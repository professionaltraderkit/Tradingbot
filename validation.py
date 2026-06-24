#!/usr/bin/env python3
"""The Checker: standardized walk-forward validation with an overfit-aware verdict.

The optimizer (the "Maker") will always find a parameter combo that looks good
on the data it was tuned on. This module is the independent "Checker" from the
maker-checker pattern: it re-runs the parameter search *out-of-sample* and then
tries to KILL the winner with three guards a single backtest can't apply:

  1. Out-of-sample survival - walk-forward (expanding window). Each fold re-picks
     its own best params on the bars seen so far and is scored on the next,
     unseen chunk. The pooled out-of-sample trades are the honest result.
  2. Multiple-testing penalty - sweeping a grid of N combos and keeping the best
     is data snooping; the luckiest of N noise strategies looks great by chance.
     We take the t-stat of the average trade and require the chosen combo to beat
     the *expected best t-stat of N noise strategies* - a deflated-Sharpe-style
     benchmark (Bailey & Lopez de Prado, 2014), built from stdlib math only.
  3. In-sample -> out-of-sample decay - if the per-trade edge mostly evaporates
     out of sample, it was a fit, not a signal.

A combo earns PASS only if it clears all three (and trades enough to mean
anything). The scoring helpers (`trade_tstat`, `expected_max_factor`, `judge`)
are metric-level and strategy-agnostic, so the portfolio strategies can reuse
them; only `select`/`walk_forward` are specific to the per-symbol candle
backtest, which they drive through backtest.run_simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import _trade_cost, compute_stats, run_simulation
# The overfit guards live in bot.validation_core so the swing strategies can
# reuse the exact same verdict logic; re-exported here for the per-symbol driver.
from bot.validation_core import (Verdict, deflated_threshold,  # noqa: F401
                                  expected_max_factor, judge, trade_tstat)


# -- per-symbol walk-forward driver -----------------------------------------
@dataclass
class BacktestContext:
    """The fixed cost/risk/sizing knobs every backtest in a sweep shares."""
    risk: object
    atr_period: int
    equity: float
    max_notional_pct: float
    cost_bps: float


@dataclass
class WalkForwardResult:
    verdict: Verdict
    chosen: list[dict]                 # the combo each fold picked (stability)
    is_return: float
    oos_return: float
    oos_trades: int
    oos_tstat: float
    fold_returns: list[float]
    n_trials: int
    n_folds: int
    threshold_t: float
    best_combo: dict | None            # combo selected on the full history (deploy)
    insufficient: bool = False


def _run(inst: dict, df: pd.DataFrame, ctx: BacktestContext):
    """One backtest -> (stats dict, list of net per-trade P&L). Isolated as a
    module function so tests can monkeypatch it with a synthetic runner."""
    trades, curve = run_simulation(inst, df, ctx.risk, ctx.atr_period, ctx.equity,
                                   ctx.max_notional_pct, ctx.cost_bps)
    net = [t["pnl"] - _trade_cost(t, ctx.cost_bps) for t in trades]
    stats = compute_stats(inst["name"], len(df), trades, curve, ctx.equity,
                          ctx.cost_bps)
    return stats, net


def select(inst: dict, df: pd.DataFrame, combos, ctx: BacktestContext,
           min_trades: int = 15):
    """Sweep the grid in-sample on `df`.

    Returns (best_combo, best_stats, rows, trial_tstats): the highest-return combo
    that cleared `min_trades` and its stats, the per-combo rows sorted for display,
    and every evaluated combo's trade t-stat (the dispersion that sets the snooping
    bar). best_combo/best_stats are None if nothing cleared the trades floor."""
    rows, tstats = [], []
    for combo in combos:
        variant = {**inst, "params": {**inst.get("params", {}), **combo}}
        try:
            stats, net = _run(variant, df, ctx)
        except Exception:                 # invalid combo for this strategy
            continue
        tstats.append(trade_tstat(net))
        rows.append({"combo": combo, "stats": stats})
    rows.sort(key=lambda r: r["stats"]["return_pct"], reverse=True)
    eligible = [r for r in rows if r["stats"]["trades"] >= min_trades]
    if eligible:                          # rows are return-sorted, so [0] is best
        return eligible[0]["combo"], eligible[0]["stats"], rows, tstats
    return None, None, rows, tstats


def _chunks(df: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """Split a frame into n contiguous, roughly equal time slices."""
    bounds = np.linspace(0, len(df), n + 1, dtype=int)
    return [df.iloc[bounds[i]:bounds[i + 1]] for i in range(n)]


def walk_forward(inst: dict, df: pd.DataFrame, combos, ctx: BacktestContext, *,
                 preselected: tuple | None = None, n_splits: int = 3,
                 min_train_trades: int = 15, min_test_trades: int = 5,
                 min_oos_trades: int = 20, gap_tol: float = 0.6) -> WalkForwardResult:
    """Expanding-window walk-forward for a per-symbol candle strategy.

    The timeline is cut into n_splits+1 contiguous chunks. Fold k trains on
    chunks[0..k] (everything seen so far), re-picks its own best combo there, and
    is scored on chunk[k+1] (the next, unseen slice). The pooled out-of-sample
    trades are the honest estimate of how the *selection procedure* generalizes.

    `best_combo` (the thing you'd actually deploy) is chosen separately on the
    full history; the IS<->OOS gap compares that combo's full-sample per-trade
    expectancy against the pooled walk-forward expectancy. Pass `preselected`
    (best_combo, trial_tstats, is_stats) to skip recomputing the full sweep when
    the caller already did it for its own ranking table."""
    combos = list(combos)
    if preselected is None:
        best_combo, is_stats, _, trial_tstats = select(inst, df, combos, ctx,
                                                       min_train_trades)
    else:
        best_combo, trial_tstats, is_stats = preselected

    n_trials = len([t for t in trial_tstats if np.isfinite(t)])
    threshold_t = deflated_threshold(trial_tstats, n_trials)

    if best_combo is None:
        v = Verdict("VETO", ["no combo met the in-sample minimum-trades floor"],
                    threshold_t=threshold_t, n_trials=n_trials)
        return WalkForwardResult(v, [], 0.0, 0.0, 0, 0.0, [], n_trials, 0,
                                 threshold_t, None, insufficient=True)

    chunks = _chunks(df, n_splits + 1)
    min_test_bars = max(30, ctx.atr_period * 3)
    pooled_net: list[float] = []
    fold_returns: list[float] = []
    chosen: list[dict] = []
    for k in range(n_splits):
        train = pd.concat(chunks[:k + 1])
        test = chunks[k + 1]
        if len(test) < min_test_bars:
            continue
        combo, _, _, _ = select(inst, train, combos, ctx, min_train_trades)
        if combo is None:
            continue
        variant = {**inst, "params": {**inst.get("params", {}), **combo}}
        try:
            stats, net = _run(variant, test, ctx)
        except Exception:
            continue
        if stats["trades"] < min_test_trades:
            continue
        chosen.append(combo)
        pooled_net.extend(net)
        fold_returns.append(sum(net) / ctx.equity * 100.0)

    oos_trades = len(pooled_net)
    oos_return = sum(pooled_net) / ctx.equity * 100.0 if pooled_net else 0.0
    oos_exp = float(np.mean(pooled_net)) if pooled_net else 0.0
    oos_tstat = trade_tstat(pooled_net)
    is_exp = is_stats["expectancy"]
    gap = ((is_exp - oos_exp) / abs(is_exp)) if is_exp else 1.0

    v = judge(is_return=is_stats["return_pct"], oos_return=oos_return,
              oos_trades=oos_trades, oos_tstat=oos_tstat, threshold_t=threshold_t,
              fold_returns=fold_returns, gap=gap, n_trials=n_trials,
              min_oos_trades=min_oos_trades, gap_tol=gap_tol)

    insufficient = len(fold_returns) < 2
    if insufficient and v.verdict == "PASS":     # never bless on <2 OOS folds
        v.verdict = "WEAK"
        v.reasons.append(f"thin walk-forward: only {len(fold_returns)} OOS fold(s)")

    return WalkForwardResult(v, chosen, is_stats["return_pct"], oos_return,
                             oos_trades, oos_tstat, fold_returns, n_trials,
                             len(fold_returns), threshold_t, best_combo,
                             insufficient=insufficient)


def ledger_row(inst: dict, wf: WalkForwardResult, *, cost_bps: float,
               config_file: str, git_sha: str | None, data_start: str,
               data_end: str, bars: int, timeframe: str | None) -> dict:
    """Flatten a walk-forward result into the dict ExperimentLedger.record wants."""
    return {
        "strategy": inst["strategy"],
        "ticker": inst["ticker"],
        "name": inst.get("name"),
        "timeframe": timeframe,
        "config_file": config_file,
        "data_start": data_start,
        "data_end": data_end,
        "bars": bars,
        "cost_bps": cost_bps,
        "n_trials": wf.n_trials,
        "n_folds": wf.n_folds,
        "params": wf.best_combo or {},
        "is_return": wf.is_return,
        "oos_return": wf.oos_return,
        "oos_trades": wf.oos_trades,
        "oos_tstat": wf.oos_tstat,
        "threshold_t": wf.threshold_t,
        "gap": wf.verdict.gap,
        "consistency": wf.verdict.consistency,
        "verdict": wf.verdict.verdict,
        "reasons": wf.verdict.reasons,
        "git_sha": git_sha,
    }
