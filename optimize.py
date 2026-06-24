#!/usr/bin/env python3
"""Parameter optimizer + walk-forward Checker for the per-symbol bots.

    python optimize.py [--config config-levels.yaml] [--days N]
                       [--wf-splits 3] [--min-trades 15] [--min-oos-trades 20]
                       [--top 12] [--no-ledger] [--ledger-summary]

Two halves of the maker-checker pattern:

  * Maker  - sweep a grid of strategy parameters over the full history and rank
    them (the exploratory table).
  * Checker - hand the winner to validation.walk_forward: an expanding-window
    walk-forward that re-picks params out-of-sample, penalizes the grid search
    for multiple testing (a deflated-Sharpe-style bar), and measures in->out
    decay. Only combos that survive earn a PASS.

Every ruling - PASS / WEAK / VETO with its numbers - is written to the experiment
ledger (bot/ledger.py) so the search compounds across runs instead of starting
cold each month. Costs are charged exactly as in backtest.py, so results are net.

    python optimize.py --ledger-summary     # what's been tried + verdicts so far
"""

import argparse
import itertools
import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

import validation
from backtest import DEFAULT_COST_BPS, build_risk
from bot.data import MarketData
from bot.ledger import ExperimentLedger, current_git_sha

log = logging.getLogger(__name__)

# Grid swept per strategy. Keys must be that strategy's constructor params.
# Keep each modest - every combo is a full backtest on the whole history AND on
# every walk-forward training window.
GRIDS = {
    "vwap_pullback": {
        "adx_min": [18, 25, 32],
        "stop_atr": [0.5, 1.0],
        "target_r": [1.5, 2.5, 0.0],   # 0.0 = no fixed target (trail only)
        "trail_atr": [0.0, 2.5],
        "use_breakeven": [True, False],
        "trade_end": ["11:30", "15:30"],  # morning-only vs full session
    },
    "opening_range_breakout": {
        "or_minutes": [15, 30, 60],
        "stop_mode": ["range", "atr"],
        "target_r": [1.0, 2.0, 0.0],   # 0.0 = no fixed target (trail only)
        "trail_atr": [0.0, 2.0],
        "use_breakeven": [True, False],
        "entry_end": ["12:00", "15:00"],
    },
}


def combos(grid: dict):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        # An exit must exist: skip "no target AND no trail" (stop/EOD only).
        if combo.get("target_r", 1) == 0.0 and combo.get("trail_atr", 1) == 0.0:
            continue
        yield combo


def _print_table(rows: list[dict], top: int) -> None:
    """The Maker's exploratory ranking - full-history fit, no overfit guard yet."""
    print(f"\n  {'Ret%':>7} {'PF':>5} {'Tr':>5} {'Exp$':>9} | params")
    print("  " + "-" * 72)
    for r in rows[:top]:
        s = r["stats"]
        pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
        params = ", ".join(f"{k}={v}" for k, v in r["combo"].items())
        print(f"  {s['return_pct']:>7.2f} {pf:>5} {s['trades']:>5} "
              f"{s['expectancy']:>9.2f} | {params}")


def _print_verdict(wf: validation.WalkForwardResult) -> None:
    v = wf.verdict
    print(f"\n  Walk-forward verdict: {v.verdict}   "
          f"({wf.n_folds} OOS folds, N={wf.n_trials} combos searched)")
    print(f"    IS return {wf.is_return:+.2f}%  ->  OOS return {wf.oos_return:+.2f}% "
          f"on {wf.oos_trades} trades")
    print(f"    OOS t-stat {wf.oos_tstat:.2f} vs snooping bar {v.threshold_t:.2f}  |  "
          f"folds +ve {v.consistency * 100:.0f}%  |  in->out decay {v.gap * 100:.0f}%")
    if wf.best_combo:                 # always show the params that were tested
        label = "deploy" if v.passed else "candidate (tested, not cleared)"
        print(f"    {label}: " +
              ", ".join(f"{k}={val}" for k, val in wf.best_combo.items()))
    if v.passed:
        print("    -> survived out-of-sample. A hypothesis worth paper-forward-testing.")
    else:
        for reason in v.reasons:
            print(f"      - {reason}")
        print("    -> not validated; don't deploy this on hope.")


def optimize_instrument(inst: dict, df: pd.DataFrame, ctx: validation.BacktestContext,
                        min_trades: int, top: int, wf_splits: int,
                        min_oos_trades: int):
    grid = GRIDS.get(inst["strategy"])
    if grid is None:
        print(f"\n=== {inst['name']} ({inst['ticker']}) - no optimizer grid for "
              f"strategy '{inst['strategy']}', skipping ===")
        return None
    all_combos = list(combos(grid))
    print(f"\n=== {inst['name']} ({inst['ticker']}) - {len(df)} bars, "
          f"{len(all_combos)} combos x {wf_splits + 1} windows ===")

    # Maker: sweep the full history -> ranked table, deploy pick, snooping spread.
    best_combo, best_stats, rows, trial_tstats = validation.select(
        inst, df, all_combos, ctx, min_trades)
    if not rows:
        print("  No combo produced any trades here. Try --days higher.")
        return None
    _print_table(rows, top)
    if best_combo is None:
        print(f"\n  No combo cleared {min_trades} trades in-sample; nothing to validate.")
        return None

    # Checker: walk-forward the selection procedure out-of-sample.
    wf = validation.walk_forward(
        inst, df, all_combos, ctx,
        preselected=(best_combo, trial_tstats, best_stats),
        n_splits=wf_splits, min_train_trades=min_trades,
        min_oos_trades=min_oos_trades)
    _print_verdict(wf)
    return wf


def _print_ledger_summary(db_path: str) -> None:
    ledger = ExperimentLedger(db_path)
    counts = ledger.summary()
    rows = ledger.recent(limit=15)
    ledger.close()
    if not rows:
        print("Experiment ledger is empty. Run the optimizer to populate it.")
        return
    print("Experiment ledger -> " +
          ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"\n  {'id':>4} {'when':<16} {'strategy':<22} {'tkr':<6} "
          f"{'verdict':<7} {'OOS%':>7} {'t':>5} {'bar':>5}")
    print("  " + "-" * 82)
    for r in rows:
        when = (r["created_at"] or "")[:16].replace("T", " ")
        print(f"  {r['id']:>4} {when:<16} {r['strategy']:<22} {r['ticker']:<6} "
              f"{r['verdict']:<7} {r['oos_return'] or 0:>7.2f} "
              f"{r['oos_tstat'] or 0:>5.2f} {r['threshold_t'] or 0:>5.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize + walk-forward-validate per-symbol strategy params")
    parser.add_argument("--config", default="config-levels.yaml")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--wf-splits", type=int, default=3,
                        help="walk-forward folds (more = stricter but needs more data)")
    parser.add_argument("--min-trades", type=int, default=15,
                        help="minimum in-sample trades for a combo to be eligible")
    parser.add_argument("--min-oos-trades", type=int, default=20,
                        help="minimum pooled out-of-sample trades to trust a verdict")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--ledger-db", default="experiments.db")
    parser.add_argument("--no-ledger", action="store_true",
                        help="don't write results to the experiment ledger")
    parser.add_argument("--ledger-summary", action="store_true",
                        help="print recent ledger entries and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    load_dotenv(Path(__file__).resolve().parent / ".env")

    if args.ledger_summary:
        _print_ledger_summary(args.ledger_db)
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)

    risk = build_risk(config)
    risk_cfg = config["risk"]
    equity = config["account"]["starting_equity"]
    max_notional_pct = risk_cfg.get("max_position_notional_pct", 100)
    atr_period = risk_cfg["atr_period"]
    cost_bps = (args.cost_bps if args.cost_bps is not None
                else config.get("costs", {}).get("per_side_bps", DEFAULT_COST_BPS))
    ctx = validation.BacktestContext(risk, atr_period, equity, max_notional_pct, cost_bps)

    ledger = None if args.no_ledger else ExperimentLedger(args.ledger_db)
    git_sha = current_git_sha()

    feed = MarketData()
    recommended = {}
    for inst in config["instruments"]:
        df = feed.fetch_candles(inst["ticker"], inst["timeframe"])
        if df.empty:
            print(f"!! no data for {inst['ticker']}, skipping")
            continue
        if args.days:
            cutoff = df.index[-1] - pd.Timedelta(days=args.days)
            df = df[df.index >= cutoff]
        wf = optimize_instrument(inst, df, ctx, args.min_trades, args.top,
                                 args.wf_splits, args.min_oos_trades)
        if wf is None:
            continue
        if ledger is not None and wf.best_combo is not None:
            row = validation.ledger_row(
                inst, wf, cost_bps=cost_bps, config_file=args.config, git_sha=git_sha,
                data_start=str(df.index[0]), data_end=str(df.index[-1]),
                bars=len(df), timeframe=inst.get("timeframe"))
            rid = ledger.record(row)
            print(f"    recorded to ledger (id={rid})")
        if wf.verdict.passed:
            recommended[inst["name"]] = {**inst.get("params", {}), **wf.best_combo}

    if ledger is not None:
        ledger.close()

    print("\n" + "=" * 64)
    if recommended:
        print("VALIDATED params (PASS only) - paste into your config, then")
        print("re-run `python backtest.py --config <file>`:\n")
        print(yaml.safe_dump(recommended, sort_keys=False, default_flow_style=False))
        print("Even these are hypotheses: the only honest test is forward paper")
        print("trading on bars the optimizer never saw.")
    else:
        print("No instrument produced a PASS. There is no validated edge to deploy")
        print("here - a real and common result, not a failure of the search.")


if __name__ == "__main__":
    main()
