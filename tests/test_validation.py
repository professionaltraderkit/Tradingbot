"""The Checker: scoring primitives, the overfit-aware verdict, and the
walk-forward plumbing (driven by a synthetic backtest, so no network/data)."""

import numpy as np
import pandas as pd
import pytest

import validation
from validation import (BacktestContext, deflated_threshold,
                        expected_max_factor, judge, trade_tstat, walk_forward)


# -- scoring primitives ------------------------------------------------------
def test_trade_tstat_matches_formula():
    net = [2.0, -1.0, 3.0, 0.0, 1.0]              # mean 1.0
    expected = 1.0 / np.std(net, ddof=1) * np.sqrt(5)
    assert trade_tstat(net) == pytest.approx(expected)


def test_trade_tstat_degenerate_cases():
    assert trade_tstat([1.0]) == 0.0              # fewer than two trades
    assert trade_tstat([3.0, 3.0, 3.0]) == 0.0    # no dispersion


def test_expected_max_factor_grows_with_trials():
    assert expected_max_factor(1) == 0.0
    assert expected_max_factor(2) < expected_max_factor(20) < expected_max_factor(200)


def test_deflated_threshold_zero_without_dispersion():
    assert deflated_threshold([1.0, 1.0, 1.0]) == 0.0   # identical trials
    assert deflated_threshold([0.5]) == 0.0             # too few trials


def test_deflated_threshold_positive_with_spread():
    assert deflated_threshold([-1.0, 0.0, 1.0, 2.0, 3.0]) > 0


# -- the verdict -------------------------------------------------------------
def test_judge_pass_when_all_guards_clear():
    v = judge(is_return=8.0, oos_return=5.0, oos_trades=80, oos_tstat=3.0,
              threshold_t=1.5, fold_returns=[1.0, 2.0, 1.5], gap=0.3, n_trials=30)
    assert v.passed and v.verdict == "PASS" and v.reasons == []


def test_judge_veto_on_negative_oos():
    v = judge(is_return=8.0, oos_return=-1.2, oos_trades=80, oos_tstat=3.0,
              threshold_t=1.5, fold_returns=[-1.0, 0.5, -2.0], gap=0.3, n_trials=30)
    assert v.verdict == "VETO"


def test_judge_weak_when_snooping_bar_not_cleared():
    # Positive OOS, but the t-stat doesn't beat the best-of-N bar -> WEAK, flagged.
    v = judge(is_return=8.0, oos_return=1.0, oos_trades=80, oos_tstat=0.7,
              threshold_t=2.5, fold_returns=[0.4, 0.3, 0.3], gap=0.3, n_trials=200)
    assert v.verdict == "WEAK"
    assert any("multiple-testing" in r for r in v.reasons)


def test_judge_weak_on_decay():
    v = judge(is_return=10.0, oos_return=0.5, oos_trades=60, oos_tstat=3.0,
              threshold_t=1.0, fold_returns=[0.2, 0.2, 0.1], gap=0.9, n_trials=20)
    assert v.verdict == "WEAK"
    assert any("decayed" in r for r in v.reasons)


def test_judge_weak_on_thin_sample():
    v = judge(is_return=5.0, oos_return=2.0, oos_trades=8, oos_tstat=3.0,
              threshold_t=1.0, fold_returns=[1.0, 1.0], gap=0.2, n_trials=20)
    assert v.verdict == "WEAK"
    assert any("thin sample" in r for r in v.reasons)


# -- walk-forward plumbing (synthetic runner) --------------------------------
def _df(n: int = 480) -> pd.DataFrame:
    idx = pd.date_range("2026-01-02 09:30", periods=n, freq="5min", tz="UTC")
    close = np.linspace(100, 110, n)
    return pd.DataFrame({"Open": close, "High": close + 0.5, "Low": close - 0.5,
                         "Close": close, "Volume": np.full(n, 1000.0)}, index=idx)


def _stats(net, equity, name):
    return {"name": name, "trades": len(net), "pnl": sum(net),
            "return_pct": sum(net) / equity * 100.0,
            "expectancy": float(np.mean(net)), "profit_factor": float("inf"),
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "max_drawdown_pct": 0.0}


def _winner_run(inst, df, ctx):
    """combo a=2 is a real edge (mean +5/trade), a=1 loses. Alternating values
    give a non-zero dispersion so the t-stat is finite."""
    a = inst["params"]["a"]
    n = max(2, len(df) // 5)         # dense enough that every WF train window qualifies
    if a == 2:
        net = [6.0 if i % 2 == 0 else 4.0 for i in range(n)]      # mean +5
    else:
        net = [-4.0 if i % 2 == 0 else -2.0 for i in range(n)]    # mean -3
    return _stats(net, ctx.equity, inst["name"]), net


def test_walk_forward_passes_a_real_edge(monkeypatch):
    ctx = BacktestContext(None, 14, 100000.0, 100, 2.0)
    monkeypatch.setattr(validation, "_run", _winner_run)
    inst = {"name": "T", "ticker": "TST", "strategy": "x", "params": {}}
    wf = walk_forward(inst, _df(), [{"a": 1}, {"a": 2}], ctx, n_splits=3)
    assert wf.best_combo == {"a": 2}        # picked the edge over the loser
    assert wf.n_folds == 3
    assert wf.oos_trades >= 20
    assert wf.verdict.passed


def test_walk_forward_vetoes_a_loser(monkeypatch):
    ctx = BacktestContext(None, 14, 100000.0, 100, 2.0)

    def all_losers(inst, df, ctx):
        n = max(2, len(df) // 5)
        net = [-3.0 if i % 2 == 0 else -1.0 for i in range(n)]
        return _stats(net, ctx.equity, inst["name"]), net

    monkeypatch.setattr(validation, "_run", all_losers)
    inst = {"name": "T", "ticker": "TST", "strategy": "x", "params": {}}
    wf = walk_forward(inst, _df(), [{"a": 1}, {"a": 2}], ctx, n_splits=3)
    assert wf.verdict.verdict == "VETO"
    assert not wf.verdict.passed


def test_walk_forward_vetoes_when_no_combo_clears_min_trades(monkeypatch):
    ctx = BacktestContext(None, 14, 100000.0, 100, 2.0)
    # Every combo trades only twice -> below min_train_trades, nothing eligible.
    monkeypatch.setattr(validation, "_run",
                        lambda inst, df, ctx: (_stats([1.0, -1.0], ctx.equity,
                                                      inst["name"]), [1.0, -1.0]))
    inst = {"name": "T", "ticker": "TST", "strategy": "x", "params": {}}
    wf = walk_forward(inst, _df(), [{"a": 1}, {"a": 2}], ctx, n_splits=3,
                      min_train_trades=15)
    assert wf.best_combo is None
    assert wf.verdict.verdict == "VETO" and wf.insufficient
