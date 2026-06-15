"""Backtester cost accounting, stats, and the optimizer's combo grid.
No network: a synthetic mean-reversion round trip drives the pipeline."""

import numpy as np
import pandas as pd
import pytest

from backtest import (DEFAULT_COST_BPS, _trade_cost, backtest_instrument,
                      compute_stats)
from bot.risk import RiskManager


def _risk():
    return RiskManager(risk_per_trade_pct=1.0, atr_stop_multiple=2.0,
                       correlation_group=[], max_same_direction=2)


def test_trade_cost_is_round_trip_bps():
    trade = {"entry_price": 100.0, "exit_price": 110.0, "quantity": 10.0}
    # (100 + 110) * 10 * 2bp/10000 = 2100 * 0.0002 = 0.42
    assert _trade_cost(trade, 2.0) == pytest.approx(0.42)
    assert _trade_cost(trade, 0.0) == 0.0


def test_compute_stats_math():
    trades = [
        {"entry_price": 100, "exit_price": 110, "quantity": 1, "pnl": 10.0},
        {"entry_price": 100, "exit_price": 95, "quantity": 1, "pnl": -5.0},
        {"entry_price": 100, "exit_price": 104, "quantity": 1, "pnl": 4.0},
    ]
    curve = [100000, 100010, 100005, 100009]
    stats = compute_stats("X", 100, trades, curve, 100000.0, cost_bps=0.0)
    assert stats["trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3 * 100)
    assert stats["pnl"] == pytest.approx(9.0)
    assert stats["profit_factor"] == pytest.approx(14.0 / 5.0)
    assert stats["expectancy"] == pytest.approx(3.0)


def test_costs_reduce_pnl():
    trades = [{"entry_price": 100, "exit_price": 110, "quantity": 100, "pnl": 1000.0}]
    free = compute_stats("X", 10, trades, [100000, 101000], 100000.0, 0.0)
    costed = compute_stats("X", 10, trades, [100000, 101000], 100000.0, 10.0)
    assert costed["pnl"] < free["pnl"]


def _mean_reversion_frame():
    """Flat, then a sharp 3-bar drop (mean-reversion LONG), then recovery to
    the mean (EXIT): exactly one deterministic round trip."""
    rng = np.random.default_rng(0)
    closes = list(5000 + rng.normal(0, 1, 60))
    closes += [4980, 4960, 4935]          # stretched > 2 sigma below -> LONG
    closes += list(np.linspace(4955, 5005, 12))  # revert to mean -> EXIT
    c = np.array(closes, dtype=float)
    idx = pd.date_range("2026-06-01 09:30", periods=len(c), freq="15min", tz="UTC")
    return pd.DataFrame({"Open": c, "High": c + 3, "Low": c - 3,
                         "Close": c, "Volume": np.full(len(c), 1000.0)}, index=idx)


def test_backtest_pipeline_runs_and_costs_bite():
    inst = {"name": "S&P 500", "ticker": "ES", "strategy": "mean_reversion",
            "params": {}}
    df = _mean_reversion_frame()
    free = backtest_instrument(inst, df, _risk(), 14, 100000.0, cost_bps=0.0)
    costed = backtest_instrument(inst, df, _risk(), 14, 100000.0, cost_bps=50.0)
    assert free["trades"] >= 1
    assert set(free) >= {"win_rate", "profit_factor", "expectancy", "pnl",
                         "max_drawdown_pct"}
    assert costed["pnl"] < free["pnl"]  # costs always reduce realized P&L


def test_optimizer_grid_excludes_no_exit_combos():
    from optimize import combos
    grid = {"target_r": [0.0, 1.5], "trail_atr": [0.0, 2.5],
            "adx_min": [20], "stop_atr": [1.0], "use_breakeven": [True],
            "trade_end": ["15:30"]}
    out = list(combos(grid))
    # The (target_r=0, trail_atr=0) combo has no exit and must be dropped.
    assert all(not (c["target_r"] == 0.0 and c["trail_atr"] == 0.0) for c in out)
    assert len(out) == 3
