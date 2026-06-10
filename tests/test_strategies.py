import numpy as np
import pandas as pd
import pytest

from bot.strategies import Signal, build_strategy
from bot.strategies.mean_reversion import MeanReversion
from bot.strategies.momentum_breakout import MomentumBreakout
from bot.strategies.trend_following import TrendFollowing


def make_ohlc(closes):
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame({
        "Open": closes,
        "High": closes * 1.001,
        "Low": closes * 0.999,
        "Close": closes,
    })


def test_build_strategy_factory():
    assert isinstance(build_strategy("mean_reversion", {}), MeanReversion)
    assert isinstance(build_strategy("momentum_breakout", {}), MomentumBreakout)
    assert isinstance(build_strategy("trend_following", {}), TrendFollowing)
    with pytest.raises(ValueError):
        build_strategy("nope", {})


class TestMeanReversion:
    def test_long_on_sharp_drop(self):
        # Flat-ish market then a hard 3-bar selloff: stretched below mean.
        closes = list(5000 + np.sin(np.linspace(0, 6, 40)) * 5)
        closes += [4985, 4965, 4940]
        df = make_ohlc(closes)
        strat = MeanReversion()
        assert strat.evaluate(df, None) == Signal.LONG

    def test_short_on_sharp_spike(self):
        closes = list(5000 + np.sin(np.linspace(0, 6, 40)) * 5)
        closes += [5015, 5035, 5060]
        df = make_ohlc(closes)
        strat = MeanReversion()
        assert strat.evaluate(df, None) == Signal.SHORT

    def test_hold_in_quiet_market(self):
        rng = np.random.default_rng(42)
        closes = 5000 + rng.normal(0, 2, 60)  # flat chop, no stretch
        df = make_ohlc(closes)
        assert MeanReversion().evaluate(df, None) == Signal.HOLD

    def test_exits_long_when_price_reverts_to_mean(self):
        # Price recovered to sit at/above the 20-bar mean -> z >= 0.
        closes = list(np.full(40, 5000.0)) + [5001.0]
        df = make_ohlc(closes)
        assert MeanReversion().evaluate(df, "long") == Signal.EXIT

    def test_holds_long_while_still_below_mean(self):
        closes = list(5000 + np.sin(np.linspace(0, 6, 40)) * 5) + [4985, 4965, 4940]
        df = make_ohlc(closes)
        assert MeanReversion().evaluate(df, "long") == Signal.HOLD


class TestMomentumBreakout:
    def test_long_on_breakout_above_channel_and_trend(self):
        closes = list(60000 + np.sin(np.linspace(0, 8, 60)) * 200)
        closes.append(61500)  # closes above 20-bar high and EMA50
        df = make_ohlc(closes)
        assert MomentumBreakout().evaluate(df, None) == Signal.LONG

    def test_short_on_breakdown(self):
        closes = list(60000 + np.sin(np.linspace(0, 8, 60)) * 200)
        closes.append(58500)
        df = make_ohlc(closes)
        assert MomentumBreakout().evaluate(df, None) == Signal.SHORT

    def test_hold_inside_channel(self):
        closes = 60000 + np.sin(np.linspace(0, 8, 60)) * 200
        df = make_ohlc(closes)
        assert MomentumBreakout().evaluate(df, None) == Signal.HOLD

    def test_exits_long_on_close_below_exit_channel(self):
        closes = list(np.linspace(60000, 65000, 60))  # steady uptrend
        closes.append(62000)  # crashes below the 10-bar low
        df = make_ohlc(closes)
        assert MomentumBreakout().evaluate(df, "long") == Signal.EXIT


class TestTrendFollowing:
    def test_long_on_fresh_golden_cross(self):
        # Long decline, then a strong recovery: fast EMA crosses above slow.
        closes = list(np.linspace(2400, 2300, 60)) + list(np.linspace(2300, 2420, 25))
        df = make_ohlc(closes)
        strat = TrendFollowing()
        signals = [strat.evaluate(df.iloc[:i], None) for i in range(strat.min_bars, len(df) + 1)]
        assert Signal.LONG in signals

    def test_short_on_fresh_death_cross(self):
        closes = list(np.linspace(2300, 2400, 60)) + list(np.linspace(2400, 2280, 25))
        df = make_ohlc(closes)
        strat = TrendFollowing()
        signals = [strat.evaluate(df.iloc[:i], None) for i in range(strat.min_bars, len(df) + 1)]
        assert Signal.SHORT in signals

    def test_no_entry_mid_trend_without_crossover(self):
        closes = np.linspace(2300, 2500, 80)  # established trend, no fresh cross
        df = make_ohlc(closes)
        assert TrendFollowing().evaluate(df, None) == Signal.HOLD

    def test_exits_long_when_trend_flips(self):
        closes = list(np.linspace(2300, 2400, 60)) + list(np.linspace(2400, 2250, 30))
        df = make_ohlc(closes)
        assert TrendFollowing().evaluate(df, "long") == Signal.EXIT
