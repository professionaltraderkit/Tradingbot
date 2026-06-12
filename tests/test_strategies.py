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


class TestLevelReversal:
    """Synthetic two-day 5m session: day 1 sets PDH=605 / PDL=595, day 2's
    premarket sets PMH=602 / PML=601, then a test bar is appended."""

    def _frame(self, last_bar):
        ts, o, h, l, c = last_bar
        rows = []
        # Day 1 regular session: flat 600 with one spike high and one spike low.
        t = pd.Timestamp("2026-06-08 09:30", tz="America/New_York")
        for i in range(78):
            o1 = h1 = l1 = c1 = 600.0
            if i == 30:
                h1 = 605.0
            if i == 40:
                l1 = 595.0
            rows.append((t + pd.Timedelta(minutes=5 * i), o1, h1, l1, c1))
        # Day 2 premarket: 601-602 range.
        t = pd.Timestamp("2026-06-09 08:00", tz="America/New_York")
        for i in range(18):
            rows.append((t + pd.Timedelta(minutes=5 * i), 601.5, 602.0, 601.0, 601.5))
        rows.append((pd.Timestamp(f"2026-06-09 {ts}", tz="America/New_York"),
                     o, h, l, c))
        idx = pd.DatetimeIndex([r[0] for r in rows])
        return pd.DataFrame(
            {"Open": [r[1] for r in rows], "High": [r[2] for r in rows],
             "Low": [r[3] for r in rows], "Close": [r[4] for r in rows]},
            index=idx)

    def test_levels_computed_from_sessions(self):
        from bot.strategies.level_reversal import LevelReversal
        df = self._frame(("12:00", 598, 598, 598, 598))
        lvls = LevelReversal().levels(df)
        assert lvls == {"PDH": 605.0, "PDL": 595.0, "PMH": 602.0, "PML": 601.0}

    def test_short_on_rejection_at_prev_day_high(self):
        from bot.strategies.level_reversal import LevelReversal
        df = self._frame(("10:30", 604.6, 605.3, 604.2, 604.0))
        assert LevelReversal().evaluate(df, None) == Signal.SHORT

    def test_long_on_rejection_at_prev_day_low(self):
        from bot.strategies.level_reversal import LevelReversal
        df = self._frame(("11:00", 595.4, 595.9, 594.8, 595.8))
        assert LevelReversal().evaluate(df, None) == Signal.LONG

    def test_hold_between_levels(self):
        from bot.strategies.level_reversal import LevelReversal
        df = self._frame(("12:00", 598, 598.2, 597.8, 598))
        assert LevelReversal().evaluate(df, None) == Signal.HOLD

    def test_no_entry_before_trade_window(self):
        from bot.strategies.level_reversal import LevelReversal
        df = self._frame(("09:35", 604.6, 605.3, 604.2, 604.0))
        assert LevelReversal().evaluate(df, None) == Signal.HOLD

    def test_long_takes_profit_at_next_level(self):
        from bot.strategies.level_reversal import LevelReversal
        # Long from support; bar crosses PML (601) from below.
        df = self._frame(("13:00", 600.8, 601.3, 600.7, 601.2))
        assert LevelReversal().evaluate(df, "long") == Signal.EXIT

    def test_short_takes_profit_at_next_level(self):
        from bot.strategies.level_reversal import LevelReversal
        df = self._frame(("13:00", 601.4, 601.5, 600.9, 601.0))
        assert LevelReversal().evaluate(df, "short") == Signal.EXIT

    def test_flat_by_end_of_session(self):
        from bot.strategies.level_reversal import LevelReversal
        df = self._frame(("15:55", 600, 600, 600, 600))
        assert LevelReversal().evaluate(df, "long") == Signal.EXIT
        assert LevelReversal().evaluate(df, "short") == Signal.EXIT

    def test_hold_position_mid_session_between_levels(self):
        from bot.strategies.level_reversal import LevelReversal
        df = self._frame(("13:00", 598, 598.2, 597.8, 598))
        assert LevelReversal().evaluate(df, "long") == Signal.HOLD

    def test_naive_index_returns_hold(self):
        from bot.strategies.level_reversal import LevelReversal
        df = self._frame(("12:00", 598, 598, 598, 598))
        df.index = df.index.tz_localize(None)
        assert LevelReversal().evaluate(df, None) == Signal.HOLD
