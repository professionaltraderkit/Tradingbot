"""Slow trend following for gold and oil on 4h candles.

Commodities move in cleaner waves, so hold the direction of the EMA
crossover and ignore intraday whipsaws: long while the fast EMA is above
the slow EMA, exit (and allow the opposite side) when it crosses back.
"""

import pandas as pd

from ..indicators import ema
from .base import Signal, Strategy


class TrendFollowing(Strategy):
    def __init__(self, fast_ema_period=20, slow_ema_period=50):
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.min_bars = slow_ema_period + 5

    def _emas(self, df: pd.DataFrame):
        close = df["Close"]
        return (
            ema(close, self.fast_ema_period),
            ema(close, self.slow_ema_period),
        )

    def evaluate(self, df: pd.DataFrame, position_side: str | None) -> Signal:
        if len(df) < self.min_bars:
            return Signal.HOLD
        fast, slow = self._emas(df)
        f_now, s_now = fast.iloc[-1], slow.iloc[-1]
        f_prev, s_prev = fast.iloc[-2], slow.iloc[-2]
        if pd.isna(f_now) or pd.isna(s_now):
            return Signal.HOLD

        if position_side == "long":
            return Signal.EXIT if f_now < s_now else Signal.HOLD
        if position_side == "short":
            return Signal.EXIT if f_now > s_now else Signal.HOLD

        # Enter only on a fresh crossover, not just because one EMA is above
        # the other (avoids buying late into an aging trend after a stop-out).
        if f_now > s_now and f_prev <= s_prev:
            return Signal.LONG
        if f_now < s_now and f_prev >= s_prev:
            return Signal.SHORT
        return Signal.HOLD

    def stance(self, df: pd.DataFrame) -> str:
        if len(df) < self.min_bars:
            return "warming up (not enough data)"
        fast, slow = self._emas(df)
        f_now, s_now = fast.iloc[-1], slow.iloc[-1]
        if pd.isna(f_now) or pd.isna(s_now):
            return "warming up (not enough data)"
        spread = (f_now - s_now) / s_now * 100
        direction = "uptrend" if f_now > s_now else "downtrend"
        return f"{direction}, EMA spread {spread:+.2f}%"
