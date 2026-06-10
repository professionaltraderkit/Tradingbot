"""Mean reversion for index futures on 15m candles.

Fades short-term overextensions: enter when price is stretched more than
`zscore_entry` standard deviations from its SMA with a fast RSI confirming
the extreme; exit when price reverts back to the mean.
"""

import pandas as pd

from ..indicators import rsi, sma, zscore
from .base import Signal, Strategy


class MeanReversion(Strategy):
    def __init__(self, sma_period=20, zscore_entry=2.0, rsi_period=2,
                 rsi_oversold=10, rsi_overbought=90):
        self.sma_period = sma_period
        self.zscore_entry = zscore_entry
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_bars = sma_period + 5

    def _signals(self, df: pd.DataFrame):
        close = df["Close"]
        return (
            zscore(close, self.sma_period).iloc[-1],
            rsi(close, self.rsi_period).iloc[-1],
        )

    def evaluate(self, df: pd.DataFrame, position_side: str | None) -> Signal:
        if len(df) < self.min_bars:
            return Signal.HOLD
        z, r = self._signals(df)
        if pd.isna(z) or pd.isna(r):
            return Signal.HOLD

        if position_side == "long":
            return Signal.EXIT if z >= 0 else Signal.HOLD
        if position_side == "short":
            return Signal.EXIT if z <= 0 else Signal.HOLD

        if z < -self.zscore_entry and r < self.rsi_oversold:
            return Signal.LONG
        if z > self.zscore_entry and r > self.rsi_overbought:
            return Signal.SHORT
        return Signal.HOLD

    def stance(self, df: pd.DataFrame) -> str:
        if len(df) < self.min_bars:
            return "warming up (not enough data)"
        z, r = self._signals(df)
        if pd.isna(z):
            return "warming up (not enough data)"
        if z < -self.zscore_entry:
            return f"stretched {abs(z):.1f}σ below mean — watching for long"
        if z > self.zscore_entry:
            return f"stretched {z:.1f}σ above mean — watching for short"
        return f"near its mean (z={z:+.1f}σ) — no edge"
