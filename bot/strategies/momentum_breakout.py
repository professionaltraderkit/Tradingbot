"""Momentum breakout for Bitcoin on 1h candles.

Crypto trends harder than indices, so ride the move instead of fading it:
enter on a close beyond the Donchian channel in the direction of the EMA
trend, exit when price closes back through the shorter opposite channel.
"""

import pandas as pd

from ..indicators import donchian_high, donchian_low, ema
from .base import Signal, Strategy


class MomentumBreakout(Strategy):
    def __init__(self, breakout_period=20, exit_period=10, trend_ema_period=50):
        self.breakout_period = breakout_period
        self.exit_period = exit_period
        self.trend_ema_period = trend_ema_period
        self.min_bars = max(breakout_period, trend_ema_period) + 5

    def evaluate(self, df: pd.DataFrame, position_side: str | None) -> Signal:
        if len(df) < self.min_bars:
            return Signal.HOLD
        close = df["Close"].iloc[-1]
        upper = donchian_high(df["High"], self.breakout_period).iloc[-1]
        lower = donchian_low(df["Low"], self.breakout_period).iloc[-1]
        exit_low = donchian_low(df["Low"], self.exit_period).iloc[-1]
        exit_high = donchian_high(df["High"], self.exit_period).iloc[-1]
        trend = ema(df["Close"], self.trend_ema_period).iloc[-1]
        if pd.isna(upper) or pd.isna(lower) or pd.isna(trend):
            return Signal.HOLD

        if position_side == "long":
            return Signal.EXIT if close < exit_low else Signal.HOLD
        if position_side == "short":
            return Signal.EXIT if close > exit_high else Signal.HOLD

        if close > upper and close > trend:
            return Signal.LONG
        if close < lower and close < trend:
            return Signal.SHORT
        return Signal.HOLD

    def stance(self, df: pd.DataFrame) -> str:
        if len(df) < self.min_bars:
            return "warming up (not enough data)"
        close = df["Close"].iloc[-1]
        upper = donchian_high(df["High"], self.breakout_period).iloc[-1]
        lower = donchian_low(df["Low"], self.breakout_period).iloc[-1]
        trend = ema(df["Close"], self.trend_ema_period).iloc[-1]
        if pd.isna(upper):
            return "warming up (not enough data)"
        side = "above" if close > trend else "below"
        room_up = (upper - close) / close * 100
        room_dn = (close - lower) / close * 100
        if room_up <= 0:
            channel = f"broke out {-room_up:.1f}% above the channel"
        elif room_dn <= 0:
            channel = f"broke down {-room_dn:.1f}% below the channel"
        else:
            channel = (f"{room_up:.1f}% below breakout level, "
                       f"{room_dn:.1f}% above breakdown level")
        return f"{side} trend EMA; {channel}"
