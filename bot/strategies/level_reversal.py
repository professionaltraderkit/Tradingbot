"""Support/resistance reversal on 5-minute candles.

Key intraday levels — the previous session's high/low (PDH/PDL) and
today's premarket high/low (PMH/PML) — act as resistance and support.
When a 5m bar pierces a level and gets rejected (closes back on the
original side with a counter-directional body), fade the move: short the
failed breakout at resistance, buy the rejection at support.

Positions take profit when price reaches the next level, and everything
is flat by the end of the session — no overnight risk.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .base import Signal, Strategy

ET = ZoneInfo("America/New_York")


def _parse_time(hhmm: str):
    return datetime.strptime(hhmm, "%H:%M").time()


class LevelReversal(Strategy):
    def __init__(self, tolerance_pct=0.1, trade_start="09:45",
                 trade_end="15:45", flatten_at="15:55"):
        self.tolerance_pct = tolerance_pct
        self.trade_start = _parse_time(trade_start)
        self.trade_end = _parse_time(trade_end)
        self.flatten_at = _parse_time(flatten_at)
        self.min_bars = 50

    # -- level computation -------------------------------------------------
    def levels(self, df: pd.DataFrame) -> dict[str, float]:
        """PDH/PDL from the previous session's regular hours, PMH/PML from
        today's 4:00–9:30 ET premarket. Requires a tz-aware index."""
        if df.index.tz is None:
            return {}
        d = df.tz_convert(ET)
        dates = sorted({ts.date() for ts in d.index})
        today = d.index[-1].date()
        out: dict[str, float] = {}

        prev_days = [x for x in dates if x < today]
        if prev_days:
            prev = d[pd.Index(d.index.date) == prev_days[-1]]
            regular = prev.between_time("09:30", "16:00")
            src = regular if not regular.empty else prev
            out["PDH"] = float(src["High"].max())
            out["PDL"] = float(src["Low"].min())

        premarket = d[pd.Index(d.index.date) == today].between_time("04:00", "09:29")
        if not premarket.empty:
            out["PMH"] = float(premarket["High"].max())
            out["PML"] = float(premarket["Low"].min())
        return out

    # -- signal logic ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, position_side: str | None) -> Signal:
        if len(df) < self.min_bars or df.index.tz is None:
            return Signal.HOLD
        lvls = self.levels(df)
        if not lvls:
            return Signal.HOLD

        last = df.iloc[-1]
        o, h, l, c = (float(last["Open"]), float(last["High"]),
                      float(last["Low"]), float(last["Close"]))
        now_et = df.index[-1].tz_convert(ET).time()

        if position_side == "long":
            if now_et >= self.flatten_at:
                return Signal.EXIT
            # Take profit when price reaches the next level from below.
            if any(o < lv <= h for lv in lvls.values()):
                return Signal.EXIT
            return Signal.HOLD
        if position_side == "short":
            if now_et >= self.flatten_at:
                return Signal.EXIT
            if any(o > lv >= l for lv in lvls.values()):
                return Signal.EXIT
            return Signal.HOLD

        if not (self.trade_start <= now_et <= self.trade_end):
            return Signal.HOLD

        tol = c * self.tolerance_pct / 100
        # Rejection at resistance: wick into the level, close back below it
        # with a bearish body. Support is the mirror image.
        short_setup = any(
            lv > c and h >= lv - tol and c < lv and c < o
            for lv in lvls.values()
        )
        long_setup = any(
            lv < c and l <= lv + tol and c > lv and c > o
            for lv in lvls.values()
        )
        if short_setup and long_setup:
            return Signal.HOLD  # conflicting rejections, stand aside
        if short_setup:
            return Signal.SHORT
        if long_setup:
            return Signal.LONG
        return Signal.HOLD

    def stance(self, df: pd.DataFrame) -> str:
        if len(df) < self.min_bars or df.index.tz is None:
            return "warming up (not enough data)"
        lvls = self.levels(df)
        if not lvls:
            return "no session levels yet"
        c = float(df["Close"].iloc[-1])
        parts = [f"{k} {v:,.2f}" for k, v in sorted(lvls.items(), key=lambda kv: -kv[1])]
        nearest = min(lvls.items(), key=lambda kv: abs(kv[1] - c))
        dist = (c - nearest[1]) / c * 100
        return (f"levels: {', '.join(parts)}; nearest {nearest[0]} "
                f"({dist:+.2f}% away)")
