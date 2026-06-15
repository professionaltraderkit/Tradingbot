"""Opening-Range Breakout (ORB) on 5-minute candles.

Define the opening range (OR) as the high/low of the first `or_minutes`
after the cash open. Once the range is set, take the FIRST breakout of the
day in either direction: go long when a bar closes above the OR high, short
when it closes below the OR low. One long attempt and one short attempt per
day at most (only the first close beyond each side counts), no new entries
after `entry_end`, always flat by the close.

Risk: stop at the opposite end of the opening range (`stop_mode: range`) or
an ATR distance (`stop_mode: atr`); position sized so that stop costs 1% of
equity. Exits: fixed R-multiple target, optional breakeven ratchet and ATR
trailing stop — same management framework as the other strategies.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ..indicators import atr
from .base import Signal, Strategy

ET = ZoneInfo("America/New_York")


def _parse_time(hhmm: str):
    return datetime.strptime(hhmm, "%H:%M").time()


class OpeningRangeBreakout(Strategy):
    def __init__(self, or_minutes=30, bar_minutes=5, buffer_pct=0.02,
                 stop_mode="range", stop_atr=1.0, target_r=1.0, trail_atr=0.0,
                 use_breakeven=True, breakeven_r=1.0, atr_len=14,
                 use_volume_filter=False, vol_len=20,
                 session_open="09:30", entry_end="12:00", flatten_at="15:50"):
        self.or_bars = max(1, or_minutes // bar_minutes)
        self.buffer_pct = buffer_pct
        if stop_mode not in ("range", "atr"):
            raise ValueError("stop_mode must be 'range' or 'atr'")
        self.stop_mode = stop_mode
        self.stop_atr = stop_atr
        self.target_r = target_r
        self.trail_atr = trail_atr
        self.use_breakeven = use_breakeven
        self.breakeven_r = breakeven_r
        self.atr_len = atr_len
        self.use_volume_filter = use_volume_filter
        self.vol_len = vol_len
        self.session_open = _parse_time(session_open)
        self.entry_end = _parse_time(entry_end)
        self.flatten_at = _parse_time(flatten_at)
        self.min_bars = atr_len + self.or_bars + 5

    # -- opening range -----------------------------------------------------
    def _today(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.tz_convert(ET)
        today = d.index[-1].date()
        return d[pd.Index(d.index.date) == today]

    def _opening_range(self, day: pd.DataFrame):
        """(or_high, or_low, last_OR_timestamp) or None if not yet formed."""
        opening = day[[t >= self.session_open for t in day.index.time]]
        opening = opening.iloc[: self.or_bars]
        if len(opening) < self.or_bars:
            return None
        return float(opening["High"].max()), float(opening["Low"].min()), opening.index[-1]

    def _setup(self, df: pd.DataFrame):
        """Return (signal, stop_price) for a flat book, else (HOLD, None)."""
        day = self._today(df)
        rng = self._opening_range(day)
        if rng is None:
            return Signal.HOLD, None
        or_high, or_low, or_end = rng

        now_et = df.index[-1].tz_convert(ET).time()
        if not (or_end.time() < now_et <= self.entry_end):
            return Signal.HOLD, None

        after = day[day.index > or_end]
        if after.empty:
            return Signal.HOLD, None
        cur = after.iloc[-1]
        prior = after.iloc[:-1]
        c = float(cur["Close"])
        up_level = or_high * (1 + self.buffer_pct / 100)
        dn_level = or_low * (1 - self.buffer_pct / 100)

        if self.use_volume_filter and "Volume" in df.columns:
            vol_sma = float(df["Volume"].rolling(self.vol_len).mean().iloc[-1])
            if pd.isna(vol_sma) or float(cur["Volume"]) <= vol_sma:
                return Signal.HOLD, None

        first_long = c > up_level and not (prior["Close"] > up_level).any()
        first_short = c < dn_level and not (prior["Close"] < dn_level).any()
        if first_long == first_short:  # neither, or both on the same bar
            return Signal.HOLD, None

        side = "long" if first_long else "short"
        stop = self._stop_for(side, c, or_high, or_low, df)
        if stop is None:
            return Signal.HOLD, None
        return (Signal.LONG if first_long else Signal.SHORT), stop

    def _stop_for(self, side, price, or_high, or_low, df):
        if self.stop_mode == "range":
            stop = or_low if side == "long" else or_high
        else:  # atr
            atr_now = float(atr(df, self.atr_len).iloc[-1])
            if pd.isna(atr_now) or atr_now <= 0:
                return None
            stop = price - self.stop_atr * atr_now if side == "long" \
                else price + self.stop_atr * atr_now
        # Stop must be on the correct side of the entry.
        if (side == "long" and stop >= price) or (side == "short" and stop <= price):
            return None
        return stop

    # -- Strategy interface ------------------------------------------------
    def evaluate(self, df: pd.DataFrame, position_side: str | None) -> Signal:
        if position_side:
            return Signal.HOLD  # exits handled by manage()
        if len(df) < self.min_bars or df.index.tz is None:
            return Signal.HOLD
        return self._setup(df)[0]

    def initial_stop(self, df: pd.DataFrame, side: str) -> float | None:
        return self._setup(df)[1]

    def manage(self, df: pd.DataFrame, position: dict) -> dict | None:
        if df.empty or df.index.tz is None:
            return None
        last = df.iloc[-1]
        if df.index[-1].tz_convert(ET).time() >= self.flatten_at:
            return {"exit": "EOD flatten"}

        entry = position["entry_price"]
        risk_per_unit = position["risk_amount"] / position["quantity"]
        if risk_per_unit <= 0:
            return None
        atr_now = float(atr(df, self.atr_len).iloc[-1]) if self.trail_atr else 0.0

        if position["side"] == "long":
            if self.target_r > 0 and float(last["High"]) >= entry + self.target_r * risk_per_unit:
                return {"exit": "target", "price": entry + self.target_r * risk_per_unit}
            new_stop = position["stop_price"]
            if self.use_breakeven and float(last["High"]) >= entry + self.breakeven_r * risk_per_unit:
                new_stop = max(new_stop, entry)
            if self.trail_atr and atr_now > 0:
                new_stop = max(new_stop, float(df["High"].tail(self.atr_len).max()) - self.trail_atr * atr_now)
            if new_stop > position["stop_price"]:
                return {"stop": new_stop}
        else:
            if self.target_r > 0 and float(last["Low"]) <= entry - self.target_r * risk_per_unit:
                return {"exit": "target", "price": entry - self.target_r * risk_per_unit}
            new_stop = position["stop_price"]
            if self.use_breakeven and float(last["Low"]) <= entry - self.breakeven_r * risk_per_unit:
                new_stop = min(new_stop, entry)
            if self.trail_atr and atr_now > 0:
                new_stop = min(new_stop, float(df["Low"].tail(self.atr_len).min()) + self.trail_atr * atr_now)
            if new_stop < position["stop_price"]:
                return {"stop": new_stop}
        return None

    def stance(self, df: pd.DataFrame) -> str:
        if len(df) < self.min_bars or df.index.tz is None:
            return "warming up (not enough data)"
        rng = self._opening_range(self._today(df))
        if rng is None:
            return "opening range not formed yet"
        or_high, or_low, _ = rng
        c = float(df["Close"].iloc[-1])
        if c > or_high:
            return f"above OR high {or_high:,.2f} — broken out"
        if c < or_low:
            return f"below OR low {or_low:,.2f} — broken down"
        return f"inside opening range {or_low:,.2f}–{or_high:,.2f}"
