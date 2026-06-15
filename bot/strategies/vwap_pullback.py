"""VWAP trend pullback on 5-minute candles.

Python port of the user's TradingView "Level Reversal v5" Pine strategy:
trade WITH the intraday trend when price pulls back to session VWAP.

Long setup: price above a rising VWAP (trend), ADX confirms trend
strength, the bar's low touches VWAP (within tolerance) and the bar
closes back above it with a bullish body, on above-average volume.
Shorts are the mirror image below a falling VWAP.

Risk/management (mirrors the Pine bracket):
  - stop: beyond the pullback extreme/VWAP minus an ATR buffer
  - target: a fixed R multiple of the entry risk
  - room filter: don't buy into PDH / short into PDL — the target must
    clear the level or price must already be beyond it
  - breakeven ratchet: at +1R the stop moves to entry (never backwards)
  - no entries during the lunch chop or outside the entry window,
    always flat by the session close
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ..indicators import adx, atr, session_vwap
from .base import Signal, Strategy

ET = ZoneInfo("America/New_York")


def _parse_time(hhmm: str):
    return datetime.strptime(hhmm, "%H:%M").time()


class VwapPullback(Strategy):
    def __init__(self, tolerance_pct=0.05, stop_atr=0.75, target_r=1.5,
                 slope_bars=3, adx_len=14, adx_min=18.0, atr_len=14,
                 vol_len=20, use_lunch_filter=True, use_room_filter=True,
                 use_breakeven=True, breakeven_r=1.0, trail_atr=0.0,
                 use_volume_filter=True,
                 trade_start="09:45", trade_end="15:30",
                 lunch_start="11:30", lunch_end="13:30", flatten_at="15:50"):
        self.tolerance_pct = tolerance_pct
        self.stop_atr = stop_atr
        self.target_r = target_r
        self.slope_bars = slope_bars
        self.adx_len = adx_len
        self.adx_min = adx_min
        self.atr_len = atr_len
        self.vol_len = vol_len
        self.use_lunch_filter = use_lunch_filter
        self.use_room_filter = use_room_filter
        self.use_breakeven = use_breakeven
        self.breakeven_r = breakeven_r
        # trail_atr > 0 enables an ATR trailing stop (Chandelier-style) using
        # the highest high / lowest low of the last `atr_len` bars. Pair with
        # target_r = 0 to let winners run with no fixed cap.
        self.trail_atr = trail_atr
        self.use_volume_filter = use_volume_filter
        self.trade_start = _parse_time(trade_start)
        self.trade_end = _parse_time(trade_end)
        self.lunch_start = _parse_time(lunch_start)
        self.lunch_end = _parse_time(lunch_end)
        self.flatten_at = _parse_time(flatten_at)
        self.min_bars = max(2 * adx_len + slope_bars, vol_len, atr_len) + 5

    # -- helpers ---------------------------------------------------------------
    def _prev_day_levels(self, df: pd.DataFrame) -> tuple[float | None, float | None]:
        """Previous session's regular-hours high/low (PDH, PDL)."""
        d = df.tz_convert(ET)
        dates = sorted({ts.date() for ts in d.index})
        today = d.index[-1].date()
        prev_days = [x for x in dates if x < today]
        if not prev_days:
            return None, None
        prev = d[pd.Index(d.index.date) == prev_days[-1]]
        regular = prev.between_time("09:30", "16:00")
        src = regular if not regular.empty else prev
        return float(src["High"].max()), float(src["Low"].min())

    def _setup(self, df: pd.DataFrame):
        """Returns (signal, stop_price) for a flat book, or (HOLD, None)."""
        vwap = session_vwap(df)
        v_now = float(vwap.iloc[-1])
        v_back = float(vwap.iloc[-1 - self.slope_bars])
        adx_now = float(adx(df, self.adx_len).iloc[-1])
        atr_now = float(atr(df, self.atr_len).iloc[-1])
        last = df.iloc[-1]
        o, h, l, c = (float(last["Open"]), float(last["High"]),
                      float(last["Low"]), float(last["Close"]))
        if pd.isna(v_now) or pd.isna(adx_now) or pd.isna(atr_now) or atr_now <= 0:
            return Signal.HOLD, None

        vol_ok = True
        if self.use_volume_filter and "Volume" in df.columns:
            vol_sma = float(df["Volume"].rolling(self.vol_len).mean().iloc[-1])
            vol_ok = not pd.isna(vol_sma) and float(last["Volume"]) > vol_sma

        tol = c * self.tolerance_pct / 100
        trend_up = c > v_now and v_now > v_back
        trend_dn = c < v_now and v_now < v_back
        trend_ok = adx_now > self.adx_min

        long_setup = (trend_up and trend_ok and l <= v_now + tol and
                      c > v_now and c > o and vol_ok)
        short_setup = (trend_dn and trend_ok and h >= v_now - tol and
                       c < v_now and c < o and vol_ok)
        if long_setup == short_setup:  # neither, or conflicting
            return Signal.HOLD, None

        pdh, pdl = self._prev_day_levels(df) if self.use_room_filter else (None, None)
        if long_setup:
            stop = min(l, v_now) - self.stop_atr * atr_now
            risk = c - stop
            if risk <= 0:
                return Signal.HOLD, None
            target = c + self.target_r * risk
            if pdh is not None and not (c >= pdh or target <= pdh):
                return Signal.HOLD, None  # no room: target buried in PDH
            return Signal.LONG, stop
        stop = max(h, v_now) + self.stop_atr * atr_now
        risk = stop - c
        if risk <= 0:
            return Signal.HOLD, None
        target = c - self.target_r * risk
        if pdl is not None and not (c <= pdl or target >= pdl):
            return Signal.HOLD, None
        return Signal.SHORT, stop

    # -- Strategy interface ------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, position_side: str | None) -> Signal:
        if position_side:
            return Signal.HOLD  # exits are handled by manage()
        if len(df) < self.min_bars or df.index.tz is None:
            return Signal.HOLD
        now_et = df.index[-1].tz_convert(ET).time()
        if not (self.trade_start <= now_et <= self.trade_end):
            return Signal.HOLD
        if self.use_lunch_filter and self.lunch_start <= now_et < self.lunch_end:
            return Signal.HOLD
        signal, _ = self._setup(df)
        return signal

    def initial_stop(self, df: pd.DataFrame, side: str) -> float | None:
        _, stop = self._setup(df)
        return stop

    def manage(self, df: pd.DataFrame, position: dict) -> dict | None:
        if df.empty or df.index.tz is None:
            return None
        last = df.iloc[-1]
        now_et = df.index[-1].tz_convert(ET).time()
        if now_et >= self.flatten_at:
            return {"exit": "EOD flatten"}

        entry = position["entry_price"]
        risk_per_unit = position["risk_amount"] / position["quantity"]
        if risk_per_unit <= 0:
            return None

        atr_now = float(atr(df, self.atr_len).iloc[-1]) if self.trail_atr else 0.0
        if position["side"] == "long":
            if self.target_r > 0:
                target = entry + self.target_r * risk_per_unit
                if float(last["High"]) >= target:
                    return {"exit": "target", "price": target}
            new_stop = position["stop_price"]
            if (self.use_breakeven and float(last["High"]) >= entry
                    + self.breakeven_r * risk_per_unit):
                new_stop = max(new_stop, entry)
            if self.trail_atr and atr_now > 0:
                trail = float(df["High"].tail(self.atr_len).max()) - self.trail_atr * atr_now
                new_stop = max(new_stop, trail)
            if new_stop > position["stop_price"]:
                return {"stop": new_stop}
        else:
            if self.target_r > 0:
                target = entry - self.target_r * risk_per_unit
                if float(last["Low"]) <= target:
                    return {"exit": "target", "price": target}
            new_stop = position["stop_price"]
            if (self.use_breakeven and float(last["Low"]) <= entry
                    - self.breakeven_r * risk_per_unit):
                new_stop = min(new_stop, entry)
            if self.trail_atr and atr_now > 0:
                trail = float(df["Low"].tail(self.atr_len).min()) + self.trail_atr * atr_now
                new_stop = min(new_stop, trail)
            if new_stop < position["stop_price"]:
                return {"stop": new_stop}
        return None

    def stance(self, df: pd.DataFrame) -> str:
        if len(df) < self.min_bars or df.index.tz is None:
            return "warming up (not enough data)"
        vwap = session_vwap(df)
        v_now = float(vwap.iloc[-1])
        adx_now = float(adx(df, self.adx_len).iloc[-1])
        c = float(df["Close"].iloc[-1])
        if pd.isna(v_now):
            return "warming up (not enough data)"
        side = "above" if c > v_now else "below"
        dist = (c - v_now) / c * 100
        trend = "trending" if adx_now > self.adx_min else "choppy"
        return (f"{side} VWAP ({dist:+.2f}%), ADX {adx_now:.0f} ({trend}) — "
                f"waiting for a pullback to VWAP")
