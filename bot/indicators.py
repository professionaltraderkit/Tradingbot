"""Technical indicators implemented on pandas, no external TA library."""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def zscore(series: pd.Series, period: int) -> pd.Series:
    mean = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # All-gain windows have zero loss -> RSI is 100 by definition.
    return out.fillna(100).where(series.notna())


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range from an OHLC frame (Wilder smoothing)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def donchian_high(series: pd.Series, period: int) -> pd.Series:
    """Highest high of the *previous* `period` bars (excludes current bar)."""
    return series.rolling(period).max().shift(1)


def donchian_low(series: pd.Series, period: int) -> pd.Series:
    """Lowest low of the *previous* `period` bars (excludes current bar)."""
    return series.rolling(period).min().shift(1)


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Average Directional Index (Wilder). Measures trend strength."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    alpha = 1 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.fillna(0).ewm(alpha=alpha, adjust=False).mean()


def session_vwap(df: pd.DataFrame, tz: str = "America/New_York") -> pd.Series:
    """Volume-weighted average price, anchored to the start of each
    calendar day in `tz` (mirrors TradingView's session VWAP closely).
    Falls back to equal weights if there is no usable volume."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        vol = df["Volume"].astype(float).clip(lower=1e-9)
    else:
        vol = pd.Series(1.0, index=df.index)
    if df.index.tz is not None:
        day = pd.Index(df.index.tz_convert(tz).date)
    else:
        day = pd.Index(df.index.date)
    pv = (tp * vol).groupby(day).cumsum()
    vv = vol.groupby(day).cumsum()
    return pv / vv
