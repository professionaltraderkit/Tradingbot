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
