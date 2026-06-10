"""Market data layer: yfinance fetching with retries and 1h -> 4h resampling.

Yahoo has no native 4h interval, so 4h instruments are fetched at 1h and
resampled. Futures tickers (ES=F, GC=F, CL=F) have session gaps and the
feed is occasionally spotty; callers get an empty DataFrame on failure and
are expected to skip the cycle rather than crash.
"""

import logging
import time

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

# Lookback windows sized to give strategies enough history while staying
# inside Yahoo's intraday limits (15m: 60 days, 1h: 730 days).
_FETCH_PLAN = {
    "15m": ("15m", "10d"),
    "1h": ("1h", "60d"),
    "4h": ("1h", "120d"),  # fetched at 1h, resampled below
}

_MAX_RETRIES = 3


def fetch_candles(ticker: str, timeframe: str) -> pd.DataFrame:
    """Return an OHLCV DataFrame for `ticker` at `timeframe`, empty on failure."""
    if timeframe not in _FETCH_PLAN:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    interval, period = _FETCH_PLAN[timeframe]

    df = pd.DataFrame()
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            df = yf.download(
                ticker,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=True,
            )
            if df is not None and not df.empty:
                break
            log.warning("Empty data for %s (attempt %d/%d)", ticker, attempt, _MAX_RETRIES)
        except Exception as exc:  # network/API hiccups
            log.warning("Fetch failed for %s (attempt %d/%d): %s", ticker, attempt, _MAX_RETRIES, exc)
        if attempt < _MAX_RETRIES:
            time.sleep(2 * attempt)

    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize(df)
    if timeframe == "4h":
        df = resample_4h(df)
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's MultiIndex columns and drop incomplete rows."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    return df[cols].dropna(subset=["Open", "High", "Low", "Close"])


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h candles into 4h candles aligned to 00/04/08/12/16/20."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    out = df.resample("4h").agg(agg)
    return out.dropna(subset=["Open", "High", "Low", "Close"])
