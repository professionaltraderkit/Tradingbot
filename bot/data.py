"""Market data layer.

Primary source is Alpaca (free plan: real-time IEX feed for stocks/ETFs,
keyless real-time crypto). When no Alpaca keys are configured it falls
back to Yahoo Finance via yfinance so backtests and the simulated broker
work without any account.

Tickers are Alpaca-style: "SPY", "GLD", "BTC/USD". Crypto is anything
with a slash. Callers get an empty DataFrame on failure and are expected
to skip the cycle rather than crash.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

log = logging.getLogger(__name__)

_LOOKBACK_DAYS = {"15m": 10, "1h": 60, "4h": 120}
_MAX_RETRIES = 3

# yfinance fallback fetch plan: (interval, period). 4h is resampled from 1h.
_YF_PLAN = {
    "15m": ("15m", "10d"),
    "1h": ("1h", "60d"),
    "4h": ("1h", "120d"),
}


def is_crypto(ticker: str) -> bool:
    return "/" in ticker


def _yf_symbol(ticker: str) -> str:
    return ticker.replace("/", "-")  # BTC/USD -> BTC-USD


class MarketData:
    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY") or None
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY") or None
        self._stock_client = None
        self._crypto_client = None
        if not self.alpaca_enabled:
            log.warning("No Alpaca keys configured — using Yahoo Finance data")

    @property
    def alpaca_enabled(self) -> bool:
        return bool(self.api_key and self.secret_key)

    # -- public API -----------------------------------------------------------
    def fetch_candles(self, ticker: str, timeframe: str) -> pd.DataFrame:
        """OHLCV DataFrame for `ticker` at `timeframe` (UTC index), empty on failure."""
        if timeframe not in _LOOKBACK_DAYS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                if self.alpaca_enabled:
                    df = self._alpaca_candles(ticker, timeframe)
                else:
                    df = self._yf_candles(ticker, timeframe)
                if df is not None and not df.empty:
                    return df
                log.warning("Empty data for %s (attempt %d/%d)", ticker, attempt, _MAX_RETRIES)
            except Exception as exc:
                log.warning("Fetch failed for %s (attempt %d/%d): %s",
                            ticker, attempt, _MAX_RETRIES, exc)
            if attempt < _MAX_RETRIES:
                time.sleep(2 * attempt)
        return pd.DataFrame()

    def latest_price(self, ticker: str) -> float | None:
        """Most recent trade price, used for the between-candle stop sweep."""
        try:
            if self.alpaca_enabled:
                return self._alpaca_latest(ticker)
            df = self._yf_candles(ticker, "15m" if not is_crypto(ticker) else "1h")
            return float(df["Close"].iloc[-1]) if not df.empty else None
        except Exception as exc:
            log.warning("Latest price failed for %s: %s", ticker, exc)
            return None

    # -- Alpaca ----------------------------------------------------------------
    def _alpaca_timeframe(self, timeframe: str):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        return {
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "1h": TimeFrame(1, TimeFrameUnit.Hour),
            "4h": TimeFrame(4, TimeFrameUnit.Hour),
        }[timeframe]

    def _stock(self):
        if self._stock_client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._stock_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self._stock_client

    def _crypto(self):
        if self._crypto_client is None:
            from alpaca.data.historical import CryptoHistoricalDataClient
            self._crypto_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
        return self._crypto_client

    def _alpaca_candles(self, ticker: str, timeframe: str) -> pd.DataFrame:
        start = datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS[timeframe])
        tf = self._alpaca_timeframe(timeframe)
        if is_crypto(ticker):
            from alpaca.data.requests import CryptoBarsRequest
            req = CryptoBarsRequest(symbol_or_symbols=ticker, timeframe=tf, start=start)
            bars = self._crypto().get_crypto_bars(req)
        else:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import StockBarsRequest
            req = StockBarsRequest(symbol_or_symbols=ticker, timeframe=tf,
                                   start=start, feed=DataFeed.IEX)
            bars = self._stock().get_stock_bars(req)
        df = bars.df
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                "close": "Close", "volume": "Volume"})
        cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        return df[cols].dropna(subset=["Open", "High", "Low", "Close"])

    def _alpaca_latest(self, ticker: str) -> float | None:
        if is_crypto(ticker):
            from alpaca.data.requests import CryptoLatestTradeRequest
            req = CryptoLatestTradeRequest(symbol_or_symbols=ticker)
            trade = self._crypto().get_crypto_latest_trade(req)
        else:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import StockLatestTradeRequest
            req = StockLatestTradeRequest(symbol_or_symbols=ticker, feed=DataFeed.IEX)
            trade = self._stock().get_stock_latest_trade(req)
        return float(trade[ticker].price)

    # -- Yahoo Finance fallback --------------------------------------------------
    def _yf_candles(self, ticker: str, timeframe: str) -> pd.DataFrame:
        import yfinance as yf
        interval, period = _YF_PLAN[timeframe]
        df = yf.download(_yf_symbol(ticker), interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        df = df[cols].dropna(subset=["Open", "High", "Low", "Close"])
        if timeframe == "4h":
            df = resample_4h(df)
        return df


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h candles into 4h candles aligned to 00/04/08/12/16/20 UTC."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    out = df.resample("4h").agg(agg)
    return out.dropna(subset=["Open", "High", "Low", "Close"])
