"""Cross-sectional (relative-strength) momentum selection logic.

This is a portfolio-level strategy, unlike the per-symbol classes in
bot/strategies/ (which look at one instrument's candles and return a
Signal). It ranks a whole universe at once, so it has a different shape —
target weights for a rebalance date — and its own engine in
momentum_backtest.py rather than the bar-by-bar backtest.py.

The edge: relative strength persists. The catch handled here is the
"momentum crash" — sharp reversals after market bottoms — which the
optional benchmark regime filter sidesteps by going to cash when the
benchmark is below its long-term average.
"""
from __future__ import annotations

import pandas as pd


class CrossSectionalMomentum:
    """Rank a universe by 12-1 momentum, hold the top N equal-weight.

    lookback_days / skip_days define the signal: the return from
    `lookback_days` ago up to `skip_days` ago. Skipping the most recent
    ~month (21 trading days) avoids short-term reversal contaminating the
    momentum signal — the canonical "12-1" academic construction.
    """

    def __init__(self, lookback_days: int = 252, skip_days: int = 21,
                 top_n: int = 4, require_positive: bool = True,
                 regime_filter: bool = True, regime_sma: int = 200):
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.top_n = top_n
        self.require_positive = require_positive
        self.regime_filter = regime_filter
        self.regime_sma = regime_sma

    @property
    def min_history(self) -> int:
        """Bars of history needed before the signal is valid."""
        return self.lookback_days + 1

    def momentum(self, panel: pd.DataFrame, date) -> pd.Series:
        """12-1 momentum for every column at `date` (NaN where history is short)."""
        i = panel.index.get_loc(date)
        if i - self.lookback_days < 0:
            return pd.Series(index=panel.columns, dtype=float)
        past = panel.iloc[i - self.lookback_days]
        recent = panel.iloc[i - self.skip_days]
        return recent / past - 1.0

    def in_uptrend(self, bench_close: pd.Series | None,
                   bench_sma: pd.Series | None, date) -> bool:
        """True when the regime filter is off, or the benchmark is at/above
        its SMA. A NaN SMA (not enough history) is treated as risk-off."""
        if not self.regime_filter:
            return True
        sma = bench_sma.loc[date]
        price = bench_close.loc[date]
        if pd.isna(sma) or pd.isna(price):
            return False
        return price >= sma

    def target_weights(self, panel: pd.DataFrame, bench_close: pd.Series | None,
                       bench_sma: pd.Series | None, date) -> dict:
        """{ticker: weight} to hold for the month starting at `date`.

        Empty dict = all cash (regime off, or nothing qualifies). Each held
        name gets 1/top_n, so when fewer than top_n names clear the
        positive-momentum hurdle the remainder stays in cash."""
        if not self.in_uptrend(bench_close, bench_sma, date):
            return {}
        mom = self.momentum(panel, date).dropna()
        if self.require_positive:
            mom = mom[mom > 0]
        chosen = mom.sort_values(ascending=False).index[:self.top_n]
        if len(chosen) == 0:
            return {}
        weight = 1.0 / self.top_n
        return {ticker: weight for ticker in chosen}
