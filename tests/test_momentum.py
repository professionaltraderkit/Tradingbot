"""Unit tests for the cross-sectional momentum selection logic (no network)."""

import numpy as np
import pandas as pd

from bot.momentum import CrossSectionalMomentum


def _panel():
    """400 business days: A strong up, B flat, C down, D mild up."""
    dates = pd.bdate_range("2020-01-01", periods=400)
    return pd.DataFrame({
        "A": np.linspace(100, 200, 400),   # +100%
        "B": np.full(400, 100.0),          # flat
        "C": np.linspace(100, 50, 400),    # -50%
        "D": np.linspace(100, 130, 400),   # +30%
    }, index=dates)


def test_ranks_strongest_first():
    panel = _panel()
    strat = CrossSectionalMomentum(lookback_days=252, skip_days=21, top_n=2,
                                   require_positive=True, regime_filter=False)
    w = strat.target_weights(panel, None, None, panel.index[-1])
    assert set(w) == {"A", "D"}                  # the two strongest positives
    assert all(abs(v - 0.5) < 1e-9 for v in w.values())   # 1/top_n each


def test_require_positive_drops_flat_and_losers():
    panel = _panel()
    strat = CrossSectionalMomentum(top_n=4, require_positive=True,
                                   regime_filter=False)
    w = strat.target_weights(panel, None, None, panel.index[-1])
    assert "C" not in w and "B" not in w         # downtrend & flat excluded
    assert set(w) == {"A", "D"}


def test_top_n_caps_holdings():
    panel = _panel()
    strat = CrossSectionalMomentum(top_n=1, require_positive=True,
                                   regime_filter=False)
    w = strat.target_weights(panel, None, None, panel.index[-1])
    assert set(w) == {"A"}                        # only the single strongest


def test_regime_filter_forces_cash_in_downtrend():
    panel = _panel()
    bench = pd.Series(np.linspace(200, 100, 400), index=panel.index)  # downtrend
    sma = bench.rolling(200).mean()
    strat = CrossSectionalMomentum(top_n=2, regime_filter=True, regime_sma=200)
    w = strat.target_weights(panel, bench, sma, panel.index[-1])
    assert w == {}                                # benchmark below SMA -> all cash


def test_regime_filter_allows_holdings_in_uptrend():
    panel = _panel()
    bench = pd.Series(np.linspace(100, 200, 400), index=panel.index)  # uptrend
    sma = bench.rolling(200).mean()
    strat = CrossSectionalMomentum(top_n=2, regime_filter=True, regime_sma=200)
    w = strat.target_weights(panel, bench, sma, panel.index[-1])
    assert set(w) == {"A", "D"}                   # benchmark above SMA -> normal book
