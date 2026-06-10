import numpy as np
import pandas as pd
import pytest

from bot.indicators import atr, donchian_high, donchian_low, ema, rsi, sma, zscore


def test_sma_matches_hand_computation():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_converges_to_constant():
    s = pd.Series([10.0] * 50)
    assert ema(s, 10).iloc[-1] == pytest.approx(10.0)


def test_zscore_zero_at_mean_and_signed_when_stretched():
    s = pd.Series([10.0] * 20 + [10.0])
    assert np.isnan(zscore(s, 20).iloc[-1])  # zero std -> no signal

    base = list(np.linspace(9.9, 10.1, 20))
    stretched = pd.Series(base + [12.0])
    assert zscore(stretched, 20).iloc[-1] > 2


def test_rsi_extremes():
    up = pd.Series(np.linspace(1, 100, 50))
    assert rsi(up, 2).iloc[-1] > 90
    down = pd.Series(np.linspace(100, 1, 50))
    assert rsi(down, 2).iloc[-1] < 10


def test_atr_simple_case():
    # Constant range of 2 with no gaps: ATR converges to 2.
    n = 200
    df = pd.DataFrame({
        "High": [11.0] * n,
        "Low": [9.0] * n,
        "Close": [10.0] * n,
    })
    assert atr(df, 14).iloc[-1] == pytest.approx(2.0, rel=1e-3)


def test_donchian_excludes_current_bar():
    highs = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    # Channel at the last bar uses the previous 3 bars only.
    assert donchian_high(highs, 3).iloc[-1] == pytest.approx(4.0)
    lows = pd.Series([5.0, 4.0, 3.0, 2.0, 0.5])
    assert donchian_low(lows, 3).iloc[-1] == pytest.approx(2.0)
