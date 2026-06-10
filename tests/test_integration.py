"""End-to-end test: synthetic market data through the full engine pipeline
(signal -> correlation filter -> ATR sizing -> paper fill -> reports),
no network required.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from bot import data, engine as engine_mod
from bot.engine import Engine
from bot.notify import Notifier


def make_ohlc(closes, freq="15min"):
    closes = pd.Series(closes, dtype=float)
    idx = pd.date_range("2026-01-05", periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({
        "Open": closes.values,
        "High": closes.values * 1.001,
        "Low": closes.values * 0.999,
        "Close": closes.values,
        "Volume": np.full(len(closes), 1000.0),
    }, index=idx)


def selloff(base):
    """Flat market ending in a sharp 3-bar drop: triggers mean-reversion LONG."""
    rng = np.random.default_rng(7)
    closes = list(base + rng.normal(0, base * 0.0002, 40))
    closes += [base * 0.997, base * 0.993, base * 0.988]
    return closes


def breakout(base):
    """Range then an upside breakout: triggers momentum LONG."""
    closes = list(base + np.sin(np.linspace(0, 8, 60)) * base * 0.003)
    closes.append(base * 1.03)
    return closes


def flat(base):
    rng = np.random.default_rng(11)
    return list(base + rng.normal(0, base * 0.0002, 80))


@pytest.fixture
def config(tmp_path):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["storage"]["db_path"] = str(tmp_path / "test.db")
    return cfg


@pytest.fixture
def market():
    return {
        "ES=F": make_ohlc(selloff(5000), "15min"),
        "NQ=F": make_ohlc(selloff(18000), "15min"),
        "BTC-USD": make_ohlc(breakout(60000), "1h"),
        "GC=F": make_ohlc(flat(2400), "4h"),
        "CL=F": make_ohlc(flat(70), "4h"),
    }


@pytest.fixture
def patched_engine(config, market, monkeypatch):
    def fake_fetch(ticker, timeframe):
        return market[ticker]
    monkeypatch.setattr(data, "fetch_candles", fake_fetch)
    monkeypatch.setattr(engine_mod.data, "fetch_candles", fake_fetch)
    return Engine(config, notifier=Notifier(token=None, chat_id=None))


def test_full_cycle_opens_positions_and_applies_correlation_filter(patched_engine):
    eng = patched_engine
    eng.run_once()

    sides = eng.portfolio.position_sides()
    # Both index longs open; BTC long (3rd risk-on long) must be blocked.
    assert sides.get("ES=F") == "long"
    assert sides.get("NQ=F") == "long"
    assert "BTC-USD" not in sides

    blocked = eng.store.events_since("2000-01-01", kind="blocked")
    assert any("BTC-USD" in ev["message"] for ev in blocked)

    # Sizing: each open position risks exactly 1% of equity at its stop.
    for pos in eng.portfolio.open_positions():
        loss_at_stop = abs(pos["entry_price"] - pos["stop_price"]) * pos["quantity"]
        assert loss_at_stop == pytest.approx(pos["risk_amount"], rel=1e-9)


def test_reports_render_after_trading(patched_engine):
    eng = patched_engine
    eng.run_once()

    morning = eng._build_morning()
    assert "Morning briefing" in morning
    for name in ("S&P 500", "Nasdaq 100", "Bitcoin", "Gold", "Oil"):
        assert name in morning
    assert "open long" in morning

    evening = eng._build_evening()
    assert "Daily report" in evening
    assert "Equity:" in evening
    assert "correlation filter" in evening


def test_stop_enforced_on_next_cycle(patched_engine, market):
    eng = patched_engine
    eng.run_once()
    pos = eng.store.get_position("ES=F")
    assert pos is not None

    # Next bar gaps down through the stop.
    crash = market["ES=F"].iloc[-1].copy()
    crash["Low"] = pos["stop_price"] * 0.99
    crash["Close"] = pos["stop_price"] * 0.995
    market["ES=F"].loc[market["ES=F"].index[-1] + pd.Timedelta(minutes=15)] = crash

    eng.run_once()
    assert eng.store.get_position("ES=F") is None
    trades = eng.store.all_trades()
    stop_trades = [t for t in trades if t["ticker"] == "ES=F" and t["reason"] == "stop"]
    assert len(stop_trades) == 1
    assert stop_trades[0]["exit_price"] == pytest.approx(pos["stop_price"])
