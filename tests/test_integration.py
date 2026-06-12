"""End-to-end test: synthetic market data through the full engine pipeline
(signal -> risk rules -> ATR sizing -> fill -> reports), no network required.
Uses the simulated broker; LivePortfolio order mechanics are covered in
test_live_portfolio.py.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

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


def breakdown(base):
    """Range then a downside break: triggers momentum SHORT."""
    closes = list(base + np.sin(np.linspace(0, 8, 60)) * base * 0.003)
    closes.append(base * 0.97)
    return closes


def flat(base):
    rng = np.random.default_rng(11)
    return list(base + rng.normal(0, base * 0.0002, 80))


class FakeFeed:
    def __init__(self, market):
        self.market = market

    def fetch_candles(self, ticker, timeframe):
        return self.market[ticker]

    def latest_price(self, ticker):
        df = self.market[ticker]
        return float(df["Close"].iloc[-1])


@pytest.fixture
def config(tmp_path):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["storage"]["db_path"] = str(tmp_path / "test.db")
    cfg["broker"]["mode"] = "simulated"
    return cfg


@pytest.fixture
def market():
    return {
        "SPY": make_ohlc(selloff(600), "15min"),
        "QQQ": make_ohlc(selloff(520), "15min"),
        "BTC/USD": make_ohlc(breakout(60000), "1h"),
        "GLD": make_ohlc(flat(220), "4h"),
        "USO": make_ohlc(flat(70), "4h"),
    }


def build_engine(config, market):
    return Engine(config, notifier=Notifier(token=None, chat_id=None),
                  feed=FakeFeed(market))


def test_full_cycle_opens_positions_and_applies_correlation_filter(config, market):
    eng = build_engine(config, market)
    eng.run_once()

    sides = eng.portfolio.position_sides()
    # Both index longs open; BTC long (3rd risk-on long) must be blocked.
    assert sides.get("SPY") == "long"
    assert sides.get("QQQ") == "long"
    assert "BTC/USD" not in sides

    blocked = eng.store.events_since("2000-01-01", kind="blocked")
    assert any("BTC/USD" in ev["message"] and "correlation" in ev["message"]
               for ev in blocked)

    # Sizing: loss at the stop equals the recorded risk, and risk never
    # exceeds 1% of starting equity.
    for pos in eng.portfolio.open_positions():
        loss_at_stop = abs(pos["entry_price"] - pos["stop_price"]) * pos["quantity"]
        assert loss_at_stop == pytest.approx(pos["risk_amount"], rel=1e-9)
        assert pos["risk_amount"] <= eng.starting_equity * 0.01 + 1e-6
        # Notional cap respected (100% of equity).
        assert pos["entry_price"] * pos["quantity"] <= eng.starting_equity * 1.0001


def test_long_only_instrument_skips_short_signal(config, market):
    market["BTC/USD"] = make_ohlc(breakdown(60000), "1h")
    eng = build_engine(config, market)
    eng.run_once()

    assert "BTC/USD" not in eng.portfolio.position_sides()
    blocked = eng.store.events_since("2000-01-01", kind="blocked")
    assert any("long-only" in ev["message"] for ev in blocked)


def test_reports_render_after_trading(config, market):
    eng = build_engine(config, market)
    eng.run_once()

    morning = eng._build_morning()
    assert "Morning briefing" in morning
    for name in ("S&P 500", "Nasdaq 100", "Bitcoin", "Gold", "Oil"):
        assert name in morning
    assert "open long" in morning

    evening = eng._build_evening()
    assert "Daily report" in evening
    assert "Equity:" in evening
    assert "skipped" in evening


def test_stop_enforced_on_next_cycle(config, market):
    eng = build_engine(config, market)
    eng.run_once()
    pos = eng.store.get_position("SPY")
    assert pos is not None

    # Next bar gaps down through the stop.
    crash = market["SPY"].iloc[-1].copy()
    crash["Low"] = pos["stop_price"] * 0.99
    crash["Close"] = pos["stop_price"] * 0.995
    market["SPY"].loc[market["SPY"].index[-1] + pd.Timedelta(minutes=15)] = crash

    eng.run_once()
    assert eng.store.get_position("SPY") is None
    trades = eng.store.all_trades()
    stop_trades = [t for t in trades if t["ticker"] == "SPY" and t["reason"] == "stop"]
    assert len(stop_trades) == 1
    assert stop_trades[0]["exit_price"] == pytest.approx(pos["stop_price"])


def test_stop_sweep_between_candles(config, market):
    eng = build_engine(config, market)
    eng.run_once()
    pos = eng.store.get_position("QQQ")
    assert pos is not None

    # Latest trade pierces the stop without a new candle being due.
    market["QQQ"].loc[market["QQQ"].index[-1], "Close"] = pos["stop_price"] * 0.99
    eng.sweep_stops()
    assert eng.store.get_position("QQQ") is None


def test_vwap_bot_config_runs_end_to_end(tmp_path):
    """Boot the second bot from config-levels.yaml: a pullback to a rising
    VWAP opens longs with the strategy's structure stop, the breakeven
    ratchet tightens the stop, and the target closes the trade."""
    from tests.test_strategies import TestVwapPullback

    with open("config-levels.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["storage"]["db_path"] = str(tmp_path / "levels.db")
    cfg["broker"]["mode"] = "simulated"

    helper = TestVwapPullback()
    market = {t: helper._with_pullback_bar(helper._trend_frame())
              for t in ("SPY", "QQQ")}
    eng = Engine(cfg, notifier=Notifier(token=None, chat_id=None),
                 feed=FakeFeed(market))
    eng.run_once()

    sides = eng.portfolio.position_sides()
    assert sides == {"SPY": "long", "QQQ": "long"}
    pos = eng.store.get_position("SPY")
    # Structure stop below the pullback bar's low, sized so the loss at the
    # stop equals the recorded risk.
    assert pos["stop_price"] < float(market["SPY"]["Low"].iloc[-1])
    loss_at_stop = (pos["entry_price"] - pos["stop_price"]) * pos["quantity"]
    assert loss_at_stop == pytest.approx(pos["risk_amount"], rel=1e-9)

    morning = eng._build_morning()
    assert "VWAP Pullback Bot" in morning
    assert "VWAP" in morning

    # Next bar runs through +1R and then the 1.5R target: the engine should
    # ratchet the stop and close at the target.
    entry, stop = pos["entry_price"], pos["stop_price"]
    rk = pos["risk_amount"] / pos["quantity"]
    for t in ("SPY", "QQQ"):
        ts = market[t].index[-1] + pd.Timedelta(minutes=5)
        market[t].loc[ts] = [entry + 0.2, entry + 2 * rk, entry + 0.1,
                             entry + 1.8 * rk, 1000.0]
    eng.run_once()

    assert eng.portfolio.position_sides() == {}
    trades = [t for t in eng.store.all_trades() if t["ticker"] == "SPY"]
    assert len(trades) == 1
    assert trades[0]["reason"] == "target"
    assert trades[0]["exit_price"] == pytest.approx(entry + 1.5 * rk)
