"""LivePortfolio against a mocked Alpaca broker: fill handling, quantity
rounding, stop anchoring to the actual fill, and P&L from real exits."""

import pytest

from bot.portfolio import LivePortfolio, round_quantity
from bot.risk import TradePlan
from bot.state import StateStore


class FakeBroker:
    def __init__(self, equity=100_000.0, fills=None, holdings=None):
        self._equity = equity
        self.fills = list(fills or [])
        self.orders = []
        # symbol -> signed quantity Alpaca "holds", mirroring a real account
        # so position_qty() reflects fills (and any seeded drift).
        self.holdings = dict(holdings or {})

    def account_equity(self):
        return self._equity

    def position_qty(self, symbol):
        return abs(self.holdings.get(symbol, 0.0))

    def submit_market_order(self, symbol, side, qty):
        self.orders.append((symbol, side, qty))
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) + (
            qty if side == "buy" else -qty)
        return self.fills.pop(0)


@pytest.fixture
def store():
    s = StateStore(":memory:", 100_000.0)
    yield s
    s.close()


def plan(side="long", qty=10.0, entry=600.0, stop=598.0):
    return TradePlan(side=side, quantity=qty, entry_price=entry,
                     stop_price=stop, risk_amount=abs(entry - stop) * qty)


def test_round_quantity_rules():
    assert round_quantity("BTC/USD", "long", 0.12345678) == pytest.approx(0.123457)
    assert round_quantity("SPY", "long", 10.567) == pytest.approx(10.57)
    assert round_quantity("SPY", "short", 10.9) == 10.0  # whole shares only
    assert round_quantity("SPY", "short", 0.9) == 0.0


def test_open_uses_fill_price_and_anchors_stop_to_fill(store):
    broker = FakeBroker(fills=[600.50])  # slipped 0.50 above planned entry
    pf = LivePortfolio(store, broker)
    pos = pf.open_position("SPY", "S&P 500", plan(entry=600.0, stop=598.0))
    assert broker.orders == [("SPY", "buy", 10.0)]
    assert pos["entry_price"] == pytest.approx(600.50)
    # Stop distance (2.0) preserved relative to the actual fill.
    assert pos["stop_price"] == pytest.approx(598.50)


def test_short_entry_sells_and_stop_is_above(store):
    broker = FakeBroker(fills=[599.80])
    pf = LivePortfolio(store, broker)
    pos = pf.open_position("SPY", "S&P 500", plan(side="short", entry=600.0, stop=602.0))
    assert broker.orders == [("SPY", "sell", 10.0)]
    assert pos["stop_price"] == pytest.approx(601.80)


def test_zero_after_rounding_skips_order(store):
    broker = FakeBroker(fills=[600.0])
    pf = LivePortfolio(store, broker)
    pos = pf.open_position("SPY", "S&P 500", plan(side="short", qty=0.8))
    assert pos is None
    assert broker.orders == []
    assert pf.open_positions() == []


def test_close_records_pnl_from_actual_fill(store):
    broker = FakeBroker(fills=[600.0, 605.25])
    pf = LivePortfolio(store, broker)
    pf.open_position("SPY", "S&P 500", plan())
    trade = pf.close_position("SPY", 605.0, reason="signal")
    assert broker.orders[1] == ("SPY", "sell", 10.0)
    assert trade["exit_price"] == pytest.approx(605.25)
    assert trade["pnl"] == pytest.approx(52.50)
    assert pf.open_positions() == []


def test_stop_closes_at_market_with_real_slippage(store):
    # Entry fills at 600; stop is 598. Market exit fills at 597.60 — the
    # reported loss includes the 0.40 slippage past the stop.
    broker = FakeBroker(fills=[600.0, 597.60])
    pf = LivePortfolio(store, broker)
    pf.open_position("SPY", "S&P 500", plan(entry=600.0, stop=598.0))
    trade = pf.check_stop("SPY", 597.9, 597.9)
    assert trade is not None
    assert trade["reason"] == "stop"
    assert trade["pnl"] == pytest.approx(-24.0)


def test_stop_not_triggered_above_stop_price(store):
    broker = FakeBroker(fills=[600.0])
    pf = LivePortfolio(store, broker)
    pf.open_position("SPY", "S&P 500", plan(entry=600.0, stop=598.0))
    assert pf.check_stop("SPY", 599.0, 599.0) is None
    assert len(pf.open_positions()) == 1


def _seed_position(store, ticker="SPY", side="long", qty=10.0, entry=600.0,
                   stop=598.0):
    store.save_position({
        "ticker": ticker, "name": "S&P 500", "side": side, "quantity": qty,
        "entry_price": entry, "stop_price": stop, "risk_amount": 200.0,
        "opened_at": "2026-01-01T00:00:00+00:00",
    })


def test_close_sizes_to_alpaca_held_quantity_when_local_drifts(store):
    # Local store claims 133.74 SPY but Alpaca only holds 1.84 (the partial
    # fill that produced the original "insufficient qty" 403). The close must
    # size to what's really there, not the stale local number.
    broker = FakeBroker(fills=[605.0], holdings={"SPY": 1.84})
    pf = LivePortfolio(store, broker)
    _seed_position(store, qty=133.74)
    trade = pf.close_position("SPY", 605.0, reason="stop")
    assert broker.orders == [("SPY", "sell", 1.84)]
    assert trade["quantity"] == pytest.approx(1.84)
    assert trade["pnl"] == pytest.approx((605.0 - 600.0) * 1.84)
    assert pf.open_positions() == []


def test_close_reconciles_when_alpaca_shows_no_position(store):
    # Alpaca holds nothing (closed out-of-band). The stale local record is
    # cleared without sending an order that would just be rejected.
    broker = FakeBroker(holdings={})
    pf = LivePortfolio(store, broker)
    _seed_position(store, qty=10.0)
    assert pf.close_position("SPY", 605.0, reason="stop") is None
    assert broker.orders == []
    assert pf.open_positions() == []


def test_equity_comes_from_broker(store):
    pf = LivePortfolio(store, FakeBroker(equity=123_456.78))
    assert pf.equity({}) == pytest.approx(123_456.78)
