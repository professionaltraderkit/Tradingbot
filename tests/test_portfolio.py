import pytest

from bot.portfolio import PaperPortfolio, position_pnl
from bot.risk import TradePlan
from bot.state import StateStore


@pytest.fixture
def portfolio():
    store = StateStore(":memory:", starting_equity=100_000.0)
    yield PaperPortfolio(store)
    store.close()


def plan(side="long", qty=10.0, entry=5000.0, stop=4950.0):
    return TradePlan(side=side, quantity=qty, entry_price=entry,
                     stop_price=stop, risk_amount=abs(entry - stop) * qty)


def test_open_and_close_long_with_profit(portfolio):
    portfolio.open_position("ES=F", "S&P 500", plan())
    trade = portfolio.close_position("ES=F", 5020.0, reason="signal")
    assert trade["pnl"] == pytest.approx(200.0)
    assert portfolio.store.get_cash() == pytest.approx(100_200.0)
    assert portfolio.open_positions() == []


def test_short_pnl_sign(portfolio):
    portfolio.open_position("NQ=F", "Nasdaq 100", plan(side="short", entry=18000, stop=18100))
    trade = portfolio.close_position("NQ=F", 17900.0, reason="signal")
    assert trade["pnl"] == pytest.approx(1000.0)


def test_stop_enforced_at_stop_price(portfolio):
    portfolio.open_position("ES=F", "S&P 500", plan(entry=5000.0, stop=4950.0))
    # Bar trades through the stop: filled at the stop, loss = risk amount.
    trade = portfolio.check_stop("ES=F", bar_high=5005.0, bar_low=4940.0)
    assert trade is not None
    assert trade["exit_price"] == pytest.approx(4950.0)
    assert trade["pnl"] == pytest.approx(-500.0)
    assert trade["reason"] == "stop"


def test_stop_not_triggered_when_price_holds(portfolio):
    portfolio.open_position("ES=F", "S&P 500", plan(entry=5000.0, stop=4950.0))
    assert portfolio.check_stop("ES=F", bar_high=5010.0, bar_low=4960.0) is None
    assert len(portfolio.open_positions()) == 1


def test_short_stop_triggers_on_high(portfolio):
    portfolio.open_position("BTC-USD", "Bitcoin", plan(side="short", qty=1.0,
                                                       entry=60000, stop=61000))
    trade = portfolio.check_stop("BTC-USD", bar_high=61500.0, bar_low=60500.0)
    assert trade["pnl"] == pytest.approx(-1000.0)


def test_equity_includes_unrealized_pnl(portfolio):
    portfolio.open_position("ES=F", "S&P 500", plan(qty=10.0, entry=5000.0))
    assert portfolio.equity({"ES=F": 5010.0}) == pytest.approx(100_100.0)
    # Without a quote the position is valued at entry.
    assert portfolio.equity({}) == pytest.approx(100_000.0)


def test_no_double_position_per_ticker(portfolio):
    portfolio.open_position("ES=F", "S&P 500", plan())
    with pytest.raises(ValueError):
        portfolio.open_position("ES=F", "S&P 500", plan())


def test_positions_survive_restart(tmp_path):
    db = str(tmp_path / "state.db")
    store = StateStore(db, 100_000.0)
    PaperPortfolio(store).open_position("GC=F", "Gold", plan(qty=5.0, entry=2400, stop=2380))
    store.close()

    store2 = StateStore(db, 100_000.0)
    positions = PaperPortfolio(store2).open_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "GC=F"
    assert positions[0]["entry_price"] == pytest.approx(2400.0)
    store2.close()


def test_position_pnl_helper():
    pos = {"side": "long", "entry_price": 100.0, "quantity": 2.0}
    assert position_pnl(pos, 105.0) == pytest.approx(10.0)
    pos["side"] = "short"
    assert position_pnl(pos, 105.0) == pytest.approx(-10.0)
