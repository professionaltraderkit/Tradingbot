import pytest

from bot.risk import RiskManager


@pytest.fixture
def risk():
    return RiskManager(
        risk_per_trade_pct=1.0,
        atr_stop_multiple=2.0,
        correlation_group=["ES=F", "NQ=F", "BTC-USD"],
        max_same_direction=2,
    )


def test_position_sized_so_stop_loss_is_one_percent(risk):
    equity, price, atr_value = 100_000.0, 5000.0, 25.0
    plan = risk.plan_trade("long", equity, price, atr_value)
    stop_distance = price - plan.stop_price
    assert stop_distance == pytest.approx(50.0)  # 2 x ATR
    loss_at_stop = stop_distance * plan.quantity
    assert loss_at_stop == pytest.approx(1000.0)  # exactly 1% of equity
    assert plan.risk_amount == pytest.approx(1000.0)


def test_quiet_market_gets_bigger_size_than_volatile(risk):
    quiet = risk.plan_trade("long", 100_000, 2000.0, 5.0)    # calm gold day
    volatile = risk.plan_trade("long", 100_000, 2000.0, 50.0)  # wild bitcoin day
    assert quiet.quantity > volatile.quantity
    # ...but dollar risk is identical.
    assert quiet.risk_amount == pytest.approx(volatile.risk_amount)


def test_short_stop_is_above_entry(risk):
    plan = risk.plan_trade("short", 100_000, 5000.0, 25.0)
    assert plan.stop_price == pytest.approx(5050.0)


def test_plan_rejects_bad_inputs(risk):
    assert risk.plan_trade("long", 100_000, 5000.0, 0.0) is None
    assert risk.plan_trade("long", 100_000, 0.0, 25.0) is None
    assert risk.plan_trade("long", 0.0, 5000.0, 25.0) is None


def test_correlation_filter_blocks_third_risk_on_long(risk):
    open_positions = {"ES=F": "long", "NQ=F": "long"}
    assert risk.correlation_blocked("BTC-USD", "long", open_positions)


def test_correlation_filter_allows_second_long_and_opposite_side(risk):
    assert not risk.correlation_blocked("NQ=F", "long", {"ES=F": "long"})
    # Shorting against two longs reduces net exposure — allowed.
    assert not risk.correlation_blocked(
        "BTC-USD", "short", {"ES=F": "long", "NQ=F": "long"}
    )


def test_correlation_filter_ignores_instruments_outside_group(risk):
    open_positions = {"ES=F": "long", "NQ=F": "long"}
    assert not risk.correlation_blocked("GC=F", "long", open_positions)
