"""Risk management: ATR-based sizing, hard 1% stop, correlation filter."""

from dataclasses import dataclass


@dataclass
class TradePlan:
    side: str          # "long" or "short"
    quantity: float    # units of the instrument (fractional allowed)
    entry_price: float
    stop_price: float
    risk_amount: float  # dollars at risk if the stop is hit


class RiskManager:
    def __init__(self, risk_per_trade_pct: float, atr_stop_multiple: float,
                 correlation_group: list[str], max_same_direction: int):
        self.risk_per_trade_pct = risk_per_trade_pct
        self.atr_stop_multiple = atr_stop_multiple
        self.correlation_group = set(correlation_group)
        self.max_same_direction = max_same_direction

    def plan_trade(self, side: str, equity: float, price: float,
                   atr_value: float) -> TradePlan | None:
        """Size a position so the loss at the ATR stop is exactly
        `risk_per_trade_pct` of equity. Returns None if inputs are unusable."""
        if price <= 0 or atr_value <= 0 or equity <= 0:
            return None
        stop_distance = self.atr_stop_multiple * atr_value
        risk_amount = equity * self.risk_per_trade_pct / 100
        quantity = risk_amount / stop_distance
        if side == "long":
            stop_price = price - stop_distance
        elif side == "short":
            stop_price = price + stop_distance
        else:
            raise ValueError(f"Invalid side: {side}")
        if stop_price <= 0:
            return None
        return TradePlan(side=side, quantity=quantity, entry_price=price,
                         stop_price=stop_price, risk_amount=risk_amount)

    def correlation_blocked(self, ticker: str, side: str,
                            open_positions: dict[str, str]) -> bool:
        """True if opening `side` on `ticker` would exceed the cap on
        same-direction exposure within the correlation group.

        `open_positions` maps ticker -> side for currently open positions.
        """
        if ticker not in self.correlation_group:
            return False
        same = sum(
            1 for t, s in open_positions.items()
            if t in self.correlation_group and s == side and t != ticker
        )
        return same >= self.max_same_direction
