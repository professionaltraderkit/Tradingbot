"""Thin wrapper around Alpaca's trading API (paper by default).

Only market orders are used; the engine owns stop logic so behavior stays
identical between the simulated broker and Alpaca execution.
"""

import logging
import os
import time

from .data import is_crypto

log = logging.getLogger(__name__)

_FILL_TIMEOUT_S = 30
_TERMINAL_FAILURES = {"canceled", "expired", "rejected"}


class OrderFailed(Exception):
    pass


class AlpacaBroker:
    def __init__(self, api_key: str | None = None, secret_key: str | None = None,
                 paper: bool = True):
        from alpaca.trading.client import TradingClient
        api_key = api_key or os.environ.get("ALPACA_API_KEY")
        secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not (api_key and secret_key):
            raise ValueError("ALPACA_API_KEY / ALPACA_SECRET_KEY not configured")
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def account_equity(self) -> float:
        return float(self.client.get_account().equity)

    def position_qty(self, symbol: str) -> float:
        """Quantity Alpaca currently holds for `symbol`, as a positive number
        of shares/units available to trade right now (0.0 when flat). Alpaca,
        not the local store, is the source of truth for what we can close."""
        from alpaca.common.exceptions import APIError

        try:
            pos = self.client.get_open_position(symbol)
        except APIError as exc:
            if exc.status_code == 404:  # no open position for this symbol
                return 0.0
            raise
        return abs(float(pos.qty_available))

    def submit_market_order(self, symbol: str, side: str, qty: float) -> float:
        """Submit a market order ('buy'/'sell') and block until filled.
        Returns the average fill price."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order = self.client.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC if is_crypto(symbol) else TimeInForce.DAY,
        ))
        deadline = time.monotonic() + _FILL_TIMEOUT_S
        while time.monotonic() < deadline:
            current = self.client.get_order_by_id(order.id)
            status = str(current.status.value if hasattr(current.status, "value")
                         else current.status)
            if status == "filled":
                return float(current.filled_avg_price)
            if status in _TERMINAL_FAILURES:
                raise OrderFailed(f"{side} {qty} {symbol}: order {status}")
            time.sleep(1)
        # Market order stuck (e.g. market closed): cancel so it can't fill later
        # without the bot knowing about it.
        try:
            self.client.cancel_order_by_id(order.id)
        except Exception:
            log.exception("Failed to cancel unfilled order %s", order.id)
        raise OrderFailed(f"{side} {qty} {symbol}: not filled within "
                          f"{_FILL_TIMEOUT_S}s (market closed?)")
