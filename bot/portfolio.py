"""Portfolio backends.

PaperPortfolio: internal simulator (fills at the given price), used by the
"simulated" broker mode and all backtests.

LivePortfolio: routes orders to an Alpaca account (paper or live). Alpaca
is the source of truth for equity and fills; stop prices, risk metadata
and trade history are mirrored in the local SQLite store because Alpaca
doesn't hold them.

Both expose the same interface: open_position / close_position /
check_stop / equity / open_positions / position_sides.
"""

import logging
from datetime import datetime, timezone

from .data import is_crypto
from .risk import TradePlan
from .state import StateStore

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def position_pnl(pos: dict, price: float) -> float:
    direction = 1 if pos["side"] == "long" else -1
    return (price - pos["entry_price"]) * pos["quantity"] * direction


def round_quantity(ticker: str, side: str, quantity: float) -> float:
    """Clamp a raw size to what Alpaca accepts: crypto trades fractionally,
    stocks fractionally only when long — shorts must be whole shares."""
    if is_crypto(ticker):
        return round(quantity, 6)
    if side == "short":
        return float(int(quantity))
    return round(quantity, 2)


class _PortfolioBase:
    def __init__(self, store: StateStore):
        self.store = store

    def open_positions(self) -> list[dict]:
        return self.store.open_positions()

    def position_sides(self) -> dict[str, str]:
        """ticker -> side for all open positions (for the correlation filter)."""
        return {p["ticker"]: p["side"] for p in self.open_positions()}

    def _save_open(self, ticker: str, name: str, plan: TradePlan,
                   entry_price: float, stop_price: float) -> dict:
        pos = {
            "ticker": ticker,
            "name": name,
            "side": plan.side,
            "quantity": plan.quantity,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "risk_amount": plan.risk_amount,
            "opened_at": _now(),
        }
        self.store.save_position(pos)
        self.store.log_event(
            "fill",
            f"OPEN {plan.side.upper()} {name} ({ticker}) qty={plan.quantity:.4f} "
            f"@ {entry_price:.2f}, stop {stop_price:.2f}, "
            f"risk ${plan.risk_amount:.2f}",
        )
        log.info("Opened %s %s @ %.2f (stop %.2f)", plan.side, ticker,
                 entry_price, stop_price)
        return pos

    def _record_close(self, pos: dict, exit_price: float, reason: str) -> dict:
        pnl = position_pnl(pos, exit_price)
        trade = {
            "ticker": pos["ticker"],
            "name": pos["name"],
            "side": pos["side"],
            "quantity": pos["quantity"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
            "opened_at": pos["opened_at"],
            "closed_at": _now(),
        }
        self.store.record_trade(trade)
        self.store.delete_position(pos["ticker"])
        self.store.log_event(
            "fill",
            f"CLOSE {pos['side'].upper()} {pos['name']} ({pos['ticker']}) "
            f"@ {exit_price:.2f} ({reason}): P&L ${pnl:+.2f}",
        )
        log.info("Closed %s %s @ %.2f (%s): pnl %+.2f", pos["side"],
                 pos["ticker"], exit_price, reason, pnl)
        return trade

    @staticmethod
    def _stop_hit(pos: dict, bar_high: float, bar_low: float) -> bool:
        stop = pos["stop_price"]
        return (pos["side"] == "long" and bar_low <= stop) or \
               (pos["side"] == "short" and bar_high >= stop)


class PaperPortfolio(_PortfolioBase):
    """Simulated fills at the requested price; cash tracked in SQLite."""

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus unrealized P&L, using `prices` (ticker -> last price).
        Positions without a quote are valued at entry (zero unrealized)."""
        total = self.store.get_cash()
        for pos in self.open_positions():
            price = prices.get(pos["ticker"], pos["entry_price"])
            total += position_pnl(pos, price)
        return total

    def open_position(self, ticker: str, name: str, plan: TradePlan) -> dict:
        if self.store.get_position(ticker):
            raise ValueError(f"Position already open for {ticker}")
        return self._save_open(ticker, name, plan, plan.entry_price, plan.stop_price)

    def close_position(self, ticker: str, price: float, reason: str) -> dict | None:
        pos = self.store.get_position(ticker)
        if not pos:
            return None
        trade = self._record_close(pos, price, reason)
        self.store.set_cash(self.store.get_cash() + trade["pnl"])
        return trade

    def check_stop(self, ticker: str, bar_high: float, bar_low: float) -> dict | None:
        """Enforce the hard stop against the latest bar's range. Fills at the
        stop price (assumes no gap through it). Returns the closed trade."""
        pos = self.store.get_position(ticker)
        if not pos or not self._stop_hit(pos, bar_high, bar_low):
            return None
        return self.close_position(ticker, pos["stop_price"], reason="stop")


class LivePortfolio(_PortfolioBase):
    """Executes on an Alpaca account via market orders."""

    def __init__(self, store: StateStore, broker):
        super().__init__(store)
        self.broker = broker

    def equity(self, prices: dict[str, float] | None = None) -> float:
        return self.broker.account_equity()

    def open_position(self, ticker: str, name: str, plan: TradePlan) -> dict | None:
        if self.store.get_position(ticker):
            raise ValueError(f"Position already open for {ticker}")
        qty = round_quantity(ticker, plan.side, plan.quantity)
        if qty <= 0:
            log.warning("Size for %s rounded to zero (raw %.6f), skipping",
                        ticker, plan.quantity)
            return None
        plan.quantity = qty
        order_side = "buy" if plan.side == "long" else "sell"
        fill = self.broker.submit_market_order(ticker, order_side, qty)
        # Keep the planned stop *distance* but anchor it to the actual fill,
        # so slippage on entry can't widen the risk.
        stop_distance = abs(plan.entry_price - plan.stop_price)
        stop = fill - stop_distance if plan.side == "long" else fill + stop_distance
        return self._save_open(ticker, name, plan, fill, stop)

    def close_position(self, ticker: str, price: float, reason: str) -> dict | None:
        """`price` is only a hint for logging; the real exit is a market fill."""
        pos = self.store.get_position(ticker)
        if not pos:
            return None
        order_side = "sell" if pos["side"] == "long" else "buy"
        fill = self.broker.submit_market_order(ticker, order_side, pos["quantity"])
        return self._record_close(pos, fill, reason)

    def check_stop(self, ticker: str, bar_high: float, bar_low: float) -> dict | None:
        """Close at market when the stop level is breached. The fill may be
        slightly past the stop price — that slippage is real and reported."""
        pos = self.store.get_position(ticker)
        if not pos or not self._stop_hit(pos, bar_high, bar_low):
            return None
        return self.close_position(ticker, pos["stop_price"], reason="stop")
