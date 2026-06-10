"""Paper broker: simulated fills at the latest close, stop enforcement,
P&L accounting. One position per instrument at a time.
"""

import logging
from datetime import datetime, timezone

from .risk import TradePlan
from .state import StateStore

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def position_pnl(pos: dict, price: float) -> float:
    direction = 1 if pos["side"] == "long" else -1
    return (price - pos["entry_price"]) * pos["quantity"] * direction


class PaperPortfolio:
    def __init__(self, store: StateStore):
        self.store = store

    # -- queries -------------------------------------------------------------
    def open_positions(self) -> list[dict]:
        return self.store.open_positions()

    def position_sides(self) -> dict[str, str]:
        """ticker -> side for all open positions (for the correlation filter)."""
        return {p["ticker"]: p["side"] for p in self.open_positions()}

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus unrealized P&L, using `prices` (ticker -> last price).
        Positions without a quote are valued at entry (zero unrealized)."""
        total = self.store.get_cash()
        for pos in self.open_positions():
            price = prices.get(pos["ticker"], pos["entry_price"])
            total += position_pnl(pos, price)
        return total

    # -- trading ---------------------------------------------------------------
    def open_position(self, ticker: str, name: str, plan: TradePlan) -> dict:
        if self.store.get_position(ticker):
            raise ValueError(f"Position already open for {ticker}")
        pos = {
            "ticker": ticker,
            "name": name,
            "side": plan.side,
            "quantity": plan.quantity,
            "entry_price": plan.entry_price,
            "stop_price": plan.stop_price,
            "risk_amount": plan.risk_amount,
            "opened_at": _now(),
        }
        self.store.save_position(pos)
        self.store.log_event(
            "fill",
            f"OPEN {plan.side.upper()} {name} ({ticker}) qty={plan.quantity:.4f} "
            f"@ {plan.entry_price:.2f}, stop {plan.stop_price:.2f}, "
            f"risk ${plan.risk_amount:.2f}",
        )
        log.info("Opened %s %s @ %.2f (stop %.2f)", plan.side, ticker,
                 plan.entry_price, plan.stop_price)
        return pos

    def close_position(self, ticker: str, price: float, reason: str) -> dict | None:
        pos = self.store.get_position(ticker)
        if not pos:
            return None
        pnl = position_pnl(pos, price)
        self.store.set_cash(self.store.get_cash() + pnl)
        trade = {
            "ticker": ticker,
            "name": pos["name"],
            "side": pos["side"],
            "quantity": pos["quantity"],
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "pnl": pnl,
            "reason": reason,
            "opened_at": pos["opened_at"],
            "closed_at": _now(),
        }
        self.store.record_trade(trade)
        self.store.delete_position(ticker)
        self.store.log_event(
            "fill",
            f"CLOSE {pos['side'].upper()} {pos['name']} ({ticker}) "
            f"@ {price:.2f} ({reason}): P&L ${pnl:+.2f}",
        )
        log.info("Closed %s %s @ %.2f (%s): pnl %+.2f", pos["side"], ticker,
                 price, reason, pnl)
        return trade

    def check_stop(self, ticker: str, bar_high: float, bar_low: float) -> dict | None:
        """Enforce the hard stop against the latest bar's range. Fills at the
        stop price (assumes no gap through it). Returns the closed trade."""
        pos = self.store.get_position(ticker)
        if not pos:
            return None
        stop = pos["stop_price"]
        hit = (pos["side"] == "long" and bar_low <= stop) or \
              (pos["side"] == "short" and bar_high >= stop)
        if not hit:
            return None
        return self.close_position(ticker, stop, reason="stop")
