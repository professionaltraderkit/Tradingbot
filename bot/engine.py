"""Main engine: schedules each instrument on its own timeframe, runs the
signal -> risk -> paper-fill pipeline, and triggers the two daily reports.
"""

import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from . import data
from .indicators import atr
from .notify import Notifier
from .portfolio import PaperPortfolio
from .reporter import evening_report, morning_briefing
from .risk import RiskManager
from .state import StateStore
from .strategies import Signal, build_strategy

log = logging.getLogger(__name__)

_TIMEFRAME_MINUTES = {"15m": 15, "1h": 60, "4h": 240}


class Engine:
    def __init__(self, config: dict, notifier: Notifier | None = None):
        self.config = config
        self.tz = ZoneInfo(config.get("timezone", "America/New_York"))
        self.instruments = config["instruments"]
        for inst in self.instruments:
            inst["_strategy"] = build_strategy(inst["strategy"], inst.get("params", {}))

        risk_cfg = config["risk"]
        corr = risk_cfg["correlation_filter"]
        self.atr_period = risk_cfg["atr_period"]
        self.risk = RiskManager(
            risk_per_trade_pct=risk_cfg["risk_per_trade_pct"],
            atr_stop_multiple=risk_cfg["atr_stop_multiple"],
            correlation_group=corr["group"],
            max_same_direction=corr["max_same_direction"],
        )

        self.starting_equity = config["account"]["starting_equity"]
        self.store = StateStore(config["storage"]["db_path"], self.starting_equity)
        self.portfolio = PaperPortfolio(self.store)
        self.notifier = notifier or Notifier()
        self._last_report: dict[str, str] = {}  # kind -> date string sent

    # -- single instrument cycle ----------------------------------------------
    def evaluate_instrument(self, inst: dict) -> pd.DataFrame:
        """Fetch data, enforce the stop, then act on the strategy signal.
        Returns the fetched candles (for reuse in reports)."""
        ticker, name = inst["ticker"], inst["name"]
        df = data.fetch_candles(ticker, inst["timeframe"])
        if df.empty:
            log.warning("No data for %s, skipping cycle", ticker)
            return df

        last = df.iloc[-1]
        # Hard stop first — no exceptions, checked before any new decision.
        stopped = self.portfolio.check_stop(ticker, last["High"], last["Low"])
        if stopped:
            return df

        pos = self.store.get_position(ticker)
        side = pos["side"] if pos else None
        signal = inst["_strategy"].evaluate(df, side)

        if signal == Signal.EXIT and pos:
            self.portfolio.close_position(ticker, last["Close"], reason="signal")
        elif signal in (Signal.LONG, Signal.SHORT) and not pos:
            self._try_open(inst, signal.value, df)
        return df

    def _try_open(self, inst: dict, side: str, df: pd.DataFrame) -> None:
        ticker, name = inst["ticker"], inst["name"]
        if self.risk.correlation_blocked(ticker, side, self.portfolio.position_sides()):
            msg = (f"{side} {name} ({ticker}) skipped — correlation cap on "
                   f"risk-on exposure reached")
            self.store.log_event("blocked", msg)
            log.info("Blocked by correlation filter: %s %s", side, ticker)
            return

        price = float(df["Close"].iloc[-1])
        atr_value = float(atr(df, self.atr_period).iloc[-1])
        equity = self.portfolio.equity({ticker: price})
        plan = self.risk.plan_trade(side, equity, price, atr_value)
        if plan is None:
            log.warning("Could not size %s trade on %s (atr=%s)", side, ticker, atr_value)
            return
        self.portfolio.open_position(ticker, name, plan)

    # -- scheduling -------------------------------------------------------------
    def _due_instruments(self, now: datetime) -> list[dict]:
        minutes = now.hour * 60 + now.minute
        due = []
        for inst in self.instruments:
            if minutes % _TIMEFRAME_MINUTES[inst["timeframe"]] == 0:
                due.append(inst)
        return due

    def _fetch_all(self) -> dict[str, pd.DataFrame]:
        return {inst["ticker"]: data.fetch_candles(inst["ticker"], inst["timeframe"])
                for inst in self.instruments}

    def _maybe_send_reports(self, now: datetime) -> None:
        reports = self.config["reports"]
        today = now.strftime("%Y-%m-%d")
        hhmm = now.strftime("%H:%M")
        for kind, build in (("morning", self._build_morning),
                            ("evening", self._build_evening)):
            if hhmm == reports[kind] and self._last_report.get(kind) != today:
                self._last_report[kind] = today
                try:
                    self.notifier.send(build())
                except Exception:
                    log.exception("Failed to send %s report", kind)

    def _build_morning(self) -> str:
        return morning_briefing(self.instruments, self._fetch_all(),
                                self.portfolio, self.atr_period)

    def _build_evening(self) -> str:
        return evening_report(self.store, self.portfolio, self._fetch_all(),
                              self.starting_equity)

    # -- main loop ---------------------------------------------------------------
    def run_once(self) -> None:
        """One evaluation pass over every instrument (used by --once and tests)."""
        for inst in self.instruments:
            try:
                self.evaluate_instrument(inst)
            except Exception:
                log.exception("Cycle failed for %s", inst["ticker"])

    def run_forever(self) -> None:
        log.info("Engine started (timezone %s). Telegram %s.", self.tz,
                 "enabled" if self.notifier.telegram_enabled else
                 "not configured — reports go to console")
        last_minute = None
        while True:
            now = datetime.now(self.tz)
            minute_key = now.strftime("%Y-%m-%d %H:%M")
            if minute_key != last_minute:
                last_minute = minute_key
                for inst in self._due_instruments(now):
                    try:
                        self.evaluate_instrument(inst)
                    except Exception:
                        log.exception("Cycle failed for %s", inst["ticker"])
                self._maybe_send_reports(now)
            time.sleep(5)
