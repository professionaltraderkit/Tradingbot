"""Main engine: schedules each instrument on its own timeframe, runs the
signal -> risk -> fill pipeline, sweeps stops between candles, and
triggers the two daily reports.

Execution backend is chosen by config `broker.mode`:
  - "alpaca_paper": orders go to your Alpaca paper account
  - "simulated":    internal paper broker (also the automatic fallback
                    when Alpaca keys are missing)
"""

import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .data import MarketData
from .indicators import atr
from .notify import Notifier
from .portfolio import LivePortfolio, PaperPortfolio
from .reporter import evening_report, morning_briefing
from .risk import RiskManager
from .state import StateStore
from .strategies import Signal, build_strategy

log = logging.getLogger(__name__)

_TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}


class Engine:
    def __init__(self, config: dict, notifier: Notifier | None = None,
                 feed: MarketData | None = None, portfolio=None):
        self.config = config
        self.bot_name = config.get("name", "Trading bot")
        self.tz = ZoneInfo(config.get("timezone", "America/New_York"))
        self.instruments = config["instruments"]
        for inst in self.instruments:
            inst["_strategy"] = build_strategy(inst["strategy"], inst.get("params", {}))

        risk_cfg = config["risk"]
        corr = risk_cfg["correlation_filter"]
        self.atr_period = risk_cfg["atr_period"]
        self.max_notional_pct = risk_cfg.get("max_position_notional_pct", 100)
        self.risk = RiskManager(
            risk_per_trade_pct=risk_cfg["risk_per_trade_pct"],
            atr_stop_multiple=risk_cfg["atr_stop_multiple"],
            correlation_group=corr["group"],
            max_same_direction=corr["max_same_direction"],
        )

        self.starting_equity = config["account"]["starting_equity"]
        self.store = StateStore(config["storage"]["db_path"], self.starting_equity)
        self.feed = feed or MarketData()
        self.portfolio = portfolio or self._build_portfolio()
        self.notifier = notifier or Notifier()
        self._last_report: dict[str, str] = {}  # kind -> date string sent

    def _build_portfolio(self):
        mode = self.config.get("broker", {}).get("mode", "simulated")
        if mode == "alpaca_paper":
            try:
                from .broker import AlpacaBroker
                broker = AlpacaBroker(paper=True)
                equity = broker.account_equity()
                log.info("Connected to Alpaca paper account, equity $%s",
                         f"{equity:,.2f}")
                return LivePortfolio(self.store, broker)
            except Exception as exc:
                log.warning("Alpaca unavailable (%s) — falling back to the "
                            "simulated broker", exc)
        elif mode != "simulated":
            raise ValueError(f"Unknown broker.mode: {mode}")
        return PaperPortfolio(self.store)

    # -- single instrument cycle ----------------------------------------------
    def evaluate_instrument(self, inst: dict) -> pd.DataFrame:
        """Fetch data, enforce the stop, then act on the strategy signal.
        Returns the fetched candles (for reuse in reports)."""
        ticker = inst["ticker"]
        df = self.feed.fetch_candles(ticker, inst["timeframe"])
        if df.empty:
            log.warning("No data for %s, skipping cycle", ticker)
            return df

        last = df.iloc[-1]
        # Hard stop first — no exceptions, checked before any new decision.
        stopped = self.portfolio.check_stop(ticker, float(last["High"]), float(last["Low"]))
        if stopped:
            return df

        pos = self.store.get_position(ticker)
        side = pos["side"] if pos else None
        signal = inst["_strategy"].evaluate(df, side)

        if signal == Signal.EXIT and pos:
            self.portfolio.close_position(ticker, float(last["Close"]), reason="signal")
        elif signal in (Signal.LONG, Signal.SHORT) and not pos:
            self._try_open(inst, signal.value, df)
        return df

    def _try_open(self, inst: dict, side: str, df: pd.DataFrame) -> None:
        ticker, name = inst["ticker"], inst["name"]
        if side == "short" and not inst.get("allow_short", True):
            msg = f"short {name} ({ticker}) skipped — instrument is long-only"
            self.store.log_event("blocked", msg)
            log.info(msg)
            return
        if self.risk.correlation_blocked(ticker, side, self.portfolio.position_sides()):
            msg = (f"{side} {name} ({ticker}) skipped — correlation cap on "
                   f"risk-on exposure reached")
            self.store.log_event("blocked", msg)
            log.info("Blocked by correlation filter: %s %s", side, ticker)
            return

        price = float(df["Close"].iloc[-1])
        atr_value = float(atr(df, self.atr_period).iloc[-1])
        equity = self.portfolio.equity({ticker: price})
        plan = self.risk.plan_trade(side, equity, price, atr_value,
                                    max_notional=equity * self.max_notional_pct / 100,
                                    stop_multiple=inst.get("atr_stop_multiple"))
        if plan is None:
            log.warning("Could not size %s trade on %s (atr=%s)", side, ticker, atr_value)
            return
        try:
            self.portfolio.open_position(ticker, name, plan)
        except Exception:
            log.exception("Order failed for %s %s", side, ticker)

    # -- stop sweep between candles ----------------------------------------------
    def sweep_stops(self) -> None:
        """Check every open position's stop against the latest trade price.
        Runs every minute so a 4h instrument doesn't wait hours for its stop."""
        for pos in self.portfolio.open_positions():
            price = self.feed.latest_price(pos["ticker"])
            if price is None:
                continue
            try:
                self.portfolio.check_stop(pos["ticker"], price, price)
            except Exception:
                log.exception("Stop sweep failed for %s", pos["ticker"])

    # -- scheduling -------------------------------------------------------------
    def _due_instruments(self, now: datetime) -> list[dict]:
        minutes = now.hour * 60 + now.minute
        return [inst for inst in self.instruments
                if minutes % _TIMEFRAME_MINUTES[inst["timeframe"]] == 0]

    def _fetch_all(self) -> dict[str, pd.DataFrame]:
        return {inst["ticker"]: self.feed.fetch_candles(inst["ticker"], inst["timeframe"])
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
                                self.portfolio, self.atr_period,
                                bot_name=self.bot_name)

    def _build_evening(self) -> str:
        return evening_report(self.store, self.portfolio, self._fetch_all(),
                              self.starting_equity, bot_name=self.bot_name)

    # -- main loop ---------------------------------------------------------------
    def run_once(self) -> None:
        """One evaluation pass over every instrument (used by --once and tests)."""
        for inst in self.instruments:
            try:
                self.evaluate_instrument(inst)
            except Exception:
                log.exception("Cycle failed for %s", inst["ticker"])

    def run_forever(self) -> None:
        backend = type(self.portfolio).__name__
        log.info("Engine started (timezone %s, broker %s). Telegram %s.",
                 self.tz, backend,
                 "enabled" if self.notifier.telegram_enabled else
                 "not configured — reports go to console")
        last_minute = None
        while True:
            now = datetime.now(self.tz)
            minute_key = now.strftime("%Y-%m-%d %H:%M")
            if minute_key != last_minute:
                last_minute = minute_key
                due = self._due_instruments(now)
                for inst in due:
                    try:
                        self.evaluate_instrument(inst)
                    except Exception:
                        log.exception("Cycle failed for %s", inst["ticker"])
                # Stops are also enforced inside evaluate_instrument; the sweep
                # covers instruments whose candle isn't due this minute.
                due_tickers = {inst["ticker"] for inst in due}
                if any(p["ticker"] not in due_tickers
                       for p in self.portfolio.open_positions()):
                    self.sweep_stops()
                self._maybe_send_reports(now)
            time.sleep(5)
