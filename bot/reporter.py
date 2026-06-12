"""Builds the two daily messages:

- 7am briefing: what's happening in the market (price, 24h change,
  volatility regime, strategy stance, open positions).
- 9pm report: how the bot did (closed trades, P&L, equity, win rate,
  blocked signals).
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from .indicators import atr
from .portfolio import PaperPortfolio, position_pnl
from .state import StateStore


def _pct(value: float) -> str:
    return f"{value:+.2f}%"


def _change_24h(df: pd.DataFrame) -> float | None:
    last_ts = df.index[-1]
    cutoff = last_ts - timedelta(hours=24)
    past = df[df.index <= cutoff]
    if past.empty:
        return None
    ref = past["Close"].iloc[-1]
    return (df["Close"].iloc[-1] / ref - 1) * 100


def _vol_regime(df: pd.DataFrame, atr_period: int) -> str:
    series = atr(df, atr_period).dropna()
    if len(series) < 30:
        return "n/a"
    current, baseline = series.iloc[-1], series.tail(200).mean()
    ratio = current / baseline if baseline else 1.0
    if ratio > 1.3:
        return f"elevated ({ratio:.1f}x normal)"
    if ratio < 0.7:
        return f"quiet ({ratio:.1f}x normal)"
    return "normal"


def morning_briefing(instruments: list[dict], market: dict[str, pd.DataFrame],
                     portfolio: PaperPortfolio, atr_period: int,
                     bot_name: str = "Trading bot") -> str:
    now = datetime.now()
    lines = [f"☀️ Morning briefing — {bot_name} — {now:%a %b %d, %Y}", ""]
    positions = {p["ticker"]: p for p in portfolio.open_positions()}

    for inst in instruments:
        ticker, name = inst["ticker"], inst["name"]
        df = market.get(ticker)
        if df is None or df.empty:
            lines.append(f"• {name}: no data available")
            continue
        price = df["Close"].iloc[-1]
        change = _change_24h(df)
        change_txt = _pct(change) if change is not None else "n/a"
        lines.append(f"• {name}: {price:,.2f} ({change_txt} 24h), "
                     f"volatility {_vol_regime(df, atr_period)}")
        stance = inst["_strategy"].stance(df)
        if stance:
            lines.append(f"  {stance}")
        pos = positions.get(ticker)
        if pos:
            upnl = position_pnl(pos, price)
            lines.append(f"  📌 open {pos['side']} from {pos['entry_price']:,.2f}, "
                         f"stop {pos['stop_price']:,.2f}, P&L ${upnl:+,.2f}")

    prices = {t: df["Close"].iloc[-1] for t, df in market.items()
              if df is not None and not df.empty}
    lines += ["", f"Equity: ${portfolio.equity(prices):,.2f} "
                  f"({len(positions)} open position(s))"]
    return "\n".join(lines)


def evening_report(store: StateStore, portfolio: PaperPortfolio,
                   market: dict[str, pd.DataFrame],
                   starting_equity: float,
                   bot_name: str = "Trading bot") -> str:
    now = datetime.now()
    day_start = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    trades = store.trades_since(day_start)
    blocked = store.events_since(day_start, kind="blocked")

    lines = [f"🌙 Daily report — {bot_name} — {now:%a %b %d, %Y}", ""]

    if trades:
        wins = [t for t in trades if t["pnl"] > 0]
        day_pnl = sum(t["pnl"] for t in trades)
        lines.append(f"Closed trades today: {len(trades)} "
                     f"({len(wins)} win / {len(trades) - len(wins)} loss), "
                     f"realized P&L ${day_pnl:+,.2f}")
        for t in trades:
            lines.append(f"  • {t['side']} {t['name']}: "
                         f"{t['entry_price']:,.2f} → {t['exit_price']:,.2f} "
                         f"= ${t['pnl']:+,.2f} ({t['reason']})")
    else:
        lines.append("No trades closed today.")

    positions = portfolio.open_positions()
    prices = {t: df["Close"].iloc[-1] for t, df in market.items()
              if df is not None and not df.empty}
    if positions:
        lines.append("")
        lines.append(f"Open positions ({len(positions)}):")
        for p in positions:
            price = prices.get(p["ticker"], p["entry_price"])
            lines.append(f"  • {p['side']} {p['name']} from "
                         f"{p['entry_price']:,.2f}, now {price:,.2f}, "
                         f"P&L ${position_pnl(p, price):+,.2f}")

    if blocked:
        lines.append("")
        lines.append("Signals skipped (risk rules):")
        for ev in blocked:
            lines.append(f"  • {ev['message']}")

    equity = portfolio.equity(prices)
    all_trades = store.all_trades()
    lines.append("")
    lines.append(f"Equity: ${equity:,.2f} "
                 f"({_pct((equity / starting_equity - 1) * 100)} all-time)")
    if all_trades:
        all_wins = sum(1 for t in all_trades if t["pnl"] > 0)
        lines.append(f"All-time: {len(all_trades)} trades, "
                     f"{all_wins / len(all_trades) * 100:.0f}% win rate")
    return "\n".join(lines)
