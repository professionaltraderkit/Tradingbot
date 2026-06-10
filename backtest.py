#!/usr/bin/env python3
"""Backtest the configured strategies on recent Yahoo Finance history.

    python backtest.py [--days N] [--config config.yaml]

Each instrument is simulated independently with its own paper account so
strategies can be judged on their own merits (the live correlation filter
is a cross-instrument, live-only constraint). Intraday history is limited
by Yahoo: ~60 days for 15m candles, ~730 days for 1h/4h.
"""

import argparse
import logging

import pandas as pd
import yaml

from bot import data
from bot.indicators import atr
from bot.portfolio import PaperPortfolio
from bot.risk import RiskManager
from bot.state import StateStore
from bot.strategies import Signal, build_strategy

log = logging.getLogger(__name__)


def backtest_instrument(inst: dict, df: pd.DataFrame, risk: RiskManager,
                        atr_period: int, starting_equity: float) -> dict:
    strategy = build_strategy(inst["strategy"], inst.get("params", {}))
    store = StateStore(":memory:", starting_equity)
    portfolio = PaperPortfolio(store)
    ticker, name = inst["ticker"], inst["name"]

    equity_curve = []
    for i in range(strategy.min_bars, len(df)):
        window = df.iloc[: i + 1]
        bar = window.iloc[-1]
        price = float(bar["Close"])

        portfolio.check_stop(ticker, float(bar["High"]), float(bar["Low"]))

        pos = store.get_position(ticker)
        signal = strategy.evaluate(window, pos["side"] if pos else None)
        if signal == Signal.EXIT and pos:
            portfolio.close_position(ticker, price, reason="signal")
        elif signal in (Signal.LONG, Signal.SHORT) and not pos:
            atr_value = float(atr(window, atr_period).iloc[-1])
            equity = portfolio.equity({ticker: price})
            plan = risk.plan_trade(signal.value, equity, price, atr_value)
            if plan:
                portfolio.open_position(ticker, name, plan)

        equity_curve.append(portfolio.equity({ticker: price}))

    last_price = float(df["Close"].iloc[-1])
    portfolio.close_position(ticker, last_price, reason="end of backtest")

    trades = store.all_trades()
    curve = pd.Series(equity_curve)
    final = portfolio.equity({})
    max_dd = ((curve - curve.cummax()) / curve.cummax()).min() * 100 if len(curve) else 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0)
    store.close()
    return {
        "name": name,
        "bars": len(df),
        "trades": len(trades),
        "win_rate": wins / len(trades) * 100 if trades else 0.0,
        "pnl": final - starting_equity,
        "return_pct": (final / starting_equity - 1) * 100,
        "max_drawdown_pct": max_dd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the bot's strategies")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--days", type=int, default=None,
                        help="restrict the backtest to the last N days")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    risk_cfg = config["risk"]
    corr = risk_cfg["correlation_filter"]
    risk = RiskManager(
        risk_per_trade_pct=risk_cfg["risk_per_trade_pct"],
        atr_stop_multiple=risk_cfg["atr_stop_multiple"],
        correlation_group=corr["group"],
        max_same_direction=corr["max_same_direction"],
    )
    starting_equity = config["account"]["starting_equity"]

    results = []
    for inst in config["instruments"]:
        df = data.fetch_candles(inst["ticker"], inst["timeframe"])
        if df.empty:
            print(f"!! no data for {inst['ticker']}, skipping")
            continue
        if args.days:
            cutoff = df.index[-1] - pd.Timedelta(days=args.days)
            df = df[df.index >= cutoff]
        results.append(backtest_instrument(inst, df, risk,
                                           risk_cfg["atr_period"], starting_equity))

    if not results:
        print("No results.")
        return

    print(f"\n{'Instrument':<14} {'Bars':>6} {'Trades':>7} {'Win%':>6} "
          f"{'P&L $':>12} {'Return%':>8} {'MaxDD%':>7}")
    print("-" * 66)
    for r in results:
        print(f"{r['name']:<14} {r['bars']:>6} {r['trades']:>7} "
              f"{r['win_rate']:>5.0f}% {r['pnl']:>12,.2f} "
              f"{r['return_pct']:>7.2f}% {r['max_drawdown_pct']:>6.2f}%")
    total_pnl = sum(r["pnl"] for r in results)
    total_trades = sum(r["trades"] for r in results)
    print("-" * 66)
    print(f"{'TOTAL':<14} {'':>6} {total_trades:>7} {'':>6} {total_pnl:>12,.2f}")
    print("\n(each instrument simulated on its own "
          f"${starting_equity:,.0f} paper account)")


if __name__ == "__main__":
    main()
