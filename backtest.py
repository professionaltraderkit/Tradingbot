#!/usr/bin/env python3
"""Backtest the configured strategies on recent market history.

    python backtest.py [--config config.yaml] [--days N] [--cost-bps B]

Each instrument is simulated independently with its own paper account so
strategies can be judged on their own merits (the live correlation filter
is a cross-instrument, live-only constraint). Data comes from the same
source as the live bot (Alpaca when keys are set, else Yahoo Finance).

Transaction costs (commission + slippage) are charged on every fill —
without them an intraday strategy that trades often looks far better than
it is. The default is a liquid-ETF estimate; override with --cost-bps.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from bot.data import MarketData
from bot.indicators import atr
from bot.portfolio import PaperPortfolio
from bot.risk import RiskManager
from bot.state import StateStore
from bot.strategies import Signal, build_strategy

log = logging.getLogger(__name__)

# Cost per side as basis points of notional (1 bp = 0.01%). ~2 bp covers the
# typical SPY/QQQ spread plus a little slippage; futures/crypto differ.
DEFAULT_COST_BPS = 2.0


def _trade_cost(trade: dict, cost_bps: float) -> float:
    """Round-trip cost (entry + exit) for one trade."""
    notional = (trade["entry_price"] + trade["exit_price"]) * trade["quantity"]
    return notional * cost_bps / 10_000.0


# Bars of history the strategy sees each step. A rolling cap keeps the
# simulation O(n) instead of O(n^2); 400 bars comfortably exceeds every
# indicator's warm-up (longest is a day-anchored VWAP, <~200 5m bars) so
# results are unchanged within rounding.
WINDOW_BARS = 400


def run_simulation(inst: dict, df: pd.DataFrame, risk: RiskManager,
                   atr_period: int, starting_equity: float,
                   max_notional_pct: float, cost_bps: float,
                   window_bars: int = WINDOW_BARS):
    """Replay `df` bar by bar; return (trades, equity_curve). Costs are
    deducted from cash as trades close so the equity curve is net."""
    strategy = build_strategy(inst["strategy"], inst.get("params", {}))
    store = StateStore(":memory:", starting_equity)
    portfolio = PaperPortfolio(store)
    ticker, name = inst["ticker"], inst["name"]

    def do_close(price: float, reason: str) -> None:
        trade = portfolio.close_position(ticker, price, reason=reason)
        if trade:  # charge the round-trip cost against cash
            store.set_cash(store.get_cash() - _trade_cost(trade, cost_bps))

    equity_curve = []
    for i in range(strategy.min_bars, len(df)):
        window = df.iloc[max(0, i + 1 - window_bars): i + 1]
        bar = window.iloc[-1]
        price = float(bar["Close"])

        # Hard stop first (charge cost if it fires).
        if portfolio.check_stop(ticker, float(bar["High"]), float(bar["Low"])):
            last = store.all_trades()[-1]
            store.set_cash(store.get_cash() - _trade_cost(last, cost_bps))

        pos = store.get_position(ticker)
        if pos:
            action = strategy.manage(window, pos)
            if action:
                new_stop = action.get("stop")
                if new_stop is not None:
                    tightens = (new_stop > pos["stop_price"] if pos["side"] == "long"
                                else new_stop < pos["stop_price"])
                    if tightens:
                        pos["stop_price"] = float(new_stop)
                        store.save_position(pos)
                if action.get("exit"):
                    do_close(float(action.get("price", price)), action["exit"])
                pos = store.get_position(ticker)

        signal = strategy.evaluate(window, pos["side"] if pos else None)
        if signal == Signal.EXIT and pos:
            do_close(price, "signal")
        elif signal in (Signal.LONG, Signal.SHORT) and not pos:
            atr_value = float(atr(window, atr_period).iloc[-1])
            equity = portfolio.equity({ticker: price})
            plan = risk.plan_trade(signal.value, equity, price, atr_value,
                                   max_notional=equity * max_notional_pct / 100,
                                   stop_multiple=inst.get("atr_stop_multiple"),
                                   stop_price=strategy.initial_stop(window, signal.value))
            if plan:
                portfolio.open_position(ticker, name, plan)

        equity_curve.append(portfolio.equity({ticker: price}))

    do_close(float(df["Close"].iloc[-1]), "end of backtest")
    trades = store.all_trades()
    store.close()
    return trades, equity_curve


def compute_stats(name: str, n_bars: int, trades: list[dict],
                  equity_curve: list[float], starting_equity: float,
                  cost_bps: float) -> dict:
    """Performance metrics on net (after-cost) P&L."""
    net = [t["pnl"] - _trade_cost(t, cost_bps) for t in trades]
    wins = [p for p in net if p > 0]
    losses = [p for p in net if p < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    total = sum(net)
    curve = pd.Series(equity_curve)
    max_dd = ((curve - curve.cummax()) / curve.cummax()).min() * 100 if len(curve) else 0.0
    return {
        "name": name,
        "bars": n_bars,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "pnl": total,
        "return_pct": total / starting_equity * 100,
        "max_drawdown_pct": max_dd,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "avg_win": gross_win / len(wins) if wins else 0.0,
        "avg_loss": -gross_loss / len(losses) if losses else 0.0,
        "expectancy": total / len(trades) if trades else 0.0,
    }


def backtest_instrument(inst: dict, df: pd.DataFrame, risk: RiskManager,
                        atr_period: int, starting_equity: float,
                        max_notional_pct: float = 100,
                        cost_bps: float = DEFAULT_COST_BPS) -> dict:
    trades, curve = run_simulation(inst, df, risk, atr_period, starting_equity,
                                   max_notional_pct, cost_bps)
    return compute_stats(inst["name"], len(df), trades, curve,
                         starting_equity, cost_bps)


def build_risk(config: dict) -> RiskManager:
    risk_cfg = config["risk"]
    corr = risk_cfg["correlation_filter"]
    return RiskManager(
        risk_per_trade_pct=risk_cfg["risk_per_trade_pct"],
        atr_stop_multiple=risk_cfg["atr_stop_multiple"],
        correlation_group=corr["group"],
        max_same_direction=corr["max_same_direction"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the bot's strategies")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--days", type=int, default=None,
                        help="restrict the backtest to the last N days")
    parser.add_argument("--cost-bps", type=float, default=None,
                        help="transaction cost per side in basis points "
                             f"(default {DEFAULT_COST_BPS}, or config costs.per_side_bps)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    # Use the same data source as the live bot (Alpaca when keys are set).
    load_dotenv(Path(__file__).resolve().parent / ".env")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    risk = build_risk(config)
    risk_cfg = config["risk"]
    starting_equity = config["account"]["starting_equity"]
    cost_bps = (args.cost_bps if args.cost_bps is not None
                else config.get("costs", {}).get("per_side_bps", DEFAULT_COST_BPS))

    feed = MarketData()
    results = []
    for inst in config["instruments"]:
        df = feed.fetch_candles(inst["ticker"], inst["timeframe"])
        if df.empty:
            print(f"!! no data for {inst['ticker']}, skipping")
            continue
        if args.days:
            cutoff = df.index[-1] - pd.Timedelta(days=args.days)
            df = df[df.index >= cutoff]
        results.append(backtest_instrument(
            inst, df, risk, risk_cfg["atr_period"], starting_equity,
            max_notional_pct=risk_cfg.get("max_position_notional_pct", 100),
            cost_bps=cost_bps))

    if not results:
        print("No results.")
        return

    print(f"\nNet of {cost_bps:.1f} bp/side costs. Each instrument on its own "
          f"${starting_equity:,.0f} paper account.\n")
    print(f"{'Instrument':<14} {'Trades':>7} {'Win%':>6} {'PF':>5} "
          f"{'Expect$':>9} {'P&L $':>11} {'Return%':>8} {'MaxDD%':>7}")
    print("-" * 72)
    for r in results:
        pf = "inf" if r["profit_factor"] == float("inf") else f"{r['profit_factor']:.2f}"
        print(f"{r['name']:<14} {r['trades']:>7} {r['win_rate']:>5.0f}% {pf:>5} "
              f"{r['expectancy']:>9.2f} {r['pnl']:>11,.2f} "
              f"{r['return_pct']:>7.2f}% {r['max_drawdown_pct']:>6.2f}%")
    total_pnl = sum(r["pnl"] for r in results)
    total_trades = sum(r["trades"] for r in results)
    print("-" * 72)
    print(f"{'TOTAL':<14} {total_trades:>7} {'':>6} {'':>5} {'':>9} "
          f"{total_pnl:>11,.2f}")
    print("\nPF = profit factor (gross wins / gross losses; >1 is profitable). "
          "Expect$ = average $ per trade.")


if __name__ == "__main__":
    main()
