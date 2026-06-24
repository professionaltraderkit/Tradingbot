#!/usr/bin/env python3
"""Cross-sectional momentum rotation  - backtest, walk-forward, today's book.

A portfolio-level bot (a different shape from the per-symbol intraday bots):
each month it ranks a fixed ETF universe by 12-1 momentum, holds the top N
equal-weight, and  - with the regime filter on  - steps aside to cash when the
benchmark is below its 200-day average. Daily split/dividend-adjusted history
comes from Yahoo via bot.data.MarketData; rebalance turnover is charged at the
same per-side-bps convention as backtest.py, so every result is net of costs.

The headline output compares the strategy WITH and WITHOUT the regime filter,
both on the full history and out-of-sample via walk-forward, against a
buy-&-hold benchmark  - so you can see exactly what the filter buys you.

    python momentum_backtest.py                      # full + walk-forward, regime on vs off
    python momentum_backtest.py --no-regime-filter   # force the regime filter off
    python momentum_backtest.py --today              # just print this month's target book
    python momentum_backtest.py --cost-bps 5 --top-n 3 --lookback 126
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from bot.data import MarketData
from bot.momentum import CrossSectionalMomentum

log = logging.getLogger(__name__)

DEFAULT_COST_BPS = 3.0


# -- data -------------------------------------------------------------------
def build_panel(feed: MarketData, universe: list[str], benchmark: str,
                years: float):
    """Fetch daily closes and align everything to the benchmark's calendar.

    Returns (universe_panel, bench_close). The universe panel is forward-
    filled across odd holiday mismatches but never before a ticker's
    inception, so a not-yet-existing ticker is simply NaN (and excluded
    from ranking) until it lists."""
    closes = {}
    for ticker in [benchmark, *universe]:
        df = feed.fetch_daily(ticker, years)
        if df.empty:
            print(f"!! no daily data for {ticker}, skipping it")
            continue
        closes[ticker] = df["Close"].rename(ticker)
    if benchmark not in closes:
        raise SystemExit(f"benchmark {benchmark} has no data  - cannot continue")

    raw = pd.concat(closes.values(), axis=1).sort_index()
    bench_close = raw[benchmark].dropna()
    have = [t for t in universe if t in closes]
    panel = raw[have].reindex(bench_close.index).ffill()
    return panel, bench_close


# -- simulation -------------------------------------------------------------
def _rebalance_dates(panel: pd.DataFrame, strat, start, end):
    """Last trading day of each month with enough history, within [start, end]."""
    month_ends = list(panel.index.to_series()
                      .groupby(panel.index.to_period("M")).last())
    out = []
    for d in month_ends:
        if panel.index.get_loc(d) < strat.min_history:
            continue
        if start is not None and d < start:
            continue
        if end is not None and d > end:
            continue
        out.append(d)
    return out


def run_backtest(panel, bench_close, bench_sma, strat, cost_bps,
                 starting_equity, start=None, end=None):
    """Monthly-rebalanced portfolio sim. Returns a dict with the daily equity
    curve, per-holding trade returns, the latest target weights, and turnover."""
    rebal = _rebalance_dates(panel, strat, start, end)
    if len(rebal) < 2:
        return None
    daily = panel.pct_change()
    equity = starting_equity
    dates, equities = [], []
    weights_prev: dict = {}
    turnovers, trades = [], []
    open_trades: dict = {}

    for k, d in enumerate(rebal):
        w = strat.target_weights(panel, bench_close, bench_sma, d)
        names = set(w) | set(weights_prev)
        turnover = sum(abs(w.get(t, 0.0) - weights_prev.get(t, 0.0)) for t in names)
        turnovers.append(turnover)
        equity *= (1 - turnover * cost_bps / 1e4)

        price_d = panel.loc[d]
        for t in list(open_trades):              # close holdings that left the book
            if t not in w:
                trades.append(price_d[t] / open_trades.pop(t) - 1.0)
        for t in w:                              # open new holdings
            open_trades.setdefault(t, price_d[t])

        if k == 0:
            dates.append(d); equities.append(equity)
        nxt = rebal[k + 1] if k + 1 < len(rebal) else panel.index[-1]
        if end is not None:
            nxt = min(nxt, end)
        seg = daily.loc[(daily.index > d) & (daily.index <= nxt)]
        if w and len(seg):
            wt = np.array([w[t] for t in w])
            seg_ret = np.nan_to_num(seg[list(w)].values) @ wt
        else:
            seg_ret = np.zeros(len(seg))
        for date, r in zip(seg.index, seg_ret):
            equity *= (1 + r)
            dates.append(date); equities.append(equity)
        weights_prev = w

    final_price = panel.loc[rebal[-1]]
    for t, entry in open_trades.items():         # mark-to-rebalance open holdings
        trades.append(final_price[t] / entry - 1.0)

    curve = pd.Series(equities, index=pd.DatetimeIndex(dates))
    curve = curve[~curve.index.duplicated(keep="last")]
    return {
        "curve": curve,
        "returns": curve.pct_change().dropna(),
        "trades": trades,
        "weights_today": weights_prev,
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "n_rebalances": len(rebal),
    }


# -- metrics ----------------------------------------------------------------
def metrics(curve: pd.Series) -> dict:
    curve = curve.dropna()
    if len(curve) < 3:
        return dict(total=0, cagr=0, vol=0, sharpe=0, sortino=0, max_dd=0,
                    pos_months=0, calmar=0, years=0)
    rets = curve.pct_change().dropna()
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    growth = curve.iloc[-1] / curve.iloc[0]
    cagr = growth ** (1 / years) - 1 if years > 0 else 0.0
    vol = rets.std() * np.sqrt(252)
    downside = rets[rets < 0].std() * np.sqrt(252)
    dd = (curve / curve.cummax() - 1).min()
    monthly = curve.groupby(curve.index.to_period("M")).last().pct_change().dropna()
    return dict(
        total=(growth - 1) * 100,
        cagr=cagr * 100,
        vol=vol * 100,
        sharpe=(rets.mean() * 252) / vol if vol > 0 else 0.0,
        sortino=(rets.mean() * 252) / downside if downside > 0 else 0.0,
        max_dd=dd * 100,
        pos_months=(monthly > 0).mean() * 100 if len(monthly) else 0.0,
        calmar=cagr / abs(dd) if dd < 0 else float("inf"),
        years=years,
    )


def trade_stats(trades: list[float]) -> dict:
    if not trades:
        return dict(n=0, win_rate=0, avg_win=0, avg_loss=0, expectancy=0)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    return dict(
        n=len(trades),
        win_rate=len(wins) / len(trades) * 100,
        avg_win=np.mean(wins) * 100 if wins else 0.0,
        avg_loss=np.mean(losses) * 100 if losses else 0.0,
        expectancy=np.mean(trades) * 100,
    )


# -- walk-forward -----------------------------------------------------------
def _wf_grid():
    return [dict(lookback_days=lb, skip_days=21, top_n=tn, require_positive=True)
            for lb in (126, 252) for tn in (2, 3, 4, 5)]


def walk_forward(panel, bench_close, bench_sma, cost_bps, starting_equity,
                 train_years, test_years, regime_filter):
    """Rolling out-of-sample test. On each in-sample window pick the params
    with the best Sharpe, then trade the next window with them; stitch the
    out-of-sample stretches into one equity curve."""
    grid = _wf_grid()
    idx = panel.index
    oos_returns, chosen = [], []
    t_start = idx[0] + pd.DateOffset(years=train_years)

    while t_start < idx[-1]:
        is_start, is_end = t_start - pd.DateOffset(years=train_years), t_start
        oos_end = min(t_start + pd.DateOffset(years=test_years), idx[-1])

        best, best_sharpe = None, -1e18
        for combo in grid:
            strat = CrossSectionalMomentum(regime_filter=regime_filter, **combo)
            res = run_backtest(panel, bench_close, bench_sma, strat, cost_bps,
                               starting_equity, start=is_start, end=is_end)
            if not res or len(res["returns"]) < 60:
                continue
            s = metrics(res["curve"])["sharpe"]
            if s > best_sharpe:
                best_sharpe, best = s, combo
        if best is not None:
            strat = CrossSectionalMomentum(regime_filter=regime_filter, **best)
            oos = run_backtest(panel, bench_close, bench_sma, strat, cost_bps,
                               starting_equity, start=t_start, end=oos_end)
            if oos and len(oos["returns"]):
                oos_returns.append(oos["returns"])
                chosen.append((t_start.year, oos_end.year, best))
        if oos_end >= idx[-1]:
            break
        t_start = oos_end

    if not oos_returns:
        return None, []
    allr = pd.concat(oos_returns)
    allr = allr[~allr.index.duplicated(keep="first")].sort_index()
    curve = (1 + allr).cumprod() * starting_equity
    return curve, chosen


# -- reporting --------------------------------------------------------------
def _row(label, m, t=None, turn=None):
    base = (f"{label:<26} {m['cagr']:>7.1f} {m['total']:>9.1f} {m['vol']:>6.1f} "
            f"{m['sharpe']:>6.2f} {m['max_dd']:>7.1f} {m['pos_months']:>7.0f} "
            f"{m['calmar']:>6.2f}")
    if t is not None:
        base += f" {t['win_rate']:>6.0f} {t['expectancy']:>8.2f}"
    if turn is not None:
        base += f" {turn*100:>7.0f}"
    return base


def _header(extra: bool):
    cols = (f"{'Strategy':<26} {'CAGR%':>7} {'Total%':>9} {'Vol%':>6} "
            f"{'Sharpe':>6} {'MaxDD%':>7} {'PosMo%':>7} {'Calmar':>6}")
    if extra:
        cols += f" {'TrWin%':>6} {'Exp/tr%':>8} {'Turn/mo%':>7}"
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-sectional momentum backtest")
    parser.add_argument("--config", default="config-momentum.yaml")
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--skip", type=int, default=None)
    parser.add_argument("--no-regime-filter", action="store_true",
                        help="run only the no-filter variant")
    parser.add_argument("--today", action="store_true",
                        help="print this month's target book and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    load_dotenv(Path(__file__).resolve().parent / ".env")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    m = config["momentum"]
    bt = config.get("backtest", {})
    equity = config["account"]["starting_equity"]
    cost_bps = (args.cost_bps if args.cost_bps is not None
                else config.get("costs", {}).get("per_side_bps", DEFAULT_COST_BPS))
    params = dict(
        lookback_days=args.lookback or m["lookback_days"],
        skip_days=args.skip or m["skip_days"],
        top_n=args.top_n or m["top_n"],
        require_positive=m.get("require_positive", True),
        regime_sma=m.get("regime_sma", 200),
    )

    print(f"Fetching {bt.get('years', 20)}y of daily data for "
          f"{len(m['universe'])} ETFs + {m['benchmark']} ...")
    feed = MarketData()
    panel, bench_close = build_panel(feed, m["universe"], m["benchmark"],
                                     bt.get("years", 20))
    bench_sma = bench_close.rolling(params["regime_sma"]).mean()
    inception = panel.dropna().index[0]
    panel = panel.loc[inception:]
    start = panel.index[params["lookback_days"] + 1]
    end = panel.index[-1]

    # -- today's book -------------------------------------------------------
    strat_on = CrossSectionalMomentum(regime_filter=True, **params)
    last_reb = _rebalance_dates(panel, strat_on, None, None)[-1]
    book = strat_on.target_weights(panel, bench_close, bench_sma, last_reb)
    regime_ok = strat_on.in_uptrend(bench_close, bench_sma, last_reb)
    mom_now = strat_on.momentum(panel, last_reb).dropna().sort_values(ascending=False)

    print(f"\nMost recent rebalance: {last_reb.date()}   "
          f"SPY regime: {'RISK-ON (>=200d)' if regime_ok else 'RISK-OFF (<200d) -> cash'}")
    print("12-1 momentum ranking (top 8):")
    for tk, v in mom_now.head(8).items():
        held = " <- HELD" if tk in book else ""
        print(f"   {tk:<5} {v*100:>7.1f}%{held}")
    if book:
        print(f"Target book ({len(book)} names, {1/params['top_n']*100:.0f}% each"
              + (f", {(1-sum(book.values()))*100:.0f}% cash" if sum(book.values()) < 0.999 else "")
              + "): " + ", ".join(book))
    else:
        print("Target book: 100% CASH")
    if args.today:
        return

    # -- full-period + walk-forward, regime on vs off -----------------------
    variants = [("regime filter OFF", False)] if args.no_regime_filter \
        else [("regime filter ON", True), ("regime filter OFF", False)]

    span_years = (end - start).days / 365.25
    print(f"\nUniverse inception {inception.date()} | tested {start.date()} -> "
          f"{end.date()} ({span_years:.1f}y) | net of {cost_bps:.1f} bp/side | "
          f"params: L{params['lookback_days']}/skip{params['skip_days']}/top{params['top_n']}")

    print("\n=== FULL-PERIOD BACKTEST (in-sample  - the optimistic view) ===")
    print(_header(extra=True))
    print("-" * 104)
    bench_curve = bench_close.loc[start:end] / bench_close.loc[start] * equity
    print(_row(f"{m['benchmark']} buy & hold", metrics(bench_curve)))
    for label, regime in variants:
        strat = CrossSectionalMomentum(regime_filter=regime, **params)
        res = run_backtest(panel, bench_close, bench_sma, strat, cost_bps,
                           equity, start=start, end=end)
        print(_row(f"Momentum ({label})", metrics(res["curve"]),
                   trade_stats(res["trades"]), res["avg_turnover"]))

    print("\n=== WALK-FORWARD (out-of-sample  - the honest view) ===")
    print(f"  in-sample {bt.get('train_years',6)}y -> trade next "
          f"{bt.get('test_years',2)}y, params re-picked each fold by Sharpe")
    print(_header(extra=False))
    print("-" * 76)
    for label, regime in variants:
        curve, chosen = walk_forward(panel, bench_close, bench_sma, cost_bps,
                                     equity, bt.get("train_years", 6),
                                     bt.get("test_years", 2), regime)
        if curve is None:
            print(f"Momentum ({label}): insufficient history for walk-forward")
            continue
        print(_row(f"Momentum ({label})", metrics(curve)))
        if regime and chosen:
            picks = "; ".join(f"{a}-{b}: top{c['top_n']}/L{c['lookback_days']}"
                              for a, b, c in chosen)
            print(f"    folds -> {picks}")

    print("\nMetrics are net of costs. Walk-forward = stitched out-of-sample only "
          "(never optimized on the data it's scored on).")
    print("PosMo% = share of positive months (the win-rate analog for a monthly "
          "rotation). TrWin%/Exp = per-holding round-trip win rate & avg % return.")


if __name__ == "__main__":
    main()
