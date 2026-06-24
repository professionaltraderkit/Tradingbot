#!/usr/bin/env python3
"""
GGR-style pairs-trading backtest (Gatev, Goetzmann & Rouwenhorst 2006).

Faithful to the canonical study's structure, which our earlier single-pair
TradingView tests were NOT:
  - DISTANCE selection (not cointegration): normalize each stock to a $1
    total-return index over the formation window, pick the N pairs with the
    smallest sum of squared deviations (SSD) between the two normalized series.
  - Rolling 12-month FORMATION -> 6-month TRADING, out-of-sample by construction.
    Pairs are re-selected every cycle.
  - Entry when the spread diverges > 2 * (formation-period sigma); positions are
    opened the NEXT day (1-day delay, avoids look-ahead / bid-ask bounce).
  - Exit on mean reversion (spread crosses 0) or forced close at period end.
  - Market-neutral, $1 long / $1 short per pair, equal-weight portfolio of N pairs.
  - "Committed capital" return = payoff averaged over ALL N pairs (conservative);
    "fully invested" = averaged over only the pairs open that day.

Caveats: universe is a fixed list of CURRENTLY-listed liquid names -> survivorship
bias (GGR used the full CRSP cross-section). Close-to-close fills, no intraday.
Costs applied per leg per transaction (configurable). Non-overlapping cycles
(GGR staggered monthly; this is the standard simplification).

Usage:
    python ggr_pairs_backtest.py
    python ggr_pairs_backtest.py --npairs 20 --entry 2.0 --cost_bps 5
    python ggr_pairs_backtest.py --start 2004-01-01 --csv ggr_equity.csv
"""
import argparse
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Sector-rich liquid universe so close substitutes exist for distance pairing.
UNIVERSE = [
    # tech / semis
    "AAPL", "MSFT", "ORCL", "IBM", "CSCO", "INTC", "TXN", "QCOM", "ADI", "AMAT",
    # financials / cards
    "V", "MA", "JPM", "BAC", "C", "WFC", "GS", "MS", "AXP", "USB",
    # consumer staples / discretionary
    "KO", "PEP", "PG", "CL", "WMT", "TGT", "COST", "MCD", "SBUX", "HD", "LOW",
    # health care
    "JNJ", "PFE", "MRK", "ABT", "BMY", "UNH", "AMGN", "GILD",
    # energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # industrials
    "CAT", "DE", "UPS", "FDX", "HON", "MMM", "GE", "EMR",
    # telecom / utilities
    "T", "VZ", "DUK", "SO",
]

FORM_DAYS = 252      # ~12 months formation
TRADE_DAYS = 126     # ~6 months trading


def select_pairs(form_prices: pd.DataFrame, n_pairs: int):
    """Distance approach: normalize to $1 index, rank pairs by SSD, return
    list of (a, b, sigma) using the smallest-distance pairs."""
    norm = form_prices / form_prices.iloc[0]
    cols = norm.columns
    results = []
    for a, b in combinations(cols, 2):
        diff = norm[a] - norm[b]
        ssd = np.sum(diff.values ** 2)
        sigma = diff.std()
        results.append((ssd, a, b, sigma))
    results.sort(key=lambda r: r[0])
    return [(a, b, sig) for _, a, b, sig in results[:n_pairs]]


def trade_pair(a, b, form_px, trade_px, sigma, entry_z, cost_bps):
    """Simulate one pair over the trading window. Returns a daily P&L Series
    (in return units on $1/$1 notional) indexed by trade dates, NaN/0 when flat."""
    base = form_px.iloc[0]                       # formation-start base for both legs
    full_a = pd.concat([form_px[a], trade_px[a]])
    full_b = pd.concat([form_px[b], trade_px[b]])
    norm_a = full_a / base[a]
    norm_b = full_b / base[b]
    spread = (norm_a - norm_b).loc[trade_px.index]
    ret_a = trade_px[a].pct_change().fillna(0)
    ret_b = trade_px[b].pct_change().fillna(0)

    thr = entry_z * sigma
    pos = 0                                       # +1 long a/short b, -1 short a/long b
    pnl = pd.Series(0.0, index=trade_px.index)
    pending = 0                                    # signal -> enter next day
    n_trades = 0
    c = cost_bps / 10000.0

    dates = list(trade_px.index)
    for i, d in enumerate(dates):
        s = spread.loc[d]
        # accrue P&L for an open position on this day's returns
        if pos != 0:
            pnl.loc[d] += pos * (ret_a.loc[d] - ret_b.loc[d])

        # apply a pending entry decided yesterday (1-day delay)
        if pending != 0 and pos == 0:
            pos = pending
            pnl.loc[d] -= 2 * c                    # two legs open
            n_trades += 1
            pending = 0

        # exit on mean reversion (spread crosses 0 relative to entry side)
        if pos == 1 and s >= 0:                    # was cheap (long a), reverted up
            pnl.loc[d] -= 2 * c
            pos = 0
        elif pos == -1 and s <= 0:                 # was rich (short a), reverted down
            pnl.loc[d] -= 2 * c
            pos = 0

        # generate a new entry signal for tomorrow if flat
        if pos == 0 and pending == 0:
            if s > thr:
                pending = -1                       # a rich -> short a / long b
            elif s < -thr:
                pending = 1                        # a cheap -> long a / short b

    # force-close at period end
    if pos != 0:
        pnl.iloc[-1] -= 2 * c
    return pnl, n_trades


def run(prices: pd.DataFrame, n_pairs, entry_z, cost_bps):
    dates = prices.index
    daily_committed = []        # portfolio return / n_pairs
    daily_invested = []         # portfolio return / open pairs
    cycle_log = []
    t = FORM_DAYS
    while t + TRADE_DAYS <= len(dates):
        form_idx = dates[t - FORM_DAYS:t]
        trade_idx = dates[t:t + TRADE_DAYS]
        form_px = prices.loc[form_idx].dropna(axis=1)         # complete formation data only
        if form_px.shape[1] < n_pairs:
            t += TRADE_DAYS
            continue
        trade_px = prices.loc[trade_idx, form_px.columns]
        valid = trade_px.dropna(axis=1)                       # need full trading data too
        form_px = form_px[valid.columns]
        trade_px = valid
        if trade_px.shape[1] < n_pairs:
            t += TRADE_DAYS
            continue

        pairs = select_pairs(form_px, n_pairs)
        pair_pnls = []
        total_trades = 0
        for a, b, sigma in pairs:
            if sigma <= 0:
                continue
            pnl, nt = trade_pair(a, b, form_px, trade_px, sigma, entry_z, cost_bps)
            pair_pnls.append(pnl)
            total_trades += nt
        if not pair_pnls:
            t += TRADE_DAYS
            continue

        mat = pd.concat(pair_pnls, axis=1)                    # dates x pairs
        committed = mat.sum(axis=1) / n_pairs                 # /all pairs
        open_count = (mat != 0).sum(axis=1).replace(0, np.nan)
        invested = (mat.sum(axis=1) / open_count).fillna(0)   # /open pairs

        daily_committed.append(committed)
        daily_invested.append(invested)
        cycle_log.append({
            "trade_start": str(trade_idx[0].date()),
            "trade_end": str(trade_idx[-1].date()),
            "n_universe": form_px.shape[1],
            "trades": total_trades,
            "cycle_ret_committed_%": round(((1 + committed).prod() - 1) * 100, 2),
        })
        t += TRADE_DAYS

    committed = pd.concat(daily_committed)
    invested = pd.concat(daily_invested)
    return committed, invested, pd.DataFrame(cycle_log)


def stats(rets: pd.Series, label: str):
    eq = (1 + rets).cumprod()
    n_years = len(rets) / 252
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    pos_days = (rets > 0).mean()
    return {
        "series": label,
        "total_%": round(total * 100, 1),
        "CAGR_%": round(cagr * 100, 2),
        "ann_vol_%": round(vol * 100, 2),
        "Sharpe": round(sharpe, 2),
        "max_DD_%": round(dd * 100, 1),
        "pos_days_%": round(pos_days * 100, 1),
        "years": round(n_years, 1),
    }, eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--npairs", type=int, default=20)
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--cost_bps", type=float, default=5.0, help="bps per leg per transaction")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    print(f"Downloading {len(UNIVERSE)} tickers from {args.start} ...", file=sys.stderr)
    raw = yf.download(UNIVERSE, start=args.start, interval="1d",
                      auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(how="all")

    committed, invested, cycles = run(prices, args.npairs, args.entry, args.cost_bps)

    print(f"\n=== GGR-style pairs backtest: {args.npairs} pairs, entry {args.entry}sigma, "
          f"{args.cost_bps}bps/leg, {args.start}->now ===")
    print(f"Universe: {len(UNIVERSE)} liquid names | 12m formation / 6m trading rolling "
          f"| {len(cycles)} cycles\n")

    rows = []
    for s, lbl in [(committed, "committed capital (/all 20 pairs)"),
                   (invested, "fully invested (/open pairs)")]:
        st, _ = stats(s, lbl)
        rows.append(st)
    print(pd.DataFrame(rows).to_string(index=False))

    # split into halves to show the post-2002/decline effect
    mid = len(committed) // 2
    print("\n--- committed-capital return, first half vs second half ---")
    h1, _ = stats(committed.iloc[:mid], f"{committed.index[0].date()}..{committed.index[mid].date()}")
    h2, _ = stats(committed.iloc[mid:], f"{committed.index[mid].date()}..{committed.index[-1].date()}")
    print(pd.DataFrame([h1, h2]).to_string(index=False))

    print(f"\nAvg trades/cycle: {cycles['trades'].mean():.0f} | "
          f"Best/worst cycle (committed): "
          f"{cycles['cycle_ret_committed_%'].max():.1f}% / "
          f"{cycles['cycle_ret_committed_%'].min():.1f}%")
    print(f"% of cycles positive: {(cycles['cycle_ret_committed_%'] > 0).mean()*100:.0f}%")

    if args.csv:
        _, eq = stats(committed, "committed")
        out = pd.DataFrame({"committed_ret": committed,
                            "committed_equity": eq,
                            "invested_ret": invested})
        out.to_csv(args.csv)
        print(f"\nSaved daily series -> {args.csv}")
        cycles.to_csv(args.csv.replace(".csv", "_cycles.csv"), index=False)


if __name__ == "__main__":
    main()
