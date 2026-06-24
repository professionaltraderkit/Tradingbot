#!/usr/bin/env python3
"""
PEAD calendar-time portfolio backtest — turns the event study into a real
tradeable equity curve with Sharpe / CAGR / max drawdown.

Strategy (long-only, the leg that actually works in large caps):
  - Read earnings events from pead_events.csv (ticker, entry_day, surprise, ear).
  - Each quarter, flag the qualifying events: top-SUE quintile, optionally also
    requiring top-EAR (the combined 'both-high' signal).
  - On each qualifying entry_day, open an equal-dollar long held HOLD days.
  - The portfolio holds ALL currently-open positions, equal-weighted, rebalanced
    daily. Daily return = mean of held names' returns. We also report the
    MARKET-EXCESS curve (minus SPY) so you can see pure alpha vs just being long.
  - Costs: cost_bps per side charged on each position's open and close day.

This is the honest 'could I have traded it' test: positions overlap, capital is
shared, and we get Sharpe + drawdown, not just average per-event drift.

Usage:
    python pead_portfolio.py --hold 60 --combined
    python pead_portfolio.py --hold 20 --bucket quintile --cost_bps 10
"""
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def qualifying_events(ev, bucket, combined):
    """Return events in the top SUE bucket (and top EAR if combined),
    ranked within each calendar quarter."""
    ev = ev.dropna(subset=["surprise"]).copy()
    ev["q_period"] = pd.to_datetime(ev["ann_date"]).dt.to_period("Q")
    nb = 5 if bucket == "quintile" else 10

    def assign(g):
        if len(g) < nb:
            g["sue_b"] = np.nan
            g["ear_b"] = np.nan
            return g
        g["sue_b"] = pd.qcut(g["surprise"].rank(method="first"), nb,
                             labels=range(1, nb + 1)).astype(float)
        if g["ear"].notna().sum() >= 5:
            g["ear_b"] = pd.qcut(g["ear"].rank(method="first"), 5,
                                 labels=range(1, 6)).astype(float)
        else:
            g["ear_b"] = np.nan
        return g
    ev = ev.groupby("q_period", group_keys=False).apply(assign)
    sel = ev[ev["sue_b"] == nb]
    if combined:
        sel = sel[sel["ear_b"] == 5]
    return sel[["ticker", "entry_day", "surprise", "ear"]].dropna(subset=["entry_day"])


def simulate(sel, hold, cost_bps):
    """Build the daily equal-weight long-only portfolio return series and the
    market-excess (vs SPY) series."""
    tickers = sorted(sel["ticker"].unique())
    start = pd.to_datetime(sel["entry_day"]).min() - pd.Timedelta(days=5)
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    print(f"Downloading {len(tickers)} names for portfolio sim...", file=sys.stderr)
    px = yf.download(tickers + ["SPY"], start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False)
    px = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px
    px = px.sort_index()
    rets = px.pct_change()
    spy_ret = rets["SPY"]
    cal = px.index
    c = cost_bps / 10000.0

    # Build a list of (ticker, open_idx, close_idx) positions
    positions = []
    for _, r in sel.iterrows():
        ed = pd.Timestamp(r["entry_day"])
        if ed not in cal:
            loc = cal.searchsorted(ed)
            if loc >= len(cal):
                continue
            ed = cal[loc]
        oi = cal.get_loc(ed)
        ci = min(oi + hold, len(cal) - 1)
        if r["ticker"] in rets.columns:
            positions.append((r["ticker"], oi, ci))

    n = len(cal)
    port_ret = np.zeros(n)
    held_count = np.zeros(n)
    cost_drag = np.zeros(n)
    for tk, oi, ci in positions:
        tr = rets[tk].values
        for d in range(oi + 1, ci + 1):           # returns accrue day after open
            if np.isfinite(tr[d]):
                port_ret[d] += tr[d]
                held_count[d] += 1
        cost_drag[oi] += c                          # open cost
        cost_drag[ci] += c                          # close cost

    held_safe = np.where(held_count > 0, held_count, np.nan)
    gross = port_ret / held_safe                    # equal-weight mean
    # cost: spread the per-position open/close cost over the number of names held
    cost_per_day = cost_drag / np.where(held_count > 0, held_count, 1)
    net = np.nan_to_num(gross) - cost_per_day
    net = pd.Series(net, index=cal)
    held = pd.Series(held_count, index=cal)
    excess = net - spy_ret.fillna(0)                # market-excess (alpha) curve
    # only count days we actually held something
    active = held > 0
    return net[active], excess[active], held[active]


def stats(rets, label):
    rets = rets.dropna()
    eq = (1 + rets).cumprod()
    yrs = len(rets) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = rets.std() * np.sqrt(252)
    sharpe = rets.mean() * 252 / vol if vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {
        "series": label,
        "total_%": round((eq.iloc[-1] - 1) * 100, 1),
        "CAGR_%": round(cagr * 100, 2),
        "vol_%": round(vol * 100, 2),
        "Sharpe": round(sharpe, 2),
        "maxDD_%": round(dd * 100, 1),
        "days": len(rets),
        "yrs": round(yrs, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pead_events.csv")
    ap.add_argument("--hold", type=int, default=60)
    ap.add_argument("--bucket", choices=["quintile", "decile"], default="quintile")
    ap.add_argument("--combined", action="store_true",
                    help="require top-EAR as well as top-SUE")
    ap.add_argument("--cost_bps", type=float, default=10.0)
    args = ap.parse_args()

    ev = pd.read_csv(args.csv, parse_dates=["ann_date", "entry_day"])
    sel = qualifying_events(ev, args.bucket, args.combined)
    print(f"Qualifying entries: {len(sel)} "
          f"({'SUE+EAR combined' if args.combined else 'SUE ' + args.bucket}), "
          f"hold {args.hold}d, {args.cost_bps}bps/side")

    net, excess, held = simulate(sel, args.hold, args.cost_bps)
    print(f"Avg names held concurrently: {held.mean():.1f}  "
          f"(min {int(held.min())}, max {int(held.max())})")

    print("\n=== PEAD long-only portfolio ===")
    rows = [stats(net, "absolute (long the basket)"),
            stats(excess, "market-excess (alpha vs SPY)")]
    print(pd.DataFrame(rows).to_string(index=False))

    # SPY buy&hold over the same window for reference
    spy = yf.download("SPY", start=net.index[0], end=net.index[-1],
                      interval="1d", auto_adjust=True, progress=False)
    spy = spy["Close"] if isinstance(spy.columns, pd.MultiIndex) else spy
    spy_r = spy.pct_change().dropna()
    if isinstance(spy_r, pd.DataFrame):
        spy_r = spy_r.iloc[:, 0]
    print("\n--- reference: SPY buy & hold, same window ---")
    print(pd.DataFrame([stats(spy_r, "SPY")]).to_string(index=False))


if __name__ == "__main__":
    main()
