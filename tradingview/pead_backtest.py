#!/usr/bin/env python3
"""
PEAD (Post-Earnings Announcement Drift) backtest.

The classic anomaly (Ball & Brown 1968; Bernard & Thomas 1989): stocks that
beat earnings keep drifting UP, and stocks that miss keep drifting DOWN, for
weeks after the announcement. A long-top / short-bottom portfolio harvests the
spread. We test whether it still pays for a retail swing trader today.

Two surprise signals, both standard in the literature:
  - SUE  : standardized unexpected earnings. We use yfinance's reported
           Surprise(%) = (actual - estimate)/|estimate| as the analyst-based
           surprise (a.k.a. SUE-analyst, the version that works best post-1990).
  - EAR  : earnings announcement return = the stock's market-excess return over
           the 3-day window around the announcement. Captures the *price*
           reaction, which Brandt/Kishore/Santa-Clara/Venkatachalam (2008) show
           predicts drift better than the accounting surprise.

Design:
  - Universe of liquid US names (configurable).
  - For every historical earnings event we know the date + Surprise(%).
  - Entry the day AFTER the announcement (avoid the gap / look-ahead), at close.
  - Hold HOLD_DAYS trading days, exit at close. Forward return is market-excess
    (subtract SPY over the same window) so we measure pure drift, not beta.
  - Each rebalance/event is sorted into quintiles by signal; the strategy is
    long Q5 (best surprise) / short Q1 (worst). We report the long-short spread
    AND each quintile, per holding horizon (1/5/20/60 days), gross and net.
  - Costs: round-trip per side in bps (entry+exit), applied to L/S.

This is an EVENT-STUDY style test (average drift per event), the standard way
PEAD is documented, plus a tradeable calendar-time long-short return.

Usage:
    python pead_backtest.py
    python pead_backtest.py --start 2010-01-01 --hold 60 --cost_bps 10
    python pead_backtest.py --signal ear --csv pead_events.csv
"""
import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Liquid, deep-history US large caps across sectors. Earnings surprises are
# cleanest (analyst coverage, reliable estimates) on names like these.
UNIVERSE = [
    # tech / semis
    "AAPL", "MSFT", "ORCL", "IBM", "CSCO", "INTC", "TXN", "QCOM", "AMD", "NVDA",
    "ADBE", "CRM", "AMAT", "MU", "AVGO",
    # internet / media
    "GOOGL", "META", "AMZN", "NFLX", "DIS", "CMCSA",
    # financials
    "JPM", "BAC", "C", "WFC", "GS", "MS", "AXP", "V", "MA", "BLK", "SCHW",
    # consumer
    "KO", "PEP", "PG", "CL", "WMT", "TGT", "COST", "MCD", "SBUX", "HD", "LOW",
    "NKE", "DIS",
    # health care
    "JNJ", "PFE", "MRK", "ABT", "BMY", "UNH", "AMGN", "GILD", "LLY", "TMO",
    # energy / industrials
    "XOM", "CVX", "COP", "SLB", "CAT", "DE", "UPS", "FDX", "HON", "MMM", "GE",
    "BA", "LMT",
    # telecom / staples
    "T", "VZ", "MO", "PM",
]

HORIZONS = [1, 5, 20, 60]   # trading-day holding windows to evaluate
EAR_WINDOW = 1              # +/- days around announcement for the EAR signal (3-day)


def get_earnings(ticker, start, max_rows=100):  # Yahoo caps limit at 100
    """Return DataFrame indexed by announcement date with Surprise(%)."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.get_earnings_dates(limit=max_rows)
    except Exception as e:
        print(f"  {ticker}: earnings fetch failed ({e})", file=sys.stderr)
        return None
    if df is None or df.empty:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="first")]
    # keep only past events with a realized surprise
    col = [c for c in df.columns if "Surprise" in c]
    if not col:
        return None
    df = df.rename(columns={col[0]: "surprise_pct"})
    df = df[["surprise_pct"]].dropna()
    df = df[df.index >= pd.Timestamp(start)]
    df = df[df.index <= pd.Timestamp.today().normalize()]
    return df if len(df) else None


def build_events(start, cost_bps):
    """Download prices + earnings for the universe, build one row per
    (ticker, announcement) with forward market-excess returns at each horizon."""
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    print(f"Downloading prices for {len(UNIVERSE)+1} tickers...", file=sys.stderr)
    px = yf.download(UNIVERSE + ["SPY"], start=start, end=end,
                     interval="1d", auto_adjust=True, progress=False)
    px = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px
    px = px.sort_index()
    spy = px["SPY"]

    events = []
    for i, tk in enumerate(UNIVERSE):
        if tk not in px.columns:
            continue
        edf = get_earnings(tk, start)
        if edf is None:
            continue
        s = px[tk].dropna()
        if s.empty:
            continue
        idx = s.index
        for ann_date, row in edf.iterrows():
            # locate the first trading day strictly AFTER the announcement
            after = idx[idx > ann_date]
            if len(after) < max(HORIZONS) + 2:
                continue
            entry_day = after[0]                 # buy at this close (T+1)
            entry_pos = idx.get_loc(entry_day)
            if entry_pos + max(HORIZONS) >= len(idx):
                continue
            entry_px = s.iloc[entry_pos]
            entry_spy = spy.loc[entry_day] if entry_day in spy.index else np.nan
            if not np.isfinite(entry_spy) or entry_spy <= 0 or entry_px <= 0:
                continue

            # EAR: 3-day market-excess return AROUND the announcement (the
            # price reaction itself) — known by entry, usable as a signal.
            ann_pos = idx.get_loc(after[0]) - 1  # last day <= ann_date ~ event day
            lo = max(0, ann_pos - EAR_WINDOW)
            hi = min(len(idx) - 1, ann_pos + EAR_WINDOW)
            ear = (s.iloc[hi] / s.iloc[lo] - 1) - (spy.loc[idx[hi]] / spy.loc[idx[lo]] - 1)

            rec = {"ticker": tk, "ann_date": ann_date, "entry_day": entry_day,
                   "surprise_pct": row["surprise_pct"], "ear": ear}
            for h in HORIZONS:
                ex_px = s.iloc[entry_pos + h]
                ex_spy = spy.loc[idx[entry_pos + h]]
                stock_ret = ex_px / entry_px - 1
                mkt_ret = ex_spy / entry_spy - 1
                rec[f"fwd_{h}"] = stock_ret - mkt_ret      # market-excess drift
            events.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(UNIVERSE)} tickers, {len(events)} events", file=sys.stderr)
        time.sleep(0.05)

    ev = pd.DataFrame(events)
    print(f"Built {len(ev)} earnings events.", file=sys.stderr)
    return ev


def quintile_table(ev, signal, horizon):
    """Sort events into surprise quintiles (cross-sectionally per CALENDAR
    QUARTER so we don't compare a 2008 event to a 2021 event), report mean
    market-excess forward return per quintile + the long-short spread."""
    df = ev.dropna(subset=[signal, f"fwd_{horizon}"]).copy()
    df["q_period"] = df["ann_date"].dt.to_period("Q")
    # rank within each quarter so quintiles are comparable cross-section
    def assign_q(g):
        if len(g) < 5:
            g["quintile"] = np.nan
            return g
        g["quintile"] = pd.qcut(g[signal].rank(method="first"), 5,
                                labels=[1, 2, 3, 4, 5]).astype(float)
        return g
    df = df.groupby("q_period", group_keys=False).apply(assign_q)
    df = df.dropna(subset=["quintile"])
    means = df.groupby("quintile")[f"fwd_{horizon}"].agg(["mean", "count"])
    means["mean_%"] = (means["mean"] * 100).round(2)
    ls = means.loc[5, "mean"] - means.loc[1, "mean"]
    return means, ls, df


def calendar_time_ls(df, horizon, cost_bps):
    """Build a tradeable calendar-time long-short return: each event in Q5 is a
    +1/H unit long, each Q1 a -1/H short, accrued over its holding window. We
    approximate the realized strategy return as the per-event L/S spread net of
    round-trip cost, annualized by overlap. Reported as mean per-event net
    return and an annualized estimate assuming the position is held H days."""
    longs = df[df["quintile"] == 5][f"fwd_{horizon}"]
    shorts = df[df["quintile"] == 1][f"fwd_{horizon}"]
    rt = 2 * cost_bps / 10000.0          # round trip, per side
    long_net = longs.mean() - rt
    short_net = -shorts.mean() - rt       # short profits when fwd<0
    ls_net = long_net + short_net         # dollar-neutral, both legs
    # annualize: ~ (252/horizon) independent holding periods per year
    periods_per_yr = 252.0 / horizon
    ann = (1 + ls_net) ** periods_per_yr - 1
    return {
        "long_gross_%": round(longs.mean() * 100, 2),
        "short_gross_%": round(-shorts.mean() * 100, 2),
        "LS_gross_%": round((longs.mean() - shorts.mean()) * 100, 2),
        "LS_net_%": round(ls_net * 100, 2),
        "ann_LS_net_%": round(ann * 100, 1),
        "n_long": len(longs), "n_short": len(shorts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--hold", type=int, default=None,
                    help="single holding horizon to spotlight (default: show all)")
    ap.add_argument("--signal", choices=["surprise", "ear", "both"], default="both")
    ap.add_argument("--cost_bps", type=float, default=10.0, help="bps per side round-trip")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    ev = build_events(args.start, args.cost_bps)
    if ev.empty:
        print("No events built — aborting.", file=sys.stderr)
        return

    ev = ev.rename(columns={"surprise_pct": "surprise"})
    signals = {"surprise": "surprise", "ear": "ear"}
    if args.signal != "both":
        signals = {args.signal: signals[args.signal]}
    horizons = [args.hold] if args.hold else HORIZONS

    print(f"\n{'='*72}")
    print(f"PEAD backtest | {args.start} -> now | {len(ev)} events | "
          f"{ev['ticker'].nunique()} names | cost {args.cost_bps}bps/side")
    print(f"Forward returns are MARKET-EXCESS (vs SPY). Entry T+1 close.")
    print(f"{'='*72}")

    for sname, scol in signals.items():
        label = {"surprise": "SUE (analyst surprise %)",
                 "ear": "EAR (3-day announcement return)"}[sname]
        print(f"\n#### SIGNAL: {label} ####")
        for h in horizons:
            means, ls, df = quintile_table(ev, scol, h)
            ct = calendar_time_ls(df, h, args.cost_bps)
            print(f"\n  Holding {h} trading days — mean market-excess return by surprise quintile:")
            qstr = "  ".join(f"Q{int(q)}:{means.loc[q,'mean_%']:+.2f}%"
                             for q in means.index)
            print(f"    {qstr}")
            print(f"    Long-Q5 short-Q1 spread (gross): {ls*100:+.2f}%   "
                  f"events L/S: {ct['n_long']}/{ct['n_short']}")
            print(f"    Tradeable L/S  net {ct['LS_net_%']:+.2f}%/event  "
                  f"-> annualized ~{ct['ann_LS_net_%']:+.1f}%  "
                  f"(long {ct['long_gross_%']:+.2f} / short {ct['short_gross_%']:+.2f} gross)")

    if args.csv:
        ev.to_csv(args.csv, index=False)
        print(f"\nSaved {len(ev)} events -> {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
