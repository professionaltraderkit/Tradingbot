#!/usr/bin/env python3
"""
PEAD live screener — surfaces stocks that JUST reported a big positive earnings
surprise, the names the backtest says drift up over the next ~60 trading days.

Strategy rule (validated by pead_backtest.py / pead_portfolio.py on 2005-2026):
  LONG names in the top SUE quintile of recent reporters, ideally with a positive
  3-day earnings announcement return (EAR) too. Hold ~60 trading days. Long-only
  (the short leg doesn't pay in large caps). Backtest: ~8-9%/yr market-excess
  alpha, Sharpe ~0.9, persists into 2019-2026.

Each run it:
  - pulls the last `--lookback` calendar days of earnings across the universe,
  - computes Surprise(%) and the 3-day EAR (excess vs SPY),
  - ranks reporters into surprise buckets vs the trailing 1y of reports,
  - flags BUY candidates = top-quintile surprise (and prints whether EAR confirms),
  - shows the suggested exit date = entry + 60 trading days.

Usage:
    python pead_live.py                 # scan default universe, last 7 days
    python pead_live.py --lookback 14
    python pead_live.py --combined      # only show SUE+EAR-confirmed names
"""
import argparse
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from pead_backtest import UNIVERSE as FULL_UNIVERSE   # reuse the liquid universe

# Sub-universes. CORE3 = the clean Mag-7 drifters (AMZN/NVDA/AAPL: big 60d
# drift + significant t-stats); MAG7 = full Magnificent 7 (MSFT/META barely
# drift, TSLA high-variance — but included for completeness).
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
CORE3 = ["AMZN", "NVDA", "AAPL"]
UNIVERSES = {"full": FULL_UNIVERSE, "mag7": MAG7, "core3": CORE3}

HOLD_DAYS = 60


def recent_reports(lookback_days, universe):
    """For each name, pull recent earnings and the trailing year of surprises
    (to rank today's number against its own + peers' distribution)."""
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
    yr_ago = pd.Timestamp.today().normalize() - pd.Timedelta(days=400)
    rows = []
    trailing = []
    for tk in universe:
        try:
            df = yf.Ticker(tk).get_earnings_dates(limit=12)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        col = [c for c in df.columns if "Surprise" in c]
        if not col:
            continue
        df = df.rename(columns={col[0]: "surprise"})[["surprise"]].dropna()
        # trailing year of realized surprises -> distribution for ranking
        for d, r in df[(df.index >= yr_ago) & (df.index <= pd.Timestamp.today())].iterrows():
            trailing.append(r["surprise"])
        # recent reporters within the lookback window
        for d, r in df[(df.index >= cutoff) & (df.index <= pd.Timestamp.today())].iterrows():
            rows.append({"ticker": tk, "ann_date": d, "surprise": r["surprise"]})
    return pd.DataFrame(rows), np.array(trailing, dtype=float)


def ear_and_quote(tk, ann_date):
    """3-day market-excess announcement return + last price."""
    try:
        px = yf.download([tk, "SPY"], start=ann_date - pd.Timedelta(days=10),
                         end=pd.Timestamp.today() + pd.Timedelta(days=1),
                         interval="1d", auto_adjust=True, progress=False)
        px = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px
        s, spy = px[tk].dropna(), px["SPY"].dropna()
        idx = s.index
        pos = idx.searchsorted(ann_date)
        lo = max(0, pos - 1)
        hi = min(len(idx) - 1, pos + 1)
        ear = (s.iloc[hi] / s.iloc[lo] - 1) - (spy.loc[idx[hi]] / spy.loc[idx[lo]] - 1)
        return ear, float(s.iloc[-1])
    except Exception:
        return np.nan, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=7, help="calendar days of reporters")
    ap.add_argument("--combined", action="store_true", help="require positive EAR confirm")
    ap.add_argument("--universe", choices=["full", "mag7", "core3"], default="full",
                    help="full=71 liquid names, mag7=Magnificent 7, "
                         "core3=AMZN/NVDA/AAPL (cleanest drifters)")
    args = ap.parse_args()

    universe = UNIVERSES[args.universe]
    print(f"Scanning {len(universe)} names ({args.universe}) for earnings in the "
          f"last {args.lookback} days...", file=sys.stderr)
    rep, trailing = recent_reports(args.lookback, universe)
    if rep.empty:
        print("No earnings reports found in the lookback window.")
        return

    # rank thresholds from trailing-year surprise distribution
    q80 = np.nanpercentile(trailing, 80) if len(trailing) else 0
    q90 = np.nanpercentile(trailing, 90) if len(trailing) else 0

    out = []
    for _, r in rep.iterrows():
        ear, last = ear_and_quote(r["ticker"], r["ann_date"])
        top_q = r["surprise"] >= q80
        exit_date = (pd.Timestamp(r["ann_date"]) +
                     pd.tseries.offsets.BDay(HOLD_DAYS + 1)).date()
        out.append({
            "ticker": r["ticker"],
            "reported": r["ann_date"].date(),
            "surprise_%": round(r["surprise"], 1),
            "EAR_%": round(ear * 100, 1) if np.isfinite(ear) else np.nan,
            "last": round(last, 2) if np.isfinite(last) else np.nan,
            "rank": "TOP10%" if r["surprise"] >= q90 else ("TOP20%" if top_q else "-"),
            "exit_~60d": exit_date,
            "BUY": "YES" if (top_q and (not args.combined or (np.isfinite(ear) and ear > 0))) else "",
        })
    res = pd.DataFrame(out).sort_values(["BUY", "surprise_%"], ascending=[False, False])

    pd.set_option("display.width", 160)
    print(f"\n=== PEAD live screen | trailing-yr surprise cut: TOP20%>={q80:.1f}%  "
          f"TOP10%>={q90:.1f}% ===")
    print(res.to_string(index=False))
    buys = res[res["BUY"] == "YES"]["ticker"].tolist()
    print(f"\nBUY candidates (top-quintile beat{' + positive EAR' if args.combined else ''}): "
          f"{buys if buys else 'none this window'}")
    print(f"Rule: long-only, equal-weight, hold ~{HOLD_DAYS} trading days, "
          f"exit on the dates shown. Backtest alpha ~8-9%/yr, Sharpe ~0.9.")


if __name__ == "__main__":
    main()
