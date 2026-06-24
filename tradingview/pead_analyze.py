#!/usr/bin/env python3
"""
PEAD deep analysis — reads pead_events.csv (built by pead_backtest.py) and
answers the questions that decide whether this is tradeable:

  1. SIGNIFICANCE: is the top-quintile drift real? t-stats + hit rates.
  2. LONG-ONLY: the short leg is dead in large caps, so size the practical
     retail strategy = long top quintile/decile, annualized.
  3. COMBINED SIGNAL: double-sort on SUE *and* EAR (both high) — the
     literature's sweet spot — vs either alone.
  4. DECAY: has the drift faded? Split by era (pre-2012 / 2012-2018 / 2019+).

No downloads — pure pandas on the saved events.
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

HORIZONS = [1, 5, 20, 60]


def tstat(x):
    x = x.dropna()
    if len(x) < 3:
        return np.nan, np.nan
    t, p = stats.ttest_1samp(x, 0.0)
    return t, p


def per_quarter_quintile(df, signal, nbuckets=5):
    df = df.dropna(subset=[signal]).copy()
    df["q_period"] = pd.to_datetime(df["ann_date"]).dt.to_period("Q")

    def assign(g):
        if len(g) < nbuckets:
            g["bucket"] = np.nan
            return g
        g["bucket"] = pd.qcut(g[signal].rank(method="first"), nbuckets,
                              labels=range(1, nbuckets + 1)).astype(float)
        return g
    df = df.groupby("q_period", group_keys=False).apply(assign)
    return df.dropna(subset=["bucket"])


def section_significance(ev):
    print("\n" + "=" * 74)
    print("1. SIGNIFICANCE OF THE DRIFT (SUE quintiles, market-excess fwd returns)")
    print("=" * 74)
    df = per_quarter_quintile(ev, "surprise", 5)
    for h in HORIZONS:
        col = f"fwd_{h}"
        q5 = df[df.bucket == 5][col].dropna()
        q1 = df[df.bucket == 1][col].dropna()
        ls = q5.reset_index(drop=True) - q1.reset_index(drop=True).reindex(
            range(len(q5))).fillna(0)
        t5, p5 = tstat(q5)
        t_ls, p_ls = stats.ttest_ind(q5, q1, equal_var=False)
        print(f"\n  Hold {h}d:")
        print(f"    Q5 (best beats): mean {q5.mean()*100:+.2f}%  t={t5:+.2f} "
              f"p={p5:.3f}  win {(q5>0).mean()*100:.0f}%  n={len(q5)}")
        print(f"    Q1 (worst miss): mean {q1.mean()*100:+.2f}%  win "
              f"{(q1>0).mean()*100:.0f}%  n={len(q1)}")
        print(f"    Q5-Q1 spread:    {(q5.mean()-q1.mean())*100:+.2f}%  "
              f"t={t_ls:+.2f} p={p_ls:.3f}  {'<-- significant' if p_ls<0.05 else ''}")


def section_long_only(ev, cost_bps):
    print("\n" + "=" * 74)
    print("2. LONG-ONLY TOP-BUCKET (the practical retail strategy)")
    print("=" * 74)
    rt = 2 * cost_bps / 10000.0     # round trip
    for nb, name in [(5, "top quintile"), (10, "top decile")]:
        df = per_quarter_quintile(ev, "surprise", nb)
        top = df[df.bucket == nb]
        print(f"\n  --- {name} by SUE (n={len(top)}) ---")
        for h in HORIZONS:
            r = top[f"fwd_{h}"].dropna()
            net = r.mean() - rt
            ann = (1 + net) ** (252.0 / h) - 1
            t, p = tstat(r)
            print(f"    {h:>2}d: gross {r.mean()*100:+.2f}%  net {net*100:+.2f}%  "
                  f"-> ann ~{ann*100:+.1f}%  (t={t:+.2f} p={p:.3f} win {(r>0).mean()*100:.0f}%)")


def section_combined(ev, cost_bps):
    print("\n" + "=" * 74)
    print("3. COMBINED SIGNAL: high SUE *and* high EAR (literature sweet spot)")
    print("=" * 74)
    rt = 2 * cost_bps / 10000.0
    df = ev.dropna(subset=["surprise", "ear"]).copy()
    df["q_period"] = pd.to_datetime(df["ann_date"]).dt.to_period("Q")

    def assign(g):
        if len(g) < 5:
            g["sue_b"] = np.nan; g["ear_b"] = np.nan; return g
        g["sue_b"] = pd.qcut(g["surprise"].rank(method="first"), 5,
                             labels=range(1, 6)).astype(float)
        g["ear_b"] = pd.qcut(g["ear"].rank(method="first"), 5,
                             labels=range(1, 6)).astype(float)
        return g
    df = df.groupby("q_period", group_keys=False).apply(assign).dropna(subset=["sue_b", "ear_b"])

    both_hi = df[(df.sue_b == 5) & (df.ear_b == 5)]
    both_lo = df[(df.sue_b == 1) & (df.ear_b == 1)]
    sue_only = df[df.sue_b == 5]
    print(f"\n  'Both high' = top-SUE AND top-EAR quintile (n={len(both_hi)})")
    print(f"  'Both low'  = bottom-SUE AND bottom-EAR (n={len(both_lo)})")
    for h in HORIZONS:
        bh = both_hi[f"fwd_{h}"].dropna()
        bl = both_lo[f"fwd_{h}"].dropna()
        so = sue_only[f"fwd_{h}"].dropna()
        net = bh.mean() - rt
        ann = (1 + net) ** (252.0 / h) - 1
        t, p = tstat(bh)
        print(f"\n    Hold {h}d:")
        print(f"      both-high long: gross {bh.mean()*100:+.2f}% net {net*100:+.2f}% "
              f"ann ~{ann*100:+.1f}%  t={t:+.2f} p={p:.3f} win {(bh>0).mean()*100:.0f}% n={len(bh)}")
        print(f"      (SUE-only top quintile was {so.mean()*100:+.2f}% — combo "
              f"{'BEATS' if bh.mean()>so.mean() else 'lags'} it)")
        print(f"      both-low  short: gross {-bl.mean()*100:+.2f}% (drift down? "
              f"{'yes' if bl.mean()<0 else 'NO — losers drift up'})")


def section_decay(ev, cost_bps):
    print("\n" + "=" * 74)
    print("4. DECAY — top-SUE-quintile 60-day drift by era")
    print("=" * 74)
    rt = 2 * cost_bps / 10000.0
    df = per_quarter_quintile(ev, "surprise", 5)
    df["year"] = pd.to_datetime(df["ann_date"]).dt.year
    top = df[df.bucket == 5].copy()
    eras = [("2005-2011", 2005, 2011), ("2012-2018", 2012, 2018),
            ("2019-2026", 2019, 2026)]
    for h in [20, 60]:
        print(f"\n  Hold {h}d:")
        for name, lo, hi in eras:
            r = top[(top.year >= lo) & (top.year <= hi)][f"fwd_{h}"].dropna()
            if len(r) < 10:
                continue
            net = r.mean() - rt
            ann = (1 + net) ** (252.0 / h) - 1
            t, p = tstat(r)
            print(f"    {name}: gross {r.mean()*100:+.2f}%  ann~{ann*100:+.1f}%  "
                  f"t={t:+.2f} p={p:.3f}  n={len(r)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pead_events.csv")
    ap.add_argument("--cost_bps", type=float, default=10.0)
    args = ap.parse_args()
    ev = pd.read_csv(args.csv, parse_dates=["ann_date", "entry_day"])
    print(f"Loaded {len(ev)} events, {ev.ticker.nunique()} names, "
          f"{ev.ann_date.min().date()}..{ev.ann_date.max().date()}")
    section_significance(ev)
    section_long_only(ev, args.cost_bps)
    section_combined(ev, args.cost_bps)
    section_decay(ev, args.cost_bps)


if __name__ == "__main__":
    main()
