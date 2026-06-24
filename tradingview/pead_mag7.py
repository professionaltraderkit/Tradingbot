#!/usr/bin/env python3
"""
PEAD on the Magnificent 7 specifically — do these momentum names drift after
earnings beats the way the broad universe does? Tests per-name + the basket.

Reuses pead_backtest.build_events by overriding its UNIVERSE to the Mag 7
(TSLA is pulled fresh; the other six were already in the main run).
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

import pead_backtest as pb

MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
HORIZONS = [5, 20, 60]


def tstat(x):
    x = x.dropna()
    if len(x) < 5:
        return np.nan, np.nan
    t, p = stats.ttest_1samp(x, 0.0)
    return t, p


def main():
    pb.UNIVERSE = MAG7
    ev = pb.build_events("2005-01-01", cost_bps=10)
    ev = ev.rename(columns={"surprise_pct": "surprise"})
    ev["ann_date"] = pd.to_datetime(ev["ann_date"])
    print(f"\n{'='*70}\nMAG-7 PEAD | {len(ev)} events | "
          f"{ev.ann_date.min().date()}..{ev.ann_date.max().date()}\n{'='*70}")

    # --- per-name: median surprise, and drift conditional on a BEAT (surprise>0) ---
    print("\nPER-NAME (market-excess fwd return AFTER A BEAT, surprise>0):")
    print(f"{'name':6} {'#beats':>6} {'beat%':>6} "
          + "  ".join(f"{f'{h}d':>14}" for h in HORIZONS))
    for tk in MAG7:
        sub = ev[ev.ticker == tk]
        beats = sub[sub.surprise > 0]
        beat_rate = len(beats) / len(sub) * 100 if len(sub) else 0
        cells = []
        for h in HORIZONS:
            r = beats[f"fwd_{h}"].dropna()
            t, p = tstat(r)
            star = "*" if (np.isfinite(p) and p < 0.10) else " "
            cells.append(f"{r.mean()*100:+5.2f}%(t{t:+.1f}){star}" if len(r) else "    n/a    ")
        print(f"{tk:6} {len(beats):>6} {beat_rate:>5.0f}% " + "  ".join(cells))

    # --- basket: all Mag-7 beats pooled, and top-half-surprise vs bottom-half ---
    print("\nBASKET (all 7 pooled):")
    beats = ev[ev.surprise > 0]
    misses = ev[ev.surprise <= 0]
    for h in HORIZONS:
        rb = beats[f"fwd_{h}"].dropna()
        rm = misses[f"fwd_{h}"].dropna()
        tb, pb_ = tstat(rb)
        print(f"  {h:>2}d  beat: {rb.mean()*100:+.2f}% (t={tb:+.2f} p={pb_:.3f} "
              f"win {(rb>0).mean()*100:.0f}% n={len(rb)})   "
              f"miss: {rm.mean()*100:+.2f}% (n={len(rm)})   "
              f"spread {(rb.mean()-rm.mean())*100:+.2f}%")

    # --- big-beat subset: surprise in the top tercile of Mag-7 beats ---
    print("\nBIG-BEAT subset (surprise in top third of all Mag-7 surprises):")
    cut = ev["surprise"].quantile(0.667)
    big = ev[ev.surprise >= cut]
    for h in HORIZONS:
        r = big[f"fwd_{h}"].dropna()
        t, p = tstat(r)
        net = r.mean() - 0.002
        ann = (1 + net) ** (252.0 / h) - 1
        print(f"  {h:>2}d  {r.mean()*100:+.2f}% gross  net {net*100:+.2f}% "
              f"-> ann ~{ann*100:+.1f}%  (t={t:+.2f} p={p:.3f} win {(r>0).mean()*100:.0f}% "
              f"n={len(r)}, surprise>={cut:.1f}%)")

    ev.to_csv("pead_mag7_events.csv", index=False)


if __name__ == "__main__":
    main()
