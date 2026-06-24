#!/usr/bin/env python3
"""
Cointegration / pairs-trading screener.

Ranks candidate pairs by how *tradeable* they are for a z-score mean-reversion
strategy. A pair is tradeable when its spread is stationary (mean-reverting) and
reverts fast enough to trade. This is the "stationarity gate" that separates the
GLD/GDX-type pairs (revert) from the XOM/CVX-type pairs (trend).

For each pair (A, B) it computes, on log prices:
  - corr        : correlation of log prices (sanity / co-movement)
  - beta        : OLS hedge ratio (slope of logA on logB)  -> dollar-neutral spread
  - coint_p     : Engle-Granger cointegration test p-value (lower = better)
  - adf_p       : ADF stationarity p-value on the OLS residual spread (lower = better)
  - half_life   : mean-reversion half-life in trading days (smaller = reverts faster)
  - hurst       : Hurst exponent of the spread (<0.5 = mean-reverting)
  - z_now       : current z-score of the spread (|z| high = setup is live now)

PASS criteria (tunable below): coint_p < 0.05 AND adf_p < 0.05 AND
                                MIN_HL <= half_life <= MAX_HL.

Usage:
    python cointegration_screener.py                 # default basket
    python cointegration_screener.py --years 3
    python cointegration_screener.py --csv out.csv
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# --- Candidate pairs: economically linked names that *might* cointegrate ---
DEFAULT_PAIRS = [
    # sector duals
    ("KO", "PEP"),      # beverages
    ("V", "MA"),        # card networks
    ("HD", "LOW"),      # home improvement
    ("GS", "MS"),       # investment banks
    ("UPS", "FDX"),     # logistics
    ("MAR", "HLT"),     # hotels
    ("CAT", "DE"),      # heavy machinery
    ("AZO", "ORLY"),    # auto parts
    ("CVS", "WBA"),     # pharmacy
    # ETF pairs
    ("XLE", "XOP"),     # energy / oil&gas E&P
    ("SPY", "IVV"),     # S&P twins
    ("EWA", "EWC"),     # Australia / Canada
    ("GDX", "GDXJ"),    # gold miners / junior miners
    ("XLF", "KRE"),     # financials / regional banks
    # commodity-linked
    ("GLD", "SLV"),     # gold / silver
    ("USO", "XLE"),     # oil / energy
    # sanity checks
    ("GLD", "GDX"),     # known: reverts (should rank well)
    ("XOM", "CVX"),     # known: trends (should rank poorly)
]

# PASS thresholds
COINT_P_MAX = 0.05
ADF_P_MAX = 0.05
MIN_HL = 1.0
MAX_HL = 60.0
Z_LOOKBACK = 20   # bars for the "current z-score" (matches the Pine strategy)


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life via AR(1): d(spread) = a + b*spread[-1]; HL=-ln2/b."""
    lag = spread.shift(1)
    delta = spread - lag
    df = pd.concat([delta, lag], axis=1).dropna()
    df.columns = ["delta", "lag"]
    if len(df) < 10:
        return np.nan
    X = sm.add_constant(df["lag"])
    b = sm.OLS(df["delta"], X).fit().params["lag"]
    if b >= 0:                       # not mean-reverting
        return np.nan
    return -np.log(2) / b


def hurst(ts: np.ndarray) -> float:
    """Hurst exponent via rescaled-range on lags. <0.5 mean-reverting, >0.5 trending."""
    ts = np.asarray(ts, dtype=float)
    lags = range(2, min(100, len(ts) // 2))
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    tau = np.array(tau)
    good = tau > 0
    if good.sum() < 5:
        return np.nan
    poly = np.polyfit(np.log(np.array(list(lags))[good]), np.log(tau[good]), 1)
    return poly[0]


def analyze_pair(a: str, b: str, prices: pd.DataFrame) -> dict:
    if a not in prices or b not in prices:
        return {"pair": f"{a}/{b}", "note": "missing data"}
    df = prices[[a, b]].dropna()
    if len(df) < 120:
        return {"pair": f"{a}/{b}", "note": f"only {len(df)} bars"}

    la, lb = np.log(df[a]), np.log(df[b])
    corr = la.corr(lb)

    # OLS hedge ratio -> residual spread
    X = sm.add_constant(lb)
    model = sm.OLS(la, X).fit()
    beta = model.params[b]
    spread = la - beta * lb - model.params["const"]

    coint_p = coint(la, lb)[1]
    adf_p = adfuller(spread.dropna(), autolag="AIC")[1]
    hl = half_life(spread)
    h = hurst(spread.values)

    # current z-score over Z_LOOKBACK bars
    s = spread.tail(Z_LOOKBACK)
    z_now = (spread.iloc[-1] - s.mean()) / s.std() if s.std() > 0 else np.nan

    passes = (
        coint_p < COINT_P_MAX
        and adf_p < ADF_P_MAX
        and (not np.isnan(hl))
        and MIN_HL <= hl <= MAX_HL
    )
    return {
        "pair": f"{a}/{b}",
        "corr": round(corr, 3),
        "beta": round(beta, 3),
        "coint_p": round(coint_p, 4),
        "adf_p": round(adf_p, 4),
        "half_life": round(hl, 1) if not np.isnan(hl) else np.nan,
        "hurst": round(h, 3) if not np.isnan(h) else np.nan,
        "z_now": round(z_now, 2) if not np.isnan(z_now) else np.nan,
        "PASS": "YES" if passes else "no",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=4.0, help="years of history")
    ap.add_argument("--csv", type=str, default=None, help="write ranked results to CSV")
    args = ap.parse_args()

    tickers = sorted({t for pair in DEFAULT_PAIRS for t in pair})
    period = f"{int(args.years * 365)}d"
    print(f"Downloading {len(tickers)} tickers, {args.years}y daily ...", file=sys.stderr)
    raw = yf.download(tickers, period=period, interval="1d",
                      auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    rows = [analyze_pair(a, b, prices) for a, b in DEFAULT_PAIRS]
    res = pd.DataFrame(rows)

    # rank: passers first, then by half_life ascending (fast reverters on top)
    res["_rank_pass"] = (res.get("PASS") == "YES").astype(int)
    res = res.sort_values(["_rank_pass", "half_life"],
                          ascending=[False, True]).drop(columns="_rank_pass")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== Cointegration / mean-reversion screen "
          f"({args.years:.0f}y daily) ===")
    print(res.to_string(index=False))
    passers = res[res.get("PASS") == "YES"]["pair"].tolist()
    print(f"\nTRADEABLE (passed gate): {passers if passers else 'none'}")
    print("Lower coint_p/adf_p = stronger reversion; shorter half_life = faster; "
          "|z_now| high = live setup.")

    if args.csv:
        res.to_csv(args.csv, index=False)
        print(f"\nSaved -> {args.csv}")


if __name__ == "__main__":
    main()
