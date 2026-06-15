#!/usr/bin/env python3
"""Parameter optimizer for the VWAP-pullback bot (config-levels.yaml).

    python optimize.py [--config config-levels.yaml] [--days N]
                       [--train-frac 0.7] [--min-trades 15] [--top 12]

For each instrument it splits the real history into an in-sample (train)
slice and an out-of-sample (test) slice, sweeps a grid of strategy
parameters, and ranks combinations by how well they hold up on BOTH
slices — not just the best in-sample fit. That ranking is the whole point:
the top in-sample result is almost always overfit, especially on a month
of data. A combo that is positive in-sample AND out-of-sample is the only
kind worth trusting, and even then only as a hypothesis to forward-test on
paper.

Costs are charged exactly as in backtest.py, so results are net.
"""

import argparse
import itertools
import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from backtest import DEFAULT_COST_BPS, backtest_instrument, build_risk
from bot.data import MarketData

log = logging.getLogger(__name__)

# Grid swept per strategy. Keys must be that strategy's constructor params.
# Keep each modest — every combo is a full backtest on both slices.
GRIDS = {
    "vwap_pullback": {
        "adx_min": [18, 25, 32],
        "stop_atr": [0.5, 1.0],
        "target_r": [1.5, 2.5, 0.0],   # 0.0 = no fixed target (trail only)
        "trail_atr": [0.0, 2.5],
        "use_breakeven": [True, False],
        "trade_end": ["11:30", "15:30"],  # morning-only vs full session
    },
    "opening_range_breakout": {
        "or_minutes": [15, 30, 60],
        "stop_mode": ["range", "atr"],
        "target_r": [1.0, 2.0, 0.0],   # 0.0 = no fixed target (trail only)
        "trail_atr": [0.0, 2.0],
        "use_breakeven": [True, False],
        "entry_end": ["12:00", "15:00"],
    },
}


def combos(grid: dict):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        # An exit must exist: skip "no target AND no trail" (stop/EOD only).
        if combo.get("target_r", 1) == 0.0 and combo.get("trail_atr", 1) == 0.0:
            continue
        yield combo


def evaluate(inst, train_df, test_df, combo, risk, atr_period,
             equity, max_notional_pct, cost_bps):
    variant = {**inst, "params": {**inst.get("params", {}), **combo}}
    try:
        tr = backtest_instrument(variant, train_df, risk, atr_period,
                                 equity, max_notional_pct, cost_bps)
        te = backtest_instrument(variant, test_df, risk, atr_period,
                                 equity, max_notional_pct, cost_bps)
    except Exception as exc:  # invalid combo for this strategy
        log.debug("combo failed: %s", exc)
        return None
    return {"combo": combo, "train": tr, "test": te,
            "robust": min(tr["return_pct"], te["return_pct"])}


def optimize_instrument(inst, df, risk, atr_period, equity, max_notional_pct,
                        cost_bps, train_frac, min_trades, top):
    grid = GRIDS.get(inst["strategy"])
    if grid is None:
        print(f"\n=== {inst['name']} ({inst['ticker']}) — no optimizer grid "
              f"for strategy '{inst['strategy']}', skipping ===")
        return None
    split = int(len(df) * train_frac)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    print(f"\n=== {inst['name']} ({inst['ticker']}) — "
          f"{len(train_df)} train bars / {len(test_df)} test bars ===")

    all_combos = list(combos(grid))
    total = len(all_combos)
    min_test = max(3, min_trades // 3)  # the test slice is smaller
    rows = []
    for n, combo in enumerate(all_combos, 1):
        if n == 1 or n % 12 == 0 or n == total:  # periodic, one line each
            print(f"  ...evaluating {n}/{total} combos")
        res = evaluate(inst, train_df, test_df, combo, risk, atr_period,
                       equity, max_notional_pct, cost_bps)
        if res and res["train"]["trades"] >= min_trades \
                and res["test"]["trades"] >= min_test:
            rows.append(res)

    if not rows:
        print("  No combo produced enough trades on both slices. "
              "Try --min-trades lower or --days higher.")
        return None

    rows.sort(key=lambda r: r["robust"], reverse=True)
    print(f"\n  {'trTr':>5} {'trRet%':>7} {'trPF':>5} | "
          f"{'teTr':>5} {'teRet%':>7} {'tePF':>5} | params")
    print("  " + "-" * 78)
    for r in rows[:top]:
        tr, te = r["train"], r["test"]
        tr_pf = "inf" if tr["profit_factor"] == float("inf") else f"{tr['profit_factor']:.2f}"
        te_pf = "inf" if te["profit_factor"] == float("inf") else f"{te['profit_factor']:.2f}"
        params = ", ".join(f"{k}={v}" for k, v in r["combo"].items())
        print(f"  {tr['trades']:>5} {tr['return_pct']:>7.2f} {tr_pf:>5} | "
              f"{te['trades']:>5} {te['return_pct']:>7.2f} {te_pf:>5} | {params}")

    best = rows[0]
    robust_positive = best["train"]["return_pct"] > 0 and best["test"]["return_pct"] > 0
    print(f"\n  Best robust combo: {best['combo']}")
    if robust_positive:
        print("  -> positive in-sample AND out-of-sample. Worth paper-forward-testing.")
    else:
        print("  -> NOT positive on both slices. No reliable edge found here; "
              "do not deploy this on hope.")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize VWAP-pullback params")
    parser.add_argument("--config", default="config-levels.yaml")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--min-trades", type=int, default=15)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--cost-bps", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    load_dotenv(Path(__file__).resolve().parent / ".env")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    risk = build_risk(config)
    risk_cfg = config["risk"]
    equity = config["account"]["starting_equity"]
    max_notional_pct = risk_cfg.get("max_position_notional_pct", 100)
    atr_period = risk_cfg["atr_period"]
    cost_bps = (args.cost_bps if args.cost_bps is not None
                else config.get("costs", {}).get("per_side_bps", DEFAULT_COST_BPS))

    feed = MarketData()
    best_params = {}
    for inst in config["instruments"]:
        df = feed.fetch_candles(inst["ticker"], inst["timeframe"])
        if df.empty:
            print(f"!! no data for {inst['ticker']}, skipping")
            continue
        if args.days:
            cutoff = df.index[-1] - pd.Timedelta(days=args.days)
            df = df[df.index >= cutoff]
        best = optimize_instrument(inst, df, risk, atr_period, equity,
                                   max_notional_pct, cost_bps,
                                   args.train_frac, args.min_trades, args.top)
        if best:
            best_params[inst["name"]] = {**inst.get("params", {}), **best["combo"]}

    if best_params:
        print("\n" + "=" * 60)
        print("Best params per instrument (paste into config-levels.yaml,")
        print("then re-run `python backtest.py --config config-levels.yaml`):\n")
        print(yaml.safe_dump(best_params, sort_keys=False, default_flow_style=False))
        print("Reminder: these are tuned on past data. The only honest test is")
        print("forward paper trading on bars the optimizer never saw.")


if __name__ == "__main__":
    main()
