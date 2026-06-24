#!/usr/bin/env python3
"""
PEAD daily push — runs the live screener and pushes NEW BUY candidates to the
phone via ntfy (reusing the swing pipeline's notifier + config).

Designed to run every weekday from Task Scheduler. It is QUIET by design:
  - between earnings seasons there are no top-quintile beats -> no push.
  - each beat alerts EXACTLY ONCE (de-duplicated via pead_alert_state.json), so
    running daily with a multi-day lookback won't spam the same name.

Push channel = ntfy.sh topic from swing_notify_config.json (same one the swing
alert uses). Email backup if email_backup=true in that config.

Usage:
    python pead_notify.py                      # full universe, push new BUYs
    python pead_notify.py --universe mag7
    python pead_notify.py --combined           # require positive EAR too
    python pead_notify.py --force              # ignore de-dup (re-alert), for testing
    python pead_notify.py --test               # send a test push and exit
"""
import argparse
import json
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SWING_DIR = r"C:\Users\xxdis"            # where swing_notify.py + config live
sys.path.insert(0, HERE)
sys.path.insert(0, SWING_DIR)

import pead_live as pl                    # recent_reports, ear_and_quote, UNIVERSES, HOLD_DAYS
import swing_notify as sn                 # send_via_ntfy, send_email, load config

STATE_FILE = os.path.join(HERE, "pead_alert_state.json")
LOG_FILE = os.path.join(HERE, "pead_notify.log")
CONFIG = os.path.join(SWING_DIR, "swing_notify_config.json")


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return set(json.load(open(STATE_FILE)))
        except Exception:
            return set()
    return set()


def save_state(seen):
    try:
        json.dump(sorted(seen), open(STATE_FILE, "w"), indent=0)
    except Exception as e:
        log(f"WARN could not save state: {e}")


def load_config():
    try:
        return json.load(open(CONFIG))
    except Exception as e:
        log(f"ERROR reading {CONFIG}: {e}")
        return {}


def scan_buys(universe, lookback, combined):
    """Return list of BUY dicts (top-quintile beats) using the live-screen logic."""
    rep, trailing = pl.recent_reports(lookback, universe)
    if rep.empty or len(trailing) == 0:
        return []
    q80 = np.nanpercentile(trailing, 80)
    q90 = np.nanpercentile(trailing, 90)
    buys = []
    for _, r in rep.iterrows():
        if r["surprise"] < q80:                       # not top-quintile
            continue
        ear, last = pl.ear_and_quote(r["ticker"], r["ann_date"])
        if combined and not (np.isfinite(ear) and ear > 0):
            continue
        exit_date = (pd.Timestamp(r["ann_date"]) +
                     pd.tseries.offsets.BDay(pl.HOLD_DAYS + 1)).date()
        buys.append({
            "ticker": r["ticker"],
            "reported": r["ann_date"].date(),
            "surprise": round(float(r["surprise"]), 1),
            "ear": round(ear * 100, 1) if np.isfinite(ear) else None,
            "last": round(last, 2) if np.isfinite(last) else None,
            "rank": "TOP10%" if r["surprise"] >= q90 else "TOP20%",
            "exit": exit_date,
        })
    return buys


def format_message(buys, universe_name):
    lines = [f"PEAD BUY signals ({universe_name}) - {dt.date.today():%b %d}"]
    for b in buys:
        ear = f" EAR{b['ear']:+.0f}%" if b["ear"] is not None else ""
        px = f" @${b['last']}" if b["last"] is not None else ""
        lines.append(f"{b['ticker']}{px}: beat {b['surprise']:+.0f}% [{b['rank']}]"
                     f"{ear} -> hold ~60d, exit {b['exit']}")
    lines.append("Long-only, equal-weight. Don't bail early.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["full", "mag7", "core3"], default="full")
    ap.add_argument("--lookback", type=int, default=4,
                    help="calendar days of reporters to scan (covers a weekend gap)")
    ap.add_argument("--combined", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore de-dup state")
    ap.add_argument("--test", action="store_true", help="send a test push and exit")
    args = ap.parse_args()

    cfg = load_config()

    if args.test:
        ok, info = sn.send_via_ntfy("[test] PEAD notifier wired up OK.", cfg)
        log(f"TEST push: ok={ok} {info}")
        return

    universe = pl.UNIVERSES[args.universe]
    log(f"Scan start: universe={args.universe} ({len(universe)} names) "
        f"lookback={args.lookback} combined={args.combined}")

    buys = scan_buys(universe, args.lookback, args.combined)
    seen = set() if args.force else load_state()

    new = [b for b in buys if f"{b['ticker']}|{b['reported']}" not in seen]
    if not new:
        log(f"No NEW BUY signals ({len(buys)} total in window, all already alerted). "
            f"Staying quiet.")
        return

    msg = format_message(new, args.universe)
    log("NEW signals:\n" + msg)

    ok, info = sn.send_via_ntfy(msg, cfg)
    log(f"ntfy: ok={ok} {info}")
    if cfg.get("email_backup"):
        try:
            eok, einfo = sn.send_email(msg, cfg)
            log(f"email: ok={eok} {einfo}")
        except Exception as e:
            log(f"email failed: {e}")

    if ok and not args.force:
        for b in new:
            seen.add(f"{b['ticker']}|{b['reported']}")
        save_state(seen)
        log(f"State updated (+{len(new)} signals).")


if __name__ == "__main__":
    main()
