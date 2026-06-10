#!/usr/bin/env python3
"""Run the paper trading bot.

    python run.py            # live loop: evaluates instruments on their
                             # timeframes, sends 7am/9pm reports
    python run.py --once     # single evaluation pass, then exit
    python run.py --report morning|evening   # build & send a report now
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from bot.engine import Engine

log = logging.getLogger("run")


def load_env() -> None:
    """Load .env from the bot's own directory and call out the common
    setup mistakes (.env never created, or saved as .env.txt) explicitly."""
    here = Path(__file__).resolve().parent
    env = here / ".env"
    if env.exists():
        load_dotenv(env)
        log.info("Loaded environment from %s", env)
        return
    txt = here / ".env.txt"
    if txt.exists():
        log.warning("Found %s — your editor added a .txt extension. "
                    "Rename it to just '.env' and run again.", txt)
    else:
        log.warning("No .env file at %s — copy .env.example to .env and "
                    "fill in your keys. Running without it (simulated "
                    "broker, console reports).", env)


def show_status(config: dict) -> None:
    """Print open positions, recent trades and recent activity from the
    local database — no network calls, works while the bot is running."""
    from bot.state import StateStore

    store = StateStore(config["storage"]["db_path"],
                       config["account"]["starting_equity"])
    positions = store.open_positions()
    print(f"\nOpen positions ({len(positions)}):")
    for p in positions:
        print(f"  {p['side']:5s} {p['name']:<12} qty {p['quantity']:.4f} "
              f"@ {p['entry_price']:,.2f}  stop {p['stop_price']:,.2f}  "
              f"since {p['opened_at'][:16]}")
    if not positions:
        print("  none")

    trades = store.all_trades()
    print(f"\nLast {min(10, len(trades))} closed trades (of {len(trades)} total):")
    for t in trades[-10:]:
        print(f"  {t['closed_at'][:16]}  {t['side']:5s} {t['name']:<12} "
              f"{t['entry_price']:,.2f} -> {t['exit_price']:,.2f}  "
              f"${t['pnl']:+,.2f}  ({t['reason']})")
    if not trades:
        print("  none yet")

    if trades:
        wins = sum(1 for t in trades if t["pnl"] > 0)
        total = sum(t["pnl"] for t in trades)
        print(f"\nAll-time: {len(trades)} trades, {wins / len(trades) * 100:.0f}% "
              f"wins, realized P&L ${total:+,.2f}")

    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    events = store.events_since(week_ago)
    print(f"\nLast {min(10, len(events))} events (7 days):")
    for ev in events[-10:]:
        print(f"  {ev['at'][:16]}  [{ev['kind']}] {ev['message']}")
    if not events:
        print("  none")
    print()
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-instrument paper trading bot")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true",
                        help="run one evaluation cycle and exit")
    parser.add_argument("--report", choices=["morning", "evening"],
                        help="build and send a report immediately, then exit")
    parser.add_argument("--status", action="store_true",
                        help="print open positions and recent trades, then exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.status:
        show_status(config)
        return

    engine = Engine(config)
    if args.report == "morning":
        engine.notifier.send(engine._build_morning())
    elif args.report == "evening":
        engine.notifier.send(engine._build_evening())
    elif args.once:
        engine.run_once()
    else:
        engine.run_forever()


if __name__ == "__main__":
    main()
