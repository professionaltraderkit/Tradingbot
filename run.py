#!/usr/bin/env python3
"""Run the paper trading bot.

    python run.py            # live loop: evaluates instruments on their
                             # timeframes, sends 7am/9pm reports
    python run.py --once     # single evaluation pass, then exit
    python run.py --report morning|evening   # build & send a report now
"""

import argparse
import logging
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-instrument paper trading bot")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true",
                        help="run one evaluation cycle and exit")
    parser.add_argument("--report", choices=["morning", "evening"],
                        help="build and send a report immediately, then exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_env()

    with open(args.config) as f:
        config = yaml.safe_load(f)

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
