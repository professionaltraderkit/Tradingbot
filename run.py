#!/usr/bin/env python3
"""Run the paper trading bot.

    python run.py            # live loop: evaluates instruments on their
                             # timeframes, sends 7am/9pm reports
    python run.py --once     # single evaluation pass, then exit
    python run.py --report morning|evening   # build & send a report now
"""

import argparse
import logging

import yaml
from dotenv import load_dotenv

from bot.engine import Engine


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
    load_dotenv()

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
