from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class Signal(Enum):
    LONG = "long"
    SHORT = "short"
    EXIT = "exit"
    HOLD = "hold"


class Strategy(ABC):
    """A strategy looks at the latest candles and the current position side
    ("long", "short" or None) and returns one Signal for the last closed bar.

    Strategies that need richer position management can additionally
    override `initial_stop` (custom structure-based stop used for sizing
    and the hard stop) and `manage` (take-profit targets and stop
    tightening while a position is open).
    """

    #: minimum number of candles required for indicators to be valid
    min_bars: int = 50

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, position_side: str | None) -> Signal:
        ...

    def initial_stop(self, df: pd.DataFrame, side: str) -> float | None:
        """Custom stop price for a new position. None = use the default
        ATR-multiple stop from the risk config."""
        return None

    def manage(self, df: pd.DataFrame, position: dict) -> dict | None:
        """Called each cycle while a position is open, before `evaluate`.
        May return {"exit": reason, "price": fill_hint} to close, and/or
        {"stop": new_stop} to tighten the stop (the engine only ever
        tightens — a stop can never be moved further away)."""
        return None

    def stance(self, df: pd.DataFrame) -> str:
        """One-line human-readable description of the current setup,
        used by the morning briefing."""
        return ""
