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
    """

    #: minimum number of candles required for indicators to be valid
    min_bars: int = 50

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, position_side: str | None) -> Signal:
        ...

    def stance(self, df: pd.DataFrame) -> str:
        """One-line human-readable description of the current setup,
        used by the morning briefing."""
        return ""
