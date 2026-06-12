from .base import Signal, Strategy
from .level_reversal import LevelReversal
from .mean_reversion import MeanReversion
from .momentum_breakout import MomentumBreakout
from .trend_following import TrendFollowing

STRATEGIES = {
    "mean_reversion": MeanReversion,
    "momentum_breakout": MomentumBreakout,
    "trend_following": TrendFollowing,
    "level_reversal": LevelReversal,
}


def build_strategy(name: str, params: dict) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"Unknown strategy: {name}") from None
    return cls(**params)
