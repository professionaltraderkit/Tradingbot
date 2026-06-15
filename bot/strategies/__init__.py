from .base import Signal, Strategy
from .level_reversal import LevelReversal
from .mean_reversion import MeanReversion
from .momentum_breakout import MomentumBreakout
from .opening_range_breakout import OpeningRangeBreakout
from .trend_following import TrendFollowing
from .vwap_pullback import VwapPullback

STRATEGIES = {
    "mean_reversion": MeanReversion,
    "momentum_breakout": MomentumBreakout,
    "trend_following": TrendFollowing,
    "level_reversal": LevelReversal,
    "vwap_pullback": VwapPullback,
    "opening_range_breakout": OpeningRangeBreakout,
}


def build_strategy(name: str, params: dict) -> Strategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"Unknown strategy: {name}") from None
    return cls(**params)
