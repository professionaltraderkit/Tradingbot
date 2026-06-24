"""Strategy-agnostic scoring for the Checker - the overfit guards, no backtest.

These primitives operate purely on a list of per-trade (or per-event) returns, so
every strategy in the repo can share one definition of "is this edge real?":
the per-symbol candle bots via validation.py, and the event-driven swing
strategies (PEAD, momentum) via their own drivers. Keeping them here - free of
any backtest/data import - is what lets a tradingview/ script reuse the exact
same verdict logic the intraday optimizer uses.

The three guards:
  * trade_tstat        - significance of the average trade (t-stat of the mean).
  * deflated_threshold - the multiple-testing bar: the t-stat you'd expect from
    the luckiest of N noise strategies (Bailey & Lopez de Prado, 2014), so the
    best of a grid search has to clear more than zero.
  * judge              - combine out-of-sample survival, that snooping bar, fold
    consistency and in->out decay into a PASS / WEAK / VETO ruling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

# Euler-Mascheroni constant, for the Gumbel expected-maximum approximation.
_EULER = 0.5772156649015329
_NORM = NormalDist()


def trade_tstat(net_pnls) -> float:
    """t-statistic of the mean trade: mean / std * sqrt(n).

    The per-trade Sharpe scaled by sqrt(n) - the natural unit for the
    multiple-testing bar. Returns 0 for fewer than two trades or no dispersion."""
    arr = np.asarray(list(net_pnls), dtype=float)
    n = arr.size
    if n < 2:
        return 0.0
    sd = arr.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(arr.mean() / sd * math.sqrt(n))


def expected_max_factor(n_trials: int) -> float:
    """Expected maximum of `n_trials` i.i.d. standard normals (Gumbel approx).

    Multiplied by the spread of the trials' t-stats, this is the t-stat you'd
    expect from the *luckiest* of N strategies with no real edge - the bar a
    grid-search winner has to clear to be more than the best of N coin flips."""
    if n_trials < 2:
        return 0.0
    a = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    b = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return (1.0 - _EULER) * a + _EULER * b


def deflated_threshold(trial_tstats, n_trials: int | None = None) -> float:
    """The multiple-testing bar: stdev(trial t-stats) * expected-max factor.

    A winner must beat this *out-of-sample* to count - beating it only in-sample
    is exactly the snooping trap. 0 when the trials show no dispersion (nothing
    to deflate) or there were too few of them."""
    arr = np.asarray([t for t in trial_tstats if np.isfinite(t)], dtype=float)
    n = arr.size if n_trials is None else n_trials
    if arr.size < 2 or n < 2:
        return 0.0
    return float(arr.std(ddof=1) * expected_max_factor(n))


@dataclass
class Verdict:
    """The Checker's ruling on one candidate, with the numbers behind it."""
    verdict: str                       # "PASS" | "WEAK" | "VETO"
    reasons: list[str] = field(default_factory=list)
    is_return: float = 0.0
    oos_return: float = 0.0
    oos_trades: int = 0
    oos_tstat: float = 0.0
    threshold_t: float = 0.0           # multiple-testing bar
    gap: float = 1.0                   # in->out per-trade decay (1.0 = all gone)
    consistency: float = 0.0           # fraction of OOS folds positive
    n_trials: int = 0

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def judge(*, is_return: float, oos_return: float, oos_trades: int,
          oos_tstat: float, threshold_t: float, fold_returns, gap: float,
          n_trials: int, min_oos_trades: int = 20, gap_tol: float = 0.6) -> Verdict:
    """Apply the three guards. PASS only with zero objections; WEAK if the edge is
    positive out-of-sample but trips a rigor gate; VETO if it isn't positive."""
    folds = list(fold_returns)
    consistency = (sum(1 for r in folds if r > 0) / len(folds)) if folds else 0.0
    reasons: list[str] = []

    if oos_trades < min_oos_trades:
        reasons.append(f"thin sample: {oos_trades} OOS trades (need {min_oos_trades})")
    if oos_return <= 0:
        reasons.append(f"OOS return {oos_return:+.2f}% not positive")
    if consistency < 0.5:
        reasons.append(f"only {consistency * 100:.0f}% of folds positive (need 50%)")
    if threshold_t > 0 and oos_tstat < threshold_t:
        reasons.append(f"OOS t-stat {oos_tstat:.2f} below multiple-testing bar "
                       f"{threshold_t:.2f} (best of N={n_trials})")
    if gap > gap_tol:
        reasons.append(f"per-trade edge decayed {gap * 100:.0f}% in->out "
                       f"(limit {gap_tol * 100:.0f}%)")

    if not reasons:
        verdict = "PASS"
    elif oos_return > 0:
        verdict = "WEAK"
    else:
        verdict = "VETO"
    return Verdict(verdict, reasons, is_return, oos_return, oos_trades, oos_tstat,
                   threshold_t, gap, consistency, n_trials)
