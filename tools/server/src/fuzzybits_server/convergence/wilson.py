"""Wilson score interval, the zero-hit bound, and the #224 goal rule.

The controller library (#224) owns intervals, convergence flags and stopping.
Until phase 2 exposes ``fuzzybits.Controller``, the server still has to execute
a stopping specification faithfully, which means evaluating *the same interval
with the same rule* the controller will use. That rule is fixed in the #224
definitions comment:

* interval: Wilson score at level ``c``; ``z = Phi^-1(1 - alpha/2)``,
  ``alpha = 1 - c``; centre ``(p + z^2/2N) / (1 + z^2/N)``, half
  ``z/(1 + z^2/N) * sqrt(p(1-p)/N + z^2/4N^2)``;
* relative half-width: ``(U - L) / (2 p_hat)``, undefined at zero hits;
* goal: ``(U - L)/2 <= max(eps * p_hat, a)``, so ``--abs-error`` is a floor;
* zero hits: the reported upper bound is ``1 - (alpha/2)^(1/N)`` (the
  Clopper-Pearson upper limit at level ``c``).

This module is deleted in phase 2, when the interval and the rule come from the
controller library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .policies import PolicyVerdict
from .special import normal_quantile_two_sided
from .tally import NodeState, RunContext

#: Levels of the fixed 95%/99% bands streamed next to the client-requested level,
#: so the probability plot keeps its p05-p95 / p01-p99 presentation in goal mode.
DISPLAY_CONFIDENCE = 0.95
DISPLAY_CONFIDENCE_99 = 0.99


def z_for_confidence(confidence: float) -> float:
    """Two-sided normal quantile for a confidence level in (0, 1)."""
    return normal_quantile_two_sided(confidence)


def wilson_interval(hits: int, trials: int, confidence: float) -> tuple[float, float]:
    r"""Wilson score interval at ``confidence`` (#224 definition 1).

    Returns ``(nan, nan)`` for ``trials <= 0``; the interval is undefined before
    the first committed pass.
    """
    n = int(trials)
    if n <= 0:
        return math.nan, math.nan
    z = z_for_confidence(confidence)
    p = min(1.0, max(0.0, int(hits) / n))
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    half = (z / denominator) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def zero_hit_upper_bound(trials: int, confidence: float) -> float:
    r"""Upper limit at zero hits: ``1 - (alpha/2)^(1/N)`` (#224 definition 4).

    This is the upper limit of the two-sided Clopper-Pearson interval at level
    ``confidence``, equivalently the one-sided bound at level ``1 - alpha/2``.

    Evaluated as ``-expm1(log(alpha/2) / N)``: the direct form loses about
    ``eps / bound`` relative accuracy for large ``N``, which is most of the digits
    at ``N = 10^9``.
    """
    n = int(trials)
    if n <= 0:
        return math.nan
    alpha = 1.0 - min(1.0, max(0.0, confidence))
    return -math.expm1(math.log(alpha / 2.0) / n)


def reported_interval(hits: int, trials: int, confidence: float) -> tuple[float, float]:
    """The interval the run reports, and the one the goal is evaluated on.

    Wilson for at least one hit; ``[0, 1 - (alpha/2)^(1/N)]`` at zero hits, so a
    zero-hit query reports a bound that narrows as trials grow instead of the
    zero-width Wald interval reported before #224.
    """
    n = int(trials)
    if n <= 0:
        return math.nan, math.nan
    if int(hits) <= 0:
        return 0.0, zero_hit_upper_bound(n, confidence)
    return wilson_interval(hits, n, confidence)


def half_width(interval: tuple[float, float]) -> float:
    """``(U - L) / 2``; ``nan`` when the interval is undefined."""
    low, high = interval
    if not (math.isfinite(low) and math.isfinite(high)):
        return math.nan
    return (high - low) / 2.0


def relative_half_width(interval: tuple[float, float], mean: float) -> float:
    """``(U - L) / (2 p_hat)``; ``nan`` at ``p_hat = 0`` (#224 definition 2)."""
    if mean <= 0.0:
        return math.nan
    half = half_width(interval)
    return half / mean if math.isfinite(half) else math.nan


def goal_bound(mean: float, rel_error: float | None, abs_error: float | None) -> float:
    """The half-width a goal run must reach: ``max(eps * p_hat, a)`` over what is set."""
    bounds: list[float] = []
    if rel_error is not None:
        bounds.append(rel_error * mean)
    if abs_error is not None:
        bounds.append(abs_error)
    return max(bounds) if bounds else math.nan


@dataclass(frozen=True, slots=True)
class GoalState:
    """One query's standing against the goal at a committed trial count."""

    hits: int
    trials: int
    mean: float
    ci_low: float
    ci_high: float
    abs_half_width: float
    rel_half_width: float
    target_half_width: float
    target_rel_half_width: float
    converged: bool
    zero_hits: bool
    zero_hit_upper_bound: float


def evaluate_goal(
    hits: int,
    trials: int,
    confidence: float,
    rel_error: float | None,
    abs_error: float | None,
) -> GoalState:
    """Evaluate the #224 goal rule for one query.

    A relative-only goal cannot be met at zero hits: the bound is ``eps * 0`` and
    the reported interval has positive width.
    """
    n = int(trials)
    k = int(hits)
    mean = k / n if n > 0 else 0.0
    low, high = reported_interval(k, n, confidence)
    abs_half = half_width((low, high))
    rel_half = relative_half_width((low, high), mean)
    target = goal_bound(mean, rel_error, abs_error)
    has_goal = rel_error is not None or abs_error is not None
    converged = bool(
        has_goal
        and n > 0
        and math.isfinite(abs_half)
        and math.isfinite(target)
        and abs_half > 0.0
        and abs_half <= target
    )
    return GoalState(
        hits=k,
        trials=n,
        mean=mean,
        ci_low=low,
        ci_high=high,
        abs_half_width=abs_half,
        rel_half_width=rel_half,
        target_half_width=target,
        target_rel_half_width=rel_error if rel_error is not None else math.nan,
        converged=converged,
        zero_hits=k == 0,
        zero_hit_upper_bound=zero_hit_upper_bound(n, confidence) if k == 0 else math.nan,
    )


def projected_trials_for_goal(
    hits: int,
    trials: int,
    confidence: float,
    rel_error: float | None,
    abs_error: float | None,
) -> int | float:
    """Smallest trial count at which the reported half-width reaches the goal.

    Solves the goal inequality with ``p_hat`` held at its current value: an
    estimate, not a guarantee, because ``p_hat`` moves as trials accrue. The
    half-width is strictly decreasing in ``N``, so a doubling search followed by
    bisection over the integer trial count finds the smallest ``N`` that meets
    it. Returns ``inf`` when no ``N <= 2**64`` reaches the bound, which includes
    a relative goal at zero hits.
    """
    if rel_error is None and abs_error is None:
        return 0
    n0 = max(1, int(trials))
    k = int(hits)
    mean = k / n0
    target = goal_bound(mean, rel_error, abs_error)
    if not math.isfinite(target) or target <= 0.0:
        return math.inf
    if k == 0 and rel_error is not None and abs_error is None:
        return math.inf

    def meets(n: int) -> bool:
        projected_hits = 0 if k <= 0 else min(n, int(round(mean * n)))
        low, high = reported_interval(projected_hits, n, confidence)
        return math.isfinite(high) and math.isfinite(low) and (high - low) / 2.0 <= target

    if meets(n0):
        return n0
    lo = n0
    hi = max(n0 + 1, n0 * 2)
    limit = 1 << 64
    while hi < limit and not meets(hi):
        lo = hi
        hi *= 2
    if hi >= limit and not meets(hi):
        return math.inf
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if meets(mid):
            hi = mid
        else:
            lo = mid
    return hi


class WilsonGoalPolicy:
    """Stopping policy for a #224 goal: Wilson interval, ``max(eps*p_hat, a)`` rule.

    ``epsilon`` is the current absolute half-width of the reported interval and
    ``target_epsilon`` the bound it must reach, so the canonical progress fields
    mean the same thing they mean for the Wald policy. With no goal the policy
    never converges and exists only to report the interval a fixed design or an
    unbounded run shows, which is the same one a goal run is evaluated on.

    Deleted in phase 2: the controller library evaluates this rule.
    """

    name = "wilson"

    def __init__(
        self,
        *,
        rel_error: float | None = None,
        abs_error: float | None = None,
    ) -> None:
        self._rel_error = rel_error
        self._abs_error = abs_error

    @property
    def rel_error(self) -> float | None:
        return self._rel_error

    @property
    def abs_error(self) -> float | None:
        return self._abs_error

    def evaluate(self, node: NodeState, ctx: RunContext) -> PolicyVerdict:
        state = evaluate_goal(
            node.num_one_bits,
            node.total_bits,
            ctx.confidence,
            self._rel_error,
            self._abs_error,
        )
        projected = projected_trials_for_goal(
            node.num_one_bits,
            node.total_bits,
            ctx.confidence,
            self._rel_error,
            self._abs_error,
        )
        ci95_low, ci95_high = reported_interval(
            node.num_one_bits, node.total_bits, DISPLAY_CONFIDENCE
        )
        ci99_low, ci99_high = reported_interval(node.num_one_bits, node.total_bits, DISPLAY_CONFIDENCE_99)
        # Canonical band fields on the same interval as the goal, so a client that
        # plots p01-p99 never mixes a Wilson centre with a Wald band.
        metrics: dict[str, Any] = {
            "ci_low": state.ci_low,
            "ci_high": state.ci_high,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "ci99_low": ci99_low,
            "ci99_high": ci99_high,
            "ci_level": ctx.confidence,
            "std_err": node.std_err,
            "p01": ci99_low,
            "p05": ci95_low,
            "p95": ci95_high,
            "p99": ci99_high,
            "abs_half_width": state.abs_half_width,
            "rel_half_width": state.rel_half_width,
            "target_half_width": state.target_half_width,
            "target_rel_half_width": state.target_rel_half_width,
            "target_abs_half_width": self._abs_error if self._abs_error is not None else None,
            "zero_hits": state.zero_hits,
            "zero_hit_upper_bound": state.zero_hit_upper_bound,
            "projected_trials": projected,
        }
        epsilon = state.abs_half_width if math.isfinite(state.abs_half_width) else 0.0
        target_epsilon = state.target_half_width if math.isfinite(state.target_half_width) else 0.0
        return PolicyVerdict(
            name=self.name,
            converged=state.converged,
            epsilon=epsilon,
            target_epsilon=target_epsilon,
            projected_trials=projected,
            metrics=metrics,
        )


__all__ = [
    "DISPLAY_CONFIDENCE",
    "GoalState",
    "WilsonGoalPolicy",
    "evaluate_goal",
    "goal_bound",
    "half_width",
    "projected_trials_for_goal",
    "relative_half_width",
    "reported_interval",
    "wilson_interval",
    "z_for_confidence",
    "zero_hit_upper_bound",
]
