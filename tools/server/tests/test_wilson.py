"""The #224 interval and goal rule: Wilson, the zero-hit bound, and the goal test.

The interval is checked against scipy's independent ``beta``/``norm`` computation
of the same closed form, and the goal rule against the definition in the #224
comment (``(U - L)/2 <= max(eps*p_hat, a)``). There is no sampling here, so
every test is exact.
"""

from __future__ import annotations

import math

import pytest
from scipy import stats

from fuzzybits_server.convergence.tally import NodeCounters, NodeState, RunContext
from fuzzybits_server.convergence.wilson import (
    DISPLAY_CONFIDENCE,
    DISPLAY_CONFIDENCE_99,
    WilsonGoalPolicy,
    evaluate_goal,
    goal_bound,
    half_width,
    projected_trials_for_goal,
    relative_half_width,
    reported_interval,
    wilson_interval,
    z_for_confidence,
    zero_hit_upper_bound,
)


def _node(hits: int, trials: int) -> NodeState:
    node = NodeState(index=1)
    node.update(NodeCounters(1, hits, trials), wall_time=0.0)
    return node


def _ctx(confidence: float = 0.95) -> RunContext:
    return RunContext(
        confidence=confidence,
        z=z_for_confidence(confidence),
        delta=1e-3,
        trials_per_pass=4096,
        target_index=1,
    )


# --------------------------------------------------------------------------- #
#  Wilson interval
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("hits", "trials", "confidence"),
    [
        (0, 1, 0.95),
        (1, 1, 0.95),
        (10, 100, 0.95),
        (500, 1000, 0.95),
        (500, 1000, 0.99),
        (3, 1_000_000, 0.95),
        (999_999, 1_000_000, 0.95),
        (38415, 3_800_000, 0.95),
        (1, 10**12, 0.999),
    ],
)
def test_wilson_matches_the_closed_form_on_the_scipy_quantile(hits, trials, confidence) -> None:
    """Same formula, with z from scipy rather than the Acklam approximation."""
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = hits / trials
    n = trials
    z2 = z * z
    centre = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    offset = (z / (1.0 + z2 / n)) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    expected = (centre - offset, centre + offset)

    low, high = wilson_interval(hits, trials, confidence)
    # The host z comes from an Acklam rational approximation (~1e-9 relative),
    # so the two agree to that, not to round-off.
    assert low == pytest.approx(expected[0], rel=1e-8, abs=1e-12)
    assert high == pytest.approx(expected[1], rel=1e-8, abs=1e-12)


def test_wilson_contains_the_point_estimate_and_stays_in_the_unit_interval() -> None:
    low, high = wilson_interval(2, 1000, 0.95)
    assert 0.0 <= low < 0.002 < high <= 1.0


def test_wilson_has_non_zero_width_at_zero_hits() -> None:
    low, high = wilson_interval(0, 1000, 0.95)
    assert low == 0.0
    assert high > 0.0


def test_wilson_is_undefined_before_the_first_pass() -> None:
    assert wilson_interval(0, 0, 0.95) == (math.nan, math.nan)


def test_wilson_narrows_as_trials_grow_at_fixed_proportion() -> None:
    widths = [half_width(wilson_interval(n // 10, n, 0.95)) for n in (10**4, 10**6, 10**8)]
    assert widths[0] > widths[1] > widths[2]
    # Half-width scales as 1/sqrt(N): a 100x budget is a 10x narrowing.
    assert widths[0] / widths[1] == pytest.approx(10.0, rel=0.02)
    assert widths[1] / widths[2] == pytest.approx(10.0, rel=0.02)


# --------------------------------------------------------------------------- #
#  Zero hits (#224 definition 4)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("trials", [1, 10, 1000, 10**7, 10**12])
def test_zero_hit_bound_is_the_clopper_pearson_upper_limit(trials: int) -> None:
    """``1 - (alpha/2)^(1/N)`` equals the two-sided CP upper limit at level c."""
    ours = zero_hit_upper_bound(trials, 0.95)
    # CP upper limit for x = 0 successes: BetaInv(1 - alpha/2; 0 + 1, N).
    expected = float(stats.beta.ppf(1.0 - 0.05 / 2.0, 1, trials))
    assert ours == pytest.approx(expected, rel=1e-12)


def test_zero_hit_bound_tends_to_3_689_over_n_at_95_percent() -> None:
    # 1 - (alpha/2)^(1/N) = -expm1(ln(alpha/2)/N) -> -ln(0.025)/N = 3.6889/N.
    assert -math.log(0.05 / 2.0) == pytest.approx(3.6888794541139363, rel=1e-12)
    # -expm1(x) = -x - x^2/2 - ... with x = ln(alpha/2)/N < 0, so N*bound
    # approaches -ln(alpha/2) from below with error O(1/N).
    constant = -math.log((1.0 - 0.95) / 2.0)
    assert constant == pytest.approx(3.6888794541139363, rel=1e-12)
    products = [trials * zero_hit_upper_bound(trials, 0.95) for trials in (10**6, 10**9, 10**12)]
    assert all(value < constant for value in products)
    assert products[0] < products[1] < products[2]
    assert products[0] == pytest.approx(constant, rel=2e-6)
    assert products[1] == pytest.approx(constant, rel=2e-9)
    assert products[2] == pytest.approx(constant, rel=2e-12)


def test_zero_hit_bound_at_99_percent_is_wider() -> None:
    assert zero_hit_upper_bound(1000, 0.99) > zero_hit_upper_bound(1000, 0.95)


def test_zero_hit_bound_is_undefined_without_trials() -> None:
    assert math.isnan(zero_hit_upper_bound(0, 0.95))


def test_reported_interval_switches_at_the_first_hit() -> None:
    assert reported_interval(0, 1000, 0.95) == (0.0, zero_hit_upper_bound(1000, 0.95))
    assert reported_interval(1, 1000, 0.95) == wilson_interval(1, 1000, 0.95)


# --------------------------------------------------------------------------- #
#  Goal rule (#224 definitions 2 and 3)
# --------------------------------------------------------------------------- #
def test_relative_half_width_is_undefined_at_zero_hits() -> None:
    low, high = reported_interval(0, 1000, 0.95)
    assert math.isnan(relative_half_width((low, high), 0.0))


def test_relative_half_width_at_the_point_estimate() -> None:
    interval = wilson_interval(100, 1000, 0.95)
    assert relative_half_width(interval, 0.1) == pytest.approx(half_width(interval) / 0.1, rel=1e-12)


def test_goal_bound_takes_the_maximum_of_the_two_errors() -> None:
    assert goal_bound(0.5, 0.01, None) == pytest.approx(0.005)
    assert goal_bound(0.5, None, 0.01) == 0.01
    # abs_error is a floor: eps*p_hat is larger here, so it wins.
    assert goal_bound(0.5, 0.01, 1e-6) == pytest.approx(0.005)
    # ... and here it is the floor that wins.
    assert goal_bound(0.001, 0.01, 1e-5) == pytest.approx(1e-5)
    assert math.isnan(goal_bound(0.5, None, None))


def test_relative_goal_is_met_once_the_wilson_half_width_is_inside_the_bound() -> None:
    # +/-1% at 95% needs about 38,400 hits whatever p is (#224: no burn-in needed).
    state = evaluate_goal(hits=38_415, trials=3_841_500, confidence=0.95, rel_error=0.01, abs_error=None)
    assert state.abs_half_width <= 0.01 * state.mean
    assert state.converged is True
    assert state.rel_half_width == pytest.approx(0.01, rel=0.01)

    tighter = evaluate_goal(hits=38_415, trials=3_841_500, confidence=0.95, rel_error=0.001, abs_error=None)
    assert tighter.converged is False


def test_a_relative_goal_cannot_be_met_at_zero_hits() -> None:
    state = evaluate_goal(hits=0, trials=10**9, confidence=0.95, rel_error=0.01, abs_error=None)
    assert state.zero_hits is True
    assert state.converged is False
    # Even with an absurd budget the bound is eps * 0 = 0 < the interval's width.
    assert state.zero_hit_upper_bound > 0.0


def test_an_absolute_goal_can_be_met_at_zero_hits() -> None:
    trials = 4_000_000  # zero-hit bound 3.689/4e6 ~ 9.2e-7
    state = evaluate_goal(hits=0, trials=trials, confidence=0.95, rel_error=None, abs_error=1e-6)
    assert state.converged is True
    state = evaluate_goal(hits=0, trials=trials, confidence=0.95, rel_error=None, abs_error=1e-7)
    assert state.converged is False


def test_abs_error_floor_lets_a_tiny_probability_converge() -> None:
    # p_hat = 1e-6: the relative bound would be 1e-8, the floor 1e-5.
    hits, trials = 10, 10_000_000
    without_floor = evaluate_goal(hits, trials, 0.95, rel_error=0.01, abs_error=None)
    with_floor = evaluate_goal(hits, trials, 0.95, rel_error=0.01, abs_error=1e-5)
    assert without_floor.converged is False
    assert with_floor.converged is True


def test_no_goal_never_converges_but_still_reports_the_interval() -> None:
    state = evaluate_goal(hits=500, trials=1000, confidence=0.95, rel_error=None, abs_error=None)
    assert state.converged is False
    assert state.ci_low < state.mean < state.ci_high
    assert math.isnan(state.target_half_width)


def test_confidence_widens_the_interval_and_delays_the_goal() -> None:
    # The relative half-width at 1e6 trials with 1000 hits is ~0.062 at 95% and
    # ~0.082 at 99% (z 1.96 vs 2.576), so a 7% goal separates the two levels.
    at_95 = evaluate_goal(1000, 1_000_000, 0.95, 0.07, None)
    at_99 = evaluate_goal(1000, 1_000_000, 0.99, 0.07, None)
    assert at_99.abs_half_width > at_95.abs_half_width
    assert at_99.rel_half_width > at_95.rel_half_width
    assert at_95.converged is True
    assert at_99.converged is False


# --------------------------------------------------------------------------- #
#  Projection
# --------------------------------------------------------------------------- #
def test_projection_is_the_smallest_trial_count_that_meets_the_goal() -> None:
    """The projection solves the goal inequality with p_hat held at its current value."""
    hits, trials, rel_error = 1000, 1_000_000, 0.005
    p_hat = hits / trials
    target = p_hat * rel_error
    projected = projected_trials_for_goal(hits, trials, 0.95, rel_error, None)
    assert isinstance(projected, int)
    assert projected > trials

    def half_width_at(n: int) -> float:
        return half_width(reported_interval(int(round(p_hat * n)), n, 0.95))

    assert half_width_at(projected) <= target
    assert half_width_at(projected - 1) > target


def test_projection_returns_the_current_count_when_already_converged() -> None:
    # 100k hits is a ~0.62% relative half-width, comfortably inside +/-1%.
    assert projected_trials_for_goal(100_000, 10_000_000, 0.95, 0.01, None) == 10_000_000


def test_projection_is_infinite_for_a_relative_goal_at_zero_hits() -> None:
    assert projected_trials_for_goal(0, 1000, 0.95, 0.01, None) == math.inf


def test_projection_finishes_for_an_absolute_goal_at_zero_hits() -> None:
    projected = projected_trials_for_goal(0, 1000, 0.95, None, 1e-3)
    assert isinstance(projected, int)
    # 3.689/N <= 2e-3  =>  N >= 1845
    assert 1800 <= projected <= 1900


def test_no_goal_projects_nothing() -> None:
    assert projected_trials_for_goal(10, 1000, 0.95, None, None) == 0


# --------------------------------------------------------------------------- #
#  The policy wrapper
# --------------------------------------------------------------------------- #
def test_policy_reports_the_interval_it_is_evaluated_on() -> None:
    verdict = WilsonGoalPolicy(rel_error=0.01).evaluate(_node(1000, 1_000_000), _ctx())
    state = evaluate_goal(1000, 1_000_000, 0.95, 0.01, None)
    assert verdict.name == "wilson"
    assert verdict.epsilon == pytest.approx(state.abs_half_width)
    assert verdict.target_epsilon == pytest.approx(state.target_half_width)
    assert verdict.metrics["ci_low"] == pytest.approx(state.ci_low)
    assert verdict.metrics["ci_high"] == pytest.approx(state.ci_high)
    assert verdict.metrics["target_rel_half_width"] == 0.01


def test_policy_bands_are_wilson_not_wald() -> None:
    verdict = WilsonGoalPolicy(rel_error=0.01).evaluate(_node(1000, 1_000_000), _ctx())
    low95, high95 = reported_interval(1000, 1_000_000, DISPLAY_CONFIDENCE)
    low99, high99 = reported_interval(1000, 1_000_000, DISPLAY_CONFIDENCE_99)
    assert verdict.metrics["p05"] == pytest.approx(low95)
    assert verdict.metrics["p95"] == pytest.approx(high95)
    assert verdict.metrics["p01"] == pytest.approx(low99)
    assert verdict.metrics["p99"] == pytest.approx(high99)
    # The Wald band at this p_hat would be narrower at the low end; Wilson is shifted out.
    wald_low = 0.001 - 1.959963984540054 * math.sqrt(0.001 * 0.999 / 1_000_000)
    assert verdict.metrics["p05"] > wald_low


def test_policy_without_a_goal_only_reports() -> None:
    verdict = WilsonGoalPolicy().evaluate(_node(1000, 1_000_000), _ctx())
    assert verdict.converged is False
    assert verdict.metrics["target_rel_half_width"] is None or math.isnan(
        verdict.metrics["target_rel_half_width"]
    )
