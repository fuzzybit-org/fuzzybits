"""Characterization: the stop reasons and progress frames a run produces.

Two rules are pinned here.

* A legacy request keeps the loop it had before #225: the same stop-reason
  strings (``converged``, ``max_trials``, ``max_passes``) and the same fields.
* A #224 stopping specification uses the controller's vocabulary
  (``goal``, ``max-trials``, ``max-time``, ``trials``) and adds only optional
  fields (``interval``, ``target_trials``, ``target_rel_half_width``,
  ``stop_reason_style``, ``eta_seconds``, ``elapsed_s``).

The sampler is the deterministic ``BernoulliSampler``, so every count and every
stop point is reproducible; with ``pass_delay_s`` the time limit is exercised
without depending on device speed.
"""

from __future__ import annotations

import pytest

from fuzzybits_server.convergence import (
    STOP_GOAL,
    STOP_MAX_TIME,
    STOP_MAX_TRIALS,
    STOP_TRIALS,
    BernoulliSampler,
    ConvergenceConfig,
    ConvergenceRunner,
    StoppingSpec,
)


def _run(options: dict, **sampler_kwargs) -> tuple[object, list[dict]]:
    sampler = BernoulliSampler(**{"true_p": 0.2, "trials_per_pass": 4096, "seed": 7, **sampler_kwargs})
    config = ConvergenceConfig.from_options(options)
    frames: list[dict] = []
    result = ConvergenceRunner(sampler, config, on_progress=frames.append).run()
    return result, frames


def _run_spec(fields: dict, options: dict | None = None, **sampler_kwargs) -> tuple[object, list[dict]]:
    sampler = BernoulliSampler(**{"true_p": 0.2, "trials_per_pass": 4096, "seed": 7, **sampler_kwargs})
    spec = StoppingSpec.model_validate(fields).resolve()
    config = ConvergenceConfig.from_stopping(spec, options)
    frames: list[dict] = []
    result = ConvergenceRunner(sampler, config, on_progress=frames.append).run()
    return result, frames


# --------------------------------------------------------------------------- #
#  Legacy loop: unchanged stop reasons and fields
# --------------------------------------------------------------------------- #
def test_legacy_early_stop_reports_converged() -> None:
    result, _ = _run({"delta": 0.05, "confidence": 0.95, "early_stop": True, "policy": "wald"})
    assert result.stop_reason == "converged"
    assert result.converged is True


def test_legacy_fixed_run_reports_max_trials() -> None:
    result, _ = _run({"num_trials": 4096 * 3, "early_stop": False})
    assert result.stop_reason == "max_trials"
    assert result.state.target.total_bits == 4096 * 3


def test_legacy_max_passes_is_still_its_own_reason() -> None:
    result, _ = _run({"max_passes": 3})
    assert result.stop_reason == "max_passes"
    assert result.state.pass_count == 3


def test_legacy_frame_has_no_controller_fields() -> None:
    _, frames = _run({"num_trials": 4096 * 2, "early_stop": False})
    for frame in frames:
        assert "stop_reason_style" not in frame
        assert "interval" not in frame
        assert "target_trials" not in frame


def test_legacy_frame_field_order_is_stable() -> None:
    """Order must not change between passes; only ``stop_reason`` may be added."""
    _, frames = _run({"num_trials": 4096 * 3, "early_stop": False})
    first = list(frames[0].keys())
    assert "stop_reason" not in first
    for frame in frames[1:-1]:
        assert list(frame.keys()) == first
    assert list(frames[-1].keys()) == [*first, "stop_reason"]


# --------------------------------------------------------------------------- #
#  #224 stop reasons
# --------------------------------------------------------------------------- #
def test_fixed_design_runs_exactly_n_and_reports_trials() -> None:
    result, frames = _run_spec({"trials": 4096 * 3})
    assert result.stop_reason == STOP_TRIALS == "trials"
    assert result.state.target.total_bits == 4096 * 3
    assert frames[-1]["stop_reason"] == "trials"


def test_fixed_design_ignores_the_goal() -> None:
    result, _ = _run_spec({"trials": 4096 * 3})
    assert result.converged is False


def test_goal_stops_where_the_goal_is_met_and_reports_goal() -> None:
    # p = 0.2: +/-5% at 95% needs ~1537 hits, so ~7685 trials (2 passes of 4096).
    result, frames = _run_spec({"rel_error": 0.05, "confidence": 0.95})
    assert result.stop_reason == STOP_GOAL == "goal"
    assert result.converged is True
    final = frames[-1]
    assert final["stop_reason"] == "goal"
    assert final["converged"] is True
    assert final["wilson_converged"] is True
    assert final["interval"] == "wilson"
    assert final["epsilon"] <= final["target_epsilon"]


def test_goal_stops_no_earlier_than_the_goal_is_met() -> None:
    _, frames = _run_spec({"rel_error": 0.05, "confidence": 0.95})
    for frame in frames[:-1]:
        assert frame["wilson_converged"] is False


def test_trial_limit_reports_max_trials() -> None:
    # A goal that needs far more than the limit, so the limit fires first.
    result, frames = _run_spec({"rel_error": 1e-4, "max_trials": 4096 * 2})
    assert result.stop_reason == STOP_MAX_TRIALS == "max-trials"
    assert result.converged is False
    assert frames[-1]["stop_reason"] == "max-trials"
    assert result.state.target.total_bits >= 4096 * 2


def test_goal_is_checked_before_the_trial_limit() -> None:
    """A run that converges at the same pass as its limit reports the goal."""
    baseline, _ = _run_spec({"rel_error": 0.05, "confidence": 0.95})
    needed = baseline.state.target.total_bits
    result, _ = _run_spec({"rel_error": 0.05, "confidence": 0.95, "max_trials": needed})
    assert result.stop_reason == "goal"


def test_time_limit_reports_max_time() -> None:
    result, frames = _run_spec({"rel_error": 1e-9, "max_time": "0.001s"}, pass_delay_s=0.01)
    assert result.stop_reason == STOP_MAX_TIME == "max-time"
    assert frames[-1]["stop_reason"] == "max-time"
    assert result.converged is False


def test_no_time_limit_runs_to_the_goal() -> None:
    result, _ = _run_spec({"rel_error": 0.05, "max_time": "none"})
    assert result.stop_reason == "goal"


def test_fixed_design_with_a_time_limit_stops_on_either() -> None:
    slow, _ = _run_spec({"trials": 4096 * 100, "max_time": "0.001s"}, pass_delay_s=0.01)
    assert slow.stop_reason == "max-time"

    fast, _ = _run_spec({"trials": 4096 * 2, "max_time": "1h"})
    assert fast.stop_reason == "trials"


# --------------------------------------------------------------------------- #
#  Controller frames
# --------------------------------------------------------------------------- #
def test_controller_frame_carries_the_interval_and_the_target() -> None:
    _, frames = _run_spec({"rel_error": 0.05, "confidence": 0.95, "max_trials": 4096 * 4})
    frame = frames[0]
    assert frame["interval"] == "wilson"
    assert frame["stop_reason_style"] == "controller"
    assert frame["target_rel_half_width"] == 0.05
    assert frame["target_trials"] == 4096 * 4
    assert frame["primary_policy"] == "wilson"
    assert frame["stopping_policy"] == "wilson"


def test_controller_frame_bands_are_the_wilson_interval() -> None:
    from fuzzybits_server.convergence.wilson import reported_interval

    _, frames = _run_spec({"rel_error": 0.05})
    frame = frames[-1]
    low95, high95 = reported_interval(frame["num_one_bits"], frame["total_bits"], 0.95)
    assert frame["p05"] == pytest.approx(low95)
    assert frame["p95"] == pytest.approx(high95)
    assert frame["ci_low"] == pytest.approx(frame["wilson_ci_low"])
    assert frame["ci_high"] == pytest.approx(frame["wilson_ci_high"])


def test_controller_frame_reports_the_eta_and_the_elapsed_time() -> None:
    _, frames = _run_spec({"rel_error": 0.05})
    assert all("elapsed_s" in frame for frame in frames)
    assert frames[-1].get("eta_seconds") is None or frames[-1]["eta_seconds"] >= 0.0


def test_controller_frame_reports_the_zero_hit_upper_bound() -> None:
    _, frames = _run_spec({"abs_error": 1e-3}, true_p=0.0)
    frame = frames[-1]
    assert frame["wilson_zero_hits"] is True
    assert frame["wilson_zero_hit_upper_bound"] > 0.0
    assert frame["ci_low"] == pytest.approx(0.0)


def test_zero_hits_meet_an_absolute_goal_but_not_a_relative_one() -> None:
    absolute, _ = _run_spec({"abs_error": 0.01}, true_p=0.0)
    assert absolute.stop_reason == "goal"

    relative, frames = _run_spec({"rel_error": 0.01, "max_trials": 4096 * 3}, true_p=0.0)
    assert relative.stop_reason == "max-trials"
    assert relative.converged is False
    assert frames[-1]["wilson_rel_half_width"] is None  # undefined at p_hat = 0


def test_summary_carries_the_stopping_block_only_via_run_calculation() -> None:
    sampler = BernoulliSampler(true_p=0.2, trials_per_pass=4096, seed=7)
    spec = StoppingSpec.model_validate({"trials": 4096}).resolve()
    config = ConvergenceConfig.from_stopping(spec)
    summary = ConvergenceRunner(sampler, config).run().summary
    assert summary["stop_reason"] == "trials"
    assert summary["trials"] == 4096
    assert "stopping" not in summary
