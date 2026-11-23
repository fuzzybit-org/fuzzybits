"""Characterization: the exact ``LiveProgressUpdate`` wire shape.

The web client reads these frames as ``ConvergenceStatsPayload``
(``webui/src/app/lib/liveConvergenceStats.ts``). This module pins, key by key and
type by type, what a simulated run and an interrupted run actually send, so the
phase 2 swap onto the controller library is checked against the same frame.

Rules the client depends on:

* every frame is JSON-serialisable with no NaN/Infinity (browsers reject them);
* the canonical fields are always present, with ``None`` where a value does not
  exist yet (never a missing key, never a sentinel number);
* additive fields are optional and appear only on the paths that define them;
* an interrupted run's last frame carries the partial counts it reached.
"""

from __future__ import annotations

import json
import math

import pytest

from fuzzybits_server.convergence import (
    BernoulliSampler,
    ConvergenceConfig,
    ConvergenceRunner,
    StoppingSpec,
)

#: Canonical fields a client may read from any frame, whatever the request was.
CANONICAL_KEYS = frozenset(
    {
        "phase",
        "pass_index",
        "iterations",
        "trials",
        "trials_per_pass",
        "mean",
        "std_err",
        "p01",
        "p05",
        "p95",
        "p99",
        "epsilon",
        "target_epsilon",
        "epsilon_converged",
        "converged",
        "projected_trials",
        "projected_iterations",
        "remaining_trials",
        "remaining_iterations",
        "num_one_bits",
        "total_bits",
        "primary_policy",
        "stopping_policy",
    }
)

#: Optional fields: additive, present only when the request defines them.
OPTIONAL_KEYS = frozenset(
    {
        "stop_reason",
        "elapsed_s",
        "eta_seconds",
        "interval",
        "stop_reason_style",
        "target_rel_half_width",
        "target_trials",
        "nodes",
        "info_bits_last",
        "info_bits_total",
    }
)

#: Canonical fields a goal run adds: the #224 interval and goal quantities, on the
#: same keys the canonical Wald fields use where they mean the same thing. They are
#: documented in `tools/server/README.md` as the additive part of the frame.
INTERVAL_KEYS = frozenset(
    {
        "ci_low",
        "ci_high",
        "ci95_low",
        "ci95_high",
        "ci99_low",
        "ci99_high",
        "ci_level",
        "abs_half_width",
        "rel_half_width",
        "target_half_width",
        "target_abs_half_width",
        "zero_hits",
        "zero_hit_upper_bound",
    }
)

#: `oracle` diagnostics, only when the request sets one.
ORACLE_KEYS = frozenset(
    {
        "oracle_p",
        "bias",
        "abs_error",
        "rel_error",
        "mse",
        "log10_abs_error",
        "abs_log10_error",
        "z_score",
        "p_value",
        "ci95_covered",
        "ci99_covered",
        "n_required",
        "n_ratio",
    }
)


def _frames(options: dict, *, true_p: float = 0.2, seed: int = 7, **sampler_kwargs) -> list[dict]:
    sampler = BernoulliSampler(true_p=true_p, trials_per_pass=4096, seed=seed, **sampler_kwargs)
    # Emit every pass, so the characterization is of the frame, not of the rate limit.
    config = ConvergenceConfig.from_options({"emit_min_interval_s": 0.0, **options})
    frames: list[dict] = []
    ConvergenceRunner(sampler, config, on_progress=frames.append).run()
    return frames


def _json_round_trip(frame: dict) -> dict:
    """What the client actually receives: the frame after a strict JSON round trip."""
    return json.loads(json.dumps(frame, allow_nan=False))


def _assert_no_non_finite(frame: dict) -> None:
    def walk(value):
        if isinstance(value, float):
            assert math.isfinite(value), f"non-finite value {value!r} would be rejected by the browser"
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(frame)


# --------------------------------------------------------------------------- #
#  Simulated run
# --------------------------------------------------------------------------- #
def test_every_frame_carries_the_canonical_fields() -> None:
    for frame in _frames({"num_trials": 4096 * 3, "early_stop": False}):
        missing = CANONICAL_KEYS - frame.keys()
        assert not missing, f"missing canonical fields: {sorted(missing)}"


#: Legacy canonical keys the Wald policy fills: the log10 band it also tracks.
WALD_LOG_KEYS = frozenset(
    {"epsilon_log10", "target_epsilon_log10", "epsilon_log10_converged", "projected_trials_log10"}
)


def test_no_frame_carries_an_unexpected_field() -> None:
    for frame in _frames({"num_trials": 4096 * 3, "early_stop": False}):
        extra = set(frame) - CANONICAL_KEYS - OPTIONAL_KEYS - ORACLE_KEYS - WALD_LOG_KEYS
        # Namespaced policy streams (`wald_*`, `gbas_*`, `wilson_*`) are declared by
        # their policy; assert the shape rather than enumerate them.
        unexpected = {key for key in extra if key.split("_")[0] not in {"wald", "gbas", "wilson"}}
        assert not unexpected, f"undeclared fields: {sorted(unexpected)}"


def test_field_types_are_pinned() -> None:
    frame = _frames({"num_trials": 4096 * 2, "early_stop": False})[-1]
    for key, python_type in (
        ("phase", str),
        ("pass_index", int),
        ("iterations", int),
        ("trials", int),
        ("trials_per_pass", int),
        ("mean", float),
        ("std_err", float),
        ("epsilon", float),
        ("target_epsilon", float),
        ("num_one_bits", int),
        ("total_bits", int),
        ("primary_policy", str),
        ("stopping_policy", str),
    ):
        assert isinstance(frame[key], python_type), f"{key} is {type(frame[key])}"
    for key in ("epsilon_converged", "converged"):
        assert isinstance(frame[key], bool), f"{key} is {type(frame[key])}"


def test_nullability_is_reserved_for_values_that_do_not_exist_yet() -> None:
    frames = _frames({"num_trials": 4096 * 2, "early_stop": False})
    first = frames[0]
    # A frame is only emitted after a pass commits, so the estimate already exists.
    assert first["mean"] is not None
    assert first["trials"] == 4096
    # The projection is a number from the first frame on; it is not "absent".
    assert isinstance(first["projected_trials"], (int, float))
    assert first["remaining_trials"] >= 0
    # A value that genuinely does not exist is null, never a sentinel.
    for frame in frames:
        assert frame["info_bits_last"] is None or isinstance(frame["info_bits_last"], float)


def test_frames_are_strict_json() -> None:
    for frame in _frames({"num_trials": 4096 * 3, "early_stop": False}):
        _assert_no_non_finite(frame)
        assert _json_round_trip(frame) == frame


def test_frames_are_monotone_in_trials() -> None:
    frames = _frames({"num_trials": 4096 * 4, "early_stop": False})
    totals = [frame["trials"] for frame in frames]
    assert totals == sorted(totals)
    assert totals[0] == 4096
    assert totals[-1] == 4096 * 4


def test_phase_is_sampling_without_burn_in() -> None:
    for frame in _frames({"num_trials": 4096 * 2, "early_stop": False}):
        assert frame["phase"] == "sampling"


def test_burn_in_phase_is_reported_as_its_own_phase() -> None:
    """Phase is decided after each pass commits, so the frame that reaches burn-in is sampling."""
    frames = _frames({"num_trials": 4096 * 3, "burn_in": 4096 * 2})
    assert [frame["phase"] for frame in frames] == ["burn_in", "sampling", "sampling"]
    assert frames[0]["trials"] == 4096
    assert frames[1]["trials"] == 4096 * 2


def test_info_gain_is_optional_and_absent_on_the_first_frame() -> None:
    frame = _frames({"num_trials": 4096, "early_stop": False})[-1]
    assert frame["info_bits_last"] is None
    assert frame["info_bits_total"] is not None


def test_oracle_keys_appear_only_when_requested() -> None:
    without = _frames({"num_trials": 4096, "early_stop": False})[-1]
    assert not (ORACLE_KEYS & without.keys())

    with_oracle = _frames({"num_trials": 4096, "early_stop": False, "oracle": 0.2})[-1]
    assert ORACLE_KEYS <= with_oracle.keys()


def test_nodes_appear_only_when_streaming_is_requested() -> None:
    sampler = BernoulliSampler(
        true_p={1: 0.2, 2: 0.1}, trials_per_pass=4096, target_index=1, seed=7
    )
    config = ConvergenceConfig.from_options({"num_trials": 4096, "stream_nodes": False})
    plain: list[dict] = []
    ConvergenceRunner(sampler, config, on_progress=plain.append).run()
    assert "nodes" not in plain[-1]

    sampler = BernoulliSampler(
        true_p={1: 0.2, 2: 0.1}, trials_per_pass=4096, target_index=1, seed=7
    )
    config = ConvergenceConfig.from_options({"num_trials": 4096, "stream_nodes": True})
    streamed: list[dict] = []
    ConvergenceRunner(sampler, config, on_progress=streamed.append).run()
    assert set(streamed[-1]["nodes"]) == {"1", "2"}
    assert set(streamed[-1]["nodes"]["1"]) == {"mean", "trials"}


def test_stop_reason_only_on_the_last_frame() -> None:
    frames = _frames({"num_trials": 4096 * 3, "early_stop": False})
    assert all("stop_reason" not in frame for frame in frames[:-1])
    assert frames[-1]["stop_reason"] == "max_trials"


# --------------------------------------------------------------------------- #
#  Interrupted run
# --------------------------------------------------------------------------- #
class _InterruptingSampler(BernoulliSampler):
    """Raises from ``step`` after ``passes`` passes, as a terminated run does."""

    def __init__(self, *, after: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._after = after
        self._calls = 0

    def step(self, passes: int = 1):
        if self._calls >= self._after:
            raise KeyboardInterrupt("stopped by the supervisor")
        self._calls += 1
        return super().step(passes)


def test_interrupted_run_keeps_the_last_frame_it_reached() -> None:
    sampler = _InterruptingSampler(after=2, true_p=0.2, trials_per_pass=4096, seed=7)
    config = ConvergenceConfig.from_options(
        {"num_trials": 4096 * 10, "early_stop": False, "emit_min_interval_s": 0.0}
    )
    frames: list[dict] = []
    with pytest.raises(KeyboardInterrupt):
        ConvergenceRunner(sampler, config, on_progress=frames.append).run()

    assert len(frames) == 2
    assert frames[-1]["trials"] == 4096 * 2
    assert "stop_reason" not in frames[-1]
    # The frame is still complete and strictly JSON despite the partial run.
    _assert_no_non_finite(frames[-1])
    assert _json_round_trip(frames[-1]) == frames[-1]


def test_interrupted_controller_run_reports_the_partial_counts() -> None:
    sampler = _InterruptingSampler(after=3, true_p=0.2, trials_per_pass=4096, seed=7)
    spec = StoppingSpec.model_validate({"rel_error": 1e-9, "max_time": "none"}).resolve()
    config = ConvergenceConfig.from_stopping(spec, {"emit_min_interval_s": 0.0})
    frames: list[dict] = []
    with pytest.raises(KeyboardInterrupt):
        ConvergenceRunner(sampler, config, on_progress=frames.append).run()
    frame = frames[-1]
    assert frame["trials"] == 4096 * 3
    assert frame["total_bits"] == frame["trials"]
    assert frame["interval"] == "wilson"
    assert "stop_reason" not in frame  # the loop never reached a stop condition


# --------------------------------------------------------------------------- #
#  Additive fields: the version-bump policy
# --------------------------------------------------------------------------- #
def test_controller_frames_add_only_optional_fields() -> None:
    """A goal run must not remove or rename anything the client already reads."""
    legacy = _frames({"num_trials": 4096 * 3, "early_stop": False})[-1]
    sampler = BernoulliSampler(true_p=0.2, trials_per_pass=4096, seed=7)
    config = ConvergenceConfig.from_stopping(StoppingSpec.model_validate({"rel_error": 0.05}).resolve())
    controller: list[dict] = []
    ConvergenceRunner(sampler, config, on_progress=controller.append).run()
    goal = controller[-1]

    added = set(goal) - set(legacy)
    documented = OPTIONAL_KEYS | INTERVAL_KEYS
    undeclared = {key for key in added if key not in documented and key.split("_")[0] != "wilson"}
    assert not undeclared, f"undocumented additions: {sorted(undeclared)}"
    removed = CANONICAL_KEYS - set(goal)
    assert not removed, f"canonical fields dropped: {sorted(removed)}"
    # Types of the canonical fields are unchanged.
    for key in CANONICAL_KEYS & set(goal):
        if key in {"mean", "std_err", "epsilon", "target_epsilon", "p01", "p05", "p95", "p99"}:
            assert goal[key] is None or isinstance(goal[key], float)
        elif key in {"converged", "epsilon_converged"}:
            assert isinstance(goal[key], bool)
        elif key in {"projected_trials", "projected_iterations", "remaining_trials", "remaining_iterations"}:
            assert isinstance(goal[key], (int, float))
        elif key in {"phase", "primary_policy", "stopping_policy"}:
            assert isinstance(goal[key], str)
        else:
            assert isinstance(goal[key], int)


def test_schema_version_of_the_frame_is_not_advertised() -> None:
    """The frame has never carried a version field, and adding one is a breaking change."""
    frame = _frames({"num_trials": 4096, "early_stop": False})[-1]
    assert "schema" not in frame
    assert "version" not in frame
