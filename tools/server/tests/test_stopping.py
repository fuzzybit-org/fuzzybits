"""The #224 stopping specification: defaults, validation rules, and resolution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fuzzybits_server.convergence.stopping import (
    DEFAULT_CONFIDENCE,
    DEFAULT_GOAL_MAX_TIME_S,
    DEFAULT_REL_ERROR,
    LegacyTranslationError,
    StoppingSpec,
    StoppingSpecError,
    UnsupportedSpecError,
    check_executable,
    format_duration,
    parse_duration,
    sampler_budget,
    translate_legacy,
)


def _spec(**fields) -> StoppingSpec:
    return StoppingSpec.model_validate(fields)


# --------------------------------------------------------------------------- #
#  Resolution: the #224 defaults
# --------------------------------------------------------------------------- #
def test_empty_spec_is_a_goal_run_with_the_documented_defaults() -> None:
    resolved = _spec().resolve()
    assert resolved.mode == "goal"
    assert resolved.rel_error == DEFAULT_REL_ERROR == 0.01
    assert resolved.abs_error is None
    assert resolved.confidence == DEFAULT_CONFIDENCE == 0.95
    assert resolved.max_trials is None
    assert resolved.max_time_s == DEFAULT_GOAL_MAX_TIME_S == 600.0
    assert resolved.fixed_trials is None
    assert resolved.goal_queries == ()
    assert resolved.require_goal is False


def test_fixed_design_has_no_goal_and_no_time_limit() -> None:
    resolved = _spec(trials=1_000_000).resolve()
    assert resolved.mode == "fixed"
    assert resolved.fixed_trials == 1_000_000
    assert resolved.rel_error is None and resolved.abs_error is None
    assert resolved.max_time_s is None


def test_fixed_design_keeps_an_explicit_time_limit() -> None:
    resolved = _spec(trials=1_000_000, max_time="5m").resolve()
    assert resolved.mode == "fixed"
    assert resolved.max_time_s == 300.0


def test_abs_error_alone_does_not_add_a_relative_goal() -> None:
    resolved = _spec(abs_error=1e-7).resolve()
    assert resolved.rel_error is None
    assert resolved.abs_error == 1e-7


def test_both_errors_are_kept_and_the_goal_takes_the_max() -> None:
    resolved = _spec(rel_error=0.02, abs_error=1e-6).resolve()
    assert resolved.rel_error == 0.02
    assert resolved.abs_error == 1e-6


def test_explicit_max_time_none_keeps_the_goal_without_a_limit() -> None:
    resolved = _spec(max_time="none").resolve()
    assert resolved.mode == "goal"
    assert resolved.max_time_s is None
    assert resolved.rel_error == DEFAULT_REL_ERROR


def test_explicit_null_max_time_is_no_limit() -> None:
    """In a JSON body ``"max_time": null`` reads as "no limit", the same as ``"none"``."""
    resolved = _spec(max_time=None).resolve()
    assert resolved.mode == "goal"
    assert resolved.max_time_s is None
    assert resolved.rel_error == DEFAULT_REL_ERROR


def test_a_specification_always_selects_a_goal_or_a_fixed_design() -> None:
    for fields in ({}, {"max_time": "none"}, {"abs_error": 1e-7}, {"max_trials": 10}):
        assert _spec(**fields).resolve().mode in {"goal", "fixed"}


def test_trial_limit_without_a_goal_is_a_goal_run_with_the_default_goal() -> None:
    resolved = _spec(max_trials=5_000_000).resolve()
    assert resolved.mode == "goal"
    assert resolved.max_trials == 5_000_000
    assert resolved.rel_error == DEFAULT_REL_ERROR


def test_confidence_applies_to_a_fixed_design_too() -> None:
    assert _spec(trials=10, confidence=0.99).resolve().confidence == 0.99


# --------------------------------------------------------------------------- #
#  Validation: message for message
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"rel_error": 0.0}, "rel_error must be greater than 0 and less than 1"),
        ({"rel_error": 1.0}, "rel_error must be greater than 0 and less than 1"),
        ({"rel_error": -0.01}, "rel_error must be greater than 0 and less than 1"),
        ({"abs_error": 0.0}, "abs_error must be greater than 0 and at most 1"),
        ({"abs_error": 1.5}, "abs_error must be greater than 0 and at most 1"),
        ({"confidence": 0.0}, "confidence must be greater than 0 and less than 1"),
        ({"confidence": 1.0}, "confidence must be greater than 0 and less than 1"),
        ({"max_trials": 0}, "max_trials must be at least 1"),
        ({"trials": 0}, "trials must be at least 1"),
        ({"trials": 1, "rel_error": 0.01}, "trials (a fixed design) cannot be combined with rel_error or abs_error"),
        ({"trials": 1, "abs_error": 0.01}, "trials (a fixed design) cannot be combined with rel_error or abs_error"),
        ({"trials": 1, "goal_queries": ["TOP"]}, "goal_queries does not apply"),
        ({"trials": 1, "require_goal": True}, "require_goal does not apply"),
        ({"goal_queries": []}, "goal_queries must name at least one query"),
        ({"max_time": "-1s"}, "max_time must be positive"),
        ({"max_time": 0}, "max_time must be positive"),
        ({"max_time": True}, "max_time must not be a boolean"),
        ({"max_time": "soon"}, "max_time 'soon' is not a duration"),
    ],
)
def test_validation_messages(fields: dict, message: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        _spec(**fields)
    assert message in str(excinfo.value)


def test_confidence_null_is_rejected_rather_than_defaulted() -> None:
    with pytest.raises(ValidationError) as excinfo:
        StoppingSpec.model_validate({"confidence": None})
    assert "confidence must be a number in (0, 1), not null" in str(excinfo.value)


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StoppingSpec.model_validate({"rel_eror": 0.01})


# --------------------------------------------------------------------------- #
#  Durations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("90s", 90.0),
        ("10m", 600.0),
        ("2h", 7200.0),
        ("1.5h", 5400.0),
        ("500ms", 0.5),
        ("90", 90.0),
        (90, 90.0),
        (90.5, 90.5),
        (" 90 s ", 90.0),
        ("none", None),
        ("off", None),
        ("no-limit", None),
        ("", None),
    ],
)
def test_parse_duration(raw, seconds) -> None:
    assert parse_duration(raw) == seconds


def test_parse_duration_rejects_nonsense() -> None:
    for raw in ("-1m", "1 lightyear", "m", 0, -1.0, float("nan"), False):
        with pytest.raises(StoppingSpecError):
            parse_duration(raw)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(None, "none"), (600.0, "10m"), (7200.0, "2h"), (90.0, "90s"), (0.5, "0.5s")],
)
def test_format_duration_round_trips(seconds, text) -> None:
    assert format_duration(seconds) == text
    assert parse_duration(text) == seconds


# --------------------------------------------------------------------------- #
#  What phase 1 cannot honour
# --------------------------------------------------------------------------- #
def test_goal_queries_are_rejected_with_the_controller_message() -> None:
    with pytest.raises(UnsupportedSpecError) as excinfo:
        check_executable(_spec(goal_queries=["TOP"]).resolve())
    assert "requires the controller library, #224" in str(excinfo.value)


def test_require_goal_is_rejected_with_the_controller_message() -> None:
    with pytest.raises(UnsupportedSpecError) as excinfo:
        check_executable(_spec(require_goal=True).resolve())
    assert "requires the controller library, #224" in str(excinfo.value)


def test_supported_specs_pass_the_executability_check() -> None:
    for fields in ({}, {"trials": 1000}, {"max_trials": 1000}, {"abs_error": 1e-6}, {"max_time": "1m"}):
        check_executable(_spec(**fields).resolve())


# --------------------------------------------------------------------------- #
#  Legacy translation
# --------------------------------------------------------------------------- #
def test_legacy_delta_and_confidence_map_onto_the_goal() -> None:
    translation = translate_legacy(
        {"num_trials": 0, "early_stop": True, "delta": 0.02, "confidence": 0.99, "policy": "gbas"}
    )
    spec = translation.spec
    assert spec.mode == "goal"
    assert spec.rel_error == 0.02
    assert spec.confidence == 0.99
    assert spec.max_trials is None  # num_trials=0 keeps the unbounded behaviour
    assert spec.max_time_s is None
    assert any("delta is deprecated" in line for line in translation.deprecations)
    assert any("gbas is deprecated" in line for line in translation.deprecations)
    assert any("confidence is deprecated" in line for line in translation.deprecations)


def test_legacy_num_trials_without_early_stop_is_a_fixed_design() -> None:
    spec = translate_legacy({"num_trials": 1e7, "early_stop": False}).spec
    assert spec.mode == "fixed"
    assert spec.fixed_trials == 10_000_000
    assert spec.rel_error is None
    assert spec.confidence == DEFAULT_CONFIDENCE


def test_legacy_num_trials_with_early_stop_is_a_trial_limit() -> None:
    spec = translate_legacy({"num_trials": 1e6, "early_stop": True, "delta": 0.01}).spec
    assert spec.mode == "goal"
    assert spec.max_trials == 1_000_000
    assert spec.rel_error == 0.01
    assert spec.max_time_s == DEFAULT_GOAL_MAX_TIME_S


def test_legacy_bare_num_trials_was_already_a_fixed_design() -> None:
    """``early_stop`` defaulted to false, so a bare budget ran exactly that many trials."""
    spec = translate_legacy({"num_trials": 1024}).spec
    assert spec.mode == "fixed"
    assert spec.fixed_trials == 1024


def test_legacy_zero_trials_without_early_stop_is_open() -> None:
    spec = translate_legacy({"num_trials": 0, "early_stop": False}).spec
    assert spec.mode == "open"
    assert spec.max_time_s is None


def test_legacy_zero_trials_defaults_to_the_unbounded_goal_run() -> None:
    spec = translate_legacy({"num_trials": 0}).spec
    assert spec.mode == "goal"
    assert spec.max_trials is None
    assert spec.max_time_s is None


def test_legacy_burn_in_in_trials_is_rejected() -> None:
    with pytest.raises(LegacyTranslationError) as excinfo:
        translate_legacy({"num_trials": 1e6, "early_stop": True, "burn_in": 1024})
    message = str(excinfo.value)
    assert "burn_in=1024 has no faithful equivalent" in message
    assert "requires the controller library, #224" in message


def test_legacy_burn_in_zero_is_not_a_rejection() -> None:
    assert translate_legacy({"num_trials": 1e6, "early_stop": True, "burn_in": 0}).spec.mode == "goal"


def test_legacy_unknown_policy_is_rejected() -> None:
    with pytest.raises(LegacyTranslationError):
        translate_legacy({"policy": "bayes"})


def test_legacy_policies_list_uses_its_last_entry() -> None:
    translation = translate_legacy({"num_trials": 1, "early_stop": True, "policies": ["wald", "gbas"]})
    assert translation.spec.mode == "goal"
    assert any("gbas is deprecated" in line for line in translation.deprecations)


def test_legacy_unknown_policy_in_the_list_is_rejected() -> None:
    with pytest.raises(LegacyTranslationError):
        translate_legacy({"num_trials": 1, "policies": ["wald", "bayes"]})


@pytest.mark.parametrize("delta", [0.0, 1.0, -0.5, 2.0])
def test_legacy_delta_out_of_range_is_rejected(delta) -> None:
    with pytest.raises(LegacyTranslationError):
        translate_legacy({"num_trials": 1, "delta": delta})


def test_legacy_delta_not_a_number_is_rejected() -> None:
    with pytest.raises(LegacyTranslationError):
        translate_legacy({"num_trials": 1, "delta": "wide"})


def test_legacy_negative_num_trials_is_rejected() -> None:
    with pytest.raises(LegacyTranslationError):
        translate_legacy({"num_trials": -1})


def test_legacy_fractional_num_trials_below_one_is_rejected() -> None:
    with pytest.raises(LegacyTranslationError):
        translate_legacy({"num_trials": 0.5})


def test_legacy_confidence_out_of_range_is_rejected() -> None:
    with pytest.raises(LegacyTranslationError):
        translate_legacy({"num_trials": 1, "confidence": 1.5})


def test_legacy_max_passes_is_retained_but_not_part_of_the_specification() -> None:
    translation = translate_legacy({"num_trials": 4096, "early_stop": True, "max_passes": 8})
    assert translation.retained == {"max_passes": 8}
    assert translation.spec.mode == "goal"
    assert any("max_passes" in line for line in translation.deprecations)


# --------------------------------------------------------------------------- #
#  Runtime plumbing
# --------------------------------------------------------------------------- #
def test_from_stopping_binds_the_goal_and_the_limits() -> None:
    from fuzzybits_server.convergence import ConvergenceConfig

    goal = ConvergenceConfig.from_stopping(
        _spec(rel_error=0.005, abs_error=1e-7, max_trials=1000, max_time="2m").resolve()
    )
    assert goal.controller_mode is True
    assert goal.goal_rel_error == 0.005
    assert goal.goal_abs_error == 1e-7
    assert goal.max_trials == 1000
    assert goal.max_time_s == 120.0
    assert goal.stopping_policy == "wilson"
    assert set(goal.policies) == {"wald", "wilson"}

    fixed = ConvergenceConfig.from_stopping(_spec(trials=500).resolve())
    assert fixed.fixed_trials == 500
    assert fixed.goal_rel_error is None
    assert fixed.max_time_s is None
    assert fixed.early_stop is False


def test_from_stopping_ignores_legacy_stopping_keys_in_the_options() -> None:
    from fuzzybits_server.convergence import ConvergenceConfig

    config = ConvergenceConfig.from_stopping(
        _spec(rel_error=0.02).resolve(),
        {"delta": 0.9, "num_trials": 10, "early_stop": False, "oracle": 0.2, "passes_per_step": 4},
    )
    # The specification is the only source of the stopping rule...
    assert config.goal_rel_error == 0.02
    assert config.max_trials == 0
    assert config.early_stop is True
    # ... but the non-stopping knobs still come through.
    assert config.oracle_p == 0.2
    assert config.passes_per_step == 4


def test_sampler_budget_sizes_the_engine_to_the_specification() -> None:
    assert sampler_budget(_spec(trials=4096).resolve()) == {"num_trials": 4096}
    assert sampler_budget(_spec(max_trials=8192).resolve()) == {"num_trials": 8192}
    assert sampler_budget(_spec().resolve()) == {}


def test_describe_is_json_ready() -> None:
    payload = _spec(rel_error=0.02, max_time="30s").describe()
    assert payload["mode"] == "goal"
    assert payload["max_time"] == "30s"
    assert payload["max_time_s"] == 30.0
    assert payload["interval"] == "wilson"


def test_summary_line_names_the_goal_and_the_limits() -> None:
    line = _spec(rel_error=0.01, max_trials=10_000, max_time="10m").resolve().summary_line()
    assert "goal +/-1%" in line
    assert "95%" in line
    assert "max 10000 trials" in line
    assert "max 10m" in line
