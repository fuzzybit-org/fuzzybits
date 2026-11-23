"""Characterization: what the run-settings request schema accepts today.

These pin the API surface a client can rely on before and after the #224
stopping specification lands: the legacy option names with their defaults, the
new ``stopping`` model, and the 422 for anything the host loop cannot honour.
Every case goes through the real router, so the FastAPI request model, the
resolution and the response shape are all exercised together.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from fuzzybits_server.app import create_app
from fuzzybits_server.config import CalculationConfig
from fuzzybits_server.convergence.stopping import (
    DEFAULT_CONFIDENCE,
    DEFAULT_GOAL_MAX_TIME_S,
    DEFAULT_REL_ERROR,
    ResolvedStopping,
)
from fuzzybits_server.event_hub import EventHub
from fuzzybits_server.request_spec import RequestSpecError, resolve_request
from fuzzybits_server.supervisor import CalculationSupervisor


class _RecordingSupervisor(CalculationSupervisor):
    """Captures the config a control call would start, without a device."""

    def __init__(self, hub: EventHub) -> None:
        super().__init__(hub)
        self.started: list[CalculationConfig] = []

    def start_sampling(self, config: CalculationConfig) -> dict[str, Any]:
        self.started.append(config)
        return {"accepted": True, "status": {"status": "running"}}

    def restart(self, config: CalculationConfig) -> dict[str, Any]:
        self.started.append(config)
        return {"accepted": True, "status": {"status": "running"}}


@pytest.fixture()
def client() -> tuple[TestClient, _RecordingSupervisor]:
    hub = EventHub()
    supervisor = _RecordingSupervisor(hub)
    app = create_app(event_hub=hub, supervisor=supervisor, autostart=False)
    return TestClient(app), supervisor


def _start(client: TestClient, body: dict[str, Any]):
    return client.post("/api/v1/start", json=body)


# --------------------------------------------------------------------------- #
#  Legacy request schema, as it is today
# --------------------------------------------------------------------------- #
LEGACY_DEFAULTS: dict[str, Any] = {
    "model_id": None,
    "compilation_level": None,
    "expand_atleast_gates": None,
    "expand_xor_gates": None,
    "recognize_atleast_groups": None,
    "recognize_xor_groups": None,
    "ccf": None,
    "keep_null_gates": None,
    "stopping": None,
    "num_trials": None,
    "requested_trials_per_pass": None,
    "confidence": None,
    "delta": None,
    "oracle": None,
    "early_stop": None,
    "policy": None,
    "stopping_policy": None,
    "policies": None,
    "burn_in": None,
    "pcfb_id": None,
    "watch": None,
}


def test_request_schema_fields_and_defaults_are_pinned() -> None:
    from fuzzybits_server.routes import SamplingControlBody

    defaults = {name: field.default for name, field in SamplingControlBody.model_fields.items()}
    assert defaults == LEGACY_DEFAULTS


def test_body_with_no_stopping_intent_keeps_the_configured_loop(client) -> None:
    http, supervisor = client
    response = _start(http, {"model_id": None, "requested_trials_per_pass": 4096})
    assert response.status_code == 200
    config = supervisor.started[-1]
    assert config.stopping is None
    assert config.options["requested_trials_per_pass"] == 4096


def test_completely_empty_body_is_accepted(client) -> None:
    http, supervisor = client
    assert http.post("/api/v1/start").status_code == 200
    assert supervisor.started[-1].stopping is None


def test_legacy_request_is_translated_and_reported(client) -> None:
    http, supervisor = client
    response = _start(
        http,
        {"num_trials": 1e6, "early_stop": True, "delta": 0.02, "confidence": 0.99, "policy": "gbas"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["stopping"]["mode"] == "goal"
    assert payload["stopping"]["rel_error"] == 0.02
    assert payload["stopping"]["confidence"] == 0.99
    assert payload["stopping"]["max_trials"] == 1_000_000
    assert payload["stopping"]["legacy"] is True
    assert any("deprecated" in line for line in payload["warnings"])

    spec = supervisor.started[-1].stopping
    assert isinstance(spec, ResolvedStopping)
    assert spec.rel_error == 0.02
    assert spec.deprecations == tuple(payload["warnings"])


def test_legacy_fixed_run_still_means_exactly_n(client) -> None:
    http, supervisor = client
    response = _start(http, {"num_trials": 4096, "early_stop": False})
    assert response.status_code == 200
    spec = supervisor.started[-1].stopping
    assert spec.mode == "fixed"
    assert spec.fixed_trials == 4096


def test_legacy_and_new_models_are_mutually_exclusive(client) -> None:
    http, _ = client
    response = _start(http, {"stopping": {"rel_error": 0.01}, "delta": 0.01})
    assert response.status_code == 400
    assert "not both" in response.json()["detail"]


# --------------------------------------------------------------------------- #
#  The new model
# --------------------------------------------------------------------------- #
def test_stopping_model_is_accepted_and_defaulted(client) -> None:
    http, supervisor = client
    response = _start(http, {"stopping": {}})
    assert response.status_code == 200
    spec = supervisor.started[-1].stopping
    assert spec.mode == "goal"
    assert spec.rel_error == DEFAULT_REL_ERROR
    assert spec.confidence == DEFAULT_CONFIDENCE
    assert spec.max_time_s == DEFAULT_GOAL_MAX_TIME_S
    assert response.json()["stopping"]["legacy"] is False


def test_stopping_model_as_a_json_string_is_accepted(client) -> None:
    http, supervisor = client
    response = _start(http, {"stopping": '{"trials": 12345}'})
    assert response.status_code == 200
    assert supervisor.started[-1].stopping.fixed_trials == 12345


def test_stopping_model_malformed_json_string_is_a_422(client) -> None:
    http, _ = client
    response = _start(http, {"stopping": "{not json"})
    assert response.status_code == 422
    assert "not valid JSON" in response.json()["detail"]


def test_fixed_design_request(client) -> None:
    http, supervisor = client
    response = _start(http, {"stopping": {"trials": 1_000_000}})
    assert response.status_code == 200
    spec = supervisor.started[-1].stopping
    assert spec.mode == "fixed"
    assert spec.fixed_trials == 1_000_000
    assert spec.max_time_s is None


def test_goal_request_with_limits(client) -> None:
    http, supervisor = client
    response = _start(
        http,
        {"stopping": {"rel_error": 0.005, "confidence": 0.99, "max_trials": 10**9, "max_time": "2h"}},
    )
    assert response.status_code == 200
    spec = supervisor.started[-1].stopping
    assert (spec.rel_error, spec.confidence, spec.max_trials, spec.max_time_s) == (
        0.005,
        0.99,
        10**9,
        7200.0,
    )


def test_the_stopping_spec_supersedes_the_legacy_options_in_the_config(client) -> None:
    http, supervisor = client
    _start(http, {"stopping": {"rel_error": 0.02}})
    config = supervisor.started[-1]
    assert config.stopping.rel_error == 0.02


@pytest.mark.parametrize(
    ("stopping", "status", "message"),
    [
        ({"rel_error": 0.0}, 422, "rel_error must be greater than 0"),
        ({"rel_error": 2.0}, 422, "rel_error must be greater than 0"),
        ({"trials": 0}, 422, "trials must be at least 1"),
        ({"trials": 10, "rel_error": 0.01}, 422, "cannot be combined"),
        ({"confidence": 1.0}, 422, "confidence must be greater than 0"),
        ({"max_time": "soon"}, 422, "is not a duration"),
        ({"goal_queries": ["TOP"]}, 422, "requires the controller library, #224"),
        ({"require_goal": True}, 422, "requires the controller library, #224"),
        ({"rel_eror": 0.01}, 422, "rel_eror"),
    ],
)
def test_unhonourable_or_invalid_specs_are_rejected(client, stopping, status, message) -> None:
    http, supervisor = client
    response = _start(http, {"stopping": stopping})
    assert response.status_code == status
    assert message in response.json()["detail"]
    assert supervisor.started == []


def test_legacy_burn_in_is_a_422_naming_the_controller_library(client) -> None:
    http, supervisor = client
    response = _start(http, {"num_trials": 1e6, "early_stop": True, "burn_in": 1000})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "burn_in=1000" in detail
    assert "requires the controller library, #224" in detail
    assert supervisor.started == []


def test_legacy_unknown_policy_is_a_422_from_the_translation(client) -> None:
    http, supervisor = client
    response = _start(http, {"num_trials": 1e6, "policy": "bayes"})
    assert response.status_code == 422
    assert "is not a stopping policy" in response.json()["detail"]
    assert supervisor.started == []


def test_legacy_env_vocabulary_names_are_accepted_and_translated(client) -> None:
    """`stopping_policy` and `policies` were reachable through the env only."""
    http, supervisor = client
    response = _start(http, {"num_trials": 0, "early_stop": True, "stopping_policy": "gbas"})
    assert response.status_code == 200
    assert supervisor.started[-1].stopping.rel_error == 0.01
    assert any("gbas is deprecated" in line for line in response.json()["warnings"])

    response = _start(http, {"num_trials": 0, "early_stop": True, "policies": ["wald", "gbas"]})
    assert response.status_code == 200
    assert any("gbas is deprecated" in line for line in response.json()["warnings"])


def test_reset_accepts_the_same_specification(client) -> None:
    http, supervisor = client
    response = http.post("/api/v1/reset", json={"stopping": {"trials": 512}})
    assert response.status_code == 200
    assert supervisor.started[-1].stopping.fixed_trials == 512


# --------------------------------------------------------------------------- #
#  resolve_request in isolation
# --------------------------------------------------------------------------- #
def test_resolve_request_returns_none_without_stopping_intent() -> None:
    assert resolve_request(None) is None
    assert resolve_request({}) is None
    assert resolve_request({"model_id": "chinese", "requested_trials_per_pass": 1024}) is None
    assert resolve_request({"num_trials": None}) is None


def test_resolve_request_rejects_a_non_object_stopping_payload() -> None:
    with pytest.raises(RequestSpecError) as excinfo:
        resolve_request({"stopping": 42})
    assert excinfo.value.status_code == 422


def test_resolve_request_reports_the_legacy_flag() -> None:
    resolved = resolve_request({"num_trials": 4096, "early_stop": False})
    assert resolved is not None
    assert resolved.legacy is True
    assert resolved.stopping.mode == "fixed"


def test_resolve_request_reports_a_new_style_specification() -> None:
    resolved = resolve_request({"stopping": {"abs_error": 1e-7}})
    assert resolved is not None
    assert resolved.legacy is False
    assert resolved.stopping.abs_error == 1e-7
    assert resolved.stopping.rel_error is None


def test_describe_is_stable_for_a_response_payload() -> None:
    resolved = resolve_request({"stopping": {"rel_error": 0.01, "max_time": "10m"}})
    payload = resolved.describe()
    assert payload["mode"] == "goal"
    assert payload["legacy"] is False
    assert payload["max_time"] == "10m"
    assert "deprecations" not in payload
