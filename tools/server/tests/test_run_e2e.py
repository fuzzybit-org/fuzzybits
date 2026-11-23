"""End-to-end: request -> calculation config -> worker -> frames -> result.

These use FastAPI's ``TestClient`` with a supervisor double that runs the real
worker entry point in-process, and ``BernoulliSampler`` in place of the device, so
the whole path from an HTTP body to the progress frames a client receives is
exercised without a GPU, a subprocess or a live server.

Covered here, because nothing else pins them together:

* a #224 stopping-specification request produces controller stop reasons, the
  Wilson interval and the frames the UI reads;
* a legacy request is translated, warns, and still runs;
* a request the host loop cannot honour is a 422 and starts nothing;
* the run result carries the stopping block, the interval and the deprecations.
"""

from __future__ import annotations

import queue
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fuzzybits_server import worker
from fuzzybits_server.app import create_app
from fuzzybits_server.config import CalculationConfig
from fuzzybits_server.convergence import BernoulliSampler, run_calculation
from fuzzybits_server.event_hub import EventHub
from fuzzybits_server.supervisor import CalculationSupervisor

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
        "num_one_bits",
        "total_bits",
        "primary_policy",
        "stopping_policy",
    }
)


class _InProcessSupervisor(CalculationSupervisor):
    """Runs the calculation in this process, publishing what the worker would."""

    def __init__(self, hub: EventHub) -> None:
        super().__init__(hub)
        self.configs: list[CalculationConfig] = []
        self.events: list[dict[str, Any]] = []

    def _run(self, config: CalculationConfig) -> dict[str, Any]:
        self.configs.append(config)
        events: queue.Queue = queue.Queue()
        worker.calculation_main(config, events)
        while not events.empty():
            self.events.append(events.get())
        return {"accepted": True, "status": {"status": "completed", "alive": False}}

    def start_sampling(self, config: CalculationConfig) -> dict[str, Any]:
        return self._run(config)

    def restart(self, config: CalculationConfig) -> dict[str, Any]:
        return self._run(config)


@pytest.fixture()
def hub_and_client(monkeypatch: pytest.MonkeyPatch):
    """A client whose runs use the deterministic simulated sampler."""
    from fuzzybits_server.convergence import runner as runner_module

    def _simulated(input_files, options=None, *, event_ids=None, pcfb=None, pcfb_path=None):
        del input_files, event_ids, pcfb, pcfb_path
        tpp = int(float((options or {}).get("requested_trials_per_pass") or 4096))
        return BernoulliSampler(true_p=0.2, trials_per_pass=tpp, target_index=1, seed=11)

    monkeypatch.setattr(runner_module, "make_sampler", _simulated)
    hub = EventHub()
    supervisor = _InProcessSupervisor(hub)
    app = create_app(event_hub=hub, supervisor=supervisor, autostart=False)
    return hub, supervisor, TestClient(app)


def _progress_events(supervisor: _InProcessSupervisor) -> list[dict]:
    return [event["update"] for event in supervisor.events if event.get("type") == "progress"]


def _completed(supervisor: _InProcessSupervisor) -> dict:
    completed = [event for event in supervisor.events if event.get("type") == "completed"]
    assert completed, f"no completed event; events were {[e.get('type') for e in supervisor.events]}"
    return completed[-1]["result"]


# --------------------------------------------------------------------------- #
#  A #224 goal request, end to end
# --------------------------------------------------------------------------- #
def test_goal_request_runs_to_the_goal_and_reports_it(hub_and_client) -> None:
    _, supervisor, client = hub_and_client
    response = client.post(
        "/api/v1/start",
        json={
            "stopping": {"rel_error": 0.05, "confidence": 0.95},
            "requested_trials_per_pass": 4096,
        },
    )
    assert response.status_code == 200
    assert response.json()["stopping"]["mode"] == "goal"

    result = _completed(supervisor)
    assert result["stop_reason"] == "goal"
    assert result["converged"] is True
    assert result["interval"] == "wilson"
    assert result["stopping"]["rel_error"] == 0.05
    assert result["stopping_summary"].startswith("goal +/-5%")
    assert result["sample_mean"] == result["mean"]  # Wilson reports the sample mean


def test_goal_frames_carry_what_the_ui_reads(hub_and_client) -> None:
    _, supervisor, client = hub_and_client
    client.post("/api/v1/start", json={"stopping": {"rel_error": 0.05}})

    frames = _progress_events(supervisor)
    assert len(frames) >= 2
    for frame in frames:
        missing = CANONICAL_KEYS - frame.keys()
        assert not missing, f"missing canonical fields: {sorted(missing)}"
        assert frame["interval"] == "wilson"
        assert frame["stop_reason_style"] == "controller"
        assert frame["target_rel_half_width"] == 0.05
        assert frame["wilson_rel_half_width"] is None or frame["wilson_rel_half_width"] > 0

    last = frames[-1]
    assert last["stop_reason"] == "goal"
    assert last["converged"] is True
    # The reported interval is Wilson's, and it contains the estimate.
    assert last["ci_low"] <= last["mean"] <= last["ci_high"]
    assert last["p05"] == last["ci_low"]
    assert last["p95"] == last["ci_high"]


def test_fixed_design_request_runs_exactly_n(hub_and_client) -> None:
    _, supervisor, client = hub_and_client
    client.post("/api/v1/start", json={"stopping": {"trials": 4096 * 3}})
    result = _completed(supervisor)
    assert result["stop_reason"] == "trials"
    assert result["trials"] == 4096 * 3


def test_trial_limit_request_reports_the_limit(hub_and_client) -> None:
    _, supervisor, client = hub_and_client
    client.post(
        "/api/v1/start",
        json={"stopping": {"rel_error": 1e-5, "max_trials": 4096 * 2}},
    )
    result = _completed(supervisor)
    assert result["stop_reason"] == "max-trials"
    assert result["converged"] is False
    assert result["trials"] >= 4096 * 2


def test_time_limit_request_reports_the_limit(hub_and_client) -> None:
    _, supervisor, client = hub_and_client
    client.post(
        "/api/v1/start",
        json={"stopping": {"rel_error": 1e-9, "max_time": "0.001s"}},
    )
    result = _completed(supervisor)
    # A goal that needs far more than the time allows stops on the clock, or
    # finishes so fast the goal is met; the sampler is deterministic, so pin the
    # contract rather than the race: a limit or the goal, never neither.
    assert result["stop_reason"] in {"max-time", "goal"}


# --------------------------------------------------------------------------- #
#  A legacy request, end to end
# --------------------------------------------------------------------------- #
def test_legacy_request_is_translated_annotated_and_run(hub_and_client) -> None:
    _, supervisor, client = hub_and_client
    response = client.post(
        "/api/v1/start",
        json={"num_trials": 4096 * 2, "early_stop": False, "delta": 0.05, "confidence": 0.95},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["stopping"]["legacy"] is True
    assert payload["warnings"]

    result = _completed(supervisor)
    assert result["stop_reason"] == "trials"
    assert result["trials"] == 4096 * 2
    assert result["stopping"]["legacy"] is True
    assert result["stopping"]["deprecations"] == payload["warnings"]

    frames = _progress_events(supervisor)
    assert frames[-1]["stop_reason"] == "trials"
    assert frames[-1]["interval"] == "wilson"


def test_legacy_early_stop_still_uses_the_goal_rule(hub_and_client) -> None:
    _, supervisor, client = hub_and_client
    client.post(
        "/api/v1/start",
        json={"num_trials": 0, "early_stop": True, "delta": 0.05, "confidence": 0.95},
    )
    result = _completed(supervisor)
    assert result["stop_reason"] == "goal"
    assert result["stopping"]["rel_error"] == 0.05
    assert result["stopping"]["max_time_s"] is None  # an unbounded legacy goal run


# --------------------------------------------------------------------------- #
#  Rejections start nothing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"stopping": {"goal_queries": ["TOP"]}}, "requires the controller library, #224"),
        ({"stopping": {"require_goal": True}}, "requires the controller library, #224"),
        ({"stopping": {"trials": 10, "rel_error": 0.01}}, "cannot be combined"),
        ({"num_trials": 1e6, "early_stop": True, "burn_in": 100}, "has no faithful equivalent"),
        ({"stopping": {"rel_error": 0.01}, "delta": 0.01}, "not both"),
    ],
)
def test_rejected_requests_start_no_run(hub_and_client, body, message) -> None:
    _, supervisor, client = hub_and_client
    response = client.post("/api/v1/start", json=body)
    assert response.status_code in {400, 422}
    assert message in response.json()["detail"]
    assert supervisor.configs == []
    assert supervisor.events == []


# --------------------------------------------------------------------------- #
#  The worker's own contract
# --------------------------------------------------------------------------- #
def test_worker_emits_progress_then_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    from fuzzybits_server.convergence import runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "make_sampler",
        lambda *a, **k: BernoulliSampler(true_p=0.5, trials_per_pass=1024, target_index=1, seed=3),
    )
    config = CalculationConfig(
        input_files=[],
        options={"requested_trials_per_pass": 1024},
        stopping=__import__(
            "fuzzybits_server.convergence.stopping", fromlist=["StoppingSpec"]
        ).StoppingSpec.model_validate({"trials": 1024 * 2}).resolve(),
    )
    events: queue.Queue = queue.Queue()
    worker.calculation_main(config, events)

    collected = []
    while not events.empty():
        collected.append(events.get())
    assert collected[-1]["type"] == "completed"
    progress = [event for event in collected if event["type"] == "progress"]
    assert progress
    assert [event["sequence"] for event in progress] == list(range(1, len(progress) + 1))
    assert all("timestamp" in event and "update" in event for event in progress)
    assert progress[-1]["update"]["stop_reason"] == "trials"
    assert collected[-1]["result"]["trials"] == 1024 * 2


def test_run_calculation_without_a_stopping_spec_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """No specification means the legacy loop, with its own stop reasons."""
    from fuzzybits_server.convergence import runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "make_sampler",
        lambda *a, **k: BernoulliSampler(true_p=0.2, trials_per_pass=4096, target_index=1, seed=7),
    )
    result = run_calculation([], {"num_trials": 4096 * 2, "early_stop": False})
    assert result["stop_reason"] == "max_trials"
    assert "stopping" not in result
    assert "interval" not in result


def test_run_calculation_with_a_spec_carries_the_block(monkeypatch: pytest.MonkeyPatch) -> None:
    from fuzzybits_server.convergence import runner as runner_module
    from fuzzybits_server.convergence.stopping import StoppingSpec

    monkeypatch.setattr(
        runner_module,
        "make_sampler",
        lambda *a, **k: BernoulliSampler(true_p=0.2, trials_per_pass=4096, target_index=1, seed=7),
    )
    spec = StoppingSpec.model_validate({"rel_error": 0.05}).resolve()
    result = run_calculation([], {}, stopping=spec)
    assert result["stop_reason"] == "goal"
    assert result["interval"] == "wilson"
    assert result["stopping"] == {**spec.as_dict(), "deprecations": []}
    assert result["stopping_summary"].startswith("goal +/-5%")
