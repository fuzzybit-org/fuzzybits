"""Characterization: an interrupted run reports why it ended.

A run stopped through ``POST /api/v1/stop`` is terminated, so it never reaches a
stop condition of its own. The frames it already sent are unchanged, and the
state it leaves behind says ``interrupted`` rather than leaving the last frame
looking like a live run.
"""

from __future__ import annotations

from typing import Any

from fuzzybits_server.state import CalculationState


def _frame(**overrides: Any) -> dict[str, Any]:
    frame = {
        "phase": "sampling",
        "pass_index": 3,
        "iterations": 3,
        "trials": 12288,
        "trials_per_pass": 4096,
        "mean": 0.2,
        "converged": False,
        "interval": "wilson",
        "stop_reason_style": "controller",
    }
    frame.update(overrides)
    return frame


def test_stop_marks_the_last_frame_interrupted() -> None:
    state = CalculationState()
    state.mark_started(1234)
    state.record_progress(_frame())
    state.mark_stopped()

    snapshot = state.snapshot(alive=False)
    assert snapshot["status"] == "stopped"
    assert snapshot["last_update"]["stop_reason"] == "interrupted"
    # Nothing else in the frame changes.
    frame = snapshot["last_update"]
    assert frame["trials"] == 12288
    assert frame["interval"] == "wilson"
    assert frame["converged"] is False


def test_stop_leaves_a_legacy_frame_alone() -> None:
    """A legacy run has no `stop_reason_style`, so its reason vocabulary is untouched."""
    state = CalculationState()
    state.mark_started(1234)
    state.record_progress({"phase": "sampling", "trials": 4096, "mean": 0.1})
    state.mark_stopped()
    assert "stop_reason" not in state.snapshot(alive=False)["last_update"]


def test_stop_before_any_progress_has_nothing_to_mark() -> None:
    state = CalculationState()
    state.mark_started(1234)
    state.mark_stopped()
    snapshot = state.snapshot(alive=False)
    assert snapshot["last_update"] is None
    assert snapshot["status"] == "stopped"


def test_stop_does_not_rewrite_a_frame_that_already_stopped() -> None:
    """A frame with a reason of its own belongs to a run that finished; keep it."""
    state = CalculationState()
    state.mark_started(1234)
    state.record_progress(_frame(stop_reason="goal", converged=True))
    state.mark_stopped()
    assert state.snapshot(alive=False)["last_update"]["stop_reason"] == "goal"
