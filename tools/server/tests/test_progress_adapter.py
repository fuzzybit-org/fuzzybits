"""Engine progress snapshot -> live-progress frame adapter (#206)."""

from __future__ import annotations

import math

import pytest

from fuzzybits_server.progress_adapter import (
    FRAME_KEYS,
    LiveProgressAdapter,
    snapshot_to_live_progress,
)


def _snapshot(**overrides):
    base = {
        "schema": 1,
        "phase": "sampling",
        "step": "",
        "trials_done": 2000,
        "trials_total": 4000,
        "passes_done": 2,
        "passes_total": 4,
        "elapsed_seconds": 1.5,
        "trials_per_second": 1000.0,
        "frac_done": 0.5,
        "eta_seconds": 2.0,
        "queries": [
            {
                "name": "TOP",
                "hits": 200,
                "trials": 2000,
                "estimate": 0.1,
                "ci_low": 0.085,
                "ci_high": 0.115,
                "rel_half_width": 0.15,
                "target_rel_half_width": float("nan"),
                "converged": False,
            }
        ],
    }
    base.update(overrides)
    return base


def test_frame_carries_the_unchanged_live_progress_keys():
    frame = snapshot_to_live_progress(_snapshot())
    for key in FRAME_KEYS:
        assert key in frame
    assert frame["phase"] == "sampling"
    assert frame["pass_index"] == 2
    assert frame["iterations"] == 2
    assert frame["trials"] == 2000
    assert frame["trials_per_pass"] == 1000
    assert frame["mean"] == pytest.approx(0.1)
    assert frame["p05"] == pytest.approx(0.085)
    assert frame["p95"] == pytest.approx(0.115)
    assert frame["epsilon"] == pytest.approx(0.1 * 0.15)
    assert frame["converged"] is False


def test_wald_std_err_matches_the_engine_formula():
    frame = snapshot_to_live_progress(_snapshot())
    expected = math.sqrt(0.1 * 0.9 / 2000)
    assert frame["std_err"] == pytest.approx(expected)


def test_no_target_means_no_epsilon_or_projection():
    frame = snapshot_to_live_progress(_snapshot())
    assert frame["target_epsilon"] is None
    assert frame["projected_trials"] is None
    assert frame["epsilon_converged"] is None


def test_target_projects_wald_trials():
    snapshot = _snapshot()
    snapshot["queries"][0]["target_rel_half_width"] = 0.05
    snapshot["queries"][0]["converged"] = True
    frame = snapshot_to_live_progress(snapshot)
    target_epsilon = 0.1 * 0.05
    expected = 1.959963984540054**2 * 0.1 * 0.9 / target_epsilon**2
    assert frame["target_epsilon"] == pytest.approx(target_epsilon)
    assert frame["projected_trials"] == pytest.approx(expected)
    assert frame["remaining_trials"] >= 0
    assert frame["epsilon_converged"] is True


def test_done_phase_is_distinguished():
    frame = snapshot_to_live_progress(_snapshot(phase="done"))
    assert frame["phase"] == "sampling_done"


def test_empty_queries_still_produce_a_frame():
    frame = snapshot_to_live_progress(_snapshot(phase="compile", queries=[]))
    for key in FRAME_KEYS:
        assert key in frame
    assert frame["mean"] is None
    assert frame["trials"] == 2000


def test_non_finite_values_become_none():
    snapshot = _snapshot()
    snapshot["queries"][0]["estimate"] = float("nan")
    frame = snapshot_to_live_progress(snapshot)
    assert frame["mean"] is None
    assert frame["std_err"] is None


def test_multi_query_nodes_only_when_asked():
    snapshot = _snapshot()
    snapshot["queries"].append(
        {
            "name": "CLASS-2A",
            "hits": 3,
            "trials": 2000,
            "estimate": 0.0015,
            "ci_low": 0.0,
            "ci_high": 0.004,
            "rel_half_width": 1.33,
            "target_rel_half_width": float("nan"),
            "converged": False,
        }
    )
    assert "nodes" not in snapshot_to_live_progress(snapshot)
    frame = snapshot_to_live_progress(snapshot, include_nodes=True)
    assert set(frame["nodes"]) == {"TOP", "CLASS-2A"}
    assert frame["nodes"]["CLASS-2A"]["trials"] == 2000


def test_adapter_deduplicates_unchanged_frames():
    published = []
    adapter = LiveProgressAdapter(published.append)
    adapter.on_progress(_snapshot())
    # Same visible state, different timing: not republished.
    adapter.on_progress(_snapshot(elapsed_seconds=9.9, trials_per_second=1.0))
    assert len(published) == 1
    adapter.on_progress(_snapshot(passes_done=3, trials_done=3000))
    assert len(published) == 2
    assert adapter.frames == 2
