"""Adapter from the engine's typed progress snapshots to the live-progress frame.

The web client consumes :class:`ConvergenceStatsPayload` (``liveConvergenceStats.ts``):
a flat frame of ``phase`` / ``iterations`` / ``trials`` / ``mean`` / ``epsilon`` /
``epsilon_converged`` / ``p05`` / ``p95`` … The host-side convergence engine fills
it in :meth:`ConvergenceRunner._build_progress`; this module fills the *same*
wire shape from a ``fuzzybits.ProgressSnapshot`` (engine schema v1), so a caller
that drives the one-shot ``fuzzybits.run(on_progress=...)`` path can publish the
unchanged ``LiveProgressUpdate`` events.

Not wired into the default server path: ``fuzzybits_server`` samples through
``fuzzybits.Session`` and decides convergence on the host
(:mod:`fuzzybits_server.convergence`), so it never runs the one-shot pcfb path
that emits these snapshots. See the PR for the follow-up.

Only Wald quantities the snapshot actually carries are derived. In particular the
Wald standard error is recomputed from ``hits`` / ``trials`` (the engine's own
formula), and projected trials are only emitted when the snapshot carries a
convergence target.
"""

from __future__ import annotations

import math
from typing import Any

# Two-sided 95% normal quantile, matching scram::mc::stats::tally::compute_ci.
Z95 = 1.959963984540054

#: Keys every emitted frame has. Values may be ``None`` when the snapshot lacks
#: the corresponding quantity (e.g. no convergence target).
FRAME_KEYS = (
    "phase",
    "pass_index",
    "iterations",
    "trials",
    "trials_per_pass",
    "mean",
    "std_err",
    "p05",
    "p95",
    "epsilon",
    "target_epsilon",
    "epsilon_converged",
    "converged",
    "projected_trials",
    "projected_iterations",
    "remaining_trials",
    "remaining_iterations",
)


def _finite(value: Any) -> float | None:
    """Return ``value`` as a float when it is finite, else ``None`` (browsers reject NaN)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _wald_std_err(mean: float | None, trials: int) -> float | None:
    if mean is None or trials <= 0:
        return None
    variance = mean * (1.0 - mean)
    if variance < 0.0:
        return None
    return math.sqrt(variance / trials)


def snapshot_to_live_progress(
    snapshot: Any,
    *,
    include_nodes: bool = False,
) -> dict[str, Any]:
    """Map one ``ProgressSnapshot`` (or its dict form) onto the live-progress frame.

    ``snapshot`` may be the dataclass from ``fuzzybits.ProgressSnapshot`` or the
    raw dict the binding passes to a callback; both expose the same keys. The
    first query is the primary stream, mirroring the host engine's choice of the
    target node.

    With ``include_nodes`` and more than one query, a ``nodes`` map of
    ``{name: {mean, trials}}`` is added, the same shape the frame already uses
    for per-node summaries.
    """
    def get(key: str, default: Any = None) -> Any:
        if isinstance(snapshot, dict):
            return snapshot.get(key, default)
        return getattr(snapshot, key, default)

    phase = str(get("phase", "sampling") or "sampling")
    if phase == "done":
        phase = "sampling_done"
    passes_done = int(get("passes_done", 0) or 0)
    trials_done = int(get("trials_done", 0) or 0)
    trials_total = int(get("trials_total", 0) or 0)
    trials_per_pass = trials_done // passes_done if passes_done > 0 else 0

    queries = list(get("queries", ()) or ())
    primary = queries[0] if queries else None
    if primary is None:
        frame: dict[str, Any] = {
            "phase": phase,
            "pass_index": passes_done,
            "iterations": passes_done,
            "trials": trials_done,
            "trials_per_pass": trials_per_pass,
            "mean": None,
            "std_err": None,
            "p05": None,
            "p95": None,
            "epsilon": None,
            "target_epsilon": None,
            "epsilon_converged": None,
            "converged": None,
            "projected_trials": None,
            "projected_iterations": None,
            "remaining_trials": None,
            "remaining_iterations": None,
        }
        return frame

    mean = _finite(primary["estimate"] if isinstance(primary, dict) else primary.estimate)
    ci_low = _finite(primary["ci_low"] if isinstance(primary, dict) else primary.ci_low)
    ci_high = _finite(primary["ci_high"] if isinstance(primary, dict) else primary.ci_high)
    rel_half_width = _finite(
        primary["rel_half_width"] if isinstance(primary, dict) else primary.rel_half_width
    )
    target_rel = _finite(
        primary["target_rel_half_width"]
        if isinstance(primary, dict)
        else primary.target_rel_half_width
    )
    converged = bool(primary["converged"] if isinstance(primary, dict) else primary.converged)
    trials = int(primary["trials"] if isinstance(primary, dict) else primary.trials)

    epsilon = mean * rel_half_width if mean is not None and rel_half_width is not None else None
    target_epsilon = mean * target_rel if mean is not None and target_rel is not None else None

    projected: float | None = None
    if mean is not None and 0.0 < mean < 1.0 and target_epsilon and target_epsilon > 0.0:
        # Wald: n = z² p(1-p) / ε².
        projected = Z95 * Z95 * mean * (1.0 - mean) / (target_epsilon * target_epsilon)
    remaining = max(0.0, projected - trials) if projected is not None else None
    projected_iterations = (
        math.ceil(projected / trials_per_pass)
        if projected is not None and trials_per_pass > 0
        else None
    )
    remaining_iterations = (
        math.ceil(remaining / trials_per_pass)
        if remaining is not None and trials_per_pass > 0
        else None
    )

    frame = {
        "phase": phase,
        "pass_index": passes_done,
        "iterations": passes_done,
        "trials": trials,
        "trials_per_pass": trials_per_pass,
        "mean": mean,
        "std_err": _wald_std_err(mean, trials),
        "p05": ci_low,
        "p95": ci_high,
        "epsilon": epsilon,
        "target_epsilon": target_epsilon,
        "epsilon_converged": converged if target_rel is not None else None,
        "converged": converged,
        "projected_trials": projected,
        "projected_iterations": projected_iterations,
        "remaining_trials": remaining,
        "remaining_iterations": remaining_iterations,
        # Derived from the totals rather than the queries so it stays meaningful
        # during compile stages.
        "frac_done": min(1.0, trials_done / trials_total) if trials_total > 0 else 0.0,
        "elapsed_seconds": _finite(get("elapsed_seconds", 0.0)) or 0.0,
        "trials_per_second": _finite(get("trials_per_second", 0.0)) or 0.0,
    }

    if include_nodes and len(queries) > 1:
        nodes: dict[str, dict[str, Any]] = {}
        for query in queries:
            name = str(query["name"] if isinstance(query, dict) else query.name)
            nodes[name] = {
                "mean": _finite(query["estimate"] if isinstance(query, dict) else query.estimate),
                "trials": int(query["trials"] if isinstance(query, dict) else query.trials),
            }
        frame["nodes"] = nodes

    return frame


class LiveProgressAdapter:
    """Stateful ``on_progress=`` target: forwards deduplicated live-progress frames.

    Attach as ``fuzzybits.run(..., on_progress=adapter.on_progress)``. Frames are
    emitted only when the visible fields change, so a caller can publish them
    directly as ``progress`` events without flooding the hub.
    """

    def __init__(self, publish: Any, *, include_nodes: bool = False) -> None:
        self._publish = publish
        self._include_nodes = include_nodes
        self._last: dict[str, Any] | None = None
        self.frames = 0

    def on_progress(self, snapshot: Any) -> None:
        frame = snapshot_to_live_progress(snapshot, include_nodes=self._include_nodes)
        if self._last is not None and _same_visible(self._last, frame):
            return
        self._last = frame
        self.frames += 1
        self._publish(frame)


def _same_visible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key, value in left.items():
        other = right[key]
        if key in {"elapsed_seconds", "trials_per_second"}:
            continue
        if value != other:
            return False
    return True


__all__ = ["FRAME_KEYS", "LiveProgressAdapter", "snapshot_to_live_progress"]
