"""Typed ``on_progress=`` for ``fuzzybits.run`` and ``fuzzybits.compile`` (#206).

The engine emits schema-v1 progress snapshots (``fuzzybits.ProgressSnapshot``),
rate-limited at the source and always including the first and the last one.
These tests check the typed shape, key stability, monotone work, agreement with
the final result, that attaching a callback does not change the numbers, and
that an exception in the callback stops the run cleanly and re-raises.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402

SNAPSHOT_FIELDS = {
    "schema",
    "phase",
    "step",
    "trials_done",
    "trials_total",
    "passes_done",
    "passes_total",
    "elapsed_seconds",
    "trials_per_second",
    "frac_done",
    "eta_seconds",
    "queries",
}

QUERY_FIELDS = {
    "name",
    "hits",
    "trials",
    "estimate",
    "ci_low",
    "ci_high",
    "rel_half_width",
    "target_rel_half_width",
    "converged",
}


@pytest.fixture(scope="module", autouse=True)
def _omp_backend():
    fuzzybits.init(backend="omp")
    yield


@pytest.fixture(scope="session")
def smalltree_pcfb(smalltree_xml: Path) -> bytes:
    return fuzzybits.compile(smalltree_xml)


def test_progress_snapshot_is_a_typed_dataclass(smalltree_pcfb: bytes):
    seen: list[fuzzybits.ProgressSnapshot] = []
    fuzzybits.run(smalltree_pcfb, trials=1 << 14, on_progress=seen.append)
    assert seen, "on_progress received no snapshot"
    for snapshot in seen:
        assert isinstance(snapshot, fuzzybits.ProgressSnapshot)
        assert dataclasses.is_dataclass(snapshot)
        assert snapshot.schema == 1
        assert snapshot.phase in {"compile", "sampling", "checking", "done"}
        for query in snapshot.queries:
            assert isinstance(query, fuzzybits.QueryState)
            assert set(QUERY_FIELDS) == {f.name for f in dataclasses.fields(query)}
    assert set(SNAPSHOT_FIELDS) == {f.name for f in dataclasses.fields(fuzzybits.ProgressSnapshot)}


def test_run_progress_first_last_and_monotone(smalltree_pcfb: bytes):
    seen: list[fuzzybits.ProgressSnapshot] = []
    results = fuzzybits.run(smalltree_pcfb, trials=1 << 16, on_progress=seen.append)
    assert len(seen) >= 2

    # Work done never decreases, and the first/last snapshots are present.
    for previous, current in zip(seen, seen[1:]):
        assert current.trials_done >= previous.trials_done
        assert current.passes_done >= previous.passes_done

    last = seen[-1]
    assert last.phase == "done"
    assert last.passes_total > 0
    assert last.passes_done == last.passes_total
    assert last.trials_total > 0
    assert last.trials_done == last.trials_total
    assert 0.0 <= last.frac_done <= 1.0
    assert last.elapsed_seconds >= 0.0

    # The final snapshot agrees exactly with the returned result.
    assert len(last.queries) == len(results)
    for query, result in zip(last.queries, results):
        assert query.name == result.name
        assert query.hits == result.hits
        assert query.trials == result.n
        assert query.estimate == result.estimate
        assert query.ci_low == result.ci_low
        assert query.ci_high == result.ci_high
        assert query.rel_half_width == pytest.approx(
            (result.ci_high - result.ci_low) / (2.0 * result.estimate)
        )
        # No convergence target is configured by `fuzzybits.run`.
        assert math_is_nan(query.target_rel_half_width)


def math_is_nan(value: float) -> bool:
    return value != value


def test_run_results_identical_with_and_without_callback(smalltree_pcfb: bytes):
    plain = fuzzybits.run(smalltree_pcfb, trials=1 << 16)
    with_progress = fuzzybits.run(smalltree_pcfb, trials=1 << 16, on_progress=lambda _s: None)
    assert len(plain) == len(with_progress)
    for a, b in zip(plain, with_progress):
        # Bit-identical, not approximately equal.
        assert (a.name, a.hits, a.n, a.estimate, a.ci_low, a.ci_high) == (
            b.name,
            b.hits,
            b.n,
            b.estimate,
            b.ci_low,
            b.ci_high,
        )


def test_run_progress_exception_stops_and_reraises(smalltree_pcfb: bytes):
    calls: list[int] = []

    class CallbackError(RuntimeError):
        pass

    def on_progress(_snapshot: fuzzybits.ProgressSnapshot) -> None:
        calls.append(len(calls))
        raise CallbackError("stop here")

    with pytest.raises(CallbackError, match="stop here"):
        fuzzybits.run(smalltree_pcfb, trials=1 << 18, on_progress=on_progress)
    # The callback ran once, then the run stopped: no second invocation.
    assert calls == [0]


def test_compile_progress_stages(smalltree_xml: Path):
    seen: list[fuzzybits.ProgressSnapshot] = []
    blob = fuzzybits.compile(smalltree_xml, on_progress=seen.append)
    assert blob

    compile_steps = [s.step for s in seen if s.phase == "compile"]
    assert compile_steps[:4] == ["parse", "lower", "passes", "write"]
    assert seen[-1].phase == "done"
    for snapshot in seen:
        # Compile snapshots carry no sampling work or queries.
        assert snapshot.passes_total == 0
        assert snapshot.queries == ()


def test_compile_progress_exception_stops_and_reraises(smalltree_xml: Path):
    def on_progress(_snapshot: fuzzybits.ProgressSnapshot) -> None:
        raise ValueError("stop compile")

    with pytest.raises(ValueError, match="stop compile"):
        fuzzybits.compile(smalltree_xml, on_progress=on_progress)


def test_legacy_run_probability_callback_is_unchanged(tiny_xml: Path):
    # The legacy string key-value callback must keep working exactly as before.
    payloads: list[dict[str, object]] = []
    native = fuzzybits.native()
    native.run_probability(
        [str(tiny_xml)],
        {"num_trials": float(1 << 12), "compilation_level": 2, "no_kn": True, "policy": "wald"},
        payloads.append,
    )
    assert payloads, "legacy callback received no update"
    assert "mean" in payloads[-1]
