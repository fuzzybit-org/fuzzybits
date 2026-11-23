"""Fixed-N equivalence: a server run against ``fzb run -N`` on a public fixture.

The claim under test is the #223 consequence the controller relies on: for a
fixed design, the server's per-query counts equal those of ``fzb run -N`` with
the same N *and the same pass size*.

Pass size is the part that has to be stated precisely, because today's engine is
not yet step-size invariant (#223 item 1):

* ``fzb run`` sizes its own geometry; for a budget of N it fills the device and
  issues the whole budget as one pass on these fixtures.
* The server's sampler uses ``requested_trials_per_pass``. Left unset, it also
  gets one pass covering N, and the counts agree bit for bit.
* Setting it smaller splits the run into several passes, and the counts differ.
  That is measured and reported, not hidden.

So the equivalence test uses the same geometry, and `test_pass_geometry_changes_counts`
pins the current difference so it cannot be mistaken for a server bug. When #223
lands this test becomes exact for every pass size.

Run with ``-m native`` and the built extension on ``sys.path``; ``FZB_BINARY`` may
point at the ``fzb`` executable. Without it the comparison falls back to
``fuzzybits.run`` (the same one-shot engine path ``fzb run`` uses) and says so.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzybits_server.convergence import StoppingSpec, run_calculation

pytestmark = pytest.mark.native

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "python" / "data" / "tiny.xml"

#: One trial budget and two geometries. Small enough for CI, large enough that a
#: pass-geometry difference shows up in the counts.
N = 1 << 20


def _pcfb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import fuzzybits

    monkeypatch.setenv("FUZZYBITS_SERVER_CACHE_DIR", str(tmp_path / "cache"))
    blob = bytes(fuzzybits.compile(str(FIXTURE), pdag_c=2))
    path = tmp_path / "tiny.pcfb"
    path.write_bytes(blob)
    return path


def _fzb_binary() -> str | None:
    explicit = os.environ.get("FZB_BINARY")
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("fzb")
    if found:
        return found
    for candidate in sorted(REPO.glob("build/*/targets/fzb/fzb")):
        if candidate.is_file():
            return str(candidate)
    return None


def _fzb_run(pcfb: Path, trials: int) -> dict:
    """``fzb run -N <trials> --json``; falls back to ``fuzzybits.run``."""
    binary = _fzb_binary()
    if binary is None:
        import fuzzybits

        result = fuzzybits.run(pcfb.read_bytes(), trials=trials)[0]
        return {"name": result.name, "n": result.n, "hits": result.hits, "source": "fuzzybits.run"}
    completed = subprocess.run(
        [binary, "run", str(pcfb), "--trials", str(trials), "--json", "--no-progress"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "ACPP_VISIBILITY_MASK": os.environ.get("ACPP_VISIBILITY_MASK", "omp")},
    )
    payload = json.loads(completed.stdout)
    query = payload["queries"][0]
    return {"name": query["name"], "n": query["n"], "hits": query["hits"], "source": binary}


def _server_run(**stopping) -> dict:
    spec = StoppingSpec.model_validate(stopping).resolve()
    return run_calculation([str(FIXTURE)], {"compilation_level": 2}, stopping=spec)


def test_fixed_design_matches_fzb_run_at_the_same_pass_size(tmp_path: Path, monkeypatch) -> None:
    if not FIXTURE.is_file():
        pytest.skip(f"missing {FIXTURE}")
    pcfb = _pcfb(tmp_path, monkeypatch)

    reference = _fzb_run(pcfb, N)
    server = _server_run(trials=N)

    # Same budget, and the same geometry: one pass of N trials on both sides.
    assert server["passes"] == 1
    assert server["trials"] == reference["n"]
    hits = round(server["mean"] * server["trials"])
    assert hits == reference["hits"], (
        f"counts differ: fzb={reference['hits']} server={hits} "
        f"(reference from {reference['source']})"
    )
    assert server["stop_reason"] == "trials"


def test_legacy_fixed_run_matches_fzb_run_at_the_same_pass_size(tmp_path: Path, monkeypatch) -> None:
    """The legacy ``num_trials`` path must agree with ``fzb run`` too."""
    if not FIXTURE.is_file():
        pytest.skip(f"missing {FIXTURE}")
    pcfb = _pcfb(tmp_path, monkeypatch)

    reference = _fzb_run(pcfb, N)
    legacy = run_calculation([str(FIXTURE)], {"num_trials": N, "compilation_level": 2})
    assert legacy["trials"] == reference["n"]
    assert round(legacy["mean"] * legacy["trials"]) == reference["hits"]


def test_pass_geometry_changes_counts_today(tmp_path: Path, monkeypatch) -> None:
    """Characterizes the #223 gap: counts depend on the pass size right now.

    ``fzb run`` issues one pass of N; the server asked for N/16 trials per pass
    issues 16 passes. Until the engine's step-size invariance lands (#223 item 1)
    the two differ, and the difference is recorded here rather than papered over.
    """
    if not FIXTURE.is_file():
        pytest.skip(f"missing {FIXTURE}")
    pcfb = _pcfb(tmp_path, monkeypatch)

    reference = _fzb_run(pcfb, N)
    split = run_calculation(
        [str(FIXTURE)],
        {"compilation_level": 2, "requested_trials_per_pass": N // 16},
        stopping=StoppingSpec.model_validate({"trials": N}).resolve(),
    )
    assert split["trials"] == N
    assert split["passes"] == 16
    hits = round(split["mean"] * split["trials"])
    # The recorded magnitude of the current gap; if this ever becomes zero, #223
    # item 1 has landed and the exact-equivalence test can drop the geometry caveat.
    assert hits != reference["hits"]
    assert abs(hits - reference["hits"]) < 0.001 * N


def test_fixed_design_commits_whole_passes(tmp_path: Path, monkeypatch) -> None:
    """A budget that is not a whole number of passes stops at the next pass boundary."""
    if not FIXTURE.is_file():
        pytest.skip(f"missing {FIXTURE}")
    _pcfb(tmp_path, monkeypatch)
    budget = 3 * 4096 + 1  # 3 passes and one trial
    result = run_calculation(
        [str(FIXTURE)],
        {"compilation_level": 2, "requested_trials_per_pass": 4096, "num_trials": budget},
        stopping=StoppingSpec.model_validate({"trials": budget}).resolve(),
    )
    assert result["trials"] == 4 * 4096
    assert result["stop_reason"] == "trials"
