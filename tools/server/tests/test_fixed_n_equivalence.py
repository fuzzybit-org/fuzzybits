"""Fixed-N equivalence: a server run against ``fzb run -N`` on a public fixture.

The claim under test is the #223 consequence the controller relies on: for a
fixed design, the server's per-query counts equal those of ``fzb run -N`` with
the same N *and the same pass size*.

The reference is ``fzb run -N`` when the CLI is usable, otherwise
``fuzzybits.run(trials=N)``, which is the same one-shot path the CLI wraps; the
test's failure message names the reference it used. Set ``FZB_REFERENCE=cli`` to
insist on the CLI (an unusable CLI then skips instead of falling back).

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
point at the ``fzb`` executable.
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


def _reference_run(pcfb: Path, trials: int) -> dict:
    """The reference a fixed-N run is compared against.

    ``fuzzybits.run`` is the one-shot path ``fzb run`` is a client of (#223:
    "the existing one-shot run_pcfb becomes a thin fixed-N client of the same
    loop"), so it is the reference when the CLI is unavailable. ``FZB_REFERENCE=cli``
    forces the CLI and turns an unavailability into a skip rather than a silent
    fallback, which is what CI wants to assert.

    The returned ``source`` always names the path that actually produced the
    numbers; :func:`_assert_reference_ran` then refuses a result that could not
    have come from a real run, so a broken reference cannot pass the comparison.
    """
    import fuzzybits

    want_cli = os.environ.get("FZB_REFERENCE", "").strip().lower() == "cli"
    binary = _fzb_binary()
    if binary is not None:
        completed = subprocess.run(
            [binary, "run", str(pcfb), "--trials", str(trials), "--json", "--no-progress"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "ACPP_VISIBILITY_MASK": os.environ.get("ACPP_VISIBILITY_MASK", "omp")},
        )
        if completed.returncode == 0:
            query = json.loads(completed.stdout)["queries"][0]
            return {"name": query["name"], "n": query["n"], "hits": query["hits"], "source": binary}
        # The CLI is a SYCL binary; some container runtimes fault it on startup
        # (SIGILL under the GitHub Actions container runtime). Fall back, and say
        # which path was used so a failure names the reference it compared against.
        if want_cli:
            pytest.skip(f"fzb {binary} is unusable here: exit {completed.returncode}")
        reason = f"fzb exited {completed.returncode}"
    else:
        if want_cli:
            pytest.skip("FZB_REFERENCE=cli but no fzb binary was found")
        reason = "no fzb binary"

    result = fuzzybits.run(pcfb.read_bytes(), trials=trials)[0]
    return {
        "name": result.name,
        "n": result.n,
        "hits": result.hits,
        "source": f"fuzzybits.run ({reason})",
    }


def _assert_reference_ran(reference: dict, trials: int) -> None:
    """Refuse a reference result that no real run could have produced.

    Without this the comparison could pass vacuously: an empty or zeroed
    reference would make ``hits == reference["hits"]`` either trivially true
    (both zero) or fail for the wrong reason.
    """
    assert reference["source"], "the reference did not say which path produced its counts"
    assert reference["n"] == trials, (
        f"the reference covered {reference['n']} trials, not {trials} "
        f"(reference from {reference['source']})"
    )
    assert reference["hits"] > 0, (
        f"the reference reported no hits at all, so it did not sample "
        f"(reference from {reference['source']})"
    )


def _server_run(**stopping) -> dict:
    spec = StoppingSpec.model_validate(stopping).resolve()
    return run_calculation([str(FIXTURE)], {"compilation_level": 2}, stopping=spec)


def test_fixed_design_matches_fzb_run_at_the_same_pass_size(tmp_path: Path, monkeypatch) -> None:
    if not FIXTURE.is_file():
        pytest.skip(f"missing {FIXTURE}")
    pcfb = _pcfb(tmp_path, monkeypatch)

    reference = _reference_run(pcfb, N)
    _assert_reference_ran(reference, N)
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

    reference = _reference_run(pcfb, N)
    _assert_reference_ran(reference, N)
    legacy = run_calculation([str(FIXTURE)], {"num_trials": N, "compilation_level": 2})
    assert legacy["trials"] == reference["n"]
    assert round(legacy["mean"] * legacy["trials"]) == reference["hits"]


def test_pass_geometry_changes_counts_today(tmp_path: Path, monkeypatch) -> None:
    """Characterizes the #223 gap: counts depend on the pass size right now.

    ``fzb run`` issues one pass of N; the server asked for N/16 trials per pass
    issues 16 passes. Until the engine's step-size invariance lands (#223 item 1)
    the two can differ, and the difference is measured and bounded here rather
    than papered over. The test does *not* require a difference: when #223 lands
    the counts will agree, and that is the outcome this test documents rather
    than a failure.
    """
    if not FIXTURE.is_file():
        pytest.skip(f"missing {FIXTURE}")
    pcfb = _pcfb(tmp_path, monkeypatch)

    reference = _reference_run(pcfb, N)
    _assert_reference_ran(reference, N)
    split = run_calculation(
        [str(FIXTURE)],
        {"compilation_level": 2, "requested_trials_per_pass": N // 16},
        stopping=StoppingSpec.model_validate({"trials": N}).resolve(),
    )
    assert split["trials"] == N
    assert split["passes"] == 16
    hits = round(split["mean"] * split["trials"])
    # Both are unbiased draws at the same N, so they can only differ by the
    # pass-boundary index assignment, not by sampling noise: bound that to 0.1%
    # of N (measured at 3.4e-4 of N on the tiny fixture with 16 passes).
    assert abs(hits - reference["hits"]) <= 0.001 * N, (
        f"pass-geometry difference is larger than recorded: "
        f"{reference['source']} {reference['hits']} vs 16 passes {hits}"
    )
    if hits == reference["hits"]:
        pytest.skip("pass geometry no longer changes counts: #223 item 1 has landed")


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
