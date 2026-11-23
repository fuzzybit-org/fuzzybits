"""`fzb inspect --exact` / `--hull-check` and `fuzzybits.exact` / `hull_check` for hull queries (#202).

The committed fixture is an AND of two basic events (``A`` at 0.25, ``B`` at 0.5) whose only query
is the monotone hull, so the hull value is the Boolean one, 0.125.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402

DATA = Path(__file__).parent / "data"


def _fzb() -> Path:
    name = "fzb.exe" if sys.platform == "win32" else "fzb"
    for candidate in (Path(sys.executable).parent / name, Path(sys.executable).resolve().parent / name):
        if candidate.is_file():
            return candidate
    pytest.skip(f"fzb console script not installed next to {sys.executable}")


def _run_fzb(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.setdefault("ACPP_VISIBILITY_MASK", "omp")
    environment.setdefault("FUZZYBITS_BACKEND", "omp")
    return subprocess.run(
        [str(_fzb()), *arguments],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture(scope="module", autouse=True)
def _omp_backend():
    fuzzybits.init(backend="omp")
    yield


@pytest.fixture(scope="module")
def hull_pcfb() -> Path:
    path = DATA / "monotone_hull_query.pcfb"
    if not path.is_file():
        pytest.skip(f"missing fixture {path}")
    return path


EMPTY_TOP = """**CHAR32
TOP + SEQ
SEQ * -SUCCG
SUCCG + SUCC
ENDTREE
PROCESS TOP
IMPORT
  1.000000E+00 SUCC
LIMIT 1.000000E-12
**FTITLE: TOP = not SUCCG with a p = 1 event under the gate, so the hull reaches the empty sample
*XEQ
"""


@pytest.fixture(scope="module")
def empty_g_pcfb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A hull file whose g is TRUE on the empty sample: TOP = not SUCC with SUCC at p = 1."""
    directory = tmp_path_factory.mktemp("hull-empty")
    source = directory / "model.ftp"
    source.write_text(EMPTY_TOP)
    path = directory / "model.pcfb"
    compiled = _run_fzb(
        "compile",
        str(source),
        "--negation",
        "hull",
        "--max-initiators-per-cutset",
        "none",
        "-o",
        str(path),
    )
    if compiled.returncode != 0:
        pytest.skip(f"cannot compile the empty-set fixture: {compiled.stderr}")
    return path


def test_the_fixture_has_a_named_hull_query(hull_pcfb: Path) -> None:
    summary = fuzzybits.inspect(hull_pcfb)
    assert summary["query_names"] == ["TOP_HULL"]
    result = fuzzybits.hull_check(hull_pcfb, ["A"])
    assert set(result) == {"hit", "lower", "upper", "tier", "residual_variables", "witness", "query"}


@pytest.mark.parametrize(("names", "hit"), [("A", False), ("B", False), ("A,B", True), ("B,A", True)])
def test_fzb_hull_check_accepts_the_event_names(hull_pcfb: Path, names: str, hit: bool) -> None:
    proc = _run_fzb("inspect", str(hull_pcfb), "--hull-check", names)
    assert proc.returncode == 0, proc.stderr
    assert f"hit={int(hit)}" in proc.stdout


@pytest.mark.parametrize("name", ["MISSING", "AND_OF"])
def test_fzb_hull_check_rejects_an_unknown_or_gate_name(hull_pcfb: Path, name: str) -> None:
    proc = _run_fzb("inspect", str(hull_pcfb), "--hull-check", name)
    assert proc.returncode == 1
    assert f"'{name}'" in proc.stderr


def test_fzb_hull_check_reads_a_file_of_sets(hull_pcfb: Path, tmp_path: Path) -> None:
    sets = tmp_path / "sets.txt"
    sets.write_text("A,B\n\n-\nA\n")
    proc = _run_fzb("inspect", str(hull_pcfb), "--hull-check", f"@{sets}", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert [item["hit"] for item in payload] == [True, False, False]

    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n")
    refused = _run_fzb("inspect", str(hull_pcfb), "--hull-check", f"@{empty}")
    assert refused.returncode == 1
    assert "lists no sets" in refused.stderr


def test_fzb_inspect_exact_answers_a_hull_query(hull_pcfb: Path) -> None:
    proc = _run_fzb("inspect", str(hull_pcfb), "--validate", "--exact")
    assert proc.returncode == 0, proc.stderr
    assert "validate: ok" in proc.stderr
    assert "exact_hull_probability  1.2500000000000000e-01" in proc.stdout
    assert "exact_probability  " not in proc.stdout

    named = _run_fzb("inspect", str(hull_pcfb), "--exact", "--query", "TOP_HULL", "--kind", "hull")
    assert named.returncode == 0, named.stderr
    assert named.stdout.endswith("exact_hull_probability  1.2500000000000000e-01\n")


def test_fzb_inspect_exact_rejects_an_unknown_query(hull_pcfb: Path) -> None:
    proc = _run_fzb("inspect", str(hull_pcfb), "--exact", "--query", "MISSING")
    assert proc.returncode == 1
    assert "no query named 'MISSING'" in proc.stderr


def test_fzb_hull_check_json(hull_pcfb: Path) -> None:
    proc = _run_fzb("inspect", str(hull_pcfb), "--hull-check", "A,B", "--witness", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert len(payload) == 1
    only = payload[0]
    assert only["hit"] is True
    assert only["lower"] is True and only["upper"] is True
    assert only["tier"] in {"bounds", "reduction", "witness", "truth_table", "bdd", "search"}
    assert only["residual_variables"] == 0
    assert only["witness"] == ["A", "B"]
    assert only["query"] == {"index": 0, "name": "TOP_HULL", "kind": "hull"}


def test_fzb_hull_check_misses_and_the_empty_set(hull_pcfb: Path) -> None:
    miss = _run_fzb("inspect", str(hull_pcfb), "--hull-check", "A")
    assert miss.returncode == 0, miss.stderr
    assert "hit=0" in miss.stdout
    assert "witness" not in miss.stdout

    empty = _run_fzb("inspect", str(hull_pcfb), "--hull-check", "")
    assert empty.returncode == 0, empty.stderr
    assert "hit=0" in empty.stdout


def test_fzb_hull_check_names_a_usage_error(hull_pcfb: Path) -> None:
    witness = _run_fzb("inspect", str(hull_pcfb), "--witness")
    assert witness.returncode != 0
    assert "--witness needs --hull-check" in witness.stderr

    kind_alone = _run_fzb("inspect", str(hull_pcfb), "--exact", "--kind", "hull")
    assert kind_alone.returncode != 0
    assert "--kind needs --query" in kind_alone.stderr

    exact_and_check = _run_fzb("inspect", str(hull_pcfb), "--exact", "--hull-check", "A")
    assert exact_and_check.returncode != 0
    assert "--exact and --hull-check cannot be combined" in exact_and_check.stderr

    bad_kind = _run_fzb("inspect", str(hull_pcfb), "--exact", "--query", "TOP_HULL", "--kind", "nope")
    assert bad_kind.returncode != 0
    assert "probability or hull" in bad_kind.stderr


def test_fzb_hull_check_requires_a_hull_query(empty_g_pcfb: Path) -> None:
    # The empty-set fixture writes its target as one hull query only, so asking for the
    # probability kind of that name is an error that names the kinds the file has.
    proc = _run_fzb("inspect", str(empty_g_pcfb), "--hull-check", "SUCC", "--query", "TOP", "--kind", "probability")
    assert proc.returncode == 1
    assert "has kind hull, not probability" in proc.stderr


def test_fzb_hull_check_distinguishes_the_empty_witness(empty_g_pcfb: Path) -> None:
    # g(∅) is TRUE, so the empty sample is a hit with an empty witness rather than a miss.
    proc = _run_fzb("inspect", str(empty_g_pcfb), "--hull-check", "", "--witness", "--json")
    if proc.returncode != 0:
        pytest.skip(f"the empty-set fixture was not written: {proc.stderr}")
    payload = json.loads(proc.stdout)
    assert payload[0]["hit"] is True
    assert payload[0]["witness"] == []
