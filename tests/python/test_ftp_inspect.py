"""``fuzzybits.read_project`` and ``fzb inspect`` on CAFTA FTP projects (#201)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402

from test_cli import _run_fzb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFORMANCE = ROOT / "input" / "ftp" / "conformance"
FIXTURES = sorted(path for path in CONFORMANCE.iterdir() if (path / "expected.json").is_file())

CLEAN = "TOP + A B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\n"
WARNING = "TOP + A B\nENDTREE\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\n"
BROKEN = "TOP + A B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n"
NAMED = (
    "**DBNAME: WOMBAT_HEADER\nZEBRA_TOP * QUOKKA_G -QUOKKA_B\nQUOKKA_G + QUOKKA_B QUOKKA_C\nENDTREE\n"
    "PROCESS ZEBRA_TOP\nIMPORT\n  1.0E-01 QUOKKA_B\n  2.0E-01 QUOKKA_C\n  3.0E-01 QUOKKA_B\n"
)
NAMES = ("ZEBRA_TOP", "QUOKKA_G", "QUOKKA_B", "QUOKKA_C", "WOMBAT_HEADER")


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _same(actual, expected, policy: str, path: str = "structure") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        for key, value in expected.items():
            assert key in actual, f"{path}.{key}"
            if path == "structure" and key == "duplicate_imports":
                _same_duplicates(actual[key], value, policy)
            else:
                _same(actual[key], value, policy, f"{path}.{key}")
    elif isinstance(expected, list):
        assert actual == expected, path
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12), path
    else:
        assert actual == expected, path


def _same_duplicates(actual: dict, expected: dict, policy: str) -> None:
    assert list(actual) == list(expected)
    for name, records in expected.items():
        chosen = records[0] if policy == "first" else records[-1]
        listed = {(record["file"], record["line"]) for record in records}
        assert len(actual[name]) == len(records), name
        assert [item for item in actual[name] if item["chosen"]] and sum(item["chosen"] for item in actual[name]) == 1
        for item in actual[name]:
            place = (Path(item["file"]).name, item["line"])
            assert place in listed, name
            if item["chosen"]:
                assert place == (chosen["file"], chosen["line"]), name
                assert item["value"] == chosen["value"], name


def _expected_ids(outcome: dict, spec: dict) -> list[str]:
    ids = [f"{item['severity']} io.ftp.{item['code']}" for item in outcome["diagnostics"]]
    structure = spec.get("structure")
    if structure is not None and not structure["process_targets"] and "warning io.ftp.missing-process" not in ids:
        ids.append("warning io.ftp.missing-process")
    return sorted(ids)


def _ids(diagnostics: list[dict]) -> list[str]:
    return sorted(f"{item['severity']} {item['id']}" for item in diagnostics if item["severity"] != "note")


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda path: path.name)
def test_read_project_matches_every_conformance_fixture(fixture: Path):
    spec = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    files = [str(fixture / name) for name in spec["files"]]
    for policy, outcome in spec["policies"].items():
        expected = _expected_ids(outcome, spec)
        if outcome["results"] is None:
            with pytest.raises(fuzzybits.FtpProjectError) as failure:
                fuzzybits.read_project(files, duplicate_import=policy)
            assert _ids(failure.value.diagnostics) == expected, policy
            continue
        document = fuzzybits.read_project(files, duplicate_import=policy)
        assert document["status"] == "ok"
        assert _ids(document["diagnostics"]) == expected, policy
        assert len(document["files"]) == len(files)
        _same(document["structure"], spec["structure"], policy)
        assert document["structure"]["default_targets"] == spec["default_targets"]


def test_read_project_accepts_a_directory_a_path_and_a_list():
    fixture = CONFORMANCE / "multi_file_project"
    by_directory = fuzzybits.read_project(fixture)
    by_files = fuzzybits.read_project(sorted(fixture.glob("*.ftp")))
    assert by_directory == by_files
    assert [Path(name).name for name in by_directory["files"]] == ["01_top.ftp", "02_systems.ftp", "03_imports.ftp"]


def test_read_project_error_carries_the_diagnostics(tmp_path: Path):
    path = _write(tmp_path, "broken.ftp", BROKEN)
    with pytest.raises(fuzzybits.FtpProjectError) as failure:
        fuzzybits.read_project(path)
    assert isinstance(failure.value, RuntimeError)
    diagnostics = failure.value.diagnostics
    assert [item["id"] for item in diagnostics] == ["io.ftp.undefined-reference"]
    assert diagnostics[0]["severity"] == "error"
    assert Path(diagnostics[0]["file"]).name == "broken.ftp"
    assert diagnostics[0]["line"] == 1
    assert "io.ftp.undefined-reference" in str(failure.value)


def test_read_project_warnings_are_returned_not_raised(tmp_path: Path):
    document = fuzzybits.read_project(_write(tmp_path, "warning.ftp", WARNING))
    assert document["status"] == "ok"
    assert [item["id"] for item in document["diagnostics"]] == ["io.ftp.missing-process"]
    assert document["diagnostics"][0]["file"] is None


def test_read_project_conventions_and_structure(tmp_path: Path):
    text = "**CHAR32\n**DBNAME: sample\nTOP * G -B -H\nG + A B\nH + A B\nENDTREE\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\nLIMIT 1.0E-09\nMAXSIZE 3\n"
    document = fuzzybits.read_project(_write(tmp_path, "s.ftp", text))
    assert document["conventions"] == {
        "LIMIT": "1.0E-09",
        "MAXSIZE": "3",
        "header_lines": ["**CHAR32", "**DBNAME: sample"],
    }
    structure = document["structure"]
    assert structure["complemented_edges_to_gates"] == 1
    assert structure["literal_operands"] == 1
    assert structure["limit"] == 1e-9


def test_read_project_duplicate_imports_name_the_chosen_record(tmp_path: Path):
    text = "TOP + A B\nENDTREE\nPROCESS TOP\nIMPORT\n  3.0E-01 A\n  1.0E-01 B\n  2.0E-01 A\n"
    path = _write(tmp_path, "d.ftp", text)
    for policy, line, value in (("last", 7, 0.2), ("first", 5, 0.3)):
        records = fuzzybits.read_project(path, duplicate_import=policy)["structure"]["duplicate_imports"]["A"]
        assert [record["line"] for record in records] == [5, 7]
        assert [record["chosen"] for record in records] == [record["line"] == line for record in records]
        assert [record["value"] for record in records if record["chosen"]] == [value]


def test_read_project_counts_only_has_no_names(tmp_path: Path):
    document = fuzzybits.read_project(_write(tmp_path, "m.ftp", NAMED), counts_only=True)
    text = json.dumps(document)
    for name in NAMES:
        assert name not in text
    assert document["structure"]["roots"] == 1
    assert document["structure"]["duplicate_imports"] == 1
    assert all("message" not in item for item in document["diagnostics"])


def test_read_project_rejects_bad_arguments(tmp_path: Path):
    path = _write(tmp_path, "a.ftp", CLEAN)
    with pytest.raises(ValueError, match="duplicate-import"):
        fuzzybits.read_project(path, duplicate_import="newest")
    # Removed: -EVENT is always a marked literal.
    with pytest.raises(TypeError, match="complement_events"):
        fuzzybits.read_project(path, complement_events="negation")
    with pytest.raises(ValueError):
        fuzzybits.read_project([])


def test_cli_text_and_json(tmp_path: Path):
    path = _write(tmp_path, "model.ftp", CLEAN)
    text = _run_fzb("inspect", str(path))
    assert text.returncode == 0, text.stderr
    assert "status: ok" in text.stdout
    assert "gates: 1 (or 1, and 0, atleast 0)" in text.stdout
    as_json = _run_fzb("inspect", str(path), "--json")
    assert as_json.returncode == 0, as_json.stderr
    assert json.loads(as_json.stdout) == fuzzybits.read_project(path)


def test_cli_exit_status_and_strict(tmp_path: Path):
    clean = _write(tmp_path, "clean.ftp", CLEAN)
    warning = _write(tmp_path, "warning.ftp", WARNING)
    broken = _write(tmp_path, "broken.ftp", BROKEN)
    assert _run_fzb("inspect", str(clean), "--strict").returncode == 0
    assert _run_fzb("inspect", str(warning)).returncode == 0
    assert _run_fzb("inspect", str(warning), "--strict").returncode == 1
    failed = _run_fzb("inspect", str(broken), "--json")
    assert failed.returncode == 1
    document = json.loads(failed.stdout)
    assert document["status"] == "error"
    assert document["structure"] is None
    duplicates = str(CONFORMANCE / "imports_tags_duplicates")
    assert _run_fzb("inspect", duplicates, "--duplicate-import", "first").returncode == 0
    assert _run_fzb("inspect", duplicates, "--duplicate-import", "error").returncode == 1


def test_cli_counts_only_directory_and_file_list(tmp_path: Path):
    project = tmp_path / "wombat"
    project.mkdir()
    _write(project, "m.ftp", NAMED)
    for extra in ((), ("--json",)):
        run = _run_fzb("inspect", str(project), "--counts-only", *extra)
        assert run.returncode == 0, run.stderr
        for name in NAMES:
            assert name not in run.stdout
            assert name not in run.stderr
        assert "io.ftp.duplicate-import" in run.stdout
    assert "ZEBRA_TOP" in _run_fzb("inspect", str(project)).stdout

    fixture = CONFORMANCE / "multi_file_project"
    names = sorted(str(path) for path in fixture.glob("*.ftp"))
    assert _run_fzb("inspect", str(fixture), "--json").stdout == _run_fzb("inspect", *names, "--json").stdout


def test_cli_usage_and_option_errors(tmp_path: Path):
    path = _write(tmp_path, "model.ftp", CLEAN)
    other = _write(tmp_path, "other.xml", "<opsa-mef/>\n")
    run = _run_fzb("inspect", str(path), "--duplicate-import", "newest")
    assert run.returncode == 1
    assert "fzb inspect: --duplicate-import must be error, first or last" in run.stderr
    run = _run_fzb("inspect", str(path), "--complement-events", "negation")
    assert run.returncode == 1
    assert "complement-events" in run.stderr
    run = _run_fzb("inspect", str(path), "--exact")
    assert run.returncode == 1
    assert "--exact applies to .pcfb input only" in run.stderr
    run = _run_fzb("inspect", str(path), str(other))
    assert run.returncode == 1
    assert "FTP and non-FTP inputs cannot be combined" in run.stderr
    run = _run_fzb("inspect", str(other), "--counts-only")
    assert run.returncode == 1
    assert "--counts-only applies to CAFTA FTP input only" in run.stderr
    assert _run_fzb("inspect").returncode == 1
