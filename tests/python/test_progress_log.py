"""``progress_log=`` / ``--progress-log``: JSON Lines and CSV snapshot logs (part of #205, #208)."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402

# One device pass of chinese is 32·2^20 trials; keep the run to a few passes.
TRIALS = 1 << 22


@pytest.fixture(scope="module", autouse=True)
def _omp_backend():
    fuzzybits.init(backend="omp")
    yield


@pytest.fixture(scope="module")
def chinese_pcfb(chinese_xml: Path, tmp_path_factory: pytest.TempPathFactory) -> bytes:
    return fuzzybits.compile([chinese_xml])


@pytest.fixture(scope="module")
def chinese_model(chinese_pcfb: bytes) -> bytes:
    return chinese_pcfb


def _jsonl_records(path: Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, f"{path} is empty"
    return [json.loads(line) for line in lines]


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check_jsonl(path: Path, rows) -> None:
    records = _jsonl_records(path)
    snaps = [r for r in records if r["event"] == "snapshot"]
    finishes = [r for r in records if r["event"] == "finish"]
    assert snaps, "no snapshot record"
    assert len(finishes) == 1, "expected exactly one finish record"
    assert records[-1]["event"] == "finish", "the finish record must be last"
    for record in records:
        assert record["schema"] == 1
        assert isinstance(record["phase"], str)
        assert isinstance(record["step"], str)
    done = [int(r["trials_done"]) for r in snaps]
    assert done == sorted(done), f"trials_done is not monotone: {done}"
    assert all(int(r["trials_done"]) >= 0 for r in snaps)
    assert all(int(r["passes_done"]) >= 0 for r in snaps)

    last = records[-1]
    assert last["queries"], "the finish record carries the queries"
    assert len(last["queries"]) == len(rows)
    for got, want in zip(last["queries"], rows, strict=True):
        assert got["name"] == want.name
        assert got["hits"] == want.hits
        assert got["trials"] == want.n
        assert got["estimate"] == want.estimate
        assert got["ci_low"] == want.ci_low
        assert got["ci_high"] == want.ci_high
    # NaN must never appear as bare text in a JSON Lines file.
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "nan" not in text and "Infinity" not in text


def _check_csv(path: Path, rows) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(csv.reader([lines[0]]))
    for column in (
        "record",
        "event",
        "interrupted",
        "schema",
        "phase",
        "step",
        "trials_done",
        "trials_total",
        "passes_done",
        "passes_total",
        "elapsed_seconds",
        "trials_per_second",
        "query_index",
        "query_name",
        "hits",
        "trials",
        "estimate",
        "ci_low",
        "ci_high",
        "rel_half_width",
        "target_rel_half_width",
        "converged",
    ):
        assert column in header, f"missing CSV column {column}"

    records = _csv_rows(path)
    assert records, "no CSV rows"
    for record in records:
        assert None not in record.values(), f"row has fewer fields than the header: {record}"
        assert len(record) == len(header)
        assert record["schema"] == "1"

    snaps = [r for r in records if r["event"] == "snapshot"]
    finishes = [r for r in records if r["event"] == "finish"]
    assert snaps, "no snapshot row"
    assert len(finishes) == len(rows), "one finish row per query"
    assert records[-1]["event"] == "finish", "the finish rows must be last"

    done = [int(r["trials_done"]) for r in snaps]
    assert done == sorted(done), f"trials_done is not monotone: {done}"

    # Long format: every snapshot scalar repeats for each query of one record.
    by_record: dict[int, list[dict[str, str]]] = {}
    for record in records:
        by_record.setdefault(int(record["record"]), []).append(record)
    for index in sorted(by_record):
        group = by_record[index]
        assert len({(r["trials_done"], r["elapsed_seconds"], r["phase"]) for r in group}) == 1

    final_rows = by_record[max(by_record)]
    assert len(final_rows) == len(rows)
    for got, want in zip(final_rows, rows, strict=True):
        assert got["query_name"] == want.name
        assert int(got["hits"]) == want.hits
        assert int(got["trials"]) == want.n
        assert float(got["estimate"]) == want.estimate
        assert float(got["ci_low"]) == want.ci_low
        assert float(got["ci_high"]) == want.ci_high
    # A missing double is an empty cell, never the text "nan".
    assert "nan" not in path.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(("suffix", "checker"), [(".jsonl", _check_jsonl), (".ndjson", _check_jsonl), (".csv", _check_csv)])
def test_run_progress_log(chinese_model: bytes, tmp_path: Path, suffix: str, checker) -> None:
    log = tmp_path / f"run{suffix}"
    rows = fuzzybits.run(chinese_model, trials=TRIALS, backend="omp", progress_log=log)
    assert log.is_file()
    assert rows
    checker(log, rows)


def test_run_progress_log_accepts_pathlike(chinese_model: bytes, tmp_path: Path) -> None:
    log = tmp_path / "pathlike.jsonl"
    rows = fuzzybits.run(chinese_model, trials=TRIALS, backend="omp", progress_log=Path(log))
    assert rows
    assert log.is_file()
    assert _jsonl_records(log)[-1]["queries"]


def test_run_progress_log_creates_nothing_by_default(chinese_model: bytes, tmp_path: Path) -> None:
    before = sorted(p.name for p in tmp_path.iterdir())
    fuzzybits.run(chinese_model, trials=TRIALS, backend="omp")
    assert sorted(p.name for p in tmp_path.iterdir()) == before


@pytest.mark.parametrize("suffix", [".txt", "", ".json"])
def test_run_progress_log_rejects_unknown_extension(
    chinese_model: bytes, tmp_path: Path, suffix: str
) -> None:
    log = tmp_path / f"run{suffix}"
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        fuzzybits.run(chinese_model, trials=TRIALS, backend="omp", progress_log=log)
    assert "extension" in str(excinfo.value).lower()
    assert not log.exists()


def test_run_progress_log_missing_directory(chinese_model: bytes, tmp_path: Path) -> None:
    log = tmp_path / "does-not-exist" / "run.jsonl"
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        fuzzybits.run(chinese_model, trials=TRIALS, backend="omp", progress_log=log)
    assert "progress log" in str(excinfo.value).lower()
    assert not log.exists()


def test_run_progress_log_empty_path(chinese_model: bytes) -> None:
    with pytest.raises(ValueError):
        fuzzybits.run(chinese_model, trials=TRIALS, backend="omp", progress_log="")


def test_compile_progress_log(chinese_xml: Path, tmp_path: Path) -> None:
    log = tmp_path / "compile.jsonl"
    blob = fuzzybits.compile([chinese_xml], progress_log=log)
    assert blob
    assert log.is_file()
    records = _jsonl_records(log)
    assert records[-1]["event"] == "finish"
    assert all(r["schema"] == 1 for r in records)
    assert all(r["phase"] in {"compile", "done"} for r in records)
    # Compile snapshots carry no queries; the finish record closes the file.
    assert records[-1]["queries"] == []


def test_compile_progress_log_csv(chinese_xml: Path, tmp_path: Path) -> None:
    log = tmp_path / "compile.csv"
    assert fuzzybits.compile([chinese_xml], progress_log=log)
    rows = _csv_rows(log)
    assert rows
    assert rows[-1]["event"] == "finish"
    assert all(len(row) == len(rows[0]) for row in rows)


def test_fzb_run_progress_log_flag(chinese_pcfb: bytes, tmp_path: Path) -> None:
    """The Python console script passes ``--progress-log`` through."""
    name = "fzb.exe" if sys.platform == "win32" else "fzb"
    exe = Path(sys.executable).parent / name
    if not exe.is_file():
        pytest.skip(f"fzb console script not installed next to {sys.executable}")
    pcfb = tmp_path / "chinese.pcfb"
    pcfb.write_bytes(chinese_pcfb)
    log = tmp_path / "cli.jsonl"
    proc = subprocess.run(
        [str(exe), "run", str(pcfb), "--trials", "1e6", "--backend", "omp", "--progress-log", str(log)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert log.is_file()
    records = _jsonl_records(log)
    assert records[-1]["event"] == "finish"
    assert records[-1]["queries"]
    assert math.isfinite(float(records[-1]["queries"][0]["estimate"]))


def test_fzb_run_progress_log_unknown_extension(chinese_pcfb: bytes, tmp_path: Path) -> None:
    name = "fzb.exe" if sys.platform == "win32" else "fzb"
    exe = Path(sys.executable).parent / name
    if not exe.is_file():
        pytest.skip(f"fzb console script not installed next to {sys.executable}")
    pcfb = tmp_path / "chinese.pcfb"
    pcfb.write_bytes(chinese_pcfb)
    log = tmp_path / "cli.txt"
    proc = subprocess.run(
        [str(exe), "run", str(pcfb), "--trials", "1e6", "--backend", "omp", "--progress-log", str(log)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "extension" in proc.stderr.lower()
    assert not log.exists()
