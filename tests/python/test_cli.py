"""Installed ``fzb`` console script (subprocess, not the function)."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402

CHINESE_REF = 1.17058e-3
Z_MAX = 4.0
INSPECT_JSON_KEYS = {
    "schema_version",
    "tensors",
    "operators",
    "samples",
    "queries",
    "opcodes",
    "sample_source",
    "query_names",
    "query_kinds",
    "metadata",
}
RUN_JSON_QUERY_KEYS = {"name", "estimate", "ci_low", "ci_high", "n", "hits"}


def _fzb() -> Path:
    name = "fzb.exe" if sys.platform == "win32" else "fzb"
    # Do not Path.resolve() sys.executable: venv python is often a symlink
    # into /usr/bin, which would miss the console script beside the venv.
    candidates = [
        Path(sys.executable).parent / name,
        Path(sys.executable).resolve().parent / name,
    ]
    for exe in candidates:
        if exe.is_file():
            return exe
    pytest.skip(f"fzb console script not installed next to {sys.executable}")


def _run_fzb(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("ACPP_VISIBILITY_MASK", "omp")
    env.setdefault("FUZZYBITS_BACKEND", "omp")
    return subprocess.run(
        [str(_fzb()), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture(scope="module", autouse=True)
def _omp_backend():
    fuzzybits.init(backend="omp")
    yield


def test_fzb_help_and_version():
    module_help = subprocess.run(
        [sys.executable, "-m", "fuzzybits._cli", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert module_help.returncode == 0, module_help.stderr
    assert "Usage: fzb" in module_help.stdout

    help_proc = _run_fzb("--help")
    assert help_proc.returncode == 0, help_proc.stderr
    assert "Usage: fzb" in help_proc.stdout
    assert "compile" in help_proc.stdout
    assert "export" in help_proc.stdout
    assert "inspect" in help_proc.stdout
    assert "run" in help_proc.stdout

    version_proc = _run_fzb("--version")
    assert version_proc.returncode == 0, version_proc.stderr
    assert version_proc.stdout.strip() == f"fzb {fuzzybits.__version__}"

    empty = _run_fzb()
    assert empty.returncode == 1
    assert "Usage: fzb" in empty.stderr

    unknown = _run_fzb("not-a-command")
    assert unknown.returncode == 1
    assert "unknown command" in unknown.stderr


def _native_options(source: str, function: str, end: str) -> set[str]:
    """Long option names of a boost::program_options chain in ``targets/fzb/main.cpp``."""
    start = source.index(function)
    body = source[start : source.index(end, start)]
    return set(re.findall(r'(?:add_options\(\)|\))\(\s*"([a-z][a-z0-9-]*)(?:,[a-z])?"', body))


def test_fzb_compile_options_match_the_native_cli():
    """The wheel's Python ``fzb compile`` takes exactly the options of the native one (docs/python.md)."""
    main_cpp = Path(__file__).resolve().parents[2] / "targets" / "fzb" / "main.cpp"
    if not main_cpp.is_file():
        pytest.skip("targets/fzb/main.cpp is not in this checkout")
    source = main_cpp.read_text(encoding="utf-8")
    native = _native_options(source, "int cmd_compile(", "add_progress_options(desc);")
    native |= _native_options(source, "void add_progress_options(", "\n}")
    native.discard("input-file")  # positional on both sides
    help_text = _run_fzb("compile", "--help").stdout
    python = set(re.findall(r"(?<![\w-])--([a-z][a-z0-9-]*)", help_text))
    assert "kind" in native and "hull-literals" in native
    assert python == native, (sorted(python - native), sorted(native - python))


def test_fzb_help_on_non_utf8_stdout():
    # Windows pipes default to cp1252, which has no "→"; --help used to raise
    # UnicodeEncodeError there. PYTHONIOENCODING reproduces it on any platform.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONUTF8"] = "0"
    proc = subprocess.run([str(_fzb()), "--help"], env=env, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "→ .pcfb" in proc.stdout.decode("utf-8")


def test_fzb_compile_inspect_run_chinese(chinese_xml: Path, tmp_path: Path):
    pcfb = tmp_path / "c.pcfb"
    compiled = _run_fzb("compile", str(chinese_xml), "-o", str(pcfb), cwd=tmp_path)
    assert compiled.returncode == 0, compiled.stderr
    assert pcfb.is_file()
    assert "wrote" in compiled.stderr

    inspected = _run_fzb("inspect", str(pcfb), "--json")
    assert inspected.returncode == 0, inspected.stderr
    payload = json.loads(inspected.stdout)
    assert INSPECT_JSON_KEYS <= set(payload)
    assert "diagnostics" not in payload
    assert int(payload["queries"]) >= 1

    ran = _run_fzb(
        "run",
        str(pcfb),
        "--trials",
        "1e6",
        "--backend",
        "omp",
        "--json",
    )
    assert ran.returncode == 0, ran.stderr
    result = json.loads(ran.stdout)
    queries = result["queries"]
    assert queries
    row = queries[0]
    assert RUN_JSON_QUERY_KEYS <= set(row)
    p_hat = float(row["estimate"])
    n = float(row["n"])
    sigma = math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / n) if n else float("nan")
    z = (p_hat - CHINESE_REF) / sigma if sigma > 0 else float("inf")
    assert abs(z) <= Z_MAX, (
        f"chinese p̂={p_hat:.6e} ref={CHINESE_REF:.6e} |z|={abs(z):.3f} n={n:.0f}"
    )


def test_fzb_run_seed_rejected(tiny_xml: Path, tmp_path: Path):
    pcfb = tmp_path / "tiny.pcfb"
    compiled = _run_fzb("compile", str(tiny_xml), "-o", str(pcfb), cwd=tmp_path)
    assert compiled.returncode == 0, compiled.stderr
    rejected = _run_fzb("run", str(pcfb), "--trials", "1024", "--seed", "1", "--backend", "omp")
    assert rejected.returncode == 1
    assert "--seed is not supported yet" in rejected.stderr

    accepted = _run_fzb(
        "run",
        str(pcfb),
        "--trials",
        "1024",
        "--seed",
        "372",
        "--backend",
        "omp",
        "--json",
    )
    assert accepted.returncode == 0, accepted.stderr
    payload = json.loads(accepted.stdout)
    assert payload["queries"]


def test_fzb_refuses_monotone_hull_query(hull_pcfb_path: Path):
    run = _run_fzb("run", str(hull_pcfb_path), "--trials", "4096", "--no-progress")
    assert run.returncode != 0
    assert "pcfb.unsupported-query-kind" in run.stderr
    assert "TOP_HULL" in run.stderr

    exact = _run_fzb("inspect", str(hull_pcfb_path), "--validate", "--exact")
    assert exact.returncode != 0
    assert "validate: ok" in exact.stderr
    assert "pcfb.unsupported-query-kind" in exact.stderr
    assert "exact_probability" not in exact.stdout
