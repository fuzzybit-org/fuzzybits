"""``fzb compile`` / ``fuzzybits.compile``: CCF expansion by default and several targets per compile."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402

from test_cli import _run_fzb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CCF_MODEL = ROOT / "tests" / "input" / "core" / "beta_factor_ccf.xml"
FTP_MODEL = "TOP + G1 E3\nG1 * E1 E2\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 E1\n  2.0E-01 E2\n  3.0E-01 E3\n"
DEPRECATED = (
    "fzb compile: --ccf is deprecated: common-cause groups are expanded by default; use --no-ccf to disable"
)
ONE_TARGET = "fzb compile: MEF and nombrilo inputs take one --target; several targets need the new MEF reader (#148)"
FTP_FLAGS = ("--negation", "boolean", "--max-initiators-per-cutset", "none", "--no-progress")


@pytest.fixture()
def ftp_model(tmp_path: Path) -> Path:
    path = tmp_path / "model.ftp"
    path.write_text(FTP_MODEL, encoding="utf-8")
    return path


def _names(blob: bytes) -> list[str]:
    return list(fuzzybits.inspect(blob)["query_names"])


def _metadata(blob: bytes) -> dict:
    return fuzzybits.inspect(blob)["metadata"]


def _compile_cli(tmp_path: Path, *args: str, name: str = "out.pcfb"):
    out = tmp_path / name
    out.unlink(missing_ok=True)
    return _run_fzb("compile", *args, "-o", str(out), cwd=tmp_path), out


# CCF


def test_api_expands_ccf_groups_by_default():
    expanded = fuzzybits.compile(str(CCF_MODEL))
    plain = fuzzybits.compile(str(CCF_MODEL), ccf=False)
    assert fuzzybits.compile(str(CCF_MODEL), ccf=True) == expanded
    assert fuzzybits.inspect(expanded)["samples"] > fuzzybits.inspect(plain)["samples"]
    assert not math.isclose(fuzzybits.exact(expanded), fuzzybits.exact(plain), rel_tol=1e-6)


def test_api_records_ccf_in_compile_metadata():
    on = _metadata(fuzzybits.compile(str(CCF_MODEL)))
    off = _metadata(fuzzybits.compile(str(CCF_MODEL), ccf=False))
    assert "--no-ccf" not in on["compile.options"]
    assert "--no-ccf" in off["compile.options"]
    assert str(on["compile.ccf"]).lower() in {"1", "true"}
    assert str(off["compile.ccf"]).lower() in {"0", "false"}


def test_api_ccf_is_ignored_for_ftp(ftp_model: Path):
    options = {"negation": "boolean", "max_initiators_per_cutset": None}
    assert fuzzybits.compile(str(ftp_model), ccf=False, **options) == fuzzybits.compile(str(ftp_model), **options)


def test_cli_default_and_no_ccf(tmp_path: Path):
    ran, expanded = _compile_cli(tmp_path, str(CCF_MODEL), "--no-progress", name="expanded.pcfb")
    assert ran.returncode == 0, ran.stderr
    assert "deprecated" not in ran.stderr
    ran, plain = _compile_cli(tmp_path, str(CCF_MODEL), "--no-ccf", "--no-progress", name="plain.pcfb")
    assert ran.returncode == 0, ran.stderr
    assert expanded.read_bytes() != plain.read_bytes()
    assert expanded.read_bytes() == fuzzybits.compile(str(CCF_MODEL), no_kn=False)
    assert plain.read_bytes() == fuzzybits.compile(str(CCF_MODEL), no_kn=False, ccf=False)


def test_cli_ccf_is_a_deprecated_noop(tmp_path: Path):
    ran, legacy = _compile_cli(tmp_path, str(CCF_MODEL), "--ccf", "--no-progress")
    assert ran.returncode == 0, ran.stderr
    assert ran.stderr.count(DEPRECATED) == 1
    assert legacy.read_bytes() == fuzzybits.compile(str(CCF_MODEL), no_kn=False)


def test_cli_ccf_with_no_ccf_is_an_error(tmp_path: Path):
    ran, out = _compile_cli(tmp_path, str(CCF_MODEL), "--ccf", "--no-ccf", "--no-progress")
    assert ran.returncode == 1
    assert "--ccf and --no-ccf cannot be combined" in ran.stderr
    assert not out.exists()


def test_cli_accepts_ccf_flags_for_ftp(tmp_path: Path, ftp_model: Path):
    ran, base = _compile_cli(tmp_path, str(ftp_model), *FTP_FLAGS)
    assert ran.returncode == 0, ran.stderr
    expected = base.read_bytes()
    ran, out = _compile_cli(tmp_path, str(ftp_model), "--no-ccf", *FTP_FLAGS)
    assert ran.returncode == 0, ran.stderr
    assert "deprecated" not in ran.stderr
    assert out.read_bytes() == expected
    ran, out = _compile_cli(tmp_path, str(ftp_model), "--ccf", *FTP_FLAGS)
    assert ran.returncode == 0, ran.stderr
    assert ran.stderr.count(DEPRECATED) == 1
    assert out.read_bytes() == expected
    ran, _ = _compile_cli(tmp_path, str(ftp_model), "--ccf", "--no-ccf", *FTP_FLAGS)
    assert ran.returncode == 1


# Targets


def test_api_ftp_target_and_watch_sequences(ftp_model: Path):
    options = {"negation": "boolean", "max_initiators_per_cutset": None}
    blob = fuzzybits.compile(str(ftp_model), target=["TOP", "G1"], watch=["E3", "E1"], **options)
    assert _names(blob) == ["TOP", "G1", "E3", "E1"]
    assert _names(fuzzybits.compile(str(ftp_model), target="G1", **options)) == ["G1"]
    # No comma splitting: this is one (unknown) name.
    with pytest.raises(RuntimeError, match="unknown-target"):
        fuzzybits.compile(str(ftp_model), target="TOP,G1", **options)


def test_api_mef_target_accepts_a_string_or_a_sequence():
    assert fuzzybits.compile(str(CCF_MODEL), target=["TopEvent"]) == fuzzybits.compile(str(CCF_MODEL), target="TopEvent")
    blob = fuzzybits.compile(str(CCF_MODEL), target="TopEvent", watch=["TrainOne", "TrainTwo"])
    assert _names(blob)[1:] == ["TrainOne", "TrainTwo"]
    with pytest.raises(ValueError, match="take one target"):
        fuzzybits.compile(str(CCF_MODEL), target=["TopEvent", "TrainOne"])
    with pytest.raises(ValueError, match="more than once"):
        fuzzybits.compile(str(CCF_MODEL), target="TopEvent", watch=["TopEvent"])


def test_cli_name_files_for_ftp(tmp_path: Path, ftp_model: Path):
    targets = tmp_path / "targets.txt"
    targets.write_bytes(b"  G1 \r\n\r\n\nE1\r\n")
    watch = tmp_path / "watch.txt"
    watch.write_text("E3\n", encoding="utf-8")
    ran, out = _compile_cli(
        tmp_path, str(ftp_model), "--target", "TOP", "--target", f"@{targets}", "--target", "E2", "--watch", f"@{watch}", *FTP_FLAGS
    )
    assert ran.returncode == 0, ran.stderr
    assert _names(out.read_bytes()) == ["TOP", "G1", "E1", "E2", "E3"]

    ran, _ = _compile_cli(tmp_path, str(ftp_model), "--target", "G1", "--target", f"@{targets}", *FTP_FLAGS)
    assert ran.returncode == 1
    assert "duplicate-target" in ran.stderr


def test_cli_missing_name_file(tmp_path: Path, ftp_model: Path):
    missing = tmp_path / "missing.txt"
    for model in (ftp_model, CCF_MODEL):
        for flag in ("--target", "--watch"):
            ran, out = _compile_cli(tmp_path, str(model), flag, f"@{missing}", "--no-progress")
            assert ran.returncode == 1
            assert f"fzb compile: cannot read {missing}" in ran.stderr
            assert not out.exists()


def test_cli_name_file_without_names_or_not_a_file(tmp_path: Path, ftp_model: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("\n  \r\n", encoding="utf-8")
    for model in (ftp_model, CCF_MODEL):
        for flag in ("--target", "--watch"):
            ran, out = _compile_cli(tmp_path, str(model), flag, f"@{empty}", *FTP_FLAGS[-1:])
            assert ran.returncode == 1
            assert f"fzb compile: {empty} lists no names" in ran.stderr
            assert not out.exists()
        ran, out = _compile_cli(tmp_path, str(model), "--target", f"@{tmp_path}", "--no-progress")
        assert ran.returncode == 1
        assert f"fzb compile: {tmp_path} is not a regular file" in ran.stderr
        assert not out.exists()


def test_api_watch_takes_integer_indices_not_floats(ftp_model: Path):
    with pytest.raises(TypeError):
        fuzzybits.compile(str(ftp_model), watch=1.5, negation="boolean", max_initiators_per_cutset=None)


def test_native_compile_expands_ccf_without_the_key():
    native = fuzzybits.native()
    default = fuzzybits.inspect(bytes(native.compile_pcfb([str(CCF_MODEL)], {})))
    plain = fuzzybits.inspect(bytes(native.compile_pcfb([str(CCF_MODEL)], {"ccf": False})))
    assert default["samples"] > plain["samples"]


def test_cli_mef_takes_one_target_and_watches_from_a_file(tmp_path: Path):
    one = tmp_path / "one.txt"
    one.write_text("TopEvent\n", encoding="utf-8")
    two = tmp_path / "two.txt"
    two.write_text("TopEvent\nTrainOne\n", encoding="utf-8")
    watch = tmp_path / "watch.txt"
    watch.write_bytes(b"TrainOne\r\nTrainTwo\r\n")

    ran, out = _compile_cli(tmp_path, str(CCF_MODEL), "--target", f"@{one}", "--watch", f"@{watch}", "--no-progress")
    assert ran.returncode == 0, ran.stderr
    assert _names(out.read_bytes())[1:] == ["TrainOne", "TrainTwo"]

    for args in (["--target", "TopEvent", "--target", "TrainOne"], ["--target", f"@{two}"]):
        ran, out = _compile_cli(tmp_path, str(CCF_MODEL), *args, "--no-progress")
        assert ran.returncode == 1, args
        assert ONE_TARGET in ran.stderr
        assert not out.exists()

    ran, _ = _compile_cli(tmp_path, str(CCF_MODEL), "--target", "TopEvent", "--watch", "TopEvent", "--no-progress")
    assert ran.returncode == 1
    assert "target 'TopEvent' is requested more than once" in ran.stderr


def test_nombrilo_input_rejects_watch(tmp_path: Path, tiny_xml: Path):
    circuit = tmp_path / "tiny.safetensors"
    circuit.write_bytes(fuzzybits.pcfb_to_nombrilo(fuzzybits.compile(str(tiny_xml))))
    assert fuzzybits.inspect(fuzzybits.compile(str(circuit)))
    with pytest.raises(Exception, match="does not take --watch"):
        fuzzybits.compile(str(circuit), watch=["x"])
    ran, out = _compile_cli(tmp_path, str(circuit), "--watch", "x", "--no-progress")
    assert ran.returncode == 1
    assert "does not take --watch" in ran.stderr
    assert not out.exists()


def test_cli_help_shows_target_and_ccf_flags():
    ran = _run_fzb("compile", "--help")
    assert ran.returncode == 0
    for needle in ("--target NAME|@FILE", "--watch NAME|@FILE", "--no-ccf"):
        assert needle in ran.stdout
