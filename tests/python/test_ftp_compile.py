"""``fuzzybits.compile`` and ``fzb compile`` on CAFTA FTP projects (#182)."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import pytest

pytest.importorskip("fuzzybits")

import fuzzybits  # noqa: E402

from test_cli import _run_fzb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFORMANCE = ROOT / "input" / "ftp" / "conformance"
FIXTURES = sorted(path for path in CONFORMANCE.iterdir() if (path / "expected.json").is_file())

SIMPLE = "TOP + G1 E3\nG1 * E1 E2\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 E1\n  2.0E-01 E2\n  3.0E-01 E3\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-15)


def _boolean(**options):
    return {"negation": "boolean", "max_initiators_per_cutset": None, **options}


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda path: path.name)
def test_conformance_fixtures_match_expected_boolean_values(fixture: Path):
    spec = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    files = [str(fixture / name) for name in spec["files"]]
    for policy, outcome in spec["policies"].items():
        options = _boolean(duplicate_import=policy)
        if outcome["results"] is None:
            with pytest.raises(RuntimeError) as failure:
                fuzzybits.compile(files, **options)
            for item in outcome["diagnostics"]:
                if item["severity"] == "error":
                    assert f"io.ftp.{item['code']}" in str(failure.value)
            continue
        for target, values in outcome["results"].items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                blob = fuzzybits.compile(files, target=target, **options)
            assert _close(fuzzybits.exact(blob), values["boolean"]), (policy, target)
    for variant in spec.get("variants", []):
        options = {
            "negation": "boolean",
            "duplicate_import": variant["policy"],
            "overrides": variant.get("overrides"),
            "max_initiators_per_cutset": variant.get("max_initiators_per_cutset"),
        }
        for target, values in variant["results"].items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                blob = fuzzybits.compile(files, target=target, **options)
            assert _close(fuzzybits.exact(blob), values["boolean"]), (variant, target)


def test_compile_returns_the_blob_with_metadata_and_names(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    blob = fuzzybits.compile([model], **_boolean())
    assert isinstance(blob, bytes)
    summary = fuzzybits.inspect(blob)
    assert summary["diagnostics"] == []
    assert summary["query_names"] == ["TOP"]
    meta = summary["metadata"]
    assert meta["general.source_format"] == "cafta-ftp"
    assert meta["general.source_sha256"] == hashlib.sha256(SIMPLE.encode()).hexdigest()
    recorded = json.loads(meta["compile.options"])
    assert recorded["format"] == "cafta-ftp"
    assert recorded["compilation_level"] == 1
    assert recorded["max_initiators_per_cutset"] is None
    assert recorded["targets"][0]["name"] == "TOP"
    assert recorded["targets"][0]["negation"] == "boolean"
    assert _close(fuzzybits.exact(blob), 0.314)
    assert summary["samples"] == 3


def test_probability_is_the_default_and_hull_is_requested(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    default = fuzzybits.compile([model])
    summary = fuzzybits.inspect(default)
    assert summary["diagnostics"] == []
    assert summary["schema_version"] == 0x00010001
    assert summary["metadata"]["compile.query.0.negation"] == "boolean"
    blob = fuzzybits.compile([model], negation="hull")
    summary = fuzzybits.inspect(blob)
    assert summary["schema_version"] == 0x00010002
    meta = summary["metadata"]
    assert meta["compile.query.0.negation"] == "monotone_hull"
    assert meta["compile.query.0.literal_mode"] == "atom_split"
    assert meta["compile.query.0.max_initiators_per_cutset"] == "1"
    both = fuzzybits.inspect(fuzzybits.compile([model], config={"targets": [{"name": "TOP", "negation": "both"}]}))
    assert both["queries"] == 2
    assert both["metadata"]["compile.query.1.negation"] == "monotone_hull"
    # The runtime guard (#217) refuses the hull file until #183.
    with pytest.raises(Exception, match="unsupported-query-kind"):
        fuzzybits.run(blob, trials=1024, backend="omp")


def test_hull_targets_with_marked_literals_are_refused(tmp_path: Path):
    model = CONFORMANCE / "initiator_rule" / "model.ftp"
    for options in ({}, {"literals": "omega_l"}):
        with pytest.raises(RuntimeError) as failure:
            fuzzybits.compile([model], negation="hull", **options)
        assert "compile.ftp.hull-literals-unsupported" in str(failure.value)
        assert "--literals negation" in str(failure.value)
    blob = fuzzybits.compile([model], negation="hull", literals="negation")
    assert fuzzybits.inspect(blob)["metadata"]["compile.query.0.literal_mode"] == "negation"
    assert fuzzybits.compile([model], negation="hull", complement_events="negation")
    # The default probability query is not affected by marked literals.
    assert fuzzybits.inspect(fuzzybits.compile([model]))["metadata"]["compile.query.0.negation"] == "boolean"


def test_targets_watch_overrides_and_levels(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    blob = fuzzybits.compile(
        [model],
        target=["G1", "E3"],
        watch="TOP",
        overrides={"E2": True},
        pdag_c=0,
        **_boolean(),
    )
    summary = fuzzybits.inspect(blob)
    assert summary["query_names"] == ["G1", "E3", "TOP"]
    assert _close(fuzzybits.exact(blob, 0), 0.1)
    assert _close(fuzzybits.exact(blob, 1), 0.3)
    assert _close(fuzzybits.exact(blob, 2), 0.37)
    assert json.loads(summary["metadata"]["compile.options"])["compilation_level"] == 0
    assert fuzzybits.compile([model], overrides=["E2=1"], **_boolean(target="TOP"))


@pytest.mark.filterwarnings("ignore:compile.pra_to_pc.unused-override")
def test_defaults_come_from_the_model_attributes_and_flags_override_the_config():
    model = CONFORMANCE / "initiator_rule" / "model.ftp"
    # ftp.max_initiators_per_cutset = 1 is the default rule: TOP is 0.288, not 0.45.
    by_attribute = fuzzybits.compile([model], negation="boolean")
    assert _close(fuzzybits.exact(by_attribute), 0.28800000000000003)
    without = fuzzybits.compile([model], negation="boolean", max_initiators_per_cutset="none")
    assert _close(fuzzybits.exact(without), 0.45000000000000007)

    config = {
        "negation": "boolean",
        "max_initiators_per_cutset": None,
        "targets": ["SEQ-3", "TOP"],
        "overrides": {"OPER": 0},
    }
    from_file = fuzzybits.compile([model], config=config)
    assert fuzzybits.inspect(from_file)["query_names"] == ["SEQ-3", "TOP"]
    assert _close(fuzzybits.exact(from_file), 0.5)

    flagged = fuzzybits.compile(
        [model],
        config=json.dumps(config),
        target="SEQ-4",
        max_initiators_per_cutset=1,
        overrides={"OPER": True},
        pdag_c=0,
    )
    assert fuzzybits.inspect(flagged)["query_names"] == ["SEQ-4"]
    assert fuzzybits.exact(flagged) == 0.0


def test_config_from_a_file(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    config = _write(tmp_path, "config.json", json.dumps({"negation": "boolean", "max_initiators_per_cutset": None}))
    for value in (config, str(config)):
        assert _close(fuzzybits.exact(fuzzybits.compile([model], config=value)), 0.314)


def test_report_returns_blob_and_reports(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    blob, report = fuzzybits.compile([model], report=True, **_boolean())
    assert fuzzybits.inspect(blob)["query_names"] == ["TOP"]
    assert report["status"] == "ok"
    assert report["source"]["format"] == "cafta-ftp"
    assert report["source"]["sha256"] == hashlib.sha256(SIMPLE.encode()).hexdigest()
    assert report["lowering"]["transform"] == "compile.pra_to_pc"
    assert report["lowering"]["pc_targets"] == 1
    assert [p["transform"] for p in report["passes"]][0] == "ir.pc.fold_zero_probability"
    assert report["circuit"]["nodes_after"] <= report["circuit"]["nodes_before"]
    assert report["options"]["targets"][0]["name"] == "TOP"
    assert all({"severity", "code", "message"} <= set(item) for item in report["diagnostics"])


def test_warnings_are_issued_and_listed_in_the_report():
    model = CONFORMANCE / "imports_tags_duplicates" / "model.ftp"
    with pytest.warns(UserWarning, match="io.ftp.duplicate-import"):
        _, report = fuzzybits.compile([model], report=True, **_boolean())
    assert any(item["code"] == "io.ftp.duplicate-import" for item in report["diagnostics"])


def test_a_directory_and_several_files_form_one_project(tmp_path: Path):
    directory = CONFORMANCE / "multi_file_project"
    expected = 0.0019254481299999999
    blob = fuzzybits.compile([directory], **_boolean())
    assert _close(fuzzybits.exact(blob), expected)
    files = sorted(directory.glob("*.ftp"))
    scrambled = fuzzybits.compile([files[2], files[0], files[1]], **_boolean())
    # Same files, same digest; the project is named after its directory only when given one.
    assert fuzzybits.inspect(scrambled)["metadata"]["general.source_sha256"] == fuzzybits.inspect(blob)["metadata"]["general.source_sha256"]
    assert fuzzybits.inspect(scrambled)["metadata"]["general.model_name"] == "01_top"
    assert _close(fuzzybits.exact(scrambled), expected)
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()
    assert fuzzybits.inspect(blob)["metadata"]["general.source_sha256"] == digest
    assert fuzzybits.inspect(blob)["metadata"]["general.model_name"] == "multi_file_project"


def test_invalid_options_and_mixed_inputs(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    xml = _write(tmp_path, "m.xml", "<opsa-mef/>\n")
    for options, needle in (
        ({"pdag_c": 2}, "compile.ftp.invalid-option"),
        ({"negation": "maybe"}, "compile.ftp.invalid-option"),
        ({"literals": "maybe"}, "compile.ftp.invalid-option"),
        ({"target": "NOPE"}, "compile.ftp.unknown-target"),
        ({"overrides": {"NOPE": True}}, "compile.pra_to_pc.unknown-element"),
        ({"config": "{"}, "compile.ftp.invalid-json"),
        ({"config": {"typo": 1}}, "compile.ftp.invalid-config"),
    ):
        with pytest.raises(RuntimeError, match=needle):
            fuzzybits.compile([model], **_boolean(**options))
    with pytest.raises(ValueError, match="cannot be combined"):
        fuzzybits.compile([model, xml])
    with pytest.raises(ValueError, match="FTP input only"):
        fuzzybits.compile([xml], negation="boolean")
    with pytest.raises(TypeError, match="do not apply"):
        fuzzybits.compile([model], keep_null=True, **_boolean())
    with pytest.raises(ValueError, match="override E1"):
        fuzzybits.compile([model], overrides={"E1": 2}, **_boolean())


def test_progress_callback_sees_the_compile_stages(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    steps: list[str] = []
    fuzzybits.compile([model], on_progress=lambda snapshot: steps.append(snapshot.step), progress=False, **_boolean())
    assert [step for step in steps if step] == ["parse", "lower", "passes", "write"]


def test_fzb_compile_cli_on_ftp(tmp_path: Path):
    model = CONFORMANCE / "initiator_rule" / "model.ftp"
    pcfb = tmp_path / "m.pcfb"
    report = tmp_path / "report.json"
    ran = _run_fzb(
        "compile",
        str(model),
        "--negation",
        "boolean",
        "--max-initiators-per-cutset",
        "none",
        "--target",
        "SEQ-3",
        "--override",
        "OPER=0",
        "--no-progress",
        "-o",
        str(pcfb),
        "--report",
        str(report),
        cwd=tmp_path,
    )
    assert ran.returncode == 0, ran.stderr
    assert "wrote" in ran.stderr and "circuit: nodes" in ran.stderr
    assert _close(fuzzybits.exact(pcfb), 0.5)
    assert json.loads(report.read_text())["options"]["overrides"][0]["name"] == "OPER"

    hull = _run_fzb("compile", str(model), "--negation", "hull", "--no-progress", "-o", str(pcfb), cwd=tmp_path)
    assert hull.returncode != 0
    assert "hull-literals-unsupported" in hull.stderr
    assert "--literals negation" in hull.stderr

    bad = _run_fzb("compile", str(model), "--negation", "boolean", "--override", "OPER", "-o", str(pcfb), cwd=tmp_path)
    assert bad.returncode != 0
    assert "NAME=0 or NAME=1" in bad.stderr

    xml = _write(tmp_path, "m.xml", "<opsa-mef/>\n")
    mef = _run_fzb("compile", str(xml), "--negation", "boolean", "-o", str(pcfb), cwd=tmp_path)
    assert mef.returncode != 0
    assert "CAFTA FTP input only" in mef.stderr
