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
    # newline="": the tests hash `text`, so keep "\n" as written (Windows would write "\r\n").
    path.write_text(text, encoding="utf-8", newline="")
    return path


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-15)


def _probability(**options):
    # The conformance fixtures tag probabilities with I, so their values are read as probabilities (#243).
    return {"kind": "probability", "initiator_values": "probability", **options}


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda path: path.name)
def test_conformance_fixtures_match_expected_boolean_values(fixture: Path):
    spec = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    files = [str(fixture / name) for name in spec["files"]]
    for policy, outcome in spec["policies"].items():
        options = _probability(duplicate_import=policy)
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
            "kind": "probability",
            "initiator_values": "probability",
            "duplicate_import": variant["policy"],
            "overrides": variant.get("overrides"),
            "hull_max_initiators": variant.get("max_initiators_per_cutset"),
        }
        if variant.get("max_initiators_per_cutset") is not None:
            # The variant's `boolean` values apply the initiator rule to the probability query, which is
            # refused: the rule belongs to hull queries (#262).
            with pytest.raises(RuntimeError, match="invalid-option"):
                fuzzybits.compile(files, **options)
            continue
        for target, values in variant["results"].items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                blob = fuzzybits.compile(files, target=target, **options)
            assert _close(fuzzybits.exact(blob), values["boolean"]), (variant, target)


def test_compile_returns_the_blob_with_metadata_and_names(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    blob = fuzzybits.compile([model], **_probability())
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
    assert recorded["kinds"] == ["probability"]
    assert recorded["hull"] == {"literals": "split", "max_initiators": None}
    assert recorded["initiator_values"] == "probability"
    assert recorded["targets"][0]["name"] == "TOP"
    assert recorded["targets"][0]["kind"] == "probability"
    assert "hull_literals" not in recorded["targets"][0]
    assert _close(fuzzybits.exact(blob), 0.314)
    assert summary["samples"] == 3


def test_probability_is_the_default_and_hull_is_requested(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    default = fuzzybits.compile([model])
    summary = fuzzybits.inspect(default)
    assert summary["diagnostics"] == []
    assert summary["schema_version"] == 0x00010001
    assert summary["query_kinds"] == ["probability"]
    assert "compile.query.0.negation" not in summary["metadata"]
    assert summary["metadata"]["compile.query.0.target"] == "TOP"
    blob = fuzzybits.compile([model], kind="hull")
    summary = fuzzybits.inspect(blob)
    assert summary["schema_version"] == 0x00010002
    assert summary["query_kinds"] == ["hull"]
    meta = summary["metadata"]
    assert meta["compile.query.0.literal_mode"] == "split"
    # No initiator rule unless requested.
    assert "compile.query.0.max_initiators_per_cutset" not in meta
    both = fuzzybits.inspect(
        fuzzybits.compile([model], config={"targets": [{"name": "TOP", "kinds": ["probability", "hull"]}]})
    )
    assert both["queries"] == 2
    assert both["query_kinds"] == ["probability", "hull"]
    assert fuzzybits.inspect(fuzzybits.compile([model], kind=["probability", "hull"]))["query_kinds"] == [
        "probability",
        "hull",
    ]
    # The runtime guard (#217) refuses the hull file until #183.
    with pytest.raises(Exception, match="unsupported-query-kind"):
        fuzzybits.run(blob, trials=1024, backend="omp")


def test_hull_targets_with_marked_literals_are_refused(tmp_path: Path):
    model = CONFORMANCE / "initiator_rule" / "model.ftp"
    for options in ({}, {"hull_literals": "fixed"}):
        with pytest.raises(RuntimeError) as failure:
            fuzzybits.compile([model], kind="hull", initiator_values="probability", **options)
        assert "compile.ftp.hull-literals-unsupported" in str(failure.value)
        assert "--hull-literals plain" in str(failure.value)
    blob = fuzzybits.compile([model], kind="hull", initiator_values="probability", hull_literals="plain")
    assert fuzzybits.inspect(blob)["metadata"]["compile.query.0.literal_mode"] == "plain"
    # complement_events is gone: -EVENT is always a marked literal; hull_literals="plain" is the plain hull.
    with pytest.raises(TypeError, match="complement_events"):
        fuzzybits.compile([model], kind="hull", initiator_values="probability", complement_events="negation")
    with pytest.raises(RuntimeError, match="invalid-config"):
        fuzzybits.compile([model], initiator_values="probability", config={"complement_events": "negation"})
    # The default probability query is not affected by marked literals.
    assert fuzzybits.inspect(fuzzybits.compile([model], initiator_values="per-year"))["query_kinds"] == [
        "probability"
    ] * 3


def test_targets_watch_overrides_and_levels(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    blob = fuzzybits.compile(
        [model],
        target=["G1", "E3"],
        watch="TOP",
        overrides={"E2": True},
        pdag_c=0,
        **_probability(),
    )
    summary = fuzzybits.inspect(blob)
    assert summary["query_names"] == ["G1", "E3", "TOP"]
    assert _close(fuzzybits.exact(blob, 0), 0.1)
    assert _close(fuzzybits.exact(blob, 1), 0.3)
    assert _close(fuzzybits.exact(blob, 2), 0.37)
    assert json.loads(summary["metadata"]["compile.options"])["compilation_level"] == 0
    assert fuzzybits.compile([model], overrides=["E2=1"], **_probability(target="TOP"))


@pytest.mark.filterwarnings("ignore:compile.pra_to_pc.unused-override")
def test_the_initiator_rule_is_requested_for_hull_queries_and_flags_override_the_config():
    model = CONFORMANCE / "initiator_rule" / "model.ftp"
    # No rule is read from the model: the probability query of TOP is P(TOP).
    default = fuzzybits.compile([model], kind="probability", initiator_values="probability")
    assert _close(fuzzybits.exact(default), 0.45000000000000007)
    assert "compile.query.0.max_initiators_per_cutset" not in fuzzybits.inspect(default)["metadata"]
    # A rule on probability queries would change them: refused.
    with pytest.raises(RuntimeError, match="hull queries only"):
        fuzzybits.compile([model], kind="probability", initiator_values="probability", hull_max_initiators=1)
    hull = fuzzybits.compile(
        [model], kind="hull", hull_literals="plain", initiator_values="probability", hull_max_initiators=1
    )
    assert fuzzybits.inspect(hull)["metadata"]["compile.query.0.max_initiators_per_cutset"] in (1, "1")

    config = {
        "kinds": "probability",
        "initiator_values": "probability",
        "hull": {"max_initiators": None},
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
        kind="hull",
        hull_literals="plain",
        hull_max_initiators=1,
        overrides={"OPER": True},
        pdag_c=0,
    )
    summary = fuzzybits.inspect(flagged)
    assert summary["query_names"] == ["SEQ-4"]
    # The flag's rule, not the document's None: SEQ-4 holds two initiating events.
    assert summary["metadata"]["compile.query.0.max_initiators_per_cutset"] in (1, "1")


def test_config_from_a_file(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    config = _write(tmp_path, "config.json", json.dumps({"kinds": "probability", "initiator_values": "probability"}))
    for value in (config, str(config)):
        assert _close(fuzzybits.exact(fuzzybits.compile([model], config=value)), 0.314)


def test_report_returns_blob_and_reports(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    blob, report = fuzzybits.compile([model], report=True, **_probability())
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
        _, report = fuzzybits.compile([model], report=True, **_probability())
    assert any(item["code"] == "io.ftp.duplicate-import" for item in report["diagnostics"])


def test_a_directory_and_several_files_form_one_project(tmp_path: Path):
    directory = CONFORMANCE / "multi_file_project"
    expected = 0.0019254481299999999
    blob = fuzzybits.compile([directory], **_probability())
    assert _close(fuzzybits.exact(blob), expected)
    files = sorted(directory.glob("*.ftp"))
    scrambled = fuzzybits.compile([files[2], files[0], files[1]], **_probability())
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
        ({"kind": "maybe"}, "compile.ftp.invalid-option"),
        ({"kind": "both"}, "compile.ftp.invalid-option"),
        ({"hull_literals": "plain"}, "hull queries only"),
        ({"kind": "hull", "hull_literals": "maybe"}, "compile.ftp.invalid-option"),
        ({"initiator_values": "frequency"}, "compile.ftp.invalid-option"),
        ({"target": "NOPE"}, "compile.ftp.unknown-target"),
        ({"overrides": {"NOPE": True}}, "compile.pra_to_pc.unknown-element"),
        ({"config": "{"}, "compile.ftp.invalid-json"),
        ({"config": {"typo": 1}}, "compile.ftp.invalid-config"),
    ):
        with pytest.raises(RuntimeError, match=needle):
            fuzzybits.compile([model], **_probability(**options))
    with pytest.raises(ValueError, match="cannot be combined"):
        fuzzybits.compile([model, xml])
    with pytest.raises(ValueError, match="FTP input only"):
        fuzzybits.compile([xml], kind="probability")
    with pytest.raises(ValueError, match="ccf applies to MEF input only"):
        fuzzybits.compile([model], ccf=False, **_probability())
    with pytest.raises(TypeError, match="do not apply"):
        fuzzybits.compile([model], keep_null=True, **_probability())
    with pytest.raises(ValueError, match="override E1"):
        fuzzybits.compile([model], overrides={"E1": 2}, **_probability())


def test_progress_callback_sees_the_compile_stages(tmp_path: Path):
    model = _write(tmp_path, "m.ftp", SIMPLE)
    steps: list[str] = []
    fuzzybits.compile([model], on_progress=lambda snapshot: steps.append(snapshot.step), progress=False, **_probability())
    assert [step for step in steps if step] == ["parse", "lower", "passes", "write"]


def test_fzb_compile_cli_on_ftp(tmp_path: Path):
    model = CONFORMANCE / "initiator_rule" / "model.ftp"
    pcfb = tmp_path / "m.pcfb"
    report = tmp_path / "report.json"
    ran = _run_fzb(
        "compile",
        str(model),
        "--kind",
        "probability",
        "--initiator-values",
        "probability",
        "--hull-max-initiators",
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

    hull = _run_fzb(
        "compile",
        str(model),
        "--kind",
        "hull",
        "--initiator-values",
        "probability",
        "--no-progress",
        "-o",
        str(pcfb),
        cwd=tmp_path,
    )
    assert hull.returncode != 0
    assert "hull-literals-unsupported" in hull.stderr
    assert "--hull-literals plain" in hull.stderr

    removed = _run_fzb(
        "compile",
        str(model),
        "--kind",
        "hull",
        "--initiator-values",
        "probability",
        "--complement-events",
        "negation",
        "--no-progress",
        "-o",
        str(pcfb),
        cwd=tmp_path,
    )
    assert removed.returncode != 0
    assert "complement-events" in removed.stderr

    bad = _run_fzb("compile", str(model), "--kind", "probability", "--override", "OPER", "-o", str(pcfb), cwd=tmp_path)
    assert bad.returncode != 0
    assert "NAME=0 or NAME=1" in bad.stderr

    xml = _write(tmp_path, "m.xml", "<opsa-mef/>\n")
    mef = _run_fzb("compile", str(xml), "--kind", "probability", "-o", str(pcfb), cwd=tmp_path)
    assert mef.returncode != 0
    assert "CAFTA FTP input only" in mef.stderr

    # Removed options are unknown; --no-ccf is for MEF input.
    for removed in (["--negation", "hull"], ["--literals", "plain"], ["--max-initiators-per-cutset", "1"],
                    ["--initiator-unit", "per-year"], ["--ccf"]):
        ran = _run_fzb("compile", str(model), *removed, "-o", str(pcfb), cwd=tmp_path)
        assert ran.returncode != 0, removed
    ran = _run_fzb("compile", str(model), "--no-ccf", "-o", str(pcfb), cwd=tmp_path)
    assert ran.returncode != 0
    assert "applies to OpenPSA/MEF input" in ran.stderr


FREQUENCY = Path(__file__).resolve().parents[2] / "input" / "ftp" / "frequency"


@pytest.mark.parametrize("name", ["F1", "F2", "F3", "F4", "F5", "F6", "F7"])
def test_initiator_frequencies_are_query_groups(name: str):
    """#243: one conditioned query per initiating event, grouped as the target's frequency."""
    spec = json.loads((FREQUENCY / "expected.json").read_text(encoding="utf-8"))["models"][name]
    model = FREQUENCY / f"{name}.ftp"
    if "error" in spec:
        with pytest.raises(RuntimeError, match=spec["error"]):
            fuzzybits.compile([model], initiator_values="per-year")
        return
    blob = fuzzybits.compile([model], initiator_values="per-year")
    summary = fuzzybits.inspect(blob)
    assert summary["query_names"] == [f"TOP|{initiator}" for initiator in spec["terms"]]
    (group,) = fuzzybits.exact_groups(blob)
    assert group.name == "TOP" and group.unit == "per-year"
    assert [weight for _, weight in group.terms] == list(spec["terms"].values())
    assert _close(group.estimate, spec["frequency"])


def test_initiator_frequency_run_and_exposure():
    blob = fuzzybits.compile([FREQUENCY / "F4.ftp"], initiator_values="per-year")
    rows = fuzzybits.run(blob, trials=1_000_000, backend="omp", exposure="1y")
    assert [row.name for row in rows] == ["TOP|IE1", "TOP|IE2"]
    (group,) = rows.groups
    assert _close(group.estimate, 20.0 * rows[0].estimate + 3.0 * rows[1].estimate)
    assert abs(group.estimate - 0.26) <= 0.05 * 0.26
    assert group.ci_low <= group.estimate <= group.ci_high
    assert _close(group.exposure[0], -math.expm1(-group.estimate))

    (exact,) = fuzzybits.exact_groups(fuzzybits.compile([FREQUENCY / "F2.ftp"], initiator_values="per-year"), exposure="1mo")
    assert _close(exact.estimate, 12.0)
    assert _close(exact.exposure[0], -math.expm1(-1.0))
    with pytest.raises(ValueError, match="exposure"):
        fuzzybits.exact_groups(blob, exposure="1fortnight")
    with pytest.raises(ValueError, match="no frequency groups"):
        fuzzybits.run(fuzzybits.compile([FREQUENCY / "F6.ftp"], initiator_values="probability",
                                        hull_max_initiators="none"), trials=1024, backend="omp", exposure="1y")


def test_initiator_values_default_to_per_year_and_the_probability_reading_stays():
    (group,) = fuzzybits.exact_groups(fuzzybits.compile([FREQUENCY / "F1.ftp"]))
    assert group.unit == "per-year"
    assert _close(group.estimate, 0.596)
    with pytest.raises(RuntimeError, match="frequency-out-of-range"):
        fuzzybits.compile([FREQUENCY / "F1.ftp"], initiator_values="probability")
    blob = fuzzybits.compile([FREQUENCY / "F6.ftp"], initiator_values="probability", hull_max_initiators="none")
    assert _close(fuzzybits.exact(blob), 0.0556)
    with pytest.raises(RuntimeError, match="invalid-option"):
        fuzzybits.compile([FREQUENCY / "F1.ftp"], initiator_values="per-year", kind="hull", hull_max_initiators=1)
    (hourly,) = fuzzybits.exact_groups(fuzzybits.compile([FREQUENCY / "F1.ftp"], initiator_values="per-hour"))
    assert hourly.unit == "per-hour"
    assert _close(hourly.estimate, 0.596)
