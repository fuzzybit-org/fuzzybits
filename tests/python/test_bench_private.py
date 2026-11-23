"""Private v2 entries in the harness (#184), with a fake private directory.

Every test builds its own fake fixture in ``tmp_path`` from a public FTP
fixture: a fake model name, a card, a ``.bench.json`` and a v2 manifest whose
entry carries the file digests. Nothing here reads the real private directory,
and the leak tests assert that no string from the fake directory reaches any
output the harness can publish.

``--execute local`` runs the real inner script with this interpreter, so the
engine-backed tests need ``fzb`` and ``fuzzybits`` on ``PATH`` (CI installs the
wheel and puts ``fzb`` beside the interpreter). The discovery, schema, verdict
and leak logic do not sample anything and run without them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.datasets import jsonschema_lite  # noqa: E402
from benchmarks.datasets.v2 import (  # noqa: E402
    Discovery,
    PrivateEntry,
    discover,
    load_v2,
    private_line,
    sanitize_private,
    settings_of,
)
from benchmarks.harness.private import (  # noqa: E402
    PrivateRunner,
    accuracy_payload,
    accuracy_verdict,
    bench_payload,
    boolean_variants,
    compile_payload,
    duplicate_policy,
    overrides_from,
    perf_payload,
    perf_verdict,
    reference_of,
    write_bench,
    z_for,
)
from benchmarks.harness.summarize import private_lines, summarize  # noqa: E402

SCHEMA_V1 = ROOT / "benchmarks" / "schemas" / "dataset-manifest.v1.json"
SCHEMA_V2 = ROOT / "benchmarks" / "schemas" / "dataset-manifest.v2.json"
V1_MANIFEST = ROOT / "benchmarks" / "datasets" / "v1" / "manifest.json"
V2_MANIFEST = ROOT / "benchmarks" / "datasets" / "v2" / "manifest.json"
# v1 is frozen: this pin is the test that it never changed.
V1_MANIFEST_SHA256 = "907efec549bcd27c8d388854679a811089efe82bd7c3d49b82fd05e9a67674b7"

# Fake identifiers and values: none of these may appear in published output.
MODEL = "QUOKKA_PRIVATE_PLANT"
ENTRY_ID = "private-ftp-1"
BAND_A = "WOMBAT_BAND_A"
BAND_B = "WOMBAT_BAND_B"
GATE = "QUOKKA_TOP_GATE"
P_REF = 0.6180339887
TRIALS = 1 << 22

# Words the report format itself uses: mode names, verdicts, reason codes and
# field names. They appear in every harness output by design.
HARNESS_VOCABULARY = frozenset(
    {
        "accuracy", "accuracy-boolean", "ambiguous", "basic_events", "boolean",
        "cafta-ftp", "compile", "complement_events", "dataset", "defaults",
        "depth", "description", "duplicate_import", "edges", "error", "estimate",
        "exact", "expected", "fail", "false", "files", "first", "format",
        "frozen", "gates", "hull", "id", "kind", "last", "level", "levels",
        "licence", "mode", "models", "negation", "no-match", "no-reference",
        "not-declared", "null", "observed", "overrides", "pass", "perf",
        "policy", "private", "private-until-audit", "probability", "reason",
        "rel_tolerance_95", "results", "rounds", "schema_version", "semantics",
        "skipped", "status", "structure", "time_budget_s", "trials", "true",
        "uncertainty", "variants", "verdict", "versioning", "warmup",
    }
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ftp_text() -> str:
    """A two-event project with the fake names, read by the real FTP reader."""
    return (
        f"{GATE} + {BAND_A} {BAND_B}\n"
        "ENDTREE\n"
        f"PROCESS {GATE}\n"
        "IMPORT\n"
        f"  3.0E-01 {BAND_A}\n"
        f"  4.0E-01 {BAND_B}\n"
    )


def _card(probability: float = P_REF) -> dict:
    return {
        "id": MODEL,
        "status": "private-until-audit",
        "format": "cafta-ftp",
        "files": [f"{MODEL}.ftp"],
        "structure": {"gates": 1, "edges": 2, "basic_events": 2, "depth": 1},
        "policy": {"duplicate_import": "first"},
        "semantics": {"variants": ["B1 boolean"]},
        "variants": {
            "B1": {
                "semantics": "boolean",
                "overrides": {BAND_B: False},
                "expected": {
                    "kind": "estimate",
                    "probability": probability,
                    "rel_tolerance_95": 0.08,
                },
                "policy": {"duplicate_import": "first"},
            },
            "B0": {
                "semantics": "boolean",
                "overrides": {BAND_A: True, BAND_B: True},
                "expected": {"kind": "exact", "probability": 1.0},
                "policy": {"duplicate_import": "first"},
            },
            "B2_observed_only": {
                "semantics": "boolean",
                "overrides": {},
                "observed": {"n": 1048576, "hits": 3, "estimate": 2.9e-06},
            },
            "H1": {
                "semantics": "monotone_hull",
                "expected": {"kind": "estimate", "probability": 0.5, "se": 0.01},
            },
        },
    }


def _settings() -> dict:
    return {
        "schema_version": 1,
        "id": ENTRY_ID,
        "compile": {
            "levels": [0, 1],
            "negation": "boolean",
            "duplicate_import": "first",
            "complement_events": "negation",
            "overrides": [],
            "time_budget_s": 300,
        },
        "perf": {
            "level": 0,
            "trials": 1 << 18,
            "warmup": 1 << 16,
            "rounds": 2,
            "time_budget_s": 120,
        },
        "accuracy": {"level": 0, "trials": 1 << 18, "warmup": 1 << 16, "time_budget_s": 120},
    }


def _build(root: Path, *, probability: float = P_REF) -> Path:
    """A fake private tree: ``root/ftp/<MODEL>/`` with its card and bench file."""
    directory = root / "ftp" / MODEL
    directory.mkdir(parents=True)
    (directory / f"{MODEL}.ftp").write_text(_ftp_text(), encoding="utf-8")
    (directory / f"{MODEL}.card.json").write_text(json.dumps(_card(probability)), encoding="utf-8")
    (directory / f"{MODEL}.bench.json").write_text(json.dumps(_settings()), encoding="utf-8")
    return root


def _manifest(root: Path, *, files: list[str] | None = None, entry_id: str = ENTRY_ID) -> Path:
    """The v2 manifest for the fake tree, beside it rather than inside it.

    Only the tree holds private content; the manifest is the repository-side
    artifact and carries the entry's digests.
    """
    digests = (
        files
        if files is not None
        else [_digest(path) for path in sorted((root / "ftp" / MODEL).glob("*.ftp"))]
    )
    document = {
        "schema_version": 2,
        "dataset": "v2",
        "frozen": False,
        "description": "fake private test dataset",
        "versioning": "draft",
        "defaults": {"private": {"compile_levels": [0, 1], "compile_budget_s": 300}},
        "models": [
            {
                "id": entry_id,
                "private": True,
                "files": digests,
                "modes": ["compile", "perf", "accuracy-boolean"],
            }
        ],
    }
    path = root.parent / "manifest.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _entry(root: Path, *, modes: list[str] | None = None) -> PrivateEntry:
    files = tuple(_digest(path) for path in sorted((root / "ftp" / MODEL).glob("*.ftp")))
    return PrivateEntry(id=ENTRY_ID, files=files, modes=tuple(modes or ("compile", "perf")))


def _cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"FUZZYBITS_PRIVATE_FIXTURES_DIR", "FUZZYBITS_PRIVATE_RESULTS"}
    }
    environment["PYTHONPATH"] = str(ROOT)
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "benchmarks", *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


# --- discovery and the manifest contract ------------------------------------


def test_discovery_matches_by_content_not_path(tmp_path: Path):
    root = _build(tmp_path / "fixtures")
    entry = _entry(root)
    found = discover(entry, root)
    assert found.status == "match"
    assert found.directory == root / "ftp" / MODEL
    assert found.card["id"] == MODEL
    # The same files under another name still match: the digest set is the key.
    renamed = tmp_path / "elsewhere"
    (renamed / "ftp" / "SOMETHING-ELSE").mkdir(parents=True)
    for source in sorted((root / "ftp" / MODEL).glob("*.ftp")):
        shutil.copy(source, renamed / "ftp" / "SOMETHING-ELSE" / source.name)
    assert discover(entry, renamed).status == "match"


def test_discovery_reports_no_match_and_ambiguous(tmp_path: Path):
    root = _build(tmp_path / "fixtures")
    entry = _entry(root)
    assert discover(entry, tmp_path / "absent").status == "no-match"
    other = PrivateEntry(id=ENTRY_ID, files=("0" * 64,), modes=("compile",))
    assert discover(other, root).status == "no-match"
    # Two directories with the same digests are ambiguous, never "the first one".
    second = root / "ftp" / f"{MODEL}-copy"
    second.mkdir()
    for source in sorted((root / "ftp" / MODEL).glob("*.ftp")):
        shutil.copy(source, second / source.name)
    assert discover(entry, root).status == "ambiguous"


def test_v2_schema_rejects_anything_but_the_reduced_private_form(tmp_path: Path):
    schema = json.loads(SCHEMA_V2.read_text(encoding="utf-8"))
    root = _build(tmp_path / "fixtures")
    good = json.loads(_manifest(root).read_text(encoding="utf-8"))
    assert jsonschema_lite.validate(schema, good) == []
    entry = {"id": ENTRY_ID, "private": True, "files": ["a" * 64], "modes": ["compile"]}
    assert jsonschema_lite.validate(schema, {**good, "models": [entry]}) == []
    # A path or a name is exactly what this design forbids in the repository.
    for extra in (
        {"path": f"ftp/{MODEL}"},
        {"name": MODEL},
        {"query": "TOP"},
        {"tags": ["large"]},
        {"structure": {}},
        {"reference": None},
    ):
        assert jsonschema_lite.validate(schema, {**good, "models": [{**entry, **extra}]}), extra
    # `hull` is declared for #183: the schema keeps the enum complete and only
    # the harness refuses to run it.
    assert jsonschema_lite.validate(schema, {**good, "models": [{**entry, "modes": ["hull"]}]}) == []
    # The opaque id must not be the model's name.
    assert jsonschema_lite.validate(schema, {**good, "models": [{**entry, "id": MODEL}]})
    assert jsonschema_lite.validate(schema, {**good, "models": [{**entry, "modes": ["accuracy"]}]})
    assert schema["title"].endswith("(draft)")


def test_v1_manifest_is_byte_identical():
    assert _digest(V1_MANIFEST) == V1_MANIFEST_SHA256, "v1 is frozen; cut v3 instead of editing it"
    schema = json.loads(SCHEMA_V1.read_text(encoding="utf-8"))
    assert jsonschema_lite.validate(schema, json.loads(V1_MANIFEST.read_text(encoding="utf-8"))) == []


def test_committed_v2_draft_validates_and_keeps_private_entries_opaque():
    schema = json.loads(SCHEMA_V2.read_text(encoding="utf-8"))
    document = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    assert jsonschema_lite.validate(schema, document) == []
    assert document["frozen"] is False
    private = [row for row in document["models"] if row.get("private") is True]
    manifest = load_v2(V2_MANIFEST, root=ROOT)
    assert {entry.id for entry in manifest.private} == {row["id"] for row in private}
    for row in private:
        assert set(row) == {"id", "private", "files", "modes"}
        assert row["files"] == sorted(set(row["files"]))
        for digest in row["files"]:
            assert len(digest) == 64 and digest == digest.lower()
    for row in document["models"]:
        if row.get("private"):
            continue
        assert row["formats"] and row["licence"]["attribution"] and row["origin"]["kind"]
        assert row["reader"]["name"] and row["structure"]["gates"] >= 0
        assert sum(1 for form in row["formats"] if form.get("primary")) == 1


def test_migrated_entries_are_present_and_complete():
    document = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in document["models"]}
    for model_id in ("das9201", "EQK-BIN4/Grp-5"):
        assert model_id in by_id
        card = by_id[model_id]
        assert card["structure"]["basic_events"] > 0
        assert card["structure"]["gates"] > 0
        assert card["structure"]["depth"] is not None
        assert card["licence"]["spdx"]
        assert card["origin"]["kind"] in {"hand-built", "published", "synthetic"}
    pwr = by_id["EQK-BIN4/Grp-5"]
    assert pwr["structure"]["event_tree_sequences"] > 0
    assert "event-tree" in pwr["tags"]
    formats = {form["format"] for form in pwr["formats"]}
    assert {"openpsa-mef", "jsinp", "saphire-project"} <= formats


# --- CLI behaviour: skipped, declared, rejected ------------------------------


def test_private_cli_without_the_variable_is_skipped_and_keeps_the_exit_status(tmp_path: Path):
    """No variable and no flag: the entry is a skip, and the exit status is unchanged."""
    root = _build(tmp_path / "fixtures")
    manifest = _manifest(root)
    out = tmp_path / "results.json"
    proc = _cli("private", "--v2-manifest", str(manifest), "--out", str(out), "--mode", "compile")
    assert proc.returncode == 0, proc.stderr
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["private"] == [
        {"id": ENTRY_ID, "mode": "compile", "verdict": "skipped", "reason": "no-match"}
    ]
    assert f"private {ENTRY_ID} compile: skipped (no-match)" in proc.stdout


def test_private_cli_no_match_and_ambiguous(tmp_path: Path):
    root = _build(tmp_path / "fixtures")
    manifest = _manifest(root, files=["b" * 64])
    out = tmp_path / "out.json"
    proc = _cli("private", "--v2-manifest", str(manifest), "--private", str(root),
                "--out", str(out), "--mode", "perf")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())["private"] == [
        {"id": ENTRY_ID, "mode": "perf", "verdict": "skipped", "reason": "no-match"}
    ]
    manifest = _manifest(root)
    second = root / "ftp" / f"{MODEL}-copy"
    second.mkdir()
    for source in sorted((root / "ftp" / MODEL).glob("*.ftp")):
        shutil.copy(source, second / source.name)
    proc = _cli("private", "--v2-manifest", str(manifest), "--private", str(root),
                "--out", str(out), "--mode", "perf")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())["private"][0]["reason"] == "ambiguous"


def test_hull_mode_is_rejected_until_183(tmp_path: Path):
    root = _build(tmp_path / "fixtures")
    manifest = _manifest(root)
    document = json.loads(manifest.read_text())
    document["models"][0]["modes"] = ["compile", "hull"]
    assert jsonschema_lite.validate(json.loads(SCHEMA_V2.read_text()), document) == []
    manifest.write_text(json.dumps(document))
    out = tmp_path / "out.json"
    proc = _cli("private", "--v2-manifest", str(manifest), "--private", str(root),
                "--out", str(out), "--mode", "hull")
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(out.read_text())["private"]
    assert {"id": ENTRY_ID, "mode": "hull", "verdict": "skipped", "reason": "hull"} in rows


def test_a_mode_the_entry_does_not_declare_is_not_declared(tmp_path: Path):
    root = _build(tmp_path / "fixtures")
    manifest = _manifest(root)
    document = json.loads(manifest.read_text())
    document["models"][0]["modes"] = ["compile"]
    manifest.write_text(json.dumps(document))
    out = tmp_path / "out.json"
    proc = _cli("private", "--v2-manifest", str(manifest), "--private", str(root),
                "--out", str(out), "--mode", "perf")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())["private"] == [
        {"id": ENTRY_ID, "mode": "perf", "verdict": "skipped", "reason": "not-declared"}
    ]


# --- verdict rules (no engine needed) ---------------------------------------


def test_boolean_variants_keep_only_what_compiles_and_report_references():
    variants = boolean_variants(_card())
    assert [variant["name"] for variant in variants] == ["B1", "B0", "B2_observed_only"]
    by_name = {variant["name"]: variant for variant in variants}
    assert by_name["B1"]["reference"]["probability"] == P_REF
    assert by_name["B1"]["overrides"] == [f"{BAND_B}=0"]
    assert by_name["B0"]["reference"]["uncertainty"] == 0.0
    assert by_name["B2_observed_only"]["reference"] is None
    # A hull variant is not a Boolean variant: it belongs to the mode #183 adds.
    assert "H1" not in by_name


def test_variants_that_need_a_file_edit_or_a_rule_are_left_out():
    card = _card()
    card["variants"]["EDITED"] = {"semantics": "boolean", "edit": "remove a term"}
    card["variants"]["RULED"] = {"semantics": "boolean", "max_initiators_per_cutset": 1}
    card["variants"]["DELETED"] = {"semantics": "boolean", "extra_delete_term": "at least 2"}
    assert [variant["name"] for variant in boolean_variants(card)] == [
        "B1",
        "B0",
        "B2_observed_only",
    ]


def test_reference_uncertainty_uses_se_or_half_width():
    assert reference_of({"expected": {"kind": "exact", "probability": 0.0}}) == {
        "probability": 0.0,
        "uncertainty": 0.0,
    }
    assert reference_of({"expected": {"kind": "estimate", "probability": 0.2, "se": 0.01}}) == {
        "probability": 0.2,
        "uncertainty": 0.01,
    }
    with_tol = reference_of(
        {"expected": {"kind": "estimate", "probability": 0.2, "rel_tolerance_95": 0.098}}
    )
    assert with_tol["uncertainty"] == pytest.approx(0.2 * 0.098 / 1.96)
    assert reference_of({"observed": {"estimate": 0.1, "n": 100}}) is None


def test_duplicate_policy_reads_the_first_word_of_an_annotated_value():
    assert duplicate_policy({"policy": {"duplicate_import": "last (vendor)"}}, {}) == "last"
    assert duplicate_policy({}, {"policy": {"duplicate_import": "first"}}) == "first"
    assert duplicate_policy({}, {}) == "last"
    assert duplicate_policy({"policy": {"duplicate_import": "nonsense"}}, {}) == "last"


def test_overrides_are_sorted_compile_words():
    assert overrides_from({"overrides": {BAND_B: True, BAND_A: 0}}) == [
        f"{BAND_A}=0",
        f"{BAND_B}=1",
    ]


def test_z_combines_the_run_variance_with_the_reference_uncertainty():
    assert z_for(0.5, 1 << 20, 0.5, 0.0) == pytest.approx(0.0, abs=1e-12)
    z = z_for(0.6, 1 << 20, 0.5, 0.02)
    sigma = ((0.6 * 0.4 / (1 << 20)) + 0.02**2) ** 0.5
    assert z == pytest.approx(0.1 / sigma)
    # A zero-hit run against a large reference is a large negative z, not inf.
    assert z_for(0.0, 1 << 20, 0.5, 0.0) < -4.0


def test_accuracy_verdict_pass_skip_and_failure():
    variants = boolean_variants(_card())
    estimates = [
        {"queries": [{"estimate": P_REF, "n": TRIALS, "hits": int(P_REF * TRIALS)}]},
        {"queries": [{"estimate": 1.0, "n": TRIALS, "hits": TRIALS}]},
        {"queries": [{"estimate": 2.9e-06, "n": TRIALS, "hits": 12}]},
    ]
    verdict, reason, detail = accuracy_verdict(estimates, variants)
    assert (verdict, reason) == ("pass", None), detail
    assert [row["verdict"] for row in detail["variants"]] == ["pass", "pass", "skipped"]
    # 0.85 against 0.618 with a ±8% (95%) reference is far past |z| = 4.
    forced = [{"queries": [{"estimate": 0.85, "n": TRIALS, "hits": int(0.85 * TRIALS)}]}]
    verdict, reason, detail = accuracy_verdict(forced, variants)
    assert (verdict, reason) == ("fail", None)
    assert detail["variants"][0]["verdict"] == "fail"
    assert abs(detail["variants"][0]["z"]) > 4.0
    # The rule is z, not a fixed relative gap: the same run passes against a
    # reference at that probability.
    matched = boolean_variants(_card(probability=0.85))[:1]
    verdict, _reason, _detail = accuracy_verdict(forced, matched)
    assert verdict == "pass"
    # Every variant without a reference: skipped, never a pass.
    only_observed = [variant for variant in variants if variant["reference"] is None]
    assert accuracy_verdict([estimates[2]], only_observed)[:2] == ("skipped", "no-reference")
    # An engine error in one variant fails the mode.
    assert accuracy_verdict([{"error": "boom"}], variants)[0] == "fail"


def test_perf_records_the_first_baseline_then_judges_later_runs():
    verdict, reason, detail = perf_verdict([1.0, 1.01, 0.99], None)
    assert (verdict, reason) == ("pass", None)
    assert detail["baseline"] == "recorded"
    assert detail["median_s"] == pytest.approx(1.0)
    baseline = {"samples": [1.0, 1.0, 1.0]}
    verdict, _reason, detail = perf_verdict([1.0, 1.005, 0.995], baseline)
    assert verdict == "pass"
    assert detail["ratio"] == pytest.approx(1.0, abs=0.01)
    # A 10% slowdown on every round warns; the CPU rule never blocks (#124).
    verdict, _reason, detail = perf_verdict([1.09, 1.11, 1.10], baseline)
    assert verdict == "warn"
    assert detail["ratio_ci"][0] > 1.0
    # A single round each side still gives a ratio, only without an interval.
    verdict, _reason, detail = perf_verdict([1.2], baseline)
    assert verdict == "warn" and "ratio_ci" not in detail
    assert perf_verdict([], None)[:2] == ("fail", "error")


def test_private_rows_are_reduced_to_four_keys():
    rows = [
        {"id": ENTRY_ID, "mode": "perf", "verdict": "pass", "samples": [1.0], "ratio": 1.02},
        {"id": ENTRY_ID, "mode": "compile", "verdict": "skipped", "reason": "no-match", "x": 1},
        {"mode": "perf", "verdict": "pass"},
    ]
    assert sanitize_private(rows) == [
        {"id": ENTRY_ID, "mode": "perf", "verdict": "pass"},
        {"id": ENTRY_ID, "mode": "compile", "verdict": "skipped", "reason": "no-match"},
    ]
    assert private_lines({"private": [rows[1]]}) == [
        f"private {ENTRY_ID} compile: skipped (no-match)"
    ]
    assert private_line({"id": ENTRY_ID, "mode": "perf", "verdict": "fail"}) == (
        f"private {ENTRY_ID} perf: fail"
    )
    rendered = summarize(
        {"kind": "run", "dataset": "v2", "context": {}, "models": [], "private": rows[:2]}
    )
    assert f"private {ENTRY_ID} perf: pass" in rendered
    assert "samples" not in rendered and "ratio" not in rendered


def test_payloads_carry_the_settings_and_never_the_model_name():
    entry = PrivateEntry(id=ENTRY_ID, files=("a" * 64,), modes=("compile", "perf"))
    settings = settings_of(Discovery(entry=entry, status="no-match"), {})
    assert settings["compile"]["duplicate_import"] == "last"
    assert compile_payload("ftp/x", settings)["compile"]["levels"] == [0, 1]
    assert perf_payload("ftp/x", settings)["perf"]["rounds"] >= 2
    assert accuracy_payload("ftp/x", settings, boolean_variants(_card()))["mode"] == (
        "accuracy-boolean"
    )
    benched = bench_payload(settings, {"perf": {"samples": [1.0]}})
    assert benched["results"]["perf"]["samples"] == [1.0]
    assert "results" not in settings


def test_writing_the_bench_file_keeps_the_settings(tmp_path: Path):
    path = tmp_path / "model.bench.json"
    write_bench(path, bench_payload(_settings(), {"compile": {"levels": []}}))
    document = json.loads(path.read_text())
    assert document["id"] == ENTRY_ID
    assert document["compile"]["levels"] == [0, 1]
    assert document["results"]["compile"] == {"levels": []}


# --- engine-backed: compile, perf, a forced accuracy failure -----------------

def _native_extension_available() -> bool:
    """True when ``fzb`` is on ``PATH`` and ``fuzzybits`` can load its native module.

    CI installs the wheel, so both hold there. A source-tree build has the
    extension in ``build/<preset>/targets/python``; the tests do not link it in,
    so they skip instead of failing.
    """
    if shutil.which("fzb") is None:
        return False
    check = (
        "import fuzzybits, pathlib\n"
        "fuzzybits.init(backend='omp')\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", check], capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
    )
    return proc.returncode == 0 and "ok" in proc.stdout


needs_engine = pytest.mark.skipif(
    not _native_extension_available(),
    reason="fzb + the fuzzybits native extension are not importable (install the wheel)",
)


@needs_engine
def test_compile_mode_reports_both_levels_and_passes(tmp_path: Path):
    root = _build(tmp_path / "fixtures")
    found = discover(_entry(root, modes=["compile"]), root)
    document = PrivateRunner(context=None, image="", volumes=(), executor="local").run(
        compile_payload(str(found.directory), settings_of(found, {}))
    )
    assert document["ok"], document.get("error")
    levels = {row["level"]: row for row in document["levels"]}
    assert set(levels) == {0, 1}
    for row in levels.values():
        assert row["rc"] == 0
        assert row["wall_s"] > 0.0
        assert row["circuit"]["nodes_before"] >= row["circuit"]["nodes_after"]
        assert row["samples"] >= 2
    # Level 1 runs the pass pipeline and reports it; level 0 has no passes.
    assert levels[0]["passes"] == []
    assert levels[1]["passes"]


@needs_engine
def test_perf_mode_records_a_baseline_on_the_first_run(tmp_path: Path):
    root = _build(tmp_path / "fixtures")
    entry = _entry(root, modes=["perf"])
    found = discover(entry, root)
    settings = settings_of(found, {})
    document = PrivateRunner(context=None, image="", volumes=(), executor="local").run(
        perf_payload(str(found.directory), settings)
    )
    assert document["ok"], document.get("error")
    samples = [row["sample_s"] for row in document["rounds"]]
    assert samples and all(value > 0.0 for value in samples)
    verdict, _reason, detail = perf_verdict(samples, None)
    assert verdict == "pass" and detail["baseline"] == "recorded"
    # The first run records the baseline; the next compares against it, and the
    # baseline it compared against is kept so a third run can compare again.
    write_bench(found.bench_path, bench_payload(settings, {"perf": {"samples": samples,
                                                                  "baseline": {"samples": samples}}}))
    again = discover(entry, root)
    verdict, _reason, detail = perf_verdict(samples, again.settings["results"]["perf"])
    assert verdict in {"pass", "warn"}
    assert "ratio" in detail and detail["baseline_rounds"] == len(samples)
    assert again.settings["results"]["perf"]["baseline"]["samples"] == samples


@needs_engine
def test_a_forced_accuracy_failure_fails_the_mode(tmp_path: Path):
    """A reference the model cannot match must fail, not pass quietly."""
    root = _build(tmp_path / "fixtures", probability=0.02)
    found = discover(_entry(root, modes=["accuracy-boolean"]), root)
    variants = boolean_variants(found.card)
    document = PrivateRunner(context=None, image="", volumes=(), executor="local").run(
        accuracy_payload(str(found.directory), settings_of(found, {}), variants)
    )
    assert document["ok"], document.get("error")
    verdict, _reason, detail = accuracy_verdict(document["variants"], variants)
    assert verdict == "fail", detail
    failed = [row for row in detail["variants"] if row["verdict"] == "fail"]
    assert failed and abs(failed[0]["z"]) > 4.0


# --- the leak test -----------------------------------------------------------


def _collect_strings(node, secrets: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _collect_strings(key, secrets)
            _collect_strings(value, secrets)
    elif isinstance(node, list):
        for item in node:
            _collect_strings(item, secrets)
    elif isinstance(node, str):
        secrets.add(node)
        secrets.update(part for part in node.split() if len(part) >= 4)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        secrets.add(str(node))


def _secret_strings(root: Path) -> list[str]:
    """Every model-specific string the fake private tree holds.

    File and directory names, the fake model name, and every string and number
    in the card and the bench file. Two things are excluded:
    the report format's own vocabulary (``HARNESS_VOCABULARY``), because those
    words name modes, verdicts and fields and occur in every harness output by
    design; and the opaque entry id, which lives in the repository manifest and
    is the one identifier the harness is meant to publish.
    """
    secrets: set[str] = {MODEL, GATE, BAND_A, BAND_B}
    for path in root.rglob("*"):
        secrets.add(path.name)
        secrets.add(path.stem)
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        _collect_strings(document, secrets)
    return sorted(
        (
            value
            for value in secrets
            if len(value) >= 4 and value not in HARNESS_VOCABULARY and value != ENTRY_ID
        ),
        key=len,
        reverse=True,
    )


def _history_output(out: Path, md: Path) -> str:
    """The published history, built from the results file as #123 does."""
    if not out.is_file():
        return ""
    from benchmarks.history.index import merge_runs, model_summaries

    payload = json.loads(out.read_text(encoding="utf-8"))
    index = json.dumps(
        model_summaries(merge_runs([payload], version="develop")["models"]), sort_keys=True
    )
    return index + ("\n" + md.read_text(encoding="utf-8") if md.is_file() else "")


def _comment(out: Path) -> str:
    """``scripts/ci/bench_comment.py`` on the same results file, as the gate does."""
    script = ROOT / "scripts" / "ci" / "bench_comment.py"
    work = out.parent / "comment"
    work.mkdir(parents=True, exist_ok=True)
    target = work / "cpu.json"
    if out.is_file():
        payload = json.loads(out.read_text(encoding="utf-8"))
        payload["context"] = {"target": "cpu", "report_only": False}
        target.write_text(json.dumps(payload), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cpu",
            str(target),
            "--trusted",
            "--run-benches",
            "--out",
            str(work / "comment.md"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    written = work / "comment.md"
    return (written.read_text(encoding="utf-8") if written.is_file() else "") + "\n" + proc.stdout


def _scan(published: dict[str, str], secrets: list[str]) -> dict[str, list[str]]:
    hits = {
        secret: [where for where, text in published.items() if secret in text]
        for secret in secrets
    }
    return {secret: where for secret, where in hits.items() if where}


def test_leak_scan_on_a_run(tmp_path: Path):
    """stdout, stderr, the results JSON, the markdown, history and the PR comment."""
    root = _build(tmp_path / "fixtures")
    manifest = _manifest(root)
    # Collected before the run: it appends a timestamp to the private bench file,
    # and a timestamp is not a model string.
    secrets = _secret_strings(root)
    out = tmp_path / "results.json"
    md = tmp_path / "results.md"
    proc = _cli(
        "private", "--v2-manifest", str(manifest), "--private", str(root),
        "--out", str(out), "--md", str(md), "--mode", "compile", "--execute", "local",
    )
    assert proc.returncode == 0, proc.stderr
    published = {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "results": out.read_text(encoding="utf-8") if out.is_file() else "",
        "markdown": md.read_text(encoding="utf-8") if md.is_file() else "",
        "history": _history_output(out, md),
        "comment": _comment(out),
    }
    assert not _scan(published, secrets), _scan(published, secrets)
    # The only private content published is the one-line verdict per entry and mode.
    document = json.loads(published["results"])
    assert document["private"]
    for row in document["private"]:
        assert set(row) <= {"id", "mode", "verdict", "reason"}
    assert f"private {ENTRY_ID} compile:" in published["stdout"]


def test_leak_scan_when_details_are_recorded(tmp_path: Path):
    """With the details going to a private file, none of them reaches an output."""
    root = _build(tmp_path / "fixtures")
    manifest = _manifest(root)
    secrets = _secret_strings(root)
    details = tmp_path / "details.json"
    out = tmp_path / "results.json"
    md = tmp_path / "results.md"
    proc = _cli(
        "private", "--v2-manifest", str(manifest), "--private", str(root),
        "--private-results", str(details), "--out", str(out), "--md", str(md),
        "--mode", "compile", "--execute", "local",
    )
    assert proc.returncode == 0, proc.stderr
    published = {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "results": out.read_text(encoding="utf-8") if out.is_file() else "",
        "markdown": md.read_text(encoding="utf-8") if md.is_file() else "",
        "history": _history_output(out, md),
        "comment": _comment(out),
    }
    assert not _scan(published, secrets), _scan(published, secrets)
    if shutil.which("fzb") is None:
        pytest.skip("fzb is not on PATH (install the wheel): no details were produced")
    # The details did go to the private file, settings included.
    document = json.loads(details.read_text(encoding="utf-8"))
    assert document["id"] == ENTRY_ID
    assert document["results"]["compile"]["levels"]
    assert document["compile"]["levels"] == [0, 1]
