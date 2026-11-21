"""Reference estimators (#179): exact values on the public FTP conformance
fixtures, Monte Carlo and hull estimates within 4 standard errors, seed
determinism, the parser rules, independence from fuzzybits, and (only when
FUZZYBITS_PRIVATE_FIXTURES_DIR is set) the private model cards. CPU only."""

from __future__ import annotations

import functools
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from benchmarks.reference import (  # noqa: E402
    LaneChecker,
    ModelError,
    estimate_boolean,
    estimate_hull,
    exact_probability,
    exact_table,
    parse_project,
    run,
    structure_counts,
)
from benchmarks.reference import hull as hull_mod  # noqa: E402
from benchmarks.reference.circuit import build_circuit, polarity  # noqa: E402
from benchmarks.reference.fixtures import (  # noqa: E402
    FIXTURES,
    compute,
    diagnostic_only,
    differences,
    fixture_dirs,
    load,
    project_of,
)

FIXTURE_NAMES = [
    "delete_term_flags",
    "header_limit",
    "imports_tags_duplicates",
    "k_of_n_continuation",
    "multi_file_project",
    "names_special",
    "no_process",
    "process_multi_target",
    "success_branch_delete_term",
]
# Diagnostic-only fixtures (#191): (severity, code, name, line) in report order,
# the same under every duplicate-import policy.
INVALID_FIXTURES = {
    "invalid_import_section_end": [
        ("error", "import-without-header", "B", 10),
        ("error", "import-without-header", "C", 12),
        ("error", "undefined-reference", "B", 3),
        ("error", "undefined-reference", "C", 4),
    ],
    "invalid_import_without_header": [
        ("error", "import-without-header", "A", 7),
        ("error", "import-without-header", "B", 8),
        ("error", "import-without-header", "C", 9),
        ("error", "undefined-reference", "A", 4),
        ("error", "undefined-reference", "B", 4),
        ("error", "undefined-reference", "C", 5),
        ("warning", "missing-process", None, None),
    ],
    "invalid_no_endtree": [("error", "missing-endtree", None, 2)],
    "invalid_second_tree_section": [
        ("error", "misplaced-gate", "G1", 4),
        ("error", "misplaced-gate", "G2", 5),
        ("error", "undefined-reference", "G1", 2),
        ("error", "undefined-reference", "G2", 2),
    ],
    "invalid_undefined_event": [("error", "undefined-reference", "C", 4)],
}
TIMING = {"seconds", "trials_per_second", "check_seconds", "lanes_per_second"}
MC_TRIALS = 1 << 16
Z_MAX = 4.0


def _cases(fixture: Path):
    """(policy, overrides, target, expected {boolean, hull}) for every result
    recorded in the fixture."""
    spec = load(fixture)
    for policy, block in spec["policies"].items():
        if policy != "first" or block["results"] is None:
            continue
        for target, values in block["results"].items():
            yield policy, {}, target, values
    for v in spec.get("variants", []):
        for target, values in v["results"].items():
            yield v["policy"], v["overrides"], target, values


@pytest.fixture(scope="module", params=FIXTURE_NAMES)
def fixture(request) -> Path:
    return FIXTURES / request.param


# --- fixtures --------------------------------------------------------------


def test_fixture_inventory():
    assert [p.name for p in fixture_dirs()] == sorted(FIXTURE_NAMES + list(INVALID_FIXTURES))
    for fx in fixture_dirs():
        spec = load(fx)
        assert spec["schema_version"] == 1
        assert all((fx / name).is_file() for name in spec["files"])
        assert diagnostic_only(spec) == fx.name.startswith("invalid_"), fx.name
        if not diagnostic_only(spec):
            assert spec["structure"]["basic_events"] <= 20, fx.name


@pytest.mark.parametrize("name", FIXTURE_NAMES + list(INVALID_FIXTURES))
def test_fixture_expected_is_current(name):
    """Structure counts, diagnostics per duplicate-import policy and exact
    probabilities recomputed from the .ftp files match expected.json."""
    fx = FIXTURES / name
    spec = load(fx)
    assert differences(spec, compute(fx, spec)) == []


@pytest.mark.parametrize("name", list(INVALID_FIXTURES))
def test_invalid_fixture_diagnostics(name):
    """Diagnostic-only fixtures: the hand-written diagnostics under every
    policy, no results, no structure, and the CLI refuses to evaluate."""
    fx = FIXTURES / name
    spec = load(fx)
    assert set(spec) == {"schema_version", "description", "covers", "files", "policies"}
    want = INVALID_FIXTURES[name]
    for policy, block in spec["policies"].items():
        got = [
            (d["severity"], d["code"], d["name"], d["locations"][0][1] if d["locations"] else None)
            for d in block["diagnostics"]
        ]
        assert got == want, policy
        assert all(len(d["locations"]) <= 1 for d in block["diagnostics"])
        assert block["results"] is None
    out = _cli("exact", str(fx))
    assert out.returncode == 1
    errors = {code for severity, code, _, _ in want if severity == "error"}
    assert errors <= set(re.findall(r"^error: ([a-z-]+):", out.stderr, re.MULTILINE))


def _exact(name: str, target: str, semantics: str, policy: str = "first", overrides=None) -> float:
    fx = FIXTURES / name
    project = project_of(fx, load(fx))
    probs, _ = project.probabilities(policy)
    return exact_probability(project, target, probs, semantics, overrides)["probability"]


def _two_of_three(a: float, b: float, c: float) -> float:
    return a * b + a * c + b * c - 2 * a * b * c


HAND = [
    # (fixture, target, semantics, policy, overrides, closed form)
    ("header_limit", "TOP", "boolean", "first", None, 0.2 + 0.8 * 0.1 * 0.3),
    ("header_limit", "TOP", "hull", "first", None, 0.2 + 0.8 * 0.1 * 0.3),
    ("imports_tags_duplicates", "TOP", "boolean", "first", None, 1 - (1 - 0.1 * 0.3 * 0.4) * (1 - 0.2 * 0.5)),
    ("imports_tags_duplicates", "TOP", "boolean", "last", None, 1 - (1 - 0.1 * 0.2 * 0.4) * (1 - 0.2 * 0.5)),
    ("delete_term_flags", "SEQ", "boolean", "error", None, 0.0),
    ("delete_term_flags", "SEQ", "hull", "error", None, 0.1 * (1 - 0.95 * 0.94)),
    ("delete_term_flags", "SEQ", "boolean", "error", {"DEL": 0},
     0.1 * ((1 - 0.95 * 0.94) - 0.4 * 0.2 * (1 - 0.95 * 0.7))),
    ("delete_term_flags", "SEQ", "hull", "error", {"DEL": 0}, 0.1 * (1 - 0.95 * 0.94)),
    ("names_special", "1-TOP", "boolean", "first", None,
     1 - (1 - 0.1 * 0.2 * 0.7) * (1 - _two_of_three(0.4, 0.5, 0.6))),
    ("names_special", "1-TOP", "hull", "first", None,
     1 - (1 - 0.1 * 0.2) * (1 - _two_of_three(0.4, 0.5, 0.6))),
    ("k_of_n_continuation", "TOP", "boolean", "first", None,
     1 - (0.9 * 0.8 * 0.7) * (1 - 0.5**3) * (0.9**4 + 4 * 0.1 * 0.9**3) * 0.95 * 0.99**9),
    ("process_multi_target", "T2", "boolean", "first", None, (1 - 0.7 * 0.6) * 0.8),
    ("process_multi_target", "T2", "hull", "first", None, 1 - 0.7 * 0.6),
    ("process_multi_target", "T3", "boolean", "first", None, _two_of_three(0.1, 0.2, 0.3)),
    ("success_branch_delete_term", "SEQ-2", "boolean", "first", None, 0.1 * (1 - 0.8 * 0.9) * 0.75 * 0.85),
    ("success_branch_delete_term", "SEQ-2", "hull", "first", None, 0.1 * (1 - 0.8 * 0.9)),
    ("success_branch_delete_term", "SEQ-3", "hull", "first", None, 0.1 * (0.1 * (1 - 0.75 * 0.85) + 0.9 * 0.2 * 0.15)),
]


@pytest.mark.parametrize("name,target,semantics,policy,overrides,want", HAND)
def test_exact_matches_closed_form(name, target, semantics, policy, overrides, want):
    assert _exact(name, target, semantics, policy, overrides) == pytest.approx(want, rel=1e-12, abs=1e-15)


def test_duplicate_import_policies():
    fx = FIXTURES / "imports_tags_duplicates"
    spec = load(fx)
    err = spec["policies"]["error"]
    assert err["results"] is None
    assert [d["code"] for d in err["diagnostics"]] == ["duplicate-import"]
    assert err["diagnostics"][0]["severity"] == "error"
    assert spec["policies"]["first"]["diagnostics"][0]["severity"] == "warning"
    project = project_of(fx, spec)
    probs, _ = project.probabilities("error")
    assert probs["A"] == 0.3  # resolved like "first", but evaluation refuses
    with pytest.raises(ModelError):
        run("exact", [str(fx / f) for f in spec["files"]], policy="error")


# --- estimators ------------------------------------------------------------


def _null_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def test_boolean_mc_within_4se(fixture):
    project = project_of(fixture, load(fixture))
    for policy, overrides, target, values in _cases(fixture):
        probs, _ = project.probabilities(policy)
        r = estimate_boolean(project, target, probs, overrides, trials=MC_TRIALS, seed=7)
        p = values["boolean"]
        if p in (0.0, 1.0):
            assert r["estimate"] == p, (target, overrides)
            continue
        assert abs(r["estimate"] - p) <= Z_MAX * _null_se(p, r["trials"]), (target, overrides, r["estimate"], p)


def test_hull_estimate_within_4se(fixture):
    project = project_of(fixture, load(fixture))
    for policy, overrides, target, values in _cases(fixture):
        probs, _ = project.probabilities(policy)
        r = estimate_hull(project, target, probs, overrides, trials=MC_TRIALS, seed=11)
        p = values["hull"]
        if p in (0.0, 1.0):
            assert r["estimate"] == p, (target, overrides)
            continue
        n = r["trials"]
        assert abs(r["estimate"] - p) <= Z_MAX * _null_se(p, n), (target, overrides, r["estimate"], p)
        if "bounds" in r and r["method"] != "constant-folding":
            assert r["bounds"]["lower"] <= r["estimate"] <= r["bounds"]["upper"]
            assert r["lanes"]["checked"] == r["lanes"]["undecided"]


def test_hull_subsampled_checks_within_4se():
    """With a check cap the stratified estimator and its delta-method SE hold."""
    fx = FIXTURES / "success_branch_delete_term"
    project = project_of(fx, load(fx))
    probs, _ = project.probabilities("first")
    for target in ("TOP", "SEQ-3"):
        p = _exact("success_branch_delete_term", target, "hull")
        r = estimate_hull(project, target, probs, trials=1 << 18, seed=3, max_checks=300)
        lanes = r["lanes"]
        assert lanes["checked"] == 300 < lanes["undecided"]
        if 0 < lanes["checked_true"] < lanes["checked"]:  # subsampling adds variance unless f is 0 or 1
            assert r["se"] > _null_se(r["estimate"], r["trials"])
        assert abs(r["estimate"] - p) <= Z_MAX * r["se"], (target, r["estimate"], p, r["se"])


def test_delete_term_flags_boolean_zero_hull_positive():
    fx = FIXTURES / "delete_term_flags"
    project = project_of(fx, load(fx))
    probs, _ = project.probabilities("error")
    b = estimate_boolean(project, "SEQ", probs, trials=MC_TRIALS, seed=1)
    h = estimate_hull(project, "SEQ", probs, trials=MC_TRIALS, seed=1)
    assert b["method"] == "constant-folding" and b["estimate"] == 0.0
    assert h["estimate"] > 0.0 and h["bounds"]["lower"] == 0.0


def _strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in TIMING}


def test_seed_determinism():
    fx = FIXTURES / "success_branch_delete_term"
    project = project_of(fx, load(fx))
    probs, _ = project.probabilities("first")
    for est in (estimate_boolean, estimate_hull):
        a = est(project, "TOP", probs, trials=1 << 17, seed=42)
        b = est(project, "TOP", probs, trials=1 << 17, seed=42)
        c = est(project, "TOP", probs, trials=1 << 17, seed=43)
        assert _strip(a) == _strip(b)
        assert a["estimate"] != c["estimate"]


# --- exact per-lane hull check against the upward closure --------------------


def _lane_check_agrees(fixture: Path, max_lanes: int = 4096, bdd_budget: int = 1 << 16) -> Counter:
    """Check lanes (all, or a sample) against the exact upward closure;
    returns how the lanes were decided."""
    spec = load(fixture)
    project = project_of(fixture, spec)
    rng = np.random.default_rng(0)
    methods: Counter = Counter()
    for policy, overrides, target, _ in _cases(fixture):
        probs, _ = project.probabilities(policy)
        tt = exact_table(project, target, probs, "hull", overrides)
        c = build_circuit(project, target, probs, overrides, constants="hull")
        if c.constant() is not None:
            continue
        checker = LaneChecker(c, polarity(c), bdd_budget)
        n = len(tt.names)
        fixed = [i for i, p in enumerate(tt.probs) if p in (0.0, 1.0)]
        idx = np.arange(1 << n)
        for i in fixed:  # lanes are reachable only with p = 0 events false and p = 1 events true
            idx = idx[((idx >> i) & 1) == int(tt.probs[i])]
        if len(idx) > max_lanes:
            idx = rng.choice(idx, size=max_lanes, replace=False)
        col = {name: i for i, name in enumerate(tt.names)}
        x = [0] + [
            int.from_bytes(np.packbits(((idx >> col[c.names[v]]) & 1).astype(bool), bitorder="little").tobytes(),
                           "little")
            for v in range(1, c.first_gate)
        ]
        got = checker.check(x, len(idx))
        assert got == tt.table[idx].tolist(), (fixture.name, target, overrides)
        methods.update(checker.methods)
    return methods


def test_lane_check_matches_upward_closure(fixture):
    _lane_check_agrees(fixture)


@pytest.mark.parametrize("mode,budget", [("bdd", 1 << 16), ("search", 3)])
def test_lane_check_fallbacks_match_upward_closure(monkeypatch, mode, budget):
    """With the truth table disabled, the BDD (and, over budget, the complete
    search) agree with the exact upward closure."""
    monkeypatch.setattr(hull_mod, "TRUTH_TABLE_MAX", 0)
    methods: Counter = Counter()
    for name in ("success_branch_delete_term", "process_multi_target", "names_special", "delete_term_flags"):
        methods += _lane_check_agrees(FIXTURES / name, max_lanes=512, bdd_budget=budget)
    assert methods[mode] > 0 and methods["truth-table"] == 0


# --- parser ----------------------------------------------------------------


def _write(tmp_path: Path, text: str, name: str = "m.ftp", newline: str = "\n") -> Path:
    path = tmp_path / name
    path.write_bytes(text.replace("\n", newline).encode("latin-1"))
    return path


def test_parser_rules(tmp_path):
    p = parse_project([_write(tmp_path, """**CHAR32
TOP +
    G1
\t-G2
G1 2 A B C
G2 * A -B
ENDTREE
PROCESS TOP
G1
IMPORT
  1E-1 A
  0.2 B I
  3.0E-01 C
LIMIT 1E-9
*XEQ
""")])
    assert p.diagnostics == []
    assert p.gates["TOP"].children == [("G1", False), ("G2", True)]
    assert p.gates["G1"].k == 2 and p.gates["G2"].children == [("A", False), ("B", True)]
    assert p.process == ["TOP", "G1"]
    imports = [(i.name, i.value, i.initiator) for i in p.imports]
    assert imports == [("A", 0.1, False), ("B", 0.2, True), ("C", 0.3, False)]
    assert p.limit == 1e-9 and p.headers == ["**CHAR32", "*XEQ"]


def test_parser_crlf_matches_lf(tmp_path):
    fx = FIXTURES / "success_branch_delete_term"
    text = (fx / "model.ftp").read_text()
    lf = parse_project([_write(tmp_path, text, "lf.ftp")])
    crlf = parse_project([_write(tmp_path, text, "crlf.ftp", "\r\n")])
    assert structure_counts(lf) == structure_counts(crlf)


@pytest.mark.parametrize(
    "text,codes",
    [
        ("  A B\nG + A\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n", {"invalid-line"}),
        ("G ? A\nG + A\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n", {"invalid-line"}),
        ("G 3 A B\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n0.2 B\n", {"invalid-gate"}),
        ("G + A X\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n", {"undefined-reference"}),
        ("G + A\nG + A\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n", {"duplicate-gate"}),
        ("G + A\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n0.2 G\n", {"gate-imported"}),
        ("G + A\nENDTREE\nPROCESS G\nIMPORT\n1.5 A\n", {"invalid-probability"}),
        ("G + --A B\nENDTREE\nPROCESS G\nIMPORT\n0.1 B\n", {"invalid-line"}),
        # #191: ENDTREE and IMPORT are required; undefined names are errors.
        ("G + A\nPROCESS G\nIMPORT\n0.1 A\n", {"missing-endtree"}),
        ("G + A\nIMPORT\n0.1 A\nPROCESS G\n", {"missing-endtree"}),
        ("G + A\nLIMIT 1E-8\nPROCESS G\nIMPORT\n0.1 A\n", {"missing-endtree"}),
        ("G + A\n", {"missing-endtree", "undefined-reference"}),
        ("G + A\n  0.1 A\nENDTREE\nPROCESS G\n", {"import-without-header", "undefined-reference"}),
        ("G + A\nENDTREE\n0.1 A\nPROCESS G\n", {"import-without-header", "undefined-reference"}),
        ("G + A\nENDTREE\nPROCESS G\n0.1 A I\n", {"import-without-header", "undefined-reference"}),
        ("G + A B\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\nLIMIT 1E-8\n0.2 B\n",
         {"import-without-header", "undefined-reference"}),
        ("G + A B\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n**FTITLE: x\n0.2 B\n",
         {"import-without-header", "undefined-reference"}),
        ("G + H\nENDTREE\nH + A\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n", {"misplaced-gate", "undefined-reference"}),
        ("G + A\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\nH + A\n  A\n", {"misplaced-gate"}),
        ("G + A\nENDTREE\nPROCESS G X\nIMPORT\n0.1 A\n", {"undefined-process-target"}),
        ("G + A\nENDTREE\nPROCESS G\n**FTITLE: x\nG\nIMPORT\n0.1 A\n", {"invalid-line"}),
    ],
)
def test_parser_errors(tmp_path, text, codes):
    p = parse_project([_write(tmp_path, text)])
    assert {d.code for d in p.diagnostics if d.severity == "error"} == codes


def test_parser_diagnostic_locations(tmp_path):
    p = parse_project([_write(tmp_path, "**CHAR32\nG + A\n  B\nPROCESS G\nIMPORT\n0.1 A\n0.2 B\n", "a.ftp"),
                       _write(tmp_path, "H * A B\n", "b.ftp")])
    endtree = [(d.key()["locations"], d.message) for d in p.diagnostics if d.code == "missing-endtree"]
    assert endtree == [
        ([["a.ftp", 2]], "the tree section starting at line 2 has no ENDTREE before PROCESS"),
        ([["b.ftp", 1]], "the tree section starting at line 1 has no ENDTREE before the end of the file"),
    ]
    assert p.gates["G"].children == [("A", False), ("B", False)] and "H" in p.gates


def test_header_lines_between_gate_lines(tmp_path):
    """FTREX and ZEBRA accept header lines inside the tree section."""
    p = parse_project([_write(tmp_path, "TOP * G1 G2\n**FTITLE: t\nG1 + A B\n*XEQ\nG2 + B C\nENDTREE\nPROCESS TOP\n"
                                        "IMPORT\n0.1 A\n0.2 B\n0.3 C\n")])
    assert p.diagnostics == [] and list(p.gates) == ["TOP", "G1", "G2"]


def test_missing_process_warns_and_targets_roots(tmp_path):
    path = _write(tmp_path, "R1 + A G\nR2 * B G\nG * A B\nENDTREE\nIMPORT\n0.1 A\n0.2 B\n")
    p = parse_project([path])
    assert [(d.severity, d.code) for d in p.diagnostics] == [("warning", "missing-process")]
    assert p.default_targets() == p.roots() == ["R1", "R2"]
    doc = run("exact", [str(path)])
    assert [r["target"] for r in doc["results"]] == ["R1", "R2"]


def test_duplicate_import_default_last_keeps_initiator(tmp_path):
    """FTREX and ZEBRA: the last value wins (with a warning) and an I tag on
    any record keeps the event an initiator."""
    path = _write(tmp_path, "TOP * IE A\nENDTREE\nPROCESS TOP\nIMPORT\n0.1 IE I\n0.3 A\n0.5 IE\n")
    p = parse_project([path])
    probs, diags = p.probabilities()
    assert probs == {"IE": 0.5, "A": 0.3} and p.initiators() == {"IE"}
    assert [(d.severity, d.code, d.name) for d in diags] == [("warning", "duplicate-import", "IE")]
    assert diags[0].key()["locations"] == [["m.ftp", 5], ["m.ftp", 7]]
    assert p.probabilities("first")[0]["IE"] == 0.1
    assert p.probabilities("error")[1][0].severity == "error"
    doc = run("exact", [str(path)])
    assert doc["duplicate_import"] == "last"
    assert doc["results"][0]["probability"] == pytest.approx(0.5 * 0.3, rel=1e-12)


def test_cycle_is_an_error(tmp_path):
    p = parse_project([_write(tmp_path, "G + H A\nH * G A\nENDTREE\nIMPORT\n0.1 A\n")])
    probs, _ = p.probabilities()
    with pytest.raises(ModelError, match="cycle"):
        build_circuit(p, "G", probs)
    assert structure_counts(p)["depth"] is None


# --- CLI and independence --------------------------------------------------


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "benchmarks.reference", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )


def test_cli_exact_and_estimate_json():
    fx = str(FIXTURES / "delete_term_flags")
    out = _cli("exact", "--semantics", "hull", "--json", fx)
    assert out.returncode == 0, out.stderr
    doc = json.loads(out.stdout)
    assert doc["results"][0]["probability"] == pytest.approx(0.0107, rel=1e-12)
    out = _cli("exact", "--semantics", "boolean", "--override", "DEL=0", "--json", fx)
    assert json.loads(out.stdout)["results"][0]["probability"] == pytest.approx(0.00802, rel=1e-12)
    args = ("estimate", "--semantics", "hull", "--trials", "65536", "--seed", "5", "--json", fx)
    a, b = json.loads(_cli(*args).stdout), json.loads(_cli(*args).stdout)
    assert _strip(a["results"][0]) == _strip(b["results"][0])


def test_cli_duplicate_import_policy():
    fx = str(FIXTURES / "imports_tags_duplicates")
    out = _cli("exact", fx)
    assert out.returncode == 0 and "warning: duplicate-import" in out.stderr
    out = _cli("exact", "--target", "TOP", "--json", fx)
    assert out.returncode == 0
    doc = json.loads(out.stdout)
    assert doc["duplicate_import"] == "last"
    assert [(d["severity"], d["code"]) for d in doc["diagnostics"]] == [("warning", "duplicate-import")]
    assert doc["results"][0]["probability"] == pytest.approx(0.1072, rel=1e-12)
    out = _cli("exact", "--duplicate-import", "first", "--target", "TOP", "--json", fx)
    assert json.loads(out.stdout)["results"][0]["probability"] == pytest.approx(0.1108, rel=1e-12)
    out = _cli("exact", "--duplicate-import", "error", fx)
    assert out.returncode == 1 and "error: duplicate-import" in out.stderr


def test_independence_from_fuzzybits():
    """Importing and running the reference package loads no fuzzybits, product
    or harness module."""
    code = (
        "import json, sys; import benchmarks.reference as r; "
        "r.run('exact', [sys.argv[1]], semantics='hull'); print(json.dumps(sorted(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code, str(FIXTURES / "header_limit")],
                         cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    mods = json.loads(out.stdout)
    banned = ("fuzzybits", "_fuzzybits", "scram", "benchmarks.harness", "benchmarks.loader",
              "benchmarks.datasets", "benchmarks.history", "src", "python")
    assert [m for m in mods if any(m == b or m.startswith(b + ".") for b in banned)] == []
    imports = re.compile(r"^\s*(?:import|from)\s+(\S+)", re.MULTILINE)
    allowed = re.compile(r"(\.[a-z_]*|__future__|argparse|collections|dataclasses|functools|json|math|numpy|"
                         r"pathlib|re|statistics|sys|time|typing)\Z")
    for src in sorted((ROOT / "benchmarks" / "reference").glob("*.py")):
        bad = [m for m in imports.findall(src.read_text()) if not allowed.match(m)]
        assert bad == [], (src.name, bad)


# --- private models (FUZZYBITS_PRIVATE_FIXTURES_DIR) -------------------------

PRIVATE_ENV = "FUZZYBITS_PRIVATE_FIXTURES_DIR"
PRIVATE_TRIALS = int(float(os.environ.get("FUZZYBITS_PRIVATE_TRIALS", 1 << 22)))
PRIVATE_HULL_TRIALS = 1 << 18
PRIVATE_HULL_CHECKS = 400


def _private_cards() -> list[Path]:
    root = os.environ.get(PRIVATE_ENV)
    if not root:
        return []
    ftp = Path(root) / "ftp"
    if not ftp.is_dir():
        return []
    return [d / f"{d.name}.card.json" for d in sorted(ftp.iterdir()) if (d / f"{d.name}.card.json").is_file()]


_CARDS = _private_cards()
_VARIANTS = [
    (i, j) for i, card in enumerate(_CARDS) for j in range(len(json.loads(card.read_text())["variants"]))
]


@functools.lru_cache(maxsize=4)
def _private_project(i: int):
    card_path = _CARDS[i]
    card = json.loads(card_path.read_text())
    return card, parse_project([card_path.parent / f for f in card["files"]])


def _same_structure(key: str, mine, want) -> bool:
    if key == "duplicate_imports":
        if set(mine) != set(want):
            return False
        for name, occ in want.items():
            if len(mine[name]) != len(occ):
                return False
            for o, w in zip(mine[name], occ, strict=True):
                if any(o.get(f) != w[f] for f in ("value", "initiator", "line") if f in w):
                    return False
        return True
    return mine == want


@pytest.mark.private
@pytest.mark.parametrize("i", list(range(len(_CARDS))) or [None], ids=lambda i: f"model{i}")
def test_private_structure(i):
    if i is None:
        pytest.skip(f"{PRIVATE_ENV} not set")
    card, project = _private_project(i)
    mine = structure_counts(project)
    keys = [k for k in card["structure"] if k in mine]
    bad = sorted(k for k in keys if not _same_structure(k, mine[k], card["structure"][k]))
    print(f"private model{i} structure: {len(keys) - len(bad)}/{len(keys)} keys match")
    assert not bad, f"model{i}: structure keys differ: {bad}"


@pytest.mark.private
@pytest.mark.parametrize("ij", _VARIANTS or [None], ids=lambda ij: f"model{ij[0]}-variant{ij[1]}" if ij else "none")
def test_private_variant(ij):
    if ij is None:
        pytest.skip(f"{PRIVATE_ENV} not set")
    i, j = ij
    label = f"model{i} variant{j}"
    card, project = _private_project(i)
    var = list(card["variants"].values())[j]
    if "edit" in var:
        pytest.skip(f"{label}: file edit, not expressible as overrides")
    semantics = "hull" if var["semantics"] == "monotone_hull" else "boolean"
    overrides = {k: int(bool(v)) for k, v in var.get("overrides", {}).items()}
    policy = var.get("policy", card.get("policy", {})).get("duplicate_import", "error")
    probs, _ = project.probabilities(policy)
    target = project.default_targets()[0]
    if semantics == "boolean":
        r = estimate_boolean(project, target, probs, overrides, trials=PRIVATE_TRIALS, seed=1)
        extra = f"trials/s={r.get('trials_per_second') or 0:.3g}"
    else:
        r = estimate_hull(project, target, probs, overrides, trials=PRIVATE_HULL_TRIALS, seed=1,
                          max_checks=PRIVATE_HULL_CHECKS)
        extra = f"checked={r.get('lanes', {}).get('checked', 0)} lanes/s={r.get('lanes_per_second') or 0:.0f}"
    est, se, n = r["estimate"], r["se"], r["trials"]
    expected = var.get("expected")
    observed = var.get("observed")
    if expected is None:
        if not isinstance(observed, dict) or "estimate" not in observed or "n" not in observed:
            pytest.skip(f"{label}: no expected value")
        ref, ref_se, kind = observed["estimate"], _null_se(observed["estimate"], observed["n"]), "observed"
    else:
        ref, kind = expected["probability"], expected["kind"]
        if kind == "exact":
            ref_se = 0.0
        elif "se" in expected:
            ref_se = expected["se"]
        else:
            ref_se = expected.get("rel_tolerance_95", 0.0) * ref / 1.96
    if kind == "exact" and r["method"] == "constant-folding":
        ok, z = est == ref, 0.0
    else:
        sigma = math.sqrt(max(se, _null_se(ref, n)) ** 2 + ref_se**2)
        z = (est - ref) / sigma if sigma > 0 else (0.0 if est == ref else math.inf)
        ok = abs(z) <= Z_MAX
    print(f"private {label}: {semantics} {r['method']} estimate={est:.4e} +- {se:.2e} z={z:+.2f} "
          f"[{'PASS' if ok else 'FAIL'}] {extra}")
    if kind == "observed":
        pytest.skip(f"{label}: observed-only reference (not asserted), z={z:+.2f}")
    assert ok, f"{label}: z={z:+.2f}"
