"""Reference estimators (#179, #190): exact values on the public FTP conformance
fixtures (Boolean, omega_L hull, plain hull, initiator rule), minimal cut sets,
Monte Carlo and hull estimates within 4 standard errors, the per-lane check
against brute-force omega_L on random circuits, seed determinism, the parser
rules, independence from fuzzybits, and (only when FUZZYBITS_PRIVATE_FIXTURES_DIR
is set) the private model cards. CPU only."""

from __future__ import annotations

import functools
import itertools
import json
import math
import os
import random
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
    exact_cut_sets,
    exact_probability,
    exact_table,
    parse_project,
    run,
    structure_counts,
)
from benchmarks.reference import hull as hull_mod  # noqa: E402
from benchmarks.reference.bitpar import ALL, atom_sampler, coherent_specs, evaluate  # noqa: E402
from benchmarks.reference.circuit import (  # noqa: E402
    BOTH,
    NEITHER,
    ONLY_NEG,
    ONLY_POS,
    build_circuit,
    event_modes,
    plan,
    polarity,
    literal_sets,
)
from benchmarks.reference.exact import closure, truth_table  # noqa: E402
from benchmarks.reference.fixtures import (  # noqa: E402
    FIXTURES,
    compute,
    diagnostic_only,
    differences,
    fixture_dirs,
    load,
    project_of,
)
from benchmarks.reference.ftp import AND, ATLEAST, OR, POLICIES, Gate, Import, Location, Project  # noqa: E402

FIXTURE_NAMES = [
    "complement_under_negation",
    "delete_term_flags",
    "dialect_names_case",
    "dialect_values_layout",
    "header_limit",
    "imports_tags_duplicates",
    "initiator_rule",
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
    """(policy, overrides, max_initiators, target, expected {boolean, hull,
    hull_positive, ...}) for every result recorded in the fixture."""
    spec = load(fixture)
    for policy, block in spec["policies"].items():
        if policy != "first" or block["results"] is None:
            continue
        for target, values in block["results"].items():
            yield policy, {}, None, target, values
    for v in spec.get("variants", []):
        for target, values in v["results"].items():
            yield v["policy"], v.get("overrides", {}), v.get("max_initiators_per_cutset"), target, values


@pytest.fixture(scope="module", params=FIXTURE_NAMES)
def fixture(request) -> Path:
    return FIXTURES / request.param


# --- fixtures --------------------------------------------------------------


def test_fixture_inventory():
    assert [p.name for p in fixture_dirs()] == sorted(FIXTURE_NAMES + list(INVALID_FIXTURES))
    for fx in fixture_dirs():
        spec = load(fx)
        assert spec["schema_version"] == 2
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


def _exact(name: str, target: str, semantics: str, policy: str = "first", overrides=None, literals="split",
           rule=None) -> float:
    fx = FIXTURES / name
    project = project_of(fx, load(fx))
    probs, _ = project.probabilities(policy)
    return exact_probability(project, target, probs, semantics, overrides, literals, rule)["probability"]


def _two_of_three(a: float, b: float, c: float) -> float:
    return a * b + a * c + b * c - 2 * a * b * c


# "hull" is the atom-split hull (default), "omega" omega_L, "positive" the plain monotone hull.
HAND = [
    # (fixture, target, semantics, policy, overrides, max_initiators, closed form)
    ("header_limit", "TOP", "boolean", "first", None, None, 0.2 + 0.8 * 0.1 * 0.3),
    ("header_limit", "TOP", "hull", "first", None, None, 0.2 + 0.8 * 0.1 * 0.3),
    ("imports_tags_duplicates", "TOP", "boolean", "first", None, None, 1 - (1 - 0.1 * 0.3 * 0.4) * (1 - 0.2 * 0.5)),
    ("imports_tags_duplicates", "TOP", "boolean", "last", None, None, 1 - (1 - 0.1 * 0.2 * 0.4) * (1 - 0.2 * 0.5)),
    # IE3 (p = 1) is an initiator: the literal conjunction removes every world with IE1 or IE2.
    ("imports_tags_duplicates", "TOP", "boolean", "last", None, 1, 0.0),
    ("imports_tags_duplicates", "TOP", "hull", "last", None, 1, 1 - (1 - 0.1 * 0.2 * 0.4) * (1 - 0.2 * 0.5)),
    ("delete_term_flags", "SEQ", "boolean", "error", None, None, 0.0),
    ("delete_term_flags", "SEQ", "hull", "error", None, None, 0.1 * (1 - 0.95 * 0.94)),
    ("delete_term_flags", "SEQ", "boolean", "error", {"DEL": 0}, None,
     0.1 * ((1 - 0.95 * 0.94) - 0.4 * 0.2 * (1 - 0.95 * 0.7))),
    ("delete_term_flags", "SEQ", "hull", "error", {"DEL": 0}, None, 0.1 * (1 - 0.95 * 0.94)),
    # -6H--I is a direct complement: omega_L keeps it, so the hull is the Boolean value.
    ("names_special", "1-TOP", "boolean", "first", None, None,
     1 - (1 - 0.1 * 0.2 * 0.7) * (1 - _two_of_three(0.4, 0.5, 0.6))),
    ("names_special", "1-TOP", "hull", "first", None, None,
     1 - (1 - 0.1 * 0.2 * 0.7) * (1 - _two_of_three(0.4, 0.5, 0.6))),
    ("names_special", "1-TOP", "positive", "first", None, None,
     1 - (1 - 0.1 * 0.2) * (1 - _two_of_three(0.4, 0.5, 0.6))),
    ("k_of_n_continuation", "TOP", "boolean", "first", None, None,
     1 - (0.9 * 0.8 * 0.7) * (1 - 0.5**3) * (0.9**4 + 4 * 0.1 * 0.9**3) * 0.95 * 0.99**9),
    ("process_multi_target", "T2", "boolean", "first", None, None, (1 - 0.7 * 0.6) * 0.8),
    ("process_multi_target", "T2", "hull", "first", None, None, (1 - 0.7 * 0.6) * 0.8),
    ("process_multi_target", "T2", "omega", "first", None, None, (1 - 0.7 * 0.6) * 0.8),
    ("process_multi_target", "T2", "positive", "first", None, None, 1 - 0.7 * 0.6),
    ("process_multi_target", "T3", "boolean", "first", None, None, _two_of_three(0.1, 0.2, 0.3)),
    ("success_branch_delete_term", "SEQ-2", "boolean", "first", None, None, 0.1 * (1 - 0.8 * 0.9) * 0.75 * 0.85),
    ("success_branch_delete_term", "SEQ-2", "hull", "first", None, None, 0.1 * (1 - 0.8 * 0.9)),
    ("success_branch_delete_term", "SEQ-3", "hull", "first", None, None,
     0.1 * (0.1 * (1 - 0.75 * 0.85) + 0.9 * 0.2 * 0.15)),
    # IE-FLAG has p = 1; kept cut sets {IE-1, PUMP-B}, {IE-FLAG, VALVE, -OPER} (and, without
    # the rule, {IE-1, IE-2, PUMP-A}, {IE-FLAG, IE-2, PUMP-A}).
    ("initiator_rule", "TOP", "boolean", "error", None, None,
     0.2 * 0.3 + 0.1 * 0.4 * 0.7 * 0.5 + 0.5 * 0.8 - 0.2 * 0.3 * 0.5 * 0.8),
    ("initiator_rule", "TOP", "hull", "error", None, None, 1 - (1 - 0.1 * 0.4) * (1 - 0.5 * 0.8) * (1 - 0.2 * 0.3)),
    ("initiator_rule", "TOP", "positive", "error", None, None, 1 - (1 - 0.1 * 0.4) * (1 - 0.5) * (1 - 0.2 * 0.3)),
    ("initiator_rule", "TOP", "boolean", "error", None, 1, 0.9 * 0.8 * 0.5 * 0.8),
    ("initiator_rule", "TOP", "hull", "error", None, 1, 1 - (1 - 0.1 * 0.4) * (1 - 0.5 * 0.8)),
    ("initiator_rule", "TOP", "positive", "error", None, 1, 1 - (1 - 0.1 * 0.4) * (1 - 0.5)),
    ("initiator_rule", "TOP", "hull", "error", None, 0, 0.0),
    ("initiator_rule", "SEQ-4", "hull", "error", None, 1, 0.0),
    ("initiator_rule", "SEQ-4", "hull", "error", None, 2, 0.2 * 0.3),
    # Direct complements under negated gates (RFC §8.11): P1 = A not(B or E) or D not B, P4 = C not(C not F).
    ("complement_under_negation", "P1", "boolean", "first", None, None, 0.8 * (1 - (1 - 0.1 * 0.6) * 0.7)),
    ("complement_under_negation", "P1", "hull", "first", None, None, 0.1 + 0.9 * 0.3 * 0.8),
    ("complement_under_negation", "P1", "omega", "first", None, None, 0.8 * (1 - 0.9 * 0.7)),
    ("complement_under_negation", "P1", "positive", "first", None, None, 1 - 0.9 * 0.7),
    ("complement_under_negation", "P4", "boolean", "first", None, None, 0.5 * 0.6),
    ("complement_under_negation", "P4", "hull", "first", None, None, 0.5),
    ("complement_under_negation", "P4", "omega", "first", None, None, 0.5 * 0.6),
    ("complement_under_negation", "P4", "positive", "first", None, None, 0.5 * 0.6),
    ("complement_under_negation", "TOP", "hull", "first", None, None, 1 - (1 - 0.316) * (1 - 0.5)),
    ("complement_under_negation", "P3", "hull", "first", None, None, 0.5 * 0.25 + 0.5 * 0.75),
    ("complement_under_negation", "P3", "omega", "first", None, None, 0.5 * 0.25 + 0.5 * 0.75),
    ("complement_under_negation", "P3", "positive", "first", None, None, 1 - (1 - 0.25 * 0.5) * (1 - 0.75)),
]


@pytest.mark.parametrize("name,target,semantics,policy,overrides,rule,want", HAND)
def test_exact_matches_closed_form(name, target, semantics, policy, overrides, rule, want):
    literals = {"hull": "split", "omega": "omega", "positive": "positive"}.get(semantics, "split")
    semantics = "boolean" if semantics == "boolean" else "hull"
    got = _exact(name, target, semantics, policy, overrides, literals, rule)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


def _cut_sets(name: str, target: str, policy: str = "first", literals: str = "split", rule=None) -> set:
    fx = FIXTURES / name
    project = project_of(fx, load(fx))
    probs, _ = project.probabilities(policy)
    return {" ".join(cs["literals"]) for cs in exact_cut_sets(project, target, probs, None, literals, rule)}


def test_cut_sets_keep_direct_complements():
    """RFC §8.8-8.9: direct complements stay in the minimal cut sets (the vendor
    cut sets); the plain hull drops them."""
    assert _cut_sets("process_multi_target", "T2") == {"-B C", "-B D"}
    assert _cut_sets("process_multi_target", "T2", literals="positive") == {"C", "D"}
    assert _cut_sets("process_multi_target", "T3") == {"A B", "A C", "B C"}
    pairs = {"7J/1 8K--2", "7J/1 9L_X", "8K--2 9L_X"}
    assert _cut_sets("names_special", "1-TOP") == pairs | {"4E 5F/G -6H--I"}
    assert _cut_sets("names_special", "1-TOP", literals="positive") == pairs | {"4E 5F/G"}
    assert _cut_sets("names_special", "1-TOP", literals="omega") == pairs | {"4E 5F/G -6H--I"}


def test_cut_sets_where_the_modes_differ():
    """RFC §8.11: over atoms the vendor cut sets; omega_L and the plain hull differ."""
    fx = "complement_under_negation"
    assert _cut_sets(fx, "TOP") == {"A", "C", "-B D"}
    assert _cut_sets(fx, "TOP", literals="omega") == {"A -B", "-B D", "C F"}
    assert _cut_sets(fx, "TOP", literals="positive") == {"A", "D", "C F"}
    assert _cut_sets(fx, "P4") == {"C"} and _cut_sets(fx, "P4", literals="omega") == {"C F"}
    # No consensus over atoms: {K, N} is not an implicant when M and -M are separate atoms.
    assert _cut_sets(fx, "P3") == {"K M", "-M N"}
    assert _cut_sets(fx, "P3", literals="omega") == {"K M", "-M N", "K N"}


def _project(tmp_path: Path, text: str):
    p = parse_project([_write(tmp_path, text)])
    assert not [d for d in p.diagnostics if d.severity == "error"], p.diagnostics
    probs, _ = p.probabilities()
    return p, probs


def test_split_excludes_event_with_its_complement_atom(tmp_path):
    """Cut sets over atoms that contain B and -B have probability 0 and are not listed."""
    p, probs = _project(tmp_path, "T * B -B\nENDTREE\nPROCESS T\nIMPORT\n0.3 B\n")
    assert exact_cut_sets(p, "T", probs) == []
    assert exact_probability(p, "T", probs, "hull")["probability"] == 0.0
    p, probs = _project(tmp_path, "T * G1 G2\nG1 + A B\nG2 + C -B\nENDTREE\nPROCESS T\nIMPORT\n"
                                  "0.1 A\n0.2 B\n0.3 C\n")
    assert {" ".join(cs["literals"]) for cs in exact_cut_sets(p, "T", probs)} == {"A C", "A -B", "B C"}


def test_split_complement_atom_of_probability_zero_event(tmp_path):
    """RFC Theorem 10(g): the atom -B of a p = 0 event is 1 in every lane and stays
    droppable; under the negated gate H, folding it to TRUE would give 0."""
    p, probs = _project(tmp_path, "T * A -H\nH * A -B\nENDTREE\nPROCESS T\nIMPORT\n0.5 A\n0.0 B\n")
    assert exact_probability(p, "T", probs, "hull")["probability"] == 0.5
    assert exact_probability(p, "T", probs, "hull", literals="omega")["probability"] == 0.0
    r = estimate_hull(p, "T", probs, trials=MC_TRIALS, seed=4)
    assert abs(r["estimate"] - 0.5) <= Z_MAX * _null_se(0.5, r["trials"])


def test_cut_sets_initiator_rule():
    """RFC §8.10: the rule deletes exactly the cut sets with two or more initiators,
    counting the p = 1 flag; the success branch deletes nothing."""
    full = {"IE-1 IE-2 PUMP-A", "IE-1 PUMP-B", "IE-FLAG -OPER VALVE", "IE-2 IE-FLAG PUMP-A"}
    assert _cut_sets("initiator_rule", "TOP", "error") == full
    assert _cut_sets("initiator_rule", "TOP", "error", rule=1) == {"IE-1 PUMP-B", "IE-FLAG -OPER VALVE"}
    assert _cut_sets("initiator_rule", "TOP", "error", "positive", 1) == {"IE-1 PUMP-B", "IE-FLAG VALVE"}
    assert _cut_sets("initiator_rule", "TOP", "error", rule=0) == set()


def _initiator_count(cs: dict, tagged: set) -> int:
    return sum(1 for lit in cs["literals"] if lit in tagged)


@pytest.mark.parametrize("rule", [0, 1, 2])
def test_initiator_rule_deletes_cut_sets(fixture, rule):
    """RFC Theorem 3.2 on every fixture: the cut sets under the rule are the cut
    sets without it that have at most ``rule`` initiators (no fixture has a
    consensus term, so the families are equal, not only their unions)."""
    project = project_of(fixture, load(fixture))
    tagged = project.initiators()
    for policy, overrides, base_rule, target, _ in _cases(fixture):
        if base_rule is not None:
            continue
        probs, _ = project.probabilities(policy)
        for literals in ("split", "omega", "positive"):
            without = exact_cut_sets(project, target, probs, overrides, literals)
            kept = [cs for cs in without if _initiator_count(cs, tagged) <= rule]
            assert exact_cut_sets(project, target, probs, overrides, literals, rule) == kept, (target, literals)


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
    for policy, overrides, rule, target, values in _cases(fixture):
        probs, _ = project.probabilities(policy)
        r = estimate_boolean(project, target, probs, overrides, trials=MC_TRIALS, seed=7, max_initiators=rule)
        p = values["boolean"]
        if p in (0.0, 1.0):
            assert r["estimate"] == p, (target, overrides)
            continue
        assert abs(r["estimate"] - p) <= Z_MAX * _null_se(p, r["trials"]), (target, overrides, r["estimate"], p)


@pytest.mark.parametrize("literals,key", [("split", "hull"), ("omega", "hull_omega"), ("positive", "hull_positive")])
def test_hull_estimate_within_4se(fixture, literals, key):
    project = project_of(fixture, load(fixture))
    for policy, overrides, rule, target, values in _cases(fixture):
        probs, _ = project.probabilities(policy)
        r = estimate_hull(project, target, probs, overrides, trials=MC_TRIALS, seed=11, literals=literals,
                          max_initiators=rule)
        assert r["complemented_events"] == values["complemented_events"]
        p = values[key]
        if p in (0.0, 1.0):
            assert r["estimate"] == p, (target, overrides)
            continue
        n = r["trials"]
        assert abs(r["estimate"] - p) <= Z_MAX * _null_se(p, n), (target, overrides, rule, r["estimate"], p)
        if "bounds" in r and r["method"] != "constant-folding":
            assert r["bounds"]["lower"] <= r["estimate"] <= r["bounds"]["upper"]
            assert r["lanes"]["checked"] == r["lanes"]["undecided"]


def test_kept_complements_tighten_the_coherent_bound():
    """With the direct complement kept (an atom, or a fixed event), the coherent
    bound equals g on T2 and 1-TOP: no lane is a candidate (RFC §4.4, Theorem 10(e))."""
    for name, target in (("process_multi_target", "T2"), ("names_special", "1-TOP")):
        fx = FIXTURES / name
        project = project_of(fx, load(fx))
        probs, _ = project.probabilities("first")
        for literals in ("split", "omega"):
            r = estimate_hull(project, target, probs, trials=MC_TRIALS, seed=2, literals=literals)
            assert r["bounds"]["lower"] == r["bounds"]["upper"] == r["estimate"]
            assert r["lanes"]["undecided"] == 0
        plain = estimate_hull(project, target, probs, trials=MC_TRIALS, seed=2, literals="positive")
        assert plain["bounds"]["upper"] > plain["bounds"]["lower"]


def test_fixtures_where_split_and_omega_agree():
    """RFC Theorem 10(c): without a complement-under-negation warning the
    atom-split hull equals omega_L; the divergence fixture has warnings and
    differs (and exceeds the plain hull on P4, Theorem 10(d))."""
    for fx in fixture_dirs():
        spec = load(fx)
        if diagnostic_only(spec):
            continue
        blocks = [b["results"] for b in spec["policies"].values() if b["results"]]
        blocks += [v["results"] for v in spec.get("variants", [])]
        for results in blocks:
            for target, r in results.items():
                if not r["complement_under_negation"]:
                    assert r["hull"] == r["hull_omega"], (fx.name, target)
                assert set(r["complement_under_negation"]) <= set(r["complemented_events"])
                assert r["boolean"] <= r["hull"] + 1e-15 and r["hull_omega"] <= r["hull"] + 1e-15
        if fx.name != "complement_under_negation":
            assert all(not r["complement_under_negation"] for results in blocks for r in results.values()), fx.name
    r = load(FIXTURES / "complement_under_negation")["policies"]["last"]["results"]
    assert r["P1"]["complement_under_negation"] == ["B"] and r["P4"]["complement_under_negation"] == ["F"]
    assert r["P1"]["hull"] != r["P1"]["hull_omega"] and r["P4"]["hull"] > r["P4"]["hull_positive"]


def test_atom_sampler_couples_event_and_complement_atom():
    """The complement atom -B and B are drawn once and are exact complements in
    every lane, in the pass and in the candidate redraw."""
    fx = FIXTURES / "complement_under_negation"
    project = project_of(fx, load(fx))
    probs, _ = project.probabilities()
    c = build_circuit(project, "P1", probs, constants="hull", split=["B"])
    a = c.names.index("-B")
    assert c.twins == {a: c.names.index("B")} and c.probs[a] == pytest.approx(0.8)
    for state_seed in (1, 2):
        rng = np.random.Generator(np.random.PCG64(state_seed))
        ones = np.full(4, ALL, dtype=np.uint64)
        sample = atom_sampler(c, rng, 4, ones)
        vals = {v: sample(v) for v in range(1, c.first_gate)}
        assert np.array_equal(vals[a], vals[c.twins[a]] ^ ones)


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


# --- exact per-lane hull check against the closure ----------------------------


def _lane_check_agrees(fixture: Path, max_lanes: int = 4096, bdd_budget: int = 1 << 16) -> Counter:
    """Check lanes (all, or a sample) against the exact omega_L closure, for
    both literal choices; returns how the lanes were decided."""
    spec = load(fixture)
    project = project_of(fixture, spec)
    rng = np.random.default_rng(0)
    methods: Counter = Counter()
    modes_all = ("split", "omega", "positive")
    for (policy, overrides, rule, target, _), literals in itertools.product(_cases(fixture), modes_all):
        probs, _ = project.probabilities(policy)
        tt = exact_table(project, target, probs, "hull", overrides, literals, rule)
        fixed_events, split = literal_sets(project, target, literals, overrides)
        c = build_circuit(project, target, probs, overrides, constants="hull", max_initiators=rule, split=split)
        if c.constant() is not None:
            continue
        modes = event_modes(c.names, c.n_vars, fixed_events)
        checker = LaneChecker(c, polarity(c), bdd_budget, modes)
        n = len(tt.names)
        fixed = [i for i, p in enumerate(tt.probs) if p in (0.0, 1.0)]
        idx = np.arange(1 << n)
        for i in fixed:  # lanes are reachable only with p = 0 variables false and p = 1 variables true
            idx = idx[((idx >> i) & 1) == int(tt.probs[i])]
        for a, e in (tt.twins or {}).items():  # and with every complement atom opposite to its event
            idx = idx[((idx >> a) & 1) != ((idx >> e) & 1)]
        if len(idx) > max_lanes:
            idx = rng.choice(idx, size=max_lanes, replace=False)
        col = {name: i for i, name in enumerate(tt.names)}
        x = [0] + [
            int.from_bytes(np.packbits(((idx >> col[c.names[v]]) & 1).astype(bool), bitorder="little").tobytes(),
                           "little")
            for v in range(1, c.first_gate)
        ]
        got = checker.check(x, len(idx))
        assert got == tt.table[idx].tolist(), (fixture.name, target, overrides, rule, literals)
        methods.update(checker.methods)
    return methods


def test_lane_check_matches_closure(fixture):
    _lane_check_agrees(fixture)


# --- brute-force omega_L on random circuits ------------------------------------


def _random_project(rng: random.Random, n_events: int, n_gates: int) -> Project:
    """A random FTP-like project: gates over earlier gates and events, random
    complements (also on events), k-of-n gates, I tags and p in {0, 1}."""
    loc = Location("random.ftp", 1)
    events = [f"E{i}" for i in range(n_events)]
    gates: dict[str, Gate] = {}
    for g in range(n_gates):
        pool = events + list(gates)
        kids = rng.sample(pool, rng.randint(1, min(4, len(pool))))
        kind = rng.choice((OR, AND, ATLEAST))
        k = rng.randint(1, len(kids)) if kind == ATLEAST else 0
        gates[f"G{g}"] = Gate(f"G{g}", kind, k, [(c, rng.random() < 0.3) for c in kids], loc)
    imports = [Import(e, rng.choice((0.0, 1.0)) if rng.random() < 0.15 else round(rng.uniform(0.05, 0.95), 2),
                      rng.random() < 0.3, loc) for e in events]
    return Project(("random.ftp",), gates, imports, [f"G{n_gates - 1}"])


def _eval_project(project: Project, name: str, x: dict, memo: dict) -> int:
    """Literal Boolean value of ``name`` (independent of benchmarks.reference)."""
    if name in x:
        return x[name]
    if name not in memo:
        g = project.gates[name]
        vals = [_eval_project(project, c, x, memo) ^ neg for c, neg in g.children]
        k = len(vals) if g.kind == AND else 1 if g.kind == OR else g.k
        memo[name] = int(sum(vals) >= k)
    return memo[name]


def _allowed(mode: str, xv: int) -> tuple[int, ...]:
    return {ONLY_POS: (0, xv), BOTH: (xv,), ONLY_NEG: (xv, 1), NEITHER: (0, 1)}[mode]


def _brute_omega(table: dict, n: int, modes: list[str]) -> dict:
    """omega_L g by its definition: exists y with y_v allowed by x_v for every v."""
    return {x: int(any(table[y] for y in itertools.product(*(_allowed(modes[i], x[i]) for i in range(n)))))
            for x in table}


def _brute_prime_l_terms(table: dict, worlds: list, index: dict, lits: list) -> set:
    """Prime implicants of ``table`` among the terms over ``lits`` ((event, value)
    pairs): the minimal cut sets of an omega_L table (Theorem 2(a))."""
    def implies(term) -> bool:
        return all(table[x] for x in worlds if all(x[index[e]] == b for e, b in term))

    terms = [t for r in range(len(index) + 1) for t in itertools.combinations(lits, r)
             if len({e for e, _ in t}) == len(t) and implies(t)]
    return {frozenset(t) for t in terms if not any(set(u) < set(t) for u in terms)}


def _brute_cone(project: Project, target: str) -> tuple[list[str], set[str]]:
    seen, stack, events, negative = set(), [target], [], set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        for c, neg in project.gates[name].children:
            if c in project.gates:
                stack.append(c)
            else:
                events.append(c)
                if neg:
                    negative.add(c)
    return sorted(set(events)), negative


def _eval_split(project: Project, name: str, atoms: dict, delta: set, memo: dict) -> int:
    """Value of ``name`` over atoms: a direct complement of an event of ``delta``
    reads the atom ``-NAME``; every other edge is Boolean."""
    if name not in project.gates:
        return atoms[name]
    if name not in memo:
        g = project.gates[name]
        vals = [atoms["-" + c] if neg and c in delta else _eval_split(project, c, atoms, delta, memo) ^ neg
                for c, neg in g.children]
        k = len(vals) if g.kind == AND else 1 if g.kind == OR else g.k
        memo[name] = int(sum(vals) >= k)
    return memo[name]


def _subsets(mask: int):
    sub = mask
    while True:
        yield sub
        if sub == 0:
            return
        sub = (sub - 1) & mask


@pytest.mark.parametrize("seed", range(120))
def test_random_projects_match_brute_force(seed):
    """End to end on random projects, complements inside and outside negated
    gates, all three literal modes, with and without the initiator rule: the
    exact value, the minimal cut sets, the per-lane check (every reachable
    world) and, for every sixth seed, the estimator equal brute force. split:
    the plain hull over atoms on the coupled valuation; omega: omega_L; positive:
    the plain hull."""
    rng = random.Random(seed)
    project = _random_project(rng, rng.randint(2, 7), rng.randint(1, 7))
    target = project.process[0]
    probs, _ = project.probabilities()
    events, negative = _brute_cone(project, target)
    tagged = project.initiators() & set(events)
    n = len(events)
    index = {e: i for i, e in enumerate(events)}
    worlds = list(itertools.product((0, 1), repeat=n))
    live = [x for x in worlds if all(x[i] == probs[e] for i, e in enumerate(events) if probs[e] in (0.0, 1.0))]
    g = {x: _eval_project(project, target, dict(zip(events, x, strict=True)), {}) for x in worlds}
    weight = {x: math.prod(probs[e] if b else 1 - probs[e] for e, b in zip(events, x, strict=True)) for x in worlds}
    atoms = events + ["-" + e for e in sorted(negative)]
    bit = {a: 1 << i for i, a in enumerate(atoms)}
    g_split = [_eval_split(project, target, {a: (m >> i) & 1 for i, a in enumerate(atoms)}, negative, {})
               for m in range(1 << len(atoms))]

    def kappa(x) -> int:
        return sum(bit[e] for e, b in zip(events, x, strict=True) if b) + sum(
            bit["-" + e] for e in negative if not x[index[e]])

    for literals, rule in itertools.product(("split", "omega", "positive"), (None, 0, 1)):
        expect = {"omega": (sorted(negative), []), "split": ([], sorted(negative)), "positive": ([], [])}
        assert literal_sets(project, target, literals) == expect[literals]
        ok_rule = [int(rule is None or sum(1 for e in tagged if m & bit[e]) <= rule) for m in range(1 << len(atoms))]
        if literals == "split":
            gm = [g_split[m] & ok_rule[m] for m in range(len(g_split))]
            want = {x: int(any(gm[sub] for sub in _subsets(kappa(x)))) for x in worlds}
            sets = [m for m in range(len(gm)) if gm[m]]
            prime = {frozenset(a for a in atoms if m & bit[a]) for m in sets
                     if not any(gm[m ^ bit[a]] for a in atoms if m & bit[a])
                     and not any(m & bit[a] and m & bit[a[1:]] for a in atoms if a.startswith("-"))}
            # gm is up-closed over atoms only after the hull; minimal sets of gm are mcs of up(gm).
            prime = {c for c in prime if not any(set(d) < set(c) for d in prime)}
        else:
            neg = negative if literals == "omega" else set()
            modes = [BOTH if e in neg else ONLY_POS for e in events]
            gr = {x: g[x] & int(rule is None or sum(x[i] for i, e in enumerate(events) if e in tagged) <= rule)
                  for x in worlds}
            want = _brute_omega(gr, n, modes)
            lits = [(e, 1) for e in events] + [(e, 0) for e in events if e in neg]
            prime = {frozenset(("-" if b == 0 else "") + e for e, b in t)
                     for t in _brute_prime_l_terms(want, worlds, index, lits)}
        p_want = math.fsum(weight[x] for x in worlds if want[x])
        got = exact_probability(project, target, probs, "hull", None, literals, rule)["probability"]
        assert got == pytest.approx(p_want, rel=1e-12, abs=1e-15), (literals, rule)
        cut_sets = exact_cut_sets(project, target, probs, None, literals, rule)
        assert {frozenset(cs["literals"]) for cs in cut_sets} == prime, (literals, rule)
        # Per-lane check on the folded estimator circuit, every reachable world.
        fixed, split = literal_sets(project, target, literals)
        c = build_circuit(project, target, probs, constants="hull", max_initiators=rule, split=split)
        if c.constant() is not None:
            assert all(want[x] == c.constant() for x in live), (literals, rule)
        else:
            checker = LaneChecker(c, polarity(c), modes=event_modes(c.names, c.n_vars, fixed))

            def value(name: str, x) -> int:
                return 1 - x[index[name[1:]]] if name.startswith("-") else x[index[name]]

            x_bits = [0] + [sum(value(c.names[v], x) << j for j, x in enumerate(live)) for v in range(1, c.first_gate)]
            assert checker.check(x_bits, len(live)) == [bool(want[x]) for x in live], (literals, rule)
        if seed % 6 == 0 and rule in (None, 1):
            r = estimate_hull(project, target, probs, trials=1 << 13, seed=seed, literals=literals,
                              max_initiators=rule)
            p_live = min(1.0, max(0.0, math.fsum(weight[x] for x in live if want[x])))
            if min(p_live, 1.0 - p_live) < 1e-12:
                assert r["estimate"] == pytest.approx(p_live, abs=1e-12), (literals, rule)
            else:
                assert abs(r["estimate"] - p_live) <= Z_MAX * _null_se(p_live, r["trials"]), (literals, rule)


def _eval_circuit(c, x: tuple) -> int:
    val = [0] + list(x) + [0] * len(c.kinds)
    for g, kids in enumerate(c.kids):
        ins = [val[lit >> 1] ^ (lit & 1) for lit in kids]
        k = len(ins) if c.kinds[g] == AND else 1 if c.kinds[g] == OR else c.ks[g]
        val[c.first_gate + g] = int(sum(ins) >= k)
    return val[c.root >> 1] ^ (c.root & 1)


@pytest.mark.parametrize("seed", range(60))
def test_lane_check_and_bounds_any_literals(seed):
    """RFC §4.4, §6.2 for any L (every event independently: only x_v, both,
    only not x_v, neither): on every world, g <= g_coh^L, the per-lane check
    and the exact closure equal brute-force omega_L g, and omega_L g <= g_coh^L."""
    rng = random.Random(1000 + seed)
    project = _random_project(rng, rng.randint(1, 7), rng.randint(1, 7))
    target = project.process[0]
    probs, _ = project.probabilities()
    c = build_circuit(project, target, probs, constants="none", fold=rng.random() < 0.5)
    if c.constant() is not None or c.n_vars == 0:
        return
    n = c.n_vars
    modes = [None] + [rng.choice((ONLY_POS, ONLY_POS, BOTH, ONLY_NEG, NEITHER)) for _ in range(n)]
    worlds = list(itertools.product((0, 1), repeat=n))
    g = {x: _eval_circuit(c, x) for x in worlds}
    want = _brute_omega(g, n, modes[1:])
    # exact closure (bit i of the table index is variable i + 1)
    tt = truth_table(c)
    up = closure(tt.table, modes[1:])
    assert [bool(up[sum(b << i for i, b in enumerate(x))]) for x in worlds] == [bool(want[x]) for x in worlds]
    # per-lane check
    x_bits = [0] + [sum(x[v - 1] << j for j, x in enumerate(worlds)) for v in range(1, n + 1)]
    checker = LaneChecker(c, polarity(c), modes=modes)
    assert checker.check(x_bits, len(worlds)) == [bool(want[x]) for x in worlds], modes
    # bounds: evaluate g and g_coh^L bit-parallel over all worlds
    mask = (1 << len(worlds)) - 1
    root, coh = evaluate(c, plan(c), x_bits.__getitem__, mask, 0, coherent=coherent_specs(c, modes))
    for j, x in enumerate(worlds):
        assert (root >> j) & 1 == g[x] <= want[x] <= (coh >> j) & 1, (modes, x)


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


_PERCENT_TILDE = """T + G~1 B-%IE
G~1 * %IE A
B-%IE * %IE -%IE2
ENDTREE
PROCESS T
IMPORT
0.1 %IE I
0.2 %IE2 I
0.5 A
"""


def _all_probabilities(project, target: str) -> list[float]:
    probs, _ = project.probabilities()
    out = []
    for rule in (None, 1):
        out.append(exact_probability(project, target, probs, "boolean", max_initiators=rule)["probability"])
        for literals in ("split", "omega", "positive"):
            out.append(exact_probability(project, target, probs, "hull", None, literals, rule)["probability"])
    return out


def test_parser_percent_and_tilde_names(tmp_path):
    """#240: '%' and '~' are name characters; the values are those of the model with them renamed away."""
    p = parse_project([_write(tmp_path, _PERCENT_TILDE, "m.ftp", "\r\n")])
    assert p.diagnostics == []
    assert p.gates["B-%IE"].children == [("%IE", False), ("%IE2", True)]
    assert p.initiators() == {"%IE", "%IE2"}
    probs, _ = p.probabilities()
    assert exact_probability(p, "T", probs, "boolean")["probability"] == pytest.approx(0.09, rel=1e-12)
    renamed = parse_project([_write(tmp_path, _PERCENT_TILDE.replace("%", "P_").replace("~", "_"), "r.ftp")])
    assert _all_probabilities(p, "T") == pytest.approx(_all_probabilities(renamed, "T"), rel=1e-12)


def test_parser_dialect_of_exported_files(tmp_path):
    """Lower-case keywords and flag, other-case names, ';' comments, layout, value forms, -1 and MAXSIZE."""
    p = parse_project([_write(tmp_path, """\xef\xbb\xbf; comment line
top + a G1 ; tail
  g2
G1 * B $C
g2 2 a B
.D
endtree now
process TOP
import
  .5 A
  5.E-1 b i
  1.0D-01 $C
  -1 .D
limit 1.E-12 x
maxsize 2
""")])
    warnings = sorted((d.code, d.name) for d in p.diagnostics if d.severity == "warning")
    assert [d for d in p.diagnostics if d.severity == "error"] == []
    assert warnings == [("ignored-tokens", None), ("ignored-tokens", None), ("minus-one-value", ".D"),
                        ("name-case", "A"), ("name-case", "b"), ("name-case", "top")]
    # A gate is named as its gate line spells it, an event as its first import record does.
    assert list(p.gates) == ["top", "G1", "g2"]
    assert p.gates["top"].children == [("A", False), ("G1", False), ("g2", False)]
    assert p.gates["G1"].children == [("b", False), ("$C", False)]
    assert p.gates["g2"].k == 2 and p.gates["g2"].children == [("A", False), ("b", False), (".D", False)]
    assert p.process == ["top"] and p.default_targets() == ["top"]
    assert [(i.name, i.value, i.initiator) for i in p.imports] == [
        ("A", 0.5, False), ("b", 0.5, True), ("$C", 0.1, False), (".D", 1.0, False)]
    assert p.limit == 1e-12 and p.maxsize == 2


@pytest.mark.parametrize(
    "text,codes",
    [
        ("G + A\nENDTREE\nPROCESS G\nIMPORT\n-0.5 A\n", {"invalid-probability"}),
        ("G + A\nENDTREE\nPROCESS G\nIMPORT\n-1.000001 A\n", {"invalid-probability"}),
        ("G + A\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\nMAXSIZE 1.5\n", {"invalid-maxsize"}),
        ("G + A\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\nLIMIT\n", {"invalid-limit"}),
        ("G + A B\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n5E B\n", {"invalid-line", "undefined-reference"}),
        ("G + A\ng * A\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n", {"duplicate-gate"}),
        ("G + A\n  + B\nENDTREE\nPROCESS G\nIMPORT\n0.1 A\n0.2 B\n", {"invalid-line"}),
        ("G + A\rENDTREE\rPROCESS G\rIMPORT\r0.1 A\r", {"missing-endtree", "undefined-reference"}),
    ],
)
def test_parser_dialect_errors(tmp_path, text, codes):
    p = parse_project([_write(tmp_path, text)])
    assert {d.code for d in p.diagnostics if d.severity == "error"} == codes


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


def test_cli_literals_rule_and_cut_sets():
    fx = str(FIXTURES / "process_multi_target")
    split = json.loads(_cli("exact", "--semantics", "hull", "--target", "T2", "--cut-sets", "--json", fx).stdout)
    r = split["results"][0]
    assert split["literals"] == "split" and r["complemented_events"] == ["B"]
    assert r["probability"] == pytest.approx(0.464, rel=1e-12)
    assert [cs["literals"] for cs in r["cut_sets"]] == [["-B", "C"], ["-B", "D"]]
    plain = json.loads(_cli("exact", "--semantics", "hull", "--literals", "positive", "--target", "T2", "--json",
                            fx).stdout)
    assert plain["results"][0]["probability"] == pytest.approx(0.58, rel=1e-12)
    assert _cli("exact", "--semantics", "boolean", "--cut-sets", fx).returncode == 2
    fx = str(FIXTURES / "initiator_rule")
    out = _cli("exact", "--semantics", "hull", "--max-initiators-per-cutset", "1", "--json", fx)
    doc = json.loads(out.stdout)
    assert doc["max_initiators_per_cutset"] == 1
    assert doc["results"][0]["probability"] == pytest.approx(0.424, rel=1e-12)
    out = _cli("estimate", "--semantics", "hull", "--max-initiators-per-cutset", "1", "--trials", "65536",
               "--json", fx)
    r = json.loads(out.stdout)["results"][0]
    assert abs(r["estimate"] - 0.424) <= Z_MAX * _null_se(0.424, r["trials"])
    assert _cli("exact", "--max-initiators-per-cutset", "-1", fx).returncode == 2


def test_cli_complement_under_negation_warning():
    """One warning per directly complemented event under a negated gate, in the
    hull modes that keep complements; none on the other fixtures."""
    fx = str(FIXTURES / "complement_under_negation")
    want = {"split": 0.658, "omega": 1 - (1 - 0.296) * 0.7, "positive": 1 - 0.63 * 0.7}
    for literals, value in want.items():
        out = _cli("exact", "--semantics", "hull", "--literals", literals, "--target", "TOP", "--json", fx)
        doc = json.loads(out.stdout)
        warned = [d["name"] for d in doc["diagnostics"] if d["code"] == "complement-under-negation"]
        assert warned == ([] if literals == "positive" else ["B", "F"]), literals
        assert doc["results"][0]["probability"] == pytest.approx(value, rel=1e-12), literals
    assert "warning: complement-under-negation" in _cli("exact", "--semantics", "hull", fx).stderr
    for name in FIXTURE_NAMES:
        if name != "complement_under_negation":
            assert "complement-under-negation" not in _cli("exact", "--semantics", "hull", str(FIXTURES / name)).stderr


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


def _policy(var: dict, card: dict) -> str:
    """Duplicate-import policy; cards may annotate it ("last (...)")."""
    raw = var.get("policy", card.get("policy", {})).get("duplicate_import", "error")
    word = re.match(r"[a-z]+", raw)
    return word.group(0) if word and word.group(0) in POLICIES else raw


# Variant keys that change the target's logic other than the structured
# initiator rule: such a variant is skipped, never run without its rule.
_UNSUPPORTED_RULE_KEYS = ("extra_delete_term",)


def _initiator_rule(var: dict, label: str) -> int | None:
    """The variant's ``max_initiators_per_cutset``; a variant that states a rule
    only in another form (``extra_delete_term``) is skipped."""
    if "max_initiators_per_cutset" in var:
        return int(var["max_initiators_per_cutset"])
    if any(k in var for k in _UNSUPPORTED_RULE_KEYS):
        pytest.skip(f"{label}: rule not given as max_initiators_per_cutset")
    return None


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
    policy = _policy(var, card)
    rule = _initiator_rule(var, label)
    probs, _ = project.probabilities(policy)
    target = project.default_targets()[0]
    if semantics == "boolean":
        r = estimate_boolean(project, target, probs, overrides, trials=PRIVATE_TRIALS, seed=1, max_initiators=rule)
        extra = f"trials/s={r.get('trials_per_second') or 0:.3g}"
    else:
        r = estimate_hull(project, target, probs, overrides, trials=PRIVATE_HULL_TRIALS, seed=1,
                          max_checks=PRIVATE_HULL_CHECKS, max_initiators=rule)
        extra = (f"complemented={len(r['complemented_events'])} rule={rule} "
                 f"checked={r.get('lanes', {}).get('checked', 0)} "
                 f"lanes/s={r.get('lanes_per_second') or 0:.0f}")
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
