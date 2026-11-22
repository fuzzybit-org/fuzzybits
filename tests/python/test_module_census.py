"""Tests for scripts/analysis/module_census.py (#213)."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

# pylint: disable-next=wrong-import-position
import module_census as mc  # noqa: E402

ARALIA = ROOT / "input" / "Aralia"


def _or(*xs: str) -> str:
    return "<or>" + "".join(xs) + "</or>"


def _and(*xs: str) -> str:
    return "<and>" + "".join(xs) + "</and>"


def _atleast(k: int, *xs: str) -> str:
    return f'<atleast min="{k}">' + "".join(xs) + "</atleast>"


def _g(name: str) -> str:
    return f'<gate name="{name}"/>'


def _b(name: str) -> str:
    return f'<basic-event name="{name}"/>'


def _mef(gates: list[tuple[str, str]], encoding: str = "utf-8") -> bytes:
    body = "".join(
        f'<define-gate name="{n}">{f}</define-gate>' for n, f in gates)
    text = (f'<?xml version="1.0" encoding="{encoding}"?>'
            f'<opsa-mef><define-fault-tree name="t">{body}'
            "</define-fault-tree></opsa-mef>")
    return text.encode(encoding)


def _analyse(gates: list[tuple[str, str]], **kwargs) -> mc.CensusDetails:
    return mc.analyse_graph(mc.parse_mef(io.BytesIO(_mef(gates))), **kwargs)


def _module(details: mc.CensusDetails, name: str) -> int:
    return details.normalized.name.index(name)


def test_shared_event_breaks_modularity():
    """A basic event under two subtrees leaves only the top as a module."""
    d = _analyse([
        ("top", _and(_g("G1"), _g("G2"))),
        ("G1", _or(_b("a"), _b("b"))),
        ("G2", _or(_b("b"), _b("c"))),
    ])
    assert d.module_names() == {"top"}
    r = d.result
    assert (r.gates, r.edges, r.modules, r.nontrivial_modules) == (3, 6, 1, 1)
    assert r.max_local_vars == 3 and r.s_inf == 1.0


def test_disjoint_subtrees_are_modules():
    """Disjoint subtrees are modules; W = 6, CP = 4, S_inf = 1.5."""
    d = _analyse([
        ("top", _and(_g("G1"), _g("G2"))),
        ("G1", _or(_b("a"), _b("b"))),
        ("G2", _or(_b("c"), _b("d"))),
    ])
    assert d.module_names() == {"top", "G1", "G2"}
    r = d.result
    assert (r.total_local_work, r.critical_path) == (6, 4)
    assert (r.tree_depth, r.max_width, r.nontrivial_modules) == (2, 2, 0)
    assert r.s_inf == pytest.approx(1.5)
    assert r.speedups == {8: pytest.approx(1.5), 48: pytest.approx(1.5)}
    assert mc.greedy_list_schedule(d.tree, 1) == 6


def test_shared_module_has_one_parent_module():
    """G3 has two parent gates, both in top's local part: one tree parent."""
    d = _analyse([
        ("top", _or(_g("G1"), _g("G2"))),
        ("G1", _and(_g("G3"), _b("a"))),
        ("G2", _and(_g("G3"), _b("b"))),
        ("G3", _or(_b("c"), _b("d"))),
    ])
    assert d.module_names() == {"top", "G3"}
    top, g3 = _module(d, "top"), _module(d, "G3")
    tree = d.tree
    assert tree.gate_parents[g3] == 2 and tree.parent[g3] == top
    assert (tree.local_gates[top], tree.local_edges[top],
            tree.local_vars[top]) == (3, 6, 3)
    assert d.result.shared_modules == 1
    assert (d.result.total_local_work, d.result.critical_path) == (8, 8)


def test_internal_sharing_keeps_parent_a_module():
    """Sharing that stays inside G1 does not break G1's modularity."""
    d = _analyse([
        ("top", _or(_g("G1"), _b("x"))),
        ("G1", _and(_g("G2"), _g("G3"))),
        ("G2", _or(_b("a"), _b("b"))),
        ("G3", _or(_b("b"), _b("c"))),
    ])
    assert d.module_names() == {"top", "G1"}
    g1 = _module(d, "G1")
    assert (d.tree.local_gates[g1], d.tree.local_vars[g1]) == (3, 3)


def test_pass_through_removal():
    """Single-operand or, bare reference and atleast k=1 are forwarded."""
    d = _analyse([
        ("top", _and(_g("P"), _b("c"), _g("Q"))),
        ("P", _or(_g("G1"))),
        ("G1", _or(_b("a"), _b("b"))),
        ("Q", _g("R")),
        ("R", _atleast(1, _g("S"))),
        ("S", _and(_b("d"), _b("e"), _b("d"))),
    ])
    r = d.result
    assert (r.gates, r.edges, r.pass_through_removed) == (3, 7, 3)
    assert d.module_names() == {"top", "G1", "S"}


def test_nested_formulas_not_and_fold_not():
    """Nested formulas become gates; fold_not forwards NOT gates."""
    gates = [("top",
              _atleast(2, _b("a"), _or(_b("b"), _b("c")),
                       "<not>" + _b("d") + "</not>"))]
    r = _analyse(gates).result
    assert (r.gates, r.anonymous_gates) == (3, 2)
    assert (r.atleast_gates, r.not_gates, r.modules) == (1, 1, 3)
    folded = _analyse(gates, fold_not=True).result
    assert (folded.gates, folded.modules) == (2, 2)


def test_multiple_roots_use_a_virtual_root():
    """Two roots sharing G: neither root is a module; G is."""
    d = _analyse([
        ("R1", _or(_b("a"), _g("G"))),
        ("R2", _and(_g("G"), _b("b"))),
        ("G", _and(_b("c"), _b("d"))),
    ])
    r = d.result
    assert r.roots == 2 and d.module_names() == {"G"}
    assert r.virtual_root_local_gates == 2 and r.gates == 3


def test_utf16_with_bom_parses():
    """UTF-16 input (with byte-order mark) gives the same graph as UTF-8."""
    gates = [("top", _or(_g("G1"), _b("x"))), ("G1", _and(_b("a"), _b("b")))]
    utf16 = mc.parse_mef(io.BytesIO(_mef(gates, encoding="utf-16")))
    utf8 = mc.parse_mef(io.BytesIO(_mef(gates)))
    assert utf16.children == utf8.children and utf16.name == utf8.name


def test_cycle_is_reported():
    """A cycle below the root raises MefError."""
    with pytest.raises(mc.MefError, match="cycle"):
        _analyse([
            ("R", _or(_g("top"), _b("z"))),
            ("top", _or(_g("G1"), _b("x"))),
            ("G1", _and(_g("top"), _b("y"))),
        ])


def test_deep_chain_needs_no_recursion():
    """A chain far deeper than the recursion limit: every gate a module."""
    depth = 30_000
    gates = [(f"G{i}", _or(_b(f"e{i}"), _g(f"G{i + 1}"))) for i in range(depth)]
    gates.append((f"G{depth}", _and(_b("x"), _b("y"))))
    assert depth > sys.getrecursionlimit()
    r = _analyse(gates).result
    assert (r.modules, r.tree_depth) == (depth + 1, depth + 1)
    assert r.nontrivial_modules == 0
    assert r.s_inf == 1.0 and r.speedups[48] == 1.0


def test_random_dags_match_definition():
    """Linear-time detection agrees with the definition on random DAGs."""
    stats = mc.self_check_random_dags(trials=600, seed=213)
    assert stats["trials"] >= 590
    assert stats["modules"] > stats["trials"]  # not only the roots
    assert stats["gates"] > 2 * stats["modules"]  # many non-module gates
    assert stats["multi_root"] > 0


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # (gates, modules, nontrivial modules, total local work, critical path)
        ("baobab1", (84, 1, 1, 219, 219)),
        ("das9205", (20, 17, 2, 72, 21)),
        ("edf9206", (360, 22, 16, 726, 576)),
    ],
)
def test_aralia_smoke(model, expected):
    """Census of public Aralia models matches the measured values (#213)."""
    path = ARALIA / f"{model}.xml"
    if not path.is_file():
        pytest.skip(f"{path} not in the source tree")
    r = mc.census_file(path)
    assert (r.gates, r.modules, r.nontrivial_modules, r.total_local_work,
            r.critical_path) == expected
    assert r.s_inf == pytest.approx(expected[3] / expected[4])
    assert r.speedups[8] <= min(8, r.s_inf) + 1e-9


def _write(path: Path, gates: list[tuple[str, str]]) -> Path:
    path.write_bytes(_mef(gates))
    return path


def test_cli_outputs_and_path_expansion(tmp_path, capsys):
    """Directory recursion, globs, --cores, CSV, JSON and markdown output."""
    sub = tmp_path / "models" / "deep"
    sub.mkdir(parents=True)
    _write(sub / "a.xml", [("top", _and(_g("G1"), _b("z"))),
                           ("G1", _or(_b("a"), _b("b")))])
    _write(tmp_path / "models" / "b.xml", [("top", _or(_b("a"), _b("b")))])
    out_csv, out_json = tmp_path / "out.csv", tmp_path / "out.json"
    status = mc.main([
        str(tmp_path / "models"), "--cores", "2,4", "--csv",
        str(out_csv), "--json",
        str(out_json), "--markdown"
    ])
    assert status == 0
    table = capsys.readouterr().out
    assert "| model |" in table and "S2" in table and "S4" in table
    with out_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["model"] for r in rows] == ["b", "a"]
    assert rows[1]["modules"] == "2" and "s_4" in rows[1]
    assert json.loads(out_json.read_text())[1]["modules"] == 2

    assert mc.main([str(tmp_path / "models" / "**" / "a.xml")]) == 0
    assert "| a |" in capsys.readouterr().out


def test_cli_parse_errors(tmp_path, capsys):
    """Parse errors name the file; --keep-going continues but still fails."""
    good = _write(tmp_path / "good.xml", [("top", _or(_b("a"), _b("b")))])
    bad = tmp_path / "bad.xml"
    bad.write_text("<opsa-mef><define-gate name='x'><or>")
    assert mc.main([str(bad), str(good)]) == 1
    captured = capsys.readouterr()
    assert "bad.xml" in captured.err and "XML parse error" in captured.err
    assert "| good |" not in captured.out

    assert mc.main(["--keep-going", str(bad), str(good)]) == 1
    captured = capsys.readouterr()
    assert "bad.xml" in captured.err and "| good |" in captured.out

    assert mc.main([str(tmp_path / "missing.xml")]) == 1
    assert "no such file" in capsys.readouterr().err
