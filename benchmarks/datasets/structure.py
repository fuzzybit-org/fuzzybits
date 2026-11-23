"""Structure counts for a v2 model card (#158, #184).

The card's ``structure`` block is computed, never typed by hand:

* OpenPSA/MEF: :func:`mef_structure`. Gate names are scoped per
  ``define-fault-tree`` (a model with several event trees shares fault trees,
  so the same gate name appears more than once), and nested anonymous formulas
  count as gates of their own.
* CAFTA FTP projects: :func:`ftp_structure`, which reuses the reference parser
  and its counter (``benchmarks.reference.structure``) — the same code the
  private cards are checked against.
* The lowered circuit: :func:`inspect_pcfb`, ``fzb inspect --json`` on the
  ``.pcfb`` built with the card's compile settings.

Command line::

    python -m benchmarks.datasets.structure input/Aralia/das9201.xml --query r1
    python -m benchmarks.datasets.structure --ftp input/ftp/conformance/multi_file_project
    python -m benchmarks.datasets.structure input/Aralia/das9201.xml --pcfb /tmp/x.pcfb
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

# MEF connective -> the card's gate type. Anything else counts as ``other``.
_GATE_TYPES = {
    "and": "AND",
    "or": "OR",
    "atleast": "VOT",
    "xor": "XOR",
    "not": "NOT",
    "nand": "NOT",
    "nor": "NOT",
}
_NEGATING = frozenset({"xor", "not", "nand", "nor", "iff"})
_CONNECTIVES = frozenset(
    {"and", "or", "atleast", "xor", "not", "nand", "nor", "null", "iff", "imply", "cardinality"}
)
_REFERENCE_TAGS = frozenset({"gate", "basic-event", "house-event", "event"})
_IGNORED = frozenset({"label", "attributes", "parameter", "float", "integer", "bool"})
GATE_TYPES = ("AND", "OR", "VOT", "XOR", "NOT")


class StructureError(ValueError):
    """The input is not a readable fault-tree model."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@dataclass
class _Tree:
    """One ``define-fault-tree``: gate name -> (connective, child names)."""

    gates: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    anonymous: int = 0


def _parse_formula(element: ElementTree.Element, tree: _Tree, owner: str) -> list[str]:
    """Children of a formula element, registering nested formulas as gates."""
    children: list[str] = []
    for child in element:
        tag = _local(child.tag)
        if tag in _IGNORED:
            continue
        if tag in _REFERENCE_TAGS:
            children.append(child.attrib.get("name", ""))
            continue
        if tag in _CONNECTIVES:
            tree.anonymous += 1
            name = f"\0{owner}#{tree.anonymous}"
            tree.gates[name] = (tag, _parse_formula(child, tree, name))
            children.append(name)
            continue
        raise StructureError(f"unsupported formula <{tag}> under {owner!r}")
    return children


def _parse_trees(root: ElementTree.Element) -> tuple[list[_Tree], dict[str, int]]:
    trees: list[_Tree] = []
    counts: Counter[str] = Counter()
    # ``define-sequence`` sits inside ``define-event-tree``, not at the root.
    for element in root.iter():
        tag = _local(element.tag)
        if tag == "define-sequence":
            counts["event_tree_sequences"] += 1
        elif tag == "define-ccf-group":
            counts["ccf_groups"] += 1
    for element in root:
        tag = _local(element.tag)
        if tag == "define-basic-event":
            counts["defined_basic_events"] += 1
        elif tag == "define-fault-tree":
            tree = _Tree()
            for block in element:
                if _local(block.tag) != "define-gate":
                    continue
                name = block.attrib.get("name", "")
                connective, children = "null", []
                for child in block:
                    if _local(child.tag) in _IGNORED:
                        continue
                    connective = _local(child.tag)
                    children = _parse_formula(child, tree, name)
                    break
                # A repeated name in one tree is the exporter's bug, not ours.
                tree.gates.setdefault(name, (connective, children))
            trees.append(tree)
    return trees, dict(counts)


def _cone(tree: _Tree, top: str) -> tuple[int, set[str], set[str]]:
    """``(depth, gates, events)`` of ``top``'s cone; depth counts gates on its
    longest root-to-event path (an event alone is depth 0)."""
    memo: dict[str, int] = {}
    gates: set[str] = set()
    events: set[str] = set()
    stack: list[tuple[str, int]] = [(top, 0)]
    on_path = {top}
    while stack:
        name, index = stack[-1]
        children = tree.gates[name][1]
        gates.add(name)
        if index < len(children):
            stack[-1] = (name, index + 1)
            child = children[index]
            if child in tree.gates:
                if child in memo:
                    continue
                if child in on_path:
                    raise StructureError(f"cycle through gate {child!r}")
                on_path.add(child)
                stack.append((child, 0))
                continue
            events.add(child)
            continue
        memo[name] = 1 + max((memo[c] for c in children if c in memo), default=0)
        on_path.discard(name)
        stack.pop()
    return memo[top], gates, events


def _roots(tree: _Tree) -> list[str]:
    children = {c for _, kids in tree.gates.values() for c in kids}
    return [name for name in tree.gates if name not in children]


def _tree_of(trees: list[_Tree], name: str) -> _Tree | None:
    for tree in trees:
        if name in tree.gates:
            return tree
    return None


def _empty() -> dict[str, Any]:
    return {"gates": 0, "basic_events": 0, "depth": 0}


def mef_structure(path: str | Path, *, query: str | None = None) -> dict[str, Any]:
    """Model-level structure counts of an OpenPSA/MEF file.

    ``depth`` is the deepest root-to-event path over every fault tree, and
    ``query_cone`` the same for the compiled query. ``coherent`` is False when
    the model contains a negating gate (NOT, NAND, NOR, XOR, IFF). Gate and
    edge counts cover every ``define-fault-tree`` in the file, so they describe
    the whole model even when only one top event is compiled.
    """
    try:
        root = ElementTree.parse(str(path)).getroot()
    except ElementTree.ParseError as exc:
        raise StructureError(f"XML parse error: {exc}") from exc
    trees, counts = _parse_trees(root)
    if not trees:
        raise StructureError(f"{Path(path).name}: no define-fault-tree")

    types: Counter[str] = Counter()
    edges = 0
    referenced: set[str] = set()
    deepest = 0
    for tree in trees:
        for connective, children in tree.gates.values():
            types[_GATE_TYPES.get(connective, "other")] += 1
            edges += len(children)
            referenced.update(child for child in children if child not in tree.gates)
        for name in _roots(tree):
            deepest = max(deepest, _cone(tree, name)[0])

    coherent = not any(
        connective in _NEGATING for tree in trees for connective, _ in tree.gates.values()
    )
    top = query
    if top is None:
        top = next((name for name in _roots(trees[0])), None)
    cone = _empty()
    if top is not None and (tree := _tree_of(trees, top)) is not None:
        depth, gates, events = _cone(tree, top)
        cone = {"gates": len(gates), "basic_events": len(events), "depth": depth}
    return {
        "basic_events": counts.get("defined_basic_events") or len(referenced),
        "gates": sum(types.values()),
        "gates_by_type": {
            **{name: types.get(name, 0) for name in GATE_TYPES},
            "other": types.get("other", 0),
        },
        "edges": edges,
        "depth": deepest,
        "event_tree_sequences": counts.get("event_tree_sequences", 0),
        "ccf_groups": counts.get("ccf_groups", 0),
        "coherent": coherent,
        "fault_trees": len(trees),
        "query_cone": cone,
    }


def ftp_structure(paths: list[str | Path], *, target: str | None = None) -> dict[str, Any]:
    """Structure counts of a CAFTA FTP project, as the reference parser reads it.

    A complemented reference is a delete term, not a negating gate, so
    ``gates_by_type.NOT`` stays 0 and ``coherent`` is False whenever the model
    has one.
    """
    from ..reference.ftp import parse_project
    from ..reference.structure import structure_counts

    project = parse_project([Path(item) for item in paths])
    counts = structure_counts(project)
    kinds = counts["gates_by_kind"]
    goals = [target] if target is not None else project.default_targets()
    structure: dict[str, Any] = {
        "basic_events": counts["basic_events"],
        "gates": counts["gates"],
        "gates_by_type": {
            "AND": kinds["and"],
            "OR": kinds["or"],
            "VOT": kinds["atleast"],
            "XOR": 0,
            "NOT": 0,
            "other": 0,
        },
        "edges": counts["edges"],
        "depth": counts["depth"],
        "event_tree_sequences": 0,
        "ccf_groups": 0,
        "coherent": counts["complemented_edges"] == 0,
    }
    if goals:
        structure["query_cone"] = _ftp_cone(project, goals[0])
    return structure


def _ftp_cone(project: Any, target: str) -> dict[str, Any]:
    gates = project.gates
    seen: set[str] = set()
    events: set[str] = set()
    stack = [target]
    while stack:
        name = stack.pop()
        if name in gates:
            if name in seen:
                continue
            seen.add(name)
            stack.extend(child for child, _ in gates[name].children)
        else:
            events.add(name)
    return {
        "gates": len(seen),
        "basic_events": len(events),
        "depth": _ftp_depth(gates, target) if seen else 0,
    }


def _ftp_depth(gates: dict[str, Any], top: str) -> int:
    memo: dict[str, int] = {}
    stack: list[tuple[str, int]] = [(top, 0)]
    on_path = {top}
    while stack:
        name, index = stack[-1]
        children = gates[name].children if name in gates else []
        if index < len(children):
            stack[-1] = (name, index + 1)
            child = children[index][0]
            if child in gates:
                if child in memo:
                    continue
                if child in on_path:
                    raise StructureError(f"cycle through gate {child!r}")
                on_path.add(child)
                stack.append((child, 0))
                continue
            continue
        memo[name] = 1 + max((memo[c] for c, _ in children if c in memo), default=0)
        on_path.discard(name)
        stack.pop()
    return memo.get(top, 0)


def inspect_pcfb(pcfb: str | Path, *, fzb: str = "fzb") -> dict[str, Any]:
    """Circuit size of a ``.pcfb``, from ``fzb inspect --json``."""
    proc = subprocess.run(
        [fzb, "inspect", str(pcfb), "--json"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise StructureError(f"fzb inspect failed: {proc.stderr.strip()[-400:]}")
    document = json.loads(proc.stdout)
    return {
        "tensors": document.get("tensors"),
        "operators": document.get("operators"),
        "samples": document.get("samples"),
        "opcodes": document.get("opcodes"),
    }


def card_structure(
    path: str | Path,
    *,
    query: str | None = None,
    pcfb: str | Path | None = None,
    fzb: str = "fzb",
) -> dict[str, Any]:
    """The card's ``structure`` block for one MEF model, plus compiled counts."""
    structure = mef_structure(path, query=query)
    if pcfb is not None:
        structure["compiled"] = inspect_pcfb(pcfb, fzb=fzb)
    return structure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.datasets.structure", description=__doc__
    )
    parser.add_argument("inputs", nargs="*", help="MEF XML files, or FTP files/directories with --ftp")
    parser.add_argument("--ftp", action="store_true", help="inputs are CAFTA FTP files or directories")
    parser.add_argument("--query", default=None, help="MEF gate whose cone is reported as query_cone")
    parser.add_argument("--target", default=None, help="FTP target for query_cone (default: the project's)")
    parser.add_argument("--pcfb", default=None, help="compiled .pcfb to report as structure.compiled")
    parser.add_argument("--fzb", default="fzb", help="fzb executable for --pcfb")
    args = parser.parse_args(argv)
    if not args.inputs:
        parser.error("pass at least one input")
    paths = [Path(item) for item in args.inputs]
    if args.ftp:
        files: list[Path] = []
        for path in paths:
            files.extend(sorted(path.glob("*.ftp")) if path.is_dir() else [path])
        payload = ftp_structure(files, target=args.target)
    else:
        payload = card_structure(paths[0], query=args.query, pcfb=args.pcfb, fzb=args.fzb)
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
