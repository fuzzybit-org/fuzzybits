#!/usr/bin/env python3
"""Module census of OpenPSA MEF fault trees (#213, #21).

Parses MEF XML into a gate/event DAG, removes pass-through gates, finds the
modules with the linear-time algorithm of Dutuit and Rauzy (1996), builds the
module tree and reports per model how the work splits into modules and how much
a bottom-up, task-parallel collapse of the module tree could gain under an
edge-count cost proxy.

Library use::

    from module_census import census_file
    result = census_file("input/Aralia/das9205.xml")
    print(result.modules, result.s_inf, result.speedups[8])

Command line::

    python scripts/analysis/module_census.py input/Aralia
    python scripts/analysis/module_census.py --csv out.csv 'models/**/*.xml'

Standard library only. See scripts/analysis/README.md for the method and its
limits.
"""
# A single-file, stdlib-only tool is easier to vendor and run than a package.
# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import heapq
import io
import json
import math
import os
import random
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, TextIO
from xml.etree import ElementTree

__all__ = [
    "CensusDetails",
    "CensusError",
    "CensusResult",
    "FaultTreeGraph",
    "MefError",
    "ModuleDetection",
    "ModuleTree",
    "NormalizedGraph",
    "analyse_graph",
    "build_module_tree",
    "census_file",
    "census_graph",
    "critical_path",
    "definition_modules",
    "detect_modules",
    "expand_paths",
    "format_markdown",
    "greedy_list_schedule",
    "main",
    "normalize_pass_through",
    "parse_mef",
    "random_mef_dag",
    "result_columns",
    "self_check_random_dags",
    "speedup_bounds",
    "total_work",
    "write_csv",
    "write_json",
]

# Node kinds.
GATE, BASIC_EVENT, HOUSE_EVENT, CONSTANT = 0, 1, 2, 3

CONNECTIVES = frozenset({
    "and", "or", "atleast", "not", "xor", "nand", "nor", "null", "iff", "imply",
    "cardinality"
})
_REFERENCES = frozenset({"gate", "basic-event", "house-event", "event"})
_IGNORED = frozenset({"label", "attributes"})
_IDEMPOTENT = frozenset({"and", "or", "nand", "nor"})
_PASS_THROUGH = frozenset({"and", "or", "null"})
_NEGATIONS = frozenset({"not", "nand", "nor"})

DEFAULT_CORES = (8, 48)
VIRTUAL_ROOT = "<virtual-root>"


class MefError(ValueError):
    """Malformed or unsupported MEF input (also raised for cyclic models)."""


class CensusError(RuntimeError):
    """A census of one input path failed; the message names the path."""

    def __init__(self, path: str | os.PathLike, reason: object) -> None:
        """Record ``path`` and prefix the message with it."""
        super().__init__(f"{os.fspath(path)}: {reason}")
        self.path = os.fspath(path)


# ----------------------------------------------------------------------------
# Graph and parsing
# ----------------------------------------------------------------------------


@dataclass(slots=True)
class FaultTreeGraph:  # pylint: disable=too-many-instance-attributes
    """Fault-tree DAG as parsed from MEF, before normalisation.

    Nodes are integers. Gates come first (named ``define-gate`` gates and
    anonymous nested formulas, in document order), then leaves (basic events,
    house events, constants) in first-reference order. ``children[v]`` lists
    the operands of gate ``v`` (with repetitions) and is empty for leaves.
    ``roots`` are the gates without a parent gate.
    """

    kind: list[int] = field(default_factory=list)
    gate_type: list[str | None] = field(default_factory=list)
    min_number: list[int] = field(default_factory=list)
    name: list[str | None] = field(default_factory=list)
    children: list[Sequence[int]] = field(default_factory=list)
    roots: list[int] = field(default_factory=list)
    named_gates: int = 0
    anonymous_gates: int = 0
    defined_basic_events: int = 0
    warnings: list[str] = field(default_factory=list)

    def add_node(self, kind: int, gate_type: str | None, min_number: int,
                 name: str | None) -> int:
        """Append a node without operands and return its id."""
        node = len(self.kind)
        self.kind.append(kind)
        self.gate_type.append(gate_type)
        self.min_number.append(min_number)
        self.name.append(name)
        return node


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class _MefBuilder:
    """Collects gates with symbolic operand references; resolves at the end."""

    def __init__(self) -> None:
        self.graph = FaultTreeGraph()
        self.operands: list[list] = []
        self.gate_id: dict[str, int] = {}
        self.defined_be: set[str] = set()
        self.defined_he: set[str] = set()

    def _label(self, gate: int) -> str:
        name = self.graph.name[gate]
        return repr(name) if name else f"#{gate}"

    def _new_gate(self, name: str | None, connective: str, element) -> int:
        k = 0
        if element is not None and element.get("min") is not None:
            try:
                k = int(element.get("min"))
            except ValueError as exc:
                raise MefError(f"gate {name!r}: bad min="
                               f"{element.get('min')!r}") from exc
        self.operands.append([])
        return self.graph.add_node(GATE, connective, k, name)

    def _reference(self, element, gate: int):
        tag = _local(element.tag)
        if tag in _REFERENCES:
            name = element.get("name")
            if name is None:
                raise MefError(f"<{tag}> without a name in gate "
                               f"{self._label(gate)}")
            return (tag, name)
        if tag == "constant":
            return ("constant", element.get("value", ""))
        raise MefError(f"unsupported formula element <{tag}> in gate "
                       f"{self._label(gate)}")

    def _fill(self, gate: int, formula) -> None:
        """Set the operands of ``gate`` from the children of ``formula``.

        Iterative, so nesting depth is not bounded by the recursion limit.
        """
        stack = [(gate, formula)]
        while stack:
            owner, element = stack.pop()
            refs = []
            for child in element:
                tag = _local(child.tag)
                if tag in CONNECTIVES:
                    sub = self._new_gate(None, tag, child)
                    self.graph.anonymous_gates += 1
                    refs.append(sub)
                    stack.append((sub, child))
                elif tag not in _IGNORED:
                    refs.append(self._reference(child, owner))
            self.operands[owner] = refs

    def define_gate(self, element) -> None:
        """Add a ``define-gate`` element and its nested formulas."""
        name = element.get("name")
        if name is None:
            raise MefError("define-gate without a name")
        if name in self.gate_id:
            raise MefError(f"gate {name!r} is defined twice")
        formulas = [c for c in element if _local(c.tag) not in _IGNORED]
        if len(formulas) != 1:
            raise MefError(f"gate {name!r} has {len(formulas)} formulas, "
                           "expected 1")
        formula = formulas[0]
        connective = _local(formula.tag)
        if connective in CONNECTIVES:
            gate = self._new_gate(name, connective, formula)
            self._fill(gate, formula)
        else:
            gate = self._new_gate(name, "null", None)
            self.operands[gate] = [self._reference(formula, gate)]
        self.gate_id[name] = gate

    def _resolve(self, ref, leaf: Callable[[int, str], int]) -> int:
        if isinstance(ref, int):
            return ref
        tag, name = ref
        if tag in ("gate", "event") and name in self.gate_id:
            return self.gate_id[name]
        if tag == "gate":
            self.graph.warnings.append(f"undefined gate {name!r} treated as "
                                       "a basic event")
            return leaf(BASIC_EVENT, name)
        if tag == "house-event" or (tag == "event" and
                                    name in self.defined_he and
                                    name not in self.defined_be):
            return leaf(HOUSE_EVENT, name)
        if tag == "constant":
            return leaf(CONSTANT, name)
        return leaf(BASIC_EVENT, name)

    def finish(self) -> FaultTreeGraph:
        """Resolve references to node ids and find the parentless gates."""
        graph = self.graph
        num_gates = len(graph.kind)
        leaf_id: dict[tuple[int, str], int] = {}

        def leaf(kind: int, name: str) -> int:
            if (kind, name) not in leaf_id:
                leaf_id[(kind, name)] = graph.add_node(kind, None, 0, name)
            return leaf_id[(kind, name)]

        graph.children = [
            [self._resolve(ref, leaf) for ref in refs] for refs in self.operands
        ]
        self.operands = []
        graph.children.extend(() for _ in range(len(graph.kind) - num_gates))
        has_parent = bytearray(len(graph.kind))
        for ops in graph.children[:num_gates]:
            for c in ops:
                has_parent[c] = 1
        graph.roots = [g for g in range(num_gates) if not has_parent[g]]
        graph.named_gates = len(self.gate_id)
        graph.defined_basic_events = len(self.defined_be)
        graph.warnings = sorted(set(graph.warnings))
        return graph


def parse_mef(source: str | os.PathLike | BinaryIO) -> FaultTreeGraph:
    """Stream-parse an OpenPSA MEF file into a :class:`FaultTreeGraph`.

    ``source`` is a path or a binary file object. The encoding comes from the
    XML declaration or the byte-order mark, so UTF-16 files work. Every
    ``define-gate`` and every nested formula (``and``, ``or``, ``atleast``,
    ``not``, ``xor``, ``nand``, ``nor``, ``null``, ``iff``, ``imply``,
    ``cardinality``) becomes a gate. ``gate``, ``basic-event``,
    ``house-event`` and untyped ``event`` references are resolved after the
    whole file is read; an untyped reference resolves to a gate, then a basic
    event, then a house event of that name. Undefined gates become basic
    events with a warning; CCF groups are ignored with a warning.

    Raises :class:`MefError` on malformed XML or unsupported formulas.
    """
    builder = _MefBuilder()
    if isinstance(source, (str, os.PathLike)):
        source = os.fspath(source)
    try:
        for _, element in ElementTree.iterparse(source, events=("end",)):
            tag = _local(element.tag)
            if tag == "define-gate":
                builder.define_gate(element)
            elif tag == "define-basic-event":
                builder.defined_be.add(element.get("name"))
            elif tag == "define-house-event":
                builder.defined_he.add(element.get("name"))
            elif tag.lower() == "define-ccf-group":
                builder.graph.warnings.append("CCF groups ignored")
                continue
            else:
                continue
            element.clear()
    except ElementTree.ParseError as exc:
        raise MefError(f"XML parse error: {exc}") from exc
    return builder.finish()


# ----------------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------------


@dataclass(slots=True)
class NormalizedGraph:  # pylint: disable=too-many-instance-attributes
    """Graph after pass-through removal, restricted to the analysed roots.

    ``top`` is the single root or, when there are several roots, a virtual
    zero-cost gate over them (``virtual_root``; -1 if absent). ``top`` is -1
    when no root is a gate.
    """

    kind: list[int]
    gate_type: list[str | None]
    name: list[str | None]
    children: list[Sequence[int]]
    roots: list[int]
    top: int
    virtual_root: int
    pass_through_removed: int


def _find(forward: list[int], x: int) -> int:
    """Forwarding target of ``x``, with path compression."""
    root = x
    while forward[root] != root:
        root = forward[root]
    while forward[x] != root:
        forward[x], x = root, forward[x]
    return root


def _forward_target(graph: FaultTreeGraph, gate: int, children: list,
                    forward: list[int], passing: frozenset) -> int | None:
    """Resolve ``gate``'s operands; return its target if it passes through."""
    ops = [_find(forward, x) for x in children[gate]]
    connective = graph.gate_type[gate]
    if connective in _IDEMPOTENT and len(ops) > 1:
        ops = list(dict.fromkeys(ops))
    children[gate] = ops
    if len(ops) != 1:
        return None
    if connective in passing or (connective == "atleast" and
                                 graph.min_number[gate] <= 1):
        if ops[0] == gate:
            raise MefError(f"cycle through gate {graph.name[gate]!r}")
        return ops[0]
    return None


def _forward_all(graph: FaultTreeGraph, children: list,
                 passing: frozenset) -> tuple[list[int], int]:
    """Iterate pass-through forwarding to a fixed point; mutates children.

    Returns the forwarding array and the number of gates removed.
    """
    forward = list(range(len(children)))
    gates = [v for v, k in enumerate(graph.kind) if k == GATE]
    removed = 0
    changed = True
    while changed:
        changed = False
        for g in gates:
            target = (_forward_target(graph, g, children, forward, passing)
                      if forward[g] == g else None)
            if target is not None:
                forward[g] = target
                removed += 1
                changed = True
    return forward, removed


def normalize_pass_through(graph: FaultTreeGraph,
                           *,
                           fold_not: bool = False) -> NormalizedGraph:
    """Forward references through pass-through gates, to a fixed point.

    Pass-through gates are single-operand ``and``/``or``/``null`` gates
    (including a ``define-gate`` whose formula is a bare reference) and
    single-operand ``atleast`` gates with k <= 1. Duplicate operands of
    idempotent gates (``and``/``or``/``nand``/``nor``) are merged, which can
    create new pass-through gates. With ``fold_not`` single-operand
    ``not``/``nand``/``nor`` gates are forwarded too (structure only: the
    negation is dropped). ``atleast`` stays one gate. The input graph is not
    modified.
    """
    kind = list(graph.kind)
    children: list[Sequence[int]] = [
        list(c) if kind[v] == GATE else () for v, c in enumerate(graph.children)
    ]
    passing = _PASS_THROUGH | (_NEGATIONS if fold_not else frozenset())
    forward, removed = _forward_all(graph, children, passing)
    roots = [
        r for r in dict.fromkeys(_find(forward, r) for r in graph.roots)
        if kind[r] == GATE
    ]
    gate_type = list(graph.gate_type)
    name = list(graph.name)
    virtual = -1
    if len(roots) > 1:
        virtual = len(kind)
        kind.append(GATE)
        gate_type.append("virtual")
        name.append(VIRTUAL_ROOT)
        children.append(list(roots))
    top = virtual if virtual >= 0 else (roots[0] if roots else -1)
    return NormalizedGraph(kind=kind,
                           gate_type=gate_type,
                           name=name,
                           children=children,
                           roots=roots,
                           top=top,
                           virtual_root=virtual,
                           pass_through_removed=removed)


# ----------------------------------------------------------------------------
# Module detection (Dutuit & Rauzy 1996)
# ----------------------------------------------------------------------------


@dataclass(slots=True)
class ModuleDetection:
    """Visit dates of the timing DFS and the resulting module flags.

    ``post_order`` holds the gates reachable from the top in DFS exit order
    (children before parents). Dates are 0 for unreachable nodes.
    """

    first_visit: list[int]
    last_visit: list[int]
    exit_time: list[int]
    post_order: list[int]
    is_module: bytearray


def _timing_dfs(graph: NormalizedGraph) -> ModuleDetection:
    """Iterative DFS from the top with a global clock (no module flags yet).

    The clock ticks on every edge and on every gate exit. First arrival sets
    first = last; a revisit sets last; a gate exit sets exit = last. In a DAG a
    gate is revisited only after it has exited, so last >= exit for gates.
    """
    kind, children = graph.kind, graph.children
    first = [0] * len(kind)
    last = [0] * len(kind)
    exit_time = [0] * len(kind)
    post: list[int] = []
    on_stack = bytearray(len(kind))
    clock = 1
    first[graph.top] = last[graph.top] = clock
    on_stack[graph.top] = 1
    stack_v = [graph.top]
    stack_i = [0]
    while stack_v:
        v = stack_v[-1]
        i = stack_i[-1]
        clock += 1
        if i == len(children[v]):
            exit_time[v] = last[v] = clock
            on_stack[v] = 0
            post.append(v)
            stack_v.pop()
            stack_i.pop()
            continue
        stack_i[-1] = i + 1
        c = children[v][i]
        if first[c]:
            if on_stack[c]:
                raise MefError(f"cycle through gate {graph.name[c]!r}")
            last[c] = clock
            continue
        first[c] = last[c] = clock
        if kind[c] == GATE:
            on_stack[c] = 1
            stack_v.append(c)
            stack_i.append(0)
    return ModuleDetection(first, last, exit_time, post, bytearray(len(kind)))


def detect_modules(graph: NormalizedGraph) -> ModuleDetection:
    """Linear-time module detection (Dutuit & Rauzy 1996).

    One iterative DFS from ``graph.top`` records, per node, the first visit,
    the last visit (updated on every revisit) and, for gates, the exit date.
    Bottom-up over the exit order, each gate gets the min of its descendants'
    first visits and the max of their last visits. A gate v is a module iff all
    those dates lie strictly inside (first(v), exit(v)): no node below v is
    reached from outside v's cone. Leaves count, so a shared basic event
    breaks modularity. O(V + E).

    Raises :class:`MefError` on a cycle.
    """
    if graph.top < 0:
        n = len(graph.kind)
        return ModuleDetection([0] * n, [0] * n, [0] * n, [], bytearray(n))
    det = _timing_dfs(graph)
    first, last, exit_time = det.first_visit, det.last_visit, det.exit_time
    never = max(last) + 1
    # Leaves keep low = never and high = 0, so the min/max below need no
    # gate/leaf test.
    low = [never] * len(first)
    high = [0] * len(first)
    for v in det.post_order:
        lo = never
        hi = 0
        for c in graph.children[v]:
            lo = min(lo, first[c], low[c])
            hi = max(hi, last[c], high[c])
        low[v] = lo
        high[v] = hi
        if lo > first[v] and hi < exit_time[v]:
            det.is_module[v] = 1
    return det


# ----------------------------------------------------------------------------
# Module tree and local work
# ----------------------------------------------------------------------------


@dataclass(slots=True)
class ModuleTree:  # pylint: disable=too-many-instance-attributes
    """Modules arranged by nesting, with their local parts.

    The local part of module m is m plus the non-module gates whose nearest
    module ancestor is m; child modules count as single variables. Module
    cones are laminar, so every non-module gate has exactly one owner and
    every module exactly one parent module: the structure is a tree even when
    a module has several parent gates (they all lie in one local part).

    ``nodes`` lists tree nodes children-before-parents. The virtual root is a
    node only if it owns real gates (i.e. some root is not a module).
    ``local_edges`` is the cost proxy: operand edges out of local gates.
    ``local_vars`` counts distinct leaves and child modules referenced from
    the local part. ``gate_parents`` is the number of distinct parent gates.
    """

    nodes: list[int]
    parent: dict[int, int | None]
    local_gates: dict[int, int]
    local_edges: dict[int, int]
    local_vars: dict[int, int]
    depth: dict[int, int]
    gate_parents: dict[int, int]
    virtual_root: int


def _local_part(graph: NormalizedGraph, module: int, is_module: bytearray,
                owner: list[int]) -> tuple[int, int, set[int], set[int]]:
    """Traverse ``module``'s local part: (gates, edges, leaves, child modules).

    Marks the visited non-module gates in ``owner``.
    """
    kind, children = graph.kind, graph.children
    gates = 0 if module == graph.virtual_root else 1
    edges = 0
    leaves: set[int] = set()
    kids: set[int] = set()
    stack = [module]
    while stack:
        g = stack.pop()
        if g != graph.virtual_root:
            edges += len(children[g])
        for c in children[g]:
            if kind[c] != GATE:
                leaves.add(c)
            elif is_module[c]:
                kids.add(c)
            elif owner[c] == -1:
                owner[c] = module
                stack.append(c)
                gates += 1
            elif owner[c] != module:
                raise RuntimeError(f"gate {c} is in two local parts")
    return gates, edges, leaves, kids


def _local_parts(graph: NormalizedGraph, is_module: bytearray,
                 modules: list[int]):
    """Parent module and local gates/edges/variables of every module."""
    owner = [-1] * len(graph.kind)
    parent_module: dict[int, int | None] = {v: None for v in modules}
    local_gates: dict[int, int] = {}
    local_edges: dict[int, int] = {}
    local_vars: dict[int, int] = {}
    for m in modules:
        owner[m] = m
        gates, edges, leaves, kids = _local_part(graph, m, is_module, owner)
        local_gates[m] = gates
        local_edges[m] = edges
        local_vars[m] = len(leaves) + len(kids)
        for c in kids:
            if parent_module[c] is not None:
                raise RuntimeError(f"module {c} has two parent modules")
            parent_module[c] = m
    return parent_module, local_gates, local_edges, local_vars


def _link_tree(modules: list[int], parent_module: dict[int, int | None],
               local_gates: dict[int, int], virtual: int):
    """Tree nodes, parents and depths; drops a virtual root with no gates."""
    drop_virtual = virtual >= 0 and local_gates[virtual] == 0
    nodes = [v for v in modules if not (drop_virtual and v == virtual)]
    parent: dict[int, int | None] = {}
    for v in nodes:
        p = parent_module[v]
        parent[v] = None if (drop_virtual and p == virtual) else p
    depth: dict[int, int] = {}
    for v in reversed(nodes):  # parents before children
        p = parent[v]
        depth[v] = 1 if p is None else depth[p] + 1
    return nodes, parent, depth


def build_module_tree(graph: NormalizedGraph,
                      detection: ModuleDetection) -> ModuleTree:
    """Assign every reachable gate to its module and link modules into a tree.

    Each non-module gate is visited by exactly one module's traversal, so the
    cost is O(V + E). Raises ``RuntimeError`` if the laminarity invariants do
    not hold (which would indicate a detection bug).
    """
    modules = [v for v in detection.post_order if detection.is_module[v]]
    parent_module, local_gates, local_edges, local_vars = _local_parts(
        graph, detection.is_module, modules)
    nodes, parent, depth = _link_tree(modules, parent_module, local_gates,
                                      graph.virtual_root)
    gate_parents = {v: 0 for v in modules}
    for v in detection.post_order:
        for c in set(graph.children[v]):
            if c in gate_parents:
                gate_parents[c] += 1
    return ModuleTree(nodes=nodes,
                      parent=parent,
                      local_gates=local_gates,
                      local_edges=local_edges,
                      local_vars=local_vars,
                      depth=depth,
                      gate_parents=gate_parents,
                      virtual_root=graph.virtual_root)


# ----------------------------------------------------------------------------
# Scheduling bounds
# ----------------------------------------------------------------------------


def total_work(tree: ModuleTree) -> int:
    """W: summed local work (edges) over the tree nodes."""
    return sum(tree.local_edges[v] for v in tree.nodes)


def critical_path(tree: ModuleTree) -> int:
    """Heaviest root-to-leaf chain of the module tree, in local work."""
    below = {v: 0 for v in tree.nodes}
    best = 0
    for v in tree.nodes:  # children before parents
        chain = tree.local_edges[v] + below[v]
        p = tree.parent[v]
        if p is None:
            best = max(best, chain)
        else:
            below[p] = max(below[p], chain)
    return best


def greedy_list_schedule(tree: ModuleTree, cores: int) -> int:
    """Makespan of a greedy list schedule of the bottom-up collapse.

    Tasks are tree nodes weighted by local work; a module may start once all
    its child modules are done. Non-preemptive and event-driven: whenever a
    core is free it takes the ready task with the largest upward rank (local
    work summed from the task up to the root, inclusive). Any such
    work-conserving schedule satisfies max(W/P, CP) <= T_P <= W/P + CP.
    """
    if cores < 1:
        raise ValueError("cores must be >= 1")
    rank: dict[int, int] = {}
    pending = {v: 0 for v in tree.nodes}
    for v in reversed(tree.nodes):
        p = tree.parent[v]
        rank[v] = tree.local_edges[v] + (0 if p is None else rank[p])
        if p is not None:
            pending[p] += 1
    ready = [(-rank[v], v) for v in tree.nodes if pending[v] == 0]
    heapq.heapify(ready)
    running: list[tuple[int, int]] = []
    now = 0
    while ready or running:
        while len(running) < cores and ready:
            _, v = heapq.heappop(ready)
            heapq.heappush(running, (now + tree.local_edges[v], v))
        now, v = heapq.heappop(running)
        p = tree.parent[v]
        if p is not None:
            pending[p] -= 1
            if pending[p] == 0:
                heapq.heappush(ready, (-rank[p], p))
    return now


def speedup_bounds(tree: ModuleTree,
                   cores: Iterable[int]) -> tuple[float, dict[int, float]]:
    """S_inf = W / CP and S_P = W / T_P for each core count P.

    Raises ``RuntimeError`` if a schedule violates the work/span or Graham
    bounds (which would indicate a scheduling bug).
    """
    work = total_work(tree)
    span = critical_path(tree)
    s_inf = work / span if span else 1.0
    speedups: dict[int, float] = {}
    for p in cores:
        makespan = greedy_list_schedule(tree, p)
        lower = max(work / p, span)
        if makespan < lower - 1e-9 or makespan > work / p + span + 1e-9:
            raise RuntimeError(f"schedule on {p} cores out of bounds")
        speedups[p] = work / makespan if makespan else 1.0
    return s_inf, speedups


# ----------------------------------------------------------------------------
# Per-model census
# ----------------------------------------------------------------------------


@dataclass(slots=True)
class CensusResult:  # pylint: disable=too-many-instance-attributes
    """Per-model metrics (a flat record: one CSV row). Work is in edges.

    Graph counts are after normalisation and restricted to the root cones;
    ``named_gates``, ``anonymous_gates`` and ``defined_basic_events`` are raw
    file counts. A module is *nontrivial* if its local part has >= 2 gates;
    the ``lw_*`` statistics are over nontrivial modules (p90 by nearest rank).
    ``largest_share`` is the largest local work over all modules divided by
    ``total_local_work``. ``speedups`` maps a core count to W / T_P.
    """

    model: str = ""
    path: str = ""
    gates: int = 0
    basic_events: int = 0
    house_events: int = 0
    constants: int = 0
    edges: int = 0
    not_gates: int = 0
    atleast_gates: int = 0
    xor_gates: int = 0
    roots: int = 0
    pass_through_removed: int = 0
    named_gates: int = 0
    anonymous_gates: int = 0
    defined_basic_events: int = 0
    modules: int = 0
    nontrivial_modules: int = 0
    shared_modules: int = 0
    total_local_work: int = 0
    lw_max: int = 0
    lw_p90: int = 0
    lw_median: float = 0.0
    largest_share: float = 0.0
    max_local_gates: int = 0
    max_local_vars: int = 0
    tree_depth: int = 0
    max_width: int = 0
    max_width_nontrivial: int = 0
    critical_path: int = 0
    s_inf: float = 1.0
    speedups: dict[int, float] = field(default_factory=dict)
    virtual_root_local_gates: int = 0
    parse_s: float = 0.0
    census_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CensusDetails:
    """Census result together with the intermediate structures."""

    graph: FaultTreeGraph
    normalized: NormalizedGraph
    detection: ModuleDetection | None
    tree: ModuleTree | None
    result: CensusResult

    def module_names(self) -> set[str]:
        """Names of the (real) modules; anonymous gates appear as ``#id``."""
        if self.detection is None:
            return set()
        names = self.normalized.name
        return {
            names[v] or f"#{v}"
            for v in self.detection.post_order
            if self.detection.is_module[v] and v != self.normalized.virtual_root
        }


def _nearest_rank(sorted_values: Sequence[int], q: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def _fill_graph_counts(res: CensusResult, ng: NormalizedGraph,
                       det: ModuleDetection) -> None:
    real_gates = [v for v in det.post_order if v != ng.virtual_root]
    res.gates = len(real_gates)
    res.edges = sum(len(ng.children[v]) for v in real_gates)
    types = [ng.gate_type[v] for v in real_gates]
    res.not_gates = types.count("not")
    res.atleast_gates = types.count("atleast")
    res.xor_gates = types.count("xor")
    leaves = [k for v, k in enumerate(ng.kind) if det.first_visit[v]]
    res.basic_events = leaves.count(BASIC_EVENT)
    res.house_events = leaves.count(HOUSE_EVENT)
    res.constants = leaves.count(CONSTANT)


def _fill_shape(res: CensusResult, tree: ModuleTree) -> None:
    """Largest module, local maxima, depth and widths over the tree nodes."""
    res.largest_share = (max(tree.local_edges[v] for v in tree.nodes) /
                         res.total_local_work if res.total_local_work else 0.0)
    res.max_local_gates = max(tree.local_gates[v] for v in tree.nodes)
    res.max_local_vars = max(tree.local_vars[v] for v in tree.nodes)
    width: dict[int, int] = {}
    width_nt: dict[int, int] = {}
    for v in tree.nodes:
        d = tree.depth[v]
        width[d] = width.get(d, 0) + 1
        if tree.local_gates[v] >= 2:
            width_nt[d] = width_nt.get(d, 0) + 1
    res.tree_depth = max(tree.depth.values())
    res.max_width = max(width.values())
    res.max_width_nontrivial = max(width_nt.values(), default=0)


def _fill_tree_metrics(res: CensusResult, tree: ModuleTree,
                       cores: Sequence[int]) -> None:
    virtual = tree.virtual_root
    real = [v for v in tree.local_gates if v != virtual]
    nontrivial = [v for v in real if tree.local_gates[v] >= 2]
    res.total_local_work = total_work(tree)
    if res.total_local_work != res.edges:
        raise RuntimeError(f"local work {res.total_local_work} != edges "
                           f"{res.edges}")
    res.modules = len(real)
    res.nontrivial_modules = len(nontrivial)
    res.shared_modules = sum(1 for v in real if tree.gate_parents[v] >= 2)
    lw = sorted(tree.local_edges[v] for v in nontrivial)
    if lw:
        res.lw_max = lw[-1]
        res.lw_p90 = _nearest_rank(lw, 0.9)
        res.lw_median = float(statistics.median(lw))
    if tree.nodes:
        _fill_shape(res, tree)
    res.critical_path = critical_path(tree)
    res.s_inf, res.speedups = speedup_bounds(tree, cores)
    if virtual >= 0:
        res.virtual_root_local_gates = tree.local_gates[virtual]


def analyse_graph(graph: FaultTreeGraph,
                  *,
                  cores: Sequence[int] = DEFAULT_CORES,
                  fold_not: bool = False,
                  model: str = "",
                  path: str = "") -> CensusDetails:
    """Normalise, detect modules, build the module tree and compute metrics."""
    ng = normalize_pass_through(graph, fold_not=fold_not)
    res = CensusResult(model=model,
                       path=path,
                       roots=len(ng.roots),
                       pass_through_removed=ng.pass_through_removed,
                       named_gates=graph.named_gates,
                       anonymous_gates=graph.anonymous_gates,
                       defined_basic_events=graph.defined_basic_events,
                       speedups={p: 1.0 for p in cores},
                       warnings=list(graph.warnings))
    if ng.top < 0:
        return CensusDetails(graph, ng, None, None, res)
    det = detect_modules(ng)
    tree = build_module_tree(ng, det)
    _fill_graph_counts(res, ng, det)
    _fill_tree_metrics(res, tree, cores)
    return CensusDetails(graph, ng, det, tree, res)


def census_graph(graph: FaultTreeGraph, **kwargs) -> CensusResult:
    """:func:`analyse_graph` without the intermediate structures."""
    return analyse_graph(graph, **kwargs).result


def census_file(path: str | os.PathLike,
                *,
                cores: Sequence[int] = DEFAULT_CORES,
                fold_not: bool = False) -> CensusResult:
    """Parse one MEF file and return its census, with parse/census timings.

    Raises :class:`CensusError` (naming the file) on unreadable, malformed or
    cyclic input.
    """
    t0 = time.perf_counter()
    try:
        graph = parse_mef(path)
        t1 = time.perf_counter()
        res = census_graph(graph,
                           cores=cores,
                           fold_not=fold_not,
                           model=Path(path).stem,
                           path=os.fspath(path))
    except (MefError, OSError, UnicodeError) as exc:
        raise CensusError(path, exc) from exc
    res.parse_s = t1 - t0
    res.census_s = time.perf_counter() - t1
    return res


# ----------------------------------------------------------------------------
# Self-check against the definition
# ----------------------------------------------------------------------------


def _reachable(children: list[Sequence[int]], start: int) -> list[int]:
    seen = {start}
    order = []
    stack = [start]
    while stack:
        v = stack.pop()
        order.append(v)
        for c in children[v]:
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return order


def definition_modules(graph: NormalizedGraph) -> set[int]:
    """Modules by definition, O(V * (V + E)): use on small graphs only.

    A gate v reachable from the top is a module iff every proper descendant
    has all of its parents inside cone(v).
    """
    if graph.top < 0:
        return set()
    reach = _reachable(graph.children, graph.top)
    parents: dict[int, set[int]] = {v: set() for v in reach}
    for v in reach:
        for c in graph.children[v]:
            parents[c].add(v)
    out = set()
    for v in reach:
        if graph.kind[v] != GATE:
            continue
        cone = set(_reachable(graph.children, v))
        if all(parents[w] <= cone for w in cone if w != v):
            out.add(v)
    return out


def _random_formula(rng: random.Random, i: int, num_gates: int,
                    num_events: int) -> str:
    ops = []
    for _ in range(rng.randint(1, 4)):
        if i + 1 < num_gates and rng.random() < 0.45:
            ops.append(f'<gate name="G{rng.randint(i + 1, num_gates - 1)}"/>')
        else:
            ops.append(
                f'<basic-event name="e{rng.randint(0, num_events - 1)}"/>')
    args = "".join(ops)
    pick = rng.random()
    if pick < 0.4:
        return f"<or>{args}</or>"
    if pick < 0.8:
        return f"<and>{args}</and>"
    if pick < 0.9:
        return f'<atleast min="{max(1, len(ops) - 1)}">{args}</atleast>'
    extra = rng.randint(0, num_events - 1)
    return f'<or><and>{args}</and><basic-event name="e{extra}"/></or>'


def random_mef_dag(rng: random.Random,
                   *,
                   max_gates: int = 30,
                   max_events: int = 25) -> bytes:
    """Random acyclic MEF model, often with several roots and shared nodes.

    Gate Gi may reference gates Gj with j > i and events e0..e{m-1}. Mixes
    ``or``, ``and``, ``atleast`` and nested anonymous formulas; single-operand
    gates exercise pass-through removal.
    """
    num_gates = rng.randint(2, max_gates)
    num_events = rng.randint(2, max_events)
    body = "".join(f'<define-gate name="G{i}">'
                   f"{_random_formula(rng, i, num_gates, num_events)}"
                   "</define-gate>" for i in range(num_gates))
    return (f'<opsa-mef><define-fault-tree name="random">{body}'
            "</define-fault-tree></opsa-mef>").encode()


def self_check_random_dags(trials: int = 300,
                           *,
                           seed: int = 1996,
                           max_gates: int = 30,
                           max_events: int = 25) -> dict[str, int]:
    """Compare :func:`detect_modules` with :func:`definition_modules`.

    Runs the full census on ``trials`` random models from
    :func:`random_mef_dag` and raises ``AssertionError`` on the first
    disagreement; the census itself raises ``RuntimeError`` if a tree or
    scheduling invariant fails. Returns coverage counts: trials (models with a
    gate root), gates, modules, multi_root.
    """
    rng = random.Random(seed)
    stats = {"trials": 0, "gates": 0, "modules": 0, "multi_root": 0}
    for trial in range(trials):
        xml = random_mef_dag(rng, max_gates=max_gates, max_events=max_events)
        details = analyse_graph(parse_mef(io.BytesIO(xml)))
        det = details.detection
        if det is None:
            continue
        got = {v for v in det.post_order if det.is_module[v]}
        want = definition_modules(details.normalized)
        if got != want:
            raise AssertionError(f"trial {trial}: modules differ on "
                                 f"{sorted(got ^ want)}")
        stats["trials"] += 1
        stats["gates"] += len(det.post_order)
        stats["modules"] += len(got)
        stats["multi_root"] += details.normalized.virtual_root >= 0
    return stats


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------


def result_columns(cores: Sequence[int]) -> list[str]:
    """CSV column names: result fields with ``s_<P>`` after ``s_inf``."""
    cols = []
    for f in dataclasses.fields(CensusResult):
        if f.name == "speedups":
            cols += [f"s_{p}" for p in cores]
        else:
            cols.append(f.name)
    return cols


def _row(res: CensusResult, cores: Sequence[int]) -> dict[str, object]:
    row: dict[str, object] = {}
    for f in dataclasses.fields(CensusResult):
        value = getattr(res, f.name)
        if f.name == "speedups":
            row.update((f"s_{p}", round(value.get(p, 1.0), 6)) for p in cores)
        elif f.name == "warnings":
            row[f.name] = "; ".join(value)
        else:
            row[f.name] = round(value, 6) if isinstance(value, float) else value
    return row


def write_csv(results: Iterable[CensusResult], cores: Sequence[int],
              out: TextIO) -> None:
    """Write one CSV row per result."""
    writer = csv.DictWriter(out, fieldnames=result_columns(cores))
    writer.writeheader()
    for res in results:
        writer.writerow(_row(res, cores))


def write_json(results: Iterable[CensusResult], cores: Sequence[int],
               out: TextIO) -> None:
    """Write the results as a JSON array of objects (same keys as the CSV)."""
    json.dump([_row(r, cores) for r in results], out, indent=2)
    out.write("\n")


def _markdown_cells(r: CensusResult, cores: Sequence[int]) -> list[str]:
    if r.nontrivial_modules:
        lw = [str(r.lw_max), str(r.lw_p90), f"{r.lw_median:g}"]
    else:
        lw = ["–", "–", "–"]
    counts = [r.gates, r.basic_events, r.edges, r.modules, r.nontrivial_modules]
    shape = [r.tree_depth, r.max_width, r.critical_path]
    return ([r.model] + [str(x) for x in counts] + lw +
            [f"{r.largest_share:.3f}"] + [str(x) for x in shape] +
            [f"{r.s_inf:.2f}"] +
            [f"{r.speedups.get(p, 1.0):.2f}" for p in cores] +
            [str(r.max_local_vars), f"{r.parse_s:.2f}", f"{r.census_s:.2f}"])


def format_markdown(results: Iterable[CensusResult],
                    cores: Sequence[int]) -> str:
    """Markdown table of the main metrics, one row per model."""
    head = [
        "model", "gates", "BE", "edges", "modules", "nontrivial", "lw max",
        "lw p90", "lw median", "largest share", "depth", "max width", "CP", "S∞"
    ] + [f"S{p}" for p in cores] + ["max local vars", "parse s", "census s"]
    lines = [
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|",
    ]
    lines += [
        "| " + " | ".join(_markdown_cells(r, cores)) + " |" for r in results
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------


def _xml_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.xml") if p.is_file())


def expand_paths(spec: str) -> list[Path]:
    """Files for one PATH argument: a file, a directory or a glob pattern.

    Directories are searched recursively for ``*.xml``; glob patterns accept
    ``**``. Raises :class:`CensusError` if nothing matches.
    """
    path = Path(spec)
    if path.is_file():
        return [path]
    if path.is_dir():
        found = _xml_files(path)
    elif any(ch in spec for ch in "*?["):
        found = []
        for match in map(Path, sorted(glob.glob(spec, recursive=True))):
            found += _xml_files(match) if match.is_dir() else [match]
    else:
        raise CensusError(spec, "no such file or directory")
    if not found:
        raise CensusError(spec, "matched no .xml files")
    return found


def _parse_cores(text: str) -> tuple[int, ...]:
    try:
        cores = tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}") from exc
    if not cores or any(p < 1 for p in cores):
        raise argparse.ArgumentTypeError("core counts must be >= 1")
    return cores


def _arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="module_census.py",
        description="Module census of OpenPSA MEF fault trees: modules, "
        "module tree, local work and task-parallel bounds (#213).")
    ap.add_argument("paths",
                    nargs="+",
                    metavar="PATH",
                    help="MEF file, directory (recursive *.xml) or glob")
    ap.add_argument("--cores",
                    type=_parse_cores,
                    default=DEFAULT_CORES,
                    help="core counts for the greedy-schedule bound "
                    "(default: 8,48)")
    ap.add_argument("--csv",
                    metavar="OUT",
                    help="write a CSV with all metrics ('-' for stdout)")
    ap.add_argument("--json",
                    metavar="OUT",
                    help="write JSON with all metrics ('-' for stdout)")
    ap.add_argument("--markdown",
                    action="store_true",
                    help="print the markdown table even with --csv/--json")
    ap.add_argument("--fold-not",
                    action="store_true",
                    help="treat single-operand not/nand/nor as pass-through")
    ap.add_argument("--keep-going",
                    action="store_true",
                    help="report failing files and continue (exit status "
                    "stays non-zero)")
    return ap


def _emit(target: str, writer, results, cores) -> None:
    if target == "-":
        writer(results, cores, sys.stdout)
        return
    with open(target, "w", newline="", encoding="utf-8") as out:
        writer(results, cores, out)


def _collect(args: argparse.Namespace) -> tuple[list[CensusResult], int]:
    """Census every PATH; returns (results, number of failures).

    Stops at the first failure unless ``--keep-going``.
    """
    results: list[CensusResult] = []
    failures = 0

    def stop(exc: CensusError) -> bool:
        nonlocal failures
        failures += 1
        print(f"module_census: error: {exc}", file=sys.stderr)
        return not args.keep_going

    paths: list[Path] = []
    for spec in args.paths:
        try:
            paths += expand_paths(spec)
        except CensusError as exc:
            if stop(exc):
                return results, failures
    for path in dict.fromkeys(paths):
        try:
            results.append(
                census_file(path, cores=args.cores, fold_not=args.fold_not))
        except CensusError as exc:
            if stop(exc):
                break
    return results, failures


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit status."""
    args = _arg_parser().parse_args(argv)
    results, failures = _collect(args)
    if failures and not args.keep_going:
        return 1
    if args.csv:
        _emit(args.csv, write_csv, results, args.cores)
    if args.json:
        _emit(args.json, write_json, results, args.cores)
    if args.markdown or not (args.csv or args.json):
        print(format_markdown(results, args.cores))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
