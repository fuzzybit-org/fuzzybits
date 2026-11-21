"""Indexed Boolean circuit for one target.

Node 0 is the constant FALSE, nodes ``1..n_vars`` are events, the remaining
nodes are gates in topological order (children first). A literal is
``2 * node + complemented``, so literal 0 is FALSE and literal 1 is TRUE.
Events are numbered in order of first use by the gates, which fixes the order
in which samplers draw them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .ftp import AND, ATLEAST, OR, ModelError, Project

FALSE, TRUE = 0, 1
POS, NEG = 1, 2  # polarity bits: reachable from the root through an even / odd number of complements

# Which import values become constants when the circuit is built.
CONSTANTS = ("boolean", "hull", "none")


@dataclass
class Circuit:
    names: list[str]
    n_vars: int
    probs: list[float]  # per node; meaningful for events only
    kinds: list[str]  # per gate
    ks: list[int]
    kids: list[tuple[int, ...]]
    root: int  # literal

    @property
    def first_gate(self) -> int:
        return 1 + self.n_vars

    @property
    def n_nodes(self) -> int:
        return 1 + self.n_vars + len(self.kinds)

    def constant(self) -> int | None:
        return self.root if self.root < 2 else None


def _fold(kind: str, k: int, lits: list[int]) -> int | tuple[str, int, tuple[int, ...]]:
    """Function-preserving local simplification; returns a literal or a gate."""
    if kind == ATLEAST:
        k -= sum(1 for lit in lits if lit == TRUE)
        lits = [lit for lit in lits if lit > TRUE]
        if k <= 0:
            return TRUE
        if k > len(lits):
            return FALSE
        if k == 1:
            kind = OR
        elif k == len(lits):
            kind = AND
    if kind in (OR, AND):
        absorbing, neutral = (TRUE, FALSE) if kind == OR else (FALSE, TRUE)
        if absorbing in lits:
            return absorbing
        uniq = list(dict.fromkeys(lit for lit in lits if lit != neutral))
        present = set(uniq)
        if any(lit ^ 1 in present for lit in uniq):
            return absorbing
        if not uniq:
            return neutral
        if len(uniq) == 1:
            return uniq[0]
        return (kind, 0, tuple(uniq))
    return (kind, k, tuple(lits))


def build_circuit(
    project: Project,
    target: str,
    probs: Mapping[str, float],
    overrides: Mapping[str, int | bool] | None = None,
    constants: str = "boolean",
    fold: bool = True,
) -> Circuit:
    """Circuit for ``target`` restricted to its cone.

    ``overrides`` replace gates or events by Boolean constants (in the function,
    not in the measure). ``constants`` selects which import values are folded:
    ``boolean`` folds p = 0 and p = 1, ``hull`` folds only p = 0 (p = 1 events
    stay droppable), ``none`` keeps every event as a variable. ``fold`` enables
    constant propagation, alias removal and structural hashing.
    """
    if constants not in CONSTANTS:
        raise ValueError(f"constants must be one of {CONSTANTS}")
    gates = project.gates
    over = {name: int(bool(v)) for name, v in (overrides or {}).items()}
    unknown = sorted(n for n in over if n not in gates and n not in probs)
    if unknown:
        raise ModelError(f"override names unknown nodes: {', '.join(unknown)}")
    if target not in gates and target not in probs and target not in over:
        raise ModelError(f"unknown target {target}")

    def expandable(name: str) -> bool:
        return name in gates and name not in over

    order: list[str] = []
    if expandable(target):
        state = {target: 1}
        stack = [(target, 0)]
        while stack:
            name, i = stack[-1]
            kids = gates[name].children
            if i < len(kids):
                stack[-1] = (name, i + 1)
                child = kids[i][0]
                if expandable(child):
                    s = state.get(child)
                    if s is None:
                        state[child] = 1
                        stack.append((child, 0))
                    elif s == 1:
                        raise ModelError(f"cycle through gate {child}")
                continue
            state[name] = 2
            order.append(name)
            stack.pop()

    # Temporary nodes: 0 = FALSE, then events and gates as they are met.
    tmp: list[tuple] = [("const",)]
    var_of: dict[str, int] = {}
    lit_of: dict[str, int] = {}
    hashed: dict[tuple, int] = {}

    def leaf(name: str) -> int:
        if name in over:
            return over[name]
        if name not in probs:
            raise ModelError(f"no probability for event {name}")
        p = probs[name]
        if constants != "none" and p == 0.0:
            return FALSE
        if constants == "boolean" and p == 1.0:
            return TRUE
        node = var_of.get(name)
        if node is None:
            node = var_of[name] = len(tmp)
            tmp.append(("var", name, p))
        return 2 * node

    def ref(name: str) -> int:
        return lit_of[name] if expandable(name) else leaf(name)

    for name in order:
        g = gates[name]
        lits = [ref(c) ^ neg for c, neg in g.children]
        if not fold:
            lit_of[name] = 2 * len(tmp)
            tmp.append(("gate", name, g.kind, g.k, tuple(lits)))
            continue
        res = _fold(g.kind, g.k, lits)
        if isinstance(res, int):
            lit_of[name] = res
            continue
        kind, k, flits = res
        key = (kind, k, tuple(sorted(flits)))
        node = hashed.get(key)
        if node is None:
            node = hashed[key] = len(tmp)
            tmp.append(("gate", name, kind, k, flits))
        lit_of[name] = 2 * node
    root = ref(target)
    return _finalize(tmp, root)


def _finalize(tmp: list[tuple], root: int) -> Circuit:
    reach = [False] * len(tmp)
    reach[root >> 1] = True
    for i in range(len(tmp) - 1, 0, -1):
        if reach[i] and tmp[i][0] == "gate":
            for lit in tmp[i][4]:
                reach[lit >> 1] = True
    gate_ids = [i for i in range(len(tmp)) if reach[i] and tmp[i][0] == "gate"]
    var_ids: list[int] = []
    seen: set[int] = set()
    if tmp[root >> 1][0] == "var":
        var_ids.append(root >> 1)
        seen.add(root >> 1)
    for i in gate_ids:
        for lit in tmp[i][4]:
            n = lit >> 1
            if tmp[n][0] == "var" and n not in seen:
                seen.add(n)
                var_ids.append(n)
    new = {0: 0}
    for i in var_ids + gate_ids:
        new[i] = len(new)
    names = ["$FALSE"] + [tmp[i][1] for i in var_ids] + [tmp[i][1] for i in gate_ids]
    probs = [0.0] + [tmp[i][2] for i in var_ids] + [0.0] * len(gate_ids)
    return Circuit(
        names=names,
        n_vars=len(var_ids),
        probs=probs,
        kinds=[tmp[i][2] for i in gate_ids],
        ks=[tmp[i][3] for i in gate_ids],
        kids=[tuple(2 * new[lit >> 1] + (lit & 1) for lit in tmp[i][4]) for i in gate_ids],
        root=2 * new[root >> 1] + (root & 1),
    )


@dataclass
class Plan:
    """Evaluation schedule: events to draw before each gate and nodes whose
    last consumer is each gate (freed afterwards)."""

    new_vars: list[range]
    free: list[tuple[int, ...]]
    peak_live: int


def plan(c: Circuit) -> Plan:
    n_gates = len(c.kinds)
    last = [-1] * c.n_nodes
    for g, kids in enumerate(c.kids):
        for lit in kids:
            last[lit >> 1] = g
    keep = c.root >> 1
    free: list[list[int]] = [[] for _ in range(n_gates)]
    for node in range(1, c.n_nodes):
        if node != keep and last[node] >= 0:
            free[last[node]].append(node)
    new_vars: list[range] = []
    nxt = 1
    for kids in c.kids:
        hi = max((lit >> 1 for lit in kids if 1 <= lit >> 1 <= c.n_vars), default=0)
        hi = max(hi + 1, nxt)
        new_vars.append(range(nxt, hi))
        nxt = hi
    live = peak = 0
    for g in range(n_gates):
        live += len(new_vars[g]) + 1
        peak = max(peak, live)
        live -= len(free[g])
    return Plan(new_vars, [tuple(f) for f in free], max(peak, 1))


def polarity(c: Circuit) -> list[int]:
    """Per node: POS if reachable from the root through an even number of
    complements, NEG if through an odd number, both bits if binate."""
    pol = [0] * c.n_nodes
    pol[c.root >> 1] = NEG if c.root & 1 else POS
    first = c.first_gate
    for g in range(len(c.kinds) - 1, -1, -1):
        p = pol[first + g]
        if not p:
            continue
        swapped = ((p & POS) << 1) | ((p & NEG) >> 1)
        for lit in c.kids[g]:
            pol[lit >> 1] |= swapped if lit & 1 else p
    return pol


def negation_below(c: Circuit) -> list[bool]:
    """Per gate: whether a complemented edge occurs in its cone."""
    first = c.first_gate
    out = [False] * len(c.kinds)
    for g, kids in enumerate(c.kids):
        out[g] = any((lit & 1) or (lit >> 1 >= first and out[(lit >> 1) - first]) for lit in kids)
    return out
