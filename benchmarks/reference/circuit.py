"""Indexed Boolean circuit for one target.

Node 0 is the constant FALSE, nodes ``1..n_vars`` are events, the remaining
nodes are gates in topological order (children first). A literal is
``2 * node + complemented``, so literal 0 is FALSE and literal 1 is TRUE.
Events are numbered in order of first use by the gates, which fixes the order
in which samplers draw them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection, Mapping

from .ftp import AND, ATLEAST, OR, Diagnostic, Location, ModelError, Project

FALSE, TRUE = 0, 1
POS, NEG = 1, 2  # polarity bits: reachable from the root through an even / odd number of complements

# Which import values become constants when the circuit is built.
CONSTANTS = ("boolean", "hull", "none")

# Literal modes of a hull query (RFC §3) for the events referenced with a direct
# complement (``-NAME`` on an event) in the target's cone: ``split`` (default)
# makes each such complement an atom of its own, ``-NAME``, droppable and valued
# not x (the atom-split hull); ``omega`` fixes those events (Rauzy's omega_L);
# ``positive`` gives the plain hull, every event droppable.
LITERALS = ("split", "omega", "positive")
# Per event under omega_L: which of x_v and not x_v are in L. A witness y may
# use y_v in [lo, hi] (``bounds``).
ONLY_POS, BOTH, ONLY_NEG, NEITHER = "pos", "both", "neg", "neither"


@dataclass
class Circuit:
    names: list[str]
    n_vars: int
    probs: list[float]  # per node; meaningful for events only
    kinds: list[str]  # per gate
    ks: list[int]
    kids: list[tuple[int, ...]]
    root: int  # literal
    twins: dict[int, int] = field(default_factory=dict)  # complement atom -> its event, when both are variables

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


def cone_gates(project: Project, target: str, overrides: Mapping[str, int | bool] | None = None) -> list[str]:
    """Gates in the cone of ``target`` in topological order (children first);
    overridden gates are leaves."""
    gates = project.gates
    over = overrides or {}

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
    return order


def cone_events(project: Project, target: str, overrides: Mapping[str, int | bool] | None = None) -> list[str]:
    """Events referenced in the cone of ``target`` (the target itself if it is
    an event), in order of first reference, including overridden events."""
    gates = project.gates
    out = dict.fromkeys([target] if target not in gates else [])
    for name in cone_gates(project, target, overrides):
        out.update((c, None) for c, _ in gates[name].children if c not in gates)
    return list(out)


def complemented_events(project: Project, target: str, overrides: Mapping[str, int | bool] | None = None) -> list[str]:
    """Events referenced with a direct complement (``-NAME``) by a gate in the
    cone of ``target``, sorted: the events whose complements are atoms for
    ``literals="split"`` and fixed for ``"omega"`` (RFC §3). Complements of gates
    (also of single-input gates) are delete terms and add nothing."""
    gates, over = project.gates, overrides or {}
    return sorted({
        c for name in cone_gates(project, target, over) for c, neg in gates[name].children
        if neg and c not in gates and c not in over
    })


def literal_sets(project: Project, target: str, literals: str,
                 overrides: Mapping[str, int | bool] | None = None) -> tuple[list[str], list[str]]:
    """(fixed, split) events under ``literals``: the directly complemented events
    are fixed for ``omega``, split into atoms for ``split``, neither for
    ``positive``."""
    if literals not in LITERALS:
        raise ValueError(f"literals must be one of {LITERALS}, got {literals!r}")
    events = complemented_events(project, target, overrides) if literals != "positive" else []
    return (events, []) if literals == "omega" else ([], events)


def complement_under_negation(project: Project, target: str, overrides: Mapping[str, int | bool] | None = None,
                              max_initiators: int | None = None) -> list[Diagnostic]:
    """Warnings ``complement-under-negation``: directly complemented events of the
    target's cone that also occur in the cone of a complemented gate (or, with
    the initiator rule, are initiators). The atom-split hull and omega_L can
    differ only there (RFC Theorem 10(c))."""
    gates, over = project.gates, overrides or {}
    cone = cone_gates(project, target, over)
    direct: dict[str, Location] = {}
    for name in cone:
        for c, neg in gates[name].children:
            if neg and c not in gates and c not in over:
                direct.setdefault(c, gates[name].location)
    if not direct:
        return []
    under: dict[str, Location | None] = {}
    for name in cone:
        for c, neg in gates[name].children:
            if neg and c in gates and c not in over:
                for e in cone_events(project, c, over):
                    under.setdefault(e, gates[name].location)
    if max_initiators is not None:
        for e in project.initiators() & set(direct):
            under.setdefault(e, None)
    out = []
    for e in sorted(direct):
        if e not in under:
            continue
        where = "the initiator rule" if under[e] is None else "a complemented gate"
        locs = (direct[e],) if under[e] is None else (direct[e], under[e])
        out.append(Diagnostic("warning", "complement-under-negation",
                              f"{e} is complemented directly and also occurs under {where} in the cone of {target}; "
                              "the atom-split hull and omega_L can differ", e, locs))
    return out


def hull_record(project: Project, target: str, overrides: Mapping[str, int | bool] | None = None,
                max_initiators: int | None = None) -> dict:
    """The directly complemented events of the target's cone (atoms under
    ``split``, fixed under ``omega``) and those that also occur under a
    complemented gate (the ``complement-under-negation`` warnings)."""
    return {
        "complemented_events": complemented_events(project, target, overrides),
        "complement_under_negation": [
            d.name for d in complement_under_negation(project, target, overrides, max_initiators)
        ],
    }


def event_modes(names: list[str], n_vars: int, negative: Collection[str]) -> list:
    """Per node (index 0 and gates unused): BOTH for events in ``negative``,
    else ONLY_POS."""
    neg = set(negative)
    return [None] + [BOTH if names[v] in neg else ONLY_POS for v in range(1, n_vars + 1)]


def bounds(mode: str, x, ones, zeros):
    """Values a witness may give an event whose value in the world is ``x``:
    the interval ``[lo, hi]`` (bitwise over lanes)."""
    if mode == ONLY_POS:
        return zeros, x
    if mode == BOTH:
        return x, x
    if mode == ONLY_NEG:
        return x, ones
    return zeros, ones


def build_circuit(
    project: Project,
    target: str,
    probs: Mapping[str, float],
    overrides: Mapping[str, int | bool] | None = None,
    constants: str = "boolean",
    fold: bool = True,
    max_initiators: int | None = None,
    split: Collection[str] = (),
) -> Circuit:
    """Circuit for ``target`` restricted to its cone.

    ``overrides`` replace gates or events by Boolean constants (in the function,
    not in the measure). ``constants`` selects which import values are folded:
    ``boolean`` folds p = 0 and p = 1, ``hull`` folds only p = 0 (p = 1 events
    stay droppable), ``none`` keeps every event as a variable. ``fold`` enables
    constant propagation, alias removal and structural hashing.

    ``max_initiators = m`` conjoins the delete term ``not atleast(m + 1; I)``,
    with I the initiator-tagged events in the cone (RFC §4.8); it is omitted
    when ``|I| <= m``.

    ``split``: events whose direct complements become atoms of their own (the
    split circuit of RFC §3): a complemented edge into such an event reads a
    variable ``-NAME`` with probability 1 - p, and ``Circuit.twins`` couples it
    to ``NAME`` when both occur. Gate complements are unchanged.
    """
    if constants not in CONSTANTS:
        raise ValueError(f"constants must be one of {CONSTANTS}")
    if max_initiators is not None and max_initiators < 0:
        raise ValueError("max_initiators must be non-negative")
    gates = project.gates
    over = {name: int(bool(v)) for name, v in (overrides or {}).items()}
    unknown = sorted(n for n in over if n not in gates and n not in probs)
    if unknown:
        raise ModelError(f"override names unknown nodes: {', '.join(unknown)}")
    if target not in gates and target not in probs and target not in over:
        raise ModelError(f"unknown target {target}")

    def expandable(name: str) -> bool:
        return name in gates and name not in over

    order = cone_gates(project, target, over)

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

    split_set = set(split)

    def atom(name: str) -> int:
        """The complement atom of event ``name`` (folded like an event of
        probability 1 - p)."""
        if name not in probs:
            raise ModelError(f"no probability for event {name}")
        q = 1.0 - probs[name]
        if constants != "none" and q == 0.0:
            return FALSE
        key = "-" + name
        node = var_of.get(key)
        if node is None:
            node = var_of[key] = len(tmp)
            tmp.append(("var", key, q))
        return 2 * node

    def child(c: str, neg: bool) -> int:
        if neg and c in split_set and not expandable(c) and c not in over:
            return atom(c)
        return ref(c) ^ neg

    def add(name: str, kind: str, k: int, lits: list[int]) -> int:
        if not fold:
            tmp.append(("gate", name, kind, k, tuple(lits)))
            return 2 * (len(tmp) - 1)
        res = _fold(kind, k, lits)
        if isinstance(res, int):
            return res
        kind, k, flits = res
        key = (kind, k, tuple(sorted(flits)))
        node = hashed.get(key)
        if node is None:
            node = hashed[key] = len(tmp)
            tmp.append(("gate", name, kind, k, flits))
        return 2 * node

    for name in order:
        g = gates[name]
        lit_of[name] = add(name, g.kind, g.k, [child(c, neg) for c, neg in g.children])
    root = ref(target)
    if max_initiators is not None:
        tagged = project.initiators()
        lits = [leaf(e) for e in cone_events(project, target, over) if e in tagged]
        if len(lits) > max_initiators:
            delete = add("$INITIATORS", ATLEAST, max_initiators + 1, lits)
            root = add("$TARGET", AND, 0, [root, delete ^ 1])
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
    var_index = {names[v]: v for v in range(1, 1 + len(var_ids))}
    twins = {v: var_index[names[v][1:]] for v in var_index.values()
             if names[v].startswith("-") and names[v][1:] in var_index}
    return Circuit(
        names=names,
        n_vars=len(var_ids),
        probs=probs,
        kinds=[tmp[i][2] for i in gate_ids],
        ks=[tmp[i][3] for i in gate_ids],
        kids=[tuple(2 * new[lit >> 1] + (lit & 1) for lit in tmp[i][4]) for i in gate_ids],
        root=2 * new[root >> 1] + (root & 1),
        twins=twins,
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


def relaxed(c: Circuit, modes: list, lit: int) -> bool:
    """Whether the coherent bound g_coh^L replaces ``lit`` by TRUE: complemented
    gates (and TRUE itself), and event literals that are not in L."""
    node = lit >> 1
    if node == 0 or node >= c.first_gate:
        return bool(lit & 1)
    if lit & 1:
        return modes[node] in (ONLY_POS, NEITHER)
    return modes[node] in (ONLY_NEG, NEITHER)


def relaxed_below(c: Circuit, modes: list) -> list[bool]:
    """Per gate: whether a relaxed literal occurs in its cone."""
    first = c.first_gate
    out = [False] * len(c.kinds)
    for g, kids in enumerate(c.kids):
        out[g] = any(relaxed(c, modes, lit) or (lit >> 1 >= first and out[(lit >> 1) - first]) for lit in kids)
    return out
