"""Exact Boolean and hull probabilities by enumeration.

Implements the definitions directly on the unfolded circuit: every event in
the cone is a variable weighted by its import value (including 0 and 1), and

    P(g)          = sum_x g(x) w(x)
    P(h)          = sum_a [exists b <= a : g_split(b)] w(a)       (atom-split)
    P(omega_L g)  = sum_x [exists y <=_L x : g(y)] w(x)

For the atom-split hull the variables are atoms (events, and one atom ``-B``
per direct complement of B), the closure is the subset-OR (zeta) transform over
all atoms, and the weight of an atom assignment is zero unless every coupled
pair ``B``, ``-B`` takes opposite values. For omega_L, ``y <=_L x`` constrains
each event separately, so the closure is one OR-transform per droppable event
axis. Nothing here relies on the folding, polarity or Kleene steps used by the
estimators, so the estimators are checked against an independent path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .bitpar import ALL, evaluate
from .circuit import (
    BOTH,
    ONLY_NEG,
    ONLY_POS,
    Circuit,
    build_circuit,
    cone_events,
    event_modes,
    hull_record,
    literal_sets,
    plan,
)
from .ftp import ModelError, Project

MAX_EXACT_VARIABLES = 22
MAX_CUT_SET_TABLE = 1 << 24  # 3^|fixed| * 2^|droppable| entries for exact_cut_sets


@dataclass
class TruthTable:
    names: list[str]  # variable i is bit i of the assignment index
    probs: list[float]
    table: np.ndarray  # bool, length 2**n
    modes: list[str] | None = None  # per variable, set for hull tables
    twins: dict[int, int] | None = None  # complement atom -> its event (variable indices)


def _patterns(n: int) -> tuple[list[np.ndarray], int]:
    n_words = max(1, (1 << n) // 64)
    words = np.arange(n_words, dtype=np.uint64)
    low = [0xAAAAAAAAAAAAAAAA, 0xCCCCCCCCCCCCCCCC, 0xF0F0F0F0F0F0F0F0,
           0xFF00FF00FF00FF00, 0xFFFF0000FFFF0000, 0xFFFFFFFF00000000]
    out = []
    for i in range(n):
        if i < 6:
            out.append(np.full(n_words, np.uint64(low[i]), dtype=np.uint64))
        else:
            bit = (words >> np.uint64(i - 6)) & np.uint64(1)
            out.append(np.where(bit == 1, ALL, np.uint64(0)).astype(np.uint64))
    return out, n_words


def truth_table(c: Circuit) -> TruthTable:
    n = c.n_vars
    if n > MAX_EXACT_VARIABLES:
        raise ModelError(f"{n} variables exceed the exact limit of {MAX_EXACT_VARIABLES}; use 'estimate'")
    pats, n_words = _patterns(n)
    ones = np.full(n_words, ALL, dtype=np.uint64)
    zeros = np.zeros(n_words, dtype=np.uint64)
    root, _ = evaluate(c, plan(c), lambda v: pats[v - 1], ones, zeros)
    bits = np.unpackbits(np.asarray(root, dtype=np.uint64).view(np.uint8), bitorder="little")[: 1 << n]
    twins = {a - 1: e - 1 for a, e in c.twins.items()}
    return TruthTable(c.names[1 : 1 + n], c.probs[1 : 1 + n], bits.astype(bool), twins=twins)


def closure(table: np.ndarray, modes: list[str]) -> np.ndarray:
    """omega_L of a truth table: for each event axis, OR over the values a
    witness may take there (``modes[i]`` for variable ``i``)."""
    up = table.copy()
    for i, mode in enumerate(modes):
        if mode == BOTH:
            continue
        view = up.reshape(-1, 2, 1 << i)
        if mode == ONLY_POS:
            view[:, 1, :] |= view[:, 0, :]
        elif mode == ONLY_NEG:
            view[:, 0, :] |= view[:, 1, :]
        else:
            either = view[:, 0, :] | view[:, 1, :]
            view[:, 0, :] = either
            view[:, 1, :] = either
    return up


def weights(probs: list[float], twins: Mapping[int, int] | None = None) -> np.ndarray:
    """Probability of every assignment (bit i = variable i). Coupled pairs
    (``twins``, either direction) take opposite values: the later variable of a
    pair carries no factor of its own."""
    partner = {**(twins or {}), **{e: a for a, e in (twins or {}).items()}}
    w = np.ones(1, dtype=np.float64)
    for i, p in enumerate(probs):
        j = partner.get(i, i)
        if j < i:
            earlier = ((np.arange(len(w)) >> j) & 1).astype(np.float64)
            w = np.concatenate([w * earlier, w * (1.0 - earlier)])
        else:
            w = np.concatenate([w * (1.0 - p), w * p])
    return w


def exact_table(
    project: Project,
    target: str,
    probs: Mapping[str, float],
    semantics: str,
    overrides: Mapping[str, int | bool] | None = None,
    literals: str = "split",
    max_initiators: int | None = None,
) -> TruthTable:
    """Truth table of g (``boolean``) or of the hull (``hull``, in the literal
    mode ``literals``) over the cone's events, or over its atoms for ``split``;
    ``max_initiators`` conjoins the initiator delete term first."""
    if semantics not in ("boolean", "hull"):
        raise ValueError("semantics must be 'boolean' or 'hull'")
    fixed, split = literal_sets(project, target, literals, overrides) if semantics == "hull" else ([], [])
    c = build_circuit(project, target, probs, overrides, constants="none", fold=False, max_initiators=max_initiators,
                      split=split)
    tt = truth_table(c)
    if semantics == "hull":
        tt.modes = event_modes(c.names, c.n_vars, fixed)[1:]
        tt.table = closure(tt.table, tt.modes)
    return tt


def exact_probability(
    project: Project,
    target: str,
    probs: Mapping[str, float],
    semantics: str,
    overrides: Mapping[str, int | bool] | None = None,
    literals: str = "split",
    max_initiators: int | None = None,
) -> dict:
    tt = exact_table(project, target, probs, semantics, overrides, literals, max_initiators)
    w = weights(tt.probs, tt.twins)
    p = math.fsum(w[tt.table].tolist())
    out = {"target": target, "semantics": semantics}
    if semantics == "hull":
        out.update(literals=literals, **hull_record(project, target, overrides, max_initiators))
    out.update(
        max_initiators_per_cutset=max_initiators,
        method="enumeration",
        probability=min(1.0, max(0.0, p)),
        variables=len(tt.names),
    )
    return out


def exact_frequency(
    project: Project,
    target: str,
    probs: Mapping[str, float],
    semantics: str = "boolean",
    overrides: Mapping[str, int | bool] | None = None,
    literals: str = "split",
) -> dict:
    """Initiating-event frequency of ``target`` (#243): the values of the initiating events (``I``
    imports) in its cone are frequencies f_i, and

        F(T) = sum_i f_i * P(T | I_i = TRUE, I_j = FALSE for every other initiating event j of the cone)

    with each conditioned target evaluated exactly (the initiating events as constants, through
    ``overrides``; under ``hull`` the hull of each conditioned target). Raises :class:`ModelError` when
    the cone holds no initiating event, or when ``T`` can hold with every initiating event FALSE: its
    value would mix a probability with a frequency."""
    over = dict(overrides or {})
    initiators = sorted(set(cone_events(project, target, over)) & project.initiators() - set(over))
    if not initiators:
        raise ModelError(f"target {target!r} has no initiating event in its cone")
    without = exact_probability(project, target, probs, semantics, {**over, **{i: 0 for i in initiators}}, literals)
    if without["probability"] > 0.0:
        raise ModelError(f"target {target!r} can hold with every initiating event FALSE: mixed units")
    terms = []
    for initiator in initiators:
        condition = {**over, **{i: int(i == initiator) for i in initiators}}
        value = exact_probability(project, target, probs, semantics, condition, literals)["probability"]
        terms.append({"initiator": initiator, "frequency": float(probs[initiator]), "probability": value})
    return {
        "target": target,
        "semantics": semantics,
        "method": "enumeration",
        "frequency": math.fsum(term["frequency"] * term["probability"] for term in terms),
        "terms": terms,
    }


def exact_cut_sets(
    project: Project,
    target: str,
    probs: Mapping[str, float],
    overrides: Mapping[str, int | bool] | None = None,
    literals: str = "split",
    max_initiators: int | None = None,
) -> list[dict]:
    """Minimal cut sets of the hull. ``split``: the minimal cut sets of the plain
    hull over atoms, without those that contain an event and its complement
    atom (RFC Theorem 10(a)); ``omega``: MC_L(g) = PI(omega_L g), the minimal
    terms over L-literals that imply omega_L g (RFC §4.2); ``positive``:
    mcs(up g). Complement atoms and negative literals are written ``-NAME``;
    ``probability`` is the product of p (positive) and 1 - p (negative). Sorted
    by order, then names."""
    tt = exact_table(project, target, probs, "hull", overrides, literals, max_initiators)
    n = len(tt.names)
    fixed = [i for i, m in enumerate(tt.modes) if m == BOTH]
    if 3 ** len(fixed) * 2 ** (n - len(fixed)) > MAX_CUT_SET_TABLE:
        raise ModelError("too many events for exact cut sets")
    # Axis i is event i; a fixed axis gets a third state "absent" (both values).
    a = tt.table.reshape((2,) * n).transpose(tuple(range(n - 1, -1, -1))) if n else tt.table.reshape(())
    for i in fixed:
        a = np.concatenate([a, np.expand_dims(a.take(0, axis=i) & a.take(1, axis=i), i)], axis=i)
    # (S, beta) implies omega_L g iff a[beta, 1_S]; it is prime iff dropping any
    # positive literal or any fixed literal breaks that (implicants are up-closed).
    prime = a.copy()
    for i in range(n):
        if i in fixed:
            absent = ~a.take([2], axis=i)
            prime &= np.concatenate([absent, absent, np.ones_like(absent)], axis=i)
        else:
            dropped = ~a.take([0], axis=i)
            prime &= np.concatenate([np.ones_like(dropped), dropped], axis=i)
    out = []
    for idx in np.argwhere(prime):
        lits, p = [], 1.0
        for i, s in enumerate(idx.tolist()):
            if s == 1:
                lits.append(tt.names[i])
                p *= tt.probs[i]
            elif s == 0 and i in fixed:
                lits.append("-" + tt.names[i])
                p *= 1.0 - tt.probs[i]
        if any(name.startswith("-") and name[1:] in lits for name in lits):
            continue  # an event and its complement atom: probability 0
        out.append({"literals": sorted(lits, key=lambda s: s.lstrip("-")), "probability": p})
    return sorted(out, key=lambda cs: (len(cs["literals"]), cs["literals"]))
