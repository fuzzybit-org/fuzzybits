"""Exact Boolean and monotone-hull probabilities by enumeration.

Implements the definitions directly on the unfolded circuit: every event in
the cone is a variable weighted by its import value (including 0 and 1), and

    P(g)      = sum_x g(x) w(x)
    P(up g)   = sum_x [exists y <= x : g(y)] w(x)

where the upward closure is a subset-OR (zeta) transform of the truth table.
Nothing here relies on the folding, polarity or Kleene steps used by the
estimators, so the estimators are checked against an independent path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .bitpar import ALL, evaluate
from .circuit import Circuit, build_circuit, plan
from .ftp import ModelError, Project

MAX_EXACT_VARIABLES = 22


@dataclass
class TruthTable:
    names: list[str]  # variable i is bit i of the assignment index
    probs: list[float]
    table: np.ndarray  # bool, length 2**n


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
    return TruthTable(c.names[1 : 1 + n], c.probs[1 : 1 + n], bits.astype(bool))


def upward_closure(table: np.ndarray, n: int) -> np.ndarray:
    """up[x] = OR of table[y] over y subset of x."""
    up = table.copy()
    for i in range(n):
        view = up.reshape(-1, 2, 1 << i)
        view[:, 1, :] |= view[:, 0, :]
    return up


def weights(probs: list[float]) -> np.ndarray:
    w = np.ones(1, dtype=np.float64)
    for p in probs:
        w = np.concatenate([w * (1.0 - p), w * p])
    return w


def exact_table(
    project: Project,
    target: str,
    probs: Mapping[str, float],
    semantics: str,
    overrides: Mapping[str, int | bool] | None = None,
) -> TruthTable:
    """Truth table of g (``boolean``) or of up(g) (``hull``) over the cone's events."""
    c = build_circuit(project, target, probs, overrides, constants="none", fold=False)
    tt = truth_table(c)
    if semantics == "hull":
        tt.table = upward_closure(tt.table, len(tt.names))
    elif semantics != "boolean":
        raise ValueError("semantics must be 'boolean' or 'hull'")
    return tt


def exact_probability(
    project: Project,
    target: str,
    probs: Mapping[str, float],
    semantics: str,
    overrides: Mapping[str, int | bool] | None = None,
) -> dict:
    tt = exact_table(project, target, probs, semantics, overrides)
    w = weights(tt.probs)
    p = math.fsum(w[tt.table].tolist())
    return {
        "target": target,
        "semantics": semantics,
        "method": "enumeration",
        "probability": min(1.0, max(0.0, p)),
        "variables": len(tt.names),
    }
