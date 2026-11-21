"""Small reduced ordered BDD with a node budget (no complement edges)."""

from __future__ import annotations

import sys
from typing import Sequence

from .circuit import Circuit
from .ftp import AND, OR


class BudgetExceeded(Exception):
    pass


class BDD:
    """Node 0 is FALSE, node 1 is TRUE; variable order is the variable index."""

    __slots__ = ("var", "lo", "hi", "unique", "budget", "memo")

    def __init__(self, budget: int):
        self.var = [sys.maxsize, sys.maxsize]
        self.lo = [0, 1]
        self.hi = [0, 1]
        self.unique: dict[tuple[int, int, int], int] = {}
        self.budget = budget
        self.memo: dict[tuple, int] = {}

    def __len__(self) -> int:
        return len(self.var)

    def mk(self, v: int, lo: int, hi: int) -> int:
        if lo == hi:
            return lo
        key = (v, lo, hi)
        node = self.unique.get(key)
        if node is None:
            if len(self.var) >= self.budget:
                raise BudgetExceeded
            node = len(self.var)
            self.var.append(v)
            self.lo.append(lo)
            self.hi.append(hi)
            self.unique[key] = node
        return node

    def neg(self, a: int) -> int:
        if a < 2:
            return 1 - a
        key = ("n", a)
        r = self.memo.get(key)
        if r is None:
            r = self.memo[key] = self.mk(self.var[a], self.neg(self.lo[a]), self.neg(self.hi[a]))
        return r

    def apply(self, op: str, a: int, b: int) -> int:
        if op == AND:
            if a == 0 or b == 0:
                return 0
            if a == 1:
                return b
            if b == 1 or a == b:
                return a
        else:
            if a == 1 or b == 1:
                return 1
            if a == 0:
                return b
            if b == 0 or a == b:
                return a
        if a > b:
            a, b = b, a
        key = (op, a, b)
        r = self.memo.get(key)
        if r is not None:
            return r
        va, vb = self.var[a], self.var[b]
        if va == vb:
            r = self.mk(va, self.apply(op, self.lo[a], self.lo[b]), self.apply(op, self.hi[a], self.hi[b]))
        elif va < vb:
            r = self.mk(va, self.apply(op, self.lo[a], b), self.apply(op, self.hi[a], b))
        else:
            r = self.mk(vb, self.apply(op, a, self.lo[b]), self.apply(op, a, self.hi[b]))
        self.memo[key] = r
        return r

    def fold(self, op: str, xs: Sequence[int]) -> int:
        acc = 1 if op == AND else 0
        for x in xs:
            acc = self.apply(op, acc, x)
        return acc

    def atleast(self, k: int, xs: Sequence[int]) -> int:
        c = [1] + [0] * k
        for x in xs:
            for j in range(k, 0, -1):
                c[j] = self.apply(OR, c[j], self.apply(AND, c[j - 1], x))
        return c[k]


def circuit_bdd(c: Circuit, budget: int) -> tuple[BDD, int]:
    """BDD of the root of ``c``; raises :class:`BudgetExceeded`."""
    bdd = BDD(budget)
    node = [0] * c.n_nodes
    for v in range(1, c.first_gate):
        node[v] = bdd.mk(v, 0, 1)
    first = c.first_gate
    for g, kids in enumerate(c.kids):
        xs = [bdd.neg(node[lit >> 1]) if lit & 1 else node[lit >> 1] for lit in kids]
        kind = c.kinds[g]
        if kind in (AND, OR):
            node[first + g] = bdd.fold(kind, xs)
        else:
            node[first + g] = bdd.atleast(c.ks[g], xs)
    root = node[c.root >> 1]
    return bdd, bdd.neg(root) if c.root & 1 else root
