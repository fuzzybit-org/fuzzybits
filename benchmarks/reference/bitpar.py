"""Bit-parallel evaluation: one bit per lane, 64 lanes per ``uint64`` word.

The gate combinators are generic over NumPy ``uint64`` arrays and Python
``int`` bit sets: both support ``&``, ``|`` and ``^``, and complement is
``x ^ ones``. In-place operators are applied only to freshly created values,
so child values are never modified.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .circuit import NEG, POS, Circuit, Plan
from .ftp import AND, OR

ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
# Below this probability an event is drawn as Binomial(n, p) distinct lane
# positions instead of n uniforms; both are exact Bernoulli(p) lanes.
SPARSE_P = 1.0 / 32.0


def combine(kind: str, k: int, xs: Sequence[Any], ones: Any, zeros: Any) -> Any:
    if kind == AND or kind == OR:
        acc = xs[0]
        if len(xs) > 1:
            if kind == AND:
                acc = acc & xs[1]
                for x in xs[2:]:
                    acc &= x
            else:
                acc = acc | xs[1]
                for x in xs[2:]:
                    acc |= x
        return acc
    return atleast(k, xs, ones, zeros)


def atleast(k: int, xs: Sequence[Any], ones: Any, zeros: Any) -> Any:
    n = len(xs)
    if k <= 0:
        return ones
    if k > n:
        return zeros
    if k == 1:
        return combine(OR, 0, xs, ones, zeros)
    if k == n:
        return combine(AND, 0, xs, ones, zeros)
    c = [ones] + [zeros] * k
    for i, x in enumerate(xs):
        # c[j] = at least j of xs[:i+1]; counts below k - (n - 1 - i) can no longer reach k.
        lo = max(1, k - (n - 1 - i))
        for j in range(min(k, i + 1), lo - 1, -1):
            c[j] = c[j] | (c[j - 1] & x)
    return c[k]


def popcount(words: np.ndarray) -> int:
    if hasattr(np, "bitwise_count"):
        return int(np.bitwise_count(words).sum(dtype=np.int64))
    return int(np.unpackbits(words.view(np.uint8)).sum(dtype=np.int64))


def lanes_set(words: np.ndarray, n_lanes: int) -> np.ndarray:
    """Indices of set lanes below ``n_lanes``."""
    bits = np.unpackbits(words.view(np.uint8), bitorder="little")[:n_lanes]
    return np.flatnonzero(bits)


def gather(words: np.ndarray, lanes: np.ndarray) -> np.ndarray:
    """Bits of ``words`` at ``lanes`` as a bool array."""
    return ((words[lanes >> 6] >> (lanes & 63).astype(np.uint64)) & np.uint64(1)).astype(bool)


def pack(bits: np.ndarray) -> np.ndarray:
    """Bool lanes to ``uint64`` words (padding lanes are 0)."""
    n_words = max(1, -(-len(bits) // 64))
    buf = np.zeros(n_words * 64, dtype=bool)
    buf[: len(bits)] = bits
    return np.packbits(buf, bitorder="little").view(np.uint64)


def to_int(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits, bitorder="little").tobytes(), "little")


def sample_event(rng: np.random.Generator, p: float, n_words: int) -> np.ndarray:
    n = n_words * 64
    if p >= 1.0:
        return np.full(n_words, ALL, dtype=np.uint64)
    if p <= 0.0:
        return np.zeros(n_words, dtype=np.uint64)
    if p < SPARSE_P:
        hits = int(rng.binomial(n, p))
        out = np.zeros(n_words, dtype=np.uint64)
        if hits:
            idx = rng.choice(n, size=hits, replace=False, shuffle=False)
            np.bitwise_or.at(out, idx >> 6, np.left_shift(np.uint64(1), (idx & 63).astype(np.uint64)))
        return out
    return np.packbits(rng.random(n) < p, bitorder="little").view(np.uint64)


def evaluate(
    c: Circuit,
    pl: Plan,
    leaf: Callable[[int], Any],
    ones: Any,
    zeros: Any,
    coherent: list | None = None,
) -> tuple[Any, Any]:
    """Root value of ``c`` (and of its coherent bound when ``coherent`` is
    given), freeing values after their last use. ``leaf(v)`` produces event
    ``v`` and is called in event order. ``coherent[g]`` is ``None`` when gate
    ``g`` has no complement below it, ``1`` when its coherent version is TRUE,
    or ``(kind, k, positive literals)``."""
    n = c.n_nodes
    vals: list[Any] = [None] * n
    vals[0] = zeros
    cvals = vals if coherent is None else [None] * n
    if coherent is not None:
        cvals[0] = zeros
    first = c.first_gate
    if not c.kinds:
        v = c.root >> 1
        if v:
            vals[v] = leaf(v)
        x = vals[v] ^ ones if c.root & 1 else vals[v]
        return x, (x if coherent is None else (ones if c.root & 1 else x))
    for g, kids in enumerate(c.kids):
        for v in pl.new_vars[g]:
            vals[v] = leaf(v)
            if coherent is not None:
                cvals[v] = vals[v]
        xs = [vals[lit >> 1] ^ ones if lit & 1 else vals[lit >> 1] for lit in kids]
        node = first + g
        vals[node] = combine(c.kinds[g], c.ks[g], xs, ones, zeros)
        if coherent is not None:
            spec = coherent[g]
            if spec is None:
                cvals[node] = vals[node]
            elif spec == 1:
                cvals[node] = ones
            else:
                kind, k, lits = spec
                cvals[node] = combine(kind, k, [cvals[lit >> 1] for lit in lits], ones, zeros)
        for f in pl.free[g]:
            vals[f] = None
            cvals[f] = None
    r = c.root >> 1
    out = vals[r] ^ ones if c.root & 1 else vals[r]
    if coherent is None:
        return out, out
    return out, (ones if c.root & 1 else cvals[r])


def coherent_specs(c: Circuit, neg_below: list[bool]) -> list:
    """Coherent bound g_coh: every complemented input replaced by TRUE."""
    specs: list = []
    for g, kids in enumerate(c.kids):
        if not neg_below[g]:
            specs.append(None)
            continue
        kind, k = c.kinds[g], c.ks[g]
        pos = [lit for lit in kids if not lit & 1]
        n_true = len(kids) - len(pos)
        if kind == OR:
            specs.append(1 if n_true else (OR, 0, tuple(pos)))
        elif kind == AND:
            specs.append((AND, 0, tuple(pos)) if pos else 1)
        else:
            k2 = k - n_true
            specs.append(1 if k2 <= 0 else (kind, k2, tuple(pos)))
    return specs


def rails(
    c: Circuit,
    pol: list[int],
    x: Callable[[int], Any],
    ones: Any,
    zeros: Any,
    pl: Plan | None = None,
) -> tuple[list[Any], list[Any]]:
    """Kleene (three-valued) evaluation in dual rail: ``T[n]`` = definitely 1,
    ``F[n]`` = definitely 0, neither = undetermined.

    Event leaves encode the hull check of a lane ``x``: events false in ``x``
    are 0; true events reachable only positively are 1 and only negatively 0
    (the optimal choice for satisfiability); true binate events are free.
    With ``pl`` values are freed after their last use (only the root rails
    remain); without it every node keeps its rails.
    """
    n = c.n_nodes
    T: list[Any] = [None] * n
    F: list[Any] = [None] * n
    T[0], F[0] = zeros, ones
    first = c.first_gate

    def put_leaf(v: int) -> None:
        p = pol[v]
        if p == NEG:
            T[v], F[v] = zeros, ones
            return
        xv = x(v)
        T[v] = xv if p == POS else zeros
        F[v] = xv ^ ones

    if pl is None:
        for v in range(1, first):
            put_leaf(v)
    for g, kids in enumerate(c.kids):
        if pl is not None:
            for v in pl.new_vars[g]:
                put_leaf(v)
        ts = [F[lit >> 1] if lit & 1 else T[lit >> 1] for lit in kids]
        fs = [T[lit >> 1] if lit & 1 else F[lit >> 1] for lit in kids]
        kind, k = c.kinds[g], c.ks[g]
        node = first + g
        if kind == AND:
            T[node] = combine(AND, 0, ts, ones, zeros)
            F[node] = combine(OR, 0, fs, ones, zeros)
        elif kind == OR:
            T[node] = combine(OR, 0, ts, ones, zeros)
            F[node] = combine(AND, 0, fs, ones, zeros)
        else:
            T[node] = atleast(k, ts, ones, zeros)
            F[node] = atleast(len(kids) - k + 1, fs, ones, zeros)
        if pl is not None:
            for f in pl.free[g]:
                T[f] = F[f] = None
    if not c.kinds and pl is not None and c.root >> 1:
        put_leaf(c.root >> 1)
    return T, F
