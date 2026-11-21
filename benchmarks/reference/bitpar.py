"""Bit-parallel evaluation: one bit per lane, 64 lanes per ``uint64`` word.

The gate combinators are generic over NumPy ``uint64`` arrays and Python
``int`` bit sets: both support ``&``, ``|`` and ``^``, and complement is
``x ^ ones``. In-place operators are applied only to freshly created values,
so child values are never modified.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .circuit import NEG, ONLY_POS, POS, Circuit, Plan, bounds, relaxed, relaxed_below
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


def atom_sampler(c: Circuit, rng: np.random.Generator, n_words: int, ones: np.ndarray) -> Callable[[int], np.ndarray]:
    """Lane values of the variables of ``c``, to be requested in increasing
    order. A complement atom and its event (``c.twins``) are coupled: the event
    is drawn once, at the first of the two, and the other gets its complement."""
    partner = {**c.twins, **{e: a for a, e in c.twins.items()}}
    drawn: dict[int, np.ndarray] = {}

    def sample(v: int) -> np.ndarray:
        u = partner.get(v)
        if u is None:
            return sample_event(rng, c.probs[v], n_words)
        if u in drawn:
            return drawn.pop(u) ^ ones
        drawn[v] = sample_event(rng, c.probs[v], n_words)
        return drawn[v]

    return sample


def evaluate(
    c: Circuit,
    pl: Plan,
    leaf: Callable[[int], Any],
    ones: Any,
    zeros: Any,
    coherent: tuple[list, bool] | None = None,
) -> tuple[Any, Any]:
    """Root value of ``c`` (and of its coherent bound when ``coherent`` is
    given), freeing values after their last use. ``leaf(v)`` produces event
    ``v`` and is called in event order. ``coherent = (specs, relax_root)`` from
    :func:`coherent_specs`: ``specs[g]`` is ``None`` when gate ``g`` has no
    relaxed literal below it, ``1`` when its coherent version is TRUE, or
    ``(kind, k, kept literals)``."""
    n = c.n_nodes
    vals: list[Any] = [None] * n
    vals[0] = zeros
    cvals = vals if coherent is None else [None] * n
    specs, relax_root = coherent if coherent is not None else (None, False)
    if coherent is not None:
        cvals[0] = zeros
    first = c.first_gate
    if not c.kinds:
        v = c.root >> 1
        if v:
            vals[v] = leaf(v)
        x = vals[v] ^ ones if c.root & 1 else vals[v]
        return x, (ones if relax_root else x)
    for g, kids in enumerate(c.kids):
        for v in pl.new_vars[g]:
            vals[v] = leaf(v)
            if coherent is not None:
                cvals[v] = vals[v]
        xs = [vals[lit >> 1] ^ ones if lit & 1 else vals[lit >> 1] for lit in kids]
        node = first + g
        vals[node] = combine(c.kinds[g], c.ks[g], xs, ones, zeros)
        if specs is not None:
            spec = specs[g]
            if spec is None:
                cvals[node] = vals[node]
            elif spec == 1:
                cvals[node] = ones
            else:
                kind, k, lits = spec
                xs = [cvals[lit >> 1] ^ ones if lit & 1 else cvals[lit >> 1] for lit in lits]
                cvals[node] = combine(kind, k, xs, ones, zeros)
        for f in pl.free[g]:
            vals[f] = None
            cvals[f] = None
    r = c.root >> 1
    out = vals[r] ^ ones if c.root & 1 else vals[r]
    if coherent is None:
        return out, out
    return out, (ones if relax_root else cvals[r] ^ ones if c.root & 1 else cvals[r])


def coherent_specs(c: Circuit, modes: list) -> tuple[list, bool]:
    """Coherent bound g_coh^L (RFC §4.4): complemented gate inputs and event
    literals not in L replaced by TRUE; literals of L kept. With L the positive
    literals this is g_coh (every complemented input TRUE)."""
    below = relaxed_below(c, modes)
    specs: list = []
    for g, kids in enumerate(c.kids):
        if not below[g]:
            specs.append(None)
            continue
        kind, k = c.kinds[g], c.ks[g]
        kept = tuple(lit for lit in kids if not relaxed(c, modes, lit))
        n_true = len(kids) - len(kept)
        if kind == OR:
            specs.append(1 if n_true else (OR, 0, kept))
        elif kind == AND:
            specs.append((AND, 0, kept) if kept else 1)
        else:
            k2 = k - n_true
            specs.append(1 if k2 <= 0 else (kind, k2, kept))
    return specs, relaxed(c, modes, c.root)


def rails(
    c: Circuit,
    pol: list[int],
    x: Callable[[int], Any],
    ones: Any,
    zeros: Any,
    pl: Plan | None = None,
    modes: list | None = None,
) -> tuple[list[Any], list[Any]]:
    """Kleene (three-valued) evaluation in dual rail: ``T[n]`` = definitely 1,
    ``F[n]`` = definitely 0, neither = undetermined.

    Event leaves encode the omega_L check of a lane ``x`` (RFC §6.2): a witness
    may give event ``v`` any value in ``bounds(modes[v], x_v)`` (default
    ONLY_POS, the plain hull: events false in ``x`` are 0, true events free).
    An event with one allowed value is fixed to it; a free event reachable only
    positively is 1 and only negatively 0 (the optimal choice for
    satisfiability, which drops p = 1 flags under a complement); free binate
    events are undetermined. With ``pl`` values are freed after their last use
    (only the root rails remain); without it every node keeps its rails.
    """
    n = c.n_nodes
    T: list[Any] = [None] * n
    F: list[Any] = [None] * n
    T[0], F[0] = zeros, ones
    first = c.first_gate

    def put_leaf(v: int) -> None:
        p = pol[v]
        mode = ONLY_POS if modes is None else modes[v]
        if mode == ONLY_POS and p == NEG:
            T[v], F[v] = zeros, ones
            return
        lo, hi = bounds(mode, x(v), ones, zeros)
        if p == POS:
            T[v], F[v] = hi, hi ^ ones
        elif p == NEG:
            T[v], F[v] = lo, lo ^ ones
        else:
            T[v], F[v] = lo, hi ^ ones

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
