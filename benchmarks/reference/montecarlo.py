"""Bit-parallel Boolean Monte Carlo for P(g) (NumPy, PCG64)."""

from __future__ import annotations

import math
import time
from typing import Mapping

import numpy as np

from .bitpar import ALL, atom_sampler, evaluate, popcount
from .circuit import Circuit, Plan, build_circuit, plan
from .ftp import Project

MAX_PASS_WORDS = 1 << 14  # 2^20 lanes per pass
DEFAULT_MEMORY_MIB = 256


def pass_words(pl: Plan, memory_mib: int, arrays_per_node: int = 1) -> int:
    """Largest power-of-two pass (in words) whose live arrays fit the budget."""
    budget = memory_mib * (1 << 20)
    words = budget // (8 * pl.peak_live * arrays_per_node)
    w = 64
    while w * 2 <= min(words, MAX_PASS_WORDS):
        w *= 2
    return w


def interval(est: float, se: float) -> list[float]:
    return [est - 1.96 * se, est + 1.96 * se]


def run_passes(c: Circuit, pl: Plan, trials: int, rng: np.random.Generator, words: int):
    """Yield ``(n_words, state, ones, zeros, sampler)`` per pass; ``state`` is
    the generator state before the pass, so the pass can be redrawn."""
    done = 0
    while done < trials:
        w = min(words, -(-(trials - done) // 64))
        state = rng.bit_generator.state
        ones = np.full(w, ALL, dtype=np.uint64)
        zeros = np.zeros(w, dtype=np.uint64)
        yield w, state, ones, zeros, atom_sampler(c, rng, w, ones)
        done += 64 * w


def estimate_boolean(
    project: Project,
    target: str,
    probs: Mapping[str, float],
    overrides: Mapping[str, int | bool] | None = None,
    trials: int = 1 << 20,
    seed: int = 0,
    memory_mib: int = DEFAULT_MEMORY_MIB,
    max_initiators: int | None = None,
) -> dict:
    t0 = time.perf_counter()
    c = build_circuit(project, target, probs, overrides, constants="boolean", fold=True, max_initiators=max_initiators)
    n = 64 * -(-trials // 64)
    base = {"target": target, "semantics": "boolean", "max_initiators_per_cutset": max_initiators, "seed": seed}
    const = c.constant()
    if const is not None:
        return {
            **base,
            "method": "constant-folding",
            "estimate": float(const),
            "se": 0.0,
            "ci95": [float(const), float(const)],
            "trials": n,
            "hits": const * n,
            "seconds": time.perf_counter() - t0,
        }
    pl = plan(c)
    words = pass_words(pl, memory_mib)
    rng = np.random.Generator(np.random.PCG64(seed))
    hits = 0
    t1 = time.perf_counter()
    for _w, _state, ones, zeros, sampler in run_passes(c, pl, n, rng, words):
        root, _ = evaluate(c, pl, sampler, ones, zeros)
        hits += popcount(root)
    dt = time.perf_counter() - t1
    est = hits / n
    se = math.sqrt(est * (1.0 - est) / n)
    return {
        **base,
        "method": "monte-carlo",
        "estimate": est,
        "se": se,
        "ci95": interval(est, se),
        "trials": n,
        "hits": hits,
        "gates": len(c.kinds),
        "events": c.n_vars,
        "pass_lanes": 64 * words,
        "seconds": time.perf_counter() - t0,
        "trials_per_second": n / dt if dt > 0 else None,
    }
