"""Monte Carlo estimator of the monotone hull P(up g) (#174).

``up g (x) = exists y <= x : g(y)``. It lies between ``g`` and ``g_coh`` (every
complemented input replaced by TRUE), so per pass:

1. Sample events bit-parallel and evaluate ``g`` and ``g_coh`` together.
   Lanes with ``g = 1`` are in the hull; lanes with ``g_coh = 0`` are not.
2. On the remaining candidates, redraw the pass, gather their lanes and run a
   Kleene (three-valued) evaluation in dual rail: events false in the lane are
   0, true events reachable only positively are 1 and only negatively 0 (the
   optimal choice, so p = 1 flags under a complement are dropped), true binate
   events are free. Lanes whose root is decided are counted directly.
3. Undecided lanes (up to ``max_checks``) get an exact check: the residual
   circuit of undetermined nodes is reduced by repeated pure-literal fixing and
   decided by a truth table (<= 16 free events), a budgeted BDD, or a complete
   branching search.

With ``n_T`` lanes decided true, ``n_C`` undecided lanes of which ``m`` were
checked and ``h`` found true, the estimate ``(n_T + n_C h / m) / N`` is unbiased
and its variance is ``[pi_T + f^2 pi_C - P^2] / N + pi_C^2 f (1 - f) / m``
(law of total variance; equals ``P (1 - P) / N`` when every undecided lane is
checked). Weights are unaffected by importance sampling because ``up g`` is a
function of the lane.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Mapping

import numpy as np

from .bdd import BudgetExceeded, circuit_bdd
from .bitpar import (
    ALL,
    coherent_specs,
    combine,
    evaluate,
    gather,
    lanes_set,
    pack,
    popcount,
    rails,
    sample_event,
    to_int,
)
from .circuit import NEG, POS, Circuit, build_circuit, negation_below, plan, polarity
from .ftp import AND, ATLEAST, OR, Project
from .montecarlo import DEFAULT_MEMORY_MIB, interval, pass_words, run_passes

TRUTH_TABLE_MAX = 16
DEFAULT_BDD_BUDGET = 1 << 16
CHECK_BATCH = 2048  # lanes per exact-check batch (bounds the int bit sets)


def extract(c: Circuit, root: int, value) -> int | Circuit:
    """Residual of ``c`` below literal ``root`` given ``value(node)`` in
    {0, 1, None}: undetermined nodes reachable through undetermined nodes,
    with determined inputs absorbed. Returns a constant or a circuit."""
    r = root >> 1
    v = value(r)
    if v is not None:
        return v ^ (root & 1)
    first = c.first_gate
    if r < first:
        return Circuit([c.names[0], c.names[r]], 1, [0.0, c.probs[r]], [], [], [], 2 + (root & 1))
    all_kids, kinds, ks = c.kids, c.kinds, c.ks
    info: dict[int, tuple[list[int], int]] = {}  # gate -> (undetermined literals, threshold)
    free: list[int] = []
    seen = {r}
    stack = [r]
    while stack:
        node = stack.pop()
        g = node - first
        kind = kinds[g]
        k = ks[g]
        lits = []
        for lit in all_kids[g]:
            ch = lit >> 1
            val = value(ch)
            if val is None:
                lits.append(lit)
                if ch not in seen:
                    seen.add(ch)
                    if ch >= first:
                        stack.append(ch)
                    else:
                        free.append(ch)
            elif kind == ATLEAST and val ^ (lit & 1):
                k -= 1
        info[node] = (lits, k)
    order = sorted(info)  # gate indices are topological
    idx = {n: i + 1 for i, n in enumerate(free)}
    for j, node in enumerate(order):
        idx[node] = 1 + len(free) + j
    out_kinds, out_ks, out_kids = [], [], []
    for node in order:
        lits, k = info[node]
        kind = kinds[node - first]
        if kind == ATLEAST:
            kind = OR if k == 1 else AND if k == len(lits) else ATLEAST
        out_kinds.append(kind)
        out_ks.append(k if kind == ATLEAST else 0)
        out_kids.append(tuple(2 * idx[lit >> 1] + (lit & 1) for lit in lits))
    names = [c.names[0]] + [c.names[n] for n in free] + [c.names[n] for n in order]
    probs = [0.0] + [c.probs[n] for n in free] + [0.0] * len(order)
    return Circuit(names, len(free), probs, out_kinds, out_ks, out_kids, 2 * idx[r] + (root & 1))


def _lane_value(T: list, D: list, j: int):
    """Value of a node in lane ``j``: 1, 0, or None (undetermined)."""

    def value(n: int):
        if not (D[n] >> j) & 1:
            return None
        return (T[n] >> j) & 1

    return value


def reduce_pure(res: Circuit) -> int | Circuit:
    """Fix events that occur only positively (to 1) or only negatively (to 0)
    in the residual and propagate, until none is left."""
    while True:
        pol = polarity(res)
        if all(pol[v] == POS | NEG for v in range(1, res.first_gate)):
            return res
        T, F = rails(res, pol, lambda v: 1, 1, 0)
        vals = [1 if t else 0 if f else None for t, f in zip(T, F, strict=True)]
        out = extract(res, res.root, vals.__getitem__)
        if isinstance(out, int):
            return out
        res = out


def kleene(c: Circuit, assign: Mapping[int, int]) -> list:
    val: list = [None] * c.n_nodes
    val[0] = 0
    for v, b in assign.items():
        val[v] = b
    first = c.first_gate
    for g, kids in enumerate(c.kids):
        ones = unknown = 0
        for lit in kids:
            x = val[lit >> 1]
            if x is None:
                unknown += 1
            else:
                ones += x ^ (lit & 1)
        n = len(kids)
        kind = c.kinds[g]
        k = n if kind == AND else 1 if kind == OR else c.ks[g]
        val[first + g] = 1 if ones >= k else 0 if ones + unknown < k else None
    return val


def truth_table_sat(res: Circuit) -> bool:
    m = res.n_vars
    size = 1 << m
    mask = (1 << size) - 1
    vals = [0] * res.n_nodes
    for i in range(m):
        block = 1 << i
        unit = ((1 << block) - 1) << block
        vals[i + 1] = unit * (mask // ((1 << (2 * block)) - 1))
    first = res.first_gate
    for g, kids in enumerate(res.kids):
        xs = [vals[lit >> 1] ^ mask if lit & 1 else vals[lit >> 1] for lit in kids]
        vals[first + g] = combine(res.kinds[g], res.ks[g], xs, mask, 0)
    root = vals[res.root >> 1] ^ (mask if res.root & 1 else 0)
    return root != 0


def search_sat(res: Circuit) -> bool:
    """Complete branching search with pure-literal reduction at every node."""
    stack: list[Circuit] = [res]
    while stack:
        cur = reduce_pure(stack.pop())
        if isinstance(cur, int):
            if cur:
                return True
            continue
        branches = []
        for b in (1, 0):
            nxt = extract(cur, cur.root, kleene(cur, {1: b}).__getitem__)
            if isinstance(nxt, int):
                if nxt:
                    return True
                continue
            branches.append(nxt)
        stack.extend(reversed(branches))
    return False


class LaneChecker:
    """Exact hull check for lanes left undecided by the Kleene stage."""

    def __init__(self, c: Circuit, pol: list[int], bdd_budget: int = DEFAULT_BDD_BUDGET):
        self.c = c
        self.pol = pol
        self.bdd_budget = bdd_budget
        self.methods = {"pure-literal": 0, "truth-table": 0, "bdd": 0, "search": 0}
        self.free_vars: list[int] = []

    def check(self, x: list[int], m: int) -> list[bool]:
        """``x[v]`` holds event ``v`` for ``m`` lanes as an int bit set."""
        mask = (1 << m) - 1
        T, F = rails(self.c, self.pol, x.__getitem__, mask, 0)
        D = [t | f for t, f in zip(T, F, strict=True)]
        return [self.decide(extract(self.c, self.c.root, _lane_value(T, D, j))) for j in range(m)]

    def decide(self, res: int | Circuit) -> bool:
        if isinstance(res, int):
            self.methods["pure-literal"] += 1
            return bool(res)
        self.free_vars.append(res.n_vars)
        red = reduce_pure(res)
        if isinstance(red, int):
            self.methods["pure-literal"] += 1
            return bool(red)
        if red.n_vars <= TRUTH_TABLE_MAX:
            self.methods["truth-table"] += 1
            return truth_table_sat(red)
        try:
            bdd, root = circuit_bdd(red, self.bdd_budget)
            self.methods["bdd"] += 1
            return root != 0
        except (BudgetExceeded, RecursionError):
            self.methods["search"] += 1
            return search_sat(red)


def estimate_hull(
    project: Project,
    target: str,
    probs: Mapping[str, float],
    overrides: Mapping[str, int | bool] | None = None,
    trials: int = 1 << 20,
    seed: int = 0,
    max_checks: int | None = None,
    bdd_budget: int = DEFAULT_BDD_BUDGET,
    memory_mib: int = DEFAULT_MEMORY_MIB,
) -> dict:
    t0 = time.perf_counter()
    c = build_circuit(project, target, probs, overrides, constants="hull", fold=True)
    n = 64 * -(-trials // 64)
    base = {"target": target, "semantics": "hull", "seed": seed}
    const = c.constant()
    if const is not None:
        v = float(const)
        return {**base, "method": "constant-folding", "estimate": v, "se": 0.0, "ci95": [v, v], "trials": n,
                "bounds": {"lower": v, "upper": v}, "seconds": time.perf_counter() - t0}
    pl = plan(c)
    pol = polarity(c)
    coh = coherent_specs(c, negation_below(c))
    checker = LaneChecker(c, pol, bdd_budget)
    words = pass_words(pl, memory_mib, arrays_per_node=2)
    rng = np.random.Generator(np.random.PCG64(seed))
    n_g = n_coh = n_kt = n_kf = n_und = checked = checked_true = 0
    t_check = 0.0
    budget = math.inf if max_checks is None else max_checks
    for w, state, ones, zeros, sampler in run_passes(c, pl, n, rng, words):
        lanes = 64 * w
        g, gc = evaluate(c, pl, sampler, ones, zeros, coherent=coh)
        n_g += popcount(g)
        n_coh += popcount(gc)
        cand = lanes_set(gc & (g ^ ones), lanes)
        if not len(cand):
            continue
        redraw = np.random.Generator(np.random.PCG64())
        redraw.bit_generator.state = state
        xs: list = [None] + [
            pack(gather(sample_event(redraw, c.probs[v], w), cand)) for v in range(1, c.first_gate)
        ]
        cw = len(xs[1]) if len(xs) > 1 else 1
        cones = np.full(cw, ALL, dtype=np.uint64)
        czeros = np.zeros(cw, dtype=np.uint64)
        T, F = rails(c, pol, xs.__getitem__, cones, czeros, pl)
        r = c.root >> 1
        tr, fr = (F[r], T[r]) if c.root & 1 else (T[r], F[r])
        m = len(cand)
        n_kt += len(lanes_set(tr, m))
        n_kf += len(lanes_set(fr, m))
        und = lanes_set((tr | fr) ^ cones, m)
        n_und += len(und)
        take = und[: int(min(len(und), budget - checked))]
        for lo in range(0, len(take), CHECK_BATCH):
            batch = take[lo : lo + CHECK_BATCH]
            t1 = time.perf_counter()
            x = [0] + [to_int(gather(xs[v], batch)) for v in range(1, c.first_gate)]
            res = checker.check(x, len(batch))
            t_check += time.perf_counter() - t1
            checked += len(batch)
            checked_true += sum(res)
    pi_t = (n_g + n_kt) / n
    pi_c = n_und / n
    lower = n_g / n
    upper = n_coh / n
    if n_und and not checked:
        est = se = math.nan
    else:
        f = checked_true / checked if checked else 0.0
        est = pi_t + pi_c * f
        var = max(pi_t + f * f * pi_c - est * est, 0.0) / n
        if checked:
            var += pi_c * pi_c * f * (1.0 - f) / checked
        se = math.sqrt(var)
    free = checker.free_vars
    return {
        **base,
        "method": "hull-estimator",
        "estimate": est,
        "se": se,
        "ci95": interval(est, se),
        "trials": n,
        "bounds": {
            "lower": lower,
            "lower_se": math.sqrt(lower * (1 - lower) / n),
            "upper": upper,
            "upper_se": math.sqrt(upper * (1 - upper) / n),
        },
        "lanes": {
            "boolean_true": n_g,
            "coherent_true": n_coh,
            "kleene_true": n_kt,
            "kleene_false": n_kf,
            "undecided": n_und,
            "checked": checked,
            "checked_true": checked_true,
        },
        "checked_fraction_of_coherent": checked / n_coh if n_coh else 0.0,
        "check_seconds": t_check,
        "lanes_per_second": checked / t_check if t_check > 0 else None,
        "residual_free_events": {
            "median": statistics.median(free) if free else 0,
            "max": max(free, default=0),
        },
        "check_methods": dict(checker.methods),
        "gates": len(c.kinds),
        "events": c.n_vars,
        "pass_lanes": 64 * words,
        "seconds": time.perf_counter() - t0,
    }
