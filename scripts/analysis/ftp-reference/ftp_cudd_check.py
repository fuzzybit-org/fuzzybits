#!/usr/bin/env python3
"""Check a ftp_cudd.py BDD dump against direct evaluation of the FTP fault tree on random assignments.

Usage: python3 -I ftp_cudd_check.py MODEL.ftp WORKDIR TARGET [--samples 20000] [--seed 1]

The .net file in WORKDIR gives the variable indices; the .bdd dump gives the BDD. Assignments are
drawn with a per-sample bias q ~ U(0.3, 0.995) so both outputs occur, then every positive sample is
also probed with each single-variable flip (the boundary of the function, where mistakes show).
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ftp_cudd import parse_ftp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ftp")
    ap.add_argument("workdir")
    ap.add_argument("target")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--flip-probes", type=int, default=200, help="positive samples whose 1-flips are probed")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    gates, order, imports, process = parse_ftp(a.ftp)
    stem = os.path.splitext(os.path.basename(a.ftp))[0]
    with open(os.path.join(a.workdir, stem + ".net")) as f:
        nv = int(f.readline().split()[1])
        vorder = [f.readline().split()[0] for _ in range(nv)]
    vidx = {v: i for i, v in enumerate(vorder)}
    with open(os.path.join(a.workdir, f"{stem}.{a.target}.bdd")) as f:
        f.readline()
        f.readline()
        hdr = f.readline().split()
        nn, root_e = int(hdr[1]), int(hdr[3])
        var = [0] * (nn + 1)
        th = [0] * (nn + 1)
        el = [0] * (nn + 1)
        for n in range(1, nn + 1):
            var[n], th[n], el[n] = map(int, f.readline().split())

    def bdd(x):
        e = root_e
        neg = 0
        while e >> 1:
            n = e >> 1
            neg ^= e & 1
            e = th[n] if x[var[n]] else el[n]
        return 1 ^ neg ^ (e & 1)  # constant ONE, complemented if parity odd

    # topological gate order for the target cone
    topo, seen = [], set()

    def visit(g):
        stack = [(g, iter(gates[g][1]))]
        seen.add(g)
        while stack:
            name, it = stack[-1]
            for op in it:
                b = op.lstrip("-")
                if b in gates and b not in seen:
                    seen.add(b)
                    stack.append((b, iter(gates[b][1])))
                    break
            else:
                stack.pop()
                topo.append(name)

    visit(a.target)

    def tree(x):
        val = {}
        for g in topo:
            kind, ops = gates[g]
            vs = []
            for op in ops:
                b = op.lstrip("-")
                v = val[b] if b in gates else x[vidx[b]]
                vs.append(1 - v if op.startswith("-") else v)
            if kind == "+":
                val[g] = int(any(vs))
            elif kind == "*":
                val[g] = int(all(vs))
            else:
                val[g] = int(sum(vs) >= int(kind))
        return val[a.target]

    rng = random.Random(a.seed)
    pos = mism = 0
    positives = []
    for _ in range(a.samples):
        q = rng.uniform(0.3, 0.995)
        x = [1 if rng.random() < q else 0 for _ in range(nv)]
        t, b = tree(x), bdd(x)
        if t != b:
            mism += 1
        if t:
            pos += 1
            if len(positives) < a.flip_probes:
                positives.append(x)
    # shrink each positive to a minimal true point (a cut set of the tree), then probe all 1-flips:
    # every removal must give 0 and every addition 1 (monotone case), and the BDD must agree
    minimal = []
    for x in positives:
        y = list(x)
        idx = list(range(nv))
        rng.shuffle(idx)
        for i in idx:
            if y[i]:
                y[i] = 0
                if not tree(y):
                    y[i] = 1
        minimal.append(y)
    cut_sizes = [sum(y) for y in minimal]
    flips = fmism = fneg = 0
    for x in positives + minimal:
        for i in range(nv):
            y = list(x)
            y[i] ^= 1
            t, b = tree(y), bdd(y)
            flips += 1
            fneg += 1 - t
            fmism += t != b
    print(f"random: {a.samples} samples, {pos} positive, {mism} mismatches; "
          f"single flips of {len(positives)} positives and {len(minimal)} minimal true points "
          f"(cut-set sizes {min(cut_sizes, default=0)}..{max(cut_sizes, default=0)}): {flips} probes, {fneg} negative, "
          f"{fmism} mismatches")
    sys.exit(1 if mism or fmism else 0)


if __name__ == "__main__":
    main()
