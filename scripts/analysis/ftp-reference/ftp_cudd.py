#!/usr/bin/env python3
"""Exact top-event probability of a CAFTA FTP fault tree via a CUDD BDD with dynamic reordering.

Pipeline: parse FTP -> write a plain .net file (variables in DFS first-occurrence order from the
targets, gates in topological order) -> run the C driver ftp_cudd (CUDD 3.0.0, autodyn reordering)
which builds the BDD, prints double / long-double probabilities and dumps the BDD -> evaluate the
dump here in exact rational arithmetic (scaled Python integers; no rounding anywhere).

Build the C driver (CUDD 3.0.0, e.g. github.com/ivmai/cudd; after `git clone`, run
`touch aclocal.m4; sleep 1; touch configure Makefile.in config.h.in` so make does not try to rerun
automake):
    ./configure --prefix=$CUDD && make -j && make install
    cc -O2 -std=c11 -I$CUDD/include ftp_cudd.c -o ftp_cudd $CUDD/lib/libcudd.a -lm

Run (writes model.net and model.<target>.bdd into --workdir; keep that out of the checkout):
    python3 -I ftp_cudd.py MODEL.ftp --bin ./ftp_cudd --workdir DIR [--targets T1,T3]
        [--order dfs|file] [--reorder sift|sift-converge|group-sift|symm-sift|none]
        [--max-mem-mb 32768] [--final-reduce]

Semantics: '+' OR, '*' AND, integer k = at-least-k-of-n (built with the DP over operands);
'-NAME' operands are complemented (plain Boolean NOT). Basic events independent with the IMPORT
values. Without PROCESS the targets are the roots (gates no other gate references).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from decimal import Decimal, getcontext
from fractions import Fraction


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_ftp(path):
    gates, order, imports, process = {}, [], {}, []
    section, current = "tree", None
    for raw in open(path):
        line = raw.rstrip("\r\n")
        text = line.strip()
        if not text or text.startswith("*"):
            continue
        parts = text.split()
        keyword = parts[0]
        if keyword == "ENDTREE":
            section = "after"
            continue
        if keyword == "PROCESS":
            section = "process"
            process += parts[1:]
            continue
        if keyword == "IMPORT":
            section = "import"
            continue
        if keyword == "LIMIT":
            section = "after"
            continue
        if section == "tree":
            if line[0].isspace() and current and not (len(parts) == 2 and is_number(parts[0])):
                gates[current][1].extend(parts)
                continue
            if len(parts) == 2 and is_number(parts[0]):  # implicit import block (no IMPORT header)
                imports[parts[1]] = parts[0]
                current = None
                continue
            current = parts[0]
            kind = parts[1]
            if kind not in ("+", "*") and not kind.isdigit():
                sys.exit(f"gate {current}: unsupported kind {kind!r}")
            if current in gates:
                sys.exit(f"duplicate gate {current}")
            gates[current] = [kind, parts[2:]]
            order.append(current)
        elif section == "process":
            process += parts
        elif section == "import":
            if len(parts) != 2 or not is_number(parts[0]):
                sys.exit(f"bad import line: {text!r}")
            if parts[1] in imports and imports[parts[1]] != parts[0]:
                sys.exit(f"conflicting duplicate import {parts[1]}")
            imports[parts[1]] = parts[0]
    for name, (_, ops) in gates.items():
        for op in ops:
            base = op[1:] if op.startswith("-") else op
            if base not in gates and base not in imports:
                sys.exit(f"gate {name}: undefined operand {op}")
    return gates, order, imports, process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ftp")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--order", default="dfs", choices=["dfs", "file"])
    ap.add_argument("--reorder", default="sift")
    ap.add_argument("--max-mem-mb", default="32768")
    ap.add_argument("--final-reduce", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--eval-only", action="store_true", help="reuse the .bdd dumps already in --workdir")
    a = ap.parse_args()
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)  # exact numerators/denominators have thousands of digits

    gates, order, imports, process = parse_ftp(a.ftp)
    if a.targets:
        targets = a.targets.split(",")
    elif process:
        targets = process
    else:
        referenced = {op.lstrip("-") for _, ops in gates.values() for op in ops}
        targets = [g for g in order if g not in referenced]
    for t in targets:
        if t not in gates:
            sys.exit(f"target {t} is not a gate")

    # topological gate order (post-order) and DFS first-occurrence variable order, from the targets
    gorder, gseen, vorder, vseen = [], set(), [], set()

    def dfs(g):
        stack = [(g, 0)]
        gseen.add(g)
        while stack:
            name, i = stack.pop()
            ops = gates[name][1]
            if i < len(ops):
                stack.append((name, i + 1))
                base = ops[i].lstrip("-")
                if base in gates:
                    if base not in gseen:
                        gseen.add(base)
                        stack.append((base, 0))
                elif base not in vseen:
                    vseen.add(base)
                    vorder.append(base)
            else:
                gorder.append(name)

    for t in targets:
        if t not in gseen:
            dfs(t)
    if a.order == "file":
        vorder = [v for v in imports if v in vseen]
    gidx = {g: i for i, g in enumerate(gorder)}
    vidx = {v: i for i, v in enumerate(vorder)}

    os.makedirs(a.workdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.ftp))[0]
    net = os.path.join(a.workdir, stem + ".net")
    with open(net, "w") as f:
        f.write(f"vars {len(vorder)}\n")
        for v in vorder:
            f.write(f"{v} {imports[v]}\n")
        f.write(f"gates {len(gorder)}\n")
        for g in gorder:
            kind, ops = gates[g]
            if kind == "+":
                kname, k = "or", 0
            elif kind == "*":
                kname, k = "and", 0
            else:
                kname, k = "atleast", int(kind)
                if not 1 <= k <= len(ops):
                    sys.exit(f"gate {g}: k={k} with {len(ops)} operands not supported")
            enc = []
            for op in ops:
                neg = "-" if op.startswith("-") else ""
                base = op.lstrip("-")
                enc.append(f"{neg}g{gidx[base]}" if base in gates else f"{neg}v{vidx[base]}")
            f.write(f"{g} {kname} {k} {len(ops)} {' '.join(enc)}\n")
        f.write(f"targets {len(targets)} {' '.join(str(gidx[t]) for t in targets)}\n")

    if a.eval_only:
        cmd = None
    else:
        cmd = [a.bin, net, os.path.join(a.workdir, stem), "--reorder", a.reorder, "--max-mem-mb", a.max_mem_mb]
    if cmd:
        if a.final_reduce:
            cmd.append("--final-reduce")
        print("running:", " ".join(cmd), flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        with open(os.path.join(a.workdir, stem + ".driver.txt"), "w") as f:
            f.write(proc.stdout)
        sys.stdout.write("\n".join(l for l in proc.stdout.splitlines() if not l.startswith("final_order")) + "\n")
        if proc.returncode != 0:
            sys.exit(f"driver failed with {proc.returncode}")
        print(f"driver_wall_seconds {time.time() - t0:.3f}")

    # exact probabilities: p_v = a_v / S with S a power of 10; a node at level l has
    # P = N / S^(L - l) with N an integer (L = number of variables; constants sit at level L).
    probs = [Fraction(Decimal(imports[v])) for v in vorder]
    e = 0
    for p in probs:
        d = p.denominator
        while (10 ** e) % d:
            e += 1
    S = 10 ** e
    A = [int(p * S) for p in probs]
    assert all(Fraction(x, S) == p for x, p in zip(A, probs))
    results = {}
    for t in targets:
        t1 = time.time()
        path = os.path.join(a.workdir, f"{stem}.{t}.bdd")
        with open(path) as f:
            L = int(f.readline().split()[1])
            perm = list(map(int, f.readline().split()[1:]))
            hdr = f.readline().split()
            nn, root_e = int(hdr[1]), int(hdr[3])
            pw = [1] * (L + 1)
            for i in range(1, L + 1):
                pw[i] = pw[i - 1] * S
            N = [1] * (nn + 1)  # node 0 = constant ONE: N = 1 over S^0
            lev = [L] * (nn + 1)
            for n in range(1, nn + 1):
                var, te, ee = map(int, f.readline().split())
                l = perm[var]
                lev[n] = l

                def edge(x):  # numerator over S^(L - l - 1)
                    c = x >> 1
                    v = N[c]
                    if x & 1:
                        v = pw[L - lev[c]] - v
                    return v * pw[lev[c] - l - 1]

                assert lev[te >> 1] > l and lev[ee >> 1] > l
                av = A[var]
                N[n] = av * edge(te) + (S - av) * edge(ee)
        c = root_e >> 1
        num = pw[L - lev[c]] - N[c] if root_e & 1 else N[c]
        exact = Fraction(num, pw[L - lev[c]])
        results[t] = exact
        getcontext().prec = 40
        dec = Decimal(exact.numerator) / Decimal(exact.denominator)
        print(f"EXACT {t} = {dec:.30E}  (double {float(exact)!r}; nodes {nn}; exact eval {time.time() - t1:.1f}s;"
              f" reduced numerator/denominator digits {len(str(exact.numerator))}/{len(str(exact.denominator))})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({t: [str(v.numerator), str(v.denominator), float(v)] for t, v in results.items()}, f)


if __name__ == "__main__":
    main()
