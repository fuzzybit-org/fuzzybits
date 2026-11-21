"""``python -m benchmarks.reference {structure,exact,estimate} ... FILES...``"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Mapping, Sequence

from .circuit import LITERALS
from .exact import exact_cut_sets, exact_probability
from .ftp import DEFAULT_POLICY, POLICIES, ModelError, parse_project
from .hull import DEFAULT_BDD_BUDGET, estimate_hull
from .montecarlo import DEFAULT_MEMORY_MIB, estimate_boolean
from .structure import structure_counts

TOOL = "benchmarks.reference"


def parse_override(text: str) -> tuple[str, int]:
    name, sep, value = text.rpartition("=")
    if not sep or not name or value not in ("0", "1"):
        raise argparse.ArgumentTypeError(f"expected NAME=0|1, got {text!r}")
    return name, int(value)


def parse_trials(text: str) -> int:
    try:
        value = int(text) if text.isdigit() else int(float(text))
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid trial count {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("trials must be positive")
    return value


def parse_count(text: str) -> int:
    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {text!r}")
    return int(text)


def run(
    command: str,
    files: Sequence[str],
    semantics: str = "boolean",
    targets: Sequence[str] = (),
    overrides: Mapping[str, int] | None = None,
    policy: str = DEFAULT_POLICY,
    trials: int = 1 << 20,
    seed: int = 0,
    max_checks: int | None = None,
    bdd_budget: int = DEFAULT_BDD_BUDGET,
    memory_mib: int = DEFAULT_MEMORY_MIB,
    literals: str = "model",
    max_initiators: int | None = None,
    cut_sets: bool = False,
) -> dict:
    """Run one command and return the JSON document. Raises :class:`ModelError`
    (with diagnostics) when the project has errors under ``policy``."""
    if literals not in LITERALS:
        raise ValueError(f"literals must be one of {LITERALS}, got {literals!r}")
    if cut_sets and (command != "exact" or semantics != "hull"):
        raise ValueError("cut sets are available for 'exact' with hull semantics only")
    project = parse_project(files)
    diags = project.all_diagnostics(policy)
    out: dict = {
        "tool": TOOL,
        "command": command,
        "files": list(project.files),
        "duplicate_import": policy,
        "diagnostics": [d.to_json() for d in diags],
    }
    if command == "structure":
        out["structure"] = structure_counts(project)
        return out
    errors = [d for d in diags if d.severity == "error"]
    if errors:
        raise ModelError(f"{len(errors)} error(s) under duplicate-import policy {policy}", diags)
    probs, _ = project.probabilities(policy)
    chosen = list(targets) or project.default_targets()
    if not chosen:
        raise ModelError("no target: the project has no PROCESS targets and no root gate")
    out.update(semantics=semantics, overrides=dict(overrides or {}), targets=chosen)
    if semantics == "hull":
        out["literals"] = literals
    out["max_initiators_per_cutset"] = max_initiators
    results = []
    for target in chosen:
        if command == "exact":
            r = exact_probability(project, target, probs, semantics, overrides, literals, max_initiators)
            if cut_sets:
                r["cut_sets"] = exact_cut_sets(project, target, probs, overrides, literals, max_initiators)
            results.append(r)
        elif semantics == "boolean":
            results.append(estimate_boolean(project, target, probs, overrides, trials, seed, memory_mib,
                                            max_initiators=max_initiators))
        else:
            results.append(
                estimate_hull(project, target, probs, overrides, trials, seed, max_checks, bdd_budget, memory_mib,
                              literals=literals, max_initiators=max_initiators)
            )
    out["results"] = results
    return out


def _fmt(x: float | None) -> str:
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.10e}"


def _print_text(doc: dict) -> None:
    if doc["command"] == "structure":
        for key, value in doc["structure"].items():
            print(f"{key}: {json.dumps(value)}")
        return
    for r in doc["results"]:
        head = f"{r['target']}  {r['semantics']}"
        if r["semantics"] == "hull" and r.get("negative_literals"):
            head += f"  L- = {{{', '.join(r['negative_literals'])}}}"
        if r.get("max_initiators_per_cutset") is not None:
            head += f"  max-initiators {r['max_initiators_per_cutset']}"
        if "probability" in r:
            print(f"{head}  exact  P = {_fmt(r['probability'])}  ({r['variables']} variables)")
            for cs in r.get("cut_sets", []):
                print(f"    {_fmt(cs['probability'])}  {' '.join(cs['literals']) or '(empty)'}")
            continue
        line = f"{head}  {r['method']}  P = {_fmt(r['estimate'])} +- {_fmt(r['se'])}  n = {r['trials']}"
        if "bounds" in r:
            b = r["bounds"]
            line += f"  bounds [{_fmt(b['lower'])}, {_fmt(b['upper'])}]"
        if "lanes" in r:
            lanes = r["lanes"]
            lps = r["lanes_per_second"]
            line += f"  checked {lanes['checked']}/{lanes['undecided']}"
            line += f"  {lps:.1f} lanes/s" if lps else ""
        print(line)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks.reference",
        description="Independent reference values for FTP projects (no fuzzybits code).",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("files", nargs="+", help=".ftp files or directories (one model, in order)")
    common.add_argument("--duplicate-import", choices=POLICIES, default=DEFAULT_POLICY)
    common.add_argument("--json", action="store_true", help="print the JSON document")
    sub.add_parser("structure", parents=[common], help="structure counts and diagnostics")
    evaluate = argparse.ArgumentParser(add_help=False, parents=[common])
    evaluate.add_argument("--semantics", choices=("boolean", "hull"), default="boolean")
    evaluate.add_argument("--literals", choices=LITERALS, default="model",
                          help="hull significant literals: 'model' adds the negative literal of every event "
                               "referenced with a direct complement in the target's cone; 'positive' is the "
                               "plain monotone hull")
    evaluate.add_argument("--max-initiators-per-cutset", type=parse_count, default=None, metavar="N",
                          help="conjoin the delete term not atleast(N+1; initiator-tagged events in the cone)")
    evaluate.add_argument("--target", action="append", default=[], help="target name (repeatable)")
    evaluate.add_argument("--override", action="append", type=parse_override, default=[], metavar="NAME=V",
                          help="replace a gate or event by a constant (repeatable)")
    exact = sub.add_parser("exact", parents=[evaluate], help="exact probability by enumeration (<= 22 events)")
    exact.add_argument("--cut-sets", action="store_true", help="also list the minimal L-cut sets (hull only)")
    est = sub.add_parser("estimate", parents=[evaluate], help="bit-parallel Monte Carlo estimate")
    est.add_argument("--trials", type=parse_trials, default=1 << 20)
    est.add_argument("--seed", type=int, default=0)
    est.add_argument("--max-checks", type=int, default=None, help="cap on exact hull lane checks")
    est.add_argument("--bdd-budget", type=int, default=DEFAULT_BDD_BUDGET, help="BDD nodes per lane check")
    est.add_argument("--memory-mib", type=int, default=DEFAULT_MEMORY_MIB, help="bound on live lane arrays")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    kwargs: dict = {"policy": args.duplicate_import}
    if args.command != "structure":
        kwargs.update(semantics=args.semantics, targets=args.target, overrides=dict(args.override),
                      literals=args.literals, max_initiators=args.max_initiators_per_cutset)
    if args.command == "exact":
        if args.cut_sets and args.semantics != "hull":
            ap.error("--cut-sets needs --semantics hull")
        kwargs.update(cut_sets=args.cut_sets)
    if args.command == "estimate":
        kwargs.update(trials=args.trials, seed=args.seed, max_checks=args.max_checks,
                      bdd_budget=args.bdd_budget, memory_mib=args.memory_mib)
    try:
        doc = run(args.command, args.files, **kwargs)
    except ModelError as exc:
        for d in exc.diagnostics:
            print(f"{d.severity}: {d.code}: {d.message}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        if args.json:
            err = {"tool": TOOL, "command": args.command, "error": str(exc),
                   "diagnostics": [d.to_json() for d in exc.diagnostics]}
            print(json.dumps(err, indent=2))
        return 1
    if args.json:
        print(json.dumps(doc, indent=2))
    else:
        for d in doc["diagnostics"]:
            print(f"{d['severity']}: {d['code']}: {d['message']}", file=sys.stderr)
        _print_text(doc)
    has_errors = any(d["severity"] == "error" for d in doc["diagnostics"])
    return 1 if has_errors else 0
