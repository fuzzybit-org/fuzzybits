"""Check or regenerate ``input/ftp/conformance/*/expected.json``.

    python -m benchmarks.reference.fixtures            # check
    python -m benchmarks.reference.fixtures --write    # regenerate computed fields

Hand-written fields (``description``, ``covers``, ``files``, ``targets``,
``variants[].policy`` / ``overrides``) are kept; structure counts, diagnostics
and exact probabilities are recomputed with the exact evaluators.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .exact import exact_probability
from .ftp import POLICIES, Project, parse_project
from .structure import structure_counts

FIXTURES = Path(__file__).resolve().parents[2] / "input" / "ftp" / "conformance"
SCHEMA_VERSION = 1
INPUT_KEYS = ("description", "covers", "files", "targets")


def fixture_dirs(root: Path = FIXTURES) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / "expected.json").is_file())


def load(fixture: Path) -> dict:
    return json.loads((fixture / "expected.json").read_text(encoding="utf-8"))


def project_of(fixture: Path, spec: Mapping[str, Any]) -> Project:
    return parse_project([fixture / name for name in spec["files"]])


def exact_results(
    project: Project, policy: str, targets: Sequence[str], overrides: Mapping[str, int] | None = None
) -> dict:
    probs, _ = project.probabilities(policy)
    out = {}
    for target in targets:
        b = exact_probability(project, target, probs, "boolean", overrides)
        h = exact_probability(project, target, probs, "hull", overrides)
        out[target] = {"boolean": b["probability"], "hull": h["probability"], "variables": b["variables"]}
    return out


def compute(fixture: Path, spec: Mapping[str, Any]) -> dict:
    project = project_of(fixture, spec)
    targets = list(spec.get("targets") or project.default_targets())
    out: dict = {"schema_version": SCHEMA_VERSION}
    out.update({k: spec[k] for k in INPUT_KEYS if k in spec})
    out["default_targets"] = project.default_targets()
    out["structure"] = structure_counts(project)
    policies = {}
    for policy in POLICIES:
        diags = project.all_diagnostics(policy)
        ok = not any(d.severity == "error" for d in diags)
        policies[policy] = {
            "diagnostics": [d.key() for d in diags],
            "results": exact_results(project, policy, targets) if ok else None,
        }
    out["policies"] = policies
    variants = []
    for v in spec.get("variants", []):
        overrides = {k: int(x) for k, x in v["overrides"].items()}
        variants.append(
            {
                "policy": v["policy"],
                "overrides": overrides,
                "results": exact_results(project, v["policy"], v.get("targets") or targets, overrides),
            }
        )
    if variants:
        out["variants"] = variants
    return out


def differences(a: Any, b: Any, path: str = "", rel: float = 1e-12) -> list[str]:
    if isinstance(a, float) or isinstance(b, float):
        numbers = isinstance(a, (int, float)) and isinstance(b, (int, float))
        if numbers and math.isclose(a, b, rel_tol=rel, abs_tol=1e-15):
            return []
        return [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}/{key}: missing on one side")
            else:
                out.extend(differences(a[key], b[key], f"{path}/{key}", rel))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in differences(x, y, f"{path}[{i}]", rel)]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.reference.fixtures")
    ap.add_argument("--write", action="store_true", help="rewrite computed fields")
    ap.add_argument("--root", type=Path, default=FIXTURES)
    args = ap.parse_args(argv)
    bad = 0
    for fixture in fixture_dirs(args.root):
        spec = load(fixture)
        got = compute(fixture, spec)
        if args.write:
            (fixture / "expected.json").write_text(json.dumps(got, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {fixture.name}")
            continue
        diff = differences(spec, got)
        bad += bool(diff)
        print(f"{'ok  ' if not diff else 'DIFF'} {fixture.name}")
        for line in diff:
            print(f"     {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
