"""Check or regenerate ``input/ftp/conformance/*/expected.json``.

    python -m benchmarks.reference.fixtures                       # check all
    python -m benchmarks.reference.fixtures --write [NAME...]     # regenerate computed fields

Hand-written fields (``description``, ``covers``, ``files``, ``targets``,
``variants[].policy`` / ``overrides`` / ``max_initiators_per_cutset``) are
kept; structure counts, diagnostics and exact probabilities are recomputed with
the exact evaluators. Per target: ``boolean`` P(g), ``hull`` P(omega_L g) with
the model's significant literals (``negative_literals``: events referenced with
a direct complement in the target's cone), ``hull_positive`` the plain monotone
hull P(up g). A fixture with errors under every duplicate-import policy is
diagnostic-only: it records the diagnostics per policy (results ``null``) and
no ``default_targets`` or ``structure``.
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
SCHEMA_VERSION = 2
INPUT_KEYS = ("description", "covers", "files", "targets")
VARIANT_KEYS = ("policy", "overrides", "max_initiators_per_cutset", "targets")


def fixture_dirs(root: Path = FIXTURES) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / "expected.json").is_file())


def load(fixture: Path) -> dict:
    return json.loads((fixture / "expected.json").read_text(encoding="utf-8"))


def project_of(fixture: Path, spec: Mapping[str, Any]) -> Project:
    return parse_project([fixture / name for name in spec["files"]])


def exact_results(
    project: Project,
    policy: str,
    targets: Sequence[str],
    overrides: Mapping[str, int] | None = None,
    max_initiators: int | None = None,
) -> dict:
    probs, _ = project.probabilities(policy)
    out = {}
    for target in targets:
        b = exact_probability(project, target, probs, "boolean", overrides, max_initiators=max_initiators)
        h = exact_probability(project, target, probs, "hull", overrides, "model", max_initiators)
        hp = exact_probability(project, target, probs, "hull", overrides, "positive", max_initiators)
        out[target] = {
            "boolean": b["probability"],
            "hull": h["probability"],
            "hull_positive": hp["probability"],
            "negative_literals": h["negative_literals"],
            "variables": b["variables"],
        }
    return out


def diagnostic_only(spec: Mapping[str, Any]) -> bool:
    """True when the fixture has errors under every policy (no results)."""
    return all(block["results"] is None for block in spec["policies"].values())


def compute(fixture: Path, spec: Mapping[str, Any]) -> dict:
    project = project_of(fixture, spec)
    targets = list(spec.get("targets") or project.default_targets())
    out: dict = {"schema_version": SCHEMA_VERSION}
    out.update({k: spec[k] for k in INPUT_KEYS if k in spec})
    policies = {}
    for policy in POLICIES:
        diags = project.all_diagnostics(policy)
        ok = not any(d.severity == "error" for d in diags)
        policies[policy] = {
            "diagnostics": [d.key() for d in diags],
            "results": exact_results(project, policy, targets) if ok else None,
        }
    if diagnostic_only({"policies": policies}):
        if spec.get("variants"):
            raise ValueError(f"{fixture.name}: variants need a project without errors")
        out["policies"] = policies
        return out
    out["default_targets"] = project.default_targets()
    out["structure"] = structure_counts(project)
    out["policies"] = policies
    variants = []
    for v in spec.get("variants", []):
        given = {k: v[k] for k in VARIANT_KEYS if k in v}
        overrides = {k: int(x) for k, x in v.get("overrides", {}).items()}
        if "overrides" in given:
            given["overrides"] = overrides
        rule = v.get("max_initiators_per_cutset")
        given["results"] = exact_results(project, v["policy"], v.get("targets") or targets, overrides, rule)
        variants.append(given)
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
    ap.add_argument("names", nargs="*", metavar="NAME", help="fixture directory names (default: all)")
    args = ap.parse_args(argv)
    fixtures = fixture_dirs(args.root)
    unknown = sorted(set(args.names) - {f.name for f in fixtures})
    if unknown:
        ap.error(f"unknown fixtures: {', '.join(unknown)}")
    bad = 0
    for fixture in fixtures:
        if args.names and fixture.name not in args.names:
            continue
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
