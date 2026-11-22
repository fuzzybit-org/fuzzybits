#!/usr/bin/env python3
"""Resolve hardware CI targets from .github/hardware.json and SKIP_* variables.

    hw_targets.py matrix TARGET...   # GITHUB_OUTPUT: matrix, targets, skipped, any
    hw_targets.py check              # validate the registry

A target is skipped when the environment variable named by its ``skip_var``
equals ``true`` (case-insensitive, like GitHub expression string comparison,
so ``vars.SKIP_X != 'true'`` in a workflow agrees with this script).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / ".github" / "hardware.json"
_SKIP_VAR = re.compile(r"^SKIP_[A-Z0-9_]+$")


def load(path: Path = REGISTRY) -> dict[str, dict[str, Any]]:
    targets = json.loads(path.read_text(encoding="utf-8")).get("targets")
    if not isinstance(targets, dict) or not targets:
        raise SystemExit(f"{path}: no targets")
    return targets


def validate(targets: Mapping[str, Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: dict[str, str] = {}
    for name, spec in targets.items():
        var = spec.get("skip_var")
        if not isinstance(var, str) or not _SKIP_VAR.match(var):
            errors.append(f"{name}: skip_var must match {_SKIP_VAR.pattern}")
        elif var in seen:
            errors.append(f"{name}: skip_var {var} already used by {seen[var]}")
        else:
            seen[var] = name
        runs_on = spec.get("runs_on")
        if isinstance(runs_on, dict):
            if not runs_on.get("group") or not runs_on.get("labels"):
                errors.append(f"{name}: self-hosted runs_on needs a group and labels")
        elif not isinstance(runs_on, str):
            errors.append(f"{name}: runs_on must be a hosted label or {{group, labels}}")
        if not spec.get("device"):
            errors.append(f"{name}: device is required")
    return errors


def is_skipped(spec: Mapping[str, Any], env: Mapping[str, str]) -> bool:
    return env.get(str(spec["skip_var"]), "").lower() == "true"


def resolve(
    names: Sequence[str],
    targets: Mapping[str, Mapping[str, Any]],
    env: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Return (enabled matrix rows, [(skipped target, skip_var)]) in request order."""
    unknown = [n for n in names if n not in targets]
    if unknown:
        raise SystemExit(f"unknown hardware target(s): {' '.join(unknown)} (see {REGISTRY.name})")
    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for name in dict.fromkeys(names):
        spec = targets[name]
        if is_skipped(spec, env):
            skipped.append((name, str(spec["skip_var"])))
        else:
            rows.append({"target": name, **spec})
    return rows, skipped


def _emit(outputs: Mapping[str, str]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={v}" for k, v in outputs.items()]
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


def _summary(rows: Sequence[Mapping[str, Any]], skipped: Sequence[tuple[str, str]]) -> None:
    for name, var in skipped:
        print(f"::notice title=Hardware skipped::{name}: {var}=true")
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = ["## Hardware targets", "", "| target | device | status |", "| --- | --- | --- |"]
    lines += [f"| {r['target']} | {r['device']} | enabled |" for r in rows]
    lines += [f"| {n} | | skipped (`{v}=true`) |" for n, v in skipped]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("matrix")
    m.add_argument("targets", nargs="+")
    sub.add_parser("check")
    args = parser.parse_args(argv)

    targets = load()
    errors = validate(targets)
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 1
    if args.cmd == "check":
        print(f"ok: {len(targets)} hardware target(s)")
        return 0

    rows, skipped = resolve(args.targets, targets, os.environ)
    _summary(rows, skipped)
    _emit(
        {
            "matrix": json.dumps({"include": rows}, separators=(",", ":")),
            "targets": " ".join(r["target"] for r in rows),
            "skipped": " ".join(n for n, _ in skipped),
            "any": "true" if rows else "false",
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
