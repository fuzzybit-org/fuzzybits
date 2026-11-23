"""Generate ``benchmarks/datasets/v2/manifest.json`` (#158, #184).

Migrates every v1 entry into the v2 card shape with computed structure counts,
plus the private FTP entry (#184) as digests only. Public card digests are
local and cheap to check; the FTP digests come from the private directory,
which is read only when ``FUZZYBITS_PRIVATE_FIXTURES_DIR`` is set.

Run from the repository root::

    python -m benchmarks.datasets.v2_freeze            # write the draft
    python -m benchmarks.datasets.v2_freeze --check    # exit 1 if the draft is stale
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..loader import ManifestError, load_manifest, repo_root, sha256_file
from .structure import mef_structure
from .v2 import DATASET_DIR, PRIVATE_ENV, digests_of

SCHEMA_VERSION = 2
DESCRIPTION = (
    "Draft v2 dataset: a card per model (#158) with every available format and its hash, "
    "computed structure, origin, licence, the fuzzybits reader that loads it, tags and the "
    "reference the harness judges it against. Private entries hold an opaque id, the sorted "
    "sha256 of their files and the modes they may run in, and nothing else (#184)."
)
VERSIONING = (
    "A dataset version never changes once used. This v2 file is a draft until it is frozen: "
    "`python -m benchmarks.datasets.freeze` refuses to set `frozen: true` without --freeze, "
    "and a frozen v2 is regenerated only as v3. The harness, PR gate (#122) and published "
    "history (#123) pin a dataset id."
)
LICENCES = {
    "aralia": {
        "spdx": "NOASSERTION",
        "attribution": (
            "Aralia, the benchmark set distributed with SCRAM "
            "(https://github.com/rakhimov/scram, input/Aralia)."
        ),
        "url": "https://github.com/rakhimov/scram",
        "note": "The source repository states no redistribution terms for the model data.",
    },
    "pwr": {
        "spdx": "CC0-1.0",
        "attribution": (
            "Generic PWR v1.2 from the openpra-org generic-pwr-model submodule "
            "(https://github.com/openpra-org/generic-pwr-model)."
        ),
        "url": "https://github.com/openpra-org/generic-pwr-model/blob/main/LICENSE",
        "note": "SAPHIRE exports converted to OpenPSA MEF; the exporter is not in the submodule.",
    },
}
ORIGINS = {
    "aralia": {
        "kind": "published",
        "citation": "https://github.com/rakhimov/scram/tree/master/input/Aralia",
        "note": "Standard benchmark set for exact top-event probabilities; see #156.",
    },
    "pwr": {
        "kind": "hand-built",
        "citation": (
            "Generic PWR v1.2 (openpra-org/generic-pwr-model): 56 event trees over 140 fault trees"
        ),
        "note": "Exported from a SAPHIRE database, so gate names repeat across fault trees.",
    },
}
READERS = {
    "aralia": {"name": "io.mef (fzb compile)", "status": "ok"},
    "pwr": {
        "name": "io.mef + datasets.compat.generic_pwr shim v1",
        "status": "ok",
        "note": (
            "The exported XML is not self-contained MEF; the harness rewrites a cached copy (#125)."
        ),
    },
}

# The private FTP entry. Its digests are read from the private directory when the
# variable is set, so this file never contains the model's name or values.
PRIVATE_ID = "private-ftp-1"
PRIVATE_MODES = ["compile", "perf", "accuracy-boolean"]

DEFAULTS: dict[str, Any] = {
    "private": {
        # `fzb compile` exposes two FTP levels: 0 is no ir::pc passes, 1 is the
        # pass pipeline (--compilation-passes). See docs/benchmarks.md.
        "compile_levels": [0, 1],
        "compile_budget_s": 600,
        "perf_level": 0,
        "perf_trials": 1 << 20,
        "perf_rounds": 5,
        "perf_budget_s": 600,
        "accuracy_level": 0,
        "accuracy_trials": 1 << 20,
        "accuracy_budget_s": 1800,
        "perf_tolerance": 0.05,
    }
}


def aralia_card(model_id: str) -> dict[str, Any]:
    """A v2 card for one Aralia model."""
    v1 = load_manifest()
    model = v1.by_id(model_id)
    structure = mef_structure(Path(repo_root()) / model.path, query=model.query)
    tags = set(model.tags)
    if structure["gates_by_type"]["NOT"] or structure["gates_by_type"]["XOR"]:
        tags.add("non-coherent")
    if structure["basic_events"] >= 1000:
        tags.add("large")
    return {
        "id": model.id,
        "private": False,
        "source": "aralia",
        "submodule_commit": None,
        "formats": [
            {
                "format": "openpsa-mef",
                "path": model.path,
                "sha256": model.sha256,
                "primary": True,
                "note": "One fault tree; the compiled query is its first define-gate.",
            }
        ],
        "query": model.query,
        "query_note": model.query_note,
        "structure": structure,
        "origin": ORIGINS["aralia"],
        "licence": LICENCES["aralia"],
        "reader": READERS["aralia"],
        "reference": {
            "probability": model.reference.probability,
            "source": model.reference.source,
            "kind": model.reference.kind,
            "uncertainty": model.reference.uncertainty,
            "citation": model.reference.citation,
        },
        "tags": sorted(tags),
        "perf_reason": model.perf_reason,
        "notes": model.notes,
        "accuracy": {"check": model.accuracy.check, "reason": model.accuracy.reason},
        "trials": {
            "accuracy": model.trials_accuracy,
            "perf": model.trials_perf,
            "cpu_perf": model.trials_cpu_perf,
        },
        "time_budget_s": dict(model.time_budget_s),
    }


def pwr_card(model_id: str, *, submodule_commit: str) -> dict[str, Any]:
    """A v2 card for one generic-PWR event-tree group: MEF, its JSInp and the BE CSV."""
    v1 = load_manifest()
    model = v1.by_id(model_id)
    root = repo_root()
    xml = root / model.path
    stem = xml.name[len("converted_") : -len(".xml")] if xml.name.startswith("converted_") else xml.stem
    rows: list[dict[str, Any]] = [
        {
            "format": "openpsa-mef",
            "path": model.path,
            "sha256": sha256_file(xml),
            "primary": True,
            "note": "Event-tree group: one event tree and its fault trees.",
        }
    ]
    jsinp = xml.parent.parent / "saphsolve" / f"{stem}.JSInp"
    if jsinp.is_file():
        rows.append(
            {
                "format": "jsinp",
                "path": str(jsinp.relative_to(root)),
                "sha256": sha256_file(jsinp),
                "note": "The SAPHSOLVE export the MEF was converted from.",
            }
        )
    csv = xml.parent / "be_name_label_and_value_V1.2.csv"
    if csv.is_file():
        rows.append(
            {
                "format": "saphire-project",
                "path": str(csv.relative_to(root)),
                "sha256": sha256_file(csv),
                "note": "Basic-event names, labels and values, shared by every group.",
            }
        )
    structure = mef_structure(xml, query=model.query)
    tags = set(model.tags)
    if structure["event_tree_sequences"]:
        tags.add("event-tree")
    if structure["gates_by_type"]["NOT"] or structure["gates_by_type"]["XOR"]:
        tags.add("non-coherent")
    return {
        "id": model.id,
        "private": False,
        "source": "pwr",
        "submodule_commit": submodule_commit,
        "formats": rows,
        "query": model.query,
        "query_note": model.query_note,
        "structure": structure,
        "origin": ORIGINS["pwr"],
        "licence": LICENCES["pwr"],
        "reader": READERS["pwr"],
        "reference": None,
        "tags": sorted(tags),
        "perf_reason": None,
        "notes": model.notes,
        "accuracy": {"check": model.accuracy.check, "reason": model.accuracy.reason},
        "trials": {
            "accuracy": model.trials_accuracy,
            "perf": model.trials_perf,
            "cpu_perf": model.trials_cpu_perf,
        },
        "time_budget_s": dict(model.time_budget_s),
        "compat": {"shim": model.compat_shim, "version": model.compat_version},
    }


def private_entry(root: str | Path) -> dict[str, Any]:
    """The reduced entry: digests only, never a name.

    Raises when the directory holds anything other than exactly one model,
    because a wrong digest set would turn into a silent ``no-match`` skip.
    """
    ftp = Path(root) / "ftp"
    directories = (
        sorted(d for d in ftp.iterdir() if d.is_dir() and any(d.glob("*.ftp")))
        if ftp.is_dir()
        else []
    )
    if len(directories) != 1:
        raise ManifestError(
            f"{PRIVATE_ENV}/ftp must hold exactly one model directory for {PRIVATE_ID}, "
            f"found {len(directories)}"
        )
    files = digests_of(sorted(directories[0].glob("*.ftp")))
    return {"id": PRIVATE_ID, "private": True, "files": list(files), "modes": list(PRIVATE_MODES)}


def build(*, private_root: str | Path | None = None, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """The whole v2 draft.

    ``previous`` supplies the private entry when the private directory is not
    available, so a regeneration without it keeps the digests rather than
    dropping the entry.
    """
    v1 = load_manifest()
    pwr_sha = v1.submodules.get("input/generic-pwr-model", "")
    models: list[dict[str, Any]] = []
    for model in v1.models:
        if model.source == "aralia":
            models.append(aralia_card(model.id))
        elif model.source == "pwr":
            models.append(pwr_card(model.id, submodule_commit=pwr_sha))
    if private_root is not None:
        models.append(private_entry(private_root))
    elif previous is not None:
        models.extend(row for row in previous.get("models") or [] if row.get("private") is True)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "v2",
        "frozen": False,
        "description": DESCRIPTION,
        "versioning": VERSIONING,
        "submodules": dict(v1.submodules),
        "defaults": DEFAULTS,
        "models": models,
    }


def render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def write(payload: dict[str, Any], *, root: Path | None = None) -> Path:
    out = (root or repo_root()) / DATASET_DIR / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(payload), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.datasets.v2_freeze", description=__doc__
    )
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed draft is stale")
    parser.add_argument(
        "--no-private", action="store_true", help="keep the existing private entry as it is"
    )
    parser.add_argument(
        "--private", default=os.environ.get(PRIVATE_ENV), help="private fixtures directory"
    )
    args = parser.parse_args(argv)
    root = repo_root()
    out = root / DATASET_DIR / "manifest.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
    if args.no_private and (previous is None or not any(
        row.get("private") is True for row in previous.get("models") or ()
    )):
        raise SystemExit(f"{out} has no private entry to keep; drop --no-private")
    payload = build(private_root=None if args.no_private else args.private, previous=previous)
    if args.check:
        current = out.read_text(encoding="utf-8") if out.is_file() else ""
        if current != render(payload):
            print(f"{out} is stale; run `python -m benchmarks.datasets.v2_freeze`", file=sys.stderr)
            return 1
        print(f"{out} is up to date ({len(payload['models'])} entries)")
        return 0
    write(payload, root=root)
    print(f"wrote {out} ({len(payload['models'])} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
