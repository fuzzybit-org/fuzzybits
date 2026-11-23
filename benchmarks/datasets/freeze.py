"""Validate and freeze dataset manifests. v1 is generated; never edit it in place.

``python -m benchmarks.datasets.freeze`` validates the committed manifests:
schema conformance for both versions, on-disk files and digests for every
public v2 card, and the v1 sha256 pin. A private v2 entry is checked only when
``FUZZYBITS_PRIVATE_FIXTURES_DIR`` is set, because it exists only there; when
the variable is unset the entry is reported as unchecked rather than failing.

``--freeze`` is the only way to set ``frozen: true`` on a v2 draft, and it also
regenerates v1 from the tree (``--write-v1``). Both refuse to run unless asked.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from ..loader import (
    DEFAULT_CPU_ACCURACY_S,
    DEFAULT_CPU_COMPILE_S,
    DEFAULT_CPU_PERF_S,
    DEFAULT_GPU_PERF_S,
    FZB_1E8,
    INFEASIBLE_HITS,
    LOW_POWER_HITS,
    NUS9601_CPU_ACCURACY_S,
    NUS9601_CPU_PERF_S,
    NUS9601_GPU_PERF_S,
    NUS9601_REF,
    NUS9601_REF_SOURCE,
    ORACLE_REL,
    PERF_REASONS,
    CPU_PERF_TRIALS,
    PERF_TRIALS,
    PWR_SUBMODULE,
    SAMPLER_MIN_P,
    TARGET_HITS,
    ManifestError,
    accuracy_trials,
    classify_accuracy,
    expected_hits,
    load_manifest,
    manifest_path,
    repo_root,
    sha256_file,
    validate_manifest,
)
from . import jsonschema_lite
from .compat import SHIM_ID, SHIM_VERSION
from .v2 import DATASET_DIR as DATASET_DIR_V2
from .v2 import PRIVATE_ENV

# v1 is frozen: its bytes must never change. This pin is the test.
V1_MANIFEST_SHA256 = "907efec549bcd27c8d388854679a811089efe82bd7c3d49b82fd05e9a67674b7"

GATE_RE = re.compile(r'<define-gate\s+name="([^"]+)"')
FT_RE = re.compile(r'<define-fault-tree\s+name="([^"]+)"')
PWR_NAME_RE = re.compile(r"converted_(.+)_et_(Grp-\d+)_")

# Remaining PWR files that the dataset-side generic_pwr shim cannot reconstruct.
# Shim v1 compiles EQK-BIN4/Grp-5 (G10/G3) and EQK-BIN5/Grp-2 (F44→FE44 + G3).
# Grp-6 still fails after G3 clones: cyclic FE56/FE57 fork nesting.
KNOWN_COMPILE_FAILURES: dict[str, dict] = {
    "EQK-BIN4/Grp-6": {
        "expected": "validity_error",
        "missing_elements": ["FE56", "FE57"],
        "reason": (
            "After cloning private G3 into referencing fault trees, MEF initialize "
            "raises ValidityError: Functional event FE56 must appear after "
            "functional event FE57 in event tree fork paths. The exported event "
            "tree nests FE56 then FE57 on some paths and FE57 then FE56 on others, "
            "so no define-functional-event reorder can satisfy "
            "CheckFunctionalEventOrder. Not reconstructable from JSInp gate/BE "
            "data without rewriting the event tree. Dataset, not the engine."
        ),
    },
}


def _first(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        raise SystemExit("no match for " + pattern.pattern)
    return match.group(1)


def _git_submodule_sha(root: Path, path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"HEAD:{path}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _git_actual_submodule_sha(root: Path, path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root / path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def freeze(root: Path | None = None) -> Path:
    repo = root or repo_root()
    oracle = json.loads((repo / ORACLE_REL).read_text(encoding="utf-8"))
    pwr_sha = _git_actual_submodule_sha(repo, PWR_SUBMODULE)
    models: list[dict] = []

    for xml in sorted((repo / "input" / "Aralia").glob("*.xml")):
        text = xml.read_text(encoding="utf-8")
        model_id = xml.stem
        tags = ["accuracy"]
        notes = None
        cpu_acc = DEFAULT_CPU_ACCURACY_S
        cpu_perf = DEFAULT_CPU_PERF_S
        gpu_perf = DEFAULT_GPU_PERF_S
        if model_id in PERF_REASONS:
            tags.append("perf")
        if model_id == "nus9601":
            probability = NUS9601_REF
            reference = {
                "probability": probability,
                "source": NUS9601_REF_SOURCE,
                "kind": "exact",
                "uncertainty": 0.0,
                "citation": "https://github.com/fuzzybit-org/fuzzybits/issues/120",
            }
            tags = ["accuracy", "perf", "hard"]
            cpu_acc = NUS9601_CPU_ACCURACY_S
            cpu_perf = NUS9601_CPU_PERF_S
            gpu_perf = NUS9601_GPU_PERF_S
            n_acc = accuracy_trials(probability)
            notes = (
                f"Accuracy n is {n_acc} (96·2^20, same as fzb --trials 1e8) so n·p ≈ "
                f"{expected_hits(n_acc, probability):.0f}. CPU accuracy budget is "
                f"{cpu_acc}s (3× the 60s default) because the sampler walks 1,567 "
                "events. fzb inspect --exact refuses 1,567 SAMPLE events (cap 30); "
                "scram-cli --bdd/--zbdd OOMs on this model."
            )
        else:
            if model_id not in oracle:
                raise SystemExit(f"no BDD oracle for {model_id}")
            probability = float(oracle[model_id])
            reference = {
                "probability": probability,
                "source": "bdd",
                "kind": "exact",
                "uncertainty": 0.0,
                "citation": str(ORACLE_REL),
            }
            n_acc = accuracy_trials(probability)
            hits = expected_hits(n_acc, probability)
            if hits < TARGET_HITS:
                tags.append("hard")
                notes = (
                    f"n·p = {hits:.3g} at the {n_acc} trial cap (fzb --trials 1e8); "
                    f"{TARGET_HITS} hits would need n ≈ {math.ceil(TARGET_HITS / probability):.3g}, "
                    "which is outside the accuracy time budget."
                )
        spec = classify_accuracy(probability, n_acc)
        models.append(
            {
                "id": model_id,
                "path": str(xml.relative_to(repo)),
                "sha256": sha256_file(xml),
                "source": "aralia",
                "submodule_commit": None,
                "query": _first(GATE_RE, text),
                "query_note": None,
                "reference": reference,
                "tags": tags,
                "perf_reason": PERF_REASONS.get(model_id),
                "notes": notes,
                "accuracy": {"check": spec.check, "reason": spec.reason},
                "trials": {
                    "accuracy": n_acc,
                    "perf": PERF_TRIALS if "perf" in tags else None,
                    "cpu_perf": CPU_PERF_TRIALS.get(model_id) if "perf" in tags else None,
                },
                "expected_hits_accuracy": expected_hits(n_acc, probability),
                "time_budget_s": {
                    "cpu_accuracy": cpu_acc,
                    "cpu_perf": cpu_perf if "perf" in tags else DEFAULT_CPU_PERF_S,
                    "gpu_perf": gpu_perf if "perf" in tags else DEFAULT_GPU_PERF_S,
                    "cpu_compile": DEFAULT_CPU_COMPILE_S,
                },
            }
        )

    pwr_dir = repo / PWR_SUBMODULE / "data" / "openpsa"
    xmls = sorted(pwr_dir.glob("converted_*.xml"))
    if len(xmls) != 95:
        raise SystemExit(f"expected 95 PWR OpenPSA XMLs, found {len(xmls)} in {pwr_dir}")
    for xml in xmls:
        text = xml.read_text(encoding="utf-8")
        match = PWR_NAME_RE.search(xml.name)
        short = f"{match.group(1)}/{match.group(2)}" if match else xml.stem
        known = KNOWN_COMPILE_FAILURES.get(short)
        notes = (
            "v1 is compile/size/throughput only. SAPHIRE MCS probabilities "
            "are mostly 1e-10 to 1e-6 (README); values below 2.3e-10 "
            "quantize to zero (#17). Revisit accuracy with #92, #17, #6. "
            f"The generic_pwr shim v{SHIM_VERSION} rewrites a cached MEF copy "
            "from the matching SAPHSOLVE JSInp (and BE CSV); the submodule XML "
            "is not modified."
        )
        if known:
            notes = notes + " Known compile failure: " + known["reason"]
        model_row = {
            "id": short,
            "path": str(xml.relative_to(repo)),
            "sha256": sha256_file(xml),
            "source": "pwr",
            "submodule_commit": pwr_sha,
            "query": _first(GATE_RE, text),
            "query_note": (
                f"first fault-tree {_first(FT_RE, text)!r} top; event-tree "
                "sequences are not compiled (#92). SAPHIRE MCS references "
                "are not used for accuracy in v1."
            ),
            "reference": None,
            "tags": ["pwr"],
            "accuracy": {
                "check": "n/a",
                "reason": (
                    "v1 PWR is compile/size/throughput only; SAPHIRE MCS "
                    "references are not a 4-sigma gate (#92, #17, #6)"
                ),
            },
            "perf_reason": None,
            "notes": notes,
            "trials": {"accuracy": None, "perf": PERF_TRIALS, "cpu_perf": None},
            "expected_hits_accuracy": None,
            "time_budget_s": {
                "cpu_accuracy": DEFAULT_CPU_ACCURACY_S,
                "cpu_perf": DEFAULT_CPU_PERF_S,
                "gpu_perf": DEFAULT_GPU_PERF_S,
                "cpu_compile": DEFAULT_CPU_COMPILE_S,
            },
            "compat": {"shim": SHIM_ID, "version": SHIM_VERSION},
        }
        if known:
            model_row["compile"] = {
                "expected": known["expected"],
                "missing_elements": list(known["missing_elements"]),
                "reason": known["reason"],
            }
        models.append(model_row)

    payload = {
        "schema_version": 1,
        "dataset": "v1",
        "description": (
            "Frozen Aralia accuracy set (43 models, exact top-event probabilities) "
            "plus generic-pwr-model (95 event-tree-group XMLs, compile/size/throughput "
            "only). synthetic-models is excluded (no license)."
        ),
        "versioning": (
            "A dataset version never changes once used. Schema or contents changes "
            "are a new version (v2, …). The harness, PR gate (#122) and published "
            "history (#123) pin a dataset id."
        ),
        "submodules": {PWR_SUBMODULE: pwr_sha},
        "defaults": {
            "accuracy_target_hits": TARGET_HITS,
            "accuracy_infeasible_hits": INFEASIBLE_HITS,
            "accuracy_low_power_hits": LOW_POWER_HITS,
            "accuracy_sampler_min_p": SAMPLER_MIN_P,
            "accuracy_min_trials": 1 << 20,
            "accuracy_max_trials": FZB_1E8,
            "perf_trials": PERF_TRIALS,
            "time_budget_s": {
                "cpu_accuracy": DEFAULT_CPU_ACCURACY_S,
                "cpu_perf": DEFAULT_CPU_PERF_S,
                "gpu_perf": DEFAULT_GPU_PERF_S,
                "cpu_compile": DEFAULT_CPU_COMPILE_S,
            },
            "compile": {"pdag_c": 2, "no_kn": True},
        },
        "models": models,
    }
    out = manifest_path(repo)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.datasets.freeze",
        description="Validate the committed dataset manifests; regenerate or freeze only when asked.",
    )
    parser.add_argument(
        "--write-v1", action="store_true", help="regenerate datasets/v1/manifest.json from the tree"
    )
    parser.add_argument(
        "--freeze", action="store_true", help="set frozen: true on the v2 draft (after validation)"
    )
    parser.add_argument("--dataset", default=None, help="validate one dataset id (default: all)")
    args = parser.parse_args(argv)

    if args.write_v1:
        out = freeze()
        print(f"wrote {out}", file=sys.stderr)
    problems = validate_all(dataset=args.dataset)
    for line in problems:
        print(line, file=sys.stderr)
    if problems:
        return 1
    if args.freeze:
        frozen = set_frozen()
        print(f"froze {frozen}", file=sys.stderr)
    return 0


def set_frozen(root: Path | None = None) -> Path:
    """Set ``frozen: true`` on the v2 draft. Callers validate first."""
    repo = root or repo_root()
    out = repo / DATASET_DIR_V2 / "manifest.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise SystemExit(f"{out} is not a v2 manifest")
    payload["frozen"] = True
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out.write_text(text, encoding="utf-8")
    return out


def validate_v1(root: Path | None = None) -> list[str]:
    """v1 is frozen: its schema, its digests and its sha256 pin."""
    repo = root or repo_root()
    path = repo / "benchmarks" / "datasets" / "v1" / "manifest.json"
    problems: list[str] = []
    digest = sha256_file(path)
    if digest != V1_MANIFEST_SHA256:
        problems.append(
            f"v1 manifest sha256 {digest} != pinned {V1_MANIFEST_SHA256}; v1 is frozen, "
            "cut v3 instead"
        )
    problems.extend(f"v1: {item}" for item in jsonschema_lite.validate_file(
        repo / "benchmarks" / "schemas" / "dataset-manifest.v1.json", path
    ))
    try:
        validate_manifest(load_manifest(path, root=repo, check_files=False), root=repo)
    except ManifestError as exc:
        problems.append(f"v1: {exc}")
    return problems


def validate_v2(
    root: Path | None = None, *, private_root: str | Path | None = None
) -> tuple[list[str], list[str]]:
    """Schema, digests and private-entry checks for the v2 draft.

    Returns ``(problems, notes)``. Notes record what was not checked: a private
    entry is checked only with the private directory present, since it is the
    only place its files exist.
    """
    from . import jsonschema_lite
    from .v2 import SCHEMA_REL, discover, load_v2

    repo = root or repo_root()
    path = repo / DATASET_DIR_V2 / "manifest.json"
    problems: list[str] = []
    notes: list[str] = []
    if not path.is_file():
        return [f"v2: {path} is missing"], notes
    problems.extend(f"v2: {item}" for item in jsonschema_lite.validate_file(repo / SCHEMA_REL, path))
    try:
        manifest = load_v2(path, root=repo)
    except ManifestError as exc:
        return [*problems, f"v2: {exc}"], notes
    if manifest.frozen:
        notes.append("v2 is frozen: edits need a new version")
    seen: set[str] = set()
    for model in manifest.models:
        if model.id in seen:
            problems.append(f"v2 {model.id}: duplicate id")
        seen.add(model.id)
        formats = list(model.raw.get("formats") or ())
        if sum(1 for form in formats if form.get("primary")) != 1:
            problems.append(f"v2 {model.id}: exactly one format must be primary")
        for form in formats:
            target = repo / str(form.get("path") or "")
            if not target.is_file():
                problems.append(f"v2 {model.id}: missing file {form.get('path')}")
                continue
            actual = sha256_file(target)
            if actual != form.get("sha256"):
                problems.append(
                    f"v2 {model.id}: sha256 mismatch for {form.get('path')}: "
                    f"manifest={form.get('sha256')} disk={actual}"
                )
        source = model.source
        if source == "pwr":
            expected = manifest.submodules.get(PWR_SUBMODULE)
            if expected and model.raw.get("submodule_commit") != expected:
                problems.append(
                    f"v2 {model.id}: submodule_commit {model.raw.get('submodule_commit')} "
                    f"!= submodules[{PWR_SUBMODULE!r}] {expected}"
                )
    for entry in manifest.private:
        if not entry.files:
            problems.append(f"v2 {entry.id}: private entry lists no files")
        if sorted(set(entry.files)) != list(entry.files):
            problems.append(f"v2 {entry.id}: files must be sorted and unique")
        if private_root is None:
            notes.append(f"v2 {entry.id}: not checked (no {PRIVATE_ENV})")
            continue
        found = discover(entry, private_root)
        if found.status != "match":
            problems.append(f"v2 {entry.id}: discovery {found.status} under {PRIVATE_ENV}")
            continue
        card = found.card
        if not card:
            problems.append(f"v2 {entry.id}: matched directory has no card")
            continue
        if card.get("files") and len(card["files"]) != len(entry.files):
            problems.append(f"v2 {entry.id}: card lists a different file count")
        notes.append(f"v2 {entry.id}: matched")
    return problems, notes


def validate_all(
    root: Path | None = None,
    *,
    dataset: str | None = None,
    private_root: str | Path | None = None,
) -> list[str]:
    """Every committed dataset: problems only. Prints nothing."""
    repo = root or repo_root()
    if private_root is None:
        private_root = os.environ.get(PRIVATE_ENV) or None
    problems: list[str] = []
    notes: list[str] = []
    if dataset in (None, "v1"):
        problems.extend(validate_v1(repo))
    if dataset in (None, "v2"):
        found, notes = validate_v2(repo, private_root=private_root)
        problems.extend(found)
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
