"""Manifest v2 drafts (#158) and private dataset entries (#184).

A v2 model is a *card*: every available form with its hash, computed
structure, origin, licence, the fuzzybits reader that loads it, tags, and the
reference the harness judges it against. A **private** entry is the reduced
form: an opaque id, ``private: true``, the sorted sha256 of its ``.ftp`` files
and the modes it may be asked for. Nothing in the repository names, locates or
values a private model.

Private files are found by content, not by path: the harness hashes the
``*.ftp`` files of every directory under ``$FUZZYBITS_PRIVATE_FIXTURES_DIR/ftp``
and takes the one whose digest set equals the entry's ``files`` exactly. Zero
or two matches is a skip, reported by reason code only.

This module is import-safe without the private directory present: nothing here
reads it until :func:`discover` is called.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..loader import (
    PRIVATE_MODES,
    PRIVATE_MODES_READY,
    ManifestError,
    repo_root,
)

PRIVATE_ENV = "FUZZYBITS_PRIVATE_FIXTURES_DIR"
PRIVATE_RESULTS_ENV = "FUZZYBITS_PRIVATE_RESULTS"
DATASET_ID = "v2"
DATASET_DIR = Path("benchmarks") / "datasets" / "v2"
SCHEMA_REL = Path("benchmarks") / "schemas" / "dataset-manifest.v2.json"
# Reason codes a private row may carry. They name the harness's own state, never
# a path, a file name or a value from the private directory.
REASONS = ("no-match", "ambiguous", "no-reference", "hull", "no-module", "not-declared", "error")
ROW_KEYS = ("id", "mode", "verdict", "reason")


@dataclass(frozen=True)
class PrivateEntry:
    """The repository-side view of a private model: no path, no values."""

    id: str
    files: tuple[str, ...]
    modes: tuple[str, ...]

    @property
    def ready_modes(self) -> tuple[str, ...]:
        """Modes the harness can run now; ``hull`` waits for #183."""
        return tuple(mode for mode in self.modes if mode in PRIVATE_MODES_READY)


@dataclass(frozen=True)
class V2Model:
    """One public card, kept as the manifest has it (validated by the schema)."""

    id: str
    raw: Mapping[str, Any]

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(str(tag) for tag in self.raw.get("tags") or ())

    @property
    def query(self) -> str:
        return str(self.raw.get("query") or "")

    @property
    def source(self) -> str:
        return str(self.raw.get("source") or "")

    @property
    def primary_path(self) -> str:
        formats = self.raw.get("formats") or ()
        for form in formats:
            if form.get("primary"):
                return str(form.get("path") or "")
        return str(formats[0].get("path") or "") if formats else ""


@dataclass(frozen=True)
class V2Manifest:
    dataset: str
    frozen: bool
    description: str
    versioning: str
    submodules: Mapping[str, str]
    defaults: Mapping[str, Any]
    models: tuple[V2Model, ...]
    private: tuple[PrivateEntry, ...]
    path: Path

    def public_by_id(self, model_id: str) -> V2Model:
        for model in self.models:
            if model.id == model_id:
                return model
        raise KeyError(model_id)

    def private_by_id(self, entry_id: str) -> PrivateEntry:
        for entry in self.private:
            if entry.id == entry_id:
                return entry
        raise KeyError(entry_id)


@dataclass(frozen=True)
class Discovery:
    """Result of matching one entry against the private directory."""

    entry: PrivateEntry
    status: str  # match | no-match | ambiguous
    directory: Path | None = None

    @property
    def card(self) -> dict[str, Any]:
        """``<dir>/<dir.name>.card.json``, the convention the FTP tests use."""
        if self.directory is None:
            return {}
        path = self.directory / f"{self.directory.name}.card.json"
        return _read_json(path)

    @property
    def bench_path(self) -> Path | None:
        """``<dir>/<dir.name>.bench.json``: harness settings and recorded results."""
        if self.directory is None:
            return None
        return self.directory / f"{self.directory.name}.bench.json"

    @property
    def settings(self) -> dict[str, Any]:
        path = self.bench_path
        return _read_json(path) if path is not None else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def digests_of(paths: Sequence[Path]) -> tuple[str, ...]:
    """Sorted sha256 of the files, deduplicated."""
    return tuple(sorted({hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}))


def discover(entry: PrivateEntry, root: str | Path) -> Discovery:
    """Match ``entry`` against the ``ftp/<dir>/`` directories under ``root``.

    A directory matches when the sorted set of its ``*.ftp`` sha256 digests
    equals ``entry.files``. Directories without a ``.ftp`` file are ignored.
    """
    ftp = Path(root) / "ftp"
    matches: list[Path] = []
    if ftp.is_dir():
        for directory in sorted(ftp.iterdir()):
            if not directory.is_dir():
                continue
            files = sorted(directory.glob("*.ftp"))
            if not files:
                continue
            if digests_of(files) == tuple(sorted(entry.files)):
                matches.append(directory)
    if len(matches) == 1:
        return Discovery(entry=entry, status="match", directory=matches[0])
    return Discovery(entry=entry, status="no-match" if not matches else "ambiguous")


def load_v2(path: str | Path | None = None, *, root: Path | None = None) -> V2Manifest:
    """Parse a v2 manifest. Schema and hash checks live in ``freeze.py``."""
    repo = root or repo_root()
    manifest_file = Path(path) if path is not None else repo / DATASET_DIR / "manifest.json"
    payload = _read_json(manifest_file)
    if not payload:
        raise ManifestError(f"v2 manifest not found or not JSON: {manifest_file.name}")
    if int(payload.get("schema_version") or 0) != 2:
        raise ManifestError("v2 manifest must set schema_version 2")
    models: list[V2Model] = []
    private: list[PrivateEntry] = []
    for index, raw in enumerate(payload.get("models") or ()):
        if not isinstance(raw, Mapping):
            raise ManifestError(f"models[{index}] is not an object")
        model_id = str(raw.get("id") or "")
        if raw.get("private") is True:
            files = tuple(str(item).lower() for item in raw.get("files") or ())
            modes = tuple(str(item) for item in raw.get("modes") or ())
            unknown = [mode for mode in modes if mode not in PRIVATE_MODES]
            if unknown:
                raise ManifestError(f"{model_id}: unknown mode(s) {unknown}; allowed {list(PRIVATE_MODES)}")
            private.append(PrivateEntry(id=model_id, files=files, modes=modes))
        else:
            models.append(V2Model(id=model_id, raw=raw))
    ids = [model.id for model in models] + [entry.id for entry in private]
    if len(ids) != len(set(ids)):
        raise ManifestError("duplicate model id in v2 manifest")
    return V2Manifest(
        dataset=str(payload.get("dataset") or ""),
        frozen=bool(payload.get("frozen")),
        description=str(payload.get("description") or ""),
        versioning=str(payload.get("versioning") or ""),
        submodules={str(k): str(v) for k, v in (payload.get("submodules") or {}).items()},
        defaults=dict(payload.get("defaults") or {}),
        models=tuple(models),
        private=tuple(private),
        path=manifest_file,
    )


def private_row(model_id: str, mode: str, verdict: str, reason: str | None = None) -> dict[str, Any]:
    """The only shape a private result may take outside the private directory."""
    if verdict not in {"pass", "fail", "skipped"}:
        raise ValueError(f"private verdict must be pass, fail or skipped, not {verdict!r}")
    if reason is not None and reason not in REASONS:
        raise ValueError(f"unknown private reason code {reason!r}")
    row: dict[str, Any] = {"id": model_id, "mode": mode, "verdict": verdict}
    if reason is not None:
        row["reason"] = reason
    return row


def private_line(row: Mapping[str, Any]) -> str:
    """``private <id> <mode>: pass|fail|skipped[ (<reason code>)]``."""
    reason = row.get("reason")
    tail = f" ({reason})" if reason else ""
    return f"private {row.get('id')} {row.get('mode')}: {row.get('verdict')}{tail}"


def sanitize_private(rows: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Keep only the four allowed keys, so a stray detail can never be published."""
    out: list[dict[str, Any]] = []
    for row in rows or ():
        clean = {key: row[key] for key in ROW_KEYS if row.get(key) is not None}
        if "id" in clean and "mode" in clean and "verdict" in clean:
            out.append(clean)
    return out


def private_lines(rows: Sequence[Mapping[str, Any]] | None) -> list[str]:
    return [private_line(row) for row in sanitize_private(rows)]


def default_settings(entry: PrivateEntry, card: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Bench settings for an entry with no ``.bench.json`` yet.

    ``defaults`` is the v2 manifest's ``defaults`` block; ``private`` in it
    carries the trial counts. The card's own ``policy.duplicate_import`` is the
    duplicate-import default (what the model records), and the query is the
    card's canonical target when it names one.
    """
    private = dict(defaults.get("private") or {})
    policy = dict(card.get("policy") or {})
    duplicate = str(policy.get("duplicate_import") or "last").split()[0]
    levels = list(private.get("compile_levels") or [0, 1])
    return {
        "schema_version": 1,
        "id": entry.id,
        "compile": {
            "levels": [int(level) for level in levels],
            "negation": "boolean",
            "duplicate_import": duplicate,
            "complement_events": "negation",
            "overrides": [],
            "time_budget_s": int(private.get("compile_budget_s") or 600),
        },
        "perf": {
            "level": int(private.get("perf_level", 0)),
            "trials": int(private.get("perf_trials") or 1 << 20),
            "rounds": int(private.get("perf_rounds") or 5),
            "time_budget_s": int(private.get("perf_budget_s") or 300),
        },
        "accuracy": {
            "level": int(private.get("accuracy_level", 0)),
            "trials": int(private.get("accuracy_trials") or 1 << 20),
            "time_budget_s": int(private.get("accuracy_budget_s") or 900),
        },
    }


def settings_of(discovery: Discovery, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """The entry's recorded settings, with any missing block filled from defaults."""
    base = default_settings(discovery.entry, discovery.card, defaults)
    recorded = discovery.settings
    for block in ("compile", "perf", "accuracy"):
        if isinstance(recorded.get(block), Mapping):
            base[block].update(dict(recorded[block]))
    if isinstance(recorded.get("results"), Mapping):
        base["results"] = dict(recorded["results"])
    return base


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
