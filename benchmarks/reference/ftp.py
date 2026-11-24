"""Reference-only CAFTA/FTREX ``.ftp`` project parser (#179).

This is **not** the product reader (``src/io/ftp``, #180). It exists so that
reference values never depend on fuzzybits code (#159), and it is kept small
enough to audit by eye. A project is an ordered set of files read as one model;
sections do not continue from one file into the next.

Line rules (``src/io/ftp/schema/ftrex_ftp.g4`` plus the conventions it leaves
implicit). Where FTREX 2.0 and ZEBRA 1.0b reject a file, or skip a line and
assume p = 1 for the events it would define, this parser reports an error
(#191):

- Names are ``[A-Za-z0-9_/%~-]+`` and may start with a digit. ``%`` and ``~``
  are ordinary name characters: CAFTA exports prefix initiator names with
  ``%`` (``%IE``, ``B-%IE``) and use ``~`` inside names, and FTREX 2.0 and
  ZEBRA 1.0b both solve such files. A leading ``-`` on a child reference
  complements it, so defined names may not start with ``-``.
- ``NAME + c...`` is OR, ``NAME * c...`` is AND, ``NAME k c...`` is at-least-k.
  An indented line continues the child list of the gate above it.
- The gate lines of a file are its tree section: it comes first and ends with
  ``ENDTREE`` (``missing-endtree`` when a ``PROCESS``, ``IMPORT`` or ``LIMIT``
  line or the end of the file comes first). A gate line after it is not read
  (``misplaced-gate``).
- ``PROCESS t...`` names targets; following lines of names only (no gate
  operator in second position) add more targets. A target that is neither a
  gate nor imported is an error (``undefined-process-target``). Without
  ``PROCESS`` the targets are the roots (``missing-process`` warning).
- ``IMPORT`` starts records ``value name [I]``. A record outside an ``IMPORT``
  section is not read (``import-without-header``). A referenced name that is
  neither a gate nor imported is an error (``undefined-reference``), never an
  implicit p = 1.
- ``LIMIT x`` is the truncation limit (recorded, not used).
- Lines starting with ``*`` are header lines (``**CHAR32``, ``**FTITLE: ...``,
  ``**DBNAME: ...``, ``*XEQ``); they are recorded and otherwise ignored. They
  may appear between gate lines, but they end a ``PROCESS`` or ``IMPORT``
  section.
- An event imported more than once takes its last value by default, with a
  warning (``first`` and ``error`` are options); an ``I`` on any of its
  records makes it an initiator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

OR, AND, ATLEAST = "or", "and", "atleast"
POLICIES = ("error", "first", "last")
DEFAULT_POLICY = "last"

_NAME = re.compile(r"[A-Za-z0-9_/%~-]+\Z")
_NUMBER = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?(?:[Ee][+-]?[0-9]+)?\Z")
_KEYWORDS = frozenset({"ENDTREE", "PROCESS", "IMPORT", "LIMIT"})


class ModelError(ValueError):
    """The project cannot be evaluated (see ``diagnostics``)."""

    def __init__(self, message: str, diagnostics: Sequence["Diagnostic"] = ()):
        super().__init__(message)
        self.diagnostics = list(diagnostics)


@dataclass(frozen=True)
class Location:
    file: str
    line: int


@dataclass(frozen=True)
class Diagnostic:
    severity: str  # "error" | "warning"
    code: str
    message: str
    name: str | None = None
    locations: tuple[Location, ...] = ()

    def key(self) -> dict:
        """Comparable form (no message) used by fixtures."""
        return {
            "severity": self.severity,
            "code": self.code,
            "name": self.name,
            "locations": [[loc.file, loc.line] for loc in self.locations],
        }

    def to_json(self) -> dict:
        out = self.key()
        out["message"] = self.message
        return out


@dataclass
class Gate:
    name: str
    kind: str
    k: int  # threshold for ATLEAST; 0 otherwise
    children: list[tuple[str, bool]]  # (name, complemented)
    location: Location


@dataclass(frozen=True)
class Import:
    name: str
    value: float
    initiator: bool
    location: Location


@dataclass
class Project:
    files: tuple[str, ...]
    gates: dict[str, Gate] = field(default_factory=dict)
    imports: list[Import] = field(default_factory=list)
    process: list[str] = field(default_factory=list)
    limit: float | None = None
    headers: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def referenced(self) -> set[str]:
        return {c for g in self.gates.values() for c, _ in g.children}

    def roots(self) -> list[str]:
        """Gates no other gate references, in definition order."""
        ref = self.referenced()
        return [name for name in self.gates if name not in ref]

    def default_targets(self) -> list[str]:
        """Defined PROCESS targets, else the roots."""
        imported = {imp.name for imp in self.imports}
        known = [t for t in dict.fromkeys(self.process) if t in self.gates or t in imported]
        return known if known else self.roots()

    def duplicate_imports(self) -> dict[str, list[Import]]:
        seen: dict[str, list[Import]] = {}
        for imp in self.imports:
            seen.setdefault(imp.name, []).append(imp)
        return {name: occ for name, occ in seen.items() if len(occ) > 1}

    def probabilities(self, policy: str = DEFAULT_POLICY) -> tuple[dict[str, float], list[Diagnostic]]:
        """Resolve import values. ``error`` reports duplicates as errors and
        resolves them like ``first``; ``first``/``last`` report warnings."""
        if policy not in POLICIES:
            raise ValueError(f"duplicate-import policy must be one of {POLICIES}, got {policy!r}")
        diags: list[Diagnostic] = []
        severity = "error" if policy == "error" else "warning"
        for name, occ in self.duplicate_imports().items():
            values = {o.value for o in occ}
            what = "with different values" if len(values) > 1 else "with the same value"
            diags.append(
                Diagnostic(
                    severity,
                    "duplicate-import",
                    f"{name} is imported {len(occ)} times {what} (policy {policy})",
                    name,
                    tuple(o.location for o in occ),
                )
            )
        probs: dict[str, float] = {}
        for imp in self.imports:
            if policy == "last" or imp.name not in probs:
                probs[imp.name] = imp.value
        return probs, diags

    def initiators(self) -> set[str]:
        return {imp.name for imp in self.imports if imp.initiator}

    def all_diagnostics(self, policy: str = DEFAULT_POLICY) -> list[Diagnostic]:
        return list(self.diagnostics) + self.probabilities(policy)[1]


def _split_ref(tok: str) -> tuple[str, bool]:
    if tok.startswith("-"):
        return tok[1:], True
    return tok, False


def _is_name(tok: str) -> bool:
    return bool(_NAME.match(tok)) and not tok.startswith("-")


def _is_ref(tok: str) -> bool:
    return _is_name(_split_ref(tok)[0])


def _is_op(tok: str) -> bool:
    return tok in ("+", "*") or tok.isdigit()


def _is_record(toks: list[str]) -> bool:
    """``value name [I]`` with a value that cannot be a name (it contains ``.``)."""
    return (
        2 <= len(toks) <= 3
        and "." in toks[0]
        and bool(_NUMBER.match(toks[0]))
        and _is_name(toks[1])
        and toks[2:] in ([], ["I"])
    )


class _Reader:
    def __init__(self, project: Project):
        self.p = project

    def error(self, code: str, message: str, loc: Location, name: str | None = None) -> None:
        self.p.diagnostics.append(Diagnostic("error", code, message, name, (loc,)))

    def read(self, path: Path, label: str) -> None:
        section: str | None = None  # "process" | "import" | None
        gate: Gate | None = None
        tree: Location | None = None  # first gate line of the open tree section
        after_tree = False
        text = path.read_text(encoding="latin-1")
        for lineno, raw in enumerate(text.splitlines(), 1):
            loc = Location(label, lineno)
            toks = raw.split()
            if not toks:
                continue
            head = toks[0]
            if head.startswith("*"):
                self.p.headers.append(raw.strip())
                section = None
                continue
            if head in _KEYWORDS:
                if tree is not None and head != "ENDTREE":
                    self.missing_endtree(tree, head)
                tree, gate, after_tree = None, None, True
                section = self.keyword(head, toks, loc)
                continue
            indented = raw[0] in " \t"
            if section == "process":
                if all(_is_name(t) for t in toks) and not (len(toks) > 1 and _is_op(toks[1])):
                    self.p.process.extend(toks)
                    continue
                section = None
            if section == "import":
                if self.import_line(toks, loc):
                    continue
                if indented or len(toks) < 2 or not _is_op(toks[1]):
                    self.error("invalid-line", "expected 'value name [I]'", loc)
                    continue
                section = None
            if _is_record(toks):
                self.error("import-without-header", "import record outside an IMPORT section is not read", loc,
                           toks[1])
                continue
            if indented:
                if gate is None:
                    self.error("invalid-line", "continuation line without an open gate", loc)
                    continue
                self.children(gate, toks, loc)
                continue
            gate = self.gate_line(toks, loc, misplaced=after_tree)
            if gate is not None and tree is None and not after_tree:
                tree = loc
        if tree is not None:
            self.missing_endtree(tree, "the end of the file")

    def missing_endtree(self, start: Location, before: str) -> None:
        message = f"the tree section starting at line {start.line} has no ENDTREE before {before}"
        self.error("missing-endtree", message, start)

    def keyword(self, head: str, toks: list[str], loc: Location) -> str | None:
        if head == "ENDTREE":
            if len(toks) > 1:
                self.error("invalid-line", "unexpected tokens after ENDTREE", loc)
            return None
        if head == "PROCESS":
            bad = [t for t in toks[1:] if not _is_name(t)]
            if bad:
                self.error("invalid-line", "PROCESS targets must be names", loc)
            self.p.process.extend(t for t in toks[1:] if _is_name(t))
            return "process"
        if head == "IMPORT":
            if len(toks) > 1:
                self.error("invalid-line", "unexpected tokens after IMPORT", loc)
            return "import"
        # LIMIT
        if len(toks) != 2 or not _NUMBER.match(toks[1]):
            self.error("invalid-line", "expected 'LIMIT value'", loc)
        else:
            self.p.limit = float(toks[1])
        return None

    def import_line(self, toks: list[str], loc: Location) -> bool:
        if not (2 <= len(toks) <= 3 and _NUMBER.match(toks[0]) and _is_name(toks[1])):
            return False
        if len(toks) == 3 and toks[2] != "I":
            return False
        value = float(toks[0])
        if not 0.0 <= value <= 1.0:
            self.error("invalid-probability", f"value {toks[0]} is outside [0, 1]", loc, toks[1])
        self.p.imports.append(Import(toks[1], value, len(toks) == 3, loc))
        return True

    def gate_line(self, toks: list[str], loc: Location, misplaced: bool = False) -> Gate | None:
        if len(toks) < 2 or not _is_name(toks[0]) or not _is_op(toks[1]):
            self.error("invalid-line", "expected 'NAME (+|*|k) children...'", loc)
            return None
        name, op = toks[0], toks[1]
        if op == "+":
            kind, k = OR, 0
        elif op == "*":
            kind, k = AND, 0
        else:
            kind, k = ATLEAST, int(op)
        gate = Gate(name, kind, k, [], loc)
        self.children(gate, toks[2:], loc)
        if misplaced:
            self.error("misplaced-gate", "gate line after the tree section of its file is not read", loc, name)
            return gate  # continuation lines still belong to it; it is not registered
        if name in self.p.gates:
            first = self.p.gates[name].location
            self.p.diagnostics.append(
                Diagnostic("error", "duplicate-gate", f"gate {name} is defined twice", name, (first, loc))
            )
            return gate  # continuation lines still belong to it; it is not registered
        self.p.gates[name] = gate
        return gate

    def children(self, gate: Gate, toks: Iterable[str], loc: Location) -> None:
        for tok in toks:
            if not _is_ref(tok):
                self.error("invalid-line", f"invalid child reference {tok!r}", loc, gate.name)
                continue
            gate.children.append(_split_ref(tok))


def _check(project: Project) -> None:
    for gate in project.gates.values():
        n = len(gate.children)
        if n == 0:
            project.diagnostics.append(
                Diagnostic("error", "invalid-gate", "gate has no children", gate.name, (gate.location,))
            )
        elif gate.kind == ATLEAST and not 1 <= gate.k <= n:
            message = f"at-least-{gate.k} gate has {n} children"
            project.diagnostics.append(Diagnostic("error", "invalid-gate", message, gate.name, (gate.location,)))
    imported: dict[str, Location] = {}
    for imp in project.imports:
        imported.setdefault(imp.name, imp.location)
    for name, loc in imported.items():
        if name in project.gates:
            project.diagnostics.append(
                Diagnostic("error", "gate-imported", "name is both a gate and an imported event", name,
                           (project.gates[name].location, loc))
            )
    missing: dict[str, Location] = {}
    for gate in project.gates.values():
        for child, _ in gate.children:
            if child not in project.gates and child not in imported:
                missing.setdefault(child, gate.location)
    for name, loc in missing.items():
        project.diagnostics.append(
            Diagnostic("error", "undefined-reference", "referenced name is neither a gate nor imported", name, (loc,))
        )
    for target in dict.fromkeys(project.process):
        if target not in project.gates and target not in imported:
            project.diagnostics.append(
                Diagnostic("error", "undefined-process-target", "PROCESS names neither a gate nor an imported event",
                           target)
            )
    if not project.process:
        project.diagnostics.append(
            Diagnostic("warning", "missing-process", "no PROCESS targets; the targets are the root gates")
        )


def project_files(paths: Iterable[str | Path]) -> list[Path]:
    """Expand directories to their ``*.ftp`` files (sorted); keep file order."""
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(p for p in path.iterdir() if p.suffix.lower() == ".ftp" and p.is_file()))
        else:
            out.append(path)
    if not out:
        raise ModelError("no .ftp files given")
    return out


def parse_project(paths: Iterable[str | Path]) -> Project:
    """Parse files (in the given order) as one model. Locations use file
    basenames, or paths relative to a common directory when basenames clash."""
    files = project_files(paths)
    labels = [f.name for f in files]
    if len(set(labels)) != len(labels):
        labels = [str(f) for f in files]
    project = Project(tuple(labels))
    reader = _Reader(project)
    for path, label in zip(files, labels, strict=True):
        reader.read(path, label)
    _check(project)
    return project
