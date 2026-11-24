"""Shared model for scope_check.py and roadmap_health.py (docs/planning.md).

Reads `.github/roadmap.toml`: the milestones with their status and allowed areas,
the path-to-area map, the surface classes and the policy limits. Fetches pull
requests and issues through the `gh` CLI, or loads them from `gh ... --json`
dumps for offline backtests. Standard library only (Python 3.11+ for tomllib).
"""
from __future__ import annotations

import json
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROADMAP = ROOT / ".github" / "roadmap.toml"

# Product statuses a PR may serve without an exception label.
ACTIVE = {"now"}
STATUSES = {"now", "next", "later", "lab", "icebox", "done"}

# `Closes #N`, `Part of #N`, `Refs #N` at the start of a body line. GitHub's own
# closing references are read separately (closingIssuesReferences).
LINK_LINE = re.compile(r"(?im)^\s*[-*]?\s*(?:closes|close|fixes|fix|resolves|resolve|part of|refs?|issue)\s*:?\s*#(\d+)")
TITLE_REF = re.compile(r"#(\d+)")


# --------------------------------------------------------------------------- config


@lru_cache(maxsize=None)
def _glob_re(pattern: str) -> re.Pattern[str]:
    """Glob to regex: `**` crosses `/`, `*` and `?` do not."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def glob_match(path: str, patterns: list[str]) -> bool:
    return any(_glob_re(p).match(path) for p in patterns)


@dataclass
class Area:
    name: str
    paths: list[str]
    frozen: bool = False  # needs scope-exception unless a milestone lists it
    note: str = ""


@dataclass
class Surface:
    name: str
    paths: list[str]
    added_line: list[str] = field(default_factory=list)  # regexes on added lines (live mode only)
    changelog: bool = True  # CHANGELOG.md must change too
    body: list[str] = field(default_factory=list)  # regexes on the PR description (declared surfaces)
    line_paths: list[str] = field(default_factory=list)  # added_line applies to these files only (empty: all)
    note: str = ""


@dataclass
class Milestone:
    number: int | None  # None until the milestone exists on GitHub; matched by title meanwhile
    key: str
    title: str
    status: str
    areas: list[str]
    exit: str = ""
    wip: int | None = None
    history: list[dict] = field(default_factory=list)  # [{from = "YYYY-MM-DD", status = "..."}]
    rank: int = 99

    def status_at(self, date: str | None) -> str:
        if not date or not self.history:
            return self.status
        current = self.status if not self.history else None
        for entry in sorted(self.history, key=lambda e: e["from"]):
            if entry["from"] <= date[:10]:
                current = entry["status"]
        return current or "later"


@dataclass
class Roadmap:
    path: Path
    policy: dict
    areas: list[Area]
    surfaces: list[Surface]
    milestones: dict  # number (or 'title:<title>') -> Milestone

    def find(self, number: int | None, title: str = "") -> Milestone | None:
        """The roadmap entry for an issue's milestone: by number, else by title."""
        if number is not None and number in self.milestones:
            return self.milestones[number]
        for ms in self.milestones.values():
            if ms.number is None and title and ms.title == title:
                return ms
        return None

    def area_of(self, path: str) -> str:
        for area in self.areas:
            if glob_match(path, area.paths):
                return area.name
        return "other"

    def area(self, name: str) -> Area | None:
        return next((a for a in self.areas if a.name == name), None)

    def label(self, key: str) -> str:
        return self.policy.get("labels", {}).get(key, key)


def load_roadmap(path: Path | str = DEFAULT_ROADMAP) -> Roadmap:
    path = Path(path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    areas = [Area(name=a["name"], paths=list(a["paths"]), frozen=bool(a.get("frozen", False)), note=a.get("note", ""))
             for a in data.get("area", [])]
    surfaces = [
        Surface(name=s["name"], paths=list(s.get("paths", [])), added_line=list(s.get("added_line", [])),
                changelog=bool(s.get("changelog", True)), body=list(s.get("body", [])), note=s.get("note", ""),
                line_paths=list(s.get("line_paths", [])))
        for s in data.get("surface", [])
    ]
    milestones: dict = {}
    names = {a.name for a in areas}
    for m in data.get("milestone", []):
        ms = Milestone(
            int(m["number"]) if "number" in m else None, m["key"], m["title"], m["status"], list(m.get("areas", [])), m.get("exit", ""), m.get("wip"),
            list(m.get("history", [])), int(m.get("rank", 99)),
        )
        if ms.status not in STATUSES:
            raise SystemExit(f"{path}: milestone {ms.number}: unknown status {ms.status!r}")
        unknown = [a for a in ms.areas if a not in names and a != "*"]
        if unknown:
            raise SystemExit(f"{path}: milestone {ms.number}: unknown areas {unknown}")
        key = ms.number if ms.number is not None else f"title:{ms.title}"
        if key in milestones:
            raise SystemExit(f"{path}: milestone {key} listed twice")
        milestones[key] = ms
    return Roadmap(path, data.get("policy", {}), areas, surfaces, milestones)


# --------------------------------------------------------------------------- data


@dataclass
class Issue:
    number: int
    title: str
    state: str
    milestone: int | None
    milestone_title: str
    labels: set[str]
    created_at: str
    closed_at: str | None
    body: str = ""
    updated_at: str = ""
    is_pr: bool = False


@dataclass
class FileChange:
    path: str
    additions: int
    deletions: int
    status: str = "modified"  # added | modified | removed | renamed
    added_lines: list[str] | None = None


@dataclass
class PullRequest:
    number: int
    title: str
    body: str
    author: str
    labels: set[str]
    head: str
    base: str
    state: str
    created_at: str
    merged_at: str | None
    files: list[FileChange]
    linked: list[int]
    closed_at: str | None = None
    updated_at: str = ""


def linked_numbers(title: str, body: str, closing: list[int]) -> list[int]:
    seen: list[int] = []
    for n in closing + [int(x) for x in LINK_LINE.findall(body or "")] + [int(x) for x in TITLE_REF.findall(title or "")]:
        if n not in seen:
            seen.append(n)
    return seen


def gh(args: list[str]) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def gh_json(args: list[str]):
    return json.loads(gh(args) or "null")


def _paginate(path: str) -> list[dict]:
    return decode_concatenated(gh(["api", "--paginate", path]))


def decode_concatenated(out: str) -> list[dict]:
    """`gh api --paginate` concatenates JSON arrays: "[...][...]"."""
    items: list[dict] = []
    decoder = json.JSONDecoder()
    i = 0
    while i < len(out):
        while i < len(out) and out[i].isspace():
            i += 1
        if i >= len(out):
            break
        value, i = decoder.raw_decode(out, i)
        items.extend(value if isinstance(value, list) else [value])
    return items


def issue_from_rest(d: dict) -> Issue:
    ms = d.get("milestone") or {}
    return Issue(
        d["number"], d.get("title", ""), d.get("state", "").upper(), ms.get("number"), ms.get("title", ""),
        {l["name"] for l in d.get("labels", [])}, d.get("created_at", ""), d.get("closed_at"), d.get("body") or "",
        d.get("updated_at", ""), "pull_request" in d,
    )


def issue_from_dump(d: dict, milestone_numbers: dict[str, int]) -> Issue:
    ms = d.get("milestone") or {}
    number = ms.get("number") or milestone_numbers.get(ms.get("title", ""))
    return Issue(
        d["number"], d.get("title", ""), d.get("state", ""), number, ms.get("title", ""),
        {l["name"] for l in d.get("labels", [])}, d.get("createdAt", ""), d.get("closedAt"), d.get("body") or "",
        d.get("updatedAt", ""),
    )


def fetch_issue(repo: str, number: int) -> Issue | None:
    try:
        return issue_from_rest(gh_json(["api", f"repos/{repo}/issues/{number}"]))
    except subprocess.CalledProcessError:
        return None


def fetch_pr(repo: str, number: int, with_patches: bool = True) -> PullRequest:
    d = gh_json(["api", f"repos/{repo}/pulls/{number}"])
    owner, name = repo.split("/", 1)
    q = (
        "query($o:String!,$n:String!,$p:Int!){repository(owner:$o,name:$n){pullRequest(number:$p){"
        "closingIssuesReferences(first:50){nodes{number}}}}}"
    )
    g = gh_json(["api", "graphql", "-f", f"query={q}", "-F", f"o={owner}", "-F", f"n={name}", "-F", f"p={number}"])
    closing = [n["number"] for n in g["data"]["repository"]["pullRequest"]["closingIssuesReferences"]["nodes"]]
    files = []
    for f in _paginate(f"repos/{repo}/pulls/{number}/files?per_page=100"):
        added = None
        if with_patches and f.get("patch") is not None:
            added = [ln[1:] for ln in f["patch"].splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        files.append(FileChange(f["filename"], f.get("additions", 0), f.get("deletions", 0), f.get("status", "modified"), added))
    state = "MERGED" if d.get("merged_at") else d.get("state", "").upper()
    return PullRequest(
        d["number"], d.get("title", ""), d.get("body") or "", d["user"]["login"], {l["name"] for l in d.get("labels", [])},
        d["head"]["ref"], d["base"]["ref"], state, d.get("created_at", ""), d.get("merged_at"), files,
        linked_numbers(d.get("title", ""), d.get("body") or "", closing), d.get("closed_at"), d.get("updated_at", ""),
    )


def pr_from_dump(d: dict) -> PullRequest:
    closing = [c["number"] for c in d.get("closingIssuesReferences") or []]
    files = [FileChange(f["path"], f.get("additions", 0), f.get("deletions", 0)) for f in d.get("files") or []]
    author = (d.get("author") or {}).get("login", "")
    return PullRequest(
        d["number"], d.get("title", ""), d.get("body") or "", author, {l["name"] for l in d.get("labels", [])},
        d.get("headRefName", ""), d.get("baseRefName", ""), d.get("state", ""), d.get("createdAt", ""), d.get("mergedAt"),
        files, linked_numbers(d.get("title", ""), d.get("body") or "", closing), d.get("closedAt"), d.get("updatedAt", ""),
    )


def is_exempt_author(roadmap: Roadmap, author: str) -> bool:
    return author in set(roadmap.policy.get("exempt_authors", []))
