#!/usr/bin/env python3
"""Check that a pull request stays inside the roadmap (docs/planning.md).

The roadmap is `.github/roadmap.toml`. For one PR, the check reads the linked
issues (`Closes #N`, `Part of #N`, GitHub closing references, `#N` in the title),
their milestones, the files changed and the PR labels, and reports:

  scope.no-issue          no linked issue (label `trivial` exempts small changes)
  scope.untriaged         a linked issue has no milestone
  scope.unknown-milestone the milestone is not in roadmap.toml
  scope.not-active        the milestone is next, later, icebox or done
  scope.lab-in-core       a lab issue's PR changes files outside lab areas
  scope.area              files in areas the milestone does not list
  scope.frozen-area       files in a frozen area (for example hull)
  scope.multi-area        more product areas than policy.max_areas (warning)
  scope.size              more changed lines than policy.size_limit (size_warn: warning)
  scope.new-directory     a new directory under a subsystem root
  surface.<name>          a user-visible surface changed (CLI, default, format, dependency, ...)
  surface.changelog       a surface that needs a CHANGELOG.md entry changed without one
  scope.lab-include       core code includes or imports lab code (no exception)

Labels applied by the maintainer accept a finding: `scope-exception` (not-active,
area, frozen-area, lab-in-core, new-directory), `size-exception` (size) and
`surface-approved` (surface.*). Issues or PRs carrying an interrupt label
(`wrong-answer`, `security`, `release-blocker`) may land from any milestone and
area except frozen ones.

Modes: `warn` reports and exits 0, `enforce` exits 1 on any error that no label
accepts. The default comes from roadmap.toml (policy.mode); `--mode` overrides it.

Usage:
  scope_check.py --pr 245                       # live, through gh
  scope_check.py --event "$GITHUB_EVENT_PATH"   # in the scope workflow (pull_request)
  scope_check.py --backtest-prs prs.json --backtest-issues issues.json \
                 --milestones milestones.json [--roadmap old.toml] [--git]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roadmap_lib as rl  # noqa: E402

LAB_INCLUDE = re.compile(r'^\s*(?:#\s*include\s*[<"]lab/|from\s+lab[.\s]|import\s+lab\b)')


@dataclass
class Finding:
    rule: str
    severity: str  # error | warn
    message: str
    accepted_by: str | None = None  # label that accepts it, if present

    @property
    def blocking(self) -> bool:
        return self.severity == "error" and self.accepted_by is None


def counted_lines(roadmap: rl.Roadmap, pr: rl.PullRequest) -> int:
    exempt = roadmap.policy.get("size_exempt_paths", [])
    return sum(f.additions + f.deletions for f in pr.files if not rl.glob_match(f.path, exempt))


def new_directories(roadmap: rl.Roadmap, pr: rl.PullRequest, dir_exists) -> list[str]:
    """Directories a PR adds directly under a subsystem root (the most specific root that holds the file)."""
    roots = sorted((r.rstrip("/") for r in roadmap.policy.get("subsystem_roots", [])), key=len, reverse=True)
    depth = int(roadmap.policy.get("subsystem_depth", 1))
    ignore = set(roadmap.policy.get("subsystem_ignore", []))
    found: set[str] = set()
    for f in pr.files:
        if f.status != "added":
            continue
        root = next((r for r in roots if f.path.startswith(r + "/")), None)
        if root is None:
            continue
        rest = f.path[len(root) + 1:].split("/")
        if len(rest) <= depth:  # a file directly in the root, not in a new directory
            continue
        if rest[depth - 1] in ignore:
            continue
        candidate = root + "/" + "/".join(rest[:depth])
        if dir_exists(candidate) is False:
            found.add(candidate)
    return sorted(found)


def check(roadmap: rl.Roadmap, pr: rl.PullRequest, issues: dict[int, rl.Issue], dir_exists=lambda d: None) -> tuple[list[Finding], dict]:
    pol = roadmap.policy
    L = roadmap.label
    labels = set(pr.labels)
    findings: list[Finding] = []
    date = pr.merged_at or pr.created_at

    def add(rule: str, severity: str, message: str, accept: str | None = None) -> None:
        findings.append(Finding(rule, severity, message, accept if accept and accept in labels else None))

    facts: dict = {"areas": Counter(), "surfaces": [], "issues": [], "lines": counted_lines(roadmap, pr)}
    for f in pr.files:
        facts["areas"][roadmap.area_of(f.path)] += f.additions + f.deletions

    if rl.is_exempt_author(roadmap, pr.author) or labels & set(pol.get("exempt_labels", [])):
        facts["exempt"] = True
        return findings, facts

    linked = [issues[n] for n in pr.linked if n in issues and not issues[n].is_pr]
    interrupt_labels = set(pol.get("interrupt_labels", []))
    interrupt = bool(labels & interrupt_labels) or any(i.labels & interrupt_labels for i in linked)

    # --- issue and milestone
    statuses: dict[int, str] = {}
    milestones: list[rl.Milestone] = []
    if not linked:
        trivial_max = int(pol.get("trivial_max_lines", 50))
        if L("trivial") in labels and facts["lines"] <= trivial_max:
            pass
        else:
            add("scope.no-issue", "error", "no linked issue: add `Closes #N` or `Part of #N` to the description"
                + (f" (`trivial` covers at most {trivial_max} lines)" if L("trivial") in labels else ""))
    for issue in linked:
        status = "untriaged"
        if issue.milestone is None and not issue.milestone_title:
            add("scope.untriaged", "error", f"#{issue.number} has no milestone: triage it first")
        elif roadmap.find(issue.milestone, issue.milestone_title) is None:
            status = "unknown"
            add("scope.unknown-milestone", "error",
                f"#{issue.number}: milestone {issue.milestone} ({issue.milestone_title}) is not in {roadmap.path.name}")
        else:
            ms = roadmap.find(issue.milestone, issue.milestone_title)
            status = ms.status_at(date)
            milestones.append(ms)
            if status in rl.ACTIVE or status == "lab" or (interrupt and status != "done"):
                pass
            elif status == "next":
                add("scope.not-active", "error", f"#{issue.number} is in `{ms.key}` (next): pulling it forward needs `{L('scope-exception')}`",
                    L("scope-exception"))
            else:
                add("scope.not-active", "error", f"#{issue.number} is in `{ms.key}` ({status}): not open for work", L("scope-exception"))
        statuses[issue.number] = status
        facts["issues"].append((issue.number, issue.milestone_title or "-", status))

    # --- areas
    always = set(pol.get("always_allowed_areas", []))
    touched = set(facts["areas"]) - always
    lab_areas = set(pol.get("lab_areas", ["lab"]))
    is_lab = bool(milestones) and all(m.status_at(date) == "lab" for m in milestones)
    if is_lab:
        outside = sorted(touched - lab_areas)
        if outside:
            add("scope.lab-in-core", "error", f"lab work changes core areas {outside}: core changes need their own product issue",
                L("scope-exception"))
    elif milestones:
        allowed: set[str] = set()
        for m in milestones:
            allowed |= set(m.areas)
        if "*" not in allowed and not interrupt:
            outside = sorted(a for a in touched - allowed if not (roadmap.area(a) and roadmap.area(a).frozen))
            if outside:
                add("scope.area", "error", f"areas {outside} are not in {sorted({m.key for m in milestones})}", L("scope-exception"))
    for a in sorted(touched):
        area = roadmap.area(a)
        if area and area.frozen and not any(a in m.areas for m in milestones):
            add("scope.frozen-area", "error", f"area `{a}` is frozen ({area.note or 'see ROADMAP.md'})", L("scope-exception"))
    product = touched - lab_areas - {"other"}
    if len(product) > int(pol.get("max_areas", 3)):
        add("scope.multi-area", "warn", f"{len(product)} areas {sorted(product)}: split by concern if they are independent")

    # --- size
    limit, warn = int(pol.get("size_limit", 2000)), int(pol.get("size_warn", 800))
    if facts["lines"] > limit:
        add("scope.size", "error", f"{facts['lines']} changed lines (limit {limit}, excluding fixtures and lockfiles)", L("size-exception"))
    elif facts["lines"] > warn:
        add("scope.size", "warn", f"{facts['lines']} changed lines (review budget {warn})")

    # --- new directories
    for d in new_directories(roadmap, pr, dir_exists):
        if any(rl.glob_match(d + "/x", roadmap.area(a).paths) for a in lab_areas if roadmap.area(a)):
            continue
        add("scope.new-directory", "error", f"new subsystem directory `{d}`", L("scope-exception"))

    # --- surfaces
    changelog = any(f.path == "CHANGELOG.md" for f in pr.files)
    needs_changelog = False
    for s in roadmap.surfaces:
        hit = [f.path for f in pr.files if rl.glob_match(f.path, s.paths)]
        if s.added_line:
            pats = [re.compile(p) for p in s.added_line]
            for f in pr.files:
                if s.line_paths and not rl.glob_match(f.path, s.line_paths):
                    continue
                if f.added_lines and any(p.search(ln) for p in pats for ln in f.added_lines):
                    hit.append(f.path)
        if any(re.search(b, pr.body or "") for b in s.body):
            hit.append("declared in the description")
        if hit:
            facts["surfaces"].append(s.name)
            add(f"surface.{s.name}", "error", f"{s.note or s.name}: {', '.join(sorted(set(hit))[:4])}"
                + (" …" if len(set(hit)) > 4 else ""), L("surface-approved"))
            needs_changelog |= s.changelog
    if needs_changelog and not changelog:
        add("surface.changelog", "error", "a user-visible surface changed without a CHANGELOG.md entry")

    # --- lab isolation (live mode: needs patches)
    for f in pr.files:
        if roadmap.area_of(f.path) in lab_areas or not f.added_lines:
            continue
        if any(LAB_INCLUDE.search(ln) for ln in f.added_lines):
            add("scope.lab-include", "error", f"{f.path} includes or imports lab code")

    facts["interrupt"] = interrupt
    return findings, facts


def render(pr: rl.PullRequest, findings: list[Finding], facts: dict, mode: str) -> str:
    out = [f"### Scope check: #{pr.number} {pr.title}", ""]
    if facts.get("exempt"):
        return "\n".join(out + ["Exempt (author or label)."])
    issues = ", ".join(f"#{n} ({t}: {s})" for n, t, s in facts["issues"]) or "none"
    areas = ", ".join(f"{a} {n}" for a, n in facts["areas"].most_common())
    out += [f"- Linked issues: {issues}", f"- Areas (changed lines): {areas}", f"- Counted lines: {facts['lines']}",
            f"- Surfaces: {', '.join(facts['surfaces']) or 'none'}", f"- Interrupt lane: {'yes' if facts.get('interrupt') else 'no'}", ""]
    if not findings:
        out.append("No findings.")
    else:
        out += ["| Rule | Severity | Accepted by | Finding |", "|---|---|---|---|"]
        for f in findings:
            out.append(f"| `{f.rule}` | {f.severity} | {f.accepted_by or '—'} | {f.message} |")
    blocking = [f for f in findings if f.blocking]
    out += ["", f"Mode `{mode}`: " + (f"{len(blocking)} blocking finding(s)." if blocking else "nothing blocks.")]
    return "\n".join(out)


def dir_exists_on_disk(d: str):
    return (rl.ROOT / d).is_dir()


def git_dir_checker(subject_index: dict[int, str]):
    """For backtests: did the directory exist in the parent of the PR's squash commit?"""

    def factory(pr_number: int):
        sha = subject_index.get(pr_number)
        if not sha:
            return lambda d: None

        def exists(d: str):
            r = subprocess.run(["git", "cat-file", "-e", f"{sha}^:{d}"], cwd=rl.ROOT, capture_output=True)
            return r.returncode == 0

        return exists

    return factory


def squash_index(ref: str) -> dict[int, str]:
    out = subprocess.run(["git", "log", ref, "--format=%H%x09%s"], cwd=rl.ROOT, capture_output=True, text=True, check=True).stdout
    index: dict[int, str] = {}
    for line in out.splitlines():
        sha, _, subject = line.partition("\t")
        m = re.search(r"\(#(\d+)\)$", subject) or re.match(r"Merge pull request #(\d+)", subject)
        if m:
            index.setdefault(int(m.group(1)), sha)
    return index


def backtest(args, roadmap: rl.Roadmap) -> int:
    prs = [rl.pr_from_dump(d) for d in json.loads(Path(args.backtest_prs).read_text())]
    if args.patches:
        for pr in prs:
            path = Path(args.patches) / f"{pr.number}.json"
            if not path.is_file() or path.stat().st_size == 0:
                continue
            files = []
            for f in rl.decode_concatenated(path.read_text()):
                added = None
                if f.get("patch") is not None:
                    added = [ln[1:] for ln in f["patch"].splitlines() if ln.startswith("+") and not ln.startswith("+++")]
                files.append(rl.FileChange(f["filename"], f.get("additions", 0), f.get("deletions", 0), f.get("status", "modified"), added))
            pr.files = files
    ms_numbers = {}
    if args.milestones:
        ms_numbers = {m["title"]: m["number"] for m in json.loads(Path(args.milestones).read_text())}
    issues = {d["number"]: rl.issue_from_dump(d, ms_numbers) for d in json.loads(Path(args.backtest_issues).read_text())}
    factory = git_dir_checker(squash_index(args.git_ref)) if args.git else (lambda n: (lambda d: None))
    rows, counts = [], Counter()
    selected = [p for p in prs if p.state == "MERGED" or (args.include_open and p.state == "OPEN")]
    for pr in sorted(selected, key=lambda p: p.number):
        findings, facts = check(roadmap, pr, issues, factory(pr.number))
        errors = [f for f in findings if f.severity == "error"]
        for f in errors:
            counts[f.rule.split(".")[0] + "." + f.rule.split(".")[1]] += 1
        rows.append((pr, findings, facts))
    flagged = [r for r in rows if any(f.severity == "error" for f in r[1])]
    print(f"# Backtest: {len(rows)} PRs against {roadmap.path}")
    print(f"PRs with at least one error finding: {len(flagged)} ({100 * len(flagged) / max(len(rows), 1):.0f}%)")
    print("Errors by rule:", ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    print()
    for pr, findings, facts in rows:
        errs = sorted({f.rule for f in findings if f.severity == "error"})
        warns = sorted({f.rule for f in findings if f.severity == "warn"})
        if args.verbose or errs:
            issues_s = ",".join(f"#{n}:{s}" for n, _, s in facts.get("issues", [])) or "-"
            print(f"#{pr.number:<4} {pr.merged_at[:10] if pr.merged_at else 'open      '} lines={facts['lines']:<6} "
                  f"issues={issues_s:<22} E={','.join(errs) or '-'} W={','.join(warns) or '-'} | {pr.title[:70]}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps([
            {"number": pr.number, "merged_at": pr.merged_at, "title": pr.title, "lines": facts["lines"],
             "areas": dict(facts["areas"]), "issues": facts.get("issues", []), "surfaces": facts.get("surfaces", []),
             "findings": [{"rule": f.rule, "severity": f.severity, "message": f.message} for f in findings]}
            for pr, findings, facts in rows], indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roadmap", default=str(rl.DEFAULT_ROADMAP))
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "fuzzybit-org/fuzzybits"))
    ap.add_argument("--pr", type=int)
    ap.add_argument("--event", help="GitHub pull_request event payload")
    ap.add_argument("--mode", choices=["warn", "enforce"])
    ap.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    ap.add_argument("--backtest-prs")
    ap.add_argument("--backtest-issues")
    ap.add_argument("--milestones", help="milestones JSON (REST) to resolve milestone titles in dumps")
    ap.add_argument("--patches", help="backtest: directory of `gh api pulls/N/files` dumps (N.json) for line rules")
    ap.add_argument("--git", action="store_true", help="backtest: detect new directories from squash commits")
    ap.add_argument("--git-ref", default="origin/develop")
    ap.add_argument("--include-open", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    roadmap = rl.load_roadmap(args.roadmap)
    if args.backtest_prs:
        return backtest(args, roadmap)

    number = args.pr
    if args.event:
        payload = json.loads(Path(args.event).read_text())
        number = number or payload.get("pull_request", {}).get("number")
    if not number:
        ap.error("--pr, --event or --backtest-prs is required")

    pr = rl.fetch_pr(args.repo, number)
    issues = {}
    for n in pr.linked:
        issue = rl.fetch_issue(args.repo, n)
        if issue:
            issues[n] = issue
    findings, facts = check(roadmap, pr, issues, dir_exists_on_disk)
    mode = args.mode or roadmap.policy.get("mode", "warn")
    text = render(pr, findings, facts, mode)
    print(text)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    for f in findings:
        if f.blocking:
            level = "error" if mode == "enforce" else "warning"
            print(f"::{level} title={f.rule}::{f.message}")
    return 1 if mode == "enforce" and any(f.blocking for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
