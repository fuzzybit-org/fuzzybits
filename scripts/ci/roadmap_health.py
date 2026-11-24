#!/usr/bin/env python3
"""Weekly roadmap health report: catch scope drift while it is still small (docs/planning.md).

Measures, over a window (default: the last 7 days):

  focus        share of merged PRs, and of their changed lines, that serve a `now`
               milestone or the interrupt lane
  drift        merged PRs whose linked issues sit in next/later/lab/icebox/done milestones,
               in no milestone, or that link no issue at all
  triage       open issues without a milestone, with their age
  wip          open issues labelled status:in-progress per milestone, against its limit
  burn         per active milestone: open, closed in the window
  surface      merged PRs that changed a surface (per scope_check), with and without approval
  rfc          RFCs still `proposed` that merged PRs implement (cited by file name or issue)
  trackers     tracking issues whose children sit in other milestones than their own
  lab          lab issues past their time box (policy.lab_days without update)
  prs          open PRs older than policy.stale_pr_days
  branches     remote branches with no open PR, older than policy.stale_branch_days
  size         p50 and p90 of counted lines per merged PR

Alarms at the top list every threshold crossed (policy.health). Data comes from gh
(live) or from dumps (`--prs`, `--issues`, `--milestones`) for a backtest.

Usage:
  roadmap_health.py [--days 7 | --since 2026-11-18 --until 2026-11-25] [--post-issue N]
  roadmap_health.py --prs prs.json --issues issues.json --milestones milestones.json --since ... --until ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roadmap_lib as rl  # noqa: E402
import scope_check as sc  # noqa: E402

TRACKING = re.compile(r"(?i)^\s*tracking\b|\(tracking\)")
CHILD_REF = re.compile(r"#(\d+)")
RFC_STATUS = re.compile(r"(?im)^Status:\s*([a-z-]+)")
RFC_ISSUES = re.compile(r"(?im)^Issue:.*$")


def iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def days_between(a: str, b: datetime) -> float:
    t = datetime.fromisoformat(a.replace("Z", "+00:00"))
    return (b - t).total_seconds() / 86400


def load_live(repo: str):
    prs_raw = rl.gh_json(["pr", "list", "--repo", repo, "--state", "all", "--limit", "300", "--json",
                          "number,title,state,labels,createdAt,closedAt,mergedAt,updatedAt,baseRefName,headRefName,body,files,"
                          "closingIssuesReferences,author"])
    issues_raw = rl.gh_json(["issue", "list", "--repo", repo, "--state", "all", "--limit", "1000", "--json",
                             "number,title,state,labels,milestone,createdAt,closedAt,updatedAt,body"])
    ms_raw = rl.gh_json(["api", f"repos/{repo}/milestones?state=all&per_page=100"])
    return prs_raw, issues_raw, ms_raw


def remote_branches(ref_prefix: str = "refs/remotes/origin/") -> list[tuple[str, str]]:
    try:
        out = subprocess.run(["git", "for-each-ref", ref_prefix, "--format=%(refname:short)\t%(committerdate:iso-strict)"],
                             cwd=rl.ROOT, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows = []
    for line in out.splitlines():
        name, _, date = line.partition("\t")
        name = name.removeprefix("origin/")
        if name in ("HEAD", "origin", "develop", "main", "gh-pages", "bench-results"):
            continue
        rows.append((name, date))
    return rows


def rfc_drift(prs: list[rl.PullRequest], since: str, until: str) -> list[tuple[str, str, list[int]]]:
    out = []
    for path in sorted((rl.ROOT / "docs" / "rfc").glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8")
        m = RFC_STATUS.search(text)
        status = (m.group(1) if m else "unknown").lower()
        if status not in ("proposed", "draft"):
            continue
        header = RFC_ISSUES.search(text)
        nums = {int(n) for n in CHILD_REF.findall(header.group(0))} if header else set()
        stem = path.name
        hits = [p.number for p in prs if p.merged_at and since <= p.merged_at <= until
                and (stem in (p.body or "") or any(n in p.linked for n in nums))]
        if hits:
            out.append((path.relative_to(rl.ROOT).as_posix(), status, hits))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roadmap", default=str(rl.DEFAULT_ROADMAP))
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "fuzzybit-org/fuzzybits"))
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--prs")
    ap.add_argument("--issues")
    ap.add_argument("--milestones")
    ap.add_argument("--patches", help="backtest: directory of `gh api pulls/N/files` dumps (N.json)")
    ap.add_argument("--no-branches", action="store_true")
    ap.add_argument("--out", help="write the markdown report here")
    ap.add_argument("--post-issue", type=int, help="replace the body of this issue with the report")
    ap.add_argument("--fail-on-alarm", action="store_true")
    args = ap.parse_args()

    roadmap = rl.load_roadmap(args.roadmap)
    pol = roadmap.policy
    health = pol.get("health", {})
    now = datetime.fromisoformat(args.until + "T23:59:59+00:00") if args.until else datetime.now(timezone.utc)
    start = datetime.fromisoformat(args.since + "T00:00:00+00:00") if args.since else now - timedelta(days=args.days)
    since, until = iso(start), iso(now)

    if args.prs:
        prs_raw = json.loads(Path(args.prs).read_text())
        issues_raw = json.loads(Path(args.issues).read_text())
        ms_raw = json.loads(Path(args.milestones).read_text()) if args.milestones else []
    else:
        prs_raw, issues_raw, ms_raw = load_live(args.repo)
    ms_numbers = {m["title"]: m["number"] for m in ms_raw}
    prs = [rl.pr_from_dump(d) for d in prs_raw]
    issues = {d["number"]: rl.issue_from_dump(d, ms_numbers) for d in issues_raw}
    L = roadmap.label
    interrupt = set(pol.get("interrupt_labels", []))

    # ---------------------------------------------------------------- merged PRs in the window
    merged = [p for p in prs if p.merged_at and since <= p.merged_at <= until and not rl.is_exempt_author(roadmap, p.author)]
    # Line-based surfaces need patches: from --patches dumps, or one REST call per merged PR when live.
    for p in merged:
        files = None
        if args.patches:
            path = Path(args.patches) / f"{p.number}.json"
            if path.is_file() and path.stat().st_size:
                files = rl.decode_concatenated(path.read_text())
        elif not args.prs:
            files = rl._paginate(f"repos/{args.repo}/pulls/{p.number}/files?per_page=100")
        if files is not None:
            p.files = [rl.FileChange(f["filename"], f.get("additions", 0), f.get("deletions", 0), f.get("status", "modified"),
                                     [ln[1:] for ln in f["patch"].splitlines() if ln.startswith("+") and not ln.startswith("+++")]
                                     if f.get("patch") is not None else None) for f in files]
    buckets: dict[str, list[tuple[rl.PullRequest, int]]] = defaultdict(list)
    surface_rows = []
    sizes = []
    for p in merged:
        findings, facts = sc.check(roadmap, p, issues)
        lines = facts["lines"]
        sizes.append(lines)
        linked = [issues[n] for n in p.linked if n in issues and not issues[n].is_pr]
        if not linked:
            bucket = "no issue"
        elif any(i.labels & interrupt for i in linked) or p.labels & interrupt:
            bucket = "interrupt"
        else:
            sts = []
            for i in linked:
                if i.milestone is None and not i.milestone_title:
                    sts.append("untriaged")
                elif roadmap.find(i.milestone, i.milestone_title):
                    sts.append(roadmap.find(i.milestone, i.milestone_title).status_at(p.merged_at))
                else:
                    sts.append("unknown")
            bucket = "now" if "now" in sts else sorted(sts)[0]
        buckets[bucket].append((p, lines))
        if facts["surfaces"]:
            approved = L("surface-approved") in p.labels
            surface_rows.append((p.number, facts["surfaces"], approved))

    n_merged = len(merged)
    lines_total = sum(sizes) or 1
    focused = buckets.get("now", []) + buckets.get("interrupt", [])
    focus_pr = len(focused) / n_merged if n_merged else 1.0
    focus_lines = sum(l for _, l in focused) / lines_total if n_merged else 1.0

    # ---------------------------------------------------------------- open issues
    open_issues = [i for i in issues.values() if i.state.upper() == "OPEN" and not i.is_pr]
    untriaged = sorted((i for i in open_issues if i.milestone is None and not i.milestone_title), key=lambda i: i.created_at)
    wip = Counter(id(roadmap.find(i.milestone, i.milestone_title)) for i in open_issues
                  if L("status:in-progress") in i.labels)
    burn = []
    for ms in roadmap.milestones.values():
        st = ms.status_at(until)
        if st not in ("now", "lab"):
            continue
        mine = [i for i in issues.values() if not i.is_pr and roadmap.find(i.milestone, i.milestone_title) is ms]
        open_n = sum(1 for i in mine if i.state.upper() == "OPEN")
        closed_w = sum(1 for i in mine if i.closed_at and since <= i.closed_at <= until)
        burn.append((ms, st, open_n, closed_w, wip.get(id(ms), 0)))

    # ---------------------------------------------------------------- trackers spanning milestones
    trackers = []
    for i in open_issues:
        if not TRACKING.search(i.title) or (i.milestone is None and not i.milestone_title):
            continue
        children = {int(n) for n in CHILD_REF.findall(i.body or "")} - {i.number}
        elsewhere = Counter()
        for c in children:
            ci = issues.get(c)
            if ci and (ci.milestone or ci.milestone_title) and ci.milestone_title != i.milestone_title:
                elsewhere[ci.milestone_title or str(ci.milestone)] += 1
        if sum(elsewhere.values()) >= int(health.get("tracker_foreign_children", 3)):
            trackers.append((i, elsewhere))

    # ---------------------------------------------------------------- lab, stale PRs, branches
    lab_days = int(pol.get("lab_days", 14))
    def ms_status(i: rl.Issue) -> str | None:
        ms = roadmap.find(i.milestone, i.milestone_title)
        return ms.status_at(until) if ms else None

    lab_issues = [i for i in open_issues if ms_status(i) == "lab"]
    lab_stale = [i for i in lab_issues if i.updated_at and days_between(i.updated_at, now) > lab_days]
    stale_pr_days = int(pol.get("stale_pr_days", 7))
    open_prs = [p for p in prs if p.state == "OPEN"]
    stale_prs = [p for p in open_prs if days_between(p.created_at, now) > stale_pr_days]
    open_heads = {p.head for p in open_prs}
    stale_branch_days = int(pol.get("stale_branch_days", 14))
    branches = [] if args.no_branches else remote_branches()
    merged_heads = {p.head for p in prs if p.state in ("MERGED", "CLOSED")}
    dead = [(b, d) for b, d in branches if b not in open_heads and (b in merged_heads or days_between(d, now) > stale_branch_days)]

    rfcs = rfc_drift(prs, since, until)

    # ---------------------------------------------------------------- alarms
    alarms = []
    if n_merged and focus_pr < float(health.get("min_focus", 0.7)):
        alarms.append(("focus", f"focus {focus_pr:.0%} of merged PRs ({focus_lines:.0%} of lines) is below {float(health.get('min_focus', 0.7)):.0%}"))
    if len(buckets.get("no issue", [])) > int(health.get("max_no_issue", 0)):
        alarms.append(("no-issue", f"{len(buckets['no issue'])} merged PRs link no issue"))
    old_untriaged = [i for i in untriaged if days_between(i.created_at, now) > float(health.get("triage_days", 2))]
    if old_untriaged:
        alarms.append(("triage", f"{len(old_untriaged)} open issues without a milestone for more than {health.get('triage_days', 2)} days"))
    for ms, st, open_n, closed_w, w in burn:
        limit = ms.wip or int(pol.get("wip_per_milestone", 3))
        if w > limit:
            alarms.append(("wip", f"`{ms.key}`: {w} issues in progress (limit {limit})"))
    n_now = sum(1 for m in roadmap.milestones.values() if m.status_at(until) == "now")
    if n_now > int(pol.get("max_now", 2)):
        alarms.append(("max-now", f"{n_now} milestones are `now` (limit {pol.get('max_now', 2)})"))
    n_lab_active = sum(1 for i in lab_issues if L("status:in-progress") in i.labels)
    if n_lab_active > int(pol.get("max_lab", 2)):
        alarms.append(("lab", f"{n_lab_active} lab experiments in progress (limit {pol.get('max_lab', 2)})"))
    unapproved = [r for r in surface_rows if not r[2]]
    if unapproved:
        alarms.append(("surface", f"{len(unapproved)} merged PRs changed a surface without `{L('surface-approved')}`"))
    if rfcs:
        alarms.append(("rfc", f"{sum(len(h) for _, _, h in rfcs)} merged PRs implement RFCs that are still proposed"))
    if trackers:
        alarms.append(("trackers", f"{len(trackers)} tracking issues have children in other milestones"))
    p90 = statistics.quantiles(sizes, n=10)[-1] if len(sizes) >= 2 else (sizes[0] if sizes else 0)
    if p90 > int(health.get("max_p90_lines", 1500)):
        alarms.append(("p90", f"p90 PR size {p90:.0f} lines exceeds {health.get('max_p90_lines', 1500)}"))
    if len(stale_prs) > int(health.get("max_stale_prs", 3)):
        alarms.append(("stale-prs", f"{len(stale_prs)} open PRs older than {stale_pr_days} days"))
    if len(dead) > int(health.get("max_dead_branches", 10)):
        alarms.append(("branches", f"{len(dead)} remote branches with no open PR (merged, closed or idle > {stale_branch_days} days)"))

    # ---------------------------------------------------------------- report
    r = [f"# Roadmap health {since[:10]} → {until[:10]}", ""]
    r += ["## Alarms", ""] + ([f"- {a}" for _, a in alarms] or ["None."]) + [""]
    r += [f"<!-- alarm-ids: {','.join(sorted({k for k, _ in alarms}))} -->", ""]
    r += ["## Focus", "", f"{n_merged} PRs merged, {sum(sizes)} counted lines. Serving `now` or the interrupt lane: "
          f"**{focus_pr:.0%}** of PRs, **{focus_lines:.0%}** of lines.", "", "| Bucket | PRs | Lines | PRs |", "|---|---|---|---|"]
    for b in ["now", "interrupt", "next", "lab", "later", "icebox", "done", "untriaged", "unknown", "no issue"]:
        rows = buckets.get(b, [])
        if rows:
            nums = " ".join(f"#{p.number}" for p, _ in sorted(rows, key=lambda x: -x[1])[:12]) + (" …" if len(rows) > 12 else "")
            r.append(f"| {b} | {len(rows)} | {sum(l for _, l in rows)} | {nums} |")
    if sizes:
        q = statistics.quantiles(sizes, n=10) if len(sizes) >= 2 else [sizes[0]] * 9
        r += ["", f"PR size (counted lines): p50 {statistics.median(sizes):.0f}, p90 {q[-1]:.0f}, max {max(sizes)}."]
    r += ["", "## Active milestones", "", "| Milestone | Status | Open | Closed in window | In progress / limit | Exit |", "|---|---|---|---|---|---|"]
    for ms, st, open_n, closed_w, w in sorted(burn, key=lambda x: (x[0].rank, x[0].title)):
        r.append(f"| {ms.title} ({'#' + str(ms.number) if ms.number is not None else 'not created yet'}) | {st} | {open_n} | {closed_w} | {w} / {ms.wip or pol.get('wip_per_milestone', 3)} | {ms.exit} |")
    r += ["", f"Lab milestones with work in progress: {n_lab_active} (limit {pol.get('max_lab', 2)})."]
    r += ["", "## Triage", ""]
    r += [f"- #{i.number} ({days_between(i.created_at, now):.0f} d) {i.title}" for i in untriaged[:30]] or ["Every open issue has a milestone."]
    r += ["", "## Surface changes", ""]
    r += [f"- #{n}: {', '.join(s)} — {'approved' if a else '**not approved**'}" for n, s, a in surface_rows] or ["None."]
    r += ["", "## RFCs implemented while proposed", ""]
    r += [f"- `{path}` ({st}): {len(h)} merged PRs: " + " ".join(f"#{n}" for n in h) for path, st, h in rfcs] or ["None."]
    r += ["", "## Tracking issues spanning milestones", ""]
    r += [f"- #{i.number} ({i.milestone_title}): children in " + ", ".join(f"{k} ×{v}" for k, v in e.items()) for i, e in trackers] or ["None."]
    r += ["", "## Lab past its time box", ""]
    r += [f"- #{i.number} {i.title} (no update for {days_between(i.updated_at, now):.0f} d)" for i in lab_stale] or ["None."]
    r += ["", f"## Open PRs older than {stale_pr_days} days", ""]
    r += [f"- #{p.number} ({days_between(p.created_at, now):.0f} d) {p.title}" for p in stale_prs] or ["None."]
    if branches:
        r += ["", "## Remote branches without an open PR", "", f"{len(dead)} of {len(branches)}: "
              + ", ".join(f"`{b}`" for b, _ in sorted(dead)[:40]) + (" …" if len(dead) > 40 else "")]
    text = "\n".join(r) + "\n"

    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    if args.post_issue:
        previous = rl.gh_json(["issue", "view", str(args.post_issue), "--repo", args.repo, "--json", "body"])["body"] or ""
        m = re.search(r"<!-- alarm-ids: ([^>]*) -->", previous)
        before = set(filter(None, (m.group(1) if m else "").split(",")))
        new_kinds = sorted({k for k, _ in alarms} - before)
        subprocess.run(["gh", "issue", "edit", str(args.post_issue), "--repo", args.repo, "--body-file", "-"], input=text,
                       text=True, check=True)
        if new_kinds:
            lines = [f"- {a}" for k, a in alarms if k in new_kinds]
            subprocess.run(["gh", "issue", "comment", str(args.post_issue), "--repo", args.repo, "--body",
                            f"New alarm(s) for {since[:10]} → {until[:10]}:\n\n" + "\n".join(lines)], check=True)
    return 1 if args.fail_on_alarm and alarms else 0


if __name__ == "__main__":
    sys.exit(main())
