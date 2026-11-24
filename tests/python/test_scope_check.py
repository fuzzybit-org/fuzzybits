"""Scope check and roadmap model (docs/planning.md): rules on synthetic PRs against the real roadmap."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))

import roadmap_lib as rl  # noqa: E402
import scope_check as sc  # noqa: E402


@pytest.fixture(scope="module")
def roadmap() -> rl.Roadmap:
    return rl.load_roadmap(ROOT / ".github" / "roadmap.toml")


def ms(roadmap: rl.Roadmap, key: str) -> rl.Milestone:
    return next(m for m in roadmap.milestones.values() if m.key == key)


def issue(n: int, milestone: rl.Milestone | None, labels=()) -> rl.Issue:
    return rl.Issue(n, f"issue {n}", "OPEN", milestone.number if milestone else None,
                    milestone.title if milestone else "", set(labels), "2026-11-24T00:00:00Z", None)


def pr(files, linked=(1,), labels=(), body="", author="someone") -> rl.PullRequest:
    changes = [rl.FileChange(f[0], f[1], 0, f[2] if len(f) > 2 else "modified", f[3] if len(f) > 3 else None) for f in files]
    return rl.PullRequest(99, "t", body, author, set(labels), "feat/x", "develop", "OPEN", "2026-11-25T00:00:00Z", None,
                          changes, list(linked))


def rules(findings, blocking_only=True):
    return {f.rule for f in findings if (f.blocking if blocking_only else True)}


def test_roadmap_loads_and_is_consistent(roadmap):
    assert sum(1 for m in roadmap.milestones.values() if m.status == "now") <= int(roadmap.policy["max_now"])
    names = {a.name for a in roadmap.areas}
    for m in roadmap.milestones.values():
        assert set(m.areas) <= names | {"*"}, m.key
    assert roadmap.area("hull").frozen


@pytest.mark.parametrize("path, area", [
    ("src/ir/pc/hull/check.cc", "hull"),
    ("src/ir/pc/passes/propagate.cpp", "passes"),
    ("src/ir/pc/builder.h", "ir-pc"),
    ("src/io/ftp/read.cpp", "io-ftp"),
    ("src/compile/ftp/compile.cpp", "io-ftp"),
    ("src/compile/pra_to_pc/lower.cpp", "compile"),
    ("src/preprocessor.cc", "scram-legacy"),
    ("tests/mc/x.cc", "runtime"),
    ("tests/python/test_cli.py", "python"),
    ("tools/server/webui/src/App.tsx", "webui"),
    ("docs/rfc/monotone-hull-queries.md", "docs"),
    ("lab/hull/README.md", "lab"),
    (".github/roadmap.toml", "governance"),
    (".github/workflows/ci.yml", "ci"),
])
def test_area_map(roadmap, path, area):
    assert roadmap.area_of(path) == area


def test_in_scope_pr_is_clean(roadmap):
    findings, _ = sc.check(roadmap, pr([("src/mc/kernel/basic_event.h", 40)]), {1: issue(1, ms(roadmap, "correct"))})
    assert rules(findings) == set()


def test_no_issue_and_trivial(roadmap):
    assert "scope.no-issue" in rules(sc.check(roadmap, pr([("README.md", 3)], linked=()), {})[0])
    assert rules(sc.check(roadmap, pr([("README.md", 3)], linked=(), labels=("trivial",)), {})[0]) == set()
    big = pr([("src/mc/x.h", 200)], linked=(), labels=("trivial",))
    assert "scope.no-issue" in rules(sc.check(roadmap, big, {})[0])


def test_untriaged_and_not_active(roadmap):
    assert "scope.untriaged" in rules(sc.check(roadmap, pr([("src/mc/x.h", 5)]), {1: issue(1, None)})[0])
    nxt = {1: issue(1, ms(roadmap, "speed"))}
    assert "scope.not-active" in rules(sc.check(roadmap, pr([("src/mc/x.h", 5)]), nxt)[0])
    accepted = pr([("src/mc/x.h", 5)], labels=("scope-exception",))
    assert "scope.not-active" not in rules(sc.check(roadmap, accepted, nxt)[0])
    ice = {1: issue(1, ms(roadmap, "icebox"))}
    assert "scope.not-active" in rules(sc.check(roadmap, pr([("src/mc/x.h", 5)]), ice)[0])


def test_interrupt_lane(roadmap):
    nxt = {1: issue(1, ms(roadmap, "ship"), labels=("security",))}
    findings, facts = sc.check(roadmap, pr([("src/mc/x.h", 5)]), nxt)
    assert facts["interrupt"] and rules(findings) == set()
    frozen = pr([("src/ir/pc/hull/check.cc", 5)])
    assert "scope.frozen-area" in rules(sc.check(roadmap, frozen, nxt)[0])


def test_area_and_frozen_area(roadmap):
    speed = {1: issue(1, ms(roadmap, "speed"))}
    p = pr([("tools/server/src/fuzzybits_server/app.py", 5)], labels=("scope-exception",))
    assert "scope.area" not in rules(sc.check(roadmap, p, speed)[0])
    assert "scope.area" in rules(sc.check(roadmap, pr([("tools/server/src/fuzzybits_server/app.py", 5)]),
                                          {1: issue(1, ms(roadmap, "evidence"), labels=())})[0], blocking_only=False)
    correct = {1: issue(1, ms(roadmap, "correct"))}
    assert "scope.frozen-area" in rules(sc.check(roadmap, pr([("src/ir/pc/hull/check.cc", 5)]), correct)[0])


def test_lab_work_stays_in_lab(roadmap):
    lab = {1: issue(1, ms(roadmap, "lab"))}
    assert rules(sc.check(roadmap, pr([("lab/hull/check.cc", 300, "added")]), lab)[0]) == set()
    assert "scope.lab-in-core" in rules(sc.check(roadmap, pr([("src/io/ftp/read.cpp", 5)]), lab)[0])


def test_core_never_includes_lab(roadmap):
    p = pr([("src/compile/ftp/compile.cpp", 1, "modified", ['#include "lab/hull/check.h"'])], labels=("scope-exception",))
    findings, _ = sc.check(roadmap, p, {1: issue(1, ms(roadmap, "correct"))})
    assert "scope.lab-include" in rules(findings)


def test_size(roadmap):
    correct = {1: issue(1, ms(roadmap, "correct"))}
    assert "scope.size" in rules(sc.check(roadmap, pr([("src/mc/x.h", 2500)]), correct)[0])
    assert "scope.size" not in rules(sc.check(roadmap, pr([("src/mc/x.h", 2500)], labels=("size-exception",)), correct)[0])
    assert "scope.size" not in rules(sc.check(roadmap, pr([("input/ftp/big.ftp", 9000)]), correct)[0], blocking_only=False)


def test_new_directory(roadmap):
    correct = {1: issue(1, ms(roadmap, "correct"))}
    p = pr([("src/kc/forest.cc", 10, "added")])
    findings, _ = sc.check(roadmap, p, correct, dir_exists=lambda d: False)
    assert "scope.new-directory" in rules(findings)
    tests_dir = pr([("src/compile/tests/new_test.cc", 10, "added")])
    assert "scope.new-directory" not in rules(sc.check(roadmap, tests_dir, correct, dir_exists=lambda d: False)[0])


def test_surfaces(roadmap):
    correct = {1: issue(1, ms(roadmap, "correct"))}
    cli = pr([("targets/fzb/main.cpp", 5)])
    assert {"surface.cli", "surface.changelog"} <= rules(sc.check(roadmap, cli, correct)[0])
    approved = pr([("targets/fzb/main.cpp", 5), ("CHANGELOG.md", 2)], labels=("surface-approved",))
    assert rules(sc.check(roadmap, approved, correct)[0]) == set()
    pin = pr([("toolchain.env", 1, "modified", ["PCFB_GIT_TAG=abc"])], labels=("scope-exception",))
    assert {"surface.format", "surface.dependency"} <= rules(sc.check(roadmap, pin, correct)[0])
    declared = pr([("src/compile/ftp/compile.cpp", 3)], body="- [x] changes a user-visible default")
    assert "surface.default" in rules(sc.check(roadmap, declared, correct)[0])
    unticked = pr([("src/compile/ftp/compile.cpp", 3)], body="- [ ] changes a user-visible default")
    assert "surface.default" not in rules(sc.check(roadmap, unticked, correct)[0])


def test_exempt_author(roadmap):
    findings, facts = sc.check(roadmap, pr([("uv.lock", 10)], linked=(), author="dependabot[bot]"), {})
    assert facts.get("exempt") and findings == []


def test_linked_numbers():
    body = "Closes #12\nPart of #7\nsee #99 for context\n- Refs: #5"
    assert rl.linked_numbers("fix(io): thing (#3)", body, [12]) == [12, 7, 5, 3]


def test_status_history(roadmap):
    p2 = ms(roadmap, "phase2")
    assert p2.status_at("2026-11-20T00:00:00Z") == "now"
    assert p2.status_at("2026-11-26T00:00:00Z") == "done"
