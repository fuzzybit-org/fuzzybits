# Planning and scope

How work enters `develop`, how it stays on the roadmap, where experiments go, and how drift is detected.
The roadmap itself is [ROADMAP.md](../ROADMAP.md). Its machine-readable form is
[`.github/roadmap.toml`](../.github/roadmap.toml), which two scripts read:

- [`scripts/ci/scope_check.py`](../scripts/ci/scope_check.py) checks every PR;
- [`scripts/ci/roadmap_health.py`](../scripts/ci/roadmap_health.py) reports daily.

These rules apply to people and coding agents alike. Agents have extra rules in §8.

## Why

Scope grows into `develop` through paths that no single PR shows:

- **Trackers.** A tracking issue lists children in other milestones and schedules work there.
- **Research.** An experiment changes core contracts (IR fields, pass semantics, format kinds) before anyone
  decided it should ship.
- **RFCs.** Code lands that implements an RFC still under discussion.
- **Defaults.** A default changes inside a feature PR, and nothing checks that the runtime accepts what the
  compiler now writes.
- **Formats.** A format version moves ahead of the runtime that reads it.
- **Volume.** PRs merge faster than they can be read, so none of the above is noticed until users hit it.

Each rule below closes one of these paths. Each names how it is enforced and how a breach is detected (§7).

## 1. Goals, ranked

When two pieces of work compete, the higher goal wins. Interrupts (§2.4) come before all of them.

1. **Correct.** Every input fuzzybits accepts gets a correct estimate, or is refused before sampling with a
   diagnostic. No silent wrong answers, and no default whose output the runtime rejects.
2. **Spine.** One pipeline: `ir::pra` → `ir::pc` → `.pcfb` → an engine that consumes `ir::pc`, with no
   SCRAM-derived code, then MIT ([#144](https://github.com/fuzzybit-org/fuzzybits/issues/144)).
3. **Speed.** Engine performance improvements, measured by the bench gate, at unchanged accuracy.
4. **Reach.** New inputs, semantics and integrations: FTP plant models, event trees, frequencies, converters,
   inverse-canopy.
5. **Ship.** Installable and operable: wheels, images, the server, CI, security.
6. **Usable.** CLI and API ergonomics, run control, progress, the web UI.
7. **Evidence.** Benchmark datasets, reference values, cross-tool comparisons.

Research is not a goal. It is what the lab is for (§6).

## 2. Structure

### 2.1 Milestones

A milestone is the unit of planning. Each one has:

- one primary goal;
- a **status**;
- an **exit**: the command or measurement that closes it;
- a **rank**;
- the **areas** its PRs may change (§4.2).

GitHub allows one milestone per issue. Goals are labels (`goal:*`), and an issue may carry several.

| Status | Meaning | Work allowed |
|---|---|---|
| `now` | in progress; at most **2** at once | issues labelled `status:ready` |
| `next` | queued, in rank order | none; pulling an issue forward needs `scope-exception` |
| `later` | planned, not scheduled | none |
| `lab` | experiments (§6) | in `lab/` only |
| `icebox` | ideas | none; starting one needs a lab charter or a product issue |
| `done` | closed | none |

A milestone moves from `next` to `now` only when a `now` milestone exits, or when the maintainer demotes one.
Either way, the same PR updates `roadmap.toml` and ROADMAP.md.

### 2.2 Work in progress

- At most **3** issues per `now` milestone carry `status:in-progress` (`wip` in `roadmap.toml`).
- At most **2** lab experiments are in progress.
- Labels: `status:triage` (new, unreviewed), `status:ready` (accepted, may start), `status:in-progress`,
  `status:blocked` (with "Blocked by #N" in the body).

### 2.3 The milestone is the tracker

A tracking issue may list only issues of its own milestone. A dependency on another milestone is written on the
child as "Blocked by #N", never as a row in a tracker. A tracker that reaches into other milestones schedules
work that no milestone owns. The health report flags any tracker with three or more children outside its
milestone.

### 2.4 Interrupt lane

Three labels let an issue land from any milestone and any area, except frozen areas:

- `security`
- `wrong-answer`: a result that is wrong with no diagnostic;
- `release-blocker`: a red `develop`, or a broken release.

They do not count against the WIP limits. Use them only for what they say.

## 3. Intake and triage

1. **New issues** start as `status:triage`. Agents file without a milestone; people may propose one.
2. **The maintainer triages within 2 days.** Triage sets the milestone and goal labels, then does one of:
   - marks the issue `status:ready`;
   - moves it to `icebox`;
   - closes it.
3. **Scope that grows during work** becomes a new triage issue. The current PR does not grow with it.
4. **Proposals and experiments** go to `icebox`, or to `lab` with a charter (§6). They never go to a product
   milestone because they would use its code.

Enforcement: the issue templates require a goal, a measurable acceptance criterion and the surfaces touched. The
scope check fails a PR whose issue has no milestone (`scope.untriaged`). The health report lists open issues
without a milestone, with their age.

## 4. Pull requests

### 4.1 One issue, one concern

- Every PR names its issue with `Closes #N` or `Part of #N`.
- A change of at most 50 lines with the `trivial` label needs no issue.
- PRs labelled `release` are exempt from the scope check.
- A PR does one thing: a fix does not carry a feature, a pin bump or a refactor.

### 4.2 Areas

`roadmap.toml` maps every path to an area (`io-ftp`, `runtime`, `hull`, ...); the first match wins.

- A PR may change only the areas its milestone lists, plus `docs`, `changelog` and `tests`.
- A **frozen** area is code that stays in place but takes no new work. Changing it needs `scope-exception`, even
  for an interrupt. `roadmap.toml` marks them `frozen = true`.
- A new directory under a subsystem root (`src/*`, `src/ir/*`, `src/ir/pc/*`, `src/io/*`, `src/compile/*`,
  `tools/*`, ...) is a new subsystem and needs `scope-exception`. Directories under `lab/` are exempt.

### 4.3 Surfaces

A surface is something users, packagers or agents depend on.

| Surface | What counts | CHANGELOG entry |
|---|---|---|
| `cli` | anything under `targets/fzb`, `python/fuzzybits/_cli.py` | yes |
| `python-api` | `__init__.py`, or a top-level `def`/`class` in `_api.py` | yes |
| `server-api` | route decorators, `config.py` | yes |
| `format` | the pcfb pin, schema version, query kinds | yes |
| `default` | declared in the PR template | yes, under **Breaking changes** with the opt-out |
| `dependency` | toolchain pins, FetchContent, Python/npm dependencies, Docker base images and packages | no |
| `ci-security` | `runs-on`, `permissions`, triggers, secrets in workflows | no |
| `governance` | `roadmap.toml`, this document, CONTRIBUTING.md, CLAUDE.md, `.claude/`, templates, these scripts | no |

Every surface change needs the `surface-approved` label from the maintainer.

### 4.4 Size

The size counts changed lines, excluding fixtures, lockfiles, model inputs and generated data.

- **800 lines** is the review budget. Above it, the check warns.
- **2000 lines** needs `size-exception`, with a reason in the description.

Split large work into a stack of PRs that each compile and pass their tests (CONTRIBUTING.md). Inference, not
measured here: review quality drops sharply past a few hundred lines per sitting, and these limits are set so a
PR can be read in one.

### 4.5 Merging

- **Method:** squash only. Each commit on `develop` is then a state CI tested, and its subject is the PR title
  plus `(#PR)`. Branches are deleted on merge.
  - Rebase merges would land commits CI never tested on their own, without the PR number.
  - Merge commits would land WIP commits and make `--first-parent` necessary everywhere.
- **Linear history:** required on `develop` by the `linear-history` ruleset. Only the repository admin can
  bypass it, and only through a PR.
- **Up to date:** a branch must contain the latest `develop` before it merges (required checks are `strict`),
  so the checks run on what will land.
- **Exceptions**, both through a PR:
  - a history import from another repository merges as a merge commit, labelled `history-import`, with merge
    commits enabled for that PR only;
  - a mechanical reformat or rename is a PR of its own, so a follow-up can add its squash commit to
    `.git-blame-ignore-revs`.
- **Direct pushes:** none to `develop`, including by admins. A direct push skips every check, including this
  one.
- **Who merges:** the maintainer. Agents never merge (§8).
- **Before merging:**
  - the required checks are green, and the scope check reports nothing blocking;
  - bench checks that ran have passed or have been explained in the PR;
  - the description's **Tests** section lists what was run.
- **Read the diff first.** The scope check finds where a PR goes, not whether its content is right. That
  needs a reader.

## 5. Defaults, formats, RFCs and benchmarks

### 5.1 RFC lifecycle

RFCs live in `docs/rfc/` ([README](rfc/README.md)). The `Status:` line is machine-read:

- `proposed`: under discussion. Implementations live in `lab/`, behind an opt-in. No core change may be justified
  by it alone.
- `accepted`: the maintainer has agreed. Product issues may now be filed in a milestone.
- `implemented`, `rejected` or `withdrawn`: final.

The health report lists merged PRs that cite a `proposed` RFC by file name or by issue.

### 5.2 Formats

- A `.pcfb` version or query-kind change needs an accepted RFC.
- The runtime evaluates a kind in the same release in which `fzb compile` can write it.
- A kind the runtime refuses is never a default output.

### 5.3 Defaults

- **A default changes in a PR of its own.** That PR declares the change in the template, carries
  `surface-approved`, and adds a **Breaking changes** entry with the opt-out.
- **Default-loop invariant:** whatever `fzb compile` writes with default options, `fzb run` accepts. A
  default-loop CI job checks this on the fixture corpus: Aralia, `input/ftp/conformance`, `input/ftp/synthetic`.
  Each model must run to |z| ≤ 4 against its reference, or be refused at compile with a diagnostic listed in its
  expected file. Adding this job is part of the Correct milestone.

### 5.4 Benchmarks measure; they do not decide

- An Evidence issue may not list product issues as its children.
- A benchmark finding that needs a product change becomes a triage issue, judged on product terms.
- Whether a benchmark model must go through the product readers ("no dataset shim") is a product decision. It is
  made in triage, not in a benchmark tracker.

## 6. The lab

The lab is where ideas are tried without committing `develop` to them.

### 6.1 Where

- **`lab/<topic>/`** holds code that should keep building against `develop`. It contains `README.md` (the
  charter), sources, tests and `FINDINGS.md`. It is built only with `-DFUZZYBITS_LAB=ON`, and never into wheels,
  images or the default `fzb`. Python prototypes and notebooks go here too.
- **`lab/<topic>` branches** are for throwaway work. Delete or tag them when the experiment ends.

### 6.2 Rules

- **Imports go one way.** Lab code may use core code. Core code never includes or imports lab code; the scope
  check fails this (`scope.lab-include`) and no label accepts it.
- **Lab work changes only `lab/`** (`scope.lab-in-core`). If an experiment needs a core change, that change is a
  separate product issue, justified on product terms. Otherwise it waits for graduation. Duplicating a pass inside
  `lab/` is cheaper than bending core for an unproven idea.
- **What the lab never changes:** defaults, formats, `toolchain.env`, required CI jobs and release artifacts.

### 6.3 Charter, time box, end

A lab issue in the `Lab` milestone, labelled `lab:<topic>`, states:

- the hypothesis;
- the measurement and the data it runs on;
- the kill criterion;
- a time box of at most 2 weeks, which may be extended once;
- the owner.

At the end the experiment does one of three things:

1. **Graduates** (§6.4).
2. **Is extended once.**
3. **Is archived.** The findings go into `FINDINGS.md` or the issue, the code is tagged `lab-archive/<topic>`, and
   the directory or branch is deleted.

The health report lists lab issues with no update for 14 days.

### 6.4 Graduation

All five must hold:

1. an accepted RFC, if the experiment changes semantics or a format;
2. a measured benefit on the benchmark dataset, or a named user need;
3. a product milestone with room within the limits;
4. no bundled default change (§5.3);
5. a plan that deletes the lab copy.

### 6.5 Not lab

An optimisation with bit-identical (or statistically equivalent) output and a bench A/B decision rule is
ordinary Speed work, for example #16, #19 and #164.

## 7. Detection

| Signal | Where | Threshold (`roadmap.toml`) |
|---|---|---|
| **Scope findings** on the PR | `scope` workflow: the base commit's script and roadmap, read-only token, hosted runner | any blocking finding |
| **Focus:** share of merged PRs and lines serving `now` or the interrupt lane | `roadmap-health`, daily | < 70 % |
| **Drift:** merged PRs whose issue is `next`, `later`, `lab`, `icebox` or untriaged, or that link no issue | daily | any PR without an issue |
| **Triage age** | daily | > 2 days |
| **WIP** per `now` milestone; lab experiments | daily | > `wip`, > 2 |
| **Unapproved surfaces** | daily | any |
| **RFC drift:** merged PRs citing a `proposed` RFC | daily | any |
| **Trackers** with children in other milestones | daily | ≥ 3 children |
| **PR size** p90 | daily | > 1500 lines |
| **Stale** open PRs and dead branches | daily | > 3 PRs older than 7 d; > 10 branches |

**The scope check** starts in `warn` mode: findings show in the job summary and as annotations, and the check
passes. After one week with no false positives that labels could not resolve, set `policy.mode = "enforce"` and
make `scope` a required check.

**The health report** replaces the body of a pinned issue (repository variable `ROADMAP_HEALTH_ISSUE`). It
comments only when a new kind of alarm appears, so a notification means something changed. The maintainer reads
it weekly, and the same day it comments. Every alarm ends in one of three ways:

- the roadmap changes (re-rank or move);
- the work stops;
- an exception label is applied, with a reason.

**Backtest.** Both scripts run offline on `gh` dumps, against any `roadmap.toml`:

- `scope_check.py --backtest-prs prs.json --backtest-issues issues.json --milestones milestones.json --patches DIR --git`;
- `roadmap_health.py --prs ... --issues ... --milestones ... --since YYYY-MM-DD --until YYYY-MM-DD`.

Running them against a roadmap that models an earlier plan shows when drift would have been reported. Usage and
dump formats are in each script's docstring. Use this to check a rule change before merging it.

## 8. Coding agents

Agents follow everything above, plus:

1. **Starting work.**
   - Start only from an issue the user names that is `status:ready` in a `now` milestone, or a lab charter.
   - Do not move issues between milestones.
2. **Filing issues.**
   - File with `status:triage` and no milestone.
   - File at most **3** per session without asking. A burst of agent-filed issues is a plan nobody reviewed.
   - Never put a private model's identifier in a title, branch name, commit or log.
3. **Labels and milestones.**
   - Never apply `status:ready`, `scope-exception`, `size-exception`, `surface-approved` or an interrupt label.
   - Never create labels or milestones.
4. **Never merge,** and never push to `develop`.
5. **Do not change governance surfaces** (`roadmap.toml`, this document, CONTRIBUTING.md, CLAUDE.md, `.claude/`,
   templates, the scope and health scripts) unless the user asks in this session.
6. **When the work outgrows its issue,** stop at the issue's boundary, file a triage issue for the rest, and say so.

`.claude/settings.json` denies `gh pr merge`. `.claude/hooks/scope-guard.sh` blocks the label, milestone and push
commands above. Both apply only to Claude Code sessions in this repository.

**Identity.** While agents push and open PRs with the maintainer's account, GitHub cannot tell their labels or
PRs from the maintainer's. Required reviews would then block every PR, because an author cannot approve their own.
A separate identity for agents (a machine user or a GitHub App without admin rights) makes two things possible:

- required reviews (one approval from the maintainer);
- label gates that verify who applied the label.

Without it, the approval labels rest on the agent rules and the local hook.

## 9. Limits

- **Areas are paths.** A PR that implements hull semantics inside `src/compile/pra_to_pc` counts as `compile`.
  Areas catch new places, not new meanings.
- **Line-based surfaces are heuristics** on added lines: a removed option or a changed default value is not seen.
  The `default` surface relies on the PR template, so honesty is still required.
- **The labels may come from the same account as the agents** (§8, Identity).
- **An exception label is a bypass.** The health report counts them, but nothing limits them.
- **Merge latency is not enforced.** GitHub has no minimum-age rule. Required reviews from a second identity are
  the real fix.
- **The backtest is approximate.** It uses issue milestones and bodies as they are when the dump is taken, not as
  they were when each PR merged.

## 10. Changing this policy

The policy is itself a `governance` surface, so a change to it needs `surface-approved`. Change it when a rule
costs more than the drift it prevents: the health report shows both, as the exceptions taken and the alarms
raised.
