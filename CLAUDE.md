# fuzzybits: agent instructions

Project conventions (branches, commits, PRs, changelog, licensing) are in CONTRIBUTING.md and apply
here:

@CONTRIBUTING.md

## What this is

fuzzybits compiles probabilistic risk assessment (PRA) models into probabilistic circuits stored as
`.pcfb` files (`fzb compile`), then runs SYCL Monte Carlo on CPU or GPU (`fzb run`). It ships as a
C++ CLI, a Python wheel (`fuzzybits`) and a server (`fuzzybits-server`, `tools/server`).

Names: **SCRAM** is the GPL upstream this forked from. **mcSCRAM** is this project's former name
(renaming is #112; `scram-cli`, `mcscram` and `scram` identifiers remain). **fuzzybits** is the
project and **`fzb`** its CLI.

## Where things are

| Area | Path | Read first |
|---|---|---|
| IR (PRA model, probabilistic circuit, passes) | `src/ir` | `docs/IR-PRA-REQUIREMENTS.md`, `docs/IR-PC-REQUIREMENTS.md`, `docs/ir-pc-passes.md` |
| Readers and writers (FTP, AIGER, pcfb) | `src/io` | `docs/pc-pcfb.md`, `docs/ftp-input.md` |
| Compiler (PRA → PC lowering, nombrilo) | `src/compile` | `docs/pra-pc-ir-architecture.md`, `docs/nombrilo.md` |
| Monte Carlo engine and runtime | `src/mc`, `src/runtime` | `docs/architecture.md` (partly dated) |
| CLI, Python, server | `targets/`, `python/`, `tools/server` | `docs/python.md`, `tools/server/README.md` |
| SCRAM-derived legacy (`scram-cli`, OpenPSA MEF path) | top-level `src/`, `src/ext`, `src/expression`, `targets/scram` | `NOTICE` |
| Build, CI, images | `CMakePresets.json`, `toolchain.env`, `docker/`, `.github/` | `docs/build-variants.md`, `docs/ci-toolchain.md` |
| Benchmarks | `benchmarks/` | `docs/benchmarks.md` |

Full docs index: `docs/README.md`. Path-specific rules load from `.claude/rules/`.

## Build and test: `scripts/dev/ws.sh`

Use `scripts/dev/ws.sh` for every build and test (details: `scripts/dev/README.md`).

- Do not run `cmake` or `ctest` on this machine.
- Do not bind-mount the checkout into Docker.
- Do not invent other build helpers.
- The one exception is macOS Metal work: see `.claude/rules/macos-metal.md`.

- Run it from the checkout you are changing. Name the workspace after the branch, e.g.
  `feat-pcfb-reader` (`[a-z0-9][a-z0-9_.-]*`).
- Build host: `ws.sh` uses `FZB_WS_CONTEXT`, else the current Docker context. If
  `scripts/dev/ws.sh doctor` fails, ask the user which host to use. Never write host names into
  tracked files. Personal values go in `.claude/settings.local.json` (`"env"`) or
  `CLAUDE.local.md`, both gitignored.
- First build: `scripts/dev/ws.sh configure <ws> --tail 3`.
- Build only what you need: `scripts/dev/ws.sh build <ws> <target>... --tail 5`. To list targets:
  `scripts/dev/ws.sh run <ws> --tail 200 -- 'cmake --build --preset generic --target help'`.
- Tests: `scripts/dev/ws.sh test <ws> '<ctest regex>' --tail 40`.
- Anything else (pytest, scripts): `scripts/dev/ws.sh run <ws> --tail 40 -- '<command>'`.
- After deleting or renaming files, or after a rebase, add `--clean`. After toolchain or CMake
  cache changes, add `--fresh`.
- Always pass `--tail`. The last stderr line names the full log: read or grep the log instead of
  re-running.
- Commands are serialised per workspace. For parallel work, use a second workspace name.
- When the PR is merged or abandoned: `scripts/dev/ws.sh down <ws>`.
- A local hook formats each C++ file you edit with LLVM 20's clang-format, when it is installed.
  Before you push, run `scripts/ci/clang-format.sh`; CI fails on any difference.

## Scope rules for agents

Full rules: `docs/planning.md` (§8 is for agents). The roadmap is `ROADMAP.md` and
`.github/roadmap.toml`.

- **Starting work.**
  - Work only on an issue the user names that is `status:ready` in a `now` milestone, or on a lab
    charter.
  - If the issue is in a `next`, `later` or `icebox` milestone, or has no milestone, say so and
    ask before starting.
- **Stay inside the issue.**
  - Do not change areas its milestone does not list, or frozen areas (`hull`, `exact-kc`).
  - Do not add a directory under `src/`, `tools/`, `targets/` or `python/`.
  - Do not change a surface (default, CLI or API, `.pcfb`, dependency, workflow, governance)
    unless the issue says to. Check with `python3 scripts/ci/scope_check.py --pr <N>`.
- **Scope that grows.** Stop at the issue's boundary, file one triage issue for the rest, and tell
  the user. Do not widen the PR.
- **Filing issues.**
  - Use `status:triage` and no milestone, at most 3 per session without asking.
  - Never put a private model's identifier in a title, branch name, commit message or log.
- **Never:**
  - merge a PR, or push to `develop`;
  - apply `status:ready`, `scope-exception`, `size-exception`, `surface-approved`,
    `wrong-answer`, `security` or `release-blocker`;
  - create labels or milestones;
  - edit `.github/roadmap.toml`, `docs/planning.md`, this file or `.claude/` unless the user asks
    in this session.

  `.claude/hooks/scope-guard.sh` blocks these commands.
- **RFCs.** Code for a `proposed` RFC goes in `lab/<topic>/`, never in core paths.
- **Benchmarks.** A benchmark finding never justifies a core change by itself. File a triage issue.

## Working rules

- **Run outputs.** Never write them into the repository root or `input/`. `fzb`/`scram-cli` runs
  and benchmarks write `.pcfb`, `convergence.csv` and similar to the current directory, so run them
  in the workspace or a temporary directory.
- **Tests.** Report test results with the exact `ws.sh` command, as in a PR's **Tests** section.
- **Pins.** `toolchain.env` changes the toolchain image hash and triggers a rebuild in CI. Change it
  only when the task is about the toolchain.
- **SCRAM.** New code must not include SCRAM-derived headers. Those are the files tagged
  `SPDX-License-Identifier: GPL-3.0-or-later`: mostly top-level `src/`, `src/ext`,
  `src/expression` and `targets/scram`. Check with `python3 scripts/ci/scram_provenance.py`.
- **Private fixtures.** Pass them only when the task asks for them (`ws.sh ... --private=<dir>`).
  They are the `update = none` submodule `extern/private-fixtures`; never fetch it in CI or change
  its `update = none`.
  Report aggregates only: never identifiers, names or values.
- **Docs.** `docs/architecture.md` predates the rename and is partly out of date. When it disagrees
  with the code or a newer doc, trust the code and say so.
