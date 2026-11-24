# Contributing to fuzzybits

Issues and pull requests: https://github.com/fuzzybit-org/fuzzybits. These conventions apply to
people and to coding agents alike; agents also read [CLAUDE.md](CLAUDE.md).

## Planning and scope

Work follows [ROADMAP.md](ROADMAP.md). The rules are in [docs/planning.md](docs/planning.md); the
hard ones:

- **Starting work.** Work starts from an issue marked `status:ready` in a `now` milestone, or from
  a lab charter. New issues are `status:triage` until the maintainer sets their milestone.
- **One PR, one issue, one concern.** Name the issue with `Closes #N` or `Part of #N`. A fix does
  not carry a feature, a pin bump or a refactor.
- **Scope check.** The `scope` check compares each PR with `.github/roadmap.toml`:
  - the milestone's status and areas, including frozen areas and new directories;
  - size: 800 counted lines is the review budget, 2000 needs `size-exception`;
  - surfaces: defaults, CLI and API, `.pcfb`, dependencies, workflows, governance. These need
    `surface-approved` and, for user-visible ones, a CHANGELOG entry.
- **Who applies labels.** The maintainer applies the exception and approval labels.
- **Defaults** change only in a PR of their own. Whatever `fzb compile` writes by default,
  `fzb run` accepts.
- **RFCs.** Code for an RFC that is still `proposed` lives in [`lab/`](lab/README.md), not in
  core ([docs/rfc/README.md](docs/rfc/README.md)).
- **Benchmarks** measure. A product change they suggest becomes a triage issue.

## Branches and pull requests

- Branch from `develop` as `<type>/<short-name>` (for example `feat/pcfb-reader`,
  `chore/dev-workspaces`). Pull requests target `develop`.
- **Experiments** use `lab/<topic>` branches or `lab/<topic>/` directories. Never merge them into
  core paths.
- Rebase on `origin/develop` before review. Never rebase commits that others have pulled.
- Every commit compiles and passes its tests. Split work over the review budget into a stack of
  PRs.
- PR description sections, as in [the template](.github/pull_request_template.md):
  - **Summary**;
  - **Changes**, by file or area;
  - **Decisions**: the trade-offs and alternatives rejected;
  - **Tests**: the exact commands you ran and their results;
  - **Scope**.
- **Merging.**
  - Squash only, and delete the branch.
  - No direct pushes to `develop`.
  - The maintainer merges after reading the diff.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org): `<type>(<scope>): <summary> (#<issue>)`.
The squash merge appends the PR number, e.g. `feat(io): .pcfb reader to ir::pc (#219) (#231)`.

- Types in use: `feat`, `fix`, `docs`, `build`, `ci`, `chore`, `bench`, `refactor`, `test`.
- Scopes are the area touched: `io`, `ir`, `compile`, `cli`, `runtime`, `hip`, `bench`, `dev`,
  `release`, `server`, `python`, …
- Summary in the imperative, lower case, no trailing period.

## Building and testing

Build and test with [`scripts/dev/ws.sh`](scripts/dev/README.md). It runs the checkout inside the
pinned CI toolchain image on a Docker host, so results match CI. This is the preferred method for
local development, for people and agents alike.

- macOS Metal work, and only that, uses `scripts/dev/macos-dev.sh`
  ([build-variants.md](docs/build-variants.md#macos-local-development)).
- Plain `cmake --preset …` on a host with the pinned toolchain also works
  ([build-variants.md](docs/build-variants.md)), but it is not what CI runs.
- CI ([`.github/workflows`](.github/workflows)) is the authority. A PR that changes behaviour adds or
  updates tests: C++ tests next to the code (`src/*/tests`, `tests/`), Python tests in
  `tests/python`.

## Changes, docs and licensing

- User-visible changes go under `## Unreleased` in [CHANGELOG.md](CHANGELOG.md). Breaking changes
  go under **Breaking changes**, with the way to opt out.
- Docs state only what CI or a reproducible command proves. Plans and vision go in
  [ROADMAP.md](ROADMAP.md) and issues. The docs index is [docs/README.md](docs/README.md).
- C++ follows [`.clang-format`](.clang-format) and [`.clang-tidy`](.clang-tidy).
  - CI checks formatting with clang-format 20.1.8 (`scripts/ci/clang-format.sh`).
  - To fix, run `scripts/ci/clang-format.sh --fix`. It uses `$CLANG_FORMAT` from
    `macos-dev.sh` when set, otherwise `uvx`.
  - Commits that only reformat go in [`.git-blame-ignore-revs`](.git-blame-ignore-revs). Run
    `git config blame.ignoreRevsFile .git-blame-ignore-revs` once.
- fuzzybits is AGPL-3.0 and still contains SCRAM-derived GPL-3.0-or-later code ([NOTICE](NOTICE)).
  - Do not add new includes of SCRAM-derived headers.
  - Do not copy the upstream copyright into new files.
  - `scripts/ci/scram_provenance.py` (run in CI) enforces both. After removing SCRAM code, run it
    with `--update-baseline` and commit the result.
  - Retiring SCRAM is tracked in [#144](https://github.com/fuzzybit-org/fuzzybits/issues/144).

### Rebasing a branch across the clang-format commit

`b173731d` reformatted every C++ source. To move a branch that started before it:

```bash
git rebase b173731d^       # 1. resolve real conflicts by hand, as usual
git rebase -X theirs --exec 'scripts/ci/clang-format.sh --fix && git add -u && git commit -q --amend --no-edit --allow-empty' b173731d
```

Step 2 is safe only after step 1. `b173731d` changes nothing but C++ layout, so the only conflicts
left are formatting, and each replayed commit is reformatted. Run step 2 without step 1 and
`-X theirs` silently drops `develop`'s side of any real conflict.

## Private fixtures

Some tests and benchmarks use private models (`--private` in `ws.sh`,
`FUZZYBITS_PRIVATE_FIXTURES_DIR`). Never print, commit or paste their identifiers or values. Report
aggregates only.

The private repositories are submodules marked `update = none` in `.gitmodules`:

| Path | Repository |
|---|---|
| `extern/private-fixtures` | the private models and their cards |
| `extern/pra-solver-images` | wine images for the licensed vendor solvers |

- Each has `update = none` in `.gitmodules`, so `git submodule update --init --recursive` and CI checkouts skip
  them.
- With ssh access to the repositories, fetch one explicitly:

  ```bash
  git submodule update --init --checkout extern/private-fixtures
  ```

- `ws.sh` never syncs them into a workspace; pass `--private=extern/private-fixtures` instead.
- `scripts/ci/private-submodules.sh` (the `hygiene` job) checks:
  - every other submodule is readable anonymously, so a new private one cannot be added without
    `update = none`;
  - no private submodule is populated in CI;
  - the job token cannot read any of them;
  - on GitHub-hosted runners, where fork pull requests run, the runner cannot read them over ssh, https or
    `gh`;
  - on self-hosted runners, whose accounts are trusted and may read them, the job-started hook that keeps other
    people's jobs off the host is configured
    ([ci-toolchain.md](docs/ci-toolchain.md#self-hosted-runner-job-guard)).
