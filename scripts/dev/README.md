# Remote build workspaces (`scripts/dev/ws.sh`)

`ws.sh` builds and tests a checkout inside the pinned toolchain image on a Docker host. Each
*workspace* is one long-lived container; the checkout's files are copied into its `/src`, and
commands run there with the same toolchain as CI. It is meant for people and for coding agents
working on separate branches in parallel, with the same commands and the same output conventions.

The Docker daemon is usually remote, so nothing is bind-mounted: `ws.sh` copies the tracked and
untracked (not ignored) files of the checkout with `tar` before each command.

## Requirements

- A Docker client and a Docker context for the build host, for example
  `docker context create build-host --docker host=ssh://USER@HOST`. The host needs CPU cores, disk
  for the build trees, and access to `ghcr.io/fuzzybit-org/fuzzybits-toolchain`
  (`docker login ghcr.io` on the host if the package is private).
- The checkout: any clone or `git worktree` of this repository.

The build host is chosen at run time and never written into the repository:

1. `--context NAME` (first argument), else
2. `FZB_WS_CONTEXT=NAME`, else
3. `DOCKER_HOST` / `DOCKER_CONTEXT`, else
4. the current Docker context.

## Quick start

```bash
export FZB_WS_CONTEXT=build-host            # or pass --context build-host
scripts/dev/ws.sh doctor                     # client/server versions, image, ccache volume
scripts/dev/ws.sh configure my-branch --tail 3
scripts/dev/ws.sh build my-branch fuzzybits_ir_pra_tests --tail 3
scripts/dev/ws.sh test my-branch 'fuzzybits_ir_pra' --tail 20
scripts/dev/ws.sh run my-branch --tail 40 -- 'ctest --test-dir build/generic -R pcfb --output-on-failure'
scripts/dev/ws.sh down my-branch
```

Run the script from the checkout you want to build (it defaults to the checkout it lives in), or
pass `--worktree DIR`.

## Commands

| Command | Does |
|---|---|
| `up NAME` | create and start the container `fzb-ws-NAME` |
| `sync NAME` | copy the checkout into `/src` |
| `run NAME -- CMD` | sync, then `bash -c CMD` in `/src` |
| `configure NAME [PRESET]` | sync, then `scripts/ci/inner-build.sh PRESET configure` |
| `build NAME [TARGET...]` | sync, then `cmake --build --preset PRESET [--target T ...]` |
| `test NAME [REGEX] [-- ARGS]` | sync, then `ctest --test-dir build/PRESET [-R REGEX] --output-on-failure ARGS` |
| `shell NAME` | interactive shell in `/src` |
| `status [NAME]` | list workspaces (name, state, checkout, owner, image) |
| `down NAME` | remove the container (the shared ccache volume stays) |
| `prune` | remove every workspace container owned by `$USER` |
| `doctor` | check the context, the image and the ccache volume |

Options for `sync`, `run`, `configure`, `build`, `test`:

| Option | Use |
|---|---|
| `--worktree DIR` | the checkout to copy |
| `--no-sync` | run without copying (the files in `/src` are as last synced) |
| `--clean` | delete `/src` except `build/` and `.cache/` first: after deleting or renaming files, or a rebase |
| `--fresh` | delete all of `/src`, including `build/`, first: after toolchain or CMake cache changes |
| `--private[=DIR]` | copy private fixtures (see below) |
| `--env KEY=VALUE` | extra environment for the command (repeatable) |
| `--tail N` | print only the last N lines; the full output is in the log |

## Conventions

- **One workspace per branch or task**, named after it (`[a-z0-9][a-z0-9_.-]*`). Workspaces are
  independent; they share only the ccache volume.
- **Copying never deletes.** A file deleted locally stays in `/src` until `--clean`.
- **Commands are serialised per workspace** (a lock inside the container), so a background build and
  a second command cannot interleave. Use separate workspaces for parallel work.
- **Logs:** every command's full output goes to
  `${FZB_WS_LOG_DIR:-${XDG_STATE_HOME:-~/.local/state}/fzb-ws}/NAME/`, and the last line on stderr
  names the file and the exit status. With `--tail N` only the last N lines are printed, which keeps
  agent transcripts short; read or grep the log for details instead of re-running.
- **Exit status** is the command's, so `ws.sh test ... && next-step` works.
- **Clean up** with `down` when the branch is merged; `prune` removes all of yours.

## Private fixtures

`--private=DIR` (or `--private` with `FUZZYBITS_PRIVATE_FIXTURES_DIR` set) copies DIR into the
container's `/private` (mode 700) and sets `FUZZYBITS_PRIVATE_FIXTURES_DIR=/private` for that command,
so private tests and cards run (#38). The copy is replaced on every such command and removed with the
container. Never print, commit or paste private identifiers or values; report aggregates only.

## Images

The image is the toolchain tag pinned by `toolchain.env`
(`ghcr.io/fuzzybit-org/fuzzybits-toolchain:${ACPP_REF}-generic-<sha256(toolchain.env)[:12]>`). A pin
change moves the hash before CI publishes the new tag; then `ws.sh` falls back to the moving tag
`${ACPP_REF}-generic` with a note (fetched dependencies such as pcfb are not part of the image).
`TOOLCHAIN_IMAGE=...` overrides both. An existing workspace keeps the image it was created with; run
`down` and start again to pick up a new one.

## Settings

| Variable | Default |
|---|---|
| `FZB_WS_CONTEXT` | current Docker context |
| `FZB_WS_PRESET` | `generic` |
| `FZB_WS_CPUS` | `12` (container CPUs and build parallelism; ctest uses half) |
| `FZB_WS_CCACHE_VOLUME` | `fzb-agent-ccache` |
| `FZB_WS_LOG_DIR` | `${XDG_STATE_HOME:-~/.local/state}/fzb-ws` |
| `TOOLCHAIN_IMAGE` | pinned tag, else the moving tag |

## Troubleshooting

- *`cannot pull ...`*: log in to GHCR on the build host, or set `TOOLCHAIN_IMAGE`.
- *A test sees an old file*: `--clean` (deleted or renamed files linger).
- *CMake complains about a changed source or toolchain*: `--fresh`.
- *A command waits*: another command holds the workspace lock; see `status` and the logs.
- *macOS `._*` files in the container*: `ws.sh` sets `COPYFILE_DISABLE=1` and strips macOS metadata;
  remove stray ones with `--clean`.

For coding agents, `scripts/dev/agent-brief.md` is a block to paste into task instructions.
