# Agent brief: building and testing

Paste this into a task, replacing the `<...>` values. The build host is given here, not in the repository.

---

**Build and test with `scripts/dev/ws.sh`** (details: `scripts/dev/README.md`). Do not use local
Docker bind mounts or other build helpers.

- Build host: `export FZB_WS_CONTEXT=<docker-context>`.
- Your checkout: `<worktree path>`; run the script from it. Your workspace name: `<name>`.
- First build: `scripts/dev/ws.sh configure <name> --tail 3`.
- Build only what you need: `scripts/dev/ws.sh build <name> <target>... --tail 5`
  (list targets with `scripts/dev/ws.sh run <name> --tail 200 -- 'cmake --build --preset generic --target help'`).
- Tests: `scripts/dev/ws.sh test <name> '<ctest regex>' --tail 40`.
- Anything else: `scripts/dev/ws.sh run <name> --tail 40 -- '<command>'`.
- After deleting or renaming files, or a rebase: add `--clean`; after toolchain or CMake cache changes: `--fresh`.
- Keep output short with `--tail`; the last stderr line names the full log. Read or grep the log
  instead of re-running a command.
- One command at a time per workspace (commands are serialised); use a second workspace name for parallel work.
- Private fixtures, only when the task asks for them: `--private=<directory>`; report aggregates only,
  never identifiers or values.
- When the PR is done: `scripts/dev/ws.sh down <name>`.
