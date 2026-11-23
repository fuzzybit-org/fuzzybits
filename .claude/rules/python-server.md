---
paths:
  - "python/**"
  - "targets/python/**"
  - "tools/server/**"
  - "tests/python/**"
  - "pyproject.toml"
---
# Python package and server

- The import name is `fuzzybits`. The wheel also installs `fzb`, which must match the native CLI
  (`docs/python.md`). `fuzzybits-server` is pinned to the same version (`docs/releasing.md`).
- `tests/python` runs against an installed wheel. Run it with
  `scripts/dev/ws.sh run <ws> --tail 40 -- '<command>'`, or let CI do it
  (`scripts/ci/python-pytest.sh`). Mark GPU tests `gpu`.
- Server docs: `tools/server/README.md`. Its web UI is built with npm in `tools/server/webui`.
