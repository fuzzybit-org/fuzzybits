---
paths:
  - "scripts/dev/macos-dev.sh"
  - "scripts/ci/macos-*"
  - ".github/workflows/macos.yml"
---
# macOS and Metal

The one exception to the `ws.sh` rule. Use `scripts/dev/macos-dev.sh` only for macOS-specific or
Metal work that needs a macOS host. Follow `docs/build-variants.md#macos-local-development`:

- `scripts/dev/macos-dev.sh` (bootstrap), then `source .cache/macos-dev.env`.
- `cmake --workflow --preset metal-local` (OpenMP tests), then `ctest --preset metal-local`
  (Metal device).
- `scripts/dev/macos-dev.sh doctor` reports state without changing anything.

Everything else, including Linux behaviour that a macOS change might affect, is verified with
`ws.sh`.
