---
paths:
  - "benchmarks/**"
  - "scripts/ci/bench*"
  - ".github/workflows/bench*.yml"
---
# Benchmarks

- `docs/benchmarks.md` is the reference for the dataset version, trial counts, the A/B PR gate,
  A/A noise, hosts and published history. Changing a dataset means a new dataset version, not an
  edit to v1.
- Reference estimators in `benchmarks/reference` must stay independent of the engine (no imports
  from `fuzzybits`).
- Private models: report aggregates only, never model names or values.
