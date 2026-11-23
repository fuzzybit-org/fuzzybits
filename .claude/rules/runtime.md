---
paths:
  - "src/mc/**"
  - "src/runtime/**"
---
# Monte Carlo engine and runtime (`src/mc`, `src/runtime`)

- SYCL through AdaptiveCpp, USM only (no SYCL buffers), Philox counter-based RNG, bit-packed
  samples. `docs/architecture.md` describes the design but predates the rename (#112).
- Results must be reproducible: same seed, same tallies on every backend. GPU behaviour is checked
  on self-hosted runners (`gpu.yml`); you cannot run GPU tests in a CPU workspace, so say so in the
  PR's **Tests** section.
- Device selection: `ACPP_VISIBILITY_MASK` (`docs/build-variants.md`). The test presets set it.
- Some files here include SCRAM-derived headers and are on the allow-list in
  `scripts/ci/scram_provenance.json`. Do not add to that list.
