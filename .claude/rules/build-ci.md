---
paths:
  - "CMakeLists.txt"
  - "CMakePresets.json"
  - "cmake/**"
  - "toolchain.env"
  - "docker/**"
  - "Dockerfile*"
  - ".github/**"
  - "scripts/ci/**"
---
# Build, toolchain and CI

- Every toolchain pin is in `toolchain.env`. The image tag hashes it
  (`scripts/ci/toolchain-sha.sh`), so any edit there means a toolchain rebuild. Docs:
  `docs/ci-toolchain.md`, `docs/build-variants.md`, `docs/cuda-builds.md`, `docs/images.md`.
- Presets in `CMakePresets.json` are the only CMake entry points used by CI, `ws.sh` and the
  wheels. Add a variant there, not as ad-hoc flags.
- Workflows are linted with actionlint (`.github/actionlint.yaml`). GPU runners and their device
  ordinals are in `.github/hardware.json` and `docs/ci-toolchain.md`. Use nvidia-smi ordinals, never
  CUDA ordinals.
- GPU workflows never run on `pull_request`, `pull_request_target` or `workflow_run`. Pre-merge
  GPU coverage comes from a same-repo push to `ci/**` (see the header of
  `.github/workflows/gpu.yml`).
- `ci.yml` uses the self-hosted `fuzzybits-ci` group for pushes and same-repo PRs, and falls back
  to GitHub-hosted `ubuntu-24.04` for fork and Dependabot PRs. Copy its `runs-on` expression for
  new jobs.
- C++ formatting is checked by `scripts/ci/clang-format.sh` (clang-format 20.1.8, `hygiene` job).
  Fix locally with `scripts/ci/clang-format.sh --fix`.
