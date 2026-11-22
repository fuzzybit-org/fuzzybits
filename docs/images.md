# Runtime images

Slim runtime images published to GHCR by [`.github/workflows/images.yml`](../.github/workflows/images.yml) (#35). They replace the manually built `ghcr.io/a-earthperson/mcscram:latest` (14.4 GB, oneAPI/L0/ROCm).

Image: **`ghcr.io/fuzzybit-org/fuzzybits-runtime`** (public). Packages are
linked to this repo via `org.opencontainers.image.source`.

Compiler toolchain (public, not this image): [`docs/ci-toolchain.md`](ci-toolchain.md).

## Why FROM the toolchain image

The publish job (self-hosted `fuzzybits-ci`) builds `docker/runtime.Dockerfile`
**on top of** the immutable toolchain tag
`ghcr.io/fuzzybit-org/fuzzybits-toolchain:<ACPP_REF>-generic-<TOOLCHAIN_SHA>`
(`scripts/ci/toolchain-sha.sh`; never the moving tag). The toolchain already
has LLVM 20, AdaptiveCpp `full` (SSCP + clang CUDA SMCP + HIP), CUDA 12.8.1
devel, ROCm 7.2 HIP/hipRTC user space, and Boost 1.89. If a push changes the
toolchain, the gate defers until `toolchain.yml` has published the new tag and
dispatched this workflow again.

Re-running the LLVM/AdaptiveCpp stages on a 4-core hosted runner is tens of
minutes (AdaptiveCpp + Boost); a LABEL-only toolchain rebuild is seconds. The
product C++ build is ~8 min per preset with `BUILD_TESTING=OFF` (see the
toolchain doc). This workflow still compiles twice per variant — once for
`scram-cli`/`fzb` (non-SKBUILD; SKBUILD skips `fzb`) and once for the
scikit-build-core wheel.

Measured on `ubuntu-24.04` (toolchain already in GHCR):

| | generic | cuda12.8-sm70 |
|---|---|---|
| first publish (empty runtime cache) | 14m25s | 14m39s |
| after registry `buildcache-<variant>` | **11m32s** | ~12–14m |

The final image does **not** contain the toolchain. `runtime` is `ubuntu:24.04`
plus:

| contents | notes |
|---|---|
| `scram-cli`, `fzb` | `/usr/local/bin`; entrypoint wrapper: `docker run IMG` → scram-cli, `docker run IMG fzb …` → fzb |
| AdaptiveCpp runtime | `libacpp-rt`, `libacpp-common`, CUDA/OMP backends (+ HIP in `generic`), `llvm-to-backend` plugins, `libLLVM.so.20.1`, `acpp-info`. Distro `libxml2` (`libxml2.so.2`) is required by LLVM; the product uses FetchContent `libxml2.so.16`. |
| SSCP JIT (`generic` only) | `libdevice.10.bc`, `opt`/`llc`/`ld.lld` at `/usr/lib/llvm-20/bin` and `hipSYCL/ext/llvm/bin` (AdaptiveCpp LLVMToHost/LLVMToPtx). `llvm-to-backend` / `libLLVM` ship in both variants because the backends DT_NEED them even for SMCP AOT. |
| `fuzzybits` Python | wheel built in the same Dockerfile, `pip install` into the image; native `/usr/local/bin/fzb` restored after pip so the console script does not overwrite it; no `PYTHONPATH` / `LD_PRELOAD` |
| `mcscram-smoke` | chinese, baobab1, das9201 vs Aralia BDD refs; `timeout 60` each |
| CUDA | `libcudart` from the toolchain (CUDA 12.8.1). Driver (`libcuda.so.1`) comes from `--gpus`. Not `FROM nvidia/cuda:*-runtime` (that image is ~3 GB of unused cublas/cufft/…). |
| HIP (`generic` only) | `librt-backend-hip`, `libllvm-to-amdgpu`, ROCm `amdgcn` device bitcode, and the ROCm 7.2 user space in `/opt/rocm/lib`: `libamdhip64.so.7` (+ the unversioned link hipRTC dlopens), `libhiprtc.so.7`, `libhiprtc-builtins`, `libamd_comgr.so.3`, `libhsa-runtime64`, `librocprofiler-register` (~210 MB). The kernel driver (`amdgpu`/KFD) is the host's; pass `/dev/kfd` and the GPU's render node. |

Runs as non-root `coder` (uid 1000) with writable JIT cache
`ACPP_APPDB_DIR=/home/coder/.cache/fuzzybits/acpp`. `MALLOC_TYPE=malloc`.

Size budget: **generic ≤ ~4 GB**, **cuda-smNN ≤ ~3.5 GB** (docker uncompressed).

`ACPP_VISIBILITY_MASK` defaults to `cuda;hip` in every image (AdaptiveCpp
reads it before `main()`, so a later narrowing such as `fzb --backend hip`
only selects among plugins this mask loaded). With no NVIDIA driver the CUDA
plugin fails to load (a warning only at `ACPP_DEBUG_LEVEL` ≥ 2; the image sets
0); with no `/dev/kfd` the HIP plugin reports zero devices. Python
`fuzzybits.init()` replaces the mask with the auto-selected backend.
CPU-only runs set `ACPP_VISIBILITY_MASK=omp`.

## Tags

Let `SHA12` be the first 12 hex chars of the fuzzybits git SHA. Variant is
`generic` (default, SSCP JIT, any NVIDIA GPU, ROCm-supported AMD GPU, or CPU) or one of the CUDA AOT
builds (SMCP SASS + embedded PTX, CUDA only, no SSCP):

| variant | `ACPP_TARGETS` | GPUs |
|---|---|---|
| `cuda12.8-sm70` | `cuda:sm_70` | Volta (V100, V100S) |
| `cuda12.8-sm86` | `cuda:sm_86` | Ampere consumer (RTX 30xx, A10, A40) |
| `cuda12.8-sm120` | `cuda:sm_120` | Blackwell consumer (RTX 50xx) |

Every variant is published on the same triggers (nightly, `develop`, `ci/**`
and release tags). `fuzzybits-server` publishes the same four variants on top
of the matching runtime image; deploys use `cuda12.8-sm70`.

| tag | when | mutability |
|---|---|---|
| `ci-<sha12>-<variant>` | push to `ci/**` | immutable (that SHA) |
| `sha-<sha12>-<variant>` | `develop`, schedule, `workflow_dispatch` on develop | immutable |
| `nightly-<variant>` | same as `sha-` | moves |
| `vX.Y.Z-<variant>` | git tag `vX.Y.Z` | immutable |
| `vX.Y.Z` | git tag, **generic only** | immutable |
| `latest` | git tag, **generic only** | moves |
| `buildcache-<variant>` | BuildKit registry cache (`mode=max`) | cache |

Pre-merge smoke on `ci/images` pulls `ci-<sha12>-<variant>` so it never races a moving `nightly-` tag.

## How to pull

```bash
# default generic (after a develop publish)
docker pull ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic

# Volta / Ampere / Blackwell AOT
docker pull ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-cuda12.8-sm70
docker pull ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-cuda12.8-sm86
docker pull ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-cuda12.8-sm120

# pin a SHA
docker pull ghcr.io/fuzzybit-org/fuzzybits-runtime:sha-<sha12>-generic
```

The package is **public**. Anonymous `docker pull` works; self-hosted smoke
does not log in to GHCR and does not get `packages: read`.

Measured uncompressed sizes (`docker image inspect`): **generic 0.95 GiB**
with HIP/ROCm (0.75 GiB before; `ci/amd-gpu-hip`), **cuda12.8-sm70 0.44 GiB**
(`ci/images` `f1f760a4ead0`).

## How to run

GPU device mapping (xerxes Docker; same as gpu.yml / #34). Indices are
nvidia-smi / NVML / `--gpus device=N`, **not** CUDA FASTEST_FIRST (the reverse
on xerxes):

| host GPU | `--gpus` | image variant |
|---|---|---|
| Tesla V100S (`sm_70`) | `device=0` | `cuda12.8-sm70` **and** `generic` |
| RTX 3070 (`sm_86`) | `device=1` | `cuda12.8-sm86` **and** `generic` |
| RTX 5060 Ti (`sm_120`, shelly-1) | `device=0` | `cuda12.8-sm120` **and** `generic` |
| Radeon Pro 5500M 8 GB (`gfx1012`, bubs / mbp-1) | none; `$(scripts/ci/amd-docker-args.sh)` | `generic` only |

The smoke job on each GPU runs both of its variants. SMCP-only images skip the
OpenMP smoke: they carry no host kernels.

```bash
IMG=ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic

# scram-cli
docker run --rm --gpus device=0 "$IMG" --help

# fzb
docker run --rm --gpus device=0 "$IMG" fzb compile /app/input/chinese.xml -o /tmp/chinese.pcfb
docker run --rm --gpus device=0 "$IMG" fzb run /tmp/chinese.pcfb --trials 1e8 --backend cuda

# in-image smoke (mcscram-smoke + fzb compile/run + Python import)
docker run --rm --gpus device=0 --entrypoint fuzzybits-image-smoke "$IMG"

# Python (no PYTHONPATH / LD_PRELOAD)
docker run --rm --gpus device=0 --entrypoint python3 "$IMG" -c \
  'import fuzzybits; fuzzybits.init(); print(fuzzybits.list_devices())'

# AdaptiveCpp backends
docker run --rm --entrypoint acpp-info "$IMG" -l

# AMD (ROCm-supported GPU, host amdgpu driver; no ROCm install needed on the
# host). amd-docker-args.sh prints --device /dev/kfd, the GPU's render node and
# --group-add <render gid>; passing all of /dev/dri also works.
docker run --rm $(scripts/ci/amd-docker-args.sh) --entrypoint bash "$IMG" -c \
  'fzb compile /app/input/chinese.xml -o /tmp/c.pcfb && fzb run /tmp/c.pcfb --trials 1e8 --backend hip'
docker run --rm $(scripts/ci/amd-docker-args.sh) --entrypoint fuzzybits-image-smoke \
  -e SMOKE_MC=0 -e SMOKE_FZB_CUDA=0 -e SMOKE_FZB_OMP=0 -e SMOKE_FZB_HIP=1 "$IMG"
```

`docker run IMG mcscram-smoke` does **not** work: the entrypoint forwards unknown
args to `scram-cli`. Use `--entrypoint mcscram-smoke` or `fuzzybits-image-smoke`.

Every Monte Carlo run inside the image is wrapped in `timeout 60`.

## Workflow

Triggers (never `pull_request` / `pull_request_target` / `workflow_run`):

- `push` to `develop` (path-filtered to engine/runtime sources, Dockerfiles,
  `docker/`, `toolchain.env`, CMake, this workflow)
- `push` to `ci/**` (always; tags are `ci-<sha12>-<variant>` only)
- git tags `v*`
- nightly `schedule` (`47 7 * * *` UTC)
- `workflow_dispatch`

Jobs:

| job | runner | permissions |
|---|---|---|
| `gate` | `fuzzybits-ci` | `contents: read`; resolves the immutable toolchain tag (anonymous) or defers |
| `publish-<variant>` | `fuzzybits-ci` | `contents: read`, `packages: write`, `id-token: write`, `attestations: write` |
| `smoke-*` | self-hosted `[self-hosted, nvidia, sm_70\|sm_86\|sm_120]` | inherits `contents: read`; anonymous GHCR pull; generic also runs `fzb --backend omp` |
| `smoke-cpu-omp` | `fuzzybits-ci` | `fzb compile` + `fzb run --backend omp --trials 1e6` on chinese, \|z\| ≤ 4 |
| `smoke-amd-gfx1012 (optional)` | `gpu-runners` `[self-hosted, amd-gpu]` (mbp-1) | `continue-on-error`; generic only, default mask; `acpp-info` must list a HIP device, then `fzb --backend hip` and Python `backend="hip"` at 1e8, \|z\| ≤ 4. Disable with repository variable `FUZZYBITS_AMD_GPU=disabled` |

Publish uses `docker/build-push-action` with `--sbom=true --provenance=mode=max`
and registry cache `ghcr.io/fuzzybit-org/fuzzybits-runtime:buildcache-<variant>`.
Pins come from [`toolchain.env`](../toolchain.env) as `--build-arg`s.

Smoke concurrency groups (shared with gpu.yml, one physical GPU each):
`gpu-xerxes-v100s`, `gpu-xerxes-3070`, `gpu-shelly-5060ti`.

`actions/checkout` uses `persist-credentials: false`. Third-party actions are
pinned to full commit SHAs. Jobs have `timeout-minutes`. Smoke uses
`docker run --rm` only and does not install packages on the host. shelly-1 is a
production host — smoke does not `docker ps` / `docker system prune` / touch
other containers.

## Follow-ups

- **`-prof` variants** (Nsight Compute/Systems on top of the slim runtime).
  Nsight versions must stay compatible with the host driver (see
  [`docs/cuda-builds.md`](cuda-builds.md) and #26). Not in this workflow.
- **Release-tag flow** is implemented in the tagging logic (`vX.Y.Z-<variant>`,
  plus `vX.Y.Z` / `latest` for generic) but has not been exercised until a
  `v*` tag is pushed.
- **arm64** generic runtime: later.
- fuzzybits-bench should switch `stack.yml` / `scripts/publish-docker-images.sh`
  to these images (fuzzybit-org/fuzzybits-bench#1, #2). Until then the
  user-owned `ghcr.io/a-earthperson/mcscram*` packages stay.
