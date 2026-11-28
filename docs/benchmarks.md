# Benchmarks

Frozen accuracy and performance measurements for the Monte Carlo engine
([#4](https://github.com/fuzzybit-org/fuzzybits/issues/4)). This is the
dataset ([#120](https://github.com/fuzzybit-org/fuzzybits/issues/120)) and
harness ([#121](https://github.com/fuzzybit-org/fuzzybits/issues/121)). The PR
gate that calls the harness is [#122](https://github.com/fuzzybit-org/fuzzybits/issues/122);
published history is [#123](https://github.com/fuzzybit-org/fuzzybits/issues/123).

Legacy scripts under `benchmarks/` (`bench_scram.py`, `benchmark_compiler.py`,
…) are from the mcSCRAM era. They stay in the tree; the harness does not use
them.

## Dataset versioning

`benchmarks/datasets/v1/` is **frozen**. A dataset version never changes once
used: the harness, the PR gate and the published history pin a dataset id.
Edits to membership, references, trial counts, tags or the schema are a new
version (`v2`, …).

Regenerate only when cutting a new version:

```bash
python -m benchmarks.datasets.freeze
```

## Dataset v1

| set | files | role |
|---|---|---|
| Aralia (`input/Aralia`, 43 XML) | accuracy (all 43) and a `perf` subset | exact top-event probabilities |
| generic-pwr-model (`input/generic-pwr-model`, 95 OpenPSA event-tree-group XML, CC0, submodule `ce2ef006…`) | `pwr`: compile / PDAG size / throughput | SAPHIRE MCS references are **not** used for accuracy |
| `synthetic-models` | out of scope | no license declared |

Each model records: path, sha256 of the XML, submodule commit (PWR only), the
query (top event / gate id), the reference probability and its source, tags
(`accuracy`, `perf`, `hard`, `pwr`), trial counts, and a time budget.

### Accuracy references

42 Aralia models use the exact BDD values in
`tools/server/src/fuzzybits_server/model_oracle_p.json` (`source: bdd`).

`nus9601` is not in that file. The v1 reference is **P = 9.94453321e-06**,
computed with nombrillo by Kirill Shumilov (`@kshumilov`), recorded as an
**exact** reference. `fzb inspect --exact` refuses 1,567 SAMPLE events (cap
30). `scram-cli --bdd` / `--zbdd` runs out of memory on this model; do not
retry that path.

The compiled query for every Aralia model is gate `r1` (the first
`define-gate`, which is the fault-tree top).

### Trial counts and power rule

Accuracy runs target **n·p ≥ 1000** so a 4-sigma check is meaningful, then
round up to a multiple of 2^20 (the engine pass size) and cap at
`fzb --trials 1e8` = 96·2^20 = 100,663,296.

The harness judges **expected hits = n·p_ref first**, then z. Thresholds live
in `manifest.defaults` (`accuracy_infeasible_hits`, `accuracy_low_power_hits`):

| expected hits n·p_ref | verdict | counts as an accuracy pass? |
|---|---|---|
| < `accuracy_infeasible_hits` (**16**) or p_ref < 2^-32 (~2.33e-10, [#17](https://github.com/fuzzybit-org/fuzzybits/issues/17)) | **`infeasible`** | no |
| 16 ≤ n·p_ref < `accuracy_low_power_hits` (**1000**) and \|z\| ≤ 4 | **`pass (low power)`** | yes, annotated |
| n·p_ref ≥ 1000 and \|z\| ≤ 4 | **`pass`** | yes |
| n·p_ref ≥ 16 and \|z\| > 4 | **`block`** | no |

16 is the vacuous-pass bound: 0 hits gives \|z\| = √(n p_ref) < 4, so the test
has no power. (The “e.g. 100” figure is the same order of magnitude;
`isp9607` at n·p ≈ 96 is therefore `pass (low power)`, not infeasible.) When
hits = 0 the table omits z; the power rule has already classified the row.

Rare-event work is [#6](https://github.com/fuzzybit-org/fuzzybits/issues/6)
(importance sampling).
GPUs help by a few orders of magnitude and still do not reach n·p ≈ 1000 for
these models. Do not enlarge the CI budget to chase them.

| model | p | accuracy n | expected hits | verdict category |
|---|---|---|---|---|
| most Aralia | ≳ 1e-4 | 2^20 | ≫ 1000 | `pass` |
| isp9605 | 1.37e-5 | 70·2^20 | ~1000 | `pass` |
| **nus9601** | 9.94453321e-06 | **96·2^20 (1e8)** | **~1001** | `pass` (CPU budget **180 s**) |
| isp9607 | 9.50e-7 | 1e8 cap | ~96 | `pass (low power)` |
| das9205 | 1.38e-8 | 1e8 cap | ~1.4 | `infeasible` (n·p ≪ 16) |
| das9204, das9209, edf9206 | ≲ 2e-11 | 1e8 cap | ≪ 1 | `infeasible` (n·p ≪ 16 **and** p < 2^-32, #17) |

z, when shown: |z| ≤ 4 with σ̂² = p̂(1−p̂)/n + σ_ref² (σ_ref = 0 for exact
references). If p̂ ∈ {0,1} on a **gated** row, the null variance
p_ref(1−p_ref)/n is used so the arithmetic is defined; those rows should
already have been classified infeasible.

### Performance subset

Chosen for coverage of the engine's cost profile (kernel breakdowns from
[PR #91](https://github.com/fuzzybit-org/fuzzybits/pull/91) and
[#20](https://github.com/fuzzybit-org/fuzzybits/issues/20)):

| model | reason |
|---|---|
| `das9201` | Sampler-dominated: on V100S at 1e8, `basic_event` is ~89% of GPU kernel time (37.6 of 42.1 ms). |
| `das9701` | Largest Aralia gate counts; fused-kernel prototype's best case in #20 (3.11×). |
| `nus9601` | 1,567 basic events; fused BE tile does not fit in shared memory (#20). |
| `edf9203` | Known fused-path regressor in #20 (2.2× slower than unfused). |
| `chinese` | Small model; launch / JIT / host overhead baseline (~9–13 ms kernels). |

The PR gate's GPU A/B uses `das9201`, `das9701`, `nus9601` and `edf9203`
(`GPU_PERF_IDS` in `bench.yml`); `chinese` stays in the CPU set and in
published history. These four are still small: a realistic large-model set
waits on manifest v2 (#158) and whole-model compile (#92, #153).

The PR gate times GPU A/B with **`--metric sample`**: in-process
`fuzzybits.run` wall time (`sample_s`) after AdaptiveCpp init and a 2^20
warm-up, with the trial count sized per model so one timed run lasts
`--seconds` (see [PR gate](#pr-gate-ci)). Pass size does not depend on the
trial count (2^20 for das9701, 2^19 for nus9601 on the RTX 3070), so the
warm-up compiles the timed launch shape; across fresh containers sharing the
JIT volume, the first timed call matched a second one within 0.5–1.4%.

`--metric kernel` (the default outside the gate) keeps per-kernel times
(`basic_event`, gate kernels, tally) via **nvprof** on V100S (`sm_70`) and
**nsys** on RTX 3070 (`sm_86`) at 1e8 trials. RTX 5060 Ti (`sm_120`, shelly)
has no Nsight Systems and falls back to `sample_s`. CPU perf trial counts are
sized per model so in-process sampling is ~2–5 s (`trials.cpu_perf` in the
manifest). Warm and cold AdaptiveCpp cache times are reported separately;
comparisons use **warm** numbers. The `basic_event` bucket holds the sampling kernels by name
(`benchmarks/harness/profile.py`): `kernel::basic_event` in builds before the per-form kernels of
`src/mc/kernel/sampler/`, `kernel::sampler::strided`, `::lockstep` and `::chunked` after.

With `--metric kernel`, the **metric is `gpu_total_ms`** (all compute kernels; memcpy
excluded), not `basic_event` alone. On das9701 at 1e8, `basic_event` is 81.5 ms
of 218 ms total GPU kernel time (~37%). A kernel-only gate would miss
regressions in gates/tally/other, and a 10% `basic_event` regression is only
~3.7% of total — right on the 3% threshold. The harness therefore **dual-gates**:
it blocks if *either* `gpu_total_ms` or `gpu_kernel_ms` (`basic_event`) has
`ci_lo > 1.03`. `gpu_kernel_ms` is skipped when `basic_event` is missing, not
treated as zero.

### generic-pwr-model

v1 records compile time, PDAG size (`fzb inspect --json`: tensors, operators,
samples) and sampling throughput. Accuracy is deferred: SAPHIRE MCS values are
mostly 1e-10 to 1e-6, and only the first fault-tree top event is compiled
([#92](https://github.com/fuzzybit-org/fuzzybits/issues/92)). Revisit with #92
and #6. The generic-pwr-model OpenPSA export is not self-contained MEF;
the harness rewrites a cached copy via `benchmarks/datasets/compat/generic_pwr.py`.

A fresh clone needs:

```bash
git submodule update --init input/generic-pwr-model
```

## Loader

```bash
python -m benchmarks validate
```

Checks: JSON structure, tags, references in (0, 1), files exist, sha256
matches, PWR submodule pin. This is the CI test (`tests/python/test_benchmark_dataset.py`);
it is CPU-only and does not import the native engine.

## Harness

```text
python -m benchmarks validate
python -m benchmarks run  --target cpu|v100s|rtx3070|rtx5060ti|gfx1012 --mode accuracy|perf|compile \
    --image IMG --out results.json [--md summary.md] [--ids chinese das9201]
python -m benchmarks ab   --target cpu|v100s|rtx3070|rtx5060ti|gfx1012 --rounds 6 \
    --base SHA --head SHA --build \
    --out ab.json [--md ab.md]
python -m benchmarks ab   --target v100s --base-image IMG --head-image IMG \
    --ids chinese --inject-regression scale:1.10 --out aa.json
python -m benchmarks summarize results.json
```

`run` and `ab` execute `fzb compile` / `inspect` / `run` **inside** a runtime
image. Models are mounted from the docker host (`/tmp/fuzzybits-bench/input` by
default). Results are versioned JSON (`benchmarks/schemas/results.v1.json`)
plus a Markdown table suitable for a PR comment.

### A/B mode (what the PR gate uses)

Numbers only compare at the few-percent level when **both commits are built
with the same recipe and timed alternately on the same host** (lesson from
[PR #91](https://github.com/fuzzybit-org/fuzzybits/pull/91#issuecomment-7019747119)).

1. Build base and head with `docker/runtime.Dockerfile --target runtime` on
   `ghcr.io/fuzzybit-org/fuzzybits-toolchain:v25.10.0-generic-<sha256(toolchain.env)[:12]>`.
   Context is `git archive <sha> | docker build -`. On macOS set
   `COPYFILE_DISABLE=1` (AppleDouble `._*` files broke corpus tests).
   Registry tags are resolved **once** with `docker buildx imagetools inspect`
   and then run as `name@sha256:<64 hex>`. Do not pass a mutable tag such as
   `nightly-generic` through to `docker run`: gateway and xerxes previously
   resolved that tag to different local config ids
   (`sha256:bb0239034e17…` vs `sha256:26b0c066974e…`). Results JSON records the
   full digest.
2. Warm both AdaptiveCpp JIT caches (`ACPP_APPDB_DIR` on a per-side volume)
   at the timed trial count. Containers run with `ACPP_ADAPTIVITY_LEVEL=1`
   and `ACPP_PERSISTENT_RUNTIME=0` (also the image defaults). Level 2 keeps
   re-specialising kernels: steady-state 64 Mi-trial das9201 runs took
   0.63–0.93 s at 2 vs 0.15 s at 1 on gfx1012; 0 fails with CU:701 on
   das9701 (#18). A persistent runtime added ~0.5 s to the first run of each
   launch shape; without it the three runs were 0.153/0.152/0.152 s. Results
   record both values in `context`.
3. Alternate base/head for N rounds (default 6; CPU A/A uses 10), swapping
   the order every round (A-B, B-A, …) so slow drift does not always land on
   one side. Profiled runs drop the first round; `--metric sample` keeps all
   rounds because each container warms up before timing. GPU rounds record
   the device's other compute processes and utilisation (`round_load`).
4. CPU: `OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores`,
   `docker --cpuset-cpus 0-7`, `ACPP_VISIBILITY_MASK=omp`. The metric is
   **in-process `fuzzybits.run(..., trials=head_trials)` time after AdaptiveCpp
   init and a 2^20 JIT warmup** (`cpu_sample_s`). Extra `--inject-regression
   scale:F` trials are that `head_trials` argument, not the warmup and not a
   second untimed call. Do not use container wall time: on chinese at
   1e8, container wall was ~1.2 s while sampling is ~1.3 s once init is
   excluded — still too short, so CPU perf trial counts are sized for
   **~2–5 s of sampling**. GPU: `ACPP_VISIBILITY_MASK=cuda`. V100S uses
   `nvprof --print-gpu-summary --profile-child-processes` (volume `pr91-nvprof`
   at `/opt/cuda`, binary `/opt/cuda/cuda/bin/nvprof`); RTX 3070 uses
   `nsys profile --stats=true --trace-fork-before-exec=true` (volume `pr91-data`).
5. Report the paired ratio: the geometric mean of the per-round
   `head / base` ratios with a 95% Student t interval on the mean log ratio,
   plus `t` = mean / standard error. Pairing within a round cancels load that
   hits both sides of that round; the t interval (not a bootstrap) stays
   honest at 6–12 rounds.
6. Verdict:
   - **block** only when the CI *lower bound* exceeds the threshold: GPU
     `sample_s` (or, with `--metric kernel`, `gpu_total_ms` **or**
     `gpu_kernel_ms`) `ci_lo > 1.03`. CPU sample
     `ci_lo > 1.05` is computed the same way but is **report-only**
     (`CPU_GATE_BLOCKS = False`, [#124](https://github.com/fuzzybit-org/fuzzybits/issues/124)).
     A point estimate above the threshold with `ci_lo` still below it is
     **warn**, not block. Accuracy |z| > 4 always blocks.
   - **warn** for a significant slowdown whose CI still overlaps the
     threshold. CPU above-threshold is always warn, never block.
   - **error** if a gated timing is missing or ≤ 0 (never a crash, never
     `block` / `pass`).
   - **pass** otherwise.
   - **inconclusive** if idle/load guards fail, or another compute process
     appears on an exclusive target during the rounds, rather than reporting
     noise as a regression.
   - Report-only targets keep the measured verdict in `measured_verdict` and
     report at most **warn**.
7. `--attempts 2` repeats the models flagged `block`, `error`, `timeout` or
   `inconclusive` once; the JSON records `attempts`, `flagged` and
   `flagged_ids`, and each retried row keeps its `first_attempt`.

### PR gate (CI)

`.github/workflows/bench.yml` runs on `pull_request` and `workflow_dispatch`
(#122). Same-repo, non-Dependabot PRs:

| job | runner | what | merge? |
|---|---|---|---|
| `bench-accuracy` | `fuzzybits-ci` | head image, every `accuracy` model | blocks on \|z\| > 4 (and harness `block`/`error`) |
| `bench-perf-rtx3070` | `public-ci` RTX 3070 (`sm_86,rtx_3070`, NVML GPU 1) | screen: A/B vs develop runtime image, `sample_s`, 6 rounds × 1.5 s, `--attempts 2` | **report-only**; a model flagged in both attempts requires `bench-perf-isolated` |
| `bench-perf-isolated` | same runner, environment `gpu-isolated` | A/B, `sample_s`, 12 rounds × 1.0 s, exclusive (`rtx3070-isolated`) | runs only after a double flag and a manual approval; blocks when `ci_lo > 1.03` |
| `bench-perf-v100s` | `public-ci` V100S (`sm_70,v100s`, NVML GPU 0) | A/B, `sample_s`, 6 rounds × 1.5 s | **report-only**; shares the GPU with the live and dev fuzzybits-server |
| `bench-perf-rtx5060ti` | `public-ci` RTX 5060 Ti (`sm_120,rtx_5060ti_8gb`, NVML GPU 0 on shelly) | same as V100S | **report-only**; ~0.5 GiB free beside VLLM |
| `bench-perf-gfx1012` | `public-ci` `amd-gpu` (mbp-1 on bubs) | A/B on the Radeon Pro 5500M via HIP, `sample_s`, 6 rounds × 1.5 s; skipped while either image predates the HIP backend | **report-only**, `continue-on-error` |
| `bench-perf-cpu` | `fuzzybits-bench` (gateway) | A/B, `cpu_sample_s` | **report-only** (#124) |
| `bench-summary` | hosted | sticky PR comment + overall verdict | the one required check |

Repository variables `SKIP_SM70`, `SKIP_SM86`, `SKIP_SM120` and `SKIP_GFX1012`
set to `true` skip the V100S, RTX 3070, RTX 5060 Ti and gfx1012 jobs (see
[ci-toolchain.md](ci-toolchain.md#hardware-targets-and-skip_-variables)); the
comment shows "hardware disabled" and the verdict ignores them. `SKIP_SM86`
also removes the only GPU gate that can block (the screen and isolated run).

Each GPU A/B spends at most `BUDGET_S` seconds timing samples per attempt
(90 s for the screen and the report-only jobs, 110 s for the isolated run):
`--seconds` shrinks if `2 × (rounds + 1) × models × seconds` would exceed it.
Container start-up, sizing and warm-up runs come on top.

**When the screen flags twice.** The PR comment says so, and
`bench-perf-isolated` waits for approval in the `gpu-isolated` environment.
Clear the RTX 3070 first (stop parakeet, the old server and any other GPU
process on GPU 1), then approve. The isolated target refuses to start while
any other compute process is on the device (`inconclusive`, so the check
fails), and marks a model `inconclusive` if one appears during its rounds.
Rerun the job after clearing the GPU. `bench-summary` waits for the isolated
job, so the PR stays blocked until it passes. If the screen job itself
fails before writing its JSON, the isolated run is required too.

Fork and Dependabot PRs skip the self-hosted jobs with a notice (push a
`ci/**` branch for numbers). Docs-only PRs skip accuracy/perf and
`bench-summary` still reports a green check so required status does not hang.

Base image is `ghcr.io/fuzzybit-org/fuzzybits-runtime:sha-<develop12>-generic`
when that tag exists, else `nightly-generic`, pinned by registry digest. Head
is rebuilt with `docker/runtime.Dockerfile --target runtime` only when product
paths change (`src/`, `python/`, `cmake/`, `docker/`, …); otherwise head is
the same digest as base (true A/A). Jobs use `--docker-context default` and
`flock` `/tmp/gpu-lock-xerxes-{0,1}` and `/tmp/gpu-lock-shelly-0` locally; the
kernel drops the lock when the holder exits, so a cancelled job cannot leave
it held. The old `mkdir` lock stayed behind when a job was cancelled, and it
failed outright while a hand-run `flock` had left a plain file at the path;
that file made `bench-perf-v100s` fail with `inconclusive`. A leftover
directory from the old scheme still works as a `flock` target. To hold a GPU
by hand, run `flock /tmp/gpu-lock-xerxes-1 <command>`. Indices
are **nvidia-smi / NVML / docker `--gpus device=N`**, matching the harness
table — on xerxes, GPU 0 = Tesla V100S (PCI 42:00.0, lock-0), GPU 1 = RTX 3070
(PCI 43:00.0, lock-1); on shelly, GPU 0 = RTX 5060 Ti (lock-0). Do not derive
them from CUDA's default FASTEST_FIRST ordering (the reverse on xerxes).
V100S / RTX 3070 / RTX 5060 Ti jobs also take GitHub `concurrency` groups
`gpu-xerxes-v100s` / `gpu-xerxes-3070` / `gpu-shelly-5060ti`
(`cancel-in-progress: false`) so they queue behind `images.yml` smoke; the
isolated job shares `gpu-xerxes-3070`. `v100s`, `rtx3070` and `rtx5060ti` are
`shared_device: true` in the harness target table: the idle guard records
foreign GPU processes (name, MiB, utilisation) and never refuses because of
them. The lock still keeps our benches from overlapping each other.
Shared-device results are flagged in the PR comment (`shared GPU, foreign
load: …`). `rtx3070-isolated` is the only exclusive GPU target.

`rtx5060ti` (shelly-1, NVML GPU 0, lock `/tmp/gpu-lock-shelly-0`, concurrency
`gpu-shelly-5060ti`) is **report-only** and `shared_device`. VLLM and an owner
Python service occupy ~7 GiB of 8 GiB; CUDA OOM is recorded as `error` (not a
hang). There is no Nsight Systems on shelly, so A/B uses fzb `sample_s` / wall
instead of `gpu_total_ms` from nsys. If generic SSCP JIT cannot target sm_120,
the harness records `unsupported` and does not block. Leave
`bench-perf-rtx5060ti` optional.

`gfx1012` (bubs / mbp-1, lock `/tmp/gpu-lock-bubs-amd-0`) is **report-only**
and `shared_device`, and the only HIP target. The host desktop keeps ~4 GiB of
the 8 GiB VRAM and the `fuzzybits-ci` runners share bubs' CPU, which feeds
the host side of each pass. Occupancy comes from sysfs, not NVML: KFD lists
every process holding `/dev/kfd` under `/sys/class/kfd/kfd/proc/<pid>` (with
per-GPU `vram_*` bytes) and amdgpu exposes `gpu_busy_percent` /
`mem_info_vram_*` on the render node. Desktop compositing does not open KFD,
so it shows up as VRAM and utilisation, not as a compute process. There is
no profiler; A/B uses `sample_s`. The job has no `concurrency` group because
mbp-1 is the only `amd-gpu` runner.

`workflow_dispatch` accepts `inject_regression` (`scale:1.10`, `scale:1.05`,
`repeat`) to confirm the screen flags twice and the isolated run blocks.

Require **`bench-summary`** only. It fails when accuracy fails, when images
fail, or when the screen flagged twice and `bench-perf-isolated` did not
pass. Requiring the GPU jobs themselves would block on report-only noise. Do
not require `windows-omp` or `windows-wheel` (#108).

### A/A noise (nightly-generic, identical images)

`nightly-generic` must be pinned by digest before A/A. Local config ids were
host-specific; the registry manifest is the pin.

| target | model | metric | rounds kept | ratio | 95% CI | half-width | IQR | verdict |
|---|---|---|---|---|---|---|---|---|
| V100S | chinese | gpu_kernel_ms (then) | 5 | 1.004 | [0.991, 1.013] | 1.1% | 0.47% | pass |
| CPU (container wall, chinese 1e8) | chinese | cpu_wall_s | 5 | 1.089 | [0.887, 1.148] | 13% | 17% | pass (unusable) |
| CPU | chinese | cpu_sample_s | 9 | 0.999 | [0.963, 1.027] | 3.2% | 4.6% | pass (report-only) |
| CPU | das9201 | cpu_sample_s | 9 | 1.011 | [0.977, 1.035] | 2.9% | 2.8% | pass (report-only) |
| CPU | das9701 | cpu_sample_s | 9 | 1.040 | [0.964, 1.097] | 6.6% | 8.9% | pass (report-only) |
| RTX 5060 Ti (shared, no nsys) | chinese | gpu_total_ms (`sample_s`) | 5 | 1.010 | [0.988, 1.039] | 2.6% | 1.7% | pass (report-only) |

Sampling times were in band (chinese ~2.9 s at 225·2^20, das9201 ~3.2 s at
80·2^20, das9701 ~3.5 s at 16·2^20) with `OMP_NUM_THREADS=8`,
`--cpuset-cpus 0-7`. CPU half-widths are **2.9–6.6%**, not ≤2%. Gateway load
rose to 10 during the run (ZFS + the 8-thread jobs). The CPU gate is
**report-only** (`CPU_GATE_BLOCKS = False`, [#124](https://github.com/fuzzybit-org/fuzzybits/issues/124)): it can warn, it does not block.

RTX 5060 Ti A/A used the same `nightly-generic` digest with VLLM
(`VLLM::EngineCore` ~3.5 GiB) and uvicorn/PaddleOCR (~3.4 GiB) resident,
~492 MiB free, util 0%. Generic SSCP JIT **does** run on sm_120 (AdaptiveCpp
25.10 / LLVM 20 `libllvm-to-ptx.so`; driver 595.99.02). LLVM warns
`'+ptx88' is not a recognized feature for this target (ignoring feature)`
and still produces a working kernel. There is no nsys on shelly, so
`gpu_total_ms` is fzb `sample_s` (~400 ms for chinese at 1e8, not the ~10 ms
kernel). The 95% CI already reaches 1.039, so a 3% `ci_lo` gate would be
unusable once VLLM is serving; the job is report-only. chinese at 1e8 fit
in the remaining ~0.5 GiB; larger models (nus9601) may record CUDA OOM as
`error` rather than hang.

Pinned-image A/A (registry digest
`sha256:97e5ffe141ce4f1a969f9a520264f940b0e5c256b9cf7780dfd6e2a9831d7803`)
and calibrated `scale:F` (later nightly
`sha256:4a0e01b84c4b5f96836f89eaa02d709bcd28136d3c16b954aa7c7c6eedc2b4ef`,
pass_size=`32·2^20` from `Session.sample_shape`):

| target | model | inject | metric | ratio | 95% CI | half-width | verdict |
|---|---|---|---|---|---|---|---|
| V100S | chinese | A/A | gpu_total_ms | 1.002 | [0.990, 1.007] | 0.87% | **pass** |
| V100S | chinese | A/A | gpu_kernel_ms | 1.001 | [0.989, 1.008] | 0.97% | pass (dual-gate) |
| V100S | chinese | scale:1.10 (10 vs 11 passes) | gpu_total_ms | 1.098 | [1.094, 1.103] | 0.44% | **block** |
| V100S | chinese | scale:1.10 | gpu_kernel_ms | 1.098 | [1.094, 1.104] | 0.52% | block (dual-gate) |
| V100S | chinese | scale:1.05 (20 vs 21 passes) | gpu_total_ms | 1.050 | [1.048, 1.051] | 0.17% | **block** |
| V100S | chinese | scale:1.05 | gpu_kernel_ms | 1.049 | [1.048, 1.051] | 0.17% | block (dual-gate) |
| CPU | chinese | A/A | cpu_sample_s | 0.993 | [0.977, 1.029] | 2.6% | pass (report-only) |
| CPU | chinese | scale:1.10 | cpu_sample_s | 1.020 | [0.861, 1.171] | 15.5% | pass (report-only; CI includes 1) |

`--inject-regression scale:F` does **not** multiply the configured trial
count and round to 2^20. It probes
`Session(..., num_trials=configured).sample_shape["trials_per_pass"]` (do
not assume 2^20; chinese on V100S is 32·2^20). A 1-trial `fzb run --json`
reports `n=64` (one uint64 bitpack) and is **not** the device pass. Then it
picks the smallest integer N ≥ ceil(configured_trials / pass_size) such
that round(N·F)/N is within 1% of F. Base runs N passes; head runs
round(N·F) passes. For F=1.10 that is 10→11 (realized 1.10×). The earlier
2^20 round-up turned chinese's 3 GPU passes into 4 (1.33×, measured 1.317);
that is no longer the acceptance path. `repeat` remains the 2× check. CPU
`scale:1.10` is reported separately; it does not block (#124).

### Timeouts

Every `fzb compile` / `inspect` / `run` is bounded by the model's
`time_budget_s` (manifest). The inner command is wrapped in GNU
`timeout --signal=TERM --kill-after=5`; the container is named
`fzb-bench-*` so a host-side wall-clock expiry can `docker stop` it.
A timeout is recorded as **`status: timeout`** with elapsed time and
whatever trials/estimate were produced — it is **not a z-test failure**
and rows are not dropped. Overall `timeout` is visible and is not an
accuracy pass. In A/B, **head timeout / base ok is a regression
(`block`)**; both sides timing out is `timeout`.

### Hosts

| target | host (docker context) | device | time? |
|---|---|---|---|
| `v100s` | xerxes `--gpus device=0` (NVML GPU 0, PCI 42:00.0) | Tesla V100S | yes; shared with live/dev fuzzybits-server + **report-only**; take `/tmp/gpu-lock-xerxes-0` |
| `rtx3070` | xerxes `--gpus device=1` (NVML GPU 1, PCI 43:00.0) | RTX 3070 | yes; shared (`shared_device`) + **report-only** screen; take `/tmp/gpu-lock-xerxes-1` |
| `rtx3070-isolated` | same device as `rtx3070` | RTX 3070 | yes; **exclusive**, blocking; take `/tmp/gpu-lock-xerxes-1` |
| `rtx5060ti` | shelly `--gpus device=0` (NVML GPU 0, sm_120) | RTX 5060 Ti 8 GB | yes; shared + **report-only**; take `/tmp/gpu-lock-shelly-0` |
| `gfx1012` | bubs (`bubs-mbp`), no `--gpus`: `scripts/ci/amd-docker-args.sh` → `/dev/kfd` + render node + render gid; backend `hip` | Radeon Pro 5500M 8 GB (Navi 14, RDNA1) | yes; shared + **report-only**; take `/tmp/gpu-lock-bubs-amd-0` |
| `cpu` | gateway | i7-9700, 8 cores | yes |
| — | xerxes CPU | noisy (Tdarr/Plex) | **no** |
| — | shelly CPU, bubs | CI (`fuzzybits-ci`) | **no** CPU timing |

GPU jobs take the lock. The exclusive target (`rtx3070-isolated`) waits for
idle compute apps and fails `inconclusive` if the device stays busy. Shared
targets (`v100s`, `rtx3070`, `rtx5060ti`, `gfx1012`) record foreign processes
and time anyway. Do not time GPU work without the lock.

### Injected regression (acceptance)

`--inject-regression scale:F` calibrates **both** sides to integer device
passes so the extra work is actually ≈F, not a 2^20/pass round-up:

1. Probe pass size: `Session(pcfb, num_trials=configured).sample_shape["trials_per_pass"]`.
   Do not use `fzb run --trials 1 --json` (`n=64` is one bitpack).
2. N = smallest integer ≥ ceil(configured_trials / pass_size) with
   round(N·F)/N within 1% of F. Examples: F=1.10 → 10 vs 11; F=1.05 → 20
   vs 21; CPU chinese at 225·2^20 scales to 230 vs 253 (exactly 1.10×).
3. Base = N passes; head = round(N·F) passes.

V100S must **block** (`ci_lo > 1.03` on `gpu_total_ms`) at F=1.10 and at
F=1.05 (noise is ~±1%). `--inject-regression repeat` still runs `fzb run`
twice under one profiler session on head (≈2× GPU time).
`--inject-regression sleep:MS` sleeps on head *outside* the sampled region, so
it no longer moves `cpu_sample_s`.

CPU extra trials **do** land inside the timed `cpu_sample_s` region:
`INNER_INPROCESS` times `fuzzybits.run(pcfb, trials=head_trials)` after
AdaptiveCpp init and a 2^20 JIT warmup. The uncalibrated CPU `scale:1.10`
of **1.020 [0.861, 1.171]** requested 248 vs 225 passes (1.102×) on that
timed call; the CI includes 1 because gateway loadavg rose to ~6 on 8
cores, not because extra trials were outside the timer. CPU stays
report-only (#124).

### generic-pwr-model compatibility shim

The converted OpenPSA XMLs (submodule `ce2ef006…`) drop some gate / basic-event
definitions into sibling files: `data/saphsolve/*.JSInp` and
`data/openpsa/be_name_label_and_value_V1.2.csv` (BE/G/S/INIT/FE/FT prefixes;
`BE0` is the single-child placeholder). That is not MEF-compliant. v1 does
**not** patch the submodule. `benchmarks/datasets/compat/generic_pwr.py`
deterministically rewrites a cached copy, keyed by source hashes and shim
version, and only for `source: pwr`:

| model | exporter gap | shim |
|---|---|---|
| `EQK-BIN4/Grp-5` | `G3` missing in FT42/FT44; `G10` missing in FT51 (not defined in any of the 95 XMLs; present in the matching JSInp as `or` of G15/G19/G24/G27/G39/G40) | inject those `define-gate`s into the referencing fault tree |
| `EQK-BIN4/Grp-6` | `G3` is private in FT3; G300 in other FTs references it without a path. After cloning G3, MEF still rejects cyclic `FE56`/`FE57` fork nesting (`FE56` then `FE57` on some paths, the reverse on others). | clone `G3`; **known failure** `validity_error` — not reconstructable without rewriting the event tree |
| `EQK-BIN5/Grp-2` | fork `functional-event="F44"`; defined name is `FE44`. Same G3 clone gap | rewrite `F44`→`FE44`; clone `G3` |

The engine MEF loader stays strict. `fzb compile` now prints
`MEF initialize: Undefined Element Error; reference=…; base_path=…` from
`boost::errinfo` so a future gap names the missing element. There is no
engine-side opt-in to accept undefined names. Private cross-FT refs that
already have a per-FT `define-gate` clone are left alone. Manifest rows
record `compat: {shim: generic_pwr, version: 1}`. If a file still cannot be
reconstructed it stays as a known compile failure with a reason. Shim v1
compiles Grp-5 and Grp-2; Grp-6 remains `compile.expected = validity_error`
because FE56/FE57 fork order is cyclic.

This shim is temporary. [#125](https://github.com/fuzzybit-org/fuzzybits/issues/125)
(SAPHSOLVE `.JSInp` import: recover generic PWR models, general parser) will
read the sibling JSInp files directly. Delete
`benchmarks/datasets/compat/generic_pwr.py` once #125 recovers those models.

## Reference estimators and FTP conformance fixtures

[#159](https://github.com/fuzzybit-org/fuzzybits/issues/159) requires every
reference value to come from a method other than the tool being scored.
`benchmarks/reference/` ([#179](https://github.com/fuzzybit-org/fuzzybits/issues/179))
provides those methods for CAFTA `.ftp` projects. Consumers: the FTP
reader (#180), the lowering (#181), the monotone-hull work (#174, #178, #183)
and the private large-model card (#185).

### Independence

The package needs only NumPy. It imports nothing from `fuzzybits`, `src/`,
`python/` or the harness (`benchmarks/__init__.py` loads the manifest loader
lazily for this reason); `test_independence_from_fuzzybits` enforces both
the loaded modules and the import statements. It has its own minimal FTP
parser (`benchmarks/reference/ftp.py`). That parser is **reference-only, not
the product reader**: it follows `src/io/ftp/schema/ftrex_ftp.g4` and the
conventions the grammar leaves implicit (a line that is not `NAME op ...`
continues the open gate, a leading `-` complements a reference, `PROCESS` may
continue on the next line, `*` lines are headers, `;` starts a comment, names
and keywords ignore ASCII case; the full list is
[FTP input: Grammar](ftp-input.md#grammar)), and it reports diagnostics instead
of guessing.

### FTP grammar checks

Vendor tools were run on small models to settle what the grammar
requires ([#191](https://github.com/fuzzybit-org/fuzzybits/issues/191)). Where
both reject a file, or skip lines and give the events those lines would define
p = 1, the reference parser reports an error, so a misread file never produces
a value. Sections never continue from one file into the next.

| diagnostic | severity | rule | Vendor tools |
|---|---|---|---|
| `missing-endtree` | error | gate lines not followed by `ENDTREE` before `PROCESS`, `IMPORT`, `LIMIT` or the end of the file | reject the file |
| `import-without-header` | error | a record `value name [I]` outside an `IMPORT` section | skip it (the event gets p = 1); after `PROCESS`, reject the file |
| `undefined-reference` | error | a referenced name is neither a gate nor imported | p = 1, with a warning from a vendor tool |
| `misplaced-gate` | error | a gate line after the tree section of its file | skip it (second tree section) or reject the file (tree after `IMPORT`) |
| `undefined-process-target` | error | a `PROCESS` name is neither a gate nor imported | reject the file |
| `missing-process` | warning | no `PROCESS`: the targets are the root gates | pick the root, with a warning from a vendor tool |
| `duplicate-import` | warning (`last`, `first`), error (`error`) | an event imported more than once | the last value, with a warning; the `I` tag is kept |

A file has at most one tree section, and it comes first. An `IMPORT` section
ends at the next keyword, header line or line that is not a record, so records
after `LIMIT` or after a header line are not read. Header lines may appear
between gate lines. Skipped lines define nothing, so `import-without-header`
and `misplaced-gate` are usually followed by `undefined-reference` for the
names they would have defined. Vendor tools read one file; a multi-file
project is a convention of this parser and #180, and its files are not
equivalent to their concatenation (the second tree section would not be read).
Vendor outputs are not committed.

### Semantics

- **Boolean:** $P(g)$ of the literal logic. Events at p = 0 and p = 1 are
  constants.
- **Hull** (#174, #190, #199; [RFC](rfc/monotone-hull-queries.md) §3): the
  plain monotone hull $\uparrow f(x) = \exists\, y \le x : f(y)$ of the
  target, with a literal mode for the events referenced with a direct
  complement (`-NAME` on an event) in the target's cone. These tools read
  only FTP, so they apply the FTP reader's convention: every `-EVENT` is a
  marked literal (the FTP reader always marks it), every `-GATE` a
  negation. Models from sources without such literals get the plain hull.
  - `--literals split` (default, the FTP convention; `fzb compile --hull-literals split`), the *atom-split hull*:
    every such `-NAME` is
    an atom of its own, valued not x and dropped independently of `NAME`; the
    hull is taken over all atoms on the coupled sample, and cut sets list the
    atom as `-NAME` (a cut set with both `NAME` and `-NAME` has probability 0
    and is not listed).
  - `--literals omega`: those events are *fixed* (a witness keeps their value):
    Rauzy's $\omega_L g$ (`fzb compile --hull-literals fixed`).
  - `--literals positive`: every event droppable, the plain hull of the literal
    target (`fzb compile --hull-literals plain`).

  A complemented gate, also a single-input one, is a delete term in every mode.
  `split` and `omega` agree wherever no directly complemented event also
  occurs under a complemented gate; elsewhere the tools report a warning
  `complement-under-negation` (they can differ there, and `split` can exceed
  the plain hull). `positive` drops the complements and differs whenever one
  matters (T2 below). Droppable variables at p = 1 are not constants;
  variables at p = 0 are constants. The atom `-NAME` of an event at p = 0 is a
  droppable variable at p = 1, not TRUE.
- **Initiator rule** (#190; `fzb compile --hull-max-initiators N`, hull queries only): `--max-initiators-per-cutset N` conjoins
  $\neg\,\mathrm{atleast}(N+1;\ I)$, with $I$ the `I`-tagged events in the
  target's cone (p = 1 initiator flags count). Under the hull this deletes the
  minimal cut sets with more than N initiators (the vendor tools use N = 1);
  under Boolean semantics it is the literal conjunction.
- **Overrides** `NAME=0|1` replace a gate or an event by a constant *in the
  function*. They are not probabilities: under the hull, an event overridden
  to 1 is a constant TRUE, while an imported p = 1 event stays droppable.
- **Duplicate imports:** `--duplicate-import error|first|last`, default
  `last` with a warning, as vendor tools do; `first` also warns, `error`
  refuses evaluation. An `I` tag on any of the records keeps the event an
  initiator.

### Methods

| command | method | scope |
|---|---|---|
| `exact --semantics boolean\|hull` | enumeration of every variable in the cone (≤ 22: events, and for `split` the complement atoms), weights include p = 0 and p = 1 (for `split`, an atom and its event take opposite values); the hull is the subset-OR (zeta) transform over the droppable axes. No folding or polarity reasoning, so it checks the estimators independently. `--cut-sets` (hull only) lists the minimal cut sets with `-NAME` for complement atoms (`split`) or negative literals (`omega`: $\operatorname{MC}_L(g)$, the prime implicants of $\omega_L g$). | small models |
| `estimate --semantics boolean` | bit-parallel Monte Carlo, PCG64 seeded, 64 lanes per word; Binomial-position draws for p < 1/32, uniforms otherwise. The circuit is folded first (constants, aliases, structural hashing); a target that folds to a constant is reported exactly (`method: constant-folding`). | any size |
| `estimate --semantics hull` | two-stage estimator below | any size |

Hull estimator, per pass:

1. Sample the events (for `split`, an atom and its event are drawn once and
   complement each other) and evaluate $g$ and the coherent bound together:
   complemented gate inputs, and event literals that are not kept, replaced by
   TRUE (atoms and fixed literals are kept). Lanes with $g = 1$ are in the
   hull, lanes with coherent bound 0 are not.
2. Redraw the pass (saved generator state), gather the candidate lanes and run
   a Kleene (three-valued) evaluation in dual rail. Variables false in the lane
   are 0 and fixed events keep their value; true droppable variables reachable
   from the target only through an even number of complements are fixed to 1,
   only through an odd number to 0 (optimal for satisfiability, which drops
   p = 1 flags under a complement); true binate droppable events are free.
3. Lanes still undetermined (up to `--max-checks`) get an exact check on the
   residual circuit of undetermined nodes: repeated pure-literal fixing, then
   a truth table (≤ 16 free events), a BDD with a node budget
   (`--bdd-budget`), or a complete branching search.

With $N$ trials, $n_T$ lanes decided true, $n_C$ undetermined lanes of which
$m$ were checked and $h$ were in the hull, the estimate
$\hat P = (n_T + n_C h / m) / N$ is unbiased, and with $\pi_T = n_T/N$,
$\pi_C = n_C/N$, $f = h/m$,

$$\widehat{\mathrm{Var}}(\hat P) = \frac{\pi_T + f^2 \pi_C - \hat P^2}{N} + \frac{\pi_C^2 f (1 - f)}{m},$$

which is $\hat P(1-\hat P)/N$ when every undetermined lane is checked. The
result reports `literals`, `complemented_events`,
`complement_under_negation`, `max_initiators_per_cutset`, `estimate`, `se`,
`ci95`, the per-run bounds $\hat P(g) \le \hat P \le$ coherent bound, lane counts per stage, the
checked fraction, `lanes_per_second` of the exact check, residual sizes and
which decision procedure was used.

Results are deterministic for a given seed, trial count and `--memory-mib`
(which fixes the pass size). Trials are rounded up to a multiple of 64.

### CLI

```bash
python -m benchmarks.reference structure input/ftp/conformance/multi_file_project
python -m benchmarks.reference exact --semantics hull input/ftp/conformance/delete_term_flags
python -m benchmarks.reference exact --semantics boolean --override DEL=0 \
    input/ftp/conformance/delete_term_flags
python -m benchmarks.reference estimate --semantics hull --trials 1e6 --seed 1 \
    --max-checks 2000 --target SEQ-3 --json input/ftp/conformance/success_branch_delete_term
python -m benchmarks.reference exact --semantics hull --cut-sets \
    --target T2 input/ftp/conformance/process_multi_target          # {-B, C}, {-B, D}
python -m benchmarks.reference exact --semantics hull --literals positive \
    --target T2 input/ftp/conformance/process_multi_target          # plain hull
python -m benchmarks.reference exact --semantics hull --literals omega --cut-sets \
    input/ftp/conformance/complement_under_negation                 # warns; differs from split
python -m benchmarks.reference exact --semantics hull --max-initiators-per-cutset 1 \
    --cut-sets input/ftp/conformance/initiator_rule
```

`FILES` are `.ftp` files or directories (all `*.ftp`, sorted) read as one
model, in order; order matters for `first`/`last`. Without `--target`, the
targets are the defined `PROCESS` targets, else the roots. Exit status 1
means error diagnostics (printed to stderr; `--json` puts them in the document).

### Conformance fixtures (`input/ftp/conformance/`)

One directory per fixture, ≤ 20 events each, with `expected.json`
(`schema_version` 2): hand-written `description`, `covers`, `files` (project
order), optional `targets` and `variants` (`policy`, `overrides`,
`max_initiators_per_cutset`); computed `default_targets`, `structure`
(policy-independent counts; an event counts as I-tagged or at a value when
any of its import lines is), and per duplicate-import policy the expected
diagnostics (`severity`, `code`, `name`, `locations`) and, per target, the
exact `boolean`, `hull` (atom-split, the default), `hull_omega` and
`hull_positive` probabilities, `complemented_events` and
`complement_under_negation` (`results` is `null` when the policy is an error).
`hull` and `hull_omega` are equal on every fixture except
`complement_under_negation`.

| fixture | covers | target | Boolean | hull (split = omega) | plain hull |
|---|---|---|---|---|---|
| `multi_file_project` | three files, cross-file gate references, separate IMPORT file | TOP | 0.00192544813 | same | same |
| `process_multi_target` | PROCESS with three targets over two lines, non-target root, complemented event | T1 / T2 / T3 | 0.622 / 0.464 / 0.098 | 0.622 / 0.464 / 0.098 | 0.622 / 0.58 / 0.098 |
| `no_process` | no PROCESS, two roots, no header lines | R1 / R2 | 0.208 / 0.024 | same | same |
| `imports_tags_duplicates` | I tags, values 0 and 1, duplicated import | TOP (first / last) | 0.1108 / 0.1072 | same | same |
| `initiator_rule` | two initiators in one cut set, p = 1 initiator flag; variant `max_initiators_per_cutset: 1` | TOP (no rule / rule) | 0.45 / 0.288 | 0.45856 / 0.424 | 0.5488 / 0.52 |
| `header_limit` | `**CHAR32`, `**DBNAME:`, `**FTITLE:`, `*XEQ`, LIMIT, no PROCESS | TOP | 0.224 | same | same |
| `success_branch_delete_term` | complemented gate references, SEQ = F ∧ ¬(S ∨ DEL) | TOP / SEQ-2 / SEQ-3 | 0.023 / 0.01785 / 0.00515 | 0.028 / 0.028 / 0.006325 | same as hull |
| `delete_term_flags` | DEL = two p = 1 flags; variants `DEL=0`, `FLAG-A=0` | SEQ | 0 (variants 0.00802) | 0.0107 | same as hull |
| `k_of_n_continuation` | k = 1, k = n, 1 < k < n, single-input gates, 564-character continuation line | TOP | 0.6372981032253088 | same | same |
| `names_special` | names starting with digits, containing `/` or `--`, complemented event | 1-TOP | 0.507 | 0.507 | 0.51 |
| `dialect_names_case` | `%`, `~` and other punctuation in names, names in another letter case, keywords and `I` flag in any case, CRLF | TOP | 0.5546875 | same | 0.5625 |
| `dialect_values_layout` | value `-1` as 1, values `.5`, `5.E-1`, `1.0D-01`, `;` comments, unindented continuation, `MAXSIZE`, tokens after `ENDTREE` | TOP | 0.4456 | same | same |

| `complement_under_negation` | target | Boolean | split | omega | plain hull |
|---|---|---|---|---|---|
| B complemented directly and under the negated gate G | P1 | 0.2736 | 0.316 | 0.296 | 0.37 |
| F complemented inside the negated gate H | P4 | 0.3 | 0.5 | 0.3 | 0.3 |
| P1 ∨ P4 | TOP | 0.49152 | 0.658 | 0.5072 | 0.559 |
| KM ∨ N¬M, no negated gate (no consensus {K, N} over atoms) | P3 | 0.5 | 0.5 | 0.5 | 0.78125 |

`imports_tags_duplicates` under `error` has no results and one
`duplicate-import` error. The vendor tools list exactly the minimal
cut sets of `exact --semantics hull --cut-sets --duplicate-import last` above
each fixture's `LIMIT`, with `--max-initiators-per-cutset 1` (the tools apply
that rule) and, with the `I` tags removed, without it, as checked for #190
([RFC](rfc/monotone-hull-queries.md) §8.8–§8.11), including
`complement_under_negation`, where omega and the plain hull do not match them.
For that check
`multi_file_project` must be merged into one tree section: the tools stop
reading a concatenation at the first `ENDTREE`/`*XEQ` and treat the gates of
later files as undefined events. Check or regenerate the computed fields:

```bash
python -m benchmarks.reference.fixtures                   # check
python -m benchmarks.reference.fixtures --write [NAME...] # after editing fixtures
```

Fixtures named `invalid_*` are diagnostic-only: every policy has errors, so
`expected.json` records the diagnostics per policy (results `null`) and no
`default_targets` or `structure`. Vendor tools reject or misread each of
them as described in its `description`.

| fixture | input | diagnostics |
|---|---|---|
| `invalid_no_endtree` | gate lines followed by `PROCESS` | `missing-endtree` |
| `invalid_import_without_header` | `header_limit` without the `IMPORT` line | `import-without-header` ×3, `undefined-reference` ×3, `missing-process` (warning) |
| `invalid_import_section_end` | records after a header line and after `LIMIT` | `import-without-header` ×2, `undefined-reference` ×2 |
| `invalid_undefined_event` | an event referenced but not imported | `undefined-reference` |
| `invalid_second_tree_section` | gate lines after `ENDTREE` | `misplaced-gate` ×2, `undefined-reference` ×2 |

`tests/python/test_reference_estimators.py` checks the fixtures (the
diagnostics of the `invalid_*` fixtures are also written out by hand in the
test), closed-form values and minimal cut sets for several of them, the
initiator rule on every fixture, Boolean and hull estimates (all three literal
modes) within 4 standard errors, the per-lane hull check against the exact
closure (truth table, BDD and search paths), brute force on random projects
(exact values, cut sets, per-lane check and estimates for all three modes,
with complements inside and outside negated gates) and random circuits
(per-lane check and bounds for every kind of significant literal), the atom
sampler's coupling, the `complement-under-negation` warning, the parser rules
and their diagnostics, seed determinism and the CLI.

### Private models

When `FUZZYBITS_PRIVATE_FIXTURES_DIR` is set, the `private` tests check every
`ftp/<MODEL>/` directory that has `<MODEL>.card.json`: the card's structure
counts, and each variant (`semantics`, `overrides`, `policy`, `expected`)
within 4 combined standard errors (Boolean: `FUZZYBITS_PRIVATE_TRIALS`,
default 2^22; hull: 2^18 trials, at most 400 checked lanes, default literal
mode). A variant's initiator rule is its `max_initiators_per_cutset`; a
variant that states a rule only otherwise (`extra_delete_term`) is skipped, as
is one that needs a file edit. A policy may carry a comment after its keyword
(`last (...)`). Variants with only an observed value are reported, not
asserted. Output is pass/fail and aggregates only: no
identifiers, per-event values or card values (#38).
`scripts/ci/docker-toolchain-run.sh` mounts the directory read-only at
`/private` and passes `FUZZYBITS_PRIVATE_FIXTURES_DIR=/private`.

```bash
FUZZYBITS_PRIVATE_FIXTURES_DIR=/path/to/private-fixtures \
  pytest tests/python/test_reference_estimators.py -m private -s
```

The fixtures are the private submodule `extern/private-fixtures` (see CONTRIBUTING.md, "Private fixtures"). CI
never fetches it.

## Published history and the Benchmarks tab

[#123](https://github.com/fuzzybit-org/fuzzybits/issues/123) publishes
single-build accuracy and performance for every `v*` tag and a nightly
`develop` point. Results are **never committed to `develop`**. They live on
the `bench-results` orphan branch and, once GitHub Pages is enabled, at

```
https://fuzzybit-org.github.io/fuzzybits/bench/v1/index.json
```

The web UI **Benchmarks** tab (`/benchmarks`, next to `/playground`) reads
that JSON. Configure the URL at build time (`VITE_BENCH_DATA_URL`) or at run
time (`window.__FUZZYBITS_BENCH_DATA_URL__`, `FUZZYBITS_BENCH_DATA_URL` on
`fuzzybits-server`). The server may serve a cached copy from
`FUZZYBITS_BENCH_CACHE_DIR` (a `bench-results` checkout, or a directory
containing `index.json`) at `/api/v1/bench/v1/index.json`. If the data
cannot be reached, the tab shows an offline message; the playground is
unaffected.

### How to read the tab

- **Latest release vs develop** per target: ratios are develop / release on
  the same runner and recipe. GPU colouring matches the PR gate: **block**
  (madder) when the point estimate exceeds 3% slower, **warn** (ochre) when
  it is slower but under 3%. CPU is **report-only** (#124): warn above 5%,
  never block; gateway noise is about ±3–7%. Accuracy `|z| > 4` is a block.
  Verdicts include `infeasible`, `pass (low power)`, and `timeout`.
- **Charts** for the selected model and target: GPU `gpu_total_ms` (all
  compute kernels) and `gpu_kernel_ms` (`basic_event`), CPU `cpu_sample_s`,
  accuracy z. A starred point changed image digest, driver, recipe, or
  runner versus the previous point on that target.
- Absolute numbers are comparable only on the same runner and recipe. CPU
  history is indicative.

### Backfill a version

Images: `ghcr.io/fuzzybit-org/fuzzybits-runtime:<tag>-generic` (and
`:nightly-generic` for develop), pinned by registry digest. The workflow
always checks out **this harness**, not the old tag source (`v0.17.0` has
no harness).

```bash
# Actions → Bench history → Run workflow
#   version: v0.17.0 | v0.17.1 | develop
#   deploy_pages: leave off until Pages is enabled
gh workflow run bench-history.yml --repo fuzzybit-org/fuzzybits \
  -f version=v0.17.1 -f deploy_pages=false
```

Or the same recipe locally (take the GPU locks; do not time CPU on
xerxes/shelly/bubs):

```bash
python3 -m benchmarks run --target v100s --mode accuracy \
  --image ghcr.io/fuzzybit-org/fuzzybits-runtime:v0.17.1-generic \
  --out /tmp/acc.json
python3 -m benchmarks run --target v100s --mode perf \
  --image ghcr.io/fuzzybit-org/fuzzybits-runtime:v0.17.1-generic \
  --out /tmp/perf.json
python3 -m benchmarks.history merge --accuracy /tmp/acc.json --perf /tmp/perf.json \
  --version v0.17.1 --out /tmp/merged.json
python3 -m benchmarks.history publish --results /tmp/merged.json --dest /path/to/bench-results
```

Older entries are never rewritten; a rerun adds a timestamped file
(`…Z-2.json`). The workflow uses per-GPU concurrency groups
`gpu-xerxes-v100s` / `gpu-xerxes-3070` / `gpu-shelly-5060ti` (same as
runtime-image smoke) plus the harness locks `/tmp/gpu-lock-xerxes-0` (V100S,
`device=0`), `/tmp/gpu-lock-xerxes-1` (RTX 3070, `device=1`), and
`/tmp/gpu-lock-shelly-0` (RTX 5060 Ti, `device=0`).

Until GitHub Pages is enabled (**Settings → Pages → Build and deployment →
Source = GitHub Actions**), keep the JSON on `bench-results` and point the
UI at a local copy:

```bash
export FUZZYBITS_BENCH_CACHE_DIR=/path/to/bench-results
fuzzybits-server --data-dir ./input
```
