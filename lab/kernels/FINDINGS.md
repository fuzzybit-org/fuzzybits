# kernels: findings

## Bit-sliced Bernoulli sampler (#164, #17)

`microbench/bench.cpp`: 122 events × 131072 bitpacks (1.02e9 trials) per launch, p = 0.01, median of 15
launches, every variant's bits checked against the host reference. Measured before the sampler merged, on the
`v25.10.0-generic` toolchain.

| device | 32-bit sampler | + split multiply (#16) | bit-sliced, merged form | speedup |
|---|---:|---:|---:|---:|
| Tesla V100S | 3.270 ms | 2.773 ms | 0.742 ms | 4.41× |
| RTX 3070 | 3.694 ms | 2.956 ms | 0.851 ms | 4.34× |
| RTX 5060 Ti | 3.769 ms | 3.248 ms | 0.672 ms | 5.61× |
| Radeon Pro 5500M (HIP) | 15.066 ms | 19.525 ms | 4.697 ms | 3.21× |
| i7-10700, 8 threads (OpenMP) | 133.2 ms | — | 35.7 ms | 3.73× |

What decided the merged form:

- **GPU: refill on decision.** A plain bit-sliced loop runs until every trial in the sub-group is decided
  (about 12 steps for 32 × 64 trials). A work-item that starts its next bitpack as soon as one is decided pays
  the mean depth (about 7 steps): 13–20% faster.
- **Uniform event index.** Taking the event from the global id made per-event values per-lane to the
  compiler (15% slower on AMD); from the work-group index they stay uniform. A 2-D range would cap CUDA at
  65,535 events (grid y), so the range is 1-D.
- **Multiply per backend.** NVIDIA is faster with a separate 32-bit low multiply (#16), AMD with one 64-bit
  product (`v_mad_u64_u32`). AdaptiveCpp's JIT reflection picks one at JIT time; guard it with `is_known()`.
- **CPU: explicit vectors.** The host JIT vectorizes across work-items only when the kernel has no
  data-dependent loop, so the scalar bit-sliced loop is no faster than the old vectorized kernel. Explicit
  16-lane vectors with even/odd `pmuludq` gave 3.7×; widening u32 vectors to u64 instead costs cross-lane
  shuffles. 16 lanes beat 8; per-lane refill was slower (masked stores, per-lane threshold bits).
- **Form as a template parameter.** A runtime flag that selects the form compiles both into one kernel (243
  registers on sm_70).
- Philox4x32-7 would be about 20% faster; not taken.
- Nsight Compute, V100S: `basic_event` 29 registers and no local memory; gate kernels 32 registers; tally 22.

End to end against develop's image (`sha-eef541a2da42-generic`), `fuzzybits.run` time head / base, 6
alternating rounds, `libcudart` preloaded on both sides for CUDA:

| model | V100S | RTX 3070 | RTX 5060 Ti | Radeon Pro 5500M | i7-9700, 8 threads |
|---|---:|---:|---:|---:|---:|
| chinese | 0.673 | 0.722 | 0.562 | 0.460 | 0.520 |
| das9201 | 0.494 | 0.528 | 0.417 | 0.439 | 0.456 |
| das9701 | 0.794 | 0.806 | 0.786 | 0.889 | 0.865 |
| edf9203 | 0.465 | 0.558 | 0.504 | 0.654 | 0.756 |
| nus9601 | 0.468 | 0.486 | 0.430 | 0.303 | 0.269 |

Two private plant models (about 91k operators and 2.9k basic events each, with some k-of-n gates), 2^26
trials (CPU: 2^20): 0.92 and 0.92 on the RTX 5060 Ti, 0.77 and 0.78 on the CPU. On the V100S (1.00, 0.94)
and the RTX 3070 (0.98, 0.96) the 95% intervals span 1: that host's load average was 13–72 during the runs.

## Measurement pitfalls

1. **CUDA context per run (#268).** With `ACPP_PERSISTENT_RUNTIME=0` (the Python default and the bench
   harness), AdaptiveCpp unloads its plugins after every `fuzzybits.run`. When the omp plugin loads before
   cuda, `libcudart` goes with them and the next run re-creates the CUDA context: about 0.3 s per call. Load
   order is the `readdir` order of `fuzzybits/lib/hipSYCL`, which differed between images and between hosts
   for the same image (both hosts use ZFS-backed Docker storage). Preloading the image's `libcudart` on both
   sides removes it. HIP on gfx1012 is not affected.
2. **One GPU A/B per host.** The private plant models are partly bound by the host on a slow, loaded CPU;
   two A/Bs on one host at once moved base times between 12.4 and 25.3 s.
3. **`ws.sh` toolchain (#269).** It hashes only `toolchain.env`, CI also hashes
   `docker/toolchain.Dockerfile`, so `ws.sh` falls back to the moving tag and keeps a stale local copy. Create
   workspaces with `TOOLCHAIN_IMAGE` set to the CI tag.
4. **`ACPP_APPDB_DIR`.** Runtime images contain a root-owned `/tmp/acpp-appdb`; pointing `ACPP_APPDB_DIR`
   there makes AdaptiveCpp abort (it cannot create its app directory). Use
   `/home/coder/.cache/fuzzybits/acpp`, as the harness does.
5. **Tracing cost.** Nsight Systems adds a cost to every launch: a traced run of 81k short kernels spanned
   2.3× its untraced wall time. Compare the traced kernel total with an untraced run.
6. **GPU % is not occupancy.** NVML's utilization counts any sample window in which a kernel ran: 98% while
   kernels covered less than half of a traced run.
7. **Short timed runs.** The harness's GPU A/B samples 1e8 trials per container (0.02–0.85 s on Aralia)
   after 5–15 s of start-up, JIT and model build, so the GPU looks idle in a monitor during an A/B.
8. **`python -m benchmarks ab --build`** cannot work as is: `git archive` omits the pcfb submodule and Docker 29
   refuses `docker build -f FILE -` with a tar on stdin. `ab/build_image.sh` builds on the docker host instead.

## Pipeline probes (head image)

- **Passes per host wait** (`Session.step(n)`, what `FUZZYBITS_PASSES_PER_STEP` sets in the server; Mtrials/s
  at n = 1 and the best n): private plant model 9.3 → 9.6 (RTX 5060 Ti), 6.3 → 6.6 (RTX 3070), 6.6 → 6.9
  (V100S); nus9601 and das9701 1–9%. Not a factor.
- **Pass size** (`budget_sweep.py`, RTX 5060 Ti, private plant models, 2^26 trials): the 1 GiB ceiling gives
  65,536 trials per pass; 2 GiB, 4 GiB and 90% of free memory give 131,072, 262,144 and 524,288, at 9.0, 9.4,
  8.7 and 8.4 Mtrials/s (peak 6,126 of 8,151 MiB). No gain there: the traced kernel total equals the untraced
  wall time, so that GPU is the bottleneck, and its time goes mostly to gates (basic events are about 3% of the
  nodes). The shaper rounds a pass down to a power of two bitpacks per node, so a pass uses between half and
  all of its budget.
- **Aralia steady state** (`Session`, 2^26 trials, Mtrials/s; V100S / RTX 3070 / RTX 5060 Ti): das9701 653 /
  378 / 377; nus9601 376 / 275 / 305.

## Open

- The budget sweep on the V100S (32 GB) and the RTX 3070, where the host is slower relative to the GPU.
- Whether pass sizes that are not a power of two, or not a multiple of 64 bitpacks, stay unbiased in the gate
  and tally kernels: not checked. The sampler's counters are per bitpack, so its stream does not depend on it.
