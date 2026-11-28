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

## Cardinality gates (#274)

`cardinality/bench.cpp`: G gates × 16384 bitpacks (1.05e6 trials per gate), G × n ≈ 2048 inputs per launch
drawn from a pool of 1024 buffers (128 MiB, larger than every L2), input densities near each gate's bounds,
median of 15–21 launches, every variant's bits checked against a host count. CPUs: 4096 bitpacks per gate.
`prod` is a replica of the at-least kernel this work replaced (one work-group of 64 work-items per bitpack,
8-bit count, two group reductions); `lowered` is the runtime's former lowering of at most (NOT over at least)
and exactly (two at-least gates, NOT and AND), each a pass of its own with the gate kernels' local range.

The new kernels against the replaced ones (at least k: the kernel; at most k: at least plus a NOT pass; exactly
k: two at-least gates, a NOT and an AND pass) with the runtime's lowering (at most k: at least n - k of the
complements; exactly k: the AND of at least k and at least n - k of the complements), median (min–max) speedup
over the shapes of 3 to 256 inputs. The new kernels use a compile-time input count up to 8 inputs and the
8-input carry-save counter above; production's slot counts up to 16 are as fast or faster (below).

| device | at least k (13 shapes) | at most k (4) | exactly k (4–5) | 2-of-3, 682 gates |
|---|---:|---:|---:|---:|
| Tesla V100S | 16.8× (12.4–25) | 19.4× (13.8–26) | 17.1× (13.7–34) | 19.07 → 0.392 ms |
| RTX 3070 | 12.2× (6.3–33) | 21.5× (7.6–33) | 19.3× (7.4–28) | 32.97 → 0.884 ms |
| RTX 5060 Ti | 11.3× (5.5–24) | 17.1× (7.2–25) | 15.4× (6.8–22) | 24.61 → 0.817 ms |
| Radeon Pro 5500M (HIP) | 28× (19–46) | 40× (20–56) | 14.0× (7.2–21) | 88.34 → 1.724 ms |
| 2× Xeon E5-2697 v2 (AVX, loaded host) | 108× (12.5–242) | 116× (83–174) | | 356 → 1.65 ms |
| i7-10700 (AVX2) | 78× (35–91) | 98× (63–118) | 52× (34–75) | 267 → 2.67 ms |

The replaced kernel's count and threshold were 8 bits: gates with 256 or more inputs or a threshold of 256 or
more gave wrong bits on every device.

What decided the forms:

- **One bitpack per work-item.** 64 work-items per bitpack each counted one trial, then two group
  reductions merged the bits. Bit-sliced counting (plane p holds bit p of each trial's count) does all the
  trials of a bitpack with whole-word logic; every new variant runs at or near the device's DRAM bandwidth
  (V100S 920–955 GB/s, RTX 3070 415 GB/s, RTX 5060 Ti 438 GB/s).
- **Fixed input counts up to 16, a runtime count above.** With the count unrolled for a compile-time number
  of slots, a work-item issues all its loads at once: 1.24× over a runtime-n thermometer loop on the V100S
  (0.390 vs 0.482 ms, 2-of-3). Runtime bounds cost nothing measurable (0.392 vs 0.390 ms for compile-time
  bounds). The production slot counts (1, 2, 3, 4, 6, 8, 12, 16, padded with an input mask whose padding
  slots are not read) are within 0.6–4.0% of an exact count without a mask, and 0–24% faster than the
  runtime-n counter (V100S, ms):

  | inputs | 3 | 5 (6 slots) | 7 (8) | 9 (12) | 12 | 13 (16) | 16 | 17 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | exact compile-time count, no mask | 0.393 | 0.356 | 0.342 | 0.329 | 0.322 | 0.326 | 0.309 | |
  | production slot counts with mask | 0.396 | 0.365 | 0.352 | 0.342 | 0.326 | 0.333 | 0.311 | |
  | runtime-n counter | 0.522 | 0.402 | 0.364 | 0.360 | 0.342 | 0.344 | 0.309 | 0.316 |

  Above 16 inputs a fixed count loses: 24, 32 and 64 slots are 3–18% slower than the runtime-n counter on
  the V100S (32 slots for 17 inputs: 0.330 ms), and 64 slots 4–13% slower on the Radeon, even without
  padding (more registers, a predicate per slot). The RTX 3070 and 5060 Ti are at bandwidth either way.

- **Counter: blocks of 8.** Harley-Seal over 16 inputs was 7–20% slower than a ripple counter on the
  Radeon; over 8 inputs it is best or within 6% everywhere. The plain ripple counter is 1.6–1.8× slower on
  the V100S for 256 inputs or more.
- **Registers decide the split.** A kernel's registers are those of its largest case: one kernel for every
  form took 122 registers on sm_70 (25% occupancy) and was 1.9× slower for 2-of-3 than the small form alone
  (20–32 registers). Capping the counter at 8 planes (fewer than 256 inputs) gained 3–16% on the V100S, so
  256 inputs or more is a third kernel.
- **Thermometer codes** (s_j = at least j + 1 so far, one three-input operation per level and input) tie the
  counter for small depths and lose above; not kept.
- **Popcount does not apply:** it counts across the trials of a word, and a gate counts across its inputs for
  each trial, so it would need a 64×64 bit transpose first (about 18 operations per word against about 4).
- **CPU: explicit vectors.** 8 bitpacks per work-item (`ext_vector_type(8)`) are 1.3–5× faster than one;
  4 and 8 lanes are within noise.
- **At most and exactly.** The PDAG has only at-least gates. At most k of n is at least n - k of the
  complements: one pass instead of two. Exactly k is the AND of the two at-least gates, one launch for both
  plus the AND pass. Former lowering / this lowering / one fused [k, k] kernel:

  | device | exactly 1 of 3, 682 gates | exactly 32 of 64, 32 gates |
  |---|---:|---:|
  | Tesla V100S | 38.6 / 1.14 / 0.41 ms | 7.84 / 0.57 / 0.28 ms |
  | RTX 3070 | 67.2 / 2.38 / 0.87 ms | 9.85 / 1.33 / 0.65 ms |
  | RTX 5060 Ti | 50.5 / 2.32 / 0.82 ms | 7.30 / 1.08 / 0.51 ms |
  | Radeon Pro 5500M | 221 / 30.0 / 1.73 ms | 64.2 / 3.10 / 1.49 ms |
  | i7-10700 | 529 / 12.3 / 2.78 ms | 174 / 2.33 / 1.58 ms |

- **The AND pass is the generic gate kernel's local range.** It gives the work-group budget to the gate
  dimension first: 682 gates get 512 gates × 2 bitpacks per work-group, so a wavefront spans 16–32 gates. The
  AND of 682 two-input gates moves 268 MB in about 26.5 ms on the Radeon (10 GB/s of 192 GB/s); NVIDIA keeps
  390–810 GB/s with the same range, the i7-10700 about a third of its bandwidth. Every generic gate kernel
  (AND, OR, NOT, …) of a layer with many gates has this range.

## AND, OR, XOR, NAND, NOR, NOT, NULL gates (#275)

`cardinality/bench_gate.cpp`: G gates × 16384 bitpacks, inputs from a pool of 1024 buffers (each read by about
2 gates; 4096-gate shapes by 4–12), densities that leave both outputs likely, every variant checked against a
host evaluation. `prod` replicates `op<OpType>` with the local range of
`working_set::compute_optimal_local_range_3d_for_gpu`, which gives the work-group budget to the gates first
(682 gates: 512 gates × 2 bitpacks per work-group).

Production forms against `prod`, ms. On GPUs: fixed slot counts with an input mask up to 16 inputs and blocks
of 8 loads above, one bitpack per work-item, block-major on NVIDIA and gate-major on the Radeon. On CPUs: one
counted kernel per connective, 8 bitpacks per work-item, block-major (NOT: one slot, gate-major).

| shape | V100S | RTX 3070 | RTX 5060 Ti | Radeon Pro 5500M | i7-10700 (4096 bitpacks) |
|---|---:|---:|---:|---:|---:|
| AND of 2, 1024 gates | 0.507 → 0.305 | 0.972 → 0.680 | 0.992 → 0.647 | 191 → 1.98 | 23.6 → 3.8 |
| OR of 3, 682 gates | 0.472 → 0.241 | 0.840 → 0.549 | 0.845 → 0.509 | 39.8 → 1.73 | 5.2 → 3.5 |
| AND of 16, 128 gates | 0.251 → 0.176 | 0.464 → 0.335 | 0.435 → 0.287 | 7.01 → 1.33 | 2.9 → 1.65 |
| NOT, 2048 gates | 0.612 → 0.423 | 1.309 → 0.876 | 1.334 → 0.858 | 246 → 2.71 | 32 → 6.1 |
| OR of 3, 4096 gates | 2.655 → 1.107 | 4.902 → 2.174 | 4.977 → 1.658 | 1121 → 10.4 | 89 → 18.9 |
| AND of 17, 120 gates (counted) | 0.247 → 0.188 | 0.466 → 0.342 | 0.438 → 0.289 | 7.17 → 1.34 | |
| AND of 64, 32 gates (counted) | 0.200 → 0.226 | 0.336 → 0.461 | 0.296 → 0.290 | 2.62 → 1.46 | 2.54 → 2.07 |

- **Work-group order is per backend.** Block-major (every gate at a block of bitpacks, then the next block)
  lets gates that share inputs read them from the cache: 1.4–2.4× over gate-major on the V100S, RTX 3070 and
  RTX 5060 Ti (beyond DRAM bandwidth on the 5060 Ti's 32 MiB L2). On the Radeon Pro 5500M it is 2.6–3.6×
  slower, also with a pool 16× smaller (so not the L2's size; the scalar cache cycling through every gate's
  descriptor and pointers is the likelier cause). CPUs do not care.
- **Fixed slot counts** are 5–21% faster than a runtime count on the V100S up to 16 inputs and tie elsewhere.
  End to end that gain didn't hold: a kernel per slot count tied the bench models and multiplied kernel
  launches. On a private model of about 88,000 gates in 147 layers it took 1,001 gate kernels per pass instead
  of 316, and ran 2.3–2.9× as long as develop on NVIDIA GPUs at the default budget. So #282 uses the counted
  form for AND, OR, XOR, NAND and NOR on every device, and the GPU rows above are no longer production's.
- **Typed loads.** Reading lanes through `memcpy` from a byte pointer hides their alignment: 1.4–3.5× slower on
  sm_70.
- **Lane width.** 64 bits per work-item is best or tied on GPUs: 32-bit lanes are 2–20% slower, aligned 128-bit
  lanes tie, 256-bit lanes through `memcpy` were up to 4.7× slower. CPUs: 4, 8 and 16 words within noise.
- **The cardinality kernel as AND, OR, NAND, NOR** ([n, n], [1, n], [0, n − 1], [0, 0]) is as fast as the
  dedicated kernels on GPUs and up to 16 inputs on CPUs, and slower above (the count costs about 3 operations
  per input against 1). XOR is parity, not a range.
- **Counted form per backend.** Above 16 inputs, blocks of 8 loads and a plain loop over the inputs are within
  1–6% of each other on every GPU (blocks of 8 ahead on the V100S and RTX 5060 Ti, the loop on the RTX 3070 and
  the Radeon); GPUs keep blocks of 8. On CPUs the loop over the positive inputs, then the negated ones, is best
  or tied, and 6–10% faster than blocks of 1, 2, 4 or 8 from 32 inputs (i7-10700, 16384 bitpacks per gate).
- **CPUs: one kernel per connective, block-major.** On CPUs the fixed slot counts tie the counted form
  (and:2 to and:16 within ±3%), and each slot count is a kernel launch whose barrier waits for the slowest
  thread. With a kernel per slot count a 1513-gate model took 145 kernels per pass instead of 42 and ran
  1.23× as long as develop on a loaded 8-core host (i7-9700, other services running), while on an idle
  i7-10700 it was level. With one counted kernel per connective it was 1.33× faster than with the slot counts
  on the loaded host and 1.125× slower on the idle one: the kernel's gates then differ in input count, and
  AdaptiveCpp's OpenMP backend hands each thread a contiguous run of work-groups (a static schedule,
  `__kmpc_for_static_init_8u`), so in gate-major order one thread can get the widest gates (about 1.9× the
  mean for a typical layer of this model). Block-major order gives every thread each gate as often; CPUs use
  it for kernels whose gates differ in input count (the counted forms), and gate-major elsewhere (block-major
  is 0–7% slower on layers of equal gates).
- **Wide gates that share inputs in small layers.** Production's work-groups span many gates at a few
  bitpacks, so an input that several of their gates read comes from the L1 once per window. With 64 inputs and
  32 gates (each input read by about 2 gates) the new kernels are 1.37× slower on the RTX 3070, 1.13× on the
  V100S, level on the RTX 5060 Ti and 1.8× faster on the Radeon; on the i7-10700 (16384 bitpacks per gate) 1.2×
  slower, and 1.5–1.75× slower for 64-input OR gates, 128 per layer, whose inputs 8 gates share. Block-major
  order recovers part of it on NVIDIA through the L2 (it is what the 1.13–1.37× already includes) and nothing
  on CPUs. Production's gate-first work-groups are 4.5–5.5× slower than the new kernels for layers of many
  small gates on CPUs and up to 100× on the Radeon, so they are no answer; tiling a few gates by a window of
  bitpacks might be (#281).

## Cardinality gates against two-level AND/OR networks

`cardinality/bench_network.cpp` (the production kernels, forms, lanes and orders): a k-of-n gate against its
sum of products (OR over the C(n, k) k-subsets of AND) and its product of sums (AND over the C(n, k - 1)
(n - k + 1)-subsets of OR), each network's bits compared word for word with the cardinality gate's; 16384
bitpacks per gate, inputs from a pool of 1024 buffers. A network's time is its two launches'. These runs used
the fixed slot counts that production then had for AND and OR gates on GPUs; the bench now takes the counted
form, as #282 does.

| device | cheaper network / cardinality, 1 < k < n | measured / naive (bytes) | OR, AND / cardinality, k = 1 or n, n ≤ 16 | n = 32, 64 |
|---|---|---:|---:|---:|
| Tesla V100S | 3.66 (2 of 3) – 46.5 (5 of 10) | 0.28–1.25 | 0.93–0.98 | 1.12–1.23 |
| RTX 3070 | 3.39 – 39.4 | 0.18–1.06 | 1.00–1.02 | 1.10–1.19 |
| RTX 5060 Ti | 3.65 – 146 | 0.22–1.12 | 0.99–1.01 | 1.00 |
| Radeon Pro 5500M | 2.72 – 62.5 (63 of 64) | 0.37–0.98 | 1.00–1.02 | 1.00–1.01 |
| i7-10700, 8 threads | 2.89 – 76.9 | 0.16–1.13 | 0.80–0.99 | 0.89–0.93 |

- No network is faster for 1 < k < n: expanding k-of-n gates into AND and OR costs 2.7–146×, and collapsing a
  two-level k-of-n network into one cardinality gate gains as much.
- Bytes moved (n + 1 against C(n, k)(k + 2) + 1 or C(n, k - 1)(n - k + 3) + 1) overstate large networks: a
  voter's first-level gates re-read its n inputs from the cache, and what remains are the intermediate words,
  written once and read once. They understate small ones on NVIDIA, whose cardinality gates read shared inputs
  from the L2 while the network's intermediates are read once. Counting distinct pool buffers U, cardinality
  moves U + G words and a network U + 2 G C + G (2 of 3 on the RTX 5060 Ti: 3.61 against 3.65 measured).
- At k = 1 or n the plain OR or AND ties or wins, except at 32 and 64 inputs on the V100S and RTX 3070. There the
  cardinality kernel's 52 registers leave 4 resident work-groups per SM against AND's 8 (V100S) or 6 (RTX 3070),
  so fewer columns of blocks are in flight and their shared inputs stay in the L2. Capping AND at 4 work-groups
  per SM (unused local memory) closes the gap to 0–5% and makes AND faster (32 inputs, 64 gates: 0.233 → 0.203
  ms on the V100S, 0.451 → 0.323 ms on the RTX 3070 at 2 per SM); gate-major order, or a pool that fits the L2
  (256 buffers) or leaves little reuse (4096), removes it.

## Open

- The budget sweep on the V100S (32 GB) and the RTX 3070, where the host is slower relative to the GPU.
- Whether pass sizes that are not a power of two, or not a multiple of 64 bitpacks, stay unbiased in the gate
  and tally kernels: not checked. The sampler's counters are per bitpack, so its stream does not depend on it.
- Why block-major order is 2.6–3.6× slower than gate-major on the Radeon Pro 5500M. It isn't the L2's size,
  since it persists with a pool 16× smaller. The scalar cache cycling through every gate's descriptor and
  pointers is a guess, not tested.
- Bounding the work-groups in flight for wide gates in small layers on NVIDIA (#281): capping residency made
  AND 1.15× faster on the V100S and 1.4× on the RTX 3070 in `bench_network --pad`, but no production form
  does it yet.
