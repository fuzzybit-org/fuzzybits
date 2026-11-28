# kernels

Issue: #164 (origin of these tools; the Lab issue labelled `lab:kernels` is the maintainer's to file) ·
Owner: maintainer · Started: 2026-11-24 · Time box: set by the next experiment that uses this directory

Tools for measuring the Monte Carlo engine's kernels, outside the pipeline (a standalone sampler
microbenchmark and readers for the JIT output) and inside it (A/B of two runtime images, occupancy,
passes per host wait, pass size). They come from the bit-sliced sampler work (#164, #17);
[FINDINGS.md](FINDINGS.md) has what they showed.

## Hypothesis

Set by the next experiment. The pass-size probe and `patches/shaper-budget.patch` are ready for the sample
shaper's device budget (#173).

The cardinality and gate kernel experiment (#274, #275) has ended. Its hypothesis was that bit-sliced kernels
with a compile-time input count beat the replaced at-least and gate kernels on every device. They did, and
their forms went into #282; [FINDINGS.md](FINDINGS.md) has the measurements.

## Measurement

| tool | what it measures | where it runs |
|---|---|---|
| `microbench/bench.cpp`, `samplers.h` | launch time of each sampler variant and of the production `strided` / `lockstep` forms (`pstrided`, `plockstep`), events × bitpacks per launch, every variant checked against a host reference | toolchain container (`microbench/build.sh`) |
| `microbench/sweep.sh` | the persistent variant's run length × work-group size | same container |
| `cardinality/bench.cpp`, `card.h` | k-of-n gates: replicas of the at-least kernel and the at-most/exactly lowering that #274 replaced, against the bit-sliced kernels of `card.h`; G gates × B bitpacks, every variant checked against a host count | toolchain container (`cardinality/build.sh`) |
| `cardinality/bench_gate.cpp` | AND, OR, XOR, NAND, NOR, NOT, NULL: a replica of the kernel #275 replaced, against fixed and runtime input counts, lane widths, load blocks, both work-group orders, and the cardinality kernels posing as AND/OR | same container |
| `cardinality/bench_network.cpp` | the production cardinality kernel against the two-level AND/OR networks that compute the same bits (sum of products, product of sums), word for word, with residency (`--pad`), order (`--order`) and pool (`--pool`) experiments | a configured checkout with #282 (`cardinality/build.sh --network`), e.g. a `scripts/dev/ws.sh` workspace container |
| `cardinality/summarize.py`, `net_summary.py`, `netexp_summary.py` | tables from the benches' output | anywhere |
| `inspect/sass.sh` | per JIT-compiled PTX module: registers, spills (`ptxas -v`), SASS opcode histogram | same container, after a run |
| `inspect/hostasm.sh` | per host-JIT kernel: instructions, ymm/xmm use, `vpmuludq` / `vpmulld` / scalar multiplies | same container |
| `inspect/amd_notes.sh`, `amd_branches.sh` | VGPR/SGPR, spills, LDS and loop branches of AMD code objects | ROCm toolchain container |
| `inspect/reflect.cpp` | the backend AdaptiveCpp's JIT reflection reports in a kernel | any image with `acpp` |
| `inspect/devq.cpp` | time of a queue construction and of the device queries the kernels make | any image with `acpp` |
| `ab/build_image.sh` | builds a runtime image from a commit or tree object exactly as `bench.yml`'s head step does | docker host over ssh |
| `ab/ab_local.py` | `python -m benchmarks ab` with a locally built head image, optionally with `libcudart` preloaded on both sides (#268) | checkout |
| `ab/pcfb_ab.py` | A/B of two images on any `.pcfb` kept on the docker host (private models stay there); targets `gpu:N`, `hip`, `cpu` | checkout, ssh |
| `ab/pcfb_summary.py` | per target and model of `pcfb_ab.py` JSON: head/base with a 95% interval, and whether base and head hits agree in every round | checkout |
| `ab/inproc_times.py` | per-call time of repeated `fuzzybits.run` in one process | runtime image |
| `pipeline/occupancy.py`, `nsys_summary.py` | one `fuzzybits.run` under Nsight Systems: kernels, GPU busy time, launch gaps, CUDA API time | runtime image + nsys |
| `pipeline/passes_per_wait.py` | throughput of `fuzzybits.Session` against passes queued per host wait (`FUZZYBITS_PASSES_PER_STEP`) | runtime image |
| `pipeline/budget_sweep.py` | throughput against the shaper's device budget, with the pass size it picks and peak memory | image built with `patches/shaper-budget.patch` |

Data: the Aralia models of benchmark dataset v1 (`python -m benchmarks`), and private models through
`ab/pcfb_ab.py`, reported as aggregates only.

Typical commands, from the repository root; `HOST` is an ssh host whose docker has the GPU, `IMAGE` a runtime
image, `LOCK` that GPU's `/tmp/gpu-lock-*` file:

```bash
lab/kernels/microbench/build.sh kernels-lab
docker exec kernels-lab bash -lc 'cd /lab && ACPP_VISIBILITY_MASK=cuda ./bench --events 122 --bitpacks 131072 --p 0.01 prod bs16 pers16'
lab/kernels/cardinality/build.sh kernels-lab
docker exec kernels-lab bash -lc 'cd /lab/cardinality && ACPP_VISIBILITY_MASK=cuda ./bench_gate --reps 15 --variants prod,fixed.bm,blk8.bm --shape and:3 --shape and:64:32' > gate.tsv
python3 lab/kernels/cardinality/summarize.py gate.tsv
lab/kernels/ab/build_image.sh "$PWD" HOST fuzzybits-runtime:lab-head "$(git rev-parse HEAD^{tree})"
PYTHONPATH=. FZB_AB_PRELOAD_CUDART=1 python3 lab/kernels/ab/ab_local.py ab --target v100s --metric sample --mode perf \
  --base-image ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic --head-image fuzzybits-runtime:lab-head \
  --base "$(git rev-parse origin/develop)" --head "$(git rev-parse HEAD^{tree})" --rounds 6 --skip-validate \
  --ids das9201 das9701 nus9601 edf9203 chinese
python3 lab/kernels/ab/pcfb_ab.py "$PWD" HOST /path/on/host gpu:0 out.json --base BASE_IMAGE --head HEAD_IMAGE \
  --models a b --trials 67108864 --rounds 6 --pin-cudart
```

The pipeline probes run in a runtime container with the probe directory mounted at `/work`:

```bash
docker run --rm --gpus device=0 -v "$PWD/lab/kernels/pipeline:/work:ro" -v /models:/m:ro \
  -e ACPP_VISIBILITY_MASK=cuda -e ACPP_PERSISTENT_RUNTIME=0 --entrypoint python3 IMAGE \
  /work/passes_per_wait.py /m/model.pcfb 67108864 auto
```

Rules for timing that the findings explain: one GPU A/B per host at a time, base and head alternated,
`libcudart` preloaded on both sides for CUDA, and no `ACPP_APPDB_DIR` under `/tmp/acpp-appdb` in runtime
images.

## Kill criterion

Set by the next experiment. The tools themselves are archived (tag `lab-archive/kernels`) when the bench
harness and CI cover what they measure.

## Graduation needs

None for the tools: they never ship. A finding that needs a core change becomes a product issue (for
example #268, #269, #270).
