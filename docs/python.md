# Python package (`fuzzybits`)

The Monte Carlo engine is installable as **`fuzzybits`** (import name and
distribution name). Wheels are self-contained:

- **Linux x86_64** (`manylinux_*`): AdaptiveCpp runtime (`libacpp-rt`,
  `libacpp-common`, CUDA/HIP/OpenMP backend plugins) and, for the `generic`
  variant, the SSCP JIT pieces (`llvm-to-backend` incl. `libllvm-to-amdgpu`,
  `libLLVM`, `libdevice.10.bc`, the ROCm `amdgcn` device bitcode,
  `opt`/`llc`). NVIDIA **driver** libraries (`libcuda.so*`) are not bundled; a
  CUDA ≥ 12.8 driver is required for NVIDIA runs. ROCm is not bundled either:
  AMD runs need the ROCm 7 user space (`hip-runtime-amd`: `libamdhip64.so.7`,
  its unversioned `libamdhip64.so` link, `libhiprtc.so.7`, `libamd_comgr.so.3`)
  on the loader path, plus read/write `/dev/kfd` and a `/dev/dri/renderD*`
  node (`render` group). No ROCm clang/hipcc: the JIT links through hipRTC.
- **macOS 26+ Apple Silicon** (`macosx_*_arm64`): AdaptiveCpp `ACPP_REF_METAL`
  runtime with Metal + OpenMP backends and SSCP JIT (`libllvm-to-metal`,
  `opt`/`llc`/`ld64.lld`, `libLLVM`, Homebrew `liblld`/`libomp`), `delocate`-repaired
  with `@loader_path` rpaths. There is no separate `libmc.dylib` (INTERFACE
  `mc` → `libscram`). Metal kernel JIT currently rejects `double` until
  [#17](https://github.com/fuzzybit-org/fuzzybits/issues/17)/[#30](https://github.com/fuzzybit-org/fuzzybits/issues/30);
  OpenMP CPU is the working backend. Built on GitHub-hosted `macos-26`.
  CPython 3.12–3.15, one wheel per interpreter (no stable ABI).
- **macOS 15+ Intel** (`macosx_15_0_x86_64`): AdaptiveCpp OpenMP + generic
  SSCP host JIT, **no Metal** (Metal is Apple Silicon only). Homebrew
  prefixes are `/usr/local/opt/{llvm@20,libomp,lld@20}`. Built on
  GitHub-hosted `macos-15-intel`; the 15.0 deployment target installs on
  Intel Macs running macOS 15+, including 26. Same CPython 3.12–3.15 matrix.

Wheels are built with `MALLOC_TYPE=malloc` — no `LD_PRELOAD`.

Publishing to PyPI is [#36](https://github.com/fuzzybit-org/fuzzybits/issues/36)
and is gated on a legal review. Until then, install a locally built wheel.

## Install a wheel

```bash
pip install dist/fuzzybits-*.whl
python -c "import fuzzybits; print(fuzzybits.__version__, fuzzybits.list_devices())"
fzb --help
```

No `PYTHONPATH` or `LD_PRELOAD`. The wheel installs an `fzb` console script
that matches the native CLI's compile / inspect / run flag surface so scripts
can use either binary. Docker images keep the native `/usr/local/bin/fzb`
(the pip script is overwritten after install).

### Migration from `mcscram_native` / `mcscram_server`

`import mcscram_native` is removed. Use `import fuzzybits`
(`fuzzybits.compile`, `fuzzybits.inspect`, `fuzzybits.run`,
`fuzzybits.Session`, or `fuzzybits.native()` for bindings that are not yet
public). The `mcscram_server` package and `mcscram-server` console script
are gone; use `fuzzybits_server` / `fuzzybits-server`.

Variants follow [#29](https://github.com/fuzzybit-org/fuzzybits/issues/29):

| wheel | native module | AdaptiveCpp flow | when to use |
|---|---|---|---|
| `generic` (default) | `fuzzybits._native_generic` | SSCP JIT | any NVIDIA GPU, AMD GPU (ROCm 7, e.g. gfx1012), CPU |
| `cuda-sm70` | `fuzzybits._native_cuda_sm70` | clang CUDA SMCP AOT + embedded PTX | Volta (V100) |
| `cuda-sm86` | `fuzzybits._native_cuda_sm86` | clang CUDA SMCP AOT + embedded PTX | Ampere consumer (RTX 30xx) |
| `cuda-sm120` | `fuzzybits._native_cuda_sm120` | clang CUDA SMCP AOT + embedded PTX | Blackwell consumer (RTX 50xx) |

macOS Metal is an experimental extra wheel ([#87](https://github.com/fuzzybit-org/fuzzybits/issues/87));
`backend="metal"` is accepted and should raise a Python exception until
[#30](https://github.com/fuzzybit-org/fuzzybits/issues/30) lands.

Import selection (override with `fuzzybits.init(variant=...)` or
`FUZZYBITS_VARIANT`): a **cuda-smNN** extension if it is installed *and* a
device of that compute capability is visible (`fuzzybits.cuda_sm_versions()`),
otherwise **generic**. Only the runtime images ship the `cuda-smNN`
extensions; PyPI carries the generic wheel.

This phase ships one variant per wheel under the same distribution name, so
pip will replace one with the other. The import machinery still auto-selects
if both `.so` files are present (future `fuzzybits-native-*` split).

## API

```python
import fuzzybits

fuzzybits.init(backend="auto")          # "cuda" | "hip" | "omp" | "metal" | "auto"; also FUZZYBITS_BACKEND
print(fuzzybits.__version__)
print(fuzzybits.list_devices())

result = fuzzybits.probability(
    "model.xml",
    trials=1e7,
    pdag_c=2,
    no_kn=True,
    policy="wald",
)
print(result.estimate, result.ci, result.std_err)

pcfb = fuzzybits.compile("model.xml", pdag_c=2, no_kn=True)   # -> bytes
summary = fuzzybits.inspect(pcfb)  # fzb inspect --json keys + diagnostics
print(summary["queries"], summary["diagnostics"])

rows = fuzzybits.run(pcfb, trials=1e6, backend="auto")        # list[QueryResult]
print(rows[0].name, rows[0].estimate, rows[0].hits, rows[0].n)

with fuzzybits.Session(pcfb, backend="auto") as session:
    print(session.sample_shape)
    graph = session.graph()              # reconstructed PDAG, same keys as Pdag.to_graph()
    plan = session.execution_plan()      # same keys as Pdag.to_execution_plan()
    for obs in session.passes(max_passes=8):
        tally = obs.nodes[obs.target_index]
        print(obs.pass_index, tally.num_one_bits, tally.total_bits)
        if tally.total_bits >= 1_000_000:
            break                        # early stop; Session.close() on __exit__
    obs = session.step(passes=16)        # 16 passes, one wait and one tally read
```

`compile` / `compile_file` lower OpenPSA/MEF, nombrilo.circuit
(safetensors, detected from the header) **or** CAFTA FTP projects to a pcfb
buffer (the same host path as `fzb compile`). FTP input takes `target=`,
`overrides=`, `duplicate_import=`, `complement_events=`, `negation=`,
`max_initiators_per_cutset=`, `literals=` and `config=`, and `report=True`
returns `(blob, report)`; see [`ftp-input.md`](ftp-input.md). MEF input expands
common-cause groups unless `ccf=False` (`fzb compile --no-ccf`); `target=` and
`watch=` take a name or a sequence of names (MEF and nombrilo input take one target). `nombrilo_to_pcfb` / `pcfb_to_nombrilo` are the
explicit converters; see [`nombrilo.md`](nombrilo.md). `inspect` always runs `pcfb::validate` and
returns its diagnostics (empty iff the circuit is core-1.1 conforming).
`run` is the one-shot `fzb run` equivalent: `list[QueryResult(name, estimate,
ci_low, ci_high, n, hits)]`. `Session` reconstructs the PDAG and streams
raw per-pass popcounts (`num_one_bits` / `total_bits`, pass index, trials
per pass) so a host-side convergence loop can replace
`run_compile` + `layer_manager.single_pass_all` without changing results.

The wheel now links `fuzzybits_compile`, pcfb, and the runtime into the
native extension (`libmc` already carries `run.cc`). Native `fzb` /
`scram-cli` binaries are not packed into the wheel; pip users get the
Python `fzb` console script instead.

Results are deterministic for a given model and trial count (Philox
counters). `seed=` raises `NotImplementedError` on `probability`, `compile`,
`run`, and `Session` ([#65](https://github.com/fuzzybit-org/fuzzybits/issues/65)).

### Progress callbacks

`run` and `compile` take `on_progress=`, called with a `ProgressSnapshot` — the
engine's typed, schema-v1 progress record, decoupled from any rendering.
Snapshots are coalesced at the source to at most ~10 per second; the first and
the last are always delivered. No snapshot is built when no callback is
attached, so results are bit-identical either way.

```python
import fuzzybits

def show(snapshot):
    if snapshot.phase != "sampling":
        return
    print(
        f"{snapshot.trials_done}/{snapshot.trials_total} trials"
        f"  {snapshot.trials_per_second:.3g}/s"
        f"  ETA {snapshot.eta_seconds:.0f}s"
    )
    for query in snapshot.queries:
        print(f"  {query.name}  {query.estimate:.4g}  ±{query.rel_half_width:.1%}")

rows = fuzzybits.run("model.pcfb", trials=1e9, on_progress=show)
fuzzybits.compile("model.xml", on_progress=show)   # phases parse/lower/passes/write
```

`ProgressSnapshot` fields: `schema`, `phase` (`"compile"`, `"sampling"`,
`"checking"`, `"done"`), `step` (the compile sub-stage, else `""`),
`trials_done` / `trials_total`, `passes_done` / `passes_total` (0 means
unknown), `elapsed_seconds`, `trials_per_second`, `frac_done`, `eta_seconds`,
and `queries`. Each `QueryState` carries `name`, `hits`, `trials`, `estimate`,
`ci_low`, `ci_high`, `rel_half_width`, `target_rel_half_width` (NaN without a
convergence target) and `converged`. Values are typed; nothing needs string
parsing. Raising from the callback stops the run cleanly and re-raises the same
exception.

The legacy `native().run_probability(..., on_progress)` string key-value
callback is unchanged. The server mirrors these records onto its unchanged
live-progress frame with `fuzzybits_server.progress_adapter`.

`init()` must run before AdaptiveCpp initialises. The first call to
`list_devices` / `probability` / `compile` / `run` / `Session` performs a
lazy `init("auto")`. The JIT kernel cache defaults to
`~/.cache/fuzzybits/acpp` (`ACPP_APPDB_DIR`); set that variable yourself
to override.

## CLI (`fzb` on the wheel)

```bash
fzb compile model.xml [more.xml ...] [-c N] [--no-kn] [--no-ccf] [--target NAME|@FILE] [--watch NAME|@FILE]... [-o model.pcfb]
fzb compile model.safetensors [--target NAME|@FILE] [--weights FILE] [-o model.pcfb]
fzb compile a.ftp [b.ftp ...]|DIR [--target NAME|@FILE]... [--watch NAME|@FILE]... [--negation boolean] [--override NAME=0|1]... [--config FILE] [--report FILE]
fzb export --to nombrilo model.pcfb -o model.safetensors [--strict]
fzb inspect model.pcfb [--validate] [--json] [--exact]
fzb run model.pcfb --trials N [--seed S] [--backend auto|cuda|hip|omp|metal] [--json]
fzb --help
fzb --version
```

`--json` keys match the native `fzb inspect --json` / `fzb run --json`
payloads. `--seed` is rejected unless it is `372` (the scram-cli default),
same as native ([#65](https://github.com/fuzzybit-org/fuzzybits/issues/65)).

### Progress log (`--progress-log FILE`, `progress_log=`)

`fzb run` and `fzb compile` accept `--progress-log FILE`; `fuzzybits.run` and
`fuzzybits.compile` accept `progress_log=` (a `str` or path-like). The format
comes from the extension: `.jsonl` and `.ndjson` are JSON Lines, `.csv` is CSV;
anything else is an error raised before any work starts. The log is written
whether or not a terminal display is active (`--json`, redirected output and a
disabled display do not affect it), and no file is ever created unless it is
asked for. The legacy `scram-cli` `<input>_<timestamp>_convergence.csv` is
unrelated and unchanged.

```bash
fzb run model.pcfb --trials 1e9 --progress-log run.jsonl
fzb compile model.xml -o model.pcfb --progress-log compile.csv
```

```python
rows = fuzzybits.run(pcfb, trials=1 << 24, progress_log="run.csv")
blob = fuzzybits.compile(["model.xml"], progress_log=Path("compile.jsonl"))
```

There is one record per coalesced snapshot (about 10 per second plus the first
and the last) and exactly one *finish* record, written last, whose per-query
values equal the run's result. Records are flushed individually, so a killed run
leaves a readable file whose final record is missing.

JSON Lines fields (one object per line; `null` for values that were not
computed, never `NaN`):

| field | meaning |
| --- | --- |
| `record` | 0-based record index in the file |
| `event` | `"snapshot"` or `"finish"` |
| `interrupted` | `true` on the finish record of a failed or cancelled run |
| `schema` | snapshot schema version, currently `1` |
| `phase` | `compile`, `sampling`, `checking`, or `done` |
| `step` | compile sub-stage (`parse`, `lower`, `passes`, `write`); empty otherwise |
| `trials_done`, `trials_total` | Bernoulli trials; `trials_total` is `0` when unknown |
| `passes_done`, `passes_total` | device passes; `passes_total` is `0` when unknown |
| `elapsed_seconds` | wall time since the run started |
| `trials_per_second` | rate over a sliding window |
| `queries` | array, in query order, of the per-query objects below |

Per query: `name`, `hits`, `trials`, `estimate`, `ci_low`, `ci_high`,
`rel_half_width` (`(ci_high - ci_low) / (2 estimate)`, `null` when the estimate
is 0), `target_rel_half_width` (`null` without a convergence target), and
`converged`.

CSV is long format: the header is

```
record,event,interrupted,schema,phase,step,trials_done,trials_total,passes_done,passes_total,
elapsed_seconds,trials_per_second,query_index,query_name,hits,trials,estimate,ci_low,ci_high,
rel_half_width,target_rel_half_width,converged
```

(written on one line) with one row per (record, query): the snapshot scalars
repeat and `query_index` / `query_name` identify the query. Names are quoted per
RFC 4180 when they contain a comma, a quote, a newline, or leading/trailing
blanks, and embedded quotes are doubled. A missing double is an empty cell. A
record without queries (compile stages) keeps one row with the query columns
empty, so no record is lost. Doubles use shortest round-trip formatting in both
formats, so a reader recovers the exact value the snapshot carried.

## Build wheels from source

Requires the pinned toolchain in [`toolchain.env`](../toolchain.env)
(AdaptiveCpp v25.10.0, clang 20, CUDA 12.8.1). CMake is driven by
**scikit-build-core** using the same cache variables as
[`CMakePresets.json`](../CMakePresets.json) (`generic` by default).

```bash
pip install . --config-settings=cmake.define.ACPP_TARGETS=generic
# or the Volta AOT variant:
pip install . --config-settings=cmake.define.ACPP_TARGETS=cuda:sm_70 \
              --config-settings=cmake.define.ACPP_USE_ACCELERATED_CPU=off \
              --config-settings=cmake.define.ACPP_STDPAR=off \
              --config-settings=cmake.define.ACPP_EXTRA_COMPILE_OPTIONS=--cuda-include-ptx=sm_70
```

Containerised build (remote daemon, writes `dist/` on the client):

```bash
# 1. AdaptiveCpp+clang toolchain (full profile — SSCP)
docker --context xerxes build -f Dockerfile.cuda12.8 \
  --build-arg ACPP_COMPILER_FEATURE_PROFILE=full \
  --target adaptivecpp-clang \
  -t fuzzybits-acpp:full .

# 2. generic + cuda-sm70 wheels
docker --context xerxes build -f docker/wheel.Dockerfile \
  --build-arg TOOLCHAIN_IMAGE=fuzzybits-acpp:full \
  --build-arg FUZZYBITS_VARIANT=generic \
  --output type=local,dest=dist/generic .

docker --context xerxes build -f docker/wheel.Dockerfile \
  --build-arg TOOLCHAIN_IMAGE=fuzzybits-acpp:full \
  --build-arg FUZZYBITS_VARIANT=cuda-sm70 \
  --output type=local,dest=dist/cuda-sm70 .
```

`auditwheel repair` on Ubuntu 24.04 tags **`manylinux_2_39`**. A
`manylinux_2_28` rebuild is a follow-up. Driver libs are passed to
`--exclude libcuda.so.1`. Approximate wheel sizes (per CPython, x86_64):
`generic` and `cuda-sm70` are both ~66 MB, dominated by `libLLVM.so.20.1`
(~153 MB uncompressed) for SSCP JIT. Release builds CPython 3.12–3.15
inside the toolchain image with uv-managed interpreters (`pybind11>=3.1`).

macOS 26 arm64 (GitHub-hosted `macos-26`). Host compiler is AppleClang;
LLVM 20 is AdaptiveCpp's `CLANG_EXECUTABLE_PATH`. Override pyproject's
Linux `clang-20` compilers:

```bash
export MACOSX_DEPLOYMENT_TARGET=26.0
pip wheel . -w dist --no-deps \
  --config-settings=cmake.define.CMAKE_C_COMPILER=/usr/bin/clang \
  --config-settings=cmake.define.CMAKE_CXX_COMPILER=/usr/bin/clang++ \
  --config-settings=cmake.define.ACPP_TARGETS=generic \
  --config-settings=cmake.define.AdaptiveCpp_DIR=$ACPP_PREFIX/lib/cmake/AdaptiveCpp
delocate-wheel --lib-sdir lib --require-archs arm64 -w dist wheelhouse/fuzzybits-*.whl
```

Hosted path: `scripts/ci/macos-wheel-versions.sh` (uv-managed 3.12–3.15)
+ `scripts/ci/macos-wheel-build.sh` + `scripts/ci/macos-smoke.py`.

macOS 15 Intel (GitHub-hosted `macos-15-intel`; `MACOSX_DEPLOYMENT_TARGET=15.0`):

```bash
export MACOSX_DEPLOYMENT_TARGET=15.0
# AdaptiveCpp: WITH_METAL_BACKEND=OFF, Homebrew /usr/local/opt/llvm@20
pip wheel . -w dist --no-deps \
  --config-settings=cmake.define.CMAKE_C_COMPILER=/usr/bin/clang \
  --config-settings=cmake.define.CMAKE_CXX_COMPILER=/usr/bin/clang++ \
  --config-settings=cmake.define.ACPP_TARGETS=generic \
  --config-settings=cmake.define.CMAKE_OSX_ARCHITECTURES=x86_64 \
  --config-settings=cmake.define.AdaptiveCpp_DIR=$ACPP_PREFIX/lib/cmake/AdaptiveCpp
delocate-wheel --lib-sdir lib --require-archs x86_64 -w dist wheelhouse/fuzzybits-*.whl
```

`cmake/bundle_acpp.py` requires `librt-backend-metal.dylib` only on arm64.

## Tests

```bash
pip install pytest
pytest tests/python -q -m "not gpu"     # OpenMP, small trial counts
pytest tests/python/test_pcfb_gpu.py -q # CUDA bit-identity (separate interpreter)
```
