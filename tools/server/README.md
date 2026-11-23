<!--
SPDX-FileCopyrightText: 2026 fuzzybit-org
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# tools/server

AGPL-3.0-or-later. Imported from `fuzzybit-org/fuzzybits-bench` (`src/server`)
with history. Compiles OpenPSA/MEF **fault trees** to `.pcfb` (once per
source SHA-256s and compile options) and samples the **top event** through
`fuzzybits.Session`. Stopping is a #224 precision goal (Wilson interval), a
trial or time limit, or a fixed trial count; the legacy Wald / GBAS options are
translated for one release ([stopping specification](#stopping-specification)).
Event-tree sequences and end states are not compiled or sampled
([#92](https://github.com/fuzzybit-org/fuzzybits/issues/92)).
Set `FUZZYBITS_SERVER_ENGINE=legacy` for the in-memory `run_compile` path.
The `MCSCRAM_CONVERGENCE_ENGINE=native` `run_probability` fallback is unchanged.

## Run

```bash
cd tools/server
uv sync --extra dev --no-install-package fuzzybits
uv run --no-sync fuzzybits-server --data-dir ../../input/Aralia
uv run --no-sync fuzzybits-server --no-ui          # headless API; GET / is a JSON pointer to /api/v1
```

CLI defaults to `127.0.0.1:8000`. Flags: `--host`, `--port`, `--data-dir`,
`--cache-dir` (platformdirs), `--backend auto|cuda|hip|omp|metal`,
`--engine pcfb|legacy`, `--no-ui`, `--open`.

Compile / sampling endpoints need the native module. The published
`fuzzybits-server` wheel depends on `fuzzybits==0.18.6`. Until that engine
wheel is on PyPI ([#36](https://github.com/fuzzybit-org/fuzzybits/issues/36)),
`uv.lock` resolves it from this checkout's static metadata and CI/Docker skip
installing it (`uv sync --no-install-package fuzzybits`) so the runtime image
or a locally built wheel can supply the engine:

```bash
uv pip install ../../dist/fuzzybits-*.whl
python -c "import fuzzybits; print(fuzzybits.__file__, fuzzybits.list_devices())"
```

Without the wheel the process still starts: `/api/v1/healthz` works, model
listing is empty, and native compile returns an install error instead of
aborting the interpreter.

### PyPI wheel

```bash
pip install fuzzybits-server
fuzzybits-server --data-dir ./models
```

The wheel is `py3-none-any` and bundles the CSR playground. Build contract
(from a clean checkout, after the UI is built):

```bash
export COPYFILE_DISABLE=1   # macOS: keep AppleDouble files out of archives
(cd tools/server/webui && npm ci && npm run build) &&   (cd tools/server && uv build --wheel --out-dir ../../dist-server)
# -> dist-server/fuzzybits_server-0.18.6-py3-none-any.whl
```

An sdist is optional; if produced, it includes `webui/dist` so building a
wheel from the sdist does not need Node.

## Test

The imported pytest suite is pure Python (BernoulliSampler, numpy, scipy)
plus config/cache helpers. It does not import `fuzzybits` unless a
test is marked `@pytest.mark.native`.

```bash
cd tools/server
uv sync --frozen --extra dev --no-install-package fuzzybits
uv run --no-sync pytest -m "not native"
```

Mark future engine tests `@pytest.mark.native`. `tests/conftest.py` skips
them when the wheel is absent.

The characterization tests for the stopping work (#225) are:

| File | Pins |
| --- | --- |
| `tests/test_live_progress_shape.py` | the `LiveProgressUpdate` wire shape, key by key, for a simulated and an interrupted run |
| `tests/test_request_schema.py` | the run-settings request schema, the legacy names and their defaults, and every 422 |
| `tests/test_stopping.py` | the #224 specification: defaults, every validation message, every legacy mapping |
| `tests/test_wilson.py` | the Wilson interval (against scipy's closed form), the zero-hit bound (against Clopper–Pearson), the goal rule and the projection |
| `tests/test_stop_reasons.py` | the stop reasons and the controller frames, legacy and new |
| `tests/test_interrupted_run.py` | an interrupted run reports `interrupted` without changing the frames it sent |
| `tests/test_fixed_n_equivalence.py` | `@pytest.mark.native`: a fixed-N server run against `fzb run -N`, and today's pass-geometry difference |

`test_fixed_n_equivalence.py` needs the extension on `sys.path` and finds `fzb`
via `FZB_BINARY` or `build/*/targets/fzb/fzb`; it falls back to
`fuzzybits.run` (the same one-shot engine path) when neither exists, and says so
in the failure message.


## UI

CSR build in `webui/`, served by FastAPI at `/` (SPA fallback, hashed
`/assets/*` with long cache). Same-origin `/api/v1` and `/api/v1/ws`. No Node
at runtime. The UI catalogs XML, draws the first `define-fault-tree`, compiled
PDAG, and kernel DAG, and streams the top-event estimate.

```bash
cd tools/server/webui
npm ci
npm run typecheck
npm test                 # node --test, the stopping-spec semantics
npm run build            # CSR assets → dist/
npm run dev              # Vite + proxy to FUZZYBITS_SERVER_URL (default :8000)
```

| Variable | Role | Default |
| --- | --- | --- |
| `FUZZYBITS_SERVER_UI` | `1` serve UI, `0`/`--no-ui` disable | `1` |
| `FUZZYBITS_SERVER_UI_DIR` | Directory with `index.html` + `assets/` | packaged `fuzzybits_server/webui`, else `tools/server/webui/dist` |

With the UI disabled, `GET /` returns `{"service":"fuzzybits-server","api":"/api/v1","ui":false}`.
When `FUZZYBITS_SERVER_UI_DIR` is unset the server uses the copy bundled in
the wheel (`importlib.resources`), then `tools/server/webui/dist` for a
source checkout or the Docker image.

## Data (Aralia XML + optional bench CSVs)

Public OpenPSA/Aralia models live in this repo at `input/Aralia/*.xml`.
Historical `compiler.csv` / `convergence.csv` stay in fuzzybits-bench
(CC BY-SA 4.0) and are optional. Point the server at a data root:

```bash
export FUZZYBITS_SERVER_DATA_DIR=/path/to/data
# docker: mount this repo's models (CSVs optional)
#   -v $PWD/input/Aralia:/data/Aralia
#   -e FUZZYBITS_SERVER_DATA_DIR=/data
```

| Path under `$FUZZYBITS_SERVER_DATA_DIR` | Used for |
| --- | --- |
| `compiler.csv`, `convergence.csv` | Optional run tables (`optional_table_path`) |
| `*.xml`, `*.pcfb`, `*.safetensors` under the root, `Aralia/`, `input/`, `input/Aralia/` | Model catalog (`GET /api/v1/model/xml-inputs`) |
| `SmallTree.xml` or `input/SmallTree/SmallTree.xml` | Default `MCSCRAM_INPUT_FILES` when that env is unset |

The catalog is recursive and lists three formats, each entry tagged with
`format`:

- OpenPSA MEF `*.xml` (`openpsa-xml`), compiled to `.pcfb` with the request's
  compile options. Model id: relative path without `.xml`, e.g. `chinese`.
- Compiled `*.pcfb` (`pcfb`), run as is; compile options don't apply. Model id
  keeps the suffix, e.g. `chinese.pcfb`.
- nombrilo circuits, `*.safetensors` files whose header says
  `format=nombrilo.circuit` (`nombrilo`), translated once into the pcfb cache.
  Variant files such as `baobab1.f3.safetensors` take their probabilities from
  the sibling `baobab1.safetensors`. Model id keeps the suffix.

The fault-tree and source-DAG views need MEF source, so they are empty for
`.pcfb` and nombrilo models; the compiled-DAG, kernel and sampling views work
for all three.

`MCSCRAM_ARALIA_INPUT_DIR` still overrides the search. If neither env is set
and `/app/input` is missing, the catalog is empty and the server stays up.

## Env

| Variable | Role |
| --- | --- |
| `FUZZYBITS_SERVER_DATA_DIR` | Bench data mount (CSVs + XML) |
| `FUZZYBITS_SERVER_ENGINE` | `pcfb` (default) or `legacy` in-memory PDAG |
| `FUZZYBITS_SERVER_CACHE_DIR` | Directory for compiled / uploaded `.pcfb` files |
| `MCSCRAM_ARALIA_INPUT_DIR` | Extra XML search root |
| `MCSCRAM_INPUT_FILES` | Explicit model path(s), comma/colon separated |
| `MCSCRAM_WATCH` | Boolean watch-mode **or** comma-separated MEF names / PDAG indices to tally |
| `MCSCRAM_WATCH_NODES` | Extra comma-separated watch names (same identification as `MCSCRAM_WATCH`) |
| `MCSCRAM_CCF` | Common-cause group expansion. On by default (as in `fzb compile`); set `false` to compile without it |
| `MCSCRAM_AUTOSTART` | Start sampling on boot (default false) |
| `FUZZYBITS_PASSES_PER_STEP` | Device passes run before the host reads tallies and checks convergence (default 1). Higher values keep the GPU busier on fast passes; convergence is checked less often. Applies to both the Python and the native (`MCSCRAM_CONVERGENCE_ENGINE=native`) loops. |
| `HOST` / `PORT` / `LOG_LEVEL` | uvicorn |

Sampling knobs (`MCSCRAM_NUM_TRIALS`, `MCSCRAM_POLICY`, …) are accepted for one
release; see [Stopping specification](#stopping-specification) for what they map
onto and which of them are rejected.

## Stopping specification

`POST /api/v1/start` and `POST /api/v1/reset` accept a stopping specification
that mirrors the controller library's (#224). It is the supported vocabulary;
the legacy option names are translated onto it, with a deprecation warning, for
one release.

```json
{
  "model_id": "chinese",
  "requested_trials_per_pass": 1048576,
  "stopping": {
    "rel_error": 0.01,
    "confidence": 0.95,
    "max_trials": 1000000000,
    "max_time": "10m"
  }
}
```

| Field | #224 flag | Meaning | Default |
| --- | --- | --- | --- |
| `rel_error` | `--rel-error` | goal: reported half-width ≤ `rel_error · p̂` | `0.01` in goal mode |
| `abs_error` | `--abs-error` | goal: reported half-width ≤ `abs_error`; a floor for tiny / zero probabilities | none |
| `confidence` | `--confidence` | interval level for the goal **and** the report | `0.95` |
| `max_trials` | `--max-trials` | limit: stop after this many committed trials | none |
| `max_time` | `--max-time` | limit: `90s`, `10m`, `2h`, a number of seconds, or `"none"` | `10m` in goal mode, none for a fixed design |
| `trials` | `-N` | design: exactly this many trials, no early stop | — |
| `goal_queries` | `--goal-query` | queries that must meet the goal | every primary query |
| `require_goal` | `--require-goal` | a limit before the goal is a failure | `false` |

Rules, all from the #224 definitions comment:

- **Interval:** Wilson at `confidence`; `z = Φ⁻¹(1 − α/2)`, centre
  `(p̂ + z²/2N)/(1 + z²/N)`, half `z/(1 + z²/N)·√(p̂(1−p̂)/N + z²/4N²)`.
- **Relative half-width:** `(U − L)/(2p̂)`; undefined at `p̂ = 0`.
- **Goal:** `(U − L)/2 ≤ max(rel_error·p̂, abs_error)`. A relative goal cannot be
  met at zero hits; `abs_error` alone can.
- **Zero hits:** the reported upper bound is `1 − (α/2)^{1/N}` (the
  Clopper–Pearson upper limit), ≈ `3.689/N` at 95%, and `converged` stays false
  against a relative goal.
- **Stop reasons:** `goal`, `max-trials`, `max-time`, `trials` (a fixed design),
  `interrupted`. The goal is evaluated at the end of each committed pass and
  before the limits, so a run that reaches a limit exactly as it converges
  reports `goal`.
- **Fixed design vs goal:** `trials` is exclusive with `rel_error`/`abs_error`,
  `goal_queries` and `require_goal`; sending both is a 422.
- **Pass quantisation:** a run stops at a pass boundary, so a fixed design of `N`
  trials commits `ceil(N / trials_per_pass) · trials_per_pass` trials when the
  engine's geometry divides unevenly. `requested_trials_per_pass` is a sampler
  knob, not part of the specification.

### Phase 1 execution

Until the controller library is wired in (phase 2), the server executes the
specification on the existing host-side loop. Anything that loop cannot honour
faithfully is rejected with **422** and a message that names the controller
library (#224) — never approximated:

| Request | Status |
| --- | --- |
| `trials`, `max_trials`, `max_time` | executed |
| `rel_error` / `abs_error` goal on the run's primary query | executed, on the Wilson interval with the rule above |
| `goal_queries` | 422: `requires the controller library, #224` |
| `require_goal` | 422: `requires the controller library, #224` |
| legacy `burn_in > 0` | 422: burn-in is counted in trials, so it has no faithful equivalent |

The Wilson implementation lives in `fuzzybits_server/convergence/wilson.py` and
the specification, its validation and the translation in
`fuzzybits_server/convergence/stopping.py`. Both are marked for deletion in
phase 2, when the interval and the rule come from the controller library.

### Legacy option mapping

Every legacy option has a disposition. A legacy request is *translated into a
specification* and executed by the same code path, so there is one implementation
of the stopping rule; the response carries the resolved `stopping` block plus
`warnings` listing the deprecations.

| Legacy option | Disposition | Transformation |
| --- | --- | --- |
| `num_trials: N > 0`, `early_stop: false` (also the default for `early_stop`) | **mapped**, deprecated | fixed design: `trials = N`. Stop reason becomes `trials` |
| `num_trials: N > 0`, `early_stop: true` | **mapped**, deprecated | goal with a limit: `max_trials = N`. The goal may stop it earlier |
| `num_trials: 0`, `early_stop: true` | **mapped**, deprecated | goal with no limits: `rel_error = delta` (or `0.01`), `max_time` omitted. Keeps today's unbounded behaviour instead of taking the ten-minute default |
| `num_trials: 0`, `early_stop: false` | **mapped**, deprecated | `max_time: "none"` and no goal: a run with no stopping criterion. Send this explicitly to mean it |
| `delta` | **deprecated → `rel_error`** | `rel_error = delta`, unchanged value. **Semantics change:** the goal is now evaluated on the Wilson interval, not the Wald one, so the same `delta` can stop at a different trial count (usually later at small `p̂`, and it no longer "converges" at zero hits) |
| `confidence` | **deprecated → `confidence`** | unchanged |
| `policy: "wald"` | **deprecated, replaced** | the Wald interval is replaced by Wilson; the request still runs, and the frame carries `wald_*` for comparison |
| `policy: "gbas"` | **deprecated, replaced** | Huber's exact relative-error control becomes a controller policy (#224). Until then the request runs the Wilson goal at the same `delta`/`confidence`, and the frame carries `wilson_*` — **no `gbas_*` stream at all**, so a client that read `gbas_mean` must read `mean`/`ci_low`/`ci_high` instead |
| `policies: [...]` | **deprecated** | the last entry is read as `policy` (the old stopping policy) |
| `burn_in > 0` | **removed** | 422. Burn-in is counted in trials, so its meaning depends on `p̂`; the #224 goal implies a hit count (`(z/rel_error)²` hits whatever `p̂` is), which makes a trial-count burn-in meaningless. Express the intent as `max_trials`; a burn-in in hits needs the controller library |
| `burn_in: 0` | **mapped** | no-op |
| `max_passes` | **deprecated, not a stopping option** | accepted and reported in `warnings`; it bounds device passes, not trials, so it has no place in the specification |
| `policies`/`stopping_policy` names other than `wald`/`gbas` | **removed** | 422 from the translation |

`sending both vocabularies` in one request is a **400**: pick `stopping` or the
legacy names.

### Resolved response

A control response that carries a stopping intent adds two fields:

```json
{
  "accepted": true,
  "status": {"status": "running"},
  "stopping": {
    "mode": "goal", "rel_error": 0.01, "abs_error": null, "confidence": 0.95,
    "max_trials": 1000000000, "max_time": "10m", "max_time_s": 600.0,
    "trials": null, "goal_queries": [], "require_goal": false,
    "interval": "wilson", "legacy": false, "deprecations": []
  },
  "stopping_summary": "goal +/-1% at 95%, max 1000000000 trials, max 10m"
}
```

A legacy request sets `"legacy": true` and fills `deprecations`. The completed
result (`last_result` in `/api/v1/status`) repeats `stopping` and
`stopping_summary` and adds `interval`, so a client never has to infer the rule
from the settings it sent.

## Live progress

The `progress` WebSocket event keeps the frame the web client already reads
(`ConvergenceStatsPayload`): `phase`, `iterations`, `trials`, `mean`, `p05`,
`p95`, `epsilon`, `target_epsilon`, `converged`, … Every frame is strict JSON
(no NaN or Infinity: a value that does not exist is `null`).

A run started with a stopping specification adds only **optional** fields; the
existing ones keep their names and types.

| Field | Type | Meaning |
| --- | --- | --- |
| `stop_reason` | `string` | on the last frame only: `goal`, `max-trials`, `max-time`, `trials` (legacy runs keep `converged`, `max_trials`, `max_passes`) |
| `interval` | `"wilson"` \| `"wald"` | which interval `p01`/`p05`/`p95`/`p99` and `epsilon` describe |
| `rel_half_width` | `number \| null` | `(U − L)/(2p̂)`; `null` at zero hits |
| `target_rel_half_width` | `number \| null` | the goal's `rel_error` |
| `abs_half_width` | `number \| null` | `(U − L)/2` |
| `target_trials` | `number` | the trial limit or fixed design the run is heading for |
| `eta_seconds` | `number \| null` | seconds to the projected trial count at the observed rate |
| `elapsed_s` | `number` | seconds since the run started |
| `stop_reason_style` | `"controller"` | marks the run's reason vocabulary |

**Compatibility policy.** New information is always a new optional field; an
existing field keeps its name, type and meaning, and a frame never gains a
required key. A change that violates either rule is a new frame version: the
`progress` event gains a `frame_version` field, the old shape is still emitted
for one release, and this table records the deprecation. The frame has never
carried a version field, and `tests/test_live_progress_shape.py` pins that, so
adding one is an explicit decision rather than a silent break.

## GBAS, Wald and information gain

The #224 controller replaces the three host-side stopping and diagnostic
mechanisms the server grew. This is their disposition, and what the controller
library needs to inherit them.

### Wald: replaced by Wilson

Today `epsilon = z·√(p̂(1−p̂)/N)` and the goal is `epsilon ≤ delta·p̂`. Wald is
unsafe at few hits for three reasons, each of which the server hit in practice:

1. **Zero width at zero hits.** `p̂ = 0` gives `epsilon = 0`, so an absolute or
   relative goal with a floor is reported "met" immediately on a model whose
   event never fired. The Wilson interval has positive width there, and the
   zero-hit bound `1 − (α/2)^{1/N}` is reported instead.
2. **Poor coverage at small `p̂ N`.** The normal approximation needs roughly
   `N p̂ ≥ 5`; below that the interval under-covers, and a goal evaluated on it
   stops too early. Wilson's coverage is close to nominal down to a handful of
   hits, and Clopper–Pearson (available in #224) is exact.
3. **The width is not monotone in the run.** `epsilon` depends on `p̂`, which
   moves, so a run can appear to converge and then un-converge. Wilson's
   half-width is monotone decreasing in `N` at fixed `p̂`, which is what makes
   "stop at the first pass that meets the goal" well defined (and is what the
   goal projection relies on).

`wald_*` stays in the frame for one release so a client can plot the old and new
intervals side by side, and `policy: "wald"` is accepted with a warning.

### GBAS: proposed #224 follow-up policy

GBAS is Huber's exact `(ε, δ)` relative-error estimator. It fixes `k` up front —
the smallest integer with `Pr(|p̂_GBAS/p − 1| > ε) ≤ δ` — and stops at the `k`-th
hit, estimating `p̂ = (k − 1)/R` where `R` is the sum of `Exp(1)` clocks over the
trials up to that hit. Its relative error law `Γ(k, k−1)` is independent of `p`,
which is why its guarantee is exact rather than asymptotic.

What it needs from the engine:

- **The exact trial index `T_k` of the `k`-th hit.** `R ~ Γ(T_k, 1)`, so `R`
  depends on `T_k`; a `k`-th hit located to a range of widths `w` gives a
  relative error in `R` of order `w/T_k`.
- `k` from `(ε, δ)` by a numerical search on the `Γ` CDF (already implemented
  and tested in `convergence/gbas.py`).
- Hits, not just trials, per pass — the engine contract already provides them.

**Granularity.** With #223, pass sizes are multiples of 64 trials (one 64-bit
bitpack). A pass-level controller observes the cumulative hit count only at pass
boundaries, so the crossing pass fixes `T_k` only to the interval
`(N_{c−1}, N_c]` of width `w ≤ 64` (exactly 64 when the pass size is a multiple
of 64 and the hit count rises by at least one in that pass).

**Effect on exactness.** With `T̂_k` anywhere in that window, `R̂ ~ Γ(T̂_k, 1)`
instead of `Γ(T_k, 1)`. Writing `u = (T̂_k − T_k)/T_k ∈ [−64/T_k, 0]`:

- the estimator's relative bias is `E[R̂]/E[R] − 1 = u`, so `|bias| ≤ 64/T_k`;
- its relative spread is inflated by at most `1/√(1 − 64/T_k) ≈ 1 + 32/T_k`;
- the `(ε, δ)` guarantee still holds if it is re-solved at
  `ε' = ε/(1 + 64/T_k)`, because the interval that matters is
  `p̂(1 ± ε')` against the true `p̂(1 ± ε)`.

`T_k ≈ k/p̂`, so the damage is worst at large `p̂`. Measured (this branch,
`convergence/gbas.py`, `relative_error_tail`):

| `p̂` | `ε` | `c` | `k` | `T_k` | `64/T_k` | `δ` achieved at `ε' = ε/(1+64/T_k)` | inflation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.05 | 0.95 | 1 537 | 1 537 | 4.2e−2 | 0.0600 vs 0.0500 | 1.20× |
| 1.0 | 0.01 | 0.95 | 38 415 | 38 415 | 1.7e−3 | 0.0504 vs 0.0500 | 1.008× |
| 0.5 | 0.05 | 0.95 | 1 537 | 3 074 | 2.1e−2 | 0.0548 vs 0.0500 | 1.10× |
| 0.1 | 0.05 | 0.95 | 1 537 | 15 370 | 4.2e−3 | 0.0509 vs 0.0500 | 1.02× |
| 0.01 | 0.01 | 0.95 | 38 415 | 3.8e6 | 1.7e−5 | 0.0500 vs 0.0500 | 1.000× |
| 1e−4 | 0.01 | 0.99 | 66 357 | 6.6e8 | 9.6e−8 | 0.0100 vs 0.0100 | 1.000× |

So at the server's default ±1% the approximation changes nothing measurable for
`p̂ ≲ 0.5`, and at ±5% it is a 2–20% widening of the failure probability. It is
*not* negligible for a loose `ε` on a high-probability event.

**Two options, for #224 to choose:**

1. **Engine statistic (preferred).** Add "index of the j-th hit within the pass"
   to the per-pass sufficient statistics contract (#223 item 6): the offset
   within `[first_trial, first_trial + trials)` of the `j`-th one-bit, for a
   caller-nominated `j`. That makes `T_k` exact and GBAS exact; the cost is a
   per-query scan of the pass's bitpack, which the kernel can do alongside the
   popcount. Pass sizing then also stops being a constraint, since a controller
   can shrink a pass to land on `k`.
2. **Accept the approximation, with the bound above.** Keep pass-level
   observation, and re-solve `k` at `ε' = ε/(1 + 64/T_k)` with `T_k` estimated
   from the crossing pass. The guarantee is then exact at `ε'`, and `ε' → ε` as
   `T_k` grows. This is faithful as long as the policy reports the `ε` it
   actually achieved (`ε` and `achieved_delta`, both already in the frame).

Either way the policy reports `k`, `T_k` (or its bounds), `R`, `p̂ = (k−1)/R`,
the exact interval `[Γ⁻¹_k(α/2)/R, Γ⁻¹_k(1−α/2)/R]`, and `achieved_delta`, and
the run reports `stop_reason: "goal"` with `interval: "gbas"`. The randomisation
in `R` is seeded from the run seed so a fixed-N rerun reproduces the estimate.

### Information gain and oracle diagnostics

| UI element | Today | Disposition |
| --- | --- | --- |
| Oracle probability field (`oracle`, `oracle_p`) | request knob, drives the oracle reference band on the p̂ plot | **kept as a request knob**; the *diagnostics* below become snapshot fields |
| Oracle reference band + its 95%/99% envelopes on the p̂ plot | computed client-side from `oracle_p` + `trials` | **kept client-side**; it is a pure function of `oracle_p` and `N` and needs nothing from the run |
| Oracle diagnostics: `bias`, `abs_error`, `rel_error`, `mse`, `z_score`, `p_value`, `ci95_covered`, `ci99_covered`, `n_required`, `n_ratio` | host metrics per pass, emitted when `oracle >= 0` | **optional snapshot field** — see below |
| Information-gain trace (`info_bits_last`, `info_bits_total`) | host metric, emitted every pass; no UI consumer (`liveConvergenceStats.ts` declares the keys, no component reads them) | **kept in phase 1** (removing a field is a change no client asked for), **proposed as an optional snapshot field** for #224 and off by default once the loop is deleted |
| `epsilon`/`target_epsilon` plot | commented out in `SamplingTrajectoryPanel.tsx` | not re-enabled; `epsilon`/`target_epsilon` stay in the frame for a client that wants it |

Proposed optional snapshot fields (one object per query, present only when the
run supplies an oracle or asks for them, absent otherwise):

```json
{
  "diagnostics": {
    "oracle_p": 0.0001243,
    "bias": -1.2e-6,
    "abs_error": 1.2e-6,
    "rel_error": 0.0097,
    "z_score": -0.31,
    "p_value": 0.756,
    "ci95_covered": true,
    "ci99_covered": true,
    "n_required": 3900000,
    "n_ratio": 0.87
  },
  "info_bits": {"last": 0.0031, "total": 11.42}
}
```

Both are plain functions of `(hits, trials, oracle_p, confidence)`; they need
nothing the controller does not already have, so they are cheap for #224 to add
as optional policy/metric outputs rather than stopping rules. Phase 1 keeps both
in the frame unchanged, because removing a field is a breaking change for a
client the server cannot enumerate; the removal is proposed for #224, where the
frame is rebuilt from snapshots and the fields become opt-in.

## Web UI run settings

The run-settings form is three separate answers, as on `fzb run`: the goal, the
limits and the design.

| Old control | Old option | New control | New field |
| --- | --- | --- | --- |
| `Trials` (mantissa + M/B/T) | `num_trials`, `early_stop: false` | `FIXED NUMBER OF TRIALS` on + `Fixed trials` | `stopping.trials` |
| `STOP ON CONVERGENCE` off | `early_stop: false` | `FIXED NUMBER OF TRIALS` on | `stopping.trials` |
| `STOP ON CONVERGENCE` on | `early_stop: true` | `FIXED NUMBER OF TRIALS` off (default) | — |
| `Delta (δ)` | `delta` | `Precision (±%)` | `stopping.rel_error = pct/100` |
| `Confidence` (default 99.999%) | `confidence` | `Confidence` (default 95%) | `stopping.confidence = pct/100` |
| `Trials` with early stop | `num_trials` as a cap | `Max trials` (empty = none) | `stopping.max_trials` |
| — | — | `Time limit` (empty = none) | `stopping.max_time` |
| policy hard-coded `"gbas"` | `policy` | removed | the Wilson goal, as #224 defaults |
| `µ` oracle field | `oracle` | unchanged | `oracle` (diagnostics only, not a stopping input) |
| `Trials / pass` | `requested_trials_per_pass` | unchanged | `requested_trials_per_pass` (sampler geometry) |

Display: the `Run` group shows the **stop reason** (`Goal met`, `Trial limit`,
`Time limit`, `Fixed design complete`, `Interrupted`) and the **per-query
convergence** — `± RELATIVE` with the achieved relative half-width and the goal
beside it — both read from the frame (`stop_reason`, `rel_half_width`,
`target_rel_half_width`). The GBAS plot renders only when the frames carry the
GBAS stream, which is the case only for a run that names no stopping intent at
all and inherits `MCSCRAM_POLICY=gbas` from the environment, plus legacy frames
replayed from before this change.

## `.pcfb` upload

`POST /api/v1/model/pcfb` accepts a raw `.pcfb` body (`Content-Type:
application/octet-stream`). The server validates with `fuzzybits.inspect`
(`pcfb::validate` diagnostics). Invalid circuits return HTTP 400:

```json
{"detail": {"error": "...", "diagnostics": [...], "inspect": {...}}}
```

Success:

```json
{"id": "<sha256>", "pcfb_id": "<sha256>", "path": "...", "inspect": {...}}
```

Start a run on the uploaded circuit:

```bash
curl -sS -X POST "$HOST/api/v1/model/pcfb" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @model.pcfb
curl -sS -X POST "$HOST/api/v1/start" \
  -H "Content-Type: application/json" \
  -d '{"pcfb_id":"<sha256>","watch":["g1"]}'
```

`GET /api/v1/model/{dag,kernel-dag,sample-shape,execution-plan,dag-bundle}`
accept `?pcfb_id=<sha256>` to serve those views from the stored circuit
instead of compiling MEF.

## Image

`ghcr.io/fuzzybit-org/fuzzybits-server` is `FROM` the slim runtime image
(`ghcr.io/fuzzybit-org/fuzzybits-runtime:nightly-generic` by default;
`scram-cli`, `fzb`, AdaptiveCpp, pip-installed `fuzzybits`). uv installs
this package into a venv with `--system-site-packages` so the engine is not
rebuilt. Variants match the runtime: `generic` (default) and `cuda12.8-sm70`.

Tags (same scheme as `fuzzybits-runtime`): `ci-<sha12>-<variant>` on `ci/**`,
`nightly-<variant>` + `sha-<sha12>-<variant>` on `develop`, `vX.Y.Z-<variant>`
on release tags.

The image runs as non-root `coder`. Default env:

| Variable | Image default |
| --- | --- |
| `FUZZYBITS_SERVER_DATA_DIR` | `/data` |
| `FUZZYBITS_SERVER_CACHE_DIR` | `/home/coder/.cache/fuzzybits/server` |
| `ACPP_APPDB_DIR` | `/home/coder/.cache/fuzzybits/acpp` |
| `ACPP_VISIBILITY_MASK` | `omp` (CLI tools only; see below) |
| `HOST` / `PORT` | `0.0.0.0` / `8000` |

The server's backend is `FUZZYBITS_BACKEND` / `--backend`
(`auto|cuda|hip|omp|metal`). Under `auto` (the default) `fuzzybits.init()`
picks CUDA if an NVIDIA device is visible, else HIP if `/dev/kfd` and an AMD
render node are passed in, else OpenMP, and **replaces** `ACPP_VISIBILITY_MASK`
with that choice — the image's `omp` mask does not pin the server to CPU. Set
`FUZZYBITS_BACKEND=omp` to keep a GPU-attached container on the CPU.

Volumes: `/data` (models) and `/home/coder/.cache/fuzzybits` (JIT + `.pcfb` cache).
`HEALTHCHECK` hits `/api/v1/healthz`.

### `docker run`

```bash
IMG=ghcr.io/fuzzybit-org/fuzzybits-server:nightly-generic

# CPU (OpenMP)
docker run --rm -p 8000:8000 \
  -e FUZZYBITS_BACKEND=omp \
  -v "$PWD/input/Aralia:/data/Aralia:ro" \
  -v fuzzybits-cache:/home/coder/.cache/fuzzybits \
  "$IMG"

# GPU (NVIDIA device N). Use the cuda12.8-sm70 tag on Volta (sm_70);
# generic SSCP covers other NVIDIA GPUs.
docker run --rm --gpus device=0 -p 8000:8000 \
  -e FUZZYBITS_BACKEND=cuda \
  -v "$PWD/input/Aralia:/data/Aralia:ro" \
  -v fuzzybits-cache:/home/coder/.cache/fuzzybits \
  ghcr.io/fuzzybit-org/fuzzybits-server:nightly-cuda12.8-sm70

# AMD GPU (generic only; ROCm 7 user space ships in the image, the host needs
# only the amdgpu driver). amd-docker-args.sh prints --device /dev/kfd, the
# GPU's render node and --group-add <render gid>. The HIP plugin must be in the
# mask AdaptiveCpp reads before main(), hence cuda;hip.
docker run --rm $(scripts/ci/amd-docker-args.sh) -p 8000:8000 \
  -e FUZZYBITS_BACKEND=hip -e ACPP_VISIBILITY_MASK='cuda;hip' \
  -v "$PWD/input/Aralia:/data/Aralia:ro" \
  -v fuzzybits-cache:/home/coder/.cache/fuzzybits \
  "$IMG"
```

xerxes mapping: `--gpus device=0` is the Tesla V100S, `device=1` is the RTX 3070.
bubs (mbp-1): Radeon Pro 5500M 8 GB, `gfx1012`. CI smokes the server on CPU only.

```bash
curl -sS http://127.0.0.1:8000/api/v1/healthz
curl -sS http://127.0.0.1:8000/api/v1/model/xml-inputs
curl -sS -X POST http://127.0.0.1:8000/api/v1/start \
  -H "Content-Type: application/json" \
  -d '{"model_id":"chinese","stopping":{"trials":1e7}}'
```

E2E helper (compile chinese + das9201, check \|z\|≤4 vs BDD refs, POST a
`fzb compile` `.pcfb`):

```bash
tools/server/scripts/e2e-smoke.sh "$IMG"

# on an AMD host
SMOKE_DOCKER_ARGS="$(scripts/ci/amd-docker-args.sh | tr '\n' ' ')" \
  ACPP_VISIBILITY_MASK='cuda;hip' FUZZYBITS_BACKEND=hip \
  tools/server/scripts/e2e-smoke.sh "$IMG"
```

### Compose

```yaml
services:
  server:
    image: ghcr.io/fuzzybit-org/fuzzybits-server:nightly-generic
    ports: ["8000:8000"]
    environment:
      FUZZYBITS_BACKEND: omp             # cuda / hip on GPU hosts (hip: ACPP_VISIBILITY_MASK: "cuda;hip")
      HOST: "0.0.0.0"
      PORT: "8000"
    volumes:
      - ../../input/Aralia:/data/Aralia:ro
      - fuzzybits-cache:/home/coder/.cache/fuzzybits
    # gpus: ["device=0"]                 # NVIDIA
    # devices: [/dev/kfd, /dev/dri/renderD128]   # AMD; group_add: [<render gid>]
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/healthz')"]
      interval: 15s
      timeout: 5s
      retries: 3

volumes:
  fuzzybits-cache:
```
