# Roadmap

READMEs state only what CI or a reproducible command proves; plans live here. The rules for changing this
file, starting work and running experiments are in [docs/planning.md](docs/planning.md). The
machine-readable copy that the scope check and the health report read is
[`.github/roadmap.toml`](.github/roadmap.toml). The two must agree.

Goals, in rank order: **correct**, **spine**, **speed**, **reach**, **ship**, **usable**, **evidence**.
Research goes to the [lab](lab/README.md).

## Now (at most two)

### 1. Correct by default

Every input fuzzybits accepts gets a correct estimate or a refusal before sampling.

**Exit:**

- No open `wrong-answer` issue.
- The default-loop job is green: every model in Aralia v1, `input/ftp/conformance` and `input/ftp/synthetic`
  either runs with default options to |z| ≤ 4 of its reference, or is refused at compile with a diagnostic
  listed in its expected file.

**Issues:**

- `wrong-answer`:
  - #65: `--seed` ignored;
  - #171 and #172: Metal;
  - #243: initiator values read as probabilities.
- #244: probability by default. PR #245: split the default flip from the frequency feature and the pin change.
- #247: the compile-level check, for Boolean targets.
- #30: verify, then close.
- To file:
  - refuse OpenPSA models with event trees until #148 lands, instead of dropping their sequences (#92);
  - the default-loop CI job.

### 2. Spine: one SCRAM-free pipeline, then MIT

The OpenPSA path (SCRAM PDAG) and the FTP path (`ir::pra`) become one pipeline.

**Exit:**

- **E1:**
  - no GPL-3.0-or-later file in the link closure of `fzb`, the wheel and the server;
  - OpenPSA runs through `ir::pra`, with tallies bit-identical to the SCRAM path on the catalog (#146, #148).
- **E2:** `scripts/ci/scram_provenance.py` reports 0, and GitHub, PyPI and the image labels say MIT (#149, #150).

**Issues:**

- The critical path, ready: #146 (engine on `ir::pc`) and #148 (OpenPSA → `ir::pra`).
- In progress: #147 (C2/C3) and #153 (event trees, CCF).
- Blocked: #149 and #150.
- Also: #92 (event trees), #112, #144, #145, #209.

## Next (in order)

3. **Speed: measured engine wins.**
   - Exit: the bench gate on Aralia v1 (V100S, RTX 3070) shows geomean pass time ≥ 1.5× faster than v0.18.7,
     with every accuracy model at |z| ≤ 4.
   - Kernel-local, can start when Correct exits: #16, #19, #95, #164, with #163 as the gate before any sampler
     change.
   - Structural, after #146: #20, #21, #204 (with #18, #173), #213.
4. **FTP plant models.**
   - Exit: the public FTP corpus compiles and runs by default as probability queries, and initiator values are
     reported as frequencies with a declared unit.
   - Hull queries join only if the hull RFC is accepted (see Lab).
   - Issues: the frequency-query half of #243 (to split off).
5. **Ship: install, operate, secure.**
   - Exit:
     - #250 closed;
     - clean-machine pip and image smoke tests green on Linux, macOS and Windows at release;
     - PR CI p50 ≤ 30 min.
   - #250 runs now, through the interrupt lane (`security`).
   - Issues: #35, #36, #37, #85, #88, #101, #102, #108, #109, #111, #203.
6. **Run control and reporting.**
   - Exit: `fzb run` and the server stop on a stated precision through one controller, with measured interval
     coverage.
   - Issues: #223 (blocked by #171), #224 (changes `fzb run` defaults: needs surface approval), #225 (PR #233),
     #96, #205, #216.
7. **Evidence: benchmark program.**
   - Exit: manifest v2 with reference kind, uncertainty and provenance for every public model, and a cross-tool
     results page with raw logs.
   - Issues: #4, #124, #156–#162, #184 (PR #236), #185.

## Later

- **Converters + train.**
  - Exit: one EBR-II back-fit with fuzzybits as the forward model.
  - Issues: #47, #56, #125, #154.

## Lab

At most two experiments at a time, with a charter and a time box ([lab/README.md](lab/README.md)).

- **Hull queries (`lab:hull`): frozen, pending review.** The RFC
  ([monotone-hull-queries.md](docs/rfc/monotone-hull-queries.md), #174) is `proposed`, and part of it is
  implemented on `develop`.
  - That code stays and keeps building. The `hull` area is frozen: new work needs `scope-exception`.
  - Probability is the FTP default and hull is opt-in (#244).
  - At the review, choose one:
    1. **Accept the RFC.** Hull becomes part of *FTP plant models*, and #183 and #202 move there.
    2. **Move the code to `lab/hull`** in one mechanical PR.
    3. **Remove it.**
  - Until then: #183, #202 and draft PRs #234, #235.

## Icebox

Starting any of these needs a lab charter or a product issue.

- Research: #5, #6, #7, #27, #28, #48–#54, #97.
- **Exact engines (`exact-kc`).**
  - #248 (OxiDD, which brings a Rust toolchain into every build) and #249 (`src/kc`, `src/sat`, CaDiCaL,
    pcfb#14).
  - They enter only with a product consumer in a `now` milestone, most likely #21 module collapse, and only the
    part that consumer needs.
  - #249's own data: on the model that motivated it, the BDD tier decided 21 of 2²⁰ lanes.

## Done

- Phase 0 "stop the bleeding" https://github.com/fuzzybit-org/fuzzybits/milestone/10
- Phase 1 "spine" https://github.com/fuzzybit-org/fuzzybits/milestone/11
- Phase 2 "measured wins" https://github.com/fuzzybit-org/fuzzybits/milestone/12: superseded by Correct, Speed,
  Ship and Run. Its closed issues remain there.
