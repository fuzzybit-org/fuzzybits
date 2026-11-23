# Changelog

## Unreleased

### Breaking changes

- Common-cause failure groups are expanded by default in `fzb compile`, `fuzzybits.compile`, the
  native `compile_pcfb` / `run_probability` bindings and `fuzzybits-server`. Results change for
  models that define CCF groups and were compiled without `--ccf`.
  - Disable it with `--no-ccf` (`fzb compile`), `ccf=False` (Python) or `MCSCRAM_CCF=false` (server).
  - The choice is recorded in the `compile.options` (`--no-ccf`) and `compile.ccf` metadata of the `.pcfb`.
  - `--ccf` is accepted for one more release as a no-op that prints a deprecation warning;
    `--ccf` together with `--no-ccf` is an error. `scram-cli` keeps its own `--ccf`.
- Python `compile(watch="a,b")` no longer splits the string on `,`, `;` or `:`: pass a list.
- The server's compile cache key now includes the effective `ccf` setting, so blobs cached
  without expansion are not reused.

### Changes

- AMD GPUs through AdaptiveCpp's HIP backend (`--backend hip`, `backend="hip"`; `auto` picks
  cuda, then hip, then omp) in the `generic` wheel and images. Needs the ROCm 7 user space and
  access to `/dev/kfd` and a render node (#228).
- `fzb compile` and `fuzzybits.compile` take `--target` and `--watch` repeatedly, and
  `--target @FILE` / `--watch @FILE` read one name per line (the Python API takes a name or a
  sequence). A file that is missing, is not a regular file or lists no names is an error.
- OpenPSA/MEF and nombrilo input take one target; several targets, a repeated name, and
  `--watch` on nombrilo input are errors instead of being ignored.
- FTP projects accept `--no-ccf` (and the deprecated `--ccf`) as no-ops.
- Benchmarks: dataset manifest v2 is a draft (`benchmarks/datasets/v2/manifest.json`) with a card
  per model — formats with hashes, computed structure, origin, licence, reader status and tags —
  and a private entry for a model that cannot be published (#158, #184).
  - `python -m benchmarks.datasets.structure` computes a card's structure counts.
  - `python -m benchmarks.datasets.v2_freeze` regenerates the draft (`--check` reports a stale
    one). `python -m benchmarks.datasets.freeze` now only validates; `--write-v1` regenerates v1
    and `--freeze` is the only way to set `frozen: true`. v1 stays byte-identical.
  - `python -m benchmarks private` runs the private entries, and `run --private DIR` runs them
    beside the public models. Private files are found by sha256, so the repository holds no path,
    name or value for them, and published output carries only
    `private <id> <mode>: pass|fail|skipped[ (<reason>)]`.

