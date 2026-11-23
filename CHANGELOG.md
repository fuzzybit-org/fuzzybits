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

- `fzb inspect --exact` follows the query kind: a `MONOTONE_HULL_PROBABILITY` query is evaluated exactly
  (`exact_hull_probability  <p>`, the plain monotone hull of the circuit as written) by the subset-OR
  transform instead of refusing the file; `PROBABILITY` output is unchanged. `--query NAME` selects a query
  by name, and `--kind probability|hull` picks a kind for a name that is written as two queries.
- `fzb inspect --hull-check EVENTS` evaluates the per-sample hull check on named samples (`@FILE` reads one
  set per line), with `--witness` (a minimal witness of the sample's true cone events) and `--json`.
- Python `fuzzybits.exact` answers hull queries and takes `query=` / `kind=`; `fuzzybits.hull_check` reports
  one sample's hit, bounds, decision tier and minimal witness. `run` and `Session` still refuse hull queries
  until the estimator of #183 lands.
- `fzb compile` and `fuzzybits.compile` take `--target` and `--watch` repeatedly, and
  `--target @FILE` / `--watch @FILE` read one name per line (the Python API takes a name or a
  sequence). A file that is missing, is not a regular file or lists no names is an error.
- OpenPSA/MEF and nombrilo input take one target; several targets, a repeated name, and
  `--watch` on nombrilo input are errors instead of being ignored.
- FTP projects accept `--no-ccf` (and the deprecated `--ccf`) as no-ops.
