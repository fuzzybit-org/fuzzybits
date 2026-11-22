# Release Notes (unreleased)

## Breaking Changes

- `fzb compile` and `fuzzybits.compile` now expand common-cause failure groups by default.
  Results change for models that define CCF groups and were compiled without `--ccf`.
  Use `--no-ccf` (`ccf=False` in Python) to keep the previous behaviour. The flag is recorded
  in the `compile.options` (`--no-ccf`) and `compile.ccf` metadata of the `.pcfb`.
- `--ccf` is accepted for one more release as a no-op that prints a deprecation warning.
  `--ccf` together with `--no-ccf` is an error.
- Python `compile(watch="a,b")` no longer splits the string on `,`, `;` or `:`: pass a list.

## Minor Changes

- `fzb compile` and `fuzzybits.compile` take `--target` and `--watch` repeatedly, and
  `--target @FILE` / `--watch @FILE` read one name per line (the Python API takes a name or a
  sequence). OpenPSA/MEF and nombrilo input take one target and reject several, or a repeated
  name, with an error instead of ignoring them. nombrilo input rejects `--watch`.
- FTP projects accept `--no-ccf` (and the deprecated `--ccf`) as no-ops.
