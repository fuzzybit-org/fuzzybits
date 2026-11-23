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

- Hull queries carry their literal atoms in the `.pcfb`: a monotone-hull target with marked `-EVENT`
  operators (`--complement-events atom`, the default) now writes `Query.literal_mode` and
  `Query.literal_samples` (pcfb 1.2 literal-atom amendment, RFC §7.3) instead of being refused. Every
  literal mode compiles: `--literals atom_split` (the default), `omega_l` and `negation`.
  - `compile.ftp.hull-literals-unsupported` and `compile.pra_to_pc.hull-literals-unsupported` are gone.
    The only remaining marked-literal error is `compile.pra_to_pc.mixed-literal` (an event with both
    marked and unmarked complemented operands in one cone, RFC §7.1), which the FTP reader never
    produces.
  - The literal mode and Δ are written in the `Query` fields; `compile.query.<i>.literal_mode` stays as
    an informative copy, and the reader takes the mode from the field (warning
    `io.pcfb.quantification-conflict` on a disagreement). `docs/pc-pcfb.md` lists every consumer.
  - `fzb inspect` and `fuzzybits.inspect` report a `literal_atoms` section: the mode and |Δ| per hull
    query.
  - New reader diagnostics: `io.pcfb.invalid-literal-fields`, `io.pcfb.unknown-literal-mode`,
    `io.pcfb.invalid-literal-sample`, `io.pcfb.invalid-literal-mode`.
  - The pass pipeline keeps atom-split hull targets atom-preserving near Δ; the differential test
    `fuzzybits_io_pcfb_tests "[literals][diff]"` checks 2000 seeded circuits through the pipeline and a
    writer → reader round trip in all three modes.
- `fzb compile` and `fuzzybits.compile` take `--target` and `--watch` repeatedly, and
  `--target @FILE` / `--watch @FILE` read one name per line (the Python API takes a name or a
  sequence). A file that is missing, is not a regular file or lists no names is an error.
- OpenPSA/MEF and nombrilo input take one target; several targets, a repeated name, and
  `--watch` on nombrilo input are errors instead of being ignored.
- FTP projects accept `--no-ccf` (and the deprecated `--ccf`) as no-ops.
