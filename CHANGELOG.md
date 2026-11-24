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
- FTP names are compared without regard to ASCII case (#240): `g1` and `G1` are one element, named
  as its gate line or first import record spells it, with an `io.ftp.name-case` warning for each
  element written in another case. A project that used two such names for different elements now
  reads them as one (or fails with `io.ftp.duplicate-gate`). To keep them apart, rename one.
- In an FTP tree section a line `NAME op ...` now always defines a gate, and any other line
  continues the open gate, indented or not. An indented continuation line whose second operand is
  an event named with digits only (`  A 7 B`) used to add operands; it now defines an at-least gate.
  To keep it a continuation, move that operand off the second position.

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
- The FTP reader accepts the files projects export (#240; rules in `docs/ftp-input.md`, Grammar):
  - names hold any punctuation, such as `%IE`, `G~1`, `SYS.A`, `PUMP#1`, `V(2)` or `$V1`;
  - keywords and the `I` flag in any letter case (`import`, `Endtree`, `i`);
  - `;` comments, values written `1.`, `.5`, `1.E-02` or `1.0D-01`, and continuation lines without
    indentation;
  - the value `-1` as probability 1 (`io.ftp.minus-one-value` warning);
  - `MAXSIZE n`, recorded as `ftp.maxsize` and in `fzb inspect` conventions (`MAXSIZE`), never applied;
  - a UTF-8 byte-order mark, and control bytes as blanks.
- Tokens after `ENDTREE`, `IMPORT`, `LIMIT x` and `MAXSIZE n` are a warning (`io.ftp.ignored-tokens`)
  instead of an error.
