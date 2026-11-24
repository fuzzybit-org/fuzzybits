---
paths:
  - "src/io/**"
  - "src/compile/**"
---
# Readers, writers and the compiler (`src/io`, `src/compile`)

- pcfb: `docs/pc-pcfb.md` is the mapping between `ir::pc` and the file format, and holds the
  diagnostics table. Keep it in step with `src/io/pcfb`.
  - pcfb is the submodule `extern/pcfb`; its gitlink is the pin (`cmake/FuzzybitsPcfb.cmake`). Schema
    changes go to `fuzzybit-org/pcfb` first, then the gitlink moves.
  - Query kinds the runtime does not implement are rejected before sampling
    (`src/io/pcfb/query_guard.h`).
- FTP: `docs/ftp-input.md` describes the FTP → `ir::pra` → `ir::pc` → `.pcfb` pipeline.
  Conformance fixtures are in `input/ftp/conformance/`. nombrilo: `docs/nombrilo.md`.
- Diagnostic ids are kebab-case (`io.pcfb.invalid-file`); C++ constants stay snake_case.
- Tests must be reproducible on libc++ and libstdc++. Use `src/io/pcfb/tests/portable_random.h`, not
  `std::*_distribution` or `std::shuffle`.
- The OpenPSA MEF path still goes through SCRAM code (`src/*.cc`). Do not extend it. New formats
  produce `ir::pra` or `ir::pc`.
