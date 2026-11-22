# FTP IO tests

- `conformance_tests.cc`: every fixture under `input/ftp/conformance/` (#179, #191) is read with
  `ftp::read_project` under each duplicate-import policy; diagnostics, structure counts, default targets
  and the exact Boolean probability of every listed target must match `expected.json`.
- `project_tests.cc`: the `ir::pra` model (gates, events, initiators, targets, attributes, source
  locations), multi-file projects, one negative case per `io.ftp.*` diagnostic, and a 1e5-gate chain.
- `write_tests.cc`: the `ir::pc` to FTP writer. Public fixtures live in `tests/testdata/` and
  `input/ftp/synthetic/`; each is read, lowered (`src/compile/ftp/tests/support.h`), written and re-read to a
  stable checksum.
- `src/compile/ftp/tests`: `fzb compile` on the conformance fixtures, options, config, report and metadata.

CAFTA / MHTGR / G-PWR / other vendor FTP models are **not** in this repository (see fuzzybits#38).
With `FUZZYBITS_PRIVATE_FIXTURES_DIR` set, every directory below it holding `.ftp` files joins the
round-trip corpus as one project, and every `ftp/<MODEL>/` with a `<MODEL>.card.json` is read and
compared with the card's `structure` under its `policy.duplicate_import`. Only pass/fail, mismatching
field names and read times are printed:

```
FUZZYBITS_PRIVATE_FIXTURES_DIR=/path/to/private-fixtures ctest -R fuzzybits_io_ftp_tests
```

Without the variable the private case `SKIP`s.
