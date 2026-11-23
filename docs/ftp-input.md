# CAFTA FTP input

`fzb compile` and `fuzzybits.compile` read CAFTA FTP projects (`*.ftp`) and write a `.pcfb`. FTP is the first PRA format
that does not go through SCRAM code:

```text
*.ftp ──► io::ftp project reader ──► ir::pra::model
      ──► quantification config (flags, --config FILE, model attributes)
      ──► compile::pra_to_pc::lower ──► ir::pc::circuit + pc_projection
      ──► ir::pc passes (-c 0 | -c 1), per-target semantics
      ──► target and event names composed through the pass maps
      ──► io::pcfb writer ──► .pcfb
```

Sources: `src/io/ftp` (reader, ids `io.ftp.*`), `src/compile/pra_to_pc` (lowering, `compile.pra_to_pc.*`),
`src/ir/pc/passes` ([passes](ir-pc-passes.md), `pc.*`), `src/io/pcfb` (writer, `io.pcfb.*`) and the driver in
`src/compile/ftp` (ids `compile.ftp.*`). Design background: [`pra-pc-ir-architecture.md`](pra-pc-ir-architecture.md),
[`IR-PRA-REQUIREMENTS.md`](IR-PRA-REQUIREMENTS.md), [`rfc/monotone-hull-queries.md`](rfc/monotone-hull-queries.md).

## Inputs

```text
fzb compile model.ftp -o model.pcfb
fzb compile top.ftp systems.ftp imports.ftp --negation boolean
fzb compile project_dir/ --literals negation
```

A file is FTP when it ends in `.ftp` (any case), or is text (no NUL byte, not XML) with a line `ENDTREE`; a directory is
FTP when it holds a `*.ftp` file, and all of them are read. Several inputs form one project: the files are sorted by
path, repeated paths dropped, and read as their ordered union (sections do not continue from one file into the next).
Mixing FTP and non-FTP inputs is an error. MEF and nombrilo inputs are unchanged.

## Supported features

| FTP construct | Meaning in `ir::pra` |
|---|---|
| `NAME + ops`, `NAME * ops`, `NAME k ops` (indented lines continue) | OR, AND and k-out-of-n gates, with file and line |
| `-name` operand | complemented operand; on a basic event also marked `literal` (`--complement-events atom`, default); on a gate always a negation |
| `ENDTREE` | closes the tree section, which comes first in each file |
| `IMPORT` records `value name [flags]` | basic events with literal probabilities |
| `I` flag | initiating event; a value above 1 is a frequency and is used as a probability only up to 1, with a warning |
| `PROCESS names` | primary targets; without `PROCESS`, the root gates (a warning) |
| `LIMIT x` | recorded in the metadata only; never a truncation in this path |
| `**CHAR32`, `**FTITLE:`, `*XEQ` and other `*` lines | project attributes, not interpreted |

An event imported twice is resolved by `--duplicate-import last|first` (a warning that names every record) or rejected
(`error`); an `I` on any record is kept. A name that is neither a gate nor imported is an error, never an implicit
probability of 1. Unknown import flags are kept with a warning.

## Options

| Flag | Meaning | Default |
|---|---|---|
| `--target NAME\|@FILE` (repeatable) | PROCESS target, gate or basic event, one query each, in the given order; `@FILE` holds one name per line | every PROCESS target |
| `--watch NAME\|@FILE` (repeatable) | extra targets with the watch role (written as ordinary queries) | none |
| `--no-ccf` | accepted and ignored: FTP projects have no common-cause groups | |
| `--override NAME=0\|1` (repeatable) | the gate, basic event or house event becomes a constant; its operands leave the cone | none |
| `--duplicate-import error\|first\|last` | duplicate `IMPORT` records | `last` |
| `--complement-events atom\|negation` | `atom`: `-EVENT` is a marked literal; `negation`: an ordinary complement | `atom` |
| `--negation boolean\|hull` | Boolean complement, or cut-set deletion through the monotone hull | `hull` (the reader records `ftp.negation = delete_term`) |
| `--max-initiators-per-cutset N\|none` | every target is ANDed with the delete term of more than N initiating events in its cone | `1` (the reader records `ftp.max_initiators_per_cutset = 1`) |
| `--literals atom_split\|omega_l\|negation` | how hull targets read marked literals | `atom_split` |
| `-c 0\|1` | `0`: no passes; `1`: `ir::pc` passes (constant folding of literal 0 and, for Boolean targets, 1; propagation; single-input elimination; coalescing) | `1` |
| `--config FILE` | quantification config (JSON, below); flags override it | none |
| `--report FILE` | JSON report: effective settings, lowering transform report, pass reports, sizes, timings, every diagnostic; native `fzb` also writes it when the compile fails | none |
| `-o FILE` | output path | `<model name>.pcfb` |
| `--no-progress`, `--quiet`, `--progress-log FILE` | as for MEF input | |

Precedence, highest first: command-line flags, the `--config` document, the model attributes the reader records
(`ftp.negation`, `ftp.max_initiators_per_cutset`), the built-in default. `--target` and `--watch` replace the document's
lists; `--negation` and `--literals` replace every per-target choice in it; `--override` replaces an entry of the same
name. `LIMIT` is never an option: it goes to the metadata only. The text summary prints counts (files, gates, events,
circuit nodes and edges before and after the passes, warnings and notes); warnings print up to ten, the report has all.

Python: `fuzzybits.compile([...], target=..., watch=..., overrides={...}, duplicate_import=..., complement_events=...,
negation=..., max_initiators_per_cutset=..., literals=..., config=..., pdag_c=..., report=True)`. With `report=True`
it returns `(blob, report)`; warnings are issued as `UserWarning`. The Python `fzb` script has the same flags.

### Config document

```json
{
  "negation": "boolean",
  "literals": "atom_split",
  "max_initiators_per_cutset": null,
  "compilation_level": 1,
  "duplicate_import": "last",
  "complement_events": "atom",
  "targets": ["TOP", {"name": "SEQ-1", "negation": ["boolean", "hull"], "literals": "omega_l"}],
  "watch": ["SEQ-3"],
  "overrides": {"PUMP-A": 0, "VALVE": true}
}
```

Every key is optional and unknown keys are an error (`compile.ftp.invalid-config`). A target is a name or an object with
`name`, `subject` (`domain_target`, `gate` or `basic_event`; default: looked up in that order), `negation` (`boolean`,
`hull`, `both` or an array) and `literals`. `max_initiators_per_cutset` is a number, `null` or `"none"`. Overrides are an
object of `name: true|false|1|0` or an array of `{name, value, kind}`. TOML is not read yet.

## Negation semantics and the `.pcfb` writer

`--negation boolean` reads every complement as logical negation: the query is the exact probability of the target
function, and the result of `fzb run` and `fzb inspect --exact` is that probability. FTP models are written for a
solver that deletes cut sets with a complemented gate (`ftp.negation = delete_term`); that is the monotone-hull
reading (`--negation hull`), which differs from the Boolean one. **Hull is the default for FTP input.** A hull target is
written as a `MONOTONE_HULL_PROBABILITY` query (pcfb 1.2, schema minor 2); a file with Boolean queries only stays at minor 1.
A target may ask for both kinds (`"negation": "both"` in the config): two queries on one tensor, both named after the
target. The runtime estimator for hull queries is #183: until then `fzb run`, `fuzzybits.run` and `fzb inspect --exact`
refuse a file with a hull query (`pcfb.unsupported-query-kind`), and `--negation boolean` is what runs today.

pcfb 1.2 has no literal atoms per query yet, so a hull query reads a marked operand (`-EVENT` under
`--complement-events atom`) as a plain complement, which is the `negation` literal mode. A hull target whose cone has
marked operands is therefore refused in `atom_split` mode (the default) and in `omega_l` mode, with
`compile.pra_to_pc.hull-literals-unsupported` per target and `compile.ftp.hull-literals-unsupported` naming the ways out:
`--literals negation` (the hull of the plain complements), `--complement-events negation` (no marks) or
`--negation boolean`. Boolean targets ignore the marks.

`--max-initiators-per-cutset N` ANDs each target with `not atleast(N + 1; initiating events in its cone)`. Under Boolean
semantics this deletes the assignments with more than N initiating events (a probability change), under the hull it
deletes the cut sets with more than N.

## `.pcfb` metadata

| Key | Value |
|---|---|
| `general.source_format` | `cafta-ftp` |
| `general.source_sha256` | SHA-256 of the file contents concatenated in sorted path order (32 bytes; `fzb inspect --json` prints hex) |
| `general.model_name` | project name, else the input directory name, else the first sorted file's stem |
| `compile.query.<i>.negation`, `.literal_mode`, `.max_initiators_per_cutset` | per query `i` in file order: `boolean` or `monotone_hull`, the literal mode, and the initiator rule when one is lowered into the target (written by the `.pcfb` writer) |
| `compile.options` | JSON of the effective settings: compilation level, reader options, initiator rule, the `ftp.limit` text, and per target (queries in file order, then watch targets) the subject kind, negation semantics, literal mode and the counts of initiating and literal events; overrides |

Query names are the target names (the PROCESS target, gate or basic event name); basic-event samples carry the event
name and gate nodes their gate name where an output node still computes them. Names survive `-c 1`: the lowering's
projection is composed with the pass node maps (`compose_write_options`), so a gate eliminated onto another node, a
folded constant or an absorbed child loses its name and nothing else does.

## Diagnostics

Every diagnostic has a stable id; the report lists all of them with severity.

| Id | Meaning |
|---|---|
| `io.ftp.missing-endtree`, `io.ftp.misplaced-gate`, `io.ftp.invalid-line`, `io.ftp.invalid-gate`, `io.ftp.duplicate-gate` | grammar errors, with `file:line` |
| `io.ftp.import-without-header`, `io.ftp.invalid-probability`, `io.ftp.invalid-limit` | `IMPORT` and `LIMIT` errors |
| `io.ftp.undefined-reference`, `io.ftp.undefined-process-target`, `io.ftp.gate-imported` | a name has no definition, or is both a gate and an imported event |
| `io.ftp.duplicate-import` | warning (error under `--duplicate-import error`), naming every record |
| `io.ftp.initiator-frequency`, `io.ftp.unknown-import-flag`, `io.ftp.duplicate-limit`, `io.ftp.missing-process` | warnings |
| `io.ftp.no-targets`, `io.ftp.no-files`, `io.ftp.open-failed`, `io.ftp.invalid-option` | project and option errors |
| `ir.pra.cycle`, `ir.pra.invalid-*`, `ir.pra.duplicate-name` | model validation errors |
| `compile.ftp.invalid-option`, `compile.ftp.invalid-config`, `compile.ftp.invalid-json` | bad flag, config document or JSON |
| `compile.ftp.unknown-target` | `--target` or `--watch` names nothing |
| `compile.ftp.hull-literals-unsupported`, `compile.pra_to_pc.hull-literals-unsupported` | a hull target has marked literal operands in `atom_split` or `omega_l` mode (see above) |
| `compile.ftp.default-negation`, `compile.ftp.default-initiator-rule` | notes: a default came from a model attribute |
| `compile.ftp.unknown-attribute` | warning: an attribute value the driver does not know is ignored |
| `compile.pra_to_pc.unknown-target`, `.unknown-element`, `.ambiguous-element`, `.duplicate-target`, `.duplicate-override`, `.unused-override` | target and override resolution |
| `compile.pra_to_pc.frequency-as-probability`, `.frequency-out-of-range` | an initiator frequency used as a probability (at most 1, warning) or rejected |
| `compile.pra_to_pc.complement-under-negation`, `.complemented-initiator`, `.mixed-literal` | hull literal checks |
| `pc.pipeline.incompatible-pass`, `pc.pipeline.pass-skipped`, `pc.pass.atom-split-skipped` | pass planning and skipped rewrites |
| `io.pcfb.*` | writer errors |

The conformance fixtures in `input/ftp/conformance/` document the expected diagnostics and values; the `invalid_*`
fixtures fail with their ids. `fzb compile --negation boolean` followed by `fzb inspect --exact` reproduces the `boolean`
values of every fixture (`src/compile/ftp/tests/cli_tests.cc`).

## Limits

- Probabilities are literals; parameters, rates and distributions are not part of FTP.
- Hull queries are written but not yet estimated (#183); hull targets with marked literals wait for literal atoms in pcfb.
- `fzb inspect` for FTP projects (#201) and `--exact` for hull queries (#202) are separate.
- `--config` reads JSON only.
