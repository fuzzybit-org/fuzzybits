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
fzb compile top.ftp systems.ftp imports.ftp --target SEQ-1
fzb compile project_dir/ --negation hull --literals negation
```

A file is FTP when it ends in `.ftp` (any case), or is text (no NUL byte, not XML) with a line `ENDTREE`; a directory is
FTP when it holds a `*.ftp` file, and all of them are read. Several inputs form one project: the files are sorted by
path, repeated paths dropped, and read as their ordered union (sections do not continue from one file into the next).
Mixing FTP and non-FTP inputs is an error. MEF and nombrilo inputs are unchanged.

## Supported features

| FTP construct | Meaning in `ir::pra` |
|---|---|
| `NAME + ops`, `NAME * ops`, `NAME k ops` (other lines continue the open gate) | OR, AND and k-out-of-n gates, with file and line |
| `-name` operand | complemented operand; on a basic event always also marked `literal` (Boolean-neutral: only hull queries read the mark, see `--literals`); on a gate never marked, always a negation |
| `ENDTREE` | closes the tree section, which comes first in each file |
| `IMPORT` records `value name [flags]` | basic events with literal probabilities; the value `-1` is probability 1 (a warning) |
| `I` flag | initiating event; its value is a frequency, of any magnitude ([Initiating-event frequencies](#initiating-event-frequencies)) |
| `PROCESS names` | primary targets; without `PROCESS`, the root gates (a warning) |
| `LIMIT x` | recorded in the metadata only; never a truncation in this path |
| `MAXSIZE n` (outside a tree section) | the cut-set order limit, recorded like `LIMIT` (`ftp.maxsize`); never applied |
| `**CHAR32`, `**FTITLE:`, `*XEQ` and other `*` lines | project attributes, not interpreted |
| `;` | a comment to the end of the line, except in `*` lines |

An event imported twice is resolved by `--duplicate-import last|first` (a warning that names every record) or rejected
(`error`); an `I` on any record is kept. A name that is neither a gate nor imported is an error, never an implicit
probability of 1. Unknown import flags are kept with a warning.

### Grammar

The reader takes files as projects write them; `src/io/ftp/tests/grammar_tests.cc` and the `dialect_*` conformance
fixtures check each rule.

- **Lines and tokens.** A line ends at LF; a CR before it, spaces, tabs and every other control byte (NUL, form
  feed, the end-of-file mark 0x1A) separate tokens, so CRLF files read as LF files and a file with CR-only line
  endings is one line. A UTF-8 byte-order mark at the start of a file is skipped.
- **Comments.** `;` starts a comment that runs to the end of the line, on every line but the `*` header lines
  (`**DBNAME: C:\a;b.rr` keeps its text).
- **Keywords** (`ENDTREE`, `PROCESS`, `IMPORT`, `LIMIT`, `MAXSIZE`) and the `I` flag are read in any letter case.
  `MAXSIZE` is a keyword only outside a tree section, so a gate may be called `MAXSIZE`. Tokens after `ENDTREE`,
  `IMPORT`, `LIMIT x` or `MAXSIZE n` are not read (`io.ftp.ignored-tokens`).
- **Names** are any token except a lone `+` or `*`; they may start with a digit and hold any punctuation (`%IE`,
  `G~1`, `SYS.A`, `PUMP#1`, `V(2)`, `$V1`) and bytes above 127. A leading `-` on an operand complements it, so a
  defined name cannot start with `-`; a `*` at the start of a line makes it a header line, so a gate name cannot
  start with `*`; and a gate cannot be named `ENDTREE`, `PROCESS`, `IMPORT` or `LIMIT` in any case (such names may
  still be events).
- **Letter case.** Names are compared without regard to ASCII case: `g1`, `G1` and `PROCESS g1` name one gate. An
  element keeps the spelling of its gate line, or of its first import record for an event; every other spelling
  is read as that one, with one `io.ftp.name-case` warning per element. Bytes above 127 compare exactly. Two
  definitions that differ only in case are a duplicate gate (`io.ftp.duplicate-gate`), and a gate imported under
  another case is `io.ftp.gate-imported`.
- **Gate lines and continuations.** A line `NAME op ...` (op `+`, `*` or an integer k) defines a gate wherever it
  starts. Any other line in the tree section continues the operand list of the open gate, indented or not.
- **Values** are `[+-]? (d+ ('.' d*)? | '.' d+) ([EeDd] [+-]? d+)?`: `1.`, `.5`, `1.E-02` and `1.0D-01` are
  values; `.`, `5E`, `0,5`, `0x1`, `inf` and `nan` are not. A value too small for a double (`1E-400`) is 0. The value
  exactly `-1` (`-1`, `-1.0`, `-1.000000E+00`) marks an event at probability 1: it is read as 1 with
  `io.ftp.minus-one-value`. Any other negative value, and a value above 1 on an event without the `I` flag, is
  `io.ftp.invalid-probability`.

Still errors, by choice: a name that is neither a gate nor imported (never an implicit probability of 1), a second
definition of a gate, an empty gate, k above the operand count, `+A` written for `+ A`, and a value above 1 on an
event without the `I` flag. Each of these is a mistake that would otherwise change the model without a trace.

## Options

| Flag | Meaning | Default |
|---|---|---|
| `--target NAME\|@FILE` (repeatable) | PROCESS target, gate or basic event, one query each, in the given order; `@FILE` holds one name per line | every PROCESS target |
| `--watch NAME\|@FILE` (repeatable) | extra targets with the watch role (written as ordinary queries) | none |
| `--no-ccf` | accepted and ignored: FTP projects have no common-cause groups | |
| `--override NAME=0\|1` (repeatable) | the gate, basic event or house event becomes a constant; its operands leave the cone | none |
| `--duplicate-import error\|first\|last` | duplicate `IMPORT` records | `last` |
| `--negation boolean\|hull\|both` | the query type of every target: the probability (complements are negations), the monotone hull (complements are delete terms) or both | `boolean` |
| `--initiator-values frequency\|probability` | `frequency`: each `I` value is a frequency, and a target becomes one conditioned query per initiating event in its cone, grouped as its frequency; `probability`: values up to 1 are probabilities, above 1 an error | `frequency` |
| `--initiator-unit per-year\|per-month\|per-week\|per-day\|per-hour` | the unit of every `I` value with frequencies; rejected with probabilities | `per-year` with frequencies |
| `--max-initiators-per-cutset N\|none` | hull queries only, with `--initiator-values probability` only: every hull query is ANDed with the delete term of more than N initiating events in its cone; probability queries never are | `none` |
| `--literals atom_split\|omega_l\|negation` | how hull targets read marked literals (every `-EVENT`); `negation` reads them as plain complements, the plain hull | `atom_split` |
| `-c 0\|1` | `0`: no passes; `1`: `ir::pc` passes (constant folding of literal 0 and, for Boolean targets, 1; propagation; single-input elimination; coalescing) | `1` |
| `--config FILE` | quantification config (JSON, below); flags override it | none |
| `--report FILE` | JSON report: effective settings, lowering transform report, pass reports, sizes, timings, every diagnostic; native `fzb` also writes it when the compile fails | none |
| `-o FILE` | output path | `<model name>.pcfb` |
| `--no-progress`, `--quiet`, `--progress-log FILE` | as for MEF input | |

Precedence, highest first: command-line flags, the `--config` document, the built-in default. Nothing is read from
the model: not the query type (`negation`), the initiator reading or the initiator rule. `--target` and `--watch`
replace the document's lists; `--negation` and `--literals` replace every per-target choice in it; `--override`
replaces an entry of the same name. `LIMIT` is never an option: it goes to the metadata only. The text summary prints
counts (files, gates, events, circuit nodes and edges before and after the passes, warnings and notes); warnings print
up to ten, the report has all.

Python: `fuzzybits.compile([...], target=..., watch=..., overrides={...}, duplicate_import=..., negation=..., initiator_values=..., initiator_unit=..., max_initiators_per_cutset=..., literals=..., config=...,
pdag_c=..., report=True)`. With `report=True`
it returns `(blob, report)`; warnings are issued as `UserWarning`. The Python `fzb` script has the same flags.

### Config document

```json
{
  "negation": "boolean",
  "literals": "atom_split",
  "initiator_values": "frequency",
  "initiator_unit": "per-year",
  "compilation_level": 1,
  "duplicate_import": "last",
  "targets": ["TOP", {"name": "SEQ-1", "negation": ["boolean", "hull"], "literals": "omega_l"}],
  "watch": ["SEQ-3"],
  "overrides": {"PUMP-A": 0, "VALVE": true}
}
```

Every key is optional and unknown keys are an error (`compile.ftp.invalid-config`). A target is a name or an object with
`name`, `subject` (`domain_target`, `gate` or `basic_event`; default: looked up in that order), `negation` (`boolean`,
`hull`, `both` or an array) and `literals`. `initiator_values` and `initiator_unit` take the values of the flags.
`max_initiators_per_cutset` is a number, `null` or `"none"`, with `"initiator_values": "probability"` only. Overrides are an
object of `name: true|false|1|0` or an array of `{name, value, kind}`. TOML is not read yet.

## Negation semantics and the `.pcfb` writer

The query type is requested, never read from the model or derived from a target's cone. **Every target is a
probability query by default** (`--negation boolean`): every complement is a logical negation, and `fzb run` and
`fzb inspect --exact` give the probability of the target function. It is a valid query for every FTP file, with or
without complements.

`--negation hull` asks for the monotone hull instead: complements are delete terms, the reading of cut-set tools that
remove the cut sets containing a cut set of a complemented gate ([RFC](rfc/monotone-hull-queries.md)). It is a
different quantity wherever the target's cone has a complement. A hull target is written as a
`MONOTONE_HULL_PROBABILITY` query (pcfb 1.2, schema minor 2); a file with probability queries only stays at minor 1. A
target may ask for both kinds (`--negation both`, or `"negation": "both"` in the config): two queries on one tensor,
both named after the target. The compiler adapts to the query type: it skips the passes that do not preserve a hull
(`fold_unit_probability`) and guards the atom-split literals ([passes](ir-pc-passes.md)). The runtime estimator for hull
queries is #183: until then `fzb run`, `fuzzybits.run` and `fzb inspect --exact` refuse a file with a hull query
(`pcfb.unsupported-query-kind`).

pcfb 1.2 has no literal atoms per query yet, so a hull query can read a marked operand (every `-EVENT`: the reader
always marks it) only as a plain complement, which is the `negation` literal mode. A hull target whose cone has marked
operands is therefore refused in `atom_split` mode (the default) and in `omega_l` mode, with
`compile.pra_to_pc.hull-literals-unsupported` per target and `compile.ftp.hull-literals-unsupported`, which names the
alternatives. Each is a different query: `--negation boolean` (the probability, the default), or the plain hull with
`--literals negation`. Probability targets ignore the marks.

`--max-initiators-per-cutset N` (with `--initiator-values probability`) ANDs each hull query with
`not atleast(N + 1; initiating events in its cone)`: under the hull it deletes exactly the cut sets with more than N
initiating events ([RFC](rfc/monotone-hull-queries.md) §4.8). It is a rule of hull queries only: under the probability
query it would delete every assignment with more than N initiating events and change the target, so a rule without a
hull query is `compile.ftp.invalid-option`. A target that asks for both kinds keeps its probability query on the target
itself and gets the hull query on the conjunction, two queries on two tensors. The rule has no default; the reader
records no initiator convention.

## Initiating-event frequencies

The value on an `I`-flagged import is an initiating-event frequency (events per unit time), whatever its magnitude
(#243). FTP keeps the value and drops its unit, so the unit is given with `--initiator-unit`, `per-year` by default.
The compiler cannot check the unit: a project in another unit compiled without the flag is labelled per year. The
unit is written into each query group and printed with every result (`fzb run`, `fzb inspect --exact`,
`GroupResult.unit`). One unit applies to every `I` value of the project: a project that mixes units cannot be
detected from the export.

With `--initiator-values frequency` (the default), a target whose cone holds initiating events `I_1 … I_m` with values
`f_i` has the frequency

```
F(T) = Σ_i f_i · P(T | I_i = TRUE, I_j = FALSE for j ≠ i)
```

- The lowering writes one query per initiating event: the target with `I_i` set TRUE and the cone's other initiating
  events FALSE (constants, as with `--override`). Query `i` is named `<target>|<initiator>` and records
  `compile.query.<i>.initiator` and `.initiator_frequency`. With `--negation hull` each conditioned query is a hull
  query: `F` uses the hull of each conditioned target.
- The `.pcfb` file holds a query group per target and kind (pcfb `SubCircuit.query_groups`, spec §5.4): the target's
  name, the unit, and the terms `(query, f_i)`. The queries stay ordinary probability queries, so a runtime that does
  not read groups still answers each of them correctly.
- `fzb run` and `fuzzybits.run` print each group after the queries: `F` as the weighted sum of the query estimates,
  and the interval `[Σ f_i lo_i, Σ f_i hi_i]`. The queries share their samples, so their estimates are correlated;
  that interval is conservative whatever the correlation (with `m` terms at level `1 − α`, coverage is at least
  `1 − mα`). `fzb inspect --exact` and `fuzzybits.exact_groups` give the exact `F` of small models.
- `--exposure DURATION` (`fzb run`, `fzb inspect --exact`; `exposure=` in Python), a positive number followed by
  `y`, `mo`, `wk`, `d` or `h`, adds `P(at least one occurrence in τ) = 1 − e^{−F·τ}`, assuming occurrences form a
  Poisson process. A year is 365.25 days and a month a twelfth of it. The period converts the result only: it is
  never applied to the initiators, because `P(I) = 1 − e^{−f·τ}` saturates a frequent initiator and loses its
  multiplicity.
- A target that may hold with every initiating event FALSE mixes a probability with a frequency
  (`compile.pra_to_pc.mixed-units`). The check evaluates the target with every initiating event FALSE in three-valued
  logic: it is sound, but conservative where a target is false only for a reason that logic cannot see (`x ∧ ¬x`).
- The initiator rule is not used: each conditioned query has exactly one initiating event TRUE, which is what the rule
  approximates in cut sets. `--max-initiators-per-cutset N` with frequencies is an error (`none` is accepted).

`--initiator-values probability` keeps the earlier reading: values up to 1 are probabilities (note
`compile.pra_to_pc.frequency-as-probability`), values above 1 are rejected (`compile.pra_to_pc.frequency-out-of-range`),
and hull queries may take the initiator rule. The conformance fixtures that tag probabilities with `I` (`initiator_rule`,
`imports_tags_duplicates`, `multi_file_project`) are checked this way; `input/ftp/frequency/` holds the models of #243
with their frequencies.

## `.pcfb` metadata

| Key | Value |
|---|---|
| `general.source_format` | `cafta-ftp` |
| `general.source_sha256` | SHA-256 of the file contents concatenated in sorted path order (32 bytes; `fzb inspect --json` prints hex) |
| `general.model_name` | project name, else the input directory name, else the first sorted file's stem |
| `compile.query.<i>.negation`, `.literal_mode`, `.max_initiators_per_cutset`, `.initiator`, `.initiator_frequency` | per query `i` in file order: `boolean` or `monotone_hull`; for a hull query its literal mode; the initiator rule when its delete term is lowered into the query's target; and for a conditioned query its initiating event and frequency (written by the `.pcfb` writer) |
| `compile.options` | JSON of the effective settings: compilation level, reader options, initiator values and unit, initiator rule, the `ftp.limit` text, and per target (queries in file order, then watch targets) the subject kind, negation semantics, literal mode and the counts of initiating and literal events; overrides |

Query names are the target names (the PROCESS target, gate or basic event name), and `<target>|<initiator>` for a
conditioned query. Basic-event samples carry the event
name and gate nodes their gate name where an output node still computes them. Names survive `-c 1`: the lowering's
projection is composed with the pass node maps (`compose_write_options`), so a gate eliminated onto another node, a
folded constant or an absorbed child loses its name and nothing else does.

## `fzb inspect` and `fuzzybits.read_project`

```text
fzb inspect model.ftp
fzb inspect top.ftp systems.ftp imports.ftp --json
fzb inspect project_dir/ --duplicate-import first --strict
fzb inspect project_dir/ --json --counts-only
```

```python
doc = fuzzybits.read_project("project_dir", duplicate_import="last")
```

`fzb inspect` runs the project reader only: no lowering, no passes, no `.pcfb`. It takes the same inputs as
`fzb compile` (FTP detection, directories, mixed input rejected), so a project can be validated and what the reader
understood reviewed before compiling it. `.pcfb` inputs keep their behaviour (`--validate`, `--exact`); those two flags
are rejected for FTP input, and the FTP flags below for `.pcfb` input.

| Flag | Meaning |
|---|---|
| `--json` | print the content below as JSON (text otherwise, with the same fields) |
| `--duplicate-import error\|first\|last` | as for `fzb compile` (default `last`) |
| `--strict` | exit 1 on warnings as well as errors |
| `--counts-only` | no element names (see below) |

Exit status: 0 without errors, 1 with errors (also when `--strict` and there are warnings). A usage or option error
prints `fzb inspect: ...` to stderr and exits 1. Reader diagnostics are part of the output, on stdout.

The JSON document (`fuzzybits.read_project` returns the same content as a dict):

| Field | Content |
|---|---|
| `schema_version`, `status`, `counts_only` | `1`; `ok` (no errors) or `error`; whether `--counts-only` was given |
| `files` | the files read, in project order (directories expanded, sorted) |
| `diagnostics` | `{id, severity, file, line, message}`; severity is `error`, `warning` or `note`; `file` and `line` are `null` when the diagnostic names no location |
| `structure` | counts below; `null` when the project has errors |
| `conventions` | `LIMIT` (the text of the value), `MAXSIZE` (the text of the value) (model attributes the reader records; `null` when absent) and `header_lines` (the `*` lines, as read) |

`structure` uses the keys of `structure` in the conformance `expected.json` files, plus `default_targets` (a top-level
key of those files), `house_events`, `complemented_edges_to_gates` and `literal_operands`:

| Key | Content |
|---|---|
| `gates`, `gates_by_kind` (`or`, `and`, `atleast`), `atleast_k_of_n` (`"k/n"` -> gates) | gates and their kinds and cardinality bounds |
| `edges`, `single_input_gates`, `max_fan_in`, `depth` | operand count, gates with one operand, widest gate, longest gate chain |
| `basic_events`, `house_events`, `import_lines` | events, and `IMPORT` records (more than one per event when imported repeatedly) |
| `initiator_tagged_events`, `initiator_tagged_events_with_p_1` | initiating events, and those with value 1 |
| `events_p_eq_1`, `events_p_eq_0`, `events_0_lt_p_lt_2^-32` | basic events by value |
| `complemented_edges`, `complemented_edges_to_basic_events`, `complemented_edges_to_gates` | complemented operands, by what they reference |
| `literal_operands` | complemented basic-event operands marked as atomic literals: every `-EVENT` operand |
| `roots` | gates no other gate references |
| `process_targets`, `default_targets` | the `PROCESS` names as declared (empty when there are none), and the model's targets: those names, or the root gates (with `io.ftp.missing-process`) |
| `limit` | the `LIMIT` value, `null` when absent |
| `names_not_valid_xml_ncname` | element names that are not XML NCNames |
| `duplicate_imports` | per repeated event, its records `{file, line, chosen}` in project order; the chosen record (the one the policy kept) also has `value` |

Under `--duplicate-import error` a repeated import is an error, so `structure` is `null`; the diagnostic names the
records. `--duplicate-import` changes `structure` only through those counts.

`--counts-only` is for logs of models whose identifiers must not be published. No element name appears: diagnostics
keep `id`, `severity` and `file`/`line` and drop `message` (messages quote names); `roots`, `process_targets`,
`default_targets`, `duplicate_imports` and `header_lines` become counts. Files are still listed. Python: `read_project(..., counts_only=True)`.

Diagnostic ids are the reader's `io.ftp.*` ids (see below; they match the codes in the conformance fixtures) and
`ir.pra.*` model validation ids. `file` and `line` are the source position the reader recorded for the diagnostic. Python:
`read_project` raises `fuzzybits.FtpProjectError` (a `RuntimeError`) when the project has errors, with the diagnostics (a
list of dicts as above) in its `diagnostics` attribute; warnings are returned, not raised. Invalid `duplicate_import` values
raise `ValueError`.

## Diagnostics

Every diagnostic has a stable id; the report and `fzb inspect` list them with severity.

| Id | Meaning |
|---|---|
| `io.ftp.missing-endtree`, `io.ftp.misplaced-gate`, `io.ftp.invalid-line`, `io.ftp.invalid-gate`, `io.ftp.duplicate-gate` | grammar errors, with `file:line` |
| `io.ftp.import-without-header`, `io.ftp.invalid-probability`, `io.ftp.invalid-limit`, `io.ftp.invalid-maxsize` | `IMPORT`, `LIMIT` and `MAXSIZE` errors |
| `io.ftp.undefined-reference`, `io.ftp.undefined-process-target`, `io.ftp.gate-imported` | a name has no definition, or is both a gate and an imported event |
| `io.ftp.duplicate-import` | warning (error under `--duplicate-import error`), naming every record |
| `io.ftp.unknown-import-flag`, `io.ftp.duplicate-limit`, `io.ftp.duplicate-maxsize`, `io.ftp.missing-process` | warnings |
| `io.ftp.name-case`, `io.ftp.minus-one-value`, `io.ftp.ignored-tokens` | warnings: a name written in another letter case, a value `-1` read as 1, tokens after a keyword not read ([Grammar](#grammar)) |
| `io.ftp.no-targets`, `io.ftp.no-files`, `io.ftp.open-failed`, `io.ftp.invalid-option` | project and option errors |
| `ir.pra.cycle`, `ir.pra.invalid-*`, `ir.pra.duplicate-name` | model validation errors |
| `compile.ftp.invalid-option`, `compile.ftp.invalid-config`, `compile.ftp.invalid-json` | bad flag, config document or JSON |
| `compile.ftp.unknown-target` | `--target` or `--watch` names nothing |
| `compile.ftp.hull-literals-unsupported`, `compile.pra_to_pc.hull-literals-unsupported` | a hull target has marked literal operands in `atom_split` or `omega_l` mode (see above) |
| `compile.pra_to_pc.unknown-target`, `.unknown-element`, `.ambiguous-element`, `.duplicate-target`, `.duplicate-override`, `.unused-override` | target and override resolution |
| `compile.pra_to_pc.frequency-as-probability`, `.frequency-out-of-range` | with `--initiator-values probability`: an initiator frequency used as a probability (at most 1, note) or rejected |
| `compile.pra_to_pc.missing-initiator-unit` | frequencies without a unit in the lowering config; the FTP driver always supplies one (`per-year` by default), so only direct callers of the lowering see it |
| `compile.pra_to_pc.mixed-units` | a target may hold with every initiating event FALSE |
| `compile.pra_to_pc.initiator-rule-with-frequencies` | an initiator rule with frequencies (the driver reports it as `compile.ftp.invalid-option`) |
| `compile.pra_to_pc.complement-under-negation`, `.complemented-initiator`, `.mixed-literal` | hull literal checks |
| `pc.pipeline.incompatible-pass`, `pc.pipeline.pass-skipped`, `pc.pass.atom-split-skipped` | pass planning and skipped rewrites |
| `io.pcfb.*` | writer errors |

The conformance fixtures in `input/ftp/conformance/` document the expected diagnostics and values; the `invalid_*`
fixtures fail with their ids. `fzb compile --negation boolean --initiator-values probability` followed by
`fzb inspect --exact` reproduces the `boolean` values of every fixture (`src/compile/ftp/tests/cli_tests.cc`).

## Limits

- Probabilities are literals; parameters, rates and distributions are not part of FTP.
- Hull queries are written but not yet estimated (#183); hull targets with marked literals wait for literal atoms in pcfb.
- `fzb inspect --exact` for hull queries (#202) is separate.
- `--config` reads JSON only.
- Frequencies carry one unit per project, which FTP does not record. Importance measures of frequency targets are not
  defined yet.
