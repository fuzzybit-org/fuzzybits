# `fuzzybits::ir::pra` Requirements

Status: R1 value model, builder and validation implemented (#175); CAFTA FTP reader implemented (#180); other readers,
lowering and evaluation deferred  
Scope: `src/ir/pra`, PRA IO adapters, PRA-to-PC lowering, projection, and PRA-domain result serialization

## Boundary

`ir::pra::model` is the canonical in-memory PRA domain model. It is not an OpenPSA MEF object model, a SAPHIRE MAR-D
object model, a KUREAS editor state, or a PC graph with names attached. External schema details either normalize into
typed PRA semantics or remain source-provenance/attribute records.

The PRA IR must not parse files, evaluate probability expressions, expand CCF groups, run solvers, or serialize reports.
Those actions belong to IO, compiler, solver, pipeline, and writer endpoints.

## Coverage Targets

The public model must be able to name and relate these information classes:

1. Project metadata, source artifacts, schema/tool versions, stable names, labels, descriptions, and external attributes.
2. Hazards, hazard groups, initiating events, initiator frequencies, and event-tree linkage.
3. Event trees, functional events, branch states, branch conditions, sequence paths, end states, transfers, and path
   frequencies.
4. Fault trees, gates, basic events, house events, gate inputs, transfers, top events, and component/failure-mode links.
5. Systems analysis concepts such as systems, components, functions, FMEA-style failure modes, and criteria formulas.
6. Parameters and expressions for probabilities, frequencies, rates, mission-time conversion, distributions, deviates,
   and unsupported external functions.
7. Unexpanded common-cause groups, members, factors, expansion state, and generated-event provenance anchors.
8. Post-processing rules including substitutions, recovery/deletion/addition rules, alignments, phases, cut-set filters,
   truncation settings, roll-ups, and imported product caches.
9. Domain targets and sparse projection bindings to lowered PC targets/events.

## Evidence From Formats

OpenPSA MEF provides explicit fault-tree, event-tree, model-data, CCF, substitution, rule, alignment, phase, consequence,
and attribute layers.

SAPHIRE MAR-D provides equivalent semantics through separate decks: `.FTL` fault-tree algebra, `.ETL` event-tree logic
matrices and node substitutions, `.BEI` basic-event quantification, `.BED` event descriptions, `.GTD` gate descriptions,
`.MTD` initiating-event taxonomy, `.FTA` analysis settings, and `.FTC` / `.SQC` / `.ESC` cut-set caches.

KUREAS `KNOW_FTL` contributes a JSON fault-tree editor representation. `KNOW_SYS` and hazard-analysis files add systems,
components, functions, failure modes, FMEA/reliability-library links, hazard hierarchy, and report/UI metadata. The core
IR should normalize only stable PRA semantics; editor state and layout remain attributes or extensions.

## Non-Commitments

The first `ir::pra` contract does not commit to:

- an evaluator for all probability/expression nodes;
- a total OpenPSA or SAPHIRE round-trip guarantee;
- a mandatory CCF expansion algorithm;
- a node-for-node PRA-to-PC map;
- canonical imported cut sets as solver truth;
- UI/report layout semantics as core PRA math.

When a reader encounters unsupported or ambiguous semantics, it should preserve the source identity and produce explicit
unsupported expression/formula/rule records plus diagnostics.

## Projection Requirement

PRA-to-PC lowering must produce sparse anchors, not a full reconciliation map. Required anchors are requested domain
targets, domain basic events needed for products/importance, generated CCF events, and aggregation bindings for
sequences/end states/hazard or system roll-ups. Intermediate PC nodes remain compiler artifacts unless they are needed
for projection, diagnostics, or provenance.

## Implemented: R1 Fault-Tree Subset (#175)

`src/ir/pra` (CMake target `fuzzybits_ir_pra`, umbrella header `ir/pra/pra.h`) implements the fault-tree subset of the
value contract, a builder and structural validation. It does not parse files, evaluate expressions, lower to PC or
expand CCF.

### Storage

- **Ids:** dense, 32-bit, strongly typed per table (`element_id`, `source_file_id`, `expression_id`, `basic_event_id`,
  `house_event_id`, `initiating_event_id`, `gate_id`, `fault_tree_id`, `domain_target_id`), assigned in insertion
  order. `domain_ref` is a typed reference (`element_kind` plus row) to any element table.
- **Tables:** one `std::vector` of trivially copyable rows per entity kind. Per-entity lists (gate operands, attributes,
  source locations, fault-tree gates and top gates) are `table_range` slices of shared columns. Strings live in one text
  pool and rows hold `text_ref` slices, so there are no per-entity heap objects.
- **Sealed model:** `pra::model` is immutable, shares its storage between copies and exposes spans. Lookup by stable
  name within a kind (`find`, `find_gate`, ...) uses an open-addressing index built at seal time (expected O(1)).
  `topological_gates()` lists every gate after the gates among its operands. `summary()` records gate and operand
  counts, complemented operands, operands marked `literal`, gate depth, and the gate and expression kinds present.
- **Handle:** `to_handle()` wraps the model in `ir::handle` with storage type and form `ir.pra.model` (endpoint
  `ir.pra`); the form profile also satisfies `ir.probabilistic_risk_model`. `pra::is_pra(handle)` and
  `pra::model_from(handle)` extract it.

### Entities

| Entity | Fields | Notes |
|---|---|---|
| `element_metadata` | owner, `stable_name`, `qualified_name`, `label`, attributes, sources | one row per entity of every kind below |
| `source_file` | `path`, `sha256`, `format` | |
| `source_location` | file, line | line 0 means unknown; an entity may have several locations |
| `attribute` | name, value | source hints that are not model semantics (FTP `LIMIT`, header lines, unknown flags) |
| `project_metadata` | name, label, attributes | |
| `expression` | kind (`literal`, `parameter`, `unsupported`), quantity (`probability`, `frequency`), value, parameter name, diagnostic id, detail | stored, never evaluated; basic events may share one |
| `basic_event` | element, expression | |
| `house_event` | element, value | |
| `initiating_event` | element, basic event | the basic event carries the value, which may be a frequency |
| `gate` | element, kind, operands, cardinality bounds, owning fault tree | kinds `and`, `or`, `cardinality(low, high)`, `not`, `xor` (odd parity), `nand`, `nor`, `xnor` (even parity); operands reference a basic event, house event or gate, each with a complement flag and a `literal` marker (see below) |
| `fault_tree` | element, gates, top gates | gates are those whose owner is the tree; top gates must be owned by the tree |
| `domain_target` | element, kind, role (`primary`, `watch`), subject | kinds `fault_tree_top` (subject: a fault tree with exactly one top gate), `gate`, `basic_event`; `sequence` and `end_state` are reserved |

At-least-k over n operands is `cardinality(k, n)`.

**Literal marker.** `operand_spec::literal` (stored as `gate_operand::literal`) says that a complemented basic-event
operand is an atomic literal of that event as the source wrote it. It is Boolean-neutral: the operand still means the
complement of the event, and `ir::pra` neither interprets the marker nor infers it. Consumers that distinguish atomic
literals from negations (for example hull queries, where a marked operand is its own droppable atom and an unmarked
complement is a delete term) read it from the operand. The builder accepts it only on a complemented operand that
resolves to a basic event. Only readers whose source format defines such literals set it; today that is the FTP
reader. A model without marked operands is read as if no source convention applied.

### Construction

`pra::builder` returns ids as entities are added and resolves references when it seals. A reference is an id or a
stable name, so a gate may name a gate, event or target subject that is added later or comes from another source file
of the same project. An unqualified name resolves against the kinds the use site admits (gate operands: gates, basic
events, house events) and must match exactly one; `reference::named(kind, name)` restricts it to one kind. Stable names
are compared exactly and may be any non-empty byte string (`1A`, `X/Y`, `A--B`). Attributes and source locations can be
attached when an entity is added or later.

`std::move(builder).seal()` returns `finalize_result { model output; api::diagnostic_list diagnostics; }`. Every error
found is reported; the model is returned only when there are none.

### Validation

Hard IR validity errors, all with severity `error`:

| Diagnostic id | Condition |
|---|---|
| `ir.pra.missing-name` | an entity has an empty stable name |
| `ir.pra.duplicate-name` | a stable name is used twice within one entity kind; the message lists the source locations |
| `ir.pra.dangling-reference` | a name or id resolves to nothing: gate operands, the basic event of an initiating event, fault-tree top gates and owners, target subjects, expression ids, source files of locations, owners of late attributes and locations |
| `ir.pra.ambiguous-reference` | an unqualified name matches several admissible kinds |
| `ir.pra.invalid-reference` | a reference resolves to a kind the use site does not admit |
| `ir.pra.cycle` | gates depend on themselves; one diagnostic per cycle found, naming its gates |
| `ir.pra.empty-gate` | an `and`, `or`, `xor`, `nand`, `nor`, `xnor` or `cardinality` gate without operands |
| `ir.pra.invalid-arity` | a `not` gate without exactly one operand |
| `ir.pra.invalid-cardinality` | bounds outside `0 <= low <= high <= n` |
| `ir.pra.invalid-literal` | a non-finite literal, a probability outside [0, 1] or a negative frequency |
| `ir.pra.invalid-literal-operand` | an operand marked `literal` that is not complemented or does not reference a basic event |
| `ir.pra.invalid-expression` | a parameter expression without a name or an unsupported expression without a diagnostic id |
| `ir.pra.invalid-fault-tree-top` | a top gate not owned by its fault tree, or listed twice |
| `ir.pra.invalid-target` | a `fault_tree_top` target on a fault tree without exactly one top gate |
| `ir.pra.unsupported-target-kind` | a `sequence` or `end_state` target |
| `ir.pra.capacity-exceeded` | a table or the text pool exceeds the 32-bit range |

Endpoint-specific rules stay in the endpoints: for example the FTP reader's duplicate-import policy, its `k = 0`
diagnostic and its handling of out-of-range initiator values (#180), and the lowering's probability policy (#181).

Every pass is linear, or expected linear for name lookups, and iterative: Kahn's algorithm orders gates and an
iterative walk over the gates it cannot order extracts the cycles. Nothing recurses on graph depth. A synthetic
100,001-gate DAG of depth 251 builds and seals in about 0.1 s in Release, and a 100,000-gate chain is ordered, or its
cycle reported, with a 128 KiB stack.

### Extension Points Left Open

- `expression_kind` grows with rates, mission-time conversion and deviates once the evaluator gate below is met.
- `domain_target_kind` already reserves `sequence` and `end_state`; `element_kind` grows with event-tree entities.
- Nested formulas (OpenPSA gates with formula children, event-tree branch conditions) need a formula arena; R1 gates
  hold one connective over event operands, and constants are house events.
- Parameter definitions are not entities yet; a `parameter` expression names its parameter.

## Deferred Implementation Gates

R1 implements the value model, builder and validation above. The following executable pieces remain intentionally
unimplemented until their input/output semantics and compatibility obligations are precise enough to avoid baking
schema-specific behavior into `ir::pra`.

### Model Construction And Validation

Implemented for the R1 subset: dense ids, `table_range` slices, reference resolution and the diagnostics listed above.

Still needed:

- construction and validation of formula arenas, event-tree entities, parameters and CCF declarations as those
  entities are added;
- invariant checks for malformed formula/expression operands once formulas and expression operands exist.

Implementation gate:

- extend the builder only with the entity kinds a reader or compiler endpoint needs;
- keep hard IR validity requirements (the `ir.pra.*` errors) separate from endpoint-specific compatibility constraints;
- define diagnostics ids for structural errors before adding silent normalization or repair behavior.

### External Readers

Still needed:

- OpenPSA MEF input reader producing `ir::pra::model`;
- SAPHIRE MAR-D reader mapping `.FTL`, `.ETL`, `.BEI`, `.BED`, `.GTD`, `.MTD`, `.FTA`, `.FTC`, `.SQC`, and `.ESC` records;
- KUREAS native readers for `KNOW_FTL`, `KNOW_SYS`, and hazard-analysis files where they carry stable PRA semantics.

Implemented: the CAFTA FTP project reader (#180, `src/io/ftp`, endpoint `io.fuzzybits.ftp.project_reader`,
`ftp::read_project`). A project is one or more `.ftp` files (or a directory, all `*.ftp` sorted) read as their ordered
union; sections do not continue from one file into the next. It is the only FTP parser (the legacy FTP to `ir::pc`
builder was removed with #182); `fzb compile` drives it ([`ftp-input.md`](ftp-input.md)).

| FTP construct | `ir::pra` | Class |
|---|---|---|
| `NAME + ops`, `NAME * ops`, `NAME k ops` (indented lines continue the operands) | `or` / `and` / `cardinality(k, n)` gate, source file:line | core |
| `-name` operand | complemented operand; on a basic event also marked `literal` (`io.ftp.complement_events = atom`, default; `negation` leaves it unmarked); on a gate never marked (a negation, i.e. a delete term under hull queries) | core |
| `ENDTREE` (the file's tree section comes first and must be closed) | — | `io.ftp.missing-endtree`, `io.ftp.misplaced-gate` |
| `IMPORT` records `value name [flags]` | basic event, `literal` probability, one source per record | core |
| `I` flag | initiating event on the basic event; a value > 1 is a `frequency` literal | core |
| other flags | element attribute `ftp.import_flag` | extension (warning) |
| event imported twice | `io.ftp.duplicate_import = last` (default) / `first` (warning) / `error`; an `I` on any record is kept | diagnosable |
| referenced name that is neither a gate nor imported | — (never an implicit p = 1) | `io.ftp.undefined-reference` |
| `PROCESS names` (continued on following name-only lines) | primary `gate` / `basic_event` domain targets | core |
| no `PROCESS` | the root gates as primary targets | `io.ftp.missing-process` (warning) |
| `LIMIT x` | project attribute `ftp.limit` | extension |
| source conventions | project attributes `ftp.negation = delete_term` (always) and `ftp.max_initiators_per_cutset = 1` (`io.ftp.max_initiators_per_cutset = 1`, default; `none` omits it); inert in `ir::pra` and the lowering; the compile driver (#182) maps them to quantification defaults | extension |
| `**CHAR32`, `**FTITLE:`, `**DBNAME:`, `*XEQ` and other `*` lines | project attributes `ftp.header` (whole line, in order) | extension |

Implementation gate:

- each reader must have a feature mapping table classifying fields as lossless core semantics, preserved extension
  attributes, diagnosable loss, or unsupported;
- parser-native structures must stay private to the IO backend;
- unsupported or ambiguous source constructs must produce diagnostics and explicit unsupported records, not guessed
  formulas or probabilities;
- at least one corpus-backed smoke/round-trip expectation must exist for every source format introduced.

### External Writers

Still needed:

- PRA-model writers for OpenPSA MEF and, later, SAPHIRE/KUREAS formats where round-trip semantics are meaningful;
- report/result writers that consume projected PRA-domain results plus the original model/projection context.

Implementation gate:

- implement only after the reader-side mapping table identifies which fields are round-trippable and which are
  extension-only;
- writers must reject raw PC handles when PRA domain data is required;
- writers must distinguish model serialization from result/report serialization;
- lossy export must be declared through diagnostics, not hidden by synthetic defaults.

### Probability And Expression Evaluation

Still needed:

- an evaluator contract for probability, frequency, rate, duration, distribution, and deviate expressions;
- evaluation context fields such as mission time, phase/alignment, units, parameter bindings, epistemic sample index, RNG
  stream identity, and external-function policy;
- lowering policy from PRA expressions to PC literal, parameter, or unsupported probability sources.

Implementation gate:

- implement only after expression domains, units, validation rules, random-variable identity, correlation semantics, and
  external-function policy are specified;
- solvers must declare which expression classes they can consume directly or through lowering;
- any expression that cannot be evaluated under a request must remain visible through diagnostics and unsupported PC/PRA
  placeholders.

### PRA-To-PC Lowering

Still needed:

- a compiler endpoint accepting `ir::pra::model` and producing a PC-bearing artifact plus `pc_projection`;
- Boolean lowering for fault-tree gates, formulas, transfers, house events, event-tree branch/path targets, and selected
  post-processing rules;
- target and event binding generation for probability, products, importance, and sequence/end-state projection.

Implementation gate:

- implement only after the supported formula/operator subset is enumerated and mapped to PC opcodes;
- define lowering options for house-event preservation, transfer flattening, generated target creation, unsupported
  probability handling, and CCF expansion policy;
- every requested domain target must either receive a binding to PC target(s) or a diagnostic explaining why it cannot be
  lowered;
- every transform that changes observable identity must update projection bindings or explicitly invalidate them through
  a transform report.

### Event-Tree Semantics And Projection

Still needed:

- precise branch-state semantics for success/failure/transfer/bypass/split states;
- conversion of event-tree paths and sequences into domain targets and generated PC formulas;
- aggregation from sequence/path targets to end states, hazards, initiators, systems, or consequence groups.

Implementation gate:

- implement only after SAPHIRE ETL logic-grid encodings and OpenPSA fork/path semantics are normalized to the same path
  model;
- define whether branch split fractions are probabilities, frequencies, weights, or conditional modifiers in each
  context;
- aggregation rules must state whether they are exact disjoint sums, rare-event approximations, weighted sums, or
  format-specific report roll-ups.

### Common-Cause Failure Support

Still needed:

- import of unexpanded beta-factor, alpha-factor, MGL, phi-factor, shock, and custom CCF declarations;
- CCF expansion transform producing generated events/nodes and original-to-generated/generated-to-original provenance;
- attribution policy for probability, product, and importance results involving generated common-shock events.

Implementation gate:

- implement only after parameterization semantics, member identity, adjusted independent probabilities, and generated
  event semantics are specified;
- expansion must be a compiler transform with a transform report, never a boolean flag on the model;
- result projection must define roll-up behavior to CCF groups, original member events, generated events, or combinations
  of those views.

### Post-Processing Rules

Still needed:

- semantics for substitutions, recovery/deletion/addition rules, alignments, phases, truncation/filter settings,
  cut-set-processing directives, and consequence collection rules;
- decision about which rules are model semantics, which are solver/compiler options, and which are report-only metadata.

Implementation gate:

- implement only after each rule kind has a declared subject, evaluation phase, ordering rule, and effect on targets,
  formulas, products, or projections;
- rules that alter Boolean/probability semantics must produce transform reports and provenance;
- report-only rules must remain attributes or extension records unless a writer explicitly consumes them.

### Imported Product/Cut-Set Caches

Still needed:

- representation and validation of imported fault-tree, sequence, end-state, and consequence cut-set decks;
- compatibility rules for when cached products may be trusted, recomputed, filtered, or rejected.

Implementation gate:

- implement only after the result/product identity model can distinguish solver-produced products from imported caches;
- cached products must carry source provenance, truncation/filter settings, and target identity;
- no solver may treat imported products as canonical truth unless its descriptor declares that behavior and validates the
  cache against the requested model/options.

### Result Projection

Still needed:

- pipeline or compiler-stage projection from PC-local solver records to PRA-domain result records;
- probability, product, importance, uncertainty, and roll-up result identities keyed by `domain_target_id` or
  `domain_ref`;
- diagnostics when solver outputs cannot be projected because bindings were missing or invalidated.

Implementation gate:

- implement only after solver results preserve PC target/event identity without lossy stringification;
- `pc_projection` must be carried alongside the PC artifact through pipeline planning and execution;
- projection must be deterministic for one-to-one bindings and explicitly declare aggregation semantics for many-to-one
  or one-to-many bindings.

### Domain-Specific Extension Growth

Still needed:

- a policy for promoting extension attributes into first-class `ir::pra` fields as new domains mature;
- compatibility tests proving promotion does not break existing readers, writers, or projection consumers.

Implementation gate:

- promote a field only when at least two endpoints or one solver/compiler workflow need typed access;
- maintain source provenance for promoted fields if external round-trip or diagnostics depend on it;
- avoid adding schema-specific columns for UI/report layout unless they affect PRA semantics or a stable writer contract.
