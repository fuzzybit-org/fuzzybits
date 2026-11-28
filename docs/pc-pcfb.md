# ir::pc ↔ pcfb

`src/io/pcfb` writes a sealed `ir::pc::circuit` as a pcfb file (opset `core-1.1`) through
the pinned `pcfb::builder`, and reads a pcfb file back into an `ir::pc::circuit`
(see [Reading](#reading)). It links no SCRAM code.

Related: [#176](https://github.com/fuzzybit-org/fuzzybits/issues/176),
[#182](https://github.com/fuzzybit-org/fuzzybits/issues/182).

## API

```cpp
#include "io/pcfb/descriptor.h"

namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;

writer::write_options options;
options.model_name = "demo";
options.source_format = "cafta-ftp";
options.node_names[pump] = "PUMP-A";      // ir::pc::bool_node_id → name
options.target_names[top] = "TOP";        // ir::pc::target_id → name

auto written = writer::write(circuit.to_handle(), options);
// written.value().bytes, .tensor_by_node, .event_id_by_node
```

`write(handle, std::ostream &, io::write_request)` is the generic IO endpoint.
It reads the scalar options from the `io.pcfb.*` keys in the writer descriptor;
symbol maps are only available through `write_options`.

## Mapping

| `ir::pc` | pcfb |
|---|---|
| `basic_event`, literal `p` | `SAMPLE` with `probability = p`, `threshold = floor(p·2^64)` and `always_true` iff `p = 1` (both from `pcfb::builder`), `event_id` below, `source = BASIC_EVENT` |
| `basic_event`, parameter or unsupported source | error (`io.pcfb.parameter-probability`, `io.pcfb.unsupported-probability`); no output |
| `constant`, `house_event` | `BITWISE_ONES` / `BITWISE_ZEROS`; a house event keeps its identity only through a name |
| `and`, `or`, `xor`, `nand`, `nor`, `xnor`, `not` | `BITWISE_AND`, `_OR`, `_XOR`, `_NAND`, `_NOR`, `_XNOR`, `_NOT` |
| complemented `edge_ref` | `inverted_inputs` |
| one source in both polarities within a gate | the complemented occurrences read one shared `NOT` tensor per source, so `inputs` and `inverted_inputs` are disjoint in every op (spec §3.1) |
| duplicate operands | kept; XOR and k-of-n count multiplicity, as in `ir::pc` |
| `cardinality(low, high)` over no operands | `BITWISE_ONES` |
| … `high = n` (so `[0, n]` is `ATLEAST 0`, valid for `0 ≤ k ≤ n`) | `ATLEAST low` |
| … `low = 0` | `ATMOST high` |
| … `low = high` | `EXACTLY low` |
| … otherwise | `AND(ATLEAST low, ATMOST high)`, two unnamed tensors before the `AND` |
| target | `Query` on the node's tensor, in target order; one per kind of `write_options::target_kinds`, else one of kind `write_options::query` (default `PROBABILITY`); see Query kinds |
| `observable_role` | primary: nothing; `watch` / `projection_anchor`: `compile.query.<i>.role` on each query of the target (see Metadata) |
| nodes no target reaches | written |

Cases are tested in that order, so `[n, n]` is `ATLEAST n` and `[0, 0]` is
`ATMOST 0`. `ir::pc` guarantees `0 ≤ low ≤ high ≤ n`, so no range is empty.

## Query kinds

A query is `probability` (`PROBABILITY`) or `monotone_hull_probability`
(`MONOTONE_HULL_PROBABILITY`, pcfb 1.2; semantics in
[`rfc/monotone-hull-queries.md`](rfc/monotone-hull-queries.md)). `write_options::target_kinds`
gives the kinds of a target; targets without an entry get `write_options::query`, which the
`io.pcfb.query_kind` option sets for the stream writer. Several kinds on one target make several
queries on one tensor, in the listed order, all named like the target; the kind tells them apart.
An empty or repeating list, or an entry for a target that is not in the circuit, is
`io.pcfb.invalid-query-kind`.

`schema_version` is the lowest minor that covers the kinds written: `0x00010001` for a
file with only `probability` queries (byte-identical to what a 1.1 writer produced), `0x00010002` as
soon as one query is a hull query. The opset stays `core-1.1`. Runtimes that do not implement a
kind reject the file before sampling (`pcfb.unsupported-query-kind`).

`write_options::quantification` records, per target, what the caller asked for. For the query at
position `i` of the file the writer sets

| Key | Value |
|---|---|
| `compile.query.<i>.target` | `target_quantification::target`: the domain target or element the query observes, if set; a conditioned query's name adds `\|<initiator>`, this does not |
| `compile.query.<i>.literal_mode` | `target_quantification::literal_mode` (`split`, `fixed`, `plain`), if set; on hull queries only, so a probability query on the same tensor has none. The reader takes it from the hull query and ignores a probability query's copy |
| `compile.query.<i>.max_initiators_per_cutset` | int64, the initiator rule already lowered into the target, if set; on hull queries only, and the FTP compile sets it only where the target holds the delete term |
| `compile.query.<i>.initiator`, `.initiator_frequency` | string and float64: the initiating event a conditioned query sets TRUE, and its frequency (the query's weight in its group), if set |
| `compile.complement_tensors` | bytes: little-endian `uint32` tensor ids in ascending order, the unnamed `NOT` tensors the writer created for sources used in both polarities; only written when there is at least one |
| `compile.query.<i>.role` | string `watch` or `projection_anchor`, only for targets that are not primary; files with only primary targets have no such entry |

`fzb inspect` lists them with the other metadata. Targets without an entry record nothing.
`compile::backend::fuzzybits::pra_to_pc::pcfb_write_options` fills all three maps from the lowering
artifact: one target per binding, of the binding's kind (the lowering makes one binding per query,
and two bindings share a node unless the hull query holds the initiator rule), and a `split` or
`fixed` hull query whose cone has marked literal events is refused with
`compile.pra_to_pc.hull-literals-unsupported` until the writer fills pcfb's literal fields (RFC
7.3, #235). Read `plain`, marked operands are ordinary complements, the plain hull a hull query
means, so the query is written. The kind of a query is `Query.kind`; no metadata key repeats it.

## Query groups

`write_options::query_groups` writes pcfb `SubCircuit.query_groups` (pcfb spec §5.4, a v1.2
amendment): each group has a name, a unit and terms `(target, weight)` for queries of one `kind`; the
writer turns each `(target, kind)` into the index of that query. A group with no terms, a target
that is not in the circuit or has no query of the group's kind, a target listed twice, or a negative
or non-finite weight is `io.pcfb.invalid-query-group`. A file with groups declares minor 2; a file
without them is unchanged. The groups add a quantity and change no query.

`pcfb_write_options` writes one group per domain target and kind under
`initiator_values::frequency` (#243): its conditioned targets, weighted by their initiators'
frequencies, in the lowering's `initiator_unit`. `io/pcfb/query_groups.h` reads a file's groups, and
`io/pcfb/frequency.h` holds the units and the `--exposure` conversion. `fzb run`, `fzb inspect --exact`
and Python's `run(...).groups` / `exact_groups` evaluate them; nombrilo export carries them in
`fuzzybits.pcfb.query_groups` (dropped by `--strict`).

## Order

Operators are written in depth-first post-order: roots in ascending node id,
operands in operand order, every node after all of its operands. A circuit whose
node ids are already topological is written in id order, so without auxiliary
tensors (`NOT` for mixed polarity, the two k-of-n ops of a general range) tensor
index equals node id. The traversal is iterative and diagnoses cycles
(`io.pcfb.cycle`, with the node path) and invalid operand ids
(`io.pcfb.invalid-edge`). `write_output::tensor_by_node` gives the tensor of each
node.

## `event_id`

`event_id` selects the counter-based PRNG stream of a basic event, so it is part
of the file contract:

> basic-event node `v` gets `event_id = 1 + |{u < v : u is a basic event}|`.

Event ids are `1..B` in ascending node id, unique within the subcircuit, and do
not depend on the emission order. For topologically numbered circuits, which is
what the FTP and AIGER readers produce, this is also the order of first
appearance in the file. Two basic-event nodes that share a probability source
are independent events with distinct ids. `write_output::event_id_by_node`
returns the assignment (0 for other nodes; pcfb runtimes reserve 0).

## Names

`ir::pc` has no names. `write_options::node_names` sets `SampleArgs.name` for
basic events and `Op.name` for any other node. `write_options::target_names` sets
`Query.name`. Missing names are generated: `e<event_id>` for basic events and
`t<target id>` for queries. Other nodes and auxiliary tensors stay unnamed. A
name for a node or target id that is not in the circuit is an error
(`io.pcfb.invalid-name`).

## Metadata

| Key | Value |
|---|---|
| `general.generator` | `fuzzybits` |
| `general.generator_version` | project version, plus `+g<git describe>` when known (as for nombrilo) |
| `general.endianness`, `pcfb.opset` | set by `pcfb::builder` (`little`, `core-1.1`) |
| `general.model_name` | `write_options::model_name`, also `Circuit.name` |
| `general.source_format` | `write_options::source_format` |
| `general.source_sha256` | `write_options::source_sha256` (32 bytes) |
| `compile.options` | `write_options::compile_options` |

Empty options are omitted.

## Checks

Before returning, the writer reads its own bytes back with
`pcfb::io::from_bytes` and runs `pcfb::validate`; any finding is an
`io.pcfb.invalid-output` error. `src/io/pcfb/tests` checks, for random circuits
with up to 16 basic events, that `compile::exact_probability` of the written
file equals exhaustive evaluation of the `ir::pc` circuit to 1e-12, runs the
FTP fixtures through `fzb inspect --validate` and `--exact`, and checks a
written circuit with `fzb run` (label `mc`) against its exact value.

`pcfb::builder::finish()` at the pinned revision names tensors in time quadratic
in the operator count (about 25 s at 10^5 operators); the writer itself is
linear.

## Reading

```cpp
auto read = writer::read(bytes);          // std::span<std::byte const>, or a std::filesystem::path
if (!read) { /* read.diagnostics() holds the errors */ }
read->circuit;                            // ir::pc::circuit
read->options;                            // write_options: names, kinds, quantification, provenance
read->node_by_tensor;                     // tensor -> ir::pc node
read->metadata;                           // every metadata entry, verbatim
read->schema_version;
```

`reader_endpoint_id()` is `io.fuzzybits.pcfb.reader`; `read(io::read_request const&)`
returns the `ir::pc` handle (`ir.probabilistic_boolean_dag`). The reader runs
`pcfb::validate` first (its findings are `io.pcfb.invalid-file`, an unknown query kind is
`io.pcfb.unknown-query-kind`, an opcode the pinned `pcfb` does not define is
`io.pcfb.unsupported-opcode`), then requires one subcircuit without external inputs and at least
one query. Any minor of schema major 1 is accepted; `MONOTONE_HULL_PROBABILITY`, `MAJORITY` and
`MINORITY` need minor 2, as in `pcfb::validate`.

### Mapping

| pcfb | `ir::pc` |
|---|---|
| `SAMPLE` | basic event with the literal `probability`; `event_id`, `name` kept in `options` / `tensor_by_node` |
| `ZEROS` / `ONES` | constant 0 / 1 |
| `AND` `OR` `XOR` `NAND` `NOR` `XNOR` `NOT` | the gate of the same name; `inverted_inputs` become complemented edges |
| `ATLEAST` / `ATMOST` / `EXACTLY` (`k`) | cardinality `[k, n]` / `[0, k]` / `[k, k]` |
| `MAJORITY` / `MINORITY` (v1.2 opcode amendment) | cardinality `[⌊n/2⌋ + 1, n]` / `[0, ⌊n/2⌋]` (`pcfb::count_bounds`; a tie is a minority) |
| `AND(ATLEAST(k), ATMOST(m))` over the same operands | one cardinality `[k, m]`; the two unnamed tensors have no node |
| `NOT` tensor listed in `compile.complement_tensors` | complemented edge in the position of the tensor in `inputs`; no node |
| every other `NOT` | `not_gate` node (this includes all NOTs of files without the key, such as other writers' files) |
| `NO_OP` | one-operand AND (`io.pcfb.no-op-as-and`) |
| `Query` | target on the tensor's node; consecutive queries with one tensor and one name and distinct kinds are one target with several kinds |

Per-query `compile.query.<i>.*` metadata fills `options.quantification`, and `SubCircuit.query_groups`
fills `options.query_groups` (term queries mapped back to their targets and kinds); `general.*` and
`compile.options` fill the provenance fields. Node and target names are the file's (a basic event
without `SampleArgs.name` has none, so a rewrite generates `e<event_id>` again).

### Numbering and what a round trip preserves

Node numbering and operand order in `ir::pc` are layout: every gate is symmetric in its operands,
so neither carries meaning. The reader therefore uses one canonical numbering and does not try to
recover the numbering a writer had:

- basic events come first, by ascending file event id (`io.pcfb.event-ids-renumbered` if the ids
  are not exactly `1..B`);
- every other node follows in file operator order. A pcfb operator reads only tensors defined
  before it, so this order is topological; the reader rejects a file in which an operator reads a
  tensor written later (`io.pcfb.invalid-file`).

A node's operands are its `inputs` in file order followed by its `inverted_inputs` in file order.
A NOT tensor folded into a complemented edge takes the position of that tensor in `inputs`.
Folding is exact and never inferred: only the tensors listed in `compile.complement_tensors` fold, so a
file from another writer reads every `NOT` as a node.

Rewriting a read file therefore gives an equivalent file, not an identical one: the same
structure, probabilities, names, kinds, roles and event ids, with operators in the writer's own
emission order. Reading that file and rewriting it again reproduces it.

Per-query `compile.query.<i>.role` restores the target role (absent: primary). Not recorded by the
writer and therefore not recovered: house events (they were written as constants) and `ir::pc`
node ids as such.

### Diagnostics

| Id | Severity | Meaning |
|---|---|---|
| `io.pcfb.unreadable` | error | not a verifiable FlatBuffer, or the file cannot be read |
| `io.pcfb.invalid-file` | error | `pcfb::validate` finding other than an unknown query kind or opcode, such as a reserved opcode or a `MAJORITY` or `MINORITY` operator in a file that declares minor 1, or an operator that reads a tensor a later operator writes |
| `io.pcfb.unknown-query-kind` | error | query kind not defined by the pinned `pcfb`, or newer than the file's minor |
| `io.pcfb.external-inputs` | error | the subcircuit has externally supplied inputs |
| `io.pcfb.unsupported-opcode` | error | an operator `ir::pc` has no op code for, such as an opcode the pinned `pcfb` does not define |
| `io.pcfb.too-large` | error | more tensors than a 32-bit node id addresses |
| `io.pcfb.format-mismatch` | error | the read request names a format other than pcfb |
| `io.pcfb.invalid-circuit` | error | the rebuilt circuit failed `ir::pc` sealing |
| `io.pcfb.sample-source-dropped` | warning | a `SAMPLE` source of `COLLAPSED_MODULE` or `HOUSE_EVENT` reads as a basic event |
| `io.pcfb.confidence-ignored` | warning | `Query.confidence` cannot be held |
| `io.pcfb.event-ids-renumbered` | warning | event ids are not exactly `1..B`; the read circuit numbers them by ascending file id |
| `io.pcfb.invalid-metadata` | warning | an interpreted metadata entry has the wrong type or range, an unknown role, or a malformed or unusable `compile.complement_tensors` entry; it is ignored (a target reads as primary, a listed tensor that is not a NOT tensor the writer could have created reads as a NOT gate) |
| `io.pcfb.quantification-conflict` | warning | queries on one target carry different `compile.query.<i>.*` values |
| `io.pcfb.no-op-as-and` | note | `NO_OP` read as a one-operand AND |

The constants are declared in `descriptor.h` (`writer::diag`).

### Tests

`read_tests.cc` round-trips FTP conformance fixtures and several hundred seeded random circuits,
checks structure through `node_by_tensor` and that a rewrite reads back to the same structure,
runs every tracked `.pcfb` file through read → write → read, and covers truncated and corrupted input, foreign constructs and unknown query kinds.
`read_scale_tests.cc` (`[scale]`, separate executable) reads a 10^5-node circuit and requires it
to take less than a second in Release.
