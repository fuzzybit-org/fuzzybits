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
| `cardinality(low, high)` over `n` operands, `[0, n]` | `BITWISE_ONES` |
| … `high = n` | `ATLEAST low` |
| … `low = 0` | `ATMOST high` |
| … `low = high` | `EXACTLY low` |
| … otherwise | `AND(ATLEAST low, ATMOST high)`, two unnamed tensors before the `AND` |
| target | `Query` on the node's tensor, in target order; one per kind of `write_options::target_kinds`, else one of kind `write_options::query` (default `PROBABILITY`); see Query kinds |
| `observable_role` | not representable; non-primary roles give the note `io.pcfb.target-role` |
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
| `compile.query.<i>.negation` | `boolean` for a `probability` query, `monotone_hull` for a hull query |
| `compile.query.<i>.literal_mode` | `target_quantification::literal_mode` (`atom_split`, `omega_l`, `negation`), if set |
| `compile.query.<i>.max_initiators_per_cutset` | int64, the initiator rule already lowered into the target, if set |

`fzb inspect` lists them with the other metadata. Targets without an entry record nothing.
`compile::backend::fuzzybits::pra_to_pc::pcfb_write_options` fills all three maps from the lowering
artifact: a target's negation set gives its kinds, and an `atom_split` or `omega_l` monotone-hull
target whose cone has marked literal events is refused with
`compile.pra_to_pc.hull-literals-unsupported` until pcfb carries literal atoms (RFC 7.3, #197). In
`negation` mode marked operands are ordinary complements, the plain hull a hull query means, so the
target is written.

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
`pcfb::validate` first (its findings are `io.pcfb.invalid_file`, an unknown query kind is
`io.pcfb.unknown_query_kind`), then requires one subcircuit without external inputs and at least
one query. Any minor of schema major 1 is accepted; `MONOTONE_HULL_PROBABILITY` needs minor 2, as
in `pcfb::validate`.

### Mapping

| pcfb | `ir::pc` |
|---|---|
| `SAMPLE` | basic event with the literal `probability`; `event_id`, `name` kept in `options` / `tensor_by_node` |
| `ZEROS` / `ONES` | constant 0 / 1 |
| `AND` `OR` `XOR` `NAND` `NOR` `XNOR` `NOT` | the gate of the same name; `inverted_inputs` become complemented edges |
| `ATLEAST` / `ATMOST` / `EXACTLY` (`k`) | cardinality `[k, n]` / `[0, k]` / `[k, k]` |
| `AND(ATLEAST(k), ATMOST(m))` over the same operands | one cardinality `[k, m]`; the two unnamed tensors have no node |
| unnamed `NOT` that only feeds operators listing its source positively | complemented edge (the writer's tensor for a source used in both polarities) |
| `NO_OP` | one-operand AND (`io.pcfb.no_op_as_and`) |
| `Query` | target on the tensor's node; consecutive queries with one tensor and one name and distinct kinds are one target with several kinds |

Per-query `compile.query.<i>.*` metadata fills `options.quantification`; `general.*` and
`compile.options` fill the provenance fields. Node and target names are the file's (a basic event
without `SampleArgs.name` has none, so a rewrite generates `e<event_id>` again).

### Numbering and what a round trip preserves

The writer emits operators by depth-first post-order, taking roots in ascending node id and
operands in operand order, and numbers basic events by ascending node id. The reader chooses node
ids so that this traversal reproduces the file's operator order: operators are grouped into the
trees the traversal would have visited, roots take ascending ids, and events take ascending ids
in event-id order as early as their trees allow. `write(read(bytes))` is therefore byte-identical
for files the writer produced, with one exception: `cardinality(0, n)` is a `ONES` tensor, but the
writer emits the operands' subtrees before it. The read circuit has no edges from the constant to
those operands, so their order may be forced differently by the ascending event ids. The rewrite
then lists the same operators in another order (`io.pcfb.emission_order_changed`, a note), and is
a fixed point from there on. Operand order among the positive and negative operands of one
operator is not recorded in the file; the reader interleaves the two lists by the order in which
the traversal first reaches their trees.

Not recorded by the writer and therefore not recovered: target roles (every read target is
primary; the writer only emits the note `io.pcfb.target-role`), house events (they were written as
constants), the operands of a full-range cardinality, and `ir::pc` node ids as such.

### Diagnostics

| Id | Severity | Meaning |
|---|---|---|
| `io.pcfb.unreadable` | error | not a verifiable FlatBuffer, or the file cannot be read |
| `io.pcfb.invalid_file` | error | `pcfb::validate` finding other than an unknown query kind |
| `io.pcfb.unknown_query_kind` | error | query kind not defined by the pinned `pcfb`, or newer than the file's minor |
| `io.pcfb.external_inputs` | error | the subcircuit has externally supplied inputs |
| `io.pcfb.unsupported_opcode` | error | an operator `ir::pc` has no op code for, or one reserved by `core-1.1` |
| `io.pcfb.too_large` | error | more tensors than a 32-bit node id addresses |
| `io.pcfb.format_mismatch` | error | the read request names a format other than pcfb |
| `io.pcfb.invalid_circuit` | error | the rebuilt circuit failed `ir::pc` sealing |
| `io.pcfb.sample_source_dropped` | warning | a `SAMPLE` source of `COLLAPSED_MODULE` or `HOUSE_EVENT` reads as a basic event |
| `io.pcfb.confidence_ignored` | warning | `Query.confidence` cannot be held |
| `io.pcfb.event_ids_renumbered` | warning | event ids are not exactly `1..B`; the read circuit numbers them by ascending file id |
| `io.pcfb.invalid_metadata` | warning | an interpreted metadata entry has the wrong type or range; it is ignored |
| `io.pcfb.quantification_conflict` | warning | queries on one target carry different `compile.query.<i>.*` values |
| `io.pcfb.no_op_as_and` | note | `NO_OP` read as a one-operand AND |
| `io.pcfb.emission_order_changed` | note | a write lists some operators in another order than the file |

The constants are declared in `descriptor.h` (`writer::diag`).

### Tests

`read_tests.cc` round-trips FTP conformance fixtures and several hundred seeded random circuits,
checks structure through `node_by_tensor`, runs every tracked `.pcfb` file through read → write →
read, and covers truncated and corrupted input, foreign constructs and unknown query kinds.
`read_scale_tests.cc` (`[scale]`, separate executable) reads a 10^5-node circuit and requires it
to take less than a second in Release.
