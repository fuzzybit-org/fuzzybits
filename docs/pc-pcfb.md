# ir::pc → pcfb

`src/io/pcfb` writes a sealed `ir::pc::circuit` as a pcfb file (opset `core-1.1`) through
the pinned `pcfb::builder`. It links no SCRAM code. The reverse direction
(`.pcfb` → `ir::pc`) belongs to
[#146](https://github.com/fuzzybit-org/fuzzybits/issues/146).

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
artifact: a target's negation set gives its kinds, and a monotone-hull target whose cone has marked
literal events is refused with `compile.pra_to_pc.hull-literals-unsupported` until pcfb carries
literal atoms (RFC 7.3, #197).

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
