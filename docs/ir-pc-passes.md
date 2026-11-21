# `ir::pc` passes and compilation levels

`src/ir/pc/passes` rewrites an `ir::pc::circuit` into a new circuit and returns a transform report
([`IR-PC-REQUIREMENTS.md`](IR-PC-REQUIREMENTS.md) §12). The library is SCRAM-free (`fuzzybits_ir_pc_passes`, links only
`fuzzybits_ir_pc`) and is not yet used by `fzb compile` (#182).

```cpp
#include "ir/pc/passes/passes.h"
namespace passes = fuzzybits::ir::pc::passes;

auto one = passes::coalesce(circuit);                          // transform_result
auto all = passes::compile(circuit, passes::compile_level::c1, // pipeline_result
                           {.semantics = per_target_semantics});
```

## Passes

Every pass rebuilds the circuit from its targets in one iterative post-order sweep (no recursion, linear in nodes plus
edges). Nodes outside every target's cone are dropped, output node ids are topological, target ids and roles are kept,
and the edge count never increases.

| Pass | Class | Preserves | Rewrite |
|---|---|---|---|
| `fold_zero_probability` | distribution | Boolean, monotone hull | basic events with literal $p = 0$ become FALSE |
| `fold_unit_probability` | distribution | Boolean only | basic events with literal $p = 1$ become TRUE |
| `propagate_constants` | function | all | see below |
| `eliminate_single_input` | function | all | single-input gates become references |
| `coalesce` | function | all | edge-guarded inlining of same-kind children |

`function_preserving`: every target computes the same Boolean function. `distribution_preserving`: target probabilities
under independent sampling are unchanged for the listed semantics only. Under the hull, $p = 1$ events are droppable,
not constants ([`rfc/monotone-hull-queries.md`](rfc/monotone-hull-queries.md) Corollary 5.1, §7.5, §8.2). Parameter
and unsupported probability sources are never folded.

**`propagate_constants`.** Constants and house events propagate through every connective. AND/OR/NAND/NOR drop
identity operands and repeated operands and fold on an absorbing operand or on $x$ together with $\neg x$. XOR/XNOR cancel
operands pairwise ($x \oplus x = 0$, $x \oplus \neg x = 1$) and absorb constants into the parity, which is put on the first
remaining operand. Cardinality $[l, h]$ over $n$ operands, counted with multiplicity: FALSE operands are removed; $t$ TRUE
operands and each $x, \neg x$ pair (exactly one is true) shift the bounds to $[\max(l - t, 0),\ \min(h - t, n')]$; the
result is FALSE if $h < t$ or $l - t > n'$, TRUE if the bounds cover $[0, n']$, AND for $[n', n']$, OR for $[1, n']$, NOR
for $[0, 0]$, NAND for $[0, n' - 1]$, and otherwise a cardinality gate. The pass simplifies first and emits second, so
nodes orphaned by a later fold are not emitted.

**`eliminate_single_input`.** Single-input AND, OR, XOR and cardinality $[1, 1]$ become their operand; NOT, NAND, NOR,
XNOR and cardinality $[0, 0]$ become its negation. The negation moves onto the consuming edges; a target whose image is
complemented gets one NOT node.

**`coalesce`.** A child AND (OR, XOR) is inlined into its parents only if every reference to it is an uncomplemented
operand of an AND or NAND (OR or NOR, XOR or XNOR) parent and no target names it. With $m$ references and $k$ operands,
inlining into all parents and deleting the child changes the edge count by $m(k - 1) - k = (m - 1)(k - 1) - 1$, so the
child is inlined iff $(m - 1)(k - 1) \le 1$: always for a single parent, never for a shared child with three or more
operands. Children are decided bottom-up, so $k$ includes what the child itself absorbed. Partial inlining (into some
parents only) always adds edges and is not done. This is the guard against the edge blow-up measured in #147.

## Reports

`transform_report`: `transform_id` (`ir.pc.<pass>`), `version`, `semantics`, `preserved_semantics`, `node_map` (by input
node id: `removed`, `absorbed`, `constant` with its value, or `reference` to an output node with a complement flag),
`target_map` (input target id to output target id), node and edge counts before and after, `rewrites` (events folded,
gates simplified or eliminated, children absorbed), `scratch_bytes` (transform-local memory) and `diagnostics`.

`pipeline_result` composes the maps across passes and keeps the per-pass reports; per-pass maps are dropped unless
`pipeline_options::keep_pass_maps` is set.

## Levels

- **C0:** no rewriting.
- **C1:** `fold_zero_probability`; `fold_unit_probability` if every target is boolean; then rounds of
  `propagate_constants`, `eliminate_single_input`, `coalesce` until a round changes neither the node nor the edge count
  (at most `max_rounds`, default 4).

Per-target semantics (`pipeline_options::semantics`, by target id; empty means all boolean) come from the lowering
(#181), which records them because `ir::pc` targets do not. `run_pipeline` refuses a schedule containing a pass that does
not preserve some target's semantics; C1 skips `fold_unit_probability` instead.

## Diagnostics

| Code | Severity | Meaning |
|---|---|---|
| `pc.pass.invalid-input` | error | the input circuit is not sealed |
| `pc.pass.no-targets` | warning | no targets; the circuit is returned unchanged |
| `pc.pass.seal-failed` | error | internal: the rewritten circuit failed validation |
| `pc.pipeline.invalid-input` | error | the input circuit is not sealed |
| `pc.pipeline.target-semantics-mismatch` | error | the semantics table is neither empty nor one entry per target |
| `pc.pipeline.incompatible-pass` | error | a scheduled pass does not preserve some target's semantics |
| `pc.pipeline.pass-skipped` | note | C1 skipped `fold_unit_probability` for monotone-hull targets |
