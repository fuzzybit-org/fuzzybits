#pragma once

#include "api/diagnostics.h"
#include "ir/pc/circuit.h"
#include "ir/pc/id.h"
#include "ir/pc/passes/transform.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace fuzzybits::ir::pc::passes {

    // ir::pc -> ir::pc rewrites (docs/ir-pc-passes.md). Every pass rebuilds the
    // circuit from its targets: nodes outside every target's cone are dropped,
    // output node ids are topological, target ids and roles are kept, and the
    // number of edges never increases. Iterative and linear in nodes + edges.

    // Monotone-hull targets in atom-split mode (#190). In their cones a
    // complemented edge into one of `events` is its own droppable atom (sampled
    // as not x_v), while gate negations stay delete terms, so the hull depends
    // on the circuit over atoms. Passes then preserve the function over atoms:
    // they never create or merge such edges, never pair an atom with its event,
    // and never fold an atom event; each rewrite left out for this reason is
    // counted in transform_report::skipped. One event set serves all `targets`.
    struct atom_split final {
        std::vector<target_id> targets;
        std::vector<bool_node_id> events;
    };

    struct pass_options final {
        atom_split atoms;
    };

    // Literal p = 0 basic events become FALSE. Distribution-preserving for
    // boolean and monotone_hull targets (RFC Corollary 5.1). Atom events with an
    // atom edge are kept: their atom has p = 1 and stays droppable.
    [[nodiscard]] auto fold_zero_probability(circuit const &input, pass_options const &options = {}) -> transform_result;

    // Literal p = 1 basic events become TRUE. Distribution-preserving for
    // boolean targets only: under every hull mode p = 1 events are droppable,
    // not constants (RFC Corollary 5.1, §8.2).
    [[nodiscard]] auto fold_unit_probability(circuit const &input, pass_options const &options = {}) -> transform_result;

    // Constants and house events propagate through every connective;
    // AND/OR drop duplicate operands and fold x with not x; XOR/XNOR cancel
    // operand pairs by parity; cardinality removes constants and x / not x
    // pairs by shifting its bounds and becomes AND, OR, NAND or NOR when the
    // bounds allow. An atom edge counts as an operand of its own. Function-
    // preserving, over atoms in atom-split cones.
    [[nodiscard]] auto propagate_constants(circuit const &input, pass_options const &options = {}) -> transform_result;

    // Single-input AND, OR, XOR, NAND, NOR, XNOR, NOT and cardinality [1, 1] or
    // [0, 0] become references; the negation moves onto the consuming edges.
    // A target whose image is complemented gets a NOT node. In atom-split cones
    // a gate whose image would be an atom event is eliminated only if it is
    // that event or that atom, unnegated, and no edge into the gate is
    // complemented. Function-preserving, over atoms in atom-split cones.
    [[nodiscard]] auto eliminate_single_input(circuit const &input, pass_options const &options = {}) -> transform_result;

    // Inlines a child AND (OR, XOR) into its parents when every parent is
    // AND or NAND (OR or NOR, XOR or XNOR), every reference is uncomplemented,
    // the child is not a target, and the edge count does not increase: with m
    // references and k operands, (m - 1)(k - 1) <= 1, which always holds for a
    // single-parent child. Moves edges verbatim, so atoms are kept.
    // Function-preserving.
    [[nodiscard]] auto coalesce(circuit const &input, pass_options const &options = {}) -> transform_result;

    enum class pass_id : std::uint8_t {
        fold_zero_probability,
        fold_unit_probability,
        propagate_constants,
        eliminate_single_input,
        coalesce,
    };

    [[nodiscard]] auto pass_name(pass_id pass) noexcept -> std::string_view;
    [[nodiscard]] auto pass_semantics(pass_id pass) noexcept -> semantic_class;
    [[nodiscard]] auto pass_preserves(pass_id pass, target_semantics kind) noexcept -> bool;
    [[nodiscard]] auto run_pass(pass_id pass, circuit const &input, pass_options const &options = {}) -> transform_result;

    // C0: no rewriting. C1: fold_zero_probability, fold_unit_probability when
    // every target is boolean, then rounds of propagate_constants,
    // eliminate_single_input and coalesce until a round changes neither the node
    // nor the edge count (at most pipeline_options::max_rounds rounds).
    enum class compile_level : std::uint8_t {
        c0,
        c1,
    };

    struct pipeline_options final {
        // By target id. Empty means every target is boolean; otherwise one entry
        // per target.
        std::vector<target_semantics> semantics;
        // By target id: the atom events of each monotone_hull target in
        // atom-split mode (the lowering's complemented_events, #181). Empty
        // means none; otherwise one entry per target. Entries of boolean
        // targets are ignored.
        std::vector<std::vector<bool_node_id>> atom_events;
        std::uint32_t max_rounds = 4;
        // Per-pass reports keep their node and target maps only if set; the
        // composed maps are always kept.
        bool keep_pass_maps = false;
    };

    struct pipeline_result final {
        circuit output;
        std::vector<transform_report> passes;
        std::vector<node_image> node_map;
        std::vector<target_id> target_map;
        std::uint32_t nodes_before = 0;
        std::uint32_t edges_before = 0;
        std::uint32_t nodes_after = 0;
        std::uint32_t edges_after = 0;
        api::diagnostic_list diagnostics;

        [[nodiscard]] auto ok() const -> bool {
            return output.valid() && !diagnostics.has_errors();
        }
    };

    // Runs `schedule` in order. Refuses (error pc.pipeline.incompatible-pass, no
    // output) if a distribution-preserving pass does not preserve the semantics
    // of some target. Atom events are carried through each pass's node map.
    [[nodiscard]] auto run_pipeline(circuit const &input, std::span<pass_id const> schedule, pipeline_options const &options = {})
        -> pipeline_result;

    [[nodiscard]] auto compile(circuit const &input, compile_level level, pipeline_options const &options = {}) -> pipeline_result;

} // namespace fuzzybits::ir::pc::passes
