#include "ir/pc/passes/detail/rewrite.h"
#include "ir/pc/passes/passes.h"

#include <optional>
#include <utility>

namespace fuzzybits::ir::pc::passes {

namespace {

// Whether a single-input node is its operand (false), its operand's
// negation (true), or neither (nullopt).
[[nodiscard]] auto single_input_negation(circuit const &input, const std::uint32_t node) -> std::optional<bool> {
    if (input.operand_counts()[node] != 1) {
        return std::nullopt;
    }
    switch (input.op_codes()[node]) {
    case op_code::and_gate:
    case op_code::or_gate:
    case op_code::xor_gate:
        return false;
    case op_code::not_gate:
    case op_code::nand_gate:
    case op_code::nor_gate:
    case op_code::xnor_gate:
        return true;
    case op_code::cardinality: {
        const auto low = input.cardinality_low()[node];
        const auto high = input.cardinality_high()[node];
        if (low == 1 && high == 1) {
            return false;
        }
        if (low == 0 && high == 0) {
            return true;
        }
        return std::nullopt;
    }
    default:
        return std::nullopt;
    }
}

} // namespace

auto eliminate_single_input(circuit const &input, pass_options const &options) -> transform_result {
    auto report = detail::begin_report(input, pass_id::eliminate_single_input);
    if (auto trivial = detail::trivial_result(input, report)) {
        return std::move(*trivial);
    }
    const auto split = detail::split_context::make(input, options.atoms, report);
    if (!split) {
        return transform_result{.output = {}, .report = std::move(report)};
    }

    const auto order = detail::cone_order(input);
    const auto codes = input.op_codes();
    std::vector<std::uint8_t> complemented_reference;
    if (split->active()) {
        complemented_reference.assign(input.node_count(), 0);
        for (const auto node : order) {
            for (auto const &operand : input.operands(bool_node_id(node))) {
                complemented_reference[operand.source.raw()] |= operand.complement ? 1U : 0U;
            }
        }
    }
    detail::emitter emitter(input);
    auto &map = report.node_map;
    std::vector<edge_ref> operands;

    for (const auto node : order) {
        const auto code = codes[node];
        if (is_leaf(code)) {
            const auto output = emitter.leaf(node);
            if (split->member(node)) {
                emitter.mark_atom(output);
            }
            map[node] = detail::reference(output);
            continue;
        }
        if (const auto negation = single_input_negation(input, node)) {
            auto const &[source, complement] = input.operands(bool_node_id(node))[0];
            const auto image = detail::through(map[source.raw()], complement != *negation);
            // Over atoms an atom event v, its atom (complemented edge) and
            // their negations are four functions, but only v and the atom
            // are edges. The gate may become one of those two only if it
            // is one, and only if no edge negates it.
            bool keep = false;
            if (split->in_cone(node)) {
                if (split->member(source.raw())) {
                    keep = *negation || complemented_reference[node] != 0;
                } else if (image.kind == image_kind::reference && emitter.is_atom(image.edge.source)) {
                    keep = complement != *negation || complemented_reference[node] != 0;
                }
            }
            if (!keep) {
                map[node] = image;
                ++report.rewrites;
                continue;
            }
            ++report.skipped;
        }
        detail::remap_operands(input, node, map, operands);
        map[node] = detail::reference(
            emitter.gate(code, operands, input.cardinality_low()[node], input.cardinality_high()[node]));
    }

    report.scratch_bytes = detail::bytes_of(order) + detail::bytes_of(operands) +
                           detail::bytes_of(complemented_reference) + split->bytes();
    return std::move(emitter).finish(std::move(report));
}

} // namespace fuzzybits::ir::pc::passes
