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

    auto eliminate_single_input(circuit const &input) -> transform_result {
        auto report = detail::begin_report(input, pass_id::eliminate_single_input);
        if (auto trivial = detail::trivial_result(input, report)) {
            return std::move(*trivial);
        }

        const auto order = detail::cone_order(input);
        const auto codes = input.op_codes();
        detail::emitter emitter(input);
        auto &map = report.node_map;
        std::vector<edge_ref> operands;

        for (const auto node : order) {
            const auto code = codes[node];
            if (is_leaf(code)) {
                map[node] = detail::reference(emitter.leaf(node));
                continue;
            }
            if (const auto negation = single_input_negation(input, node)) {
                auto const &operand = input.operands(bool_node_id(node))[0];
                map[node] = detail::through(map[operand.source.raw()], operand.complement != *negation);
                ++report.rewrites;
                continue;
            }
            detail::remap_operands(input, node, map, operands);
            map[node] = detail::reference(emitter.gate(code, operands, input.cardinality_low()[node], input.cardinality_high()[node]));
        }

        report.scratch_bytes = detail::bytes_of(order) + detail::bytes_of(operands);
        return std::move(emitter).finish(std::move(report));
    }

} // namespace fuzzybits::ir::pc::passes
