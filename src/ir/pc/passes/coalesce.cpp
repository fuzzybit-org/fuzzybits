#include "ir/pc/passes/detail/rewrite.h"
#include "ir/pc/passes/passes.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pc::passes {

namespace {

// The child kind a parent can absorb without changing its function.
[[nodiscard]] constexpr auto absorbable_kind(const op_code parent) noexcept -> std::optional<op_code> {
    switch (parent) {
    case op_code::and_gate:
    case op_code::nand_gate:
        return op_code::and_gate;
    case op_code::or_gate:
    case op_code::nor_gate:
        return op_code::or_gate;
    case op_code::xor_gate:
    case op_code::xnor_gate:
        return op_code::xor_gate;
    default:
        return std::nullopt;
    }
}

} // namespace

auto coalesce(circuit const &input, pass_options const &) -> transform_result {
    auto report = detail::begin_report(input, pass_id::coalesce);
    if (auto trivial = detail::trivial_result(input, report)) {
        return std::move(*trivial);
    }

    const auto order = detail::cone_order(input);
    const auto codes = input.op_codes();
    const auto node_count = input.node_count();

    // A child is a candidate if every reference to it is an uncomplemented
    // operand of a parent that absorbs its kind, and no target names it.
    std::vector<std::uint32_t> references(node_count, 0);
    std::vector<std::uint8_t> candidate(node_count, 1);
    for (const auto node : order) {
        const auto absorbs = absorbable_kind(codes[node]);
        for (auto const &[source, complement] : input.operands(bool_node_id(node))) {
            const auto child = source.raw();
            ++references[child];
            if (complement || absorbs != codes[child]) {
                candidate[child] = 0;
            }
        }
    }
    for (auto const &target : input.targets()) {
        candidate[target.node.raw()] = 0;
    }

    // Absorbed operand lists are expanded only when a kept gate is emitted,
    // never copied into the absorbing parent: a chain of n nested ANDs then
    // costs O(n), not O(n^2). Each expansion step yields an output edge, so
    // the total work is bounded by the output edge count.
    detail::emitter emitter(input);
    auto &map = report.node_map;
    std::vector<std::uint64_t> expanded_fan_in(node_count, 0);
    std::vector<edge_ref> operands;
    struct frame final {
        std::uint32_t node;
        std::uint32_t next;
    };
    std::vector<frame> stack;

    for (const auto node : order) {
        const auto code = codes[node];
        if (is_leaf(code)) {
            map[node] = detail::reference(emitter.leaf(node));
            continue;
        }

        std::uint64_t fan_in = 0;
        for (auto const &operand : input.operands(bool_node_id(node))) {
            const auto child = operand.source.raw();
            fan_in += map[child].kind == image_kind::absorbed ? expanded_fan_in[child] : 1;
        }

        // Inlining into all m references and deleting the node changes the
        // edge count by m(k - 1) - k = (m - 1)(k - 1) - 1.
        const std::uint64_t parents = references[node];
        if (candidate[node] != 0 && (parents - 1) * (fan_in - 1) <= 1) {
            expanded_fan_in[node] = fan_in;
            map[node] = node_image{.kind = image_kind::absorbed};
            ++report.rewrites;
            continue;
        }

        operands.clear();
        stack.push_back(frame{.node = node, .next = 0});
        while (!stack.empty()) {
            auto &top = stack.back();
            const auto top_operands = input.operands(bool_node_id(top.node));
            if (top.next == top_operands.size()) {
                stack.pop_back();
                continue;
            }
            auto const &[source, complement] = top_operands[top.next++];
            if (map[source.raw()].kind == image_kind::absorbed) {
                stack.push_back(frame{.node = source.raw(), .next = 0});
            } else {
                operands.push_back(detail::through(map[source.raw()], complement).edge);
            }
        }
        map[node] = detail::reference(
            emitter.gate(code, operands, input.cardinality_low()[node], input.cardinality_high()[node]));
    }

    report.scratch_bytes = detail::bytes_of(order) + detail::bytes_of(references) + detail::bytes_of(candidate) +
                           detail::bytes_of(expanded_fan_in) + detail::bytes_of(operands) + detail::bytes_of(stack);
    return std::move(emitter).finish(std::move(report));
}

} // namespace fuzzybits::ir::pc::passes
