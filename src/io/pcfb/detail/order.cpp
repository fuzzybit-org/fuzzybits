#include "io/pcfb/detail/model.h"

#include "api/diagnostics.h"
#include "api/result.h"
#include "ir/pc/id.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::pcfb::detail {
    namespace {

        using order_result = api::result<std::vector<std::uint32_t>>;

        enum class VisitState : std::uint8_t {
            unseen,
            open,
            done,
        };

        struct Frame final {
            std::uint32_t node;
            std::uint32_t next_operand;
        };

        constexpr std::size_t kMaxCycleNodesShown = 8;

        [[nodiscard]] auto checkColumns(graph_view const &graph) -> api::diagnostic_list {
            api::diagnostic_list diagnostics;
            const auto nodeCount = graph.op_codes.size();
            if (nodeCount > ir::pc::bool_node_id::invalid_value) {
                diagnostics.push(api::make_error("io.pcfb.invalid-ir", "PC node count exceeds the 32-bit node id range"));
                return diagnostics;
            }
            if (graph.operand_offsets.size() != nodeCount || graph.operand_counts.size() != nodeCount) {
                diagnostics.push(api::make_error("io.pcfb.invalid-ir", "PC operand columns do not match the node count"));
                return diagnostics;
            }
            for (std::size_t node = 0; node < nodeCount; ++node) {
                const auto offset = std::uint64_t{graph.operand_offsets[node]};
                const auto count = std::uint64_t{graph.operand_counts[node]};
                if (offset + count > graph.edges.size()) {
                    diagnostics.push(api::make_error("io.pcfb.invalid-ir",
                                                     "node " + std::to_string(node) + " operand range exceeds the edge column"));
                    continue;
                }
                for (std::uint64_t edge = offset; edge < offset + count; ++edge) {
                    if (const auto source = graph.edges[edge].source; !source.valid() || source.raw() >= nodeCount) {
                        diagnostics.push(api::make_error("io.pcfb.invalid-edge",
                                                         "node " + std::to_string(node) + " references invalid operand node " +
                                                             (source.valid() ? std::to_string(source.raw()) : std::string("<invalid>"))));
                    }
                }
            }
            return diagnostics;
        }

        [[nodiscard]] auto cycleMessage(std::vector<Frame> const &stack, const std::uint32_t reentered) -> std::string {
            std::size_t first = stack.size();
            while (first > 0 && stack[first - 1].node != reentered) {
                --first;
            }
            first = first > 0 ? first - 1 : 0;

            std::string message = "PC circuit contains a dependency cycle through nodes ";
            const auto length = stack.size() - first;
            for (std::size_t index = 0; index < length && index < kMaxCycleNodesShown; ++index) {
                message += std::to_string(stack[first + index].node) + " -> ";
            }
            if (length > kMaxCycleNodesShown) {
                message += "... -> ";
            }
            message += std::to_string(reentered);
            return message;
        }

    } // namespace

    auto emission_order(graph_view const &graph) -> order_result {
        if (auto diagnostics = checkColumns(graph); diagnostics.has_errors()) {
            return order_result::fail(std::move(diagnostics));
        }

        const auto nodeCount = static_cast<std::uint32_t>(graph.op_codes.size());
        std::vector<VisitState> states(nodeCount, VisitState::unseen);
        std::vector<std::uint32_t> order;
        order.reserve(nodeCount);
        std::vector<Frame> stack;

        for (std::uint32_t root = 0; root < nodeCount; ++root) {
            if (states[root] != VisitState::unseen) {
                continue;
            }
            states[root] = VisitState::open;
            stack.push_back(Frame{.node = root, .next_operand = 0});
            while (!stack.empty()) {
                const auto node = stack.back().node;
                if (const auto operand = stack.back().next_operand; operand < graph.operand_counts[node]) {
                    ++stack.back().next_operand;
                    const auto source = graph.edges[graph.operand_offsets[node] + operand].source.raw();
                    if (states[source] == VisitState::unseen) {
                        states[source] = VisitState::open;
                        stack.push_back(Frame{.node = source, .next_operand = 0});
                    } else if (states[source] == VisitState::open) {
                        return order_result::fail(api::make_error("io.pcfb.cycle", cycleMessage(stack, source)));
                    }
                    continue;
                }
                states[node] = VisitState::done;
                order.push_back(node);
                stack.pop_back();
            }
        }
        return order_result::ok(std::move(order));
    }

} // namespace fuzzybits::io::backend::fuzzybits::pcfb::detail
