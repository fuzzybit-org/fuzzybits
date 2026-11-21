#include "ir/pc/passes/detail/rewrite.h"
#include "ir/pc/passes/passes.h"

#include <utility>

namespace fuzzybits::ir::pc::passes {

    namespace {

        [[nodiscard]] auto fold_probability(circuit const &input, const pass_id pass, const double probability, const bool value)
            -> transform_result {
            auto report = detail::begin_report(input, pass);
            if (auto trivial = detail::trivial_result(input, report)) {
                return std::move(*trivial);
            }

            const auto order = detail::cone_order(input);
            const auto codes = input.op_codes();
            const auto source_refs = input.probability_source_refs();
            const auto sources = input.probability_sources();
            detail::emitter emitter(input);
            auto &map = report.node_map;
            std::vector<edge_ref> operands;

            for (const auto node : order) {
                const auto code = codes[node];
                if (code == op_code::basic_event) {
                    auto const &source = sources[source_refs[node].raw()];
                    if (source.kind == probability_source_kind::literal && source.literal == probability) {
                        map[node] = detail::reference(emitter.constant(value));
                        ++report.rewrites;
                        continue;
                    }
                }
                if (is_leaf(code)) {
                    map[node] = detail::reference(emitter.leaf(node));
                    continue;
                }
                detail::remap_operands(input, node, map, operands);
                map[node] = detail::reference(emitter.gate(code, operands, input.cardinality_low()[node], input.cardinality_high()[node]));
            }

            report.scratch_bytes = detail::bytes_of(order) + detail::bytes_of(operands);
            return std::move(emitter).finish(std::move(report));
        }

    } // namespace

    auto fold_zero_probability(circuit const &input) -> transform_result {
        return fold_probability(input, pass_id::fold_zero_probability, 0.0, false);
    }

    auto fold_unit_probability(circuit const &input) -> transform_result {
        return fold_probability(input, pass_id::fold_unit_probability, 1.0, true);
    }

} // namespace fuzzybits::ir::pc::passes
