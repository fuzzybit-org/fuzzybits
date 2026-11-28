#include "ir/pc/passes/detail/rewrite.h"
#include "ir/pc/passes/passes.h"

#include <optional>
#include <utility>

namespace fuzzybits::ir::pc::passes {

namespace {

[[nodiscard]] auto fold_probability(circuit const &input, pass_options const &options, const pass_id pass, const double probability, const bool value)
    -> transform_result {
    auto report = detail::begin_report(input, pass);
    if (auto trivial = detail::trivial_result(input, report)) {
        return std::move(*trivial);
    }
    // Folding an atom event would turn its atom into a constant, which
    // the hull cannot drop. p = 1 folding is never valid for the hull.
    const auto split = pass == pass_id::fold_zero_probability ? detail::split_context::make(input, options.atoms, report)
                                                              : std::optional<detail::split_context>(detail::split_context{});
    if (!split) {
        return transform_result{.output = {}, .report = std::move(report)};
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
                if (split->has_atom_edge(node)) {
                    ++report.skipped;
                    map[node] = detail::reference(emitter.leaf(node));
                    continue;
                }
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

    report.scratch_bytes = detail::bytes_of(order) + detail::bytes_of(operands) + split->bytes();
    return std::move(emitter).finish(std::move(report));
}

} // namespace

auto fold_zero_probability(circuit const &input, pass_options const &options) -> transform_result {
    return fold_probability(input, options, pass_id::fold_zero_probability, 0.0, false);
}

auto fold_unit_probability(circuit const &input, pass_options const &options) -> transform_result {
    return fold_probability(input, options, pass_id::fold_unit_probability, 1.0, true);
}

} // namespace fuzzybits::ir::pc::passes
