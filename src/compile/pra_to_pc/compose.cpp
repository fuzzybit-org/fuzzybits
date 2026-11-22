#include "compile/pra_to_pc/compose.h"

#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/passes/transform.h"

#include <utility>

namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc {

    auto compose_write_options(io::backend::fuzzybits::pcfb::write_options options, ir::pc::circuit const &input,
                               ir::pc::passes::pipeline_result const &passes) -> io::backend::fuzzybits::pcfb::write_options {
        using namespace ir::pc;

        auto names = std::move(options.target_names);
        options.target_names.clear();
        for (auto &[target, name] : names) {
            if (target.raw() < passes.target_map.size() && passes.target_map[target.raw()].valid()) {
                options.target_names.emplace(passes.target_map[target.raw()], std::move(name));
            }
        }

        auto nodes = std::move(options.node_names);
        options.node_names.clear();
        // Per-target settings travel with the target.
        const auto rekey = [&passes](auto &table) {
            auto moved = std::move(table);
            table.clear();
            for (auto &[target, value] : moved) {
                if (target.raw() < passes.target_map.size() && passes.target_map[target.raw()].valid()) {
                    table.emplace(passes.target_map[target.raw()], std::move(value));
                }
            }
        };
        rekey(options.target_kinds);
        rekey(options.quantification);

        auto const output_ops = passes.output.op_codes();
        auto const input_ops = input.op_codes();
        for (auto &[node, name] : nodes) {
            if (node.raw() >= passes.node_map.size() || node.raw() >= input_ops.size()) {
                continue;
            }
            auto const &image = passes.node_map[node.raw()];
            if (image.kind != passes::image_kind::reference || image.edge.complement || image.edge.source.raw() >= output_ops.size()) {
                continue;
            }
            const bool event_image = output_ops[image.edge.source.raw()] == op_code::basic_event;
            if (event_image && input_ops[node.raw()] != op_code::basic_event) {
                continue;
            }
            options.node_names.emplace(image.edge.source, std::move(name));
        }
        return options;
    }

} // namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc
