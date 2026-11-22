#include "compile/pra_to_pc/descriptor.h"

#include "api/diagnostics.h"
#include "io/pcfb/descriptor.h"
#include "ir/pra/model.h"
#include "ir/pra/projection.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc {
    namespace {

        namespace writer = io::backend::fuzzybits::pcfb;

        [[nodiscard]] auto literal_mode_name(const ir::pra::literal_mode mode) noexcept -> std::string_view {
            switch (mode) {
            case ir::pra::literal_mode::atom_split:
                return "atom_split";
            case ir::pra::literal_mode::omega_l:
                return "omega_l";
            case ir::pra::literal_mode::negation:
                return "negation";
            }
            return "atom_split";
        }

    } // namespace

    auto pcfb_write_options(pra_pc_lowering const &lowering) -> api::result<writer::write_options> {
        using options_result = api::result<writer::write_options>;
        auto const *model = ir::pra::model_from(lowering.pra_model);
        if (model == nullptr || !model->valid()) {
            return options_result::fail(
                api::make_error("compile.pra_to_pc.invalid-artifact", "the lowering artifact does not hold an ir::pra model"));
        }

        writer::write_options options;
        api::diagnostic_list diagnostics;
        for (auto const &target : lowering.projection.target_bindings()) {
            const auto name = target.domain_target.valid() ? model->stable_name(ir::pra::domain_ref(target.domain_target))
                                                           : model->stable_name(target.subject);
            std::vector<writer::query_kind> kinds;
            if (target.negation.boolean) {
                kinds.push_back(writer::query_kind::probability);
            }
            if (target.negation.monotone_hull) {
                kinds.push_back(writer::query_kind::monotone_hull_probability);
                // pcfb 1.2 has no literal atoms per query (literal_samples, RFC 7.3), so a hull query reads
                // marked operands as plain complements: exactly the `negation` mode, and not what
                // atom_split and omega_l ask for.
                const auto literals = lowering.projection.literal_events(target);
                if (target.literals != ir::pra::literal_mode::negation && !literals.empty()) {
                    diagnostics.push(api::make_error(
                        "compile.pra_to_pc.hull-literals-unsupported",
                        "target '" + std::string(name) + "' requests monotone-hull semantics and its cone has " + std::to_string(literals.size()) +
                            " event(s) with marked literal operands (first: '" + std::string(model->stable_name(ir::pra::domain_ref(literals.front()))) +
                            "'); pcfb 1.2 cannot carry literal atoms yet"));
                }
            }
            if (kinds.empty()) {
                diagnostics.push(api::make_error("compile.pra_to_pc.invalid-artifact",
                                                 "target '" + std::string(name) + "' requests no negation semantics"));
            }
            options.target_names.emplace(target.pc_target, std::string(name));
            options.target_kinds.emplace(target.pc_target, std::move(kinds));
            options.quantification.emplace(target.pc_target,
                                           writer::target_quantification{
                                               .literal_mode = std::string(literal_mode_name(target.literals)),
                                               .max_initiators_per_cutset = lowering.report.max_initiators_per_cutset,
                                           });
        }
        for (auto const &event : lowering.projection.event_bindings()) {
            options.node_names.emplace(event.node, std::string(model->stable_name(event.element)));
        }
        options.model_name = std::string(model->text(model->project().name));
        if (diagnostics.has_errors()) {
            return options_result::fail(std::move(diagnostics));
        }
        return options_result::ok(std::move(options));
    }

} // namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc
