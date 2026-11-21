#include "compile/pra_to_pc/descriptor.h"

#include "api/diagnostics.h"
#include "io/pcfb/descriptor.h"
#include "ir/pra/model.h"
#include "ir/pra/projection.h"

#include <string>
#include <utility>

namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc {

    auto pcfb_write_options(pra_pc_lowering const &lowering) -> api::result<io::backend::fuzzybits::pcfb::write_options> {
        using options_result = api::result<io::backend::fuzzybits::pcfb::write_options>;
        auto const *model = ir::pra::model_from(lowering.pra_model);
        if (model == nullptr || !model->valid()) {
            return options_result::fail(
                api::make_error("compile.pra_to_pc.invalid-artifact", "the lowering artifact does not hold an ir::pra model"));
        }

        io::backend::fuzzybits::pcfb::write_options options;
        api::diagnostic_list diagnostics;
        for (auto const &target : lowering.projection.target_bindings()) {
            const auto name = target.domain_target.valid() ? model->stable_name(ir::pra::domain_ref(target.domain_target))
                                                           : model->stable_name(target.subject);
            if (target.negation.monotone_hull) {
                diagnostics.push(api::make_error("compile.pra_to_pc.unsupported-query-kind",
                                                 "target '" + std::string(name) +
                                                     "' requests monotone-hull semantics, which pcfb core-1.1 cannot express"));
            }
            options.target_names.emplace(target.pc_target, std::string(name));
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
