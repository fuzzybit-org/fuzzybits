#include "compile/pra_to_pc/descriptor.h"

#include "api/diagnostics.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/query.h"
#include "ir/pra/model.h"
#include "ir/pra/projection.h"

#include <cstddef>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc {
namespace {

namespace writer = io::backend::fuzzybits::pcfb;

[[nodiscard]] auto writer_kind(const ir::pc::query_kind kind) noexcept -> writer::query_kind {
    return kind == ir::pc::query_kind::probability ? writer::query_kind::probability
                                                   : writer::query_kind::monotone_hull_probability;
}

} // namespace

auto pcfb_write_options(pra_pc_lowering const &lowering) -> api::result<writer::write_options> {
    using options_result = api::result<writer::write_options>;
    auto const *model = ir::pra::model_from(lowering.pra_model);
    if (model == nullptr || !model->valid()) {
        return options_result::fail(api::make_error("compile.pra_to_pc.invalid-artifact",
                                                    "the lowering artifact does not hold an ir::pra model"));
    }

    writer::write_options options;
    api::diagnostic_list diagnostics;
    // Conditioned queries of one domain target (or element) and kind form one group, in binding
    // order; the group is the target's frequency (descriptor.h, initiator_values).
    struct pending_group final {
        std::string name;
        ir::pc::query_kind kind = ir::pc::query_kind::probability;
        std::vector<std::pair<ir::pc::target_id, double>> terms;
    };
    std::vector<pending_group> groups;
    std::map<std::tuple<ir::pra::domain_target_id, ir::pra::domain_ref, ir::pc::query_kind>, std::size_t> group_of;
    for (auto const &target : lowering.projection.target_bindings()) {
        const auto base = target.domain_target.valid() ? model->stable_name(ir::pra::domain_ref(target.domain_target))
                                                       : model->stable_name(target.subject);
        std::string name(base);
        // The target's name, the hull reading of a hull query, and the initiator rule where the lowering
        // put its delete term into this target.
        writer::target_quantification quantification{.target = std::string(base)};
        if (target.hull) {
            quantification.literal_mode = std::string(ir::pra::to_string(target.hull->literals));
            // The writer does not fill pcfb's per-query literal fields yet (RFC 7.3, #235), so a hull query
            // reads marked operands as plain complements: the plain reading, not split or fixed.
            const auto literals = lowering.projection.literal_events(target);
            if (target.hull->literals != ir::pra::hull_literals::plain && !literals.empty()) {
                diagnostics.push(api::make_error(
                    "compile.pra_to_pc.hull-literals-unsupported",
                    "target '" + std::string(name) + "' asks for a hull query that reads marked literals as " +
                        std::string(ir::pra::to_string(target.hull->literals)) + ", and its cone has " +
                        std::to_string(literals.size()) + " event(s) with marked literal operands (first: '" +
                        std::string(model->stable_name(ir::pra::domain_ref(literals.front()))) +
                        "'); the .pcfb writer cannot record literal atoms yet"));
            }
            if (target.initiator_delete_term.valid()) {
                quantification.max_initiators_per_cutset = target.hull->max_initiators;
            }
        }
        if (target.condition.valid()) {
            const auto initiator = std::string(model->stable_name(ir::pra::domain_ref(target.condition)));
            const auto frequency =
                model->expressions()[model->basic_events()[target.condition.raw()].expression.raw()].value;
            name += "|" + initiator;
            quantification.initiator = initiator;
            quantification.initiator_frequency = frequency;
            const auto key = std::make_tuple(target.domain_target, target.subject, target.kind);
            auto [found, added] = group_of.try_emplace(key, groups.size());
            if (added) {
                groups.push_back(pending_group{.name = std::string(base), .kind = target.kind});
            }
            groups[found->second].terms.emplace_back(target.pc_target, frequency);
        }
        options.target_names.emplace(target.pc_target, std::move(name));
        options.target_kinds.emplace(target.pc_target, std::vector<writer::query_kind>{writer_kind(target.kind)});
        options.quantification.emplace(target.pc_target, std::move(quantification));
    }
    const auto unit = lowering.report.initiator_unit.value_or(std::string{});
    for (auto &group : groups) {
        options.query_groups.push_back(writer::query_group{
            .name = group.name, .unit = unit, .kind = writer_kind(group.kind), .terms = std::move(group.terms)});
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
