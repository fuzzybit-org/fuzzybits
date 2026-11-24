#include "io/ftp/detail/model.h"

#include "ir/pra/builder.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {

// Basic events take ids in event order and gates in definition order, so
// every reference is passed as an id and never resolved by name.
auto build_pra(ParsedProject const &project, PraOptions const &options) -> ir::pra::finalize_result {
    ir::pra::builder builder;

    for (const auto header : project.headers) {
        builder.add_project_attribute(kFtpHeaderAttribute, header);
    }
    if (project.limit) {
        builder.add_project_attribute(kFtpLimitAttribute, project.limitText);
    }
    if (project.maxSize) {
        builder.add_project_attribute(kFtpMaxSizeAttribute, project.maxSizeText);
    }
    builder.add_project_attribute(kFtpNegationAttribute, kFtpNegationDeleteTerm);
    if (options.maxOneInitiator) {
        builder.add_project_attribute(kFtpMaxInitiatorsAttribute, kFtpMaxInitiatorsOne);
    }

    std::vector<ir::pra::source_file_id> files;
    files.reserve(project.paths.size());
    for (auto const &path : project.paths) {
        files.push_back(builder.add_source_file(ir::pra::source_file_spec{.path = path, .format = kFtpFormat}));
    }
    const auto source = [&files](const Location location) {
        return ir::pra::source_location{.file = files[location.file], .line = location.line};
    };

    std::size_t operandCount = 0;
    for (auto const &definition : project.definitions) {
        operandCount += definition.operands.size();
    }
    builder.reserve_basic_events(static_cast<std::uint32_t>(project.events.size()));
    builder.reserve_gates(static_cast<std::uint32_t>(project.definitions.size()),
                          static_cast<std::uint32_t>(operandCount));

    std::vector<ir::pra::source_location> sources;
    std::vector<ir::pra::attribute_spec> attributes;
    for (std::uint32_t index = 0; index < project.events.size(); ++index) {
        auto const &event = project.events[index];
        sources.clear();
        attributes.clear();
        for (const auto record : event.records) {
            sources.push_back(source(project.imports[record].location));
            for (const auto flag : project.imports[record].flags) {
                attributes.push_back(ir::pra::attribute_spec{.name = kFtpImportFlagAttribute, .value = flag});
            }
        }
        const auto quantity = event.frequency ? ir::pra::quantity_kind::frequency : ir::pra::quantity_kind::probability;
        const auto id = builder.add_basic_event(
            ir::pra::element_spec{.stable_name = event.name, .attributes = attributes, .sources = sources},
            ir::pra::expression_spec::literal(event.value, quantity));
        if (event.initiator) {
            const auto initiator = builder.add_initiating_event(
                ir::pra::element_spec{.stable_name = event.name, .sources = std::span{sources}.first(1)}, id);
            (void)initiator;
        }
    }

    const auto markLiterals = options.complementEvents == complement_events_policy::atom;
    const auto operandSpec = [&project, markLiterals](Operand const &operand) -> ir::pra::operand_spec {
        if (const auto gate = project.definitionByName.find(operand.name); gate != project.definitionByName.end()) {
            return ir::pra::operand_spec{.event = ir::pra::gate_id(gate->second), .complement = operand.complement};
        }
        return ir::pra::operand_spec{.event = ir::pra::basic_event_id(project.eventByName.at(operand.name)),
                                     .complement = operand.complement,
                                     .literal = operand.complement && markLiterals};
    };

    std::vector<ir::pra::operand_spec> operands;
    for (auto const &definition : project.definitions) {
        operands.clear();
        for (auto const &operand : definition.operands) {
            operands.push_back(operandSpec(operand));
        }
        const auto location = source(definition.location);
        const ir::pra::element_spec element{.stable_name = definition.name, .sources = std::span{&location, 1}};
        ir::pra::gate_id id;
        switch (definition.kind) {
        case DefinitionKind::and_gate:
            id = builder.add_gate(element, ir::pra::gate_kind::and_gate, operands);
            break;
        case DefinitionKind::or_gate:
            id = builder.add_gate(element, ir::pra::gate_kind::or_gate, operands);
            break;
        case DefinitionKind::at_least:
            id = builder.add_cardinality(
                element,
                ir::pra::cardinality_bounds{.low = definition.threshold,
                                            .high = static_cast<std::uint32_t>(operands.size())},
                operands);
            break;
        }
        (void)id;
    }

    for (auto const &target : project.targets) {
        const auto location = source(target.location);
        const ir::pra::element_spec element{.stable_name = target.name, .sources = std::span{&location, 1}};
        const auto id =
            target.gate ? builder.add_target(element, ir::pra::domain_target_kind::gate, ir::pra::gate_id(target.index))
                        : builder.add_target(element, ir::pra::domain_target_kind::basic_event,
                                             ir::pra::basic_event_id(target.index));
        (void)id;
    }

    return std::move(builder).seal();
}

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
