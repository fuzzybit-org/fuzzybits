#include "io/ftp/summary.h"

#include "ir/pra/model.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <string_view>
#include <system_error>
#include <utility>

namespace fuzzybits::io::backend::fuzzybits::ftp {

namespace {

namespace pra = ir::pra;

constexpr double tiny_probability = 1.0 / 4294967296.0;

[[nodiscard]] auto is_ncname(const std::string_view name) -> bool {
    const auto start = [](const char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; };
    const auto rest = [&start](const char c) { return start(c) || (c >= '0' && c <= '9') || c == '.' || c == '-'; };
    return !name.empty() && start(name.front()) && std::ranges::all_of(name.substr(1), rest);
}

[[nodiscard]] auto literal_value(pra::model const &model, pra::basic_event const &event) -> double {
    return model.expressions()[event.expression.raw()].value;
}

[[nodiscard]] auto source_path(pra::model const &model, const pra::source_location location) -> std::string {
    return std::string(model.text(model.source_files()[location.file.raw()].path));
}

[[nodiscard]] auto same_place(pra::source_location const &left, pra::source_location const &right) -> bool {
    return left.file.raw() == right.file.raw() && left.line == right.line;
}

// The model keeps one target per PROCESS name, located at its PROCESS line; an implicit target
// is located at its subject's definition.
[[nodiscard]] auto implicit_targets(pra::model const &model) -> bool {
    if (model.targets().empty()) {
        return false;
    }
    return std::ranges::all_of(model.targets(), [&model](pra::domain_target const &target) {
        const auto own = model.sources(target.element);
        const auto subject = model.sources(model.element_of(target.subject));
        return !own.empty() && !subject.empty() && same_place(own.front(), subject.front());
    });
}

void add_duplicate(structure_summary &out, pra::model const &model, pra::basic_event const &event, const double value) {
    const auto sources = model.sources(event.element);
    duplicate_import duplicate{.name = std::string(model.stable_name(event.element)), .value = value};
    std::vector<std::pair<pra::source_location, bool>> records;
    for (std::size_t position = 0; position < sources.size(); ++position) {
        records.emplace_back(sources[position], position == 0);
    }
    std::ranges::sort(records, [](auto const &left, auto const &right) {
        return std::pair(left.first.file.raw(), left.first.line) < std::pair(right.first.file.raw(), right.first.line);
    });
    for (auto const &[location, chosen] : records) {
        duplicate.occurrences.push_back(
            import_occurrence{.file = source_path(model, location), .line = location.line, .chosen = chosen});
    }
    out.duplicate_imports.push_back(std::move(duplicate));
}

} // namespace

auto structure_of(pra::model const &model) -> structure_summary {
    structure_summary out;
    out.gates = model.gates().size();
    std::vector<std::uint8_t> referenced(model.gates().size(), 0);
    for (std::uint32_t index = 0; index < model.gates().size(); ++index) {
        auto const &gate = model.gates()[index];
        const auto operands = model.operands(pra::gate_id(index));
        switch (gate.kind) {
        case pra::gate_kind::or_gate:
            ++out.or_gates;
            break;
        case pra::gate_kind::and_gate:
            ++out.and_gates;
            break;
        case pra::gate_kind::cardinality:
            ++out.atleast_gates;
            ++out.atleast_k_of_n[std::to_string(gate.bounds.low) + "/" + std::to_string(operands.size())];
            break;
        default:
            break;
        }
        out.edges += operands.size();
        out.single_input_gates += operands.size() == 1 ? 1 : 0;
        out.max_fan_in = std::max(out.max_fan_in, operands.size());
        for (auto const &operand : operands) {
            const auto kind = operand.event.kind();
            const bool to_basic_event = kind == pra::element_kind::basic_event;
            out.complemented_edges += operand.complement ? 1 : 0;
            out.complemented_edges_to_basic_events += operand.complement && to_basic_event ? 1 : 0;
            out.complemented_edges_to_gates += operand.complement && kind == pra::element_kind::gate ? 1 : 0;
            out.literal_operands += operand.literal ? 1 : 0;
            if (const auto child = operand.event.as<pra::gate_id>(); child.valid()) {
                referenced[child.raw()] = 1;
            }
        }
        out.names_not_ncname += is_ncname(model.stable_name(gate.element)) ? 0 : 1;
    }
    for (std::uint32_t index = 0; index < model.gates().size(); ++index) {
        if (referenced[index] == 0) {
            out.roots.emplace_back(model.stable_name(model.gates()[index].element));
        }
    }
    out.depth = model.summary().gate_depth;

    out.basic_events = model.basic_events().size();
    out.house_events = model.house_events().size();
    for (auto const &event : model.basic_events()) {
        const auto sources = model.sources(event.element);
        out.import_lines += sources.size();
        const auto value = literal_value(model, event);
        out.events_p_1 += value == 1.0 ? 1 : 0;
        out.events_p_0 += value == 0.0 ? 1 : 0;
        out.events_p_tiny += value > 0.0 && value < tiny_probability ? 1 : 0;
        out.names_not_ncname += is_ncname(model.stable_name(event.element)) ? 0 : 1;
        if (sources.size() > 1) {
            add_duplicate(out, model, event, value);
        }
    }
    out.initiating_events = model.initiating_events().size();
    for (auto const &initiator : model.initiating_events()) {
        const auto value = literal_value(model, model.basic_events()[initiator.event.raw()]);
        out.initiating_events_p_1 += value == 1.0 ? 1 : 0;
    }
    const bool implicit = implicit_targets(model);
    for (auto const &target : model.targets()) {
        out.default_targets.emplace_back(model.stable_name(target.element));
        if (!implicit) {
            out.process_targets.emplace_back(model.stable_name(target.element));
        }
    }
    for (auto const &attribute : model.project_attributes()) {
        if (model.text(attribute.name) == "ftp.limit") {
            const auto text = model.text(attribute.value);
            double value = 0.0;
            if (const auto [end, status] = std::from_chars(text.data(), text.data() + text.size(), value);
                status == std::errc{} && end == text.data() + text.size()) {
                out.limit = value;
            }
        }
    }
    return out;
}

} // namespace fuzzybits::io::backend::fuzzybits::ftp
