#pragma once

// Structure counts of an ir::pra model read from FTP, with the keys of the
// conformance fixtures and the private model cards.

#include "api/diagnostics.h"
#include "ir/pra/model.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace ftp_test {

    struct structure final {
        std::size_t gates = 0;
        std::size_t or_gates = 0;
        std::size_t and_gates = 0;
        std::size_t atleast_gates = 0;
        std::map<std::string, std::size_t> atleast_k_of_n;
        std::size_t edges = 0;
        std::size_t single_input_gates = 0;
        std::size_t max_fan_in = 0;
        std::uint32_t depth = 0;
        std::size_t basic_events = 0;
        std::size_t import_lines = 0;
        std::size_t complemented_edges = 0;
        std::size_t complemented_edges_to_basic_events = 0;
        std::vector<std::string> roots;
        std::vector<std::string> targets;
        std::optional<double> limit;
        std::size_t initiators = 0;
        std::size_t initiators_p_1 = 0;
        std::size_t events_p_1 = 0;
        std::size_t events_p_0 = 0;
        std::size_t names_not_ncname = 0;
        std::size_t duplicate_imports = 0;
    };

    [[nodiscard]] inline auto is_ncname(const std::string_view name) -> bool {
        const auto start = [](const char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; };
        const auto rest = [&start](const char c) { return start(c) || (c >= '0' && c <= '9') || c == '.' || c == '-'; };
        return !name.empty() && start(name.front()) && std::ranges::all_of(name.substr(1), rest);
    }

    [[nodiscard]] inline auto literal_value(fuzzybits::ir::pra::model const &model, const fuzzybits::ir::pra::basic_event_id event) -> double {
        return model.expressions()[model.basic_events()[event.raw()].expression.raw()].value;
    }

    [[nodiscard]] inline auto structure_of(fuzzybits::ir::pra::model const &model, fuzzybits::api::diagnostic_list const &diagnostics)
        -> structure {
        namespace pra = fuzzybits::ir::pra;
        structure out;
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
            default:
                ++out.atleast_gates;
                ++out.atleast_k_of_n[std::to_string(gate.bounds.low) + "/" + std::to_string(operands.size())];
                break;
            }
            out.edges += operands.size();
            out.single_input_gates += operands.size() == 1 ? 1 : 0;
            out.max_fan_in = std::max(out.max_fan_in, operands.size());
            for (auto const &operand : operands) {
                out.complemented_edges += operand.complement ? 1 : 0;
                out.complemented_edges_to_basic_events +=
                    operand.complement && operand.event.kind() == pra::element_kind::basic_event ? 1 : 0;
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
        for (std::uint32_t index = 0; index < model.basic_events().size(); ++index) {
            auto const &event = model.basic_events()[index];
            out.import_lines += model.sources(event.element).size();
            const auto value = literal_value(model, pra::basic_event_id(index));
            out.events_p_1 += value == 1.0 ? 1 : 0;
            out.events_p_0 += value == 0.0 ? 1 : 0;
            out.names_not_ncname += is_ncname(model.stable_name(event.element)) ? 0 : 1;
        }
        out.initiators = model.initiating_events().size();
        for (auto const &initiator : model.initiating_events()) {
            out.initiators_p_1 += literal_value(model, initiator.event) == 1.0 ? 1 : 0;
        }
        for (auto const &target : model.targets()) {
            out.targets.emplace_back(model.stable_name(target.element));
        }
        for (auto const &attribute : model.project_attributes()) {
            if (model.text(attribute.name) == "ftp.limit") {
                const auto text = model.text(attribute.value);
                double value = 0.0;
                if (std::from_chars(text.data(), text.data() + text.size(), value).ec == std::errc{}) {
                    out.limit = value;
                }
            }
        }
        for (auto const &diagnostic : diagnostics.items()) {
            out.duplicate_imports += diagnostic.code == "io.ftp.duplicate-import" ? 1 : 0;
        }
        return out;
    }

    // Exact Boolean probability of a gate by enumerating every basic event (tests only).
    [[nodiscard]] inline auto brute_force_probability(fuzzybits::ir::pra::model const &model, const fuzzybits::ir::pra::domain_ref target)
        -> double {
        namespace pra = fuzzybits::ir::pra;
        const auto events = model.basic_events().size();
        std::vector<double> probability(events);
        for (std::uint32_t index = 0; index < events; ++index) {
            probability[index] = literal_value(model, pra::basic_event_id(index));
        }
        std::vector<std::uint8_t> eventValue(events);
        std::vector<std::uint8_t> gateValue(model.gates().size());
        double total = 0.0;
        for (std::uint64_t world = 0; world < (std::uint64_t{1} << events); ++world) {
            double weight = 1.0;
            for (std::size_t index = 0; index < events; ++index) {
                eventValue[index] = static_cast<std::uint8_t>((world >> index) & 1U);
                weight *= eventValue[index] != 0 ? probability[index] : 1.0 - probability[index];
            }
            if (weight == 0.0) {
                continue;
            }
            for (const auto gate : model.topological_gates()) {
                std::uint32_t trueCount = 0;
                const auto operands = model.operands(gate);
                for (auto const &operand : operands) {
                    const auto child = operand.event;
                    const bool value = child.kind() == pra::element_kind::gate ? gateValue[child.raw()] != 0 : eventValue[child.raw()] != 0;
                    trueCount += value != operand.complement ? 1 : 0;
                }
                auto const &row = model.gates()[gate.raw()];
                bool value = false;
                switch (row.kind) {
                case pra::gate_kind::and_gate:
                    value = trueCount == operands.size();
                    break;
                case pra::gate_kind::or_gate:
                    value = trueCount > 0;
                    break;
                case pra::gate_kind::cardinality:
                    value = trueCount >= row.bounds.low && trueCount <= row.bounds.high;
                    break;
                default:
                    break;
                }
                gateValue[gate.raw()] = value ? 1 : 0;
            }
            const bool observed = target.kind() == pra::element_kind::gate ? gateValue[target.raw()] != 0 : eventValue[target.raw()] != 0;
            total += observed ? weight : 0.0;
        }
        return total;
    }

} // namespace ftp_test
