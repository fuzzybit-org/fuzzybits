#pragma once

// Test-only helpers over an ir::pra model read from FTP.

#include "ir/pra/model.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ftp_test {

[[nodiscard]] inline auto literal_value(fuzzybits::ir::pra::model const &model,
                                        const fuzzybits::ir::pra::basic_event_id event) -> double {
    return model.expressions()[model.basic_events()[event.raw()].expression.raw()].value;
}

// Exact Boolean probability of a gate by enumerating every basic event (tests only).
[[nodiscard]] inline auto brute_force_probability(fuzzybits::ir::pra::model const &model,
                                                  const fuzzybits::ir::pra::domain_ref target) -> double {
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
                const bool value = child.kind() == pra::element_kind::gate ? gateValue[child.raw()] != 0
                                                                           : eventValue[child.raw()] != 0;
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
        const bool observed =
            target.kind() == pra::element_kind::gate ? gateValue[target.raw()] != 0 : eventValue[target.raw()] != 0;
        total += observed ? weight : 0.0;
    }
    return total;
}

} // namespace ftp_test
