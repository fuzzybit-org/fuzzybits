#pragma once

#include "api/diagnostics.h"
#include "ir/pra/builder.h"
#include "ir/pra/model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace fuzzybits::ir::pra::test {

    [[nodiscard]] inline auto named(const std::string_view name, const bool complement = false) -> operand_spec {
        return operand_spec{.event = reference::named(name), .complement = complement};
    }

    [[nodiscard]] inline auto codes(api::diagnostic_list const &diagnostics) -> std::vector<std::string> {
        std::vector<std::string> out;
        for (auto const &item : diagnostics.items()) {
            out.push_back(item.code);
        }
        return out;
    }

    [[nodiscard]] inline auto count(api::diagnostic_list const &diagnostics, const std::string_view code) -> std::size_t {
        return static_cast<std::size_t>(
            std::ranges::count_if(diagnostics.items(), [code](api::diagnostic const &item) { return item.code == code; }));
    }

    [[nodiscard]] inline auto message(api::diagnostic_list const &diagnostics, const std::string_view code) -> std::string {
        for (auto const &item : diagnostics.items()) {
            if (item.code == code) {
                return item.message;
            }
        }
        return {};
    }

    [[nodiscard]] inline auto render(api::diagnostic_list const &diagnostics) -> std::string {
        std::string out;
        for (auto const &item : diagnostics.items()) {
            out += item.code + ": " + item.message + "\n";
        }
        return out;
    }

    [[nodiscard]] inline auto topologically_ordered(model const &sealed) -> bool {
        const auto order = sealed.topological_gates();
        if (order.size() != sealed.gates().size()) {
            return false;
        }
        std::vector<std::uint32_t> position(order.size(), std::numeric_limits<std::uint32_t>::max());
        for (std::uint32_t index = 0; index < order.size(); ++index) {
            position[order[index].raw()] = index;
        }
        for (std::uint32_t gate = 0; gate < sealed.gates().size(); ++gate) {
            for (auto const &operand : sealed.operands(gate_id(gate))) {
                if (const auto child = operand.event.as<gate_id>(); child.valid() && position[child.raw()] >= position[gate]) {
                    return false;
                }
            }
        }
        return true;
    }

} // namespace fuzzybits::ir::pra::test
