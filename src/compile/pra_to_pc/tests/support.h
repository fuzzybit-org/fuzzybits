#pragma once

#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "compile/pra_to_pc/descriptor.h"
#include "ir/pc/circuit.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pra/builder.h"
#include "ir/pra/id.h"
#include "ir/pra/model.h"
#include "ir/pra/projection.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pra_to_pc_test {

namespace pra = fuzzybits::ir::pra;
namespace pc = fuzzybits::ir::pc;
namespace lowering = fuzzybits::compile::backend::fuzzybits::pra_to_pc;

[[nodiscard]] inline auto render(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
    std::string out;
    for (auto const &item : diagnostics.items()) {
        out += item.code + ": " + item.message + "\n";
    }
    return out;
}

[[nodiscard]] inline auto codes(fuzzybits::api::diagnostic_list const &diagnostics) -> std::vector<std::string> {
    std::vector<std::string> out;
    for (auto const &item : diagnostics.items()) {
        out.push_back(item.code);
    }
    return out;
}

[[nodiscard]] inline auto count(fuzzybits::api::diagnostic_list const &diagnostics, const std::string_view code) -> std::size_t {
    return static_cast<std::size_t>(std::ranges::count_if(diagnostics.items(), [code](fuzzybits::api::diagnostic const &item) { return item.code == code; }));
}

[[nodiscard]] inline auto events_of_span(std::span<pra::basic_event_id const> events) -> std::vector<pra::basic_event_id> {
    return {events.begin(), events.end()};
}

[[nodiscard]] inline auto seal(pra::builder &&builder) -> pra::model {
    auto sealed = std::move(builder).seal();
    INFO(render(sealed.diagnostics));
    REQUIRE(sealed.ok());
    return sealed.output;
}

[[nodiscard]] inline auto require_lower(pra::model const &model, lowering::quantification_config const &config = {}) -> lowering::pra_pc_lowering {
    auto lowered = lowering::lower(model, config);
    INFO(render(lowered.diagnostics()));
    REQUIRE(lowered);
    return std::move(lowered).value();
}

struct reference_override final {
    pra::domain_ref element;
    bool value = false;
};

// ir::pra semantics written from gate.h, independent of the lowering. A
// world is a bit mask over all basic events of the model (bit i: event i).
class pra_reference final {
  public:
    pra_reference(pra::model const &model, std::vector<reference_override> overrides) : model_(model), overrides_(std::move(overrides)) {
        REQUIRE(model.basic_events().size() <= 20);
    }

    [[nodiscard]] auto values(const std::uint64_t world) const -> std::vector<std::uint8_t> {
        std::vector<std::uint8_t> gate_values(model_.gates().size(), 0);
        for (const auto gate : model_.topological_gates()) {
            if (const auto fixed = override_of(gate)) {
                gate_values[gate.raw()] = *fixed ? 1 : 0;
                continue;
            }
            auto const &row = model_.gates()[gate.raw()];
            const auto operands = model_.operands(gate);
            std::uint32_t trues = 0;
            for (auto const &operand : operands) {
                trues += leaf_or_gate(operand.event, world, gate_values) != operand.complement ? 1U : 0U;
            }
            const auto n = static_cast<std::uint32_t>(operands.size());
            bool value = false;
            switch (row.kind) {
            case pra::gate_kind::and_gate:
                value = trues == n;
                break;
            case pra::gate_kind::or_gate:
                value = trues >= 1;
                break;
            case pra::gate_kind::cardinality:
                value = row.bounds.low <= trues && trues <= row.bounds.high;
                break;
            case pra::gate_kind::not_gate:
                value = trues == 0;
                break;
            case pra::gate_kind::xor_gate:
                value = (trues % 2) == 1;
                break;
            case pra::gate_kind::nand_gate:
                value = trues != n;
                break;
            case pra::gate_kind::nor_gate:
                value = trues == 0;
                break;
            case pra::gate_kind::xnor_gate:
                value = (trues % 2) == 0;
                break;
            }
            gate_values[gate.raw()] = value ? 1 : 0;
        }
        return gate_values;
    }

    [[nodiscard]] auto value(const pra::domain_ref element, const std::uint64_t world) const -> bool { return leaf_or_gate(element, world, values(world)); }

    // Basic events reachable from `subject` without entering an overridden
    // element; with `literals`, only those referenced through an operand marked `literal`.
    [[nodiscard]] auto cone_events(const pra::domain_ref subject, const bool literals) const -> std::vector<pra::basic_event_id> {
        std::vector<std::uint8_t> seen_gate(model_.gates().size(), 0);
        std::vector<std::uint8_t> seen_event(model_.basic_events().size(), 0);
        std::vector<pra::basic_event_id> out;
        std::vector<std::pair<pra::domain_ref, bool>> stack{{subject, false}};
        while (!stack.empty()) {
            const auto [element, literal] = stack.back();
            stack.pop_back();
            if (override_of(element)) {
                continue;
            }
            if (const auto gate = element.as<pra::gate_id>(); gate.valid()) {
                if (seen_gate[gate.raw()] == 0) {
                    seen_gate[gate.raw()] = 1;
                    for (auto const &operand : model_.operands(gate)) {
                        stack.emplace_back(operand.event, operand.literal);
                    }
                }
                continue;
            }
            const auto event = element.as<pra::basic_event_id>();
            if (event.valid() && (!literals || literal) && seen_event[event.raw()] == 0) {
                seen_event[event.raw()] = 1;
                out.push_back(event);
            }
        }
        std::ranges::sort(out);
        return out;
    }

    [[nodiscard]] auto cone_initiators(const pra::domain_ref subject) const -> std::vector<pra::basic_event_id> {
        std::vector<pra::basic_event_id> out;
        for (const auto event : cone_events(subject, false)) {
            if (std::ranges::any_of(model_.initiating_events(), [event](auto const &row) { return row.event == event; })) {
                out.push_back(event);
            }
        }
        return out;
    }

    // The function a target has under the config: the subject, ANDed with
    // "at most k initiating events of the cone are true" when the rule is set.
    [[nodiscard]] auto target_value(const pra::domain_ref subject, const std::optional<std::uint32_t> max_initiators, const std::uint64_t world) const -> bool {
        if (!value(subject, world)) {
            return false;
        }
        if (!max_initiators) {
            return true;
        }
        std::uint32_t initiators = 0;
        for (const auto event : cone_initiators(subject)) {
            initiators += ((world >> event.raw()) & 1U) != 0 ? 1U : 0U;
        }
        return initiators <= *max_initiators;
    }

    [[nodiscard]] auto probability(const pra::basic_event_id event) const -> double {
        return model_.expressions()[model_.basic_events()[event.raw()].expression.raw()].value;
    }

    [[nodiscard]] auto target_probability(const pra::domain_ref subject, const std::optional<std::uint32_t> max_initiators) const -> double {
        const auto events = model_.basic_events().size();
        double sum = 0.0;
        for (std::uint64_t world = 0; world < (std::uint64_t{1} << events); ++world) {
            if (!target_value(subject, max_initiators, world)) {
                continue;
            }
            double mass = 1.0;
            for (std::uint32_t event = 0; event < events; ++event) {
                const auto p = probability(pra::basic_event_id(event));
                mass *= ((world >> event) & 1U) != 0 ? p : 1.0 - p;
            }
            sum += mass;
        }
        return sum;
    }

  private:
    [[nodiscard]] auto override_of(const pra::domain_ref element) const -> std::optional<bool> {
        for (auto const &item : overrides_) {
            if (item.element == element) {
                return item.value;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] auto leaf_or_gate(const pra::domain_ref element, const std::uint64_t world, std::vector<std::uint8_t> const &gate_values) const -> bool {
        if (const auto fixed = override_of(element)) {
            return *fixed;
        }
        switch (element.kind()) {
        case pra::element_kind::basic_event:
            return ((world >> element.raw()) & 1U) != 0;
        case pra::element_kind::house_event:
            return model_.house_events()[element.raw()].value;
        case pra::element_kind::gate:
            return gate_values[element.raw()] != 0;
        default:
            FAIL("operand of unexpected kind");
            return false;
        }
    }

    pra::model const &model_;
    std::vector<reference_override> overrides_;
};

// ir::pc semantics over a node order computed here (Kahn), with each
// basic-event node an independent variable.
class pc_reference final {
  public:
    explicit pc_reference(pc::circuit const &circuit) : circuit_(circuit) {
        const auto nodes = circuit.node_count();
        std::vector<std::uint32_t> pending(nodes, 0);
        std::vector<std::vector<std::uint32_t>> users(nodes);
        for (std::uint32_t node = 0; node < nodes; ++node) {
            for (auto const &edge : circuit.operands(pc::bool_node_id(node))) {
                users[edge.source.raw()].push_back(node);
                ++pending[node];
            }
            if (circuit.op_codes()[node] == pc::op_code::basic_event) {
                events_.push_back(node);
            }
        }
        for (std::uint32_t node = 0; node < nodes; ++node) {
            if (pending[node] == 0) {
                order_.push_back(node);
            }
        }
        for (std::size_t cursor = 0; cursor < order_.size(); ++cursor) {
            for (const auto user : users[order_[cursor]]) {
                if (--pending[user] == 0) {
                    order_.push_back(user);
                }
            }
        }
        REQUIRE(order_.size() == nodes);
    }

    [[nodiscard]] auto events() const -> std::vector<std::uint32_t> const & { return events_; }

    // `event_values[node]` gives each basic-event node's value.
    [[nodiscard]] auto values(std::vector<std::uint8_t> event_values) const -> std::vector<std::uint8_t> {
        auto &value = event_values;
        value.resize(circuit_.node_count(), 0);
        for (const auto node : order_) {
            const auto code = circuit_.op_codes()[node];
            if (code == pc::op_code::basic_event) {
                continue;
            }
            if (code == pc::op_code::constant || code == pc::op_code::house_event) {
                value[node] = circuit_.constant_values()[node];
                continue;
            }
            const auto operands = circuit_.operands(pc::bool_node_id(node));
            std::uint32_t trues = 0;
            for (auto const &[source, complement] : operands) {
                trues += (value[source.raw()] != 0) != complement ? 1U : 0U;
            }
            const auto n = static_cast<std::uint32_t>(operands.size());
            bool result = false;
            switch (code) {
            case pc::op_code::and_gate:
                result = trues == n;
                break;
            case pc::op_code::or_gate:
                result = trues != 0;
                break;
            case pc::op_code::xor_gate:
                result = (trues & 1U) != 0;
                break;
            case pc::op_code::not_gate:
            case pc::op_code::nor_gate:
                result = trues == 0;
                break;
            case pc::op_code::nand_gate:
                result = trues != n;
                break;
            case pc::op_code::xnor_gate:
                result = (trues & 1U) == 0;
                break;
            case pc::op_code::cardinality:
                result = trues >= circuit_.cardinality_low()[node] && trues <= circuit_.cardinality_high()[node];
                break;
            case pc::op_code::basic_event:
            case pc::op_code::constant:
            case pc::op_code::house_event:
                break;
            }
            value[node] = result ? 1 : 0;
        }
        return value;
    }

    [[nodiscard]] auto target_probabilities() const -> std::vector<double> {
        REQUIRE(events_.size() <= 20);
        std::vector<double> sum(circuit_.targets().size(), 0.0);
        std::vector<std::uint8_t> event_values(circuit_.node_count(), 0);
        for (std::uint64_t world = 0; world < (std::uint64_t{1} << events_.size()); ++world) {
            double mass = 1.0;
            for (std::size_t index = 0; index < events_.size(); ++index) {
                const bool bit = ((world >> index) & 1U) != 0;
                const auto node = events_[index];
                event_values[node] = bit ? 1 : 0;
                const auto p = circuit_.probability_sources()[circuit_.probability_source_refs()[node].raw()].literal;
                mass *= bit ? p : 1.0 - p;
            }
            const auto value = values(event_values);
            for (std::size_t target = 0; target < sum.size(); ++target) {
                sum[target] += value[circuit_.targets()[target].node.raw()] != 0 ? mass : 0.0;
            }
        }
        return sum;
    }

  private:
    pc::circuit const &circuit_;
    std::vector<std::uint32_t> events_;
    std::vector<std::uint32_t> order_;
};

// For every world of the model's basic events, every PC target equals its
// domain target's function under the config; and every target probability
// of the circuit equals the PRA one.
// The initiator rule as it applies to one binding: hull queries only. A probability query is the
// subject itself and never holds a delete term or hull parameters.
[[nodiscard]] inline auto rule_of(pra::target_binding const &target) -> std::optional<std::uint32_t> {
    if (target.kind == pc::query_kind::probability) {
        REQUIRE_FALSE(target.hull);
        REQUIRE_FALSE(target.initiator_delete_term.valid());
        REQUIRE(target.node == target.root);
        return std::nullopt;
    }
    REQUIRE(target.hull);
    return target.hull->max_initiators;
}

inline void require_equivalent(pra::model const &model, lowering::pra_pc_lowering const &lowered, std::vector<reference_override> const &overrides) {
    const pra_reference reference(model, overrides);
    const pc_reference circuit(lowered.circuit);
    auto const targets = lowered.projection.target_bindings();
    REQUIRE(targets.size() == lowered.circuit.targets().size());

    std::vector<pra::basic_event_id> event_of_node(lowered.circuit.node_count());
    for (const auto node : circuit.events()) {
        auto const *binding = lowered.projection.find_event(pc::bool_node_id(node));
        REQUIRE(binding != nullptr);
        REQUIRE(binding->kind == pra::event_binding_kind::basic_event);
        event_of_node[node] = binding->element.as<pra::basic_event_id>();
        REQUIRE(event_of_node[node].valid());
    }

    const auto events = model.basic_events().size();
    std::vector<std::uint8_t> event_values(lowered.circuit.node_count(), 0);
    for (std::uint64_t world = 0; world < (std::uint64_t{1} << events); ++world) {
        for (const auto node : circuit.events()) {
            event_values[node] = ((world >> event_of_node[node].raw()) & 1U) != 0 ? 1 : 0;
        }
        const auto value = circuit.values(event_values);
        for (auto const &target : targets) {
            REQUIRE(lowered.circuit.targets()[target.pc_target.raw()].node == target.node);
            CAPTURE(world, target.pc_target.raw(), model.stable_name(target.subject));
            REQUIRE((value[target.node.raw()] != 0) == reference.target_value(target.subject, rule_of(target), world));
        }
    }

    const auto probabilities = circuit.target_probabilities();
    for (auto const &target : targets) {
        const auto expected = reference.target_probability(target.subject, rule_of(target));
        CAPTURE(model.stable_name(target.subject), expected, probabilities[target.pc_target.raw()]);
        REQUIRE(std::abs(probabilities[target.pc_target.raw()] - expected) <= 1e-12);
    }
}

} // namespace pra_to_pc_test
