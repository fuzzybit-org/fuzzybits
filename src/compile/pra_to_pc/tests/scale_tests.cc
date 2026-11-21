#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include "compile/pra_to_pc/descriptor.h"
#include "ir/pra/builder.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace pra_to_pc_test;

TEST_CASE("pra_to_pc lowers a 1e5-gate DAG in linear time", "[compile][pra_to_pc][scale]") {
    constexpr std::uint32_t layers = 250;
    constexpr std::uint32_t width = 400;
    constexpr std::uint32_t events = 2000;
    constexpr std::uint32_t initiators = 50;
    constexpr std::array kinds{pra::gate_kind::or_gate,  pra::gate_kind::and_gate, pra::gate_kind::cardinality, pra::gate_kind::xor_gate,
                               pra::gate_kind::nand_gate, pra::gate_kind::nor_gate, pra::gate_kind::xnor_gate, pra::gate_kind::not_gate};

    pra::builder builder;
    builder.reserve_gates(layers * width + 1, layers * width * 3 + width);
    builder.reserve_basic_events(events);
    std::vector<pra::basic_event_id> leaves;
    for (std::uint32_t event = 0; event < events; ++event) {
        leaves.push_back(builder.add_basic_event({.stable_name = "B" + std::to_string(event)}, pra::expression_spec::literal(1e-3)));
    }
    for (std::uint32_t event = 0; event < initiators; ++event) {
        (void)builder.add_initiating_event({.stable_name = "IE" + std::to_string(event)}, leaves[event * 7]);
    }
    std::vector<pra::gate_id> previous;
    std::vector<pra::gate_id> current;
    std::vector<pra::operand_spec> operands;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        current.clear();
        for (std::uint32_t index = 0; index < width; ++index) {
            const auto kind = kinds[(layer + index) % kinds.size()];
            const std::uint32_t fan_in = kind == pra::gate_kind::not_gate ? 1 : 3;
            operands.clear();
            for (std::uint32_t operand = 0; operand < fan_in; ++operand) {
                const pra::domain_ref source = layer == 0 ? pra::domain_ref(leaves[(index * 3 + operand * 7) % events])
                                                          : pra::domain_ref(previous[(index * 7 + operand * 13 + layer) % width]);
                operands.push_back(pra::operand_spec{.event = source, .complement = (index + operand) % 5 == 0});
            }
            const auto name = "G" + std::to_string(layer) + "_" + std::to_string(index);
            const pra::element_spec spec{.stable_name = name};
            current.push_back(kind == pra::gate_kind::cardinality ? builder.add_cardinality(spec, {.low = 2, .high = 3}, operands)
                                                                  : builder.add_gate(spec, kind, operands));
        }
        std::swap(previous, current);
    }
    operands.clear();
    for (const auto gate : previous) {
        operands.push_back(pra::operand_spec{.event = gate});
    }
    for (std::uint32_t event = 0; event < events; event += 3) {
        operands.push_back(pra::operand_spec{.event = leaves[event], .complement = event % 2 == 0});
    }
    const auto top = builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, operands);
    (void)builder.add_target({.stable_name = "TOP"}, pra::domain_target_kind::gate, top);
    const auto model = seal(std::move(builder));

    lowering::quantification_config config;
    config.max_initiators_per_cutset = 1;
    const auto start = std::chrono::steady_clock::now();
    const auto artifact = require_lower(model, config);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    CAPTURE(artifact.report.source_gates);
    CHECK(artifact.report.source_gates > model.gates().size() / 2);
    CHECK(artifact.circuit.node_count() == artifact.report.source_gates + artifact.report.source_basic_events + 2);
    CHECK(artifact.circuit.edge_count() == artifact.report.source_operands + artifact.report.generated_edges);
    auto const &binding = artifact.projection.target_bindings()[0];
    CHECK(binding.initiators.count > 1);
    CHECK(binding.initiators.count <= initiators);
    CHECK(binding.complemented_events.count > 0);

    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    CAPTURE(milliseconds);
#ifdef NDEBUG
    CHECK(elapsed < std::chrono::seconds(1));
#endif
}

TEST_CASE("pra_to_pc lowers a 1e5-gate chain without recursion", "[compile][pra_to_pc][scale]") {
    constexpr std::uint32_t length = 100'000;
    pra::builder builder;
    builder.reserve_gates(length, 2 * length);
    const auto event = builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::literal(0.5));
    pra::domain_ref below = event;
    for (std::uint32_t index = 0; index < length; ++index) {
        below = builder.add_gate({.stable_name = "G" + std::to_string(index)}, pra::gate_kind::or_gate,
                                 {{.event = below, .complement = index % 2 == 0}, {.event = event}});
    }
    (void)builder.add_target({.stable_name = "TOP"}, pra::domain_target_kind::gate, below);
    const auto model = seal(std::move(builder));
    const auto artifact = require_lower(model);
    CHECK(artifact.circuit.node_count() == length + 1);
    CHECK(artifact.circuit.edge_count() == 2 * length);
    CHECK(events_of_span(artifact.projection.complemented_events(artifact.projection.target_bindings()[0])) ==
          std::vector<pra::basic_event_id>{event});
}
