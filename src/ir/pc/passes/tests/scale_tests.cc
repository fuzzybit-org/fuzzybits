#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

    using namespace pc_passes_test;

    constexpr std::uint32_t scale_nodes = 100'000;
    constexpr std::uint32_t scale_events = 4'000;
    constexpr std::uint32_t scale_layers = 160;

    [[nodiscard]] auto depth_of(pc::circuit const &model) -> std::uint32_t {
        std::vector<std::uint32_t> depth(model.node_count(), 0);
        std::uint32_t deepest = 0;
        for (const auto node : topological_order(model)) {
            for (auto const &operand : model.operands(pc::bool_node_id(node))) {
                depth[node] = std::max(depth[node], depth[operand.source.raw()] + 1);
            }
            deepest = std::max(deepest, depth[node]);
        }
        return deepest;
    }

    // Layered DAG: every gate takes one operand from the previous layer and the
    // rest from anywhere below, so subtrees are widely shared. FTP-like mix of
    // OR, AND and at-least gates, some single-input gates, complements and
    // p in {0, 1} events.
    [[nodiscard]] auto layered_dag(std::mt19937_64 &rng) -> pc::circuit {
        const auto uniform = [&](const std::uint32_t low, const std::uint32_t high) {
            return std::uniform_int_distribution<std::uint32_t>(low, high)(rng);
        };
        pc::builder builder;
        builder.reserve_nodes(scale_nodes + 1);
        std::vector<pc::bool_node_id> nodes;
        for (std::uint32_t index = 0; index < scale_events; ++index) {
            const auto draw = uniform(0, 199);
            const double p = draw == 0 ? 0.0 : draw == 1 ? 1.0 : 1.0e-4 * (1 + uniform(0, 999));
            nodes.push_back(builder.add_basic_event(builder.add_literal_probability(p)));
        }
        const auto gates_per_layer = (scale_nodes - scale_events - 1) / scale_layers;
        std::uint32_t previous_begin = 0;
        std::uint32_t previous_end = scale_events;
        std::vector<edge> operands;
        for (std::uint32_t layer = 0; layer < scale_layers; ++layer) {
            const auto begin = static_cast<std::uint32_t>(nodes.size());
            for (std::uint32_t gate = 0; gate < gates_per_layer; ++gate) {
                operands.clear();
                const auto fan_in = uniform(0, 19) == 0 ? 1U : uniform(2, 6);
                operands.push_back(edge{.source = nodes[uniform(previous_begin, previous_end - 1)]});
                for (std::uint32_t operand = 1; operand < fan_in; ++operand) {
                    const auto source = uniform(0, 1) == 0 ? uniform(0, scale_events - 1) : uniform(0, begin - 1);
                    operands.push_back(edge{.source = nodes[source], .complement = uniform(0, 49) == 0});
                }
                const auto kind = uniform(0, 19);
                if (kind < 2 && fan_in >= 2) {
                    nodes.push_back(builder.add_cardinality(uniform(1, fan_in), fan_in, operands));
                } else {
                    nodes.push_back(builder.add_gate(kind < 11 ? pc::op_code::or_gate : pc::op_code::and_gate, operands));
                }
            }
            previous_begin = begin;
            previous_end = static_cast<std::uint32_t>(nodes.size());
        }
        operands.clear();
        for (auto index = previous_begin; index < previous_end; ++index) {
            operands.push_back(edge{.source = nodes[index]});
        }
        (void)builder.add_target(builder.add_or(operands));
        for (std::uint32_t index = 0; index < 4; ++index) {
            (void)builder.add_target(nodes[uniform(scale_events, previous_end - 1)], pc::observable_role::watch);
        }
        return seal(std::move(builder));
    }

    // Targets agree on random worlds in which p = 0 / p = 1 events take their
    // almost-sure value.
    void require_same_targets(pc::circuit const &input, passes::pipeline_result const &result, std::mt19937_64 &rng) {
        const evaluator in(input);
        const evaluator out(result.output);
        std::vector<word> input_words(in.events().size());
        std::vector<word> output_words(out.events().size(), 0);
        std::vector<word> input_values;
        std::vector<word> output_values;
        for (std::uint32_t block = 0; block < 8; ++block) {
            for (std::uint32_t slot = 0; slot < in.events().size(); ++slot) {
                const auto p = literal(input, in.events()[slot]);
                input_words[slot] = p == 0.0 ? 0 : p == 1.0 ? all_lanes : rng();
                auto const &image = result.node_map[in.events()[slot]];
                if (image.kind == passes::image_kind::reference &&
                    result.output.op_codes()[image.edge.source.raw()] == pc::op_code::basic_event) {
                    output_words[out.slot(image.edge.source.raw())] = input_words[slot];
                }
            }
            in.run(input_words, input_values);
            out.run(output_words, output_values);
            for (std::uint32_t target = 0; target < input.targets().size(); ++target) {
                const auto out_target = result.output.targets()[result.target_map[target].raw()].node.raw();
                REQUIRE(input_values[input.targets()[target].node.raw()] == output_values[out_target]);
            }
        }
    }

} // namespace

TEST_CASE("C1 on a 1e5-node DAG of depth > 140 is fast, never adds edges and keeps every target", "[ir][pc][passes][scale]") {
    std::mt19937_64 rng(223);
    const auto input = layered_dag(rng);
    REQUIRE(input.node_count() >= scale_nodes - scale_layers);
    REQUIRE(depth_of(input) > 140);

    const auto start = std::chrono::steady_clock::now();
    const auto result = passes::compile(input, passes::compile_level::c1);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    INFO(diagnostics_text(result.diagnostics));
    REQUIRE(result.ok());
    std::cout << "C1 scale: nodes " << result.nodes_before << " -> " << result.nodes_after << ", edges " << result.edges_before << " -> "
              << result.edges_after << ", passes " << result.passes.size() << ", " << elapsed << " s\n";
    REQUIRE(result.edges_after <= result.edges_before);
    for (auto const &report : result.passes) {
        REQUIRE(report.edges_after <= report.edges_before);
    }
#ifdef NDEBUG
    REQUIRE(elapsed < 2.0);
#endif
    require_same_targets(input, result, rng);

    for (const auto pass : {passes::pass_id::propagate_constants, passes::pass_id::eliminate_single_input, passes::pass_id::coalesce}) {
        const auto pass_start = std::chrono::steady_clock::now();
        const auto single = passes::run_pass(pass, input);
        const auto pass_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - pass_start).count();
        REQUIRE(single.ok());
        REQUIRE(single.report.edges_after <= single.report.edges_before);
#ifdef NDEBUG
        REQUIRE(pass_elapsed < 0.5);
#endif
    }
}

TEST_CASE("Passes traverse 1e5-deep chains without recursion", "[ir][pc][passes][scale]") {
    constexpr std::uint32_t length = 100'000;
    SECTION("nested AND coalesces into one gate") {
        pc::builder builder;
        auto chain = builder.add_basic_event(builder.add_literal_probability(0.5));
        for (std::uint32_t index = 0; index < length; ++index) {
            const auto next = builder.add_basic_event(builder.add_literal_probability(0.5));
            chain = builder.add_and({edge{.source = next}, edge{.source = chain}});
        }
        (void)builder.add_target(chain);
        const auto input = seal(std::move(builder));
        const auto start = std::chrono::steady_clock::now();
        const auto result = passes::coalesce(input);
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        REQUIRE(result.ok());
#ifdef NDEBUG
        REQUIRE(elapsed < 0.5);
#endif
        const auto top = result.output.targets()[0].node.raw();
        REQUIRE(result.output.op_codes()[top] == pc::op_code::and_gate);
        REQUIRE(result.output.operand_counts()[top] == length + 1);
        REQUIRE(result.report.edges_after == length + 1);
        REQUIRE(result.report.nodes_after == length + 2);
    }
    SECTION("NOT chain collapses to its event") {
        pc::builder builder;
        const auto event = builder.add_basic_event(builder.add_literal_probability(0.5));
        auto chain = event;
        for (std::uint32_t index = 0; index < length; ++index) {
            chain = builder.add_not(edge{.source = chain});
        }
        (void)builder.add_target(chain);
        const auto input = seal(std::move(builder));
        const auto result = passes::compile(input, passes::compile_level::c1);
        REQUIRE(result.ok());
        REQUIRE(result.output.node_count() == 1);
        REQUIRE(result.output.op_codes()[result.output.targets()[0].node.raw()] == pc::op_code::basic_event);
    }
}
