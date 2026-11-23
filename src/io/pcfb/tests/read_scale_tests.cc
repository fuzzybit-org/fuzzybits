#include <catch2/catch_test_macros.hpp>

#include "io/pcfb/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "portable_random.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace {

namespace pc = fuzzybits::ir::pc;
namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;

constexpr std::uint32_t scale_nodes = 100'000;
constexpr std::uint32_t scale_events = 4'000;
constexpr std::uint32_t scale_layers = 160;

// Layered DAG as in the pass scale tests: every gate takes one operand from the previous layer and the
// rest from anywhere below. With `shuffled`, node ids are a random permutation of the topological
// order, so the writer's depth-first order differs from the id order.
[[nodiscard]] auto layered(std::mt19937_64 &rng, const bool shuffled) -> pc::circuit {
    const auto uniform = [&](const std::uint32_t low, const std::uint32_t high) {
        return static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, low, high));
    };
    struct logical final {
        pc::op_code code;
        std::vector<pc::edge_ref> operands;
        std::uint32_t low = 0;
        std::uint32_t high = 0;
        double probability = 0.0;
    };
    std::vector<logical> nodes;
    for (std::uint32_t index = 0; index < scale_events; ++index) {
        nodes.push_back({.code = pc::op_code::basic_event, .probability = 1.0e-4 * (1 + uniform(0, 999))});
    }
    const auto gates_per_layer = (scale_nodes - scale_events - 1) / scale_layers;
    std::uint32_t previous_begin = 0;
    std::uint32_t previous_end = scale_events;
    for (std::uint32_t layer = 0; layer < scale_layers; ++layer) {
        const auto begin = static_cast<std::uint32_t>(nodes.size());
        for (std::uint32_t gate = 0; gate < gates_per_layer; ++gate) {
            logical node{.code = pc::op_code::or_gate};
            const auto fan_in = uniform(0, 19) == 0 ? 1U : uniform(2, 6);
            node.operands.push_back({.source = pc::bool_node_id(uniform(previous_begin, previous_end - 1))});
            for (std::uint32_t operand = 1; operand < fan_in; ++operand) {
                const auto source = uniform(0, 1) == 0 ? uniform(0, scale_events - 1) : uniform(0, begin - 1);
                node.operands.push_back({.source = pc::bool_node_id(source), .complement = uniform(0, 49) == 0});
            }
            switch (uniform(0, 9)) {
            case 0:
            case 1:
            case 2:
                node.code = pc::op_code::and_gate;
                break;
            case 3:
                if (node.operands.size() >= 3) {
                    node.code = pc::op_code::cardinality;
                    node.low = 2;
                    node.high = static_cast<std::uint32_t>(node.operands.size());
                }
                break;
            case 4:
                if (node.operands.size() >= 4) {
                    node.code = pc::op_code::cardinality;
                    node.low = 1;
                    node.high = 2;
                }
                break;
            default:
                break;
            }
            nodes.push_back(std::move(node));
        }
        previous_begin = begin;
        previous_end = static_cast<std::uint32_t>(nodes.size());
    }
    nodes.push_back({.code = pc::op_code::or_gate, .operands = {{.source = pc::bool_node_id(previous_end - 1)}}});

    std::vector<std::uint32_t> storage(nodes.size());
    std::iota(storage.begin(), storage.end(), 0U);
    if (shuffled) {
        pcfb_test_random::shuffle(std::span{storage}, rng);
    }
    std::vector<std::uint32_t> logical_at(nodes.size());
    for (std::uint32_t index = 0; index < nodes.size(); ++index) {
        logical_at[storage[index]] = index;
    }
    pc::builder builder;
    builder.reserve_nodes(static_cast<std::uint32_t>(nodes.size()));
    std::vector<pc::edge_ref> operands;
    for (std::uint32_t id = 0; id < nodes.size(); ++id) {
        auto const &node = nodes[logical_at[id]];
        operands.clear();
        for (auto const &edge : node.operands) {
            operands.push_back({.source = pc::bool_node_id(storage[edge.source.raw()]), .complement = edge.complement});
        }
        switch (node.code) {
        case pc::op_code::basic_event:
            (void)builder.add_basic_event(builder.add_literal_probability(node.probability));
            break;
        case pc::op_code::cardinality:
            (void)builder.add_cardinality(node.low, node.high, std::span<pc::edge_ref const>{operands});
            break;
        default:
            (void)builder.add_gate(node.code, std::span<pc::edge_ref const>{operands});
            break;
        }
    }
    (void)builder.add_target(pc::bool_node_id(storage[nodes.size() - 1]));
    auto sealed = std::move(builder).seal();
    REQUIRE(sealed.ok());
    return std::move(sealed.output);
}

} // namespace

TEST_CASE("pcfb reader reads a 1e5-node circuit in under a second", "[io][pcfb][read][scale]") {
    for (const bool shuffled : {false, true}) {
        std::mt19937_64 rng(0x219);
        const auto model = layered(rng, shuffled);
        REQUIRE(model.node_count() >= scale_nodes - scale_nodes / 100);
        // The pinned pcfb::builder::finish is quadratic in the operator count, so writing takes a while;
        // only the read is timed.
        auto written = writer::write(model.to_handle(), writer::write_options{});
        REQUIRE(written);
        const auto &bytes = written.value().bytes;

        const auto start = std::chrono::steady_clock::now();
        auto read = writer::read(bytes);
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        REQUIRE(read);
        std::cout << "pcfb read scale (" << (shuffled ? "shuffled ids" : "topological ids") << "): nodes "
                  << read.value().circuit.node_count() << ", file " << bytes.size() << " bytes, " << elapsed << " s\n";
        CHECK(read.value().circuit.node_count() >= model.node_count());
        REQUIRE(elapsed < 1.0);

        auto again = writer::write(read.value().circuit.to_handle(), read.value().options);
        REQUIRE(again);
        auto reread = writer::read(again.value().bytes);
        REQUIRE(reread);
        CHECK(reread.value().circuit.node_count() == read.value().circuit.node_count());
        CHECK(reread.value().circuit.edge_count() == read.value().circuit.edge_count());
    }
}
