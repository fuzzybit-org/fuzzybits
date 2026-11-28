#include <catch2/catch_test_macros.hpp>

#include "pra_test_support.h"

#include "ir/pra/builder.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pra = fuzzybits::ir::pra;
using pra::test::codes;
using pra::test::message;
using pra::test::render;
using pra::test::topologically_ordered;

[[nodiscard]] auto gate_name(const std::uint32_t layer, const std::uint32_t index) -> std::string {
    return "G" + std::to_string(layer) + "_" + std::to_string(index);
}

// A chain G0 -> G1 -> ... -> G{length-1}, each also reading one basic
// event, defined in that order so every gate operand is a forward
// reference. With `closed`, the last gate reads G0 and the chain is one
// long cycle.
[[nodiscard]] auto seal_chain(const std::uint32_t length, const bool closed) -> pra::finalize_result {
    pra::builder builder;
    builder.reserve_gates(length, 2 * length);
    for (std::uint32_t index = 0; index < length; ++index) {
        const auto next = index + 1 < length ? "G" + std::to_string(index + 1) : std::string(closed ? "G0" : "");
        const auto name = "G" + std::to_string(index);
        if (next.empty()) {
            (void)builder.add_gate({.stable_name = name}, pra::gate_kind::or_gate, {pra::test::named("B")});
        } else {
            (void)builder.add_gate({.stable_name = name}, pra::gate_kind::or_gate, {pra::test::named(next), pra::test::named("B")});
        }
    }
    (void)builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::literal(0.5));
    (void)builder.add_target({.stable_name = "TOP"}, pra::domain_target_kind::gate, pra::reference::named("G0"));
    return std::move(builder).seal();
}

} // namespace

TEST_CASE("PRA builder seals a 1e5-gate DAG of depth >= 200 without recursion", "[ir][pra][scale]") {
    constexpr std::uint32_t layers = 250;
    constexpr std::uint32_t width = 400;
    constexpr std::uint32_t events = 2000;
    constexpr std::array kinds{pra::gate_kind::or_gate,   pra::gate_kind::and_gate, pra::gate_kind::cardinality, pra::gate_kind::xor_gate,
                               pra::gate_kind::nand_gate, pra::gate_kind::nor_gate, pra::gate_kind::xnor_gate,   pra::gate_kind::not_gate};

    const auto start = std::chrono::steady_clock::now();
    pra::builder builder;
    builder.reserve_gates(layers * width + 1, layers * width * 3 + width);
    builder.reserve_basic_events(events);
    const auto tree = builder.add_fault_tree({.stable_name = "FT"});

    // Gates are added top-down and events last, so every name is a forward reference.
    std::vector<std::string> names;
    std::vector<pra::operand_spec> operands;
    for (std::uint32_t index = 0; index < width; ++index) {
        names.push_back(gate_name(layers - 1, index));
    }
    for (auto const &name : names) {
        operands.push_back(pra::test::named(name));
    }
    const auto top = builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, operands, tree);

    for (std::uint32_t layer = layers; layer-- > 0;) {
        for (std::uint32_t index = 0; index < width; ++index) {
            const auto kind = kinds[(layer + index) % kinds.size()];
            const std::uint32_t fan_in = kind == pra::gate_kind::not_gate ? 1 : 3;
            names.clear();
            operands.clear();
            for (std::uint32_t operand = 0; operand < fan_in; ++operand) {
                names.push_back(layer == 0 ? "B" + std::to_string((index * 3 + operand * 7) % events)
                                           : gate_name(layer - 1, (index * 7 + operand * 13 + layer) % width));
            }
            for (std::uint32_t operand = 0; operand < fan_in; ++operand) {
                operands.push_back(pra::test::named(names[operand], (index + operand) % 5 == 0));
            }
            const auto name = gate_name(layer, index);
            if (kind == pra::gate_kind::cardinality) {
                (void)builder.add_cardinality({.stable_name = name}, {.low = 2, .high = 3}, operands, tree);
            } else {
                (void)builder.add_gate({.stable_name = name}, kind, operands, tree);
            }
        }
    }
    for (std::uint32_t event = 0; event < events; ++event) {
        (void)builder.add_basic_event({.stable_name = "B" + std::to_string(event)}, pra::expression_spec::literal(1e-3));
    }
    builder.add_fault_tree_top(tree, top);
    const auto target = builder.add_target({.stable_name = "FT"}, pra::domain_target_kind::fault_tree_top, tree);

    auto result = std::move(builder).seal();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    INFO(render(result.diagnostics));
    REQUIRE(result.ok());

    auto const &model = result.output;
    CHECK(model.gates().size() == layers * width + 1);
    CHECK(model.gates(tree).size() == layers * width + 1);
    CHECK(model.summary().gate_depth == layers + 1);
    CHECK(model.summary().gate_kinds_present.size() == kinds.size());
    CHECK(model.target_event(target) == pra::domain_ref(top));
    CHECK(model.topological_gates().back() == top);
    CHECK(topologically_ordered(model));
    CHECK(model.find_gate(gate_name(0, 17)).valid());

    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    CAPTURE(milliseconds);
#ifdef NDEBUG
    CHECK(elapsed < std::chrono::seconds(1));
#endif
}

TEST_CASE("PRA builder orders a 1e5-gate chain without recursion", "[ir][pra][scale]") {
    constexpr std::uint32_t length = 100'000;
    auto result = seal_chain(length, false);
    INFO(render(result.diagnostics));
    REQUIRE(result.ok());
    CHECK(result.output.summary().gate_depth == length);
    CHECK(result.output.topological_gates().front() == result.output.find_gate("G" + std::to_string(length - 1)));
    CHECK(result.output.topological_gates().back() == result.output.find_gate("G0"));
}

TEST_CASE("PRA builder reports a 1e5-gate cycle without recursion", "[ir][pra][scale]") {
    constexpr std::uint32_t length = 100'000;
    auto result = seal_chain(length, true);
    REQUIRE_FALSE(result.ok());
    REQUIRE(codes(result.diagnostics) == std::vector<std::string>{"ir.pra.cycle"});
    const auto text = message(result.diagnostics, "ir.pra.cycle");
    CHECK(text.find("cycle of length 100000: 'G0' -> 'G1' -> ") != std::string::npos);
    CHECK(text.find("-> ... -> 'G0'") != std::string::npos);
}
