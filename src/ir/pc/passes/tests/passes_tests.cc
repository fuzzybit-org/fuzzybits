#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

    using namespace pc_passes_test;

    constexpr std::array all_passes{
        passes::pass_id::fold_zero_probability,
        passes::pass_id::fold_unit_probability,
        passes::pass_id::propagate_constants,
        passes::pass_id::eliminate_single_input,
        passes::pass_id::coalesce,
    };

    [[nodiscard]] auto mode_for(const passes::pass_id pass) -> worlds {
        return passes::pass_semantics(pass) == passes::semantic_class::function_preserving ? worlds::all : worlds::almost_sure;
    }

    [[nodiscard]] auto require_pass(const passes::pass_id pass, pc::circuit const &input) -> passes::transform_result {
        auto result = passes::run_pass(pass, input);
        require_report(result, input);
        require_images(input, result.output, result.report.node_map, result.report.target_map, mode_for(pass));
        return result;
    }

    void require_pipeline(passes::pipeline_result const &result, pc::circuit const &input) {
        INFO(diagnostics_text(result.diagnostics));
        REQUIRE(result.ok());
        REQUIRE(result.nodes_before == input.node_count());
        REQUIRE(result.edges_before == input.edge_count());
        REQUIRE(result.nodes_after == result.output.node_count());
        REQUIRE(result.edges_after == result.output.edge_count());
        REQUIRE(result.edges_after <= result.edges_before);
        for (auto const &report : result.passes) {
            REQUIRE(report.edges_after <= report.edges_before);
        }
        require_images(input, result.output, result.node_map, result.target_map, worlds::almost_sure);
    }

    [[nodiscard]] auto all_hull(pc::circuit const &model) -> passes::pipeline_options {
        return passes::pipeline_options{.semantics = std::vector(model.targets().size(), passes::target_semantics::monotone_hull)};
    }

    [[nodiscard]] auto single_target(pc::builder &&builder, const pc::bool_node_id node) -> pc::circuit {
        (void)builder.add_target(node);
        return seal(std::move(builder));
    }

    [[nodiscard]] auto event(pc::builder &builder, const double p = 0.5) -> pc::bool_node_id {
        return builder.add_basic_event(builder.add_literal_probability(p));
    }

    [[nodiscard]] auto target_node(passes::transform_result const &result, const std::uint32_t target = 0) -> std::uint32_t {
        return result.output.targets()[target].node.raw();
    }

    [[nodiscard]] auto target_code(passes::transform_result const &result, const std::uint32_t target = 0) -> pc::op_code {
        return result.output.op_codes()[target_node(result, target)];
    }

} // namespace

TEST_CASE("Every pass and C1 preserve Boolean probability, images and hull on random circuits", "[ir][pc][passes]") {
    std::mt19937_64 rng(177);
    std::array<std::uint64_t, all_passes.size()> rewrites{};
    std::uint64_t absorbed = 0;
    std::uint64_t constant_targets = 0;
    for (std::uint32_t iteration = 0; iteration < 400; ++iteration) {
        INFO("iteration " << iteration);
        const auto input = random_circuit(rng);
        const auto boolean = exact_probabilities(input);
        const auto hull = hull_probabilities(input);

        for (std::size_t index = 0; index < all_passes.size(); ++index) {
            const auto pass = all_passes[index];
            INFO("pass " << passes::pass_name(pass));
            const auto result = require_pass(pass, input);
            rewrites[index] += result.report.rewrites;
            require_close(boolean, exact_probabilities(result.output));
            if (passes::pass_preserves(pass, passes::target_semantics::monotone_hull)) {
                require_close(hull, hull_probabilities(result.output));
            }
            for (auto const &image : result.report.node_map) {
                absorbed += image.kind == passes::image_kind::absorbed ? 1 : 0;
            }
        }

        const auto compiled = passes::compile(input, passes::compile_level::c1);
        require_pipeline(compiled, input);
        require_close(boolean, exact_probabilities(compiled.output));
        for (auto const &target : compiled.output.targets()) {
            constant_targets += compiled.output.op_codes()[target.node.raw()] == pc::op_code::constant ? 1 : 0;
        }

        const auto hull_compiled = passes::compile(input, passes::compile_level::c1, all_hull(input));
        require_pipeline(hull_compiled, input);
        REQUIRE(has_code(hull_compiled.diagnostics, "pc.pipeline.pass-skipped"));
        require_close(hull, hull_probabilities(hull_compiled.output));
        require_close(boolean, exact_probabilities(hull_compiled.output));

        const auto chained = passes::run_pipeline(input, all_passes);
        require_pipeline(chained, input);
        require_close(boolean, exact_probabilities(chained.output));
    }
    for (const auto count : rewrites) {
        REQUIRE(count > 0);
    }
    REQUIRE(absorbed > 0);
    REQUIRE(constant_targets > 0);
}

TEST_CASE("Delete-term pair of p = 1 flags: only fold_unit_probability changes the hull", "[ir][pc][passes][hull]") {
    // g = (and(A) | B | Z) & !(X & Y) & ((C | TRUE) & (A | B)), RFC §8.2 plus
    // structure for every pass: Boolean P = 0, hull P(A | B) = 0.625.
    pc::builder builder;
    const auto a = event(builder, 0.25);
    const auto b = event(builder, 0.5);
    const auto z = event(builder, 0.0);
    const auto x = event(builder, 1.0);
    const auto y = event(builder, 1.0);
    const auto c = event(builder, 0.3);
    const auto on = builder.add_house_event(true);
    const auto failure = builder.add_or({edge{.source = builder.add_and({edge{.source = a}})}, edge{.source = b}, edge{.source = z}});
    const auto deletion = builder.add_and({edge{.source = x}, edge{.source = y}});
    const auto context = builder.add_and({edge{.source = builder.add_or({edge{.source = c}, edge{.source = on}})},
                                          edge{.source = builder.add_or({edge{.source = a}, edge{.source = b}})}});
    const auto top = builder.add_and({edge{.source = failure}, edge{.source = deletion, .complement = true}, edge{.source = context}});
    const auto input = single_target(std::move(builder), top);

    REQUIRE(exact_probabilities(input)[0] == 0.0);
    REQUIRE(std::abs(hull_probabilities(input)[0] - 0.625) <= 1.0e-15);

    for (const auto pass : all_passes) {
        INFO("pass " << passes::pass_name(pass));
        const auto result = require_pass(pass, input);
        REQUIRE(result.report.rewrites > 0);
        REQUIRE(exact_probabilities(result.output)[0] == 0.0);
        const auto hull = hull_probabilities(result.output)[0];
        if (pass == passes::pass_id::fold_unit_probability) {
            REQUIRE(hull == 0.0);
            REQUIRE_FALSE(result.report.preserves(passes::target_semantics::monotone_hull));
            REQUIRE(result.report.preserves(passes::target_semantics::boolean));
        } else {
            REQUIRE(std::abs(hull - 0.625) <= 1.0e-15);
            REQUIRE(result.report.preserves(passes::target_semantics::monotone_hull));
        }
    }

    const passes::pass_id hull_safe[] = {passes::pass_id::fold_zero_probability, passes::pass_id::propagate_constants,
                                         passes::pass_id::eliminate_single_input, passes::pass_id::coalesce};
    const auto safe = passes::run_pipeline(input, hull_safe, all_hull(input));
    require_pipeline(safe, input);
    REQUIRE(std::abs(hull_probabilities(safe.output)[0] - 0.625) <= 1.0e-15);
    REQUIRE(safe.edges_after < safe.edges_before);

    const auto refused = passes::run_pipeline(input, all_passes, all_hull(input));
    REQUIRE_FALSE(refused.ok());
    REQUIRE_FALSE(refused.output.valid());
    REQUIRE(has_code(refused.diagnostics, "pc.pipeline.incompatible-pass"));
    REQUIRE(refused.passes.empty());

    const auto hull_c1 = passes::compile(input, passes::compile_level::c1, all_hull(input));
    require_pipeline(hull_c1, input);
    REQUIRE(has_code(hull_c1.diagnostics, "pc.pipeline.pass-skipped"));
    REQUIRE(std::abs(hull_probabilities(hull_c1.output)[0] - 0.625) <= 1.0e-15);

    const auto boolean_c1 = passes::compile(input, passes::compile_level::c1);
    require_pipeline(boolean_c1, input);
    REQUIRE(boolean_c1.output.node_count() == 1);
    REQUIRE(boolean_c1.output.op_codes()[0] == pc::op_code::constant);
    REQUIRE(boolean_c1.output.constant_values()[0] == 0);
    REQUIRE(boolean_c1.node_map[top.raw()].kind == passes::image_kind::constant);
}

TEST_CASE("Edges never increase on random DAGs with shared subtrees", "[ir][pc][passes][coalesce]") {
    std::mt19937_64 rng(147);
    std::uint64_t kept_shared = 0;
    for (std::uint32_t iteration = 0; iteration < 60; ++iteration) {
        INFO("iteration " << iteration);
        const auto uniform = [&](const std::uint32_t low, const std::uint32_t high) {
            return std::uniform_int_distribution<std::uint32_t>(low, high)(rng);
        };
        pc::builder builder;
        std::vector<pc::bool_node_id> nodes;
        const auto events = uniform(4, 12);
        for (std::uint32_t index = 0; index < events; ++index) {
            nodes.push_back(event(builder, 0.1 + 0.05 * index));
        }
        std::vector<edge> operands;
        const auto gates = uniform(20, 200);
        for (std::uint32_t gate = 0; gate < gates; ++gate) {
            operands.clear();
            const auto fan_in = uniform(1, 6);
            for (std::uint32_t operand = 0; operand < fan_in; ++operand) {
                const auto available = static_cast<std::uint32_t>(nodes.size());
                const auto source = available > events && uniform(0, 3) != 0 ? uniform(events, available - 1) : uniform(0, available - 1);
                operands.push_back(edge{.source = nodes[source], .complement = uniform(0, 9) == 0});
            }
            nodes.push_back(builder.add_gate(uniform(0, 3) == 0 ? pc::op_code::and_gate : pc::op_code::or_gate, operands));
        }
        (void)builder.add_target(nodes.back());
        (void)builder.add_target(nodes[uniform(events, static_cast<std::uint32_t>(nodes.size()) - 1)], pc::observable_role::watch);
        const auto input = seal(std::move(builder));

        const auto coalesced = require_pass(passes::pass_id::coalesce, input);
        for (std::uint32_t node = 0; node < input.node_count(); ++node) {
            if (coalesced.report.node_map[node].kind == passes::image_kind::reference && input.operand_counts()[node] > 2) {
                ++kept_shared;
            }
        }
        const auto compiled = passes::compile(input, passes::compile_level::c1);
        require_pipeline(compiled, input);
        require_close(exact_probabilities(input), exact_probabilities(compiled.output));
    }
    REQUIRE(kept_shared > 0);
}

TEST_CASE("coalesce inlines only when the edge count does not increase", "[ir][pc][passes][coalesce]") {
    SECTION("shared OR with three parents and four inputs is kept") {
        pc::builder builder;
        const auto shared = builder.add_or({edge{.source = event(builder)}, edge{.source = event(builder)}, edge{.source = event(builder)},
                                            edge{.source = event(builder)}});
        std::vector<edge> parents;
        for (std::uint32_t index = 0; index < 3; ++index) {
            parents.push_back(edge{.source = builder.add_or({edge{.source = shared}, edge{.source = event(builder)}})});
        }
        const auto top = builder.add_and(parents);
        const auto input = single_target(std::move(builder), top);
        const auto result = require_pass(passes::pass_id::coalesce, input);
        REQUIRE(result.report.rewrites == 0);
        REQUIRE(result.report.edges_after == result.report.edges_before);
        REQUIRE(result.report.node_map[shared.raw()].kind == passes::image_kind::reference);
    }
    SECTION("single-parent child is inlined") {
        pc::builder builder;
        const auto a = event(builder);
        const auto child = builder.add_and({edge{.source = event(builder)}, edge{.source = event(builder)}});
        const auto top = builder.add_and({edge{.source = a}, edge{.source = child}});
        const auto input = single_target(std::move(builder), top);
        const auto result = require_pass(passes::pass_id::coalesce, input);
        REQUIRE(result.report.node_map[child.raw()].kind == passes::image_kind::absorbed);
        REQUIRE(result.report.edges_after + 1 == result.report.edges_before);
        REQUIRE(result.output.operand_counts()[target_node(result)] == 3);
    }
    SECTION("two parents and two inputs keep the edge count") {
        pc::builder builder;
        const auto child = builder.add_or({edge{.source = event(builder)}, edge{.source = event(builder)}});
        const auto left = builder.add_or({edge{.source = child}, edge{.source = event(builder)}});
        const auto right = builder.add_gate(pc::op_code::nor_gate, {edge{.source = child}, edge{.source = event(builder)}});
        const auto top = builder.add_and({edge{.source = left}, edge{.source = right}});
        const auto input = single_target(std::move(builder), top);
        const auto result = require_pass(passes::pass_id::coalesce, input);
        REQUIRE(result.report.node_map[child.raw()].kind == passes::image_kind::absorbed);
        REQUIRE(result.report.edges_after == result.report.edges_before);
        REQUIRE(result.report.nodes_after + 1 == result.report.nodes_before);
    }
    SECTION("complemented references, targets and other kinds block inlining") {
        pc::builder builder;
        const auto negated = builder.add_and({edge{.source = event(builder)}, edge{.source = event(builder)}});
        const auto watched = builder.add_and({edge{.source = event(builder)}, edge{.source = event(builder)}});
        const auto other_kind = builder.add_or({edge{.source = event(builder)}, edge{.source = event(builder)}});
        const auto nand_child = builder.add_gate(pc::op_code::nand_gate, {edge{.source = event(builder)}, edge{.source = event(builder)}});
        const auto top = builder.add_and(
            {edge{.source = negated, .complement = true}, edge{.source = watched}, edge{.source = other_kind}, edge{.source = nand_child}});
        (void)builder.add_target(top);
        (void)builder.add_target(watched, pc::observable_role::watch);
        const auto input = seal(std::move(builder));
        const auto result = require_pass(passes::pass_id::coalesce, input);
        REQUIRE(result.report.rewrites == 0);
    }
    SECTION("NAND absorbs AND and XNOR absorbs XOR") {
        pc::builder builder;
        const auto conjunction = builder.add_and({edge{.source = event(builder)}, edge{.source = event(builder)}});
        const auto parity = builder.add_gate(pc::op_code::xor_gate, {edge{.source = event(builder)}, edge{.source = event(builder)}});
        const auto nand = builder.add_gate(pc::op_code::nand_gate, {edge{.source = conjunction}, edge{.source = event(builder)}});
        const auto xnor = builder.add_gate(pc::op_code::xnor_gate, {edge{.source = parity}, edge{.source = event(builder)}});
        const auto top = builder.add_or({edge{.source = nand}, edge{.source = xnor}});
        const auto input = single_target(std::move(builder), top);
        const auto result = require_pass(passes::pass_id::coalesce, input);
        REQUIRE(result.report.rewrites == 2);
        REQUIRE(result.report.edges_after + 2 == result.report.edges_before);
    }
}

TEST_CASE("propagate_constants folds every connective", "[ir][pc][passes][constants]") {
    struct expectation final {
        pc::op_code code;
        std::uint32_t fan_in;
        std::uint32_t low = 0;
        std::uint32_t high = 0;
    };
    const auto check = [](pc::circuit const &input, expectation const &expected) {
        const auto result = require_pass(passes::pass_id::propagate_constants, input);
        require_close(exact_probabilities(input), exact_probabilities(result.output));
        REQUIRE(target_code(result) == expected.code);
        const auto node = target_node(result);
        REQUIRE(result.output.operand_counts()[node] == expected.fan_in);
        if (expected.code == pc::op_code::cardinality) {
            REQUIRE(result.output.cardinality_low()[node] == expected.low);
            REQUIRE(result.output.cardinality_high()[node] == expected.high);
        }
        return result;
    };
    const auto constant_target = [](pc::circuit const &input, const bool value) {
        const auto result = require_pass(passes::pass_id::propagate_constants, input);
        REQUIRE(target_code(result) == pc::op_code::constant);
        REQUIRE((result.output.constant_values()[target_node(result)] != 0) == value);
        REQUIRE(result.output.node_count() == 1);
    };

    SECTION("AND / OR / NAND / NOR with constants, duplicates and x with not x") {
        for (const auto code : {pc::op_code::and_gate, pc::op_code::or_gate, pc::op_code::nand_gate, pc::op_code::nor_gate}) {
            const bool conjunction = code == pc::op_code::and_gate || code == pc::op_code::nand_gate;
            const bool negate = code == pc::op_code::nand_gate || code == pc::op_code::nor_gate;
            {
                pc::builder builder;
                const auto a = event(builder);
                const auto identity = builder.add_constant(conjunction);
                const auto top = builder.add_gate(
                    code, {edge{.source = a}, edge{.source = identity}, edge{.source = a}, edge{.source = event(builder)}});
                (void)check(single_target(std::move(builder), top), expectation{.code = code, .fan_in = 2});
            }
            {
                pc::builder builder;
                const auto absorbing = builder.add_house_event(!conjunction);
                const auto top = builder.add_gate(code, {edge{.source = event(builder)}, edge{.source = absorbing}});
                constant_target(single_target(std::move(builder), top), !conjunction != negate);
            }
            {
                pc::builder builder;
                const auto a = event(builder);
                const auto top =
                    builder.add_gate(code, {edge{.source = a}, edge{.source = event(builder)}, edge{.source = a, .complement = true}});
                constant_target(single_target(std::move(builder), top), !conjunction != negate);
            }
            {
                pc::builder builder;
                const auto identity = builder.add_constant(conjunction);
                const auto top = builder.add_gate(code, {edge{.source = identity}, edge{.source = identity}});
                constant_target(single_target(std::move(builder), top), conjunction != negate);
            }
        }
    }
    SECTION("NOT of a constant") {
        pc::builder builder;
        const auto top = builder.add_not(edge{.source = builder.add_constant(true)});
        constant_target(single_target(std::move(builder), top), false);
    }
    SECTION("XOR / XNOR parity") {
        {
            pc::builder builder;
            const auto a = event(builder);
            const auto top =
                builder.add_gate(pc::op_code::xor_gate, {edge{.source = a}, edge{.source = event(builder)}, edge{.source = a}});
            (void)check(single_target(std::move(builder), top), expectation{.code = pc::op_code::xor_gate, .fan_in = 1});
        }
        {
            pc::builder builder;
            const auto a = event(builder);
            const auto top = builder.add_gate(pc::op_code::xor_gate, {edge{.source = a}, edge{.source = a, .complement = true}});
            constant_target(single_target(std::move(builder), top), true);
        }
        {
            pc::builder builder;
            const auto a = event(builder);
            const auto top = builder.add_gate(pc::op_code::xnor_gate, {edge{.source = a}, edge{.source = a, .complement = true}});
            constant_target(single_target(std::move(builder), top), false);
        }
        {
            pc::builder builder;
            const auto a = event(builder);
            const auto top =
                builder.add_gate(pc::op_code::xnor_gate, {edge{.source = a, .complement = true}, edge{.source = builder.add_constant(true)},
                                                          edge{.source = event(builder), .complement = true}});
            (void)check(single_target(std::move(builder), top), expectation{.code = pc::op_code::xnor_gate, .fan_in = 2});
        }
        {
            pc::builder builder;
            const auto a = event(builder);
            const auto b = event(builder);
            const auto top = builder.add_gate(pc::op_code::xor_gate, {edge{.source = a, .complement = true}, edge{.source = b}});
            const auto result = check(single_target(std::move(builder), top), expectation{.code = pc::op_code::xor_gate, .fan_in = 2});
            REQUIRE(result.report.rewrites == 0);
        }
    }
    SECTION("cardinality shifts its bounds and degenerates to AND / OR / NAND / NOR") {
        const auto build = [](const std::uint32_t low, const std::uint32_t high, const std::uint32_t events, const std::uint32_t trues,
                              const std::uint32_t falses, const bool pair) {
            pc::builder builder;
            std::vector<edge> operands;
            for (std::uint32_t index = 0; index < events; ++index) {
                operands.push_back(edge{.source = event(builder)});
            }
            for (std::uint32_t index = 0; index < trues; ++index) {
                operands.push_back(edge{.source = builder.add_constant(true)});
            }
            for (std::uint32_t index = 0; index < falses; ++index) {
                operands.push_back(edge{.source = builder.add_constant(false)});
            }
            if (pair) {
                const auto x = event(builder);
                operands.push_back(edge{.source = x});
                operands.push_back(edge{.source = x, .complement = true});
            }
            const auto top = builder.add_cardinality(low, high, operands);
            return single_target(std::move(builder), top);
        };
        (void)check(build(2, 4, 3, 1, 0, false), expectation{.code = pc::op_code::or_gate, .fan_in = 3});
        (void)check(build(2, 3, 2, 0, 0, true), expectation{.code = pc::op_code::or_gate, .fan_in = 2});
        (void)check(build(2, 2, 2, 0, 0, true), expectation{.code = pc::op_code::cardinality, .fan_in = 2, .low = 1, .high = 1});
        (void)check(build(3, 4, 4, 0, 1, false), expectation{.code = pc::op_code::cardinality, .fan_in = 4, .low = 3, .high = 4});
        (void)check(build(3, 3, 3, 0, 2, false), expectation{.code = pc::op_code::and_gate, .fan_in = 3});
        (void)check(build(0, 0, 3, 0, 0, false), expectation{.code = pc::op_code::nor_gate, .fan_in = 3});
        (void)check(build(1, 3, 3, 1, 0, false), expectation{.code = pc::op_code::nand_gate, .fan_in = 3});
        (void)check(build(0, 1, 3, 0, 0, false), expectation{.code = pc::op_code::cardinality, .fan_in = 3, .low = 0, .high = 1});
        constant_target(build(3, 3, 2, 0, 1, false), false);
        constant_target(build(1, 2, 1, 0, 0, true), true);
        constant_target(build(0, 1, 2, 2, 0, false), false);
        constant_target(build(1, 4, 0, 1, 3, false), true);
    }
    SECTION("cardinality counts duplicate operands with multiplicity") {
        pc::builder builder;
        const auto a = event(builder, 0.3);
        const auto top = builder.add_cardinality(2, 2, {edge{.source = a}, edge{.source = a}, edge{.source = event(builder, 0.6)}});
        const auto input = single_target(std::move(builder), top);
        const auto result = check(input, expectation{.code = pc::op_code::cardinality, .fan_in = 3, .low = 2, .high = 2});
        REQUIRE(std::abs(exact_probabilities(result.output)[0] - 0.3 * 0.4) <= 1.0e-15);
    }
}

TEST_CASE("eliminate_single_input folds negations into edges", "[ir][pc][passes][single-input]") {
    pc::builder builder;
    const auto a = event(builder, 0.2);
    const auto b = event(builder, 0.7);
    const auto negation = builder.add_not(edge{.source = builder.add_not(edge{.source = builder.add_not(edge{.source = a})})});
    const auto exactly_one = builder.add_cardinality(1, 1, {edge{.source = b}});
    const auto none = builder.add_cardinality(0, 0, {edge{.source = b}});
    const auto consumer =
        builder.add_and({edge{.source = builder.add_gate(pc::op_code::nand_gate, {edge{.source = a}})}, edge{.source = exactly_one},
                         edge{.source = builder.add_gate(pc::op_code::xnor_gate, {edge{.source = none}})}});
    (void)builder.add_target(negation);
    (void)builder.add_target(consumer, pc::observable_role::watch);
    (void)builder.add_target(builder.add_or({edge{.source = b}}), pc::observable_role::projection_anchor);
    (void)builder.add_target(builder.add_gate(pc::op_code::nor_gate, {edge{.source = a}}), pc::observable_role::watch);
    const auto input = seal(std::move(builder));

    const auto result = require_pass(passes::pass_id::eliminate_single_input, input);
    require_close(exact_probabilities(input), exact_probabilities(result.output));
    REQUIRE(result.report.node_map[negation.raw()].kind == passes::image_kind::reference);
    REQUIRE(result.report.node_map[negation.raw()].edge.complement);
    REQUIRE(target_code(result, 0) == pc::op_code::not_gate);
    REQUIRE(target_node(result, 0) == target_node(result, 3));
    REQUIRE(target_code(result, 2) == pc::op_code::basic_event);
    const auto consumer_node = target_node(result, 1);
    REQUIRE(result.output.op_codes()[consumer_node] == pc::op_code::and_gate);
    const auto operands = result.output.operands(pc::bool_node_id(consumer_node));
    REQUIRE(operands.size() == 3);
    for (auto const &operand : operands) {
        REQUIRE(result.output.op_codes()[operand.source.raw()] == pc::op_code::basic_event);
    }
    REQUIRE(operands[0].complement);
    REQUIRE_FALSE(operands[1].complement);
    REQUIRE_FALSE(operands[2].complement);
    REQUIRE(result.report.edges_after < result.report.edges_before);
}

TEST_CASE("Folding passes fold only literal p = 0 / p = 1 events and state their class", "[ir][pc][passes][fold]") {
    pc::builder builder;
    const auto zero = event(builder, 0.0);
    const auto one = event(builder, 1.0);
    const auto named = builder.add_basic_event(builder.add_parameter_probability("P"));
    const auto tiny = event(builder, 1.0e-300);
    const auto top = builder.add_or({edge{.source = zero}, edge{.source = one}, edge{.source = named}, edge{.source = tiny}});
    const auto input = single_target(std::move(builder), top);

    const auto zero_result = passes::fold_zero_probability(input);
    require_report(zero_result, input);
    REQUIRE(zero_result.report.transform_id == "ir.pc.fold_zero_probability");
    REQUIRE(zero_result.report.semantics == passes::semantic_class::distribution_preserving);
    REQUIRE(zero_result.report.preserves(passes::target_semantics::boolean));
    REQUIRE(zero_result.report.preserves(passes::target_semantics::monotone_hull));
    REQUIRE(zero_result.report.rewrites == 1);
    const auto zero_image = zero_result.report.node_map[zero.raw()];
    REQUIRE(zero_result.output.op_codes()[zero_image.edge.source.raw()] == pc::op_code::constant);
    REQUIRE(zero_result.output.constant_values()[zero_image.edge.source.raw()] == 0);
    REQUIRE(zero_result.output.op_codes()[zero_result.report.node_map[named.raw()].edge.source.raw()] == pc::op_code::basic_event);
    REQUIRE(zero_result.output.op_codes()[zero_result.report.node_map[tiny.raw()].edge.source.raw()] == pc::op_code::basic_event);

    const auto unit_result = passes::fold_unit_probability(input);
    require_report(unit_result, input);
    REQUIRE(unit_result.report.semantics == passes::semantic_class::distribution_preserving);
    REQUIRE(unit_result.report.preserved_semantics == std::vector{passes::target_semantics::boolean});
    REQUIRE(unit_result.report.rewrites == 1);
    REQUIRE(unit_result.output.constant_values()[unit_result.report.node_map[one.raw()].edge.source.raw()] == 1);

    for (const auto pass : {passes::pass_id::propagate_constants, passes::pass_id::eliminate_single_input, passes::pass_id::coalesce}) {
        const auto result = passes::run_pass(pass, input);
        REQUIRE(result.report.semantics == passes::semantic_class::function_preserving);
        REQUIRE(result.report.preserves(passes::target_semantics::boolean));
        REQUIRE(result.report.preserves(passes::target_semantics::monotone_hull));
    }
}

TEST_CASE("Pipeline presets, maps and diagnostics", "[ir][pc][passes][pipeline]") {
    std::mt19937_64 rng(181);
    const auto input = random_circuit(rng, random_options{.max_events = 8, .max_gates = 30});

    SECTION("C0 is the identity") {
        const auto result = passes::compile(input, passes::compile_level::c0);
        require_pipeline(result, input);
        REQUIRE(result.passes.empty());
        REQUIRE(result.nodes_after == input.node_count());
        REQUIRE(result.edges_after == input.edge_count());
    }
    SECTION("C1 schedule and map retention") {
        const auto dropped = passes::compile(input, passes::compile_level::c1);
        require_pipeline(dropped, input);
        REQUIRE(dropped.passes.size() >= 5);
        REQUIRE(dropped.passes[0].transform_id == "ir.pc.fold_zero_probability");
        REQUIRE(dropped.passes[1].transform_id == "ir.pc.fold_unit_probability");
        REQUIRE((dropped.passes.size() - 2) % 3 == 0);
        for (auto const &report : dropped.passes) {
            REQUIRE(report.node_map.empty());
        }
        const auto kept = passes::compile(input, passes::compile_level::c1, passes::pipeline_options{.keep_pass_maps = true});
        require_pipeline(kept, input);
        auto composed = kept.passes.front().node_map;
        for (std::size_t index = 1; index < kept.passes.size(); ++index) {
            REQUIRE(kept.passes[index].node_map.size() == kept.passes[index - 1].nodes_after);
            composed = passes::compose(composed, kept.passes[index].node_map);
        }
        REQUIRE(composed.size() == kept.node_map.size());
        for (std::size_t node = 0; node < composed.size(); ++node) {
            REQUIRE(composed[node].kind == kept.node_map[node].kind);
            REQUIRE(composed[node].edge.source == kept.node_map[node].edge.source);
        }
        const auto one_round = passes::compile(input, passes::compile_level::c1, passes::pipeline_options{.max_rounds = 1});
        require_pipeline(one_round, input);
        REQUIRE(one_round.passes.size() == 5);
    }
    SECTION("mixed semantics skip fold_unit_probability in C1 and refuse it explicitly") {
        passes::pipeline_options options{.semantics = std::vector(input.targets().size(), passes::target_semantics::boolean)};
        options.semantics.back() = passes::target_semantics::monotone_hull;
        const auto compiled = passes::compile(input, passes::compile_level::c1, options);
        require_pipeline(compiled, input);
        REQUIRE(has_code(compiled.diagnostics, "pc.pipeline.pass-skipped"));
        for (auto const &report : compiled.passes) {
            REQUIRE(report.transform_id != "ir.pc.fold_unit_probability");
        }
        const passes::pass_id schedule[] = {passes::pass_id::propagate_constants, passes::pass_id::fold_unit_probability};
        const auto refused = passes::run_pipeline(input, schedule, options);
        REQUIRE_FALSE(refused.ok());
        REQUIRE(has_code(refused.diagnostics, "pc.pipeline.incompatible-pass"));
    }
    SECTION("semantics table must cover every target") {
        const auto result = passes::compile(
            input, passes::compile_level::c1,
            passes::pipeline_options{.semantics = std::vector(input.targets().size() + 1, passes::target_semantics::boolean)});
        REQUIRE_FALSE(result.ok());
        REQUIRE(has_code(result.diagnostics, "pc.pipeline.target-semantics-mismatch"));
    }
    SECTION("invalid input and circuits without targets") {
        const auto invalid = passes::coalesce(pc::circuit{});
        REQUIRE_FALSE(invalid.ok());
        REQUIRE(has_code(invalid.report.diagnostics, "pc.pass.invalid-input"));

        pc::builder builder;
        (void)builder.add_and({edge{.source = event(builder)}});
        const auto untargeted = seal(std::move(builder));
        for (const auto pass : all_passes) {
            const auto result = passes::run_pass(pass, untargeted);
            REQUIRE(result.ok());
            REQUIRE(has_code(result.report.diagnostics, "pc.pass.no-targets"));
            REQUIRE(result.output.node_count() == untargeted.node_count());
        }
    }
}
