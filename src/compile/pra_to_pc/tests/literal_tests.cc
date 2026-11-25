#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include "compile/pra_to_pc/descriptor.h"
#include "ir/pc/op_code.h"
#include "ir/pra/builder.h"
#include "ir/pra/projection.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace pra_to_pc_test;

constexpr std::string_view divergence = "compile.pra_to_pc.complement-under-negation";
constexpr std::string_view complemented_initiator = "compile.pra_to_pc.complemented-initiator";

// A complemented basic-event operand marked as a literal, as the FTP reader
// writes `-B`.
[[nodiscard]] auto literal_of(const pra::basic_event_id event) -> pra::operand_spec {
    return {.event = event, .complement = true, .literal = true};
}

[[nodiscard]] auto hull(std::string name) -> lowering::target_request {
    return {.name = std::move(name), .kinds = {pc::query_kind::monotone_hull}};
}

[[nodiscard]] auto lower_ok(pra::model const &model, lowering::quantification_config const &config)
    -> fuzzybits::api::diagnostic_list {
    auto lowered = lowering::lower(model, config);
    INFO(render(lowered.diagnostics()));
    REQUIRE(lowered);
    return lowered.diagnostics();
}

// Stable names, in report order, of the events a divergence warning names.
[[nodiscard]] auto divergent(fuzzybits::api::diagnostic_list const &diagnostics) -> std::vector<std::string> {
    std::vector<std::string> out;
    for (auto const &item : diagnostics.items()) {
        if (item.code != divergence) {
            continue;
        }
        const auto at = item.message.find("basic event '");
        REQUIRE(at != std::string::npos);
        const auto begin = at + std::string_view("basic event '").size();
        out.push_back(item.message.substr(begin, item.message.find('\'', begin) - begin));
    }
    return out;
}

struct small_model final {
    pra::builder builder;
    pra::basic_event_id a;
    pra::basic_event_id b;
    pra::basic_event_id c;
    pra::basic_event_id d;
    pra::basic_event_id e;

    small_model() {
        a = builder.add_basic_event({.stable_name = "A"}, pra::expression_spec::literal(0.1));
        b = builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::literal(0.2));
        c = builder.add_basic_event({.stable_name = "C"}, pra::expression_spec::literal(0.3));
        d = builder.add_basic_event({.stable_name = "D"}, pra::expression_spec::literal(0.4));
        e = builder.add_basic_event({.stable_name = "E"}, pra::expression_spec::literal(0.5));
    }

    [[nodiscard]] auto seal_with_target(const pra::domain_ref top) && -> pra::model {
        (void)builder.add_target({.stable_name = "T"}, pra::domain_target_kind::gate, top);
        return seal(std::move(builder));
    }
};

// T = (A and not(B or E)) or (D and -B), with -B marked or not.
[[nodiscard]] auto negated_and_direct(const bool marked) -> pra::model {
    small_model m;
    const auto either =
        m.builder.add_gate({.stable_name = "B_OR_E"}, pra::gate_kind::or_gate, {{.event = m.b}, {.event = m.e}});
    const auto left = m.builder.add_gate({.stable_name = "L"}, pra::gate_kind::and_gate,
                                         {{.event = m.a}, {.event = either, .complement = true}});
    const auto right = m.builder.add_gate({.stable_name = "R"}, pra::gate_kind::and_gate,
                                          {{.event = m.d}, {.event = m.b, .complement = true, .literal = marked}});
    const auto top =
        m.builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = left}, {.event = right}});
    return std::move(m).seal_with_target(top);
}

// T = or(and(A, -B), context(B, C)).
[[nodiscard]] auto with_context(const pra::gate_kind kind,
                                const std::optional<pra::cardinality_bounds> bounds = std::nullopt) -> pra::model {
    small_model m;
    const auto direct =
        m.builder.add_gate({.stable_name = "DIRECT"}, pra::gate_kind::and_gate, {{.event = m.a}, literal_of(m.b)});
    const std::vector<pra::operand_spec> operands =
        kind == pra::gate_kind::not_gate ? std::vector<pra::operand_spec>{{.event = m.b}}
                                         : std::vector<pra::operand_spec>{{.event = m.b}, {.event = m.c}};
    const auto context = bounds ? m.builder.add_cardinality({.stable_name = "CONTEXT"}, *bounds, operands)
                                : m.builder.add_gate({.stable_name = "CONTEXT"}, kind, operands);
    const auto top =
        m.builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = direct}, {.event = context}});
    return std::move(m).seal_with_target(top);
}

} // namespace

TEST_CASE("pra_to_pc records the hull reading on hull queries only", "[compile][pra_to_pc][literals]") {
    small_model m;
    const auto g =
        m.builder.add_gate({.stable_name = "G"}, pra::gate_kind::and_gate, {{.event = m.a}, literal_of(m.b)});
    (void)m.builder.add_target({.stable_name = "U"}, pra::domain_target_kind::gate, g);
    const auto model = std::move(m).seal_with_target(g);

    auto artifact = require_lower(model);
    for (auto const &target : artifact.projection.target_bindings()) {
        CHECK(target.kind == pc::query_kind::probability);
        CHECK_FALSE(target.hull);
    }

    lowering::quantification_config config;
    config.default_hull.literals = pra::hull_literals::fixed;
    config.targets = {hull("T"),
                      {.name = "U",
                       .kinds = {pc::query_kind::monotone_hull},
                       .hull = pra::hull_query{.literals = pra::hull_literals::plain}}};
    config.watch_targets = {{.name = "A", .element = pra::element_kind::basic_event}};
    artifact = require_lower(model, config);
    auto const bindings = artifact.projection.target_bindings();
    REQUIRE(bindings.size() == 3);
    REQUIRE(bindings[0].hull);
    CHECK(bindings[0].hull->literals == pra::hull_literals::fixed);
    REQUIRE(bindings[1].hull);
    CHECK(bindings[1].hull->literals == pra::hull_literals::plain);
    CHECK(bindings[2].kind == pc::query_kind::probability);
    CHECK_FALSE(bindings[2].hull);
}

TEST_CASE("pra_to_pc warns where the literal modes can differ", "[compile][pra_to_pc][literals]") {
    lowering::quantification_config config;
    config.targets = {hull("T")};

    SECTION("A not(B or E) or D -B with -B marked warns for B") {
        const auto model = negated_and_direct(true);
        const auto diagnostics = lower_ok(model, config);
        CHECK(divergent(diagnostics) == std::vector<std::string>{"B"});
        for (auto const &item : diagnostics.items()) {
            if (item.code == divergence) {
                CHECK(item.severity == fuzzybits::api::diagnostic_severity::warning);
                CHECK(item.message.find("target 'T'") != std::string::npos);
            }
        }
        // Boolean targets read no literal mode.
        CHECK(count(lower_ok(model, {}), divergence) == 0);
    }

    SECTION("the same model with -B unmarked has no literals and does not warn") {
        CHECK(count(lower_ok(negated_and_direct(false), config), divergence) == 0);
    }

    SECTION("AB or C not B does not warn") {
        small_model m;
        const auto both =
            m.builder.add_gate({.stable_name = "AB"}, pra::gate_kind::and_gate, {{.event = m.a}, {.event = m.b}});
        const auto other =
            m.builder.add_gate({.stable_name = "C_NOT_B"}, pra::gate_kind::and_gate, {{.event = m.c}, literal_of(m.b)});
        const auto top =
            m.builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = both}, {.event = other}});
        const auto model = std::move(m).seal_with_target(top);
        CHECK(divergent(lower_ok(model, config)).empty());
    }

    SECTION("A not(A not B) warns for B") {
        small_model m;
        const auto inner =
            m.builder.add_gate({.stable_name = "INNER"}, pra::gate_kind::and_gate, {{.event = m.a}, literal_of(m.b)});
        const auto top = m.builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::and_gate,
                                            {{.event = m.a}, {.event = inner, .complement = true}});
        const auto model = std::move(m).seal_with_target(top);
        CHECK(divergent(lower_ok(model, config)) == std::vector<std::string>{"B"});
    }

    SECTION("non-monotone gate contexts warn") {
        for (const auto kind : {pra::gate_kind::not_gate, pra::gate_kind::nand_gate, pra::gate_kind::nor_gate,
                                pra::gate_kind::xor_gate, pra::gate_kind::xnor_gate}) {
            CAPTURE(static_cast<int>(kind));
            CHECK(divergent(lower_ok(with_context(kind), config)) == std::vector<std::string>{"B"});
        }
        CHECK(
            divergent(lower_ok(with_context(pra::gate_kind::cardinality, pra::cardinality_bounds{.low = 1, .high = 1}),
                               config)) == std::vector<std::string>{"B"});
    }

    SECTION("monotone gate contexts do not warn") {
        for (const auto kind : {pra::gate_kind::and_gate, pra::gate_kind::or_gate}) {
            CAPTURE(static_cast<int>(kind));
            CHECK(divergent(lower_ok(with_context(kind), config)).empty());
        }
        CHECK(
            divergent(lower_ok(with_context(pra::gate_kind::cardinality, pra::cardinality_bounds{.low = 1, .high = 2}),
                               config))
                .empty());
        CHECK(
            divergent(lower_ok(with_context(pra::gate_kind::cardinality, pra::cardinality_bounds{.low = 0, .high = 2}),
                               config))
                .empty());
    }

    SECTION("a shared gate first reached monotonically is rescanned below a negation") {
        small_model m;
        const auto shared =
            m.builder.add_gate({.stable_name = "SHARED"}, pra::gate_kind::and_gate, {{.event = m.c}, {.event = m.b}});
        const auto direct = m.builder.add_gate({.stable_name = "DIRECT"}, pra::gate_kind::and_gate,
                                               {{.event = m.a}, literal_of(m.b), {.event = shared}});
        const auto negated = m.builder.add_gate({.stable_name = "NEG"}, pra::gate_kind::or_gate,
                                                {{.event = shared, .complement = true}});
        const auto top = m.builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate,
                                            {{.event = negated}, {.event = direct}});
        const auto model = std::move(m).seal_with_target(top);
        CHECK(divergent(lower_ok(model, config)) == std::vector<std::string>{"B"});
    }

    SECTION("once per hull target and event; overridden subtrees do not count") {
        small_model m;
        const auto right =
            m.builder.add_gate({.stable_name = "R"}, pra::gate_kind::and_gate, {{.event = m.d}, literal_of(m.b)});
        const auto again =
            m.builder.add_gate({.stable_name = "R2"}, pra::gate_kind::and_gate, {{.event = m.e}, literal_of(m.b)});
        const auto g_not = m.builder.add_gate({.stable_name = "NOT_B"}, pra::gate_kind::not_gate, {{.event = m.b}});
        const auto g_nor =
            m.builder.add_gate({.stable_name = "NOR_B"}, pra::gate_kind::nor_gate, {{.event = m.b}, {.event = m.c}});
        const auto top = m.builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate,
                                            {{.event = right}, {.event = again}, {.event = g_not}, {.event = g_nor}});
        (void)m.builder.add_target({.stable_name = "U"}, pra::domain_target_kind::gate, top);
        (void)m.builder.add_target({.stable_name = "W"}, pra::domain_target_kind::gate, top);
        const auto model = std::move(m).seal_with_target(top);

        config.targets = {hull("T"), hull("U"), {.name = "W"}};
        CHECK(divergent(lower_ok(model, config)) == std::vector<std::string>{"B", "B"});

        config.targets = {hull("T")};
        config.overrides = {{.name = "NOT_B", .value = false}, {.name = "NOR_B", .value = true}};
        CHECK(divergent(lower_ok(model, config)).empty());
    }

    SECTION("the message list is capped") {
        pra::builder builder;
        const auto a = builder.add_basic_event({.stable_name = "A"}, pra::expression_spec::literal(0.5));
        std::vector<pra::operand_spec> direct{{.event = a}};
        std::vector<pra::operand_spec> negated;
        for (int index = 0; index < 40; ++index) {
            const auto event = builder.add_basic_event({.stable_name = "B" + std::to_string(index)},
                                                       pra::expression_spec::literal(0.5));
            direct.push_back(literal_of(event));
            negated.push_back({.event = event});
        }
        const auto left = builder.add_gate({.stable_name = "L"}, pra::gate_kind::and_gate, direct);
        const auto right = builder.add_gate({.stable_name = "R"}, pra::gate_kind::nor_gate, negated);
        const auto top =
            builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = left}, {.event = right}});
        (void)builder.add_target({.stable_name = "T"}, pra::domain_target_kind::gate, top);
        const auto model = seal(std::move(builder));
        const auto diagnostics = lower_ok(model, config);
        REQUIRE(count(diagnostics, divergence) == 33);
        CHECK(diagnostics.items().back().message.starts_with("8 further"));
    }
}

TEST_CASE("pra_to_pc warns about complemented initiators under the initiator rule", "[compile][pra_to_pc][literals]") {
    small_model m;
    (void)m.builder.add_initiating_event({.stable_name = "IE_A"}, m.a);
    (void)m.builder.add_initiating_event({.stable_name = "IE_C"}, m.c);
    const auto left =
        m.builder.add_gate({.stable_name = "L"}, pra::gate_kind::and_gate, {literal_of(m.a), {.event = m.b}});
    const auto right =
        m.builder.add_gate({.stable_name = "R"}, pra::gate_kind::and_gate, {{.event = m.c}, {.event = m.a}});
    const auto top =
        m.builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = left}, {.event = right}});
    (void)m.builder.add_target({.stable_name = "U"}, pra::domain_target_kind::gate, left);
    const auto model = std::move(m).seal_with_target(top);

    lowering::quantification_config config;
    config.initiators = lowering::initiator_values::probability;
    config.targets = {{.name = "T"}, {.name = "U"}};
    CHECK(count(lower_ok(model, config), complemented_initiator) == 0);

    config.default_hull.max_initiators = 1;
    // Probability queries get no rule, so there is nothing to warn about.
    CHECK(count(lower_ok(model, config), complemented_initiator) == 0);

    const auto ruled = pra::hull_query{.literals = pra::hull_literals::plain, .max_initiators = 1};
    config.targets = {{.name = "T", .kinds = {pc::query_kind::monotone_hull}, .hull = ruled},
                      {.name = "U", .kinds = {pc::query_kind::monotone_hull}, .hull = ruled}};
    const auto diagnostics = lower_ok(model, config);
    REQUIRE(count(diagnostics, complemented_initiator) == 1);
    for (auto const &item : diagnostics.items()) {
        if (item.code == complemented_initiator) {
            CHECK(item.message.find("'A'") != std::string::npos);
        }
    }

    config.targets = {{.name = "R", .element = pra::element_kind::gate}};
    CHECK(count(lower_ok(model, config), complemented_initiator) == 0);
}

TEST_CASE("pra_to_pc keeps gate negations on gates", "[compile][pra_to_pc][literals]") {
    small_model m;
    const auto g_not = m.builder.add_gate({.stable_name = "NOT_B"}, pra::gate_kind::not_gate, {{.event = m.b}});
    const auto single = m.builder.add_gate({.stable_name = "SINGLE"}, pra::gate_kind::or_gate, {{.event = m.c}});
    const auto g_nand = m.builder.add_gate({.stable_name = "NAND_D"}, pra::gate_kind::nand_gate, {{.event = m.d}});
    const auto top = m.builder.add_gate(
        {.stable_name = "TOP"}, pra::gate_kind::and_gate,
        {{.event = m.a}, {.event = g_not}, {.event = single, .complement = true}, {.event = g_nand}});
    const auto model = std::move(m).seal_with_target(top);

    for (const auto kind : {pc::query_kind::probability, pc::query_kind::monotone_hull}) {
        lowering::quantification_config config;
        config.targets = {{.name = "T", .kinds = {kind}}};
        const auto artifact = require_lower(model, config);
        require_equivalent(model, artifact, {});
        auto const &circuit = artifact.circuit;
        auto const &projection = artifact.projection;
        const auto node = [&](const pra::domain_ref element) { return projection.node_of(element); };

        CHECK(circuit.op_codes()[node(g_not).raw()] == pc::op_code::not_gate);
        REQUIRE(circuit.operands(node(g_not)).size() == 1);
        CHECK(circuit.operands(node(g_not))[0].source == node(m.b));
        CHECK_FALSE(circuit.operands(node(g_not))[0].complement);

        CHECK(circuit.op_codes()[node(single).raw()] == pc::op_code::or_gate);
        REQUIRE(circuit.operands(node(single)).size() == 1);
        CHECK_FALSE(circuit.operands(node(single))[0].complement);

        CHECK(circuit.op_codes()[node(g_nand).raw()] == pc::op_code::nand_gate);
        CHECK_FALSE(circuit.operands(node(g_nand))[0].complement);

        const auto top_operands = circuit.operands(node(top));
        REQUIRE(top_operands.size() == 4);
        CHECK(top_operands[1].source == node(g_not));
        CHECK_FALSE(top_operands[1].complement);
        CHECK(top_operands[2].source == node(single));
        CHECK(top_operands[2].complement);
        for (auto const &edge : top_operands) {
            CHECK((circuit.op_codes()[edge.source.raw()] != pc::op_code::basic_event || !edge.complement));
        }

        CHECK(projection.literal_events(projection.target_bindings()[0]).empty());
        CHECK(projection.literal_events().empty());
    }
}

TEST_CASE("pra_to_pc takes literal events only from marked operands", "[compile][pra_to_pc][literals]") {
    small_model m;
    (void)m.builder.add_initiating_event({.stable_name = "IE_C"}, m.c);
    // -B marked; -C and -D unmarked; E only below a complemented gate.
    const auto inner = m.builder.add_gate({.stable_name = "INNER"}, pra::gate_kind::or_gate, {{.event = m.e}});
    const auto left = m.builder.add_gate({.stable_name = "L"}, pra::gate_kind::and_gate,
                                         {{.event = m.a}, literal_of(m.b), {.event = m.c, .complement = true}});
    const auto right = m.builder.add_gate({.stable_name = "R"}, pra::gate_kind::and_gate,
                                          {{.event = m.d, .complement = true}, {.event = inner, .complement = true}});
    const auto top =
        m.builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = left}, {.event = right}});
    (void)m.builder.add_target({.stable_name = "U"}, pra::domain_target_kind::gate, right);
    const auto model = std::move(m).seal_with_target(top);

    lowering::quantification_config config;
    config.initiators = lowering::initiator_values::probability;
    config.targets = {hull("T"), hull("U")};
    config.default_hull.max_initiators = 1;
    auto lowered = lowering::lower(model, config);
    INFO(render(lowered.diagnostics()));
    REQUIRE(lowered);
    auto const &projection = lowered.value().projection;
    CHECK(events_of_span(projection.literal_events(projection.target_bindings()[0])) ==
          std::vector<pra::basic_event_id>{m.b});
    CHECK(projection.literal_events(projection.target_bindings()[1]).empty());
    CHECK(events_of_span(projection.literal_events()) == std::vector<pra::basic_event_id>{m.b});
    // C is an initiator with an unmarked complement only.
    CHECK(count(lowered.diagnostics(), complemented_initiator) == 0);
    CHECK(count(lowered.diagnostics(), divergence) == 0);
}

TEST_CASE("pra_to_pc rejects literal and unmarked complements of one event in a hull target",
          "[compile][pra_to_pc][literals]") {
    small_model m;
    const auto left =
        m.builder.add_gate({.stable_name = "L"}, pra::gate_kind::and_gate, {{.event = m.a}, literal_of(m.b)});
    const auto right = m.builder.add_gate({.stable_name = "R"}, pra::gate_kind::and_gate,
                                          {{.event = m.c}, {.event = m.b, .complement = true}});
    const auto top =
        m.builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = left}, {.event = right}});
    const auto model = std::move(m).seal_with_target(top);

    lowering::quantification_config config;
    config.targets = {hull("T")};
    for (const auto mode : {pra::hull_literals::split, pra::hull_literals::fixed}) {
        config.default_hull.literals = mode;
        auto lowered = lowering::lower(model, config);
        REQUIRE_FALSE(lowered);
        CHECK(count(lowered.diagnostics(), "compile.pra_to_pc.mixed-literal") == 1);
    }

    config.default_hull.literals = pra::hull_literals::plain;
    CHECK(count(lower_ok(model, config), "compile.pra_to_pc.mixed-literal") == 0);
    CHECK(count(lower_ok(model, lowering::quantification_config{}), "compile.pra_to_pc.mixed-literal") == 0);
}
