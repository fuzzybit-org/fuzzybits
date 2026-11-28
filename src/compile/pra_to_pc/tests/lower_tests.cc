#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include "compile/pra_to_pc/descriptor.h"
#include "ir/pc/op_code.h"
#include "ir/pc/target.h"
#include "ir/pra/builder.h"
#include "ir/pra/projection.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace pra_to_pc_test;

struct random_case final {
    pra::model model;
    lowering::quantification_config config;
    std::vector<reference_override> overrides;
};

// Up to 8 basic events, every connective, complements on events and gates,
// house events, shared subgraphs, initiators, overrides and several targets.
[[nodiscard]] auto random_case_from(std::mt19937_64 &rng) -> random_case {
    const auto uniform = [&](const std::uint32_t low, const std::uint32_t high) { return std::uniform_int_distribution<std::uint32_t>(low, high)(rng); };
    const auto chance = [&](const double p) { return std::bernoulli_distribution(p)(rng); };
    const auto pick = [&](auto const &values) { return uniform(0, static_cast<std::uint32_t>(values.size()) - 1); };
    constexpr std::array kinds{pra::gate_kind::and_gate, pra::gate_kind::or_gate,   pra::gate_kind::cardinality, pra::gate_kind::not_gate,
                               pra::gate_kind::xor_gate, pra::gate_kind::nand_gate, pra::gate_kind::nor_gate,    pra::gate_kind::xnor_gate};
    constexpr std::array probabilities{0.0, 1.0, 0.5, 0.1, 0.9, 0.25};

    pra::builder builder;
    std::vector<pra::domain_ref> elements;
    std::vector<pra::domain_ref> gates;
    const auto event_count = uniform(1, 8);
    for (std::uint32_t index = 0; index < event_count; ++index) {
        const auto name = "B" + std::to_string(index);
        const auto event = builder.add_basic_event({.stable_name = name}, pra::expression_spec::literal(probabilities[pick(probabilities)]));
        elements.emplace_back(event);
        if (chance(0.3)) {
            (void)builder.add_initiating_event({.stable_name = "I" + std::to_string(index)}, event);
        }
    }
    const auto house_count = uniform(0, 2);
    for (std::uint32_t index = 0; index < house_count; ++index) {
        elements.emplace_back(builder.add_house_event({.stable_name = "H" + std::to_string(index)}, chance(0.5)));
    }
    const auto gate_count = uniform(1, 12);
    std::vector<pra::operand_spec> operands;
    for (std::uint32_t index = 0; index < gate_count; ++index) {
        const auto kind = kinds[pick(kinds)];
        const auto fan_in = kind == pra::gate_kind::not_gate ? 1U : uniform(1, 4);
        operands.clear();
        for (std::uint32_t operand = 0; operand < fan_in; ++operand) {
            const auto recent = std::min<std::uint32_t>(static_cast<std::uint32_t>(elements.size()), 4);
            const auto source = chance(0.5) ? static_cast<std::uint32_t>(elements.size()) - 1 - uniform(0, recent - 1) : pick(elements);
            const auto complement = chance(0.35);
            const auto literal = complement && elements[source].kind() == pra::element_kind::basic_event && chance(0.5);
            operands.push_back(pra::operand_spec{.event = elements[source], .complement = complement, .literal = literal});
        }
        const auto gate_name = "G" + std::to_string(index);
        const pra::element_spec spec{.stable_name = gate_name};
        pra::domain_ref gate;
        if (kind == pra::gate_kind::cardinality) {
            const auto low = uniform(0, fan_in);
            gate = builder.add_cardinality(spec, {.low = low, .high = uniform(low, fan_in)}, operands);
        } else {
            gate = builder.add_gate(spec, kind, operands);
        }
        elements.push_back(gate);
        gates.push_back(gate);
    }
    // Names must outlive seal(); element_spec only views them.
    const auto target_count = uniform(1, 3);
    std::vector<std::string> names;
    for (std::uint32_t index = 0; index < target_count; ++index) {
        names.push_back("T" + std::to_string(index));
    }
    for (std::uint32_t index = 0; index < target_count; ++index) {
        const auto role = index == 0 || chance(0.5) ? pra::target_role::primary : pra::target_role::watch;
        if (chance(0.2)) {
            (void)builder.add_target({.stable_name = names[index]}, pra::domain_target_kind::basic_event, elements[uniform(0, event_count - 1)], role);
        } else {
            (void)builder.add_target({.stable_name = names[index]}, pra::domain_target_kind::gate, gates[pick(gates)], role);
        }
    }
    random_case out{.model = seal(std::move(builder))};
    auto const &model = out.model;

    if (chance(0.3)) {
        const auto gate = gates[pick(gates)];
        out.config.watch_targets.push_back(lowering::target_request{.name = std::string(model.stable_name(gate)), .element = pra::element_kind::gate});
    }
    const auto override_count = uniform(0, 2);
    for (std::uint32_t index = 0; index < override_count; ++index) {
        const auto element = elements[pick(elements)];
        if (std::ranges::any_of(out.overrides, [element](auto const &item) { return item.element == element; })) {
            continue;
        }
        const bool value = chance(0.5);
        out.overrides.push_back(reference_override{.element = element, .value = value});
        out.config.overrides.push_back(lowering::override_request{.name = std::string(model.stable_name(element)), .value = value});
    }
    if (chance(0.6)) {
        out.config.default_hull.max_initiators = uniform(0, 2);
    }
    // Probability, hull or both: the rule goes into hull queries only, and a target with both kinds
    // keeps its probability query on the subject.
    const auto query_kinds = [&]() -> std::vector<pc::query_kind> {
        switch (uniform(0, 2)) {
        case 0:
            return {pc::query_kind::probability};
        case 1:
            return {pc::query_kind::monotone_hull};
        default:
            return {pc::query_kind::probability, pc::query_kind::monotone_hull};
        }
    };
    out.config.default_kinds = query_kinds();
    // The random models mix marked and unmarked complements of one event, which only the plain reading
    // of hull queries accepts (compile.pra_to_pc.mixed-literal); the other readings are tested elsewhere.
    out.config.default_hull.literals = pra::hull_literals::plain;
    for (auto &request : out.config.watch_targets) {
        request.kinds = query_kinds();
    }
    // Initiating-event values as probabilities: the initiator rule needs them.
    out.config.initiators = lowering::initiator_values::probability;
    return out;
}

} // namespace

TEST_CASE("pra_to_pc lowers every connective function-preservingly", "[compile][pra_to_pc]") {
    pra::builder builder;
    const auto a = builder.add_basic_event({.stable_name = "A"}, pra::expression_spec::literal(0.1));
    const auto b = builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::literal(0.2));
    const auto c = builder.add_basic_event({.stable_name = "C"}, pra::expression_spec::literal(0.3));
    const auto d = builder.add_basic_event({.stable_name = "D"}, pra::expression_spec::literal(0.4));
    const auto shared = builder.add_expression(pra::expression_spec::literal(0.5));
    const auto e = builder.add_basic_event({.stable_name = "E"}, shared);
    const auto f = builder.add_basic_event({.stable_name = "F"}, shared);
    const auto on = builder.add_house_event({.stable_name = "H_ON"}, true);
    const auto off = builder.add_house_event({.stable_name = "H_OFF"}, false);
    const auto tree = builder.add_fault_tree({.stable_name = "FT"});
    const auto g_and = builder.add_gate({.stable_name = "G_AND"}, pra::gate_kind::and_gate, {{.event = a}, {.event = b, .complement = true}}, tree);
    const auto g_or = builder.add_gate({.stable_name = "G_OR"}, pra::gate_kind::or_gate, {{.event = g_and}, {.event = on}, {.event = f}}, tree);
    const auto g_vote = builder.add_cardinality({.stable_name = "G_2OF3"}, {.low = 2, .high = 3}, {{.event = a}, {.event = b}, {.event = c}}, tree);
    const auto g_range =
        builder.add_cardinality({.stable_name = "G_RANGE"}, {.low = 1, .high = 2}, {{.event = a}, {.event = c, .complement = true}, {.event = d}}, tree);
    const auto g_not = builder.add_gate({.stable_name = "G_NOT"}, pra::gate_kind::not_gate, {{.event = g_vote}}, tree);
    const auto g_xor =
        builder.add_gate({.stable_name = "G_XOR"}, pra::gate_kind::xor_gate, {{.event = c}, {.event = d}, {.event = e, .complement = true}}, tree);
    const auto g_nand = builder.add_gate({.stable_name = "G_NAND"}, pra::gate_kind::nand_gate, {{.event = a}, {.event = g_or, .complement = true}}, tree);
    const auto g_nor = builder.add_gate({.stable_name = "G_NOR"}, pra::gate_kind::nor_gate, {{.event = d}, {.event = off}}, tree);
    const auto g_xnor =
        builder.add_gate({.stable_name = "G_XNOR"}, pra::gate_kind::xnor_gate, {{.event = b}, {.event = c, .complement = true}, {.event = g_nand}}, tree);
    const auto g_shared = builder.add_gate({.stable_name = "G_SHARED"}, pra::gate_kind::and_gate, {{.event = g_xor}, {.event = g_range}}, tree);
    const auto g_single = builder.add_gate({.stable_name = "G_SINGLE"}, pra::gate_kind::or_gate, {{.event = e, .complement = true}}, tree);
    const auto top = builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate,
                                      {{.event = g_or}, {.event = g_not}, {.event = g_nand}, {.event = g_nor}, {.event = g_xnor}, {.event = g_shared}}, tree);
    builder.add_fault_tree_top(tree, top);
    const auto side = builder.add_gate({.stable_name = "SIDE"}, pra::gate_kind::and_gate, {{.event = g_shared}, {.event = g_single, .complement = true}});
    const auto unused = builder.add_gate({.stable_name = "UNUSED"}, pra::gate_kind::or_gate, {{.event = a}});
    (void)builder.add_target({.stable_name = "FT"}, pra::domain_target_kind::fault_tree_top, tree);
    (void)builder.add_target({.stable_name = "SIDE"}, pra::domain_target_kind::gate, side, pra::target_role::watch);
    (void)builder.add_target({.stable_name = "A"}, pra::domain_target_kind::basic_event, a);
    const auto model = seal(std::move(builder));

    // Defaults are the primary targets; the watch-role domain target is requested.
    lowering::quantification_config config;
    config.watch_targets = {{.name = "SIDE"}};
    const auto lowered = require_lower(model, config);
    require_equivalent(model, lowered, {});

    auto const &circuit = lowered.circuit;
    auto const &projection = lowered.projection;
    REQUIRE(circuit.targets().size() == 3);
    CHECK(circuit.targets()[0].role == pc::observable_role::primary);
    CHECK(circuit.targets()[1].role == pc::observable_role::primary);
    CHECK(circuit.targets()[2].role == pc::observable_role::watch);
    CHECK(projection.target_bindings()[0].subject == pra::domain_ref(top));
    CHECK(projection.target_bindings()[1].subject == pra::domain_ref(a));
    CHECK(projection.target_bindings()[2].subject == pra::domain_ref(side));
    CHECK(require_lower(model).circuit.targets().size() == 2);

    // One node per element of the union cone: 6 events, 2 house events, 13 gates.
    CHECK(circuit.node_count() == 21);
    CHECK(projection.node_of(unused) == pc::bool_node_id::invalid());
    CHECK(lowered.report.source_gates == 13);
    CHECK(lowered.report.source_basic_events == 6);
    CHECK(lowered.report.source_house_events == 2);
    CHECK(lowered.report.pc_nodes == circuit.node_count());
    CHECK(lowered.report.pc_edges == circuit.edge_count());
    CHECK(lowered.report.source_operands == circuit.edge_count());
    CHECK(lowered.report.generated_nodes == 0);
    CHECK(lowered.report.semantics == lowering::semantic_class::function_preserving);

    // Shared PRA expressions stay shared probability sources.
    CHECK(circuit.probability_sources().size() == 5);
    CHECK(circuit.probability_source_refs()[projection.node_of(e).raw()] == circuit.probability_source_refs()[projection.node_of(f).raw()]);

    const auto code = [&](const pra::domain_ref element) { return circuit.op_codes()[projection.node_of(element).raw()]; };
    CHECK(code(g_and) == pc::op_code::and_gate);
    CHECK(code(g_or) == pc::op_code::or_gate);
    CHECK(code(g_vote) == pc::op_code::cardinality);
    CHECK(code(g_not) == pc::op_code::not_gate);
    CHECK(code(g_xor) == pc::op_code::xor_gate);
    CHECK(code(g_nand) == pc::op_code::nand_gate);
    CHECK(code(g_nor) == pc::op_code::nor_gate);
    CHECK(code(g_xnor) == pc::op_code::xnor_gate);
    CHECK(code(g_single) == pc::op_code::or_gate);
    CHECK(code(on) == pc::op_code::house_event);
    CHECK(code(a) == pc::op_code::basic_event);
    const auto range = projection.node_of(g_range).raw();
    CHECK(circuit.cardinality_low()[range] == 1);
    CHECK(circuit.cardinality_high()[range] == 2);
    // Complements stay on the edges they were on.
    const auto and_operands = circuit.operands(projection.node_of(g_and));
    REQUIRE(and_operands.size() == 2);
    CHECK_FALSE(and_operands[0].complement);
    CHECK(and_operands[1].complement);
    CHECK(and_operands[1].source == projection.node_of(b));
}

TEST_CASE("pra_to_pc matches brute-force PRA semantics on random models", "[compile][pra_to_pc][random]") {
    std::mt19937_64 rng(0x181);
    std::size_t with_rule = 0;
    std::size_t with_overrides = 0;
    std::size_t with_delete_terms = 0;
    std::size_t with_split = 0;
    std::size_t with_complements = 0;
    for (int iteration = 0; iteration < 400; ++iteration) {
        CAPTURE(iteration);
        const auto sample = random_case_from(rng);
        const auto lowered = require_lower(sample.model, sample.config);
        require_equivalent(sample.model, lowered, sample.overrides);

        const pra_reference reference(sample.model, sample.overrides);
        for (auto const &target : lowered.projection.target_bindings()) {
            CHECK(events_of_span(lowered.projection.literal_events(target)) == reference.cone_events(target.subject, true));
            CHECK(events_of_span(lowered.projection.initiators(target)) == reference.cone_initiators(target.subject));
            with_delete_terms += target.initiator_delete_term.valid() ? 1 : 0;
            with_split += target.kind == pc::query_kind::probability && sample.config.default_hull.max_initiators &&
                                  std::ranges::any_of(lowered.projection.target_bindings(),
                                                      [&](auto const &other) {
                                                          return other.pc_target != target.pc_target && other.subject == target.subject &&
                                                                 other.initiator_delete_term.valid();
                                                      })
                              ? 1
                              : 0;
            with_complements += target.literal_events.count != 0 ? 1 : 0;
        }
        with_rule += sample.config.default_hull.max_initiators ? 1 : 0;
        with_overrides += sample.overrides.empty() ? 0 : 1;
    }
    CHECK(with_rule > 100);
    CHECK(with_overrides > 100);
    CHECK(with_delete_terms > 20);
    // Targets asking for both kinds under the rule: the probability query on the subject, the hull query
    // on the conjunction.
    CHECK(with_split > 5);
    CHECK(with_complements > 100);
}

TEST_CASE("pra_to_pc overrides become constants and cut the cone", "[compile][pra_to_pc][override]") {
    pra::builder builder;
    const auto a = builder.add_basic_event({.stable_name = "A"}, pra::expression_spec::literal(0.1));
    const auto b = builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::literal(1.0));
    const auto hidden = builder.add_basic_event({.stable_name = "HIDDEN"}, pra::expression_spec::parameter("lambda"));
    const auto house = builder.add_house_event({.stable_name = "H"}, true);
    const auto inner = builder.add_gate({.stable_name = "INNER"}, pra::gate_kind::and_gate, {{.event = hidden}, {.event = a}});
    const auto top = builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = inner}, {.event = b, .complement = true}, {.event = house}});
    const auto other = builder.add_gate({.stable_name = "OTHER"}, pra::gate_kind::and_gate, {{.event = a}});
    (void)builder.add_target({.stable_name = "TOP"}, pra::domain_target_kind::gate, top);
    // A gate and a basic event share the name "X".
    (void)builder.add_gate({.stable_name = "X"}, pra::gate_kind::or_gate, {{.event = a}});
    (void)builder.add_basic_event({.stable_name = "X"}, pra::expression_spec::literal(0.5));
    const auto model = seal(std::move(builder));

    SECTION("gate, basic event and house event overrides") {
        lowering::quantification_config config;
        config.overrides = {
            {.name = "INNER", .value = false},
            {.name = "B", .value = true},
            {.name = "H", .value = false},
            {.name = "OTHER", .value = true},
        };
        auto lowered = lowering::lower(model, config);
        INFO(render(lowered.diagnostics()));
        REQUIRE(lowered);
        // INNER's parameter-probability operand left the cone, so it is not checked.
        CHECK(codes(lowered.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.unused-override"});
        auto const &artifact = lowered.value();
        require_equivalent(
            model, artifact,
            {{.element = inner, .value = false}, {.element = b, .value = true}, {.element = house, .value = false}, {.element = other, .value = true}});

        auto const &projection = artifact.projection;
        for (const pra::domain_ref element : {pra::domain_ref(inner), pra::domain_ref(b), pra::domain_ref(house)}) {
            auto const *binding = projection.find_event(element);
            REQUIRE(binding != nullptr);
            CHECK(binding->kind == pra::event_binding_kind::constant_override);
            CHECK(artifact.circuit.op_codes()[binding->node.raw()] == pc::op_code::constant);
        }
        CHECK(artifact.circuit.constant_values()[projection.node_of(b).raw()] == 1);
        CHECK(projection.node_of(hidden) == pc::bool_node_id::invalid());
        CHECK(projection.node_of(a) == pc::bool_node_id::invalid());
        REQUIRE(projection.override_bindings().size() == 4);
        CHECK(projection.override_bindings()[0].element == pra::domain_ref(inner));
        CHECK(projection.override_bindings()[3].element == pra::domain_ref(other));
        CHECK_FALSE(projection.override_bindings()[3].node.valid());
        CHECK(artifact.report.overridden_elements == 3);
        // An overridden event is a constant, not a significant literal.
        CHECK(projection.literal_events(projection.target_bindings()[0]).empty());
    }

    SECTION("kinds disambiguate names and resolution errors are reported") {
        lowering::quantification_config config;
        config.overrides = {
            {.name = "X", .value = true},
            {.name = "NOPE", .value = true},
            {.name = "A", .kind = pra::element_kind::domain_target, .value = true},
            {.name = "A", .value = true},
            {.name = "A", .kind = pra::element_kind::basic_event, .value = false},
        };
        auto lowered = lowering::lower(model, config);
        REQUIRE_FALSE(lowered);
        CHECK(codes(lowered.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.ambiguous-element", "compile.pra_to_pc.unknown-element",
                                                                       "compile.pra_to_pc.invalid-override", "compile.pra_to_pc.duplicate-override"});

        lowering::quantification_config qualified;
        qualified.overrides = {{.name = "X", .kind = pra::element_kind::basic_event, .value = true}, {.name = "INNER", .value = false}};
        auto ok = lowering::lower(model, qualified);
        REQUIRE(ok.has_value());
        CHECK(count(ok.diagnostics(), "compile.pra_to_pc.unused-override") == 1);
    }

    SECTION("without overrides the parameter probability is an error") {
        auto lowered = lowering::lower(model);
        REQUIRE_FALSE(lowered);
        CHECK(codes(lowered.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.parameter-probability"});
    }
}

TEST_CASE("pra_to_pc initiator rule deletes exactly the worlds with too many initiators", "[compile][pra_to_pc][initiators]") {
    pra::builder builder;
    const auto i1 = builder.add_basic_event({.stable_name = "I1"}, pra::expression_spec::literal(0.5, pra::quantity_kind::frequency));
    const auto i2 = builder.add_basic_event({.stable_name = "I2"}, pra::expression_spec::literal(0.2));
    const auto i3 = builder.add_basic_event({.stable_name = "I3"}, pra::expression_spec::literal(1.0));
    const auto b = builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::literal(0.3));
    const auto i4 = builder.add_basic_event({.stable_name = "I4"}, pra::expression_spec::literal(0.4));
    for (const auto event : {i1, i2, i3, i4}) {
        (void)builder.add_initiating_event({.stable_name = "IE-" + std::to_string(event.raw())}, event);
    }
    const auto both = builder.add_gate({.stable_name = "I3_B"}, pra::gate_kind::and_gate, {{.event = i3}, {.event = b}});
    const auto top = builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = i1}, {.event = i2}, {.event = both}});
    const auto single = builder.add_gate({.stable_name = "SINGLE"}, pra::gate_kind::and_gate, {{.event = i1}, {.event = b}});
    (void)builder.add_gate({.stable_name = "ELSEWHERE"}, pra::gate_kind::or_gate, {{.event = i4}});
    (void)builder.add_target({.stable_name = "TOP"}, pra::domain_target_kind::gate, top);
    (void)builder.add_target({.stable_name = "SINGLE"}, pra::domain_target_kind::gate, single);
    const auto model = seal(std::move(builder));
    const pra_reference reference(model, {});

    // The values are read as probabilities here; the lowering reads them as frequencies by default.
    lowering::quantification_config probabilities;
    probabilities.initiators = lowering::initiator_values::probability;

    SECTION("rule unset: the targets are the subjects") {
        auto lowered = lowering::lower(model, probabilities);
        REQUIRE(lowered);
        CHECK(codes(lowered.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.frequency-as-probability"});
        auto const &artifact = lowered.value();
        require_equivalent(model, artifact, {});
        CHECK(artifact.projection.generated_bindings().empty());
        for (auto const &target : artifact.projection.target_bindings()) {
            CHECK(target.node == target.root);
            CHECK_FALSE(target.initiator_delete_term.valid());
        }
        CHECK(events_of_span(artifact.projection.initiators(artifact.projection.target_bindings()[0])) == std::vector<pra::basic_event_id>{i1, i2, i3});
    }

    SECTION("frequencies need a unit in the lowering config") {
        // The FTP driver supplies per-year by default; direct callers of the lowering name the unit.
        lowering::quantification_config config;
        CHECK(config.initiators == lowering::initiator_values::frequency);
        auto unitless = lowering::lower(model, config);
        REQUIRE_FALSE(unitless);
        CHECK(count(unitless.diagnostics(), "compile.pra_to_pc.missing-initiator-unit") == 1);
        config.initiator_unit = "per-hour";
        const auto artifact = require_lower(model, config);
        CHECK(artifact.report.initiator_unit == "per-hour");
    }

    SECTION("at most one initiator") {
        auto config = probabilities;
        config.default_hull.max_initiators = 1;
        config.default_kinds = {pc::query_kind::monotone_hull};
        const auto artifact = require_lower(model, config);
        require_equivalent(model, artifact, {});

        auto const &top_binding = artifact.projection.target_bindings()[0];
        REQUIRE(top_binding.initiator_delete_term.valid());
        const auto term = top_binding.initiator_delete_term.raw();
        CHECK(artifact.circuit.op_codes()[term] == pc::op_code::cardinality);
        CHECK(artifact.circuit.cardinality_low()[term] == 2);
        CHECK(artifact.circuit.cardinality_high()[term] == 3);
        const auto conjunction = artifact.circuit.operands(top_binding.node);
        REQUIRE(conjunction.size() == 2);
        CHECK(conjunction[0].source == top_binding.root);
        CHECK_FALSE(conjunction[0].complement);
        CHECK(conjunction[1].source == top_binding.initiator_delete_term);
        CHECK(conjunction[1].complement);
        REQUIRE(artifact.projection.generated_bindings().size() == 2);
        CHECK(artifact.projection.find_generated(top_binding.initiator_delete_term)->kind == pra::generated_kind::initiator_delete_term);
        CHECK(artifact.projection.find_generated(top_binding.node)->kind == pra::generated_kind::initiator_rule_conjunction);
        CHECK(artifact.projection.find_generated(top_binding.node)->target == top_binding.pc_target);
        CHECK(artifact.report.generated_nodes == 2);
        CHECK(artifact.report.generated_edges == 5);

        // SINGLE's cone holds one initiator: nothing to delete.
        auto const &single_binding = artifact.projection.target_bindings()[1];
        CHECK(single_binding.node == single_binding.root);
        CHECK(events_of_span(artifact.projection.initiators(single_binding)) == std::vector<pra::basic_event_id>{i1});

        // The removed worlds are exactly those of TOP with >= 2 of I1, I2, I3
        // true; I4 is outside the cone and does not count.
        const pc_reference circuit(artifact.circuit);
        std::size_t removed = 0;
        for (std::uint64_t world = 0; world < 32; ++world) {
            std::vector<std::uint8_t> values(artifact.circuit.node_count(), 0);
            for (const auto event : {i1, i2, i3, b, i4}) {
                if (const auto node = artifact.projection.node_of(event); node.valid()) {
                    values[node.raw()] = ((world >> event.raw()) & 1U) != 0 ? 1 : 0;
                }
            }
            const auto value = circuit.values(values);
            const auto initiators = std::popcount(world & 0b0111U);
            const bool subject = reference.value(top, world);
            CHECK((value[top_binding.root.raw()] != 0) == subject);
            CHECK((value[top_binding.node.raw()] != 0) == (subject && initiators <= 1));
            removed += subject && value[top_binding.node.raw()] == 0 ? 1 : 0;
            CHECK((subject && value[top_binding.node.raw()] == 0) == (subject && initiators >= 2));
        }
        // 4 initiator patterns with >= 2 of 3 set, each with B and I4 free.
        CHECK(removed == 4 * 4);
    }

    SECTION("the rule goes into hull queries only") {
        auto config = probabilities;
        config.default_hull.max_initiators = 1;
        // Probability queries: the rule would remove assignments, so it is not lowered.
        const auto probability = require_lower(model, config);
        require_equivalent(model, probability, {});
        CHECK(probability.projection.generated_bindings().empty());
        CHECK(probability.projection.target_bindings().size() == 2);
        for (auto const &target : probability.projection.target_bindings()) {
            CHECK(target.node == target.root);
            CHECK_FALSE(target.initiator_delete_term.valid());
        }

        // Both kinds: TOP's probability query stays on its subject and its hull query gets the rule, on two
        // nodes; SINGLE has one initiator, nothing to delete, so its two queries share one node.
        config.default_kinds = {pc::query_kind::probability, pc::query_kind::monotone_hull};
        const auto both_kinds = require_lower(model, config);
        require_equivalent(model, both_kinds, {});
        auto const bindings = both_kinds.projection.target_bindings();
        REQUIRE(bindings.size() == 4);
        CHECK(bindings[0].subject == pra::domain_ref(top));
        CHECK(bindings[0].kind == pc::query_kind::probability);
        CHECK_FALSE(bindings[0].hull);
        CHECK(bindings[0].node == bindings[0].root);
        CHECK(bindings[1].subject == pra::domain_ref(top));
        CHECK(bindings[1].kind == pc::query_kind::monotone_hull);
        CHECK(bindings[1].hull == pra::hull_query{.literals = pra::hull_literals::split, .max_initiators = 1});
        CHECK(bindings[1].initiator_delete_term.valid());
        CHECK(bindings[1].root == bindings[0].root);
        CHECK(bindings[1].node != bindings[1].root);
        // SINGLE: one query per kind, both on its subject.
        CHECK(bindings[2].subject == pra::domain_ref(single));
        CHECK(bindings[2].kind == pc::query_kind::probability);
        CHECK(bindings[3].subject == pra::domain_ref(single));
        CHECK(bindings[3].kind == pc::query_kind::monotone_hull);
        CHECK_FALSE(bindings[3].initiator_delete_term.valid());
        CHECK(bindings[2].node == bindings[3].node);
        CHECK(both_kinds.circuit.targets().size() == 4);
        CHECK(both_kinds.report.pc_targets == 4);
    }

    SECTION("no initiators at all") {
        auto config = probabilities;
        config.default_hull.max_initiators = 0;
        config.default_kinds = {pc::query_kind::monotone_hull};
        const auto artifact = require_lower(model, config);
        require_equivalent(model, artifact, {});
        CHECK(artifact.circuit.cardinality_low()[artifact.projection.target_bindings()[0].initiator_delete_term.raw()] == 1);
        CHECK(artifact.projection.target_bindings()[1].initiator_delete_term.valid());
    }

    SECTION("an overridden initiator is a constant and does not count") {
        auto config = probabilities;
        config.default_hull.max_initiators = 1;
        config.default_kinds = {pc::query_kind::monotone_hull};
        config.overrides = {{.name = "I2", .value = true}};
        const auto artifact = require_lower(model, config);
        require_equivalent(model, artifact, {{.element = i2, .value = true}});
        CHECK(events_of_span(artifact.projection.initiators(artifact.projection.target_bindings()[0])) == std::vector<pra::basic_event_id>{i1, i3});
    }
}

TEST_CASE("pra_to_pc records significant literals per target", "[compile][pra_to_pc][literals]") {
    pra::builder builder;
    std::vector<pra::basic_event_id> b;
    for (int index = 0; index <= 10; ++index) {
        b.push_back(builder.add_basic_event({.stable_name = "B" + std::to_string(index)}, pra::expression_spec::literal(0.5)));
    }
    const auto house = builder.add_house_event({.stable_name = "H"}, true);
    // B1: a marked literal. B2: also directly complemented, unmarked: a negation, never a literal.
    const auto direct = builder.add_gate({.stable_name = "DIRECT"}, pra::gate_kind::and_gate,
                                         {{.event = b[1], .complement = true, .literal = true}, {.event = b[0]}, {.event = b[2], .complement = true}});
    // B2: positive inside a complemented gate reference. B3: under not. B4: under nor. B5: under nand.
    const auto negated = builder.add_gate({.stable_name = "NEGATED"}, pra::gate_kind::or_gate, {{.event = b[2]}, {.event = b[0]}});
    const auto g_not = builder.add_gate({.stable_name = "NOT_B3"}, pra::gate_kind::not_gate, {{.event = b[3]}});
    const auto g_nor = builder.add_gate({.stable_name = "NOR_B4"}, pra::gate_kind::nor_gate, {{.event = b[4]}, {.event = b[0]}});
    const auto g_nand = builder.add_gate({.stable_name = "NAND_B5"}, pra::gate_kind::nand_gate, {{.event = b[5]}});
    // B6: a marked literal inside a gate that is itself complemented; still a literal.
    const auto inner =
        builder.add_gate({.stable_name = "INNER"}, pra::gate_kind::and_gate, {{.event = b[6], .complement = true, .literal = true}, {.event = b[0]}});
    // B7: a literal, overridden. B8: a literal only in OTHER's cone. B9 twice, both polarities.
    const auto overridden = builder.add_gate({.stable_name = "OVERRIDDEN"}, pra::gate_kind::or_gate, {{.event = b[7], .complement = true, .literal = true}});
    const auto both =
        builder.add_gate({.stable_name = "BOTH"}, pra::gate_kind::xor_gate, {{.event = b[9]}, {.event = b[9], .complement = true, .literal = true}});
    const auto top = builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate,
                                      {{.event = direct},
                                       {.event = negated, .complement = true},
                                       {.event = g_not},
                                       {.event = g_nor},
                                       {.event = g_nand},
                                       {.event = inner, .complement = true},
                                       {.event = overridden},
                                       {.event = both},
                                       {.event = house, .complement = true}});
    const auto other =
        builder.add_gate({.stable_name = "OTHER"}, pra::gate_kind::and_gate, {{.event = b[8], .complement = true, .literal = true}, {.event = direct}});
    (void)builder.add_target({.stable_name = "TOP"}, pra::domain_target_kind::gate, top);
    (void)builder.add_target({.stable_name = "OTHER"}, pra::domain_target_kind::gate, other);
    (void)builder.add_target({.stable_name = "B10"}, pra::domain_target_kind::basic_event, b[10]);
    const auto model = seal(std::move(builder));

    lowering::quantification_config config;
    config.overrides = {{.name = "B7", .value = false}};
    const auto artifact = require_lower(model, config);
    require_equivalent(model, artifact, {{.element = b[7], .value = false}});

    auto const &projection = artifact.projection;
    REQUIRE(projection.target_bindings().size() == 3);
    CHECK(events_of_span(projection.literal_events(projection.target_bindings()[0])) == std::vector<pra::basic_event_id>{b[1], b[6], b[9]});
    CHECK(events_of_span(projection.literal_events(projection.target_bindings()[1])) == std::vector<pra::basic_event_id>{b[1], b[8]});
    CHECK(projection.literal_events(projection.target_bindings()[2]).empty());
    CHECK(projection.literal_events().size() == 5);
}

TEST_CASE("pra_to_pc probability policy", "[compile][pra_to_pc][probability]") {
    pra::builder builder;
    const auto literal = builder.add_basic_event({.stable_name = "LIT"}, pra::expression_spec::literal(0.25));
    const auto parameter = builder.add_basic_event({.stable_name = "PAR"}, pra::expression_spec::parameter("lambda"));
    const auto unsupported = builder.add_basic_event({.stable_name = "UNS"}, pra::expression_spec::unsupported("io.ftp.missing-import", "no IMPORT record"));
    const auto rate = builder.add_basic_event({.stable_name = "IE_LOW"}, pra::expression_spec::literal(0.75, pra::quantity_kind::frequency));
    const auto high = builder.add_basic_event({.stable_name = "IE_HIGH"}, pra::expression_spec::literal(2.5, pra::quantity_kind::frequency));
    (void)builder.add_initiating_event({.stable_name = "IE_LOW"}, rate);
    (void)builder.add_initiating_event({.stable_name = "IE_HIGH"}, high);
    const auto good = builder.add_gate({.stable_name = "GOOD"}, pra::gate_kind::or_gate, {{.event = literal}, {.event = rate}});
    const auto bad = builder.add_gate({.stable_name = "BAD"}, pra::gate_kind::or_gate, {{.event = parameter}, {.event = unsupported}, {.event = high}});
    (void)builder.add_target({.stable_name = "GOOD"}, pra::domain_target_kind::gate, good);
    (void)builder.add_target({.stable_name = "BAD"}, pra::domain_target_kind::gate, bad, pra::target_role::watch);
    const auto model = seal(std::move(builder));

    // The initiating-event values are read as probabilities here (the lowering's default is frequencies).
    lowering::quantification_config probabilities;
    probabilities.initiators = lowering::initiator_values::probability;

    SECTION("events outside the cone are not checked; frequencies up to 1 warn") {
        auto lowered = lowering::lower(model, probabilities);
        REQUIRE(lowered);
        REQUIRE(codes(lowered.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.frequency-as-probability"});
        CHECK(lowered.diagnostics().items()[0].message.find("'IE_LOW' (initiating event)") != std::string::npos);
        auto const &artifact = lowered.value();
        const auto node = artifact.projection.node_of(rate).raw();
        CHECK(artifact.circuit.probability_sources()[artifact.circuit.probability_source_refs()[node].raw()].literal == 0.75);
        require_equivalent(model, artifact, {});
    }

    SECTION("every offending event in the cone is reported") {
        auto config = probabilities;
        config.watch_targets = {{.name = "BAD"}};
        auto lowered = lowering::lower(model, config);
        REQUIRE_FALSE(lowered);
        CHECK(codes(lowered.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.parameter-probability", "compile.pra_to_pc.unsupported-probability",
                                                                       "compile.pra_to_pc.frequency-as-probability",
                                                                       "compile.pra_to_pc.frequency-out-of-range"});
        CHECK(lowered.diagnostics().items()[1].message.find("io.ftp.missing-import: no IMPORT record") != std::string::npos);
    }
}

TEST_CASE("pra_to_pc target selection", "[compile][pra_to_pc][targets]") {
    pra::builder builder;
    const auto a = builder.add_basic_event({.stable_name = "A"}, pra::expression_spec::literal(0.5));
    const auto g = builder.add_gate({.stable_name = "G"}, pra::gate_kind::or_gate, {{.event = a}});
    const auto h = builder.add_gate({.stable_name = "H"}, pra::gate_kind::and_gate, {{.event = g}, {.event = a, .complement = true}});
    const auto p = builder.add_target({.stable_name = "P"}, pra::domain_target_kind::gate, g);
    const auto w = builder.add_target({.stable_name = "W"}, pra::domain_target_kind::gate, h, pra::target_role::watch);
    const auto model = seal(std::move(builder));

    SECTION("default: the primary targets with the default kinds, one binding per query") {
        lowering::quantification_config config;
        config.default_kinds = {pc::query_kind::monotone_hull, pc::query_kind::probability};
        const auto artifact = require_lower(model, config);
        auto const bindings = artifact.projection.target_bindings();
        REQUIRE(bindings.size() == 2);
        CHECK(bindings[0].domain_target == p);
        CHECK(bindings[0].kind == pc::query_kind::probability);
        CHECK(bindings[1].domain_target == p);
        CHECK(bindings[1].kind == pc::query_kind::monotone_hull);
        CHECK(bindings[1].hull == pra::hull_query{});
        CHECK(bindings[0].node == bindings[1].node);
    }

    SECTION("explicit targets, watch targets and element targets") {
        lowering::quantification_config config;
        config.targets = {{.name = "W", .kinds = {pc::query_kind::monotone_hull}}, {.name = "H", .element = pra::element_kind::gate}};
        config.watch_targets = {{.name = "P"}, {.name = "A", .element = pra::element_kind::basic_event}};
        const auto artifact = require_lower(model, config);
        auto const targets = artifact.projection.target_bindings();
        REQUIRE(targets.size() == 4);
        CHECK(targets[0].domain_target == w);
        CHECK(targets[0].role == pc::observable_role::watch);
        CHECK(targets[0].kind == pc::query_kind::monotone_hull);
        CHECK(targets[1].kind == pc::query_kind::probability);
        CHECK_FALSE(targets[1].domain_target.valid());
        CHECK(targets[1].subject == pra::domain_ref(h));
        CHECK(targets[1].role == pc::observable_role::primary);
        CHECK(targets[1].node == targets[0].node);
        CHECK(targets[2].domain_target == p);
        CHECK(targets[2].role == pc::observable_role::watch);
        CHECK(targets[3].subject == pra::domain_ref(a));
        CHECK(targets[3].role == pc::observable_role::watch);
        require_equivalent(model, artifact, {});
    }

    SECTION("errors") {
        lowering::quantification_config config;
        config.targets = {{.name = "NOPE"},
                          {.name = "P"},
                          {.name = "P"},
                          {.name = "A", .element = pra::element_kind::house_event},
                          {.name = "G", .element = pra::element_kind::basic_event},
                          {.name = "G", .element = pra::element_kind::gate, .kinds = {pc::query_kind::probability}, .hull = pra::hull_query{}}};
        auto lowered = lowering::lower(model, config);
        REQUIRE_FALSE(lowered);
        CHECK(codes(lowered.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.unknown-target", "compile.pra_to_pc.duplicate-target",
                                                                       "compile.pra_to_pc.invalid-target", "compile.pra_to_pc.unknown-target",
                                                                       "compile.pra_to_pc.hull-parameters-without-hull"});

        lowering::quantification_config none;
        none.default_kinds = {};
        auto empty = lowering::lower(model, none);
        REQUIRE_FALSE(empty);
        CHECK(codes(empty.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.no-query-kinds"});
    }

    SECTION("nothing to lower") {
        pra::builder empty;
        const auto event = empty.add_basic_event({.stable_name = "E"}, pra::expression_spec::literal(0.5));
        (void)empty.add_target({.stable_name = "E"}, pra::domain_target_kind::basic_event, event, pra::target_role::watch);
        auto lowered = lowering::lower(seal(std::move(empty)));
        REQUIRE_FALSE(lowered);
        CHECK(codes(lowered.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.no-targets"});
    }

    SECTION("handles") {
        auto from_handle = lowering::lower(model.to_handle());
        REQUIRE(from_handle);
        CHECK(pra::model_from(from_handle.value().pra_model) != nullptr);
        auto wrong = lowering::lower(from_handle.value().circuit.to_handle());
        REQUIRE_FALSE(wrong);
        CHECK(codes(wrong.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.unsupported-ir"});
        auto invalid = lowering::lower(pra::model{});
        REQUIRE_FALSE(invalid);
        CHECK(codes(invalid.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.invalid-model"});
    }
}
