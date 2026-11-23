#include <catch2/catch_test_macros.hpp>

#include "pra_test_support.h"

#include "ir/forms.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pra/builder.h"
#include "ir/pra/descriptor.h"
#include "ir/pra/model.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace pra = fuzzybits::ir::pra;
using pra::test::named;
using pra::test::render;
using pra::test::topologically_ordered;

[[nodiscard]] auto contains(std::span<pra::gate_kind const> kinds, const pra::gate_kind kind) -> bool {
    return std::ranges::find(kinds, kind) != kinds.end();
}

} // namespace

TEST_CASE("PRA builder seals every R1 entity kind", "[ir][pra][builder]") {
    pra::builder builder;
    builder.set_project("demo", "Demonstration project");
    builder.add_project_attribute("ftp.limit", "1e-12");
    const auto file = builder.add_source_file({.path = "model/a.ftp", .sha256 = "abc123", .format = "ftp"});
    const pra::source_location line_3{.file = file, .line = 3};
    const pra::source_location line_9{.file = file, .line = 9};
    const pra::attribute_spec flags[] = {{.name = "ftp.flag", .value = "X"}};

    const auto pump =
        builder.add_basic_event({.stable_name = "PUMP", .label = "Pump fails to start", .sources = {&line_3, 1}},
                                pra::expression_spec::literal(1e-3));
    const auto valve =
        builder.add_basic_event({.stable_name = "VALVE"}, pra::expression_spec::parameter("lambda_valve"));
    const auto crew =
        builder.add_basic_event({.stable_name = "CREW", .attributes = flags},
                                pra::expression_spec::unsupported("io.ftp.missing-import", "no IMPORT record"));
    const auto loss = builder.add_basic_event({.stable_name = "LOSP"},
                                              pra::expression_spec::literal(2.5, pra::quantity_kind::frequency));
    const auto shared = builder.add_expression(pra::expression_spec::literal(0.5));
    const auto left = builder.add_basic_event({.stable_name = "L"}, shared);
    const auto right = builder.add_basic_event({.stable_name = "R"}, shared);
    const auto maintenance = builder.add_house_event({.stable_name = "MAINT"}, true);
    const auto initiator = builder.add_initiating_event({.stable_name = "LOSP", .qualified_name = "IE.LOSP"}, loss);

    const auto tree = builder.add_fault_tree({.stable_name = "FT", .label = "Loss of cooling"});
    const auto both = builder.add_gate({.stable_name = "G_AND"}, pra::gate_kind::and_gate,
                                       {{.event = pump}, {.event = valve, .complement = true}}, tree);
    const auto either = builder.add_gate({.stable_name = "G_OR"}, pra::gate_kind::or_gate,
                                         {{.event = both}, {.event = maintenance}}, tree);
    const auto vote = builder.add_cardinality({.stable_name = "G_2OF3"}, {.low = 2, .high = 3},
                                              {{.event = pump}, {.event = valve}, {.event = crew}}, tree);
    const auto negation = builder.add_gate({.stable_name = "G_NOT"}, pra::gate_kind::not_gate, {{.event = vote}}, tree);
    const auto parity =
        builder.add_gate({.stable_name = "G_XOR"}, pra::gate_kind::xor_gate, {{.event = left}, {.event = right}}, tree);
    const auto nand = builder.add_gate({.stable_name = "G_NAND"}, pra::gate_kind::nand_gate,
                                       {{.event = left}, {.event = right, .complement = true}}, tree);
    const auto nor =
        builder.add_gate({.stable_name = "G_NOR"}, pra::gate_kind::nor_gate, {{.event = left}, {.event = right}}, tree);
    const auto xnor = builder.add_gate({.stable_name = "G_XNOR"}, pra::gate_kind::xnor_gate,
                                       {{.event = left}, {.event = right, .complement = true}}, tree);
    const auto top = builder.add_gate(
        {.stable_name = "TOP"}, pra::gate_kind::or_gate,
        {{.event = either}, {.event = negation}, {.event = parity}, {.event = nand}, {.event = nor}, {.event = xnor}},
        tree);
    builder.add_fault_tree_top(tree, top);

    const auto top_target = builder.add_target({.stable_name = "FT"}, pra::domain_target_kind::fault_tree_top, tree);
    const auto gate_target =
        builder.add_target({.stable_name = "G_OR"}, pra::domain_target_kind::gate, either, pra::target_role::watch);
    const auto event_target = builder.add_target({.stable_name = "PUMP"}, pra::domain_target_kind::basic_event, pump,
                                                 pra::target_role::watch);
    builder.add_attribute(top, "note", "added after the gate");
    builder.add_source(top, line_9);

    auto result = std::move(builder).seal();
    INFO(render(result.diagnostics));
    REQUIRE(result.ok());
    REQUIRE(result.diagnostics.empty());
    auto const &model = result.output;

    SECTION("tables have one row per entity, in insertion order") {
        CHECK(model.basic_events().size() == 6);
        CHECK(model.house_events().size() == 1);
        CHECK(model.initiating_events().size() == 1);
        CHECK(model.gates().size() == 9);
        CHECK(model.fault_trees().size() == 1);
        CHECK(model.targets().size() == 3);
        CHECK(model.expressions().size() == 5);
        CHECK(model.elements().size() == 21);
        CHECK(pump == pra::basic_event_id(0));
        CHECK(top == pra::gate_id(8));
        for (std::uint32_t index = 0; index < model.elements().size(); ++index) {
            CHECK(model.element_of(model.elements()[index].owner) == pra::element_id(index));
        }
    }

    SECTION("names are unique per kind and found by kind") {
        CHECK(model.find_basic_event("PUMP") == pump);
        CHECK(model.find_basic_event("LOSP") == loss);
        CHECK(model.find_initiating_event("LOSP") == initiator);
        CHECK(model.find_house_event("MAINT") == maintenance);
        CHECK(model.find_gate("TOP") == top);
        CHECK(model.find_fault_tree("FT") == tree);
        CHECK(model.find_target("FT") == top_target);
        CHECK(model.find_target("G_OR") == gate_target);
        CHECK(model.find(pra::element_kind::domain_target, "PUMP") == pra::domain_ref(event_target));
        CHECK_FALSE(model.find_gate("PUMP").valid());
        CHECK_FALSE(model.find_gate("top").valid());
        CHECK_FALSE(model.find_basic_event("").valid());
        CHECK(model.stable_name(pra::domain_ref(vote)) == "G_2OF3");
        CHECK(model.text(model.elements()[model.element_of(initiator).raw()].qualified_name) == "IE.LOSP");
    }

    SECTION("gates keep their connective, operands, complements and bounds") {
        auto const and_operands = model.operands(both);
        REQUIRE(and_operands.size() == 2);
        CHECK(and_operands[0].event == pra::domain_ref(pump));
        CHECK_FALSE(and_operands[0].complement);
        CHECK(and_operands[1].event == pra::domain_ref(valve));
        CHECK(and_operands[1].complement);
        CHECK(model.operands(either)[1].event == pra::domain_ref(maintenance));

        auto const &cardinality = model.gates()[vote.raw()];
        CHECK(cardinality.kind == pra::gate_kind::cardinality);
        CHECK(cardinality.bounds.low == 2);
        CHECK(cardinality.bounds.high == 3);
        CHECK(model.gates()[both.raw()].bounds.high == 0);
        CHECK(model.gates()[negation.raw()].kind == pra::gate_kind::not_gate);
        CHECK(model.gates()[parity.raw()].kind == pra::gate_kind::xor_gate);
        CHECK(model.gates()[nand.raw()].kind == pra::gate_kind::nand_gate);
        CHECK(model.gates()[nor.raw()].kind == pra::gate_kind::nor_gate);
        CHECK(model.gates()[xnor.raw()].kind == pra::gate_kind::xnor_gate);
        CHECK(model.operands(xnor)[1].complement);
        CHECK(model.operands(top).size() == 6);
        CHECK(topologically_ordered(model));
    }

    SECTION("events keep their values") {
        auto const expressions = model.expressions();
        auto const &literal = expressions[model.basic_events()[pump.raw()].expression.raw()];
        CHECK(literal.kind == pra::expression_kind::literal);
        CHECK(literal.quantity == pra::quantity_kind::probability);
        CHECK(literal.value == 1e-3);
        auto const &parameter = expressions[model.basic_events()[valve.raw()].expression.raw()];
        CHECK(parameter.kind == pra::expression_kind::parameter);
        CHECK(model.text(parameter.parameter_name) == "lambda_valve");
        auto const &unsupported = expressions[model.basic_events()[crew.raw()].expression.raw()];
        CHECK(unsupported.kind == pra::expression_kind::unsupported);
        CHECK(model.text(unsupported.diagnostic) == "io.ftp.missing-import");
        CHECK(model.text(unsupported.detail) == "no IMPORT record");
        auto const &frequency = expressions[model.basic_events()[loss.raw()].expression.raw()];
        CHECK(frequency.quantity == pra::quantity_kind::frequency);
        CHECK(frequency.value == 2.5);
        CHECK(model.basic_events()[left.raw()].expression == shared);
        CHECK(model.basic_events()[right.raw()].expression == shared);
        CHECK(model.house_events()[maintenance.raw()].value);
        CHECK(model.initiating_events()[initiator.raw()].event == loss);
    }

    SECTION("fault trees group gates and name their top gate") {
        CHECK(model.gates(tree).size() == 9);
        CHECK(std::ranges::is_sorted(model.gates(tree)));
        REQUIRE(model.top_gates(tree).size() == 1);
        CHECK(model.top_gates(tree)[0] == top);
        CHECK(model.gates()[top.raw()].fault_tree == tree);
    }

    SECTION("targets resolve to the observed event") {
        auto const targets = model.targets();
        CHECK(targets[top_target.raw()].kind == pra::domain_target_kind::fault_tree_top);
        CHECK(targets[top_target.raw()].role == pra::target_role::primary);
        CHECK(targets[top_target.raw()].subject == pra::domain_ref(tree));
        CHECK(model.target_event(top_target) == pra::domain_ref(top));
        CHECK(targets[gate_target.raw()].role == pra::target_role::watch);
        CHECK(model.target_event(gate_target) == pra::domain_ref(either));
        CHECK(model.target_event(event_target) == pra::domain_ref(pump));
    }

    SECTION("metadata keeps labels, attributes and source locations") {
        const auto pump_element = model.element_of(pump);
        CHECK(model.text(model.elements()[pump_element.raw()].label) == "Pump fails to start");
        REQUIRE(model.sources(pump_element).size() == 1);
        CHECK(model.sources(pump_element)[0].file == file);
        CHECK(model.sources(pump_element)[0].line == 3);

        const auto crew_attributes = model.attributes(model.element_of(crew));
        REQUIRE(crew_attributes.size() == 1);
        CHECK(model.text(crew_attributes[0].name) == "ftp.flag");
        CHECK(model.text(crew_attributes[0].value) == "X");

        const auto top_element = model.element_of(top);
        REQUIRE(model.attributes(top_element).size() == 1);
        CHECK(model.text(model.attributes(top_element)[0].value) == "added after the gate");
        REQUIRE(model.sources(top_element).size() == 1);
        CHECK(model.sources(top_element)[0].line == 9);
        CHECK(model.attributes(model.element_of(pump)).empty());

        REQUIRE(model.source_files().size() == 1);
        CHECK(model.text(model.source_files()[0].path) == "model/a.ftp");
        CHECK(model.text(model.source_files()[0].sha256) == "abc123");
        CHECK(model.text(model.source_files()[0].format) == "ftp");

        CHECK(model.text(model.project().name) == "demo");
        CHECK(model.text(model.project().label) == "Demonstration project");
        REQUIRE(model.project_attributes().size() == 1);
        CHECK(model.text(model.project_attributes()[0].name) == "ftp.limit");
        CHECK(model.text(model.project_attributes()[0].value) == "1e-12");
    }

    SECTION("summary describes the shape") {
        auto const &summary = model.summary();
        CHECK(summary.gate_count == 9);
        CHECK(summary.operand_count == 22);
        CHECK(summary.complemented_operand_count == 3);
        CHECK(summary.literal_operand_count == 0);
        CHECK(summary.gate_depth == 3);
        CHECK(summary.gate_kinds_present.size() == pra::gate_kind_count);
        CHECK(contains(summary.gate_kinds_present, pra::gate_kind::cardinality));
        CHECK(summary.expression_kinds_present.size() == 3);
        CHECK(summary.has_frequency_expressions);
    }
}

TEST_CASE("PRA builder resolves forward references across source files", "[ir][pra][builder]") {
    pra::builder builder;
    const auto first_file = builder.add_source_file({.path = "a.ftp", .format = "ftp"});
    const auto second_file = builder.add_source_file({.path = "b.ftp", .format = "ftp"});
    const pra::source_location top_line{.file = first_file, .line = 1};
    const pra::source_location sub_line{.file = second_file, .line = 4};

    // Everything below refers to names that are only defined later.
    const auto top_target =
        builder.add_target({.stable_name = "TOP"}, pra::domain_target_kind::gate, pra::reference::named("TOP"));
    const auto tree_target = builder.add_target({.stable_name = "TREE"}, pra::domain_target_kind::fault_tree_top,
                                                pra::reference::named("TOP"));
    const auto initiator = builder.add_initiating_event({.stable_name = "X"}, pra::reference::named("X"));
    // The fault tree shares the name TOP with a gate; references resolve by the kinds each use site admits.
    const auto tree = builder.add_fault_tree({.stable_name = "TOP"});
    builder.add_fault_tree_top(tree, pra::reference::named(pra::element_kind::gate, "TOP"));
    const auto top = builder.add_gate({.stable_name = "TOP", .sources = {&top_line, 1}}, pra::gate_kind::or_gate,
                                      {named("SUB"), named("X", true)}, tree);
    const auto sub =
        builder.add_gate({.stable_name = "SUB", .sources = {&sub_line, 1}}, pra::gate_kind::and_gate,
                         {named("Y"), {.event = pra::reference::named(pra::element_kind::house_event, "H")}}, tree);
    const auto house = builder.add_house_event({.stable_name = "H"}, false);
    const auto x = builder.add_basic_event({.stable_name = "X"}, pra::expression_spec::literal(0.1));
    const auto y = builder.add_basic_event({.stable_name = "Y"}, pra::expression_spec::literal(0.2));

    auto result = std::move(builder).seal();
    INFO(render(result.diagnostics));
    REQUIRE(result.ok());
    auto const &model = result.output;

    REQUIRE(model.operands(top).size() == 2);
    CHECK(model.operands(top)[0].event == pra::domain_ref(sub));
    CHECK(model.operands(top)[1].event == pra::domain_ref(x));
    CHECK(model.operands(top)[1].complement);
    CHECK(model.operands(sub)[0].event == pra::domain_ref(y));
    CHECK(model.operands(sub)[1].event == pra::domain_ref(house));
    CHECK(model.initiating_events()[initiator.raw()].event == x);
    CHECK(model.targets()[top_target.raw()].subject == pra::domain_ref(top));
    CHECK(model.targets()[tree_target.raw()].subject == pra::domain_ref(tree));
    CHECK(model.target_event(tree_target) == pra::domain_ref(top));
    CHECK(model.sources(model.element_of(top))[0].file == first_file);
    CHECK(model.sources(model.element_of(sub))[0].file == second_file);
    REQUIRE(model.topological_gates().size() == 2);
    CHECK(model.topological_gates()[0] == sub);
    CHECK(model.topological_gates()[1] == top);
}

TEST_CASE("PRA stable names are arbitrary tokens", "[ir][pra][builder]") {
    const std::vector<std::string_view> names{"1A", "X/Y", "A--B", "0.5", "G.1", "a b", "\xC3\x84"};
    pra::builder builder;
    std::vector<pra::operand_spec> operands;
    for (const auto name : names) {
        operands.push_back(named(name));
    }
    const auto top = builder.add_gate({.stable_name = "-TOP"}, pra::gate_kind::or_gate, operands);
    for (const auto name : names) {
        (void)builder.add_basic_event({.stable_name = name}, pra::expression_spec::literal(0.5));
    }

    auto result = std::move(builder).seal();
    INFO(render(result.diagnostics));
    REQUIRE(result.ok());
    auto const &model = result.output;
    CHECK(model.find_gate("-TOP") == top);
    for (std::uint32_t index = 0; index < names.size(); ++index) {
        CHECK(model.find_basic_event(names[index]) == pra::basic_event_id(index));
        CHECK(model.operands(top)[index].event == pra::domain_ref(pra::basic_event_id(index)));
    }
}

TEST_CASE("PRA builder seals an empty model", "[ir][pra][builder]") {
    auto result = pra::builder{}.seal();
    REQUIRE(result.ok());
    CHECK(result.output.gates().empty());
    CHECK(result.output.topological_gates().empty());
    CHECK(result.output.summary().gate_depth == 0);
    CHECK_FALSE(result.output.find_gate("G").valid());
}

TEST_CASE("PRA default model is invalid and empty", "[ir][pra][model]") {
    const pra::model model;
    CHECK_FALSE(model.valid());
    CHECK(model.elements().empty());
    CHECK(model.operands(pra::gate_id(0)).empty());
    CHECK(model.text(pra::text_ref{.offset = 0, .size = 4}).empty());
    CHECK_FALSE(model.find_gate("G").valid());
    CHECK_FALSE(model.target_event(pra::domain_target_id(0)).valid());
    CHECK_FALSE(model.to_handle().valid());
}

TEST_CASE("PRA model round-trips through ir::handle", "[ir][pra][handle]") {
    pra::builder builder;
    const auto event = builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::literal(0.1));
    const auto gate = builder.add_gate({.stable_name = "G"}, pra::gate_kind::or_gate, {{.event = event}});
    (void)builder.add_target({.stable_name = "G"}, pra::domain_target_kind::gate, gate);
    auto result = std::move(builder).seal();
    REQUIRE(result.ok());

    auto const &descriptor = result.output.descriptor();
    CHECK(descriptor.form == pra::model_form());
    CHECK(descriptor.form.str() == "ir.pra.model");
    CHECK(descriptor.storage_type == pra::storage_type_id());
    CHECK(descriptor.endpoint == pra::endpoint_id());
    CHECK(descriptor.capabilities.contains(fuzzybits::api::capability_id("ir.pra.model")));
    CHECK(fuzzybits::ir::descriptor_satisfies_form(descriptor, fuzzybits::ir::forms::probabilistic_risk_model()));
    CHECK_FALSE(fuzzybits::ir::descriptor_satisfies_form(descriptor, fuzzybits::ir::forms::bool_dag()));

    const auto handle = result.output.to_handle();
    REQUIRE(handle.valid());
    CHECK(pra::is_pra(handle));
    CHECK(fuzzybits::ir::pc::circuit_from(handle) == nullptr);
    auto const *model = pra::model_from(handle);
    REQUIRE(model != nullptr);
    CHECK(model->find_gate("G") == gate);
    CHECK(model->gates().data() == result.output.gates().data());

    fuzzybits::ir::pc::builder circuit_builder;
    const auto probability = circuit_builder.add_literal_probability(0.5);
    const auto node = circuit_builder.add_basic_event(probability);
    (void)circuit_builder.add_target(node);
    auto circuit = std::move(circuit_builder).seal();
    REQUIRE(circuit.ok());
    CHECK(pra::model_from(circuit.output.to_handle()) == nullptr);
}

TEST_CASE("PRA literal markers round-trip through the model and its summary", "[ir][pra][builder]") {
    pra::builder builder;
    const auto a = builder.add_basic_event({.stable_name = "A"}, pra::expression_spec::literal(0.1));
    const auto b = builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::literal(0.2));
    const auto inner =
        builder.add_gate({.stable_name = "INNER"}, pra::gate_kind::or_gate, {{.event = a}, {.event = b}});
    const auto top = builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::and_gate,
                                      {{.event = a},
                                       {.event = b, .complement = true, .literal = true},
                                       {.event = inner, .complement = true},
                                       named("B", true),
                                       {.event = pra::reference::named("A"), .complement = true, .literal = true}});
    auto result = std::move(builder).seal();
    INFO(render(result.diagnostics));
    REQUIRE(result.ok());

    const auto handle = result.output.to_handle();
    auto const *model = pra::model_from(handle);
    REQUIRE(model != nullptr);
    const auto operands = model->operands(top);
    REQUIRE(operands.size() == 5);
    CHECK_FALSE(operands[0].literal);
    CHECK(operands[1].event == pra::domain_ref(b));
    CHECK(operands[1].complement);
    CHECK(operands[1].literal);
    CHECK(operands[2].complement);
    CHECK_FALSE(operands[2].literal);
    CHECK(operands[3].complement);
    CHECK_FALSE(operands[3].literal);
    CHECK(operands[4].event == pra::domain_ref(a));
    CHECK(operands[4].literal);
    CHECK(model->gate_operands()[model->gates()[top.raw()].operands.offset + 1].literal);

    CHECK(model->summary().complemented_operand_count == 4);
    CHECK(model->summary().literal_operand_count == 2);
}
