#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include "compile/pra_to_pc/descriptor.h"
#include "ir/pra/builder.h"
#include "ir/pra/projection.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace pra_to_pc_test;

TEST_CASE("pra_to_pc projection resolves in both directions", "[compile][pra_to_pc][projection]") {
    pra::builder builder;
    const auto a = builder.add_basic_event({.stable_name = "A"}, pra::expression_spec::literal(0.1));
    const auto b = builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::literal(0.2));
    const auto i = builder.add_basic_event({.stable_name = "I"}, pra::expression_spec::literal(0.3));
    const auto j = builder.add_basic_event({.stable_name = "J"}, pra::expression_spec::literal(0.4));
    const auto unused = builder.add_basic_event({.stable_name = "UNUSED"}, pra::expression_spec::literal(0.5));
    (void)builder.add_initiating_event({.stable_name = "I"}, i);
    (void)builder.add_initiating_event({.stable_name = "J"}, j);
    const auto house = builder.add_house_event({.stable_name = "H"}, false);
    const auto tree = builder.add_fault_tree({.stable_name = "FT"});
    const auto left = builder.add_gate({.stable_name = "L"}, pra::gate_kind::and_gate, {{.event = a}, {.event = i}}, tree);
    const auto right =
        builder.add_gate({.stable_name = "R"}, pra::gate_kind::and_gate, {{.event = b, .complement = true, .literal = true}, {.event = j}}, tree);
    const auto flag = builder.add_gate({.stable_name = "FLAG"}, pra::gate_kind::or_gate, {{.event = house}, {.event = a}}, tree);
    const auto top = builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = left}, {.event = right}, {.event = flag}}, tree);
    builder.add_fault_tree_top(tree, top);
    const auto ft = builder.add_target({.stable_name = "FT"}, pra::domain_target_kind::fault_tree_top, tree);
    const auto watched = builder.add_target({.stable_name = "R"}, pra::domain_target_kind::gate, right, pra::target_role::watch);
    const auto model = seal(std::move(builder));

    lowering::quantification_config config;
    config.initiators = lowering::initiator_values::probability;
    config.watch_targets = {{.name = "R"}, {.name = "L", .element = pra::element_kind::gate}};
    config.overrides = {{.name = "FLAG", .value = false}};
    config.default_kinds = {pc::query_kind::monotone_hull};
    config.default_hull.literals = pra::hull_literals::plain;
    config.default_hull.max_initiators = 1;
    const auto artifact = require_lower(model, config);
    require_equivalent(model, artifact, {{.element = flag, .value = false}});
    auto const &projection = artifact.projection;

    // Every PC node is bound to an element or generated for a target.
    CHECK(projection.event_bindings().size() + projection.generated_bindings().size() == artifact.circuit.node_count());
    std::set<std::uint32_t> nodes;
    for (auto const &binding : projection.event_bindings()) {
        CHECK(nodes.insert(binding.node.raw()).second);
        REQUIRE(projection.find_event(binding.node) == &binding);
        REQUIRE(projection.find_event(binding.element) == &binding);
        CHECK(projection.node_of(binding.element) == binding.node);
        CHECK(projection.find_generated(binding.node) == nullptr);
    }
    for (auto const &generated : projection.generated_bindings()) {
        CHECK(nodes.insert(generated.node.raw()).second);
        REQUIRE(projection.find_generated(generated.node) == &generated);
        CHECK(projection.find_event(generated.node) == nullptr);
        CHECK(projection.find_target(generated.target) != nullptr);
    }
    for (const pra::domain_ref element : {pra::domain_ref(a), pra::domain_ref(b), pra::domain_ref(i), pra::domain_ref(j)}) {
        REQUIRE(projection.find_event(element) != nullptr);
        CHECK(projection.find_event(element)->kind == pra::event_binding_kind::basic_event);
    }
    CHECK(projection.find_event(pra::domain_ref(unused)) == nullptr);
    CHECK(projection.find_event(pra::domain_ref(house)) == nullptr);
    CHECK(projection.find_event(pra::domain_ref(flag))->kind == pra::event_binding_kind::constant_override);
    CHECK(projection.find_event(pra::domain_ref(top))->kind == pra::event_binding_kind::gate);

    auto const targets = projection.target_bindings();
    REQUIRE(targets.size() == 3);
    for (std::uint32_t index = 0; index < targets.size(); ++index) {
        auto const &binding = targets[index];
        CHECK(binding.pc_target == pc::target_id(index));
        REQUIRE(projection.find_target(binding.pc_target) == &binding);
        CHECK(artifact.circuit.targets()[index].node == binding.node);
        CHECK(projection.node_of(binding.subject) == binding.root);
        if (binding.domain_target.valid()) {
            CHECK(projection.find_target(binding.domain_target) == &binding);
        }
    }
    CHECK(targets[0].domain_target == ft);
    CHECK(targets[0].subject == pra::domain_ref(top));
    CHECK(targets[1].domain_target == watched);
    CHECK_FALSE(targets[2].domain_target.valid());
    CHECK(projection.find_target(pc::target_id(3)) == nullptr);
    CHECK(projection.find_target(pra::domain_target_id(7)) == nullptr);

    CHECK(targets[0].initiator_delete_term.valid());
    CHECK_FALSE(targets[1].initiator_delete_term.valid());
    CHECK(events_of_span(projection.initiators(targets[0])) == std::vector<pra::basic_event_id>{i, j});
    CHECK(events_of_span(projection.literal_events(targets[0])) == std::vector<pra::basic_event_id>{b});
    CHECK(events_of_span(projection.literal_events(targets[2])).empty());
}

TEST_CASE("pc_projection default is empty", "[compile][pra_to_pc][projection]") {
    const pra::pc_projection projection;
    CHECK(projection.target_bindings().empty());
    CHECK(projection.find_target(pc::target_id(0)) == nullptr);
    CHECK(projection.find_target(pra::domain_target_id(0)) == nullptr);
    CHECK(projection.find_event(pc::bool_node_id(0)) == nullptr);
    CHECK_FALSE(projection.node_of(pra::domain_ref(pra::gate_id(0))).valid());
    CHECK(projection.initiators(pra::target_binding{.initiators = {.offset = 3, .count = 1}}).empty());
}
