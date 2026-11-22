#include <catch2/catch_test_macros.hpp>

#include "compile/pra_to_pc/compose.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/op_code.h"
#include "ir/pc/passes/passes.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

    namespace pc = fuzzybits::ir::pc;
    namespace passes = fuzzybits::ir::pc::passes;
    namespace lowering = fuzzybits::compile::backend::fuzzybits::pra_to_pc;
    namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;

    using edge = pc::edge_ref;

    [[nodiscard]] auto name_of(writer::write_options const &options, const pc::bool_node_id node) -> std::string {
        const auto found = options.node_names.find(node);
        return found == options.node_names.end() ? std::string{} : found->second;
    }

    [[nodiscard]] auto node_named(writer::write_options const &options, std::string const &name) -> pc::bool_node_id {
        for (auto const &[node, text] : options.node_names) {
            if (text == name) {
                return node;
            }
        }
        return pc::bool_node_id::invalid();
    }

} // namespace

TEST_CASE("names move to the nodes that compute them through C1", "[compile][pra_to_pc][compose]") {
    pc::builder builder;
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
    const auto b = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto zero = builder.add_basic_event(builder.add_literal_probability(0.0));
    const auto wrapper = builder.add_and({edge{.source = a}});                     // eliminated onto a
    const auto inverted = builder.add_not(edge{.source = b});                       // a complemented image of b
    const auto dead = builder.add_and({edge{.source = zero}, edge{.source = b}});   // folds to FALSE
    const auto top = builder.add_or({edge{.source = wrapper}, edge{.source = b}});
    const auto other = builder.add_or({edge{.source = a}, edge{.source = inverted}, edge{.source = dead}});
    const auto first_target = builder.add_target(top);
    const auto second_target = builder.add_target(other, pc::observable_role::watch);
    auto sealed = std::move(builder).seal();
    REQUIRE(sealed.ok());
    auto const &input = sealed.output;

    writer::write_options names;
    names.target_names = {{first_target, "TOP"}, {second_target, "OTHER"}};
    names.target_kinds = {{first_target, {writer::query_kind::probability, writer::query_kind::monotone_hull_probability}},
                          {second_target, {writer::query_kind::monotone_hull_probability}}};
    names.quantification = {{first_target, {.literal_mode = "atom_split", .max_initiators_per_cutset = 1}},
                            {second_target, {.literal_mode = "negation", .max_initiators_per_cutset = std::nullopt}}};
    names.node_names = {{a, "A"}, {b, "B"}, {zero, "ZERO"}, {wrapper, "WRAP"}, {inverted, "NOT-B"}, {dead, "DEAD"}, {top, "TOP-GATE"}, {other, "OTHER-GATE"}};

    const auto compiled = passes::compile(input, passes::compile_level::c1);
    REQUIRE(compiled.ok());
    const auto composed = lowering::compose_write_options(names, input, compiled);

    REQUIRE(composed.target_names.size() == 2);
    REQUIRE(composed.target_names.at(compiled.target_map[first_target.raw()]) == "TOP");
    REQUIRE(composed.target_names.at(compiled.target_map[second_target.raw()]) == "OTHER");

    // Query kinds and the quantification record follow their target.
    REQUIRE(composed.target_kinds.size() == 2);
    REQUIRE(composed.target_kinds.at(compiled.target_map[first_target.raw()]) ==
            std::vector<writer::query_kind>{writer::query_kind::probability, writer::query_kind::monotone_hull_probability});
    REQUIRE(composed.target_kinds.at(compiled.target_map[second_target.raw()]) ==
            std::vector<writer::query_kind>{writer::query_kind::monotone_hull_probability});
    REQUIRE(composed.quantification.size() == 2);
    REQUIRE(composed.quantification.at(compiled.target_map[first_target.raw()]).literal_mode == "atom_split");
    REQUIRE(composed.quantification.at(compiled.target_map[first_target.raw()]).max_initiators_per_cutset == 1U);
    REQUIRE(composed.quantification.at(compiled.target_map[second_target.raw()]).literal_mode == "negation");
    REQUIRE_FALSE(composed.quantification.at(compiled.target_map[second_target.raw()]).max_initiators_per_cutset);

    // The events keep their names on event nodes, and the gate eliminated onto `a` does not take `a`'s node.
    const auto out_a = node_named(composed, "A");
    const auto out_b = node_named(composed, "B");
    REQUIRE(out_a.valid());
    REQUIRE(out_b.valid());
    REQUIRE(compiled.output.op_codes()[out_a.raw()] == pc::op_code::basic_event);
    REQUIRE(compiled.output.op_codes()[out_b.raw()] == pc::op_code::basic_event);
    REQUIRE(name_of(composed, out_a) == "A");
    REQUIRE_FALSE(node_named(composed, "WRAP").valid());
    // Folded, complemented and removed nodes are unnamed.
    REQUIRE_FALSE(node_named(composed, "ZERO").valid());
    REQUIRE_FALSE(node_named(composed, "DEAD").valid());
    REQUIRE_FALSE(node_named(composed, "NOT-B").valid());
    // Every name belongs to a node of the output circuit.
    for (auto const &[node, text] : composed.node_names) {
        INFO(text);
        REQUIRE(node.raw() < compiled.output.node_count());
    }
}

TEST_CASE("C0 keeps every name", "[compile][pra_to_pc][compose]") {
    pc::builder builder;
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
    const auto b = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto top = builder.add_and({edge{.source = a}, edge{.source = b, .complement = true}});
    const auto target = builder.add_target(top);
    auto sealed = std::move(builder).seal();
    REQUIRE(sealed.ok());

    writer::write_options names;
    names.target_names = {{target, "TOP"}};
    names.node_names = {{a, "A"}, {b, "B"}, {top, "G"}};
    const auto compiled = passes::compile(sealed.output, passes::compile_level::c0);
    REQUIRE(compiled.ok());
    const auto composed = lowering::compose_write_options(names, sealed.output, compiled);
    REQUIRE(composed.target_names == names.target_names);
    REQUIRE(composed.node_names == names.node_names);
    REQUIRE(composed.target_kinds == names.target_kinds);
    REQUIRE(composed.quantification.size() == names.quantification.size());
}
