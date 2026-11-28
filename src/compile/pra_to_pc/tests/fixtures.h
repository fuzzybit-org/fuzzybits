#pragma once

#include "support.h"

#include "ir/pra/builder.h"
#include "ir/pra/id.h"
#include "ir/pra/model.h"

#include <span>
#include <utility>
#include <vector>

namespace pra_to_pc_test {

[[nodiscard]] inline auto ids(std::span<pra::basic_event_id const> events) -> std::vector<pra::basic_event_id> { return {events.begin(), events.end()}; }

// Every R1 connective, complements on events and gates, house events, a
// non-monotone cardinality range, a complemented single-input gate and
// subgraphs shared between targets.
struct connective_model final {
    pra::model model;
    pra::basic_event_id a, b, c, d, e, f;
    pra::house_event_id house_true, house_false;
    pra::gate_id both, either, vote, range, negation, parity, nand, nor, single, shared, with_house, top;
    pra::fault_tree_id tree;
    pra::domain_target_id tree_target, vote_target, single_target, event_target;
};

[[nodiscard]] inline auto make_connective_model() -> connective_model {
    connective_model out;
    pra::builder builder;
    builder.set_project("connectives");
    out.a = builder.add_basic_event({.stable_name = "A"}, pra::expression_spec::literal(0.1));
    out.b = builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::literal(0.2));
    out.c = builder.add_basic_event({.stable_name = "C"}, pra::expression_spec::literal(0.3));
    out.d = builder.add_basic_event({.stable_name = "D"}, pra::expression_spec::literal(0.4));
    const auto half = builder.add_expression(pra::expression_spec::literal(0.5));
    out.e = builder.add_basic_event({.stable_name = "E"}, half);
    out.f = builder.add_basic_event({.stable_name = "F"}, half);
    (void)builder.add_basic_event({.stable_name = "UNUSED"}, pra::expression_spec::parameter("lambda"));
    out.house_true = builder.add_house_event({.stable_name = "H_TRUE"}, true);
    out.house_false = builder.add_house_event({.stable_name = "H_FALSE"}, false);
    out.tree = builder.add_fault_tree({.stable_name = "FT"});
    out.both = builder.add_gate({.stable_name = "G_AND"}, pra::gate_kind::and_gate, {{.event = out.a}, {.event = out.b, .complement = true}}, out.tree);
    out.either =
        builder.add_gate({.stable_name = "G_OR"}, pra::gate_kind::or_gate, {{.event = out.both}, {.event = out.house_false}, {.event = out.c}}, out.tree);
    out.vote = builder.add_cardinality({.stable_name = "G_2OF3"}, {.low = 2, .high = 3},
                                       {{.event = out.a}, {.event = out.c, .complement = true}, {.event = out.d}}, out.tree);
    out.range = builder.add_cardinality({.stable_name = "G_RANGE"}, {.low = 1, .high = 2},
                                        {{.event = out.b}, {.event = out.d}, {.event = out.e}, {.event = out.f}}, out.tree);
    out.negation = builder.add_gate({.stable_name = "G_NOT"}, pra::gate_kind::not_gate, {{.event = out.vote}}, out.tree);
    out.parity = builder.add_gate({.stable_name = "G_XOR"}, pra::gate_kind::xor_gate,
                                  {{.event = out.a}, {.event = out.d}, {.event = out.e, .complement = true}}, out.tree);
    out.nand = builder.add_gate({.stable_name = "G_NAND"}, pra::gate_kind::nand_gate, {{.event = out.both}, {.event = out.f, .complement = true}}, out.tree);
    out.nor = builder.add_gate({.stable_name = "G_NOR"}, pra::gate_kind::nor_gate, {{.event = out.c}, {.event = out.d, .complement = true}}, out.tree);
    out.single = builder.add_gate({.stable_name = "G_SINGLE"}, pra::gate_kind::or_gate, {{.event = out.either, .complement = true}}, out.tree);
    out.shared = builder.add_gate({.stable_name = "G_SHARED"}, pra::gate_kind::and_gate, {{.event = out.either}, {.event = out.vote}}, out.tree);
    out.with_house = builder.add_gate({.stable_name = "G_HOUSE"}, pra::gate_kind::and_gate, {{.event = out.house_true}, {.event = out.e}}, out.tree);
    out.top = builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate,
                               {{.event = out.shared},
                                {.event = out.negation},
                                {.event = out.parity},
                                {.event = out.nand},
                                {.event = out.nor},
                                {.event = out.range},
                                {.event = out.single, .complement = true},
                                {.event = out.with_house}},
                               out.tree);
    builder.add_fault_tree_top(out.tree, out.top);
    out.tree_target = builder.add_target({.stable_name = "FT"}, pra::domain_target_kind::fault_tree_top, out.tree);
    out.vote_target = builder.add_target({.stable_name = "VOTE"}, pra::domain_target_kind::gate, out.vote);
    out.single_target = builder.add_target({.stable_name = "SINGLE"}, pra::domain_target_kind::gate, out.single);
    out.event_target = builder.add_target({.stable_name = "D_WATCH"}, pra::domain_target_kind::basic_event, out.d, pra::target_role::watch);
    out.model = seal(std::move(builder));
    return out;
}

} // namespace pra_to_pc_test
