// Known bugs of ir::pc sealing and the ir::pc pass pipeline. Each case states the contract it checks
// (docs/IR-PC-REQUIREMENTS.md, docs/ir-pc-passes.md) and fails on develop until its issue is fixed. Cases
// tagged [crash] read out of bounds today and run as their own ctest entry. Labelled `bug`.

#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/op_code.h"
#include "ir/pc/passes/passes.h"
#include "ir/pc/query.h"

#include <algorithm>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace pc = fuzzybits::ir::pc;
namespace passes = fuzzybits::ir::pc::passes;
using edge = pc::edge_ref;

[[nodiscard]] auto has_code(fuzzybits::api::diagnostic_list const &diagnostics, std::string_view code) -> bool {
    return std::ranges::any_of(diagnostics.items(), [code](auto const &item) { return item.code == code; });
}

// TOP = AND(e0, NOT-free gate over e1): two basic events, one gate, one target.
struct small_circuit final {
    pc::circuit circuit;
    pc::bool_node_id event;
    pc::bool_node_id gate;
};

[[nodiscard]] auto make_small() -> small_circuit {
    pc::builder builder;
    const auto e0 = builder.add_basic_event(builder.add_literal_probability(0.5));
    const auto e1 = builder.add_basic_event(builder.add_literal_probability(0.25));
    const auto inner = builder.add_or({edge{.source = e1}});
    const auto top = builder.add_and({edge{.source = e0}, edge{.source = inner}});
    (void)builder.add_target(top);
    auto sealed = std::move(builder).seal();
    REQUIRE(sealed.ok());
    return {std::move(sealed.output), e0, top};
}

} // namespace

// docs/ir-pc-passes.md (Diagnostics): pc.pass.invalid-atom-split is an error when "an atom-split target does
// not exist or an atom event is not a basic event". run_pipeline() copies pipeline_options::atom_events into
// the pass options unchecked, and after each pass carries them through the pass's node map by indexing it
// with the event id (pipeline.cpp, apply). coalesce never validates its options, so a schedule that starts
// with coalesce indexes node_map with an invalid id.
TEST_CASE("run_pipeline rejects an atom event that is not a node, whatever the schedule",
          "[bug][passes][crash][issue-292]") {
    const auto input = make_small();
    const passes::pass_id schedule[] = {passes::pass_id::coalesce};
    const auto result = passes::run_pipeline(input.circuit, schedule,
                                             passes::pipeline_options{.kinds = {pc::query_kind::monotone_hull},
                                                                      .atom_events = {{pc::bool_node_id(1'000'000)}}});
    CHECK_FALSE(result.ok());
    CHECK(has_code(result.diagnostics, "pc.pass.invalid-atom-split"));
}

TEST_CASE("run_pipeline rejects an atom event that is a gate, whatever the schedule", "[bug][passes][issue-297]") {
    const auto input = make_small();
    const passes::pass_id schedule[] = {passes::pass_id::coalesce};
    const auto result = passes::run_pipeline(
        input.circuit, schedule,
        passes::pipeline_options{.kinds = {pc::query_kind::monotone_hull}, .atom_events = {{input.gate}}});
    CHECK_FALSE(result.ok());
    CHECK(has_code(result.diagnostics, "pc.pass.invalid-atom-split"));
}

// The same rule for single passes: propagate_constants with a target that does not exist and no events, and
// coalesce with a gate as atom event, both return ok() without pc.pass.invalid-atom-split
// (rewrite.cpp split_context::make returns an inactive context when either list is empty; coalesce never
// builds one).
TEST_CASE("every pass validates its atom-split options", "[bug][passes][issue-297]") {
    const auto input = make_small();
    SECTION("propagate_constants, target out of range, no events") {
        const auto result =
            passes::propagate_constants(input.circuit, passes::pass_options{.atoms = {.targets = {pc::target_id(5)}}});
        CHECK(has_code(result.report.diagnostics, "pc.pass.invalid-atom-split"));
    }
    SECTION("coalesce, target out of range") {
        const auto result = passes::coalesce(
            input.circuit, passes::pass_options{.atoms = {.targets = {pc::target_id(5)}, .events = {input.event}}});
        CHECK(has_code(result.report.diagnostics, "pc.pass.invalid-atom-split"));
    }
    SECTION("coalesce, a gate as atom event") {
        const auto result = passes::coalesce(
            input.circuit, passes::pass_options{.atoms = {.targets = {pc::target_id(0)}, .events = {input.gate}}});
        CHECK(has_code(result.report.diagnostics, "pc.pass.invalid-atom-split"));
    }
}

// docs/ir-pc-passes.md (Levels) and passes.h: C1 runs rounds "until a round changes neither the node nor the
// edge count (at most max_rounds, default 4)". compile() runs std::max(max_rounds, 1) rounds, so
// max_rounds = 0 still runs propagate_constants, eliminate_single_input and coalesce once.
TEST_CASE("C1 with max_rounds = 0 runs no rewrite round", "[bug][passes][issue-297]") {
    const auto input = make_small();
    const auto result =
        passes::compile(input.circuit, passes::compile_level::c1, passes::pipeline_options{.max_rounds = 0});
    REQUIRE(result.ok());
    // fold_zero_probability and fold_unit_probability only.
    CHECK(result.passes.size() == 2);
}

// docs/IR-PC-REQUIREMENTS.md section 4 limits the opcode set, and section 6 says sealing validates the
// circuit. An op code outside the enum seals without a diagnostic, and the passes then carry it as an
// opaque node.
TEST_CASE("seal rejects an op code outside the ir::pc opcode set", "[bug][ir-pc][issue-297]") {
    pc::builder builder;
    const auto event = builder.add_basic_event(builder.add_literal_probability(0.5));
    const auto gate = builder.add_gate(static_cast<pc::op_code>(200), {edge{.source = event}});
    (void)builder.add_target(gate);
    const auto sealed = std::move(builder).seal();
    CHECK_FALSE(sealed.ok());
}
