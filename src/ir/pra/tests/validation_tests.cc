#include <catch2/catch_test_macros.hpp>

#include "pra_test_support.h"

#include "ir/pra/builder.h"

#include <array>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace pra = fuzzybits::ir::pra;
using pra::test::codes;
using pra::test::count;
using pra::test::message;
using pra::test::named;
using pra::test::render;

[[nodiscard]] auto literal(const double value, const pra::quantity_kind quantity = pra::quantity_kind::probability)
    -> pra::expression_spec {
    return pra::expression_spec::literal(value, quantity);
}

[[nodiscard]] auto mentions(std::string const &text, const std::string_view part) -> bool {
    return text.find(part) != std::string::npos;
}

// Seals and requires rejection with exactly one diagnostic, of `code`.
[[nodiscard]] auto rejected_with(pra::builder &&builder, std::string const &code) -> std::string {
    auto result = std::move(builder).seal();
    INFO(render(result.diagnostics));
    REQUIRE_FALSE(result.ok());
    REQUIRE_FALSE(result.output.valid());
    REQUIRE(codes(result.diagnostics) == std::vector<std::string>{code});
    return result.diagnostics.items().front().message;
}

void require_ok(pra::builder &&builder) {
    auto result = std::move(builder).seal();
    INFO(render(result.diagnostics));
    REQUIRE(result.ok());
    REQUIRE(result.diagnostics.empty());
}

} // namespace

TEST_CASE("PRA validation: every element needs a stable name", "[ir][pra][validation]") {
    pra::builder builder;
    (void)builder.add_basic_event({}, literal(0.1));
    const auto text = rejected_with(std::move(builder), "ir.pra.missing-name");
    CHECK(mentions(text, "basic event #0"));
}

TEST_CASE("PRA validation: stable names are unique within a kind", "[ir][pra][validation]") {
    pra::builder builder;
    const auto first_file = builder.add_source_file({.path = "a.ftp", .format = "ftp"});
    const auto second_file = builder.add_source_file({.path = "b.ftp", .format = "ftp"});
    const pra::source_location first{.file = first_file, .line = 3};
    const pra::source_location second{.file = second_file, .line = 7};
    const auto event = builder.add_basic_event({.stable_name = "B"}, literal(0.1));

    SECTION("gates defined in two files") {
        (void)builder.add_gate({.stable_name = "G1", .sources = {&first, 1}}, pra::gate_kind::or_gate,
                               {{.event = event}});
        (void)builder.add_gate({.stable_name = "G1", .sources = {&second, 1}}, pra::gate_kind::and_gate,
                               {{.event = event}});
        const auto text = rejected_with(std::move(builder), "ir.pra.duplicate-name");
        CHECK(mentions(text, "duplicate gate name 'G1'"));
        CHECK(mentions(text, "2 definitions"));
        CHECK(mentions(text, "a.ftp:3"));
        CHECK(mentions(text, "b.ftp:7"));
    }

    SECTION("one diagnostic per duplicated name") {
        (void)builder.add_basic_event({.stable_name = "B"}, literal(0.2));
        (void)builder.add_basic_event({.stable_name = "B"}, literal(0.3));
        const auto text = rejected_with(std::move(builder), "ir.pra.duplicate-name");
        CHECK(mentions(text, "basic event name 'B' (3 definitions)"));
    }

    SECTION("the same name in different kinds is not a duplicate") {
        const auto gate = builder.add_gate({.stable_name = "B"}, pra::gate_kind::or_gate, {{.event = event}});
        (void)builder.add_house_event({.stable_name = "B"}, true);
        (void)builder.add_initiating_event({.stable_name = "B"}, event);
        (void)builder.add_target({.stable_name = "B"}, pra::domain_target_kind::gate, gate);
        require_ok(std::move(builder));
    }
}

TEST_CASE("PRA validation: unqualified names must match exactly one admissible kind", "[ir][pra][validation]") {
    pra::builder builder;
    const auto event = builder.add_basic_event({.stable_name = "B"}, literal(0.1));
    (void)builder.add_gate({.stable_name = "X"}, pra::gate_kind::or_gate, {{.event = event}});
    (void)builder.add_basic_event({.stable_name = "X"}, literal(0.2));

    SECTION("ambiguous") {
        (void)builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {named("X")});
        const auto text = rejected_with(std::move(builder), "ir.pra.ambiguous-reference");
        CHECK(mentions(text, "gate 'TOP' operand 1 references 'X', which names a basic event and a gate"));
    }

    SECTION("qualified") {
        (void)builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate,
                               {{.event = pra::reference::named(pra::element_kind::gate, "X")}});
        require_ok(std::move(builder));
    }
}

TEST_CASE("PRA validation: dangling references", "[ir][pra][validation]") {
    pra::builder builder;
    const auto event = builder.add_basic_event({.stable_name = "B"}, literal(0.1));
    const auto tree = builder.add_fault_tree({.stable_name = "FT"});
    const auto gate = builder.add_gate({.stable_name = "G"}, pra::gate_kind::or_gate, {{.event = event}}, tree);
    builder.add_fault_tree_top(tree, gate);
    const std::string code = "ir.pra.dangling-reference";

    SECTION("operand name") {
        (void)builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = event}, named("NOPE")});
        CHECK(
            mentions(rejected_with(std::move(builder), code), "gate 'TOP' operand 2 references unknown event 'NOPE'"));
    }
    SECTION("operand id") {
        (void)builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = pra::gate_id(42)}});
        CHECK(mentions(rejected_with(std::move(builder), code), "nonexistent gate #42"));
    }
    SECTION("empty operand reference") {
        (void)builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = pra::reference{}}});
        CHECK(mentions(rejected_with(std::move(builder), code), "empty reference"));
    }
    SECTION("initiating event") {
        (void)builder.add_initiating_event({.stable_name = "IE"}, pra::reference::named("NOPE"));
        CHECK(mentions(rejected_with(std::move(builder), code), "unknown basic event 'NOPE'"));
    }
    SECTION("fault-tree top gate") {
        builder.add_fault_tree_top(tree, pra::reference::named("NOPE"));
        CHECK(mentions(rejected_with(std::move(builder), code),
                       "fault tree 'FT' top gate references unknown gate 'NOPE'"));
    }
    SECTION("top gate of a nonexistent fault tree") {
        builder.add_fault_tree_top(pra::fault_tree_id(3), gate);
        CHECK(mentions(rejected_with(std::move(builder), code), "nonexistent fault tree #3"));
    }
    SECTION("gate owner") {
        (void)builder.add_gate({.stable_name = "H"}, pra::gate_kind::or_gate, {{.event = event}},
                               pra::fault_tree_id(5));
        CHECK(mentions(rejected_with(std::move(builder), code), "belongs to nonexistent fault tree #5"));
    }
    SECTION("expression") {
        (void)builder.add_basic_event({.stable_name = "C"}, pra::expression_id(9));
        CHECK(mentions(rejected_with(std::move(builder), code), "nonexistent expression #9"));
    }
    SECTION("source file") {
        const pra::source_location where{.file = pra::source_file_id(4), .line = 2};
        (void)builder.add_basic_event({.stable_name = "C", .sources = {&where, 1}}, literal(0.1));
        CHECK(mentions(rejected_with(std::move(builder), code), "nonexistent source file #4"));
    }
    SECTION("attribute owner") {
        builder.add_attribute(pra::gate_id(7), "key", "value");
        CHECK(mentions(rejected_with(std::move(builder), code), "attribute 'key' is attached to nonexistent gate #7"));
    }
    SECTION("source owner") {
        builder.add_source(pra::basic_event_id(9), {});
        CHECK(mentions(rejected_with(std::move(builder), code), "nonexistent basic event #9"));
    }
    SECTION("target subject name") {
        (void)builder.add_target({.stable_name = "T"}, pra::domain_target_kind::gate, pra::reference::named("NOPE"));
        CHECK(mentions(rejected_with(std::move(builder), code), "target 'T' references unknown gate 'NOPE'"));
    }
    SECTION("target subject id") {
        (void)builder.add_target({.stable_name = "T"}, pra::domain_target_kind::basic_event, pra::basic_event_id(3));
        CHECK(mentions(rejected_with(std::move(builder), code), "nonexistent basic event #3"));
    }
    SECTION("target subject named only in another kind") {
        (void)builder.add_target({.stable_name = "T"}, pra::domain_target_kind::gate, pra::reference::named("B"));
        CHECK(mentions(rejected_with(std::move(builder), code), "unknown gate 'B'"));
    }
}

TEST_CASE("PRA validation: references to a kind the use site does not admit", "[ir][pra][validation]") {
    pra::builder builder;
    const auto event = builder.add_basic_event({.stable_name = "B"}, literal(0.1));
    const auto tree = builder.add_fault_tree({.stable_name = "FT"});
    const auto gate = builder.add_gate({.stable_name = "G"}, pra::gate_kind::or_gate, {{.event = event}}, tree);
    builder.add_fault_tree_top(tree, gate);
    const std::string code = "ir.pra.invalid-reference";

    SECTION("operand id of a fault tree") {
        (void)builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate, {{.event = tree}});
        CHECK(mentions(rejected_with(std::move(builder), code), "references fault tree 'FT' where an event"));
    }
    SECTION("operand name qualified with a non-event kind") {
        (void)builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::or_gate,
                               {{.event = pra::reference::named(pra::element_kind::fault_tree, "FT")}});
        CHECK(mentions(rejected_with(std::move(builder), code), "names a fault tree 'FT' where an event"));
    }
    SECTION("target subject of another kind") {
        (void)builder.add_target({.stable_name = "T"}, pra::domain_target_kind::gate, event);
        CHECK(mentions(rejected_with(std::move(builder), code), "references basic event 'B' where a gate is required"));
    }
    SECTION("initiating event on a gate") {
        (void)builder.add_initiating_event({.stable_name = "IE"}, gate);
        CHECK(mentions(rejected_with(std::move(builder), code), "where a basic event is required"));
    }
    SECTION("fault-tree top on a basic event") {
        builder.add_fault_tree_top(tree, event);
        CHECK(mentions(rejected_with(std::move(builder), code), "where a gate is required"));
    }
}

TEST_CASE("PRA validation: gate dependency cycles", "[ir][pra][validation]") {
    pra::builder builder;
    const auto event = builder.add_basic_event({.stable_name = "E"}, literal(0.1));

    SECTION("self loop") {
        (void)builder.add_gate({.stable_name = "G"}, pra::gate_kind::or_gate, {named("G"), {.event = event}});
        const auto text = rejected_with(std::move(builder), "ir.pra.cycle");
        CHECK(mentions(text, "length 1: 'G' -> 'G'"));
    }

    SECTION("through complements and a cardinality gate") {
        (void)builder.add_gate({.stable_name = "A"}, pra::gate_kind::and_gate, {named("BB"), {.event = event}});
        (void)builder.add_cardinality({.stable_name = "BB"}, {.low = 1, .high = 2},
                                      {named("C", true), {.event = event}});
        (void)builder.add_gate({.stable_name = "C"}, pra::gate_kind::or_gate, {named("A")});
        (void)builder.add_gate({.stable_name = "D"}, pra::gate_kind::or_gate, {named("A")});
        const auto text = rejected_with(std::move(builder), "ir.pra.cycle");
        CHECK(mentions(text, "gate 'A' is on a gate dependency cycle of length 3: 'A' -> 'BB' -> 'C' -> 'A'"));
        CHECK_FALSE(mentions(text, "'D'"));
    }

    SECTION("each disjoint cycle is reported") {
        (void)builder.add_gate({.stable_name = "X"}, pra::gate_kind::or_gate, {named("Y")});
        (void)builder.add_gate({.stable_name = "Y"}, pra::gate_kind::or_gate, {named("X")});
        (void)builder.add_gate({.stable_name = "P"}, pra::gate_kind::or_gate, {named("Q")});
        (void)builder.add_gate({.stable_name = "Q"}, pra::gate_kind::nor_gate, {named("P")});
        auto result = std::move(builder).seal();
        INFO(render(result.diagnostics));
        REQUIRE_FALSE(result.ok());
        CHECK(count(result.diagnostics, "ir.pra.cycle") == 2);
        CHECK(result.diagnostics.size() == 2);
    }
}

TEST_CASE("PRA validation: n-ary gates need operands", "[ir][pra][validation]") {
    for (const auto kind : std::array{pra::gate_kind::and_gate, pra::gate_kind::or_gate, pra::gate_kind::xor_gate,
                                      pra::gate_kind::nand_gate, pra::gate_kind::nor_gate, pra::gate_kind::xnor_gate}) {
        pra::builder builder;
        (void)builder.add_gate({.stable_name = "G"}, kind, {});
        CHECK(mentions(rejected_with(std::move(builder), "ir.pra.empty-gate"), "gate without operands"));
    }
    pra::builder builder;
    (void)builder.add_cardinality({.stable_name = "K"}, {.low = 0, .high = 0}, {});
    CHECK(mentions(rejected_with(std::move(builder), "ir.pra.empty-gate"), "cardinality gate without operands"));
}

TEST_CASE("PRA validation: not gates take exactly one operand", "[ir][pra][validation]") {
    pra::builder builder;
    const auto event = builder.add_basic_event({.stable_name = "E"}, literal(0.1));

    SECTION("none") {
        (void)builder.add_gate({.stable_name = "N"}, pra::gate_kind::not_gate, {});
        CHECK(mentions(rejected_with(std::move(builder), "ir.pra.invalid-arity"), "not gate with 0 operands"));
    }
    SECTION("two") {
        (void)builder.add_gate({.stable_name = "N"}, pra::gate_kind::not_gate, {{.event = event}, {.event = event}});
        CHECK(mentions(rejected_with(std::move(builder), "ir.pra.invalid-arity"), "not gate with 2 operands"));
    }
}

TEST_CASE("PRA validation: literal markers only on complemented basic-event operands", "[ir][pra][validation]") {
    pra::builder builder;
    const auto event = builder.add_basic_event({.stable_name = "E"}, literal(0.1));
    const auto house = builder.add_house_event({.stable_name = "H"}, true);
    const auto inner = builder.add_gate({.stable_name = "I"}, pra::gate_kind::or_gate, {{.event = event}});

    SECTION("uncomplemented basic event") {
        (void)builder.add_gate({.stable_name = "G"}, pra::gate_kind::and_gate, {{.event = event, .literal = true}});
        CHECK(mentions(rejected_with(std::move(builder), "ir.pra.invalid-literal-operand"),
                       "marked literal but not complemented"));
    }
    SECTION("complemented gate") {
        (void)builder.add_gate({.stable_name = "G"}, pra::gate_kind::and_gate,
                               {{.event = inner, .complement = true, .literal = true}});
        CHECK(mentions(rejected_with(std::move(builder), "ir.pra.invalid-literal-operand"), "gate 'I'"));
    }
    SECTION("complemented gate by name, resolved at seal") {
        (void)builder.add_gate({.stable_name = "G"}, pra::gate_kind::and_gate,
                               {{.event = pra::reference::named("I"), .complement = true, .literal = true}});
        CHECK(mentions(rejected_with(std::move(builder), "ir.pra.invalid-literal-operand"), "not a basic event"));
    }
    SECTION("complemented house event") {
        (void)builder.add_gate({.stable_name = "G"}, pra::gate_kind::and_gate,
                               {{.event = house, .complement = true, .literal = true}});
        CHECK(mentions(rejected_with(std::move(builder), "ir.pra.invalid-literal-operand"), "house event 'H'"));
    }
    SECTION("dangling operands are reported once, as dangling") {
        (void)builder.add_gate({.stable_name = "G"}, pra::gate_kind::and_gate,
                               {{.event = pra::reference::named("NOPE"), .complement = true, .literal = true}});
        (void)rejected_with(std::move(builder), "ir.pra.dangling-reference");
    }
    SECTION("complemented basic event, by id or by name") {
        (void)builder.add_gate({.stable_name = "G"}, pra::gate_kind::and_gate,
                               {{.event = event, .complement = true, .literal = true},
                                {.event = pra::reference::named("E"), .complement = true, .literal = true}});
        require_ok(std::move(builder));
    }
}

TEST_CASE("PRA validation: cardinality bounds satisfy 0 <= low <= high <= n", "[ir][pra][validation]") {
    pra::builder builder;
    std::vector<pra::operand_spec> operands;
    for (const auto *name : {"A", "B", "C"}) {
        operands.push_back({.event = builder.add_basic_event({.stable_name = name}, literal(0.1))});
    }
    const std::string code = "ir.pra.invalid-cardinality";

    SECTION("low above high") {
        (void)builder.add_cardinality({.stable_name = "K"}, {.low = 3, .high = 2}, operands);
        CHECK(mentions(rejected_with(std::move(builder), code), "bounds (3, 2) outside 0 <= low <= high <= 3"));
    }
    SECTION("high above the fan-in") {
        (void)builder.add_cardinality({.stable_name = "K"}, {.low = 2, .high = 4}, operands);
        CHECK(mentions(rejected_with(std::move(builder), code), "bounds (2, 4)"));
    }
    SECTION("at-least-k with k above the fan-in") {
        (void)builder.add_cardinality({.stable_name = "K"}, {.low = 4, .high = 3}, operands);
        CHECK(mentions(rejected_with(std::move(builder), code), "bounds (4, 3)"));
    }
    SECTION("boundary bounds are accepted") {
        (void)builder.add_cardinality({.stable_name = "NONE"}, {.low = 0, .high = 0}, operands);
        (void)builder.add_cardinality({.stable_name = "ANY"}, {.low = 0, .high = 3}, operands);
        (void)builder.add_cardinality({.stable_name = "ALL"}, {.low = 3, .high = 3}, operands);
        (void)builder.add_cardinality({.stable_name = "TWO_OF_THREE"}, {.low = 2, .high = 3}, operands);
        require_ok(std::move(builder));
    }
}

TEST_CASE("PRA validation: literal values lie in their domain", "[ir][pra][validation]") {
    pra::builder builder;
    const std::string code = "ir.pra.invalid-literal";

    SECTION("probability above one") {
        (void)builder.add_basic_event({.stable_name = "B"}, literal(1.5));
        CHECK(mentions(rejected_with(std::move(builder), code),
                       "basic event 'B' has literal probability 1.5 outside [0, 1]"));
    }
    SECTION("negative probability") {
        (void)builder.add_basic_event({.stable_name = "B"}, literal(-0.25));
        CHECK(mentions(rejected_with(std::move(builder), code), "-0.25"));
    }
    SECTION("NaN") {
        (void)builder.add_basic_event({.stable_name = "B"}, literal(std::numeric_limits<double>::quiet_NaN()));
        CHECK(mentions(rejected_with(std::move(builder), code), "non-finite"));
    }
    SECTION("infinite frequency") {
        (void)builder.add_basic_event({.stable_name = "B"},
                                      literal(std::numeric_limits<double>::infinity(), pra::quantity_kind::frequency));
        CHECK(mentions(rejected_with(std::move(builder), code), "non-finite"));
    }
    SECTION("negative frequency") {
        (void)builder.add_basic_event({.stable_name = "B"}, literal(-1.0, pra::quantity_kind::frequency));
        CHECK(mentions(rejected_with(std::move(builder), code), "negative literal frequency -1"));
    }
    SECTION("domain boundaries are accepted") {
        (void)builder.add_basic_event({.stable_name = "ZERO"}, literal(0.0));
        (void)builder.add_basic_event({.stable_name = "ONE"}, literal(1.0));
        (void)builder.add_basic_event({.stable_name = "F0"}, literal(0.0, pra::quantity_kind::frequency));
        (void)builder.add_basic_event({.stable_name = "F3"}, literal(3.0, pra::quantity_kind::frequency));
        require_ok(std::move(builder));
    }
}

TEST_CASE("PRA validation: parameter and unsupported expressions carry their identity", "[ir][pra][validation]") {
    pra::builder builder;
    const std::string code = "ir.pra.invalid-expression";

    SECTION("parameter without a name") {
        (void)builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::parameter(""));
        CHECK(mentions(rejected_with(std::move(builder), code), "without a parameter name"));
    }
    SECTION("unsupported without a diagnostic id") {
        (void)builder.add_basic_event({.stable_name = "B"}, pra::expression_spec::unsupported(""));
        CHECK(mentions(rejected_with(std::move(builder), code), "without a diagnostic id"));
    }
    SECTION("unreferenced expressions are checked too") {
        (void)builder.add_expression(pra::expression_spec::parameter(""));
        CHECK(mentions(rejected_with(std::move(builder), code), "expression #0"));
    }
}

TEST_CASE("PRA validation: fault-tree top gates belong to their tree", "[ir][pra][validation]") {
    pra::builder builder;
    const auto event = builder.add_basic_event({.stable_name = "E"}, literal(0.1));
    const auto first_tree = builder.add_fault_tree({.stable_name = "FT1"});
    const auto second_tree = builder.add_fault_tree({.stable_name = "FT2"});
    const auto first_gate =
        builder.add_gate({.stable_name = "G1"}, pra::gate_kind::or_gate, {{.event = event}}, first_tree);
    const auto second_gate =
        builder.add_gate({.stable_name = "G2"}, pra::gate_kind::or_gate, {{.event = event}}, second_tree);
    const auto loose_gate = builder.add_gate({.stable_name = "G3"}, pra::gate_kind::or_gate, {{.event = event}});
    const std::string code = "ir.pra.invalid-fault-tree-top";

    SECTION("owned by another tree") {
        builder.add_fault_tree_top(first_tree, second_gate);
        CHECK(mentions(rejected_with(std::move(builder), code), "but the gate belongs to fault tree 'FT2'"));
    }
    SECTION("owned by no tree") {
        builder.add_fault_tree_top(first_tree, loose_gate);
        CHECK(mentions(rejected_with(std::move(builder), code), "but the gate belongs to no fault tree"));
    }
    SECTION("listed twice") {
        builder.add_fault_tree_top(first_tree, first_gate);
        builder.add_fault_tree_top(first_tree, pra::reference::named("G1"));
        CHECK(mentions(rejected_with(std::move(builder), code), "as a top gate more than once"));
    }
}

TEST_CASE("PRA validation: fault_tree_top targets need exactly one top gate", "[ir][pra][validation]") {
    pra::builder builder;
    const auto event = builder.add_basic_event({.stable_name = "E"}, literal(0.1));
    const auto tree = builder.add_fault_tree({.stable_name = "FT"});
    const auto first_gate = builder.add_gate({.stable_name = "G1"}, pra::gate_kind::or_gate, {{.event = event}}, tree);
    const auto second_gate = builder.add_gate({.stable_name = "G2"}, pra::gate_kind::or_gate, {{.event = event}}, tree);
    const std::string code = "ir.pra.invalid-target";

    SECTION("no top gate") {
        (void)builder.add_target({.stable_name = "T"}, pra::domain_target_kind::fault_tree_top, tree);
        CHECK(mentions(rejected_with(std::move(builder), code), "which has 0 top gates"));
    }
    SECTION("two top gates") {
        builder.add_fault_tree_top(tree, first_gate);
        builder.add_fault_tree_top(tree, second_gate);
        (void)builder.add_target({.stable_name = "T"}, pra::domain_target_kind::fault_tree_top, tree);
        CHECK(mentions(rejected_with(std::move(builder), code), "which has 2 top gates"));
    }
}

TEST_CASE("PRA validation: sequence and end-state targets are reserved", "[ir][pra][validation]") {
    for (const auto kind : {pra::domain_target_kind::sequence, pra::domain_target_kind::end_state}) {
        pra::builder builder;
        (void)builder.add_target({.stable_name = "S"}, kind, pra::reference::named("S1"));
        CHECK(mentions(rejected_with(std::move(builder), "ir.pra.unsupported-target-kind"),
                       "reserved until event trees"));
    }
}

TEST_CASE("PRA validation reports every error, not only the first", "[ir][pra][validation]") {
    pra::builder builder;
    const auto event = builder.add_basic_event({}, literal(2.0));
    (void)builder.add_gate({.stable_name = "G"}, pra::gate_kind::or_gate, {named("NOPE")});
    (void)builder.add_gate({.stable_name = "G"}, pra::gate_kind::and_gate, {});
    (void)builder.add_gate({.stable_name = "N"}, pra::gate_kind::not_gate, {{.event = event}, {.event = event}});
    (void)builder.add_gate({.stable_name = "C"}, pra::gate_kind::or_gate, {named("C")});
    (void)builder.add_target({.stable_name = "T"}, pra::domain_target_kind::sequence, pra::reference::named("S"));

    auto result = std::move(builder).seal();
    INFO(render(result.diagnostics));
    REQUIRE_FALSE(result.ok());
    for (const auto *code :
         {"ir.pra.missing-name", "ir.pra.invalid-literal", "ir.pra.dangling-reference", "ir.pra.duplicate-name",
          "ir.pra.empty-gate", "ir.pra.invalid-arity", "ir.pra.cycle", "ir.pra.unsupported-target-kind"}) {
        CHECK(count(result.diagnostics, code) == 1);
    }
    CHECK(result.diagnostics.size() == 8);
}
