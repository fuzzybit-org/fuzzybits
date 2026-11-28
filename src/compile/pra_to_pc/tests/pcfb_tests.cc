#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include "compile/exact_eval.h"
#include "compile/pra_to_pc/descriptor.h"
#include "io/pcfb/descriptor.h"
#include "ir/pra/builder.h"

#include "pcfb.h"
#include "pcfb/io.h"
#include "pcfb/validate.h"
#include "pcfb_generated.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace pra_to_pc_test;
namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;

[[nodiscard]] auto build_model() -> pra::model {
    pra::builder builder;
    builder.set_project("demo");
    const auto ie = builder.add_basic_event({.stable_name = "IE"}, pra::expression_spec::literal(0.05));
    const auto ie2 = builder.add_basic_event({.stable_name = "IE2"}, pra::expression_spec::literal(0.02));
    const auto pump = builder.add_basic_event({.stable_name = "PUMP"}, pra::expression_spec::literal(0.1));
    const auto valve = builder.add_basic_event({.stable_name = "VALVE"}, pra::expression_spec::literal(0.2));
    const auto flag = builder.add_basic_event({.stable_name = "FLAG"}, pra::expression_spec::literal(1.0));
    const auto crew = builder.add_basic_event({.stable_name = "CREW"}, pra::expression_spec::literal(0.3));
    (void)builder.add_initiating_event({.stable_name = "IE"}, ie);
    (void)builder.add_initiating_event({.stable_name = "IE2"}, ie2);
    const auto maintenance = builder.add_house_event({.stable_name = "MAINT"}, true);
    const auto tree = builder.add_fault_tree({.stable_name = "FT"});
    const auto cooling = builder.add_cardinality({.stable_name = "COOLING"}, {.low = 2, .high = 3},
                                                 {{.event = pump}, {.event = valve}, {.event = crew, .complement = true}}, tree);
    const auto deleted = builder.add_gate({.stable_name = "DEL"}, pra::gate_kind::and_gate, {{.event = flag}, {.event = valve}}, tree);
    const auto failed = builder.add_gate({.stable_name = "FAILED"}, pra::gate_kind::and_gate,
                                         {{.event = cooling}, {.event = deleted, .complement = true}, {.event = maintenance}}, tree);
    const auto initiators = builder.add_gate({.stable_name = "INIT"}, pra::gate_kind::or_gate, {{.event = ie}, {.event = ie2}}, tree);
    const auto top = builder.add_gate({.stable_name = "TOP"}, pra::gate_kind::and_gate, {{.event = initiators}, {.event = failed}}, tree);
    const auto parity = builder.add_gate({.stable_name = "PARITY"}, pra::gate_kind::xor_gate, {{.event = pump}, {.event = ie}}, tree);
    builder.add_fault_tree_top(tree, top);
    (void)builder.add_target({.stable_name = "FT"}, pra::domain_target_kind::fault_tree_top, tree);
    (void)builder.add_target({.stable_name = "PARITY"}, pra::domain_target_kind::gate, parity, pra::target_role::watch);
    return seal(std::move(builder));
}

void require_valid(::pcfb::ir::Circuit const &circuit) {
    std::ostringstream text;
    const auto diagnostics = ::pcfb::validate(circuit);
    for (auto const &diagnostic : diagnostics) {
        text << diagnostic.location << ": " << diagnostic.message << '\n';
    }
    INFO(text.str());
    REQUIRE(diagnostics.empty());
}

} // namespace

TEST_CASE("pra_to_pc lowering writes a valid, named .pcfb with exact probabilities", "[compile][pra_to_pc][pcfb]") {
    const auto model = build_model();
    for (const auto rule : {std::optional<std::uint32_t>{}, std::optional<std::uint32_t>{1}}) {
        for (const bool override_flag : {false, true}) {
            CAPTURE(rule.has_value(), override_flag);
            lowering::quantification_config config;
            config.initiators = lowering::initiator_values::probability;
            config.default_hull.max_initiators = rule;
            config.watch_targets = {{.name = "PARITY"}};
            std::vector<reference_override> overrides;
            if (override_flag) {
                config.overrides = {{.name = "DEL", .value = false}};
                overrides.push_back({.element = model.find_gate("DEL"), .value = false});
            }
            const auto artifact = require_lower(model, config);
            require_equivalent(model, artifact, overrides);

            auto options = lowering::pcfb_write_options(artifact);
            INFO(render(options.diagnostics()));
            REQUIRE(options);
            CHECK(options.value().model_name == "demo");
            CHECK(options.value().target_names.at(pc::target_id(0)) == "FT");
            CHECK(options.value().target_names.at(pc::target_id(1)) == "PARITY");
            CHECK(options.value().node_names.at(artifact.projection.node_of(model.find_basic_event("PUMP"))) == "PUMP");

            auto written = writer::write(artifact.circuit.to_handle(), options.value());
            INFO(render(written.diagnostics()));
            REQUIRE(written);
            auto circuit = ::pcfb::io::from_bytes(written.value().bytes);
            REQUIRE(circuit);
            require_valid(*circuit);

            auto const *sub = (*circuit)->subcircuits()->Get(0);
            REQUIRE(sub->queries()->size() == 2);
            CHECK(sub->queries()->Get(0)->name()->str() == "FT");
            CHECK(sub->queries()->Get(1)->name()->str() == "PARITY");
            const auto top_tensor = written.value().tensor_by_node[artifact.projection.node_of(model.find_gate("TOP")).raw()];
            CHECK(sub->operators()->Get(top_tensor)->name()->str() == "TOP");

            const pra_reference reference(model, overrides);
            for (std::uint32_t query = 0; query < 2; ++query) {
                // Probability queries: the rule is never lowered into them.
                auto const &binding = artifact.projection.target_bindings()[query];
                const auto expected = reference.target_probability(binding.subject, rule_of(binding));
                const auto exact = fuzzybits::compile::exact_probability(*circuit, query);
                CAPTURE(query, expected, exact);
                CHECK(std::abs(exact - expected) <= 1e-12);
            }
        }
    }
}

namespace {

[[nodiscard]] auto query_summary(::pcfb::ir::Circuit const &circuit) -> std::vector<std::pair<std::string, ::pcfb::QueryKind>> {
    std::vector<std::pair<std::string, ::pcfb::QueryKind>> rows;
    for (auto const *query : *circuit->subcircuits()->Get(0)->queries()) {
        rows.emplace_back(query->name()->str(), query->kind());
    }
    return rows;
}

[[nodiscard]] auto string_meta(::pcfb::ir::Circuit const &circuit, std::string const &key) -> std::optional<std::string> {
    auto const *entry = circuit->metadata()->LookupByKey(key.c_str());
    if (entry == nullptr || entry->value_type() != ::pcfb::MetadataValue::StringValue) {
        return std::nullopt;
    }
    return entry->value_as_StringValue()->value()->str();
}

[[nodiscard]] auto int_meta(::pcfb::ir::Circuit const &circuit, std::string const &key) -> std::optional<std::int64_t> {
    auto const *entry = circuit->metadata()->LookupByKey(key.c_str());
    if (entry == nullptr || entry->value_type() != ::pcfb::MetadataValue::IntValue) {
        return std::nullopt;
    }
    return entry->value_as_IntValue()->value();
}

// SAFE and LIT observe the same gate shape; LIT's cone has an operand marked `literal`, SAFE's has a plain complement.
[[nodiscard]] auto build_literal_model() -> pra::model {
    pra::builder builder;
    builder.set_project("literals");
    const auto x = builder.add_basic_event({.stable_name = "X"}, pra::expression_spec::literal(0.1));
    const auto y = builder.add_basic_event({.stable_name = "Y"}, pra::expression_spec::literal(0.2));
    const auto z = builder.add_basic_event({.stable_name = "Z"}, pra::expression_spec::literal(0.3));
    const auto tree = builder.add_fault_tree({.stable_name = "FT"});
    const auto safe = builder.add_gate({.stable_name = "SAFE_GATE"}, pra::gate_kind::and_gate, {{.event = x}, {.event = z, .complement = true}}, tree);
    const auto lit =
        builder.add_gate({.stable_name = "LIT_GATE"}, pra::gate_kind::and_gate, {{.event = x}, {.event = y, .complement = true, .literal = true}}, tree);
    (void)builder.add_target({.stable_name = "SAFE"}, pra::domain_target_kind::gate, safe);
    (void)builder.add_target({.stable_name = "LIT"}, pra::domain_target_kind::gate, lit);
    return seal(std::move(builder));
}

} // namespace

TEST_CASE("pra_to_pc write options write one query per binding, of its kind", "[compile][pra_to_pc][pcfb]") {
    const auto model = build_model();
    lowering::quantification_config config;
    config.initiators = lowering::initiator_values::probability;
    config.default_kinds = {pc::query_kind::probability, pc::query_kind::monotone_hull};
    // A request without kinds takes the default kinds; PARITY asks for the probability only.
    config.watch_targets = {{.name = "PARITY", .kinds = {pc::query_kind::probability}}};
    const auto artifact = require_lower(model, config);
    auto const bindings = artifact.projection.target_bindings();
    REQUIRE(bindings.size() == 3);
    CHECK(bindings[0].kind == pc::query_kind::probability);
    CHECK(bindings[1].kind == pc::query_kind::monotone_hull);
    CHECK(bindings[0].node == bindings[1].node);

    auto options = lowering::pcfb_write_options(artifact);
    INFO(render(options.diagnostics()));
    REQUIRE(options);
    using kind = writer::query_kind;
    CHECK(options.value().target_kinds.at(pc::target_id(0)) == std::vector<kind>{kind::probability});
    CHECK(options.value().target_kinds.at(pc::target_id(1)) == std::vector<kind>{kind::monotone_hull_probability});
    CHECK(options.value().target_kinds.at(pc::target_id(2)) == std::vector<kind>{kind::probability});

    auto written = writer::write(artifact.circuit.to_handle(), options.value());
    INFO(render(written.diagnostics()));
    REQUIRE(written);
    auto circuit = ::pcfb::io::from_bytes(written.value().bytes);
    REQUIRE(circuit);
    require_valid(*circuit);
    CHECK((*circuit)->schema_version() == ::pcfb::kSchemaVersionV12);
    using rows = std::vector<std::pair<std::string, ::pcfb::QueryKind>>;
    CHECK(query_summary(*circuit) ==
          rows{{"FT", ::pcfb::QueryKind::PROBABILITY}, {"FT", ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY}, {"PARITY", ::pcfb::QueryKind::PROBABILITY}});
    auto const *queries = (*circuit)->subcircuits()->Get(0)->queries();
    CHECK(queries->Get(0)->target() == queries->Get(1)->target());

    // The kind is Query.kind only; the metadata names the target.
    CHECK_FALSE(string_meta(*circuit, "compile.query.0.negation"));
    CHECK(string_meta(*circuit, "compile.query.0.target") == "FT");
    CHECK(string_meta(*circuit, "compile.query.1.target") == "FT");
    CHECK(string_meta(*circuit, "compile.query.2.target") == "PARITY");
    // The hull reading is a parameter of the hull query only; no rule was asked for.
    CHECK_FALSE(string_meta(*circuit, "compile.query.0.literal_mode"));
    CHECK(string_meta(*circuit, "compile.query.1.literal_mode") == "split");
    CHECK_FALSE(string_meta(*circuit, "compile.query.2.literal_mode"));
    for (const auto index : {0, 1, 2}) {
        CHECK_FALSE(int_meta(*circuit, "compile.query." + std::to_string(index) + ".max_initiators_per_cutset"));
    }
}

TEST_CASE("pra_to_pc write options put the initiator rule on hull queries only", "[compile][pra_to_pc][pcfb]") {
    const auto model = build_model();
    lowering::quantification_config config;
    config.initiators = lowering::initiator_values::probability;
    config.default_kinds = {pc::query_kind::probability, pc::query_kind::monotone_hull};
    config.default_hull.max_initiators = 1;
    config.watch_targets = {{.name = "PARITY", .kinds = {pc::query_kind::probability}}};
    const auto artifact = require_lower(model, config);
    // FT's cone holds two initiating events: its probability query stays on the subject, its hull query
    // gets the rule on another node. PARITY (watch, probability only) never gets it.
    auto const bindings = artifact.projection.target_bindings();
    REQUIRE(bindings.size() == 3);
    CHECK(bindings[0].kind == pc::query_kind::probability);
    CHECK_FALSE(bindings[0].initiator_delete_term.valid());
    CHECK(bindings[1].kind == pc::query_kind::monotone_hull);
    CHECK(bindings[1].hull->max_initiators == 1U);
    CHECK(bindings[1].initiator_delete_term.valid());
    CHECK_FALSE(bindings[2].initiator_delete_term.valid());
    require_equivalent(model, artifact, {});

    auto options = lowering::pcfb_write_options(artifact);
    INFO(render(options.diagnostics()));
    REQUIRE(options);
    using kind = writer::query_kind;
    CHECK(options.value().target_kinds.at(pc::target_id(0)) == std::vector<kind>{kind::probability});
    CHECK(options.value().target_kinds.at(pc::target_id(1)) == std::vector<kind>{kind::monotone_hull_probability});
    CHECK(options.value().target_kinds.at(pc::target_id(2)) == std::vector<kind>{kind::probability});

    auto written = writer::write(artifact.circuit.to_handle(), options.value());
    INFO(render(written.diagnostics()));
    REQUIRE(written);
    auto circuit = ::pcfb::io::from_bytes(written.value().bytes);
    REQUIRE(circuit);
    require_valid(*circuit);
    using rows = std::vector<std::pair<std::string, ::pcfb::QueryKind>>;
    CHECK(query_summary(*circuit) ==
          rows{{"FT", ::pcfb::QueryKind::PROBABILITY}, {"FT", ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY}, {"PARITY", ::pcfb::QueryKind::PROBABILITY}});
    auto const *queries = (*circuit)->subcircuits()->Get(0)->queries();
    CHECK(queries->Get(0)->target() != queries->Get(1)->target());
    CHECK_FALSE(int_meta(*circuit, "compile.query.0.max_initiators_per_cutset"));
    CHECK(int_meta(*circuit, "compile.query.1.max_initiators_per_cutset") == 1);
    CHECK_FALSE(int_meta(*circuit, "compile.query.2.max_initiators_per_cutset"));
}

TEST_CASE("pra_to_pc write options record the literal mode and omit an absent initiator rule", "[compile][pra_to_pc][pcfb]") {
    const auto model = build_model();
    lowering::quantification_config config;
    config.initiators = lowering::initiator_values::probability;
    config.targets = {{.name = "FT", .kinds = {pc::query_kind::monotone_hull}, .hull = pra::hull_query{.literals = pra::hull_literals::fixed}},
                      {.name = "PARITY"}};
    const auto artifact = require_lower(model, config);
    auto options = lowering::pcfb_write_options(artifact);
    INFO(render(options.diagnostics()));
    REQUIRE(options);
    auto written = writer::write(artifact.circuit.to_handle(), options.value());
    INFO(render(written.diagnostics()));
    REQUIRE(written);
    auto circuit = ::pcfb::io::from_bytes(written.value().bytes);
    REQUIRE(circuit);
    require_valid(*circuit);
    using rows = std::vector<std::pair<std::string, ::pcfb::QueryKind>>;
    CHECK(query_summary(*circuit) == rows{{"FT", ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY}, {"PARITY", ::pcfb::QueryKind::PROBABILITY}});
    CHECK(string_meta(*circuit, "compile.query.0.literal_mode") == "fixed");
    CHECK_FALSE(string_meta(*circuit, "compile.query.1.literal_mode"));
    CHECK_FALSE(int_meta(*circuit, "compile.query.0.max_initiators_per_cutset"));
}

TEST_CASE("pra_to_pc write options refuse a monotone-hull target with marked literals", "[compile][pra_to_pc][pcfb]") {
    const auto model = build_literal_model();
    const std::vector<pc::query_kind> monotone{pc::query_kind::monotone_hull};
    const std::vector<pc::query_kind> both{pc::query_kind::probability, pc::query_kind::monotone_hull};

    SECTION("a split or fixed hull target whose cone has a marked literal") {
        for (const auto mode : {pra::hull_literals::split, pra::hull_literals::fixed}) {
            for (auto const &kinds : {monotone, both}) {
                lowering::quantification_config config;
                config.initiators = lowering::initiator_values::probability;
                config.initiators = lowering::initiator_values::probability;
                config.targets = {{.name = "LIT", .kinds = kinds, .hull = pra::hull_query{.literals = mode}}};
                const auto artifact = require_lower(model, config);
                REQUIRE_FALSE(artifact.projection.literal_events(artifact.projection.target_bindings()[0]).empty());
                auto options = lowering::pcfb_write_options(artifact);
                REQUIRE_FALSE(options);
                CHECK(codes(options.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.hull-literals-unsupported"});
                const auto text = render(options.diagnostics());
                CHECK(text.find("'LIT'") != std::string::npos);
                CHECK(text.find("'Y'") != std::string::npos);
            }
        }
    }

    SECTION("a plain hull target reads marked literals as ordinary complements and is written") {
        for (auto const &kinds : {monotone, both}) {
            lowering::quantification_config config;
            config.initiators = lowering::initiator_values::probability;
            config.targets = {{.name = "LIT", .kinds = kinds, .hull = pra::hull_query{.literals = pra::hull_literals::plain}}};
            const auto artifact = require_lower(model, config);
            REQUIRE_FALSE(artifact.projection.literal_events(artifact.projection.target_bindings()[0]).empty());
            auto options = lowering::pcfb_write_options(artifact);
            INFO(render(options.diagnostics()));
            REQUIRE(options);
            auto written = writer::write(artifact.circuit.to_handle(), options.value());
            INFO(render(written.diagnostics()));
            REQUIRE(written);
            auto circuit = ::pcfb::io::from_bytes(written.value().bytes);
            REQUIRE(circuit);
            require_valid(*circuit);
            CHECK((*circuit)->schema_version() == ::pcfb::kSchemaVersionV12);
            const auto rows = query_summary(*circuit);
            REQUIRE_FALSE(rows.empty());
            CHECK(rows.back() == std::pair<std::string, ::pcfb::QueryKind>{"LIT", ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY});
            CHECK(rows.size() == kinds.size());
            const auto hull = "compile.query." + std::to_string(rows.size() - 1) + ".";
            CHECK(string_meta(*circuit, hull + "target") == "LIT");
            CHECK(string_meta(*circuit, hull + "literal_mode") == "plain");
        }
    }

    SECTION("a probability target with marked literals is unaffected") {
        lowering::quantification_config config;
        config.initiators = lowering::initiator_values::probability;
        config.targets = {{.name = "LIT"}};
        const auto artifact = require_lower(model, config);
        REQUIRE_FALSE(artifact.projection.literal_events(artifact.projection.target_bindings()[0]).empty());
        auto options = lowering::pcfb_write_options(artifact);
        INFO(render(options.diagnostics()));
        REQUIRE(options);
        auto written = writer::write(artifact.circuit.to_handle(), options.value());
        REQUIRE(written);
        auto circuit = ::pcfb::io::from_bytes(written.value().bytes);
        REQUIRE(circuit);
        require_valid(*circuit);
        CHECK((*circuit)->schema_version() == ::pcfb::kSchemaVersionV11);
    }

    SECTION("a hull target without marked literals next to a probability target with them") {
        lowering::quantification_config config;
        config.initiators = lowering::initiator_values::probability;
        config.targets = {{.name = "SAFE", .kinds = monotone}, {.name = "LIT"}};
        const auto artifact = require_lower(model, config);
        auto options = lowering::pcfb_write_options(artifact);
        INFO(render(options.diagnostics()));
        REQUIRE(options);
        auto written = writer::write(artifact.circuit.to_handle(), options.value());
        REQUIRE(written);
        auto circuit = ::pcfb::io::from_bytes(written.value().bytes);
        REQUIRE(circuit);
        require_valid(*circuit);
        using rows = std::vector<std::pair<std::string, ::pcfb::QueryKind>>;
        CHECK(query_summary(*circuit) == rows{{"SAFE", ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY}, {"LIT", ::pcfb::QueryKind::PROBABILITY}});
        CHECK((*circuit)->schema_version() == ::pcfb::kSchemaVersionV12);
    }
}

TEST_CASE("pra_to_pc write options reject a missing artifact", "[compile][pra_to_pc][pcfb]") {
    auto missing = lowering::pcfb_write_options(lowering::pra_pc_lowering{});
    REQUIRE_FALSE(missing);
    CHECK(codes(missing.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.invalid-artifact"});
}
