#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include "compile/exact_eval.h"
#include "compile/pra_to_pc/descriptor.h"
#include "io/pcfb/descriptor.h"
#include "ir/pra/builder.h"

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
            config.max_initiators_per_cutset = rule;
            config.watch_targets = {{.name = "PARITY"}};
            std::vector<reference_override> overrides;
            if (override_flag) {
                config.overrides = {{.name = "DEL", .value = false}};
                overrides.push_back({.element = model.find_gate("DEL"), .value = false});
            }
            const auto artifact = require_lower(model, config);
            require_equivalent(model, artifact, overrides, rule);

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
                const auto subject = artifact.projection.target_bindings()[query].subject;
                const auto expected = reference.target_probability(subject, rule);
                const auto exact = fuzzybits::compile::exact_probability(*circuit, query);
                CAPTURE(query, expected, exact);
                CHECK(std::abs(exact - expected) <= 1e-12);
            }
        }
    }
}

TEST_CASE("pra_to_pc write options reject monotone-hull targets", "[compile][pra_to_pc][pcfb]") {
    const auto model = build_model();
    lowering::quantification_config config;
    config.default_negation = pra::negation_set::both();
    const auto artifact = require_lower(model, config);
    CHECK(artifact.projection.target_bindings()[0].negation == pra::negation_set::both());
    auto options = lowering::pcfb_write_options(artifact);
    REQUIRE_FALSE(options);
    CHECK(codes(options.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.unsupported-query-kind"});

    auto missing = lowering::pcfb_write_options(lowering::pra_pc_lowering{});
    REQUIRE_FALSE(missing);
    CHECK(codes(missing.diagnostics()) == std::vector<std::string>{"compile.pra_to_pc.invalid-artifact"});
}
