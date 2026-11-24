#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "compile/exact_eval.h"
#include "compile/ftp/compile.h"
#include "compile/ftp/json.h"
#include "compile/sha256.h"
#include "ir/pc/circuit.h"
#include "ir/pc/op_code.h"
#include "ir/pra/projection.h"
#include "pcfb.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace ftp = fuzzybits::compile::ftp;
namespace json = fuzzybits::compile::ftp::json;
namespace reader = fuzzybits::io::backend::fuzzybits::ftp;
using fuzzybits::api::diagnostic_list;

constexpr auto simple =
    "TOP + G1 E3\nG1 * E1 E2\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 E1\n  2.0E-01 E2\n  3.0E-01 E3\n";

[[nodiscard]] auto render(diagnostic_list const &diagnostics) -> std::string {
    std::ostringstream out;
    for (auto const &item : diagnostics.items()) {
        out << item.code << ": " << item.message << '\n';
    }
    return out.str();
}

[[nodiscard]] auto has_code(diagnostic_list const &diagnostics, std::string_view code) -> bool {
    return std::ranges::any_of(diagnostics.items(), [code](auto const &item) { return item.code == code; });
}

[[nodiscard]] auto read_file(std::filesystem::path const &path) -> std::string {
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

[[nodiscard]] auto fixture(std::string_view name) -> std::filesystem::path {
    return std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "conformance" / name;
}

[[nodiscard]] auto sources_of(std::vector<std::pair<std::string, std::string>> files)
    -> std::vector<reader::source_text> {
    std::vector<reader::source_text> out;
    for (auto &[path, text] : files) {
        out.push_back(reader::source_text{.path = std::move(path), .text = std::move(text)});
    }
    return out;
}

[[nodiscard]] auto boolean_options() -> ftp::options {
    ftp::options options;
    options.negation = "boolean";
    return options;
}

[[nodiscard]] auto hull_options() -> ftp::options {
    ftp::options options;
    options.negation = "hull";
    return options;
}

[[nodiscard]] auto compile_text(std::string text, ftp::options const &options) -> ftp::compile_result {
    const auto sources = sources_of({{"m.ftp", std::move(text)}});
    return ftp::compile_sources(sources, options);
}

[[nodiscard]] auto compile_ok(std::string text, ftp::options const &options) -> ftp::compiled {
    auto result = compile_text(std::move(text), options);
    INFO(render(result.diagnostics()));
    REQUIRE(result);
    return std::move(result).value();
}

[[nodiscard]] auto view(std::vector<std::byte> const &bytes) -> ::pcfb::ir::Circuit {
    auto circuit = ::pcfb::io::from_bytes(bytes);
    INFO((circuit ? std::string{} : circuit.error().message));
    REQUIRE(circuit);
    const auto diagnostics = ::pcfb::validate(*circuit);
    REQUIRE(diagnostics.empty());
    return *circuit;
}

[[nodiscard]] auto string_meta(::pcfb::ir::Circuit const &circuit, char const *key) -> std::optional<std::string> {
    auto const *entry = circuit->metadata() == nullptr ? nullptr : circuit->metadata()->LookupByKey(key);
    if (entry == nullptr || entry->value_type() != ::pcfb::MetadataValue::StringValue) {
        return std::nullopt;
    }
    return entry->value_as_StringValue()->value()->str();
}

[[nodiscard]] auto query_names(::pcfb::ir::Circuit const &circuit) -> std::vector<std::string> {
    std::vector<std::string> out;
    for (auto const *query : *circuit->subcircuits()->Get(0)->queries()) {
        out.push_back(query->name() == nullptr ? std::string{} : query->name()->str());
    }
    return out;
}

[[nodiscard]] auto sample_names(::pcfb::ir::Circuit const &circuit) -> std::set<std::string> {
    std::set<std::string> out;
    for (auto const *op : *circuit->subcircuits()->Get(0)->operators()) {
        if (op->opcode() == ::pcfb::OpCode::SAMPLE && op->args_as_SampleArgs() != nullptr &&
            op->args_as_SampleArgs()->name() != nullptr) {
            out.insert(op->args_as_SampleArgs()->name()->str());
        }
    }
    return out;
}

[[nodiscard]] auto exact(ftp::compiled const &compiled, const std::uint32_t query = 0) -> double {
    return fuzzybits::compile::exact_probability(view(compiled.bytes), query);
}

} // namespace

TEST_CASE("an FTP project compiles to Boolean queries named by their targets", "[compile][ftp]") {
    const auto compiled = compile_ok(simple, boolean_options());
    const auto circuit = view(compiled.bytes);
    REQUIRE(query_names(circuit) == std::vector<std::string>{"TOP"});
    REQUIRE(std::abs(exact(compiled) - 0.314) <= 1e-12);
    REQUIRE(sample_names(circuit) == std::set<std::string>{"E1", "E2", "E3"});
    REQUIRE(compiled.targets.size() == 1);
    REQUIRE(compiled.targets[0].name == "TOP");
    REQUIRE(compiled.targets[0].subject == "domain_target");
    REQUIRE(compiled.targets[0].boolean);
    REQUIRE_FALSE(compiled.targets[0].monotone_hull);
    REQUIRE(compiled.lowering.pc_targets == 1);
    REQUIRE(compiled.nodes_after <= compiled.nodes_before);
    REQUIRE(compiled.level == 1);
}

TEST_CASE("the metadata names the source, its digest and the effective options", "[compile][ftp][metadata]") {
    const auto directory = fixture("multi_file_project");
    const auto sources = sources_of({{"03_imports.ftp", read_file(directory / "03_imports.ftp")},
                                     {"01_top.ftp", read_file(directory / "01_top.ftp")},
                                     {"02_systems.ftp", read_file(directory / "02_systems.ftp")}});
    auto options = boolean_options();
    options.max_initiators_per_cutset = std::optional<std::uint32_t>{};
    const auto result = ftp::compile_sources(sources, options);
    INFO(render(result.diagnostics()));
    REQUIRE(result);
    auto const &compiled = result.value();
    REQUIRE(compiled.files == std::vector<std::string>{"01_top.ftp", "02_systems.ftp", "03_imports.ftp"});
    REQUIRE(std::abs(exact(compiled) - 0.0019254481299999999) <= 1e-12);

    const auto circuit = view(compiled.bytes);
    REQUIRE(string_meta(circuit, "general.source_format") == "cafta-ftp");
    auto const *sha = circuit->metadata()->LookupByKey("general.source_sha256");
    REQUIRE(sha != nullptr);
    std::string joined = sources[1].text + sources[2].text + sources[0].text;
    const auto expected = fuzzybits::compile::sha256(std::as_bytes(std::span{joined.data(), joined.size()}));
    REQUIRE(compiled.source_sha256 == fuzzybits::compile::sha256_hex(expected));
    auto const *digest = sha->value_as_BytesValue()->value();
    REQUIRE(digest->size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        REQUIRE(digest->Get(static_cast<flatbuffers::uoffset_t>(index)) == expected[index]);
    }
    REQUIRE(compiled.limit == "1.000000E-12");

    const auto recorded = json::parse(*string_meta(circuit, "compile.options"));
    REQUIRE(recorded);
    auto const &root = recorded.value();
    REQUIRE(root.find("format")->string == "cafta-ftp");
    REQUIRE(root.find("compilation_level")->integer == 1);
    REQUIRE(root.find("max_initiators_per_cutset")->kind == json::value::kind_type::null);
    REQUIRE(root.find("ftp_limit")->string == "1.000000E-12");
    auto const &targets = root.find("targets")->items;
    REQUIRE(targets.size() == 1);
    REQUIRE(targets[0].find("name")->string == "TOP");
    REQUIRE(targets[0].find("negation")->string == "boolean");
    REQUIRE(targets[0].find("literals")->string == "atom_split");
    REQUIRE(string_meta(circuit, "compile.options") == compiled.options_json);

    auto reordered = sources;
    std::ranges::reverse(reordered);
    const auto again = ftp::compile_sources(reordered, options);
    REQUIRE(again);
    REQUIRE(again.value().source_sha256 == compiled.source_sha256);
    REQUIRE(again.value().bytes == compiled.bytes);
}

[[nodiscard]] auto query_kinds(::pcfb::ir::Circuit const &circuit) -> std::vector<::pcfb::QueryKind> {
    std::vector<::pcfb::QueryKind> out;
    for (auto const *query : *circuit->subcircuits()->Get(0)->queries()) {
        out.push_back(query->kind());
    }
    return out;
}

constexpr auto minor_1 = 0x00010001U;
constexpr auto minor_2 = 0x00010002U;

TEST_CASE("FTP input defaults to probability queries", "[compile][ftp]") {
    // The query type is requested, never read from the model: the default is the probability.
    const auto compiled = compile_ok(simple, {});
    const auto circuit = view(compiled.bytes);
    REQUIRE(circuit.schema_version() == minor_1);
    REQUIRE(query_names(circuit) == std::vector<std::string>{"TOP"});
    REQUIRE(query_kinds(circuit) == std::vector<::pcfb::QueryKind>{::pcfb::QueryKind::PROBABILITY});
    REQUIRE(string_meta(circuit, "compile.query.0.negation") == "boolean");
    REQUIRE(compiled.targets[0].boolean);
    REQUIRE_FALSE(compiled.targets[0].monotone_hull);
    const auto result = compile_text(simple, {});
    REQUIRE_FALSE(has_code(result.diagnostics(), "compile.ftp.default-negation"));
    REQUIRE_FALSE(has_code(result.diagnostics(), "pc.pipeline.pass-skipped"));
    REQUIRE(std::abs(exact(compiled) - 0.314) <= 1e-12);
    const auto recorded = json::parse(*string_meta(circuit, "compile.options"));
    REQUIRE(recorded);
    REQUIRE(recorded.value().find("targets")->items[0].find("negation")->string == "boolean");

    // Complemented events and gates change nothing about the default: every target is a probability query.
    const auto marked = compile_ok(read_file(fixture("initiator_rule") / "model.ftp"), {});
    REQUIRE(query_kinds(view(marked.bytes)) == std::vector<::pcfb::QueryKind>{::pcfb::QueryKind::PROBABILITY});
}

TEST_CASE("a hull query is written only when requested, with its metadata", "[compile][ftp][hull]") {
    // `simple` has no complemented event, so no marked literals: the requested hull is written as asked,
    // although its value equals the probability.
    const auto compiled = compile_ok(simple, hull_options());
    const auto circuit = view(compiled.bytes);
    REQUIRE(circuit.schema_version() == minor_2);
    REQUIRE(query_kinds(circuit) == std::vector<::pcfb::QueryKind>{::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY});
    REQUIRE(string_meta(circuit, "compile.query.0.negation") == "monotone_hull");
    REQUIRE(string_meta(circuit, "compile.query.0.literal_mode") == "atom_split");
    auto const *rule = circuit->metadata()->LookupByKey("compile.query.0.max_initiators_per_cutset");
    REQUIRE(rule != nullptr);
    REQUIRE(rule->value_as_IntValue()->value() == 1);
    REQUIRE(compiled.targets[0].monotone_hull);
    REQUIRE_FALSE(compiled.targets[0].boolean);
    REQUIRE(has_code(compile_text(simple, hull_options()).diagnostics(), "pc.pipeline.pass-skipped"));
    // The runtime refuses hull queries until #183, the exact evaluator included.
    REQUIRE_THROWS_AS(exact(compiled), fuzzybits::compile::ExactEvalError);

    // The config document requests it too; a flag overrides the document.
    ftp::options document;
    document.config = R"({"negation": "hull"})";
    REQUIRE(query_kinds(view(compile_ok(simple, document).bytes)) ==
            std::vector<::pcfb::QueryKind>{::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY});
    document.negation = "boolean";
    REQUIRE(query_kinds(view(compile_ok(simple, document).bytes)) ==
            std::vector<::pcfb::QueryKind>{::pcfb::QueryKind::PROBABILITY});
}

TEST_CASE("--negation boolean writes minor-1 Boolean files", "[compile][ftp][hull]") {
    const auto compiled = compile_ok(simple, boolean_options());
    const auto circuit = view(compiled.bytes);
    REQUIRE(circuit.schema_version() == minor_1);
    REQUIRE(query_kinds(circuit) == std::vector<::pcfb::QueryKind>{::pcfb::QueryKind::PROBABILITY});
    REQUIRE(string_meta(circuit, "compile.query.0.negation") == "boolean");
}

TEST_CASE("a target can ask for both kinds; names and kinds survive C1", "[compile][ftp][hull]") {
    const auto text = "TOP + W1 W2\nW1 * U1\nU1 + A B\nW2 * C ONE\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  "
                      "2.0E-01 B\n  3.0E-01 C\n  1.0E+00 ONE\n";
    ftp::options options;
    options.max_initiators_per_cutset = std::optional<std::uint32_t>{};
    options.config =
        R"({"targets": [{"name": "TOP", "negation": "both"}, {"name": "W1", "negation": "hull"}, {"name": "U1", "negation": "boolean"}]})";
    for (const int level : {0, 1}) {
        options.level = level;
        const auto compiled = compile_ok(text, options);
        const auto circuit = view(compiled.bytes);
        INFO("level " << level);
        REQUIRE(circuit.schema_version() == minor_2);
        REQUIRE(query_names(circuit) == std::vector<std::string>{"TOP", "TOP", "W1", "U1"});
        REQUIRE(query_kinds(circuit) == std::vector<::pcfb::QueryKind>{::pcfb::QueryKind::PROBABILITY,
                                                                       ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY,
                                                                       ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY,
                                                                       ::pcfb::QueryKind::PROBABILITY});
        const std::vector<std::string> negations{"boolean", "monotone_hull", "monotone_hull", "boolean"};
        for (std::size_t index = 0; index < negations.size(); ++index) {
            REQUIRE(string_meta(circuit, ("compile.query." + std::to_string(index) + ".negation").c_str()) ==
                    negations[index]);
        }
        // The Boolean file of the same targets and level, for the function the queries share.
        auto boolean = options;
        boolean.negation = "boolean";
        boolean.config.reset();
        boolean.targets = {"TOP", "W1"};
        const auto reference = compile_ok(text, boolean);
        REQUIRE(std::abs(exact(reference, 0) - 0.496) <= 1e-12);
        REQUIRE(std::abs(exact(reference, 1) - 0.28) <= 1e-12);
        REQUIRE(compiled.targets[0].boolean);
        REQUIRE(compiled.targets[0].monotone_hull);
        if (level == 1) {
            // A hull target keeps p = 1 events: fold_unit_probability is skipped for the whole circuit.
            REQUIRE(sample_names(circuit) == std::set<std::string>{"A", "B", "C", "ONE"});
        }
    }
}

TEST_CASE("hull targets with marked literals are refused with the ways out", "[compile][ftp][hull]") {
    const auto text = read_file(fixture("initiator_rule") / "model.ftp");
    for (const char *mode : {"atom_split", "omega_l"}) {
        auto options = hull_options();
        options.literals = mode;
        const auto refused = compile_text(text, options);
        INFO(mode << "\n" << render(refused.diagnostics()));
        REQUIRE_FALSE(refused);
        REQUIRE(has_code(refused.diagnostics(), "compile.pra_to_pc.hull-literals-unsupported"));
        REQUIRE(has_code(refused.diagnostics(), "compile.ftp.hull-literals-unsupported"));
        const auto hint = std::ranges::find_if(refused.diagnostics().items(), [](auto const &item) {
            return item.code == "compile.ftp.hull-literals-unsupported";
        });
        REQUIRE(hint->message.find("--literals negation") != std::string::npos);
        REQUIRE(hint->message.find("--negation boolean") != std::string::npos);
    }
    // The default literal mode of a requested hull is atom_split.
    REQUIRE_FALSE(compile_text(text, hull_options()));
    // The default query is the probability, which marks do not affect.
    REQUIRE(compile_text(text, {}));

    // Marked operands read as plain complements: `negation` mode, or no marks.
    auto plain = hull_options();
    plain.literals = "negation";
    const auto negation = compile_ok(text, plain);
    const auto circuit = view(negation.bytes);
    REQUIRE(circuit.schema_version() == minor_2);
    REQUIRE(query_kinds(circuit) == std::vector<::pcfb::QueryKind>{::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY});
    REQUIRE(string_meta(circuit, "compile.query.0.literal_mode") == "negation");
    auto unmarked = hull_options();
    unmarked.complement_events = "negation";
    REQUIRE(compile_text(text, unmarked));

    // Boolean targets are unaffected by marks.
    REQUIRE(compile_text(text, boolean_options()));
}

TEST_CASE("the initiator rule defaults to the model attribute and flags override it", "[compile][ftp][initiator]") {
    const auto text = read_file(fixture("initiator_rule") / "model.ftp");

    auto none = boolean_options();
    none.max_initiators_per_cutset = std::optional<std::uint32_t>{};
    const auto without = compile_ok(text, none);
    REQUIRE(std::abs(exact(without) - 0.45000000000000007) <= 1e-12);

    const auto by_attribute = compile_ok(text, boolean_options());
    REQUIRE(std::abs(exact(by_attribute) - 0.28800000000000003) <= 1e-12);
    REQUIRE(by_attribute.targets[0].max_initiators_per_cutset == 1U);
    REQUIRE(by_attribute.targets[0].initiating_events == 3);
    const auto result = compile_text(text, boolean_options());
    REQUIRE(has_code(result.diagnostics(), "compile.ftp.default-initiator-rule"));

    // SEQ-1 holds two initiators: allowed by 2, deleted by 1.
    auto two = boolean_options();
    two.targets = {"SEQ-1"};
    two.max_initiators_per_cutset = std::optional<std::uint32_t>{2};
    REQUIRE(std::abs(exact(compile_ok(text, two)) - 0.006000000000000001) <= 1e-12);
    two.max_initiators_per_cutset = std::optional<std::uint32_t>{1};
    REQUIRE(exact(compile_ok(text, two)) == 0.0);

    // The rule is lifted by the reader option too: no attribute, no default.
    auto flag = boolean_options();
    flag.config = R"({"max_initiators_per_cutset": "none"})";
    REQUIRE(std::abs(exact(compile_ok(text, flag)) - 0.45000000000000007) <= 1e-12);
}

TEST_CASE("flags override the config document, which overrides the model defaults", "[compile][ftp][config]") {
    const auto text = read_file(fixture("initiator_rule") / "model.ftp");

    ftp::options file_only;
    file_only.config = R"({
        "negation": "boolean",
        "max_initiators_per_cutset": null,
        "compilation_level": 0,
        "targets": [{"name": "SEQ-3"}, "TOP"],
        "watch": ["SEQ-1"],
        "overrides": {"OPER": 1}
    })";
    const auto from_file = compile_ok(text, file_only);
    REQUIRE(from_file.level == 0);
    REQUIRE(from_file.nodes_before == from_file.nodes_after);
    const auto circuit = view(from_file.bytes);
    REQUIRE(query_names(circuit) == std::vector<std::string>{"SEQ-3", "TOP", "SEQ-1"});
    // SEQ-3 = IE-FLAG and VALVE and not OPER; OPER forced TRUE makes it FALSE.
    REQUIRE(exact(from_file, 0) == 0.0);
    REQUIRE(from_file.targets[2].watch);

    auto flags = file_only;
    flags.targets = {"SEQ-4"};
    flags.watch = {};
    flags.max_initiators_per_cutset = std::optional<std::uint32_t>{1};
    flags.level = 1;
    flags.overrides = {ftp::override_spec{.name = "OPER", .value = false}};
    const auto from_flags = compile_ok(text, flags);
    REQUIRE(from_flags.level == 1);
    // The document's watch list stays: only a non-empty flag list replaces it.
    REQUIRE(query_names(view(from_flags.bytes)) == std::vector<std::string>{"SEQ-4", "SEQ-1"});
    // SEQ-4 = IE-FLAG and IE-2 and PUMP-A holds two initiators: the rule deletes it.
    REQUIRE(exact(from_flags) == 0.0);

    flags.max_initiators_per_cutset = std::optional<std::uint32_t>{};
    flags.targets = {"SEQ-3"};
    REQUIRE(std::abs(exact(compile_ok(text, flags)) - 0.5) <= 1e-12);

    auto per_target = boolean_options();
    per_target.config = R"({"targets": [{"name": "TOP", "negation": ["boolean", "hull"], "literals": "omega_l"}]})";
    per_target.negation.reset();
    const auto lowered = ftp::lower_sources(sources_of({{"m.ftp", text}}), per_target);
    REQUIRE(lowered);
    auto const &binding = lowered.value().projection.target_bindings().front();
    REQUIRE(binding.negation.boolean);
    REQUIRE(binding.negation.monotone_hull);
    REQUIRE(binding.literals == fuzzybits::ir::pra::literal_mode::omega_l);

    auto overruled = per_target;
    overruled.negation = "boolean";
    overruled.literals = "negation";
    const auto flagged = ftp::lower_sources(sources_of({{"m.ftp", text}}), overruled);
    REQUIRE(flagged);
    auto const &flagged_binding = flagged.value().projection.target_bindings().front();
    REQUIRE_FALSE(flagged_binding.negation.monotone_hull);
    REQUIRE(flagged_binding.literals == fuzzybits::ir::pra::literal_mode::negation);
}

TEST_CASE("invalid settings are rejected with stable ids", "[compile][ftp][config]") {
    const auto fails = [](ftp::options const &options, std::string_view code) {
        const auto result = compile_text(simple, options);
        INFO(code << "\n" << render(result.diagnostics()));
        REQUIRE_FALSE(result);
        REQUIRE(has_code(result.diagnostics(), code));
    };
    auto options = boolean_options();
    options.level = 2;
    fails(options, "compile.ftp.invalid-option");
    options = boolean_options();
    options.negation = "both-ways";
    fails(options, "compile.ftp.invalid-option");
    options = boolean_options();
    options.literals = "maybe";
    fails(options, "compile.ftp.invalid-option");
    options = boolean_options();
    options.duplicate_import = "newest";
    fails(options, "compile.ftp.invalid-option");
    options = boolean_options();
    options.complement_events = "both";
    fails(options, "compile.ftp.invalid-option");
    options = boolean_options();
    options.targets = {"NOPE"};
    fails(options, "compile.ftp.unknown-target");
    options = boolean_options();
    options.overrides = {ftp::override_spec{.name = "NOPE", .value = true}};
    fails(options, "compile.pra_to_pc.unknown-element");
    for (const char *config : {"{", "", R"({"negation": "boolean", "negation": "hull"})",
                               R"({"negation": "boolean"} trailing)", "[1e999]"}) {
        INFO(config);
        options = boolean_options();
        options.config = config;
        fails(options, "compile.ftp.invalid-json");
    }
    for (const char *config :
         {"[]", R"({"typo": 1})", R"({"negation": "maybe"})", R"({"targets": [1]})", R"({"overrides": {"E1": 2}})",
          R"({"compilation_level": 3})", R"({"max_initiators_per_cutset": -1})", R"({"literals": ["x"]})"}) {
        INFO(config);
        options = boolean_options();
        options.config = config;
        fails(options, "compile.ftp.invalid-config");
    }
}

TEST_CASE("overrides, watch targets and element targets are lowered", "[compile][ftp]") {
    auto options = boolean_options();
    options.targets = {"G1", "E3"};
    options.watch = {"TOP"};
    options.overrides = {ftp::override_spec{.name = "E2", .value = true}};
    const auto compiled = compile_ok(simple, options);
    REQUIRE(query_names(view(compiled.bytes)) == std::vector<std::string>{"G1", "E3", "TOP"});
    REQUIRE(std::abs(exact(compiled, 0) - 0.1) <= 1e-12);
    REQUIRE(std::abs(exact(compiled, 1) - 0.3) <= 1e-12);
    REQUIRE(std::abs(exact(compiled, 2) - 0.37) <= 1e-12);
    REQUIRE(compiled.lowering.overridden_elements == 1);
    REQUIRE(compiled.targets[0].subject == "gate");
    REQUIRE(compiled.targets[1].subject == "basic_event");
    REQUIRE(compiled.targets[2].watch);
    // E2 is a constant now, not a sampled event.
    REQUIRE(sample_names(view(compiled.bytes)) == std::set<std::string>{"E1", "E3"});
}

TEST_CASE("target and event names survive C1", "[compile][ftp][names]") {
    // Single-input gates and constant folding remove nodes; names stay on the nodes that compute them.
    const auto text = "TOP + W1 W2\nW1 * U1\nU1 + A B\nW2 * C ONE\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  "
                      "2.0E-01 B\n  3.0E-01 C\n  1.0E+00 ONE\n";
    auto c0 = boolean_options();
    c0.level = 0;
    c0.targets = {"TOP", "W1", "U1", "W2"};
    auto c1 = c0;
    c1.level = 1;
    const auto before = compile_ok(text, c0);
    const auto after = compile_ok(text, c1);
    REQUIRE(after.nodes_after < before.nodes_after);
    const auto circuit = view(after.bytes);
    REQUIRE(query_names(circuit) == std::vector<std::string>{"TOP", "W1", "U1", "W2"});
    REQUIRE(sample_names(circuit) == std::set<std::string>{"A", "B", "C"});
    for (std::uint32_t query = 0; query < 4; ++query) {
        REQUIRE(std::abs(exact(after, query) - exact(before, query)) <= 1e-12);
    }
    REQUIRE(std::abs(exact(after, 0) - (1.0 - (1.0 - 0.28) * (1.0 - 0.3))) <= 1e-12);
    REQUIRE(after.passes.size() >= 4);
    REQUIRE(after.passes.front().transform == "ir.pc.fold_zero_probability");
}

TEST_CASE("duplicate-import and complement-events pass through to the reader", "[compile][ftp][reader]") {
    const auto text = read_file(fixture("imports_tags_duplicates") / "model.ftp");
    auto options = boolean_options();
    options.duplicate_import = "error";
    const auto rejected = compile_text(text, options);
    REQUIRE_FALSE(rejected);
    REQUIRE(has_code(rejected.diagnostics(), "io.ftp.duplicate-import"));

    options.max_initiators_per_cutset = std::optional<std::uint32_t>{};
    options.duplicate_import = "first";
    const auto first = compile_text(text, options);
    REQUIRE(first);
    REQUIRE(has_code(first.diagnostics(), "io.ftp.duplicate-import"));
    REQUIRE(std::abs(exact(first.value()) - 0.11080000000000001) <= 1e-12);
    options.duplicate_import.reset();
    const auto last = compile_text(text, options);
    REQUIRE(last);
    REQUIRE(std::abs(exact(last.value()) - 0.10720000000000002) <= 1e-12);

    const auto complement = read_file(fixture("complement_under_negation") / "model.ftp");
    for (const char *policy : {"atom", "negation"}) {
        auto marked = boolean_options();
        marked.complement_events = policy;
        marked.max_initiators_per_cutset = std::optional<std::uint32_t>{};
        const auto lowered = ftp::lower_sources(sources_of({{"m.ftp", complement}}), marked);
        REQUIRE(lowered);
        const auto events = lowered.value().projection.literal_events().size();
        REQUIRE((events != 0) == (std::string(policy) == "atom"));
    }
}

TEST_CASE("a failed compile reports every diagnostic and a report document", "[compile][ftp][report]") {
    const auto text = read_file(fixture("invalid_undefined_event") / "model.ftp");
    const auto result = compile_text(text, boolean_options());
    REQUIRE_FALSE(result);
    const auto failed = json::parse(ftp::report_json(result));
    REQUIRE(failed);
    REQUIRE(failed.value().find("status")->string == "error");
    REQUIRE(failed.value().find("circuit") == nullptr);
    REQUIRE_FALSE(failed.value().find("diagnostics")->items.empty());
    REQUIRE(failed.value().find("diagnostics")->items[0].find("severity")->string == "error");

    const auto ok = compile_text(simple, boolean_options());
    REQUIRE(ok);
    const auto report = json::parse(ftp::report_json(ok));
    REQUIRE(report);
    auto const &root = report.value();
    REQUIRE(root.find("status")->string == "ok");
    REQUIRE(root.find("source")->find("format")->string == "cafta-ftp");
    REQUIRE(root.find("source")->find("sha256")->string == ok.value().source_sha256);
    REQUIRE(root.find("lowering")->find("pc_targets")->integer == 1);
    REQUIRE(root.find("lowering")->find("transform")->string == "compile.pra_to_pc");
    REQUIRE(root.find("passes")->items.size() == ok.value().passes.size());
    REQUIRE(root.find("passes")->items[0].find("nodes_before")->integer >= 0);
    REQUIRE(root.find("circuit")->find("nodes_after")->integer == ok.value().nodes_after);
    REQUIRE(root.find("options")->find("targets")->items.size() == 1);
    REQUIRE(root.find("seconds")->find("write")->kind == json::value::kind_type::number);
}

TEST_CASE("inputs are classified by extension, directory and content", "[compile][ftp][detect]") {
    const auto scratch = std::filesystem::temp_directory_path() / "fuzzybits-compile-ftp-detect";
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch / "project");
    std::filesystem::create_directories(scratch / "empty");
    const auto write = [](std::filesystem::path const &path, std::string_view text) {
        std::ofstream(path, std::ios::binary) << text;
    };
    write(scratch / "by_extension.FTP", "anything");
    write(scratch / "project" / "a.ftp", simple);
    write(scratch / "by_content.txt", simple);
    write(scratch / "model.xml", "<?xml version=\"1.0\"?>\n<opsa-mef>\nENDTREE\n</opsa-mef>\n");
    write(scratch / "binary.bin", std::string("\0ENDTREE\n", 9));
    write(scratch / "plain.txt", "nothing here\n");

    using kind = ftp::input_kind;
    const auto classify = [](std::vector<std::filesystem::path> inputs) { return ftp::classify_inputs(inputs); };
    REQUIRE(classify({scratch / "by_extension.FTP"}) == kind::ftp);
    REQUIRE(classify({scratch / "project"}) == kind::ftp);
    REQUIRE(classify({scratch / "by_content.txt"}) == kind::ftp);
    REQUIRE(classify({scratch / "model.xml"}) == kind::other);
    REQUIRE(classify({scratch / "binary.bin"}) == kind::other);
    REQUIRE(classify({scratch / "plain.txt"}) == kind::other);
    REQUIRE(classify({scratch / "empty"}) == kind::other);
    REQUIRE(classify({scratch / "missing.xml"}) == kind::other);
    REQUIRE(classify({scratch / "project", scratch / "by_content.txt"}) == kind::ftp);
    REQUIRE(classify({scratch / "project", scratch / "model.xml"}) == kind::mixed);

    std::vector<std::filesystem::path> directory{scratch / "project"};
    const auto compiled = ftp::compile_project(directory, boolean_options());
    INFO(render(compiled.diagnostics()));
    REQUIRE(compiled);
    REQUIRE(compiled.value().model_name == "project");
    std::filesystem::remove_all(scratch);
}

TEST_CASE("FTP gates keep their shape through the lowering", "[compile][ftp][lowering]") {
    namespace pc = fuzzybits::ir::pc;
    const auto lower = [](std::string const &text) {
        auto lowered = ftp::lower_sources(sources_of({{"m.ftp", text}}), boolean_options());
        INFO(render(lowered.diagnostics()));
        REQUIRE(lowered);
        return std::move(lowered).value();
    };

    const auto simple_lowering = lower(
        "TOP + A MID\nMID * B C\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\n  3.0E-01 C\nLIMIT 1.0E-05\n");
    auto const &circuit = simple_lowering.circuit;
    REQUIRE(circuit.node_count() == 5);
    REQUIRE(circuit.edge_count() == 4);
    REQUIRE(circuit.targets().size() == 1);
    REQUIRE(circuit.probability_sources().size() == 3);
    REQUIRE(circuit.op_codes()[circuit.targets()[0].node.raw()] == pc::op_code::or_gate);

    const auto atleast = lower("TOP 2 A B C\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\n  3.0E-01 C\n");
    const auto top = atleast.circuit.targets()[0].node;
    REQUIRE(atleast.circuit.op_codes()[top.raw()] == pc::op_code::cardinality);
    REQUIRE(atleast.circuit.operands(top).size() == 3);
    REQUIRE(atleast.circuit.cardinality_low()[top.raw()] == 2);
    REQUIRE(atleast.circuit.cardinality_high()[top.raw()] == 3);

    const auto complemented = lower("TOP * A -B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\n");
    const auto operands = complemented.circuit.operands(complemented.circuit.targets()[0].node);
    REQUIRE(operands.size() == 2);
    REQUIRE_FALSE(operands[0].complement);
    REQUIRE(operands[1].complement);
}

TEST_CASE("override flags parse as NAME=0 or NAME=1", "[compile][ftp]") {
    const auto set = ftp::parse_override("PUMP-A=1");
    REQUIRE(set);
    REQUIRE(set.value().name == "PUMP-A");
    REQUIRE(set.value().value);
    REQUIRE_FALSE(ftp::parse_override("A=B=0").value().value);
    REQUIRE(ftp::parse_override("A=B=0").value().name == "A=B");
    for (const char *bad : {"", "A", "A=", "=1", "A=2", "A=true"}) {
        INFO(bad);
        REQUIRE_FALSE(ftp::parse_override(bad));
    }
}
