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
#include <cstdint>
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

// The conformance fixtures tag probabilities with I, so the tests that check their values read
// initiating-event values as probabilities (#243).
[[nodiscard]] auto boolean_options() -> ftp::options {
    ftp::options options;
    options.negation = "boolean";
    options.initiator_values = "probability";
    return options;
}

[[nodiscard]] auto hull_options() -> ftp::options {
    ftp::options options;
    options.negation = "hull";
    options.initiator_values = "probability";
    return options;
}

[[nodiscard]] auto frequency_options() -> ftp::options {
    ftp::options options;
    options.initiator_unit = "per-year";
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

[[nodiscard]] auto int_meta(::pcfb::ir::Circuit const &circuit, char const *key) -> std::optional<std::int64_t> {
    auto const *entry = circuit->metadata() == nullptr ? nullptr : circuit->metadata()->LookupByKey(key);
    if (entry == nullptr || entry->value_type() != ::pcfb::MetadataValue::IntValue) {
        return std::nullopt;
    }
    return entry->value_as_IntValue()->value();
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
    REQUIRE(root.find("complement_events") == nullptr);
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

    // Complemented events and gates change nothing about the default: every query is a probability query, here
    // one per initiating event of TOP's cone.
    const auto marked = compile_ok(read_file(fixture("initiator_rule") / "model.ftp"), frequency_options());
    REQUIRE(query_kinds(view(marked.bytes)) == std::vector<::pcfb::QueryKind>(3, ::pcfb::QueryKind::PROBABILITY));
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
    // No initiator rule unless requested: nothing is read from the model.
    REQUIRE(circuit->metadata()->LookupByKey("compile.query.0.max_initiators_per_cutset") == nullptr);
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
    options.initiator_values = "probability";
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
    REQUIRE(compile_text(text, frequency_options()));

    // Marked operands read as plain complements: the `negation` mode, the plain hull.
    auto plain = hull_options();
    plain.literals = "negation";
    const auto negation = compile_ok(text, plain);
    const auto circuit = view(negation.bytes);
    REQUIRE(circuit.schema_version() == minor_2);
    REQUIRE(query_kinds(circuit) == std::vector<::pcfb::QueryKind>{::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY});
    REQUIRE(string_meta(circuit, "compile.query.0.literal_mode") == "negation");

    // Boolean targets are unaffected by marks.
    REQUIRE(compile_text(text, boolean_options()));
}

TEST_CASE("the initiator rule is a rule of hull queries, requested explicitly", "[compile][ftp][initiator]") {
    const auto text = read_file(fixture("initiator_rule") / "model.ftp");

    // Probability queries are the probability of the target: no rule from the model, none by default.
    const auto plain = compile_ok(text, boolean_options());
    REQUIRE(std::abs(exact(plain) - 0.45000000000000007) <= 1e-12);
    REQUIRE_FALSE(plain.targets[0].max_initiators_per_cutset);
    REQUIRE(plain.targets[0].initiating_events == 3);
    REQUIRE_FALSE(has_code(compile_text(text, boolean_options()).diagnostics(), "compile.ftp.default-initiator-rule"));
    REQUIRE(view(plain.bytes)->metadata()->LookupByKey("compile.query.0.max_initiators_per_cutset") == nullptr);

    // A rule without a hull query would change the probability query: refused.
    auto boolean_rule = boolean_options();
    boolean_rule.max_initiators_per_cutset = std::optional<std::uint32_t>{1};
    const auto refused = compile_text(text, boolean_rule);
    REQUIRE_FALSE(refused);
    REQUIRE(has_code(refused.diagnostics(), "compile.ftp.invalid-option"));
    REQUIRE(render(refused.diagnostics()).find("hull queries only") != std::string::npos);
    auto document_rule = boolean_options();
    document_rule.max_initiators_per_cutset.reset();
    document_rule.config = R"({"max_initiators_per_cutset": 1})";
    REQUIRE(has_code(compile_text(text, document_rule).diagnostics(), "compile.ftp.invalid-option"));
    // `none` is no rule and needs no hull query.
    document_rule.config = R"({"max_initiators_per_cutset": "none"})";
    REQUIRE(std::abs(exact(compile_ok(text, document_rule)) - 0.45000000000000007) <= 1e-12);

    // A hull query takes the rule; its metadata says so.
    auto hull_rule = hull_options();
    hull_rule.literals = "negation";
    hull_rule.max_initiators_per_cutset = std::optional<std::uint32_t>{1};
    const auto hull = compile_ok(text, hull_rule);
    REQUIRE(hull.targets.size() == 1);
    REQUIRE(hull.targets[0].max_initiators_per_cutset == 1U);
    REQUIRE(int_meta(view(hull.bytes), "compile.query.0.max_initiators_per_cutset") == 1);

    // Both kinds: the probability query reads TOP itself, the hull query TOP under the rule, on two
    // tensors. SEQ-3 holds one initiating event, so its two queries share one tensor and neither has a rule.
    auto both = hull_rule;
    both.negation = "both";
    both.targets = {"TOP", "SEQ-3"};
    const auto both_kinds = compile_ok(text, both);
    REQUIRE(both_kinds.targets.size() == 3);
    REQUIRE(both_kinds.targets[0].boolean);
    REQUIRE_FALSE(both_kinds.targets[0].monotone_hull);
    REQUIRE_FALSE(both_kinds.targets[0].max_initiators_per_cutset);
    REQUIRE(both_kinds.targets[1].monotone_hull);
    REQUIRE_FALSE(both_kinds.targets[1].boolean);
    REQUIRE(both_kinds.targets[1].max_initiators_per_cutset == 1U);
    REQUIRE(both_kinds.targets[2].boolean);
    REQUIRE(both_kinds.targets[2].monotone_hull);
    REQUIRE_FALSE(both_kinds.targets[2].max_initiators_per_cutset);
    const auto circuit = view(both_kinds.bytes);
    REQUIRE(query_names(circuit) == std::vector<std::string>{"TOP", "TOP", "SEQ-3", "SEQ-3"});
    auto const *queries = circuit->subcircuits()->Get(0)->queries();
    REQUIRE(queries->Get(0)->target() != queries->Get(1)->target());
    REQUIRE(queries->Get(2)->target() == queries->Get(3)->target());
    REQUIRE(circuit->metadata()->LookupByKey("compile.query.0.max_initiators_per_cutset") == nullptr);
    REQUIRE(int_meta(circuit, "compile.query.1.max_initiators_per_cutset") == 1);
    REQUIRE(circuit->metadata()->LookupByKey("compile.query.0.literal_mode") == nullptr);
    REQUIRE(string_meta(circuit, "compile.query.1.literal_mode") == "negation");

    // The lowered probability query is TOP's subject; the hull query is the conjunction with the delete term.
    const auto lowered = ftp::lower_sources(sources_of({{"m.ftp", text}}), both);
    REQUIRE(lowered);
    auto const bindings = lowered.value().projection.target_bindings();
    REQUIRE(bindings.size() == 3);
    REQUIRE(bindings[0].node == bindings[0].root);
    REQUIRE_FALSE(bindings[0].initiator_delete_term.valid());
    REQUIRE(bindings[1].root == bindings[0].root);
    REQUIRE(bindings[1].initiator_delete_term.valid());
}

TEST_CASE("flags override the config document, which overrides the model defaults", "[compile][ftp][config]") {
    const auto text = read_file(fixture("initiator_rule") / "model.ftp");

    ftp::options file_only;
    file_only.config = R"({
        "negation": "boolean",
        "initiator_values": "probability",
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
    flags.negation = "hull";
    flags.max_initiators_per_cutset = std::optional<std::uint32_t>{1};
    flags.level = 1;
    flags.overrides = {ftp::override_spec{.name = "OPER", .value = false}};
    const auto from_flags = compile_ok(text, flags);
    REQUIRE(from_flags.level == 1);
    // The document's watch list stays: only a non-empty flag list replaces it.
    REQUIRE(query_names(view(from_flags.bytes)) == std::vector<std::string>{"SEQ-4", "SEQ-1"});
    // The flags' rule, not the document's `null`: SEQ-4 = IE-FLAG and IE-2 and PUMP-A holds two initiators,
    // so its hull query gets the delete term.
    REQUIRE(from_flags.targets[0].monotone_hull);
    REQUIRE(from_flags.targets[0].max_initiators_per_cutset == 1U);

    flags.negation = "boolean";
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
          R"({"compilation_level": 3})", R"({"max_initiators_per_cutset": -1})", R"({"literals": ["x"]})",
          // Removed: -EVENT is always a marked literal.
          R"({"complement_events": "negation"})"}) {
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

TEST_CASE("duplicate-import passes through to the reader; -EVENT is always a marked literal",
          "[compile][ftp][reader]") {
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
    const auto lowered = ftp::lower_sources(sources_of({{"m.ftp", complement}}), boolean_options());
    REQUIRE(lowered);
    REQUIRE_FALSE(lowered.value().projection.literal_events().empty());
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

namespace {

[[nodiscard]] auto as_double(json::value const *value) -> double {
    REQUIRE(value != nullptr);
    return value->kind == json::value::kind_type::integer ? static_cast<double>(value->integer) : value->number;
}

[[nodiscard]] auto frequency_fixture(std::string_view name) -> std::filesystem::path {
    return std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "frequency" / name;
}

// F(T) of a compiled file: the weighted sum of the exact values of its first group's queries.
[[nodiscard]] auto group_value(ftp::compiled const &compiled) -> double {
    const auto circuit = view(compiled.bytes);
    auto const *groups = circuit->subcircuits()->Get(0)->query_groups();
    REQUIRE(groups != nullptr);
    double sum = 0.0;
    for (auto const *term : *groups->Get(0)->terms()) {
        sum += term->weight() * fuzzybits::compile::exact_probability(circuit, term->query());
    }
    return sum;
}

} // namespace

TEST_CASE("initiating-event values are frequencies: one conditioned query per initiator, grouped (#243)",
          "[compile][ftp][frequency]") {
    auto const parsed = json::parse(read_file(frequency_fixture("expected.json")));
    REQUIRE(parsed);
    auto const &models = *parsed.value().find("models");
    for (std::size_t index = 0; index < models.keys.size(); ++index) {
        auto const &name = models.keys[index];
        auto const &model = models.items[index];
        INFO(name);
        const auto text = read_file(frequency_fixture(name + ".ftp"));
        const auto result = compile_text(text, frequency_options());
        if (auto const *error = model.find("error")) {
            REQUIRE_FALSE(result);
            REQUIRE(has_code(result.diagnostics(), error->string));
            continue;
        }
        INFO(render(result.diagnostics()));
        REQUIRE(result);
        auto const &compiled = result.value();
        auto const &terms = *model.find("terms");
        const auto circuit = view(compiled.bytes);
        REQUIRE(circuit.schema_version() == minor_2);
        auto const *queries = circuit->subcircuits()->Get(0)->queries();
        REQUIRE(queries->size() == terms.keys.size());
        auto const *groups = circuit->subcircuits()->Get(0)->query_groups();
        REQUIRE(groups != nullptr);
        REQUIRE(groups->size() == 1);
        auto const *group = groups->Get(0);
        REQUIRE(group->name()->str() == "TOP");
        REQUIRE(group->unit()->str() == "per-year");
        REQUIRE(group->terms()->size() == terms.keys.size());
        for (std::uint32_t query = 0; query < queries->size(); ++query) {
            auto const *term = group->terms()->Get(query);
            REQUIRE(term->query() == query);
            const auto prefix = "compile.query." + std::to_string(query) + ".";
            const auto initiator = string_meta(circuit, (prefix + "initiator").c_str());
            REQUIRE(initiator);
            REQUIRE(queries->Get(query)->name()->str() == "TOP|" + *initiator);
            REQUIRE(queries->Get(query)->kind() == ::pcfb::QueryKind::PROBABILITY);
            REQUIRE(term->weight() == as_double(terms.find(*initiator)));
            REQUIRE(compiled.targets[query].initiator == *initiator);
        }
        REQUIRE(std::abs(group_value(compiled) - as_double(model.find("frequency"))) <= 1e-12);
        REQUIRE(compiled.lowering.conditioned_targets == terms.keys.size());
        REQUIRE_FALSE(has_code(result.diagnostics(), "compile.ftp.default-initiator-rule"));
    }
}

TEST_CASE("frequencies default to per year; probabilities keep today's reading", "[compile][ftp][frequency]") {
    const auto f1 = read_file(frequency_fixture("F1.ftp"));
    const auto f6 = read_file(frequency_fixture("F6.ftp"));

    // No unit: per year, in the file's group and in the report.
    const auto unitless = compile_ok(f1, {});
    REQUIRE(std::abs(group_value(unitless) - 0.596) <= 1e-12);
    REQUIRE(view(unitless.bytes)->subcircuits()->Get(0)->query_groups()->Get(0)->unit()->str() == "per-year");
    REQUIRE(unitless.lowering.initiator_unit == "per-year");

    // A model without initiating events needs no unit.
    REQUIRE(compile_text(simple, {}));

    // Probabilities: a value above 1 is rejected, values up to 1 are read as probabilities (F6 without the rule
    // is 0.0556, the value #243 reports for that reading).
    auto probability = boolean_options();
    REQUIRE(has_code(compile_text(f1, probability).diagnostics(), "compile.pra_to_pc.frequency-out-of-range"));
    probability.max_initiators_per_cutset = std::optional<std::uint32_t>{};
    REQUIRE(std::abs(exact(compile_ok(f6, probability)) - 0.0556) <= 1e-12);

    // The initiator rule and the unit each belong to one reading; `none` (no rule) agrees with frequencies.
    auto rule = frequency_options();
    rule.max_initiators_per_cutset = std::optional<std::uint32_t>{1};
    REQUIRE(has_code(compile_text(f1, rule).diagnostics(), "compile.ftp.invalid-option"));
    rule.max_initiators_per_cutset = std::optional<std::uint32_t>{};
    REQUIRE(std::abs(group_value(compile_ok(f1, rule)) - 0.596) <= 1e-12);
    auto unit = boolean_options();
    unit.initiator_unit = "per-year";
    REQUIRE(has_code(compile_text(f6, unit).diagnostics(), "compile.ftp.invalid-option"));
    for (auto const &[values, units] :
         {std::pair<const char *, const char *>{"frequencies", "per-year"}, {"frequency", "per-fortnight"}}) {
        ftp::options bad;
        bad.initiator_values = values;
        bad.initiator_unit = units;
        REQUIRE(has_code(compile_text(f1, bad).diagnostics(), "compile.ftp.invalid-option"));
    }

    // The config document sets both; a flag overrides the document.
    ftp::options document;
    document.config = R"({"initiator_values": "frequency", "initiator_unit": "per-day"})";
    const auto by_document = compile_ok(f6, document);
    REQUIRE(std::abs(group_value(by_document) - 0.066) <= 1e-12);
    REQUIRE(view(by_document.bytes)->subcircuits()->Get(0)->query_groups()->Get(0)->unit()->str() == "per-day");
    // The flag wins: probabilities, so the document's rule is allowed (F6 without the rule is 0.0556).
    document.initiator_values = "probability";
    document.config = R"({"initiator_values": "frequency", "max_initiators_per_cutset": null})";
    const auto by_flag = compile_ok(f6, document);
    REQUIRE(std::abs(exact(by_flag) - 0.0556) <= 1e-12);
    REQUIRE(view(by_flag.bytes)->subcircuits()->Get(0)->query_groups() == nullptr);
    // Frequencies from the document with a rule (a count) from the document is an error; `null` is no rule.
    document.initiator_values.reset();
    const auto no_rule = compile_ok(f6, document);
    REQUIRE(std::abs(group_value(no_rule) - 0.066) <= 1e-12);
    REQUIRE(view(no_rule.bytes)->subcircuits()->Get(0)->query_groups()->Get(0)->unit()->str() == "per-year");
    document.config =
        R"({"initiator_values": "frequency", "initiator_unit": "per-year", "max_initiators_per_cutset": 1})";
    REQUIRE(has_code(compile_text(f6, document).diagnostics(), "compile.ftp.invalid-option"));
    document.config = R"({"initiator_unit": "per-fortnight"})";
    document.initiator_values.reset();
    REQUIRE(has_code(compile_text(f6, document).diagnostics(), "compile.ftp.invalid-config"));
}

TEST_CASE("a frequency target asking for both kinds gets one group per kind", "[compile][ftp][frequency][hull]") {
    auto options = frequency_options();
    options.negation = "both";
    const auto compiled = compile_ok(read_file(frequency_fixture("F4.ftp")), options);
    const auto circuit = view(compiled.bytes);
    REQUIRE(query_kinds(circuit) == std::vector<::pcfb::QueryKind>{
                                        ::pcfb::QueryKind::PROBABILITY, ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY,
                                        ::pcfb::QueryKind::PROBABILITY, ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY});
    auto const *groups = circuit->subcircuits()->Get(0)->query_groups();
    REQUIRE(groups->size() == 2);
    REQUIRE(groups->Get(0)->terms()->Get(0)->query() == 0);
    REQUIRE(groups->Get(0)->terms()->Get(1)->query() == 2);
    REQUIRE(groups->Get(1)->terms()->Get(0)->query() == 1);
    REQUIRE(groups->Get(1)->terms()->Get(1)->query() == 3);
}
