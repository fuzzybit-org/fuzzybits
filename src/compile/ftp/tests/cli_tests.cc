#include <catch2/catch_test_macros.hpp>

#include "compile/ftp/json.h"

#include "runtime/tests/test_backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef FZB_PATH
#error "FZB_PATH must be provided by CMake"
#endif

namespace {

namespace json = fuzzybits::compile::ftp::json;

struct run_result final {
    int status = -1;
    std::string out;
    std::string err;
};

[[nodiscard]] auto quoted(std::string const &text) -> std::string {
#ifdef _WIN32
    return "\"" + text + "\"";
#else
    return "'" + text + "'";
#endif
}

[[nodiscard]] auto read_file(std::filesystem::path const &path) -> std::string {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

[[nodiscard]] auto run_fzb(std::string const &arguments, std::filesystem::path const &scratch) -> run_result {
    const auto out_path = scratch / "fzb.out";
    const auto err_path = scratch / "fzb.err";
    const auto command =
        quoted(FZB_PATH) + " " + arguments + " > " + quoted(out_path.string()) + " 2> " + quoted(err_path.string());
#ifdef _WIN32
    const int status = std::system(("cmd /S /C \"" + command + "\"").c_str());
#else
    const int status = std::system(command.c_str());
#endif
    return run_result{.status = status, .out = read_file(out_path), .err = read_file(err_path)};
}

[[nodiscard]] auto scratch_directory(std::string const &name) -> std::filesystem::path {
    const auto path = std::filesystem::temp_directory_path() / ("fuzzybits-compile-ftp-cli-" + name);
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

[[nodiscard]] auto number_after(std::string const &text, std::string const &key) -> double {
    const auto position = text.rfind(key);
    INFO(text);
    REQUIRE(position != std::string::npos);
    return std::strtod(text.c_str() + position + key.size(), nullptr);
}

[[nodiscard]] auto fixtures_root() -> std::filesystem::path {
    return std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "conformance";
}

[[nodiscard]] auto parse_file(std::filesystem::path const &path) -> json::value {
    auto parsed = json::parse(read_file(path));
    INFO(path.string());
    REQUIRE(parsed);
    return std::move(parsed).value();
}

// `fzb inspect --exact` of query 0.
[[nodiscard]] auto exact_of(std::filesystem::path const &pcfb, std::filesystem::path const &scratch) -> double {
    const auto inspected = run_fzb("inspect " + quoted(pcfb.string()) + " --validate --exact", scratch);
    INFO(inspected.out << inspected.err);
    REQUIRE(inspected.status == 0);
    REQUIRE(inspected.err.find("validate: ok") != std::string::npos);
    return number_after(inspected.out, "exact_probability");
}

// The error ids a fixture expects, "io.ftp." prefixed.
[[nodiscard]] auto expected_errors(json::value const &outcome) -> std::multiset<std::string> {
    std::multiset<std::string> out;
    for (auto const &item : outcome.find("diagnostics")->items) {
        if (item.find("severity")->string == "error") {
            out.insert("io.ftp." + item.find("code")->string);
        }
    }
    return out;
}

[[nodiscard]] auto report_errors(std::filesystem::path const &report) -> std::multiset<std::string> {
    const auto document = parse_file(report);
    std::multiset<std::string> out;
    for (auto const &item : document.find("diagnostics")->items) {
        if (item.find("severity")->string == "error") {
            out.insert(item.find("code")->string);
        }
    }
    return out;
}

void require_close(const double actual, const double expected) {
    INFO("actual " << actual << " expected " << expected);
    REQUIRE(std::abs(actual - expected) <= 1e-9 * std::abs(expected) + 1e-15);
}

[[nodiscard]] auto inputs_of(json::value const &spec, std::filesystem::path const &fixture) -> std::string {
    std::string out;
    for (auto const &file : spec.find("files")->items) {
        out += " " + quoted((fixture / file.string).string());
    }
    return out;
}

} // namespace

TEST_CASE("fzb compile --negation boolean reproduces the conformance values under fzb inspect --exact",
          "[compile][ftp][cli][conformance]") {
    std::vector<std::filesystem::path> fixtures;
    for (auto const &entry : std::filesystem::directory_iterator(fixtures_root())) {
        if (entry.is_directory() && std::filesystem::is_regular_file(entry.path() / "expected.json")) {
            fixtures.push_back(entry.path());
        }
    }
    std::ranges::sort(fixtures);
    REQUIRE(fixtures.size() >= 16);
    const auto scratch = scratch_directory("conformance");

    for (auto const &fixture : fixtures) {
        DYNAMIC_SECTION(fixture.filename().string()) {
            std::size_t compared = 0;
            std::size_t rejected = 0;
            const auto spec = parse_file(fixture / "expected.json");
            const auto inputs = inputs_of(spec, fixture);
            const auto pcfb = scratch / "model.pcfb";
            const auto report = scratch / "report.json";

            for (std::size_t policy_index = 0; policy_index < spec.find("policies")->items.size(); ++policy_index) {
                const auto &policy = spec.find("policies")->keys[policy_index];
                auto const &outcome = spec.find("policies")->items[policy_index];
                INFO("policy " << policy);
                std::filesystem::remove(pcfb);
                const auto base = "compile" + inputs +
                                  " --negation boolean --initiator-values probability --max-initiators-per-cutset none "
                                  "--duplicate-import " +
                                  policy + " --no-progress -o " + quoted(pcfb.string()) + " --report " +
                                  quoted(report.string());

                if (outcome.find("results")->is(json::value::kind_type::null)) {
                    const auto failed = run_fzb(base, scratch);
                    INFO(failed.err);
                    REQUIRE(failed.status != 0);
                    REQUIRE_FALSE(std::filesystem::exists(pcfb));
                    REQUIRE(report_errors(report) == expected_errors(outcome));
                    for (auto const &code : expected_errors(outcome)) {
                        REQUIRE(failed.err.find(code) != std::string::npos);
                    }
                    ++rejected;
                    continue;
                }

                auto const &results = outcome.find("results");
                for (std::size_t index = 0; index < results->items.size(); ++index) {
                    const auto &target = results->keys[index];
                    INFO("target " << target);
                    const auto ran = run_fzb(base + " --target " + quoted(target), scratch);
                    INFO(ran.err);
                    REQUIRE(ran.status == 0);
                    require_close(exact_of(pcfb, scratch), results->items[index].find("boolean")->number);
                    ++compared;
                }
                // The default targets are the PROCESS targets, one query each, in file order.
                const auto all = run_fzb(base, scratch);
                INFO(all.err);
                REQUIRE(all.status == 0);
                REQUIRE(report_errors(report).empty());
            }

            if (auto const *variants = spec.find("variants")) {
                for (auto const &variant : variants->items) {
                    std::string flags;
                    if (auto const *overrides = variant.find("overrides")) {
                        for (std::size_t index = 0; index < overrides->keys.size(); ++index) {
                            flags += " --override " + quoted(overrides->keys[index] + "=" +
                                                             std::to_string(overrides->items[index].integer));
                        }
                    }
                    flags += " --max-initiators-per-cutset ";
                    flags += variant.find("max_initiators_per_cutset") != nullptr
                                 ? std::to_string(variant.find("max_initiators_per_cutset")->integer)
                                 : "none";
                    auto const &results = variant.find("results");
                    for (std::size_t index = 0; index < results->items.size(); ++index) {
                        const auto &target = results->keys[index];
                        INFO("variant target " << target << flags);
                        const auto ran =
                            run_fzb("compile" + inputs +
                                        " --negation boolean --initiator-values probability --duplicate-import " +
                                        variant.find("policy")->string + flags + " --no-progress --target " +
                                        quoted(target) + " -o " + quoted(pcfb.string()),
                                    scratch);
                        INFO(ran.err);
                        REQUIRE(ran.status == 0);
                        require_close(exact_of(pcfb, scratch), results->items[index].find("boolean")->number);
                        ++compared;
                    }
                }
            }
            REQUIRE(compared + rejected >= 3);
            REQUIRE((fixture.filename().string().starts_with("invalid_") ? compared == 0 : compared > 0));
        }
    }
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb compile writes probability queries for FTP input unless hull is requested",
          "[compile][ftp][cli][hull]") {
    const auto scratch = scratch_directory("hull");
    const auto out = scratch / "m.pcfb";
    const auto model = scratch / "m.ftp";
    std::ofstream(model)
        << "TOP + G1 E3\nG1 * E1 E2\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 E1\n  2.0E-01 E2\n  3.0E-01 E3\n";

    // The default: a minor-1 file with one probability query, which the exact evaluator reads.
    const auto ran =
        run_fzb("compile " + quoted(model.string()) + " --no-progress -o " + quoted(out.string()), scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    auto inspected = run_fzb("inspect " + quoted(out.string()) + " --validate --json", scratch);
    REQUIRE(inspected.status == 0);
    auto summary = json::parse(inspected.out);
    REQUIRE(summary);
    REQUIRE(summary.value().find("schema_version")->integer == 0x00010001);
    REQUIRE(summary.value().find("metadata")->find("compile.query.0.negation")->string == "boolean");
    require_close(exact_of(out, scratch), 0.314);

    // --negation hull: a minor-2 file with the hull query and its metadata.
    const auto hull = run_fzb(
        "compile " + quoted(model.string()) + " --negation hull --no-progress -o " + quoted(out.string()), scratch);
    INFO(hull.err);
    REQUIRE(hull.status == 0);
    summary = json::parse(run_fzb("inspect " + quoted(out.string()) + " --validate --json", scratch).out);
    REQUIRE(summary);
    REQUIRE(summary.value().find("schema_version")->integer == 0x00010002);
    auto const *metadata = summary.value().find("metadata");
    REQUIRE(metadata->find("compile.query.0.negation")->string == "monotone_hull");
    REQUIRE(metadata->find("compile.query.0.literal_mode")->string == "atom_split");
    // Initiating-event values are frequencies by default, so no initiator rule is lowered or recorded.
    REQUIRE(metadata->find("compile.query.0.max_initiators_per_cutset") == nullptr);

    // Both kinds on one target: two queries with the target's name.
    const auto config = scratch / "both.json";
    std::ofstream(config) << R"({"targets": [{"name": "TOP", "negation": "both"}]})";
    const auto both = run_fzb("compile " + quoted(model.string()) + " --config " + quoted(config.string()) +
                                  " --no-progress -o " + quoted(out.string()),
                              scratch);
    INFO(both.err);
    REQUIRE(both.status == 0);
    summary = json::parse(run_fzb("inspect " + quoted(out.string()) + " --json", scratch).out);
    REQUIRE(summary.value().find("schema_version")->integer == 0x00010002);
    REQUIRE(summary.value().find("queries")->integer == 2);
    REQUIRE(summary.value().find("metadata")->find("compile.query.0.negation")->string == "boolean");
    REQUIRE(summary.value().find("metadata")->find("compile.query.1.negation")->string == "monotone_hull");
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb compile refuses hull targets with marked literals and says how to proceed",
          "[compile][ftp][cli][hull]") {
    const auto scratch = scratch_directory("literals");
    const auto model = quoted((fixtures_root() / "initiator_rule" / "model.ftp").string());
    const auto out = (scratch / "m.pcfb");
    for (const char *flags : {" --negation hull --initiator-values probability",
                              " --negation hull --initiator-values probability --literals omega_l"}) {
        const auto refused = run_fzb("compile " + model + flags + " --no-progress -o " + quoted(out.string()), scratch);
        INFO(flags << "\n" << refused.err);
        REQUIRE(refused.status != 0);
        REQUIRE_FALSE(std::filesystem::exists(out));
        REQUIRE(refused.err.find("compile.ftp.hull-literals-unsupported") != std::string::npos);
        REQUIRE(refused.err.find("--literals negation") != std::string::npos);
        REQUIRE(refused.err.find("--negation boolean") != std::string::npos);
    }
    for (const char *flags : {" --initiator-unit per-year", " --negation boolean --initiator-values probability",
                              " --negation hull --initiator-values probability --literals negation",
                              " --negation hull --initiator-values probability --complement-events negation"}) {
        const auto ran = run_fzb("compile " + model + flags + " --no-progress -o " + quoted(out.string()), scratch);
        INFO(flags << "\n" << ran.err);
        REQUIRE(ran.status == 0);
        REQUIRE(std::filesystem::exists(out));
        std::filesystem::remove(out);
    }
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb run refuses a compiled hull file before sampling", "[compile][ftp][cli][hull][mc]") {
    const auto scratch = scratch_directory("run");
    const auto model = scratch / "m.ftp";
    const auto out = scratch / "m.pcfb";
    std::ofstream(model)
        << "TOP + G1 E3\nG1 * E1 E2\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 E1\n  2.0E-01 E2\n  3.0E-01 E3\n";
    REQUIRE(run_fzb("compile " + quoted(model.string()) + " --negation hull --no-progress -o " + quoted(out.string()),
                    scratch)
                .status == 0);
    const auto run =
        run_fzb("run " + quoted(out.string()) + " --trials 1024" + fuzzybits::testing::backend_flag(), scratch);
    INFO(run.err);
    REQUIRE(run.status != 0);
    REQUIRE(run.err.find("pcfb.unsupported-query-kind") != std::string::npos);
    REQUIRE(run.err.find("MONOTONE_HULL_PROBABILITY") != std::string::npos);
    REQUIRE(run.out.find("estimate") == std::string::npos);

    // The same model compiled with the default probability query runs.
    REQUIRE(
        run_fzb("compile " + quoted(model.string()) + " --no-progress -o " + quoted(out.string()), scratch).status ==
        0);
    const auto boolean = run_fzb(
        "run " + quoted(out.string()) + " --trials 65536" + fuzzybits::testing::backend_flag() + " --json", scratch);
    INFO(boolean.err);
    REQUIRE(boolean.status == 0);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb compile options, config precedence, report and metadata", "[compile][ftp][cli]") {
    const auto scratch = scratch_directory("options");
    const auto model = quoted((fixtures_root() / "initiator_rule" / "model.ftp").string());
    const auto pcfb = scratch / "m.pcfb";
    const auto report = scratch / "report.json";
    const auto config = scratch / "config.json";
    std::ofstream(config)
        << R"({"negation": "boolean", "initiator_values": "probability", "max_initiators_per_cutset": null, "targets": ["SEQ-3", "TOP"], "overrides": {"OPER": 0}})";

    // The file alone: no initiator rule, SEQ-3 first with OPER forced FALSE.
    auto ran = run_fzb("compile " + model + " --config " + quoted(config.string()) + " --no-progress -o " +
                           quoted(pcfb.string()) + " --report " + quoted(report.string()),
                       scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    require_close(exact_of(pcfb, scratch), 0.5);
    {
        const auto document = parse_file(report);
        REQUIRE(document.find("status")->string == "ok");
        REQUIRE(document.find("options")->find("max_initiators_per_cutset")->is(json::value::kind_type::null));
        REQUIRE(document.find("options")->find("targets")->items.size() == 2);
        REQUIRE(document.find("options")->find("overrides")->items[0].find("name")->string == "OPER");
        REQUIRE(document.find("lowering")->find("overridden_elements")->integer == 1);
        REQUIRE(document.find("circuit")->find("nodes_after")->integer <=
                document.find("circuit")->find("nodes_before")->integer);
        REQUIRE(document.find("passes")->items.size() >= 4);
        REQUIRE(document.find("source")->find("files")->items.size() == 1);
    }

    // Flags override the file: another target, the initiator rule on, the override flipped.
    ran = run_fzb("compile " + model + " --config " + quoted(config.string()) +
                      " --target SEQ-4 --target TOP --max-initiators-per-cutset 1 --override OPER=1 "
                      "--no-progress -c 0 -o " +
                      quoted(pcfb.string()) + " --report " + quoted(report.string()),
                  scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    require_close(exact_of(pcfb, scratch), 0.0);
    {
        const auto document = parse_file(report);
        REQUIRE(document.find("options")->find("max_initiators_per_cutset")->integer == 1);
        REQUIRE(document.find("options")->find("compilation_level")->integer == 0);
        REQUIRE(document.find("options")->find("targets")->items[0].find("name")->string == "SEQ-4");
        REQUIRE(document.find("options")->find("overrides")->items[0].find("value")->boolean);
        REQUIRE(document.find("passes")->items.empty());
    }

    // Metadata: the source format, the digest of the file set, the effective options.
    const auto inspected = run_fzb("inspect " + quoted(pcfb.string()) + " --json", scratch);
    REQUIRE(inspected.status == 0);
    const auto summary = json::parse(inspected.out);
    REQUIRE(summary);
    auto const *metadata = summary.value().find("metadata");
    REQUIRE(metadata->find("general.source_format")->string == "cafta-ftp");
    REQUIRE(metadata->find("general.source_sha256")->string.size() == 64);
    const auto recorded = json::parse(metadata->find("compile.options")->string);
    REQUIRE(recorded);
    REQUIRE(recorded.value().find("targets")->items[1].find("name")->string == "TOP");
    const auto names = summary.value().find("query_names")->items;
    REQUIRE(names.size() == 2);
    REQUIRE(names[0].string == "SEQ-4");
    REQUIRE(names[1].string == "TOP");

    // The text summary prints the counts.
    ran = run_fzb("compile " + model + " --negation boolean --initiator-values probability --no-progress -o " +
                      quoted(pcfb.string()),
                  scratch);
    REQUIRE(ran.status == 0);
    REQUIRE(ran.err.find("wrote ") != std::string::npos);
    REQUIRE(ran.err.find("circuit: nodes ") != std::string::npos);
    REQUIRE(ran.err.find("diagnostics: ") != std::string::npos);

    // The default output name follows the first file's stem, like the MEF path; a directory names the project.
    const auto previous = std::filesystem::current_path();
    std::filesystem::current_path(scratch);
    ran = run_fzb("compile " + model + " --negation boolean --initiator-values probability --no-progress --quiet",
                  scratch);
    std::filesystem::current_path(previous);
    REQUIRE(ran.status == 0);
    REQUIRE(ran.err.empty());
    REQUIRE(std::filesystem::exists(scratch / "model.pcfb"));
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb compile takes several files and a directory as one project", "[compile][ftp][cli]") {
    const auto scratch = scratch_directory("project");
    const auto directory = fixtures_root() / "multi_file_project";
    const auto pcfb = scratch / "p.pcfb";
    const auto flags =
        std::string(
            " --negation boolean --initiator-values probability --max-initiators-per-cutset none --no-progress -o ") +
        quoted(pcfb.string());

    const auto by_directory = run_fzb("compile " + quoted(directory.string()) + flags, scratch);
    INFO(by_directory.err);
    REQUIRE(by_directory.status == 0);
    require_close(exact_of(pcfb, scratch), 0.0019254481299999999);
    const auto digest = [&]() {
        const auto summary = json::parse(run_fzb("inspect " + quoted(pcfb.string()) + " --json", scratch).out);
        REQUIRE(summary);
        return summary.value().find("metadata")->find("general.source_sha256")->string;
    };
    const auto first = digest();
    const auto scrambled = run_fzb("compile " + quoted((directory / "03_imports.ftp").string()) + " " +
                                       quoted((directory / "01_top.ftp").string()) + " " +
                                       quoted((directory / "02_systems.ftp").string()) + flags,
                                   scratch);
    REQUIRE(scrambled.status == 0);
    REQUIRE(digest() == first);
    const auto summary = json::parse(run_fzb("inspect " + quoted(pcfb.string()) + " --json", scratch).out);
    REQUIRE(summary.value().find("metadata")->find("general.model_name")->string == "01_top");
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb compile rejects bad FTP options and mixed inputs", "[compile][ftp][cli]") {
    const auto scratch = scratch_directory("errors");
    const auto model = quoted((fixtures_root() / "initiator_rule" / "model.ftp").string());
    const auto out = quoted((scratch / "m.pcfb").string());
    const auto fails = [&](std::string const &arguments, std::string const &needle) {
        const auto ran = run_fzb("compile " + arguments + " --no-progress -o " + out, scratch);
        INFO(arguments << "\n" << ran.err);
        REQUIRE(ran.status != 0);
        REQUIRE(ran.err.find(needle) != std::string::npos);
    };
    fails(model + " --negation boolean -c 2", "compile.ftp.invalid-option");
    fails(model + " --negation maybe", "compile.ftp.invalid-option");
    fails(model + " --negation boolean --literals maybe", "compile.ftp.invalid-option");
    fails(model + " --negation boolean --override OPER", "NAME=0 or NAME=1");
    fails(model + " --negation boolean --override NOPE=1", "compile.pra_to_pc.unknown-element");
    fails(model + " --negation boolean --target NOPE", "compile.ftp.unknown-target");
    fails(model + " --negation boolean --max-initiators-per-cutset many", "--max-initiators-per-cutset");
    fails(model + " --negation boolean --config " + quoted((scratch / "missing.json").string()), "cannot read");
    fails(model + " --negation boolean --no-kn", "applies to OpenPSA/MEF input");

    std::ofstream(scratch / "m.xml") << "<opsa-mef/>\n";
    fails(model + " " + quoted((scratch / "m.xml").string()), "cannot be combined");
    fails(quoted((scratch / "m.xml").string()) + " --negation boolean", "applies to CAFTA FTP input only");
    std::filesystem::remove_all(scratch);
}

namespace {

// `std::system` reports the wait status on POSIX.
[[nodiscard]] auto exit_code(run_result const &ran) -> int {
#ifdef _WIN32
    return ran.status;
#else
    return ran.status >> 8;
#endif
}

constexpr const char *kDeprecatedCcf =
    "fzb compile: --ccf is deprecated: common-cause groups are expanded by default; use --no-ccf to disable";

[[nodiscard]] auto ccf_model() -> std::string {
    return quoted(
        (std::filesystem::path(PROJECT_SOURCE_DIR) / "tests" / "input" / "core" / "beta_factor_ccf.xml").string());
}

[[nodiscard]] auto count_of(std::string const &text, std::string const &needle) -> std::size_t {
    std::size_t count = 0;
    for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) {
        ++count;
    }
    return count;
}

[[nodiscard]] auto summary_of(std::filesystem::path const &pcfb, std::filesystem::path const &scratch) -> json::value {
    const auto inspected = run_fzb("inspect " + quoted(pcfb.string()) + " --json", scratch);
    REQUIRE(inspected.status == 0);
    auto parsed = json::parse(inspected.out);
    REQUIRE(parsed);
    return std::move(parsed).value();
}

[[nodiscard]] auto query_names_of(json::value const &summary) -> std::vector<std::string> {
    std::vector<std::string> out;
    for (auto const &item : summary.find("query_names")->items) {
        out.push_back(item.string);
    }
    return out;
}

} // namespace

TEST_CASE("fzb compile expands common-cause groups by default and --no-ccf disables it", "[compile][mef][cli][ccf]") {
    const auto scratch = scratch_directory("ccf");
    const auto expanded = scratch / "expanded.pcfb";
    const auto plain = scratch / "plain.pcfb";

    auto ran = run_fzb("compile " + ccf_model() + " --no-progress -o " + quoted(expanded.string()), scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    REQUIRE(ran.err.find("deprecated") == std::string::npos);
    ran = run_fzb("compile " + ccf_model() + " --no-ccf --no-progress -o " + quoted(plain.string()), scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);

    // Beta-factor groups add common-cause events, so the circuits and the probabilities differ.
    const auto with_ccf = summary_of(expanded, scratch);
    const auto without_ccf = summary_of(plain, scratch);
    REQUIRE(with_ccf.find("samples")->integer > without_ccf.find("samples")->integer);
    REQUIRE(std::abs(exact_of(expanded, scratch) - exact_of(plain, scratch)) > 1e-6);

    // The effective choice is recorded both in the flag fragment and as a boolean.
    REQUIRE(with_ccf.find("metadata")->find("compile.options")->string.find("--no-ccf") == std::string::npos);
    REQUIRE(without_ccf.find("metadata")->find("compile.options")->string.find("--no-ccf") != std::string::npos);
    REQUIRE(with_ccf.find("metadata")->find("compile.ccf")->string == "1");
    REQUIRE(without_ccf.find("metadata")->find("compile.ccf")->string == "0");
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb compile --ccf is a deprecated no-op and conflicts with --no-ccf", "[compile][mef][cli][ccf]") {
    const auto scratch = scratch_directory("ccf-deprecated");
    const auto expanded = scratch / "expanded.pcfb";
    const auto legacy = scratch / "legacy.pcfb";

    auto ran = run_fzb("compile " + ccf_model() + " --no-progress -o " + quoted(expanded.string()), scratch);
    REQUIRE(ran.status == 0);
    ran = run_fzb("compile " + ccf_model() + " --ccf --no-progress -o " + quoted(legacy.string()), scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    REQUIRE(count_of(ran.err, kDeprecatedCcf) == 1);
    require_close(exact_of(legacy, scratch), exact_of(expanded, scratch));
    REQUIRE(summary_of(legacy, scratch).find("metadata")->find("compile.ccf")->string == "1");

    ran = run_fzb("compile " + ccf_model() + " --ccf --no-ccf --no-progress -o " + quoted(legacy.string()), scratch);
    INFO(ran.err);
    REQUIRE(exit_code(ran) == 1);
    REQUIRE(ran.err.find("--ccf and --no-ccf cannot be combined") != std::string::npos);

    const auto ftp = quoted((fixtures_root() / "initiator_rule" / "model.ftp").string());
    ran = run_fzb("compile " + ftp + " --ccf --no-ccf --no-progress -o " + quoted(legacy.string()), scratch);
    REQUIRE(exit_code(ran) == 1);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb compile accepts --no-ccf and the deprecated --ccf for FTP input", "[compile][ftp][cli][ccf]") {
    const auto scratch = scratch_directory("ccf-ftp");
    const auto model = quoted((fixtures_root() / "initiator_rule" / "model.ftp").string());
    const auto flags = std::string(" --negation boolean --initiator-values probability --no-progress -o ");
    const auto base = scratch / "base.pcfb";
    const auto other = scratch / "other.pcfb";

    REQUIRE(run_fzb("compile " + model + flags + quoted(base.string()), scratch).status == 0);
    auto ran = run_fzb("compile " + model + " --no-ccf" + flags + quoted(other.string()), scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    REQUIRE(ran.err.find("deprecated") == std::string::npos);
    REQUIRE(read_file(other) == read_file(base));

    ran = run_fzb("compile " + model + " --ccf" + flags + quoted(other.string()), scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    REQUIRE(count_of(ran.err, kDeprecatedCcf) == 1);
    REQUIRE(read_file(other) == read_file(base));
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb compile reads --target and --watch names from @FILE", "[compile][ftp][cli][target]") {
    const auto scratch = scratch_directory("name-files");
    const auto model = scratch / "model.ftp";
    std::ofstream(model)
        << "TOP + G1 E3\nG1 * E1 E2\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 E1\n  2.0E-01 E2\n  3.0E-01 E3\n";
    const auto targets = scratch / "targets.txt";
    // Blank lines, surrounding blanks and CRLF line ends are dropped; there is no comment syntax and no comma split.
    std::ofstream(targets, std::ios::binary) << "  G1 \r\n\r\n\nE1\r\n";
    const auto watched = scratch / "watch.txt";
    std::ofstream(watched) << "E3\n";
    const auto pcfb = scratch / "m.pcfb";
    const auto flags =
        std::string(
            " --negation boolean --initiator-values probability --max-initiators-per-cutset none --no-progress -o ") +
        quoted(pcfb.string());

    // Plain names and files mix; file contents expand in place.
    auto ran = run_fzb("compile " + quoted(model.string()) + " --target TOP --target @" + quoted(targets.string()) +
                           " --target E2 --watch @" + quoted(watched.string()) + flags,
                       scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    REQUIRE(query_names_of(summary_of(pcfb, scratch)) == std::vector<std::string>{"TOP", "G1", "E1", "E2", "E3"});

    ran = run_fzb("compile " + quoted(model.string()) + " --target @" + quoted((scratch / "missing.txt").string()) +
                      flags,
                  scratch);
    INFO(ran.err);
    REQUIRE(exit_code(ran) == 1);
    REQUIRE(ran.err.find("cannot read " + (scratch / "missing.txt").string()) != std::string::npos);
    ran =
        run_fzb("compile " + quoted(model.string()) + " --watch @" + quoted((scratch / "missing.txt").string()) + flags,
                scratch);
    REQUIRE(exit_code(ran) == 1);

    // A file without names must not fall back to every PROCESS target; neither may a directory.
    const auto empty = scratch / "empty.txt";
    std::ofstream(empty) << "\n  \r\n";
    for (auto const &flag : {"--target", "--watch"}) {
        ran =
            run_fzb("compile " + quoted(model.string()) + " " + flag + " @" + quoted(empty.string()) + flags, scratch);
        INFO(ran.err);
        REQUIRE(exit_code(ran) == 1);
        REQUIRE(ran.err.find("fzb compile: " + empty.string() + " lists no names") != std::string::npos);
    }
    ran = run_fzb("compile " + quoted(model.string()) + " --target @" + quoted(scratch.string()) + flags, scratch);
    INFO(ran.err);
    REQUIRE(exit_code(ran) == 1);
    REQUIRE(ran.err.find(scratch.string() + " is not a regular file") != std::string::npos);

    // The FTP driver rejects a repeated target; so does an expanded one.
    ran = run_fzb("compile " + quoted(model.string()) + " --target G1 --target @" + quoted(targets.string()) + flags,
                  scratch);
    INFO(ran.err);
    REQUIRE(exit_code(ran) == 1);
    REQUIRE(ran.err.find("compile.pra_to_pc.duplicate-target") != std::string::npos);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb compile MEF input takes one target and watches from @FILE", "[compile][mef][cli][target]") {
    const auto scratch = scratch_directory("mef-targets");
    const auto pcfb = scratch / "m.pcfb";
    const auto flags = " --no-progress -o " + quoted(pcfb.string());
    const auto one = scratch / "one.txt";
    std::ofstream(one) << "TopEvent\n";
    const auto two = scratch / "two.txt";
    std::ofstream(two) << "TopEvent\nTrainOne\n";
    const auto watched = scratch / "watch.txt";
    std::ofstream(watched) << "TrainOne\r\nTrainTwo\r\n";

    auto ran = run_fzb("compile " + ccf_model() + " --target @" + quoted(one.string()) + " --watch @" +
                           quoted(watched.string()) + flags,
                       scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    const auto names = query_names_of(summary_of(pcfb, scratch));
    REQUIRE(names.size() == 3);
    REQUIRE(names[1] == "TrainOne");
    REQUIRE(names[2] == "TrainTwo");

    const auto several =
        "fzb compile: MEF and nombrilo inputs take one --target; several targets need the new MEF reader (#148)";
    for (auto const &arguments :
         std::vector<std::string>{" --target TopEvent --target TrainOne", " --target @" + quoted(two.string())}) {
        ran = run_fzb("compile " + ccf_model() + arguments + flags, scratch);
        INFO(arguments << "\n" << ran.err);
        REQUIRE(exit_code(ran) == 1);
        REQUIRE(ran.err.find(several) != std::string::npos);
    }
    ran =
        run_fzb("compile " + ccf_model() + " --target @" + quoted((scratch / "missing.txt").string()) + flags, scratch);
    REQUIRE(exit_code(ran) == 1);
    REQUIRE(ran.err.find("cannot read " + (scratch / "missing.txt").string()) != std::string::npos);

    const auto empty = scratch / "empty.txt";
    std::ofstream(empty) << "\n";
    ran = run_fzb("compile " + ccf_model() + " --watch @" + quoted(empty.string()) + flags, scratch);
    REQUIRE(exit_code(ran) == 1);
    REQUIRE(ran.err.find(empty.string() + " lists no names") != std::string::npos);
    ran = run_fzb("compile " + ccf_model() + " --target @" + quoted(scratch.string()) + flags, scratch);
    REQUIRE(exit_code(ran) == 1);
    REQUIRE(ran.err.find("is not a regular file") != std::string::npos);

    ran = run_fzb("compile " + ccf_model() + " --target TopEvent --watch TopEvent" + flags, scratch);
    REQUIRE(exit_code(ran) == 1);
    REQUIRE(ran.err.find("target 'TopEvent' is requested more than once") != std::string::npos);
    ran = run_fzb("compile " + ccf_model() + " --watch TrainOne --watch @" + quoted(watched.string()) + flags, scratch);
    REQUIRE(exit_code(ran) == 1);
    REQUIRE(ran.err.find("target 'TrainOne' is requested more than once") != std::string::npos);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb compile help shows the new target and CCF flags", "[compile][cli]") {
    const auto scratch = scratch_directory("help");
    const auto ran = run_fzb("compile --help", scratch);
    REQUIRE(ran.status == 0);
    for (auto const *needle : {"--target NAME|@FILE", "--watch NAME|@FILE", "--no-ccf"}) {
        INFO(needle);
        REQUIRE(ran.out.find(needle) != std::string::npos);
    }
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb run and fzb inspect --exact report frequency groups and exposure",
          "[compile][ftp][cli][frequency][mc]") {
    const auto scratch = scratch_directory("frequency");
    const auto root = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "frequency";
    const auto f2 = scratch / "f2.pcfb";
    const auto f4 = scratch / "f4.pcfb";
    for (auto const &[model, out] : {std::pair{root / "F2.ftp", f2}, std::pair{root / "F4.ftp", f4}}) {
        const auto ran = run_fzb("compile " + quoted(model.string()) + " --initiator-unit per-year --no-progress -o " +
                                     quoted(out.string()),
                                 scratch);
        INFO(ran.err);
        REQUIRE(ran.status == 0);
    }

    // F2: F = 12 per year; over one month 1 - exp(-1).
    auto inspected = run_fzb("inspect " + quoted(f2.string()) + " --exact --exposure 1mo", scratch);
    INFO(inspected.err);
    REQUIRE(inspected.status == 0);
    REQUIRE(inspected.out.find("exact_group  TOP  1.20000000e+01") != std::string::npos);
    REQUIRE(inspected.out.find("per year  (1 conditioned queries)") != std::string::npos);
    REQUIRE(inspected.out.find("P(>=1 in 1mo)  6.32120559e-01") != std::string::npos);
    REQUIRE(run_fzb("inspect " + quoted(f2.string()) + " --exposure 1mo", scratch).status != 0);
    REQUIRE(run_fzb("inspect " + quoted(f2.string()) + " --exact --exposure 1fortnight", scratch).status != 0);

    // F4: two conditioned queries; the group estimate is the weighted sum, near F = 0.26 per year.
    const auto ran = run_fzb("run " + quoted(f4.string()) + " --trials 1e6 --exposure 1y --json --no-progress" +
                                 fuzzybits::testing::backend_flag(),
                             scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    auto document = json::parse(ran.out);
    REQUIRE(document);
    auto const &queries = document.value().find("queries")->items;
    REQUIRE(queries.size() == 2);
    REQUIRE(queries[0].find("name")->string == "TOP|IE1");
    REQUIRE(queries[1].find("name")->string == "TOP|IE2");
    auto const &group = document.value().find("groups")->items.at(0);
    REQUIRE(group.find("name")->string == "TOP");
    REQUIRE(group.find("unit")->string == "per-year");
    const auto estimate = group.find("estimate")->number;
    REQUIRE(std::abs(estimate - (20.0 * queries[0].find("estimate")->number +
                                 3.0 * queries[1].find("estimate")->number)) <= 1e-12 * estimate);
    REQUIRE(std::abs(estimate - 0.26) <= 0.05 * 0.26);
    REQUIRE(group.find("ci_low")->number <= estimate);
    REQUIRE(group.find("ci_high")->number >= estimate);
    REQUIRE(std::abs(group.find("exposure")->find("probability")->number - -std::expm1(-estimate)) <= 1e-12);

    // --exposure on a file without groups is refused before sampling.
    const auto plain = scratch / "plain.pcfb";
    REQUIRE(run_fzb("compile " + quoted((fixtures_root() / "header_limit" / "model.ftp").string()) +
                        " --no-progress -o " + quoted(plain.string()),
                    scratch)
                .status == 0);
    const auto refused = run_fzb("run " + quoted(plain.string()) + " --trials 1024 --exposure 1y --no-progress" +
                                     fuzzybits::testing::backend_flag(),
                                 scratch);
    REQUIRE(refused.status != 0);
    REQUIRE(refused.err.find("no frequency groups") != std::string::npos);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb run answers every query of a multi-target file", "[compile][ftp][cli][mc]") {
    // Queries outside the first query's cone used to fail with "unknown event" (the PDAG root was the first query).
    const auto scratch = scratch_directory("multi");
    const auto out = scratch / "m.pcfb";
    REQUIRE(run_fzb("compile " + quoted((fixtures_root() / "process_multi_target").string()) + " --no-progress -o " +
                        quoted(out.string()),
                    scratch)
                .status == 0);
    const auto ran = run_fzb("run " + quoted(out.string()) + " --trials 1e6 --json --no-progress" +
                                 fuzzybits::testing::backend_flag(),
                             scratch);
    INFO(ran.err);
    REQUIRE(ran.status == 0);
    auto document = json::parse(ran.out);
    REQUIRE(document);
    auto const &queries = document.value().find("queries")->items;
    REQUIRE(queries.size() == 3);
    const std::array expected{0.622, 0.464, 0.098};
    for (std::size_t index = 0; index < queries.size(); ++index) {
        INFO(queries[index].find("name")->string);
        REQUIRE(std::abs(queries[index].find("estimate")->number - expected[index]) <= 0.01);
    }
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb run agrees with fzb inspect --exact on constants from overrides and folding (#259)",
          "[compile][ftp][cli][mc][constants]") {
    // `--override NAME=1` at -c 0 writes a named ONES, which fzb run used to read as FALSE. At the default level a
    // p = 1 event under an OR folds the whole target to ONES, which read as FALSE too.
    const auto scratch = scratch_directory("constants");
    const auto fixture = (fixtures_root() / "success_branch_delete_term" / "model.ftp").string();
    const auto unit = scratch / "unit.ftp";
    {
        std::ofstream out(unit);
        out << "**CHAR32\nTOP + A B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.000000E+00 A\n  5.000000E-01 B\n"
               "LIMIT 1.000000E-12\n*XEQ\n";
        REQUIRE(out);
    }
    struct run_case final {
        std::string input;
        std::string target;
        std::string options;
    };
    std::vector<run_case> cases;
    for (const char *target : {"TOP", "SEQ-2", "SEQ-3"}) {
        cases.push_back({fixture, target, "--override IE=1 -c 0"});
        cases.push_back({fixture, target, "--initiator-values probability --override P1=1 -c 0"});
    }
    cases.push_back({unit.string(), "TOP", ""});

    for (std::size_t index = 0; index < cases.size(); ++index) {
        auto const &item = cases[index];
        INFO(item.input << " --target " << item.target << " " << item.options);
        const auto out = scratch / ("case" + std::to_string(index) + ".pcfb");
        const auto compiled = run_fzb("compile " + quoted(item.input) + " --target " + item.target + " " +
                                          item.options + " --no-progress -o " + quoted(out.string()),
                                      scratch);
        INFO(compiled.err);
        REQUIRE(compiled.status == 0);
        const auto exact = exact_of(out, scratch);

        const auto ran = run_fzb("run " + quoted(out.string()) + " --trials 262144 --json --no-progress" +
                                     fuzzybits::testing::backend_flag(),
                                 scratch);
        INFO(ran.err);
        REQUIRE(ran.status == 0);
        auto document = json::parse(ran.out);
        REQUIRE(document);
        auto const &query = document.value().find("queries")->items.at(0);
        const auto n = static_cast<double>(query.find("n")->integer);
        const auto hits = static_cast<double>(query.find("hits")->integer);
        INFO("exact " << exact << ", hits " << hits << " of " << n);
        REQUIRE(n >= 262144.0);
        if (exact < 1.0e-12 || exact > 1.0 - 1.0e-12) {
            CHECK(hits == (exact > 0.5 ? n : 0.0));
        } else {
            CHECK(std::abs(hits - n * exact) <= 4.0 * std::sqrt(n * exact * (1.0 - exact)) + 1.0);
        }
    }
    std::filesystem::remove_all(scratch);
}
