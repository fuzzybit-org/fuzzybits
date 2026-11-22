#include <catch2/catch_test_macros.hpp>

#include "compile/ftp/inspect.h"
#include "compile/ftp/json.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

    namespace ftp = fuzzybits::compile::ftp;
    namespace json = fuzzybits::compile::ftp::json;
    namespace reader = fuzzybits::io::backend::fuzzybits::ftp;

    [[nodiscard]] auto read_file(std::filesystem::path const &path) -> std::string {
        std::ifstream input(path, std::ios::binary);
        REQUIRE(input);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    [[nodiscard]] auto parse_text(std::string_view text) -> json::value {
        auto parsed = json::parse(text);
        REQUIRE(parsed);
        return std::move(parsed).value();
    }

    [[nodiscard]] auto fixture_directories() -> std::vector<std::filesystem::path> {
        std::vector<std::filesystem::path> out;
        const auto root = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "conformance";
        for (auto const &entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_directory() && std::filesystem::is_regular_file(entry.path() / "expected.json")) {
                out.push_back(entry.path());
            }
        }
        std::ranges::sort(out);
        return out;
    }

    [[nodiscard]] auto as_number(json::value const &value) -> double {
        return value.is(json::value::kind_type::integer) ? static_cast<double>(value.integer) : value.number;
    }

    [[nodiscard]] auto is_numeric(json::value const &value) -> bool {
        return value.is(json::value::kind_type::integer) || value.is(json::value::kind_type::number);
    }

    // Every key of `expected` is present in `actual` with the same value. The
    // duplicate-import entries only agree on the names, and an empty
    // `process_targets` stands for the implicit root targets.
    void require_same(json::value const &actual, json::value const &expected, std::string const &path) {
        INFO("at " << path);
        if (is_numeric(expected)) {
            REQUIRE(is_numeric(actual));
            REQUIRE(std::abs(as_number(actual) - as_number(expected)) <= 1e-12 * std::abs(as_number(expected)));
        } else if (expected.is_string()) {
            REQUIRE(actual.is_string());
            REQUIRE(actual.string == expected.string);
        } else if (expected.is(json::value::kind_type::null)) {
            REQUIRE(actual.is(json::value::kind_type::null));
        } else if (expected.is_array()) {
            REQUIRE(actual.is_array());
            REQUIRE(actual.items.size() == expected.items.size());
            for (std::size_t index = 0; index < expected.items.size(); ++index) {
                require_same(actual.items[index], expected.items[index], path + "[" + std::to_string(index) + "]");
            }
        } else if (expected.is_object()) {
            REQUIRE(actual.is_object());
            REQUIRE(actual.keys.size() >= expected.keys.size());
            for (std::size_t index = 0; index < expected.keys.size(); ++index) {
                auto const *found = actual.find(expected.keys[index]);
                INFO("key " << expected.keys[index]);
                REQUIRE(found != nullptr);
                if (path == "structure" && expected.keys[index] == "duplicate_imports") {
                    REQUIRE(found->is_object());
                    REQUIRE(found->keys == expected.items[index].keys);
                } else if (path == "structure" && expected.keys[index] == "process_targets" && expected.items[index].items.empty()) {
                    continue;
                } else {
                    require_same(*found, expected.items[index], path.empty() ? expected.keys[index] : path + "." + expected.keys[index]);
                }
            }
        }
    }

    [[nodiscard]] auto diagnostic_ids(json::value const &list) -> std::multiset<std::string> {
        std::multiset<std::string> out;
        for (auto const &item : list.items) {
            const auto severity = item.find("severity")->string;
            if (severity != "note") {
                out.insert(severity + " " + item.find("id")->string);
            }
        }
        return out;
    }

    [[nodiscard]] auto expected_ids(json::value const &outcome, const bool documents_structure_without_targets) -> std::multiset<std::string> {
        std::multiset<std::string> out;
        bool missing_process = false;
        for (auto const &item : outcome.find("diagnostics")->items) {
            const auto id = "io.ftp." + item.find("code")->string;
            missing_process = missing_process || id == "io.ftp.missing-process";
            out.insert(item.find("severity")->string + " " + id);
        }
        if (documents_structure_without_targets && !missing_process) {
            out.insert("warning io.ftp.missing-process");
        }
        return out;
    }

} // namespace

TEST_CASE("inspect: JSON structure and diagnostic ids of every conformance fixture under every duplicate-import policy",
          "[compile][ftp][inspect][conformance]") {
    const auto fixtures = fixture_directories();
    REQUIRE(fixtures.size() >= 16);

    for (auto const &fixture : fixtures) {
        DYNAMIC_SECTION(fixture.filename().string()) {
            const auto spec = parse_text(read_file(fixture / "expected.json"));
            std::vector<std::filesystem::path> files;
            for (auto const &file : spec.find("files")->items) {
                files.push_back(fixture / file.string);
            }
            auto const *structure = spec.find("structure");
            const bool without_targets = structure != nullptr && structure->find("process_targets")->items.empty();

            for (std::size_t policy = 0; policy < spec.find("policies")->keys.size(); ++policy) {
                const auto &name = spec.find("policies")->keys[policy];
                auto const &outcome = spec.find("policies")->items[policy];
                INFO("policy " << name);

                ftp::inspect_options options;
                options.duplicate_import = name;
                const auto result = ftp::inspect_project(files, options);
                const auto document = parse_text(json::dump(ftp::inspection_json(result), 2));

                REQUIRE(diagnostic_ids(*document.find("diagnostics")) == expected_ids(outcome, without_targets));
                REQUIRE(document.find("files")->items.size() == files.size());

                const bool read = !outcome.find("results")->is(json::value::kind_type::null);
                REQUIRE(document.find("status")->string == (read ? "ok" : "error"));
                if (!read) {
                    REQUIRE(document.find("structure")->is(json::value::kind_type::null));
                    continue;
                }
                REQUIRE(structure != nullptr);
                require_same(*document.find("structure"), *structure, "structure");
            }
        }
    }
}

TEST_CASE("inspect: diagnostics carry the file and line named by the reader", "[compile][ftp][inspect]") {
    const auto fixture = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "conformance" / "invalid_undefined_event";
    const auto spec = parse_text(read_file(fixture / "expected.json"));
    const std::vector<std::filesystem::path> files{fixture / "model.ftp"};
    const auto result = ftp::inspect_project(files);

    REQUIRE_FALSE(result.ok());
    REQUIRE(result.diagnostics.size() == 1);
    auto const &item = result.diagnostics.front();
    auto const &expected = spec.find("policies")->find("last")->find("diagnostics")->items.front();
    REQUIRE(item.id == "io.ftp.undefined-reference");
    REQUIRE(std::filesystem::path(item.file).filename() == "model.ftp");
    REQUIRE(item.line == expected.find("locations")->items.front().items[1].integer);
}

TEST_CASE("inspect: duplicate imports name the chosen record and its value", "[compile][ftp][inspect]") {
    const std::vector<reader::source_text> sources{
        {.path = "d.ftp", .text = "TOP + A B\nENDTREE\nPROCESS TOP\nIMPORT\n  3.0E-01 A\n  1.0E-01 B\n  2.0E-01 A\n"}};

    ftp::inspect_options options;
    for (const auto &[policy, line, value] : {std::tuple{"last", 7U, 0.2}, std::tuple{"first", 5U, 0.3}}) {
        options.duplicate_import = policy;
        const auto result = ftp::inspect_sources(sources, options);
        REQUIRE(result.ok());
        REQUIRE(result.warnings() == 1);
        REQUIRE(result.structure->duplicate_imports.size() == 1);
        auto const &duplicate = result.structure->duplicate_imports.front();
        REQUIRE(duplicate.name == "A");
        REQUIRE(duplicate.value == value);
        REQUIRE(duplicate.occurrences.size() == 2);
        REQUIRE(duplicate.occurrences[0].line == 5);
        REQUIRE(duplicate.occurrences[1].line == 7);
        for (auto const &occurrence : duplicate.occurrences) {
            REQUIRE(occurrence.chosen == (occurrence.line == line));
        }
    }

    options.duplicate_import = "error";
    const auto rejected = ftp::inspect_sources(sources, options);
    REQUIRE_FALSE(rejected.ok());
    REQUIRE_FALSE(rejected.structure);
    REQUIRE(rejected.diagnostics.front().file == "d.ftp");
    REQUIRE(rejected.diagnostics.front().line == 5);
}

TEST_CASE("inspect: structure counts literal operands, complemented gates, house events and conventions", "[compile][ftp][inspect]") {
    const std::vector<reader::source_text> sources{{.path = "s.ftp",
                                                    .text = "**CHAR32\n**DBNAME: sample\nTOP * G -B -H\nG + A B\nH + A B\nENDTREE\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\n"
                                                            "LIMIT 1.0E-09\n"}};
    ftp::inspect_options options;
    const auto atom = ftp::inspect_sources(sources, options);
    REQUIRE(atom.ok());
    REQUIRE(atom.structure->complemented_edges == 2);
    REQUIRE(atom.structure->complemented_edges_to_basic_events == 1);
    REQUIRE(atom.structure->complemented_edges_to_gates == 1);
    REQUIRE(atom.structure->literal_operands == 1);
    REQUIRE(atom.structure->house_events == 0);
    REQUIRE(atom.structure->limit == 1e-9);
    REQUIRE(atom.conventions.limit == "1.0E-09");
    REQUIRE(atom.conventions.negation == "delete_term");
    REQUIRE(atom.conventions.max_initiators_per_cutset == "1");
    REQUIRE(atom.conventions.header_lines == std::vector<std::string>{"**CHAR32", "**DBNAME: sample"});

    options.complement_events = "negation";
    const auto negation = ftp::inspect_sources(sources, options);
    REQUIRE(negation.structure->literal_operands == 0);
    REQUIRE(negation.structure->complemented_edges == 2);
}

TEST_CASE("inspect: --counts-only output holds no element, file-content or header text", "[compile][ftp][inspect]") {
    const std::vector<reader::source_text> sources{
        {.path = "c.ftp",
         .text = "**DBNAME: WOMBAT_HEADER\nZEBRA_TOP * QUOKKA_G -QUOKKA_B\nQUOKKA_G + QUOKKA_B QUOKKA_C\nENDTREE\nPROCESS ZEBRA_TOP\nIMPORT\n"
                 "  1.0E-01 QUOKKA_B\n  2.0E-01 QUOKKA_C\n  3.0E-01 QUOKKA_B\n  1.0E-01 NO_GATE_USES_ME\n"}};
    const auto result = ftp::inspect_sources(sources);
    REQUIRE(result.ok());
    REQUIRE(result.warnings() >= 1);

    const std::string json_out = json::dump(ftp::inspection_json(result, true), 2);
    const std::string text_out = ftp::inspection_text(result, true);
    for (auto const *name : {"ZEBRA", "QUOKKA", "WOMBAT", "NO_GATE"}) {
        INFO(name);
        REQUIRE(json_out.find(name) == std::string::npos);
        REQUIRE(text_out.find(name) == std::string::npos);
    }
    const auto document = parse_text(json_out);
    REQUIRE(document.find("counts_only")->boolean);
    auto const &structure = *document.find("structure");
    REQUIRE(structure.find("roots")->integer == 1);
    REQUIRE(structure.find("process_targets")->integer == 1);
    REQUIRE(structure.find("duplicate_imports")->integer == 1);
    REQUIRE(document.find("conventions")->find("header_lines")->integer == 1);
    for (auto const &item : document.find("diagnostics")->items) {
        REQUIRE(item.find("message") == nullptr);
    }

    const auto full = parse_text(json::dump(ftp::inspection_json(result, false), 2));
    REQUIRE(full.find("structure")->find("roots")->is_array());
    REQUIRE(full.find("diagnostics")->items.front().find("message") != nullptr);
}

TEST_CASE("inspect: exit status and option checks", "[compile][ftp][inspect]") {
    const std::vector<reader::source_text> clean{{.path = "a.ftp", .text = "TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n"}};
    const std::vector<reader::source_text> warns{{.path = "w.ftp", .text = "TOP + A\nENDTREE\nIMPORT\n  1.0E-01 A\n"}};
    const std::vector<reader::source_text> broken{{.path = "b.ftp", .text = "TOP + A B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n"}};

    REQUIRE(ftp::inspect_exit_status(ftp::inspect_sources(clean), false) == 0);
    REQUIRE(ftp::inspect_exit_status(ftp::inspect_sources(clean), true) == 0);
    REQUIRE(ftp::inspect_exit_status(ftp::inspect_sources(warns), false) == 0);
    REQUIRE(ftp::inspect_exit_status(ftp::inspect_sources(warns), true) == 1);
    REQUIRE(ftp::inspect_exit_status(ftp::inspect_sources(broken), false) == 1);

    REQUIRE(ftp::check_options({}).empty());
    REQUIRE_FALSE(ftp::check_options({.duplicate_import = "newest"}).empty());
    REQUIRE_FALSE(ftp::check_options({.complement_events = "both"}).empty());
}
