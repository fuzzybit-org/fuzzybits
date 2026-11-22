#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "io/ftp/descriptor.h"
#include "ir/pc/circuit.h"
#include "ir/pra/model.h"

#include "json.h"
#include "structure.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

    namespace ftp = fuzzybits::io::backend::fuzzybits::ftp;
    namespace pra = fuzzybits::ir::pra;
    using fuzzybits::api::diagnostic_severity;

    struct expected_diagnostic final {
        diagnostic_severity severity = diagnostic_severity::error;
        std::string code;
        std::string name;
        std::vector<std::pair<std::string, std::size_t>> locations;
    };

    [[nodiscard]] auto readText(std::filesystem::path const &path) -> std::string {
        std::ifstream input(path, std::ios::binary);
        REQUIRE(input);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    [[nodiscard]] auto diagnosticsText(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
        std::ostringstream output;
        for (auto const &diagnostic : diagnostics.items()) {
            output << diagnostic.code << ": " << diagnostic.message << '\n';
        }
        return output.str();
    }

    // "<file>:<line>" not followed by another digit, after a path separator or at the start.
    [[nodiscard]] auto mentionsLocation(std::string_view message, std::string_view file, const std::size_t line) -> bool {
        const auto needle = std::string(file) + ":" + std::to_string(line);
        for (auto at = message.find(needle); at != std::string_view::npos; at = message.find(needle, at + 1)) {
            const auto end = at + needle.size();
            const bool boundaryBefore = at == 0 || message[at - 1] == '/' || message[at - 1] == ' ' || message[at - 1] == '(';
            const bool boundaryAfter = end == message.size() || message[end] < '0' || message[end] > '9';
            if (boundaryBefore && boundaryAfter) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] auto matches(fuzzybits::api::diagnostic const &actual, expected_diagnostic const &expected) -> bool {
        if (actual.severity != expected.severity || actual.code != expected.code) {
            return false;
        }
        if (!expected.name.empty() && actual.message.find("'" + expected.name + "'") == std::string::npos) {
            return false;
        }
        return std::ranges::all_of(expected.locations, [&](auto const &location) {
            return mentionsLocation(actual.message, location.first, location.second);
        });
    }

    void requireDiagnostics(fuzzybits::api::diagnostic_list const &actual, std::vector<expected_diagnostic> const &expected) {
        INFO(diagnosticsText(actual));
        std::vector<bool> used(actual.size(), false);
        for (auto const &want : expected) {
            INFO("expected " << want.code << " '" << want.name << "'");
            bool found = false;
            for (std::size_t index = 0; index < actual.size() && !found; ++index) {
                if (!used[index] && matches(actual.items()[index], want)) {
                    used[index] = true;
                    found = true;
                }
            }
            REQUIRE(found);
        }
        const auto reported = std::ranges::count_if(actual.items(), [](auto const &item) {
            return item.severity != diagnostic_severity::note;
        });
        REQUIRE(static_cast<std::size_t>(reported) == expected.size());
    }

    [[nodiscard]] auto expectedDiagnostics(ftp_test::json const &list) -> std::vector<expected_diagnostic> {
        std::vector<expected_diagnostic> out;
        for (auto const &item : list.array()) {
            expected_diagnostic diagnostic;
            diagnostic.severity = item["severity"].string() == "error" ? diagnostic_severity::error : diagnostic_severity::warning;
            diagnostic.code = "io.ftp." + item["code"].string();
            if (!item["name"].is_null()) {
                diagnostic.name = item["name"].string();
            }
            for (auto const &location : item["locations"].array()) {
                diagnostic.locations.emplace_back(location.array()[0].string(), static_cast<std::size_t>(location.array()[1].number()));
            }
            out.push_back(std::move(diagnostic));
        }
        return out;
    }

    // Until #191 lands, header_limit has neither ENDTREE nor an IMPORT header;
    // both vendors reject or misread it, and so does the reader.
    [[nodiscard]] auto legacyHeaderLimitDiagnostics() -> std::vector<expected_diagnostic> {
        const auto error = [](std::string code, std::string name, std::size_t line) {
            expected_diagnostic out{.severity = diagnostic_severity::error, .code = "io.ftp." + std::move(code), .name = std::move(name)};
            out.locations.emplace_back("model.ftp", line);
            return out;
        };
        return {
            error("missing-endtree", "", 3),
            error("import-without-header", "A", 6),
            error("import-without-header", "B", 7),
            error("import-without-header", "C", 8),
            error("undefined-reference", "A", 4),
            error("undefined-reference", "B", 4),
            error("undefined-reference", "C", 5),
            expected_diagnostic{.severity = diagnostic_severity::warning, .code = "io.ftp.missing-process"},
        };
    }

    [[nodiscard]] auto names(ftp_test::json const &list) -> std::vector<std::string> {
        std::vector<std::string> out;
        for (auto const &item : list.array()) {
            out.push_back(item.string());
        }
        return out;
    }

    void requireStructure(ftp_test::structure const &actual, ftp_test::json const &expected) {
        REQUIRE(actual.gates == static_cast<std::size_t>(expected["gates"].number()));
        REQUIRE(actual.or_gates == static_cast<std::size_t>(expected["gates_by_kind"]["or"].number()));
        REQUIRE(actual.and_gates == static_cast<std::size_t>(expected["gates_by_kind"]["and"].number()));
        REQUIRE(actual.atleast_gates == static_cast<std::size_t>(expected["gates_by_kind"]["atleast"].number()));
        REQUIRE(actual.atleast_k_of_n.size() == expected["atleast_k_of_n"].object().size());
        for (auto const &[key, count] : expected["atleast_k_of_n"].object()) {
            REQUIRE(actual.atleast_k_of_n.at(key) == static_cast<std::size_t>(count.number()));
        }
        REQUIRE(actual.edges == static_cast<std::size_t>(expected["edges"].number()));
        REQUIRE(actual.single_input_gates == static_cast<std::size_t>(expected["single_input_gates"].number()));
        REQUIRE(actual.max_fan_in == static_cast<std::size_t>(expected["max_fan_in"].number()));
        REQUIRE(actual.depth == static_cast<std::uint32_t>(expected["depth"].number()));
        REQUIRE(actual.basic_events == static_cast<std::size_t>(expected["basic_events"].number()));
        REQUIRE(actual.import_lines == static_cast<std::size_t>(expected["import_lines"].number()));
        REQUIRE(actual.complemented_edges == static_cast<std::size_t>(expected["complemented_edges"].number()));
        REQUIRE(actual.complemented_edges_to_basic_events ==
                static_cast<std::size_t>(expected["complemented_edges_to_basic_events"].number()));
        REQUIRE(actual.roots == names(expected["roots"]));
        if (!expected["process_targets"].array().empty()) {
            REQUIRE(actual.targets == names(expected["process_targets"]));
        }
        if (expected["limit"].is_null()) {
            REQUIRE_FALSE(actual.limit);
        } else {
            REQUIRE(actual.limit);
            REQUIRE(std::abs(*actual.limit - expected["limit"].number()) <= 1e-12 * std::abs(expected["limit"].number()));
        }
        REQUIRE(actual.initiators == static_cast<std::size_t>(expected["initiator_tagged_events"].number()));
        REQUIRE(actual.initiators_p_1 == static_cast<std::size_t>(expected["initiator_tagged_events_with_p_1"].number()));
        REQUIRE(actual.events_p_1 == static_cast<std::size_t>(expected["events_p_eq_1"].number()));
        REQUIRE(actual.events_p_0 == static_cast<std::size_t>(expected["events_p_eq_0"].number()));
        REQUIRE(actual.names_not_ncname == static_cast<std::size_t>(expected["names_not_valid_xml_ncname"].number()));
        REQUIRE(actual.duplicate_imports == expected["duplicate_imports"].object().size());
    }

    [[nodiscard]] auto fixtureDirectories() -> std::vector<std::filesystem::path> {
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

} // namespace

TEST_CASE("FTP conformance fixtures match expected.json under every duplicate-import policy", "[io][ftp][conformance]") {
    const auto fixtures = fixtureDirectories();
    REQUIRE(fixtures.size() >= 9);

    for (auto const &fixture : fixtures) {
        DYNAMIC_SECTION(fixture.filename().string()) {
            const auto spec = ftp_test::parse_json(readText(fixture / "expected.json"));
            REQUIRE(spec);
            std::vector<std::filesystem::path> files;
            for (auto const &file : (*spec)["files"].array()) {
                files.push_back(fixture / file.string());
            }
            const bool legacyHeaderLimit =
                fixture.filename() == "header_limit" && readText(fixture / "model.ftp").find("ENDTREE") == std::string::npos;

            for (auto const &[policy, outcome] : (*spec)["policies"].object()) {
                INFO("policy " << policy);
                fuzzybits::api::option_set options;
                options.set(ftp::duplicate_import_option(), policy);
                const auto read = ftp::read_project(files, options);

                auto expected = legacyHeaderLimit ? legacyHeaderLimitDiagnostics() : expectedDiagnostics(outcome["diagnostics"]);
                // Fixtures generated before #191 do not list the missing-process warning.
                const bool documentsMissingProcess = std::ranges::any_of(expected, [](auto const &item) {
                    return item.code == "io.ftp.missing-process";
                });
                if (auto const *structure = spec->find("structure"); !documentsMissingProcess && structure != nullptr &&
                                                                     (*structure)["process_targets"].array().empty()) {
                    expected.push_back(expected_diagnostic{.severity = diagnostic_severity::warning, .code = "io.ftp.missing-process"});
                }
                requireDiagnostics(read.diagnostics(), expected);

                const bool success = !legacyHeaderLimit && !outcome["results"].is_null();
                REQUIRE(static_cast<bool>(read) == success);

                if (!success) {
                    continue;
                }

                auto const *model = pra::model_from(read.value());
                REQUIRE(model != nullptr);
                const auto counts = ftp_test::structure_of(*model, read.diagnostics());
                requireStructure(counts, (*spec)["structure"]);
                REQUIRE(counts.targets == names((*spec)["default_targets"]));

                for (auto const &[target, values] : outcome["results"].object()) {
                    INFO("target " << target);
                    auto subject = model->find(pra::element_kind::gate, target);
                    if (!subject.valid()) {
                        subject = model->find(pra::element_kind::basic_event, target);
                    }
                    REQUIRE(subject.valid());
                    const auto expectedValue = values["boolean"].number();
                    REQUIRE(std::abs(ftp_test::brute_force_probability(*model, subject) - expectedValue) <= 1e-9 * expectedValue + 1e-15);
                }
            }
        }
    }
}
