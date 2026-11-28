#include <catch2/catch_test_macros.hpp>

#include "compile/ftp/json.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

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

void write_file(std::filesystem::path const &path, std::string const &text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
}

[[nodiscard]] auto exit_code(const int status) -> int {
#ifdef _WIN32
    return status;
#else
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

[[nodiscard]] auto run_fzb(std::string const &arguments, std::filesystem::path const &scratch) -> run_result {
    const auto out_path = scratch / "fzb.out";
    const auto err_path = scratch / "fzb.err";
    const auto command = quoted(FZB_PATH) + " " + arguments + " > " + quoted(out_path.string()) + " 2> " + quoted(err_path.string());
#ifdef _WIN32
    const int status = std::system(("cmd /S /C \"" + command + "\"").c_str());
#else
    const int status = std::system(command.c_str());
#endif
    return run_result{.status = exit_code(status), .out = read_file(out_path), .err = read_file(err_path)};
}

[[nodiscard]] auto scratch_directory(std::string const &name) -> std::filesystem::path {
    const auto path = std::filesystem::temp_directory_path() / ("fuzzybits-inspect-ftp-cli-" + name);
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

[[nodiscard]] auto fixtures_root() -> std::filesystem::path { return std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "conformance"; }

[[nodiscard]] auto parse_output(std::string const &text) -> json::value {
    INFO(text);
    auto parsed = json::parse(text);
    REQUIRE(parsed);
    return std::move(parsed).value();
}

constexpr auto clean_project = "TOP + A B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\n";
constexpr auto warning_project = "TOP + A B\nENDTREE\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\n";
constexpr auto error_project = "TOP + A B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n";

[[nodiscard]] auto contains_word(std::string const &text, std::string const &word) -> bool {
    const auto is_word = [](const char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_'; };
    for (auto at = text.find(word); at != std::string::npos; at = text.find(word, at + 1)) {
        const auto end = at + word.size();
        if ((at == 0 || !is_word(text[at - 1])) && (end == text.size() || !is_word(text[end]))) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST_CASE("fzb inspect FTP: text output of a project", "[compile][ftp][inspect][cli]") {
    const auto scratch = scratch_directory("text");
    write_file(scratch / "model.ftp", clean_project);
    const auto run = run_fzb("inspect " + quoted((scratch / "model.ftp").string()), scratch);
    INFO(run.out << run.err);
    REQUIRE(run.status == 0);
    REQUIRE(run.out.find("status: ok") != std::string::npos);
    REQUIRE(run.out.find("model.ftp") != std::string::npos);
    REQUIRE(run.out.find("gates: 1 (or 1, and 0, atleast 0)") != std::string::npos);
    REQUIRE(run.out.find("basic_events: 2") != std::string::npos);
    REQUIRE(run.out.find("conventions:") != std::string::npos);
    REQUIRE(run.out.find("process_targets: 1") != std::string::npos);
}

TEST_CASE("fzb inspect FTP: JSON output and the conformance structure", "[compile][ftp][inspect][cli]") {
    const auto scratch = scratch_directory("json");
    const auto fixture = fixtures_root() / "imports_tags_duplicates";
    const auto run = run_fzb("inspect " + quoted(fixture.string()) + " --json", scratch);
    INFO(run.err);
    REQUIRE(run.status == 0);
    const auto document = parse_output(run.out);
    REQUIRE(document.find("status")->string == "ok");
    REQUIRE(document.find("files")->items.size() == 1);
    REQUIRE(document.find("conventions")->find("LIMIT")->string.empty() == false);

    const auto expected = parse_output(read_file(fixture / "expected.json"));
    auto const &structure = *document.find("structure");
    auto const &want = *expected.find("structure");
    REQUIRE(structure.find("gates")->integer == want.find("gates")->integer);
    REQUIRE(structure.find("edges")->integer == want.find("edges")->integer);
    REQUIRE(structure.find("initiator_tagged_events")->integer == want.find("initiator_tagged_events")->integer);
    REQUIRE(structure.find("duplicate_imports")->keys == want.find("duplicate_imports")->keys);
    REQUIRE(document.find("diagnostics")->items.size() == 1);
    auto const &diagnostic = document.find("diagnostics")->items.front();
    REQUIRE(diagnostic.find("id")->string == "io.ftp.duplicate-import");
    REQUIRE(diagnostic.find("severity")->string == "warning");
    REQUIRE(diagnostic.find("line")->integer == 12);
    REQUIRE(diagnostic.find("message") != nullptr);
}

TEST_CASE("fzb inspect FTP: exit status, --strict and the duplicate-import policy", "[compile][ftp][inspect][cli]") {
    const auto scratch = scratch_directory("exit");
    write_file(scratch / "clean.ftp", clean_project);
    write_file(scratch / "warning.ftp", warning_project);
    write_file(scratch / "error.ftp", error_project);
    const auto at = [&](std::string const &name) { return quoted((scratch / name).string()); };

    REQUIRE(run_fzb("inspect " + at("clean.ftp"), scratch).status == 0);
    REQUIRE(run_fzb("inspect " + at("clean.ftp") + " --strict", scratch).status == 0);
    REQUIRE(run_fzb("inspect " + at("warning.ftp"), scratch).status == 0);
    REQUIRE(run_fzb("inspect " + at("warning.ftp") + " --strict", scratch).status == 1);

    const auto broken = run_fzb("inspect " + at("error.ftp") + " --json", scratch);
    REQUIRE(broken.status == 1);
    const auto document = parse_output(broken.out);
    REQUIRE(document.find("status")->string == "error");
    REQUIRE(document.find("structure")->is(json::value::kind_type::null));
    REQUIRE(document.find("diagnostics")->items.front().find("id")->string == "io.ftp.undefined-reference");

    const auto duplicates = quoted((fixtures_root() / "imports_tags_duplicates").string());
    REQUIRE(run_fzb("inspect " + duplicates + " --duplicate-import first", scratch).status == 0);
    REQUIRE(run_fzb("inspect " + duplicates + " --duplicate-import last --strict", scratch).status == 1);
    REQUIRE(run_fzb("inspect " + duplicates + " --duplicate-import error", scratch).status == 1);
}

TEST_CASE("fzb inspect FTP: usage and option errors", "[compile][ftp][inspect][cli]") {
    const auto scratch = scratch_directory("usage");
    write_file(scratch / "model.ftp", clean_project);
    write_file(scratch / "other.xml", "<opsa-mef/>\n");
    const auto ftp = quoted((scratch / "model.ftp").string());

    auto run = run_fzb("inspect " + ftp + " --duplicate-import newest", scratch);
    REQUIRE(run.status == 1);
    REQUIRE(run.err.find("fzb inspect: --duplicate-import must be error, first or last") != std::string::npos);

    // Removed: -EVENT is always a marked literal.
    run = run_fzb("inspect " + ftp + " --complement-events negation", scratch);
    REQUIRE(run.status == 1);
    REQUIRE(run.err.find("complement-events") != std::string::npos);
    REQUIRE(run.out.empty());

    run = run_fzb("inspect " + ftp + " --exact", scratch);
    REQUIRE(run.status == 1);
    REQUIRE(run.err.find("--exact applies to .pcfb input only") != std::string::npos);

    run = run_fzb("inspect " + ftp + " " + quoted((scratch / "other.xml").string()), scratch);
    REQUIRE(run.status == 1);
    REQUIRE(run.err.find("FTP and non-FTP inputs cannot be combined") != std::string::npos);

    run = run_fzb("inspect " + quoted((scratch / "other.xml").string()) + " --counts-only", scratch);
    REQUIRE(run.status == 1);
    REQUIRE(run.err.find("--counts-only applies to CAFTA FTP input only") != std::string::npos);

    run = run_fzb("inspect " + ftp + " --no-such-flag", scratch);
    REQUIRE(run.status == 1);
    REQUIRE(run.err.find("Option error") != std::string::npos);

    run = run_fzb("inspect", scratch);
    REQUIRE(run.status == 1);

    run = run_fzb("inspect " + quoted((scratch / "missing.ftp").string()) + " --json", scratch);
    REQUIRE(run.status == 1);
    REQUIRE(parse_output(run.out).find("diagnostics")->items.front().find("id")->string == "io.ftp.open-failed");
}

TEST_CASE("fzb inspect FTP: a directory and a file list read the same project", "[compile][ftp][inspect][cli]") {
    const auto scratch = scratch_directory("list");
    const auto fixture = fixtures_root() / "multi_file_project";
    const auto by_directory = run_fzb("inspect " + quoted(fixture.string()) + " --json", scratch);
    REQUIRE(by_directory.status == 0);

    std::string list;
    for (auto const &name : {"01_top.ftp", "02_systems.ftp", "03_imports.ftp"}) {
        list += " " + quoted((fixture / name).string());
    }
    const auto by_files = run_fzb("inspect" + list + " --json", scratch);
    REQUIRE(by_files.status == 0);
    REQUIRE(by_files.out == by_directory.out);

    const auto document = parse_output(by_files.out);
    REQUIRE(document.find("files")->items.size() == 3);
    REQUIRE(std::filesystem::path(document.find("files")->items[0].string).filename() == "01_top.ftp");
    REQUIRE(std::filesystem::path(document.find("files")->items[2].string).filename() == "03_imports.ftp");

    const auto reversed = run_fzb("inspect " + quoted((fixture / "03_imports.ftp").string()) + " " + quoted((fixture / "02_systems.ftp").string()) + " " +
                                      quoted((fixture / "01_top.ftp").string()) + " --json",
                                  scratch);
    const auto order = parse_output(reversed.out);
    REQUIRE(std::filesystem::path(order.find("files")->items[0].string).filename() == "03_imports.ftp");
}

TEST_CASE("fzb inspect FTP: --counts-only prints no element names", "[compile][ftp][inspect][cli]") {
    const auto scratch = scratch_directory("counts");
    const auto project = scratch / "wombat";
    std::filesystem::create_directories(project);
    write_file(project / "m.ftp", "**DBNAME: WOMBAT_HEADER\nZEBRA_TOP * QUOKKA_G -QUOKKA_B\nQUOKKA_G + QUOKKA_B QUOKKA_C\n"
                                  "ENDTREE\nPROCESS ZEBRA_TOP\nIMPORT\n"
                                  "  1.0E-01 QUOKKA_B\n  2.0E-01 QUOKKA_C\n  3.0E-01 QUOKKA_B\n");
    const std::vector<std::string> names{"ZEBRA_TOP", "QUOKKA_G", "QUOKKA_B", "QUOKKA_C", "WOMBAT_HEADER"};

    for (auto const *format : {"", " --json"}) {
        const auto run = run_fzb("inspect " + quoted(project.string()) + " --counts-only" + format, scratch);
        INFO(run.out);
        REQUIRE(run.status == 0);
        for (auto const &name : names) {
            REQUIRE(run.out.find(name) == std::string::npos);
            REQUIRE(run.err.find(name) == std::string::npos);
        }
        REQUIRE(run.out.find("io.ftp.duplicate-import") != std::string::npos);
        REQUIRE(run.out.find("m.ftp") != std::string::npos);
    }

    const auto full = run_fzb("inspect " + quoted(project.string()), scratch);
    REQUIRE(full.out.find("ZEBRA_TOP") != std::string::npos);
    REQUIRE(full.out.find("WOMBAT_HEADER") != std::string::npos);

    const auto duplicates = quoted((fixtures_root() / "imports_tags_duplicates").string());
    const auto fixture = run_fzb("inspect " + duplicates + " --counts-only", scratch);
    for (auto const *name : {"TOP", "A"}) {
        REQUIRE_FALSE(contains_word(fixture.out, name));
    }
}
