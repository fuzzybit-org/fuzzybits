#include <catch2/catch_test_macros.hpp>

#include "runtime/tests/test_backend.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef FZB_PATH
#error "FZB_PATH must be provided by CMake"
#endif
#ifndef PROJECT_SOURCE_DIR
#error "PROJECT_SOURCE_DIR must be provided by CMake"
#endif

namespace {

int run_cmd(const std::string &cmd) {
#ifdef _WIN32
    return std::system(("cmd /S /C \"" + cmd + "\"").c_str());
#else
    return std::system(cmd.c_str());
#endif
}

std::string slurp(const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::vector<std::string> split_lines(const std::string &text) {
    std::vector<std::string> lines;
    std::string current;
    for (const char c : text) {
        if (c == '\n') {
            lines.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        lines.push_back(current);
    }
    return lines;
}

/// The scalar field `"key":value` of one JSON Lines record, as text.
std::string json_value(const std::string &record, const std::string &key) {
    const std::string pattern = "\"" + key + "\":";
    const auto pos = record.find(pattern);
    REQUIRE(pos != std::string::npos);
    const auto start = pos + pattern.size();
    auto end = record.find_first_of(",}", start);
    REQUIRE(end != std::string::npos);
    return record.substr(start, end - start);
}

std::uint64_t json_uint(const std::string &record, const std::string &key) { return static_cast<std::uint64_t>(std::stoull(json_value(record, key))); }

/// ``"name":"TOP"`` style string field.
std::string json_string(const std::string &record, const std::string &key) {
    const std::string pattern = "\"" + key + "\":\"";
    const auto pos = record.find(pattern);
    REQUIRE(pos != std::string::npos);
    const auto start = pos + pattern.size();
    const auto end = record.find('"', start);
    REQUIRE(end != std::string::npos);
    return record.substr(start, end - start);
}

std::filesystem::path scratch(const std::string &name) {
    const auto dir = std::filesystem::temp_directory_path() / "fuzzybits-progress-log-cli";
    std::filesystem::create_directories(dir);
    const auto path = dir / name;
    std::filesystem::remove(path);
    return path;
}

std::string quote(const std::string &text) {
#ifdef _WIN32
    return "\"" + text + "\"";
#else
    return "'" + text + "'";
#endif
}

/// Compile chinese once per test binary, into the scratch directory.
std::filesystem::path chinese_pcfb() {
    static const auto path = [] {
        const auto out = scratch("chinese.pcfb");
        const auto xml = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia" / "chinese.xml";
        REQUIRE(std::filesystem::exists(xml));
        const std::string cmd = quote(FZB_PATH) + " compile " + quote(xml.string()) + " -c 2 --no-kn -o " + quote(out.string());
        REQUIRE(run_cmd(cmd) == 0);
        REQUIRE(std::filesystem::exists(out));
        return out;
    }();
    return path;
}

std::string run_json(const std::filesystem::path &pcfb, const std::vector<std::string> &extra) {
    const auto json = scratch("run.json");
    std::string cmd = quote(FZB_PATH) + " run " + quote(pcfb.string()) + " --trials 1048576" + fuzzybits::testing::backend_flag() + " --json";
    for (const auto &item : extra) {
        cmd += " " + item;
    }
    cmd += " > " + quote(json.string());
    REQUIRE(run_cmd(cmd) == 0);
    return slurp(json);
}

} // namespace

TEST_CASE("fzb run --progress-log writes JSON Lines with schema and a matching finish record", "[mc][cli][progress-log]") {
    const auto pcfb = chinese_pcfb();
    const auto log = scratch("run.jsonl");
    const auto printed = run_json(pcfb, {"--progress-log", quote(log.string())});
    REQUIRE(std::filesystem::exists(log));

    const auto lines = split_lines(slurp(log));
    REQUIRE_FALSE(lines.empty());
    CHECK(lines.back().find("\"event\":\"finish\"") != std::string::npos);

    std::vector<std::uint64_t> done;
    for (const auto &line : lines) {
        REQUIRE(line.front() == '{');
        CHECK(json_uint(line, "schema") == 1);
        done.push_back(json_uint(line, "trials_done"));
    }
    CHECK(std::is_sorted(done.begin(), done.end()));

    // The finish record repeats the values the CLI printed: same hits and n.
    const auto &last = lines.back();
    const auto n = static_cast<std::uint64_t>(std::stoull(json_value(printed, "n")));
    const auto hits = static_cast<std::uint64_t>(std::stoull(json_value(printed, "hits")));
    CHECK(json_uint(last, "trials_done") == n);
    CHECK(last.find("\"trials\":" + std::to_string(n)) != std::string::npos);
    CHECK(last.find("\"hits\":" + std::to_string(hits)) != std::string::npos);
    CHECK(last.find("\"estimate\":") != std::string::npos);
    CHECK(json_string(last, "name").find("r1") != std::string::npos);
}

TEST_CASE("fzb run --progress-log writes CSV with a header and one row per query", "[mc][cli][progress-log]") {
    const auto pcfb = chinese_pcfb();
    const auto log = scratch("run.csv");
    run_json(pcfb, {"--progress-log", quote(log.string())});
    REQUIRE(std::filesystem::exists(log));

    const auto lines = split_lines(slurp(log));
    REQUIRE(lines.size() >= 2);
    CHECK(lines.front().find("record,event,interrupted,schema") == 0);
    CHECK(lines.front().find("query_name") != std::string::npos);
    CHECK(lines.front().find("rel_half_width") != std::string::npos);
    CHECK(lines.front().find("converged") != std::string::npos);
    // Long format: one row per (snapshot, query); the finish rows come last.
    CHECK(lines.back().find(",finish,") != std::string::npos);
    std::size_t finish_rows = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
        if (lines[i].find(",finish,") != std::string::npos) {
            ++finish_rows;
        }
    }
    CHECK(finish_rows == 1);
}

TEST_CASE("fzb run --progress-log is independent of the terminal display", "[mc][cli][progress-log]") {
    const auto pcfb = chinese_pcfb();
    // stderr redirected to a file is not a terminal, and the log is still written.
    const auto log = scratch("quiet.jsonl");
    const auto printed = run_json(pcfb, {"--progress-log", quote(log.string())});
    CHECK_FALSE(printed.empty());
    CHECK(std::filesystem::exists(log));
    CHECK_FALSE(split_lines(slurp(log)).empty());
}

TEST_CASE("fzb run --progress-log rejects an unknown extension before sampling", "[mc][cli][progress-log]") {
    const auto pcfb = chinese_pcfb();
    const auto log = scratch("run.txt");
#ifdef _WIN32
    const std::string cmd = quote(FZB_PATH) + " run " + quote(pcfb.string()) + " --trials 1048576 --progress-log " + quote(log.string()) + " >NUL 2>&1";
#else
    const std::string cmd = quote(FZB_PATH) + " run " + quote(pcfb.string()) + " --trials 1048576 --progress-log " + quote(log.string()) + " >/dev/null 2>&1";
#endif
    CHECK(run_cmd(cmd) != 0);
    CHECK_FALSE(std::filesystem::exists(log));
}

TEST_CASE("fzb run --progress-log rejects a missing directory", "[mc][cli][progress-log]") {
    const auto pcfb = chinese_pcfb();
    const auto log = scratch("no-such-dir/run.jsonl");
#ifdef _WIN32
    const std::string cmd = quote(FZB_PATH) + " run " + quote(pcfb.string()) + " --trials 1048576 --progress-log " + quote(log.string()) + " >NUL 2>&1";
#else
    const std::string cmd = quote(FZB_PATH) + " run " + quote(pcfb.string()) + " --trials 1048576 --progress-log " + quote(log.string()) + " >/dev/null 2>&1";
#endif
    CHECK(run_cmd(cmd) != 0);
    CHECK_FALSE(std::filesystem::exists(log));
}

TEST_CASE("fzb compile --progress-log records the compile stages", "[mc][cli][progress-log]") {
    const auto xml = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia" / "chinese.xml";
    const auto out = scratch("compile-out.pcfb");
    const auto log = scratch("compile.jsonl");
    const std::string cmd = quote(FZB_PATH) + " compile " + quote(xml.string()) + " -o " + quote(out.string()) + " --progress-log " + quote(log.string());
    REQUIRE(run_cmd(cmd) == 0);
    REQUIRE(std::filesystem::exists(out));
    const auto lines = split_lines(slurp(log));
    REQUIRE_FALSE(lines.empty());
    for (const auto &line : lines) {
        CHECK(json_uint(line, "schema") == 1);
    }
    CHECK(lines.back().find("\"event\":\"finish\"") != std::string::npos);
}

TEST_CASE("fzb without --progress-log creates no file", "[mc][cli][progress-log]") {
    const auto pcfb = chinese_pcfb();
    const auto before = std::filesystem::exists(scratch("unasked.jsonl"));
    REQUIRE_FALSE(before);
    run_json(pcfb, {});
    CHECK_FALSE(std::filesystem::exists(scratch("unasked.jsonl")));
}
