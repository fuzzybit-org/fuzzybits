#include "runtime/progress_log.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace progress = fuzzybits::runtime::progress;

std::filesystem::path scratch_dir() {
    const auto dir = std::filesystem::temp_directory_path() / "fuzzybits-progress-log-tests";
    std::filesystem::create_directories(dir);
    return dir;
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

/// RFC 4180 reader for one line: quoted fields may hold commas and doubled quotes.
std::vector<std::string> parse_csv_fields(const std::string &line) {
    std::vector<std::string> fields(1);
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                fields.back().push_back('"');
                ++i;
            } else if (c == '"') {
                quoted = false;
            } else {
                fields.back().push_back(c);
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == ',') {
            fields.emplace_back();
        } else {
            fields.back().push_back(c);
        }
    }
    return fields;
}

progress::query_state query(std::string name, std::uint64_t hits, std::uint64_t trials, double estimate, double ci_low,
                            double ci_high) {
    progress::query_state q;
    q.name = std::move(name);
    q.hits = hits;
    q.trials = trials;
    q.estimate = estimate;
    q.ci_low = ci_low;
    q.ci_high = ci_high;
    q.rel_half_width = progress::relative_half_width(estimate, ci_low, ci_high);
    return q;
}

progress::snapshot two_query_snapshot() {
    progress::snapshot s;
    s.stage = progress::phase::sampling;
    s.trials_done = 4096;
    s.trials_total = 8192;
    s.passes_done = 1;
    s.passes_total = 2;
    s.elapsed_seconds = 0.25;
    s.trials_per_second = 16384.0;
    s.queries.push_back(query("TOP", 12, 4096, 0.0029296875, 0.0012, 0.0046));
    // Not yet computed: every double is NaN, which JSON must spell as null.
    s.queries.push_back(query("no,comma\"quote", 0, 4096, std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()));
    return s;
}

} // namespace

TEST_CASE("progress log format comes from the extension", "[runtime][progress-log]") {
    CHECK(progress::log_format_from_path("run.jsonl") == progress::log_format::json_lines);
    CHECK(progress::log_format_from_path("run.ndjson") == progress::log_format::json_lines);
    CHECK(progress::log_format_from_path("run.CSV") == progress::log_format::csv);
    CHECK_THROWS_AS(progress::log_format_from_path("run.txt"), std::invalid_argument);
    CHECK_THROWS_AS(progress::log_format_from_path("run"), std::invalid_argument);
    CHECK_THROWS_AS(progress::log_format_from_path("run.csv.gz"), std::invalid_argument);
}

TEST_CASE("progress log refuses a path it cannot open", "[runtime][progress-log]") {
    const auto path = scratch_dir() / "missing-directory" / "run.jsonl";
    CHECK_THROWS_AS(progress::log_sink(path), std::runtime_error);
}

TEST_CASE("JSON Lines progress log: every field, NaN as null, one finish record", "[runtime][progress-log]") {
    const auto path = scratch_dir() / "jsonl.jsonl";
    progress::log_sink log(path);
    log.on_snapshot(two_query_snapshot());
    progress::snapshot last = two_query_snapshot();
    last.stage = progress::phase::done;
    last.trials_done = 8192;
    last.passes_done = 2;
    log.on_finish(last, /*interrupted=*/false);

    const auto lines = split_lines(slurp(path));
    REQUIRE(lines.size() == 2);

    const std::string &first = lines[0];
    CHECK(first.front() == '{');
    CHECK(first.back() == '}');
    for (const char *key :
         {"\"record\":0", "\"event\":\"snapshot\"", "\"interrupted\":false", "\"schema\":1", "\"phase\":\"sampling\"",
          "\"step\":\"\"", "\"trials_done\":4096", "\"trials_total\":8192", "\"passes_done\":1", "\"passes_total\":2",
          "\"elapsed_seconds\":0.25", "\"trials_per_second\":16384", "\"name\":\"TOP\"", "\"hits\":12",
          "\"trials\":4096", "\"estimate\":0.0029296875", "\"ci_low\":0.0012", "\"ci_high\":0.0046",
          "\"converged\":false"}) {
        INFO("missing " << key);
        CHECK(first.find(key) != std::string::npos);
    }
    // A NaN never leaks into JSON: the second query serialises every double as null.
    CHECK(first.find("nan") == std::string::npos);
    CHECK(first.find("NaN") == std::string::npos);
    CHECK(first.find("inf") == std::string::npos);
    const auto second_query = first.substr(first.find("no,comma"));
    CHECK(second_query.find("\"estimate\":null") != std::string::npos);
    CHECK(second_query.find("\"ci_low\":null") != std::string::npos);
    CHECK(second_query.find("\"rel_half_width\":null") != std::string::npos);
    CHECK(second_query.find("\"target_rel_half_width\":null") != std::string::npos);

    CHECK(lines[1].find("\"event\":\"finish\"") != std::string::npos);
    CHECK(lines[1].find("\"phase\":\"done\"") != std::string::npos);

    // A snapshot after on_finish must not land behind the finish record.
    log.on_snapshot(two_query_snapshot());
    CHECK(split_lines(slurp(path)).size() == 2);
}

TEST_CASE("CSV progress log: header, one row per query, RFC 4180 quoting, empty cells", "[runtime][progress-log]") {
    const auto path = scratch_dir() / "run.csv";
    progress::log_sink log(path);
    log.on_snapshot(two_query_snapshot());
    log.on_finish(two_query_snapshot(), /*interrupted=*/true);

    const auto lines = split_lines(slurp(path));
    REQUIRE(lines.size() >= 5); // header + 2 queries + 2 queries

    const auto header = lines.front();
    std::vector<std::string> columns;
    std::string current;
    for (const char c : header) {
        if (c == ',') {
            columns.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    columns.push_back(current);
    CHECK(columns.size() == progress::log_csv_column_count());
    CHECK(columns.front() == "record");
    CHECK(columns.back() == "converged");
    for (const char *required :
         {"schema", "phase", "step", "trials_done", "trials_total", "query_name", "hits", "trials", "estimate",
          "ci_low", "ci_high", "rel_half_width", "target_rel_half_width"}) {
        INFO("missing column " << required);
        CHECK(std::find(columns.begin(), columns.end(), required) != columns.end());
    }

    // The name holds a comma and a quote: RFC 4180 quotes it and doubles the quote.
    CHECK(lines[2].find("\"no,comma\"\"quote\"") != std::string::npos);
    // Its doubles are NaN: empty cells, not the text "nan".
    CHECK(lines[2].find("nan") == std::string::npos);
    CHECK(lines[2].find("0,4096,,,,,,false") != std::string::npos);

    CHECK(lines[3].find("finish") != std::string::npos);
    CHECK(lines[3].find("true") != std::string::npos); // interrupted
}

TEST_CASE("CSV progress log keeps a query-less snapshot as one row", "[runtime][progress-log]") {
    const auto path = scratch_dir() / "compile.csv";
    progress::log_sink log(path);
    progress::snapshot s;
    s.stage = progress::phase::compile;
    s.step = "passes";
    s.passes_done = 2;
    s.passes_total = 5;
    log.on_snapshot(s);
    progress::finish_log(log);

    const auto lines = split_lines(slurp(path));
    REQUIRE(lines.size() == 3);
    CHECK(lines[1].find("passes") != std::string::npos);
    // Count fields with the RFC 4180 reader, not commas: a name may hold commas.
    CHECK(parse_csv_fields(lines[0]).size() == progress::log_csv_column_count());
    CHECK(parse_csv_fields(lines[1]).size() == progress::log_csv_column_count());
    CHECK(parse_csv_fields(lines[2]).size() == progress::log_csv_column_count());
}

TEST_CASE("progress log finish record matches the returned results", "[runtime][progress-log]") {
    struct Row {
        std::string name;
        std::uint64_t hits;
        std::uint64_t n;
        double estimate;
        double ci_low;
        double ci_high;
    };
    const std::vector<Row> results{{"TOP", 300, 1000, 0.3, 0.27, 0.33}, {"ALT", 40, 1000, 0.04, 0.03, 0.05}};

    const auto snapshot_done = progress::final_snapshot(results);
    CHECK(snapshot_done.stage == progress::phase::done);
    REQUIRE(snapshot_done.queries.size() == 2);
    CHECK(snapshot_done.trials_done == 1000);
    CHECK(snapshot_done.queries[0].estimate == 0.3);
    CHECK(snapshot_done.queries[0].trials == 1000);
    CHECK(snapshot_done.queries[0].rel_half_width == (0.33 - 0.27) / (2 * 0.3));

    // estimate == 0 has no relative half-width.
    CHECK(std::isnan(progress::relative_half_width(0.0, 0.0, 0.0)));
    CHECK(std::isnan(progress::relative_half_width(std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0)));

    const auto path = scratch_dir() / "results.jsonl";
    progress::log_sink log(path);
    progress::finish_log(log, results);
    const auto lines = split_lines(slurp(path));
    REQUIRE(lines.size() == 1);
    CHECK(lines.front().find("\"event\":\"finish\"") != std::string::npos);
    CHECK(lines.front().find("\"event\":\"snapshot\"") == std::string::npos);
    CHECK(lines.front().find("\"estimate\":0.3") != std::string::npos);
    CHECK(lines.front().find("\"name\":\"ALT\"") != std::string::npos);
}
