#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "runtime/progress.h"

namespace fuzzybits::runtime::progress {

enum class log_format { json_lines, csv };

/// Format implied by the file extension: `.jsonl` and `.ndjson` are JSON Lines,
/// `.csv` is CSV (long format). Anything else throws `std::invalid_argument`.
[[nodiscard]] log_format log_format_from_path(const std::filesystem::path &path);

/// Stable token for the format, as used in messages and tests.
[[nodiscard]] const char *log_format_name(log_format format) noexcept;

/// Column count of the CSV header; every row has exactly this many fields.
[[nodiscard]] std::size_t log_csv_column_count() noexcept;

/// `rel_half_width` as the snapshot model defines it: `(ci_high - ci_low) /
/// (2 estimate)`, NaN when the estimate is 0 or not finite.
[[nodiscard]] double relative_half_width(double estimate, double ci_low, double ci_high);

/// The finishing snapshot for `results` (anything with `name`, `hits`, `n`,
/// `estimate`, `ci_low`, `ci_high` per element), in query order. The engine
/// emits its own last snapshot; this gives the log a finish record that matches
/// the result the caller prints when the engine did not deliver one.
template <class Results> [[nodiscard]] snapshot final_snapshot(const Results &results) {
    snapshot s;
    s.stage = phase::done;
    s.queries.reserve(results.size());
    for (const auto &r : results) {
        query_state q;
        q.name = r.name;
        q.hits = static_cast<std::uint64_t>(r.hits);
        q.trials = static_cast<std::uint64_t>(r.n);
        q.estimate = r.estimate;
        q.ci_low = r.ci_low;
        q.ci_high = r.ci_high;
        q.rel_half_width = relative_half_width(r.estimate, r.ci_low, r.ci_high);
        if (s.queries.empty()) {
            s.trials_done = q.trials;
        }
        s.queries.push_back(std::move(q));
    }
    return s;
}

/// Finishing snapshot for a run with no query results (a compile, or a run that
/// failed before producing any).
[[nodiscard]] snapshot final_snapshot();

/// Writes one record per `on_snapshot` and exactly one finish record per
/// `on_finish`, flushing after each record so a killed run leaves a readable
/// file. The constructor opens (truncates) the file, so a bad path or an
/// unsupported extension fails before sampling starts. A write that fails later
/// sets `ok()` to false and drops the remaining records instead of throwing on
/// the run thread.
class log_sink : public sink {
  public:
    explicit log_sink(const std::filesystem::path &path);
    log_sink(std::filesystem::path path, log_format format);

    log_sink(const log_sink &) = delete;
    log_sink &operator=(const log_sink &) = delete;

    void on_snapshot(const snapshot &s) override;
    void on_finish(const snapshot &last, bool interrupted) override;

    [[nodiscard]] log_format format() const noexcept { return format_; }
    [[nodiscard]] const std::filesystem::path &path() const noexcept { return path_; }
    [[nodiscard]] bool ok() const noexcept { return !failed_; }
    [[nodiscard]] bool finished() const noexcept { return finished_; }

  private:
    void write(const snapshot &s, bool finish, bool interrupted);

    std::filesystem::path path_;
    log_format format_;
    std::ofstream out_;
    std::uint64_t records_ = 0;
    bool finished_ = false;
    bool failed_ = false;
};

/// Guarantees exactly one finish record: a no-op when the engine already called
/// `on_finish`. A run passes its results so the record matches them; a compile
/// (or a run that produced nothing) passes none.
template <class Results> void finish_log(log_sink &log, const Results &results, bool interrupted = false) {
    if (!log.finished()) {
        log.on_finish(final_snapshot(results), interrupted);
    }
}

inline void finish_log(log_sink &log, bool interrupted = false) {
    if (!log.finished()) {
        log.on_finish(final_snapshot(), interrupted);
    }
}

} // namespace fuzzybits::runtime::progress
