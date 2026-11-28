#include "runtime/progress_log.h"

#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace fuzzybits::runtime::progress {
namespace {

constexpr std::string_view kJsonLinesExt = ".jsonl";
constexpr std::string_view kJsonLinesAltExt = ".ndjson";
constexpr std::string_view kCsvExt = ".csv";

const char *phase_token(phase p) noexcept {
    switch (p) {
    case phase::compile:
        return "compile";
    case phase::sampling:
        return "sampling";
    case phase::checking:
        return "checking";
    case phase::done:
        return "done";
    }
    return "sampling";
}

/// Shortest round-trip decimal representation (`std::to_chars`), so a reader
/// recovers the exact double the snapshot carried.
void append_number(std::string &out, double value) {
    std::array<char, 64> buf{};
    const auto [end, ec] = std::to_chars(buf.data(), buf.data() + buf.size(), value);
    if (ec == std::errc{}) {
        out.append(buf.data(), static_cast<std::size_t>(end - buf.data()));
    }
}

void append_uint(std::string &out, std::uint64_t value) {
    std::array<char, 32> buf{};
    const auto [end, ec] = std::to_chars(buf.data(), buf.data() + buf.size(), value);
    if (ec == std::errc{}) {
        out.append(buf.data(), static_cast<std::size_t>(end - buf.data()));
    }
}

void append_json_string(std::string &out, std::string_view text) {
    out.push_back('"');
    for (const char raw : text) {
        const auto c = static_cast<unsigned char>(raw);
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                static constexpr char kHex[] = "0123456789abcdef";
                out += "\\u00";
                out.push_back(kHex[c >> 4]);
                out.push_back(kHex[c & 0x0f]);
            } else {
                out.push_back(static_cast<char>(c));
            }
        }
    }
    out.push_back('"');
}

/// JSON has no NaN/Infinity: non-finite values become `null`.
void append_json_double(std::string &out, double value) {
    if (std::isfinite(value)) {
        append_number(out, value);
    } else {
        out += "null";
    }
}

bool csv_needs_quotes(std::string_view text) {
    if (text.empty()) {
        return false;
    }
    if (text.front() == ' ' || text.back() == ' ') {
        return true;
    }
    for (const char c : text) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            return true;
        }
    }
    return false;
}

/// RFC 4180: quote when the field holds a comma, a quote, a newline, or
/// leading/trailing blanks; embedded quotes are doubled.
void append_csv_field(std::string &out, std::string_view text) {
    if (!csv_needs_quotes(text)) {
        out.append(text);
        return;
    }
    out.push_back('"');
    for (const char c : text) {
        if (c == '"') {
            out.push_back('"');
        }
        out.push_back(c);
    }
    out.push_back('"');
}

/// Long format cannot spell a missing double: an empty cell says "not
/// computed", matching the JSON `null`.
void append_csv_double(std::string &out, double value) {
    if (std::isfinite(value)) {
        append_number(out, value);
    }
}

void append_csv_bool(std::string &out, bool value) { out += value ? "true" : "false"; }

constexpr std::string_view kCsvHeader = "record,event,interrupted,schema,phase,step,trials_done,trials_total,passes_done,passes_total,"
                                        "elapsed_seconds,trials_per_second,query_index,query_name,hits,trials,estimate,ci_low,ci_high,"
                                        "rel_half_width,target_rel_half_width,converged";

constexpr std::size_t kCsvColumns = 22;

std::string json_record(const snapshot &s, std::uint64_t record, bool finish, bool interrupted) {
    std::string out;
    out.reserve(320 + 192 * s.queries.size());
    out += "{\"record\":";
    append_uint(out, record);
    out += ",\"event\":";
    out += finish ? "\"finish\"" : "\"snapshot\"";
    out += ",\"interrupted\":";
    out += interrupted ? "true" : "false";
    out += ",\"schema\":";
    append_uint(out, s.schema);
    out += ",\"phase\":";
    append_json_string(out, phase_token(s.stage));
    out += ",\"step\":";
    append_json_string(out, s.step);
    out += ",\"trials_done\":";
    append_uint(out, s.trials_done);
    out += ",\"trials_total\":";
    append_uint(out, s.trials_total);
    out += ",\"passes_done\":";
    append_uint(out, s.passes_done);
    out += ",\"passes_total\":";
    append_uint(out, s.passes_total);
    out += ",\"elapsed_seconds\":";
    append_json_double(out, s.elapsed_seconds);
    out += ",\"trials_per_second\":";
    append_json_double(out, s.trials_per_second);
    out += ",\"queries\":[";
    for (std::size_t i = 0; i < s.queries.size(); ++i) {
        const query_state &q = s.queries[i];
        out += i == 0 ? "{\"name\":" : ",{\"name\":";
        append_json_string(out, q.name);
        out += ",\"hits\":";
        append_uint(out, q.hits);
        out += ",\"trials\":";
        append_uint(out, q.trials);
        out += ",\"estimate\":";
        append_json_double(out, q.estimate);
        out += ",\"ci_low\":";
        append_json_double(out, q.ci_low);
        out += ",\"ci_high\":";
        append_json_double(out, q.ci_high);
        out += ",\"rel_half_width\":";
        append_json_double(out, q.rel_half_width);
        out += ",\"target_rel_half_width\":";
        append_json_double(out, q.target_rel_half_width);
        out += ",\"converged\":";
        out += q.converged ? "true" : "false";
        out.push_back('}');
    }
    out += "]}\n";
    return out;
}

/// One row per (snapshot, query), with the snapshot scalars repeated. A
/// snapshot with no queries (compile stages) still emits one row whose query
/// columns are empty, so no record is lost.
std::string csv_records(const snapshot &s, std::uint64_t record, bool finish, bool interrupted) {
    std::string out;
    // Fields are appended into a fixed-order vector and joined, so a row always
    // has exactly the header's column count even when a value is empty.
    std::vector<std::string> fields;
    fields.reserve(kCsvColumns);
    const auto row = [&](std::size_t query_index, const query_state *q) {
        fields.clear();
        const auto add = [&](std::string value) { fields.push_back(std::move(value)); };
        const auto add_uint = [&](std::uint64_t value) {
            std::string text;
            append_uint(text, value);
            fields.push_back(std::move(text));
        };
        const auto add_double = [&](double value) {
            std::string text;
            append_csv_double(text, value);
            fields.push_back(std::move(text));
        };
        const auto add_text = [&](std::string_view value) {
            std::string text;
            append_csv_field(text, value);
            fields.push_back(std::move(text));
        };

        add_uint(record);
        add(finish ? "finish" : "snapshot");
        add(interrupted ? "true" : "false");
        add_uint(s.schema);
        add_text(phase_token(s.stage));
        add_text(s.step);
        add_uint(s.trials_done);
        add_uint(s.trials_total);
        add_uint(s.passes_done);
        add_uint(s.passes_total);
        add_double(s.elapsed_seconds);
        add_double(s.trials_per_second);
        if (q == nullptr) {
            // A snapshot without queries (compile stages) keeps one row with the
            // query columns empty rather than dropping the record.
            fields.resize(kCsvColumns);
        } else {
            add_uint(query_index);
            add_text(q->name);
            add_uint(q->hits);
            add_uint(q->trials);
            add_double(q->estimate);
            add_double(q->ci_low);
            add_double(q->ci_high);
            add_double(q->rel_half_width);
            add_double(q->target_rel_half_width);
            add(q->converged ? "true" : "false");
        }
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (i != 0) {
                out.push_back(',');
            }
            out += fields[i];
        }
        out.push_back('\n');
    };

    if (s.queries.empty()) {
        row(0, nullptr);
    } else {
        for (std::size_t i = 0; i < s.queries.size(); ++i) {
            row(i, &s.queries[i]);
        }
    }
    return out;
}

} // namespace

log_format log_format_from_path(const std::filesystem::path &path) {
    std::string ext = path.extension().string();
    for (char &c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (ext == kJsonLinesExt || ext == kJsonLinesAltExt) {
        return log_format::json_lines;
    }
    if (ext == kCsvExt) {
        return log_format::csv;
    }
    throw std::invalid_argument("unsupported progress log extension '" + ext + "' (expected .jsonl, .ndjson, or .csv): " + path.string());
}

const char *log_format_name(const log_format format) noexcept { return format == log_format::csv ? "csv" : "jsonl"; }

std::size_t log_csv_column_count() noexcept { return kCsvColumns; }

double relative_half_width(const double estimate, const double ci_low, const double ci_high) {
    if (!std::isfinite(estimate) || estimate == 0.0 || !std::isfinite(ci_low) || !std::isfinite(ci_high)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (ci_high - ci_low) / (2.0 * estimate);
}

snapshot final_snapshot() {
    snapshot s;
    s.stage = phase::done;
    return s;
}

log_sink::log_sink(const std::filesystem::path &path) : log_sink(path, log_format_from_path(path)) {}

log_sink::log_sink(std::filesystem::path path, const log_format format) : path_(std::move(path)), format_(format) {
    out_.open(path_, std::ios::binary | std::ios::trunc);
    if (!out_) {
        throw std::runtime_error("cannot write progress log " + path_.string());
    }
    if (format_ == log_format::csv) {
        out_.write(kCsvHeader.data(), static_cast<std::streamsize>(kCsvHeader.size()));
        out_.put('\n');
        out_.flush();
        if (!out_) {
            failed_ = true;
            throw std::runtime_error("cannot write progress log " + path_.string());
        }
    }
}

void log_sink::on_snapshot(const snapshot &s) { write(s, /*finish=*/false, /*interrupted=*/false); }

void log_sink::on_finish(const snapshot &last, const bool interrupted) {
    if (finished_) {
        return;
    }
    finished_ = true;
    write(last, /*finish=*/true, interrupted);
}

void log_sink::write(const snapshot &s, const bool finish, const bool interrupted) {
    // `on_snapshot` may follow `on_finish` when the emitter broadcasts the last
    // snapshot; the finish record must stay last.
    if (failed_ || (finished_ && !finish)) {
        return;
    }
    const std::string record = format_ == log_format::csv ? csv_records(s, records_, finish, interrupted) : json_record(s, records_, finish, interrupted);
    ++records_;
    out_.write(record.data(), static_cast<std::streamsize>(record.size()));
    out_.flush();
    if (!out_) {
        failed_ = true;
    }
}

} // namespace fuzzybits::runtime::progress
