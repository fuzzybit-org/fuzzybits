#pragma once

// Initiating-event frequencies (#243): the units an FTP compile accepts, the durations of
// --exposure, and the probability of at least one occurrence over an exposure period. Header-only
// and independent of pcfb; io/pcfb/query_groups.h reads a file's groups.

#include <array>
#include <cmath>
#include <cstdint>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

namespace fuzzybits::io::pcfb_frequency {

// Average Gregorian year: 365.25 days. A month is a twelfth of it.
inline constexpr double hours_per_year = 365.25 * 24.0;

struct unit_spec final {
    std::string_view unit;     // initiator unit, e.g. "per-year"
    std::string_view duration; // the matching --exposure suffix, e.g. "y"
    std::string_view label;    // for results, e.g. "per year"
    double hours = 1.0;        // length of one period
};

inline constexpr std::array<unit_spec, 5> units{{
    {"per-year", "y", "per year", hours_per_year},
    {"per-month", "mo", "per month", hours_per_year / 12.0},
    {"per-week", "wk", "per week", 7.0 * 24.0},
    {"per-day", "d", "per day", 24.0},
    {"per-hour", "h", "per hour", 1.0},
}};

// The FTP driver's unit when none is given: PRA initiating-event frequencies are conventionally per year.
inline constexpr std::string_view default_unit = units[0].unit;

// "per-year|per-month|per-week|per-day|per-hour", for messages and help.
[[nodiscard]] inline auto unit_names() -> std::string {
    std::string out;
    for (auto const &unit : units) {
        out += (out.empty() ? "" : "|") + std::string(unit.unit);
    }
    return out;
}

[[nodiscard]] inline auto find_unit(const std::string_view unit) -> std::optional<unit_spec> {
    for (auto const &candidate : units) {
        if (candidate.unit == unit) {
            return candidate;
        }
    }
    return std::nullopt;
}

// "<positive decimal><y|mo|wk|d|h>", e.g. "1y", "0.5mo", "720h": the duration in hours.
[[nodiscard]] inline auto parse_duration_hours(const std::string_view text) -> std::optional<double> {
    for (auto const &unit : units) {
        if (text.size() <= unit.duration.size() || !text.ends_with(unit.duration)) {
            continue;
        }
        const auto number = std::string(text.substr(0, text.size() - unit.duration.size()));
        std::istringstream in(number);
        in.imbue(std::locale::classic());
        double value = 0.0;
        in >> std::noskipws >> value;
        if (in.fail() || !in.eof() || !std::isfinite(value) || value <= 0.0) {
            return std::nullopt;
        }
        return value * unit.hours;
    }
    return std::nullopt;
}

// P(at least one occurrence in the period) = 1 - exp(-F * tau), occurrences a Poisson process at
// rate F in `unit`. Unset when the unit is unknown.
[[nodiscard]] inline auto exposure_probability(const double frequency, const std::string_view unit, const double hours) -> std::optional<double> {
    const auto spec = find_unit(unit);
    if (!spec) {
        return std::nullopt;
    }
    return -std::expm1(-frequency * hours / spec->hours);
}

} // namespace fuzzybits::io::pcfb_frequency
