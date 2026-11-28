#include <catch2/catch_test_macros.hpp>

#include "io/charconv.h"

#include <array>
#include <bit>
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if FUZZYBITS_IO_HAS_FROM_CHARS_FALLBACK

namespace {

using fuzzybits::io::detail::from_chars_fallback;

struct parsed {
    std::ptrdiff_t consumed = 0;
    std::errc error{};
    double value = 0.0;
};

constexpr double kUnset = -12345.678;

auto parse_fallback(const std::string_view text) -> parsed {
    double value = kUnset;
    const auto [end, error] = from_chars_fallback(text.data(), text.data() + text.size(), value);
    return {end - text.data(), error, value};
}

// Inputs whose std::from_chars result is fully specified (no gradual underflow).
const std::vector<std::string_view> kSpecified = {
    "0",
    "-0",
    "1",
    "-1",
    "1.5",
    ".5",
    "5.",
    "-.5",
    "0.1",
    "1e10",
    "1E-5",
    "1e+5",
    "007",
    "123456789012345678901234567890",
    "0.30000000000000004",
    "2.2250738585072014e-308",
    "1.7976931348623157e308",
    "9007199254740993",
    "inf",
    "-INF",
    "Infinity",
    "-infinity",
    "nan",
    "-NaN",
    "nan(abc_1)",
    "nan(",
    "nan(x",
    "infinit",
    "1e",
    "1e+",
    "1e-x",
    "1.5x",
    "1..5",
    "0x1p3",
    "",
    "-",
    ".",
    "+1",
    " 1",
    "e5",
    "x",
    "1e400",
    "-1e400",
    "1e309",
};

} // namespace

TEST_CASE("from_chars fallback follows the documented contract", "[io][charconv]") {
    CHECK(parse_fallback("1.5").value == 1.5);
    CHECK(parse_fallback(".5").value == 0.5);
    CHECK(parse_fallback("5.").value == 5.0);
    CHECK(parse_fallback("-0").value == 0.0);
    CHECK(std::signbit(parse_fallback("-0").value));

    for (const std::string_view invalid : {"", "-", ".", "+1", " 1", "e5", "x", "-.", "-e1"}) {
        const auto result = parse_fallback(invalid);
        CHECK(result.error == std::errc::invalid_argument);
        CHECK(result.consumed == 0);
        CHECK(result.value == kUnset);
    }

    // Longest prefix: trailing text is left for the caller.
    CHECK(parse_fallback("1e").consumed == 1);
    CHECK(parse_fallback("1e+").consumed == 1);
    CHECK(parse_fallback("1.5x").consumed == 3);
    CHECK(parse_fallback("1..5").consumed == 2);
    CHECK(parse_fallback("0x1p3").consumed == 1);
    CHECK(parse_fallback("infinit").consumed == 3);
    CHECK(parse_fallback("nan(").consumed == 3);
    CHECK(parse_fallback("nan(abc_1)").consumed == 10);

    for (const std::string_view huge : {"1e400", "-1e400", "1e309"}) {
        const auto result = parse_fallback(huge);
        CHECK(result.error == std::errc::result_out_of_range);
        CHECK(result.consumed == static_cast<std::ptrdiff_t>(huge.size()));
        CHECK(result.value == kUnset);
    }
    {
        const auto result = parse_fallback("1e-400");
        CHECK(result.error == std::errc::result_out_of_range);
        CHECK(result.consumed == 6);
        CHECK(result.value == kUnset);
    }
    {
        const auto result = parse_fallback("4.9406564584124654e-324"); // smallest subnormal
        CHECK(result.error == std::errc{});
        CHECK(result.value == std::nextafter(0.0, 1.0));
    }
}

TEST_CASE("from_chars fallback ignores the global locale", "[io][charconv]") {
    const std::string previous = std::setlocale(LC_NUMERIC, nullptr);
    bool switched = false;
    for (const char *name : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8"}) {
        if (std::setlocale(LC_NUMERIC, name) != nullptr) {
            switched = true;
            break;
        }
    }
    const auto result = parse_fallback("1.5");
    std::setlocale(LC_NUMERIC, previous.c_str());
    if (!switched) {
        SKIP("no locale with a ',' decimal separator is installed");
    }
    CHECK(result.error == std::errc{});
    CHECK(result.consumed == 3);
    CHECK(result.value == 1.5);
}

#if FUZZYBITS_IO_HAS_STD_FROM_CHARS_DOUBLE
namespace {

auto same_double(const double a, const double b) -> bool {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b) && std::signbit(a) == std::signbit(b);
    }
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

} // namespace

TEST_CASE("from_chars fallback matches std::from_chars", "[io][charconv]") {
    const auto compare = [](const std::string_view text) {
        double expected_value = kUnset;
        const auto expected = std::from_chars(text.data(), text.data() + text.size(), expected_value);
        const auto actual = parse_fallback(text);
        INFO("input: '" << std::string(text) << "'");
        CHECK(actual.consumed == expected.ptr - text.data());
        CHECK(actual.error == expected.ec);
        // On error the standard leaves value unchanged; libc++ stores +-inf on overflow instead.
        if (expected.ec == std::errc{}) {
            CHECK(same_double(actual.value, expected_value));
        }
    };
    for (const auto text : kSpecified) {
        compare(text);
    }

    // Shortest round-trip text of random finite doubles, plus 17 significant digits.
    std::mt19937_64 rng(0x5eedf00dULL);
    std::array<char, 64> buffer{};
    for (int i = 0; i < 20000; ++i) {
        const double value = std::bit_cast<double>(rng());
        if (!std::isfinite(value) || (value != 0.0 && std::fabs(value) < 2.2250738585072014e-308)) {
            continue; // subnormals: implementations may differ on underflow
        }
        const auto shortest = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        compare(std::string_view(buffer.data(), static_cast<std::size_t>(shortest.ptr - buffer.data())));
        const auto fixed = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::scientific, 16);
        compare(std::string_view(buffer.data(), static_cast<std::size_t>(fixed.ptr - buffer.data())));
    }
}
#endif

#endif // FUZZYBITS_IO_HAS_FROM_CHARS_FALLBACK

TEST_CASE("fuzzybits::io::from_chars parses doubles", "[io][charconv]") {
    const std::string_view text = "0.125e1";
    double value = 0.0;
    const auto [end, error] = fuzzybits::io::from_chars(text.data(), text.data() + text.size(), value);
    CHECK(error == std::errc{});
    CHECK(end == text.data() + text.size());
    CHECK(value == 1.25);
}
