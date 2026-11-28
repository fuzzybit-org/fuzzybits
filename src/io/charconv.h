#pragma once

// Floating-point std::from_chars that also builds where the standard library lacks it.
//
// Apple libc++ ships the double overload only from macOS 26 (it is "unavailable: introduced in
// macOS 26.0" when MACOSX_DEPLOYMENT_TARGET is lower, and missing entirely before LLVM 20's
// libc++, e.g. Xcode 16.4). The Intel macOS wheel targets macOS 15.0, so it needs the fallback.
// libstdc++ (Linux) and the MSVC STL (Windows) always take the std::from_chars branch.
//
// fuzzybits::io::from_chars(first, last, value) has the std::from_chars contract for
// std::chars_format::general:
//   - parses the longest prefix of [first, last) that matches the pattern
//     -? (digits [. digits?] | . digits) ([eE] [+-]? digits)?, or -? inf | infinity | nan | nan(chars)
//     (case-insensitive), with no leading whitespace, no '+' and no hex;
//   - returns {first, invalid_argument} and leaves value unchanged when nothing matches;
//   - returns {end of match, result_out_of_range} and leaves value unchanged when the number
//     overflows, or is non-zero but rounds to zero;
//   - is locale-independent.
//
// detail::from_chars_fallback implements that contract with strtod_l in the "C" locale, which is
// correctly rounded on macOS and glibc. It is built on every POSIX target (not Windows, which has
// no strtod_l) so tests can compare it with std::from_chars where both exist. It reports a
// subnormal result as success; the standard leaves implementations room to differ on gradual
// underflow.

#include <charconv>
#include <system_error>

#if defined(_LIBCPP_VERSION)
#if _LIBCPP_VERSION < 200000
#define FUZZYBITS_IO_STD_FROM_CHARS_DOUBLE 0
#elif defined(_LIBCPP_AVAILABILITY_HAS_FROM_CHARS_FLOATING_POINT)
#define FUZZYBITS_IO_STD_FROM_CHARS_DOUBLE _LIBCPP_AVAILABILITY_HAS_FROM_CHARS_FLOATING_POINT
#else
#define FUZZYBITS_IO_STD_FROM_CHARS_DOUBLE 1
#endif
#else
#define FUZZYBITS_IO_STD_FROM_CHARS_DOUBLE 1
#endif

#if !defined(_WIN32)
#define FUZZYBITS_IO_HAS_FROM_CHARS_FALLBACK 1
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <locale.h>
#include <string>
#if defined(__APPLE__)
#include <xlocale.h>
#endif
#else
#define FUZZYBITS_IO_HAS_FROM_CHARS_FALLBACK 0
#endif

#if !FUZZYBITS_IO_STD_FROM_CHARS_DOUBLE && !FUZZYBITS_IO_HAS_FROM_CHARS_FALLBACK
#error "fuzzybits::io::from_chars: no floating-point std::from_chars and no strtod_l"
#endif

namespace fuzzybits::io {

#if FUZZYBITS_IO_HAS_FROM_CHARS_FALLBACK
namespace detail {

[[nodiscard]] constexpr auto lower(const char c) noexcept -> char { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

[[nodiscard]] constexpr auto is_digit(const char c) noexcept -> bool { return c >= '0' && c <= '9'; }

// Case-insensitive match of the lower-case `word` at `p`; returns the end of the match or nullptr.
[[nodiscard]] constexpr auto match_word(const char *p, const char *last, const char *word) noexcept -> const char * {
    for (; *word != '\0'; ++word, ++p) {
        if (p == last || lower(*p) != *word) {
            return nullptr;
        }
    }
    return p;
}

// End of the longest prefix of [first, last) that std::from_chars (general) would parse, or first.
[[nodiscard]] constexpr auto scan_general(const char *first, const char *last) noexcept -> const char * {
    const char *p = first;
    if (p != last && *p == '-') {
        ++p;
    }
    if (const char *inf = match_word(p, last, "inf")) {
        const char *infinity = match_word(p, last, "infinity");
        return infinity ? infinity : inf;
    }
    if (const char *nan = match_word(p, last, "nan")) {
        if (nan != last && *nan == '(') {
            const char *q = nan + 1;
            while (q != last && (is_digit(*q) || (lower(*q) >= 'a' && lower(*q) <= 'z') || *q == '_')) {
                ++q;
            }
            if (q != last && *q == ')') {
                return q + 1;
            }
        }
        return nan;
    }
    bool digits = false;
    while (p != last && is_digit(*p)) {
        ++p;
        digits = true;
    }
    if (p != last && *p == '.') {
        ++p;
        while (p != last && is_digit(*p)) {
            ++p;
            digits = true;
        }
    }
    if (!digits) {
        return first;
    }
    if (p != last && (*p == 'e' || *p == 'E')) {
        const char *q = p + 1;
        if (q != last && (*q == '+' || *q == '-')) {
            ++q;
        }
        if (q != last && is_digit(*q)) {
            while (q != last && is_digit(*q)) {
                ++q;
            }
            p = q;
        }
    }
    return p;
}

[[nodiscard]] inline auto c_locale() noexcept -> locale_t {
    static const locale_t locale = newlocale(LC_ALL_MASK, "C", static_cast<locale_t>(nullptr));
    return locale;
}

inline auto from_chars_fallback(const char *first, const char *last, double &value) -> std::from_chars_result {
    const char *end = scan_general(first, last);
    if (end == first) {
        return {first, std::errc::invalid_argument};
    }
    const std::string text(first, end); // strtod_l needs a terminated string
    char *parsed = nullptr;
    errno = 0;
    const double result = strtod_l(text.c_str(), &parsed, c_locale());
    const int error = errno;
    if (parsed != text.c_str() + text.size()) {
        return {first, std::errc::invalid_argument}; // unreachable: the scan matches strtod's grammar
    }
    if (error == ERANGE && (std::isinf(result) || result == 0.0)) {
        return {end, std::errc::result_out_of_range};
    }
    value = result;
    return {end, std::errc{}};
}

} // namespace detail
#endif

#if FUZZYBITS_IO_STD_FROM_CHARS_DOUBLE
inline auto from_chars(const char *first, const char *last, double &value) noexcept -> std::from_chars_result { return std::from_chars(first, last, value); }
#else
inline auto from_chars(const char *first, const char *last, double &value) -> std::from_chars_result { return detail::from_chars_fallback(first, last, value); }
#endif

} // namespace fuzzybits::io

// Tests use this to know whether std::from_chars(double) exists to compare against.
#define FUZZYBITS_IO_HAS_STD_FROM_CHARS_DOUBLE FUZZYBITS_IO_STD_FROM_CHARS_DOUBLE
