#include "cli/progress_line.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <csignal>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace fuzzybits::runtime::progress {
namespace {

// ---------------------------------------------------------------------------------------------
// Small string helpers.
// ---------------------------------------------------------------------------------------------

char ascii_lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c; }

bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (ascii_lower(a[i]) != ascii_lower(b[i])) {
            return false;
        }
    }
    return true;
}

bool is_ascii(std::string_view text) {
    for (const char c : text) {
        if (static_cast<unsigned char>(c) >= 0x80) {
            return false;
        }
    }
    return true;
}

/// 1 = on, 0 = off, -1 = unset or not a recognised value.
int env_flag(const char *value) {
    if (value == nullptr) {
        return -1;
    }
    const std::string_view text(value);
    if (text == "0" || iequals(text, "false") || iequals(text, "no") || iequals(text, "off")) {
        return 0;
    }
    if (text == "1" || iequals(text, "true") || iequals(text, "yes") || iequals(text, "on")) {
        return 1;
    }
    return -1;
}

bool locale_is_utf8(const char *lc_all, const char *lc_ctype, const char *lang) {
    const char *value = lc_all;
    if (value == nullptr || *value == '\0') {
        value = lc_ctype;
    }
    if (value == nullptr || *value == '\0') {
        value = lang;
    }
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::string_view text(value);
    for (const std::string_view code : {"UTF-8", "utf8", "UTF8", "utf-8"}) {
        if (text.find(code) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// Width: a wcwidth-style classifier over decoded code points.
// ---------------------------------------------------------------------------------------------

struct code_point {
    char32_t value = 0;
    std::size_t bytes = 1;
    bool valid = false;
};

code_point decode(std::string_view text, std::size_t offset) {
    const auto first = static_cast<unsigned char>(text[offset]);
    code_point cp;
    if (first < 0x80) {
        cp.value = first;
        cp.bytes = 1;
        cp.valid = true;
        return cp;
    }
    std::size_t extra = 0;
    char32_t value = 0;
    if ((first & 0xE0) == 0xC0) {
        extra = 1;
        value = first & 0x1Fu;
    } else if ((first & 0xF0) == 0xE0) {
        extra = 2;
        value = first & 0x0Fu;
    } else if ((first & 0xF8) == 0xF0) {
        extra = 3;
        value = first & 0x07u;
    } else {
        return cp;
    }
    if (offset + extra >= text.size()) {
        return cp;
    }
    for (std::size_t i = 1; i <= extra; ++i) {
        const auto byte = static_cast<unsigned char>(text[offset + i]);
        if ((byte & 0xC0) != 0x80) {
            return cp;
        }
        value = (value << 6) | (byte & 0x3Fu);
    }
    cp.value = value;
    cp.bytes = extra + 1;
    cp.valid = true;
    return cp;
}

bool in_range(char32_t cp, char32_t low, char32_t high) { return cp >= low && cp <= high; }

/// Zero cells: combining marks, joiners, variation selectors, Hangul medial/final jamo.
bool is_zero_width(char32_t cp) {
    static constexpr std::array<std::pair<char32_t, char32_t>, 14> kRanges{{
        {0x0300, 0x036F},
        {0x0483, 0x0489},
        {0x0591, 0x05BD},
        {0x0610, 0x061A},
        {0x064B, 0x065F},
        {0x0670, 0x0670},
        {0x06D6, 0x06DC},
        {0x0E31, 0x0E3A},
        {0x1160, 0x11FF},
        {0x1AB0, 0x1AFF},
        {0x1DC0, 0x1DFF},
        {0x200B, 0x200F},
        {0xFE00, 0xFE0F},
        {0xFE20, 0xFE2F},
    }};
    for (const auto &[low, high] : kRanges) {
        if (in_range(cp, low, high)) {
            return true;
        }
    }
    return in_range(cp, 0xE0100, 0xE01EF);
}

/// Two cells: East Asian Wide/Fullwidth and emoji presentation.
bool is_wide(char32_t cp) {
    static constexpr std::array<std::pair<char32_t, char32_t>, 16> kRanges{{
        {0x1100, 0x115F},
        {0x2E80, 0x303E},
        {0x3041, 0x33FF},
        {0x3400, 0x4DBF},
        {0x4E00, 0x9FFF},
        {0xA000, 0xA4CF},
        {0xA960, 0xA97F},
        {0xAC00, 0xD7A3},
        {0xF900, 0xFAFF},
        {0xFE10, 0xFE19},
        {0xFE30, 0xFE6F},
        {0xFF00, 0xFF60},
        {0xFFE0, 0xFFE6},
        {0x16FE0, 0x16FFF},
        {0x1F000, 0x1FAFF},
        {0x20000, 0x3FFFD},
    }};
    for (const auto &[low, high] : kRanges) {
        if (in_range(cp, low, high)) {
            return true;
        }
    }
    return false;
}

std::size_t code_point_width(char32_t cp) {
    if (cp == 0 || is_zero_width(cp)) {
        return 0;
    }
    return is_wide(cp) ? 2 : 1;
}

// ---------------------------------------------------------------------------------------------
// Values. Every formatter is total: NaN, infinity and 0 have defined text.
// ---------------------------------------------------------------------------------------------

/// `7.452e-02`, always a signed two-digit exponent (portable across libcs).
std::string fmt_estimate(double v) {
    if (!std::isfinite(v)) {
        return "n/a";
    }
    if (v == 0.0) {
        return "0.000e+00";
    }
    int exponent = static_cast<int>(std::floor(std::log10(std::fabs(v))));
    double mantissa = v / std::pow(10.0, exponent);
    if (std::fabs(mantissa) >= 10.0) {
        mantissa /= 10.0;
        ++exponent;
    } else if (std::fabs(mantissa) < 1.0) {
        mantissa *= 10.0;
        --exponent;
    }
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.3fe%s%02d", mantissa, exponent < 0 ? "-" : "+", std::abs(exponent));
    return std::string(buffer.data());
}

/// `0.19%`: two significant digits, so narrow and wide targets read comparably.
std::string fmt_rel_half_width(double rel) {
    if (!std::isfinite(rel) || rel < 0.0) {
        return "n/a";
    }
    const double percent = rel * 100.0;
    if (percent >= 100.0) {
        return ">99%";
    }
    if (percent < 0.01) {
        return "<0.01%";
    }
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.2g%%", percent);
    return std::string(buffer.data());
}

/// `1.6e9`, `12582912`.
std::string fmt_count(std::uint64_t v) {
    if (v < 100000) {
        return std::to_string(v);
    }
    int exponent = 0;
    double mantissa = static_cast<double>(v);
    while (mantissa >= 10.0) {
        mantissa /= 10.0;
        ++exponent;
    }
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.1f", mantissa);
    std::string text(buffer.data());
    if (text.size() > 2 && text.compare(text.size() - 2, 2, ".0") == 0) {
        text.erase(text.size() - 2);
    }
    return text + "e" + std::to_string(exponent);
}

/// `3.2 Gtrial/s`; empty when the rate is not known yet.
std::string fmt_rate(double trials_per_second) {
    if (!std::isfinite(trials_per_second) || trials_per_second <= 0.0) {
        return {};
    }
    static constexpr std::array<const char *, 5> kPrefix{"", "k", "M", "G", "T"};
    std::size_t index = 0;
    double scaled = trials_per_second;
    while (scaled >= 1000.0 && index + 1 < kPrefix.size()) {
        scaled /= 1000.0;
        ++index;
    }
    std::array<char, 32> buffer{};
    if (index == 0) {
        std::snprintf(buffer.data(), buffer.size(), "%.0f trial/s", scaled);
    } else {
        std::snprintf(buffer.data(), buffer.size(), "%.1f %strial/s", scaled, kPrefix[index]);
    }
    return std::string(buffer.data());
}

/// `0:05`, `1:02:03`.
std::string fmt_clock(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return {};
    }
    const auto total = static_cast<long long>(std::llround(seconds));
    const long long hours = total / 3600;
    const long long minutes = (total % 3600) / 60;
    const long long secs = total % 60;
    std::array<char, 32> buffer{};
    if (hours > 0) {
        std::snprintf(buffer.data(), buffer.size(), "%lld:%02lld:%02lld", hours, minutes, secs);
    } else {
        std::snprintf(buffer.data(), buffer.size(), "%lld:%02lld", minutes, secs);
    }
    return std::string(buffer.data());
}

/// `3.2 s` below a minute, `0:08` above.
std::string fmt_stage_time(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return {};
    }
    if (seconds < 60.0) {
        std::array<char, 32> buffer{};
        std::snprintf(buffer.data(), buffer.size(), "%.1f s", seconds);
        return std::string(buffer.data());
    }
    return fmt_clock(seconds);
}

struct marks {
    std::string_view fill;
    std::string_view track;
    std::string_view ellipsis;
    std::string_view converged;
    std::string_view plus_minus;
    std::string_view arrow;
    std::string_view slowest;
    std::string_view slowest_short;
};

const marks &marks_for(glyphs g) {
    static constexpr marks kAscii{"#", "-", "...", "ok", "+/-", "->", "<- slowest", "<-"};
    static constexpr marks kUtf8{"\u2501", "\u2500", "\u2026",         "\u2713",
                                 "\u00b1", "\u2192", "\u2190 slowest", "\u2190"};
    return g == glyphs::ascii ? kAscii : kUtf8;
}

/// Append `segment`, two spaces from whatever is already there.
void join(std::string &out, std::string_view segment) {
    if (segment.empty()) {
        return;
    }
    if (!out.empty()) {
        out += "  ";
    }
    out += segment;
}

/// `text` padded or cut to exactly `width` cells.
std::string fit_name(std::string_view text, std::size_t width, glyphs g) {
    if (display_width(text) > width) {
        return truncate_to_width(text, width, g);
    }
    std::string out(text);
    out.append(width - display_width(text), ' ');
    return out;
}

std::string fmt_percent(const snapshot &s) {
    if (s.trials_total == 0) {
        return {};
    }
    const double fraction =
        std::clamp(static_cast<double>(s.trials_done) / static_cast<double>(s.trials_total), 0.0, 1.0);
    std::array<char, 16> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.0f%%", fraction * 100.0);
    return std::string(buffer.data());
}

std::string bar(double fraction, std::size_t cells, glyphs g) {
    const marks &m = marks_for(g);
    const auto filled =
        static_cast<std::size_t>(std::llround(std::clamp(fraction, 0.0, 1.0) * static_cast<double>(cells)));
    std::string out;
    out.reserve(cells * 3);
    for (std::size_t i = 0; i < cells; ++i) {
        out += i < filled ? m.fill : m.track;
    }
    return out;
}

std::string phase_label(phase p) {
    switch (p) {
    case phase::compile:
        return "compiling";
    case phase::sampling:
        return "sampling";
    case phase::checking:
        return "checking";
    case phase::done:
        return "done";
    }
    return "sampling";
}

/// The least-known target: largest relative half width. An unknown (NaN) width counts as the
/// worst, so a target that has not measured anything yet is always listed.
std::size_t slowest_index(const std::vector<query_state> &queries) {
    std::size_t best = 0;
    double best_width = -1.0;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const double width = std::isfinite(queries[i].rel_half_width) ? queries[i].rel_half_width
                                                                      : std::numeric_limits<double>::infinity();
        if (width > best_width) {
            best_width = width;
            best = i;
        }
    }
    return best;
}

/// Which targets to list and whether a `+N more` row follows. The list is at most `max_rows`
/// rows *including* that summary row: query order, the slowest target always present.
struct row_plan {
    std::vector<std::size_t> rows;
    bool more = false;
};

row_plan pick_rows(const snapshot &s, std::size_t max_rows) {
    const std::size_t n = s.queries.size();
    row_plan plan;
    if (n == 0 || max_rows == 0) {
        return plan;
    }
    if (n <= max_rows) {
        for (std::size_t i = 0; i < n; ++i) {
            plan.rows.push_back(i);
        }
        return plan;
    }
    const std::size_t slowest = slowest_index(s.queries);
    if (max_rows == 1) {
        // No room for the summary row: the slowest target is the one that matters.
        plan.rows.push_back(slowest);
        return plan;
    }
    // One of `max_rows` is the `+N more` summary, so at most `max_rows - 1` targets are listed.
    const std::size_t target_rows = max_rows - 1;
    for (std::size_t i = 0; i < n && plan.rows.size() + 1 < target_rows; ++i) {
        if (i != slowest) {
            plan.rows.push_back(i);
        }
    }
    plan.rows.push_back(slowest);
    std::sort(plan.rows.begin(), plan.rows.end());
    plan.more = plan.rows.size() < n;
    return plan;
}

/// `    NAME  estimate ±width  [check]  [marker]`. The marker is the last thing added, so it is
/// also the first thing given up when the terminal is narrow; the estimate and its precision,
/// which are what the row is for, are never traded away for it.
std::string target_row(const query_state &q, std::size_t name_width, bool slowest, const line_style &style,
                       std::size_t width) {
    const marks &m = marks_for(style.glyph_set);
    std::string row = "    ";
    row += fit_name(q.name, name_width, style.glyph_set);
    row += "  ";
    row += fmt_estimate(q.estimate);
    row += " ";
    row += m.plus_minus;
    row += fmt_rel_half_width(q.rel_half_width);
    if (q.converged) {
        row += "  ";
        row += m.converged;
    }
    if (slowest) {
        row += "  ";
        if (display_width(row) + display_width(m.slowest) <= width) {
            row += m.slowest; // "<<- slowest" / "← slowest"
        } else {
            row += m.slowest_short;
        }
    }
    return row;
}

/// `parse -> [lower] -> passes -> write`, bracketing the current stage.
std::string compile_chain(const snapshot &s, const line_style &style) {
    static constexpr std::array<std::string_view, 4> kStages{"parse", "lower", "passes", "write"};
    const marks &m = marks_for(style.glyph_set);
    std::string chain;
    bool matched = false;
    for (const std::string_view stage : kStages) {
        if (!chain.empty()) {
            chain += " ";
            chain += m.arrow;
            chain += " ";
        }
        const bool current = !s.step.empty() && s.step == stage;
        matched = matched || current;
        if (current) {
            chain += "[";
            chain += stage;
            chain += "]";
        } else {
            chain += stage;
        }
    }
    if (!s.step.empty() && !matched) {
        chain += " ";
        chain += m.arrow;
        chain += " [";
        chain += s.step;
        chain += "]";
    }
    return chain;
}

std::string compile_line(const snapshot &s, std::size_t width, const line_style &style) {
    std::string line = "  compiling  " + compile_chain(s, style);
    const std::string elapsed = fmt_stage_time(s.elapsed_seconds);
    if (!elapsed.empty()) {
        line += "  " + elapsed;
    }
    if (s.passes_total > 0) {
        line += "  C" + std::to_string(s.passes_done) + "/" + std::to_string(s.passes_total);
    }
    return truncate_to_width(line, width, style.glyph_set);
}

std::string shared_line(const snapshot &s, std::size_t width, const line_style &style) {
    const marks &m = marks_for(style.glyph_set);
    const std::string percent = fmt_percent(s);
    const std::string rate = fmt_rate(s.trials_per_second);

    std::string counts;
    if (s.trials_total > 0) {
        counts = fmt_count(s.trials_done) + "/" + fmt_count(s.trials_total) + " trials";
    } else if (s.trials_done > 0) {
        counts = fmt_count(s.trials_done) + " trials";
    } else if (s.passes_done > 0) {
        counts = std::to_string(s.passes_done) + (s.passes_done == 1 ? " pass" : " passes");
    }

    std::string eta;
    if (!percent.empty() && s.trials_per_second > 0.0 && s.trials_total > s.trials_done) {
        const double remaining = static_cast<double>(s.trials_total - s.trials_done) / s.trials_per_second;
        const std::string clock = fmt_clock(remaining);
        if (!clock.empty()) {
            eta = "ETA " + clock;
        }
    }

    // With a single target the estimate and its precision fold into the shared line.
    std::string fold;
    if (s.queries.size() == 1) {
        fold = "TOP " + fmt_estimate(s.queries.front().estimate) + " " + std::string(m.plus_minus) +
               fmt_rel_half_width(s.queries.front().rel_half_width);
    }

    const std::string label = phase_label(s.stage);
    const double fraction =
        s.trials_total > 0 ? static_cast<double>(s.trials_done) / static_cast<double>(s.trials_total) : 0.0;

    auto compose = [&](std::size_t bar_cells, bool with_counts, bool with_fold, bool with_rate, bool with_eta) {
        std::string line = "  " + label;
        if (!percent.empty()) {
            if (bar_cells > 0) {
                line += " " + bar(fraction, bar_cells, style.glyph_set);
            }
            line += " " + percent;
        }
        if (with_counts) {
            join(line, counts);
        }
        if (with_fold) {
            join(line, fold);
        }
        if (with_rate) {
            join(line, rate);
        }
        if (with_eta) {
            join(line, eta);
        }
        return line;
    };

    // Preference ladder. The bar shrinks before anything is dropped, and content goes in inverse
    // order of usefulness: the raw trial counts are the first to go because the bar and the
    // percentage already say how far the run is, while the rate and the ETA are the only place that
    // timing is shown. The estimate survives longest because it is why anyone watches the run; a
    // narrow terminal keeps whatever fits rather than going blank.
    struct keep {
        bool counts : 1;
        bool fold : 1;
        bool rate : 1;
        bool eta : 1;
    };
    static constexpr std::array<keep, 6> kLadder{{
        {true, true, true, true},    // everything
        {false, true, true, true},   // counts yield first
        {false, true, false, true},  // then the rate
        {false, true, false, false}, // then the ETA
        {true, false, false, false}, // very narrow: counts instead of the estimate
        {false, false, false, false},
    }};
    constexpr std::size_t kMaxBar = 24;
    constexpr std::size_t kMinBar = 4;
    for (const keep &k : kLadder) {
        for (std::size_t cells = kMaxBar; cells >= kMinBar; cells -= 2) {
            const std::string line = compose(cells, k.counts, k.fold, k.rate, k.eta);
            if (display_width(line) <= width) {
                return line;
            }
        }
    }
    return truncate_to_width(compose(0, false, false, false, false), width, style.glyph_set);
}

std::string summary_targets(const snapshot &s) {
    if (s.queries.empty()) {
        return {};
    }
    const query_state &top = s.queries.front();
    std::string text = "TOP " + fmt_estimate(top.estimate) + " " + std::string(marks_for(glyphs::utf8).plus_minus) +
                       fmt_rel_half_width(top.rel_half_width);
    if (s.queries.size() > 1) {
        text += "  +" + std::to_string(s.queries.size() - 1) + " more";
    }
    return text;
}

// ---------------------------------------------------------------------------------------------
// Terminal plumbing.
// ---------------------------------------------------------------------------------------------

void raw_write(int fd, std::string_view bytes) {
    std::size_t written = 0;
    while (written < bytes.size()) {
#if defined(_WIN32)
        const int chunk = _write(fd, bytes.data() + written, static_cast<unsigned int>(bytes.size() - written));
#else
        const ssize_t chunk = ::write(fd, bytes.data() + written, bytes.size() - written);
#endif
        if (chunk <= 0) {
            return; // the terminal went away; nothing useful is left to do
        }
        written += static_cast<std::size_t>(chunk);
    }
}

std::size_t query_terminal_width(int fd) {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO info{};
    const HANDLE handle = GetStdHandle(fd == 2 ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE);
    if (handle != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(handle, &info)) {
        return static_cast<std::size_t>(info.srWindow.Right - info.srWindow.Left + 1);
    }
    return 0;
#else
    winsize size{};
    if (::ioctl(fd, TIOCGWINSZ, &size) == 0 && size.ws_col > 0) {
        return size.ws_col;
    }
    return 0;
#endif
}

/// Rows in the terminal. 0 when it cannot be measured; callers then keep their default.
std::size_t query_terminal_height(int fd) {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO info{};
    const HANDLE handle = GetStdHandle(fd == 2 ? STD_ERROR_HANDLE : STD_OUTPUT_HANDLE);
    if (handle != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(handle, &info)) {
        return static_cast<std::size_t>(info.srWindow.Bottom - info.srWindow.Top + 1);
    }
    return 0;
#else
    winsize size{};
    if (::ioctl(fd, TIOCGWINSZ, &size) == 0 && size.ws_row > 0) {
        return size.ws_row;
    }
    return 0;
#endif
}

bool fd_is_tty(int fd) {
#if defined(_WIN32)
    return ::_isatty(fd) != 0;
#else
    return ::isatty(fd) != 0;
#endif
}

constexpr std::string_view kHideCursor = "\x1b[?25l";
constexpr std::string_view kShowCursor = "\x1b[?25h";
constexpr std::string_view kEraseLine = "\x1b[2K";
constexpr std::string_view kCursorUp = "\x1b[1A";

#if !defined(_WIN32)
/// The block stays on screen and the cursor stays hidden if the process dies mid-run, so
/// SIGINT/SIGTERM restore the cursor before re-raising with the default action. Handlers are
/// installed only when the disposition is still SIG_DFL, so a host that handles its own signals
/// (CPython turns SIGINT into KeyboardInterrupt) keeps its behaviour; the destructor then runs on
/// the unwinding path and restores the cursor exactly once.
struct sigaction g_previous_int{};
struct sigaction g_previous_term{};
bool g_signals_installed = false;
/// Live channel, or nullptr when no sink owns the handlers. Assigned only by the thread that
/// installs/removes the handlers; the handler loads it once with acquire.
std::atomic<signal_channel *> g_signal_channel{nullptr};

/// One channel per sink, taken from a fixed pool that is never freed: a handler can still be
/// reading the channel while the sink it belongs to is destroyed, so nothing the handler touches
/// may be reclaimed. Exhausting the pool (more than four live sinks in one process) falls back to
/// the first channel, which only degrades the interrupt summary, never safety.
signal_channel &signal_channel_for(int fd) {
    static std::array<signal_channel, 4> channels{};
    static std::uint32_t next = 0;
    signal_channel &channel = channels[next < channels.size() ? next : 0];
    if (next < channels.size()) {
        ++next;
    }
    channel.fd = fd;
    channel.payload.withdraw();
    return channel;
}

void restore_cursor_and_reraise(int signo) {
    // errno belongs to the interrupted thread and the handler may run on any of them.
    const int saved_errno = errno;
    if (signal_channel *channel = g_signal_channel.load(std::memory_order_acquire)) {
        // One acquire load here as well, so the bytes and their length come from the same record.
        const std::string_view bytes = channel->payload.current();
        const int fd = channel->fd;
        if (fd >= 0 && !bytes.empty()) {
            const char *data = bytes.data();
            std::size_t left = bytes.size();
            while (left > 0) {
                const ssize_t wrote = ::write(fd, data, left);
                if (wrote <= 0) {
                    break;
                }
                data += wrote;
                left -= static_cast<std::size_t>(wrote);
            }
        }
    }
    // SA_RESETHAND already restored the default action; this is the belt-and-braces path for a
    // handler installed by something that ignored it.
    ::signal(signo, SIG_DFL);
    errno = saved_errno;
    ::raise(signo);
}

void handle_interrupt(int) { restore_cursor_and_reraise(SIGINT); }
void handle_terminate(int) { restore_cursor_and_reraise(SIGTERM); }

bool install_signal_handlers(int fd, signal_channel &channel) {
    struct sigaction current{};
    if (::sigaction(SIGINT, nullptr, &current) != 0 || current.sa_handler != SIG_DFL) {
        return false;
    }
    // Publish the channel before any handler can run, so a signal arriving right after
    // installation already finds the descriptor and the payload with the right ordering.
    channel.fd = fd;
    g_signal_channel.store(&channel, std::memory_order_release);
    struct sigaction action{};
    // `sa_mask` is already empty: the value-initialised struct is all zeroes, and `sigemptyset` is a
    // macro on Darwin that cannot be qualified. SA_RESETHAND: the handler restores the default
    // action itself, so a second Ctrl-C during the partial summary cannot re-enter it.
    action.sa_flags = SA_RESETHAND;
    action.sa_handler = &handle_interrupt;
    if (::sigaction(SIGINT, &action, &g_previous_int) != 0) {
        g_signal_channel.store(nullptr, std::memory_order_release);
        return false;
    }
    action.sa_handler = &handle_terminate;
    if (::sigaction(SIGTERM, &action, &g_previous_term) != 0) {
        ::sigaction(SIGINT, &g_previous_int, nullptr);
        g_signal_channel.store(nullptr, std::memory_order_release);
        return false;
    }
    g_signals_installed = true;
    return true;
}

void remove_signal_handlers() {
    if (!g_signals_installed) {
        return;
    }
    ::sigaction(SIGINT, &g_previous_int, nullptr);
    ::sigaction(SIGTERM, &g_previous_term, nullptr);
    g_signals_installed = false;
    // Withdraw before dropping the channel: a handler that already loaded it writes nothing more,
    // and the channel (with its payload records) stays alive for the life of the process.
    if (signal_channel *channel = g_signal_channel.exchange(nullptr, std::memory_order_acq_rel)) {
        channel->payload.withdraw();
    }
}
#endif

} // namespace

// ---------------------------------------------------------------------------------------------
// Public formatting.
// ---------------------------------------------------------------------------------------------

std::size_t display_width(std::string_view text) {
    std::size_t cells = 0;
    std::size_t offset = 0;
    while (offset < text.size()) {
        const code_point cp = decode(text, offset);
        if (!cp.valid) {
            ++cells; // a stray byte still occupies a cell
            ++offset;
            continue;
        }
        cells += code_point_width(cp.value);
        offset += cp.bytes;
    }
    return cells;
}

std::string truncate_to_width(std::string_view text, std::size_t width, glyphs g) {
    if (display_width(text) <= width && (g == glyphs::utf8 || is_ascii(text))) {
        return std::string(text);
    }
    if (width == 0) {
        return {};
    }
    const std::string_view ellipsis = marks_for(g).ellipsis;
    const std::size_t ellipsis_cells = display_width(ellipsis);
    // A budget smaller than the ellipsis still has to fit: fall back to the ellipsis' own head.
    const std::size_t budget = width > ellipsis_cells ? width - ellipsis_cells : 0;
    std::string out;
    std::size_t cells = 0;
    std::size_t offset = 0;
    while (offset < text.size()) {
        const code_point cp = decode(text, offset);
        const std::size_t cells_here = cp.valid ? code_point_width(cp.value) : 1;
        if (cells + cells_here > budget) {
            break;
        }
        if (!cp.valid) {
            out += '?';
            ++offset;
        } else if (g == glyphs::ascii && cp.value > 0x7F) {
            out += '?';
            offset += cp.bytes;
        } else {
            out.append(text.substr(offset, cp.bytes));
            offset += cp.bytes;
        }
        cells += cells_here;
    }
    out += ellipsis;
    if (display_width(out) > width) {
        // Only reachable for width < ellipsis width (a one- or two-cell terminal): return the head of
        // the ellipsis itself so the result still fits.
        std::string head;
        std::size_t cells_left = width;
        std::size_t offset2 = 0;
        while (offset2 < ellipsis.size() && cells_left > 0) {
            const code_point cp = decode(ellipsis, offset2);
            const std::size_t cells_here = cp.valid ? code_point_width(cp.value) : 1;
            if (cells_here > cells_left) {
                break;
            }
            head.append(ellipsis.substr(offset2, cp.valid ? cp.bytes : 1));
            cells_left -= cells_here;
            offset2 += cp.valid ? cp.bytes : 1;
        }
        return head;
    }
    return out;
}

std::vector<std::string> format_block(const snapshot &s, std::size_t width, const line_style &style,
                                      std::size_t height) {
    const std::size_t usable = std::max<std::size_t>(width, 8);
    std::vector<std::string> lines;
    if (s.stage == phase::compile && s.queries.empty()) {
        lines.push_back(compile_line(s, usable, style));
        return lines;
    }
    lines.push_back(shared_line(s, usable, style));
    if (s.queries.size() <= 1) {
        return lines; // a single target is folded into the shared line
    }

    // Never taller than the terminal: the shared line needs one row, so the list gets the rest,
    // capped by the style. 0 (unmeasurable) leaves the style cap in charge.
    std::size_t max_rows = std::max<std::size_t>(style.max_target_rows, 1);
    if (height > 0) {
        // The shared line takes one row; the list (with its summary row) gets the rest.
        max_rows = std::min(max_rows, height > 1 ? height - 1 : 1);
    }
    const row_plan plan = pick_rows(s, max_rows);
    const std::vector<std::size_t> &rows = plan.rows;
    const std::size_t slowest = slowest_index(s.queries);

    std::size_t name_width = 3;
    for (const std::size_t index : rows) {
        name_width = std::max(name_width, std::min<std::size_t>(display_width(s.queries[index].name), 12));
    }
    // 4 indent + name + 2 + 9 estimate + 1 + 3 sign + 7 width: keep 30 columns of room for them.
    const std::size_t room = usable > 30 ? usable - 30 : 3;
    name_width = std::clamp<std::size_t>(std::min(name_width, room), 3, 12);

    for (const std::size_t index : rows) {
        lines.push_back(truncate_to_width(target_row(s.queries[index], name_width, index == slowest, style, usable),
                                          usable, style.glyph_set));
    }
    if (plan.more) {
        lines.push_back(truncate_to_width("    " + std::string(marks_for(style.glyph_set).ellipsis) + " +" +
                                              std::to_string(s.queries.size() - rows.size()) + " more",
                                          usable, style.glyph_set));
    }
    return lines;
}

std::string format_summary(const snapshot &s, bool interrupted, std::size_t width) {
    std::string line;
    if (interrupted) {
        line = "interrupted after " + fmt_clock(s.elapsed_seconds);
        if (s.trials_done > 0) {
            line += "  " + fmt_count(s.trials_done) + " trials";
        }
    } else if (s.stage == phase::compile) {
        line = "compiled in " + fmt_stage_time(s.elapsed_seconds);
    } else {
        line = "sampled " + fmt_count(s.trials_done) + " trials in " + fmt_clock(s.elapsed_seconds);
    }
    const std::string rate = fmt_rate(s.trials_per_second);
    if (!rate.empty()) {
        line += " (" + rate + ")";
    }
    const std::string targets = summary_targets(s);
    if (!targets.empty() && (width == 0 || display_width(line) + 2 + display_width(targets) <= width)) {
        line += "  " + targets;
    }
    // The summary shares the block's guarantee: it never wraps either. Truncation is a last resort
    // because the fields above are ordered so the important ones come first.
    return width == 0 ? line : truncate_to_width(line, width, glyphs::utf8);
}

// ---------------------------------------------------------------------------------------------
// Signal payload.
// ---------------------------------------------------------------------------------------------

std::size_t signal_payload::publish(std::string_view bytes) {
    // Never touch the record the handler may be reading: only the inactive one is rewritten.
    const std::uint32_t target = 1u - active_;
    const std::size_t length = std::min(bytes.size(), signal_payload_capacity);
    std::memcpy(buffers_[target].data(), bytes.data(), length);
    lengths_[target] = length;
    active_ = target;
    // Release: the record and its length are visible to any handler that acquires this index. A
    // handler that observes the previous index keeps writing the previous record, which is intact.
    published_.store(target, std::memory_order_release);
    return length;
}

void signal_payload::withdraw() { published_.store(kNone, std::memory_order_release); }

std::string_view signal_payload::current() const {
    const std::uint32_t index = published_.load(std::memory_order_acquire);
    if (index > 1) {
        return {};
    }
    return {buffers_[index].data(), lengths_[index]};
}

// ---------------------------------------------------------------------------------------------
// Environment policy.
// ---------------------------------------------------------------------------------------------

bool stderr_is_tty() { return fd_is_tty(2); }

bool locale_uses_utf8(const char *lc_all, const char *lc_ctype, const char *lang) {
    return locale_is_utf8(lc_all, lc_ctype, lang);
}

display_policy resolve_display(bool no_progress, bool quiet) {
    return resolve_display(no_progress, quiet, std::getenv("FUZZYBITS_PROGRESS"), std::getenv("TERM"), stderr_is_tty());
}

display_policy resolve_display(bool no_progress, bool quiet, const char *progress_env, const char *term, bool is_tty) {
    if (no_progress || quiet) {
        return {};
    }
    const int forced = env_flag(progress_env);
    if (forced == 0) {
        return {};
    }
    if (forced == 1) {
        return {.animate = true, .summary = true};
    }
    if (!is_tty || (term != nullptr && iequals(term, "dumb"))) {
        return {}; // pipes, files, CI and dumb terminals get no escape sequences at all
    }
    return {.animate = true, .summary = true};
}

line_style style_from_environment() {
    line_style style;
    style.glyph_set = locale_uses_utf8(std::getenv("LC_ALL"), std::getenv("LC_CTYPE"), std::getenv("LANG"))
                          ? glyphs::utf8
                          : glyphs::ascii;
    return style;
}

// ---------------------------------------------------------------------------------------------
// Sink.
// ---------------------------------------------------------------------------------------------

line_sink::line_sink(int fd)
    : opt_{
          .style = style_from_environment(),
          .write = [](std::string_view bytes) { raw_write(2, bytes); },
          .terminal_width = [] { return query_terminal_width(2); },
          .terminal_height = [] { return query_terminal_height(2); },
          .fallback_width = 80,
          .hide_cursor = true,
          .own_signals = true,
      },
      fd_(fd) {
    if (fd_ != 2) {
        const int target = fd_;
        opt_.write = [target](std::string_view bytes) { raw_write(target, bytes); };
        opt_.terminal_width = [target] { return query_terminal_width(target); };
        opt_.terminal_height = [target] { return query_terminal_height(target); };
    }
}

line_sink::line_sink(options o) : opt_(std::move(o)) {
    if (!opt_.terminal_width) {
        opt_.terminal_width = [] { return query_terminal_width(2); };
    }
    if (!opt_.terminal_height) {
        opt_.terminal_height = [] { return query_terminal_height(2); };
    }
    if (!opt_.write) {
        opt_.write = [](std::string_view bytes) { raw_write(2, bytes); };
    }
    if (opt_.fallback_width == 0) {
        opt_.fallback_width = 80;
    }
    if (opt_.style.max_target_rows == 0) {
        opt_.style.max_target_rows = 4;
    }
    fd_ = 2;
}

line_sink::~line_sink() { stop(); }

std::size_t line_sink::width() const {
    const std::size_t columns = opt_.terminal_width();
    const std::size_t usable = columns > 1 ? columns : opt_.fallback_width;
    // One cell of slack: a line that exactly fills the terminal leaves a pending wrap, and the
    // following write would then cost an extra row and push the block down.
    return usable > 1 ? usable - 1 : 1;
}

void line_sink::write_out(std::string_view bytes) {
    if (!bytes.empty()) {
        opt_.write(bytes);
    }
}

namespace {

/// Erase `rows` lines in place and leave the cursor at the top of the block.
void append_erase(std::string &out, std::size_t rows) {
    if (rows == 0) {
        return;
    }
    out += "\r";
    out += kEraseLine;
    for (std::size_t i = 1; i < rows; ++i) {
        out += kCursorUp;
        out += "\r";
        out += kEraseLine;
    }
}

} // namespace

void line_sink::erase_block(std::string &out) {
    append_erase(out, rows_);
    rows_ = 0;
}

void line_sink::draw(const snapshot &s) {
    const std::vector<std::string> lines =
        format_block(s, width(), opt_.style, opt_.terminal_height ? opt_.terminal_height() : 0);
    std::string out;
    if (rows_ > 0) {
        erase_block(out);
    } else if (opt_.hide_cursor && !cursor_hidden_) {
        out += kHideCursor;
        cursor_hidden_ = true;
#if !defined(_WIN32)
        if (opt_.own_signals) {
            signal_channel_ = &signal_channel_for(fd_);
            signals_owned_ = install_signal_handlers(fd_, *signal_channel_);
        }
#endif
    }
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) {
            out += "\r\n";
        }
        out += lines[i];
    }
    rows_ = lines.size();

    // Publish the signal payload before the frame becomes visible. The cursor-hide sequence is what
    // a test (or a user) treats as "a frame was drawn", so a handler that fires right after it must
    // already find a payload; publishing afterwards left a window in which the handler saw none and
    // the block was left on screen.
    if (signals_owned_ && signal_channel_ != nullptr) {
        // Pre-render what a SIGINT/SIGTERM has to print: the same erase the normal path does, the
        // partial summary, then the cursor. Refreshed every frame, so the handler always has current
        // numbers without formatting anything (which is not async-signal-safe).
        std::string bytes;
        append_erase(bytes, rows_);
        bytes += format_summary(s, /*interrupted=*/true, width());
        bytes += "\r\n";
        // The cursor restore is appended last and the payload is truncated at a fixed capacity, so a
        // long partial summary loses its own tail, never the restore.
        if (cursor_hidden_) {
            if (bytes.size() + kShowCursor.size() > signal_payload_capacity) {
                bytes.resize(signal_payload_capacity - kShowCursor.size());
            }
            bytes += kShowCursor;
        }
        (void)signal_channel_->payload.publish(bytes);
    }

    write_out(out);
}

void line_sink::on_snapshot(const snapshot &s) {
    if (stopped_) {
        return;
    }
    if (s.stage != phase::done) {
        // `phase::done` is the terminal marker the emitter broadcasts just before on_finish; it
        // carries no work, so the summary is built from the last real snapshot instead.
        saw_snapshot_ = true;
        last_ = s;
        draw(s);
    }
}

void line_sink::on_finish(const snapshot &last, bool interrupted) {
    if (saw_snapshot_) {
        // The summary is built from the last snapshot that carried work. The emitter's terminal
        // `phase::done` marker has no trials, so it only contributes its fresher elapsed time.
        if (last.stage != phase::done) {
            last_ = last;
        } else if (last.elapsed_seconds > last_.elapsed_seconds) {
            last_.elapsed_seconds = last.elapsed_seconds;
        }
    }
    finish_now(interrupted);
}

void line_sink::finish_now(bool interrupted) {
    if (stopped_) {
        return;
    }
    std::string out;
    erase_block(out);
    if (saw_snapshot_) {
        out += format_summary(last_, interrupted, width());
        out += "\r\n";
    }
    if (cursor_hidden_) {
        out += kShowCursor;
        cursor_hidden_ = false;
#if !defined(_WIN32)
        if (signals_owned_) {
            remove_signal_handlers();
            signals_owned_ = false;
        }
#endif
    }
    stopped_ = true;
    write_out(out);
}

void line_sink::stop() {
    if (stopped_) {
        return;
    }
    std::string out;
    erase_block(out);
    if (cursor_hidden_) {
        out += kShowCursor;
        cursor_hidden_ = false;
#if !defined(_WIN32)
        if (signals_owned_) {
            remove_signal_handlers();
            signals_owned_ = false;
        }
#endif
    }
    stopped_ = true;
    write_out(out);
}

} // namespace fuzzybits::runtime::progress
