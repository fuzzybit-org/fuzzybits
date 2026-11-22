#include "cli/progress_line.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace progress = fuzzybits::runtime::progress;

namespace {

progress::query_state make_query(std::string name, double estimate, double rel) {
  progress::query_state q;
  q.name = std::move(name);
  q.estimate = estimate;
  q.rel_half_width = rel;
  return q;
}

progress::snapshot sampling_snapshot() {
  progress::snapshot s;
  s.stage = progress::phase::sampling;
  s.trials_done = 1600000000;
  s.trials_total = 2600000000;
  s.elapsed_seconds = 21.0;
  s.trials_per_second = 3.2e9;
  return s;
}

progress::line_style utf8_style() { return progress::line_style{}; }

progress::line_style ascii_style() {
  progress::line_style style;
  style.glyph_set = progress::glyphs::ascii;
  return style;
}

constexpr std::size_t kWidths[] = {40, 80, 120};

}  // namespace

TEST_CASE("display_width counts cells, not bytes", "[progress][unit]") {
  REQUIRE(progress::display_width("") == 0);
  REQUIRE(progress::display_width("abc") == 3);
  // U+2501, U+2192, U+00B1: three boxes, one cell each.
  REQUIRE(progress::display_width("\u2501\u2192\u00b1") == 3);
  // CJK is two cells.
  REQUIRE(progress::display_width("\u4e2d\u6587") == 4);
  // A truncation must never split a multi-byte code point into stray bytes.
  REQUIRE(progress::display_width("a\u2501b") == 3);
}

TEST_CASE("truncate_to_width never exceeds the budget", "[progress][unit]") {
  const std::string wide = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  for (const std::size_t width : {0u, 1u, 2u, 5u, 10u, 40u}) {
    for (const auto style : {utf8_style(), ascii_style()}) {
      const std::string cut = progress::truncate_to_width(wide, width, style.glyph_set);
      REQUIRE(progress::display_width(cut) <= width);
    }
  }
  // Short input is returned untouched.
  REQUIRE(progress::truncate_to_width("abc", 10, progress::glyphs::utf8) == "abc");
  // Non-ASCII bytes are replaced under ASCII, so a C locale cannot emit raw UTF-8.
  const std::string replaced = progress::truncate_to_width("\u2501\u2501\u2501\u2501", 4,
                                                          progress::glyphs::ascii);
  for (const char c : replaced) {
    REQUIRE(static_cast<unsigned char>(c) < 0x80);
  }
}

TEST_CASE("single-target block folds the estimate into the shared line", "[progress][unit]") {
  progress::snapshot s = sampling_snapshot();
  s.queries.push_back(make_query("TOP", 7.452e-2, 0.0019));

  const auto block = progress::format_block(s, 120, utf8_style());
  REQUIRE(block.size() == 1);
  REQUIRE(block.front().find("sampling") != std::string::npos);
  REQUIRE(block.front().find("62%") != std::string::npos);
  REQUIRE(block.front().find("1.6e9/2.6e9 trials") != std::string::npos);
  REQUIRE(block.front().find("TOP 7.452e-02 \u00b10.19%") != std::string::npos);
  REQUIRE(block.front().find("3.2 Gtrial/s") != std::string::npos);
  REQUIRE(block.front().find("ETA 0:") != std::string::npos);
}

TEST_CASE("every line fits the terminal width", "[progress][unit]") {
  progress::snapshot s = sampling_snapshot();
  s.queries.push_back(make_query("TOP", 7.452e-2, 0.0019));
  for (int i = 0; i < 20; ++i) {
    s.queries.push_back(make_query("TARGET-" + std::to_string(i), 1.1e-5 * (i + 1), 0.041 * (i + 1)));
  }
  for (const std::size_t width : kWidths) {
    for (const auto style : {utf8_style(), ascii_style()}) {
      for (const std::string &line : progress::format_block(s, width, style)) {
        INFO("width=" << width << " line='" << line << "' cells=" << progress::display_width(line));
        REQUIRE(progress::display_width(line) <= width);
      }
    }
  }
}

TEST_CASE("multi-target block is bounded and keeps the slowest", "[progress][unit]") {
  progress::snapshot s = sampling_snapshot();
  s.queries.push_back(make_query("TOP", 7.452e-2, 0.0019));
  for (int i = 0; i < 17; ++i) {
    s.queries.push_back(make_query("CLASS-" + std::to_string(i), 6.27e-5, 0.01 * (i + 1)));
  }
  // Index 17 (CLASS-16) has the widest interval and must be listed.
  progress::line_style style;
  style.max_target_rows = 4;
  const auto block = progress::format_block(s, 80, style);

  // 1 shared line + 3 target rows + the `+N more` row.
  REQUIRE(block.size() == 5);
  const std::string joined = [&] {
    std::string all;
    for (const auto &line : block) {
      all += line;
      all += "\n";
    }
    return all;
  }();
  REQUIRE(joined.find("+15 more") != std::string::npos);
  // The widest interval (CLASS-16, 0.01 * 17) must be listed and marked as the slowest.
  const auto slowest_row = std::find_if(block.begin(), block.end(), [](const std::string &line) {
    return line.find("CLASS-16") != std::string::npos;
  });
  REQUIRE(slowest_row != block.end());
  REQUIRE(slowest_row->find("\u2190") != std::string::npos);
  // Top row is query order.
  REQUIRE(block[1].find("TOP") != std::string::npos);
}

TEST_CASE("block height is constant across snapshots", "[progress][unit]") {
  progress::snapshot early = sampling_snapshot();
  early.trials_done = 1;
  early.trials_per_second = 0.0;
  progress::snapshot late = sampling_snapshot();
  for (int i = 0; i < 7; ++i) {
    progress::query_state q = make_query("T" + std::to_string(i), 1e-3, 0.1);
    q.converged = i < 2;  // the `converged` mark must not change the height
    early.queries.push_back(q);
    late.queries.push_back(q);
  }
  for (const std::size_t width : kWidths) {
    REQUIRE(progress::format_block(early, width, utf8_style()).size() ==
            progress::format_block(late, width, utf8_style()).size());
  }
}

TEST_CASE("block height is clamped by the terminal height", "[progress][unit]") {
  progress::snapshot s = sampling_snapshot();
  for (int i = 0; i < 19; ++i) {
    s.queries.push_back(make_query("T" + std::to_string(i), 1e-3, 0.01 * (i + 1)));
  }
  // Unknown height: the style cap decides (4 target rows + the shared line).
  REQUIRE(progress::format_block(s, 80, utf8_style(), 0).size() == 5);
  // A short terminal wins over the style, and the height still does not depend on the data.
  REQUIRE(progress::format_block(s, 80, utf8_style(), 4).size() == 4);
  REQUIRE(progress::format_block(s, 80, utf8_style(), 8).size() == 5);
  // Degenerate heights still produce one row rather than nothing.
  REQUIRE(progress::format_block(s, 80, utf8_style(), 1).size() == 2);
  REQUIRE(progress::format_block(s, 80, utf8_style(), 2).size() == 2);
}

TEST_CASE("seven targets at 80x24 use the whole row budget", "[progress][unit]") {
  progress::snapshot s = sampling_snapshot();
  s.queries.push_back(make_query("r1", 1.172e-3, 0.0022));
  for (const char *name : {"e1", "e2", "e3", "e4", "e10", "e20"}) {
    s.queries.push_back(make_query(name, 1.0e-2, 0.05));
  }
  progress::line_style style;  // max_target_rows = 4
  const auto block = progress::format_block(s, 79, style, 23 /* rows - 1 */);
  INFO("block=" << block.size());
  for (const auto &line : block) {
    INFO("  '" << line << "'");
  }
  // 1 shared line + 3 target rows + the `+N more` row.
  REQUIRE(block.size() == 5);
}

TEST_CASE("unknown totals and missing estimates degrade cleanly", "[progress][unit]") {
  progress::snapshot s;
  s.stage = progress::phase::sampling;
  s.trials_done = 0;
  s.trials_total = 0;  // unknown
  s.trials_per_second = std::numeric_limits<double>::quiet_NaN();
  s.queries.push_back(make_query("TOP", std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::quiet_NaN()));

  const auto block = progress::format_block(s, 80, utf8_style());
  REQUIRE(block.size() == 1);
  REQUIRE(block.front().find('%') == std::string::npos);
  REQUIRE(block.front().find("ETA") == std::string::npos);
  REQUIRE(block.front().find("TOP n/a \u00b1n/a") != std::string::npos);
  for (const char c : block.front()) {
    REQUIRE(c != '\n');
    REQUIRE(c != '\r');
  }
}

TEST_CASE("zero and infinite estimates are stable", "[progress][unit]") {
  progress::snapshot s = sampling_snapshot();
  s.queries.push_back(make_query("ZERO", 0.0, std::numeric_limits<double>::quiet_NaN()));
  s.queries.push_back(make_query("INF", std::numeric_limits<double>::infinity(), 0.5));
  for (const std::string &line : progress::format_block(s, 80, utf8_style())) {
    REQUIRE(line.find("nan") == std::string::npos);
    REQUIRE(line.find("inf") == std::string::npos);
  }
}

TEST_CASE("compile snapshots render the stage chain", "[progress][unit]") {
  progress::snapshot s;
  s.stage = progress::phase::compile;
  s.step = "passes";
  s.elapsed_seconds = 3.2;
  s.passes_done = 2;
  s.passes_total = 5;

  const auto block = progress::format_block(s, 80, utf8_style());
  REQUIRE(block.size() == 1);
  REQUIRE(block.front().find("compiling") != std::string::npos);
  REQUIRE(block.front().find("[passes]") != std::string::npos);
  REQUIRE(block.front().find("3.2 s") != std::string::npos);
  REQUIRE(block.front().find("C2/5") != std::string::npos);
  REQUIRE(progress::display_width(block.front()) <= 80);

  const auto ascii_block = progress::format_block(s, 80, ascii_style());
  REQUIRE(ascii_block.front().find("\u2192") == std::string::npos);
  REQUIRE(ascii_block.front().find("->") != std::string::npos);
}

TEST_CASE("summary reports elapsed, trials, rate and the top estimate", "[progress][unit]") {
  progress::snapshot s = sampling_snapshot();
  s.trials_done = 2600000000;
  s.elapsed_seconds = 8.0;
  s.queries.push_back(make_query("TOP", 7.452e-2, 0.0019));

  const std::string line = progress::format_summary(s, /*interrupted=*/false);
  REQUIRE(line.find("sampled") != std::string::npos);
  REQUIRE(line.find("trials in 0:08") != std::string::npos);
  REQUIRE(line.find("3.2 Gtrial/s") != std::string::npos);
  REQUIRE(line.find("7.452e-02") != std::string::npos);
  REQUIRE(line.find("0.19%") != std::string::npos);
  REQUIRE(line.find('\n') == std::string::npos);

  progress::snapshot many = s;
  many.queries.push_back(make_query("OTHER", 1e-5, 0.5));
  const std::string many_line = progress::format_summary(many, false);
  REQUIRE(many_line.find("+1 more") != std::string::npos);

  const std::string stopped = progress::format_summary(s, /*interrupted=*/true);
  REQUIRE(stopped.find("interrupted") != std::string::npos);
}

TEST_CASE("summary honours the width so it cannot wrap", "[progress][unit]") {
  progress::snapshot s = sampling_snapshot();
  s.trials_done = 2600000000;
  s.elapsed_seconds = 8.0;
  for (int i = 0; i < 8; ++i) {
    s.queries.push_back(make_query("TARGET-" + std::to_string(i), 1e-3 * (i + 1), 0.1));
  }
  // Unbounded stays whole.
  REQUIRE(progress::format_summary(s, false, 0).find("+7 more") != std::string::npos);
  for (const std::size_t width : {40u, 80u, 120u}) {
    const std::string line = progress::format_summary(s, false, width);
    INFO("width=" << width << " line='" << line << "'");
    REQUIRE(progress::display_width(line) <= width);
    REQUIRE(line.find('\n') == std::string::npos);
  }
  // At 40 columns the per-target tail is the first thing dropped, not the timing.
  const std::string narrow = progress::format_summary(s, false, 40);
  REQUIRE(narrow.find("+7 more") == std::string::npos);
  REQUIRE(narrow.find("sampled") != std::string::npos);
}

TEST_CASE("progress policy: flags win, env disables, TTY is required", "[progress][unit]") {
  using progress::resolve_display;
  // Default: on, but only for a terminal.
  REQUIRE(resolve_display(false, false, nullptr, "xterm-256color", true).animate);
  REQUIRE_FALSE(resolve_display(false, false, nullptr, "xterm-256color", false).animate);
  // Flags win over the environment.
  REQUIRE_FALSE(resolve_display(true, false, "1", "xterm", true).animate);
  REQUIRE_FALSE(resolve_display(false, true, "1", "xterm", true).animate);
  // FUZZYBITS_PROGRESS=0 disables; a truthy value forces on without a TTY.
  REQUIRE_FALSE(resolve_display(false, false, "0", "xterm", true).animate);
  REQUIRE_FALSE(resolve_display(false, false, "off", "xterm", true).animate);
  REQUIRE(resolve_display(false, false, "1", "xterm", false).animate);
  // TERM=dumb is off; an unknown value leaves the default untouched.
  REQUIRE_FALSE(resolve_display(false, false, nullptr, "dumb", true).animate);
  REQUIRE(resolve_display(false, false, "maybe", "xterm", true).animate);
  // A display never comes without its summary.
  REQUIRE(resolve_display(false, false, nullptr, "xterm", true).summary);
}

TEST_CASE("locale sniffing", "[progress][unit]") {
  REQUIRE(progress::locale_uses_utf8("C.UTF-8", nullptr, nullptr));
  REQUIRE(progress::locale_uses_utf8(nullptr, "en_US.utf8", nullptr));
  REQUIRE(progress::locale_uses_utf8(nullptr, nullptr, "de_DE.UTF-8"));
  REQUIRE_FALSE(progress::locale_uses_utf8("C", "en_US.UTF-8", nullptr));  // LC_ALL wins
  REQUIRE_FALSE(progress::locale_uses_utf8(nullptr, nullptr, "C"));
  REQUIRE_FALSE(progress::locale_uses_utf8(nullptr, nullptr, nullptr));
}

// Not run by default (`[.]`): `fuzzybits_progress_line_tests "[example]" -s` prints the blocks
// used in the PR description and the docs. Kept as a test so the examples cannot rot.
TEST_CASE("example renders", "[.example]") {
  progress::snapshot single = sampling_snapshot();
  single.queries.push_back(make_query("TOP", 7.452e-2, 0.0019));
  const auto single_block = progress::format_block(single, 80, utf8_style());
  std::printf("--- single target, 80 columns ---\n");
  for (const auto &line : single_block) {
    std::printf("%s|  w=%zu\n", line.c_str(), progress::display_width(line));
  }

  progress::snapshot many = sampling_snapshot();
  many.queries.push_back(make_query("TOP", 7.452e-2, 0.0019));
  many.queries.push_back(make_query("CLASS-1A", 6.270e-5, 0.041));
  many.queries.push_back(make_query("CLASS-2A", 1.110e-5, 0.098));
  for (int i = 0; i < 14; ++i) {
    many.queries.push_back(make_query("CLASS-" + std::to_string(i + 4), 3.3e-6, 0.02 * (i + 1)));
  }
  progress::line_style style;
  style.max_target_rows = 4;
  std::printf("--- 17 targets, 80 columns ---\n");
  for (const auto &line : progress::format_block(many, 80, style)) {
    std::printf("%s|  w=%zu\n", line.c_str(), progress::display_width(line));
  }
  std::printf("--- 17 targets, 40 columns, ASCII ---\n");
  for (const auto &line : progress::format_block(many, 40, ascii_style())) {
    std::printf("%s|  w=%zu\n", line.c_str(), progress::display_width(line));
  }
  std::printf("--- summary ---\n");
  std::printf("%s\n", progress::format_summary(many, false).c_str());

  progress::snapshot compiled;
  compiled.stage = progress::phase::compile;
  compiled.step = "passes";
  compiled.elapsed_seconds = 3.2;
  std::printf("--- compile, 80 columns ---\n");
  for (const auto &line : progress::format_block(compiled, 80, utf8_style())) {
    std::printf("%s|  w=%zu\n", line.c_str(), progress::display_width(line));
  }
}

TEST_CASE("signal payload publishes and withdraws consistently", "[progress][unit]") {
  progress::signal_payload payload;
  // Nothing published yet: a handler writes nothing.
  REQUIRE(payload.current().empty());

  const std::string first = "\x1b[2Kinterrupted after 0:06\r\n\x1b[?25h";
  REQUIRE(payload.publish(first) == first.size());
  REQUIRE(payload.current() == first);

  // A second publish must alternate records so the first one stays intact for a handler that
  // already acquired it: the old bytes are never overwritten in place.
  const std::string second = "\x1b[2Kinterrupted after 0:07\r\n\x1b[?25h";
  REQUIRE(payload.publish(second) == second.size());
  REQUIRE(payload.current() == second);
  REQUIRE(payload.current().data() != first.data());

  // Publishing again returns to the first record, whose bytes were rewritten for this publish.
  REQUIRE(payload.publish(first) == first.size());
  REQUIRE(payload.current() == first);

  payload.withdraw();
  REQUIRE(payload.current().empty());
}

TEST_CASE("signal payload truncates at the fixed capacity", "[progress][unit]") {
  progress::signal_payload payload;
  // An oversized payload is cut, never allowed to overrun the record.
  const std::string huge(progress::signal_payload_capacity + 100, 'x');
  REQUIRE(payload.publish(huge) == progress::signal_payload_capacity);
  REQUIRE(payload.current().size() == progress::signal_payload_capacity);
  REQUIRE(payload.current().front() == 'x');
}

TEST_CASE("signal payload never reuses the record a reader may hold", "[progress][unit]") {
  progress::signal_payload payload;
  const std::string first = "first-payload";
  (void)payload.publish(first);
  // A handler holds this view while it writes. The next publish must go to the other record, so
  // the bytes behind `held` stay exactly as they were.
  const std::string_view held = payload.current();
  const std::string copy(held);
  const char *const address = held.data();

  (void)payload.publish("second-payload");
  REQUIRE(payload.current().data() != address);
  REQUIRE(std::string_view{address, copy.size()} == copy);  // still the first record, untouched

  // The third publish returns to that record, which is allowed now: the handler that could still
  // be writing had acquired the second record, not this one.
  (void)payload.publish("third-payload");
  REQUIRE(payload.current() == "third-payload");
}
