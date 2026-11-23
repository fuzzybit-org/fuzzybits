#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "runtime/progress.h"

namespace fuzzybits::runtime::progress {

/// Glyphs for the bar. `utf8` uses box-drawing characters; `ascii` is printable ASCII only, so the
/// renderer stays correct under a non-UTF-8 ctype (where the legacy display measured box glyphs as
/// zero cells and emitted mangled bars).
enum class glyphs : std::uint8_t { ascii, utf8 };

struct line_style {
    glyphs glyph_set = glyphs::utf8;
    /// Per-target rows, including the `+N more` row.
    std::size_t max_target_rows = 4;
};

// ---------------------------------------------------------------------------------------------
// Environment policy, shared by the native CLI and the Python CLI so both decide identically.
// ---------------------------------------------------------------------------------------------

struct display_policy {
    /// Redraw the animated block on stderr.
    bool animate = false;
    /// Print the one-line summary on stderr when the run ends (only meaningful with `animate`).
    bool summary = false;

    [[nodiscard]] bool enabled() const noexcept { return animate || summary; }
};

[[nodiscard]] bool stderr_is_tty();
[[nodiscard]] bool locale_uses_utf8(const char *lc_all, const char *lc_ctype, const char *lang);

/// Policy for the real environment: `--no-progress` / `--quiet` / `FUZZYBITS_PROGRESS` / `TERM`.
[[nodiscard]] display_policy resolve_display(bool no_progress, bool quiet);
/// Same policy with every input supplied, so it is testable without a terminal.
/// `progress_env == nullptr` means unset; `0/false/no/off` disables, `1/true/yes/on` forces the
/// display on without a TTY (stderr only). Flags win over the environment.
[[nodiscard]] display_policy resolve_display(bool no_progress, bool quiet, const char *progress_env, const char *term,
                                             bool is_tty);

/// Glyphs for the current stderr locale.
[[nodiscard]] line_style style_from_environment();

// ---------------------------------------------------------------------------------------------
// Pure formatting: no terminal, no globals. Every returned line is at most `width` cells wide, so
// the terminal never wraps it and the block cannot drift.
// ---------------------------------------------------------------------------------------------

/// Cells `text` occupies on a UTF-8 terminal (double-width and combining code points counted;
/// a stray byte counts as one cell).
[[nodiscard]] std::size_t display_width(std::string_view text);

/// Cut `text` to `width` cells, adding an ellipsis when anything was dropped. Non-ASCII bytes
/// become `?` under `glyphs::ascii`.
[[nodiscard]] std::string truncate_to_width(std::string_view text, std::size_t width, glyphs g);

/// Block for one snapshot: a shared progress line, plus a bounded per-target list when the run
/// watches several targets. The line count depends only on the query count and the style, so the
/// height is fixed for the run and every redraw moves the cursor by an exact count.
/// `height` (0 = unknown) caps the block so it never exceeds the terminal.
[[nodiscard]] std::vector<std::string> format_block(const snapshot &s, std::size_t width, const line_style &style,
                                                    std::size_t height = 0);

/// One line that replaces the block when the run ends. `width` (0 = unbounded) truncates it, so a
/// narrow terminal cannot wrap it either; the per-target tail is dropped before the timing is.
[[nodiscard]] std::string format_summary(const snapshot &s, bool interrupted, std::size_t width = 0);

// ---------------------------------------------------------------------------------------------
// Signal payload: the bytes a SIGINT/SIGTERM handler writes (erase the block, the partial
// summary, restore the cursor). A handler may call nothing but async-signal-safe functions, so the
// bytes are formatted here and only `write(2)` runs in the handler.
// ---------------------------------------------------------------------------------------------

/// Largest payload the handler publishes. A wider block (an 8K-wide terminal) is truncated; the
/// cursor restore is appended after truncation, so it is never the part that is cut.
inline constexpr std::size_t signal_payload_capacity = 4096;

/// Two fixed-size records and one atomic index. The publisher writes the inactive record and then
/// stores its index with release; a handler does a single acquire load and writes that record.
///
/// Nothing is allocated, freed or moved while handlers are installed, so the publisher cannot pull
/// the bytes out from under a concurrent handler, and the release/acquire pair means a reader never
/// observes a partially written record. A process-directed signal can be delivered to any thread
/// that does not block it, so this has to hold against the OpenMP/SYCL workers too.
class signal_payload {
  public:
    /// Records must be lock-free: a handler cannot take a lock.
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free, "the signal handler needs a lock-free atomic index");

    signal_payload() = default;
    signal_payload(const signal_payload &) = delete;
    signal_payload &operator=(const signal_payload &) = delete;

    /// Publisher side. Copies at most `signal_payload_capacity` bytes into the inactive record and
    /// publishes it. Returns the published length.
    [[nodiscard]] std::size_t publish(std::string_view bytes);

    /// Withdraws the payload: after this returns a handler writes nothing. A handler already inside
    /// `write` still finishes, because the records outlive the sink.
    void withdraw();

    /// Handler side: the single acquire load that yields a consistent (bytes, length) pair. Empty
    /// when nothing is published or after `withdraw`.
    [[nodiscard]] std::string_view current() const;

  private:
    static constexpr std::uint32_t kNone = 2;

    std::array<char, signal_payload_capacity> buffers_[2]{};
    std::array<std::size_t, 2> lengths_{};
    std::atomic<std::uint32_t> published_{kNone};
    /// Publisher-owned; never read by the handler.
    std::uint32_t active_ = 0;
};

/// What a handler needs: the descriptor plus the payload. Published as one pointer so a handler
/// gets a consistent pair from a single acquire load, and never freed, so a handler that outlives
/// the sink still reads valid memory.
struct signal_channel {
    int fd = -1;
    signal_payload payload;
};

// ---------------------------------------------------------------------------------------------
// Terminal sink.
// ---------------------------------------------------------------------------------------------

/// Draws snapshots on a file descriptor (stderr by default). Redraws in place with cursor-up plus
/// erase-line, hides the cursor for the duration of the run and shows it exactly once, and leaves
/// only the summary line behind.
class line_sink final : public sink {
  public:
    struct options {
        line_style style;
        /// Receives every byte written.
        std::function<void(std::string_view)> write;
        /// Terminal width in cells, re-read for every frame so a resize is picked up.
        std::function<std::size_t()> terminal_width;
        /// Terminal height in rows, re-read for every frame. 0 when unmeasurable.
        std::function<std::size_t()> terminal_height;
        /// Width used when the terminal cannot be measured.
        std::size_t fallback_width = 80;
        /// Hide the cursor while drawing and restore it once at the end.
        bool hide_cursor = true;
        /// Restore the cursor from a SIGINT/SIGTERM handler, but only when the process still has the
        /// default disposition (the native CLI). A host with its own handler (CPython) keeps it; the
        /// destructor restores the cursor when the run unwinds.
        bool own_signals = true;
    };

    /// Writing to `fd` (2 = stderr).
    explicit line_sink(int fd = 2);
    explicit line_sink(options o);
    ~line_sink() override;

    line_sink(const line_sink &) = delete;
    line_sink &operator=(const line_sink &) = delete;
    line_sink(line_sink &&) = delete;
    line_sink &operator=(line_sink &&) = delete;

    void on_snapshot(const snapshot &s) override;
    void on_finish(const snapshot &last, bool interrupted) override;

    /// Summarise the last snapshot seen and restore the terminal, for a run whose emitter never
    /// reached `on_finish` (a compile pipeline that returned early, an exception). A no-op when the
    /// sink already finished, so the caller can call it unconditionally.
    void finish_now(bool interrupted);

    /// Erase the block and restore the terminal. Idempotent, and called by the destructor, so an
    /// exception or an early return cannot leave the cursor hidden.
    void stop();

    /// Rows currently on screen (0 before the first frame).
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }

  private:
    void draw(const snapshot &s);
    void erase_block(std::string &out);
    [[nodiscard]] std::size_t width() const;
    void write_out(std::string_view bytes);

    options opt_;
    int fd_ = -1;
    std::size_t rows_ = 0;
    bool cursor_hidden_ = false;
    bool stopped_ = false;
    bool signals_owned_ = false;
    bool saw_snapshot_ = false;
    snapshot last_;
    /// Refreshed on every frame: the bytes a signal handler needs (erase the block, print the
    /// partial summary, show the cursor). A handler may only call async-signal-safe functions, so it
    /// writes the published record with `write(2)` instead of formatting anything itself.
    /// The buffer is the channel's, not the sink's, so it outlives the sink on an interrupt path.
    signal_channel *signal_channel_ = nullptr;
};

} // namespace fuzzybits::runtime::progress
