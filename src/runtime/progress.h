#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::runtime::progress {

inline constexpr std::uint32_t schema_version = 1;

enum class phase : std::uint8_t { compile, sampling, checking, done };

struct query_state {
    std::string name;
    std::uint64_t hits = 0;
    std::uint64_t trials = 0;
    double estimate = std::numeric_limits<double>::quiet_NaN();
    double ci_low = std::numeric_limits<double>::quiet_NaN();
    double ci_high = std::numeric_limits<double>::quiet_NaN();
    // (ci_high - ci_low) / (2 estimate); NaN if estimate == 0.
    double rel_half_width = std::numeric_limits<double>::quiet_NaN();
    // NaN when there is no convergence target.
    double target_rel_half_width = std::numeric_limits<double>::quiet_NaN();
    bool converged = false;
};

struct snapshot {
    std::uint32_t schema = schema_version;
    phase stage = phase::sampling;
    // Compile sub-stage ("parse", "lower", "passes", "write"); empty otherwise.
    std::string step;
    std::uint64_t trials_done = 0;
    std::uint64_t trials_total = 0; // 0 = unknown
    std::uint64_t passes_done = 0;
    std::uint64_t passes_total = 0; // 0 = unknown
    double elapsed_seconds = 0.0;
    double trials_per_second = 0.0;   // sliding window
    std::vector<query_state> queries; // query order of the run
};

class sink {
  public:
    virtual ~sink() = default;
    // Called on the run thread, coalesced.
    virtual void on_snapshot(const snapshot &s) = 0;
    // Exactly once per run.
    virtual void on_finish(const snapshot &last, bool interrupted) = 0;
};

// Thrown by a sink to stop the run early (e.g. a Python callback raised). The engine unwinds
// without touching the results, reports `finish(..., interrupted=true)`, and rethrows this type
// unchanged so the caller can restore its own error state.
class interrupted : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Owned by a run. Coalesces to at most max_rate_hz; always delivers the first and the last snapshot;
// builds nothing when no sink is attached.
class emitter {
  public:
    using clock = std::chrono::steady_clock;
    using now_fn = std::function<clock::time_point()>;

    // `max_rate_hz <= 0` means unlimited (every maybe_emit reaches the sinks). `now` exists so tests
    // can drive time; it defaults to steady_clock::now.
    explicit emitter(std::vector<sink *> sinks, double max_rate_hz = 10.0, now_fn now = {})
        : sinks_(std::move(sinks)), now_(now ? std::move(now) : now_fn([] { return clock::now(); })) {
        if (max_rate_hz > 0.0) {
            interval_ = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / max_rate_hz));
            if (interval_.count() <= 0) {
                interval_ = clock::duration{1};
            }
        } else {
            unlimited_ = true;
        }
    }

    [[nodiscard]] bool active() const noexcept { return !sinks_.empty(); }

    // Whether the next `maybe_emit` would reach the sinks. Lets a caller skip building the inputs for
    // a snapshot (e.g. a tally collection) that would be coalesced away.
    [[nodiscard]] bool due() const {
        if (sinks_.empty()) {
            return false;
        }
        return !emitted_ || unlimited_ || now_() - last_ >= interval_;
    }

    template <class MakeSnapshot> void maybe_emit(MakeSnapshot &&make) {
        if (!due()) {
            return;
        }
        emit_now(make());
    }

    // Delivers `s` immediately, regardless of the coalescing interval.
    void emit_now(snapshot s) {
        if (sinks_.empty()) {
            return;
        }
        last_ = now_();
        emitted_ = true;
        broadcast(s);
    }

    // Bypasses the rate limit: delivers `last` to every sink and then on_finish, exactly once. A sink
    // that throws is skipped for the rest of this call, but on_finish still runs for every sink, and
    // the emitter raises its own typed signal afterwards so a failing callback is never silent.
    void finish(snapshot last, bool was_interrupted) {
        if (sinks_.empty() || finished_) {
            return;
        }
        finished_ = true;
        bool failed = false;
        for (sink *target : sinks_) {
            try {
                target->on_snapshot(last);
            } catch (...) {
                failed = true;
            }
        }
        for (sink *target : sinks_) {
            try {
                target->on_finish(last, was_interrupted);
            } catch (...) {
                failed = true;
            }
        }
        if (failed) {
            // Typed and catchable: the sinks' own messages are already reported (a Python callback stores
            // its exception for the binding to re-raise), so this only signals that finishing failed.
            throw interrupted("progress: a sink raised while finishing");
        }
    }

  private:
    void broadcast(const snapshot &s) {
        for (sink *target : sinks_) {
            target->on_snapshot(s);
        }
    }

    std::vector<sink *> sinks_;
    now_fn now_;
    clock::duration interval_{};
    bool unlimited_ = false;
    bool emitted_ = false;
    bool finished_ = false;
    clock::time_point last_{};
};

// Calls `finish()` from its destructor unless the emitting scope completed, so every sink sees
// exactly one `on_finish` even when the work fails for a reason unrelated to progress (device error,
// a failed compile stage, a Python callback raising). `finish()` is expected to report
// `interrupted=true`; errors it raises are swallowed because a destructor must not throw and must
// never replace the in-flight exception.
//
// Wrap the whole emitting scope, and call `complete()` only after the normal-path `finish()`:
//
//     finish_guard guard([&] { progress.finish(&tallies, passes_done, true); });
//     ... work; guard-arm stays live for every throw ...
//     progress.finish(&tallies, passes_done, false);
//     guard.complete();
template <class Finish> class finish_guard {
  public:
    explicit finish_guard(Finish finish) : finish_(std::move(finish)) {}

    ~finish_guard() {
        if (!armed_) {
            return;
        }
        try {
            finish_();
        } catch (...) {
        }
    }

    finish_guard(const finish_guard &) = delete;
    finish_guard &operator=(const finish_guard &) = delete;
    finish_guard(finish_guard &&other) noexcept : finish_(std::move(other.finish_)), armed_(std::exchange(other.armed_, false)) {}
    finish_guard &operator=(finish_guard &&) = delete;

    // The normal path already ran `finish()`.
    void complete() noexcept { armed_ = false; }

    [[nodiscard]] bool armed() const noexcept { return armed_; }

  private:
    Finish finish_;
    bool armed_ = true;
};

} // namespace fuzzybits::runtime::progress
