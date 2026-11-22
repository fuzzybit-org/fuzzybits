#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
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
  double rel_half_width = std::numeric_limits<double>::quiet_NaN();        // (ci_high - ci_low) / (2 estimate); NaN if estimate == 0
  double target_rel_half_width = std::numeric_limits<double>::quiet_NaN(); // NaN when there is no convergence target
  bool converged = false;
};

struct snapshot {
  std::uint32_t schema = schema_version;
  phase stage = phase::sampling;
  std::string step;                         // compile sub-stage ("parse", "lower", "passes", "write"); empty otherwise
  std::uint64_t trials_done = 0;
  std::uint64_t trials_total = 0;           // 0 = unknown
  std::uint64_t passes_done = 0;
  std::uint64_t passes_total = 0;           // 0 = unknown
  double elapsed_seconds = 0.0;
  double trials_per_second = 0.0;           // sliding window
  std::vector<query_state> queries;         // query order of the run
};

class sink {
 public:
  virtual ~sink() = default;
  virtual void on_snapshot(const snapshot &s) = 0;                    // called on the run thread, coalesced
  virtual void on_finish(const snapshot &last, bool interrupted) = 0; // exactly once per run
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
      interval_ = std::chrono::duration_cast<clock::duration>(
          std::chrono::duration<double>(1.0 / max_rate_hz));
      if (interval_.count() <= 0) {
        interval_ = clock::duration{1};
      }
    } else {
      unlimited_ = true;
    }
  }

  [[nodiscard]] bool active() const noexcept { return !sinks_.empty(); }

  template <class MakeSnapshot>
  void maybe_emit(MakeSnapshot &&make) {
    if (sinks_.empty()) {
      return;
    }
    const auto now = now_();
    if (emitted_ && !unlimited_ && now - last_ < interval_) {
      return;
    }
    snapshot s = make();
    last_ = now;
    emitted_ = true;
    broadcast(s);
  }

  // Bypasses the rate limit: delivers `last` to every sink and then on_finish, exactly once.
  void finish(snapshot last, bool interrupted) {
    if (sinks_.empty() || finished_) {
      return;
    }
    finished_ = true;
    broadcast(last);
    for (sink *s : sinks_) {
      s->on_finish(last, interrupted);
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

}  // namespace fuzzybits::runtime::progress
