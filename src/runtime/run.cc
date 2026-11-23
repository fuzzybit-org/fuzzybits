#include "runtime/runtime.h"

#include "mc/queue/layer_manager.h"
#include "mc/scheduler/convergence_controller.h"
#include "mc/stats/tally_node_map.h"
#include "pdag.h"
#include "settings.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fuzzybits::runtime {
namespace {

using bitpack_t = std::uint64_t;
using prob_t = std::double_t;
using size_t_ = std::size_t;

void collect_nodes(const std::shared_ptr<scram::core::Gate> &gate,
                   std::unordered_map<int, std::shared_ptr<scram::core::Node>> &nodes) {
    if (!gate || !nodes.emplace(gate->index(), gate).second) {
        return;
    }
    for (const auto &arg : gate->args<scram::core::Gate>()) {
        collect_nodes(arg.second, nodes);
    }
    for (const auto &arg : gate->args<scram::core::Variable>()) {
        nodes.emplace(arg.second->index(), arg.second);
    }
}

/// Builds `progress::snapshot`s for one `run_graph` call. Every field comes from tallies the run
/// loop already collected, so attaching a sink never changes sampling.
class sampling_progress {
  public:
    sampling_progress(const RunOptions &options, const std::vector<ReconstructedQuery> &queries,
                      std::uint64_t trials_per_pass, std::uint64_t passes_total)
        : emitter_(options.progress_sinks, options.progress_max_rate_hz), trials_per_pass_(trials_per_pass),
          passes_total_(passes_total), start_(clock::now()) {
        names_.reserve(queries.size());
        for (const auto &q : queries) {
            names_.push_back(q.node->index());
            query_names_.push_back(q.name);
        }
    }

    [[nodiscard]] bool active() const noexcept { return emitter_.active(); }

    /// Whether the next collection is worth doing: false when the snapshot would be coalesced away.
    [[nodiscard]] bool due() const { return emitter_.due(); }

    /// Emits at most once per coalescing interval. `tallies` must hold cumulative counters for every
    /// query node.
    void maybe_emit(const scram::mc::stats::TallyNodeMap &tallies, std::uint64_t passes_done) {
        if (!emitter_.active()) {
            return;
        }
        emitter_.maybe_emit([&] { return build_live(tallies, passes_done); });
    }

    /// Emits `last` regardless of the rate and then `on_finish`. Exceptions from the sinks are
    /// swallowed when the run is already unwinding, so the original error is not replaced.
    void finish(const scram::mc::stats::TallyNodeMap *tallies, std::uint64_t passes_done, bool interrupted) {
        if (!emitter_.active()) {
            return;
        }
        auto last = tallies ? build_final(*tallies, passes_done) : snapshot_without_tallies(passes_done);
        last.stage = interrupted ? progress::phase::sampling : progress::phase::done;
        last.elapsed_seconds = std::chrono::duration<double>(clock::now() - start_).count();
        if (!interrupted) {
            emitter_.finish(std::move(last), false);
            return;
        }
        try {
            emitter_.finish(std::move(last), true);
        } catch (...) {
        }
    }

    /// `finish(..., interrupted=true)`, for the RAII guard. Never throws.
    void finish_interrupted(const scram::mc::stats::TallyNodeMap *tallies, std::uint64_t passes_done) noexcept {
        try {
            finish(tallies, passes_done, /*interrupted=*/true);
        } catch (...) {
        }
    }

  private:
    using clock = progress::emitter::clock;

    [[nodiscard]] progress::snapshot snapshot_without_tallies(std::uint64_t passes_done) const {
        progress::snapshot s;
        s.stage = progress::phase::sampling;
        s.passes_done = passes_done;
        s.passes_total = passes_total_;
        s.trials_done = passes_done * trials_per_pass_;
        s.trials_total = passes_total_ * trials_per_pass_;
        s.elapsed_seconds = std::chrono::duration<double>(clock::now() - start_).count();
        s.queries.resize(query_names_.size());
        for (std::size_t i = 0; i < query_names_.size(); ++i) {
            s.queries[i].name = query_names_[i];
        }
        return s;
    }

    /// Fills the per-query states from `tallies`; the only difference between an intermediate and the
    /// final snapshot is the throughput value (resampled vs. the last delivered rate).
    [[nodiscard]] progress::snapshot build(const scram::mc::stats::TallyNodeMap &tallies, std::uint64_t passes_done,
                                           double trials_per_second) const {
        progress::snapshot s = snapshot_without_tallies(passes_done);
        for (std::size_t i = 0; i < names_.size(); ++i) {
            const auto &t = tallies.at(names_[i]).tally_stats;
            progress::query_state &q = s.queries[i];
            q.hits = static_cast<std::uint64_t>(t.num_one_bits);
            q.trials = static_cast<std::uint64_t>(t.total_bits);
            q.estimate = t.mean;
            q.ci_low = t.ci[0];
            q.ci_high = t.ci[1];
            const double width = t.ci[1] - t.ci[0];
            q.rel_half_width = t.mean != 0.0 && std::isfinite(width) ? width / (2.0 * t.mean)
                                                                     : std::numeric_limits<double>::quiet_NaN();
        }
        s.trials_per_second = trials_per_second;
        return s;
    }

    /// An intermediate snapshot: resamples the throughput window and remembers the result.
    [[nodiscard]] progress::snapshot build_live(const scram::mc::stats::TallyNodeMap &tallies,
                                                std::uint64_t passes_done) {
        last_rate_ = update_rate(passes_done * trials_per_pass_);
        return build(tallies, passes_done, last_rate_);
    }

    /// The final snapshot reuses the last delivered rate (see `update_rate`).
    [[nodiscard]] progress::snapshot build_final(const scram::mc::stats::TallyNodeMap &tallies,
                                                 std::uint64_t passes_done) const {
        return build(tallies, passes_done, last_rate_);
    }

    /// Trials per second over a sliding window of ~2 s (or the whole run when it is shorter).
    ///
    /// The window only advances when a snapshot is actually delivered, so the "whole run" case is two
    /// real samples apart. `finish()` does not call this: it reuses the last delivered rate, otherwise
    /// a final sample taken microseconds after the previous one would report ~0.
    double update_rate(std::uint64_t trials_done) {
        const auto now = clock::now();
        window_.emplace_back(now, trials_done);
        while (window_.size() > 2 && now - window_.front().first > std::chrono::seconds(2)) {
            window_.pop_front();
        }
        const auto &[first_time, first_trials] = window_.front();
        const double seconds = std::chrono::duration<double>(now - first_time).count();
        // Never extrapolate the first sample of a window (the delta would be 0 over 0 s).
        if (window_.size() < 2 || seconds <= 0.0 || trials_done < first_trials) {
            return 0.0;
        }
        last_rate_ = static_cast<double>(trials_done - first_trials) / seconds;
        return last_rate_;
    }

    progress::emitter emitter_;
    std::vector<int> names_;
    std::vector<std::string> query_names_;
    std::uint64_t trials_per_pass_ = 0;
    std::uint64_t passes_total_ = 0;
    clock::time_point start_;
    std::deque<std::pair<clock::time_point, std::uint64_t>> window_;
    double last_rate_ = 0.0;
};

} // namespace

namespace {
thread_local std::uint64_t fail_run_after_passes = 0;
} // namespace

namespace testing {

void fail_next_run_after_passes(std::uint64_t passes) { fail_run_after_passes = passes; }

} // namespace testing

std::vector<QueryResult> run_graph(scram::core::Pdag &pdag, const std::vector<ReconstructedQuery> &queries,
                                   const RunOptions &options) {
    if (options.trials == 0) {
        throw RuntimeError("run: --trials must be positive");
    }
    if (queries.empty()) {
        throw RuntimeError("run: no queries");
    }
    if (!pdag.root_ptr()) {
        throw RuntimeError("run: PDAG has no root");
    }

    apply_backend_mask(options.backend);

    scram::core::Settings settings;
    settings.num_trials(static_cast<std::double_t>(options.trials));
    settings.early_stop(false);
    settings.seed(options.seed);
    settings.ci_policy(scram::core::CIPolicy::kWald);
    settings.watch_mode(false);

    scram::mc::stats::TallyNodeMap monitored;
    for (const auto &q : queries) {
        if (!q.node) {
            throw RuntimeError("run: query '" + q.name + "' has no node");
        }
        monitored.insert_or_assign(q.node->index(), scram::mc::stats::TallyNode{
                                                        .status = scram::mc::stats::not_tracked,
                                                        .tally_stats = {},
                                                        .node = q.node,
                                                    });
    }

    try {
        scram::mc::queue::layer_manager<bitpack_t, prob_t, size_t_> manager(&pdag, static_cast<size_t_>(options.trials),
                                                                            monitored, /*requested_trials_per_pass=*/0);

        const auto total_passes = manager.shaper().TOTAL_ITERATIONS;
        if (total_passes == 0) {
            throw RuntimeError("run: sample shaper produced zero iterations");
        }
        const auto &shape = manager.shaper().SAMPLE_SHAPE;
        const auto trials_per_pass =
            static_cast<std::uint64_t>(shape.batch_size) * shape.bitpacks_per_batch * sizeof(bitpack_t) * 8;

        sampling_progress progress(options, queries, trials_per_pass, total_passes);

        // Every sink sees exactly one on_finish, including when the device fails or a result cannot be
        // built. `collected_pass` is the last pass whose tallies were folded into `monitored`, so the
        // interrupted snapshot pairs `passes_done * trials_per_pass` with per-query counts from the same
        // collection; the guard's finish never throws.
        std::uint64_t collected_pass = 0;
        progress::finish_guard guard([&progress, &monitored, &collected_pass] {
            progress.finish_interrupted(progress.active() && collected_pass > 0 ? &monitored : nullptr, collected_pass);
        });
        // One-shot test seam: a deterministic non-sink failure inside the active path.
        const auto fail_after = std::exchange(fail_run_after_passes, 0);

        if (!progress.active()) {
            manager.pass_wait_collect(monitored, total_passes, /*passes_between_waits=*/1);
            collected_pass = total_passes;
        } else {
            // Same device work as pass_wait_collect() with passes_between_waits=1, split so a snapshot can
            // be taken at the pass waits. Tallies are only read back when a snapshot is due; sampling and
            // the final tally are unchanged.
            for (std::uint64_t pass = 1; pass <= total_passes; ++pass) {
                manager.pass(1);
                scram::mc::queue::wait_sycl_queue(manager.queue());
                if (progress.due()) {
                    manager.collect_tallies(monitored);
                    collected_pass = pass;
                    progress.maybe_emit(monitored, pass);
                }
                if (fail_after != 0 && pass >= fail_after) {
                    throw RuntimeError("run: injected failure after pass " + std::to_string(pass));
                }
            }
            manager.collect_tallies(monitored);
            collected_pass = total_passes;
        }

        std::vector<QueryResult> results;
        results.reserve(queries.size());
        for (const auto &q : queries) {
            const auto &t = monitored.at(q.node->index()).tally_stats;
            QueryResult r;
            r.name = q.name;
            r.estimate = t.mean;
            r.ci_low = t.ci[0];
            r.ci_high = t.ci[1];
            r.n = static_cast<std::uint64_t>(t.total_bits);
            r.hits = static_cast<std::uint64_t>(t.num_one_bits);
            results.push_back(std::move(r));
        }
        progress.finish(&monitored, total_passes, /*interrupted=*/false);
        guard.complete();
        return results;
    } catch (const progress::interrupted &) {
        throw;
    } catch (const RuntimeError &) {
        throw;
    } catch (const sycl::exception &e) {
        throw RuntimeError(std::string("SYCL/AdaptiveCpp kernel failed: ") + e.what());
    } catch (const std::exception &e) {
        throw RuntimeError(std::string("Monte Carlo run failed: ") + e.what());
    }
}

std::vector<QueryResult> run_pcfb(const pcfb::ir::Circuit &circuit, const RunOptions &options) {
    apply_backend_mask(options.backend);
    auto reconstructed = pdag_from_circuit(circuit);
    return run_graph(*reconstructed.graph, reconstructed.queries, options);
}

namespace {

pcfb::ir::Circuit require_valid_circuit(pcfb::Result<pcfb::ir::Circuit> circuit, const char *op) {
    if (!circuit) {
        throw RuntimeError(std::string(op) + ": " + circuit.error().message);
    }
    require_implemented_queries(*circuit, op);
    const auto diags = pcfb::validate(*circuit);
    if (!diags.empty()) {
        throw InvalidPcfbError(
            std::string(op) + ": pcfb validation failed (" + std::to_string(diags.size()) + " diagnostics)", diags);
    }
    return std::move(*circuit);
}

} // namespace

pcfb::ir::Circuit load_pcfb(const std::filesystem::path &path) {
    return require_valid_circuit(pcfb::io::read(path), "load_pcfb");
}

pcfb::ir::Circuit load_pcfb(std::span<const std::byte> bytes) {
    return require_valid_circuit(pcfb::io::from_bytes(bytes), "load_pcfb");
}

std::vector<QueryResult> run_pcfb(const std::filesystem::path &path, const RunOptions &options) {
    apply_backend_mask(options.backend);
    return run_pcfb(load_pcfb(path), options);
}

std::vector<QueryResult> run_pcfb(std::span<const std::byte> bytes, const RunOptions &options) {
    apply_backend_mask(options.backend);
    return run_pcfb(load_pcfb(bytes), options);
}

struct Session::Impl {
    using bitpack_t = std::uint64_t;
    using prob_t = std::double_t;
    using size_t_ = std::size_t;
    using Manager = scram::mc::queue::layer_manager<bitpack_t, prob_t, size_t_>;

    ReconstructedPdag reconstructed;
    std::unique_ptr<Manager> manager;
    std::uint64_t pass_index = 0;
    bool closed = false;
};

Session::Session(ReconstructedPdag reconstructed, const std::string &backend, std::size_t requested_trials_per_pass,
                 std::vector<std::string> watch, std::size_t num_trials) {
    apply_backend_mask(backend);
    auto impl = std::make_unique<Impl>();
    impl->reconstructed = std::move(reconstructed);
    if (!impl->reconstructed.graph || !impl->reconstructed.graph->root_ptr()) {
        throw RuntimeError("session: reconstructed PDAG has no root");
    }
    if (impl->reconstructed.queries.empty()) {
        throw RuntimeError("session: no queries");
    }

    scram::mc::stats::TallyNodeMap tallies;
    for (const auto &q : impl->reconstructed.queries) {
        if (!q.node) {
            throw RuntimeError("session: query '" + q.name + "' has no node");
        }
        tallies.insert_or_assign(q.node->index(), scram::mc::stats::TallyNode{
                                                      .status = scram::mc::stats::not_tracked,
                                                      .tally_stats = {},
                                                      .node = q.node,
                                                  });
    }
    if (!watch.empty()) {
        std::unordered_map<int, std::shared_ptr<scram::core::Node>> nodes;
        collect_nodes(impl->reconstructed.graph->root_ptr(), nodes);
        for (int index : resolve_watch(impl->reconstructed, watch)) {
            auto it = nodes.find(index);
            if (it == nodes.end()) {
                throw RuntimeError("session: watched index " + std::to_string(index) + " is not in the PDAG");
            }
            tallies.insert_or_assign(index, scram::mc::stats::TallyNode{
                                                .status = scram::mc::stats::not_tracked,
                                                .tally_stats = {},
                                                .node = it->second,
                                            });
        }
    }
    try {
        impl->manager = std::make_unique<Impl::Manager>(impl->reconstructed.graph.get(), num_trials, tallies,
                                                        requested_trials_per_pass);
    } catch (const RuntimeError &) {
        throw;
    } catch (const sycl::exception &e) {
        throw RuntimeError(std::string("SYCL/AdaptiveCpp kernel failed: ") + e.what());
    } catch (const std::exception &e) {
        throw RuntimeError(std::string("Monte Carlo session failed: ") + e.what());
    }
    impl_ = std::move(impl);
}

Session::Session(Session &&) noexcept = default;
Session &Session::operator=(Session &&) noexcept = default;
Session::~Session() = default;

Session Session::from_circuit(const pcfb::ir::Circuit &circuit, const std::string &backend,
                              std::size_t requested_trials_per_pass, std::vector<std::string> watch,
                              std::size_t num_trials) {
    const auto diags = pcfb::validate(circuit);
    if (!diags.empty()) {
        throw InvalidPcfbError("session: pcfb validation failed (" + std::to_string(diags.size()) + " diagnostics)",
                               diags);
    }
    return Session(pdag_from_circuit(circuit), backend, requested_trials_per_pass, std::move(watch), num_trials);
}

Session Session::from_bytes(std::span<const std::byte> bytes, const std::string &backend,
                            std::size_t requested_trials_per_pass, std::vector<std::string> watch,
                            std::size_t num_trials) {
    return from_circuit(load_pcfb(bytes), backend, requested_trials_per_pass, std::move(watch), num_trials);
}

Session Session::from_path(const std::filesystem::path &path, const std::string &backend,
                           std::size_t requested_trials_per_pass, std::vector<std::string> watch,
                           std::size_t num_trials) {
    return from_circuit(load_pcfb(path), backend, requested_trials_per_pass, std::move(watch), num_trials);
}

scram::core::Pdag &Session::pdag() {
    if (!impl_ || impl_->closed) {
        throw RuntimeError("session: closed");
    }
    return *impl_->reconstructed.graph;
}

const scram::core::Pdag &Session::pdag() const {
    if (!impl_ || impl_->closed) {
        throw RuntimeError("session: closed");
    }
    return *impl_->reconstructed.graph;
}

const std::vector<ReconstructedQuery> &Session::queries() const {
    if (!impl_ || impl_->closed) {
        throw RuntimeError("session: closed");
    }
    return impl_->reconstructed.queries;
}

const GraphMeta &Session::meta() const {
    if (!impl_ || impl_->closed) {
        throw RuntimeError("session: closed");
    }
    return impl_->reconstructed.meta;
}

SampleShape Session::sample_shape() const {
    if (!impl_ || impl_->closed || !impl_->manager) {
        throw RuntimeError("session: closed");
    }
    const auto &shape = impl_->manager->shaper().SAMPLE_SHAPE;
    SampleShape out;
    out.batch_size = shape.batch_size;
    out.bitpacks_per_batch = shape.bitpacks_per_batch;
    out.bits_per_bitpack = sizeof(Impl::bitpack_t) * 8;
    out.trials_per_pass =
        static_cast<std::uint64_t>(shape.batch_size) * shape.bitpacks_per_batch * out.bits_per_bitpack;
    out.total_iterations = impl_->manager->shaper().TOTAL_ITERATIONS;
    return out;
}

bool Session::closed() const { return !impl_ || impl_->closed; }

PassResult Session::step(const std::size_t passes) {
    if (!impl_ || impl_->closed || !impl_->manager) {
        throw RuntimeError("session: closed");
    }
    if (passes == 0) {
        throw RuntimeError("session: passes must be at least 1");
    }
    scram::mc::stats::TallyNodeMap tallies;
    try {
        impl_->manager->pass_wait_collect(tallies, passes, 0);
    } catch (const RuntimeError &) {
        throw;
    } catch (const sycl::exception &e) {
        throw RuntimeError(std::string("SYCL/AdaptiveCpp kernel failed: ") + e.what());
    } catch (const std::exception &e) {
        throw RuntimeError(std::string("Monte Carlo session failed: ") + e.what());
    }
    impl_->pass_index += passes;

    PassResult out;
    out.pass_index = impl_->pass_index;
    out.trials_per_pass = sample_shape().trials_per_pass;
    out.target_index = impl_->reconstructed.queries.front().node_index;
    out.nodes.reserve(tallies.size());
    for (const auto &[index, node_tally] : tallies) {
        out.nodes.push_back(PassTally{
            .index = index,
            .num_one_bits = static_cast<std::uint64_t>(node_tally.tally_stats.num_one_bits),
            .total_bits = static_cast<std::uint64_t>(node_tally.tally_stats.total_bits),
        });
    }
    return out;
}

void Session::close() {
    if (!impl_) {
        return;
    }
    impl_->manager.reset();
    impl_->closed = true;
}

} // namespace fuzzybits::runtime
