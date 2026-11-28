/*
 * Copyright (C) 2025 Arjun Earthperson
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "mc/event/node.h"
#include "mc/executor/builder/builder.h"
#include "mc/executor/descriptor/execution_plan.h"
#include "mc/executor/layer_manager.h"
#include "mc/stats/ci_utils.h"

namespace fuzzybits::mc::executor {

template <typename bitpack_t_, typename prob_t_, typename size_t_>
builder::program<typename layer_manager<bitpack_t_, prob_t_, size_t_>::node_types>
layer_manager<bitpack_t_, prob_t_, size_t_>::build_program(const stats::TallyNodeMap &to_tally) {
    builder::program_builder<node_types, size_t_> build(queue_, sample_shape_, stream_, arena_, graph_.indices());
    for (std::size_t layer = 0; layer < graph_.layers(); ++layer) {
        build.add(graph_.layer<node_types>(layer), layer);
    }
    // The tallies read any layer's nodes, so they are a layer of their own after the last.
    build.tally(compat::tallied(to_tally), graph_.layers());
    return std::move(build).finish();
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
event::tally_estimate layer_manager<bitpack_t_, prob_t_, size_t_>::fetch_tally_for_event_with_index(const index_t_ evt_idx) {
    // The kernel counts the true trials only; the trials, and the estimate, are the host's.
    event::tally_estimate estimate{.num_one_bits = static_cast<std::size_t>(program_.ones(evt_idx)), .total_bits = static_cast<std::size_t>(total_bits())};
    stats::populate_point_estimates(estimate);
    return estimate;
}

template <typename bitpack_t_, typename prob_t_, typename size_t_> std::size_t layer_manager<bitpack_t_, prob_t_, size_t_>::node_count() const {
    return graph_.nodes();
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
descriptor::kernel_dag layer_manager<bitpack_t_, prob_t_, size_t_>::describe_kernel_dag() const {
    descriptor::kernel_dag dag = graph_.kernel_dag();
    dag.batch_size = sample_shape_.batch_size;
    dag.bitpacks_per_batch = sample_shape_.bitpacks_per_batch;
    dag.total_iterations = sample_shaper_.TOTAL_ITERATIONS;
    return dag;
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
typename layer_manager<bitpack_t_, prob_t_, size_t_>::execution_plan_description layer_manager<bitpack_t_, prob_t_, size_t_>::describe_execution_plan() const {
    return descriptor::execution_plan_of<bitpack_t_>(program_.plan, queue_.get_device(), sample_shaper_.describe(), node_count(),
                                                     graph_.describe(program_.plan));
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
[[gnu::always_inline]] sycl::queue layer_manager<bitpack_t_, prob_t_, size_t_>::single_pass() {
    // Pass p samples bitpacks [p·B, (p + 1)·B) of each node's run; their indices must stay below the counter's
    // prng::kMaxBitpacks, and the pass index below 2^32.
    const std::uint64_t end = (std::uint64_t{passes_} + 1) * sample_shape_.num_bitpacks();
    if (passes_ == std::numeric_limits<std::uint32_t>::max() || end > prng::kMaxBitpacks) {
        throw std::overflow_error("layer_manager: pass " + std::to_string(passes_) + " would run past the Philox counter's bitpack indices");
    }
    for (const auto &queueable : program_.queueables) {
        queueable->submit(passes_);
    }
    ++passes_;
    return queue_;
}

template <typename bitpack_t_, typename prob_t_, typename size_t_> sycl::queue layer_manager<bitpack_t_, prob_t_, size_t_>::pass(const size_t count) {
    for (size_t i = 0; i < count; i++) {
        single_pass();
    }
    return queue_;
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
event::tally_estimate layer_manager<bitpack_t_, prob_t_, size_t_>::single_pass_and_tally(const index_t_ evt_idx) {
    single_pass();
    wait_sycl_queue(queue_);
    return fetch_tally_for_event_with_index(evt_idx);
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
stats::TallyNodeMap &layer_manager<bitpack_t_, prob_t_, size_t_>::pass_wait_collect(stats::TallyNodeMap &stats, const std::size_t total_passes,
                                                                                    const std::size_t passes_between_waits) {
    /*
     * The purpose of this routine is to
     *  1. enqueue `total_passes` Monte-Carlo passes, potentially amounting to
     *     hundreds of thousands of kernel launches (10k+ queueables per pass!),
     *  2. periodically wait for the device to drain the queue so that we do not
     *     accumulate an unbounded number of in-flight kernels / SYCL events, and
     *  3. finally aggregate the resulting tallies on the host.
     *
     *  `passes_between_waits` controls the waiting cadence:
     *      – 0   : never wait in-between; we only wait once at the end.
     *      – n>0 : wait after every *n* passes (or sooner if less than *n*
     *              passes remain).
     */

    if (total_passes == 0) {
        throw std::invalid_argument("pass_wait_collect: total_passes must be greater than 0");
    }

    // A value of 0 means "wait only after the final batch".
    const auto interval = passes_between_waits == 0 ? total_passes : passes_between_waits;

    auto passes_remaining = total_passes;
    while (passes_remaining > 0) {
        const auto batch_size = std::min(interval, passes_remaining);

        // Enqueue `batch_size` passes without blocking.
        pass(batch_size);

        // Flush the device so we do not let the queue grow without bound.
        // Do not wait_and_throw: AdaptiveCpp's handler + ~queue double-throw
        // terminates. wait_sycl_queue records async errors and rethrows once.
        wait_sycl_queue(queue_);

        passes_remaining -= batch_size;
    }

    // Once all passes have completed on the device we can safely harvest the
    // tallies and fold them into the caller-supplied statistics map.
    return collect_tallies(stats);
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
stats::TallyNodeMap &layer_manager<bitpack_t_, prob_t_, size_t_>::collect_tallies(stats::TallyNodeMap &stats) {
    // The trials are those of every pass submitted, so the counts must be those of every pass too: wait for the
    // device first (a no-op on the idle queue every current caller leaves).
    wait_sycl_queue(queue_);
    const auto trials = static_cast<std::size_t>(total_bits());
    for (const auto &[index, location] : program_.tallies) {
        stats[index].tally_stats.update(static_cast<std::size_t>(program_.ones(location)), trials);
    }
    return stats;
}

template <typename bitpack_t_, typename prob_t_, typename size_t_> layer_manager<bitpack_t_, prob_t_, size_t_>::~layer_manager() {
    // The members free their USM after this body: wait for every kernel that may still use it. async_error_drop_
    // drops the recorded asynchronous errors once they have. A destructor must not throw.
    try {
        queue_.wait();
    } catch (...) {
    }
}

/// The MEF event behind PDAG node `event_id`: only a Variable has one, its BasicEvent (compat::graph::mef_event_of).
template <typename bitpack_t_, typename prob_t_, typename size_t_>
const compat::mef_event *layer_manager<bitpack_t_, prob_t_, size_t_>::get_mef_event(const index_t_ event_id) const {
    return graph_.mef_event_of(event_id);
}

template <typename bitpack_t_, typename prob_t_, typename size_t_>
layer_manager<bitpack_t_, prob_t_, size_t_>::layer_manager(compat::pdag *pdag, const size_t_ num_trials, const stats::TallyNodeMap &to_tally,
                                                           const std::size_t requested_trials_per_pass, const prng::stream stream)
    : stream_(stream), graph_(pdag) {
    sample_shaper_ = sample_shaper<bitpack_t_>(queue_, num_trials, node_count(), requested_trials_per_pass);
    sample_shape_ = sample_shaper_.SAMPLE_SHAPE;

    // One arena slot per node: the sample shaper sized the pass so that every node's output fits one allocation.
    arena_ = event::output_arena<node_types>::create(queue_, node_count(), sample_shape_.num_bitpacks());

    // The kernels of every layer, then the tallies. The queue is in order and submissions carry no dependency events,
    // so every kernel comes after the kernels it reads (descriptor::plan::add checks it).
    program_ = build_program(to_tally);

    compat::debug2([this](std::ostream &log) { log << sample_shaper_; });
}

} // namespace fuzzybits::mc::executor