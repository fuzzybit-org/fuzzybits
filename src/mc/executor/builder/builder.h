/**
 * @file builder.h
 * @brief The kernel builder: from the layers of a PDAG, as layer.h takes them, to the program a layer manager submits
 *        every pass
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details Per layer, the constants' bitpacks are written once, one kernel samples the basic events, and one kernel per
 * op code and gate kernel evaluates the gates; after the last layer, one kernel counts the tallied nodes' true trials.
 * Each kernel becomes a partition of the program: a queueable, submitted after the partitions whose outputs it reads,
 * and its description in the plan.
 *
 *     op.h           the op codes the builder computes nodes by (kernel::op_code, which extends ir::pc::op_code)
 *     layer.h        a layer as the builder takes it (the layer manager reads the PDAG into it)
 *     state.h        the build so far: every node's output and the partition that computes it, and the program
 *     program.h      what the builder makes: queueables, node blocks, tallies, and the plan
 *     basic_event.h  the basic events and constants of a layer
 *     gate.h         the gates of a layer, by kernel
 *     tally.h        the tallies
 *     launch.h       a kernel as a queueable
 *     dispatch.h     from a run-time choice to the kernel instantiated for it
 *     compat.h       SCRAM's PDAG gates in these op codes, for the layer manager; no builder file includes it
 */

#pragma once

#include "mc/event/node.h"
#include "mc/event/sample_shape.h"
#include "mc/executor/builder/basic_event.h"
#include "mc/executor/builder/gate.h"
#include "mc/executor/builder/layer.h"
#include "mc/executor/builder/op.h"
#include "mc/executor/builder/program.h"
#include "mc/executor/builder/state.h"
#include "mc/executor/builder/tally.h"
#include "mc/prng/stream.h"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <span>
#include <utility>

namespace fuzzybits::mc::executor::builder {

/// Builds a program for a queue and its arena, a layer at a time, then the tallies.
template <event::node_types_like T, typename size_t_> class program_builder {
  public:
    using index_type = typename T::index_type;

    /// A builder for `queue` and its `arena`, of nodes whose PDAG indices are below `indices`, sampled in `stream`.
    program_builder(sycl::queue &queue, const event::sample_shape<size_t_> &shape, const prng::stream &stream, event::output_arena<T> &arena,
                    const std::size_t indices)
        : state_(queue, shape, stream, arena, indices) {}

    /// The kernels of the next layer, layer `index`: every node it reads is in an earlier layer.
    program_builder &add(const layer<T> &nodes, const std::size_t index) {
        build_basic_events(state_, std::span<const event::basic_event<T>>(nodes.basic_events), index);
        for (const op_code op : kGateOps) {
            build_gates(state_, nodes, op, index);
        }
        return *this;
    }

    /// The kernel that tallies the nodes `nodes`, as layer `index`, after every layer of nodes.
    program_builder &tally(const std::span<const index_type> nodes, const std::size_t index) {
        build_tallies(state_, nodes, index);
        return *this;
    }

    [[nodiscard]] program<T> finish() && { return std::move(state_).finish(); }

  private:
    state<T, size_t_> state_;
};

} // namespace fuzzybits::mc::executor::builder
