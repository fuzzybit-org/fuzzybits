/**
 * @file tally.h
 * @brief The tallies: one kernel that adds each pass's true trials of the tallied nodes to their counts
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/event/tally.h"
#include "mc/executor/builder/state.h"
#include "mc/executor/descriptor/partition.h"
#include "mc/kernel/tally.h"

#include <cstddef>
#include <span>
#include <vector>

namespace fuzzybits::mc::executor::builder {

/// One kernel, of layer `layer_index`, that tallies the nodes `nodes`. Each tally reads the arena slot of its node, after the
/// kernel that computes it. The kernel counts only: the host derives the trials and the estimate.
///
/// @throws std::logic_error for a node that no kernel computes, or that is tallied twice
template <event::node_types_like T, typename size_t_>
void build_tallies(state<T, size_t_> &build, const std::span<const typename T::index_type> nodes, const std::size_t layer_index) {
    if (nodes.empty()) {
        return;
    }
    std::vector<event::tally<T>> records;
    records.reserve(nodes.size());
    dependencies reads;
    for (const auto index : nodes) {
        records.emplace_back(index, build.read(index, reads));
    }
    auto block = event::tally_block<T>::create(build.queue(), build.arena().view(), records);
    (void)build.add(kernel::tally<T>(block.view(), build.device()), block, reads,
                    {.layer = layer_index, .kind = descriptor::partition_kind::tally, .operation = "popcount"});
    build.tallied(std::move(block));
}

} // namespace fuzzybits::mc::executor::builder
