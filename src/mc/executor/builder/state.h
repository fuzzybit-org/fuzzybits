/**
 * @file state.h
 * @brief The builder's state: where every node built so far has its output and which partition computes it, and the
 *        program as built so far
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/event/basic_event.h"
#include "mc/event/node.h"
#include "mc/event/sample_shape.h"
#include "mc/event/tally.h"
#include "mc/executor/builder/launch.h"
#include "mc/executor/builder/program.h"
#include "mc/executor/descriptor/partition.h"
#include "mc/executor/queueable.h"
#include "mc/prng/stream.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::mc::executor::builder {

/// The partitions a partition reads, gathered as it reads its inputs: their ids, ascending and distinct.
class dependencies {
  public:
    void add(const std::size_t partition) {
        if (const auto at = std::ranges::lower_bound(ids_, partition); at == ids_.end() || *at != partition) {
            ids_.insert(at, partition);
        }
    }

    [[nodiscard]] const std::vector<std::size_t> &ids() const noexcept { return ids_; }

  private:
    std::vector<std::size_t> ids_;
};

/**
 * @brief What the builders read and extend: the queue, the pass's shape, the stream and the arena they build for, every
 *        built node's output, and the program
 *
 * @details A node is built once: by a partition, which computes its output every pass, or when its arena slot is
 * allocated, for an output that never changes. A partition reads only nodes built before it, so the partitions are in
 * an order an in-order queue can run them in, which descriptor::plan::add checks again.
 */
template <event::node_types_like T, typename size_t_> class state {
  public:
    using index_type = typename T::index_type;
    using extent_type = typename T::extent_type;

    /// The state of a build for `queue` and its arena, of nodes whose PDAG indices are below `indices`.
    state(sycl::queue &queue, const event::sample_shape<size_t_> &shape, const prng::stream &stream, event::output_arena<T> &arena, const std::size_t indices)
        : queue_(queue), device_(queue.get_device()), shape_(shape), stream_(stream), arena_(arena), outputs_(indices) {}

    [[nodiscard]] sycl::queue &queue() const noexcept { return queue_; }
    [[nodiscard]] const sycl::device &device() const noexcept { return device_; }
    [[nodiscard]] const prng::stream &stream() const noexcept { return stream_; }
    [[nodiscard]] event::output_arena<T> &arena() const noexcept { return arena_; }

    /// The arena slot of node `index`'s output; the partition that computes it, if any, is added to `reads`. Throws
    /// std::logic_error for a node that nothing built so far writes: a kernel reads only the outputs of kernels before
    /// it.
    [[nodiscard]] extent_type read(const index_type index, dependencies &reads) const {
        const output *const node = built(index);
        if (node == nullptr) {
            throw std::logic_error("kernel builder: PDAG node " + std::to_string(index) + " is read before any kernel computes it");
        }
        if (node->partition != output::kWritten) {
            reads.add(node->partition);
        }
        return node->slot;
    }

    /**
     * @brief Adds `kernel`, which computes `block`'s nodes or counts their trials, as the next partition: after the
     *        partitions in `reads`. Returns its id.
     *
     * @details `description` says what the builder knows of the partition (its layer, kind, operation and gates'
     * inputs); the state adds the rest: its id, nodes, dependencies, launch and pass shape.
     */
    template <typename kernel_type, class Block>
    std::size_t add(const kernel_type &kernel, const Block &block, const dependencies &reads, descriptor::partition description) {
        std::vector<std::shared_ptr<queueable_base>> after;
        after.reserve(reads.ids().size());
        for (const std::size_t id : reads.ids()) {
            after.push_back(program_.queueables[id]);
        }
        description.nodes.reserve(block.size());
        for (const auto &record : block.records()) {
            description.nodes.push_back(static_cast<std::int32_t>(record.id_in_pdag));
        }
        description.dependencies = reads.ids();
        description.shape = descriptor::launch::of(kernel.get_range(), block.size());
        description.batches = shape_.batch_size;
        description.bitpacks_per_batch = shape_.bitpacks_per_batch;
        description.bitpack_bytes = sizeof(typename T::bitpack_type);
        const std::size_t id = program_.plan.add(std::move(description));
        program_.queueables.push_back(queueable_of(queue_, kernel, after));
        return id;
    }

    /// `block`'s nodes are computed by partition `partition`; keeps the block. Throws std::logic_error for a node built
    /// already.
    template <class Block> void computed(Block &&block, const std::size_t partition) {
        for (const auto &node : block.nodes()) {
            build(node.record.id_in_pdag, node.slot, partition);
        }
        program_.blocks.keep(std::forward<Block>(block));
    }

    /// `block`'s nodes have their outputs in their slots already, and no kernel computes them; keeps the block. Throws
    /// std::logic_error for a node built already.
    void written(event::constant_event_block<T> &&block) { computed(std::move(block), output::kWritten); }

    /// `block` holds the tallies of its nodes; keeps the block. Throws std::logic_error for a node tallied already.
    void tallied(event::tally_block<T> &&block) {
        const std::size_t at = program_.blocks.tallies.size();
        const auto records = block.records();
        for (std::size_t position = 0; position < records.size(); ++position) {
            if (!program_.tallies.emplace(records[position].id_in_pdag, tally_location{.block = at, .position = position}).second) {
                throw std::logic_error("kernel builder: PDAG node " + std::to_string(records[position].id_in_pdag) + " is tallied twice");
            }
        }
        program_.blocks.keep(std::move(block));
    }

    /// The program, once every layer and the tallies are built.
    [[nodiscard]] program<T> finish() && { return std::move(program_); }

  private:
    /// A node's output: its arena slot and the partition that computes it, or kWritten when its slot was written as it
    /// was allocated. Indexed by PDAG index, which a PDAG gives its nodes densely.
    struct output {
        static constexpr std::uint32_t kWritten = std::numeric_limits<std::uint32_t>::max();
        static constexpr extent_type kUnbuilt = std::numeric_limits<extent_type>::max();
        extent_type slot = kUnbuilt;
        std::uint32_t partition = kWritten;
    };

    [[nodiscard]] const output *built(const index_type index) const noexcept {
        const auto i = static_cast<std::size_t>(index);
        return index >= 0 && i < outputs_.size() && outputs_[i].slot != output::kUnbuilt ? &outputs_[i] : nullptr;
    }

    void build(const index_type index, const extent_type slot, const std::size_t partition) {
        if (index < 0 || static_cast<std::size_t>(index) >= outputs_.size()) {
            throw std::logic_error("kernel builder: PDAG node " + std::to_string(index) + " is outside the indices it was given");
        }
        output &node = outputs_[static_cast<std::size_t>(index)];
        if (node.slot != output::kUnbuilt) {
            throw std::logic_error("kernel builder: PDAG node " + std::to_string(index) + " is computed by two kernels");
        }
        node = {.slot = slot, .partition = static_cast<std::uint32_t>(partition)};
    }

    sycl::queue &queue_;
    sycl::device device_;
    const event::sample_shape<size_t_> &shape_;
    const prng::stream &stream_;
    event::output_arena<T> &arena_;
    std::vector<output> outputs_;
    program<T> program_;
};

} // namespace fuzzybits::mc::executor::builder
