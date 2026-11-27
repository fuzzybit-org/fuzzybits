#pragma once

#include <sycl/sycl.hpp>
#include "mc/event/node.tpp"
#include "mc/event/sample_shape.h"


namespace scram::mc::event {
/**
 * @struct basic_event
 * @brief Represents a basic event with associated probability and identification
 *
 * @details Basic events are the leaf nodes of the computation graph and serve as inputs
 * to higher-level gate operations. They are processed in parallel using SYCL
 * kernels to generate random samples based on their failure probabilities.
 *
 * @par Probability-threshold mapping
 * The kernels never compare floating-point probabilities. Each basic event stores
 * `threshold` \f$T=\min(\lfloor p\cdot2^{64}\rfloor, 2^{64}-1)\f$, computed exactly from the
 * binary64 \f$p\f$ by `pcfb::probability_to_threshold`, and a trial is true iff its 64-bit uniform
 * is below \f$T\f$ (prng/bernoulli.h). \f$p=1\f$ sets `always_true` instead, since no 64-bit
 * threshold is always satisfied. Resolution is \f$2^{-64}\f$: every \f$p\ge2^{-64}\f$ samples at a
 * nonzero rate (#17).
 *
 * @tparam bitpack_t_ Integer type for bit-packed result storage
 * * @tparam index_t_ Signed integer type for the node index.
 *
 * @note Probability values should be in the range [0.0, 1.0]
 * @note Index values must be unique within the computation graph
 *
 * @example
 * @code
 * // Create a basic event with 0.01 failure probability
 * basic_event<double, std::uint64_t> pump_failure;
 * pump_failure.threshold = pcfb::probability_to_threshold(0.01);
 * pump_failure.index = 42;
 * pump_failure.buffer = working_set<std::size_t, std::uint64_t>::smart_alloc_device<std::uint64_t>(queue,
 * num_bitpacks);
 * @endcode
 */
// TODO:: index_t should have no default type
template <typename bitpack_t_, typename pdag_index_t_> struct basic_event : node<bitpack_t_, pdag_index_t_> {
    /// @brief floor(p * 2^64): a trial is true iff its 64-bit uniform is below it
    std::uint64_t threshold{};
};

// TODO:: index_t should have no default type
template <typename bitpack_t_, typename pdag_index_t_>
inline std::ostream &operator<<(std::ostream &os, const basic_event<bitpack_t_, pdag_index_t_> &be) {
    os << "index= " << be.index << "  |  "
       << "p(double)= " << std::scientific << std::setprecision(6)
       << pcfb::threshold_to_probability(be.threshold) << "  |  "
       << "threshold= " << be.threshold;
    return os;
}

/**
 * @brief Specialized node_block for <code>basic_event</code> that also owns the
 * contiguous device buffer block used for bit-packed samples.
 *
 * The <code>buffers</code> pointer is the *sole* allocation for all bit-packed
 * outputs of all events in this block.  Each individual event’s
 * <code>buffer</code> member points somewhere inside this big block at an
 * offset of <code>event_index * bitpacks_per_event</code>.
 */
// TODO:: stride/bitpacks_per_type should be templated, and have no default type
template <typename bitpack_t_, typename pdag_index_t_, typename block_size_t_>
struct basic_event_block : public node_block<basic_event<bitpack_t_, pdag_index_t_>, block_size_t_> {
    bitpack_t_ *buffers = nullptr;      ///< single contiguous device allocation
    std::size_t bitpacks_per_event = 0; ///< stride between successive event buffers
};

/**
 * @brief Factory that allocates and initializes a contiguous block of
 * <code>basic_event</code> objects *and* their shared output-buffer block.
 *
 * The layout mimics the one previously produced by <code>create_basic_events</code>
 * but returns a richer wrapper that makes ownership explicit and therefore
 * easier to free safely.
 */
// TODO:: index_t should have no default type
// TODO:: num_bitpacks should be templated, and should have no default type
template <typename bitpack_t_, typename pdag_index_t_, typename block_size_t_>
[[nodiscard]]
basic_event_block<bitpack_t_, pdag_index_t_, block_size_t_>
create_basic_event_block(const sycl::queue &queue,
                         const std::vector<std::pair<pdag_index_t_, std::double_t>> &indexed_probabilities,
                         const auto num_bitpacks) {
    // TODO:: derive from size_t_
    const auto num_events = indexed_probabilities.size();

    // 1. Allocate host-visible node array (USM shared memory).
    using event_t = basic_event<bitpack_t_, pdag_index_t_>;
    //TODO:: This value should be device-scoped once we support multiple devices
    using cache_line_bytes = sycl::info::device::global_mem_cache_line_size;
    const auto align = queue.get_device().get_info<cache_line_bytes>();
    event_t *events = sycl::aligned_alloc_shared<event_t>(align, num_events, queue);

    // 2. Allocate device-side bit-packed buffer for *all* events.
    const auto buffer_bytes = num_events * num_bitpacks;
    bitpack_t_ *buffer_block = sycl::aligned_alloc_device<bitpack_t_>(align, buffer_bytes, queue);

    // 3. Populate per-event fields.
    for (auto i = 0; i < num_events; ++i) {
        events[i].index = indexed_probabilities[i].first;
        events[i].threshold = indexed_probabilities[i].second;
        events[i].buffer = buffer_block + i * num_bitpacks;
        LOG(DEBUG4) << events[i];
    }

    // 4. Wrap everything in a basic_event_block and return.
    basic_event_block<bitpack_t_, pdag_index_t_, block_size_t_> blk;
    blk.data = events;
    blk.count = num_events;
    blk.buffers = buffer_block;
    blk.bitpacks_per_event = num_bitpacks;
    return blk;
}

/**
 * @brief Releases all device memory owned by a <code>basic_event_block</code>.
 *
 * After the call the block is zeroed out so that double-free / use-after-free
 * errors are easier to spot during debugging.
 */
template <typename prob_t_, typename bitpack_t_, typename index_t_ = int32_t>
void destroy_basic_event_block(const sycl::queue &queue, basic_event_block<prob_t_, bitpack_t_, index_t_> &blk) {
    if (blk.buffers) {
        sycl::free(blk.buffers, queue);
        blk.buffers = nullptr;
    }
    if (blk.data) {
        sycl::free(blk.data, queue);
        blk.data = nullptr;
    }
    blk.count = 0;
    blk.bitpacks_per_event = 0;
}
}