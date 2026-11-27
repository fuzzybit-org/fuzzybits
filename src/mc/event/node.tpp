/**
 * @file node.tpp
 * @brief SYCL-based node structures for parallel graph computation
 * @author Arjun Earthperson
 * @date 2025
 *
 * @details This file defines the core data structures used in SYCL-based parallel
 * computation of probabilistic directed acyclic graphs (PDAGs). It provides template
 * structures for nodes, basic events, gates, and tally events, along with factory
 * functions for efficient device memory allocation and management.
 *
 * The structures are designed for optimal performance on SYCL devices, using
 * bit-packed representations for efficient memory utilization and parallel processing.
 * All memory allocations are performed using SYCL unified shared memory for seamless
 * host-device data access.
 */

#pragma once

#include "logger.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <vector>

#include "mc/event/sample_shape.h"
#include "pcfb/threshold.h"

namespace scram::mc::event {

/**
 * @struct node
 * @brief Base structure for all computation nodes in the graph
 *
 * @details This is the fundamental building block for all node types in the
 * computation graph. It provides a device-accessible buffer for storing
 * computation results in a bit-packed format for memory efficiency.
 *
 * @tparam bitpack_t_ Integer type used for bit-packed data storage
 *
 * @note This structure is designed for SYCL device access
 * @note All derived structures inherit this buffer-based design
 *
 * @example
 * @code
 * // Basic node with 64-bit packing
 * node<std::uint64_t> basic_node;
 * basic_node.buffer = working_set<std::size_t, std::uint64_t>::smart_alloc_device<std::uint64_t>(queue, 1024);
 * @endcode
 */
template <typename bitpack_t_, typename pdag_index_t_> struct node {
    /// @brief Unique identifier for this event within the computation graph
    pdag_index_t_ id_in_pdag;

    /// TODO:: replace with index into the block-allocated event bitpack buffer.
    /// @brief Device-accessible buffer for storing computation results in bit-packed format
    bitpack_t_ *buffer;
};

/**
 * @brief Thin wrapper describing one contiguous allocation of Node objects.
 *
 * A node_block owns (or at least refers to) a *single* contiguous allocation
 * that stores <code>count</code> nodes of type <code>node_t</code>.  The core
 * motivation is to make the memory-layout property explicit so that callers
 * can reason about lifetime and de-allocation without having to keep track
 * of individual interior pointers.
 *
 * The wrapper itself is intentionally minimal – just a pointer and a size –
 * because different node types sometimes need extra bookkeeping (e.g. a
 * separate buffer block for `basic_event`).  Specializations or derived
 * wrappers can extend it with such fields.
 */
// TODO:: size_t should be templated, and have no default type
template <typename node_t_, typename block_size_t_> struct node_block {
    node_t_ *data = nullptr; ///< first element of the contiguous allocation
    block_size_t_ count = 0;  ///< number of valid nodes in <code>data</code>

    [[nodiscard]] node_t_ &operator[](block_size_t_ i) { return data[i]; }
    [[nodiscard]] const node_t_ &operator[](block_size_t_ i) const { return data[i]; }
};

} // namespace scram::mc::event
