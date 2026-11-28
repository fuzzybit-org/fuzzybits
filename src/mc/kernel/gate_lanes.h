/**
 * @file gate_lanes.h
 * @brief What the gate kernels (gate.h, cardinality.h) share: lanes, their loads and stores, and the launch
 *        geometry with its work-group order
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details A work-item evaluates one bitpack of one gate on GPUs, and kGateCpuLanes consecutive bitpacks as one
 * vector on CPUs, whose JIT does not vectorize across work-items. Wider lanes on GPUs measured no faster (32-bit
 * lanes were 2-20% slower, aligned 128-bit lanes tied). The range is one dimension of gates x work-groups per
 * gate, and the gate comes from the work-group index, so its fields are uniform across the group.
 */

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace fuzzybits::mc::kernel {

/// Order of a gate kernel's work-groups over (gate, block of bitpacks).
enum class gate_order {
    /// All blocks of a gate, then the next gate.
    gate_major,
    /// Every gate at a block, then the next block, so gates that share inputs read them from the cache.
    block_major,
};

/// Block-major on NVIDIA GPUs (1.4-2.4x over gate-major on the V100S, RTX 3070 and RTX 5060 Ti); gate-major
/// elsewhere: block-major was 2.6-3.6x slower on a Radeon Pro 5500M, and 0-7% slower on CPUs.
[[nodiscard]] inline gate_order gate_order_for(const sycl::device &device) {
    return device.get_backend() == sycl::backend::cuda ? gate_order::block_major : gate_order::gate_major;
}

/// The order for a kernel whose gates differ in input count (mixed_, the counted forms): block-major on CPUs
/// too. Their OpenMP runtime gives each thread a contiguous run of work-groups (a static schedule), so in
/// gate-major order one thread can get the widest gates while the others wait at the kernel's barrier; in
/// block-major order every thread gets each gate as often.
[[nodiscard]] inline gate_order gate_order_for(const sycl::device &device, const bool mixed_) {
    return mixed_ && device.is_cpu() ? gate_order::block_major : gate_order_for(device);
}

/// Bitpacks per work-item on CPUs.
inline constexpr std::uint32_t kGateCpuLanes = 8;

namespace gate_detail {

// Lanes go by reference and results through out-parameters, as in prng/bernoulli/simd.h: a vector passed by value
// would change the host calling convention (-Wpsabi), and 8- and 16-bit bitpacks promote to int in
// expressions, so every result is assigned back to its lane type.

template <typename T, int W> struct lanes_of {
    using type = T __attribute__((ext_vector_type(W)));
};
template <typename T> struct lanes_of<T, 1> {
    using type = T;
};

/// A lane at p. One bitpack is a typed load; a vector of them a copy (CPUs only).
template <typename V, typename bitpack_t_> [[gnu::always_inline]] inline void load(V &out, const bitpack_t_ *p) {
    if constexpr (std::is_same_v<V, bitpack_t_>) {
        out = *p;
    } else {
        __builtin_memcpy(&out, p, sizeof(V));
    }
}

template <typename V, typename bitpack_t_> [[gnu::always_inline]] inline void store(bitpack_t_ *p, const V &v) {
    if constexpr (std::is_same_v<V, bitpack_t_>) {
        *p = v;
    } else {
        __builtin_memcpy(p, &v, sizeof(V));
    }
}

} // namespace gate_detail

/// One dimension of gates x work-groups per gate, in either order.
template <std::uint32_t lanes_, std::size_t work_group_> class gate_geometry {
    std::uint32_t gates_ = 0;
    /// Bitpacks per gate per iteration (batch_size * bitpacks_per_batch).
    std::uint32_t bitpacks_ = 0;
    /// Work-items per gate, and work-groups per gate (rounded up, so that no work-group spans two gates).
    std::uint32_t items_per_gate_ = 0;
    std::uint32_t groups_per_gate_ = 0;
    bool block_major_ = false;

  public:
    struct position {
        std::uint32_t gate;
        /// The work-item's first bitpack in the gate's buffer.
        std::size_t first;
    };

    gate_geometry() = default;
    gate_geometry(const std::size_t gates, const std::size_t bitpacks, const gate_order order)
        : gates_(static_cast<std::uint32_t>(gates)), bitpacks_(static_cast<std::uint32_t>(bitpacks)), items_per_gate_((bitpacks_ + lanes_ - 1) / lanes_),
          groups_per_gate_(static_cast<std::uint32_t>((items_per_gate_ + work_group_ - 1) / work_group_)), block_major_(order == gate_order::block_major) {}

    [[nodiscard]] sycl::nd_range<1> range() const {
        return {sycl::range<1>{std::size_t{gates_} * groups_per_gate_ * work_group_}, sycl::range<1>{work_group_}};
    }
    [[nodiscard]] std::size_t gates() const { return gates_; }
    [[nodiscard]] std::uint32_t bitpacks() const { return bitpacks_; }
    [[nodiscard]] std::size_t items_per_gate() const { return std::size_t{groups_per_gate_} * work_group_; }

    /// The work-item's gate and first bitpack; false when it has none (past the end of its gate's buffer).
    [[gnu::always_inline]] bool locate(const sycl::nd_item<1> &item, position &at) const {
        const auto group = static_cast<std::uint32_t>(item.get_group(0));
        const std::uint32_t gate = block_major_ ? group % gates_ : group / groups_per_gate_;
        const std::uint32_t block = block_major_ ? group / gates_ : group - gate * groups_per_gate_;
        const auto item_idx = static_cast<std::uint32_t>(block * work_group_ + item.get_local_id(0));
        if (item_idx >= items_per_gate_) {
            return false;
        }
        at = {gate, std::size_t{item_idx} * lanes_};
        return true;
    }
};

} // namespace fuzzybits::mc::kernel
