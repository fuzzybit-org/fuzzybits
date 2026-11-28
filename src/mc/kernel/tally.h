/**
 * @file tally.h
 * @brief SYCL kernel that adds the true trials of one pass of each tallied node to its tally
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details The range is one dimension of tallies x work-groups per tally, and the tally comes from the work-group
 * index, so no work-group spans two tallies and a tally's fields are uniform across its groups. A work-item adds
 * the popcounts of a strided run of the node's bitpacks: at most kMaxRun of them, fewer when the tallies are too
 * small to keep every compute unit busy, as in the strided sampling kernel, kernel/sampler/strided.h (a run of one bitpack
 * per work-item on small passes, where a long run would make the tally's latency that of a few work-groups' loops). The
 * work-group reduces the sums in 32 bits (Metal's group reductions reject 64-bit integers; a group's sum is at most
 * kWorkGroup x kMaxRun x 64, far below 2^32), and its first work-item adds the total to the tally's 64-bit count with
 * one relaxed atomic. Every work-item reaches the reduction, including those past the end of the node's bitpacks,
 * which add nothing.
 *
 * The kernel writes counts only. The number of trials is passes x bitpacks per node x bits per bitpack, which the
 * host knows, and the host derives the estimate (stats/ci_utils.h).
 */

#pragma once

#include "mc/event/tally.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace fuzzybits::mc::kernel {

template <event::node_types_like T> class tally {
    using bitpack_type = typename T::bitpack_type;

  public:
    static constexpr std::size_t kWorkGroup = 256;
    /// Bitpacks per work-item at most, which bounds a work-group's 32-bit sum.
    static constexpr std::uint32_t kMaxRun = 32;
    static_assert(std::uint64_t{kWorkGroup} * kMaxRun * 64 < (std::uint64_t{1} << 32));

  private:
    event::tally_block_view<T> tallies_;
    /// Work-items per tally: a whole number of work-groups, each work-item reading a run of at most kMaxRun bitpacks.
    std::uint32_t items_per_tally_ = 0;
    std::uint32_t groups_per_tally_ = 0;

  public:
    tally(const event::tally_block_view<T> &tallies, const sycl::device &device) : tallies_(tallies) {
        // Up to kMaxRun bitpacks per work-item, fewer when there is too little work to keep every compute unit busy
        // (2048 work-items per unit bounds what a GPU keeps in flight).
        const std::uint64_t bitpacks = tallies_.bitpacks_per_node();
        const std::uint64_t units = std::max<std::uint64_t>(device.get_info<sycl::info::device::max_compute_units>(), 1);
        const std::uint64_t total = std::uint64_t{tallies_.size()} * bitpacks;
        const std::uint64_t run = std::clamp<std::uint64_t>(total / (units * 2048), 1, kMaxRun);
        const std::uint64_t items = (bitpacks + run - 1) / run;
        groups_per_tally_ = static_cast<std::uint32_t>(std::max<std::uint64_t>((items + kWorkGroup - 1) / kWorkGroup, 1));
        items_per_tally_ = static_cast<std::uint32_t>(groups_per_tally_ * kWorkGroup);
    }

    [[nodiscard]] sycl::nd_range<1> get_range() const { return {sycl::range<1>{std::size_t{tallies_.size()} * items_per_tally_}, sycl::range<1>{kWorkGroup}}; }
    [[nodiscard]] std::size_t tallies() const { return tallies_.size(); }
    [[nodiscard]] std::size_t items_per_tally() const { return items_per_tally_; }

    void operator()(const sycl::nd_item<1> &item) const {
        const auto group = static_cast<std::uint32_t>(item.get_group(0));
        const auto t = group / groups_per_tally_;
        const std::size_t first = std::size_t{group - t * groups_per_tally_} * kWorkGroup + item.get_local_id(0);
        const auto observed = tallies_.observed(t);
        std::uint32_t ones = 0;
        for (std::size_t w = first; w < observed.size(); w += items_per_tally_) {
            ones += static_cast<std::uint32_t>(sycl::popcount(observed.data()[w]));
        }
        const std::uint32_t sum = sycl::reduce_over_group(item.get_group(), ones, sycl::plus<std::uint32_t>());
        if (item.get_local_linear_id() == 0) {
            sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> count(
                tallies_[t].ones);
            count.fetch_add(std::uint64_t{sum});
        }
    }
};

} // namespace fuzzybits::mc::kernel
