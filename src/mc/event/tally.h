/**
 * @file tally.h
 * @brief Tallies: how many of an observed node's trials are true, counted on the device and estimated on the host
 */

#pragma once

#include "mc/event/node.h"

#include <sycl/sycl.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace fuzzybits::mc::event {

/// A tally as its kernel accumulates it: the observed node, its arena slot, and how many of the node's trials were
/// true over every pass so far. The number of trials is not stored: it is passes × bitpacks per node × bits per
/// bitpack, which the host knows.
template <node_types_like T> struct tally {
    /// The tally of the node `id` whose output is arena slot `observed`, counting from `count`. There is no
    /// aggregate form, so a tally cannot be made without its slot.
    constexpr tally(const typename T::index_type id, const typename T::extent_type observed, const std::uint64_t count = 0) noexcept
        : id_in_pdag(id), slot(observed), ones(count) {}

    typename T::index_type id_in_pdag;
    typename T::extent_type slot;
    std::uint64_t ones;
};

/// The tallies of one kernel, as the kernel holds them (by value): the records, whose counts it adds to, and the
/// arena it reads the observed nodes from.
template <node_types_like T> class tally_block_view {
  public:
    using bitpack_type = typename T::bitpack_type;
    using extent_type = typename T::extent_type;
    using observed_span = bitpack_span<const bitpack_type, extent_type>;

    constexpr tally_block_view() noexcept = default;
    constexpr tally_block_view(tally<T> *records, const arena_view<T> &arena, const extent_type size) noexcept
        : records_(records), arena_(arena), size_(size) {}

    [[nodiscard]] constexpr extent_type size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] constexpr extent_type bitpacks_per_node() const noexcept { return arena_.bitpacks_per_node(); }
    [[nodiscard]] constexpr tally<T> &operator[](const extent_type i) const noexcept { return records_[i]; }
    /// The bitpacks of the node tally i observes.
    [[nodiscard]] constexpr observed_span observed(const extent_type i) const noexcept { return arena_.output(records_[i].slot).as_const(); }

  private:
    tally<T> *records_ = nullptr;
    arena_view<T> arena_{};
    extent_type size_ = 0;
};

/// Owns the tallies of one kernel, in shared USM: the device adds to their counts, and the host reads them while no
/// tally kernel is in flight. The block reads the arena and holds no slot of it. Move-only; a moved-from block is
/// empty.
template <node_types_like T> class tally_block {
  public:
    using view_type = tally_block_view<T>;
    using extent_type = typename T::extent_type;

    tally_block() = default;
    tally_block(const tally_block &) = delete;
    tally_block &operator=(const tally_block &) = delete;
    tally_block(tally_block &&other) noexcept : records_(std::move(other.records_)), arena_(std::exchange(other.arena_, {})) {}
    tally_block &operator=(tally_block &&other) noexcept {
        records_ = std::move(other.records_);
        arena_ = std::exchange(other.arena_, {});
        return *this;
    }
    ~tally_block() = default;

    /// The tallies `records`, in order, counting from their `ones`. Refuses (std::out_of_range) a slot outside
    /// `arena`, and (std::length_error) more tallies than extent_type can count.
    template <std::ranges::sized_range R>
        requires std::constructible_from<tally<T>, std::ranges::range_reference_t<R>>
    [[nodiscard]] static tally_block create(const sycl::queue &queue, const arena_view<T> &arena, R &&records) {
        if (static_cast<std::size_t>(std::ranges::size(records)) > std::numeric_limits<extent_type>::max()) {
            throw std::length_error("tally_block: more tallies than the extent type counts");
        }
        tally_block block;
        block.records_ = usm_array<tally<T>>::copy_of(queue, std::forward<R>(records), sycl::usm::alloc::shared);
        for (const tally<T> &record : block.records_.host()) {
            if (record.slot >= arena.slots()) {
                throw std::out_of_range("tally_block: node " + std::to_string(record.id_in_pdag) + " has slot " + std::to_string(record.slot) +
                                        " of an arena of " + std::to_string(arena.slots()));
            }
        }
        block.arena_ = arena;
        return block;
    }

    [[nodiscard]] view_type view() const noexcept {
        static_assert(std::is_trivially_copyable_v<view_type>, "kernels hold views by value");
        return {records_.data(), arena_, size()};
    }
    [[nodiscard]] extent_type size() const noexcept { return static_cast<extent_type>(records_.size()); }
    [[nodiscard]] bool empty() const noexcept { return records_.empty(); }
    /// The records, on the host. Their counts are current once no tally kernel is in flight.
    [[nodiscard]] std::span<const tally<T>> records() const { return records_.host(); }

  private:
    usm_array<tally<T>> records_;
    arena_view<T> arena_{};
};

/// What the host makes of a tally: its counts, and their normal-approximation estimate, which
/// stats::populate_point_estimates (stats/ci_utils.h) fills from the counts.
struct tally_estimate {
    std::size_t num_one_bits = 0;
    std::size_t total_bits = 0;
    std::double_t mean = 0.;
    std::double_t std_err = 0.;
    /// [lower 95 %, upper 95 %, lower 99 %, upper 99 %]
    std::array<std::double_t, 4> ci = {0., 0., 0., 0.};
};

inline std::ostream &operator<<(std::ostream &os, const tally_estimate &t) {
    os << "p01= " << t.ci[2] << "  |  "
       << "p05= " << t.ci[0] << "  |  "
       << "mu = " << t.mean << "  |  "
       << "p95= " << t.ci[1] << "  |  "
       << "p99= " << t.ci[3] << "  |  ";
    return os;
}

} // namespace fuzzybits::mc::event
