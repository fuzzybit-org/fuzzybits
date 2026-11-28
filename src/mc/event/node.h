/**
 * @file node.h
 * @brief Nodes of the sampled graph: the engine's integer types, the output arena that holds every node's bitpacks,
 *        and the blocks of nodes that one kernel evaluates
 *
 * @details
 * - **Types.** The engine's integer choices travel as one `node_types` parameter. The concepts hold each to what
 *   the kernels can represent: bitpacks of at most 64 bits, and ids and extents of at most 32, the width of a
 *   Philox counter word (prng/stream.h) and of the kernels' indices.
 * - **Arena.** Every node's output for one pass lives in one device allocation, node-major: slot s holds bitpacks
 *   [s·B, (s + 1)·B), B the bitpacks per node per pass. A node's output is found from its slot, so no record or
 *   table stores a pointer to it: a gate's inputs and a tally's node are slots, and a kernel reaches every node it
 *   reads or writes through one pointer, the arena's base. The arena is device memory: the host passes its addresses
 *   to kernels and to queue.memcpy, and never dereferences them.
 * - **Blocks.** A `node_block` is the nodes one kernel evaluates: their records, each of which names its node
 *   (`id_in_pdag`) and holds what the kernel needs, and the run of arena slots their outputs occupy, in record
 *   order. The host owns the block; the kernel holds its `node_block_view` by value.
 */

#pragma once

#include "mc/event/span.h"
#include "mc/event/usm.h"

#include <sycl/sycl.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace fuzzybits::mc::event {

// ── vocabulary ──────────────────────────────────────────────────────────────────────────────────────────────────

/// An unsigned word of at most 64 bits; bit j is trial j. A wider word is refused here, at the type, rather than in
/// each kernel: the gate lanes, the cardinality counter and the tally's popcount all assume at most 64 bits.
template <class B>
concept bitpack_word = std::unsigned_integral<B> && !std::same_as<std::remove_cv_t<B>, bool> && sizeof(B) <= sizeof(std::uint64_t);

/// At most 32 bits: ids and extents become Philox counter words (prng/stream.h) and 32-bit kernel indices, so a wider
/// type would be truncated there without a diagnostic.
template <class I>
concept counter_word = std::integral<I> && !std::same_as<std::remove_cv_t<I>, bool> && sizeof(I) <= sizeof(std::uint32_t);

/// A count or a position: of nodes, arena slots, inputs or bitpacks.
template <class E>
concept node_extent = counter_word<E> && std::unsigned_integral<E>;

/// The engine's integer choices, made once: the bitpack word, the PDAG index of a node, and the extent of every
/// count and position.
template <bitpack_word Bitpack, counter_word Index, node_extent Extent> struct node_types {
    using bitpack_type = Bitpack;
    using index_type = Index;
    using extent_type = Extent;
};

template <class T>
concept node_types_like = requires {
    typename T::bitpack_type;
    typename T::index_type;
    typename T::extent_type;
} && bitpack_word<typename T::bitpack_type> && counter_word<typename T::index_type> && node_extent<typename T::extent_type>;

/// One node's bitpacks for one pass.
template <bitpack_word B, node_extent E> using bitpack_span = device_span<B, E>;

/// A record that names its node, as every block holds one per node: the node's PDAG index as `id_in_pdag`, and
/// trivially copyable, since it lives in USM.
template <class R, class T>
concept node_record = node_types_like<T> && std::is_trivially_copyable_v<R> &&
                      requires(const R &record) { requires std::same_as<std::remove_cvref_t<decltype(record.id_in_pdag)>, typename T::index_type>; };

// ── the output arena ────────────────────────────────────────────────────────────────────────────────────────────

/// Arena slots [first, first + count).
template <node_extent E> struct slot_range {
    E first = 0;
    E count = 0;

    [[nodiscard]] constexpr E end() const noexcept { return static_cast<E>(first + count); }
    [[nodiscard]] constexpr bool contains(const E slot) const noexcept { return slot >= first && slot - first < count; }
};

/// Every node's output for one pass, node-major: slot s holds bitpacks [s·B, (s + 1)·B), B = bitpacks_per_node().
/// What a kernel holds, by value (a pointer and two counts), to reach any node's output from its slot.
template <node_types_like T> class arena_view {
  public:
    using bitpack_type = typename T::bitpack_type;
    using extent_type = typename T::extent_type;
    using output_span = bitpack_span<bitpack_type, extent_type>;

    constexpr arena_view() noexcept = default;
    constexpr arena_view(bitpack_type *base, const extent_type slots, const extent_type bitpacks_per_node) noexcept
        : base_(base), slots_(slots), bitpacks_per_node_(bitpacks_per_node) {}

    [[nodiscard]] constexpr extent_type slots() const noexcept { return slots_; }
    [[nodiscard]] constexpr extent_type bitpacks_per_node() const noexcept { return bitpacks_per_node_; }
    /// Bitpacks in the arena, in 64 bits: slots() × bitpacks_per_node() can exceed 2^32.
    [[nodiscard]] constexpr std::size_t size() const noexcept { return std::size_t{slots_} * bitpacks_per_node_; }

    /// The first bitpack of `slot`. The offset is a 64-bit product: one wide multiply-add on GPUs.
    [[nodiscard]] constexpr bitpack_type *at(const extent_type slot) const noexcept { return base_ + std::size_t{slot} * bitpacks_per_node_; }
    /// The bitpacks of `slot`: device memory, for kernels.
    [[nodiscard]] constexpr output_span output(const extent_type slot) const noexcept { return {at(slot), bitpacks_per_node_}; }
    /// The whole arena, slot after slot.
    [[nodiscard]] constexpr bitpack_type *data() const noexcept { return base_; }

  private:
    bitpack_type *base_ = nullptr;
    extent_type slots_ = 0;
    extent_type bitpacks_per_node_ = 0;
};

/// Owns the arena: one device allocation of slots() × bitpacks_per_node() bitpacks, whose slots it hands to blocks in
/// order. Move-only; a moved-from arena is empty.
template <node_types_like T> class output_arena {
  public:
    using view_type = arena_view<T>;
    using bitpack_type = typename T::bitpack_type;
    using extent_type = typename T::extent_type;

    output_arena() = default;
    output_arena(const output_arena &) = delete;
    output_arena &operator=(const output_arena &) = delete;
    output_arena(output_arena &&other) noexcept
        : storage_(std::move(other.storage_)), slots_(std::exchange(other.slots_, 0)), bitpacks_per_node_(std::exchange(other.bitpacks_per_node_, 0)),
          reserved_(std::exchange(other.reserved_, 0)) {}
    output_arena &operator=(output_arena &&other) noexcept {
        storage_ = std::move(other.storage_);
        slots_ = std::exchange(other.slots_, 0);
        bitpacks_per_node_ = std::exchange(other.bitpacks_per_node_, 0);
        reserved_ = std::exchange(other.reserved_, 0);
        return *this;
    }
    ~output_arena() = default;

    /// `slots` slots of `bitpacks_per_node` bitpacks, in device memory. Refuses zero bitpacks per node
    /// (std::invalid_argument) and a count that extent_type cannot hold (std::length_error).
    [[nodiscard]] static output_arena create(const sycl::queue &queue, const std::size_t slots, const std::size_t bitpacks_per_node) {
        constexpr auto max = std::size_t{std::numeric_limits<extent_type>::max()};
        if (bitpacks_per_node == 0) {
            throw std::invalid_argument("output_arena: zero bitpacks per node");
        }
        if (slots > max || bitpacks_per_node > max) {
            throw std::length_error("output_arena: " + std::to_string(slots) + " slots of " + std::to_string(bitpacks_per_node) +
                                    " bitpacks exceed the extent type");
        }
        output_arena arena;
        arena.storage_ = usm_array<bitpack_type>::allocate(queue, slots * bitpacks_per_node, sycl::usm::alloc::device);
        arena.slots_ = static_cast<extent_type>(slots);
        arena.bitpacks_per_node_ = static_cast<extent_type>(bitpacks_per_node);
        return arena;
    }

    /// The next `count` slots, for one block's nodes. Throws std::length_error when fewer are left: every node needs a
    /// slot of its own.
    [[nodiscard]] slot_range<extent_type> reserve(const std::size_t count) {
        const std::size_t left = std::size_t{slots_} - reserved_;
        if (count > left) {
            throw std::length_error("output_arena: " + std::to_string(count) + " slots requested, " + std::to_string(left) + " left");
        }
        const slot_range<extent_type> range{reserved_, static_cast<extent_type>(count)};
        reserved_ = static_cast<extent_type>(reserved_ + count);
        return range;
    }

    [[nodiscard]] view_type view() const noexcept {
        static_assert(std::is_trivially_copyable_v<view_type>, "kernels hold views by value");
        return {storage_.data(), slots_, bitpacks_per_node_};
    }
    [[nodiscard]] extent_type slots() const noexcept { return slots_; }
    [[nodiscard]] extent_type bitpacks_per_node() const noexcept { return bitpacks_per_node_; }
    /// Slots handed out so far: [0, reserved()) belong to blocks.
    [[nodiscard]] extent_type reserved() const noexcept { return reserved_; }

  private:
    usm_array<bitpack_type> storage_;
    extent_type slots_ = 0;
    extent_type bitpacks_per_node_ = 0;
    extent_type reserved_ = 0;
};

// ── blocks ──────────────────────────────────────────────────────────────────────────────────────────────────────

/// The nodes of one kernel, as the kernel holds them (by value): their records, read-only, and the arena slots
/// [slots().first, slots().end()) their outputs occupy, in record order.
template <class Record, node_types_like T> class node_block_view {
  public:
    using record_type = Record;
    using bitpack_type = typename T::bitpack_type;
    using extent_type = typename T::extent_type;
    using output_span = bitpack_span<bitpack_type, extent_type>;

    constexpr node_block_view() noexcept = default;
    constexpr node_block_view(const Record *records, const arena_view<T> &arena, const slot_range<extent_type> slots) noexcept
        : records_(records), arena_(arena), slots_(slots) {}

    [[nodiscard]] constexpr extent_type size() const noexcept { return slots_.count; }
    [[nodiscard]] constexpr bool empty() const noexcept { return slots_.count == 0; }
    [[nodiscard]] constexpr extent_type bitpacks_per_node() const noexcept { return arena_.bitpacks_per_node(); }
    [[nodiscard]] constexpr slot_range<extent_type> slots() const noexcept { return slots_; }
    [[nodiscard]] constexpr const arena_view<T> &arena() const noexcept { return arena_; }

    [[nodiscard]] constexpr const Record &operator[](const extent_type i) const noexcept { return records_[i]; }
    /// The arena slot of node i.
    [[nodiscard]] constexpr extent_type slot(const extent_type i) const noexcept { return static_cast<extent_type>(slots_.first + i); }
    /// The bitpacks node i writes.
    [[nodiscard]] constexpr output_span output(const extent_type i) const noexcept { return arena_.output(slot(i)); }

  private:
    const Record *records_ = nullptr;
    arena_view<T> arena_{};
    slot_range<extent_type> slots_{};
};

/// A node of a block, on the host: its position in the block, its arena slot and its record. Its output is device
/// memory, reached from the slot (arena_view::at, for kernels and queue.memcpy).
template <class Record, node_types_like T> struct node_ref {
    typename T::extent_type position;
    typename T::extent_type slot;
    const Record &record;
};

/// Owns one block: its records, in shared USM (written by the host, read by the device), and its run of arena slots.
/// The arena belongs to whoever created it and must outlive the block's kernels. Move-only; a moved-from block is
/// empty.
template <class Record, node_types_like T>
    requires node_record<Record, T>
class node_block {
  public:
    using view_type = node_block_view<Record, T>;
    using record_type = Record;
    using bitpack_type = typename T::bitpack_type;
    using extent_type = typename T::extent_type;

    node_block() = default;
    node_block(const node_block &) = delete;
    node_block &operator=(const node_block &) = delete;
    node_block(node_block &&other) noexcept
        : records_(std::move(other.records_)), arena_(std::exchange(other.arena_, {})), slots_(std::exchange(other.slots_, {})) {}
    node_block &operator=(node_block &&other) noexcept {
        records_ = std::move(other.records_);
        arena_ = std::exchange(other.arena_, {});
        slots_ = std::exchange(other.slots_, {});
        return *this;
    }
    ~node_block() = default;

    /// `records`, in order, each given the next free slot of `arena`.
    template <std::ranges::sized_range R>
        requires std::constructible_from<Record, std::ranges::range_reference_t<R>>
    [[nodiscard]] static node_block create(const sycl::queue &queue, output_arena<T> &arena, R &&records) {
        const auto count = static_cast<std::size_t>(std::ranges::size(records));
        node_block block;
        block.records_ = usm_array<Record>::copy_of(queue, std::forward<R>(records), sycl::usm::alloc::shared);
        block.slots_ = arena.reserve(count);
        block.arena_ = arena.view();
        return block;
    }

    [[nodiscard]] view_type view() const noexcept {
        static_assert(std::is_trivially_copyable_v<view_type>, "kernels hold views by value");
        return {records_.data(), arena_, slots_};
    }
    [[nodiscard]] extent_type size() const noexcept { return slots_.count; }
    [[nodiscard]] bool empty() const noexcept { return slots_.count == 0; }
    [[nodiscard]] slot_range<extent_type> slots() const noexcept { return slots_; }
    /// The arena slot of node i.
    [[nodiscard]] extent_type slot(const extent_type i) const noexcept { return static_cast<extent_type>(slots_.first + i); }
    /// The records, on the host.
    [[nodiscard]] std::span<const Record> records() const { return records_.host(); }

    /// The nodes, on the host, as node_refs in record order: for example to map PDAG indices to slots.
    [[nodiscard]] auto nodes() const {
        return std::views::iota(extent_type{0}, size()) |
               std::views::transform([v = view()](const extent_type i) { return node_ref<Record, T>{i, v.slot(i), v[i]}; });
    }

  private:
    usm_array<Record> records_;
    arena_view<T> arena_{};
    slot_range<extent_type> slots_{};
};

} // namespace fuzzybits::mc::event
