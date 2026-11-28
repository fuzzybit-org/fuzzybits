/**
 * @file gate.h
 * @brief Gates as their kernels read them: records that find their inputs through a table of arena slots
 *
 * @details A gate reads other nodes' outputs. Its record holds no pointer to them: it holds the run
 * [first_input, first_input + num_inputs) of its block's input table, whose entries are arena slots
 * (event/node.h). Inputs [0, negated_from) count as they are, inputs [negated_from, num_inputs) complemented.
 *
 * Kernels with a fixed number of input slots (kernel::kGateSlots, kernel::kCardinalitySlots) read slot j only when
 * j < num_inputs, so the table holds exactly the inputs, without padding.
 */

#pragma once

#include "mc/event/node.h"

#include <sycl/sycl.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fuzzybits::mc::event {

/// An AND, OR, XOR, NAND, NOR, NOT or NULL gate; the connective is its kernel's.
template <node_types_like T> struct gate {
    typename T::index_type id_in_pdag;
    /// The position of input 0 in the block's input table.
    typename T::extent_type first_input;
    typename T::extent_type num_inputs;
    /// Inputs from here on count complemented.
    typename T::extent_type negated_from;
};

/// A gate that is true where the number of its true inputs lies in [low, high]. high >= num_inputs is no upper
/// bound, so at least k is [k, n], at most k is [0, k] and exactly k is [k, k]; low > high is never true.
template <node_types_like T> struct cardinality_gate {
    typename T::index_type id_in_pdag;
    typename T::extent_type first_input;
    typename T::extent_type num_inputs;
    typename T::extent_type negated_from;
    typename T::extent_type low;
    typename T::extent_type high;
};

/// The bounds of a cardinality gate: [low, high].
template <node_extent E> struct cardinality_bounds {
    E low = 0;
    E high = 0;
};

/// A record of a node that reads other nodes: a node record with its run of the block's input table.
template <class R, class T>
concept gate_record = node_record<R, T> && requires(const R &g) {
    requires std::same_as<std::remove_cvref_t<decltype(g.first_input)>, typename T::extent_type>;
    requires std::same_as<std::remove_cvref_t<decltype(g.num_inputs)>, typename T::extent_type>;
    requires std::same_as<std::remove_cvref_t<decltype(g.negated_from)>, typename T::extent_type>;
};

/// A block of gates as its kernel holds it (by value): a node block and the input table, whose entries are arena
/// slots.
template <class Record, node_types_like T> class input_block_view : public node_block_view<Record, T> {
  public:
    using nodes_view = node_block_view<Record, T>;
    using typename nodes_view::bitpack_type;
    using typename nodes_view::extent_type;

    constexpr input_block_view() noexcept = default;
    constexpr input_block_view(const nodes_view &nodes, const extent_type *inputs) noexcept : nodes_view(nodes), inputs_(inputs) {}

    /// The arena slot of input i of gate g.
    [[nodiscard]] constexpr extent_type input_slot(const Record &g, const extent_type i) const noexcept { return inputs_[g.first_input + i]; }
    /// The first bitpack of input i of gate g, which the gate reads and never writes.
    [[nodiscard]] constexpr const bitpack_type *input(const Record &g, const extent_type i) const noexcept { return this->arena().at(input_slot(g, i)); }

  private:
    const extent_type *inputs_ = nullptr;
};

template <class Record, node_types_like T>
    requires gate_record<Record, T>
class input_block_builder;

/// Owns a block of gates: the node block and its input table, both in shared USM. input_block_builder builds it.
/// Move-only; a moved-from block is empty.
template <class Record, node_types_like T>
    requires gate_record<Record, T>
class input_block : public node_block<Record, T> {
  public:
    using nodes_block = node_block<Record, T>;
    using view_type = input_block_view<Record, T>;
    using typename nodes_block::extent_type;

    input_block() = default;

    [[nodiscard]] view_type view() const noexcept {
        static_assert(std::is_trivially_copyable_v<view_type>, "kernels hold views by value");
        return {nodes_block::view(), inputs_.data()};
    }
    /// The input table, on the host: arena slots.
    [[nodiscard]] std::span<const extent_type> inputs() const { return inputs_.host(); }
    /// The inputs of gate i, as arena slots: positive, then negated.
    [[nodiscard]] std::span<const extent_type> inputs_of(const extent_type i) const {
        const Record &g = this->records()[i];
        return inputs().subspan(g.first_input, g.num_inputs);
    }

  private:
    friend class input_block_builder<Record, T>;

    input_block(nodes_block nodes, usm_array<extent_type> inputs) noexcept : nodes_block(std::move(nodes)), inputs_(std::move(inputs)) {}

    usm_array<extent_type> inputs_;
};

/// Collects one block's gates on the host, as records and one flat input table, then copies both to USM and
/// reserves the block's arena slots in one step.
template <class Record, node_types_like T>
    requires gate_record<Record, T>
class input_block_builder {
  public:
    using index_type = typename T::index_type;
    using extent_type = typename T::extent_type;

    /// Room for `gates` gates with `inputs` inputs in all.
    void reserve(const std::size_t gates, const std::size_t inputs) {
        records_.reserve(gates);
        inputs_.reserve(inputs);
    }

    /// A gate whose inputs are the arena slots `positive`, then `negated`, which count complemented. Not for cardinality
    /// gates, whose bounds have no default: they take the overload below.
    void add(const index_type id_in_pdag, const std::span<const extent_type> positive, const std::span<const extent_type> negated)
        requires(!std::same_as<Record, cardinality_gate<T>>)
    {
        (void)append(id_in_pdag, positive, negated);
    }

    /// A cardinality gate, with its bounds.
    void add(const index_type id_in_pdag, const std::span<const extent_type> positive, const std::span<const extent_type> negated,
             const cardinality_bounds<extent_type> bounds)
        requires std::same_as<Record, cardinality_gate<T>>
    {
        Record &record = append(id_in_pdag, positive, negated);
        record.low = bounds.low;
        record.high = bounds.high;
    }

    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
    [[nodiscard]] bool empty() const noexcept { return records_.empty(); }
    [[nodiscard]] std::size_t input_count() const noexcept { return inputs_.size(); }

    /// The block: records and input table in shared USM, and the next size() slots of `arena`. Refuses
    /// (std::out_of_range) an input slot that no earlier block holds: a gate reads only outputs that a kernel before
    /// it writes.
    [[nodiscard]] input_block<Record, T> build(const sycl::queue &queue, output_arena<T> &arena) const {
        for (const extent_type slot : inputs_) {
            if (slot >= arena.reserved()) {
                throw std::out_of_range("input_block_builder: input slot " + std::to_string(slot) + " belongs to no earlier block (" +
                                        std::to_string(arena.reserved()) + " slots reserved)");
            }
        }
        auto inputs = usm_array<extent_type>::copy_of(queue, inputs_, sycl::usm::alloc::shared);
        auto nodes = node_block<Record, T>::create(queue, arena, records_);
        return input_block<Record, T>(std::move(nodes), std::move(inputs));
    }

  private:
    /// The record of a new gate, with its inputs appended to the table; valid until the next append.
    Record &append(const index_type id_in_pdag, const std::span<const extent_type> positive, const std::span<const extent_type> negated) {
        constexpr auto max = std::size_t{std::numeric_limits<extent_type>::max()};
        const std::size_t n = positive.size() + negated.size();
        if (n > max - inputs_.size() || records_.size() >= max) {
            throw std::length_error("input_block_builder: " + std::to_string(inputs_.size() + n) + " inputs of " + std::to_string(records_.size() + 1) +
                                    " gates exceed the extent type");
        }
        Record record{};
        record.id_in_pdag = id_in_pdag;
        record.first_input = static_cast<extent_type>(inputs_.size());
        record.num_inputs = static_cast<extent_type>(n);
        record.negated_from = static_cast<extent_type>(positive.size());
        inputs_.insert(inputs_.end(), positive.begin(), positive.end());
        inputs_.insert(inputs_.end(), negated.begin(), negated.end());
        return records_.emplace_back(record);
    }

    std::vector<Record> records_;
    std::vector<extent_type> inputs_;
};

template <node_types_like T> using gate_block = input_block<gate<T>, T>;
template <node_types_like T> using gate_block_view = input_block_view<gate<T>, T>;
template <node_types_like T> using gate_block_builder = input_block_builder<gate<T>, T>;

template <node_types_like T> using cardinality_gate_block = input_block<cardinality_gate<T>, T>;
template <node_types_like T> using cardinality_gate_block_view = input_block_view<cardinality_gate<T>, T>;
template <node_types_like T> using cardinality_gate_block_builder = input_block_builder<cardinality_gate<T>, T>;

} // namespace fuzzybits::mc::event
