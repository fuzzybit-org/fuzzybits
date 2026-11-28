/**
 * @file layer.h
 * @brief A layer of nodes as the builder takes them, by op code (op.h): its basic events with their laws, its constants,
 *        and its gates with their inputs
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details The builder does not read the PDAG: the layer manager reads each layer into this form (executor/compat.h),
 * which holds what a kernel needs and nothing else. A layer's nodes do not read each other, so every kernel of a layer
 * reads only the outputs of earlier layers.
 */

#pragma once

#include "mc/event/basic_event.h"
#include "mc/executor/builder/op.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace fuzzybits::mc::executor::builder {

/// An input of a gate: the PDAG index of the node it reads, and whether the gate reads its complement.
struct input {
    std::int32_t node = 0;
    bool negated = false;
};

/// A counting gate is true where the number of its true inputs lies in [low, high]: at least k of n is [k, n], at most
/// k is [0, k] and exactly k is [k, k].
struct bounds {
    std::uint32_t low = 0;
    std::uint32_t high = 0;
};

/// A gate: its PDAG index, its inputs (layer::inputs_of) and, for a counting gate, its bounds.
struct gate {
    std::int32_t index = 0;
    bounds range;
    std::uint32_t first_input = 0;
    std::uint32_t inputs = 0;
};

/// A layer: its basic events and constants, each with its exact law, and its gates by op code, every gate's inputs in
/// one array.
template <event::node_types_like T> class layer {
  public:
    /// The nodes of a law: those it samples (op_code::sample), and the constants, whose law is never or always true
    /// (op_code::zeros, op_code::ones) and whose bitpacks are written once, by no kernel (builder/basic_event.h).
    std::vector<event::basic_event<T>> basic_events;

    /// Adds node `index`, true in every trial (op_code::ones) if `value` and in none (op_code::zeros) otherwise.
    void add_constant(const std::int32_t index, const bool value) {
        basic_events.push_back(
            {.id_in_pdag = static_cast<typename T::index_type>(index), .law = value ? event::bernoulli_law::always() : event::bernoulli_law::never()});
    }

    /// Adds Boolean gate `index` of op code `op` (is_boolean), with `inputs`. A gate of no inputs folds none: AND, NOR
    /// and XNOR are true, OR, NAND and XOR false.
    ///
    /// @throws std::invalid_argument if `op` is not a Boolean gate, or a NOT or NULL gate has other than one input
    void add_gate(const op_code op, const std::int32_t index, const std::span<const input> inputs) {
        if (!is_boolean(op)) {
            refuse(op, index, "is not a Boolean gate; a counting gate takes its bounds");
        }
        if (is_unary(op) && inputs.size() != 1) {
            refuse(op, index, "has " + std::to_string(inputs.size()) + " inputs, not one");
        }
        add(op, index, inputs, {});
    }

    /// Adds counting gate `index` of op code `op` (is_counting), with `inputs`: true where the number of its true
    /// inputs lies in `range`.
    ///
    /// @throws std::invalid_argument if `op` is not a counting gate, `range` is not low <= high <= the number of inputs
    ///         (as ir::pc requires of its cardinality gates), or a majority or minority gate's range is not its own
    void add_gate(const op_code op, const std::int32_t index, const std::span<const input> inputs, const bounds range) {
        if (!is_counting(op)) {
            refuse(op, index, "is not a counting gate");
        }
        if (range.low > range.high || range.high > inputs.size()) {
            refuse(op, index,
                   "has bounds [" + std::to_string(range.low) + ", " + std::to_string(range.high) + "] outside its " + std::to_string(inputs.size()) +
                       " inputs");
        }
        // A majority or minority gate computes one range of its own: more than half its inputs, or not more than half.
        if (op == op_code::majority || op == op_code::minority) {
            const std::size_t half = inputs.size() / 2;
            const bool majority = op == op_code::majority;
            if (range.low != (majority ? half + 1 : 0) || range.high != (majority ? inputs.size() : half)) {
                refuse(op, index,
                       "has bounds [" + std::to_string(range.low) + ", " + std::to_string(range.high) + "], not its own over " + std::to_string(inputs.size()) +
                           " inputs: a majority is [n / 2 + 1, n], a minority [0, n / 2]");
            }
        }
        add(op, index, inputs, range);
    }

    /// The gates of op code `op`, in the order they were added. Throws std::out_of_range if `op` is not a gate.
    [[nodiscard]] std::span<const gate> gates(const op_code op) const { return gates_.at(gate_position(op)); }

    [[nodiscard]] std::span<const input> inputs_of(const gate &g) const { return std::span<const input>(inputs_).subspan(g.first_input, g.inputs); }

  private:
    void add(const op_code op, const std::int32_t index, const std::span<const input> inputs, const bounds range) {
        gates_.at(gate_position(op))
            .push_back({.index = index,
                        .range = range,
                        .first_input = static_cast<std::uint32_t>(inputs_.size()),
                        .inputs = static_cast<std::uint32_t>(inputs.size())});
        inputs_.insert(inputs_.end(), inputs.begin(), inputs.end());
    }

    [[noreturn]] static void refuse(const op_code op, const std::int32_t index, const std::string &why) {
        throw std::invalid_argument("kernel builder: gate G" + std::to_string(index) + " (" + std::string(name_of(op)) + ") " + why);
    }

    std::array<std::vector<gate>, kGateOps.size()> gates_;
    std::vector<input> inputs_;
};

} // namespace fuzzybits::mc::executor::builder
