/**
 * @file gate.h
 * @brief SYCL kernels for the Boolean gates (AND, OR, XOR, NOT, NAND, NOR, XNOR, NULL) on bit-packed data
 * @author Arjun Earthperson
 * @date 2025
 *
 * @details A gate folds its inputs with its op code's operation (op_code.h: AND, OR or XOR, complemented after for
 * NAND, NOR and XNOR; NOT complements its one input and NULL passes it through); inputs at or after the gate's
 * negated_from are complemented first. Counting gates (at least, at most, exactly k of n) are in cardinality.h. A gate reads its inputs and writes
 * its output in the output arena (event/node.h), through its block's input table of arena slots (event/gate.h).
 *
 * A work-item evaluates one bitpack of one gate on GPUs, and kGateCpuLanes consecutive bitpacks as one vector on
 * CPUs (gate_lanes.h). Two forms (`gate_slots_of` gives the kernel builder's choice):
 *
 * - counted: the input count read at run time. The kernel builder uses it for every AND, OR, XOR, NAND, NOR and
 *   XNOR gate on every device, so that a layer has one kernel per op code. GPUs read it in blocks of 8 so that 8
 *   loads are in flight; CPUs read the positive inputs, then the negated ones, without blocks (blocks of 8 were
 *   6-10% slower from 32 inputs on an i7-10700, and tied below). Its gates differ in input count, so on CPUs it
 *   runs in block-major order (`gate_order_for`), which balances them across threads.
 * - fixed: a compile-time number of slots (kGateSlots); slot j holds an input iff j < num_inputs, and a slot without
 *   one is the fold's identity. The kernel builder uses one slot for NOT and NULL. More slots were 5-21% faster than a runtime
 *   count in the V100S microbenchmark (16,384 bitpacks per gate) but tied in end-to-end runs, and a kernel per slot
 *   count multiplies kernel launches: on a model of about 88,000 gates in 147 layers, with passes of
 *   1,024 bitpacks per node, it took 1,001 gate kernels per pass instead of 316, and 2.3-2.8x as long on NVIDIA
 *   GPUs, whose host spends about 100 us per launch. On CPUs every kernel is also a barrier that waits for the
 *   slowest thread.
 *
 * A gate of no inputs folds none: AND, NOR and XNOR are true, OR, NAND and XOR false.
 */

#pragma once

#include "mc/event/gate.h"
#include "mc/kernel/gate_lanes.h"
#include "mc/kernel/op_code.h"

#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace fuzzybits::mc::kernel {

/// Slot counts of the fixed form; a gate takes the smallest that holds its inputs (at most a quarter padding).
inline constexpr std::array<std::uint32_t, 8> kGateSlots{1, 2, 3, 4, 6, 8, 12, 16};

/// Slots of the fixed form for a gate with num_inputs inputs; 0 for the counted form.
[[nodiscard]] constexpr std::uint32_t gate_slots_for(const std::size_t num_inputs) {
    for (const auto slots : kGateSlots) {
        if (num_inputs <= slots) {
            return slots;
        }
    }
    return 0;
}

/// Slots the kernel builder gives a Boolean gate of op code `op`: one for NOT and NULL, the counted form (0) for the
/// others.
[[nodiscard]] constexpr std::uint32_t gate_slots_of(const op_code op) noexcept { return is_unary(op) ? 1 : 0; }

/**
 * @brief Evaluates a block of gates of one op code and, for the fixed form, one slot count.
 *
 * @tparam op_ a Boolean gate (is_boolean).
 * @tparam T the engine's node_types.
 * @tparam slots_ the fixed form's slot count (one of kGateSlots), 0 for the counted form.
 * @tparam lanes_form_ true on CPUs: a work-item evaluates kLanes consecutive bitpacks as one vector.
 */
template <op_code op_, event::node_types_like T, std::uint32_t slots_, bool lanes_form_> class op {
    static_assert(is_boolean(op_), "counting gates are cardinality gates (cardinality.h)");
    static_assert(!is_unary(op_) || slots_ == 1, "NOT and NULL have one input");

  public:
    /// Bitpacks per work-item.
    static constexpr std::uint32_t kLanes = lanes_form_ ? kGateCpuLanes : 1;
    static constexpr std::size_t kWorkGroup = lanes_form_ ? 64 : 256;

  private:
    using bitpack_type = typename T::bitpack_type;
    using block_t = event::gate_block_view<T>;
    using gate_t = event::gate<T>;
    using lanes_t = typename gate_detail::lanes_of<bitpack_type, static_cast<int>(kLanes)>::type;

    static constexpr bool kAndLike = op_ == op_code::and_gate || op_ == op_code::nand_gate;
    static constexpr bool kOrLike = op_ == op_code::or_gate || op_ == op_code::nor_gate;
    static constexpr bool kXorLike = op_ == op_code::xor_gate || op_ == op_code::xnor_gate;
    /// NAND, NOR, XNOR and NOT complement the fold of their inputs (NOT's: its one input; NULL passes it through).
    static constexpr bool kComplemented = op_ == op_code::nand_gate || op_ == op_code::nor_gate || op_ == op_code::xnor_gate || op_ == op_code::not_gate;

    block_t gates_;
    gate_geometry<kLanes, kWorkGroup> geometry_;

    template <typename V> [[gnu::always_inline]] static void combine(V &acc, const V &x) {
        if constexpr (kAndLike) {
            acc &= x;
        } else if constexpr (kOrLike) {
            acc |= x;
        } else if constexpr (kXorLike) {
            acc ^= x;
        } else {
            acc = x;
        }
    }

    /// Input i of gate g at bitpack w, complemented from `negated` on.
    template <typename V>
    [[gnu::always_inline]] void input(V &x, const gate_t &g, const std::size_t w, const std::uint32_t i, const std::uint32_t negated) const {
        gate_detail::load(x, gates_.input(g, i) + w);
        if (i >= negated) {
            x = ~x;
        }
    }

    template <typename V> [[gnu::always_inline]] void evaluate(V &out, const gate_t &g, const std::size_t w) const {
        V acc = V{};
        if constexpr (kAndLike) {
            acc = ~V{};
        }
        const auto negated = static_cast<std::uint32_t>(g.negated_from);
        const auto n = static_cast<std::uint32_t>(g.num_inputs);
        if constexpr (slots_ > 0) {
#pragma unroll
            for (std::uint32_t j = 0; j < slots_; ++j) {
                if (j < n) {
                    V x;
                    input(x, g, w, j, negated);
                    combine(acc, x);
                }
            }
        } else if constexpr (lanes_form_) {
            const auto positive = negated < n ? negated : n;
            std::uint32_t i = 0;
#pragma unroll 4
            for (; i < positive; ++i) {
                V x;
                gate_detail::load(x, gates_.input(g, i) + w);
                combine(acc, x);
            }
#pragma unroll 4
            for (; i < n; ++i) {
                V x;
                gate_detail::load(x, gates_.input(g, i) + w);
                x = ~x;
                combine(acc, x);
            }
        } else {
            std::uint32_t i = 0;
            for (; i + 8 <= n; i += 8) {
                V x[8];
#pragma unroll
                for (std::uint32_t j = 0; j < 8; ++j) {
                    input(x[j], g, w, i + j, negated);
                }
#pragma unroll
                for (std::uint32_t j = 0; j < 8; ++j) {
                    combine(acc, x[j]);
                }
            }
            for (; i < n; ++i) {
                V x;
                input(x, g, w, i, negated);
                combine(acc, x);
            }
        }
        if constexpr (kComplemented) {
            acc = ~acc;
        }
        out = acc;
    }

  public:
    explicit op(const block_t &gates, const gate_order order = gate_order::gate_major)
        : gates_(gates), geometry_(gates.size(), gates.bitpacks_per_node(), order) {}

    [[nodiscard]] sycl::nd_range<1> get_range() const { return geometry_.range(); }
    [[nodiscard]] std::size_t gates() const { return geometry_.gates(); }
    [[nodiscard]] std::size_t items_per_gate() const { return geometry_.items_per_gate(); }

    void operator()(const sycl::nd_item<1> &item) const {
        typename gate_geometry<kLanes, kWorkGroup>::position at;
        if (!geometry_.locate(item, at)) {
            return;
        }
        const auto &g = gates_[at.gate];
        bitpack_type *out = gates_.output(at.gate).data();
        const std::size_t bitpacks = geometry_.bitpacks();
        if (at.first + kLanes <= bitpacks) {
            lanes_t r;
            evaluate(r, g, at.first);
            gate_detail::store(out + at.first, r);
        } else {
            for (std::size_t w = at.first; w < bitpacks; ++w) {
                bitpack_type r;
                evaluate(r, g, w);
                out[w] = r;
            }
        }
    }
};

} // namespace fuzzybits::mc::kernel
