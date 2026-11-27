/**
 * @file gate.h
 * @brief SYCL kernels for AND, OR, XOR, NAND, NOR, NOT and NULL gates on bit-packed data
 * @author Arjun Earthperson
 * @date 2025
 *
 * @details A gate reduces its inputs with its connective; inputs at or after negated_inputs_offset are
 * complemented first. Cardinality gates (at least, at most, exactly k of n) are in cardinality.h.
 *
 * A work-item evaluates one bitpack of one gate on GPUs, and kGateCpuLanes consecutive bitpacks as one vector on
 * CPUs (gate_lanes.h). Two forms (`gate_slots_of` gives the kernel builder's choice):
 *
 * - counted: the input count read at run time. The kernel builder uses it for every AND, OR, XOR, NAND and NOR
 *   gate on every device, so that a layer has one kernel per connective. GPUs read it in blocks of 8 so that 8
 *   loads are in flight; CPUs read the positive inputs, then the negated ones, without blocks (blocks of 8 were
 *   6-10% slower from 32 inputs on an i7-10700, and tied below). Its gates differ in input count, so on CPUs it
 *   runs in block-major order (`gate_order_for`), which balances them across threads.
 * - fixed: a compile-time number of slots (kGateSlots) into which a gate is padded; the gate's input_mask says
 *   which slots hold an input, and a slot without one is the connective's identity. The kernel builder uses one
 *   slot for NOT and NULL. More slots were 5-21% faster than a runtime count in the V100S microbenchmark (16,384
 *   bitpacks per gate) but tied in end-to-end runs, and a kernel per slot count multiplies kernel launches: on a
 *   model of about 88,000 gates in 147 layers, with passes of 1,024 bitpacks per node, it took 1,001 gate
 *   kernels per pass instead of 316, and 2.3-2.8x as long on NVIDIA GPUs, whose host spends about 100 us per
 *   launch. On CPUs every kernel is also a barrier that waits for the slowest thread.
 *
 * Constant gates are AND (TRUE) or OR (FALSE) over no inputs.
 */

#pragma once

#include "mc/event/node.tpp"
#include "mc/event/sample_shape.h"
#include "mc/kernel/gate_lanes.h"
#include "pdag.h"

#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace scram::mc::kernel {

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

/// Slots the kernel builder gives a gate of connective op: one for NOT and NULL, the counted form (0) for the
/// others.
[[nodiscard]] constexpr std::uint32_t gate_slots_of(const core::Connective op) {
    return op == core::kNot || op == core::kNull ? 1 : 0;
}

/**
 * @brief Evaluates a block of gates of one connective and, for the fixed form, one slot count.
 *
 * @tparam slots_ the fixed form's slot count (one of kGateSlots), 0 for the counted form.
 * @tparam lanes_form_ true on CPUs: a work-item evaluates kLanes consecutive bitpacks as one vector.
 */
template <core::Connective OpType, typename bitpack_t_, typename size_t_, std::uint32_t slots_, bool lanes_form_>
class op {
    static_assert(OpType == core::kAnd || OpType == core::kOr || OpType == core::kXor || OpType == core::kNand ||
                      OpType == core::kNor || OpType == core::kNot || OpType == core::kNull,
                  "at-least gates are cardinality gates (cardinality.h)");
    static_assert((OpType != core::kNot && OpType != core::kNull) || slots_ == 1, "NOT and NULL have one input");

  public:
    static constexpr bool kSupported = std::is_unsigned_v<bitpack_t_> && sizeof(bitpack_t_) <= sizeof(std::uint64_t);
    static_assert(kSupported, "gates need an unsigned bitpack of at most 64 bits");

    /// Bitpacks per work-item.
    static constexpr std::uint32_t kLanes = lanes_form_ ? kGateCpuLanes : 1;
    static constexpr std::size_t kWorkGroup = lanes_form_ ? 64 : 256;

  private:
    using block_t = event::gate_block<bitpack_t_, size_t_>;
    using gate_t = event::gate<bitpack_t_, size_t_>;
    using word_t = std::conditional_t<kSupported, bitpack_t_, std::uint64_t>;
    using lanes_t = typename gate_detail::lanes_of<word_t, static_cast<int>(kLanes)>::type;

    static constexpr bool kAndLike = OpType == core::kAnd || OpType == core::kNand;
    static constexpr bool kOrLike = OpType == core::kOr || OpType == core::kNor;

    block_t gates_;
    gate_geometry<kLanes, kWorkGroup> geometry_;

    template <typename V> [[gnu::always_inline]] static void combine(V &acc, const V &x) {
        if constexpr (kAndLike) {
            acc &= x;
        } else if constexpr (kOrLike) {
            acc |= x;
        } else if constexpr (OpType == core::kXor) {
            acc ^= x;
        } else {
            acc = x;
        }
    }

    template <typename V>
    [[gnu::always_inline]] static void input(V &x, const gate_t &g, const std::size_t w, const std::uint32_t i,
                                             const std::uint32_t negated) {
        gate_detail::load(x, g.inputs[i] + w);
        if (i >= negated) {
            x = ~x;
        }
    }

    template <typename V> [[gnu::always_inline]] static void evaluate(V &out, const gate_t &g, const std::size_t w) {
        V acc = V{};
        if constexpr (kAndLike) {
            acc = ~V{};
        }
        const auto negated = static_cast<std::uint32_t>(g.negated_inputs_offset);
        if constexpr (slots_ > 0) {
#pragma unroll
            for (std::uint32_t j = 0; j < slots_; ++j) {
                if (((g.input_mask >> j) & 1u) != 0) {
                    V x;
                    input(x, g, w, j, negated);
                    combine(acc, x);
                }
            }
        } else if constexpr (lanes_form_) {
            const auto n = static_cast<std::uint32_t>(g.num_inputs);
            const auto positive = negated < n ? negated : n;
            std::uint32_t i = 0;
#pragma unroll 4
            for (; i < positive; ++i) {
                V x;
                gate_detail::load(x, g.inputs[i] + w);
                combine(acc, x);
            }
#pragma unroll 4
            for (; i < n; ++i) {
                V x;
                gate_detail::load(x, g.inputs[i] + w);
                x = ~x;
                combine(acc, x);
            }
        } else {
            const auto n = static_cast<std::uint32_t>(g.num_inputs);
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
        if constexpr (OpType == core::kNand || OpType == core::kNor || OpType == core::kNot) {
            acc = ~acc;
        }
        out = acc;
    }

  public:
    op(const block_t &gates, const event::sample_shape<size_t_> &sample_shape,
       const gate_order order = gate_order::gate_major)
        : gates_(gates), geometry_(gates.count, sample_shape.num_bitpacks(), order) {}

    [[nodiscard]] sycl::nd_range<1> get_range() const { return geometry_.range(); }
    [[nodiscard]] std::size_t gates() const { return geometry_.gates(); }
    [[nodiscard]] std::size_t items_per_gate() const { return geometry_.items_per_gate(); }

    void operator()(const sycl::nd_item<1> &item) const {
        if constexpr (kSupported) {
            typename gate_geometry<kLanes, kWorkGroup>::position at;
            if (!geometry_.locate(item, at)) {
                return;
            }
            const auto &g = gates_[at.gate];
            const std::size_t bitpacks = geometry_.bitpacks();
            if (at.first + kLanes <= bitpacks) {
                lanes_t r;
                evaluate(r, g, at.first);
                gate_detail::store(g.buffer + at.first, r);
            } else {
                for (std::size_t w = at.first; w < bitpacks; ++w) {
                    bitpack_t_ r;
                    evaluate(r, g, w);
                    g.buffer[w] = r;
                }
            }
        }
    }
};

} // namespace scram::mc::kernel
