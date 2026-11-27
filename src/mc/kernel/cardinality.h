/**
 * @file cardinality.h
 * @brief SYCL kernels for cardinality gates: true where the number of true inputs lies in [low, high]
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details At least k of n is [k, n], at most k is [0, k] and exactly k is [k, k]. Inputs at or after the
 * gate's negated_inputs_offset count when they are false.
 *
 * Counting is bit-sliced: bit j of a bitpack is trial j, and the per-trial counts are held as bit planes
 * (plane p holds bit p of every trial's count), so whole-word logic counts all of a bitpack's trials at
 * once. A work-item evaluates one bitpack of one gate on GPUs, and kLanes consecutive bitpacks as one vector
 * on CPUs, whose JIT does not vectorize across work-items. The gate comes from the work-group index, so its
 * fields are uniform across the group. Three forms, which the host picks per gate (`cardinality_form_for`):
 *
 * - fixed: up to 16 inputs. A compile-time number of slots (`kCardinalitySlots`), into which the kernel
 *   builder pads each gate; the gate's input_mask says which slots hold an input. The count is unrolled, so
 *   a work-item issues all its loads at once and needs few registers.
 * - counter: 17 to 255 inputs. Carry-save adders over blocks of 8 inputs (Harley-Seal), so an input costs
 *   about two three-input logic operations, and the planes are compared with low and high once at the end.
 *   Above 16 inputs this beats a fixed slot count, whose unrolled loads raise register pressure.
 * - wide: 256 inputs or more, the counter with up to 32 planes. It is its own kernel because a kernel's
 *   registers are those of its largest case, and 32 planes would lower the counter's occupancy.
 *
 * Every form writes the same bitpacks on every backend.
 */

#pragma once

#include "mc/event/node.tpp"
#include "mc/event/sample_shape.h"
#include "mc/kernel/gate_lanes.h"

#include <sycl/sycl.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace scram::mc::kernel {

enum class cardinality_form {
    /// Up to 16 inputs, in a compile-time number of slots (`kCardinalitySlots`) given by an input mask.
    fixed,
    /// 17 to 255 inputs, counted by carry-save adders over blocks of 8 inputs into at most 8 planes.
    counter,
    /// 256 inputs or more: the counter with up to 32 planes.
    wide,
};

/// Slot counts of the fixed form. A gate takes the smallest that holds its inputs, so at most a quarter of
/// its slots are padding (9 inputs in 12); each slot count is a kernel of its own.
inline constexpr std::array<std::uint32_t, 8> kCardinalitySlots{1, 2, 3, 4, 6, 8, 12, 16};

[[nodiscard]] constexpr cardinality_form cardinality_form_for(const std::size_t num_inputs) {
    if (num_inputs <= kCardinalitySlots.back()) {
        return cardinality_form::fixed;
    }
    return num_inputs < 256 ? cardinality_form::counter : cardinality_form::wide;
}

/// Slots of the fixed form for a gate with num_inputs inputs; 0 for the other forms.
[[nodiscard]] constexpr std::uint32_t cardinality_slots_for(const std::size_t num_inputs) {
    for (const auto slots : kCardinalitySlots) {
        if (num_inputs <= slots) {
            return slots;
        }
    }
    return 0;
}

namespace cardinality_detail {

using gate_detail::load;
using gate_detail::store;

/// Planes c[F..Q) += x, rippling.
template <int F, int Q, typename V> [[gnu::always_inline]] inline void add_at(V (&c)[Q], const V &x) {
    V carry = x;
#pragma unroll
    for (int p = F; p < Q; ++p) {
        const V t = c[p] & carry;
        c[p] ^= carry;
        carry = t;
    }
}

/// a + b + c = 2 * carry + sum: two three-input operations where the hardware has them. The outputs may alias
/// the inputs.
template <typename V>
[[gnu::always_inline]] inline void full_add(V &carry, V &sum, const V &a, const V &b, const V &c) {
    const V u = a ^ b;
    const V ab = a & b;
    const V uc = u & c;
    const V s = u ^ c;
    carry = ab | uc;
    sum = s;
}

/// Adds 8 inputs into the ones, twos and fours planes c[0..2] with 7 full adders; the eights word ripples
/// into c[3..Q).
template <int Q, typename V> [[gnu::always_inline]] inline void add8(V (&c)[Q], const V (&x)[8]) {
    static_assert(Q >= 4);
    V twos_a, twos_b, fours_a, fours_b, eights;
    full_add(twos_a, c[0], c[0], x[0], x[1]);
    full_add(twos_b, c[0], c[0], x[2], x[3]);
    full_add(fours_a, c[1], c[1], twos_a, twos_b);
    full_add(twos_a, c[0], c[0], x[4], x[5]);
    full_add(twos_b, c[0], c[0], x[6], x[7]);
    full_add(fours_b, c[1], c[1], twos_a, twos_b);
    full_add(eights, c[2], c[2], fours_a, fours_b);
    add_at<3>(c, eights);
}

/// Trials whose count is at least k: the carry out of count + (2^Q - k), one majority per plane.
template <int Q, typename V>
[[gnu::always_inline]] inline void at_least(V &out, const V (&c)[Q], const std::uint64_t k) {
    if (k == 0) {
        out = ~V{};
        return;
    }
    if (k >= (std::uint64_t{1} << Q)) {
        out = V{};
        return;
    }
    const std::uint64_t m = (std::uint64_t{1} << Q) - k;
    V carry = V{};
#pragma unroll
    for (int p = 0; p < Q; ++p) {
        V bit = V{};
        if (((m >> p) & 1u) != 0) {
            bit = ~V{};
        }
        carry = (c[p] & carry) | (bit & (c[p] | carry));
    }
    out = carry;
}

/// Trials whose count lies in [low, high]; high >= n means no upper bound.
template <int Q, typename V>
[[gnu::always_inline]] inline void in_range(V &out, const V (&c)[Q], const std::uint64_t low, const std::uint64_t high,
                                            const std::uint64_t n) {
    at_least<Q>(out, c, low);
    if (high < n) {
        V above;
        at_least<Q>(above, c, high + 1);
        out &= ~above;
    }
}

/// Slot j of a fixed-form gate: zero unless bit j of its input mask is set (the slot's pointer is not read
/// then), complemented from negated_inputs_offset on.
template <typename V, typename gate_t_>
[[gnu::always_inline]] inline void slot(V &x, const gate_t_ &g, const std::size_t w, const std::uint32_t j,
                                        const std::uint32_t negated) {
    x = V{};
    if (((g.input_mask >> j) & 1u) != 0) {
        load(x, g.inputs[j] + w);
        if (j >= negated) {
            x = ~x;
        }
    }
}

/// Fixed form with N slots: blocks of 8 through the carry-save adder, the rest rippled.
template <int N, typename V, typename gate_t_>
[[gnu::always_inline]] inline void fixed(V &out, const gate_t_ &g, const std::size_t w) {
    static_assert(N >= 1 && N <= 64);
    constexpr int Q = std::bit_width(static_cast<unsigned>(N));
    constexpr int blocks = N > 8 ? N / 8 : 0;
    V c[Q];
#pragma unroll
    for (int p = 0; p < Q; ++p) {
        c[p] = V{};
    }
    const auto negated = static_cast<std::uint32_t>(g.negated_inputs_offset);
    if constexpr (blocks > 0) {
#pragma unroll
        for (int b = 0; b < blocks; ++b) {
            V x[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                slot(x[j], g, w, static_cast<std::uint32_t>(8 * b + j), negated);
            }
            add8(c, x);
        }
    }
#pragma unroll
    for (int j = 8 * blocks; j < N; ++j) {
        V x;
        slot(x, g, w, static_cast<std::uint32_t>(j), negated);
        add_at<0>(c, x);
    }
    in_range<Q>(out, c, g.low, g.high, g.num_inputs);
}

template <bool negated_, typename V, typename bitpack_t_>
[[gnu::always_inline]] inline void load8(V (&x)[8], bitpack_t_ *const *inputs, const std::size_t w) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        load(x[j], inputs[j] + w);
        if constexpr (negated_) {
            x[j] = ~x[j];
        }
    }
}

/// Inputs [i, i + 8): complemented from `negated` on, zero from `n` on.
template <typename V, typename bitpack_t_>
[[gnu::always_inline]] inline void load8_masked(V (&x)[8], bitpack_t_ *const *inputs, const std::size_t w,
                                                const std::uint32_t i, const std::uint32_t negated,
                                                const std::uint32_t n) {
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j) {
        x[j] = V{};
        if (i + j < n) {
            load(x[j], inputs[i + j] + w);
            if (i + j >= negated) {
                x[j] = ~x[j];
            }
        }
    }
}

/// Counter form with Q planes (n < 2^Q).
template <int Q, typename V, typename gate_t_>
[[gnu::always_inline]] inline void counter(V &out, const gate_t_ &g, const std::size_t w) {
    V c[Q];
#pragma unroll
    for (int p = 0; p < Q; ++p) {
        c[p] = V{};
    }
    const auto n = static_cast<std::uint32_t>(g.num_inputs);
    const auto negated = static_cast<std::uint32_t>(g.negated_inputs_offset);
    std::uint32_t i = 0;
    V x[8];
    // Whole blocks of one polarity take the complement at compile time; at most two blocks per gate (the one
    // that straddles `negated`, and the last) need the masked load.
    for (; i + 8 <= negated; i += 8) {
        load8<false>(x, g.inputs + i, w);
        add8(c, x);
    }
    if (i < negated) {
        load8_masked(x, g.inputs, w, i, negated, n);
        add8(c, x);
        i += 8;
    }
    for (; i + 8 <= n; i += 8) {
        load8<true>(x, g.inputs + i, w);
        add8(c, x);
    }
    if (i < n) {
        load8_masked(x, g.inputs, w, i, negated, n);
        add8(c, x);
    }
    in_range<Q>(out, c, g.low, g.high, n);
}

/// Counter form: 16 < n < 256, so at most 8 planes.
template <typename V, typename gate_t_>
[[gnu::always_inline]] inline void counter_any(V &out, const gate_t_ &g, const std::size_t w) {
    switch (std::bit_width(static_cast<std::uint32_t>(g.num_inputs))) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
        return counter<4>(out, g, w);
    case 5:
        return counter<5>(out, g, w);
    case 6:
        return counter<6>(out, g, w);
    case 7:
        return counter<7>(out, g, w);
    default:
        return counter<8>(out, g, w);
    }
}

/// Wide form: plane counts are bucketed so that a few instantiations cover every input count below 2^32.
template <typename V, typename gate_t_>
[[gnu::always_inline]] inline void wide_any(V &out, const gate_t_ &g, const std::size_t w) {
    switch (std::bit_width(static_cast<std::uint32_t>(g.num_inputs))) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
        return counter<10>(out, g, w);
    case 11:
    case 12:
        return counter<12>(out, g, w);
    case 13:
    case 14:
    case 15:
    case 16:
        return counter<16>(out, g, w);
    default:
        return counter<32>(out, g, w);
    }
}

} // namespace cardinality_detail

/**
 * @brief Evaluates a block of cardinality gates of one form.
 *
 * @tparam slots_ the fixed form's slot count (one of kCardinalitySlots), 0 for the other forms.
 * @tparam lanes_form_ true on CPUs: a work-item evaluates kLanes consecutive bitpacks as one vector.
 */
template <typename bitpack_t_, typename size_t_, cardinality_form form_, std::uint32_t slots_, bool lanes_form_>
class cardinality {
    static_assert((form_ == cardinality_form::fixed) == (slots_ != 0));

  public:
    static constexpr bool kSupported = std::is_unsigned_v<bitpack_t_> && sizeof(bitpack_t_) <= sizeof(std::uint64_t);
    static_assert(kSupported, "cardinality gates need an unsigned bitpack of at most 64 bits");

    /// Bitpacks per work-item.
    static constexpr std::uint32_t kLanes = lanes_form_ ? kGateCpuLanes : 1;
    static constexpr std::size_t kWorkGroup = lanes_form_ ? 64 : 256;

  private:
    using block_t = event::cardinality_gate_block<bitpack_t_, size_t_>;
    using word_t = std::conditional_t<kSupported, bitpack_t_, std::uint64_t>;
    using lanes_t = typename gate_detail::lanes_of<word_t, static_cast<int>(kLanes)>::type;

    block_t gates_;
    gate_geometry<kLanes, kWorkGroup> geometry_;

    template <typename V>
    [[gnu::always_inline]] static void evaluate(V &out, const typename block_t::gate_t &g, const std::size_t w) {
        if constexpr (form_ == cardinality_form::fixed) {
            cardinality_detail::fixed<static_cast<int>(slots_)>(out, g, w);
        } else if constexpr (form_ == cardinality_form::counter) {
            cardinality_detail::counter_any(out, g, w);
        } else {
            cardinality_detail::wide_any(out, g, w);
        }
    }

  public:
    cardinality(const block_t &gates, const event::sample_shape<size_t_> &sample_shape,
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
                cardinality_detail::store(g.buffer + at.first, r);
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
