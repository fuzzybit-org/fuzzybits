// Cardinality gate evaluation for the lab microbenchmark, templated on the lane
// type V: one 64-bit word (64 trials) on GPUs, or a clang extended vector of
// words on CPUs. A gate is true where the number of counted inputs lies in [lo,
// hi]: inputs [0, neg) count as they are, [neg, n) complemented, and hi >= n
// means no upper bound. atleast(k) = [k, n], atmost(k) = [0, k], exactly(k) =
// [k, k].
#pragma once

#include <sycl/sycl.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>

namespace card {

using word = std::uint64_t;
typedef word v4 __attribute__((ext_vector_type(4)));
typedef word v8 __attribute__((ext_vector_type(8)));

template <typename V> inline constexpr int lanes_v = sizeof(V) / sizeof(word);

template <typename V> [[gnu::always_inline]] inline V ld(const word *p) {
    if constexpr (lanes_v<V> == 1) {
        return *p;
    } else {
        V v;
        __builtin_memcpy(&v, p, sizeof(V));
        return v;
    }
}

template <typename V> [[gnu::always_inline]] inline void st(word *p, V v) {
    if constexpr (lanes_v<V> == 1) {
        *p = v;
    } else {
        __builtin_memcpy(p, &v, sizeof(V));
    }
}

template <typename V> [[gnu::always_inline]] inline V all(bool b) { return b ? ~V{} : V{}; }

struct gate_desc {
    const word *const *in; // n input buffers
    word *out;
    std::uint32_t n;
    std::uint32_t neg; // inputs [neg, n) are complemented
    std::uint32_t lo;
    std::uint32_t hi;   // hi >= n: no upper bound
    std::uint32_t cls;  // klass, for the auto kernel
    std::uint32_t arg;  // thermometer depth, or binary counter planes
    std::uint64_t mask; // fixed-size kernels: bit j set iff slot j is an input
};

enum klass : std::uint32_t { k_false = 0, k_true, k_therm, k_bin };

// ------------------------------------------------------------------------------------------------
// thermometer s[j] = lanes with at least j + 1 counted inputs so far,
// saturating at depth D. One 3-input op per level.
template <std::uint32_t D, typename V> [[gnu::always_inline]] inline void therm_step(V (&s)[D], V x) {
#pragma unroll
    for (std::uint32_t j = D - 1; j > 0; --j) {
        s[j] |= s[j - 1] & x;
    }
    s[0] |= x;
}

// Needs lo <= D and (hi >= n or hi < D): the host picks the counted side (ones
// or zeros) that satisfies this.
template <std::uint32_t D, typename V>
[[gnu::always_inline]] inline V therm_range(const V (&s)[D], std::uint32_t lo, std::uint32_t hi) {
    V r = ~V{};
    V u = ~V{};
#pragma unroll
    for (std::uint32_t j = 0; j < D; ++j) {
        r = (lo == j + 1) ? s[j] : r;
        u = (hi == j) ? ~s[j] : u;
    }
    return r & u;
}

template <std::uint32_t D, typename V> [[gnu::always_inline]] inline V eval_therm(const gate_desc &g, std::size_t w) {
    V s[D];
#pragma unroll
    for (std::uint32_t j = 0; j < D; ++j) {
        s[j] = V{};
    }
    const word *const *in = g.in;
    const std::uint32_t n = g.n;
    const std::uint32_t neg = g.neg;
    std::uint32_t i = 0;
#pragma unroll 4
    for (; i < neg; ++i) {
        therm_step<D>(s, ld<V>(in[i] + w));
    }
#pragma unroll 4
    for (; i < n; ++i) {
        therm_step<D>(s, ~ld<V>(in[i] + w));
    }
    return therm_range<D>(s, g.lo, g.hi);
}

// -----------------------------------------------------------------------------------------
// binary counters c[j] = bit j of the per-lane count (bit-sliced).
template <int F, int Q, typename V> [[gnu::always_inline]] inline void add_at(V (&c)[Q], V carry) {
#pragma unroll
    for (int j = F; j < Q; ++j) {
        const V t = c[j] & carry;
        c[j] ^= carry;
        carry = t;
    }
}

// Full adder: a + b + c = 2 * carry + sum. Two 3-input ops (LOP3 / vpternlog),
// five 2-input ones.
template <typename V> [[gnu::always_inline]] inline void fa(V &carry, V &sum, V a, V b, V c) {
    const V u = a ^ b;
    sum = u ^ c;
    carry = (a & b) | (u & c);
}

// Harley-Seal: 16 inputs into the ones/twos/fours/eights planes c[0..3] with 15
// full adders; the sixteens word ripples into c[4..Q).
template <int Q, typename V> [[gnu::always_inline]] inline void hs16(V (&c)[Q], const V (&x)[16]) {
    static_assert(Q >= 4);
    V tA, tB, fA, fB, eA, eB, s16;
    fa(tA, c[0], c[0], x[0], x[1]);
    fa(tB, c[0], c[0], x[2], x[3]);
    fa(fA, c[1], c[1], tA, tB);
    fa(tA, c[0], c[0], x[4], x[5]);
    fa(tB, c[0], c[0], x[6], x[7]);
    fa(fB, c[1], c[1], tA, tB);
    fa(eA, c[2], c[2], fA, fB);
    fa(tA, c[0], c[0], x[8], x[9]);
    fa(tB, c[0], c[0], x[10], x[11]);
    fa(fA, c[1], c[1], tA, tB);
    fa(tA, c[0], c[0], x[12], x[13]);
    fa(tB, c[0], c[0], x[14], x[15]);
    fa(fB, c[1], c[1], tA, tB);
    fa(eB, c[2], c[2], fA, fB);
    fa(s16, c[3], c[3], eA, eB);
    add_at<4>(c, s16);
}

// Harley-Seal over 8 inputs: 7 full adders into c[0..2]; the eights word
// ripples into c[3..Q). Fewer live registers than hs16.
template <int Q, typename V> [[gnu::always_inline]] inline void hs8(V (&c)[Q], const V (&x)[8]) {
    static_assert(Q >= 4);
    V tA, tB, fA, fB, e8;
    fa(tA, c[0], c[0], x[0], x[1]);
    fa(tB, c[0], c[0], x[2], x[3]);
    fa(fA, c[1], c[1], tA, tB);
    fa(tA, c[0], c[0], x[4], x[5]);
    fa(tB, c[0], c[0], x[6], x[7]);
    fa(fB, c[1], c[1], tA, tB);
    fa(e8, c[2], c[2], fA, fB);
    add_at<3>(c, e8);
}

template <int BLK, int Q, typename V> [[gnu::always_inline]] inline void hs(V (&c)[Q], const V (&x)[BLK]) {
    if constexpr (BLK == 16) {
        hs16(c, x);
    } else {
        hs8(c, x);
    }
}

template <bool NEG, int BLK, typename V>
[[gnu::always_inline]] inline void load_blk(V (&x)[BLK], const word *const *in, std::size_t w) {
#pragma unroll
    for (int j = 0; j < BLK; ++j) {
        const V v = ld<V>(in[j] + w);
        x[j] = NEG ? ~v : v;
    }
}

// Inputs [i, i + BLK) of the gate, complementing those >= neg and zeroing those
// >= n.
template <int BLK, typename V>
[[gnu::always_inline]] inline void load_blk_masked(V (&x)[BLK], const word *const *in, std::size_t w, std::uint32_t i,
                                                   std::uint32_t neg, std::uint32_t n) {
#pragma unroll
    for (std::uint32_t j = 0; j < BLK; ++j) {
        const std::uint32_t k = i + j;
        V v = V{};
        if (k < n) {
            v = ld<V>(in[k] + w);
            if (k >= neg) {
                v = ~v;
            }
        }
        x[j] = v;
    }
}

// Lanes whose count is >= K: the carry out of count + (2^Q - K), one majority
// per plane.
template <int Q, typename V> [[gnu::always_inline]] inline V count_ge(const V (&c)[Q], std::uint64_t K) {
    if (K == 0) {
        return ~V{};
    }
    if (K >= (std::uint64_t{1} << Q)) {
        return V{};
    }
    const std::uint64_t M = (std::uint64_t{1} << Q) - K;
    V carry = V{};
#pragma unroll
    for (int j = 0; j < Q; ++j) {
        const V m = all<V>(((M >> j) & 1u) != 0);
        carry = (c[j] & carry) | (m & (c[j] | carry));
    }
    return carry;
}

template <int Q, typename V>
[[gnu::always_inline]] inline V count_in(const V (&c)[Q], std::uint32_t lo, std::uint32_t hi, std::uint32_t n) {
    V r = count_ge<Q>(c, lo);
    if (hi < n) {
        r &= ~count_ge<Q>(c, std::uint64_t{hi} + 1);
    }
    return r;
}

template <int P, int BLK, typename V> [[gnu::always_inline]] inline V eval_bin(const gate_desc &g, std::size_t w) {
    constexpr int Q = P < 4 ? 4 : P;
    V c[Q];
#pragma unroll
    for (int j = 0; j < Q; ++j) {
        c[j] = V{};
    }
    const word *const *in = g.in;
    const std::uint32_t n = g.n;
    const std::uint32_t neg = g.neg;
    std::uint32_t i = 0;
    V x[BLK];
    for (; i + BLK <= neg; i += BLK) {
        load_blk<false>(x, in + i, w);
        hs<BLK>(c, x);
    }
    if (i < neg) { // the block that straddles neg
        load_blk_masked(x, in, w, i, neg, n);
        hs<BLK>(c, x);
        i += BLK;
    }
    for (; i + BLK <= n; i += BLK) {
        load_blk<true>(x, in + i, w);
        hs<BLK>(c, x);
    }
    if (i < n) {
        load_blk_masked(x, in, w, i, neg, n);
        hs<BLK>(c, x);
    }
    return count_in<Q>(c, g.lo, g.hi, n);
}

template <int P, typename V> [[gnu::always_inline]] inline V eval_bin16(const gate_desc &g, std::size_t w) {
    return eval_bin<P, 16, V>(g, w);
}

template <int P, typename V> [[gnu::always_inline]] inline V eval_bin8(const gate_desc &g, std::size_t w) {
    return eval_bin<P, 8, V>(g, w);
}

template <int P, typename V> [[gnu::always_inline]] inline V eval_ripple(const gate_desc &g, std::size_t w) {
    V c[P];
#pragma unroll
    for (int j = 0; j < P; ++j) {
        c[j] = V{};
    }
    const word *const *in = g.in;
    const std::uint32_t n = g.n;
    const std::uint32_t neg = g.neg;
    std::uint32_t i = 0;
    for (; i < neg; ++i) {
        add_at<0>(c, ld<V>(in[i] + w));
    }
    for (; i < n; ++i) {
        add_at<0>(c, ~ld<V>(in[i] + w));
    }
    return count_in<P>(c, g.lo, g.hi, n);
}

// -----------------------------------------------------------------------------------------------
// dispatch
template <typename V> [[gnu::always_inline]] inline V eval_therm_upto8(const gate_desc &g, std::size_t w) {
    switch (g.arg) {
    case 1:
        return eval_therm<1, V>(g, w);
    case 2:
        return eval_therm<2, V>(g, w);
    case 3:
        return eval_therm<3, V>(g, w);
    case 4:
        return eval_therm<4, V>(g, w);
    case 5:
        return eval_therm<5, V>(g, w);
    case 6:
        return eval_therm<6, V>(g, w);
    case 7:
        return eval_therm<7, V>(g, w);
    default:
        return eval_therm<8, V>(g, w);
    }
}

template <typename V> [[gnu::always_inline]] inline V eval_therm_upto4(const gate_desc &g, std::size_t w) {
    switch (g.arg) {
    case 1:
        return eval_therm<1, V>(g, w);
    case 2:
        return eval_therm<2, V>(g, w);
    case 3:
        return eval_therm<3, V>(g, w);
    default:
        return eval_therm<4, V>(g, w);
    }
}

#define CARD_PLANES_SWITCH(FN)                                                                                         \
    switch (planes) {                                                                                                  \
    case 0:                                                                                                            \
    case 1:                                                                                                            \
    case 2:                                                                                                            \
    case 3:                                                                                                            \
    case 4:                                                                                                            \
        return FN<4, V>(g, w);                                                                                         \
    case 5:                                                                                                            \
        return FN<5, V>(g, w);                                                                                         \
    case 6:                                                                                                            \
        return FN<6, V>(g, w);                                                                                         \
    case 7:                                                                                                            \
        return FN<7, V>(g, w);                                                                                         \
    case 8:                                                                                                            \
        return FN<8, V>(g, w);                                                                                         \
    case 9:                                                                                                            \
        return FN<9, V>(g, w);                                                                                         \
    case 10:                                                                                                           \
        return FN<10, V>(g, w);                                                                                        \
    case 11:                                                                                                           \
        return FN<11, V>(g, w);                                                                                        \
    case 12:                                                                                                           \
        return FN<12, V>(g, w);                                                                                        \
    case 13:                                                                                                           \
    case 14:                                                                                                           \
    case 15:                                                                                                           \
    case 16:                                                                                                           \
        return FN<16, V>(g, w);                                                                                        \
    default:                                                                                                           \
        return FN<32, V>(g, w);                                                                                        \
    }

template <typename V, int BLK = 16> [[gnu::always_inline]] inline V eval_bin_any(const gate_desc &g, std::size_t w) {
    const std::uint32_t planes = g.arg;
    if constexpr (BLK == 16) {
        CARD_PLANES_SWITCH(eval_bin16)
    } else {
        CARD_PLANES_SWITCH(eval_bin8)
    }
}

template <typename V> [[gnu::always_inline]] inline V eval_ripple_any(const gate_desc &g, std::size_t w) {
    const std::uint32_t planes = g.arg;
    CARD_PLANES_SWITCH(eval_ripple)
}

template <typename V> [[gnu::always_inline]] inline V eval_auto(const gate_desc &g, std::size_t w) {
    switch (g.cls) {
    case k_false:
        return V{};
    case k_true:
        return ~V{};
    case k_therm:
        return eval_therm_upto4<V>(g, w);
    default:
        return eval_bin_any<V>(g, w);
    }
}

// -----------------------------------------------------------------------------------------
// fixed shapes Everything but the complement mask is a compile-time constant,
// so the compiler folds the thermometer into the minimal network (2-of-3 is one
// majority).
template <int N, int LO, int HI, typename V>
[[gnu::always_inline]] inline V eval_fixed(const gate_desc &g, std::size_t w) {
    constexpr int H = HI < N ? HI : N;
    constexpr int D1 = H < N ? H + 1 : LO;
    constexpr int D0 = LO > 0 ? N - LO + 1 : N - H;
    constexpr bool flip = D0 < D1;
    constexpr int D = flip ? D0 : D1;
    constexpr int lo = flip ? N - H : LO;
    constexpr int hi = flip ? N - LO : H;
    V s[D];
#pragma unroll
    for (int j = 0; j < D; ++j) {
        s[j] = V{};
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        V x = ld<V>(g.in[j] + w);
        x ^= all<V>((static_cast<std::uint32_t>(j) >= g.neg) != flip);
        therm_step<D>(s, x);
    }
    V r = lo == 0 ? ~V{} : s[lo > 0 ? lo - 1 : 0];
    if constexpr (hi < N) {
        r &= ~s[hi];
    }
    return r;
}

// ----------------------------------------------------------- small-n kernels
// Compile-time N, runtime lo/hi/neg: fully unrolled bit-sliced count, then the
// carry-chain comparison.
template <int N, typename V> [[gnu::always_inline]] inline V eval_fixedN(const gate_desc &g, std::size_t w) {
    constexpr int P = std::bit_width(static_cast<unsigned>(N));
    V c[P];
#pragma unroll
    for (int j = 0; j < P; ++j) {
        c[j] = V{};
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        V x = ld<V>(g.in[j] + w);
        x ^= all<V>(static_cast<std::uint32_t>(j) >= g.neg);
        add_at<0>(c, x);
    }
    return count_in<P>(c, g.lo, g.hi, N);
}

// Compile-time N and thermometer depth D, runtime lo/hi/neg on the counted side.
template <int N, int D, typename V> [[gnu::always_inline]] inline V eval_fixedND(const gate_desc &g, std::size_t w) {
    V s[D];
#pragma unroll
    for (int j = 0; j < D; ++j) {
        s[j] = V{};
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        V x = ld<V>(g.in[j] + w);
        x ^= all<V>(static_cast<std::uint32_t>(j) >= g.neg);
        therm_step<D>(s, x);
    }
    return therm_range<D>(s, g.lo, g.hi);
}

// Runtime n, inputs loaded in blocks of BLK (predicated tail), so the loads of
// a block are in flight together.
template <std::uint32_t D, int BLK, typename V>
[[gnu::always_inline]] inline V eval_therm_blk(const gate_desc &g, std::size_t w) {
    V s[D];
#pragma unroll
    for (std::uint32_t j = 0; j < D; ++j) {
        s[j] = V{};
    }
    const word *const *in = g.in;
    const std::uint32_t n = g.n;
    const std::uint32_t neg = g.neg;
    for (std::uint32_t i = 0; i < n; i += BLK) {
        V x[BLK];
        load_blk_masked(x, in, w, i, neg, n);
#pragma unroll
        for (int j = 0; j < BLK; ++j) {
            therm_step<D>(s, x[j]);
        }
    }
    return therm_range<D>(s, g.lo, g.hi);
}

template <typename V, int BLK> [[gnu::always_inline]] inline V eval_therm_blk_upto8(const gate_desc &g, std::size_t w) {
    switch (g.arg) {
    case 1:
        return eval_therm_blk<1, BLK, V>(g, w);
    case 2:
        return eval_therm_blk<2, BLK, V>(g, w);
    case 3:
        return eval_therm_blk<3, BLK, V>(g, w);
    case 4:
        return eval_therm_blk<4, BLK, V>(g, w);
    case 5:
        return eval_therm_blk<5, BLK, V>(g, w);
    case 6:
        return eval_therm_blk<6, BLK, V>(g, w);
    case 7:
        return eval_therm_blk<7, BLK, V>(g, w);
    default:
        return eval_therm_blk<8, BLK, V>(g, w);
    }
}

template <typename V> [[gnu::always_inline]] inline V eval_small_switch(const gate_desc &g, std::size_t w) {
    switch (g.n) {
    case 1:
        return eval_fixedN<1, V>(g, w);
    case 2:
        return eval_fixedN<2, V>(g, w);
    case 3:
        return eval_fixedN<3, V>(g, w);
    case 4:
        return eval_fixedN<4, V>(g, w);
    case 5:
        return eval_fixedN<5, V>(g, w);
    case 6:
        return eval_fixedN<6, V>(g, w);
    case 7:
        return eval_fixedN<7, V>(g, w);
    default:
        return eval_fixedN<8, V>(g, w);
    }
}

// Counter limited to 8 planes (n < 256), so its registers do not grow with the
// 32-plane path.
template <typename V> [[gnu::always_inline]] inline V eval_bin8_mid(const gate_desc &g, std::size_t w) {
    switch (g.arg) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
        return eval_bin<4, 8, V>(g, w);
    case 5:
        return eval_bin<5, 8, V>(g, w);
    case 6:
        return eval_bin<6, 8, V>(g, w);
    case 7:
        return eval_bin<7, 8, V>(g, w);
    default:
        return eval_bin<8, 8, V>(g, w);
    }
}

template <typename V> [[gnu::always_inline]] inline V eval_small16_switch(const gate_desc &g, std::size_t w) {
    switch (g.n) {
    case 1:
        return eval_fixedN<1, V>(g, w);
    case 2:
        return eval_fixedN<2, V>(g, w);
    case 3:
        return eval_fixedN<3, V>(g, w);
    case 4:
        return eval_fixedN<4, V>(g, w);
    case 5:
        return eval_fixedN<5, V>(g, w);
    case 6:
        return eval_fixedN<6, V>(g, w);
    case 7:
        return eval_fixedN<7, V>(g, w);
    case 8:
        return eval_fixedN<8, V>(g, w);
    case 9:
        return eval_fixedN<9, V>(g, w);
    case 10:
        return eval_fixedN<10, V>(g, w);
    case 11:
        return eval_fixedN<11, V>(g, w);
    case 12:
        return eval_fixedN<12, V>(g, w);
    case 13:
        return eval_fixedN<13, V>(g, w);
    case 14:
        return eval_fixedN<14, V>(g, w);
    case 15:
        return eval_fixedN<15, V>(g, w);
    default:
        return eval_fixedN<16, V>(g, w);
    }
}

// ------------------------------------------------- fixed size with an input mask
// Compile-time slot count N; slot j is an input iff bit j of g.mask is set (its
// pointer is not read otherwise), complemented iff j >= g.neg. Up to 8 slots the
// count ripples; above, 8-slot blocks go through the Harley-Seal adder.
template <int N, typename V>
[[gnu::always_inline]] inline void masked_slot(V &x, const gate_desc &g, std::size_t w, int j) {
    x = V{};
    if (((g.mask >> j) & 1u) != 0) {
        x = ld<V>(g.in[j] + w);
        if (static_cast<std::uint32_t>(j) >= g.neg) {
            x = ~x;
        }
    }
}

template <int N, typename V> [[gnu::always_inline]] inline V eval_fixedM(const gate_desc &g, std::size_t w) {
    // As src/mc/kernel/cardinality.h: blocks of 8 slots through the Harley-Seal
    // adder above 8 slots, the rest rippled.
    static_assert(N >= 1 && N <= 64);
    constexpr int P = std::bit_width(static_cast<unsigned>(N));
    constexpr int blocks = N > 8 ? N / 8 : 0;
    V c[P];
#pragma unroll
    for (int j = 0; j < P; ++j) {
        c[j] = V{};
    }
    if constexpr (blocks > 0) {
#pragma unroll
        for (int b = 0; b < blocks; ++b) {
            V x[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                masked_slot<N>(x[j], g, w, 8 * b + j);
            }
            hs8(c, x);
        }
    }
#pragma unroll
    for (int j = 8 * blocks; j < N; ++j) {
        V x;
        masked_slot<N>(x, g, w, j);
        add_at<0>(c, x);
    }
    return count_in<P>(c, g.lo, g.hi, g.n);
}

} // namespace card
