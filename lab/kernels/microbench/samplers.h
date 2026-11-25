// Sampler variants for the Philox Bernoulli microbenchmark (bench.cpp): a replica of the 32-bit sampler that
// develop used before the bit-sliced one, the split low multiply (#16), and the bit-sliced variants that were
// tried for #164 (plain, 16-lane, persistent, explicit SIMD). Not the production code: that is
// src/mc/prng/bernoulli.h, which bench.cpp includes.
#pragma once
#include <cstdint>
#include <sycl/sycl.hpp>

namespace lab {

// ---------------------------------------------------------------- Philox
// 4x32-10, production replica
struct state128 {
    std::uint32_t x[4];
};

static constexpr std::uint32_t kM0 = 0xD2511F53u;
static constexpr std::uint32_t kM1 = 0xCD9E8D57u;
static constexpr std::uint32_t kW0 = 0x9E3779B9u;
static constexpr std::uint32_t kW1 = 0xBB67AE85u;
static constexpr std::uint32_t kK0 = 382307844u;
static constexpr std::uint32_t kK1 = 293830103u;

// Production mulhilo (one 64-bit product, both halves from it).
[[gnu::always_inline]] inline std::uint32_t mulhilo_wide(std::uint32_t a, std::uint32_t b, std::uint32_t *hi) {
    std::uint64_t p = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
    *hi = static_cast<std::uint32_t>(p >> 32);
    return static_cast<std::uint32_t>(p);
}
// #16: low half with a separate 32-bit multiply.
[[gnu::always_inline]] inline std::uint32_t mulhilo_split(std::uint32_t a, std::uint32_t b, std::uint32_t *hi) {
    *hi = static_cast<std::uint32_t>((static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b)) >> 32);
    return a * b;
}

template <bool Split> [[gnu::always_inline]] inline void philox_round(state128 &c, std::uint32_t k0, std::uint32_t k1) {
    std::uint32_t hi0, hi1;
    const std::uint32_t lo0 = Split ? mulhilo_split(kM0, c.x[0], &hi0) : mulhilo_wide(kM0, c.x[0], &hi0);
    const std::uint32_t lo1 = Split ? mulhilo_split(kM1, c.x[2], &hi1) : mulhilo_wide(kM1, c.x[2], &hi1);
    c.x[0] = hi1 ^ c.x[1] ^ k0;
    c.x[1] = lo1;
    c.x[2] = hi0 ^ c.x[3] ^ k1;
    c.x[3] = lo0;
}

template <bool Split, int Rounds = 10> [[gnu::always_inline]] inline state128 philox(state128 c) {
#pragma unroll
    for (int i = 0; i < Rounds; ++i) {
        philox_round<Split>(c, kK0 + static_cast<std::uint32_t>(i) * kW0, kK1 + static_cast<std::uint32_t>(i) * kW1);
    }
    return c;
}

[[gnu::always_inline]] inline state128 counter(std::uint32_t pdag_idx, std::uint32_t event_idx,
                                               std::uint32_t sample_idx, std::uint32_t iteration) {
    return state128{{pdag_idx + 1, event_idx + 1, sample_idx + 1, (iteration + 1) << 6}};
}

// ---------------------------------------------------------------- A:
// production sampler (32-bit thresholds)
template <bool Split>
[[gnu::always_inline]] inline std::uint64_t sample_production(const state128 &base, std::uint64_t threshold32) {
    std::uint64_t out = 0;
#pragma unroll
    for (std::uint32_t g = 0; g < 16; ++g) {
        state128 c = base;
        c.x[3] += g;
        const state128 r = philox<Split>(c);
        const std::uint32_t off = 4 * g;
        out |= (static_cast<std::uint64_t>(r.x[0]) < threshold32 ? 1ull : 0ull) << (off + 0);
        out |= (static_cast<std::uint64_t>(r.x[1]) < threshold32 ? 1ull : 0ull) << (off + 1);
        out |= (static_cast<std::uint64_t>(r.x[2]) < threshold32 ? 1ull : 0ull) << (off + 2);
        out |= (static_cast<std::uint64_t>(r.x[3]) < threshold32 ? 1ull : 0ull) << (off + 3);
    }
    return out;
}

// ---------------------------------------------------------------- bit-sliced
// definition Lane j of the bitpack is 1 iff U_j < T, where U_j = sum_k r_k[j]
// 2^(63-k) and r_k is the k-th 64-bit random word of this bitpack: r_{2g} = w0
// | w1<<32, r_{2g+1} = w2 | w3<<32 with (w0..w3) = Philox(counter + g). T =
// floor(p 2^64); p = 1 is handled by the caller.
struct bs_state {
    std::uint32_t lt0, lt1, eq0, eq1;
};

// One comparison step against T bit `m` (all ones or all zeros).
[[gnu::always_inline]] inline void bs_step(bs_state &s, std::uint32_t r0, std::uint32_t r1, std::uint32_t m) {
    s.lt0 |= s.eq0 & ~r0 & m;
    s.eq0 &= ~(r0 ^ m);
    s.lt1 |= s.eq1 & ~r1 & m;
    s.eq1 &= ~(r1 ^ m);
}

// Host reference: scalar, one step at a time, no early-exit shortcuts beyond
// correctness.
inline std::uint64_t bs_reference(const state128 &base, std::uint64_t T) {
    if (T == 0) {
        return 0;
    }
    std::uint64_t lt = 0, eq = ~0ull;
    for (std::uint32_t g = 0; g < 32; ++g) {
        state128 c = base;
        c.x[3] += g;
        const state128 r = philox<false>(c);
        const std::uint64_t rr[2] = {static_cast<std::uint64_t>(r.x[0]) | (static_cast<std::uint64_t>(r.x[1]) << 32),
                                     static_cast<std::uint64_t>(r.x[2]) | (static_cast<std::uint64_t>(r.x[3]) << 32)};
        for (int h = 0; h < 2; ++h) {
            const int k = 2 * static_cast<int>(g) + h;
            const bool tb = (T >> (63 - k)) & 1u;
            if (tb) {
                lt |= eq & ~rr[h];
                eq &= rr[h];
            } else {
                eq &= ~rr[h];
            }
        }
    }
    return lt;
}

// C: rolled loop, one Philox per iteration, early exit when every lane is
// decided or T has no 1 bits left. `Tsh` holds the not-yet-compared bits of T
// at the top.
template <bool Split>
[[gnu::always_inline]] inline std::uint64_t sample_bitsliced(const state128 &base, std::uint64_t T) {
    if (T == 0) {
        return 0;
    }
    bs_state s{0u, 0u, ~0u, ~0u};
    std::uint64_t Tsh = T;
    std::uint32_t g = 0;
    while (true) {
        state128 c = base;
        c.x[3] += g;
        const state128 r = philox<Split>(c);
        const std::uint32_t th = static_cast<std::uint32_t>(Tsh >> 32);
        const std::uint32_t m0 = static_cast<std::uint32_t>(static_cast<std::int32_t>(th) >> 31);
        const std::uint32_t m1 = static_cast<std::uint32_t>(static_cast<std::int32_t>(th << 1) >> 31);
        bs_step(s, r.x[0], r.x[1], m0);
        bs_step(s, r.x[2], r.x[3], m1);
        Tsh <<= 2;
        ++g;
        if (((s.eq0 | s.eq1) == 0u) || Tsh == 0) {
            break;
        }
    }
    return static_cast<std::uint64_t>(s.lt0) | (static_cast<std::uint64_t>(s.lt1) << 32);
}

// D: unrolled prefix of P generations (no exit checks), then the rolled loop.
template <bool Split, int P>
[[gnu::always_inline]] inline std::uint64_t sample_bitsliced_prefix(const state128 &base, std::uint64_t T) {
    if (T == 0) {
        return 0;
    }
    bs_state s{0u, 0u, ~0u, ~0u};
    std::uint64_t Tsh = T;
#pragma unroll
    for (std::uint32_t g = 0; g < P; ++g) {
        state128 c = base;
        c.x[3] += g;
        const state128 r = philox<Split>(c);
        const std::uint32_t th = static_cast<std::uint32_t>(Tsh >> 32);
        bs_step(s, r.x[0], r.x[1], static_cast<std::uint32_t>(static_cast<std::int32_t>(th) >> 31));
        bs_step(s, r.x[2], r.x[3], static_cast<std::uint32_t>(static_cast<std::int32_t>(th << 1) >> 31));
        Tsh <<= 2;
    }
    std::uint32_t g = P;
    while (((s.eq0 | s.eq1) != 0u) && Tsh != 0) {
        state128 c = base;
        c.x[3] += g;
        const state128 r = philox<Split>(c);
        const std::uint32_t th = static_cast<std::uint32_t>(Tsh >> 32);
        bs_step(s, r.x[0], r.x[1], static_cast<std::uint32_t>(static_cast<std::int32_t>(th) >> 31));
        bs_step(s, r.x[2], r.x[3], static_cast<std::uint32_t>(static_cast<std::int32_t>(th << 1) >> 31));
        Tsh <<= 2;
        ++g;
    }
    return static_cast<std::uint64_t>(s.lt0) | (static_cast<std::uint64_t>(s.lt1) << 32);
}

// E: persistent. A work-item walks `count` bitpacks of one event (sample = first + i*stride), one
// Philox call per loop iteration; a decided bitpack is stored and the next one starts at once, so a
// SIMT group pays for the mean decision depth instead of the maximum over its lanes.
template <bool Split>
[[gnu::always_inline]] inline void
sample_bitsliced_persistent(std::uint64_t *row, std::uint32_t pdag_idx, std::uint32_t event_idx, std::uint32_t first,
                            std::uint32_t stride, std::uint32_t count, std::uint32_t iteration, std::uint64_t T) {
    if (T == 0) {
        for (std::uint32_t i = 0; i < count; ++i) {
            row[first + i * stride] = 0;
        }
        return;
    }
    std::uint32_t i = 0;
    std::uint32_t sample = first;
    bs_state s{0u, 0u, ~0u, ~0u};
    std::uint64_t Tsh = T;
    std::uint32_t g = 0;
    while (i < count) {
        state128 c = counter(pdag_idx, event_idx, sample, iteration);
        c.x[3] += g;
        const state128 r = philox<Split>(c);
        const std::uint32_t th = static_cast<std::uint32_t>(Tsh >> 32);
        bs_step(s, r.x[0], r.x[1], static_cast<std::uint32_t>(static_cast<std::int32_t>(th) >> 31));
        bs_step(s, r.x[2], r.x[3], static_cast<std::uint32_t>(static_cast<std::int32_t>(th << 1) >> 31));
        Tsh <<= 2;
        ++g;
        if (((s.eq0 | s.eq1) == 0u) || Tsh == 0) {
            row[sample] = static_cast<std::uint64_t>(s.lt0) | (static_cast<std::uint64_t>(s.lt1) << 32);
            ++i;
            sample += stride;
            s = bs_state{0u, 0u, ~0u, ~0u};
            Tsh = T;
            g = 0;
        }
    }
}

// ---------------------------------------------------------------- explicit
// SIMD (CPU): W bitpacks per work-item
template <int W> using u32v = std::uint32_t __attribute__((ext_vector_type(W)));
template <int W> using u64v = std::uint64_t __attribute__((ext_vector_type(W)));

template <int W>
[[gnu::always_inline]] inline void philox_round_v(u32v<W> &x0, u32v<W> &x1, u32v<W> &x2, u32v<W> &x3, std::uint32_t k0,
                                                  std::uint32_t k1) {
    const u64v<W> p0 = __builtin_convertvector(x0, u64v<W>) * static_cast<std::uint64_t>(kM0);
    const u64v<W> p1 = __builtin_convertvector(x2, u64v<W>) * static_cast<std::uint64_t>(kM1);
    const u32v<W> hi0 = __builtin_convertvector(p0 >> 32, u32v<W>);
    const u32v<W> lo0 = __builtin_convertvector(p0, u32v<W>);
    const u32v<W> hi1 = __builtin_convertvector(p1 >> 32, u32v<W>);
    const u32v<W> lo1 = __builtin_convertvector(p1, u32v<W>);
    const u32v<W> n0 = hi1 ^ x1 ^ k0;
    const u32v<W> n2 = hi0 ^ x3 ^ k1;
    x0 = n0;
    x1 = lo1;
    x2 = n2;
    x3 = lo0;
}

template <int W>
[[gnu::always_inline]] inline void sample_bitsliced_simd(std::uint64_t *out, std::uint32_t pdag_idx,
                                                         std::uint32_t event_idx, std::uint32_t sample0,
                                                         std::uint32_t iteration, std::uint64_t T) {
    if (T == 0) {
#pragma unroll
        for (int i = 0; i < W; ++i) {
            out[i] = 0;
        }
        return;
    }
    u32v<W> iota;
#pragma unroll
    for (int i = 0; i < W; ++i) {
        iota[i] = static_cast<std::uint32_t>(i);
    }
    const u32v<W> c2 = iota + (sample0 + 1u);
    u32v<W> lt0 = 0u, lt1 = 0u, eq0 = ~0u, eq1 = ~0u;
    std::uint64_t Tsh = T;
    std::uint32_t g = 0;
    while (true) {
        u32v<W> x0 = pdag_idx + 1u;
        u32v<W> x1 = event_idx + 1u;
        u32v<W> x2 = c2;
        u32v<W> x3 = ((iteration + 1u) << 6) + g;
#pragma unroll
        for (int i = 0; i < 10; ++i) {
            philox_round_v<W>(x0, x1, x2, x3, kK0 + static_cast<std::uint32_t>(i) * kW0,
                              kK1 + static_cast<std::uint32_t>(i) * kW1);
        }
        const std::uint32_t th = static_cast<std::uint32_t>(Tsh >> 32);
        const std::uint32_t m0 = static_cast<std::uint32_t>(static_cast<std::int32_t>(th) >> 31);
        const std::uint32_t m1 = static_cast<std::uint32_t>(static_cast<std::int32_t>(th << 1) >> 31);
        lt0 |= eq0 & ~x0 & m0;
        eq0 &= ~(x0 ^ m0);
        lt1 |= eq1 & ~x1 & m0;
        eq1 &= ~(x1 ^ m0);
        lt0 |= eq0 & ~x2 & m1;
        eq0 &= ~(x2 ^ m1);
        lt1 |= eq1 & ~x3 & m1;
        eq1 &= ~(x3 ^ m1);
        Tsh <<= 2;
        ++g;
        if (__builtin_reduce_or(eq0 | eq1) == 0u || Tsh == 0) {
            break;
        }
    }
#pragma unroll
    for (int i = 0; i < W; ++i) {
        out[i] = static_cast<std::uint64_t>(lt0[i]) | (static_cast<std::uint64_t>(lt1[i]) << 32);
    }
}

// Vector Philox round in 64-bit-lane form: a u32 vector of W lanes is viewed as W/2 u64 lanes.
// Even u32 lanes multiply as the low halves (pmuludq ignores the high 32 bits), odd lanes after a
// 32-bit right shift; hi/lo are reassembled in place with shifts and masks, so nothing crosses a
// 128-bit lane and no u32<->u64 conversions are needed.
template <int W>
[[gnu::always_inline]] inline void philox_round_v64(u32v<W> &x0, u32v<W> &x1, u32v<W> &x2, u32v<W> &x3,
                                                    std::uint32_t k0, std::uint32_t k1) {
    constexpr int H = W / 2;
    constexpr std::uint64_t lo_mask = 0xFFFFFFFFull;
    const u64v<H> a = __builtin_bit_cast(u64v<H>, x0);
    const u64v<H> c = __builtin_bit_cast(u64v<H>, x2);
    const u64v<H> pa_e = (a & lo_mask) * static_cast<std::uint64_t>(kM0);
    const u64v<H> pa_o = (a >> 32) * static_cast<std::uint64_t>(kM0);
    const u64v<H> pc_e = (c & lo_mask) * static_cast<std::uint64_t>(kM1);
    const u64v<H> pc_o = (c >> 32) * static_cast<std::uint64_t>(kM1);
    const u32v<W> hi0 = __builtin_bit_cast(u32v<W>, (pa_e >> 32) | (pa_o & ~lo_mask));
    const u32v<W> lo0 = __builtin_bit_cast(u32v<W>, (pa_e & lo_mask) | (pa_o << 32));
    const u32v<W> hi1 = __builtin_bit_cast(u32v<W>, (pc_e >> 32) | (pc_o & ~lo_mask));
    const u32v<W> lo1 = __builtin_bit_cast(u32v<W>, (pc_e & lo_mask) | (pc_o << 32));
    const u32v<W> n0 = hi1 ^ x1 ^ k0;
    const u32v<W> n2 = hi0 ^ x3 ^ k1;
    x0 = n0;
    x1 = lo1;
    x2 = n2;
    x3 = lo0;
}

template <int W>
[[gnu::always_inline]] inline void sample_bitsliced_simd64(std::uint64_t *out, std::uint32_t pdag_idx,
                                                           std::uint32_t event_idx, std::uint32_t sample0,
                                                           std::uint32_t iteration, std::uint64_t T) {
    if (T == 0) {
#pragma unroll
        for (int i = 0; i < W; ++i) {
            out[i] = 0;
        }
        return;
    }
    u32v<W> iota;
#pragma unroll
    for (int i = 0; i < W; ++i) {
        iota[i] = static_cast<std::uint32_t>(i);
    }
    const u32v<W> c2 = iota + (sample0 + 1u);
    u32v<W> lt0 = 0u, lt1 = 0u, eq0 = ~0u, eq1 = ~0u;
    std::uint64_t Tsh = T;
    std::uint32_t g = 0;
    while (true) {
        u32v<W> x0 = pdag_idx + 1u;
        u32v<W> x1 = event_idx + 1u;
        u32v<W> x2 = c2;
        u32v<W> x3 = ((iteration + 1u) << 6) + g;
#pragma unroll
        for (int i = 0; i < 10; ++i) {
            philox_round_v64<W>(x0, x1, x2, x3, kK0 + static_cast<std::uint32_t>(i) * kW0,
                                kK1 + static_cast<std::uint32_t>(i) * kW1);
        }
        const std::uint32_t th = static_cast<std::uint32_t>(Tsh >> 32);
        const std::uint32_t m0 = static_cast<std::uint32_t>(static_cast<std::int32_t>(th) >> 31);
        const std::uint32_t m1 = static_cast<std::uint32_t>(static_cast<std::int32_t>(th << 1) >> 31);
        lt0 |= eq0 & ~x0 & m0;
        eq0 &= ~(x0 ^ m0);
        lt1 |= eq1 & ~x1 & m0;
        eq1 &= ~(x1 ^ m0);
        lt0 |= eq0 & ~x2 & m1;
        eq0 &= ~(x2 ^ m1);
        lt1 |= eq1 & ~x3 & m1;
        eq1 &= ~(x3 ^ m1);
        Tsh <<= 2;
        ++g;
        if (__builtin_reduce_or(eq0 | eq1) == 0u || Tsh == 0) {
            break;
        }
    }
#pragma unroll
    for (int i = 0; i < W; ++i) {
        out[i] = static_cast<std::uint64_t>(lt0[i]) | (static_cast<std::uint64_t>(lt1[i]) << 32);
    }
}

// Persistent SIMD (CPU): each of the W vector lanes walks its own run of bitpacks
// (sample = sample0 + lane + W*i, i < count), one Philox generation per loop iteration. A lane whose
// bitpack is decided stores it and restarts on its next bitpack, so the vector pays for the mean
// decision depth instead of the maximum over W*64 trials.
template <int W>
[[gnu::always_inline]] inline void sample_bitsliced_simd64_persistent(std::uint64_t *row, std::uint32_t pdag_idx,
                                                                      std::uint32_t event_idx, std::uint32_t sample0,
                                                                      std::uint32_t count, std::uint32_t iteration,
                                                                      std::uint64_t T) {
    if (T == 0) {
        for (std::uint32_t i = 0; i < count * W; ++i) {
            row[sample0 + i] = 0;
        }
        return;
    }
    // Generations needed at most: ceil((64 - ctz(T)) / 2).
    const std::uint32_t gmax = (64u - static_cast<std::uint32_t>(__builtin_ctzll(T)) + 1u) / 2u;
    const std::uint32_t t_hi = static_cast<std::uint32_t>(T >> 32);
    const std::uint32_t t_lo = static_cast<std::uint32_t>(T);
    using V = u32v<W>;
    using SV = std::int32_t __attribute__((ext_vector_type(W)));
    V lane;
#pragma unroll
    for (int i = 0; i < W; ++i) {
        lane[i] = static_cast<std::uint32_t>(i);
    }
    V idx = 0u;                // bitpack ordinal within this lane's run
    V sample = lane + sample0; // current sample index
    V g = 0u;                  // generation within the current bitpack
    V lt0 = 0u, lt1 = 0u, eq0 = ~0u, eq1 = ~0u;
    const std::uint32_t x3base = (iteration + 1u) << 6;
    while (true) {
        const SV live = idx < count; // -1 where the lane still has work
        if (__builtin_reduce_or(live) == 0) {
            break;
        }
        V x0 = pdag_idx + 1u;
        V x1 = event_idx + 1u;
        V x2 = sample + 1u;
        V x3 = g + x3base;
#pragma unroll
        for (int i = 0; i < 10; ++i) {
            philox_round_v64<W>(x0, x1, x2, x3, kK0 + static_cast<std::uint32_t>(i) * kW0,
                                kK1 + static_cast<std::uint32_t>(i) * kW1);
        }
        // T bits for steps 2g and 2g+1 (g < 16: high word, else low word).
        const V k0 = g * 2u;
        const V hiw = __builtin_bit_cast(V, k0 < 32u);
        const V sh0 = (31u - (k0 & 31u));
        const V word = (hiw & t_hi) | (~hiw & t_lo);
        const V m0 = __builtin_bit_cast(V, -__builtin_bit_cast(SV, (word >> sh0) & 1u));
        const V m1 = __builtin_bit_cast(V, -__builtin_bit_cast(SV, (word >> (sh0 - 1u)) & 1u));
        lt0 |= eq0 & ~x0 & m0;
        eq0 &= ~(x0 ^ m0);
        lt1 |= eq1 & ~x1 & m0;
        eq1 &= ~(x1 ^ m0);
        lt0 |= eq0 & ~x2 & m1;
        eq0 &= ~(x2 ^ m1);
        lt1 |= eq1 & ~x3 & m1;
        eq1 &= ~(x3 ^ m1);
        g += 1u;
        const SV done = (((eq0 | eq1) == 0u) | (g >= gmax)) & live;
        if (__builtin_reduce_or(done) != 0) {
#pragma unroll
            for (int i = 0; i < W; ++i) {
                if (done[i]) {
                    row[sample[i]] = static_cast<std::uint64_t>(lt0[i]) | (static_cast<std::uint64_t>(lt1[i]) << 32);
                }
            }
            const V d = __builtin_bit_cast(V, done);
            idx += d & 1u;
            sample += d & static_cast<std::uint32_t>(W);
            g &= ~d;
            lt0 &= ~d;
            lt1 &= ~d;
            eq0 |= d;
            eq1 |= d;
        }
    }
}

} // namespace lab
