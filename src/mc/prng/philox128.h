#pragma once
#include <cstdint>

#include "mc/prng/state128.h"

/// Philox4x32-10 (Salmon, Moraes, Dror and Shaw, "Parallel random numbers: as easy as 1, 2, 3", SC'11),
/// in a scalar form for GPUs and a lane-wise vector form for host SIMD. Both compute the same function.
namespace scram::mc::prng::philox {

inline constexpr std::uint32_t kMultiplier0 = 0xD2511F53u;
inline constexpr std::uint32_t kMultiplier1 = 0xCD9E8D57u;
inline constexpr std::uint32_t kWeyl0 = 0x9E3779B9u;
inline constexpr std::uint32_t kWeyl1 = 0xBB67AE85u;
inline constexpr int kRounds = 10;

struct key64 {
    std::uint32_t k0;
    std::uint32_t k1;
};

/// The engine's fixed key; a user seed is #65.
inline constexpr key64 kEngineKey{382307844u, 293830103u};

[[gnu::always_inline]] inline void round(state128 &c, const std::uint32_t k0, const std::uint32_t k1) {
    std::uint32_t hi0;
    std::uint32_t hi1;
    const std::uint32_t lo0 = prng::mulhilo(kMultiplier0, c.x[0], &hi0);
    const std::uint32_t lo1 = prng::mulhilo(kMultiplier1, c.x[2], &hi1);
    c.x[0] = hi1 ^ c.x[1] ^ k0;
    c.x[1] = lo1;
    c.x[2] = hi0 ^ c.x[3] ^ k1;
    c.x[3] = lo0;
}

/// Philox4x32-10 of counter `c` under `key`.
[[gnu::always_inline]] inline state128 generate(state128 c, const key64 key = kEngineKey) {
#pragma unroll
    //TODO:: understand why i is declared as 'int', then static-cast to std::uint32_t anyway. Why not just use std::uint32_t?
    for (int i = 0; i < kRounds; ++i) {
        round(c, key.k0 + static_cast<std::uint32_t>(i) * kWeyl0, key.k1 + static_cast<std::uint32_t>(i) * kWeyl1);
    }
    return c;
}

template <int W> using u32xN = std::uint32_t __attribute__((ext_vector_type(W)));
template <int W> using u64xN = std::uint64_t __attribute__((ext_vector_type(W)));

/// One round on W counters held lane-wise: lane l of x[i] is word i of counter l.
///
/// The 32x32 -> 64-bit products are formed on W/2 64-bit lanes, even 32-bit lanes in place (the multiply
/// only reads the low half) and odd lanes after a 32-bit shift, and the halves are recombined in place.
/// x86 lowers this to pmuludq, shifts and blends without crossing 128-bit lanes; widening the u32 vectors
/// to u64 instead costs lane-crossing shuffles that bound the loop on one port.
template <int W>
[[gnu::always_inline]] inline void round_lanes(u32xN<W> (&x)[4], const std::uint32_t k0, const std::uint32_t k1) {
    static_assert(W % 2 == 0, "lanes pair up into 64-bit lanes");
    using u64v = u64xN<W / 2>;
    constexpr std::uint64_t low = 0xFFFFFFFFull;
    const u64v a = __builtin_bit_cast(u64v, x[0]);
    const u64v c = __builtin_bit_cast(u64v, x[2]);
    const u64v a_even = (a & low) * static_cast<std::uint64_t>(kMultiplier0);
    const u64v a_odd = (a >> 32) * static_cast<std::uint64_t>(kMultiplier0);
    const u64v c_even = (c & low) * static_cast<std::uint64_t>(kMultiplier1);
    const u64v c_odd = (c >> 32) * static_cast<std::uint64_t>(kMultiplier1);
    const auto hi0 = __builtin_bit_cast(u32xN<W>, (a_even >> 32) | (a_odd & ~low));
    const auto lo0 = __builtin_bit_cast(u32xN<W>, (a_even & low) | (a_odd << 32));
    const auto hi1 = __builtin_bit_cast(u32xN<W>, (c_even >> 32) | (c_odd & ~low));
    const auto lo1 = __builtin_bit_cast(u32xN<W>, (c_even & low) | (c_odd << 32));
    x[0] = hi1 ^ x[1] ^ k0;
    x[1] = lo1;
    x[2] = hi0 ^ x[3] ^ k1;
    x[3] = lo0;
}

/// Philox4x32-10 of W lane-wise counters, in place.
template <int W> [[gnu::always_inline]] inline void generate_lanes(u32xN<W> (&x)[4], const key64 key = kEngineKey) {
#pragma unroll
    //TODO:: understand why i is declared as 'int', then static-cast to std::uint32_t anyway. Why not just use std::uint32_t?
    for (int i = 0; i < kRounds; ++i) {
        round_lanes<W>(x, key.k0 + static_cast<std::uint32_t>(i) * kWeyl0,
                       key.k1 + static_cast<std::uint32_t>(i) * kWeyl1);
    }
}

} // namespace scram::mc::prng::philox
