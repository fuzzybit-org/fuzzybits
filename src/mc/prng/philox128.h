#pragma once

#include "mc/prng/prng.h"

#include <cstdint>

/// Philox4x32-10 (Salmon, Moraes, Dror and Shaw, "Parallel random numbers: as easy as 1, 2, 3", SC'11), written once
/// over prng::variant: on one counter in plain words (GPUs) or on W counters lane-wise in vectors (CPUs). The variants
/// compute the same function; only the round's products are formed differently (prng::mulhilo).
namespace fuzzybits::mc::prng::philox {

inline constexpr std::uint32_t kMultiplier0 = 0xD2511F53u;
inline constexpr std::uint32_t kMultiplier1 = 0xCD9E8D57u;
inline constexpr std::uint32_t kWeyl0 = 0x9E3779B9u;
inline constexpr std::uint32_t kWeyl1 = 0xBB67AE85u;
inline constexpr std::uint32_t kRounds = 10;

struct key64 {
    std::uint32_t k0;
    std::uint32_t k1;
};

/// The round keys of Philox4x32-10 under one key: round i XORs in the key plus i Weyl increments. Expanded once on the
/// host, so that a kernel takes every round key from its arguments instead of adding it per call or holding twenty
/// registers: under AdaptiveCpp's SSCP JIT the schedule is a specialized argument and its words become immediates;
/// in the AOT CUDA builds, and a JIT at adaptivity level 0, it is an ordinary argument in the constant bank, which
/// NVIDIA's XOR takes as an operand (kernel/sampler/sampler.h).
struct key_schedule {
    std::uint32_t k0[kRounds];
    std::uint32_t k1[kRounds];

    [[nodiscard]] static constexpr key_schedule of(const key64 key) noexcept {
        key_schedule s{};
        for (std::uint32_t i = 0; i < kRounds; ++i) {
            s.k0[i] = key.k0 + i * kWeyl0;
            s.k1[i] = key.k1 + i * kWeyl1;
        }
        return s;
    }

    friend constexpr bool operator==(const key_schedule &, const key_schedule &) = default;
};

/// One round on the counters in `c`: words 0 and 2 multiplied, and the halves of the products mixed with words 1 and 3
/// and the round keys.
template <variant V, int W> [[gnu::always_inline]] inline void round(state<V, W> &c, const std::uint32_t k0, const std::uint32_t k1) {
    word<V, W> hi0;
    word<V, W> lo0;
    word<V, W> hi1;
    word<V, W> lo1;
    mulhilo<V, W>(kMultiplier0, c.x[0], hi0, lo0);
    mulhilo<V, W>(kMultiplier1, c.x[2], hi1, lo1);
    c.x[0] = hi1 ^ c.x[1] ^ k0;
    c.x[1] = lo1;
    c.x[2] = hi0 ^ c.x[3] ^ k1;
    c.x[3] = lo0;
}

/// Philox4x32-10 of the counters in `c` under the round keys `keys`, in place: the counters go in, the random words
/// come out.
template <variant V, int W> [[gnu::always_inline]] inline void generate(state<V, W> &c, const key_schedule &keys) {
#pragma unroll
    for (std::uint32_t i = 0; i < kRounds; ++i) {
        round(c, keys.k0[i], keys.k1[i]);
    }
}

/// Philox4x32-10 of the counters in `c` under `key`, in place.
template <variant V, int W> [[gnu::always_inline]] inline void generate(state<V, W> &c, const key64 key) { generate(c, key_schedule::of(key)); }

} // namespace fuzzybits::mc::prng::philox
