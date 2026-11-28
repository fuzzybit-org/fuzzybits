/**
 * @file stream.h
 * @brief Where a trial's random bits come from: the Philox key and counter of one bitpack of one node
 *
 * @details Philox4x32-10 maps a 64-bit key and a 128-bit counter to 128 random bits. The engine fills all 192 input
 * bits, each with a coordinate of its own:
 *
 *     key     = seed                                                (64 bits)
 *     counter = (node, device, b mod 2^32, (b >> 32) << 6 | g)       (32, 32, 32, 26 + 6 bits)
 *
 * `node` is the node's PDAG index, `device` the logical index of the device in the run, `b` the bitpack's index
 * among all of the node's bitpacks in the run (pass × bitpacks per pass + its index in the pass), and `g` the
 * generation (prng/bernoulli/sample.h). Nothing is hashed or overlapped, so distinct coordinates are distinct Philox inputs,
 * and a trial's bits are a function of (seed, device, node, b, g) only: not of the node's block or layer, the pass
 * size, the work-group geometry, the sampler form or the backend.
 */

#pragma once

#include "mc/prng/philox128.h"
#include "mc/prng/prng.h"

#include <cstdint>

namespace fuzzybits::mc::prng {

/// Bits of a generation in counter word 3.
inline constexpr std::uint32_t kGenerationBits = 6;

/// Bitpack indices have 58 bits: 32 in counter word 2, 26 above the generation in word 3.
inline constexpr std::uint64_t kMaxBitpacks = std::uint64_t{1} << (64 - kGenerationBits);

/// What selects a run's random numbers apart from the node and the trial: the seed, which is the Philox key, and
/// the logical index of the device in the run. The device index is a position in the run (0 for a one-device run),
/// never a hardware identity, so that a run reproduces on other machines.
struct stream {
    /// The seed of runs that name none: 372, the default seed of every entry point (scram::core::Settings,
    /// runtime::RunOptions, `fzb run --seed`).
    static constexpr std::uint64_t kDefaultSeed = 372;

    philox::key64 key{.k0 = static_cast<std::uint32_t>(kDefaultSeed), .k1 = static_cast<std::uint32_t>(kDefaultSeed >> 32)};
    std::uint32_t device = 0;

    /// The stream of `seed` on device `device` of the run.
    [[nodiscard]] static constexpr stream of(const std::uint64_t seed, const std::uint32_t device = 0) noexcept {
        return {.key = {.k0 = static_cast<std::uint32_t>(seed), .k1 = static_cast<std::uint32_t>(seed >> 32)}, .device = device};
    }
    [[nodiscard]] constexpr std::uint64_t seed() const noexcept { return (std::uint64_t{key.k1} << 32) | std::uint64_t{key.k0}; }

    friend constexpr bool operator==(const stream &a, const stream &b) noexcept { return a.key.k0 == b.key.k0 && a.key.k1 == b.key.k1 && a.device == b.device; }
};

/// A stream as kernels hold it: the seed's round keys, expanded on the host (philox::key_schedule), and the device.
/// 84 bytes, built once per run; per launch only the pass index changes.
struct expanded_stream {
    philox::key_schedule keys;
    std::uint32_t device;

    [[nodiscard]] static constexpr expanded_stream of(const stream &run) noexcept { return {.keys = philox::key_schedule::of(run.key), .device = run.device}; }
};

/// The coordinates of one bitpack of one node in a run, apart from the seed.
struct coordinates {
    /// The node's PDAG index.
    std::uint32_t node = 0;
    /// stream::device.
    std::uint32_t device = 0;
    /// The bitpack's index among all of the node's bitpacks in the run; below kMaxBitpacks.
    std::uint64_t bitpack = 0;
};

/// The Philox counter of a bitpack at generation 0. Generation g adds g to its last word.
[[gnu::always_inline]] constexpr state128 counter(const coordinates &c) noexcept {
    return state128{{c.node, c.device, static_cast<std::uint32_t>(c.bitpack), static_cast<std::uint32_t>(c.bitpack >> 32) << kGenerationBits}};
}

} // namespace fuzzybits::mc::prng
