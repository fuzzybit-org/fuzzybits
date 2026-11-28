/**
 * @file pass_stream.h
 * @brief One node's bitpacks in one pass: their Philox key and counters, one bitpack at a time (scalar variant) or
 *        lane by lane (vector variant)
 */

#pragma once

#include "mc/prng/philox128.h"
#include "mc/prng/prng.h"
#include "mc/prng/stream.h"

#include <cstdint>

namespace fuzzybits::mc::prng::bernoulli {

/// One node's bitpacks in one pass: the run's round keys, and the coordinates of the pass's bitpack 0. Bitpack s of
/// the pass is the node's bitpack first + s of the run. Built from the run's stream only, so its key is always the
/// run's.
class pass_stream {
  public:
    /// Node `node`'s bitpacks in the pass whose bitpack 0 is the node's run-wide bitpack `first`.
    constexpr pass_stream(const expanded_stream &run, const std::uint32_t node, const std::uint64_t first) noexcept
        : keys_(run.keys), node_(node), device_(run.device), first_lo_(static_cast<std::uint32_t>(first)), first_hi_(static_cast<std::uint32_t>(first >> 32)) {}
    constexpr pass_stream(const stream &run, const std::uint32_t node, const std::uint64_t first) noexcept
        : pass_stream(expanded_stream::of(run), node, first) {}

    /// Counter word 2 of the pass's bitpack s: the low 32 bits of its run-wide index first + s.
    [[nodiscard]] constexpr std::uint32_t word2_of(const std::uint32_t s) const noexcept { return first_lo_ + s; }

    /// Counter word 3 of the pass's bitpack s at generation 0: the bits of first + s above the low 32, shifted above
    /// the generation bits. The low word's carry is added in 32 bits.
    [[nodiscard]] constexpr std::uint32_t word3_of(const std::uint32_t s) const noexcept {
        return (first_hi_ + (word2_of(s) < first_lo_ ? 1u : 0u)) << kGenerationBits;
    }

    /// Counter words 2 and 3 at generation 0 of the pass's bitpacks `s`, lane by lane: word2_of and word3_of of each
    /// lane. Out-parameters, as a vector passed or returned by value would change the host calling convention
    /// (-Wpsabi).
    template <int W> [[gnu::always_inline]] void word23_of(lanes<W> &word2, lanes<W> &word3, const lanes<W> &s) const noexcept {
        word2 = s + first_lo_;
        // A lane-wise comparison is -1 where true.
        word3 = (first_hi_ + (__builtin_convertvector(word2 < first_lo_, lanes<W>) & 1u)) << kGenerationBits;
    }

    /// The counter of the pass's bitpack s at generation 0: counter({node, device, first + s}).
    [[nodiscard]] constexpr state128 counter_of(const std::uint32_t s) const noexcept { return state128{{node_, device_, word2_of(s), word3_of(s)}}; }

    /// Whether word 3 differs among the pass's bitpacks [0, count): their run-wide indices cross a multiple of 2^32. A
    /// pass of B bitpacks starts at a multiple of B, so with B a power of two, as the sample shaper makes it, none does.
    [[nodiscard]] constexpr bool word3_varies(const std::uint32_t count) const noexcept { return count != 0 && count - 1 > ~first_lo_; }

    [[nodiscard]] constexpr const philox::key_schedule &keys() const noexcept { return keys_; }
    [[nodiscard]] constexpr std::uint32_t node() const noexcept { return node_; }
    [[nodiscard]] constexpr std::uint32_t device() const noexcept { return device_; }
    [[nodiscard]] constexpr std::uint64_t first() const noexcept { return (std::uint64_t{first_hi_} << 32) | first_lo_; }

  private:
    philox::key_schedule keys_;
    std::uint32_t node_;
    std::uint32_t device_;
    std::uint32_t first_lo_;
    std::uint32_t first_hi_;
};

} // namespace fuzzybits::mc::prng::bernoulli
