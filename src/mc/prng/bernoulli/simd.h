/**
 * @file simd.h
 * @brief The vector variant's building blocks: lane helpers, and the generation loop that runs W lanes in lockstep
 *
 * @details Both vector forms run their lanes with run_lockstep: the lockstep form until every lane is decided, the
 * chunked form in two phases. Vectors go by reference and results through out-parameters: a vector passed or returned
 * by value would change the host calling convention (-Wpsabi).
 */

#pragma once

#include "mc/prng/bernoulli/pass_stream.h"
#include "mc/prng/bernoulli/sample.h"
#include "mc/prng/prng.h"

#include <cstdint>

namespace fuzzybits::mc::prng::bernoulli {

/// The lane indices: lane[l] = l.
template <int W> [[gnu::always_inline]] inline void lane_indices(lanes<W> &lane) {
#pragma unroll
    for (std::uint32_t l = 0; l < static_cast<std::uint32_t>(W); ++l) {
        lane[l] = l;
    }
}

/// The lanes of `v` that are not zero, as the low W bits of an integer.
template <int W> [[gnu::always_inline]] inline std::uint32_t lane_mask(const lanes<W> &v) {
    using mask = bool __attribute__((ext_vector_type(W)));
    const mask m = __builtin_convertvector(v, mask);
    if constexpr (W <= 8) {
        return __builtin_bit_cast(std::uint8_t, m) & ((1u << W) - 1u);
    } else if constexpr (W == 16) {
        return __builtin_bit_cast(std::uint16_t, m);
    } else {
        static_assert(W == 32, "lanes come in vectors of at most 32");
        return __builtin_bit_cast(std::uint32_t, m);
    }
}

/// The bitpacks of lanes [0, valid) of `s` into out[0 .. valid), valid <= W.
template <int W> [[gnu::always_inline]] inline void store_lanes(std::uint64_t *out, const comparison<variant::vector, W> &s, const std::uint32_t valid) {
#pragma unroll
    for (std::uint32_t l = 0; l < static_cast<std::uint32_t>(W); ++l) {
        if (l < valid) {
            out[l] = s.bits(l);
        }
    }
}

/// W lanes in lockstep: generations g, g + 1, ... of the bitpacks whose counter words 2 and 3 at generation 0 are
/// `word2` and `word3`, with `rest` = T shifted left by 2g, until every trial of every lane is decided, T has no bits
/// left, or generation `stop`. Stopping changes no bit. Returns `rest` after the last generation run.
template <int W>
[[gnu::always_inline]] inline std::uint64_t run_lockstep(comparison<variant::vector, W> &s, const lanes<W> &word2, const lanes<W> &word3,
                                                         const pass_stream &stream, std::uint64_t rest, std::uint32_t g, const std::uint32_t stop) {
    for (; g < stop; ++g) {
        compare_generation(s, word2, word3, g, stream, rest);
        if (s.decided() || rest == 0) {
            break;
        }
    }
    return rest;
}

} // namespace fuzzybits::mc::prng::bernoulli
