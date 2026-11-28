/**
 * @file lockstep.h
 * @brief A vector variant's form: W consecutive bitpacks in W lanes, generation by generation, until every lane's
 *        bitpack is decided
 */

#pragma once

#include "mc/prng/bernoulli/pass_stream.h"
#include "mc/prng/bernoulli/sample.h"
#include "mc/prng/bernoulli/simd.h"
#include "mc/prng/prng.h"

#include <cstdint>

namespace fuzzybits::mc::prng::bernoulli {

/// Vector form: bitpacks [first, first + count) of `stream` into out[first .. first + count), W consecutive bitpacks at
/// a time in W lanes, each group until all of its trials are decided. The host JIT vectorizes kernels across
/// work-items only when their body has no loop of data-dependent length, so this loop carries its own vector width. A
/// group runs until its slowest bitpack is decided, 1.51 times the generations its bitpacks need at p = 0.01 with
/// W = 16; sample_chunked does not wait as long. Requires T > 0.
template <int W>
[[gnu::always_inline]] inline void sample_lockstep(std::uint64_t *out, const std::uint32_t first, const std::uint32_t count, const pass_stream &stream,
                                                   const std::uint64_t threshold) {
    constexpr auto kW = static_cast<std::uint32_t>(W);
    lanes<W> lane;
    lane_indices<W>(lane);
    const std::uint32_t end = first + count;
    for (std::uint32_t group = first; group < end; group += kW) {
        lanes<W> word2;
        lanes<W> word3;
        stream.word23_of<W>(word2, word3, lane + group);
        comparison<variant::vector, W> s;
        (void)run_lockstep<W>(s, word2, word3, stream, threshold, 0, kMaxGenerations);
        // Fewer than W only in the last group: the lanes past `end` are not this call's bitpacks.
        store_lanes<W>(out + group, s, end - group < kW ? end - group : kW);
    }
}

} // namespace fuzzybits::mc::prng::bernoulli
