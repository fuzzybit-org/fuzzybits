/**
 * @file strided.h
 * @brief The scalar variant's form, for GPUs: a work-item samples bitpacks first, first + stride, ... one at a time
 */

#pragma once

#include "mc/prng/bernoulli/pass_stream.h"
#include "mc/prng/bernoulli/sample.h"
#include "mc/prng/prng.h"

#include <cstdint>

namespace fuzzybits::mc::prng::bernoulli {

namespace detail {

/// sample_strided's loop. With `word3_varies` false, every bitpack of the pass has the same counter word 3, and the loop
/// never recomputes it.
template <bool word3_varies>
[[gnu::always_inline]] inline void sample_strided_loop(std::uint64_t *out, const std::uint32_t first, const std::uint32_t stride, const std::uint32_t end,
                                                       const pass_stream &stream, const std::uint64_t threshold) {
    std::uint32_t sample_idx = first;
    std::uint32_t word3 = stream.word3_of(sample_idx);
    std::uint32_t g = 0;
    comparison<variant::scalar> s;
    std::uint64_t rest = threshold;
    while (sample_idx < end) {
        compare_generation(s, stream.word2_of(sample_idx), word3, g, stream, rest);
        ++g;
        if (s.decided() || rest == 0) {
            out[sample_idx] = s.bits();
            sample_idx += stride;
            if constexpr (word3_varies) {
                word3 = stream.word3_of(sample_idx);
            }
            g = 0;
            s = comparison<variant::scalar>{};
            rest = threshold;
        }
    }
}

} // namespace detail

/// Scalar form: the bitpacks first, first + stride, ... below `end` of `stream`, one at a time, into out[first],
/// out[first + stride], ... A decided bitpack is stored and the next one starts in the same loop iteration, so a
/// sub-group pays for the mean number of generations per bitpack rather than the maximum over its work-items. Requires
/// T > 0.
///
/// The loop carries only 32-bit scalars and builds each generation's counter from them, so that starting the next
/// bitpack stays a few instructions, which compilers predicate inside the loop body. Two things would make it a
/// divergent branch that a warp takes almost every generation (measured: the sampler 1.07 and 1.17 times slower on an
/// RTX 5060 Ti):
/// - recomputing the high word of the bitpack index, with its carry, on every restart. Only a pass whose indices
///   cross a multiple of 2^32 needs that; the test is the same for every work-item of a launch, so it never diverges;
/// - a state128 passed or returned by value on the kernel path, such as a counter carried across iterations: the host
///   calling convention packs its words in pairs into 64-bit values, which AdaptiveCpp's generic JIT receives as such.
[[gnu::always_inline]] inline void sample_strided(std::uint64_t *out, const std::uint32_t first, const std::uint32_t stride, const std::uint32_t end,
                                                  const pass_stream &stream, const std::uint64_t threshold) {
    if (stream.word3_varies(end)) {
        detail::sample_strided_loop<true>(out, first, stride, end, stream, threshold);
    } else {
        detail::sample_strided_loop<false>(out, first, stride, end, stream, threshold);
    }
}

} // namespace fuzzybits::mc::prng::bernoulli
