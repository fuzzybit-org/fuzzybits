/**
 * @file chunked.h
 * @brief The vector variant's form for CPUs: a chunk of bitpacks in lockstep for a few generations, then the bitpacks
 *        still undecided, W at a time
 */

#pragma once

#include "mc/prng/bernoulli/pass_stream.h"
#include "mc/prng/bernoulli/sample.h"
#include "mc/prng/bernoulli/simd.h"
#include "mc/prng/prng.h"

#include <cstdint>

namespace fuzzybits::mc::prng::bernoulli {

/// Generations a group runs in lockstep before its undecided bitpacks are set aside (sample_chunked). Under a threshold
/// of 2k bits or more, a bitpack is decided by generation k with probability (1 - 4^-k)^64: 2 % by generation 2, 37 %
/// by 3, 78 % by 4. 3 and 5 were slower than 4 on an i7-10700 at p = 0.01.
inline constexpr std::uint32_t kChunkGenerations = 4;

/// Bitpacks sample_chunked takes at most: the size of its buffer of set-aside bitpacks.
inline constexpr std::uint32_t kMaxChunk = 1024;

/// Vector form: bitpacks [first, first + count) of `stream` into out[first .. first + count), count <= kMaxChunk, in W
/// vector lanes and two phases, each in lockstep (run_lockstep):
/// 1. groups of W consecutive bitpacks for at most kChunkGenerations generations; the bitpacks still undecided are set
///    aside with their comparison state;
/// 2. the set-aside bitpacks, W at a time, from generation kChunkGenerations until decided. All are at the same
///    generation, so the threshold bits are the same for every lane, as in phase 1.
/// A group in lockstep runs until its slowest bitpack is decided: 1.51 times the generations its bitpacks need at
/// p = 0.01 with W = 16 (sample_lockstep). Phase 1 runs 4 generations where its bitpacks need 3.62 on average, and
/// phase 2 waits only on the slowest of W set-aside bitpacks. On one core of an i7-10700 this was 1.13 to 1.20 times as
/// fast as sample_lockstep over the same bitpacks at p = 0.01 and p = 1e-6, and 1.02 at p = 0.5, where T has one bit
/// and every bitpack takes one generation. Requires T > 0.
template <int W>
[[gnu::always_inline]] inline void sample_chunked(std::uint64_t *out, const std::uint32_t first, const std::uint32_t count, const pass_stream &stream,
                                                  const std::uint64_t threshold) {
    using V = lanes<W>;
    constexpr auto kW = static_cast<std::uint32_t>(W);
    V lane;
    lane_indices<W>(lane);
    // The set-aside bitpacks, field by field, with one vector of decided padding past the last.
    alignas(64) std::uint32_t aside_s[kMaxChunk + kW];
    alignas(64) std::uint32_t aside_lt0[kMaxChunk + kW];
    alignas(64) std::uint32_t aside_lt1[kMaxChunk + kW];
    alignas(64) std::uint32_t aside_eq0[kMaxChunk + kW];
    alignas(64) std::uint32_t aside_eq1[kMaxChunk + kW];
    alignas(64) std::uint32_t copy0[kW];
    alignas(64) std::uint32_t copy1[kW];
    alignas(64) std::uint32_t copy2[kW];
    alignas(64) std::uint32_t copy3[kW];
    std::uint32_t aside = 0;
    const std::uint32_t end = first + count;
    for (std::uint32_t group = first; group < end; group += kW) {
        V word2;
        V word3;
        stream.word23_of<W>(word2, word3, lane + group);
        comparison<variant::vector, W> s;
        const std::uint64_t rest = run_lockstep<W>(s, word2, word3, stream, threshold, 0, kChunkGenerations);
        const std::uint32_t valid = end - group < kW ? end - group : kW;
        store_lanes<W>(out + group, s, valid);
        if (rest == 0) {
            continue; // T has no bits left: every trial is final
        }
        std::uint32_t open = lane_mask<W>(__builtin_convertvector((s.eq0 | s.eq1) != 0u, V) & __builtin_convertvector(lane < valid, V));
        if (open == 0) {
            continue;
        }
        __builtin_memcpy(copy0, &s.lt0, sizeof(V));
        __builtin_memcpy(copy1, &s.lt1, sizeof(V));
        __builtin_memcpy(copy2, &s.eq0, sizeof(V));
        __builtin_memcpy(copy3, &s.eq1, sizeof(V));
        for (; open != 0; open &= open - 1) {
            const auto l = static_cast<std::uint32_t>(__builtin_ctz(open));
            aside_s[aside] = group + l;
            aside_lt0[aside] = copy0[l];
            aside_lt1[aside] = copy1[l];
            aside_eq0[aside] = copy2[l];
            aside_eq1[aside] = copy3[l];
            ++aside;
        }
    }
    if (aside == 0) {
        return;
    }
    // Every set-aside bitpack ran exactly kChunkGenerations generations: run_lockstep stops sooner only when a group is
    // decided or T has no bits left.
    const std::uint64_t rest = threshold << (2 * kChunkGenerations);
    // Decided lanes pad the last vector: they change nothing and are not stored.
    for (std::uint32_t l = aside; l < aside + kW; ++l) {
        aside_s[l] = first;
        aside_lt0[l] = 0u;
        aside_lt1[l] = 0u;
        aside_eq0[l] = 0u;
        aside_eq1[l] = 0u;
    }
    for (std::uint32_t at = 0; at < aside; at += kW) {
        V s_of;
        comparison<variant::vector, W> s;
        __builtin_memcpy(&s_of, aside_s + at, sizeof(V));
        __builtin_memcpy(&s.lt0, aside_lt0 + at, sizeof(V));
        __builtin_memcpy(&s.lt1, aside_lt1 + at, sizeof(V));
        __builtin_memcpy(&s.eq0, aside_eq0 + at, sizeof(V));
        __builtin_memcpy(&s.eq1, aside_eq1 + at, sizeof(V));
        V word2;
        V word3;
        stream.word23_of<W>(word2, word3, s_of);
        (void)run_lockstep<W>(s, word2, word3, stream, rest, kChunkGenerations, kMaxGenerations);
        __builtin_memcpy(copy0, &s.lt0, sizeof(V));
        __builtin_memcpy(copy1, &s.lt1, sizeof(V));
        const std::uint32_t n = aside - at < kW ? aside - at : kW;
        for (std::uint32_t l = 0; l < n; ++l) {
            out[aside_s[at + l]] = static_cast<std::uint64_t>(copy0[l]) | (static_cast<std::uint64_t>(copy1[l]) << 32);
        }
    }
}

} // namespace fuzzybits::mc::prng::bernoulli
