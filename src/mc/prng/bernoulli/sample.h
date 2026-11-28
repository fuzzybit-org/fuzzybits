/**
 * @file sample.h
 * @brief The bit-sliced Bernoulli sampler's definition: the comparison every form runs, generation by generation, and
 *        the reference, which samples one bitpack at a time
 *
 * @details **Definition.** A basic event with probability p has threshold T = floor(p * 2^64) (pcfb's
 * `probability_to_threshold`; p = 1 is event::bernoulli_kind::always and samples no randomness). Bitpack b of a node
 * has the Philox counter of prng/stream.h's `counter` (its node, the run's device, b) under the run's key; generation
 * g, 0 <= g < 32, yields (w0, w1, w2, w3) = Philox4x32-10 of that counter with g added to its last word, and the
 * bitpack's 64-bit random words are
 *
 *     r[2g] = w0 | w1 << 32,    r[2g + 1] = w2 | w3 << 32.
 *
 * Trial j (bit j of the bitpack) is 1 iff U_j < T, where U_j = sum_k bit_j(r[k]) * 2^(63 - k): the j-th bits of the
 * words, most significant first. Each U_j is uniform on [0, 2^64) and independent of the other trials, so each trial
 * is an exact Bernoulli(T / 2^64) draw.
 *
 * **Evaluation.** The comparison runs most significant bit first over all 64 trials at once, one random word per step.
 * A trial is decided at the first step where its bit differs from T's (probability 1/2 per step for any p), so a
 * bitpack needs about 7 steps (4 Philox calls) on average where the 32-bit sampler it replaces drew one 32-bit word per
 * trial (16 calls). Trials still tied once T has no set bits left have U_j >= T. Stopping as soon as every trial is
 * decided does not change any bit, so every form (bernoulli.h), and any future one, writes the same bitpacks on every
 * backend.
 *
 * One generation's comparison is written once over Philox's variant (compare, compare_generation): on one bitpack in
 * plain words, or on W bitpacks lane-wise. The forms differ only in which bitpacks they compare at which generation.
 */

#pragma once

#include "mc/prng/bernoulli/pass_stream.h"
#include "mc/prng/philox128.h"
#include "mc/prng/prng.h"
#include "mc/prng/stream.h"

#include <cstdint>

namespace fuzzybits::mc::prng::bernoulli {

/// Generations a bitpack can use: 64 words, two per Philox call. Counter word 3's low bits hold the generation.
inline constexpr std::uint32_t kMaxGenerations = 32;
static_assert(kMaxGenerations <= (1u << kGenerationBits), "a generation fits counter word 3's low bits");

/// Comparison state of a bitpack's 64 trials in 32-bit halves (trials 0-31 and 32-63): `lt` marks trials decided below
/// T, `eq` trials whose bits so far equal T's. Of one bitpack in plain words (scalar variant), or of W bitpacks
/// lane-wise, lane l holding bitpack l (vector variant).
template <variant V, int W = 1> struct comparison {
    word<V, W> lt0 = word<V, W>(0u);
    word<V, W> lt1 = word<V, W>(0u);
    word<V, W> eq0 = word<V, W>(~0u);
    word<V, W> eq1 = word<V, W>(~0u);

    /// Whether every trial is decided: of the bitpack, or of every lane's bitpack.
    [[nodiscard]] bool decided() const {
        if constexpr (V == variant::scalar) {
            return (eq0 | eq1) == 0u;
        } else {
            return __builtin_reduce_or(eq0 | eq1) == 0u; // an OR across the lanes
        }
    }

    /// The bitpack: bit j is 1 iff trial j was decided below T.
    [[nodiscard]] std::uint64_t bits() const
        requires(V == variant::scalar)
    {
        return static_cast<std::uint64_t>(lt0) | (static_cast<std::uint64_t>(lt1) << 32);
    }

    /// Lane `l`'s bitpack.
    [[nodiscard]] std::uint64_t bits(const std::uint32_t l) const
        requires(V == variant::vector)
    {
        return static_cast<std::uint64_t>(lt0[l]) | (static_cast<std::uint64_t>(lt1[l]) << 32);
    }
};

/// One step: words `r` against T bit `t`, given as a mask of all ones or all zeros. A tied trial with a 0 where T has
/// a 1 drops below T; a trial whose bit differs from T's leaves the tie.
template <typename Word> [[gnu::always_inline]] inline void step(Word &lt, Word &eq, const Word &r, const std::uint32_t t) {
    lt |= eq & ~r & t;
    eq &= ~(r ^ t);
}

/// One generation g: the bitpacks' words r[2g] = (w.x[0], w.x[1]) and r[2g + 1] = (w.x[2], w.x[3]) against the top
/// two bits of `rest`, which holds T shifted left by 2g (the bits not compared yet); then `rest` advances by two bits.
/// Lanes are all at generation g, so T's bits are the same for every lane.
template <variant V, int W> [[gnu::always_inline]] inline void compare(comparison<V, W> &s, const state<V, W> &w, std::uint64_t &rest) {
    const auto top = static_cast<std::uint32_t>(rest >> 32);
    const auto t0 = static_cast<std::uint32_t>(static_cast<std::int32_t>(top) >> 31);
    const auto t1 = static_cast<std::uint32_t>(static_cast<std::int32_t>(top << 1) >> 31);
    step(s.lt0, s.eq0, w.x[0], t0);
    step(s.lt1, s.eq1, w.x[1], t0);
    step(s.lt0, s.eq0, w.x[2], t1);
    step(s.lt1, s.eq1, w.x[3], t1);
    rest <<= 2;
}

/// Generation g of the bitpacks whose counter words 2 and 3 at generation 0 are `word2` and `word3`: Philox4x32-10 of
/// their counters (node, device, word2, word3 + g) under the pass's round keys, compared with T (compare).
template <variant V, int W>
[[gnu::always_inline]] inline void compare_generation(comparison<V, W> &s, const word<V, W> &word2, const word<V, W> &word3, const std::uint32_t g,
                                                      const pass_stream &stream, std::uint64_t &rest) {
    state<V, W> w{{word<V, W>(stream.node()), word<V, W>(stream.device()), word2, word3 + g}};
    philox::generate(w, stream.keys());
    compare(s, w, rest);
}

/// The bitpack for threshold T, with generation g's words given by `words(g)`: the evaluation every form follows,
/// separated from Philox so tests can supply the words.
template <typename Words> [[gnu::always_inline]] inline std::uint64_t evaluate(const Words &words, const std::uint64_t threshold) {
    if (threshold == 0) {
        return 0;
    }
    comparison<variant::scalar> s;
    std::uint64_t rest = threshold;
    for (std::uint32_t g = 0; g < kMaxGenerations; ++g) {
        compare(s, words(g), rest);
        if (s.decided() || rest == 0) {
            break;
        }
    }
    return s.bits();
}

/// The bitpack at counter `base` under the round keys `keys` for threshold T.
[[gnu::always_inline]] inline std::uint64_t sample(const state128 &base, const philox::key_schedule &keys, const std::uint64_t threshold) {
    return evaluate(
        [&base, &keys](const std::uint32_t g) {
            state128 c = base;
            c.x[3] += g;
            philox::generate(c, keys);
            return c;
        },
        threshold);
}

/// The bitpack at counter `base` under `key` for threshold T.
[[gnu::always_inline]] inline std::uint64_t sample(const state128 &base, const philox::key64 key, const std::uint64_t threshold) {
    return sample(base, philox::key_schedule::of(key), threshold);
}

/// Bitpack s of `stream` for threshold T: the reference for every form and backend.
[[gnu::always_inline]] inline std::uint64_t sample(const pass_stream &stream, const std::uint32_t s, const std::uint64_t threshold) {
    return sample(stream.counter_of(s), stream.keys(), threshold);
}

} // namespace fuzzybits::mc::prng::bernoulli
