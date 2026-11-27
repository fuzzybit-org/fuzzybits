#pragma once
#include <cstdint>

#include "mc/prng/philox128.h"
#include "mc/prng/state128.h"

/// Bit-sliced Bernoulli sampling of 64-trial bitpacks (#164) with exact 64-bit thresholds (#17).
///
/// **Definition.** A basic event with probability p has threshold T = floor(p * 2^64) (pcfb's
/// `probability_to_threshold`; p = 1 is `always_true` and samples no randomness). A bitpack has a Philox
/// counter base (see `counter`); generation g, 0 <= g < 32, yields (w0, w1, w2, w3) = Philox4x32-10 of
/// the base with g added to its last word, and the bitpack's 64-bit random words are
///
///     r[2g] = w0 | w1 << 32,    r[2g + 1] = w2 | w3 << 32.
///
/// Trial j (bit j of the bitpack) is 1 iff U_j < T, where U_j = sum_k bit_j(r[k]) * 2^(63 - k): the j-th
/// bits of the words, most significant first. Each U_j is uniform on [0, 2^64) and independent of the other
/// trials, so each trial is an exact Bernoulli(T / 2^64) draw.
///
/// **Evaluation.** The comparison runs most significant bit first over all 64 trials at once, one random
/// word per step. A trial is decided at the first step where its bit differs from T's (probability 1/2 per
/// step for any p), so a bitpack needs about 7 steps (4 Philox calls) on average where the 32-bit sampler
/// it replaces drew one 32-bit word per trial (16 calls). Trials still tied once T has no set bits left
/// have U_j >= T. Stopping as soon as every trial is decided does not change any bit, so every
/// implementation below, and any future one, produces the same bitpacks on every backend.
namespace scram::mc::prng::bernoulli {

/// Generations a bitpack can use: 64 words, two per Philox call. The counter's low 6 bits hold it.
inline constexpr std::uint32_t kMaxGenerations = 32;

/// Philox counter base of a bitpack: (pdag_idx + 1, event_idx + 1, sample_idx + 1, (iteration + 1) << 6).
/// The iteration field has 26 bits.
[[gnu::always_inline]] inline state128 counter(const std::uint32_t pdag_idx, const std::uint32_t event_idx,
                                               const std::uint32_t sample_idx, const std::uint32_t iteration) {
    return state128{{pdag_idx + 1u, event_idx + 1u, sample_idx + 1u, (iteration + 1u) << 6}};
}

/// Comparison state of a bitpack's 64 trials in 32-bit halves (trials 0-31 and 32-63): `lt` marks trials
/// decided below T, `eq` trials whose bits so far equal T's.
struct comparison {
    std::uint32_t lt0 = 0;
    std::uint32_t lt1 = 0;
    std::uint32_t eq0 = ~0u;
    std::uint32_t eq1 = ~0u;

    [[nodiscard]] bool decided() const { return (eq0 | eq1) == 0u; }
    [[nodiscard]] std::uint64_t bits() const {
        return static_cast<std::uint64_t>(lt0) | (static_cast<std::uint64_t>(lt1) << 32);
    }
};

/// One step: word (r0, r1) against T bit `t`, given as a mask of all ones or all zeros. A tied trial with a
/// 0 where T has a 1 drops below T; a trial whose bit differs from T's leaves the tie.
template <typename V> [[gnu::always_inline]] inline void step(V &lt, V &eq, const V &r, const std::uint32_t t) {
    lt |= eq & ~r & t;
    eq &= ~(r ^ t);
}

/// One generation: words r[2g] and r[2g+1] against the top two bits of `rest`, which holds T shifted left
/// by 2g (the bits not compared yet), then `rest` advances by two bits.
[[gnu::always_inline]] inline void compare(comparison &s, const state128 &w, std::uint64_t &rest) {
    const auto top = static_cast<std::uint32_t>(rest >> 32);
    const auto t0 = static_cast<std::uint32_t>(static_cast<std::int32_t>(top) >> 31);
    const auto t1 = static_cast<std::uint32_t>(static_cast<std::int32_t>(top << 1) >> 31);
    step(s.lt0, s.eq0, w.x[0], t0);
    step(s.lt1, s.eq1, w.x[1], t0);
    step(s.lt0, s.eq0, w.x[2], t1);
    step(s.lt1, s.eq1, w.x[3], t1);
    rest <<= 2;
}

/// The bitpack for threshold T, with generation g's words given by `words(g)`: the evaluation every form
/// below follows, separated from Philox so tests can supply the words.
template <typename Words>
[[gnu::always_inline]] inline std::uint64_t evaluate(const Words &words, const std::uint64_t threshold) {
    if (threshold == 0) {
        return 0;
    }
    comparison s;
    std::uint64_t rest = threshold;
    for (std::uint32_t g = 0; g < kMaxGenerations; ++g) {
        compare(s, words(g), rest);
        if (s.decided() || rest == 0) {
            break;
        }
    }
    return s.bits();
}

/// The bitpack at counter `base` for threshold T. The reference for every backend.
[[gnu::always_inline]] inline std::uint64_t sample(const state128 &base, const std::uint64_t threshold) {
    return evaluate(
        [&base](const std::uint32_t g) {
            state128 c = base;
            c.x[3] += g;
            return philox::generate(c);
        },
        threshold);
}

/// GPU form: the bitpacks first, first + stride, ... below `end` of one event, one at a time. A decided
/// bitpack is stored and the next one starts in the same loop iteration, so a sub-group pays for the mean
/// number of generations per bitpack rather than the maximum over its work-items. Requires T > 0.
[[gnu::always_inline]] inline void sample_strided(std::uint64_t *out, const std::uint32_t first,
                                                  const std::uint32_t stride, const std::uint32_t end,
                                                  const std::uint32_t pdag_idx, const std::uint32_t event_idx,
                                                  const std::uint32_t iteration, const std::uint64_t threshold) {
    std::uint32_t sample_idx = first;
    std::uint32_t g = 0;
    comparison s;
    std::uint64_t rest = threshold;
    while (sample_idx < end) {
        state128 c = counter(pdag_idx, event_idx, sample_idx, iteration);
        c.x[3] += g;
        compare(s, philox::generate(c), rest);
        ++g;
        if (s.decided() || rest == 0) {
            out[sample_idx] = s.bits();
            sample_idx += stride;
            g = 0;
            s = comparison{};
            rest = threshold;
        }
    }
}

/// CPU form: bitpacks sample0 .. sample0 + W - 1 of one event in W vector lanes, until all are decided;
/// the first `valid` are stored to out[0 .. valid). The host JIT vectorises kernels across work-items only
/// when their body has no loop of data-dependent length, so this loop carries its own vector width.
/// Requires T > 0.
template <int W>
[[gnu::always_inline]] inline void
sample_lanes(std::uint64_t *out, const std::uint32_t valid, const std::uint32_t sample0, const std::uint32_t pdag_idx,
             const std::uint32_t event_idx, const std::uint32_t iteration, const std::uint64_t threshold) {
    using V = philox::u32xN<W>;
    V lane;
    //TODO:: understand why i is declared as 'int', then static-cast to std::uint32_t anyway. Why not just use std::uint32_t?
#pragma unroll
    for (int i = 0; i < W; ++i) {
        lane[i] = static_cast<std::uint32_t>(i);
    }
    const state128 base = counter(pdag_idx, event_idx, sample0, iteration);
    const V sample_word = lane + base.x[2];
    V lt0 = 0u;
    V lt1 = 0u;
    V eq0 = ~0u;
    V eq1 = ~0u;
    std::uint64_t rest = threshold;
    for (std::uint32_t g = 0; g < kMaxGenerations; ++g) {
        V w[4] = {V(base.x[0]), V(base.x[1]), sample_word, V(base.x[3] + g)};
        philox::generate_lanes<W>(w);
        const auto top = static_cast<std::uint32_t>(rest >> 32);
        const auto t0 = static_cast<std::uint32_t>(static_cast<std::int32_t>(top) >> 31);
        const auto t1 = static_cast<std::uint32_t>(static_cast<std::int32_t>(top << 1) >> 31);
        step(lt0, eq0, w[0], t0);
        step(lt1, eq1, w[1], t0);
        step(lt0, eq0, w[2], t1);
        step(lt1, eq1, w[3], t1);
        rest <<= 2;
        //TODO:: inspect where `__builtin_reduce_or` is
        //TODO:: also check if its worth it to just do all the extra work anyway, and clear the extra work after
        // the loop. i.e. measure the cost vs accuracy of having the conditional inside the loop
        if (__builtin_reduce_or(eq0 | eq1) == 0u || rest == 0) {
            break;
        }
    }
#pragma unroll
    for (int i = 0; i < W; ++i) {
        //TODO:: check why i, valid are being checked inside a loop when valid is a constant to this function.
        if (static_cast<std::uint32_t>(i) < valid) {
            out[i] = static_cast<std::uint64_t>(lt0[i]) | (static_cast<std::uint64_t>(lt1[i]) << 32);
        }
    }
}

} // namespace scram::mc::prng::bernoulli
