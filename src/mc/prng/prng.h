/**
 * @file prng.h
 * @brief The vocabulary of the engine's counter-based generator: its two variants, the words each works on, and the
 *        Philox state
 *
 * @details The generator runs in one of two variants, which compute the same function on every counter:
 * - **scalar**: a work-item holds one counter, in plain 32-bit words. GPUs run work-items in hardware lanes, so the
 *   code is scalar and the lanes are implicit.
 * - **vector**: a work-item holds W counters lane-wise, in W-lane vectors: word i of counter l is lane l of vector i.
 *   A CPU JIT vectorizes across work-items only when the kernel body has no loop of data-dependent length, which the
 *   samplers have (prng/bernoulli/bernoulli.h), so their lanes are explicit.
 * Code that works on words is written once over the variant (philox::round, philox::generate, and the samplers'
 * bernoulli::compare_generation); the variants differ only in how a 32 x 32 -> 64-bit product is formed (mulhilo).
 */

#pragma once

#include <cstdint>

#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
#include <sycl/sycl.hpp>
#endif

namespace fuzzybits::mc::prng {

/// How a work-item holds its counters: one in plain words, or W lane-wise in vectors.
enum struct variant { scalar, vector };

/// W lanes of 32-bit and of 64-bit words.
template <int W> using u32xN = std::uint32_t __attribute__((ext_vector_type(W)));
template <int W> using u64xN = std::uint64_t __attribute__((ext_vector_type(W)));

namespace detail {

template <variant V, int W> struct word_of;
template <int W> struct word_of<variant::scalar, W> {
    static_assert(W == 1, "the scalar variant holds one counter, in plain words");
    using type = std::uint32_t;
};
template <int W> struct word_of<variant::vector, W> {
    static_assert(W >= 2 && W % 2 == 0, "vector lanes pair up into 64-bit lanes for the products (mulhilo)");
    using type = u32xN<W>;
};

} // namespace detail

/// The 32-bit word a variant works on: one std::uint32_t, or W lanes.
template <variant V, int W = 1> using word = typename detail::word_of<V, W>::type;

/// Philox4x32's state, four 32-bit words: a 128-bit counter going in, 128 random bits coming out. The scalar variant
/// holds one counter; the vector variant W, word i of counter l in x[i][l].
template <variant V, int W = 1> struct state {
    word<V, W> x[4];
};

/// One counter in plain words: what a GPU work-item holds, and the counter of a bitpack (prng/stream.h).
using state128 = state<variant::scalar>;

/// The vector variant's word, W lanes: what the samplers' vector forms hold their counters and comparisons in
/// (prng/bernoulli/simd.h).
template <int W> using lanes = word<variant::vector, W>;

/// The high and low halves of the 32 x 32 -> 64-bit products m · a, lane by lane. Out-parameters, because a vector
/// passed or returned by value would change the host calling convention (-Wpsabi). Every form reads `a` before it
/// writes, so `hi` or `lo` may be `a`.
///
/// - scalar: the best form depends on the GPU, so the generic (SSCP) build picks it when the kernel is JIT-compiled.
///   NVIDIA: the low half is a separate 32-bit multiply, and ptxas emits one IMAD.WIDE.U32 or an IMAD.HI/IMAD pair;
///   truncating the 64-bit product instead adds instructions per multiply (#16). AMD: one 64-bit product, a single
///   v_mad_u64_u32, against v_mul_hi_u32 plus v_mul_lo_u32.
/// - vector: the products are formed on W/2 64-bit lanes, even 32-bit lanes in place (the multiply reads only the low
///   half) and odd lanes after a 32-bit shift, and the halves are recombined in place. x86 lowers this to pmuludq,
///   shifts and blends without crossing 128-bit lanes; widening the vectors to 64-bit lanes instead costs lane-crossing
///   shuffles that bound the loop on one port.
template <variant V, int W> [[gnu::always_inline]] inline void mulhilo(const std::uint32_t m, const word<V, W> &a, word<V, W> &hi, word<V, W> &lo) {
    if constexpr (V == variant::scalar) {
        const std::uint32_t x = a;
#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
        __acpp_if_target_sscp(namespace jit = sycl::AdaptiveCpp_jit;
                              if (jit::reflection_query::compiler_backend::is_known() &&
                                  jit::reflect<jit::reflection_query::compiler_backend>() == jit::compiler_backend::amdgpu) {
                                  const std::uint64_t product = std::uint64_t{m} * x;
                                  hi = static_cast<std::uint32_t>(product >> 32);
                                  lo = static_cast<std::uint32_t>(product);
                                  return;
                              });
#endif
        hi = static_cast<std::uint32_t>((std::uint64_t{m} * x) >> 32);
        lo = m * x;
    } else {
        using pairs = u64xN<W / 2>;
        constexpr std::uint64_t low = 0xFFFFFFFFull;
        const pairs both = __builtin_bit_cast(pairs, a);
        const pairs even = (both & low) * std::uint64_t{m};
        const pairs odd = (both >> 32) * std::uint64_t{m};
        hi = __builtin_bit_cast(word<V, W>, (even >> 32) | (odd & ~low));
        lo = __builtin_bit_cast(word<V, W>, (even & low) | (odd << 32));
    }
}

} // namespace fuzzybits::mc::prng
