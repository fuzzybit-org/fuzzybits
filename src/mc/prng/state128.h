#pragma once
#include <cstdint>

#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
#include <sycl/sycl.hpp>
#endif

namespace scram::mc::prng {
struct state128 {
    /// @brief Four 32-bit state values forming the complete 128-bit state
    std::uint32_t x[4];
};

/// High and low halves of the 32x32 -> 64-bit product a * b.
///
/// The best form depends on the GPU, so the generic (SSCP) build picks it when the kernel is JIT-compiled:
/// - NVIDIA: the low half is a separate 32-bit multiply. ptxas then emits one IMAD.WIDE.U32 or an
///   IMAD.HI/IMAD pair; truncating the 64-bit product instead adds instructions per multiply (#16).
/// - AMD: one 64-bit product, which is a single v_mad_u64_u32, against v_mul_hi_u32 plus v_mul_lo_u32.
/// Host SIMD code does not use this: it has the lane-wise form in philox128.h.
[[gnu::always_inline]] inline std::uint32_t mulhilo(const std::uint32_t a, const std::uint32_t b, std::uint32_t *hi) {
#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
    __acpp_if_target_sscp(
        namespace jit = sycl::AdaptiveCpp_jit;
        if (jit::reflection_query::compiler_backend::is_known() &&
            jit::reflect<jit::reflection_query::compiler_backend>() == jit::compiler_backend::amdgpu) {
            const std::uint64_t product = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
            *hi = static_cast<std::uint32_t>(product >> 32);
            return static_cast<std::uint32_t>(product);
        });
#endif
    *hi = static_cast<std::uint32_t>((static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b)) >> 32);
    return a * b;
}
} // namespace scram::mc::prng
