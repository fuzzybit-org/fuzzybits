/**
 * @file form.h
 * @brief The sampler's forms and the Philox variant each runs on, apart from their code (bernoulli.h includes it all)
 */

#pragma once

#include "mc/prng/prng.h"

namespace fuzzybits::mc::prng::bernoulli {

/// The sampler's forms. Each runs on one Philox variant (variant_of), and all write the same bitpacks.
enum struct form {
    /// Scalar: bitpacks first, first + stride, ... one at a time (sample_strided).
    strided,
    /// Vector: W consecutive bitpacks in W lanes, until all are decided (sample_lockstep).
    lockstep,
    /// Vector: lockstep for kChunkGenerations generations, then the bitpacks still undecided W at a time
    /// (sample_chunked).
    chunked,
};

/// The Philox variant that form `f` runs on.
[[nodiscard]] constexpr variant variant_of(const form f) noexcept { return f == form::strided ? variant::scalar : variant::vector; }

} // namespace fuzzybits::mc::prng::bernoulli
