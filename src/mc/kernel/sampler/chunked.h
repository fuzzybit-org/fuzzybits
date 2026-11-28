/**
 * @file chunked.h
 * @brief The sampling kernel of the chunked form (vector variant), which CPUs sample with
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/kernel/sampler/vector.h"

namespace fuzzybits::mc::kernel::sampler {

/// Samples events in the chunked form (prng/bernoulli/chunked.h): a work-item takes a chunk of an event's consecutive
/// bitpacks, samples them kLanes at a time in vector lanes for a few generations, sets aside the bitpacks still
/// undecided, and finishes those kLanes at a time.
template <event::node_types_like T> using chunked = vector_kernel<T, form::chunked>;

} // namespace fuzzybits::mc::kernel::sampler
