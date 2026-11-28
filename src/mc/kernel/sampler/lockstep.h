/**
 * @file lockstep.h
 * @brief The sampling kernel of the lockstep form (vector variant). No device samples with it (form_for); the tests
 *        check it on the device like the other kernels
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/kernel/sampler/vector.h"

namespace fuzzybits::mc::kernel::sampler {

/// Samples events in the lockstep form (prng/bernoulli/lockstep.h): a work-item takes the chunk of an event's
/// consecutive bitpacks that the chunked kernel would, and samples them kLanes at a time in vector lanes, each group
/// until all of its bitpacks are decided.
template <event::node_types_like T> using lockstep = vector_kernel<T, form::lockstep>;

} // namespace fuzzybits::mc::kernel::sampler
