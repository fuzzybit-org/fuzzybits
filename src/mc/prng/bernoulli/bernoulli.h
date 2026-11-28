/**
 * @file bernoulli.h
 * @brief The bit-sliced Bernoulli sampler: exact Bernoulli(T / 2^64) trials, 64 to a bitpack, from Philox4x32-10, and
 *        its forms
 *
 * @details A node's bitpack is a function of the run's stream, the node, the bitpack's run-wide index and the threshold
 * T only (sample.h defines it, and samples it one bitpack at a time as the reference). The forms compute that function
 * for many bitpacks at once, each on one of Philox's variants (prng/prng.h; form.h):
 *
 *     variant  form      header       bitpacks of a work-item
 *     scalar   strided   strided.h    first, first + stride, ..., one at a time (GPU kernels)
 *     vector   lockstep  lockstep.h   W consecutive ones in W lanes, until all are decided
 *     vector   chunked   chunked.h    a chunk in lockstep for a few generations, then the undecided W at a time
 *                                     (CPU kernels)
 *
 * Every form compares generation by generation with compare_generation, which is written once over the variant, so
 * every form writes the reference's bitpacks. A form depends only on what the forms share, never on another form:
 * pass_stream.h (the counters of a pass's bitpacks), sample.h (the comparison and the reference) and, for the vector
 * forms, simd.h (lane helpers and the lockstep generation loop).
 */

#pragma once

#include "mc/prng/bernoulli/chunked.h"
#include "mc/prng/bernoulli/form.h"
#include "mc/prng/bernoulli/lockstep.h"
#include "mc/prng/bernoulli/pass_stream.h"
#include "mc/prng/bernoulli/sample.h"
#include "mc/prng/bernoulli/simd.h"
#include "mc/prng/bernoulli/strided.h"
