/**
 * @file vector.h
 * @brief The vector variant's sampling kernel: a work-item samples a chunk of an event's consecutive bitpacks, kLanes
 *        at a time in vector lanes, in the lockstep or the chunked form
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/event/basic_event.h"
#include "mc/kernel/sampler/sampler.h"
#include "mc/prng/bernoulli/chunked.h"
#include "mc/prng/bernoulli/form.h"
#include "mc/prng/bernoulli/lockstep.h"
#include "mc/prng/bernoulli/pass_stream.h"
#include "mc/prng/stream.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace fuzzybits::mc::kernel::sampler {

/// Samples events in a vector form, `form_` (lockstep.h, chunked.h): a work-item takes a chunk of an event's consecutive
/// bitpacks and samples them kLanes at a time. A work-group is what an OpenMP thread takes at a time, so it stays a few
/// chunks.
template <event::node_types_like T, form form_> class vector_kernel final : public event_kernel<vector_kernel<T, form_>, T, 4> {
    static_assert(prng::bernoulli::variant_of(form_) == prng::variant::vector, "a vector kernel samples in a vector form");
    using base = event_kernel<vector_kernel<T, form_>, T, 4>;
    friend base;

  public:
    static constexpr form kForm = form_;
    static constexpr int kLanes = 16;

    vector_kernel(const event::sampled_event_block_view<T> &events, const prng::stream &stream, const sycl::device &device)
        : vector_kernel(events, stream, chunk_for(std::uint64_t{events.size()} * events.bitpacks_per_node(), compute_units(device))) {}

  private:
    vector_kernel(const event::sampled_event_block_view<T> &events, const prng::stream &stream, const std::uint32_t chunk)
        : base(events, stream, (std::uint64_t{events.bitpacks_per_node()} + chunk - 1) / chunk), chunk_(chunk) {}

    /// Bitpacks per work-item for `total` bitpacks over `units` compute units: a multiple of kLanes, up to
    /// prng::bernoulli::kMaxChunk, fewer when the pass is too small for 16 work-groups per compute unit. The set-aside
    /// bitpacks of a larger chunk fill their vectors better (prng::bernoulli::sample_chunked).
    [[nodiscard]] static std::uint32_t chunk_for(const std::uint64_t total, const std::uint64_t units) {
        constexpr auto lanes = static_cast<std::uint64_t>(kLanes);
        return static_cast<std::uint32_t>(
            std::clamp<std::uint64_t>(total / (units * 16 * base::kWorkGroup) / lanes * lanes, lanes, prng::bernoulli::kMaxChunk));
    }

    /// Bitpacks [item · chunk, (item + 1) · chunk) of the event's pass, the last chunk cut at `bitpacks`.
    [[gnu::always_inline]] void sample(std::uint64_t *out, const std::uint32_t bitpacks, const prng::bernoulli::pass_stream &source,
                                       const std::uint64_t threshold, const std::uint32_t item) const {
        const std::uint32_t first = item * chunk_;
        const std::uint32_t count = std::min(chunk_, bitpacks - first);
        if constexpr (form_ == form::chunked) {
            prng::bernoulli::sample_chunked<kLanes>(out, first, count, source, threshold);
        } else {
            prng::bernoulli::sample_lockstep<kLanes>(out, first, count, source, threshold);
        }
    }

    /// Bitpacks per work-item (chunk_for).
    std::uint32_t chunk_;
};

} // namespace fuzzybits::mc::kernel::sampler
