/**
 * @file strided.h
 * @brief The sampling kernel of the strided form (scalar variant), which GPUs sample with
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/event/basic_event.h"
#include "mc/kernel/sampler/sampler.h"
#include "mc/prng/bernoulli/pass_stream.h"
#include "mc/prng/bernoulli/strided.h"
#include "mc/prng/stream.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace fuzzybits::mc::kernel::sampler {

/// Samples events in the strided form (prng/bernoulli/strided.h): a work-item takes a strided run of up to kMaxRun of
/// an event's bitpacks, and samples them one at a time, starting the next as soon as one is decided.
template <event::node_types_like T> class strided final : public event_kernel<strided<T>, T, 128> {
    using base = event_kernel<strided<T>, T, 128>;
    friend base;

  public:
    static constexpr form kForm = form::strided;
    static constexpr std::uint32_t kMaxRun = 32;

    strided(const event::sampled_event_block_view<T> &events, const prng::stream &stream, const sycl::device &device)
        : base(events, stream, items_for(events.bitpacks_per_node(), std::uint64_t{events.size()} * events.bitpacks_per_node(), compute_units(device))) {}

  private:
    /// Work-items per event of `bitpacks` bitpacks, `total` bitpacks over `units` compute units: runs of up to kMaxRun,
    /// shorter when the pass is too small to keep every compute unit busy (2048 work-items per unit bounds what a GPU
    /// keeps in flight).
    [[nodiscard]] static std::uint64_t items_for(const std::uint64_t bitpacks, const std::uint64_t total, const std::uint64_t units) {
        const std::uint64_t run = std::clamp<std::uint64_t>(total / (units * 2048), 1, kMaxRun);
        return (bitpacks + run - 1) / run;
    }

    /// Bitpacks item, item + items(), ... of the event's pass.
    [[gnu::always_inline]] void sample(std::uint64_t *out, const std::uint32_t bitpacks, const prng::bernoulli::pass_stream &source,
                                       const std::uint64_t threshold, const std::uint32_t item) const {
        prng::bernoulli::sample_strided(out, item, this->items(), bitpacks, source, threshold);
    }
};

} // namespace fuzzybits::mc::kernel::sampler
