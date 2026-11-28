/**
 * @file sampler.h
 * @brief What the basic-event sampling kernels share: the form a device samples with, the run's stream as a kernel
 *        argument, the work-item's way to its share of an event's pass (event_kernel), and the bitpacks of constant
 *        laws, which no kernel samples
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details Each sampled basic event's bitpacks are exact Bernoulli(T / 2^64) draws from the bit-sliced Philox sampler
 * (prng/bernoulli/bernoulli.h) at the counters of prng/stream.h: the event's node, the run's device and each bitpack's
 * run-wide index, under the run's seed. There is one kernel per form of the sampler, and all write the same bitpacks:
 *
 *     kernel      form (variant)      a work-item samples                                    device
 *     strided.h   strided (scalar)    a strided run of an event's bitpacks, one at a time    GPUs
 *     lockstep.h  lockstep (vector)   a chunk of bitpacks, kLanes at a time in lockstep       none: tests
 *     chunked.h   chunked (vector)    a chunk of bitpacks, the slow ones set aside            CPUs
 *
 * Every kernel is an event_kernel, which finds a work-item's event and builds the counters of the event's pass; the
 * kernel then samples its share of the pass in its form. The two vector forms share their kernel (vector.h).
 *
 * The host picks the form per device type (form_for). The kernels read only events whose trials are random
 * (event::sampled_event), so none has a case for a constant law. An event whose law is `never` or `always` has the same
 * bitpacks in every pass: write_constant_outputs writes them once, when the event's arena slot is allocated, and no
 * kernel is built for it.
 *
 * The run's stream is expanded on the host into the seed's ten Philox round keys and the device (prng::expanded_stream,
 * 84 bytes), held as a `sycl::specialized` argument:
 * - AdaptiveCpp's generic (SSCP) JIT compiles the value into the kernel, so the round keys are immediates, as they were
 *   for the fixed key the seed replaced. One JIT compile per seed, device and kernel.
 * - The CUDA AOT presets (cuda-sm70, cuda-sm86, cuda-sm120), and a generic JIT at ACPP_ADAPTIVITY_LEVEL=0, specialize
 *   nothing: the round keys are an ordinary argument, in the constant bank on NVIDIA GPUs, never computed per call.
 * The only argument that changes between launches is the pass index.
 */

#pragma once

#include "mc/event/basic_event.h"
#include "mc/kernel/sampler/geometry.h"
#include "mc/prng/bernoulli/form.h"
#include "mc/prng/bernoulli/pass_stream.h"
#include "mc/prng/stream.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fuzzybits::mc::kernel::sampler {

/// The sampler's forms (prng/bernoulli/form.h), one kernel each.
using form = prng::bernoulli::form;

/// The form the host samples with on `device`: chunked on CPUs, strided on other devices. No device gets lockstep; the
/// tests check its kernel on the device like the others.
[[nodiscard]] inline form form_for(const sycl::device &device) { return device.is_cpu() ? form::chunked : form::strided; }

/// The run's expanded stream as a kernel argument. AdaptiveCpp defines ACPP_EXT_SPECIALIZED in every compilation flow;
/// the plain type is for other SYCL implementations.
#if defined(ACPP_EXT_SPECIALIZED)
using stream_argument = sycl::specialized<prng::expanded_stream>;
#else
using stream_argument = prng::expanded_stream;
#endif

/// The device's compute units, at least one.
[[nodiscard]] inline std::uint64_t compute_units(const sycl::device &device) {
    return std::max<std::uint64_t>(device.get_info<sycl::info::device::max_compute_units>(), 1);
}

/**
 * @brief A sampling kernel: its events, the run's stream and its launch geometry, and the steps every work-item takes
 *        before it samples
 *
 * @details A work-item finds its event from its work-group (geometry.h) and builds the counters of the event's pass
 * (prng::bernoulli::pass_stream); `Derived`, the kernel, then samples the work-item's share of the pass:
 *
 *     void sample(std::uint64_t *out, std::uint32_t bitpacks, const prng::bernoulli::pass_stream &source,
 *                 std::uint64_t threshold, std::uint32_t item) const;
 *
 * out[0 .. bitpacks) is the event's pass of bitpacks, `source` their counters, `threshold` the event's T, and `item`
 * the work-item's index among the event's `items()` work-items. Derived chooses `items` when it is built.
 */
template <class Derived, event::node_types_like T, std::size_t work_group_> class event_kernel {
    static_assert(std::same_as<typename T::bitpack_type, std::uint64_t>, "the bit-sliced sampler defines 64-trial bitpacks");

  public:
    static constexpr std::size_t kWorkGroup = work_group_;

    [[nodiscard]] sycl::nd_range<1> get_range() const { return geometry_.range(); }
    [[nodiscard]] std::size_t events() const { return geometry_.events(); }
    [[nodiscard]] std::size_t items_per_event() const { return geometry_.items_per_event(); }

    /// Samples pass `pass` (from 0): bitpacks [pass·B, (pass + 1)·B) of every event's run, B its bitpacks per pass.
    void operator()(const sycl::nd_item<1> &item, const std::uint32_t pass) const {
        typename geometry<work_group_>::position at;
        if (!geometry_.locate(item, at)) {
            return;
        }
        const auto &ev = events_[at.event];
        const auto out = events_.output(at.event);
        const prng::expanded_stream run = stream_;
        const prng::bernoulli::pass_stream source(run, static_cast<std::uint32_t>(ev.id_in_pdag), std::uint64_t{pass} * out.size());
        static_cast<const Derived &>(*this).sample(out.data(), static_cast<std::uint32_t>(out.size()), source, ev.law.threshold(), at.item);
    }

  protected:
    /// `events`, sampled in `stream`, `items` work-items per event.
    event_kernel(const event::sampled_event_block_view<T> &events, const prng::stream &stream, const std::uint64_t items)
        : events_(events), stream_(prng::expanded_stream::of(stream)), geometry_(events.size(), items) {}

    /// Work-items per event that have bitpacks.
    [[nodiscard]] std::uint32_t items() const { return geometry_.items(); }

  private:
    event::sampled_event_block_view<T> events_;
    stream_argument stream_;
    geometry<work_group_> geometry_;
};

/// Writes the bitpacks of `block`'s events once: all zeros for `never`, all ones for `always`. Call it when the block's
/// arena slots are allocated; no pass writes them again, so no kernel is built for these events. Events next to each
/// other with the same value are one fill. Returns when the bitpacks are written.
template <event::node_types_like T> void write_constant_outputs(sycl::queue &queue, const event::constant_event_block<T> &block) {
    using bitpack_type = typename T::bitpack_type;
    const auto view = block.view();
    const std::span<const event::constant_event<T>> events = block.records();
    std::vector<sycl::event> writes;
    for (std::size_t first = 0; first < events.size();) {
        std::size_t last = first + 1;
        while (last < events.size() && events[last].value == events[first].value) {
            ++last;
        }
        // Consecutive events of a block hold consecutive slots, whose bitpacks are adjacent in the arena.
        const bitpack_type value = events[first].value ? ~bitpack_type{0} : bitpack_type{0};
        writes.push_back(queue.fill(view.output(first).data(), value, (last - first) * std::size_t{view.bitpacks_per_node()}));
        first = last;
    }
    sycl::event::wait_and_throw(writes);
}

} // namespace fuzzybits::mc::kernel::sampler
