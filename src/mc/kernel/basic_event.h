/**
 * @file basic_event.h
 * @brief SYCL kernel that samples basic events into bitpacks
 * @author Arjun Earthperson
 * @date 2025
 *
 * @details Each basic event's bitpacks are exact Bernoulli(T / 2^64) draws from the bit-sliced Philox
 * sampler defined in prng/bernoulli.h. The kernel has two forms that produce the same bitpacks; the host
 * picks one per device type (`sampler_form_for`).
 *
 * @copyright Copyright (C) 2025 Arjun Earthperson
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "mc/event/node.h"
#include "mc/event/sample_shape.h"
#include "mc/prng/bernoulli.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace scram::mc::kernel {

/// How the basic-event kernel spreads an event's bitpacks over work-items.
enum class sampler_form {
    /// GPU: a work-item samples a strided run of up to `kMaxRun` bitpacks one at a time, starting the next
    /// as soon as one is decided (`bernoulli::sample_strided`).
    strided,
    /// CPU: a work-item samples `kLanes` consecutive bitpacks in vector registers (`bernoulli::sample_lanes`).
    lanes,
};

inline sampler_form sampler_form_for(const sycl::device &device) {
    return device.is_cpu() ? sampler_form::lanes : sampler_form::strided;
}

template <typename prob_t_, typename bitpack_t_, typename size_t_, sampler_form form_> class basic_event {
    static_assert(std::is_same_v<bitpack_t_, std::uint64_t>, "the bit-sliced sampler defines 64-trial bitpacks");

    using be_block = event::basic_event_block<prob_t_, bitpack_t_>;
    be_block basic_events_block_;

    /// Bitpacks per event per iteration (batch_size * bitpacks_per_batch).
    std::uint32_t bitpacks_;

    /// Work-items per event: `strided` runs, or `lanes` groups of kLanes bitpacks.
    std::uint32_t items_per_event_;

    /// Work-groups per event: items_per_event_ rounded up to whole work-groups, so none spans two events.
    std::uint32_t groups_per_event_;

  public:
    static constexpr int kLanes = 16;
    static constexpr std::uint32_t kMaxRun = 32;
    static constexpr std::size_t kWorkGroup = form_ == sampler_form::strided ? 128 : 64;

    basic_event(const be_block &basic_events_block, const event::sample_shape<size_t_> &sample_shape,
                const sycl::device &device)
        : basic_events_block_(basic_events_block),
          bitpacks_(static_cast<std::uint32_t>(sample_shape.batch_size * sample_shape.bitpacks_per_batch)) {
        if constexpr (form_ == sampler_form::strided) {
            // Up to kMaxRun bitpacks per work-item, fewer when the iteration is too small to keep every compute
            // unit busy (2048 work-items per unit bounds what a GPU keeps in flight).
            const std::size_t units =
                std::max<std::size_t>(device.get_info<sycl::info::device::max_compute_units>(), 1);
            const std::size_t total = basic_events_block_.count * static_cast<std::size_t>(bitpacks_);
            const auto run = static_cast<std::uint32_t>(std::clamp<std::size_t>(total / (units * 2048), 1, kMaxRun));
            items_per_event_ = (bitpacks_ + run - 1) / run;
        } else {
            (void)device;
            items_per_event_ = (bitpacks_ + kLanes - 1) / kLanes;
        }
        groups_per_event_ = static_cast<std::uint32_t>((items_per_event_ + kWorkGroup - 1) / kWorkGroup);
    }

    /// One dimension of events x work-groups per event. A second dimension for events would cap them at the
    /// CUDA grid's 65535 blocks in y.
    [[nodiscard]] sycl::nd_range<1> get_range() const {
        return {sycl::range<1>{basic_events_block_.count * groups_per_event_ * kWorkGroup}, sycl::range<1>{kWorkGroup}};
    }

    [[nodiscard]] std::size_t events() const { return basic_events_block_.count; }
    [[nodiscard]] std::size_t items_per_event() const { return groups_per_event_ * kWorkGroup; }

    void operator()(const sycl::nd_item<1> &item, const std::uint32_t iteration) const {
        // The event comes from the work-group index, which compilers know is uniform across the group, so
        // per-event values stay in scalar registers instead of being loaded per work-item.
        const auto group = static_cast<std::uint32_t>(item.get_group(0));
        const auto event_idx = group / groups_per_event_;
        const auto item_idx =
            static_cast<std::uint32_t>((group - event_idx * groups_per_event_) * kWorkGroup + item.get_local_id(0));
        if (item_idx >= items_per_event_) {
            return;
        }
        const auto &event = basic_events_block_[event_idx];
        bitpack_t_ *output = event.buffer;
        const auto pdag_idx = static_cast<std::uint32_t>(event.index);

        if constexpr (form_ == sampler_form::strided) {
            if (event.always_true || event.threshold == 0) {
                const bitpack_t_ value = event.always_true ? ~bitpack_t_{0} : bitpack_t_{0};
                for (std::uint32_t s = item_idx; s < bitpacks_; s += items_per_event_) {
                    output[s] = value;
                }
                return;
            }
            prng::bernoulli::sample_strided(output, item_idx, items_per_event_, bitpacks_, pdag_idx, event_idx,
                                            iteration, event.threshold);
        } else {
            const std::uint32_t first = item_idx * kLanes;
            const std::uint32_t valid = std::min<std::uint32_t>(kLanes, bitpacks_ - first);
            if (event.always_true || event.threshold == 0) {
                const bitpack_t_ value = event.always_true ? ~bitpack_t_{0} : bitpack_t_{0};
                for (std::uint32_t i = 0; i < valid; ++i) {
                    output[first + i] = value;
                }
                return;
            }
            prng::bernoulli::sample_lanes<kLanes>(output + first, valid, first, pdag_idx, event_idx, iteration,
                                                  event.threshold);
        }
    }
};
} // namespace scram::mc::kernel
