/**
 * @file geometry.h
 * @brief The launch geometry of the sampling kernels: one dimension of events x work-groups per event
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace fuzzybits::mc::kernel::sampler {

/// One dimension of events x work-groups per event: each event's `items` work-items, in whole work-groups of
/// `work_group_` so that none spans two events. The event comes from the work-group index, which compilers know is
/// uniform across the group, so per-event values stay in scalar registers instead of being loaded per work-item. A
/// second dimension for events would cap them at the CUDA grid's 65535 blocks in y.
template <std::size_t work_group_> class geometry {
    std::uint32_t events_ = 0;
    /// Work-items per event that have bitpacks.
    std::uint32_t items_ = 0;
    /// Work-groups per event: items_ rounded up.
    std::uint32_t groups_per_event_ = 0;

  public:
    /// A work-item's event, and its index among the event's work-items.
    struct position {
        std::uint32_t event;
        std::uint32_t item;
    };

    geometry() = default;
    geometry(const std::size_t events, const std::uint64_t items)
        : events_(static_cast<std::uint32_t>(events)), items_(static_cast<std::uint32_t>(items)),
          groups_per_event_(static_cast<std::uint32_t>((items + work_group_ - 1) / work_group_)) {}

    [[nodiscard]] sycl::nd_range<1> range() const {
        return {sycl::range<1>{std::size_t{events_} * groups_per_event_ * work_group_}, sycl::range<1>{work_group_}};
    }
    [[nodiscard]] std::size_t events() const { return events_; }
    /// Work-items per event that have bitpacks.
    [[nodiscard]] std::uint32_t items() const { return items_; }
    /// Work-items per event in the range: items() in whole work-groups.
    [[nodiscard]] std::size_t items_per_event() const { return std::size_t{groups_per_event_} * work_group_; }

    /// The work-item's event and index; false past the event's last work-item with bitpacks.
    [[gnu::always_inline]] bool locate(const sycl::nd_item<1> &item, position &at) const {
        const auto group = static_cast<std::uint32_t>(item.get_group(0));
        const std::uint32_t event = group / groups_per_event_;
        const auto index = static_cast<std::uint32_t>((group - event * groups_per_event_) * work_group_ + item.get_local_id(0));
        if (index >= items_) {
            return false;
        }
        at = {event, index};
        return true;
    }
};

} // namespace fuzzybits::mc::kernel::sampler
