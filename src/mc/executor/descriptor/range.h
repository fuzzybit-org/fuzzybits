/**
 * @file range.h
 * @brief A three-dimensional range, as descriptions report work-item ranges and device limits
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>

namespace fuzzybits::mc::executor::descriptor {

/// Three extents and their product: a range of work-items, or of the work a kernel covers.
struct range3 {
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t z = 0;

    [[nodiscard]] constexpr std::size_t product() const noexcept { return x * y * z; }

    friend constexpr bool operator==(const range3 &, const range3 &) = default;
};

/// `range` as a range3, its missing dimensions 1.
template <int dimensions> [[nodiscard]] range3 range_of(const sycl::range<dimensions> &range) {
    static_assert(dimensions >= 1 && dimensions <= 3, "SYCL ranges have one to three dimensions");
    range3 extents{.x = range[0], .y = 1, .z = 1};
    if constexpr (dimensions >= 2) {
        extents.y = range[1];
    }
    if constexpr (dimensions == 3) {
        extents.z = range[2];
    }
    return extents;
}

} // namespace fuzzybits::mc::executor::descriptor
