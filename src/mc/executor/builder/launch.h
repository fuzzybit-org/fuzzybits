/**
 * @file launch.h
 * @brief A kernel as a queueable: the submission that runs it over its range, after the queueables it reads
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/executor/queueable.h"

#include <sycl/sycl.hpp>

#include <concepts>
#include <cstdint>
#include <memory>
#include <vector>

namespace fuzzybits::mc::executor::builder {

/// A kernel that takes, besides its work-item, the pass it belongs to: kernel(item, pass), as the sampling kernels do,
/// whose Philox counters depend on it.
template <typename kernel_type>
concept takes_pass = std::invocable<const kernel_type &, const sycl::nd_item<1> &, std::uint32_t>;

/// `kernel` over its range (get_range), as a queueable that runs after `dependencies` and hands the kernel its pass if
/// it takes one.
template <typename kernel_type>
[[nodiscard]] std::shared_ptr<queueable_base> queueable_of(sycl::queue &queue, const kernel_type &kernel,
                                                           const std::vector<std::shared_ptr<queueable_base>> &dependencies) {
    if constexpr (takes_pass<kernel_type>) {
        return std::make_shared<iterable_queueable<kernel_type, 1>>(queue, kernel, kernel.get_range(), dependencies);
    } else {
        return std::make_shared<queueable<kernel_type, 1>>(queue, kernel, kernel.get_range(), dependencies);
    }
}

} // namespace fuzzybits::mc::executor::builder
