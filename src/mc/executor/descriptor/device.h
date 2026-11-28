/**
 * @file device.h
 * @brief The device a plan runs on, as descriptions report it, and the names of SYCL's device enumerations
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/executor/descriptor/range.h"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace fuzzybits::mc::executor::descriptor {

[[nodiscard]] constexpr std::string_view name_of(const sycl::info::device_type type) noexcept {
    switch (type) {
    case sycl::info::device_type::cpu:
        return "cpu";
    case sycl::info::device_type::gpu:
        return "gpu";
    case sycl::info::device_type::accelerator:
        return "accelerator";
    case sycl::info::device_type::custom:
        return "custom";
    case sycl::info::device_type::automatic:
        return "automatic";
    case sycl::info::device_type::host:
        return "host";
    case sycl::info::device_type::all:
        return "all";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view name_of(const sycl::info::global_mem_cache_type type) noexcept {
    switch (type) {
    case sycl::info::global_mem_cache_type::none:
        return "none";
    case sycl::info::global_mem_cache_type::read_only:
        return "read_only";
    case sycl::info::global_mem_cache_type::read_write:
        return "read_write";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view name_of(const sycl::info::local_mem_type type) noexcept {
    switch (type) {
    case sycl::info::local_mem_type::none:
        return "none";
    case sycl::info::local_mem_type::local:
        return "local";
    case sycl::info::local_mem_type::global:
        return "global";
    }
    return "unknown";
}

/// A device's identity, limits and memory, from its SYCL queries.
struct device_profile {
    std::string type;
    std::string name;
    std::uint32_t vendor_id = 0;
    std::string vendor;
    std::string driver_version;
    std::string profile;
    std::string version;
    std::string opencl_c_version;
    std::vector<int> aspects;
    std::vector<std::string> extensions;
    std::uint32_t max_compute_units = 0;
    std::uint32_t max_clock_frequency = 0;
    std::uint32_t max_work_item_dimensions = 0;
    range3 max_work_item_sizes_1d{};
    range3 max_work_item_sizes_2d{};
    range3 max_work_item_sizes_3d{};
    /// SYCL 2020 has no query for it; always false.
    bool work_item_independent_forward_progress = false;
    std::size_t max_work_group_size = 0;
    std::uint32_t max_num_sub_groups = 0;
    bool sub_group_independent_forward_progress = false;
    std::vector<std::size_t> sub_group_sizes;
    std::uint32_t preferred_vector_width_char = 0;
    std::uint64_t max_mem_alloc_size = 0;
    std::uint64_t global_mem_size = 0;
    std::uint64_t global_mem_cache_size = 0;
    std::uint32_t global_mem_cache_line_size = 0;
    std::string global_mem_cache_type;
    std::string local_mem_type;
    std::uint64_t local_mem_size = 0;
};

/// The profile of `device`.
[[nodiscard]] inline device_profile profile_of(const sycl::device &device) {
    namespace info = sycl::info::device;
    std::vector<int> aspects;
    for (const sycl::aspect aspect : device.get_info<info::aspects>()) {
        aspects.push_back(static_cast<int>(aspect));
    }
    return {
        .type = std::string(name_of(device.get_info<info::device_type>())),
        .name = device.get_info<info::name>(),
        .vendor_id = static_cast<std::uint32_t>(device.get_info<info::vendor_id>()),
        .vendor = device.get_info<info::vendor>(),
        .driver_version = device.get_info<info::driver_version>(),
        .profile = device.get_info<info::profile>(),
        .version = device.get_info<info::version>(),
        .opencl_c_version = device.get_info<info::opencl_c_version>(),
        .aspects = std::move(aspects),
        .extensions = device.get_info<info::extensions>(),
        .max_compute_units = static_cast<std::uint32_t>(device.get_info<info::max_compute_units>()),
        .max_clock_frequency = static_cast<std::uint32_t>(device.get_info<info::max_clock_frequency>()),
        .max_work_item_dimensions = static_cast<std::uint32_t>(device.get_info<info::max_work_item_dimensions>()),
        .max_work_item_sizes_1d = range_of(device.get_info<info::max_work_item_sizes<1>>()),
        .max_work_item_sizes_2d = range_of(device.get_info<info::max_work_item_sizes<2>>()),
        .max_work_item_sizes_3d = range_of(device.get_info<info::max_work_item_sizes<3>>()),
        .work_item_independent_forward_progress = false,
        .max_work_group_size = device.get_info<info::max_work_group_size>(),
        .max_num_sub_groups = static_cast<std::uint32_t>(device.get_info<info::max_num_sub_groups>()),
        .sub_group_independent_forward_progress = device.get_info<info::sub_group_independent_forward_progress>(),
        .sub_group_sizes = device.get_info<info::sub_group_sizes>(),
        .preferred_vector_width_char = static_cast<std::uint32_t>(device.get_info<info::preferred_vector_width_char>()),
        .max_mem_alloc_size = static_cast<std::uint64_t>(device.get_info<info::max_mem_alloc_size>()),
        .global_mem_size = static_cast<std::uint64_t>(device.get_info<info::global_mem_size>()),
        .global_mem_cache_size = static_cast<std::uint64_t>(device.get_info<info::global_mem_cache_size>()),
        .global_mem_cache_line_size = static_cast<std::uint32_t>(device.get_info<info::global_mem_cache_line_size>()),
        .global_mem_cache_type = std::string(name_of(device.get_info<info::global_mem_cache_type>())),
        .local_mem_type = std::string(name_of(device.get_info<info::local_mem_type>())),
        .local_mem_size = static_cast<std::uint64_t>(device.get_info<info::local_mem_size>()),
    };
}

} // namespace fuzzybits::mc::executor::descriptor
