/**
 * @file execution_plan.h
 * @brief Everything a layer manager reports about how it runs a pass: the device, the pass's shape, the partitions and
 *        their edges, and the PDAG's nodes
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/executor/descriptor/device.h"
#include "mc/executor/descriptor/node.h"
#include "mc/executor/descriptor/partition.h"
#include "mc/executor/descriptor/plan.h"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fuzzybits::mc::executor::descriptor {

/// The plan of a pass, as the `schema` below reports it. `shaper_description` is the sample shaper's own
/// report of how it shaped the pass (sample_shaper::description).
template <typename shaper_description> struct execution_plan {
    static constexpr std::string_view schema = "fuzzybits.execution_plan.v1";

    std::string bitpack_type;
    std::size_t bits_per_bitpack = 0;
    std::size_t node_count = 0;
    device_profile device;
    shaper_description sample_shaper;
    std::vector<partition> partitions;
    std::vector<edge> dependency_edges;
    std::vector<node> nodes;
};

/// The execution plan of `plan` on `device`, with bitpacks of type `bitpack`, the shaper's description, and the
/// descriptions of the PDAG's `node_count` nodes.
template <typename bitpack, typename shaper_description>
[[nodiscard]] execution_plan<shaper_description> execution_plan_of(const plan &plan, const sycl::device &device, shaper_description shaper,
                                                                   const std::size_t node_count, std::vector<node> nodes) {
    static_assert(sizeof(bitpack) == sizeof(std::uint64_t) || sizeof(bitpack) == sizeof(std::uint32_t), "the plan names 32- and 64-bit bitpacks");
    return {
        .bitpack_type = sizeof(bitpack) == sizeof(std::uint64_t) ? "uint64" : "uint32",
        .bits_per_bitpack = sizeof(bitpack) * 8,
        .node_count = node_count,
        .device = profile_of(device),
        .sample_shaper = std::move(shaper),
        .partitions = {plan.partitions().begin(), plan.partitions().end()},
        .dependency_edges = plan.edges(),
        .nodes = std::move(nodes),
    };
}

} // namespace fuzzybits::mc::executor::descriptor
