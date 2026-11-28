/**
 * @file partition.h
 * @brief A partition of the execution plan: one kernel, the nodes it computes, the partitions it reads, and how it is
 *        launched
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/executor/descriptor/range.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace fuzzybits::mc::executor::descriptor {

/// What a partition's kernel does.
enum class partition_kind : std::uint8_t {
    /// Samples basic events (kernel/sampler/).
    sampler,
    /// Evaluates gates (kernel/gate.h, kernel/cardinality.h).
    gate,
    /// Counts the true trials of tallied nodes (kernel/tally.h).
    tally,
};

[[nodiscard]] constexpr std::string_view name_of(const partition_kind kind) noexcept {
    switch (kind) {
    case partition_kind::sampler:
        return "sampler";
    case partition_kind::gate:
        return "gate";
    case partition_kind::tally:
        return "tally";
    }
    return "unknown";
}

/// The inputs of a partition's gates: the least, greatest and total count of a gate's inputs, and how many inputs are
/// positive and how many negated. All zero for a partition without gates.
struct fan_in {
    std::size_t gates = 0;
    std::size_t min = 0;
    std::size_t max = 0;
    std::size_t total = 0;
    std::size_t positive = 0;
    std::size_t negated = 0;

    /// Counts a gate of `positive_inputs` positive and `negated_inputs` negated inputs.
    constexpr void add(const std::size_t positive_inputs, const std::size_t negated_inputs) noexcept {
        const std::size_t inputs = positive_inputs + negated_inputs;
        min = gates == 0 ? inputs : std::min(min, inputs);
        max = std::max(max, inputs);
        total += inputs;
        positive += positive_inputs;
        negated += negated_inputs;
        ++gates;
    }
};

/// How a partition's kernel is launched. Every kernel's range is one dimension of its nodes × work-items per node, in
/// work-groups of one size, which descriptions report in three: global {nodes, 1, work-items per node} and local
/// {1, 1, work-group}.
struct launch {
    range3 global{};
    range3 local{};

    /// The launch of a kernel of `nodes` nodes over `range`.
    [[nodiscard]] static launch of(const sycl::nd_range<1> &range, const std::size_t nodes) {
        const std::size_t items = range.get_global_range()[0];
        return {.global = {.x = nodes, .y = 1, .z = nodes == 0 ? 0 : items / nodes}, .local = {.x = 1, .y = 1, .z = range.get_local_range()[0]}};
    }
};

/// One kernel of the plan. The fields are what the plan knows; the functions what follows from them.
struct partition {
    /// Its position in the plan, which is its submission order.
    std::size_t id = 0;
    std::size_t layer = 0;
    partition_kind kind = partition_kind::sampler;
    /// What the kernel computes: the op code of its nodes (builder::name_of: `sample` for basic events, `and`,
    /// `atleast`, ... for gates), `popcount` for tallies.
    std::string operation;
    /// The PDAG indices of its nodes, in its block's order.
    std::vector<std::int32_t> nodes;
    /// The partitions whose outputs it reads, ascending.
    std::vector<std::size_t> dependencies;
    launch shape{};
    /// A pass: `batches` × `bitpacks_per_batch` bitpacks per node, of `bitpack_bytes` bytes each.
    std::size_t batches = 0;
    std::size_t bitpacks_per_batch = 0;
    std::size_t bitpack_bytes = 0;
    fan_in inputs{};

    [[nodiscard]] std::size_t node_count() const noexcept { return nodes.size(); }
    /// The work it covers: nodes × batches × bitpacks per batch.
    [[nodiscard]] range3 logical_range() const noexcept { return {.x = nodes.size(), .y = batches, .z = bitpacks_per_batch}; }
    [[nodiscard]] std::size_t logical_work_items() const noexcept { return logical_range().product(); }
    [[nodiscard]] std::size_t work_items() const noexcept { return shape.global.product(); }
    [[nodiscard]] std::size_t work_groups() const noexcept { return work_items() / std::max<std::size_t>(shape.local.product(), 1); }
    /// Work-items beyond the work: those that round the range up to whole work-groups, or that cover several bitpacks.
    [[nodiscard]] std::size_t overprovisioned_work_items() const noexcept {
        return work_items() > logical_work_items() ? work_items() - logical_work_items() : 0;
    }
    [[nodiscard]] std::size_t bitpacks_per_node() const noexcept { return batches * bitpacks_per_batch; }
    [[nodiscard]] std::size_t bytes_per_node() const noexcept { return bitpacks_per_node() * bitpack_bytes; }
    [[nodiscard]] std::size_t bits_per_node() const noexcept { return bytes_per_node() * 8; }
    /// The bytes of its nodes' outputs, which stay on the device.
    [[nodiscard]] std::size_t resident_bytes() const noexcept { return nodes.size() * bytes_per_node(); }
};

/// An edge of the plan: partition `target` reads an output of partition `source`.
struct edge {
    /// Why `target` waits for `source`: the only reason the plan knows.
    static constexpr std::string_view reason = "buffer_dependency";

    std::size_t source = 0;
    std::size_t target = 0;
};

} // namespace fuzzybits::mc::executor::descriptor
