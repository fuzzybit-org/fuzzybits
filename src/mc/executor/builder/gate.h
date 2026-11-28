/**
 * @file gate.h
 * @brief The gates of a layer: one kernel per op code, and for counting gates one per cardinality form and slot count,
 *        so that every kernel is compiled for its gates' inputs
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/event/gate.h"
#include "mc/executor/builder/dispatch.h"
#include "mc/executor/builder/layer.h"
#include "mc/executor/builder/op.h"
#include "mc/executor/builder/state.h"
#include "mc/executor/descriptor/partition.h"
#include "mc/kernel/cardinality.h"
#include "mc/kernel/gate.h"
#include "mc/kernel/gate_lanes.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace fuzzybits::mc::executor::builder {

/// The kernel a gate needs: its op code, and the slots its inputs take (kernel::gate_slots_of; for a counting gate, its
/// cardinality form and kernel::cardinality_slots_for). Gates that need the same kernel share one.
struct gate_kernel {
    op_code op = op_code::and_gate;
    /// Counting gates only.
    kernel::cardinality_form form = kernel::cardinality_form::fixed;
    std::uint32_t slots = 0;

    /// The kernel for a gate of op code `op` and `inputs` inputs.
    [[nodiscard]] static constexpr gate_kernel of(const op_code op, const std::size_t inputs) noexcept {
        if (is_counting(op)) {
            return {.op = op, .form = kernel::cardinality_form_for(inputs), .slots = kernel::cardinality_slots_for(inputs)};
        }
        return {.op = op, .slots = kernel::gate_slots_of(op)};
    }

    /// Whether its gates may differ in input count: the counted forms, which take no fixed number of slots. Their
    /// work-groups go in the order for such gates (kernel::gate_order_for).
    [[nodiscard]] constexpr bool counted() const noexcept { return slots == 0; }

    friend constexpr auto operator<=>(const gate_kernel &, const gate_kernel &) = default;
};

/// `choose.template operator()<K>()` with K the kernel type of `variant`, for gates of op code `Op`, in the lanes form
/// if `lanes` (CPUs: vectors of bitpacks per work-item).
template <op_code Op, event::node_types_like T, typename F> [[nodiscard]] auto with_gate_kernel(const gate_kernel &variant, const bool lanes, F &&choose) {
    return select<false, true>(
        lanes,
        [&]<bool lanes_>() {
            if constexpr (is_counting(Op)) {
                using kernel::cardinality_form;
                return select<cardinality_form::fixed, cardinality_form::counter, cardinality_form::wide>(
                    variant.form,
                    [&]<cardinality_form form_>() {
                        if constexpr (form_ == cardinality_form::fixed) {
                            return select_in<kernel::kCardinalitySlots>(
                                variant.slots,
                                [&]<std::uint32_t slots_>() { return choose.template operator()<kernel::cardinality<T, form_, slots_, lanes_>>(); },
                                "cardinality slot count");
                        } else {
                            return choose.template operator()<kernel::cardinality<T, form_, 0, lanes_>>();
                        }
                    },
                    "cardinality form");
            } else {
                return choose.template operator()<kernel::op<Op, T, kernel::gate_slots_of(Op), lanes_>>();
            }
        },
        "lanes form");
}

/**
 * @brief One kernel that evaluates `gates`, gates of `nodes` (layer `layer_index`), all of op code `Op` and needing
 *        `variant`. Returns its partition
 *
 * @details Each gate's inputs are the arena slots of the nodes it reads, the negated ones after the positive ones; a
 * counting gate takes its bounds. The gates take the next arena slots, in the order of `gates`.
 *
 * @throws std::logic_error if a gate reads a node that no earlier kernel computes
 */
template <op_code Op, event::node_types_like T, typename size_t_>
std::size_t build_kernel_of(state<T, size_t_> &build, const layer<T> &nodes, const gate_kernel &variant, const std::span<const gate> gates,
                            const std::size_t layer_index) {
    using index_type = typename T::index_type;
    using extent_type = typename T::extent_type;
    using record = std::conditional_t<is_counting(Op), event::cardinality_gate<T>, event::gate<T>>;

    event::input_block_builder<record, T> records;
    dependencies reads;
    descriptor::fan_in fan_in;
    std::vector<extent_type> positive;
    std::vector<extent_type> negated;
    for (const gate &g : gates) {
        positive.clear();
        negated.clear();
        for (const input &in : nodes.inputs_of(g)) {
            (in.negated ? negated : positive).push_back(build.read(static_cast<index_type>(in.node), reads));
        }
        fan_in.add(positive.size(), negated.size());
        const auto id = static_cast<index_type>(g.index);
        if constexpr (is_counting(Op)) {
            // The layer holds low <= high <= inputs, and the records refuse more inputs than extent_type counts.
            records.add(id, positive, negated, {.low = static_cast<extent_type>(g.range.low), .high = static_cast<extent_type>(g.range.high)});
        } else {
            records.add(id, positive, negated);
        }
    }

    auto block = records.build(build.queue(), build.arena());
    const std::size_t partition = with_gate_kernel<Op, T>(variant, build.device().is_cpu(), [&]<typename kernel_type>() {
        return build.add(kernel_type(block.view(), kernel::gate_order_for(build.device(), variant.counted())), block, reads,
                         {.layer = layer_index, .kind = descriptor::partition_kind::gate, .operation = std::string(name_of(Op)), .inputs = fan_in});
    });
    build.computed(std::move(block), partition);
    return partition;
}

/**
 * @brief The kernel for `gates` of op code `Op` (build_kernel_of), or, for an op code without one (has_kernel), the
 *        stub: it builds nothing and throws
 *
 * @throws not_implemented if no kernel computes gates of op code `Op` yet
 */
template <op_code Op, event::node_types_like T, typename size_t_>
std::size_t build_gate_kernel(state<T, size_t_> &build, const layer<T> &nodes, const gate_kernel &variant, const std::span<const gate> gates,
                              const std::size_t layer_index) {
    if constexpr (has_kernel(Op)) {
        return build_kernel_of<Op>(build, nodes, variant, gates, layer_index);
    } else {
        // TODO:: a kernel for these gates (kernel::has_kernel).
        throw not_implemented("kernel builder: " + std::string(name_of(Op)) + " gates are not implemented: no kernel computes them yet");
    }
}

/// The kernels of the gates of op code `op` in `nodes` (layer `layer_index`): one per gate_kernel they need, in
/// gate_kernel order.
template <event::node_types_like T, typename size_t_>
void build_gates(state<T, size_t_> &build, const layer<T> &nodes, const op_code op, const std::size_t layer_index) {
    const std::span<const gate> gates = nodes.gates(op);
    if (gates.empty()) {
        return;
    }
    std::map<gate_kernel, std::vector<gate>> by_kernel;
    for (const gate &g : gates) {
        by_kernel[gate_kernel::of(op, g.inputs)].push_back(g);
    }
    for (const auto &[variant, group] : by_kernel) {
        (void)select_in<kGateOps>(op, [&]<op_code Op>() { return build_gate_kernel<Op>(build, nodes, variant, group, layer_index); }, "gate op code");
    }
}

} // namespace fuzzybits::mc::executor::builder
