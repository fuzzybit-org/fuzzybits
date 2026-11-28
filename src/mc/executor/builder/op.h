/**
 * @file op.h
 * @brief The op codes the builder computes nodes by (kernel/op_code.h): the order in which it builds a layer's gate
 *        kernels, and the names descriptions give them
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details The builder takes nodes by kernel::op_code only, which extends the native IR's (ir::pc::op_code). compat.h says SCRAM's PDAG in
 * them, for the layer manager.
 */

#pragma once

#include "mc/kernel/op_code.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

namespace fuzzybits::mc::executor::builder {

using kernel::has_kernel;
using kernel::is_boolean;
using kernel::is_counting;
using kernel::is_gate;
using kernel::is_unary;
using kernel::op_code;

namespace detail {

static_assert(sizeof(std::underlying_type_t<op_code>) <= 2, "op_code's values are counted at compile time");

/// The number of values of op_code's underlying type, each an op_code, named or not.
inline constexpr std::uint32_t kOpValues = std::uint32_t{std::numeric_limits<std::underlying_type_t<op_code>>::max()} + 1;

/// The number of op codes a gate kernel evaluates.
[[nodiscard]] consteval std::size_t gate_count() {
    std::size_t count = 0;
    for (std::uint32_t value = 0; value < kOpValues; ++value) {
        count += is_gate(static_cast<op_code>(value)) ? 1 : 0;
    }
    return count;
}

} // namespace detail

/// Every op code a gate kernel evaluates (is_gate), in op code order: the order in which the builder builds a layer's
/// gate kernels and descriptions list them.
inline constexpr auto kGateOps = [] {
    std::array<op_code, detail::gate_count()> ops{};
    std::size_t next = 0;
    for (std::uint32_t value = 0; value < detail::kOpValues; ++value) {
        if (const auto op = static_cast<op_code>(value); is_gate(op)) {
            ops[next++] = op;
        }
    }
    return ops;
}();

/// The position of `op` in kGateOps, or kGateOps.size() if no gate kernel evaluates it.
[[nodiscard]] constexpr std::size_t gate_position(const op_code op) noexcept {
    return static_cast<std::size_t>(std::ranges::find(kGateOps, op) - kGateOps.begin());
}

/// The name descriptions give `op`: its op code's, without `_gate` (`and`, `null`, `atleast`, `sample`, `ones`).
[[nodiscard]] constexpr std::string_view name_of(const op_code op) noexcept {
    switch (op) {
    case op_code::basic_event:
        return "basic_event";
    case op_code::constant:
        return "constant";
    case op_code::sample:
        return "sample";
    case op_code::ones:
        return "ones";
    case op_code::zeros:
        return "zeros";
    case op_code::house_event:
        return "house_event";
    case op_code::and_gate:
        return "and";
    case op_code::or_gate:
        return "or";
    case op_code::xor_gate:
        return "xor";
    case op_code::not_gate:
        return "not";
    case op_code::nand_gate:
        return "nand";
    case op_code::nor_gate:
        return "nor";
    case op_code::xnor_gate:
        return "xnor";
    case op_code::null_gate:
        return "null";
    case op_code::cardinality:
        return "cardinality";
    case op_code::atleast:
        return "atleast";
    case op_code::atmost:
        return "atmost";
    case op_code::exactly:
        return "exactly";
    case op_code::majority:
        return "majority";
    case op_code::minority:
        return "minority";
    }
    return "unknown";
}

} // namespace fuzzybits::mc::executor::builder
