/**
 * @file compat.h
 * @brief SCRAM's PDAG gates in the builder's op codes (op.h): the op code a gate is computed by, and a counting gate's
 *        bounds
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details The builder names what computes a node by kernel::op_code, which extends the native IR's (ir::pc::op_code); SCRAM's PDAG names a
 * gate's operation by its connective (core::Connective), and holds its constants as gates. This file says the one in
 * the other. Only executor/compat.h, which reads the PDAG, includes it: no builder file reads SCRAM's code. Retiring
 * SCRAM (#144) deletes it with executor/compat.h.
 */

#pragma once

#include "pdag.h"

#include "mc/executor/builder/layer.h"
#include "mc/executor/builder/op.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace fuzzybits::mc::executor::compat {

/// SCRAM's PDAG namespace, by the name its code gives it.
namespace core = scram::core;

/// The op code of SCRAM's connective `c`.
[[nodiscard]] constexpr builder::op_code op_code_of(const core::Connective c) {
    using builder::op_code;
    switch (c) {
    case core::kAnd:
        return op_code::and_gate;
    case core::kOr:
        return op_code::or_gate;
    case core::kAtleast:
        return op_code::atleast;
    case core::kXor:
        return op_code::xor_gate;
    case core::kNot:
        return op_code::not_gate;
    case core::kNand:
        return op_code::nand_gate;
    case core::kNor:
        return op_code::nor_gate;
    case core::kNull:
        return op_code::null_gate;
    }
    throw std::invalid_argument("compat: SCRAM connective " + std::to_string(static_cast<int>(c)) + " has no op code");
}

/// The op code `gate` is computed by. The kernels read a gate's Variable and Gate arguments only; the PDAG holds its
/// constant apart (`Gate::constant()`), so a gate that holds it is handled here:
///
/// - a constant gate (its only argument is the constant, as `Gate::MakeConstant` leaves it) is `ones` (TRUE) or
///   `zeros` (FALSE), whose bitpacks the builder writes once;
/// - a gate that holds the constant beside other arguments is refused before sampling. Dropping the constant would
///   silently change the gate's function (#259).
///
/// Any other gate is its connective's op code.
[[nodiscard]] inline builder::op_code op_code_of(const core::Gate &gate) {
    if (!gate.constant()) {
        return op_code_of(gate.type());
    }
    if (gate.args().size() != 1 || !gate.args<core::Gate>().empty() || !gate.args<core::Variable>().empty()) {
        throw std::runtime_error("PDAG gate G" + std::to_string(gate.index()) +
                                 " holds the Boolean constant beside other arguments; the Monte Carlo kernels cannot evaluate it");
    }
    // The one argument is the constant's index, negative for FALSE.
    return *gate.args().begin() > 0 ? builder::op_code::ones : builder::op_code::zeros;
}

/// The bounds of counting gate `gate`, of `inputs` inputs. SCRAM's one counting gate is at least k of n, [k, n].
///
/// @throws std::logic_error if `gate` is not an at-least gate, or its k is negative
[[nodiscard]] inline builder::bounds bounds_of(const core::Gate &gate, const std::size_t inputs) {
    if (gate.type() != core::kAtleast || gate.min_number() < 0) {
        throw std::logic_error("PDAG gate G" + std::to_string(gate.index()) + " is not an at-least gate of a non-negative threshold");
    }
    return {.low = static_cast<std::uint32_t>(gate.min_number()), .high = static_cast<std::uint32_t>(inputs)};
}

} // namespace fuzzybits::mc::executor::compat
