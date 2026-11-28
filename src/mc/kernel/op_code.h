/**
 * @file op_code.h
 * @brief The op codes the kernels compute nodes by, as the native IR names them (ir::pc::op_code), and which kernel
 *        evaluates which
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details Two kernels evaluate gates:
 *
 * - kernel::op (gate.h), the Boolean gates: AND, OR and XOR fold their inputs, NAND, NOR and XNOR complement the fold,
 *   NOT complements its one input and NULL passes it through;
 * - kernel::cardinality (cardinality.h), the counting gates: true where the number of their true inputs lies in their
 *   bounds. The op code names the gate (`cardinality`, `atleast`, `atmost`, `exactly`); its bounds alone say what it
 *   computes.
 *
 * The other op codes name nodes that no gate kernel computes: those sampled from their law (`sample`), and constants
 * (`ones`, `zeros`).
 *
 * These names are not those of ir::pc's own predicates (ir::pc::is_boolean_gate), which argument-dependent lookup
 * finds for any op code, so that an unqualified call is never ambiguous.
 */

#pragma once

#include "ir/pc/op_code.h"

namespace fuzzybits::mc::kernel {

/// What computes a node (fuzzybits::ir::pc::op_code).
using op_code = fuzzybits::ir::pc::op_code;

/// Whether kernel::op evaluates gates of op code `op`: the Boolean gates, and NULL.
[[nodiscard]] constexpr bool is_boolean(const op_code op) noexcept {
    return op == op_code::and_gate || op == op_code::or_gate || op == op_code::xor_gate || op == op_code::not_gate || op == op_code::nand_gate ||
           op == op_code::nor_gate || op == op_code::xnor_gate || op == op_code::null_gate;
}

/// Whether kernel::cardinality evaluates gates of op code `op`: the counting gates.
[[nodiscard]] constexpr bool is_counting(const op_code op) noexcept {
    return op == op_code::cardinality || op == op_code::atleast || op == op_code::atmost || op == op_code::exactly;
}

/// Whether a gate kernel evaluates gates of op code `op`.
[[nodiscard]] constexpr bool is_gate(const op_code op) noexcept { return is_boolean(op) || is_counting(op); }

/// Whether a gate of op code `op` reads exactly one input: NOT and NULL.
[[nodiscard]] constexpr bool is_unary(const op_code op) noexcept { return op == op_code::not_gate || op == op_code::null_gate; }

} // namespace fuzzybits::mc::kernel
