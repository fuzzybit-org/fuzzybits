/**
 * @file op_code.h
 * @brief The op codes the kernels compute nodes by: ir::pc::op_code, extended with the operations only execution has,
 *        and which kernel evaluates which
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details kernel::op_code keeps every ir::pc::op_code at its value, so an IR op code converts to it unchanged (of),
 * and adds, from ir::pc::first_extension_op_code, the execution operations that docs/IR-PC-REQUIREMENTS.md §4 keeps
 * out of the IR: nodes sampled from their law (`sample`), constants (`ones`, `zeros`), NULL, and the counting gates'
 * specializations (`atleast`, `atmost`, `exactly`, `majority`, `minority`). Its values are pinned as the IR's are: an
 * extension takes the next free value and is never renumbered.
 *
 * Two kernels evaluate gates:
 *
 * - kernel::op (gate.h), the Boolean gates: AND, OR and XOR fold their inputs, NAND, NOR and XNOR complement the fold,
 *   NOT complements its one input and NULL passes it through;
 * - kernel::cardinality (cardinality.h), the counting gates: true where the number of their true inputs lies in their
 *   bounds. The op code names the gate; its bounds alone say what it computes.
 *
 * No gate kernel computes `sample`, `ones` or `zeros`. `majority` and `minority` have no kernel yet (has_kernel): the
 * kernel builder refuses their gates.
 *
 * These names are not those of ir::pc's own predicates (ir::pc::is_boolean_gate), so that an unqualified call never
 * means both.
 */

#pragma once

#include "ir/pc/op_code.h"

#include <cstdint>
#include <utility>

namespace fuzzybits::mc::kernel {

/// What computes a node: ir::pc::op_code at its own values, then the execution operations.
enum class op_code : std::uint8_t {
    // ir::pc::op_code
    basic_event = 0,
    constant = 1,
    house_event = 2,
    and_gate = 3,
    or_gate = 4,
    xor_gate = 5,
    not_gate = 6,
    nand_gate = 7,
    nor_gate = 8,
    xnor_gate = 9,
    cardinality = 10,
    // Extensions, from ir::pc::first_extension_op_code
    sample = 64,    ///< sampled from its law
    ones = 65,      ///< true in every trial
    zeros = 66,     ///< true in none
    null_gate = 67, ///< its one input
    atleast = 68,   ///< at least k of n inputs true
    atmost = 69,    ///< at most k of n
    exactly = 70,   ///< exactly k of n
    majority = 71,  ///< more than half: at least floor(n / 2) + 1 of n
    minority = 72,  ///< the complement of majority: at most floor(n / 2) of n
};

/// The kernel op code of IR op code `code`: the same value.
[[nodiscard]] constexpr op_code of(const ir::pc::op_code code) noexcept { return static_cast<op_code>(std::to_underlying(code)); }

/// Whether `op` is an IR op code, not an extension.
[[nodiscard]] constexpr bool is_ir(const op_code op) noexcept { return std::to_underlying(op) < ir::pc::first_extension_op_code; }

static_assert(of(ir::pc::op_code::basic_event) == op_code::basic_event && of(ir::pc::op_code::constant) == op_code::constant &&
                  of(ir::pc::op_code::house_event) == op_code::house_event && of(ir::pc::op_code::and_gate) == op_code::and_gate &&
                  of(ir::pc::op_code::or_gate) == op_code::or_gate && of(ir::pc::op_code::xor_gate) == op_code::xor_gate &&
                  of(ir::pc::op_code::not_gate) == op_code::not_gate && of(ir::pc::op_code::nand_gate) == op_code::nand_gate &&
                  of(ir::pc::op_code::nor_gate) == op_code::nor_gate && of(ir::pc::op_code::xnor_gate) == op_code::xnor_gate &&
                  of(ir::pc::op_code::cardinality) == op_code::cardinality,
              "kernel::op_code keeps every ir::pc::op_code at its value");
static_assert(std::to_underlying(op_code::sample) == ir::pc::first_extension_op_code && !is_ir(op_code::sample) && is_ir(op_code::cardinality),
              "the extensions start where ir::pc's codes end");

/// Whether kernel::op evaluates gates of op code `op`: the Boolean gates, and NULL.
[[nodiscard]] constexpr bool is_boolean(const op_code op) noexcept {
    return op == op_code::and_gate || op == op_code::or_gate || op == op_code::xor_gate || op == op_code::not_gate || op == op_code::nand_gate ||
           op == op_code::nor_gate || op == op_code::xnor_gate || op == op_code::null_gate;
}

/// Whether kernel::cardinality evaluates gates of op code `op`: the counting gates.
[[nodiscard]] constexpr bool is_counting(const op_code op) noexcept {
    return op == op_code::cardinality || op == op_code::atleast || op == op_code::atmost || op == op_code::exactly || op == op_code::majority ||
           op == op_code::minority;
}

/// Whether a gate kernel evaluates gates of op code `op`.
[[nodiscard]] constexpr bool is_gate(const op_code op) noexcept { return is_boolean(op) || is_counting(op); }

/// Whether a gate of op code `op` reads exactly one input: NOT and NULL.
[[nodiscard]] constexpr bool is_unary(const op_code op) noexcept { return op == op_code::not_gate || op == op_code::null_gate; }

/// Whether a kernel computes gates of op code `op` today. kernel::cardinality computes every counting gate from its
/// bounds, but nothing builds `majority` or `minority` gates yet, so they have a stub: the kernel builder refuses them
/// before sampling (builder::not_implemented).
// TODO:: kernels for majority and minority gates: kernel::cardinality over [floor(n / 2) + 1, n] and [0, floor(n / 2)],
// the bounds the layer already holds them to (builder::layer::add_gate).
[[nodiscard]] constexpr bool has_kernel(const op_code op) noexcept { return op != op_code::majority && op != op_code::minority; }

} // namespace fuzzybits::mc::kernel
