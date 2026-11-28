#pragma once

#include <cstdint>

namespace fuzzybits::ir::pc {

// R1 intentionally admits only probabilistic Boolean semantics. Solver scratch
// operations, pcfb tensor operators, and expression opcodes belong elsewhere.
//
// The values are pinned, as the pcfb schema pins its OpCode values: a value is never renumbered or reused, and a new
// code takes the next free value below first_extension_op_code.
enum class op_code : std::uint8_t {
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
};

// The first value of op_code's type that ir::pc leaves to extensions outside the IR (mc::kernel::op_code adds the
// execution operations of docs/IR-PC-REQUIREMENTS.md §4 from here).
inline constexpr std::uint8_t first_extension_op_code = 64;

// Leaf classification is semantic, not layout-driven: leaves have no Boolean
// operands even if an importer keeps domain identity such as house events.
[[nodiscard]] constexpr auto is_leaf(const op_code code) noexcept -> bool {
    return code == op_code::basic_event || code == op_code::constant || code == op_code::house_event;
}

[[nodiscard]] constexpr auto is_boolean_gate(const op_code code) noexcept -> bool {
    return code == op_code::and_gate || code == op_code::or_gate || code == op_code::xor_gate || code == op_code::not_gate || code == op_code::nand_gate ||
           code == op_code::nor_gate || code == op_code::xnor_gate;
}

[[nodiscard]] constexpr auto accepts_operands(const op_code code) noexcept -> bool { return is_boolean_gate(code) || code == op_code::cardinality; }

} // namespace fuzzybits::ir::pc
