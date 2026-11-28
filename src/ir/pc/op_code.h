#pragma once

#include <cstdint>

namespace fuzzybits::ir::pc {

// R1 intentionally admits only probabilistic Boolean semantics. Solver scratch
// operations, pcfb tensor operators, and expression opcodes belong elsewhere.
enum class op_code : std::uint8_t {
    basic_event,
    constant,
    sample,
    ones,
    zeros,
    house_event,
    and_gate,
    or_gate,
    xor_gate,
    not_gate,
    null_gate,
    nand_gate,
    nor_gate,
    xnor_gate,
    cardinality,
    atleast,
    atmost,
    exactly,
};

// Leaf classification is semantic, not layout-driven: leaves have no Boolean
// operands even if an importer keeps domain identity such as house events.
[[nodiscard]] constexpr auto is_leaf(const op_code code) noexcept -> bool {
    return code == op_code::basic_event || code == op_code::constant || code == op_code::house_event;
}

[[nodiscard]] constexpr auto is_boolean_gate(const op_code code) noexcept -> bool {
    switch (code) {
    case op_code::constant:
    case op_code::sample:
    case op_code::ones:
    case op_code::zeros:
    case op_code::basic_event:
    case op_code::house_event:
        return false;
    default:
        return true;
    }
}

[[nodiscard]] constexpr auto accepts_operands(const op_code code) noexcept -> bool { return is_boolean_gate(code) || code == op_code::cardinality; }

} // namespace fuzzybits::ir::pc
