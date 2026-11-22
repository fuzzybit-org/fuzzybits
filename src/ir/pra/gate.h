#pragma once

#include "ir/pra/id.h"

#include <cstddef>
#include <cstdint>

namespace fuzzybits::ir::pra {

    // One connective per gate. With n operands and t of them true:
    //   and: t = n, or: t >= 1, not: the single operand is false, xor: t is odd,
    //   nand: not and, nor: not or, xnor: t is even, cardinality(low, high): low <= t <= high.
    // At-least-k is cardinality(k, n). The kinds lower 1:1 to pc op codes.
    enum class gate_kind : std::uint8_t {
        and_gate,
        or_gate,
        cardinality,
        not_gate,
        xor_gate,
        nand_gate,
        nor_gate,
        xnor_gate,
    };

    inline constexpr std::size_t gate_kind_count = 8;

    struct cardinality_bounds final {
        std::uint32_t low = 0;
        std::uint32_t high = 0;
    };

    // `event` is a basic event, a house event or a gate. `literal` marks a
    // complemented basic-event operand that the source defines as an atomic
    // literal; it never changes the Boolean meaning, which is the complement.
    struct gate_operand final {
        domain_ref event;
        bool complement = false;
        bool literal = false;
    };

    struct gate final {
        element_id element;
        gate_kind kind = gate_kind::and_gate;
        table_range<gate_operand> operands;
        // Zero unless `kind` is cardinality.
        cardinality_bounds bounds;
        // Invalid when the gate belongs to no fault tree.
        fault_tree_id fault_tree;
    };

} // namespace fuzzybits::ir::pra
