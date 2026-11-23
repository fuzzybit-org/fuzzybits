#pragma once

#include "ir/pra/id.h"

#include <cstdint>

namespace fuzzybits::ir::pra {

// The subject's kind follows the target kind: fault_tree_top -> fault tree
// (which must have exactly one top gate), gate -> gate, basic_event -> basic
// event. Sequence and end-state targets are reserved; validation rejects them
// until event trees exist.
enum class domain_target_kind : std::uint8_t {
    fault_tree_top,
    gate,
    basic_event,
    sequence,
    end_state,
};

// Lowering maps these to pc::observable_role.
enum class target_role : std::uint8_t {
    primary,
    watch,
};

// A domain query. Result kinds and negation semantics belong to the
// quantification config, not to the target.
struct domain_target final {
    element_id element;
    domain_target_kind kind = domain_target_kind::gate;
    target_role role = target_role::primary;
    domain_ref subject;
};

} // namespace fuzzybits::ir::pra
