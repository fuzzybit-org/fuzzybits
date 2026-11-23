#pragma once

#include "ir/pra/id.h"

namespace fuzzybits::ir::pra {

// A named group of gates. `gates` slices model::fault_tree_gates() (every gate
// whose owner is this tree, in id order); `top_gates` slices
// model::fault_tree_top_gates(). Top gates are owned by the tree.
struct fault_tree final {
    element_id element;
    table_range<gate_id> gates;
    table_range<gate_id> top_gates;
};

} // namespace fuzzybits::ir::pra
