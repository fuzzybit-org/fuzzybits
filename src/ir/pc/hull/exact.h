#pragma once

#include "api/result.h"
#include "ir/pc/circuit.h"
#include "ir/pc/id.h"

#include <cstdint>

// Exact P(↑g) by the subset-OR (zeta) transform
// (docs/rfc/monotone-hull-queries.md §6.5).
namespace fuzzybits::ir::pc::hull {

    // Random events (0 < p) in the target's cone that the transform accepts.
    inline constexpr std::uint32_t kMaxExactHullEvents = 24;

    // Every event is `down`, so the value is ↑g of the circuit as written: the
    // plain monotone hull, ∃ y <= x : g(y). A probability-0 event becomes FALSE,
    // because x_e = 0 forces y_e = 0. A probability-1 event stays a variable with
    // weight 1: x_e = 1 still allows y_e = 0, so substituting TRUE would be wrong.
    [[nodiscard]] auto exact_hull_probability(circuit const &model, bool_node_id target) -> api::result<double>;

} // namespace fuzzybits::ir::pc::hull
