#pragma once

#include "ir/pc/hull/detail/residual.h"

#include <cstdint>
#include <span>
#include <vector>

namespace fuzzybits::ir::pc::hull::detail {

// Conflict-driven clause learning (DPLL with unit propagation on two watched
// literals, first-UIP learning and non-chronological backtracking, VSIDS
// decisions with phase saving, Luby restarts). Literals are
// variable << 1 | negated. Complete and deterministic; there is no budget.
class sat_solver final {
  public:
    void reset(std::uint32_t variables);
    [[nodiscard]] auto add_variable() -> std::uint32_t;
    // Clauses may only be added before `solve`.
    void add_clause(std::span<std::uint32_t const> literals);
    [[nodiscard]] auto solve() -> bool;

  private:
    struct clause final {
        std::uint32_t begin;
        std::uint32_t size;
    };

    struct watcher final {
        std::uint32_t clause;
        std::uint32_t blocker;
    };

    [[nodiscard]] auto value(std::uint32_t literal) const noexcept -> std::uint8_t;
    [[nodiscard]] auto decision_level() const noexcept -> std::uint32_t;
    void assign(std::uint32_t literal, std::uint32_t reason);
    void attach(std::uint32_t index);
    [[nodiscard]] auto propagate() -> std::uint32_t;
    [[nodiscard]] auto analyze(std::uint32_t conflict) -> std::uint32_t;
    void backtrack(std::uint32_t level);
    void bump(std::uint32_t variable);
    void heap_insert(std::uint32_t variable);
    void heap_up(std::uint32_t position);
    void heap_down(std::uint32_t position);
    [[nodiscard]] auto heap_pop() -> std::uint32_t;

    std::vector<std::uint32_t> arena_;
    std::vector<clause> clauses_;
    std::vector<std::vector<watcher>> watches_;
    std::vector<std::uint8_t> assigns_;
    std::vector<std::uint8_t> phase_;
    std::vector<std::uint8_t> seen_;
    std::vector<std::uint32_t> level_;
    std::vector<std::uint32_t> reason_;
    std::vector<std::uint32_t> trail_;
    std::vector<std::uint32_t> trail_limits_;
    std::vector<double> activity_;
    std::vector<std::uint32_t> heap_;
    std::vector<std::uint32_t> heap_position_;
    std::vector<std::uint32_t> learnt_;
    std::vector<std::uint32_t> buffer_;
    std::uint32_t head_ = 0;
    double increment_ = 1.0;
    bool inconsistent_ = false;
};

// Tseitin encoding of the residual (sequential counters for cardinality
// gates, chains for XOR) with the root asserted, then `solve`.
[[nodiscard]] auto search_satisfiable(residual const &circuit, sat_solver &solver) -> bool;

} // namespace fuzzybits::ir::pc::hull::detail
