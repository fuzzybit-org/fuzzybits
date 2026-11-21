#pragma once

#include "ir/pc/hull/detail/residual.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace fuzzybits::ir::pc::hull::detail {

    // Reduced ordered BDD with complement edges (Bryant 1986; Brace, Rudell and
    // Bryant 1990) under a node budget. An edge is node << 1 | complement;
    // node 0 is the terminal, so edge 0 is TRUE and edge 1 is FALSE. The high
    // edge of a stored node is never complemented, which makes the
    // representation canonical. Variable order is the residual's variable
    // ordinal (depth-first discovery from the root).
    class bdd_manager final {
    public:
        // Whether the residual is satisfiable, or nullopt when building its BDD
        // would exceed `node_budget` nodes. Recursion depth is bounded by the
        // number of variables.
        [[nodiscard]] auto satisfiable(residual const &circuit, std::uint32_t node_budget) -> std::optional<bool>;

    private:
        struct node final {
            std::uint32_t variable;
            std::uint32_t low;
            std::uint32_t high;
        };

        struct cache_entry final {
            std::uint32_t generation = 0;
            std::uint32_t operation = 0;
            std::uint32_t left = 0;
            std::uint32_t right = 0;
            std::uint32_t result = 0;
        };

        static constexpr std::uint32_t edge_true = 0;
        static constexpr std::uint32_t edge_false = 1;

        void reset(std::uint32_t node_budget);
        [[nodiscard]] auto make_node(std::uint32_t variable, std::uint32_t low, std::uint32_t high) -> std::uint32_t;
        [[nodiscard]] auto top(std::uint32_t edge) const noexcept -> std::uint32_t;
        [[nodiscard]] auto low(std::uint32_t edge) const noexcept -> std::uint32_t;
        [[nodiscard]] auto high(std::uint32_t edge) const noexcept -> std::uint32_t;
        [[nodiscard]] auto conjunction(std::uint32_t left, std::uint32_t right) -> std::uint32_t;
        [[nodiscard]] auto exclusive_or(std::uint32_t left, std::uint32_t right) -> std::uint32_t;
        [[nodiscard]] auto disjunction(std::uint32_t left, std::uint32_t right) -> std::uint32_t;
        [[nodiscard]] auto lookup(std::uint32_t operation, std::uint32_t left, std::uint32_t right,
                                  std::uint32_t &result) const noexcept -> bool;
        void store(std::uint32_t operation, std::uint32_t left, std::uint32_t right, std::uint32_t result) noexcept;

        std::vector<node> nodes_;
        std::vector<std::uint32_t> unique_;
        std::vector<cache_entry> cache_;
        std::vector<std::uint32_t> edges_;
        std::vector<std::uint32_t> counts_;
        std::uint32_t generation_ = 0;
        std::uint32_t budget_ = 0;
        bool exceeded_ = false;
    };

} // namespace fuzzybits::ir::pc::hull::detail
