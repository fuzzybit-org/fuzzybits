#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace fuzzybits::ir::pc::hull::detail {

    inline constexpr std::uint8_t value_false = 0;
    inline constexpr std::uint8_t value_true = 1;
    inline constexpr std::uint8_t value_unknown = 2;

    inline constexpr std::uint8_t polarity_positive = 1;
    inline constexpr std::uint8_t polarity_negative = 2;

    [[nodiscard]] constexpr auto swap_polarity(const std::uint8_t polarity) noexcept -> std::uint8_t {
        return static_cast<std::uint8_t>(((polarity & polarity_positive) << 1) | ((polarity & polarity_negative) >> 1));
    }

    enum class residual_kind : std::uint8_t {
        variable,
        and_gate,
        or_gate,
        xor_gate,
        // low <= #true inputs <= high, with 1 <= low and high <= count.
        cardinality,
    };

    struct residual_node final {
        residual_kind kind = residual_kind::variable;
        std::uint32_t begin = 0;
        std::uint32_t count = 0;
        std::uint32_t low = 0;
        std::uint32_t high = 0;
        // Variables: ordinal among the residual's variables, and the cone event
        // it stands for.
        std::uint32_t variable = 0;
        std::uint32_t source = 0;
    };

    // The undetermined part of one lane's target cone, with determined inputs
    // folded into the gates. Nodes are in topological order; edges are
    // literals (node << 1 | complement). Gates have no output negation.
    struct residual final {
        std::vector<residual_node> nodes;
        std::vector<std::uint32_t> literals;
        std::vector<std::uint32_t> variables;
        std::uint32_t root = 0;

        void clear() noexcept {
            nodes.clear();
            literals.clear();
            variables.clear();
            root = 0;
        }

        [[nodiscard]] auto inputs(residual_node const &node) const noexcept -> std::span<std::uint32_t const> {
            return std::span<std::uint32_t const>{literals}.subspan(node.begin, node.count);
        }

        [[nodiscard]] auto variable_count() const noexcept -> std::uint32_t {
            return static_cast<std::uint32_t>(variables.size());
        }
    };

    // Three-valued (Kleene) evaluation. `values` must hold the variables'
    // values on entry; gate entries are overwritten.
    void evaluate_kleene(residual const &circuit, std::span<std::uint8_t> values);

    // Two-valued evaluation of the root under an assignment of the variables
    // (indexed by ordinal); `scratch` is resized as needed.
    [[nodiscard]] auto evaluate(residual const &circuit, std::span<std::uint8_t const> assignment,
                                std::vector<std::uint8_t> &scratch) -> bool;

    // Per node: which polarities reach it from the root.
    void polarities(residual const &circuit, std::vector<std::uint8_t> &out);

    // Exhaustive enumeration, 64 assignments per word, stopping at the first
    // satisfying assignment.
    [[nodiscard]] auto truth_table_satisfiable(residual const &circuit, std::vector<std::uint64_t> &scratch) -> bool;

} // namespace fuzzybits::ir::pc::hull::detail
