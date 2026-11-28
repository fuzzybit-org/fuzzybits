#include "ir/pc/hull/detail/bdd.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace fuzzybits::ir::pc::hull::detail {

namespace {

constexpr std::uint32_t terminal_variable = std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t cache_size = std::size_t{1} << 15;
constexpr std::uint32_t operation_and = 1;
constexpr std::uint32_t operation_xor = 2;

[[nodiscard]] auto mix(std::uint64_t value) noexcept -> std::uint64_t {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

[[nodiscard]] auto hash3(const std::uint32_t a, const std::uint32_t b, const std::uint32_t c) noexcept -> std::uint64_t {
    return mix((std::uint64_t{a} << 32) ^ mix((std::uint64_t{b} << 32) | c));
}

} // namespace

void bdd_manager::reset(const std::uint32_t node_budget) {
    budget_ = node_budget;
    exceeded_ = false;
    nodes_.clear();
    nodes_.push_back(node{.variable = terminal_variable, .low = 0, .high = 0});
    const auto table = std::bit_ceil(std::max<std::size_t>(1024, 2 * (std::size_t{node_budget} + 1)));
    if (unique_.size() != table) {
        unique_.assign(table, 0);
    } else {
        std::ranges::fill(unique_, 0U);
    }
    if (cache_.size() != cache_size) {
        cache_.assign(cache_size, cache_entry{});
        generation_ = 0;
    }
    if (++generation_ == 0) {
        std::ranges::fill(cache_, cache_entry{});
        generation_ = 1;
    }
}

auto bdd_manager::make_node(const std::uint32_t variable, std::uint32_t low, std::uint32_t high) -> std::uint32_t {
    if (low == high) {
        return low;
    }
    const auto complement = high & 1U;
    low ^= complement;
    high ^= complement;
    const auto mask = unique_.size() - 1;
    for (auto slot = hash3(variable, low, high) & mask;; slot = (slot + 1) & mask) {
        const auto index = unique_[slot];
        if (index == 0) {
            if (nodes_.size() > budget_) {
                exceeded_ = true;
                return edge_false;
            }
            const auto created = static_cast<std::uint32_t>(nodes_.size());
            nodes_.push_back(node{.variable = variable, .low = low, .high = high});
            unique_[slot] = created;
            return (created << 1) | complement;
        }
        auto const &existing = nodes_[index];
        if (existing.variable == variable && existing.low == low && existing.high == high) {
            return (index << 1) | complement;
        }
    }
}

auto bdd_manager::top(const std::uint32_t edge) const noexcept -> std::uint32_t { return nodes_[edge >> 1].variable; }

auto bdd_manager::low(const std::uint32_t edge) const noexcept -> std::uint32_t { return nodes_[edge >> 1].low ^ (edge & 1U); }

auto bdd_manager::high(const std::uint32_t edge) const noexcept -> std::uint32_t { return nodes_[edge >> 1].high ^ (edge & 1U); }

auto bdd_manager::lookup(const std::uint32_t operation, const std::uint32_t left, const std::uint32_t right, std::uint32_t &result) const noexcept -> bool {
    auto const &entry = cache_[hash3(operation, left, right) & (cache_.size() - 1)];
    if (entry.generation == generation_ && entry.operation == operation && entry.left == left && entry.right == right) {
        result = entry.result;
        return true;
    }
    return false;
}

void bdd_manager::store(const std::uint32_t operation, const std::uint32_t left, const std::uint32_t right, const std::uint32_t result) noexcept {
    cache_[hash3(operation, left, right) & (cache_.size() - 1)] =
        cache_entry{.generation = generation_, .operation = operation, .left = left, .right = right, .result = result};
}

auto bdd_manager::conjunction(std::uint32_t left, std::uint32_t right) -> std::uint32_t {
    if (exceeded_ || left == edge_false || right == edge_false || left == (right ^ 1U)) {
        return edge_false;
    }
    if (left == edge_true) {
        return right;
    }
    if (right == edge_true || left == right) {
        return left;
    }
    if (left > right) {
        std::swap(left, right);
    }
    std::uint32_t result = 0;
    if (lookup(operation_and, left, right, result)) {
        return result;
    }
    const auto variable = std::min(top(left), top(right));
    const auto left_low = top(left) == variable ? low(left) : left;
    const auto left_high = top(left) == variable ? high(left) : left;
    const auto right_low = top(right) == variable ? low(right) : right;
    const auto right_high = top(right) == variable ? high(right) : right;
    const auto low_result = conjunction(left_low, right_low);
    const auto high_result = conjunction(left_high, right_high);
    result = make_node(variable, low_result, high_result);
    store(operation_and, left, right, result);
    return result;
}

auto bdd_manager::exclusive_or(std::uint32_t left, std::uint32_t right) -> std::uint32_t {
    if (exceeded_) {
        return edge_false;
    }
    const auto complement = (left ^ right) & 1U;
    left &= ~1U;
    right &= ~1U;
    if (left == edge_true) {
        return right ^ 1U ^ complement;
    }
    if (right == edge_true) {
        return left ^ 1U ^ complement;
    }
    if (left == right) {
        return edge_false ^ complement;
    }
    if (left > right) {
        std::swap(left, right);
    }
    std::uint32_t result = 0;
    if (lookup(operation_xor, left, right, result)) {
        return result ^ complement;
    }
    const auto variable = std::min(top(left), top(right));
    const auto left_low = top(left) == variable ? low(left) : left;
    const auto left_high = top(left) == variable ? high(left) : left;
    const auto right_low = top(right) == variable ? low(right) : right;
    const auto right_high = top(right) == variable ? high(right) : right;
    const auto low_result = exclusive_or(left_low, right_low);
    const auto high_result = exclusive_or(left_high, right_high);
    result = make_node(variable, low_result, high_result);
    store(operation_xor, left, right, result);
    return result ^ complement;
}

auto bdd_manager::disjunction(const std::uint32_t left, const std::uint32_t right) -> std::uint32_t { return conjunction(left ^ 1U, right ^ 1U) ^ 1U; }

auto bdd_manager::satisfiable(residual const &circuit, const std::uint32_t node_budget) -> std::optional<bool> {
    reset(node_budget);
    edges_.resize(circuit.nodes.size());
    for (std::size_t index = 0; index < circuit.nodes.size(); ++index) {
        auto const &gate = circuit.nodes[index];
        const auto input = [&](const std::uint32_t literal) { return edges_[literal >> 1] ^ (literal & 1U); };
        std::uint32_t edge = edge_false;
        switch (gate.kind) {
        case residual_kind::variable:
            edge = make_node(gate.variable, edge_false, edge_true);
            break;
        case residual_kind::and_gate:
            edge = edge_true;
            for (const auto literal : circuit.inputs(gate)) {
                edge = conjunction(edge, input(literal));
            }
            break;
        case residual_kind::or_gate:
            for (const auto literal : circuit.inputs(gate)) {
                edge = disjunction(edge, input(literal));
            }
            break;
        case residual_kind::xor_gate:
            for (const auto literal : circuit.inputs(gate)) {
                edge = exclusive_or(edge, input(literal));
            }
            break;
        case residual_kind::cardinality: {
            // counts_[j]: at least j of the inputs seen so far are true.
            const auto limit = gate.high + 1;
            counts_.assign(std::size_t{limit} + 1, edge_false);
            counts_[0] = edge_true;
            std::uint32_t seen = 0;
            for (const auto literal : circuit.inputs(gate)) {
                const auto value = input(literal);
                ++seen;
                for (auto count = std::min(seen, limit); count > 0; --count) {
                    counts_[count] = disjunction(counts_[count], conjunction(counts_[count - 1], value));
                }
            }
            edge = conjunction(counts_[gate.low], counts_[limit] ^ 1U);
            break;
        }
        }
        if (exceeded_) {
            return std::nullopt;
        }
        edges_[index] = edge;
    }
    return (edges_[circuit.root >> 1] ^ (circuit.root & 1U)) != edge_false;
}

} // namespace fuzzybits::ir::pc::hull::detail
