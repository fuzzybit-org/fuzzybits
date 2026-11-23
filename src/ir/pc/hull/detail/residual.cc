#include "ir/pc/hull/detail/residual.h"

#include "ir/pc/hull/detail/bits.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fuzzybits::ir::pc::hull::detail {

void evaluate_kleene(residual const &circuit, std::span<std::uint8_t> values) {
    for (std::size_t index = 0; index < circuit.nodes.size(); ++index) {
        auto const &node = circuit.nodes[index];
        if (node.kind == residual_kind::variable) {
            continue;
        }
        std::uint32_t ones = 0;
        std::uint32_t unknown = 0;
        for (const auto literal : circuit.inputs(node)) {
            const auto value = values[literal >> 1];
            if (value == value_unknown) {
                ++unknown;
            } else {
                ones += value ^ (literal & 1U);
            }
        }
        std::uint8_t out = value_unknown;
        switch (node.kind) {
        case residual_kind::and_gate:
            out = ones == node.count ? value_true : ones + unknown < node.count ? value_false : value_unknown;
            break;
        case residual_kind::or_gate:
            out = ones > 0 ? value_true : unknown == 0 ? value_false : value_unknown;
            break;
        case residual_kind::xor_gate:
            out = unknown == 0 ? static_cast<std::uint8_t>(ones & 1U) : value_unknown;
            break;
        case residual_kind::cardinality:
            if (ones >= node.low && ones + unknown <= node.high) {
                out = value_true;
            } else if (ones > node.high || ones + unknown < node.low) {
                out = value_false;
            }
            break;
        case residual_kind::variable:
            break;
        }
        values[index] = out;
    }
}

auto evaluate(residual const &circuit, std::span<std::uint8_t const> assignment, std::vector<std::uint8_t> &scratch)
    -> bool {
    scratch.resize(circuit.nodes.size());
    for (std::size_t index = 0; index < circuit.nodes.size(); ++index) {
        auto const &node = circuit.nodes[index];
        if (node.kind == residual_kind::variable) {
            scratch[index] = assignment[node.variable];
            continue;
        }
        std::uint32_t ones = 0;
        for (const auto literal : circuit.inputs(node)) {
            ones += scratch[literal >> 1] ^ (literal & 1U);
        }
        switch (node.kind) {
        case residual_kind::and_gate:
            scratch[index] = ones == node.count ? 1 : 0;
            break;
        case residual_kind::or_gate:
            scratch[index] = ones > 0 ? 1 : 0;
            break;
        case residual_kind::xor_gate:
            scratch[index] = static_cast<std::uint8_t>(ones & 1U);
            break;
        case residual_kind::cardinality:
            scratch[index] = ones >= node.low && ones <= node.high ? 1 : 0;
            break;
        case residual_kind::variable:
            break;
        }
    }
    return (scratch[circuit.root >> 1] ^ (circuit.root & 1U)) != 0;
}

void polarities(residual const &circuit, std::vector<std::uint8_t> &out) {
    out.assign(circuit.nodes.size(), 0);
    out[circuit.root >> 1] = (circuit.root & 1U) != 0 ? polarity_negative : polarity_positive;
    for (auto index = circuit.nodes.size(); index-- > 0;) {
        auto const &node = circuit.nodes[index];
        auto polarity = out[index];
        if (polarity == 0 || node.kind == residual_kind::variable) {
            continue;
        }
        if (node.kind == residual_kind::xor_gate ||
            (node.kind == residual_kind::cardinality && node.high < node.count)) {
            polarity = polarity_positive | polarity_negative;
        }
        for (const auto literal : circuit.inputs(node)) {
            out[literal >> 1] |= (literal & 1U) != 0 ? swap_polarity(polarity) : polarity;
        }
    }
}

auto truth_table_satisfiable(residual const &circuit, std::vector<std::uint64_t> &scratch) -> bool {
    static constexpr std::uint64_t patterns[6] = {
        0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
        0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL, 0xFFFFFFFF00000000ULL,
    };
    const auto variables = circuit.variable_count();
    const std::uint64_t chunks = variables <= 6 ? 1 : std::uint64_t{1} << (variables - 6);
    scratch.resize(circuit.nodes.size());
    const auto root_flip = (circuit.root & 1U) != 0 ? all_ones : 0;
    for (std::uint64_t chunk = 0; chunk < chunks; ++chunk) {
        for (std::size_t index = 0; index < circuit.nodes.size(); ++index) {
            auto const &node = circuit.nodes[index];
            const auto input = [&](const std::uint32_t literal) {
                return scratch[literal >> 1] ^ ((literal & 1U) != 0 ? all_ones : 0);
            };
            std::uint64_t word = 0;
            switch (node.kind) {
            case residual_kind::variable:
                word = node.variable < 6 ? patterns[node.variable]
                                         : (((chunk >> (node.variable - 6)) & 1U) != 0 ? all_ones : 0);
                break;
            case residual_kind::and_gate:
                word = all_ones;
                for (const auto literal : circuit.inputs(node)) {
                    word &= input(literal);
                }
                break;
            case residual_kind::or_gate:
                for (const auto literal : circuit.inputs(node)) {
                    word |= input(literal);
                }
                break;
            case residual_kind::xor_gate:
                for (const auto literal : circuit.inputs(node)) {
                    word ^= input(literal);
                }
                break;
            case residual_kind::cardinality: {
                sliced_counter counter;
                for (const auto literal : circuit.inputs(node)) {
                    counter.add(input(literal));
                }
                word = counter.at_least(node.low) & ~counter.at_least(std::uint64_t{node.high} + 1);
                break;
            }
            }
            scratch[index] = word;
        }
        if ((scratch[circuit.root >> 1] ^ root_flip) != 0) {
            return true;
        }
    }
    return false;
}

} // namespace fuzzybits::ir::pc::hull::detail
