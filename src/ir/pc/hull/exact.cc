#include "ir/pc/hull/exact.h"

#include "api/diagnostics.h"
#include "ir/pc/hull/detail/bits.h"
#include "ir/pc/op_code.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pc::hull {

    namespace {

        constexpr std::uint32_t kNoAxis = std::numeric_limits<std::uint32_t>::max();

        // The target's cone in topological order, with one axis per event that
        // has a probability above 0.
        struct exact_cone final {
            std::vector<op_code> codes;
            std::vector<std::uint32_t> low;
            std::vector<std::uint32_t> high;
            std::vector<std::uint8_t> value; // constants and house events
            std::vector<std::uint32_t> input_offsets;
            std::vector<std::uint32_t> inputs; // literals: (position << 1) | complement
            std::vector<std::uint32_t> axis;
            std::vector<double> probability;
            std::uint32_t root = 0;

            [[nodiscard]] auto axis_count() const noexcept -> std::uint32_t {
                return static_cast<std::uint32_t>(probability.size());
            }
        };

        [[nodiscard]] auto literal_probability(circuit const &model, const std::uint32_t node)
            -> std::optional<double> {
            const auto source = model.probability_source_refs()[node];
            if (!source.valid() || source.raw() >= model.probability_sources().size()) {
                return std::nullopt;
            }
            auto const &probability = model.probability_sources()[source.raw()];
            if (probability.kind != probability_source_kind::literal) {
                return std::nullopt;
            }
            return probability.literal;
        }

        // Post-order over the operand relation, restricted to the target's cone.
        [[nodiscard]] auto build_cone(circuit const &model, const bool_node_id target) -> api::result<exact_cone> {
            using cone_result = api::result<exact_cone>;
            const auto node_count = model.node_count();
            const auto op_codes = model.op_codes();
            std::vector<std::uint8_t> reached(node_count, 0);
            std::vector<std::uint32_t> position_of(node_count, kNoAxis);
            std::vector<std::pair<std::uint32_t, std::uint32_t>> stack;
            std::vector<std::uint32_t> order;
            stack.emplace_back(target.raw(), 0);
            reached[target.raw()] = 1;
            while (!stack.empty()) {
                const auto node = stack.back().first;
                const auto operands = model.operands(bool_node_id(node));
                auto next = stack.back().second;
                bool descended = false;
                while (next < operands.size()) {
                    const auto operand = operands[next++].source.raw();
                    if (reached[operand] == 0) {
                        reached[operand] = 1;
                        stack.back().second = next;
                        stack.emplace_back(operand, 0);
                        descended = true;
                        break;
                    }
                }
                if (descended) {
                    continue;
                }
                stack.pop_back();
                order.push_back(node);
            }

            exact_cone cone;
            const auto size = static_cast<std::uint32_t>(order.size());
            for (std::uint32_t position = 0; position < size; ++position) {
                position_of[order[position]] = position;
            }
            cone.codes.resize(size);
            cone.low.assign(size, 0);
            cone.high.assign(size, 0);
            cone.value.assign(size, 0);
            cone.axis.assign(size, kNoAxis);
            cone.input_offsets.assign(std::size_t{size} + 1, 0);
            for (std::uint32_t position = 0; position < size; ++position) {
                const auto node = order[position];
                const auto code = op_codes[node];
                cone.codes[position] = code;
                if (code == op_code::constant || code == op_code::house_event) {
                    cone.value[position] = model.constant_values()[node] != 0 ? 1 : 0;
                } else if (code == op_code::basic_event) {
                    const auto probability = literal_probability(model, node);
                    if (!probability) {
                        return cone_result::fail(api::make_error(
                            "pc.hull.exact-probability", "basic event " + std::to_string(node) +
                                                             " in the target's cone has no literal probability"));
                    }
                    // A probability-0 event is FALSE in every hull world (x_e = 0
                    // forces y_e = 0); a probability-1 event stays a variable with
                    // weight 1 (x_e = 1 still allows y_e = 0).
                    if (*probability > 0.0) {
                        cone.axis[position] = cone.axis_count();
                        cone.probability.push_back(*probability);
                    }
                } else if (code == op_code::cardinality) {
                    cone.low[position] = model.cardinality_low()[node];
                    cone.high[position] = model.cardinality_high()[node];
                }
                for (auto const &edge : model.operands(bool_node_id(node))) {
                    cone.inputs.push_back((position_of[edge.source.raw()] << 1) | (edge.complement ? 1U : 0U));
                }
                cone.input_offsets[position + 1] = static_cast<std::uint32_t>(cone.inputs.size());
            }
            cone.root = position_of[target.raw()];
            return cone_result::ok(std::move(cone));
        }

        [[nodiscard]] auto inputs_of(exact_cone const &cone, const std::uint32_t position)
            -> std::span<std::uint32_t const> {
            return std::span<std::uint32_t const>{cone.inputs}.subspan(
                cone.input_offsets[position], cone.input_offsets[position + 1] - cone.input_offsets[position]);
        }

        // Word-parallel value of every cone position for the 64 assignments whose
        // axes above 5 are `chunk` and whose five-toe axes run over the word's bits.
        void evaluate_word(exact_cone const &cone, const std::uint64_t chunk, std::vector<std::uint64_t> &values) {
            static constexpr std::uint64_t patterns[6] = {
                0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
                0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL, 0xFFFFFFFF00000000ULL,
            };
            for (std::size_t position = 0; position < cone.codes.size(); ++position) {
                const auto inputs = inputs_of(cone, static_cast<std::uint32_t>(position));
                const auto input = [&](const std::uint32_t literal) {
                    return values[literal >> 1] ^ ((literal & 1U) != 0 ? detail::all_ones : 0);
                };
                std::uint64_t word = 0;
                switch (cone.codes[position]) {
                case op_code::basic_event: {
                    const auto axis = cone.axis[position];
                    word = axis == kNoAxis                ? 0
                           : axis < 6                     ? patterns[axis]
                           : ((chunk >> (axis - 6)) & 1U) != 0 ? detail::all_ones
                                                               : 0;
                    break;
                }
                case op_code::constant:
                case op_code::house_event:
                    word = cone.value[position] != 0 ? detail::all_ones : 0;
                    break;
                case op_code::and_gate:
                    word = detail::all_ones;
                    for (const auto literal : inputs) {
                        word &= input(literal);
                    }
                    break;
                case op_code::nand_gate:
                    word = detail::all_ones;
                    for (const auto literal : inputs) {
                        word &= input(literal);
                    }
                    word = ~word;
                    break;
                case op_code::or_gate:
                    for (const auto literal : inputs) {
                        word |= input(literal);
                    }
                    break;
                case op_code::nor_gate:
                    for (const auto literal : inputs) {
                        word |= input(literal);
                    }
                    word = ~word;
                    break;
                case op_code::not_gate:
                    word = ~input(inputs.front());
                    break;
                case op_code::xor_gate:
                    for (const auto literal : inputs) {
                        word ^= input(literal);
                    }
                    break;
                case op_code::xnor_gate:
                    for (const auto literal : inputs) {
                        word ^= input(literal);
                    }
                    word = ~word;
                    break;
                case op_code::cardinality: {
                    detail::sliced_counter ones;
                    for (const auto literal : inputs) {
                        ones.add(input(literal));
                    }
                    word = ones.at_least(cone.low[position]) & ~ones.at_least(std::uint64_t{cone.high[position]} + 1);
                    break;
                }
                }
                values[position] = word;
            }
        }

        // T[x] |= T[x \ {i}] for every x containing axis i: the subset-OR (zeta)
        // transform, after which T[x] is TRUE iff some y <= x satisfies g.
        void subset_or(std::vector<std::uint64_t> &table, const std::uint32_t axes) {
            static constexpr std::uint64_t low_masks[6] = {
                0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
                0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL, 0xFFFFFFFF00000000ULL,
            };
            for (std::uint32_t axis = 0; axis < axes; ++axis) {
                if (axis < 6) {
                    const auto shift = std::uint64_t{1} << axis;
                    for (auto &word : table) {
                        word |= (word << shift) & low_masks[axis];
                    }
                    continue;
                }
                const auto stride = std::size_t{1} << (axis - 6);
                for (std::size_t word = 0; word < table.size(); ++word) {
                    if ((word & stride) != 0) {
                        table[word] |= table[word ^ stride];
                    }
                }
            }
        }

        // Weights of every assignment of one contiguous axis range.
        [[nodiscard]] auto assignment_weights(std::span<double const> probability) -> std::vector<double> {
            std::vector<double> weights(std::size_t{1} << probability.size(), 1.0);
            std::size_t size = 1;
            for (const auto p : probability) {
                for (std::size_t assignment = 0; assignment < size; ++assignment) {
                    weights[size + assignment] = weights[assignment] * p;
                    weights[assignment] *= 1.0 - p;
                }
                size *= 2;
            }
            return weights;
        }

        // Σ_x P(X = x) T[x], blockwise over the low axes so that the block
        // weights are a table rather than a product per assignment. Neumaier
        // summation: every term is non-negative, so this is accurate to a few ulp.
        [[nodiscard]] auto weighted_sum(std::vector<std::uint64_t> const &table, std::span<double const> probability)
            -> double {
            const auto axes = static_cast<std::uint32_t>(probability.size());
            const auto low_axes = std::min<std::uint32_t>(axes, 12);
            const auto block = std::size_t{1} << low_axes;
            const auto words_per_block = std::max<std::size_t>(1, block / 64);
            const auto block_count = std::max<std::size_t>(1, table.size() / words_per_block);
            const auto low_weight = assignment_weights(probability.first(low_axes));
            const auto high_weight = assignment_weights(probability.subspan(low_axes));

            double total = 0.0;
            double compensation = 0.0;
            for (std::size_t index = 0; index < block_count; ++index) {
                const auto scale = high_weight[index];
                if (scale == 0.0) {
                    continue;
                }
                double sum = 0.0;
                double sum_compensation = 0.0;
                for (std::size_t word = 0; word < words_per_block; ++word) {
                    auto bits = table[index * words_per_block + word];
                    const auto base = word * 64;
                    while (bits != 0) {
                        const auto bit = static_cast<std::size_t>(std::countr_zero(bits));
                        bits &= bits - 1;
                        const auto term = low_weight[base + bit];
                        const auto next = sum + term;
                        sum_compensation += std::abs(sum) >= std::abs(term) ? (sum - next) + term : (term - next) + sum;
                        sum = next;
                    }
                }
                sum += sum_compensation;
                const auto term = scale * sum;
                const auto next = total + term;
                compensation += std::abs(total) >= std::abs(term) ? (total - next) + term : (term - next) + total;
                total = next;
            }
            return total + compensation;
        }

    } // namespace

    auto exact_hull_probability(circuit const &model, const bool_node_id target) -> api::result<double> {
        using probability_result = api::result<double>;
        if (!model.valid()) {
            return probability_result::fail(api::make_error("pc.hull.invalid-circuit", "circuit is not sealed"));
        }
        if (!target.valid() || target.raw() >= model.node_count()) {
            return probability_result::fail(
                api::make_error("pc.hull.invalid-target", "target does not name a node of the circuit"));
        }
        auto made = build_cone(model, target);
        if (!made) {
            return probability_result::fail(made.diagnostics());
        }
        auto const &cone = made.value();
        const auto axes = cone.axis_count();
        if (axes > kMaxExactHullEvents) {
            return probability_result::fail(api::make_error(
                "pc.hull.exact-too-many-events",
                "the target's cone has " + std::to_string(axes) + " events with a probability above 0; the exact hull "
                "evaluator is limited to " + std::to_string(kMaxExactHullEvents)));
        }

        std::vector<std::uint64_t> table(std::size_t{1} << (axes > 6 ? axes - 6 : 0), 0);
        std::vector<std::uint64_t> values(cone.codes.size(), 0);
        const std::uint64_t chunks = axes <= 6 ? 1 : std::uint64_t{1} << (axes - 6);
        for (std::uint64_t chunk = 0; chunk < chunks; ++chunk) {
            evaluate_word(cone, chunk, values);
            table[chunk] = values[cone.root];
        }
        if (axes < 6) {
            // evaluate_word fills all 64 lanes; only the 2^axes assignments exist.
            table[0] &= (std::uint64_t{1} << (std::uint64_t{1} << axes)) - 1;
        }
        subset_or(table, axes);
        return probability_result::ok(weighted_sum(table, cone.probability));
    }

} // namespace fuzzybits::ir::pc::hull
