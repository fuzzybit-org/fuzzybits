#pragma once

#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/passes/passes.h"
#include "ir/pc/passes/transform.h"
#include "ir/pc/probability.h"
#include "ir/pc/target.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pc_passes_test {

    namespace pc = fuzzybits::ir::pc;
    namespace passes = fuzzybits::ir::pc::passes;

    using edge = pc::edge_ref;
    using word = std::uint64_t;

    constexpr word all_lanes = ~word{0};

    [[nodiscard]] inline auto diagnostics_text(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
        std::ostringstream output;
        for (auto const &diagnostic : diagnostics.items()) {
            output << diagnostic.code << ": " << diagnostic.message << '\n';
        }
        return output.str();
    }

    [[nodiscard]] inline auto has_code(fuzzybits::api::diagnostic_list const &diagnostics, std::string const &code) -> bool {
        return std::ranges::any_of(diagnostics.items(), [&](auto const &diagnostic) { return diagnostic.code == code; });
    }

    [[nodiscard]] inline auto seal(pc::builder &&builder) -> pc::circuit {
        auto sealed = std::move(builder).seal();
        INFO(diagnostics_text(sealed.diagnostics));
        REQUIRE(sealed.ok());
        return std::move(sealed.output);
    }

    [[nodiscard]] inline auto literal(pc::circuit const &model, const std::uint32_t node) -> double {
        return model.probability_sources()[model.probability_source_refs()[node].raw()].literal;
    }

    // Kahn order, independent of the passes' DFS.
    [[nodiscard]] inline auto topological_order(pc::circuit const &model) -> std::vector<std::uint32_t> {
        const auto count = model.node_count();
        std::vector<std::uint32_t> pending(count, 0);
        std::vector<std::vector<std::uint32_t>> users(count);
        for (std::uint32_t node = 0; node < count; ++node) {
            for (auto const &operand : model.operands(pc::bool_node_id(node))) {
                users[operand.source.raw()].push_back(node);
                ++pending[node];
            }
        }
        std::vector<std::uint32_t> order;
        order.reserve(count);
        for (std::uint32_t node = 0; node < count; ++node) {
            if (pending[node] == 0) {
                order.push_back(node);
            }
        }
        for (std::size_t cursor = 0; cursor < order.size(); ++cursor) {
            for (const auto user : users[order[cursor]]) {
                if (--pending[user] == 0) {
                    order.push_back(user);
                }
            }
        }
        REQUIRE(order.size() == count);
        return order;
    }

    // Evaluates every node on 64 worlds at once; lane i of an event word is the
    // event's value in world i.
    class evaluator final {
      public:
        explicit evaluator(pc::circuit const &model) : model_(model), order_(topological_order(model)), event_slot_(model.node_count(), 0) {
            for (std::uint32_t node = 0; node < model.node_count(); ++node) {
                if (model.op_codes()[node] == pc::op_code::basic_event) {
                    event_slot_[node] = static_cast<std::uint32_t>(events_.size());
                    events_.push_back(node);
                }
            }
        }

        [[nodiscard]] auto events() const noexcept -> std::span<std::uint32_t const> {
            return events_;
        }

        [[nodiscard]] auto slot(const std::uint32_t node) const noexcept -> std::uint32_t {
            return event_slot_[node];
        }

        void run(std::span<word const> event_words, std::vector<word> &values) const {
            values.resize(model_.node_count());
            for (const auto node : order_) {
                values[node] = value(node, event_words, values);
            }
        }

      private:
        [[nodiscard]] auto value(const std::uint32_t node, std::span<word const> event_words, std::vector<word> const &values) const
            -> word {
            const auto code = model_.op_codes()[node];
            if (code == pc::op_code::basic_event) {
                return event_words[event_slot_[node]];
            }
            if (code == pc::op_code::constant || code == pc::op_code::house_event) {
                return model_.constant_values()[node] != 0 ? all_lanes : 0;
            }
            const auto operands = model_.operands(pc::bool_node_id(node));
            const auto operand = [&](edge const &e) { return values[e.source.raw()] ^ (e.complement ? all_lanes : 0); };
            word conjunction = all_lanes;
            word disjunction = 0;
            word parity = 0;
            for (auto const &e : operands) {
                conjunction &= operand(e);
                disjunction |= operand(e);
                parity ^= operand(e);
            }
            switch (code) {
            case pc::op_code::and_gate:
                return conjunction;
            case pc::op_code::or_gate:
                return disjunction;
            case pc::op_code::xor_gate:
                return parity;
            case pc::op_code::not_gate:
            case pc::op_code::nor_gate:
                return ~disjunction;
            case pc::op_code::nand_gate:
                return ~conjunction;
            case pc::op_code::xnor_gate:
                return ~parity;
            case pc::op_code::cardinality: {
                std::array<std::uint32_t, 64> trues{};
                for (auto const &e : operands) {
                    const auto bits = operand(e);
                    for (std::uint32_t lane = 0; lane < 64; ++lane) {
                        trues[lane] += static_cast<std::uint32_t>((bits >> lane) & 1U);
                    }
                }
                const auto low = model_.cardinality_low()[node];
                const auto high = model_.cardinality_high()[node];
                word result = 0;
                for (std::uint32_t lane = 0; lane < 64; ++lane) {
                    result |= trues[lane] >= low && trues[lane] <= high ? word{1} << lane : 0;
                }
                return result;
            }
            default:
                return 0;
            }
        }

        pc::circuit const &model_;
        std::vector<std::uint32_t> order_;
        std::vector<std::uint32_t> events_;
        std::vector<std::uint32_t> event_slot_;
    };

    // Calls visit(base, valid_lanes, words) for blocks of 64 worlds over `count`
    // free variables; world base + lane assigns bit i to variable i.
    template <typename visitor_type> void for_each_block(const std::uint32_t count, visitor_type &&visit) {
        REQUIRE(count <= 24);
        const std::uint64_t worlds = std::uint64_t{1} << count;
        std::vector<word> words(count);
        for (std::uint64_t base = 0; base < worlds; base += 64) {
            for (std::uint32_t variable = 0; variable < count; ++variable) {
                if (variable < 6) {
                    word pattern = 0;
                    for (std::uint32_t lane = 0; lane < 64; ++lane) {
                        pattern |= ((lane >> variable) & 1U) != 0 ? word{1} << lane : 0;
                    }
                    words[variable] = pattern;
                } else {
                    words[variable] = ((base >> variable) & 1U) != 0 ? all_lanes : 0;
                }
            }
            const auto lanes = worlds - base >= 64 ? all_lanes : (word{1} << (worlds - base)) - 1;
            visit(base, lanes, std::span<word const>{words});
        }
    }

    struct neumaier final {
        double sum = 0.0;
        double compensation = 0.0;

        void add(const double value) {
            const auto total = sum + value;
            compensation += std::abs(sum) >= std::abs(value) ? (sum - total) + value : (value - total) + sum;
            sum = total;
        }

        [[nodiscard]] auto value() const -> double {
            return sum + compensation;
        }
    };

    [[nodiscard]] inline auto world_mass(pc::circuit const &model, std::span<std::uint32_t const> events, const std::uint64_t world)
        -> double {
        double mass = 1.0;
        for (std::size_t index = 0; index < events.size(); ++index) {
            const auto p = literal(model, events[index]);
            mass *= ((world >> index) & 1U) != 0 ? p : 1.0 - p;
        }
        return mass;
    }

    // Exact P(target) for every target; every basic-event node is an
    // independent variable (IR-PC semantics).
    [[nodiscard]] inline auto exact_probabilities(pc::circuit const &model) -> std::vector<double> {
        const evaluator eval(model);
        const auto events = eval.events();
        const auto targets = model.targets();
        std::vector<neumaier> sums(targets.size());
        std::vector<word> values;
        for_each_block(static_cast<std::uint32_t>(events.size()),
                       [&](const std::uint64_t base, const word lanes, std::span<word const> words) {
                           eval.run(words, values);
                           for (std::uint32_t lane = 0; lane < 64; ++lane) {
                               if (((lanes >> lane) & 1U) == 0) {
                                   continue;
                               }
                               const auto mass = world_mass(model, events, base + lane);
                               for (std::size_t target = 0; target < targets.size(); ++target) {
                                   if (((values[targets[target].node.raw()] >> lane) & 1U) != 0) {
                                       sums[target].add(mass);
                                   }
                               }
                           }
                       });
        std::vector<double> result;
        for (auto const &sum : sums) {
            result.push_back(sum.value());
        }
        return result;
    }

    // P(up g) with up g(x) = OR_{y <= x} g(y) over the circuit's basic events
    // (RFC §3): truth table, superset closure, then the weighted sum.
    [[nodiscard]] inline auto hull_probabilities(pc::circuit const &model) -> std::vector<double> {
        const evaluator eval(model);
        const auto events = eval.events();
        const auto count = static_cast<std::uint32_t>(events.size());
        const auto worlds = std::uint64_t{1} << count;
        const auto targets = model.targets();
        std::vector<std::vector<std::uint8_t>> tables(targets.size(), std::vector<std::uint8_t>(worlds, 0));
        std::vector<word> values;
        for_each_block(count, [&](const std::uint64_t base, const word lanes, std::span<word const> words) {
            eval.run(words, values);
            for (std::uint32_t lane = 0; lane < 64; ++lane) {
                if (((lanes >> lane) & 1U) != 0) {
                    for (std::size_t target = 0; target < targets.size(); ++target) {
                        tables[target][base + lane] = static_cast<std::uint8_t>((values[targets[target].node.raw()] >> lane) & 1U);
                    }
                }
            }
        });
        std::vector<double> result;
        for (auto &table : tables) {
            for (std::uint32_t bit = 0; bit < count; ++bit) {
                for (std::uint64_t world = 0; world < worlds; ++world) {
                    if (((world >> bit) & 1U) != 0) {
                        table[world] |= table[world ^ (std::uint64_t{1} << bit)];
                    }
                }
            }
            neumaier sum;
            for (std::uint64_t world = 0; world < worlds; ++world) {
                if (table[world] != 0) {
                    sum.add(world_mass(model, events, world));
                }
            }
            result.push_back(sum.value());
        }
        return result;
    }

    inline void require_close(std::vector<double> const &expected, std::vector<double> const &actual, const double tolerance = 1.0e-12) {
        REQUIRE(expected.size() == actual.size());
        for (std::size_t index = 0; index < expected.size(); ++index) {
            INFO("target " << index << " expected " << expected[index] << " actual " << actual[index]);
            REQUIRE(std::abs(expected[index] - actual[index]) <= tolerance);
        }
    }

    enum class worlds : std::uint8_t {
        // Every assignment of the input's basic events.
        all,
        // Literal p = 0 and p = 1 events fixed at their almost-sure value.
        almost_sure,
    };

    // Every node image and every target of `output` agrees with `input` on the
    // given worlds. Also checks the maps' shape and that every output basic event
    // is the image of exactly one input event.
    inline void require_images(pc::circuit const &input, pc::circuit const &output, std::span<passes::node_image const> node_map,
                               std::span<pc::target_id const> target_map, const worlds mode) {
        REQUIRE(node_map.size() == input.node_count());
        REQUIRE(target_map.size() == input.targets().size());
        const evaluator in(input);
        const evaluator out(output);

        std::vector<std::uint32_t> free;
        std::vector<word> fixed(in.events().size(), 0);
        std::vector<std::uint8_t> is_free(in.events().size(), 1);
        for (std::uint32_t slot = 0; slot < in.events().size(); ++slot) {
            const auto node = in.events()[slot];
            const auto p = literal(input, node);
            const bool literal_source =
                input.probability_sources()[input.probability_source_refs()[node].raw()].kind == pc::probability_source_kind::literal;
            if (mode == worlds::almost_sure && literal_source && (p == 0.0 || p == 1.0)) {
                is_free[slot] = 0;
                fixed[slot] = p == 1.0 ? all_lanes : 0;
            } else {
                free.push_back(slot);
            }
        }

        std::vector<std::int64_t> preimage(output.node_count(), -1);
        for (std::uint32_t slot = 0; slot < in.events().size(); ++slot) {
            auto const &image = node_map[in.events()[slot]];
            if (image.kind != passes::image_kind::reference) {
                continue;
            }
            REQUIRE(image.edge.source.raw() < output.node_count());
            if (output.op_codes()[image.edge.source.raw()] == pc::op_code::basic_event) {
                REQUIRE_FALSE(image.edge.complement);
                REQUIRE(preimage[image.edge.source.raw()] == -1);
                preimage[image.edge.source.raw()] = slot;
                REQUIRE(literal(output, image.edge.source.raw()) == literal(input, in.events()[slot]));
            } else {
                REQUIRE(is_free[slot] == 0);
            }
        }
        for (const auto node : out.events()) {
            REQUIRE(preimage[node] >= 0);
        }
        for (std::uint32_t target = 0; target < target_map.size(); ++target) {
            REQUIRE(target_map[target].valid());
            REQUIRE(output.targets()[target_map[target].raw()].role == input.targets()[target].role);
        }

        std::vector<word> input_words(in.events().size());
        std::vector<word> output_words(out.events().size());
        std::vector<word> input_values;
        std::vector<word> output_values;
        std::uint64_t mismatches = 0;
        std::uint32_t first_mismatch = 0;
        for_each_block(static_cast<std::uint32_t>(free.size()), [&](std::uint64_t, const word lanes, std::span<word const> words) {
            input_words = fixed;
            for (std::size_t index = 0; index < free.size(); ++index) {
                input_words[free[index]] = words[index];
            }
            for (const auto node : out.events()) {
                output_words[out.slot(node)] = input_words[static_cast<std::size_t>(preimage[node])];
            }
            in.run(input_words, input_values);
            out.run(output_words, output_values);
            for (std::uint32_t node = 0; node < input.node_count(); ++node) {
                auto const &image = node_map[node];
                word expected = 0;
                if (image.kind == passes::image_kind::constant) {
                    expected = image.constant_value ? all_lanes : 0;
                } else if (image.kind == passes::image_kind::reference) {
                    expected = output_values[image.edge.source.raw()] ^ (image.edge.complement ? all_lanes : 0);
                } else {
                    continue;
                }
                if (((expected ^ input_values[node]) & lanes) != 0) {
                    first_mismatch = mismatches++ == 0 ? node : first_mismatch;
                }
            }
            for (std::uint32_t target = 0; target < target_map.size(); ++target) {
                const auto in_node = input.targets()[target].node.raw();
                const auto out_node = output.targets()[target_map[target].raw()].node.raw();
                if (((input_values[in_node] ^ output_values[out_node]) & lanes) != 0) {
                    first_mismatch = mismatches++ == 0 ? in_node : first_mismatch;
                }
            }
        });
        INFO("first mismatching input node " << first_mismatch);
        REQUIRE(mismatches == 0);
    }

    inline void require_report(passes::transform_result const &result, pc::circuit const &input) {
        INFO(result.report.transform_id << '\n' << diagnostics_text(result.report.diagnostics));
        REQUIRE(result.ok());
        REQUIRE(result.report.nodes_before == input.node_count());
        REQUIRE(result.report.edges_before == input.edge_count());
        REQUIRE(result.report.nodes_after == result.output.node_count());
        REQUIRE(result.report.edges_after == result.output.edge_count());
        REQUIRE(result.report.edges_after <= result.report.edges_before);
        REQUIRE(result.output.targets().size() == input.targets().size());
        for (auto const &image : result.report.node_map) {
            if (image.kind == passes::image_kind::reference) {
                REQUIRE(image.edge.source.raw() < result.output.node_count());
            }
        }
        for (auto const &target : input.targets()) {
            const auto kind = result.report.node_map[target.node.raw()].kind;
            REQUIRE((kind == passes::image_kind::reference || kind == passes::image_kind::constant));
        }
    }

    struct random_options final {
        std::uint32_t max_events = 10;
        std::uint32_t max_gates = 40;
        double extreme_probability = 0.2;
    };

    // Random circuit over every op code: complements, duplicate and x / not x
    // operands, every cardinality form, constants, house events, p in {0, 1},
    // shared nodes with same-kind parents, non-topological ids, 1-3 targets.
    [[nodiscard]] inline auto random_circuit(std::mt19937_64 &rng, random_options const &options = {}) -> pc::circuit {
        struct logical final {
            pc::op_code code = pc::op_code::basic_event;
            std::vector<std::pair<std::uint32_t, bool>> operands;
            std::uint32_t low = 0;
            std::uint32_t high = 0;
            bool value = false;
            double probability = 0.0;
        };
        const auto uniform = [&](const std::uint32_t low, const std::uint32_t high) {
            return std::uniform_int_distribution<std::uint32_t>(low, high)(rng);
        };
        const auto chance = [&](const double p) { return std::bernoulli_distribution(p)(rng); };

        std::vector<logical> nodes;
        const auto event_count = uniform(1, options.max_events);
        for (std::uint32_t event = 0; event < event_count; ++event) {
            const auto draw = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
            double p = std::uniform_real_distribution<double>(0.05, 0.95)(rng);
            if (draw < options.extreme_probability / 2) {
                p = 0.0;
            } else if (draw < options.extreme_probability) {
                p = 1.0;
            } else if (draw < options.extreme_probability + 0.1) {
                p = 0.5;
            }
            nodes.push_back(logical{.probability = p});
        }
        const auto constants = uniform(0, 2);
        for (std::uint32_t index = 0; index < constants; ++index) {
            nodes.push_back(logical{.code = chance(0.5) ? pc::op_code::constant : pc::op_code::house_event, .value = chance(0.5)});
        }
        std::ranges::shuffle(nodes, rng);
        const auto leaves = static_cast<std::uint32_t>(nodes.size());

        constexpr std::array codes{
            pc::op_code::and_gate, pc::op_code::or_gate,   pc::op_code::and_gate, pc::op_code::or_gate,   pc::op_code::xor_gate,
            pc::op_code::not_gate, pc::op_code::nand_gate, pc::op_code::nor_gate, pc::op_code::xnor_gate, pc::op_code::cardinality,
        };
        const auto gates = uniform(1, options.max_gates);
        for (std::uint32_t gate = 0; gate < gates; ++gate) {
            const auto available = static_cast<std::uint32_t>(nodes.size());
            logical node{.code = codes[uniform(0, static_cast<std::uint32_t>(codes.size()) - 1)]};
            const auto fan_in = node.code == pc::op_code::not_gate ? 1U : uniform(1, 5);
            for (std::uint32_t operand = 0; operand < fan_in; ++operand) {
                std::uint32_t source = 0;
                if (available > leaves && chance(0.5)) {
                    source = available - 1 - uniform(0, std::min<std::uint32_t>(available - leaves, 6) - 1);
                } else {
                    source = uniform(0, available - 1);
                }
                node.operands.emplace_back(source, chance(0.25));
            }
            if (node.code != pc::op_code::not_gate && chance(0.25)) {
                auto const repeated = node.operands[uniform(0, static_cast<std::uint32_t>(node.operands.size()) - 1)];
                node.operands.emplace_back(repeated.first, repeated.second != chance(0.5));
            }
            if (node.code == pc::op_code::cardinality) {
                const auto n = static_cast<std::uint32_t>(node.operands.size());
                node.low = uniform(0, n);
                node.high = uniform(node.low, n);
            }
            nodes.push_back(std::move(node));
        }

        const auto count = static_cast<std::uint32_t>(nodes.size());
        std::vector<std::uint32_t> storage(count);
        std::iota(storage.begin(), storage.end(), 0U);
        if (chance(0.5)) {
            std::ranges::shuffle(storage, rng);
        }
        std::vector<std::uint32_t> logical_at(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            logical_at[storage[index]] = index;
        }

        pc::builder builder;
        std::vector<edge> operands;
        for (std::uint32_t id = 0; id < count; ++id) {
            auto const &node = nodes[logical_at[id]];
            operands.clear();
            for (auto const &[source, complement] : node.operands) {
                operands.push_back(edge{.source = pc::bool_node_id(storage[source]), .complement = complement});
            }
            switch (node.code) {
            case pc::op_code::basic_event:
                (void)builder.add_basic_event(builder.add_literal_probability(node.probability));
                break;
            case pc::op_code::constant:
                (void)builder.add_constant(node.value);
                break;
            case pc::op_code::house_event:
                (void)builder.add_house_event(node.value);
                break;
            case pc::op_code::cardinality:
                (void)builder.add_cardinality(node.low, node.high, std::span<edge const>{operands});
                break;
            default:
                (void)builder.add_gate(node.code, std::span<edge const>{operands});
                break;
            }
        }
        constexpr std::array roles{pc::observable_role::primary, pc::observable_role::watch, pc::observable_role::projection_anchor};
        const auto targets = uniform(1, 3);
        for (std::uint32_t target = 0; target < targets; ++target) {
            const auto logical_node = chance(0.8) ? uniform(leaves, count - 1) : uniform(0, count - 1);
            (void)builder.add_target(pc::bool_node_id(storage[logical_node]), roles[uniform(0, 2)]);
        }
        return seal(std::move(builder));
    }

} // namespace pc_passes_test
