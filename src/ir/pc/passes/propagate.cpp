#include "ir/pc/passes/detail/rewrite.h"
#include "ir/pc/passes/passes.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pc::passes {

namespace {

// Per-gate operand bookkeeping keyed by output node id. Entries are valid
// only while stamp_ holds the current generation, so a gate costs
// O(fan-in) without clearing.
class operand_scratch final {
  public:
    void next_gate(const std::uint32_t node_count) {
        if (stamp_.size() < node_count) {
            stamp_.resize(node_count, 0);
            first_.resize(node_count, 0);
            second_.resize(node_count, 0);
        }
        generation_ += 2;
    }

    // Drops repeated operands in place; true if some x occurs with both
    // polarities.
    [[nodiscard]] auto dedupe(std::vector<edge_ref> &edges) -> bool {
        std::size_t kept = 0;
        for (auto const &edge : edges) {
            const auto node = edge.source.raw();
            const std::uint32_t polarity = edge.complement ? 2U : 1U;
            if (stamp_[node] != generation_) {
                stamp_[node] = generation_;
                first_[node] = polarity;
                edges[kept++] = edge;
            } else if ((first_[node] & polarity) == 0) {
                return true;
            }
        }
        edges.resize(kept);
        return false;
    }

    // Cancels operands pairwise under XOR: x ^ x = 0, not x = x ^ 1.
    // Returns the parity of the complements and whether anything cancelled;
    // on cancellation `edges` becomes the odd-multiplicity operands,
    // uncomplemented, in first-occurrence order.
    [[nodiscard]] auto cancel_parity(std::vector<edge_ref> &edges) -> std::pair<bool, bool> {
        bool parity = false;
        bool repeated = false;
        for (auto const &edge : edges) {
            const auto node = edge.source.raw();
            parity = parity != edge.complement;
            if (stamp_[node] != generation_) {
                stamp_[node] = generation_;
                first_[node] = 1;
            } else {
                first_[node] ^= 1U;
                repeated = true;
            }
        }
        if (!repeated) {
            return {parity, false};
        }
        std::size_t kept = 0;
        for (std::size_t index = 0; index < edges.size(); ++index) {
            const auto node = edges[index].source.raw();
            if (first_[node] == 1) {
                first_[node] = 2;
                edges[kept++] = edge_ref{.source = edges[index].source};
            }
        }
        edges.resize(kept);
        return {parity, true};
    }

    // Removes min(#x, #not x) occurrences of each polarity of every x;
    // each removed pair contributes exactly one true operand.
    [[nodiscard]] auto remove_complement_pairs(std::vector<edge_ref> &edges) -> std::uint32_t {
        for (auto const &edge : edges) {
            const auto node = edge.source.raw();
            if (stamp_[node] != generation_) {
                stamp_[node] = generation_;
                first_[node] = 0;
                second_[node] = 0;
            }
            ++(edge.complement ? second_[node] : first_[node]);
        }
        std::uint32_t pairs = 0;
        std::size_t kept = 0;
        for (auto const &edge : edges) {
            const auto node = edge.source.raw();
            if (stamp_[node] == generation_) {
                stamp_[node] = generation_ + 1;
                const auto removable = std::min(first_[node], second_[node]);
                first_[node] = removable;
                second_[node] = removable;
                pairs += removable;
            }
            auto &remaining = edge.complement ? second_[node] : first_[node];
            if (remaining > 0) {
                --remaining;
                continue;
            }
            edges[kept++] = edge;
        }
        edges.resize(kept);
        return pairs;
    }

    [[nodiscard]] auto bytes() const noexcept -> std::size_t { return detail::bytes_of(stamp_) + detail::bytes_of(first_) + detail::bytes_of(second_); }

  private:
    std::vector<std::uint32_t> stamp_;
    std::vector<std::uint32_t> first_;
    std::vector<std::uint32_t> second_;
    std::uint32_t generation_ = 0;
};

struct simplified final {
    std::optional<bool> constant;
    op_code code = op_code::and_gate;
    std::uint32_t low = 0;
    std::uint32_t high = 0;
};

// AND (conjunction) or OR over non-constant `edges` plus `absorbing`
// constant operands (FALSE for AND, TRUE for OR).
[[nodiscard]] auto reduce_junction(operand_scratch &scratch, std::vector<edge_ref> &edges, const bool conjunction, const std::uint32_t absorbing)
    -> std::optional<bool> {
    const bool absorbed = !conjunction;
    if (absorbing > 0 || scratch.dedupe(edges)) {
        return absorbed;
    }
    if (edges.empty()) {
        return !absorbed;
    }
    return std::nullopt;
}

[[nodiscard]] auto negated(std::optional<bool> value, const bool negate) -> std::optional<bool> {
    if (value) {
        *value = *value != negate;
    }
    return value;
}

// Cardinality [low, high] over `edges` plus `trues` TRUE operands.
[[nodiscard]] auto reduce_cardinality(operand_scratch &scratch, std::vector<edge_ref> &edges, std::uint32_t trues, const std::uint32_t low,
                                      const std::uint32_t high) -> simplified {
    trues += scratch.remove_complement_pairs(edges);
    if (high < trues) {
        return simplified{.constant = false};
    }
    const auto count = static_cast<std::uint32_t>(edges.size());
    const auto shifted_low = low > trues ? low - trues : 0U;
    const auto shifted_high = std::min(high - trues, count);
    if (shifted_low > count) {
        return simplified{.constant = false};
    }
    if (shifted_low == 0 && shifted_high == count) {
        return simplified{.constant = true};
    }
    if (shifted_low == count) {
        return simplified{.constant = reduce_junction(scratch, edges, true, 0), .code = op_code::and_gate};
    }
    if (shifted_low == 1 && shifted_high == count) {
        return simplified{.constant = reduce_junction(scratch, edges, false, 0), .code = op_code::or_gate};
    }
    if (shifted_high == 0) {
        return simplified{.constant = negated(reduce_junction(scratch, edges, false, 0), true), .code = op_code::nor_gate};
    }
    if (shifted_low == 0 && shifted_high + 1 == count) {
        return simplified{.constant = negated(reduce_junction(scratch, edges, true, 0), true), .code = op_code::nand_gate};
    }
    return simplified{.code = op_code::cardinality, .low = shifted_low, .high = shifted_high};
}

// Edges with source >= atoms are atoms (see propagate_constants). Parity
// never moves onto an atom or an atom event: negating either edge
// would read the other variable.
[[nodiscard]] auto reduce(operand_scratch &scratch, const op_code code, std::vector<edge_ref> &edges, const std::uint32_t trues, const std::uint32_t falses,
                          const std::uint32_t low, const std::uint32_t high, const std::uint32_t atoms, detail::split_context const &split,
                          const bool split_gate) -> simplified {
    switch (code) {
    case op_code::and_gate:
    case op_code::nand_gate:
        return simplified{.constant = negated(reduce_junction(scratch, edges, true, falses), code == op_code::nand_gate), .code = code};
    case op_code::or_gate:
    case op_code::nor_gate:
        return simplified{.constant = negated(reduce_junction(scratch, edges, false, trues), code == op_code::nor_gate), .code = code};
    case op_code::not_gate:
        return simplified{.constant = edges.empty() ? std::optional<bool>(trues == 0) : std::nullopt, .code = code};
    case op_code::xor_gate:
    case op_code::xnor_gate: {
        auto [parity, cancelled] = scratch.cancel_parity(edges);
        if (!cancelled) {
            if (trues + falses == 0) {
                return simplified{.code = code};
            }
            for (auto &edge : edges) {
                edge.complement = false;
            }
        }
        parity = parity != ((trues & 1U) != 0);
        if (edges.empty()) {
            return simplified{.constant = parity != (code == op_code::xnor_gate), .code = code};
        }
        if (!parity) {
            return simplified{.code = code};
        }
        const auto negatable =
            std::ranges::find_if(edges, [&](edge_ref const &edge) { return edge.source.raw() < atoms && !(split_gate && split.member(edge.source.raw())); });
        if (negatable == edges.end()) {
            return simplified{.code = code == op_code::xor_gate ? op_code::xnor_gate : op_code::xor_gate};
        }
        negatable->complement = true;
        return simplified{.code = code};
    }
    case op_code::cardinality:
        return reduce_cardinality(scratch, edges, trues, low, high);
    default:
        return simplified{.code = code};
    }
}

// Whether an atom and its event both occur: the x / not x rules would
// apply under Boolean semantics.
[[nodiscard]] auto skips_pair(std::vector<edge_ref> const &edges, const std::uint32_t atoms, std::vector<std::uint32_t> &present,
                              const std::uint32_t generation) -> bool {
    for (auto const &edge : edges) {
        if (edge.source.raw() < atoms) {
            present[edge.source.raw()] = generation;
        }
    }
    return std::ranges::any_of(edges, [&](edge_ref const &edge) { return edge.source.raw() >= atoms && present[edge.source.raw() - atoms] == generation; });
}

} // namespace

auto propagate_constants(circuit const &input, pass_options const &options) -> transform_result {
    auto report = detail::begin_report(input, pass_id::propagate_constants);
    if (auto trivial = detail::trivial_result(input, report)) {
        return std::move(*trivial);
    }
    const auto split = detail::split_context::make(input, options.atoms, report);
    if (!split) {
        return transform_result{.output = {}, .report = std::move(report)};
    }

    // Phase 1 simplifies every node into `definitions`, keyed by input id: a
    // non-constant image is always the node itself, so operands stay input
    // ids. Phase 2 emits only what the targets still reach, because folding a
    // gate to a constant can orphan nodes simplified before it.
    const auto order = detail::cone_order(input);
    const auto codes = input.op_codes();
    const auto node_count = input.node_count();
    auto &map = report.node_map;
    // An atom edge (v, complement) becomes the uncomplemented edge
    // (node_count + v, false): a variable of its own, so no rule pairs,
    // cancels or deduplicates it with v.
    const auto atoms = node_count;
    operand_scratch scratch;
    scratch.next_gate(2 * node_count);
    std::vector<std::uint32_t> present(split->active() ? node_count : 0, 0);
    std::uint32_t generation = 0;
    std::vector<edge_ref> edges;
    std::vector<edge_ref> definitions;
    std::vector<std::uint32_t> definition_offset(node_count, 0);
    std::vector<std::uint32_t> definition_count(node_count, 0);
    std::vector<op_code> definition_code(codes.begin(), codes.end());
    std::vector<std::uint32_t> definition_low(node_count, 0);
    std::vector<std::uint32_t> definition_high(node_count, 0);

    for (const auto node : order) {
        const auto code = codes[node];
        if (code == op_code::constant || code == op_code::house_event) {
            map[node] = detail::constant(input.constant_values()[node] != 0);
            continue;
        }
        if (is_leaf(code)) {
            map[node] = detail::reference(bool_node_id(node));
            continue;
        }

        edges.clear();
        std::uint32_t trues = 0;
        std::uint32_t falses = 0;
        const auto operands = input.operands(bool_node_id(node));
        bool has_atom = false;
        for (auto const &edge : operands) {
            const auto image = detail::through(map[edge.source.raw()], edge.complement);
            if (image.kind == image_kind::constant) {
                ++(image.constant_value ? trues : falses);
            } else if (split->atom_edge(node, edge)) {
                edges.push_back(edge_ref{.source = bool_node_id(atoms + edge.source.raw())});
                has_atom = true;
            } else {
                edges.push_back(image.edge);
            }
        }
        if (has_atom && skips_pair(edges, atoms, present, ++generation)) {
            ++report.skipped;
        }

        scratch.next_gate(2 * node_count);
        const auto low = input.cardinality_low()[node];
        const auto high = input.cardinality_high()[node];
        const auto result = reduce(scratch, code, edges, trues, falses, low, high, atoms, *split, split->in_cone(node));
        if (result.constant) {
            map[node] = detail::constant(*result.constant);
            ++report.rewrites;
            continue;
        }
        if (result.code != code || edges.size() != operands.size() || result.low != low || result.high != high) {
            ++report.rewrites;
        }
        definition_offset[node] = static_cast<std::uint32_t>(definitions.size());
        definition_count[node] = static_cast<std::uint32_t>(edges.size());
        definition_code[node] = result.code;
        definition_low[node] = result.low;
        definition_high[node] = result.high;
        for (auto const &edge : edges) {
            definitions.push_back(edge.source.raw() < atoms ? edge : edge_ref{.source = bool_node_id(edge.source.raw() - atoms), .complement = true});
        }
        map[node] = detail::reference(bool_node_id(node));
    }

    std::vector<std::uint8_t> live(node_count, 0);
    for (auto const &target : input.targets()) {
        if (auto const &image = map[target.node.raw()]; image.kind == image_kind::reference) {
            live[image.edge.source.raw()] = 1;
        }
    }
    for (auto node = order.rbegin(); node != order.rend(); ++node) {
        if (live[*node] == 0) {
            continue;
        }
        for (std::uint32_t index = 0; index < definition_count[*node]; ++index) {
            live[definitions[definition_offset[*node] + index].source.raw()] = 1;
        }
    }

    detail::emitter emitter(input);
    for (const auto node : order) {
        auto &image = map[node];
        if (image.kind != image_kind::reference) {
            continue;
        }
        if (live[node] == 0) {
            image = node_image{};
            continue;
        }
        if (is_leaf(codes[node])) {
            image.edge.source = emitter.leaf(node);
            continue;
        }
        edges.clear();
        for (std::uint32_t index = 0; index < definition_count[node]; ++index) {
            auto const &operand = definitions[definition_offset[node] + index];
            edges.push_back(edge_ref{.source = map[operand.source.raw()].edge.source, .complement = operand.complement});
        }
        image.edge.source = emitter.gate(definition_code[node], edges, definition_low[node], definition_high[node]);
    }

    report.scratch_bytes = detail::bytes_of(order) + detail::bytes_of(edges) + scratch.bytes() + split->bytes() + detail::bytes_of(present) +
                           detail::bytes_of(definitions) + detail::bytes_of(definition_offset) + detail::bytes_of(definition_count) +
                           detail::bytes_of(definition_code) + detail::bytes_of(definition_low) + detail::bytes_of(definition_high) + detail::bytes_of(live);
    return std::move(emitter).finish(std::move(report));
}

} // namespace fuzzybits::ir::pc::passes
