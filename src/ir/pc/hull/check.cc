#include "ir/pc/hull/hull.h"

#include "api/diagnostics.h"
#include "ir/pc/hull/detail/bdd.h"
#include "ir/pc/hull/detail/bits.h"
#include "ir/pc/hull/detail/residual.h"
#include "ir/pc/hull/detail/sat.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace fuzzybits::ir::pc::hull {

namespace detail {

enum class cone_kind : std::uint8_t {
    event,
    constant,
    and_gate,
    or_gate,
    xor_gate,
    cardinality,
};

// A block is 256 lanes: one node's dual-rail value fills a cache line, so
// the random accesses of the bound pass are amortised over 4 words.
inline constexpr std::size_t block_words = 4;
inline constexpr std::size_t block_lanes = 64 * block_words;

// Kleene value of one node for a block: a lane is true, false, or (in
// neither rail) undetermined.
struct alignas(64) rail final {
    std::array<std::uint64_t, block_words> is_true;
    std::array<std::uint64_t, block_words> is_false;

    friend auto operator==(rail const &, rail const &) -> bool = default;
};

// Leaf value for event bits x, in dual rail:
// true = (x & true_mask) | true_constant, false = (~x & false_mask) | false_constant.
// A lane where neither rail is set is a free event.
struct leaf_rule final {
    std::uint64_t true_mask = 0;
    std::uint64_t true_constant = 0;
    std::uint64_t false_mask = 0;
    std::uint64_t false_constant = 0;
};

struct event_group final {
    std::uint32_t word = 0;
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
};

struct workspace;

// The target's cone in topological order ("positions"), with fan-in and
// fan-out lists and one leaf rule per cone event. `base` is the
// dual-rail value of every position when each event takes its more
// likely value (`base_events`); a block propagates only the events
// that differ from it.
struct cone final {
    check_options options;
    std::vector<bool_node_id> event_nodes;
    std::vector<std::uint32_t> cone_events;
    // Per atom (ordinal): the event, whether the atom samples its
    // complement, its position, leaf rule and base value.
    std::vector<std::uint32_t> events;
    std::vector<std::uint8_t> complemented;
    std::vector<std::uint32_t> event_positions;
    std::vector<leaf_rule> rules;
    std::vector<std::uint64_t> base_events;
    std::vector<event_group> groups;
    std::vector<cone_kind> kinds;
    std::vector<std::uint8_t> negated;
    std::vector<std::uint32_t> low;
    std::vector<std::uint32_t> high;
    std::vector<std::uint32_t> input_offsets;
    std::vector<std::uint32_t> inputs;
    std::vector<std::uint32_t> output_offsets;
    std::vector<std::uint32_t> outputs;
    std::vector<std::uint32_t> sources;
    std::vector<rail> base;
    std::uint32_t root = 0;

    mutable std::mutex pool_mutex;
    mutable std::vector<std::unique_ptr<workspace>> pool;

    [[nodiscard]] auto size() const noexcept -> std::uint32_t { return static_cast<std::uint32_t>(kinds.size()); }
    [[nodiscard]] auto inputs_of(const std::uint32_t position) const noexcept -> std::span<std::uint32_t const> {
        return std::span<std::uint32_t const>{inputs}.subspan(input_offsets[position], input_offsets[position + 1] - input_offsets[position]);
    }
    [[nodiscard]] auto outputs_of(const std::uint32_t position) const noexcept -> std::span<std::uint32_t const> {
        return std::span<std::uint32_t const>{outputs}.subspan(output_offsets[position], output_offsets[position + 1] - output_offsets[position]);
    }
};

struct workspace final {
    explicit workspace(cone const &shape)
        : rails(shape.base), touched_flags(shape.size(), 0), dirty((std::size_t{shape.size()} + 63) / 64, 0), event_words(shape.events.size() * block_words, 0),
          stamps(shape.size(), 0), mapped(shape.size(), 0) {}

    std::vector<rail> rails;
    std::vector<std::uint8_t> touched_flags;
    std::vector<std::uint32_t> touched;
    std::vector<std::uint64_t> dirty;
    std::size_t dirty_first = 0;
    std::size_t dirty_last = 0;
    std::vector<std::uint64_t> event_words;
    std::array<std::uint64_t, 64> rows{};

    std::vector<std::uint32_t> stamps;
    std::uint32_t epoch = 0;
    std::vector<std::uint32_t> mapped;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> stack;
    std::vector<std::uint32_t> collected;
    residual current;
    residual next;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> polarity;
    std::vector<std::uint8_t> assignment;
    std::vector<std::uint8_t> scratch;
    std::vector<std::uint64_t> table;
    bdd_manager bdd;
    sat_solver sat;
};

namespace {

constexpr std::uint32_t no_event = std::numeric_limits<std::uint32_t>::max();
constexpr std::size_t no_word = std::numeric_limits<std::size_t>::max();

[[nodiscard]] auto evaluate_position(cone const &shape, const std::uint32_t position, rail const *rails) -> rail {
    rail out{};
    const auto inputs = shape.inputs_of(position);
    // Input value with the edge's complement applied.
    const auto input = [&](const std::uint32_t literal) -> std::pair<std::uint64_t const *, std::uint64_t const *> {
        auto const &source = rails[literal >> 1];
        return (literal & 1U) != 0 ? std::pair{source.is_false.data(), source.is_true.data()} : std::pair{source.is_true.data(), source.is_false.data()};
    };
    switch (shape.kinds[position]) {
    case cone_kind::and_gate:
        out.is_true.fill(all_ones);
        for (const auto literal : inputs) {
            const auto [input_true, input_false] = input(literal);
            for (std::size_t word = 0; word < block_words; ++word) {
                out.is_true[word] &= input_true[word];
                out.is_false[word] |= input_false[word];
            }
        }
        break;
    case cone_kind::or_gate:
        out.is_false.fill(all_ones);
        for (const auto literal : inputs) {
            const auto [input_true, input_false] = input(literal);
            for (std::size_t word = 0; word < block_words; ++word) {
                out.is_true[word] |= input_true[word];
                out.is_false[word] &= input_false[word];
            }
        }
        break;
    case cone_kind::xor_gate:
        out.is_false.fill(all_ones);
        for (const auto literal : inputs) {
            const auto [input_true, input_false] = input(literal);
            for (std::size_t word = 0; word < block_words; ++word) {
                const auto next_true = (out.is_true[word] & input_false[word]) | (out.is_false[word] & input_true[word]);
                out.is_false[word] = (out.is_true[word] & input_true[word]) | (out.is_false[word] & input_false[word]);
                out.is_true[word] = next_true;
            }
        }
        break;
    case cone_kind::cardinality: {
        const std::uint64_t count = inputs.size();
        const std::uint64_t low = shape.low[position];
        const std::uint64_t high = shape.high[position];
        for (std::size_t word = 0; word < block_words; ++word) {
            sliced_counter ones;
            sliced_counter zeros;
            for (const auto literal : inputs) {
                const auto [input_true, input_false] = input(literal);
                ones.add(input_true[word]);
                zeros.add(input_false[word]);
            }
            out.is_true[word] = ones.at_least(low) & zeros.at_least(count - high);
            out.is_false[word] = ones.at_least(high + 1) | zeros.at_least(count - low + 1);
        }
        break;
    }
    case cone_kind::event:
    case cone_kind::constant:
        return rails[position];
    }
    if (shape.negated[position] != 0) {
        std::swap(out.is_true, out.is_false);
    }
    return out;
}

[[nodiscard]] constexpr auto to_residual_kind(const cone_kind kind) noexcept -> residual_kind {
    switch (kind) {
    case cone_kind::and_gate:
        return residual_kind::and_gate;
    case cone_kind::or_gate:
        return residual_kind::or_gate;
    case cone_kind::xor_gate:
        return residual_kind::xor_gate;
    case cone_kind::cardinality:
        return residual_kind::cardinality;
    case cone_kind::event:
    case cone_kind::constant:
        break;
    }
    return residual_kind::variable;
}

// One lane of the dual-rail cone values.
struct lane_view final {
    cone const &shape;
    workspace const &space;
    std::uint32_t lane;

    [[nodiscard]] auto value(const std::uint32_t position) const noexcept -> std::uint8_t {
        auto const &value = space.rails[position];
        if (((value.is_true[lane / 64] >> (lane % 64)) & 1U) != 0) {
            return value_true;
        }
        return ((value.is_false[lane / 64] >> (lane % 64)) & 1U) != 0 ? value_false : value_unknown;
    }
    [[nodiscard]] auto is_variable(const std::uint32_t position) const noexcept -> bool { return shape.kinds[position] == cone_kind::event; }
    [[nodiscard]] auto kind(const std::uint32_t position) const noexcept -> residual_kind { return to_residual_kind(shape.kinds[position]); }
    [[nodiscard]] auto negated(const std::uint32_t position) const noexcept -> std::uint32_t { return shape.negated[position]; }
    [[nodiscard]] auto low(const std::uint32_t position) const noexcept -> std::uint32_t { return shape.low[position]; }
    [[nodiscard]] auto high(const std::uint32_t position) const noexcept -> std::uint32_t { return shape.high[position]; }
    [[nodiscard]] auto inputs(const std::uint32_t position) const noexcept -> std::span<std::uint32_t const> { return shape.inputs_of(position); }
    [[nodiscard]] auto source(const std::uint32_t position) const noexcept -> std::uint32_t { return shape.sources[position]; }
};

// A residual under a three-valued evaluation.
struct residual_view final {
    residual const &circuit;
    std::span<std::uint8_t const> values;

    [[nodiscard]] auto value(const std::uint32_t node) const noexcept -> std::uint8_t { return values[node]; }
    [[nodiscard]] auto is_variable(const std::uint32_t node) const noexcept -> bool { return circuit.nodes[node].kind == residual_kind::variable; }
    [[nodiscard]] auto kind(const std::uint32_t node) const noexcept -> residual_kind { return circuit.nodes[node].kind; }
    [[nodiscard]] static auto negated(std::uint32_t /*node*/) noexcept -> std::uint32_t { return 0; }
    [[nodiscard]] auto low(const std::uint32_t node) const noexcept -> std::uint32_t { return circuit.nodes[node].low; }
    [[nodiscard]] auto high(const std::uint32_t node) const noexcept -> std::uint32_t { return circuit.nodes[node].high; }
    [[nodiscard]] auto inputs(const std::uint32_t node) const noexcept -> std::span<std::uint32_t const> { return circuit.inputs(circuit.nodes[node]); }
    [[nodiscard]] auto source(const std::uint32_t node) const noexcept -> std::uint32_t { return circuit.nodes[node].source; }
};

// Residual literal of an undetermined node whose undetermined inputs
// are already mapped. Determined inputs are folded: they are neutral
// for AND/OR (an absorbing input would have determined the node),
// flip XOR parity, and shift cardinality bounds.
template <typename view_type> [[nodiscard]] auto emit(view_type const &view, const std::uint32_t node, workspace &space, residual &out) -> std::uint32_t {
    if (view.is_variable(node)) {
        const auto index = static_cast<std::uint32_t>(out.nodes.size());
        out.nodes.push_back(residual_node{
            .kind = residual_kind::variable,
            .variable = static_cast<std::uint32_t>(out.variables.size()),
            .source = view.source(node),
        });
        out.variables.push_back(index);
        return index << 1;
    }
    auto kind = view.kind(node);
    auto negated = view.negated(node);
    space.collected.clear();
    std::uint32_t ones = 0;
    for (const auto literal : view.inputs(node)) {
        const auto input = literal >> 1;
        const auto value = view.value(input);
        if (value == value_unknown) {
            space.collected.push_back(space.mapped[input] ^ (literal & 1U));
        } else {
            ones += value ^ (literal & 1U);
        }
    }
    const auto count = static_cast<std::uint32_t>(space.collected.size());
    std::uint32_t low = 0;
    std::uint32_t high = 0;
    if (kind == residual_kind::xor_gate) {
        negated ^= ones & 1U;
    } else if (kind == residual_kind::cardinality) {
        const auto shifted_low = view.low(node) > ones ? view.low(node) - ones : 0;
        const auto shifted_high = std::min(view.high(node) - ones, count);
        if (shifted_low > 0 && shifted_high < count) {
            low = shifted_low;
            high = shifted_high;
        } else {
            // One-sided: at least `low` of `count`, possibly negated.
            if (shifted_low == 0) {
                negated ^= 1U;
                low = shifted_high + 1;
            } else {
                low = shifted_low;
            }
            high = count;
            kind = low == 1 ? residual_kind::or_gate : low == count ? residual_kind::and_gate : residual_kind::cardinality;
        }
    }
    if (count == 1 && kind != residual_kind::cardinality) {
        return space.collected[0] ^ negated;
    }
    const auto index = static_cast<std::uint32_t>(out.nodes.size());
    out.nodes.push_back(residual_node{
        .kind = kind,
        .begin = static_cast<std::uint32_t>(out.literals.size()),
        .count = count,
        .low = low,
        .high = high,
    });
    out.literals.insert(out.literals.end(), space.collected.begin(), space.collected.end());
    return (index << 1) ^ negated;
}

// Residual of the undetermined nodes reachable from `root` through
// undetermined nodes (iterative depth-first search, post-order), or
// the root's value when it is determined.
template <typename view_type> [[nodiscard]] auto extract(view_type const &view, const std::uint32_t root, workspace &space, residual &out) -> std::uint8_t {
    out.clear();
    const auto top = root >> 1;
    if (const auto value = view.value(top); value != value_unknown) {
        return static_cast<std::uint8_t>(value ^ (root & 1U));
    }
    if (++space.epoch == 0) {
        std::ranges::fill(space.stamps, 0U);
        space.epoch = 1;
    }
    space.stack.clear();
    space.stack.emplace_back(top, 0);
    space.stamps[top] = space.epoch;
    while (!space.stack.empty()) {
        const auto node = space.stack.back().first;
        auto next = space.stack.back().second;
        const auto inputs = view.inputs(node);
        bool descended = false;
        while (next < inputs.size()) {
            const auto input = inputs[next++] >> 1;
            if (space.stamps[input] != space.epoch && view.value(input) == value_unknown) {
                space.stamps[input] = space.epoch;
                space.stack.back().second = next;
                space.stack.emplace_back(input, 0);
                descended = true;
                break;
            }
        }
        if (descended) {
            continue;
        }
        space.stack.pop_back();
        space.mapped[node] = emit(view, node, space, out);
    }
    out.root = space.mapped[top] ^ (root & 1U);
    return value_unknown;
}

[[nodiscard]] auto decide_lane(cone const &shape, workspace &space, const std::uint32_t lane, check_stats &stats) -> bool {
    auto const &options = shape.options;
    (void)extract(lane_view{.shape = shape, .space = space, .lane = lane}, shape.root << 1, space, space.current);
    if (options.pure_literals) {
        while (true) {
            polarities(space.current, space.polarity);
            space.values.assign(space.current.nodes.size(), value_unknown);
            bool fixed = false;
            for (const auto node : space.current.variables) {
                const auto polarity = space.polarity[node];
                if (polarity == (polarity_positive | polarity_negative)) {
                    continue;
                }
                space.values[node] = polarity == polarity_positive ? value_true : value_false;
                fixed = true;
            }
            if (!fixed) {
                break;
            }
            evaluate_kleene(space.current, space.values);
            const auto value = extract(residual_view{.circuit = space.current, .values = space.values}, space.current.root, space, space.next);
            if (value != value_unknown) {
                ++stats.reduction;
                return value == value_true;
            }
            std::swap(space.current, space.next);
        }
    }

    auto const &circuit = space.current;
    const auto variables = circuit.variable_count();
    if (options.witnesses) {
        space.assignment.resize(variables);
        for (std::uint32_t variable = 0; variable < variables; ++variable) {
            const auto source = circuit.nodes[circuit.variables[variable]].source;
            const auto word = space.event_words[source * block_words + lane / 64];
            space.assignment[variable] = static_cast<std::uint8_t>((word >> (lane % 64)) & 1U);
        }
        bool found = evaluate(circuit, space.assignment, space.scratch);
        for (const std::uint8_t constant : {std::uint8_t{0}, std::uint8_t{1}}) {
            if (found) {
                break;
            }
            std::ranges::fill(space.assignment, constant);
            found = evaluate(circuit, space.assignment, space.scratch);
        }
        if (found) {
            ++stats.witness;
            return true;
        }
    }

    stats.residual_variables_total += variables;
    stats.residual_variables_max = std::max(stats.residual_variables_max, variables);
    const std::uint64_t size = circuit.nodes.size() + circuit.literals.size();
    if (variables <= std::min<std::uint32_t>(options.truth_table_max_variables, 40)) {
        const std::uint64_t chunks = variables <= 6 ? 1 : std::uint64_t{1} << (variables - 6);
        if (size <= options.truth_table_budget / chunks) {
            ++stats.truth_table;
            return truth_table_satisfiable(circuit, space.table);
        }
    }
    if (options.bdd_node_budget > 0 && variables <= options.bdd_max_variables) {
        if (const auto satisfiable = space.bdd.satisfiable(circuit, options.bdd_node_budget)) {
            ++stats.bdd;
            return *satisfiable;
        }
        ++stats.bdd_budget_exceeded;
    }
    ++stats.search;
    return search_satisfiable(circuit, space.sat);
}

void mark_dirty(workspace &space, const std::uint32_t position) noexcept {
    const auto word = std::size_t{position} >> 6;
    space.dirty[word] |= std::uint64_t{1} << (position & 63U);
    space.dirty_first = std::min(space.dirty_first, word);
    space.dirty_last = space.dirty_last == no_word ? word : std::max(space.dirty_last, word);
}

void update(cone const &shape, workspace &space, const std::uint32_t position, rail const &value) {
    auto &current = space.rails[position];
    if (current == value) {
        return;
    }
    if (space.touched_flags[position] == 0) {
        space.touched_flags[position] = 1;
        space.touched.push_back(position);
    }
    current = value;
    for (const auto output : shape.outputs_of(position)) {
        mark_dirty(space, output);
    }
}

// Event bits of the block's lanes, block_words words per cone event.
void load_events(cone const &shape, workspace &space, lane_matrix const &lanes, const std::size_t block, std::array<std::uint64_t, block_words> const &valid) {
    const auto first_word = block * block_words;
    if (lanes.layout == lane_layout::event_major) {
        const auto words = (lanes.lane_count + 63) / 64;
        for (std::size_t ordinal = 0; ordinal < shape.events.size(); ++ordinal) {
            auto const *row = lanes.words.data() + std::size_t{shape.events[ordinal]} * lanes.stride;
            for (std::size_t word = 0; word < block_words; ++word) {
                const auto sample = first_word + word < words ? row[first_word + word] : 0;
                space.event_words[ordinal * block_words + word] = (shape.complemented[ordinal] != 0 ? ~sample : sample) & valid[word];
            }
        }
        return;
    }
    for (auto const &group : shape.groups) {
        for (std::size_t word = 0; word < block_words; ++word) {
            const auto first_lane = (first_word + word) * 64;
            for (std::size_t lane = 0; lane < 64; ++lane) {
                space.rows[lane] = first_lane + lane < lanes.lane_count ? lanes.words[(first_lane + lane) * lanes.stride + group.word] : 0;
            }
            transpose64(space.rows);
            for (auto ordinal = group.begin; ordinal < group.end; ++ordinal) {
                const auto sample = space.rows[shape.events[ordinal] & 63U];
                space.event_words[std::size_t{ordinal} * block_words + word] = shape.complemented[ordinal] != 0 ? ~sample : sample;
            }
        }
    }
}

// Answers for one block of up to block_lanes lanes, written to
// out[block * block_words + word]. The workspace starts and ends in
// the base state.
void run_block(cone const &shape, workspace &space, lane_matrix const &lanes, const std::size_t block, std::span<std::uint64_t> out, check_stats &stats) {
    std::array<std::uint64_t, block_words> valid{};
    for (std::size_t word = 0; word < block_words; ++word) {
        const auto first_lane = (block * block_words + word) * 64;
        if (first_lane < lanes.lane_count) {
            const auto count = std::min<std::size_t>(64, lanes.lane_count - first_lane);
            valid[word] = count == 64 ? all_ones : (std::uint64_t{1} << count) - 1;
        }
    }
    load_events(shape, space, lanes, block, valid);

    space.dirty_first = no_word;
    space.dirty_last = no_word;
    rail leaf{};
    for (std::size_t ordinal = 0; ordinal < shape.events.size(); ++ordinal) {
        auto *words = space.event_words.data() + ordinal * block_words;
        const auto base = shape.base_events[ordinal];
        std::uint64_t differs = 0;
        for (std::size_t word = 0; word < block_words; ++word) {
            words[word] = (words[word] & valid[word]) | (base & ~valid[word]);
            differs |= words[word] ^ base;
        }
        if (differs == 0) {
            continue;
        }
        auto const &rule = shape.rules[ordinal];
        for (std::size_t word = 0; word < block_words; ++word) {
            leaf.is_true[word] = (words[word] & rule.true_mask) | rule.true_constant;
            leaf.is_false[word] = (~words[word] & rule.false_mask) | rule.false_constant;
        }
        update(shape, space, shape.event_positions[ordinal], leaf);
    }
    if (space.dirty_first != no_word) {
        for (auto word = space.dirty_first; word <= space.dirty_last; ++word) {
            while (space.dirty[word] != 0) {
                const auto bit = static_cast<std::uint32_t>(std::countr_zero(space.dirty[word]));
                space.dirty[word] &= space.dirty[word] - 1;
                const auto position = static_cast<std::uint32_t>(word * 64 + bit);
                update(shape, space, position, evaluate_position(shape, position, space.rails.data()));
            }
        }
    }

    auto const &root = space.rails[shape.root];
    for (std::size_t word = 0; word < block_words; ++word) {
        if (valid[word] == 0) {
            break;
        }
        const auto root_true = root.is_true[word] & valid[word];
        const auto root_false = root.is_false[word] & valid[word];
        auto undecided = valid[word] & ~(root_true | root_false);
        auto result = root_true;
        stats.lanes += static_cast<std::uint64_t>(std::popcount(valid[word]));
        stats.bounds += static_cast<std::uint64_t>(std::popcount(root_true | root_false));
        while (undecided != 0) {
            const auto bit = static_cast<std::uint32_t>(std::countr_zero(undecided));
            undecided &= undecided - 1;
            if (decide_lane(shape, space, static_cast<std::uint32_t>(word * 64 + bit), stats)) {
                result |= std::uint64_t{1} << bit;
            }
        }
        stats.hits += static_cast<std::uint64_t>(std::popcount(result));
        out[block * block_words + word] = result;
    }

    for (const auto position : space.touched) {
        space.rails[position] = shape.base[position];
        space.touched_flags[position] = 0;
    }
    space.touched.clear();
}

[[nodiscard]] auto acquire(cone const &shape) -> std::unique_ptr<workspace> {
    {
        const std::scoped_lock lock(shape.pool_mutex);
        if (!shape.pool.empty()) {
            auto space = std::move(shape.pool.back());
            shape.pool.pop_back();
            return space;
        }
    }
    return std::make_unique<workspace>(shape);
}

void release(cone const &shape, std::unique_ptr<workspace> space) {
    const std::scoped_lock lock(shape.pool_mutex);
    shape.pool.push_back(std::move(space));
}

[[nodiscard]] auto literal_probability(circuit const &model, const std::uint32_t node) -> std::optional<double> {
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

[[nodiscard]] auto rule_for(const event_mode mode, std::uint8_t polarity, const bool zero, const bool pure_literals) -> leaf_rule {
    constexpr leaf_rule same{.true_mask = all_ones, .false_mask = all_ones};
    constexpr leaf_rule always_false{.false_constant = all_ones};
    constexpr leaf_rule always_true{.true_constant = all_ones};
    constexpr leaf_rule free_if_true{.false_mask = all_ones};
    constexpr leaf_rule free_if_false{.true_mask = all_ones};
    if (zero || polarity == 0) {
        return always_false;
    }
    if (!pure_literals) {
        polarity = polarity_positive | polarity_negative;
    }
    switch (mode) {
    case event_mode::fixed:
        return same;
    case event_mode::down:
    case event_mode::split:
        return polarity == polarity_positive ? same : polarity == polarity_negative ? always_false : free_if_true;
    case event_mode::up:
        return polarity == polarity_positive ? always_true : polarity == polarity_negative ? same : free_if_false;
    }
    return same;
}

} // namespace

} // namespace detail

void check_stats::merge(check_stats const &other) noexcept {
    lanes += other.lanes;
    hits += other.hits;
    bounds += other.bounds;
    reduction += other.reduction;
    witness += other.witness;
    truth_table += other.truth_table;
    bdd += other.bdd;
    bdd_budget_exceeded += other.bdd_budget_exceeded;
    search += other.search;
    residual_variables_total += other.residual_variables_total;
    residual_variables_max = std::max(residual_variables_max, other.residual_variables_max);
}

auto checker::make(circuit const &model, const bool_node_id target, std::span<event_mode const> modes, check_options const &options) -> api::result<checker> {
    using detail::cone_kind;
    if (!model.valid()) {
        return api::result<checker>::fail(api::make_error("pc.hull.invalid-circuit", "circuit is not sealed"));
    }
    const auto node_count = model.node_count();
    if (!target.valid() || target.raw() >= node_count) {
        return api::result<checker>::fail(api::make_error("pc.hull.invalid-target", "target does not name a node of the circuit"));
    }
    const auto op_codes = model.op_codes();

    auto shape = std::make_shared<detail::cone>();
    shape->options = options;
    std::vector<std::uint32_t> event_of(node_count, detail::no_event);
    for (std::uint32_t node = 0; node < node_count; ++node) {
        if (op_codes[node] == op_code::basic_event) {
            event_of[node] = static_cast<std::uint32_t>(shape->event_nodes.size());
            shape->event_nodes.emplace_back(node);
        }
    }
    if (!modes.empty() && modes.size() != shape->event_nodes.size()) {
        return api::result<checker>::fail(api::make_error("pc.hull.mode-count", "mode vector has " + std::to_string(modes.size()) + " entries for " +
                                                                                    std::to_string(shape->event_nodes.size()) + " basic events"));
    }

    // Cone in post-order (a topological order), by iterative depth-first
    // search. Keys are nodes, and node_count + e for the complement atom of
    // a `split` event e, which complemented edges into e read instead.
    const auto mode_of = [&](const std::uint32_t event) { return modes.empty() ? event_mode::down : modes[event]; };
    const auto key_of = [&](edge_ref const &edge) {
        const auto source = edge.source.raw();
        const auto event = event_of[source];
        return edge.complement && event != detail::no_event && mode_of(event) == event_mode::split ? node_count + event : source;
    };
    const auto key_count = std::size_t{node_count} + shape->event_nodes.size();
    constexpr auto unplaced = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> position_of(key_count, unplaced);
    std::vector<std::uint8_t> visited(key_count, 0);
    std::vector<std::uint32_t> order;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> stack;
    stack.emplace_back(target.raw(), 0);
    visited[target.raw()] = 1;
    while (!stack.empty()) {
        const auto node = stack.back().first;
        const auto operands = node < node_count ? model.operands(bool_node_id(node)) : std::span<edge_ref const>{};
        auto next = stack.back().second;
        bool descended = false;
        while (next < operands.size()) {
            const auto operand = key_of(operands[next++]);
            if (visited[operand] == 0) {
                visited[operand] = 1;
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
        position_of[node] = static_cast<std::uint32_t>(order.size());
        order.push_back(node);
    }

    const auto size = static_cast<std::uint32_t>(order.size());
    shape->root = position_of[target.raw()];
    shape->kinds.resize(size);
    shape->negated.assign(size, 0);
    shape->low.assign(size, 0);
    shape->high.assign(size, 0);
    shape->sources.assign(size, 0);
    shape->input_offsets.assign(std::size_t{size} + 1, 0);
    std::vector<std::uint8_t> constants(size, 0);
    for (std::uint32_t position = 0; position < size; ++position) {
        const auto node = order[position];
        const auto code = node < node_count ? op_codes[node] : op_code::basic_event;
        auto kind = cone_kind::and_gate;
        switch (code) {
        case op_code::basic_event:
            kind = cone_kind::event;
            break;
        case op_code::constant:
        case op_code::house_event:
            kind = cone_kind::constant;
            constants[position] = model.constant_values()[node] != 0 ? 1 : 0;
            break;
        case op_code::not_gate:
        case op_code::nand_gate:
            shape->negated[position] = 1;
            break;
        case op_code::and_gate:
            break;
        case op_code::nor_gate:
            shape->negated[position] = 1;
            kind = cone_kind::or_gate;
            break;
        case op_code::or_gate:
            kind = cone_kind::or_gate;
            break;
        case op_code::xnor_gate:
            shape->negated[position] = 1;
            kind = cone_kind::xor_gate;
            break;
        case op_code::xor_gate:
            kind = cone_kind::xor_gate;
            break;
        case op_code::cardinality:
            kind = cone_kind::cardinality;
            shape->low[position] = model.cardinality_low()[node];
            shape->high[position] = model.cardinality_high()[node];
            break;
        }
        shape->kinds[position] = kind;
        for (auto const &edge : node < node_count ? model.operands(bool_node_id(node)) : std::span<edge_ref const>{}) {
            const auto key = key_of(edge);
            shape->inputs.push_back((position_of[key] << 1) | (edge.complement && key < node_count ? 1U : 0U));
        }
        shape->input_offsets[position + 1] = static_cast<std::uint32_t>(shape->inputs.size());
    }

    shape->output_offsets.assign(std::size_t{size} + 1, 0);
    for (const auto literal : shape->inputs) {
        ++shape->output_offsets[(literal >> 1) + 1];
    }
    for (std::uint32_t position = 0; position < size; ++position) {
        shape->output_offsets[position + 1] += shape->output_offsets[position];
    }
    shape->outputs.resize(shape->inputs.size());
    {
        auto cursor = shape->output_offsets;
        for (std::uint32_t position = 0; position < size; ++position) {
            for (const auto literal : shape->inputs_of(position)) {
                shape->outputs[cursor[literal >> 1]++] = position;
            }
        }
    }

    // Which polarities of each position reach the root (syntactic unateness).
    std::vector<std::uint8_t> polarity(size, 0);
    polarity[shape->root] = detail::polarity_positive;
    for (auto position = size; position-- > 0;) {
        auto passed = polarity[position];
        if (passed == 0) {
            continue;
        }
        if (shape->negated[position] != 0) {
            passed = detail::swap_polarity(passed);
        }
        const auto inputs = shape->inputs_of(position);
        if (shape->kinds[position] == cone_kind::xor_gate) {
            passed = detail::polarity_positive | detail::polarity_negative;
        } else if (shape->kinds[position] == cone_kind::cardinality) {
            const auto low = shape->low[position];
            const auto high = shape->high[position];
            if (low == 0 && high >= inputs.size()) {
                passed = 0;
            } else if (low == 0) {
                passed = detail::swap_polarity(passed);
            } else if (high < inputs.size()) {
                passed = detail::polarity_positive | detail::polarity_negative;
            }
        }
        for (const auto literal : inputs) {
            polarity[literal >> 1] |= (literal & 1U) != 0 ? detail::swap_polarity(passed) : passed;
        }
    }

    // Atoms ordered by (event, complement); a complement atom samples not x_e.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> atoms;
    for (std::uint32_t position = 0; position < size; ++position) {
        if (shape->kinds[position] == cone_kind::event) {
            const auto key = order[position];
            atoms.emplace_back(key < node_count ? 2 * event_of[key] : 2 * (key - node_count) + 1, position);
        }
    }
    std::ranges::sort(atoms);
    for (auto const &[atom, position] : atoms) {
        if (shape->cone_events.empty() || shape->cone_events.back() != atom / 2) {
            shape->cone_events.push_back(atom / 2);
        }
    }
    shape->events.resize(atoms.size());
    shape->complemented.resize(atoms.size());
    shape->event_positions.resize(atoms.size());
    shape->rules.resize(atoms.size());
    shape->base_events.resize(atoms.size());
    for (std::uint32_t ordinal = 0; ordinal < atoms.size(); ++ordinal) {
        const auto [atom, position] = atoms[ordinal];
        const auto event = atom / 2;
        const bool complemented = (atom & 1U) != 0;
        const auto node = shape->event_nodes[event].raw();
        const auto probability = detail::literal_probability(model, node);
        const auto mode = mode_of(event);
        // A complement atom has probability 1 - p and is FALSE when that is 0.
        const bool zero = complemented ? probability == 1.0 : probability == 0.0;
        const bool likely = complemented ? probability < 0.5 : probability > 0.5;
        shape->events[ordinal] = event;
        shape->complemented[ordinal] = complemented ? 1 : 0;
        shape->event_positions[ordinal] = position;
        shape->sources[position] = ordinal;
        shape->rules[ordinal] = detail::rule_for(mode, polarity[position], zero, options.pure_literals);
        shape->base_events[ordinal] = probability.has_value() && likely ? detail::all_ones : 0;
        const auto word = event >> 6;
        if (shape->groups.empty() || shape->groups.back().word != word) {
            shape->groups.push_back(detail::event_group{.word = word, .begin = ordinal, .end = ordinal});
        }
        shape->groups.back().end = ordinal + 1;
    }

    shape->base.assign(size, detail::rail{});
    for (std::uint32_t position = 0; position < size; ++position) {
        auto &base = shape->base[position];
        switch (shape->kinds[position]) {
        case cone_kind::event: {
            const auto ordinal = shape->sources[position];
            auto const &rule = shape->rules[ordinal];
            const auto x = shape->base_events[ordinal];
            base.is_true.fill((x & rule.true_mask) | rule.true_constant);
            base.is_false.fill((~x & rule.false_mask) | rule.false_constant);
            break;
        }
        case cone_kind::constant:
            base.is_true.fill(constants[position] != 0 ? detail::all_ones : 0);
            base.is_false.fill(constants[position] != 0 ? 0 : detail::all_ones);
            break;
        default:
            base = detail::evaluate_position(*shape, position, shape->base.data());
            break;
        }
    }
    return api::result<checker>::ok(checker(std::move(shape)));
}

auto checker::event_count() const noexcept -> std::uint32_t { return static_cast<std::uint32_t>(cone_->event_nodes.size()); }

auto checker::event_node(const std::uint32_t event) const noexcept -> bool_node_id {
    return event < cone_->event_nodes.size() ? cone_->event_nodes[event] : bool_node_id::invalid();
}

auto checker::cone_events() const noexcept -> std::span<std::uint32_t const> { return cone_->cone_events; }

auto checker::cone_size() const noexcept -> std::uint32_t { return cone_->size(); }

auto checker::options() const noexcept -> check_options const & { return cone_->options; }

auto checker::check(std::span<std::uint64_t const> sample) const -> bool {
    std::uint64_t out = 0;
    check(lane_matrix{.words = sample, .lane_count = 1, .stride = sample.size(), .layout = lane_layout::lane_major}, std::span<std::uint64_t>{&out, 1});
    return (out & 1U) != 0;
}

void checker::check(lane_matrix const &lanes, std::span<std::uint64_t> out, check_stats *stats) const {
    auto const &shape = *cone_;
    const auto words = (lanes.lane_count + 63) / 64;
    const auto blocks = (lanes.lane_count + detail::block_lanes - 1) / detail::block_lanes;
    const std::size_t events = shape.event_nodes.size();
    if (out.size() < words) {
        throw std::invalid_argument("hull check: output has fewer than ceil(lanes / 64) words");
    }
    const bool lane_major = lanes.layout == lane_layout::lane_major;
    const auto rows = lane_major ? lanes.lane_count : events;
    const auto columns = lane_major ? events : lanes.lane_count;
    if (lanes.stride * 64 < columns || lanes.words.size() < rows * lanes.stride) {
        throw std::invalid_argument("hull check: lane matrix is smaller than its dimensions");
    }
    if (blocks == 0) {
        return;
    }

    check_stats total;
#if defined(_OPENMP)
    if (shape.options.parallel && blocks > 1 && omp_get_max_threads() > 1) {
        const auto block_count = static_cast<std::int64_t>(blocks);
#pragma omp parallel
        {
            auto space = detail::acquire(shape);
            check_stats local;
#pragma omp for schedule(dynamic, 1)
            for (std::int64_t block = 0; block < block_count; ++block) {
                detail::run_block(shape, *space, lanes, static_cast<std::size_t>(block), out, local);
            }
#pragma omp critical(fuzzybits_hull_check_stats)
            total.merge(local);
            detail::release(shape, std::move(space));
        }
        if (stats != nullptr) {
            stats->merge(total);
        }
        return;
    }
#endif
    auto space = detail::acquire(shape);
    for (std::size_t block = 0; block < blocks; ++block) {
        detail::run_block(shape, *space, lanes, block, out, total);
    }
    detail::release(shape, std::move(space));
    if (stats != nullptr) {
        stats->merge(total);
    }
}

} // namespace fuzzybits::ir::pc::hull
