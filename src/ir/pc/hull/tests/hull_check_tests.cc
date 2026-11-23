#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ir/pc/builder.h"
#include "ir/pc/edge.h"
#include "ir/pc/hull/hull.h"
#include "ir/pc/hull/tests/hull_test_support.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace pc = fuzzybits::ir::pc;
namespace hull = fuzzybits::ir::pc::hull;
using hull::event_mode;

struct generated final {
    pc::circuit model;
    pc::bool_node_id target;
    std::uint32_t events = 0;
    std::uint32_t zero_mask = 0;
    std::uint32_t one_mask = 0;
};

// Random DAG over at most 14 events with every op code, complemented edges,
// constants, house events, probability-0 and probability-1 events and
// shared sub-circuits.
auto generate(std::mt19937_64 &random) -> generated {
    std::uniform_int_distribution<std::uint32_t> pick_events(1, 14);
    std::uniform_int_distribution<std::uint32_t> pick_gates(1, 40);
    std::uniform_int_distribution<std::uint32_t> percent(0, 99);
    generated out;
    out.events = pick_events(random);
    const auto gates = pick_gates(random);
    pc::builder builder;
    std::vector<pc::bool_node_id> nodes;
    for (std::uint32_t event = 0; event < out.events; ++event) {
        const auto roll = percent(random);
        const double probability = roll < 10 ? 0.0 : roll < 20 ? 1.0 : 0.05 + 0.9 * (percent(random) / 100.0);
        if (probability == 0.0) {
            out.zero_mask |= 1U << event;
        }
        if (probability == 1.0) {
            out.one_mask |= 1U << event;
        }
        nodes.push_back(builder.add_basic_event(builder.add_literal_probability(probability)));
    }
    if (percent(random) < 30) {
        nodes.push_back(builder.add_constant(percent(random) < 50));
    }
    if (percent(random) < 30) {
        nodes.push_back(builder.add_house_event(percent(random) < 50));
    }
    constexpr std::array kinds{pc::op_code::and_gate,  pc::op_code::or_gate,    pc::op_code::xor_gate,
                               pc::op_code::not_gate,  pc::op_code::nand_gate,  pc::op_code::nor_gate,
                               pc::op_code::xnor_gate, pc::op_code::cardinality};
    std::vector<pc::edge_ref> operands;
    for (std::uint32_t gate = 0; gate < gates; ++gate) {
        const auto kind = kinds[random() % kinds.size()];
        const auto arity = kind == pc::op_code::not_gate ? 1U : 1U + static_cast<std::uint32_t>(random() % 5);
        operands.clear();
        for (std::uint32_t operand = 0; operand < arity; ++operand) {
            // Prefer recent nodes so that deep, shared structure appears.
            const auto span = static_cast<std::uint32_t>(nodes.size());
            const auto recent = std::min<std::uint32_t>(span, 8);
            const auto index = percent(random) < 60 ? span - 1 - static_cast<std::uint32_t>(random() % recent)
                                                    : static_cast<std::uint32_t>(random() % span);
            operands.push_back(pc::edge_ref{.source = nodes[index], .complement = percent(random) < 30});
        }
        if (kind == pc::op_code::cardinality) {
            const auto low = static_cast<std::uint32_t>(random() % (arity + 1));
            const auto high = low + static_cast<std::uint32_t>(random() % (arity - low + 1));
            nodes.push_back(builder.add_cardinality(low, high, operands));
        } else {
            nodes.push_back(builder.add_gate(kind, operands));
        }
    }
    out.target = percent(random) < 90 ? nodes.back() : nodes[random() % nodes.size()];
    (void)builder.add_target(out.target);
    auto sealed = std::move(builder).seal();
    REQUIRE(sealed.ok());
    out.model = sealed.output;
    return out;
}

// Reference for one mode vector: a truth table over the atoms (bit e is
// y+_e for every event, bit n + k is y-_e for the k-th split event), and an
// enumeration of every atom assignment allowed for sample x.
class reference final {
  public:
    reference(generated const &case_, std::span<event_mode const> modes) : events_(case_.events) {
        const auto mode = [&](std::uint32_t event) { return modes.empty() ? event_mode::down : modes[event]; };
        std::vector<std::uint32_t> split_bit(events_, 0);
        for (std::uint32_t event = 0; event < events_; ++event) {
            const auto bit = 1U << event;
            switch (mode(event)) {
            case event_mode::down:
                down_ |= bit;
                break;
            case event_mode::up:
                up_ |= bit;
                break;
            case event_mode::fixed:
                break;
            case event_mode::split:
                down_ |= bit;
                split_bit[event] = 1U << (events_ + static_cast<std::uint32_t>(split_.size()));
                split_.push_back(event);
                break;
            }
        }
        zero_ = case_.zero_mask;
        one_ = case_.one_mask;
        const auto bits = events_ + static_cast<std::uint32_t>(split_.size());
        table_.resize(std::size_t{1} << bits);
        std::vector<std::uint8_t> values(events_);
        std::vector<std::uint8_t> complements(events_, hull::testing::unset);
        for (std::size_t world = 0; world < table_.size(); ++world) {
            for (std::uint32_t event = 0; event < events_; ++event) {
                values[event] = static_cast<std::uint8_t>((world >> event) & 1U);
                if (split_bit[event] != 0) {
                    complements[event] = (world & split_bit[event]) != 0 ? 1 : 0;
                }
            }
            table_[world] = hull::testing::evaluate_nodes(case_.model, values, complements)[case_.target.raw()];
        }
    }

    [[nodiscard]] auto split_count() const -> std::size_t { return split_.size(); }

    [[nodiscard]] auto operator()(std::uint32_t x) const -> bool {
        const auto all = (1U << events_) - 1;
        const auto fixed_part = x & ~down_ & ~zero_ & all;
        auto free = ((x & down_) | (~x & up_)) & ~zero_ & all;
        for (std::size_t k = 0; k < split_.size(); ++k) {
            const auto event = split_[k];
            if (((x >> event) & 1U) == 0 && ((one_ >> event) & 1U) == 0) {
                free |= 1U << (events_ + k);
            }
        }
        for (std::uint32_t subset = free;; subset = (subset - 1) & free) {
            if (table_[fixed_part | subset] != 0) {
                return true;
            }
            if (subset == 0) {
                return false;
            }
        }
    }

  private:
    std::uint32_t events_ = 0;
    std::uint32_t down_ = 0;
    std::uint32_t up_ = 0;
    std::uint32_t zero_ = 0;
    std::uint32_t one_ = 0;
    std::vector<std::uint32_t> split_;
    std::vector<std::uint8_t> table_;
};

struct named_options final {
    std::string name;
    hull::check_options options;
};

auto option_sets() -> std::vector<named_options> {
    std::vector<named_options> out;
    out.push_back({"default", {}});
    hull::check_options exact{};
    exact.pure_literals = false;
    exact.witnesses = false;
    exact.truth_table_max_variables = 20;
    exact.bdd_node_budget = 0;
    out.push_back({"truth-table", exact});
    exact.truth_table_max_variables = 0;
    exact.bdd_node_budget = 1U << 16;
    out.push_back({"bdd", exact});
    exact.bdd_node_budget = 0;
    out.push_back({"search", exact});
    exact.bdd_node_budget = 3;
    exact.pure_literals = true;
    out.push_back({"bdd-overflow", exact});
    return out;
}

auto random_modes(std::mt19937_64 &random, std::uint32_t events) -> std::vector<event_mode> {
    std::vector<event_mode> modes(events);
    for (auto &mode : modes) {
        mode = static_cast<event_mode>(random() % 4);
    }
    return modes;
}

// The lowering's default (#190): split for events referenced through a
// complemented edge, down otherwise. Generated events are nodes 0..n-1.
auto split_complemented(pc::circuit const &model, std::uint32_t events) -> std::vector<event_mode> {
    std::vector<event_mode> modes(events, event_mode::down);
    for (auto const &edge : model.edges()) {
        if (edge.complement && model.op_codes()[edge.source.raw()] == pc::op_code::basic_event) {
            modes[edge.source.raw()] = event_mode::split;
        }
    }
    return modes;
}

// Keeps the reference table at most 2^18 entries.
void limit_splits(std::vector<event_mode> &modes, std::uint32_t events) {
    auto budget = 18 - static_cast<std::int32_t>(events);
    for (auto &mode : modes) {
        if (mode == event_mode::split && budget-- <= 0) {
            mode = event_mode::down;
        }
    }
}

auto build(std::initializer_list<double> probabilities, pc::builder &builder) -> std::vector<pc::bool_node_id> {
    std::vector<pc::bool_node_id> events;
    for (const auto probability : probabilities) {
        events.push_back(builder.add_basic_event(builder.add_literal_probability(probability)));
    }
    return events;
}

auto pos(pc::bool_node_id node) -> pc::edge_ref { return pc::edge_ref{.source = node}; }
auto neg(pc::bool_node_id node) -> pc::edge_ref { return pc::edge_ref{.source = node, .complement = true}; }

auto seal(pc::builder &builder) -> pc::circuit {
    auto sealed = std::move(builder).seal();
    REQUIRE(sealed.ok());
    return sealed.output;
}

auto hull_probability(pc::circuit const &model, pc::bool_node_id target, std::vector<event_mode> const &modes = {})
    -> double {
    auto made = hull::checker::make(model, target, modes);
    REQUIRE(made);
    return hull::testing::exact_probability(model, made.value());
}

// Answers for all 2^n samples of an n-event circuit, lane-major, one word per lane.
auto all_samples(hull::checker const &check, std::uint32_t events, hull::lane_layout layout,
                 hull::check_stats *stats = nullptr) -> std::vector<std::uint64_t> {
    const std::size_t lanes = std::size_t{1} << events;
    const std::size_t blocks = (lanes + 63) / 64;
    std::vector<std::uint64_t> words;
    std::size_t stride = 1;
    if (layout == hull::lane_layout::lane_major) {
        words.resize(lanes);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            words[lane] = lane;
        }
    } else {
        stride = blocks;
        words.assign(std::size_t{events} * stride, 0);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            for (std::uint32_t event = 0; event < events; ++event) {
                if (((lane >> event) & 1U) != 0) {
                    words[event * stride + lane / 64] |= std::uint64_t{1} << (lane % 64);
                }
            }
        }
    }
    std::vector<std::uint64_t> out(blocks, ~std::uint64_t{0});
    check.check(hull::lane_matrix{.words = words, .lane_count = lanes, .stride = stride, .layout = layout}, out, stats);
    return out;
}

} // namespace

TEST_CASE("hull check equals brute force on random circuits", "[ir][pc][hull]") {
    std::mt19937_64 random(0x5eed178);
    const auto configurations = option_sets();
    std::vector<hull::check_stats> totals(configurations.size());
    std::size_t splits = 0;
    for (int round = 0; round < 160; ++round) {
        const auto case_ = generate(random);
        std::vector<std::vector<event_mode>> mode_vectors{{},
                                                          random_modes(random, case_.events),
                                                          std::vector<event_mode>(case_.events, event_mode::fixed),
                                                          split_complemented(case_.model, case_.events)};
        for (auto &modes : mode_vectors) {
            limit_splits(modes, case_.events);
            const reference expected_value(case_, modes);
            splits += expected_value.split_count();
            const std::size_t lanes = std::size_t{1} << case_.events;
            std::vector<std::uint8_t> expected(lanes);
            for (std::size_t x = 0; x < lanes; ++x) {
                expected[x] = expected_value(static_cast<std::uint32_t>(x));
            }
            for (std::size_t index = 0; index < configurations.size(); ++index) {
                auto const &configuration = configurations[index];
                auto made = hull::checker::make(case_.model, case_.target, modes, configuration.options);
                REQUIRE(made);
                auto const &check = made.value();
                for (const auto layout : {hull::lane_layout::lane_major, hull::lane_layout::event_major}) {
                    const auto out = all_samples(check, case_.events, layout, &totals[index]);
                    for (std::size_t x = 0; x < lanes; ++x) {
                        const bool got = ((out[x / 64] >> (x % 64)) & 1U) != 0;
                        if (got != (expected[x] != 0)) {
                            FAIL("round " << round << " options " << configuration.name << " sample " << x
                                          << " expected " << int{expected[x]});
                        }
                    }
                    if (lanes % 64 != 0) {
                        REQUIRE((out.back() >> (lanes % 64)) == 0);
                    }
                }
                const auto x = static_cast<std::uint32_t>(random() % lanes);
                const std::array<std::uint64_t, 1> sample{x};
                REQUIRE(check.check(sample) == (expected[x] != 0));
            }
        }
    }
    REQUIRE(splits > 100);
    // Every tier must have decided some lanes.
    REQUIRE(totals[0].bounds > 0);
    REQUIRE(totals[0].reduction > 0);
    REQUIRE(totals[0].witness > 0);
    REQUIRE(totals[1].truth_table > 0);
    REQUIRE(totals[2].bdd > 0);
    REQUIRE(totals[3].search > 0);
    REQUIRE(totals[4].bdd_budget_exceeded > 0);
    for (auto const &total : totals) {
        REQUIRE(total.lanes ==
                total.bounds + total.reduction + total.witness + total.truth_table + total.bdd + total.search);
    }
}

TEST_CASE("hull check is independent of the thread count", "[ir][pc][hull]") {
    std::mt19937_64 random(7);
    for (int round = 0; round < 20; ++round) {
        const auto case_ = generate(random);
        hull::check_options serial{};
        serial.parallel = false;
        auto parallel_check = hull::checker::make(case_.model, case_.target);
        auto serial_check = hull::checker::make(case_.model, case_.target, {}, serial);
        REQUIRE(parallel_check);
        REQUIRE(serial_check);
        hull::check_stats parallel_stats;
        hull::check_stats serial_stats;
        const auto a =
            all_samples(parallel_check.value(), case_.events, hull::lane_layout::lane_major, &parallel_stats);
        const auto b = all_samples(serial_check.value(), case_.events, hull::lane_layout::lane_major, &serial_stats);
        REQUIRE(a == b);
        REQUIRE(parallel_stats.hits == serial_stats.hits);
        REQUIRE(parallel_stats.search == serial_stats.search);
        REQUIRE(parallel_stats.residual_variables_total == serial_stats.residual_variables_total);
    }
}

TEST_CASE("hull check rejects bad arguments", "[ir][pc][hull]") {
    pc::builder builder;
    const auto events = build({0.5, 0.5}, builder);
    const auto gate = builder.add_and({pos(events[0]), neg(events[1])});
    const auto model = seal(builder);
    REQUIRE_FALSE(hull::checker::make(model, pc::bool_node_id(99)));
    const std::vector<event_mode> short_modes{event_mode::down};
    REQUIRE_FALSE(hull::checker::make(model, gate, short_modes));
    REQUIRE_FALSE(hull::checker::make(pc::circuit{}, gate));
    auto made = hull::checker::make(model, gate);
    REQUIRE(made);
    REQUIRE(made.value().event_count() == 2);
    REQUIRE(made.value().cone_size() == 3);
    std::vector<std::uint64_t> out;
    const std::array<std::uint64_t, 1> words{0};
    REQUIRE_THROWS(made.value().check(hull::lane_matrix{.words = words, .lane_count = 1, .stride = 1}, out));
}

// docs/rfc/monotone-hull-queries.md §8; values are exact.
TEST_CASE("hull check reproduces the RFC worked examples", "[ir][pc][hull]") {
    SECTION("8.1 probability-1 event in a delete term") {
        pc::builder builder;
        const auto e = build({1.0, 0.1, 0.2}, builder);
        const auto f = builder.add_or({pos(e[0]), pos(builder.add_and({pos(e[1]), pos(e[2])}))});
        const auto g = builder.add_and({pos(f), neg(builder.add_and({pos(e[0]), pos(e[1])}))});
        const auto model = seal(builder);
        REQUIRE(hull_probability(model, g) == Catch::Approx(1.0).epsilon(1e-12));
        const std::vector<event_mode> fixed(3, event_mode::fixed);
        REQUIRE(hull_probability(model, g, fixed) == Catch::Approx(0.9).epsilon(1e-12));
    }
    SECTION("8.2 two-flag delete term") {
        pc::builder builder;
        const auto e = build({0.01, 0.02, 1.0, 1.0}, builder);
        const auto g = builder.add_and(
            {pos(builder.add_or({pos(e[0]), pos(e[1])})), neg(builder.add_and({pos(e[2]), pos(e[3])}))});
        const auto model = seal(builder);
        REQUIRE(hull_probability(model, g) == Catch::Approx(0.0298).epsilon(1e-12));
        const std::vector<event_mode> fixed(4, event_mode::fixed);
        REQUIRE(hull_probability(model, g, fixed) == Catch::Approx(0.0).margin(1e-15));
    }
    SECTION("8.3 exclusion between high-probability events") {
        pc::builder builder;
        const auto e = build({0.9, 0.9, 0.01, 0.01}, builder);
        const auto f = builder.add_or(
            {pos(builder.add_and({pos(e[0]), pos(e[2])})), pos(builder.add_and({pos(e[1]), pos(e[3])}))});
        const auto g = builder.add_and({pos(f), neg(builder.add_and({pos(e[0]), pos(e[1])}))});
        const auto model = seal(builder);
        REQUIRE(hull_probability(model, g) == Catch::Approx(0.017919).epsilon(1e-12));
        const std::vector<event_mode> fixed(4, event_mode::fixed);
        REQUIRE(hull_probability(model, g, fixed) == Catch::Approx(0.0018).epsilon(1e-12));
    }
    SECTION("8.4 a deletion that matters") {
        pc::builder builder;
        const auto e = build({0.5, 0.5, 0.1, 0.01}, builder);
        const auto ab = builder.add_and({pos(e[0]), pos(e[1])});
        const auto f = builder.add_or({pos(ab), pos(builder.add_and({pos(e[0]), pos(e[2])})), pos(e[3])});
        const auto g = builder.add_and({pos(f), neg(ab)});
        const auto model = seal(builder);
        REQUIRE(hull_probability(model, g) == Catch::Approx(0.0595).epsilon(1e-12));
        const std::vector<event_mode> fixed(4, event_mode::fixed);
        REQUIRE(hull_probability(model, g, fixed) == Catch::Approx(0.03225).epsilon(1e-12));
    }
    SECTION("8.5 two sequences of one event tree") {
        pc::builder builder;
        const auto e = build({0.01, 0.1, 0.2}, builder);
        const auto s1 = builder.add_and({pos(e[0]), pos(e[1]), neg(e[2])});
        const auto s2 = builder.add_and({pos(e[0]), pos(e[1]), pos(e[2])});
        const auto both = builder.add_or({pos(s1), pos(s2)});
        const auto model = seal(builder);
        REQUIRE(hull_probability(model, s1) == Catch::Approx(0.001).epsilon(1e-12));
        REQUIRE(hull_probability(model, s2) == Catch::Approx(0.0002).epsilon(1e-12));
        REQUIRE(hull_probability(model, both) == Catch::Approx(0.001).epsilon(1e-12));
    }
    SECTION("8.6 probability-0 event") {
        pc::builder builder;
        const auto e = build({0.1, 0.0, 0.5}, builder);
        const auto g = builder.add_and(
            {pos(builder.add_or({pos(e[0]), pos(e[1])})), neg(builder.add_and({pos(e[1]), pos(e[2])}))});
        const auto model = seal(builder);
        REQUIRE(hull_probability(model, g) == Catch::Approx(0.1).epsilon(1e-12));
        const std::vector<event_mode> up(3, event_mode::up);
        REQUIRE(hull_probability(model, g, up) == Catch::Approx(1.0).epsilon(1e-12));
    }
    SECTION("8.7 non-monotone module") {
        pc::builder builder;
        const auto e = build({0.5, 0.25, 0.1}, builder);
        const auto g = builder.add_or({pos(builder.add_and({pos(e[0]), neg(e[1])})), pos(e[2])});
        const auto model = seal(builder);
        REQUIRE(hull_probability(model, g) == Catch::Approx(0.55).epsilon(1e-12));
    }
}

TEST_CASE("hull check matches the closed forms of the RFC's examples", "[ir][pc][hull]") {
    SECTION("Rauzy: up(ab or not(a) c) = ab or c") {
        pc::builder builder;
        const auto e = build({0.5, 0.5, 0.5}, builder);
        const auto g = builder.add_or(
            {pos(builder.add_and({pos(e[0]), pos(e[1])})), pos(builder.add_and({neg(e[0]), pos(e[2])}))});
        const auto model = seal(builder);
        auto made = hull::checker::make(model, g);
        REQUIRE(made);
        for (std::uint64_t x = 0; x < 8; ++x) {
            const std::array<std::uint64_t, 1> sample{x};
            const bool expected = ((x & 3U) == 3U) || ((x & 4U) != 0);
            REQUIRE(made.value().check(sample) == expected);
        }
    }
    SECTION("KSC22 sequence 4: only {IE, B, E, F'} survives") {
        pc::builder builder;
        // IE A B C D E F' G
        const auto e = build({0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, builder);
        const auto f =
            builder.add_and({pos(e[0]), pos(builder.add_or({pos(e[3]), pos(builder.add_and({pos(e[2]), pos(e[5])}))})),
                             pos(builder.add_or({pos(e[6]), pos(builder.add_and({pos(e[4]), pos(e[7])}))}))});
        const auto s = builder.add_or({pos(e[1]), pos(e[3]), pos(builder.add_and({pos(e[2]), pos(e[4])}))});
        const auto g = builder.add_and({pos(f), neg(s)});
        const auto model = seal(builder);
        auto made = hull::checker::make(model, g);
        REQUIRE(made);
        constexpr std::uint64_t cut = (1U << 0) | (1U << 2) | (1U << 5) | (1U << 6);
        for (std::uint64_t x = 0; x < 256; ++x) {
            const std::array<std::uint64_t, 1> sample{x};
            REQUIRE(made.value().check(sample) == ((x & cut) == cut));
        }
    }
    SECTION("#190: a directly complemented event is a literal under mode fixed") {
        // process_multi_target T2 = (C or D) and not B; names_special 1-TOP.
        pc::builder builder;
        const auto e = build({0.1, 0.2, 0.3, 0.4}, builder);
        const auto t2 = builder.add_and({pos(builder.add_or({pos(e[2]), pos(e[3])})), neg(e[1])});
        const auto model = seal(builder);
        REQUIRE(hull_probability(model, t2) == Catch::Approx(0.58).epsilon(1e-12));
        const std::vector<event_mode> literal_b{event_mode::down, event_mode::fixed, event_mode::down,
                                                event_mode::down};
        REQUIRE(hull_probability(model, t2, literal_b) == Catch::Approx(0.464).epsilon(1e-12));

        pc::builder special;
        const auto s = build({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, special);
        const auto top = special.add_or({pos(special.add_and({pos(s[0]), pos(s[1]), neg(s[2])})),
                                         pos(special.add_cardinality(2, 3, {pos(s[3]), pos(s[4]), pos(s[5])}))});
        const auto special_model = seal(special);
        REQUIRE(hull_probability(special_model, top) == Catch::Approx(0.51).epsilon(1e-12));
        std::vector<event_mode> literal_6h(6, event_mode::down);
        literal_6h[2] = event_mode::fixed;
        REQUIRE(hull_probability(special_model, top, literal_6h) == Catch::Approx(0.507).epsilon(1e-12));
    }
}

// Theorem 8: CNF-SAT reduces to the check at x = 1_V, which leaves every event
// binate and free. Pigeonhole formulas are hard for resolution, so these
// exercise the complete search.
TEST_CASE("hull check decides the Theorem 8 reduction of pigeonhole formulas", "[ir][pc][hull]") {
    const auto pigeonhole = [](std::uint32_t pigeons, std::uint32_t holes) {
        pc::builder builder;
        const auto variables = pigeons * holes;
        std::vector<pc::bool_node_id> positive;
        std::vector<pc::bool_node_id> negative;
        for (std::uint32_t variable = 0; variable < variables; ++variable) {
            positive.push_back(builder.add_basic_event(builder.add_literal_probability(0.5)));
            negative.push_back(builder.add_basic_event(builder.add_literal_probability(0.5)));
        }
        const auto at = [holes](std::uint32_t pigeon, std::uint32_t hole) { return pigeon * holes + hole; };
        std::vector<pc::edge_ref> clauses;
        std::vector<pc::edge_ref> literals;
        for (std::uint32_t pigeon = 0; pigeon < pigeons; ++pigeon) {
            literals.clear();
            for (std::uint32_t hole = 0; hole < holes; ++hole) {
                literals.push_back(pos(positive[at(pigeon, hole)]));
            }
            clauses.push_back(pos(builder.add_or(literals)));
        }
        for (std::uint32_t hole = 0; hole < holes; ++hole) {
            for (std::uint32_t first = 0; first < pigeons; ++first) {
                for (std::uint32_t second = first + 1; second < pigeons; ++second) {
                    clauses.push_back(
                        pos(builder.add_or({pos(negative[at(first, hole)]), pos(negative[at(second, hole)])})));
                }
            }
        }
        std::vector<pc::edge_ref> clashes;
        for (std::uint32_t variable = 0; variable < variables; ++variable) {
            clauses.push_back(pos(builder.add_or({pos(positive[variable]), pos(negative[variable])})));
            clashes.push_back(pos(builder.add_and({pos(positive[variable]), pos(negative[variable])})));
        }
        const auto f = builder.add_and(clauses);
        const auto d = builder.add_or(clashes);
        const auto g = builder.add_and({pos(f), neg(d)});
        auto sealed = std::move(builder).seal();
        REQUIRE(sealed.ok());
        return std::pair{sealed.output, g};
    };
    for (const auto [pigeons, holes, satisfiable] :
         {std::tuple{4U, 4U, true}, std::tuple{5U, 4U, false}, std::tuple{6U, 5U, false}, std::tuple{3U, 4U, true}}) {
        const auto [model, target] = pigeonhole(pigeons, holes);
        auto made = hull::checker::make(model, target);
        REQUIRE(made);
        std::vector<std::uint64_t> all((made.value().event_count() + 63) / 64, ~std::uint64_t{0});
        hull::check_stats stats;
        std::uint64_t out = 0;
        made.value().check(hull::lane_matrix{.words = all, .lane_count = 1, .stride = all.size()},
                           std::span<std::uint64_t>{&out, 1}, &stats);
        REQUIRE((out == 1) == satisfiable);
        if (!satisfiable) {
            REQUIRE(stats.search + stats.bdd == 1);
        }
    }
}

// #190 decision: atom-split hull. Each case is checked for every sample.
TEST_CASE("hull check implements the atom-split hull", "[ir][pc][hull]") {
    const auto every_sample = [](pc::circuit const &model, pc::bool_node_id target,
                                 std::vector<event_mode> const &modes, std::uint32_t events, auto expected) {
        auto made = hull::checker::make(model, target, modes);
        REQUIRE(made);
        for (std::uint64_t x = 0; x < (1U << events); ++x) {
            const std::array<std::uint64_t, 1> sample{x};
            INFO("sample " << x);
            REQUIRE(made.value().check(sample) == expected(x));
        }
    };
    const auto bit = [](std::uint64_t x, int event) { return ((x >> event) & 1U) != 0; };
    SECTION("T = A not(B or E) or D not B with B split gives A or D not B") {
        // A B D E
        pc::builder builder;
        const auto e = build({0.5, 0.5, 0.5, 0.5}, builder);
        const auto t = builder.add_or({pos(builder.add_and({pos(e[0]), neg(builder.add_or({pos(e[1]), pos(e[3])}))})),
                                       pos(builder.add_and({pos(e[2]), neg(e[1])}))});
        const auto model = seal(builder);
        const std::vector<event_mode> modes{event_mode::down, event_mode::split, event_mode::down, event_mode::down};
        every_sample(model, t, modes, 4, [&](std::uint64_t x) { return bit(x, 0) || (bit(x, 2) && !bit(x, 1)); });
    }
    SECTION("A and not(A and not B) with B split gives A") {
        pc::builder builder;
        const auto e = build({0.5, 0.5}, builder);
        const auto g = builder.add_and({pos(e[0]), neg(builder.add_and({pos(e[0]), neg(e[1])}))});
        const auto model = seal(builder);
        const std::vector<event_mode> modes{event_mode::down, event_mode::split};
        every_sample(model, g, modes, 2, [&](std::uint64_t x) { return bit(x, 0); });
        // The plain hull differs: AB.
        every_sample(model, g, {}, 2, [&](std::uint64_t x) { return bit(x, 0) && bit(x, 1); });
    }
    SECTION("AB or C not B with B split has no consensus term AC") {
        // A B C
        pc::builder builder;
        const auto e = build({0.5, 0.5, 0.5}, builder);
        const auto g = builder.add_or(
            {pos(builder.add_and({pos(e[0]), pos(e[1])})), pos(builder.add_and({pos(e[2]), neg(e[1])}))});
        const auto model = seal(builder);
        const std::vector<event_mode> modes{event_mode::down, event_mode::split, event_mode::down};
        every_sample(model, g, modes, 3,
                     [&](std::uint64_t x) { return (bit(x, 0) && bit(x, 1)) || (bit(x, 2) && !bit(x, 1)); });
    }
    SECTION("probability of the #190 fixtures under split") {
        pc::builder builder;
        const auto e = build({0.1, 0.2, 0.3, 0.4}, builder);
        const auto t2 = builder.add_and({pos(builder.add_or({pos(e[2]), pos(e[3])})), neg(e[1])});
        const auto model = seal(builder);
        const std::vector<event_mode> modes{event_mode::down, event_mode::split, event_mode::down, event_mode::down};
        REQUIRE(hull_probability(model, t2, modes) == Catch::Approx(0.464).epsilon(1e-12));
    }
}
