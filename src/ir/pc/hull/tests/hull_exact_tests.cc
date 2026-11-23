#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "ir/pc/builder.h"
#include "ir/pc/edge.h"
#include "ir/pc/hull/exact.h"
#include "ir/pc/hull/hull.h"
#include "ir/pc/hull/tests/hull_test_support.h"

#include "api/options.h"
#include "compile/ftp/compile.h"
#include "compile/ftp/tests/support.h"
#include "io/ftp/descriptor.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/op_code.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <iterator>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

    namespace pc = fuzzybits::ir::pc;
    namespace hull = fuzzybits::ir::pc::hull;

    auto pos(pc::bool_node_id node) -> pc::edge_ref {
        return pc::edge_ref{.source = node};
    }

    auto neg(pc::bool_node_id node) -> pc::edge_ref {
        return pc::edge_ref{.source = node, .complement = true};
    }

    auto seal(pc::builder &builder) -> pc::circuit {
        auto sealed = std::move(builder).seal();
        REQUIRE(sealed.ok());
        return sealed.output;
    }

    auto build(std::initializer_list<double> probabilities, pc::builder &builder) -> std::vector<pc::bool_node_id> {
        std::vector<pc::bool_node_id> events;
        for (const auto probability : probabilities) {
            events.push_back(builder.add_basic_event(builder.add_literal_probability(probability)));
        }
        return events;
    }

    // Draws on the raw std::mt19937_64 output only: the standard distributions and
    // std::ranges::shuffle differ between libc++ and libstdc++.
    auto uniform(std::mt19937_64 &random, const std::uint32_t low, const std::uint32_t high) -> std::uint32_t {
        return low + static_cast<std::uint32_t>(random() % (std::uint64_t{high} - low + 1));
    }

    auto unit(std::mt19937_64 &random) -> double {
        return static_cast<double>(random() >> 11U) * 0x1.0p-53;
    }

    struct generated final {
        pc::circuit model;
        pc::bool_node_id target;
        std::uint32_t events = 0;
        std::vector<double> probability;
    };

    // A random circuit over at most 14 events with every op code, complemented
    // edges, constants, cardinality gates, probabilities 0, 1 and in between, and
    // sometimes a not-atleast(k+1; S) rule over a subset of the events, as the
    // initiator rule is lowered.
    auto generate(std::mt19937_64 &random) -> generated {
        const auto events = uniform(random, 1, 14);
        const auto gates = uniform(random, 1, 40);
        generated out;
        out.events = events;
        pc::builder builder;
        std::vector<pc::bool_node_id> nodes;
        std::vector<pc::bool_node_id> initiating;
        for (std::uint32_t event = 0; event < events; ++event) {
            const auto roll = uniform(random, 0, 99);
            const double probability = roll < 10 ? 0.0 : roll < 20 ? 1.0 : 0.05 + 0.9 * unit(random);
            out.probability.push_back(probability);
            nodes.push_back(builder.add_basic_event(builder.add_literal_probability(probability)));
            if (uniform(random, 0, 99) < 30) {
                initiating.push_back(nodes.back());
            }
        }
        if (uniform(random, 0, 99) < 30) {
            nodes.push_back(builder.add_constant(uniform(random, 0, 99) < 50));
        }
        if (uniform(random, 0, 99) < 30) {
            nodes.push_back(builder.add_house_event(uniform(random, 0, 99) < 50));
        }
        constexpr std::array kinds{pc::op_code::and_gate,  pc::op_code::or_gate,   pc::op_code::xor_gate,
                                   pc::op_code::not_gate,  pc::op_code::nand_gate, pc::op_code::nor_gate,
                                   pc::op_code::xnor_gate, pc::op_code::cardinality};
        std::vector<pc::edge_ref> operands;
        for (std::uint32_t gate = 0; gate < gates; ++gate) {
            const auto kind = kinds[uniform(random, 0, static_cast<std::uint32_t>(kinds.size()) - 1)];
            const auto arity = kind == pc::op_code::not_gate ? 1U : uniform(random, 1, 5);
            operands.clear();
            for (std::uint32_t operand = 0; operand < arity; ++operand) {
                const auto span = static_cast<std::uint32_t>(nodes.size());
                const auto recent = std::min<std::uint32_t>(span, 8);
                const auto index = uniform(random, 0, 99) < 60 ? span - 1 - uniform(random, 0, recent - 1)
                                                               : uniform(random, 0, span - 1);
                operands.push_back(pc::edge_ref{.source = nodes[index], .complement = uniform(random, 0, 99) < 30});
            }
            if (kind == pc::op_code::cardinality) {
                const auto low = uniform(random, 0, arity);
                const auto high = uniform(random, low, arity);
                nodes.push_back(builder.add_cardinality(low, high, operands));
            } else {
                nodes.push_back(builder.add_gate(kind, operands));
            }
        }
        out.target = uniform(random, 0, 99) < 90
                         ? nodes.back()
                         : nodes[uniform(random, 0, static_cast<std::uint32_t>(nodes.size()) - 1)];
        if (initiating.size() >= 2 && uniform(random, 0, 2) == 0) {
            std::vector<pc::edge_ref> rule;
            for (const auto event : initiating) {
                rule.push_back(pos(event));
            }
            const auto keep = uniform(random, 0, static_cast<std::uint32_t>(initiating.size()) - 1);
            const auto count = static_cast<std::uint32_t>(initiating.size());
            const auto atleast = builder.add_cardinality(std::min(keep + 1, count), count, rule);
            out.target = builder.add_and({pos(out.target), neg(atleast)});
        }
        (void)builder.add_target(out.target);
        out.model = seal(builder);
        return out;
    }

    // Σ_x P(x)·[check(x)] over every sample, with Neumaier summation.
    auto enumerated_probability(generated const &case_, hull::checker const &check) -> double {
        const auto lanes = std::size_t{1} << case_.events;
        std::vector<double> weight(lanes, 1.0);
        for (std::uint32_t bit = 0; bit < case_.events; ++bit) {
            const auto p = case_.probability[bit];
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                weight[lane] *= ((lane >> bit) & 1U) != 0 ? p : 1.0 - p;
            }
        }
        double total = 0.0;
        double compensation = 0.0;
        std::vector<std::uint64_t> sample(1, 0);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            sample[0] = lane;
            if (!check.check(sample)) {
                continue;
            }
            const auto term = weight[lane];
            const auto next = total + term;
            compensation += std::abs(total) >= std::abs(term) ? (total - next) + term : (term - next) + total;
            total = next;
        }
        return total + compensation;
    }

    auto as_sample(const std::uint32_t mask) -> std::vector<std::uint64_t> {
        return std::vector<std::uint64_t>{mask};
    }

    // A sample whose true cone events are exactly `events`.
    auto sample_of(std::span<std::uint32_t const> events, const std::uint32_t width) -> std::vector<std::uint64_t> {
        std::vector<std::uint64_t> sample((width + 63) / 64, 0);
        for (const auto event : events) {
            sample[event / 64] |= std::uint64_t{1} << (event % 64);
        }
        return sample;
    }

} // namespace

TEST_CASE("exact hull probability equals the enumerated checker hull", "[ir][pc][hull][exact]") {
    std::mt19937_64 random(0x5eed202);
    for (int round = 0; round < 120; ++round) {
        const auto case_ = generate(random);
        auto made = hull::checker::make(case_.model, case_.target);
        REQUIRE(made);
        const auto expected = enumerated_probability(case_, made.value());
        auto exact = hull::exact_hull_probability(case_.model, case_.target);
        INFO("round " << round << " events " << case_.events << " expected " << expected);
        REQUIRE(exact);
        REQUIRE(exact.value() == Catch::Approx(expected).epsilon(1e-12).margin(1e-15));
    }
}

TEST_CASE("trace follows the check on every sample and its bounds sandwich the answer", "[ir][pc][hull][trace]") {
    std::mt19937_64 random(0x7ea0e);
    for (int round = 0; round < 24; ++round) {
        const auto case_ = generate(random);
        auto made = hull::checker::make(case_.model, case_.target);
        REQUIRE(made);
        auto const &check = made.value();
        // Every sample up to 12 events; above that a random spread, since `trace`
        // rebuilds a workspace per lane.
        std::vector<std::uint32_t> lanes;
        const auto total = 1U << case_.events;
        if (case_.events <= 12) {
            for (std::uint32_t x = 0; x < total; ++x) {
                lanes.push_back(x);
            }
        } else {
            for (int draw = 0; draw < 256; ++draw) {
                lanes.push_back(uniform(random, 0, total - 1));
            }
        }
        for (const auto x : lanes) {
            const auto sample = as_sample(x);
            const auto trace = check.trace(sample);
            INFO("round " << round << " sample " << x);
            REQUIRE(trace.hit == check.check(sample));
            REQUIRE(trace.lower <= trace.hit);
            REQUIRE(trace.hit <= trace.upper);
        }
    }
}

TEST_CASE("a witness is a minimal true subset of the sample's cone", "[ir][pc][hull][witness]") {
    std::mt19937_64 random(0x11e55);
    std::size_t hits = 0;
    std::size_t empty_witnesses = 0;
    std::size_t brute_forced = 0;
    for (int round = 0; round < 40; ++round) {
        const auto case_ = generate(random);
        auto made = hull::checker::make(case_.model, case_.target);
        REQUIRE(made);
        auto const &check = made.value();
        const auto cone = check.cone_events();
        std::vector<std::uint32_t> lanes;
        const auto total = 1U << case_.events;
        if (case_.events <= 10) {
            for (std::uint32_t x = 0; x < total; ++x) {
                lanes.push_back(x);
            }
        } else {
            for (int draw = 0; draw < 128; ++draw) {
                lanes.push_back(uniform(random, 0, total - 1));
            }
        }
        for (const auto lane : lanes) {
            const auto sample = as_sample(lane);
            auto found = check.witness(sample);
            REQUIRE(found);
            auto const &witness = found.value();
            INFO("round " << round << " sample " << lane);
            if (!check.check(sample)) {
                REQUIRE_FALSE(witness);
                continue;
            }
            ++hits;
            REQUIRE(witness);
            const std::vector<std::uint32_t> events(witness->begin(), witness->end());
            REQUIRE(std::ranges::is_sorted(events));
            for (const auto event : events) {
                REQUIRE((lane >> event & 1U) != 0);
                REQUIRE(std::ranges::find(cone, event) != cone.end());
            }
            REQUIRE(check.check(sample_of(events, check.event_count())));
            for (std::size_t removed = 0; removed < events.size(); ++removed) {
                auto shorter = events;
                shorter.erase(shorter.begin() + static_cast<std::ptrdiff_t>(removed));
                REQUIRE_FALSE(check.check(sample_of(shorter, check.event_count())));
            }
            if (events.empty()) {
                ++empty_witnesses;
            }
            if (case_.events <= 10) {
                ++brute_forced;
                // No proper subset of the witness is a hit.
                for (std::uint32_t subset = 0; subset + 1 < (1U << events.size()); ++subset) {
                    std::vector<std::uint32_t> proper;
                    for (std::uint32_t bit = 0; bit < events.size(); ++bit) {
                        if (((subset >> bit) & 1U) != 0) {
                            proper.push_back(events[bit]);
                        }
                    }
                    INFO("subset " << subset);
                    REQUIRE_FALSE(check.check(sample_of(proper, check.event_count())));
                }
            }
        }
    }
    REQUIRE(hits > 100);
    REQUIRE(brute_forced > 100);
    // g(∅) can be TRUE: an empty witness is a hit, not a miss.
    REQUIRE(empty_witnesses > 0);
}

TEST_CASE("a witness is refused for a checker whose cone is not all down", "[ir][pc][hull][witness]") {
    pc::builder builder;
    const auto e = build({0.5, 0.5}, builder);
    const auto g = builder.add_and({pos(e[0]), neg(e[1])});
    (void)builder.add_target(g);
    const auto model = seal(builder);

    auto down = hull::checker::make(model, g);
    REQUIRE(down);
    CHECK(down.value().witness(as_sample(0)).has_value());

    const std::vector<hull::event_mode> modes{hull::event_mode::fixed, hull::event_mode::down};
    auto fixed = hull::checker::make(model, g, modes);
    REQUIRE(fixed);
    auto refused = fixed.value().witness(as_sample(3));
    REQUIRE_FALSE(refused);
    REQUIRE(refused.diagnostics().has_errors());
    CHECK(refused.diagnostics().items().front().code == "pc.hull.witness-mode");

    // A mode on an event outside the target's cone is not a restriction.
    auto outside = hull::checker::make(model, e[0], std::vector<hull::event_mode>{hull::event_mode::down,
                                                                                  hull::event_mode::up});
    REQUIRE(outside);
    CHECK(outside.value().witness(as_sample(1)).has_value());
}

TEST_CASE("trace names the tier that decided the lane", "[ir][pc][hull][trace]") {
    SECTION("the bounds decide an event") {
        pc::builder builder;
        const auto e = build({0.5}, builder);
        (void)builder.add_target(e[0]);
        const auto model = seal(builder);
        auto made = hull::checker::make(model, e[0]);
        REQUIRE(made);
        const auto hit = made.value().trace(as_sample(1));
        CHECK(hit.hit);
        CHECK(hit.tier == hull::check_tier::bounds);
        CHECK(hit.residual_variables == 0);
        const auto miss = made.value().trace(as_sample(0));
        CHECK_FALSE(miss.hit);
        CHECK(miss.tier == hull::check_tier::bounds);
    }
    SECTION("a binate event leaves the bounds open and a later tier decides") {
        pc::builder builder;
        const auto e = build({0.5, 0.5}, builder);
        const auto g = builder.add_gate(pc::op_code::xor_gate, {pos(e[0]), pos(e[1])});
        (void)builder.add_target(g);
        const auto model = seal(builder);
        auto made = hull::checker::make(model, g);
        REQUIRE(made);
        // x = {A}: B is forced to 0, and A may be 1, so the lane is a hit the
        // bounds cannot see.
        const auto trace = made.value().trace(as_sample(1));
        CHECK(trace.hit);
        CHECK_FALSE(trace.lower);
        CHECK(trace.upper);
        CHECK(trace.tier != hull::check_tier::bounds);
        CHECK(trace.residual_variables == 0);
        // x = ∅: nothing can be true.
        const auto empty = made.value().trace(as_sample(0));
        CHECK_FALSE(empty.hit);
        CHECK_FALSE(empty.upper);
    }
}

TEST_CASE("exact hull keeps a probability-1 event existential", "[ir][pc][hull][exact]") {
    // The initiator rule as `fzb compile` lowers it, with a probability-1 initiator that
    // is not in any cut set: g = I2 AND X AND NOT atleast(2; ONE, I2). Under the hull
    // y_ONE may be 0, so ↑g = I2 AND X and P = 0.3 · 0.5. Substituting ONE by TRUE turns
    // the rule into I2 and leaves g identically FALSE, which is the value the test
    // asserts against.
    const auto make_circuit = [](const bool one_as_true) {
        pc::builder builder;
        const auto one = one_as_true ? builder.add_constant(true)
                                     : builder.add_basic_event(builder.add_literal_probability(1.0));
        const auto second = builder.add_basic_event(builder.add_literal_probability(0.3));
        const auto other = builder.add_basic_event(builder.add_literal_probability(0.5));
        const auto sequence = builder.add_and({pos(second), pos(other)});
        const auto rule = builder.add_cardinality(2, 2, {pos(one), pos(second)});
        const auto g = builder.add_and({pos(sequence), neg(rule)});
        (void)builder.add_target(g);
        return std::pair{seal(builder), g};
    };
    const auto [model, target] = make_circuit(false);
    auto made = hull::checker::make(model, target);
    REQUIRE(made);
    const auto expected = enumerated_probability(generated{.model = model, .target = target, .events = 3,
                                                           .probability = {1.0, 0.3, 0.5}},
                                                 made.value());
    REQUIRE(expected == Catch::Approx(0.15).epsilon(1e-12));

    auto exact = hull::exact_hull_probability(model, target);
    REQUIRE(exact);
    CHECK(exact.value() == Catch::Approx(expected).epsilon(1e-12));

    const auto [substituted, substituted_target] = make_circuit(true);
    auto replaced = hull::exact_hull_probability(substituted, substituted_target);
    REQUIRE(replaced);
    INFO("substituting the probability-1 event by TRUE gives " << replaced.value());
    CHECK(replaced.value() == Catch::Approx(0.0).margin(1e-15));
    CHECK_FALSE(exact.value() == Catch::Approx(replaced.value()).epsilon(1e-6));
}

TEST_CASE("exact hull limits the random events of the cone", "[ir][pc][hull][exact]") {
    const auto or_of = [](const std::uint32_t random, const std::uint32_t zero) {
        pc::builder builder;
        std::vector<pc::bool_node_id> events;
        for (std::uint32_t index = 0; index < random; ++index) {
            events.push_back(builder.add_basic_event(builder.add_literal_probability(0.5)));
        }
        for (std::uint32_t index = 0; index < zero; ++index) {
            events.push_back(builder.add_basic_event(builder.add_literal_probability(0.0)));
        }
        std::vector<pc::edge_ref> operands;
        for (const auto event : events) {
            operands.push_back(pos(event));
        }
        const auto gate = builder.add_or(operands);
        (void)builder.add_target(gate);
        return std::pair{seal(builder), gate};
    };

    {
        const auto [model, gate] = or_of(hull::kMaxExactHullEvents + 1, 0);
        auto refused = hull::exact_hull_probability(model, gate);
        REQUIRE_FALSE(refused);
        REQUIRE(refused.diagnostics().has_errors());
        const auto &item = refused.diagnostics().items().front();
        CHECK(item.code == "pc.hull.exact-too-many-events");
        CHECK(item.message.find(std::to_string(hull::kMaxExactHullEvents + 1)) != std::string::npos);
        CHECK(item.message.find(std::to_string(hull::kMaxExactHullEvents)) != std::string::npos);
    }
    // A probability-0 event is FALSE in every hull world, so it is not an axis: the
    // hull is the OR of the 24 remaining events, 1 - 2^-24. (Enumerating 2^25 lanes
    // through the checker would take a minute for the same number.)
    {
        const auto [model, gate] = or_of(hull::kMaxExactHullEvents, 1);
        auto exact = hull::exact_hull_probability(model, gate);
        REQUIRE(exact);
        const auto expected = 1.0 - std::exp2(-static_cast<double>(hull::kMaxExactHullEvents));
        CHECK(exact.value() == Catch::Approx(expected).epsilon(1e-12));
    }
}

TEST_CASE("check_nodes indexes the lane by event ordinal, not node id", "[ir][pc][hull][witness]") {
    // A gate before the events, so node ids and event ordinals differ: without the
    // remap, the sample would set the wrong lane bits.
    pc::builder builder;
    const auto seed = builder.add_basic_event(builder.add_literal_probability(0.5));
    const auto early = builder.add_or({pos(seed)});
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.5));
    const auto b = builder.add_basic_event(builder.add_literal_probability(0.5));
    const auto g = builder.add_and({pos(early), pos(a), pos(b)});
    (void)builder.add_target(g);
    const auto model = seal(builder);

    auto made = hull::checker::make(model, g);
    REQUIRE(made);
    auto const &check = made.value();
    // Node ids 1, 2 and 3 are events; node 0 is a gate and node 4 the target.
    REQUIRE(check.event_count() == 3);
    CHECK(check.event_index(seed) == 0);
    CHECK(check.event_index(a) == 1);
    CHECK(check.event_index(b) == 2);
    CHECK(check.event_index(g) == std::numeric_limits<std::uint32_t>::max());
    CHECK(check.event_node(2) == b);

    const std::array<pc::bool_node_id, 3> all{seed, a, b};
    auto checked = check.check_nodes(all, true);
    REQUIRE(checked);
    CHECK(checked.value().hit);
    CHECK(checked.value().witness.has_value());
    CHECK(*checked.value().witness == std::vector<std::uint32_t>{0, 1, 2});

    // Only the later events: the seed is false, so the early OR is false.
    const std::array<pc::bool_node_id, 2> later{a, b};
    auto missed = check.check_nodes(later);
    REQUIRE(missed);
    CHECK_FALSE(missed.value().hit);

    const std::array<pc::bool_node_id, 2> with_gate{g, a};
    auto refused = check.check_nodes(with_gate);
    REQUIRE_FALSE(refused);
    CHECK(refused.diagnostics().items().front().code == "pc.hull.not-an-event");
}

TEST_CASE("exact hull rejects bad arguments", "[ir][pc][hull][exact]") {
    pc::builder builder;
    const auto events = build({0.5}, builder);
    (void)builder.add_target(events[0]);
    const auto model = seal(builder);
    CHECK_FALSE(hull::exact_hull_probability(pc::circuit{}, events[0]));
    CHECK_FALSE(hull::exact_hull_probability(model, pc::bool_node_id(99)));
    auto exact = hull::exact_hull_probability(model, events[0]);
    REQUIRE(exact);
    CHECK(exact.value() == Catch::Approx(0.5).epsilon(1e-15));
}
namespace {

    namespace ftp = fuzzybits::compile::ftp;

    auto read_text(std::filesystem::path const &path) -> std::string {
        std::ifstream input(path, std::ios::binary);
        REQUIRE(input);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    auto has_code(fuzzybits::api::diagnostic_list const &diagnostics, std::string_view code) -> bool {
        return std::ranges::any_of(diagnostics.items(), [&](auto const &item) { return item.code == code; });
    }

    template <typename value_type>
    auto diagnostics_of(fuzzybits::api::result<value_type> const &result) -> std::string {
        std::string out;
        for (auto const &item : result.diagnostics().items()) {
            out += item.code + ": " + item.message + "; ";
        }
        return out;
    }

} // namespace

// docs/rfc/monotone-hull-queries.md §6.5 on the conformance fixtures: the exact hull
// of the written circuit is the reference tools' `hull` (atom-split, the default) or
// `hull_positive` (--literals negation) value.
TEST_CASE("the exact hull reproduces the FTP conformance hull values", "[ir][pc][hull][ftp][exact]") {
    const auto root = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "conformance";
    std::vector<std::filesystem::path> fixtures;
    for (auto const &entry : std::filesystem::directory_iterator(root)) {
        if (std::filesystem::exists(entry.path() / "expected.json")) {
            fixtures.push_back(entry.path());
        }
    }
    std::ranges::sort(fixtures);
    REQUIRE(fixtures.size() >= 9);

    std::size_t compared = 0;
    std::size_t refused = 0;
    for (const auto &fixture : fixtures) {
        const auto expected = hull::testing::json_reader(read_text(fixture / "expected.json")).parse();
        std::vector<fuzzybits::io::backend::fuzzybits::ftp::source_text> sources;
        for (auto const &file : expected.find("files")->items) {
            const auto path = fixture / file.text;
            sources.push_back(fuzzybits::io::backend::fuzzybits::ftp::source_text{
                .path = path.generic_string(), .text = read_text(path)});
        }
        auto const *results = expected.find("policies")->find("last")->find("results");
        REQUIRE(results != nullptr);
        if (results->type == hull::testing::json::kind::null) {
            continue;
        }
        REQUIRE(results->type == hull::testing::json::kind::object);
        for (std::size_t member = 0; member < results->keys.size(); ++member) {
            auto const &target = results->keys[member];
            auto const &values = results->items[member];
            INFO(fixture.filename().string() << " / " << target);
            // The default literal mode is atom_split, which pcfb 1.2 cannot carry for a
            // cone with marked operands: those targets are refused, and the value they
            // would have is `hull` (the atom-split hull) of the reference tools.
            // No initiator rule: expected.json's `hull` is the plain value, and the
            // `variants` hold the rule's.
            auto atom_split = ftp_lowering_test::boolean_options({target});
            atom_split.negation = "hull";
            atom_split.literals = "atom_split";
            auto compiled = ftp::compile_sources(sources, atom_split);
            if (!compiled) {
                REQUIRE(has_code(compiled.diagnostics(), "compile.ftp.hull-literals-unsupported"));
                REQUIRE_FALSE(values.find("complemented_events")->items.empty());
                ++refused;
                continue;
            }

            const auto hull_of = [&](fuzzybits::io::backend::fuzzybits::pcfb::read_output const &file) -> double {
                REQUIRE(file.queries.size() == 1);
                REQUIRE(file.queries.front().kind ==
                        fuzzybits::io::backend::fuzzybits::pcfb::query_kind::monotone_hull_probability);
                const auto target_node = file.circuit.targets()[file.queries.front().target.raw()].node;
                auto exact = hull::exact_hull_probability(file.circuit, target_node);
                INFO(diagnostics_of(exact));
                REQUIRE(exact);
                return exact.value();
            };

            auto read = fuzzybits::io::backend::fuzzybits::pcfb::read(compiled.value().bytes);
            REQUIRE(read);
            const auto atom_split_value = hull_of(read.value());
            CHECK(atom_split_value == Catch::Approx(values.find("hull")->number).epsilon(1e-12));
            ++compared;

            if (values.find("hull_positive") == nullptr) {
                continue;
            }
            auto negation = atom_split;
            negation.literals = "negation";
            auto plain = ftp::compile_sources(sources, negation);
            INFO(diagnostics_of(plain));
            REQUIRE(plain);
            auto plain_read = fuzzybits::io::backend::fuzzybits::pcfb::read(plain.value().bytes);
            REQUIRE(plain_read);
            const auto plain_value = hull_of(plain_read.value());
            CHECK(plain_value == Catch::Approx(values.find("hull_positive")->number).epsilon(1e-12));
            ++compared;
        }
    }
    REQUIRE(compared >= 14);
    REQUIRE(refused > 0);
}

// Exact-hull cost at the limit: about 16 million table words at 24 events.
TEST_CASE("the exact hull runs within its time budget at 20 and 24 events", "[ir][pc][hull][exact][.]") {
    const auto chain = [](const std::uint32_t events) {
        pc::builder builder;
        std::vector<pc::bool_node_id> all;
        for (std::uint32_t index = 0; index < events; ++index) {
            all.push_back(builder.add_basic_event(builder.add_literal_probability(0.3)));
        }
        std::vector<pc::edge_ref> first;
        std::vector<pc::edge_ref> second;
        for (std::uint32_t index = 0; index < events; ++index) {
            (index % 3 == 0 ? first : second).push_back(pos(all[index]));
        }
        const auto left = builder.add_or(first);
        const auto right = builder.add_gate(pc::op_code::xor_gate, second);
        const auto top = builder.add_and({pos(left), pos(builder.add_or({pos(right), pos(all[1])}))});
        (void)builder.add_target(top);
        return std::pair{seal(builder), top};
    };
    for (const auto events : {20U, 24U}) {
        const auto [model, top] = chain(events);
        const auto start = std::chrono::steady_clock::now();
        auto exact = hull::exact_hull_probability(model, top);
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        REQUIRE(exact);
        WARN("exact hull at " << events << " events: " << seconds << " s, p = " << exact.value());
    }
}
