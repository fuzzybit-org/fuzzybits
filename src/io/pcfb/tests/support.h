#pragma once

#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"
#include "ir/pc/target.h"
#include "portable_random.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace pcfb_writer_test {

namespace pc = fuzzybits::ir::pc;

[[nodiscard]] inline auto diagnosticsText(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
    std::ostringstream output;
    for (auto const &diagnostic : diagnostics.items()) {
        output << diagnostic.code << ": " << diagnostic.message << '\n';
    }
    return output.str();
}

[[nodiscard]] inline auto hasCode(fuzzybits::api::diagnostic_list const &diagnostics, std::string const &code) -> bool {
    return std::ranges::any_of(diagnostics.items(), [&](auto const &diagnostic) { return diagnostic.code == code; });
}

[[nodiscard]] inline auto requireWrite(pc::circuit const &model,
                                       fuzzybits::io::backend::fuzzybits::pcfb::write_options const &options = {})
    -> fuzzybits::io::backend::fuzzybits::pcfb::write_output {
    auto written = fuzzybits::io::backend::fuzzybits::pcfb::write(model.to_handle(), options);
    INFO(diagnosticsText(written.diagnostics()));
    REQUIRE(written);
    return std::move(written).value();
}

// Kahn order over the operand relation, deliberately not the writer's DFS.
[[nodiscard]] inline auto referenceOrder(pc::circuit const &model) -> std::vector<std::uint32_t> {
    const auto nodeCount = model.node_count();
    std::vector<std::uint32_t> pending(nodeCount, 0);
    std::vector<std::vector<std::uint32_t>> users(nodeCount);
    for (std::uint32_t node = 0; node < nodeCount; ++node) {
        for (auto const &edge : model.operands(pc::bool_node_id(node))) {
            users[edge.source.raw()].push_back(node);
            ++pending[node];
        }
    }
    std::vector<std::uint32_t> order;
    order.reserve(nodeCount);
    for (std::uint32_t node = 0; node < nodeCount; ++node) {
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
    REQUIRE(order.size() == nodeCount);
    return order;
}

// Exhaustive P(target) for every target under ir::pc semantics: every
// basic-event node is an independent Bernoulli variable (also when two nodes
// share a probability source) and k-of-n counts operands with multiplicity.
[[nodiscard]] inline auto referenceProbabilities(pc::circuit const &model) -> std::vector<double> {
    const auto order = referenceOrder(model);
    std::vector<std::uint32_t> events;
    std::vector<double> probability;
    for (std::uint32_t node = 0; node < model.node_count(); ++node) {
        if (model.op_codes()[node] == pc::op_code::basic_event) {
            events.push_back(node);
            probability.push_back(model.probability_sources()[model.probability_source_refs()[node].raw()].literal);
        }
    }
    REQUIRE(events.size() <= 20);

    const auto targets = model.targets();
    std::vector<double> sum(targets.size(), 0.0);
    std::vector<double> compensation(targets.size(), 0.0);
    std::vector<std::uint8_t> value(model.node_count(), 0);
    for (std::uint64_t mask = 0; mask < (std::uint64_t{1} << events.size()); ++mask) {
        double mass = 1.0;
        for (std::size_t event = 0; event < events.size(); ++event) {
            const bool bit = ((mask >> event) & 1U) != 0;
            value[events[event]] = bit ? 1 : 0;
            mass *= bit ? probability[event] : 1.0 - probability[event];
        }
        for (const auto node : order) {
            const auto code = model.op_codes()[node];
            if (code == pc::op_code::basic_event) {
                continue;
            }
            if (code == pc::op_code::constant || code == pc::op_code::house_event) {
                value[node] = model.constant_values()[node];
                continue;
            }
            const auto operands = model.operands(pc::bool_node_id(node));
            std::uint32_t trues = 0;
            for (auto const &[source, complement] : operands) {
                trues += (value[source.raw()] != 0) != complement ? 1U : 0U;
            }
            const auto fanIn = static_cast<std::uint32_t>(operands.size());
            bool result = false;
            switch (code) {
            case pc::op_code::and_gate:
                result = trues == fanIn;
                break;
            case pc::op_code::or_gate:
                result = trues != 0;
                break;
            case pc::op_code::xor_gate:
                result = (trues & 1U) != 0;
                break;
            case pc::op_code::not_gate:
                result = trues == 0;
                break;
            case pc::op_code::nand_gate:
                result = trues != fanIn;
                break;
            case pc::op_code::nor_gate:
                result = trues == 0;
                break;
            case pc::op_code::xnor_gate:
                result = (trues & 1U) == 0;
                break;
            case pc::op_code::cardinality:
                result = trues >= model.cardinality_low()[node] && trues <= model.cardinality_high()[node];
                break;
            case pc::op_code::basic_event:
            case pc::op_code::constant:
            case pc::op_code::house_event:
                break;
            }
            value[node] = result ? 1 : 0;
        }
        for (std::size_t target = 0; target < targets.size(); ++target) {
            if (value[targets[target].node.raw()] == 0) {
                continue;
            }
            const auto total = sum[target] + mass;
            compensation[target] +=
                std::abs(sum[target]) >= mass ? (sum[target] - total) + mass : (mass - total) + sum[target];
            sum[target] = total;
        }
    }
    for (std::size_t target = 0; target < targets.size(); ++target) {
        sum[target] += compensation[target];
    }
    return sum;
}

enum class CardinalityForm : std::uint8_t {
    ones,
    at_least,
    at_most,
    exactly,
    range,
};

[[nodiscard]] inline auto cardinalityForm(const std::uint32_t low, const std::uint32_t high, const std::uint32_t fanIn)
    -> CardinalityForm {
    if (low == 0 && high == fanIn) {
        return CardinalityForm::ones;
    }
    if (high == fanIn) {
        return CardinalityForm::at_least;
    }
    if (low == 0) {
        return CardinalityForm::at_most;
    }
    return low == high ? CardinalityForm::exactly : CardinalityForm::range;
}

struct Coverage final {
    std::array<std::size_t, 11> op_codes{};
    std::array<std::size_t, 5> cardinality_forms{};
    std::size_t complemented_edges = 0;
    std::size_t mixed_polarity_gates = 0;
    std::size_t duplicate_operands = 0;
    std::size_t shared_nodes = 0;
    std::size_t shared_probability_sources = 0;
    std::size_t non_topological_circuits = 0;
    std::size_t extreme_probabilities = 0;
    std::size_t multi_target_circuits = 0;
    std::size_t sixteen_event_circuits = 0;
};

struct RandomSize final {
    std::uint32_t max_events = 16;
    std::uint32_t max_gates = 30;
};

// Random PC circuit with 1..16 basic events (by default), every opcode, complements, every
// cardinality form, constants, house events, shared nodes and shared
// probability sources. Node ids are a random permutation of a topological
// order, so most circuits are not topologically numbered.
[[nodiscard]] inline auto randomCircuit(std::mt19937_64 &rng, Coverage &coverage, const RandomSize size = {})
    -> pc::finalize_result {
    struct Logical final {
        pc::op_code code = pc::op_code::basic_event;
        std::vector<std::pair<std::uint32_t, bool>> operands;
        std::uint32_t low = 0;
        std::uint32_t high = 0;
        bool value = false;
        std::uint32_t source = 0;
    };
    const auto uniform = [&](const std::uint32_t low, const std::uint32_t high) {
        return static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, low, high));
    };
    const auto chance = [&](const double p) { return pcfb_test_random::chance(rng, p); };

    const auto eventCount = uniform(1, size.max_events);
    const auto constantCount = uniform(0, 2);
    const auto houseCount = uniform(0, 2);
    const auto gateCount = uniform(1, size.max_gates);
    coverage.sixteen_event_circuits += eventCount == 16 ? 1 : 0;

    std::vector<double> probabilities;
    std::vector<Logical> nodes;
    for (std::uint32_t event = 0; event < eventCount; ++event) {
        Logical node;
        if (event > 0 && chance(0.1)) {
            node.source = uniform(0, static_cast<std::uint32_t>(probabilities.size()) - 1);
            ++coverage.shared_probability_sources;
        } else {
            const auto draw = pcfb_test_random::unit(rng);
            double p = pcfb_test_random::real(rng, 0.01, 0.99);
            if (draw < 0.05) {
                p = 0.0;
            } else if (draw < 0.1) {
                p = 1.0;
            } else if (draw < 0.2) {
                p = 0.5;
            } else if (draw < 0.3) {
                p = 1.0e-6 * pcfb_test_random::unit(rng);
            }
            coverage.extreme_probabilities += p == 0.0 || p == 1.0 ? 1 : 0;
            node.source = static_cast<std::uint32_t>(probabilities.size());
            probabilities.push_back(p);
        }
        nodes.push_back(node);
    }
    for (std::uint32_t index = 0; index < constantCount + houseCount; ++index) {
        nodes.push_back(Logical{.code = index < constantCount ? pc::op_code::constant : pc::op_code::house_event,
                                .value = chance(0.5)});
    }
    pcfb_test_random::shuffle(std::span{nodes}, rng);
    const auto leafCount = static_cast<std::uint32_t>(nodes.size());

    constexpr std::array gateCodes{
        pc::op_code::and_gate,  pc::op_code::or_gate,  pc::op_code::xor_gate,  pc::op_code::not_gate,
        pc::op_code::nand_gate, pc::op_code::nor_gate, pc::op_code::xnor_gate, pc::op_code::cardinality,
    };
    for (std::uint32_t gate = 0; gate < gateCount; ++gate) {
        const auto available = static_cast<std::uint32_t>(nodes.size());
        Logical node{.code = gateCodes[uniform(0, static_cast<std::uint32_t>(gateCodes.size()) - 1)]};
        const auto fanIn = node.code == pc::op_code::not_gate ? 1U : uniform(1, 5);
        for (std::uint32_t operand = 0; operand < fanIn; ++operand) {
            const auto recent = std::min<std::uint32_t>(available, 5);
            const auto source = chance(0.5) ? available - 1 - uniform(0, recent - 1) : uniform(0, available - 1);
            node.operands.emplace_back(source, chance(0.35));
        }
        if (node.code != pc::op_code::not_gate && chance(0.2)) {
            auto const repeated = node.operands[uniform(0, static_cast<std::uint32_t>(node.operands.size()) - 1)];
            const bool flip = chance(0.5);
            node.operands.emplace_back(repeated.first, repeated.second != flip);
            ++(flip ? coverage.mixed_polarity_gates : coverage.duplicate_operands);
        }
        if (node.code == pc::op_code::cardinality) {
            const auto n = static_cast<std::uint32_t>(node.operands.size());
            std::vector<CardinalityForm> feasible{CardinalityForm::ones, CardinalityForm::at_least,
                                                  CardinalityForm::at_most};
            if (n >= 2) {
                feasible.push_back(CardinalityForm::exactly);
            }
            if (n >= 3) {
                feasible.push_back(CardinalityForm::range);
            }
            switch (feasible[uniform(0, static_cast<std::uint32_t>(feasible.size()) - 1)]) {
            case CardinalityForm::ones:
                node.low = 0;
                node.high = n;
                break;
            case CardinalityForm::at_least:
                node.low = uniform(1, n);
                node.high = n;
                break;
            case CardinalityForm::at_most:
                node.low = 0;
                node.high = uniform(0, n - 1);
                break;
            case CardinalityForm::exactly:
                node.low = node.high = uniform(1, n - 1);
                break;
            case CardinalityForm::range:
                node.low = uniform(1, n - 2);
                node.high = uniform(node.low + 1, n - 1);
                break;
            }
            ++coverage.cardinality_forms[static_cast<std::size_t>(cardinalityForm(node.low, node.high, n))];
        }
        nodes.push_back(std::move(node));
    }

    const auto nodeCount = static_cast<std::uint32_t>(nodes.size());
    std::vector<std::uint32_t> users(nodeCount, 0);
    for (auto const &node : nodes) {
        for (auto const &[source, complement] : node.operands) {
            ++users[source];
            coverage.complemented_edges += complement ? 1 : 0;
        }
    }
    coverage.shared_nodes +=
        static_cast<std::size_t>(std::ranges::count_if(users, [](const auto count) { return count >= 2; }));

    std::vector<std::uint32_t> storage(nodeCount);
    std::iota(storage.begin(), storage.end(), 0U);
    if (chance(0.75)) {
        pcfb_test_random::shuffle(std::span{storage}, rng);
    }
    std::vector<std::uint32_t> logicalAt(nodeCount);
    for (std::uint32_t logical = 0; logical < nodeCount; ++logical) {
        logicalAt[storage[logical]] = logical;
    }
    bool topological = true;
    for (std::uint32_t logical = 0; logical < nodeCount; ++logical) {
        for (auto const &[source, complement] : nodes[logical].operands) {
            topological = topological && storage[source] < storage[logical];
        }
    }
    coverage.non_topological_circuits += topological ? 0 : 1;

    pc::builder builder;
    builder.reserve_nodes(nodeCount);
    std::vector<pc::probability_source_id> sources;
    for (const auto p : probabilities) {
        sources.push_back(builder.add_literal_probability(p));
    }
    std::vector<pc::edge_ref> operands;
    for (std::uint32_t id = 0; id < nodeCount; ++id) {
        auto const &node = nodes[logicalAt[id]];
        ++coverage.op_codes[static_cast<std::size_t>(node.code)];
        operands.clear();
        for (auto const &[source, complement] : node.operands) {
            operands.push_back(pc::edge_ref{.source = pc::bool_node_id(storage[source]), .complement = complement});
        }
        switch (node.code) {
        case pc::op_code::basic_event:
            (void)builder.add_basic_event(sources[node.source]);
            break;
        case pc::op_code::constant:
            (void)builder.add_constant(node.value);
            break;
        case pc::op_code::house_event:
            (void)builder.add_house_event(node.value);
            break;
        case pc::op_code::cardinality:
            (void)builder.add_cardinality(node.low, node.high, std::span<pc::edge_ref const>{operands});
            break;
        default:
            (void)builder.add_gate(node.code, std::span<pc::edge_ref const>{operands});
            break;
        }
    }

    constexpr std::array roles{pc::observable_role::primary, pc::observable_role::primary, pc::observable_role::watch,
                               pc::observable_role::projection_anchor};
    const auto targetCount = uniform(1, 4);
    coverage.multi_target_circuits += targetCount > 1 ? 1 : 0;
    for (std::uint32_t target = 0; target < targetCount; ++target) {
        const auto logical = chance(0.7) ? uniform(leafCount, nodeCount - 1) : uniform(0, nodeCount - 1);
        (void)builder.add_target(pc::bool_node_id(storage[logical]),
                                 roles[uniform(0, static_cast<std::uint32_t>(roles.size()) - 1)]);
    }
    return std::move(builder).seal();
}

namespace pcfb_io = fuzzybits::io::backend::fuzzybits::pcfb;

[[nodiscard]] inline auto sealed(pc::builder &&builder) -> pc::circuit {
    auto result = std::move(builder).seal();
    INFO(diagnosticsText(result.diagnostics));
    REQUIRE(result.ok());
    return std::move(result.output);
}

// Three targets over events a, b, c: a AND b, b OR c, and the event a.
[[nodiscard]] inline auto threeTargetModel() -> pc::circuit {
    pc::builder builder;
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
    const auto b = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto c = builder.add_basic_event(builder.add_literal_probability(0.3));
    const auto both = builder.add_and({pc::edge_ref{.source = a}, pc::edge_ref{.source = b}});
    const auto either = builder.add_or({pc::edge_ref{.source = b}, pc::edge_ref{.source = c}});
    (void)builder.add_target(both);
    (void)builder.add_target(either);
    (void)builder.add_target(a);
    return sealed(std::move(builder));
}

[[nodiscard]] inline auto threeTargetNames() -> pcfb_io::write_options {
    pcfb_io::write_options options;
    options.target_names = {{pc::target_id(0), "AND"}, {pc::target_id(1), "OR"}, {pc::target_id(2), "A"}};
    return options;
}

// A random circuit with the names, query kinds, quantification and metadata a writer can record.
struct RandomModel final {
    pc::circuit circuit;
    pcfb_io::write_options options;
};

[[nodiscard]] inline auto randomModel(std::mt19937_64 &rng, Coverage &coverage, const RandomSize size = {})
    -> RandomModel {
    auto result = randomCircuit(rng, coverage, size);
    INFO(diagnosticsText(result.diagnostics));
    REQUIRE(result.ok());
    const auto chance = [&](const double p) { return pcfb_test_random::chance(rng, p); };
    const auto uniform = [&](const std::uint32_t low, const std::uint32_t high) {
        return static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, low, high));
    };
    RandomModel model{.circuit = std::move(result.output)};
    auto &options = model.options;
    constexpr std::array nameStyles{"n", "node ", "Ünï/", "A-B.", "x_"};
    for (std::uint32_t node = 0; node < model.circuit.node_count(); ++node) {
        if (chance(0.5)) {
            options.node_names[pc::bool_node_id(node)] =
                std::string(nameStyles[uniform(0, nameStyles.size() - 1)]) + std::to_string(node);
        }
    }
    options.query = chance(0.2) ? pcfb_io::query_kind::monotone_hull_probability : pcfb_io::query_kind::probability;
    constexpr std::array literalModes{"", "split", "fixed", "plain"};
    for (auto const &target : model.circuit.targets()) {
        // Unique per target: targets that share a node and a name would read back as one target with two kinds.
        if (chance(0.7)) {
            options.target_names[target.id] = "T" + std::to_string(target.id.raw()) + (chance(0.3) ? " top/ü" : "");
        }
        using kind = pcfb_io::query_kind;
        switch (uniform(0, 5)) {
        case 0:
            options.target_kinds[target.id] = {kind::monotone_hull_probability};
            break;
        case 1:
            options.target_kinds[target.id] = {kind::probability, kind::monotone_hull_probability};
            break;
        case 2:
            options.target_kinds[target.id] = {kind::monotone_hull_probability, kind::probability};
            break;
        case 3:
            options.target_kinds[target.id] = {kind::probability};
            break;
        default:
            break;
        }
        if (chance(0.4)) {
            // The target name is always recorded, as the lowering does, so every entry reads back.
            pcfb_io::target_quantification entry{.target = "D" + std::to_string(target.id.raw())};
            entry.literal_mode = literalModes[uniform(0, literalModes.size() - 1)];
            if (chance(0.5)) {
                entry.max_initiators_per_cutset = uniform(0, 7);
            }
            options.quantification[target.id] = entry;
        }
    }
    if (chance(0.7)) {
        options.model_name = "model " + std::to_string(uniform(0, 999));
    }
    if (chance(0.5)) {
        options.source_format = chance(0.5) ? "cafta-ftp" : "openpsa-mef-2.0";
    }
    if (chance(0.5)) {
        std::array<std::byte, 32> digest{};
        for (auto &byte : digest) {
            byte = static_cast<std::byte>(uniform(0, 255));
        }
        options.source_sha256 = digest;
    }
    if (chance(0.5)) {
        options.compile_options = R"({"kinds": ["hull"], "level": )" + std::to_string(uniform(0, 1)) + "}";
    }
    return model;
}

[[nodiscard]] inline auto projectRoot() -> std::filesystem::path { return std::filesystem::path(PROJECT_SOURCE_DIR); }

// Public FTP fixtures plus FUZZYBITS_PRIVATE_FIXTURES_DIR, as in the FTP IO tests.
[[nodiscard]] inline auto ftpCorpusFiles() -> std::vector<std::filesystem::path> {
    std::vector<std::filesystem::path> roots{
        projectRoot() / "input" / "ftp" / "synthetic",
        projectRoot() / "src" / "io" / "ftp" / "tests" / "testdata",
    };
    if (const char *privateDir = std::getenv("FUZZYBITS_PRIVATE_FIXTURES_DIR");
        privateDir != nullptr && privateDir[0] != '\0') {
        roots.emplace_back(privateDir);
    }
    std::vector<std::filesystem::path> files;
    for (auto const &root : roots) {
        std::error_code error;
        constexpr auto skip = std::filesystem::directory_options::skip_permission_denied;
        for (auto iterator = std::filesystem::recursive_directory_iterator(root, skip, error);
             !error && iterator != std::filesystem::recursive_directory_iterator(); iterator.increment(error)) {
            auto extension = iterator->path().extension().string();
            std::ranges::transform(extension, extension.begin(),
                                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
            std::error_code statusError;
            if (iterator->is_regular_file(statusError) && extension == ".ftp") {
                files.push_back(iterator->path());
            }
        }
    }
    std::ranges::sort(files);
    return files;
}

} // namespace pcfb_writer_test
