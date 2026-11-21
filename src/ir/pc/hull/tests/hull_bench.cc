// Throughput of the hull check, single-threaded and on all threads.
//
//   fuzzybits_ir_pc_hull_bench [--nodes N] [--lanes N] [--candidates N] [--seed N]
//                              [--exclusive P] [--ftp FILE ... --target NAME] [--split-complements]
//                              [--verify]
//
// Lanes are sampled from the event probabilities. "Candidates" are lanes where
// the RFC's bounds disagree (L = 0, U = 1, docs/rfc/monotone-hull-queries.md
// §4.4, §5.1): the lanes a runtime sends to the check. Only aggregates are
// printed, so private models can be measured without disclosing them.
// --exclusive P gives the synthetic model's delete-term events probabilities
// around P (larger residuals). --verify re-decides the candidates by complete
// search alone and by BDDs alone and fails on any difference.

#include "api/options.h"
#include "io/ftp/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/hull/detail/bits.h"
#include "ir/pc/hull/hull.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

    namespace pc = fuzzybits::ir::pc;
    namespace hull = fuzzybits::ir::pc::hull;
    using hull::detail::all_ones;

    struct model final {
        pc::circuit circuit;
        pc::bool_node_id target;
    };

    // A PRA-like model. Components are ORs of rare failures, some with a
    // frequent unavailability; above them, levels of OR, AND and 2-of-3 gates
    // over neighbouring gates (with some shared inputs) end in about 64
    // systems. Sequences are SEQ = IE and F and not(S or DEL): F is one or two
    // gates whose probability (estimated as if inputs were independent) is a
    // few percent, S is one of the top systems, and DEL holds pairwise
    // exclusions among frequent events that also occur in components, plus a
    // two-flag term. The target is the OR of all sequences; nearly the whole
    // circuit is in its cone.
    auto synthetic(const std::uint32_t nodes, const double exclusive_probability, std::mt19937_64 &random) -> model {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const auto events = std::max<std::uint32_t>(256, nodes / 4);
        const auto sequences = std::max<std::uint32_t>(8, nodes / 500);
        pc::builder builder;
        builder.reserve_nodes(nodes + 64);
        std::vector<pc::bool_node_id> rare;
        std::vector<pc::bool_node_id> frequent;
        std::vector<pc::bool_node_id> flags;
        std::vector<double> estimate;
        const auto add_event = [&](double probability) {
            const auto node = builder.add_basic_event(builder.add_literal_probability(probability));
            estimate.resize(node.raw() + 1, 0.0);
            estimate[node.raw()] = probability;
            return node;
        };
        for (std::uint32_t event = 0; event < events; ++event) {
            const auto roll = unit(random);
            double probability = 1.0;
            if (roll < 0.9) {
                probability = std::pow(10.0, -4.0 + 2.0 * unit(random));
            } else if (roll < 0.98) {
                probability = 0.02 + 0.2 * unit(random);
            } else if (roll < 0.995) {
                probability = 0.6 + 0.35 * unit(random);
            }
            const auto node = add_event(probability);
            (probability == 1.0 ? flags : probability >= 0.02 ? frequent : rare).push_back(node);
        }
        std::vector<pc::bool_node_id> exclusive(frequent.begin(),
                                                frequent.begin() + std::min<std::size_t>(60, frequent.size()));
        if (exclusive_probability > 0.0) {
            for (auto &node : exclusive) {
                node = add_event(std::clamp(exclusive_probability * (0.75 + 0.5 * unit(random)), 0.0, 1.0));
            }
        }
        const auto pick = [&](std::vector<pc::bool_node_id> const &from) { return from[random() % from.size()]; };
        const auto edge = [](pc::bool_node_id node, bool complement = false) {
            return pc::edge_ref{.source = node, .complement = complement};
        };
        const auto add_gate = [&](pc::op_code code, std::vector<pc::edge_ref> const &operands) {
            double p = code == pc::op_code::and_gate ? 1.0 : 0.0;
            if (code == pc::op_code::or_gate) {
                double miss = 1.0;
                for (auto const &operand : operands) {
                    miss *= 1.0 - estimate[operand.source.raw()];
                }
                p = 1.0 - miss;
            } else if (code == pc::op_code::and_gate) {
                for (auto const &operand : operands) {
                    p *= estimate[operand.source.raw()];
                }
            } else {
                // At least 2: distribution of the count by dynamic programming.
                std::vector<double> count{1.0};
                for (auto const &operand : operands) {
                    const auto q = estimate[operand.source.raw()];
                    count.push_back(0.0);
                    for (auto k = count.size() - 1; k > 0; --k) {
                        count[k] = count[k] * (1.0 - q) + count[k - 1] * q;
                    }
                    count[0] *= 1.0 - q;
                }
                p = 1.0 - count[0] - count[1];
            }
            const auto node = code == pc::op_code::cardinality
                                  ? builder.add_cardinality(2, static_cast<std::uint32_t>(operands.size()), operands)
                                  : builder.add_gate(code, operands);
            estimate.resize(node.raw() + 1, 0.0);
            estimate[node.raw()] = p;
            return node;
        };

        // Gates above the components: n/3 + n/6 + ... ~ n/2, so components get
        // about two thirds of the gate budget.
        const auto budget = nodes - events - 6 * sequences;
        std::vector<pc::bool_node_id> level;
        std::vector<pc::bool_node_id> failure_candidates;
        std::vector<pc::edge_ref> operands;
        for (std::uint32_t index = 0; index < budget * 2 / 3; ++index) {
            operands.clear();
            const auto arity = 2 + random() % 3;
            for (std::uint32_t operand = 0; operand < arity; ++operand) {
                operands.push_back(edge(pick(rare)));
            }
            if (unit(random) < 0.25) {
                operands.push_back(edge(unit(random) < 0.5 ? pick(exclusive) : pick(frequent)));
            }
            level.push_back(add_gate(pc::op_code::or_gate, operands));
        }
        constexpr std::array<pc::op_code, 4> cycle{pc::op_code::or_gate, pc::op_code::and_gate, pc::op_code::or_gate,
                                                   pc::op_code::cardinality};
        for (std::size_t depth = 0; level.size() > 64; ++depth) {
            const auto code = cycle[depth % cycle.size()];
            const std::size_t width = code == pc::op_code::and_gate ? 2 : 3;
            std::vector<pc::bool_node_id> next;
            for (std::size_t first = 0; first < level.size(); first += width) {
                operands.clear();
                for (auto index = first; index < std::min(level.size(), first + width); ++index) {
                    operands.push_back(edge(level[index]));
                }
                if (unit(random) < 0.25) {
                    // A shared input from nearby, as support systems are shared.
                    const auto low = first > 200 ? first - 200 : 0;
                    const auto high = std::min(level.size(), first + 200);
                    operands.push_back(edge(level[low + random() % (high - low)]));
                }
                if (operands.size() == 1) {
                    next.push_back(operands[0].source);
                    continue;
                }
                const auto node = add_gate(code, operands);
                next.push_back(node);
                if (depth >= 3 && estimate[node.raw()] >= 0.01 && estimate[node.raw()] <= 0.05) {
                    failure_candidates.push_back(node);
                }
            }
            level = std::move(next);
        }
        const auto systems = level;
        if (failure_candidates.empty()) {
            failure_candidates = systems;
        }

        operands.clear();
        for (std::uint32_t pair = 0; pair < 2 * sequences; ++pair) {
            operands.push_back(edge(builder.add_and({edge(pick(exclusive)), edge(pick(exclusive))})));
        }
        if (flags.size() >= 2) {
            operands.push_back(edge(builder.add_and({edge(flags[0]), edge(flags[1])})));
        }
        const auto del = builder.add_or(operands);

        std::vector<pc::edge_ref> sequence_edges;
        for (std::uint32_t sequence = 0; sequence < sequences; ++sequence) {
            operands.clear();
            operands.push_back(edge(pick(frequent)));
            const auto failed = 1 + random() % 2;
            for (std::uint32_t index = 0; index < failed; ++index) {
                operands.push_back(edge(pick(failure_candidates)));
            }
            const auto success = builder.add_or({edge(systems[sequence % systems.size()]), edge(del)});
            operands.push_back(edge(success, true));
            sequence_edges.push_back(edge(builder.add_and(operands)));
        }
        const auto target = builder.add_or(sequence_edges);
        (void)builder.add_target(target);
        auto sealed = std::move(builder).seal();
        if (!sealed.ok()) {
            std::fprintf(stderr, "synthetic circuit failed to seal\n");
            std::exit(1);
        }
        return {sealed.output, target};
    }

    auto read_ftp(std::vector<std::string> const &files, std::string const &target) -> model {
        std::string text;
        for (auto const &file : files) {
            std::ifstream input(file, std::ios::binary);
            if (!input) {
                std::fprintf(stderr, "cannot open an input file\n");
                std::exit(1);
            }
            text.append(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
            text.push_back('\n');
        }
        fuzzybits::api::option_set options;
        options.set(fuzzybits::io::backend::fuzzybits::ftp::target_option(), target);
        std::istringstream stream(text);
        auto read = fuzzybits::io::backend::fuzzybits::ftp::read(stream, options);
        if (!read) {
            std::fprintf(stderr, "FTP read failed (%zu diagnostics)\n", read.diagnostics().size());
            std::exit(1);
        }
        auto const *circuit = pc::circuit_from(read.value().model);
        return {*circuit, circuit->targets()[0].node};
    }

    auto probabilities(pc::circuit const &circuit) -> std::vector<double> {
        std::vector<double> out;
        for (std::uint32_t node = 0; node < circuit.node_count(); ++node) {
            if (circuit.op_codes()[node] != pc::op_code::basic_event) {
                continue;
            }
            auto const &source = circuit.probability_sources()[circuit.probability_source_refs()[node].raw()];
            if (source.kind != pc::probability_source_kind::literal) {
                std::fprintf(stderr, "an event has no literal probability\n");
                std::exit(1);
            }
            out.push_back(source.literal);
        }
        return out;
    }

    // Per-node syntactic polarity w.r.t. the target (1 positive, 2 negative).
    auto polarities(pc::circuit const &circuit, pc::bool_node_id target) -> std::vector<std::uint8_t> {
        std::vector<std::uint8_t> polarity(circuit.node_count(), 0);
        polarity[target.raw()] = 1;
        const auto swap = [](std::uint8_t p) { return static_cast<std::uint8_t>(((p & 1) << 1) | ((p & 2) >> 1)); };
        for (auto node = circuit.node_count(); node-- > 0;) {
            auto p = polarity[node];
            if (p == 0) {
                continue;
            }
            const auto code = circuit.op_codes()[node];
            const auto operands = circuit.operands(pc::bool_node_id(node));
            if (code == pc::op_code::not_gate || code == pc::op_code::nand_gate || code == pc::op_code::nor_gate) {
                p = swap(p);
            } else if (code == pc::op_code::xor_gate || code == pc::op_code::xnor_gate) {
                p = 3;
            } else if (code == pc::op_code::cardinality) {
                const auto low = circuit.cardinality_low()[node];
                const auto high = circuit.cardinality_high()[node];
                if (low == 0 && high < operands.size()) {
                    p = swap(p);
                } else if (low > 0 && high < operands.size()) {
                    p = 3;
                }
            }
            for (auto const &operand : operands) {
                polarity[operand.source.raw()] |= operand.complement ? swap(p) : p;
            }
        }
        return polarity;
    }

    // Lower bound L (negative-unate events zeroed) and upper bound U
    // (negation-normal relaxation, negative literals TRUE) for 64 lanes.
    class bounds final {
    public:
        bounds(pc::circuit const &circuit, pc::bool_node_id target)
            : circuit_(circuit), target_(target), polarity_(polarities(circuit, target)),
              plain_(circuit.node_count()), positive_(circuit.node_count()), negative_(circuit.node_count()) {}

        // `events[e]`: lane word of event e.
        auto evaluate(std::vector<std::uint64_t> const &events) -> std::pair<std::uint64_t, std::uint64_t> {
            std::uint32_t event = 0;
            for (std::uint32_t node = 0; node < circuit_.node_count(); ++node) {
                const auto code = circuit_.op_codes()[node];
                const auto operands = circuit_.operands(pc::bool_node_id(node));
                if (code == pc::op_code::basic_event) {
                    const auto word = events[event++];
                    plain_[node] = polarity_[node] == 2 ? 0 : word;
                    positive_[node] = word;
                    negative_[node] = all_ones;
                    continue;
                }
                if (code == pc::op_code::constant || code == pc::op_code::house_event) {
                    plain_[node] = positive_[node] = circuit_.constant_values()[node] != 0 ? all_ones : 0;
                    negative_[node] = ~positive_[node];
                    continue;
                }
                std::uint64_t value = 0;
                std::uint64_t positive = 0;
                std::uint64_t negative = 0;
                const auto input = [&](pc::edge_ref const &edge) {
                    const auto source = edge.source.raw();
                    return edge.complement ? std::tuple{~plain_[source], negative_[source], positive_[source]}
                                           : std::tuple{plain_[source], positive_[source], negative_[source]};
                };
                bool invert = false;
                switch (code) {
                case pc::op_code::nand_gate:
                case pc::op_code::not_gate:
                    invert = true;
                    [[fallthrough]];
                case pc::op_code::and_gate:
                    value = positive = all_ones;
                    for (auto const &operand : operands) {
                        const auto [v, p, n] = input(operand);
                        value &= v;
                        positive &= p;
                        negative |= n;
                    }
                    break;
                case pc::op_code::nor_gate:
                    invert = true;
                    [[fallthrough]];
                case pc::op_code::or_gate:
                    negative = all_ones;
                    for (auto const &operand : operands) {
                        const auto [v, p, n] = input(operand);
                        value |= v;
                        positive |= p;
                        negative &= n;
                    }
                    break;
                case pc::op_code::xnor_gate:
                    invert = true;
                    [[fallthrough]];
                case pc::op_code::xor_gate:
                    negative = all_ones;
                    for (auto const &operand : operands) {
                        const auto [v, p, n] = input(operand);
                        value ^= v;
                        const auto next = (positive & n) | (negative & p);
                        negative = (positive & p) | (negative & n);
                        positive = next;
                    }
                    break;
                case pc::op_code::cardinality: {
                    hull::detail::sliced_counter values;
                    hull::detail::sliced_counter ones;
                    hull::detail::sliced_counter zeros;
                    for (auto const &operand : operands) {
                        const auto [v, p, n] = input(operand);
                        values.add(v);
                        ones.add(p);
                        zeros.add(n);
                    }
                    const std::uint64_t count = operands.size();
                    const std::uint64_t low = circuit_.cardinality_low()[node];
                    const std::uint64_t high = circuit_.cardinality_high()[node];
                    value = values.at_least(low) & ~values.at_least(high + 1);
                    positive = ones.at_least(low) & zeros.at_least(count - high);
                    negative = ones.at_least(high + 1) | zeros.at_least(count - low + 1);
                    break;
                }
                default:
                    break;
                }
                if (invert) {
                    value = ~value;
                    std::swap(positive, negative);
                }
                plain_[node] = value;
                positive_[node] = positive;
                negative_[node] = negative;
            }
            return {plain_[target_.raw()], positive_[target_.raw()]};
        }

    private:
        pc::circuit const &circuit_;
        pc::bool_node_id target_;
        std::vector<std::uint8_t> polarity_;
        std::vector<std::uint64_t> plain_;
        std::vector<std::uint64_t> positive_;
        std::vector<std::uint64_t> negative_;
    };

    struct run final {
        double seconds = 0.0;
        hull::check_stats stats;
    };

    auto time_check(hull::checker const &check, std::vector<std::uint64_t> const &words, std::size_t lanes,
                    std::size_t stride, std::vector<std::uint64_t> *answers = nullptr) -> run {
        std::vector<std::uint64_t> out((lanes + 63) / 64, 0);
        run result;
        const auto start = std::chrono::steady_clock::now();
        check.check(hull::lane_matrix{.words = words, .lane_count = lanes, .stride = stride}, out, &result.stats);
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (answers != nullptr) {
            *answers = std::move(out);
        }
        return result;
    }

    void report(char const *label, run const &result, int threads) {
        auto const &s = result.stats;
        const auto exact = s.truth_table + s.bdd + s.search;
        std::printf("%-28s lanes %9llu  %8.3f s  %12.0f lanes/s  %12.0f lanes/s/core  hit %.4f\n", label,
                    static_cast<unsigned long long>(s.lanes), result.seconds, s.lanes / result.seconds,
                    s.lanes / result.seconds / threads, s.lanes != 0 ? double(s.hits) / double(s.lanes) : 0.0);
        std::printf("%-28s decided by: bounds %llu, reduction %llu, witness %llu, truth table %llu, bdd %llu "
                    "(over budget %llu), search %llu; free events mean %.2f max %u\n",
                    "", static_cast<unsigned long long>(s.bounds), static_cast<unsigned long long>(s.reduction),
                    static_cast<unsigned long long>(s.witness), static_cast<unsigned long long>(s.truth_table),
                    static_cast<unsigned long long>(s.bdd), static_cast<unsigned long long>(s.bdd_budget_exceeded),
                    static_cast<unsigned long long>(s.search),
                    exact != 0 ? double(s.residual_variables_total) / double(exact) : 0.0, s.residual_variables_max);
    }

} // namespace

auto main(int argc, char **argv) -> int {
    std::uint32_t nodes = 100000;
    std::size_t lanes = std::size_t{1} << 16;
    std::size_t candidates_wanted = std::size_t{1} << 14;
    std::uint64_t seed = 178;
    double exclusive_probability = 0.0;
    bool split_complements = false;
    bool verify = false;
    std::vector<std::string> files;
    std::string target_name;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&]() -> std::string {
            if (index + 1 >= argc) {
                std::fprintf(stderr, "missing value\n");
                std::exit(2);
            }
            return argv[++index];
        };
        if (argument == "--nodes") {
            nodes = static_cast<std::uint32_t>(std::stoul(next()));
        } else if (argument == "--lanes") {
            lanes = std::stoull(next());
        } else if (argument == "--candidates") {
            candidates_wanted = std::stoull(next());
        } else if (argument == "--seed") {
            seed = std::stoull(next());
        } else if (argument == "--exclusive") {
            exclusive_probability = std::stod(next());
        } else if (argument == "--ftp") {
            files.push_back(next());
        } else if (argument == "--target") {
            target_name = next();
        } else if (argument == "--split-complements") {
            split_complements = true;
        } else if (argument == "--verify") {
            verify = true;
        } else {
            std::fprintf(stderr, "unknown argument\n");
            return 2;
        }
    }
    lanes = std::max<std::size_t>(64, (lanes + 63) / 64 * 64);

    std::mt19937_64 random(seed);
    const auto subject = files.empty() ? synthetic(nodes, exclusive_probability, random) : read_ftp(files, target_name);
    const auto probability = probabilities(subject.circuit);
    const auto events = static_cast<std::uint32_t>(probability.size());

    std::vector<hull::event_mode> modes;
    if (split_complements) {
        modes.assign(events, hull::event_mode::down);
        std::vector<std::uint32_t> event_of(subject.circuit.node_count(), 0);
        for (std::uint32_t node = 0, event = 0; node < subject.circuit.node_count(); ++node) {
            if (subject.circuit.op_codes()[node] == pc::op_code::basic_event) {
                event_of[node] = event++;
            }
        }
        for (auto const &edge : subject.circuit.edges()) {
            if (edge.complement && subject.circuit.op_codes()[edge.source.raw()] == pc::op_code::basic_event) {
                modes[event_of[edge.source.raw()]] = hull::event_mode::split;
            }
        }
    }

    hull::check_options serial_options{};
    serial_options.parallel = false;
    const auto prepare_start = std::chrono::steady_clock::now();
    auto serial = hull::checker::make(subject.circuit, subject.target, modes, serial_options);
    const auto prepare_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - prepare_start).count();
    auto parallel = hull::checker::make(subject.circuit, subject.target, modes);
    if (!serial || !parallel) {
        std::fprintf(stderr, "checker preparation failed\n");
        return 1;
    }
    std::printf("circuit: %u nodes, %u edges, %u events; cone %u positions, %zu events; prepare %.3f s\n",
                subject.circuit.node_count(), subject.circuit.edge_count(), events, serial.value().cone_size(),
                serial.value().cone_events().size(), prepare_seconds);

    // Sample lanes; keep the first `lanes` and the candidates (L = 0, U = 1).
    const std::size_t stride = (std::size_t{events} + 63) / 64;
    std::vector<std::uint64_t> all_words(lanes * stride, 0);
    std::vector<std::uint64_t> candidate_words;
    std::size_t candidates = 0;
    std::size_t sampled = 0;
    std::size_t drawn = 0;
    std::size_t lower_true = 0;
    std::size_t upper_true = 0;
    bounds bound(subject.circuit, subject.target);
    std::vector<std::uint64_t> event_words(stride * 64, 0);
    std::vector<double> log_miss(events);
    for (std::uint32_t event = 0; event < events; ++event) {
        log_miss[event] = probability[event] > 0.0 && probability[event] < 1.0 ? std::log1p(-probability[event]) : 0.0;
    }
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    // Bernoulli lanes by geometric skipping: about 1 + 64 p draws per event and block.
    const auto skip = [&](std::uint32_t event) {
        const auto gap = std::floor(std::log(1.0 - unit(random)) / log_miss[event]);
        return gap >= 64.0 ? 64U : static_cast<std::uint32_t>(gap);
    };
    std::array<std::uint64_t, 64> rows{};
    const auto bounds_start = std::chrono::steady_clock::now();
    while (sampled < lanes || (candidates < candidates_wanted && drawn < 16 * lanes)) {
        for (std::uint32_t event = 0; event < events; ++event) {
            std::uint64_t word = 0;
            if (probability[event] >= 1.0) {
                word = all_ones;
            } else if (probability[event] > 0.0) {
                for (auto lane = skip(event); lane < 64; lane += 1 + skip(event)) {
                    word |= std::uint64_t{1} << lane;
                }
            }
            event_words[event] = word;
        }
        drawn += 64;
        const auto [lower, upper] = bound.evaluate(event_words);
        if (sampled < lanes) {
            lower_true += static_cast<std::size_t>(std::popcount(lower));
            upper_true += static_cast<std::size_t>(std::popcount(upper));
        }
        const auto keep_all = sampled < lanes ? all_ones : 0;
        auto keep_candidates = ~lower & upper;
        const auto room = candidates_wanted - std::min(candidates, candidates_wanted);
        while (static_cast<std::size_t>(std::popcount(keep_candidates)) > room) {
            keep_candidates &= keep_candidates - 1;
        }
        const auto first_candidate = candidate_words.size();
        candidate_words.resize(first_candidate + static_cast<std::size_t>(std::popcount(keep_candidates)) * stride);
        for (std::size_t group = 0; group < stride; ++group) {
            std::copy_n(event_words.begin() + static_cast<std::ptrdiff_t>(group * 64), 64, rows.begin());
            hull::detail::transpose64(rows);
            std::size_t candidate = 0;
            for (std::uint32_t lane = 0; lane < 64; ++lane) {
                if (((keep_all >> lane) & 1U) != 0) {
                    all_words[(sampled + lane) * stride + group] = rows[lane];
                }
                if (((keep_candidates >> lane) & 1U) != 0) {
                    candidate_words[first_candidate + (candidate++) * stride + group] = rows[lane];
                }
            }
        }
        sampled += keep_all != 0 ? 64 : 0;
        candidates += static_cast<std::size_t>(std::popcount(keep_candidates));
    }
    const auto bounds_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - bounds_start).count();
    std::printf("sampling + bounds: %.3f s for %zu lanes; P(L) ~ %.5f, P(U) ~ %.5f; %zu candidate lanes kept\n",
                bounds_seconds, drawn, double(lower_true) / double(lanes), double(upper_true) / double(lanes),
                candidates);

    int threads = 1;
#if defined(_OPENMP)
    threads = omp_get_max_threads();
#endif
    report("all lanes, 1 thread", time_check(serial.value(), all_words, lanes, stride), 1);
    std::vector<std::uint64_t> answers;
    if (candidates > 0) {
        report("candidates, 1 thread", time_check(serial.value(), candidate_words, candidates, stride, &answers), 1);
    }
    if (verify && candidates > 0) {
        // The same lanes through other tiers: no unate fixing or witnesses,
        // then complete search only, and BDDs only (unbounded).
        hull::check_options plain{};
        plain.pure_literals = false;
        plain.witnesses = false;
        plain.truth_table_max_variables = 0;
        plain.bdd_node_budget = 0;
        hull::check_options bdd_only = plain;
        bdd_only.bdd_max_variables = 1U << 20;
        bdd_only.bdd_node_budget = 1U << 24;
        for (auto const &[label, options] : {std::pair{"verify: search only", plain}, std::pair{"verify: bdd only", bdd_only}}) {
            auto other = hull::checker::make(subject.circuit, subject.target, modes, options);
            std::vector<std::uint64_t> other_answers;
            report(label, time_check(other.value(), candidate_words, candidates, stride, &other_answers), threads);
            std::size_t mismatches = 0;
            for (std::size_t word = 0; word < answers.size(); ++word) {
                mismatches += static_cast<std::size_t>(std::popcount(answers[word] ^ other_answers[word]));
            }
            std::printf("%-28s mismatches %zu\n", label, mismatches);
            if (mismatches != 0) {
                return 1;
            }
        }
    }
    report("all lanes, all threads", time_check(parallel.value(), all_words, lanes, stride), threads);
    if (candidates > 0) {
        report("candidates, all threads", time_check(parallel.value(), candidate_words, candidates, stride), threads);
    }
    std::printf("threads: %d\n", threads);
    return 0;
}
