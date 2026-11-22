// Throughput of the hull check, single-threaded and on all threads.
//
//   fuzzybits_ir_pc_hull_bench [--nodes N] [--lanes N] [--candidates N] [--seed N]
//                              [--exclusive P] [--ftp FILE ... --target NAME] [--split-complements]
//                              [--verify]
//   fuzzybits_ir_pc_hull_bench --ftp FILE ... --hull [--target NAME ...] [--max-initiators K|none]
//                              [--literals atom_split|omega_l|negation] [--level 0|1]
//                              [--decompose-initiators [--verify-decomposition]]
//                              [--check-threads 1|all|both] [--lanes N] [--seed N]
//
// Lanes are sampled from the event probabilities. "Candidates" are lanes where
// the RFC's bounds disagree (L = 0, U = 1, docs/rfc/monotone-hull-queries.md
// §4.4, §5.1): the lanes a runtime sends to the check. Only aggregates are
// printed, so private models can be measured without disclosing them.
// --exclusive P gives the synthetic model's delete-term events probabilities
// around P (larger residuals). --verify re-decides the candidates by complete
// search alone and by BDDs alone and fails on any difference.
//
// --hull lowers the targets as `fzb compile` does for hull queries (negation
// hull, the initiator rule from the model attribute unless --max-initiators
// says otherwise), applies the ir::pc passes of --level, and measures the rule
// as lowered: bounds L and U per lane, the check on the candidates. Results are
// labelled by target index. With k = 1, --decompose-initiators decides the
// same lanes by h = ↑g∅ ∨ ⋁ᵢ (xᵢ ∧ ↑gᵢ), where g∅ and gᵢ are g with the
// initiating events replaced by constants and no rule; --verify-decomposition
// compares both decisions on every lane and fails on any difference.

#include "api/options.h"
#include "compile/ftp/compile.h"
#include "compile/ftp/tests/support.h"
#include "io/ftp/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/hull/detail/bits.h"
#include "ir/pc/hull/hull.h"
#include "ir/pc/op_code.h"
#include "ir/pc/passes/passes.h"
#include "ir/pc/probability.h"
#include "ir/pra/projection.h"

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
#include <memory>
#include <optional>
#include <random>
#include <span>
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
    namespace passes = fuzzybits::ir::pc::passes;
    namespace pra = fuzzybits::ir::pra;
    namespace compile_ftp = fuzzybits::compile::ftp;
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

    auto read_sources(std::vector<std::string> const &files) -> std::vector<ftp_lowering_test::reader::source_text> {
        std::vector<ftp_lowering_test::reader::source_text> sources;
        for (auto const &file : files) {
            std::ifstream input(file, std::ios::binary);
            if (!input) {
                std::fprintf(stderr, "cannot open an input file\n");
                std::exit(1);
            }
            sources.push_back(ftp_lowering_test::reader::source_text{
                .path = file, .text = {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()}});
        }
        return sources;
    }

    auto read_ftp(std::vector<std::string> const &files, std::string const &target) -> model {
        const auto sources = read_sources(files);
        auto read = ftp_lowering_test::lower(sources, ftp_lowering_test::boolean_options({target}));
        if (!read) {
            std::fprintf(stderr, "FTP read failed (%zu diagnostics)\n", read.diagnostics().size());
            std::exit(1);
        }
        auto const &circuit = read.value().circuit;
        return {circuit, circuit.targets()[0].node};
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
    // `split` (by node, may be empty) marks atom-split events: a complemented
    // edge into one reads the atom, whose polarity is its parent's and goes to
    // `atoms` instead.
    auto polarities(pc::circuit const &circuit, pc::bool_node_id target, std::vector<std::uint8_t> const &split = {},
                    std::vector<std::uint8_t> *atoms = nullptr) -> std::vector<std::uint8_t> {
        std::vector<std::uint8_t> polarity(circuit.node_count(), 0);
        if (atoms != nullptr) {
            atoms->assign(circuit.node_count(), 0);
        }
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
                if (operand.complement && !split.empty() && split[operand.source.raw()] != 0) {
                    (*atoms)[operand.source.raw()] |= p;
                } else {
                    polarity[operand.source.raw()] |= operand.complement ? swap(p) : p;
                }
            }
        }
        return polarity;
    }

    // Lower bound L (negative-unate events zeroed) and upper bound U
    // (negation-normal relaxation, negative literals TRUE) for 64 lanes.
    // `modes` (by event, may be empty: all down) are the checker's; a split
    // event's complement atom is at most not x in U and is zeroed in L when it
    // is negative-unate.
    class bounds final {
    public:
        bounds(pc::circuit const &circuit, pc::bool_node_id target, std::span<hull::event_mode const> modes = {})
            : circuit_(circuit), target_(target), modes_(modes.begin(), modes.end()), plain_(circuit.node_count()),
              positive_(circuit.node_count()), negative_(circuit.node_count()) {
            std::vector<std::uint8_t> split;
            if (std::ranges::find(modes_, hull::event_mode::split) != modes_.end()) {
                split.assign(circuit.node_count(), 0);
                for (std::uint32_t node = 0, event = 0; node < circuit.node_count(); ++node) {
                    if (circuit.op_codes()[node] == pc::op_code::basic_event) {
                        split[node] = modes_[event++] == hull::event_mode::split ? 1 : 0;
                    }
                }
                atom_plain_.resize(circuit.node_count());
                atom_positive_.resize(circuit.node_count());
            }
            polarity_ = polarities(circuit, target, split, &atom_polarity_);
            split_ = std::move(split);
        }

        // `events[e]`: lane word of event e.
        auto evaluate(std::span<std::uint64_t const> events) -> std::pair<std::uint64_t, std::uint64_t> {
            std::uint32_t event = 0;
            for (std::uint32_t node = 0; node < circuit_.node_count(); ++node) {
                const auto code = circuit_.op_codes()[node];
                const auto operands = circuit_.operands(pc::bool_node_id(node));
                if (code == pc::op_code::basic_event) {
                    const auto word = events[event];
                    const auto mode = modes_.empty() ? hull::event_mode::down : modes_[event];
                    ++event;
                    plain_[node] = polarity_[node] == 2 ? 0 : word;
                    positive_[node] = word;
                    negative_[node] = all_ones;
                    if (mode == hull::event_mode::fixed) {
                        plain_[node] = word;
                        negative_[node] = ~word;
                    } else if (mode == hull::event_mode::up) {
                        plain_[node] = polarity_[node] == 1 ? all_ones : word;
                        positive_[node] = all_ones;
                        negative_[node] = ~word;
                    } else if (mode == hull::event_mode::split) {
                        atom_plain_[node] = atom_polarity_[node] == 2 ? 0 : ~word;
                        atom_positive_[node] = ~word;
                    }
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
                    if (edge.complement && !split_.empty() && split_[source] != 0) {
                        return std::tuple{atom_plain_[source], atom_positive_[source], all_ones};
                    }
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
        std::vector<hull::event_mode> modes_;
        std::vector<std::uint8_t> polarity_;
        std::vector<std::uint8_t> atom_polarity_;
        std::vector<std::uint8_t> split_;
        std::vector<std::uint64_t> plain_;
        std::vector<std::uint64_t> positive_;
        std::vector<std::uint64_t> negative_;
        std::vector<std::uint64_t> atom_plain_;
        std::vector<std::uint64_t> atom_positive_;
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

    using clock = std::chrono::steady_clock;

    auto seconds_since(clock::time_point start) -> double {
        return std::chrono::duration<double>(clock::now() - start).count();
    }

    constexpr std::uint32_t outside = ~std::uint32_t{0};
    constexpr std::size_t block_groups = 256;

    auto literal_probability(pc::circuit const &circuit, std::uint32_t node) -> double {
        auto const &source = circuit.probability_sources()[circuit.probability_source_refs()[node].raw()];
        if (source.kind != pc::probability_source_kind::literal) {
            std::fprintf(stderr, "an event has no literal probability\n");
            std::exit(1);
        }
        return source.literal;
    }

    // The cone of `target`, rebuilt in topological order through the builder.
    // `fixed` (by node, may be empty): 0 replaces the node by FALSE, 1 by TRUE,
    // -1 keeps it; nothing else is folded.
    struct restricted final {
        pc::circuit circuit;
        pc::bool_node_id target;
        // New node to source node, and source node to new node (`outside` when not in the cone).
        std::vector<std::uint32_t> origin;
        std::vector<std::uint32_t> image;
    };

    auto restrict_cone(pc::circuit const &source, pc::bool_node_id target, std::span<std::int8_t const> fixed = {})
        -> restricted {
        restricted out;
        out.image.assign(source.node_count(), outside);
        std::vector<std::uint8_t> seen(source.node_count(), 0);
        std::vector<std::pair<std::uint32_t, std::uint32_t>> stack{{target.raw(), 0}};
        seen[target.raw()] = 1;
        pc::builder builder;
        std::vector<pc::edge_ref> operands;
        while (!stack.empty()) {
            const auto node = stack.back().first;
            const auto kept = fixed.empty() || fixed[node] < 0;
            const auto children = kept ? source.operands(pc::bool_node_id(node)) : std::span<pc::edge_ref const>{};
            if (stack.back().second < children.size()) {
                const auto child = children[stack.back().second++].source.raw();
                if (seen[child] == 0) {
                    seen[child] = 1;
                    stack.emplace_back(child, 0);
                }
                continue;
            }
            stack.pop_back();
            const auto code = source.op_codes()[node];
            pc::bool_node_id made;
            if (!kept) {
                made = builder.add_constant(fixed[node] != 0);
            } else if (code == pc::op_code::basic_event) {
                made = builder.add_basic_event(builder.add_literal_probability(literal_probability(source, node)));
            } else if (code == pc::op_code::constant) {
                made = builder.add_constant(source.constant_values()[node] != 0);
            } else if (code == pc::op_code::house_event) {
                made = builder.add_house_event(source.constant_values()[node] != 0);
            } else {
                operands.clear();
                for (auto const &operand : children) {
                    operands.push_back(pc::edge_ref{.source = pc::bool_node_id(out.image[operand.source.raw()]),
                                                    .complement = operand.complement});
                }
                made = code == pc::op_code::cardinality
                           ? builder.add_cardinality(source.cardinality_low()[node], source.cardinality_high()[node],
                                                     operands)
                           : builder.add_gate(code, operands);
            }
            out.image[node] = made.raw();
            out.origin.push_back(node);
        }
        out.target = pc::bool_node_id(out.image[target.raw()]);
        (void)builder.add_target(out.target);
        auto sealed = std::move(builder).seal();
        if (!sealed.ok()) {
            std::fprintf(stderr, "restricted circuit failed to seal\n");
            std::exit(1);
        }
        out.circuit = std::move(sealed.output);
        return out;
    }

    // Source node of each event node that a pass pipeline kept as itself
    // (`outside` for every other output node).
    auto event_origin(pc::circuit const &input, std::span<passes::node_image const> map, pc::circuit const &output)
        -> std::vector<std::uint32_t> {
        std::vector<std::uint32_t> origin(output.node_count(), outside);
        for (std::uint32_t node = 0; node < input.node_count(); ++node) {
            if (input.op_codes()[node] != pc::op_code::basic_event || map[node].kind != passes::image_kind::reference ||
                map[node].edge.complement) {
                continue;
            }
            const auto image = map[node].edge.source.raw();
            if (output.op_codes()[image] == pc::op_code::basic_event && origin[image] == outside) {
                origin[image] = node;
            }
        }
        return origin;
    }

    // A circuit as the checker sees it. `source` is the lowering node of each
    // event, which is how events of different circuits of one target are matched.
    struct subject final {
        pc::circuit circuit;
        pc::bool_node_id target;
        std::vector<std::uint32_t> source;
        std::vector<hull::event_mode> modes;
    };

    auto make_subject(pc::circuit circuit, pc::bool_node_id target, std::span<std::uint32_t const> source_of_node,
                      std::vector<std::uint8_t> const &literal, hull::event_mode literal_mode) -> subject {
        subject out{.circuit = std::move(circuit), .target = target, .source = {}, .modes = {}};
        for (std::uint32_t node = 0; node < out.circuit.node_count(); ++node) {
            if (out.circuit.op_codes()[node] != pc::op_code::basic_event) {
                continue;
            }
            if (source_of_node[node] == outside) {
                std::fprintf(stderr, "an event has no source in the lowering\n");
                std::exit(1);
            }
            out.source.push_back(source_of_node[node]);
            out.modes.push_back(literal[source_of_node[node]] != 0 ? literal_mode : hull::event_mode::down);
        }
        return out;
    }

    class sampler final {
    public:
        sampler(std::vector<double> probability, std::uint64_t seed)
            : probability_(std::move(probability)), random_(seed), log_miss_(probability_.size()) {
            for (std::size_t event = 0; event < probability_.size(); ++event) {
                log_miss_[event] =
                    probability_[event] > 0.0 && probability_[event] < 1.0 ? std::log1p(-probability_[event]) : 0.0;
            }
        }

        // One word of 64 lanes per event.
        void draw(std::span<std::uint64_t> words) {
            for (std::size_t event = 0; event < probability_.size(); ++event) {
                std::uint64_t word = 0;
                if (probability_[event] >= 1.0) {
                    word = all_ones;
                } else if (probability_[event] > 0.0) {
                    for (auto lane = skip(event); lane < 64; lane += 1 + skip(event)) {
                        word |= std::uint64_t{1} << lane;
                    }
                }
                words[event] = word;
            }
        }

    private:
        // Bernoulli lanes by geometric skipping: about 1 + 64 p draws per event and block.
        auto skip(std::size_t event) -> std::uint32_t {
            const auto gap = std::floor(std::log(1.0 - unit_(random_)) / log_miss_[event]);
            return gap >= 64.0 ? 64U : static_cast<std::uint32_t>(gap);
        }

        std::vector<double> probability_;
        std::mt19937_64 random_;
        std::vector<double> log_miss_;
        std::uniform_real_distribution<double> unit_{0.0, 1.0};
    };

    // Appends, for each lane in `mask` in ascending order, its row (bit e is
    // event e) to `out`. `words` holds one word per event, padded to `stride` * 64.
    void append_rows(std::span<std::uint64_t const> words, std::size_t stride, std::uint64_t mask,
                     std::vector<std::uint64_t> &out) {
        if (mask == 0) {
            return;
        }
        const auto first = out.size();
        out.resize(first + static_cast<std::size_t>(std::popcount(mask)) * stride);
        std::array<std::uint64_t, 64> rows{};
        for (std::size_t group = 0; group < stride; ++group) {
            std::copy_n(words.begin() + static_cast<std::ptrdiff_t>(group * 64), 64, rows.begin());
            hull::detail::transpose64(rows);
            auto row = first + group;
            for (auto bits = mask; bits != 0; bits &= bits - 1) {
                out[row] = rows[static_cast<std::size_t>(std::countr_zero(bits))];
                row += stride;
            }
        }
    }

    // The lanes of `mask` whose answer bit is set; `next` is the answer index of its first lane.
    auto scatter(std::uint64_t mask, std::span<std::uint64_t const> answers, std::size_t &next) -> std::uint64_t {
        std::uint64_t out = 0;
        for (auto bits = mask; bits != 0; bits &= bits - 1, ++next) {
            if (((answers[next >> 6] >> (next & 63)) & 1U) != 0) {
                out |= std::uint64_t{1} << std::countr_zero(bits);
            }
        }
        return out;
    }

    struct checkers final {
        std::optional<hull::checker> serial;
        std::optional<hull::checker> parallel;
    };

    // Cost of deciding the lanes one way: bound evaluation and bookkeeping once, the
    // checks timed for each requested thread count on the same candidates.
    struct way final {
        double bounds_seconds = 0.0;
        double serial_seconds = 0.0;
        double parallel_seconds = 0.0;
        hull::check_stats stats;
    };

    void decide(checkers const &check, std::vector<std::uint64_t> const &rows, std::size_t count, std::size_t stride,
                std::vector<std::uint64_t> &answers, way &cost) {
        answers.assign((count + 63) / 64, 0);
        if (count == 0) {
            return;
        }
        std::optional<run> first;
        std::vector<std::uint64_t> other;
        if (check.serial) {
            first = time_check(*check.serial, rows, count, stride, &answers);
            cost.serial_seconds += first->seconds;
        }
        if (check.parallel) {
            const auto second = time_check(*check.parallel, rows, count, stride, first ? &other : &answers);
            cost.parallel_seconds += second.seconds;
            if (!first) {
                first = second;
            } else if (other != answers) {
                std::fprintf(stderr, "single-threaded and parallel answers differ\n");
                std::exit(1);
            }
        }
        cost.stats.merge(first->stats);
    }

    struct hull_options final {
        std::vector<std::string> targets;
        std::optional<std::optional<std::uint32_t>> max_initiators;
        std::string literals = "atom_split";
        int level = 1;
        bool decompose = false;
        bool verify = false;
        bool serial = true;
        bool parallel = true;
        std::size_t lanes = 0;
        std::uint64_t seed = 0;
    };

    // One circuit of the decomposition: g with the initiating events replaced by
    // constants; `initiator` is the rule circuit's event whose lane bit gates
    // the part, `outside` for g∅.
    struct part final {
        part(subject view_, std::vector<std::uint32_t> from_rule_, std::uint32_t initiator_)
            : view(std::move(view_)), from_rule(std::move(from_rule_)), initiator(initiator_),
              bound(view.circuit, view.target, view.modes), stride((view.source.size() + 63) / 64),
              events(stride * 64, 0) {}

        subject view;
        std::vector<std::uint32_t> from_rule;
        std::uint32_t initiator;
        bounds bound;
        std::size_t stride;
        std::vector<std::uint64_t> events;
        checkers check;
        std::vector<std::uint64_t> rows;
        std::vector<std::uint64_t> candidate_lanes = std::vector<std::uint64_t>(block_groups);
        std::size_t active = 0;
        std::size_t candidates = 0;
    };

    void print_run(char const *label, double seconds, hull::check_stats const &stats, int threads) {
        if (stats.lanes == 0) {
            std::printf("%-28s no lanes checked\n", label);
            return;
        }
        report(label, run{.seconds = seconds, .stats = stats}, threads);
    }

    auto hull_target(ftp_lowering_test::lowering const &lowering, passes::pipeline_result const &compiled,
                     std::size_t index, hull_options const &options, int threads) -> int {
        auto const &binding = lowering.projection.target_bindings()[index];
        std::printf("target %zu:\n", index);
        const auto literal_mode = binding.literals == pra::literal_mode::atom_split ? hull::event_mode::split
                                  : binding.literals == pra::literal_mode::omega_l  ? hull::event_mode::fixed
                                                                                    : hull::event_mode::down;
        std::vector<std::uint8_t> literal(lowering.circuit.node_count(), 0);
        std::vector<std::uint32_t> literal_nodes;
        if (binding.negation.monotone_hull && binding.literals != pra::literal_mode::negation) {
            for (const auto event : lowering.projection.literal_events(binding)) {
                if (const auto node = lowering.projection.node_of(pra::domain_ref(event)); node.valid()) {
                    literal[node.raw()] = 1;
                    literal_nodes.push_back(node.raw());
                }
            }
        }
        std::vector<std::uint32_t> initiator_nodes;
        for (const auto event : lowering.projection.initiators(binding)) {
            if (const auto node = lowering.projection.node_of(pra::domain_ref(event)); node.valid()) {
                initiator_nodes.push_back(node.raw());
            }
        }

        const auto output_target = compiled.target_map[index];
        if (!output_target.valid()) {
            std::fprintf(stderr, "the passes dropped the target\n");
            return 1;
        }
        auto cone = restrict_cone(compiled.output, compiled.output.targets()[output_target.raw()].node);
        const auto origin = event_origin(lowering.circuit, compiled.node_map, compiled.output);
        std::vector<std::uint32_t> source_of_node(cone.circuit.node_count(), outside);
        for (std::uint32_t node = 0; node < cone.circuit.node_count(); ++node) {
            source_of_node[node] = origin[cone.origin[node]];
        }
        const auto rule = make_subject(std::move(cone.circuit), cone.target, source_of_node, literal, literal_mode);
        const auto events = rule.source.size();
        const auto stride = (events + 63) / 64;
        std::printf(
            "circuit: %u nodes, %u edges, %zu events (%td split, %td fixed); %zu initiating events in the cone\n",
            rule.circuit.node_count(), rule.circuit.edge_count(), events,
            std::ranges::count(rule.modes, hull::event_mode::split),
            std::ranges::count(rule.modes, hull::event_mode::fixed), initiator_nodes.size());

        checkers rule_check;
        hull::check_options serial_options{};
        serial_options.parallel = false;
        double prepare_seconds = 0.0;
        for (const auto parallel : {false, true}) {
            if (!(parallel ? options.parallel : options.serial)) {
                continue;
            }
            const auto start = clock::now();
            auto made = hull::checker::make(rule.circuit, rule.target, rule.modes,
                                            parallel ? hull::check_options{} : serial_options);
            if (!made) {
                std::fprintf(stderr, "checker preparation failed\n");
                return 1;
            }
            (parallel ? rule_check.parallel : rule_check.serial) = std::move(made).value();
            prepare_seconds += seconds_since(start);
        }
        auto const &any = rule_check.serial ? *rule_check.serial : *rule_check.parallel;
        std::printf("checker: cone %u positions, %zu events; prepare %.3f s\n", any.cone_size(),
                    any.cone_events().size(), prepare_seconds);

        std::vector<std::unique_ptr<part>> parts;
        std::size_t zero_initiators = 0;
        std::size_t refused = 0;
        if (options.decompose) {
            std::vector<std::uint32_t> event_of(lowering.circuit.node_count(), outside);
            for (std::uint32_t event = 0; event < events; ++event) {
                event_of[rule.source[event]] = event;
            }
            std::vector<std::uint32_t> active_nodes;
            for (const auto node : initiator_nodes) {
                if (literal_probability(lowering.circuit, node) == 0.0) {
                    ++zero_initiators;
                } else if (event_of[node] == outside) {
                    std::fprintf(stderr, "an initiating event is missing from the rule circuit\n");
                    return 1;
                } else if (rule.modes[event_of[node]] != hull::event_mode::down) {
                    ++refused;
                } else {
                    active_nodes.push_back(node);
                }
            }
            if (refused != 0) {
                std::printf("decomposition refused: %zu of %zu initiating events are not in mode down\n", refused,
                            initiator_nodes.size());
                active_nodes.clear();
            } else {
                const auto start = clock::now();
                double passes_seconds = 0.0;
                for (std::size_t which = 0; which <= active_nodes.size(); ++which) {
                    std::vector<std::int8_t> fixed(lowering.circuit.node_count(), -1);
                    for (const auto node : initiator_nodes) {
                        fixed[node] = 0;
                    }
                    if (which > 0) {
                        fixed[active_nodes[which - 1]] = 1;
                    }
                    auto restricted_part = restrict_cone(lowering.circuit, binding.root, fixed);
                    passes::pipeline_options pipeline;
                    pipeline.semantics.push_back(passes::target_semantics::monotone_hull);
                    pipeline.atom_events.resize(1);
                    if (binding.literals == pra::literal_mode::atom_split) {
                        for (const auto node : literal_nodes) {
                            if (restricted_part.image[node] != outside) {
                                pipeline.atom_events[0].push_back(pc::bool_node_id(restricted_part.image[node]));
                            }
                        }
                    }
                    const auto passes_start = clock::now();
                    auto lowered_part = passes::compile(
                        restricted_part.circuit,
                        options.level == 0 ? passes::compile_level::c0 : passes::compile_level::c1, pipeline);
                    passes_seconds += seconds_since(passes_start);
                    if (!lowered_part.ok() || !lowered_part.target_map[0].valid()) {
                        std::fprintf(stderr, "the passes failed on a part of the decomposition\n");
                        return 1;
                    }
                    const auto part_origin =
                        event_origin(restricted_part.circuit, lowered_part.node_map, lowered_part.output);
                    std::vector<std::uint32_t> part_source(lowered_part.output.node_count(), outside);
                    for (std::uint32_t node = 0; node < part_source.size(); ++node) {
                        part_source[node] =
                            part_origin[node] == outside ? outside : restricted_part.origin[part_origin[node]];
                    }
                    auto view = make_subject(lowered_part.output,
                                             lowered_part.output.targets()[lowered_part.target_map[0].raw()].node,
                                             part_source, literal, literal_mode);
                    std::vector<std::uint32_t> from_rule;
                    for (const auto node : view.source) {
                        if (event_of[node] == outside) {
                            std::fprintf(stderr, "an event of a part is missing from the rule circuit\n");
                            return 1;
                        }
                        from_rule.push_back(event_of[node]);
                    }
                    auto &added = parts.emplace_back(
                        std::make_unique<part>(std::move(view), std::move(from_rule),
                                               which == 0 ? outside : event_of[active_nodes[which - 1]]));
                    for (const auto parallel : {false, true}) {
                        if (!(parallel ? options.parallel : options.serial)) {
                            continue;
                        }
                        auto made = hull::checker::make(added->view.circuit, added->view.target, added->view.modes,
                                                        parallel ? hull::check_options{} : serial_options);
                        if (!made) {
                            std::fprintf(stderr, "checker preparation failed\n");
                            return 1;
                        }
                        (parallel ? added->check.parallel : added->check.serial) = std::move(made).value();
                    }
                }
                std::size_t nodes = 0;
                std::size_t edges = 0;
                for (auto const &item : parts) {
                    nodes += item->view.circuit.node_count();
                    edges += item->view.circuit.edge_count();
                }
                std::printf(
                    "decomposition: %zu parts (%zu initiating events, %zu with probability 0 left out), %zu nodes, "
                    "%zu edges "
                    "in all; passes %.3f s, parts and checkers prepared in %.3f s\n",
                    parts.size(), initiator_nodes.size(), zero_initiators, nodes, edges, passes_seconds,
                    seconds_since(start));
            }
        }
        std::fflush(stdout);

        const auto probability = probabilities(rule.circuit);
        sampler draw(probability, options.seed);
        bounds rule_bounds(rule.circuit, rule.target, rule.modes);
        const auto groups = options.lanes / 64;
        std::vector<std::uint64_t> words(block_groups * stride * 64, 0);
        std::vector<std::uint64_t> rule_rows;
        std::vector<std::uint64_t> answers;
        std::vector<std::uint64_t> rule_true(block_groups);
        std::vector<std::uint64_t> rule_candidates(block_groups);
        std::vector<std::uint64_t> decomposed(block_groups);
        std::size_t lower_ones = 0;
        std::size_t upper_ones = 0;
        std::size_t rule_candidate_count = 0;
        std::size_t rule_hits = 0;
        std::size_t decomposition_hits = 0;
        std::size_t compared = 0;
        std::size_t differences = 0;
        double sampling_seconds = 0.0;
        way rule_way;
        way decomposition_way;
        for (std::size_t first = 0; first < groups; first += block_groups) {
            const auto count = std::min(block_groups, groups - first);
            auto start = clock::now();
            for (std::size_t group = 0; group < count; ++group) {
                draw.draw(std::span<std::uint64_t>(words).subspan(group * stride * 64, events));
            }
            sampling_seconds += seconds_since(start);

            start = clock::now();
            rule_rows.clear();
            std::size_t candidates = 0;
            for (std::size_t group = 0; group < count; ++group) {
                const std::span<std::uint64_t const> lane_words(words.data() + group * stride * 64, stride * 64);
                const auto [lower, upper] = rule_bounds.evaluate(lane_words.first(events));
                lower_ones += static_cast<std::size_t>(std::popcount(lower));
                upper_ones += static_cast<std::size_t>(std::popcount(upper));
                rule_true[group] = lower;
                rule_candidates[group] = ~lower & upper;
                candidates += static_cast<std::size_t>(std::popcount(rule_candidates[group]));
                append_rows(lane_words, stride, rule_candidates[group], rule_rows);
            }
            rule_way.bounds_seconds += seconds_since(start);
            decide(rule_check, rule_rows, candidates, stride, answers, rule_way);
            rule_candidate_count += candidates;
            start = clock::now();
            std::size_t next = 0;
            for (std::size_t group = 0; group < count; ++group) {
                rule_true[group] |= scatter(rule_candidates[group], answers, next);
                rule_hits += static_cast<std::size_t>(std::popcount(rule_true[group]));
            }
            rule_way.bounds_seconds += seconds_since(start);

            if (parts.empty()) {
                continue;
            }
            std::fill_n(decomposed.begin(), count, 0);
            for (auto &item : parts) {
                start = clock::now();
                item->rows.clear();
                std::size_t part_candidates = 0;
                for (std::size_t group = 0; group < count; ++group) {
                    const auto lane_words = words.data() + group * stride * 64;
                    for (std::size_t event = 0; event < item->from_rule.size(); ++event) {
                        item->events[event] = lane_words[item->from_rule[event]];
                    }
                    const auto [lower, upper] = item->bound.evaluate(
                        std::span<std::uint64_t const>(item->events).first(item->from_rule.size()));
                    const auto active = item->initiator == outside ? all_ones : lane_words[item->initiator];
                    item->active += static_cast<std::size_t>(std::popcount(active));
                    decomposed[group] |= active & lower;
                    item->candidate_lanes[group] = active & ~decomposed[group] & upper;
                    part_candidates += static_cast<std::size_t>(std::popcount(item->candidate_lanes[group]));
                    append_rows(item->events, item->stride, item->candidate_lanes[group], item->rows);
                }
                item->candidates += part_candidates;
                decomposition_way.bounds_seconds += seconds_since(start);
                decide(item->check, item->rows, part_candidates, item->stride, answers, decomposition_way);
                start = clock::now();
                next = 0;
                for (std::size_t group = 0; group < count; ++group) {
                    decomposed[group] |= scatter(item->candidate_lanes[group], answers, next);
                }
                decomposition_way.bounds_seconds += seconds_since(start);
            }
            for (std::size_t group = 0; group < count; ++group) {
                compared += 64;
                differences += static_cast<std::size_t>(std::popcount(rule_true[group] ^ decomposed[group]));
                decomposition_hits += static_cast<std::size_t>(std::popcount(decomposed[group]));
            }
        }

        const auto lanes = static_cast<double>(options.lanes);
        std::printf("sampling: %.3f s for %zu lanes\n", sampling_seconds, options.lanes);
        std::printf("rule: N %zu; L = 1 %.6f; U = 1 %.6f; candidates %.6f (%zu); hits %.6f; bounds %.3f s\n",
                    options.lanes, double(lower_ones) / lanes, double(upper_ones) / lanes,
                    double(rule_candidate_count) / lanes, rule_candidate_count, double(rule_hits) / lanes,
                    rule_way.bounds_seconds);
        if (options.serial) {
            print_run("rule checks, 1 thread", rule_way.serial_seconds, rule_way.stats, 1);
            std::printf("%-28s %.3f s, %.0f lanes/s\n", "rule bounds + checks, 1 thread",
                        rule_way.bounds_seconds + rule_way.serial_seconds,
                        lanes / (rule_way.bounds_seconds + rule_way.serial_seconds));
        }
        if (options.parallel) {
            print_run("rule checks, all threads", rule_way.parallel_seconds, rule_way.stats, threads);
            std::printf("%-28s %.3f s, %.0f lanes/s\n", "rule bounds + checks, all thr.",
                        rule_way.bounds_seconds + rule_way.parallel_seconds,
                        lanes / (rule_way.bounds_seconds + rule_way.parallel_seconds));
        }
        if (parts.empty()) {
            return options.verify && refused != 0 ? 1 : 0;
        }
        double candidate_sum = 0.0;
        double candidate_max = 0.0;
        double active_sum = 0.0;
        std::size_t checks = 0;
        for (auto const &item : parts) {
            const auto fraction = double(item->candidates) / lanes;
            candidate_sum += fraction;
            candidate_max = std::max(candidate_max, fraction);
            active_sum += double(item->active) / lanes;
            checks += item->candidates;
        }
        std::printf(
            "decomposition: N %zu; %zu parts; candidates per part: mean %.6f max %.6f of N; active mean %.6f of N; "
            "checks per lane %.6f (rule %.6f); hits %.6f; bounds %.3f s\n",
            options.lanes, parts.size(), candidate_sum / double(parts.size()), candidate_max,
            active_sum / double(parts.size()), double(checks) / lanes, double(rule_candidate_count) / lanes,
            double(decomposition_hits) / lanes, decomposition_way.bounds_seconds);
        if (options.serial) {
            print_run("part checks, 1 thread", decomposition_way.serial_seconds, decomposition_way.stats, 1);
            std::printf("%-28s %.3f s, %.0f lanes/s\n", "decision, 1 thread",
                        decomposition_way.bounds_seconds + decomposition_way.serial_seconds,
                        lanes / (decomposition_way.bounds_seconds + decomposition_way.serial_seconds));
        }
        if (options.parallel) {
            print_run("part checks, all threads", decomposition_way.parallel_seconds, decomposition_way.stats, threads);
            std::printf("%-28s %.3f s, %.0f lanes/s\n", "decision, all threads",
                        decomposition_way.bounds_seconds + decomposition_way.parallel_seconds,
                        lanes / (decomposition_way.bounds_seconds + decomposition_way.parallel_seconds));
        }
        if (options.verify) {
            std::printf("verify: lanes compared %zu, differences %zu\n", compared, differences);
            return differences != 0 || compared == 0 ? 1 : 0;
        }
        return 0;
    }

    auto run_hull(std::vector<std::string> const &files, hull_options const &options) -> int {
        const auto sources = read_sources(files);
        compile_ftp::options settings;
        settings.negation = "hull";
        settings.literals = options.literals;
        settings.max_initiators_per_cutset = options.max_initiators;
        settings.targets = options.targets;
        auto start = clock::now();
        auto lowered = ftp_lowering_test::lower(sources, settings);
        const auto lower_seconds = seconds_since(start);
        if (!lowered) {
            std::fprintf(stderr, "FTP lowering failed (%zu diagnostics)\n", lowered.diagnostics().size());
            return 1;
        }
        auto const &lowering = lowered.value();
        const auto rule = lowering.report.max_initiators_per_cutset;
        if (options.decompose && rule != std::optional<std::uint32_t>{1}) {
            std::fprintf(stderr, "--decompose-initiators needs an initiator rule with k = 1\n");
            return 2;
        }
        auto const bindings = lowering.projection.target_bindings();
        passes::pipeline_options pipeline;
        for (auto const &binding : bindings) {
            pipeline.semantics.push_back(binding.negation.monotone_hull ? passes::target_semantics::monotone_hull
                                                                        : passes::target_semantics::boolean);
            auto &atoms = pipeline.atom_events.emplace_back();
            if (binding.negation.monotone_hull && binding.literals == pra::literal_mode::atom_split) {
                for (const auto event : lowering.projection.literal_events(binding)) {
                    if (const auto node = lowering.projection.node_of(pra::domain_ref(event)); node.valid()) {
                        atoms.push_back(node);
                    }
                }
            }
        }
        const auto rule_text = rule ? "k = " + std::to_string(*rule) : std::string("none");
        std::printf(
            "lowered: %u nodes, %u edges, %zu targets; initiator rule %s (%zu nodes, %zu edges); lower %.3f s\n",
            lowering.circuit.node_count(), lowering.circuit.edge_count(), bindings.size(), rule_text.c_str(),
            lowering.report.generated_nodes, lowering.report.generated_edges, lower_seconds);
        start = clock::now();
        auto compiled = passes::compile(
            lowering.circuit, options.level == 0 ? passes::compile_level::c0 : passes::compile_level::c1, pipeline);
        const auto passes_seconds = seconds_since(start);
        if (!compiled.ok()) {
            std::fprintf(stderr, "the passes failed\n");
            return 1;
        }
        std::printf("level %d: %u nodes, %u edges; passes %.3f s\n", options.level, compiled.nodes_after,
                    compiled.edges_after, passes_seconds);
        int threads = 1;
#if defined(_OPENMP)
        threads = omp_get_max_threads();
#endif
        int status = 0;
        for (std::size_t index = 0; index < bindings.size(); ++index) {
            status = std::max(status, hull_target(lowering, compiled, index, options, threads));
        }
        std::printf("threads: %d\n", threads);
        return status;
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
    std::vector<std::string> target_names;
    hull_options hull_settings;
    bool hull_mode = false;
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
            target_names.push_back(next());
        } else if (argument == "--split-complements") {
            split_complements = true;
        } else if (argument == "--verify") {
            verify = true;
        } else if (argument == "--hull") {
            hull_mode = true;
        } else if (argument == "--max-initiators") {
            const auto value = next();
            hull_settings.max_initiators =
                value == "none" ? std::optional<std::uint32_t>{}
                                : std::optional<std::uint32_t>{static_cast<std::uint32_t>(std::stoul(value))};
        } else if (argument == "--literals") {
            hull_settings.literals = next();
        } else if (argument == "--level") {
            hull_settings.level = std::stoi(next());
        } else if (argument == "--decompose-initiators") {
            hull_settings.decompose = true;
        } else if (argument == "--verify-decomposition") {
            hull_settings.verify = true;
        } else if (argument == "--check-threads") {
            const auto value = next();
            hull_settings.serial = value == "1" || value == "both";
            hull_settings.parallel = value == "all" || value == "both";
            if (!hull_settings.serial && !hull_settings.parallel) {
                std::fprintf(stderr, "--check-threads takes 1, all or both\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown argument\n");
            return 2;
        }
    }
    lanes = std::max<std::size_t>(64, (lanes + 63) / 64 * 64);

    if (hull_mode) {
        if (files.empty() || (hull_settings.level != 0 && hull_settings.level != 1) ||
            (hull_settings.literals != "atom_split" && hull_settings.literals != "omega_l" &&
             hull_settings.literals != "negation") ||
            (hull_settings.verify && !hull_settings.decompose)) {
            std::fprintf(stderr, "--hull needs --ftp, --level 0|1, --literals atom_split|omega_l|negation, and "
                                 "--verify-decomposition needs --decompose-initiators\n");
            return 2;
        }
        hull_settings.targets = target_names;
        hull_settings.lanes = lanes;
        hull_settings.seed = seed;
        return run_hull(files, hull_settings);
    }
    if (hull_settings.decompose || hull_settings.verify) {
        std::fprintf(stderr, "--decompose-initiators and --verify-decomposition need --hull\n");
        return 2;
    }

    std::mt19937_64 random(seed);
    const auto subject = files.empty() ? synthetic(nodes, exclusive_probability, random)
                                       : read_ftp(files, target_names.empty() ? std::string{} : target_names.front());
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
