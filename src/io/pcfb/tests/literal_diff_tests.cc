#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <pcfb.h>

#include "support.h"

#include "compile/pra_to_pc/compose.h"
#include "compile/pra_to_pc/descriptor.h"
#include "io/pcfb/descriptor.h"
#include "io/pcfb/detail/model.h"
#include "ir/pc/op_code.h"
#include "ir/pc/passes/passes.h"
#include "ir/pra/builder.h"
#include "ir/pra/gate.h"
#include "ir/pra/model.h"
#include "ir/pra/projection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Literal-atom differential test (RFC §7.5, pcfb SPEC §5.1.1).
//
// Reference: a brute-force evaluator of the three modes over the *lowered* ir::pc circuit (the pipeline input),
// built from the definitions only. For every target it is compared with
//   (a) the same target after the C1 pass pipeline, evaluated with the Δ the pipeline carried, and
//   (b) the same target after writer → reader, evaluated from the read circuit with the read
//       `Query.literal_mode` and `Query.literal_samples` (mapped back to nodes through `node_by_tensor`).
// Both must match to kTolerance, so a pass that is not atom-preserving near Δ, or a writer or reader that loses
// or shifts Δ, fails. The evaluator shares no code with the lowering, the passes or the codec.
namespace {

    namespace pc = fuzzybits::ir::pc;
    namespace pra = fuzzybits::ir::pra;
    namespace passes = fuzzybits::ir::pc::passes;
    namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;
    namespace lowering = fuzzybits::compile::backend::fuzzybits::pra_to_pc;

    // Absolute tolerance on a probability. Both sides sum the same world masses in binary64, so only the
    // accumulation order differs. A side sums at most 2^12 terms of magnitude at most 1; each product's relative
    // error is at most (nodes + 1) · 2^-53 < 2^-46 and plain left-to-right summation adds at most 2^12 · 2^-53 <
    // 2^-41, so 2^-41 ≈ 4.5e-13 bounds one side and 1e-12 bounds the difference. The evaluator uses Neumaier
    // compensation, so the real error is far below that.
    constexpr double kTolerance = 1e-12;

    constexpr std::uint32_t kMaxEvents = 12;
    constexpr std::uint32_t kMaxAtoms = 16;

    // --- evaluation ------------------------------------------------------------------------------------

    // Value of `target` at one atom assignment. `event_bits` gives the events in `events` order; `atom_bits`
    // gives the complement atom of each entry of `delta`, in order. A complemented edge into a delta event reads
    // its complement atom; every other complemented edge (in particular every gate complement) is a Boolean
    // complement.
    [[nodiscard]] auto evaluate(pc::circuit const &model, std::span<std::uint32_t const> events, std::span<std::uint32_t const> delta,
                                const std::uint64_t event_bits, const std::uint64_t atom_bits, const std::uint32_t target) -> bool {
        std::vector<std::uint8_t> value(model.node_count(), 0);
        for (std::size_t index = 0; index < events.size(); ++index) {
            value[events[index]] = ((event_bits >> index) & 1U) != 0 ? 1 : 0;
        }
        for (std::uint32_t node = 0; node < model.node_count(); ++node) {
            const auto code = model.op_codes()[node];
            if (code == pc::op_code::basic_event) {
                continue;
            }
            if (code == pc::op_code::constant || code == pc::op_code::house_event) {
                value[node] = model.constant_values()[node] != 0 ? 1 : 0;
                continue;
            }
            std::uint32_t trues = 0;
            std::uint32_t arity = 0;
            for (auto const &edge : model.operands(pc::bool_node_id(node))) {
                bool bit = value[edge.source.raw()] != 0;
                if (edge.complement) {
                    const auto found = std::ranges::find(delta, edge.source.raw());
                    if (found != delta.end()) {
                        bit = ((atom_bits >> static_cast<std::size_t>(found - delta.begin())) & 1U) != 0;
                    } else {
                        bit = !bit;
                    }
                }
                trues += bit ? 1U : 0U;
                ++arity;
            }
            bool result = false;
            switch (code) {
            case pc::op_code::and_gate:
                result = trues == arity;
                break;
            case pc::op_code::or_gate:
                result = trues != 0;
                break;
            case pc::op_code::xor_gate:
                result = (trues & 1U) != 0;
                break;
            case pc::op_code::not_gate:
            case pc::op_code::nor_gate:
                result = trues == 0;
                break;
            case pc::op_code::nand_gate:
                result = trues != arity;
                break;
            case pc::op_code::xnor_gate:
                result = (trues & 1U) == 0;
                break;
            case pc::op_code::cardinality:
                result = trues >= model.cardinality_low()[node] && trues <= model.cardinality_high()[node];
                break;
            default:
                break;
            }
            value[node] = result ? 1 : 0;
        }
        return value[target] != 0;
    }

    struct circuit_view final {
        std::vector<std::uint32_t> events;
        std::vector<double> probability;
    };

    [[nodiscard]] auto view_of(pc::circuit const &model) -> circuit_view {
        circuit_view view;
        for (std::uint32_t node = 0; node < model.node_count(); ++node) {
            if (model.op_codes()[node] == pc::op_code::basic_event) {
                view.events.push_back(node);
                view.probability.push_back(model.probability_sources()[model.probability_source_refs()[node].raw()].literal);
            }
        }
        REQUIRE(view.events.size() <= kMaxEvents);
        return view;
    }

    // The exact value of one hull query, brute force over the definitions of SPEC §5.1.1. `delta` may name nodes
    // that are not events of this circuit (a listed sample the pipeline dropped): such entries have no effect,
    // which is what the definitions say, so they are ignored. This is why Δ ∩ cone may shrink.
    [[nodiscard]] auto exact_hull(pc::circuit const &model, circuit_view const &view, std::span<std::uint32_t const> delta,
                                  const pcfb::LiteralMode mode, const std::uint32_t target) -> double {
        std::vector<std::uint32_t> listed;
        if (mode != pcfb::LiteralMode::NEGATION) {
            for (std::size_t index = 0; index < view.events.size(); ++index) {
                if (std::ranges::find(delta, view.events[index]) != delta.end()) {
                    listed.push_back(view.events[index]);
                }
            }
        }
        const auto n_events = view.events.size();
        const auto n_atoms = n_events + listed.size();
        REQUIRE(n_atoms <= kMaxAtoms);

        const std::uint64_t worlds = std::uint64_t{1} << n_events;
        std::vector<double> mass(worlds, 1.0);
        for (std::uint64_t world = 0; world < worlds; ++world) {
            for (std::size_t index = 0; index < n_events; ++index) {
                mass[world] *= ((world >> index) & 1U) != 0 ? view.probability[index] : 1.0 - view.probability[index];
            }
        }

        double sum = 0.0;
        double compensation = 0.0;
        const auto accumulate = [&](const double term) {
            // Neumaier compensation.
            const double adjusted = term - compensation;
            const double next = sum + adjusted;
            compensation = (next - sum) - adjusted;
            sum = next;
        };

        if (mode == pcfb::LiteralMode::ATOM_SPLIT) {
            // g̃ over the atoms, closed over every atom axis, weighed at the coupled valuation κ(x): the
            // complement atom of a listed event s is 1 when x_s = 0.
            const std::uint64_t points = std::uint64_t{1} << n_atoms;
            std::vector<std::uint8_t> hull(points, 0);
            for (std::uint64_t point = 0; point < points; ++point) {
                hull[point] = evaluate(model, view.events, listed, point & (worlds - 1), point >> n_events, target) ? 1 : 0;
            }
            for (std::uint32_t bit = 0; bit < n_atoms; ++bit) {
                for (std::uint64_t point = 0; point < points; ++point) {
                    if (((point >> bit) & 1U) != 0) {
                        hull[point] = static_cast<std::uint8_t>(hull[point] | hull[point ^ (std::uint64_t{1} << bit)]);
                    }
                }
            }
            for (std::uint64_t world = 0; world < worlds; ++world) {
                std::uint64_t point = world;
                for (std::size_t at = 0; at < listed.size(); ++at) {
                    const auto slot = static_cast<std::size_t>(std::ranges::find(view.events, listed[at]) - view.events.begin());
                    if (((world >> slot) & 1U) == 0) {
                        point |= std::uint64_t{1} << (n_events + at);
                    }
                }
                if (hull[point] != 0) {
                    accumulate(mass[world]);
                }
            }
            return sum;
        }

        // OMEGA_L and NEGATION read the plain target g. OMEGA_L does not close the axis of a listed event, because
        // ω_L fixes it (RFC §3); NEGATION closes every axis.
        std::vector<std::uint8_t> hull(worlds, 0);
        for (std::uint64_t world = 0; world < worlds; ++world) {
            hull[world] = evaluate(model, view.events, {}, world, 0, target) ? 1 : 0;
        }
        for (std::size_t index = 0; index < n_events; ++index) {
            if (std::ranges::find(listed, view.events[index]) != listed.end()) {
                continue;
            }
            for (std::uint64_t world = 0; world < worlds; ++world) {
                if (((world >> index) & 1U) != 0) {
                    hull[world] = static_cast<std::uint8_t>(hull[world] | hull[world ^ (std::uint64_t{1} << index)]);
                }
            }
        }
        for (std::uint64_t world = 0; world < worlds; ++world) {
            if (hull[world] != 0) {
                accumulate(mass[world]);
            }
        }
        return sum;
    }

    // --- one seeded case --------------------------------------------------------------------------------

    [[nodiscard]] auto delta_nodes(lowering::pra_pc_lowering const &artifact, const std::size_t target_index) -> std::vector<std::uint32_t> {
        std::vector<std::uint32_t> delta;
        for (const auto event : artifact.projection.literal_events(artifact.projection.target_bindings()[target_index])) {
            const auto node = artifact.projection.node_of(pra::domain_ref(event));
            if (node.valid()) {
                delta.push_back(node.raw());
            }
        }
        std::ranges::sort(delta);
        delta.erase(std::ranges::unique(delta).begin(), delta.end());
        return delta;
    }

    [[nodiscard]] auto probability_of(std::mt19937_64 &rng) -> double {
        switch (pcfb_test_random::uniform(rng, 0, 5)) {
        case 0:
            return 0.0;
        case 1:
            return 1.0;
        case 2:
            return 0.5;
        default:
            return 0.01 + 0.98 * pcfb_test_random::unit(rng);
        }
    }

    struct Case final {
        lowering::pra_pc_lowering artifact;
        std::vector<std::vector<std::uint32_t>> delta;
    };

    // A random model: 2..8 basic events with p ∈ {0, 1, 0.5, random}; a DAG of AND / OR / XOR / NAND / NOR /
    // XNOR / NOT / cardinality gates over events and gates, with complements and `literal` marks on complemented
    // basic-event operands, and shared subtrees; 1..3 targets on gates.
    [[nodiscard]] auto random_case(std::mt19937_64 &rng, const pra::literal_mode mode) -> Case {
        pra::builder builder;
        builder.set_project("literals");
        const auto n_events = static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, 2, 8));
        std::vector<pra::domain_ref> nodes;
        for (std::uint32_t index = 0; index < n_events; ++index) {
            nodes.push_back(pra::domain_ref(builder.add_basic_event({.stable_name = "E" + std::to_string(index)},
                                                                  pra::expression_spec::literal(probability_of(rng)))));
        }
        const auto tree = builder.add_fault_tree({.stable_name = "FT"});
        const auto n_gates = static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, 1, 6));
        for (std::uint32_t gate_index = 0; gate_index < n_gates; ++gate_index) {
            const auto kind = static_cast<pra::gate_kind>(pcfb_test_random::uniform(rng, 0, 7));
            const auto arity = kind == pra::gate_kind::not_gate ? 1U : static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, 1, 3));
            std::vector<pra::operand_spec> operands;
            for (std::uint32_t at = 0; at < arity; ++at) {
                const auto child = nodes[static_cast<std::size_t>(pcfb_test_random::uniform(rng, 0, nodes.size() - 1))];
                pra::operand_spec operand{.event = child};
                // A `literal` mark is admitted only on a complemented basic-event operand (ir::pra rule).
                if (child.kind() == pra::element_kind::basic_event && pcfb_test_random::chance(rng, 0.35)) {
                    operand.complement = true;
                    operand.literal = true;
                } else if (pcfb_test_random::chance(rng, 0.35)) {
                    operand.complement = true;
                }
                operands.push_back(operand);
            }
            const auto name = "G" + std::to_string(gate_index);
            if (kind == pra::gate_kind::cardinality) {
                const auto n = static_cast<std::uint32_t>(operands.size());
                const auto low = static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, 0, n));
                const auto high = static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, low, n));
                nodes.push_back(pra::domain_ref(
                    builder.add_cardinality({.stable_name = name}, pra::cardinality_bounds{.low = low, .high = high}, operands, tree)));
            } else {
                nodes.push_back(pra::domain_ref(builder.add_gate({.stable_name = name}, kind, operands, tree)));
            }
        }
        std::vector<pra::domain_ref> targets;
        const auto n_targets = static_cast<std::uint32_t>(pcfb_test_random::uniform(rng, 1, 3));
        for (std::uint32_t index = 0; index < n_targets; ++index) {
            const auto choice = nodes[static_cast<std::size_t>(pcfb_test_random::uniform(rng, n_events, nodes.size() - 1))];
            if (std::ranges::find(targets, choice) != targets.end()) {
                continue;
            }
            targets.push_back(choice);
            (void)builder.add_target({.stable_name = "T" + std::to_string(targets.size() - 1)}, pra::domain_target_kind::gate, choice);
        }
        auto sealed = std::move(builder).seal();
        INFO(pcfb_writer_test::diagnosticsText(sealed.diagnostics));
        REQUIRE(sealed.ok());

        lowering::quantification_config config;
        for (std::size_t index = 0; index < targets.size(); ++index) {
            config.targets.push_back({.name = "T" + std::to_string(index), .negation = pra::negation_set::both(), .literals = mode});
        }
        auto lowered = lowering::lower(sealed.output, config);
        // `compile.pra_to_pc.mixed-literal` (RFC §7.1) is the one model the per-query list cannot represent: an
        // event with both marked and unmarked complemented operands in one cone. The FTP reader never produces it,
        // so such a case is regenerated rather than counted; the diagnostic itself is covered by
        // src/compile/pra_to_pc/tests/literal_tests.cc.
        if (!lowered && pcfb_writer_test::hasCode(lowered.diagnostics(), "compile.pra_to_pc.mixed-literal")) {
            return random_case(rng, mode);
        }
        INFO(pcfb_writer_test::diagnosticsText(lowered.diagnostics()));
        REQUIRE(lowered);
        Case out{.artifact = std::move(lowered).value()};
        for (std::size_t index = 0; index < out.artifact.projection.target_bindings().size(); ++index) {
            out.delta.push_back(delta_nodes(out.artifact, index));
        }
        return out;
    }

    // The pipeline options fzb compile builds: per-target semantics, and Δ for every atom-split hull target.
    [[nodiscard]] auto pipeline_for(lowering::pra_pc_lowering const &artifact, std::span<std::vector<std::uint32_t> const> delta)
        -> passes::pipeline_options {
        passes::pipeline_options options;
        const auto bindings = artifact.projection.target_bindings();
        options.semantics.reserve(bindings.size());
        options.atom_events.resize(bindings.size());
        for (std::size_t index = 0; index < bindings.size(); ++index) {
            // A target asking for both kinds takes the stricter pass rules.
            options.semantics.push_back(bindings[index].negation.monotone_hull ? passes::target_semantics::monotone_hull
                                                                              : passes::target_semantics::boolean);
            if (bindings[index].negation.monotone_hull && bindings[index].literals == pra::literal_mode::atom_split) {
                for (const auto node : delta[index]) {
                    options.atom_events[index].push_back(pc::bool_node_id(node));
                }
            }
        }
        return options;
    }

    // Δ through the pipeline's composed node map, kept only where an output node still computes the event.
    [[nodiscard]] auto carried_delta(passes::pipeline_result const &compiled, std::span<std::uint32_t const> delta) -> std::vector<std::uint32_t> {
        std::vector<std::uint32_t> out;
        for (const auto node : delta) {
            if (node >= compiled.node_map.size()) {
                continue;
            }
            const auto &image = compiled.node_map[node];
            if (image.kind == passes::image_kind::reference && !image.edge.complement &&
                image.edge.source.raw() < compiled.output.op_codes().size() &&
                compiled.output.op_codes()[image.edge.source.raw()] == pc::op_code::basic_event) {
                out.push_back(image.edge.source.raw());
            }
        }
        std::ranges::sort(out);
        out.erase(std::ranges::unique(out).begin(), out.end());
        return out;
    }

    // What the writer is told about each target of a post-pipeline circuit: name, both kinds, the literal mode and
    // the carried Δ. `compose_write_options` does the same re-keying in the real driver.
    [[nodiscard]] auto options_for(lowering::pra_pc_lowering const &artifact, passes::pipeline_result const &compiled,
                                   std::span<std::vector<std::uint32_t> const> delta, const pra::literal_mode mode) -> writer::write_options {
        writer::write_options options;
        const auto bindings = artifact.projection.target_bindings();
        for (std::size_t index = 0; index < bindings.size(); ++index) {
            const auto output_target = compiled.target_map[bindings[index].pc_target.raw()];
            REQUIRE(output_target.valid());
            options.target_names[output_target] = "T" + std::to_string(index);
            options.target_kinds[output_target] = {writer::query_kind::probability, writer::query_kind::monotone_hull_probability};
            writer::target_quantification quantification{
                .literal_mode = mode == pra::literal_mode::atom_split ? "atom_split"
                                                                     : mode == pra::literal_mode::omega_l ? "omega_l" : "negation",
            };
            for (const auto node : carried_delta(compiled, delta[index])) {
                quantification.literal_events.push_back(pc::bool_node_id(node));
            }
            options.quantification[output_target] = std::move(quantification);
        }
        options.model_name = "literals";
        return options;
    }

} // namespace

namespace {

    // pcfb object-API plumbing, as read_tests.cc has it, for files the builder cannot produce.
    [[nodiscard]] auto pack_file(pcfb::CircuitT const &model) -> std::vector<std::byte> {
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(pcfb::Circuit::Pack(builder, &model), pcfb::CircuitIdentifier());
        const auto *data = reinterpret_cast<const std::byte *>(builder.GetBufferPointer());
        return {data, data + builder.GetSize()};
    }

    [[nodiscard]] auto unpack_file(std::span<const std::byte> bytes) -> std::unique_ptr<pcfb::CircuitT> {
        auto circuit = pcfb::io::from_bytes(bytes);
        REQUIRE(circuit);
        return std::unique_ptr<pcfb::CircuitT>((*circuit).get()->UnPack());
    }

    [[nodiscard]] auto read_file_ok(std::span<const std::byte> bytes) -> writer::read_output {
        auto read = writer::read(bytes);
        INFO(pcfb_writer_test::diagnosticsText(read.diagnostics()));
        REQUIRE(read);
        return std::move(read).value();
    }

} // namespace


TEST_CASE("literal atoms: the pass pipeline and the pcfb round trip preserve the exact hull value (2000 seeded cases)",
          "[io][pcfb][literals][diff]") {
    // Seeded and deterministic: std::mt19937_64 with portable draws only (portable_random.h), so libc++ and
    // libstdc++ produce the same cases.
    std::mt19937_64 rng(0x5ca1ab1e1eeULL);

    struct tally final {
        std::uint64_t circuits = 0;
        std::array<std::uint64_t, 3> comparisons{};
        std::uint64_t non_empty_delta = 0;
        std::uint64_t rewrites = 0;
        std::uint64_t skipped = 0;
    } stats;

    constexpr std::array requested{pra::literal_mode::atom_split, pra::literal_mode::omega_l, pra::literal_mode::negation};
    constexpr std::array modes{pcfb::LiteralMode::ATOM_SPLIT, pcfb::LiteralMode::OMEGA_L, pcfb::LiteralMode::NEGATION};

    for (std::uint32_t iteration = 0; iteration < 2000; ++iteration) {
        // Cycle the literal mode, so each sees a third of the random circuits.
        const auto mode_index = static_cast<std::size_t>(iteration % modes.size());
        INFO("iteration " << iteration << " mode " << static_cast<int>(modes[mode_index]));
        auto test = random_case(rng, requested[mode_index]);
        ++stats.circuits;

        const auto &input = test.artifact.circuit;
        const auto input_view = view_of(input);

        // The pipeline exactly as fzb compile runs it.
        auto compiled = passes::compile(input, passes::compile_level::c1, pipeline_for(test.artifact, test.delta));
        INFO(pcfb_writer_test::diagnosticsText(compiled.diagnostics));
        REQUIRE(compiled.ok());
        for (const auto &report : compiled.passes) {
            stats.rewrites += report.rewrites;
            stats.skipped += report.skipped;
        }
        const auto output_view = view_of(compiled.output);

        // Writer → reader of the post-pipeline circuit.
        auto options = options_for(test.artifact, compiled, test.delta, requested[mode_index]);
        auto written = writer::write(compiled.output.to_handle(), options);
        INFO(pcfb_writer_test::diagnosticsText(written.diagnostics()));
        REQUIRE(written);
        auto read = writer::read(std::span<std::byte const>{written.value().bytes});
        INFO(pcfb_writer_test::diagnosticsText(read.diagnostics()));
        REQUIRE(read);
        const auto read_view = view_of(read.value().circuit);

        const auto bindings = test.artifact.projection.target_bindings();
        for (std::size_t index = 0; index < bindings.size(); ++index) {
            if (!bindings[index].negation.monotone_hull) {
                continue;
            }
            INFO("target " << index);
            REQUIRE(test.delta[index].size() == bindings[index].literal_events.count);
            const auto expected = exact_hull(input, input_view, test.delta[index], modes[mode_index], bindings[index].node.raw());
            const auto output_target = compiled.target_map[bindings[index].pc_target.raw()];
            REQUIRE(output_target.valid());
            const auto carried = carried_delta(compiled, test.delta[index]);
            if (!test.delta[index].empty() && !carried.empty()) {
                ++stats.non_empty_delta;
            }

            // (a) after the pass pipeline, with the Δ the pipeline carried.
            const auto after_pipeline = exact_hull(compiled.output, output_view, carried, modes[mode_index],
                                                   compiled.output.targets()[output_target.raw()].node.raw());
            INFO("after pipeline: expected " << expected << " actual " << after_pipeline);
            REQUIRE(std::abs(after_pipeline - expected) <= kTolerance);
            ++stats.comparisons[mode_index];

            // (b) after writer → reader, from the file's own mode and Δ. The reader numbers nodes canonically
            // (basic events first, by ascending event id), so Δ is compared as a set of events through the
            // tensors, which is what any consumer of the file does.
            const auto &quantification = read.value().options.quantification.at(output_target);
            std::vector<std::uint32_t> file_delta;
            for (const auto node : quantification.literal_events) {
                file_delta.push_back(node.raw());
            }
            std::vector<std::uint32_t> expected_file_delta;
            for (const auto node : carried) {
                const auto tensor = written.value().tensor_by_node.at(node);
                REQUIRE(tensor < read.value().node_by_tensor.size());
                REQUIRE(read.value().node_by_tensor[tensor].valid());
                expected_file_delta.push_back(read.value().node_by_tensor[tensor].raw());
            }
            std::ranges::sort(expected_file_delta);
            expected_file_delta.erase(std::ranges::unique(expected_file_delta).begin(), expected_file_delta.end());
            // The file's mode agrees with the requested one; the read Δ is the carried Δ.
            const auto expected_mode = modes[mode_index];
            const auto read_mode = quantification.literal_mode == "atom_split" ? pcfb::LiteralMode::ATOM_SPLIT
                                    : quantification.literal_mode == "omega_l" ? pcfb::LiteralMode::OMEGA_L
                                                                               : pcfb::LiteralMode::NEGATION;
            REQUIRE(read_mode == expected_mode);
            REQUIRE(file_delta == expected_file_delta);
            const auto after_round_trip = exact_hull(read.value().circuit, read_view, file_delta, read_mode,
                                                     read.value().circuit.targets()[output_target.raw()].node.raw());
            INFO("after round trip: expected " << expected << " actual " << after_round_trip);
            REQUIRE(std::abs(after_round_trip - expected) <= kTolerance);
        }
    }
    // The suite must actually exercise what it claims.
    for (const auto count : stats.comparisons) {
        REQUIRE(count > 0);
    }
    REQUIRE(stats.non_empty_delta > 0);
    WARN("circuits " << stats.circuits << ", hull comparisons atom_split " << stats.comparisons[0] << ", omega_l "
                     << stats.comparisons[1] << ", negation " << stats.comparisons[2] << ", with non-empty Δ ∩ cone "
                     << stats.non_empty_delta << ", rewrites " << stats.rewrites << ", atom-split skips " << stats.skipped);
}

namespace {

    // A two-event hull target with one marked literal: `A AND -B`, B marked.
    [[nodiscard]] auto literal_model() -> pc::circuit {
        pc::builder builder;
        const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
        const auto b = builder.add_basic_event(builder.add_literal_probability(0.2));
        const auto top = builder.add_and({pc::edge_ref{.source = a}, pc::edge_ref{.source = b, .complement = true}});
        (void)builder.add_target(top);
        return pcfb_writer_test::sealed(std::move(builder));
    }

    [[nodiscard]] auto literal_options(const char *mode, std::vector<pc::bool_node_id> events) -> writer::write_options {
        writer::write_options options;
        options.target_names = {{pc::target_id(0), "TOP"}};
        options.target_kinds = {{pc::target_id(0), {writer::query_kind::monotone_hull_probability}}};
        options.quantification = {
            {pc::target_id(0), writer::target_quantification{.literal_mode = mode, .literal_events = std::move(events)}},
        };
        return options;
    }

} // namespace

TEST_CASE("literal mode and samples round-trip through write → read for every mode", "[io][pcfb][literals][roundtrip]") {
    const auto model = literal_model();
    constexpr std::array modes{pcfb::LiteralMode::ATOM_SPLIT, pcfb::LiteralMode::OMEGA_L, pcfb::LiteralMode::NEGATION};
    constexpr std::array names{"atom_split", "omega_l", "negation"};
    const auto b = pc::bool_node_id(1);
    for (std::size_t index = 0; index < modes.size(); ++index) {
        SECTION(names[index]) {
            for (const auto with_list : {false, true}) {
                // The empty Δ case writes the mode alone (a mode with no samples is legal on a hull query).
                const auto events = with_list ? std::vector<pc::bool_node_id>{b} : std::vector<pc::bool_node_id>{};
                auto options = literal_options(names[index], events);
                if (!with_list && modes[index] == pcfb::LiteralMode::ATOM_SPLIT) {
                    // A non-default mode is needed for the field to be written at all with an empty list.
                    continue;
                }
                const auto written = pcfb_writer_test::requireWrite(model, options);
                const auto read = read_file_ok(written.bytes);
                const auto *query = [&] {
                    auto file = pcfb::io::from_bytes(written.bytes);
                    REQUIRE(file);
                    REQUIRE(file->subcircuits()->Get(0)->queries()->size() == 1);
                    return file->subcircuits()->Get(0)->queries()->Get(0);
                }();
                const auto literals = pcfb::read_literals(*query);
                CHECK(literals.mode == modes[index]);
                CHECK(literals.samples.size() == events.size());
                // The reader's own view carries the same mode and a Δ of the same size.
                const auto &quantification = read.options.quantification.at(pc::target_id(0));
                CHECK(quantification.literal_mode == names[index]);
                CHECK(quantification.literal_events.size() == events.size());
            }
        }
    }
}

TEST_CASE("a file without hull literals is byte-identical to one written before the amendment",
          "[io][pcfb][literals][bytes]") {
    // A PROBABILITY-only target records no literal fields on the wire, so the bytes are those of a v1.1 file.
    const auto model = pcfb_writer_test::threeTargetModel();
    auto options = pcfb_writer_test::threeTargetNames();
    options.target_kinds = {{pc::target_id(0), {writer::query_kind::probability}},
                            {pc::target_id(1), {writer::query_kind::probability}},
                            {pc::target_id(2), {writer::query_kind::probability}}};
    const auto plain = pcfb_writer_test::requireWrite(model, options);
    CHECK(read_file_ok(plain.bytes).schema_version == 0x00010001U);

    // Adding quantification records with an empty literal mode and no events changes no query bytes.
    auto recorded = options;
    recorded.quantification = {{pc::target_id(0), writer::target_quantification{}}};
    const auto with_record = pcfb_writer_test::requireWrite(model, recorded);
    auto a = pcfb::io::from_bytes(plain.bytes);
    auto c = pcfb::io::from_bytes(with_record.bytes);
    REQUIRE(a);
    REQUIRE(c);
    const auto *qa = a->subcircuits()->Get(0)->queries();
    const auto *qc = c->subcircuits()->Get(0)->queries();
    REQUIRE(qa->size() == qc->size());
    for (flatbuffers::uoffset_t index = 0; index < qa->size(); ++index) {
        CHECK(qa->Get(index)->literal_mode() == pcfb::LiteralMode::ATOM_SPLIT);
        CHECK(qa->Get(index)->literal_samples() == nullptr);
        CHECK(qc->Get(index)->literal_mode() == pcfb::LiteralMode::ATOM_SPLIT);
        CHECK(qc->Get(index)->literal_samples() == nullptr);
    }
}

TEST_CASE("the reader rejects the literal fields the validator rejects", "[io][pcfb][literals][negative]") {
    const auto model = literal_model();
    const auto written = pcfb_writer_test::requireWrite(model, literal_options("atom_split", {pc::bool_node_id(1)}));

    // Rewrite the query's literal fields into the shapes the validator rejects, and check the reader's own
    // diagnostic id for each. `writer::read` returns the diagnostics only on failure, so the check is on the
    // failed result before any assertion consumes it.
    const auto diagnose = [&](auto &&change) {
        auto file = unpack_file(written.bytes);
        change(*file->subcircuits[0]->queries[0]);
        const auto read = writer::read(std::span<std::byte const>{pack_file(*file)});
        REQUIRE_FALSE(read);
        const auto text = pcfb_writer_test::diagnosticsText(read.diagnostics());
        INFO(text);
        return text;
    };

    SECTION("an index that is not a SAMPLE output") {
        // Tensor 2 is the AND gate, not a SAMPLE.
        const auto text = diagnose([](pcfb::QueryT &query) { query.literal_samples = {2}; });
        CHECK(text.find(std::string(writer::diag::invalid_literal_fields)) != std::string::npos);
    }
    SECTION("a descending list") {
        const auto text = diagnose([](pcfb::QueryT &query) { query.literal_samples = {2, 1}; });
        CHECK(text.find(std::string(writer::diag::invalid_literal_fields)) != std::string::npos);
    }
    SECTION("literal fields on a PROBABILITY query") {
        const auto text = diagnose([](pcfb::QueryT &query) {
            query.kind = pcfb::QueryKind::PROBABILITY;
            query.literal_samples = {1};
        });
        CHECK(text.find(std::string(writer::diag::invalid_literal_fields)) != std::string::npos);
    }
    SECTION("an unknown literal mode") {
        const auto text = diagnose([](pcfb::QueryT &query) { query.literal_mode = static_cast<pcfb::LiteralMode>(7); });
        CHECK(text.find(std::string(writer::diag::unknown_literal_mode)) != std::string::npos);
    }
}

TEST_CASE("the writer rejects a literal event that is not a basic event", "[io][pcfb][literals][negative]") {
    const auto model = literal_model();
    // Node 2 is the AND gate, not a basic event.
    auto options = literal_options("atom_split", {pc::bool_node_id(2)});
    const auto written = writer::write(model.to_handle(), options);
    REQUIRE_FALSE(written);
    CHECK(pcfb_writer_test::hasCode(written.diagnostics(), std::string(writer::diag::invalid_literal_sample)));
}
