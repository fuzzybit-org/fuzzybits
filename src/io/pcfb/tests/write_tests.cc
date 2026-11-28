#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include "api/diagnostics.h"
#include "api/options.h"
#include "compile/exact_eval.h"
#include "compile/ftp/tests/support.h"
#include "io/descriptor.h"
#include "io/formats.h"
#include "io/pcfb/descriptor.h"
#include "io/pcfb/detail/model.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/target.h"

#include "pcfb/io.h"
#include "pcfb/threshold.h"
#include "pcfb/validate.h"
#include "pcfb_generated.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace pcfb_writer_test;
namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;

using edge = pc::edge_ref;

[[nodiscard]] auto view(writer::write_output const &written) -> ::pcfb::ir::Circuit {
    auto circuit = ::pcfb::io::from_bytes(written.bytes);
    INFO((circuit ? std::string{} : circuit.error().message));
    REQUIRE(circuit);
    return *circuit;
}

[[nodiscard]] auto subcircuit(::pcfb::ir::Circuit const &circuit) -> ::pcfb::SubCircuit const * {
    REQUIRE(circuit.subcircuits() != nullptr);
    REQUIRE(circuit.subcircuits()->size() == 1);
    return circuit.subcircuits()->Get(0);
}

[[nodiscard]] auto opAt(::pcfb::SubCircuit const *sub, const std::uint32_t tensor) -> ::pcfb::Op const * {
    REQUIRE(tensor < sub->operators()->size());
    auto const *op = sub->operators()->Get(tensor);
    REQUIRE(op->output() == tensor);
    return op;
}

[[nodiscard]] auto indices(flatbuffers::Vector<std::uint32_t> const *values) -> std::vector<std::uint32_t> {
    return values == nullptr ? std::vector<std::uint32_t>{} : std::vector<std::uint32_t>(values->begin(), values->end());
}

[[nodiscard]] auto kOf(::pcfb::Op const *op) -> std::uint32_t {
    REQUIRE(op->args_type() == ::pcfb::OpArgs::KofNArgs);
    return op->args_as_KofNArgs()->k();
}

[[nodiscard]] auto stringMeta(::pcfb::ir::Circuit const &circuit, char const *key) -> std::optional<std::string> {
    auto const *entry = circuit->metadata() == nullptr ? nullptr : circuit->metadata()->LookupByKey(key);
    if (entry == nullptr || entry->value_type() != ::pcfb::MetadataValue::StringValue) {
        return std::nullopt;
    }
    return entry->value_as_StringValue()->value()->str();
}

void requireValid(::pcfb::ir::Circuit const &circuit) {
    std::ostringstream text;
    const auto diagnostics = ::pcfb::validate(circuit);
    for (auto const &diagnostic : diagnostics) {
        text << diagnostic.location << ": " << diagnostic.message << '\n';
    }
    INFO(text.str());
    REQUIRE(diagnostics.empty());
}

// Structural invariants of every written file, independent of the circuit.
void requireWriterInvariants(pc::circuit const &model, writer::write_output const &written, ::pcfb::ir::Circuit const &circuit) {
    requireValid(circuit);
    auto const *sub = subcircuit(circuit);
    REQUIRE(written.tensor_by_node.size() == model.node_count());
    REQUIRE(written.event_id_by_node.size() == model.node_count());

    for (auto const *op : *sub->operators()) {
        const auto inputs = indices(op->inputs());
        const auto inverted = indices(op->inverted_inputs());
        for (const auto tensor : inverted) {
            REQUIRE(std::ranges::find(inputs, tensor) == inputs.end());
        }
    }

    std::uint32_t expectedEventId = 0;
    for (std::uint32_t node = 0; node < model.node_count(); ++node) {
        auto const *op = opAt(sub, written.tensor_by_node[node]);
        if (model.op_codes()[node] != pc::op_code::basic_event) {
            REQUIRE(written.event_id_by_node[node] == 0);
            REQUIRE(op->opcode() != ::pcfb::OpCode::SAMPLE);
            continue;
        }
        REQUIRE(written.event_id_by_node[node] == ++expectedEventId);
        REQUIRE(op->opcode() == ::pcfb::OpCode::SAMPLE);
        REQUIRE(op->args_as_SampleArgs()->event_id() == expectedEventId);
        REQUIRE(op->args_as_SampleArgs()->source() == ::pcfb::SampleSource::BASIC_EVENT);
        REQUIRE(op->args_as_SampleArgs()->probability() == model.probability_sources()[model.probability_source_refs()[node].raw()].literal);
    }
    std::size_t samples = 0;
    for (auto const *op : *sub->operators()) {
        samples += op->opcode() == ::pcfb::OpCode::SAMPLE ? 1 : 0;
    }
    REQUIRE(samples == expectedEventId);

    REQUIRE(sub->queries()->size() == model.targets().size());
    for (std::size_t index = 0; index < model.targets().size(); ++index) {
        auto const *query = sub->queries()->Get(static_cast<flatbuffers::uoffset_t>(index));
        REQUIRE(query->target() == written.tensor_by_node[model.targets()[index].node.raw()]);
        REQUIRE(query->kind() == ::pcfb::QueryKind::PROBABILITY);
    }
}

void requireExact(pc::circuit const &model, ::pcfb::ir::Circuit const &circuit) {
    const auto expected = referenceProbabilities(model);
    for (std::uint32_t query = 0; query < expected.size(); ++query) {
        const auto exact = fuzzybits::compile::exact_probability(circuit, query);
        INFO("query " << query << " exact_probability " << exact << " reference " << expected[query]);
        REQUIRE(std::abs(exact - expected[query]) <= 1.0e-12);
    }
}

} // namespace

TEST_CASE("pcfb writer descriptor advertises the pcfb format and its options", "[io][pcfb]") {
    const auto descriptor = writer::writer_descriptor();
    REQUIRE(descriptor.provider == fuzzybits::api::provider_id("fuzzybits"));
    REQUIRE(descriptor.endpoint == writer::writer_endpoint_id());
    REQUIRE(descriptor.output_format == fuzzybits::io::formats::pcfb());

    std::set<std::string> keys;
    for (auto const &entry : descriptor.options.entries) {
        keys.insert(entry.key.str());
    }
    REQUIRE(keys ==
            std::set<std::string>{"io.pcfb.compile_options", "io.pcfb.model_name", "io.pcfb.query_kind", "io.pcfb.source_format", "io.pcfb.source_sha256"});
}

TEST_CASE("pcfb writer maps every PC opcode", "[io][pcfb]") {
    pc::builder builder;
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.25));
    const auto b = builder.add_basic_event(builder.add_literal_probability(1.0));
    const auto c = builder.add_basic_event(builder.add_literal_probability(0.0));
    const auto constantTrue = builder.add_constant(true);
    const auto constantFalse = builder.add_constant(false);
    const auto houseTrue = builder.add_house_event(true);
    const auto houseFalse = builder.add_house_event(false);
    const auto andGate = builder.add_and({edge{.source = a}, edge{.source = c, .complement = true}});
    const auto orGate = builder.add_or({edge{.source = a, .complement = true}, edge{.source = b}});
    const auto xorGate = builder.add_gate(pc::op_code::xor_gate, {edge{.source = a}, edge{.source = b}, edge{.source = c, .complement = true}});
    const auto notGate = builder.add_not(edge{.source = a});
    const auto notComplement = builder.add_not(edge{.source = b, .complement = true});
    const auto nandGate = builder.add_gate(pc::op_code::nand_gate, {edge{.source = a}, edge{.source = b}});
    const auto norGate = builder.add_gate(pc::op_code::nor_gate, {edge{.source = a, .complement = true}, edge{.source = c}});
    const auto xnorGate = builder.add_gate(pc::op_code::xnor_gate, {edge{.source = a}, edge{.source = b, .complement = true}});
    const auto atLeast = builder.add_cardinality(2, 3, {edge{.source = a}, edge{.source = b, .complement = true}, edge{.source = c}});
    const auto top = builder.add_or({
        edge{.source = constantTrue, .complement = true},
        edge{.source = constantFalse},
        edge{.source = houseTrue, .complement = true},
        edge{.source = houseFalse},
        edge{.source = andGate},
        edge{.source = orGate, .complement = true},
        edge{.source = xorGate},
        edge{.source = notGate, .complement = true},
        edge{.source = notComplement, .complement = true},
        edge{.source = nandGate, .complement = true},
        edge{.source = norGate},
        edge{.source = xnorGate},
        edge{.source = atLeast},
    });
    (void)builder.add_target(top);
    (void)builder.add_target(atLeast, pc::observable_role::watch);
    (void)builder.add_target(a, pc::observable_role::projection_anchor);
    const auto model = sealed(std::move(builder));

    auto written = fuzzybits::io::backend::fuzzybits::pcfb::write(model.to_handle(), {});
    INFO(diagnosticsText(written.diagnostics()));
    REQUIRE(written);
    const auto output = std::move(written).value();
    const auto circuit = view(output);
    requireWriterInvariants(model, output, circuit);
    auto const *sub = subcircuit(circuit);

    // Topologically numbered and no auxiliary ops: tensor index == node id.
    REQUIRE(sub->operators()->size() == model.node_count());
    for (std::uint32_t node = 0; node < model.node_count(); ++node) {
        REQUIRE(output.tensor_by_node[node] == node);
    }

    auto const *sampleA = opAt(sub, a.raw())->args_as_SampleArgs();
    REQUIRE(sampleA->probability() == 0.25);
    REQUIRE(sampleA->threshold() == ::pcfb::probability_to_threshold(0.25));
    REQUIRE_FALSE(sampleA->always_true());
    REQUIRE(sampleA->name()->str() == "e1");
    REQUIRE(opAt(sub, b.raw())->args_as_SampleArgs()->always_true());
    REQUIRE(opAt(sub, c.raw())->args_as_SampleArgs()->threshold() == 0);
    REQUIRE(opAt(sub, c.raw())->args_as_SampleArgs()->event_id() == 3);

    REQUIRE(opAt(sub, constantTrue.raw())->opcode() == ::pcfb::OpCode::BITWISE_ONES);
    REQUIRE(opAt(sub, constantFalse.raw())->opcode() == ::pcfb::OpCode::BITWISE_ZEROS);
    REQUIRE(opAt(sub, houseTrue.raw())->opcode() == ::pcfb::OpCode::BITWISE_ONES);
    REQUIRE(opAt(sub, houseFalse.raw())->opcode() == ::pcfb::OpCode::BITWISE_ZEROS);

    const auto requireGate = [&](const pc::bool_node_id node, const ::pcfb::OpCode opcode, std::vector<std::uint32_t> const &inputs,
                                 std::vector<std::uint32_t> const &inverted) {
        auto const *op = opAt(sub, node.raw());
        REQUIRE(op->opcode() == opcode);
        REQUIRE(indices(op->inputs()) == inputs);
        REQUIRE(indices(op->inverted_inputs()) == inverted);
    };
    requireGate(andGate, ::pcfb::OpCode::BITWISE_AND, {a.raw()}, {c.raw()});
    requireGate(orGate, ::pcfb::OpCode::BITWISE_OR, {b.raw()}, {a.raw()});
    requireGate(xorGate, ::pcfb::OpCode::BITWISE_XOR, {a.raw(), b.raw()}, {c.raw()});
    requireGate(notGate, ::pcfb::OpCode::BITWISE_NOT, {a.raw()}, {});
    requireGate(notComplement, ::pcfb::OpCode::BITWISE_NOT, {}, {b.raw()});
    requireGate(nandGate, ::pcfb::OpCode::BITWISE_NAND, {a.raw(), b.raw()}, {});
    requireGate(norGate, ::pcfb::OpCode::BITWISE_NOR, {c.raw()}, {a.raw()});
    requireGate(xnorGate, ::pcfb::OpCode::BITWISE_XNOR, {a.raw()}, {b.raw()});
    requireGate(atLeast, ::pcfb::OpCode::BITWISE_ATLEAST, {a.raw(), c.raw()}, {b.raw()});
    REQUIRE(kOf(opAt(sub, atLeast.raw())) == 2);

    REQUIRE(sub->queries()->Get(0)->name()->str() == "t0");
    REQUIRE(sub->queries()->Get(1)->name()->str() == "t1");
    REQUIRE(sub->queries()->Get(2)->target() == a.raw());
    requireExact(model, circuit);
}

TEST_CASE("pcfb writer lowers cardinality(low, high) by range", "[io][pcfb]") {
    pc::builder builder;
    std::array<pc::bool_node_id, 4> events{};
    const std::array probabilities{0.1, 0.2, 0.3, 0.4};
    for (std::size_t index = 0; index < events.size(); ++index) {
        events[index] = builder.add_basic_event(builder.add_literal_probability(probabilities[index]));
    }
    const std::array operands{
        edge{.source = events[0]},
        edge{.source = events[1], .complement = true},
        edge{.source = events[2]},
        edge{.source = events[3]},
    };
    struct Range final {
        std::uint32_t low;
        std::uint32_t high;
        pc::bool_node_id node;
    };
    std::vector<Range> ranges{{0, 4, {}}, {2, 4, {}}, {4, 4, {}}, {0, 2, {}}, {0, 0, {}}, {2, 2, {}}, {1, 3, {}}};
    for (auto &range : ranges) {
        range.node = builder.add_cardinality(range.low, range.high, std::span<edge const>{operands});
        (void)builder.add_target(range.node);
    }
    const auto model = sealed(std::move(builder));

    writer::write_options options;
    options.node_names[ranges.back().node] = "RANGE";
    const auto output = requireWrite(model, options);
    const auto circuit = view(output);
    requireWriterInvariants(model, output, circuit);
    auto const *sub = subcircuit(circuit);

    const std::vector<std::uint32_t> inputs{events[0].raw(), events[2].raw(), events[3].raw()};
    const std::vector<std::uint32_t> inverted{events[1].raw()};
    const auto requireKofN = [&](::pcfb::Op const *op, const ::pcfb::OpCode opcode, const std::uint32_t k) {
        REQUIRE(op->opcode() == opcode);
        REQUIRE(kOf(op) == k);
        REQUIRE(indices(op->inputs()) == inputs);
        REQUIRE(indices(op->inverted_inputs()) == inverted);
    };

    requireKofN(opAt(sub, output.tensor_by_node[ranges[0].node.raw()]), ::pcfb::OpCode::BITWISE_ATLEAST, 0);
    requireKofN(opAt(sub, output.tensor_by_node[ranges[1].node.raw()]), ::pcfb::OpCode::BITWISE_ATLEAST, 2);
    requireKofN(opAt(sub, output.tensor_by_node[ranges[2].node.raw()]), ::pcfb::OpCode::BITWISE_ATLEAST, 4);
    requireKofN(opAt(sub, output.tensor_by_node[ranges[3].node.raw()]), ::pcfb::OpCode::BITWISE_ATMOST, 2);
    requireKofN(opAt(sub, output.tensor_by_node[ranges[4].node.raw()]), ::pcfb::OpCode::BITWISE_ATMOST, 0);
    requireKofN(opAt(sub, output.tensor_by_node[ranges[5].node.raw()]), ::pcfb::OpCode::BITWISE_EXACTLY, 2);

    const auto rangeTensor = output.tensor_by_node[ranges[6].node.raw()];
    auto const *both = opAt(sub, rangeTensor);
    REQUIRE(both->opcode() == ::pcfb::OpCode::BITWISE_AND);
    REQUIRE(both->name()->str() == "RANGE");
    REQUIRE(indices(both->inputs()) == std::vector<std::uint32_t>{rangeTensor - 2, rangeTensor - 1});
    requireKofN(opAt(sub, rangeTensor - 2), ::pcfb::OpCode::BITWISE_ATLEAST, 1);
    requireKofN(opAt(sub, rangeTensor - 1), ::pcfb::OpCode::BITWISE_ATMOST, 3);
    REQUIRE(opAt(sub, rangeTensor - 2)->name() == nullptr);
    REQUIRE(sub->operators()->size() == model.node_count() + 2);
    requireExact(model, circuit);
}

TEST_CASE("pcfb writer reads mixed-polarity operands through one NOT per source", "[io][pcfb]") {
    pc::builder builder;
    const auto x = builder.add_basic_event(builder.add_literal_probability(0.3));
    const auto y = builder.add_basic_event(builder.add_literal_probability(0.6));
    const auto andGate = builder.add_and({edge{.source = x}, edge{.source = x, .complement = true}, edge{.source = y}});
    const auto orGate = builder.add_or({edge{.source = x, .complement = true}, edge{.source = x}});
    const auto xorGate =
        builder.add_gate(pc::op_code::xor_gate, {edge{.source = x}, edge{.source = x, .complement = true}, edge{.source = y, .complement = true}});
    const auto range = builder.add_cardinality(1, 2, {edge{.source = x}, edge{.source = x, .complement = true}, edge{.source = y}});
    const auto duplicate = builder.add_cardinality(2, 3, {edge{.source = y}, edge{.source = y}, edge{.source = x}});
    for (const auto node : {andGate, orGate, xorGate, range, duplicate}) {
        (void)builder.add_target(node);
    }
    const auto model = sealed(std::move(builder));
    const auto output = requireWrite(model);
    const auto circuit = view(output);
    requireWriterInvariants(model, output, circuit);
    auto const *sub = subcircuit(circuit);

    std::vector<std::uint32_t> nots;
    for (auto const *op : *sub->operators()) {
        if (op->opcode() == ::pcfb::OpCode::BITWISE_NOT) {
            nots.push_back(op->output());
            REQUIRE(indices(op->inputs()) == std::vector<std::uint32_t>{output.tensor_by_node[x.raw()]});
        }
    }
    REQUIRE(nots.size() == 1);
    auto const *andOp = opAt(sub, output.tensor_by_node[andGate.raw()]);
    REQUIRE(indices(andOp->inputs()) == std::vector<std::uint32_t>{output.tensor_by_node[x.raw()], nots[0], output.tensor_by_node[y.raw()]});
    REQUIRE(indices(andOp->inverted_inputs()).empty());
    auto const *duplicateOp = opAt(sub, output.tensor_by_node[duplicate.raw()]);
    REQUIRE(indices(duplicateOp->inputs()).size() == 3);
    requireExact(model, circuit);
}

TEST_CASE("pcfb writer assigns event ids by basic-event node id, not emission order", "[io][pcfb]") {
    pc::builder builder;
    const auto top = builder.add_and({edge{.source = pc::bool_node_id(1)}, edge{.source = pc::bool_node_id(3)}});
    const auto first = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto other = builder.add_or({edge{.source = pc::bool_node_id(4)}, edge{.source = pc::bool_node_id(3), .complement = true}});
    const auto second = builder.add_basic_event(builder.add_literal_probability(0.3));
    const auto third = builder.add_basic_event(builder.add_literal_probability(0.4));
    (void)builder.add_target(top);
    (void)builder.add_target(other);
    const auto model = sealed(std::move(builder));

    auto const order = writer::detail::emission_order(writer::detail::graph_view::of(model));
    REQUIRE(order);
    REQUIRE(order.value() == std::vector<std::uint32_t>{first.raw(), second.raw(), top.raw(), third.raw(), other.raw()});

    const auto output = requireWrite(model);
    const auto circuit = view(output);
    requireWriterInvariants(model, output, circuit);
    REQUIRE(output.tensor_by_node == std::vector<std::uint32_t>{2, 0, 4, 1, 3});
    REQUIRE(output.event_id_by_node == std::vector<std::uint32_t>{0, 1, 0, 2, 3});
    REQUIRE(opAt(subcircuit(circuit), 3)->args_as_SampleArgs()->event_id() == 3);
    requireExact(model, circuit);
}

TEST_CASE("pcfb writer keeps basic events that share a probability source independent", "[io][pcfb]") {
    pc::builder builder;
    const auto source = builder.add_literal_probability(0.5);
    const auto left = builder.add_basic_event(source);
    const auto right = builder.add_basic_event(source);
    (void)builder.add_target(builder.add_and({edge{.source = left}, edge{.source = right}}));
    const auto model = sealed(std::move(builder));
    const auto output = requireWrite(model);
    const auto circuit = view(output);
    requireWriterInvariants(model, output, circuit);
    REQUIRE(fuzzybits::compile::exact_probability(circuit) == 0.25);
}

TEST_CASE("pcfb writer takes names and metadata from options", "[io][pcfb]") {
    pc::builder builder;
    const auto pump = builder.add_basic_event(builder.add_literal_probability(0.1));
    const auto valve = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto flag = builder.add_house_event(true);
    const auto gate = builder.add_and({edge{.source = pump}, edge{.source = valve}, edge{.source = flag}});
    const auto other = builder.add_or({edge{.source = pump}, edge{.source = valve}});
    const auto named = builder.add_target(gate);
    (void)builder.add_target(other);
    const auto model = sealed(std::move(builder));

    writer::write_options options;
    options.node_names[pump] = "PUMP-A";
    options.node_names[flag] = "FLAG";
    options.node_names[gate] = "G1";
    options.target_names[named] = "TOP";
    options.model_name = "demo";
    options.source_format = "cafta-ftp";
    options.compile_options = "--no-passes";
    std::array<std::byte, 32> digest{};
    for (std::size_t index = 0; index < digest.size(); ++index) {
        digest[index] = static_cast<std::byte>(index * 7 + 1);
    }
    options.source_sha256 = digest;
    const auto output = requireWrite(model, options);
    const auto circuit = view(output);
    requireWriterInvariants(model, output, circuit);
    auto const *sub = subcircuit(circuit);

    REQUIRE(opAt(sub, pump.raw())->args_as_SampleArgs()->name()->str() == "PUMP-A");
    REQUIRE(opAt(sub, pump.raw())->name()->str() == "PUMP-A");
    REQUIRE(opAt(sub, valve.raw())->args_as_SampleArgs()->name()->str() == "e2");
    REQUIRE(opAt(sub, flag.raw())->name()->str() == "FLAG");
    REQUIRE(opAt(sub, gate.raw())->name()->str() == "G1");
    REQUIRE(opAt(sub, other.raw())->name() == nullptr);
    REQUIRE(sub->queries()->Get(0)->name()->str() == "TOP");
    REQUIRE(sub->queries()->Get(1)->name()->str() == "t1");

    REQUIRE(circuit->name()->str() == "demo");
    REQUIRE(stringMeta(circuit, "general.generator") == "fuzzybits");
    REQUIRE_FALSE(stringMeta(circuit, "general.generator_version").value_or("").empty());
    REQUIRE(stringMeta(circuit, "general.endianness") == "little");
    REQUIRE(stringMeta(circuit, "pcfb.opset") == "core-1.1");
    REQUIRE(stringMeta(circuit, "general.model_name") == "demo");
    REQUIRE(stringMeta(circuit, "general.source_format") == "cafta-ftp");
    REQUIRE(stringMeta(circuit, "compile.options") == "--no-passes");
    auto const *sha = circuit->metadata()->LookupByKey("general.source_sha256");
    REQUIRE(sha != nullptr);
    REQUIRE(sha->value_type() == ::pcfb::MetadataValue::BytesValue);
    auto const *bytes = sha->value_as_BytesValue()->value();
    REQUIRE(bytes->size() == digest.size());
    for (std::size_t index = 0; index < digest.size(); ++index) {
        REQUIRE(bytes->Get(static_cast<flatbuffers::uoffset_t>(index)) == static_cast<std::uint8_t>(digest[index]));
    }

    const auto bare = view(requireWrite(model));
    REQUIRE_FALSE(stringMeta(bare, "general.model_name"));
    REQUIRE_FALSE(stringMeta(bare, "general.source_format"));
    REQUIRE_FALSE(stringMeta(bare, "compile.options"));
    REQUIRE(bare->metadata()->LookupByKey("general.source_sha256") == nullptr);
}

TEST_CASE("pcfb writer output is deterministic", "[io][pcfb]") {
    std::mt19937_64 rng(176);
    Coverage coverage;
    for (int round = 0; round < 20; ++round) {
        auto sealed = randomCircuit(rng, coverage);
        REQUIRE(sealed.ok());
        REQUIRE(requireWrite(sealed.output).bytes == requireWrite(sealed.output).bytes);
    }
}

TEST_CASE("pcfb writer rejects parameter and unsupported probabilities", "[io][pcfb]") {
    pc::builder builder;
    const auto parameter = builder.add_basic_event(builder.add_parameter_probability("lambda"));
    const auto unsupported = builder.add_basic_event(builder.add_unsupported_probability("lognormal deviate"));
    const auto literal = builder.add_basic_event(builder.add_literal_probability(0.5));
    (void)builder.add_target(builder.add_or({edge{.source = parameter}, edge{.source = unsupported}, edge{.source = literal}}));
    const auto model = sealed(std::move(builder));

    writer::write_options options;
    options.node_names[parameter] = "PARAM";
    const auto written = writer::write(model.to_handle(), options);
    REQUIRE_FALSE(written);
    REQUIRE_FALSE(written.has_value());
    REQUIRE(hasCode(written.diagnostics(), "io.pcfb.parameter-probability"));
    REQUIRE(hasCode(written.diagnostics(), "io.pcfb.unsupported-probability"));
    const auto text = diagnosticsText(written.diagnostics());
    REQUIRE(text.find("'PARAM'") != std::string::npos);
    REQUIRE(text.find("lambda") != std::string::npos);
    REQUIRE(text.find("lognormal deviate") != std::string::npos);
}

TEST_CASE("pcfb writer rejects circuits without targets and names for unknown ids", "[io][pcfb]") {
    pc::builder untargeted;
    (void)untargeted.add_basic_event(untargeted.add_literal_probability(0.5));
    const auto noTargets = writer::write(sealed(std::move(untargeted)).to_handle(), {});
    REQUIRE_FALSE(noTargets);
    REQUIRE(hasCode(noTargets.diagnostics(), "io.pcfb.no-targets"));

    pc::builder builder;
    const auto event = builder.add_basic_event(builder.add_literal_probability(0.5));
    (void)builder.add_target(event);
    const auto model = sealed(std::move(builder));

    writer::write_options badNode;
    badNode.node_names[pc::bool_node_id(7)] = "GHOST";
    REQUIRE(hasCode(writer::write(model.to_handle(), badNode).diagnostics(), "io.pcfb.invalid-name"));

    writer::write_options badTarget;
    badTarget.target_names[pc::target_id(3)] = "GHOST";
    REQUIRE(hasCode(writer::write(model.to_handle(), badTarget).diagnostics(), "io.pcfb.invalid-name"));

    REQUIRE(hasCode(writer::write(fuzzybits::ir::handle{}, writer::write_options{}).diagnostics(), "io.pcfb.unsupported-ir"));
}

TEST_CASE("pcfb stream writer reads scalar options from the request", "[io][pcfb]") {
    pc::builder builder;
    const auto event = builder.add_basic_event(builder.add_literal_probability(0.5));
    (void)builder.add_target(event);
    const auto model = sealed(std::move(builder));

    fuzzybits::io::write_request request{.format = fuzzybits::io::formats::pcfb()};
    request.options.set(writer::model_name_option(), std::string("streamed"));
    request.options.set(writer::source_format_option(), std::string("cafta-ftp"));
    request.options.set(writer::compile_options_option(), std::string("-x"));
    request.options.set(writer::source_sha256_option(), std::string(64, 'A'));
    request.options.set(writer::query_kind_option(), std::string("probability"));
    std::ostringstream stream;
    const auto written = writer::write(model.to_handle(), stream, request);
    INFO(diagnosticsText(written.diagnostics()));
    REQUIRE(written);

    const auto text = stream.str();
    std::vector<std::byte> bytes(text.size());
    std::ranges::transform(text, bytes.begin(), [](const char c) { return static_cast<std::byte>(c); });
    const auto circuit = ::pcfb::io::from_bytes(bytes);
    REQUIRE(circuit);
    requireValid(*circuit);
    REQUIRE(stringMeta(*circuit, "general.model_name") == "streamed");
    REQUIRE(stringMeta(*circuit, "compile.options") == "-x");
    auto const *sha = (*circuit)->metadata()->LookupByKey("general.source_sha256")->value_as_BytesValue()->value();
    REQUIRE(sha->size() == 32);
    REQUIRE(sha->Get(0) == 0xAA);

    fuzzybits::io::write_request badDigest{.format = fuzzybits::io::formats::pcfb()};
    badDigest.options.set(writer::source_sha256_option(), std::string("abc"));
    std::ostringstream ignored;
    REQUIRE(hasCode(writer::write(model.to_handle(), ignored, badDigest).diagnostics(), "io.pcfb.invalid-option"));

    fuzzybits::io::write_request badKind{.format = fuzzybits::io::formats::pcfb()};
    badKind.options.set(writer::query_kind_option(), std::string("monotone_hull"));
    REQUIRE(hasCode(writer::write(model.to_handle(), ignored, badKind).diagnostics(), "io.pcfb.invalid-option"));

    const auto mismatch = writer::write(model.to_handle(), ignored, fuzzybits::io::write_request{.format = fuzzybits::io::formats::ftp()});
    REQUIRE(hasCode(mismatch.diagnostics(), "io.pcfb.format-mismatch"));
    REQUIRE(ignored.str().empty());

    std::ostringstream defaults;
    REQUIRE(writer::write(model.to_handle(), defaults));
    REQUIRE_FALSE(defaults.str().empty());
}

TEST_CASE("pcfb emission order diagnoses cycles and invalid endpoints", "[io][pcfb]") {
    const std::vector<pc::op_code> codes(3, pc::op_code::and_gate);
    const std::vector<std::uint32_t> offsets{0, 1, 2};
    const std::vector<std::uint32_t> counts{1, 1, 1};
    const auto order = [&](std::vector<edge> const &edges) {
        return writer::detail::emission_order(writer::detail::graph_view{
            .op_codes = codes,
            .operand_offsets = offsets,
            .operand_counts = counts,
            .edges = edges,
        });
    };

    const auto cycle = order({edge{.source = pc::bool_node_id(1)}, edge{.source = pc::bool_node_id(2)}, edge{.source = pc::bool_node_id(0)}});
    REQUIRE_FALSE(cycle);
    REQUIRE(hasCode(cycle.diagnostics(), "io.pcfb.cycle"));
    REQUIRE(diagnosticsText(cycle.diagnostics()).find("0 -> 1 -> 2 -> 0") != std::string::npos);

    const auto selfLoop = order({edge{.source = pc::bool_node_id(1)}, edge{.source = pc::bool_node_id(1)}, edge{.source = pc::bool_node_id(1)}});
    REQUIRE(hasCode(selfLoop.diagnostics(), "io.pcfb.cycle"));
    REQUIRE(diagnosticsText(selfLoop.diagnostics()).find("1 -> 1") != std::string::npos);

    const auto invalid = order({edge{.source = pc::bool_node_id(7)}, edge{.source = pc::bool_node_id::invalid()}, edge{.source = pc::bool_node_id(0)}});
    REQUIRE(hasCode(invalid.diagnostics(), "io.pcfb.invalid-edge"));
    REQUIRE(invalid.diagnostics().size() == 2);

    const std::vector<std::uint32_t> shortCounts{1, 1};
    const auto mismatched = writer::detail::emission_order(writer::detail::graph_view{
        .op_codes = codes,
        .operand_offsets = offsets,
        .operand_counts = shortCounts,
        .edges = {},
    });
    REQUIRE(hasCode(mismatched.diagnostics(), "io.pcfb.invalid-ir"));
}

TEST_CASE("pcfb emission order is iterative at 1e5 depth", "[io][pcfb]") {
    constexpr std::uint32_t kDepth = 100'000;
    std::vector<pc::op_code> codes(kDepth, pc::op_code::not_gate);
    codes.back() = pc::op_code::basic_event;
    std::vector<std::uint32_t> offsets(kDepth);
    std::iota(offsets.begin(), offsets.end(), 0U);
    std::vector<std::uint32_t> counts(kDepth, 1);
    counts.back() = 0;
    std::vector<edge> edges;
    for (std::uint32_t node = 0; node + 1 < kDepth; ++node) {
        edges.push_back(edge{.source = pc::bool_node_id(node + 1)});
    }
    const auto order = writer::detail::emission_order(writer::detail::graph_view{
        .op_codes = codes,
        .operand_offsets = offsets,
        .operand_counts = counts,
        .edges = edges,
    });
    REQUIRE(order);
    REQUIRE(order.value().size() == kDepth);
    for (std::uint32_t index = 0; index < kDepth; ++index) {
        REQUIRE(order.value()[index] == kDepth - 1 - index);
    }
}

TEST_CASE("pcfb writer handles a deep top-down circuit", "[io][pcfb]") {
    constexpr std::uint32_t kDepth = 10'000;
    pc::builder builder;
    builder.reserve_nodes(kDepth + 1);
    for (std::uint32_t node = 0; node < kDepth; ++node) {
        const auto next = pc::bool_node_id(node + 1);
        if (node % 2 == 0) {
            (void)builder.add_not(edge{.source = next});
        } else {
            (void)builder.add_or({edge{.source = next}, edge{.source = pc::bool_node_id(kDepth), .complement = true}});
        }
    }
    (void)builder.add_basic_event(builder.add_literal_probability(0.5));
    (void)builder.add_target(pc::bool_node_id(0));
    const auto model = sealed(std::move(builder));
    const auto output = requireWrite(model);
    const auto circuit = view(output);
    requireWriterInvariants(model, output, circuit);
    // Node kDepth - 1 is OR(e, ¬e), so exactly one NOT tensor is added for e.
    REQUIRE(subcircuit(circuit)->operators()->size() == kDepth + 2);
    REQUIRE(output.tensor_by_node[kDepth] == 0);
    REQUIRE(output.tensor_by_node[0] == kDepth + 1);
}

TEST_CASE("pcfb writer is exact on random circuits", "[io][pcfb][random]") {
    std::mt19937_64 rng(0x176176);
    Coverage coverage;
    for (int round = 0; round < 300; ++round) {
        INFO("round " << round);
        auto sealed = randomCircuit(rng, coverage);
        INFO(diagnosticsText(sealed.diagnostics));
        REQUIRE(sealed.ok());
        const auto output = requireWrite(sealed.output);
        const auto circuit = view(output);
        requireWriterInvariants(sealed.output, output, circuit);
        requireExact(sealed.output, circuit);
    }

    for (std::size_t code = 0; code < coverage.op_codes.size(); ++code) {
        INFO("op_code " << code);
        REQUIRE(coverage.op_codes[code] > 0);
    }
    for (std::size_t form = 0; form < coverage.cardinality_forms.size(); ++form) {
        INFO("cardinality form " << form);
        REQUIRE(coverage.cardinality_forms[form] > 0);
    }
    REQUIRE(coverage.complemented_edges > 0);
    REQUIRE(coverage.mixed_polarity_gates > 0);
    REQUIRE(coverage.duplicate_operands > 0);
    REQUIRE(coverage.shared_nodes > 0);
    REQUIRE(coverage.shared_probability_sources > 0);
    REQUIRE(coverage.non_topological_circuits > 0);
    REQUIRE(coverage.extreme_probabilities > 0);
    REQUIRE(coverage.multi_target_circuits > 0);
    REQUIRE(coverage.sixteen_event_circuits > 0);
}

TEST_CASE("FTP fixtures write valid pcfb with exact probabilities", "[io][pcfb][ftp]") {
    const std::map<std::string, double> expected{
        {"and_or_limit.ftp", 0.5884},
        {"atleast.ftp", 0.098},
        {"complement.ftp", 0.08},
        {"import_process.ftp", 0.006},
    };
    const auto files = ftpCorpusFiles();
    REQUIRE_FALSE(files.empty());

    std::size_t checked = 0;
    for (auto const &file : files) {
        INFO(file.filename().string());
        std::ifstream input(file, std::ios::binary);
        REQUIRE(input);
        auto read = ftp_lowering_test::lower_text({std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()});
        INFO(diagnosticsText(read.diagnostics()));
        REQUIRE(read);
        auto const *model = &read.value().circuit;

        writer::write_options options;
        options.model_name = file.stem().string();
        options.source_format = "cafta-ftp";
        const auto written = writer::write(model->to_handle(), options);
        const bool literalOnly =
            std::ranges::all_of(model->probability_sources(), [](auto const &source) { return source.kind == pc::probability_source_kind::literal; });
        if (!literalOnly) {
            REQUIRE_FALSE(written);
            REQUIRE(hasCode(written.diagnostics(), "io.pcfb.parameter-probability"));
            continue;
        }
        INFO(diagnosticsText(written.diagnostics()));
        REQUIRE(written);
        const auto circuit = view(written.value());
        requireWriterInvariants(*model, written.value(), circuit);
        REQUIRE(stringMeta(circuit, "general.model_name") == file.stem().string());

        if (std::ranges::count(model->op_codes(), pc::op_code::basic_event) <= 20) {
            requireExact(*model, circuit);
        }
        if (const auto known = expected.find(file.filename().string()); known != expected.end()) {
            INFO("known P(TOP) " << known->second);
            REQUIRE(std::abs(fuzzybits::compile::exact_probability(circuit) - known->second) <= 1.0e-12);
            ++checked;
        }
    }
    REQUIRE(checked == expected.size());
}
