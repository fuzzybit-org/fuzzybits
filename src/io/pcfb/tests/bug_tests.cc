// Known bugs of the .pcfb writer, the .pcfb reader and the exact evaluator that reads .pcfb files. Each case
// states the docs/pc-pcfb.md rule or ir::pc contract it checks and fails on develop until its issue is fixed.
// Cases tagged [crash] dereference null or write out of bounds today and run as their own ctest entry.
// Labelled `bug`.

#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "compile/exact_eval.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "pcfb.h"
#include "pcfb/builder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pc = fuzzybits::ir::pc;
namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;
using bytes_type = std::vector<std::byte>;

[[nodiscard]] auto diagnosticsText(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
    std::ostringstream output;
    for (auto const &diagnostic : diagnostics.items()) {
        output << diagnostic.code << ": " << diagnostic.message << '\n';
    }
    return output.str();
}

[[nodiscard]] auto hasCode(fuzzybits::api::diagnostic_list const &diagnostics, std::string const &code) -> bool {
    return std::ranges::any_of(diagnostics.items(), [&code](auto const &item) { return item.code == code; });
}

// x (0.1), y (0.2), z (0.3); AND(x, NOT x) and friends make the writer emit its own NOT tensors and the
// compile.complement_tensors entry.
[[nodiscard]] auto mixedPolarityCircuit() -> pc::circuit {
    pc::builder builder;
    const auto x = builder.add_basic_event(builder.add_literal_probability(0.1));
    const auto y = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto first = builder.add_and(
        {pc::edge_ref{.source = x}, pc::edge_ref{.source = x, .complement = true}, pc::edge_ref{.source = y}});
    (void)builder.add_target(first);
    auto sealed = std::move(builder).seal();
    REQUIRE(sealed.ok());
    return std::move(sealed.output);
}

[[nodiscard]] auto requireWrite(pc::circuit const &circuit, writer::write_options const &options = {}) -> bytes_type {
    auto written = writer::write(circuit.to_handle(), options);
    INFO(diagnosticsText(written.diagnostics()));
    REQUIRE(written);
    return std::move(written.value().bytes);
}

[[nodiscard]] auto pack(::pcfb::CircuitT const &model) -> bytes_type {
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(::pcfb::Circuit::Pack(builder, &model), ::pcfb::CircuitIdentifier());
    auto const *data = reinterpret_cast<std::byte const *>(builder.GetBufferPointer());
    return {data, data + builder.GetSize()};
}

[[nodiscard]] auto unpack(std::span<std::byte const> bytes) -> std::unique_ptr<::pcfb::CircuitT> {
    auto circuit = ::pcfb::io::from_bytes(bytes);
    REQUIRE(circuit);
    return std::unique_ptr<::pcfb::CircuitT>((*circuit).get()->UnPack());
}

[[nodiscard]] auto le32(const std::uint32_t value) -> std::vector<std::uint8_t> {
    return {static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8U),
            static_cast<std::uint8_t>(value >> 16U), static_cast<std::uint8_t>(value >> 24U)};
}

} // namespace

// docs/IR-PC-REQUIREMENTS.md section 2: "the R1 target scale is millions of nodes", with 32-bit ids, and
// "Builders must diagnose attempts to exceed the representable R1 id or offset range". pcfb::io::from_bytes
// verifies with flatbuffers::Verifier's default options (at most 1,000,000 tables), and a file takes about
// two tables per tensor (Tensor and Op) plus one per SAMPLE, so every circuit above about 3.3e5 to 5e5
// nodes fails the writer's own read-back check with io.pcfb.invalid-output ("FlatBuffer verification
// failed"), and no reader (fzb run, fzb inspect, Python) could load it either. #285 is this limit, reached
// through the frequency lowering.
TEST_CASE("the writer writes, and the reader reads, a valid 6e5-node circuit (#285)", "[bug][pcfb][scale][issue-285]") {
    constexpr std::uint32_t events = 300'000;
    pc::builder builder;
    builder.reserve_nodes(2 * events);
    const auto source = builder.add_literal_probability(1.0e-3);
    auto chain = builder.add_basic_event(source);
    for (std::uint32_t index = 1; index < events; ++index) {
        const auto event = builder.add_basic_event(source);
        chain = builder.add_or({pc::edge_ref{.source = chain}, pc::edge_ref{.source = event}});
    }
    (void)builder.add_target(chain);
    auto sealed = std::move(builder).seal();
    REQUIRE(sealed.ok());
    REQUIRE(sealed.output.node_count() == 2 * events - 1);

    auto written = writer::write(sealed.output.to_handle(), {});
    INFO(diagnosticsText(written.diagnostics()));
    CHECK_FALSE(hasCode(written.diagnostics(), "io.pcfb.invalid-output"));
    REQUIRE(written);
    auto read = writer::read(std::span<std::byte const>(written.value().bytes));
    INFO(diagnosticsText(read.diagnostics()));
    REQUIRE(read);
    CHECK(read.value().circuit.node_count() == 2 * events - 1);
}

// docs/pc-pcfb.md (Reading, Diagnostics): an interpreted metadata entry of the wrong type or range is
// io.pcfb.invalid-metadata and ignored, and every metadata entry is kept verbatim. A KeyValue whose union
// type is set but whose value table is absent passes the FlatBuffer verifier and pcfb::validate; the reader
// then dereferences the missing value (reader.cpp, readMetadata).
TEST_CASE("pcfb reader: a metadata entry with a type and no value is not a crash",
          "[bug][pcfb][read][crash][crash-null-metadata][issue-292]") {
    const auto model = unpack(requireWrite(mixedPolarityCircuit()));
    flatbuffers::FlatBufferBuilder fbb;
    std::vector<flatbuffers::Offset<::pcfb::SubCircuit>> subcircuits;
    for (auto const &subcircuit : model->subcircuits) {
        subcircuits.push_back(::pcfb::SubCircuit::Pack(fbb, subcircuit.get()));
    }
    std::vector<flatbuffers::Offset<::pcfb::Buffer>> buffers;
    for (auto const &buffer : model->buffers) {
        buffers.push_back(::pcfb::Buffer::Pack(fbb, buffer.get()));
    }
    std::vector<flatbuffers::Offset<::pcfb::KeyValue>> metadata;
    for (auto const &entry : model->metadata) {
        metadata.push_back(::pcfb::KeyValue::Pack(fbb, entry.get()));
    }
    // Type IntValue, value offset 0 (absent).
    metadata.push_back(::pcfb::CreateKeyValueDirect(fbb, "zz.no_value", ::pcfb::MetadataValue::IntValue));
    const auto circuit = ::pcfb::CreateCircuit(fbb, fbb.CreateVector(subcircuits), fbb.CreateVector(buffers),
                                               fbb.CreateString(model->name), 0, model->schema_version,
                                               fbb.CreateVectorOfSortedTables(&metadata));
    fbb.Finish(circuit, ::pcfb::CircuitIdentifier());
    auto const *data = reinterpret_cast<std::byte const *>(fbb.GetBufferPointer());
    const bytes_type bytes(data, data + fbb.GetSize());

    auto view = ::pcfb::io::from_bytes(bytes);
    REQUIRE(view);
    const auto read = writer::read(std::span<std::byte const>(bytes));
    INFO(diagnosticsText(read.diagnostics()));
    CHECK(read);
}

// docs/pc-pcfb.md (Diagnostics): "a malformed or unusable compile.complement_tensors entry ... is ignored"
// with io.pcfb.invalid-metadata. The reader checks only that each listed id is below the tensor count; a
// tensor in range that no operator writes maps to the producer index kNone (0xFFFFFFFF), which is then used
// as an index (reader.cpp, foldComplementTensors).
TEST_CASE("pcfb reader: compile.complement_tensors naming a tensor no operator writes is ignored",
          "[bug][pcfb][read][crash][crash-orphan-complement][issue-292]") {
    auto model = unpack(requireWrite(mixedPolarityCircuit()));
    auto &tensors = model->subcircuits[0]->tensors;
    const auto orphan = static_cast<std::uint32_t>(tensors.size());
    tensors.push_back(std::make_unique<::pcfb::TensorT>());
    bool listed = false;
    for (auto &entry : model->metadata) {
        if (entry->key == "compile.complement_tensors") {
            ::pcfb::BytesValueT value;
            value.value = le32(orphan);
            entry->value.Set(value);
            listed = true;
        }
    }
    REQUIRE(listed);
    const auto bytes = pack(*model);
    auto view = ::pcfb::io::from_bytes(bytes);
    REQUIRE(view);
    REQUIRE(::pcfb::validate(*view).empty());

    const auto read = writer::read(std::span<std::byte const>(bytes));
    INFO(diagnosticsText(read.diagnostics()));
    REQUIRE(read);
    CHECK(hasCode(read.diagnostics(), "io.pcfb.invalid-metadata"));
}

// compile::exact_probability backs `fzb inspect --exact` (which runs on after a failed --validate) and
// Python's exact_pcfb_bytes/exact_pcfb_path, which do not validate first. It indexes its value table by
// operator outputs and inputs without bounds checks, so a file whose operator writes tensor 1000 of 3 is an
// out-of-bounds write. Expected: pcfb validation errors or ExactEvalError, not memory corruption.
TEST_CASE("exact_probability rejects a file whose operator writes a tensor out of range",
          "[bug][pcfb][exact][crash][crash-exact-bounds][issue-292]") {
    ::pcfb::builder builder;
    auto always = builder.add_sample(1.0, "a", 1);
    auto other = builder.add_sample(0.5, "b", 2);
    REQUIRE(always);
    REQUIRE(other);
    const std::uint32_t inputs[] = {*always, *other};
    auto gate = builder.add_op(::pcfb::OpCode::BITWISE_AND, inputs);
    REQUIRE(gate);
    REQUIRE(builder.add_query(*gate));
    auto finished = builder.finish();
    REQUIRE(finished);

    auto model = unpack(*finished);
    model->subcircuits[0]->operators[0]->output = 1000;
    const auto bytes = pack(*model);
    auto view = ::pcfb::io::from_bytes(bytes);
    REQUIRE(view);
    REQUIRE_FALSE(::pcfb::validate(*view).empty());
    CHECK_THROWS_AS(fuzzybits::compile::exact_probability(*view), fuzzybits::compile::ExactEvalError);
}

// docs/pc-pcfb.md (Diagnostics): io.pcfb.invalid-metadata is "an interpreted metadata entry [that] has the wrong
// type or range". compile.query.<i>.initiator_frequency is a query's weight in its frequency group, and the
// writer refuses a negative or non-finite group weight (io.pcfb.invalid-query-group); the same value as
// per-query metadata is read without a diagnostic.
TEST_CASE("pcfb reader: a negative or non-finite initiator_frequency is invalid metadata",
          "[bug][pcfb][read][issue-297]") {
    for (const double frequency : {-1.0, std::numeric_limits<double>::quiet_NaN()}) {
        INFO("initiator_frequency " << frequency);
        ::pcfb::builder builder;
        auto event = builder.add_sample(0.5, "e", 1);
        REQUIRE(event);
        REQUIRE(builder.add_query(*event, ::pcfb::QueryKind::PROBABILITY, "T|IE"));
        REQUIRE(builder.set("compile.query.0.target", std::string_view("T")));
        REQUIRE(builder.set("compile.query.0.initiator", std::string_view("IE")));
        REQUIRE(builder.set("compile.query.0.initiator_frequency", frequency));
        auto bytes = builder.finish();
        REQUIRE(bytes);
        const auto read = writer::read(std::span<std::byte const>(*bytes));
        INFO(diagnosticsText(read.diagnostics()));
        REQUIRE(read);
        CHECK(hasCode(read.diagnostics(), "io.pcfb.invalid-metadata"));
    }
}
