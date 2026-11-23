#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include "api/diagnostics.h"
#include "api/ids.h"
#include "compile/exact_eval.h"
#include "compile/ftp/compile.h"
#include "io/descriptor.h"
#include "io/formats.h"
#include "io/ftp/descriptor.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/target.h"

#include "pcfb.h"
#include "pcfb/builder.h"
#include "pcfb/io.h"
#include "pcfb/validate.h"
#include "pcfb_generated.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <sstream>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

    using namespace pcfb_writer_test;
    namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;
    namespace ftp = fuzzybits::compile::ftp;

    using kind = writer::query_kind;
    using bytes_type = std::vector<std::byte>;

    [[nodiscard]] auto readOk(std::span<std::byte const> bytes) -> writer::read_output {
        auto result = writer::read(bytes);
        INFO(diagnosticsText(result.diagnostics()));
        REQUIRE(result);
        return std::move(result).value();
    }

    [[nodiscard]] auto writeBack(writer::read_output const &read) -> bytes_type {
        auto written = writer::write(read.circuit.to_handle(), read.options);
        INFO(diagnosticsText(written.diagnostics()));
        REQUIRE(written);
        return std::move(written).value().bytes;
    }

    // The ids of every diagnostic, for exact-set assertions.
    [[nodiscard]] auto codesOf(fuzzybits::api::diagnostic_list const &diagnostics) -> std::set<std::string> {
        std::set<std::string> codes;
        for (auto const &item : diagnostics.items()) {
            codes.insert(item.code);
        }
        return codes;
    }

    struct edge_key final {
        std::uint32_t source;
        bool complement;
        friend auto operator<=>(edge_key const &, edge_key const &) = default;
    };

    // The operands of a node as a multiset of (mapped source, complement).
    template <typename mapper> [[nodiscard]] auto operandsOf(pc::circuit const &circuit, const pc::bool_node_id node, mapper &&map) -> std::vector<edge_key> {
        std::vector<edge_key> edges;
        for (auto const &edge : circuit.operands(node)) {
            edges.push_back({map(edge.source), edge.complement});
        }
        std::ranges::sort(edges);
        return edges;
    }

    [[nodiscard]] auto eventIds(pc::circuit const &circuit) -> std::vector<std::uint32_t> {
        std::vector<std::uint32_t> ids(circuit.node_count(), 0);
        std::uint32_t next = 0;
        for (std::uint32_t node = 0; node < circuit.node_count(); ++node) {
            if (circuit.op_codes()[node] == pc::op_code::basic_event) {
                ids[node] = ++next;
            }
        }
        return ids;
    }

    [[nodiscard]] auto effectiveKinds(writer::write_options const &options, pc::target_id target) -> std::vector<kind> {
        const auto found = options.target_kinds.find(target);
        return found == options.target_kinds.end() ? std::vector<kind>{options.query} : found->second;
    }

    // The read circuit has the structure of the written one, seen through the maps, node for node. What
    // pcfb records differently: house events and a cardinality without operands (a ONES tensor) read as
    // constants, and the order of the operands is the file's.
    void requireSameStructure(RandomModel const &model, writer::write_output const &written, writer::read_output const &read) {
        auto const &original = model.circuit;
        auto const &options = model.options;
        auto const &circuit = read.circuit;
        REQUIRE(circuit.valid());
        REQUIRE(written.tensor_by_node.size() == original.node_count());
        const auto readEvents = eventIds(circuit);

        const auto mapped = [&](const pc::bool_node_id node) -> pc::bool_node_id {
            return read.node_by_tensor.at(written.tensor_by_node.at(node.raw()));
        };
        const auto mappedId = [&](const pc::bool_node_id node) {
            const auto result = mapped(node);
            REQUIRE(result.valid());
            return result.raw();
        };

        for (std::uint32_t v = 0; v < original.node_count(); ++v) {
            INFO("node " << v);
            const auto node = pc::bool_node_id(v);
            const auto code = original.op_codes()[v];
            const auto fanIn = original.operand_counts()[v];
            const auto name = options.node_names.find(node);
            REQUIRE(mapped(node).valid());
            const auto r = mapped(node).raw();
            const bool fullRange = code == pc::op_code::cardinality && fanIn == 0;
            auto expected = code == pc::op_code::house_event || fullRange ? pc::op_code::constant : code;
            REQUIRE(circuit.op_codes()[r] == expected);
            REQUIRE(read.tensor_by_node.at(r) == written.tensor_by_node[v]);

            std::string expectedName = name == options.node_names.end() ? std::string{} : name->second;
            if (code == pc::op_code::basic_event && expectedName.empty()) {
                expectedName = "e" + std::to_string(written.event_id_by_node[v]);
            }
            const auto readName = read.options.node_names.find(pc::bool_node_id(r));
            REQUIRE((readName == read.options.node_names.end() ? std::string{} : readName->second) == expectedName);

            switch (code) {
            case pc::op_code::basic_event: {
                REQUIRE(readEvents[r] == written.event_id_by_node[v]);
                auto const &sourceOf = original.probability_sources()[original.probability_source_refs()[v].raw()];
                auto const &readSource = circuit.probability_sources()[circuit.probability_source_refs()[r].raw()];
                REQUIRE(readSource.kind == pc::probability_source_kind::literal);
                REQUIRE(readSource.literal == sourceOf.literal);
                break;
            }
            case pc::op_code::constant:
            case pc::op_code::house_event:
                REQUIRE(circuit.constant_values()[r] == original.constant_values()[v]);
                break;
            default:
                if (fullRange) {
                    REQUIRE(circuit.constant_values()[r] == 1);
                    break;
                }
                if (code == pc::op_code::cardinality) {
                    REQUIRE(circuit.cardinality_low()[r] == original.cardinality_low()[v]);
                    REQUIRE(circuit.cardinality_high()[r] == original.cardinality_high()[v]);
                }
                REQUIRE(operandsOf(original, node, mappedId) ==
                        operandsOf(circuit, pc::bool_node_id(r), [](const pc::bool_node_id source) { return source.raw(); }));
                break;
            }
        }
        REQUIRE(circuit.node_count() == original.node_count());

        REQUIRE(circuit.targets().size() == original.targets().size());
        for (std::size_t index = 0; index < original.targets().size(); ++index) {
            INFO("target " << index);
            auto const &before = original.targets()[index];
            auto const &after = circuit.targets()[index];
            REQUIRE(after.id == before.id);
            REQUIRE(after.node == mapped(before.node));
            REQUIRE(after.role == before.role);
            const auto named = options.target_names.find(before.id);
            const auto readNamed = read.options.target_names.find(after.id);
            REQUIRE(readNamed != read.options.target_names.end());
            REQUIRE(readNamed->second == (named == options.target_names.end() ? "t" + std::to_string(before.id.raw()) : named->second));
            REQUIRE(read.options.target_kinds.at(after.id) == effectiveKinds(options, before.id));
        }

        REQUIRE(read.options.quantification.size() == options.quantification.size());
        for (auto const &[target, entry] : options.quantification) {
            INFO("quantification of target " << target.raw());
            const auto found = read.options.quantification.find(target);
            REQUIRE(found != read.options.quantification.end());
            REQUIRE(found->second.literal_mode == entry.literal_mode);
            REQUIRE(found->second.max_initiators_per_cutset == entry.max_initiators_per_cutset);
        }
        REQUIRE(read.options.model_name == options.model_name);
        REQUIRE(read.options.source_format == options.source_format);
        REQUIRE(read.options.source_sha256 == options.source_sha256);
        REQUIRE(read.options.compile_options == options.compile_options);
        REQUIRE(read.options.query == kind::probability);
    }

    // Two read circuits are the same structure, id for id.
    void requireEqual(writer::read_output const &left, writer::read_output const &right) {
        auto const &a = left.circuit;
        auto const &b = right.circuit;
        REQUIRE(a.node_count() == b.node_count());
        REQUIRE(a.edge_count() == b.edge_count());
        for (std::uint32_t node = 0; node < a.node_count(); ++node) {
            INFO("node " << node);
            REQUIRE(a.op_codes()[node] == b.op_codes()[node]);
            REQUIRE(a.cardinality_low()[node] == b.cardinality_low()[node]);
            REQUIRE(a.cardinality_high()[node] == b.cardinality_high()[node]);
            REQUIRE(a.constant_values()[node] == b.constant_values()[node]);
            const auto id = pc::bool_node_id(node);
            REQUIRE(a.operands(id).size() == b.operands(id).size());
            for (std::size_t edge = 0; edge < a.operands(id).size(); ++edge) {
                REQUIRE(a.operands(id)[edge].source == b.operands(id)[edge].source);
                REQUIRE(a.operands(id)[edge].complement == b.operands(id)[edge].complement);
            }
            if (a.op_codes()[node] == pc::op_code::basic_event) {
                REQUIRE(a.probability_sources()[a.probability_source_refs()[node].raw()].literal ==
                        b.probability_sources()[b.probability_source_refs()[node].raw()].literal);
            }
        }
        REQUIRE(a.targets().size() == b.targets().size());
        for (std::size_t target = 0; target < a.targets().size(); ++target) {
            REQUIRE(a.targets()[target].node == b.targets()[target].node);
        }
        REQUIRE(left.options.node_names == right.options.node_names);
        REQUIRE(left.options.target_names == right.options.target_names);
        REQUIRE(left.options.target_kinds == right.options.target_kinds);
        REQUIRE(left.options.model_name == right.options.model_name);
        REQUIRE(left.options.source_format == right.options.source_format);
        REQUIRE(left.options.compile_options == right.options.compile_options);
        REQUIRE(left.options.quantification.size() == right.options.quantification.size());
    }

    // Rewriting a read circuit and reading that back keeps its structure through the tensor maps, and
    // a second rewrite reproduces the numbering exactly.
    void requireStable(writer::read_output const &read) {
        auto rewritten = writer::write(read.circuit.to_handle(), read.options);
        INFO(diagnosticsText(rewritten.diagnostics()));
        REQUIRE(rewritten);
        const auto reread = readOk(rewritten.value().bytes);
        requireSameStructure(RandomModel{read.circuit, read.options}, rewritten.value(), reread);
        requireEqual(reread, readOk(writeBack(reread)));
    }

    struct round_trip_stats final {
        std::size_t circuits = 0;
        std::size_t nodes = 0;
    };

    // The read circuit has the structure of the written one, and rewriting it and reading that back
    // changes nothing under the same comparison.
    void roundTripRandom(RandomModel const &model, round_trip_stats &stats) {
        auto written = writer::write(model.circuit.to_handle(), model.options);
        INFO(diagnosticsText(written.diagnostics()));
        REQUIRE(written);
        auto const &output = written.value();
        auto result = writer::read(output.bytes);
        INFO(diagnosticsText(result.diagnostics()));
        REQUIRE(result);
        auto const &read = result.value();
        const bool hull = std::ranges::any_of(model.circuit.targets(), [&](auto const &target) {
            const auto kinds = effectiveKinds(model.options, target.id);
            return std::ranges::find(kinds, kind::monotone_hull_probability) != kinds.end();
        });
        REQUIRE(read.schema_version == (hull ? 0x00010002U : 0x00010001U));
        requireSameStructure(model, output, read);
        requireStable(read);
        ++stats.circuits;
        stats.nodes += model.circuit.node_count();
    }

    [[nodiscard]] auto readTextFile(std::filesystem::path const &path) -> std::string {
        std::ifstream input(path, std::ios::binary);
        REQUIRE(input);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    [[nodiscard]] auto readBinaryFile(std::filesystem::path const &path) -> bytes_type {
        const auto text = readTextFile(path);
        bytes_type bytes(text.size());
        std::ranges::transform(text, bytes.begin(), [](const char c) { return static_cast<std::byte>(c); });
        return bytes;
    }

    [[nodiscard]] auto asBytes(std::string const &text) -> bytes_type {
        bytes_type bytes(text.size());
        std::ranges::transform(text, bytes.begin(), [](const char c) { return static_cast<std::byte>(c); });
        return bytes;
    }

    // pcfb object-API plumbing for files the pcfb builder cannot produce.
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

    [[nodiscard]] auto sampleBytes(std::vector<double> const &probabilities, std::vector<std::uint32_t> const &eventIds,
                                   ::pcfb::SampleSource source = ::pcfb::SampleSource::BASIC_EVENT) -> ::pcfb::builder {
        ::pcfb::builder builder;
        for (std::size_t index = 0; index < probabilities.size(); ++index) {
            REQUIRE(builder.add_sample(probabilities[index], "s" + std::to_string(index), eventIds[index], source));
        }
        return builder;
    }

} // namespace

TEST_CASE("pcfb reader descriptor and endpoint", "[io][pcfb][read]") {
    const auto descriptor = writer::reader_descriptor();
    CHECK(descriptor.input_format == fuzzybits::io::formats::pcfb());
    CHECK(descriptor.endpoint == writer::reader_endpoint_id());
    CHECK(descriptor.endpoint != writer::writer_endpoint_id());
    CHECK(descriptor.produces.contains(fuzzybits::api::capability_id("ir.probabilistic_boolean_dag")));

    const auto model = threeTargetModel();
    const auto written = requireWrite(model, threeTargetNames());
    const auto path = std::filesystem::temp_directory_path() / ("fzb_pcfb_reader_endpoint_" + std::to_string(std::random_device{}()) + ".pcfb");
    {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<char const *>(written.bytes.data()), static_cast<std::streamsize>(written.bytes.size()));
    }
    fuzzybits::io::read_request request{.path = path, .format = fuzzybits::io::formats::pcfb()};
    const auto handle = writer::read(request);
    INFO(diagnosticsText(handle.diagnostics()));
    REQUIRE(handle);
    auto const *circuit = pc::circuit_from(handle.value());
    REQUIRE(circuit != nullptr);
    CHECK(circuit->satisfies_form(fuzzybits::api::form_id("ir.probabilistic_boolean_dag")));
    CHECK(circuit->node_count() == model.node_count());

    const auto overPath = writer::read(path);
    REQUIRE(overPath);
    CHECK(overPath.value().tensor_by_node.size() == model.node_count());

    request.path = path.string() + ".missing";
    const auto missing = writer::read(request);
    CHECK_FALSE(missing);
    CHECK(hasCode(missing.diagnostics(), std::string(writer::diag::unreadable)));

    request.path = path;
    request.format = fuzzybits::io::formats::ftp();
    CHECK(hasCode(writer::read(request).diagnostics(), std::string(writer::diag::format_mismatch)));
    std::filesystem::remove(path);
}

TEST_CASE("pcfb reader recovers names, kinds and the compile metadata", "[io][pcfb][read]") {
    const auto model = threeTargetModel();
    auto options = threeTargetNames();
    options.target_kinds = {{pc::target_id(0), {kind::probability, kind::monotone_hull_probability}},
                            {pc::target_id(1), {kind::monotone_hull_probability}}};
    options.quantification = {
        {pc::target_id(0), writer::target_quantification{.literal_mode = "atom_split", .max_initiators_per_cutset = 1}},
        {pc::target_id(1), writer::target_quantification{.literal_mode = "omega_l"}},
    };
    options.node_names = {{pc::bool_node_id(0), "PUMP-A"}, {pc::bool_node_id(3), "BOTH"}};
    options.model_name = "demo";
    options.source_format = "cafta-ftp";
    options.compile_options = "-c 2";
    std::array<std::byte, 32> digest{};
    digest[0] = std::byte{0xAB};
    digest[31] = std::byte{0x01};
    options.source_sha256 = digest;

    const auto written = requireWrite(model, options);
    const auto read = readOk(written.bytes);
    CHECK(read.schema_version == 0x00010002U);
    CHECK(read.options.target_names == options.target_names);
    CHECK(read.options.target_kinds.at(pc::target_id(0)) == std::vector<kind>{kind::probability, kind::monotone_hull_probability});
    CHECK(read.options.target_kinds.at(pc::target_id(1)) == std::vector<kind>{kind::monotone_hull_probability});
    CHECK(read.options.target_kinds.at(pc::target_id(2)) == std::vector<kind>{kind::probability});
    CHECK(read.circuit.targets().size() == 3);
    CHECK(read.options.quantification.size() == 2);
    CHECK(read.options.quantification.at(pc::target_id(0)).literal_mode == "atom_split");
    CHECK(read.options.quantification.at(pc::target_id(0)).max_initiators_per_cutset == 1U);
    CHECK(read.options.quantification.at(pc::target_id(1)).literal_mode == "omega_l");
    CHECK_FALSE(read.options.quantification.at(pc::target_id(1)).max_initiators_per_cutset.has_value());
    CHECK(read.options.node_names.at(pc::bool_node_id(0)) == "PUMP-A");
    CHECK(read.options.node_names.at(pc::bool_node_id(3)) == "BOTH");
    CHECK(read.options.node_names.at(pc::bool_node_id(1)) == "e2");
    CHECK(read.options.model_name == "demo");
    CHECK(read.options.source_format == "cafta-ftp");
    CHECK(read.options.compile_options == "-c 2");
    CHECK(read.options.source_sha256 == digest);
    CHECK(read.options.query == kind::probability);

    // Every metadata entry verbatim, in file order.
    auto file = ::pcfb::io::from_bytes(written.bytes);
    REQUIRE(file);
    REQUIRE((*file)->metadata() != nullptr);
    REQUIRE(read.metadata.size() == (*file)->metadata()->size());
    for (std::size_t index = 0; index < read.metadata.size(); ++index) {
        CHECK(read.metadata[index].first == (*file)->metadata()->Get(static_cast<flatbuffers::uoffset_t>(index))->key()->str());
    }
    const auto find = [&](std::string const &key) -> writer::metadata_value const * {
        for (auto const &entry : read.metadata) {
            if (entry.first == key) {
                return &entry.second;
            }
        }
        return nullptr;
    };
    REQUIRE(find("general.generator") != nullptr);
    CHECK(std::get<std::string>(*find("general.generator")) == "fuzzybits");
    CHECK(std::get<std::string>(*find("pcfb.opset")) == "core-1.1");
    CHECK(std::get<std::int64_t>(*find("compile.query.0.max_initiators_per_cutset")) == 1);
    CHECK(std::get<std::string>(*find("compile.query.2.negation")) == "monotone_hull");
    CHECK(std::get<std::vector<std::byte>>(*find("general.source_sha256")).size() == 32);

    requireStable(read);
}

TEST_CASE("pcfb reader accepts any minor of schema major 1 and keeps the version it read", "[io][pcfb][read]") {
    const auto written = requireWrite(threeTargetModel(), threeTargetNames());
    auto model = unpack(written.bytes);
    model->schema_version = 0x00010007U;
    const auto patched = pack(*model);
    const auto read = readOk(patched);
    CHECK(read.schema_version == 0x00010007U);
    // The writer picks the lowest minor again.
    const auto rewritten = writeBack(read);
    const auto view = ::pcfb::io::from_bytes(rewritten);
    REQUIRE(view);
    CHECK((*view)->schema_version() == 0x00010001U);
}

TEST_CASE("pcfb reader keeps the structure of random circuits and rewrites them stably", "[io][pcfb][read][random]") {
    std::mt19937_64 rng(0x219219);
    Coverage coverage;
    round_trip_stats stats;
    for (int round = 0; round < 360; ++round) {
        INFO("round " << round);
        const auto model = round % 3 == 0 ? randomModel(rng, coverage) : randomModel(rng, coverage, RandomSize{.max_events = 300, .max_gates = 700});
        roundTripRandom(model, stats);
    }
    CHECK(stats.circuits == 360);
    CHECK(stats.nodes > 100'000);
    for (std::size_t code = 0; code < coverage.op_codes.size(); ++code) {
        INFO("op_code " << code);
        REQUIRE(coverage.op_codes[code] > 0);
    }
    for (std::size_t form = 0; form < coverage.cardinality_forms.size(); ++form) {
        INFO("cardinality form " << form);
        REQUIRE(coverage.cardinality_forms[form] > 0);
    }
    REQUIRE(coverage.mixed_polarity_gates > 0);
    REQUIRE(coverage.duplicate_operands > 0);
    REQUIRE(coverage.non_topological_circuits > 0);
    REQUIRE(coverage.multi_target_circuits > 0);
}

TEST_CASE("pcfb reader keeps the event order of a circuit whose gates come first", "[io][pcfb][read]") {
    // Events a < b < c by node id, listed by the top gate as c, a, b: the file emits the samples in that
    // order and the read circuit must still number a, b, c.
    pc::builder builder;
    const auto top = builder.add_and({pc::edge_ref{.source = pc::bool_node_id(3)}, pc::edge_ref{.source = pc::bool_node_id(1)},
                                      pc::edge_ref{.source = pc::bool_node_id(2), .complement = true}});
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
    const auto b = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto c = builder.add_basic_event(builder.add_literal_probability(0.3));
    (void)builder.add_target(top);
    const auto model = sealed(std::move(builder));
    REQUIRE(model.node_count() == 4);
    const auto written = requireWrite(model);
    auto file = ::pcfb::io::from_bytes(written.bytes);
    REQUIRE(file);
    CHECK(written.event_id_by_node[a.raw()] == 1);
    CHECK(written.event_id_by_node[c.raw()] == 3);
    CHECK(written.tensor_by_node[c.raw()] < written.tensor_by_node[a.raw()]);

    const auto read = readOk(written.bytes);
    std::vector<double> probabilities;
    for (std::uint32_t node = 0; node < read.circuit.node_count(); ++node) {
        if (read.circuit.op_codes()[node] == pc::op_code::basic_event) {
            probabilities.push_back(read.circuit.probability_sources()[read.circuit.probability_source_refs()[node].raw()].literal);
        }
    }
    CHECK(probabilities == std::vector<double>{0.1, 0.2, 0.3});
    // Canonical numbering: events by file event id, then the other operators in file order; operands
    // as the file lists them, inputs before inverted_inputs.
    for (std::uint32_t node = 0; node < 3; ++node) {
        CHECK(read.circuit.op_codes()[node] == pc::op_code::basic_event);
    }
    CHECK(read.circuit.op_codes()[3] == pc::op_code::and_gate);
    const auto operands = read.circuit.operands(pc::bool_node_id(3));
    REQUIRE(operands.size() == 3);
    CHECK(operands[0].source == read.node_by_tensor[written.tensor_by_node[c.raw()]]);
    CHECK_FALSE(operands[0].complement);
    CHECK(operands[1].source == read.node_by_tensor[written.tensor_by_node[a.raw()]]);
    CHECK_FALSE(operands[1].complement);
    CHECK(operands[2].source == read.node_by_tensor[written.tensor_by_node[b.raw()]]);
    CHECK(operands[2].complement);
    requireStable(read);
}

TEST_CASE("pcfb reader: FTP conformance fixtures compile, read and rewrite to the same structure", "[io][pcfb][read][ftp]") {
    const auto root = projectRoot() / "input" / "ftp" / "conformance";
    std::vector<std::filesystem::path> directories;
    for (auto const &entry : std::filesystem::directory_iterator(root)) {
        if (entry.is_directory() && entry.path().filename().string().rfind("invalid", 0) != 0) {
            directories.push_back(entry.path());
        }
    }
    std::ranges::sort(directories);
    REQUIRE(directories.size() >= 10);

    struct variant final {
        std::string label;
        ftp::options settings;
    };
    std::vector<variant> variants;
    {
        ftp::options boolean;
        boolean.negation = "boolean";
        variants.push_back({"boolean", boolean});
        ftp::options hull;
        hull.complement_events = "negation";
        variants.push_back({"hull", hull});
        ftp::options both;
        both.complement_events = "negation";
        both.config = R"({"negation": "both"})";
        variants.push_back({"both", both});
        ftp::options literals;
        literals.literals = "negation";
        variants.push_back({"hull with atoms as negation", literals});
    }

    std::size_t compiled = 0;
    std::size_t hullFiles = 0;
    std::size_t twoKindTargets = 0;
    std::size_t roundTrips = 0;
    for (auto const &directory : directories) {
        std::vector<std::filesystem::path> files;
        for (auto const &entry : std::filesystem::directory_iterator(directory)) {
            if (entry.path().extension() == ".ftp") {
                files.push_back(entry.path());
            }
        }
        std::ranges::sort(files);
        std::vector<fuzzybits::io::backend::fuzzybits::ftp::source_text> sources;
        for (auto const &file : files) {
            sources.push_back({.path = file.filename().string(), .text = readTextFile(file)});
        }
        for (auto const &entry : variants) {
            INFO(directory.filename().string() << " / " << entry.label);
            const auto result = ftp::compile_sources(sources, entry.settings);
            if (!result) {
                continue;
            }
            ++compiled;
            auto const &bytes = result.value().bytes;
            const auto read = readOk(bytes);
            requireStable(read);
            ++roundTrips;
            hullFiles += read.schema_version == 0x00010002U ? 1 : 0;
            for (auto const &[target, kinds] : read.options.target_kinds) {
                twoKindTargets += kinds.size() > 1 ? 1 : 0;
            }
        }
    }
    INFO(compiled << " compiled files, " << hullFiles << " with hull queries, " << twoKindTargets << " two-kind targets");
    CHECK(compiled >= 20);
    CHECK(roundTrips == compiled);
    CHECK(hullFiles > 0);
    CHECK(twoKindTargets > 0);
}

TEST_CASE("pcfb reader: tracked .pcfb files read to a fixed point or are rejected with a documented id", "[io][pcfb][read]") {
    std::vector<std::filesystem::path> files;
    const auto root = projectRoot();
    for (auto iterator = std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::skip_permission_denied);
         iterator != std::filesystem::recursive_directory_iterator(); ++iterator) {
        const auto name = iterator->path().filename().string();
        if (iterator->is_directory() && (name == ".git" || name == "_deps" || name == "node_modules" || name.rfind("build", 0) == 0 ||
                                         name.rfind(".venv", 0) == 0 || name.rfind("cmake-build", 0) == 0)) {
            iterator.disable_recursion_pending();
            continue;
        }
        if (iterator->is_regular_file() && iterator->path().extension() == ".pcfb") {
            files.push_back(iterator->path());
        }
    }
    std::ranges::sort(files);
    REQUIRE_FALSE(files.empty());

    for (auto const &file : files) {
        INFO(file.string());
        const auto bytes = readBinaryFile(file);
        const auto first = writer::read(bytes);
        INFO(diagnosticsText(first.diagnostics()));
        if (!first) {
            for (auto const &item : first.diagnostics().items()) {
                CHECK(item.code.rfind("io.pcfb.", 0) == 0);
            }
            CHECK(first.diagnostics().has_errors());
            continue;
        }
        const auto second = readOk(writeBack(first.value()));
        CHECK(first.value().schema_version == second.schema_version);
        requireStable(first.value());
    }
}

TEST_CASE("pcfb reader: the hull query fixture keeps its kinds", "[io][pcfb][read]") {
    const auto bytes = readBinaryFile(projectRoot() / "tests" / "python" / "data" / "monotone_hull_query.pcfb");
    const auto read = readOk(bytes);
    CHECK(read.schema_version == 0x00010002U);
    bool hull = false;
    for (auto const &[target, kinds] : read.options.target_kinds) {
        hull = hull || std::ranges::find(kinds, kind::monotone_hull_probability) != kinds.end();
    }
    CHECK(hull);
}

TEST_CASE("pcfb reader: a file written by another tool reads with its meaning", "[io][pcfb][read]") {
    // NO_OP and EXACTLY, a source other than BASIC_EVENT, non-contiguous event ids in descending file
    // order, a query with confidence levels.
    auto builder = sampleBytes({0.5, 0.25, 0.125}, {9, 5, 7}, ::pcfb::SampleSource::COLLAPSED_MODULE);
    const std::uint32_t pair[] = {0, 1};
    const std::uint32_t inverted[] = {2};
    const auto both = builder.add_op(::pcfb::OpCode::BITWISE_NO_OP, std::span<std::uint32_t const>{pair, 1});
    REQUIRE(both);
    const auto exactly = builder.add_op(::pcfb::OpCode::BITWISE_EXACTLY, pair, inverted, 1U, "exactly one");
    REQUIRE(exactly);
    const double levels[] = {0.95};
    REQUIRE(builder.add_query(*exactly, ::pcfb::QueryKind::PROBABILITY, "q", levels));
    REQUIRE(builder.add_query(*both, ::pcfb::QueryKind::PROBABILITY, "h"));
    auto bytes = builder.finish();
    REQUIRE(bytes);

    auto result = writer::read(*bytes);
    INFO(diagnosticsText(result.diagnostics()));
    REQUIRE(result);
    CHECK(codesOf(result.diagnostics()) == std::set<std::string>{std::string(writer::diag::sample_source_dropped),
                                                                std::string(writer::diag::confidence_ignored),
                                                                std::string(writer::diag::event_ids_renumbered),
                                                                std::string(writer::diag::no_op_as_and)});
    auto const &read = result.value();
    auto const &circuit = read.circuit;
    REQUIRE(circuit.targets().size() == 2);
    // Events are numbered by ascending file id: 5 (0.25), 7 (0.125), 9 (0.5).
    std::vector<double> probabilities;
    for (std::uint32_t node = 0; node < circuit.node_count(); ++node) {
        if (circuit.op_codes()[node] == pc::op_code::basic_event) {
            probabilities.push_back(circuit.probability_sources()[circuit.probability_source_refs()[node].raw()].literal);
        }
    }
    CHECK(probabilities == std::vector<double>{0.25, 0.125, 0.5});

    const auto exactlyNode = circuit.targets()[0].node;
    CHECK(circuit.op_codes()[exactlyNode.raw()] == pc::op_code::cardinality);
    CHECK(circuit.cardinality_low()[exactlyNode.raw()] == 1);
    CHECK(circuit.cardinality_high()[exactlyNode.raw()] == 1);
    REQUIRE(circuit.operands(exactlyNode).size() == 3);
    CHECK(circuit.operands(exactlyNode)[2].complement);
    CHECK(read.options.node_names.at(exactlyNode) == "exactly one");
    CHECK(circuit.op_codes()[circuit.targets()[1].node.raw()] == pc::op_code::and_gate);
    CHECK(read.options.target_kinds.at(pc::target_id(1)) == std::vector<kind>{kind::probability});

    // Semantics: exact probability of the file equals the exhaustive value of the read circuit.
    auto file = ::pcfb::io::from_bytes(*bytes);
    REQUIRE(file);
    CHECK(std::abs(fuzzybits::compile::exact_probability(*file, 0) - referenceProbabilities(circuit)[0]) <= 1.0e-12);
}

TEST_CASE("pcfb reader: an unnamed ATLEAST/ATMOST pair over one operand list is one cardinality node", "[io][pcfb][read]") {
    auto builder = sampleBytes({0.1, 0.2, 0.3, 0.4}, {1, 2, 3, 4});
    const std::uint32_t operands[] = {0, 1, 2, 3};
    const auto least = builder.add_op(::pcfb::OpCode::BITWISE_ATLEAST, operands, {}, 1U);
    const auto most = builder.add_op(::pcfb::OpCode::BITWISE_ATMOST, operands, {}, 3U);
    REQUIRE(least);
    REQUIRE(most);
    const std::uint32_t both[] = {*least, *most};
    const auto range = builder.add_op(::pcfb::OpCode::BITWISE_AND, both, {}, std::nullopt, "range");
    REQUIRE(range);
    REQUIRE(builder.add_query(*range, ::pcfb::QueryKind::PROBABILITY, "q"));
    // The same pair with a named half is two nodes and an AND.
    const auto namedLeast = builder.add_op(::pcfb::OpCode::BITWISE_ATLEAST, operands, {}, 2U, "named");
    const auto mostAgain = builder.add_op(::pcfb::OpCode::BITWISE_ATMOST, operands, {}, 3U);
    REQUIRE(namedLeast);
    REQUIRE(mostAgain);
    const std::uint32_t pair[] = {*namedLeast, *mostAgain};
    const auto kept = builder.add_op(::pcfb::OpCode::BITWISE_AND, pair, {}, std::nullopt, "kept");
    REQUIRE(kept);
    REQUIRE(builder.add_query(*kept, ::pcfb::QueryKind::PROBABILITY, "k"));
    auto bytes = builder.finish();
    REQUIRE(bytes);

    const auto read = readOk(*bytes);
    auto const &circuit = read.circuit;
    CHECK(std::ranges::count(circuit.op_codes(), pc::op_code::cardinality) == 3);
    CHECK(std::ranges::count(circuit.op_codes(), pc::op_code::and_gate) == 1);
    const auto rangeNode = read.node_by_tensor[*range];
    REQUIRE(rangeNode.valid());
    CHECK(circuit.op_codes()[rangeNode.raw()] == pc::op_code::cardinality);
    CHECK(circuit.cardinality_low()[rangeNode.raw()] == 1);
    CHECK(circuit.cardinality_high()[rangeNode.raw()] == 3);
    CHECK_FALSE(read.node_by_tensor[*least].valid());
    CHECK_FALSE(read.node_by_tensor[*most].valid());
    CHECK(read.node_by_tensor[*namedLeast].valid());
    requireStable(read);
}

TEST_CASE("pcfb reader: unreadable and unrepresentable files are errors with stable ids", "[io][pcfb][read][negative]") {
    const auto written = requireWrite(threeTargetModel(), threeTargetNames());
    const auto codeOfError = [](fuzzybits::api::diagnostic_list const &diagnostics) {
        for (auto const &item : diagnostics.items()) {
            if (item.severity == fuzzybits::api::diagnostic_severity::error) {
                return item.code;
            }
        }
        return std::string{};
    };

    SECTION("empty and truncated bytes") {
        CHECK(codeOfError(writer::read(std::span<std::byte const>{}).diagnostics()) == writer::diag::unreadable);
        for (const auto length : {std::size_t{1}, std::size_t{4}, std::size_t{8}, written.bytes.size() / 2, written.bytes.size() * 3 / 4}) {
            INFO("length " << length);
            const auto result = writer::read(std::span<std::byte const>{written.bytes.data(), length});
            CHECK_FALSE(result);
            CHECK_FALSE(result.has_value());
            CHECK(codeOfError(result.diagnostics()).rfind("io.pcfb.", 0) == 0);
        }
    }

    SECTION("corrupted bytes never crash and fail with a pcfb id when they fail") {
        std::mt19937_64 rng(7);
        std::size_t rejected = 0;
        for (int round = 0; round < 400; ++round) {
            auto bytes = written.bytes;
            const auto flips = 1 + round % 4;
            for (int flip = 0; flip < flips; ++flip) {
                bytes[static_cast<std::size_t>(pcfb_test_random::uniform(rng, 0, bytes.size() - 1))] ^= static_cast<std::byte>(1 + rng() % 255);
            }
            const auto result = writer::read(bytes);
            if (!result) {
                ++rejected;
                CHECK(codeOfError(result.diagnostics()).rfind("io.pcfb.", 0) == 0);
            } else {
                CHECK(result.value().circuit.valid());
            }
        }
        CHECK(rejected > 0);
    }

    SECTION("a wrong file identifier") {
        auto bytes = written.bytes;
        bytes[4] = std::byte{'X'};
        CHECK_FALSE(writer::read(bytes));
    }

    SECTION("a query kind this reader does not know") {
        auto model = unpack(written.bytes);
        model->subcircuits[0]->queries[1]->kind = static_cast<::pcfb::QueryKind>(9);
        model->schema_version = 0x00010002U;
        const auto result = writer::read(pack(*model));
        CHECK_FALSE(result);
        CHECK_FALSE(result.has_value());
        CHECK(codeOfError(result.diagnostics()) == writer::diag::unknown_query_kind);
    }

    SECTION("a hull query in a file that declares minor 1") {
        auto hull = threeTargetNames();
        hull.query = kind::monotone_hull_probability;
        auto model = unpack(requireWrite(threeTargetModel(), hull).bytes);
        model->schema_version = 0x00010001U;
        const auto result = writer::read(pack(*model));
        CHECK_FALSE(result);
        CHECK(codeOfError(result.diagnostics()) == writer::diag::unknown_query_kind);
    }

    SECTION("externally supplied inputs") {
        auto model = unpack(written.bytes);
        model->subcircuits[0]->inputs = {0};
        const auto result = writer::read(pack(*model));
        CHECK_FALSE(result);
        CHECK_FALSE(result.has_value());
        const auto code = codeOfError(result.diagnostics());
        CHECK((code == writer::diag::external_inputs || code == writer::diag::invalid_file));
    }

    SECTION("a reserved operator") {
        auto model = unpack(written.bytes);
        model->subcircuits[0]->operators[3]->opcode = ::pcfb::OpCode::BITWISE_EVENS;
        const auto result = writer::read(pack(*model));
        CHECK_FALSE(result);
        CHECK_FALSE(result.has_value());
        CHECK(codeOfError(result.diagnostics()) == writer::diag::invalid_file);
    }

    SECTION("a major version other than 1") {
        auto model = unpack(written.bytes);
        model->schema_version = 0x00020001U;
        CHECK_FALSE(writer::read(pack(*model)));
    }

    SECTION("an operator listed before an operator whose tensor it reads") {
        auto model = unpack(written.bytes);
        std::ranges::reverse(model->subcircuits[0]->operators);
        const auto result = writer::read(pack(*model));
        INFO(diagnosticsText(result.diagnostics()));
        CHECK_FALSE(result);
        CHECK(codeOfError(result.diagnostics()) == writer::diag::invalid_file);
    }

    SECTION("a second subcircuit") {
        auto model = unpack(written.bytes);
        model->subcircuits.push_back(std::make_unique<::pcfb::SubCircuitT>(*model->subcircuits[0]));
        const auto result = writer::read(pack(*model));
        CHECK_FALSE(result);
        CHECK(codeOfError(result.diagnostics()) == writer::diag::invalid_file);
    }
}

TEST_CASE("pcfb reader: metadata the reader interprets is checked", "[io][pcfb][read][negative]") {
    auto options = threeTargetNames();
    options.model_name = "demo";
    options.target_kinds = {{pc::target_id(0), {kind::probability}}};
    options.quantification = {{pc::target_id(0), writer::target_quantification{.literal_mode = "omega_l", .max_initiators_per_cutset = 2}}};
    options.source_sha256 = std::array<std::byte, 32>{};
    auto model = unpack(requireWrite(threeTargetModel(), options).bytes);
    for (auto &entry : model->metadata) {
        if (entry->key == "compile.query.0.max_initiators_per_cutset") {
            ::pcfb::IntValueT value;
            value.value = -3;
            entry->value.Set(value);
        } else if (entry->key == "general.source_sha256") {
            ::pcfb::BytesValueT value;
            value.value = {1, 2, 3};
            entry->value.Set(value);
        } else if (entry->key == "general.model_name") {
            ::pcfb::IntValueT value;
            value.value = 5;
            entry->value.Set(value);
        }
    }
    auto result = writer::read(pack(*model));
    INFO(diagnosticsText(result.diagnostics()));
    REQUIRE(result);
    CHECK(codesOf(result.diagnostics()) == std::set<std::string>{std::string(writer::diag::invalid_metadata)});
    CHECK(result.diagnostics().size() == 3);
    auto const &read = result.value();
    CHECK_FALSE(read.options.source_sha256.has_value());
    CHECK_FALSE(read.options.quantification.at(pc::target_id(0)).max_initiators_per_cutset.has_value());
    CHECK(read.options.quantification.at(pc::target_id(0)).literal_mode == "omega_l");
    // Circuit.name is the fallback of general.model_name.
    CHECK(read.options.model_name == "demo");
    // The entries stay as they were.
    CHECK(std::ranges::any_of(read.metadata, [](auto const &entry) { return entry.first == "general.model_name" && std::holds_alternative<std::int64_t>(entry.second); }));
}

TEST_CASE("pcfb reader: a watch or projection-anchor target keeps its role", "[io][pcfb][read]") {
    pc::builder builder;
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
    const auto b = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto both = builder.add_and({pc::edge_ref{.source = a}, pc::edge_ref{.source = b}});
    (void)builder.add_target(both);
    (void)builder.add_target(a, pc::observable_role::watch);
    (void)builder.add_target(b, pc::observable_role::projection_anchor);
    (void)builder.add_target(both, pc::observable_role::watch);
    const auto model = sealed(std::move(builder));
    auto options = writer::write_options{};
    options.target_kinds = {{pc::target_id(0), {kind::probability, kind::monotone_hull_probability}},
                            {pc::target_id(3), {kind::monotone_hull_probability}}};
    const auto written = requireWrite(model, options);

    auto file = ::pcfb::io::from_bytes(written.bytes);
    REQUIRE(file);
    std::map<std::string, std::string> roles;
    for (auto const *entry : *(*file)->metadata()) {
        if (entry->key()->str().ends_with(".role")) {
            roles[entry->key()->str()] = entry->value_as_StringValue()->value()->str();
        }
    }
    // One entry per query of a non-primary target, none for the primary one.
    CHECK(roles == std::map<std::string, std::string>{{"compile.query.2.role", "watch"},
                                                     {"compile.query.3.role", "projection_anchor"},
                                                     {"compile.query.4.role", "watch"}});

    const auto read = readOk(written.bytes);
    REQUIRE(read.circuit.targets().size() == 4);
    CHECK(read.circuit.targets()[0].role == pc::observable_role::primary);
    CHECK(read.circuit.targets()[1].role == pc::observable_role::watch);
    CHECK(read.circuit.targets()[2].role == pc::observable_role::projection_anchor);
    CHECK(read.circuit.targets()[3].role == pc::observable_role::watch);
    requireStable(read);
}

TEST_CASE("pcfb reader: files with only primary targets carry no role metadata", "[io][pcfb][read]") {
    const auto written = requireWrite(threeTargetModel());
    auto file = ::pcfb::io::from_bytes(written.bytes);
    REQUIRE(file);
    for (auto const *entry : *(*file)->metadata()) {
        CHECK_FALSE(entry->key()->str().ends_with(".role"));
    }
}

TEST_CASE("pcfb reader: an unknown role value reads as primary with a warning", "[io][pcfb][read][negative]") {
    pc::builder builder;
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
    (void)builder.add_target(a, pc::observable_role::watch);
    auto model = unpack(requireWrite(sealed(std::move(builder))).bytes);
    for (auto &entry : model->metadata) {
        if (entry->key == "compile.query.0.role") {
            ::pcfb::StringValueT value;
            value.value = "decoration";
            entry->value.Set(value);
        }
    }
    auto result = writer::read(pack(*model));
    INFO(diagnosticsText(result.diagnostics()));
    REQUIRE(result);
    CHECK(codesOf(result.diagnostics()) == std::set<std::string>{std::string(writer::diag::invalid_metadata)});
    CHECK(result.value().circuit.targets()[0].role == pc::observable_role::primary);
}

TEST_CASE("pcfb reader: FTP --watch compiles to a file that reads back with the watch role", "[io][pcfb][read][ftp]") {
    const std::string text = "TOP + G1 E3\nG1 * E1 E2\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 E1\n  2.0E-01 E2\n  3.0E-01 E3\n";
    ftp::options settings;
    settings.negation = "boolean";
    settings.watch = {"G1"};
    const std::vector<fuzzybits::io::backend::fuzzybits::ftp::source_text> sources{{.path = "model.ftp", .text = text}};
    const auto compiled = ftp::compile_sources(sources, settings);
    INFO(diagnosticsText(compiled.diagnostics()));
    REQUIRE(compiled);
    const auto read = readOk(compiled.value().bytes);
    REQUIRE(read.circuit.targets().size() == 2);
    CHECK(read.circuit.targets()[0].role == pc::observable_role::primary);
    CHECK(read.circuit.targets()[1].role == pc::observable_role::watch);
    CHECK(read.options.target_names.at(read.circuit.targets()[1].id) == "G1");
    requireStable(read);
}

TEST_CASE("pcfb reader: cardinality(0, n) is written as ATLEAST 0 and read back with its operands", "[io][pcfb][read]") {
    pc::builder builder;
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
    const auto b = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto any = builder.add_cardinality(0, 2, {pc::edge_ref{.source = a}, pc::edge_ref{.source = b, .complement = true}});
    (void)builder.add_target(any);
    const auto model = sealed(std::move(builder));
    const auto written = requireWrite(model);
    auto file = ::pcfb::io::from_bytes(written.bytes);
    REQUIRE(file);
    auto const *last = (*file)->subcircuits()->Get(0)->operators()->Get(written.tensor_by_node[any.raw()]);
    CHECK(last->opcode() == ::pcfb::OpCode::BITWISE_ATLEAST);
    CHECK(last->args_as_KofNArgs()->k() == 0);

    const auto read = readOk(written.bytes);
    const auto node = read.circuit.targets()[0].node;
    CHECK(read.circuit.op_codes()[node.raw()] == pc::op_code::cardinality);
    CHECK(read.circuit.cardinality_low()[node.raw()] == 0);
    CHECK(read.circuit.cardinality_high()[node.raw()] == 2);
    CHECK(read.circuit.operands(node).size() == 2);
    requireStable(read);
}

namespace {

    // A NOT gate of the circuit in front of operators that list its source in both polarities, which the
    // writer reads through its own NOT tensors: three sources, the gate first, then operators on x (two
    // of them share the writer's tensor) and one on z.
    struct mixed_polarity_model final {
        pc::circuit circuit;
        pc::bool_node_id x;
        pc::bool_node_id gate;
    };

    [[nodiscard]] auto mixedPolarityModel() -> mixed_polarity_model {
        pc::builder builder;
        const auto x = builder.add_basic_event(builder.add_literal_probability(0.1));
        const auto y = builder.add_basic_event(builder.add_literal_probability(0.2));
        const auto z = builder.add_basic_event(builder.add_literal_probability(0.3));
        const auto gate = builder.add_not(pc::edge_ref{.source = x});
        const auto beside = builder.add_and({pc::edge_ref{.source = x}, pc::edge_ref{.source = gate}});
        const auto first = builder.add_and({pc::edge_ref{.source = x}, pc::edge_ref{.source = x, .complement = true}, pc::edge_ref{.source = y}});
        const auto second = builder.add_or({pc::edge_ref{.source = x, .complement = true}, pc::edge_ref{.source = x}, pc::edge_ref{.source = y}});
        const auto third = builder.add_and({pc::edge_ref{.source = z}, pc::edge_ref{.source = z, .complement = true}});
        for (const auto node : {beside, first, second, third}) {
            (void)builder.add_target(node);
        }
        return {sealed(std::move(builder)), x, gate};
    }

    [[nodiscard]] auto complementTensorsOf(writer::read_output const &read) -> std::optional<std::vector<std::byte>> {
        for (auto const &[key, value] : read.metadata) {
            if (key == "compile.complement_tensors") {
                return std::get<std::vector<std::byte>>(value);
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] auto withComplementTensors(bytes_type const &bytes, std::vector<std::uint8_t> const &packed) -> bytes_type {
        auto model = unpack(bytes);
        for (auto &entry : model->metadata) {
            if (entry->key == "compile.complement_tensors") {
                ::pcfb::BytesValueT value;
                value.value = packed;
                entry->value.Set(value);
            }
        }
        return pack(*model);
    }

    [[nodiscard]] auto withoutComplementTensors(bytes_type const &bytes) -> bytes_type {
        auto model = unpack(bytes);
        std::erase_if(model->metadata, [](auto const &entry) { return entry->key == "compile.complement_tensors"; });
        return pack(*model);
    }

    [[nodiscard]] auto nodeCountOf(std::span<std::byte const> bytes) -> std::pair<std::size_t, std::set<std::string>> {
        const auto read = writer::read(bytes);
        REQUIRE(read);
        return {read.value().circuit.node_count(), codesOf(read.diagnostics())};
    }

} // namespace

TEST_CASE("pcfb reader: a NOT gate beside the writer's NOT tensors stays a node and the tensors fold", "[io][pcfb][read]") {
    const auto model = mixedPolarityModel();
    REQUIRE(model.circuit.node_count() == 8);
    const auto written = requireWrite(model.circuit);
    const auto read = readOk(written.bytes);
    // Every node of the circuit maps, and the writer's tensors are complemented edges, not nodes.
    requireSameStructure(RandomModel{model.circuit, {}}, written, read);
    CHECK(read.circuit.node_count() == model.circuit.node_count());
    const auto gate = read.node_by_tensor.at(written.tensor_by_node.at(model.gate.raw()));
    REQUIRE(gate.valid());
    CHECK(read.circuit.op_codes()[gate.raw()] == pc::op_code::not_gate);
    CHECK(read.circuit.operands(gate)[0].source == read.node_by_tensor.at(written.tensor_by_node.at(model.x.raw())));
    // One tensor for x shared by two operators, one for z.
    const auto listed = complementTensorsOf(read);
    REQUIRE(listed.has_value());
    CHECK(listed->size() == 8);
    requireStable(read);
}

TEST_CASE("pcfb reader: without compile.complement_tensors every NOT operator is a node", "[io][pcfb][read]") {
    const auto model = mixedPolarityModel();
    const auto written = requireWrite(model.circuit);
    const auto bytes = withoutComplementTensors(written.bytes);
    const auto [nodes, codes] = nodeCountOf(bytes);
    // The two tensors become NOT gates; the file is otherwise read as before.
    CHECK(nodes == model.circuit.node_count() + 2);
    CHECK(codes.empty());
}

TEST_CASE("pcfb reader: a writer file without mixed polarity has no compile.complement_tensors entry", "[io][pcfb][read]") {
    pc::builder builder;
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
    const auto b = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto gate = builder.add_not(pc::edge_ref{.source = a});
    (void)builder.add_target(builder.add_and({pc::edge_ref{.source = gate}, pc::edge_ref{.source = b, .complement = true}}));
    const auto written = requireWrite(sealed(std::move(builder)));
    const auto read = readOk(written.bytes);
    CHECK_FALSE(complementTensorsOf(read).has_value());
    requireStable(read);
}

TEST_CASE("pcfb reader: compile.complement_tensors is checked", "[io][pcfb][read][negative]") {
    const auto model = mixedPolarityModel();
    const auto written = requireWrite(model.circuit);
    const auto expected = model.circuit.node_count();
    const auto invalid = std::set<std::string>{std::string(writer::diag::invalid_metadata)};
    const auto listed = *complementTensorsOf(readOk(written.bytes));
    REQUIRE(listed.size() == 8);
    std::vector<std::uint8_t> tensors;
    for (const auto byte : listed) {
        tensors.push_back(std::to_integer<std::uint8_t>(byte));
    }
    // The tensor of the event y, which is not the source of a NOT tensor.
    const auto ofEvent = written.tensor_by_node.at(1);

    SECTION("a length that is not a multiple of four") {
        auto packed = tensors;
        packed.pop_back();
        const auto [nodes, codes] = nodeCountOf(withComplementTensors(written.bytes, packed));
        CHECK(codes == invalid);
        CHECK(nodes == expected + 2);
    }

    SECTION("ids that are not strictly ascending") {
        auto packed = tensors;
        std::rotate(packed.begin(), packed.begin() + 4, packed.end());
        const auto [nodes, codes] = nodeCountOf(withComplementTensors(written.bytes, packed));
        CHECK(codes == invalid);
        CHECK(nodes == expected + 2);
    }

    SECTION("an id out of range") {
        auto packed = tensors;
        packed[7] = 0x7F;
        const auto [nodes, codes] = nodeCountOf(withComplementTensors(written.bytes, packed));
        CHECK(codes == invalid);
        CHECK(nodes == expected + 2);
    }

    SECTION("a tensor that is not a NOT operator") {
        std::vector<std::uint32_t> ids{ofEvent};
        for (std::size_t at = 0; at < tensors.size(); at += 4) {
            ids.push_back(static_cast<std::uint32_t>(tensors[at]) | static_cast<std::uint32_t>(tensors[at + 1]) << 8U);
        }
        std::ranges::sort(ids);
        std::vector<std::uint8_t> packed;
        for (const auto id : ids) {
            for (unsigned shift = 0; shift < 32; shift += 8) {
                packed.push_back(static_cast<std::uint8_t>(id >> shift));
            }
        }
        const auto result = writer::read(withComplementTensors(written.bytes, packed));
        INFO(diagnosticsText(result.diagnostics()));
        REQUIRE(result);
        // Only the event is refused; the writer's two tensors still fold.
        CHECK(codesOf(result.diagnostics()) == invalid);
        CHECK(result.diagnostics().size() == 1);
        CHECK(result.value().circuit.node_count() == expected);
    }
}
