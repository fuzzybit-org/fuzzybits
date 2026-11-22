#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include "io/descriptor.h"
#include "io/formats.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"

#include "pcfb.h"
#include "pcfb/io.h"
#include "pcfb/validate.h"
#include "pcfb_generated.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

    using namespace pcfb_writer_test;
    namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;

    using edge = pc::edge_ref;
    using kind = writer::query_kind;

    // Three targets over events a, b, c: a AND b, b OR c, and the event a.
    [[nodiscard]] auto buildModel() -> pc::circuit {
        pc::builder builder;
        const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
        const auto b = builder.add_basic_event(builder.add_literal_probability(0.2));
        const auto c = builder.add_basic_event(builder.add_literal_probability(0.3));
        const auto both = builder.add_and({edge{.source = a}, edge{.source = b}});
        const auto either = builder.add_or({edge{.source = b}, edge{.source = c}});
        (void)builder.add_target(both);
        (void)builder.add_target(either);
        (void)builder.add_target(a);
        auto sealed = std::move(builder).seal();
        INFO(diagnosticsText(sealed.diagnostics));
        REQUIRE(sealed.ok());
        return std::move(sealed.output);
    }

    struct written_file final {
        std::vector<std::byte> bytes;
        ::pcfb::ir::Circuit circuit;
        std::vector<std::uint32_t> tensor_by_node;
    };

    [[nodiscard]] auto writeAndRead(pc::circuit const &model, writer::write_options const &options) -> written_file {
        auto written = writer::write(model.to_handle(), options);
        INFO(diagnosticsText(written.diagnostics()));
        REQUIRE(written);
        auto bytes = std::move(written.value().bytes);
        auto circuit = ::pcfb::io::from_bytes(bytes);
        REQUIRE(circuit);
        written_file file{.bytes = std::move(bytes), .circuit = *circuit, .tensor_by_node = std::move(written.value().tensor_by_node)};
        const auto diagnostics = ::pcfb::validate(file.circuit);
        INFO("pcfb::validate reported " << diagnostics.size() << " finding(s)");
        REQUIRE(diagnostics.empty());
        return file;
    }

    struct query_row final {
        std::string name;
        ::pcfb::QueryKind kind;
        std::uint32_t target;
        friend auto operator==(query_row const &, query_row const &) -> bool = default;
    };

    [[nodiscard]] auto queriesOf(::pcfb::ir::Circuit const &circuit) -> std::vector<query_row> {
        std::vector<query_row> rows;
        for (auto const *query : *circuit.subcircuits()->Get(0)->queries()) {
            rows.push_back({query->name()->str(), query->kind(), query->target()});
        }
        return rows;
    }

    [[nodiscard]] auto minorOf(::pcfb::ir::Circuit const &circuit) -> std::uint32_t {
        return circuit->schema_version() & 0xFFFFU;
    }

    [[nodiscard]] auto meta(::pcfb::ir::Circuit const &circuit, std::string const &key) -> ::pcfb::KeyValue const * {
        return circuit->metadata() == nullptr ? nullptr : circuit->metadata()->LookupByKey(key.c_str());
    }

    [[nodiscard]] auto stringMeta(::pcfb::ir::Circuit const &circuit, std::string const &key) -> std::optional<std::string> {
        auto const *entry = meta(circuit, key);
        if (entry == nullptr || entry->value_type() != ::pcfb::MetadataValue::StringValue) {
            return std::nullopt;
        }
        return entry->value_as_StringValue()->value()->str();
    }

    [[nodiscard]] auto intMeta(::pcfb::ir::Circuit const &circuit, std::string const &key) -> std::optional<std::int64_t> {
        auto const *entry = meta(circuit, key);
        if (entry == nullptr || entry->value_type() != ::pcfb::MetadataValue::IntValue) {
            return std::nullopt;
        }
        return entry->value_as_IntValue()->value();
    }

    [[nodiscard]] auto namesOf(pc::circuit const &model) -> writer::write_options {
        writer::write_options options;
        options.target_names = {{pc::target_id(0), "AND"}, {pc::target_id(1), "OR"}, {pc::target_id(2), "A"}};
        return options;
    }

} // namespace

TEST_CASE("pcfb writer: Boolean-only query sets stay at schema minor 1", "[io][pcfb][kind]") {
    const auto model = buildModel();
    auto const options = namesOf(model);
    const auto file = writeAndRead(model, options);
    CHECK(file.circuit->schema_version() == ::pcfb::kSchemaVersionV11);
    CHECK(queriesOf(file.circuit) == std::vector<query_row>{{"AND", ::pcfb::QueryKind::PROBABILITY, file.tensor_by_node[3]},
                                                            {"OR", ::pcfb::QueryKind::PROBABILITY, file.tensor_by_node[4]},
                                                            {"A", ::pcfb::QueryKind::PROBABILITY, file.tensor_by_node[0]}});

    // An explicit probability kind is the same file.
    auto explicitKinds = namesOf(model);
    explicitKinds.target_kinds = {{pc::target_id(0), {kind::probability}}, {pc::target_id(2), {kind::probability}}};
    CHECK(writeAndRead(model, explicitKinds).bytes == file.bytes);
}

TEST_CASE("pcfb writer: a monotone-hull target makes the file schema minor 2 and keeps names and kinds", "[io][pcfb][kind]") {
    const auto model = buildModel();
    auto options = namesOf(model);
    options.target_kinds = {{pc::target_id(1), {kind::monotone_hull_probability}}};
    const auto file = writeAndRead(model, options);
    CHECK(file.circuit->schema_version() == ::pcfb::kSchemaVersionV12);
    CHECK(minorOf(file.circuit) == 2);
    CHECK(queriesOf(file.circuit) == std::vector<query_row>{{"AND", ::pcfb::QueryKind::PROBABILITY, file.tensor_by_node[3]},
                                                            {"OR", ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, file.tensor_by_node[4]},
                                                            {"A", ::pcfb::QueryKind::PROBABILITY, file.tensor_by_node[0]}});
    // pcfb 1.2 adds no operators.
    CHECK(stringMeta(file.circuit, "pcfb.opset") == "core-1.1");
}

TEST_CASE("pcfb writer: options.query is the default for targets without an explicit kind", "[io][pcfb][kind]") {
    const auto model = buildModel();
    auto options = namesOf(model);
    options.query = kind::monotone_hull_probability;
    options.target_kinds = {{pc::target_id(0), {kind::probability}}};
    const auto file = writeAndRead(model, options);
    CHECK(minorOf(file.circuit) == 2);
    CHECK(queriesOf(file.circuit) == std::vector<query_row>{{"AND", ::pcfb::QueryKind::PROBABILITY, file.tensor_by_node[3]},
                                                            {"OR", ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, file.tensor_by_node[4]},
                                                            {"A", ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, file.tensor_by_node[0]}});

    // Overriding every target back to probability returns to minor 1 whatever the default says.
    options.target_kinds = {{pc::target_id(0), {kind::probability}}, {pc::target_id(1), {kind::probability}}, {pc::target_id(2), {kind::probability}}};
    CHECK(minorOf(writeAndRead(model, options).circuit) == 1);
}

TEST_CASE("pcfb writer: several kinds on one target make several queries on one tensor", "[io][pcfb][kind]") {
    const auto model = buildModel();
    auto options = namesOf(model);
    options.target_kinds = {{pc::target_id(0), {kind::probability, kind::monotone_hull_probability}}};
    const auto file = writeAndRead(model, options);
    CHECK(minorOf(file.circuit) == 2);
    CHECK(queriesOf(file.circuit) == std::vector<query_row>{{"AND", ::pcfb::QueryKind::PROBABILITY, file.tensor_by_node[3]},
                                                            {"AND", ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, file.tensor_by_node[3]},
                                                            {"OR", ::pcfb::QueryKind::PROBABILITY, file.tensor_by_node[4]},
                                                            {"A", ::pcfb::QueryKind::PROBABILITY, file.tensor_by_node[0]}});
}

TEST_CASE("pcfb writer: io.pcfb.query_kind selects the default kind of the stream writer", "[io][pcfb][kind]") {
    const auto model = buildModel();
    const auto written = [&](std::string const &value) {
        fuzzybits::io::write_request request{.format = fuzzybits::io::formats::pcfb()};
        request.options.set(writer::query_kind_option(), value);
        std::ostringstream stream;
        const auto result = writer::write(model.to_handle(), stream, request);
        INFO(diagnosticsText(result.diagnostics()));
        return std::pair{static_cast<bool>(result), stream.str()};
    };
    const auto parse = [](std::string const &text) {
        std::vector<std::byte> bytes(text.size());
        for (std::size_t i = 0; i < text.size(); ++i) {
            bytes[i] = static_cast<std::byte>(text[i]);
        }
        return bytes;
    };

    const auto hull = written("monotone_hull_probability");
    REQUIRE(hull.first);
    const auto hullBytes = parse(hull.second);
    auto hullCircuit = ::pcfb::io::from_bytes(hullBytes);
    REQUIRE(hullCircuit);
    CHECK(minorOf(*hullCircuit) == 2);
    for (auto const &row : queriesOf(*hullCircuit)) {
        CHECK(row.kind == ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY);
    }

    const auto boolean = written("probability");
    REQUIRE(boolean.first);
    const auto booleanBytes = parse(boolean.second);
    auto booleanCircuit = ::pcfb::io::from_bytes(booleanBytes);
    REQUIRE(booleanCircuit);
    CHECK(minorOf(*booleanCircuit) == 1);

    CHECK_FALSE(written("monotone_hull").first);
}

TEST_CASE("pcfb writer: per-target quantification is recorded next to the queries", "[io][pcfb][kind]") {
    const auto model = buildModel();
    auto options = namesOf(model);
    options.target_kinds = {{pc::target_id(0), {kind::probability, kind::monotone_hull_probability}},
                            {pc::target_id(1), {kind::monotone_hull_probability}}};
    options.quantification = {
        {pc::target_id(0), writer::target_quantification{.literal_mode = "atom_split", .max_initiators_per_cutset = 1}},
        {pc::target_id(1), writer::target_quantification{.literal_mode = "omega_l", .max_initiators_per_cutset = std::nullopt}},
    };
    const auto file = writeAndRead(model, options);
    REQUIRE(queriesOf(file.circuit).size() == 4);

    // Query 0 and 1 belong to target 0 (Boolean, then hull), query 2 to target 1, query 3 to target 2.
    CHECK(stringMeta(file.circuit, "fuzzybits.query.0.negation") == "boolean");
    CHECK(stringMeta(file.circuit, "fuzzybits.query.1.negation") == "monotone_hull");
    CHECK(stringMeta(file.circuit, "fuzzybits.query.2.negation") == "monotone_hull");
    for (const auto index : {0, 1}) {
        const auto prefix = "fuzzybits.query." + std::to_string(index) + ".";
        CHECK(stringMeta(file.circuit, prefix + "literal_mode") == "atom_split");
        CHECK(intMeta(file.circuit, prefix + "max_initiators_per_cutset") == 1);
    }
    CHECK(stringMeta(file.circuit, "fuzzybits.query.2.literal_mode") == "omega_l");
    CHECK_FALSE(meta(file.circuit, "fuzzybits.query.2.max_initiators_per_cutset"));
    // A target without an entry records nothing.
    CHECK_FALSE(meta(file.circuit, "fuzzybits.query.3.negation"));

    // No quantification entries, no keys: the metadata of a plain file is unchanged.
    const auto plain = writeAndRead(model, namesOf(model));
    for (const auto *entry : *plain.circuit->metadata()) {
        CHECK(entry->key()->str().rfind("fuzzybits.query.", 0) == std::string::npos);
    }
}

TEST_CASE("pcfb writer: invalid query kinds and quantification entries are diagnosed", "[io][pcfb][kind]") {
    const auto model = buildModel();
    const auto codeOf = [&](writer::write_options const &options) { return hasCode(writer::write(model.to_handle(), options).diagnostics(), "io.pcfb.invalid-query-kind"); };

    auto unknown = namesOf(model);
    unknown.target_kinds = {{pc::target_id(9), {kind::probability}}};
    CHECK(codeOf(unknown));

    auto empty = namesOf(model);
    empty.target_kinds = {{pc::target_id(0), {}}};
    CHECK(codeOf(empty));

    auto repeated = namesOf(model);
    repeated.target_kinds = {{pc::target_id(0), {kind::monotone_hull_probability, kind::monotone_hull_probability}}};
    CHECK(codeOf(repeated));

    auto orphan = namesOf(model);
    orphan.quantification = {{pc::target_id(9), writer::target_quantification{.literal_mode = "atom_split"}}};
    CHECK(hasCode(writer::write(model.to_handle(), orphan).diagnostics(), "io.pcfb.invalid-quantification"));
}
