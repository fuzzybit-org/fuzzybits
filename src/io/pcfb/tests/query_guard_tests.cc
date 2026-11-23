#include <catch2/catch_test_macros.hpp>

#include "query_guard_files.h"

#include "compile/exact_eval.h"
#include "io/pcfb/query_guard.h"
#include "pcfb/io.h"
#include "pcfb/validate.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

namespace guard = fuzzybits::io::pcfb_guard;
namespace files = query_guard_files;

struct loaded final {
    std::vector<std::byte> bytes;
    pcfb::ir::Circuit circuit;
};

[[nodiscard]] auto load(std::vector<std::byte> bytes) -> loaded {
    auto circuit = pcfb::io::from_bytes(bytes);
    REQUIRE(circuit);
    return loaded{std::move(bytes), std::move(*circuit)};
}

[[nodiscard]] auto contains(std::string const &text, std::string const &part) -> bool {
    return text.find(part) != std::string::npos;
}

} // namespace

TEST_CASE("the guard accepts Boolean queries whatever the schema minor", "[io][pcfb][guard]") {
    CHECK_FALSE(guard::reject_unimplemented_queries(load(files::boolean_only()).circuit));
    CHECK_FALSE(guard::reject_unimplemented_queries(load(files::boolean_with_minor(2)).circuit));
    // Readers accept any minor of major 1 (pcfb spec 1.1 section 2).
    CHECK_FALSE(guard::reject_unimplemented_queries(load(files::boolean_with_minor(3)).circuit));
    CHECK_FALSE(guard::reject_unimplemented_queries(load(files::boolean_with_minor(9)).circuit));
}

TEST_CASE("the guard rejects a monotone-hull query and names it", "[io][pcfb][guard]") {
    const auto hull = load(files::hull_only());
    REQUIRE(pcfb::validate(hull.circuit).empty());
    const auto rejected = guard::reject_unimplemented_queries(hull.circuit);
    REQUIRE(rejected);
    CHECK(rejected->id == "pcfb.unsupported-query-kind");
    CHECK(contains(rejected->message, files::hull_name));
    CHECK(contains(rejected->message, "MONOTONE_HULL_PROBABILITY"));
}

TEST_CASE("the guard looks past a leading Boolean query", "[io][pcfb][guard]") {
    const auto mixed = load(files::mixed());
    REQUIRE(pcfb::validate(mixed.circuit).empty());
    const auto rejected = guard::reject_unimplemented_queries(mixed.circuit);
    REQUIRE(rejected);
    CHECK(rejected->id == "pcfb.unsupported-query-kind");
    CHECK(contains(rejected->message, files::hull_name));
    CHECK_FALSE(contains(rejected->message, std::string("'") + files::boolean_name + "'"));
}

TEST_CASE("the guard rejects a query kind newer than the pin, whatever minor the file declares", "[io][pcfb][guard]") {
    const auto unknown = load(files::repacked(files::boolean_only(), [](pcfb::CircuitT &model) {
        model.schema_version = (1U << 16U) | 9U;
        model.subcircuits[0]->queries[0]->kind = static_cast<pcfb::QueryKind>(200);
    }));
    const auto diagnostics = pcfb::validate(unknown.circuit);
    REQUIRE_FALSE(diagnostics.empty());
    CHECK(diagnostics.front().code == pcfb::Diagnostic::Code::UnknownQueryKind);
    const auto rejected = guard::reject_unimplemented_queries(unknown.circuit);
    REQUIRE(rejected);
    CHECK(rejected->id == "pcfb.unsupported-query-kind");
    CHECK(contains(rejected->message, files::boolean_name));
    CHECK(contains(rejected->message, "unknown(200)"));
}

TEST_CASE("exact evaluation answers Boolean queries of any minor and refuses the others", "[io][pcfb][guard][exact]") {
    const auto boolean = load(files::boolean_only());
    CHECK(std::abs(fuzzybits::compile::exact_probability(boolean.circuit) - 0.125) < 1e-15);

    for (auto bytes : {files::hull_only(), files::mixed()}) {
        const auto file = load(std::move(bytes));
        for (std::uint32_t query = 0; query < 2; ++query) {
            try {
                static_cast<void>(fuzzybits::compile::exact_probability(file.circuit, query));
                FAIL("exact_probability accepted a monotone-hull query");
            } catch (fuzzybits::compile::ExactEvalError const &error) {
                const std::string message = error.what();
                CHECK(contains(message, "pcfb.unsupported-query-kind"));
                CHECK(contains(message, files::hull_name));
            }
        }
    }
    const auto future = load(files::boolean_with_minor(3));
    CHECK(std::abs(fuzzybits::compile::exact_probability(future.circuit) - 0.125) < 1e-15);
}

TEST_CASE("the committed hull fixture is a valid 1.2 file with one hull query", "[io][pcfb][guard]") {
    const auto path =
        std::filesystem::path(PROJECT_SOURCE_DIR) / "tests" / "python" / "data" / "monotone_hull_query.pcfb";
    auto circuit = pcfb::io::read(path);
    REQUIRE(circuit);
    CHECK(pcfb::validate(*circuit).empty());
    CHECK(circuit->get()->schema_version() == pcfb::kSchemaVersionV12);
    const auto *queries = circuit->subcircuits()->Get(0)->queries();
    REQUIRE(queries->size() == 1);
    CHECK(queries->Get(0)->kind() == pcfb::QueryKind::MONOTONE_HULL_PROBABILITY);
    CHECK(queries->Get(0)->name()->str() == files::hull_name);
    const auto rejected = guard::reject_unimplemented_queries(*circuit);
    REQUIRE(rejected);
    CHECK(rejected->id == "pcfb.unsupported-query-kind");
}

// Regenerates tests/python/data/monotone_hull_query.pcfb:
//   FZB_HULL_FIXTURE_OUT=<path> fuzzybits_io_pcfb_guard_tests "[.write-fixture]"
TEST_CASE("write the hull fixture", "[.write-fixture]") {
    const char *out = std::getenv("FZB_HULL_FIXTURE_OUT");
    REQUIRE(out != nullptr);
    files::write(out, files::hull_only());
}
