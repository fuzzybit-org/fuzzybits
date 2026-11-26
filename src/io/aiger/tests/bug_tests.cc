// Known bugs of the AIGER reader. Each case states the AIGER rule or ir::pc contract it checks and fails
// on develop until its issue is fixed; the fixtures are in input/bugs/aiger/. Cases tagged [crash]
// terminate the process today and run as their own ctest entry. This executable is labelled `bug`.
//
// AIGER rules cited from the AIGER 1.9 format description (aiger/FORMAT): the ASCII AND gates "can be
// in any order"; the symbol table holds `i<pos>`, `l<pos>` and `o<pos>` entries with `pos` below the
// count of that kind and at most one symbol per position; the comment section starts with a line `c`.

#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "io/aiger/descriptor.h"
#include "ir/pc/circuit.h"
#include "ir/pc/probability.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

namespace {

namespace aiger = fuzzybits::io::backend::fuzzybits::aiger;

[[nodiscard]] auto diagnosticsText(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
    std::ostringstream output;
    for (auto const &diagnostic : diagnostics.items()) {
        output << diagnostic.code << ": " << diagnostic.message << '\n';
    }
    return output.str();
}

[[nodiscard]] auto readText(std::string text) -> aiger::aiger_read_result {
    std::istringstream input(std::move(text));
    return aiger::read(input);
}

[[nodiscard]] auto readFixture(std::string_view name) -> aiger::aiger_read_result {
    const auto path = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "bugs" / "aiger" / name;
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input);
    return aiger::read(input);
}

[[nodiscard]] auto errorCount(fuzzybits::api::diagnostic_list const &diagnostics) -> std::size_t {
    std::size_t count = 0;
    for (auto const &diagnostic : diagnostics.items()) {
        count += diagnostic.severity == fuzzybits::api::diagnostic_severity::error ? 1 : 0;
    }
    return count;
}

} // namespace

// AND gates of an ASCII file may come in any order. The builder resolves operands by recursion
// (buildVariable -> buildLiteralEdge -> buildVariable), one stack frame pair per gate of the longest
// dependency chain, so a 10^6-gate chain listed from the output down overflows the stack.
TEST_CASE("AIGER reader imports a 10^6-gate ASCII chain listed top-down", "[bug][aiger][crash][issue-292]") {
    constexpr std::uint32_t gates = 1'000'000;
    std::string text;
    text.reserve(std::size_t{gates} * 24);
    // M = gates + 1, one input (variable 1), one output (the last AND).
    text += "aag " + std::to_string(gates + 1) + " 1 0 1 " + std::to_string(gates) + "\n2\n";
    text += std::to_string(2 * (gates + 1)) + "\n";
    for (std::uint32_t variable = gates + 1; variable >= 2; --variable) {
        text += std::to_string(2 * variable) + " " + std::to_string(2 * (variable - 1)) + " 2\n";
    }

    const auto result = readText(std::move(text));
    INFO(diagnosticsText(result.diagnostics()));
    REQUIRE(result);
    auto const *circuit = fuzzybits::ir::pc::circuit_from(result.value().model);
    REQUIRE(circuit != nullptr);
    CHECK(circuit->node_count() == gates + 1);
}

// A body shorter than the header promises is one error. Today every missing line of the output and
// AND sections is its own io.aiger.unexpected-eof, so the work and the diagnostics grow with the
// header counts, not with the file (a 23-byte file gives 10^5 errors here).
TEST_CASE("AIGER reader reports a truncated body once", "[bug][aiger][issue-297]") {
    const auto result = readText("aag 0 0 0 100000 0\n");
    INFO(result.diagnostics().items().size() << " diagnostics");
    REQUIRE_FALSE(result);
    CHECK(errorCount(result.diagnostics()) <= 2);
}

// CRLF line ends: every numeric line is read with the CR stripped, but a symbol name keeps it
// ("x\r"), and the comment header `c\r` is not recognised, so the comment text that follows is read as
// a symbol line and the read fails.
TEST_CASE("AIGER reader reads symbol names and comments of a CRLF file", "[bug][aiger][issue-297]") {
    SECTION("symbol names") {
        const auto result = readFixture("crlf_symbols.aag");
        INFO(diagnosticsText(result.diagnostics()));
        REQUIRE(result);
        auto const *circuit = fuzzybits::ir::pc::circuit_from(result.value().model);
        REQUIRE(circuit != nullptr);
        REQUIRE(circuit->probability_sources().size() == 2);
        CHECK(circuit->probability_sources()[0].parameter_name == "x");
        CHECK(circuit->probability_sources()[1].parameter_name == "y");
    }
    SECTION("comment section") {
        const auto result = readFixture("crlf_comment.aag");
        INFO(diagnosticsText(result.diagnostics()));
        CHECK(result);
    }
}

// Output and latch symbols are not checked at all: an index beyond the outputs, two symbols for one
// output, and a line `cfoo` inside the symbol table (not the comment header `c`) are all accepted
// without a diagnostic. Input symbols get these checks (io.aiger.invalid-symbol,
// io.aiger.duplicate-symbol).
TEST_CASE("AIGER reader checks output symbols as it checks input symbols", "[bug][aiger][issue-297]") {
    SECTION("output symbol index out of range") {
        const auto result = readText("aag 1 1 0 1 0\n2\n2\no7 a\n");
        CHECK(result.diagnostics().has_errors());
    }
    SECTION("two symbols for one output") {
        const auto result = readText("aag 1 1 0 1 0\n2\n2\no0 a\no0 b\n");
        CHECK(result.diagnostics().has_errors());
    }
    SECTION("a symbol line that starts with c but is not the comment header") {
        const auto result = readText("aag 1 1 0 1 0\n2\n2\ncfoo\n");
        CHECK(result.diagnostics().has_errors());
    }
}

// An unnamed input gets the generated parameter name `i<pos>`, which can equal another input's own
// symbol: input 0 unnamed and input 1 named `i0` give two parameter sources both named `i0`, so a
// binding by name gives both inputs one value without a diagnostic.
TEST_CASE("AIGER reader gives every input a distinct parameter name", "[bug][aiger][issue-297]") {
    const auto result = readText("aag 3 2 0 1 1\n2\n4\n6\n6 2 4\ni1 i0\n");
    INFO(diagnosticsText(result.diagnostics()));
    REQUIRE(result);
    auto const *circuit = fuzzybits::ir::pc::circuit_from(result.value().model);
    REQUIRE(circuit != nullptr);
    std::set<std::string> names;
    for (auto const &source : circuit->probability_sources()) {
        names.insert(source.parameter_name);
    }
    CHECK(names.size() == circuit->probability_sources().size());
}
