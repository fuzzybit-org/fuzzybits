// Known bugs of the AIGER reader. Each case states the AIGER rule or ir::pc contract it checks and fails
// on develop until its issue is fixed; the fixtures are in input/bugs/aiger/. Cases tagged [crash]
// terminate the process today and run as their own ctest entry. This executable is labelled `bug`.
//
// AIGER rules cited from src/io/aiger/schema/format.md (the AIGER format description): the ASCII AND gates "can be
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
#include <vector>

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

// A file with CRLF line ends is not AIGER (src/io/aiger/schema/format.md): "The header line ends with a new
// line character immediately after the last digit of the number of ands", a symbol is "an arbitrary ASCII
// string of printable characters excluding the new line character" (CR is not printable), and the comment
// section "starts with a 'c' character followed by a new line". Both fixtures are therefore malformed at
// their first line and must be refused with an error, for the same reason.
//
// Today the reader trims the CR from numeric lines, so it accepts crlf_symbols.aag and imports the symbol
// names as "x\r" and "y\r" (writing the circuit back then fails), and it refuses crlf_comment.aag only
// because `c\r` is not the comment header, with io.aiger.invalid-symbol for the comment text: a diagnostic
// for the wrong defect, at the wrong line. (Kept out of input/aiger, whose corpus test reads every file.)
TEST_CASE("AIGER reader refuses a CRLF file with an error, whatever follows the header", "[bug][aiger][issue-297]") {
    std::vector<std::string> first_errors;
    for (const auto *name : {"crlf_symbols.aag", "crlf_comment.aag"}) {
        INFO(name);
        const auto result = readFixture(name);
        INFO(diagnosticsText(result.diagnostics()));
        CHECK_FALSE(result);
        CHECK(errorCount(result.diagnostics()) >= 1);
        for (auto const &diagnostic : result.diagnostics().items()) {
            if (diagnostic.severity == fuzzybits::api::diagnostic_severity::error) {
                first_errors.push_back(diagnostic.code);
                break;
            }
        }
    }
    // One defect, one diagnostic: the line end, found before the body is read.
    REQUIRE(first_errors.size() == 2);
    CHECK(first_errors[0] == first_errors[1]);
    CHECK(first_errors[1] != "io.aiger.invalid-symbol");
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
