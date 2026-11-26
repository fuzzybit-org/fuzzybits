// Known bugs of `fzb compile` on OpenPSA MEF input. The cases drive the fzb binary and evaluate the written
// .pcfb exhaustively (compile::exact_probability), so they check the `fzb compile` contract, not the reader
// behind it: today the SCRAM-derived MEF path, later the fuzzybits OpenPSA reader into ir::pra (#148), which
// must pass them unchanged. The fixtures are in input/bugs/openpsa/; each states its expected value. This
// test links no SCRAM code. Fails on develop until its issue is fixed. Labelled `bug`.

#include "compile/exact_eval.h"
#include "pcfb.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef FZB_PATH
#error "FZB_PATH must be provided by CMake"
#endif

namespace {

struct run_result final {
    int status = -1;
    std::string out;
    std::string err;
};

[[nodiscard]] auto quoted(std::string const &text) -> std::string {
#ifdef _WIN32
    return "\"" + text + "\"";
#else
    return "'" + text + "'";
#endif
}

[[nodiscard]] auto read_file(std::filesystem::path const &path) -> std::string {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

[[nodiscard]] auto scratch_directory(std::string const &name) -> std::filesystem::path {
    const auto path = std::filesystem::temp_directory_path() / ("fuzzybits-mef-bug-" + name);
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

[[nodiscard]] auto run_fzb(std::string const &arguments, std::filesystem::path const &scratch) -> run_result {
    const auto out_path = scratch / "fzb.out";
    const auto err_path = scratch / "fzb.err";
    const auto command =
        quoted(FZB_PATH) + " " + arguments + " > " + quoted(out_path.string()) + " 2> " + quoted(err_path.string());
#ifdef _WIN32
    const int status = std::system(("cmd /S /C \"" + command + "\"").c_str());
#else
    const int status = std::system(command.c_str());
#endif
    return run_result{.status = status, .out = read_file(out_path), .err = read_file(err_path)};
}

[[nodiscard]] auto fixture(std::string_view name) -> std::filesystem::path {
    return std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "bugs" / "openpsa" / name;
}

struct compiled final {
    run_result run;
    std::vector<std::byte> bytes;
};

// `fzb compile INPUT -o OUT --no-progress EXTRA`; bytes are empty when it failed.
[[nodiscard]] auto compile(std::filesystem::path const &input, std::string const &extra,
                           std::filesystem::path const &scratch) -> compiled {
    const auto output = scratch / "out.pcfb";
    std::filesystem::remove(output);
    compiled out;
    out.run = run_fzb(
        "compile " + quoted(input.string()) + " -o " + quoted(output.string()) + " --no-progress " + extra, scratch);
    if (out.run.status == 0 && std::filesystem::exists(output)) {
        const auto text = read_file(output);
        out.bytes.resize(text.size());
        for (std::size_t index = 0; index < text.size(); ++index) {
            out.bytes[index] = static_cast<std::byte>(text[index]);
        }
    }
    return out;
}

// Exact P of every query, by query name.
[[nodiscard]] auto exact_by_name(std::vector<std::byte> const &bytes) -> std::map<std::string, double> {
    auto circuit = pcfb::io::from_bytes(bytes);
    REQUIRE(circuit);
    REQUIRE(pcfb::validate(*circuit).empty());
    std::map<std::string, double> out;
    auto const *queries = (*circuit)->subcircuits()->Get(0)->queries();
    for (std::uint32_t index = 0; index < queries->size(); ++index) {
        auto const *query = queries->Get(index);
        out[query->name() == nullptr ? std::string{} : query->name()->str()] =
            fuzzybits::compile::exact_probability(*circuit, index);
    }
    return out;
}

// Requires a successful compile whose query `name` has exact probability `expected`.
void require_query(std::filesystem::path const &input, std::string const &extra, std::string const &name,
                   const double expected, std::filesystem::path const &scratch) {
    INFO("fzb compile " << input.filename().string() << " " << extra);
    const auto result = compile(input, extra, scratch);
    INFO("stdout: " << result.run.out << "\nstderr: " << result.run.err);
    REQUIRE(result.run.status == 0);
    const auto values = exact_by_name(result.bytes);
    REQUIRE(values.contains(name));
    CHECK(values.at(name) == Catch::Approx(expected).margin(1e-12));
}

} // namespace

// An at-least gate whose operands reach one event twice: through a one-operand (pass-through) gate, or
// through two gates with the same definition. ir::pc, pcfb and the MEF at-least count operands with
// multiplicity. The PDAG rewrite for a repeated at-least argument keeps the wrong one when two arguments are
// left (pdag.cc ProcessAtleastGateDuplicateArg: @(2,[x,x,z]) becomes z, not x): null-gate removal reaches it
// at every level unless --keep-null, and the merge of equal gates at -c 2 (the default) and above. The
// compiled file is a different, silently wrong function: 0.2 instead of 0.1, and 0.3 instead of 0.02.
TEST_CASE("fzb compile MEF: at-least gates whose operands reach one event twice",
          "[bug][mef][wrong-answer][issue-290]") {
    const auto scratch = scratch_directory("atleast");
    SECTION("through a one-operand gate") {
        for (const auto *level : {"", "-c 0", "-c 1", "-c 8"}) {
            require_query(fixture("atleast_operand_through_pass_through_gate.xml"), level, "TOP", 0.1, scratch);
        }
    }
    SECTION("through two equal gates") {
        for (const auto *level : {"", "-c 1", "-c 4", "-c 8"}) {
            require_query(fixture("atleast_operands_equal_gates.xml"), level, "TOP", 0.02, scratch);
        }
    }
    std::filesystem::remove_all(scratch);
}

// --watch NAME adds a query on the gate NAME. When NAME is a NOR (NAND) gate, normalisation rewrites it to an OR
// (AND) under its own MEF name and moves the negation onto the parent's edge; the watch binds to that node and
// reports the complement: 0.28 instead of 0.72. The target itself stays right.
TEST_CASE("fzb compile MEF: --watch on a NOR gate reports the gate", "[bug][mef][wrong-answer][issue-290]") {
    const auto scratch = scratch_directory("watch");
    for (const auto *level : {"--watch G", "--watch G -c 1"}) {
        require_query(fixture("watch_nor_gate.xml"), level, "TOP", 0.36, scratch);
        require_query(fixture("watch_nor_gate.xml"), level, "G", 0.72, scratch);
    }
    std::filesystem::remove_all(scratch);
}

// --target matches id, name or full path over a hash table and takes the first hit, so a bare name that two
// fault trees use for private gates compiles one of them, chosen by hash order, without a diagnostic.
TEST_CASE("fzb compile MEF: --target with a name two fault trees use is refused as ambiguous",
          "[bug][mef][issue-290]") {
    const auto scratch = scratch_directory("ambiguous");
    const auto result = compile(fixture("target_name_in_two_fault_trees.xml"), "--target TOP", scratch);
    INFO("stdout: " << result.run.out << "\nstderr: " << result.run.err);
    CHECK(result.run.status != 0);
    // The qualified names stay usable.
    require_query(fixture("target_name_in_two_fault_trees.xml"), "--target FT1.TOP", "FT1.TOP", 0.28, scratch);
    require_query(fixture("target_name_in_two_fault_trees.xml"), "--target FT2.TOP", "FT2.TOP", 0.02, scratch);
    std::filesystem::remove_all(scratch);
}

// A fault tree with two top gates and no --target: the top list comes from an unordered set of gate pointers and
// the first one is compiled, without a diagnostic (related: #92). Either a refusal or a query per top is right.
TEST_CASE("fzb compile MEF: two top gates without --target are refused or both queried", "[bug][mef][issue-290]") {
    const auto scratch = scratch_directory("two-tops");
    const auto result = compile(fixture("two_top_gates.xml"), "", scratch);
    INFO("stdout: " << result.run.out << "\nstderr: " << result.run.err);
    if (result.run.status == 0) {
        const auto values = exact_by_name(result.bytes);
        CHECK(values.size() == 2);
        CHECK(values.contains("T1"));
        CHECK(values.contains("T2"));
    }
    std::filesystem::remove_all(scratch);
}

// Valid MEF the compile refuses:
//  - atleast min="1" and atleast with min equal to the operand count ("Min number cannot be less than 2",
//    "must have more arguments than its min number"): OR and AND;
//  - A and not A in one gate (the duplicate-argument check compares ids and ignores the complement): FALSE;
//  - nand with one argument (share/input.rng allows one or more): NOT;
//  - beta-deviate(1, 1), the uniform distribution on [0, 1]: its range is computed as [0, 1 / I_0.99(1, 1)] =
//    [0, 1.0101] and refused as a probability. Expected at the mean of A, the convention every other deviate
//    gets today (see the issue for the convention itself).
TEST_CASE("fzb compile MEF: valid gates and distributions are compiled, not refused", "[bug][mef][issue-295]") {
    const auto scratch = scratch_directory("valid");
    SECTION("atleast min=1") { require_query(fixture("atleast_min_1.xml"), "", "TOP", 0.28, scratch); }
    SECTION("atleast with min equal to the operand count") {
        require_query(fixture("atleast_min_equals_operands.xml"), "", "TOP", 0.02, scratch);
    }
    SECTION("an event and its complement in one gate") {
        require_query(fixture("event_and_its_complement.xml"), "", "TOP", 0.0, scratch);
    }
    SECTION("nand with one argument") { require_query(fixture("nand_one_operand.xml"), "", "TOP", 0.9, scratch); }
    SECTION("beta-deviate(1, 1)") { require_query(fixture("beta_deviate_uniform.xml"), "", "TOP", 0.55, scratch); }
    std::filesystem::remove_all(scratch);
}

// --no-kn expands every at-least gate into AND/OR. NormalizeAtleastGate builds new, unshared sub-gates at each
// step, T(n, k) = 3 + T(n-1, k-1) + T(n-1, k), and later passes merge them again, so the written circuit is
// small (210 operators for 10 of 20) but the compile time grows like C(n, k): 0.3 s for 8 of 16, 1.9 s for
// 9 of 18, 23 s for 10 of 20 and more than 2 min for 11 of 22 (input/bugs/openpsa/atleast_15_of_30.xml does
// not finish). A shared expansion needs O(k (n - k)) gates, so 10 of 20 should take well under a second.
TEST_CASE("fzb compile MEF --no-kn expands a 10-of-20 gate in polynomial time", "[bug][mef][issue-280]") {
    const auto scratch = scratch_directory("no-kn");
    constexpr int n = 20;
    constexpr int k = 10;
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n<opsa-mef>\n<define-fault-tree name=\"FT\">\n"
           "<define-gate name=\"TOP\"><atleast min=\""
        << k << "\">";
    for (int index = 0; index < n; ++index) {
        xml << "<basic-event name=\"E" << index << "\"/>";
    }
    xml << "</atleast></define-gate>\n";
    for (int index = 0; index < n; ++index) {
        xml << "<define-basic-event name=\"E" << index << "\"><float value=\"0.1\"/></define-basic-event>\n";
    }
    xml << "</define-fault-tree>\n</opsa-mef>\n";
    const auto input = scratch / "atleast_10_of_20.xml";
    std::ofstream(input, std::ios::binary) << xml.str();

    // P(at least k of n at p = 0.1).
    double expected = 0.0;
    for (int hits = k; hits <= n; ++hits) {
        double choose = 1.0;
        for (int i = 0; i < hits; ++i) {
            choose = choose * (n - i) / (i + 1);
        }
        expected += choose * std::pow(0.1, hits) * std::pow(0.9, n - hits);
    }
    const auto start = std::chrono::steady_clock::now();
    const auto result = compile(input, "--no-kn", scratch);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    INFO("stdout: " << result.run.out << "\nstderr: " << result.run.err);
    REQUIRE(result.run.status == 0);
    CHECK(exact_by_name(result.bytes).at("TOP") == Catch::Approx(expected).margin(1e-15));
    CHECK(elapsed.count() < 5.0);
    std::filesystem::remove_all(scratch);
}
