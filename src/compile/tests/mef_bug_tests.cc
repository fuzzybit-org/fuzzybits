// Known bugs of `fzb compile` on OpenPSA MEF input. The cases drive the fzb binary and evaluate the written
// .pcfb exhaustively (compile::exact_probability), so they check the `fzb compile` contract, not the reader
// behind it: today the SCRAM-derived MEF path, later the fuzzybits OpenPSA reader into ir::pra (#148), which
// must pass them unchanged. The fixtures are in input/bugs/openpsa/; each states its expected value. This
// test links no SCRAM code. Fails on develop until its issue is fixed. Labelled `bug`.

#include "compile/exact_eval.h"
#include "io/pcfb/tests/portable_random.h"
#include "pcfb.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
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

// From -c 3, ancestor processing simplifies a gate with what its parent already fixes (inside TOP = A & G1, A is
// TRUE, so G1 = B | (A & C) becomes B | C) and keeps the gate's MEF name; a watch on G1 then reads the simplified
// function: 0.44 instead of 0.32. The OR-parent variant (TOP = A | G1, G1 = B & (A | C), where A is FALSE inside
// the parent) reads 0.06 instead of 0.13, run by hand.
TEST_CASE("fzb compile MEF: --watch on a gate simplified in its parent's context reports the gate",
          "[bug][mef][wrong-answer][issue-290]") {
    const auto scratch = scratch_directory("watch-context");
    for (const auto *level : {"-c 2", "-c 3", "-c 5", "-c 8"}) {
        SECTION(std::string("level ") + level) {
            require_query(fixture("watch_gate_simplified_in_context.xml"), std::string("--watch G1 ") + level, "TOP",
                          0.22, scratch);
            require_query(fixture("watch_gate_simplified_in_context.xml"), std::string("--watch G1 ") + level, "G1",
                          0.32, scratch);
        }
    }
    std::filesystem::remove_all(scratch);
}

// At -c 7 and 8, a negation on a reference to a gate (TOP = A & not G1) is pushed into the gate, which keeps its
// MEF name, so a watch on G1 reports its complement: 0.94 instead of 0.06. OR and XOR gates behave the same
// (0.56 for 0.44, 0.62 for 0.38), run by hand.
TEST_CASE("fzb compile MEF: --watch on a gate referenced through not reports the gate",
          "[bug][mef][wrong-answer][issue-290]") {
    const auto scratch = scratch_directory("watch-complemented");
    for (const auto *level : {"-c 2", "-c 7", "-c 8"}) {
        SECTION(std::string("level ") + level) {
            require_query(fixture("watch_complemented_gate.xml"), std::string("--watch G1 ") + level, "TOP", 0.47,
                          scratch);
            require_query(fixture("watch_complemented_gate.xml"), std::string("--watch G1 ") + level, "G1", 0.06,
                          scratch);
        }
    }
    std::filesystem::remove_all(scratch);
}

// --watch NAME on a gate of the target's cone is refused ("watched node not in compiled PDAG") when the PDAG no
// longer has a node for it: a one-operand gate is removed by null-gate removal at every level unless --keep-null,
// and every gate below a target that folds to a constant goes with it. The watched function still exists and has
// a value (0.2 and 0.1 here); a removed gate is a reference to another node or a constant, so its query can be
// bound to that image (docs/IR-PC-REQUIREMENTS.md section 12, node maps of transforms).
TEST_CASE("fzb compile MEF: --watch on a gate that preprocessing removes is answered, not refused",
          "[bug][mef][issue-290]") {
    const auto scratch = scratch_directory("watch-removed");
    for (const auto *level : {"-c 0", "-c 1", "", "-c 8"}) {
        SECTION(std::string("one-operand gate ") + level) {
            require_query(fixture("watch_pass_through_gate.xml"), std::string("--watch G1 ") + level, "G1", 0.2,
                          scratch);
        }
        SECTION(std::string("gate under a constant target ") + level) {
            require_query(fixture("watch_gate_under_constant_top.xml"), std::string("--watch G1 ") + level, "G1", 0.1,
                          scratch);
        }
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

// MEF constructs that change the model are accepted and then dropped without a diagnostic: the PDAG keeps
// non-declarative substitutions (those with a <source>) for the ZBDD cut-set path only, and pdag_to_pcfb never
// reads them; alignments and their phases are read by risk_analysis only. The written file answers the model
// without them (0.02, 0.02, 0.58355). docs/IR-PRA-REQUIREMENTS.md: unsupported constructs "must produce
// diagnostics ..., not guessed formulas or probabilities". Applying them or refusing them is right; dropping them
// is not.
TEST_CASE("fzb compile MEF: substitutions with a source and alignments are applied or refused, never dropped",
          "[bug][mef][wrong-answer][issue-299]") {
    const auto scratch = scratch_directory("ignored");
    const auto applied_or_refused = [&scratch](const char *name, const char *target, const double applied) {
        INFO(name);
        const auto result = compile(fixture(name), "", scratch);
        INFO("stdout: " << result.run.out << "\nstderr: " << result.run.err);
        if (result.run.status != 0) {
            SUCCEED("refused");
            return;
        }
        CHECK(exact_by_name(result.bytes).at(target) == Catch::Approx(applied).margin(1e-9));
    };
    SECTION("exchange-event substitution") { applied_or_refused("substitution_exchange_event.xml", "TOP", 0.1); }
    SECTION("recovery rule with a source") { applied_or_refused("substitution_recovery_with_source.xml", "TOP", 0.3); }
    SECTION("alignment phase setting a house event") {
        applied_or_refused("alignment_set_house_event.xml", "TOP", 1.0);
    }
    std::filesystem::remove_all(scratch);
}

// NaN passes the XML number reader (strtod) and the probability range check (NaN < 0 and NaN > 1 are both false);
// max/min (fmax/fmin) and comparisons then drop it, so max(NaN, 0.3) compiles as 0.3 and ite(NaN > 0, 0.9, 0.1)
// as 0.1. A NaN in an expression has no value; the model must be refused.
TEST_CASE("fzb compile MEF: an expression with NaN is refused", "[bug][mef][issue-299]") {
    const auto scratch = scratch_directory("nan");
    for (const auto *name : {"nan_through_max.xml", "nan_through_ite.xml"}) {
        INFO(name);
        const auto result = compile(fixture(name), "", scratch);
        INFO("stdout: " << result.run.out << "\nstderr: " << result.run.err);
        CHECK(result.run.status != 0);
    }
    std::filesystem::remove_all(scratch);
}

// More valid MEF the compile refuses (each fixture states its value):
//  - sin(0.5) and cos(0): the interval of sin and cos is taken as [-1, 1] and checked as a probability;
//  - or, mean and add with one argument (input.rng allows one or more);
//  - a repeated <constant value="true"/> or a repeated event in one gate (the duplicate-argument check);
//  - GLM with failure rate 0, which is gamma exp(-mu t);
//  - an alpha-factor group with only alpha_1, refused as "Expression requires 2 or more arguments".
TEST_CASE("fzb compile MEF: more valid expressions, gates and CCF groups are compiled, not refused",
          "[bug][mef][issue-295]") {
    const auto scratch = scratch_directory("valid-more");
    SECTION("sin") { require_query(fixture("expression_sin.xml"), "", "TOP", 0.479425538604203, scratch); }
    SECTION("cos") { require_query(fixture("expression_cos.xml"), "", "TOP", 1.0, scratch); }
    SECTION("or with one argument") { require_query(fixture("gate_or_one_operand.xml"), "", "TOP", 0.1, scratch); }
    SECTION("mean with one argument") {
        require_query(fixture("expression_mean_one_operand.xml"), "", "TOP", 0.3, scratch);
    }
    SECTION("add with one argument") {
        require_query(fixture("expression_add_one_operand.xml"), "", "TOP", 0.3, scratch);
    }
    SECTION("a repeated constant") { require_query(fixture("repeated_constant_operand.xml"), "", "TOP", 0.1, scratch); }
    SECTION("a repeated event") { require_query(fixture("repeated_event_operand.xml"), "", "TOP", 0.1, scratch); }
    SECTION("GLM with failure rate 0") {
        require_query(fixture("glm_zero_failure_rate.xml"), "", "TOP", 0.0367879441171442, scratch);
    }
    SECTION("alpha-factor group with alpha_1 only") {
        require_query(fixture("alpha_factor_level_1_only.xml"), "", "TOP", 0.01, scratch);
    }
    std::filesystem::remove_all(scratch);
}

// --watch on a CCF member is refused ("watched node not in compiled PDAG"): CCF expansion replaces the member by a
// proxy gate (its independent event OR its common events) that is never added to the watched gates. The member
// has a value, 0.0984 here.
TEST_CASE("fzb compile MEF: --watch on a CCF group member is answered", "[bug][mef][issue-290]") {
    const auto scratch = scratch_directory("watch-ccf");
    require_query(fixture("watch_ccf_member.xml"), "--watch A", "AND2", 0.026272, scratch);
    require_query(fixture("watch_ccf_member.xml"), "--watch A", "A", 0.0984, scratch);
    std::filesystem::remove_all(scratch);
}

// Parts of the SCRAM PDAG construction and preprocessing recurse once per level of gate depth, so a 10^5-deep
// chain of alternating AND and OR gates (valid MEF, 2e5 elements) ends `fzb compile` with SIGSEGV at -c 0 and at
// the default level. The same chain compiles from FTP and from nombrilo, whose readers and passes are iterative; the
// runtime loader has the same recursion (fzb_runtime_pcfb_bug_tests, "[crash]"). Expected: a compile, or a refusal
// with a diagnostic.
TEST_CASE("fzb compile MEF: a 10^5-deep gate chain compiles", "[bug][mef][issue-300]") {
    const auto scratch = scratch_directory("deep");
    constexpr int depth = 100'000;
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n<opsa-mef><define-fault-tree name=\"FT\">\n";
    for (int index = 0; index < depth; ++index) {
        const char *kind = index % 2 == 0 ? "and" : "or";
        xml << "<define-gate name=\"G" << index << "\"><" << kind << "><gate name=\"G" << index + 1
            << "\"/><basic-event name=\"E" << index << "\"/></" << kind << "></define-gate>\n";
    }
    xml << "<define-gate name=\"G" << depth << "\"><basic-event name=\"E" << depth << "\"/></define-gate>\n";
    for (int index = 0; index <= depth; ++index) {
        xml << "<define-basic-event name=\"E" << index << "\"><float value=\"0.5\"/></define-basic-event>\n";
    }
    xml << "</define-fault-tree></opsa-mef>\n";
    const auto input = scratch / "chain.xml";
    std::ofstream(input, std::ios::binary) << xml.str();
    for (const auto *level : {"-c 0 --keep-null", "-c 2"}) {
        INFO(level);
        const auto result = compile(input, std::string("--target G0 ") + level, scratch);
        INFO("stderr: " << result.run.err);
        CHECK(result.run.status == 0);
    }
    std::filesystem::remove_all(scratch);
}

// The .pcfb size limit of #285 reaches the MEF path too, and silently: pcfb::io::from_bytes verifies with
// flatbuffers::Verifier's default options (at most 10^6 tables, about three per basic event), so a model with
// 3.5e5 basic events (250 OR gates of 1,400 events each) is written, 88 MB with exit status 0, and then cannot be
// read by any reader ("FlatBuffer verification failed"). The FTP writer checks its own output; the MEF writer
// (src/compile/pcfb_writer.cc) does not. Expected: a readable file, or a compile that fails with a diagnostic.
TEST_CASE("fzb compile MEF of 3.5e5 basic events writes a readable .pcfb or fails (#285)",
          "[bug][mef][pcfb][scale][issue-285]") {
    const auto scratch = scratch_directory("size");
    constexpr int gates = 250;
    constexpr int fan_in = 1400;
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n<opsa-mef><define-fault-tree name=\"FT\">\n<define-gate name=\"TOP\"><or>";
    for (int gate = 0; gate < gates; ++gate) {
        xml << "<gate name=\"G" << gate << "\"/>";
    }
    xml << "</or></define-gate>\n";
    for (int gate = 0; gate < gates; ++gate) {
        xml << "<define-gate name=\"G" << gate << "\"><or>";
        for (int event = 0; event < fan_in; ++event) {
            xml << "<basic-event name=\"E" << gate << '_' << event << "\"/>";
        }
        xml << "</or></define-gate>\n";
    }
    for (int gate = 0; gate < gates; ++gate) {
        for (int event = 0; event < fan_in; ++event) {
            xml << "<define-basic-event name=\"E" << gate << '_' << event
                << "\"><float value=\"1e-6\"/></define-basic-event>\n";
        }
    }
    xml << "</define-fault-tree></opsa-mef>\n";
    const auto input = scratch / "wide.xml";
    std::ofstream(input, std::ios::binary) << xml.str();
    const auto result = compile(input, "-c 0 --keep-null", scratch);
    INFO("stderr: " << result.run.err);
    if (result.run.status != 0) {
        SUCCEED("refused");
    } else {
        auto view = pcfb::io::from_bytes(result.bytes);
        INFO((view ? std::string{} : view.error().message));
        CHECK(view);
    }
    std::filesystem::remove_all(scratch);
}

// The MEF path spends time quadratic in a gate's operand count: one OR over 1e4, 2e4 and 4e4 basic events compiles
// in 0.8 s, 2.2 s and 7.8 s (and 4e5 had not finished after 12 minutes), where the FTP path takes 0.1 to 0.2 s for
// the same gates. 6e4 operands take about 17 s today; a linear reader takes well under a second.
TEST_CASE("fzb compile MEF: one gate over 6e4 events compiles in linear time", "[bug][mef][issue-300]") {
    const auto scratch = scratch_directory("fan-in");
    constexpr int events = 60'000;
    std::ostringstream xml;
    xml << "<?xml version=\"1.0\"?>\n<opsa-mef><define-fault-tree name=\"FT\">\n<define-gate name=\"TOP\"><or>";
    for (int event = 0; event < events; ++event) {
        xml << "<basic-event name=\"E" << event << "\"/>";
    }
    xml << "</or></define-gate>\n";
    for (int event = 0; event < events; ++event) {
        xml << "<define-basic-event name=\"E" << event << "\"><float value=\"1e-6\"/></define-basic-event>\n";
    }
    xml << "</define-fault-tree></opsa-mef>\n";
    const auto input = scratch / "fan_in.xml";
    std::ofstream(input, std::ios::binary) << xml.str();
    const auto start = std::chrono::steady_clock::now();
    const auto result = compile(input, "-c 0", scratch);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    INFO("stderr: " << result.run.err);
    REQUIRE(result.run.status == 0);
    CHECK(elapsed.count() < 5.0);
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

namespace {

namespace random = pcfb_test_random;

// A random MEF fault tree over 2..6 basic events: gates G0..Gn-1, each over events, house events and gates of a
// higher index (so the graph is acyclic and G0 can reach any of them), with complemented operands. Only forms
// share/input.rng and the SCRAM reader accept today, so a refusal here is a finding too.
struct random_gate final {
    std::string kind;                                  // and or atleast xor not nand nor iff imply cardinality null
    std::vector<std::tuple<char, int, bool>> operands; // ('e'|'h'|'g', index, complemented)
    int low = 0;
    int high = 0;
};

struct random_model final {
    std::vector<double> probabilities;
    std::vector<bool> houses;
    std::vector<random_gate> gates;
};

[[nodiscard]] auto random_fault_tree(std::mt19937_64 &rng) -> random_model {
    static constexpr std::array<double, 8> kProbabilities{0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0};
    static constexpr std::array<const char *, 11> kKinds{"and", "or",  "atleast", "xor",         "not", "nand",
                                                         "nor", "iff", "imply",   "cardinality", "null"};
    random_model model;
    const auto events = random::uniform(rng, 2, 6);
    for (std::uint64_t index = 0; index < events; ++index) {
        model.probabilities.push_back(random::chance(rng, 0.25)
                                          ? kProbabilities[random::uniform(rng, 0, 7)]
                                          : 0.1 * static_cast<double>(random::uniform(rng, 1, 6)));
    }
    for (std::uint64_t index = random::uniform(rng, 0, 2); index > 0; --index) {
        model.houses.push_back(random::chance(rng, 0.5));
    }
    const auto gates = static_cast<int>(random::uniform(rng, 2, 7));
    for (int index = 0; index < gates; ++index) {
        random_gate gate{.kind = kKinds[random::uniform(rng, 0, kKinds.size() - 1)]};
        std::vector<std::pair<char, int>> pool;
        for (std::uint64_t event = 0; event < events; ++event) {
            pool.emplace_back('e', static_cast<int>(event));
        }
        for (std::size_t house = 0; house < model.houses.size(); ++house) {
            pool.emplace_back('h', static_cast<int>(house));
        }
        for (int child = index + 1; child < gates; ++child) {
            pool.emplace_back('g', child);
        }
        std::size_t count = 0;
        if (gate.kind == "not" || gate.kind == "null") {
            count = 1;
        } else if (gate.kind == "xor" || gate.kind == "iff" || gate.kind == "imply") {
            count = 2;
        } else {
            count = static_cast<std::size_t>(random::uniform(rng, 2, std::min<std::uint64_t>(5, pool.size())));
        }
        random::shuffle(std::span(pool), rng);
        const bool unary = count == 1;
        for (std::size_t at = 0; at < count; ++at) {
            // A complemented operand of a one-operand form would be a nested formula, which the schema refuses.
            gate.operands.emplace_back(pool[at].first, pool[at].second, !unary && random::chance(rng, 0.25));
        }
        const auto n = static_cast<int>(count);
        if (gate.kind == "atleast") {
            if (n < 3) {
                gate.kind = "and";
            } else {
                gate.low = static_cast<int>(random::uniform(rng, 2, static_cast<std::uint64_t>(n - 1)));
            }
        } else if (gate.kind == "cardinality") {
            gate.low = static_cast<int>(random::uniform(rng, 0, static_cast<std::uint64_t>(n)));
            gate.high = static_cast<int>(
                random::uniform(rng, static_cast<std::uint64_t>(gate.low), static_cast<std::uint64_t>(n)));
        }
        model.gates.push_back(std::move(gate));
    }
    return model;
}

[[nodiscard]] auto to_xml(random_model const &model) -> std::string {
    std::ostringstream out;
    out << "<?xml version=\"1.0\"?>\n<opsa-mef>\n<define-fault-tree name=\"FT\">\n";
    const auto reference = [&](char type, int index) {
        std::ostringstream tag;
        tag << (type == 'e'   ? "<basic-event name=\"E"
                : type == 'h' ? "<house-event name=\"H"
                              : "<gate name=\"G")
            << index << "\"/>";
        return tag.str();
    };
    for (std::size_t index = 0; index < model.gates.size(); ++index) {
        auto const &gate = model.gates[index];
        std::string operands;
        for (auto const &[type, at, complemented] : gate.operands) {
            operands += complemented ? "<not>" + reference(type, at) + "</not>" : reference(type, at);
        }
        out << "<define-gate name=\"G" << index << "\">";
        if (gate.kind == "null") {
            out << operands;
        } else if (gate.kind == "atleast") {
            out << "<atleast min=\"" << gate.low << "\">" << operands << "</atleast>";
        } else if (gate.kind == "cardinality") {
            out << "<cardinality min=\"" << gate.low << "\" max=\"" << gate.high << "\">" << operands
                << "</cardinality>";
        } else {
            out << '<' << gate.kind << '>' << operands << "</" << gate.kind << '>';
        }
        out << "</define-gate>\n";
    }
    out.precision(17);
    for (std::size_t index = 0; index < model.probabilities.size(); ++index) {
        out << "<define-basic-event name=\"E" << index << "\"><float value=\"" << model.probabilities[index]
            << "\"/></define-basic-event>\n";
    }
    for (std::size_t index = 0; index < model.houses.size(); ++index) {
        out << "<define-house-event name=\"H" << index << "\"><constant value=\""
            << (model.houses[index] ? "true" : "false") << "\"/></define-house-event>\n";
    }
    out << "</define-fault-tree>\n</opsa-mef>\n";
    return out.str();
}

// The probability of every gate, by truth table over the basic events.
[[nodiscard]] auto truth_table(random_model const &model) -> std::vector<double> {
    const auto events = model.probabilities.size();
    std::vector<double> probability(model.gates.size(), 0.0);
    std::vector<int> value(model.gates.size(), 0);
    for (std::uint32_t assignment = 0; assignment < (1U << events); ++assignment) {
        double weight = 1.0;
        for (std::size_t event = 0; event < events; ++event) {
            const auto p = model.probabilities[event];
            weight *= ((assignment >> event) & 1U) != 0 ? p : 1.0 - p;
        }
        for (auto index = static_cast<int>(model.gates.size()) - 1; index >= 0; --index) {
            auto const &gate = model.gates[static_cast<std::size_t>(index)];
            std::vector<int> in;
            for (auto const &[type, at, complemented] : gate.operands) {
                int v = type == 'e'   ? static_cast<int>((assignment >> at) & 1U)
                        : type == 'h' ? static_cast<int>(model.houses[static_cast<std::size_t>(at)])
                                      : value[static_cast<std::size_t>(at)];
                in.push_back(complemented ? 1 - v : v);
            }
            const auto n = static_cast<int>(in.size());
            int ones = 0;
            for (const int v : in) {
                ones += v;
            }
            int out = 0;
            if (gate.kind == "and")
                out = ones == n;
            else if (gate.kind == "or")
                out = ones > 0;
            else if (gate.kind == "xor")
                out = ones % 2;
            else if (gate.kind == "not")
                out = 1 - in[0];
            else if (gate.kind == "null")
                out = in[0];
            else if (gate.kind == "nand")
                out = ones != n;
            else if (gate.kind == "nor")
                out = ones == 0;
            else if (gate.kind == "iff")
                out = in[0] == in[1];
            else if (gate.kind == "imply")
                out = in[0] == 0 || in[1] == 1;
            else if (gate.kind == "atleast")
                out = ones >= gate.low;
            else if (gate.kind == "cardinality")
                out = ones >= gate.low && ones <= gate.high;
            value[static_cast<std::size_t>(index)] = out;
        }
        for (std::size_t index = 0; index < model.gates.size(); ++index) {
            probability[index] += weight * value[index];
        }
    }
    return probability;
}

} // namespace

// Differential check of the whole MEF path: random fault trees (fixed seeds, portable draws), compiled at every
// level and with each preprocessing switch, with G0 as the target and every other gate of its cone watched. Every
// query must equal the truth table of the model as written. Preprocessing levels and switches are documented as
// optimisations, so they may change the circuit but never a value. This is how the watch cases above were found.
// On develop 201 of the 336 compiles fail: 183 watches refused at every level, -c 0 included, and 18 watched values
// wrong from -c 1 (mostly complements). No target is wrong in these 24 models; the at-least case above is rarer.
TEST_CASE("fzb compile MEF: random fault trees keep their truth-table values at every level",
          "[bug][mef][wrong-answer][differential][issue-290]") {
    const auto scratch = scratch_directory("differential");
    const std::vector<std::string> configurations{"-c 0",
                                                  "-c 1",
                                                  "-c 2",
                                                  "-c 3",
                                                  "-c 4",
                                                  "-c 5",
                                                  "-c 6",
                                                  "-c 7",
                                                  "-c 8",
                                                  "-c 0 --keep-null",
                                                  "-c 2 --keep-null",
                                                  "-c 2 --no-kn",
                                                  "-c 2 --no-xor",
                                                  "-c 8 --no-kn --no-xor"};
    std::mt19937_64 rng(290);
    std::size_t checked = 0;
    std::size_t failed = 0;
    for (int round = 0; round < 24; ++round) {
        const auto model = random_fault_tree(rng);
        const auto expected = truth_table(model);
        // The cone of G0.
        std::vector<bool> reached(model.gates.size(), false);
        std::vector<int> stack{0};
        reached[0] = true;
        while (!stack.empty()) {
            const auto gate = stack.back();
            stack.pop_back();
            for (auto const &[type, at, complemented] : model.gates[static_cast<std::size_t>(gate)].operands) {
                if (type == 'g' && !reached[static_cast<std::size_t>(at)]) {
                    reached[static_cast<std::size_t>(at)] = true;
                    stack.push_back(at);
                }
            }
        }
        std::string watch;
        for (std::size_t index = 1; index < model.gates.size(); ++index) {
            if (reached[index]) {
                watch += " --watch G" + std::to_string(index);
            }
        }
        const auto input = scratch / ("model_" + std::to_string(round) + ".xml");
        std::ofstream(input, std::ios::binary) << to_xml(model);
        for (auto const &configuration : configurations) {
            INFO("round " << round << ", fzb compile " << configuration << " --target G0" << watch << "\n"
                          << to_xml(model));
            const auto result = compile(input, configuration + " --target G0" + watch, scratch);
            INFO("stderr: " << result.run.err);
            ++checked;
            CHECK(result.run.status == 0);
            if (result.run.status != 0) {
                ++failed;
                continue;
            }
            for (auto const &[name, value] : exact_by_name(result.bytes)) {
                const auto gate = static_cast<std::size_t>(std::stoul(name.substr(1)));
                INFO("query " << name << ": truth table " << expected[gate] << ", compiled " << value);
                CHECK(value == Catch::Approx(expected[gate]).margin(1e-9));
                failed += std::abs(value - expected[gate]) > 1e-9 ? 1 : 0;
            }
        }
    }
    INFO(failed << " of " << checked << " compiles refused or wrong");
    CHECK(failed == 0);
    std::filesystem::remove_all(scratch);
}
