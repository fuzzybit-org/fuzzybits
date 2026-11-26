// Known bugs of the FTP compile driver, `fzb inspect` and the frequency lowering. Each case states the
// contract it checks and fails on develop until its issue is fixed; the fixtures are in
// input/bugs/ftp/. This executable is labelled `bug` (ctest -L bug).

#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "compile/exact_eval.h"
#include "compile/ftp/compile.h"
#include "compile/ftp/inspect.h"
#include "compile/ftp/json.h"
#include "pcfb.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace ftp = fuzzybits::compile::ftp;
namespace reader = fuzzybits::io::backend::fuzzybits::ftp;
using fuzzybits::api::diagnostic_list;

[[nodiscard]] auto render(diagnostic_list const &diagnostics) -> std::string {
    std::ostringstream out;
    for (auto const &item : diagnostics.items()) {
        out << item.code << ": " << item.message << '\n';
    }
    return out.str();
}

[[nodiscard]] auto render(ftp::inspection const &inspection) -> std::string {
    std::ostringstream out;
    for (auto const &item : inspection.diagnostics) {
        out << item.id << ": " << item.message << '\n';
    }
    return out.str();
}

[[nodiscard]] auto bug_fixture(std::string_view name) -> std::filesystem::path {
    return std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "bugs" / "ftp" / name;
}

[[nodiscard]] auto probability_options() -> ftp::options {
    ftp::options options;
    options.kinds = {"probability"};
    options.initiator_values = "probability";
    return options;
}

[[nodiscard]] auto exact(std::vector<std::byte> const &bytes, const std::uint32_t query = 0) -> double {
    auto circuit = ::pcfb::io::from_bytes(bytes);
    INFO((circuit ? std::string{} : circuit.error().message));
    REQUIRE(circuit);
    return fuzzybits::compile::exact_probability(*circuit, query);
}

[[nodiscard]] auto query_names(std::vector<std::byte> const &bytes) -> std::vector<std::string> {
    auto circuit = ::pcfb::io::from_bytes(bytes);
    REQUIRE(circuit);
    std::vector<std::string> out;
    for (auto const *query : *(*circuit)->subcircuits()->Get(0)->queries()) {
        out.push_back(query->name() == nullptr ? std::string{} : query->name()->str());
    }
    return out;
}

[[nodiscard]] auto compile_text(std::string text, ftp::options const &options) -> ftp::compile_result {
    const std::vector<reader::source_text> sources{reader::source_text{.path = "m.ftp", .text = std::move(text)}};
    return ftp::compile_sources(sources, options);
}

// The value the duplicate-import policy chose for `name`, as `fzb inspect` reports it.
[[nodiscard]] auto chosen_value(ftp::inspection const &inspection, std::string_view name) -> std::optional<double> {
    if (!inspection.structure) {
        return std::nullopt;
    }
    for (auto const &duplicate : inspection.structure->duplicate_imports) {
        if (duplicate.name == name) {
            return duplicate.value;
        }
    }
    return std::nullopt;
}

// A model whose single target needs an initiating event on every path, so that under
// `--initiator-values per-year` (the default) every gate depends on the initiators and is lowered
// once per initiator: `initiators` x `3 * levels` conditioned gates. Level k is
//   G_k = OR(C_k, D_k), C_k = AND(G_{k-1}, E_k), D_k = AND(IE_{k mod m}, F_k)
// and G_1 = OR(AND(IE_1, F_1), AND(IE_2, F_1)), so with every initiator FALSE the target is FALSE
// (no compile.pra_to_pc.mixed-units).
[[nodiscard]] auto initiator_chain(const int initiators, const int levels) -> std::string {
    std::ostringstream out;
    out << "G1 + A1 B1\nA1 * IE1 F1\nB1 * IE" << (initiators > 1 ? 2 : 1) << " F1\n";
    for (int k = 2; k <= levels; ++k) {
        out << 'G' << k << " + C" << k << " D" << k << '\n';
        out << 'C' << k << " * G" << (k - 1) << " E" << k << '\n';
        out << 'D' << k << " * IE" << (k % initiators) + 1 << " F" << k << '\n';
    }
    out << "ENDTREE\nPROCESS G" << levels << "\nIMPORT\n";
    for (int i = 1; i <= initiators; ++i) {
        out << "1.0E-02 IE" << i << " I\n";
    }
    for (int k = 1; k <= levels; ++k) {
        out << "0.5 E" << k << "\n1.0E-03 F" << k << '\n';
    }
    return out.str();
}

// RFC 3629 well-formedness (no overlong two-byte forms, no surrogates, at most U+10FFFF).
[[nodiscard]] auto valid_utf8(std::string_view text) -> bool {
    std::size_t index = 0;
    while (index < text.size()) {
        const auto lead = static_cast<unsigned char>(text[index]);
        std::size_t follow = 0;
        std::uint32_t code = 0;
        if (lead < 0x80) {
            ++index;
            continue;
        }
        if (lead >= 0xC2 && lead <= 0xDF) {
            follow = 1;
            code = lead & 0x1FU;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            follow = 2;
            code = lead & 0x0FU;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            follow = 3;
            code = lead & 0x07U;
        } else {
            return false;
        }
        if (index + follow >= text.size()) {
            return false;
        }
        for (std::size_t offset = 1; offset <= follow; ++offset) {
            const auto next = static_cast<unsigned char>(text[index + offset]);
            if ((next & 0xC0U) != 0x80U) {
                return false;
            }
            code = (code << 6U) | (next & 0x3FU);
        }
        if ((follow == 2 && (code < 0x800 || (code >= 0xD800 && code <= 0xDFFF))) ||
            (follow == 3 && (code < 0x10000 || code > 0x10FFFF))) {
            return false;
        }
        index += follow + 1;
    }
    return true;
}

[[nodiscard]] auto config_options(std::string document) -> ftp::options {
    auto options = probability_options();
    options.config = std::move(document);
    return options;
}

[[nodiscard]] auto has_code(diagnostic_list const &diagnostics, std::string_view code) -> bool {
    return std::ranges::any_of(diagnostics.items(), [code](auto const &item) { return item.code == code; });
}

[[nodiscard]] auto read_fixture_text(std::filesystem::path const &path) -> std::string {
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

} // namespace

// docs/ftp-input.md (Inputs): "Several inputs form one project: the files are sorted by path, repeated
// paths dropped, and read as their ordered union", and (fzb inspect) "It takes the same inputs as fzb
// compile ..., so a project can be validated and what the reader understood reviewed before compiling
// it". inspect_project() reads the files in argument order, so with a duplicate import under the
// default policy `last` it reports a different record and value than the one fzb compile uses.
TEST_CASE("fzb inspect reads a multi-file project in sorted path order, as fzb compile does",
          "[bug][ftp][inspect][issue-293]") {
    const auto directory = bug_fixture("inspect_file_order");
    // Argument order b, a. Sorted, the project is a_tree.ftp (X = 0.1) then b_imports.ftp (X = 0.2).
    const std::vector<std::filesystem::path> inputs{directory / "b_imports.ftp", directory / "a_tree.ftp"};

    auto compiled = ftp::compile_project(inputs, probability_options());
    INFO(render(compiled.diagnostics()));
    REQUIRE(compiled);
    // TOP = X * Y with the last record X = 0.2 and Y = 0.5.
    CHECK(std::abs(exact(compiled.value().bytes) - 0.1) < 1e-12);

    const auto inspected = ftp::inspect_project(inputs);
    INFO(render(inspected));
    REQUIRE(inspected.ok());
    CHECK(inspected.files == compiled.value().files);
    const auto value = chosen_value(inspected, "X");
    REQUIRE(value);
    CHECK(*value == 0.2);
}

TEST_CASE("fzb inspect drops a repeated path, as fzb compile does", "[bug][ftp][inspect][issue-293]") {
    const auto file = bug_fixture("inspect_file_order") / "a_tree.ftp";
    const std::vector<std::filesystem::path> inputs{file, file};

    auto compiled = ftp::compile_project(inputs, probability_options());
    INFO(render(compiled.diagnostics()));
    REQUIRE(compiled);
    CHECK(compiled.value().files.size() == 1);

    // Today: io.ftp.duplicate-gate for TOP, and every import is a duplicate import.
    const auto inspected = ftp::inspect_project(inputs);
    INFO(render(inspected));
    CHECK(inspected.ok());
    CHECK(inspected.files.size() == 1);
}

// "Repeated paths dropped": sorted_sources() compares the path strings, so the same file spelled
// `dir/a.ftp` and `dir/./a.ftp` is read twice and every gate is a duplicate gate.
TEST_CASE("fzb compile drops one file named by two spellings of its path", "[bug][ftp][compile][issue-293]") {
    const auto directory = bug_fixture("inspect_file_order");
    const std::vector<std::filesystem::path> inputs{directory / "a_tree.ftp", directory / "." / "a_tree.ftp"};

    auto compiled = ftp::compile_project(inputs, probability_options());
    INFO(render(compiled.diagnostics()));
    REQUIRE(compiled);
    CHECK(compiled.value().files.size() == 1);
}

// docs/ftp-input.md (Grammar, Letter case): "Names are compared without regard to ASCII case: g1, G1 and
// PROCESS g1 name one gate." The file's `PROCESS top` and `0.1 x` resolve to TOP and X, but the same
// spellings given to --target, --watch and --override are compared exactly (pra::model::find), so
// they name nothing.
TEST_CASE("--target, --watch and --override compare FTP names without regard to case",
          "[bug][ftp][compile][issue-293]") {
    constexpr auto text = "TOP + G1 X\nG1 * Y Z\nENDTREE\nPROCESS top\nIMPORT\n0.1 X\n0.2 Y\n0.3 Z\n";

    SECTION("--target") {
        auto options = probability_options();
        options.targets = {"g1"};
        auto compiled = compile_text(text, options);
        INFO(render(compiled.diagnostics()));
        REQUIRE(compiled);
        CHECK(query_names(compiled.value().bytes) == std::vector<std::string>{"G1"});
        CHECK(std::abs(exact(compiled.value().bytes) - 0.06) < 1e-12);
    }
    SECTION("--watch") {
        auto options = probability_options();
        options.watch = {"g1"};
        auto compiled = compile_text(text, options);
        INFO(render(compiled.diagnostics()));
        REQUIRE(compiled);
        CHECK(query_names(compiled.value().bytes) == std::vector<std::string>{"TOP", "G1"});
    }
    SECTION("--override") {
        auto options = probability_options();
        options.overrides = {ftp::override_spec{.name = "x", .value = true}};
        auto compiled = compile_text(text, options);
        INFO(render(compiled.diagnostics()));
        REQUIRE(compiled);
        CHECK(std::abs(exact(compiled.value().bytes) - 1.0) < 1e-12);
    }
}

// docs/ftp-input.md (Options, -c): `0` and `1` are both valid levels, and the default initiator reading is
// `per-year` (#243). #285: a model with 10 initiating events in its target's cone compiles as
// probabilities (~1e5 circuit nodes) but fails under the default frequency reading at -c 0, because the
// lowering duplicates every initiator-dependent gate once per initiator (6.4e5 nodes) and the written
// file exceeds the FlatBuffer verifier's default table limit (1e6 tables), so the writer's own check
// rejects it with io.pcfb.invalid-output. The R1 ir::pc limit is 2^32 - 2 nodes
// (docs/IR-PC-REQUIREMENTS.md section 2); the compile should succeed, or be refused before writing
// with a capacity diagnostic that names the limit.
TEST_CASE("a frequency-mode compile at -c 0 of a 6.4e5-node lowering writes a valid .pcfb (#285)",
          "[bug][ftp][frequency][pcfb][issue-285]") {
    const auto text = initiator_chain(10, 20000);

    {
        // The same model with initiator values as probabilities: about 1e5 nodes, written fine.
        auto as_probabilities = probability_options();
        as_probabilities.level = 0;
        auto baseline = compile_text(text, as_probabilities);
        INFO(render(baseline.diagnostics()));
        REQUIRE(baseline);
    }

    ftp::options frequencies;
    frequencies.level = 0; // initiator_values unset: per-year, the default
    auto compiled = compile_text(text, frequencies);
    INFO(render(compiled.diagnostics()));
    CHECK(compiled.diagnostics().has_errors() == false);
    REQUIRE(compiled);
    CHECK(compiled.value().lowering.conditioned_targets == 10);
}

// docs/ftp-input.md (fzb inspect): `structure.limit` is "the LIMIT value, null when absent", and the reader
// accepts the values of the Grammar (`1.0D-12`, `+1E-12`, and `1E-400` read as 0). summary.cpp parses the
// recorded text again with io::from_chars, which takes none of these, so a LIMIT the reader accepted and
// records under `conventions` is reported as absent.
TEST_CASE("fzb inspect reports every LIMIT value the reader accepts", "[bug][ftp][inspect][issue-294]") {
    for (const auto *limit : {"1.0D-12", "+1E-12", "1E-400"}) {
        INFO("LIMIT " << limit);
        const std::vector<reader::source_text> sources{reader::source_text{
            .path = "m.ftp",
            .text = std::string("TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n0.1 A\nLIMIT ") + limit + "\n"}};
        const auto inspected = ftp::inspect_sources(sources);
        INFO(render(inspected));
        REQUIRE(inspected.ok());
        REQUIRE(inspected.conventions.limit == std::optional<std::string>(limit));
        CHECK(inspected.structure->limit.has_value());
    }
}

// RFC 8259 section 8.1: JSON exchanged between systems "MUST be encoded using UTF-8". FTP names and header
// lines may hold any byte above 127 (docs/ftp-input.md, Grammar), and json::dump copies them unchanged, so
// `fzb inspect --json`, `fzb compile --report` and the Python read_project / compile(report=True), which
// decode that text as UTF-8, get invalid UTF-8 from a Latin-1 export.
TEST_CASE("inspect and compile reports are valid UTF-8 for Latin-1 names", "[bug][ftp][json][issue-294]") {
    // 0xE9 is e-acute in Latin-1; alone it is not UTF-8.
    const std::string text =
        "**FTITLE: Syst\xE8me\nSYST\xC8ME + POMPE-\xC9\nENDTREE\nPROCESS SYST\xC8ME\nIMPORT\n0.1 POMPE-\xC9\n";
    const std::vector<reader::source_text> sources{reader::source_text{.path = "m.ftp", .text = text}};

    const auto inspected = ftp::inspect_sources(sources);
    INFO(render(inspected));
    REQUIRE(inspected.ok());
    CHECK(valid_utf8(ftp::json::dump(ftp::inspection_json(inspected))));

    const auto compiled = ftp::compile_sources(sources, probability_options());
    INFO(render(compiled.diagnostics()));
    REQUIRE(compiled);
    CHECK(valid_utf8(ftp::report_json(compiled)));
}

// docs/ftp-input.md (Config document): "Every key is optional and unknown keys are an error
// (compile.ftp.invalid-config)", and `kinds` is "\"probability\", \"hull\" or an array of them". Target and
// hull objects reject unknown keys; override entries of the array form do not, so a misspelt `kind` is
// dropped silently. A `kinds` string is split at commas (the CLI flag's syntax), so "probability,hull" is
// accepted as both kinds.
TEST_CASE("the config document rejects unknown override keys and a comma list as kinds",
          "[bug][ftp][config][issue-294]") {
    constexpr auto text = "TOP + P Q\nENDTREE\nPROCESS TOP\nIMPORT\n0.1 P\n0.2 Q\n";
    SECTION("override entry with an unknown key") {
        const auto compiled =
            compile_text(text, config_options(R"({"overrides": [{"name": "P", "value": 1, "knd": "gate"}]})"));
        INFO(render(compiled.diagnostics()));
        CHECK(has_code(compiled.diagnostics(), "compile.ftp.invalid-config"));
    }
    SECTION("kinds as a comma-separated string") {
        auto options = config_options(R"({"kinds": "probability,hull", "hull": {"literals": "plain"}})");
        options.kinds.clear();
        const auto compiled = compile_text(text, options);
        INFO(render(compiled.diagnostics()));
        CHECK(has_code(compiled.diagnostics(), "compile.ftp.invalid-config"));
    }
}

// docs/ftp-input.md (Config document): "a target's `hull` replaces the top-level one". The driver merges the
// two key by key (compile.cpp, `add`), so a target whose own hull object sets only `literals` still gets
// the top-level `max_initiators` and its query is ANDed with the initiator rule.
TEST_CASE("a target's hull object replaces the top-level hull object", "[bug][ftp][config][issue-294]") {
    const auto text = read_fixture_text(std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "conformance" /
                                        "initiator_rule" / "model.ftp");
    ftp::options options;
    options.config = R"({"initiator_values": "probability", "kinds": "hull",
                         "hull": {"literals": "plain", "max_initiators": 1},
                         "targets": [{"name": "TOP", "hull": {"literals": "plain"}}]})";
    const auto compiled = compile_text(text, options);
    INFO(render(compiled.diagnostics()));
    REQUIRE(compiled);
    REQUIRE(compiled.value().targets.size() == 1);
    CHECK_FALSE(compiled.value().targets[0].hull_max_initiators.has_value());
}

// docs/ftp-input.md (Inputs): "A file is FTP when it ends in .ftp (any case), or is text (no NUL byte, not XML)
// with a line ENDTREE", and (Grammar) keywords are "read in any letter case" with `;` comments and ignored
// trailing tokens. Content detection reads only the first 64 KiB and wants a line exactly `ENDTREE`, so a
// `.txt` project written `endtree`, `ENDTREE ; end`, or with its tree section longer than 64 KiB is handed
// to the MEF reader instead.
TEST_CASE("content detection finds every ENDTREE line the reader accepts", "[bug][ftp][detect][issue-293]") {
    const auto scratch = std::filesystem::temp_directory_path() / "fuzzybits-compile-ftp-bug-detect";
    std::filesystem::remove_all(scratch);
    std::filesystem::create_directories(scratch);
    const auto write = [](std::filesystem::path const &path, std::string const &text) {
        std::ofstream(path, std::ios::binary) << text;
    };
    std::string long_tree = "TOP + A0\n";
    for (int index = 0; index < 12000; ++index) {
        long_tree += "  A" + std::to_string(index) + "\n";
    }
    long_tree += "ENDTREE\nPROCESS TOP\nIMPORT\n";
    for (int index = 0; index < 12000; ++index) {
        long_tree += "0.001 A" + std::to_string(index) + "\n";
    }
    REQUIRE(long_tree.find("ENDTREE") > 64 * 1024);
    write(scratch / "lower.txt", "TOP + A\nendtree\nPROCESS TOP\nIMPORT\n0.1 A\n");
    write(scratch / "comment.txt", "TOP + A\nENDTREE ; end of tree\nPROCESS TOP\nIMPORT\n0.1 A\n");
    write(scratch / "long.txt", long_tree);

    for (const auto *name : {"lower.txt", "comment.txt", "long.txt"}) {
        INFO(name);
        const std::vector<std::filesystem::path> inputs{scratch / name};
        // The reader takes each of them.
        auto compiled = ftp::compile_project(inputs, probability_options());
        INFO(render(compiled.diagnostics()));
        REQUIRE(compiled);
        CHECK(ftp::classify_inputs(inputs) == ftp::input_kind::ftp);
    }
    std::filesystem::remove_all(scratch);
}

// docs/ftp-input.md (fzb inspect): `names_not_valid_xml_ncname` counts "element names that are not XML
// NCNames". XML NCName start characters include U+00C0..U+00D6 and the rest of the Unicode letters, but
// is_ncname() (summary.cpp) admits only ASCII letters and `_`, so the NCName `É1` is counted.
TEST_CASE("names_not_valid_xml_ncname does not count non-ASCII NCNames", "[bug][ftp][inspect][issue-294]") {
    const std::vector<reader::source_text> sources{
        reader::source_text{.path = "m.ftp",
                            .text = "TOP + \xC3\x89"
                                    "1\nENDTREE\nPROCESS TOP\nIMPORT\n0.1 \xC3\x89"
                                    "1\n"}};
    const auto inspected = ftp::inspect_sources(sources);
    INFO(render(inspected));
    REQUIRE(inspected.ok());
    CHECK(inspected.structure->names_not_ncname == 0);
}

// Cross-format: the Latin-1 names of an FTP project are copied into the .pcfb unchanged. FlatBuffers strings are
// UTF-8, and the readers decode them as such: Python fuzzybits.inspect and fuzzybits.run raise UnicodeDecodeError on
// this file, `fzb run --json` prints invalid UTF-8, and nombrilo export writes a safetensors header that is not
// UTF-8 JSON. Expected: every string written into the .pcfb is valid UTF-8 (the names transcoded, or the project
// refused with a diagnostic that names the encoding).
TEST_CASE("fzb compile writes only UTF-8 strings into the .pcfb", "[bug][ftp][pcfb][utf8][issue-300]") {
    const std::string text =
        "**FTITLE: Syst\xE8me\nSYST\xC8ME + POMPE-\xC9 B\nENDTREE\nPROCESS SYST\xC8ME\nIMPORT\n0.1 POMPE-\xC9\n0.2 B\n";
    const auto compiled = compile_text(text, probability_options());
    INFO(render(compiled.diagnostics()));
    if (!compiled) {
        SUCCEED("refused");
        return;
    }
    auto circuit = ::pcfb::io::from_bytes(compiled.value().bytes);
    REQUIRE(circuit);
    auto const *sub = (*circuit)->subcircuits()->Get(0);
    std::vector<std::string> strings;
    for (auto const *op : *sub->operators()) {
        if (op->name() != nullptr) {
            strings.push_back(op->name()->str());
        }
        if (auto const *sample = op->args_as_SampleArgs(); sample != nullptr && sample->name() != nullptr) {
            strings.push_back(sample->name()->str());
        }
    }
    for (auto const *query : *sub->queries()) {
        if (query->name() != nullptr) {
            strings.push_back(query->name()->str());
        }
    }
    for (auto const *entry : *(*circuit)->metadata()) {
        if (auto const *value = entry->value_as_StringValue(); value != nullptr && value->value() != nullptr) {
            strings.push_back(value->value()->str());
        }
    }
    REQUIRE_FALSE(strings.empty());
    for (auto const &string : strings) {
        INFO(string);
        CHECK(valid_utf8(string));
    }
}
