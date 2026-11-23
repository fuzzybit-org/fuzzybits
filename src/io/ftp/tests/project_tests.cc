#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "io/formats.h"
#include "io/ftp/descriptor.h"
#include "io/ftp/summary.h"
#include "ir/pc/circuit.h"
#include "ir/pra/model.h"

#include "json.h"
#include "structure.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

    namespace ftp = fuzzybits::io::backend::fuzzybits::ftp;
    namespace pra = fuzzybits::ir::pra;
    using fuzzybits::api::diagnostic_severity;

    [[nodiscard]] auto diagnosticsText(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
        std::ostringstream output;
        for (auto const &diagnostic : diagnostics.items()) {
            output << diagnostic.code << ": " << diagnostic.message << '\n';
        }
        return output.str();
    }

    [[nodiscard]] auto sources(std::initializer_list<std::pair<std::string_view, std::string_view>> files) -> std::vector<ftp::source_text> {
        std::vector<ftp::source_text> out;
        for (auto const &[path, text] : files) {
            out.push_back(ftp::source_text{.path = std::string(path), .text = std::string(text)});
        }
        return out;
    }

    [[nodiscard]] auto readText(std::string_view text, fuzzybits::api::option_set const &options = {}) -> fuzzybits::io::read_result {
        return ftp::read_project(sources({{"m.ftp", text}}), options);
    }

    [[nodiscard]] auto withPolicy(std::string policy) -> fuzzybits::api::option_set {
        fuzzybits::api::option_set options;
        options.set(ftp::duplicate_import_option(), std::move(policy));
        return options;
    }

    [[nodiscard]] auto count(fuzzybits::api::diagnostic_list const &diagnostics, const diagnostic_severity severity, std::string_view code,
                             std::string_view fragment = {}) -> std::size_t {
        return static_cast<std::size_t>(std::ranges::count_if(diagnostics.items(), [&](auto const &item) {
            return item.severity == severity && item.code == code && item.message.find(fragment) != std::string::npos;
        }));
    }

    [[nodiscard]] auto requireModel(fuzzybits::io::read_result const &read) -> pra::model const & {
        INFO(diagnosticsText(read.diagnostics()));
        REQUIRE(read);
        auto const *model = pra::model_from(read.value());
        REQUIRE(model != nullptr);
        return *model;
    }

    [[nodiscard]] auto attribute(pra::model const &model, std::span<pra::attribute const> attributes, std::string_view name)
        -> std::vector<std::string> {
        std::vector<std::string> out;
        for (auto const &item : attributes) {
            if (model.text(item.name) == name) {
                out.emplace_back(model.text(item.value));
            }
        }
        return out;
    }

    [[nodiscard]] auto firstSource(pra::model const &model, const pra::domain_ref ref) -> std::pair<std::string, std::uint32_t> {
        const auto where = model.sources(model.element_of(ref));
        REQUIRE_FALSE(where.empty());
        return {std::string(model.text(model.source_files()[where.front().file.raw()].path)), where.front().line};
    }

    constexpr std::string_view kSimple = R"ftp(**CHAR32
**DBNAME: unit
TOP + G1 -G2
G1 * A B
G2 2 A B
    C
ENDTREE
PROCESS TOP
IMPORT
  1.0E-01 A
  2.0E-01 B
  3.0E-01 C
LIMIT 1.0E-09
**FTITLE: simple
*XEQ
)ftp";

} // namespace

TEST_CASE("FTP project reader descriptor produces ir.pra.model", "[io][ftp][project]") {
    const auto descriptor = ftp::project_reader_descriptor();
    REQUIRE(descriptor.endpoint == ftp::project_reader_endpoint_id());
    REQUIRE(descriptor.input_format == fuzzybits::io::formats::ftp());
    REQUIRE(descriptor.produces.contains(fuzzybits::api::capability_id("ir.pra.model")));
    REQUIRE(descriptor.options.entries.size() == 3);
    REQUIRE(descriptor.options.entries[0].key == ftp::duplicate_import_option());
    REQUIRE(std::get<std::string>(descriptor.options.entries[0].default_value) == "last");
    REQUIRE(descriptor.options.entries[1].key == ftp::complement_events_option());
    REQUIRE(std::get<std::string>(descriptor.options.entries[1].default_value) == "atom");
    REQUIRE(descriptor.options.entries[2].key == ftp::max_initiators_option());
    REQUIRE(std::get<std::string>(descriptor.options.entries[2].default_value) == "1");
}

TEST_CASE("FTP project reader builds gates, events, targets and attributes", "[io][ftp][project]") {
    const auto read = readText(kSimple);
    auto const &model = requireModel(read);
    REQUIRE(read.diagnostics().empty());

    REQUIRE(model.gates().size() == 3);
    REQUIRE(model.basic_events().size() == 3);
    REQUIRE(model.initiating_events().empty());
    const auto top = model.find_gate("TOP");
    REQUIRE(model.gates()[top.raw()].kind == pra::gate_kind::or_gate);
    const auto topOperands = model.operands(top);
    REQUIRE(topOperands.size() == 2);
    REQUIRE_FALSE(topOperands[0].complement);
    REQUIRE(topOperands[1].complement);
    REQUIRE(topOperands[1].event == pra::domain_ref(model.find_gate("G2")));

    const auto vote = model.find_gate("G2");
    REQUIRE(model.gates()[vote.raw()].kind == pra::gate_kind::cardinality);
    REQUIRE(model.gates()[vote.raw()].bounds.low == 2);
    REQUIRE(model.gates()[vote.raw()].bounds.high == 3);
    REQUIRE(model.operands(vote).size() == 3);

    const auto c = model.find_basic_event("C");
    auto const &expression = model.expressions()[model.basic_events()[c.raw()].expression.raw()];
    REQUIRE(expression.kind == pra::expression_kind::literal);
    REQUIRE(expression.quantity == pra::quantity_kind::probability);
    REQUIRE(expression.value == 0.3);

    REQUIRE(model.targets().size() == 1);
    REQUIRE(model.targets()[0].kind == pra::domain_target_kind::gate);
    REQUIRE(model.targets()[0].role == pra::target_role::primary);
    REQUIRE(model.target_event(pra::domain_target_id(0)) == pra::domain_ref(top));

    REQUIRE(firstSource(model, top) == std::pair<std::string, std::uint32_t>{"m.ftp", 3});
    REQUIRE(firstSource(model, c) == std::pair<std::string, std::uint32_t>{"m.ftp", 12});
    REQUIRE(firstSource(model, pra::domain_target_id(0)) == std::pair<std::string, std::uint32_t>{"m.ftp", 8});
    REQUIRE(model.text(model.source_files()[0].format) == "ftp");

    REQUIRE(attribute(model, model.project_attributes(), "ftp.limit") == std::vector<std::string>{"1.0E-09"});
    REQUIRE(attribute(model, model.project_attributes(), "ftp.header") ==
            std::vector<std::string>{"**CHAR32", "**DBNAME: unit", "**FTITLE: simple", "*XEQ"});
    REQUIRE(attribute(model, model.project_attributes(), "ftp.negation") == std::vector<std::string>{"delete_term"});
    REQUIRE(attribute(model, model.project_attributes(), "ftp.max_initiators_per_cutset") == std::vector<std::string>{"1"});
}

TEST_CASE("FTP -EVENT operands are marked literal, -GATE operands are not", "[io][ftp][project]") {
    constexpr std::string_view text = R"ftp(TOP + G1 -G2 -C
G1 * A -B
G2 * -A C
ENDTREE
PROCESS TOP
IMPORT
  1.0E-01 A
  2.0E-01 B
  3.0E-01 C
)ftp";
    const auto marks = [](pra::model const &model, std::string_view gate) {
        std::vector<std::pair<bool, bool>> out;
        for (auto const &operand : model.operands(model.find_gate(gate))) {
            out.emplace_back(operand.complement, operand.literal);
        }
        return out;
    };
    using marking = std::vector<std::pair<bool, bool>>;

    SECTION("atom (default)") {
        const auto read = readText(text);
        auto const &model = requireModel(read);
        REQUIRE(marks(model, "TOP") == marking{{false, false}, {true, false}, {true, true}});
        REQUIRE(marks(model, "G1") == marking{{false, false}, {true, true}});
        REQUIRE(marks(model, "G2") == marking{{true, true}, {false, false}});
        REQUIRE(model.summary().complemented_operand_count == 4);
        REQUIRE(model.summary().literal_operand_count == 3);
    }
    SECTION("explicit atom") {
        fuzzybits::api::option_set options;
        options.set(ftp::complement_events_option(), std::string("atom"));
        REQUIRE(requireModel(readText(text, options)).summary().literal_operand_count == 3);
    }
    SECTION("negation leaves every complement unmarked") {
        fuzzybits::api::option_set options;
        options.set(ftp::complement_events_option(), std::string("negation"));
        const auto read = readText(text, options);
        auto const &model = requireModel(read);
        REQUIRE(marks(model, "TOP") == marking{{false, false}, {true, false}, {true, false}});
        REQUIRE(marks(model, "G2") == marking{{true, false}, {false, false}});
        REQUIRE(model.summary().complemented_operand_count == 4);
        REQUIRE(model.summary().literal_operand_count == 0);
    }
    SECTION("other values are rejected") {
        fuzzybits::api::option_set options;
        options.set(ftp::complement_events_option(), std::string("literal"));
        const auto read = readText(text, options);
        REQUIRE_FALSE(read);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-option", "io.ftp.complement_events") == 1);
    }
}

TEST_CASE("FTP conventions are recorded as model attributes", "[io][ftp][project]") {
    const auto initiators = [](fuzzybits::api::option_set const &options) {
        const auto read = readText(kSimple, options);
        auto const &model = requireModel(read);
        REQUIRE(attribute(model, model.project_attributes(), "ftp.negation") == std::vector<std::string>{"delete_term"});
        return attribute(model, model.project_attributes(), "ftp.max_initiators_per_cutset");
    };
    fuzzybits::api::option_set options;
    REQUIRE(initiators(options) == std::vector<std::string>{"1"});
    options.set(ftp::max_initiators_option(), std::string("1"));
    REQUIRE(initiators(options) == std::vector<std::string>{"1"});
    options.set(ftp::max_initiators_option(), std::int64_t{1});
    REQUIRE(initiators(options) == std::vector<std::string>{"1"});
    options.set(ftp::max_initiators_option(), std::string("none"));
    REQUIRE(initiators(options).empty());

    for (auto const &value : {fuzzybits::api::option_value{std::string("2")}, fuzzybits::api::option_value{std::int64_t{2}}}) {
        options.set(ftp::max_initiators_option(), value);
        const auto read = readText(kSimple, options);
        REQUIRE_FALSE(read);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-option", "io.ftp.max_initiators_per_cutset") == 1);
    }
}

TEST_CASE("FTP project reader reads several files as one model", "[io][ftp][project]") {
    const auto project = sources({
        {"a.ftp", "TOP * G1 IE\nENDTREE\nPROCESS TOP\n"},
        {"b.ftp", "G1 + X\nENDTREE\n"},
        {"c.ftp", "IMPORT\n  1.0E-02 IE I\n  5.0E-01 X\n"},
    });
    const auto read = ftp::read_project(project);
    auto const &model = requireModel(read);
    REQUIRE(model.source_files().size() == 3);
    REQUIRE(firstSource(model, model.find_gate("G1")) == std::pair<std::string, std::uint32_t>{"b.ftp", 1});
    REQUIRE(firstSource(model, model.find_basic_event("X")) == std::pair<std::string, std::uint32_t>{"c.ftp", 3});
    REQUIRE(model.operands(model.find_gate("G1")).size() == 1);

    REQUIRE(model.initiating_events().size() == 1);
    REQUIRE(model.initiating_events()[0].event == model.find_basic_event("IE"));
    REQUIRE(model.find_initiating_event("IE").valid());
}

TEST_CASE("FTP project reader expands a directory to its sorted .ftp files", "[io][ftp][project]") {
    const auto directory = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "conformance" / "multi_file_project";
    const auto files = ftp::project_files(std::vector{directory});
    REQUIRE(files);
    REQUIRE(files.value().size() == 3);
    REQUIRE(files.value()[0].filename() == "01_top.ftp");
    REQUIRE(files.value()[2].filename() == "03_imports.ftp");

    const auto read = ftp::read_project(fuzzybits::io::read_request{.path = directory, .format = fuzzybits::io::formats::ftp()});
    auto const &model = requireModel(read);
    REQUIRE(model.gates().size() == 7);
    REQUIRE(model.basic_events().size() == 8);

    const auto missing = ftp::read_project(std::vector{directory / "absent.ftp"});
    REQUIRE(count(missing.diagnostics(), diagnostic_severity::error, "io.ftp.open-failed") == 1);
    const auto empty = ftp::project_files(std::span<std::filesystem::path const>{});
    REQUIRE(count(empty.diagnostics(), diagnostic_severity::error, "io.ftp.no-files") == 1);
    const auto mismatch = ftp::read_project(fuzzybits::io::read_request{.path = directory, .format = fuzzybits::io::formats::aiger()});
    REQUIRE(count(mismatch.diagnostics(), diagnostic_severity::error, "io.ftp.format-mismatch") == 1);
}

TEST_CASE("FTP names keep every character the grammar allows", "[io][ftp][project]") {
    const auto read = readText("1A + X/Y A--B 100\n100 * 7 -A--B\nENDTREE\nPROCESS 1A\nIMPORT\n  1.0E-01 X/Y\n  2.0E-01 A--B\n  3.0E-01 7\n");
    auto const &model = requireModel(read);
    REQUIRE(model.find_gate("1A").valid());
    REQUIRE(model.find_gate("100").valid());
    REQUIRE(model.find_basic_event("X/Y").valid());
    REQUIRE(model.find_basic_event("A--B").valid());
    REQUIRE(model.find_basic_event("7").valid());
    REQUIRE(model.operands(model.find_gate("100"))[1].complement);
}

TEST_CASE("FTP single-input gates, CRLF and tabs are read as written", "[io][ftp][project]") {
    const auto read = readText("TOP *\tONE\r\nONE + A\r\nENDTREE\r\nPROCESS TOP\r\nIMPORT\r\n\t1.0E-01\tA\r\n");
    auto const &model = requireModel(read);
    REQUIRE(model.gates().size() == 2);
    REQUIRE(model.operands(model.find_gate("ONE")).size() == 1);
    REQUIRE(model.find_basic_event("A").valid());
}

TEST_CASE("FTP PROCESS continues on following lines and may name an event", "[io][ftp][project]") {
    const auto read = readText("T1 + A\nT2 * A B\nENDTREE\nPROCESS T1\n  T2 B\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\n");
    auto const &model = requireModel(read);
    REQUIRE(model.targets().size() == 3);
    REQUIRE(model.targets()[2].kind == pra::domain_target_kind::basic_event);
    REQUIRE(model.target_event(pra::domain_target_id(2)) == pra::domain_ref(model.find_basic_event("B")));
}

TEST_CASE("FTP without PROCESS targets the root gates with a warning", "[io][ftp][project]") {
    const auto read = readText("R1 + A G\nR2 * B G\nG + A B\nENDTREE\nIMPORT\n  1.0E-01 A\n  2.0E-01 B\n");
    auto const &model = requireModel(read);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.missing-process") == 1);
    REQUIRE(model.targets().size() == 2);
    REQUIRE(model.stable_name(model.targets()[1].element) == "R2");

    const auto none = readText("IMPORT\n  1.0E-01 A\n");
    REQUIRE_FALSE(none);
    REQUIRE(count(none.diagnostics(), diagnostic_severity::error, "io.ftp.no-targets") == 1);
}

TEST_CASE("FTP duplicate imports follow the duplicate-import policy", "[io][ftp][project]") {
    constexpr std::string_view text = "TOP + A IE\nENDTREE\nPROCESS TOP\nIMPORT\n  3.0E-01 A\n  1.0E-02 IE\n  2.0E-01 A\n  2.0E-02 IE I\n";
    const auto value = [](pra::model const &model, std::string_view name) {
        return ftp_test::literal_value(model, model.find_basic_event(name));
    };

    SECTION("last is the default and warns with every record") {
        const auto read = readText(text);
        auto const &model = requireModel(read);
        REQUIRE(value(model, "A") == 0.2);
        REQUIRE(value(model, "IE") == 0.02);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.duplicate-import", "'A'") == 1);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.duplicate-import", "m.ftp:5 = 3.0E-01, m.ftp:7 = 2.0E-01") == 1);
        REQUIRE(model.initiating_events().size() == 1);
        REQUIRE(firstSource(model, model.find_basic_event("A")) == std::pair<std::string, std::uint32_t>{"m.ftp", 7});
        REQUIRE(model.sources(model.basic_events()[model.find_basic_event("A").raw()].element).size() == 2);
    }
    SECTION("first keeps the first record; an I tag on any record is kept") {
        const auto read = readText(text, withPolicy("first"));
        auto const &model = requireModel(read);
        REQUIRE(value(model, "A") == 0.3);
        REQUIRE(value(model, "IE") == 0.01);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.duplicate-import") == 2);
        REQUIRE(model.initiating_events().size() == 1);
        REQUIRE(model.initiating_events()[0].event == model.find_basic_event("IE"));
    }
    SECTION("error rejects the project") {
        const auto read = readText(text, withPolicy("error"));
        REQUIRE_FALSE(read);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::error, "io.ftp.duplicate-import") == 2);
    }
    SECTION("an unknown policy is an error") {
        const auto read = readText(text, withPolicy("average"));
        REQUIRE_FALSE(read);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-option") == 1);
    }
}

TEST_CASE("FTP initiator values above 1 are frequencies; other values must be probabilities", "[io][ftp][project]") {
    const auto read = readText("TOP * IE A\nENDTREE\nPROCESS TOP\nIMPORT\n  2.5E+00 IE I\n  1.0E-01 A\n");
    auto const &model = requireModel(read);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.initiator-frequency", "'IE'") == 1);
    auto const &expression = model.expressions()[model.basic_events()[model.find_basic_event("IE").raw()].expression.raw()];
    REQUIRE(expression.quantity == pra::quantity_kind::frequency);
    REQUIRE(expression.value == 2.5);

    for (const auto *bad : {"  1.5E+00 A\n", "  -1.0E-01 A\n", "  1.0E+999 A\n", "  -2.0E+00 A I\n"}) {
        const auto rejected = readText(std::string("TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n") + bad);
        INFO(bad);
        REQUIRE_FALSE(rejected);
        REQUIRE(count(rejected.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-probability", "m.ftp:5") == 1);
    }
}

TEST_CASE("FTP unknown import flags are attributes with a warning", "[io][ftp][project]") {
    const auto read = readText("TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A X9 I\n");
    auto const &model = requireModel(read);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.unknown-import-flag", "'X9'") == 1);
    const auto event = model.basic_events()[model.find_basic_event("A").raw()];
    REQUIRE(attribute(model, model.attributes(event.element), "ftp.import_flag") == std::vector<std::string>{"X9"});
    REQUIRE(model.initiating_events().size() == 1);
}

TEST_CASE("FTP structural errors have stable diagnostics", "[io][ftp][project]") {
    struct error_case final {
        std::string_view text;
        std::string_view code;
        std::string_view fragment;
    };
    const auto cases = std::vector<error_case>{
        {"TOP + A\nIMPORT\n  1.0E-01 A\n", "io.ftp.missing-endtree", "m.ftp:1 has no ENDTREE before the IMPORT line at m.ftp:2"},
        {"TOP + A\n", "io.ftp.missing-endtree", "before the end of the file"},
        {"TOP + A\nLIMIT 1.0E-05\n", "io.ftp.missing-endtree", "LIMIT line"},
        {"TOP + A\nENDTREE\nPROCESS TOP\n  1.0E-01 A\n", "io.ftp.import-without-header", "m.ftp:4"},
        {"TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n**FTITLE: x\n  2.0E-01 B\n", "io.ftp.import-without-header", "'B'"},
        {"TOP + G\nENDTREE\nG + A\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n", "io.ftp.misplaced-gate", "'G'"},
        {"TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\nG + A\n", "io.ftp.misplaced-gate", "m.ftp:6"},
        {"TOP + A\nTOP * A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n", "io.ftp.duplicate-gate", "(m.ftp:1, m.ftp:2)"},
        {"TOP +\nENDTREE\nPROCESS TOP\n", "io.ftp.invalid-gate", "no operands"},
        {"TOP 0 A B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  1.0E-01 B\n", "io.ftp.invalid-gate", "at-least-0"},
        {"TOP 3 A B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  1.0E-01 B\n", "io.ftp.invalid-gate", "has 2 operands"},
        {"TOP + G\nG + A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  1.0E-01 G\n", "io.ftp.gate-imported", "'G'"},
        {"TOP + A B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n", "io.ftp.undefined-reference", "'B'"},
        {"TOP + A\nENDTREE\nPROCESS TOP NOPE\nIMPORT\n  1.0E-01 A\n", "io.ftp.undefined-process-target", "'NOPE'"},
        {"TOP x A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n", "io.ftp.invalid-line", "m.ftp:1"},
        {"TOP 99999999999 A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n", "io.ftp.invalid-line", "out of range"},
        {"  A B\nTOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n", "io.ftp.invalid-line", "without an open gate"},
        {"TOP + A $B\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n", "io.ftp.invalid-line", "'$B'"},
        {"TOP + A\nENDTREE now\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n", "io.ftp.invalid-line", "after ENDTREE"},
        {"TOP + A\nENDTREE\nPROCESS TOP $X\nIMPORT\n  1.0E-01 A\n", "io.ftp.invalid-line", "'$X'"},
        {"TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n  0x1 B\n", "io.ftp.invalid-line", "IMPORT section"},
        {"TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\nLIMIT x\n", "io.ftp.invalid-limit", "m.ftp:6"},
    };
    for (auto const &[text, code, fragment] : cases) {
        INFO(text);
        const auto read = readText(text);
        INFO(diagnosticsText(read.diagnostics()));
        REQUIRE_FALSE(read);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::error, code, fragment) == 1);
    }
}

TEST_CASE("FTP duplicate gates are errors across files", "[io][ftp][project]") {
    const auto read = ftp::read_project(sources({
        {"a.ftp", "TOP + G\nG + A\nENDTREE\nPROCESS TOP\n"},
        {"b.ftp", "G * A\nENDTREE\nIMPORT\n  1.0E-01 A\n"},
    }));
    REQUIRE_FALSE(read);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::error, "io.ftp.duplicate-gate", "(a.ftp:2, b.ftp:1)") == 1);
}

TEST_CASE("FTP LIMIT repeated with another value warns", "[io][ftp][project]") {
    const auto read = readText("TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\nLIMIT 1.0E-05\nLIMIT 1.0E-06\n");
    auto const &model = requireModel(read);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.duplicate-limit") == 1);
    REQUIRE(attribute(model, model.project_attributes(), "ftp.limit") == std::vector<std::string>{"1.0E-06"});
}

TEST_CASE("FTP gate cycles are reported", "[io][ftp][project]") {
    constexpr std::string_view text = "TOP + G1\nG1 * G2 A\nG2 + G1 A\nENDTREE\nPROCESS TOP\nIMPORT\n  1.0E-01 A\n";
    const auto read = readText(text);
    REQUIRE_FALSE(read);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::error, "ir.pra.cycle") == 1);
}

TEST_CASE("FTP reader handles a 1e5-gate chain without recursion", "[io][ftp][project][scale]") {
    constexpr std::uint32_t gates = 100'000;
    std::string text;
    text.reserve(gates * 24);
    for (std::uint32_t gate = 0; gate < gates; ++gate) {
        text += "G" + std::to_string(gate) + (gate % 2 == 0 ? " * " : " + ");
        text += gate + 1 < gates ? "G" + std::to_string(gate + 1) + " " : std::string{};
        text += "-E" + std::to_string(gate % 64) + " E" + std::to_string((gate + 7) % 64) + "\n";
    }
    text += "ENDTREE\nPROCESS G0\nIMPORT\n";
    for (std::uint32_t event = 0; event < 64; ++event) {
        text += "  5.0E-01 E" + std::to_string(event) + "\n";
    }

    const auto start = std::chrono::steady_clock::now();
    const auto read = readText(text);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    auto const &model = requireModel(read);
    REQUIRE(model.gates().size() == gates);
    REQUIRE(model.summary().gate_depth == gates);
#ifdef NDEBUG
    REQUIRE(elapsed < 1.0);
#else
    (void)elapsed;
#endif
}

namespace {

    [[nodiscard]] auto fileText(std::filesystem::path const &path) -> std::string {
        std::ifstream input(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    [[nodiscard]] auto cardNames(ftp_test::json const &list) -> std::vector<std::string> {
        std::vector<std::string> out;
        for (auto const &item : list.array()) {
            out.push_back(item.string());
        }
        return out;
    }

    // Field names that differ from the card; never their values (#38).
    [[nodiscard]] auto cardMismatches(ftp::structure_summary const &actual, ftp_test::json const &card)
        -> std::vector<std::string> {
        std::vector<std::string> out;
        const auto number = [](ftp_test::json const &value) { return static_cast<std::size_t>(value.number()); };
        const auto check = [&out](const bool equal, std::string field) {
            if (!equal) {
                out.push_back(std::move(field));
            }
        };
        check(actual.gates == number(card["gates"]), "gates");
        check(actual.or_gates == number(card["gates_by_kind"]["or"]), "gates_by_kind.or");
        check(actual.and_gates == number(card["gates_by_kind"]["and"]), "gates_by_kind.and");
        check(actual.atleast_gates == number(card["gates_by_kind"]["atleast"]), "gates_by_kind.atleast");
        bool kOfN = actual.atleast_k_of_n.size() == card["atleast_k_of_n"].object().size();
        for (auto const &[key, value] : card["atleast_k_of_n"].object()) {
            kOfN = kOfN && actual.atleast_k_of_n.contains(key) && actual.atleast_k_of_n.at(key) == number(value);
        }
        check(kOfN, "atleast_k_of_n");
        check(actual.edges == number(card["edges"]), "edges");
        check(actual.single_input_gates == number(card["single_input_gates"]), "single_input_gates");
        check(actual.max_fan_in == number(card["max_fan_in"]), "max_fan_in");
        check(actual.depth == number(card["depth"]), "depth");
        check(actual.basic_events == number(card["basic_events"]), "basic_events");
        check(actual.import_lines == number(card["import_lines"]), "import_lines");
        check(actual.complemented_edges == number(card["complemented_edges"]), "complemented_edges");
        check(actual.complemented_edges_to_basic_events == number(card["complemented_edges_to_basic_events"]),
              "complemented_edges_to_basic_events");
        check(actual.roots == cardNames(card["roots"]), "roots");
        check(actual.process_targets == cardNames(card["process_targets"]), "process_targets");
        check(actual.initiating_events == number(card["initiator_tagged_events"]), "initiator_tagged_events");
        check(actual.names_not_ncname == number(card["names_not_valid_xml_ncname"]), "names_not_valid_xml_ncname");
        check(actual.duplicate_imports.size() == card["duplicate_imports"].object().size(), "duplicate_imports");
        check(actual.limit.has_value() == !card["limit"].is_null() &&
                  (!actual.limit || *actual.limit == card["limit"].number()),
              "limit");
        return out;
    }

} // namespace

// FUZZYBITS_PRIVATE_FIXTURES_DIR/ftp/<MODEL>/ with <MODEL>.card.json: read the
// project under the card's duplicate-import policy and compare the structure.
// Output is limited to pass/fail, mismatching field names and timings (#38).
TEST_CASE("FTP private projects match their cards", "[io][ftp][private]") {
    const char *root = std::getenv("FUZZYBITS_PRIVATE_FIXTURES_DIR");
    if (root == nullptr || root[0] == '\0' || !std::filesystem::is_directory(std::filesystem::path(root) / "ftp")) {
        SKIP("FUZZYBITS_PRIVATE_FIXTURES_DIR is not set");
    }
    std::vector<std::filesystem::path> projects;
    for (auto const &entry : std::filesystem::directory_iterator(std::filesystem::path(root) / "ftp")) {
        const auto card = entry.path() / (entry.path().filename().string() + ".card.json");
        if (entry.is_directory() && std::filesystem::is_regular_file(card)) {
            projects.push_back(entry.path());
        }
    }
    std::ranges::sort(projects);

    std::size_t passed = 0;
    for (std::size_t index = 0; index < projects.size(); ++index) {
        auto const &directory = projects[index];
        const auto card = ftp_test::parse_json(fileText(directory / (directory.filename().string() + ".card.json")));
        bool ok = card.has_value() && card->find("structure") != nullptr;
        std::string policy = "last";
        if (ok) {
            if (auto const *section = card->find("policy"); section != nullptr && section->find("duplicate_import") != nullptr) {
                const auto &text = (*section)["duplicate_import"].string();
                policy = text.substr(0, text.find_first_of(" ("));
            }
        }

        const auto start = std::chrono::steady_clock::now();
        const auto read = ftp::read_project(std::vector{directory}, withPolicy(policy));
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        std::vector<std::string> mismatches;
        if (!ok || !read) {
            ok = false;
            std::size_t errors = 0;
            for (auto const &item : read.diagnostics().items()) {
                errors += item.severity == diagnostic_severity::error ? 1 : 0;
            }
            std::cout << "private FTP project " << index + 1 << ": FAIL (read failed with " << errors << " errors)\n";
        } else {
            const auto counts = ftp::structure_of(*pra::model_from(read.value()));
            mismatches = cardMismatches(counts, (*card)["structure"]);
            ok = mismatches.empty();
            std::cout << "private FTP project " << index + 1 << ": " << (ok ? "pass" : "FAIL") << " (policy " << policy << ", read "
                      << seconds << " s";
            for (auto const &field : mismatches) {
                std::cout << ", mismatch " << field;
            }
            std::cout << ")\n";
#ifdef NDEBUG
            CHECK(seconds < 2.0);
#endif
        }
        passed += ok ? 1 : 0;
    }
    std::cout << "private FTP projects: " << passed << "/" << projects.size() << " pass\n";
    CHECK(passed == projects.size());
}
