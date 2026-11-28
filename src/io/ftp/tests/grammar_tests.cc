// The FTP line grammar as files in use write it: the name alphabet, case, comments,
// layout, value forms, the -1 value and MAXSIZE. See docs/ftp-input.md ("Grammar").

#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "io/ftp/descriptor.h"
#include "ir/pra/model.h"

#include "structure.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace ftp = fuzzybits::io::backend::fuzzybits::ftp;
namespace pra = fuzzybits::ir::pra;
using fuzzybits::api::diagnostic_severity;
using namespace std::string_literals;

[[nodiscard]] auto diagnosticsText(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
    std::ostringstream output;
    for (auto const &diagnostic : diagnostics.items()) {
        output << diagnostic.code << ": " << diagnostic.message << '\n';
    }
    return output.str();
}

[[nodiscard]] auto readText(std::string_view text, fuzzybits::api::option_set const &options = {}) -> fuzzybits::io::read_result {
    const std::vector<ftp::source_text> sources{ftp::source_text{.path = "m.ftp", .text = std::string(text)}};
    return ftp::read_project(sources, options);
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

[[nodiscard]] auto projectAttribute(pra::model const &model, std::string_view name) -> std::vector<std::string> {
    std::vector<std::string> out;
    for (auto const &item : model.project_attributes()) {
        if (model.text(item.name) == name) {
            out.emplace_back(model.text(item.value));
        }
    }
    return out;
}

[[nodiscard]] auto targetNames(pra::model const &model) -> std::vector<std::string> {
    std::vector<std::string> out;
    for (auto const &target : model.targets()) {
        out.emplace_back(model.stable_name(target.element));
    }
    return out;
}

[[nodiscard]] auto probability(pra::model const &model, std::string_view gate) -> double {
    const auto id = model.find_gate(gate);
    REQUIRE(id.valid());
    return ftp_test::brute_force_probability(model, pra::domain_ref(id));
}

[[nodiscard]] auto value(pra::model const &model, std::string_view event) -> double {
    const auto id = model.find_basic_event(event);
    REQUIRE(id.valid());
    return ftp_test::literal_value(model, id);
}

// The model without its names: gate kinds and bounds, operands by index and complement,
// event values and initiator flags, targets by index.
[[nodiscard]] auto shape(pra::model const &model) -> std::string {
    std::ostringstream out;
    for (std::uint32_t index = 0; index < model.gates().size(); ++index) {
        auto const &gate = model.gates()[index];
        out << 'G' << static_cast<int>(gate.kind) << '/' << gate.bounds.low << ':';
        for (auto const &operand : model.operands(pra::gate_id(index))) {
            out << ' ' << (operand.event.kind() == pra::element_kind::gate ? 'g' : 'e') << operand.event.raw() << (operand.complement ? "-" : "")
                << (operand.literal ? "L" : "");
        }
        out << '\n';
    }
    for (std::uint32_t index = 0; index < model.basic_events().size(); ++index) {
        out << 'E' << ftp_test::literal_value(model, pra::basic_event_id(index)) << '\n';
    }
    for (auto const &initiator : model.initiating_events()) {
        out << 'I' << initiator.event.raw() << '\n';
    }
    for (std::uint32_t index = 0; index < model.targets().size(); ++index) {
        out << 'T' << model.target_event(pra::domain_target_id(index)).raw() << '\n';
    }
    return out.str();
}

void replaceAll(std::string &text, std::string_view from, std::string_view to) {
    for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
}

// Initiating-event names with a '%' prefix, and '~' inside gate names (#240), CRLF.
constexpr std::string_view kPercentTilde = "T + G~1 B-%IE\r\nG~1 * %IE A\r\nB-%IE * %IE -%IE2\r\nENDTREE\r\nPROCESS T\r\n"
                                           "IMPORT\r\n0.1 %IE I\r\n0.2 %IE2 I\r\n0.5 A\r\n";

} // namespace

TEST_CASE("FTP names with % and ~ read as plain names (#240)", "[io][ftp][grammar]") {
    const auto read = readText(kPercentTilde);
    auto const &model = requireModel(read);
    REQUIRE(read.diagnostics().empty());

    const auto gate = model.find_gate("B-%IE");
    REQUIRE(gate.valid());
    REQUIRE(model.find_gate("G~1").valid());
    const auto operands = model.operands(gate);
    REQUIRE(operands.size() == 2);
    REQUIRE(operands[0].event == pra::domain_ref(model.find_basic_event("%IE")));
    REQUIRE_FALSE(operands[0].complement);
    REQUIRE(operands[1].event == pra::domain_ref(model.find_basic_event("%IE2")));
    REQUIRE(operands[1].complement);
    REQUIRE(model.initiating_events().size() == 2);
    // T = %IE (A + not %IE2) = 0.1 (1 - 0.5 * 0.2)
    REQUIRE(std::abs(probability(model, "T") - 0.09) < 1e-15);

    SECTION("the characters change nothing but the names") {
        std::string renamed(kPercentTilde);
        replaceAll(renamed, "%", "P_");
        replaceAll(renamed, "~", "_");
        const auto plain = readText(renamed);
        auto const &plainModel = requireModel(plain);
        REQUIRE(plain.diagnostics().empty());
        REQUIRE(shape(plainModel) == shape(model));
        REQUIRE(model.find_gate("G~1").raw() == plainModel.find_gate("G_1").raw());
        REQUIRE(model.find_basic_event("%IE2").raw() == plainModel.find_basic_event("P_IE2").raw());
    }
}

TEST_CASE("FTP names are any token except a lone operator, a leading '-' or ';'", "[io][ftp][grammar]") {
    for (const char character : "$#.()[]:@!?'\"=<>,|\\^{}*&+%~`/_\xE9"s) {
        const std::string c(1, character);
        // As the middle of a name, and as its first character (not at the start of a line).
        const auto text = "TOP + A" + c + "B " + c + "C\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 A" + c + "B\n  0.2 " + c + "C\n";
        INFO(text);
        const auto read = readText(text);
        auto const &model = requireModel(read);
        REQUIRE(read.diagnostics().empty());
        REQUIRE(model.find_basic_event("A" + c + "B").valid());
        REQUIRE(model.find_basic_event(c + "C").valid());
        REQUIRE(std::abs(probability(model, "TOP") - 0.28) < 1e-15);
    }

    SECTION("a leading '-' complements, so '--A' is not a reference") {
        const auto read = readText("TOP + --A B\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 B\n");
        REQUIRE_FALSE(read);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-line", "'--A'") == 1);
    }
    SECTION("a lone '+' or '*' is an operator, never an operand") {
        const auto read = readText("TOP + A\n  + B\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 A\n  0.2 B\n");
        REQUIRE_FALSE(read);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-line", "'+'") == 1);
    }
    SECTION("an operator must stand alone: '+A' is not an OR of A") {
        const auto read = readText("TOP +A B\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 A\n  0.2 B\n");
        REQUIRE_FALSE(read);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-line", "m.ftp:1") == 1);
    }
}

TEST_CASE("FTP keywords and the I flag ignore case", "[io][ftp][grammar]") {
    // The layout of exported benchmark trees: lower-case keywords, indented records.
    const auto read = readText("root1 *  G4 B5\nG4 + B6 B7\nendtree\nProcess root1\nimport\n"
                               "  0.5 B5\n  0.25 B6 i\n  0.5 B7\nLimit 1E-300\nmaxSize 3\n");
    auto const &model = requireModel(read);
    REQUIRE(read.diagnostics().empty());
    REQUIRE(targetNames(model) == std::vector<std::string>{"root1"});
    REQUIRE(model.initiating_events().size() == 1);
    REQUIRE(model.initiating_events()[0].event == model.find_basic_event("B6"));
    REQUIRE(projectAttribute(model, "ftp.limit") == std::vector<std::string>{"1E-300"});
    REQUIRE(projectAttribute(model, "ftp.maxsize") == std::vector<std::string>{"3"});
    // 0.5 * (1 - 0.75 * 0.5)
    REQUIRE(std::abs(probability(model, "root1") - 0.3125) < 1e-15);
}

TEST_CASE("FTP names ignore ASCII case and keep their declared spelling", "[io][ftp][grammar]") {
    const auto read = readText("TOP + a g1\nG1 * A c\ng2 + A\nENDTREE\nPROCESS top G2\nIMPORT\n  0.1 A\n  0.2 C\n");
    auto const &model = requireModel(read);
    REQUIRE(model.gates().size() == 3);
    REQUIRE(model.basic_events().size() == 2);
    REQUIRE(model.find_basic_event("A").valid());
    REQUIRE_FALSE(model.find_basic_event("a").valid());
    REQUIRE(model.find_gate("G1").valid());
    REQUIRE(model.find_gate("g2").valid());
    REQUIRE(targetNames(model) == std::vector<std::string>{"TOP", "g2"});
    REQUIRE(model.operands(model.find_gate("TOP"))[1].event == pra::domain_ref(model.find_gate("G1")));
    // TOP = A + A C
    REQUIRE(std::abs(probability(model, "TOP") - 0.1) < 1e-15);

    // One warning per element spelled another way, naming both spellings and places.
    const auto &diagnostics = read.diagnostics();
    INFO(diagnosticsText(diagnostics));
    REQUIRE(diagnostics.size() == 5);
    REQUIRE(count(diagnostics, diagnostic_severity::warning, "io.ftp.name-case", "'a' is read as 'A' (m.ftp:7)") == 1);
    REQUIRE(count(diagnostics, diagnostic_severity::warning, "io.ftp.name-case", "'g1' is read as 'G1' (m.ftp:2)") == 1);
    REQUIRE(count(diagnostics, diagnostic_severity::warning, "io.ftp.name-case", "'c' is read as 'C' (m.ftp:8)") == 1);
    REQUIRE(count(diagnostics, diagnostic_severity::warning, "io.ftp.name-case", "m.ftp:5: 'top' is read as 'TOP'") == 1);
    REQUIRE(count(diagnostics, diagnostic_severity::warning, "io.ftp.name-case", "'G2' is read as 'g2'") == 1);

    SECTION("spellings of one name are one gate, one event or a clash") {
        const auto gates = readText("TOP + G1\nG1 * A\ng1 + A\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 A\n");
        REQUIRE_FALSE(gates);
        REQUIRE(count(gates.diagnostics(), diagnostic_severity::error, "io.ftp.duplicate-gate", "(m.ftp:2, m.ftp:3)") == 1);

        const auto imported = readText("TOP + G1\nG1 + A\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 A\n  0.2 g1\n");
        REQUIRE_FALSE(imported);
        REQUIRE(count(imported.diagnostics(), diagnostic_severity::error, "io.ftp.gate-imported", "'G1'") == 1);

        const auto twice = readText("TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 A\n  0.3 a\n");
        auto const &last = requireModel(twice);
        REQUIRE(value(last, "A") == 0.3);
        REQUIRE(count(twice.diagnostics(), diagnostic_severity::warning, "io.ftp.duplicate-import", "'A'") == 1);
        REQUIRE(count(twice.diagnostics(), diagnostic_severity::warning, "io.ftp.name-case", "'a' is read as 'A'") == 1);
    }
    SECTION("only ASCII letters fold") {
        const auto read = readText("TOP * \xE9 \xC9\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 \xE9\n  0.2 \xC9\n");
        auto const &model = requireModel(read);
        REQUIRE(read.diagnostics().empty());
        REQUIRE(model.basic_events().size() == 2);
    }
}

TEST_CASE("FTP ';' starts a comment except in header lines", "[io][ftp][grammar]") {
    const auto read = readText("; a comment before the tree\n"
                               "TOP + A C ; D is not an operand\n"
                               "G2 * A C;D\n"
                               "  ; a comment line inside the operand list\n"
                               "  E\n"
                               "ENDTREE ; end of the tree\n"
                               "PROCESS TOP G2 ; X\n"
                               "IMPORT ; records\n"
                               "  0.1 A ; note\n"
                               "  0.2 C\n"
                               "  0.3 E\n"
                               "**DBNAME: C:\\models\\a;b.rr\n");
    auto const &model = requireModel(read);
    REQUIRE(read.diagnostics().empty());
    REQUIRE(model.operands(model.find_gate("TOP")).size() == 2);
    REQUIRE(model.operands(model.find_gate("G2")).size() == 3);
    REQUIRE(targetNames(model) == std::vector<std::string>{"TOP", "G2"});
    REQUIRE(projectAttribute(model, "ftp.header") == std::vector<std::string>{"**DBNAME: C:\\models\\a;b.rr"});
}

TEST_CASE("FTP indentation does not decide between a gate line and a continuation", "[io][ftp][grammar]") {
    // `NAME op ...` defines a gate wherever it starts; any other line continues the open gate.
    const auto read = readText("TOP + A\nC\n  G2 * A C\n    G3\nG3 2 A C\nENDTREE\nPROCESS TOP G2\nIMPORT\n"
                               "  0.1 A\n  0.2 C\n");
    auto const &model = requireModel(read);
    REQUIRE(read.diagnostics().empty());
    REQUIRE(model.operands(model.find_gate("TOP")).size() == 2);
    REQUIRE(model.operands(model.find_gate("G2")).size() == 3);
    REQUIRE(std::abs(probability(model, "G2") - 0.1 * 0.2) < 1e-15);

    SECTION("a line that is neither, with no open gate, is an error") {
        const auto stray = readText("ENDTREE\n  A B\nPROCESS TOP\n");
        REQUIRE_FALSE(stray);
        REQUIRE(count(stray.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-line", "continuation line without an open gate") == 1);
    }
}

TEST_CASE("FTP values take every decimal form, including a D exponent", "[io][ftp][grammar]") {
    const std::vector<std::pair<std::string_view, double>> accepted{
        {"1.", 1.0},    {".5", 0.5},    {"5.E-1", 0.5}, {"5.e-1", 0.5}, {"1.0D-01", 0.1}, {"1.0d-01", 0.1}, {"+.5", 0.5},
        {"5E-01", 0.5}, {"000.5", 0.5}, {"1", 1.0},     {"0", 0.0},     {"-0.0", 0.0},    {"1E-400", 0.0},  {"1.000000E-02", 0.01},
    };
    for (auto const &[text, expected] : accepted) {
        INFO(text);
        const auto read = readText("TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  " + std::string(text) + " A\nLIMIT " + std::string(text) + "\n");
        auto const &model = requireModel(read);
        REQUIRE(read.diagnostics().empty());
        REQUIRE(value(model, "A") == expected);
    }
    for (const auto *text : {"5E", ".", "0,5", "0x1", "inf", "nan", "1.0E-1.5", "10%", "1.0Q-1", "E5"}) {
        INFO(text);
        const auto read = readText("TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 A\n  " + std::string(text) + " B\n");
        REQUIRE_FALSE(read);
        REQUIRE(count(read.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-line", "IMPORT section") == 1);
    }
    const auto overflow = readText("TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  1E400 A\n");
    REQUIRE_FALSE(overflow);
    REQUIRE(count(overflow.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-probability", "'A'") == 1);
}

TEST_CASE("FTP value -1 is probability 1; other negative values are errors", "[io][ftp][grammar]") {
    // Flag events written with -1, as in exported plant models.
    const auto read = readText("TOP + F1 F2\nF1 * A X1\nF2 + X2 -X3\nENDTREE\nPROCESS TOP\nIMPORT\n"
                               "  1.0E-01 A\n  -1.000000E+00 X1\n  -1 X2 I\n  -1.0 X3\n");
    auto const &model = requireModel(read);
    for (const auto *name : {"X1", "X2", "X3"}) {
        REQUIRE(value(model, name) == 1.0);
    }
    REQUIRE(model.initiating_events().size() == 1);
    REQUIRE(read.diagnostics().size() == 3);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.minus-one-value",
                  "m.ftp:8: value -1.000000E+00 of 'X1' is read as probability 1") == 1);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.minus-one-value", "'X2'") == 1);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.minus-one-value", "'X3'") == 1);
    // F2 is TRUE through X2 alone.
    REQUIRE(std::abs(probability(model, "TOP") - 1.0) < 1e-15);
    REQUIRE(std::abs(probability(model, "F1") - 0.1) < 1e-15);

    for (const auto *bad : {"-0.5", "-2", "-1.0E-03", "-0.999999", "-1.000001"}) {
        INFO(bad);
        const auto rejected = readText(std::string("TOP + A\nENDTREE\nPROCESS TOP\nIMPORT\n  ") + bad + " A\n");
        REQUIRE_FALSE(rejected);
        REQUIRE(count(rejected.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-probability", "'A'") == 1);
    }
}

TEST_CASE("FTP MAXSIZE outside the tree section is recorded like LIMIT", "[io][ftp][grammar]") {
    constexpr std::string_view tree = "TOP + A G1\nG1 * C D\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 A\n  0.2 C\n  0.3 D\n";
    const auto read = readText(std::string(tree) + "MAXSIZE 1\nLIMIT 1E-12\n");
    auto const &model = requireModel(read);
    REQUIRE(read.diagnostics().empty());
    REQUIRE(projectAttribute(model, "ftp.maxsize") == std::vector<std::string>{"1"});
    // Nothing truncates: the order-2 cut set {C, D} is in the probability.
    REQUIRE(std::abs(probability(model, "TOP") - (1.0 - 0.9 * (1.0 - 0.06))) < 1e-15);

    SECTION("a later MAXSIZE with another value replaces it with a warning") {
        const auto twice = readText(std::string(tree) + "MAXSIZE 1\nmaxsize 0\n");
        auto const &last = requireModel(twice);
        REQUIRE(projectAttribute(last, "ftp.maxsize") == std::vector<std::string>{"0"});
        REQUIRE(count(twice.diagnostics(), diagnostic_severity::warning, "io.ftp.duplicate-maxsize", "MAXSIZE 0 replaces MAXSIZE 1") == 1);
    }
    SECTION("its value is a non-negative integer") {
        for (const auto *line : {"MAXSIZE\n", "MAXSIZE 1.5\n", "MAXSIZE -1\n", "MAXSIZE x\n", "MAXSIZE 99999999999\n"}) {
            INFO(line);
            const auto bad = readText(std::string(tree) + line);
            REQUIRE_FALSE(bad);
            REQUIRE(count(bad.diagnostics(), diagnostic_severity::error, "io.ftp.invalid-maxsize", "m.ftp:9") == 1);
        }
    }
    SECTION("inside a tree section MAXSIZE is a gate name") {
        const auto gate = readText("TOP + A MAXSIZE\nMAXSIZE 2 A C D\nENDTREE\nPROCESS TOP\nIMPORT\n  0.1 A\n"
                                   "  0.2 C\n  0.3 D\n");
        auto const &model = requireModel(gate);
        REQUIRE(gate.diagnostics().empty());
        REQUIRE(model.find_gate("MAXSIZE").valid());
        REQUIRE(projectAttribute(model, "ftp.maxsize").empty());
    }
}

TEST_CASE("FTP tokens after ENDTREE, IMPORT, LIMIT x and MAXSIZE n are ignored with a warning", "[io][ftp][grammar]") {
    const auto read = readText("TOP + A\nENDTREE now\nPROCESS TOP\nIMPORT records\n  0.1 A\nLIMIT 1E-5 junk\nMAXSIZE 2 more junk\n");
    auto const &model = requireModel(read);
    REQUIRE(projectAttribute(model, "ftp.limit") == std::vector<std::string>{"1E-5"});
    REQUIRE(read.diagnostics().size() == 4);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.ignored-tokens", "m.ftp:2: 'now' after ENDTREE is not read") == 1);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.ignored-tokens", "'records' after IMPORT") == 1);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.ignored-tokens", "'junk' after LIMIT") == 1);
    REQUIRE(count(read.diagnostics(), diagnostic_severity::warning, "io.ftp.ignored-tokens", "'more junk' after MAXSIZE") == 1);
}

TEST_CASE("FTP control bytes are blanks and a UTF-8 byte-order mark is skipped", "[io][ftp][grammar]") {
    const auto text = "\xEF\xBB\xBFTOP + A\x00 C\nENDTREE\f\nPROCESS TOP\nIMPORT\n\v 0.1 A\n  0.2 C\n\x1A"s;
    const auto read = readText(text);
    auto const &model = requireModel(read);
    REQUIRE(read.diagnostics().empty());
    REQUIRE(targetNames(model) == std::vector<std::string>{"TOP"});
    REQUIRE(model.operands(model.find_gate("TOP")).size() == 2);

    SECTION("a lone CR does not end a line") {
        const auto mac = readText("TOP + A\rENDTREE\rPROCESS TOP\rIMPORT\r  0.1 A\r");
        REQUIRE_FALSE(mac);
        REQUIRE(count(mac.diagnostics(), diagnostic_severity::error, "io.ftp.missing-endtree") == 1);
    }
}
