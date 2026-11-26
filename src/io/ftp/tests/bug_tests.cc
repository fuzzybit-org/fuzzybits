// Known bugs of the FTP project reader. Each case states the docs/ftp-input.md rule it checks and fails on
// develop until its issue is fixed. This executable is labelled `bug` (ctest -L bug).

#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "io/ftp/descriptor.h"
#include "ir/pra/model.h"

#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ftp = fuzzybits::io::backend::fuzzybits::ftp;
namespace pra = fuzzybits::ir::pra;

[[nodiscard]] auto diagnosticsText(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
    std::ostringstream output;
    for (auto const &diagnostic : diagnostics.items()) {
        output << diagnostic.code << ": " << diagnostic.message << '\n';
    }
    return output.str();
}

[[nodiscard]] auto readText(std::string_view text) -> fuzzybits::io::read_result {
    const std::vector<ftp::source_text> sources{ftp::source_text{.path = "m.ftp", .text = std::string(text)}};
    return ftp::read_project(sources, {});
}

} // namespace

// docs/ftp-input.md (Grammar, Keywords): "MAXSIZE is a keyword only outside a tree section, so a gate may
// be called MAXSIZE." The tree section starts at its first gate line, so a gate named MAXSIZE on that
// line is read as the keyword: `MAXSIZE + A B` is io.ftp.invalid-maxsize, and `MAXSIZE 1 A B` silently
// sets the cut-set limit to 1 and drops the gate (io.ftp.ignored-tokens, a warning). A gate named
// MAXSIZE later in the tree is read as a gate.
TEST_CASE("a gate named MAXSIZE on the first line of the tree section is a gate", "[bug][ftp][grammar][issue-293]") {
    SECTION("OR gate") {
        const auto read = readText("MAXSIZE + A B\nENDTREE\nPROCESS MAXSIZE\nIMPORT\n0.1 A\n0.2 B\n");
        INFO(diagnosticsText(read.diagnostics()));
        REQUIRE(read);
        auto const *model = pra::model_from(read.value());
        REQUIRE(model != nullptr);
        CHECK(model->find_gate("MAXSIZE").valid());
    }
    SECTION("at-least gate") {
        const auto read = readText("MAXSIZE 1 A B\nENDTREE\nPROCESS MAXSIZE\nIMPORT\n0.1 A\n0.2 B\n");
        INFO(diagnosticsText(read.diagnostics()));
        REQUIRE(read);
        auto const *model = pra::model_from(read.value());
        REQUIRE(model != nullptr);
        CHECK(model->find_gate("MAXSIZE").valid());
        for (auto const &attribute : model->project_attributes()) {
            CHECK(model->text(attribute.name) != "ftp.maxsize");
        }
    }
}
