#pragma once

#include "api/diagnostics.h"
#include "compile/ftp/json.h"
#include "io/ftp/descriptor.h"
#include "io/ftp/summary.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fuzzybits::compile::ftp {

// Reader policies only; inspecting never lowers or writes.
struct inspect_options final {
    // error | first | last
    std::string duplicate_import = "last";
};

// Empty when the options are valid, else the message of the first problem.
[[nodiscard]] auto check_options(inspect_options const &options) -> std::string;

// `file` is empty and `line` 0 when the diagnostic has no source position.
struct inspected_diagnostic final {
    std::string id;
    api::diagnostic_severity severity = api::diagnostic_severity::note;
    std::string file;
    std::uint32_t line = 0;
    std::string message;
};

struct conventions_summary final {
    // The text of the `LIMIT` line as read.
    std::optional<std::string> limit;
    // The value of the `MAXSIZE` line as read.
    std::optional<std::string> maxsize;
    std::vector<std::string> header_lines;
};

struct inspection final {
    // Project order; the directory arguments expanded.
    std::vector<std::string> files;
    std::vector<inspected_diagnostic> diagnostics;
    // Present when the project was read without errors.
    std::optional<io::backend::fuzzybits::ftp::structure_summary> structure;
    conventions_summary conventions;

    [[nodiscard]] auto count(api::diagnostic_severity severity) const -> std::size_t;
    [[nodiscard]] auto errors() const -> std::size_t { return count(api::diagnostic_severity::error); }
    [[nodiscard]] auto warnings() const -> std::size_t { return count(api::diagnostic_severity::warning); }
    [[nodiscard]] auto ok() const -> bool { return errors() == 0; }
};

// Files and directories (all `*.ftp`) form one project, as for `compile_project`.
// Invalid options are not checked here; see `check_options`.
[[nodiscard]] auto inspect_project(std::span<std::filesystem::path const> inputs, inspect_options const &options = {}) -> inspection;
[[nodiscard]] auto inspect_sources(std::span<io::backend::fuzzybits::ftp::source_text const> sources, inspect_options const &options = {}) -> inspection;

// With `counts_only` no element name appears: diagnostics keep id, severity and
// position, and the roots, targets and duplicate imports are given as counts.
[[nodiscard]] auto inspection_json(inspection const &result, bool counts_only = false) -> json::value;
[[nodiscard]] auto inspection_text(inspection const &result, bool counts_only = false) -> std::string;

// 0 without errors, 1 with errors, and 1 with warnings when `strict`.
[[nodiscard]] auto inspect_exit_status(inspection const &result, bool strict) -> int;

} // namespace fuzzybits::compile::ftp
