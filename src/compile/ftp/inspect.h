#pragma once

#include "api/diagnostics.h"
#include "compile/ftp/json.h"
#include "io/ftp/descriptor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fuzzybits::ir::pra {
    class model;
}

namespace fuzzybits::compile::ftp {

    // Reader policies only; inspecting never lowers or writes.
    struct inspect_options final {
        // error | first | last
        std::string duplicate_import = "last";
        // atom | negation
        std::string complement_events = "atom";
    };

    // Empty when the options are valid, else the message of the first problem.
    [[nodiscard]] auto check_options(inspect_options const &options) -> std::string;

    // `file` is empty and `line` 0 when the message names no project location.
    struct inspected_diagnostic final {
        std::string id;
        api::diagnostic_severity severity = api::diagnostic_severity::note;
        std::string file;
        std::uint32_t line = 0;
        std::string message;
    };

    struct import_occurrence final {
        std::string file;
        std::uint32_t line = 0;
        // The record whose value (and position in the model) the policy kept.
        bool chosen = false;
    };

    struct duplicate_import final {
        std::string name;
        double value = 0.0;
        // Project order.
        std::vector<import_occurrence> occurrences;
    };

    struct structure_summary final {
        std::size_t gates = 0;
        std::size_t or_gates = 0;
        std::size_t and_gates = 0;
        std::size_t atleast_gates = 0;
        // "k/n" -> number of at-least gates.
        std::map<std::string, std::size_t> atleast_k_of_n;
        std::size_t edges = 0;
        std::size_t single_input_gates = 0;
        std::size_t max_fan_in = 0;
        std::uint32_t depth = 0;
        std::size_t basic_events = 0;
        std::size_t house_events = 0;
        std::size_t import_lines = 0;
        std::size_t initiating_events = 0;
        std::size_t initiating_events_p_1 = 0;
        std::size_t events_p_1 = 0;
        std::size_t events_p_0 = 0;
        // Basic events with 0 < p < 2^-32.
        std::size_t events_p_tiny = 0;
        std::size_t complemented_edges = 0;
        std::size_t complemented_edges_to_basic_events = 0;
        std::size_t complemented_edges_to_gates = 0;
        // Complemented basic-event operands marked as atomic literals.
        std::size_t literal_operands = 0;
        std::vector<std::string> roots;
        // The model's targets: the PROCESS names, or the roots when there are none.
        std::vector<std::string> process_targets;
        std::optional<double> limit;
        std::size_t names_not_ncname = 0;
        std::vector<duplicate_import> duplicate_imports;
    };

    struct conventions_summary final {
        // The text of the `LIMIT` line as read.
        std::optional<std::string> limit;
        std::optional<std::string> negation;
        std::optional<std::string> max_initiators_per_cutset;
        std::vector<std::string> header_lines;
    };

    struct inspection final {
        // Project order; the directory arguments expanded.
        std::vector<std::string> files;
        std::vector<inspected_diagnostic> diagnostics;
        // Present when the project was read without errors.
        std::optional<structure_summary> structure;
        conventions_summary conventions;

        [[nodiscard]] auto count(api::diagnostic_severity severity) const -> std::size_t;
        [[nodiscard]] auto errors() const -> std::size_t {
            return count(api::diagnostic_severity::error);
        }
        [[nodiscard]] auto warnings() const -> std::size_t {
            return count(api::diagnostic_severity::warning);
        }
        [[nodiscard]] auto ok() const -> bool {
            return errors() == 0;
        }
    };

    // Counts of a model read from FTP; the record the reader kept for a repeated import is the
    // model's first source of the event.
    [[nodiscard]] auto structure_of(ir::pra::model const &model) -> structure_summary;

    // Files and directories (all `*.ftp`) form one project, as for `compile_project`.
    // Invalid options are not checked here; see `check_options`.
    [[nodiscard]] auto inspect_project(std::span<std::filesystem::path const> inputs, inspect_options const &options = {}) -> inspection;
    [[nodiscard]] auto inspect_sources(std::span<io::backend::fuzzybits::ftp::source_text const> sources,
                                       inspect_options const &options = {}) -> inspection;

    // With `counts_only` no element name appears: diagnostics keep id, severity and
    // position, and the roots, targets and duplicate imports are given as counts.
    [[nodiscard]] auto inspection_json(inspection const &result, bool counts_only = false) -> json::value;
    [[nodiscard]] auto inspection_text(inspection const &result, bool counts_only = false) -> std::string;

    // 0 without errors, 1 with errors, and 1 with warnings when `strict`.
    [[nodiscard]] auto inspect_exit_status(inspection const &result, bool strict) -> int;

} // namespace fuzzybits::compile::ftp
