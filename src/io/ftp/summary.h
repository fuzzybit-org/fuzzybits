#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fuzzybits::ir::pra {
class model;
}

namespace fuzzybits::io::backend::fuzzybits::ftp {

struct import_occurrence final {
    std::string file;
    std::uint32_t line = 0;
    // The record whose value (and position in the model) the policy kept.
    bool chosen = false;
};

struct duplicate_import final {
    std::string name;
    // The value of the chosen record.
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
    // The names of the PROCESS lines; empty when the project declares none.
    std::vector<std::string> process_targets;
    // The model's targets: `process_targets`, or the root gates when none are declared.
    std::vector<std::string> default_targets;
    std::optional<double> limit;
    std::size_t names_not_ncname = 0;
    std::vector<duplicate_import> duplicate_imports;
};

// Counts of a model read from FTP. The reader keeps the record a repeated import resolved to
// as the event's first source. A target is implicit when it sits where its subject is defined.
[[nodiscard]] auto structure_of(ir::pra::model const &model) -> structure_summary;

} // namespace fuzzybits::io::backend::fuzzybits::ftp
