#pragma once

#include "api/diagnostics.h"
#include "api/result.h"
#include "compile/pra_to_pc/descriptor.h"
#include "io/ftp/descriptor.h"
#include "ir/pc/passes/passes.h"
#include "runtime/progress.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fuzzybits::compile::ftp {

// `element := TRUE | FALSE` from `--override NAME=0|1`.
struct override_spec final {
    std::string name;
    bool value = false;
};

// Settings of one compile. Unset values take, in this order, the `config`
// document, the model attribute the reader records
// (`ftp.max_initiators_per_cutset`) and the built-in default. Flags override
// the document: `targets` and `watch` replace its lists, `negation` and
// `literals` replace every per-target choice in it, `overrides` replace its
// entries of the same name.
struct options final {
    // Domain target (a PROCESS target), gate or basic event names. Empty:
    // the document's targets, else every PROCESS target.
    std::vector<std::string> targets;
    // Extra targets, with the watch role.
    std::vector<std::string> watch;
    std::vector<override_spec> overrides;
    // error | first | last (default last).
    std::optional<std::string> duplicate_import;
    // atom | negation (default atom).
    std::optional<std::string> complement_events;
    // boolean | hull | both: the query type of every target (default
    // boolean, the probability query). Never read from the model.
    std::optional<std::string> negation;
    // Outer unset: from the document or the model attribute. Inner unset: no
    // rule (`none`). Only with initiator_values = probability.
    std::optional<std::optional<std::uint32_t>> max_initiators_per_cutset;
    // frequency | probability (default frequency, #243): how `I` values are
    // quantified. frequency: one conditioned query per initiating event in a
    // target's cone, grouped as the target's frequency; probability: values up
    // to 1 used as probabilities, above 1 rejected.
    std::optional<std::string> initiator_values;
    // per-year | per-month | per-week | per-day | per-hour: the unit of every
    // `I` value (default per-year with initiator_values = frequency); rejected
    // with probability.
    std::optional<std::string> initiator_unit;
    // atom_split | omega_l | negation (default atom_split).
    std::optional<std::string> literals;
    // 0 or 1 (default 1).
    std::optional<int> level;
    // A quantification config document (JSON text); see docs/ftp-input.md.
    std::optional<std::string> config;

    // Not owned; they must outlive the compile.
    std::vector<runtime::progress::sink *> progress_sinks;
    double progress_max_rate_hz = 10.0;
};

struct target_summary final {
    std::string name;
    // domain_target | gate | basic_event
    std::string subject;
    bool watch = false;
    bool boolean = true;
    bool monotone_hull = false;
    std::string literals;
    std::optional<std::uint32_t> max_initiators_per_cutset;
    // A query conditioned on this initiating event (initiator_values =
    // frequency); empty otherwise. Its frequency is the query's weight.
    std::string initiator;
    std::optional<double> initiator_frequency;
    std::size_t initiating_events = 0;
    std::size_t literal_events = 0;
};

struct pass_summary final {
    std::string transform;
    std::string semantics;
    std::uint32_t nodes_before = 0;
    std::uint32_t edges_before = 0;
    std::uint32_t nodes_after = 0;
    std::uint32_t edges_after = 0;
    std::uint64_t rewrites = 0;
    std::uint64_t skipped = 0;
};

struct timings final {
    double read = 0.0;
    double lower = 0.0;
    double passes = 0.0;
    double write = 0.0;
};

struct compiled final {
    std::vector<std::byte> bytes;
    std::string model_name;
    // The sorted file set the digest covers.
    std::vector<std::string> files;
    std::string source_sha256;
    // The `ftp.limit` attribute, recorded for the metadata only.
    std::string limit;
    int level = 1;
    std::vector<target_summary> targets;
    backend::fuzzybits::pra_to_pc::transform_report lowering;
    std::vector<pass_summary> passes;
    std::uint32_t nodes_before = 0;
    std::uint32_t edges_before = 0;
    std::uint32_t nodes_after = 0;
    std::uint32_t edges_after = 0;
    // The effective settings, as written to `compile.options`.
    std::string options_json;
    timings seconds;
};

using compile_result = api::result<compiled>;

enum class input_kind : std::uint8_t {
    // Not FTP: MEF XML, nombrilo or unreadable (the caller's own reader reports it).
    other,
    ftp,
    // Some inputs are FTP and some are not.
    mixed,
};

// A file is FTP when it ends in `.ftp`, or is text (no NUL, not XML) with a
// line `ENDTREE`; a directory is FTP when it holds a `*.ftp` file.
[[nodiscard]] auto classify_inputs(std::span<std::filesystem::path const> inputs) -> input_kind;

// "NAME=0" or "NAME=1".
[[nodiscard]] auto parse_override(std::string_view text) -> api::result<override_spec>;

// Files and directories (all `*.ftp`) form one project, read in sorted path
// order. Reader, config, lowering, passes and `.pcfb` writer; every
// diagnostic of every stage is on the result, errors only on failure.
[[nodiscard]] auto compile_project(std::span<std::filesystem::path const> inputs, options const &settings = {})
    -> compile_result;
[[nodiscard]] auto compile_sources(std::span<io::backend::fuzzybits::ftp::source_text const> sources,
                                   options const &settings = {}) -> compile_result;

// Reader and lowering only, under the same settings; no passes, no writer.
[[nodiscard]] auto lower_sources(std::span<io::backend::fuzzybits::ftp::source_text const> sources,
                                 options const &settings = {})
    -> api::result<backend::fuzzybits::pra_to_pc::pra_pc_lowering>;

// JSON: effective settings, the lowering report, the pass reports, circuit
// sizes, timings and every diagnostic. Written for failures too (then only
// `status` and `diagnostics`).
[[nodiscard]] auto report_json(compile_result const &result, unsigned indent = 2) -> std::string;

} // namespace fuzzybits::compile::ftp
