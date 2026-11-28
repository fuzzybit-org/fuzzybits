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

// Settings of one compile. Unset values take the `config` document, else the
// built-in default; nothing is read from the model. Flags override the
// document: `targets` and `watch` replace its lists, `kinds`,
// `hull_literals` and `hull_max_initiators` replace every per-target choice
// in it, `overrides` replace its entries of the same name.
struct options final {
    // Domain target (a PROCESS target), gate or basic event names. Empty:
    // the document's targets, else every PROCESS target.
    std::vector<std::string> targets;
    // Extra targets, with the watch role.
    std::vector<std::string> watch;
    std::vector<override_spec> overrides;
    // error | first | last (default last).
    std::optional<std::string> duplicate_import;
    // The queries of every target: probability | hull, each entry one kind
    // or a comma-separated list. Empty: the document, else probability.
    // Never read from the model.
    std::vector<std::string> kinds;
    // split | fixed | plain: how hull queries read marked literals (-EVENT);
    // default split. Only when some target asks for the hull.
    std::optional<std::string> hull_literals;
    // The initiator rule of hull queries: every hull query is ANDed with
    // not atleast(N + 1; initiating events in its cone); probability queries
    // never are. Outer unset: the document, else no rule. Inner unset: no
    // rule (`none`). A count needs a hull query and initiator_values =
    // probability.
    std::optional<std::optional<std::uint32_t>> hull_max_initiators;
    // What `I` values are (#243): a frequency unit, per-year (default) |
    // per-month | per-week | per-day | per-hour, giving one conditioned query
    // per initiating event in a target's cone, grouped as the target's
    // frequency in that unit; or `probability`, values up to 1 used as
    // probabilities, above 1 rejected.
    std::optional<std::string> initiator_values;
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
    // probability | hull: one summary per query.
    std::string kind;
    // A hull query's reading of marked literals (split | fixed | plain); empty for a probability query.
    std::string hull_literals;
    // The initiator rule lowered into this query's target; unset when it holds no delete term.
    std::optional<std::uint32_t> hull_max_initiators;
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
[[nodiscard]] auto compile_project(std::span<std::filesystem::path const> inputs, options const &settings = {}) -> compile_result;
[[nodiscard]] auto compile_sources(std::span<io::backend::fuzzybits::ftp::source_text const> sources, options const &settings = {}) -> compile_result;

// Reader and lowering only, under the same settings; no passes, no writer.
[[nodiscard]] auto lower_sources(std::span<io::backend::fuzzybits::ftp::source_text const> sources, options const &settings = {})
    -> api::result<backend::fuzzybits::pra_to_pc::pra_pc_lowering>;

// JSON: effective settings, the lowering report, the pass reports, circuit
// sizes, timings and every diagnostic. Written for failures too (then only
// `status` and `diagnostics`).
[[nodiscard]] auto report_json(compile_result const &result, unsigned indent = 2) -> std::string;

} // namespace fuzzybits::compile::ftp
