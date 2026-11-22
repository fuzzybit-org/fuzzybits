#pragma once

#include "api/diagnostics.h"
#include "api/options.h"
#include "io/ftp/descriptor.h"
#include "io/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pra/builder.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {

    // On-wire FTP tokens shared by the parser and writer (single source of truth).
    inline constexpr char kFtpOrGateToken = '+';
    inline constexpr char kFtpAndGateToken = '*';
    inline constexpr char kFtpComplementToken = '-';
    inline constexpr char kFtpHeaderPrefix = '*';
    inline constexpr char kFtpBasicEventNodePrefix = 'E';
    inline constexpr char kFtpCompositeNodePrefix = 'N';
    inline constexpr char kFtpGeneratedTargetPrefix = 'T';
    inline constexpr std::string_view kFtpInitiatorFlag = "I";
    inline constexpr std::string_view kFtpEncodingLine = "**CHAR32\n";
    inline constexpr std::string_view kFtpFtitleLine = "**FTITLE:\n";
    inline constexpr std::string_view kFtpXeqLine = "*XEQ\n";
    inline constexpr std::string_view kFtpLimitOutputPrefix = "LIMIT    ";
    inline constexpr std::string_view kFtpEndTreeToken = "ENDTREE";
    inline constexpr std::string_view kFtpLimitToken = "LIMIT";
    inline constexpr std::string_view kFtpProcessToken = "PROCESS";
    inline constexpr std::string_view kFtpImportToken = "IMPORT";
    inline constexpr std::string_view kFtpTargetLineJoin = " + ";
    inline constexpr std::string_view kFtpFormat = "ftp";

    // Attribute names in the `ir::pra` model.
    inline constexpr std::string_view kFtpHeaderAttribute = "ftp.header";
    inline constexpr std::string_view kFtpLimitAttribute = "ftp.limit";
    inline constexpr std::string_view kFtpImportFlagAttribute = "ftp.import_flag";
    // Source conventions, recorded for the compile driver; nothing in `ir::pra`
    // or the lowering reads them.
    inline constexpr std::string_view kFtpNegationAttribute = "ftp.negation";
    inline constexpr std::string_view kFtpNegationDeleteTerm = "delete_term";
    inline constexpr std::string_view kFtpMaxInitiatorsAttribute = "ftp.max_initiators_per_cutset";
    inline constexpr std::string_view kFtpMaxInitiatorsOne = "1";
    inline constexpr std::string_view kFtpMaxInitiatorsNone = "none";

    enum class DefinitionKind : std::uint8_t {
        and_gate,
        or_gate,
        at_least,
    };

    struct Location final {
        std::uint32_t file = 0;
        std::uint32_t line = 0;
    };

    struct Operand final {
        std::string_view name;
        bool complement = false;
    };

    struct Definition final {
        std::string_view name;
        DefinitionKind kind = DefinitionKind::or_gate;
        std::uint32_t threshold = 0;
        std::vector<Operand> operands;
        Location location;
    };

    struct ImportRecord final {
        std::string_view name;
        std::string_view valueText;
        double value = 0.0;
        bool initiator = false;
        std::vector<std::string_view> flags;
        Location location;
    };

    struct NamedLocation final {
        std::string_view name;
        Location location;
    };

    // A basic event: a non-gate name with at least one import record. `records`
    // indexes ParsedProject::imports, the record that supplies the value first.
    struct Event final {
        std::string_view name;
        double value = 0.0;
        bool initiator = false;
        bool frequency = false;
        std::vector<std::uint32_t> records;
    };

    struct Target final {
        std::string_view name;
        // Index into definitions when `gate`, otherwise into events.
        std::uint32_t index = 0;
        bool gate = true;
        Location location;
    };

    // Every string_view points into `texts`, which the project owns.
    struct ParsedProject final {
        std::vector<std::string> paths;
        std::vector<std::string> texts;
        std::vector<Definition> definitions;
        std::unordered_map<std::string_view, std::uint32_t> definitionByName;
        std::vector<ImportRecord> imports;
        std::vector<NamedLocation> process;
        std::vector<std::string_view> headers;
        std::optional<double> limit;
        std::string_view limitText;
        std::vector<Event> events;
        std::unordered_map<std::string_view, std::uint32_t> eventByName;
        std::vector<Target> targets;
        bool implicitTargets = false;
        api::diagnostic_list diagnostics;
    };

    struct PraOptions final {
        complement_events_policy complementEvents = complement_events_policy::atom;
        bool maxOneInitiator = true;
    };

    [[nodiscard]] auto parse_options(api::option_set const &options, api::diagnostic_list &diagnostics) -> duplicate_import_policy;
    [[nodiscard]] auto parse_pra_options(api::option_set const &options, api::diagnostic_list &diagnostics) -> PraOptions;
    [[nodiscard]] auto load_files(std::span<std::filesystem::path const> inputs) -> api::result<std::vector<source_text>>;
    // Parses every source in order and resolves names, imports and targets.
    [[nodiscard]] auto parse(std::vector<source_text> sources, duplicate_import_policy policy) -> ParsedProject;
    [[nodiscard]] auto location_text(ParsedProject const &project, Location location) -> std::string;

    [[nodiscard]] auto build_pra(ParsedProject const &project, PraOptions const &options) -> ir::pra::finalize_result;
    [[nodiscard]] auto write_pc(ir::pc::circuit const &model, std::ostream &output, io::write_request const &request)
        -> io::write_result;

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
