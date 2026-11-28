#pragma once

#include "api/diagnostics.h"
#include "api/options.h"
#include "io/descriptor.h"
#include "io/ftp/descriptor.h"
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
#include <unordered_set>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {

// On-wire FTP tokens shared by the parser and writer (single source of truth).
inline constexpr char kFtpOrGateToken = '+';
inline constexpr char kFtpAndGateToken = '*';
inline constexpr char kFtpComplementToken = '-';
inline constexpr char kFtpHeaderPrefix = '*';
inline constexpr char kFtpCommentToken = ';';
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
inline constexpr std::string_view kFtpMaxSizeToken = "MAXSIZE";
inline constexpr std::string_view kFtpProcessToken = "PROCESS";
inline constexpr std::string_view kFtpImportToken = "IMPORT";
// A record value of exactly -1 stands for probability 1.
inline constexpr double kFtpMinusOneValue = -1.0;
inline constexpr std::string_view kFtpTargetLineJoin = " + ";
inline constexpr std::string_view kFtpFormat = "ftp";

// Attribute names in the `ir::pra` model.
inline constexpr std::string_view kFtpHeaderAttribute = "ftp.header";
inline constexpr std::string_view kFtpLimitAttribute = "ftp.limit";
inline constexpr std::string_view kFtpMaxSizeAttribute = "ftp.maxsize";
inline constexpr std::string_view kFtpImportFlagAttribute = "ftp.import_flag";
// The reader records what the files say. Quantification choices (the query kind, how hull queries read
// complements, the initiator rule) are never recorded: the compile request makes them.

// FTP names, keywords and import flags are compared without regard to ASCII
// case: `g1` and `G1` name one element. Other bytes compare exactly.
[[nodiscard]] constexpr auto ascii_upper(const char character) noexcept -> char {
    return character >= 'a' && character <= 'z' ? static_cast<char>(character - 'a' + 'A') : character;
}

[[nodiscard]] constexpr auto equal_ignoring_case(const std::string_view left, const std::string_view right) noexcept -> bool {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (ascii_upper(left[index]) != ascii_upper(right[index])) {
            return false;
        }
    }
    return true;
}

struct NameHash final {
    using is_transparent = void;
    [[nodiscard]] auto operator()(const std::string_view name) const noexcept -> std::size_t {
        std::uint64_t hash = 14695981039346656037ULL; // FNV-1a
        for (const auto character : name) {
            hash = (hash ^ static_cast<unsigned char>(ascii_upper(character))) * 1099511628211ULL;
        }
        return static_cast<std::size_t>(hash);
    }
};

struct NameEqual final {
    using is_transparent = void;
    [[nodiscard]] auto operator()(const std::string_view left, const std::string_view right) const noexcept -> bool { return equal_ignoring_case(left, right); }
};

using NameIndex = std::unordered_map<std::string_view, std::uint32_t, NameHash, NameEqual>;
using NameSet = std::unordered_set<std::string_view, NameHash, NameEqual>;

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
    // The value was written as -1 and is read as probability 1.
    bool minusOne = false;
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

// Every string_view points into `texts`, which the project owns. A gate is
// named as its definition spells it, an event as its first import record does.
struct ParsedProject final {
    std::vector<std::string> paths;
    std::vector<std::string> texts;
    std::vector<Definition> definitions;
    NameIndex definitionByName;
    std::vector<ImportRecord> imports;
    std::vector<NamedLocation> process;
    std::vector<std::string_view> headers;
    std::optional<double> limit;
    std::string_view limitText;
    std::optional<std::uint32_t> maxSize;
    std::string_view maxSizeText;
    std::vector<Event> events;
    NameIndex eventByName;
    std::vector<Target> targets;
    bool implicitTargets = false;
    api::diagnostic_list diagnostics;
};

[[nodiscard]] auto parse_options(api::option_set const &options, api::diagnostic_list &diagnostics) -> duplicate_import_policy;
[[nodiscard]] auto load_files(std::span<std::filesystem::path const> inputs) -> api::result<std::vector<source_text>>;
// Parses every source in order and resolves names, imports and targets.
[[nodiscard]] auto parse(std::vector<source_text> sources, duplicate_import_policy policy) -> ParsedProject;
[[nodiscard]] auto location_text(ParsedProject const &project, Location location) -> std::string;

// Every complemented basic-event operand (`-EVENT`) is marked `literal`; a
// complemented gate (`-GATE`) never is.
[[nodiscard]] auto build_pra(ParsedProject const &project) -> ir::pra::finalize_result;
[[nodiscard]] auto write_pc(ir::pc::circuit const &model, std::ostream &output, io::write_request const &request) -> io::write_result;

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
