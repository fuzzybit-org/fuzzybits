#include "io/ftp/detail/model.h"

#include "api/diagnostics.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {
    namespace {

        [[nodiscard]] constexpr auto isBlank(const char character) noexcept -> bool {
            return character == ' ' || character == '\t' || character == '\r' || character == '\f' || character == '\v';
        }

        [[nodiscard]] constexpr auto isDigit(const char character) noexcept -> bool {
            return character >= '0' && character <= '9';
        }

        [[nodiscard]] constexpr auto isNameCharacter(const char character) noexcept -> bool {
            return isDigit(character) || (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z') ||
                   character == '_' || character == '/' || character == '-';
        }

        [[nodiscard]] auto trim(std::string_view text) noexcept -> std::string_view {
            while (!text.empty() && isBlank(text.front())) {
                text.remove_prefix(1);
            }
            while (!text.empty() && isBlank(text.back())) {
                text.remove_suffix(1);
            }
            return text;
        }

        void tokenize(std::string_view line, std::vector<std::string_view> &tokens) {
            tokens.clear();
            std::size_t cursor = 0;
            while (cursor < line.size()) {
                while (cursor < line.size() && isBlank(line[cursor])) {
                    ++cursor;
                }
                const auto begin = cursor;
                while (cursor < line.size() && !isBlank(line[cursor])) {
                    ++cursor;
                }
                if (cursor > begin) {
                    tokens.push_back(line.substr(begin, cursor - begin));
                }
            }
        }

        // EVENT_ID of the grammar; a leading '-' is reserved for complemented references.
        [[nodiscard]] auto isName(const std::string_view token) noexcept -> bool {
            return !token.empty() && token.front() != kFtpComplementToken && std::ranges::all_of(token, isNameCharacter);
        }

        [[nodiscard]] auto isReference(std::string_view token) noexcept -> bool {
            if (!token.empty() && token.front() == kFtpComplementToken) {
                token.remove_prefix(1);
            }
            return isName(token);
        }

        [[nodiscard]] auto isDigits(const std::string_view token) noexcept -> bool {
            return !token.empty() && std::ranges::all_of(token, isDigit);
        }

        [[nodiscard]] auto isOperator(const std::string_view token) noexcept -> bool {
            return token == "+" || token == "*" || isDigits(token);
        }

        // NUMBER of the grammar: [+-]?[0-9]+('.'[0-9]+)?([Ee][+-]?[0-9]+)?
        [[nodiscard]] auto isNumber(std::string_view token) noexcept -> bool {
            const auto digits = [&token] {
                std::size_t count = 0;
                while (count < token.size() && isDigit(token[count])) {
                    ++count;
                }
                token.remove_prefix(count);
                return count;
            };
            if (!token.empty() && (token.front() == '+' || token.front() == '-')) {
                token.remove_prefix(1);
            }
            if (digits() == 0) {
                return false;
            }
            if (!token.empty() && token.front() == '.') {
                token.remove_prefix(1);
                if (digits() == 0) {
                    return false;
                }
            }
            if (!token.empty() && (token.front() == 'E' || token.front() == 'e')) {
                token.remove_prefix(1);
                if (!token.empty() && (token.front() == '+' || token.front() == '-')) {
                    token.remove_prefix(1);
                }
                if (digits() == 0) {
                    return false;
                }
            }
            return token.empty();
        }

        [[nodiscard]] auto parseNumber(std::string_view token) noexcept -> std::optional<double> {
            if (!isNumber(token)) {
                return std::nullopt;
            }
            if (token.front() == '+') {
                token.remove_prefix(1);
            }
            double value = 0.0;
            const auto end = token.data() + token.size();
            if (const auto [cursor, error] = std::from_chars(token.data(), end, value); error != std::errc{} || cursor != end) {
                return std::nullopt;
            }
            return value;
        }

        [[nodiscard]] auto parseThreshold(const std::string_view token) noexcept -> std::optional<std::uint32_t> {
            std::uint32_t value = 0;
            const auto end = token.data() + token.size();
            if (const auto [cursor, error] = std::from_chars(token.data(), end, value); error != std::errc{} || cursor != end) {
                return std::nullopt;
            }
            return value;
        }

        // `value name` outside an IMPORT section; the value cannot be a name.
        [[nodiscard]] auto isStrayRecord(std::vector<std::string_view> const &tokens) noexcept -> bool {
            return tokens.size() >= 2 && tokens[0].find('.') != std::string_view::npos && isNumber(tokens[0]) && isName(tokens[1]);
        }

        [[nodiscard]] auto quoted(const std::string_view name) -> std::string {
            std::string out;
            out.reserve(name.size() + 2);
            out += '\'';
            out += name;
            out += '\'';
            return out;
        }

        [[nodiscard]] auto formatValue(const double value) -> std::string {
            std::ostringstream out;
            out << value;
            return out.str();
        }

        [[nodiscard]] auto policyName(const duplicate_import_policy policy) -> std::string_view {
            switch (policy) {
            case duplicate_import_policy::last:
                return "last";
            case duplicate_import_policy::first:
                return "first";
            case duplicate_import_policy::error:
                return "error";
            }
            return "last";
        }

        class Parser final {
        public:
            Parser(ParsedProject &project, const duplicate_import_policy policy) : project_(project), policy_(policy) {}

            void parseFile(const std::uint32_t file) {
                const std::string_view text = project_.texts[file];
                section_ = Section::none;
                openGate_ = kNoGate;
                treeStart_.reset();
                afterTree_ = false;

                std::uint32_t lineNumber = 0;
                for (std::size_t cursor = 0; cursor < text.size();) {
                    auto end = text.find('\n', cursor);
                    if (end == std::string_view::npos) {
                        end = text.size();
                    }
                    ++lineNumber;
                    parseLine(text.substr(cursor, end - cursor), Location{.file = file, .line = lineNumber});
                    cursor = end + 1;
                }
                if (treeStart_) {
                    missingEndtree(*treeStart_, "the end of the file");
                }
            }

            void resolve() {
                collectEvents();
                resolveDuplicates();
                checkValues();
                checkGates();
                checkReferences();
                selectTargets();
                if (project_.limit) {
                    project_.metadata.truncation_limit = project_.limit;
                }
            }

        private:
            enum class Section : std::uint8_t {
                none,
                process,
                import,
            };

            static constexpr std::uint32_t kNoGate = std::numeric_limits<std::uint32_t>::max();
            static constexpr std::uint32_t kDiscardedGate = kNoGate - 1;

            [[nodiscard]] auto where(const Location location) const -> std::string {
                return location_text(project_, location);
            }

            void error(std::string code, std::string message) {
                project_.diagnostics.push(api::make_error(std::move(code), std::move(message)));
            }

            void warning(std::string code, std::string message) {
                project_.diagnostics.push(api::make_warning(std::move(code), std::move(message)));
            }

            void parseLine(const std::string_view line, const Location location) {
                tokenize(line, tokens_);
                if (tokens_.empty()) {
                    return;
                }
                const auto head = tokens_[0];
                if (head.front() == kFtpHeaderPrefix) {
                    project_.headers.push_back(trim(line));
                    section_ = Section::none;
                    return;
                }
                if (head == kFtpEndTreeToken || head == kFtpProcessToken || head == kFtpImportToken || head == kFtpLimitToken) {
                    if (treeStart_ && head != kFtpEndTreeToken) {
                        missingEndtree(*treeStart_, "the " + std::string(head) + " line at " + where(location));
                    }
                    treeStart_.reset();
                    openGate_ = kNoGate;
                    afterTree_ = true;
                    section_ = keyword(head, location);
                    return;
                }

                const auto indented = isBlank(line.front());
                if (section_ == Section::process) {
                    const auto names = std::ranges::all_of(tokens_, isName);
                    if (names && !(tokens_.size() > 1 && isOperator(tokens_[1]))) {
                        for (const auto token : tokens_) {
                            project_.process.push_back(NamedLocation{.name = token, .location = location});
                        }
                        return;
                    }
                    section_ = Section::none;
                }
                if (section_ == Section::import) {
                    if (importRecord(location)) {
                        return;
                    }
                    if (indented || tokens_.size() < 2 || !isOperator(tokens_[1])) {
                        error("io.ftp.invalid-line", where(location) + ": expected 'value name [flags]' in the IMPORT section");
                        return;
                    }
                    section_ = Section::none;
                }
                if (isStrayRecord(tokens_)) {
                    error("io.ftp.import-without-header", where(location) + ": import record for " + quoted(tokens_[1]) +
                                                               " outside an IMPORT section is not read");
                    return;
                }
                if (indented) {
                    if (openGate_ == kNoGate) {
                        error("io.ftp.invalid-line", where(location) + ": continuation line without an open gate");
                        return;
                    }
                    auto &gate = openGate_ == kDiscardedGate ? discarded_ : project_.definitions[openGate_];
                    appendOperands(gate, 0, location);
                    return;
                }
                gateLine(location);
            }

            [[nodiscard]] auto keyword(const std::string_view head, const Location location) -> Section {
                if (head == kFtpEndTreeToken || head == kFtpImportToken) {
                    if (tokens_.size() > 1) {
                        error("io.ftp.invalid-line", where(location) + ": unexpected tokens after " + std::string(head));
                    }
                    return head == kFtpImportToken ? Section::import : Section::none;
                }
                if (head == kFtpProcessToken) {
                    for (std::size_t index = 1; index < tokens_.size(); ++index) {
                        if (isName(tokens_[index])) {
                            project_.process.push_back(NamedLocation{.name = tokens_[index], .location = location});
                        } else {
                            error("io.ftp.invalid-line", where(location) + ": PROCESS target " + quoted(tokens_[index]) + " is not a name");
                        }
                    }
                    return Section::process;
                }
                const auto value = tokens_.size() == 2 ? parseNumber(tokens_[1]) : std::nullopt;
                if (!value) {
                    error("io.ftp.invalid-limit", where(location) + ": expected 'LIMIT value'");
                    return Section::none;
                }
                if (project_.limit && *project_.limit != *value) {
                    warning("io.ftp.duplicate-limit", where(location) + ": LIMIT " + std::string(tokens_[1]) + " replaces LIMIT " +
                                                          std::string(project_.limitText));
                }
                project_.limit = *value;
                project_.limitText = tokens_[1];
                return Section::none;
            }

            [[nodiscard]] auto importRecord(const Location location) -> bool {
                if (tokens_.size() < 2 || !isNumber(tokens_[0]) || !isName(tokens_[1])) {
                    return false;
                }
                ImportRecord record{.name = tokens_[1], .valueText = tokens_[0], .location = location};
                if (const auto value = parseNumber(tokens_[0])) {
                    record.value = *value;
                } else {
                    record.value = std::numeric_limits<double>::infinity();
                }
                for (std::size_t index = 2; index < tokens_.size(); ++index) {
                    if (tokens_[index] == kFtpInitiatorFlag) {
                        record.initiator = true;
                    } else {
                        record.flags.push_back(tokens_[index]);
                    }
                }
                project_.imports.push_back(std::move(record));
                return true;
            }

            void gateLine(const Location location) {
                if (tokens_.size() < 2 || !isName(tokens_[0]) || !isOperator(tokens_[1])) {
                    openGate_ = kNoGate;
                    error("io.ftp.invalid-line", where(location) + ": expected 'NAME (+|*|k) operands...'");
                    return;
                }
                Definition definition{.name = tokens_[0], .location = location};
                if (tokens_[1] == "+") {
                    definition.kind = DefinitionKind::or_gate;
                } else if (tokens_[1] == "*") {
                    definition.kind = DefinitionKind::and_gate;
                } else if (const auto threshold = parseThreshold(tokens_[1])) {
                    definition.kind = DefinitionKind::at_least;
                    definition.threshold = *threshold;
                } else {
                    openGate_ = kNoGate;
                    error("io.ftp.invalid-line", where(location) + ": threshold " + std::string(tokens_[1]) + " of gate " +
                                                     quoted(tokens_[0]) + " is out of range");
                    return;
                }
                definition.operands.reserve(tokens_.size() - 2);
                appendOperands(definition, 2, location);

                if (afterTree_) {
                    error("io.ftp.misplaced-gate", where(location) + ": gate " + quoted(definition.name) +
                                                       " after the tree section of its file is not read");
                    discarded_ = std::move(definition);
                    openGate_ = kDiscardedGate;
                    return;
                }
                if (!treeStart_) {
                    treeStart_ = location;
                }
                const auto index = static_cast<std::uint32_t>(project_.definitions.size());
                if (const auto [found, inserted] = project_.definitionByName.emplace(definition.name, index); !inserted) {
                    error("io.ftp.duplicate-gate", "gate " + quoted(definition.name) + " is defined twice (" +
                                                       where(project_.definitions[found->second].location) + ", " + where(location) +
                                                       "); the second definition is not read");
                    discarded_ = std::move(definition);
                    openGate_ = kDiscardedGate;
                    return;
                }
                project_.definitions.push_back(std::move(definition));
                openGate_ = index;
            }

            void appendOperands(Definition &gate, const std::size_t first, const Location location) {
                for (auto index = first; index < tokens_.size(); ++index) {
                    auto token = tokens_[index];
                    if (!isReference(token)) {
                        error("io.ftp.invalid-line",
                              where(location) + ": operand " + quoted(token) + " of gate " + quoted(gate.name) + " is not a name");
                        continue;
                    }
                    const auto complement = token.front() == kFtpComplementToken;
                    if (complement) {
                        token.remove_prefix(1);
                    }
                    gate.operands.push_back(Operand{.name = token, .complement = complement});
                }
            }

            void missingEndtree(const Location start, const std::string &before) {
                error("io.ftp.missing-endtree", "the tree section starting at " + where(start) + " has no ENDTREE before " + before);
            }

            void collectEvents() {
                std::unordered_set<std::string_view> reported;
                project_.events.reserve(project_.imports.size());
                project_.eventByName.reserve(project_.imports.size());
                for (std::uint32_t index = 0; index < project_.imports.size(); ++index) {
                    auto const &record = project_.imports[index];
                    if (const auto gate = project_.definitionByName.find(record.name); gate != project_.definitionByName.end()) {
                        if (reported.insert(record.name).second) {
                            error("io.ftp.gate-imported", quoted(record.name) + " is both a gate (" +
                                                              where(project_.definitions[gate->second].location) +
                                                              ") and an imported event (" + where(record.location) + ")");
                        }
                        continue;
                    }
                    const auto [found, inserted] =
                        project_.eventByName.emplace(record.name, static_cast<std::uint32_t>(project_.events.size()));
                    if (inserted) {
                        project_.events.push_back(Event{.name = record.name});
                    }
                    project_.events[found->second].records.push_back(index);
                }
            }

            void resolveDuplicates() {
                for (auto &event : project_.events) {
                    auto &records = event.records;
                    for (const auto record : records) {
                        event.initiator = event.initiator || project_.imports[record].initiator;
                    }
                    if (records.size() > 1) {
                        reportDuplicate(event);
                        if (policy_ == duplicate_import_policy::last) {
                            std::rotate(records.begin(), records.end() - 1, records.end());
                        }
                    }
                    event.value = project_.imports[records.front()].value;
                }
            }

            void reportDuplicate(Event const &event) {
                const auto selected = policy_ == duplicate_import_policy::last ? event.records.back() : event.records.front();
                bool differ = false;
                std::string records;
                for (const auto index : event.records) {
                    auto const &record = project_.imports[index];
                    differ = differ || record.value != project_.imports[event.records.front()].value;
                    records += records.empty() ? "" : ", ";
                    records += where(record.location) + " = " + std::string(record.valueText);
                    if (record.initiator) {
                        records += " I";
                    }
                }
                auto message = "basic event " + quoted(event.name) + " is imported " + std::to_string(event.records.size()) + " times" +
                               (differ ? " with different values" : " with the same value") + " (" + records + ")";
                if (policy_ == duplicate_import_policy::error) {
                    error("io.ftp.duplicate-import", std::move(message) + "; policy error");
                    return;
                }
                warning("io.ftp.duplicate-import", std::move(message) + "; policy " + std::string(policyName(policy_)) + " uses " +
                                                       where(project_.imports[selected].location) +
                                                       (event.initiator ? ", initiator tag kept" : ""));
            }

            void checkValues() {
                for (auto &event : project_.events) {
                    for (const auto index : event.records) {
                        auto const &record = project_.imports[index];
                        if (!std::isfinite(record.value) || record.value < 0.0 || (record.value > 1.0 && !event.initiator)) {
                            error("io.ftp.invalid-probability", where(record.location) + ": value " + std::string(record.valueText) +
                                                                     " of " + quoted(record.name) + " is outside [0, 1]");
                        }
                        for (const auto flag : record.flags) {
                            warning("io.ftp.unknown-import-flag", where(record.location) + ": unknown flag " + quoted(flag) +
                                                                       " on the import of " + quoted(record.name) +
                                                                       " is kept as an attribute");
                        }
                    }
                    if (event.initiator && std::isfinite(event.value) && event.value > 1.0) {
                        event.frequency = true;
                        warning("io.ftp.initiator-frequency", "initiating event " + quoted(event.name) + " at " +
                                                                   where(project_.imports[event.records.front()].location) +
                                                                   " has value " + formatValue(event.value) +
                                                                   " > 1; it is kept as a frequency, not a probability");
                    }
                }
            }

            void checkGates() {
                for (auto const &gate : project_.definitions) {
                    const auto count = gate.operands.size();
                    if (count == 0) {
                        error("io.ftp.invalid-gate", "gate " + quoted(gate.name) + " at " + where(gate.location) + " has no operands");
                    } else if (gate.kind == DefinitionKind::at_least && (gate.threshold == 0 || gate.threshold > count)) {
                        error("io.ftp.invalid-gate", "at-least-" + std::to_string(gate.threshold) + " gate " + quoted(gate.name) + " at " +
                                                         where(gate.location) + " has " + std::to_string(count) +
                                                         " operands; k must be in [1, n]");
                    }
                }
            }

            void checkReferences() {
                std::unordered_set<std::string_view> reported;
                for (auto const &gate : project_.definitions) {
                    for (auto const &operand : gate.operands) {
                        if (project_.definitionByName.contains(operand.name) || project_.eventByName.contains(operand.name) ||
                            !reported.insert(operand.name).second) {
                            continue;
                        }
                        error("io.ftp.undefined-reference", "gate " + quoted(gate.name) + " at " + where(gate.location) +
                                                                " references " + quoted(operand.name) +
                                                                ", which is neither a gate nor an imported event");
                    }
                }
            }

            void selectTargets() {
                if (!project_.process.empty()) {
                    std::unordered_set<std::string_view> seen;
                    for (auto const &[name, location] : project_.process) {
                        if (!seen.insert(name).second) {
                            continue;
                        }
                        if (const auto gate = project_.definitionByName.find(name); gate != project_.definitionByName.end()) {
                            project_.targets.push_back(Target{.name = name, .index = gate->second, .gate = true, .location = location});
                        } else if (const auto event = project_.eventByName.find(name); event != project_.eventByName.end()) {
                            project_.targets.push_back(Target{.name = name, .index = event->second, .gate = false, .location = location});
                        } else {
                            error("io.ftp.undefined-process-target", where(location) + ": PROCESS target " + quoted(name) +
                                                                         " is neither a gate nor an imported event");
                        }
                    }
                    return;
                }

                project_.implicitTargets = true;
                std::unordered_set<std::string_view> referenced;
                for (auto const &gate : project_.definitions) {
                    for (auto const &operand : gate.operands) {
                        referenced.insert(operand.name);
                    }
                }
                for (std::uint32_t index = 0; index < project_.definitions.size(); ++index) {
                    auto const &gate = project_.definitions[index];
                    if (!referenced.contains(gate.name)) {
                        project_.targets.push_back(Target{.name = gate.name, .index = index, .gate = true, .location = gate.location});
                    }
                }
                warning("io.ftp.missing-process",
                        "no PROCESS targets; the targets are the " + std::to_string(project_.targets.size()) + " root gates");
            }

            ParsedProject &project_;
            duplicate_import_policy policy_;
            std::vector<std::string_view> tokens_;
            Section section_ = Section::none;
            std::uint32_t openGate_ = kNoGate;
            Definition discarded_;
            std::optional<Location> treeStart_;
            bool afterTree_ = false;
        };

    } // namespace

    auto location_text(ParsedProject const &project, const Location location) -> std::string {
        return project.paths[location.file] + ':' + std::to_string(location.line);
    }

    auto parse(std::vector<source_text> sources, const duplicate_import_policy policy) -> ParsedProject {
        ParsedProject project;
        project.paths.reserve(sources.size());
        project.texts.reserve(sources.size());
        for (auto &source : sources) {
            project.paths.push_back(std::move(source.path));
            project.texts.push_back(std::move(source.text));
        }

        Parser parser(project, policy);
        for (std::uint32_t file = 0; file < project.texts.size(); ++file) {
            parser.parseFile(file);
        }
        parser.resolve();
        return project;
    }

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
