#include "compile/ftp/inspect.h"

#include "ir/pra/model.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

namespace fuzzybits::compile::ftp {

    namespace {

        namespace pra = ir::pra;
        namespace reader = io::backend::fuzzybits::ftp;

        constexpr double tiny_probability = 1.0 / 4294967296.0;

        [[nodiscard]] auto is_ncname(const std::string_view name) -> bool {
            const auto start = [](const char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; };
            const auto rest = [&start](const char c) { return start(c) || (c >= '0' && c <= '9') || c == '.' || c == '-'; };
            return !name.empty() && start(name.front()) && std::ranges::all_of(name.substr(1), rest);
        }

        [[nodiscard]] auto literal_value(pra::model const &model, auto const &event) -> double {
            return model.expressions()[event.expression.raw()].value;
        }

        [[nodiscard]] auto attribute_text(pra::model const &model, const std::string_view name) -> std::optional<std::string> {
            for (auto const &attribute : model.project_attributes()) {
                if (model.text(attribute.name) == name) {
                    return std::string(model.text(attribute.value));
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] auto parse_number(const std::string_view text) -> std::optional<double> {
            double value = 0.0;
            const auto [end, status] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (status != std::errc{} || end != text.data() + text.size()) {
                return std::nullopt;
            }
            return value;
        }

        [[nodiscard]] auto severity_name(const api::diagnostic_severity severity) -> std::string {
            switch (severity) {
            case api::diagnostic_severity::error:
                return "error";
            case api::diagnostic_severity::warning:
                return "warning";
            case api::diagnostic_severity::note:
                break;
            }
            return "note";
        }

        [[nodiscard]] auto format_number(const double value) -> std::string {
            std::array<char, 40> buffer{};
            const auto [end, status] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
            return status == std::errc{} ? std::string(buffer.data(), end) : std::string("nan");
        }

        // The earliest `<file>:<line>` in a message, for the files of the project; the
        // reader reports positions only inside the message text.
        void locate(std::string_view message, std::vector<std::string> const &files, std::string &file, std::uint32_t &line) {
            std::size_t best = std::string_view::npos;
            for (auto const &candidate : files) {
                if (candidate.empty()) {
                    continue;
                }
                for (auto at = message.find(candidate); at != std::string_view::npos; at = message.find(candidate, at + 1)) {
                    const auto digits = at + candidate.size() + 1;
                    const bool boundary = at == 0 || message[at - 1] == ' ' || message[at - 1] == '(' || message[at - 1] == '\'';
                    if (!boundary || digits >= message.size() || message[digits - 1] != ':' || message[digits] < '0' || message[digits] > '9') {
                        continue;
                    }
                    if (at < best || (at == best && candidate.size() > file.size())) {
                        std::uint32_t value = 0;
                        std::from_chars(message.data() + digits, message.data() + message.size(), value);
                        best = at;
                        file = candidate;
                        line = value;
                    }
                    break;
                }
            }
            if (best == std::string_view::npos) {
                file.clear();
                line = 0;
            }
        }

        [[nodiscard]] auto source_path(pra::model const &model, const pra::source_location location) -> std::string {
            return std::string(model.text(model.source_files()[location.file.raw()].path));
        }

        [[nodiscard]] auto read_options(inspect_options const &options) -> api::option_set {
            api::option_set out;
            out.set(reader::duplicate_import_option(), options.duplicate_import);
            out.set(reader::complement_events_option(), options.complement_events);
            return out;
        }

        void collect(inspection &out, io::read_result const &read) {
            for (auto const &item : read.diagnostics().items()) {
                inspected_diagnostic row{.id = item.code, .severity = item.severity, .message = item.message};
                locate(item.message, out.files, row.file, row.line);
                out.diagnostics.push_back(std::move(row));
            }
            if (!read || read.diagnostics().has_errors()) {
                return;
            }
            auto const *model = pra::model_from(read.value());
            if (model == nullptr) {
                return;
            }
            out.structure = structure_of(*model);
            auto &conventions = out.conventions;
            conventions.limit = attribute_text(*model, "ftp.limit");
            conventions.negation = attribute_text(*model, "ftp.negation");
            conventions.max_initiators_per_cutset = attribute_text(*model, "ftp.max_initiators_per_cutset");
            for (auto const &attribute : model->project_attributes()) {
                if (model->text(attribute.name) == "ftp.header") {
                    conventions.header_lines.emplace_back(model->text(attribute.value));
                }
            }
        }

        [[nodiscard]] auto count_value(const std::size_t value) -> json::value {
            return json::value::make_int(static_cast<std::int64_t>(value));
        }

        [[nodiscard]] auto name_list(std::vector<std::string> const &names) -> json::value {
            auto out = json::value::make_array();
            for (auto const &name : names) {
                out.push(json::value::make_string(name));
            }
            return out;
        }

        [[nodiscard]] auto optional_string(std::optional<std::string> const &text) -> json::value {
            return text ? json::value::make_string(*text) : json::value::make_null();
        }

        void position_json(std::string const &file, const std::uint32_t line, json::value &into) {
            into.set("file", file.empty() ? json::value::make_null() : json::value::make_string(file));
            into.set("line", line == 0 ? json::value::make_null() : json::value::make_int(line));
        }

        [[nodiscard]] auto structure_json(structure_summary const &s, const bool counts_only) -> json::value {
            auto out = json::value::make_object();
            out.set("gates", count_value(s.gates));
            auto kinds = json::value::make_object();
            kinds.set("or", count_value(s.or_gates));
            kinds.set("and", count_value(s.and_gates));
            kinds.set("atleast", count_value(s.atleast_gates));
            out.set("gates_by_kind", std::move(kinds));
            auto bounds = json::value::make_object();
            for (auto const &[key, count] : s.atleast_k_of_n) {
                bounds.set(key, count_value(count));
            }
            out.set("atleast_k_of_n", std::move(bounds));
            out.set("edges", count_value(s.edges));
            out.set("single_input_gates", count_value(s.single_input_gates));
            out.set("max_fan_in", count_value(s.max_fan_in));
            out.set("depth", count_value(s.depth));
            out.set("basic_events", count_value(s.basic_events));
            out.set("house_events", count_value(s.house_events));
            out.set("import_lines", count_value(s.import_lines));
            out.set("complemented_edges", count_value(s.complemented_edges));
            out.set("complemented_edges_to_basic_events", count_value(s.complemented_edges_to_basic_events));
            out.set("complemented_edges_to_gates", count_value(s.complemented_edges_to_gates));
            out.set("literal_operands", count_value(s.literal_operands));
            out.set("roots", counts_only ? count_value(s.roots.size()) : name_list(s.roots));
            out.set("process_targets", counts_only ? count_value(s.process_targets.size()) : name_list(s.process_targets));
            out.set("limit", s.limit ? json::value::make_number(*s.limit) : json::value::make_null());
            out.set("initiator_tagged_events", count_value(s.initiating_events));
            out.set("initiator_tagged_events_with_p_1", count_value(s.initiating_events_p_1));
            out.set("events_p_eq_1", count_value(s.events_p_1));
            out.set("events_p_eq_0", count_value(s.events_p_0));
            out.set("events_0_lt_p_lt_2^-32", count_value(s.events_p_tiny));
            out.set("names_not_valid_xml_ncname", count_value(s.names_not_ncname));
            if (counts_only) {
                out.set("duplicate_imports", count_value(s.duplicate_imports.size()));
                return out;
            }
            auto duplicates = json::value::make_object();
            for (auto const &duplicate : s.duplicate_imports) {
                auto records = json::value::make_array();
                for (auto const &occurrence : duplicate.occurrences) {
                    auto record = json::value::make_object();
                    position_json(occurrence.file, occurrence.line, record);
                    record.set("chosen", json::value::make_bool(occurrence.chosen));
                    if (occurrence.chosen) {
                        record.set("value", json::value::make_number(duplicate.value));
                    }
                    records.push(std::move(record));
                }
                duplicates.set(duplicate.name, std::move(records));
            }
            out.set("duplicate_imports", std::move(duplicates));
            return out;
        }

        void put_line(std::string &out, const std::string_view text) {
            out += text;
            out += '\n';
        }

    } // namespace

    auto check_options(inspect_options const &options) -> std::string {
        if (options.duplicate_import != "error" && options.duplicate_import != "first" && options.duplicate_import != "last") {
            return "--duplicate-import must be error, first or last, not '" + options.duplicate_import + "'";
        }
        if (options.complement_events != "atom" && options.complement_events != "negation") {
            return "--complement-events must be atom or negation, not '" + options.complement_events + "'";
        }
        return {};
    }

    auto inspection::count(const api::diagnostic_severity severity) const -> std::size_t {
        return static_cast<std::size_t>(std::ranges::count_if(diagnostics, [severity](auto const &item) { return item.severity == severity; }));
    }

    auto structure_of(pra::model const &model) -> structure_summary {
        structure_summary out;
        out.gates = model.gates().size();
        std::vector<std::uint8_t> referenced(model.gates().size(), 0);
        for (std::uint32_t index = 0; index < model.gates().size(); ++index) {
            auto const &gate = model.gates()[index];
            const auto operands = model.operands(pra::gate_id(index));
            switch (gate.kind) {
            case pra::gate_kind::or_gate:
                ++out.or_gates;
                break;
            case pra::gate_kind::and_gate:
                ++out.and_gates;
                break;
            case pra::gate_kind::cardinality:
                ++out.atleast_gates;
                ++out.atleast_k_of_n[std::to_string(gate.bounds.low) + "/" + std::to_string(operands.size())];
                break;
            default:
                break;
            }
            out.edges += operands.size();
            out.single_input_gates += operands.size() == 1 ? 1 : 0;
            out.max_fan_in = std::max(out.max_fan_in, operands.size());
            for (auto const &operand : operands) {
                const auto kind = operand.event.kind();
                out.complemented_edges += operand.complement ? 1 : 0;
                out.complemented_edges_to_basic_events += operand.complement && kind == pra::element_kind::basic_event ? 1 : 0;
                out.complemented_edges_to_gates += operand.complement && kind == pra::element_kind::gate ? 1 : 0;
                out.literal_operands += operand.literal ? 1 : 0;
                if (const auto child = operand.event.as<pra::gate_id>(); child.valid()) {
                    referenced[child.raw()] = 1;
                }
            }
            out.names_not_ncname += is_ncname(model.stable_name(gate.element)) ? 0 : 1;
        }
        for (std::uint32_t index = 0; index < model.gates().size(); ++index) {
            if (referenced[index] == 0) {
                out.roots.emplace_back(model.stable_name(model.gates()[index].element));
            }
        }
        out.depth = model.summary().gate_depth;

        out.basic_events = model.basic_events().size();
        out.house_events = model.house_events().size();
        for (auto const &event : model.basic_events()) {
            const auto sources = model.sources(event.element);
            out.import_lines += sources.size();
            const auto value = literal_value(model, event);
            out.events_p_1 += value == 1.0 ? 1 : 0;
            out.events_p_0 += value == 0.0 ? 1 : 0;
            out.events_p_tiny += value > 0.0 && value < tiny_probability ? 1 : 0;
            out.names_not_ncname += is_ncname(model.stable_name(event.element)) ? 0 : 1;
            if (sources.size() > 1) {
                duplicate_import duplicate{.name = std::string(model.stable_name(event.element)), .value = value};
                // The reader keeps the chosen record first.
                std::vector<std::pair<pra::source_location, bool>> records;
                for (std::size_t position = 0; position < sources.size(); ++position) {
                    records.emplace_back(sources[position], position == 0);
                }
                std::ranges::sort(records, [](auto const &left, auto const &right) {
                    return std::pair(left.first.file.raw(), left.first.line) < std::pair(right.first.file.raw(), right.first.line);
                });
                for (auto const &[location, chosen] : records) {
                    duplicate.occurrences.push_back(
                        import_occurrence{.file = source_path(model, location), .line = location.line, .chosen = chosen});
                }
                out.duplicate_imports.push_back(std::move(duplicate));
            }
        }
        out.initiating_events = model.initiating_events().size();
        for (auto const &initiator : model.initiating_events()) {
            out.initiating_events_p_1 += literal_value(model, model.basic_events()[initiator.event.raw()]) == 1.0 ? 1 : 0;
        }
        for (auto const &target : model.targets()) {
            out.process_targets.emplace_back(model.stable_name(target.element));
        }
        if (const auto limit = attribute_text(model, "ftp.limit")) {
            out.limit = parse_number(*limit);
        }
        return out;
    }

    auto inspect_sources(const std::span<reader::source_text const> sources, inspect_options const &options) -> inspection {
        inspection out;
        for (auto const &source : sources) {
            out.files.push_back(source.path);
        }
        const auto read = reader::read_project(sources, read_options(options));
        collect(out, read);
        return out;
    }

    auto inspect_project(const std::span<std::filesystem::path const> inputs, inspect_options const &options) -> inspection {
        inspection out;
        if (auto files = reader::project_files(inputs); files) {
            for (auto const &path : files.value()) {
                out.files.push_back(path.generic_string());
            }
        }
        const auto read = reader::read_project(inputs, read_options(options));
        collect(out, read);
        return out;
    }

    auto inspection_json(inspection const &result, const bool counts_only) -> json::value {
        auto root = json::value::make_object();
        root.set("schema_version", json::value::make_int(1));
        root.set("status", json::value::make_string(result.ok() ? "ok" : "error"));
        root.set("counts_only", json::value::make_bool(counts_only));
        root.set("files", name_list(result.files));

        auto diagnostics = json::value::make_array();
        for (auto const &item : result.diagnostics) {
            auto row = json::value::make_object();
            row.set("id", json::value::make_string(item.id));
            row.set("severity", json::value::make_string(severity_name(item.severity)));
            position_json(item.file, item.line, row);
            if (!counts_only) {
                row.set("message", json::value::make_string(item.message));
            }
            diagnostics.push(std::move(row));
        }
        root.set("diagnostics", std::move(diagnostics));

        root.set("structure", result.structure ? structure_json(*result.structure, counts_only) : json::value::make_null());

        auto conventions = json::value::make_object();
        conventions.set("LIMIT", optional_string(result.conventions.limit));
        conventions.set("ftp.negation", optional_string(result.conventions.negation));
        conventions.set("ftp.max_initiators_per_cutset", optional_string(result.conventions.max_initiators_per_cutset));
        conventions.set("header_lines", counts_only ? count_value(result.conventions.header_lines.size())
                                                    : name_list(result.conventions.header_lines));
        root.set("conventions", std::move(conventions));
        return root;
    }

    auto inspection_text(inspection const &result, const bool counts_only) -> std::string {
        std::string out;
        put_line(out, std::string("status: ") + (result.ok() ? "ok" : "error"));
        put_line(out, "files: " + std::to_string(result.files.size()));
        for (auto const &file : result.files) {
            put_line(out, "  " + file);
        }
        put_line(out, "diagnostics: " + std::to_string(result.errors()) + " errors, " + std::to_string(result.warnings()) + " warnings, " +
                      std::to_string(result.count(api::diagnostic_severity::note)) + " notes");
        for (auto const &item : result.diagnostics) {
            std::string text = "  " + severity_name(item.severity) + " " + item.id;
            if (!item.file.empty()) {
                text += " " + item.file + ":" + std::to_string(item.line);
            }
            if (!counts_only) {
                text += ": " + item.message;
            }
            put_line(out, text);
        }
        if (result.structure) {
            auto const &s = *result.structure;
            const auto field = [&out](const std::string_view name, const std::string &value) {
                put_line(out, "  " + std::string(name) + ": " + value);
            };
            const auto number = [](const std::size_t value) { return std::to_string(value); };
            put_line(out, "structure:");
            field("gates", number(s.gates) + " (or " + number(s.or_gates) + ", and " + number(s.and_gates) + ", atleast " +
                               number(s.atleast_gates) + ")");
            for (auto const &[key, count] : s.atleast_k_of_n) {
                field("  atleast " + key, number(count));
            }
            field("edges", number(s.edges));
            field("single_input_gates", number(s.single_input_gates));
            field("max_fan_in", number(s.max_fan_in));
            field("depth", std::to_string(s.depth));
            field("basic_events", number(s.basic_events));
            field("house_events", number(s.house_events));
            field("import_lines", number(s.import_lines));
            field("initiator_tagged_events", number(s.initiating_events) + " (p = 1: " + number(s.initiating_events_p_1) + ")");
            field("events_p_eq_1", number(s.events_p_1));
            field("events_p_eq_0", number(s.events_p_0));
            field("events_0_lt_p_lt_2^-32", number(s.events_p_tiny));
            field("complemented_edges", number(s.complemented_edges) + " (to basic events " +
                                            number(s.complemented_edges_to_basic_events) + ", to gates " +
                                            number(s.complemented_edges_to_gates) + ")");
            field("literal_operands", number(s.literal_operands));
            field("names_not_valid_xml_ncname", number(s.names_not_ncname));
            const auto names = [&](const std::string_view name, std::vector<std::string> const &list) {
                field(name, number(list.size()));
                if (!counts_only) {
                    for (auto const &item : list) {
                        put_line(out, "    " + item);
                    }
                }
            };
            names("roots", s.roots);
            names("process_targets", s.process_targets);
            field("duplicate_imports", number(s.duplicate_imports.size()));
            if (!counts_only) {
                for (auto const &duplicate : s.duplicate_imports) {
                    std::string text = "    " + duplicate.name + " = " + format_number(duplicate.value) + " at";
                    for (auto const &occurrence : duplicate.occurrences) {
                        text += " " + occurrence.file + ":" + std::to_string(occurrence.line) + (occurrence.chosen ? "*" : "");
                    }
                    put_line(out, text);
                }
            }
        }
        put_line(out, "conventions:");
        auto const &c = result.conventions;
        put_line(out, "  LIMIT: " + c.limit.value_or("-"));
        put_line(out, "  ftp.negation: " + c.negation.value_or("-"));
        put_line(out, "  ftp.max_initiators_per_cutset: " + c.max_initiators_per_cutset.value_or("-"));
        put_line(out, "  header_lines: " + std::to_string(c.header_lines.size()));
        if (!counts_only) {
            for (auto const &header : c.header_lines) {
                put_line(out, "    " + header);
            }
        }
        return out;
    }

    auto inspect_exit_status(inspection const &result, const bool strict) -> int {
        return result.errors() > 0 || (strict && result.warnings() > 0) ? 1 : 0;
    }

} // namespace fuzzybits::compile::ftp
