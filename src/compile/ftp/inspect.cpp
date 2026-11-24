#include "compile/ftp/inspect.h"

#include "ir/pra/model.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

namespace fuzzybits::compile::ftp {

namespace {

namespace pra = ir::pra;
namespace reader = io::backend::fuzzybits::ftp;

[[nodiscard]] auto attribute_text(pra::model const &model, const std::string_view name) -> std::optional<std::string> {
    for (auto const &attribute : model.project_attributes()) {
        if (model.text(attribute.name) == name) {
            return std::string(model.text(attribute.value));
        }
    }
    return std::nullopt;
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

[[nodiscard]] auto read_options(inspect_options const &options) -> api::option_set {
    api::option_set out;
    out.set(reader::duplicate_import_option(), options.duplicate_import);
    out.set(reader::complement_events_option(), options.complement_events);
    return out;
}

void collect(inspection &out, io::read_result const &read) {
    for (auto const &item : read.diagnostics().items()) {
        out.diagnostics.push_back(inspected_diagnostic{
            .id = item.code, .severity = item.severity, .file = item.file, .line = item.line, .message = item.message});
    }
    if (!read || read.diagnostics().has_errors()) {
        return;
    }
    auto const *model = pra::model_from(read.value());
    if (model == nullptr) {
        return;
    }
    out.structure = reader::structure_of(*model);
    auto &conventions = out.conventions;
    conventions.limit = attribute_text(*model, "ftp.limit");
    conventions.maxsize = attribute_text(*model, "ftp.maxsize");
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

[[nodiscard]] auto names_json(std::vector<std::string> const &names, const bool counts_only) -> json::value {
    return counts_only ? count_value(names.size()) : name_list(names);
}

[[nodiscard]] auto structure_json(reader::structure_summary const &s, const bool counts_only) -> json::value {
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
    out.set("roots", names_json(s.roots, counts_only));
    out.set("process_targets", names_json(s.process_targets, counts_only));
    out.set("default_targets", names_json(s.default_targets, counts_only));
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
    if (options.duplicate_import != "error" && options.duplicate_import != "first" &&
        options.duplicate_import != "last") {
        return "--duplicate-import must be error, first or last, not '" + options.duplicate_import + "'";
    }
    if (options.complement_events != "atom" && options.complement_events != "negation") {
        return "--complement-events must be atom or negation, not '" + options.complement_events + "'";
    }
    return {};
}

auto inspection::count(const api::diagnostic_severity severity) const -> std::size_t {
    return static_cast<std::size_t>(
        std::ranges::count_if(diagnostics, [severity](auto const &item) { return item.severity == severity; }));
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

auto inspect_project(const std::span<std::filesystem::path const> inputs, inspect_options const &options)
    -> inspection {
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
    conventions.set("MAXSIZE", optional_string(result.conventions.maxsize));
    conventions.set("ftp.max_initiators_per_cutset", optional_string(result.conventions.max_initiators_per_cutset));
    conventions.set("header_lines", names_json(result.conventions.header_lines, counts_only));
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
    put_line(out, "diagnostics: " + std::to_string(result.errors()) + " errors, " + std::to_string(result.warnings()) +
                      " warnings, " + std::to_string(result.count(api::diagnostic_severity::note)) + " notes");
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
        field("initiator_tagged_events",
              number(s.initiating_events) + " (p = 1: " + number(s.initiating_events_p_1) + ")");
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
        names("default_targets", s.default_targets);
        field("duplicate_imports", number(s.duplicate_imports.size()));
        if (!counts_only) {
            for (auto const &duplicate : s.duplicate_imports) {
                std::string text = "    " + duplicate.name + " = " + format_number(duplicate.value) + " at";
                for (auto const &occurrence : duplicate.occurrences) {
                    text +=
                        " " + occurrence.file + ":" + std::to_string(occurrence.line) + (occurrence.chosen ? "*" : "");
                }
                put_line(out, text);
            }
        }
    }
    put_line(out, "conventions:");
    auto const &c = result.conventions;
    put_line(out, "  LIMIT: " + c.limit.value_or("-"));
    put_line(out, "  MAXSIZE: " + c.maxsize.value_or("-"));
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
