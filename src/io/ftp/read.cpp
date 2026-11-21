#include "io/ftp/descriptor.h"
#include "io/ftp/detail/model.h"

#include "api/capabilities.h"
#include "api/diagnostics.h"
#include "api/ids.h"
#include "api/options.h"
#include "io/descriptor.h"
#include "io/formats.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <istream>
#include <iterator>
#include <span>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp {

    namespace detail {

        auto parse_options(api::option_set const &options, api::diagnostic_list &diagnostics) -> duplicate_import_policy {
            const auto value = options.get<std::string>(duplicate_import_option());
            if (!value || *value == "last") {
                return duplicate_import_policy::last;
            }
            if (*value == "first") {
                return duplicate_import_policy::first;
            }
            if (*value == "error") {
                return duplicate_import_policy::error;
            }
            diagnostics.push(api::make_error("io.ftp.invalid-option", "io.ftp.duplicate_import must be last, first or error, not '" +
                                                                          *value + "'"));
            return duplicate_import_policy::last;
        }

        auto load_files(std::span<std::filesystem::path const> inputs) -> api::result<std::vector<source_text>> {
            auto files = project_files(inputs);
            if (!files) {
                return api::result<std::vector<source_text>>::fail(std::move(files.diagnostics()));
            }
            std::vector<source_text> sources;
            sources.reserve(files.value().size());
            for (auto const &path : files.value()) {
                std::ifstream input(path, std::ios::binary);
                if (!input) {
                    return api::result<std::vector<source_text>>::fail(
                        api::make_error("io.ftp.open-failed", "failed to open FTP input file '" + path.generic_string() + "'"));
                }
                std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
                sources.push_back(source_text{.path = path.generic_string(), .text = std::move(text)});
            }
            return api::result<std::vector<source_text>>::ok(std::move(sources));
        }

    } // namespace detail

    namespace {

        [[nodiscard]] auto hasFtpExtension(std::filesystem::path const &path) -> bool {
            auto extension = path.extension().string();
            std::ranges::transform(extension, extension.begin(), [](const unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            return extension == ".ftp";
        }

        [[nodiscard]] auto parseSources(std::vector<source_text> sources, api::option_set const &options) -> detail::ParsedProject {
            api::diagnostic_list optionDiagnostics;
            const auto policy = detail::parse_options(options, optionDiagnostics);
            auto project = detail::parse(std::move(sources), policy);
            optionDiagnostics.append(std::move(project.diagnostics));
            project.diagnostics = std::move(optionDiagnostics);
            return project;
        }

        [[nodiscard]] auto toPra(detail::ParsedProject project) -> io::read_result {
            if (project.diagnostics.has_errors()) {
                return io::read_result::fail(std::move(project.diagnostics));
            }
            if (project.targets.empty()) {
                project.diagnostics.push(api::make_error("io.ftp.no-targets", "FTP project has no PROCESS target and no root gate"));
                return io::read_result::fail(std::move(project.diagnostics));
            }
            auto [model, diagnostics] = detail::build_pra(project);
            project.diagnostics.append(std::move(diagnostics));
            if (project.diagnostics.has_errors() || !model.valid()) {
                return io::read_result::fail(std::move(project.diagnostics));
            }
            return io::read_result::ok(model.to_handle(), std::move(project.diagnostics));
        }

        [[nodiscard]] auto toPc(detail::ParsedProject project, api::option_set const &options) -> ftp_read_result {
            if (auto overrideLimit = options.get<double>(truncation_limit_option())) {
                project.metadata.truncation_limit = *overrideLimit;
            }
            const auto targets = detail::selected_targets(project, options, project.diagnostics);
            if (project.diagnostics.has_errors()) {
                return ftp_read_result::fail(std::move(project.diagnostics));
            }
            if (targets.empty()) {
                project.diagnostics.push(api::make_error("io.ftp.no-targets", "FTP input did not contain an importable target gate"));
                return ftp_read_result::fail(std::move(project.diagnostics));
            }

            auto [output, diagnostics] = detail::build_pc(project, targets);
            project.diagnostics.append(std::move(diagnostics));
            if (project.diagnostics.has_errors() || !output.valid()) {
                return ftp_read_result::fail(std::move(project.diagnostics));
            }
            return ftp_read_result::ok(
                read_output{
                    .model = output.to_handle(),
                    .metadata = project.metadata,
                },
                std::move(project.diagnostics));
        }

        [[nodiscard]] auto formatMismatch(io::read_request const &request) -> bool {
            return !request.format.empty() && request.format != formats::ftp();
        }

    } // namespace

    auto reader_endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("io.fuzzybits.ftp.reader");
    }

    auto project_reader_endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("io.fuzzybits.ftp.project_reader");
    }

    auto truncation_limit_option() -> api::option_key {
        return api::option_key("io.ftp.truncation_limit");
    }

    auto target_option() -> api::option_key {
        return api::option_key("io.ftp.target");
    }

    auto duplicate_import_option() -> api::option_key {
        return api::option_key("io.ftp.duplicate_import");
    }

    namespace {

        [[nodiscard]] auto duplicateImportSpec() -> api::option_spec {
            return api::option_spec{
                .key = duplicate_import_option(),
                .name = "Duplicate import",
                .default_value = std::string("last"),
                .help = "Value of an event imported more than once: last (default) or first, with a warning naming every record, "
                        "or error. An I tag on any record makes the event an initiating event.",
            };
        }

    } // namespace

    auto reader_descriptor() -> io::reader_descriptor {
        return io::reader_descriptor{
            .name = "FuzzyBits FTP reader",
            .provider = api::provider_id("fuzzybits"),
            .endpoint = reader_endpoint_id(),
            .input_format = formats::ftp(),
            .produces =
                api::capability_set{
                    api::capability_id("ir.probabilistic_boolean_dag"),
                    api::capability_id("ir.explicit_edge_polarity"),
                    api::capability_id("ir.targets"),
                    api::capability_id("ir.literal_probabilities"),
                    api::capability_id("ir.parameter_probabilities"),
                },
            .options =
                api::option_schema{.entries = {
                                       api::option_spec{
                                           .key = truncation_limit_option(),
                                           .name = "Truncation limit",
                                           .default_value = std::monostate{},
                                           .help = "Optional reader-side cut/truncation limit override for FTP files.",
                                       },
                                       api::option_spec{
                                           .key = target_option(),
                                           .name = "Target",
                                           .default_value = std::string{},
                                           .help = "Optional FTP symbol to import as the sole PC target.",
                                       },
                                       duplicateImportSpec(),
                                   }},
        };
    }

    auto project_reader_descriptor() -> io::reader_descriptor {
        return io::reader_descriptor{
            .name = "FuzzyBits CAFTA FTP project reader",
            .provider = api::provider_id("fuzzybits"),
            .endpoint = project_reader_endpoint_id(),
            .input_format = formats::ftp(),
            .produces =
                api::capability_set{
                    api::capability_id("ir.pra.model"),
                    api::capability_id("ir.pra.domain_targets"),
                    api::capability_id("ir.pra.source_locations"),
                },
            .options = api::option_schema{.entries = {duplicateImportSpec()}},
        };
    }

    auto project_files(std::span<std::filesystem::path const> inputs) -> api::result<std::vector<std::filesystem::path>> {
        std::vector<std::filesystem::path> files;
        for (auto const &input : inputs) {
            std::error_code error;
            if (!std::filesystem::is_directory(input, error)) {
                files.push_back(input);
                continue;
            }
            std::vector<std::filesystem::path> entries;
            for (auto iterator = std::filesystem::directory_iterator(input, error); !error && iterator != std::filesystem::directory_iterator{};
                 iterator.increment(error)) {
                if (std::error_code status; iterator->is_regular_file(status) && hasFtpExtension(iterator->path())) {
                    entries.push_back(iterator->path());
                }
            }
            if (error) {
                return api::result<std::vector<std::filesystem::path>>::fail(api::make_error(
                    "io.ftp.open-failed", "failed to list FTP project directory '" + input.generic_string() + "': " + error.message()));
            }
            std::ranges::sort(entries);
            files.insert(files.end(), entries.begin(), entries.end());
        }
        if (files.empty()) {
            return api::result<std::vector<std::filesystem::path>>::fail(api::make_error("io.ftp.no-files", "no .ftp files given"));
        }
        return api::result<std::vector<std::filesystem::path>>::ok(std::move(files));
    }

    auto read_project(std::span<std::filesystem::path const> inputs, api::option_set const &options) -> io::read_result {
        auto sources = detail::load_files(inputs);
        if (!sources) {
            return io::read_result::fail(std::move(sources.diagnostics()));
        }
        return toPra(parseSources(std::move(sources).value(), options));
    }

    auto read_project(std::span<source_text const> sources, api::option_set const &options) -> io::read_result {
        return toPra(parseSources(std::vector<source_text>(sources.begin(), sources.end()), options));
    }

    auto read_project(io::read_request const &request) -> io::read_result {
        if (formatMismatch(request)) {
            return io::read_result::fail(api::make_error("io.ftp.format-mismatch", "FTP reader received a non-FTP format request"));
        }
        return read_project(std::span{&request.path, 1}, request.options);
    }

    auto read(std::istream &input) -> ftp_read_result {
        const api::option_set options;
        return read(input, options);
    }

    auto read(std::istream &input, api::option_set const &options) -> ftp_read_result {
        std::ostringstream text;
        text << input.rdbuf();
        std::vector<source_text> sources;
        sources.push_back(source_text{.path = "<input>", .text = std::move(text).str()});
        return toPc(parseSources(std::move(sources), options), options);
    }

    auto read(std::span<source_text const> sources, api::option_set const &options) -> ftp_read_result {
        return toPc(parseSources(std::vector<source_text>(sources.begin(), sources.end()), options), options);
    }

    auto read(io::read_request const &request) -> io::read_result {
        if (formatMismatch(request)) {
            return io::read_result::fail(api::make_error("io.ftp.format-mismatch", "FTP reader received a non-FTP format request"));
        }

        auto sources = detail::load_files(std::span{&request.path, 1});
        if (!sources) {
            return io::read_result::fail(std::move(sources.diagnostics()));
        }
        auto result = toPc(parseSources(std::move(sources).value(), request.options), request.options);
        if (!result) {
            return io::read_result::fail(result.diagnostics());
        }
        auto diagnostics = result.diagnostics();
        auto model = std::move(result).value().model;
        return io::read_result::ok(std::move(model), std::move(diagnostics));
    }

} // namespace fuzzybits::io::backend::fuzzybits::ftp
