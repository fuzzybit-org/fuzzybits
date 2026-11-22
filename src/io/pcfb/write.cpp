#include "io/pcfb/descriptor.h"
#include "io/pcfb/detail/model.h"

#include "api/capabilities.h"
#include "api/diagnostics.h"
#include "api/ids.h"
#include "api/options.h"
#include "io/descriptor.h"
#include "io/formats.h"
#include "ir/handle.h"
#include "ir/pc/circuit.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <ios>
#include <span>
#include <vector>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace fuzzybits::io::backend::fuzzybits::pcfb {
    namespace {

        [[nodiscard]] auto parseQueryKind(std::string_view text) -> std::optional<query_kind> {
            for (const auto kind : {query_kind::probability, query_kind::monotone_hull_probability}) {
                if (text == to_string(kind)) {
                    return kind;
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] auto optionsFrom(io::write_request const &request) -> api::result<write_options> {
            write_options options;
            api::diagnostic_list diagnostics;
            if (auto value = request.options.get<std::string>(model_name_option())) {
                options.model_name = std::move(*value);
            }
            if (auto value = request.options.get<std::string>(source_format_option())) {
                options.source_format = std::move(*value);
            }
            if (auto value = request.options.get<std::string>(compile_options_option())) {
                options.compile_options = std::move(*value);
            }
            if (auto value = request.options.get<std::string>(source_sha256_option()); value && !value->empty()) {
                options.source_sha256 = detail::parse_sha256_hex(*value);
                if (!options.source_sha256) {
                    diagnostics.push(api::make_error("io.pcfb.invalid-option", "io.pcfb.source_sha256 must be 64 hexadecimal digits"));
                }
            }
            if (auto value = request.options.get<std::string>(query_kind_option())) {
                if (const auto kind = parseQueryKind(*value)) {
                    options.query = *kind;
                } else {
                    diagnostics.push(api::make_error("io.pcfb.invalid-option", "io.pcfb.query_kind '" + *value + "' is not a pcfb query kind (expected '" +
                                                                                   std::string(to_string(query_kind::probability)) + "' or '" +
                                                                                   std::string(to_string(query_kind::monotone_hull_probability)) + "')"));
                }
            }
            if (diagnostics.has_errors()) {
                return api::result<write_options>::fail(std::move(diagnostics));
            }
            return api::result<write_options>::ok(std::move(options));
        }

    } // namespace

    auto to_string(const query_kind kind) noexcept -> std::string_view {
        switch (kind) {
        case query_kind::probability:
            return "probability";
        case query_kind::monotone_hull_probability:
            return "monotone_hull_probability";
        }
        return "probability";
    }

    auto writer_endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("io.fuzzybits.pcfb.writer");
    }

    auto model_name_option() -> api::option_key {
        return api::option_key("io.pcfb.model_name");
    }

    auto source_format_option() -> api::option_key {
        return api::option_key("io.pcfb.source_format");
    }

    auto source_sha256_option() -> api::option_key {
        return api::option_key("io.pcfb.source_sha256");
    }

    auto compile_options_option() -> api::option_key {
        return api::option_key("io.pcfb.compile_options");
    }

    auto query_kind_option() -> api::option_key {
        return api::option_key("io.pcfb.query_kind");
    }

    auto writer_descriptor() -> io::writer_descriptor {
        return io::writer_descriptor{
            .name = "FuzzyBits pcfb writer",
            .provider = api::provider_id("fuzzybits"),
            .endpoint = writer_endpoint_id(),
            .output_format = formats::pcfb(),
            .accepts =
                api::capability_set{
                    api::capability_id("ir.probabilistic_boolean_dag"),
                    api::capability_id("ir.explicit_edge_polarity"),
                    api::capability_id("ir.targets"),
                    api::capability_id("ir.literal_probabilities"),
                },
            .options =
                api::option_schema{.entries = {
                                       api::option_spec{
                                           .key = model_name_option(),
                                           .name = "Model name",
                                           .default_value = std::string{},
                                           .help = "Circuit name and general.model_name.",
                                       },
                                       api::option_spec{
                                           .key = source_format_option(),
                                           .name = "Source format",
                                           .default_value = std::string{},
                                           .help = "general.source_format, e.g. cafta-ftp.",
                                       },
                                       api::option_spec{
                                           .key = source_sha256_option(),
                                           .name = "Source SHA-256",
                                           .default_value = std::string{},
                                           .help = "general.source_sha256 as 64 hexadecimal digits.",
                                       },
                                       api::option_spec{
                                           .key = compile_options_option(),
                                           .name = "Compile options",
                                           .default_value = std::string{},
                                           .help = "compile.options, the flags that produced the circuit.",
                                       },
                                       api::option_spec{
                                           .key = query_kind_option(),
                                           .name = "Query kind",
                                           .default_value = std::string{to_string(query_kind::probability)},
                                           .help = "Query.kind of every target without an explicit kind: 'probability' or "
                                                   "'monotone_hull_probability' (pcfb 1.2, schema minor 2).",
                                       },
                                   }},
        };
    }

    auto write(ir::handle const &model, write_options const &options) -> pcfb_write_result {
        auto const *pc = ir::pc::circuit_from(model);
        if (pc == nullptr) {
            return pcfb_write_result::fail(api::make_error("io.pcfb.unsupported-ir", "pcfb writer requires a PC IR handle"));
        }
        return detail::write_pc(*pc, options);
    }

    auto write(ir::handle const &model, std::ostream &output) -> io::write_result {
        return write(model, output, io::write_request{.format = formats::pcfb()});
    }

    auto write(ir::handle const &model, std::ostream &output, io::write_request const &request) -> io::write_result {
        if (!request.format.empty() && request.format != formats::pcfb()) {
            return io::write_result::fail(api::make_error("io.pcfb.format-mismatch", "pcfb writer received a non-pcfb format request"));
        }
        auto options = optionsFrom(request);
        if (!options) {
            return io::write_result::fail(std::move(options.diagnostics()));
        }
        auto written = write(model, options.value());
        if (!written) {
            return io::write_result::fail(std::move(written.diagnostics()));
        }
        auto const &bytes = written.value().bytes;
        output.write(reinterpret_cast<char const *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!output) {
            return io::write_result::fail(api::make_error("io.pcfb.write-failed", "failed to write pcfb bytes to the output stream"));
        }
        return io::write_result::ok(true, std::move(written.diagnostics()));
    }

    auto reader_endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("io.fuzzybits.pcfb.reader");
    }

    auto reader_descriptor() -> io::reader_descriptor {
        return io::reader_descriptor{
            .name = "FuzzyBits pcfb reader",
            .provider = api::provider_id("fuzzybits"),
            .endpoint = reader_endpoint_id(),
            .input_format = formats::pcfb(),
            .produces =
                api::capability_set{
                    api::capability_id("ir.probabilistic_boolean_dag"),
                    api::capability_id("ir.explicit_edge_polarity"),
                    api::capability_id("ir.targets"),
                    api::capability_id("ir.literal_probabilities"),
                },
            .options = {},
        };
    }

    auto read(std::span<std::byte const> bytes, api::option_set const &) -> pcfb_read_result {
        return detail::read_pcfb(bytes);
    }

    auto read(std::filesystem::path const &path, api::option_set const &options) -> pcfb_read_result {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return pcfb_read_result::fail(api::make_error(std::string(diag::unreadable), "failed to open '" + path.generic_string() + "'"));
        }
        const std::vector<char> text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        if (input.bad()) {
            return pcfb_read_result::fail(api::make_error(std::string(diag::unreadable), "failed to read '" + path.generic_string() + "'"));
        }
        std::vector<std::byte> bytes(text.size());
        std::ranges::transform(text, bytes.begin(), [](const char c) { return static_cast<std::byte>(c); });
        return read(std::span<std::byte const>{bytes}, options);
    }

    auto read(io::read_request const &request) -> io::read_result {
        if (!request.format.empty() && request.format != formats::pcfb()) {
            return io::read_result::fail(api::make_error(std::string(diag::format_mismatch), "pcfb reader received a non-pcfb format request"));
        }
        auto read = pcfb::read(request.path, request.options);
        if (!read.has_value()) {
            return io::read_result::fail(std::move(read.diagnostics()));
        }
        auto diagnostics = std::move(read.diagnostics());
        if (diagnostics.has_errors()) {
            return io::read_result::fail(std::move(diagnostics));
        }
        return io::read_result::ok(read.value().circuit.to_handle(), std::move(diagnostics));
    }

} // namespace fuzzybits::io::backend::fuzzybits::pcfb
