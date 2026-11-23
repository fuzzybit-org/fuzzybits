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
#include <ios>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::pcfb {

auto reader_endpoint_id() -> api::endpoint_id { return api::endpoint_id("io.fuzzybits.pcfb.reader"); }

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
        return pcfb_read_result::fail(
            api::make_error(std::string(diag::unreadable), "failed to open '" + path.generic_string() + "'"));
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (!input || size < 0) {
        return pcfb_read_result::fail(
            api::make_error(std::string(diag::unreadable), "failed to read '" + path.generic_string() + "'"));
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        return pcfb_read_result::fail(
            api::make_error(std::string(diag::unreadable), "failed to read '" + path.generic_string() + "'"));
    }
    return read(std::span<std::byte const>{bytes}, options);
}

auto read(io::read_request const &request) -> io::read_result {
    if (!request.format.empty() && request.format != formats::pcfb()) {
        return io::read_result::fail(
            api::make_error(std::string(diag::format_mismatch), "pcfb reader received a non-pcfb format request"));
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
