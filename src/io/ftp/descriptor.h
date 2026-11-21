#pragma once

#include "api/ids.h"
#include "api/options.h"
#include "api/result.h"
#include "io/descriptor.h"
#include "ir/handle.h"

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp {

    struct read_metadata final {
        std::optional<double> truncation_limit;
    };

    struct read_output final {
        ir::handle model;
        read_metadata metadata;
    };

    using ftp_read_result = api::result<read_output>;

    // How an event imported more than once is resolved (`io.ftp.duplicate_import`).
    // `last` and `first` keep one record and warn; `error` rejects the project.
    enum class duplicate_import_policy : std::uint8_t {
        last,
        first,
        error,
    };

    // One file of a project held in memory; `path` is used in source locations
    // and diagnostics.
    struct source_text final {
        std::string path;
        std::string text;
    };

    [[nodiscard]] auto reader_endpoint_id() -> api::endpoint_id;
    [[nodiscard]] auto project_reader_endpoint_id() -> api::endpoint_id;
    [[nodiscard]] auto writer_endpoint_id() -> api::endpoint_id;
    [[nodiscard]] auto truncation_limit_option() -> api::option_key;
    [[nodiscard]] auto target_option() -> api::option_key;
    [[nodiscard]] auto duplicate_import_option() -> api::option_key;
    [[nodiscard]] auto reader_descriptor() -> io::reader_descriptor;
    [[nodiscard]] auto project_reader_descriptor() -> io::reader_descriptor;
    [[nodiscard]] auto writer_descriptor() -> io::writer_descriptor;

    // Expands directories to their `*.ftp` files (sorted by name); files keep
    // the given order.
    [[nodiscard]] auto project_files(std::span<std::filesystem::path const> inputs) -> api::result<std::vector<std::filesystem::path>>;

    // A CAFTA FTP project (one or more files read as their ordered union) into
    // an `ir.pra.model` handle.
    [[nodiscard]] auto read_project(std::span<std::filesystem::path const> inputs, api::option_set const &options = {})
        -> io::read_result;
    [[nodiscard]] auto read_project(std::span<source_text const> sources, api::option_set const &options = {}) -> io::read_result;
    // `request.path` is a file or a directory.
    [[nodiscard]] auto read_project(io::read_request const &request) -> io::read_result;

    // The same grammar lowered directly to `ir.pc` (removed with #182).
    [[nodiscard]] auto read(std::istream &input) -> ftp_read_result;
    [[nodiscard]] auto read(std::istream &input, api::option_set const &options) -> ftp_read_result;
    [[nodiscard]] auto read(std::span<source_text const> sources, api::option_set const &options = {}) -> ftp_read_result;
    [[nodiscard]] auto read(io::read_request const &request) -> io::read_result;

    [[nodiscard]] auto write(ir::handle const &model, std::ostream &output) -> io::write_result;
    [[nodiscard]] auto write(ir::handle const &model, std::ostream &output, io::write_request const &request) -> io::write_result;

} // namespace fuzzybits::io::backend::fuzzybits::ftp
