#pragma once

#include "ir/pra/id.h"

#include <cstdint>

namespace fuzzybits::ir::pra {

// External key/value data that is not model semantics: format header lines,
// quantification hints such as FTP `LIMIT`, unknown source flags.
struct attribute final {
    text_ref name;
    text_ref value;
};

struct source_file final {
    text_ref path;
    // Lowercase hex digest of the file contents, or empty when not computed.
    text_ref sha256;
    // An io format id, for example `ftp`.
    text_ref format;
};

// `line` is 1-based; 0 means the line is unknown.
struct source_location final {
    source_file_id file;
    std::uint32_t line = 0;
};

// One row per entity. `owner` points back at the entity's row in its own table.
struct element_metadata final {
    domain_ref owner;
    text_ref stable_name;
    text_ref qualified_name;
    text_ref label;
    table_range<attribute> attributes;
    table_range<source_location> sources;
};

struct project_metadata final {
    text_ref name;
    text_ref label;
    table_range<attribute> attributes;
};

} // namespace fuzzybits::ir::pra
