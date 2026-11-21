#pragma once

#include "api/ids.h"
#include "api/options.h"
#include "api/result.h"
#include "io/descriptor.h"
#include "ir/handle.h"
#include "ir/pc/id.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::pcfb {

    // pcfb core-1.1 has QueryKind::PROBABILITY only; monotone-hull queries (#174)
    // extend this enum together with the pinned schema.
    enum class query_kind : std::uint8_t {
        probability,
    };

    // ir::pc carries no names or provenance, so everything here comes from the
    // caller (for example an ir::pra projection). Empty strings mean "absent".
    struct write_options final {
        // Query names by PC target id; missing targets are named "t<target id>".
        std::map<ir::pc::target_id, std::string> target_names;
        // Op names by PC node id: SampleArgs.name for basic events, Op.name for
        // every other node. Missing basic events are named "e<event_id>"; other
        // nodes stay unnamed.
        std::map<ir::pc::bool_node_id, std::string> node_names;
        std::string model_name;
        std::string source_format;
        std::optional<std::array<std::byte, 32>> source_sha256;
        std::string compile_options;
        query_kind query = query_kind::probability;
    };

    struct write_output final {
        std::vector<std::byte> bytes;
        // pcfb tensor holding each PC node's value, indexed by node id.
        std::vector<std::uint32_t> tensor_by_node;
        // SampleArgs.event_id per node id; 0 (reserved by pcfb runtimes) for
        // nodes that are not basic events.
        std::vector<std::uint32_t> event_id_by_node;
    };

    using pcfb_write_result = api::result<write_output>;

    [[nodiscard]] auto writer_endpoint_id() -> api::endpoint_id;
    [[nodiscard]] auto model_name_option() -> api::option_key;
    [[nodiscard]] auto source_format_option() -> api::option_key;
    [[nodiscard]] auto source_sha256_option() -> api::option_key;
    [[nodiscard]] auto compile_options_option() -> api::option_key;
    [[nodiscard]] auto query_kind_option() -> api::option_key;
    [[nodiscard]] auto writer_descriptor() -> io::writer_descriptor;

    // event_id contract (selects the counter-based PRNG stream, see
    // docs/pc-pcfb.md): basic-event node v gets 1 + |{u < v : u is a basic
    // event}|, i.e. 1..B in ascending node id, independent of emission order.
    [[nodiscard]] auto write(ir::handle const &model, write_options const &options) -> pcfb_write_result;

    [[nodiscard]] auto write(ir::handle const &model, std::ostream &output) -> io::write_result;
    [[nodiscard]] auto write(ir::handle const &model, std::ostream &output, io::write_request const &request) -> io::write_result;

} // namespace fuzzybits::io::backend::fuzzybits::pcfb
