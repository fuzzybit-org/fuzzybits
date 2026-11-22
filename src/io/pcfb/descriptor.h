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
#include <string_view>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::pcfb {

    // pcfb QueryKind. monotone_hull_probability is a pcfb 1.2 feature (docs/rfc/monotone-hull-queries.md):
    // a file that uses it declares schema minor 2, any other file stays at minor 1.
    enum class query_kind : std::uint8_t {
        probability,
        monotone_hull_probability,
    };

    // The spelling of the io.pcfb.query_kind option and of the quantification metadata.
    [[nodiscard]] auto to_string(query_kind kind) noexcept -> std::string_view;

    // What the caller asked of one target, recorded as pcfb metadata next to its queries
    // (compile.query.<i>.*, i the position in the file's query list) so that fzb inspect and the
    // runtime can show it. The negation semantics of a query is its kind; this adds the rest.
    struct target_quantification final {
        // ir::pra::literal_mode as text (atom_split, omega_l, negation); empty: not recorded.
        std::string literal_mode;
        // The initiator rule already lowered into the target's logic; unset: no rule.
        std::optional<std::uint32_t> max_initiators_per_cutset;
    };

    // ir::pc carries no names or provenance, so everything here comes from the
    // caller (for example an ir::pra projection). Empty strings mean "absent".
    struct write_options final {
        // Query names by PC target id; missing targets are named "t<target id>".
        std::map<ir::pc::target_id, std::string> target_names;
        // Query kinds by PC target id, in file order. Several kinds make several queries on one
        // tensor, all carrying the target's name (the kind tells them apart). An entry must be
        // non-empty, without repeats, and name a target of the circuit.
        std::map<ir::pc::target_id, std::vector<query_kind>> target_kinds;
        // Recorded per target; entries must name a target of the circuit.
        std::map<ir::pc::target_id, target_quantification> quantification;
        // Op names by PC node id: SampleArgs.name for basic events, Op.name for
        // every other node. Missing basic events are named "e<event_id>"; other
        // nodes stay unnamed.
        std::map<ir::pc::bool_node_id, std::string> node_names;
        std::string model_name;
        std::string source_format;
        std::optional<std::array<std::byte, 32>> source_sha256;
        std::string compile_options;
        // The kind of every target without an entry in target_kinds (option io.pcfb.query_kind).
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
