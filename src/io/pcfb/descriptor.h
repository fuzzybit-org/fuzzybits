#pragma once

#include "api/ids.h"
#include "api/options.h"
#include "api/result.h"
#include "io/descriptor.h"
#include "ir/handle.h"
#include "ir/pc/circuit.h"
#include "ir/pc/id.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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

    // Diagnostic ids of the reader. Errors: no ir::pc circuit is produced.
    namespace diag {
        // The bytes are not a verifiable FlatBuffer, or the file cannot be read.
        inline constexpr std::string_view unreadable = "io.pcfb.unreadable";
        // pcfb::validate reported a finding other than an unknown query kind (one error per finding).
        inline constexpr std::string_view invalid_file = "io.pcfb.invalid_file";
        // A query kind the pinned pcfb library does not define, or a minor too old for its kind.
        inline constexpr std::string_view unknown_query_kind = "io.pcfb.unknown_query_kind";
        // The subcircuit lists externally supplied input tensors; ir::pc has no such leaf.
        inline constexpr std::string_view external_inputs = "io.pcfb.external_inputs";
        // An operator that core-1.1 reserves or ir::pc has no op code for.
        inline constexpr std::string_view unsupported_opcode = "io.pcfb.unsupported_opcode";
        // The file has more tensors than a 32-bit ir::pc node id can address.
        inline constexpr std::string_view too_large = "io.pcfb.too_large";
        // The read request names a format other than pcfb.
        inline constexpr std::string_view format_mismatch = "io.pcfb.format_mismatch";
        // The reconstructed ir::pc circuit failed its own sealing checks.
        inline constexpr std::string_view invalid_circuit = "io.pcfb.invalid_circuit";

        // Warnings: the circuit is meaning-preserving, something recorded in the file is dropped.
        // A SAMPLE whose source is COLLAPSED_MODULE or HOUSE_EVENT; it reads as a basic event and is
        // written back as BASIC_EVENT.
        inline constexpr std::string_view sample_source_dropped = "io.pcfb.sample_source_dropped";
        // Query.confidence: a request that ir::pc and write_options cannot hold.
        inline constexpr std::string_view confidence_ignored = "io.pcfb.confidence_ignored";
        // Event ids are not exactly 1..B; the read circuit numbers its events 1..B by ascending file id,
        // so a writer assigns other ids than the file carried and counter-based streams differ.
        inline constexpr std::string_view event_ids_renumbered = "io.pcfb.event_ids_renumbered";
        // A metadata entry the reader interprets has the wrong type or range; it is ignored (it stays in
        // read_output::metadata).
        inline constexpr std::string_view invalid_metadata = "io.pcfb.invalid_metadata";
        // Queries on one target carry different compile.query.<i>.* values; the first query's are kept.
        inline constexpr std::string_view quantification_conflict = "io.pcfb.quantification_conflict";

        // Notes: nothing is lost.
        // BITWISE_NO_OP reads as a one-operand AND.
        inline constexpr std::string_view no_op_as_and = "io.pcfb.no_op_as_and";
        // The file's operator order cannot be reproduced by the writer's emission order for any numbering
        // of the read circuit that keeps basic events in ascending event id; a write of the circuit lists
        // some operators in another order than the file.
        inline constexpr std::string_view emission_order_changed = "io.pcfb.emission_order_changed";
    } // namespace diag

    using metadata_value = std::variant<std::string, std::int64_t, double, std::vector<std::byte>>;

    struct read_output final {
        ir::pc::circuit circuit;
        // Everything the file records that write_options can hold; `query` stays at its default. Every
        // target has an entry in target_kinds, and names are recorded as the file has them (a basic
        // event without a SampleArgs.name has none).
        write_options options;
        // The node holding each tensor's value, indexed by tensor. Tensors that belong to a lowered
        // construct (the ATLEAST and ATMOST of an `AND(ATLEAST, ATMOST)` pair, a NOT tensor that only
        // stands for a complemented edge) have no node: invalid id.
        std::vector<ir::pc::bool_node_id> node_by_tensor;
        // The tensor holding each node's value, indexed by node id.
        std::vector<std::uint32_t> tensor_by_node;
        // Every metadata entry in file order, values verbatim.
        std::vector<std::pair<std::string, metadata_value>> metadata;
        // Circuit.schema_version as read (major << 16 | minor).
        std::uint32_t schema_version = 0;
    };

    using pcfb_read_result = api::result<read_output>;

    [[nodiscard]] auto reader_endpoint_id() -> api::endpoint_id;
    [[nodiscard]] auto reader_descriptor() -> io::reader_descriptor;

    // The inverse of write(): the circuit is numbered so that write() of it emits the operators in the
    // file's order and event ids in the file's order of ascending event id (see docs/pc-pcfb.md,
    // "Reading"). Any minor of schema major 1 is accepted.
    [[nodiscard]] auto read(std::span<std::byte const> bytes, api::option_set const &options = {}) -> pcfb_read_result;
    [[nodiscard]] auto read(std::filesystem::path const &path, api::option_set const &options = {}) -> pcfb_read_result;
    // The circuit as an ir.probabilistic_boolean_dag handle; `request.path` is a file.
    [[nodiscard]] auto read(io::read_request const &request) -> io::read_result;

} // namespace fuzzybits::io::backend::fuzzybits::pcfb
