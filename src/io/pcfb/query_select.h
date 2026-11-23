#pragma once

#include "api/result.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/id.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Choosing one query of a .pcfb by position, name and kind (#202).
namespace fuzzybits::io::backend::fuzzybits::pcfb {

    // "probability" or "hull": the spelling of `--kind` and of the JSON `kind` field.
    [[nodiscard]] auto query_kind_name(query_kind kind) noexcept -> std::string_view;
    [[nodiscard]] auto parse_query_kind(std::string_view text) -> std::optional<query_kind>;

    // One query of a file, whatever read the file: `name` is the query's own name,
    // or "t<target>` when it has none.
    struct query_info final {
        // Position in the file's query list.
        std::uint32_t index = 0;
        ir::pc::target_id target;
        query_kind kind = query_kind::probability;
        std::string name;
    };

    [[nodiscard]] auto queries_of(read_output const &file) -> std::vector<query_info>;

    struct query_selection final {
        // Position in the file's query list.
        std::uint32_t index = 0;
        ir::pc::target_id target;
        query_kind kind = query_kind::probability;
        std::string name;
    };

    // Query 0, or the one query named `name`. Without `kind`, a name that the file
    // writes as several kinds (two queries on one target, #218) is an error that
    // lists them; with `kind`, that kind of that name.
    [[nodiscard]] auto select_query(std::span<query_info const> queries, std::optional<std::string_view> name,
                                    std::optional<query_kind> kind) -> api::result<query_selection>;

    // As above, but only a hull query will do: with no name and no kind, the file's
    // only hull query is selected; with several, the selection is an error.
    [[nodiscard]] auto select_hull_query(std::span<query_info const> queries, std::optional<std::string_view> name,
                                         std::optional<query_kind> kind) -> api::result<query_selection>;

    // The read circuit's basic-event node with that name, or an error that names it.
    // A named node that is not a basic event is `io.pcfb.not-an-event`.
    [[nodiscard]] auto basic_event_named(read_output const &file, std::string_view name)
        -> api::result<ir::pc::bool_node_id>;

    // The node's name, or "e<node id>" when the file records none.
    [[nodiscard]] auto node_name(read_output const &file, ir::pc::bool_node_id node) -> std::string;

} // namespace fuzzybits::io::backend::fuzzybits::pcfb
