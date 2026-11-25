#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace fuzzybits::ir::pc {

// The quantity a query asks of one target (docs/rfc/monotone-hull-queries.md). ir::pc targets never
// carry it (IR-PC §2): the ir::pra lowering records it per target binding, the passes take it per
// target, and the .pcfb writer turns it into a Query kind.
enum class query_kind : std::uint8_t {
    // P(target): the probability of the target function.
    probability,
    // P(h): the monotone hull, the delete-term reading of complements that cut-set tools apply.
    monotone_hull,
};

// The spelling of the CLI, the config document and the reports: `probability` or `hull`.
[[nodiscard]] constexpr auto to_string(const query_kind kind) noexcept -> std::string_view {
    return kind == query_kind::probability ? std::string_view{"probability"} : std::string_view{"hull"};
}

[[nodiscard]] constexpr auto parse_query_kind(const std::string_view text) noexcept -> std::optional<query_kind> {
    if (text == "probability") {
        return query_kind::probability;
    }
    if (text == "hull") {
        return query_kind::monotone_hull;
    }
    return std::nullopt;
}

} // namespace fuzzybits::ir::pc
