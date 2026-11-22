#pragma once

#include "pcfb.h"
#include "pcfb/ir/circuit.h"

#include <cstdint>
#include <optional>
#include <string>

namespace fuzzybits::io::pcfb_guard {

    inline constexpr char unsupported_query_kind_id[] = "pcfb.unsupported-query-kind";

    struct rejection final {
        // unsupported_query_kind_id; stable across releases.
        std::string id;
        std::string message;
    };

    [[nodiscard]] inline auto query_kind_name(::pcfb::QueryKind kind) -> std::string {
        switch (kind) {
        case ::pcfb::QueryKind::PROBABILITY:
            return "PROBABILITY";
        case ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY:
            return "MONOTONE_HULL_PROBABILITY";
        }
        return "unknown(" + std::to_string(static_cast<unsigned>(kind)) + ")";
    }

    // Sampling and exact evaluation answer Boolean probabilities only. A runtime that does not implement a
    // kind must reject the file before computing anything and never substitute another kind (pcfb spec 1.2
    // section 8), so this checks every query of every subcircuit, not only the one about to be evaluated.
    // The schema minor is not checked: readers accept any minor of major 1 (spec 1.1 section 2), and an
    // unknown kind is rejected here whichever minor the file declares. Structural problems are left to
    // pcfb::validate.
    [[nodiscard]] inline auto reject_unimplemented_queries(::pcfb::ir::Circuit const &circuit) -> std::optional<rejection> {
        if (!circuit.valid()) {
            return std::nullopt;
        }
        auto const *root = circuit.get();
        if (root->subcircuits() == nullptr) {
            return std::nullopt;
        }
        for (auto const *sub : *root->subcircuits()) {
            if (sub == nullptr || sub->queries() == nullptr) {
                continue;
            }
            for (flatbuffers::uoffset_t i = 0; i < sub->queries()->size(); ++i) {
                auto const *query = sub->queries()->Get(i);
                if (query == nullptr || query->kind() == ::pcfb::QueryKind::PROBABILITY) {
                    continue;
                }
                auto const name = query->name() != nullptr ? query->name()->str() : "q" + std::to_string(i);
                return rejection{unsupported_query_kind_id,
                                 "query '" + name + "' has kind " + query_kind_name(query->kind()) +
                                     ", which this runtime does not implement; only PROBABILITY queries can be " +
                                     "answered, and a query is never answered as another kind"};
            }
        }
        return std::nullopt;
    }

} // namespace fuzzybits::io::pcfb_guard
