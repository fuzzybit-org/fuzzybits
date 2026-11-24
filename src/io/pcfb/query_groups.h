#pragma once

// The query groups of a .pcfb (pcfb SubCircuit.query_groups, spec §5.4), such as initiating-event
// frequencies. Header-only: consumers already link pcfb.

#include "pcfb.h"
#include "pcfb/ir/circuit.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fuzzybits::io::pcfb_frequency {

struct group_term final {
    std::uint32_t query = 0;
    double weight = 0.0;
};

struct query_group final {
    std::string name;
    std::string unit;
    std::vector<group_term> terms;
};

// The query groups of the file's first subcircuit (pcfb::validate has checked them, rule 9).
[[nodiscard]] inline auto query_groups_of(::pcfb::ir::Circuit const &circuit) -> std::vector<query_group> {
    std::vector<query_group> out;
    if (!circuit.valid() || circuit.get()->subcircuits() == nullptr || circuit.get()->subcircuits()->size() == 0) {
        return out;
    }
    auto const *groups = circuit.get()->subcircuits()->Get(0)->query_groups();
    if (groups == nullptr) {
        return out;
    }
    for (auto const *group : *groups) {
        query_group entry;
        entry.name = group->name() != nullptr ? group->name()->str() : std::string{};
        entry.unit = group->unit() != nullptr ? group->unit()->str() : std::string{};
        if (group->terms() != nullptr) {
            for (auto const *term : *group->terms()) {
                entry.terms.push_back(group_term{.query = term->query(), .weight = term->weight()});
            }
        }
        out.push_back(std::move(entry));
    }
    return out;
}

} // namespace fuzzybits::io::pcfb_frequency
