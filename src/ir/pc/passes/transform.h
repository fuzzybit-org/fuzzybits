#pragma once

#include "api/diagnostics.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace fuzzybits::ir::pc::passes {

    // Query semantics of one target. ir::pc targets do not carry it (IR-PC §2);
    // the ir::pra -> ir::pc lowering (#181) records it and callers pass it in.
    enum class target_semantics : std::uint8_t {
        boolean,
        monotone_hull,
    };

    // docs/rfc/monotone-hull-queries.md §7.5.
    enum class semantic_class : std::uint8_t {
        // Every target computes the same Boolean function, so every query kind
        // defined from the function is preserved.
        function_preserving,
        // Target probabilities under independent sampling are unchanged for the
        // semantics listed in transform_report::preserved_semantics only.
        distribution_preserving,
    };

    enum class image_kind : std::uint8_t {
        // Not in the cone of any target; the output does not contain it.
        removed,
        // Inlined into every parent by coalescing; no output node computes it.
        absorbed,
        // Folded to node_image::constant_value; there may be no output node.
        constant,
        // Computed by output node edge.source, complemented if edge.complement.
        reference,
    };

    struct node_image final {
        image_kind kind = image_kind::removed;
        bool constant_value = false;
        edge_ref edge;
    };

    // IR-PC §12. Node and target maps are indexed by input ids.
    struct transform_report final {
        std::string transform_id;
        std::uint32_t version = 1;
        semantic_class semantics = semantic_class::function_preserving;
        std::vector<target_semantics> preserved_semantics;
        std::vector<node_image> node_map;
        std::vector<target_id> target_map;
        std::uint32_t nodes_before = 0;
        std::uint32_t edges_before = 0;
        std::uint32_t nodes_after = 0;
        std::uint32_t edges_after = 0;
        // Pass-specific count: events folded, gates simplified or eliminated,
        // children absorbed.
        std::uint64_t rewrites = 0;
        // Rewrites not applied because they would change an atom-split hull
        // (#190); also reported as note pc.pass.atom-split-skipped.
        std::uint64_t skipped = 0;
        // Transform-local scratch (excluding input and output circuits).
        std::size_t scratch_bytes = 0;
        api::diagnostic_list diagnostics;

        [[nodiscard]] auto preserves(const target_semantics kind) const -> bool {
            return std::ranges::find(preserved_semantics, kind) != preserved_semantics.end();
        }
    };

    struct transform_result final {
        circuit output;
        transform_report report;

        [[nodiscard]] auto ok() const -> bool {
            return output.valid() && !report.diagnostics.has_errors();
        }
    };

    // Image under `second` of an image under the preceding transform.
    [[nodiscard]] inline auto compose(node_image const &image, std::span<node_image const> second) -> node_image {
        if (image.kind != image_kind::reference) {
            return image;
        }
        auto next = second[image.edge.source.raw()];
        if (next.kind == image_kind::constant) {
            next.constant_value = next.constant_value != image.edge.complement;
        } else if (next.kind == image_kind::reference) {
            next.edge.complement = next.edge.complement != image.edge.complement;
        }
        return next;
    }

    [[nodiscard]] inline auto compose(std::span<node_image const> first, std::span<node_image const> second) -> std::vector<node_image> {
        std::vector<node_image> composed(first.size());
        for (std::size_t node = 0; node < first.size(); ++node) {
            composed[node] = compose(first[node], second);
        }
        return composed;
    }

    [[nodiscard]] inline auto compose(std::span<target_id const> first, std::span<target_id const> second) -> std::vector<target_id> {
        std::vector<target_id> composed(first.size());
        for (std::size_t target = 0; target < first.size(); ++target) {
            composed[target] = first[target].valid() ? second[first[target].raw()] : target_id::invalid();
        }
        return composed;
    }

} // namespace fuzzybits::ir::pc::passes
