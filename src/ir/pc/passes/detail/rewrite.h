#pragma once

#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/passes/passes.h"
#include "ir/pc/passes/transform.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace fuzzybits::ir::pc::passes::detail {

// Union of the target cones in post-order (operands before users), from an
// explicit stack: depth is bounded by memory, not by the call stack.
[[nodiscard]] auto cone_order(circuit const &input) -> std::vector<std::uint32_t>;

[[nodiscard]] inline auto reference(const bool_node_id node, const bool complement = false) -> node_image {
    return node_image{.kind = image_kind::reference, .edge = edge_ref{.source = node, .complement = complement}};
}

[[nodiscard]] inline auto constant(const bool value) -> node_image {
    return node_image{.kind = image_kind::constant, .constant_value = value};
}

// Image of an input edge whose source has image `image`.
[[nodiscard]] inline auto through(node_image image, const bool complement) -> node_image {
    if (image.kind == image_kind::constant) {
        image.constant_value = image.constant_value != complement;
    } else if (image.kind == image_kind::reference) {
        image.edge.complement = image.edge.complement != complement;
    }
    return image;
}

// Operands of input node `node` through `map`, whose images must all be
// references.
inline void remap_operands(circuit const &input, const std::uint32_t node, std::span<node_image const> map,
                           std::vector<edge_ref> &out) {
    out.clear();
    for (auto const &[source, complement] : input.operands(bool_node_id(node))) {
        auto const &image = map[source.raw()];
        out.push_back(edge_ref{.source = image.edge.source, .complement = image.edge.complement != complement});
    }
}

template <typename value_type>
[[nodiscard]] auto bytes_of(std::vector<value_type> const &values) noexcept -> std::size_t {
    return values.capacity() * sizeof(value_type);
}

// Transform-local construction state: copies leaves and probability sources
// on first use, creates at most one node per constant value, and seals with
// the input targets resolved through the report's node map.
class emitter final {
  public:
    explicit emitter(circuit const &input);

    [[nodiscard]] auto leaf(std::uint32_t node) -> bool_node_id;
    [[nodiscard]] auto constant(bool value) -> bool_node_id;
    [[nodiscard]] auto gate(op_code code, std::span<edge_ref const> operands, std::uint32_t low = 0,
                            std::uint32_t high = 0) -> bool_node_id;

    [[nodiscard]] auto node_count() const noexcept -> std::uint32_t { return node_count_; }

    [[nodiscard]] auto scratch_bytes() const noexcept -> std::size_t {
        return bytes_of(source_map_) + bytes_of(atoms_);
    }

    // Output node `node` is an atom event: a target whose image is its
    // complemented edge gets and(not node) (the atom), not not(node) (a
    // delete term). Both are the same Boolean function.
    void mark_atom(bool_node_id node);
    [[nodiscard]] auto is_atom(const bool_node_id node) const noexcept -> bool {
        return node.raw() < atoms_.size() && atoms_[node.raw()] != 0;
    }

    // Every target's image must be a constant or a reference.
    [[nodiscard]] auto finish(transform_report report) && -> transform_result;

  private:
    [[nodiscard]] auto source(probability_source_id input_source) -> probability_source_id;
    [[nodiscard]] auto counted(bool_node_id node) noexcept -> bool_node_id;

    circuit const &input_;
    builder builder_;
    std::vector<probability_source_id> source_map_;
    std::vector<std::uint8_t> atoms_;
    std::optional<bool_node_id> constants_[2];
    std::uint32_t node_count_ = 0;
};

// Atom-split context of a pass (#190): which input gates lie in the cone of
// an atom-split hull target, and which basic events are atom events there.
// A complemented edge from such a gate into such an event is an atom, not a
// negation.
class split_context final {
  public:
    [[nodiscard]] auto active() const noexcept -> bool { return !in_cone_.empty(); }
    [[nodiscard]] auto in_cone(const std::uint32_t node) const noexcept -> bool {
        return active() && in_cone_[node] != 0;
    }
    [[nodiscard]] auto member(const std::uint32_t node) const noexcept -> bool {
        return active() && member_[node] != 0;
    }
    [[nodiscard]] auto atom_edge(const std::uint32_t gate, edge_ref const &edge) const noexcept -> bool {
        return edge.complement && in_cone(gate) && member(edge.source.raw());
    }
    // Some gate in an atom-split cone reads this event's atom.
    [[nodiscard]] auto has_atom_edge(const std::uint32_t node) const noexcept -> bool {
        return active() && has_atom_edge_[node] != 0;
    }
    [[nodiscard]] auto bytes() const noexcept -> std::size_t {
        return bytes_of(in_cone_) + bytes_of(member_) + bytes_of(has_atom_edge_);
    }

    // Errors pc.pass.invalid-atom-split on out-of-range targets or events
    // that are not basic events.
    [[nodiscard]] static auto make(circuit const &input, atom_split const &atoms, transform_report &report)
        -> std::optional<split_context>;

  private:
    std::vector<std::uint8_t> in_cone_;
    std::vector<std::uint8_t> member_;
    std::vector<std::uint8_t> has_atom_edge_;
};
[[nodiscard]] auto begin_report(circuit const &input, pass_id pass) -> transform_report;

// Invalid input fails; a circuit without targets is returned unchanged with
// a warning. Otherwise nullopt: the pass proceeds.
[[nodiscard]] auto trivial_result(circuit const &input, transform_report &report) -> std::optional<transform_result>;
} // namespace fuzzybits::ir::pc::passes::detail
