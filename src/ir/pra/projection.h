#pragma once

#include "ir/pc/id.h"
#include "ir/pc/query.h"
#include "ir/pc/target.h"
#include "ir/pra/id.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pra {

// How a monotone-hull query reads a complemented basic-event operand marked
// `literal` (target_binding::literal_events; docs/rfc/monotone-hull-queries.md
// §3). Unmarked complements and complemented gates are delete terms under
// every reading; probability queries have no reading.
enum class hull_literals : std::uint8_t {
    // Each marked operand is an atom of its own, sampled as not x_B and
    // droppable on its own: the plain hull over atoms (the hull check's
    // `split` mode; the reading of cut-set tools).
    split,
    // Rauzy's widening omega_L, with L the positive literals plus not B for
    // each marked event: those events are fixed (the hull check's `fixed`).
    fixed,
    // Marked operands are ordinary complements: the plain hull of the target.
    plain,
};

// The spelling of the CLI, the config document and the metadata.
[[nodiscard]] constexpr auto to_string(const hull_literals literals) noexcept -> std::string_view {
    switch (literals) {
    case hull_literals::split:
        return "split";
    case hull_literals::fixed:
        return "fixed";
    case hull_literals::plain:
        return "plain";
    }
    return "split";
}

[[nodiscard]] constexpr auto parse_hull_literals(const std::string_view text) noexcept -> std::optional<hull_literals> {
    if (text == "split") {
        return hull_literals::split;
    }
    if (text == "fixed") {
        return hull_literals::fixed;
    }
    if (text == "plain") {
        return hull_literals::plain;
    }
    return std::nullopt;
}

// The parameters of one monotone-hull query.
struct hull_query final {
    hull_literals literals = hull_literals::split;
    // The initiator rule: k deletes the cut sets with more than k initiating
    // events in the target's cone (RFC 4.8); unset: no rule.
    std::optional<std::uint32_t> max_initiators;

    friend constexpr auto operator==(hull_query const &, hull_query const &) noexcept -> bool = default;
};

enum class event_binding_kind : std::uint8_t {
    basic_event,
    house_event,
    gate,
    // A gate, basic event or house event replaced by a constant node by a
    // quantification-config override.
    constant_override,
};

// One PC node per bound element; `element` is a basic event, house event or gate.
struct event_binding final {
    domain_ref element;
    pc::bool_node_id node;
    event_binding_kind kind = event_binding_kind::basic_event;
};

struct override_binding final {
    domain_ref element;
    bool value = false;
    // Invalid when the element is outside every target's cone.
    pc::bool_node_id node;
};

enum class generated_kind : std::uint8_t {
    // atleast(k + 1; initiating events in the target's cone), for a hull
    // query with max_initiators = k.
    initiator_delete_term,
    // The target's subject AND NOT its initiator delete term.
    initiator_rule_conjunction,
};

// A PC node with no domain element, created for one target.
struct generated_binding final {
    pc::bool_node_id node;
    generated_kind kind = generated_kind::initiator_delete_term;
    pc::target_id target;
};

struct target_binding final {
    // Invalid for a target placed directly on a gate or basic event that no
    // domain target names.
    domain_target_id domain_target;
    // The observed gate or basic event (the top gate of a fault_tree_top target).
    domain_ref subject;
    pc::target_id pc_target;
    pc::observable_role role = pc::observable_role::primary;
    // The subject's node. The target node is `root`, or the initiator-rule
    // conjunction over it.
    pc::bool_node_id root;
    pc::bool_node_id node;
    // One binding, and one PC target, per query: a target asking for both
    // kinds has two bindings, which share a node unless the hull query holds
    // the initiator rule.
    pc::query_kind kind = pc::query_kind::probability;
    // Set exactly for a monotone-hull query.
    std::optional<hull_query> hull;
    // Valid when the hull query's rule is lowered into this target: the cone
    // holds more initiating events than hull->max_initiators allows. `node`
    // is then the conjunction of `root` and the complement of this term.
    pc::bool_node_id initiator_delete_term;
    // Slices of pc_projection::initiator_events() and
    // pc_projection::literal_events(), ascending basic-event ids.
    // `initiators`: basic events of initiating events in the cone.
    // `literal_events`: basic events that some gate in the cone references
    // through an operand marked `literal` (#199). The marker is the only
    // source: an unmarked complemented operand is never a literal,
    // whatever the source format, and an event negated only through a gate
    // is not in the set. Both sets come from the PRA model, so passes over
    // the circuit need not keep complemented edges where they were.
    // Elements a config override replaced are constants, not events, and
    // are in neither set; their operands are outside the cone.
    table_range<basic_event_id> initiators;
    table_range<basic_event_id> literal_events;
    // Under initiator_values::frequency: the initiating event this PC target is conditioned on
    // (TRUE; every other initiating event of the cone FALSE). The domain target's frequency is the
    // sum over its conditioned PC targets of the initiator's value times the target's value.
    // Invalid for an unconditioned target.
    basic_event_id condition;
};

// Sparse maps between a PRA model and the PC circuit lowered from it
// (docs/pra-pc-ir-architecture.md). Lookups in both directions are binary
// searches over sorted columns.
class pc_projection final {
  public:
    pc_projection() = default;

    pc_projection(std::vector<target_binding> targets, std::vector<event_binding> events,
                  std::vector<override_binding> overrides, std::vector<generated_binding> generated,
                  std::vector<basic_event_id> initiator_events, std::vector<basic_event_id> literal_events)
        : targets_(std::move(targets)), events_(std::move(events)), overrides_(std::move(overrides)),
          generated_(std::move(generated)), initiator_events_(std::move(initiator_events)),
          literal_events_(std::move(literal_events)) {
        std::ranges::stable_sort(targets_, {}, &target_binding::pc_target);
        std::ranges::stable_sort(events_, {}, &event_binding::node);
        std::ranges::stable_sort(generated_, {}, &generated_binding::node);
        targets_by_domain_ = order_by(targets_, &target_binding::domain_target);
        events_by_element_ = order_by(events_, &event_binding::element);
    }

    // Ordered by PC target id; binding i observes pc::target_id(i).
    [[nodiscard]] auto target_bindings() const noexcept -> std::span<target_binding const> { return targets_; }
    // Ordered by PC node id.
    [[nodiscard]] auto event_bindings() const noexcept -> std::span<event_binding const> { return events_; }
    // In config order.
    [[nodiscard]] auto override_bindings() const noexcept -> std::span<override_binding const> { return overrides_; }
    // Ordered by PC node id.
    [[nodiscard]] auto generated_bindings() const noexcept -> std::span<generated_binding const> { return generated_; }
    [[nodiscard]] auto initiator_events() const noexcept -> std::span<basic_event_id const> {
        return initiator_events_;
    }
    [[nodiscard]] auto literal_events() const noexcept -> std::span<basic_event_id const> { return literal_events_; }

    [[nodiscard]] auto initiators(target_binding const &target) const noexcept -> std::span<basic_event_id const> {
        return slice(initiator_events_, target.initiators);
    }
    [[nodiscard]] auto literal_events(target_binding const &target) const noexcept -> std::span<basic_event_id const> {
        return slice(literal_events_, target.literal_events);
    }

    [[nodiscard]] auto find_target(const pc::target_id target) const noexcept -> target_binding const * {
        const auto found = std::ranges::lower_bound(targets_, target, {}, &target_binding::pc_target);
        return found != targets_.end() && found->pc_target == target ? &*found : nullptr;
    }

    // The first binding, in PC target order, of a domain target.
    [[nodiscard]] auto find_target(const domain_target_id target) const noexcept -> target_binding const * {
        const auto found = std::ranges::lower_bound(
            targets_by_domain_, target, {}, [this](const std::uint32_t row) { return targets_[row].domain_target; });
        return found != targets_by_domain_.end() && targets_[*found].domain_target == target ? &targets_[*found]
                                                                                             : nullptr;
    }

    [[nodiscard]] auto find_event(const pc::bool_node_id node) const noexcept -> event_binding const * {
        const auto found = std::ranges::lower_bound(events_, node, {}, &event_binding::node);
        return found != events_.end() && found->node == node ? &*found : nullptr;
    }

    [[nodiscard]] auto find_event(const domain_ref element) const noexcept -> event_binding const * {
        const auto found = std::ranges::lower_bound(events_by_element_, element, {},
                                                    [this](const std::uint32_t row) { return events_[row].element; });
        return found != events_by_element_.end() && events_[*found].element == element ? &events_[*found] : nullptr;
    }

    // Invalid when the element has no node.
    [[nodiscard]] auto node_of(const domain_ref element) const noexcept -> pc::bool_node_id {
        auto const *binding = find_event(element);
        return binding == nullptr ? pc::bool_node_id::invalid() : binding->node;
    }

    [[nodiscard]] auto find_generated(const pc::bool_node_id node) const noexcept -> generated_binding const * {
        const auto found = std::ranges::lower_bound(generated_, node, {}, &generated_binding::node);
        return found != generated_.end() && found->node == node ? &*found : nullptr;
    }

  private:
    template <typename row_type, typename key_type>
    [[nodiscard]] static auto order_by(std::vector<row_type> const &rows, key_type row_type::*key)
        -> std::vector<std::uint32_t> {
        std::vector<std::uint32_t> order(rows.size());
        std::iota(order.begin(), order.end(), 0U);
        std::ranges::stable_sort(order, {}, [&rows, key](const std::uint32_t row) { return rows[row].*key; });
        return order;
    }

    [[nodiscard]] static auto slice(std::vector<basic_event_id> const &column,
                                    const table_range<basic_event_id> range) noexcept
        -> std::span<basic_event_id const> {
        if (range.offset > column.size() || range.count > column.size() - range.offset) {
            return {};
        }
        return std::span<basic_event_id const>{column}.subspan(range.offset, range.count);
    }

    std::vector<target_binding> targets_;
    std::vector<event_binding> events_;
    std::vector<override_binding> overrides_;
    std::vector<generated_binding> generated_;
    std::vector<basic_event_id> initiator_events_;
    std::vector<basic_event_id> literal_events_;
    std::vector<std::uint32_t> targets_by_domain_;
    std::vector<std::uint32_t> events_by_element_;
};

} // namespace fuzzybits::ir::pra
