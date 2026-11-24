#pragma once

#include "ir/pc/id.h"
#include "ir/pc/target.h"
#include "ir/pra/id.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pra {

// How a target reads negation: as Boolean complement, or as cut-set
// deletion through the monotone hull (docs/rfc/monotone-hull-queries.md).
enum class negation_semantics : std::uint8_t {
    boolean,
    monotone_hull,
};

// The result kinds requested for one target; a target may carry both, which
// the writer emits as two queries on one tensor.
struct negation_set final {
    bool boolean = true;
    bool monotone_hull = false;

    [[nodiscard]] static constexpr auto of(const negation_semantics semantics) noexcept -> negation_set {
        return semantics == negation_semantics::boolean ? negation_set{.boolean = true, .monotone_hull = false}
                                                        : negation_set{.boolean = false, .monotone_hull = true};
    }
    [[nodiscard]] static constexpr auto both() noexcept -> negation_set {
        return negation_set{.boolean = true, .monotone_hull = true};
    }
    [[nodiscard]] constexpr auto contains(const negation_semantics semantics) const noexcept -> bool {
        return semantics == negation_semantics::boolean ? boolean : monotone_hull;
    }
    [[nodiscard]] constexpr auto empty() const noexcept -> bool { return !boolean && !monotone_hull; }

    friend constexpr auto operator==(negation_set const &, negation_set const &) noexcept -> bool = default;
};

// How a monotone-hull target reads a complemented basic-event operand
// marked `literal` (see target_binding::literal_events). Unmarked
// complements and negated gates stay delete terms in every mode; Boolean
// targets ignore the mode.
enum class literal_mode : std::uint8_t {
    // Each marked operand is its own atom, sampled as not x_B; the query is
    // the plain hull over atoms (the hull check's `split` treatment).
    atom_split,
    // The widening omega_L with L the positive literals plus not B for each
    // event in literal_events (those events fixed in the hull check).
    omega_l,
    // Marked operands are ordinary complements: the plain hull.
    negation,
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
    // atleast(k + 1; initiating events in the target's cone), for
    // max_initiators_per_cutset = k.
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
    negation_set negation;
    literal_mode literals = literal_mode::atom_split;
    // Invalid unless the config sets max_initiators_per_cutset and the cone
    // holds more initiating events than it allows.
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
