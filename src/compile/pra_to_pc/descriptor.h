#pragma once

#include "api/ids.h"
#include "api/result.h"
#include "io/pcfb/descriptor.h"
#include "ir/handle.h"
#include "ir/pc/circuit.h"
#include "ir/pra/id.h"
#include "ir/pra/model.h"
#include "ir/pra/projection.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc {

    // A target to quantify, named by stable name.
    struct target_request final {
        std::string name;
        // Unset: `name` is a domain target. `gate` or `basic_event`: a target
        // placed directly on the element of that kind and name.
        std::optional<ir::pra::element_kind> element;
        ir::pra::negation_set negation;
        // Unset: quantification_config::default_literals.
        std::optional<ir::pra::literal_mode> literals;
    };

    // `element := TRUE | FALSE`. The element becomes a constant under every
    // negation semantics: it is never droppable, and its operands leave the cone.
    struct override_request final {
        std::string name;
        // Unset: the name must match exactly one gate, basic event or house event.
        std::optional<ir::pra::element_kind> kind;
        bool value = false;
    };

    enum class packing_mode : std::uint8_t {
        // One circuit holding every target.
        single_circuit,
    };

    // R1 quantification config. Probability policy is fixed: every basic event
    // in the cone needs a literal; a frequency is used as a probability only if
    // it is at most 1, with a warning.
    struct quantification_config final {
        // Empty: every primary domain target of the model, in id order, with
        // `default_negation`.
        std::vector<target_request> targets;
        ir::pra::negation_set default_negation;
        // How monotone-hull targets read operands marked `literal` in the PRA
        // model; recorded on every target, ignored by Boolean targets and by
        // models without marked operands. A hull target warns
        // (compile.pra_to_pc.complement-under-negation) for each literal event
        // that also occurs below a non-monotone step, where the modes can differ.
        ir::pra::literal_mode default_literals = ir::pra::literal_mode::atom_split;
        // Added after `targets`, with pc::observable_role::watch.
        std::vector<target_request> watch_targets;
        std::vector<override_request> overrides;
        // Unset: no rule. k: every target is ANDed with
        // not atleast(k + 1; initiating events in its cone), which deletes the
        // assignments (and, under the monotone hull, exactly the cut sets) with
        // more than k initiating events.
        std::optional<std::uint32_t> max_initiators_per_cutset;
        packing_mode packing = packing_mode::single_circuit;
    };

    enum class semantic_class : std::uint8_t {
        function_preserving,
        distribution_preserving,
    };

    struct transform_report final {
        std::string transform = "compile.pra_to_pc";
        std::uint32_t version = 1;
        // Every PC target computes, on every assignment of the basic events, the
        // Boolean function its domain target has under the config: the subject
        // with overrides applied, ANDed with the initiator delete term if any.
        semantic_class semantics = semantic_class::function_preserving;

        // The union of the targets' cones in the PRA model.
        std::size_t source_gates = 0;
        std::size_t source_operands = 0;
        std::size_t source_basic_events = 0;
        std::size_t source_house_events = 0;
        std::size_t overridden_elements = 0;

        std::size_t pc_nodes = 0;
        std::size_t pc_edges = 0;
        std::size_t pc_targets = 0;
        // quantification_config::max_initiators_per_cutset as applied: unset means no rule. Recorded
        // even where a target's cone holds too few initiating events for the rule to add a node.
        std::optional<std::uint32_t> max_initiators_per_cutset;
        // Nodes and edges of the initiator rule; included in pc_nodes and pc_edges.
        std::size_t generated_nodes = 0;
        std::size_t generated_edges = 0;
    };

    struct pra_pc_lowering final {
        ir::handle pra_model;
        ir::pc::circuit circuit;
        ir::pra::pc_projection projection;
        transform_report report;
    };

    using lowering_result = api::result<pra_pc_lowering>;

    [[nodiscard]] auto endpoint_id() -> api::endpoint_id;

    // Lowers the targets' cones of a sealed model, gate for gate. Every
    // diagnostic is reported; the artifact is returned only without errors.
    // Linear in the union cone, plus one walk of each target's cone when the
    // model has initiating events or operands marked `literal`.
    [[nodiscard]] auto lower(ir::pra::model const &model, quantification_config const &config = {}) -> lowering_result;
    [[nodiscard]] auto lower(ir::handle const &model, quantification_config const &config = {}) -> lowering_result;

    // `.pcfb` writer options naming every query by its domain target (or element), choosing its kind
    // from the target's negation semantics (boolean: probability, monotone_hull: monotone-hull
    // probability; a target with both gets both queries on one tensor) and recording the negation
    // semantics, literal mode and initiator rule of each target. Fails with
    // compile.pra_to_pc.hull-literals-unsupported for an atom_split or omega_l monotone-hull target whose
    // cone has marked literal events: pcfb 1.2 has no per-query literal atoms yet
    // (docs/rfc/monotone-hull-queries.md 7.3). In negation mode marked operands are ordinary
    // complements, which is the plain hull a hull query means, so such a target is written.
    [[nodiscard]] auto pcfb_write_options(pra_pc_lowering const &lowering) -> api::result<io::backend::fuzzybits::pcfb::write_options>;

} // namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc
