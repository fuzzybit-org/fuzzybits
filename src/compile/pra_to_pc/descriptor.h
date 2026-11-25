#pragma once

#include "api/ids.h"
#include "api/result.h"
#include "io/pcfb/descriptor.h"
#include "ir/handle.h"
#include "ir/pc/circuit.h"
#include "ir/pc/query.h"
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
    // The queries of this target, one per kind; empty: quantification_config::default_kinds.
    std::vector<ir::pc::query_kind> kinds;
    // The parameters of its hull query; unset: quantification_config::default_hull. Only with a
    // monotone_hull kind (compile.pra_to_pc.hull-parameters-without-hull otherwise).
    std::optional<ir::pra::hull_query> hull;
};

// `element := TRUE | FALSE`. The element becomes a constant under every
// query kind: it is never droppable, and its operands leave the cone.
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

// How the value of an initiating event is quantified.
enum class initiator_values : std::uint8_t {
    // Its value is a probability. A frequency literal up to 1 is used as one (note
    // compile.pra_to_pc.frequency-as-probability); above 1 it is an error.
    probability,
    // Its value is a frequency (#243). A target whose cone holds initiating events I_1..I_m becomes
    // m conditioned targets, I_i := TRUE and every other I_j := FALSE, and the target's value is
    // F = sum_i f_i * value(target | I_i), in initiator_unit. A target that may hold with every
    // initiating event FALSE mixes a probability with a frequency and is an error
    // (compile.pra_to_pc.mixed-units). No hull query may set max_initiators: each conditioned
    // target has exactly one initiating event TRUE.
    frequency,
};

// R1 quantification config. Probability policy is fixed: every basic event
// in the cone needs a literal; frequencies follow `initiators`.
struct quantification_config final {
    // Empty: every primary domain target of the model, in id order, with
    // `default_kinds` and `default_hull`.
    std::vector<target_request> targets;
    // The kinds of a target whose request names none: the probability query.
    std::vector<ir::pc::query_kind> default_kinds{ir::pc::query_kind::probability};
    // The hull parameters of a hull target whose request sets none.
    //  - literals: how the query reads operands marked `literal` in the PRA model; irrelevant to
    //    models without marked operands. A hull target warns
    //    (compile.pra_to_pc.complement-under-negation) for each literal event that also occurs
    //    below a non-monotone step, where the readings can differ.
    //  - max_initiators: k ANDs the hull query with not atleast(k + 1; initiating events in its
    //    cone), which deletes exactly the cut sets with more than k initiating events
    //    (docs/rfc/monotone-hull-queries.md 4.8). Probability queries never get the rule: it would
    //    remove assignments and change the target. A target with both kinds then has the
    //    probability query on its subject and the hull query on the conjunction.
    ir::pra::hull_query default_hull;
    // Added after `targets`, with pc::observable_role::watch.
    std::vector<target_request> watch_targets;
    std::vector<override_request> overrides;
    initiator_values initiators = initiator_values::frequency;
    // The unit of every initiating-event frequency, e.g. "per-year". Required under
    // initiator_values::frequency when a target's cone holds an initiating event
    // (compile.pra_to_pc.missing-initiator-unit; the FTP driver always supplies one);
    // recorded, never interpreted.
    std::optional<std::string> initiator_unit;
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
    // with overrides applied, ANDed with the initiator delete term for a
    // monotone-hull query under the initiator rule.
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
    // Nodes and edges of the initiator rules of hull queries; included in pc_nodes and pc_edges.
    std::size_t generated_nodes = 0;
    std::size_t generated_edges = 0;
    // quantification_config::initiators and ::initiator_unit as applied.
    initiator_values initiators = initiator_values::frequency;
    std::optional<std::string> initiator_unit;
    // PC targets conditioned on one initiating event, and the nodes lowered once per condition
    // (gates whose cone holds an initiating event); both included in pc_targets and pc_nodes.
    std::size_t conditioned_targets = 0;
    std::size_t conditioned_nodes = 0;
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

// `.pcfb` writer options: one query per target binding, named by its domain target (or element),
// of the binding's kind (two queries share a tensor where two bindings share a node), with its
// metadata: the target name, the hull reading of a hull query, the initiator rule where its delete
// term is in the target, and the initiating event of a conditioned query. Fails with
// compile.pra_to_pc.hull-literals-unsupported for a split or fixed hull query whose cone has
// marked literal events: the writer does not yet fill pcfb's per-query literal fields
// (docs/rfc/monotone-hull-queries.md 7.3, #235). With plain, marked operands are ordinary
// complements, which is the plain hull a hull query means without literal fields, so such a
// query is written.
[[nodiscard]] auto pcfb_write_options(pra_pc_lowering const &lowering)
    -> api::result<io::backend::fuzzybits::pcfb::write_options>;

} // namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc
