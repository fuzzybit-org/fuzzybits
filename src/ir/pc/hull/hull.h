#pragma once

#include "api/result.h"
#include "ir/pc/circuit.h"
#include "ir/pc/id.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

// Exact per-sample monotone-hull check (docs/rfc/monotone-hull-queries.md §6).
//
// For a target g and a sample x, the check decides whether some y in the
// sample's neighbourhood Y(x) satisfies g, where Y(x) constrains every basic
// event v by its mode:
//
//   down   y_v <= x_v   only the positive literal is significant (plain hull ↑g)
//   fixed  y_v == x_v   both literals are significant
//   up     y_v >= x_v   only the negative literal is significant
//
// With every event `down` this is ↑g(x); a mixed mode vector gives Rauzy's
// widening ω_L(g)(x) for the corresponding set L of significant literals.
// Events whose probability source is the literal 0 are FALSE in every y
// regardless of mode; probability-1 events are ordinary events. Constants and
// house events keep their value.
namespace fuzzybits::ir::pc::hull {

    enum class event_mode : std::uint8_t {
        down,
        fixed,
        up,
    };

    // Budgets are counted in deterministic units, so they change cost, never
    // answers: every lane is decided exactly by the last tier (complete search).
    struct check_options final {
        // Fix events on which the (residual) target is unate to their best value.
        bool pure_literals = true;
        // Try a few concrete assignments of the residual before exact solving.
        bool witnesses = true;
        // Enumerate residuals with at most this many free events...
        std::uint32_t truth_table_max_variables = 20;
        // ...if (residual gates + edges) x 2^max(0, free - 6) stays within this.
        std::uint64_t truth_table_budget = std::uint64_t{1} << 22;
        // Build a BDD for residuals with at most this many free events, up to
        // this many BDD nodes; 0 disables the tier.
        std::uint32_t bdd_max_variables = 512;
        std::uint32_t bdd_node_budget = std::uint32_t{1} << 14;
        // Run blocks of 256 lanes on OpenMP threads (when built with OpenMP).
        bool parallel = true;
    };

    enum class lane_layout : std::uint8_t {
        // One row per lane; bit e of a row is event e.
        lane_major,
        // One row per event; bit l of a row is lane l.
        event_major,
    };

    // Read-only bit matrix of samples. Bit (r, c) is bit c % 64 of
    // words[r * stride + c / 64]. Events are numbered by `checker::event_node`.
    struct lane_matrix final {
        std::span<std::uint64_t const> words;
        std::size_t lane_count = 0;
        std::size_t stride = 0;
        lane_layout layout = lane_layout::lane_major;
    };

    // How each lane was decided. Counts are integers and independent of the
    // thread count.
    struct check_stats final {
        std::uint64_t lanes = 0;
        std::uint64_t hits = 0;
        // Decided by three-valued propagation with unate events fixed.
        std::uint64_t bounds = 0;
        // Decided after repeated unate fixing on the residual.
        std::uint64_t reduction = 0;
        std::uint64_t witness = 0;
        std::uint64_t truth_table = 0;
        std::uint64_t bdd = 0;
        std::uint64_t bdd_budget_exceeded = 0;
        std::uint64_t search = 0;
        // Free events of residuals that reached an exact tier.
        std::uint64_t residual_variables_total = 0;
        std::uint32_t residual_variables_max = 0;

        void merge(check_stats const &other) noexcept;
    };

    namespace detail {
        struct cone;
    } // namespace detail

    // Per-target preparation (cone, topological order, fan-out lists, event
    // polarities, base values) shared by all lanes. Immutable after `make`;
    // `check` may be called concurrently from any number of threads.
    class checker final {
    public:
        [[nodiscard]] static auto make(circuit const &model, bool_node_id target, std::span<event_mode const> modes = {},
                                       check_options const &options = {}) -> api::result<checker>;

        // Basic events of the circuit in node-id order; lane bit e is event_node(e).
        [[nodiscard]] auto event_count() const noexcept -> std::uint32_t;
        [[nodiscard]] auto event_node(std::uint32_t event) const noexcept -> bool_node_id;
        // Events in the target's cone, ascending; other lane bits are ignored.
        [[nodiscard]] auto cone_events() const noexcept -> std::span<std::uint32_t const>;
        [[nodiscard]] auto cone_size() const noexcept -> std::uint32_t;
        [[nodiscard]] auto options() const noexcept -> check_options const &;

        // One lane: bit e of `sample` is x_e.
        [[nodiscard]] auto check(std::span<std::uint64_t const> sample) const -> bool;

        // Bit l of `out` (at least ceil(lane_count / 64) words) receives the
        // answer for lane l; bits past lane_count are cleared.
        void check(lane_matrix const &lanes, std::span<std::uint64_t> out, check_stats *stats = nullptr) const;

    private:
        explicit checker(std::shared_ptr<detail::cone const> cone) : cone_(std::move(cone)) {}

        std::shared_ptr<detail::cone const> cone_;
    };

} // namespace fuzzybits::ir::pc::hull
