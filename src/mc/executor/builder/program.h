/**
 * @file program.h
 * @brief What the builder makes: the kernels of a pass in submission order, the node blocks they read, where the
 *        tallied nodes' counts are, and the plan that describes it all
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/event/basic_event.h"
#include "mc/event/gate.h"
#include "mc/event/tally.h"
#include "mc/executor/descriptor/plan.h"
#include "mc/executor/queueable.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fuzzybits::mc::executor::builder {

/// The node blocks, by kind. Each holds its records and its run of arena slots (a tally block: its records only); the
/// arena belongs to the layer manager and outlives them.
template <event::node_types_like T> struct node_blocks {
    std::vector<event::sampled_event_block<T>> sampled_events;
    /// Basic events of constant law: their bitpacks were written when their slots were allocated, and no kernel reads
    /// these blocks.
    std::vector<event::constant_event_block<T>> constant_events;
    std::vector<event::gate_block<T>> gates;
    std::vector<event::cardinality_gate_block<T>> cardinality_gates;
    std::vector<event::tally_block<T>> tallies;

    [[nodiscard]] std::size_t size() const noexcept {
        return sampled_events.size() + constant_events.size() + gates.size() + cardinality_gates.size() + tallies.size();
    }

    /// Keeps `block` with the blocks of its kind, and returns its position among them.
    template <class Block> std::size_t keep(Block &&block) {
        auto &kind = of<std::remove_cvref_t<Block>>();
        kind.push_back(std::move(block));
        return kind.size() - 1;
    }

  private:
    template <class Block> std::vector<Block> &of() noexcept {
        if constexpr (std::same_as<Block, event::sampled_event_block<T>>) {
            return sampled_events;
        } else if constexpr (std::same_as<Block, event::constant_event_block<T>>) {
            return constant_events;
        } else if constexpr (std::same_as<Block, event::gate_block<T>>) {
            return gates;
        } else if constexpr (std::same_as<Block, event::cardinality_gate_block<T>>) {
            return cardinality_gates;
        } else {
            static_assert(std::same_as<Block, event::tally_block<T>>, "a block of a kind the builder keeps");
            return tallies;
        }
    }
};

/// Where a tallied node's count is: its block in node_blocks::tallies, and its position in the block.
struct tally_location {
    std::size_t block = 0;
    std::size_t position = 0;
};

/// A built program: the kernels a layer manager submits every pass, what they read, and what it reads back after.
template <event::node_types_like T> struct program {
    using index_type = typename T::index_type;

    /// The kernels of a pass, in submission order: every kernel after those whose outputs it reads, so that an in-order
    /// queue runs them without dependency events. The plan describes them in the same order; they are kept apart from
    /// it so that a pass walks these alone.
    std::vector<std::shared_ptr<queueable_base>> queueables;
    node_blocks<T> blocks;
    /// PDAG index → where the node's tally is.
    std::unordered_map<index_type, tally_location> tallies;
    descriptor::plan plan;

    /// The true trials counted at `location`, over the passes run so far.
    [[nodiscard]] std::uint64_t ones(const tally_location &location) const { return blocks.tallies.at(location.block).records()[location.position].ones; }

    /// The true trials counted for node `index`. Throws std::out_of_range for a node without a tally.
    [[nodiscard]] std::uint64_t ones(const index_type index) const {
        const auto location = tallies.find(index);
        if (location == tallies.end()) {
            throw std::out_of_range("layer_manager: PDAG node " + std::to_string(index) + " has no tally");
        }
        return ones(location->second);
    }
};

} // namespace fuzzybits::mc::executor::builder
