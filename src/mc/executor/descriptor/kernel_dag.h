/**
 * @file kernel_dag.h
 * @brief The PDAG's nodes grouped per layer by the kind of kernel that computes them: the execution plan's coarse view
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::mc::executor::descriptor {

/// Per layer, one group of basic events and one group of gates per op code (builder::name_of), in op code order. The
/// plan may split a group into several partitions (basic events by law, gates by kernel form) and computes constants
/// in none, so the groups are the plan's coarse view, and each node is in exactly one.
struct kernel_dag {
    struct node {
        std::int32_t index = 0;
        std::size_t layer = 0;
        /// The id of its group.
        std::size_t partition = 0;
        std::string kind;
        std::string connective;
    };

    struct group {
        std::size_t id = 0;
        std::size_t layer = 0;
        std::string kind;
        std::string connective;
        std::vector<std::int32_t> node_indices;
    };

    std::vector<node> nodes;
    std::vector<group> partitions;
    std::size_t batch_size = 0;
    std::size_t bitpacks_per_batch = 0;
    std::size_t total_iterations = 0;

    /// Starts the next group, `kind` nodes of `connective` in layer `layer`; add() then puts nodes in it.
    void open(const std::size_t layer, std::string kind, std::string connective) {
        partitions.push_back({.id = partitions.size(), .layer = layer, .kind = std::move(kind), .connective = std::move(connective), .node_indices = {}});
    }

    /// Puts node `index` in the group opened last.
    void add(const std::int32_t index) {
        auto &[id, layer, kind, connective, node_indices] = partitions.back();
        node_indices.push_back(index);
        nodes.push_back({.index = index, .layer = layer, .partition = id, .kind = kind, .connective = connective});
    }
};

} // namespace fuzzybits::mc::executor::descriptor
