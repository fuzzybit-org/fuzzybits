/**
 * @file plan.h
 * @brief The execution plan: the partitions of a pass in submission order, and the edges between them
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/executor/descriptor/partition.h"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fuzzybits::mc::executor::descriptor {

/// The partitions of a pass in submission order. A partition reads only partitions before it, which an in-order queue
/// then runs first; add() refuses any other. The edges follow from the partitions' dependencies, so they are not kept.
class plan {
  public:
    /// Adds `partition` as the next one, and returns its id. Throws std::logic_error if it reads a partition that is
    /// not before it.
    std::size_t add(partition next) {
        next.id = partitions_.size();
        if (!next.dependencies.empty() && next.dependencies.back() >= next.id) {
            throw std::logic_error("execution plan: partition " + std::to_string(next.id) + " reads partition " + std::to_string(next.dependencies.back()) +
                                   ", which is not submitted before it");
        }
        partitions_.push_back(std::move(next));
        return partitions_.back().id;
    }

    [[nodiscard]] std::span<const partition> partitions() const noexcept { return partitions_; }
    [[nodiscard]] std::size_t size() const noexcept { return partitions_.size(); }

    /// One edge per dependency, by target and then source.
    [[nodiscard]] std::vector<edge> edges() const {
        std::vector<edge> out;
        for (const partition &target : partitions_) {
            for (const std::size_t source : target.dependencies) {
                out.push_back({.source = source, .target = target.id});
            }
        }
        return out;
    }

  private:
    std::vector<partition> partitions_;
};

} // namespace fuzzybits::mc::executor::descriptor
