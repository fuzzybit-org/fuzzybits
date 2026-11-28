/**
 * @file node.h
 * @brief A node of the PDAG as the execution plan reports it
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace fuzzybits::mc::executor::descriptor {

/// A node of the PDAG: what it is, and where the plan computes it.
struct node {
    std::int32_t index = 0;
    std::size_t layer = 0;
    /// The partition that computes it. A node no partition computes (a constant: a basic event of constant law or a
    /// constant gate, written when its arena slot was allocated) reports 0.
    std::size_t partition = 0;
    /// `basic_event` or `gate`, or `unknown`.
    std::string kind;
    /// The op code it is computed by (builder::name_of): `sample` for a basic event; for a gate, `ones` or `zeros` if
    /// it is a constant, else its own (`and`, `xor`, `atleast`, ...); or `unknown`.
    std::string connective;
    std::size_t fan_in = 0;
    std::size_t positive_input_count = 0;
    std::size_t negated_input_count = 0;
    /// An at-least gate's least number of true inputs; 0 for other nodes.
    std::size_t atleast_min = 0;
    /// Whether the node is a module gate of the PDAG.
    bool module = false;
};

} // namespace fuzzybits::mc::executor::descriptor
