/**
 * @file compat.h
 * @brief The executor's only door to SCRAM's code: the PDAG it runs, the MEF events behind its basic events, and SCRAM's
 *        log
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details Everything in mc/executor that reads SCRAM's PDAG, MEF or logger does it here, and builder/compat.h says the
 * PDAG's gates in the builder's op codes (the native IR's, ir::pc::op_code); no other file of the executor includes
 * SCRAM. The layer manager holds the PDAG as a compat::graph: its nodes in topological layers, which the graph reads
 * into the builder's layers (builder/layer.h) and describes (descriptor/). Retiring SCRAM (#144) deletes these two
 * files.
 */

#pragma once

#include "logger.h"
#include "pdag.h"
#include "preprocessor.h"

#include "mc/event/basic_event.h"
#include "mc/executor/builder/compat.h"
#include "mc/executor/builder/layer.h"
#include "mc/executor/builder/op.h"
#include "mc/executor/descriptor/kernel_dag.h"
#include "mc/executor/descriptor/node.h"
#include "mc/executor/descriptor/partition.h"
#include "mc/executor/descriptor/plan.h"
#include "mc/stats/tally_node_map.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fuzzybits::mc::executor::compat {

/// SCRAM's namespaces, by the names its code gives them: the PDAG (`core`) and the MEF model (`mef`).
namespace core = scram::core;
namespace mef = scram::mef;

/// The PDAG the executor runs (SCRAM's core::Pdag).
using pdag = core::Pdag;

/// The MEF event a basic event of the PDAG comes from (SCRAM's mef::Event).
using mef_event = mef::Event;

/// Calls `write(log)` with SCRAM's log stream at DEBUG2 when the log reports that level, and not otherwise: what
/// `write` writes is not computed when no one reads it.
template <typename F> void debug2(F &&write) {
    if (scram::DEBUG2 <= scram::Logger::report_level()) {
        write(scram::Logger().Get(scram::DEBUG2));
    }
}

/// The PDAG indices of the nodes `to_tally` tallies, in its order. Throws std::invalid_argument for a tally of no node.
[[nodiscard]] inline std::vector<std::int32_t> tallied(const stats::TallyNodeMap &to_tally) {
    std::vector<std::int32_t> indices;
    indices.reserve(to_tally.size());
    for (const auto &[key, tally] : to_tally) {
        if (!tally.node) {
            throw std::invalid_argument("kernel builder: a tally names no PDAG node");
        }
        indices.push_back(tally.node->index());
    }
    return indices;
}

/**
 * @brief A PDAG as the executor reads it: the nodes reachable from its root, in topological layers
 *
 * @details A layer's nodes do not read each other: every node reads only nodes of earlier layers. Within a layer, the
 * Variables come first and then the gates, each by PDAG index, so that the order of their records and arena slots is a
 * function of the PDAG alone (a basic event's random bits do not depend on it: prng/stream.h). The builder's layer
 * (builder/layer.h) holds the gates by op code (builder/compat.h).
 */
class graph {
  public:
    using node_ptr = std::shared_ptr<core::Node>;

    /// The nodes of `pdag`, which this orders topologically by layer (core::pdag::LayeredTopologicalOrder) and whose visit
    /// marks it clears. Throws std::runtime_error for two nodes of one index.
    explicit graph(pdag *pdag) {
        core::pdag::LayeredTopologicalOrder(pdag);
        pdag->Clear<core::Pdag::kVisit>();
        gather(pdag->root_ptr());
        std::ranges::sort(nodes_, {}, [](const node_ptr &node) { return node->order(); });
        layers_.resize(nodes_.empty() ? 0 : static_cast<std::size_t>(nodes_.back()->order()) + 1);
        for (const node_ptr &node : nodes_) {
            layers_[static_cast<std::size_t>(node->order())].push_back(node);
            indices_ = std::max(indices_, static_cast<std::size_t>(node->index()) + 1);
        }
        for (auto &layer : layers_) {
            std::ranges::sort(layer, [](const node_ptr &lhs, const node_ptr &rhs) { return position(*lhs) < position(*rhs); });
        }
        LOG(scram::DEBUG5) << "num_nodes: " << nodes_.size();
        LOG(scram::DEBUG5) << "num_layers: " << layers_.size();
    }

    [[nodiscard]] std::size_t nodes() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t layers() const noexcept { return layers_.size(); }
    [[nodiscard]] std::size_t variables() const noexcept { return variables_; }
    [[nodiscard]] std::size_t gates() const noexcept { return nodes_.size() - variables_; }
    /// One more than the greatest PDAG index of a node: a PDAG gives its nodes dense indices.
    [[nodiscard]] std::size_t indices() const noexcept { return indices_; }

    /// Layer `index` as the builder takes it: each Variable's exact law, each constant gate's value, and each other gate's
    /// inputs, by op code (op_code_of). Throws std::runtime_error for a Variable without a point probability in [0, 1].
    template <event::node_types_like T> [[nodiscard]] builder::layer<T> layer(const std::size_t index) const {
        builder::layer<T> layer;
        std::vector<builder::input> inputs;
        for_each_by_op(
            layers_.at(index),
            [&layer](core::Variable &variable) {
                const int pdag_index = variable.index();
                const mef::BasicEvent *definition = variable.graph().basic_events().at(pdag_index - core::Pdag::kVariableStartIndex);
                if (definition == nullptr || !definition->HasExpression()) {
                    throw std::runtime_error("basic event kernel: PDAG index " + std::to_string(pdag_index) + " has no point probability expression");
                }
                const auto law = event::bernoulli_law::from_probability(definition->expression().value());
                if (!law) {
                    throw std::runtime_error("basic event kernel: PDAG index " + std::to_string(pdag_index) + " point probability is not in [0, 1]");
                }
                layer.basic_events.push_back({.id_in_pdag = static_cast<typename T::index_type>(pdag_index), .law = *law});
            },
            [&layer, &inputs](core::Gate &gate, const builder::op_code op) {
                if (op == builder::op_code::ones || op == builder::op_code::zeros) {
                    layer.add_constant(gate.index(), op == builder::op_code::ones);
                    return;
                }
                inputs.clear();
                for_each_input(gate,
                               [&inputs](const int argument, const core::Node &input) { inputs.push_back({.node = input.index(), .negated = argument < 0}); });
                if (builder::is_counting(op)) {
                    layer.add_gate(op, gate.index(), inputs, bounds_of(gate, inputs.size()));
                } else {
                    layer.add_gate(op, gate.index(), inputs);
                }
            });
        return layer;
    }

    /// The nodes grouped per layer: the basic events (`sample`), then the gates by op code, in op code order (constant
    /// gates as `ones` and `zeros`). The pass's shape is the caller's to fill in.
    [[nodiscard]] descriptor::kernel_dag kernel_dag() const {
        descriptor::kernel_dag dag;
        for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
            std::vector<std::int32_t> variables;
            std::map<builder::op_code, std::vector<std::int32_t>> gates;
            for_each_by_op(
                layers_[layer], [&variables](const core::Variable &variable) { variables.push_back(variable.index()); },
                [&gates](const core::Gate &gate, const builder::op_code op) { gates[op].push_back(gate.index()); });
            const auto group = [&dag, layer](const std::vector<std::int32_t> &indices, const std::string_view kind, const builder::op_code op) {
                if (indices.empty()) {
                    return;
                }
                dag.open(layer, std::string(kind), std::string(builder::name_of(op)));
                for (const std::int32_t index : indices) {
                    dag.add(index);
                }
            };
            group(variables, "basic_event", builder::op_code::sample);
            for (const auto &[op, indices] : gates) {
                group(indices, "gate", op);
            }
        }
        return dag;
    }

    /// Every node, in topological order, with the partition of `plan` that computes it; a node that none computes (a
    /// constant: a basic event of constant law, or a constant gate) reports 0.
    [[nodiscard]] std::vector<descriptor::node> describe(const descriptor::plan &plan) const {
        std::unordered_map<std::int32_t, std::size_t> computed_by;
        for (const descriptor::partition &partition : plan.partitions()) {
            if (partition.kind != descriptor::partition_kind::tally) {
                for (const std::int32_t index : partition.nodes) {
                    computed_by.emplace(index, partition.id);
                }
            }
        }
        std::vector<descriptor::node> out;
        out.reserve(nodes_.size());
        for (const node_ptr &node : nodes_) {
            const auto partition = computed_by.find(node->index());
            out.push_back(describe(*node, partition == computed_by.end() ? 0 : partition->second));
        }
        return out;
    }

    /// The MEF event behind node `index`: only a Variable has one, its BasicEvent. Throws std::runtime_error for an
    /// unknown node or one that is not a Variable.
    [[nodiscard]] const mef_event *mef_event_of(const std::int32_t index) const {
        const auto it = by_index_.find(index);
        if (it == by_index_.end() || !it->second) {
            throw std::runtime_error("layer_manager::get_mef_event – unknown event_id " + std::to_string(index));
        }
        // Not const: SCRAM's Node::graph() is not.
        auto *variable = dynamic_cast<core::Variable *>(it->second.get());
        if (variable == nullptr) {
            throw std::runtime_error("layer_manager::get_mef_event – resolution for non-variable nodes (gates / constants) is not implemented");
        }
        // Variables are indexed from core::Pdag::kVariableStartIndex.
        const int position = variable->index() - core::Pdag::kVariableStartIndex;
        const auto &basic_events = variable->graph().basic_events();
        if (position < 0 || static_cast<std::size_t>(position) >= basic_events.size()) {
            throw std::runtime_error("layer_manager::get_mef_event – BasicEvent not found for variable index " + std::to_string(index));
        }
        return basic_events.at(position);
    }

  private:
    /// Calls `f(argument, node)` for every input of `gate`, its Variables and then its Gates; a negative argument is a
    /// negated input. The PDAG constant is no input (op_code_of).
    template <typename F> static void for_each_input(core::Gate &gate, F &&f) {
        for (const auto &[argument, variable] : gate.args<core::Variable>()) {
            f(argument, static_cast<const core::Node &>(*variable));
        }
        for (const auto &[argument, child] : gate.args<core::Gate>()) {
            f(argument, static_cast<const core::Node &>(*child));
        }
    }

    /// Calls `variable(v)` for every Variable of `nodes` and `gate(g, op_code_of(g))` for every Gate, in order. Throws
    /// std::logic_error for a node that is neither: the kernels compute only these.
    template <typename OnVariable, typename OnGate> static void for_each_by_op(const std::vector<node_ptr> &nodes, OnVariable &&variable, OnGate &&gate) {
        for (const node_ptr &node : nodes) {
            if (auto *v = dynamic_cast<core::Variable *>(node.get())) {
                variable(*v);
            } else if (auto *g = dynamic_cast<core::Gate *>(node.get())) {
                gate(*g, op_code_of(*g));
            } else {
                throw std::logic_error("PDAG node " + std::to_string(node->index()) + " is neither a Variable nor a Gate");
            }
        }
    }

    /// A node's place in its layer: Variables, then Gates, each by index.
    [[nodiscard]] static std::pair<bool, int> position(const core::Node &node) { return {dynamic_cast<const core::Gate *>(&node) != nullptr, node.index()}; }

    /// Node `node`, computed by partition `partition`, as the execution plan reports it.
    [[nodiscard]] static descriptor::node describe(core::Node &node, const std::size_t partition) {
        descriptor::node description{.index = node.index(), .layer = static_cast<std::size_t>(node.order()), .partition = partition};
        if (dynamic_cast<const core::Variable *>(&node) != nullptr) {
            description.kind = "basic_event";
            description.connective = std::string(builder::name_of(builder::op_code::sample));
        } else if (auto *gate = dynamic_cast<core::Gate *>(&node)) {
            description.kind = "gate";
            description.connective = std::string(builder::name_of(op_code_of(*gate)));
            description.atleast_min = gate->type() == core::kAtleast ? static_cast<std::size_t>(gate->min_number()) : 0;
            description.module = gate->module();
            for_each_input(*gate, [&description](const int argument, const core::Node &) {
                ++(argument < 0 ? description.negated_input_count : description.positive_input_count);
            });
            description.fan_in = description.positive_input_count + description.negated_input_count;
        } else {
            description.kind = "unknown";
            description.connective = "unknown";
        }
        return description;
    }

    /// Gathers `gate` and every node it reaches, once each, by the PDAG's visit marks. Throws std::runtime_error for two
    /// nodes of one index.
    void gather(const std::shared_ptr<core::Gate> &gate) {
        if (gate->Visited()) {
            return;
        }
        gate->Visit(1);
        add(gate);
        for (const auto &[argument, child] : gate->args<core::Gate>()) {
            gather(child);
        }
        for (const auto &[argument, variable] : gate->args<core::Variable>()) {
            if (!variable->Visited()) {
                variable->Visit(1);
                add(variable);
                ++variables_;
            }
        }
    }

    void add(const node_ptr &node) {
        if (!by_index_.emplace(node->index(), node).second) {
            LOG(scram::ERROR) << "Found a node with a duplicate index while gathering all nodes";
            throw std::runtime_error("gather all nodes failed");
        }
        nodes_.push_back(node);
    }

    std::vector<node_ptr> nodes_;
    std::unordered_map<std::int32_t, node_ptr> by_index_;
    std::vector<std::vector<node_ptr>> layers_;
    std::size_t variables_ = 0;
    std::size_t indices_ = 0;
};

} // namespace fuzzybits::mc::executor::compat
