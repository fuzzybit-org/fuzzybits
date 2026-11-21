#include "ir/pc/builder.h"
#include "io/ftp/detail/model.h"

#include "api/diagnostics.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::ftp::detail {
    namespace {

        // Nodes are numbered in depth-first post-order from the targets (events
        // when first reached, gates after their operands), so a written and
        // re-read circuit numbers its nodes identically.
        class PcBuilder final {
        public:
            explicit PcBuilder(ParsedProject const &model)
                : model_(model), gateNodes_(model.definitions.size()), eventNodes_(model.events.size()),
                  gateState_(model.definitions.size(), Visit::unvisited) {}

            [[nodiscard]] auto build(std::vector<Target> const &targets) -> ir::pc::finalize_result {
                for (auto const &target : targets) {
                    const auto resolved = collapse(target);
                    const auto node = resolved.gate ? buildDefinition(resolved.index) : eventNode(resolved.index);
                    if (!node.valid()) {
                        break;
                    }
                    const auto targetId = builder_.add_target(node);
                    (void)targetId;
                }
                if (diagnostics_.has_errors()) {
                    return ir::pc::finalize_result{
                        .output = {},
                        .diagnostics = std::move(diagnostics_),
                    };
                }
                auto finalized = std::move(builder_).seal();
                finalized.diagnostics.append(std::move(diagnostics_));
                return finalized;
            }

        private:
            enum class Visit : std::uint8_t {
                unvisited,
                active,
                done,
            };

            struct Frame final {
                std::uint32_t gate = 0;
                std::uint32_t next = 0;
            };

            [[nodiscard]] auto resolve(const std::string_view name) const -> Target {
                if (const auto gate = model_.definitionByName.find(name); gate != model_.definitionByName.end()) {
                    return Target{.name = name, .index = gate->second, .gate = true};
                }
                return Target{.name = name, .index = model_.eventByName.at(name), .gate = false};
            }

            // A target on an identity gate (one plain operand) observes the operand:
            // the writer's `T<k> + node` target lines read back to the same circuit.
            [[nodiscard]] auto collapse(Target target) const -> Target {
                for (std::size_t steps = 0; target.gate && steps < model_.definitions.size(); ++steps) {
                    auto const &definition = model_.definitions[target.index];
                    const auto identity = definition.kind != DefinitionKind::at_least || definition.threshold == 1;
                    if (definition.operands.size() != 1 || definition.operands.front().complement || !identity) {
                        break;
                    }
                    target = resolve(definition.operands.front().name);
                }
                return target;
            }

            [[nodiscard]] auto eventNode(const std::uint32_t index) -> ir::pc::bool_node_id {
                if (eventNodes_[index].valid()) {
                    return eventNodes_[index];
                }
                auto const &event = model_.events[index];
                ir::pc::probability_source_id probability;
                if (event.frequency) {
                    probability = builder_.add_parameter_probability(std::string(event.name));
                    diagnostics_.push(api::make_warning("io.ftp.non-probability-import",
                                                        "imported value for basic event '" + std::string(event.name) +
                                                            "' is a frequency outside the literal probability range [0, 1]"));
                } else {
                    probability = builder_.add_literal_probability(event.value);
                }
                eventNodes_[index] = builder_.add_basic_event(probability);
                return eventNodes_[index];
            }

            [[nodiscard]] auto buildDefinition(const std::uint32_t root) -> ir::pc::bool_node_id {
                if (gateState_[root] == Visit::done) {
                    return gateNodes_[root];
                }
                std::vector<Frame> stack{Frame{.gate = root}};
                gateState_[root] = Visit::active;
                while (!stack.empty()) {
                    auto &frame = stack.back();
                    auto const &definition = model_.definitions[frame.gate];
                    if (frame.next < definition.operands.size()) {
                        const auto operand = resolve(definition.operands[frame.next++].name);
                        if (!operand.gate) {
                            (void)eventNode(operand.index);
                            continue;
                        }
                        if (gateState_[operand.index] == Visit::active) {
                            diagnostics_.push(api::make_error("io.ftp.cycle", "FTP gate cycle involving '" + std::string(operand.name) + "'"));
                            return ir::pc::bool_node_id::invalid();
                        }
                        if (gateState_[operand.index] == Visit::unvisited) {
                            gateState_[operand.index] = Visit::active;
                            stack.push_back(Frame{.gate = operand.index});
                        }
                        continue;
                    }

                    std::vector<ir::pc::edge_ref> operands;
                    operands.reserve(definition.operands.size());
                    for (auto const &[name, complement] : definition.operands) {
                        const auto operand = resolve(name);
                        operands.push_back(ir::pc::edge_ref{
                            .source = operand.gate ? gateNodes_[operand.index] : eventNodes_[operand.index],
                            .complement = complement,
                        });
                    }
                    ir::pc::bool_node_id node;
                    switch (definition.kind) {
                    case DefinitionKind::and_gate:
                        node = builder_.add_and(operands);
                        break;
                    case DefinitionKind::or_gate:
                        node = builder_.add_or(operands);
                        break;
                    case DefinitionKind::at_least:
                        node = builder_.add_cardinality(definition.threshold, static_cast<std::uint32_t>(operands.size()), operands);
                        break;
                    }
                    gateNodes_[frame.gate] = node;
                    gateState_[frame.gate] = Visit::done;
                    stack.pop_back();
                }
                return gateNodes_[root];
            }

            ParsedProject const &model_;
            ir::pc::builder builder_;
            api::diagnostic_list diagnostics_;
            std::vector<ir::pc::bool_node_id> gateNodes_;
            std::vector<ir::pc::bool_node_id> eventNodes_;
            std::vector<Visit> gateState_;
        };

    } // namespace

    auto selected_targets(ParsedProject const &project, api::option_set const &options, api::diagnostic_list &diagnostics)
        -> std::vector<Target> {
        const auto requested = options.get<std::string>(target_option());
        if (!requested || requested->empty()) {
            return project.targets;
        }
        if (const auto gate = project.definitionByName.find(*requested); gate != project.definitionByName.end()) {
            return {Target{.name = gate->first, .index = gate->second, .gate = true}};
        }
        if (const auto event = project.eventByName.find(*requested); event != project.eventByName.end()) {
            return {Target{.name = event->first, .index = event->second, .gate = false}};
        }
        diagnostics.push(api::make_error("io.ftp.unknown-target", "requested target '" + *requested + "' is neither a gate nor an imported event"));
        return {};
    }

    auto build_pc(ParsedProject const &model, std::vector<Target> const &targets) -> ir::pc::finalize_result {
        return PcBuilder(model).build(targets);
    }

} // namespace fuzzybits::io::backend::fuzzybits::ftp::detail
