#include "compile/pra_to_pc/descriptor.h"

#include "api/diagnostics.h"
#include "api/ids.h"
#include "ir/handle.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/target.h"
#include "ir/pra/event.h"
#include "ir/pra/expression.h"
#include "ir/pra/gate.h"
#include "ir/pra/id.h"
#include "ir/pra/model.h"
#include "ir/pra/projection.h"
#include "ir/pra/target.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc {
    namespace {

        namespace pc = ir::pc;
        namespace pra = ir::pra;

        [[nodiscard]] auto in_quotes(const std::string_view text) -> std::string {
            return "'" + std::string(text) + "'";
        }

        [[nodiscard]] auto format_value(const double value) -> std::string {
            std::ostringstream out;
            out.precision(17);
            out << value;
            return out.str();
        }

        [[nodiscard]] auto pc_op(const pra::gate_kind kind) noexcept -> pc::op_code {
            switch (kind) {
            case pra::gate_kind::and_gate:
                return pc::op_code::and_gate;
            case pra::gate_kind::or_gate:
                return pc::op_code::or_gate;
            case pra::gate_kind::cardinality:
                return pc::op_code::cardinality;
            case pra::gate_kind::not_gate:
                return pc::op_code::not_gate;
            case pra::gate_kind::xor_gate:
                return pc::op_code::xor_gate;
            case pra::gate_kind::nand_gate:
                return pc::op_code::nand_gate;
            case pra::gate_kind::nor_gate:
                return pc::op_code::nor_gate;
            case pra::gate_kind::xnor_gate:
                return pc::op_code::xnor_gate;
            }
            return pc::op_code::and_gate;
        }

        [[nodiscard]] auto pc_role(const pra::target_role role) noexcept -> pc::observable_role {
            return role == pra::target_role::watch ? pc::observable_role::watch : pc::observable_role::primary;
        }

        [[nodiscard]] constexpr auto lowerable(const pra::element_kind kind) noexcept -> bool {
            return kind == pra::element_kind::gate || kind == pra::element_kind::basic_event || kind == pra::element_kind::house_event;
        }

        // One value per row of each lowerable element table.
        template <typename value_type> class element_columns final {
        public:
            element_columns(pra::model const &model, const value_type initial)
                : basic_events_(model.basic_events().size(), initial), house_events_(model.house_events().size(), initial),
                  gates_(model.gates().size(), initial) {}

            // `ref` is a gate, basic event or house event.
            [[nodiscard]] auto operator[](const pra::domain_ref ref) -> value_type & {
                return column(*this, ref.kind())[ref.raw()];
            }
            [[nodiscard]] auto operator[](const pra::domain_ref ref) const -> value_type const & {
                return column(*this, ref.kind())[ref.raw()];
            }

        private:
            template <typename self_type> [[nodiscard]] static auto column(self_type &self, const pra::element_kind kind) -> auto & {
                switch (kind) {
                case pra::element_kind::house_event:
                    return self.house_events_;
                case pra::element_kind::gate:
                    return self.gates_;
                default:
                    return self.basic_events_;
                }
            }

            std::vector<value_type> basic_events_;
            std::vector<value_type> house_events_;
            std::vector<value_type> gates_;
        };

        enum class override_state : std::uint8_t {
            none,
            constant_false,
            constant_true,
        };

        struct resolved_target final {
            pra::domain_target_id domain_target;
            pra::domain_ref subject;
            pc::observable_role role = pc::observable_role::primary;
            pra::negation_set negation;
            pra::literal_mode literals = pra::literal_mode::atom_split;
        };

        // Operands of these gates are not monotone in the gate's value.
        [[nodiscard]] auto non_monotone(pra::gate const &row, const std::size_t operands) noexcept -> bool {
            switch (row.kind) {
            case pra::gate_kind::not_gate:
            case pra::gate_kind::nand_gate:
            case pra::gate_kind::nor_gate:
            case pra::gate_kind::xor_gate:
            case pra::gate_kind::xnor_gate:
                return true;
            case pra::gate_kind::cardinality:
                return row.bounds.high < operands;
            default:
                return false;
            }
        }

        constexpr std::size_t listed_warning_limit = 32;

        constexpr std::uint64_t id_limit = std::numeric_limits<std::uint32_t>::max() - 1U;

        class lowering final {
        public:
            lowering(ir::handle handle, pra::model const &model, quantification_config const &config)
                : handle_(std::move(handle)), model_(model), config_(config), overridden_(model, override_state::none), reached_(model, 0),
                  nodes_(model, pc::bool_node_id::invalid()), is_initiator_(model.basic_events().size(), 0),
                  sources_(model.expressions().size(), pc::probability_source_id::invalid()) {
                for (auto const &initiator : model.initiating_events()) {
                    is_initiator_[initiator.event.raw()] = 1;
                }
            }

            [[nodiscard]] auto run() && -> lowering_result {
                resolve_targets();
                resolve_overrides();
                if (diagnostics_.has_errors()) {
                    return lowering_result::fail(std::move(diagnostics_));
                }
                mark_cone();
                warn_unused_overrides();
                check_probabilities();
                if (diagnostics_.has_errors()) {
                    return lowering_result::fail(std::move(diagnostics_));
                }
                collect_target_events();
                check_capacity();
                if (diagnostics_.has_errors()) {
                    return lowering_result::fail(std::move(diagnostics_));
                }
                return emit();
            }

        private:
            void error(std::string code, std::string message) {
                diagnostics_.push(api::make_error(std::move(code), std::move(message)));
            }

            [[nodiscard]] auto name(const pra::domain_ref ref) const -> std::string {
                return in_quotes(model_.stable_name(ref));
            }

            [[nodiscard]] auto is_overridden(const pra::domain_ref ref) const -> bool {
                return overridden_[ref] != override_state::none;
            }

            void resolve_targets() {
                std::set<pra::domain_ref> seen;
                const auto add = [&](target_request const &request, const bool watch) {
                    if (request.negation.empty()) {
                        error("compile.pra_to_pc.empty-negation", "target " + in_quotes(request.name) + " requests no negation semantics");
                        return;
                    }
                    resolved_target out{.negation = request.negation, .literals = request.literals.value_or(config_.default_literals)};
                    pra::domain_ref key;
                    if (!request.element) {
                        const auto target = model_.find_target(request.name);
                        if (!target.valid()) {
                            error("compile.pra_to_pc.unknown-target", "no domain target is named " + in_quotes(request.name));
                            return;
                        }
                        out.domain_target = target;
                        out.subject = model_.target_event(target);
                        out.role = watch ? pc::observable_role::watch : pc_role(model_.targets()[target.raw()].role);
                        key = target;
                    } else {
                        const auto kind = *request.element;
                        if (kind != pra::element_kind::gate && kind != pra::element_kind::basic_event) {
                            error("compile.pra_to_pc.invalid-target", "target " + in_quotes(request.name) +
                                                                          " must be a domain target, a gate or a basic event, not a " +
                                                                          std::string(pra::to_string(kind)));
                            return;
                        }
                        const auto subject = model_.find(kind, request.name);
                        if (!subject.valid()) {
                            error("compile.pra_to_pc.unknown-target", "no " + std::string(pra::to_string(kind)) + " is named " + in_quotes(request.name));
                            return;
                        }
                        out.subject = subject;
                        out.role = watch ? pc::observable_role::watch : pc::observable_role::primary;
                        key = subject;
                    }
                    if (!seen.insert(key).second) {
                        error("compile.pra_to_pc.duplicate-target", "target " + in_quotes(request.name) + " is requested more than once");
                        return;
                    }
                    targets_.push_back(out);
                };

                if (config_.targets.empty()) {
                    auto const targets = model_.targets();
                    for (std::uint32_t index = 0; index < targets.size(); ++index) {
                        if (targets[index].role == pra::target_role::primary) {
                            const pra::domain_ref target = pra::domain_target_id(index);
                            add(target_request{.name = std::string(model_.stable_name(target)), .negation = config_.default_negation}, false);
                        }
                    }
                } else {
                    for (auto const &request : config_.targets) {
                        add(request, false);
                    }
                }
                for (auto const &request : config_.watch_targets) {
                    add(request, true);
                }
                if (targets_.empty() && !diagnostics_.has_errors()) {
                    error("compile.pra_to_pc.no-targets", "nothing to lower: the config names no targets and the model has no primary targets");
                }
            }

            void resolve_overrides() {
                overrides_.assign(config_.overrides.size(), pra::domain_ref{});
                for (std::size_t index = 0; index < config_.overrides.size(); ++index) {
                    auto const &request = config_.overrides[index];
                    pra::domain_ref element;
                    if (request.kind) {
                        if (!lowerable(*request.kind)) {
                            error("compile.pra_to_pc.invalid-override", "override " + in_quotes(request.name) +
                                                                            " must name a gate, basic event or house event, not a " +
                                                                            std::string(pra::to_string(*request.kind)));
                            continue;
                        }
                        element = model_.find(*request.kind, request.name);
                        if (!element.valid()) {
                            error("compile.pra_to_pc.unknown-element",
                                  "override names no " + std::string(pra::to_string(*request.kind)) + " " + in_quotes(request.name));
                            continue;
                        }
                    } else {
                        std::uint32_t matches = 0;
                        for (const auto kind : {pra::element_kind::gate, pra::element_kind::basic_event, pra::element_kind::house_event}) {
                            if (const auto found = model_.find(kind, request.name); found.valid()) {
                                element = found;
                                ++matches;
                            }
                        }
                        if (matches == 0) {
                            error("compile.pra_to_pc.unknown-element", "override names no gate, basic event or house event " + in_quotes(request.name));
                            continue;
                        }
                        if (matches > 1) {
                            error("compile.pra_to_pc.ambiguous-element",
                                  "override " + in_quotes(request.name) + " names elements of several kinds; give the kind");
                            continue;
                        }
                    }
                    auto &state = overridden_[element];
                    if (state != override_state::none) {
                        error("compile.pra_to_pc.duplicate-override",
                              std::string(pra::to_string(element.kind())) + " " + name(element) + " is overridden more than once");
                        continue;
                    }
                    state = request.value ? override_state::constant_true : override_state::constant_false;
                    overrides_[index] = element;
                }
            }

            // Marks every element the targets reach; an overridden element is
            // reached but its operands are not.
            void mark_cone() {
                std::vector<pra::gate_id> stack;
                const auto reach = [&](const pra::domain_ref element) {
                    auto &reached = reached_[element];
                    if (reached != 0) {
                        return;
                    }
                    reached = 1;
                    if (const auto gate = element.as<pra::gate_id>(); gate.valid() && !is_overridden(element)) {
                        stack.push_back(gate);
                    }
                };
                for (auto const &target : targets_) {
                    reach(target.subject);
                }
                while (!stack.empty()) {
                    const auto gate = stack.back();
                    stack.pop_back();
                    for (auto const &operand : model_.operands(gate)) {
                        reach(operand.event);
                    }
                }
            }

            void warn_unused_overrides() {
                for (std::size_t index = 0; index < overrides_.size(); ++index) {
                    if (reached_[overrides_[index]] == 0) {
                        diagnostics_.push(api::make_warning("compile.pra_to_pc.unused-override",
                                                            "override of " + std::string(pra::to_string(overrides_[index].kind())) + " " +
                                                                name(overrides_[index]) + " has no effect: it is outside every target's cone"));
                    }
                }
            }

            void check_probabilities() {
                auto const events = model_.basic_events();
                auto const expressions = model_.expressions();
                for (std::uint32_t index = 0; index < events.size(); ++index) {
                    const pra::domain_ref element = pra::basic_event_id(index);
                    if (reached_[element] == 0 || is_overridden(element)) {
                        continue;
                    }
                    const auto expression_id = events[index].expression;
                    auto const &expression = expressions[expression_id.raw()];
                    const auto subject = "basic event " + name(element) + (is_initiator_[index] != 0 ? " (initiating event)" : "");
                    switch (expression.kind) {
                    case pra::expression_kind::literal:
                        if (expression.quantity == pra::quantity_kind::frequency) {
                            if (expression.value > 1.0) {
                                error("compile.pra_to_pc.frequency-out-of-range", subject + " has frequency " + format_value(expression.value) +
                                                                                      ", which cannot be used as a probability");
                                continue;
                            }
                            diagnostics_.push(api::make_warning("compile.pra_to_pc.frequency-as-probability",
                                                                subject + " has frequency " + format_value(expression.value) +
                                                                    ", used as a probability"));
                        }
                        if (!sources_[expression_id.raw()].valid()) {
                            sources_[expression_id.raw()] = builder_.add_literal_probability(expression.value);
                        }
                        break;
                    case pra::expression_kind::parameter:
                        error("compile.pra_to_pc.parameter-probability", subject + " has parameter probability " +
                                                                             in_quotes(model_.text(expression.parameter_name)) +
                                                                             "; the R1 lowering requires literal probabilities");
                        break;
                    case pra::expression_kind::unsupported: {
                        auto detail = std::string(model_.text(expression.diagnostic));
                        if (const auto text = model_.text(expression.detail); !text.empty()) {
                            detail += ": " + std::string(text);
                        }
                        error("compile.pra_to_pc.unsupported-probability",
                              subject + " has an unsupported probability expression (" + detail + ")");
                        break;
                    }
                    }
                }
            }

            // Per target: the initiating events of its cone, the basic events it
            // reaches through operands marked `literal`, and which events occur
            // below a non-monotone step (a complemented gate reference, a
            // non_monotone() operand, or an unmarked complemented event operand).
            // Literals come only from the marker, never from a complement alone.
            // Each target walks its own cone, so the sets do not depend on what
            // else is packed into the circuit. A gate is expanded at most twice
            // per target: once when reached, once more if later reached below a
            // non-monotone step.
            void collect_target_events() {
                initiator_ranges_.assign(targets_.size(), {});
                literal_ranges_.assign(targets_.size(), {});
                if (model_.initiating_events().empty() && model_.summary().literal_operand_count == 0) {
                    return;
                }
                const auto event_count = model_.basic_events().size();
                std::vector<std::uint32_t> gate_stamp(model_.gates().size(), 0);
                std::vector<std::uint32_t> gate_below_stamp(model_.gates().size(), 0);
                std::vector<std::uint32_t> initiator_stamp(event_count, 0);
                std::vector<std::uint32_t> literal_stamp(event_count, 0);
                std::vector<std::uint32_t> event_below_stamp(event_count, 0);
                std::vector<std::uint32_t> unmarked_stamp(event_count, 0);
                std::vector<std::pair<pra::gate_id, bool>> stack;
                std::vector<pra::basic_event_id> divergent;
                std::vector<std::uint32_t> warned_initiator(event_count, 0);
                std::size_t literal_warnings = 0;
                std::size_t initiator_warnings = 0;
                for (std::uint32_t target = 0; target < targets_.size(); ++target) {
                    const auto stamp = target + 1;
                    const auto initiators_begin = initiator_events_.size();
                    const auto literals_begin = literal_events_.size();
                    const auto visit = [&](const pra::domain_ref element, const bool complement, const bool literal, const bool below) {
                        if (is_overridden(element)) {
                            return;
                        }
                        if (const auto gate = element.as<pra::gate_id>(); gate.valid()) {
                            const bool gate_below = below || complement;
                            if (gate_stamp[gate.raw()] != stamp) {
                                gate_stamp[gate.raw()] = stamp;
                                gate_below_stamp[gate.raw()] = gate_below ? stamp : 0;
                                stack.emplace_back(gate, gate_below);
                            } else if (gate_below && gate_below_stamp[gate.raw()] != stamp) {
                                gate_below_stamp[gate.raw()] = stamp;
                                stack.emplace_back(gate, true);
                            }
                            return;
                        }
                        const auto event = element.as<pra::basic_event_id>();
                        if (!event.valid()) {
                            return;
                        }
                        if (is_initiator_[event.raw()] != 0 && initiator_stamp[event.raw()] != stamp) {
                            initiator_stamp[event.raw()] = stamp;
                            initiator_events_.push_back(event);
                        }
                        if (literal && literal_stamp[event.raw()] != stamp) {
                            literal_stamp[event.raw()] = stamp;
                            literal_events_.push_back(event);
                        }
                        if (below || (complement && !literal)) {
                            event_below_stamp[event.raw()] = stamp;
                        }
                        if (complement && !literal) {
                            unmarked_stamp[event.raw()] = stamp;
                        }
                    };
                    visit(targets_[target].subject, false, false, false);
                    while (!stack.empty()) {
                        const auto [gate, below] = stack.back();
                        stack.pop_back();
                        if (!below && gate_below_stamp[gate.raw()] == stamp) {
                            continue;
                        }
                        const auto operands = model_.operands(gate);
                        const bool operands_below = below || non_monotone(model_.gates()[gate.raw()], operands.size());
                        for (auto const &operand : operands) {
                            visit(operand.event, operand.complement, operand.literal, operands_below);
                        }
                    }
                    initiator_ranges_[target] = sorted_range(initiator_events_, initiators_begin);
                    literal_ranges_[target] = sorted_range(literal_events_, literals_begin);

                    auto const &request = targets_[target];
                    auto const &range = literal_ranges_[target];
                    divergent.clear();
                    for (std::uint32_t offset = 0; offset < range.count; ++offset) {
                        const auto event = literal_events_[range.offset + offset];
                        // The hull check reads every complemented edge into an atom-split or
                        // omega_L event as the literal, so an unmarked complement of the same
                        // event cannot keep its negation reading.
                        if (request.negation.monotone_hull && request.literals != pra::literal_mode::negation &&
                            unmarked_stamp[event.raw()] == stamp) {
                            error("compile.pra_to_pc.mixed-literal", "target " + target_name(request) + ": basic event " + name(event) +
                                                                         " has both literal and unmarked complemented operands");
                        } else if (request.negation.monotone_hull && event_below_stamp[event.raw()] == stamp) {
                            divergent.push_back(event);
                        }
                        if (config_.max_initiators_per_cutset && is_initiator_[event.raw()] != 0 && warned_initiator[event.raw()] == 0) {
                            warned_initiator[event.raw()] = 1;
                            if (initiator_warnings++ < listed_warning_limit) {
                                diagnostics_.push(api::make_warning(
                                    "compile.pra_to_pc.complemented-initiator",
                                    "initiating event " + name(event) +
                                        " has a literal complement; max_initiators_per_cutset is exact only for uncomplemented initiators"));
                            }
                        }
                    }
                    for (const auto event : divergent) {
                        if (literal_warnings++ < listed_warning_limit) {
                            diagnostics_.push(api::make_warning("compile.pra_to_pc.complement-under-negation",
                                                                "target " + target_name(request) + ": basic event " + name(event) +
                                                                    " is a literal and also occurs below a negation, "
                                                                    "xor or upper-bounded cardinality; the literal modes can differ"));
                        }
                    }
                }
                const auto summarize = [&](const std::size_t total, const char *code, const char *what) {
                    if (total > listed_warning_limit) {
                        diagnostics_.push(api::make_warning(code, std::to_string(total - listed_warning_limit) + " further " + what + " not listed"));
                    }
                };
                summarize(literal_warnings, "compile.pra_to_pc.complement-under-negation", "literal events below a negation");
                summarize(initiator_warnings, "compile.pra_to_pc.complemented-initiator", "initiating events with a literal complement");
            }

            [[nodiscard]] auto target_name(resolved_target const &target) const -> std::string {
                return target.domain_target.valid() ? name(target.domain_target) : name(target.subject);
            }

            [[nodiscard]] static auto sorted_range(std::vector<pra::basic_event_id> &column, const std::size_t begin)
                -> pra::table_range<pra::basic_event_id> {
                std::sort(column.begin() + static_cast<std::ptrdiff_t>(begin), column.end());
                return {.offset = static_cast<std::uint32_t>(begin), .count = static_cast<std::uint32_t>(column.size() - begin)};
            }

            [[nodiscard]] auto delete_term_size(const std::size_t target) const -> std::uint32_t {
                if (!config_.max_initiators_per_cutset) {
                    return 0;
                }
                const auto count = initiator_ranges_[target].count;
                return count > *config_.max_initiators_per_cutset ? count : 0;
            }

            void check_capacity() {
                std::uint64_t nodes = 0;
                std::uint64_t edges = 0;
                const auto count = [&](const pra::domain_ref element) {
                    if (reached_[element] == 0) {
                        return;
                    }
                    ++nodes;
                    if (const auto gate = element.as<pra::gate_id>(); gate.valid() && !is_overridden(element)) {
                        edges += model_.operands(gate).size();
                    }
                };
                for (std::uint32_t index = 0; index < model_.basic_events().size(); ++index) {
                    count(pra::basic_event_id(index));
                }
                for (std::uint32_t index = 0; index < model_.house_events().size(); ++index) {
                    count(pra::house_event_id(index));
                }
                for (std::uint32_t index = 0; index < model_.gates().size(); ++index) {
                    count(pra::gate_id(index));
                }
                for (std::size_t target = 0; target < targets_.size(); ++target) {
                    if (const auto size = delete_term_size(target); size != 0) {
                        nodes += 2;
                        edges += std::uint64_t{size} + 2;
                    }
                }
                if (nodes > id_limit || edges > id_limit || initiator_events_.size() > id_limit || literal_events_.size() > id_limit) {
                    error("compile.pra_to_pc.capacity-exceeded", "the lowered circuit exceeds the R1 limit of 2^32 - 2 nodes or edges");
                }
            }

            [[nodiscard]] auto emit() -> lowering_result {
                transform_report report;
                std::vector<pra::event_binding> events;
                std::vector<pra::override_binding> overrides;
                std::vector<pra::generated_binding> generated;
                std::vector<pra::target_binding> targets;

                const auto bind = [&](const pra::domain_ref element, const pc::bool_node_id node, const pra::event_binding_kind kind) {
                    nodes_[element] = node;
                    events.push_back(pra::event_binding{.element = element, .node = node, .kind = kind});
                };
                const auto lowered = [&](const pra::domain_ref element) { return reached_[element] != 0 && !is_overridden(element); };

                for (std::size_t index = 0; index < overrides_.size(); ++index) {
                    const auto element = overrides_[index];
                    const bool value = config_.overrides[index].value;
                    if (reached_[element] == 0) {
                        overrides.push_back(pra::override_binding{.element = element, .value = value});
                        continue;
                    }
                    const auto node = builder_.add_constant(value);
                    bind(element, node, pra::event_binding_kind::constant_override);
                    overrides.push_back(pra::override_binding{.element = element, .value = value, .node = node});
                    ++report.overridden_elements;
                }
                auto const basic_events = model_.basic_events();
                for (std::uint32_t index = 0; index < basic_events.size(); ++index) {
                    if (const pra::domain_ref element = pra::basic_event_id(index); lowered(element)) {
                        bind(element, builder_.add_basic_event(sources_[basic_events[index].expression.raw()]), pra::event_binding_kind::basic_event);
                        ++report.source_basic_events;
                    }
                }
                auto const house_events = model_.house_events();
                for (std::uint32_t index = 0; index < house_events.size(); ++index) {
                    if (const pra::domain_ref element = pra::house_event_id(index); lowered(element)) {
                        bind(element, builder_.add_house_event(house_events[index].value), pra::event_binding_kind::house_event);
                        ++report.source_house_events;
                    }
                }
                std::vector<pc::edge_ref> edges;
                for (const auto gate : model_.topological_gates()) {
                    if (!lowered(gate)) {
                        continue;
                    }
                    edges.clear();
                    for (auto const &operand : model_.operands(gate)) {
                        edges.push_back(pc::edge_ref{.source = nodes_[operand.event], .complement = operand.complement});
                    }
                    auto const &row = model_.gates()[gate.raw()];
                    const auto node = row.kind == pra::gate_kind::cardinality ? builder_.add_cardinality(row.bounds.low, row.bounds.high, edges)
                                                                              : builder_.add_gate(pc_op(row.kind), edges);
                    bind(gate, node, pra::event_binding_kind::gate);
                    ++report.source_gates;
                    report.source_operands += edges.size();
                }

                for (std::uint32_t index = 0; index < targets_.size(); ++index) {
                    auto const &request = targets_[index];
                    const auto target = pc::target_id(index);
                    pra::target_binding binding{
                        .domain_target = request.domain_target,
                        .subject = request.subject,
                        .pc_target = target,
                        .role = request.role,
                        .root = nodes_[request.subject],
                        .negation = request.negation,
                        .literals = request.literals,
                        .initiators = initiator_ranges_[index],
                        .literal_events = literal_ranges_[index],
                    };
                    binding.node = binding.root;
                    if (const auto size = delete_term_size(index); size != 0) {
                        edges.clear();
                        auto const &range = initiator_ranges_[index];
                        for (std::uint32_t offset = 0; offset < range.count; ++offset) {
                            edges.push_back(pc::edge_ref{.source = nodes_[pra::domain_ref(initiator_events_[range.offset + offset])]});
                        }
                        const auto delete_term = builder_.add_cardinality(*config_.max_initiators_per_cutset + 1, size, edges);
                        const auto conjunction = builder_.add_and({pc::edge_ref{.source = binding.root}, pc::edge_ref{.source = delete_term, .complement = true}});
                        generated.push_back(pra::generated_binding{.node = delete_term, .kind = pra::generated_kind::initiator_delete_term, .target = target});
                        generated.push_back(
                            pra::generated_binding{.node = conjunction, .kind = pra::generated_kind::initiator_rule_conjunction, .target = target});
                        binding.initiator_delete_term = delete_term;
                        binding.node = conjunction;
                        report.generated_nodes += 2;
                        report.generated_edges += std::size_t{size} + 2;
                    }
                    (void)builder_.add_target(binding.node, binding.role);
                    targets.push_back(binding);
                }

                auto sealed = std::move(builder_).seal();
                if (!sealed.ok()) {
                    diagnostics_.append(std::move(sealed.diagnostics));
                    error("compile.pra_to_pc.invalid-circuit", "the lowered circuit failed ir::pc validation");
                    return lowering_result::fail(std::move(diagnostics_));
                }
                report.max_initiators_per_cutset = config_.max_initiators_per_cutset;
                report.pc_nodes = sealed.output.node_count();
                report.pc_edges = sealed.output.edge_count();
                report.pc_targets = sealed.output.targets().size();

                return lowering_result::ok(
                    pra_pc_lowering{
                        .pra_model = std::move(handle_),
                        .circuit = std::move(sealed.output),
                        .projection = pra::pc_projection(std::move(targets), std::move(events), std::move(overrides), std::move(generated),
                                                         std::move(initiator_events_), std::move(literal_events_)),
                        .report = report,
                    },
                    std::move(diagnostics_));
            }

            ir::handle handle_;
            pra::model const &model_;
            quantification_config const &config_;
            api::diagnostic_list diagnostics_;
            pc::builder builder_;
            std::vector<resolved_target> targets_;
            // Parallel to config_.overrides; invalid where resolution failed.
            std::vector<pra::domain_ref> overrides_;
            element_columns<override_state> overridden_;
            element_columns<std::uint8_t> reached_;
            element_columns<pc::bool_node_id> nodes_;
            std::vector<std::uint8_t> is_initiator_;
            // By expression id: one PC probability source per shared PRA expression.
            std::vector<pc::probability_source_id> sources_;
            std::vector<pra::basic_event_id> initiator_events_;
            std::vector<pra::basic_event_id> literal_events_;
            std::vector<pra::table_range<pra::basic_event_id>> initiator_ranges_;
            std::vector<pra::table_range<pra::basic_event_id>> literal_ranges_;
        };

    } // namespace

    auto endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("compile.fuzzybits.pra_to_pc");
    }

    auto lower(ir::pra::model const &model, quantification_config const &config) -> lowering_result {
        if (!model.valid()) {
            return lowering_result::fail(api::make_error("compile.pra_to_pc.invalid-model", "pra_to_pc requires a sealed ir::pra model"));
        }
        return lowering(model.to_handle(), model, config).run();
    }

    auto lower(ir::handle const &model, quantification_config const &config) -> lowering_result {
        auto const *pra_model = ir::pra::model_from(model);
        if (pra_model == nullptr || !pra_model->valid()) {
            return lowering_result::fail(api::make_error("compile.pra_to_pc.unsupported-ir", "pra_to_pc requires an ir::pra model handle"));
        }
        return lowering(model, *pra_model, config).run();
    }

} // namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc
