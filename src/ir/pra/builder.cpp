#include "ir/pra/builder.h"

#include "api/diagnostics.h"
#include "ir/pra/descriptor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pra {

    namespace {

        constexpr std::size_t row_limit = std::numeric_limits<std::uint32_t>::max();
        constexpr std::uint32_t invalid_row = std::numeric_limits<std::uint32_t>::max();
        constexpr std::size_t cycle_names_shown = 8;
        constexpr std::size_t duplicate_locations_shown = 8;

        using kind_mask = std::uint8_t;

        [[nodiscard]] constexpr auto bit(const element_kind kind) noexcept -> kind_mask {
            return static_cast<kind_mask>(1U << static_cast<unsigned>(kind));
        }

        constexpr kind_mask event_kinds = bit(element_kind::basic_event) | bit(element_kind::house_event) | bit(element_kind::gate);

        constexpr std::array all_kinds{
            element_kind::basic_event, element_kind::house_event, element_kind::initiating_event,
            element_kind::gate,        element_kind::fault_tree,  element_kind::domain_target,
        };

        [[nodiscard]] auto quoted(const std::string_view name) -> std::string {
            std::string out;
            out.reserve(name.size() + 2);
            out += '\'';
            out += name;
            out += '\'';
            return out;
        }

        [[nodiscard]] auto with_article(const std::string_view noun) -> std::string {
            const bool vowel = !noun.empty() && std::string_view("aeiou").find(noun.front()) != std::string_view::npos;
            return (vowel ? "an " : "a ") + std::string(noun);
        }

        [[nodiscard]] auto join_kinds(const kind_mask kinds, const std::string_view separator, const bool articles) -> std::string {
            std::string out;
            for (const auto kind : all_kinds) {
                if ((kinds & bit(kind)) == 0) {
                    continue;
                }
                if (!out.empty()) {
                    out += separator;
                }
                out += articles ? with_article(to_string(kind)) : std::string(to_string(kind));
            }
            return out;
        }

        [[nodiscard]] auto kind_noun(const kind_mask kinds) -> std::string {
            return kinds == event_kinds ? std::string("event") : join_kinds(kinds, " or ", false);
        }

        [[nodiscard]] auto expected_kinds(const kind_mask kinds) -> std::string {
            return kinds == event_kinds ? std::string("an event (gate, basic event or house event)") : join_kinds(kinds, " or ", true);
        }

        [[nodiscard]] constexpr auto gate_kind_name(const gate_kind kind) noexcept -> std::string_view {
            switch (kind) {
            case gate_kind::and_gate:
                return "and";
            case gate_kind::or_gate:
                return "or";
            case gate_kind::cardinality:
                return "cardinality";
            case gate_kind::not_gate:
                return "not";
            case gate_kind::xor_gate:
                return "xor";
            case gate_kind::nand_gate:
                return "nand";
            case gate_kind::nor_gate:
                return "nor";
            }
            return "unknown";
        }

        [[nodiscard]] constexpr auto target_kind_name(const domain_target_kind kind) noexcept -> std::string_view {
            switch (kind) {
            case domain_target_kind::fault_tree_top:
                return "fault_tree_top";
            case domain_target_kind::gate:
                return "gate";
            case domain_target_kind::basic_event:
                return "basic_event";
            case domain_target_kind::sequence:
                return "sequence";
            case domain_target_kind::end_state:
                return "end_state";
            }
            return "unknown";
        }

        [[nodiscard]] auto format_value(const double value) -> std::string {
            std::ostringstream out;
            out << value;
            return out.str();
        }

        template <typename pending_type>
        [[nodiscard]] auto owner_offsets(std::vector<pending_type> const &pending, const std::size_t owners) -> std::vector<std::uint32_t> {
            std::vector<std::uint32_t> offsets(owners + 1, 0);
            for (auto const &item : pending) {
                ++offsets[item.owner.raw() + 1];
            }
            std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
            return offsets;
        }

        template <typename value_type> void add_unique(std::vector<value_type> &values, const value_type value) {
            if (std::ranges::find(values, value) == values.end()) {
                values.push_back(value);
            }
        }

        // Every pass is linear (expected linear for name lookups) and iterative:
        // real models have 1e5 gates and depth > 140, so nothing may recurse on
        // graph depth.
        class sealer final {
        public:
            sealer(detail::model_storage &storage, detail::builder_state const &state, api::diagnostic_list &diagnostics) noexcept
                : storage_(storage), state_(state), diagnostics_(diagnostics) {}

            void run() {
                group_attributes();
                group_sources();
                index_names();
                resolve_operands();
                resolve_initiators();
                check_gates();
                check_expressions();
                build_fault_trees();
                resolve_targets();
                order_gates();
            }

        private:
            struct name_match final {
                domain_ref first;
                kind_mask kinds = 0;
            };

            void error(std::string code, std::string message) {
                diagnostics_.push(api::make_error(std::move(code), std::move(message)));
            }

            [[nodiscard]] auto text(const text_ref ref) const -> std::string_view {
                return detail::text_of(storage_, ref);
            }

            [[nodiscard]] auto reference_name(const text_ref ref) const -> std::string_view {
                return std::string_view{state_.names}.substr(ref.offset, ref.size);
            }

            [[nodiscard]] auto location(const source_location where) const -> std::string {
                std::string out = where.file.valid() && where.file.raw() < storage_.source_files.size()
                                      ? std::string(text(storage_.source_files[where.file.raw()].path))
                                      : std::string("<unknown file>");
                if (where.line != 0) {
                    out += ':';
                    out += std::to_string(where.line);
                }
                return out;
            }

            [[nodiscard]] auto first_location(const element_id element) const -> std::string {
                auto const &row = storage_.elements[element.raw()];
                return row.sources.count == 0 ? std::string{} : location(storage_.source_locations[row.sources.offset]);
            }

            [[nodiscard]] auto short_name(const element_id element) const -> std::string {
                auto const &row = storage_.elements[element.raw()];
                return row.stable_name.empty() ? "#" + std::to_string(row.owner.raw()) : quoted(text(row.stable_name));
            }

            // "gate 'G1' at a.ftp:12"
            [[nodiscard]] auto describe(const element_id element) const -> std::string {
                auto const &row = storage_.elements[element.raw()];
                std::string out(to_string(row.owner.kind()));
                out += ' ';
                out += short_name(element);
                if (row.sources.count != 0) {
                    out += " at ";
                    out += location(storage_.source_locations[row.sources.offset]);
                }
                return out;
            }

            [[nodiscard]] auto describe(const domain_ref ref) const -> std::string {
                if (detail::exists(storage_, ref)) {
                    return describe(detail::element_of(storage_, ref));
                }
                return std::string(to_string(ref.kind())) + " #" + std::to_string(ref.raw());
            }

            void group_attributes() {
                const auto element_count = storage_.elements.size();
                const auto base = static_cast<std::uint32_t>(storage_.attributes.size());
                storage_.project.attributes = table_range<attribute>{.offset = 0, .count = base};

                const auto offsets = owner_offsets(state_.attributes, element_count);
                auto cursor = offsets;
                storage_.attributes.resize(base + state_.attributes.size());
                for (auto const &pending : state_.attributes) {
                    storage_.attributes[base + cursor[pending.owner.raw()]++] = attribute{.name = pending.name, .value = pending.value};
                }
                for (std::size_t element = 0; element < element_count; ++element) {
                    storage_.elements[element].attributes =
                        table_range<attribute>{.offset = base + offsets[element], .count = offsets[element + 1] - offsets[element]};
                }
            }

            void group_sources() {
                const auto element_count = storage_.elements.size();
                const auto offsets = owner_offsets(state_.sources, element_count);
                auto cursor = offsets;
                storage_.source_locations.resize(state_.sources.size());
                for (auto const &pending : state_.sources) {
                    storage_.source_locations[cursor[pending.owner.raw()]++] = pending.location;
                }
                for (std::size_t element = 0; element < element_count; ++element) {
                    storage_.elements[element].sources =
                        table_range<source_location>{.offset = offsets[element], .count = offsets[element + 1] - offsets[element]};
                }

                for (std::uint32_t element = 0; element < element_count; ++element) {
                    for (auto const &where : detail::slice(storage_.source_locations, storage_.elements[element].sources)) {
                        if (!where.file.valid() || where.file.raw() >= storage_.source_files.size()) {
                            error("ir.pra.dangling-reference",
                                  describe(element_id(element)) + " has a source location in a nonexistent source file" +
                                      (where.file.valid() ? " #" + std::to_string(where.file.raw()) : std::string{}));
                        }
                    }
                }
            }

            void index_names() {
                std::size_t named = 0;
                for (auto const &row : storage_.elements) {
                    named += row.stable_name.empty() ? 0 : 1;
                }
                std::size_t capacity = 8;
                while (capacity < 2 * named) {
                    capacity *= 2;
                }
                auto &slots = storage_.name_slots;
                slots.assign(named == 0 ? 0 : capacity, element_id::invalid());
                const auto mask = capacity - 1;

                // (earliest element with the name and kind, later duplicate)
                std::vector<std::pair<element_id, element_id>> duplicates;
                for (std::uint32_t index = 0; index < storage_.elements.size(); ++index) {
                    auto const &row = storage_.elements[index];
                    if (row.stable_name.empty()) {
                        error("ir.pra.missing-name", describe(element_id(index)) + " has no stable name");
                        continue;
                    }
                    const auto name = text(row.stable_name);
                    element_id first;
                    auto slot = detail::name_hash(name) & mask;
                    for (; slots[slot].valid(); slot = (slot + 1) & mask) {
                        auto const &other = storage_.elements[slots[slot].raw()];
                        if (!first.valid() && other.owner.kind() == row.owner.kind() && text(other.stable_name) == name) {
                            first = slots[slot];
                        }
                    }
                    slots[slot] = element_id(index);
                    if (first.valid()) {
                        duplicates.emplace_back(first, element_id(index));
                    }
                }
                report_duplicates(duplicates);
            }

            void report_duplicates(std::vector<std::pair<element_id, element_id>> &duplicates) {
                std::ranges::stable_sort(duplicates, {}, &std::pair<element_id, element_id>::first);
                for (std::size_t begin = 0; begin < duplicates.size();) {
                    const auto first = duplicates[begin].first;
                    auto end = begin;
                    while (end < duplicates.size() && duplicates[end].first == first) {
                        ++end;
                    }
                    const auto definitions = end - begin + 1;
                    std::string locations;
                    const auto append_location = [&](const element_id element) {
                        if (auto where = first_location(element); !where.empty()) {
                            locations += locations.empty() ? ": " : ", ";
                            locations += where;
                        }
                    };
                    append_location(first);
                    for (auto index = begin; index < end && index - begin + 1 < duplicate_locations_shown; ++index) {
                        append_location(duplicates[index].second);
                    }
                    if (definitions > duplicate_locations_shown && !locations.empty()) {
                        locations += ", ...";
                    }
                    auto const &row = storage_.elements[first.raw()];
                    error("ir.pra.duplicate-name", "duplicate " + std::string(to_string(row.owner.kind())) + " name " +
                                                       quoted(text(row.stable_name)) + " (" + std::to_string(definitions) +
                                                       " definitions" + locations + ")");
                    begin = end;
                }
            }

            // One walk of the shared probe chain finds the earliest element of each
            // wanted kind with this name.
            [[nodiscard]] auto lookup(const kind_mask wanted, const std::string_view name) const -> name_match {
                name_match match;
                auto const &slots = storage_.name_slots;
                if (slots.empty()) {
                    return match;
                }
                const auto mask = slots.size() - 1;
                for (auto slot = detail::name_hash(name) & mask; slots[slot].valid(); slot = (slot + 1) & mask) {
                    auto const &row = storage_.elements[slots[slot].raw()];
                    const auto kind = bit(row.owner.kind());
                    if ((wanted & kind) != 0 && (match.kinds & kind) == 0 && text(row.stable_name) == name) {
                        if (match.kinds == 0) {
                            match.first = row.owner;
                        }
                        match.kinds |= kind;
                    }
                }
                return match;
            }

            template <typename context_type>
            [[nodiscard]] auto resolve(detail::pending_reference const &ref, const kind_mask admissible, context_type const &context)
                -> domain_ref {
                if (ref.mode == detail::reference_mode::id) {
                    return resolve_id(ref.target, admissible, context);
                }
                const auto name = reference_name(ref.name);
                auto wanted = admissible;
                if (ref.mode == detail::reference_mode::qualified) {
                    const auto kind = ref.target.kind();
                    if ((admissible & bit(kind)) == 0) {
                        error("ir.pra.invalid-reference", context() + " names " + with_article(to_string(kind)) + " " + quoted(name) +
                                                              " where " + expected_kinds(admissible) + " is required");
                        return {};
                    }
                    wanted = bit(kind);
                }
                const auto match = lookup(wanted, name);
                if (match.kinds == 0) {
                    error("ir.pra.dangling-reference", context() + " references unknown " + kind_noun(wanted) + " " + quoted(name));
                    return {};
                }
                if (std::popcount(match.kinds) > 1) {
                    error("ir.pra.ambiguous-reference", context() + " references " + quoted(name) + ", which names " +
                                                            join_kinds(match.kinds, " and ", true) + "; qualify the reference with a kind");
                    return {};
                }
                return match.first;
            }

            template <typename context_type>
            [[nodiscard]] auto resolve_id(const domain_ref target, const kind_mask admissible, context_type const &context) -> domain_ref {
                if (!target.valid()) {
                    error("ir.pra.dangling-reference", context() + " is an empty reference");
                    return {};
                }
                if ((admissible & bit(target.kind())) == 0) {
                    error("ir.pra.invalid-reference",
                          context() + " references " + describe(target) + " where " + expected_kinds(admissible) + " is required");
                    return {};
                }
                if (!detail::exists(storage_, target)) {
                    error("ir.pra.dangling-reference", context() + " references nonexistent " + describe(target));
                    return {};
                }
                return target;
            }

            void resolve_operands() {
                for (auto const &row : storage_.gates) {
                    for (std::uint32_t index = 0; index < row.operands.count; ++index) {
                        const auto slot = row.operands.offset + index;
                        storage_.gate_operands[slot].event = resolve(state_.operands[slot], event_kinds, [&] {
                            return describe(row.element) + " operand " + std::to_string(index + 1);
                        });
                    }
                }
            }

            void resolve_initiators() {
                for (std::size_t index = 0; index < storage_.initiating_events.size(); ++index) {
                    auto &row = storage_.initiating_events[index];
                    row.event = resolve(state_.initiators[index], bit(element_kind::basic_event), [&] {
                                    return describe(row.element);
                                }).as<basic_event_id>();
                }
            }

            void check_gates() {
                for (auto const &row : storage_.gates) {
                    const auto count = row.operands.count;
                    switch (row.kind) {
                    case gate_kind::not_gate:
                        if (count != 1) {
                            error("ir.pra.invalid-arity", describe(row.element) + " is a not gate with " + std::to_string(count) +
                                                              " operands; it takes exactly one");
                        }
                        break;
                    case gate_kind::cardinality:
                        if (count == 0) {
                            error("ir.pra.empty-gate", describe(row.element) + " is a cardinality gate without operands");
                        } else if (row.bounds.low > row.bounds.high || row.bounds.high > count) {
                            error("ir.pra.invalid-cardinality",
                                  describe(row.element) + " has cardinality bounds (" + std::to_string(row.bounds.low) + ", " +
                                      std::to_string(row.bounds.high) + ") outside 0 <= low <= high <= " + std::to_string(count));
                        }
                        break;
                    case gate_kind::and_gate:
                    case gate_kind::or_gate:
                    case gate_kind::xor_gate:
                    case gate_kind::nand_gate:
                    case gate_kind::nor_gate:
                        if (count == 0) {
                            error("ir.pra.empty-gate",
                                  describe(row.element) + " is " + with_article(gate_kind_name(row.kind)) + " gate without operands");
                        }
                        break;
                    }
                    if (row.fault_tree.valid() && row.fault_tree.raw() >= storage_.fault_trees.size()) {
                        error("ir.pra.dangling-reference",
                              describe(row.element) + " belongs to nonexistent fault tree #" + std::to_string(row.fault_tree.raw()));
                    }
                }
            }

            void check_expressions() {
                std::vector<std::uint8_t> checked(storage_.expressions.size(), 0);
                for (auto const &row : storage_.basic_events) {
                    const auto index = row.expression.raw();
                    if (!row.expression.valid() || index >= storage_.expressions.size()) {
                        error("ir.pra.dangling-reference", describe(row.element) + " references a nonexistent expression" +
                                                               (row.expression.valid() ? " #" + std::to_string(index) : std::string{}));
                        continue;
                    }
                    if (checked[index] == 0) {
                        checked[index] = 1;
                        check_expression(storage_.expressions[index], [&] { return describe(row.element); });
                    }
                }
                for (std::size_t index = 0; index < storage_.expressions.size(); ++index) {
                    if (checked[index] == 0) {
                        check_expression(storage_.expressions[index], [&] { return "expression #" + std::to_string(index); });
                    }
                }
            }

            template <typename context_type> void check_expression(expression const &row, context_type const &context) {
                switch (row.kind) {
                case expression_kind::literal:
                    if (!std::isfinite(row.value)) {
                        error("ir.pra.invalid-literal", context() + " has a non-finite literal value");
                    } else if (row.quantity == quantity_kind::probability && (row.value < 0.0 || row.value > 1.0)) {
                        error("ir.pra.invalid-literal",
                              context() + " has literal probability " + format_value(row.value) + " outside [0, 1]");
                    } else if (row.quantity == quantity_kind::frequency && row.value < 0.0) {
                        error("ir.pra.invalid-literal", context() + " has negative literal frequency " + format_value(row.value));
                    }
                    break;
                case expression_kind::parameter:
                    if (row.parameter_name.empty()) {
                        error("ir.pra.invalid-expression", context() + " has a parameter expression without a parameter name");
                    }
                    break;
                case expression_kind::unsupported:
                    if (row.diagnostic.empty()) {
                        error("ir.pra.invalid-expression", context() + " has an unsupported expression without a diagnostic id");
                    }
                    break;
                }
            }

            void build_fault_trees() {
                const auto tree_count = storage_.fault_trees.size();
                const auto gate_count = storage_.gates.size();
                const auto owned = [tree_count](gate const &row) {
                    return row.fault_tree.valid() && row.fault_tree.raw() < tree_count;
                };

                std::vector<std::uint32_t> offsets(tree_count + 1, 0);
                for (auto const &row : storage_.gates) {
                    if (owned(row)) {
                        ++offsets[row.fault_tree.raw() + 1];
                    }
                }
                std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
                auto cursor = offsets;
                storage_.fault_tree_gates.resize(offsets.back());
                for (std::uint32_t index = 0; index < gate_count; ++index) {
                    if (owned(storage_.gates[index])) {
                        storage_.fault_tree_gates[cursor[storage_.gates[index].fault_tree.raw()]++] = gate_id(index);
                    }
                }
                for (std::size_t tree = 0; tree < tree_count; ++tree) {
                    storage_.fault_trees[tree].gates =
                        table_range<gate_id>{.offset = offsets[tree], .count = offsets[tree + 1] - offsets[tree]};
                }

                std::vector<std::uint32_t> top_offsets(tree_count + 1, 0);
                for (auto const &top : state_.tops) {
                    if (top.tree.valid() && top.tree.raw() < tree_count) {
                        ++top_offsets[top.tree.raw() + 1];
                    } else {
                        error("ir.pra.dangling-reference", "a top gate is listed for nonexistent fault tree" +
                                                               (top.tree.valid() ? " #" + std::to_string(top.tree.raw()) : std::string{}));
                    }
                }
                std::partial_sum(top_offsets.begin(), top_offsets.end(), top_offsets.begin());
                auto top_cursor = top_offsets;
                std::vector<std::uint32_t> tops_by_tree(top_offsets.back());
                for (std::uint32_t index = 0; index < state_.tops.size(); ++index) {
                    if (const auto tree = state_.tops[index].tree; tree.valid() && tree.raw() < tree_count) {
                        tops_by_tree[top_cursor[tree.raw()]++] = index;
                    }
                }

                std::vector<fault_tree_id> listed_by(gate_count);
                storage_.fault_tree_top_gates.reserve(tops_by_tree.size());
                for (std::uint32_t tree_index = 0; tree_index < tree_count; ++tree_index) {
                    const fault_tree_id tree(tree_index);
                    auto &tree_row = storage_.fault_trees[tree_index];
                    const auto begin = static_cast<std::uint32_t>(storage_.fault_tree_top_gates.size());
                    for (auto index = top_offsets[tree_index]; index < top_offsets[tree_index + 1]; ++index) {
                        const auto top = resolve(state_.tops[tops_by_tree[index]].gate, bit(element_kind::gate), [&] {
                                             return describe(tree_row.element) + " top gate";
                                         }).as<gate_id>();
                        if (!top.valid()) {
                            continue;
                        }
                        auto const &gate_row = storage_.gates[top.raw()];
                        if (gate_row.fault_tree != tree) {
                            const auto owner = owned(gate_row) ? describe(storage_.fault_trees[gate_row.fault_tree.raw()].element)
                                                               : std::string("no fault tree");
                            error("ir.pra.invalid-fault-tree-top", describe(tree_row.element) + " lists " + describe(gate_row.element) +
                                                                       " as a top gate, but the gate belongs to " + owner);
                            continue;
                        }
                        if (listed_by[top.raw()] == tree) {
                            error("ir.pra.invalid-fault-tree-top",
                                  describe(tree_row.element) + " lists " + describe(gate_row.element) + " as a top gate more than once");
                            continue;
                        }
                        listed_by[top.raw()] = tree;
                        storage_.fault_tree_top_gates.push_back(top);
                    }
                    tree_row.top_gates = table_range<gate_id>{
                        .offset = begin, .count = static_cast<std::uint32_t>(storage_.fault_tree_top_gates.size()) - begin};
                }
            }

            void resolve_targets() {
                for (std::size_t index = 0; index < storage_.targets.size(); ++index) {
                    auto &row = storage_.targets[index];
                    auto const &subject = state_.subjects[index];
                    const auto context = [&] { return describe(row.element); };
                    switch (row.kind) {
                    case domain_target_kind::fault_tree_top: {
                        row.subject = resolve(subject, bit(element_kind::fault_tree), context);
                        if (const auto tree = row.subject.as<fault_tree_id>(); tree.valid()) {
                            auto const &tree_row = storage_.fault_trees[tree.raw()];
                            if (tree_row.top_gates.count != 1) {
                                error("ir.pra.invalid-target",
                                      describe(row.element) + " observes the top of " + describe(tree_row.element) + ", which has " +
                                          std::to_string(tree_row.top_gates.count) + " top gates; a fault_tree_top target needs exactly one");
                            }
                        }
                        break;
                    }
                    case domain_target_kind::gate:
                        row.subject = resolve(subject, bit(element_kind::gate), context);
                        break;
                    case domain_target_kind::basic_event:
                        row.subject = resolve(subject, bit(element_kind::basic_event), context);
                        break;
                    case domain_target_kind::sequence:
                    case domain_target_kind::end_state:
                        error("ir.pra.unsupported-target-kind", describe(row.element) + " has kind " +
                                                                    std::string(target_kind_name(row.kind)) +
                                                                    ", which is reserved until event trees are modeled");
                        break;
                    }
                }
            }

            [[nodiscard]] auto operands_of(const std::uint32_t gate) const -> std::span<gate_operand const> {
                return detail::slice(storage_.gate_operands, storage_.gates[gate].operands);
            }

            // Kahn's algorithm over gate -> gate operand edges. Unresolved operands
            // were already reported and are skipped.
            void order_gates() {
                const auto gate_count = static_cast<std::uint32_t>(storage_.gates.size());
                std::vector<std::uint32_t> remaining(gate_count, 0);
                std::vector<std::uint32_t> offsets(gate_count + 1, 0);
                for (std::uint32_t gate = 0; gate < gate_count; ++gate) {
                    for (auto const &operand : operands_of(gate)) {
                        if (const auto child = operand.event.as<gate_id>(); child.valid()) {
                            ++remaining[gate];
                            ++offsets[child.raw() + 1];
                        }
                    }
                }
                std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
                auto cursor = offsets;
                std::vector<std::uint32_t> users(offsets.back());
                for (std::uint32_t gate = 0; gate < gate_count; ++gate) {
                    for (auto const &operand : operands_of(gate)) {
                        if (const auto child = operand.event.as<gate_id>(); child.valid()) {
                            users[cursor[child.raw()]++] = gate;
                        }
                    }
                }

                auto &order = storage_.topological_gates;
                order.reserve(gate_count);
                for (std::uint32_t gate = 0; gate < gate_count; ++gate) {
                    if (remaining[gate] == 0) {
                        order.emplace_back(gate);
                    }
                }
                for (std::size_t next = 0; next < order.size(); ++next) {
                    const auto gate = order[next].raw();
                    for (auto index = offsets[gate]; index < offsets[gate + 1]; ++index) {
                        if (--remaining[users[index]] == 0) {
                            order.emplace_back(users[index]);
                        }
                    }
                }
                if (order.size() != gate_count) {
                    report_cycles(remaining);
                }
            }

            // Every gate Kahn could not order has an unordered gate operand, so
            // following such operands from an unordered gate must close a cycle.
            // Each gate is walked at most once.
            void report_cycles(std::vector<std::uint32_t> const &remaining) {
                enum class walk : std::uint8_t { unvisited, on_path, done };
                const auto gate_count = static_cast<std::uint32_t>(storage_.gates.size());
                std::vector<walk> state(gate_count, walk::unvisited);
                std::vector<std::uint32_t> position(gate_count, 0);
                std::vector<std::uint32_t> path;
                for (std::uint32_t start = 0; start < gate_count; ++start) {
                    if (remaining[start] == 0 || state[start] != walk::unvisited) {
                        continue;
                    }
                    path.clear();
                    auto current = start;
                    while (state[current] == walk::unvisited) {
                        state[current] = walk::on_path;
                        position[current] = static_cast<std::uint32_t>(path.size());
                        path.push_back(current);
                        current = unordered_operand(current, remaining);
                    }
                    if (state[current] == walk::on_path) {
                        report_cycle(std::span<std::uint32_t const>{path}.subspan(position[current]));
                    }
                    for (const auto gate : path) {
                        state[gate] = walk::done;
                    }
                }
            }

            [[nodiscard]] auto unordered_operand(const std::uint32_t gate, std::vector<std::uint32_t> const &remaining) const
                -> std::uint32_t {
                for (auto const &operand : operands_of(gate)) {
                    if (const auto child = operand.event.as<gate_id>(); child.valid() && remaining[child.raw()] != 0) {
                        return child.raw();
                    }
                }
                return gate;
            }

            void report_cycle(const std::span<std::uint32_t const> cycle) {
                const auto shown = std::min(cycle.size(), cycle_names_shown);
                std::string names;
                for (std::size_t index = 0; index < shown; ++index) {
                    names += short_name(storage_.gates[cycle[index]].element);
                    names += " -> ";
                }
                if (cycle.size() > shown) {
                    names += "... -> ";
                }
                names += short_name(storage_.gates[cycle.front()].element);
                error("ir.pra.cycle", describe(storage_.gates[cycle.front()].element) + " is on a gate dependency cycle of length " +
                                          std::to_string(cycle.size()) + ": " + names);
            }

            detail::model_storage &storage_;
            detail::builder_state const &state_;
            api::diagnostic_list &diagnostics_;
        };

        [[nodiscard]] auto summarize(detail::model_storage const &storage) -> model_summary {
            model_summary summary;
            summary.gate_count = storage.gates.size();
            summary.operand_count = storage.gate_operands.size();
            for (auto const &operand : storage.gate_operands) {
                summary.complemented_operand_count += operand.complement ? 1 : 0;
            }
            for (auto const &row : storage.gates) {
                add_unique(summary.gate_kinds_present, row.kind);
            }
            for (auto const &row : storage.expressions) {
                add_unique(summary.expression_kinds_present, row.kind);
                summary.has_frequency_expressions = summary.has_frequency_expressions || row.quantity == quantity_kind::frequency;
            }

            std::vector<std::uint32_t> depth(storage.gates.size(), 0);
            for (const auto gate : storage.topological_gates) {
                std::uint32_t below = 0;
                for (auto const &operand : detail::slice(storage.gate_operands, storage.gates[gate.raw()].operands)) {
                    if (const auto child = operand.event.as<gate_id>(); child.valid()) {
                        below = std::max(below, depth[child.raw()]);
                    }
                }
                depth[gate.raw()] = below + 1;
                summary.gate_depth = std::max(summary.gate_depth, below + 1);
            }
            return summary;
        }

    } // namespace

    void builder::reserve_gates(const std::uint32_t gates, const std::uint32_t operands) {
        storage_.elements.reserve(storage_.elements.size() + gates);
        storage_.gates.reserve(gates);
        storage_.gate_operands.reserve(operands);
        state_.operands.reserve(operands);
    }

    void builder::reserve_basic_events(const std::uint32_t count) {
        storage_.elements.reserve(storage_.elements.size() + count);
        storage_.basic_events.reserve(count);
        storage_.expressions.reserve(count);
    }

    void builder::set_project(const std::string_view name, const std::string_view label) {
        storage_.project.name = intern(storage_.text, name);
        storage_.project.label = intern(storage_.text, label);
    }

    void builder::add_project_attribute(const std::string_view name, const std::string_view value) {
        if (next_row(storage_.attributes.size() + state_.attributes.size()) == invalid_row) {
            return;
        }
        storage_.attributes.push_back(attribute{.name = intern(storage_.text, name), .value = intern(storage_.text, value)});
    }

    auto builder::add_source_file(source_file_spec const &file) -> source_file_id {
        const auto row = next_row(storage_.source_files.size());
        if (row == invalid_row) {
            return {};
        }
        storage_.source_files.push_back(source_file{
            .path = intern(storage_.text, file.path),
            .sha256 = intern(storage_.text, file.sha256),
            .format = intern(storage_.text, file.format),
        });
        return source_file_id(row);
    }

    auto builder::add_expression(expression_spec const &expression) -> expression_id {
        const auto row = next_row(storage_.expressions.size());
        if (row == invalid_row) {
            return {};
        }
        storage_.expressions.push_back(pra::expression{
            .kind = expression.kind,
            .quantity = expression.quantity,
            .value = expression.value,
            .parameter_name = intern(storage_.text, expression.parameter_name),
            .diagnostic = intern(storage_.text, expression.diagnostic),
            .detail = intern(storage_.text, expression.detail),
        });
        return expression_id(row);
    }

    auto builder::add_basic_event(element_spec const &element, expression_spec const &value) -> basic_event_id {
        return add_basic_event(element, add_expression(value));
    }

    auto builder::add_basic_event(element_spec const &element, const expression_id value) -> basic_event_id {
        const auto row = next_row(storage_.basic_events.size());
        if (row == invalid_row) {
            return {};
        }
        const basic_event_id id(row);
        storage_.basic_events.push_back(basic_event{.element = add_element(id, element), .expression = value});
        return id;
    }

    auto builder::add_house_event(element_spec const &element, const bool value) -> house_event_id {
        const auto row = next_row(storage_.house_events.size());
        if (row == invalid_row) {
            return {};
        }
        const house_event_id id(row);
        storage_.house_events.push_back(house_event{.element = add_element(id, element), .value = value});
        return id;
    }

    auto builder::add_initiating_event(element_spec const &element, const reference event) -> initiating_event_id {
        const auto row = next_row(storage_.initiating_events.size());
        if (row == invalid_row) {
            return {};
        }
        const initiating_event_id id(row);
        storage_.initiating_events.push_back(initiating_event{.element = add_element(id, element)});
        state_.initiators.push_back(pending(event));
        return id;
    }

    auto builder::add_fault_tree(element_spec const &element) -> fault_tree_id {
        const auto row = next_row(storage_.fault_trees.size());
        if (row == invalid_row) {
            return {};
        }
        const fault_tree_id id(row);
        storage_.fault_trees.push_back(fault_tree{.element = add_element(id, element)});
        return id;
    }

    void builder::add_fault_tree_top(const fault_tree_id tree, const reference gate) {
        if (next_row(state_.tops.size()) == invalid_row) {
            return;
        }
        state_.tops.push_back(detail::pending_top{.tree = tree, .gate = pending(gate)});
    }

    auto builder::add_gate(element_spec const &element, const gate_kind kind, const std::span<operand_spec const> operands,
                           const fault_tree_id tree) -> gate_id {
        return push_gate(element, kind, cardinality_bounds{}, operands, tree);
    }

    auto builder::add_cardinality(element_spec const &element, const cardinality_bounds bounds,
                                  const std::span<operand_spec const> operands, const fault_tree_id tree) -> gate_id {
        return push_gate(element, gate_kind::cardinality, bounds, operands, tree);
    }

    auto builder::add_target(element_spec const &element, const domain_target_kind kind, const reference subject,
                             const target_role role) -> domain_target_id {
        const auto row = next_row(storage_.targets.size());
        if (row == invalid_row) {
            return {};
        }
        const domain_target_id id(row);
        storage_.targets.push_back(domain_target{.element = add_element(id, element), .kind = kind, .role = role});
        state_.subjects.push_back(pending(subject));
        return id;
    }

    void builder::add_attribute(const domain_ref owner, const std::string_view name, const std::string_view value) {
        const auto element = detail::element_of(storage_, owner);
        if (!element.valid()) {
            diagnostics_.push(api::make_error("ir.pra.dangling-reference", "attribute " + quoted(name) + " is attached to nonexistent " +
                                                                               std::string(to_string(owner.kind())) + " #" +
                                                                               std::to_string(owner.raw())));
            return;
        }
        push_attribute(element, name, value);
    }

    void builder::add_source(const domain_ref owner, const source_location location) {
        const auto element = detail::element_of(storage_, owner);
        if (!element.valid()) {
            diagnostics_.push(api::make_error("ir.pra.dangling-reference", "a source location is attached to nonexistent " +
                                                                               std::string(to_string(owner.kind())) + " #" +
                                                                               std::to_string(owner.raw())));
            return;
        }
        push_source(element, location);
    }

    void builder::push_attribute(const element_id owner, const std::string_view name, const std::string_view value) {
        if (next_row(storage_.attributes.size() + state_.attributes.size()) == invalid_row) {
            return;
        }
        state_.attributes.push_back(
            detail::pending_attribute{.owner = owner, .name = intern(storage_.text, name), .value = intern(storage_.text, value)});
    }

    void builder::push_source(const element_id owner, const source_location location) {
        if (next_row(state_.sources.size()) == invalid_row) {
            return;
        }
        state_.sources.push_back(detail::pending_source{.owner = owner, .location = location});
    }

    auto builder::seal() && -> finalize_result {
        auto diagnostics = std::move(diagnostics_);
        if (capacity_exceeded_) {
            diagnostics.push(api::make_error("ir.pra.capacity-exceeded",
                                             "PRA model exceeds the R1 limit of 2^32 - 1 rows per table or bytes of text"));
            return finalize_result{.output = {}, .diagnostics = std::move(diagnostics)};
        }

        auto storage = std::make_shared<detail::model_storage>(std::move(storage_));
        sealer(*storage, state_, diagnostics).run();
        if (diagnostics.has_errors()) {
            return finalize_result{.output = {}, .diagnostics = std::move(diagnostics)};
        }
        auto summary = summarize(*storage);
        return finalize_result{
            .output = model(std::move(storage), std::move(summary)),
            .diagnostics = std::move(diagnostics),
        };
    }

    auto builder::next_row(const std::size_t size) -> std::uint32_t {
        if (size >= row_limit) {
            capacity_exceeded_ = true;
            return invalid_row;
        }
        return static_cast<std::uint32_t>(size);
    }

    auto builder::intern(std::string &pool, const std::string_view text) -> text_ref {
        if (text.empty()) {
            return {};
        }
        if (text.size() > row_limit || pool.size() > row_limit - text.size()) {
            capacity_exceeded_ = true;
            return {};
        }
        const text_ref ref{.offset = static_cast<std::uint32_t>(pool.size()), .size = static_cast<std::uint32_t>(text.size())};
        pool.append(text);
        return ref;
    }

    auto builder::add_element(const domain_ref owner, element_spec const &element) -> element_id {
        const auto row = next_row(storage_.elements.size());
        if (row == invalid_row) {
            return {};
        }
        const element_id id(row);
        storage_.elements.push_back(element_metadata{
            .owner = owner,
            .stable_name = intern(storage_.text, element.stable_name),
            .qualified_name = intern(storage_.text, element.qualified_name),
            .label = intern(storage_.text, element.label),
        });
        for (auto const &[name, value] : element.attributes) {
            push_attribute(id, name, value);
        }
        for (auto const &location : element.sources) {
            push_source(id, location);
        }
        return id;
    }

    auto builder::pending(reference const &ref) -> detail::pending_reference {
        if (!ref.is_named()) {
            return detail::pending_reference{.target = ref.target(), .mode = detail::reference_mode::id};
        }
        return detail::pending_reference{
            .target = ref.target(),
            .name = intern(state_.names, ref.name()),
            .mode = ref.is_qualified() ? detail::reference_mode::qualified : detail::reference_mode::named,
        };
    }

    auto builder::push_gate(element_spec const &element, const gate_kind kind, const cardinality_bounds bounds,
                            const std::span<operand_spec const> operands, const fault_tree_id tree) -> gate_id {
        const auto row = next_row(storage_.gates.size());
        if (row == invalid_row || next_row(storage_.gate_operands.size() + operands.size()) == invalid_row) {
            return {};
        }
        const gate_id id(row);
        const auto offset = static_cast<std::uint32_t>(storage_.gate_operands.size());
        for (auto const &operand : operands) {
            storage_.gate_operands.push_back(gate_operand{.complement = operand.complement});
            state_.operands.push_back(pending(operand.event));
        }
        storage_.gates.push_back(gate{
            .element = add_element(id, element),
            .kind = kind,
            .operands = table_range<gate_operand>{.offset = offset, .count = static_cast<std::uint32_t>(operands.size())},
            .bounds = kind == gate_kind::cardinality ? bounds : cardinality_bounds{},
            .fault_tree = tree,
        });
        return id;
    }

} // namespace fuzzybits::ir::pra
