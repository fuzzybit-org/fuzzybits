#pragma once

#include "ir/descriptor.h"
#include "ir/handle.h"
#include "ir/pra/descriptor.h"
#include "ir/pra/event.h"
#include "ir/pra/expression.h"
#include "ir/pra/fault_tree.h"
#include "ir/pra/gate.h"
#include "ir/pra/id.h"
#include "ir/pra/metadata.h"
#include "ir/pra/target.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pra {

    class builder;

    namespace detail {

        struct model_storage final {
            project_metadata project;
            std::string text;
            std::vector<element_metadata> elements;
            std::vector<attribute> attributes;
            std::vector<source_file> source_files;
            std::vector<source_location> source_locations;
            std::vector<expression> expressions;
            std::vector<basic_event> basic_events;
            std::vector<house_event> house_events;
            std::vector<initiating_event> initiating_events;
            std::vector<gate> gates;
            std::vector<gate_operand> gate_operands;
            std::vector<fault_tree> fault_trees;
            std::vector<gate_id> fault_tree_gates;
            std::vector<gate_id> fault_tree_top_gates;
            std::vector<domain_target> targets;
            std::vector<gate_id> topological_gates;
            // Open-addressing hash index over every named element (all kinds);
            // empty or a power of two in size, at most half full.
            std::vector<element_id> name_slots;
        };

        template <typename row_type>
        [[nodiscard]] auto slice(std::vector<row_type> const &column, const table_range<row_type> range) noexcept
            -> std::span<row_type const> {
            return std::span<row_type const>{column}.subspan(range.offset, range.count);
        }

        [[nodiscard]] inline auto row_count(model_storage const &storage, const element_kind kind) noexcept -> std::size_t {
            switch (kind) {
            case element_kind::basic_event:
                return storage.basic_events.size();
            case element_kind::house_event:
                return storage.house_events.size();
            case element_kind::initiating_event:
                return storage.initiating_events.size();
            case element_kind::gate:
                return storage.gates.size();
            case element_kind::fault_tree:
                return storage.fault_trees.size();
            case element_kind::domain_target:
                return storage.targets.size();
            }
            return 0;
        }

        [[nodiscard]] inline auto exists(model_storage const &storage, const domain_ref ref) noexcept -> bool {
            return ref.valid() && ref.raw() < row_count(storage, ref.kind());
        }

        [[nodiscard]] inline auto element_of(model_storage const &storage, const domain_ref ref) noexcept -> element_id {
            if (!exists(storage, ref)) {
                return element_id::invalid();
            }
            switch (ref.kind()) {
            case element_kind::basic_event:
                return storage.basic_events[ref.raw()].element;
            case element_kind::house_event:
                return storage.house_events[ref.raw()].element;
            case element_kind::initiating_event:
                return storage.initiating_events[ref.raw()].element;
            case element_kind::gate:
                return storage.gates[ref.raw()].element;
            case element_kind::fault_tree:
                return storage.fault_trees[ref.raw()].element;
            case element_kind::domain_target:
                return storage.targets[ref.raw()].element;
            }
            return element_id::invalid();
        }

        [[nodiscard]] inline auto text_of(model_storage const &storage, const text_ref ref) noexcept -> std::string_view {
            const std::string_view text{storage.text};
            if (ref.offset > text.size() || ref.size > text.size() - ref.offset) {
                return {};
            }
            return text.substr(ref.offset, ref.size);
        }

        [[nodiscard]] inline auto name_hash(const std::string_view name) noexcept -> std::size_t {
            return std::hash<std::string_view>{}(name);
        }

        // Elements with equal names share a probe chain in insertion order, so the
        // first match is the earliest element with that kind and name.
        [[nodiscard]] inline auto find_element(model_storage const &storage, const element_kind kind,
                                               const std::string_view name) noexcept -> element_id {
            auto const &slots = storage.name_slots;
            if (slots.empty()) {
                return element_id::invalid();
            }
            const auto mask = slots.size() - 1;
            for (auto slot = name_hash(name) & mask;; slot = (slot + 1) & mask) {
                const auto element = slots[slot];
                if (!element.valid()) {
                    return element_id::invalid();
                }
                auto const &row = storage.elements[element.raw()];
                if (row.owner.kind() == kind && text_of(storage, row.stable_name) == name) {
                    return element;
                }
            }
        }

    } // namespace detail

    // Immutable after sealing; copies share storage. Every table is a dense
    // column indexed by its id type, and every per-entity list is a
    // table_range slice of a shared column.
    class model final {
    public:
        model() = default;

        [[nodiscard]] auto valid() const noexcept -> bool {
            return storage_ != nullptr;
        }
        [[nodiscard]] explicit operator bool() const noexcept {
            return valid();
        }
        [[nodiscard]] auto descriptor() const noexcept -> ir::descriptor const & {
            return descriptor_;
        }
        [[nodiscard]] auto summary() const noexcept -> model_summary const & {
            return summary_;
        }
        [[nodiscard]] auto project() const noexcept -> project_metadata {
            return storage_ == nullptr ? project_metadata{} : storage_->project;
        }

        [[nodiscard]] auto text(const text_ref ref) const noexcept -> std::string_view {
            return storage_ == nullptr ? std::string_view{} : detail::text_of(*storage_, ref);
        }

        [[nodiscard]] auto elements() const noexcept -> std::span<element_metadata const> {
            return column(&detail::model_storage::elements);
        }
        [[nodiscard]] auto attributes() const noexcept -> std::span<attribute const> {
            return column(&detail::model_storage::attributes);
        }
        [[nodiscard]] auto source_files() const noexcept -> std::span<source_file const> {
            return column(&detail::model_storage::source_files);
        }
        [[nodiscard]] auto source_locations() const noexcept -> std::span<source_location const> {
            return column(&detail::model_storage::source_locations);
        }
        [[nodiscard]] auto expressions() const noexcept -> std::span<expression const> {
            return column(&detail::model_storage::expressions);
        }
        [[nodiscard]] auto basic_events() const noexcept -> std::span<basic_event const> {
            return column(&detail::model_storage::basic_events);
        }
        [[nodiscard]] auto house_events() const noexcept -> std::span<house_event const> {
            return column(&detail::model_storage::house_events);
        }
        [[nodiscard]] auto initiating_events() const noexcept -> std::span<initiating_event const> {
            return column(&detail::model_storage::initiating_events);
        }
        [[nodiscard]] auto gates() const noexcept -> std::span<gate const> {
            return column(&detail::model_storage::gates);
        }
        [[nodiscard]] auto gate_operands() const noexcept -> std::span<gate_operand const> {
            return column(&detail::model_storage::gate_operands);
        }
        [[nodiscard]] auto fault_trees() const noexcept -> std::span<fault_tree const> {
            return column(&detail::model_storage::fault_trees);
        }
        [[nodiscard]] auto fault_tree_gates() const noexcept -> std::span<gate_id const> {
            return column(&detail::model_storage::fault_tree_gates);
        }
        [[nodiscard]] auto fault_tree_top_gates() const noexcept -> std::span<gate_id const> {
            return column(&detail::model_storage::fault_tree_top_gates);
        }
        [[nodiscard]] auto targets() const noexcept -> std::span<domain_target const> {
            return column(&detail::model_storage::targets);
        }

        // Every gate appears after all gates among its operands.
        [[nodiscard]] auto topological_gates() const noexcept -> std::span<gate_id const> {
            return column(&detail::model_storage::topological_gates);
        }

        [[nodiscard]] auto operands(const gate_id gate) const noexcept -> std::span<gate_operand const> {
            if (storage_ == nullptr || !gate.valid() || gate.raw() >= storage_->gates.size()) {
                return {};
            }
            return detail::slice(storage_->gate_operands, storage_->gates[gate.raw()].operands);
        }

        [[nodiscard]] auto attributes(const element_id element) const noexcept -> std::span<attribute const> {
            if (storage_ == nullptr || !element.valid() || element.raw() >= storage_->elements.size()) {
                return {};
            }
            return detail::slice(storage_->attributes, storage_->elements[element.raw()].attributes);
        }

        [[nodiscard]] auto project_attributes() const noexcept -> std::span<attribute const> {
            return storage_ == nullptr ? std::span<attribute const>{} : detail::slice(storage_->attributes, storage_->project.attributes);
        }

        [[nodiscard]] auto sources(const element_id element) const noexcept -> std::span<source_location const> {
            if (storage_ == nullptr || !element.valid() || element.raw() >= storage_->elements.size()) {
                return {};
            }
            return detail::slice(storage_->source_locations, storage_->elements[element.raw()].sources);
        }

        [[nodiscard]] auto gates(const fault_tree_id tree) const noexcept -> std::span<gate_id const> {
            if (storage_ == nullptr || !tree.valid() || tree.raw() >= storage_->fault_trees.size()) {
                return {};
            }
            return detail::slice(storage_->fault_tree_gates, storage_->fault_trees[tree.raw()].gates);
        }

        [[nodiscard]] auto top_gates(const fault_tree_id tree) const noexcept -> std::span<gate_id const> {
            if (storage_ == nullptr || !tree.valid() || tree.raw() >= storage_->fault_trees.size()) {
                return {};
            }
            return detail::slice(storage_->fault_tree_top_gates, storage_->fault_trees[tree.raw()].top_gates);
        }

        [[nodiscard]] auto element_of(const domain_ref ref) const noexcept -> element_id {
            return storage_ == nullptr ? element_id::invalid() : detail::element_of(*storage_, ref);
        }

        [[nodiscard]] auto stable_name(const element_id element) const noexcept -> std::string_view {
            if (storage_ == nullptr || !element.valid() || element.raw() >= storage_->elements.size()) {
                return {};
            }
            return detail::text_of(*storage_, storage_->elements[element.raw()].stable_name);
        }

        [[nodiscard]] auto stable_name(const domain_ref ref) const noexcept -> std::string_view {
            return stable_name(element_of(ref));
        }

        // Exact, case-sensitive lookup of a stable name within one kind; invalid
        // when absent. Expected O(1).
        [[nodiscard]] auto find(const element_kind kind, const std::string_view name) const -> domain_ref {
            if (storage_ == nullptr) {
                return {};
            }
            const auto element = detail::find_element(*storage_, kind, name);
            return element.valid() ? storage_->elements[element.raw()].owner : domain_ref{};
        }

        [[nodiscard]] auto find_basic_event(const std::string_view name) const -> basic_event_id {
            return find(element_kind::basic_event, name).as<basic_event_id>();
        }
        [[nodiscard]] auto find_house_event(const std::string_view name) const -> house_event_id {
            return find(element_kind::house_event, name).as<house_event_id>();
        }
        [[nodiscard]] auto find_initiating_event(const std::string_view name) const -> initiating_event_id {
            return find(element_kind::initiating_event, name).as<initiating_event_id>();
        }
        [[nodiscard]] auto find_gate(const std::string_view name) const -> gate_id {
            return find(element_kind::gate, name).as<gate_id>();
        }
        [[nodiscard]] auto find_fault_tree(const std::string_view name) const -> fault_tree_id {
            return find(element_kind::fault_tree, name).as<fault_tree_id>();
        }
        [[nodiscard]] auto find_target(const std::string_view name) const -> domain_target_id {
            return find(element_kind::domain_target, name).as<domain_target_id>();
        }

        // The event a target observes: the single top gate of a fault_tree_top
        // target, otherwise the subject itself.
        [[nodiscard]] auto target_event(const domain_target_id target) const noexcept -> domain_ref {
            if (storage_ == nullptr || !target.valid() || target.raw() >= storage_->targets.size()) {
                return {};
            }
            auto const &row = storage_->targets[target.raw()];
            if (row.kind != domain_target_kind::fault_tree_top) {
                return row.subject;
            }
            const auto tops = top_gates(row.subject.as<fault_tree_id>());
            return tops.size() == 1 ? domain_ref(tops.front()) : domain_ref{};
        }

        [[nodiscard]] auto to_handle() const -> ir::handle {
            if (!valid()) {
                return {};
            }
            return ir::handle::make(std::make_shared<model const>(*this), descriptor_);
        }

    private:
        friend class builder;

        model(std::shared_ptr<detail::model_storage const> storage, model_summary summary)
            : storage_(std::move(storage)), summary_(std::move(summary)) {}

        template <typename row_type>
        [[nodiscard]] auto column(std::vector<row_type> detail::model_storage::*member) const noexcept -> std::span<row_type const> {
            return storage_ == nullptr ? std::span<row_type const>{} : std::span<row_type const>{(*storage_).*member};
        }

        std::shared_ptr<detail::model_storage const> storage_;
        model_summary summary_;
        ir::descriptor descriptor_ = make_descriptor();
    };

    [[nodiscard]] inline auto is_pra(ir::handle const &handle) -> bool {
        return handle.valid() && handle.storage_type() == storage_type_id();
    }

    [[nodiscard]] inline auto model_from(ir::handle const &handle) -> model const * {
        return is_pra(handle) ? handle.storage_as<model>() : nullptr;
    }

} // namespace fuzzybits::ir::pra
