#pragma once

#include "api/diagnostics.h"
#include "ir/pra/event.h"
#include "ir/pra/expression.h"
#include "ir/pra/fault_tree.h"
#include "ir/pra/gate.h"
#include "ir/pra/id.h"
#include "ir/pra/metadata.h"
#include "ir/pra/model.h"
#include "ir/pra/target.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fuzzybits::ir::pra {

    // Construction inputs. The builder copies every string, so views only need to
    // outlive the call that receives them.

    struct attribute_spec final {
        std::string_view name;
        std::string_view value;
    };

    struct element_spec final {
        std::string_view stable_name;
        std::string_view qualified_name;
        std::string_view label;
        std::span<attribute_spec const> attributes;
        std::span<source_location const> sources;
    };

    struct source_file_spec final {
        std::string_view path;
        std::string_view sha256;
        std::string_view format;
    };

    struct expression_spec final {
        expression_kind kind = expression_kind::unsupported;
        quantity_kind quantity = quantity_kind::probability;
        double value = 0.0;
        std::string_view parameter_name;
        std::string_view diagnostic;
        std::string_view detail;

        [[nodiscard]] static constexpr auto literal(const double value,
                                                    const quantity_kind quantity = quantity_kind::probability) noexcept
            -> expression_spec {
            return expression_spec{.kind = expression_kind::literal, .quantity = quantity, .value = value};
        }

        [[nodiscard]] static constexpr auto parameter(const std::string_view name,
                                                      const quantity_kind quantity = quantity_kind::probability) noexcept
            -> expression_spec {
            return expression_spec{.kind = expression_kind::parameter, .quantity = quantity, .parameter_name = name};
        }

        [[nodiscard]] static constexpr auto unsupported(const std::string_view diagnostic, const std::string_view detail = {},
                                                        const quantity_kind quantity = quantity_kind::probability) noexcept
            -> expression_spec {
            return expression_spec{
                .kind = expression_kind::unsupported, .quantity = quantity, .diagnostic = diagnostic, .detail = detail};
        }
    };

    // A reference to an element: an id returned by the builder, or a stable name
    // resolved when the builder seals, so it may name an element added later.
    // An unqualified name resolves against every kind admissible where it is used
    // (gate operands: gates, basic events and house events) and must match exactly
    // one of them.
    class reference final {
    public:
        constexpr reference() = default;

        constexpr reference(const domain_ref target) noexcept : target_(target) {} // NOLINT(google-explicit-constructor)

        template <element_tag tag_type>
        constexpr reference(const id<tag_type> target) noexcept : target_(target) {} // NOLINT(google-explicit-constructor)

        [[nodiscard]] static constexpr auto named(const std::string_view name) noexcept -> reference {
            reference out;
            out.name_ = name;
            out.named_ = true;
            return out;
        }

        [[nodiscard]] static constexpr auto named(const element_kind kind, const std::string_view name) noexcept -> reference {
            reference out;
            out.target_ = domain_ref(kind, std::numeric_limits<std::uint32_t>::max());
            out.name_ = name;
            out.named_ = true;
            out.qualified_ = true;
            return out;
        }

        [[nodiscard]] constexpr auto is_named() const noexcept -> bool {
            return named_;
        }
        [[nodiscard]] constexpr auto is_qualified() const noexcept -> bool {
            return qualified_;
        }
        [[nodiscard]] constexpr auto name() const noexcept -> std::string_view {
            return name_;
        }
        // The referenced element, or for a qualified name an invalid ref of the
        // required kind.
        [[nodiscard]] constexpr auto target() const noexcept -> domain_ref {
            return target_;
        }

    private:
        domain_ref target_;
        std::string_view name_;
        bool named_ = false;
        bool qualified_ = false;
    };

    struct operand_spec final {
        reference event;
        bool complement = false;
    };

    struct finalize_result final {
        model output;
        api::diagnostic_list diagnostics;

        [[nodiscard]] auto ok() const -> bool {
            return !diagnostics.has_errors() && output.valid();
        }
    };

    namespace detail {

        enum class reference_mode : std::uint8_t {
            id,
            named,
            qualified,
        };

        struct pending_reference final {
            domain_ref target;
            text_ref name;
            reference_mode mode = reference_mode::id;
        };

        struct pending_top final {
            fault_tree_id tree;
            pending_reference gate;
        };

        struct pending_attribute final {
            element_id owner;
            text_ref name;
            text_ref value;
        };

        struct pending_source final {
            element_id owner;
            source_location location;
        };

        // Everything the builder needs only until it seals.
        struct builder_state final {
            // Names of pending references; not part of the model's text.
            std::string names;
            std::vector<pending_reference> operands;
            std::vector<pending_reference> initiators;
            std::vector<pending_reference> subjects;
            std::vector<pending_top> tops;
            std::vector<pending_attribute> attributes;
            std::vector<pending_source> sources;
        };

    } // namespace detail

    // The only mutable construction surface. It assigns dense ids in insertion
    // order, defers name resolution and validation to seal(), and seals into an
    // immutable model. seal() reports every error it finds; the model is
    // returned only when there are none.
    class builder final {
    public:
        builder() = default;

        void reserve_gates(std::uint32_t gates, std::uint32_t operands);
        void reserve_basic_events(std::uint32_t count);

        void set_project(std::string_view name, std::string_view label = {});
        void add_project_attribute(std::string_view name, std::string_view value);
        [[nodiscard]] auto add_source_file(source_file_spec const &file) -> source_file_id;

        [[nodiscard]] auto add_expression(expression_spec const &expression) -> expression_id;
        [[nodiscard]] auto add_basic_event(element_spec const &element, expression_spec const &value) -> basic_event_id;
        [[nodiscard]] auto add_basic_event(element_spec const &element, expression_id value) -> basic_event_id;
        [[nodiscard]] auto add_house_event(element_spec const &element, bool value) -> house_event_id;
        [[nodiscard]] auto add_initiating_event(element_spec const &element, reference event) -> initiating_event_id;

        [[nodiscard]] auto add_fault_tree(element_spec const &element) -> fault_tree_id;
        void add_fault_tree_top(fault_tree_id tree, reference gate);

        [[nodiscard]] auto add_gate(element_spec const &element, gate_kind kind, std::span<operand_spec const> operands,
                                    fault_tree_id tree = {}) -> gate_id;

        [[nodiscard]] auto add_gate(element_spec const &element, const gate_kind kind,
                                    const std::initializer_list<operand_spec> operands, const fault_tree_id tree = {})
            -> gate_id {
            return add_gate(element, kind, std::span<operand_spec const>{operands.begin(), operands.size()}, tree);
        }

        [[nodiscard]] auto add_cardinality(element_spec const &element, cardinality_bounds bounds,
                                           std::span<operand_spec const> operands, fault_tree_id tree = {}) -> gate_id;

        [[nodiscard]] auto add_cardinality(element_spec const &element, const cardinality_bounds bounds,
                                           const std::initializer_list<operand_spec> operands, const fault_tree_id tree = {})
            -> gate_id {
            return add_cardinality(element, bounds, std::span<operand_spec const>{operands.begin(), operands.size()}, tree);
        }

        [[nodiscard]] auto add_target(element_spec const &element, domain_target_kind kind, reference subject,
                                      target_role role = target_role::primary) -> domain_target_id;

        // Attach metadata to an element after it was added.
        void add_attribute(domain_ref owner, std::string_view name, std::string_view value);
        void add_source(domain_ref owner, source_location location);

        [[nodiscard]] auto seal() && -> finalize_result;

    private:
        [[nodiscard]] auto next_row(std::size_t size) -> std::uint32_t;
        [[nodiscard]] auto intern(std::string &pool, std::string_view text) -> text_ref;
        [[nodiscard]] auto add_element(domain_ref owner, element_spec const &element) -> element_id;
        void push_attribute(element_id owner, std::string_view name, std::string_view value);
        void push_source(element_id owner, source_location location);
        [[nodiscard]] auto pending(reference const &ref) -> detail::pending_reference;
        [[nodiscard]] auto push_gate(element_spec const &element, gate_kind kind, cardinality_bounds bounds,
                                     std::span<operand_spec const> operands, fault_tree_id tree) -> gate_id;

        detail::model_storage storage_;
        detail::builder_state state_;
        api::diagnostic_list diagnostics_;
        bool capacity_exceeded_ = false;
    };

} // namespace fuzzybits::ir::pra
