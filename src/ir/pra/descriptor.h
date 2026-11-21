#pragma once

#include "api/capabilities.h"
#include "api/ids.h"
#include "ir/descriptor.h"
#include "ir/form_profile.h"
#include "ir/forms.h"
#include "ir/pra/expression.h"
#include "ir/pra/gate.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fuzzybits::ir::pra {

    struct model_summary final {
        std::size_t gate_count = 0;
        std::size_t operand_count = 0;
        std::size_t complemented_operand_count = 0;
        // Longest chain of gates; a gate whose operands are all events has depth 1.
        std::uint32_t gate_depth = 0;
        std::vector<gate_kind> gate_kinds_present;
        std::vector<expression_kind> expression_kinds_present;
        bool has_frequency_expressions = false;
    };

    [[nodiscard]] inline auto storage_type_id() -> api::type_id {
        return api::type_id("ir.pra.model");
    }
    [[nodiscard]] inline auto endpoint_id() -> api::endpoint_id {
        return api::endpoint_id("ir.pra");
    }
    [[nodiscard]] inline auto model_form() -> api::form_id {
        return api::form_id("ir.pra.model");
    }

    [[nodiscard]] inline auto make_descriptor() -> ir::descriptor {
        auto form_profile = ir::form_profile::from_exact_form(model_form());
        form_profile.add_satisfied_form(forms::probabilistic_risk_model());

        return ir::descriptor{
            .provider = api::provider_id("fuzzybits"),
            .endpoint = endpoint_id(),
            .storage_type = storage_type_id(),
            .form = model_form(),
            .name = "PRA model",
            .capabilities =
                api::capability_set{
                    api::capability_id("ir.pra.model"),
                    api::capability_id("ir.pra.fault_trees"),
                    api::capability_id("ir.pra.domain_targets"),
                    api::capability_id("ir.pra.source_locations"),
                },
            .satisfied_forms = form_profile.satisfied_form_ids(),
        };
    }

} // namespace fuzzybits::ir::pra
