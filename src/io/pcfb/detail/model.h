#pragma once

#include "api/result.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/op_code.h"

#include <cstdint>
#include <span>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::pcfb::detail {

    // Operand columns of a PC circuit. emission_order diagnoses cycles and invalid
    // endpoints itself instead of relying on what seal() already checked.
    struct graph_view final {
        std::span<ir::pc::op_code const> op_codes;
        std::span<std::uint32_t const> operand_offsets;
        std::span<std::uint32_t const> operand_counts;
        std::span<ir::pc::edge_ref const> edges;

        [[nodiscard]] static auto of(ir::pc::circuit const &model) -> graph_view {
            return graph_view{
                .op_codes = model.op_codes(),
                .operand_offsets = model.operand_offsets(),
                .operand_counts = model.operand_counts(),
                .edges = model.edges(),
            };
        }
    };

    // Depth-first post-order with roots taken in ascending node id and operands in
    // operand order: every node follows all of its operands, and a circuit whose
    // ids are already topological is emitted in id order. Iterative, O(V + E).
    [[nodiscard]] auto emission_order(graph_view const &graph) -> api::result<std::vector<std::uint32_t>>;

    [[nodiscard]] auto write_pc(ir::pc::circuit const &model, write_options const &options) -> pcfb_write_result;

    [[nodiscard]] auto read_pcfb(std::span<std::byte const> bytes) -> pcfb_read_result;

    [[nodiscard]] auto parse_sha256_hex(std::string const &text) -> std::optional<std::array<std::byte, 32>>;

} // namespace fuzzybits::io::backend::fuzzybits::pcfb::detail
