#include "io/pcfb/detail/model.h"
#include "io/pcfb/descriptor.h"

#include "api/diagnostics.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"
#include "ir/pc/target.h"

#include "pcfb/builder.h"
#include "pcfb/io.h"
#include "pcfb/validate.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef FUZZYBITS_PCFB_GENERATOR_VERSION
#error "FUZZYBITS_PCFB_GENERATOR_VERSION must be provided by CMake"
#endif

namespace fuzzybits::io::backend::fuzzybits::pcfb::detail {
    namespace {

        constexpr std::uint32_t kNoTensor = std::numeric_limits<std::uint32_t>::max();

        template <typename id_type> [[nodiscard]] auto idText(const id_type id) -> std::string {
            return id.valid() ? std::to_string(id.raw()) : std::string("<invalid>");
        }

        [[nodiscard]] auto nameOf(write_options const &options, const std::uint32_t node) -> std::string_view {
            const auto found = options.node_names.find(ir::pc::bool_node_id(node));
            return found == options.node_names.end() ? std::string_view{} : std::string_view{found->second};
        }

        [[nodiscard]] auto nodeLabel(write_options const &options, const std::uint32_t node) -> std::string {
            auto label = "node " + std::to_string(node);
            if (const auto name = nameOf(options, node); !name.empty()) {
                label += " ('" + std::string(name) + "')";
            }
            return label;
        }

        [[nodiscard]] auto pcfbQueryKind(const query_kind kind) -> ::pcfb::QueryKind {
            switch (kind) {
            case query_kind::probability:
                return ::pcfb::QueryKind::PROBABILITY;
            }
            return ::pcfb::QueryKind::PROBABILITY;
        }

        [[nodiscard]] auto gateOpcode(const ir::pc::op_code code) -> ::pcfb::OpCode {
            switch (code) {
            case ir::pc::op_code::and_gate:
                return ::pcfb::OpCode::BITWISE_AND;
            case ir::pc::op_code::or_gate:
                return ::pcfb::OpCode::BITWISE_OR;
            case ir::pc::op_code::xor_gate:
                return ::pcfb::OpCode::BITWISE_XOR;
            case ir::pc::op_code::not_gate:
                return ::pcfb::OpCode::BITWISE_NOT;
            case ir::pc::op_code::nand_gate:
                return ::pcfb::OpCode::BITWISE_NAND;
            case ir::pc::op_code::nor_gate:
                return ::pcfb::OpCode::BITWISE_NOR;
            case ir::pc::op_code::xnor_gate:
                return ::pcfb::OpCode::BITWISE_XNOR;
            case ir::pc::op_code::basic_event:
            case ir::pc::op_code::constant:
            case ir::pc::op_code::house_event:
            case ir::pc::op_code::cardinality:
                break;
            }
            return ::pcfb::OpCode::BITWISE_NO_OP;
        }

        void checkProbability(ir::pc::circuit const &model, write_options const &options, const std::uint32_t node,
                              api::diagnostic_list &diagnostics) {
            const auto source = model.probability_source_refs()[node];
            if (!source.valid() || source.raw() >= model.probability_sources().size()) {
                diagnostics.push(api::make_error("io.pcfb.invalid-probability-source",
                                                 "basic event " + nodeLabel(options, node) + " references an invalid probability source"));
                return;
            }
            switch (auto const &probability = model.probability_sources()[source.raw()]; probability.kind) {
            case ir::pc::probability_source_kind::literal:
                if (!std::isfinite(probability.literal) || probability.literal < 0.0 || probability.literal > 1.0) {
                    diagnostics.push(api::make_error("io.pcfb.invalid-probability",
                                                     "basic event " + nodeLabel(options, node) + " has a literal probability outside [0, 1]"));
                }
                break;
            case ir::pc::probability_source_kind::parameter:
                diagnostics.push(api::make_error("io.pcfb.parameter-probability",
                                                 "basic event " + nodeLabel(options, node) + " has parameter probability '" +
                                                     probability.parameter_name + "'; pcfb SAMPLE requires a literal probability"));
                break;
            case ir::pc::probability_source_kind::unsupported:
                diagnostics.push(api::make_error("io.pcfb.unsupported-probability",
                                                 "basic event " + nodeLabel(options, node) + " has an unsupported probability source" +
                                                     (probability.diagnostic.empty() ? std::string{} : ": " + probability.diagnostic)));
                break;
            }
        }

        void checkNode(ir::pc::circuit const &model, write_options const &options, const std::uint32_t node,
                       api::diagnostic_list &diagnostics) {
            const auto code = model.op_codes()[node];
            const auto count = model.operand_counts()[node];
            if (ir::pc::is_leaf(code) && count != 0) {
                diagnostics.push(api::make_error("io.pcfb.invalid-arity", "leaf " + nodeLabel(options, node) + " has operands"));
            }
            switch (code) {
            case ir::pc::op_code::basic_event:
                checkProbability(model, options, node, diagnostics);
                break;
            case ir::pc::op_code::constant:
            case ir::pc::op_code::house_event:
                break;
            case ir::pc::op_code::not_gate:
                if (count != 1) {
                    diagnostics.push(
                        api::make_error("io.pcfb.invalid-arity", "not gate " + nodeLabel(options, node) + " must have exactly one operand"));
                }
                break;
            case ir::pc::op_code::and_gate:
            case ir::pc::op_code::or_gate:
            case ir::pc::op_code::xor_gate:
            case ir::pc::op_code::nand_gate:
            case ir::pc::op_code::nor_gate:
            case ir::pc::op_code::xnor_gate:
                if (count == 0) {
                    diagnostics.push(
                        api::make_error("io.pcfb.invalid-arity", "gate " + nodeLabel(options, node) + " must have at least one operand"));
                }
                break;
            case ir::pc::op_code::cardinality:
                if (const auto low = model.cardinality_low()[node], high = model.cardinality_high()[node];
                    count == 0 || low > high || high > count) {
                    diagnostics.push(api::make_error("io.pcfb.invalid-cardinality", "cardinality " + nodeLabel(options, node) + " bounds [" +
                                                                                        std::to_string(low) + ", " + std::to_string(high) +
                                                                                        "] are outside its fan-in " + std::to_string(count)));
                }
                break;
            }
        }

        [[nodiscard]] auto validateSerializable(ir::pc::circuit const &model, write_options const &options) -> api::diagnostic_list {
            api::diagnostic_list diagnostics;
            if (!model.valid()) {
                diagnostics.push(api::make_error("io.pcfb.invalid-ir", "pcfb writer requires a valid PC circuit"));
                return diagnostics;
            }
            if (model.targets().empty()) {
                diagnostics.push(api::make_error("io.pcfb.no-targets", "pcfb core-1.1 requires at least one query; the PC circuit has no targets"));
            }

            const auto nodeCount = model.node_count();
            for (std::uint32_t node = 0; node < nodeCount; ++node) {
                checkNode(model, options, node, diagnostics);
            }
            for (auto const &target : model.targets()) {
                if (!target.node.valid() || target.node.raw() >= nodeCount) {
                    diagnostics.push(
                        api::make_error("io.pcfb.invalid-target", "target " + std::to_string(target.id.raw()) + " references an invalid node"));
                }
            }

            for (auto const &[node, name] : options.node_names) {
                if (!node.valid() || node.raw() >= nodeCount) {
                    diagnostics.push(api::make_error("io.pcfb.invalid-name",
                                                     "node name '" + name + "' refers to node " + idText(node) + ", which is not in the circuit"));
                }
            }
            for (auto const &[target, name] : options.target_names) {
                if (std::ranges::none_of(model.targets(), [&](auto const &view) { return view.id == target; })) {
                    diagnostics.push(api::make_error("io.pcfb.invalid-name", "target name '" + name + "' refers to target " + idText(target) +
                                                                                 ", which is not in the circuit"));
                }
            }
            return diagnostics;
        }

        [[nodiscard]] auto validateOutput(std::span<std::byte const> bytes) -> api::diagnostic_list {
            api::diagnostic_list diagnostics;
            auto circuit = ::pcfb::io::from_bytes(bytes);
            if (!circuit) {
                diagnostics.push(api::make_error("io.pcfb.invalid-output", "written pcfb failed verification: " + circuit.error().message));
                return diagnostics;
            }
            for (auto const &diagnostic : ::pcfb::validate(*circuit)) {
                diagnostics.push(api::make_error("io.pcfb.invalid-output",
                                                 "written pcfb failed validation at " + diagnostic.location + ": " + diagnostic.message));
            }
            return diagnostics;
        }

        class Emitter final {
        public:
            Emitter(ir::pc::circuit const &model, write_options const &options)
                : model_(model), options_(options), tensorByNode_(model.node_count(), kNoTensor), eventIdByNode_(model.node_count(), 0),
                  notTensorByNode_(model.node_count(), kNoTensor), positiveEpoch_(model.node_count(), 0) {}

            [[nodiscard]] auto run(std::vector<std::uint32_t> const &order) && -> pcfb_write_result {
                assignEventIds();
                if (!options_.model_name.empty()) {
                    builder_.name(options_.model_name);
                }
                for (const auto node : order) {
                    if (!emitNode(node)) {
                        return pcfb_write_result::fail(std::move(diagnostics_));
                    }
                }
                if (!emitQueries() || !emitMetadata()) {
                    return pcfb_write_result::fail(std::move(diagnostics_));
                }

                auto bytes = builder_.finish();
                if (!bytes) {
                    diagnostics_.push(api::make_error("io.pcfb.builder", "pcfb::builder::finish failed: " + bytes.error().message));
                    return pcfb_write_result::fail(std::move(diagnostics_));
                }
                if (auto invalid = validateOutput(*bytes); invalid.has_errors()) {
                    diagnostics_.append(std::move(invalid));
                    return pcfb_write_result::fail(std::move(diagnostics_));
                }
                return pcfb_write_result::ok(
                    write_output{
                        .bytes = std::move(*bytes),
                        .tensor_by_node = std::move(tensorByNode_),
                        .event_id_by_node = std::move(eventIdByNode_),
                    },
                    std::move(diagnostics_));
            }

        private:
            void assignEventIds() {
                std::uint32_t next = 0;
                for (std::uint32_t node = 0; node < model_.node_count(); ++node) {
                    if (model_.op_codes()[node] == ir::pc::op_code::basic_event) {
                        eventIdByNode_[node] = ++next;
                    }
                }
            }

            [[nodiscard]] auto accept(::pcfb::Result<std::uint32_t> const &tensor, const std::uint32_t node) -> std::optional<std::uint32_t> {
                if (!tensor) {
                    diagnostics_.push(
                        api::make_error("io.pcfb.builder", "pcfb::builder rejected " + nodeLabel(options_, node) + ": " + tensor.error().message));
                    return std::nullopt;
                }
                return *tensor;
            }

            [[nodiscard]] auto roomForTensor(const std::uint32_t node) -> bool {
                if (tensorCount_ < kNoTensor) {
                    ++tensorCount_;
                    return true;
                }
                diagnostics_.push(
                    api::make_error("io.pcfb.tensor-overflow", "pcfb output for " + nodeLabel(options_, node) + " exceeds the 32-bit tensor range"));
                return false;
            }

            [[nodiscard]] auto addOp(const std::uint32_t node, const ::pcfb::OpCode opcode, std::span<std::uint32_t const> inputs,
                                     std::span<std::uint32_t const> inverted, const std::optional<std::uint32_t> k, std::string_view name)
                -> std::optional<std::uint32_t> {
                if (!roomForTensor(node)) {
                    return std::nullopt;
                }
                return accept(builder_.add_op(opcode, inputs, inverted, k, name), node);
            }

            // pcfb permits one tensor in both `inputs` and `inverted_inputs` only for
            // XOR-family ops (spec §3.1). Every op is written with disjoint lists: a
            // source used in both polarities is read through one NOT tensor per source.
            [[nodiscard]] auto partitionOperands(const std::uint32_t node) -> bool {
                inputs_.clear();
                inverted_.clear();
                ++epoch_;
                const auto operands = model_.operands(ir::pc::bool_node_id(node));
                for (auto const &[source, complement] : operands) {
                    if (!complement) {
                        positiveEpoch_[source.raw()] = epoch_;
                    }
                }
                for (auto const &[source, complement] : operands) {
                    const auto tensor = tensorByNode_[source.raw()];
                    if (!complement) {
                        inputs_.push_back(tensor);
                    } else if (positiveEpoch_[source.raw()] != epoch_) {
                        inverted_.push_back(tensor);
                    } else {
                        auto &negated = notTensorByNode_[source.raw()];
                        if (negated == kNoTensor) {
                            const std::uint32_t operand[] = {tensor};
                            const auto added = addOp(node, ::pcfb::OpCode::BITWISE_NOT, operand, {}, std::nullopt, {});
                            if (!added) {
                                return false;
                            }
                            negated = *added;
                        }
                        inputs_.push_back(negated);
                    }
                }
                return true;
            }

            [[nodiscard]] auto emitCardinality(const std::uint32_t node, std::string_view name) -> std::optional<std::uint32_t> {
                const auto fanIn = model_.operand_counts()[node];
                const auto low = model_.cardinality_low()[node];
                const auto high = model_.cardinality_high()[node];
                if (low == 0 && high == fanIn) {
                    return addOp(node, ::pcfb::OpCode::BITWISE_ONES, {}, {}, std::nullopt, name);
                }
                if (!partitionOperands(node)) {
                    return std::nullopt;
                }
                if (high == fanIn) {
                    return addOp(node, ::pcfb::OpCode::BITWISE_ATLEAST, inputs_, inverted_, low, name);
                }
                if (low == 0) {
                    return addOp(node, ::pcfb::OpCode::BITWISE_ATMOST, inputs_, inverted_, high, name);
                }
                if (low == high) {
                    return addOp(node, ::pcfb::OpCode::BITWISE_EXACTLY, inputs_, inverted_, low, name);
                }
                const auto atLeast = addOp(node, ::pcfb::OpCode::BITWISE_ATLEAST, inputs_, inverted_, low, {});
                if (!atLeast) {
                    return std::nullopt;
                }
                const auto atMost = addOp(node, ::pcfb::OpCode::BITWISE_ATMOST, inputs_, inverted_, high, {});
                if (!atMost) {
                    return std::nullopt;
                }
                const std::uint32_t both[] = {*atLeast, *atMost};
                return addOp(node, ::pcfb::OpCode::BITWISE_AND, both, {}, std::nullopt, name);
            }

            [[nodiscard]] auto emitNode(const std::uint32_t node) -> bool {
                const auto name = nameOf(options_, node);
                std::optional<std::uint32_t> tensor;
                switch (const auto code = model_.op_codes()[node]) {
                case ir::pc::op_code::basic_event: {
                    if (!roomForTensor(node)) {
                        return false;
                    }
                    const auto eventId = eventIdByNode_[node];
                    const auto probability = model_.probability_sources()[model_.probability_source_refs()[node].raw()].literal;
                    const auto sampleName = name.empty() ? "e" + std::to_string(eventId) : std::string(name);
                    tensor = accept(builder_.add_sample(probability, sampleName, eventId, ::pcfb::SampleSource::BASIC_EVENT), node);
                    break;
                }
                case ir::pc::op_code::constant:
                case ir::pc::op_code::house_event:
                    tensor = addOp(node, model_.constant_values()[node] != 0 ? ::pcfb::OpCode::BITWISE_ONES : ::pcfb::OpCode::BITWISE_ZEROS, {}, {},
                                   std::nullopt, name);
                    break;
                case ir::pc::op_code::cardinality:
                    tensor = emitCardinality(node, name);
                    break;
                case ir::pc::op_code::and_gate:
                case ir::pc::op_code::or_gate:
                case ir::pc::op_code::xor_gate:
                case ir::pc::op_code::not_gate:
                case ir::pc::op_code::nand_gate:
                case ir::pc::op_code::nor_gate:
                case ir::pc::op_code::xnor_gate:
                    if (partitionOperands(node)) {
                        tensor = addOp(node, gateOpcode(code), inputs_, inverted_, std::nullopt, name);
                    }
                    break;
                }
                if (!tensor) {
                    return false;
                }
                tensorByNode_[node] = *tensor;
                return true;
            }

            [[nodiscard]] auto emitQueries() -> bool {
                std::size_t nonPrimary = 0;
                for (auto const &target : model_.targets()) {
                    const auto named = options_.target_names.find(target.id);
                    const auto name = named == options_.target_names.end() || named->second.empty() ? "t" + std::to_string(target.id.raw())
                                                                                                     : named->second;
                    if (auto added = builder_.add_query(tensorByNode_[target.node.raw()], pcfbQueryKind(options_.query), name); !added) {
                        diagnostics_.push(api::make_error("io.pcfb.builder", "pcfb::builder rejected target " + std::to_string(target.id.raw()) +
                                                                                 ": " + added.error().message));
                        return false;
                    }
                    nonPrimary += target.role == ir::pc::observable_role::primary ? 0 : 1;
                }
                if (nonPrimary != 0) {
                    diagnostics_.push(api::make_note("io.pcfb.target-role", std::to_string(nonPrimary) +
                                                                                " non-primary target(s) written as ordinary queries; pcfb "
                                                                                "core-1.1 has no observable roles"));
                }
                return true;
            }

            [[nodiscard]] auto setMetadata(std::string_view key, ::pcfb::Result<void> const &result) -> bool {
                if (!result) {
                    diagnostics_.push(api::make_error("io.pcfb.builder", "pcfb::builder rejected metadata key '" + std::string(key) +
                                                                             "': " + result.error().message));
                    return false;
                }
                return true;
            }

            [[nodiscard]] auto emitMetadata() -> bool {
                bool ok = setMetadata("general.generator", builder_.set("general.generator", std::string_view{"fuzzybits"})) &&
                          setMetadata("general.generator_version",
                                      builder_.set("general.generator_version", std::string_view{FUZZYBITS_PCFB_GENERATOR_VERSION}));
                if (ok && !options_.model_name.empty()) {
                    ok = setMetadata("general.model_name", builder_.set("general.model_name", std::string_view{options_.model_name}));
                }
                if (ok && !options_.source_format.empty()) {
                    ok = setMetadata("general.source_format", builder_.set("general.source_format", std::string_view{options_.source_format}));
                }
                if (ok && options_.source_sha256) {
                    ok = setMetadata("general.source_sha256",
                                     builder_.set("general.source_sha256", std::span<std::byte const>{*options_.source_sha256}));
                }
                if (ok && !options_.compile_options.empty()) {
                    ok = setMetadata("compile.options", builder_.set("compile.options", std::string_view{options_.compile_options}));
                }
                return ok;
            }

            ir::pc::circuit const &model_;
            write_options const &options_;
            ::pcfb::builder builder_;
            api::diagnostic_list diagnostics_;
            std::vector<std::uint32_t> tensorByNode_;
            std::vector<std::uint32_t> eventIdByNode_;
            std::vector<std::uint32_t> notTensorByNode_;
            std::vector<std::uint32_t> positiveEpoch_;
            std::vector<std::uint32_t> inputs_;
            std::vector<std::uint32_t> inverted_;
            std::uint32_t epoch_ = 0;
            std::uint32_t tensorCount_ = 0;
        };

    } // namespace

    auto write_pc(ir::pc::circuit const &model, write_options const &options) -> pcfb_write_result {
        if (auto diagnostics = validateSerializable(model, options); diagnostics.has_errors()) {
            return pcfb_write_result::fail(std::move(diagnostics));
        }
        auto order = emission_order(graph_view::of(model));
        if (!order) {
            return pcfb_write_result::fail(std::move(order.diagnostics()));
        }
        return Emitter(model, options).run(order.value());
    }

    auto parse_sha256_hex(std::string const &text) -> std::optional<std::array<std::byte, 32>> {
        std::array<std::byte, 32> digest{};
        if (text.size() != 2 * digest.size()) {
            return std::nullopt;
        }
        const auto nibble = [](const char character) -> int {
            if (character >= '0' && character <= '9') {
                return character - '0';
            }
            if (character >= 'a' && character <= 'f') {
                return character - 'a' + 10;
            }
            if (character >= 'A' && character <= 'F') {
                return character - 'A' + 10;
            }
            return -1;
        };
        for (std::size_t index = 0; index < digest.size(); ++index) {
            const auto high = nibble(text[2 * index]);
            const auto low = nibble(text[2 * index + 1]);
            if (high < 0 || low < 0) {
                return std::nullopt;
            }
            digest[index] = static_cast<std::byte>((high << 4) | low);
        }
        return digest;
    }

} // namespace fuzzybits::io::backend::fuzzybits::pcfb::detail
