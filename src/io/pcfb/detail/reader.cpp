#include "io/pcfb/descriptor.h"
#include "io/pcfb/detail/model.h"

#include "api/diagnostics.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/id.h"
#include "ir/pc/op_code.h"
#include "ir/pc/target.h"

#include "pcfb/io.h"
#include "pcfb/validate.h"
#include "pcfb_generated.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace fuzzybits::io::backend::fuzzybits::pcfb::detail {
namespace {

constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] auto textOf(flatbuffers::String const *value) -> std::string_view {
    return value == nullptr ? std::string_view{} : std::string_view{value->c_str(), value->size()};
}

// flatbuffers::Vector<uint32_t> that may be absent.
class Indices final {
  public:
    Indices() = default;
    explicit Indices(flatbuffers::Vector<std::uint32_t> const *values) : values_(values) {}

    [[nodiscard]] auto size() const -> std::uint32_t { return values_ == nullptr ? 0U : values_->size(); }
    [[nodiscard]] auto operator[](const std::uint32_t index) const -> std::uint32_t { return values_->Get(index); }
    [[nodiscard]] friend auto operator==(Indices const &left, Indices const &right) -> bool {
        if (left.size() != right.size()) {
            return false;
        }
        for (std::uint32_t index = 0; index < left.size(); ++index) {
            if (left[index] != right[index]) {
                return false;
            }
        }
        return true;
    }

  private:
    flatbuffers::Vector<std::uint32_t> const *values_ = nullptr;
};

enum class Role : std::uint8_t {
    node,
    // ATLEAST or ATMOST of a lowered range cardinality.
    cardinality_part,
    // A NOT tensor the writer created so that inputs and inverted_inputs stay disjoint.
    complement_tensor,
};

struct Operand final {
    std::uint32_t source;
    bool complement;
};

struct NodeInfo final {
    std::uint32_t op = 0;
    std::uint32_t list_op = 0;
    ir::pc::op_code code = ir::pc::op_code::basic_event;
    std::uint32_t low = 0;
    std::uint32_t high = 0;
    bool value = false;
    // Operands as the file lists them: `inputs` (plain, or complemented through a NOT tensor) and
    // `inverted_inputs`, each in file order.
    std::uint32_t positive_begin = 0;
    std::uint32_t positive_end = 0;
    std::uint32_t negative_begin = 0;
    std::uint32_t negative_end = 0;
};

[[nodiscard]] auto gateCode(const ::pcfb::OpCode opcode) -> std::optional<ir::pc::op_code> {
    switch (opcode) {
    case ::pcfb::OpCode::BITWISE_AND:
    case ::pcfb::OpCode::BITWISE_NO_OP:
        return ir::pc::op_code::and_gate;
    case ::pcfb::OpCode::BITWISE_OR:
        return ir::pc::op_code::or_gate;
    case ::pcfb::OpCode::BITWISE_XOR:
        return ir::pc::op_code::xor_gate;
    case ::pcfb::OpCode::BITWISE_NOT:
        return ir::pc::op_code::not_gate;
    case ::pcfb::OpCode::BITWISE_NAND:
        return ir::pc::op_code::nand_gate;
    case ::pcfb::OpCode::BITWISE_NOR:
        return ir::pc::op_code::nor_gate;
    case ::pcfb::OpCode::BITWISE_XNOR:
        return ir::pc::op_code::xnor_gate;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] auto supported(const ::pcfb::OpCode opcode) -> bool {
    switch (opcode) {
    case ::pcfb::OpCode::SAMPLE:
    case ::pcfb::OpCode::BITWISE_ZEROS:
    case ::pcfb::OpCode::BITWISE_ONES:
    case ::pcfb::OpCode::BITWISE_ATLEAST:
    case ::pcfb::OpCode::BITWISE_ATMOST:
    case ::pcfb::OpCode::BITWISE_EXACTLY:
        return true;
    default:
        return gateCode(opcode).has_value();
    }
}

[[nodiscard]] auto error(std::string_view code, std::string message) -> api::diagnostic {
    return api::make_error(std::string(code), std::move(message));
}

[[nodiscard]] auto warning(std::string_view code, std::string message) -> api::diagnostic {
    return api::make_warning(std::string(code), std::move(message));
}

[[nodiscard]] auto note(std::string_view code, std::string message) -> api::diagnostic {
    return api::make_note(std::string(code), std::move(message));
}

class Reader final {
  public:
    explicit Reader(::pcfb::ir::Circuit const &file) : file_(file) {}

    [[nodiscard]] auto run() && -> pcfb_read_result {
        if (!decode()) {
            return pcfb_read_result::fail(std::move(diagnostics_));
        }
        readMetadata();
        if (!lowerStructure() || !buildNodes()) {
            return pcfb_read_result::fail(std::move(diagnostics_));
        }
        layout();
        if (!assemble()) {
            return pcfb_read_result::fail(std::move(diagnostics_));
        }
        fillOptions();
        output_.schema_version = file_->schema_version();
        return pcfb_read_result::ok(std::move(output_), std::move(diagnostics_));
    }

  private:
    [[nodiscard]] auto op(const std::uint32_t index) const -> ::pcfb::Op const * { return ops_->Get(index); }
    [[nodiscard]] static auto inputsOf(::pcfb::Op const *operation) -> Indices { return Indices(operation->inputs()); }
    [[nodiscard]] static auto invertedOf(::pcfb::Op const *operation) -> Indices {
        return Indices(operation->inverted_inputs());
    }
    [[nodiscard]] auto kOf(::pcfb::Op const *operation) const -> std::uint32_t {
        auto const *args = operation->args_as_KofNArgs();
        return args == nullptr ? 0U : args->k();
    }

    [[nodiscard]] auto decode() -> bool {
        auto const *subcircuits = file_->subcircuits();
        if (subcircuits == nullptr || subcircuits->size() != 1) {
            diagnostics_.push(error(diag::invalid_file, "the file does not have exactly one subcircuit"));
            return false;
        }
        auto const *sub = subcircuits->Get(0);
        if (sub->inputs() != nullptr && sub->inputs()->size() != 0) {
            diagnostics_.push(
                error(diag::external_inputs, "the subcircuit lists " + std::to_string(sub->inputs()->size()) +
                                                 " externally supplied input tensor(s); ir::pc has no such leaf"));
            return false;
        }
        ops_ = sub->operators();
        queries_ = sub->queries();
        groups_ = sub->query_groups();
        opCount_ = ops_ == nullptr ? 0U : ops_->size();
        const std::uint64_t tensorCount = sub->tensors() == nullptr ? 0U : sub->tensors()->size();
        if (opCount_ >= kNone - 1 || tensorCount >= kNone - 1) {
            diagnostics_.push(error(diag::too_large, "the file has more tensors than a 32-bit node id can address"));
            return false;
        }
        opOfTensor_.assign(static_cast<std::size_t>(tensorCount), kNone);
        output_.node_by_tensor.assign(static_cast<std::size_t>(tensorCount), ir::pc::bool_node_id::invalid());
        for (std::uint32_t index = 0; index < opCount_; ++index) {
            auto const *operation = op(index);
            if (!supported(operation->opcode())) {
                diagnostics_.push(
                    error(diag::unsupported_opcode, "operator " + std::to_string(index) + " has opcode " +
                                                        std::to_string(static_cast<unsigned>(operation->opcode())) +
                                                        ", which ir::pc cannot represent"));
                return false;
            }
            if (operation->output() >= opOfTensor_.size()) {
                diagnostics_.push(
                    error(diag::invalid_file, "operator " + std::to_string(index) + " writes a tensor out of range"));
                return false;
            }
            opOfTensor_[operation->output()] = index;
        }
        if (queries_ == nullptr) {
            diagnostics_.push(error(diag::invalid_file, "the file has no queries"));
            return false;
        }
        return true;
    }

    [[nodiscard]] auto producer(const std::uint32_t tensor) -> std::uint32_t {
        const auto found = tensor < opOfTensor_.size() ? opOfTensor_[tensor] : kNone;
        if (found == kNone && !missingReported_) {
            missingReported_ = true;
            diagnostics_.push(
                error(diag::invalid_file, "tensor " + std::to_string(tensor) + " has no producing operator"));
        }
        return found;
    }

    // Resolves operand and query tensors to operators, counts uses, and decides which operators
    // are the lowered form of a construct that ir::pc holds as one node or one edge flag.
    [[nodiscard]] auto lowerStructure() -> bool {
        fanout_.assign(opCount_, 0);
        queried_.assign(opCount_, 0);
        role_.assign(opCount_, Role::node);
        pairLow_.assign(opCount_, kNone);
        for (std::uint32_t index = 0; index < opCount_; ++index) {
            for (const auto list : {inputsOf(op(index)), invertedOf(op(index))}) {
                for (std::uint32_t position = 0; position < list.size(); ++position) {
                    const auto source = producer(list[position]);
                    if (source == kNone) {
                        return false;
                    }
                    if (source >= index) {
                        diagnostics_.push(error(diag::invalid_file, "operator " + std::to_string(index) +
                                                                        " reads tensor " +
                                                                        std::to_string(list[position]) +
                                                                        ", which a later operator writes"));
                        return false;
                    }
                    ++fanout_[source];
                }
            }
        }
        for (auto const *query : *queries_) {
            const auto target = producer(query->target());
            if (target == kNone) {
                return false;
            }
            ++fanout_[target];
            queried_[target] = 1;
        }
        foldCardinality();
        foldComplementTensors();
        return true;
    }

    // The writer lowers cardinality(low, high) with 0 < low < high < n to two unnamed operators
    // over one operand list, used only by the named AND that follows.
    void foldCardinality() {
        for (std::uint32_t index = 0; index < opCount_; ++index) {
            auto const *conjunction = op(index);
            if (conjunction->opcode() != ::pcfb::OpCode::BITWISE_AND || inputsOf(conjunction).size() != 2 ||
                invertedOf(conjunction).size() != 0) {
                continue;
            }
            const auto least = producer(inputsOf(conjunction)[0]);
            const auto most = producer(inputsOf(conjunction)[1]);
            if (least == most || least == kNone || most == kNone) {
                continue;
            }
            auto const *leastOp = op(least);
            auto const *mostOp = op(most);
            if (leastOp->opcode() != ::pcfb::OpCode::BITWISE_ATLEAST ||
                mostOp->opcode() != ::pcfb::OpCode::BITWISE_ATMOST || !textOf(leastOp->name()).empty() ||
                !textOf(mostOp->name()).empty() || fanout_[least] != 1 || fanout_[most] != 1 ||
                inputsOf(leastOp) != inputsOf(mostOp) || invertedOf(leastOp) != invertedOf(mostOp) ||
                kOf(leastOp) > kOf(mostOp)) {
                continue;
            }
            role_[least] = Role::cardinality_part;
            role_[most] = Role::cardinality_part;
            pairLow_[index] = least;
        }
    }

    // The writer lists the NOT tensors it creates to keep inputs and inverted_inputs disjoint in
    // compile.complement_tensors; those, and only those, are complemented edges of ir::pc.
    void foldComplementTensors() {
        const auto found = metadataIndex_.find("compile.complement_tensors");
        if (found == metadataIndex_.end()) {
            return;
        }
        auto const *packed = std::get_if<std::vector<std::byte>>(&output_.metadata[found->second].second);
        if (packed == nullptr || packed->size() % 4 != 0) {
            diagnostics_.push(
                warning(diag::invalid_metadata,
                        "metadata 'compile.complement_tensors' is not a list of 32-bit tensor ids and is ignored"));
            return;
        }
        std::vector<std::uint32_t> tensors;
        for (std::size_t at = 0; at < packed->size(); at += 4) {
            std::uint32_t tensor = 0;
            for (unsigned byte = 0; byte < 4; ++byte) {
                tensor |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>((*packed)[at + byte]))
                          << (8U * byte);
            }
            if (tensor >= opOfTensor_.size() || (!tensors.empty() && tensor <= tensors.back())) {
                diagnostics_.push(warning(diag::invalid_metadata,
                                          "metadata 'compile.complement_tensors' lists tensor " +
                                              std::to_string(tensor) + " out of range or out of order and is ignored"));
                return;
            }
            tensors.push_back(tensor);
        }
        std::vector<std::uint8_t> invertedUse(opCount_, 0);
        for (std::uint32_t index = 0; index < opCount_; ++index) {
            const auto negative = invertedOf(op(index));
            for (std::uint32_t position = 0; position < negative.size(); ++position) {
                invertedUse[producer(negative[position])] = 1;
            }
        }
        std::vector<std::uint8_t> listed(opCount_, 0);
        for (const auto tensor : tensors) {
            listed[opOfTensor_[tensor]] = 1;
        }
        for (const auto tensor : tensors) {
            const auto index = opOfTensor_[tensor];
            auto const *operation = op(index);
            const bool shaped = operation->opcode() == ::pcfb::OpCode::BITWISE_NOT &&
                                textOf(operation->name()).empty() && inputsOf(operation).size() == 1 &&
                                invertedOf(operation).size() == 0;
            const auto source = shaped ? producer(inputsOf(operation)[0]) : kNone;
            if (!shaped || queried_[index] != 0 || invertedUse[index] != 0 || role_[index] != Role::node ||
                source == kNone || listed[source] != 0 || role_[source] != Role::node) {
                diagnostics_.push(
                    warning(diag::invalid_metadata,
                            "metadata 'compile.complement_tensors' lists tensor " + std::to_string(tensor) +
                                ", which is not a NOT tensor the writer created; it is read as a NOT gate"));
                continue;
            }
            role_[index] = Role::complement_tensor;
        }
    }

    void appendOperands(NodeInfo &info) {
        const auto listOp = info.list_op;
        info.positive_begin = static_cast<std::uint32_t>(positive_.size());
        const auto inputs = inputsOf(op(listOp));
        for (std::uint32_t position = 0; position < inputs.size(); ++position) {
            const auto used = producer(inputs[position]);
            if (role_[used] == Role::complement_tensor) {
                const auto source = producer(inputsOf(op(used))[0]);
                positive_.push_back({nodeOfOp_[source], true});
            } else {
                positive_.push_back({nodeOfOp_[used], false});
            }
        }
        info.positive_end = static_cast<std::uint32_t>(positive_.size());
        info.negative_begin = static_cast<std::uint32_t>(negative_.size());
        const auto inverted = invertedOf(op(listOp));
        for (std::uint32_t position = 0; position < inverted.size(); ++position) {
            const auto used = producer(inverted[position]);
            negative_.push_back({nodeOfOp_[used], true});
        }
        info.negative_end = static_cast<std::uint32_t>(negative_.size());
    }

    [[nodiscard]] auto buildNodes() -> bool {
        nodeOfOp_.assign(opCount_, kNone);
        std::size_t noOps = 0;
        for (std::uint32_t index = 0; index < opCount_; ++index) {
            if (role_[index] != Role::node) {
                continue;
            }
            auto const *operation = op(index);
            NodeInfo info{.op = index, .list_op = index};
            switch (operation->opcode()) {
            case ::pcfb::OpCode::SAMPLE: {
                info.code = ir::pc::op_code::basic_event;
                auto const *sample = operation->args_as_SampleArgs();
                if (sample == nullptr) {
                    diagnostics_.push(
                        error(diag::invalid_file, "SAMPLE operator " + std::to_string(index) + " has no SampleArgs"));
                    return false;
                }
                if (sample->source() != ::pcfb::SampleSource::BASIC_EVENT) {
                    ++sourcesDropped_;
                }
                break;
            }
            case ::pcfb::OpCode::BITWISE_ZEROS:
            case ::pcfb::OpCode::BITWISE_ONES:
                info.code = ir::pc::op_code::constant;
                info.value = operation->opcode() == ::pcfb::OpCode::BITWISE_ONES;
                break;
            case ::pcfb::OpCode::BITWISE_ATLEAST:
            case ::pcfb::OpCode::BITWISE_ATMOST:
            case ::pcfb::OpCode::BITWISE_EXACTLY: {
                info.code = ir::pc::op_code::cardinality;
                const auto fanIn = inputsOf(operation).size() + invertedOf(operation).size();
                const auto k = kOf(operation);
                info.low = operation->opcode() == ::pcfb::OpCode::BITWISE_ATMOST ? 0 : k;
                info.high = operation->opcode() == ::pcfb::OpCode::BITWISE_ATLEAST ? fanIn : k;
                break;
            }
            default:
                info.code = *gateCode(operation->opcode());
                noOps += operation->opcode() == ::pcfb::OpCode::BITWISE_NO_OP ? 1 : 0;
                if (pairLow_[index] != kNone) {
                    info.code = ir::pc::op_code::cardinality;
                    info.list_op = pairLow_[index];
                    info.low = kOf(op(pairLow_[index]));
                    info.high = kOf(op(producer(inputsOf(operation)[1])));
                }
                break;
            }
            nodeOfOp_[index] = static_cast<std::uint32_t>(nodes_.size());
            info.positive_begin = info.positive_end = static_cast<std::uint32_t>(positive_.size());
            info.negative_begin = info.negative_end = static_cast<std::uint32_t>(negative_.size());
            if (ir::pc::accepts_operands(info.code)) {
                appendOperands(info);
            }
            nodes_.push_back(info);
        }
        if (missingReported_) {
            return false;
        }
        if (noOps != 0) {
            diagnostics_.push(
                note(diag::no_op_as_and, std::to_string(noOps) + " BITWISE_NO_OP operator(s) read as one-operand AND"));
        }
        if (sourcesDropped_ != 0) {
            diagnostics_.push(
                warning(diag::sample_source_dropped,
                        std::to_string(sourcesDropped_) +
                            " SAMPLE operator(s) with source COLLAPSED_MODULE or HOUSE_EVENT read as basic events"));
        }
        return true;
    }

    [[nodiscard]] auto sampleOf(const std::uint32_t node) const -> ::pcfb::SampleArgs const * {
        return op(nodes_[node].op)->args_as_SampleArgs();
    }

    // The canonical numbering: basic events first, by ascending file event id, then every other node
    // in file operator order. The file lists an operator after the operators it reads, so every
    // operand has a smaller id than its consumer, except through the events, which come first.
    void layout() {
        const auto count = static_cast<std::uint32_t>(nodes_.size());
        std::vector<std::uint32_t> events;
        for (std::uint32_t node = 0; node < count; ++node) {
            if (nodes_[node].code == ir::pc::op_code::basic_event) {
                events.push_back(node);
            }
        }
        std::ranges::sort(events, [&](const std::uint32_t left, const std::uint32_t right) {
            return sampleOf(left)->event_id() < sampleOf(right)->event_id();
        });
        for (std::uint32_t rank = 0; rank < events.size(); ++rank) {
            if (sampleOf(events[rank])->event_id() != rank + 1) {
                diagnostics_.push(warning(diag::event_ids_renumbered,
                                          "event ids are not 1.." + std::to_string(events.size()) +
                                              "; the read circuit numbers its events by ascending file id"));
                break;
            }
        }
        order_ = std::move(events);
        order_.reserve(count);
        for (std::uint32_t node = 0; node < count; ++node) {
            if (nodes_[node].code != ir::pc::op_code::basic_event) {
                order_.push_back(node);
            }
        }
        idOfNode_.assign(count, 0);
        for (std::uint32_t id = 0; id < count; ++id) {
            idOfNode_[order_[id]] = id;
        }
    }

    [[nodiscard]] auto assemble() -> bool {
        ir::pc::builder builder;
        const auto count = static_cast<std::uint32_t>(nodes_.size());
        builder.reserve_nodes(count);
        std::vector<ir::pc::edge_ref> operands;
        output_.tensor_by_node.resize(count);
        for (std::uint32_t id = 0; id < count; ++id) {
            auto const &info = nodes_[order_[id]];
            operands.clear();
            for (const auto list :
                 {std::span{positive_}.subspan(info.positive_begin, info.positive_end - info.positive_begin),
                  std::span{negative_}.subspan(info.negative_begin, info.negative_end - info.negative_begin)}) {
                for (auto const &operand : list) {
                    operands.push_back(ir::pc::edge_ref{.source = ir::pc::bool_node_id(idOfNode_[operand.source]),
                                                        .complement = operand.complement});
                }
            }
            switch (info.code) {
            case ir::pc::op_code::basic_event:
                (void)builder.add_basic_event(builder.add_literal_probability(sampleOf(order_[id])->probability()));
                break;
            case ir::pc::op_code::constant:
                (void)builder.add_constant(info.value);
                break;
            case ir::pc::op_code::cardinality:
                (void)builder.add_cardinality(info.low, info.high, std::span<ir::pc::edge_ref const>{operands});
                break;
            default:
                (void)builder.add_gate(info.code, std::span<ir::pc::edge_ref const>{operands});
                break;
            }
            const auto tensor = op(info.op)->output();
            output_.tensor_by_node[id] = tensor;
            output_.node_by_tensor[tensor] = ir::pc::bool_node_id(id);
        }
        if (!addTargets(builder)) {
            return false;
        }
        auto sealed = std::move(builder).seal();
        diagnostics_.append(std::move(sealed.diagnostics));
        if (!sealed.ok()) {
            diagnostics_.push(error(diag::invalid_circuit, "the circuit rebuilt from the file failed ir::pc sealing"));
            return false;
        }
        output_.circuit = std::move(sealed.output);
        return true;
    }

    // Consecutive queries on one tensor with one name and distinct kinds are the kinds of one
    // target, which is how the writer lays out a target with several kinds.
    [[nodiscard]] auto addTargets(ir::pc::builder &builder) -> bool {
        std::uint32_t previousNode = kNone;
        std::string_view previousName;
        for (std::uint32_t index = 0; index < queries_->size(); ++index) {
            auto const *query = queries_->Get(index);
            const auto node = nodeOfOp_[producer(query->target())];
            if (node == kNone) {
                diagnostics_.push(
                    error(diag::invalid_file, "query " + std::to_string(index) + " targets a tensor without a node"));
                return false;
            }
            const auto name = textOf(query->name());
            query_kind kind = query_kind::probability;
            switch (query->kind()) {
            case ::pcfb::QueryKind::PROBABILITY:
                break;
            case ::pcfb::QueryKind::MONOTONE_HULL_PROBABILITY:
                kind = query_kind::monotone_hull_probability;
                break;
            default:
                diagnostics_.push(
                    error(diag::unknown_query_kind, "query " + std::to_string(index) + " has kind " +
                                                        std::to_string(static_cast<unsigned>(query->kind())) +
                                                        ", which this reader does not implement"));
                return false;
            }
            if (query->confidence() != nullptr && query->confidence()->size() != 0) {
                ++confidenceDropped_;
            }
            const auto role = roleOfQuery(index);
            auto &kinds = targetKinds_;
            const bool joins = !kinds.empty() && node == previousNode && name == previousName &&
                               role == targetRoles_.back() &&
                               std::ranges::find(kinds.back(), kind) == kinds.back().end();
            if (!joins) {
                (void)builder.add_target(ir::pc::bool_node_id(idOfNode_[node]), role);
                targetRoles_.push_back(role);
                targetNames_.emplace_back(name);
                kinds.emplace_back();
            }
            kinds.back().push_back(kind);
            queryTarget_.push_back(static_cast<std::uint32_t>(kinds.size() - 1));
            queryKind_.push_back(kind);
            previousNode = node;
            previousName = name;
        }
        if (confidenceDropped_ != 0) {
            diagnostics_.push(
                warning(diag::confidence_ignored, std::to_string(confidenceDropped_) +
                                                      " query(ies) request confidence levels, which are not read"));
        }
        return true;
    }

    void readMetadata() {
        auto const *entries = file_->metadata();
        if (entries == nullptr) {
            return;
        }
        for (auto const *entry : *entries) {
            metadata_value value;
            switch (entry->value_type()) {
            case ::pcfb::MetadataValue::StringValue:
                value = std::string(textOf(entry->value_as_StringValue()->value()));
                break;
            case ::pcfb::MetadataValue::IntValue:
                value = entry->value_as_IntValue()->value();
                break;
            case ::pcfb::MetadataValue::FloatValue:
                value = entry->value_as_FloatValue()->value();
                break;
            case ::pcfb::MetadataValue::BytesValue: {
                auto const *bytes = entry->value_as_BytesValue()->value();
                std::vector<std::byte> copy;
                if (bytes != nullptr) {
                    copy.reserve(bytes->size());
                    for (const auto byte : *bytes) {
                        copy.push_back(static_cast<std::byte>(byte));
                    }
                }
                value = std::move(copy);
                break;
            }
            default:
                continue;
            }
            metadataIndex_.try_emplace(std::string(textOf(entry->key())), output_.metadata.size());
            output_.metadata.emplace_back(std::string(textOf(entry->key())), std::move(value));
        }
    }

    [[nodiscard]] auto metadataText(std::string const &key) -> std::optional<std::string> {
        const auto found = metadataIndex_.find(key);
        if (found == metadataIndex_.end()) {
            return std::nullopt;
        }
        if (auto const *text = std::get_if<std::string>(&output_.metadata[found->second].second)) {
            return *text;
        }
        diagnostics_.push(warning(diag::invalid_metadata, "metadata '" + key + "' is not a string and is ignored"));
        return std::nullopt;
    }

    // compile.query.<i>.role is written only for targets that are not primary.
    [[nodiscard]] auto roleOfQuery(const std::uint32_t index) -> ir::pc::observable_role {
        const auto key = "compile.query." + std::to_string(index) + ".role";
        if (!metadataIndex_.contains(key)) {
            return ir::pc::observable_role::primary;
        }
        const auto text = metadataText(key);
        if (text && *text == "watch") {
            return ir::pc::observable_role::watch;
        }
        if (text && *text == "projection_anchor") {
            return ir::pc::observable_role::projection_anchor;
        }
        if (text) {
            diagnostics_.push(warning(diag::invalid_metadata,
                                      "metadata '" + key + "' is '" + *text +
                                          "', not watch or projection_anchor; the target is read as primary"));
        }
        return ir::pc::observable_role::primary;
    }

    void fillQuantification() {
        std::vector<std::uint8_t> recorded(targetKinds_.size(), 0);
        for (std::uint32_t index = 0; index < queryTarget_.size(); ++index) {
            const auto target = queryTarget_[index];
            const auto prefix = "compile.query." + std::to_string(index) + ".";
            target_quantification entry;
            bool present = metadataIndex_.contains(prefix + "negation");
            // The literal mode belongs to the hull query; a probability query's copy, which earlier
            // writers recorded, is ignored.
            if (const auto mode = queryKind_[index] == query_kind::monotone_hull_probability &&
                                          metadataIndex_.contains(prefix + "literal_mode")
                                      ? metadataText(prefix + "literal_mode")
                                      : std::nullopt) {
                entry.literal_mode = *mode;
                present = true;
            }
            if (const auto found = metadataIndex_.find(prefix + "max_initiators_per_cutset");
                found != metadataIndex_.end()) {
                auto const *number = std::get_if<std::int64_t>(&output_.metadata[found->second].second);
                if (number != nullptr && *number >= 0 && *number <= std::numeric_limits<std::uint32_t>::max()) {
                    entry.max_initiators_per_cutset = static_cast<std::uint32_t>(*number);
                    present = true;
                } else {
                    diagnostics_.push(warning(diag::invalid_metadata,
                                              "metadata '" + prefix +
                                                  "max_initiators_per_cutset' is not a 32-bit count and is ignored"));
                }
            }
            if (const auto initiator =
                    metadataIndex_.contains(prefix + "initiator") ? metadataText(prefix + "initiator") : std::nullopt) {
                entry.initiator = *initiator;
                present = true;
            }
            if (const auto found = metadataIndex_.find(prefix + "initiator_frequency"); found != metadataIndex_.end()) {
                if (auto const *number = std::get_if<double>(&output_.metadata[found->second].second)) {
                    entry.initiator_frequency = *number;
                    present = true;
                } else {
                    diagnostics_.push(warning(diag::invalid_metadata, "metadata '" + prefix +
                                                                          "initiator_frequency' is not a number and is "
                                                                          "ignored"));
                }
            }
            if (!present) {
                continue;
            }
            const auto id = ir::pc::target_id(target);
            if (recorded[target] == 0) {
                recorded[target] = 1;
                output_.options.quantification[id] = entry;
            } else if (auto &kept = output_.options.quantification[id];
                       (!kept.literal_mode.empty() && !entry.literal_mode.empty() &&
                        kept.literal_mode != entry.literal_mode) ||
                       kept.max_initiators_per_cutset != entry.max_initiators_per_cutset ||
                       kept.initiator != entry.initiator || kept.initiator_frequency != entry.initiator_frequency) {
                diagnostics_.push(
                    warning(diag::quantification_conflict, "queries of target " + std::to_string(target) +
                                                               " record different quantification; the first is kept"));
            } else if (kept.literal_mode.empty()) {
                // Only the target's hull query records the literal mode.
                kept.literal_mode = entry.literal_mode;
            }
        }
    }

    void fillOptions() {
        auto &options = output_.options;
        for (std::uint32_t id = 0; id < nodes_.size(); ++id) {
            auto const &info = nodes_[order_[id]];
            const auto name = info.code == ir::pc::op_code::basic_event ? textOf(sampleOf(order_[id])->name())
                                                                        : textOf(op(info.op)->name());
            if (!name.empty()) {
                options.node_names.emplace(ir::pc::bool_node_id(id), std::string(name));
            }
        }
        for (std::uint32_t target = 0; target < targetKinds_.size(); ++target) {
            const auto id = ir::pc::target_id(target);
            if (!targetNames_[target].empty()) {
                options.target_names.emplace(id, targetNames_[target]);
            }
            options.target_kinds.emplace(id, targetKinds_[target]);
        }
        fillQuantification();
        // pcfb::validate has checked the groups (rule 9): terms name queries of one kind, once each.
        if (groups_ != nullptr) {
            for (auto const *group : *groups_) {
                query_group out{.name = std::string(textOf(group->name())), .unit = std::string(textOf(group->unit()))};
                if (group->terms() != nullptr) {
                    for (auto const *term : *group->terms()) {
                        out.kind = queryKind_[term->query()];
                        out.terms.emplace_back(ir::pc::target_id(queryTarget_[term->query()]), term->weight());
                    }
                }
                options.query_groups.push_back(std::move(out));
            }
        }
        if (auto name = metadataText("general.model_name")) {
            options.model_name = std::move(*name);
        } else {
            options.model_name = std::string(textOf(file_->name()));
        }
        if (auto format = metadataText("general.source_format")) {
            options.source_format = std::move(*format);
        }
        if (auto compile = metadataText("compile.options")) {
            options.compile_options = std::move(*compile);
        }
        if (const auto found = metadataIndex_.find("general.source_sha256"); found != metadataIndex_.end()) {
            auto const *bytes = std::get_if<std::vector<std::byte>>(&output_.metadata[found->second].second);
            if (bytes != nullptr && bytes->size() == 32) {
                std::array<std::byte, 32> digest{};
                std::ranges::copy(*bytes, digest.begin());
                options.source_sha256 = digest;
            } else {
                diagnostics_.push(
                    warning(diag::invalid_metadata, "metadata 'general.source_sha256' is not 32 bytes and is ignored"));
            }
        }
    }

    ::pcfb::ir::Circuit const &file_;
    api::diagnostic_list diagnostics_;
    read_output output_;

    flatbuffers::Vector<flatbuffers::Offset<::pcfb::Op>> const *ops_ = nullptr;
    flatbuffers::Vector<flatbuffers::Offset<::pcfb::Query>> const *queries_ = nullptr;
    flatbuffers::Vector<flatbuffers::Offset<::pcfb::QueryGroup>> const *groups_ = nullptr;
    std::uint32_t opCount_ = 0;
    bool missingReported_ = false;
    std::size_t sourcesDropped_ = 0;
    std::size_t confidenceDropped_ = 0;

    std::vector<std::uint32_t> opOfTensor_;
    std::vector<std::uint32_t> fanout_;
    std::vector<std::uint8_t> queried_;
    std::vector<Role> role_;
    // For the AND of a lowered cardinality: its ATLEAST operator.
    std::vector<std::uint32_t> pairLow_;
    std::vector<std::uint32_t> nodeOfOp_;

    std::vector<NodeInfo> nodes_;
    std::vector<Operand> positive_;
    std::vector<Operand> negative_;
    std::vector<std::uint32_t> order_;
    std::vector<std::uint32_t> idOfNode_;

    std::vector<std::string> targetNames_;
    std::vector<std::vector<query_kind>> targetKinds_;
    std::vector<std::uint32_t> queryTarget_;
    std::vector<query_kind> queryKind_;
    std::vector<ir::pc::observable_role> targetRoles_;
    std::unordered_map<std::string, std::size_t> metadataIndex_;
};

} // namespace

auto read_pcfb(std::span<std::byte const> bytes) -> pcfb_read_result {
    auto file = ::pcfb::io::from_bytes(bytes);
    if (!file) {
        return pcfb_read_result::fail(error(diag::unreadable, "pcfb verification failed: " + file.error().message));
    }
    api::diagnostic_list diagnostics;
    for (auto const &finding : ::pcfb::validate(*file)) {
        const auto kind = finding.code == ::pcfb::Diagnostic::Code::UnknownQueryKind ||
                                  finding.code == ::pcfb::Diagnostic::Code::SchemaVersionTooOld
                              ? diag::unknown_query_kind
                              : diag::invalid_file;
        diagnostics.push(error(kind, finding.location + ": " + finding.message));
    }
    if (diagnostics.has_errors()) {
        return pcfb_read_result::fail(std::move(diagnostics));
    }
    auto read = Reader(*file).run();
    diagnostics.append(std::move(read.diagnostics()));
    if (!read.has_value() || diagnostics.has_errors()) {
        return pcfb_read_result::fail(std::move(diagnostics));
    }
    return pcfb_read_result::ok(std::move(read).value(), std::move(diagnostics));
}

} // namespace fuzzybits::io::backend::fuzzybits::pcfb::detail
