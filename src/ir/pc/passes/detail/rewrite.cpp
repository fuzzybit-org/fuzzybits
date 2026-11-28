#include "ir/pc/passes/detail/rewrite.h"

#include "api/diagnostics.h"

#include <algorithm>
#include <string>
#include <utility>

namespace fuzzybits::ir::pc::passes::detail {

auto cone_order(circuit const &input) -> std::vector<std::uint32_t> {
    const auto node_count = input.node_count();
    const auto offsets = input.operand_offsets();
    const auto counts = input.operand_counts();
    const auto edges = input.edges();

    enum : std::uint8_t { unseen, on_stack, finished };
    std::vector<std::uint8_t> state(node_count, unseen);
    std::vector<std::uint32_t> order;
    order.reserve(node_count);
    struct frame final {
        std::uint32_t node;
        std::uint32_t next;
    };
    std::vector<frame> stack;

    for (auto const &target : input.targets()) {
        const auto root = target.node.raw();
        if (state[root] != unseen) {
            continue;
        }
        state[root] = on_stack;
        stack.push_back(frame{.node = root, .next = 0});
        while (!stack.empty()) {
            auto &top = stack.back();
            if (top.next < counts[top.node]) {
                const auto child = edges[offsets[top.node] + top.next].source.raw();
                ++top.next;
                if (state[child] == unseen) {
                    state[child] = on_stack;
                    stack.push_back(frame{.node = child, .next = 0});
                }
                continue;
            }
            state[top.node] = finished;
            order.push_back(top.node);
            stack.pop_back();
        }
    }
    return order;
}

emitter::emitter(circuit const &input) : input_(input), source_map_(input.probability_sources().size(), probability_source_id::invalid()) {
    builder_.reserve_nodes(input.node_count());
}

auto emitter::counted(const bool_node_id node) noexcept -> bool_node_id {
    ++node_count_;
    return node;
}

auto emitter::source(const probability_source_id input_source) -> probability_source_id {
    auto &mapped = source_map_[input_source.raw()];
    if (!mapped.valid()) {
        auto const &payload = input_.probability_sources()[input_source.raw()];
        switch (payload.kind) {
        case probability_source_kind::literal:
            mapped = builder_.add_literal_probability(payload.literal);
            break;
        case probability_source_kind::parameter:
            mapped = builder_.add_parameter_probability(payload.parameter_name);
            break;
        case probability_source_kind::unsupported:
            mapped = builder_.add_unsupported_probability(payload.diagnostic);
            break;
        }
    }
    return mapped;
}

auto emitter::leaf(const std::uint32_t node) -> bool_node_id {
    switch (input_.op_codes()[node]) {
    case op_code::basic_event:
        return counted(builder_.add_basic_event(source(input_.probability_source_refs()[node])));
    case op_code::house_event:
        return counted(builder_.add_house_event(input_.constant_values()[node] != 0));
    default:
        return counted(builder_.add_constant(input_.constant_values()[node] != 0));
    }
}

auto emitter::constant(const bool value) -> bool_node_id {
    auto &slot = constants_[value ? 1 : 0];
    if (!slot) {
        slot = counted(builder_.add_constant(value));
    }
    return *slot;
}

auto emitter::gate(const op_code code, const std::span<edge_ref const> operands, const std::uint32_t low, const std::uint32_t high) -> bool_node_id {
    if (code == op_code::cardinality) {
        return counted(builder_.add_cardinality(low, high, operands));
    }
    return counted(builder_.add_gate(code, operands));
}

auto emitter::finish(transform_report report) && -> transform_result {
    std::unordered_map<std::uint32_t, bool_node_id> negations;
    const auto targets = input_.targets();
    report.target_map.resize(targets.size());
    for (auto const &target : targets) {
        auto const &image = report.node_map[target.node.raw()];
        bool_node_id node;
        if (image.kind == image_kind::constant) {
            node = constant(image.constant_value);
        } else if (!image.edge.complement) {
            node = image.edge.source;
        } else if (const auto found = negations.find(image.edge.source.raw()); found != negations.end()) {
            node = found->second;
        } else {
            const auto source = image.edge.source.raw();
            const edge_ref negated[] = {edge_ref{.source = image.edge.source, .complement = true}};
            node = counted(source < atoms_.size() && atoms_[source] != 0 ? builder_.add_and(negated) : builder_.add_not(edge_ref{.source = image.edge.source}));
            negations.emplace(source, node);
        }
        report.target_map[target.id.raw()] = builder_.add_target(node, target.role);
    }
    report.scratch_bytes += scratch_bytes();
    if (report.skipped > 0) {
        report.diagnostics.push(api::make_note("pc.pass.atom-split-skipped", report.transform_id + " left " + std::to_string(report.skipped) +
                                                                                 " rewrite(s) out to preserve atom-split hull targets"));
    }

    auto sealed = std::move(builder_).seal();
    report.diagnostics.append(std::move(sealed.diagnostics));
    if (!sealed.output.valid()) {
        report.diagnostics.push(api::make_error("pc.pass.seal-failed", report.transform_id + " produced an invalid circuit"));
        return transform_result{.output = {}, .report = std::move(report)};
    }
    report.nodes_after = sealed.output.node_count();
    report.edges_after = sealed.output.edge_count();
    return transform_result{.output = std::move(sealed.output), .report = std::move(report)};
}

void emitter::mark_atom(const bool_node_id node) {
    if (atoms_.size() <= node.raw()) {
        atoms_.resize(std::max<std::size_t>(node.raw() + 1, input_.node_count()), 0);
    }
    atoms_[node.raw()] = 1;
}

auto split_context::make(circuit const &input, atom_split const &atoms, transform_report &report) -> std::optional<split_context> {
    split_context context;
    if (atoms.targets.empty() || atoms.events.empty()) {
        return context;
    }
    const auto node_count = input.node_count();
    const auto targets = input.targets();
    for (const auto target : atoms.targets) {
        if (!target.valid() || target.raw() >= targets.size()) {
            report.diagnostics.push(
                api::make_error("pc.pass.invalid-atom-split", report.transform_id + ": atom-split target " + std::to_string(target.raw()) + " does not exist"));
            return std::nullopt;
        }
    }
    for (const auto event : atoms.events) {
        if (!event.valid() || event.raw() >= node_count || input.op_codes()[event.raw()] != op_code::basic_event) {
            report.diagnostics.push(
                api::make_error("pc.pass.invalid-atom-split", report.transform_id + ": atom event " + std::to_string(event.raw()) + " is not a basic event"));
            return std::nullopt;
        }
    }

    context.in_cone_.assign(node_count, 0);
    context.member_.assign(node_count, 0);
    context.has_atom_edge_.assign(node_count, 0);
    std::vector<std::uint32_t> stack;
    for (const auto target : atoms.targets) {
        const auto root = targets[target.raw()].node.raw();
        if (context.in_cone_[root] == 0) {
            context.in_cone_[root] = 1;
            stack.push_back(root);
        }
        while (!stack.empty()) {
            const auto node = stack.back();
            stack.pop_back();
            for (auto const &operand : input.operands(bool_node_id(node))) {
                if (context.in_cone_[operand.source.raw()] == 0) {
                    context.in_cone_[operand.source.raw()] = 1;
                    stack.push_back(operand.source.raw());
                }
            }
        }
    }
    for (const auto event : atoms.events) {
        context.member_[event.raw()] = context.in_cone_[event.raw()];
    }
    for (std::uint32_t node = 0; node < node_count; ++node) {
        if (context.in_cone_[node] == 0) {
            continue;
        }
        for (auto const &operand : input.operands(bool_node_id(node))) {
            if (operand.complement && context.member_[operand.source.raw()] != 0) {
                context.has_atom_edge_[operand.source.raw()] = 1;
            }
        }
    }
    return context;
}

auto begin_report(circuit const &input, const pass_id pass) -> transform_report {
    transform_report report;
    report.transform_id = "ir.pc." + std::string(pass_name(pass));
    report.semantics = pass_semantics(pass);
    for (const auto kind : {query_kind::probability, query_kind::monotone_hull}) {
        if (pass_preserves(pass, kind)) {
            report.preserved_kinds.push_back(kind);
        }
    }
    report.nodes_before = input.node_count();
    report.edges_before = input.edge_count();
    return report;
}

auto trivial_result(circuit const &input, transform_report &report) -> std::optional<transform_result> {
    if (!input.valid()) {
        report.diagnostics.push(api::make_error("pc.pass.invalid-input", report.transform_id + " requires a sealed circuit"));
        return transform_result{.output = {}, .report = std::move(report)};
    }
    if (!input.targets().empty()) {
        report.node_map.assign(input.node_count(), node_image{});
        return std::nullopt;
    }
    report.diagnostics.push(api::make_warning("pc.pass.no-targets", report.transform_id + ": circuit has no targets; returned unchanged"));
    report.node_map.resize(input.node_count());
    for (std::uint32_t node = 0; node < input.node_count(); ++node) {
        report.node_map[node] = reference(bool_node_id(node));
    }
    report.nodes_after = report.nodes_before;
    report.edges_after = report.edges_before;
    return transform_result{.output = input, .report = std::move(report)};
}

} // namespace fuzzybits::ir::pc::passes::detail
