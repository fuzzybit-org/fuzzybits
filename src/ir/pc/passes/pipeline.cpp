#include "ir/pc/passes/detail/rewrite.h"
#include "ir/pc/passes/passes.h"

#include "api/diagnostics.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace fuzzybits::ir::pc::passes {

auto pass_name(const pass_id pass) noexcept -> std::string_view {
    switch (pass) {
    case pass_id::fold_zero_probability:
        return "fold_zero_probability";
    case pass_id::fold_unit_probability:
        return "fold_unit_probability";
    case pass_id::propagate_constants:
        return "propagate_constants";
    case pass_id::eliminate_single_input:
        return "eliminate_single_input";
    case pass_id::coalesce:
        return "coalesce";
    }
    return "unknown";
}

auto pass_semantics(const pass_id pass) noexcept -> semantic_class {
    switch (pass) {
    case pass_id::fold_zero_probability:
    case pass_id::fold_unit_probability:
        return semantic_class::distribution_preserving;
    default:
        return semantic_class::function_preserving;
    }
}

auto pass_preserves(const pass_id pass, const query_kind kind) noexcept -> bool {
    return pass != pass_id::fold_unit_probability || kind == query_kind::probability;
}

auto run_pass(const pass_id pass, circuit const &input, pass_options const &options) -> transform_result {
    switch (pass) {
    case pass_id::fold_zero_probability:
        return fold_zero_probability(input, options);
    case pass_id::fold_unit_probability:
        return fold_unit_probability(input, options);
    case pass_id::propagate_constants:
        return propagate_constants(input, options);
    case pass_id::eliminate_single_input:
        return eliminate_single_input(input, options);
    case pass_id::coalesce:
        return coalesce(input, options);
    }
    return {};
}

namespace {

constexpr std::size_t targets_listed = 8;

class pipeline final {
  public:
    pipeline(circuit const &input, pipeline_options const &options) : options_(options), current_(input) {
        result_.nodes_before = input.node_count();
        result_.edges_before = input.edge_count();
        result_.node_map.resize(input.node_count());
        for (std::uint32_t node = 0; node < input.node_count(); ++node) {
            result_.node_map[node] = detail::reference(bool_node_id(node));
        }
        result_.target_map.resize(input.targets().size());
        for (std::uint32_t target = 0; target < result_.target_map.size(); ++target) {
            result_.target_map[target] = target_id(target);
        }
    }

    // Errors for a malformed kind table or a pass in `schedule` that does
    // not preserve some target's query kind.
    [[nodiscard]] auto check(std::span<pass_id const> schedule) -> bool {
        if (!current_.valid()) {
            result_.diagnostics.push(
                api::make_error("pc.pipeline.invalid-input", "pipeline requires a sealed circuit"));
            return false;
        }
        const auto targets = current_.targets().size();
        if (!options_.kinds.empty() && options_.kinds.size() != targets) {
            result_.diagnostics.push(api::make_error("pc.pipeline.query-kinds-mismatch",
                                                     "query kinds has " + std::to_string(options_.kinds.size()) +
                                                         " entries for " + std::to_string(targets) + " targets"));
            return false;
        }
        if (!options_.atom_events.empty() && options_.atom_events.size() != targets) {
            result_.diagnostics.push(api::make_error("pc.pipeline.atom-events-mismatch",
                                                     "atom events has " + std::to_string(options_.atom_events.size()) +
                                                         " entries for " + std::to_string(targets) + " targets"));
            return false;
        }
        for (std::uint32_t target = 0; target < options_.atom_events.size(); ++target) {
            if (options_.atom_events[target].empty() || options_.kinds.empty() ||
                options_.kinds[target] != query_kind::monotone_hull) {
                continue;
            }
            pass_options_.atoms.targets.push_back(target_id(target));
            pass_options_.atoms.events.insert(pass_options_.atoms.events.end(), options_.atom_events[target].begin(),
                                              options_.atom_events[target].end());
        }
        deduplicate(pass_options_.atoms.events);
        bool ok = true;
        for (const auto pass : schedule) {
            const auto refused = refused_targets(pass);
            if (refused.empty()) {
                continue;
            }
            ok = false;
            std::string listed;
            for (std::size_t index = 0; index < std::min(refused.size(), targets_listed); ++index) {
                listed += (index == 0 ? "" : ", ") + std::to_string(refused[index]);
            }
            if (refused.size() > targets_listed) {
                listed += ", ...";
            }
            result_.diagnostics.push(api::make_error(
                "pc.pipeline.incompatible-pass",
                std::string(pass_name(pass)) + " is distribution-preserving for probability queries only; " +
                    std::to_string(refused.size()) + " target(s) are monotone-hull queries (target ids " + listed +
                    ")"));
        }
        return ok;
    }

    [[nodiscard]] auto refused_targets(const pass_id pass) const -> std::vector<std::uint32_t> {
        std::vector<std::uint32_t> refused;
        for (std::uint32_t target = 0; target < options_.kinds.size(); ++target) {
            if (!pass_preserves(pass, options_.kinds[target])) {
                refused.push_back(target);
            }
        }
        return refused;
    }

    [[nodiscard]] auto apply(const pass_id pass) -> bool {
        auto applied = run_pass(pass, current_, pass_options_);
        result_.diagnostics.append(applied.report.diagnostics);
        if (!applied.ok()) {
            return false;
        }
        auto &events = pass_options_.atoms.events;
        std::size_t kept = 0;
        for (const auto event : events) {
            if (auto const &image = applied.report.node_map[event.raw()];
                image.kind == image_kind::reference &&
                applied.output.op_codes()[image.edge.source.raw()] == op_code::basic_event) {
                events[kept++] = image.edge.source;
            }
        }
        events.resize(kept);
        deduplicate(events);
        for (auto &image : result_.node_map) {
            image = compose(image, applied.report.node_map);
        }
        for (auto &target : result_.target_map) {
            target = applied.report.target_map[target.raw()];
        }
        if (!options_.keep_pass_maps) {
            std::vector<node_image>().swap(applied.report.node_map);
            std::vector<target_id>().swap(applied.report.target_map);
        }
        current_ = std::move(applied.output);
        result_.passes.push_back(std::move(applied.report));
        return true;
    }

    void note(std::string code, std::string message) {
        result_.diagnostics.push(api::make_note(std::move(code), std::move(message)));
    }

    [[nodiscard]] auto size() const noexcept -> std::pair<std::uint32_t, std::uint32_t> {
        return {current_.node_count(), current_.edge_count()};
    }

    [[nodiscard]] auto finish(const bool ok) && -> pipeline_result {
        if (!ok) {
            result_.output = {};
            return std::move(result_);
        }
        result_.output = std::move(current_);
        result_.nodes_after = result_.output.node_count();
        result_.edges_after = result_.output.edge_count();
        return std::move(result_);
    }

  private:
    static void deduplicate(std::vector<bool_node_id> &events) {
        std::ranges::sort(events);
        events.erase(std::ranges::unique(events).begin(), events.end());
    }

    pipeline_options const &options_;
    // Atom events in the current circuit's node ids.
    pass_options pass_options_;
    circuit current_;
    pipeline_result result_;
};

} // namespace

auto run_pipeline(circuit const &input, const std::span<pass_id const> schedule, pipeline_options const &options)
    -> pipeline_result {
    pipeline state(input, options);
    if (!state.check(schedule)) {
        return std::move(state).finish(false);
    }
    for (const auto pass : schedule) {
        if (!state.apply(pass)) {
            return std::move(state).finish(false);
        }
    }
    return std::move(state).finish(true);
}

auto compile(circuit const &input, const compile_level level, pipeline_options const &options) -> pipeline_result {
    pipeline state(input, options);
    if (!state.check({})) {
        return std::move(state).finish(false);
    }
    if (level == compile_level::c0) {
        return std::move(state).finish(true);
    }

    if (!state.apply(pass_id::fold_zero_probability)) {
        return std::move(state).finish(false);
    }
    if (state.refused_targets(pass_id::fold_unit_probability).empty()) {
        if (!state.apply(pass_id::fold_unit_probability)) {
            return std::move(state).finish(false);
        }
    } else {
        state.note("pc.pipeline.pass-skipped", "fold_unit_probability skipped: some targets are monotone-hull queries");
    }

    constexpr pass_id round[] = {pass_id::propagate_constants, pass_id::eliminate_single_input, pass_id::coalesce};
    for (std::uint32_t iteration = 0; iteration < std::max(options.max_rounds, 1U); ++iteration) {
        const auto before = state.size();
        for (const auto pass : round) {
            if (!state.apply(pass)) {
                return std::move(state).finish(false);
            }
        }
        if (state.size() == before) {
            break;
        }
    }
    return std::move(state).finish(true);
}

} // namespace fuzzybits::ir::pc::passes
