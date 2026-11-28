#pragma once

#include "mc/executor/layer_manager.h"
#include "mc/logger/csv.h"
#include "mc/logger/log_sample_shaper.h"

#include <memory>
#include <string>
#include <vector>

namespace scram::log::layers {

/**
 * Return a vector of (header,value) pairs suitable for CSV output that describe
 * the configuration of a `fuzzybits::mc::executor::layer_manager` instance.  The
 * function is modeled after `scram::log::sample_shaper::csv_pairs()` so that
 * it can be used seamlessly with the generic CSV helpers found in
 * `mc/logger/csv.h`.
 *
 * We purposely make this a free function (instead of a member) so that it can
 * be specialized for any layer-manager template instantiation while still
 * having direct access to the class’ private data via a `friend` declaration
 * inside `layer_manager`.
 */

template <typename bitpack_t_, typename prob_t_, typename size_t_>
inline std::vector<std::pair<std::string, std::string>> csv_pairs(const fuzzybits::mc::executor::layer_manager<bitpack_t_, prob_t_, size_t_> &lm) {
    using scram::log::csv_string;
    using pair_t = std::pair<std::string, std::string>;

    std::vector<pair_t> out;
    out.reserve(32); // rough upper-bound – avoids reallocs.

    // ---------------------------------------------------------------------
    //  Basic structural information
    // ---------------------------------------------------------------------
    out.emplace_back("num_nodes", csv_string(lm.graph_.nodes()));
    out.emplace_back("num_layers", csv_string(lm.graph_.layers()));
    out.emplace_back("num_kernels", csv_string(lm.program_.queueables.size()));

    const auto avg_kernels_per_layer = static_cast<double>(lm.program_.queueables.size()) / static_cast<double>(lm.graph_.layers());
    out.emplace_back("avg_kernels_per_layer", csv_string(avg_kernels_per_layer));

    // Device-side resource blocks.
    out.emplace_back("device_basic_event_blocks", csv_string(lm.program_.blocks.sampled_events.size()));
    // Basic events with a constant law: written once when their slots were allocated, read by no kernel.
    out.emplace_back("device_constant_event_blocks", csv_string(lm.program_.blocks.constant_events.size()));
    out.emplace_back("device_gate_blocks", csv_string(lm.program_.blocks.gates.size()));
    out.emplace_back("device_atleast_gate_blocks", csv_string(lm.program_.blocks.cardinality_gates.size()));
    out.emplace_back("device_tally_blocks", csv_string(lm.program_.blocks.tallies.size()));

    const auto all_blocks = lm.program_.blocks.size();
    out.emplace_back("device_total_blocks", csv_string(all_blocks));

    const auto kernel_blocks = all_blocks - lm.program_.blocks.constant_events.size();
    const auto avg_blocks_per_kernel = static_cast<double>(kernel_blocks) / static_cast<double>(lm.program_.queueables.size());
    out.emplace_back("avg_blocks_per_kernel", csv_string(avg_blocks_per_kernel));

    // Tally-related bookkeeping.
    out.emplace_back("tally_events_tracked", csv_string(lm.program_.tallies.size()));

    // ---------------------------------------------------------------------
    //  Detailed node composition – count variables vs. gates.
    // ---------------------------------------------------------------------
    out.emplace_back("num_variables", csv_string(lm.graph_.variables()));
    out.emplace_back("num_gates", csv_string(lm.graph_.gates()));

    // ---------------------------------------------------------------------
    //  Flatten the associated sample-shaper statistics (prefixed with "shaper_").
    // ---------------------------------------------------------------------
    const auto shaper_pairs = scram::log::sample_shaper::csv_pairs(lm.sample_shaper_);
    for (const auto &p : shaper_pairs) {
        out.emplace_back("shaper_" + p.first, p.second);
    }

    return out;
}

} // namespace scram::log::layers