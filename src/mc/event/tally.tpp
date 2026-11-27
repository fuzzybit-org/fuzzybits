#pragma once

#include "mc/event/node.tpp"
#include <sycl/sycl.hpp>

namespace scram::mc::event {

// TODO:: std::size_t should be templated, and have no default type
template <typename bitpack_t_> struct device_tally : node<bitpack_t_> {
    /// @brief Count of positive outcomes (1-bits) across all samples
    std::size_t num_one_bits = 0;

    /// @brief Count of total outcomes evaluated so far
    std::size_t total_bits = 0;
};

/**
 * @struct tally
 * @brief Accumulates and stores statistical results from Monte Carlo sampling
 *
 * @details A tally event collects statistical information from multiple simulation
 * runs, computing mean probability estimates, standard errors, and confidence
 * intervals. This structure is used to aggregate results from parallel computations
 * and provide robust statistical measures for reliability analysis.
 *
 * The tally process counts positive outcomes across all bit-packed samples and
 * computes statistical measures including confidence intervals at multiple levels
 * (95% and 99%). This enables quantification of uncertainty in probability estimates.
 *
 * @tparam bitpack_t_ Integer type for bit-packed data storage
 *
 * @note Statistical measures are computed using standard Monte Carlo methods
 * @note Confidence intervals use normal approximation for large sample sizes
 *
 * @example
 * @code
 * // Process tally results
 * tally_event<std::uint64_t> tally_result;
 * if (tally_result.mean > 0.0) {
 *     std::cout << "Probability: " << tally_result.mean
 *               << " ± " << tally_result.std_err << std::endl;
 *     std::cout << "95% CI: [" << tally_result.ci[0] << ", " << tally_result.ci[1] << "]" << std::endl;
 * }
 * @endcode
 */
template <typename bitpack_t_> struct tally : device_tally<bitpack_t_> {
    /// @brief Estimated mean probability based on sample proportion
    std::double_t mean = 0.;

    /// @brief Standard error of the probability estimate
    std::double_t std_err = 0.;

    /// @brief Confidence intervals: [lower_95, upper_95, lower_99, upper_99]
    std::array<double_t, 4> ci = {0., 0., 0., 0.};
};

/**
 * @brief Destroys a tally event and frees associated device memory
 *
 * @details Properly releases device memory allocated for a tally event structure.
 * This function should be called for all tally events created with create_tally_events()
 * to prevent memory leaks.
 *
 * @tparam bitpack_t_ Integer type for bit-packed data storage
 *
 * @param queue SYCL queue for memory deallocation
 * @param event Pointer to tally event to destroy
 *
 * @note This function only frees the tally event structure, not the associated buffer
 * @note Buffers should be freed separately if they were allocated independently
 *
 * @example
 * @code
 * auto tally = create_tally_events(queue, buffers, counts);
 * // Use tally...
 * destroy_tally_event(queue, tally);
 * @endcode
 */
template <typename bitpack_t_> void destroy_tally_event(const sycl::queue queue, device_tally<bitpack_t_> *event) {
    sycl::free(event, queue);
}

// ---------------------------------------------------------------------
//  Tally block wrappers (one contiguous allocation of tally nodes)
// ---------------------------------------------------------------------

/**
 * @brief Thin wrapper describing one contiguous allocation of `tally` nodes.
 *
 * Contrary to `basic_event_block`, a tally block owns only the array of
 * `tally` structs; the bit-packed sample buffers it references are owned by
 * the producer nodes (basic events or gates).  Therefore, the wrapper holds
 * no extra fields beyond the base `node_block`.
 */
template <typename bitpack_t_> struct tally_block : public node_block<device_tally<bitpack_t_>> {
    // no additional state – buffers are external
};

/**
 * @brief Allocates and initialises a contiguous array of `tally` objects.
 *
 * Each tally’s `buffer` member is set to the corresponding entry in
 * `source_buffers`, and `num_one_bits` is initialised to 0.
 */
template <typename bitpack_t_>
[[nodiscard]]
tally_block<bitpack_t_> create_tally_block(const sycl::queue &queue, const std::vector<bitpack_t_ *> &source_buffers) {
    const auto n = source_buffers.size();
    using tally_t = device_tally<bitpack_t_>;
    //TODO:: This value should be device-scoped once we support multiple devices
    using cache_line_bytes = sycl::info::device::global_mem_cache_line_size;
    const auto align = queue.get_device().get_info<cache_line_bytes>();
    tally_t *tallies = sycl::aligned_alloc_shared<tally_t>(align, n, queue);

    for (auto i = 0; i < n; ++i) {
        tallies[i].buffer = source_buffers[i];
        tallies[i].num_one_bits = 0;
        tallies[i].total_bits = 0;
    }

    tally_block<bitpack_t_> blk;
    blk.data = tallies;
    blk.count = n;
    return blk;
}

/**
 * @brief Frees the USM allocation owned by a `tally_block`.
 *
 * Does **not** free the sample buffers the individual tally nodes point to;
 * those are owned by the nodes that produced the samples.
 */
template <typename bitpack_t_> void destroy_tally_block(const sycl::queue &queue, tally_block<bitpack_t_> &blk) {
    if (blk.data) {
        sycl::free(blk.data, queue);
        blk.data = nullptr;
    }
    blk.count = 0;
}

template <typename bitpack_t_> inline std::ostream &operator<<(std::ostream &os, const event::tally<bitpack_t_> &t) {
    os << "p01= " << t.ci[2] << "  |  "
       << "p05= " << t.ci[0] << "  |  "
       << "mu = " << t.mean << "  |  "
       << "p95= " << t.ci[1] << "  |  "
       << "p99= " << t.ci[3] << "  |  ";
    return os;
}
}