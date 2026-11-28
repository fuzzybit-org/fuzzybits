/**
 * @file layer_manager.h
 * @brief SYCL-based layer management for the compute graph
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/event/basic_event.h"
#include "mc/event/node.h"
#include "mc/event/tally.h"
#include "mc/executor/builder/layer.h"
#include "mc/executor/compat.h"
#include "mc/executor/builder/program.h"
#include "mc/executor/descriptor/execution_plan.h"
#include "mc/executor/descriptor/kernel_dag.h"
#include "mc/executor/descriptor/node.h"
#include "mc/executor/queueable.h"
#include "mc/executor/sample_shaper.h"
#include "mc/prng/stream.h"
#include "mc/stats/tally_node_map.h"


#include <hipSYCL/runtime/settings.hpp>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
//  Forward declarations to break circular header dependencies.
// ---------------------------------------------------------------------------

// Forward-declare layer_manager itself so that subsequent declarations can
// reference it before its full definition appears later in this file.
namespace fuzzybits::mc::executor {
template <typename bitpack_t_, typename prob_t_, typename size_t_> class layer_manager;
}

// Forward declaration of the CSV logging helper.
namespace scram::log::layers {
template <typename bt_, typename pt_, typename st_>
std::vector<std::pair<std::string, std::string>> csv_pairs(const fuzzybits::mc::executor::layer_manager<bt_, pt_, st_> &);
}

namespace fuzzybits::mc::executor {

/// AdaptiveCpp's default async handler prints the error list and std::terminate()s.
/// Do not throw from the queue handler: `wait_and_throw` / `~queue` both invoke
/// it, so a throw from `wait_and_throw` plus a second throw from `~queue` during
/// unwind calls std::terminate. Record the message and rethrow on `wait()`.
inline thread_local std::string g_sycl_async_error;

inline std::string &sycl_async_error_slot() { return g_sycl_async_error; }

/// AdaptiveCpp reads ACPP_VISIBILITY_MASK once, when the __acpp_register_hcf
/// static initializers run before main(). A mask set later (fzb --backend,
/// runtime::apply_backend_mask) therefore filters nothing, and the default
/// selector picks any loaded backend. Re-apply the mask's backend list when
/// the queue is created (per-device conditions such as `cuda:1` are not
/// re-applied). A mask naming no backend this AdaptiveCpp knows (`metal` on
/// v25.10) selects nothing rather than falling back to the default device.
inline int visibility_mask_selector(const sycl::device &device) {
    const char *env = std::getenv("ACPP_VISIBILITY_MASK");
    if (env != nullptr && *env != '\0') {
        hipsycl::rt::visibility_mask_t mask;
        std::istringstream in{env};
        in >> mask;
        if (mask.find(device.get_backend()) == mask.end()) {
            return -1;
        }
    }
    return sycl::default_selector_v(device);
}

/// The Monte Carlo queue: in order, so that kernels run in submission order and are submitted without dependency
/// events (queueable_base::submit, check_submission_order).
inline sycl::queue make_throwing_sycl_queue() {
    return sycl::queue{visibility_mask_selector,
                       [](sycl::exception_list errors) {
                           if (errors.empty()) {
                               return;
                           }
                           std::string msg;
                           for (std::exception_ptr const &err : errors) {
                               try {
                                   if (err) {
                                       std::rethrow_exception(err);
                                   }
                               } catch (const std::exception &e) {
                                   if (!msg.empty()) {
                                       msg += "; ";
                                   }
                                   msg += e.what();
                               } catch (...) {
                                   if (!msg.empty()) {
                                       msg += "; ";
                                   }
                                   msg += "unknown SYCL asynchronous error";
                               }
                           }
                           auto &slot = sycl_async_error_slot();
                           if (!slot.empty()) {
                               slot += "; ";
                           }
                           slot += msg.empty() ? "SYCL asynchronous error" : msg;
                       },
                       sycl::property_list{sycl::property::queue::in_order{}}};
}

inline void wait_sycl_queue(sycl::queue &queue) {
    queue.wait();
    queue.throw_asynchronous();
    auto &slot = sycl_async_error_slot();
    if (slot.empty()) {
        return;
    }
    std::string msg = std::move(slot);
    slot.clear();
    throw std::runtime_error(msg);
}

/**
 * @class layer_manager
 * @brief Manages layered execution of pdag computations on SYCL devices
 *
 * @details The layer_manager class orchestrates the execution of pdag
 * computations by organizing nodes into layers based on topological ordering. It builds
 * and manages SYCL kernels for parallel execution of basic events and gates, handles
 * memory allocation, and provides tally computation for probabilistic analysis.
 *
 * The class performs several key operations:
 * - Topological sorting of PDAG nodes into execution layers
 * - SYCL kernel generation for each layer of computations
 * - Memory management for device-side data structures
 * - Execution scheduling and synchronization
 * - Tally computation with confidence intervals
 *
 * @tparam bitpack_t_ Integer type for bit packing operations (default: std::uint64_t)
 * @tparam prob_t_ Floating-point type for probability calculations (default: std::double_t)
 * @tparam size_t_ Integer type for size and indexing (default: std::uint32_t)
 *
 * @example Basic usage:
 * @code
 * // Create a layer manager for a pdag
 * fuzzybits::mc::executor::layer_manager<> manager(pdag, 1024, 16);
 *
 * // Submit all computations and wait for completion
 * manager.submit_all().wait_and_throw();
 *
 * // Get tally results for a specific event
 * auto tally_result = manager.tally(event_index, 1000);
 * std::cout << "Probability: " << tally_result.mean << " ± " << tally_result.std_err << std::endl;
 * @endcode
 *
 * @note This class requires SYCL-compatible hardware and runtime
 * @warning The class manages GPU memory and should be used with proper exception handling
 *
 * @since Version 1.0
 */
template <typename bitpack_t_, typename prob_t_, typename size_t_> class layer_manager {

    /// @brief Index type for node identification
    using index_t_ = std::int32_t;

    /// @brief The engine's integer choices for its node blocks: bitpacks, PDAG indices, and 32-bit extents.
    using node_types = event::node_types<bitpack_t_, index_t_, std::uint32_t>;

    /// @brief SYCL queue for device operations and kernel execution
    sycl::queue queue_{make_throwing_sycl_queue()};

    /// @brief Drops the queue's recorded asynchronous errors when the manager goes. Declared right after the queue, so
    /// that it is destroyed after every other member has freed its USM, and also when the constructor throws: a later
    /// run on this thread must not see this run's errors (make_throwing_sycl_queue records them per thread).
    struct async_error_drop {
        sycl::queue &queue;
        explicit async_error_drop(sycl::queue &q) noexcept : queue(q) {}
        async_error_drop(const async_error_drop &) = delete;
        async_error_drop &operator=(const async_error_drop &) = delete;
        ~async_error_drop() {
            try {
                queue.wait();
                queue.throw_asynchronous();
            } catch (...) {
            }
            sycl_async_error_slot().clear();
        }
    } async_error_drop_{queue_};

    /// @brief Sample shape configuration defining batch size and bitpack dimensions
    event::sample_shape<size_t_> sample_shape_;

    sample_shaper<bitpack_t_> sample_shaper_{};

    /// @brief The run's seed and device: with a node and a bitpack index, they select its random bits (prng/stream.h).
    prng::stream stream_{};

    /// @brief The PDAG's nodes in topological layers (compat.h): what the kernels compute, layer by layer.
    compat::graph graph_;

    /// @brief Every node's output for one pass, one arena slot per node. Declared before the program, whose blocks
    /// hold views of it, so that it is destroyed after them.
    event::output_arena<node_types> arena_;

    /// @brief The kernels of a pass in submission order, the node blocks they read, the tallies, and their plan
    /// (builder/program.h).
    builder::program<node_types> program_;

    /// @brief Passes submitted so far; pass p samples bitpacks [p·B, (p + 1)·B) of every node's run.
    std::uint32_t passes_ = 0;

    /// The program of the layers and of the tallies of `to_tally`, built for this manager's queue and arena.
    [[nodiscard]] builder::program<node_types> build_program(const stats::TallyNodeMap &to_tally);

    /// The tally of node `evt_idx` and its estimate. Throws std::out_of_range for a node without a tally.
    event::tally_estimate fetch_tally_for_event_with_index(index_t_ evt_idx);

  public:
    /// The plan this manager reports (descriptor/execution_plan.h), with its sample shaper's description.
    using execution_plan_description = descriptor::execution_plan<typename sample_shaper<bitpack_t_>::description>;

    /**
     * @brief Constructs a layer manager for the given PDAG
     *
     * @details Initializes the layer manager with the specified computation parameters,
     * performs topological sorting of the PDAG, and builds all necessary kernels for
     * parallel execution. The sample shape is automatically rounded to device-optimal
     * dimensions.
     *
     * @throws std::runtime_error if PDAG processing fails
     * @throws std::runtime_error if kernel building fails
     *
     * @note The constructor performs all setup operations synchronously
     * @note Sample shape is optimized for the target SYCL device
     *
     * @param stream the run's seed and device (prng/stream.h); the default is the default seed on device 0.
     */
    layer_manager(compat::pdag *pdag, size_t_ num_trials, const stats::TallyNodeMap &to_tally, std::size_t requested_trials_per_pass, prng::stream stream = {});

    /**
     * @brief Submits all queued computations to the SYCL device
     *
     * @details Submits all built kernels to the SYCL queue for execution. Returns
     * the SYCL queue object that can be used for synchronization and error checking.
     * This method does not block; use wait_and_throw() on the returned queue for
     * synchronous execution.
     *
     * @return SYCL queue object for synchronization and error handling
     *
     * @note This method is non-blocking; kernels execute asynchronously
     * @note Call wait_and_throw() on the returned queue to ensure completion
     *
     * @example
     * @code
     * // Submit and wait for completion
     * manager.submit_all().wait_and_throw();
     *
     * // Submit asynchronously
     * auto queue = manager.single_pass();
     * // Do other work...
     * queue.wait_and_throw();
     * @endcode
     */
    sycl::queue single_pass();

    sycl::queue pass(size_t count = 1);

    /// Passes submitted so far.
    [[nodiscard]] std::uint32_t passes() const noexcept { return passes_; }

    /// Trials per node over the passes submitted so far: passes × bitpacks per pass × bits per bitpack. The tallies'
    /// counts cover these trials once the queue is idle.
    [[nodiscard]] std::uint64_t total_bits() const noexcept { return std::uint64_t{passes_} * sample_shape_.num_bitpacks() * sizeof(bitpack_t_) * 8; }

    /// The run's seed and device.
    [[nodiscard]] const prng::stream &stream() const noexcept { return stream_; }

    /**
     * @brief Computes tally statistics for a specific event
     *
     * @details Performs Monte Carlo sampling to compute probability statistics for
     * the specified event, including mean probability, standard error, and confidence
     * intervals. The computation is repeated for the specified count to improve accuracy.
     *
     * @param evt_idx Index of the event to compute tally for
     * @param count Number of tally computations to perform
     *
     * @return tally_event structure containing computed statistics
     * @retval tally_event::mean Mean probability estimate
     * @retval tally_event::std_err Standard error of the estimate
     * @retval tally_event::ci[0] 5th percentile confidence bound
     * @retval tally_event::ci[1] 95th percentile confidence bound
     * @retval tally_event::num_one_bits Number of positive outcomes
     *
     * @note This method blocks until all computations complete
     * @throws std::out_of_range if the event has no tally
     *
     * @example
     * @code
     * // Compute tally for event 42
     * auto result = manager.tally(42);
     *
     * if (result.mean > 0.0) {
     *     std::cout << "Event 42 probability: " << result.mean
     *               << " ± " << result.std_err << std::endl;
     *     std::cout << "95% CI: [" << result.ci[0] << ", " << result.ci[1] << "]" << std::endl;
     * }
     * @endcode
     */
    event::tally_estimate single_pass_and_tally(index_t_ evt_idx);

    stats::TallyNodeMap &pass_wait_collect(stats::TallyNodeMap &stats, std::size_t total_passes = 1, std::size_t passes_between_waits = 0);

    stats::TallyNodeMap &collect_tallies(stats::TallyNodeMap &stats);

    [[nodiscard]] std::size_t node_count() const;

    /// The PDAG's nodes grouped per layer by the kind of kernel that computes them (descriptor/kernel_dag.h).
    [[nodiscard]] descriptor::kernel_dag describe_kernel_dag() const;

    /// The execution plan of a pass: the device, the pass's shape, every partition and edge, and every node.
    [[nodiscard]] execution_plan_description describe_execution_plan() const;

    /**
     * @brief Retrieves the original MEF event (BasicEvent) corresponding to a PDAG node index.
     *
     * Given a PDAG node index, this function resolves the underlying MEF event that was
     * used to construct the node.  At the moment only Variable nodes map directly back
     * to MEF::BasicEvent instances.  If the index is unknown or the mapping cannot be
     * resolved an exception is thrown.
     */
    [[nodiscard]] const compat::mef_event *get_mef_event(index_t_ event_id) const;

    /**
     * @brief Accessor for the internal sample shaper configuration
     *
     * @details Returns a const reference to the sample_shaper instance that
     * controls how the requested number of trials is split across device
     * iterations.  External convergence-management utilities can rely on
     * this accessor to obtain TOTAL_ITERATIONS as well as the chosen
     * SAMPLE_SHAPE.
     */
    [[nodiscard]] inline const sample_shaper<bitpack_t_> &shaper() const { return sample_shaper_; }
    [[nodiscard]] sycl::queue &queue() { return queue_; }

    // ------------------------------------------------------------------
    //  Logging helpers – grant access to scram::log::layers::csv_pairs
    // ------------------------------------------------------------------
    template <typename bt_, typename pt_, typename st_>
    friend std::vector<std::pair<std::string, std::string>> scram::log::layers::csv_pairs(const fuzzybits::mc::executor::layer_manager<bt_, pt_, st_> &);
    /**
     * @brief Waits for the device, then lets the members free their memory
     *
     * @details The arena and the blocks free their USM when they are destroyed, after this body; kernels in flight
     * may still read or write it, so the body waits on the queue first. Nothing throws from here.
     */
    ~layer_manager();
};
} // namespace fuzzybits::mc::executor

#include "layer_manager.tpp"
