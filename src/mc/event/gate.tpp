#pragma once

#include "mc/event/node.tpp"
#include <sycl/sycl.hpp>

namespace scram::mc::event {

/**
 * @struct gate
 * @brief Represents a logical gate with multiple inputs and configurable logic
 *
 * @details A gate performs logical operations on multiple input signals, supporting
 * both positive and negated inputs. The gate structure is designed for efficient
 * parallel processing of boolean operations in pdags.
 *
 * Gates can handle mixed positive and negative logic by specifying an offset where
 * negative inputs begin. This allows for efficient representation of complex logical
 * expressions without requiring separate NOT gates.
 *
 * @tparam bitpack_t_ Integer type for bit-packed data storage
 * @tparam size_t_ Integer type for size and indexing
 *
 * @note Input buffers are stored as an array of pointers for indirect access
 * @note The negated_inputs_offset defines the boundary between positive and negative inputs
 *
 * @example
 * @code
 * // Gate with 3 positive inputs and 2 negated inputs
 * gate<std::uint64_t, std::uint32_t> or_gate;
 * or_gate.num_inputs = 5;
 * or_gate.negated_inputs_offset = 3;  // inputs[0-2] positive, inputs[3-4] negated
 * @endcode
 */
template <typename bitpack_t_, typename size_t_> struct gate : node<bitpack_t_> {
    /// @brief Array of pointers to input buffers from other nodes
    bitpack_t_ **inputs;

    /// @brief Total number of input connections to this gate
    size_t_ num_inputs;

    /// @brief Offset where negated inputs begin (inputs[0..offset-1] are positive, inputs[offset..num_inputs-1] are
    /// negated)
    size_t_ negated_inputs_offset;

    // TODO:: input_mask's type should follow the appropriate value it's tracking's type, and not be a default.
    /// @brief Kernels with a fixed number of input slots: bit j is set iff slot j holds an input. The block pads
    /// a gate's inputs with null pointers to the slot count, and the kernels never read a slot whose bit is clear.
    std::uint64_t input_mask = 0;
};

/**
 * @struct cardinality_gate
 * @brief Gate that is true where the number of its true inputs lies in [low, high]
 *
 * @details Inputs at or after negated_inputs_offset count when they are false. high >= num_inputs means no
 * upper bound, so at least k is [k, n], at most k is [0, k] and exactly k is [k, k]; low > high is never
 * true. kernel::cardinality evaluates these gates.
 *
 * Gates of the kernel's fixed form hold a fixed number of input slots (kernel::kCardinalitySlots), padded
 * after the inputs (see gate::input_mask). num_inputs is the number of inputs, not of slots.
 *
 * @example
 * @code
 * // 2-out-of-3 voting gate
 * cardinality_gate<std::uint64_t, std::uint32_t> voting_gate;
 * voting_gate.num_inputs = 3;
 * voting_gate.negated_inputs_offset = 3;  // all inputs positive
 * voting_gate.low = 2;
 * voting_gate.high = 3;
 * @endcode
 */
template <typename bitpack_t_, typename size_t_> struct cardinality_gate : gate<bitpack_t_, size_t_> {
    size_t_ low = 0;
    size_t_ high = 0;
};

/**
 * @brief Destroys a gate and frees associated device memory.
 *
 * @details Properly releases device memory allocated for a gate structure.
 * This function should be called for all gates created with create_atleast_gates()
 * to prevent memory leaks.
 *
 * @tparam bitpack_t_ Integer type for bit-packed data storage
 *
 * @param queue SYCL queue for memory deallocation
 * @param gate Pointer to gate to destroy
 */
template <typename bitpack_t_, typename size_t_>
void destroy_gate(const sycl::queue queue, gate<bitpack_t_, size_t_> *gate) {
    sycl::free(gate->inputs, queue);
    sycl::free(gate, queue);
}

// ---------------------------------------------------------------------
//  Gate block wrappers (one contiguous allocation of standard `gate`s)
// ---------------------------------------------------------------------

/**
 * @brief Thin wrapper describing one contiguous allocation of `gate` nodes.
 *
 * In addition to the base `node_block` data/size pair, a `gate_block` owns
 * a single device allocation that stores all bit-packed output buffers for
 * every gate in the block.  Each individual gate’s `buffer` member points
 * somewhere *inside* this big allocation at an offset of
 * `gate_index * bitpacks_per_gate`.
 */
template <typename bitpack_t_, typename size_t_> struct gate_block : public node_block<gate<bitpack_t_, size_t_>> {
    /* contiguous device block that stores every gate’s computed output */
    bitpack_t_ *buffers = nullptr;

    /* single USM-shared array containing *all* input buffer pointers
     * for *every* gate in this block.  Each gate’s `inputs` member is
     * simply a slice into this big array.                           */
    bitpack_t_ **all_inputs = nullptr;
    // TODO:: needs to be its own templated (no default type)
    std::size_t total_inputs = 0; ///< length of all_inputs
    // TODO:: needs to be its own templated (no default type)
    std::size_t bitpacks_per_gate = 0; ///< stride between successive gate buffers
};

/**
 * @brief Allocates and initializes a contiguous block of `gate` objects and
 * their shared output-buffer block.
 *
 * `inputs_per_gate` follows the same convention used previously in
 * `create_gates`: each entry is a pair whose first element is the *ordered*
 * vector of input buffers and whose second element is the number of *negated*
 * inputs at the tail of that vector. With `slots` > 0 (at most 64), every gate
 * gets that many input slots: its inputs, then null pointers, with
 * `input_mask` marking the inputs.
 */
template <typename bitpack_t_, typename size_t_>
[[nodiscard]]
gate_block<bitpack_t_, size_t_>
create_gate_block(const sycl::queue &queue,
                  const std::vector<std::pair<std::vector<bitpack_t_ *>, size_t_>> &inputs_per_gate,
                  const std::size_t num_bitpacks, const std::size_t slots = 0) {
    const auto num_gates = inputs_per_gate.size();
    // TODO:: no assert!
    assert(slots <= 64);

    using gate_t = gate<bitpack_t_, size_t_>;

    /* First pass: count total unique pointer slots needed */
    std::size_t total_input_ptrs = 0;
    for (const auto &g : inputs_per_gate) {
        assert(slots == 0 || g.first.size() <= slots);
        total_input_ptrs += slots > 0 ? slots : g.first.size();
    }
    //TODO:: This value should be device-scoped once we support multiple devices
    using cache_line_bytes = sycl::info::device::global_mem_cache_line_size;
    const auto align = queue.get_device().get_info<cache_line_bytes>();

    // 1) Allocate primary USM-shared structures. At least one pointer slot: a zero-size shared allocation
    //    fails on HIP (#277), and a block may hold only gates without inputs (constants).
    gate_t *gates = sycl::aligned_alloc_shared<gate_t>(align, num_gates, queue);
    bitpack_t_ **all_inputs =
        sycl::aligned_alloc_shared<bitpack_t_ *>(align, std::max<std::size_t>(total_input_ptrs, 1), queue);

    // 2) Allocate single device block for all gate outputs.
    bitpack_t_ *buffer_block = sycl::aligned_alloc_device<bitpack_t_>(align, num_gates * num_bitpacks, queue);

    // 3) Populate gate structs and fill the all_inputs array.
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < num_gates; ++i) {
        const auto &gate_inputs = inputs_per_gate[i].first;
        const std::size_t n_in = gate_inputs.size();
        const std::size_t n_neg = inputs_per_gate[i].second;

        assert(n_neg <= n_in);

        // Slice in all_inputs where this gate’s pointers will live.
        gates[i].inputs = all_inputs + cursor;
        gates[i].num_inputs = static_cast<size_t_>(n_in);
        gates[i].negated_inputs_offset = static_cast<size_t_>(n_in - n_neg);
        gates[i].buffer = buffer_block + i * num_bitpacks;
        gates[i].input_mask = n_in >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << n_in) - 1;

        // Copy actual buffer pointers, then the padding.
        std::copy(gate_inputs.begin(), gate_inputs.end(), all_inputs + cursor);
        const std::size_t width = slots > 0 ? slots : n_in;
        std::fill(all_inputs + cursor + n_in, all_inputs + cursor + width, nullptr);
        cursor += width;
    }

    // 4) Wrap and return.
    gate_block<bitpack_t_, size_t_> blk;
    blk.data = gates;
    blk.count = num_gates;
    blk.buffers = buffer_block;
    blk.all_inputs = all_inputs;
    blk.total_inputs = total_input_ptrs;
    blk.bitpacks_per_gate = num_bitpacks;
    return blk;
}

/**
 * @brief Releases all device memory owned by a `gate_block`.
 *
 * Frees, in order:
 *   1. Each per-gate `inputs` array
 *   2. The shared output-buffer block
 *   3. The contiguous array of `gate` structures
 */
template <typename bitpack_t_, typename size_t_>
void destroy_gate_block(const sycl::queue &queue, gate_block<bitpack_t_, size_t_> &blk) {
    if (blk.all_inputs) {
        sycl::free(blk.all_inputs, queue);
        blk.all_inputs = nullptr;
    }

    if (blk.buffers) {
        sycl::free(blk.buffers, queue);
        blk.buffers = nullptr;
    }

    if (blk.data) {
        sycl::free(blk.data, queue);
        blk.data = nullptr;
    }

    blk.count = 0;
    blk.total_inputs = 0;
    blk.bitpacks_per_gate = 0;
}

// ---------------------------------------------------------------------------------
//  Cardinality gate block wrappers (one contiguous allocation of cardinality gates)
// ---------------------------------------------------------------------------------

/**
 * @brief Thin wrapper describing one contiguous allocation of `cardinality_gate` nodes.
 *
 * In addition to the base `node_block` data/size pair, a `cardinality_gate_block` owns
 * a single device allocation that stores all bit-packed output buffers for
 * every gate in the block.  Each individual gate’s `buffer` member points
 * somewhere *inside* this big allocation at an offset of
 * `gate_index * bitpacks_per_gate`.
 */
template <typename bitpack_t_, typename size_t_>
struct cardinality_gate_block : public node_block<cardinality_gate<bitpack_t_, size_t_>> {
    using gate_t = cardinality_gate<bitpack_t_, size_t_>;

    /* contiguous device block that stores every gate’s computed output */
    bitpack_t_ *buffers = nullptr;

    /* single USM-shared array containing *all* input buffer pointers
     * for *every* gate in this block.  Each gate’s `inputs` member is
     * simply a slice into this big array.                           */
    bitpack_t_ **all_inputs = nullptr;
    std::size_t total_inputs = 0; ///< length of all_inputs

    std::size_t bitpacks_per_gate = 0; ///< stride between successive gate buffers
};

/**
 * @brief Allocates and initialises a contiguous block of `cardinality_gate` objects and
 * their shared output-buffer block.
 *
 * `inputs_per_gate` follows the same convention used previously in
 * `create_gates`: each entry is a pair whose first element is the *ordered*
 * vector of input buffers and whose second element is the number of *negated*
 * inputs at the tail of that vector. `bounds_per_gate` holds each gate's
 * [low, high]; high >= the gate's input count means no upper bound. With
 * `slots` > 0 (at most 64), every gate gets that many input slots: its inputs,
 * then null pointers, with `input_mask` marking the inputs.
 */
template <typename bitpack_t_, typename size_t_>
[[nodiscard]]
cardinality_gate_block<bitpack_t_, size_t_>
create_cardinality_gate_block(const sycl::queue &queue,
                              const std::vector<std::pair<std::vector<bitpack_t_ *>, size_t_>> &inputs_per_gate,
                              const std::vector<std::pair<size_t_, size_t_>> &bounds_per_gate,
                              const std::size_t num_bitpacks, const std::size_t slots = 0) {
    const std::size_t num_gates = inputs_per_gate.size();
    assert(bounds_per_gate.size() == num_gates);
    assert(slots <= 64);

    using gate_t = cardinality_gate<bitpack_t_, size_t_>;

    /* First pass: count total unique pointer slots needed */
    std::size_t total_input_ptrs = 0;
    for (const auto &g : inputs_per_gate) {
        assert(slots == 0 || g.first.size() <= slots);
        total_input_ptrs += slots > 0 ? slots : g.first.size();
    }

    //TODO:: This value should be device-scoped once we support multiple devices
    using cache_line_bytes = sycl::info::device::global_mem_cache_line_size;
    const auto align = queue.get_device().get_info<cache_line_bytes>();

    // 1) Allocate primary USM-shared structures. At least one pointer slot: a zero-size shared allocation
    //    fails on HIP, and a block may hold only gates without inputs.
    gate_t *gates = sycl::aligned_alloc_shared<gate_t>(align, num_gates, queue);
    bitpack_t_ **all_inputs =
        sycl::aligned_alloc_shared<bitpack_t_ *>(align, std::max<std::size_t>(total_input_ptrs, 1), queue);

    // 2) Allocate single device block for all gate outputs.
    bitpack_t_ *buffer_block = sycl::aligned_alloc_device<bitpack_t_>(align, num_gates * num_bitpacks, queue);

    // 3) Populate gate structs and fill the all_inputs array.
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < num_gates; ++i) {
        const auto &gate_inputs = inputs_per_gate[i].first;
        const std::size_t n_in = gate_inputs.size();
        const std::size_t n_neg = inputs_per_gate[i].second;

        assert(n_neg <= n_in);

        // Slice in all_inputs where this gate’s pointers will live.
        gates[i].inputs = all_inputs + cursor;
        gates[i].num_inputs = static_cast<size_t_>(n_in);
        gates[i].negated_inputs_offset = static_cast<size_t_>(n_in - n_neg);
        gates[i].buffer = buffer_block + i * num_bitpacks;
        gates[i].low = bounds_per_gate[i].first;
        gates[i].high = bounds_per_gate[i].second;
        gates[i].input_mask = n_in >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << n_in) - 1;

        // Copy actual buffer pointers, then the padding.
        std::copy(gate_inputs.begin(), gate_inputs.end(), all_inputs + cursor);
        const std::size_t width = slots > 0 ? slots : n_in;
        std::fill(all_inputs + cursor + n_in, all_inputs + cursor + width, nullptr);
        cursor += width;
    }

    // 4) Wrap and return.
    cardinality_gate_block<bitpack_t_, size_t_> blk;
    blk.data = gates;
    blk.count = num_gates;
    blk.buffers = buffer_block;
    blk.all_inputs = all_inputs;
    blk.total_inputs = total_input_ptrs;
    blk.bitpacks_per_gate = num_bitpacks;
    return blk;
}

/**
 * @brief Releases all device memory owned by a `cardinality_gate_block`.
 *
 * Frees, in order:
 *   1. Each per-gate `inputs` array
 *   2. The shared output-buffer block
 *   3. The contiguous array of `gate` structures
 */
template <typename bitpack_t_, typename size_t_>
void destroy_cardinality_gate_block(const sycl::queue &queue, cardinality_gate_block<bitpack_t_, size_t_> &blk) {
    if (blk.all_inputs) {
        sycl::free(blk.all_inputs, queue);
        blk.all_inputs = nullptr;
    }

    if (blk.buffers) {
        sycl::free(blk.buffers, queue);
        blk.buffers = nullptr;
    }

    if (blk.data) {
        sycl::free(blk.data, queue);
        blk.data = nullptr;
    }

    blk.count = 0;
    blk.total_inputs = 0;
    blk.bitpacks_per_gate = 0;
}

}