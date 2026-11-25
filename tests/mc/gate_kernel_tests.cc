// Device tests of the AND, OR, XOR, NAND, NOR, NOT and NULL kernels (src/mc/kernel/gate.h) against a host
// evaluation: every slot count of the fixed form, with and without padding slots, the counted form at every
// input count (CPUs use it for every gate), both lane forms (GPU, CPU), both work-group orders, gates without
// inputs (the kernel builder's constants), and 8- to 64-bit bitpacks. Each gate's inputs have a density that
// leaves both outputs likely.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "mc/event/node.h"
#include "mc/event/sample_shape.h"
#include "mc/kernel/gate.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace {

using size_t_ = std::size_t;
using scram::core::Connective;
using scram::mc::kernel::gate_order;
using scram::mc::kernel::gate_slots_for;
using scram::mc::kernel::gate_slots_on;
using scram::mc::kernel::kGateSlots;

struct gate_spec {
    std::size_t positive;
    std::size_t negated;
};

// Fixed form: no inputs, then every slot count exact and padded (5 in 6, 7 in 8, 9 and 10 in 12, 13 and 14 in
// 16); counted form: all of these, and 17 to 100 inputs; both polarities.
const std::vector<gate_spec> kGates{
    {0, 0}, {1, 0}, {0, 1}, {1, 1},  {2, 1}, {3, 1}, {2, 3},   {4, 2},   {5, 2},   {6, 2},   {5, 4},
    {6, 4}, {6, 6}, {7, 6}, {10, 4}, {8, 8}, {9, 8}, {12, 12}, {20, 20}, {32, 32}, {60, 40},
};
const std::vector<gate_spec> kUnary{{1, 0}, {0, 1}};

constexpr bool is_unary(const Connective op) { return op == Connective::kNot || op == Connective::kNull; }

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// A word whose bits are 1 with probability eighths / 8.
template <typename bitpack_t> bitpack_t random_word(const unsigned eighths, const std::uint64_t seed) {
    std::uint64_t x = 0;
    for (unsigned digit = 0; digit < 3; ++digit) {
        const std::uint64_t r = splitmix64(seed * 3 + digit);
        x = ((eighths >> digit) & 1u) != 0 ? (x | r) : (x & r);
    }
    return static_cast<bitpack_t>(x);
}

// Counted inputs are true with probability density(op) / 8: AND-like gates high, OR-like low.
constexpr unsigned density(const Connective op) {
    if (op == Connective::kAnd || op == Connective::kNand) {
        return 7;
    }
    if (op == Connective::kOr || op == Connective::kNor) {
        return 1;
    }
    return 4;
}

template <typename bitpack_t>
bitpack_t host_gate(const Connective op, const std::vector<bitpack_t> &inputs, const std::size_t positive) {
    const bool and_like = op == Connective::kAnd || op == Connective::kNand;
    bitpack_t acc = and_like ? static_cast<bitpack_t>(~bitpack_t{0}) : bitpack_t{0};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto x = i < positive ? inputs[i] : static_cast<bitpack_t>(~inputs[i]);
        if (and_like) {
            acc = static_cast<bitpack_t>(acc & x);
        } else if (op == Connective::kXor) {
            acc = static_cast<bitpack_t>(acc ^ x);
        } else if (op == Connective::kOr || op == Connective::kNor) {
            acc = static_cast<bitpack_t>(acc | x);
        } else {
            acc = x;
        }
    }
    if (op == Connective::kNand || op == Connective::kNor || op == Connective::kNot) {
        acc = static_cast<bitpack_t>(~acc);
    }
    return acc;
}

template <typename bitpack_t, Connective op_, std::uint32_t slots_, bool lanes_form_>
void check(sycl::queue &queue, const scram::mc::event::sample_shape<size_t_> &shape, const gate_order order) {
    constexpr std::size_t kBits = sizeof(bitpack_t) * 8;
    const std::size_t words = shape.num_bitpacks();
    std::vector<const gate_spec *> specs;
    for (const auto &g : is_unary(op_) ? kUnary : kGates) {
        if (slots_ == 0 || gate_slots_for(g.positive + g.negated) == slots_) {
            specs.push_back(&g);
        }
    }
    REQUIRE(!specs.empty());

    std::size_t total_inputs = 0;
    for (const auto *g : specs) {
        total_inputs += g->positive + g->negated;
    }
    // Inputs at density d (positive) or 8 - d (negated), so that every counted input is true with probability
    // d / 8. One allocation for all of them (at least one word).
    std::vector<std::vector<std::vector<bitpack_t>>> host(specs.size());
    bitpack_t *device = sycl::malloc_device<bitpack_t>(std::max<std::size_t>(total_inputs, 1) * words, queue);
    std::vector<std::pair<std::vector<bitpack_t *>, size_t_>> inputs_per_gate;
    std::size_t slot = 0;
    for (std::size_t s = 0; s < specs.size(); ++s) {
        const auto &g = *specs[s];
        std::vector<bitpack_t *> ptrs;
        for (std::size_t i = 0; i < g.positive + g.negated; ++i, ++slot) {
            const unsigned eighths = i < g.positive ? density(op_) : 8 - density(op_);
            auto &h = host[s].emplace_back(words);
            for (std::size_t w = 0; w < words; ++w) {
                h[w] = random_word<bitpack_t>(eighths, slot * words + w);
            }
            queue.memcpy(device + slot * words, h.data(), words * sizeof(bitpack_t));
            ptrs.push_back(device + slot * words);
        }
        inputs_per_gate.emplace_back(std::move(ptrs), g.negated);
    }
    queue.wait_and_throw();

    auto block = scram::mc::event::create_gate_block<bitpack_t, size_t_>(queue, inputs_per_gate, words, slots_);
    using kernel_t = scram::mc::kernel::op<op_, bitpack_t, size_t_, slots_, lanes_form_>;
    const kernel_t kernel(block, shape, order);
    queue.submit([&](sycl::handler &cgh) { cgh.parallel_for(kernel.get_range(), kernel); });
    queue.wait_and_throw();

    std::vector<bitpack_t> out(specs.size() * words);
    queue.memcpy(out.data(), block.buffers, out.size() * sizeof(bitpack_t)).wait_and_throw();

    std::size_t mismatches = 0;
    std::size_t ones = 0;
    for (std::size_t s = 0; s < specs.size(); ++s) {
        std::vector<bitpack_t> column(host[s].size());
        for (std::size_t w = 0; w < words; ++w) {
            for (std::size_t i = 0; i < column.size(); ++i) {
                column[i] = host[s][i][w];
            }
            const bitpack_t want = host_gate(op_, column, specs[s]->positive);
            const bitpack_t got = out[s * words + w];
            ones += static_cast<std::size_t>(std::popcount(want));
            if (got != want && ++mismatches <= 8) {
                UNSCOPED_INFO("op=" << static_cast<int>(op_) << " slots=" << slots_
                                    << " n=" << specs[s]->positive + specs[s]->negated << " word=" << w << std::hex
                                    << " got=0x" << static_cast<std::uint64_t>(got) << " want=0x"
                                    << static_cast<std::uint64_t>(want));
            }
        }
    }
    CHECK(mismatches == 0);
    CHECK(ones > 0);
    CHECK(ones < specs.size() * words * kBits);

    scram::mc::event::destroy_gate_block(queue, block);
    sycl::free(device, queue);
}

template <typename bitpack_t, Connective op_, bool lanes_form_, std::size_t... I>
void check_slot_counts(sycl::queue &queue, const scram::mc::event::sample_shape<size_t_> &shape, const gate_order order,
                       std::index_sequence<I...>) {
    (check<bitpack_t, op_, kGateSlots[I], lanes_form_>(queue, shape, order), ...);
}

// Every slot count and the counted form, in both lane forms.
template <typename bitpack_t, Connective op_>
void check_all(sycl::queue &queue, const scram::mc::event::sample_shape<size_t_> &shape, const gate_order order) {
    if constexpr (is_unary(op_)) {
        check<bitpack_t, op_, 1, false>(queue, shape, order);
        check<bitpack_t, op_, 1, true>(queue, shape, order);
    } else {
        constexpr auto each = std::make_index_sequence<kGateSlots.size()>{};
        check_slot_counts<bitpack_t, op_, false>(queue, shape, order, each);
        check_slot_counts<bitpack_t, op_, true>(queue, shape, order, each);
        check<bitpack_t, op_, 0, false>(queue, shape, order);
        check<bitpack_t, op_, 0, true>(queue, shape, order);
    }
}

// Exact, padded and counted: 1, 3 and 16 slots, and the counted form.
template <typename bitpack_t, Connective op_>
void check_some(sycl::queue &queue, const scram::mc::event::sample_shape<size_t_> &shape, const gate_order order) {
    check<bitpack_t, op_, 1, false>(queue, shape, order);
    check<bitpack_t, op_, 3, true>(queue, shape, order);
    check<bitpack_t, op_, 16, false>(queue, shape, order);
    check<bitpack_t, op_, 16, true>(queue, shape, order);
    check<bitpack_t, op_, 0, false>(queue, shape, order);
    check<bitpack_t, op_, 0, true>(queue, shape, order);
}

} // namespace

TEST_CASE("gate kernels match a host evaluation", "[mc][gate]") {
    // 37 bitpacks per batch leaves a tail for the lanes form; batches are contiguous in a gate's buffer.
    const std::size_t batch_size = GENERATE(1, 8);
    const std::size_t bitpacks_per_batch = GENERATE(4, 37);
    const scram::mc::event::sample_shape<size_t_> shape{batch_size, bitpacks_per_batch};
    const auto order = GENERATE(gate_order::gate_major, gate_order::block_major);

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>() << " bitpacks=" << shape.num_bitpacks()
                   << " block_major=" << (order == gate_order::block_major));

    SECTION("AND, every slot count, 64- and 8-bit") {
        check_all<std::uint64_t, Connective::kAnd>(queue, shape, order);
        check_all<std::uint8_t, Connective::kAnd>(queue, shape, order);
    }
    SECTION("XOR, every slot count, 16-bit") { check_all<std::uint16_t, Connective::kXor>(queue, shape, order); }
    SECTION("OR, NAND, NOR, XOR") {
        check_some<std::uint64_t, Connective::kOr>(queue, shape, order);
        check_some<std::uint64_t, Connective::kNand>(queue, shape, order);
        check_some<std::uint64_t, Connective::kNor>(queue, shape, order);
        check_some<std::uint32_t, Connective::kXor>(queue, shape, order);
    }
    SECTION("NOT and NULL") {
        check_all<std::uint64_t, Connective::kNot>(queue, shape, order);
        check_all<std::uint64_t, Connective::kNull>(queue, shape, order);
        check_all<std::uint8_t, Connective::kNot>(queue, shape, order);
    }
}

TEST_CASE("gate slot counts by input count", "[mc][gate]") {
    CHECK(gate_slots_for(0) == 1);
    CHECK(gate_slots_for(1) == 1);
    CHECK(gate_slots_for(5) == 6);
    CHECK(gate_slots_for(9) == 12);
    CHECK(gate_slots_for(16) == 16);
    CHECK(gate_slots_for(17) == 0);

    // The kernel builder's choice: one counted kernel per connective on CPUs, the fixed form on GPUs.
    const sycl::queue queue;
    const auto device = queue.get_device();
    const bool cpu = device.is_cpu();
    INFO("device=" << device.get_info<sycl::info::device::name>());
    CHECK(gate_slots_on(device, Connective::kAnd, 3) == (cpu ? 0u : 3u));
    CHECK(gate_slots_on(device, Connective::kOr, 0) == (cpu ? 0u : 1u));
    CHECK(gate_slots_on(device, Connective::kXor, 40) == 0);
    CHECK(gate_slots_on(device, Connective::kNot, 1) == 1);
    CHECK(gate_slots_on(device, Connective::kNull, 1) == 1);
}
