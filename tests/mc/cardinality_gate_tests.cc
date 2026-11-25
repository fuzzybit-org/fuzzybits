// Device tests of the cardinality gate kernels (src/mc/kernel/cardinality.h): every slot count of the fixed
// form, with and without padding slots, and the counter and wide forms, in both the GPU and the CPU (lanes)
// form, at every bitpack width, against a host count. The gates cover at least, at most,
// exactly and other ranges, no inputs, k = 0, k > n, low > high, both polarities, and the 256-or-more inputs
// and thresholds that an 8-bit count cannot hold. Each gate's inputs have a density near its bounds, so both
// outputs occur.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "mc/event/node.h"
#include "mc/event/sample_shape.h"
#include "mc/kernel/cardinality.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

// Widths of 128 and 256 bits are rejected at compile time; see
// bitpack_width_compile_fail.cc and the mc.compile_fail.* tests.

namespace {

using size_t_ = std::size_t;
using scram::mc::kernel::cardinality_form;
using scram::mc::kernel::gate_order;

struct gate_spec {
    std::size_t positive;
    std::size_t negated;
    std::size_t low;
    std::size_t high; // >= positive + negated: no upper bound
    unsigned density; // counted inputs are true with probability density / 8
};

const std::vector<gate_spec> kGates{
    // fixed: slot counts 1, 2, 3, 4, 6, 8, 12 and 16, exact and padded (0 in 1, 5 in 6, 7 in 8, 9 in 12, 13
    // and 14 in 16)
    {0, 0, 0, 0, 4},
    {0, 0, 1, 1, 4},
    {1, 0, 1, 1, 4},
    {0, 1, 1, 1, 4},
    {1, 1, 1, 1, 4},
    {2, 0, 2, 2, 4},
    {3, 0, 2, 3, 4},
    {2, 1, 2, 3, 4},
    {0, 3, 2, 3, 4},
    {3, 0, 0, 1, 4},
    {2, 2, 2, 2, 4},
    {4, 0, 0, 4, 4},
    {3, 1, 5, 9, 4},
    {3, 2, 3, 2, 4},
    {3, 2, 2, 3, 4},
    {4, 2, 2, 6, 4},
    {7, 0, 0, 3, 4},
    {5, 3, 4, 8, 4},
    {8, 0, 1, 8, 2},
    {7, 1, 8, 8, 7},
    {9, 0, 5, 9, 4},
    {5, 4, 0, 4, 4},
    {6, 6, 6, 6, 4},
    {13, 0, 7, 13, 4},
    {7, 7, 5, 9, 4},
    {16, 0, 8, 8, 4},
    // counter
    {9, 8, 9, 17, 4},
    {12, 20, 16, 16, 4},
    {40, 24, 1, 64, 1},
    {0, 64, 64, 64, 7},
    {100, 155, 120, 135, 4},
    // wide
    {256, 0, 128, 256, 4},
    {150, 150, 150, 300, 4},
    {200, 100, 262, 300, 7},
    {257, 0, 0, 255, 7},
    {300, 300, 290, 310, 4},
    {1000, 24, 512, 512, 4},
};

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

template <typename bitpack_t, cardinality_form form_, std::uint32_t slots_, bool lanes_form_>
void check_form(sycl::queue &queue, const scram::mc::event::sample_shape<size_t_> &shape, const gate_order order) {
    constexpr std::size_t kBits = sizeof(bitpack_t) * 8;
    const std::size_t words = shape.num_bitpacks();

    std::vector<const gate_spec *> specs;
    for (const auto &g : kGates) {
        const auto inputs = g.positive + g.negated;
        if (scram::mc::kernel::cardinality_form_for(inputs) == form_ &&
            scram::mc::kernel::cardinality_slots_for(inputs) == slots_) {
            specs.push_back(&g);
        }
    }
    REQUIRE(!specs.empty());

    // Host inputs per gate: positive ones at the gate's density, negated ones at its complement, so that every
    // counted input is true with probability density / 8. One device allocation holds them all.
    std::size_t total_inputs = 0;
    for (const auto *g : specs) {
        total_inputs += g->positive + g->negated;
    }
    std::vector<std::vector<std::vector<bitpack_t>>> host(specs.size());
    bitpack_t *device = sycl::malloc_device<bitpack_t>(total_inputs * words, queue);
    std::vector<std::pair<std::vector<bitpack_t *>, size_t_>> inputs_per_gate;
    std::vector<std::pair<size_t_, size_t_>> bounds_per_gate;
    std::size_t slot = 0;
    for (std::size_t s = 0; s < specs.size(); ++s) {
        const auto &g = *specs[s];
        std::vector<bitpack_t *> ptrs;
        for (std::size_t i = 0; i < g.positive + g.negated; ++i, ++slot) {
            const unsigned eighths = i < g.positive ? g.density : 8 - g.density;
            auto &h = host[s].emplace_back(words);
            for (std::size_t w = 0; w < words; ++w) {
                h[w] = random_word<bitpack_t>(eighths, slot * words + w);
            }
            bitpack_t *d = device + slot * words;
            queue.memcpy(d, h.data(), words * sizeof(bitpack_t));
            ptrs.push_back(d);
        }
        inputs_per_gate.emplace_back(std::move(ptrs), g.negated);
        bounds_per_gate.emplace_back(g.low, g.high);
    }
    queue.wait_and_throw();

    auto block = scram::mc::event::create_cardinality_gate_block<bitpack_t, size_t_>(queue, inputs_per_gate,
                                                                                     bounds_per_gate, words, slots_);
    using kernel_t = scram::mc::kernel::cardinality<bitpack_t, size_t_, form_, slots_, lanes_form_>;
    const kernel_t kernel(block, shape, order);
    queue.submit([&](sycl::handler &cgh) { cgh.parallel_for(kernel.get_range(), kernel); });
    queue.wait_and_throw();

    std::vector<bitpack_t> out(specs.size() * words);
    queue.memcpy(out.data(), block.buffers, out.size() * sizeof(bitpack_t)).wait_and_throw();

    std::size_t mismatches = 0;
    std::size_t ones = 0;
    for (std::size_t s = 0; s < specs.size(); ++s) {
        const auto &g = *specs[s];
        for (std::size_t w = 0; w < words; ++w) {
            bitpack_t want = 0;
            for (std::size_t bit = 0; bit < kBits; ++bit) {
                std::size_t count = 0;
                for (std::size_t i = 0; i < g.positive + g.negated; ++i) {
                    const bool value = ((host[s][i][w] >> bit) & 1u) != 0;
                    count += (i < g.positive) == value ? 1 : 0;
                }
                if (count >= g.low && count <= g.high) {
                    want = static_cast<bitpack_t>(want | (bitpack_t{1} << bit));
                }
            }
            const bitpack_t got = out[s * words + w];
            ones += static_cast<std::size_t>(std::popcount(want));
            if (got != want && ++mismatches <= 8) {
                UNSCOPED_INFO("gate n=" << g.positive + g.negated << " negated=" << g.negated << " [" << g.low << ", "
                                        << g.high << "] word=" << w << std::hex << " got=0x"
                                        << static_cast<std::uint64_t>(got) << " want=0x"
                                        << static_cast<std::uint64_t>(want));
            }
        }
    }
    CHECK(mismatches == 0);
    // Both outputs occur, so a kernel that wrote a constant would fail.
    CHECK(ones > 0);
    CHECK(ones < specs.size() * words * kBits);

    scram::mc::event::destroy_cardinality_gate_block(queue, block);
    sycl::free(device, queue);
}

} // namespace

TEMPLATE_TEST_CASE("cardinality kernels match a host count in every form", "[mc][cardinality]", std::uint8_t,
                   std::uint16_t, std::uint32_t, std::uint64_t) {
    using bitpack_t = TestType;
    // 37 bitpacks per batch leaves a tail for the lanes form; batches are contiguous in a gate's buffer.
    const std::size_t batch_size = GENERATE(1, 8);
    const std::size_t bitpacks_per_batch = GENERATE(4, 37);
    const scram::mc::event::sample_shape<size_t_> shape{batch_size, bitpacks_per_batch};
    const auto order = GENERATE(gate_order::gate_major, gate_order::block_major);

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>() << " bits=" << sizeof(bitpack_t) * 8
                   << " bitpacks=" << shape.num_bitpacks() << " block_major=" << (order == gate_order::block_major));

    SECTION("fixed") {
        // Every slot count, as the kernel builder pads gates into them.
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (check_form<bitpack_t, cardinality_form::fixed, scram::mc::kernel::kCardinalitySlots[I], false>(
                 queue, shape, order),
             ...);
            (check_form<bitpack_t, cardinality_form::fixed, scram::mc::kernel::kCardinalitySlots[I], true>(queue, shape,
                                                                                                           order),
             ...);
        }(std::make_index_sequence<scram::mc::kernel::kCardinalitySlots.size()>{});
    }
    SECTION("counter") {
        check_form<bitpack_t, cardinality_form::counter, 0, false>(queue, shape, order);
        check_form<bitpack_t, cardinality_form::counter, 0, true>(queue, shape, order);
    }
    SECTION("wide") {
        check_form<bitpack_t, cardinality_form::wide, 0, false>(queue, shape, order);
        check_form<bitpack_t, cardinality_form::wide, 0, true>(queue, shape, order);
    }
}

TEST_CASE("cardinality forms and slot counts by input count", "[mc][cardinality]") {
    using scram::mc::kernel::cardinality_form_for;
    using scram::mc::kernel::cardinality_slots_for;
    CHECK(cardinality_form_for(0) == cardinality_form::fixed);
    CHECK(cardinality_form_for(16) == cardinality_form::fixed);
    CHECK(cardinality_form_for(17) == cardinality_form::counter);
    CHECK(cardinality_form_for(255) == cardinality_form::counter);
    CHECK(cardinality_form_for(256) == cardinality_form::wide);
    CHECK(cardinality_slots_for(0) == 1);
    CHECK(cardinality_slots_for(3) == 3);
    CHECK(cardinality_slots_for(5) == 6);
    CHECK(cardinality_slots_for(7) == 8);
    CHECK(cardinality_slots_for(9) == 12);
    CHECK(cardinality_slots_for(13) == 16);
    CHECK(cardinality_slots_for(17) == 0);
}
