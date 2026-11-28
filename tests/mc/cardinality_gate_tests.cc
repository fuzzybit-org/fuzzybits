// Device tests of the cardinality gate kernels (src/mc/kernel/cardinality.h): every slot count of the fixed form,
// with every slot used and with unused slots (fewer inputs than slots), and the counter and wide forms, in both the
// GPU and the CPU (lanes) form, in both work-group orders, at every bitpack width, against a host count. The gates
// cover at least, at most, exactly and other ranges, no inputs, k = 0, k > n, low > high, both polarities, and the
// 256-or-more inputs and thresholds that an 8-bit count cannot hold. Each gate's inputs have a density near its
// bounds, so both outputs occur.
//
// The inputs are in arena slots reserved before the gate block, each gate's at descending slots, so a kernel that does
// not look its inputs up in the block's input table reads the wrong slots. The gates' slots start as the complement of
// their expected bitpacks, so a bitpack the kernel does not write fails; the input slots, and a guard slot after the
// block, must be unchanged.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "mc/event/gate.h"
#include "mc/event/node.h"
#include "mc/kernel/cardinality.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <span>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

// Widths of 128 and 256 bits are rejected at compile time; see
// bitpack_width_compile_fail.cc and the mc.compile_fail.* tests.

namespace {

namespace ev = fuzzybits::mc::event;
using fuzzybits::mc::kernel::cardinality_form;
using fuzzybits::mc::kernel::gate_order;

template <typename Bitpack> using types = ev::node_types<Bitpack, std::int32_t, std::uint32_t>;

struct gate_spec {
    std::uint32_t positive;
    std::uint32_t negated;
    std::uint32_t low;
    std::uint32_t high; // >= positive + negated: no upper bound
    unsigned density;   // counted inputs are true with probability density / 8
};

const std::vector<gate_spec> kGates{
    // fixed: slot counts 1, 2, 3, 4, 6, 8, 12 and 16, every slot used and some unused (0 in 1, 5 in 6, 7 in 8,
    // 9 in 12, 13 and 14 in 16)
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
template <typename Bitpack> Bitpack random_word(const unsigned eighths, const std::uint64_t seed) {
    std::uint64_t x = 0;
    for (unsigned digit = 0; digit < 3; ++digit) {
        const std::uint64_t r = splitmix64(seed * 3 + digit);
        x = ((eighths >> digit) & 1u) != 0 ? (x | r) : (x & r);
    }
    return static_cast<Bitpack>(x);
}

// The gates of kGates whose input count takes form form_ and slot count slots_, as one block, evaluated on the device
// in lanes form lanes_form_ and in `order`, against a host count.
template <typename Bitpack, cardinality_form form_, std::uint32_t slots_, bool lanes_form_>
void check_form(sycl::queue &queue, const std::uint32_t bitpacks_per_node, const gate_order order) {
    using T = types<Bitpack>;
    constexpr std::size_t kBits = sizeof(Bitpack) * 8;
    const std::size_t words = bitpacks_per_node;
    INFO("slots=" << slots_ << " lanes_form=" << lanes_form_);

    // The gates, by position in kGates, which is also their PDAG index.
    std::vector<std::size_t> picked;
    std::size_t total_inputs = 0;
    for (std::size_t k = 0; k < kGates.size(); ++k) {
        const std::uint32_t n = kGates[k].positive + kGates[k].negated;
        if (fuzzybits::mc::kernel::cardinality_form_for(n) == form_ && fuzzybits::mc::kernel::cardinality_slots_for(n) == slots_) {
            picked.push_back(k);
            total_inputs += n;
        }
    }
    REQUIRE(!picked.empty());

    // The arena holds the inputs, then the gates, then a guard slot that the kernel must not write. `image` is the
    // arena as the host writes it before the kernel.
    auto arena = ev::output_arena<T>::create(queue, total_inputs + picked.size() + 1, words);
    std::vector<Bitpack> image(arena.view().size());
    const auto bitpacks_of = [&image, words](const std::uint32_t slot) { return std::span<Bitpack>(image).subspan(std::size_t{slot} * words, words); };

    // Input k takes slot inputs.end() - 1 - k. Positive inputs are true with the gate's density and negated ones with
    // its complement, so every counted input is true with probability density / 8.
    const auto inputs = arena.reserve(total_inputs);
    std::vector<std::vector<std::uint32_t>> input_slots(picked.size()); // per gate: positive, then negated
    ev::cardinality_gate_block_builder<T> builder;
    builder.reserve(picked.size(), total_inputs);
    std::uint32_t k = 0;
    for (std::size_t s = 0; s < picked.size(); ++s) {
        const gate_spec &g = kGates[picked[s]];
        for (std::uint32_t i = 0; i < g.positive + g.negated; ++i, ++k) {
            const auto slot = static_cast<std::uint32_t>(inputs.end() - 1 - k);
            const unsigned eighths = i < g.positive ? g.density : 8 - g.density;
            const auto bits = bitpacks_of(slot);
            for (std::size_t w = 0; w < words; ++w) {
                bits[w] = random_word<Bitpack>(eighths, k * words + w);
            }
            input_slots[s].push_back(slot);
        }
        const std::span<const std::uint32_t> slots(input_slots[s]);
        builder.add(static_cast<std::int32_t>(picked[s]), slots.first(g.positive), slots.subspan(g.positive), {.low = g.low, .high = g.high});
    }
    const auto block = builder.build(queue, arena);
    const auto guard = arena.reserve(1);
    REQUIRE(block.size() == picked.size());
    // The input table holds exactly the inputs: the fixed form reads slot j only when j < num_inputs, so a gate with
    // unused slots has no entries for them.
    CHECK(block.inputs().size() == total_inputs);
    {
        const auto bits = bitpacks_of(guard.first);
        for (std::size_t w = 0; w < words; ++w) {
            bits[w] = random_word<Bitpack>(4, k * words + w);
        }
    }

    // The host count. A gate's slot starts as the complement of its expected bitpacks, so a bitpack the kernel does
    // not write cannot match.
    std::vector<Bitpack> want(picked.size() * words);
    std::size_t ones = 0;
    for (std::uint32_t s = 0; s < block.size(); ++s) {
        const gate_spec &g = kGates[picked[s]];
        std::vector<std::span<const Bitpack>> in;
        for (const std::uint32_t slot : input_slots[s]) {
            in.emplace_back(bitpacks_of(slot));
        }
        const auto out = bitpacks_of(block.slot(s));
        std::vector<Bitpack> counted(in.size());
        for (std::size_t w = 0; w < words; ++w) {
            // Input i as it counts: complemented when negated.
            for (std::size_t i = 0; i < in.size(); ++i) {
                counted[i] = i < g.positive ? in[i][w] : static_cast<Bitpack>(~in[i][w]);
            }
            Bitpack expected = 0;
            for (std::size_t bit = 0; bit < kBits; ++bit) {
                std::uint32_t count = 0;
                for (const Bitpack x : counted) {
                    count += ((x >> bit) & 1u) != 0 ? 1 : 0;
                }
                if (count >= g.low && count <= g.high) {
                    expected = static_cast<Bitpack>(expected | (Bitpack{1} << bit));
                }
            }
            want[s * words + w] = expected;
            out[w] = static_cast<Bitpack>(~expected);
            ones += static_cast<std::size_t>(std::popcount(expected));
        }
    }
    // Both outputs occur, so a kernel that wrote a constant would fail.
    CHECK(ones > 0);
    CHECK(ones < picked.size() * words * kBits);

    queue.memcpy(arena.view().data(), image.data(), image.size() * sizeof(Bitpack)).wait_and_throw();
    using kernel_t = fuzzybits::mc::kernel::cardinality<T, form_, slots_, lanes_form_>;
    const kernel_t kernel(block.view(), order);
    queue.submit([&](sycl::handler &cgh) { cgh.parallel_for(kernel.get_range(), kernel); });
    queue.wait_and_throw();

    std::vector<Bitpack> after(image.size());
    queue.memcpy(after.data(), arena.view().data(), after.size() * sizeof(Bitpack)).wait_and_throw();

    std::size_t mismatches = 0;
    for (std::uint32_t s = 0; s < block.size(); ++s) {
        const gate_spec &g = kGates[picked[s]];
        for (std::size_t w = 0; w < words; ++w) {
            const Bitpack got = after[std::size_t{block.slot(s)} * words + w];
            const Bitpack expected = want[s * words + w];
            if (got != expected && ++mismatches <= 8) {
                UNSCOPED_INFO("gate " << picked[s] << " n=" << g.positive + g.negated << " negated=" << g.negated << " [" << g.low << ", " << g.high
                                      << "] bitpack=" << w << std::hex << " got=0x" << static_cast<std::uint64_t>(got) << " want=0x"
                                      << static_cast<std::uint64_t>(expected));
            }
        }
    }
    CHECK(mismatches == 0);

    // The kernel writes only its gates' slots: the inputs, and the guard slot after the block, are as the host wrote
    // them.
    const auto run = [words](const std::vector<Bitpack> &copy, const ev::slot_range<std::uint32_t> range) {
        return std::span<const Bitpack>(copy).subspan(std::size_t{range.first} * words, std::size_t{range.count} * words);
    };
    CHECK(std::ranges::equal(run(after, inputs), run(image, inputs)));
    CHECK(std::ranges::equal(run(after, guard), run(image, guard)));
}

} // namespace

TEMPLATE_TEST_CASE("cardinality kernels match a host count in every form", "[mc][cardinality]", std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t) {
    using Bitpack = TestType;
    // Bitpacks per node per pass. In the lanes form (kGateCpuLanes = 8 bitpacks per work-item) 4 is a tail alone, 32
    // whole vectors and 37 whole vectors and a tail; in the GPU form (256 work-items per work-group) 296 takes two
    // work-groups per gate, and in the lanes form (64 work-items of 8 bitpacks) 592 does, so the two orders map
    // work-groups to gates differently in both forms.
    const std::uint32_t bitpacks_per_node = GENERATE(4u, 32u, 37u, 296u, 592u);
    const auto order = GENERATE(gate_order::gate_major, gate_order::block_major);

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>() << " bits=" << sizeof(Bitpack) * 8 << " bitpacks_per_node=" << bitpacks_per_node
                   << " block_major=" << (order == gate_order::block_major));

    SECTION("fixed") {
        // Every slot count; a gate takes the smallest that holds its inputs (cardinality_slots_for).
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (check_form<Bitpack, cardinality_form::fixed, fuzzybits::mc::kernel::kCardinalitySlots[I], false>(queue, bitpacks_per_node, order), ...);
            (check_form<Bitpack, cardinality_form::fixed, fuzzybits::mc::kernel::kCardinalitySlots[I], true>(queue, bitpacks_per_node, order), ...);
        }(std::make_index_sequence<fuzzybits::mc::kernel::kCardinalitySlots.size()>{});
    }
    SECTION("counter") {
        check_form<Bitpack, cardinality_form::counter, 0, false>(queue, bitpacks_per_node, order);
        check_form<Bitpack, cardinality_form::counter, 0, true>(queue, bitpacks_per_node, order);
    }
    SECTION("wide") {
        check_form<Bitpack, cardinality_form::wide, 0, false>(queue, bitpacks_per_node, order);
        check_form<Bitpack, cardinality_form::wide, 0, true>(queue, bitpacks_per_node, order);
    }
}

TEST_CASE("cardinality forms and slot counts by input count", "[mc][cardinality]") {
    using fuzzybits::mc::kernel::cardinality_form_for;
    using fuzzybits::mc::kernel::cardinality_slots_for;
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
