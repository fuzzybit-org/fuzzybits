// Device tests of the Boolean gate kernels, AND, OR, XOR, NOT, NAND, NOR, XNOR and NULL (src/mc/kernel/gate.h),
// against a host evaluation: every slot count of the fixed form, with as many inputs as slots and with fewer, the
// counted form at the same input counts and at 17 to 100 inputs (the kernel builder uses it for every Boolean gate but
// NOT and NULL), both lane forms (GPU, CPU), both work-group orders, gates without inputs (the empty fold), and 8- to
// 64-bit bitpacks. Each gate's inputs have a density that leaves both outputs likely.
//
// Inputs and gates share one output arena (event/node.h), which holds a guard slot, the inputs, the gate block and
// another guard slot, in that order. The inputs' slots run in the reverse of their order in the block's input table.
// The kernel must write every bitpack of its block's slots and no other bitpack of the arena.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "mc/event/gate.h"
#include "mc/event/node.h"
#include "mc/kernel/gate.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <span>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace {

namespace ev = fuzzybits::mc::event;
using fuzzybits::mc::kernel::gate_order;
using fuzzybits::mc::kernel::gate_slots_for;
using fuzzybits::mc::kernel::gate_slots_of;
using fuzzybits::mc::kernel::is_unary;
using fuzzybits::mc::kernel::kGateSlots;
using fuzzybits::mc::kernel::op_code;

template <typename Bitpack> using types = ev::node_types<Bitpack, std::int32_t, std::uint32_t>;

struct gate_spec {
    std::size_t positive;
    std::size_t negated;
};

// Fixed form: no inputs, then every slot count with as many inputs as slots and with fewer (5 in 6, 7 in 8, 9 and 10
// in 12, 13 and 14 in 16); counted form: all of these, and 17 to 100 inputs; both polarities. The input table holds
// no padding, and in each fixed-form block a gate with fewer inputs than slots comes before another gate, whose
// inputs follow its own in the table: a kernel that read slot j >= num_inputs would read them.
const std::vector<gate_spec> kGates{
    {0, 0}, {1, 0}, {0, 1}, {1, 1},  {2, 1}, {3, 1}, {2, 3},   {4, 2},   {5, 2},   {6, 2},   {5, 4},
    {6, 4}, {6, 6}, {7, 6}, {10, 4}, {8, 8}, {9, 8}, {12, 12}, {20, 20}, {32, 32}, {60, 40},
};
const std::vector<gate_spec> kUnary{{1, 0}, {0, 1}};

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

// Counted inputs are true with probability density(op) / 8: AND-like gates high, OR-like low.
constexpr unsigned density(const op_code op) {
    if (op == op_code::and_gate || op == op_code::nand_gate) {
        return 7;
    }
    if (op == op_code::or_gate || op == op_code::nor_gate) {
        return 1;
    }
    return 4;
}

// The gate's value, bit by bit: the AND, OR or XOR of its inputs (the negated ones complemented), complemented for
// NAND, NOR and XNOR; NOT complements its one input and NULL passes it through.
template <typename Bitpack> Bitpack host_gate(const op_code op, const std::vector<Bitpack> &inputs, const std::size_t positive) {
    const bool and_like = op == op_code::and_gate || op == op_code::nand_gate;
    Bitpack acc = and_like ? static_cast<Bitpack>(~Bitpack{0}) : Bitpack{0};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto x = i < positive ? inputs[i] : static_cast<Bitpack>(~inputs[i]);
        if (and_like) {
            acc = static_cast<Bitpack>(acc & x);
        } else if (op == op_code::xor_gate || op == op_code::xnor_gate) {
            acc = static_cast<Bitpack>(acc ^ x);
        } else if (op == op_code::or_gate || op == op_code::nor_gate) {
            acc = static_cast<Bitpack>(acc | x);
        } else {
            acc = x;
        }
    }
    if (op == op_code::nand_gate || op == op_code::nor_gate || op == op_code::xnor_gate || op == op_code::not_gate) {
        acc = static_cast<Bitpack>(~acc);
    }
    return acc;
}

// The gates of kGates (kUnary for NOT and NULL) that take slots_ slots, or all of them for the counted form
// (slots_ == 0), as one block evaluated on the device with bitpacks_per_node bitpacks per node.
template <typename Bitpack, op_code op_, std::uint32_t slots_, bool lanes_form_>
void check(sycl::queue &queue, const std::uint32_t bitpacks_per_node, const gate_order order) {
    using T = types<Bitpack>;
    constexpr std::size_t kBits = sizeof(Bitpack) * 8;
    const std::size_t words = bitpacks_per_node;
    INFO("op=" << static_cast<int>(op_) << " slots=" << slots_ << " lanes_form=" << lanes_form_ << " bits=" << kBits);

    std::vector<gate_spec> specs;
    for (const auto &g : is_unary(op_) ? kUnary : kGates) {
        if (slots_ == 0 || gate_slots_for(g.positive + g.negated) == slots_) {
            specs.push_back(g);
        }
    }
    REQUIRE(!specs.empty());
    std::size_t total_inputs = 0;
    for (const auto &g : specs) {
        total_inputs += g.positive + g.negated;
    }

    // The arena's slots in the order they are reserved: a guard, the inputs, the gate block and another guard. The
    // inputs come first: build() refuses an input slot that is not reserved yet.
    auto arena = ev::output_arena<T>::create(queue, total_inputs + specs.size() + 2, words);
    const auto low_guard = arena.reserve(1);
    const auto inputs = arena.reserve(total_inputs);

    // The arena's contents on the host: bitpack w of slot s is image[at(s, w)].
    std::vector<Bitpack> image(arena.view().size());
    const auto at = [words](const std::size_t slot, const std::size_t w) { return slot * words + w; };

    // Input k of the table (gate after gate, positive inputs then negated) is at density d (positive) or 8 - d
    // (negated), so that every counted input is true with probability d / 8. It holds slot inputs.end() - 1 - k: the
    // slots run against the table, so a kernel that read slot first_input + i, or a gate's inputs from consecutive
    // slots upwards, would read other inputs.
    ev::gate_block_builder<T> builder;
    builder.reserve(specs.size(), total_inputs);
    std::vector<std::vector<std::uint32_t>> input_slots(specs.size());
    std::size_t k = 0;
    for (std::size_t s = 0; s < specs.size(); ++s) {
        const auto &g = specs[s];
        for (std::size_t i = 0; i < g.positive + g.negated; ++i, ++k) {
            const auto slot = static_cast<std::uint32_t>(inputs.end() - 1 - k);
            const unsigned eighths = i < g.positive ? density(op_) : 8 - density(op_);
            for (std::size_t w = 0; w < words; ++w) {
                image[at(slot, w)] = random_word<Bitpack>(eighths, k * words + w);
            }
            input_slots[s].push_back(slot);
        }
        const std::span<const std::uint32_t> of_gate(input_slots[s]);
        builder.add(static_cast<std::int32_t>(s), of_gate.first(g.positive), of_gate.subspan(g.positive));
    }
    const auto block = builder.build(queue, arena);
    const auto high_guard = arena.reserve(1);

    // A gate's slot starts as the complement of its expected output, so that a bitpack the kernel does not write
    // differs; the guards hold random words.
    std::vector<Bitpack> want(specs.size() * words);
    std::vector<Bitpack> column;
    for (std::size_t s = 0; s < specs.size(); ++s) {
        const std::size_t slot = block.slot(static_cast<std::uint32_t>(s));
        column.resize(input_slots[s].size());
        for (std::size_t w = 0; w < words; ++w) {
            for (std::size_t i = 0; i < column.size(); ++i) {
                column[i] = image[at(input_slots[s][i], w)];
            }
            want[s * words + w] = host_gate(op_, column, specs[s].positive);
            image[at(slot, w)] = static_cast<Bitpack>(~want[s * words + w]);
        }
    }
    for (const std::size_t guard : {low_guard.first, high_guard.first}) {
        for (std::size_t w = 0; w < words; ++w) {
            image[at(guard, w)] = random_word<Bitpack>(4, (total_inputs + guard) * words + w);
        }
    }
    const auto view = arena.view();
    queue.memcpy(view.data(), image.data(), image.size() * sizeof(Bitpack)).wait_and_throw();

    using kernel_t = fuzzybits::mc::kernel::op<op_, T, slots_, lanes_form_>;
    const kernel_t kernel(block.view(), order);
    queue.submit([&](sycl::handler &cgh) { cgh.parallel_for(kernel.get_range(), kernel); });
    queue.wait_and_throw();

    std::vector<Bitpack> out(image.size());
    queue.memcpy(out.data(), view.data(), out.size() * sizeof(Bitpack)).wait_and_throw();

    std::size_t mismatches = 0;
    std::size_t ones = 0;
    for (std::size_t s = 0; s < specs.size(); ++s) {
        const std::size_t slot = block.slot(static_cast<std::uint32_t>(s));
        for (std::size_t w = 0; w < words; ++w) {
            const Bitpack expected = want[s * words + w];
            const Bitpack got = out[at(slot, w)];
            ones += static_cast<std::size_t>(std::popcount(expected));
            if (got != expected && ++mismatches <= 8) {
                UNSCOPED_INFO("gate=" << s << " n=" << specs[s].positive + specs[s].negated << " word=" << w << std::hex << " got=0x"
                                      << static_cast<std::uint64_t>(got) << " want=0x" << static_cast<std::uint64_t>(expected));
            }
        }
    }
    CHECK(mismatches == 0);
    CHECK(ones > 0);
    CHECK(ones < specs.size() * words * kBits);

    // Every slot outside the block, the inputs and both guards, holds what was written to it.
    std::size_t changed = 0;
    for (std::uint32_t slot = 0; slot < arena.slots(); ++slot) {
        if (block.slots().contains(slot)) {
            continue;
        }
        for (std::size_t w = 0; w < words; ++w) {
            if (out[at(slot, w)] != image[at(slot, w)] && ++changed <= 8) {
                UNSCOPED_INFO("slot=" << slot << " word=" << w << std::hex << " got=0x" << static_cast<std::uint64_t>(out[at(slot, w)]) << " written=0x"
                                      << static_cast<std::uint64_t>(image[at(slot, w)]));
            }
        }
    }
    CHECK(changed == 0);
}

template <typename Bitpack, op_code op_, bool lanes_form_, std::size_t... I>
void check_slot_counts(sycl::queue &queue, const std::uint32_t bitpacks_per_node, const gate_order order, std::index_sequence<I...>) {
    (check<Bitpack, op_, kGateSlots[I], lanes_form_>(queue, bitpacks_per_node, order), ...);
}

// Every slot count and the counted form, in both lane forms.
template <typename Bitpack, op_code op_> void check_all(sycl::queue &queue, const std::uint32_t bitpacks_per_node, const gate_order order) {
    if constexpr (is_unary(op_)) {
        check<Bitpack, op_, 1, false>(queue, bitpacks_per_node, order);
        check<Bitpack, op_, 1, true>(queue, bitpacks_per_node, order);
    } else {
        constexpr auto each = std::make_index_sequence<kGateSlots.size()>{};
        check_slot_counts<Bitpack, op_, false>(queue, bitpacks_per_node, order, each);
        check_slot_counts<Bitpack, op_, true>(queue, bitpacks_per_node, order, each);
        check<Bitpack, op_, 0, false>(queue, bitpacks_per_node, order);
        check<Bitpack, op_, 0, true>(queue, bitpacks_per_node, order);
    }
}

// Filled, not filled and counted: 1, 3 and 16 slots, and the counted form.
template <typename Bitpack, op_code op_> void check_some(sycl::queue &queue, const std::uint32_t bitpacks_per_node, const gate_order order) {
    check<Bitpack, op_, 1, false>(queue, bitpacks_per_node, order);
    check<Bitpack, op_, 3, true>(queue, bitpacks_per_node, order);
    check<Bitpack, op_, 16, false>(queue, bitpacks_per_node, order);
    check<Bitpack, op_, 16, true>(queue, bitpacks_per_node, order);
    check<Bitpack, op_, 0, false>(queue, bitpacks_per_node, order);
    check<Bitpack, op_, 0, true>(queue, bitpacks_per_node, order);
}

} // namespace

TEST_CASE("gate kernels match a host evaluation", "[mc][gate]") {
    // Bitpacks per node per pass, batch_size x bitpacks_per_batch as the layer manager sizes its arena: 1 x 4, 1 x 37,
    // 8 x 4, 8 x 37 and 16 x 37. In the lanes form (8 bitpacks per work-item) 4 is all tail and 37 ends in a partial
    // vector. 296 takes two work-groups per gate in the GPU form, and 592 two in the lanes form and three in the GPU
    // form: with one work-group per gate, block-major and gate-major order assign work-groups alike.
    const std::uint32_t bitpacks_per_node = GENERATE(4u, 37u, 32u, 296u, 592u);
    const auto order = GENERATE(gate_order::gate_major, gate_order::block_major);

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>() << " bitpacks=" << bitpacks_per_node
                   << " block_major=" << (order == gate_order::block_major));

    SECTION("AND, every slot count, 64- and 8-bit") {
        check_all<std::uint64_t, op_code::and_gate>(queue, bitpacks_per_node, order);
        check_all<std::uint8_t, op_code::and_gate>(queue, bitpacks_per_node, order);
    }
    SECTION("XOR and XNOR, every slot count, 16-bit") {
        check_all<std::uint16_t, op_code::xor_gate>(queue, bitpacks_per_node, order);
        check_all<std::uint16_t, op_code::xnor_gate>(queue, bitpacks_per_node, order);
    }
    SECTION("OR, NAND, NOR, XOR, XNOR") {
        check_some<std::uint64_t, op_code::or_gate>(queue, bitpacks_per_node, order);
        check_some<std::uint64_t, op_code::nand_gate>(queue, bitpacks_per_node, order);
        check_some<std::uint64_t, op_code::nor_gate>(queue, bitpacks_per_node, order);
        check_some<std::uint32_t, op_code::xor_gate>(queue, bitpacks_per_node, order);
        check_some<std::uint64_t, op_code::xnor_gate>(queue, bitpacks_per_node, order);
    }
    SECTION("NOT and NULL") {
        check_all<std::uint64_t, op_code::not_gate>(queue, bitpacks_per_node, order);
        check_all<std::uint64_t, op_code::null_gate>(queue, bitpacks_per_node, order);
        check_all<std::uint8_t, op_code::not_gate>(queue, bitpacks_per_node, order);
    }
}

TEST_CASE("gate slot counts by input count", "[mc][gate]") {
    CHECK(gate_slots_for(0) == 1);
    CHECK(gate_slots_for(1) == 1);
    CHECK(gate_slots_for(5) == 6);
    CHECK(gate_slots_for(9) == 12);
    CHECK(gate_slots_for(16) == 16);
    CHECK(gate_slots_for(17) == 0);

    // The kernel builder's choice: one counted kernel per op code, one slot for NOT and NULL.
    CHECK(gate_slots_of(op_code::and_gate) == 0);
    CHECK(gate_slots_of(op_code::or_gate) == 0);
    CHECK(gate_slots_of(op_code::xor_gate) == 0);
    CHECK(gate_slots_of(op_code::nand_gate) == 0);
    CHECK(gate_slots_of(op_code::nor_gate) == 0);
    CHECK(gate_slots_of(op_code::xnor_gate) == 0);
    CHECK(gate_slots_of(op_code::not_gate) == 1);
    CHECK(gate_slots_of(op_code::null_gate) == 1);
}
