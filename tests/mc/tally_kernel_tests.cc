// Device tests of the tally kernel (src/mc/kernel/tally.h) and its block (src/mc/event/tally.h). Each launch adds the
// number of true trials in one pass of each tallied node to its tally's 64-bit count: for random, all-ones and
// all-zeros nodes, 8- to 64-bit bitpacks, and geometries that give work-items runs of one bitpack, of two and of kMaxRun
// on any device (the kernel sizes its runs by the device's compute units); and from a count that carries past 2^32. A
// tally reads its node at its arena slot and writes nothing but its count. Its block refuses a slot outside the arena.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "mc/event/node.h"
#include "mc/event/tally.h"
#include "mc/kernel/tally.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <ranges>
#include <span>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <vector>

// Widths of 128 and 256 bits are rejected at compile time; see bitpack_width_compile_fail.cc and the
// mc.compile_fail.* tests.

namespace {

namespace ev = fuzzybits::mc::event;
template <class bitpack_t> using types_of = ev::node_types<bitpack_t, std::int32_t, std::uint32_t>;
template <class bitpack_t> using kernel_of = fuzzybits::mc::kernel::tally<types_of<bitpack_t>>;

/// Bitpacks per node that one work-group reads in full runs: kWorkGroup work-items of kMaxRun bitpacks each.
template <class bitpack_t> constexpr std::size_t kFullGroup = kernel_of<bitpack_t>::kWorkGroup * kernel_of<bitpack_t>::kMaxRun;

/// Work-items per compute unit by which kernel::tally sizes its runs (src/mc/kernel/tally.h).
constexpr std::size_t kItemsPerUnit = 2048;

constexpr std::size_t ceil_div(const std::size_t n, const std::size_t d) { return (n + d - 1) / d; }

/// The device's compute units as kernel::tally counts them: at least one.
std::size_t compute_units(const sycl::device &device) { return std::max<std::size_t>(device.get_info<sycl::info::device::max_compute_units>(), 1); }

/// The geometry kernel::tally gives `tallies` tallies of `bitpacks` bitpacks on `units` compute units: a work-item reads
/// a strided run of at most `run` bitpacks of its tally's node (kMaxRun, fewer when tallies × bitpacks is under
/// kMaxRun × units × kItemsPerUnit), and a tally takes `items_per_tally` work-items, a whole number of work-groups.
struct geometry {
    std::size_t run;
    std::size_t items_per_tally;
};

template <class Kernel> geometry geometry_of(const std::size_t units, const std::size_t tallies, const std::size_t bitpacks) {
    const std::size_t run = std::clamp<std::size_t>(tallies * bitpacks / (units * kItemsPerUnit), 1, Kernel::kMaxRun);
    const std::size_t groups = std::max<std::size_t>(ceil_div(ceil_div(bitpacks, run), Kernel::kWorkGroup), 1);
    return {.run = run, .items_per_tally = groups * Kernel::kWorkGroup};
}

/// A geometry of the counting test: bitpacks per node, copies of each of its three tallies, and the run kernel::tally
/// gives them.
struct layout {
    std::size_t bitpacks;
    std::size_t copies;
    std::size_t run;
};

std::ostream &operator<<(std::ostream &os, const layout &l) { return os << "bitpacks=" << l.bitpacks << " copies=" << l.copies << " run=" << l.run; }

/// The counting test's geometries on a device of `units` compute units, each with the same run on any device:
/// - runs of 1, as 3 tallies × 257 bitpacks is under 2 × kItemsPerUnit, the least that gives a run of 2 (on one
///   compute unit): 1 bitpack per node, one busy work-item in one work-group; 255, one bitpack for every work-item but
///   the last; 257, a second work-group whose first work-item alone is busy;
/// - runs of 2, from one bitpack more than the fewest that give them, so that some work-items read one;
/// - runs of kMaxRun, from enough copies of the tallies, which raise tallies × bitpacks without a larger arena:
///   kFullGroup, one work-group whose work-items each read kMaxRun bitpacks, the largest all-ones sum the 32-bit
///   group reduction carries; kFullGroup + 1, a second work-group; 2 × kFullGroup, two work-groups of full runs.
template <class bitpack_t> std::vector<layout> layouts(const std::size_t units) {
    constexpr std::size_t kTallies = 3;
    constexpr std::size_t kMaxRun = kernel_of<bitpack_t>::kMaxRun;
    constexpr std::size_t kFull = kFullGroup<bitpack_t>;
    // Tallies × bitpacks for each bitpack of run.
    const std::size_t per_run = units * kItemsPerUnit;
    const auto max_run = [&](const std::size_t bitpacks) { return layout{bitpacks, ceil_div(kMaxRun * per_run, kTallies * bitpacks), kMaxRun}; };
    return {{1, 1, 1}, {255, 1, 1}, {257, 1, 1}, {ceil_div(2 * per_run, kTallies) + 1, 1, 2}, max_run(kFull), max_run(kFull + 1), max_run(2 * kFull)};
}

enum class pattern { random, all_ones, all_zeros, quarter, three_quarters };

// The arena of the counting test. The observed nodes (slots 1 to 3) lie between unobserved slots whose bitpacks have
// a quarter and three quarters of their bits set: a tally that read past either end of its node would count another
// slot's bits, not memory outside the arena. Fixed fractions keep every slot's count apart from the random node's at
// node sizes that depend on the device, where a second random slot's count can equal it.
constexpr std::array<pattern, 5> kSlots{pattern::quarter, pattern::all_zeros, pattern::all_ones, pattern::random, pattern::three_quarters};

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

/// Writes `p` into `words`; random bits depend on `seed` and the bitpack index.
template <typename bitpack_t> void fill(const std::span<bitpack_t> words, const pattern p, const std::uint64_t seed) {
    for (std::size_t j = 0; j < words.size(); ++j) {
        switch (p) {
        case pattern::random:
            words[j] = static_cast<bitpack_t>(splitmix64((seed << 32) | j));
            break;
        case pattern::all_ones:
            words[j] = static_cast<bitpack_t>(~bitpack_t{0});
            break;
        case pattern::all_zeros:
            words[j] = bitpack_t{0};
            break;
        case pattern::quarter:
            words[j] = static_cast<bitpack_t>(0x1111'1111'1111'1111ull);
            break;
        case pattern::three_quarters:
            words[j] = static_cast<bitpack_t>(0x7777'7777'7777'7777ull);
            break;
        }
    }
}

template <std::ranges::input_range R> std::uint64_t host_popcount(const R &words) {
    std::uint64_t n = 0;
    for (const auto w : words) {
        n += static_cast<std::uint64_t>(std::popcount(w));
    }
    return n;
}

/// One pass: one launch of `kernel`, waited for.
template <class Kernel> void launch(sycl::queue &queue, const Kernel &kernel) {
    queue.submit([&](sycl::handler &cgh) { cgh.parallel_for(kernel.get_range(), kernel); });
    queue.wait_and_throw();
}

} // namespace

TEMPLATE_TEST_CASE("tally kernel counts set bits and accumulates across launches", "[mc][tally]", std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t) {
    using bitpack_t = TestType;
    using types = types_of<bitpack_t>;
    using kernel_t = kernel_of<bitpack_t>;
    using tally_t = ev::tally<types>;
    constexpr std::size_t kBits = sizeof(bitpack_t) * 8;

    sycl::queue queue;
    const std::size_t units = compute_units(queue.get_device());
    const layout shape = GENERATE_COPY(from_range(layouts<bitpack_t>(units)));
    const std::size_t bitpacks = shape.bitpacks;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>() << " units=" << units << " bits=" << kBits << " " << shape);

    std::vector<bitpack_t> host(kSlots.size() * bitpacks);
    std::array<std::uint64_t, kSlots.size()> ones_per_pass{};
    for (std::size_t s = 0; s < kSlots.size(); ++s) {
        const std::span<bitpack_t> node(host.data() + s * bitpacks, bitpacks);
        fill(node, kSlots[s], s);
        ones_per_pass[s] = host_popcount(node);
    }

    auto arena = ev::output_arena<types>::create(queue, kSlots.size(), bitpacks);
    (void)arena.reserve(kSlots.size());
    queue.memcpy(arena.view().data(), host.data(), host.size() * sizeof(bitpack_t)).wait_and_throw();

    // (id_in_pdag, slot): the random, all-ones and all-zeros nodes, in an order other than their slots'. Their PDAG
    // indices are other slots, so a kernel that found a node by its index instead of its slot would count another node.
    const std::array observed{tally_t(4, 3), tally_t(0, 2), tally_t(3, 1)};
    // Each observed node's count differs from every other slot's, so a tally that read another slot would fail.
    for (const auto &record : observed) {
        for (std::size_t s = 0; s < kSlots.size(); ++s) {
            INFO("slot=" << s << " observed=" << record.slot);
            REQUIRE((s == record.slot || ones_per_pass[s] != ones_per_pass[record.slot]));
        }
    }
    // Copies of the three tallies observe the same nodes and count independently.
    std::vector<tally_t> records;
    records.reserve(observed.size() * shape.copies);
    for (std::size_t c = 0; c < shape.copies; ++c) {
        records.insert(records.end(), observed.begin(), observed.end());
    }

    const auto block = ev::tally_block<types>::create(queue, arena.view(), records);
    const kernel_t kernel(block.view(), queue.get_device());
    // The geometry on this device, and the run the layout is chosen for.
    const geometry expected = geometry_of<kernel_t>(units, records.size(), bitpacks);
    REQUIRE(expected.run == shape.run);
    CHECK(kernel.tallies() == records.size());
    CHECK(kernel.items_per_tally() == expected.items_per_tally);
    CHECK(kernel.get_range().get_local_range()[0] == kernel_t::kWorkGroup);
    CHECK(kernel.get_range().get_global_range()[0] == records.size() * kernel.items_per_tally());
    // Work-item i of a tally reads bitpacks i, i + items_per_tally, ... of its node: more than one only for runs over
    // one, and at most kMaxRun.
    CHECK((kernel.items_per_tally() < bitpacks) == (shape.run > 1));
    CHECK(kernel.items_per_tally() * kernel_t::kMaxRun >= bitpacks);

    // Each launch is one pass and adds that pass's ones to the counts. The kernel writes no other field.
    for (std::uint64_t passes = 1; passes <= 2; ++passes) {
        launch(queue, kernel);
        const auto counts = block.records();
        for (std::size_t t = 0; t < records.size(); ++t) {
            INFO("tally=" << t << " passes=" << passes);
            CHECK(counts[t].ones == passes * ones_per_pass[records[t].slot]);
            CHECK(counts[t].id_in_pdag == records[t].id_in_pdag);
            CHECK(counts[t].slot == records[t].slot);
            if (kSlots[records[t].slot] == pattern::all_ones) {
                // Every trial of an all-ones node is true: its count is the trial count the host derives.
                CHECK(counts[t].ones == passes * bitpacks * kBits);
            }
        }
    }

    // The kernel reads the arena and does not write it.
    std::vector<bitpack_t> after(host.size());
    queue.memcpy(after.data(), arena.view().data(), after.size() * sizeof(bitpack_t)).wait_and_throw();
    CHECK(std::ranges::equal(after, host));
}

TEST_CASE("tally counts are 64-bit", "[mc][tally]") {
    using bitpack_t = std::uint64_t;
    using types = types_of<bitpack_t>;
    using kernel_t = kernel_of<bitpack_t>;
    constexpr std::size_t kBits = 64;
    // One tally of kFullGroup + 1 bitpacks has runs of at most 8193 / kItemsPerUnit = 4 on any device, so 2049
    // work-items or more: several work-groups, each of which adds to the count.
    constexpr std::size_t kBitpacks = kFullGroup<bitpack_t> + 1;
    // The work-group sums are 32-bit and the count is not: from 2^32 - 1, the first addition carries into the count's
    // high word.
    constexpr std::uint64_t kStart = (std::uint64_t{1} << 32) - 1;

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());

    auto arena = ev::output_arena<types>::create(queue, 1, kBitpacks);
    (void)arena.reserve(1);
    const std::vector<bitpack_t> ones(kBitpacks, ~bitpack_t{0});
    queue.memcpy(arena.view().data(), ones.data(), kBitpacks * sizeof(bitpack_t)).wait_and_throw();

    // (id_in_pdag, slot, ones)
    const std::array records{ev::tally<types>(0, 0, kStart)};
    const auto block = ev::tally_block<types>::create(queue, arena.view(), records);
    const kernel_t kernel(block.view(), queue.get_device());
    REQUIRE(kernel.items_per_tally() > kernel_t::kWorkGroup);
    launch(queue, kernel);
    CHECK(block.records()[0].ones == kStart + kBitpacks * kBits);
}

TEST_CASE("tally_block refuses a slot outside the arena", "[mc][tally]") {
    using types = types_of<std::uint64_t>;
    using tally_t = ev::tally<types>;
    sycl::queue queue;
    auto arena = ev::output_arena<types>::create(queue, 3, 1);
    (void)arena.reserve(3);

    // (id_in_pdag, slot)
    const std::array last{tally_t(0, 2)};
    CHECK(ev::tally_block<types>::create(queue, arena.view(), last).size() == 1u);

    // Every record is checked, not only the first.
    const std::array past_end{tally_t(0, 0), tally_t(1, 3)};
    CHECK_THROWS_AS(ev::tally_block<types>::create(queue, arena.view(), past_end), std::out_of_range);

    const std::array far{tally_t(0, std::numeric_limits<std::uint32_t>::max())};
    CHECK_THROWS_AS(ev::tally_block<types>::create(queue, arena.view(), far), std::out_of_range);
}
