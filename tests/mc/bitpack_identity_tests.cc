// Device tests of the sampling kernels (src/mc/kernel/sampler/): the kernel of every sampler form (strided, lockstep,
// chunked) writes, on every backend, exactly the bitpacks of the host reference (prng/bernoulli/sample.h) at the
// counters of prng/stream.h, which a pinned checksum fixes. The run's seed and device are not the defaults, and the
// pass puts the run-wide bitpack indices across a 2^32 boundary inside the first vector of the vector forms, so every
// counter word is exercised. A node's bits depend only on its counter coordinates: not on its block, its arena slot or
// the pass size. Further device cases run the strided form's in-loop restart (several bitpacks per work-item), a
// bitpack that takes all 32 generations, and kernels of streams that differ only in seed or in device, in one
// process.
//
// Events are made as the kernel builder makes them: those whose law is a constant (never, always) are written once,
// when their arena slots are allocated, and only the others reach the sampling kernel, whose record type cannot hold a
// constant law. Each launch's arena holds two guard slots, the events' slots and one more guard slot, and every bitpack
// of it starts as a sentinel word of its own: nothing may change a bitpack outside the events' slots, and a bitpack of
// an event that is not written keeps its pseudo-random sentinel, which equals the reference's word only by a 2^-64
// chance.

#include <catch2/catch_test_macros.hpp>

#include "mc/event/basic_event.h"
#include "mc/event/node.h"
#include "mc/kernel/sampler/chunked.h"
#include "mc/kernel/sampler/lockstep.h"
#include "mc/kernel/sampler/sampler.h"
#include "mc/kernel/sampler/strided.h"
#include "mc/prng/bernoulli/bernoulli.h"
#include "mc/prng/philox128.h"
#include "mc/prng/prng.h"
#include "mc/prng/stream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <ios>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace ev = fuzzybits::mc::event;
namespace prng = fuzzybits::mc::prng;
namespace bernoulli = prng::bernoulli;
using types = ev::node_types<std::uint64_t, std::int32_t, std::uint32_t>;
namespace sampler = fuzzybits::mc::kernel::sampler;
using strided_kernel = sampler::strided<types>;
using lockstep_kernel = sampler::lockstep<types>;
using chunked_kernel = sampler::chunked<types>;

// A bitpack's 64 random words come from its generations, two per generation.
static_assert(2 * bernoulli::kMaxGenerations == 64);

// The sampling kernels read only events whose trials are random: a block of general basic events, which may hold a
// constant law, does not make one.
template <typename kernel_type>
constexpr bool reads_only_sampled_events = !std::is_constructible_v<kernel_type, ev::basic_event_block_view<types>, prng::stream, sycl::device> &&
                                           std::is_constructible_v<kernel_type, ev::sampled_event_block_view<types>, prng::stream, sycl::device>;
static_assert(reads_only_sampled_events<strided_kernel>);
static_assert(reads_only_sampled_events<lockstep_kernel>);
static_assert(reads_only_sampled_events<chunked_kernel>);

// Each kernel samples in its own form.
static_assert(strided_kernel::kForm == sampler::form::strided && lockstep_kernel::kForm == sampler::form::lockstep &&
              chunked_kernel::kForm == sampler::form::chunked);

// 40 bitpacks per node per pass: the vector forms end on a partial group.
constexpr std::uint32_t kBitpacksPerPass = 40;
// 40 · kPass = 3 · 2^32 - 8: the pass's bitpacks 0-7 have high word 2 and 8-39 high word 3.
constexpr std::uint32_t kPass = 322122547;
static_assert(std::uint64_t{kPass} * kBitpacksPerPass == 3 * (std::uint64_t{1} << 32) - 8);
constexpr prng::stream kStream = prng::stream::of(0x243f6a8885a308d3ull, 3);

// (PDAG index, p): ordinary, half, never, always, a few significant threshold bits, and p = 1e-12 < 2^-32, whose
// threshold has high word 0 (#17: the 32-bit sampler collapsed it to 0). On the device that last event exercises only
// the `sampled` dispatch and shallow ties, which end within the threshold's 39 leading zeros: its bitpacks here have
// no ones, so it cannot by itself detect a #17-style collapse to zero. The host tests (bernoulli_sampler_tests.cc)
// cover exactness; on the device, the tied thresholds of the test case below reach a threshold's last bit.
constexpr std::array<std::pair<std::int32_t, double>, 6> kEvents{{
    {2, 0.1},
    {3, 0.5},
    {4, 0.0},
    {5, 1.0},
    {6, 0.3125},
    {7, 1e-12},
}};

// FNV-1a 64 of the 240 little-endian uint64 bitpacks of the bit-sliced sampler (prng/bernoulli/sample.h) at the
// counters above. Every backend and every kernel form must reproduce it; a change to the random stream fails this test
// everywhere.
constexpr std::uint64_t kExpectedChecksum = 0xb016dd42494810ffull;

constexpr const char *form_name(const sampler::form form) {
    switch (form) {
    case sampler::form::strided:
        return "strided";
    case sampler::form::lockstep:
        return "lockstep";
    case sampler::form::chunked:
        return "chunked";
    }
    return "?";
}

std::uint64_t fnv1a64(const std::vector<std::uint64_t> &data) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (const std::uint64_t v : data) {
        unsigned char bytes[8];
        std::memcpy(bytes, &v, 8);
        for (unsigned char b : bytes) {
            h ^= b;
            h *= 0x100000001b3ull;
        }
    }
    return h;
}

// SplitMix64, a bijection of 64-bit words: distinct arena indices get distinct sentinel words.
std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

std::vector<ev::basic_event<types>> records() {
    std::vector<ev::basic_event<types>> out;
    out.reserve(kEvents.size());
    for (const auto &[id, p] : kEvents) {
        out.push_back({.id_in_pdag = id, .law = *ev::bernoulli_law::from_probability(p)});
    }
    return out;
}

/// Bitpacks [first, first + count) of node `id`'s run in `stream` under `law`, from the host reference.
std::vector<std::uint64_t> reference(const prng::stream &stream, const std::int32_t id, const ev::bernoulli_law law, const std::uint64_t first,
                                     const std::uint32_t count) {
    const bernoulli::pass_stream source(stream, static_cast<std::uint32_t>(id), first);
    std::vector<std::uint64_t> packs(count);
    for (std::uint32_t s = 0; s < count; ++s) {
        switch (law.kind()) {
        case ev::bernoulli_kind::never:
            packs[s] = 0;
            break;
        case ev::bernoulli_kind::always:
            packs[s] = ~std::uint64_t{0};
            break;
        case ev::bernoulli_kind::sampled:
            packs[s] = bernoulli::sample(source, s, law.threshold());
            break;
        }
    }
    return packs;
}

/// Pass `pass` of `events` in `stream`, `bitpacks_per_pass` bitpacks per node, from the host reference: the block's
/// bitpacks, slot after slot.
std::vector<std::uint64_t> block_reference(const prng::stream &stream, const std::vector<ev::basic_event<types>> &events, const std::uint32_t bitpacks_per_pass,
                                           const std::uint32_t pass) {
    std::vector<std::uint64_t> packs;
    packs.reserve(events.size() * bitpacks_per_pass);
    for (const auto &event : events) {
        const auto node = reference(stream, event.id_in_pdag, event.law, std::uint64_t{pass} * bitpacks_per_pass, bitpacks_per_pass);
        packs.insert(packs.end(), node.begin(), node.end());
    }
    return packs;
}

/// The bitpacks in which a block's `got` differs from `want`, `bitpacks_per_node` per node; the first eight are
/// reported with the next assertion.
std::size_t mismatches(const std::vector<std::uint64_t> &got, const std::vector<std::uint64_t> &want, const std::uint32_t bitpacks_per_node) {
    if (got.size() != want.size()) {
        UNSCOPED_INFO(got.size() << " bitpacks, want " << want.size());
        return std::max(got.size(), want.size());
    }
    std::size_t n = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i] && ++n <= 8) {
            UNSCOPED_INFO("node " << i / bitpacks_per_node << " of the block, bitpack " << i % bitpacks_per_node << ": got 0x" << std::hex << got[i]
                                  << ", want 0x" << want[i]);
        }
    }
    return n;
}

/// What a launch wrote to its block, slot after slot, and its kernel's work-items per node (whole work-groups).
struct launch_output {
    std::vector<std::uint64_t> bitpacks;
    std::size_t items_per_node = 0;
};

/// Pass `pass` of `events` in `stream`, `bitpacks_per_pass` bitpacks per node, on the device with the sampling kernel
/// `kernel_type`, made as the kernel builder makes it: the bitpacks of events with a constant law are written once, when
/// their slots are allocated (kernel::sampler::write_constant_outputs), and the kernel samples the other events. The arena holds two
/// guard slots, the events' slots and one more guard slot. Every bitpack of the arena is set to its sentinel word before
/// anything is written, and every bitpack outside the events' slots must still hold it after. Returns the events'
/// bitpacks in the order of `events`.
template <typename kernel_type>
launch_output sample_on_device(sycl::queue &queue, const prng::stream &stream, const std::vector<ev::basic_event<types>> &events,
                               const std::uint32_t bitpacks_per_pass, const std::uint32_t pass) {
    INFO("form=" << form_name(kernel_type::kForm) << " bitpacks per pass=" << bitpacks_per_pass << " pass=" << pass);
    std::vector<ev::sampled_event<types>> sampled;
    std::vector<ev::constant_event<types>> constant;
    for (const auto &event : events) {
        if (const auto random = ev::sampled_event<types>::of(event)) {
            sampled.push_back(*random);
        } else {
            constant.push_back(*ev::constant_event<types>::of(event));
        }
    }
    auto arena = ev::output_arena<types>::create(queue, events.size() + 3, bitpacks_per_pass);
    (void)arena.reserve(2);
    const auto sampled_block = ev::sampled_event_block<types>::create(queue, arena, sampled);
    const auto constant_block = ev::constant_event_block<types>::create(queue, arena, constant);
    (void)arena.reserve(1);
    const auto view = arena.view();

    std::vector<std::uint64_t> sentinel(view.size());
    for (std::size_t i = 0; i < sentinel.size(); ++i) {
        sentinel[i] = splitmix64(i);
    }
    queue.memcpy(view.data(), sentinel.data(), sentinel.size() * sizeof(std::uint64_t)).wait_and_throw();

    sampler::write_constant_outputs(queue, constant_block);
    std::size_t items_per_node = 0;
    if (!sampled.empty()) {
        const kernel_type sampling(sampled_block.view(), stream, queue.get_device());
        queue.parallel_for(sampling.get_range(), [=](sycl::nd_item<1> item) { sampling(item, pass); });
        queue.wait_and_throw();
        items_per_node = sampling.items_per_event();
    }

    std::vector<std::uint64_t> written(view.size());
    queue.memcpy(written.data(), view.data(), written.size() * sizeof(std::uint64_t)).wait_and_throw();

    const std::size_t words = bitpacks_per_pass;
    std::size_t changed = 0;
    for (std::uint32_t slot = 0; slot < arena.slots(); ++slot) {
        if (sampled_block.slots().contains(slot) || constant_block.slots().contains(slot)) {
            continue;
        }
        for (std::size_t w = 0; w < words; ++w) {
            const std::size_t i = slot * words + w;
            if (written[i] != sentinel[i] && ++changed <= 8) {
                UNSCOPED_INFO("guard slot " << slot << ", bitpack " << w << ": got 0x" << std::hex << written[i] << ", sentinel 0x" << sentinel[i]);
            }
        }
    }
    CHECK(changed == 0);

    launch_output out{.items_per_node = items_per_node};
    out.bitpacks.reserve(events.size() * words);
    std::size_t next_sampled = 0;
    std::size_t next_constant = 0;
    for (const auto &event : events) {
        const std::uint32_t slot = event.law.kind() == ev::bernoulli_kind::sampled ? sampled_block.slot(next_sampled++) : constant_block.slot(next_constant++);
        const auto first = written.begin() + static_cast<std::ptrdiff_t>(std::size_t{slot} * words);
        out.bitpacks.insert(out.bitpacks.end(), first, first + static_cast<std::ptrdiff_t>(words));
    }
    return out;
}

/// The Philox words of the 32 generations of bitpack `bitpack` of node `node`'s run in `stream`, at prng::counter.
std::array<prng::state128, bernoulli::kMaxGenerations> generations(const prng::stream &stream, const std::uint32_t node, const std::uint64_t bitpack) {
    const auto keys = prng::philox::key_schedule::of(stream.key);
    const prng::state128 base = prng::counter({.node = node, .device = stream.device, .bitpack = bitpack});
    std::array<prng::state128, bernoulli::kMaxGenerations> words{};
    for (std::uint32_t g = 0; g < bernoulli::kMaxGenerations; ++g) {
        prng::state128 c = base;
        c.x[3] += g;
        prng::philox::generate(c, keys);
        words[g] = c;
    }
    return words;
}

/// The uniforms of a bitpack's 64 trials, U_j = sum_k bit_j(r[k]) 2^(63-k), from its generations' words:
/// r[2g] = w0 | w1 << 32 and r[2g + 1] = w2 | w3 << 32 (as uniforms_from_words in bernoulli_sampler_tests.cc).
std::array<std::uint64_t, 64> uniforms(const std::array<prng::state128, bernoulli::kMaxGenerations> &words) {
    std::array<std::uint64_t, 64> u{};
    for (std::uint32_t k = 0; k < 64; ++k) {
        const prng::state128 &w = words[k / 2];
        const std::uint64_t r = k % 2 == 0 ? (std::uint64_t{w.x[0]} | (std::uint64_t{w.x[1]} << 32)) : (std::uint64_t{w.x[2]} | (std::uint64_t{w.x[3]} << 32));
        for (std::uint32_t j = 0; j < 64; ++j) {
            u[j] |= ((r >> j) & 1u) << (63 - k);
        }
    }
    return u;
}

/// Bit j is 1 iff U_j < T.
std::uint64_t brute_force(const std::array<std::uint64_t, 64> &u, const std::uint64_t threshold) {
    std::uint64_t bits = 0;
    for (std::uint32_t j = 0; j < 64; ++j) {
        bits |= std::uint64_t{u[j] < threshold} << j;
    }
    return bits;
}

} // namespace

TEST_CASE("basic_event bitpacks are bit-identical at fixed Philox counters", "[mc][bit-identity]") {
    const auto events = records();
    const auto expected = block_reference(kStream, events, kBitpacksPerPass, kPass);
    const auto host = fnv1a64(expected);
    INFO("host checksum=0x" << std::hex << host);
    REQUIRE(host == kExpectedChecksum);

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());
    CHECK(mismatches(sample_on_device<strided_kernel>(queue, kStream, events, kBitpacksPerPass, kPass).bitpacks, expected, kBitpacksPerPass) == 0);
    CHECK(mismatches(sample_on_device<lockstep_kernel>(queue, kStream, events, kBitpacksPerPass, kPass).bitpacks, expected, kBitpacksPerPass) == 0);
    CHECK(mismatches(sample_on_device<chunked_kernel>(queue, kStream, events, kBitpacksPerPass, kPass).bitpacks, expected, kBitpacksPerPass) == 0);
}

TEST_CASE("a node's bitpacks depend only on its counter coordinates", "[mc][bit-identity]") {
    // Node 6 (p = 0.3125) alone, and behind another node at a later slot, with passes of 40 and of 20 bitpacks: the
    // run-wide bitpacks [40·kPass, 40·kPass + 40) are the same bits every time.
    const auto law = *ev::bernoulli_law::from_probability(0.3125);
    const ev::basic_event<types> node{.id_in_pdag = 6, .law = law};
    const ev::basic_event<types> other{.id_in_pdag = 9, .law = law};
    const auto expected = reference(kStream, 6, law, std::uint64_t{kPass} * kBitpacksPerPass, kBitpacksPerPass);

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());
    const auto check = [&]<typename kernel_type>() {
        INFO("form=" << form_name(kernel_type::kForm));
        CHECK(mismatches(sample_on_device<kernel_type>(queue, kStream, {node}, kBitpacksPerPass, kPass).bitpacks, expected, kBitpacksPerPass) == 0);

        const auto behind = sample_on_device<kernel_type>(queue, kStream, {other, node}, kBitpacksPerPass, kPass).bitpacks;
        CHECK(mismatches(std::vector<std::uint64_t>(behind.begin() + kBitpacksPerPass, behind.end()), expected, kBitpacksPerPass) == 0);

        // Passes of 20 bitpacks: passes 2·kPass and 2·kPass + 1 cover the same run-wide bitpacks.
        auto halves = sample_on_device<kernel_type>(queue, kStream, {node}, kBitpacksPerPass / 2, 2 * kPass).bitpacks;
        const auto second = sample_on_device<kernel_type>(queue, kStream, {node}, kBitpacksPerPass / 2, 2 * kPass + 1).bitpacks;
        halves.insert(halves.end(), second.begin(), second.end());
        CHECK(mismatches(halves, expected, kBitpacksPerPass) == 0);
    };
    check.template operator()<strided_kernel>();
    check.template operator()<lockstep_kernel>();
    check.template operator()<chunked_kernel>();
}

TEST_CASE("work-items of the strided form sample runs of bitpacks on the device", "[mc][bit-identity]") {
    // One node (p = 0.3125) with B = 4 · 2048 · units bitpacks per pass, units the device's compute units: the strided
    // form gives each work-item a run of 4 bitpacks (kernel/sampler/strided.h), and its loop restarts on the next bitpack of
    // the run as soon as one is decided. B + 512 = 512 · (16 · units + 1) also gives runs of 4. Each pass is the one that
    // holds run-wide bitpack 3 · 2^32, the first whose index has high word 3. B + 512 has an odd factor of at least 17,
    // so it does not divide 3 · 2^32: in its pass the high word changes between restarts of some work-items' runs.
    sycl::queue queue;
    const sycl::device device = queue.get_device();
    const std::uint64_t units = std::max<std::uint64_t>(device.get_info<sycl::info::device::max_compute_units>(), 1);
    INFO("device=" << device.get_info<sycl::info::device::name>() << " compute units=" << units);
    const std::vector<ev::basic_event<types>> events{{.id_in_pdag = 6, .law = *ev::bernoulli_law::from_probability(0.3125)}};
    for (const std::uint64_t size : {4 * 2048 * units, 4 * 2048 * units + 512}) {
        const auto bitpacks = static_cast<std::uint32_t>(size);
        const auto pass = static_cast<std::uint32_t>((std::uint64_t{3} << 32) / size);
        INFO("bitpacks per pass=" << bitpacks << " pass=" << pass);
        const auto expected = block_reference(kStream, events, bitpacks, pass);

        const auto runs = sample_on_device<strided_kernel>(queue, kStream, events, bitpacks, pass);
        // At least 4 bitpacks per work-item: a geometry that gave fewer would not run the restarts this case is for.
        CHECK(size >= 4 * runs.items_per_node);
        CHECK(mismatches(runs.bitpacks, expected, bitpacks) == 0);
        CHECK(mismatches(sample_on_device<lockstep_kernel>(queue, kStream, events, bitpacks, pass).bitpacks, expected, bitpacks) == 0);
        CHECK(mismatches(sample_on_device<chunked_kernel>(queue, kStream, events, bitpacks, pass).bitpacks, expected, bitpacks) == 0);
    }
}

TEST_CASE("a bitpack tied to its threshold down to the last bit takes all 32 generations on the device", "[mc][bit-identity]") {
    // Bitpack 8 of the pass (run-wide index 3 · 2^32) of nodes 11 and 12. For each node, u is the uniform of the first
    // trial j of that bitpack whose last two bits are 01 or 10, from the bitpack's 32 generations of Philox words: a
    // uniform's last two bits come from generation 31. Node 11 samples T = u: trial j ties with T in all 64 bits and is
    // 0, and T's lowest set bit is one of its last two, so the comparison runs through generation 31. Node 12 samples
    // T = u + 1: trial j first differs from T in one of its last two bits, where it drops below T, so it is 1 only if
    // generation 31 is computed. Last bits 01 or 10 give 0 < u < 2^64 - 1, so both laws are `sampled`.
    constexpr std::uint32_t kTied = 8;
    std::vector<ev::basic_event<types>> events;
    std::array<std::size_t, 2> trial{};
    events.reserve(trial.size());
    for (std::size_t n = 0; n < trial.size(); ++n) {
        const auto id = static_cast<std::int32_t>(11 + n);
        const auto words = generations(kStream, static_cast<std::uint32_t>(id), std::uint64_t{kPass} * kBitpacksPerPass + kTied);
        const auto u = uniforms(words);
        trial[n] = static_cast<std::size_t>(std::ranges::find_if(u, [](const std::uint64_t x) { return ((x ^ (x >> 1)) & 1u) != 0; }) - u.begin());
        REQUIRE(trial[n] < u.size());
        const std::uint64_t threshold = n == 0 ? u[trial[n]] : u[trial[n]] + 1;
        INFO("node " << id << ": trial " << trial[n] << ", T=0x" << std::hex << threshold);

        // On the host, the bitpack asks for all 32 generations, agrees with the brute-force comparison and has trial j
        // 0 at T = u and 1 at T = u + 1.
        std::uint32_t asked = 0;
        const std::uint64_t evaluated = bernoulli::evaluate(
            [&](const std::uint32_t g) {
                asked = g + 1;
                return words[g];
            },
            threshold);
        REQUIRE(asked == bernoulli::kMaxGenerations);
        CHECK(evaluated == brute_force(u, threshold));
        CHECK(((evaluated >> trial[n]) & 1u) == n);

        events.push_back({.id_in_pdag = id, .law = ev::bernoulli_law::from_threshold(threshold)});
        REQUIRE(events.back().law.kind() == ev::bernoulli_kind::sampled);
        CHECK(reference(kStream, id, events.back().law, std::uint64_t{kPass} * kBitpacksPerPass, kBitpacksPerPass)[kTied] == evaluated);
    }
    const auto expected = block_reference(kStream, events, kBitpacksPerPass, kPass);

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>() << " trials " << trial[0] << " and " << trial[1]);
    const auto check = [&]<typename kernel_type>() {
        INFO("form=" << form_name(kernel_type::kForm));
        const auto got = sample_on_device<kernel_type>(queue, kStream, events, kBitpacksPerPass, kPass).bitpacks;
        CHECK(mismatches(got, expected, kBitpacksPerPass) == 0);
        // Trial j of bitpack 8: 0 at T = u (U_j < T fails at equality), 1 at T = u + 1 (decided in generation 31).
        CHECK(((got[kTied] >> trial[0]) & 1u) == 0u);
        CHECK(((got[kBitpacksPerPass + kTied] >> trial[1]) & 1u) == 1u);
    };
    check.template operator()<strided_kernel>();
    check.template operator()<lockstep_kernel>();
    check.template operator()<chunked_kernel>();
}

TEST_CASE("the seed and the device select distinct streams", "[mc][bit-identity]") {
    const auto law = *ev::bernoulli_law::from_probability(0.5);
    const std::uint64_t first = std::uint64_t{kPass} * kBitpacksPerPass;
    const auto bits = [&](const prng::stream &stream) { return reference(stream, 6, law, first, kBitpacksPerPass); };
    const auto base = bits(kStream);
    CHECK(bits(kStream) == base);
    CHECK(bits(prng::stream::of(kStream.seed(), kStream.device + 1)) != base);
    CHECK(bits(prng::stream::of(kStream.seed() + 1, kStream.device)) != base);
    CHECK(prng::stream{}.seed() == prng::stream::kDefaultSeed);
}

TEST_CASE("kernels of streams that differ in seed or in device write their own bitpacks", "[mc][bit-identity]") {
    // In one process, each form runs with kStream, the next seed, the next device and kStream again. The stream is a
    // specialized kernel argument (kernel/sampler/sampler.h), which the generic JIT compiles into the kernel once per stream
    // and form: a kernel compiled for one stream and launched with another would write the first stream's bitpacks.
    const auto events = records();
    const auto expected = block_reference(kStream, events, kBitpacksPerPass, kPass);
    const std::array<prng::stream, 2> others{prng::stream::of(kStream.seed() + 1, kStream.device), prng::stream::of(kStream.seed(), kStream.device + 1)};

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());
    const auto check = [&]<typename kernel_type>() {
        INFO("form=" << form_name(kernel_type::kForm));
        const auto base = sample_on_device<kernel_type>(queue, kStream, events, kBitpacksPerPass, kPass).bitpacks;
        CHECK(mismatches(base, expected, kBitpacksPerPass) == 0);
        for (const prng::stream &other : others) {
            INFO("seed=0x" << std::hex << other.seed() << std::dec << " device=" << other.device);
            const auto bits = sample_on_device<kernel_type>(queue, other, events, kBitpacksPerPass, kPass).bitpacks;
            CHECK(mismatches(bits, block_reference(other, events, kBitpacksPerPass, kPass), kBitpacksPerPass) == 0);
            CHECK(bits != base);
        }
        CHECK(mismatches(sample_on_device<kernel_type>(queue, kStream, events, kBitpacksPerPass, kPass).bitpacks, expected, kBitpacksPerPass) == 0);
    };
    check.template operator()<strided_kernel>();
    check.template operator()<lockstep_kernel>();
    check.template operator()<chunked_kernel>();
}
