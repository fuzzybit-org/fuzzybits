#include <catch2/catch_test_macros.hpp>

#include "mc/event/node.h"
#include "mc/kernel/basic_event.h"
#include "mc/prng/bernoulli.h"

#include "pcfb/threshold.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <ios>
#include <sycl/sycl.hpp>
#include <tuple>
#include <vector>

namespace {

using bitpack_t = std::uint64_t;
using prob_t = double;
using size_t_ = std::size_t;
using scram::mc::kernel::sampler_form;

constexpr std::size_t kNumEvents = 6;
// 40 bitpacks per event: the lanes form ends on a partial group of 8.
constexpr std::size_t kBatchSize = 4;
constexpr std::size_t kBitpacksPerBatch = 10;
constexpr std::uint32_t kIteration = 3;

// (event_idx, pdag_idx, p): ordinary, half, never, always, a few significant threshold bits, and
// p < 2^-32 (threshold high word 0), which the 32-bit sampler collapsed to 0 (#17).
constexpr std::array<std::tuple<std::uint32_t, std::uint32_t, double>, kNumEvents> kEvents{{
    {0, 2, 0.1},
    {1, 3, 0.5},
    {2, 4, 0.0},
    {3, 5, 1.0},
    {4, 6, 0.3125},
    {5, 7, 1e-12},
}};

// FNV-1a 64 of the 240 little-endian uint64 bitpacks of the bit-sliced sampler
// (prng/bernoulli.h) at the counters above. Every backend and both kernel forms must
// reproduce it; a change to the random stream fails this test everywhere.
constexpr std::uint64_t kExpectedChecksum = 0xdefa0692846b7d85ull;

std::uint64_t fnv1a64(const std::vector<bitpack_t> &data) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (const bitpack_t v : data) {
        unsigned char bytes[8];
        std::memcpy(bytes, &v, 8);
        for (unsigned char b : bytes) {
            h ^= b;
            h *= 0x100000001b3ull;
        }
    }
    return h;
}

std::vector<bitpack_t> host_reference() {
    namespace bernoulli = scram::mc::prng::bernoulli;
    std::vector<bitpack_t> packs;
    for (const auto &[event_idx, pdag_idx, p] : kEvents) {
        const auto threshold = pcfb::probability_to_threshold(p);
        for (std::uint32_t s = 0; s < kBatchSize * kBitpacksPerBatch; ++s) {
            packs.push_back(p >= 1.0
                                ? ~bitpack_t{0}
                                : bernoulli::sample(bernoulli::counter(pdag_idx, event_idx, s, kIteration), threshold));
        }
    }
    return packs;
}

template <sampler_form form_> std::vector<bitpack_t> device_bitpacks(sycl::queue &queue) {
    using block_t = scram::mc::event::basic_event_block<prob_t, bitpack_t>;
    using kernel_t = scram::mc::kernel::basic_event<prob_t, bitpack_t, size_t_, form_>;

    std::vector<std::pair<std::int32_t, prob_t>> indexed;
    for (const auto &[event_idx, pdag_idx, p] : kEvents) {
        (void)event_idx;
        indexed.emplace_back(static_cast<std::int32_t>(pdag_idx), p);
    }
    const scram::mc::event::sample_shape<size_t_> shape{kBatchSize, kBitpacksPerBatch};
    block_t block = scram::mc::event::create_basic_event_block<prob_t, bitpack_t>(queue, indexed, shape.num_bitpacks());

    const kernel_t kernel(block, shape, queue.get_device());
    queue.parallel_for(kernel.get_range(), [=](sycl::nd_item<1> item) { kernel(item, kIteration); });
    queue.wait_and_throw();

    std::vector<bitpack_t> host(kNumEvents * shape.num_bitpacks());
    queue.memcpy(host.data(), block.buffers, host.size() * sizeof(bitpack_t)).wait_and_throw();
    scram::mc::event::destroy_basic_event_block(queue, block);
    return host;
}

} // namespace

TEST_CASE("basic_event bitpacks are bit-identical at fixed Philox counters", "[mc][bit-identity]") {
    const auto reference = host_reference();
    const auto host = fnv1a64(reference);
    INFO("host checksum=0x" << std::hex << host);
    REQUIRE(host == kExpectedChecksum);

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());
    CHECK(device_bitpacks<sampler_form::strided>(queue) == reference);
    CHECK(device_bitpacks<sampler_form::lanes>(queue) == reference);
}
