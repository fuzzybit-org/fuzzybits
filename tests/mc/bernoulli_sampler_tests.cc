// Host tests of the bit-sliced Bernoulli sampler (src/mc/prng/bernoulli.h): Philox4x32-10 known answers,
// exactness of the lazy comparison down to the last threshold bit, agreement of the GPU and CPU forms with
// the reference, and frequency, bit-position and neighbour-independence checks. The kernels' device output
// is compared with this reference in bitpack_identity_tests.cc, so these statistics carry over to every
// backend. Inputs are fixed counters, so results are deterministic.

#include <catch2/catch_test_macros.hpp>

#include "mc/prng/bernoulli.h"
#include "mc/prng/philox128.h"

#include "pcfb/threshold.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

namespace bernoulli = scram::mc::prng::bernoulli;
namespace philox = scram::mc::prng::philox;
using scram::mc::prng::state128;

// SplitMix64, for test inputs only.
struct splitmix64 {
    std::uint64_t s;
    std::uint64_t operator()() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};

// U_j for all 64 trials of a bitpack, from all 64 words: U_j = sum_k bit_j(r[k]) 2^(63-k).
std::array<std::uint64_t, 64> uniforms_from_words(const std::array<state128, bernoulli::kMaxGenerations> &w) {
    std::array<std::uint64_t, 64> u{};
    for (std::uint32_t k = 0; k < 64; ++k) {
        const state128 &g = w[k / 2];
        const std::uint64_t r = k % 2 == 0 ? (std::uint64_t{g.x[0]} | (std::uint64_t{g.x[1]} << 32))
                                           : (std::uint64_t{g.x[2]} | (std::uint64_t{g.x[3]} << 32));
        for (int j = 0; j < 64; ++j) {
            u[j] |= ((r >> j) & 1u) << (63 - k);
        }
    }
    return u;
}

// The 64 words whose bit-slices are the given uniforms (inverse of uniforms_from_words).
std::array<state128, bernoulli::kMaxGenerations> words_from_uniforms(const std::array<std::uint64_t, 64> &u) {
    std::array<state128, bernoulli::kMaxGenerations> w{};
    for (std::uint32_t k = 0; k < 64; ++k) {
        std::uint64_t r = 0;
        for (int j = 0; j < 64; ++j) {
            r |= ((u[j] >> (63 - k)) & 1u) << j;
        }
        state128 &g = w[k / 2];
        g.x[(k % 2) * 2] = static_cast<std::uint32_t>(r);
        g.x[(k % 2) * 2 + 1] = static_cast<std::uint32_t>(r >> 32);
    }
    return w;
}

std::uint64_t brute_force(const std::array<std::uint64_t, 64> &u, const std::uint64_t threshold) {
    std::uint64_t bits = 0;
    for (int j = 0; j < 64; ++j) {
        bits |= std::uint64_t{u[j] < threshold} << j;
    }
    return bits;
}

std::array<state128, bernoulli::kMaxGenerations> philox_words(const state128 &base) {
    std::array<state128, bernoulli::kMaxGenerations> w{};
    for (std::uint32_t g = 0; g < bernoulli::kMaxGenerations; ++g) {
        state128 c = base;
        c.x[3] += g;
        w[g] = philox::generate(c);
    }
    return w;
}

// One-sided upper 1e-7 quantile of chi-square with k degrees of freedom (Wilson-Hilferty).
double chi2_limit(const double k) {
    const double z = 5.2;
    const double a = 2.0 / (9.0 * k);
    return k * std::pow(1.0 - a + z * std::sqrt(a), 3.0);
}

constexpr double kZLimit = 5.5; // two-sided p ~ 4e-8 per check

} // namespace

TEST_CASE("Philox4x32-10 matches the Random123 known-answer vectors", "[mc][prng]") {
    struct kat {
        state128 counter;
        philox::key64 key;
        state128 expected;
    };
    // Random123 kat_vectors, philox4x32 10.
    const std::array<kat, 3> kats{{
        {{{0u, 0u, 0u, 0u}}, {0u, 0u}, {{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}}},
        {{{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}},
         {0xffffffffu, 0xffffffffu},
         {{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}}},
        {{{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}},
         {0xa4093822u, 0x299f31d0u},
         {{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}}},
    }};
    for (const auto &k : kats) {
        const state128 out = philox::generate(k.counter, k.key);
        for (int i = 0; i < 4; ++i) {
            CHECK(out.x[i] == k.expected.x[i]);
        }
        // The lane-wise vector form computes the same function in every lane.
        philox::u32xN<8> x[4];
        for (int i = 0; i < 4; ++i) {
            x[i] = k.counter.x[i];
        }
        philox::generate_lanes<8>(x, k.key);
        for (int lane = 0; lane < 8; ++lane) {
            for (int i = 0; i < 4; ++i) {
                CHECK(x[i][lane] == k.expected.x[i]);
            }
        }
    }
}

TEST_CASE("the lazy comparison decides U < T exactly at every depth", "[mc][prng][exact]") {
    splitmix64 rng{12345};
    for (int round = 0; round < 2000; ++round) {
        const std::uint64_t threshold = rng() >> (rng() % 64);
        std::array<std::uint64_t, 64> u{};
        for (int j = 0; j < 64; ++j) {
            // Share a prefix of random length with T, so ties end at every depth, including none.
            const auto shared = static_cast<int>(rng() % 65);
            const std::uint64_t keep = shared == 0 ? 0 : (shared == 64 ? ~0ull : ~0ull << (64 - shared));
            u[j] = (threshold & keep) | (rng() & ~keep);
        }
        u[0] = threshold; // equal: not below
        if (threshold > 0) {
            u[1] = threshold - 1; // just below
        }
        if (threshold < ~0ull) {
            u[2] = threshold + 1; // just above
        }
        const auto words = words_from_uniforms(u);
        const std::uint64_t got = bernoulli::evaluate([&words](const std::uint32_t g) { return words[g]; }, threshold);
        REQUIRE(got == brute_force(u, threshold));
    }
}

TEST_CASE("Philox bitpacks are exact for thresholds tied to a trial over all 64 bits", "[mc][prng][exact]") {
    // T equal to one trial's uniform (that trial is 0) or one above it (that trial is 1): the comparison of that
    // trial runs to the last bit of T, which a 32-bit sampler never reaches (#17).
    for (std::uint32_t sample_idx = 0; sample_idx < 64; ++sample_idx) {
        const state128 base = bernoulli::counter(7, 5, sample_idx, 11);
        const auto u = uniforms_from_words(philox_words(base));
        const auto j = static_cast<int>(sample_idx % 64);
        for (const std::uint64_t threshold : {u[j], u[j] + 1, u[j] >> 40, (u[j] >> 40) + 1}) {
            if (threshold == 0) {
                continue;
            }
            REQUIRE(bernoulli::sample(base, threshold) == brute_force(u, threshold));
        }
    }
    // p < 2^-32 has threshold high word 0 and is not 0 (#17).
    CHECK(pcfb::probability_to_threshold(1e-12) == 18446744ull);
}

TEST_CASE("the strided and lanes forms reproduce the reference bitpacks", "[mc][prng]") {
    constexpr std::uint32_t kPacks = 203; // not a multiple of the lane widths
    const std::uint32_t pdag_idx = 9;
    const std::uint32_t event_idx = 4;
    const std::uint32_t iteration = 2;
    for (const double p : {0.5, 0.3125, 0.1, 1e-3, 1e-12, 1.0 - 1e-9}) {
        const auto threshold = pcfb::probability_to_threshold(p);
        std::vector<std::uint64_t> reference(kPacks);
        for (std::uint32_t s = 0; s < kPacks; ++s) {
            reference[s] = bernoulli::sample(bernoulli::counter(pdag_idx, event_idx, s, iteration), threshold);
        }
        for (const std::uint32_t stride : {1u, 3u, 7u, 32u, 500u}) {
            std::vector<std::uint64_t> out(kPacks, 0xdeadbeefull);
            for (std::uint32_t first = 0; first < std::min(stride, kPacks); ++first) {
                bernoulli::sample_strided(out.data(), first, stride, kPacks, pdag_idx, event_idx, iteration, threshold);
            }
            CHECK(out == reference);
        }
        auto lanes = [&]<int W>() {
            std::vector<std::uint64_t> out(kPacks, 0xdeadbeefull);
            for (std::uint32_t first = 0; first < kPacks; first += W) {
                const auto valid = std::min<std::uint32_t>(W, kPacks - first);
                bernoulli::sample_lanes<W>(out.data() + first, valid, first, pdag_idx, event_idx, iteration, threshold);
            }
            return out;
        };
        CHECK(lanes.template operator()<4>() == reference);
        CHECK(lanes.template operator()<8>() == reference);
        CHECK(lanes.template operator()<16>() == reference);
    }
}

TEST_CASE("trials are Bernoulli(T / 2^64), uniform over bit positions and independent of neighbours",
          "[mc][prng][statistics]") {
    constexpr std::uint32_t kPacks = 1u << 16; // 4 Mi trials per probability
    for (const double p : {0.5, 0.3125, 0.1, 0.01, 1e-3, 0.999}) {
        INFO("p=" << p);
        const auto threshold = pcfb::probability_to_threshold(p);
        const double pe = std::ldexp(static_cast<double>(threshold), -64);
        // Bitpacks of two adjacent events and two adjacent iterations at the same samples.
        std::vector<std::uint64_t> a(kPacks), b(kPacks), c(kPacks);
        for (std::uint32_t s = 0; s < kPacks; ++s) {
            a[s] = bernoulli::sample(bernoulli::counter(3, 1, s, 0), threshold);
            b[s] = bernoulli::sample(bernoulli::counter(4, 2, s, 0), threshold);
            c[s] = bernoulli::sample(bernoulli::counter(3, 1, s, 1), threshold);
        }
        auto z_of = [](const double hits, const double n, const double q) {
            return (hits - n * q) / std::sqrt(n * q * (1.0 - q));
        };

        // Frequency.
        double ones = 0;
        std::array<double, 64> per_bit{};
        for (const auto v : a) {
            ones += std::popcount(v);
            for (int j = 0; j < 64; ++j) {
                per_bit[j] += static_cast<double>((v >> j) & 1u);
            }
        }
        CHECK(std::fabs(z_of(ones, 64.0 * kPacks, pe)) < kZLimit);

        // Bit positions: 64 independent binomial cells, chi-square with 64 degrees of freedom.
        double chi2 = 0;
        for (int j = 0; j < 64; ++j) {
            const double e = kPacks * pe;
            chi2 += (per_bit[j] - e) * (per_bit[j] - e) / (e * (1.0 - pe));
        }
        CHECK(chi2 < chi2_limit(64));

        // Pairs: adjacent bits of a bitpack, the same bit of adjacent samples, adjacent events and adjacent
        // iterations. Joint ones must match p^2.
        double in_pack = 0, across_samples = 0, across_events = 0, across_iterations = 0;
        for (std::uint32_t s = 0; s < kPacks; ++s) {
            in_pack += std::popcount(a[s] & (a[s] >> 1) & 0x7FFFFFFFFFFFFFFFull);
            if (s + 1 < kPacks) {
                across_samples += std::popcount(a[s] & a[s + 1]);
            }
            across_events += std::popcount(a[s] & b[s]);
            across_iterations += std::popcount(a[s] & c[s]);
        }
        CHECK(std::fabs(z_of(in_pack, 63.0 * kPacks, pe * pe)) < kZLimit);
        CHECK(std::fabs(z_of(across_samples, 64.0 * (kPacks - 1), pe * pe)) < kZLimit);
        CHECK(std::fabs(z_of(across_events, 64.0 * kPacks, pe * pe)) < kZLimit);
        CHECK(std::fabs(z_of(across_iterations, 64.0 * kPacks, pe * pe)) < kZLimit);
    }
}

static_assert(bernoulli::kMaxGenerations <= 64, "generations must fit the counter's 6-bit field");
