// Host tests of the bit-sliced Bernoulli sampler (src/mc/prng/bernoulli/) at the Philox keys and counters of
// src/mc/prng/stream.h: Philox4x32-10 known answers and its round-key schedule against its key, the default stream, the
// layout of the key and counter, a pass's counters against the run's, one bitpack at a time and lane by lane, exactness
// of the lazy comparison down to the last threshold bit, agreement of the strided, lockstep and chunked forms with the
// reference, also where a pass's bitpack indices cross a 2^32 boundary, and frequency, bit-position and
// neighbour-independence checks. The kernels' device output is compared with this reference in
// bitpack_identity_tests.cc, so these statistics carry over to every backend. Inputs are fixed seeds and counters, so
// results are deterministic.

#include <catch2/catch_test_macros.hpp>

#include "mc/prng/bernoulli/bernoulli.h"
#include "mc/prng/philox128.h"
#include "mc/prng/prng.h"
#include "mc/prng/stream.h"

#include "pcfb/threshold.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace prng = fuzzybits::mc::prng;
namespace bernoulli = prng::bernoulli;
namespace philox = prng::philox;
using prng::state128;

// The helpers below take a bitpack's 64 random words from its generations, two per generation.
static_assert(2 * bernoulli::kMaxGenerations == 64, "a generation yields two of a bitpack's 64 random words");

// The sampler's forms by Philox variant: one form on the scalar variant, two on the vector variant.
static_assert(bernoulli::variant_of(bernoulli::form::strided) == prng::variant::scalar);
static_assert(bernoulli::variant_of(bernoulli::form::lockstep) == prng::variant::vector);
static_assert(bernoulli::variant_of(bernoulli::form::chunked) == prng::variant::vector);

// Neither the default seed nor device 0, so a form that used another key or device than its pass_stream's would
// differ from the reference.
constexpr prng::stream kStream = prng::stream::of(0xa4093822299f31d0ull, 5);

// The run-wide bitpack index at which counter word 2 wraps to 0 and the index's high word, in word 3, becomes 1.
constexpr std::uint64_t kWrap = std::uint64_t{1} << 32;

// The high words a bitpack index can have: 2^26, above the generation in counter word 3.
constexpr std::uint64_t kHighWords = prng::kMaxBitpacks / kWrap;

// A counter's words, as an array Catch2 compares and prints.
constexpr std::array<std::uint32_t, 4> words_of(const state128 &c) { return {c.x[0], c.x[1], c.x[2], c.x[3]}; }

// Lane `l` of W lane-wise counters (the vector variant's state).
template <int W> std::array<std::uint32_t, 4> lane_of(const prng::state<prng::variant::vector, W> &c, const int l) {
    return {c.x[0][l], c.x[1][l], c.x[2][l], c.x[3][l]};
}

// Philox4x32-10 of one counter, by value.
template <class Key> state128 philox_of(state128 c, const Key &key) {
    philox::generate(c, key);
    return c;
}

// A schedule's round keys, k0 of rounds 0-9 then k1 of rounds 0-9, as an array Catch2 compares and prints.
constexpr std::array<std::uint32_t, 2 * philox::kRounds> round_keys_of(const philox::key_schedule &s) {
    std::array<std::uint32_t, 2 * philox::kRounds> keys{};
    for (std::uint32_t i = 0; i < philox::kRounds; ++i) {
        keys[i] = s.k0[i];
        keys[philox::kRounds + i] = s.k1[i];
    }
    return keys;
}

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

// A key from one draw, low word first.
philox::key64 random_key(splitmix64 &rng) {
    const std::uint64_t r = rng();
    return {static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(r >> 32)};
}

// A counter from two draws.
state128 random_counter(splitmix64 &rng) {
    const std::uint64_t a = rng();
    const std::uint64_t b = rng();
    return state128{{static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(a >> 32), static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(b >> 32)}};
}

// U_j for all 64 trials of a bitpack, from all 64 words: U_j = sum_k bit_j(r[k]) 2^(63-k).
std::array<std::uint64_t, 64> uniforms_from_words(const std::array<state128, bernoulli::kMaxGenerations> &w) {
    std::array<std::uint64_t, 64> u{};
    for (std::uint32_t k = 0; k < 64; ++k) {
        const state128 &g = w[k / 2];
        const std::uint64_t r = k % 2 == 0 ? (std::uint64_t{g.x[0]} | (std::uint64_t{g.x[1]} << 32)) : (std::uint64_t{g.x[2]} | (std::uint64_t{g.x[3]} << 32));
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

// The generations of the bitpack at counter `base` under `key`.
std::array<state128, bernoulli::kMaxGenerations> philox_words(const state128 &base, const philox::key64 key) {
    std::array<state128, bernoulli::kMaxGenerations> w{};
    for (std::uint32_t g = 0; g < bernoulli::kMaxGenerations; ++g) {
        state128 c = base;
        c.x[3] += g;
        w[g] = philox_of(c, key);
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
        {{{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}}, {0xffffffffu, 0xffffffffu}, {{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}}},
        {{{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}}, {0xa4093822u, 0x299f31d0u}, {{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}}},
    }};
    for (const auto &k : kats) {
        // Under the key, and under its round keys expanded once, as the kernels hold them.
        const philox::key_schedule keys = philox::key_schedule::of(k.key);
        CHECK(words_of(philox_of(k.counter, k.key)) == words_of(k.expected));
        CHECK(words_of(philox_of(k.counter, keys)) == words_of(k.expected));
        // The vector variant computes the same function in every lane, at 2 to 32 lanes and under the key or its
        // schedule: with the known-answer counter in every lane, and with lane l's counter l more in every word,
        // which the pairing of even and odd lanes into 64-bit products must keep apart.
        const auto lanes = [&k]<int W>(const auto &key, const std::string_view form) {
            prng::lanes<W> lane{};
            for (int l = 0; l < W; ++l) {
                lane[l] = static_cast<std::uint32_t>(l);
            }
            prng::state<prng::variant::vector, W> same{};
            prng::state<prng::variant::vector, W> apart{};
            for (int i = 0; i < 4; ++i) {
                same.x[i] = k.counter.x[i];
                apart.x[i] = lane + k.counter.x[i];
            }
            philox::generate(same, key);
            philox::generate(apart, key);
            for (int l = 0; l < W; ++l) {
                INFO(form << " W=" << W << " lane=" << l);
                state128 c = k.counter;
                for (auto &word : c.x) {
                    word += static_cast<std::uint32_t>(l);
                }
                CHECK(lane_of<W>(same, l) == words_of(k.expected));
                CHECK(lane_of<W>(apart, l) == words_of(philox_of(c, k.key)));
            }
        };
        lanes.template operator()<2>(k.key, "key");
        lanes.template operator()<4>(k.key, "key");
        lanes.template operator()<8>(k.key, "key");
        lanes.template operator()<16>(k.key, "key");
        lanes.template operator()<32>(k.key, "key");
        lanes.template operator()<2>(keys, "schedule");
        lanes.template operator()<4>(keys, "schedule");
        lanes.template operator()<8>(keys, "schedule");
        lanes.template operator()<16>(keys, "schedule");
        lanes.template operator()<32>(keys, "schedule");
    }
}

TEST_CASE("a key schedule holds the key plus i Weyl increments in round i", "[mc][prng]") {
    // Random123's Philox adds (kWeyl0, kWeyl1) to the key after each round, mod 2^32 in each word, so round i takes the
    // key plus i increments. Keys with no bit and every bit set, the engine's fixed key before runs had seeds, the
    // default stream's, kStream's, and random keys.
    splitmix64 rng{0x6b65795f73636864ull};
    std::vector<philox::key64> keys{{0u, 0u}, {~0u, ~0u}, {382307844u, 293830103u}, prng::stream{}.key, kStream.key};
    for (int i = 0; i < 256; ++i) {
        keys.push_back(random_key(rng));
    }
    for (const philox::key64 key : keys) {
        INFO("key=(" << key.k0 << ", " << key.k1 << ")");
        std::array<std::uint32_t, 2 * philox::kRounds> bumped{};
        philox::key64 round_key = key;
        for (std::uint32_t i = 0; i < philox::kRounds; ++i) {
            bumped[i] = round_key.k0;
            bumped[philox::kRounds + i] = round_key.k1;
            round_key.k0 += philox::kWeyl0;
            round_key.k1 += philox::kWeyl1;
        }
        CHECK(round_keys_of(philox::key_schedule::of(key)) == bumped);
    }

    // Equality compares every round key of both words.
    const philox::key_schedule schedule = philox::key_schedule::of(kStream.key);
    CHECK(schedule == philox::key_schedule::of(kStream.key));
    CHECK_FALSE(schedule == philox::key_schedule::of(prng::stream{}.key));
    for (std::uint32_t i = 0; i < philox::kRounds; ++i) {
        INFO("round=" << i);
        philox::key_schedule other = schedule;
        other.k0[i] ^= 1u;
        CHECK_FALSE(other == schedule);
        other = schedule;
        other.k1[i] ^= 0x80000000u;
        CHECK_FALSE(other == schedule);
    }
}

TEST_CASE("Philox under a key's schedule equals Philox under the key, in the scalar variant and lane by lane", "[mc][prng]") {
    // Random keys, and a random counter in every lane. The arbitrary schedule holds the round keys of no key, so the
    // vector variant must also take each round's keys from the same words of the schedule as the scalar variant.
    splitmix64 rng{0x67656e6572617465ull};
    for (int round = 0; round < 128; ++round) {
        INFO("round=" << round);
        const philox::key64 key = random_key(rng);
        const philox::key_schedule schedule = philox::key_schedule::of(key);
        philox::key_schedule arbitrary{};
        for (std::uint32_t i = 0; i < philox::kRounds; ++i) {
            const philox::key64 words = random_key(rng);
            arbitrary.k0[i] = words.k0;
            arbitrary.k1[i] = words.k1;
        }
        const auto lanes = [&]<int W>() {
            std::array<state128, W> counters{};
            for (auto &counter : counters) {
                counter = random_counter(rng);
            }
            prng::state<prng::variant::vector, W> under_key{};
            for (int i = 0; i < 4; ++i) {
                for (int l = 0; l < W; ++l) {
                    under_key.x[i][l] = counters[l].x[i];
                }
            }
            prng::state<prng::variant::vector, W> under_schedule = under_key;
            prng::state<prng::variant::vector, W> under_arbitrary = under_key;
            philox::generate(under_key, key);
            philox::generate(under_schedule, schedule);
            philox::generate(under_arbitrary, arbitrary);
            for (int l = 0; l < W; ++l) {
                INFO("W=" << W << " lane=" << l);
                const auto scalar = words_of(philox_of(counters[l], key));
                CHECK(words_of(philox_of(counters[l], schedule)) == scalar);
                CHECK(lane_of<W>(under_key, l) == scalar);
                CHECK(lane_of<W>(under_schedule, l) == scalar);
                CHECK(lane_of<W>(under_arbitrary, l) == words_of(philox_of(counters[l], arbitrary)));
            }
        };
        lanes.template operator()<4>();
        lanes.template operator()<8>();
        lanes.template operator()<16>();
    }
}

TEST_CASE("the default stream is seed 372 on device 0, and a stream expands to its key's round keys", "[mc][prng]") {
    // The seed of every entry point that names none (prng/stream.h).
    const prng::stream defaults{};
    CHECK(prng::stream::kDefaultSeed == 372u);
    CHECK(defaults.seed() == 372u);
    CHECK(defaults.key.k0 == 372u);
    CHECK(defaults.key.k1 == 0u);
    CHECK(defaults.device == 0u);
    CHECK(prng::stream::of(372) == defaults);

    // of(seed, device) keeps both, the seed's low word in k0, and names device 0 when given none. Streams that differ in
    // a key word or the device are unequal. The expanded stream holds the key's schedule and the device.
    splitmix64 rng{0x73747265616d73ull};
    std::vector<std::uint64_t> seeds{0, 1, 372, kWrap - 1, kWrap, kWrap + 372, ~0ull, kStream.seed()};
    for (int i = 0; i < 64; ++i) {
        seeds.push_back(rng());
    }
    for (const std::uint64_t seed : seeds) {
        const auto random_device = static_cast<std::uint32_t>(rng());
        for (const std::uint32_t device : {0u, 1u, 5u, ~0u, random_device}) {
            INFO("seed=" << seed << " device=" << device);
            const auto run = prng::stream::of(seed, device);
            CHECK(run.seed() == seed);
            CHECK(run.key.k0 == static_cast<std::uint32_t>(seed));
            CHECK(run.key.k1 == static_cast<std::uint32_t>(seed >> 32));
            CHECK(run.device == device);
            CHECK_FALSE(run == prng::stream::of(seed ^ 1u, device));
            CHECK_FALSE(run == prng::stream::of(seed ^ kWrap, device));
            CHECK_FALSE(run == prng::stream::of(seed, device ^ 1u));
            const auto expanded = prng::expanded_stream::of(run);
            CHECK(round_keys_of(expanded.keys) == round_keys_of(philox::key_schedule::of(run.key)));
            CHECK(expanded.device == device);
        }
        INFO("seed=" << seed);
        CHECK(prng::stream::of(seed) == prng::stream::of(seed, 0));
    }
}

TEST_CASE("a bitpack's Philox key and counter have the layout prng/stream.h documents", "[mc][prng]") {
    // key = seed, low word first. The device is a counter word, not part of the key.
    const auto stream = prng::stream::of(0x0123456789abcdefull, 3);
    CHECK(stream.key.k0 == 0x89abcdefu);
    CHECK(stream.key.k1 == 0x01234567u);
    CHECK(stream.device == 3u);
    CHECK(stream.seed() == 0x0123456789abcdefull);
    CHECK(prng::stream::of(prng::stream::kDefaultSeed) == prng::stream{});

    // counter = (node, device, b mod 2^32, (b >> 32) << 6), with no offsets and no hashing.
    struct layout {
        prng::coordinates at;
        std::array<std::uint32_t, 4> words;
    };
    constexpr std::array<layout, 7> kLayouts{{
        {{.node = 0, .device = 0, .bitpack = 0}, {0u, 0u, 0u, 0u}},
        {{.node = 7, .device = 5, .bitpack = 11}, {7u, 5u, 11u, 0u}},
        // Node and device keep all 32 bits; the last index below 2^32 leaves word 3 at 0.
        {{.node = ~0u, .device = ~0u, .bitpack = kWrap - 1}, {~0u, ~0u, ~0u, 0u}},
        {{.node = 1, .device = 2, .bitpack = kWrap}, {1u, 2u, 0u, 0x40u}},
        {{.node = 1, .device = 2, .bitpack = kWrap + 3}, {1u, 2u, 3u, 0x40u}},
        // High word 0x01234567 in bits 6-31 of word 3.
        {{.node = 3, .device = 1, .bitpack = 0x0123456789abcdefull}, {3u, 1u, 0x89abcdefu, 0x48d159c0u}},
        // The last bitpack index of a run: high word 2^26 - 1 fills bits 6-31 of word 3.
        {{.node = 3, .device = 1, .bitpack = prng::kMaxBitpacks - 1}, {3u, 1u, 0xffffffffu, 0xffffffc0u}},
    }};
    constexpr std::uint32_t generation_mask = (1u << prng::kGenerationBits) - 1;
    for (const auto &[at, words] : kLayouts) {
        INFO("node=" << at.node << " device=" << at.device << " bitpack=" << at.bitpack);
        const state128 c = prng::counter(at);
        CHECK(words_of(c) == words);
        // Generation g adds g to word 3, whose low kGenerationBits the bitpack leaves clear, so up to the last
        // generation the sum keeps the bitpack's high word above and g below: no two (bitpack, generation) pairs of a
        // node share a counter.
        const std::uint32_t last = c.x[3] + (bernoulli::kMaxGenerations - 1);
        CHECK((c.x[3] & generation_mask) == 0u);
        CHECK(last >> prng::kGenerationBits == static_cast<std::uint32_t>(at.bitpack >> 32));
        CHECK((last & generation_mask) == bernoulli::kMaxGenerations - 1);
    }

    // Bitpack s of a pass is the run's bitpack first + s: s is added to the low word, its carry to the high word.
    const bernoulli::pass_stream pass(kStream, 9, kWrap - 1);
    CHECK(words_of(pass.counter_of(0)) == std::array<std::uint32_t, 4>{9u, kStream.device, 0xffffffffu, 0u});
    CHECK(words_of(pass.counter_of(1)) == std::array<std::uint32_t, 4>{9u, kStream.device, 0u, 0x40u});
    CHECK(words_of(pass.counter_of(~0u)) == std::array<std::uint32_t, 4>{9u, kStream.device, 0xfffffffeu, 0x40u});
}

TEST_CASE("bitpack s of a pass has the run's counter of bitpack first + s, also across 32-bit wraps", "[mc][prng]") {
    // pass_stream::counter_of adds s to the low word of `first` and the carry to its high word in 32-bit words;
    // prng::counter splits the 64-bit index first + s. Every point has a random node and device; first + s stays below
    // kMaxBitpacks.
    splitmix64 rng{0x636f756e746572ull};
    const auto check = [&rng](const std::uint64_t first, const std::uint32_t s) {
        const auto node = static_cast<std::uint32_t>(rng());
        const prng::stream run = prng::stream::of(kStream.seed(), static_cast<std::uint32_t>(rng()));
        const bernoulli::pass_stream pass(run, node, first);
        INFO("node=" << node << " device=" << run.device << " first=" << first << " s=" << s);
        CHECK(words_of(pass.counter_of(s)) == words_of(prng::counter({.node = node, .device = run.device, .bitpack = first + s})));
    };

    // Random first and s: the low word wraps for about half of them.
    for (int i = 0; i < 4096; ++i) {
        const std::uint64_t first = rng() % (prng::kMaxBitpacks - kWrap + 1);
        check(first, static_cast<std::uint32_t>(rng()));
    }
    // Small ranges across a wrap of the low word, which comes k = 1 .. 64 bitpacks into the pass, from the first high
    // word, from 2^25 - 1, whose carry runs through 25 bits, from the last high word a carry leaves and from random ones.
    std::vector<std::uint64_t> highs{0, kHighWords / 2 - 1, kHighWords - 2};
    for (int i = 0; i < 6; ++i) {
        highs.push_back(rng() % (kHighWords - 1));
    }
    for (const std::uint64_t high : highs) {
        for (std::uint32_t k = 1; k <= 64; ++k) {
            for (std::uint32_t s = 0; s < 2 * k; ++s) {
                check(high * kWrap + (kWrap - k), s);
            }
        }
    }
    // pass_stream::word23_of gives each lane counter words 2 and 3 of its own bitpack: of W consecutive bitpacks, as the
    // vector forms' groups, with the low word wrapping at lane k, and of W arbitrary ones, as sample_chunked's
    // set-aside bitpacks.
    const auto lanes = [&rng]<int W>(const std::uint64_t first) {
        const bernoulli::pass_stream pass(kStream, 9, first);
        prng::lanes<W> of[2];
        for (int l = 0; l < W; ++l) {
            of[0][l] = static_cast<std::uint32_t>(l);
            of[1][l] = static_cast<std::uint32_t>(rng());
        }
        for (const prng::lanes<W> &s : of) {
            prng::lanes<W> word2;
            prng::lanes<W> word3;
            pass.word23_of<W>(word2, word3, s);
            for (int l = 0; l < W; ++l) {
                INFO("W=" << W << " first=" << first << " s=" << s[l]);
                const state128 c = pass.counter_of(s[l]);
                CHECK(word2[l] == c.x[2]);
                CHECK(word3[l] == c.x[3]);
            }
        }
    };
    for (const std::uint64_t high : highs) {
        for (std::uint32_t k = 1; k <= 32; ++k) {
            lanes.template operator()<2>(high * kWrap + (kWrap - k));
            lanes.template operator()<4>(high * kWrap + (kWrap - k));
            lanes.template operator()<8>(high * kWrap + (kWrap - k));
            lanes.template operator()<16>(high * kWrap + (kWrap - k));
            lanes.template operator()<32>(high * kWrap + (kWrap - k));
        }
    }
    // The largest s, from low words 0 (no wrap), 1 (wraps to 0) and 2^32 - 1.
    for (const std::uint64_t low : {std::uint64_t{0}, std::uint64_t{1}, kWrap - 1}) {
        check(7 * kWrap + low, ~0u);
    }
    // The last 256 bitpack indices of a run, up to kMaxBitpacks - 1, from passes that start up to 2^32 - 1 bitpacks
    // before them.
    for (std::uint64_t below = 1; below <= 256; ++below) {
        for (const std::uint32_t s : {0u, 1u, 255u, 0x80000000u, ~0u}) {
            check(prng::kMaxBitpacks - below - s, s);
        }
    }

    // The accessors return what built the pass, and a pass built from the run's expanded stream, as the kernels build
    // it, has the keys and counters of one built from the stream.
    for (const std::uint32_t node : {0u, 9u, ~0u}) {
        for (const prng::stream run : {kStream, prng::stream{}, prng::stream::of(~0ull, ~0u)}) {
            for (const std::uint64_t first : {std::uint64_t{0}, kWrap - 1, kWrap, kStream.seed() % prng::kMaxBitpacks, prng::kMaxBitpacks - 1}) {
                INFO("node=" << node << " seed=" << run.seed() << " device=" << run.device << " first=" << first);
                const bernoulli::pass_stream from_stream(run, node, first);
                const bernoulli::pass_stream from_expanded(prng::expanded_stream::of(run), node, first);
                for (const bernoulli::pass_stream &pass : {from_stream, from_expanded}) {
                    CHECK(pass.node() == node);
                    CHECK(pass.device() == run.device);
                    CHECK(pass.first() == first);
                    CHECK(round_keys_of(pass.keys()) == round_keys_of(philox::key_schedule::of(run.key)));
                    CHECK(words_of(pass.counter_of(0)) == words_of(prng::counter({.node = node, .device = run.device, .bitpack = first})));
                }
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
    // trial runs to the last bit of T, which a 32-bit sampler never reaches (#17). The 64 bitpacks cross a 2^32
    // boundary of the run-wide index at bitpack 32. The counter form of `sample`, under the run's key and under its
    // schedule, the pass_stream form, and every form of the sampler at every vector width, all agree with the
    // brute-force comparison: the forms too run through generation 31 (kMaxGenerations).
    const bernoulli::pass_stream pass(kStream, 7, 11 * kWrap - 32);
    const philox::key_schedule keys = philox::key_schedule::of(kStream.key);
    for (std::uint32_t s = 0; s < 64; ++s) {
        const state128 base = pass.counter_of(s);
        const auto u = uniforms_from_words(philox_words(base, kStream.key));
        const auto j = static_cast<int>(s); // the trial tied to T
        for (const std::uint64_t threshold : {u[j], u[j] + 1, u[j] >> 40, (u[j] >> 40) + 1}) {
            if (threshold == 0) {
                continue;
            }
            const std::uint64_t expected = brute_force(u, threshold);
            REQUIRE(bernoulli::sample(base, kStream.key, threshold) == expected);
            REQUIRE(bernoulli::sample(base, keys, threshold) == expected);
            REQUIRE(bernoulli::sample(pass, s, threshold) == expected);
            std::array<std::uint64_t, 64> out{};
            bernoulli::sample_strided(out.data(), s, 1, s + 1, pass, threshold);
            REQUIRE(out[s] == expected);
            const auto vector_forms = [&]<int W>() {
                out[s] = 0;
                bernoulli::sample_lockstep<W>(out.data(), s, 1, pass, threshold);
                REQUIRE(out[s] == expected);
                out[s] = 0;
                bernoulli::sample_chunked<W>(out.data(), s, 1, pass, threshold);
                REQUIRE(out[s] == expected);
            };
            vector_forms.template operator()<2>();
            vector_forms.template operator()<4>();
            vector_forms.template operator()<8>();
            vector_forms.template operator()<16>();
            vector_forms.template operator()<32>();
        }
    }
    // p < 2^-32 has threshold high word 0 and is not 0 (#17).
    CHECK(pcfb::probability_to_threshold(1e-12) == 18446744ull);
}

TEST_CASE("the strided, lockstep and chunked forms reproduce the reference bitpacks", "[mc][prng]") {
    constexpr std::uint32_t kPacks = 203; // not a multiple of the lane widths
    // The run-wide index of the pass's bitpack 0, and where counter word 2 wraps to 0 in the pass. From 2^32 - 100 it
    // wraps at bitpack 100: inside a vector of 8, 16 or 32 lanes, at the start of a vector of 2 or 4. From 2^32 - 102
    // it wraps at bitpack 102, inside a vector of 4 to 32 lanes, and so it does from 2^57 - 102, where the carry runs
    // through the high word, 2^25 - 1 to 2^25 (word 3 0x7fffffc0 to 0x80000000). From 2^32 - 101 and 2^57 - 101 it
    // wraps at the odd bitpack 101, inside a vector of every width, 2 included. From kMaxBitpacks - kPacks, the pass
    // ends on the last bitpack index of a run.
    constexpr std::array<std::uint64_t, 7> kFirsts{
        0, kWrap - 100, kWrap - 101, kWrap - 102, (kHighWords / 2) * kWrap - 101, (kHighWords / 2) * kWrap - 102, prng::kMaxBitpacks - kPacks};
    // Guard words past the pass, and the word every bitpack starts as: a form may write only its own bitpacks.
    constexpr std::uint32_t kGuard = 32;
    constexpr std::uint64_t kSentinel = 0xdeadbeefdeadbeefull;
    for (const std::uint64_t first : kFirsts) {
        INFO("first=" << first);
        // Built from the run's expanded stream, as the sampling kernels build it.
        const bernoulli::pass_stream pass(prng::expanded_stream::of(kStream), 9, first);
        for (const double p : {0.5, 0.3125, 0.1, 1e-3, 1e-12, 1.0 - 1e-9}) {
            INFO("p=" << p);
            const auto threshold = pcfb::probability_to_threshold(p);
            std::vector<std::uint64_t> reference(kPacks);
            for (std::uint32_t s = 0; s < kPacks; ++s) {
                reference[s] = bernoulli::sample(pass, s, threshold);
            }
            for (const std::uint32_t stride : {1u, 3u, 7u, 32u, 500u}) {
                std::vector<std::uint64_t> out(kPacks + kGuard, kSentinel);
                for (std::uint32_t item = 0; item < std::min(stride, kPacks); ++item) {
                    bernoulli::sample_strided(out.data(), item, stride, kPacks, pass, threshold);
                }
                CHECK(std::vector<std::uint64_t>(out.begin(), out.begin() + kPacks) == reference);
                CHECK(std::all_of(out.begin() + kPacks, out.end(), [](const std::uint64_t w) { return w == kSentinel; }));
            }
            // The vector forms over chunks that are and are not multiples of W, and the whole pass in one chunk: a
            // chunk's last group is partial, sample_chunked's set-aside bitpacks come from several groups, and the wrap
            // of counter word 2 falls inside a chunk. W = 2 and 32 are the narrowest and widest vectors lane_mask takes.
            // Each call writes into a buffer of its own, so that a store outside [at, at + count), which the next
            // call would overwrite or which would land past the pass, is seen: it counts as stray.
            struct sampled {
                std::vector<std::uint64_t> bits;
                std::size_t stray = 0;
            };
            const auto chunks = [&]<int W>(const bernoulli::form form, const std::uint32_t chunk) {
                sampled result{.bits = std::vector<std::uint64_t>(kPacks, kSentinel)};
                std::vector<std::uint64_t> buffer(kPacks + kGuard);
                for (std::uint32_t at = 0; at < kPacks; at += chunk) {
                    const auto count = std::min<std::uint32_t>(chunk, kPacks - at);
                    std::fill(buffer.begin(), buffer.end(), kSentinel);
                    if (form == bernoulli::form::lockstep) {
                        bernoulli::sample_lockstep<W>(buffer.data(), at, count, pass, threshold);
                    } else {
                        bernoulli::sample_chunked<W>(buffer.data(), at, count, pass, threshold);
                    }
                    for (std::uint32_t i = 0; i < buffer.size(); ++i) {
                        if (i >= at && i < at + count) {
                            result.bits[i] = buffer[i];
                        } else if (buffer[i] != kSentinel) {
                            ++result.stray;
                        }
                    }
                }
                return result;
            };
            const auto check = [&]<int W>(const bernoulli::form form, const std::uint32_t chunk) {
                INFO("W=" << W);
                const sampled got = chunks.template operator()<W>(form, chunk);
                CHECK(got.bits == reference);
                CHECK(got.stray == 0);
            };
            for (const bernoulli::form form : {bernoulli::form::lockstep, bernoulli::form::chunked}) {
                for (const std::uint32_t chunk : {1u, 16u, 37u, 64u, kPacks}) {
                    INFO((form == bernoulli::form::lockstep ? "lockstep" : "chunked") << " chunk=" << chunk);
                    check.template operator()<2>(form, chunk);
                    check.template operator()<4>(form, chunk);
                    check.template operator()<8>(form, chunk);
                    check.template operator()<16>(form, chunk);
                    check.template operator()<32>(form, chunk);
                }
            }
        }
    }
}

TEST_CASE("trials are Bernoulli(T / 2^64), uniform over bit positions and independent of neighbours", "[mc][prng][statistics]") {
    constexpr std::uint32_t kPacks = 1u << 16; // 4 Mi trials per probability; also the bitpacks per pass
    // Node 3's first pass in the default stream (seed 372, device 0), and neighbours that each differ from it in one
    // Philox input.
    const prng::stream defaults{};
    const bernoulli::pass_stream base(defaults, 3, 0);
    const std::array<std::pair<std::string_view, bernoulli::pass_stream>, 5> neighbours{{
        {"adjacent node", bernoulli::pass_stream(defaults, 4, 0)},                               // counter word 0
        {"adjacent device", bernoulli::pass_stream(prng::stream::of(defaults.seed(), 1), 3, 0)}, // counter word 1
        {"adjacent pass", bernoulli::pass_stream(defaults, 3, kPacks)},                          // counter word 2
        {"next high word", bernoulli::pass_stream(defaults, 3, kWrap)},                          // counter word 3
        {"next seed", bernoulli::pass_stream(prng::stream::of(defaults.seed() + 1), 3, 0)},      // the key
    }};
    for (const double p : {0.5, 0.3125, 0.1, 0.01, 1e-3, 0.999}) {
        INFO("p=" << p);
        const auto threshold = pcfb::probability_to_threshold(p);
        const double pe = std::ldexp(static_cast<double>(threshold), -64);
        const auto bitpacks = [threshold](const bernoulli::pass_stream &pass) {
            std::vector<std::uint64_t> out(kPacks);
            for (std::uint32_t s = 0; s < kPacks; ++s) {
                out[s] = bernoulli::sample(pass, s, threshold);
            }
            return out;
        };
        auto z_of = [](const double hits, const double n, const double q) { return (hits - n * q) / std::sqrt(n * q * (1.0 - q)); };
        const auto a = bitpacks(base);

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

        // Pairs in the base pass: adjacent bits of a bitpack and the same bit of adjacent bitpacks. Joint ones must
        // match p^2.
        double in_pack = 0, across_samples = 0;
        for (std::uint32_t s = 0; s < kPacks; ++s) {
            in_pack += std::popcount(a[s] & (a[s] >> 1) & 0x7FFFFFFFFFFFFFFFull);
            if (s + 1 < kPacks) {
                across_samples += std::popcount(a[s] & a[s + 1]);
            }
        }
        CHECK(std::fabs(z_of(in_pack, 63.0 * kPacks, pe * pe)) < kZLimit);
        CHECK(std::fabs(z_of(across_samples, 64.0 * (kPacks - 1), pe * pe)) < kZLimit);

        // Each neighbour is Bernoulli(p) too, and its joint ones with the base pass at the same bit of the same
        // bitpack match p^2.
        for (const auto &[name, pass] : neighbours) {
            INFO(name);
            const auto b = bitpacks(pass);
            double ones_b = 0, joint = 0;
            for (std::uint32_t s = 0; s < kPacks; ++s) {
                ones_b += std::popcount(b[s]);
                joint += std::popcount(a[s] & b[s]);
            }
            CHECK(std::fabs(z_of(ones_b, 64.0 * kPacks, pe)) < kZLimit);
            CHECK(std::fabs(z_of(joint, 64.0 * kPacks, pe * pe)) < kZLimit);
        }
    }
}

// The layout prng/stream.h documents: 32 + 26 bits of bitpack index, and 6 bits of generation in counter word 3, which
// hold every generation a bitpack uses.
static_assert(prng::kGenerationBits == 6 && prng::kMaxBitpacks == std::uint64_t{1} << 58, "the counter layout of prng/stream.h");
static_assert(bernoulli::kMaxGenerations <= (1u << prng::kGenerationBits), "generations must fit the counter's 6-bit field");
// The default stream and a stream's round keys are constant expressions.
static_assert(prng::stream::kDefaultSeed == 372 && prng::stream{}.seed() == 372 && prng::stream{}.device == 0, "the default stream: seed 372 on device 0");
static_assert(prng::expanded_stream::of(kStream).keys == philox::key_schedule::of(kStream.key), "a stream expands to its key's round keys");
