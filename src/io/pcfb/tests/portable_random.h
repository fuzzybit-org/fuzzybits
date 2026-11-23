#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <utility>

// The sequence of std::mt19937_64 is fixed by the standard; the standard distributions and
// std::ranges::shuffle are not, so libc++ and libstdc++ would draw different circuits from one seed.
namespace pcfb_test_random {

// Uniform in [low, high] by modulo with rejection.
[[nodiscard]] inline auto uniform(std::mt19937_64 &rng, const std::uint64_t low, const std::uint64_t high)
    -> std::uint64_t {
    const std::uint64_t span = high - low + 1;
    if (span == 0) {
        return rng();
    }
    const std::uint64_t limit =
        std::numeric_limits<std::uint64_t>::max() - std::numeric_limits<std::uint64_t>::max() % span;
    std::uint64_t draw = rng();
    while (draw >= limit) {
        draw = rng();
    }
    return low + draw % span;
}

// Uniform in [0, 1).
[[nodiscard]] inline auto unit(std::mt19937_64 &rng) -> double { return static_cast<double>(rng() >> 11U) * 0x1.0p-53; }

[[nodiscard]] inline auto real(std::mt19937_64 &rng, const double low, const double high) -> double {
    return low + (high - low) * unit(rng);
}

[[nodiscard]] inline auto chance(std::mt19937_64 &rng, const double p) -> bool { return unit(rng) < p; }

// Fisher-Yates.
template <typename Element> inline void shuffle(std::span<Element> values, std::mt19937_64 &rng) {
    for (std::size_t at = values.size(); at > 1; --at) {
        std::swap(values[at - 1], values[static_cast<std::size_t>(uniform(rng, 0, at - 1))]);
    }
}

} // namespace pcfb_test_random
