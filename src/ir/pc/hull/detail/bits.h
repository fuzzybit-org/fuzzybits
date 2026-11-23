#pragma once

#include <array>
#include <cstdint>

namespace fuzzybits::ir::pc::hull::detail {

inline constexpr std::uint64_t all_ones = ~std::uint64_t{0};

// Per-lane population count of a stream of 64-lane words, stored bit-sliced
// (bits[b] holds bit b of every lane's count).
class sliced_counter final {
  public:
    void add(std::uint64_t word) noexcept {
        for (std::uint32_t bit = 0; word != 0; ++bit) {
            if (bit == width_) {
                bits_[width_++] = 0;
            }
            const auto carry = bits_[bit] & word;
            bits_[bit] ^= word;
            word = carry;
        }
    }

    // Lanes whose count is at least `threshold`.
    [[nodiscard]] auto at_least(const std::uint64_t threshold) const noexcept -> std::uint64_t {
        if (threshold == 0) {
            return all_ones;
        }
        if ((threshold >> width_) != 0) {
            return 0;
        }
        std::uint64_t greater = 0;
        std::uint64_t equal = all_ones;
        for (auto bit = width_; bit-- > 0;) {
            if (((threshold >> bit) & 1U) != 0) {
                equal &= bits_[bit];
            } else {
                greater |= equal & bits_[bit];
                equal &= ~bits_[bit];
            }
        }
        return greater | equal;
    }

  private:
    // Only bits_[0, width_) are ever read; each is zeroed when width_ grows.
    std::array<std::uint64_t, 33> bits_;
    std::uint32_t width_ = 0;
};

// In-place transpose of a 64 x 64 bit matrix: bit c of rows[r] becomes bit
// r of rows[c]. Recursive block swap (Hacker's Delight, §7-3).
inline void transpose64(std::array<std::uint64_t, 64> &rows) noexcept {
    std::uint64_t mask = 0x00000000FFFFFFFFULL;
    for (std::uint32_t width = 32; width != 0; width >>= 1, mask ^= mask << width) {
        for (std::uint32_t row = 0; row < 64; row = ((row | width) + 1) & ~width) {
            const auto swap = ((rows[row] >> width) ^ rows[row | width]) & mask;
            rows[row] ^= swap << width;
            rows[row | width] ^= swap;
        }
    }
}

} // namespace fuzzybits::ir::pc::hull::detail
