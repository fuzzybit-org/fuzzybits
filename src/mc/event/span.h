/**
 * @file span.h
 * @brief device_span: the span a kernel holds by value, a pointer and a count no wider than its indices
 */

#pragma once

#include <concepts>
#include <span>
#include <type_traits>

namespace fuzzybits::mc::event {

/// A contiguous run of `T`. Two words and trivially copyable, so a kernel argument that holds one costs a pointer
/// and a count. The count has the type `E` the kernels index in (at most 32 bits in the engine), so an index is
/// neither widened nor truncated on the way in.
///
/// There are no bounds checks: kernels index spans on the hot path, and whatever hands a span out (an arena, a
/// block) checked its shape when it was built.
template <class T, std::unsigned_integral E> class device_span {
  public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = E;

    constexpr device_span() noexcept = default;
    constexpr device_span(T *data, const E size) noexcept : data_(data), size_(size) {}

    /// The same elements, read-only.
    [[nodiscard]] constexpr device_span<const T, E> as_const() const noexcept { return {data_, size_}; }

    [[nodiscard]] constexpr T *data() const noexcept { return data_; }
    [[nodiscard]] constexpr E size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] constexpr T &operator[](const E i) const noexcept { return data_[i]; }
    [[nodiscard]] constexpr T *begin() const noexcept { return data_; }
    [[nodiscard]] constexpr T *end() const noexcept { return data_ + size_; }

    /// Elements [first, first + count).
    [[nodiscard]] constexpr device_span subspan(const E first, const E count) const noexcept { return {data_ + first, count}; }

    /// The same elements as a std::span, for host algorithms over memory the host can read: not over device USM, such
    /// as an arena's outputs, which only kernels and queue.memcpy reach.
    [[nodiscard]] constexpr std::span<T> as_std() const noexcept { return {data_, size_}; }

  private:
    T *data_ = nullptr;
    E size_ = 0;
};

} // namespace fuzzybits::mc::event
