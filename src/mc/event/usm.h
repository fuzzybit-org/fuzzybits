/**
 * @file usm.h
 * @brief usm_array: one USM allocation, owned and freed by one object
 */

#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace fuzzybits::mc::event {

/// The device's global-memory cache line when it reports a power of two, else alignof(T).
template <class T> [[nodiscard]] std::size_t cache_line_alignment(const sycl::device &device) {
    const auto line = static_cast<std::size_t>(device.get_info<sycl::info::device::global_mem_cache_line_size>());
    return std::has_single_bit(line) ? std::max(line, alignof(T)) : alignof(T);
}

/// `size()` objects of `T` in one USM allocation of kind `kind()`, freed through the allocation's context when the
/// owner is destroyed. Move-only; a moved-from array is empty.
///
/// Device and host memory are aligned to the device's cache line. Shared memory has the backend's allocation
/// alignment, which AdaptiveCpp does not let a caller choose (aligned_alloc_shared ignores its alignment); every kind
/// is checked to meet alignof(T).
///
/// USM memory is not constructed or destroyed, so `T` must be trivially copyable (the bytes on the device are the
/// object) and trivially destructible. The owner must outlive every kernel that uses the memory: RAII does not see
/// the queue, so whoever owns both waits on the queue first (layer_manager does, in its destructor).
template <class T> class usm_array {
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>, "USM holds trivially copyable, trivially destructible objects");

  public:
    usm_array() noexcept = default;
    usm_array(const usm_array &) = delete;
    usm_array &operator=(const usm_array &) = delete;
    usm_array(usm_array &&other) noexcept { swap(other); }
    usm_array &operator=(usm_array &&other) noexcept {
        usm_array(std::move(other)).swap(*this);
        return *this;
    }
    ~usm_array() {
        if (data_ != nullptr) {
            sycl::free(data_, *context_);
        }
    }

    /// `n` uninitialised objects of kind `kind`. `n == 0` allocates nothing: a zero-size shared allocation fails on
    /// HIP (#277). SYCL returns null when an allocation fails; this throws std::bad_alloc instead, and
    /// std::bad_array_new_length (a std::bad_alloc) when n·sizeof(T) bytes overflow, where SYCL would allocate the
    /// wrapped size.
    [[nodiscard]] static usm_array allocate(const sycl::queue &queue, const std::size_t n, const sycl::usm::alloc kind) {
        usm_array a;
        a.kind_ = kind;
        if (n == 0) {
            return a;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length{};
        }
        a.context_.emplace(queue.get_context());
        const std::size_t alignment = kind == sycl::usm::alloc::shared ? alignof(T) : cache_line_alignment<T>(queue.get_device());
        a.data_ = sycl::aligned_alloc<T>(alignment, n, queue, kind);
        if (a.data_ == nullptr || reinterpret_cast<std::uintptr_t>(a.data_) % alignof(T) != 0) {
            throw std::bad_alloc{};
        }
        a.size_ = n;
        return a;
    }

    /// `values`, in order, copied into a new allocation of `kind`, which the host must be able to write (shared or
    /// host memory).
    template <std::ranges::sized_range R>
        requires std::constructible_from<T, std::ranges::range_reference_t<R>>
    [[nodiscard]] static usm_array copy_of(const sycl::queue &queue, R &&values, const sycl::usm::alloc kind) {
        if (kind == sycl::usm::alloc::device) {
            throw std::invalid_argument("usm_array::copy_of: the host cannot write device memory");
        }
        auto a = allocate(queue, static_cast<std::size_t>(std::ranges::size(values)), kind);
        std::ranges::uninitialized_copy(values, a.host());
        return a;
    }

    [[nodiscard]] T *data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] sycl::usm::alloc kind() const noexcept { return kind_; }

    /// The objects, on the host. Refused for device memory, which the host cannot dereference.
    [[nodiscard]] std::span<T> host() const {
        if (kind_ == sycl::usm::alloc::device && data_ != nullptr) {
            throw std::logic_error("usm_array: device memory has no host view");
        }
        return {data_, size_};
    }

    void swap(usm_array &other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(kind_, other.kind_);
        std::swap(context_, other.context_);
    }

  private:
    T *data_ = nullptr;
    std::size_t size_ = 0;
    sycl::usm::alloc kind_ = sycl::usm::alloc::unknown;
    /// Engaged iff data_ is not null. Optional, so that an empty array constructs no context.
    std::optional<sycl::context> context_;
};

} // namespace fuzzybits::mc::event
