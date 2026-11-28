/**
 * @file basic_event.h
 * @brief Basic events as their kernel reads them: a node and the Bernoulli law it samples
 */

#pragma once

#include "mc/event/node.h"

#include "pcfb/threshold.h"

#include <concepts>
#include <cstdint>
#include <optional>

namespace fuzzybits::mc::event {

enum class bernoulli_kind : std::uint8_t { never, sampled, always };

/// The law a basic event samples. A trial is true iff its 64-bit uniform is below threshold(): exactly
/// Bernoulli(threshold / 2^64) (prng/bernoulli/sample.h). p = 1 has no 64-bit threshold, so it is a kind of its own.
///
/// Invariant: `sampled` ⇔ threshold > 0 and not `always`, which is the samplers' precondition ("Requires T > 0").
/// There is no conversion from an arithmetic value: a probability enters only through from_probability(double), and
/// a threshold only through from_threshold(std::uint64_t).
class bernoulli_law {
  public:
    [[nodiscard]] static constexpr bernoulli_law never() noexcept { return {0, false}; }
    [[nodiscard]] static constexpr bernoulli_law always() noexcept { return {pcfb::kThresholdAlwaysTrue, true}; }
    [[nodiscard]] static constexpr bernoulli_law from_threshold(const std::uint64_t t) noexcept { return {t, false}; }

    /// Exact floor(p · 2^64) (pcfb::probability_to_threshold); nullopt unless 0 <= p <= 1, NaN included. p below
    /// 2^-64 is `never`.
    [[nodiscard]] static constexpr std::optional<bernoulli_law> from_probability(const double p) noexcept {
        if (!(p >= 0.0 && p <= 1.0)) {
            return std::nullopt;
        }
        return p == 1.0 ? always() : from_threshold(pcfb::probability_to_threshold(p));
    }

    // Refused at compile time: a float or integer probability would be rounded or truncated before the exact
    // mapping, and a threshold that is not a 64-bit unsigned integer would be converted without a diagnostic. Every
    // 64-bit unsigned type is one (std::uint64_t is unsigned long on Linux, unsigned long long elsewhere).
    template <class P>
        requires(!std::same_as<P, double>)
    static std::optional<bernoulli_law> from_probability(P) = delete;
    template <class U>
        requires(!(std::unsigned_integral<U> && sizeof(U) == sizeof(std::uint64_t)))
    static bernoulli_law from_threshold(U) = delete;

    [[nodiscard]] constexpr bernoulli_kind kind() const noexcept {
        return always_ ? bernoulli_kind::always : threshold_ == 0 ? bernoulli_kind::never : bernoulli_kind::sampled;
    }
    [[nodiscard]] constexpr std::uint64_t threshold() const noexcept { return threshold_; }
    /// threshold / 2^64, or 1 for `always`: exact for thresholds of at most 53 significant bits.
    [[nodiscard]] constexpr double probability() const noexcept { return always_ ? 1.0 : pcfb::threshold_to_probability(threshold_); }

    friend constexpr bool operator==(const bernoulli_law &, const bernoulli_law &) = default;

  private:
    constexpr bernoulli_law(const std::uint64_t t, const bool is_always) noexcept : threshold_(t), always_(is_always) {}

    std::uint64_t threshold_;
    bool always_;
};

/// A basic event and its law, of any kind. Its kind decides how its output is made, once, when the kernels are built:
/// a sampled law becomes a sampled_event, which the sampling kernel reads; a constant law becomes a constant_event,
/// whose bitpacks are written when its arena slot is allocated, and which no kernel reads.
template <node_types_like T> struct basic_event { // NOLINT(cppcoreguidelines-pro-type-member-init): law has no default by design
    /// The node's PDAG index: also the node word of its Philox counter (prng/stream.h).
    typename T::index_type id_in_pdag;
    /// No default: a record cannot exist without its law.
    bernoulli_law law;
};

/// The law of a basic event whose trials are random: bernoulli_kind::sampled, so 0 < T and not `always`, which is the
/// samplers' precondition. There is no other way to make one than from such a law.
class sampled_law {
  public:
    /// `law` if its trials are random; nullopt for `never` and `always`.
    [[nodiscard]] static constexpr std::optional<sampled_law> of(const bernoulli_law &law) noexcept {
        if (law.kind() != bernoulli_kind::sampled) {
            return std::nullopt;
        }
        return sampled_law(law.threshold());
    }

    /// T, above 0.
    [[nodiscard]] constexpr std::uint64_t threshold() const noexcept { return threshold_; }
    [[nodiscard]] constexpr bernoulli_law law() const noexcept { return bernoulli_law::from_threshold(threshold_); }

    friend constexpr bool operator==(const sampled_law &, const sampled_law &) = default;

  private:
    explicit constexpr sampled_law(const std::uint64_t t) noexcept : threshold_(t) {}

    std::uint64_t threshold_;
};

/// A basic event whose trials are random, as the sampling kernel reads it: 16 bytes with 32-bit ids. The kernel has
/// no case for a constant law because this record cannot hold one.
template <node_types_like T> struct sampled_event { // NOLINT(cppcoreguidelines-pro-type-member-init): law has no default by design
    /// The node's PDAG index: also the node word of its Philox counter (prng/stream.h).
    typename T::index_type id_in_pdag;
    sampled_law law;

    /// `event` if its trials are random; nullopt if its law is a constant.
    [[nodiscard]] static constexpr std::optional<sampled_event> of(const basic_event<T> &event) noexcept {
        const std::optional<sampled_law> law = sampled_law::of(event.law);
        if (!law) {
            return std::nullopt;
        }
        return sampled_event{.id_in_pdag = event.id_in_pdag, .law = *law};
    }
};

/// A basic event whose law is a constant (`never`, `always`). Its bitpacks are the same in every pass, so they are
/// written once, when its arena slot is allocated (kernel::sampler::write_constant_outputs); no kernel reads it.
template <node_types_like T> struct constant_event {
    /// The node's PDAG index.
    typename T::index_type id_in_pdag;
    /// Every trial's value: false for `never`, true for `always`.
    bool value;

    /// `event` if its law is a constant; nullopt if its trials are random.
    [[nodiscard]] static constexpr std::optional<constant_event> of(const basic_event<T> &event) noexcept {
        switch (event.law.kind()) {
        case bernoulli_kind::never:
            return constant_event{.id_in_pdag = event.id_in_pdag, .value = false};
        case bernoulli_kind::always:
            return constant_event{.id_in_pdag = event.id_in_pdag, .value = true};
        case bernoulli_kind::sampled:
            break;
        }
        return std::nullopt;
    }
};

template <node_types_like T> using basic_event_block = node_block<basic_event<T>, T>;
template <node_types_like T> using basic_event_block_view = node_block_view<basic_event<T>, T>;
template <node_types_like T> using sampled_event_block = node_block<sampled_event<T>, T>;
template <node_types_like T> using sampled_event_block_view = node_block_view<sampled_event<T>, T>;
template <node_types_like T> using constant_event_block = node_block<constant_event<T>, T>;

} // namespace fuzzybits::mc::event
