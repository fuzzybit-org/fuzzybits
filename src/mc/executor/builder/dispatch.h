/**
 * @file dispatch.h
 * @brief From a value known at run time to the template instantiated for it
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details A kernel's form is a template argument, so that each form compiles to a kernel of its own, but it is chosen
 * at run time: by the device, by the op code, by the number of inputs. select() makes that choice in one place: it
 * calls a generic lambda with the compile-time value equal to the run-time one, among the values the kernels are
 * instantiated for.
 */

#pragma once

#include <cstddef>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace fuzzybits::mc::executor::builder {

namespace detail {
template <auto first, auto...> inline constexpr auto first_of = first;
} // namespace detail

/// Thrown when a model needs a kernel that does not exist yet (kernel::has_kernel): before any kernel runs.
class not_implemented : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

/// `choose.template operator()<v>()` for the value v among `values` equal to `value`. Every choice returns the same type.
/// Throws std::logic_error, naming `what`, when no value is equal: no kernel is instantiated for `value`.
template <auto... values, typename F> [[nodiscard]] auto select(const auto value, F &&choose, const char *what) {
    static_assert(sizeof...(values) > 0, "select chooses among at least one value");
    using result = decltype(choose.template operator()<detail::first_of<values...>>());
    std::optional<result> chosen;
    (void)((value == values && (chosen.emplace(choose.template operator()<values>()), true)) || ...);
    if (!chosen) {
        throw std::logic_error(std::string("kernel builder: no kernel for this ") + what);
    }
    return std::move(*chosen);
}

/// select() among the elements of the constant array `values`.
template <const auto &values, typename F> [[nodiscard]] auto select_in(const auto value, F &&choose, const char *what) {
    return [&]<std::size_t... i>(std::index_sequence<i...>) {
        return select<values[i]...>(value, std::forward<F>(choose), what);
    }(std::make_index_sequence<std::size(values)>{});
}

} // namespace fuzzybits::mc::executor::builder
