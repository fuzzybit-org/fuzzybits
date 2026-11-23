#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace fuzzybits::ir::pra {

// One 32-bit row index per table, typed by a tag so ids of different tables
// cannot be mixed. Ids are local to one sealed model.
template <typename tag_type> class id final {
  public:
    using tag = tag_type;
    using value_type = std::uint32_t;

    static constexpr value_type invalid_value = std::numeric_limits<value_type>::max();

    constexpr id() = default;

    explicit constexpr id(const value_type value) noexcept : value_(value) {}

    [[nodiscard]] static constexpr auto invalid() noexcept -> id { return id(); }

    [[nodiscard]] constexpr auto valid() const noexcept -> bool { return value_ != invalid_value; }

    [[nodiscard]] constexpr auto raw() const noexcept -> value_type { return value_; }

    friend auto operator<=>(id const &, id const &) = default;

  private:
    value_type value_ = invalid_value;
};

namespace id_tags {
struct element;
struct source_file;
struct expression;
struct basic_event;
struct house_event;
struct initiating_event;
struct gate;
struct fault_tree;
struct domain_target;
} // namespace id_tags

using element_id = id<id_tags::element>;
using source_file_id = id<id_tags::source_file>;
using expression_id = id<id_tags::expression>;
using basic_event_id = id<id_tags::basic_event>;
using house_event_id = id<id_tags::house_event>;
using initiating_event_id = id<id_tags::initiating_event>;
using gate_id = id<id_tags::gate>;
using fault_tree_id = id<id_tags::fault_tree>;
using domain_target_id = id<id_tags::domain_target>;

// Entity kinds that carry element metadata. Stable names are unique within a
// kind, not across kinds.
enum class element_kind : std::uint8_t {
    basic_event,
    house_event,
    initiating_event,
    gate,
    fault_tree,
    domain_target,
};

inline constexpr std::size_t element_kind_count = 6;

[[nodiscard]] constexpr auto to_string(const element_kind kind) noexcept -> std::string_view {
    switch (kind) {
    case element_kind::basic_event:
        return "basic event";
    case element_kind::house_event:
        return "house event";
    case element_kind::initiating_event:
        return "initiating event";
    case element_kind::gate:
        return "gate";
    case element_kind::fault_tree:
        return "fault tree";
    case element_kind::domain_target:
        return "target";
    }
    return "element";
}

template <typename tag_type> struct element_kind_of {};
template <> struct element_kind_of<id_tags::basic_event> {
    static constexpr element_kind value = element_kind::basic_event;
};
template <> struct element_kind_of<id_tags::house_event> {
    static constexpr element_kind value = element_kind::house_event;
};
template <> struct element_kind_of<id_tags::initiating_event> {
    static constexpr element_kind value = element_kind::initiating_event;
};
template <> struct element_kind_of<id_tags::gate> {
    static constexpr element_kind value = element_kind::gate;
};
template <> struct element_kind_of<id_tags::fault_tree> {
    static constexpr element_kind value = element_kind::fault_tree;
};
template <> struct element_kind_of<id_tags::domain_target> {
    static constexpr element_kind value = element_kind::domain_target;
};

template <typename tag_type>
concept element_tag = requires { element_kind_of<tag_type>::value; };

// A typed reference to a row of any element table.
class domain_ref final {
  public:
    constexpr domain_ref() = default;

    constexpr domain_ref(const element_kind kind, const std::uint32_t index) noexcept : index_(index), kind_(kind) {}

    template <element_tag tag_type>
    constexpr domain_ref(const id<tag_type> value) noexcept // NOLINT(google-explicit-constructor)
        : index_(value.raw()), kind_(element_kind_of<tag_type>::value) {}

    [[nodiscard]] constexpr auto kind() const noexcept -> element_kind { return kind_; }

    [[nodiscard]] constexpr auto raw() const noexcept -> std::uint32_t { return index_; }

    [[nodiscard]] constexpr auto valid() const noexcept -> bool {
        return index_ != std::numeric_limits<std::uint32_t>::max();
    }

    // Returns an invalid id when the reference is of another kind.
    template <typename id_type> [[nodiscard]] constexpr auto as() const noexcept -> id_type {
        return kind_ == element_kind_of<typename id_type::tag>::value ? id_type(index_) : id_type::invalid();
    }

    friend auto operator<=>(domain_ref const &, domain_ref const &) = default;

  private:
    std::uint32_t index_ = std::numeric_limits<std::uint32_t>::max();
    element_kind kind_ = element_kind::basic_event;
};

// A slice of the model's text pool.
struct text_ref final {
    std::uint32_t offset = 0;
    std::uint32_t size = 0;

    [[nodiscard]] constexpr auto empty() const noexcept -> bool { return size == 0; }
};

// A contiguous slice of one model column; `row_type` names the column's row type.
template <typename row_type> struct table_range final {
    std::uint32_t offset = 0;
    std::uint32_t count = 0;
};

} // namespace fuzzybits::ir::pra
