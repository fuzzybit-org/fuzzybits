#pragma once

#include "api/result.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fuzzybits::compile::ftp::json {

    // A JSON document. Objects keep their key order; arrays and objects both
    // store their members in `items`, an object's keys in `keys` (same index).
    struct value final {
        enum class kind_type : std::uint8_t { null, boolean, integer, number, string, array, object };

        kind_type kind = kind_type::null;
        bool boolean = false;
        std::int64_t integer = 0;
        double number = 0.0;
        std::string string;
        std::vector<value> items;
        std::vector<std::string> keys;

        [[nodiscard]] static auto make_null() -> value {
            return {};
        }
        [[nodiscard]] static auto make_bool(const bool v) -> value {
            value out;
            out.kind = kind_type::boolean;
            out.boolean = v;
            return out;
        }
        [[nodiscard]] static auto make_int(const std::int64_t v) -> value {
            value out;
            out.kind = kind_type::integer;
            out.integer = v;
            return out;
        }
        [[nodiscard]] static auto make_number(const double v) -> value {
            value out;
            out.kind = kind_type::number;
            out.number = v;
            return out;
        }
        [[nodiscard]] static auto make_string(std::string v) -> value {
            value out;
            out.kind = kind_type::string;
            out.string = std::move(v);
            return out;
        }
        [[nodiscard]] static auto make_array() -> value {
            value out;
            out.kind = kind_type::array;
            return out;
        }
        [[nodiscard]] static auto make_object() -> value {
            value out;
            out.kind = kind_type::object;
            return out;
        }

        value &push(value member) {
            items.push_back(std::move(member));
            return *this;
        }
        // Appends a member; the caller keeps keys unique.
        value &set(std::string key, value member) {
            keys.push_back(std::move(key));
            items.push_back(std::move(member));
            return *this;
        }

        [[nodiscard]] auto is(const kind_type expected) const noexcept -> bool {
            return kind == expected;
        }
        [[nodiscard]] auto is_object() const noexcept -> bool {
            return kind == kind_type::object;
        }
        [[nodiscard]] auto is_array() const noexcept -> bool {
            return kind == kind_type::array;
        }
        [[nodiscard]] auto is_string() const noexcept -> bool {
            return kind == kind_type::string;
        }
        [[nodiscard]] auto is_integer() const noexcept -> bool {
            return kind == kind_type::integer;
        }
        [[nodiscard]] auto find(const std::string_view key) const noexcept -> value const * {
            if (kind != kind_type::object) {
                return nullptr;
            }
            for (std::size_t index = 0; index < keys.size(); ++index) {
                if (keys[index] == key) {
                    return &items[index];
                }
            }
            return nullptr;
        }
    };

    // Strict RFC 8259 parser (UTF-8 passes through; \uXXXX escapes, including
    // surrogate pairs, are decoded). Diagnostic id compile.ftp.invalid-json.
    [[nodiscard]] auto parse(std::string_view text) -> api::result<value>;

    // Compact when `indent` is 0, otherwise pretty-printed with `indent` spaces.
    [[nodiscard]] auto dump(value const &document, unsigned indent = 0) -> std::string;

} // namespace fuzzybits::compile::ftp::json
