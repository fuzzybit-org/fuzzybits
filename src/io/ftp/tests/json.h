#pragma once

// A minimal JSON reader for the conformance fixtures and model cards: objects,
// arrays, strings (escapes other than \uXXXX), numbers, booleans and null.

#include <charconv>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace ftp_test {

    struct json;
    using json_object = std::map<std::string, json, std::less<>>;
    using json_array = std::vector<json>;

    struct json final {
        std::variant<std::nullptr_t, bool, double, std::string, std::shared_ptr<json_array>, std::shared_ptr<json_object>> value = nullptr;

        [[nodiscard]] auto is_null() const -> bool {
            return std::holds_alternative<std::nullptr_t>(value);
        }
        [[nodiscard]] auto number() const -> double {
            return std::get<double>(value);
        }
        [[nodiscard]] auto string() const -> std::string const & {
            return std::get<std::string>(value);
        }
        [[nodiscard]] auto array() const -> json_array const & {
            return *std::get<std::shared_ptr<json_array>>(value);
        }
        [[nodiscard]] auto object() const -> json_object const & {
            return *std::get<std::shared_ptr<json_object>>(value);
        }
        [[nodiscard]] auto is_object() const -> bool {
            return std::holds_alternative<std::shared_ptr<json_object>>(value);
        }
        [[nodiscard]] auto find(std::string_view key) const -> json const * {
            if (!is_object()) {
                return nullptr;
            }
            const auto found = object().find(key);
            return found == object().end() ? nullptr : &found->second;
        }
        [[nodiscard]] auto operator[](std::string_view key) const -> json const & {
            return object().find(key)->second;
        }
    };

    class json_parser final {
    public:
        explicit json_parser(std::string_view text) : text_(text) {}

        [[nodiscard]] auto parse() -> std::optional<json> {
            auto out = value();
            skip();
            if (!out || cursor_ != text_.size()) {
                return std::nullopt;
            }
            return out;
        }

    private:
        void skip() {
            while (cursor_ < text_.size() && (text_[cursor_] == ' ' || text_[cursor_] == '\n' || text_[cursor_] == '\r' || text_[cursor_] == '\t')) {
                ++cursor_;
            }
        }

        [[nodiscard]] auto consume(const char expected) -> bool {
            skip();
            if (cursor_ < text_.size() && text_[cursor_] == expected) {
                ++cursor_;
                return true;
            }
            return false;
        }

        [[nodiscard]] auto literal(const std::string_view word) -> bool {
            if (text_.substr(cursor_, word.size()) == word) {
                cursor_ += word.size();
                return true;
            }
            return false;
        }

        [[nodiscard]] auto string() -> std::optional<std::string> {
            if (!consume('"')) {
                return std::nullopt;
            }
            std::string out;
            while (cursor_ < text_.size() && text_[cursor_] != '"') {
                if (text_[cursor_] == '\\' && cursor_ + 1 < text_.size()) {
                    ++cursor_;
                    switch (text_[cursor_]) {
                    case 'n':
                        out += '\n';
                        break;
                    case 't':
                        out += '\t';
                        break;
                    default:
                        out += text_[cursor_];
                        break;
                    }
                } else {
                    out += text_[cursor_];
                }
                ++cursor_;
            }
            if (cursor_ >= text_.size()) {
                return std::nullopt;
            }
            ++cursor_;
            return out;
        }

        [[nodiscard]] auto value() -> std::optional<json> {
            skip();
            if (cursor_ >= text_.size()) {
                return std::nullopt;
            }
            const auto head = text_[cursor_];
            if (head == '{') {
                ++cursor_;
                auto object = std::make_shared<json_object>();
                if (consume('}')) {
                    return json{object};
                }
                do {
                    auto key = string();
                    if (!key || !consume(':')) {
                        return std::nullopt;
                    }
                    auto item = value();
                    if (!item) {
                        return std::nullopt;
                    }
                    object->insert_or_assign(std::move(*key), std::move(*item));
                } while (consume(','));
                return consume('}') ? std::optional<json>(json{object}) : std::nullopt;
            }
            if (head == '[') {
                ++cursor_;
                auto array = std::make_shared<json_array>();
                if (consume(']')) {
                    return json{array};
                }
                do {
                    auto item = value();
                    if (!item) {
                        return std::nullopt;
                    }
                    array->push_back(std::move(*item));
                } while (consume(','));
                return consume(']') ? std::optional<json>(json{array}) : std::nullopt;
            }
            if (head == '"') {
                auto text = string();
                return text ? std::optional<json>(json{std::move(*text)}) : std::nullopt;
            }
            if (literal("true")) {
                return json{true};
            }
            if (literal("false")) {
                return json{false};
            }
            if (literal("null")) {
                return json{nullptr};
            }
            double number = 0.0;
            const auto begin = text_.data() + cursor_;
            const auto [end, error] = std::from_chars(begin, text_.data() + text_.size(), number);
            if (error != std::errc{} || end == begin) {
                return std::nullopt;
            }
            cursor_ += static_cast<std::size_t>(end - begin);
            return json{number};
        }

        std::string_view text_;
        std::size_t cursor_ = 0;
    };

    [[nodiscard]] inline auto parse_json(std::string_view text) -> std::optional<json> {
        return json_parser(text).parse();
    }

} // namespace ftp_test
