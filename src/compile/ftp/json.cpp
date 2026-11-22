#include "compile/ftp/json.h"

#include "api/diagnostics.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>
#include <system_error>

namespace fuzzybits::compile::ftp::json {

    namespace {

        constexpr unsigned max_depth = 128;

        class parser final {
        public:
            explicit parser(const std::string_view text) : text_(text) {}

            [[nodiscard]] auto run() -> api::result<value> {
                skip();
                auto document = parse_value(0);
                if (!document) {
                    return api::result<value>::fail(api::make_error("compile.ftp.invalid-json", error_));
                }
                skip();
                if (cursor_ != text_.size()) {
                    return api::result<value>::fail(api::make_error("compile.ftp.invalid-json", at("unexpected trailing characters")));
                }
                return api::result<value>::ok(std::move(*document));
            }

        private:
            [[nodiscard]] auto at(const std::string &message) const -> std::string {
                std::size_t line = 1;
                std::size_t column = 1;
                for (std::size_t index = 0; index < cursor_ && index < text_.size(); ++index) {
                    if (text_[index] == '\n') {
                        ++line;
                        column = 1;
                    } else {
                        ++column;
                    }
                }
                return message + " at line " + std::to_string(line) + ", column " + std::to_string(column);
            }

            auto fail(const std::string &message) -> std::optional<value> {
                error_ = at(message);
                return std::nullopt;
            }

            void skip() {
                while (cursor_ < text_.size() && (text_[cursor_] == ' ' || text_[cursor_] == '\n' || text_[cursor_] == '\r' || text_[cursor_] == '\t')) {
                    ++cursor_;
                }
            }

            [[nodiscard]] auto consume(const char expected) -> bool {
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

            auto parse_value(const unsigned depth) -> std::optional<value> {
                if (depth > max_depth) {
                    return fail("nesting is too deep");
                }
                if (cursor_ >= text_.size()) {
                    return fail("unexpected end of input");
                }
                const char head = text_[cursor_];
                if (head == '{') {
                    return parse_object(depth);
                }
                if (head == '[') {
                    return parse_array(depth);
                }
                if (head == '"') {
                    auto text = parse_string();
                    return text ? std::optional<value>(value::make_string(std::move(*text))) : std::nullopt;
                }
                if (literal("true")) {
                    return value::make_bool(true);
                }
                if (literal("false")) {
                    return value::make_bool(false);
                }
                if (literal("null")) {
                    return value::make_null();
                }
                return parse_number();
            }

            auto parse_object(const unsigned depth) -> std::optional<value> {
                ++cursor_;
                auto out = value::make_object();
                skip();
                if (consume('}')) {
                    return out;
                }
                while (true) {
                    skip();
                    if (cursor_ >= text_.size() || text_[cursor_] != '"') {
                        return fail("expected a string key");
                    }
                    auto key = parse_string();
                    if (!key) {
                        return std::nullopt;
                    }
                    skip();
                    if (!consume(':')) {
                        return fail("expected ':'");
                    }
                    skip();
                    auto member = parse_value(depth + 1);
                    if (!member) {
                        return std::nullopt;
                    }
                    if (out.find(*key) != nullptr) {
                        return fail("duplicate key '" + *key + "'");
                    }
                    out.set(std::move(*key), std::move(*member));
                    skip();
                    if (consume(',')) {
                        continue;
                    }
                    if (consume('}')) {
                        return out;
                    }
                    return fail("expected ',' or '}'");
                }
            }

            auto parse_array(const unsigned depth) -> std::optional<value> {
                ++cursor_;
                auto out = value::make_array();
                skip();
                if (consume(']')) {
                    return out;
                }
                while (true) {
                    skip();
                    auto member = parse_value(depth + 1);
                    if (!member) {
                        return std::nullopt;
                    }
                    out.push(std::move(*member));
                    skip();
                    if (consume(',')) {
                        continue;
                    }
                    if (consume(']')) {
                        return out;
                    }
                    return fail("expected ',' or ']'");
                }
            }

            [[nodiscard]] auto hex4(std::uint32_t &out) -> bool {
                if (cursor_ + 4 > text_.size()) {
                    return false;
                }
                out = 0;
                for (int digit = 0; digit < 4; ++digit) {
                    const char c = text_[cursor_++];
                    out <<= 4;
                    if (c >= '0' && c <= '9') {
                        out |= static_cast<std::uint32_t>(c - '0');
                    } else if (c >= 'a' && c <= 'f') {
                        out |= static_cast<std::uint32_t>(c - 'a' + 10);
                    } else if (c >= 'A' && c <= 'F') {
                        out |= static_cast<std::uint32_t>(c - 'A' + 10);
                    } else {
                        return false;
                    }
                }
                return true;
            }

            static void append_utf8(std::string &out, const std::uint32_t code) {
                if (code < 0x80) {
                    out.push_back(static_cast<char>(code));
                } else if (code < 0x800) {
                    out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                } else if (code < 0x10000) {
                    out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                    out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                } else {
                    out.push_back(static_cast<char>(0xF0 | (code >> 18)));
                    out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                }
            }

            auto parse_string() -> std::optional<std::string> {
                ++cursor_;
                std::string out;
                while (cursor_ < text_.size()) {
                    const char c = text_[cursor_++];
                    if (c == '"') {
                        return out;
                    }
                    if (static_cast<unsigned char>(c) < 0x20) {
                        --cursor_;
                        error_ = at("control character in string");
                        return std::nullopt;
                    }
                    if (c != '\\') {
                        out.push_back(c);
                        continue;
                    }
                    if (cursor_ >= text_.size()) {
                        break;
                    }
                    const char escape = text_[cursor_++];
                    switch (escape) {
                    case '"':
                    case '\\':
                    case '/':
                        out.push_back(escape);
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    case 'u': {
                        std::uint32_t code = 0;
                        if (!hex4(code)) {
                            error_ = at("invalid \\u escape");
                            return std::nullopt;
                        }
                        if (code >= 0xD800 && code <= 0xDBFF) {
                            std::uint32_t low = 0;
                            if (!literal("\\u") || !hex4(low) || low < 0xDC00 || low > 0xDFFF) {
                                error_ = at("unpaired surrogate in \\u escape");
                                return std::nullopt;
                            }
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                        } else if (code >= 0xDC00 && code <= 0xDFFF) {
                            error_ = at("unpaired surrogate in \\u escape");
                            return std::nullopt;
                        }
                        append_utf8(out, code);
                        break;
                    }
                    default:
                        --cursor_;
                        error_ = at("invalid escape");
                        return std::nullopt;
                    }
                }
                error_ = at("unterminated string");
                return std::nullopt;
            }

            auto parse_number() -> std::optional<value> {
                const auto start = cursor_;
                bool integral = true;
                consume('-');
                if (cursor_ >= text_.size() || text_[cursor_] < '0' || text_[cursor_] > '9') {
                    cursor_ = start;
                    return fail("unexpected character");
                }
                if (text_[cursor_] == '0') {
                    ++cursor_;
                } else {
                    while (cursor_ < text_.size() && text_[cursor_] >= '0' && text_[cursor_] <= '9') {
                        ++cursor_;
                    }
                }
                if (consume('.')) {
                    integral = false;
                    const auto digits = cursor_;
                    while (cursor_ < text_.size() && text_[cursor_] >= '0' && text_[cursor_] <= '9') {
                        ++cursor_;
                    }
                    if (cursor_ == digits) {
                        return fail("expected digits after '.'");
                    }
                }
                if (cursor_ < text_.size() && (text_[cursor_] == 'e' || text_[cursor_] == 'E')) {
                    integral = false;
                    ++cursor_;
                    if (cursor_ < text_.size() && (text_[cursor_] == '+' || text_[cursor_] == '-')) {
                        ++cursor_;
                    }
                    const auto digits = cursor_;
                    while (cursor_ < text_.size() && text_[cursor_] >= '0' && text_[cursor_] <= '9') {
                        ++cursor_;
                    }
                    if (cursor_ == digits) {
                        return fail("expected exponent digits");
                    }
                }
                const auto begin = text_.data() + start;
                const auto end = text_.data() + cursor_;
                if (integral) {
                    std::int64_t integer = 0;
                    if (const auto [pointer, status] = std::from_chars(begin, end, integer); status == std::errc{} && pointer == end) {
                        return value::make_int(integer);
                    }
                }
                double number = 0.0;
                if (const auto [pointer, status] = std::from_chars(begin, end, number); status != std::errc{} || pointer != end ||
                                                                                           !std::isfinite(number)) {
                    cursor_ = start;
                    return fail("number out of range");
                }
                return value::make_number(number);
            }

            std::string_view text_;
            std::size_t cursor_ = 0;
            std::string error_;
        };

        void escape(std::string &out, std::string_view text) {
            out.push_back('"');
            for (const char c : text) {
                switch (c) {
                case '"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                case '\b':
                    out += "\\b";
                    break;
                case '\f':
                    out += "\\f";
                    break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        std::array<char, 8> buffer{};
                        std::snprintf(buffer.data(), buffer.size(), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                        out += buffer.data();
                    } else {
                        out.push_back(c);
                    }
                }
            }
            out.push_back('"');
        }

        void newline(std::string &out, const unsigned indent, const unsigned depth) {
            if (indent == 0) {
                return;
            }
            out.push_back('\n');
            out.append(static_cast<std::size_t>(indent) * depth, ' ');
        }

        void emit(std::string &out, value const &node, const unsigned indent, const unsigned depth) {
            using kind = value::kind_type;
            switch (node.kind) {
            case kind::null:
                out += "null";
                return;
            case kind::boolean:
                out += node.boolean ? "true" : "false";
                return;
            case kind::integer:
                out += std::to_string(node.integer);
                return;
            case kind::number: {
                if (!std::isfinite(node.number)) {
                    out += "null";
                    return;
                }
                std::array<char, 32> buffer{};
                const auto [end, status] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), node.number);
                out.append(buffer.data(), status == std::errc{} ? end : buffer.data());
                return;
            }
            case kind::string:
                escape(out, node.string);
                return;
            case kind::array:
            case kind::object: {
                const bool object = node.kind == kind::object;
                out.push_back(object ? '{' : '[');
                for (std::size_t index = 0; index < node.items.size(); ++index) {
                    if (index != 0) {
                        out.push_back(',');
                    }
                    newline(out, indent, depth + 1);
                    if (object) {
                        escape(out, node.keys[index]);
                        out += indent == 0 ? ":" : ": ";
                    }
                    emit(out, node.items[index], indent, depth + 1);
                }
                if (!node.items.empty()) {
                    newline(out, indent, depth);
                }
                out.push_back(object ? '}' : ']');
                return;
            }
            }
        }

    } // namespace

    auto parse(const std::string_view text) -> api::result<value> {
        return parser(text).run();
    }

    auto dump(value const &document, const unsigned indent) -> std::string {
        std::string out;
        emit(out, document, indent, 0);
        return out;
    }

} // namespace fuzzybits::compile::ftp::json
