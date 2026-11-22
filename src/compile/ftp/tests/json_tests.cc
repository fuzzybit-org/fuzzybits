#include <catch2/catch_test_macros.hpp>

#include "compile/ftp/json.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace {

    namespace json = fuzzybits::compile::ftp::json;

} // namespace

TEST_CASE("json parses documents and keeps key order", "[compile][ftp][json]") {
    const auto parsed = json::parse(R"({"b": [1, -2.5e1, true, null, "x\n\u00e9\ud83d\ude00"], "a": {"k": 0}})");
    REQUIRE(parsed);
    auto const &root = parsed.value();
    REQUIRE(root.is_object());
    REQUIRE(root.keys == std::vector<std::string>{"b", "a"});
    auto const &list = root.find("b")->items;
    REQUIRE(list.size() == 5);
    REQUIRE(list[0].is_integer());
    REQUIRE(list[0].integer == 1);
    REQUIRE(list[1].kind == json::value::kind_type::number);
    REQUIRE(list[1].number == -25.0);
    REQUIRE(list[2].boolean);
    REQUIRE(list[3].kind == json::value::kind_type::null);
    REQUIRE(list[4].string == "x\n\xC3\xA9\xF0\x9F\x98\x80");
    REQUIRE(root.find("a")->find("k")->integer == 0);
    REQUIRE(root.find("missing") == nullptr);
}

TEST_CASE("json rejects malformed documents with a stable id", "[compile][ftp][json]") {
    for (const char *text : {"", "{", "[1,]", "{\"a\":1,}", "{\"a\":1 \"b\":2}", "{\"a\":1,\"a\":2}", "01", "1.", "\"\\x\"", "\"\\ud800\"",
                             "[1] 2", "nul", "\"unterminated", "{\"a\":\"\t\"}", "1e999"}) {
        INFO(text);
        const auto parsed = json::parse(text);
        REQUIRE_FALSE(parsed);
        REQUIRE(parsed.diagnostics().items().front().code == "compile.ftp.invalid-json");
    }
    REQUIRE_FALSE(json::parse(std::string(200, '[') + std::string(200, ']')));
}

TEST_CASE("json dump round-trips", "[compile][ftp][json]") {
    auto root = json::value::make_object();
    root.set("text", json::value::make_string("a\"b\\c\n\x01"));
    root.set("list", json::value::make_array().push(json::value::make_int(-3)).push(json::value::make_number(0.1)).push(json::value::make_bool(false)));
    root.set("none", json::value::make_null());
    root.set("empty", json::value::make_object());
    const auto compact = json::dump(root);
    REQUIRE(compact == R"({"text":"a\"b\\c\n\u0001","list":[-3,0.1,false],"none":null,"empty":{}})");
    for (const unsigned indent : {0U, 2U}) {
        const auto again = json::parse(json::dump(root, indent));
        REQUIRE(again);
        REQUIRE(json::dump(again.value()) == compact);
    }
}

namespace {

    [[nodiscard]] auto parse_ok(std::string const &text) -> json::value {
        auto parsed = json::parse(text);
        INFO(text);
        REQUIRE(parsed);
        return std::move(parsed).value();
    }

    void require_rejected(std::string const &text) {
        const auto parsed = json::parse(text);
        INFO(text.substr(0, 60));
        REQUIRE_FALSE(parsed);
        REQUIRE(parsed.diagnostics().items().front().code == "compile.ftp.invalid-json");
    }

    [[nodiscard]] auto nested(const std::size_t levels, const char open, const char close) -> std::string {
        std::string out;
        for (std::size_t level = 0; level < levels; ++level) {
            out += open;
            if (open == '{') {
                out += "\"k\":";
            }
        }
        out += "0";
        out.append(levels, close);
        return out;
    }

} // namespace

TEST_CASE("json string escapes and surrogate pairs", "[compile][ftp][json]") {
    REQUIRE(parse_ok(R"("\"\\\/\b\f\n\r\t")").string == "\"\\/\b\f\n\r\t");
    REQUIRE(parse_ok(R"("\u0041\u00e9\u20ac")").string == "A\xC3\xA9\xE2\x82\xAC");
    REQUIRE(parse_ok(R"("\uD83D\uDE00")").string == "\xF0\x9F\x98\x80");
    REQUIRE(parse_ok(R"("\uD83d\ude00")").string == "\xF0\x9F\x98\x80");
    REQUIRE(parse_ok(R"("\u0000")").string == std::string(1, '\0'));
    REQUIRE(parse_ok("\"caf\xC3\xA9 \xE2\x82\xAC\"").string == "caf\xC3\xA9 \xE2\x82\xAC");
    // Keys decode like values.
    REQUIRE(parse_ok(R"({"\u0041\n": 1})").keys.front() == "A\n");
    for (const char *bad : {R"("\uD83D")", R"("\uD83Dx")", R"("\uD83D\u0041")", R"("\uDE00")", R"("\uD83D\uD83D")", R"("\u12G4")", R"("\u12")",
                            R"("\U0041")", R"("\a")", R"("\')", "\"\n\"", "\"\x1f\"", "\"abc"}) {
        require_rejected(bad);
    }
}

TEST_CASE("json numbers", "[compile][ftp][json]") {
    for (const auto &[text, expected] : std::vector<std::pair<std::string, double>>{
             {"0", 0.0}, {"-0", 0.0}, {"7", 7.0}, {"-12", -12.0}, {"1.5", 1.5}, {"-0.25", -0.25}, {"1e2", 100.0}, {"1E+2", 100.0}, {"1e-2", 0.01},
             {"-12.5e3", -12500.0}, {"0.0", 0.0}, {"1.0e0", 1.0}, {"2.5e-3", 0.0025}}) {
        const auto value = parse_ok(text);
        INFO(text);
        const auto as_double = value.is_integer() ? static_cast<double>(value.integer) : value.number;
        REQUIRE(as_double == expected);
        // An integer is a number without fraction and exponent.
        REQUIRE(value.is_integer() == (text.find_first_of(".eE") == std::string::npos));
    }
    REQUIRE(parse_ok("9223372036854775807").integer == 9223372036854775807LL);
    REQUIRE(parse_ok("-9223372036854775808").integer == (-9223372036854775807LL - 1));
    // Beyond int64 an integral literal becomes a double instead of wrapping.
    const auto big = parse_ok("9223372036854775808");
    REQUIRE(big.kind == json::value::kind_type::number);
    REQUIRE(big.number == 9223372036854775808.0);
    REQUIRE(parse_ok("1e308").number == 1e308);
    for (const char *bad : {"1e309", "-1e309", "1e99999", "1e-99999", "+1", ".5", "5.", "-", "--1", "1e", "1e+", "1.e2", "0x10", "01", "-01", "1_000",
                            "NaN", "Infinity", "-Infinity", "1 2", "1,2"}) {
        require_rejected(bad);
    }
}

TEST_CASE("json nesting limit", "[compile][ftp][json]") {
    // 128 levels of containers parse; the 129th is refused, for arrays and objects, without recursing further.
    REQUIRE(json::parse(nested(128, '[', ']')));
    REQUIRE(json::parse(nested(128, '{', '}')));
    require_rejected(nested(129, '[', ']'));
    require_rejected(nested(129, '{', '}'));
    require_rejected(std::string(100000, '['));
    require_rejected(std::string(100000, '[') + std::string(100000, ']'));
    require_rejected(nested(5000, '{', '}'));
    // Wide is fine.
    std::string wide = "[";
    for (int index = 0; index < 20000; ++index) {
        wide += (index == 0 ? "" : ",") + std::string("[1]");
    }
    wide += "]";
    REQUIRE(parse_ok(wide).items.size() == 20000);
}

TEST_CASE("json duplicate keys and trailing data", "[compile][ftp][json]") {
    require_rejected(R"({"a": 1, "a": 2})");
    require_rejected(R"({"a": 1, "b": {"c": 1, "c": 2}})");
    require_rejected(R"({"a": 1, "\u0061": 2})");
    // The same key in different objects is fine.
    REQUIRE(parse_ok(R"({"a": {"k": 1}, "b": {"k": 2}, "c": [{"k": 3}, {"k": 4}]})").items.size() == 3);

    for (const char *bad : {"{} x", "[1]]", "{}{}", "1 2", "true false", "{} ,", "[] //", "\"a\" \"b\"", "null0"}) {
        require_rejected(bad);
    }
    // Surrounding whitespace is not data; a byte-order mark and other whitespace are.
    REQUIRE(parse_ok(" \t\r\n{} \t\r\n").is_object());
    require_rejected("\xEF\xBB\xBF{}");
    require_rejected("\f{}");
    require_rejected("");
    require_rejected(" \n ");
    // Structure errors.
    for (const char *bad : {"{", "}", "[", "]", "{\"a\"}", "{\"a\":}", "{\"a\" 1}", "{1: 2}", "{'a': 1}", "{\"a\": 1,}", "[,1]", "[1,,2]", "[1 2]", "tru", "nul",
                            "True", "NULL"}) {
        require_rejected(bad);
    }
}

TEST_CASE("json errors say where", "[compile][ftp][json]") {
    const auto parsed = json::parse("{\n  \"a\": 1,\n  \"b\": ?\n}");
    REQUIRE_FALSE(parsed);
    REQUIRE(parsed.diagnostics().items().front().message.find("line 3, column 8") != std::string::npos);
    const auto trailing = json::parse("{}\nx");
    REQUIRE_FALSE(trailing);
    REQUIRE(trailing.diagnostics().items().front().message.find("line 2, column 1") != std::string::npos);
}
