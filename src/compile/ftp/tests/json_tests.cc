#include <catch2/catch_test_macros.hpp>

#include "compile/ftp/json.h"

#include <string>

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
