#pragma once

#include "ir/pc/circuit.h"
#include "ir/pc/hull/hull.h"
#include "ir/pc/op_code.h"
#include "ir/pc/probability.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pc::hull::testing {

// Plain two-valued evaluation of every node; node ids are topological in
// circuits built through pc::builder. A complemented edge into an event e
// with complement_values[e] != unset reads that value instead of not x_e
// (the atom-split reading).
inline constexpr std::uint8_t unset = 2;

inline auto evaluate_nodes(circuit const &model, std::span<std::uint8_t const> event_values, std::span<std::uint8_t const> complement_values = {})
    -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> value(model.node_count(), 0);
    std::vector<std::uint32_t> event_of(model.node_count(), 0);
    std::uint32_t event = 0;
    for (std::uint32_t node = 0; node < model.node_count(); ++node) {
        const auto code = model.op_codes()[node];
        std::uint32_t ones = 0;
        std::uint32_t count = 0;
        for (auto const &edge : model.operands(bool_node_id(node))) {
            const auto source = edge.source.raw();
            if (edge.complement && model.op_codes()[source] == op_code::basic_event && !complement_values.empty() &&
                complement_values[event_of[source]] != unset) {
                ones += complement_values[event_of[source]];
            } else {
                ones += value[source] ^ (edge.complement ? 1U : 0U);
            }
            ++count;
        }
        switch (code) {
        case op_code::basic_event:
            event_of[node] = event;
            value[node] = event_values[event++];
            break;
        case op_code::constant:
        case op_code::house_event:
            value[node] = model.constant_values()[node];
            break;
        case op_code::and_gate:
            value[node] = ones == count;
            break;
        case op_code::or_gate:
            value[node] = ones > 0;
            break;
        case op_code::xor_gate:
            value[node] = ones & 1U;
            break;
        case op_code::not_gate:
            value[node] = ones == 0;
            break;
        case op_code::nand_gate:
            value[node] = ones != count;
            break;
        case op_code::nor_gate:
            value[node] = ones == 0;
            break;
        case op_code::xnor_gate:
            value[node] = (ones & 1U) == 0;
            break;
        case op_code::cardinality:
            value[node] = ones >= model.cardinality_low()[node] && ones <= model.cardinality_high()[node];
            break;
        }
    }
    return value;
}

inline auto event_probability(circuit const &model, bool_node_id node) -> double {
    const auto source = model.probability_source_refs()[node.raw()];
    auto const &probability = model.probability_sources()[source.raw()];
    if (probability.kind != probability_source_kind::literal) {
        throw std::runtime_error("event without a literal probability");
    }
    return probability.literal;
}

// Exact P(check = 1) by enumerating every assignment of the cone's events.
inline auto exact_probability(circuit const &model, checker const &check) -> double {
    auto const &cone = check.cone_events();
    const auto variables = cone.size();
    if (variables > 24) {
        throw std::runtime_error("too many events to enumerate");
    }
    std::vector<double> probability(variables);
    for (std::size_t bit = 0; bit < variables; ++bit) {
        probability[bit] = event_probability(model, check.event_node(cone[bit]));
    }
    const std::size_t lanes = std::size_t{1} << variables;
    const std::size_t stride = std::max<std::size_t>(1, (std::size_t{check.event_count()} + 63) / 64);
    std::vector<std::uint64_t> words(lanes * stride, 0);
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        for (std::size_t bit = 0; bit < variables; ++bit) {
            if (((lane >> bit) & 1U) != 0) {
                const auto event = cone[bit];
                words[lane * stride + event / 64] |= std::uint64_t{1} << (event % 64);
            }
        }
    }
    std::vector<std::uint64_t> out((lanes + 63) / 64, 0);
    check.check(lane_matrix{.words = words, .lane_count = lanes, .stride = stride}, out);
    // Neumaier summation: long double is double on some platforms.
    double total = 0.0;
    double compensation = 0.0;
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        if (((out[lane / 64] >> (lane % 64)) & 1U) == 0) {
            continue;
        }
        double weight = 1.0;
        for (std::size_t bit = 0; bit < variables; ++bit) {
            weight *= ((lane >> bit) & 1U) != 0 ? probability[bit] : 1.0 - probability[bit];
        }
        const auto sum = total + weight;
        compensation += std::abs(total) >= std::abs(weight) ? (total - sum) + weight : (weight - sum) + total;
        total = sum;
    }
    return total + compensation;
}

// Minimal JSON reader for fixture expectations (objects, arrays, strings
// without escapes beyond \" and \\, numbers, literals).
struct json final {
    enum class kind : std::uint8_t { null, boolean, number, string, array, object };
    kind type = kind::null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    // Arrays use `items`; objects use `keys` with the values in `items`.
    std::vector<json> items;
    std::vector<std::string> keys;

    [[nodiscard]] auto find(std::string_view key) const -> json const * {
        for (std::size_t index = 0; index < keys.size(); ++index) {
            if (keys[index] == key) {
                return &items[index];
            }
        }
        return nullptr;
    }
};

class json_reader final {
  public:
    explicit json_reader(std::string_view input) : input_(input) {}

    [[nodiscard]] auto parse() -> json {
        auto value = parse_value();
        skip();
        if (position_ != input_.size()) {
            fail();
        }
        return value;
    }

  private:
    [[noreturn]] void fail() const { throw std::runtime_error("invalid JSON at offset " + std::to_string(position_)); }

    void skip() {
        while (position_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[position_])) != 0) {
            ++position_;
        }
    }

    [[nodiscard]] auto consume(const char expected) -> bool {
        skip();
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    [[nodiscard]] auto parse_string() -> std::string {
        if (!consume('"')) {
            fail();
        }
        std::string out;
        while (position_ < input_.size() && input_[position_] != '"') {
            if (input_[position_] == '\\') {
                ++position_;
                if (position_ >= input_.size()) {
                    fail();
                }
            }
            out.push_back(input_[position_++]);
        }
        if (position_++ >= input_.size()) {
            fail();
        }
        return out;
    }

    [[nodiscard]] auto parse_value() -> json {
        skip();
        if (position_ >= input_.size()) {
            fail();
        }
        json value;
        const auto first = input_[position_];
        if (first == '{') {
            ++position_;
            value.type = json::kind::object;
            if (consume('}')) {
                return value;
            }
            do {
                skip();
                auto key = parse_string();
                if (!consume(':')) {
                    fail();
                }
                value.keys.push_back(std::move(key));
                value.items.push_back(parse_value());
            } while (consume(','));
            if (!consume('}')) {
                fail();
            }
        } else if (first == '[') {
            ++position_;
            value.type = json::kind::array;
            if (consume(']')) {
                return value;
            }
            do {
                value.items.push_back(parse_value());
            } while (consume(','));
            if (!consume(']')) {
                fail();
            }
        } else if (first == '"') {
            value.type = json::kind::string;
            value.text = parse_string();
        } else if (input_.substr(position_, 4) == "null") {
            position_ += 4;
        } else if (input_.substr(position_, 4) == "true") {
            position_ += 4;
            value.type = json::kind::boolean;
            value.boolean = true;
        } else if (input_.substr(position_, 5) == "false") {
            position_ += 5;
            value.type = json::kind::boolean;
        } else {
            const std::string token(input_.substr(position_, input_.find_first_of(",]} \n\r\t", position_) - position_));
            char *end = nullptr;
            value.number = std::strtod(token.c_str(), &end);
            if (token.empty() || end != token.c_str() + token.size()) {
                fail();
            }
            value.type = json::kind::number;
            position_ += token.size();
        }
        return value;
    }

    std::string_view input_;
    std::size_t position_ = 0;
};

} // namespace fuzzybits::ir::pc::hull::testing
