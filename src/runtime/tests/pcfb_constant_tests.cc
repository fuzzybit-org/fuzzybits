// fzb run on pcfb constants and repeated operands (#259).
//
// The reconstruction used to hand ONES / ZEROS to the PDAG as `Gate::AddArg(constant)`, which is constant
// propagation and assumes the gate's other arguments are already present, and the kernels dropped a constant
// argument. The result depended on the opcode, on whether the constant was named, and on how many operands
// came before it. Repeated operands went through the same kind of rewrite. These cases cover every opcode with
// each kind of constant after zero, one and several operands, in both polarities, and compare every query with
// exact enumeration.

#include "compile/exact_eval.h"
#include "io/pcfb/tests/portable_random.h"
#include "pcfb.h"
#include "pcfb/builder.h"
#include "runtime/runtime.h"
#include "runtime/tests/test_backend.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using pcfb::OpCode;
using tensor = std::uint32_t;

constexpr std::uint64_t kTrials = std::uint64_t{1} << 18;

struct operands final {
    std::vector<tensor> inputs;
    std::vector<tensor> inverted;
};

class circuit_builder final {
  public:
    auto sample(double probability) -> tensor {
        auto t = builder_.add_sample(probability, "x" + std::to_string(next_event_), next_event_);
        ++next_event_;
        REQUIRE(t);
        return *t;
    }

    auto op(OpCode opcode, operands const &in, std::optional<std::uint32_t> k = std::nullopt,
            std::string const &name = {}) -> tensor {
        auto t = builder_.add_op(opcode, in.inputs, in.inverted, k, name);
        REQUIRE(t);
        return *t;
    }

    void query(tensor target) {
        REQUIRE(builder_.add_query(target, pcfb::QueryKind::PROBABILITY, "q" + std::to_string(queries_)));
        ++queries_;
    }

    auto finish() -> std::vector<std::byte> {
        auto bytes = builder_.finish();
        REQUIRE(bytes);
        return std::move(*bytes);
    }

  private:
    pcfb::builder builder_;
    std::uint32_t next_event_ = 1;
    std::uint32_t queries_ = 0;
};

// Every query against exact enumeration: hits = 0 or n where the query is constant, |z| <= 4 elsewhere.
void require_matches_exact(std::vector<std::byte> const &bytes) {
    const auto circuit = fuzzybits::runtime::load_pcfb(bytes);
    fuzzybits::runtime::RunOptions options;
    options.trials = kTrials;
    options.seed = 372;
    options.backend = fuzzybits::testing::backend();
    const auto results = fuzzybits::runtime::run_pcfb(circuit, options);
    const auto count = circuit.subcircuits()->Get(0)->queries()->size();
    REQUIRE(results.size() == count);
    for (std::uint32_t index = 0; index < count; ++index) {
        const double exact = fuzzybits::compile::exact_probability(circuit, index);
        auto const &result = results[index];
        INFO(result.name << ": exact " << exact << ", hits " << result.hits << " of " << result.n);
        REQUIRE(result.n >= kTrials);
        const auto n = static_cast<double>(result.n);
        if (exact < 1.0e-12) {
            CHECK(result.hits == 0);
        } else if (exact > 1.0 - 1.0e-12) {
            CHECK(result.hits == result.n);
        } else {
            const double spread = std::sqrt(n * exact * (1.0 - exact));
            CHECK(std::abs(static_cast<double>(result.hits) - n * exact) <= 4.0 * spread + 1.0);
        }
    }
}

struct constant_kind final {
    OpCode opcode;
    const char *name; // empty: anonymous
};

constexpr std::array kConstants{
    constant_kind{OpCode::BITWISE_ONES, ""},
    constant_kind{OpCode::BITWISE_ZEROS, ""},
    constant_kind{OpCode::BITWISE_ONES, "IE"}, // an FTP override or condition keeps the element's name
    constant_kind{OpCode::BITWISE_ZEROS, "OFF"},
};

struct gate_kind final {
    OpCode opcode;
    std::optional<std::uint32_t> k;
};

// Four operands: three samples and the constant.
constexpr std::array kGates{
    gate_kind{OpCode::BITWISE_AND, std::nullopt}, gate_kind{OpCode::BITWISE_OR, std::nullopt},
    gate_kind{OpCode::BITWISE_XOR, std::nullopt}, gate_kind{OpCode::BITWISE_NAND, std::nullopt},
    gate_kind{OpCode::BITWISE_NOR, std::nullopt}, gate_kind{OpCode::BITWISE_XNOR, std::nullopt},
    gate_kind{OpCode::BITWISE_ATLEAST, 2},        gate_kind{OpCode::BITWISE_ATLEAST, 1},
    gate_kind{OpCode::BITWISE_ATLEAST, 4},        gate_kind{OpCode::BITWISE_ATMOST, 1},
    gate_kind{OpCode::BITWISE_ATMOST, 3},         gate_kind{OpCode::BITWISE_EXACTLY, 2},
    gate_kind{OpCode::BITWISE_EXACTLY, 0},
};

} // namespace

TEST_CASE("fzb run evaluates ONES and ZEROS under every opcode, in every position and polarity (#259)",
          "[runtime][mc][pcfb][constants]") {
    circuit_builder b;
    const std::array<tensor, 3> x{b.sample(0.3), b.sample(0.6), b.sample(0.45)};
    const auto y = b.sample(0.7);

    std::vector<tensor> constants;
    for (auto const &kind : kConstants) {
        constants.push_back(b.op(kind.opcode, {}, std::nullopt, kind.name));
    }

    for (const auto constant : constants) {
        // A query on the constant itself.
        b.query(constant);
        // NOT and NO_OP over it, in both polarities.
        for (const auto opcode : {OpCode::BITWISE_NOT, OpCode::BITWISE_NO_OP}) {
            b.query(b.op(opcode, {.inputs = {constant}}));
            b.query(b.op(opcode, {.inverted = {constant}}));
        }
        for (auto const &gate : kGates) {
            // The reader adds `inputs` and then `inverted_inputs`, so the constant comes after 0, 1 and 3
            // operands in `inputs`, and after 0, 1 and 3 inverted operands with no positive ones before.
            for (const std::size_t position : {0U, 1U, 3U}) {
                std::vector<tensor> list(x.begin(), x.end());
                list.insert(list.begin() + static_cast<std::ptrdiff_t>(position), constant);
                b.query(b.op(gate.opcode, {.inputs = list}, gate.k));
                b.query(b.op(gate.opcode, {.inverted = list}, gate.k));
            }
            // A positive operand before an inverted constant.
            b.query(b.op(gate.opcode, {.inputs = {x[0]}, .inverted = {constant, x[1], x[2]}}, gate.k));
        }
    }

    // Constants used by several parents and nested: OR(AND(x0, ONES), ZEROS, NOR(IE, x1)) and the like.
    const auto t = constants[0];
    const auto f = constants[1];
    const auto named_t = constants[2];
    const auto inner = b.op(OpCode::BITWISE_AND, {.inputs = {x[0], t}});
    const auto nor = b.op(OpCode::BITWISE_NOR, {.inputs = {named_t, x[1]}});
    b.query(b.op(OpCode::BITWISE_OR, {.inputs = {inner, f, nor}}));
    b.query(b.op(OpCode::BITWISE_AND, {.inputs = {y, inner}, .inverted = {f}}));
    b.query(b.op(OpCode::BITWISE_XOR, {.inputs = {t, inner, named_t, y}}));

    require_matches_exact(b.finish());
}

TEST_CASE("fzb run counts repeated operands as pcfb does (#259)", "[runtime][mc][pcfb][constants]") {
    circuit_builder b;
    const auto x0 = b.sample(0.3);
    const auto x1 = b.sample(0.6);
    const auto x2 = b.sample(0.45);
    const auto t = b.op(OpCode::BITWISE_ONES, {});
    const auto f = b.op(OpCode::BITWISE_ZEROS, {}, std::nullopt, "OFF");
    const auto g = b.op(OpCode::BITWISE_AND, {.inputs = {x1, x2}});

    // Idempotent folds: AND(x0, x0, x1, x2) used to become a pass-through of its last operand.
    for (const auto opcode : {OpCode::BITWISE_AND, OpCode::BITWISE_OR, OpCode::BITWISE_NAND, OpCode::BITWISE_NOR}) {
        b.query(b.op(opcode, {.inputs = {x0, x0, x1, x2}}));
        b.query(b.op(opcode, {.inputs = {x1, g, g}, .inverted = {x0, x0}}));
        b.query(b.op(opcode, {.inputs = {t, t, x0}}));
        b.query(b.op(opcode, {.inputs = {x0}, .inverted = {f, f}}));
    }
    // XOR and XNOR cancel in pairs, and may read one tensor in both polarities (spec §3.1).
    for (const auto opcode : {OpCode::BITWISE_XOR, OpCode::BITWISE_XNOR}) {
        b.query(b.op(opcode, {.inputs = {x0, x0, x1}}));
        b.query(b.op(opcode, {.inputs = {x0, x1}, .inverted = {x0}}));
        b.query(b.op(opcode, {.inputs = {x0, x0, x0, x2}}));
        b.query(b.op(opcode, {.inputs = {t, t, x1}}));
    }
    // k-of-n counts multiplicity.
    b.query(b.op(OpCode::BITWISE_ATLEAST, {.inputs = {x0, x0, x1}}, 2));
    b.query(b.op(OpCode::BITWISE_ATLEAST, {.inputs = {x0, x0, x1, x2}}, 3));
    b.query(b.op(OpCode::BITWISE_ATMOST, {.inputs = {x0, x0, x1}}, 1));
    b.query(b.op(OpCode::BITWISE_EXACTLY, {.inputs = {x0, x1, x0, x2}}, 2));
    b.query(b.op(OpCode::BITWISE_EXACTLY, {.inputs = {g, g}, .inverted = {x0, x0}}, 2));
    b.query(b.op(OpCode::BITWISE_ATLEAST, {.inputs = {t, t, x0}}, 3));
    // Both polarities outside the XOR family: not written by conforming writers, but accepted by
    // `pcfb::validate`, so answered like any other file.
    b.query(b.op(OpCode::BITWISE_AND, {.inputs = {x0, x1}, .inverted = {x0}}));
    b.query(b.op(OpCode::BITWISE_OR, {.inputs = {x0}, .inverted = {x0, x1}}));
    b.query(b.op(OpCode::BITWISE_ATLEAST, {.inputs = {x0, x1}, .inverted = {x0}}, 2));

    require_matches_exact(b.finish());
}

TEST_CASE("fzb run matches exact enumeration on random circuits with constants and repeated operands (#259)",
          "[runtime][mc][pcfb][constants]") {
    namespace random = pcfb_test_random;
    std::mt19937_64 rng(259);
    constexpr std::array kOpcodes{
        OpCode::BITWISE_AND,     OpCode::BITWISE_OR,   OpCode::BITWISE_XOR,     OpCode::BITWISE_NAND,
        OpCode::BITWISE_NOR,     OpCode::BITWISE_XNOR, OpCode::BITWISE_ATLEAST, OpCode::BITWISE_ATMOST,
        OpCode::BITWISE_EXACTLY, OpCode::BITWISE_NOT,  OpCode::BITWISE_NO_OP,
    };

    for (int round = 0; round < 4; ++round) {
        circuit_builder b;
        std::vector<tensor> pool;
        for (int event = 0; event < 6; ++event) {
            pool.push_back(b.sample(random::real(rng, 0.2, 0.8)));
        }
        for (auto const &kind : kConstants) {
            const auto constant = b.op(kind.opcode, {}, std::nullopt, kind.name);
            pool.push_back(constant);
            b.query(constant);
        }
        for (int index = 0; index < 60; ++index) {
            const auto opcode = kOpcodes[random::uniform(rng, 0, kOpcodes.size() - 1)];
            const bool unary = opcode == OpCode::BITWISE_NOT || opcode == OpCode::BITWISE_NO_OP;
            const bool xor_family = opcode == OpCode::BITWISE_XOR || opcode == OpCode::BITWISE_XNOR;
            const auto arity = unary ? 1U : static_cast<std::uint32_t>(random::uniform(rng, 1, 5));
            operands in;
            for (std::uint32_t slot = 0; slot < arity; ++slot) {
                const auto source = pool[random::uniform(rng, 0, pool.size() - 1)];
                bool inverted = random::chance(rng, 0.3);
                // Outside the XOR family a tensor keeps one polarity within an op (spec §3.1).
                if (!xor_family) {
                    if (std::ranges::find(in.inputs, source) != in.inputs.end()) {
                        inverted = false;
                    } else if (std::ranges::find(in.inverted, source) != in.inverted.end()) {
                        inverted = true;
                    }
                }
                (inverted ? in.inverted : in.inputs).push_back(source);
            }
            std::optional<std::uint32_t> k;
            if (opcode == OpCode::BITWISE_ATLEAST || opcode == OpCode::BITWISE_ATMOST ||
                opcode == OpCode::BITWISE_EXACTLY) {
                k = static_cast<std::uint32_t>(random::uniform(rng, 0, arity));
            }
            const auto output = b.op(opcode, in, k);
            pool.push_back(output);
            if (random::chance(rng, 0.5)) {
                b.query(output);
            }
        }
        INFO("round " << round);
        require_matches_exact(b.finish());
    }
}

TEST_CASE("fzb run evaluates k-of-n gates in every cardinality kernel form", "[runtime][mc][pcfb][cardinality]") {
    // Repeated operands give gates of up to 331 inputs over three events, which exact enumeration still covers:
    // up to 8 inputs take the small kernel, 9 to 255 the counter, 256 or more the wide one. Thresholds of 256
    // and more are past what an 8-bit count holds.
    circuit_builder b;
    const auto x0 = b.sample(0.3);
    const auto x1 = b.sample(0.6);
    const auto x2 = b.sample(0.45);
    const auto repeat = [](std::initializer_list<std::pair<tensor, std::size_t>> parts) {
        std::vector<tensor> out;
        for (const auto &[t, count] : parts) {
            out.insert(out.end(), count, t);
        }
        return out;
    };
    b.query(b.op(OpCode::BITWISE_ATLEAST, {.inputs = repeat({{x0, 2}, {x1, 3}, {x2, 3}})}, 5));
    b.query(b.op(OpCode::BITWISE_ATMOST, {.inputs = repeat({{x0, 10}, {x1, 10}})}, 10));
    b.query(b.op(OpCode::BITWISE_EXACTLY, {.inputs = repeat({{x0, 4}, {x1, 5}})}, 5));
    b.query(b.op(OpCode::BITWISE_ATLEAST, {.inputs = repeat({{x0, 150}, {x1, 150}})}, 200));
    b.query(b.op(OpCode::BITWISE_ATLEAST, {.inputs = repeat({{x0, 100}, {x1, 200}})}, 300));
    b.query(
        b.op(OpCode::BITWISE_EXACTLY, {.inputs = repeat({{x0, 128}, {x1, 200}}), .inverted = repeat({{x2, 3}})}, 131));
    b.query(b.op(OpCode::BITWISE_ATMOST, {.inputs = repeat({{x0, 300}}), .inverted = repeat({{x1, 20}})}, 280));
    require_matches_exact(b.finish());
}
