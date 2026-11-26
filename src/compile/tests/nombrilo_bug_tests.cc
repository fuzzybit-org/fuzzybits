// Known bugs of the nombrilo.circuit (.safetensors) import and export. Each case states the docs/nombrilo.md
// rule or the pcfb semantics it checks and fails on develop until its issue is fixed; the malformed files
// are in input/bugs/nombrilo/. Cases tagged [crash] terminate the process today and run as their own ctest
// entry. Labelled `bug`.
//
// Semantics used as the oracle: pcfb core-1.1 allows ATLEAST / EXACTLY with 0 <= k <= n
// (pcfb::validate), and compile::exact_probability evaluates a file exhaustively.

#include "compile/exact_eval.h"
#include "compile/nombrilo.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace compile = fuzzybits::compile;
using compile::NombriloCircuit;
using compile::NombriloCompileOptions;
using compile::NombriloExportOptions;

std::uint32_t lit(std::uint32_t var, bool pos = true) { return (var << 1) | (pos ? 1u : 0u); }

NombriloCircuit make_circuit(std::uint32_t inputs, std::vector<std::uint32_t> ops,
                             std::vector<std::vector<std::uint32_t>> kids,
                             std::vector<std::pair<std::string, std::uint32_t>> names, std::vector<double> weights) {
    NombriloCircuit c;
    c.base_count = inputs;
    c.operators = std::move(ops);
    c.offsets.push_back(0);
    for (const auto &k : kids) {
        c.children.insert(c.children.end(), k.begin(), k.end());
        c.offsets.push_back(static_cast<std::uint32_t>(c.children.size()));
    }
    for (auto &[n, l] : names) {
        c.names.push_back(std::move(n));
        c.name_lits.push_back(l);
    }
    c.weights = std::move(weights);
    c.metadata["format"] = compile::kNombriloFormat;
    c.metadata["version"] = compile::kNombriloVersion;
    c.metadata["base_count"] = std::to_string(inputs);
    return c;
}

std::vector<std::byte> to_pcfb(const NombriloCircuit &c, NombriloCompileOptions options = {}) {
    std::array<std::byte, 32> sha{};
    auto compiled = compile::nombrilo_to_pcfb(c, options, sha, "fixture");
    REQUIRE(compiled);
    return std::move(compiled->bytes);
}

double exact(std::vector<std::byte> const &bytes) {
    auto circuit = pcfb::io::from_bytes(bytes);
    REQUIRE(circuit);
    REQUIRE(pcfb::validate(*circuit).empty());
    return compile::exact_probability(*circuit);
}

// pcfb -> nombrilo (with extras, the `fzb export` default) -> pcfb, as `fzb export` then `fzb compile` do.
std::vector<std::byte> round_trip(std::vector<std::byte> const &bytes, NombriloCompileOptions options = {}) {
    auto circuit = pcfb::io::from_bytes(bytes);
    REQUIRE(circuit);
    auto exported = compile::pcfb_to_nombrilo(*circuit, NombriloExportOptions{true});
    REQUIRE(exported);
    auto written = compile::write_nombrilo(*exported);
    REQUIRE(written);
    auto read = compile::read_nombrilo(*written);
    REQUIRE(read);
    return to_pcfb(*read, std::move(options));
}

std::vector<std::byte> finish(pcfb::builder &b) {
    auto bytes = b.finish();
    REQUIRE(bytes);
    return std::move(*bytes);
}

std::vector<std::byte> read_bytes(std::filesystem::path const &path) {
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input);
    const std::vector<char> chars{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    std::vector<std::byte> out(chars.size());
    std::memcpy(out.data(), chars.data(), chars.size());
    return out;
}

std::filesystem::path bug_fixture(std::string_view name) {
    return std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "bugs" / "nombrilo" / name;
}

// A safetensors file: little-endian u64 header length, the header JSON, the data.
std::vector<std::byte> safetensors(std::string const &header, std::size_t data_bytes = 0) {
    std::vector<std::byte> out(8 + header.size() + data_bytes, std::byte{0});
    std::uint64_t size = header.size();
    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<std::byte>((size >> (8 * i)) & 0xFF);
    }
    std::memcpy(out.data() + 8, header.data(), header.size());
    return out;
}

} // namespace

// pcfb ATLEAST k and EXACTLY k allow k = 0 (count >= 0 is TRUE; exactly 0 is NOR). The export writes an
// at-least gate as nombrilo operator code k, and code 0 is kNombriloAnd, so ATLEAST 0 comes back as AND and
// EXACTLY 0 as AND(kids) AND NOT OR(kids), which is FALSE. Both the general and the 1:1 export do this.
TEST_CASE("pcfb ATLEAST 0 and EXACTLY 0 survive export to nombrilo and back", "[bug][nombrilo][issue-291]") {
    for (const bool one_to_one : {false, true}) {
        for (const auto opcode : {pcfb::OpCode::BITWISE_ATLEAST, pcfb::OpCode::BITWISE_EXACTLY}) {
            INFO((one_to_one ? "1:1 layout" : "general layout")
                 << ", " << (opcode == pcfb::OpCode::BITWISE_ATLEAST ? "ATLEAST 0" : "EXACTLY 0"));
            pcfb::builder b;
            if (one_to_one) {
                REQUIRE(b.add_op(pcfb::OpCode::BITWISE_ONES, {}));
            }
            auto a = b.add_sample(0.5, "a", 1);
            auto c = b.add_sample(0.25, "c", 2);
            REQUIRE(a);
            REQUIRE(c);
            const std::array<std::uint32_t, 2> inputs{*a, *c};
            auto gate = b.add_op(opcode, inputs, {}, 0U, "g");
            REQUIRE(gate);
            REQUIRE(b.add_query(*gate, pcfb::QueryKind::PROBABILITY, "g"));
            const auto bytes = finish(b);
            // ATLEAST 0: TRUE. EXACTLY 0: neither input, 0.5 * 0.75.
            const double expected = opcode == pcfb::OpCode::BITWISE_ATLEAST ? 1.0 : 0.375;
            REQUIRE(exact(bytes) == Catch::Approx(expected).margin(1e-15));
            CHECK(exact(round_trip(bytes)) == Catch::Approx(expected).margin(1e-15));
        }
    }
}

// docs/nombrilo.md (Extras): export with extras writes `fuzzybits.pcfb.queries`, and "our own fzb compile
// output can round-trip pcfb -> nombrilo -> pcfb". A file in the 1:1 layout takes the 1:1 export
// (try_derive_1to1), which copies only `fuzzybits.pcfb.*` keys already present in the file's metadata, and
// import consumes those, so the queries are never written: `fzb compile m.safetensors --target nx` then
// `fzb export --to nombrilo` then `fzb compile` gives the default query (every root gate) instead of nx.
TEST_CASE("a 1:1 nombrilo-compiled file keeps its query through export with extras", "[bug][nombrilo][issue-291]") {
    // One input x (p = 0.25), gate g0 = OR(x); the target is the name on the negated literal of x.
    const auto c = make_circuit(1, {compile::kNombriloOr}, {{lit(1)}}, {{"x", lit(1)}, {"nx", lit(1, false)}}, {0.25});
    NombriloCompileOptions options;
    options.target = "nx";
    const auto bytes = to_pcfb(c, options);
    REQUIRE(exact(bytes) == Catch::Approx(0.75).margin(1e-15));

    auto circuit = pcfb::io::from_bytes(bytes);
    REQUIRE(circuit);
    auto exported = compile::pcfb_to_nombrilo(*circuit, NombriloExportOptions{true});
    REQUIRE(exported);
    CHECK(exported->metadata.contains("fuzzybits.pcfb.queries"));
    CHECK(exact(round_trip(bytes)) == Catch::Approx(0.75).margin(1e-15));
}

// docs/nombrilo.md (CLI, Python): `--target` / `target=` names the query of the compiled file. When the
// nombrilo file carries `fuzzybits.pcfb.queries` (every `fzb export` with extras of a non-1:1 file), import
// restores those queries and ignores the requested target, while still reporting it as the target.
TEST_CASE("--target overrides the queries restored from fuzzybits.pcfb.queries", "[bug][nombrilo][issue-291]") {
    pcfb::builder b;
    auto x = b.add_sample(0.5, "x", 1);
    auto y = b.add_sample(0.25, "y", 2);
    REQUIRE(x);
    REQUIRE(y);
    const std::array<std::uint32_t, 2> inputs{*x, *y};
    auto both = b.add_op(pcfb::OpCode::BITWISE_AND, inputs, {}, std::nullopt, "g_and");
    auto either = b.add_op(pcfb::OpCode::BITWISE_OR, inputs, {}, std::nullopt, "g_or");
    REQUIRE(both);
    REQUIRE(either);
    REQUIRE(b.add_query(*both, pcfb::QueryKind::PROBABILITY, "g_and"));
    const auto bytes = finish(b);
    REQUIRE(exact(bytes) == Catch::Approx(0.125).margin(1e-15));

    NombriloCompileOptions options;
    options.target = "g_or";
    // P(x or y) = 1 - 0.5 * 0.75.
    CHECK(exact(round_trip(bytes, options)) == Catch::Approx(0.625).margin(1e-15));
}

// The header JSON is skipped by mutual recursion (skip_json_value -> skip_json_array -> skip_json_value)
// with no depth limit. 10^5 nested arrays (200 KB, far below the 16 MiB header limit) under an unknown key
// of a tensor's info overflow the stack. `fzb compile` sniffs every non-FTP input this way, so any file can
// crash it.
TEST_CASE("a nombrilo header with 10^5 nested arrays is rejected, not a stack overflow",
          "[bug][nombrilo][crash][issue-292]") {
    constexpr std::size_t depth = 100'000;
    std::string header = R"({"operators":{"dtype":"U32","shape":[0],"data_offsets":[0,0],"x":)";
    header.append(depth, '[');
    header.append(depth, ']');
    header += "}}";
    const auto bytes = safetensors(header);
    CHECK_FALSE(compile::read_nombrilo(bytes));
}

// Structurally invalid safetensors files are accepted (the safetensors format requires unique keys and
// tensors that tile the data buffer without gaps or overlaps, and the header must be one JSON object):
//  - dup_weights: two `weights` entries (0.9 and 0.1); the first one is used silently;
//  - overlap: `children` aliases the second word of `offsets` (the value 1, literal TRUE), with a gap and
//    trailing bytes; read as a circuit whose root is TRUE;
//  - trailing_header: junk after the header object, and the dtype `u32` in lower case.
TEST_CASE("structurally invalid safetensors files are rejected", "[bug][nombrilo][issue-297]") {
    for (const auto *name : {"dup_weights.safetensors", "overlap.safetensors", "trailing_header.safetensors"}) {
        INFO(name);
        const auto bytes = read_bytes(bug_fixture(name));
        CHECK_FALSE(compile::read_nombrilo(bytes));
    }
}
