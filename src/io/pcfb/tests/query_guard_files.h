#pragma once

#include <catch2/catch_test_macros.hpp>

#include "pcfb.h"
#include "pcfb/builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>

// pcfb files for the query-kind guard (#217): one AND gate over two basic events, queried as
// PROBABILITY, as MONOTONE_HULL_PROBABILITY, or both.
namespace query_guard_files {

inline constexpr char boolean_name[] = "TOP";
inline constexpr char hull_name[] = "TOP_HULL";

[[nodiscard]] inline auto build(bool boolean_query, bool hull_query) -> std::vector<std::byte> {
    pcfb::builder b;
    auto a = b.add_sample(0.25, "A", 1);
    auto c = b.add_sample(0.5, "B", 2);
    REQUIRE(a);
    REQUIRE(c);
    const std::array<std::uint32_t, 2> inputs{*a, *c};
    auto gate = b.add_op(pcfb::OpCode::BITWISE_AND, inputs, {}, std::nullopt, "G");
    REQUIRE(gate);
    if (boolean_query) {
        REQUIRE(b.add_query(*gate, pcfb::QueryKind::PROBABILITY, boolean_name));
    }
    if (hull_query) {
        REQUIRE(b.add_query(*gate, pcfb::QueryKind::MONOTONE_HULL_PROBABILITY, hull_name));
    }
    auto bytes = b.finish();
    REQUIRE(bytes);
    return std::move(*bytes);
}

[[nodiscard]] inline auto boolean_only() -> std::vector<std::byte> { return build(true, false); }
[[nodiscard]] inline auto hull_only() -> std::vector<std::byte> { return build(false, true); }
// PROBABILITY first, so a reader that only looks at query 0 would not notice the hull query.
[[nodiscard]] inline auto mixed() -> std::vector<std::byte> { return build(true, true); }

// `source` after `edit` ran on its object-API form; the builder itself only writes valid combinations.
[[nodiscard]] inline auto repacked(std::vector<std::byte> const &source, std::function<void(pcfb::CircuitT &)> const &edit) -> std::vector<std::byte> {
    const std::unique_ptr<pcfb::CircuitT> model(pcfb::GetCircuit(source.data())->UnPack());
    edit(*model);
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(pcfb::Circuit::Pack(fbb, model.get()), pcfb::CircuitIdentifier());
    const auto *data = reinterpret_cast<const std::byte *>(fbb.GetBufferPointer());
    return {data, data + fbb.GetSize()};
}

// A Boolean-only file that declares schema minor `minor` (the builder always writes the lowest minor).
[[nodiscard]] inline auto boolean_with_minor(std::uint16_t minor) -> std::vector<std::byte> {
    return repacked(boolean_only(), [&](pcfb::CircuitT &model) { model.schema_version = (1U << 16U) | minor; });
}

inline void write(std::filesystem::path const &path, std::vector<std::byte> const &bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<char const *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(output);
}

} // namespace query_guard_files
