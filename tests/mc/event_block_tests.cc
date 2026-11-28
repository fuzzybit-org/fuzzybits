// Tests of the event layer (src/mc/event): USM ownership, alignment and byte-count overflow (usm.h), device_span
// (span.h), the output arena and the node blocks that take its slots (node.h), gate blocks, their input tables of arena
// slots and the bounds a cardinality gate is added with (gate.h), tally records and blocks (tally.h), the Bernoulli law
// and its compile-time refusals (basic_event.h), and the concepts that hold the engine's integer types to what the
// kernels can represent. One device round trip writes through a block's view into the arena; the kernels that read
// these blocks are tested in gate_kernel_tests.cc, cardinality_gate_tests.cc, tally_kernel_tests.cc and
// bitpack_identity_tests.cc.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "mc/event/basic_event.h"
#include "mc/event/gate.h"
#include "mc/event/node.h"
#include "mc/event/span.h"
#include "mc/event/tally.h"
#include "mc/event/usm.h"

#include "pcfb/threshold.h"

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace ev = fuzzybits::mc::event;

/// The engine's types: 64-bit bitpacks, 32-bit PDAG indices and extents.
using types = ev::node_types<std::uint64_t, std::int32_t, std::uint32_t>;
/// 16-bit indices and extents: at most 65,535 slots of at most 65,535 bitpacks, and 65,535 nodes or inputs a block.
using narrow = ev::node_types<std::uint64_t, std::int16_t, std::uint16_t>;

/// Arena slots, as a gate builder takes a gate's inputs.
using slot_list = std::vector<std::uint32_t>;

std::vector<ev::basic_event<types>> some_basic_events() {
    return {
        {.id_in_pdag = 7, .law = ev::bernoulli_law::never()},
        {.id_in_pdag = 3, .law = ev::bernoulli_law::always()},
        {.id_in_pdag = 11, .law = *ev::bernoulli_law::from_probability(0.25)},
    };
}

// Outside a template, a requires-expression whose expression is ill-formed is an error rather than false, so the
// refusals are tested through concepts.

template <class P>
concept takes_probability = requires(const P p) { ev::bernoulli_law::from_probability(p); };
template <class U>
concept takes_threshold = requires(const U u) { ev::bernoulli_law::from_threshold(u); };
/// A record initialized from its PDAG index alone, every other field from {}.
template <class R>
concept made_from_id_alone = requires { R{.id_in_pdag = 1}; };
/// A gate added from its inputs alone, without bounds.
template <class Builder>
concept adds_without_bounds = requires(Builder &builder, const std::vector<typename Builder::extent_type> &slots) { builder.add(1, slots, slots); };
/// A gate added from its inputs and its bounds.
template <class Builder>
concept adds_with_bounds = requires(Builder &builder, const std::vector<typename Builder::extent_type> &slots,
                                    const ev::cardinality_bounds<typename Builder::extent_type> bounds) { builder.add(1, slots, slots, bounds); };

template <class B>
concept engine_bitpack = requires { typename ev::node_types<B, std::int32_t, std::uint32_t>; };
template <class I>
concept engine_index = requires { typename ev::node_types<std::uint64_t, I, std::uint32_t>; };
template <class E>
concept engine_extent = requires { typename ev::node_types<std::uint64_t, std::int32_t, E>; };

/// No copies, and moves that cannot throw: what an owner of USM is.
template <class X>
concept move_only =
    !std::is_copy_constructible_v<X> && !std::is_copy_assignable_v<X> && std::is_nothrow_move_constructible_v<X> && std::is_nothrow_move_assignable_v<X>;

#if defined(__SIZEOF_INT128__)
/// node_types' member types written out by hand, with a bitpack of 128 bits that node_types itself refuses.
struct wide_types {
    using bitpack_type = unsigned __int128;
    using index_type = std::int32_t;
    using extent_type = std::uint32_t;
};
#endif

/// A tally's fields as an aggregate, which event::tally is not.
struct tally_fields {
    std::int32_t id_in_pdag;
    std::uint32_t slot;
    std::uint64_t ones;
};

/// Trivially copyable, and aligned to 16: more than any bitpack word.
struct alignas(16) aligned_record {
    std::uint64_t low;
    std::uint64_t high;
};

} // namespace

TEST_CASE("usm_array owns one allocation of its kind", "[mc][event]") {
    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());

    const std::size_t line = ev::cache_line_alignment<std::uint64_t>(queue.get_device());
    CHECK(std::has_single_bit(line));
    CHECK(line >= alignof(std::uint64_t));

    // No objects, no allocation, of either kind: a zero-size shared allocation fails on HIP (#277). With nothing to
    // dereference, host() of empty device memory is an empty span rather than a refusal.
    for (const auto kind : {sycl::usm::alloc::shared, sycl::usm::alloc::device}) {
        INFO("kind=" << static_cast<int>(kind));
        const auto none = ev::usm_array<std::uint64_t>::allocate(queue, 0, kind);
        CHECK(none.data() == nullptr);
        CHECK(none.size() == 0);
        CHECK(none.empty());
        CHECK(none.kind() == kind);
        CHECK(none.host().empty());
    }

    // Shared memory: the host writes it through host(), whose elements are data()'s.
    auto shared = ev::usm_array<std::uint64_t>::allocate(queue, 5, sycl::usm::alloc::shared);
    REQUIRE(shared.data() != nullptr);
    CHECK(shared.size() == 5);
    CHECK_FALSE(shared.empty());
    CHECK(shared.kind() == sycl::usm::alloc::shared);
    const std::span<std::uint64_t> host = shared.host();
    CHECK(host.data() == shared.data());
    CHECK(host.size() == 5);
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = 10 * i + 1;
    }
    for (std::size_t i = 0; i < host.size(); ++i) {
        CHECK(shared.data()[i] == 10 * i + 1);
    }

    // Device memory: allocated, and refused a host view.
    const auto device = ev::usm_array<std::uint64_t>::allocate(queue, 5, sycl::usm::alloc::device);
    CHECK(device.data() != nullptr);
    CHECK(device.size() == 5);
    CHECK(device.kind() == sycl::usm::alloc::device);
    CHECK_THROWS_AS(device.host(), std::logic_error);

    // copy_of: the values in order, into memory the host can write; device memory is refused, and no values make no
    // allocation.
    const std::vector<std::int32_t> values{4, -1, 7};
    const auto copy = ev::usm_array<std::int32_t>::copy_of(queue, values, sycl::usm::alloc::shared);
    CHECK(copy.kind() == sycl::usm::alloc::shared);
    CHECK(std::ranges::equal(copy.host(), values));
    CHECK_THROWS_AS(ev::usm_array<std::int32_t>::copy_of(queue, values, sycl::usm::alloc::device), std::invalid_argument);
    const auto no_copy = ev::usm_array<std::int32_t>::copy_of(queue, std::vector<std::int32_t>{}, sycl::usm::alloc::shared);
    CHECK(no_copy.data() == nullptr);
    CHECK(no_copy.empty());

    // Moved from, by construction and by assignment (which frees the target's own allocation first): empty.
    std::uint64_t *const data = shared.data();
    auto moved = std::move(shared);
    CHECK(moved.data() == data);
    CHECK(moved.size() == 5);
    CHECK(moved.kind() == sycl::usm::alloc::shared);
    auto assigned = ev::usm_array<std::uint64_t>::allocate(queue, 2, sycl::usm::alloc::shared);
    assigned = std::move(moved);
    CHECK(assigned.data() == data);
    CHECK(assigned.size() == 5);
    // NOLINTBEGIN(bugprone-use-after-move)
    CHECK(shared.data() == nullptr);
    CHECK(shared.empty());
    CHECK(shared.host().empty());
    CHECK(moved.data() == nullptr);
    CHECK(moved.empty());
    // NOLINTEND(bugprone-use-after-move)
}

TEST_CASE("usm_array refuses a count whose bytes overflow size_t", "[mc][event]") {
    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());

    // n·8 bytes wrap: SIZE_MAX / 8 + 1 words to 0 bytes, one more word to 8. The count is refused before anything is
    // allocated, as std::bad_array_new_length. Allocating the wrapped size would instead give AdaptiveCpp's null for 0
    // bytes (a plain std::bad_alloc) or a valid 8 bytes (no exception).
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
    constexpr std::size_t kWrapsToZero = kMax / sizeof(std::uint64_t) + 1;
    STATIC_CHECK(kWrapsToZero * sizeof(std::uint64_t) == 0);
    STATIC_CHECK((kWrapsToZero + 1) * sizeof(std::uint64_t) == sizeof(std::uint64_t));
    for (const auto kind : {sycl::usm::alloc::device, sycl::usm::alloc::shared, sycl::usm::alloc::host}) {
        INFO("kind=" << static_cast<int>(kind));
        for (const std::size_t n : {kWrapsToZero, kWrapsToZero + 1, kMax}) {
            INFO("n=" << n);
            CHECK_THROWS_AS(ev::usm_array<std::uint64_t>::allocate(queue, n, kind), std::bad_array_new_length);
        }
    }
}

TEST_CASE("usm_array aligns its objects in memory of every kind", "[mc][event]") {
    STATIC_CHECK(std::is_trivially_copyable_v<aligned_record>);
    STATIC_CHECK(alignof(aligned_record) == 16);

    sycl::queue queue;
    const sycl::device device = queue.get_device();
    INFO("device=" << device.get_info<sycl::info::device::name>());

    std::vector<sycl::usm::alloc> kinds{sycl::usm::alloc::device, sycl::usm::alloc::shared};
    if (device.has(sycl::aspect::usm_host_allocations)) {
        kinds.push_back(sycl::usm::alloc::host);
    } else {
        WARN("the device has no host USM: host memory is not checked");
    }
    for (const auto kind : kinds) {
        INFO("kind=" << static_cast<int>(kind));
        // Allocations of different sizes, live at once, so that one aligned by chance does not pass alone.
        std::vector<ev::usm_array<aligned_record>> live;
        for (const std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{5}, std::size_t{17}}) {
            INFO("n=" << n);
            const auto &objects = live.emplace_back(ev::usm_array<aligned_record>::allocate(queue, n, kind));
            REQUIRE(objects.data() != nullptr);
            CHECK(objects.size() == n);
            CHECK(objects.kind() == kind);
            CHECK(reinterpret_cast<std::uintptr_t>(objects.data()) % alignof(aligned_record) == 0);
        }
    }
}

TEST_CASE("device_span is a pointer and a count", "[mc][event]") {
    using span_t = ev::device_span<int, std::uint32_t>;
    STATIC_CHECK(std::is_trivially_copyable_v<span_t>);
    STATIC_CHECK(sizeof(span_t) <= 2 * sizeof(int *));
    STATIC_CHECK(std::same_as<span_t::size_type, std::uint32_t>);
    STATIC_CHECK(span_t{}.empty());

    std::array<int, 6> values{0, 1, 2, 3, 4, 5};
    const span_t all(values.data(), 6);
    CHECK(all.data() == values.data());
    CHECK(all.size() == 6);
    CHECK_FALSE(all.empty());
    CHECK(all.begin() == values.data());
    CHECK(all.end() == values.data() + 6);
    // A const span of mutable elements writes them.
    all[2] = 20;
    CHECK(values[2] == 20);
    int sum = 0;
    for (const int x : all) {
        sum += x;
    }
    CHECK(sum == 0 + 1 + 20 + 3 + 4 + 5);

    const auto middle = all.subspan(1, 3);
    CHECK(middle.data() == values.data() + 1);
    CHECK(middle.size() == 3);
    CHECK(middle[1] == 20);
    const auto tail = all.subspan(6, 0);
    CHECK(tail.empty());
    CHECK(tail.data() == values.data() + 6);

    const auto read_only = all.as_const();
    STATIC_CHECK(std::same_as<decltype(read_only), const ev::device_span<const int, std::uint32_t>>);
    CHECK(read_only.data() == values.data());
    CHECK(read_only.size() == 6);

    const std::span<int> standard = all.as_std();
    CHECK(standard.data() == values.data());
    CHECK(standard.size() == 6);

    const span_t none;
    CHECK(none.data() == nullptr);
    CHECK(none.size() == 0);
    CHECK(none.empty());
}

TEST_CASE("output_arena refuses a shape its extent type cannot index", "[mc][event]") {
    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());

    CHECK_THROWS_AS(ev::output_arena<types>::create(queue, 4, 0), std::invalid_argument);
    CHECK_THROWS_AS(ev::output_arena<types>::create(queue, std::size_t{std::numeric_limits<std::uint32_t>::max()} + 1, 1), std::length_error);
    // 16-bit extents: at most 65,535 slots of at most 65,535 bitpacks.
    CHECK_THROWS_AS(ev::output_arena<narrow>::create(queue, 70000, 1), std::length_error);
    CHECK_THROWS_AS(ev::output_arena<narrow>::create(queue, 65536, 1), std::length_error);
    CHECK_THROWS_AS(ev::output_arena<narrow>::create(queue, 1, 70000), std::length_error);
    const auto largest = ev::output_arena<narrow>::create(queue, 65535, 1);
    CHECK(largest.slots() == 65535);
    CHECK(largest.view().size() == 65535);
}

TEST_CASE("output_arena hands out consecutive slots of B bitpacks", "[mc][event]") {
    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());
    constexpr std::uint32_t kBitpacks = 3;

    auto arena = ev::output_arena<types>::create(queue, 10, kBitpacks);
    CHECK(arena.slots() == 10);
    CHECK(arena.bitpacks_per_node() == kBitpacks);
    CHECK(arena.reserved() == 0);

    const auto head = arena.reserve(3);
    CHECK(head.first == 0);
    CHECK(head.count == 3);
    const auto nothing = arena.reserve(0);
    CHECK(nothing.first == 3);
    CHECK(nothing.count == 0);
    const auto middle = arena.reserve(4);
    CHECK(middle.first == 3);
    CHECK(middle.count == 4);
    CHECK(arena.reserved() == 7);

    // Over-reserving is refused and leaves the arena as it was.
    CHECK_THROWS_AS(arena.reserve(4), std::length_error);
    CHECK(arena.reserved() == 7);
    const auto tail = arena.reserve(3);
    CHECK(tail.first == 7);
    CHECK(tail.count == 3);
    CHECK(arena.reserved() == 10);
    CHECK_THROWS_AS(arena.reserve(1), std::length_error);
    CHECK(arena.reserve(0).first == 10);

    // Slot s holds bitpacks [s·B, (s + 1)·B).
    const auto view = arena.view();
    REQUIRE(view.data() != nullptr);
    CHECK(view.slots() == 10);
    CHECK(view.bitpacks_per_node() == kBitpacks);
    CHECK(view.size() == 10 * kBitpacks);
    for (std::uint32_t s = 0; s < view.slots(); ++s) {
        INFO("slot=" << s);
        CHECK(view.at(s) == view.data() + std::size_t{s} * kBitpacks);
        CHECK(view.output(s).data() == view.at(s));
        CHECK(view.output(s).size() == kBitpacks);
    }
    // size() is 64 bits: slots × bitpacks per node can exceed 2^32.
    STATIC_CHECK(ev::arena_view<types>{nullptr, 1u << 20, 1u << 13}.size() == std::size_t{1} << 33);

    // [first, first + count), up to the top of the extent type.
    constexpr ev::slot_range<std::uint32_t> range{.first = 3, .count = 4};
    STATIC_CHECK(range.end() == 7);
    STATIC_CHECK_FALSE(range.contains(2));
    STATIC_CHECK(range.contains(3));
    STATIC_CHECK(range.contains(6));
    STATIC_CHECK_FALSE(range.contains(7));
    STATIC_CHECK_FALSE(ev::slot_range<std::uint32_t>{.first = 5, .count = 0}.contains(5));
    constexpr ev::slot_range<std::uint16_t> top{.first = 65530, .count = 5};
    STATIC_CHECK(top.end() == 65535);
    STATIC_CHECK(top.contains(65534));
    STATIC_CHECK_FALSE(top.contains(65535));

    // An arena of no slots allocates nothing and hands out nothing.
    auto empty = ev::output_arena<types>::create(queue, 0, kBitpacks);
    CHECK(empty.view().data() == nullptr);
    CHECK(empty.view().size() == 0);
    CHECK(empty.reserve(0).count == 0);
    CHECK_THROWS_AS(empty.reserve(1), std::length_error);

    // Moved from, by construction and by assignment: empty, with no slot to hand out.
    auto moved = std::move(arena);
    CHECK(moved.view().data() == view.data());
    CHECK(moved.slots() == 10);
    CHECK(moved.bitpacks_per_node() == kBitpacks);
    CHECK(moved.reserved() == 10);
    ev::output_arena<types> assigned;
    assigned = std::move(moved);
    CHECK(assigned.view().data() == view.data());
    CHECK(assigned.reserved() == 10);
    // NOLINTBEGIN(bugprone-use-after-move)
    for (auto *from : {&arena, &moved}) {
        CHECK(from->slots() == 0);
        CHECK(from->bitpacks_per_node() == 0);
        CHECK(from->reserved() == 0);
        CHECK(from->view().data() == nullptr);
        CHECK(from->view().size() == 0);
        CHECK_THROWS_AS(from->reserve(1), std::length_error);
    }
    // NOLINTEND(bugprone-use-after-move)
}

TEST_CASE("node_block takes the arena's next slots, in record order", "[mc][event]") {
    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());
    constexpr std::uint32_t kBitpacks = 3;

    auto arena = ev::output_arena<types>::create(queue, 8, kBitpacks);
    const auto earlier = arena.reserve(2);
    const auto records = some_basic_events();
    auto block = ev::basic_event_block<types>::create(queue, arena, records);
    CHECK(arena.reserved() == 5);
    CHECK(block.size() == 3);
    CHECK_FALSE(block.empty());
    CHECK(block.slots().first == earlier.end());
    CHECK(block.slots().count == 3);
    REQUIRE(block.records().size() == records.size());
    for (std::uint32_t i = 0; i < block.size(); ++i) {
        INFO("node=" << i);
        CHECK(block.slot(i) == 2 + i);
        CHECK(block.records()[i].id_in_pdag == records[i].id_in_pdag);
        CHECK(block.records()[i].law == records[i].law);
    }

    // nodes(): each node's position, slot and record. Its output is device memory, reached from the slot:
    // arena_view::at(slot) is the slot's first bitpack, where the block's view writes the node.
    const auto arena_view = arena.view();
    std::uint32_t position = 0;
    for (const auto &node : block.nodes()) {
        INFO("position=" << position);
        CHECK(node.position == position);
        CHECK(node.slot == 2 + position);
        CHECK(node.slot == block.slot(position));
        CHECK(&node.record == &block.records()[position]);
        CHECK(node.record.id_in_pdag == records[position].id_in_pdag);
        CHECK(node.record.law == records[position].law);
        CHECK(arena_view.at(node.slot) == arena_view.data() + std::size_t{node.slot} * kBitpacks);
        CHECK(block.view().output(node.position).data() == arena_view.at(node.slot));
        ++position;
    }
    CHECK(position == 3);
    CHECK(block.nodes().size() == 3);

    // The view, as a kernel holds it.
    const auto view = block.view();
    CHECK(view.size() == 3);
    CHECK_FALSE(view.empty());
    CHECK(view.bitpacks_per_node() == kBitpacks);
    CHECK(view.slots().first == 2);
    CHECK(view.slots().count == 3);
    CHECK(view.arena().data() == arena_view.data());
    CHECK(view.arena().slots() == 8);
    for (std::uint32_t i = 0; i < view.size(); ++i) {
        INFO("node=" << i);
        CHECK(&view[i] == &block.records()[i]);
        CHECK(view.slot(i) == block.slot(i));
        CHECK(view.output(i).data() == arena_view.at(2 + i));
        CHECK(view.output(i).size() == kBitpacks);
    }

    // The next block starts where this one ends, and so does the next reservation.
    const auto next = ev::basic_event_block<types>::create(queue, arena, std::vector{records[0]});
    CHECK(next.slots().first == 5);
    CHECK(next.slots().count == 1);
    CHECK(arena.reserve(1).first == 6);

    // A block of no nodes holds no slot.
    const auto empty = ev::basic_event_block<types>::create(queue, arena, std::vector<ev::basic_event<types>>{});
    CHECK(empty.empty());
    CHECK(empty.size() == 0);
    CHECK(empty.slots().first == 7);
    CHECK(empty.slots().count == 0);
    CHECK(empty.records().empty());
    CHECK(empty.nodes().empty());
    CHECK(empty.view().empty());
    CHECK(arena.reserved() == 7);

    // Moved from, by construction and by assignment: empty.
    const auto *const first_record = block.records().data();
    auto moved = std::move(block);
    CHECK(moved.size() == 3);
    CHECK(moved.slots().first == 2);
    CHECK(moved.records().data() == first_record);
    ev::basic_event_block<types> assigned;
    assigned = std::move(moved);
    CHECK(assigned.size() == 3);
    CHECK(assigned.records().data() == first_record);
    CHECK(assigned.view().output(0).data() == arena_view.at(2));
    // NOLINTBEGIN(bugprone-use-after-move)
    for (const auto *from : {&block, &moved}) {
        CHECK(from->empty());
        CHECK(from->size() == 0);
        CHECK(from->slots().count == 0);
        CHECK(from->records().empty());
        CHECK(from->nodes().empty());
        CHECK(from->view().empty());
    }
    // NOLINTEND(bugprone-use-after-move)
}

TEST_CASE("input_block_builder lays out records and an input table of arena slots", "[mc][event]") {
    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());

    auto arena = ev::output_arena<types>::create(queue, 16, 2);
    (void)arena.reserve(6);

    ev::gate_block_builder<types> builder;
    builder.reserve(4, 9);
    CHECK(builder.empty());
    builder.add(10, slot_list{0, 1, 2}, slot_list{});
    builder.add(11, slot_list{}, slot_list{3, 4});
    // No inputs: a constant (AND over none is TRUE, OR over none FALSE).
    builder.add(12, slot_list{}, slot_list{});
    // Slots read by other gates too, and twice by this one.
    builder.add(13, slot_list{5, 0}, slot_list{1, 0});
    CHECK(builder.size() == 4);
    CHECK_FALSE(builder.empty());
    CHECK(builder.input_count() == 9);

    auto block = builder.build(queue, arena);
    CHECK(arena.reserved() == 10);
    REQUIRE(block.size() == 4);
    CHECK(block.slots().first == 6);
    // Each gate's inputs, positive then negated, one gate after another.
    CHECK(std::ranges::equal(block.inputs(), slot_list{0, 1, 2, 3, 4, 5, 0, 1, 0}));

    struct expected_gate {
        std::int32_t id_in_pdag;
        std::uint32_t first_input;
        std::uint32_t num_inputs;
        std::uint32_t negated_from;
        slot_list inputs;
    };
    const std::array<expected_gate, 4> expected{{
        {10, 0, 3, 3, {0, 1, 2}},
        {11, 3, 2, 0, {3, 4}},
        {12, 5, 0, 0, {}},
        {13, 5, 4, 2, {5, 0, 1, 0}},
    }};
    const auto view = block.view();
    const auto arena_view = arena.view();
    for (std::uint32_t i = 0; i < block.size(); ++i) {
        INFO("gate=" << i);
        const auto &g = block.records()[i];
        CHECK(g.id_in_pdag == expected[i].id_in_pdag);
        CHECK(g.first_input == expected[i].first_input);
        CHECK(g.num_inputs == expected[i].num_inputs);
        CHECK(g.negated_from == expected[i].negated_from);
        CHECK(std::ranges::equal(block.inputs_of(i), expected[i].inputs));
        CHECK(&view[i] == &g);
        CHECK(view.output(i).data() == arena_view.at(6 + i));
        for (std::uint32_t j = 0; j < g.num_inputs; ++j) {
            INFO("input=" << j);
            CHECK(view.input_slot(g, j) == expected[i].inputs[j]);
            CHECK(view.input(g, j) == arena_view.at(expected[i].inputs[j]));
        }
    }

    // A builder of no gates builds a block of no slots and no inputs.
    const auto none = ev::gate_block_builder<types>{}.build(queue, arena);
    CHECK(none.empty());
    CHECK(none.slots().first == 10);
    CHECK(none.slots().count == 0);
    CHECK(none.inputs().empty());
    CHECK(arena.reserved() == 10);

    // Moved from, by construction and by assignment: empty, input table included.
    auto moved = std::move(block);
    CHECK(moved.size() == 4);
    CHECK(moved.inputs().size() == 9);
    ev::gate_block<types> assigned;
    assigned = std::move(moved);
    CHECK(assigned.size() == 4);
    CHECK(std::ranges::equal(assigned.inputs_of(3), expected[3].inputs));
    // NOLINTBEGIN(bugprone-use-after-move)
    for (const auto *from : {&block, &moved}) {
        CHECK(from->empty());
        CHECK(from->records().empty());
        CHECK(from->inputs().empty());
        CHECK(from->view().empty());
    }
    // NOLINTEND(bugprone-use-after-move)
}

TEST_CASE("a cardinality gate's record holds its bounds", "[mc][event]") {
    // The bounds have no default: a cardinality gate is added with them, a plain gate without.
    STATIC_CHECK_FALSE(adds_without_bounds<ev::cardinality_gate_block_builder<types>>);
    STATIC_CHECK(adds_with_bounds<ev::cardinality_gate_block_builder<types>>);
    STATIC_CHECK(adds_without_bounds<ev::gate_block_builder<types>>);
    STATIC_CHECK_FALSE(adds_with_bounds<ev::gate_block_builder<types>>);

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());

    auto arena = ev::output_arena<types>::create(queue, 12, 2);
    (void)arena.reserve(5);

    ev::cardinality_gate_block_builder<types> builder;
    // 2 or 3 of 4 inputs, the last of them negated.
    builder.add(20, slot_list{0, 1, 2}, slot_list{3}, {.low = 2, .high = 3});
    // At least 1 of 1: high >= num_inputs is no upper bound, and is stored as given.
    builder.add(21, slot_list{4}, slot_list{}, {.low = 1, .high = 7});
    // low > high: never true, and stored as given.
    builder.add(22, slot_list{0, 1}, slot_list{2}, {.low = 2, .high = 1});
    const auto block = builder.build(queue, arena);
    REQUIRE(block.size() == 3);
    CHECK(block.slots().first == 5);
    CHECK(std::ranges::equal(block.inputs(), slot_list{0, 1, 2, 3, 4, 0, 1, 2}));

    struct expected_gate {
        std::int32_t id_in_pdag;
        std::uint32_t first_input;
        std::uint32_t num_inputs;
        std::uint32_t negated_from;
        std::uint32_t low;
        std::uint32_t high;
    };
    const std::array<expected_gate, 3> expected{{
        {20, 0, 4, 3, 2, 3},
        {21, 4, 1, 1, 1, 7},
        {22, 5, 3, 2, 2, 1},
    }};
    for (std::uint32_t i = 0; i < block.size(); ++i) {
        INFO("gate=" << i);
        const auto &g = block.records()[i];
        CHECK(g.id_in_pdag == expected[i].id_in_pdag);
        CHECK(g.first_input == expected[i].first_input);
        CHECK(g.num_inputs == expected[i].num_inputs);
        CHECK(g.negated_from == expected[i].negated_from);
        CHECK(g.low == expected[i].low);
        CHECK(g.high == expected[i].high);
    }
}

TEST_CASE("input_block_builder refuses an input slot that no earlier block holds", "[mc][event]") {
    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());

    auto arena = ev::output_arena<types>::create(queue, 8, 2);
    (void)arena.reserve(3);

    // Each block below would take slots 3 and up, so slots 3 and 4 are its own outputs, which the kernel that reads
    // them would also write; slots 5 to 7 belong to no block, and slot 8 is outside the arena.
    for (const std::uint32_t slot : {3u, 4u, 5u, 7u, 8u}) {
        INFO("slot=" << slot);
        ev::gate_block_builder<types> gates;
        gates.add(1, slot_list{0, 2}, slot_list{});
        gates.add(2, slot_list{1}, slot_list{slot});
        CHECK_THROWS_AS(gates.build(queue, arena), std::out_of_range);

        ev::cardinality_gate_block_builder<types> counted;
        counted.add(3, slot_list{slot}, slot_list{0}, {.low = 1, .high = 2});
        CHECK_THROWS_AS(counted.build(queue, arena), std::out_of_range);
    }
    // A refused block reserves nothing.
    CHECK(arena.reserved() == 3);

    // The last reserved slot is an input like any other, and a built block's slots are inputs of the blocks after it.
    ev::gate_block_builder<types> gates;
    gates.add(1, slot_list{2}, slot_list{});
    const auto block = gates.build(queue, arena);
    CHECK(block.slots().first == 3);
    ev::gate_block_builder<types> next;
    next.add(2, slot_list{}, slot_list{block.slot(0)});
    const auto next_block = next.build(queue, arena);
    CHECK(next_block.slots().first == 4);
    CHECK(std::ranges::equal(next_block.inputs_of(0), slot_list{3}));
    CHECK(arena.reserved() == 5);
}

TEST_CASE("blocks refuse more nodes or inputs than their extent type counts", "[mc][event]") {
    using narrow_slots = std::vector<std::uint16_t>;

    // 65,535 inputs in a block's table at most.
    ev::gate_block_builder<narrow> wide;
    CHECK_THROWS_AS(wide.add(0, narrow_slots(65536), {}), std::length_error);
    CHECK(wide.empty());
    CHECK(wide.input_count() == 0);
    wide.add(0, narrow_slots(65535), {});
    CHECK(wide.input_count() == 65535);
    CHECK_THROWS_AS(wide.add(1, {}, narrow_slots(1)), std::length_error);
    CHECK(wide.size() == 1);
    CHECK(wide.input_count() == 65535);

    // 65,535 gates in a block at most.
    ev::gate_block_builder<narrow> many;
    for (std::uint32_t i = 0; i < 65535; ++i) {
        many.add(0, {}, {});
    }
    CHECK_THROWS_AS(many.add(0, {}, {}), std::length_error);
    CHECK(many.size() == 65535);

    // 65,535 tallies in a block at most.
    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());
    const auto arena = ev::output_arena<narrow>::create(queue, 1, 1);
    std::vector<ev::tally<narrow>> tallies(65536, ev::tally<narrow>{0, 0});
    CHECK_THROWS_AS(ev::tally_block<narrow>::create(queue, arena.view(), tallies), std::length_error);
    tallies.pop_back();
    CHECK(ev::tally_block<narrow>::create(queue, arena.view(), tallies).size() == 65535);
}

TEST_CASE("a tally is made with its slot", "[mc][event]") {
    // No aggregate form and no default; the count is 0 unless given. The same fields as an aggregate are made from the
    // id alone, so the refusal is the constructor's.
    STATIC_CHECK_FALSE(made_from_id_alone<ev::tally<types>>);
    STATIC_CHECK(made_from_id_alone<tally_fields>);
    STATIC_CHECK_FALSE(std::is_default_constructible_v<ev::tally<types>>);
    STATIC_CHECK_FALSE(std::is_constructible_v<ev::tally<types>, std::int32_t>);
    STATIC_CHECK(std::is_constructible_v<ev::tally<types>, std::int32_t, std::uint32_t>);
    STATIC_CHECK(std::is_constructible_v<ev::tally<types>, std::int32_t, std::uint32_t, std::uint64_t>);

    constexpr ev::tally<types> fresh{4, 5};
    STATIC_CHECK(fresh.id_in_pdag == 4);
    STATIC_CHECK(fresh.slot == 5);
    STATIC_CHECK(fresh.ones == 0);
    constexpr ev::tally<types> resumed{9, 0, 17};
    STATIC_CHECK(resumed.id_in_pdag == 9);
    STATIC_CHECK(resumed.slot == 0);
    STATIC_CHECK(resumed.ones == 17);
}

TEST_CASE("tally_block reads the arena at its records' slots", "[mc][event]") {
    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>());
    constexpr std::uint32_t kBitpacks = 4;

    // A tally reads a slot of the arena whether or not a block holds it yet.
    auto arena = ev::output_arena<types>::create(queue, 6, kBitpacks);
    const std::vector<ev::tally<types>> records{
        {4, 5},
        {9, 0, 17},
        {2, 3, std::uint64_t{1} << 40},
    };
    auto block = ev::tally_block<types>::create(queue, arena.view(), records);
    REQUIRE(block.size() == 3);
    CHECK_FALSE(block.empty());

    // The counts start from the records' `ones`, 0 for the first.
    const auto view = block.view();
    const auto arena_view = arena.view();
    CHECK(view.size() == 3);
    CHECK_FALSE(view.empty());
    CHECK(view.bitpacks_per_node() == kBitpacks);
    STATIC_CHECK(std::same_as<decltype(view.observed(0)), ev::bitpack_span<const std::uint64_t, std::uint32_t>>);
    for (std::uint32_t i = 0; i < block.size(); ++i) {
        INFO("tally=" << i);
        const auto &record = block.records()[i];
        CHECK(record.id_in_pdag == records[i].id_in_pdag);
        CHECK(record.slot == records[i].slot);
        CHECK(record.ones == records[i].ones);
        CHECK(&view[i] == &record);
        CHECK(view.observed(i).data() == arena_view.at(records[i].slot));
        CHECK(view.observed(i).size() == kBitpacks);
    }

    // A slot outside the arena is refused.
    const std::vector<ev::tally<types>> outside{
        {4, 5},
        {8, 6},
    };
    CHECK_THROWS_AS(ev::tally_block<types>::create(queue, arena_view, outside), std::out_of_range);

    const auto empty = ev::tally_block<types>::create(queue, arena_view, std::vector<ev::tally<types>>{});
    CHECK(empty.empty());
    CHECK(empty.records().empty());
    CHECK(empty.view().empty());

    // Moved from, by construction and by assignment: empty.
    auto moved = std::move(block);
    CHECK(moved.size() == 3);
    ev::tally_block<types> assigned;
    assigned = std::move(moved);
    CHECK(assigned.size() == 3);
    CHECK(assigned.view().observed(2).data() == arena_view.at(3));
    // NOLINTBEGIN(bugprone-use-after-move)
    for (const auto *from : {&block, &moved}) {
        CHECK(from->empty());
        CHECK(from->size() == 0);
        CHECK(from->records().empty());
        CHECK(from->view().empty());
        CHECK(from->view().bitpacks_per_node() == 0);
    }
    // NOLINTEND(bugprone-use-after-move)
}

TEST_CASE("bernoulli_law maps a probability to its exact threshold", "[mc][event]") {
    using ev::bernoulli_kind;
    using ev::bernoulli_law;

    struct law_case {
        double p;
        bernoulli_kind kind;
        std::uint64_t threshold;
    };
    const std::array<law_case, 8> cases{{
        {0.0, bernoulli_kind::never, 0},
        {-0.0, bernoulli_kind::never, 0},
        // Below 2^-64, floor(p · 2^64) is 0; the largest double below 2^-64 included.
        {0x1p-65, bernoulli_kind::never, 0},
        {0x1.fffffffffffffp-65, bernoulli_kind::never, 0},
        {0x1p-64, bernoulli_kind::sampled, 1},
        {0.5, bernoulli_kind::sampled, std::uint64_t{1} << 63},
        // The largest double below 1: 2^64 - 2^11.
        {1.0 - 0x1p-53, bernoulli_kind::sampled, ~std::uint64_t{0} - 2047},
        {1.0, bernoulli_kind::always, pcfb::kThresholdAlwaysTrue},
    }};
    for (const auto &c : cases) {
        INFO("p=" << std::hexfloat << c.p);
        const auto law = bernoulli_law::from_probability(c.p);
        REQUIRE(law.has_value());
        CHECK(law->kind() == c.kind);
        CHECK(law->threshold() == c.threshold);
    }

    // Outside [0, 1], NaN included.
    for (const double p :
         {std::numeric_limits<double>::quiet_NaN(), -1.0, 1.0 + 0x1p-52, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
        INFO("p=" << std::hexfloat << p);
        CHECK_FALSE(bernoulli_law::from_probability(p).has_value());
    }
    // Subnormals, evaluated at compile time, where no denormals-are-zero mode applies.
    STATIC_CHECK(bernoulli_law::from_probability(std::numeric_limits<double>::denorm_min())->kind() == bernoulli_kind::never);
    STATIC_CHECK_FALSE(bernoulli_law::from_probability(-std::numeric_limits<double>::denorm_min()).has_value());
    STATIC_CHECK(bernoulli_law::from_probability(0x1p-64)->threshold() == 1);

    // probability() is exact for thresholds of at most 53 significant bits.
    CHECK(bernoulli_law::from_probability(0.5)->probability() == 0.5);
    CHECK(bernoulli_law::from_probability(0x1p-64)->probability() == 0x1p-64);
    CHECK(bernoulli_law::from_probability(1.0 - 0x1p-53)->probability() == 1.0 - 0x1p-53);
    CHECK(bernoulli_law::never().probability() == 0.0);
    CHECK(bernoulli_law::always().probability() == 1.0);

    CHECK(*bernoulli_law::from_probability(0.5) == bernoulli_law::from_threshold(std::uint64_t{1} << 63));
    CHECK(*bernoulli_law::from_probability(0.0) == bernoulli_law::never());
    CHECK(*bernoulli_law::from_probability(-0.0) == bernoulli_law::never());
    CHECK(*bernoulli_law::from_probability(0x1p-65) == bernoulli_law::never());
    CHECK(bernoulli_law::from_threshold(std::uint64_t{0}) == bernoulli_law::never());
    CHECK(bernoulli_law::from_threshold(std::uint64_t{0}).kind() == bernoulli_kind::never);
    CHECK(*bernoulli_law::from_probability(1.0) == bernoulli_law::always());
    CHECK(*bernoulli_law::from_probability(0.5) != *bernoulli_law::from_probability(0.25));
    CHECK(bernoulli_law::never() != bernoulli_law::always());

    // p = 1 is a kind of its own: the largest threshold, 2^64 - 1, is sampled, and false with probability 2^-64.
    const auto largest = bernoulli_law::from_threshold(std::numeric_limits<std::uint64_t>::max());
    CHECK(largest.kind() == bernoulli_kind::sampled);
    CHECK(largest.threshold() == bernoulli_law::always().threshold());
    CHECK(largest != bernoulli_law::always());
}

TEST_CASE("bernoulli_law takes only a double probability and a 64-bit threshold", "[mc][event]") {
    STATIC_CHECK(takes_probability<double>);
    STATIC_CHECK_FALSE(takes_probability<float>);
    STATIC_CHECK_FALSE(takes_probability<long double>);
    STATIC_CHECK_FALSE(takes_probability<int>);
    STATIC_CHECK_FALSE(takes_probability<std::uint64_t>);

    // Any 64-bit unsigned type is a threshold, whichever of them std::uint64_t names on the platform.
    STATIC_CHECK(takes_threshold<std::uint64_t>);
    STATIC_CHECK(takes_threshold<unsigned long long>);
    STATIC_CHECK(takes_threshold<std::uintmax_t>);
    STATIC_CHECK_FALSE(takes_threshold<double>);
    STATIC_CHECK_FALSE(takes_threshold<int>);
    STATIC_CHECK_FALSE(takes_threshold<std::int64_t>);
    STATIC_CHECK_FALSE(takes_threshold<std::uint32_t>);
    STATIC_CHECK_FALSE(takes_threshold<bool>);

    // No conversion from an arithmetic value, and no default: a basic event cannot exist without its law. A gate
    // record, whose other fields are made from {}, is made from its id alone.
    STATIC_CHECK_FALSE(std::is_constructible_v<ev::bernoulli_law, double>);
    STATIC_CHECK_FALSE(std::is_constructible_v<ev::bernoulli_law, std::uint64_t>);
    STATIC_CHECK_FALSE(std::is_default_constructible_v<ev::bernoulli_law>);
    STATIC_CHECK_FALSE(std::is_default_constructible_v<ev::basic_event<types>>);
    STATIC_CHECK_FALSE(made_from_id_alone<ev::basic_event<types>>);
    STATIC_CHECK(made_from_id_alone<ev::gate<types>>);
}

TEST_CASE("a basic event's law decides its record: sampled for the kernel, or a constant written once", "[mc][event]") {
    const auto never = ev::basic_event<types>{.id_in_pdag = 2, .law = ev::bernoulli_law::never()};
    const auto always = ev::basic_event<types>{.id_in_pdag = 3, .law = ev::bernoulli_law::always()};
    const auto tiny = ev::basic_event<types>{.id_in_pdag = 4, .law = *ev::bernoulli_law::from_probability(1e-30)};
    const auto lowest = ev::basic_event<types>{.id_in_pdag = 5, .law = ev::bernoulli_law::from_threshold(std::uint64_t{1})};
    const auto highest = ev::basic_event<types>{.id_in_pdag = 6, .law = ev::bernoulli_law::from_threshold(~std::uint64_t{0})};
    const auto half = ev::basic_event<types>{.id_in_pdag = 7, .law = *ev::bernoulli_law::from_probability(0.5)};

    // Every law is exactly one of the two: p below 2^-64 is `never`, and T = 2^64 - 1 is still sampled, not `always`.
    for (const auto &event : {never, always, tiny, lowest, highest, half}) {
        INFO("event " << event.id_in_pdag);
        CHECK(ev::sampled_event<types>::of(event).has_value() != ev::constant_event<types>::of(event).has_value());
        CHECK(ev::sampled_event<types>::of(event).has_value() == (event.law.kind() == ev::bernoulli_kind::sampled));
    }
    CHECK(ev::constant_event<types>::of(never)->value == false);
    CHECK(ev::constant_event<types>::of(always)->value == true);
    CHECK(ev::constant_event<types>::of(tiny)->value == false);
    CHECK(ev::constant_event<types>::of(always)->id_in_pdag == 3);
    CHECK(ev::sampled_event<types>::of(lowest)->law.threshold() == 1);
    CHECK(ev::sampled_event<types>::of(highest)->law.threshold() == ~std::uint64_t{0});
    CHECK(ev::sampled_event<types>::of(half)->law.law() == half.law);
    CHECK(ev::sampled_event<types>::of(half)->id_in_pdag == 7);
    CHECK_FALSE(ev::sampled_law::of(ev::bernoulli_law::never()).has_value());
    CHECK_FALSE(ev::sampled_law::of(ev::bernoulli_law::always()).has_value());

    // A sampled law is made only from a sampled bernoulli_law, so the sampling kernel's records cannot hold a constant.
    STATIC_CHECK_FALSE(std::is_default_constructible_v<ev::sampled_law>);
    STATIC_CHECK_FALSE(std::is_constructible_v<ev::sampled_law, std::uint64_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<ev::sampled_law, ev::bernoulli_law>);
    STATIC_CHECK_FALSE(std::is_default_constructible_v<ev::sampled_event<types>>);
    STATIC_CHECK(ev::node_record<ev::sampled_event<types>, types>);
    STATIC_CHECK(ev::node_record<ev::constant_event<types>, types>);
    STATIC_CHECK(sizeof(ev::sampled_event<types>) == 16);
}

TEST_CASE("the engine's types are held to what the kernels can represent", "[mc][event]") {
    // Bitpacks: unsigned words of at most 64 bits, bool excluded.
    STATIC_CHECK(ev::bitpack_word<std::uint8_t>);
    STATIC_CHECK(ev::bitpack_word<std::uint16_t>);
    STATIC_CHECK(ev::bitpack_word<std::uint32_t>);
    STATIC_CHECK(ev::bitpack_word<std::uint64_t>);
#if defined(__SIZEOF_INT128__)
    STATIC_CHECK_FALSE(ev::bitpack_word<unsigned __int128>);
    STATIC_CHECK_FALSE(engine_bitpack<unsigned __int128>);
    STATIC_CHECK_FALSE(ev::node_types_like<wide_types>);
#endif
    STATIC_CHECK_FALSE(ev::bitpack_word<bool>);
    STATIC_CHECK_FALSE(ev::bitpack_word<const bool>);
    STATIC_CHECK_FALSE(ev::bitpack_word<std::int64_t>);
    STATIC_CHECK_FALSE(ev::bitpack_word<double>);

    // Indices and extents: at most 32 bits, a Philox counter word; extents unsigned.
    STATIC_CHECK(ev::counter_word<std::int32_t>);
    STATIC_CHECK(ev::counter_word<std::uint32_t>);
    STATIC_CHECK(ev::counter_word<std::int16_t>);
    STATIC_CHECK_FALSE(ev::counter_word<std::int64_t>);
    STATIC_CHECK_FALSE(ev::counter_word<std::uint64_t>);
    STATIC_CHECK_FALSE(ev::counter_word<bool>);
    STATIC_CHECK(ev::node_extent<std::uint32_t>);
    STATIC_CHECK(ev::node_extent<std::uint16_t>);
    STATIC_CHECK_FALSE(ev::node_extent<std::int32_t>);
    STATIC_CHECK_FALSE(ev::node_extent<std::uint64_t>);

    // node_types holds its arguments to the same concepts.
    STATIC_CHECK(ev::node_types_like<types>);
    STATIC_CHECK(ev::node_types_like<narrow>);
    STATIC_CHECK(engine_bitpack<std::uint8_t>);
    STATIC_CHECK_FALSE(engine_bitpack<bool>);
    STATIC_CHECK_FALSE(engine_bitpack<std::int64_t>);
    STATIC_CHECK(engine_index<std::int16_t>);
    STATIC_CHECK_FALSE(engine_index<std::int64_t>);
    STATIC_CHECK(engine_extent<std::uint16_t>);
    STATIC_CHECK_FALSE(engine_extent<std::int32_t>);
    STATIC_CHECK_FALSE(engine_extent<std::uint64_t>);

    // Records name their node with exactly the index type; gate records add their run of the input table.
    STATIC_CHECK(ev::node_record<ev::basic_event<types>, types>);
    STATIC_CHECK(ev::node_record<ev::gate<types>, types>);
    STATIC_CHECK(ev::node_record<ev::cardinality_gate<types>, types>);
    STATIC_CHECK(ev::node_record<ev::tally<types>, types>);
    STATIC_CHECK(ev::gate_record<ev::gate<types>, types>);
    STATIC_CHECK(ev::gate_record<ev::cardinality_gate<types>, types>);
    STATIC_CHECK_FALSE(ev::gate_record<ev::basic_event<types>, types>);
    STATIC_CHECK_FALSE(ev::gate_record<ev::tally<types>, types>);
    STATIC_CHECK_FALSE(ev::node_record<ev::basic_event<narrow>, types>);
    STATIC_CHECK_FALSE(ev::gate_record<ev::gate<narrow>, types>);
    STATIC_CHECK(sizeof(ev::basic_event<types>) == 24);

    // Kernels hold views by value.
    STATIC_CHECK(std::is_trivially_copyable_v<ev::device_span<std::uint64_t, std::uint32_t>>);
    STATIC_CHECK(std::is_trivially_copyable_v<ev::slot_range<std::uint32_t>>);
    STATIC_CHECK(std::is_trivially_copyable_v<ev::arena_view<types>>);
    STATIC_CHECK(std::is_trivially_copyable_v<ev::basic_event_block_view<types>>);
    STATIC_CHECK(std::is_trivially_copyable_v<ev::gate_block_view<types>>);
    STATIC_CHECK(std::is_trivially_copyable_v<ev::cardinality_gate_block_view<types>>);
    STATIC_CHECK(std::is_trivially_copyable_v<ev::tally_block_view<types>>);

    // The owners are move-only.
    STATIC_CHECK(move_only<ev::usm_array<std::uint64_t>>);
    STATIC_CHECK(move_only<ev::output_arena<types>>);
    STATIC_CHECK(move_only<ev::basic_event_block<types>>);
    STATIC_CHECK(move_only<ev::gate_block<types>>);
    STATIC_CHECK(move_only<ev::cardinality_gate_block<types>>);
    STATIC_CHECK(move_only<ev::tally_block<types>>);
}

TEMPLATE_TEST_CASE("a kernel writes through a block's view into the block's slots and no others", "[mc][event]", std::uint8_t, std::uint16_t, std::uint32_t,
                   std::uint64_t) {
    using bitpack_t = TestType;
    using T = ev::node_types<bitpack_t, std::int32_t, std::uint32_t>;
    // 5 bitpacks per node; 3 slots before the block, 4 in it and 2 after it.
    constexpr std::uint32_t kBitpacks = 5;
    constexpr std::uint32_t kBefore = 3;
    constexpr std::uint32_t kNodes = 4;
    constexpr std::uint32_t kSlots = kBefore + kNodes + 2;
    constexpr auto kUntouched = static_cast<bitpack_t>(0xa5a5a5a5a5a5a5a5ull);
    constexpr std::size_t kWorkGroup = 32;

    sycl::queue queue;
    INFO("device=" << queue.get_device().get_info<sycl::info::device::name>() << " bits=" << sizeof(bitpack_t) * 8);

    auto arena = ev::output_arena<T>::create(queue, kSlots, kBitpacks);
    const std::vector<bitpack_t> initial(arena.view().size(), kUntouched);
    queue.memcpy(arena.view().data(), initial.data(), initial.size() * sizeof(bitpack_t)).wait_and_throw();

    const auto before = arena.reserve(kBefore);
    std::vector<ev::basic_event<T>> records;
    for (std::uint32_t i = 0; i < kNodes; ++i) {
        records.push_back({.id_in_pdag = static_cast<std::int32_t>(100 + i), .law = ev::bernoulli_law::never()});
    }
    const auto block = ev::basic_event_block<T>::create(queue, arena, records);
    REQUIRE(block.slots().first == before.end());
    REQUIRE(block.size() == kNodes);

    // Work-item k writes bitpack k mod B of node k / B: the node's slot number.
    const auto view = block.view();
    const std::size_t items = std::size_t{view.size()} * view.bitpacks_per_node();
    const sycl::nd_range<1> range{sycl::range<1>{(items + kWorkGroup - 1) / kWorkGroup * kWorkGroup}, sycl::range<1>{kWorkGroup}};
    queue.parallel_for(range, [=](const sycl::nd_item<1> item) {
        const std::size_t k = item.get_global_id(0);
        if (k >= items) {
            return;
        }
        const auto i = static_cast<std::uint32_t>(k / view.bitpacks_per_node());
        const auto w = static_cast<std::uint32_t>(k % view.bitpacks_per_node());
        view.output(i)[w] = static_cast<bitpack_t>(view.slot(i));
    });
    queue.wait_and_throw();

    std::vector<bitpack_t> host(initial.size());
    queue.memcpy(host.data(), arena.view().data(), host.size() * sizeof(bitpack_t)).wait_and_throw();
    for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
        INFO("slot=" << slot);
        const std::uint64_t expected = block.slots().contains(slot) ? slot : std::uint64_t{kUntouched};
        for (std::uint32_t w = 0; w < kBitpacks; ++w) {
            CHECK(std::uint64_t{host[std::size_t{slot} * kBitpacks + w]} == expected);
        }
    }
}
