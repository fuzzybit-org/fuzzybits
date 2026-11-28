// Tests of the kernel builder and the descriptions it records, without a PDAG: the plan's order and edges, the
// numbers a partition derives, the choice of kernel among its instantiations, the op codes a layer takes, and a
// two-layer program built on the device, partition by partition and run for a pass, with the errors the builder's
// state refuses.

#include <catch2/catch_test_macros.hpp>

#include "mc/event/basic_event.h"
#include "mc/event/node.h"
#include "mc/event/sample_shape.h"
#include "mc/executor/builder/builder.h"
#include "mc/executor/builder/dispatch.h"
#include "mc/executor/builder/gate.h"
#include "mc/executor/builder/layer.h"
#include "mc/executor/builder/op.h"
#include "mc/executor/builder/state.h"
#include "mc/executor/descriptor/partition.h"
#include "mc/executor/descriptor/plan.h"
#include "mc/prng/stream.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace builder = fuzzybits::mc::executor::builder;
namespace descriptor = fuzzybits::mc::executor::descriptor;
namespace ev = fuzzybits::mc::event;
using builder::op_code;
using types = ev::node_types<std::uint64_t, std::int32_t, std::uint32_t>;

descriptor::partition reading(const std::vector<std::size_t> &dependencies) { return {.dependencies = dependencies}; }

/// The arena slot of node `id` in `block`.
template <class Block> std::uint32_t slot_of(const Block &block, const std::int32_t id) {
    const auto records = block.records();
    const auto at = std::ranges::find_if(records, [id](const auto &record) { return record.id_in_pdag == id; });
    REQUIRE(at != records.end());
    return block.slot(static_cast<std::uint32_t>(at - records.begin()));
}

} // namespace

TEST_CASE("a plan numbers its partitions in order and refuses a partition that reads one not before it", "[mc][executor]") {
    descriptor::plan plan;
    CHECK(plan.add(reading({})) == 0);
    CHECK(plan.add(reading({0})) == 1);
    CHECK(plan.add(reading({0, 1})) == 2);
    CHECK_THROWS_AS(plan.add(reading({3})), std::logic_error);
    CHECK_THROWS_AS(plan.add(reading({1, 4})), std::logic_error);
    CHECK(plan.size() == 3);

    // One edge per dependency, by target, then source.
    const auto edges = plan.edges();
    REQUIRE(edges.size() == 3);
    CHECK((edges[0].source == 0 && edges[0].target == 1));
    CHECK((edges[1].source == 0 && edges[1].target == 2));
    CHECK((edges[2].source == 1 && edges[2].target == 2));
    CHECK(descriptor::edge::reason == "buffer_dependency");
}

TEST_CASE("a partition derives its work, memory and fan-in from what it knows", "[mc][executor]") {
    descriptor::fan_in none;
    CHECK((none.gates == 0 && none.min == 0 && none.max == 0 && none.total == 0));
    descriptor::fan_in inputs;
    inputs.add(3, 1);
    inputs.add(1, 0);
    inputs.add(2, 2);
    CHECK((inputs.gates == 3 && inputs.min == 1 && inputs.max == 4 && inputs.total == 9 && inputs.positive == 6 && inputs.negated == 3));

    // 5 nodes of 96 work-items each, in work-groups of 32.
    const auto shape = descriptor::launch::of(sycl::nd_range<1>{sycl::range<1>{5 * 96}, sycl::range<1>{32}}, 5);
    CHECK(shape.global == descriptor::range3{.x = 5, .y = 1, .z = 96});
    CHECK(shape.local == descriptor::range3{.x = 1, .y = 1, .z = 32});

    const descriptor::partition partition{.kind = descriptor::partition_kind::gate,
                                          .nodes = {7, 8, 9, 10, 11},
                                          .shape = shape,
                                          .batches = 2,
                                          .bitpacks_per_batch = 40,
                                          .bitpack_bytes = 8,
                                          .inputs = inputs};
    CHECK(partition.node_count() == 5);
    CHECK(partition.logical_range() == descriptor::range3{.x = 5, .y = 2, .z = 40});
    CHECK(partition.work_items() == 480);
    CHECK(partition.work_groups() == 15);
    CHECK(partition.logical_work_items() == 400);
    CHECK(partition.overprovisioned_work_items() == 80);
    CHECK(partition.bitpacks_per_node() == 80);
    CHECK(partition.bytes_per_node() == 640);
    CHECK(partition.bits_per_node() == 5120);
    CHECK(partition.resident_bytes() == 3200);
    CHECK(descriptor::name_of(partition.kind) == "gate");
}

TEST_CASE("select calls the instantiation for the value it is given, and no other", "[mc][executor]") {
    const auto tenfold = []<int v>() { return v * 10; };
    CHECK(builder::select<1, 2, 3>(2, tenfold, "value") == 20);
    CHECK(builder::select<1, 2, 3>(3, tenfold, "value") == 30);
    CHECK_THROWS_AS((builder::select<1, 2, 3>(4, tenfold, "value")), std::logic_error);
    static constexpr std::array<std::uint32_t, 3> kChoices{4, 8, 16};
    CHECK(builder::select_in<kChoices>(std::uint32_t{8}, []<std::uint32_t v>() { return v + 1; }, "choice") == 9);
    CHECK_THROWS_AS((builder::select_in<kChoices>(std::uint32_t{5}, []<std::uint32_t v>() { return v; }, "choice")), std::logic_error);
}

TEST_CASE("gates share a kernel when they need the same op code, form and slots", "[mc][executor]") {
    using fuzzybits::mc::kernel::cardinality_form;
    const auto nine = builder::gate_kernel::of(op_code::atleast, 9);
    CHECK((nine.form == cardinality_form::fixed && nine.slots == 12 && !nine.counted()));
    CHECK(builder::gate_kernel::of(op_code::atleast, 10) == nine);
    const auto many = builder::gate_kernel::of(op_code::atleast, 17);
    CHECK((many.form == cardinality_form::counter && many.slots == 0 && many.counted()));
    CHECK(builder::gate_kernel::of(op_code::atleast, 300).form == cardinality_form::wide);
    CHECK(builder::gate_kernel::of(op_code::cardinality, 9).slots == 12);
    CHECK(builder::gate_kernel::of(op_code::not_gate, 1).slots == 1);
    CHECK(builder::gate_kernel::of(op_code::null_gate, 1).slots == 1);
    CHECK(builder::gate_kernel::of(op_code::and_gate, 5).counted());
    CHECK(builder::gate_kernel::of(op_code::xnor_gate, 5).counted());
    CHECK(builder::gate_kernel::of(op_code::and_gate, 5) == builder::gate_kernel::of(op_code::and_gate, 50));
    CHECK(builder::gate_kernel::of(op_code::and_gate, 5) < builder::gate_kernel::of(op_code::or_gate, 5));
    CHECK(nine < many);

    builder::dependencies reads;
    for (const std::size_t partition : {4, 1, 4, 0, 1}) {
        reads.add(partition);
    }
    CHECK(reads.ids() == std::vector<std::size_t>{0, 1, 4});
}

TEST_CASE("the builder makes a program partition by partition, after the partitions each reads", "[mc][executor]") {
    sycl::queue queue;
    const ev::sample_shape<std::size_t> shape{.batch_size = 1, .bitpacks_per_batch = 4};
    const fuzzybits::mc::prng::stream stream = fuzzybits::mc::prng::stream::of(0x243f6a8885a308d3ull, 1);
    auto arena = ev::output_arena<types>::create(queue, 6, shape.num_bitpacks());

    // Layer 0: event 2 is sampled, event 3 is never true, and node 6 is the constant TRUE. Layer 1: AND(2, not 3),
    // NOT(3) and exactly 2 of (2, 6). The AND gate is tallied.
    builder::layer<types> events;
    events.basic_events.push_back({.id_in_pdag = 2, .law = *ev::bernoulli_law::from_probability(0.5)});
    events.basic_events.push_back({.id_in_pdag = 3, .law = ev::bernoulli_law::never()});
    events.add_constant(6, true);
    builder::layer<types> gates;
    const std::array<builder::input, 2> and_inputs{{{.node = 2, .negated = false}, {.node = 3, .negated = true}}};
    const std::array<builder::input, 1> not_inputs{{{.node = 3, .negated = false}}};
    const std::array<builder::input, 2> exactly_inputs{{{.node = 2, .negated = false}, {.node = 6, .negated = false}}};
    gates.add_gate(op_code::exactly, 7, exactly_inputs, {.low = 2, .high = 2});
    gates.add_gate(op_code::not_gate, 5, not_inputs);
    gates.add_gate(op_code::and_gate, 4, and_inputs);
    const std::array<std::int32_t, 1> tallied{4};

    builder::program_builder<types, std::size_t> build(queue, shape, stream, arena, 8);
    build.add(events, 0).add(gates, 1).tally(tallied, 2);
    const builder::program<types> program = std::move(build).finish();

    // The sampled event, then the gates in op code order (AND, NOT, exactly), then the tally. The constants are written
    // when their slots are allocated: no partition computes them, and none waits for them.
    const auto partitions = program.plan.partitions();
    REQUIRE(partitions.size() == 5);
    REQUIRE(program.queueables.size() == 5);
    CHECK(partitions[0].kind == descriptor::partition_kind::sampler);
    CHECK(partitions[0].operation == "sample");
    CHECK(partitions[0].nodes == std::vector<std::int32_t>{2});
    CHECK(partitions[0].dependencies.empty());
    CHECK(partitions[1].kind == descriptor::partition_kind::gate);
    CHECK(partitions[1].operation == "and");
    CHECK(partitions[1].nodes == std::vector<std::int32_t>{4});
    CHECK(partitions[1].dependencies == std::vector<std::size_t>{0});
    CHECK((partitions[1].inputs.positive == 1 && partitions[1].inputs.negated == 1 && partitions[1].inputs.total == 2));
    CHECK(partitions[2].operation == "not");
    CHECK(partitions[2].dependencies.empty());
    CHECK(partitions[3].kind == descriptor::partition_kind::gate);
    CHECK(partitions[3].operation == "exactly");
    CHECK(partitions[3].nodes == std::vector<std::int32_t>{7});
    CHECK(partitions[3].dependencies == std::vector<std::size_t>{0});
    CHECK(partitions[4].kind == descriptor::partition_kind::tally);
    CHECK(partitions[4].layer == 2);
    CHECK(partitions[4].dependencies == std::vector<std::size_t>{1});
    for (const descriptor::partition &partition : partitions) {
        INFO("partition " << partition.id);
        CHECK(partition.bitpacks_per_node() == shape.num_bitpacks());
        CHECK(partition.bytes_per_node() == shape.num_bitpacks() * sizeof(std::uint64_t));
        CHECK(partition.work_items() % partition.node_count() == 0);
    }
    // The queueables wait for exactly the partitions the plan says.
    CHECK(program.queueables[1]->dependencies() == std::vector{program.queueables[0]});
    CHECK(program.queueables[2]->dependencies().empty());
    CHECK(program.queueables[3]->dependencies() == std::vector{program.queueables[0]});
    CHECK(program.queueables[4]->dependencies() == std::vector{program.queueables[1]});

    REQUIRE(program.blocks.sampled_events.size() == 1);
    REQUIRE(program.blocks.constant_events.size() == 1);
    CHECK(program.blocks.constant_events.front().size() == 2);
    REQUIRE(program.blocks.gates.size() == 2);
    REQUIRE(program.blocks.cardinality_gates.size() == 1);
    REQUIRE(program.blocks.tallies.size() == 1);
    REQUIRE(program.tallies.contains(4));
    CHECK(program.ones(4) == 0);
    CHECK_THROWS_AS(program.ones(5), std::out_of_range);

    // One pass: node 6 is all ones and node 3 all zeros, written once; AND(2, not 3) and exactly 2 of (2, 6) are
    // event 2's trials, NOT(3) is all ones, and the tally counts event 2's true trials.
    for (const auto &queueable : program.queueables) {
        queueable->submit(0);
    }
    queue.wait_and_throw();
    const std::size_t bitpacks = shape.num_bitpacks();
    std::vector<std::uint64_t> image(arena.view().size());
    queue.memcpy(image.data(), arena.view().data(), image.size() * sizeof(std::uint64_t)).wait_and_throw();
    const auto bits = [&](const std::uint32_t slot) {
        const auto first = image.begin() + static_cast<std::ptrdiff_t>(slot * bitpacks);
        return std::vector<std::uint64_t>(first, first + static_cast<std::ptrdiff_t>(bitpacks));
    };
    const std::vector<std::uint64_t> ones(bitpacks, ~std::uint64_t{0});
    const std::vector<std::uint64_t> zeros(bitpacks, 0);
    const auto event = bits(slot_of(program.blocks.sampled_events.front(), 2));
    CHECK(event != ones);
    CHECK(event != zeros);
    CHECK(bits(slot_of(program.blocks.constant_events.front(), 6)) == ones);
    CHECK(bits(slot_of(program.blocks.constant_events.front(), 3)) == zeros);
    CHECK(bits(slot_of(program.blocks.gates[0], 4)) == event);
    CHECK(bits(slot_of(program.blocks.gates[1], 5)) == ones);
    CHECK(bits(slot_of(program.blocks.cardinality_gates.front(), 7)) == event);
    std::uint64_t true_trials = 0;
    for (const std::uint64_t word : event) {
        true_trials += static_cast<std::uint64_t>(std::popcount(word));
    }
    CHECK(program.ones(4) == true_trials);
}

TEST_CASE("a layer takes each node by its op code, and refuses a gate no kernel evaluates as given", "[mc][executor]") {
    const std::array<builder::input, 2> two{{{.node = 2, .negated = false}, {.node = 3, .negated = true}}};
    const std::array<builder::input, 1> one{{{.node = 2, .negated = false}}};
    builder::layer<types> nodes;
    nodes.add_gate(op_code::xnor_gate, 10, two);
    nodes.add_gate(op_code::null_gate, 11, one);
    nodes.add_gate(op_code::atmost, 12, two, {.low = 0, .high = 1});
    nodes.add_gate(op_code::and_gate, 13, std::span<const builder::input>{});
    nodes.add_constant(14, false);
    REQUIRE(nodes.gates(op_code::xnor_gate).size() == 1);
    CHECK(nodes.inputs_of(nodes.gates(op_code::xnor_gate).front()).size() == 2);
    CHECK(nodes.gates(op_code::null_gate).front().index == 11);
    const builder::gate &atmost = nodes.gates(op_code::atmost).front();
    CHECK((atmost.range.low == 0 && atmost.range.high == 1 && atmost.inputs == 2));
    CHECK(nodes.gates(op_code::and_gate).front().inputs == 0);
    REQUIRE(nodes.basic_events.size() == 1);
    CHECK(nodes.basic_events.front().id_in_pdag == 14);
    CHECK(nodes.basic_events.front().law.kind() == ev::bernoulli_kind::never);

    // A counting gate takes bounds within its inputs, a Boolean gate none, NOT and NULL one input; no kernel computes
    // the other op codes as gates.
    CHECK_THROWS_AS(nodes.add_gate(op_code::atleast, 20, two), std::invalid_argument);
    CHECK_THROWS_AS(nodes.add_gate(op_code::or_gate, 20, two, {.low = 1, .high = 2}), std::invalid_argument);
    CHECK_THROWS_AS(nodes.add_gate(op_code::exactly, 20, two, {.low = 2, .high = 1}), std::invalid_argument);
    CHECK_THROWS_AS(nodes.add_gate(op_code::cardinality, 20, two, {.low = 1, .high = 3}), std::invalid_argument);
    CHECK_THROWS_AS(nodes.add_gate(op_code::not_gate, 20, two), std::invalid_argument);
    CHECK_THROWS_AS(nodes.add_gate(op_code::null_gate, 20, std::span<const builder::input>{}), std::invalid_argument);
    CHECK_THROWS_AS(nodes.add_gate(op_code::sample, 20, two), std::invalid_argument);
    CHECK_THROWS_AS(nodes.add_gate(op_code::ones, 20, std::span<const builder::input>{}), std::invalid_argument);
    // A majority or minority of two inputs computes [2, 2] or [0, 1], and no other range.
    CHECK_THROWS_AS(nodes.add_gate(op_code::majority, 20, two, {.low = 1, .high = 2}), std::invalid_argument);
    CHECK_THROWS_AS(nodes.add_gate(op_code::minority, 20, two, {.low = 0, .high = 0}), std::invalid_argument);
    CHECK_THROWS_AS(nodes.gates(op_code::house_event), std::out_of_range);

    // The gates, in op code order, are every op code a gate kernel evaluates; descriptions name them without `_gate`.
    CHECK(std::ranges::is_sorted(builder::kGateOps));
    // The seven Boolean gates and NULL, then cardinality, atleast, atmost, exactly, majority and minority.
    CHECK(builder::kGateOps.size() == 14);
    for (const op_code op : builder::kGateOps) {
        CHECK(builder::is_gate(op));
        CHECK(builder::is_boolean(op) != builder::is_counting(op));
    }
    CHECK(builder::gate_position(op_code::sample) == builder::kGateOps.size());
    CHECK(builder::name_of(op_code::and_gate) == "and");
    CHECK(builder::name_of(op_code::null_gate) == "null");
    CHECK(builder::name_of(op_code::xnor_gate) == "xnor");
    CHECK(builder::name_of(op_code::atleast) == "atleast");
    CHECK(builder::name_of(op_code::majority) == "majority");
    CHECK(builder::name_of(op_code::minority) == "minority");
    CHECK(builder::name_of(op_code::sample) == "sample");
    CHECK(builder::name_of(op_code::ones) == "ones");
}

TEST_CASE("the builder refuses a node read before it is built, built twice, or tallied twice", "[mc][executor]") {
    sycl::queue queue;
    const ev::sample_shape<std::size_t> shape{.batch_size = 1, .bitpacks_per_batch = 2};
    auto arena = ev::output_arena<types>::create(queue, 8, shape.num_bitpacks());
    builder::layer<types> events;
    events.basic_events.push_back({.id_in_pdag = 2, .law = *ev::bernoulli_law::from_probability(0.25)});

    SECTION("a gate that reads a node no layer before it computes") {
        builder::program_builder<types, std::size_t> build(queue, shape, {}, arena, 8);
        build.add(events, 0);
        builder::layer<types> gates;
        const std::array<builder::input, 2> inputs{{{.node = 2, .negated = false}, {.node = 6, .negated = false}}};
        gates.add_gate(op_code::or_gate, 7, inputs);
        CHECK_THROWS_AS(build.add(gates, 1), std::logic_error);
    }
    SECTION("a node in two layers") {
        builder::program_builder<types, std::size_t> build(queue, shape, {}, arena, 8);
        build.add(events, 0);
        CHECK_THROWS_AS(build.add(events, 1), std::logic_error);
    }
    SECTION("a node tallied twice, or a node that is not built") {
        builder::program_builder<types, std::size_t> build(queue, shape, {}, arena, 8);
        build.add(events, 0);
        const std::array<std::int32_t, 2> twice{2, 2};
        CHECK_THROWS_AS(build.tally(twice, 1), std::logic_error);
        const std::array<std::int32_t, 1> unbuilt{3};
        builder::program_builder<types, std::size_t> other(queue, shape, {}, arena, 8);
        other.add(events, 0);
        CHECK_THROWS_AS(other.tally(unbuilt, 1), std::logic_error);
    }
}

TEST_CASE("majority and minority gates have no kernel yet: the builder refuses them before any kernel runs", "[mc][executor]") {
    // TODO:: a kernel test instead, once kernels compute them (kernel::has_kernel).
    CHECK_FALSE(builder::has_kernel(op_code::majority));
    CHECK_FALSE(builder::has_kernel(op_code::minority));
    CHECK(builder::has_kernel(op_code::atleast));
    CHECK(builder::has_kernel(op_code::exactly));

    sycl::queue queue;
    const ev::sample_shape<std::size_t> shape{.batch_size = 1, .bitpacks_per_batch = 4};
    const fuzzybits::mc::prng::stream stream = fuzzybits::mc::prng::stream::of(0x243f6a8885a308d3ull, 1);
    for (const op_code op : {op_code::majority, op_code::minority}) {
        INFO(builder::name_of(op));
        auto arena = ev::output_arena<types>::create(queue, 4, shape.num_bitpacks());
        builder::layer<types> events;
        events.basic_events.push_back({.id_in_pdag = 1, .law = *ev::bernoulli_law::from_probability(0.5)});
        events.basic_events.push_back({.id_in_pdag = 2, .law = *ev::bernoulli_law::from_probability(0.25)});
        events.basic_events.push_back({.id_in_pdag = 3, .law = *ev::bernoulli_law::from_probability(0.125)});
        builder::layer<types> gates;
        const std::array<builder::input, 3> inputs{{{.node = 1, .negated = false}, {.node = 2, .negated = false}, {.node = 3, .negated = false}}};
        gates.add_gate(op, 4, inputs, op == op_code::majority ? builder::bounds{.low = 2, .high = 3} : builder::bounds{.low = 0, .high = 1});
        builder::program_builder<types, std::size_t> build(queue, shape, stream, arena, 8);
        build.add(events, 0);
        CHECK_THROWS_AS(build.add(gates, 1), builder::not_implemented);
    }
}
