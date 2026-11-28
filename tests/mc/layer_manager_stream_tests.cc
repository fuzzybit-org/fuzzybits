// Engine-level test of the random stream's plumbing through queue::layer_manager (src/mc/executor/layer_manager.h): from
// the stream given to the constructor and the pass index to the tally counts.
//
// The model is one basic event, p = 0.3125, written as a pcfb circuit and reconstructed by the runtime
// (runtime::pdag_from_circuit), as fzb run reconstructs it, so no SCRAM-derived header is included here. Its query is
// a pass-through (NULL) gate over the event, which the reconstruction adds because the PDAG root must be a gate. The
// tally observes that gate, so it counts exactly the event's true trials.
//
// Pass p (from 0) of a layer manager with B bitpacks per node per pass samples the event's run-wide bitpacks
// [p·B, (p + 1)·B): the Philox counters of its PDAG index, the stream's device and those bitpack indices, under the
// stream's seed (prng/stream.h). After k passes the tally's count is therefore the host reference
//
//     sum over p < k and s < B of popcount(bernoulli::sample(pass_stream(stream, id, p·B), s, T)),
//
// an exact integer. Checked after each of three passes, with a seed whose two key words are both set and device 2,
// it pins:
// - the constructor's stream: not the default stream, the seed on device 0, the default seed on device 2, or the
//   seed's low 32 bits;
// - the event's PDAG index as the counters' node word;
// - the pass index from 0: not passes counted from 1;
// - first = p·B: not first = p, nor every pass at first = 0 or at the last pass's bitpacks.
// Each of these faults changes the counts at the pass size the device realizes; the test checks that on the host
// before it samples, so a change of constants that would let a fault through fails instead. It also checks passes()
// and total_bits(), the count as single_pass_and_tally and collect_tallies read it, and three passes submitted with
// no wait between them (pass_wait_collect, as runtime::Session::step(3) submits them).
//
// A second case pins how events whose law is a constant are made: in a layer with a sampled event, an event with p = 0
// and one with p = 1, the only basic-event kernel samples the sampled event; the constants' bitpacks are written once,
// when their arena slots are allocated, so their tallies count exactly 0 and all trials after every pass.
//
// Not pinned here: run-wide bitpack indices of 2^32 and more, and the sampler forms against each other
// (mc.bitpack_identity); how the entry points (fzb run, Session, the Python bindings) choose the stream.

#include <catch2/catch_test_macros.hpp>

#include "mc/event/basic_event.h"
#include "mc/executor/layer_manager.h"
#include "mc/kernel/sampler/sampler.h"
#include "mc/prng/bernoulli/sample.h"
#include "mc/prng/stream.h"
#include "mc/stats/tally_node_map.h"
#include "runtime/runtime.h"
#include "runtime/tests/test_backend.h"

#include "pcfb/builder.h"

#include <sycl/sycl.hpp>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace prng = fuzzybits::mc::prng;
namespace stats = fuzzybits::mc::stats;
using manager_type = fuzzybits::mc::executor::layer_manager<std::uint64_t, std::double_t, std::size_t>;

// Neither the default seed nor device 0, and both key words set, so a stream that kept only the seed's low 32 bits
// is another stream.
constexpr std::uint64_t kSeed = 0x13198a2e03707344ull;
constexpr std::uint32_t kDevice = 2;
constexpr prng::stream kStream = prng::stream::of(kSeed, kDevice);
static_assert(kStream.key.k0 != 0 && kStream.key.k1 != 0 && kStream != prng::stream{});

// The event's pcfb event_id, which the reconstruction makes its PDAG index and so the node word of its counters. Not
// 0, the event's arena slot.
constexpr std::uint32_t kEventId = 5;
constexpr double kProbability = 0.3125;
// floor(0.3125 · 2^64) = 5 · 2^60: its lowest set bit is bit 60, so a bitpack takes at most two generations.
constexpr std::uint64_t kThreshold = std::uint64_t{5} << 60;
static_assert(fuzzybits::mc::event::bernoulli_law::from_probability(kProbability)->threshold() == kThreshold);

// 64 bitpacks per pass. The sample shaper rounds the request down to whole cache lines and a power of two, so the test
// reads the realized B from the manager.
constexpr std::size_t kRequestedTrialsPerPass = 4096;
constexpr std::uint32_t kPasses = 3;

// The test backend (FUZZYBITS_BACKEND, else omp) as the visibility mask when none is set, before the first queue. An
// explicit ACPP_VISIBILITY_MASK wins. The layer manager's queue honours the mask (queue::make_throwing_sycl_queue).
struct default_mask {
    default_mask() {
        const std::string &mask = fuzzybits::testing::backend();
        if (mask.empty() || std::getenv("ACPP_VISIBILITY_MASK") != nullptr) {
            return;
        }
#ifdef _WIN32
        _putenv_s("ACPP_VISIBILITY_MASK", mask.c_str());
#else
        setenv("ACPP_VISIBILITY_MASK", mask.c_str(), 0);
#endif
    }
};
const default_mask kDefaultMask{};

/// The one-event model, written as pcfb and reconstructed as fzb run reconstructs it. The reconstruction outlives the
/// bytes and the circuit, as in runtime::Session::from_bytes.
fuzzybits::runtime::ReconstructedPdag one_event_model() {
    pcfb::builder builder;
    const auto event = builder.add_sample(kProbability, "x", kEventId);
    REQUIRE(event);
    REQUIRE(builder.add_query(*event, pcfb::QueryKind::PROBABILITY, "x"));
    const auto bytes = builder.finish();
    REQUIRE(bytes);
    return fuzzybits::runtime::pdag_from_circuit(fuzzybits::runtime::load_pcfb(*bytes));
}

// The PDAG indices of the events with a constant law in mixed_law_model.
constexpr std::uint32_t kNeverId = 7;
constexpr std::uint32_t kAlwaysId = 8;

/// Three basic events, each queried: the event of one_event_model, one with p = 0 and one with p = 1.
fuzzybits::runtime::ReconstructedPdag mixed_law_model() {
    pcfb::builder builder;
    const auto sampled = builder.add_sample(kProbability, "x", kEventId);
    const auto never = builder.add_sample(0.0, "never", kNeverId);
    const auto always = builder.add_sample(1.0, "always", kAlwaysId);
    REQUIRE(sampled);
    REQUIRE(never);
    REQUIRE(always);
    REQUIRE(builder.add_query(*sampled, pcfb::QueryKind::PROBABILITY, "x"));
    REQUIRE(builder.add_query(*never, pcfb::QueryKind::PROBABILITY, "never"));
    REQUIRE(builder.add_query(*always, pcfb::QueryKind::PROBABILITY, "always"));
    const auto bytes = builder.finish();
    REQUIRE(bytes);
    return fuzzybits::runtime::pdag_from_circuit(fuzzybits::runtime::load_pcfb(*bytes));
}

/// The query's tally, as fzb run and Session ask for it.
stats::TallyNodeMap tally_of(const fuzzybits::runtime::ReconstructedQuery &query) {
    stats::TallyNodeMap tallies;
    tallies.insert_or_assign(query.node_index, stats::TallyNode{.status = stats::not_tracked, .tally_stats = {}, .node = query.node});
    return tallies;
}

/// One-bits of node `node`'s bitpacks [first, first + bitpacks) in `run`, from the host reference (prng/bernoulli/sample.h).
std::uint64_t reference_ones(const prng::stream &run, const std::uint32_t node, const std::uint64_t first, const std::uint32_t bitpacks) {
    const prng::bernoulli::pass_stream source(run, node, first);
    std::uint64_t ones = 0;
    for (std::uint32_t s = 0; s < bitpacks; ++s) {
        ones += static_cast<std::uint64_t>(std::popcount(prng::bernoulli::sample(source, s, kThreshold)));
    }
    return ones;
}

/// The tally's count after each of passes 1 .. kPasses when pass p samples bitpacks [first(p), first(p) + bitpacks) of
/// node `node` in `run`.
template <class First> std::vector<std::uint64_t> counts(const prng::stream &run, const std::uint32_t node, const std::uint32_t bitpacks, const First first) {
    std::vector<std::uint64_t> out;
    std::uint64_t ones = 0;
    for (std::uint32_t p = 0; p < kPasses; ++p) {
        ones += reference_ones(run, node, first(p), bitpacks);
        out.push_back(ones);
    }
    return out;
}

/// A fault in the plumbing from the constructor's stream and the pass index to the counters, and the counts it gives.
struct fault {
    const char *what;
    std::vector<std::uint64_t> counts;
};

std::vector<fault> faults_at(const std::uint32_t bitpacks) {
    const auto from_pass = [bitpacks](const std::uint32_t p) { return std::uint64_t{p} * bitpacks; };
    return {
        {"the default stream", counts(prng::stream{}, kEventId, bitpacks, from_pass)},
        {"the seed on device 0", counts(prng::stream::of(kSeed, 0), kEventId, bitpacks, from_pass)},
        {"the default seed on the device", counts(prng::stream::of(prng::stream::kDefaultSeed, kDevice), kEventId, bitpacks, from_pass)},
        {"the seed's low 32 bits", counts(prng::stream::of(kSeed & 0xffffffffull, kDevice), kEventId, bitpacks, from_pass)},
        {"node word 0 (the event's arena slot)", counts(kStream, 0, bitpacks, from_pass)},
        // The event's index among the PDAG's basic events (its PDAG index less core::Pdag::kVariableStartIndex, 2),
        // which the kernel builder computes next to its PDAG index.
        {"node word PDAG index - 2 (the event's basic_events() position)", counts(kStream, kEventId - 2, bitpacks, from_pass)},
        {"passes counted from 1", counts(kStream, kEventId, bitpacks, [bitpacks](const std::uint32_t p) { return std::uint64_t{p + 1} * bitpacks; })},
        {"first = p", counts(kStream, kEventId, bitpacks, [](const std::uint32_t p) { return std::uint64_t{p}; })},
        {"every pass at first = 0", counts(kStream, kEventId, bitpacks, [](std::uint32_t) { return std::uint64_t{0}; })},
        {"every pass at the last pass's bitpacks",
         counts(kStream, kEventId, bitpacks, [bitpacks](std::uint32_t) { return std::uint64_t{kPasses - 1} * bitpacks; })},
        // Kernels that read the pass counter when they run rather than when they are submitted: with every pass
        // submitted before the first runs, each samples the bitpacks after the last pass's.
        {"every pass after the last pass's bitpacks (the pass index read after its increment)",
         counts(kStream, kEventId, bitpacks, [bitpacks](std::uint32_t) { return std::uint64_t{kPasses} * bitpacks; })},
    };
}

/// The faults whose count after pass k is `observed`, for a failure message.
std::string faults_giving(const std::vector<fault> &faults, const std::uint32_t k, const std::uint64_t observed) {
    std::string out;
    for (const auto &f : faults) {
        if (f.counts[k - 1] == observed) {
            out += (out.empty() ? "" : "; ") + std::string{f.what};
        }
    }
    return out.empty() ? "none" : out;
}

} // namespace

TEST_CASE("layer_manager tallies pass p at the counters of its stream from bitpack p*B", "[mc][stream]") {
    auto model = one_event_model();
    REQUIRE(model.graph);
    REQUIRE(model.queries.size() == 1);
    const auto &query = model.queries.front();
    REQUIRE(query.node);

    manager_type manager(model.graph.get(), 0, tally_of(query), kRequestedTrialsPerPass, kStream);
    REQUIRE(manager.stream() == kStream);
    REQUIRE(manager.passes() == 0);

    // The model as the engine sees it: the event at its event_id, and the query's pass-through gate, which is tallied.
    const auto dag = manager.describe_kernel_dag();
    REQUIRE(dag.nodes.size() == 2);
    REQUIRE(dag.nodes[0].kind == "basic_event");
    REQUIRE(dag.nodes[0].index == static_cast<std::int32_t>(kEventId));
    REQUIRE(dag.nodes[1].kind == "gate");
    REQUIRE(dag.nodes[1].connective == "null");
    REQUIRE(dag.nodes[1].index == query.node_index);

    // The shaper rounds the request down. B >= 2, or first = p would be first = p·B.
    const auto realized = manager.shaper().SAMPLE_SHAPE.num_bitpacks();
    REQUIRE(realized >= 2);
    REQUIRE(realized <= kRequestedTrialsPerPass / 64);
    const auto bitpacks = static_cast<std::uint32_t>(realized);
    const auto expected = counts(kStream, kEventId, bitpacks, [bitpacks](const std::uint32_t p) { return std::uint64_t{p} * bitpacks; });
    const auto faults = faults_at(bitpacks);

    const sycl::device device = manager.queue().get_device();
    const bool chunked = fuzzybits::mc::kernel::sampler::form_for(device) == fuzzybits::mc::kernel::sampler::form::chunked;
    INFO("device=" << device.get_info<sycl::info::device::name>() << ", sampler form " << (chunked ? "chunked" : "strided") << ", B=" << bitpacks);

    SECTION("one pass at a time") {
        for (const auto &f : faults) {
            INFO("fault: " << f.what);
            REQUIRE(f.counts != expected);
        }
        stats::TallyNodeMap collected;
        for (std::uint32_t k = 1; k <= kPasses; ++k) {
            INFO("after pass " << k);
            const std::uint64_t trials = std::uint64_t{k} * bitpacks * 64;
            const auto estimate = manager.single_pass_and_tally(query.node_index);
            REQUIRE(manager.passes() == k);
            REQUIRE(manager.total_bits() == trials);
            REQUIRE(estimate.total_bits == trials);
            INFO("faults that give this count: " << faults_giving(faults, k, estimate.num_one_bits));
            REQUIRE(estimate.num_one_bits == expected[k - 1]);

            manager.collect_tallies(collected);
            REQUIRE(collected.at(query.node_index).tally_stats.num_one_bits == expected[k - 1]);
            REQUIRE(collected.at(query.node_index).tally_stats.total_bits == trials);
        }
    }

    SECTION("three passes submitted with no wait between them, as Session::step(3) submits them") {
        // Only the count after the last pass is observed.
        for (const auto &f : faults) {
            INFO("fault: " << f.what);
            REQUIRE(f.counts.back() != expected.back());
        }
        stats::TallyNodeMap collected;
        manager.pass_wait_collect(collected, kPasses, 0);
        const std::uint64_t trials = std::uint64_t{kPasses} * bitpacks * 64;
        REQUIRE(manager.passes() == kPasses);
        REQUIRE(manager.total_bits() == trials);
        const auto &tally = collected.at(query.node_index).tally_stats;
        INFO("faults that give this count: " << faults_giving(faults, kPasses, tally.num_one_bits));
        REQUIRE(tally.num_one_bits == expected.back());
        REQUIRE(tally.total_bits == trials);
    }
}

TEST_CASE("basic events with a constant law are written when their slots are allocated, and get no kernel", "[mc][stream]") {
    auto model = mixed_law_model();
    REQUIRE(model.graph);
    REQUIRE(model.queries.size() == 3);
    stats::TallyNodeMap tallies;
    std::unordered_map<std::string, int> tally_of_query;
    for (const auto &query : model.queries) {
        REQUIRE(query.node);
        tallies.insert_or_assign(query.node_index, stats::TallyNode{.status = stats::not_tracked, .tally_stats = {}, .node = query.node});
        tally_of_query[query.name] = query.node_index;
    }
    REQUIRE(tally_of_query.size() == 3);

    manager_type manager(model.graph.get(), 0, tallies, kRequestedTrialsPerPass, kStream);

    // The constants' layer has one basic-event kernel, and it samples only the sampled event.
    std::size_t sampling_kernels = 0;
    for (const auto &partition : manager.describe_execution_plan().partitions) {
        if (partition.kind == fuzzybits::mc::executor::descriptor::partition_kind::sampler) {
            ++sampling_kernels;
            CHECK(partition.nodes == std::vector<std::int32_t>{static_cast<std::int32_t>(kEventId)});
        }
    }
    CHECK(sampling_kernels == 1);

    const auto bitpacks = static_cast<std::uint32_t>(manager.shaper().SAMPLE_SHAPE.num_bitpacks());
    REQUIRE(bitpacks >= 2);
    const auto expected = counts(kStream, kEventId, bitpacks, [bitpacks](const std::uint32_t p) { return std::uint64_t{p} * bitpacks; });
    stats::TallyNodeMap collected;
    for (std::uint32_t k = 1; k <= kPasses; ++k) {
        INFO("after pass " << k << ", B=" << bitpacks);
        const std::uint64_t trials = std::uint64_t{k} * bitpacks * 64;
        (void)manager.single_pass_and_tally(tally_of_query.at("x"));
        manager.collect_tallies(collected);
        CHECK(collected.at(tally_of_query.at("x")).tally_stats.num_one_bits == expected[k - 1]);
        CHECK(collected.at(tally_of_query.at("never")).tally_stats.num_one_bits == 0);
        CHECK(collected.at(tally_of_query.at("always")).tally_stats.num_one_bits == trials);
        for (const auto &[name, index] : tally_of_query) {
            INFO("query " << name);
            CHECK(collected.at(index).tally_stats.total_bits == trials);
        }
    }
}
