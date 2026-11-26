// Known bugs of the runtime's .pcfb loader (pdag_from_circuit): valid files that `fzb run` refuses. Each case
// builds its file with the pinned pcfb::builder, checks that pcfb::validate accepts it, and expects run_pcfb to
// answer it like compile::exact_probability. Fails on develop until its issue is fixed. Labelled `bug` and `mc`.

#include "compile/exact_eval.h"
#include "pcfb.h"
#include "pcfb/builder.h"
#include "pcfb/threshold.h"
#include "runtime/runtime.h"
#include "runtime/tests/test_backend.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kTrials = std::uint64_t{1} << 16;

void require_runs_like_exact(std::vector<std::byte> const &bytes) {
    auto view = pcfb::io::from_bytes(bytes);
    REQUIRE(view);
    REQUIRE(pcfb::validate(*view).empty());
    const double exact = fuzzybits::compile::exact_probability(*view);

    fuzzybits::runtime::RunOptions options;
    options.trials = kTrials;
    options.seed = 7;
    options.backend = fuzzybits::testing::backend();
    std::vector<fuzzybits::runtime::QueryResult> results;
    try {
        const auto circuit = fuzzybits::runtime::load_pcfb(bytes);
        results = fuzzybits::runtime::run_pcfb(circuit, options);
    } catch (std::exception const &error) {
        FAIL("run_pcfb threw: " << error.what());
    }
    REQUIRE(results.size() == 1);
    const auto n = static_cast<double>(results[0].n);
    const double spread = std::sqrt(n * exact * (1.0 - exact));
    CHECK(std::abs(static_cast<double>(results[0].hits) - n * exact) <= 4.0 * spread + 1.0);
}

} // namespace

// pcfb SampleArgs.event_id is a uint32, unique within the subcircuit (pcfb::validate checks nothing else), and
// docs/pc-pcfb.md (`event_id`) makes it part of the file contract. The loader places each SAMPLE at PDAG index
// `static_cast<int>(event_id) + shift`, so an id of 2^31 or more becomes a negative index and
// Pdag::SetNextNodeIndex throws std::logic_error. (Ids just below 2^31 are accepted but make
// Pdag::SetBasicEvent resize a pointer table to the id: about 16 GiB for one event.)
TEST_CASE("fzb run accepts SAMPLE event ids of 2^31 and above", "[bug][runtime][mc][pcfb][issue-296]") {
    pcfb::builder b;
    auto low = b.add_sample(0.5, "low", 1);
    auto high = b.add_sample(0.25, "high", 3'000'000'000U);
    REQUIRE(low);
    REQUIRE(high);
    const std::uint32_t inputs[] = {*low, *high};
    auto gate = b.add_op(pcfb::OpCode::BITWISE_OR, inputs);
    REQUIRE(gate);
    REQUIRE(b.add_query(*gate, pcfb::QueryKind::PROBABILITY, "top"));
    auto bytes = b.finish();
    REQUIRE(bytes);
    require_runs_like_exact(*bytes);
}

// pcfb::validate accepts a SampleArgs.threshold within 1 ULP of floor(p * 2^64), and pcfb::builder's explicit
// threshold overload writes such files (probability = threshold / 2^64, rounded). The loader requires the
// threshold to equal probability_to_threshold(p) exactly and throws "SampleArgs.threshold does not match
// p*2^64" otherwise, so a file the format accepts is refused before sampling.
TEST_CASE("fzb run accepts a SAMPLE threshold that pcfb::validate accepts", "[bug][runtime][mc][pcfb][issue-296]") {
    pcfb::builder b;
    // 2^63 + 1: its probability rounds to 0.5, whose exact threshold is 2^63, one below.
    const std::uint64_t threshold = (std::uint64_t{1} << 63U) + 1U;
    auto event = b.add_sample(threshold, false, "e", 1);
    REQUIRE(event);
    REQUIRE(pcfb::probability_to_threshold(0.5) != threshold);
    const std::uint32_t inputs[] = {*event};
    auto gate = b.add_op(pcfb::OpCode::BITWISE_OR, inputs);
    REQUIRE(gate);
    REQUIRE(b.add_query(*gate, pcfb::QueryKind::PROBABILITY, "top"));
    auto bytes = b.finish();
    REQUIRE(bytes);
    require_runs_like_exact(*bytes);
}

// The loader rebuilds a SCRAM PDAG and names its nodes with MEF rules: a basic-event name has every '.' replaced by
// '_' (sanitize_name, because a MEF element name cannot hold a '.'), and a gate name is split at its last '.' into
// a base path and a name. Names from other writers follow no such rule: FTP names may hold any punctuation
// (docs/ftp-input.md, Grammar), so `SYS.A` and `SYS_A` are two events. Watch names (Session, Python PcfbSession,
// the server) resolve against the renamed nodes: `SYS.A` matches nothing, and `SYS_A` matches both renamed nodes, so
// the first one in an unordered map's iteration order wins (in this build the right one). Each name must resolve
// to its own event (the loader places event ids 1 and 2 at PDAG indices 2 and 3).
TEST_CASE("watch names resolve to the event of that name, whatever punctuation it holds",
          "[bug][runtime][mc][pcfb][issue-300]") {
    pcfb::builder b;
    auto dotted = b.add_sample(0.1, "SYS.A", 1);
    auto underscored = b.add_sample(0.9, "SYS_A", 2);
    REQUIRE(dotted);
    REQUIRE(underscored);
    const std::uint32_t inputs[] = {*dotted, *underscored};
    auto gate = b.add_op(pcfb::OpCode::BITWISE_OR, inputs, {}, std::nullopt, "TOP");
    REQUIRE(gate);
    REQUIRE(b.add_query(*gate, pcfb::QueryKind::PROBABILITY, "TOP"));
    auto bytes = b.finish();
    REQUIRE(bytes);
    const auto circuit = fuzzybits::runtime::load_pcfb(*bytes);
    const auto reconstructed = fuzzybits::runtime::pdag_from_circuit(circuit);

    std::vector<int> dotted_index;
    try {
        dotted_index = fuzzybits::runtime::Session::resolve_watch(reconstructed, {"SYS.A"});
    } catch (std::exception const &error) {
        FAIL_CHECK("resolve_watch(SYS.A) threw: " << error.what());
    }
    CHECK(dotted_index == std::vector<int>{2});
    CHECK(fuzzybits::runtime::Session::resolve_watch(reconstructed, {"SYS_A"}) == std::vector<int>{3});
}

// fzb run rebuilds every .pcfb as a SCRAM PDAG, whatever wrote it, and parts of the SCRAM PDAG and kernel set-up
// recurse once per level of gate depth. A 10^5-deep chain of alternating AND and OR gates (so no pass can flatten
// it), compiled without error from FTP (-c 0 and -c 1) or imported from nombrilo, ends `fzb run` with SIGSEGV. The
// FTP reader, the ir::pc passes and the .pcfb writer are iterative and handle the same chain. Expected: the run
// answers it (P(G0) by the chain recurrence) or refuses it with a diagnostic, never a crash.
TEST_CASE("fzb run evaluates a 10^5-deep gate chain", "[bug][runtime][mc][pcfb][crash][issue-300]") {
    constexpr std::uint32_t depth = 100'000;
    pcfb::builder b;
    std::vector<std::uint32_t> events;
    for (std::uint32_t index = 0; index <= depth; ++index) {
        auto event = b.add_sample(0.5, "E" + std::to_string(index), index + 1);
        REQUIRE(event);
        events.push_back(*event);
    }
    // G_depth = E_depth; G_i = AND(G_{i+1}, E_i) for even i, OR for odd i.
    std::uint32_t chain = events[depth];
    double probability = 0.5;
    for (std::uint32_t index = depth; index-- > 0;) {
        const std::uint32_t inputs[] = {chain, events[index]};
        const bool conjunction = index % 2 == 0;
        auto gate = b.add_op(conjunction ? pcfb::OpCode::BITWISE_AND : pcfb::OpCode::BITWISE_OR, inputs);
        REQUIRE(gate);
        chain = *gate;
        probability = conjunction ? probability * 0.5 : 1.0 - (1.0 - probability) * 0.5;
    }
    REQUIRE(b.add_query(chain, pcfb::QueryKind::PROBABILITY, "G0"));
    auto bytes = b.finish();
    REQUIRE(bytes);

    fuzzybits::runtime::RunOptions options;
    options.trials = kTrials;
    options.backend = fuzzybits::testing::backend();
    const auto circuit = fuzzybits::runtime::load_pcfb(*bytes);
    const auto results = fuzzybits::runtime::run_pcfb(circuit, options);
    REQUIRE(results.size() == 1);
    const auto n = static_cast<double>(results[0].n);
    const double spread = std::sqrt(n * probability * (1.0 - probability));
    CHECK(std::abs(static_cast<double>(results[0].hits) - n * probability) <= 4.0 * spread + 1.0);
}
