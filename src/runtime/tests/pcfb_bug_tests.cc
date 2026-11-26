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
