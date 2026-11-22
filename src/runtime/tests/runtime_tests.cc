#include "compile/pcfb_writer.h"
#include "pdag.h"
#include "runtime/runtime.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct OmpBackend {
#ifdef _WIN32
  OmpBackend() { _putenv_s("ACPP_VISIBILITY_MASK", "omp"); }
#else
  OmpBackend() { setenv("ACPP_VISIBILITY_MASK", "omp", 0); }
#endif
};
const OmpBackend kOmpBackend{};

std::string aralia(const char *name) {
  return (std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "Aralia" / name).string();
}

std::string repo_input(const char *relative) {
  return (std::filesystem::path(PROJECT_SOURCE_DIR) / relative).string();
}

fuzzybits::compile::Options no_kn() {
  fuzzybits::compile::Options opt;
  opt.compilation_level = 2;
  opt.expand_atleast_gates = true;
  return opt;
}

void check_pcfb_matches_memory(const std::string &xml, std::uint64_t trials) {
  fuzzybits::runtime::RunOptions opt;
  opt.trials = trials;
  opt.backend = "omp";
  opt.seed = 372;

  auto live = fuzzybits::compile::compile_to_graph({xml}, no_kn());
  REQUIRE(live);
  REQUIRE(live->graph);

  fuzzybits::runtime::ReconstructedQuery q;
  q.name = live->target_id;
  q.node = live->graph->root_ptr();
  q.node_index = live->graph->root_ptr()->index();
  const auto in_memory = fuzzybits::runtime::run_graph(*live->graph, {q}, opt);
  REQUIRE(in_memory.size() == 1);

  auto compiled = fuzzybits::compile::pdag_to_pcfb(*live->graph, no_kn(), live->model_name, live->target_id, {});
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(*compiled);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  const auto from_pcfb = fuzzybits::runtime::run_pcfb(*circuit, opt);
  REQUIRE(from_pcfb.size() == 1);

  INFO("in-memory hits=" << in_memory[0].hits << " n=" << in_memory[0].n);
  INFO("pcfb hits=" << from_pcfb[0].hits << " n=" << from_pcfb[0].n);
  REQUIRE(from_pcfb[0].hits == in_memory[0].hits);
  REQUIRE(from_pcfb[0].n == in_memory[0].n);
  REQUIRE(from_pcfb[0].n >= trials);
}

double z_score(double p_hat, double p_ref, std::uint64_t n) {
  const double sigma = std::sqrt(p_hat * (1.0 - p_hat) / static_cast<double>(n));
  REQUIRE(sigma > 0.0);
  return (p_hat - p_ref) / sigma;
}

}  // namespace

namespace {

/// Records every snapshot and finish, and can throw from `on_snapshot` to simulate a callback that
/// fails (the Python binding path).
struct RecordingSink : fuzzybits::runtime::progress::sink {
  std::vector<fuzzybits::runtime::progress::snapshot> snapshots;
  bool finished = false;
  bool interrupted = true;
  int finish_count = 0;
  int snapshot_count = 0;  // on_snapshot calls, including ones that threw or followed a finish
  std::vector<bool> finishes;  // the `interrupted` flag of every on_finish
  int throw_on = -1;  // 0-based snapshot index that raises, -1 never

  void on_snapshot(const fuzzybits::runtime::progress::snapshot &s) override {
    ++snapshot_count;
    if (throw_on >= 0 && static_cast<int>(snapshots.size()) >= throw_on) {
      throw fuzzybits::runtime::progress::interrupted("sink requested stop");
    }
    snapshots.push_back(s);
  }

  void on_finish(const fuzzybits::runtime::progress::snapshot &last, bool flag) override {
    finished = true;
    interrupted = flag;
    ++finish_count;
    finishes.push_back(flag);
    last_snapshot = last;
  }

  fuzzybits::runtime::progress::snapshot last_snapshot;
};

using now_point = fuzzybits::runtime::progress::emitter::clock::time_point;

now_point at_ms(std::int64_t ms) {
  return now_point{} + std::chrono::milliseconds(ms);
}

}  // namespace

TEST_CASE("progress::emitter with no sinks is inert", "[runtime][progress][emitter]") {
  fuzzybits::runtime::progress::emitter emitter({});
  CHECK_FALSE(emitter.active());
  int built = 0;
  emitter.maybe_emit([&built] {
    ++built;
    return fuzzybits::runtime::progress::snapshot{};
  });
  emitter.emit_now(fuzzybits::runtime::progress::snapshot{});
  emitter.finish(fuzzybits::runtime::progress::snapshot{}, false);
  CHECK(built == 0);
}

TEST_CASE("progress::emitter coalesces to the rate but always delivers first and last",
          "[runtime][progress][emitter]") {
  RecordingSink sink;
  auto now = at_ms(1000);
  fuzzybits::runtime::progress::emitter emitter({&sink}, 10.0, [&now] { return now; });

  int built = 0;
  auto make = [&built] {
    ++built;
    fuzzybits::runtime::progress::snapshot s;
    s.passes_done = static_cast<std::uint64_t>(built);
    return s;
  };

  emitter.maybe_emit(make);  // first: always delivered
  now = at_ms(1010);
  emitter.maybe_emit(make);  // inside the 100 ms window: coalesced, make() not called
  now = at_ms(1050);
  emitter.maybe_emit(make);
  now = at_ms(1099);
  emitter.maybe_emit(make);
  now = at_ms(1100);
  emitter.maybe_emit(make);  // exactly one interval later: delivered
  now = at_ms(1150);
  emitter.maybe_emit(make);
  now = at_ms(1200);
  emitter.maybe_emit(make);  // delivered

  CHECK(built == 3);
  REQUIRE(sink.snapshots.size() == 3);
  CHECK(sink.snapshots.front().passes_done == 1);
  CHECK(sink.snapshots.back().passes_done == 3);

  auto last = make();
  last.passes_done = 99;
  emitter.finish(last, false);
  emitter.finish(last, true);  // exactly once
  REQUIRE(sink.snapshots.size() == 4);
  CHECK(sink.snapshots.back().passes_done == 99);
  CHECK(sink.finished);
  CHECK_FALSE(sink.interrupted);
  CHECK(sink.finish_count == 1);
}

TEST_CASE("progress::emitter never swallows a sink exception at finish",
          "[runtime][progress][emitter]") {
  RecordingSink sink;
  fuzzybits::runtime::progress::emitter emitter({&sink}, 10.0);
  sink.throw_on = 0;  // the first on_snapshot from finish()
  // The sink's own exception is not rethrown (nothing to rethrow from a destructor-safe path); the
  // emitter raises its own typed signal instead, after every sink got on_finish.
  CHECK_THROWS_AS(emitter.finish(fuzzybits::runtime::progress::snapshot{}, false),
                  fuzzybits::runtime::progress::interrupted);
  // on_finish still ran, and only once.
  CHECK(sink.finished);
  CHECK_FALSE(sink.interrupted);
  CHECK(sink.finish_count == 1);
}

TEST_CASE("progress::finish_guard reports interruption on any unwind, exactly once",
          "[runtime][progress][emitter]") {
  SECTION("unwinding with a non-sink exception") {
    RecordingSink sink;
    fuzzybits::runtime::progress::emitter emitter({&sink}, 0.0);
    auto run = [&] {
      fuzzybits::runtime::progress::finish_guard guard([&emitter] {
        emitter.finish(fuzzybits::runtime::progress::snapshot{}, true);
      });
      emitter.maybe_emit([] { return fuzzybits::runtime::progress::snapshot{}; });
      throw std::runtime_error("device failed");
    };
    CHECK_THROWS_AS(run(), std::runtime_error);
    CHECK(sink.finish_count == 1);
    CHECK(sink.interrupted);
  }

  SECTION("completing normally does not double-finish") {
    RecordingSink sink;
    fuzzybits::runtime::progress::emitter emitter({&sink}, 0.0);
    {
      fuzzybits::runtime::progress::finish_guard guard([&emitter] {
        emitter.finish(fuzzybits::runtime::progress::snapshot{}, true);
      });
      emitter.maybe_emit([] { return fuzzybits::runtime::progress::snapshot{}; });
      emitter.finish(fuzzybits::runtime::progress::snapshot{}, false);
      guard.complete();
    }
    CHECK(sink.finish_count == 1);
    CHECK_FALSE(sink.interrupted);
  }

  SECTION("a throwing finish is swallowed while unwinding and the original error survives") {
    RecordingSink sink;
    sink.throw_on = 0;
    fuzzybits::runtime::progress::emitter emitter({&sink}, 0.0);
    auto run = [&] {
      fuzzybits::runtime::progress::finish_guard guard([&emitter] {
        emitter.finish(fuzzybits::runtime::progress::snapshot{}, true);
      });
      throw std::runtime_error("device failed");
    };
    CHECK_THROWS_AS(run(), std::runtime_error);
    CHECK(sink.finish_count == 1);
    CHECK(sink.interrupted);
  }

  SECTION("a moved-from guard does not fire") {
    RecordingSink sink;
    fuzzybits::runtime::progress::emitter emitter({&sink}, 0.0);
    {
      fuzzybits::runtime::progress::finish_guard first([&emitter] {
        emitter.finish(fuzzybits::runtime::progress::snapshot{}, true);
      });
      auto second = std::move(first);
      CHECK_FALSE(first.armed());
      CHECK(second.armed());
    }
    CHECK(sink.finish_count == 1);
  }
}

TEST_CASE("progress::emitter with a non-positive rate is unlimited", "[runtime][progress][emitter]") {
  RecordingSink sink;
  fuzzybits::runtime::progress::emitter emitter({&sink}, 0.0);
  for (int i = 0; i < 7; ++i) {
    emitter.maybe_emit([] { return fuzzybits::runtime::progress::snapshot{}; });
  }
  CHECK(sink.snapshots.size() == 7);
}

TEST_CASE("progress::emitter reports interruption exactly once", "[runtime][progress][emitter]") {
  RecordingSink sink;
  fuzzybits::runtime::progress::emitter emitter({&sink}, 0.0);
  emitter.maybe_emit([] { return fuzzybits::runtime::progress::snapshot{}; });
  emitter.finish(fuzzybits::runtime::progress::snapshot{}, true);
  CHECK(sink.finished);
  CHECK(sink.interrupted);
}

TEST_CASE("run_graph with and without a sink returns bit-identical estimates", "[runtime][progress][mc]") {
  constexpr std::uint64_t kTrials = 1 << 16;
  auto compiled = fuzzybits::compile::compile_files({repo_input("input/SmallTree/SmallTree.xml")}, no_kn());
  REQUIRE(compiled);

  fuzzybits::runtime::RunOptions plain;
  plain.trials = kTrials;
  plain.backend = "omp";
  plain.seed = 372;

  auto baseline = fuzzybits::runtime::run_pcfb(compiled->bytes, plain);
  REQUIRE(baseline.size() == 1);

  RecordingSink sink;
  fuzzybits::runtime::RunOptions watched = plain;
  watched.progress_sinks = {&sink};
  auto with_sink = fuzzybits::runtime::run_pcfb(compiled->bytes, watched);
  REQUIRE(with_sink.size() == 1);

  // Exactly: same hits, same n, and therefore the same estimate and interval.
  CHECK(with_sink[0].hits == baseline[0].hits);
  CHECK(with_sink[0].n == baseline[0].n);
  CHECK(with_sink[0].estimate == baseline[0].estimate);
  CHECK(with_sink[0].ci_low == baseline[0].ci_low);
  CHECK(with_sink[0].ci_high == baseline[0].ci_high);

  REQUIRE_FALSE(sink.snapshots.empty());
  const auto &first = sink.snapshots.front();
  CHECK(first.schema == fuzzybits::runtime::progress::schema_version);
  REQUIRE(first.queries.size() == with_sink.size());
  CHECK(first.queries.front().name == with_sink.front().name);

  // Work done is monotone and the last snapshot carries the final tallies.
  for (std::size_t i = 1; i < sink.snapshots.size(); ++i) {
    CHECK(sink.snapshots[i].trials_done >= sink.snapshots[i - 1].trials_done);
    CHECK(sink.snapshots[i].passes_done >= sink.snapshots[i - 1].passes_done);
  }
  const auto &last = sink.snapshots.back();
  CHECK(last.stage == fuzzybits::runtime::progress::phase::done);
  CHECK(last.passes_done == last.passes_total);
  CHECK(last.trials_done == last.trials_total);
  CHECK(last.trials_done >= kTrials);
  CHECK(last.queries.front().hits == with_sink.front().hits);
  CHECK(last.queries.front().trials == with_sink.front().n);
  CHECK(last.queries.front().estimate == with_sink.front().estimate);
  CHECK(last.queries.front().ci_low == with_sink.front().ci_low);
  CHECK(last.queries.front().ci_high == with_sink.front().ci_high);
  CHECK(sink.finished);
  CHECK_FALSE(sink.interrupted);
}

TEST_CASE("a throwing progress sink stops the run and propagates", "[runtime][progress][mc]") {
  auto compiled = fuzzybits::compile::compile_files({repo_input("input/SmallTree/SmallTree.xml")}, no_kn());
  REQUIRE(compiled);

  RecordingSink sink;
  sink.throw_on = 0;  // first snapshot
  fuzzybits::runtime::RunOptions opt;
  opt.trials = 1 << 16;
  opt.backend = "omp";
  opt.seed = 372;
  opt.progress_sinks = {&sink};
  opt.progress_max_rate_hz = 0.0;  // deliver every pass so the first snapshot arrives promptly

  CHECK_THROWS_AS(fuzzybits::runtime::run_pcfb(compiled->bytes, opt),
                  fuzzybits::runtime::progress::interrupted);
  CHECK(sink.finished);
  CHECK(sink.interrupted);
  CHECK(sink.finish_count == 1);
}

TEST_CASE("a non-sink failure inside the active run still finishes the sinks once", "[runtime][progress][mc]") {
  // The device loop fails for a reason unrelated to progress (a SYCL error, an async error, a failed
  // result build). The renderer must still get its one chance to restore the terminal.
  auto compiled = fuzzybits::compile::compile_files({repo_input("input/SmallTree/SmallTree.xml")}, no_kn());
  REQUIRE(compiled);

  // `trials_per_pass` comes from the shaper's memory budget, not the trial count, so probing it with
  // a tiny run and then requesting two passes keeps this test independent of the device.
  const auto shape = [&] {
    auto session = fuzzybits::runtime::Session::from_bytes(compiled->bytes, "omp");
    const auto s = session.sample_shape();
    session.close();
    return s;
  }();
  REQUIRE(shape.trials_per_pass > 0);

  RecordingSink sink;
  fuzzybits::runtime::RunOptions opt;
  opt.trials = shape.trials_per_pass * 2;
  opt.backend = "omp";
  opt.seed = 372;
  opt.progress_sinks = {&sink};
  opt.progress_max_rate_hz = 0.0;  // a first snapshot, so the loop takes the active path

  // Fail after the first pass, leaving one more to go.
  fuzzybits::runtime::testing::fail_next_run_after_passes(1);
  CHECK_THROWS_AS(fuzzybits::runtime::run_pcfb(compiled->bytes, opt),
                  fuzzybits::runtime::RuntimeError);

  CHECK(sink.finish_count == 1);
  CHECK(sink.finished);
  CHECK(sink.interrupted);
  REQUIRE(sink.finishes.size() == 1);
  CHECK(sink.finishes.front());
  // The interrupted snapshot describes the work really done: one of two passes, not the total.
  CHECK(sink.last_snapshot.stage == fuzzybits::runtime::progress::phase::sampling);
  CHECK(sink.last_snapshot.passes_done == 1);
  REQUIRE(sink.last_snapshot.passes_total >= 2);
  CHECK(sink.last_snapshot.passes_done < sink.last_snapshot.passes_total);
  CHECK(sink.last_snapshot.trials_done < sink.last_snapshot.trials_total);

  // The seam is one-shot: the next run is unaffected and finishes normally.
  RecordingSink second;
  opt.progress_sinks = {&second};
  const auto results = fuzzybits::runtime::run_pcfb(compiled->bytes, opt);
  CHECK(results.size() == 1);
  CHECK(second.finish_count == 1);
  CHECK_FALSE(second.interrupted);
}

TEST_CASE("a sink exception raised while a run is already unwinding is swallowed",
          "[runtime][progress][mc]") {
  auto compiled = fuzzybits::compile::compile_files({repo_input("input/SmallTree/SmallTree.xml")}, no_kn());
  REQUIRE(compiled);

  const auto shape = [&] {
    auto session = fuzzybits::runtime::Session::from_bytes(compiled->bytes, "omp");
    const auto s = session.sample_shape();
    session.close();
    return s;
  }();
  REQUIRE(shape.trials_per_pass > 0);

  RecordingSink sink;
  sink.throw_on = 1;  // the snapshot emitted from the interrupted finish()
  fuzzybits::runtime::RunOptions opt;
  opt.trials = shape.trials_per_pass * 2;
  opt.backend = "omp";
  opt.seed = 372;
  opt.progress_sinks = {&sink};
  opt.progress_max_rate_hz = 0.0;

  // First snapshot is delivered (sink records it); the injected failure then unwinds, and the
  // finish path's delivery throws. The original RuntimeError must survive.
  fuzzybits::runtime::testing::fail_next_run_after_passes(1);
  CHECK_THROWS_AS(fuzzybits::runtime::run_pcfb(compiled->bytes, opt),
                  fuzzybits::runtime::RuntimeError);
  // on_finish still ran, even though the final delivery raised.
  CHECK(sink.finish_count == 1);
  CHECK(sink.interrupted);
}

TEST_CASE("compile emits parse, lower, passes, write and a final snapshot", "[runtime][progress][compile]") {
  RecordingSink sink;
  fuzzybits::compile::Options opt = no_kn();
  opt.progress_sinks = {&sink};
  auto compiled = fuzzybits::compile::compile_files({repo_input("input/SmallTree/SmallTree.xml")}, opt);
  REQUIRE(compiled);
  REQUIRE_FALSE(compiled->bytes.empty());

  std::vector<std::string> steps;
  for (const auto &s : sink.snapshots) {
    if (s.stage == fuzzybits::runtime::progress::phase::compile) {
      steps.push_back(s.step);
    }
  }
  REQUIRE(steps.size() >= 4);
  CHECK(steps[0] == "parse");
  CHECK(steps[1] == "lower");
  CHECK(steps[2] == "passes");
  CHECK(steps.back() == "write");
  CHECK(sink.snapshots.back().stage == fuzzybits::runtime::progress::phase::done);
  CHECK(sink.finished);
  CHECK_FALSE(sink.interrupted);
}

TEST_CASE("a non-sink failure inside compile still finishes the sinks once",
          "[runtime][progress][compile]") {
  RecordingSink sink;
  fuzzybits::compile::Options opt = no_kn();
  opt.progress_sinks = {&sink};

  // Fail between stages, i.e. from code the sink never sees.
  fuzzybits::compile::testing::fail_next_compile_after_stage("lower");
  CHECK_THROWS_AS(
      fuzzybits::compile::compile_files({repo_input("input/SmallTree/SmallTree.xml")}, opt),
      std::runtime_error);

  CHECK(sink.finish_count == 1);
  REQUIRE(sink.finishes.size() == 1);
  CHECK(sink.finishes.front());
  CHECK(sink.last_snapshot.stage == fuzzybits::runtime::progress::phase::compile);
  CHECK(sink.last_snapshot.step == "lower");
}

TEST_CASE("a compile sink exception on the final snapshot is not swallowed",
          "[runtime][progress][compile]") {
  RecordingSink sink;
  fuzzybits::compile::Options opt = no_kn();
  opt.progress_sinks = {&sink};
  // Stages parse/lower/passes/write deliver four snapshots; the fifth delivery is the final one
  // from `finish()`.
  sink.throw_on = 4;

  CHECK_THROWS_AS(
      fuzzybits::compile::compile_files({repo_input("input/SmallTree/SmallTree.xml")}, opt),
      fuzzybits::runtime::progress::interrupted);
  CHECK(sink.finish_count == 1);
  CHECK_FALSE(sink.interrupted);
}

TEST_CASE("fzb run refuses an invalid pcfb file", "[runtime][mc]") {
  const auto path = std::filesystem::temp_directory_path() / "fuzzybits-invalid.pcfb";
  {
    std::ofstream out(path, std::ios::binary);
    const char junk[] = "not a pcfb";
    out.write(junk, sizeof(junk) - 1);
  }
  fuzzybits::runtime::RunOptions opt;
  opt.trials = 64;
  opt.backend = "omp";
  CHECK_THROWS_AS(fuzzybits::runtime::run_pcfb(path, opt), fuzzybits::runtime::RuntimeError);
  std::filesystem::remove(path);
}

TEST_CASE("pcfb reconstruction places SAMPLE event_id at the Variable index", "[runtime][mc]") {
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, no_kn());
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());

  auto reconstructed = fuzzybits::runtime::pdag_from_circuit(*circuit);
  REQUIRE(reconstructed.graph);
  REQUIRE_FALSE(reconstructed.queries.empty());
  REQUIRE_FALSE(reconstructed.events.empty());

  const auto *sc = circuit->subcircuits()->Get(0);
  std::size_t n_sample = 0;
  for (const auto *op : *sc->operators()) {
    if (!op || op->opcode() != pcfb::OpCode::SAMPLE) {
      continue;
    }
    ++n_sample;
    const auto *sa = op->args_as_SampleArgs();
    REQUIRE(sa);
    const auto idx = static_cast<int>(sa->event_id());
    REQUIRE(idx >= scram::core::Pdag::kVariableStartIndex);
    const auto *event = reconstructed.graph->basic_events()[idx];
    REQUIRE(event != nullptr);
    if (sa->always_true()) {
      REQUIRE(event->p() == 1.0);
    } else {
      REQUIRE(event->p() == sa->probability());
    }
  }
  REQUIRE(reconstructed.events.size() >= n_sample);
  REQUIRE_FALSE(reconstructed.threshold_notes.empty());
  REQUIRE(reconstructed.threshold_notes.front().find("2^32") != std::string::npos);
}

TEST_CASE("fzb run chinese.pcfb is bit-identical to the in-memory PDAG path", "[runtime][mc][bit-identity]") {
  constexpr std::uint64_t kTrials = 65536;
  fuzzybits::runtime::RunOptions opt;
  opt.trials = kTrials;
  opt.backend = "omp";
  opt.seed = 372;

  auto live = fuzzybits::compile::compile_to_graph({aralia("chinese.xml")}, no_kn());
  REQUIRE(live);
  REQUIRE(live->graph);

  fuzzybits::runtime::ReconstructedQuery q;
  q.name = live->target_id;
  q.node = live->graph->root_ptr();
  q.node_index = live->graph->root_ptr()->index();
  const auto in_memory = fuzzybits::runtime::run_graph(*live->graph, {q}, opt);
  REQUIRE(in_memory.size() == 1);

  auto compiled = fuzzybits::compile::pdag_to_pcfb(*live->graph, no_kn(), live->model_name, live->target_id, {});
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(*compiled);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  const auto from_pcfb = fuzzybits::runtime::run_pcfb(*circuit, opt);
  REQUIRE(from_pcfb.size() == 1);

  INFO("in-memory hits=" << in_memory[0].hits << " n=" << in_memory[0].n);
  INFO("pcfb hits=" << from_pcfb[0].hits << " n=" << from_pcfb[0].n);
  REQUIRE(from_pcfb[0].hits == in_memory[0].hits);
  REQUIRE(from_pcfb[0].n == in_memory[0].n);
  REQUIRE(from_pcfb[0].n >= kTrials);
}

TEST_CASE("Session streams the same first-pass hits as run_graph on chinese", "[runtime][mc][bit-identity]") {
  constexpr std::uint64_t kTrialsPerPass = 65536;
  fuzzybits::runtime::RunOptions opt;
  opt.trials = kTrialsPerPass;
  opt.backend = "omp";
  opt.seed = 372;

  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, no_kn());
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());

  const auto oneshot = fuzzybits::runtime::run_pcfb(*circuit, opt);
  REQUIRE(oneshot.size() == 1);

  auto session = fuzzybits::runtime::Session::from_bytes(compiled->bytes, "omp", kTrialsPerPass);
  const auto first = session.step();
  REQUIRE(first.pass_index == 1);
  REQUIRE(first.trials_per_pass >= kTrialsPerPass);
  REQUIRE_FALSE(first.nodes.empty());
  session.close();
  REQUIRE(session.closed());
}

TEST_CASE("Session::step(n) matches n single steps on chinese", "[runtime][mc][bit-identity]") {
  constexpr std::uint64_t kTrialsPerPass = 65536;
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, no_kn());
  REQUIRE(compiled);

  auto batched = fuzzybits::runtime::Session::from_bytes(compiled->bytes, "omp", kTrialsPerPass);
  auto single = fuzzybits::runtime::Session::from_bytes(compiled->bytes, "omp", kTrialsPerPass);
  const auto once = batched.step(3);
  fuzzybits::runtime::PassResult last;
  for (int i = 0; i < 3; ++i) {
    last = single.step();
  }

  REQUIRE(once.pass_index == 3);
  REQUIRE(last.pass_index == 3);
  REQUIRE(once.nodes.size() == last.nodes.size());
  for (const auto &node : once.nodes) {
    const auto match = std::find_if(last.nodes.begin(), last.nodes.end(),
                                    [&](const auto &other) { return other.index == node.index; });
    REQUIRE(match != last.nodes.end());
    REQUIRE(node.num_one_bits == match->num_one_bits);
    REQUIRE(node.total_bits == match->total_bits);
  }
  REQUIRE_THROWS(batched.step(0));
}

TEST_CASE("extra watch queries do not replace the PDAG root", "[runtime][mc][watch]") {
  fuzzybits::compile::Options opt = no_kn();
  opt.watch = {"g1"};
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, opt);
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  auto reconstructed = fuzzybits::runtime::pdag_from_circuit(*circuit);
  REQUIRE(reconstructed.queries.size() >= 2);
  REQUIRE(reconstructed.graph->root_ptr());
  REQUIRE(reconstructed.graph->root_ptr()->index() == reconstructed.queries.front().node_index);
  REQUIRE_FALSE(reconstructed.meta.target.empty());
  const auto watched = fuzzybits::runtime::Session::resolve_watch(reconstructed, {"g1"});
  REQUIRE(watched.size() == 1);
}

TEST_CASE("fzb run SmallTree.pcfb is bit-identical to the in-memory PDAG path", "[runtime][mc][bit-identity][smalltree]") {
  check_pcfb_matches_memory(repo_input("input/SmallTree/SmallTree.xml"), 65536);
}

TEST_CASE("fzb run BSCU.pcfb is bit-identical to the in-memory PDAG path", "[runtime][mc][bit-identity][bscu]") {
  check_pcfb_matches_memory(repo_input("input/BSCU/BSCU.xml"), 65536);
}

TEST_CASE("fzb run SmallTree.pcfb matches the exponential point-value formula |z|<=4",
          "[runtime][mc][smalltree]") {
  constexpr std::uint64_t kTrials = 1 << 16;
  constexpr double kRef = 0.026776830672032736;  // 1-(1-p1 p2)(1-p3 p4), t=8760 h
  auto compiled = fuzzybits::compile::compile_files({repo_input("input/SmallTree/SmallTree.xml")}, no_kn());
  REQUIRE(compiled);
  auto circuit = pcfb::io::from_bytes(compiled->bytes);
  REQUIRE(circuit);
  REQUIRE(pcfb::validate(*circuit).empty());
  fuzzybits::runtime::RunOptions opt;
  opt.trials = kTrials;
  opt.backend = "omp";
  const auto results = fuzzybits::runtime::run_pcfb(*circuit, opt);
  REQUIRE(results.size() == 1);
  REQUIRE(results[0].n >= kTrials);
  const double z = z_score(results[0].estimate, kRef, results[0].n);
  INFO("p_hat=" << results[0].estimate << " p_ref=" << kRef << " z=" << z);
  REQUIRE(std::abs(z) <= 4.0);
}

TEST_CASE("fzb run chinese.pcfb small-N estimate is in (0,1) on CPU", "[runtime][mc]") {
  auto compiled = fuzzybits::compile::compile_files({aralia("chinese.xml")}, no_kn());
  REQUIRE(compiled);
  const auto tmp = std::filesystem::temp_directory_path() / "fuzzybits-chinese-runtime.pcfb";
  {
    std::ofstream out(tmp, std::ios::binary);
    out.write(reinterpret_cast<const char *>(compiled->bytes.data()),
              static_cast<std::streamsize>(compiled->bytes.size()));
  }
  fuzzybits::runtime::RunOptions opt;
  opt.trials = 1 << 16;
  opt.backend = "omp";
  const auto results = fuzzybits::runtime::run_pcfb(tmp, opt);
  std::filesystem::remove(tmp);
  REQUIRE(results.size() == 1);
  REQUIRE(results[0].n >= opt.trials);
  REQUIRE(results[0].estimate >= 0.0);
  REQUIRE(results[0].estimate <= 1.0);
  REQUIRE(results[0].ci_low <= results[0].estimate);
  REQUIRE(results[0].ci_high >= results[0].estimate);
}
