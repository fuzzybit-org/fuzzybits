#include <catch2/catch_test_macros.hpp>

#include "query_guard_files.h"
#include "support.h"

#include "compile/ftp/tests/support.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/op_code.h"

#include "runtime/tests/test_backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef FZB_PATH
#error "FZB_PATH must be provided by CMake"
#endif

namespace {

using namespace pcfb_writer_test;
namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;
using edge = pc::edge_ref;

[[nodiscard]] auto quoted(std::string const &text) -> std::string {
#ifdef _WIN32
    return "\"" + text + "\"";
#else
    return "'" + text + "'";
#endif
}

// Runs `fzb <arguments>` with stdout and stderr captured to files.
struct FzbRun final {
    int status = -1;
    std::string out;
    std::string err;
};

[[nodiscard]] auto readText(std::filesystem::path const &path) -> std::string {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

[[nodiscard]] auto runFzb(std::string const &arguments, std::filesystem::path const &scratch) -> FzbRun {
    const auto outPath = scratch / "fzb.out";
    const auto errPath = scratch / "fzb.err";
    const auto command =
        quoted(FZB_PATH) + " " + arguments + " > " + quoted(outPath.string()) + " 2> " + quoted(errPath.string());
#ifdef _WIN32
    const int status = std::system(("cmd /S /C \"" + command + "\"").c_str());
#else
    const int status = std::system(command.c_str());
#endif
    return FzbRun{.status = status, .out = readText(outPath), .err = readText(errPath)};
}

[[nodiscard]] auto scratchDirectory(std::string const &name) -> std::filesystem::path {
    const auto path = std::filesystem::temp_directory_path() / ("fuzzybits-io-pcfb-" + name);
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}

void writeFile(std::filesystem::path const &path, std::vector<std::byte> const &bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<char const *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(output);
}

[[nodiscard]] auto numberAfter(std::string const &text, std::string const &key) -> double {
    const auto position = text.rfind(key);
    INFO(text);
    REQUIRE(position != std::string::npos);
    return std::strtod(text.c_str() + position + key.size(), nullptr);
}

} // namespace

TEST_CASE("fzb inspect validates and exactly evaluates written FTP fixtures", "[io][pcfb][cli]") {
    const auto files = ftpCorpusFiles();
    REQUIRE_FALSE(files.empty());
    const auto scratch = scratchDirectory("inspect");

    std::size_t exact = 0;
    for (auto const &file : files) {
        INFO(file.filename().string());
        std::ifstream input(file, std::ios::binary);
        REQUIRE(input);
        auto read =
            ftp_lowering_test::lower_text({std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()});
        INFO(diagnosticsText(read.diagnostics()));
        REQUIRE(read);
        auto const *model = &read.value().circuit;
        if (!std::ranges::all_of(model->probability_sources(), [](auto const &source) {
                return source.kind == pc::probability_source_kind::literal;
            })) {
            continue;
        }

        const auto written =
            writer::write(model->to_handle(), writer::write_options{.model_name = file.stem().string()});
        INFO(diagnosticsText(written.diagnostics()));
        REQUIRE(written);
        const auto path = scratch / (file.stem().string() + ".pcfb");
        writeFile(path, written.value().bytes);

        const auto validated = runFzb("inspect " + quoted(path.string()) + " --validate", scratch);
        INFO(validated.out << validated.err);
        REQUIRE(validated.status == 0);
        REQUIRE(validated.err.find("validate: ok") != std::string::npos);

        const auto events = std::ranges::count(model->op_codes(), pc::op_code::basic_event);
        if (events > 30) {
            continue;
        }
        const auto evaluated = runFzb("inspect " + quoted(path.string()) + " --exact", scratch);
        INFO(evaluated.out << evaluated.err);
        REQUIRE(evaluated.status == 0);
        const auto probability = numberAfter(evaluated.out, "exact_probability");
        if (events <= 20) {
            const auto expected = referenceProbabilities(*model).front();
            INFO("fzb inspect --exact " << probability << " reference " << expected);
            REQUIRE(std::abs(probability - expected) <= 1.0e-12);
        }
        ++exact;
    }
    REQUIRE(exact > 0);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb run estimates a written pcfb within 4 standard errors of the exact value", "[mc][io][pcfb]") {
    pc::builder builder;
    const std::vector<double> probabilities{0.3, 0.45, 0.2, 0.6, 0.15, 0.5};
    std::vector<pc::bool_node_id> e;
    for (const auto p : probabilities) {
        e.push_back(builder.add_basic_event(builder.add_literal_probability(p)));
    }
    const auto yes = builder.add_constant(true);
    const auto no = builder.add_house_event(false);
    const auto parity =
        builder.add_gate(pc::op_code::xor_gate, {edge{.source = e[0]}, edge{.source = e[1], .complement = true}});
    const auto nand = builder.add_gate(pc::op_code::nand_gate, {edge{.source = e[2]}, edge{.source = e[3]}});
    const auto range = builder.add_cardinality(
        1, 2,
        {edge{.source = e[0]}, edge{.source = e[3], .complement = true}, edge{.source = e[4]}, edge{.source = e[5]}});
    const auto exactly = builder.add_cardinality(
        2, 2, {edge{.source = e[1]}, edge{.source = e[2]}, edge{.source = e[5], .complement = true}});
    const auto atMost =
        builder.add_cardinality(0, 1, {edge{.source = e[0]}, edge{.source = e[1]}, edge{.source = e[2]}});
    const auto atLeast =
        builder.add_cardinality(2, 3, {edge{.source = range}, edge{.source = e[3]}, edge{.source = e[4]}});
    const auto contradiction = builder.add_and({edge{.source = e[4]}, edge{.source = e[4], .complement = true}});
    const auto equivalent =
        builder.add_gate(pc::op_code::xnor_gate, {edge{.source = e[4]}, edge{.source = e[5], .complement = true}});
    const auto inverted = builder.add_not(edge{.source = nand, .complement = true});
    const auto neither = builder.add_gate(pc::op_code::nor_gate, {edge{.source = atMost}, edge{.source = no}});
    const auto top = builder.add_or({
        edge{.source = builder.add_and({edge{.source = parity}, edge{.source = range}, edge{.source = yes}})},
        edge{.source = builder.add_and({edge{.source = inverted}, edge{.source = exactly}})},
        edge{.source = builder.add_and({edge{.source = equivalent}, edge{.source = neither}})},
        edge{.source = builder.add_and({edge{.source = atLeast}, edge{.source = e[1], .complement = true}})},
        edge{.source = contradiction},
    });
    (void)builder.add_target(top);
    auto sealed = std::move(builder).seal();
    INFO(diagnosticsText(sealed.diagnostics));
    REQUIRE(sealed.ok());
    const auto expected = referenceProbabilities(sealed.output).front();
    REQUIRE(expected > 0.05);
    REQUIRE(expected < 0.95);

    const auto written = writer::write(sealed.output.to_handle(), writer::write_options{.model_name = "io_pcfb_mc"});
    INFO(diagnosticsText(written.diagnostics()));
    REQUIRE(written);
    const auto scratch = scratchDirectory("run");
    const auto path = scratch / "io_pcfb_mc.pcfb";
    writeFile(path, written.value().bytes);

    const auto run = runFzb(
        "run " + quoted(path.string()) + " --trials 1048576" + fuzzybits::testing::backend_flag() + " --json", scratch);
    INFO(run.out << run.err);
    REQUIRE(run.status == 0);
    const auto estimate = numberAfter(run.out, "\"estimate\": ");
    const auto trials = numberAfter(run.out, "\"n\": ");
    REQUIRE(trials >= 1048576.0);
    const auto standardError = std::sqrt(expected * (1.0 - expected) / trials);
    INFO("estimate " << estimate << " exact " << expected << " n " << trials << " standard error " << standardError);
    REQUIRE(std::abs(estimate - expected) <= 4.0 * standardError);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb inspect --exact refuses monotone-hull queries but still summarises them", "[io][pcfb][cli][guard]") {
    const auto scratch = scratchDirectory("guard-exact");
    const auto hull = scratch / "hull.pcfb";
    const auto mixed = scratch / "mixed.pcfb";
    const auto boolean = scratch / "boolean.pcfb";
    query_guard_files::write(hull, query_guard_files::hull_only());
    query_guard_files::write(mixed, query_guard_files::mixed());
    query_guard_files::write(boolean, query_guard_files::boolean_only());

    for (auto const &path : {hull, mixed}) {
        const auto refused = runFzb("inspect " + quoted(path.string()) + " --validate --exact", scratch);
        INFO(refused.out << refused.err);
        CHECK(refused.status != 0);
        CHECK(refused.err.find("validate: ok") != std::string::npos);
        CHECK(refused.err.find("pcfb.unsupported-query-kind") != std::string::npos);
        CHECK(refused.err.find(query_guard_files::hull_name) != std::string::npos);
        CHECK(refused.out.find("exact_probability") == std::string::npos);
    }

    const auto summary = runFzb("inspect " + quoted(hull.string()), scratch);
    INFO(summary.out << summary.err);
    CHECK(summary.status == 0);

    const auto accepted = runFzb("inspect " + quoted(boolean.string()) + " --exact", scratch);
    INFO(accepted.out << accepted.err);
    REQUIRE(accepted.status == 0);
    CHECK(std::abs(numberAfter(accepted.out, "exact_probability") - 0.125) < 1.0e-12);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb run refuses monotone-hull queries before sampling and accepts later minors", "[mc][io][pcfb][guard]") {
    const auto scratch = scratchDirectory("guard-run");
    const auto hull = scratch / "hull.pcfb";
    const auto mixed = scratch / "mixed.pcfb";
    const auto future = scratch / "future.pcfb";
    query_guard_files::write(hull, query_guard_files::hull_only());
    query_guard_files::write(mixed, query_guard_files::mixed());
    query_guard_files::write(future, query_guard_files::boolean_with_minor(3));

    for (auto const &path : {hull, mixed}) {
        const auto run = runFzb("run " + quoted(path.string()) + " --trials 1024" + fuzzybits::testing::backend_flag() +
                                    " --json --no-progress",
                                scratch);
        INFO(run.out << run.err);
        CHECK(run.status != 0);
        CHECK(run.err.find("pcfb.unsupported-query-kind") != std::string::npos);
        CHECK(run.err.find(query_guard_files::hull_name) != std::string::npos);
        CHECK(run.out.find("estimate") == std::string::npos);
    }
    // Readers accept any minor of major 1 (pcfb spec 1.1 section 2).
    const auto later = runFzb("run " + quoted(future.string()) + " --trials 1024" + fuzzybits::testing::backend_flag() +
                                  " --json --no-progress",
                              scratch);
    INFO(later.out << later.err);
    CHECK(later.status == 0);
    CHECK(later.out.find("\"name\": \"TOP\"") != std::string::npos);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb inspect validates a written mixed-kind file and lists the recorded quantification",
          "[io][pcfb][cli][kind]") {
    pc::builder builder;
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.25));
    const auto b = builder.add_basic_event(builder.add_literal_probability(0.5));
    const auto both = builder.add_and({edge{.source = a}, edge{.source = b}});
    (void)builder.add_target(both);
    (void)builder.add_target(a);
    auto sealed = std::move(builder).seal();
    REQUIRE(sealed.ok());

    writer::write_options options;
    options.target_names = {{pc::target_id(0), "TOP"}, {pc::target_id(1), "A"}};
    options.target_kinds = {
        {pc::target_id(0), {writer::query_kind::probability, writer::query_kind::monotone_hull_probability}}};
    options.quantification = {
        {pc::target_id(0), writer::target_quantification{.literal_mode = "split", .max_initiators_per_cutset = 1}}};
    const auto written = writer::write(sealed.output.to_handle(), options);
    INFO(diagnosticsText(written.diagnostics()));
    REQUIRE(written);
    const auto scratch = scratchDirectory("kinds");
    const auto path = scratch / "mixed.pcfb";
    writeFile(path, written.value().bytes);

    const auto inspected = runFzb("inspect " + quoted(path.string()) + " --validate", scratch);
    INFO(inspected.out << inspected.err);
    REQUIRE(inspected.status == 0);
    CHECK(inspected.err.find("validate: ok") != std::string::npos);
    CHECK(inspected.out.find("schema_version  0x10002") != std::string::npos);
    CHECK(inspected.out.find("compile.query.1.literal_mode  split") != std::string::npos);
    CHECK(inspected.out.find("compile.query.1.max_initiators_per_cutset  1") != std::string::npos);

    // The runtime and the exact evaluator refuse the hull query instead of answering it as a probability.
    const auto exact = runFzb("inspect " + quoted(path.string()) + " --exact", scratch);
    CHECK(exact.status != 0);
    CHECK(exact.err.find("pcfb.unsupported-query-kind") != std::string::npos);
    std::filesystem::remove_all(scratch);
}
