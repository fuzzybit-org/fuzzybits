#include <catch2/catch_test_macros.hpp>

#include "query_guard_files.h"
#include "support.h"

#include "compile/ftp/tests/support.h"
#include "io/pcfb/descriptor.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/op_code.h"

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
        const auto command = quoted(FZB_PATH) + " " + arguments + " > " + quoted(outPath.string()) + " 2> " + quoted(errPath.string());
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
        auto read = ftp_lowering_test::lower_text({std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()});
        INFO(diagnosticsText(read.diagnostics()));
        REQUIRE(read);
        auto const *model = &read.value().circuit;
        if (!std::ranges::all_of(model->probability_sources(),
                                 [](auto const &source) { return source.kind == pc::probability_source_kind::literal; })) {
            continue;
        }

        const auto written = writer::write(model->to_handle(), writer::write_options{.model_name = file.stem().string()});
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
    const auto parity = builder.add_gate(pc::op_code::xor_gate, {edge{.source = e[0]}, edge{.source = e[1], .complement = true}});
    const auto nand = builder.add_gate(pc::op_code::nand_gate, {edge{.source = e[2]}, edge{.source = e[3]}});
    const auto range = builder.add_cardinality(
        1, 2, {edge{.source = e[0]}, edge{.source = e[3], .complement = true}, edge{.source = e[4]}, edge{.source = e[5]}});
    const auto exactly = builder.add_cardinality(2, 2, {edge{.source = e[1]}, edge{.source = e[2]}, edge{.source = e[5], .complement = true}});
    const auto atMost = builder.add_cardinality(0, 1, {edge{.source = e[0]}, edge{.source = e[1]}, edge{.source = e[2]}});
    const auto atLeast = builder.add_cardinality(2, 3, {edge{.source = range}, edge{.source = e[3]}, edge{.source = e[4]}});
    const auto contradiction = builder.add_and({edge{.source = e[4]}, edge{.source = e[4], .complement = true}});
    const auto equivalent = builder.add_gate(pc::op_code::xnor_gate, {edge{.source = e[4]}, edge{.source = e[5], .complement = true}});
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

    const auto run = runFzb("run " + quoted(path.string()) + " --trials 1048576 --backend omp --json", scratch);
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

TEST_CASE("fzb inspect --exact follows the query kind and leaves Boolean output unchanged", "[io][pcfb][cli][guard]") {
    const auto scratch = scratchDirectory("guard-exact");
    const auto hull = scratch / "hull.pcfb";
    const auto mixed = scratch / "mixed.pcfb";
    const auto boolean = scratch / "boolean.pcfb";
    query_guard_files::write(hull, query_guard_files::hull_only());
    query_guard_files::write(mixed, query_guard_files::mixed());
    query_guard_files::write(boolean, query_guard_files::boolean_only());

    // The golden Bytes of a Boolean answer: `exact_probability`, 16 significant digits, and the
    // same summary before it, whatever else the file holds.
    const auto expected_boolean = [&](std::filesystem::path const &path) {
        const auto output = runFzb("inspect " + quoted(path.string()) + " --exact", scratch);
        INFO(output.out << output.err);
        REQUIRE(output.status == 0);
        const auto line = output.out.substr(output.out.rfind("exact_probability"));
        CHECK(line == "exact_probability  1.2500000000000000e-01\n");
        CHECK(output.out.find("exact_hull_probability") == std::string::npos);
        CHECK(output.err.find("validate:") == std::string::npos);
        return output.out;
    };
    const auto golden = expected_boolean(boolean);
    // Query 0 of the mixed file is the Boolean one, so it prints the same line; the
    // summary in front of it differs (two queries instead of one).
    const auto tail = [](std::string const &text) {
        const auto start = text.rfind("\nexact_probability");
        return start == std::string::npos ? text : text.substr(start + 1);
    };
    CHECK(tail(expected_boolean(mixed)) == tail(golden));

    // A hull query is answered instead, by the subset-OR transform. This AND of two events has
    // ↑g = g, so the value equals the Boolean one.
    for (auto const &path : {hull, mixed}) {
        const auto answered = runFzb("inspect " + quoted(path.string()) + " --validate --exact --query " +
                                         query_guard_files::hull_name + " --kind hull",
                                     scratch);
        INFO(answered.out << answered.err);
        REQUIRE(answered.status == 0);
        CHECK(answered.err.find("validate: ok") != std::string::npos);
        CHECK(answered.out.find("exact_probability") == std::string::npos);
        CHECK(std::abs(numberAfter(answered.out, "exact_hull_probability") - 0.125) < 1.0e-12);
    }
    // A hull file has one hull query, so it is the default even when the file also has a
    // Boolean query named differently.
    const auto by_default = runFzb("inspect " + quoted(hull.string()) + " --exact", scratch);
    INFO(by_default.out << by_default.err);
    REQUIRE(by_default.status == 0);
    CHECK(std::abs(numberAfter(by_default.out, "exact_hull_probability") - 0.125) < 1.0e-12);

    // The summary is still printed for a hull file, with or without a query to evaluate.
    const auto summary = runFzb("inspect " + quoted(hull.string()), scratch);
    INFO(summary.out << summary.err);
    CHECK(summary.status == 0);

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
        const auto run = runFzb("run " + quoted(path.string()) + " --trials 1024 --backend omp --json --no-progress", scratch);
        INFO(run.out << run.err);
        CHECK(run.status != 0);
        CHECK(run.err.find("pcfb.unsupported-query-kind") != std::string::npos);
        CHECK(run.err.find(query_guard_files::hull_name) != std::string::npos);
        CHECK(run.out.find("estimate") == std::string::npos);
    }
    // Readers accept any minor of major 1 (pcfb spec 1.1 section 2).
    const auto later = runFzb("run " + quoted(future.string()) + " --trials 1024 --backend omp --json --no-progress", scratch);
    INFO(later.out << later.err);
    CHECK(later.status == 0);
    CHECK(later.out.find("\"name\": \"TOP\"") != std::string::npos);
    std::filesystem::remove_all(scratch);
}

TEST_CASE("fzb inspect validates a written mixed-kind file and lists the recorded quantification", "[io][pcfb][cli][kind]") {
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
    options.target_kinds = {{pc::target_id(0), {writer::query_kind::probability, writer::query_kind::monotone_hull_probability}}};
    options.quantification = {{pc::target_id(0), writer::target_quantification{.literal_mode = "atom_split", .max_initiators_per_cutset = 1}}};
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
    CHECK(inspected.out.find("compile.query.1.negation  monotone_hull") != std::string::npos);
    CHECK(inspected.out.find("compile.query.1.max_initiators_per_cutset  1") != std::string::npos);

    // The evaluator follows the kind: query 0 is the Boolean AND, query 1 its hull.
    const auto exact = runFzb("inspect " + quoted(path.string()) + " --exact", scratch);
    INFO(exact.out << exact.err);
    REQUIRE(exact.status == 0);
    CHECK(std::abs(numberAfter(exact.out, "exact_probability") - 0.125) < 1.0e-12);
    const auto hull = runFzb("inspect " + quoted(path.string()) + " --exact --query TOP --kind hull", scratch);
    INFO(hull.out << hull.err);
    REQUIRE(hull.status == 0);
    CHECK(std::abs(numberAfter(hull.out, "exact_hull_probability") - 0.125) < 1.0e-12);
    // The name alone is ambiguous: the target is written as two kinds.
    const auto ambiguous = runFzb("inspect " + quoted(path.string()) + " --exact --query TOP", scratch);
    CHECK(ambiguous.status != 0);
    CHECK(ambiguous.err.find("probability and hull") != std::string::npos);
    std::filesystem::remove_all(scratch);
}
TEST_CASE("fzb inspect --hull-check evaluates named samples of a hull query", "[io][pcfb][cli][hull]") {
    // g = A AND XOR(B, NOT C). x = {A} satisfies it with B and C low, so the bounds decide
    // that lane; x = {A, C} is a hit only with y_C = 0, which the bounds cannot see, so a
    // later tier decides it. D is outside the target's cone.
    pc::builder builder;
    const auto a = builder.add_basic_event(builder.add_literal_probability(0.1));
    const auto b = builder.add_basic_event(builder.add_literal_probability(0.2));
    const auto c = builder.add_basic_event(builder.add_literal_probability(0.3));
    const auto d = builder.add_basic_event(builder.add_literal_probability(0.4));
    const auto not_c = builder.add_not(edge{.source = c});
    const auto parity = builder.add_gate(pc::op_code::xor_gate, {edge{.source = b}, edge{.source = not_c}});
    const auto g = builder.add_and({edge{.source = a}, edge{.source = parity}});
    (void)builder.add_target(g);
    const auto plain = builder.add_and({edge{.source = a}, edge{.source = d}});
    (void)builder.add_target(plain);
    auto sealed = std::move(builder).seal();
    REQUIRE(sealed.ok());

    writer::write_options options;
    options.target_names = {{pc::target_id(0), "TOP_HULL"}, {pc::target_id(1), "PLAIN"}};
    options.target_kinds = {{pc::target_id(0), {writer::query_kind::monotone_hull_probability}}};
    options.node_names = {{a, "A"}, {b, "B"}, {c, "C"}, {d, "D"}, {not_c, "NOT_C"}, {parity, "PARITY"}, {g, "TOP"}};
    const auto written = writer::write(sealed.output.to_handle(), options);
    INFO(diagnosticsText(written.diagnostics()));
    REQUIRE(written);
    const auto scratch = scratchDirectory("hull-check");
    const auto path = scratch / "hull.pcfb";
    writeFile(path, written.value().bytes);

    SECTION("one set per argument") {
        // x = {A}: A satisfies the OR and C is false, so g holds outright.
        const auto decided = runFzb("inspect " + quoted(path.string()) + " --hull-check A --witness", scratch);
        INFO(decided.out << decided.err);
        REQUIRE(decided.status == 0);
        CHECK(decided.out.find("hull_check  hit=1  L=1  U=1  tier=bounds  residual=0") != std::string::npos);
        CHECK(decided.out.find("witness  A\n") != std::string::npos);
        // x = {A, C}: A satisfies the OR, but the deletion may be dropped (y_C = 0),
        // so the bounds leave the lane open and a later tier decides it.
        const auto open = runFzb("inspect " + quoted(path.string()) + " --hull-check A,C --witness", scratch);
        INFO(open.out << open.err);
        REQUIRE(open.status == 0);
        CHECK(open.out.find("hull_check  hit=1  L=0  U=1") != std::string::npos);
        CHECK(open.out.find("tier=bounds") == std::string::npos);
        // The witness is {A}: C is true in the sample but need not be for the hit.
        CHECK(open.out.find("witness  A\n") != std::string::npos);
        // x = {C} is a miss: the OR has no true event to work with.
        const auto only_c = runFzb("inspect " + quoted(path.string()) + " --hull-check C", scratch);
        INFO(only_c.out << only_c.err);
        REQUIRE(only_c.status == 0);
        CHECK(only_c.out.find("hit=0") != std::string::npos);
        const auto miss = runFzb("inspect " + quoted(path.string()) + " --hull-check D", scratch);
        INFO(miss.out << miss.err);
        REQUIRE(miss.status == 0);
        CHECK(miss.out.find("hit=0") != std::string::npos);
        CHECK(miss.out.find("witness") == std::string::npos);
    }
    SECTION("events outside the cone are ignored") {
        const auto with_d = runFzb("inspect " + quoted(path.string()) + " --hull-check A,D --witness", scratch);
        const auto without_d = runFzb("inspect " + quoted(path.string()) + " --hull-check A --witness", scratch);
        INFO(with_d.out << with_d.err);
        REQUIRE(with_d.status == 0);
        REQUIRE(without_d.status == 0);
        CHECK(with_d.out == without_d.out);
    }
    SECTION("the empty set is a miss, and a hit only when g(empty) is TRUE") {
        const auto empty = runFzb("inspect " + quoted(path.string()) + " --hull-check ''", scratch);
        INFO(empty.out << empty.err);
        REQUIRE(empty.status == 0);
        CHECK(empty.out.find("hit=0") != std::string::npos);
    }
    SECTION("a file of sets, with `-` for the empty set and blank lines skipped") {
        const auto sets = scratch / "sets.txt";
        {
            std::ofstream out(sets, std::ios::binary);
            out << "A,B\n\n-\nD\nC\n";
            REQUIRE(out);
        }
        const auto listed = runFzb("inspect " + quoted(path.string()) + " --hull-check @" + quoted(sets.string()) +
                                       " --witness --json",
                                   scratch);
        INFO(listed.out << listed.err);
        REQUIRE(listed.status == 0);
        CHECK(listed.out.find("\"hit\": true") != std::string::npos);
        CHECK(listed.out.find("\"hit\": false, \"lower\": false, \"upper\": false") != std::string::npos);
        CHECK(listed.out.find("\"witness\": [\"A\"]") != std::string::npos);
        CHECK(listed.out.find("\"query\": {\"index\": 0, \"name\": \"TOP_HULL\", \"kind\": \"hull\"}") !=
              std::string::npos);
        // The JSON is an array of four objects, one per set.
        CHECK(listed.out.find("\"hit\":") != std::string::npos);
        CHECK(listed.out.rfind("\"hit\":") != listed.out.find("\"hit\":"));
        CHECK(listed.out.find("\"residual_variables\": 0, \"witness\": null") != std::string::npos);
    }
    SECTION("an empty file of sets is an error") {
        const auto sets = scratch / "empty.txt";
        {
            std::ofstream out(sets, std::ios::binary);
            out << "\n\n";
            REQUIRE(out);
        }
        const auto listed =
            runFzb("inspect " + quoted(path.string()) + " --hull-check @" + quoted(sets.string()), scratch);
        CHECK(listed.status != 0);
        CHECK(listed.err.find("lists no sets") != std::string::npos);
    }
    SECTION("an unknown name or a gate name is an error naming it") {
        const auto unknown = runFzb("inspect " + quoted(path.string()) + " --hull-check NOPE", scratch);
        CHECK(unknown.status != 0);
        CHECK(unknown.err.find("'NOPE'") != std::string::npos);
        const auto gate = runFzb("inspect " + quoted(path.string()) + " --hull-check PARITY", scratch);
        CHECK(gate.status != 0);
        CHECK(gate.err.find("'PARITY' is not a basic event") != std::string::npos);
    }
    SECTION("a Boolean query is refused") {
        const auto plain_only = runFzb("inspect " + quoted(path.string()) + " --hull-check A --query PLAIN", scratch);
        CHECK(plain_only.status != 0);
        CHECK(plain_only.err.find("not hull") != std::string::npos);
    }
    SECTION("the only hull query is the default") {
        const auto defaulted = runFzb("inspect " + quoted(path.string()) + " --hull-check A", scratch);
        const auto named = runFzb("inspect " + quoted(path.string()) + " --hull-check A --query TOP_HULL", scratch);
        REQUIRE(defaulted.status == 0);
        REQUIRE(named.status == 0);
        CHECK(defaulted.out == named.out);
    }
    SECTION("event names map to lane bits even when a gate precedes the events") {
        // A gate is node 0, so the read circuit's node ids and event ordinals differ;
        // a wrong mapping would decide a different lane.
        pc::builder builder;
        const auto seed = builder.add_basic_event(builder.add_literal_probability(0.5));
        const auto early = builder.add_or({edge{.source = seed}});
        const auto a = builder.add_basic_event(builder.add_literal_probability(0.5));
        const auto b = builder.add_basic_event(builder.add_literal_probability(0.5));
        const auto g = builder.add_and({edge{.source = early}, edge{.source = a}, edge{.source = b}});
        (void)builder.add_target(g);
        auto sealed = std::move(builder).seal();
        REQUIRE(sealed.ok());
        writer::write_options named_options;
        named_options.target_names = {{pc::target_id(0), "G"}};
        named_options.target_kinds = {{pc::target_id(0), {writer::query_kind::monotone_hull_probability}}};
        named_options.node_names = {{seed, "SEED"}, {a, "A"}, {b, "B"}, {early, "EARLY"}, {g, "GATE"}};
        const auto written = writer::write(sealed.output.to_handle(), named_options);
        INFO(diagnosticsText(written.diagnostics()));
        REQUIRE(written);
        const auto gate_first = scratch / "gate-first.pcfb";
        writeFile(gate_first, written.value().bytes);

        const auto all = runFzb("inspect " + quoted(gate_first.string()) + " --hull-check SEED,A,B --json", scratch);
        INFO(all.out << all.err);
        REQUIRE(all.status == 0);
        CHECK(all.out.find("\"hit\": true, \"lower\": true, \"upper\": true") != std::string::npos);
        // Without SEED the early OR is false, so the target is a miss.
        const auto later = runFzb("inspect " + quoted(gate_first.string()) + " --hull-check A,B --json", scratch);
        INFO(later.out << later.err);
        REQUIRE(later.status == 0);
        CHECK(later.out.find("\"hit\": false") != std::string::npos);
    }
    SECTION("--json without --hull-check still prints the summary") {
        const auto summary = runFzb("inspect " + quoted(path.string()) + " --json", scratch);
        INFO(summary.out << summary.err);
        REQUIRE(summary.status == 0);
        CHECK(summary.out.find("\"schema_version\": 65538") != std::string::npos);
    }
    SECTION("nonsensical combinations are usage errors") {
        for (const auto &arguments :
             {std::string("--witness --exact"), std::string("--hull-check A --exact"), std::string("--kind hull"),
              std::string("--hull-check A --kind hull"), std::string("--query TOP_HULL")}) {
            const auto refused = runFzb("inspect " + quoted(path.string()) + " " + arguments, scratch);
            INFO(arguments << ": " << refused.err);
            CHECK(refused.status != 0);
            CHECK(refused.err.find("Option error:") != std::string::npos);
        }
    }
    std::filesystem::remove_all(scratch);
}
