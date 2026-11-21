#include <catch2/catch_test_macros.hpp>

#include "support.h"

#include "compile/exact_eval.h"
#include "io/ftp/descriptor.h"
#include "io/pcfb/descriptor.h"

#include "pcfb/io.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace {

    using namespace pc_passes_test;
    namespace ftp = fuzzybits::io::backend::fuzzybits::ftp;
    namespace writer = fuzzybits::io::backend::fuzzybits::pcfb;

    [[nodiscard]] auto ftp_files(std::filesystem::path const &root) -> std::vector<std::filesystem::path> {
        std::vector<std::filesystem::path> files;
        std::error_code error;
        constexpr auto skip = std::filesystem::directory_options::skip_permission_denied;
        for (auto iterator = std::filesystem::recursive_directory_iterator(root, skip, error);
             !error && iterator != std::filesystem::recursive_directory_iterator(); iterator.increment(error)) {
            auto extension = iterator->path().extension().string();
            std::ranges::transform(extension, extension.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
            std::error_code status;
            if (iterator->is_regular_file(status) && extension == ".ftp") {
                files.push_back(iterator->path());
            }
        }
        std::ranges::sort(files);
        return files;
    }

    [[nodiscard]] auto read_ftp(std::filesystem::path const &file) -> std::optional<pc::circuit> {
        std::ifstream input(file);
        auto read = ftp::read(input);
        if (!read) {
            return std::nullopt;
        }
        auto const *model = pc::circuit_from(read.value().model);
        return model == nullptr ? std::nullopt : std::optional<pc::circuit>(*model);
    }

    [[nodiscard]] auto literal_only(pc::circuit const &model) -> bool {
        return std::ranges::all_of(model.probability_sources(),
                                   [](auto const &source) { return source.kind == pc::probability_source_kind::literal; });
    }

    [[nodiscard]] auto event_count(pc::circuit const &model) -> std::size_t {
        return static_cast<std::size_t>(std::ranges::count(model.op_codes(), pc::op_code::basic_event));
    }

    // The compiled circuit, written to .pcfb, has the input's exact probabilities.
    void require_pcfb_exact(pc::circuit const &input, pc::circuit const &compiled) {
        auto written = writer::write(compiled.to_handle(), writer::write_options{});
        INFO(diagnostics_text(written.diagnostics()));
        REQUIRE(written);
        auto loaded = ::pcfb::io::from_bytes(written.value().bytes);
        REQUIRE(loaded);
        const auto expected = exact_probabilities(input);
        for (std::uint32_t query = 0; query < expected.size(); ++query) {
            INFO("query " << query);
            REQUIRE(std::abs(fuzzybits::compile::exact_probability(*loaded, query) - expected[query]) <= 1.0e-12);
        }
    }

} // namespace

TEST_CASE("C1 output written to .pcfb keeps exact probabilities", "[ir][pc][passes][pcfb]") {
    std::mt19937_64 rng(186);
    for (std::uint32_t iteration = 0; iteration < 150; ++iteration) {
        INFO("iteration " << iteration);
        const auto input = random_circuit(rng);
        const auto compiled = passes::compile(input, passes::compile_level::c1);
        INFO(diagnostics_text(compiled.diagnostics));
        REQUIRE(compiled.ok());
        require_pcfb_exact(input, compiled.output);
    }
}

TEST_CASE("C1 on public FTP fixtures: fewer or equal edges, same probabilities", "[ir][pc][passes][ftp]") {
    const auto root = std::filesystem::path(PROJECT_SOURCE_DIR);
    std::vector<std::filesystem::path> files;
    for (auto const &directory : {root / "input" / "ftp", root / "src" / "io" / "ftp" / "tests" / "testdata"}) {
        const auto found = ftp_files(directory);
        files.insert(files.end(), found.begin(), found.end());
    }
    REQUIRE_FALSE(files.empty());
    std::size_t compiled_files = 0;
    for (auto const &file : files) {
        INFO(file.lexically_relative(root).string());
        const auto input = read_ftp(file);
        if (!input) {
            continue;
        }
        const auto compiled = passes::compile(*input, passes::compile_level::c1);
        INFO(diagnostics_text(compiled.diagnostics));
        REQUIRE(compiled.ok());
        REQUIRE(compiled.edges_after <= compiled.edges_before);
        ++compiled_files;
        if (literal_only(*input) && event_count(*input) <= 20) {
            require_close(exact_probabilities(*input), exact_probabilities(compiled.output));
            require_close(hull_probabilities(*input),
                          hull_probabilities(
                              passes::compile(*input, passes::compile_level::c1,
                                              passes::pipeline_options{.semantics = std::vector(input->targets().size(),
                                                                                                passes::target_semantics::monotone_hull)})
                                  .output));
            require_pcfb_exact(*input, compiled.output);
        }
    }
    REQUIRE(compiled_files > 0);
}

// Aggregates only: no names, paths or values of private models are printed.
TEST_CASE("C1 on private FTP models reports aggregate node and edge counts", "[ir][pc][passes][private]") {
    const char *directory = std::getenv("FUZZYBITS_PRIVATE_FIXTURES_DIR");
    if (directory == nullptr || directory[0] == '\0') {
        SUCCEED("FUZZYBITS_PRIVATE_FIXTURES_DIR not set");
        return;
    }
    std::uint64_t models = 0;
    std::uint64_t unread = 0;
    std::uint64_t nodes = 0;
    std::uint64_t edges = 0;
    std::uint64_t boolean_nodes = 0;
    std::uint64_t boolean_edges = 0;
    std::uint64_t hull_nodes = 0;
    std::uint64_t hull_edges = 0;
    double seconds = 0.0;
    for (auto const &file : ftp_files(directory)) {
        const auto input = read_ftp(file);
        if (!input) {
            ++unread;
            continue;
        }
        ++models;
        nodes += input->node_count();
        edges += input->edge_count();

        const auto start = std::chrono::steady_clock::now();
        const auto boolean = passes::compile(*input, passes::compile_level::c1);
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        REQUIRE(boolean.ok());
        REQUIRE(boolean.edges_after <= boolean.edges_before);
        boolean_nodes += boolean.nodes_after;
        boolean_edges += boolean.edges_after;

        const auto hull = passes::compile(
            *input, passes::compile_level::c1,
            passes::pipeline_options{.semantics = std::vector(input->targets().size(), passes::target_semantics::monotone_hull)});
        REQUIRE(hull.ok());
        REQUIRE(hull.edges_after <= hull.edges_before);
        hull_nodes += hull.nodes_after;
        hull_edges += hull.edges_after;
    }
    std::cout << "private FTP aggregate: models " << models << " (unreadable " << unread << "), raw nodes " << nodes << ", raw edges "
              << edges << "; C1 boolean nodes " << boolean_nodes << ", edges " << boolean_edges << " (" << seconds
              << " s); C1 monotone_hull nodes " << hull_nodes << ", edges " << hull_edges << '\n';
}
