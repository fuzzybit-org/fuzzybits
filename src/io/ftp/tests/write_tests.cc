#include <catch2/catch_test_macros.hpp>

#include "api/diagnostics.h"
#include "compile/ftp/tests/support.h"
#include "io/ftp/descriptor.h"
#include "io/formats.h"
#include "ir/pc/builder.h"
#include "ir/pc/circuit.h"
#include "ir/pc/edge.h"
#include "ir/pc/passes/passes.h"
#include "ir/pc/probability.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

    [[nodiscard]] auto diagnosticsText(fuzzybits::api::diagnostic_list const &diagnostics) -> std::string {
        std::ostringstream output;
        for (auto const &diagnostic : diagnostics.items()) {
            output << diagnostic.code << ": " << diagnostic.message << '\n';
        }
        return output.str();
    }

    [[nodiscard]] auto checksum(std::string_view text) -> std::uint64_t {
        auto value = std::uint64_t{14695981039346656037ULL};
        for (const auto character : text) {
            value ^= static_cast<unsigned char>(character);
            value *= std::uint64_t{1099511628211ULL};
        }
        return value;
    }

    [[nodiscard]] auto hasFtpExtension(std::filesystem::path const &path) -> bool {
        auto extension = path.extension().string();
        std::ranges::transform(extension, extension.begin(), [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return extension == ".ftp";
    }

    [[nodiscard]] auto projectRoot() -> std::filesystem::path {
        return std::filesystem::path(PROJECT_SOURCE_DIR);
    }

    void collectFtpFiles(std::filesystem::path const &root, std::vector<std::filesystem::path> &files) {
        std::error_code error;
        auto iterator = std::filesystem::directory_iterator(root, std::filesystem::directory_options::skip_permission_denied, error);
        if (error) {
            return;
        }

        for (const auto end = std::filesystem::directory_iterator{}; iterator != end; iterator.increment(error)) {
            if (error) {
                error.clear();
                continue;
            }

            auto const &entry = *iterator;
            if (entry.is_directory(error)) {
                if (!error) {
                    collectFtpFiles(entry.path(), files);
                }
                error.clear();
                continue;
            }
            if (entry.is_regular_file(error) && hasFtpExtension(entry.path())) {
                files.push_back(entry.path());
            }
            error.clear();
        }
    }

    struct corpus_project final {
        std::string label;
        std::vector<std::filesystem::path> files;
    };

    // Public corpus: every file under input/ftp/synthetic/ and tests/testdata/
    // is a project. Vendor CAFTA/MHTGR/G-PWR FTP models are not in this public
    // tree (#38); every directory with .ftp files under
    // FUZZYBITS_PRIVATE_FIXTURES_DIR is read as one project, labelled by index only.
    [[nodiscard]] auto ftpCorpus() -> std::vector<corpus_project> {
        std::vector<std::filesystem::path> files;
        collectFtpFiles(projectRoot() / "input" / "ftp" / "synthetic", files);
        collectFtpFiles(projectRoot() / "src" / "io" / "ftp" / "tests" / "testdata", files);
        std::ranges::sort(files);
        std::vector<corpus_project> projects;
        for (auto const &file : files) {
            projects.push_back(corpus_project{.label = file.lexically_relative(projectRoot()).generic_string(), .files = {file}});
        }

        if (const char *privateDir = std::getenv("FUZZYBITS_PRIVATE_FIXTURES_DIR"); privateDir != nullptr && privateDir[0] != '\0') {
            files.clear();
            collectFtpFiles(privateDir, files);
            std::map<std::filesystem::path, std::vector<std::filesystem::path>> byDirectory;
            for (auto const &file : files) {
                byDirectory[file.parent_path()].push_back(file);
            }
            for (auto &[directory, members] : byDirectory) {
                std::ranges::sort(members);
                projects.push_back(corpus_project{.label = "private project " + std::to_string(projects.size()), .files = members});
            }
        }
        return projects;
    }

    [[nodiscard]] auto readProject(corpus_project const &project) -> fuzzybits::api::result<ftp_lowering_test::lowering> {
        std::vector<fuzzybits::io::backend::fuzzybits::ftp::source_text> sources;
        for (auto const &file : project.files) {
            std::ifstream input(file, std::ios::binary);
            REQUIRE(input);
            sources.push_back({.path = file.generic_string(), .text = {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()}});
        }
        return ftp_lowering_test::lower(sources);
    }

    // The writer puts a single-input gate in front of every target and the
    // lowering keeps such gates, so a stable round trip needs them eliminated.
    [[nodiscard]] auto requirePcModel(ftp_lowering_test::lowering const &lowered) -> fuzzybits::ir::pc::circuit {
        const auto eliminated = fuzzybits::ir::pc::passes::eliminate_single_input(lowered.circuit);
        REQUIRE(eliminated.ok());
        auto const &pc = eliminated.output;
        REQUIRE(pc.valid());
        REQUIRE(pc.node_count() > 0);
        REQUIRE_FALSE(pc.targets().empty());
        return pc;
    }

    [[nodiscard]] auto writeOutput(fuzzybits::ir::pc::circuit const &model, std::optional<double> truncationLimit = std::nullopt)
        -> std::string {
        fuzzybits::io::write_request request{
            .format = fuzzybits::io::formats::ftp(),
        };
        if (truncationLimit) {
            request.options.set(fuzzybits::io::backend::fuzzybits::ftp::truncation_limit_option(), *truncationLimit);
        }

        std::ostringstream output;
        const auto written = fuzzybits::io::backend::fuzzybits::ftp::write(model.to_handle(), output, request);
        INFO(diagnosticsText(written.diagnostics()));
        REQUIRE(written);
        return output.str();
    }

} // namespace

TEST_CASE("FTP writer descriptor advertises the FTP format", "[io][ftp]") {
    const auto descriptor = fuzzybits::io::backend::fuzzybits::ftp::writer_descriptor();

    REQUIRE(descriptor.provider == fuzzybits::api::provider_id("fuzzybits"));
    REQUIRE(descriptor.endpoint == fuzzybits::io::backend::fuzzybits::ftp::writer_endpoint_id());
    REQUIRE(descriptor.output_format == fuzzybits::io::formats::ftp());
}

TEST_CASE("FTP writer round-trips the writable PC subset", "[io][ftp]") {
    fuzzybits::ir::pc::builder builder;
    const auto aProbability = builder.add_literal_probability(0.1);
    const auto bProbability = builder.add_literal_probability(0.2);
    const auto a = builder.add_basic_event(aProbability);
    const auto b = builder.add_basic_event(bProbability);
    const auto top = builder.add_and({
        fuzzybits::ir::pc::edge_ref{.source = a},
        fuzzybits::ir::pc::edge_ref{.source = b, .complement = true},
    });
    const auto target = builder.add_target(top);
    (void)target;

    auto finalized = std::move(builder).seal();
    REQUIRE(finalized.ok());

    const auto text = writeOutput(finalized.output, 1.0e-6);

    REQUIRE(text.find("LIMIT") != std::string::npos);
    REQUIRE(text.find("T0 + N") != std::string::npos);
    REQUIRE(text.find(" -E") != std::string::npos);

    auto read = ftp_lowering_test::lower_text(text);
    INFO(diagnosticsText(read.diagnostics()));
    REQUIRE(read);

    auto const *pc = &read.value().circuit;
    REQUIRE(pc->targets().size() == 1);
    REQUIRE(pc->probability_sources().size() == 2);
    // The lowering keeps the single-input gate the writer put in front of the target.
    auto node = pc->targets()[0].node;
    while (pc->operands(node).size() == 1) {
        REQUIRE_FALSE(pc->operands(node)[0].complement);
        node = pc->operands(node)[0].source;
    }
    const auto operands = pc->operands(node);
    REQUIRE(operands.size() == 2);
    REQUIRE(operands[1].complement);
}

TEST_CASE("FTP input corpus round-trips to stable write checksums", "[io][ftp][corpus]") {
    const auto projects = ftpCorpus();
    if (projects.empty()) {
        SKIP("No public FTP fixtures; set FUZZYBITS_PRIVATE_FIXTURES_DIR to a directory of .ftp files");
    }

    for (auto const &project : projects) {
        DYNAMIC_SECTION(project.label) {
            auto firstRead = readProject(project);
            INFO(diagnosticsText(firstRead.diagnostics()));
            REQUIRE(firstRead);

            const auto firstPc = requirePcModel(firstRead.value());
            const auto firstLimit = ftp_lowering_test::limit_of(firstRead.value());
            if (!std::ranges::all_of(firstPc.probability_sources(), [](auto const &source) {
                    return source.kind == fuzzybits::ir::pc::probability_source_kind::literal;
                })) {
                SUCCEED("frequencies are not literal probabilities; the FTP writer cannot round-trip them");
                continue;
            }
            const auto firstOutput = writeOutput(firstPc, firstLimit);
            auto secondRead = ftp_lowering_test::lower_text(firstOutput);
            INFO(diagnosticsText(secondRead.diagnostics()));
            REQUIRE(secondRead);

            const auto secondPc = requirePcModel(secondRead.value());
            const auto secondOutput = writeOutput(secondPc, ftp_lowering_test::limit_of(secondRead.value()));
            INFO("first checksum: " << checksum(firstOutput));
            INFO("second checksum: " << checksum(secondOutput));
            REQUIRE(firstOutput.size() == secondOutput.size());
            REQUIRE(checksum(firstOutput) == checksum(secondOutput));
        }
    }
}
