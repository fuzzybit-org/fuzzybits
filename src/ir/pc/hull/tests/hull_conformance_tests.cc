#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "api/options.h"
#include "io/ftp/descriptor.h"
#include "ir/pc/circuit.h"
#include "ir/pc/hull/hull.h"
#include "ir/pc/hull/tests/hull_test_support.h"
#include "ir/pc/op_code.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

    namespace pc = fuzzybits::ir::pc;
    namespace hull = fuzzybits::ir::pc::hull;
    namespace ftp = fuzzybits::io::backend::fuzzybits::ftp;

    auto read_text(std::filesystem::path const &path) -> std::string {
        std::ifstream input(path, std::ios::binary);
        REQUIRE(input);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    auto close(const double left, const double right) -> bool {
        return std::abs(left - right) <= 1e-12 * std::max(1.0, std::abs(right));
    }

    // Events referenced through a complemented edge somewhere in the circuit:
    // both of their literals are significant once #190 lands.
    auto literal_modes(pc::circuit const &model, hull::checker const &check) -> std::vector<hull::event_mode> {
        std::vector<hull::event_mode> modes(check.event_count(), hull::event_mode::down);
        for (std::uint32_t event = 0; event < check.event_count(); ++event) {
            const auto node = check.event_node(event);
            for (auto const &edge : model.edges()) {
                if (edge.complement && edge.source == node) {
                    modes[event] = hull::event_mode::fixed;
                }
            }
        }
        return modes;
    }

} // namespace

// Exhaustive enumeration over each target's events, with the hull check as
// the indicator, reproduces the reference tools' exact values (#179).
TEST_CASE("hull check reproduces the FTP conformance fixtures", "[ir][pc][hull][ftp]") {
    const auto root = std::filesystem::path(PROJECT_SOURCE_DIR) / "input" / "ftp" / "conformance";
    std::vector<std::filesystem::path> fixtures;
    for (auto const &entry : std::filesystem::directory_iterator(root)) {
        if (std::filesystem::exists(entry.path() / "expected.json")) {
            fixtures.push_back(entry.path());
        }
    }
    std::ranges::sort(fixtures);
    REQUIRE(fixtures.size() >= 9);

    std::size_t targets = 0;
    for (auto const &fixture : fixtures) {
        const auto expected = hull::testing::json_reader(read_text(fixture / "expected.json")).parse();
        std::string text;
        for (auto const &file : expected.find("files")->items) {
            text += read_text(fixture / file.text);
            text += '\n';
        }
        // The FTP reader keeps the last of duplicated imports. Fixtures that the
        // reference tools reject have no results.
        auto const *results = expected.find("policies")->find("last")->find("results");
        REQUIRE(results != nullptr);
        if (results->type == hull::testing::json::kind::null) {
            continue;
        }
        REQUIRE(results->type == hull::testing::json::kind::object);
        for (std::size_t member = 0; member < results->keys.size(); ++member) {
            auto const &target = results->keys[member];
            auto const &values = results->items[member];
            INFO(fixture.filename().string() << " / " << target);
            fuzzybits::api::option_set options;
            options.set(ftp::target_option(), target);
            std::istringstream input(text);
            auto read = ftp::read(input, options);
            REQUIRE(read);
            auto const *model = pc::circuit_from(read.value().model);
            REQUIRE(model != nullptr);
            REQUIRE(model->targets().size() == 1);
            const auto node = model->targets()[0].node;

            auto plain = hull::checker::make(*model, node);
            REQUIRE(plain);
            REQUIRE(plain.value().cone_events().size() == static_cast<std::size_t>(values.find("variables")->number));

            const std::vector<hull::event_mode> all_fixed(plain.value().event_count(), hull::event_mode::fixed);
            auto boolean = hull::checker::make(*model, node, all_fixed);
            REQUIRE(boolean);
            const auto boolean_value = hull::testing::exact_probability(*model, boolean.value());
            REQUIRE(close(boolean_value, values.find("boolean")->number));

            const auto modes = literal_modes(*model, plain.value());
            auto literal = hull::checker::make(*model, node, modes);
            REQUIRE(literal);
            const auto plain_value = hull::testing::exact_probability(*model, plain.value());
            const auto literal_value = hull::testing::exact_probability(*model, literal.value());
            const auto hull_value = values.find("hull")->number;
            INFO("plain hull " << plain_value << ", with complemented events as literals " << literal_value);
            // Before #190 the fixtures hold the plain hull; after it, the hull
            // with directly complemented events fixed.
            REQUIRE((close(plain_value, hull_value) || close(literal_value, hull_value)));
            if (std::ranges::count(modes, hull::event_mode::fixed) == 0) {
                REQUIRE(close(plain_value, hull_value));
            }
            ++targets;
        }
    }
    REQUIRE(targets >= 14);
}
