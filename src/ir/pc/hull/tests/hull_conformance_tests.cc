#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "api/options.h"
#include "compile/ftp/tests/support.h"
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

    // The lowering's default for hull targets (#190): directly complemented
    // events are split into an atom and a complement atom.
    auto split_modes(pc::circuit const &model, hull::checker const &check) -> std::vector<hull::event_mode> {
        std::vector<hull::event_mode> modes(check.event_count(), hull::event_mode::down);
        for (std::uint32_t event = 0; event < check.event_count(); ++event) {
            const auto node = check.event_node(event);
            for (auto const &edge : model.edges()) {
                if (edge.complement && edge.source == node) {
                    modes[event] = hull::event_mode::split;
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
        std::vector<ftp::source_text> sources;
        for (auto const &file : expected.find("files")->items) {
            const auto path = fixture / file.text;
            sources.push_back(ftp::source_text{.path = path.generic_string(), .text = read_text(path)});
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
            auto read = ftp_lowering_test::lower(sources, ftp_lowering_test::boolean_options({target}));
            INFO(read.diagnostics().size() << " diagnostics");
            REQUIRE(read);
            auto const *model = &read.value().circuit;
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

            const auto modes = split_modes(*model, plain.value());
            auto split = hull::checker::make(*model, node, modes);
            REQUIRE(split);
            const auto plain_value = hull::testing::exact_probability(*model, plain.value());
            const auto split_value = hull::testing::exact_probability(*model, split.value());
            const auto hull_value = values.find("hull")->number;
            INFO("plain hull " << plain_value << ", atom-split hull " << split_value);
            if (auto const *positive = values.find("hull_positive"); positive != nullptr) {
                // Fixtures regenerated by #197: `hull` is the atom-split hull.
                REQUIRE(close(split_value, hull_value));
                REQUIRE(close(plain_value, positive->number));
            } else {
                REQUIRE((close(plain_value, hull_value) || close(split_value, hull_value)));
                if (std::ranges::count(modes, hull::event_mode::split) == 0) {
                    REQUIRE(close(plain_value, hull_value));
                }
            }
            ++targets;
        }
    }
    REQUIRE(targets >= 14);
}
