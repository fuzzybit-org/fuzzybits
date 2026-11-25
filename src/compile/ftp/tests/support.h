#pragma once

// FTP text to the ir::pc circuit the reader and the lowering produce under
// Boolean semantics and no initiator rule, for tests of other components.

#include "api/diagnostics.h"
#include "api/result.h"
#include "compile/ftp/compile.h"
#include "io/ftp/descriptor.h"
#include "ir/pra/model.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ftp_lowering_test {

namespace ftp = fuzzybits::compile::ftp;
namespace reader = fuzzybits::io::backend::fuzzybits::ftp;
using lowering = fuzzybits::compile::backend::fuzzybits::pra_to_pc::pra_pc_lowering;

// `targets` empty: every PROCESS target.
[[nodiscard]] inline auto probability_options(std::vector<std::string> targets = {},
                                              std::string duplicate_import = "last") -> ftp::options {
    ftp::options options;
    options.kinds = {"probability"};
    // The conformance fixtures tag probabilities with I (#243 keeps them meaningful this way).
    options.initiator_values = "probability";
    options.duplicate_import = std::move(duplicate_import);
    options.targets = std::move(targets);
    return options;
}

[[nodiscard]] inline auto lower(std::span<reader::source_text const> sources,
                                ftp::options const &options = probability_options())
    -> fuzzybits::api::result<lowering> {
    return ftp::lower_sources(sources, options);
}

[[nodiscard]] inline auto lower_text(std::string text, ftp::options const &options = probability_options())
    -> fuzzybits::api::result<lowering> {
    const std::vector<reader::source_text> sources{reader::source_text{.path = "<input>", .text = std::move(text)}};
    return ftp::lower_sources(sources, options);
}

// The `ftp.limit` model attribute as a number.
[[nodiscard]] inline auto limit_of(lowering const &lowered) -> std::optional<double> {
    auto const *model = fuzzybits::ir::pra::model_from(lowered.pra_model);
    if (model == nullptr) {
        return std::nullopt;
    }
    for (auto const &attribute : model->project_attributes()) {
        if (model->text(attribute.name) == "ftp.limit") {
            return std::strtod(std::string(model->text(attribute.value)).c_str(), nullptr);
        }
    }
    return std::nullopt;
}

} // namespace ftp_lowering_test
