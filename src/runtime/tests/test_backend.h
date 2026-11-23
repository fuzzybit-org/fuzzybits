#pragma once

#include <cstdlib>
#include <string>

namespace fuzzybits::testing {

/// Backend for tests that run the Monte Carlo engine, as `RunOptions::backend`, `Session::from_*`
/// or `fzb run --backend`. GPU and Metal CI jobs set FUZZYBITS_BACKEND (cuda, hip, metal) so these
/// tests run on the device under test; unset or empty, they run on omp. "auto" leaves the choice to
/// AdaptiveCpp and ACPP_VISIBILITY_MASK.
inline const std::string &backend() {
    static const std::string value = [] {
        const char *env = std::getenv("FUZZYBITS_BACKEND");
        const std::string chosen = env != nullptr && *env != '\0' ? env : "omp";
        return chosen == "auto" ? std::string{} : chosen;
    }();
    return value;
}

/// The same backend as a `fzb run --backend` argument ("" for "auto").
inline std::string backend_flag() { return backend().empty() ? std::string{} : " --backend " + backend(); }

} // namespace fuzzybits::testing
