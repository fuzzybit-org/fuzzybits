/// Host-only stubs for ProbabilityAnalyzer<DirectEval>.
///
/// Layering inversion: libscram.so references these symbols, which live in
/// SYCL libmc. This TU is compiled into `fzb` / `fzb_compile_tests` only when
/// FUZZYBITS_WITH_MC=OFF (see targets/fzb/CMakeLists.txt). If it is also
/// linked when libmc is present, ELF interposition binds libscram.so to the
/// stubs and `fzb run` (#44) would throw / observe an empty set. The proper
/// fix — move DirectEval instantiations out of libscram, or make the core
/// analyzer independent of mc — is out of scope for Phase 1.

#include "probability_analysis.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace scram::core {

ProbabilityAnalyzer<DirectEval>::ProbabilityAnalyzer(FaultTreeAnalyzer<DirectEval> *fta, mef::MissionTime *mission_time)
    : ProbabilityAnalyzerBase(fta, mission_time) {}

ProbabilityAnalyzer<DirectEval>::~ProbabilityAnalyzer() = default;

double ProbabilityAnalyzer<DirectEval>::CalculateTotalProbability(const Pdag::IndexMap<double> &) {
    throw std::runtime_error("host-only fzb: ProbabilityAnalyzer<DirectEval> requires libmc");
}

std::vector<std::pair<double, double>> ProbabilityAnalyzer<DirectEval>::CalculateProbabilityOverTime() { return {}; }

void ProbabilityAnalyzer<DirectEval>::ComputeTallies(bool) { throw std::runtime_error("host-only fzb: ProbabilityAnalyzer<DirectEval> requires libmc"); }

std::unordered_set<int> ProbabilityAnalyzer<DirectEval>::ObserveNodes(Pdag *, fuzzybits::mc::stats::TallyNodeMap &, const std::unordered_set<index_t> &, bool,
                                                                      bool) {
    return {};
}

auto ProbabilityAnalyzer<DirectEval>::GatherGates(Pdag *, std::unordered_set<index_t> *) {}

std::unordered_set<int> ProbabilityAnalyzer<DirectEval>::GatherGates(Pdag *) { return {}; }

void ProbabilityAnalyzer<DirectEval>::SanitizeWatchState() {}

} // namespace scram::core
