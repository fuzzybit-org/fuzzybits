#pragma once

#include "pcfb/ir/circuit.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace fuzzybits::compile {

inline constexpr std::uint32_t kMaxExactEvents = 30;

class ExactEvalError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Exhaustive exact P(query) over independent SAMPLE events.
/// Constant SAMPLE (p ∈ {0,1} / always_true) are substituted, not enumerated.
/// Throws ExactEvalError if the circuit has more than kMaxExactEvents random events, or if it has a
/// query of a kind this evaluator cannot answer (it never substitutes another kind).
[[nodiscard]] double exact_probability(const pcfb::ir::Circuit &circuit, std::uint32_t query_index = 0);

/// As `exact_probability`, but the query is known to be PROBABILITY: other queries of the file,
/// including kinds this library does not answer, are ignored. Callers that dispatch on the kind
/// themselves (`fzb inspect --exact`, `fuzzybits.exact`) use this; they answer every kind present.
[[nodiscard]] double exact_boolean_probability(const pcfb::ir::Circuit &circuit, std::uint32_t query_index = 0);

[[nodiscard]] std::uint32_t random_sample_count(const pcfb::ir::Circuit &circuit);

}  // namespace fuzzybits::compile
