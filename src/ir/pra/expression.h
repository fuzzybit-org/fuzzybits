#pragma once

#include "ir/pra/id.h"

#include <cstdint>

namespace fuzzybits::ir::pra {

// R1 expression kinds. Rates, mission-time conversion and deviates are later
// kinds of this table; until then readers record them as `unsupported`.
enum class expression_kind : std::uint8_t {
    literal,
    parameter,
    unsupported,
};

// What the value measures. Initiating events may carry a frequency (per unit
// time), which is not a probability and is not bounded by 1.
enum class quantity_kind : std::uint8_t {
    probability,
    frequency,
};

// The model stores expressions; it never evaluates them.
struct expression final {
    expression_kind kind = expression_kind::unsupported;
    quantity_kind quantity = quantity_kind::probability;
    double value = 0.0;
    text_ref parameter_name;
    // Stable diagnostic id of an unsupported expression, for example
    // `io.ftp.missing-import`.
    text_ref diagnostic;
    text_ref detail;
};

} // namespace fuzzybits::ir::pra
