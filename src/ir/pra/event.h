#pragma once

#include "ir/pra/id.h"

namespace fuzzybits::ir::pra {

    struct basic_event final {
        element_id element;
        expression_id expression;
    };

    struct house_event final {
        element_id element;
        bool value = false;
    };

    // An initiator overlay on a basic event (the FTP `I` tag); the basic event
    // carries the value, which may be a frequency.
    struct initiating_event final {
        element_id element;
        basic_event_id event;
    };

} // namespace fuzzybits::ir::pra
