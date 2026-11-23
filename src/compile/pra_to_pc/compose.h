#pragma once

#include "io/pcfb/descriptor.h"
#include "ir/pc/circuit.h"
#include "ir/pc/passes/passes.h"

namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc {

    // Re-keys the names in `options`, which `pcfb_write_options` derives from the
    // PC ids of a lowering, to the ids of the circuit `passes` produced from
    // `input` (the lowering's circuit), through the pipeline's composed node and
    // target maps. A target keeps its name, its query kinds, its quantification
    // record and its literal event set (everything the writer options hold per
    // target); a node keeps its name when some output node computes it without
    // complement. A literal event the passes folded or dropped leaves Δ. An event
    // name only ever goes to the event's own node, never to a gate that was
    // eliminated onto it. A node folded to a constant, absorbed or complemented
    // loses its name.
    [[nodiscard]] auto compose_write_options(io::backend::fuzzybits::pcfb::write_options options, ir::pc::circuit const &input,
                                             ir::pc::passes::pipeline_result const &passes)
        -> io::backend::fuzzybits::pcfb::write_options;

} // namespace fuzzybits::compile::backend::fuzzybits::pra_to_pc
