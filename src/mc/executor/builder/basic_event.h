/**
 * @file basic_event.h
 * @brief The basic events and constants of a layer: one kernel samples those whose trials are random (op_code::sample),
 *        and the bitpacks of those whose law is a constant (op_code::ones, op_code::zeros) are written once
 * @author Arjun Earthperson
 * @date 2026
 */

#pragma once

#include "mc/event/basic_event.h"
#include "mc/executor/builder/dispatch.h"
#include "mc/executor/builder/op.h"
#include "mc/executor/builder/state.h"
#include "mc/executor/descriptor/partition.h"
#include "mc/kernel/sampler/sampler.h"
#include "mc/kernel/sampler/strided.h"
#include "mc/kernel/sampler/vector.h"

#include <cstddef>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace fuzzybits::mc::executor::builder {

/// The sampling kernel of form `form_` (kernel/sampler/).
template <kernel::sampler::form form_, event::node_types_like T>
using sampler_kernel = std::conditional_t<form_ == kernel::sampler::form::strided, kernel::sampler::strided<T>, kernel::sampler::vector_kernel<T, form_>>;

/**
 * @brief The outputs of `events`, the basic events of layer `layer_index`, each with its exact law (event::bernoulli_law)
 *
 * @details An event whose law is `never` or `always` has the same bitpacks in every pass: they are written when its
 * arena slot is allocated (kernel::sampler::write_constant_outputs), and no kernel is built for it. The others take
 * the next arena slots, in the order of `events`, and the kernel of the device's sampler form samples them
 * (kernel::sampler::form_for); every form writes the same bitpacks.
 */
template <event::node_types_like T, typename size_t_>
void build_basic_events(state<T, size_t_> &build, const std::span<const event::basic_event<T>> events, const std::size_t layer_index) {
    std::vector<event::sampled_event<T>> sampled;
    std::vector<event::constant_event<T>> constant;
    for (const event::basic_event<T> &record : events) {
        if (const auto random = event::sampled_event<T>::of(record)) {
            sampled.push_back(*random);
        } else {
            constant.push_back(*event::constant_event<T>::of(record));
        }
    }
    if (!constant.empty()) {
        auto block = event::constant_event_block<T>::create(build.queue(), build.arena(), constant);
        kernel::sampler::write_constant_outputs(build.queue(), block);
        build.written(std::move(block));
    }
    if (sampled.empty()) {
        return;
    }
    auto block = event::sampled_event_block<T>::create(build.queue(), build.arena(), sampled);
    using kernel::sampler::form;
    // form_for picks strided or chunked; only the tests build the lockstep kernel.
    const std::size_t partition = select<form::strided, form::chunked>(
        kernel::sampler::form_for(build.device()),
        [&]<form form_>() {
            return build.add(sampler_kernel<form_, T>(block.view(), build.stream(), build.device()), block, dependencies{},
                             {.layer = layer_index, .kind = descriptor::partition_kind::sampler, .operation = std::string(name_of(op_code::sample))});
        },
        "sampler form");
    build.computed(std::move(block), partition);
}

} // namespace fuzzybits::mc::executor::builder
