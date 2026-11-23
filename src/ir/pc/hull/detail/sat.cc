#include "ir/pc/hull/detail/sat.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace fuzzybits::ir::pc::hull::detail {

namespace {

constexpr std::uint32_t no_clause = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t no_position = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint8_t unassigned = 2;
constexpr double activity_decay = 0.95;
constexpr std::uint64_t restart_unit = 100;

// Luby, Sinclair and Zuckerman (1993) restart sequence 1 1 2 1 1 2 4 ...
[[nodiscard]] auto luby(std::uint64_t index) -> std::uint64_t {
    std::uint64_t size = 1;
    std::uint32_t exponent = 0;
    while (size < index + 1) {
        ++exponent;
        size = 2 * size + 1;
    }
    while (size - 1 != index) {
        size = (size - 1) >> 1;
        --exponent;
        index %= size;
    }
    return std::uint64_t{1} << exponent;
}

[[nodiscard]] constexpr auto negate(const std::uint32_t literal) noexcept -> std::uint32_t { return literal ^ 1U; }

} // namespace

void sat_solver::reset(const std::uint32_t variables) {
    arena_.clear();
    clauses_.clear();
    for (auto &list : watches_) {
        list.clear();
    }
    watches_.resize(2 * std::size_t{variables});
    assigns_.assign(variables, unassigned);
    phase_.assign(variables, 0);
    seen_.assign(variables, 0);
    level_.assign(variables, 0);
    reason_.assign(variables, no_clause);
    activity_.assign(variables, 0.0);
    trail_.clear();
    trail_limits_.clear();
    heap_.clear();
    heap_position_.assign(variables, no_position);
    for (std::uint32_t variable = 0; variable < variables; ++variable) {
        heap_insert(variable);
    }
    head_ = 0;
    increment_ = 1.0;
    inconsistent_ = false;
}

auto sat_solver::add_variable() -> std::uint32_t {
    const auto variable = static_cast<std::uint32_t>(assigns_.size());
    watches_.resize(2 * std::size_t{variable} + 2);
    assigns_.push_back(unassigned);
    phase_.push_back(0);
    seen_.push_back(0);
    level_.push_back(0);
    reason_.push_back(no_clause);
    activity_.push_back(0.0);
    heap_position_.push_back(no_position);
    heap_insert(variable);
    return variable;
}

auto sat_solver::value(const std::uint32_t literal) const noexcept -> std::uint8_t {
    const auto assigned = assigns_[literal >> 1];
    return assigned == unassigned ? unassigned : static_cast<std::uint8_t>(assigned ^ (literal & 1U));
}

auto sat_solver::decision_level() const noexcept -> std::uint32_t {
    return static_cast<std::uint32_t>(trail_limits_.size());
}

void sat_solver::assign(const std::uint32_t literal, const std::uint32_t reason) {
    const auto variable = literal >> 1;
    assigns_[variable] = static_cast<std::uint8_t>(1U ^ (literal & 1U));
    level_[variable] = decision_level();
    reason_[variable] = reason;
    trail_.push_back(literal);
}

void sat_solver::attach(const std::uint32_t index) {
    auto const &stored = clauses_[index];
    const auto first = arena_[stored.begin];
    const auto second = arena_[stored.begin + 1];
    watches_[first].push_back(watcher{.clause = index, .blocker = second});
    watches_[second].push_back(watcher{.clause = index, .blocker = first});
}

void sat_solver::add_clause(std::span<std::uint32_t const> literals) {
    if (inconsistent_) {
        return;
    }
    buffer_.assign(literals.begin(), literals.end());
    std::ranges::sort(buffer_);
    const auto duplicates = std::ranges::unique(buffer_);
    buffer_.erase(duplicates.begin(), duplicates.end());
    std::size_t kept = 0;
    for (std::size_t index = 0; index < buffer_.size(); ++index) {
        const auto literal = buffer_[index];
        if (index + 1 < buffer_.size() && buffer_[index + 1] == negate(literal)) {
            return;
        }
        const auto current = value(literal);
        if (current == 1) {
            return;
        }
        if (current == unassigned) {
            buffer_[kept++] = literal;
        }
    }
    buffer_.resize(kept);
    if (buffer_.empty()) {
        inconsistent_ = true;
        return;
    }
    if (buffer_.size() == 1) {
        assign(buffer_[0], no_clause);
        return;
    }
    const auto index = static_cast<std::uint32_t>(clauses_.size());
    clauses_.push_back(
        clause{.begin = static_cast<std::uint32_t>(arena_.size()), .size = static_cast<std::uint32_t>(buffer_.size())});
    arena_.insert(arena_.end(), buffer_.begin(), buffer_.end());
    attach(index);
}

auto sat_solver::propagate() -> std::uint32_t {
    while (head_ < trail_.size()) {
        const auto falsified = negate(trail_[head_++]);
        auto &watching = watches_[falsified];
        std::size_t read = 0;
        std::size_t write = 0;
        while (read < watching.size()) {
            const auto current = watching[read++];
            if (value(current.blocker) == 1) {
                watching[write++] = current;
                continue;
            }
            auto const &stored = clauses_[current.clause];
            auto *literals = arena_.data() + stored.begin;
            if (literals[0] == falsified) {
                std::swap(literals[0], literals[1]);
            }
            const auto first = literals[0];
            if (first != current.blocker && value(first) == 1) {
                watching[write++] = watcher{.clause = current.clause, .blocker = first};
                continue;
            }
            bool moved = false;
            for (std::uint32_t index = 2; index < stored.size; ++index) {
                if (value(literals[index]) != 0) {
                    std::swap(literals[1], literals[index]);
                    watches_[literals[1]].push_back(watcher{.clause = current.clause, .blocker = first});
                    moved = true;
                    break;
                }
            }
            if (moved) {
                continue;
            }
            watching[write++] = watcher{.clause = current.clause, .blocker = first};
            if (value(first) == 0) {
                while (read < watching.size()) {
                    watching[write++] = watching[read++];
                }
                watching.resize(write);
                head_ = static_cast<std::uint32_t>(trail_.size());
                return current.clause;
            }
            assign(first, current.clause);
        }
        watching.resize(write);
    }
    return no_clause;
}

auto sat_solver::analyze(std::uint32_t conflict) -> std::uint32_t {
    learnt_.clear();
    learnt_.push_back(0);
    std::uint32_t pending = 0;
    std::uint32_t implied = no_position;
    auto index = trail_.size();
    const auto level = decision_level();
    do {
        auto const &stored = clauses_[conflict];
        for (std::uint32_t position = implied == no_position ? 0 : 1; position < stored.size; ++position) {
            const auto literal = arena_[stored.begin + position];
            const auto variable = literal >> 1;
            if (seen_[variable] != 0 || level_[variable] == 0) {
                continue;
            }
            bump(variable);
            seen_[variable] = 1;
            if (level_[variable] >= level) {
                ++pending;
            } else {
                learnt_.push_back(literal);
            }
        }
        while (seen_[trail_[--index] >> 1] == 0) {
        }
        implied = trail_[index];
        conflict = reason_[implied >> 1];
        seen_[implied >> 1] = 0;
        --pending;
    } while (pending > 0);
    learnt_[0] = negate(implied);

    std::uint32_t backtrack_level = 0;
    if (learnt_.size() > 1) {
        std::size_t deepest = 1;
        for (std::size_t position = 2; position < learnt_.size(); ++position) {
            if (level_[learnt_[position] >> 1] > level_[learnt_[deepest] >> 1]) {
                deepest = position;
            }
        }
        std::swap(learnt_[1], learnt_[deepest]);
        backtrack_level = level_[learnt_[1] >> 1];
    }
    for (const auto literal : learnt_) {
        seen_[literal >> 1] = 0;
    }
    return backtrack_level;
}

void sat_solver::backtrack(const std::uint32_t level) {
    if (decision_level() <= level) {
        return;
    }
    const auto keep = trail_limits_[level];
    for (auto position = trail_.size(); position-- > keep;) {
        const auto variable = trail_[position] >> 1;
        phase_[variable] = assigns_[variable];
        assigns_[variable] = unassigned;
        reason_[variable] = no_clause;
        heap_insert(variable);
    }
    trail_.resize(keep);
    trail_limits_.resize(level);
    head_ = static_cast<std::uint32_t>(trail_.size());
}

void sat_solver::bump(const std::uint32_t variable) {
    activity_[variable] += increment_;
    if (activity_[variable] > 1e100) {
        for (auto &activity : activity_) {
            activity *= 1e-100;
        }
        increment_ *= 1e-100;
    }
    if (heap_position_[variable] != no_position) {
        heap_up(heap_position_[variable]);
    }
}

void sat_solver::heap_insert(const std::uint32_t variable) {
    if (heap_position_[variable] != no_position) {
        return;
    }
    heap_position_[variable] = static_cast<std::uint32_t>(heap_.size());
    heap_.push_back(variable);
    heap_up(heap_position_[variable]);
}

// Max-heap on activity; ties go to the smaller variable for determinism.
void sat_solver::heap_up(std::uint32_t position) {
    const auto variable = heap_[position];
    const auto before = [&](const std::uint32_t other) {
        return activity_[variable] > activity_[other] || (activity_[variable] == activity_[other] && variable < other);
    };
    while (position > 0) {
        const auto parent = (position - 1) / 2;
        if (!before(heap_[parent])) {
            break;
        }
        heap_[position] = heap_[parent];
        heap_position_[heap_[position]] = position;
        position = parent;
    }
    heap_[position] = variable;
    heap_position_[variable] = position;
}

void sat_solver::heap_down(std::uint32_t position) {
    const auto variable = heap_[position];
    const auto better = [&](const std::uint32_t left, const std::uint32_t right) {
        return activity_[left] > activity_[right] || (activity_[left] == activity_[right] && left < right);
    };
    const auto size = static_cast<std::uint32_t>(heap_.size());
    while (true) {
        auto child = 2 * position + 1;
        if (child >= size) {
            break;
        }
        if (child + 1 < size && better(heap_[child + 1], heap_[child])) {
            ++child;
        }
        if (!better(heap_[child], variable)) {
            break;
        }
        heap_[position] = heap_[child];
        heap_position_[heap_[position]] = position;
        position = child;
    }
    heap_[position] = variable;
    heap_position_[variable] = position;
}

auto sat_solver::heap_pop() -> std::uint32_t {
    const auto top = heap_.front();
    heap_position_[top] = no_position;
    const auto last = heap_.back();
    heap_.pop_back();
    if (!heap_.empty()) {
        heap_[0] = last;
        heap_position_[last] = 0;
        heap_down(0);
    }
    return top;
}

auto sat_solver::solve() -> bool {
    if (inconsistent_ || propagate() != no_clause) {
        return false;
    }
    std::uint64_t conflicts = 0;
    std::uint64_t restarts = 0;
    auto next_restart = luby(restarts) * restart_unit;
    while (true) {
        const auto conflict = propagate();
        if (conflict != no_clause) {
            if (decision_level() == 0) {
                return false;
            }
            const auto level = analyze(conflict);
            backtrack(level);
            if (learnt_.size() == 1) {
                assign(learnt_[0], no_clause);
            } else {
                const auto index = static_cast<std::uint32_t>(clauses_.size());
                clauses_.push_back(clause{.begin = static_cast<std::uint32_t>(arena_.size()),
                                          .size = static_cast<std::uint32_t>(learnt_.size())});
                arena_.insert(arena_.end(), learnt_.begin(), learnt_.end());
                attach(index);
                assign(learnt_[0], index);
            }
            increment_ /= activity_decay;
            if (++conflicts >= next_restart) {
                backtrack(0);
                next_restart = conflicts + luby(++restarts) * restart_unit;
            }
            continue;
        }
        std::uint32_t variable = no_position;
        while (!heap_.empty()) {
            const auto candidate = heap_pop();
            if (assigns_[candidate] == unassigned) {
                variable = candidate;
                break;
            }
        }
        if (variable == no_position) {
            return true;
        }
        trail_limits_.push_back(static_cast<std::uint32_t>(trail_.size()));
        assign((variable << 1) | (phase_[variable] != 0 ? 0U : 1U), no_clause);
    }
}

auto search_satisfiable(residual const &circuit, sat_solver &solver) -> bool {
    const auto node_count = static_cast<std::uint32_t>(circuit.nodes.size());
    solver.reset(node_count);
    const auto constant_true = solver.add_variable() << 1;
    const auto constant_false = negate(constant_true);
    std::vector<std::uint32_t> clause;
    const auto add = [&](std::initializer_list<std::uint32_t> literals) {
        solver.add_clause(std::span<std::uint32_t const>{literals.begin(), literals.size()});
    };
    add({constant_true});

    std::vector<std::uint32_t> previous;
    std::vector<std::uint32_t> current;
    for (std::uint32_t index = 0; index < node_count; ++index) {
        auto const &node = circuit.nodes[index];
        const auto output = index << 1;
        const auto inputs = circuit.inputs(node);
        switch (node.kind) {
        case residual_kind::variable:
            break;
        case residual_kind::and_gate:
        case residual_kind::or_gate: {
            // OR is AND with every literal negated.
            const auto flip = node.kind == residual_kind::or_gate ? 1U : 0U;
            clause.assign(1, output ^ flip);
            for (const auto literal : inputs) {
                add({negate(output) ^ flip, literal ^ flip});
                clause.push_back(negate(literal) ^ flip);
            }
            solver.add_clause(clause);
            break;
        }
        case residual_kind::xor_gate: {
            auto accumulated = inputs[0];
            if (inputs.size() == 1) {
                add({negate(output), accumulated});
                add({output, negate(accumulated)});
                break;
            }
            for (std::size_t position = 1; position < inputs.size(); ++position) {
                const auto next = inputs[position];
                const auto result = position + 1 == inputs.size() ? output : solver.add_variable() << 1;
                add({negate(result), accumulated, next});
                add({negate(result), negate(accumulated), negate(next)});
                add({result, negate(accumulated), next});
                add({result, accumulated, negate(next)});
                accumulated = result;
            }
            break;
        }
        case residual_kind::cardinality: {
            // Sinz (2005) sequential counter with both implication directions:
            // r[i][j] <-> r[i-1][j] or (r[i-1][j-1] and x_i), "at least j of the first i".
            const auto count = node.count;
            const auto width = std::min(node.high + 1, count);
            previous.assign(std::size_t{width} + 1, constant_false);
            previous[0] = constant_true;
            current.assign(std::size_t{width} + 1, constant_false);
            current[0] = constant_true;
            for (std::uint32_t row = 1; row <= count; ++row) {
                const auto input = inputs[row - 1];
                for (std::uint32_t column = 1; column <= width; ++column) {
                    if (column > row) {
                        current[column] = constant_false;
                        continue;
                    }
                    const auto counter = solver.add_variable() << 1;
                    add({negate(previous[column]), counter});
                    add({negate(previous[column - 1]), negate(input), counter});
                    add({negate(counter), previous[column], previous[column - 1]});
                    add({negate(counter), previous[column], input});
                    current[column] = counter;
                }
                std::swap(previous, current);
            }
            const auto reached_low = previous[node.low];
            const auto exceeded_high = node.high + 1 <= count ? previous[node.high + 1] : constant_false;
            add({negate(output), reached_low});
            add({negate(output), negate(exceeded_high)});
            add({output, negate(reached_low), exceeded_high});
            break;
        }
        }
    }
    add({circuit.root});
    return solver.solve();
}

} // namespace fuzzybits::ir::pc::hull::detail
