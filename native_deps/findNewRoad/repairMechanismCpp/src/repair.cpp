#include "repair.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>

namespace g001::repair {
namespace {

fastkag::Position moved(fastkag::Position position, fastkag::Op operation) {
    if (operation == fastkag::Op::NORTH) --position.y;
    if (operation == fastkag::Op::SOUTH) ++position.y;
    if (operation == fastkag::Op::EAST) ++position.x;
    if (operation == fastkag::Op::WEST) --position.x;
    return position;
}

int distance(fastkag::Position left, fastkag::Position right) {
    return std::abs(int(left.x) - int(right.x))
         + std::abs(int(left.y) - int(right.y));
}

std::vector<fastkag::Op> movement_sequence(
    const std::vector<TimedAction>& planned,
    const std::vector<int>& schedule
) {
    std::vector<fastkag::Op> result;
    for (const auto source : schedule) {
        if (source >= 0 && source < int(planned.size()) &&
            is_movement(planned[static_cast<std::size_t>(source)].action.op)) {
            result.push_back(planned[static_cast<std::size_t>(source)].action.op);
        }
    }
    return result;
}

int edit_distance(const std::vector<fastkag::Op>& left,
                  const std::vector<fastkag::Op>& right) {
    std::vector<int> previous(right.size() + 1), current(right.size() + 1);
    for (std::size_t j = 0; j <= right.size(); ++j) previous[j] = int(j);
    for (std::size_t i = 1; i <= left.size(); ++i) {
        current[0] = int(i);
        for (std::size_t j = 1; j <= right.size(); ++j) {
            current[j] = std::min({previous[j] + 1, current[j - 1] + 1,
                previous[j - 1] + int(left[i - 1] != right[j - 1])});
        }
        previous.swap(current);
    }
    return previous.back();
}

void count_failure(AlignmentMetrics& metrics, fastkag::Op operation) {
    ++metrics.critical_fill_failures;
    if (operation == fastkag::Op::HARVEST) ++metrics.harvest_failures;
    if (operation == fastkag::Op::FEED) ++metrics.feed_failures;
    if (operation == fastkag::Op::PLANT) ++metrics.plant_failures;
}

AlignmentPlan evaluate_schedule(const std::vector<TimedAction>& planned,
                                fastkag::Position start,
                                std::vector<int> schedule,
                                int skipped,
                                const AlignmentWeights& weights) {
    AlignmentPlan result{std::move(schedule), {}};
    result.metrics.skipped_source_index = skipped;
    auto planned_position = start;
    auto actual_position = start;
    std::vector<bool> executed(planned.size(), false);
    for (std::size_t actual_step = 0; actual_step < result.source_index_by_actual_step.size();
         ++actual_step) {
        if (actual_step < planned.size()) {
            planned_position = moved(planned_position, planned[actual_step].action.op);
        }
        const int source = result.source_index_by_actual_step[actual_step];
        if (source >= 0 && source < int(planned.size())) {
            const auto& action = planned[static_cast<std::size_t>(source)];
            const auto before = actual_position;
            actual_position = moved(actual_position, action.action.op);
            executed[static_cast<std::size_t>(source)] = true;
            result.metrics.delayed_action_steps += std::max(0, int(actual_step) - source);
            if (action.expected_fill &&
                (before.x != action.required_position.x ||
                 before.y != action.required_position.y ||
                 int(actual_step) < action.earliest_step ||
                 int(actual_step) > action.latest_step)) {
                count_failure(result.metrics, action.action.op);
                result.metrics.economic_value_lost += std::max(
                    action.economic_value, default_economic_value(action.action.op));
            }
        }
        result.metrics.path_l1_area += distance(actual_position, planned_position);
    }
    for (std::size_t source = 0; source < planned.size(); ++source) {
        if (!executed[source] && planned[source].expected_fill) {
            count_failure(result.metrics, planned[source].action.op);
            result.metrics.economic_value_lost += std::max(
                planned[source].economic_value,
                default_economic_value(planned[source].action.op));
        }
    }
    std::vector<fastkag::Op> baseline_moves;
    for (const auto& action : planned) {
        if (is_movement(action.action.op)) baseline_moves.push_back(action.action.op);
    }
    result.metrics.movement_edits = edit_distance(
        baseline_moves, movement_sequence(planned, result.source_index_by_actual_step));
    result.metrics.objective =
        std::int64_t(result.metrics.movement_edits) * weights.movement_edit +
        std::int64_t(result.metrics.critical_fill_failures) * weights.critical_fill_failure +
        std::int64_t(result.metrics.economic_value_lost) * weights.economic_value_loss +
        std::int64_t(result.metrics.path_l1_area) * weights.path_l1_step +
        std::int64_t(result.metrics.delayed_action_steps) * weights.delayed_action_step;
    return result;
}

std::vector<int> schedule_for_skip(std::size_t size, int skipped) {
    std::vector<int> schedule(size, -2);
    if (size == 0) return schedule;
    schedule[0] = -1;  // forced DIG
    int source = 0;
    for (std::size_t actual = 1; actual < size; ++actual) {
        if (source == skipped) ++source;
        schedule[actual] = source < int(size) ? source : -2;
        ++source;
    }
    return schedule;
}

std::size_t animal_index(Animal animal) {
    return static_cast<std::size_t>(animal);
}

}  // namespace

bool is_movement(fastkag::Op operation) {
    return operation == fastkag::Op::NORTH || operation == fastkag::Op::SOUTH ||
           operation == fastkag::Op::EAST || operation == fastkag::Op::WEST;
}

int default_economic_value(fastkag::Op operation) {
    switch (operation) {
        case fastkag::Op::HARVEST: return 12;
        case fastkag::Op::FEED: return 10;
        case fastkag::Op::PLANT: return 9;
        case fastkag::Op::PLACE: return 9;
        case fastkag::Op::WATER: return 7;
        case fastkag::Op::CARE: return 6;
        case fastkag::Op::FERTILIZE: return 5;
        case fastkag::Op::COLLECT_FERTILIZER: return 5;
        case fastkag::Op::PICKUP:
        case fastkag::Op::DROP: return 4;
        case fastkag::Op::BUILD_COOP:
        case fastkag::Op::BUILD_PASTURE: return 8;
        default: return 0;
    }
}

AlignmentPlan legacy_fixed_window(const std::vector<TimedAction>& planned,
                                  fastkag::Position start,
                                  const AlignmentWeights& weights) {
    // With eight replay actions, source index 9 is the first source action not
    // replayed before the transaction expires.
    const int skipped = planned.size() > 9 ? 9 : int(planned.size()) - 1;
    return evaluate_schedule(planned, start, schedule_for_skip(planned.size(), skipped),
                             skipped, weights);
}

AlignmentPlan search_minimum_loss_realign(
    const std::vector<TimedAction>& planned,
    fastkag::Position start,
    int maximum_delayed_source_index,
    int minimum_skipped_source_index,
    const AlignmentWeights& weights
) {
    AlignmentPlan best;
    best.metrics.objective = std::numeric_limits<std::int64_t>::max();
    const int last = std::min(int(planned.size()) - 1,
                              std::max(0, maximum_delayed_source_index));
    for (int skipped = std::max(0, minimum_skipped_source_index); skipped <= last; ++skipped) {
        // The route's movement order is a hard invariant. The extra DIG must
        // be paid for by a non-movement action.
        if (is_movement(planned[static_cast<std::size_t>(skipped)].action.op)) continue;
        auto candidate = evaluate_schedule(
            planned, start, schedule_for_skip(planned.size(), skipped), skipped, weights);
        if (candidate.metrics.objective < best.metrics.objective ||
            (candidate.metrics.objective == best.metrics.objective &&
             candidate.metrics.skipped_source_index < best.metrics.skipped_source_index)) {
            best = std::move(candidate);
        }
    }
    return best;
}

StationaryWorkRecovery search_stationary_work_replacement(
    fastkag::Position current_position,
    int current_step,
    const std::vector<TimedAction>& future,
    int maximum_lookahead
) {
    StationaryWorkRecovery best;
    best.reason = "no_same_position_productive_work";
    const int last = std::min(int(future.size()) - 1, std::max(0, maximum_lookahead));
    for (int index = 0; index <= last; ++index) {
        const auto& candidate = future[static_cast<std::size_t>(index)];
        if (is_movement(candidate.action.op) || !candidate.expected_fill ||
            candidate.required_position.x != current_position.x ||
            candidate.required_position.y != current_position.y ||
            current_step < candidate.earliest_step || current_step > candidate.latest_step)
            continue;
        // Another empty-stall operation is not recovery.
        if (candidate.action.op == fastkag::Op::FEED ||
            candidate.action.op == fastkag::Op::CARE ||
            candidate.action.op == fastkag::Op::HARVEST ||
            candidate.action.op == fastkag::Op::COLLECT_FERTILIZER) continue;
        const int value = std::max(candidate.economic_value,
                                   default_economic_value(candidate.action.op));
        if (!best.recovered || value > best.economic_value ||
            (value == best.economic_value && index < best.source_index)) {
            best.recovered = true;
            best.source_index = index;
            best.economic_value = value;
            best.replacement = candidate.action;
            best.reason = "same_position_future_work";
        }
    }
    return best;
}

int animal_cost(Animal animal) {
    constexpr std::array<int, 3> costs{300, 400, 500};
    return costs[animal_index(animal)];
}

AnimalRetryController::AnimalRetryController(AnimalRetryParameters parameters)
    : parameters_(parameters) {
    reset();
}

void AnimalRetryController::reset() {
    intents_ = {};
    latest_total_ = {};
    latest_step_ = -1;
    latest_money_ = 0;
    latest_capacity_room_ = 0;
}

void AnimalRetryController::observe(const AnimalObservation& observation) {
    if (latest_step_ >= 0 && observation.step <= latest_step_) reset();
    for (std::size_t index = 0; index < intents_.size(); ++index) {
        auto& intent = intents_[index];
        if (intent.awaiting_observation && observation.step > intent.last_attempt_step) {
            const int purchased = std::clamp(
                observation.own_total[index] + observation.known_losses[index]
                    - intent.total_before_attempt,
                0, intent.last_attempt_quantity);
            intent.public_state.inferred_fills += purchased;
            const int failure = intent.last_attempt_quantity - purchased;
            intent.public_state.inferred_failures += failure;
            intent.public_state.outstanding += failure;
            if (failure > 0) {
                const auto animal = static_cast<Animal>(index);
                const int cash_capacity = intent.money_before_attempt / animal_cost(animal);
                const int shed_capacity = intent.capacity_before_attempt;
                bool classified = false;
                if (purchased >= cash_capacity && cash_capacity < intent.last_attempt_quantity) {
                    intent.public_state.cash_limited_failures += failure;
                    classified = true;
                }
                if (purchased >= shed_capacity && shed_capacity < intent.last_attempt_quantity) {
                    intent.public_state.capacity_limited_failures += failure;
                    classified = true;
                }
                if (!classified) intent.public_state.unclassified_failures += failure;
            }
            intent.awaiting_observation = false;
        }
        if (intent.public_state.outstanding > 0) {
            const int deadline = observation.next_place_step[index] >= 0
                ? observation.next_place_step[index] : intent.public_state.deadline;
            intent.public_state.deadline = deadline;
            if (observation.empty_structures[index] <= 0) {
                intent.public_state.retired = true;
                intent.public_state.retire_reason = "no_empty_structure";
            } else if (deadline >= 0 &&
                       observation.step > deadline + parameters_.retire_after_deadline) {
                intent.public_state.retired = true;
                intent.public_state.retire_reason = "place_deadline_passed";
            } else if (intent.public_state.attempts >= parameters_.maximum_attempts) {
                intent.public_state.retired = true;
                intent.public_state.retire_reason = "attempt_limit";
            }
        }
    }
    latest_total_ = observation.own_total;
    latest_money_ = observation.money;
    latest_capacity_room_ = std::max(0, observation.shed_capacity - observation.shed_used);
    latest_step_ = observation.step;
}

void AnimalRetryController::record_attempt(Animal animal, int quantity, int deadline) {
    if (quantity <= 0) return;
    const auto index = animal_index(animal);
    auto& intent = intents_[index];
    intent.last_attempt_step = latest_step_;
    intent.last_attempt_quantity = quantity;
    intent.total_before_attempt = latest_total_[index];
    intent.money_before_attempt = latest_money_;
    intent.capacity_before_attempt = latest_capacity_room_;
    intent.awaiting_observation = true;
    ++intent.public_state.attempts;
    if (deadline >= 0) intent.public_state.deadline = deadline;
}

std::optional<AnimalRetryDecision> AnimalRetryController::decide(
    const AnimalObservation& observation
) {
    if (observation.market_slots_used >= 10) return std::nullopt;
    for (std::size_t index = 0; index < intents_.size(); ++index) {
        auto& intent = intents_[index];
        auto& state = intent.public_state;
        if (state.outstanding <= 0 || state.retired || observation.planned_buy[index] > 0)
            continue;
        const bool future_pickup = observation.next_pickup_step[index] > observation.step;
        const bool shifted_pickup = observation.next_pickup_step[index] == observation.step &&
            observation.allow_same_step_pickup_realign[index];
        if (!observation.pickup_place_feasible[index] || (!future_pickup && !shifted_pickup) ||
            observation.next_place_step[index] < observation.next_pickup_step[index]) continue;
        if (intent.awaiting_observation) continue;
        if (observation.step - intent.last_attempt_step <= parameters_.retry_backoff_steps)
            continue;
        const auto animal = static_cast<Animal>(index);
        const int available_cash = observation.money - observation.protected_cash_reserve;
        const int capacity = observation.shed_capacity - observation.shed_used;
        const int affordable = available_cash / animal_cost(animal);
        const int quantity = std::min({state.outstanding, parameters_.maximum_quantity,
                                       capacity, affordable});
        if (quantity <= 0) continue;
        state.outstanding -= quantity;  // restored causally if the retry fails
        return AnimalRetryDecision{
            animal, quantity, observation.next_pickup_step[index], shifted_pickup,
            shifted_pickup ? "causal_failed_buy_retry_with_pickup_realign"
                           : "causal_failed_buy_retry"
        };
    }
    return std::nullopt;
}

const AnimalIntentSnapshot& AnimalRetryController::snapshot(Animal animal) const {
    return intents_[animal_index(animal)].public_state;
}

bool suppress_empty_stall_work(fastkag::Op operation,
                               fastkag::TileKind tile_kind,
                               bool animal_present) {
    if (animal_present || (tile_kind != fastkag::TileKind::COOP &&
                           tile_kind != fastkag::TileKind::PASTURE)) return false;
    return operation == fastkag::Op::FEED || operation == fastkag::Op::CARE ||
           operation == fastkag::Op::HARVEST ||
           operation == fastkag::Op::COLLECT_FERTILIZER;
}

}  // namespace g001::repair
