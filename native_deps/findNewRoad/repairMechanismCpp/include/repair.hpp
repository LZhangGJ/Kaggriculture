#pragma once

#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace g001::repair {

struct TimedAction {
    fastkag::Action action{};
    fastkag::Position required_position{};
    int earliest_step = 0;
    int latest_step = 1'000'000;
    int economic_value = 0;
    bool expected_fill = false;
};

struct AlignmentWeights {
    std::int64_t movement_edit = 1'000'000;
    std::int64_t critical_fill_failure = 100'000;
    std::int64_t economic_value_loss = 100;
    std::int64_t path_l1_step = 20;
    std::int64_t delayed_action_step = 1;
};

struct AlignmentMetrics {
    std::int64_t objective = 0;
    int skipped_source_index = -1;
    int movement_edits = 0;
    int critical_fill_failures = 0;
    int economic_value_lost = 0;
    int path_l1_area = 0;
    int delayed_action_steps = 0;
    int harvest_failures = 0;
    int feed_failures = 0;
    int plant_failures = 0;
};

struct AlignmentPlan {
    // -1 is the inserted DIG. Otherwise this is an index into the planned tape.
    std::vector<int> source_index_by_actual_step;
    AlignmentMetrics metrics{};
};

[[nodiscard]] bool is_movement(fastkag::Op operation);
[[nodiscard]] int default_economic_value(fastkag::Op operation);

// Reproduce the deployed bounded transaction: insert DIG, delay source
// actions 0..8 by one turn, and discard source action 9.
[[nodiscard]] AlignmentPlan legacy_fixed_window(
    const std::vector<TimedAction>& planned,
    fastkag::Position start,
    const AlignmentWeights& weights = {}
);

// Exhaustively choose which non-movement source action absorbs the inserted
// DIG. Movement order is invariant; a PASS/slack or low-value work action is
// preferred when that avoids production loss.
[[nodiscard]] AlignmentPlan search_minimum_loss_realign(
    const std::vector<TimedAction>& planned,
    fastkag::Position start,
    int maximum_delayed_source_index,
    int minimum_skipped_source_index = 0,
    const AlignmentWeights& weights = {}
);

struct StationaryWorkRecovery {
    bool recovered = false;
    int source_index = -1;
    int economic_value = 0;
    fastkag::Action replacement{};
    std::string reason;
};

// Replace an invalid empty-stall work action with a future valid non-movement
// action at the exact same position. The caller must suppress source_index
// when its original turn arrives; no movement is inserted, deleted or moved.
[[nodiscard]] StationaryWorkRecovery search_stationary_work_replacement(
    fastkag::Position current_position,
    int current_step,
    const std::vector<TimedAction>& future,
    int maximum_lookahead
);

enum class Animal : std::uint8_t { Goose = 0, Cow = 1, Sheep = 2 };

struct AnimalObservation {
    int step = 0;
    std::array<int, 3> own_total{};       // shed + carried + placed
    std::array<int, 3> known_losses{};    // public deaths/removals since prior obs
    std::array<int, 3> empty_structures{};
    std::array<int, 3> next_pickup_step{{-1, -1, -1}};
    std::array<int, 3> next_place_step{{-1, -1, -1}};
    std::array<bool, 3> pickup_place_feasible{};
    std::array<bool, 3> allow_same_step_pickup_realign{};
    int shed_used = 0;
    int shed_capacity = 100;
    int money = 0;
    int protected_cash_reserve = 0;
    int market_slots_used = 0;
    std::array<int, 3> planned_buy{};
};

struct AnimalRetryParameters {
    int maximum_attempts = 3;     // original request included
    int maximum_quantity = 2;
    int retry_backoff_steps = 0;
    int retire_after_deadline = 1;
};

struct AnimalRetryDecision {
    Animal animal = Animal::Goose;
    int quantity = 0;
    int deadline = -1;
    bool requires_pickup_realign = false;
    std::string reason;
};

struct AnimalIntentSnapshot {
    int outstanding = 0;
    int attempts = 0;
    int inferred_fills = 0;
    int inferred_failures = 0;
    int cash_limited_failures = 0;
    int capacity_limited_failures = 0;
    int unclassified_failures = 0;
    int deadline = -1;
    bool retired = false;
    std::string retire_reason;
};

class AnimalRetryController {
public:
    explicit AnimalRetryController(AnimalRetryParameters parameters = {});
    void reset();
    // Call exactly once at the start of each observed step.
    void observe(const AnimalObservation& observation);
    // Record each original or retry BUY_ANIMAL emitted this step.
    void record_attempt(Animal animal, int quantity, int deadline);
    [[nodiscard]] std::optional<AnimalRetryDecision> decide(
        const AnimalObservation& observation
    );
    [[nodiscard]] const AnimalIntentSnapshot& snapshot(Animal animal) const;

private:
    struct Intent {
        AnimalIntentSnapshot public_state{};
        int last_attempt_step = -1;
        int last_attempt_quantity = 0;
        int total_before_attempt = 0;
        int money_before_attempt = 0;
        int capacity_before_attempt = 0;
        bool awaiting_observation = false;
    };
    AnimalRetryParameters parameters_{};
    std::array<Intent, 3> intents_{};
    std::array<int, 3> latest_total_{};
    int latest_money_ = 0;
    int latest_capacity_room_ = 0;
    int latest_step_ = -1;
};

[[nodiscard]] int animal_cost(Animal animal);
[[nodiscard]] bool suppress_empty_stall_work(
    fastkag::Op operation,
    fastkag::TileKind tile_kind,
    bool animal_present
);

}  // namespace g001::repair
