#pragma once

#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace g001::weed_absorption {

enum class CandidatePolicy : std::uint8_t {
    SameDayPassOnly,
    CrossDayPassOnly,
    AnyNonMovement,
};

struct SearchOptions {
    int maximum_absorbed_source = 24;
    int replay_horizon = 48;
    CandidatePolicy candidate_policy = CandidatePolicy::AnyNonMovement;
};

struct DependencyMetrics {
    std::array<int, 24> attempted_by_op{};
    std::array<int, 24> failed_by_op{};
    int productive_attempts = 0;
    int productive_failures = 0;
    int market_unfilled_units = 0;
    int end_of_day_overflow = 0;
};

struct BranchMetrics {
    int skipped_source = -1;
    fastkag::Op skipped_operation = fastkag::Op::PASS;
    int movement_source_edits = 0;
    int movement_coordinate_violations = 0;
    int simulated_steps = 0;
    double cash = 0;
    double inventory_value = 0;
    double seed_value = 0;
    double standing_asset_value = 0;
    double economic_equity = 0;
    double objective = 0;
    DependencyMetrics dependencies{};
};

struct SearchRequest {
    const fastkag::Simulator* state = nullptr;
    int player = 0;
    int actor = 0;
    int absolute_start_step = 0;
    // Raw source tape for the repaired actor and raw market timing.
    const std::vector<fastkag::PlayerAction>* own_tape = nullptr;
    // Actual-step own actions, already containing any older independent repair
    // transactions on other actors. Empty means raw own_tape.
    std::vector<fastkag::PlayerAction> fixed_own_scenario;
    // A causal scenario, not a policy feature. Offline evaluation can use an
    // exact opponent tape; online callers may supply a conservative public
    // forecast. Candidate ranking never reads opponent identity.
    std::vector<fastkag::PlayerAction> opponent_scenario;
    SearchOptions options{};
};

struct SearchResult {
    bool feasible = false;
    int selected_source = -1;
    BranchMetrics selected{};
    std::vector<BranchMetrics> candidates;
    std::int64_t elapsed_nanoseconds = 0;
    std::string reason;
};

[[nodiscard]] bool is_movement(fastkag::Op operation);
[[nodiscard]] bool is_productive(fastkag::Op operation);
// Exact own unit-phase legality for one actor after applying all earlier actors
// in the same PlayerAction, including the simulator's all-or-none PLANT demand
// guard. This does not advance market, town, or the clock.
[[nodiscard]] bool projected_unit_action_succeeds(
    const fastkag::Simulator& state, int player,
    const fastkag::PlayerAction& action, int actor);
[[nodiscard]] std::vector<int> enumerate_candidates(const SearchRequest& request);
[[nodiscard]] BranchMetrics replay_candidate(const SearchRequest& request,
                                             int skipped_source);
[[nodiscard]] SearchResult select_absorption(const SearchRequest& request);

}  // namespace g001::weed_absorption
