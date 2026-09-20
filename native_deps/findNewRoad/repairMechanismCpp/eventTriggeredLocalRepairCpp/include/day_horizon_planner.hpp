#pragma once

#include "event_triggered_local_repair.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace g001::day_horizon_repair {

using event_local_repair::Action;
using event_local_repair::Position;

struct ActorPlan {
  int actor{-1};
  Position start;
  std::vector<Action> raw;
  // True only when the caller proves that replacing this blocked production
  // slot with an objective transition is state-equivalent.
  std::vector<bool> blocked_equivalent_capacity;
};

struct Objective {
  std::uint64_t id{};
  Position tile;
  std::vector<Action> remaining_transitions;
  int deadline_turn{};
  int value{};
  bool critical{};
};

struct Assignment {
  std::uint64_t objective_id{};
  int transition_index{};
  int actor{-1};
  int turn{-1};
  Action action;
  friend bool operator==(const Assignment&, const Assignment&) = default;
};

struct Result {
  std::vector<std::vector<Action>> manifest;
  std::vector<std::vector<Position>> positions_before;
  std::vector<Assignment> assignments;
  std::vector<std::uint64_t> unscheduled_objectives;
  bool move_slots_exact{};
};

struct ResourceSnapshot {
  // Seed inventory by crop item id at the planning observation.
  std::map<int, int> seeds;
};

struct ReceiptOutcome {
  std::uint64_t objective_id{};
  int transition_index{};
  bool confirmed{};
};

struct RouteSequenceConfig {
  int move_timing_deviation_cost{1};
  int unfinished_move_penalty{1000};
  bool fail_on_unfinished_route{true};
};

struct RouteSequenceResult : Result {
  bool ordered_move_sequences_exact{};
  int move_timing_deviation{};
  int timing_deviation_cost{};
  int terminal_unexecuted_moves{};
  int terminal_unexecuted_raw_actions{};
  int terminal_penalty{};
  int completed_objectives{};
};

struct HarvestProofToken {
  std::uint64_t observation_epoch{};
  std::uint64_t objective_id{};
  int transition_index{};
  friend bool operator==(const HarvestProofToken&,
                         const HarvestProofToken&) = default;
  friend bool operator<(const HarvestProofToken& left,
                        const HarvestProofToken& right) {
    return std::tie(left.observation_epoch, left.objective_id,
                    left.transition_index) <
        std::tie(right.observation_epoch, right.objective_id,
                 right.transition_index);
  }
};

struct ObservationReadiness {
  std::uint64_t observation_epoch{};
  // Transitions present here are explicit WATERED_IMMATURE wait nodes.
  std::set<std::pair<std::uint64_t, int>> harvest_legal_required;
  // Exact (objective id, transition index) proofs from this observation.
  std::set<HarvestProofToken> harvest_legal;
  // Receipt/planner owner feeds previously returned tokens back here. A proof
  // is one-shot even if the same objective remains visible.
  std::set<HarvestProofToken> consumed_harvest_legal;
};

struct TimeExpandedConfig {
  int move_timing_deviation_cost{1};
  int unfinished_objective_value_cost{1};
  int terminal_route_hard_penalty{1'000'000};
  std::size_t max_states{250'000};
  int time_budget_ms{1'000};
};

struct TimeExpandedResult : RouteSequenceResult {
  int completed_objective_value{};
  int unfinished_objective_value{};
  int total_cost{};
  std::size_t expanded_states{};
  std::uint64_t planning_elapsed_us{};
  bool exact{};
  bool fell_back_to_greedy{};
  bool within_time_budget{};
  std::uint64_t observation_epoch{};
  std::vector<std::uint64_t> waiting_objectives;
  std::set<HarvestProofToken> consumed_harvest_legal;
  int conflict_components{};
  int largest_component_objectives{};
  int exact_component_objectives{};
  int joint_search_objectives{};
  int single_actor_exact_components{};
  int joint_search_components{};
  int fallback_components{};
  std::map<int, int> component_objective_size_histogram;
};

// Bounded executable MVP for the offline day-horizon boundary. It freezes all
// raw MOVE slots, derives the position timeline, and assigns complete objective
// suffixes to PASS or caller-certified equivalent capacity. Priority is
// critical, earliest deadline, then value. This is not yet the release
// min-cost optimizer and intentionally has no native activation seam.
[[nodiscard]] Result compile(const std::vector<ActorPlan>& actors,
                             std::vector<Objective> objectives,
                             const ResourceSnapshot& resources = {});

// Re-solves only turns strictly after frozen_through_turn. Past manifest bytes
// come from prior, future MOVE bytes come from raw actors, and only a
// contiguous confirmed transition prefix is retired. resources is the
// post-receipt observation, not a forecast.
[[nodiscard]] Result recompile_suffix(
    const std::vector<ActorPlan>& actors, std::vector<Objective> objectives,
    const ResourceSnapshot& resources, int frozen_through_turn,
    const Result& prior, const std::vector<ReceiptOutcome>& receipts);

// Timing-variable research mode. Every raw non-PASS action remains in its
// actor's ordered queue; objectives may consume only count-proven slack and
// thereby delay, never advance, the queue. Position is recomputed after every
// selected action. No native activation seam exists.
[[nodiscard]] RouteSequenceResult compile_route_sequence(
    const std::vector<ActorPlan>& actors, std::vector<Objective> objectives,
    const ResourceSnapshot& resources = {},
    RouteSequenceConfig config = {});

// Small-component exact time-expanded DP. State contains turn, actor route
// cursor/position, objective mask/progress segment, and the current seed
// snapshot. Larger joint components fail over to the already hard-gated
// RouteSequence manifest, so this offline research API is Pareto-safe versus
// that baseline rather than pretending a partial search is optimal.
[[nodiscard]] TimeExpandedResult compile_time_expanded(
    const std::vector<ActorPlan>& actors, std::vector<Objective> objectives,
    const ResourceSnapshot& resources = {},
    const ObservationReadiness& readiness = {},
    TimeExpandedConfig config = {});

}  // namespace g001::day_horizon_repair
