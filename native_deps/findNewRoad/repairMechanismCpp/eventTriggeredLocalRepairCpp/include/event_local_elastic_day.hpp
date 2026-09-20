#pragma once

#include "exact_move_slot_planner.hpp"

#include <cstdint>
#include <vector>

namespace g001::day_horizon_repair {

struct ElasticDayActor {
  int actor{-1};
  Position start;
  std::vector<Action> raw;
  // PASS or an exact no-effect/stale production source. Only these sources
  // may be consumed without replaying their byte.
  std::vector<bool> certified_sink;
  std::vector<bool> trigger_source;
};

struct ElasticDayInput {
  ElasticDayActor actor;
  event_local_repair::TileObservation plot_state;
  Position plot;
  int desired_crop{-1};
  // Exact observed seed count at each rolling tick. This is receipt state,
  // never a forecast or an oracle grant.
  // Scenario-only receipt trace. A native owner may pass only a length-one
  // current suffix and re-invoke next tick; it must never read future entries.
  std::vector<int> scenario_seed_receipt_by_turn;
  int day{};
};

struct ElasticDayResult {
  std::vector<Action> manifest;
  std::vector<Position> positions_before;
  std::vector<std::uint64_t> move_source_order;
  PersistentPlotIntent debt;
  bool has_debt{};
  bool purchase_requested{};
  int purchase_request_turn{-1};
  int delayed_moves{};
  int total_move_delay{};
  int maximum_move_delay{};
  int absorbed_pass_sinks{};
  int absorbed_no_effect_sinks{};
  int assignments{};
  int terminal_move_tokens{};
  int terminal_hard_raw_tokens{};
  bool move_source_order_exact{};
  bool move_payload_exact{};
  bool capacity_proof_held{};
  bool scenario_shadow_only{true};
};

// Single-affected-actor scenario-shadow MVP. Input must start at the already
// observed trigger (`trigger_source[0] == true`); there is no pre-trigger
// prefix to advance or rewrite. It consumes MOVE as source tokens rather
// than shifting an action array. A MOVE is delayed only while the remaining
// horizon proves capacity for every MOVE, every non-sink raw obligation, and
// the lifecycle transitions still needed at the current plot.
[[nodiscard]] ElasticDayResult compile_event_local_elastic_day(
    const ElasticDayInput& input);

}  // namespace g001::day_horizon_repair
