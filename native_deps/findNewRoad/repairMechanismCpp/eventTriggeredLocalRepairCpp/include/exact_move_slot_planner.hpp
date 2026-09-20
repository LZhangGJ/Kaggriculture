#pragma once

#include "day_horizon_planner.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace g001::day_horizon_repair {

struct ExactMoveSlotActor {
  int actor{-1};
  Position start;
  std::vector<Action> raw;
  // True means the caller has compiled this stationary raw action into an
  // intent, or has certified PASS as a service opportunity. MOVE is never a
  // service slot even if the bit is accidentally set.
  std::vector<bool> service_slot;
  // Optional. When present, only these raw crop actions create/refresh an
  // intent. The caller sets this after proving an observation deviation.
  // Empty preserves the full-recompile oracle behavior.
  std::vector<bool> trigger_source;
};

struct PersistentPlotIntent {
  std::uint64_t id{};
  Position tile;
  int desired_crop{-1};
  bool harvest_before_replant{};
  int origin_day{};
};

enum class ExactSlotReject : std::uint8_t {
  MissingDesiredCrop,
  SeedUnavailable,
  MaturityWait,
  TileUnsupported,
  TileSerialized,
  OngoingHarvestUnsupported,
  GlobalSeedManifestUnsafe,
};

struct ExactMoveSlotResult {
  std::vector<std::vector<Action>> manifest;
  std::vector<std::vector<Position>> positions_before;
  std::vector<PersistentPlotIntent> active_intents;
  std::vector<std::uint64_t> completed_intents;
  std::map<Position, event_local_repair::TileObservation> final_tiles;
  std::map<int, int> remaining_seeds;
  std::map<ExactSlotReject, int> rejected;
  int absorbed_raw_intents{};
  int assignments{};
  int state_equivalent_changes{};
  int certified_service_changes{};
  std::set<Position> affected_tiles;
  std::set<int> affected_actors;
  std::map<event_local_repair::Op, int> raw_nonmove_changes;
  bool move_slots_exact{};
  bool move_positions_exact{};
  int original_move_slot_changes{};
  int original_move_payload_changes{};
};

// Offline crop MVP. Every raw MOVE remains at its exact original actor/turn
// slot. Known crop production and PASS slots are state-driven service
// opportunities. Unfinished plot intents are returned as plot-owned debt and
// may be supplied to the next day/visit; they are never attached to an actor.
[[nodiscard]] ExactMoveSlotResult compile_exact_move_slots(
    const std::vector<ExactMoveSlotActor>& actors,
    std::map<Position, event_local_repair::TileObservation> tiles,
    std::map<int, int> seeds,
    std::vector<PersistentPlotIntent> carried = {}, int day = 0);

}  // namespace g001::day_horizon_repair
