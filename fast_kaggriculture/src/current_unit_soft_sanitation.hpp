// Licensed under the Apache License, Version 2.0.
#pragma once

#include "simulator.hpp"

#include <array>
#include <string>
#include <vector>

namespace fastkag {

constexpr int kProductionProtectionSteps = 220;

enum class SoftMissKind : std::int8_t {
  MissingSeed,
  MissingCarriedWheat,
  MissingCarriedFertilizer,
  MissingCarriedPlaceItem,
};

enum class SoftRecovery : std::int8_t { Pickup, Pass };

struct SoftMiss {
  int step = -1;
  int actor = -1;
  Op original_op = Op::PASS;
  Item item = Item::NONE;
  int required = 0;
  int available = 0;
  int shortfall = 0;
  SoftMissKind kind = SoftMissKind::MissingSeed;
  SoftRecovery recovery = SoftRecovery::Pass;
  // Future predictable needs belong to the hard-obligation compiler. Every
  // miss handled here is an unexpected, already-unrecoverable current phase.
  bool unexpected_current_miss = true;
  bool production_protection_window = false;
  // Uncalibrated mechanical severity: one lost unit-action slot, resource
  // shortfall, and a one-step PICKUP or two-step/unknown PASS recovery delay.
  double loss_proxy = 0.0;
  std::string reason;
};

// Own current-state snapshot only.  It deliberately has no market orders,
// future frames, route/opponent identity, clone, or realized-future field.
struct CurrentUnitSanitationInput {
  int step = 0;
  int board_size = 10;
  int shed_capacity = 100;
  std::array<int, N_ITEMS> shed{};
  std::array<int, N_CROPS> seeds{};
  std::vector<Position> actor_positions;
  std::vector<std::array<int, N_ITEMS>> carried;
  std::vector<std::vector<int8_t>> carried_order;
  std::vector<Tile> tiles;
  std::vector<Action> units;
};

struct CurrentUnitSanitationResult {
  std::vector<Action> units;
  std::vector<SoftMiss> misses;
  bool production_protection_window = false;
  int pickup_substitutions = 0;
  int pass_substitutions = 0;
  int move_actions_preserved = 0;
  int unexpected_current_misses = 0;
  int production_protection_misses = 0;
  double total_loss_proxy = 0.0;
};

[[nodiscard]] CurrentUnitSanitationInput current_unit_sanitation_input(
    const Simulator& env, int player, const std::vector<Action>& units);

[[nodiscard]] CurrentUnitSanitationResult sanitize_current_units(
    const CurrentUnitSanitationInput& input);

const char* soft_miss_kind_name(SoftMissKind kind);
const char* soft_recovery_name(SoftRecovery recovery);

}  // namespace fastkag
