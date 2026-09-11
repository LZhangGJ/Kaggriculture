// Licensed under the Apache License, Version 2.0.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace fastkag {

// Five crops plus three animals.  The candidate kernel deliberately knows
// only project indices and public execution constraints; it contains no
// Replay route, author, opponent identity, calendar or map coordinate.
constexpr int ADAPTIVE_PROJECTS = 8;
constexpr int ADAPTIVE_PRODUCTS = 9;

enum class CandidateFamily : int8_t {
  KEEP = 0,
  SCHEDULE_LAYOUT = 1,
  CONTINUOUS_SCALE = 2,
  UNILATERAL = 3,
  MULTI_PROJECT = 4,
  TIMING = 5,
  MARKET_TRANSACTION = 6,
  PHASE_SUFFIX = 7,
  LOCAL_RECOVERY = 8,
};

enum class ScheduleProfile : int8_t {
  CURRENT = 0,
  MIN_TOTAL_TRAVEL = 1,
  DEADLINE_FIRST = 2,
  VALUE_PER_STEP = 3,
  REGION_BALANCED = 4,
  GLOBAL_MATCHING = 5,
  CHAIN_CONTINUITY = 6,
  COMPACT_LAYOUT = 7,
  SERVICE_LANES = 8,
};

enum class MarketProfile : int8_t {
  CURRENT = 0,
  SELL_NOW = 1,
  HOLD_FOR_DEMAND = 2,
  PARTIAL_SELL = 3,
  SELL_TO_FINANCE = 4,
};

enum class RecoveryProfile : int8_t {
  CURRENT = 0,
  REPAIR_NOW = 1,
  DEFER_LOW_VALUE = 2,
  ABANDON_LOW_VALUE = 3,
};

enum RecoveryIssue : uint8_t {
  RECOVERY_NONE = 0,
  RECOVERY_WEED = 1 << 0,
  RECOVERY_PARTIAL_FILL = 1 << 1,
  RECOVERY_UNIT_MISALIGNMENT = 1 << 2,
  RECOVERY_BROKEN_CHAIN = 1 << 3,
};

struct AdaptiveCandidateContext {
  int day = 0;
  std::array<int16_t, ADAPTIVE_PROJECTS> targets{};
  std::array<int16_t, ADAPTIVE_PROJECTS> irreversible_floor{};
  std::array<int16_t, ADAPTIVE_PROJECTS> caps{};
  // Approximate marginal values are only a cheap first-stage screen.  The
  // full native continuation remains the authority for offline labels.
  std::array<double, ADAPTIVE_PROJECTS> marginal_value{};
  std::array<double, ADAPTIVE_PROJECTS> purchase_cost{};
  std::array<double, ADAPTIVE_PROJECTS> daily_action_load{};
  std::array<int16_t, ADAPTIVE_PROJECTS> first_cash_lag_days{};

  double liquid_cash = 0.0;
  double protected_cash = 0.0;
  double financeable_inventory_value = 0.0;
  int unlocked_quadrants = 1;
  int maximum_quadrants = 4;
  int productive_tiles = 0;
  int hands = 0;
  int maximum_hands = 12;
  double next_hand_cost = 0.0;
  double next_quadrant_cost = 0.0;
  int tiles_per_quadrant = 25;
  double current_daily_action_load = 0.0;
  double hard_deadline_load = 0.0;
  double estimated_travel_load = 0.0;
  double delayed_loss = 0.0;
  int market_slots_available = 10;
  std::array<int16_t, ADAPTIVE_PRODUCTS> sellable_inventory{};
  std::array<int16_t, ADAPTIVE_PRODUCTS> market_prices{};
  std::array<int16_t, ADAPTIVE_PRODUCTS> demand_within_day{};
  uint8_t recovery_issues = RECOVERY_NONE;
};

struct AdaptivePlanDelta {
  CandidateFamily family = CandidateFamily::KEEP;
  std::array<int16_t, ADAPTIVE_PROJECTS> target_delta{};
  // Capacity belongs to the same transaction as the projects that require
  // it; it is not an independent route choice.
  int8_t hand_delta = 0;
  int8_t quadrant_delta = 0;
  int8_t effective_delay_days = 0;
  int8_t schedule_profile = int8_t(ScheduleProfile::CURRENT);
  int8_t market_profile = int8_t(MarketProfile::CURRENT);
  int8_t recovery_profile = int8_t(RecoveryProfile::CURRENT);
  int8_t suffix_project = -1;
  int8_t market_item = -1;
  uint8_t recovery_issue = RECOVERY_NONE;
  double estimated_value = 0.0;
  double estimated_cash_cost = 0.0;
  double estimated_daily_action_load = 0.0;
  uint64_t signature = 0;
};

struct AdaptiveCandidateSet {
  int raw_count = 0;
  int feasible_count = 0;
  int duplicate_count = 0;
  std::array<int16_t, 9> raw_by_family{};
  std::array<int16_t, 9> feasible_by_family{};
  std::array<int16_t, 9> shortlisted_by_family{};
  std::vector<AdaptivePlanDelta> feasible;
  std::vector<AdaptivePlanDelta> shortlist;
};

std::string_view candidate_family_name(CandidateFamily family);

// Generate a broad but structured set and keep a diverse, soft-quota
// shortlist.  max_raw bounds only the accepted unique raw set; generation is
// deterministic, so identical public state always produces identical output.
AdaptiveCandidateSet generate_adaptive_candidates(
    const AdaptiveCandidateContext& context,
    int max_shortlist = 64,
    int max_raw = 1000);

}  // namespace fastkag
