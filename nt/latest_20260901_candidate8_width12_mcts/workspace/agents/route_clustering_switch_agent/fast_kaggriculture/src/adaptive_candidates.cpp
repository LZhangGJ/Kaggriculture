// Licensed under the Apache License, Version 2.0.
#include "adaptive_candidates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_set>
#include <utility>

namespace fastkag {
namespace {

constexpr std::array<int, 6> SCALE_STEPS = {1, 2, 3, 4, 6, 8};

uint64_t mix(uint64_t hash, int64_t value) {
  // Stable FNV-1a over integer fields.  Candidate generation is offline and
  // deterministic; this signature is only a duplicate key, never a policy
  // feature.
  constexpr uint64_t prime = 1099511628211ULL;
  uint64_t bits = static_cast<uint64_t>(value);
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (bits >> (8 * byte)) & 0xffULL;
    hash *= prime;
  }
  return hash;
}

uint64_t signature_for(const AdaptivePlanDelta& delta) {
  uint64_t hash = 1469598103934665603ULL;
  hash = mix(hash, int(delta.family));
  for (const int value : delta.target_delta) hash = mix(hash, value);
  hash = mix(hash, delta.hand_delta);
  hash = mix(hash, delta.quadrant_delta);
  hash = mix(hash, delta.effective_delay_days);
  hash = mix(hash, delta.schedule_profile);
  hash = mix(hash, delta.market_profile);
  hash = mix(hash, delta.recovery_profile);
  hash = mix(hash, delta.suffix_project);
  hash = mix(hash, delta.market_item);
  hash = mix(hash, delta.recovery_issue);
  return hash;
}

bool changes_behavior(const AdaptivePlanDelta& delta) {
  if (delta.family == CandidateFamily::KEEP) return true;
  if (std::any_of(delta.target_delta.begin(), delta.target_delta.end(),
                  [](int value) { return value != 0; }))
    return true;
  return delta.hand_delta != 0 || delta.quadrant_delta != 0 ||
      delta.effective_delay_days != 0 ||
      delta.schedule_profile != int8_t(ScheduleProfile::CURRENT) ||
      delta.market_profile != int8_t(MarketProfile::CURRENT) ||
      delta.recovery_profile != int8_t(RecoveryProfile::CURRENT) ||
      delta.suffix_project >= 0 || delta.market_item >= 0 ||
      delta.recovery_issue != RECOVERY_NONE;
}

double family_bonus(const AdaptiveCandidateContext& context,
                    const AdaptivePlanDelta& delta) {
  switch (delta.family) {
    case CandidateFamily::KEEP:
      return 0.0;
    case CandidateFamily::SCHEDULE_LAYOUT: {
      switch (ScheduleProfile(delta.schedule_profile)) {
        case ScheduleProfile::MIN_TOTAL_TRAVEL:
          return 0.20 * context.estimated_travel_load;
        case ScheduleProfile::DEADLINE_FIRST:
          return 0.35 * context.hard_deadline_load +
                 0.20 * context.delayed_loss;
        case ScheduleProfile::VALUE_PER_STEP:
          return 0.10 * context.delayed_loss +
                 0.08 * context.estimated_travel_load;
        case ScheduleProfile::REGION_BALANCED:
        case ScheduleProfile::GLOBAL_MATCHING:
          return 0.16 * context.estimated_travel_load +
                 0.08 * context.hard_deadline_load;
        case ScheduleProfile::CHAIN_CONTINUITY:
          return 0.14 * context.estimated_travel_load +
                 0.06 * context.delayed_loss;
        case ScheduleProfile::COMPACT_LAYOUT:
        case ScheduleProfile::SERVICE_LANES:
          return 0.12 * context.estimated_travel_load;
        default:
          return 0.0;
      }
    }
    case CandidateFamily::TIMING: {
      double outstanding = 0.0;
      for (int project = 0; project < ADAPTIVE_PROJECTS; ++project)
        if (delta.target_delta[project] < 0)
          outstanding += -delta.target_delta[project] *
              std::max(0.0, context.purchase_cost[project]);
      // Deferral protects liquidity but loses a fraction of future value.
      return 0.12 * outstanding -
          0.04 * delta.effective_delay_days * context.delayed_loss;
    }
    case CandidateFamily::MARKET_TRANSACTION: {
      if (delta.market_item < 0 ||
          delta.market_item >= ADAPTIVE_PRODUCTS)
        return -1e9;
      const int item = delta.market_item;
      const double stock = context.sellable_inventory[item];
      const double quote = context.market_prices[item];
      const double demand = context.demand_within_day[item];
      switch (MarketProfile(delta.market_profile)) {
        case MarketProfile::SELL_NOW:
          return stock * quote;
        case MarketProfile::HOLD_FOR_DEMAND:
          return stock * (quote + std::max(0.0, 0.25 * demand));
        case MarketProfile::PARTIAL_SELL:
          return 0.55 * stock * quote;
        case MarketProfile::SELL_TO_FINANCE:
          return stock * quote + 0.20 * context.delayed_loss;
        default:
          return 0.0;
      }
    }
    case CandidateFamily::PHASE_SUFFIX:
      return delta.suffix_project >= 0
          ? 0.25 * context.marginal_value[delta.suffix_project]
          : 0.0;
    case CandidateFamily::LOCAL_RECOVERY: {
      switch (RecoveryProfile(delta.recovery_profile)) {
        case RecoveryProfile::REPAIR_NOW:
          return context.delayed_loss + 0.25 * context.hard_deadline_load;
        case RecoveryProfile::DEFER_LOW_VALUE:
          return 0.35 * context.delayed_loss;
        case RecoveryProfile::ABANDON_LOW_VALUE:
          return context.delayed_loss < 100.0
              ? 0.45 * (100.0 - context.delayed_loss)
              : -0.50 * context.delayed_loss;
        default:
          return 0.0;
      }
    }
    default:
      return 0.0;
  }
}

bool normalise_and_score(const AdaptiveCandidateContext& context,
                         AdaptivePlanDelta& delta) {
  if (!changes_behavior(delta)) return false;
  // KEEP is the already-selected executable plan, not a request to repurchase
  // every outstanding commitment in one transaction.  It must remain the
  // safe baseline even when today's free cash cannot fund all future capacity
  // at once; otherwise shortlist rank zero silently becomes a risky edit.
  if (delta.family == CandidateFamily::KEEP) {
    delta.estimated_value = 0.0;
    delta.estimated_cash_cost = 0.0;
    delta.estimated_daily_action_load = context.current_daily_action_load;
    delta.signature = signature_for(delta);
    return true;
  }
  std::array<int, ADAPTIVE_PROJECTS> targets{};
  double cash_cost = 0.0;
  double daily_load = context.current_daily_action_load;
  int tile_delta = 0;
  double value = 0.0;
  for (int project = 0; project < ADAPTIVE_PROJECTS; ++project) {
    const int target = int(context.targets[project]) +
        int(delta.target_delta[project]);
    // A live project may already exceed today's newly-computed investment
    // cap.  Grandfather the current commitment: unchanged and reducing edits
    // stay legal, while further expansion is still blocked by this ceiling.
    const int effective_cap = std::max(
        int(context.caps[project]), int(context.targets[project]));
    if (target < int(context.irreversible_floor[project]) ||
        target > effective_cap)
      return false;
    targets[project] = target;
    const int change = int(delta.target_delta[project]);
    if (change > 0) {
      if (context.day + delta.effective_delay_days +
              int(context.first_cash_lag_days[project]) >= 30)
        return false;
      cash_cost += change * std::max(0.0, context.purchase_cost[project]);
    }
    value += change * context.marginal_value[project];
    daily_load += change * context.daily_action_load[project];
    tile_delta += change;
  }
  daily_load = std::max(0.0, daily_load);

  const int tiles_per_quadrant = std::max(1, context.tiles_per_quadrant);
  const int required_tiles = std::max(0, context.productive_tiles + tile_delta);
  const int required_quadrants = std::max(
      context.unlocked_quadrants,
      (required_tiles + tiles_per_quadrant - 1) / tiles_per_quadrant);
  if (required_quadrants > context.maximum_quadrants) return false;
  delta.quadrant_delta = int8_t(std::max(
      int(delta.quadrant_delta), required_quadrants - context.unlocked_quadrants));

  const int required_units = std::max(
      1, int(std::ceil(daily_load / 24.0)));
  const int required_hands = required_units - 1;
  if (required_hands > context.maximum_hands) return false;
  delta.hand_delta = int8_t(std::max(
      int(delta.hand_delta), required_hands - context.hands));
  if (context.hands + delta.hand_delta > context.maximum_hands)
    return false;

  cash_cost += std::max(0, int(delta.hand_delta)) *
      std::max(0.0, context.next_hand_cost);
  cash_cost += std::max(0, int(delta.quadrant_delta)) *
      std::max(0.0, context.next_quadrant_cost);
  double deployable = std::max(0.0, context.liquid_cash -
      context.protected_cash);
  if (delta.market_profile == int8_t(MarketProfile::SELL_TO_FINANCE))
    deployable += std::max(0.0, context.financeable_inventory_value);
  if (cash_cost > deployable + 1e-9) return false;

  if (delta.market_profile != int8_t(MarketProfile::CURRENT)) {
    const int slots_needed = delta.market_profile ==
            int8_t(MarketProfile::SELL_TO_FINANCE)
        ? 2 : 1;
    if (context.market_slots_available < slots_needed) return false;
    if (delta.market_item < 0 ||
        context.sellable_inventory[delta.market_item] <= 0)
      return false;
  }
  if (delta.family == CandidateFamily::LOCAL_RECOVERY &&
      (context.recovery_issues & delta.recovery_issue) == 0)
    return false;

  // Capacity costs are part of the same project transaction.  The heuristic
  // is intentionally cheap and conservative; native common-random-number
  // continuations supply the actual label.
  value -= cash_cost;
  value -= 0.20 * std::max(0.0, daily_load -
      24.0 * (context.hands + 1));
  value += family_bonus(context, delta);
  delta.estimated_cash_cost = cash_cost;
  delta.estimated_daily_action_load = daily_load;
  delta.estimated_value = value;
  delta.signature = signature_for(delta);
  return std::isfinite(value);
}

void add_candidate(std::vector<AdaptivePlanDelta>& raw,
                   AdaptivePlanDelta candidate,
                   int max_raw) {
  if (int(raw.size()) >= max_raw) return;
  raw.push_back(std::move(candidate));
}

double approximate_transaction_cost(
    const AdaptiveCandidateContext& context,
    const AdaptivePlanDelta& delta) {
  double cash_cost = 0.0;
  double daily_load = context.current_daily_action_load;
  int tile_delta = 0;
  for (int project = 0; project < ADAPTIVE_PROJECTS; ++project) {
    const int change = int(delta.target_delta[project]);
    if (change > 0)
      cash_cost += change * std::max(0.0, context.purchase_cost[project]);
    daily_load += change * context.daily_action_load[project];
    tile_delta += change;
  }
  daily_load = std::max(0.0, daily_load);
  const int tiles_per_quadrant = std::max(1, context.tiles_per_quadrant);
  const int required_tiles = std::max(0, context.productive_tiles + tile_delta);
  const int required_quadrants = std::max(
      context.unlocked_quadrants,
      (required_tiles + tiles_per_quadrant - 1) / tiles_per_quadrant);
  const int required_hands = std::max(
      0, int(std::ceil(daily_load / 24.0)) - 1);
  cash_cost += std::max(0, required_hands - context.hands) *
      std::max(0.0, context.next_hand_cost);
  cash_cost += std::max(0, required_quadrants - context.unlocked_quadrants) *
      std::max(0.0, context.next_quadrant_cost);
  return cash_cost;
}

int best_finance_item(const AdaptiveCandidateContext& context) {
  int best = -1;
  int64_t best_value = 0;
  for (int item = 0; item < ADAPTIVE_PRODUCTS; ++item) {
    const int64_t value = int64_t(context.sellable_inventory[item]) *
        int64_t(std::max<int16_t>(0, context.market_prices[item]));
    if (value > best_value) {
      best = item;
      best_value = value;
    }
  }
  return best;
}

// Add a bounded set of compositional transaction variants after all eight
// semantic families have been generated. A positive project edit that is
// unaffordable from protected liquid cash but affordable after selling public
// inventory must remain expressible as one ordered plan: SELL first, then
// capacity/project purchases. Only the best currently sellable item is used
// as the forced first sale; the live transaction compiler may sell additional
// inventory if the projected cash ledger still has a shortfall.
void add_financed_project_variants(
    const AdaptiveCandidateContext& context,
    std::vector<AdaptivePlanDelta>& raw,
    int max_raw) {
  if (context.market_slots_available < 2 ||
      context.financeable_inventory_value <= 0.0)
    return;
  const int finance_item = best_finance_item(context);
  if (finance_item < 0) return;
  const double liquid = std::max(
      0.0, context.liquid_cash - context.protected_cash);
  const double deployable = liquid +
      std::max(0.0, context.financeable_inventory_value);

  struct Variant {
    AdaptivePlanDelta delta;
    double rough_value = 0.0;
  };
  std::vector<Variant> variants;
  variants.reserve(raw.size());
  for (const auto& base : raw) {
    if (base.family == CandidateFamily::KEEP ||
        base.family == CandidateFamily::MARKET_TRANSACTION ||
        base.market_profile != int8_t(MarketProfile::CURRENT) ||
        !std::any_of(base.target_delta.begin(), base.target_delta.end(),
                     [](int change) { return change > 0; }))
      continue;
    const double transaction_cost =
        approximate_transaction_cost(context, base);
    if (transaction_cost <= liquid + 1e-9 ||
        transaction_cost > deployable + 1e-9)
      continue;
    Variant variant;
    variant.delta = base;
    variant.delta.market_profile = int8_t(MarketProfile::SELL_TO_FINANCE);
    variant.delta.market_item = int8_t(finance_item);
    variant.rough_value = -transaction_cost;
    for (int project = 0; project < ADAPTIVE_PROJECTS; ++project)
      variant.rough_value += base.target_delta[project] *
          context.marginal_value[project];
    variants.push_back(std::move(variant));
  }
  std::sort(variants.begin(), variants.end(), [](const auto& lhs,
                                                 const auto& rhs) {
    if (lhs.rough_value != rhs.rough_value)
      return lhs.rough_value > rhs.rough_value;
    return signature_for(lhs.delta) < signature_for(rhs.delta);
  });
  // Keep room for every ordinary semantic family and prevent a low-cash state
  // from being monopolised by financing variants of the same edit grammar.
  const int variant_budget = std::min(
      96, std::max(0, max_raw - int(raw.size())));
  for (int i = 0; i < std::min<int>(variant_budget, variants.size()); ++i)
    add_candidate(raw, std::move(variants[i].delta), max_raw);
}

int family_index(CandidateFamily family) {
  return std::clamp(int(family), 0, 8);
}

}  // namespace

std::string_view candidate_family_name(CandidateFamily family) {
  switch (family) {
    case CandidateFamily::KEEP: return "KEEP";
    case CandidateFamily::SCHEDULE_LAYOUT: return "SCHEDULE_LAYOUT";
    case CandidateFamily::CONTINUOUS_SCALE: return "CONTINUOUS_SCALE";
    case CandidateFamily::UNILATERAL: return "UNILATERAL";
    case CandidateFamily::MULTI_PROJECT: return "MULTI_PROJECT";
    case CandidateFamily::TIMING: return "TIMING";
    case CandidateFamily::MARKET_TRANSACTION: return "MARKET_TRANSACTION";
    case CandidateFamily::PHASE_SUFFIX: return "PHASE_SUFFIX";
    case CandidateFamily::LOCAL_RECOVERY: return "LOCAL_RECOVERY";
  }
  return "UNKNOWN";
}

AdaptiveCandidateSet generate_adaptive_candidates(
    const AdaptiveCandidateContext& context,
    int max_shortlist,
    int max_raw) {
  max_shortlist = std::clamp(max_shortlist, 1, 256);
  max_raw = std::clamp(max_raw, max_shortlist, 4096);
  std::vector<AdaptivePlanDelta> raw;
  raw.reserve(std::min(max_raw, 1024));

  add_candidate(raw, AdaptivePlanDelta{}, max_raw);

  // 1. Same business commitment, several generic execution/layout solvers.
  for (int profile = int(ScheduleProfile::MIN_TOTAL_TRAVEL);
       profile <= int(ScheduleProfile::SERVICE_LANES); ++profile) {
    AdaptivePlanDelta candidate;
    candidate.family = CandidateFamily::SCHEDULE_LAYOUT;
    candidate.schedule_profile = int8_t(profile);
    add_candidate(raw, candidate, max_raw);
  }

  // 2. Continuous scale applies only to an already active project.  Each
  // quantity is a parameter, not an observed expert target.
  for (int project = 0; project < ADAPTIVE_PROJECTS; ++project) {
    if (context.targets[project] <= context.irreversible_floor[project])
      continue;
    for (const int amount : SCALE_STEPS) {
      for (const int direction : {-1, 1}) {
        AdaptivePlanDelta candidate;
        candidate.family = CandidateFamily::CONTINUOUS_SCALE;
        candidate.target_delta[project] = int16_t(direction * amount);
        add_candidate(raw, candidate, max_raw);
      }
    }
  }

  // 3. Start or retire one project without forcing a compensating edit.
  for (int project = 0; project < ADAPTIVE_PROJECTS; ++project) {
    if (context.targets[project] <= context.irreversible_floor[project]) {
      for (const int amount : SCALE_STEPS) {
        AdaptivePlanDelta candidate;
        candidate.family = CandidateFamily::UNILATERAL;
        candidate.target_delta[project] = int16_t(amount);
        add_candidate(raw, candidate, max_raw);
      }
    } else {
      AdaptivePlanDelta candidate;
      candidate.family = CandidateFamily::UNILATERAL;
      candidate.target_delta[project] = int16_t(
          int(context.irreversible_floor[project]) -
          int(context.targets[project]));
      add_candidate(raw, candidate, max_raw);
    }
  }

  // 4a. Two-project release/reinvest and joint expansion.
  constexpr std::array<int, 4> bundle_steps = {1, 2, 4, 6};
  for (int first = 0; first < ADAPTIVE_PROJECTS; ++first) {
    for (int second = first + 1; second < ADAPTIVE_PROJECTS; ++second) {
      for (const int amount : bundle_steps) {
        for (const auto directions : {
                 std::pair{-1, 1}, std::pair{1, -1}, std::pair{1, 1}}) {
          AdaptivePlanDelta candidate;
          candidate.family = CandidateFamily::MULTI_PROJECT;
          candidate.target_delta[first] = int16_t(directions.first * amount);
          candidate.target_delta[second] = int16_t(directions.second * amount);
          add_candidate(raw, candidate, max_raw);
        }
      }
    }
  }
  // 4b. Three-project edits use a small generic beam seed.  The later value
  // and diversity screen, not hard-coded industries, decides which survive.
  for (int source = 0; source < ADAPTIVE_PROJECTS; ++source) {
    for (int first = 0; first < ADAPTIVE_PROJECTS; ++first) {
      if (first == source) continue;
      for (int second = first + 1; second < ADAPTIVE_PROJECTS; ++second) {
        if (second == source) continue;
        // Two small beam seeds keep the complete broad pool below roughly
        // one thousand candidates while still allowing capacity-crossing
        // three-project edits.  Larger scales remain available through the
        // two-project and continuous-scale families.
        for (const int amount : {1, 2}) {
          AdaptivePlanDelta candidate;
          candidate.family = CandidateFamily::MULTI_PROJECT;
          candidate.target_delta[source] = int16_t(-amount);
          candidate.target_delta[first] = int16_t(amount);
          candidate.target_delta[second] = int16_t(amount);
          add_candidate(raw, candidate, max_raw);
        }
      }
    }
  }

  // 5. Timing changes only future reversible commitments.  A one/two-day
  // delay and a stop-adding arm are distinct candidates.
  for (int project = 0; project < ADAPTIVE_PROJECTS; ++project) {
    const int reversible = int(context.targets[project]) -
        int(context.irreversible_floor[project]);
    if (reversible <= 0) continue;
    for (const int delay : {1, 2}) {
      AdaptivePlanDelta candidate;
      candidate.family = CandidateFamily::TIMING;
      candidate.effective_delay_days = int8_t(delay);
      candidate.target_delta[project] = int16_t(-reversible);
      add_candidate(raw, candidate, max_raw);
    }
    AdaptivePlanDelta stop;
    stop.family = CandidateFamily::TIMING;
    stop.effective_delay_days = -1;  // persistent STOP_ADDING semantics
    stop.target_delta[project] = int16_t(-reversible);
    add_candidate(raw, stop, max_raw);
  }

  // 6. Ordered transaction profiles exist only for currently sellable stock.
  for (int item = 0; item < ADAPTIVE_PRODUCTS; ++item) {
    if (context.sellable_inventory[item] <= 0) continue;
    for (int profile = int(MarketProfile::SELL_NOW);
         profile <= int(MarketProfile::SELL_TO_FINANCE); ++profile) {
      AdaptivePlanDelta candidate;
      candidate.family = CandidateFamily::MARKET_TRANSACTION;
      candidate.market_profile = int8_t(profile);
      candidate.market_item = int8_t(item);
      add_candidate(raw, candidate, max_raw);
    }
  }

  // 7. Phase suffixes change only future crop capacity.  Both add and stop
  // arms are produced for every crop whose official horizon remains feasible.
  for (int crop = 0; crop < 5; ++crop) {
    for (const int amount : {2, 4, 6}) {
      AdaptivePlanDelta candidate;
      candidate.family = CandidateFamily::PHASE_SUFFIX;
      candidate.suffix_project = int8_t(crop);
      candidate.target_delta[crop] = int16_t(amount);
      add_candidate(raw, candidate, max_raw);
    }
    const int reversible = int(context.targets[crop]) -
        int(context.irreversible_floor[crop]);
    if (reversible > 0) {
      AdaptivePlanDelta candidate;
      candidate.family = CandidateFamily::PHASE_SUFFIX;
      candidate.suffix_project = int8_t(crop);
      candidate.target_delta[crop] = int16_t(-reversible);
      add_candidate(raw, candidate, max_raw);
    }
  }

  // 8. Recovery candidates are generated only for a real live issue.  The
  // unchanged remainder of the plan is implicit in the sparse delta.
  for (const uint8_t issue : {uint8_t(RECOVERY_WEED),
                              uint8_t(RECOVERY_PARTIAL_FILL),
                              uint8_t(RECOVERY_UNIT_MISALIGNMENT),
                              uint8_t(RECOVERY_BROKEN_CHAIN)}) {
    if ((context.recovery_issues & issue) == 0) continue;
    for (int profile = int(RecoveryProfile::REPAIR_NOW);
         profile <= int(RecoveryProfile::ABANDON_LOW_VALUE); ++profile) {
      AdaptivePlanDelta candidate;
      candidate.family = CandidateFamily::LOCAL_RECOVERY;
      candidate.recovery_profile = int8_t(profile);
      candidate.recovery_issue = issue;
      add_candidate(raw, candidate, max_raw);
    }
  }

  // Compose project/capacity edits with a sell-first financing transaction
  // only when the live cash ledger proves it is needed and affordable.
  add_financed_project_variants(context, raw, max_raw);

  AdaptiveCandidateSet result;
  result.raw_count = int(raw.size());
  for (const auto& candidate : raw)
    result.raw_by_family[family_index(candidate.family)]++;

  std::unordered_set<uint64_t> seen;
  result.feasible.reserve(raw.size());
  for (auto candidate : raw) {
    if (!normalise_and_score(context, candidate)) continue;
    if (!seen.insert(candidate.signature).second) {
      result.duplicate_count++;
      continue;
    }
    result.feasible_by_family[family_index(candidate.family)]++;
    result.feasible.push_back(std::move(candidate));
  }
  result.feasible_count = int(result.feasible.size());

  // Soft reservations.  Missing capacity returns to the global value/diversity
  // pool instead of producing an empty fixed slot.
  // Replay capability audits suggested broad SCALE and multi-project capacity
  // edits; stable 32-future counterfactuals then confirmed these two families
  // caused most old-shortlist regret.  This remains a soft reservation (empty
  // slots return to the global pool), not a claim about any hidden policy
  // implementation or a fixed route frequency.
  const std::array<int, 9> quota64 = {1, 8, 15, 5, 18, 5, 6, 3, 3};
  std::array<std::vector<int>, 9> by_family;
  for (int index = 0; index < int(result.feasible.size()); ++index)
    by_family[family_index(result.feasible[index].family)].push_back(index);
  for (auto& indices : by_family)
    std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
      if (result.feasible[lhs].estimated_value !=
          result.feasible[rhs].estimated_value)
        return result.feasible[lhs].estimated_value >
               result.feasible[rhs].estimated_value;
      return result.feasible[lhs].signature <
             result.feasible[rhs].signature;
    });

  std::vector<uint8_t> selected(result.feasible.size(), 0);
  auto select_index = [&](int index) {
    if (index < 0 || index >= int(result.feasible.size()) || selected[index])
      return;
    selected[index] = 1;
    result.shortlisted_by_family[
        family_index(result.feasible[index].family)]++;
    result.shortlist.push_back(result.feasible[index]);
  };
  for (int family = 0; family < 9 &&
                       int(result.shortlist.size()) < max_shortlist; ++family) {
    const int scaled_quota = std::max(
        family == 0 ? 1 : 0,
        int(std::floor(quota64[family] * max_shortlist / 64.0)));
    const int take = std::min<int>(scaled_quota, by_family[family].size());
    for (int i = 0; i < take; ++i) select_index(by_family[family][i]);
  }

  std::vector<int> remainder;
  remainder.reserve(result.feasible.size());
  for (int index = 0; index < int(result.feasible.size()); ++index)
    if (!selected[index]) remainder.push_back(index);
  std::sort(remainder.begin(), remainder.end(), [&](int lhs, int rhs) {
    const int lhs_family = family_index(result.feasible[lhs].family);
    const int rhs_family = family_index(result.feasible[rhs].family);
    // Mild diversity penalty stops the largest combinatorial family from
    // consuming every returned slot.
    const double lhs_score = result.feasible[lhs].estimated_value /
        (1.0 + 0.20 * result.shortlisted_by_family[lhs_family]);
    const double rhs_score = result.feasible[rhs].estimated_value /
        (1.0 + 0.20 * result.shortlisted_by_family[rhs_family]);
    if (lhs_score != rhs_score) return lhs_score > rhs_score;
    return result.feasible[lhs].signature < result.feasible[rhs].signature;
  });
  for (const int index : remainder) {
    if (int(result.shortlist.size()) >= max_shortlist) break;
    select_index(index);
  }
  return result;
}

}  // namespace fastkag
