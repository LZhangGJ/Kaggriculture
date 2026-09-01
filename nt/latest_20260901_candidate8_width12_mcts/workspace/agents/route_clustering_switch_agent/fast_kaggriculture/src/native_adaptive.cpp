// Licensed under the Apache License, Version 2.0.
#include "native_adaptive.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace fastkag {
namespace {

constexpr int TOTAL_DAYS = 30;
constexpr int BASE_INVENTORY = 10000;
constexpr std::array<int, N_CROPS> SEED_COST{10, 20, 50, 100, 80};
constexpr std::array<int, N_CROPS> CROP_FIRST{2, 2, 8, 10, 10};
constexpr std::array<int, N_CROPS> CROP_MAX_DAY{4, 3, 8, 10, 12};
constexpr std::array<int, N_CROPS> CROP_INTERVAL{0, 0, 1, 2, 0};
constexpr std::array<int, N_CROPS> CROP_MAX_YIELD{6, 4, 4, 4, 6};
constexpr std::array<bool, N_CROPS> CROP_ONGOING{false, false, true, true, false};
constexpr std::array<int, N_ANIMALS> ANIMAL_COST{300, 400, 500};
constexpr std::array<int, N_ANIMALS> ANIMAL_FIRST{4, 8, 6};
constexpr std::array<int, N_ANIMALS> ANIMAL_INTERVAL{1, 2, 3};
constexpr std::array<int, N_ANIMALS> ANIMAL_PRODUCT{5, 6, 7};
constexpr std::array<int, N_ANIMALS> ANIMAL_MAX_HELD{4, 6, 6};
constexpr std::array<int, N_PRODUCTS> BASE_PRICE{25, 35, 60, 120, 250, 50, 160, 200, 100};
constexpr std::array<int, N_PRODUCTS> MARKET_T{400, 450, 200, 100, 300, 332, 122, 105, 200};
constexpr std::array<int, 3> LAND_COST{1000, 2000, 4000};
constexpr std::array<Position, 4> SHED_POS{{{4, 4}, {5, 4}, {4, 5}, {5, 5}}};
constexpr int SHOP_PET_CAFE = 4;
constexpr int SHOP_YARN_STORE = 7;

bool crop_project_open(const AdaptiveGenome& genome, int crop, int day) {
  if (crop < 0 || crop >= N_CROPS) return false;
  // Preserve the legacy scalar gate exactly when the capability is disabled,
  // so controlled A/B tests have a genuine rollback arm.
  if (genome.crop_specific_terminal_horizon <= 0.0)
    return day <= genome.stop_new_crops_day;
  return day + CROP_FIRST[crop] < TOTAL_DAYS;
}

int latest_generic_crop_start(const AdaptiveGenome& genome) {
  if (genome.crop_specific_terminal_horizon <= 0.0)
    return genome.stop_new_crops_day;
  int latest = -1;
  for (int crop = 0; crop < N_CROPS; ++crop)
    latest = std::max(latest, TOTAL_DAYS - 1 - CROP_FIRST[crop]);
  return latest;
}

// Per shop and product daily-equivalent demand.  Simulator consumption occurs
// every four steps; the planner only needs a stable relative forecast.
constexpr int SHOP_DEMAND[8][N_PRODUCTS] = {
    {1, 0, 0, 0, 0, 1, 0, 0, 0},  // bakery
    {1, 0, 0, 1, 0, 1, 0, 0, 0},  // brunch
    {1, 1, 1, 1, 0, 0, 0, 0, 0},  // farmers market
    {1, 0, 0, 1, 0, 0, 1, 0, 0},  // ice cream
    {0, 2, 0, 0, 0, 0, 0, 0, 0},  // pet cafe
    {1, 0, 1, 0, 0, 0, 1, 0, 0},  // pizza
    {0, 0, 0, 1, 0, 0, 1, 0, 0},  // smoothie
    {0, 0, 0, 0, 0, 0, 0, 2, 0},  // yarn
};

enum Shape { LINEAR, SQ, SQRT, LOG, HINGE };
constexpr std::array<Shape, N_PRODUCTS> BELOW{
    SQRT, HINGE, HINGE, SQRT, LOG, HINGE, SQRT, LOG, LINEAR};
constexpr std::array<Shape, N_PRODUCTS> ABOVE{
    LOG, SQRT, SQRT, LINEAR, SQ, LOG, LINEAR, SQ, LINEAR};
constexpr std::array<double, N_PRODUCTS> BELOW_TARGET{.8, 1., .4, .7, .2, .4, .6, .2, .4};
constexpr std::array<double, N_PRODUCTS> ABOVE_TARGET{.2, .7, .6, 1.6, 3.6, .2, 1.6, 3.2, .4};

double shaped(Shape kind, double x, double t) {
  x = std::max(0.0, x);
  switch (kind) {
    case LINEAR: return x;
    case SQ: return x * x;
    case SQRT: return std::sqrt(x);
    case LOG: return std::log1p(x);
    case HINGE: {
      const double u = x / t;
      return u + 8.0 * std::pow(std::max(0.0, u - 1.0), 2.0);
    }
  }
  return x;
}

int predicted_price(int item, int inventory) {
  if (item < 0 || item >= N_PRODUCTS) return 1;
  const double base = BASE_PRICE[item], t = MARKET_T[item];
  double price = base;
  if (inventory < BASE_INVENTORY) {
    const double denom = shaped(BELOW[item], t, t);
    const double amp = denom > 0 ? BELOW_TARGET[item] * base / denom : 0.0;
    price = base + amp * shaped(BELOW[item], BASE_INVENTORY - inventory, t);
  } else {
    const double denom = shaped(ABOVE[item], t, t);
    const double amp = denom > 0 ? ABOVE_TARGET[item] * base / denom : 0.0;
    price = base - amp * shaped(ABOVE[item], inventory - BASE_INVENTORY, t);
  }
  return std::max(1, int(std::nearbyint(price)));
}

int pos_index(Position p, int board_size) { return int(p.y) * board_size + int(p.x); }
int distance(Position a, Position b) { return std::abs(int(a.x) - int(b.x)) + std::abs(int(a.y) - int(b.y)); }
bool same(Position a, Position b) { return a.x == b.x && a.y == b.y; }
bool shed_adjacent(Position p) {
  return (p.x == 4 || p.x == 5) && (p.y == 4 || p.y == 5);
}
int inventory_sum(const std::array<int32_t, N_ITEMS>& inv) {
  return std::accumulate(inv.begin(), inv.end(), 0);
}
int shed_sum(const PrivateState& state) {
  return std::accumulate(state.shed.begin(), state.shed.end(), 0);
}
int popcount(uint8_t value) { return __builtin_popcount(unsigned(value)); }

uint64_t behavior_mix(uint64_t hash, int64_t value) {
  constexpr uint64_t prime = 1099511628211ULL;
  const uint64_t bits = static_cast<uint64_t>(value);
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (bits >> (8 * byte)) & 0xffULL;
    hash *= prime;
  }
  return hash;
}

uint64_t player_action_signature(const PlayerAction& action) {
  uint64_t hash = 1469598103934665603ULL;
  hash = behavior_mix(hash, int(action.units.size()));
  for (const Action& unit : action.units) {
    hash = behavior_mix(hash, int(unit.op));
    hash = behavior_mix(hash, int(unit.item));
    hash = behavior_mix(hash, unit.quantity);
  }
  hash = behavior_mix(hash, -314159);
  hash = behavior_mix(hash, int(action.market.size()));
  for (const Action& market : action.market) {
    hash = behavior_mix(hash, int(market.op));
    hash = behavior_mix(hash, int(market.item));
    hash = behavior_mix(hash, market.quantity);
  }
  return hash;
}

uint64_t simulator_effect_signature(const Simulator& env) {
  uint64_t hash = 1469598103934665603ULL;
  hash = behavior_mix(hash, env.step_count());
  for (const Farm& farm : env.farms()) {
    hash = behavior_mix(hash, int64_t(std::nearbyint(100.0 * farm.money)));
    hash = behavior_mix(hash, farm.farmer.x);
    hash = behavior_mix(hash, farm.farmer.y);
    hash = behavior_mix(hash, int(farm.hands.size()));
    for (const Position hand : farm.hands) {
      hash = behavior_mix(hash, hand.x);
      hash = behavior_mix(hash, hand.y);
    }
    hash = behavior_mix(hash, farm.unlocked_mask);
    hash = behavior_mix(hash, farm.hires_today);
    hash = behavior_mix(hash, int(farm.tiles.size()));
    for (const Tile& tile : farm.tiles) {
      hash = behavior_mix(hash, int(tile.kind));
      hash = behavior_mix(hash, int(tile.crop));
      hash = behavior_mix(hash, int(tile.animal));
      hash = behavior_mix(hash, tile.planted_day);
      hash = behavior_mix(hash, tile.placed_day);
      hash = behavior_mix(hash, tile.yield_units);
      hash = behavior_mix(hash, tile.consecutive_unwatered);
      hash = behavior_mix(hash, tile.consecutive_unfed);
      hash = behavior_mix(hash, tile.fertilized_until_day);
      hash = behavior_mix(hash, tile.pending_care_bonus);
      hash = behavior_mix(hash, tile.max_lifespan_step);
      hash = behavior_mix(hash, tile.watered_today);
      hash = behavior_mix(hash, tile.fed_today);
      hash = behavior_mix(hash, tile.cared_today);
      hash = behavior_mix(hash, tile.fertilizer_available);
    }
  }
  for (const PrivateState& private_state : env.privates()) {
    for (const int value : private_state.shed)
      hash = behavior_mix(hash, value);
    for (const int value : private_state.seeds)
      hash = behavior_mix(hash, value);
    hash = behavior_mix(hash, int(private_state.inventories.size()));
    for (const auto& inventory : private_state.inventories)
      for (const int value : inventory) hash = behavior_mix(hash, value);
    hash = behavior_mix(hash, int(private_state.inventory_order.size()));
    for (const auto& order : private_state.inventory_order) {
      hash = behavior_mix(hash, int(order.size()));
      for (const int value : order) hash = behavior_mix(hash, value);
    }
  }
  for (const int value : env.market().inventory)
    hash = behavior_mix(hash, value);
  for (const int value : env.market().prices)
    hash = behavior_mix(hash, value);
  hash = behavior_mix(hash, int(env.shops().size()));
  for (const int value : env.shops()) hash = behavior_mix(hash, value);
  return hash;
}

int daily_hire_cost(int hands) {
  int a = 1, b = 1, total = 0;
  for (int i = 0; i < hands; ++i) {
    total += a;
    const int next = a + b;
    a = b;
    b = next;
  }
  return total;
}

Action move_toward(Position from, Position target) {
  if (from.x < target.x) return Action{Op::EAST};
  if (from.x > target.x) return Action{Op::WEST};
  if (from.y < target.y) return Action{Op::SOUTH};
  if (from.y > target.y) return Action{Op::NORTH};
  return Action{Op::PASS};
}

struct Counts {
  std::array<int, N_ANIMALS> field_animals{};
  std::array<int, N_ANIMALS> total_animals{};
  std::array<int, N_CROPS> field_crops{};
  std::array<int, N_CROPS> committed_crops{};
  int pastures = 0;
  int coops = 0;
  int empty_pastures = 0;
  int empty_coops = 0;
  int empty = 0;
  int weeds = 0;
  int ready_harvest = 0;
  int hard_water = 0;
  int hard_feed = 0;
};

Counts counts_for(const Simulator& env, int player) {
  Counts out;
  const auto& farm = env.farms()[player];
  const auto& priv = env.privates()[player];
  for (const auto& tile : farm.tiles) {
    if (tile.kind == TileKind::PLANT) {
      const int crop = int(tile.crop);
      if (crop >= 0 && crop < N_CROPS) out.field_crops[crop]++;
      out.ready_harvest += tile.yield_units > 0;
      out.hard_water += !tile.watered_today && tile.consecutive_unwatered >= 1;
    } else if (tile.kind == TileKind::ANIMAL) {
      const int animal = int(tile.animal) - 9;
      if (animal >= 0 && animal < N_ANIMALS) {
        out.field_animals[animal]++;
        // An occupied animal tile is still an occupied structure.  Counting
        // only empty structures made the planner build a second pasture/coop
        // for every placed animal and silently consumed half of the map.
        if (animal == 0) out.coops++;
        else out.pastures++;
      }
      out.ready_harvest += tile.yield_units > 0;
      out.hard_feed += !tile.fed_today && tile.consecutive_unfed >= 1;
    } else if (tile.kind == TileKind::PASTURE) {
      out.pastures++;
      out.empty_pastures++;
    } else if (tile.kind == TileKind::COOP) {
      out.coops++;
      out.empty_coops++;
    } else if (tile.kind == TileKind::EMPTY) {
      out.empty++;
    } else if (tile.kind == TileKind::WEED) {
      out.weeds++;
    }
  }
  out.total_animals = out.field_animals;
  for (int a = 0; a < N_ANIMALS; ++a) {
    out.total_animals[a] += priv.shed[9 + a];
    for (const auto& inv : priv.inventories) out.total_animals[a] += inv[9 + a];
  }
  for (int c = 0; c < N_CROPS; ++c)
    out.committed_crops[c] = out.field_crops[c] + priv.seeds[c];
  return out;
}

// Candidate8 O1.4 canonical short-preview contract. The fields report what
// the existing planner/executor actually does, rather than another algebraic
// estimate of requested target deltas. Every arm receives the same fixed
// synthetic future and the other seat always passes. No true future seed,
// opponent route, hidden inventory or terminal reward reaches these fields.
struct CandidatePreviewAccumulator {
  double minimum_money = std::numeric_limits<double>::infinity();
  std::array<int32_t, 18> unit_op_count{};  // PASS .. CARE
  std::array<int32_t, 6> market_op_count{};
  std::array<int32_t, 6> market_quantity{};
  int32_t cash_shortfall_count = 0;
  double cash_shortfall_total = 0.0;
  int32_t overflow_total = 0;
  int32_t sell_before_buy_steps = 0;
  int32_t buy_before_sell_steps = 0;
  int32_t mixed_market_steps = 0;
};

// O1.5 rival response scenarios.  They are fixed before any label split and
// depend only on the public checkpoint and official time horizon.  No route
// id, author, Replay calendar, true future action or hidden rival inventory is
// available to these policies.
AdaptiveGenome response_scenario_genome(
    const AdaptiveGenome& base, int scenario, int day) {
  AdaptiveGenome result = base;
  result.offline_expose_all_feasible_switches = false;
  result.offline_candidate8_enabled = false;
  result.runtime_market_preempt_fraction = 0.0;
  result.r8_execution_enabled = false;
  // Empty backbones are supplied by the caller, so the policy always rebuilds
  // its portfolio from the live state instead of replaying an operating prior.
  result.operating_prior_selection = 1.0;
  result.backbone_crop_flex = 1.0;
  result.backbone_animal_flex = 1.0;
  result.autonomous_project_state_machine = 1.0;
  result.autonomous_commitment_persistence = 0.50;
  result.crop_specific_terminal_horizon = 1.0;
  result.repeating_crop_economics = 1.0;
  result.future_shop_expectation_weight = 1.0;
  result.live_commitment_projection_weight = 1.0;

  if (scenario == 2) {  // expansion under optimistic capital conversion
    result.cash_reserve = 0.0;
    result.risk_multiplier = std::min(result.risk_multiplier, 0.85);
    result.capital_lockup_weight *= 0.25;
    result.max_hands = 12;
    result.max_quadrants = 4;
    result.max_total_animals = std::max(result.max_total_animals, 24);
    result.max_cows = std::max(result.max_cows, 16);
    result.max_sheep = std::max(result.max_sheep, 12);
    result.max_geese = std::max(result.max_geese, 8);
    result.proactive_land_investment = 1.0;
    result.land_capacity_trigger_fraction = 0.65;
  } else if (scenario == 3) {  // sell/finance now from reconstructed stock
    result.stop_new_animals_day = day;
    result.stop_new_crops_day = day;
    result.liquidation_day = day;
    result.runtime_market_preempt_fraction = 1.0;
    result.runtime_market_preempt_min_ready = 1;
    result.runtime_market_preempt_min_price_ratio = 0.0;
    result.runtime_market_preempt_max_quantity = 100;
    result.town_demand_sell_timing = 0.0;
  } else if (scenario == 4) {  // protect capital and wait for public demand
    result.cash_reserve = std::max(result.cash_reserve, 1000.0);
    result.risk_multiplier = std::max(result.risk_multiplier, 1.50);
    result.stop_new_animals_day = std::min(result.stop_new_animals_day, day);
    result.town_demand_sell_timing = 1.0;
    result.liquidation_day = TOTAL_DAYS - 1;
  }
  return result;
}

PrivateState public_opponent_private_belief(
    const Simulator& env, int opponent,
    const AdaptivePlannerState& observer_state) {
  PrivateState belief;
  const size_t units = env.farms()[opponent].hands.size() + 1;
  belief.inventories.resize(units);
  belief.inventory_order.resize(units);
  int remaining_capacity = env.config().shed_capacity;
  // A public fall in on-map ready yield is the only inventory evidence used.
  // Treat it as recently collected stock, clipped by official shed capacity.
  // This is an intentionally simple belief scenario, not reconstruction of
  // the true private shed.
  for (int project = 0; project < N_CROPS + N_ANIMALS; ++project) {
    const int observed_drop = std::max(
        0, -int(observer_state.opponent_recent_project_yield_delta[project]));
    const int quantity = std::min(remaining_capacity, observed_drop);
    if (quantity <= 0) continue;
    const int item = project < N_CROPS
        ? project : (N_CROPS + (project - N_CROPS));
    belief.shed[item] = quantity;
    remaining_capacity -= quantity;
  }
  return belief;
}

int preview_market_index(Op op) {
  switch (op) {
    case Op::HIRE: return 0;
    case Op::BUY_LAND: return 1;
    case Op::BUY_SEED: return 2;
    case Op::BUY_PRODUCT: return 3;
    case Op::BUY_ANIMAL: return 4;
    case Op::SELL: return 5;
    default: return -1;
  }
}

void preview_accumulate_action(const PlayerAction& action,
                               CandidatePreviewAccumulator& accumulator) {
  for (const auto& unit : action.units) {
    const int op = int(unit.op);
    if (op >= int(Op::PASS) && op <= int(Op::CARE))
      accumulator.unit_op_count[op]++;
  }
  int first_sell = -1;
  int first_buy = -1;
  for (int order = 0; order < int(action.market.size()); ++order) {
    const auto& market = action.market[order];
    const int index = preview_market_index(market.op);
    if (index < 0) continue;
    accumulator.market_op_count[index]++;
    accumulator.market_quantity[index] += std::max(0, market.quantity);
    if (market.op == Op::SELL && first_sell < 0) first_sell = order;
    if (market.op != Op::SELL && first_buy < 0) first_buy = order;
  }
  if (first_sell >= 0 && first_buy >= 0) {
    accumulator.mixed_market_steps++;
    if (first_sell < first_buy) accumulator.sell_before_buy_steps++;
    else accumulator.buy_before_sell_steps++;
  }
}

void preview_accumulate_effects(const Simulator& env, int player,
                                CandidatePreviewAccumulator& accumulator) {
  accumulator.minimum_money = std::min(
      accumulator.minimum_money, env.farms()[player].money);
  for (const double shortfall : env.last_market_cash_shortfalls()[player]) {
    if (shortfall <= 0.0) continue;
    accumulator.cash_shortfall_count++;
    accumulator.cash_shortfall_total += shortfall;
  }
  accumulator.overflow_total += env.last_end_of_day_overflow()[player];
}

void capture_candidate_preview(
    const Simulator& env, int player,
    const CandidatePreviewAccumulator& accumulator,
    std::array<int32_t,
        AdaptiveCandidate8CounterfactualResult::CONSEQUENCE_FEATURE_DIM>& out,
    int& feature) {
  const auto& farm = env.farms()[player];
  const auto& priv = env.privates()[player];
  const auto counts = counts_for(env, player);
  int32_t shed_quantity = 0;
  int32_t shed_value_x100 = 0;
  int32_t carry_quantity = 0;
  int32_t carry_value_x100 = 0;
  int32_t seed_quantity = 0;
  int32_t ready_yield = 0;
  std::array<int32_t, N_CROPS + N_ANIMALS> project_count{};
  std::array<int32_t, N_CROPS + N_ANIMALS> project_yield{};
  for (int item = 0; item < N_ITEMS; ++item) {
    shed_quantity += priv.shed[item];
    if (item < N_PRODUCTS)
      shed_value_x100 += 100 * priv.shed[item] * env.market().prices[item];
  }
  for (const auto& inventory : priv.inventories) {
    for (int item = 0; item < N_ITEMS; ++item) {
      carry_quantity += inventory[item];
      if (item < N_PRODUCTS)
        carry_value_x100 += 100 * inventory[item] * env.market().prices[item];
    }
  }
  for (const int quantity : priv.seeds) seed_quantity += quantity;
  for (const auto& tile : farm.tiles) {
    ready_yield += std::max(0, int(tile.yield_units));
    if (tile.kind == TileKind::PLANT) {
      const int crop = int(tile.crop);
      if (crop >= 0 && crop < N_CROPS) {
        project_count[crop]++;
        project_yield[crop] += std::max(0, int(tile.yield_units));
      }
    } else if (tile.kind == TileKind::ANIMAL) {
      const int animal = int(tile.animal) - 9;
      if (animal >= 0 && animal < N_ANIMALS) {
        project_count[N_CROPS + animal]++;
        project_yield[N_CROPS + animal] +=
            std::max(0, int(tile.yield_units));
      }
    }
  }
  int crop_total = 0;
  int animal_total = 0;
  for (const int value : counts.field_crops) crop_total += value;
  for (const int value : counts.total_animals) animal_total += value;

  out[feature++] = int32_t(std::nearbyint(100.0 * farm.money));
  out[feature++] = int32_t(std::nearbyint(
      100.0 * (std::isfinite(accumulator.minimum_money)
          ? accumulator.minimum_money : farm.money)));
  out[feature++] = int32_t(farm.hands.size());
  out[feature++] = popcount(farm.unlocked_mask);
  out[feature++] = shed_quantity;
  out[feature++] = shed_value_x100;
  out[feature++] = carry_quantity;
  out[feature++] = carry_value_x100;
  out[feature++] = seed_quantity;
  out[feature++] = crop_total;
  out[feature++] = animal_total;
  out[feature++] = counts.weeds;
  out[feature++] = ready_yield;
  out[feature++] = counts.hard_water;
  out[feature++] = counts.hard_feed;
  for (const int value : project_count) out[feature++] = value;
  for (const int value : project_yield) out[feature++] = value;
  for (const int value : env.market().inventory) out[feature++] = value;
  for (const int value : accumulator.unit_op_count) out[feature++] = value;
  for (const int value : accumulator.market_op_count) out[feature++] = value;
  for (const int value : accumulator.market_quantity) out[feature++] = value;
  out[feature++] = accumulator.cash_shortfall_count;
  out[feature++] = int32_t(std::nearbyint(
      100.0 * accumulator.cash_shortfall_total));
  out[feature++] = accumulator.overflow_total;
  out[feature++] = accumulator.sell_before_buy_steps;
  out[feature++] = accumulator.buy_before_sell_steps;
  out[feature++] = accumulator.mixed_market_steps;
}

std::array<double, N_PRODUCTS> shop_demand(const Simulator& env) {
  std::array<double, N_PRODUCTS> demand{};
  // The town center consumes one unit of each sellable product once per day
  // (every 24 steps).  Only unlocked shops consume every four steps, i.e. six
  // times per day.  Treating the town-center unit as six units systematically
  // inflated every project's future price and made animal/crop caps, rather
  // than diminishing economics, choose the portfolio size.
  for (int i = 0; i < 8; ++i) demand[i] = 1.0;
  for (int shop : env.shops()) {
    if (shop < 0 || shop >= 8) continue;
    for (int item = 0; item < N_PRODUCTS; ++item)
      demand[item] += 6.0 * SHOP_DEMAND[shop][item];
  }
  return demand;
}

std::array<double, N_PRODUCTS> visible_daily_supply(const Simulator& env, int player) {
  std::array<double, N_PRODUCTS> supply{};
  const auto& farm = env.farms()[player];
  for (const auto& tile : farm.tiles) {
    if (tile.kind == TileKind::ANIMAL) {
      const int animal = int(tile.animal) - 9;
      if (animal >= 0 && animal < N_ANIMALS)
        supply[ANIMAL_PRODUCT[animal]] += 1.0;
    } else if (tile.kind == TileKind::PLANT) {
      const int crop = int(tile.crop);
      if (crop >= 0 && crop < N_CROPS) {
        if (CROP_ONGOING[crop]) supply[crop] += 1.0;
        else supply[crop] += double(CROP_MAX_YIELD[crop]) / std::max(1, CROP_MAX_DAY[crop]);
      }
    }
  }
  return supply;
}

// Convert only public opponent production capacity into a bounded first-seller
// pressure for each crop.  The signal is deliberately prospective: waiting
// until product is already harvestable is too late for a crop whose production
// lead time is several days.  Capacity contributes only when it can mature
// inside the same window as a fresh commitment of that product, and its weight
// is the actual quote loss produced by the official market curve.
std::array<double, N_CROPS> public_crop_market_race_pressure(
    const Simulator& env, int player) {
  std::array<double, N_CROPS> pressure{};
  const auto& opponent = env.farms()[1 - player];
  const int day = env.day();
  for (int crop = 0; crop < N_CROPS; ++crop) {
    const int horizon = CROP_FIRST[crop] + 2;
    int earliest = horizon + 1;
    double projected_units = 0.0;
    for (const auto& tile : opponent.tiles) {
      if (tile.kind != TileKind::PLANT || int(tile.crop) != crop) continue;
      const int wait = std::max(0, int(tile.planted_day) +
                                      CROP_FIRST[crop] - day);
      earliest = std::min(earliest, wait);
      double units = std::max(0, int(tile.yield_units));
      if (wait <= horizon) {
        if (CROP_ONGOING[crop]) {
          const int interval = std::max(1, CROP_INTERVAL[crop]);
          const int cycles = 1 + std::max(0, horizon - wait) / interval;
          units = std::max<double>(units,
              std::min(CROP_MAX_YIELD[crop], cycles));
        } else {
          units = std::max<double>(units, CROP_MAX_YIELD[crop]);
        }
      }
      projected_units += units;
    }
    if (projected_units <= 0.0 || earliest > horizon) continue;

    const int item = crop;
    const int inventory = env.market().inventory[item];
    const int current_quote = std::max(1, predicted_price(item, inventory));
    const int adverse_quote = predicted_price(
        item, inventory + int(std::ceil(projected_units)));
    const double quote_impact = std::max(
        0.0, (current_quote - adverse_quote) / double(current_quote));
    // If the opponent is already far ahead of a new commitment, planting more
    // is not automatically a race.  Equal or later maturity retains full
    // pressure; an earlier opponent is discounted smoothly rather than by a
    // brittle day threshold.
    const int opponent_lead = std::max(0, CROP_FIRST[crop] - earliest);
    const double timing = 1.0 / (1.0 + 0.25 * opponent_lead);
    pressure[crop] = std::clamp(quote_impact * timing / 0.12, 0.0, 1.0);
  }
  return pressure;
}

int cap_for_crop(const AdaptiveGenome& g, int crop) {
  const int values[N_CROPS] = {g.max_wheat, g.max_carrot, g.max_tomato,
                               g.max_strawberry, g.max_melon};
  return values[crop];
}

bool unlocked(const Farm& farm, Position p) {
  const int q = (p.y < 5 ? 0 : 2) + (p.x < 5 ? 0 : 1);
  return (farm.unlocked_mask & (1 << q)) != 0;
}

}  // namespace

const std::array<const char*, AdaptiveGenome::DIM>& AdaptiveGenome::names() {
  static const std::array<const char*, DIM> values{{
      "cash_reserve", "action_cost", "move_cost", "risk_multiplier",
      "market_impact_weight", "opponent_supply_weight", "demand_drift_weight",
      "sell_drop_limit", "price_replan_fraction", "task_stickiness",
      "deadline_weight", "fertilizer_value_fraction", "max_hands",
      "max_quadrants", "max_total_animals", "max_cows", "max_sheep",
      "max_geese", "max_wheat", "max_carrot", "max_tomato",
      "max_strawberry", "max_melon", "min_wheat_buffer", "liquidation_day",
      "stop_new_animals_day", "stop_new_crops_day", "replan_interval_steps",
      "routine_priority_scale", "task_value_scale", "plant_priority",
      "drop_value_threshold", "preempt_quantity",
      "wheat_relay_capacity_fraction", "wheat_relay_opponent_risk",
      "industry_affinity_scale", "route_density_scale",
      "flow_pace_priority", "prerequisite_chain_strength",
      "local_candidate_order", "region_ownership_scale",
      "animal_flow_control", "workload_region_mode",
      "routine_cell_exclusivity", "hard_travel_slack_scale",
      "future_structure_reservation_fraction",
      "yarn_animal_flex",
      "yarn_wool_price_threshold", "yarn_opponent_sheep_gate",
      "pet_crop_flex",
      "pet_carrot_price_threshold",
      "weed_capacity_priority",
      "animal_flow_pressure_gate",
      "animal_flow_executable_gate",
      "terminal_animal_economics",
      "crop_lane_layout_mode",
      "region_assignment_mode",
      "hard_latest_start_reservation",
      "enroute_task_reservation",
      "global_routine_matching",
      "backbone_crop_flex",
      "region_owner_first",
      "quadrant_affinity_scale",
      "opening_cash_reserve_fraction",
      "region_anchor_distance_scale",
      "town_demand_sell_timing",
      "repeating_crop_economics",
      "proactive_land_investment",
      "capital_lockup_weight",
      "backbone_animal_flex",
      "animal_branch_economic_selector",
      "animal_branch_observation_days",
      "seed_transaction_value_order",
      "animal_counter_crowding_gate",
      "due_flow_cash_release",
      "market_race_acceleration",
      "operating_prior_selection",
      "operating_prior_switch_margin",
      "autonomous_commitment_persistence",
      "land_capacity_trigger_fraction",
      "crop_specific_terminal_horizon",
      "autonomous_project_state_machine",
      "future_shop_expectation_weight",
      "portfolio_supply_impact_weight",
      "live_commitment_projection_weight",
      "portfolio_bundle_switch_margin",
      "portfolio_switch_cooldown_days",
      "portfolio_switch_candidate_rank",
      "portfolio_local_edit_hold_days"}};
  return values;
}

AdaptiveGenome AdaptiveGenome::from_vector(const std::vector<double>& v) {
  AdaptiveGenome g;
  if (v.empty()) return g;
  if (int(v.size()) != DIM) throw std::invalid_argument(
      "adaptive genome has incorrect dimension");
  int i = 0;
  g.cash_reserve = std::max(0.0, v[i++]);
  g.action_cost = std::max(0.0, v[i++]);
  g.move_cost = std::max(0.0, v[i++]);
  g.risk_multiplier = std::max(0.5, v[i++]);
  g.market_impact_weight = std::max(0.0, v[i++]);
  g.opponent_supply_weight = std::max(0.0, v[i++]);
  g.demand_drift_weight = std::clamp(v[i++], 0.0, 2.0);
  g.sell_drop_limit = std::clamp(v[i++], 0.0, 1.0);
  g.price_replan_fraction = std::clamp(v[i++], 0.01, 2.0);
  g.task_stickiness = std::max(0.0, v[i++]);
  g.deadline_weight = std::max(0.0, v[i++]);
  g.fertilizer_value_fraction = std::clamp(v[i++], 0.0, 2.0);
  auto integer = [&](int lo, int hi) { return std::clamp(int(std::nearbyint(v[i++])), lo, hi); };
  g.max_hands = integer(0, 24);
  g.max_quadrants = integer(1, 4);
  g.max_total_animals = integer(0, 50);
  g.max_cows = integer(0, 50);
  g.max_sheep = integer(0, 50);
  g.max_geese = integer(0, 50);
  g.max_wheat = integer(0, 100);
  g.max_carrot = integer(0, 100);
  g.max_tomato = integer(0, 100);
  g.max_strawberry = integer(0, 100);
  g.max_melon = integer(0, 100);
  g.min_wheat_buffer = integer(0, 100);
  g.liquidation_day = integer(20, 30);
  g.stop_new_animals_day = integer(0, 29);
  g.stop_new_crops_day = integer(0, 29);
  g.replan_interval_steps = integer(1, 720);
  g.routine_priority_scale = std::clamp(v[i++], 0.0, 20.0);
  g.task_value_scale = std::clamp(v[i++], 0.0, 5.0);
  g.plant_priority = std::clamp(v[i++], 400.0, 1000.0);
  // This threshold is intentionally allowed to exceed every realistic cargo
  // value so experiments can disable value-only mid-day returns.  The former
  // 10,000 cap silently converted a configured 100,000 "off" value into an
  // active trigger and sent workers on unnecessary shed round trips.
  g.drop_value_threshold = std::clamp(v[i++], 0.0, 1.0e9);
  g.preempt_quantity = integer(1, 100);
  g.wheat_relay_capacity_fraction = std::clamp(v[i++], 0.0, 0.9);
  g.wheat_relay_opponent_risk = std::clamp(v[i++], 0.0, 4.0);
  g.industry_affinity_scale = std::clamp(v[i++], 0.0, 8.0);
  g.route_density_scale = std::clamp(v[i++], 0.0, 20.0);
  g.flow_pace_priority = std::clamp(v[i++], 0.0, 300.0);
  g.prerequisite_chain_strength = std::clamp(v[i++], 0.0, 1.0);
  g.local_candidate_order = std::clamp(v[i++], 0.0, 1.0);
  g.region_ownership_scale = std::clamp(v[i++], 0.0, 10.0);
  g.animal_flow_control = std::clamp(v[i++], 0.0, 1.0);
  g.workload_region_mode = std::clamp(v[i++], 0.0, 1.0);
  g.routine_cell_exclusivity = std::clamp(v[i++], 0.0, 1.0);
  g.hard_travel_slack_scale = std::clamp(v[i++], 0.0, 10.0);
  g.future_structure_reservation_fraction = std::clamp(v[i++], 0.0, 1.0);
  g.yarn_animal_flex = std::clamp(v[i++], 0.0, 1.0);
  g.yarn_wool_price_threshold = std::clamp(v[i++], 1.0, 1000.0);
  g.yarn_opponent_sheep_gate = std::clamp(v[i++], 0.0, 100.0);
  g.pet_crop_flex = std::clamp(v[i++], 0.0, 1.0);
  g.pet_carrot_price_threshold = std::clamp(v[i++], 1.0, 1000.0);
  g.weed_capacity_priority = std::clamp(v[i++], 0.0, 1000.0);
  g.animal_flow_pressure_gate = std::clamp(v[i++], 0.0, 4.0);
  g.animal_flow_executable_gate = std::clamp(v[i++], 0.0, 4.0);
  g.terminal_animal_economics = std::clamp(v[i++], 0.0, 1.0);
  g.crop_lane_layout_mode = std::clamp(v[i++], 0.0, 2.0);
  g.region_assignment_mode = std::clamp(v[i++], 0.0, 1.0);
  g.hard_latest_start_reservation = std::clamp(v[i++], 0.0, 1.0);
  g.enroute_task_reservation = std::clamp(v[i++], 0.0, 1.0);
  g.global_routine_matching = std::clamp(v[i++], 0.0, 1.0);
  g.backbone_crop_flex = std::clamp(v[i++], 0.0, 1.0);
  g.region_owner_first = std::clamp(v[i++], 0.0, 1.0);
  g.quadrant_affinity_scale = std::clamp(v[i++], 0.0, 8.0);
  g.opening_cash_reserve_fraction = std::clamp(v[i++], 0.0, 1.0);
  g.region_anchor_distance_scale = std::clamp(v[i++], 0.0, 10.0);
  g.town_demand_sell_timing = std::clamp(v[i++], 0.0, 1.0);
  g.repeating_crop_economics = std::clamp(v[i++], 0.0, 1.0);
  g.proactive_land_investment = std::clamp(v[i++], 0.0, 1.0);
  g.capital_lockup_weight = std::clamp(v[i++], 0.0, 3.0);
  g.backbone_animal_flex = std::clamp(v[i++], 0.0, 1.0);
  g.animal_branch_economic_selector = std::clamp(v[i++], 0.0, 4.0);
  g.animal_branch_observation_days = integer(0, 6);
  g.seed_transaction_value_order = std::clamp(v[i++], 0.0, 1.0);
  g.animal_counter_crowding_gate = integer(0, 100);
  g.due_flow_cash_release = std::clamp(v[i++], 0.0, 1.0);
  g.market_race_acceleration = std::clamp(v[i++], 0.0, 1.0);
  g.operating_prior_selection = std::clamp(v[i++], 0.0, 1.0);
  g.operating_prior_switch_margin = std::clamp(v[i++], 0.0, 1.0);
  g.autonomous_commitment_persistence = std::clamp(v[i++], 0.0, 1.0);
  g.land_capacity_trigger_fraction = std::clamp(v[i++], 0.0, 1.0);
  g.crop_specific_terminal_horizon = std::clamp(v[i++], 0.0, 1.0);
  g.autonomous_project_state_machine = std::clamp(v[i++], 0.0, 1.0);
  g.future_shop_expectation_weight = std::clamp(v[i++], 0.0, 1.0);
  g.portfolio_supply_impact_weight = std::max(0.0, v[i++]);
  g.live_commitment_projection_weight = std::clamp(v[i++], 0.0, 1.0);
  g.portfolio_bundle_switch_margin = std::clamp(v[i++], 0.0, 0.50);
  g.portfolio_switch_cooldown_days = integer(0, 30);
  g.portfolio_switch_candidate_rank = integer(0, 15);
  g.portfolio_local_edit_hold_days = integer(0, 14);
  return g;
}

void AdaptivePlannerState::reset() {
  *this = AdaptivePlannerState{};
  candidate8_persistent_target.fill(int16_t(-1));
}

const AdaptiveBackbone& NativeAdaptivePlanner::active_backbone(
    const AdaptivePlannerState& state) const {
  static const AdaptiveBackbone disabled{};
  if (backbones_.empty()) return disabled;
  if (genome_.operating_prior_selection > 0.0 &&
      state.operating_prior_index >= -1) {
    if (state.operating_prior_index < 0) return disabled;
    const int index = int(state.operating_prior_index);
    if (index >= 0 && index < int(backbones_.size()))
      return backbones_[index];
    return disabled;
  }
  if (backbones_.size() != 4) return backbones_.front();
  const bool sheep = state.animal_branch == 1;
  const bool carrot = state.crop_suffix == 1;
  // Before either public decision window resolves, use the corresponding
  // wheat/cow prior.  This is an operating prior, not a frozen action tape;
  // build_plan still revalues projects from the current market every replan.
  const int index = sheep ? (carrot ? 2 : 3) : (carrot ? 0 : 1);
  return backbones_[index];
}

bool NativeAdaptivePlanner::should_replan(const Simulator& env, int,
                                          const AdaptivePlannerState& state) const {
  if (state.last_plan_step < 0) return true;
  if (env.day() != state.plan.generated_day) return true;
  if (int(env.shops().size()) != state.last_shop_count) return true;
  if (env.step_count() - state.last_plan_step >= genome_.replan_interval_steps) return true;
  for (int item = 0; item < N_PRODUCTS; ++item) {
    const int old_price = state.last_prices[item];
    const int new_price = env.market().prices[item];
    if (old_price > 0 && std::abs(new_price - old_price) / double(old_price) > genome_.price_replan_fraction)
      return true;
  }
  return false;
}

AdaptivePlan NativeAdaptivePlanner::build_plan(const Simulator& env, int player,
                                                const AdaptivePlannerState& state) const {
  const auto& backbone = active_backbone(state);
  const int day = env.day();
  const auto own = counts_for(env, player);
  const auto opp_supply = visible_daily_supply(env, 1 - player);
  const auto own_supply = visible_daily_supply(env, player);
  const auto demand = shop_demand(env);
  const auto& market = env.market();
  const auto& farm = env.farms()[player];
  const auto& priv = env.privates()[player];

  // Cache public committed production by player, product and horizon.  The
  // same few lags are queried hundreds of times by the marginal allocator;
  // lazy caching keeps the semantic projection out of the inner tile scan.
  const double live_projection = genome_.live_commitment_projection_weight;
  std::array<std::array<std::array<double, N_PRODUCTS>, TOTAL_DAYS + 1>, 2>
      committed_supply_cache;
  std::array<std::array<std::array<uint8_t, N_PRODUCTS>, TOTAL_DAYS + 1>, 2>
      committed_supply_cached;
  bool committed_supply_cache_ready = false;
  auto ensure_committed_supply_cache = [&]() {
    if (committed_supply_cache_ready) return;
    for (auto& owner : committed_supply_cached)
      for (auto& horizon : owner) horizon.fill(uint8_t(0));
    committed_supply_cache_ready = true;
  };
  // This projection is an optional experimental feature.  In the accepted
  // rollback configuration its weight is zero, so do not clear roughly 5 KB
  // of cache state on every replan or enter the public-tile scan at all.
  if (live_projection > 0.0) {
    ensure_committed_supply_cache();
  }
  auto committed_supply_until = [&](int owner, int item, int raw_lag) {
    ensure_committed_supply_cache();
    const int lag = std::clamp(raw_lag, 0, TOTAL_DAYS);
    if (item < 0 || item >= N_PRODUCTS) return 0.0;
    if (committed_supply_cached[owner][lag][item])
      return committed_supply_cache[owner][lag][item];
    double units = 0.0;
    const int horizon_day = std::min(TOTAL_DAYS - 1, day + lag);
    for (const auto& tile : env.farms()[owner].tiles) {
      if (tile.kind == TileKind::PLANT && int(tile.crop) == item &&
          item < N_CROPS) {
        const int crop = item;
        const int age_now = day - int(tile.planted_day);
        const int age_horizon = horizon_day - int(tile.planted_day);
        if (age_horizon < CROP_FIRST[crop]) continue;
        double projected = std::max(0, int(tile.yield_units));
        if (CROP_ONGOING[crop]) {
          const int interval = std::max(1, CROP_INTERVAL[crop]);
          const int first_future_age = std::max(
              age_now + 1, CROP_FIRST[crop]);
          for (int age = first_future_age; age <= age_horizon; ++age) {
            const int delta = age - CROP_FIRST[crop];
            if (delta >= 0 && delta % interval == 0 &&
                delta / interval < CROP_MAX_YIELD[crop])
              projected += 1.0;
          }
        } else {
          // A maintained one-shot crop gains one unit on each watering day in
          // the yield window (fertilizer upside is intentionally omitted).
          const int water_start = (CROP_MAX_DAY[crop] + 1) / 2;
          const int from_age = std::max(age_now + 1, water_start);
          const int to_age = std::min(age_horizon, CROP_MAX_DAY[crop]);
          if (to_age >= from_age) projected += to_age - from_age + 1;
          projected = std::min<double>(CROP_MAX_YIELD[crop], projected);
        }
        units += projected;
      } else if (tile.kind == TileKind::ANIMAL) {
        const int animal = int(tile.animal) - 9;
        if (animal < 0 || animal >= N_ANIMALS) continue;
        if (item == 8) {
          units += lag;
          continue;
        }
        if (ANIMAL_PRODUCT[animal] != item) continue;
        double projected = std::max(0, int(tile.yield_units));
        const int first = ANIMAL_FIRST[animal];
        const int interval = std::max(1, ANIMAL_INTERVAL[animal]);
        const int age_now = day - int(tile.placed_day);
        const int age_horizon = horizon_day - int(tile.placed_day);
        const int first_future_age = std::max(age_now + 1, first);
        for (int age = first_future_age; age <= age_horizon; ++age) {
          const int delta = age - first;
          if (delta >= 0 && delta % interval == 0)
            projected += 1.0 + interval;
        }
        units += projected;
      }
    }
    committed_supply_cached[owner][lag][item] = 1;
    committed_supply_cache[owner][lag][item] = units;
    return units;
  };

  AdaptivePlan plan;
  plan.generated_day = int16_t(day);
  plan.quadrant_target = int16_t(popcount(farm.unlocked_mask));
  plan.deferred_quadrant_target = plan.quadrant_target;
  plan.wheat_buffer = int16_t(genome_.min_wheat_buffer);
  for (int a = 0; a < N_ANIMALS; ++a) {
    plan.animal_targets[a] = int16_t(own.total_animals[a]);
    // An animal already bought is an irreversible project commitment even
    // while it is still in the shed or a worker inventory.  Initialising the
    // placement/service target from field animals only made a newly purchased
    // animal disappear from the next day's plan before it reached a structure.
    // It then remained stranded until the terminal state.  Existing field
    // animals and every unplaced owned animal therefore share the same normal
    // operating floor.  Deliberate terminal release remains controlled by the
    // separate terminal-animal policy in build_tasks().
    plan.animal_service_targets[a] = int16_t(own.total_animals[a]);
    plan.deferred_animal_targets[a] = int16_t(own.total_animals[a]);
    plan.animal_project_modes[a] = AdaptivePlan::INACTIVE;
  }
  // Animals are irreversible commitments, whereas harvested crops may be
  // replaced by a different crop or by an animal structure.  Initialising the
  // desired crop portfolio from the current field made every crop permanent:
  // the greedy allocator could only append projects and therefore got stuck
  // at the first full-map mix it happened to build.  Rebuild the *future*
  // crop portfolio on every plan instead.  Existing plants are still fully
  // maintained and harvested by build_tasks(); a lower target merely means
  // "do not replant this slot after harvest".
  for (int c = 0; c < N_CROPS; ++c) {
    plan.crop_targets[c] = 0;
    plan.deferred_crop_targets[c] = 0;
    plan.crop_project_modes[c] = AdaptivePlan::INACTIVE;
  }
  if (!backbone.enabled && state.plan.generated_day >= 0 &&
      genome_.autonomous_commitment_persistence > 0.0 &&
      genome_.autonomous_project_state_machine <= 0.0) {
    const double keep = genome_.autonomous_commitment_persistence;
    if (day <= genome_.stop_new_animals_day) {
      for (int animal = 0; animal < N_ANIMALS; ++animal) {
        const int owned = own.total_animals[animal];
        const int outstanding = std::max(
            0, int(state.plan.animal_targets[animal]) - owned);
        const int retained = int(std::nearbyint(keep * outstanding));
        plan.animal_targets[animal] = int16_t(owned + retained);
        const int prior_service = std::max(
            0, int(state.plan.animal_service_targets[animal]));
        plan.animal_service_targets[animal] = int16_t(std::max(
            own.total_animals[animal],
            std::min<int>(plan.animal_targets[animal],
                          int(std::nearbyint(keep * prior_service)))));
      }
    }
    {
      for (int crop = 0; crop < N_CROPS; ++crop) {
        if (!crop_project_open(genome_, crop, day)) continue;
        // Bought seeds are sunk, directly usable capacity.  Existing plants
        // remain protected by build_tasks(), but do not by themselves force a
        // future replant when another project has become more valuable.
        const int retained = int(std::nearbyint(
            keep * std::max(0, int(state.plan.crop_targets[crop]))));
        plan.crop_targets[crop] = int16_t(std::max(
            int(priv.seeds[crop]), retained));
      }
    }
    if (day <= std::max(genome_.stop_new_animals_day,
                        latest_generic_crop_start(genome_))) {
      const int unlocked = popcount(farm.unlocked_mask);
      const int outstanding = std::max(
          0, int(state.plan.quadrant_target) - unlocked);
      plan.quadrant_target = int16_t(std::max(
          unlocked, unlocked + int(std::nearbyint(keep * outstanding))));
    }
  }
  if (backbone.enabled) {
    const int reference_day = std::clamp(day, 0, TOTAL_DAYS - 1);
    plan.quadrant_target = int16_t(std::max<int>(
        plan.quadrant_target, backbone.quadrant_target[reference_day]));
    plan.wheat_buffer = int16_t(std::max<int>(
        plan.wheat_buffer, backbone.wheat_buffer[reference_day]));
    for (int animal = 0; animal < N_ANIMALS; ++animal) {
      int purchase_target = backbone.animal_targets[reference_day][animal];
      int service_target = backbone.animal_service_targets[reference_day][animal];
      if (backbones_.size() == 4 && state.animal_branch < 0 &&
          genome_.animal_branch_economic_selector > 0.0) {
        // Until the observation window closes, commit only the capacity shared
        // by both animal portfolios.  Cows and sheep use the same PASTURE
        // structure, so generic space may still be prepared without making
        // the irreversible animal purchase before public evidence arrives.
        const int cow_index = state.crop_suffix == 1 ? 0 : 1;
        const int sheep_index = state.crop_suffix == 1 ? 2 : 3;
        purchase_target = std::min<int>(
            backbones_[cow_index].animal_targets[reference_day][animal],
            backbones_[sheep_index].animal_targets[reference_day][animal]);
        service_target = std::min<int>(
            backbones_[cow_index].animal_service_targets[reference_day][animal],
            backbones_[sheep_index].animal_service_targets[reference_day][animal]);
      }
      // A semantic calendar is an operating prior, not an irreversible future
      // purchase tape.  Release a searchable fraction of outstanding animal
      // commitments before live economic allocation, while preserving every
      // animal already bought or placed.
      const int owned = own.total_animals[animal];
      purchase_target = owned + int(std::nearbyint(
          std::max(0, purchase_target - owned) *
          (1.0 - genome_.backbone_animal_flex)));
      const int serviced = own.field_animals[animal];
      service_target = serviced + int(std::nearbyint(
          std::max(0, service_target - serviced) *
          (1.0 - genome_.backbone_animal_flex)));
      // The YARN branch releases only not-yet-bought goose/cow commitments.
      // Existing animals remain irreversible and the demonstrated sheep floor
      // remains intact.  The general project evaluator must earn any added
      // sheep/cow capacity from current public economics.
      if (!force_backbone_exact_ && state.animal_branch == 1 && animal != 2) {
        purchase_target = owned + int(std::nearbyint(
            std::max(0, purchase_target - owned) *
            (1.0 - genome_.yarn_animal_flex)));
        service_target = serviced + int(std::nearbyint(
            std::max(0, service_target - serviced) *
            (1.0 - genome_.yarn_animal_flex)));
      }
      plan.animal_targets[animal] = int16_t(std::max<int>(
          plan.animal_targets[animal], purchase_target));
      plan.animal_service_targets[animal] = int16_t(std::max<int>(
          plan.animal_service_targets[animal], service_target));
    }
    for (int crop = 0; crop < N_CROPS; ++crop) {
      int target = backbone.crop_targets[reference_day][crop];
      // Crop commitments that have not yet reached the field are reversible.
      // Release a searchable fraction before the live project allocator runs;
      // existing plants are still maintained by build_tasks() regardless of
      // the lower desired next-cycle target.
      const int planted = own.field_crops[crop];
      target = planted + int(std::nearbyint(
          std::max(0, target - planted) *
          (1.0 - genome_.backbone_crop_flex)));
      // WHEAT remains protected as feed/working inventory and CARROT is the
      // candidate suffix itself.  Only future replant commitments of the
      // replaceable competing crops are released; existing fields are still
      // maintained and harvested by build_tasks().
      if (!force_backbone_exact_ && state.crop_suffix == 1 && crop != 0 && crop != 1)
        target = int(std::nearbyint(target * (1.0 - genome_.pet_crop_flex)));
      plan.crop_targets[crop] = int16_t(std::max(0, target));
    }
  }

  // Preserve the *direction* of the two targets touched by an accepted local
  // SWITCH.  A source edit is a temporary ceiling (the allocator may shrink it
  // further when economics deteriorate); a destination edit is a temporary
  // floor (the allocator may scale it further when economics improve).
  //
  // The previous exact-target lock turned a local edit into a brittle frozen
  // mini-route.  It also prevented useful scaling and made unrelated capacity
  // allocation cascade around two exact numbers.  Directional bounds retain
  // the causal intervention long enough to execute while preserving daily
  // replanning.  The bounds are project-generic and contain no route, opponent
  // identity, calendar constant or coordinate.
  constexpr int PROJECTS = N_CROPS + N_ANIMALS;
  std::array<int16_t, PROJECTS> local_project_floor{};
  std::array<int16_t, PROJECTS> local_project_cap{};
  local_project_floor.fill(int16_t(-1));
  local_project_cap.fill(int16_t(-1));
  const bool bundle_lock_active = !backbone.enabled &&
      state.bundle_lock_until_day >= day;
  if (bundle_lock_active) {
    if (state.bundle_locked_source >= 0 &&
      state.bundle_locked_source < PROJECTS)
      local_project_cap[state.bundle_locked_source] =
          state.bundle_locked_source_target;
    if (state.bundle_locked_destination >= 0 &&
        state.bundle_locked_destination < PROJECTS)
      local_project_floor[state.bundle_locked_destination] =
          state.bundle_locked_destination_target;
    for (int project = 0; project < PROJECTS; ++project) {
      if (local_project_floor[project] < 0 &&
          local_project_cap[project] < 0) continue;
      if (project < N_CROPS) {
        const int crop = project;
        const int irreversible = own.committed_crops[crop];
        const int normal_cap = crop_project_open(genome_, crop, day)
            ? cap_for_crop(genome_, crop) : irreversible;
        const int effective_cap = local_project_cap[project] >= 0
            ? std::min(normal_cap, std::max(
                  irreversible, int(local_project_cap[project])))
            : normal_cap;
        const int effective_floor = local_project_floor[project] >= 0
            ? std::min(effective_cap, std::max(
                  irreversible, int(local_project_floor[project])))
            : irreversible;
        plan.crop_targets[crop] = int16_t(std::clamp(
            int(plan.crop_targets[crop]), effective_floor, effective_cap));
        if (local_project_cap[project] >= 0)
          local_project_cap[project] = int16_t(effective_cap);
      } else {
        const int animal = project - N_CROPS;
        const int animal_caps[N_ANIMALS] = {
            genome_.max_geese, genome_.max_cows, genome_.max_sheep};
        const int irreversible = own.total_animals[animal];
        const int normal_cap = day <= genome_.stop_new_animals_day
            ? animal_caps[animal] : irreversible;
        const int effective_cap = local_project_cap[project] >= 0
            ? std::min(normal_cap, std::max(
                  irreversible, int(local_project_cap[project])))
            : normal_cap;
        const int effective_floor = local_project_floor[project] >= 0
            ? std::min(effective_cap, std::max(
                  irreversible, int(local_project_floor[project])))
            : irreversible;
        plan.animal_targets[animal] = int16_t(std::clamp(
            int(plan.animal_targets[animal]), effective_floor, effective_cap));
        plan.animal_service_targets[animal] = int16_t(std::max(
            own.total_animals[animal], int(plan.animal_targets[animal])));
        if (local_project_cap[project] >= 0)
          local_project_cap[project] = int16_t(effective_cap);
      }
    }
  }

  const int days_left = TOTAL_DAYS - day;
  double budget = std::max(0.0, farm.money - genome_.cash_reserve);
  // A small fraction of liquid inventory may finance same-step projects.
  for (int item = 0; item < N_PRODUCTS; ++item)
    budget += 0.75 * priv.shed[item] * market.prices[item];

  auto future_price = [&](int item, int lag, double added_supply) {
    const double drift = demand[item] + genome_.demand_drift_weight * state.observed_daily_drift[item];
    const double projection = live_projection;
    double own_committed = own_supply[item] * lag;
    double opposing_committed = opp_supply[item] * lag;
    if (projection > 0.0) {
      own_committed = (1.0 - projection) * own_committed
          + projection * committed_supply_until(player, item, lag);
      opposing_committed = (1.0 - projection) * opposing_committed
          + projection * committed_supply_until(1 - player, item, lag);
    }
    const double opposing =
        genome_.opponent_supply_weight * opposing_committed;
    const double planned_impact = genome_.portfolio_supply_impact_weight > 0.0
        ? genome_.portfolio_supply_impact_weight
        : genome_.market_impact_weight;
    const double committed_impact =
        genome_.portfolio_supply_impact_weight > 0.0 && projection > 0.0
        ? genome_.portfolio_supply_impact_weight
        : genome_.market_impact_weight;
    const double own = committed_impact * own_committed
        + planned_impact * added_supply;
    // Future shops are unknown, but their distribution is part of the public
    // rules: one uniformly sampled shop is added every unlock interval until
    // eight are present.  Integrate only that expectation; never inspect the
    // RNG seed or a future event bank.  A shop unlocked at future_day affects
    // the remaining full days before the candidate cash-out horizon.
    double expected_future_shop_demand = 0.0;
    if (genome_.future_shop_expectation_weight > 0.0 &&
        item >= 0 && item < N_PRODUCTS && lag > 0) {
      double mean_shop_units_per_tick = 0.0;
      for (int shop = 0; shop < 8; ++shop)
        mean_shop_units_per_tick += SHOP_DEMAND[shop][item];
      mean_shop_units_per_tick /= 8.0;
      const int unlock_interval = std::max(
          1, env.config().town_shop_unlock_interval);
      const int sell_interval = std::max(
          1, env.config().town_shop_sell_interval);
      const double ticks_per_day =
          double(env.config().turns_per_day) / sell_interval;
      int remaining_unlocks = std::max(
          0, 8 - int(env.shops().size()));
      const int horizon_day = day + lag;
      for (int future_day = day + 1;
           future_day <= horizon_day && remaining_unlocks > 0;
           ++future_day) {
        if (future_day % unlock_interval != 0) continue;
        const int active_days = std::max(0, horizon_day - future_day);
        expected_future_shop_demand += active_days * ticks_per_day *
            mean_shop_units_per_tick;
        --remaining_unlocks;
      }
      expected_future_shop_demand *=
          genome_.future_shop_expectation_weight;
    }
    const int inv = int(std::nearbyint(
        market.inventory[item] - drift * lag - expected_future_shop_demand +
        opposing + own));
    return predicted_price(item, std::max(1, inv));
  };

  // Project-level autonomous memory.  Previous targets are not hard lower
  // bounds: that creates a ratchet in which every merely-positive project is
  // retained forever and the farm over-expands.  Instead, remember which
  // projects remain economically viable and let the joint marginal allocator
  // below rebuild the portfolio.  Yesterday's still-viable units receive a
  // small continuity bonus, so near-ties KEEP the operating plan while a
  // materially better use of cash/labour can SCALE, DEFER, SWITCH or CANCEL it.
  // No replay day, coordinate, quantity or opponent identity is available.
  std::array<int, N_ANIMALS> previous_animal_desired{};
  std::array<int, N_CROPS> previous_crop_desired{};
  std::array<uint8_t, N_ANIMALS> previous_animal_positive{};
  std::array<uint8_t, N_CROPS> previous_crop_positive{};
  int previous_quadrant_desired = popcount(farm.unlocked_mask);
  if (!backbone.enabled && state.plan.generated_day >= 0) {
    for (int animal = 0; animal < N_ANIMALS; ++animal)
      previous_animal_desired[animal] = std::max<int>(
          state.plan.animal_targets[animal],
          state.plan.deferred_animal_targets[animal]);
    for (int crop = 0; crop < N_CROPS; ++crop)
      previous_crop_desired[crop] = std::max<int>(
          state.plan.crop_targets[crop],
          state.plan.deferred_crop_targets[crop]);
    previous_quadrant_desired = std::max<int>(
        state.plan.quadrant_target, state.plan.deferred_quadrant_target);
  }
  if (!backbone.enabled && state.plan.generated_day >= 0 &&
      genome_.autonomous_project_state_machine > 0.0) {
    const int days_left_now = TOTAL_DAYS - day;
    for (int animal = 0; animal < N_ANIMALS; ++animal) {
      const int owned = own.total_animals[animal];
      const int desired = std::max(owned, previous_animal_desired[animal]);
      const int outstanding = std::max(0, desired - owned);
      if (outstanding <= 0) {
        previous_animal_positive[animal] = owned > 0;
        continue;
      }
      if (day > genome_.stop_new_animals_day ||
          day + ANIMAL_FIRST[animal] >= TOTAL_DAYS) {
        continue;
      }
      const int first = ANIMAL_FIRST[animal];
      const int interval = std::max(1, ANIMAL_INTERVAL[animal]);
      const int cycles = 1 + std::max(
          0, (TOTAL_DAYS - 1 - (day + first)) / interval);
      const double product_units = cycles * (1.0 + interval);
      const int product = ANIMAL_PRODUCT[animal];
      const int lag = std::max(first, first +
          std::max(0, days_left_now - first) / 2);
      const double revenue = product_units * future_price(
          product, lag, 0.5 * outstanding * product_units);
      const int service_days = std::max(0, TOTAL_DAYS - 1 - day);
      const double fertilizer = service_days * future_price(
          8, std::max(1, days_left_now / 2),
          0.5 * outstanding * service_days) *
          genome_.fertilizer_value_fraction;
      const double feed = service_days * future_price(
          0, std::max(1, days_left_now / 2),
          -0.5 * outstanding * service_days);
      const double actions = 3.0 * service_days + cycles + 5.0;
      const double unit_npv = revenue + fertilizer - ANIMAL_COST[animal] -
          feed - 0.25 * genome_.action_cost * actions -
          0.20 * genome_.move_cost * service_days;
      previous_animal_positive[animal] = unit_npv > 0.0;
    }
    for (int crop = 0; crop < N_CROPS; ++crop) {
      const int desired = previous_crop_desired[crop];
      if (desired <= 0) continue;
      if (!crop_project_open(genome_, crop, day)) continue;
      const int first = CROP_FIRST[crop];
      double units = CROP_MAX_YIELD[crop];
      if (CROP_ONGOING[crop]) {
        const int cycles = 1 + std::max(
            0, (TOTAL_DAYS - 1 - (day + first)) /
                   std::max(1, CROP_INTERVAL[crop]));
        units = std::min<double>(CROP_MAX_YIELD[crop], cycles);
      }
      const int lag = std::max(
          first, std::min(CROP_MAX_DAY[crop],
                          first + std::max(0, days_left_now - first) / 2));
      const double revenue = units * future_price(
          crop, lag, 0.5 * desired * units);
      const double actions = CROP_ONGOING[crop]
          ? 2.0 + std::ceil(std::min(days_left_now,
                CROP_MAX_DAY[crop] + 1) / 2.0) + units
          : 3.0 + CROP_MAX_DAY[crop];
      const double unit_npv = revenue - SEED_COST[crop] -
          0.25 * genome_.action_cost * actions -
          0.15 * genome_.move_cost * std::max(2, CROP_MAX_DAY[crop] / 2);
      previous_crop_positive[crop] = unit_npv > 0.0;
    }
  }

  struct Project { int kind; int id; double cost; double npv; double actions; double score; };
  // kind 0 crop, 1 animal
  double committed_cost = 0.0;
  {
    // Reserve cash for retained plan commitments before considering optional
    // expansion.  Market execution remains authoritative and may partially
    // fill orders; this only prevents a new project from crowding out work the
    // autonomous planner has already committed to finish.
    for (int animal = 0; animal < N_ANIMALS; ++animal)
      committed_cost += std::max(
          0, int(plan.animal_targets[animal]) - own.total_animals[animal]) *
          ANIMAL_COST[animal];
    for (int crop = 0; crop < N_CROPS; ++crop)
      committed_cost += std::max(
          0, int(plan.crop_targets[crop]) - own.committed_crops[crop]) *
          SEED_COST[crop];
    for (int quadrant = popcount(farm.unlocked_mask);
         quadrant < plan.quadrant_target && quadrant <= 3; ++quadrant)
      committed_cost += LAND_COST[quadrant - 1];
  }
  const double liquid_capital = budget + genome_.cash_reserve;
  if (!backbone.enabled && genome_.proactive_land_investment > 0.0 &&
      plan.quadrant_target < genome_.max_quadrants) {
    const int productive_stop = std::max(
        genome_.stop_new_animals_day, latest_generic_crop_start(genome_));
    const int next_land_index = std::clamp(
        int(plan.quadrant_target) - 1, 0, 2);
    const double land_cost = LAND_COST[next_land_index];
    // Six remaining days is the shortest horizon in which the faster crop
    // chains can still repay extra capacity.  The searchable gate controls how
    // much of the normal reserve must remain after the purchase.
    const double protected_cash = genome_.cash_reserve *
        genome_.proactive_land_investment;
    if (day + 6 <= productive_stop &&
        land_cost + protected_cash <= liquid_capital) {
      plan.quadrant_target++;
      committed_cost += land_cost;
    }
  }
  const int unlocked_tiles = 25 * popcount(farm.unlocked_mask);
  const double unit_day_capacity = 12.0 / std::max(0.5, genome_.risk_multiplier);
  auto workload_for = [&](const AdaptivePlan& candidate) {
    double workload = 4.0;
    for (int animal = 0; animal < N_ANIMALS; ++animal)
      workload += 3.1 * candidate.animal_service_targets[animal];
    for (int crop = 0; crop < N_CROPS; ++crop)
      // Current field crops remain obligations even when the desired next
      // portfolio intends to replace them.
      workload += (CROP_ONGOING[crop] ? 1.0 : 1.15) *
                  std::max<int>(candidate.crop_targets[crop], own.field_crops[crop]);
    return workload;
  };
  auto hands_for_workload = [&](double workload) {
    return std::max(0, int(std::ceil(workload / unit_day_capacity)) - 1);
  };

  // Estimate the *whole remaining project*, not one isolated marginal unit.
  // The old estimator priced every additional cow/sheep/crop as if it were the
  // first one.  Consequently the analytic score never saw its own planned
  // supply depressing the market and caps, rather than economics, selected the
  // final portfolio.  These helpers accumulate the public-state plan volume so
  // each new unit is valued after the units already selected in this plan.
  auto animal_units_until_terminal = [&](int animal) {
    const int first = ANIMAL_FIRST[animal];
    const int interval = ANIMAL_INTERVAL[animal];
    if (day + first >= TOTAL_DAYS) return 0.0;
    const int cycles = 1 + std::max(0, (TOTAL_DAYS - 1 - (day + first)) / interval);
    // FEED + CARE is an explicit obligation in build_tasks().  A completed care
    // cycle contributes one base unit plus the accumulated daily care bonus.
    return double(cycles * (1 + interval));
  };
  struct CropHorizon {
    double units = 0.0;
    double seed_cost = 0.0;
    double actions = 0.0;
    double weighted_lag = 0.0;
    int service_days = 0;
  };
  auto crop_horizon = [&](int crop) {
    CropHorizon out;
    const int first = CROP_FIRST[crop];
    if (day + first >= TOTAL_DAYS) return out;
    if (genome_.repeating_crop_economics <= 0.0) {
      out.units = CROP_MAX_YIELD[crop];
      if (CROP_ONGOING[crop]) {
        const int interval = std::max(1, CROP_INTERVAL[crop]);
        const int cycles = 1 + std::max(
            0, (TOTAL_DAYS - 1 - (day + first)) / interval);
        out.units = std::min<double>(CROP_MAX_YIELD[crop], cycles);
      }
      out.seed_cost = SEED_COST[crop];
      out.weighted_lag = out.units * std::max(first, CROP_MAX_DAY[crop] / 2);
      out.service_days = std::min(
          TOTAL_DAYS - day, CROP_MAX_DAY[crop] + 1);
      return out;
    }

    // A crop target denotes a maintained production slot, not one seed.  Walk
    // every lifecycle that can return product before the terminal market.
    // The loop is day-level and intentionally ignores hidden future random
    // events; it uses only official crop timings and the searchable stop day.
    int start = day;
    while (crop_project_open(genome_, crop, start)) {
      int harvests = 0;
      double units = 0.0;
      int next_start = TOTAL_DAYS;
      if (CROP_ONGOING[crop]) {
        const int interval = std::max(1, CROP_INTERVAL[crop]);
        harvests = 1 + std::max(
            0, (TOTAL_DAYS - 1 - (start + first)) / interval);
        harvests = std::min(harvests, CROP_MAX_YIELD[crop]);
        units = harvests;
        const int last_yield_age = first + (harvests - 1) * interval;
        next_start = start + last_yield_age + 1;
        const int active_days = std::max(1, next_start - start);
        out.actions += 1.0 + std::ceil(active_days / 2.0) + harvests + 1.0;
        for (int h = 0; h < harvests; ++h)
          out.weighted_lag += first + h * interval;
      } else {
        const int last_age = std::min(CROP_MAX_DAY[crop],
                                      TOTAL_DAYS - 1 - start);
        if (last_age < first) break;
        const int bonus_start = (CROP_MAX_DAY[crop] + 1) / 2;
        units = 1.0 + std::max(0, last_age - bonus_start + 1);
        units = std::min<double>(units, CROP_MAX_YIELD[crop]);
        harvests = 1;
        // A successful harvest clears a one-time crop, so the slot may be
        // replanted later in the same game day.  Day-level planning treats the
        // next cycle as starting at the harvested crop's target age.
        next_start = start + std::max(first, last_age);
        out.actions += 2.0 + std::max(1, last_age);
        out.weighted_lag += units * last_age;
      }
      if (units <= 0.0) break;
      out.units += units;
      out.seed_cost += SEED_COST[crop];
      out.service_days = std::max(out.service_days,
          std::min(TOTAL_DAYS - day, next_start - day + 1));
      if (next_start <= start) break;
      start = next_start;
    }
    return out;
  };
  auto crop_units_until_terminal = [&](int crop) {
    return crop_horizon(crop).units;
  };
  auto planned_new_supply = [&](const AdaptivePlan& candidate, int item) {
    double units = 0.0;
    for (int animal = 0; animal < N_ANIMALS; ++animal) {
      if (ANIMAL_PRODUCT[animal] != item) continue;
      const int added = std::max(0, int(candidate.animal_targets[animal])
                                      - own.field_animals[animal]);
      units += added * animal_units_until_terminal(animal);
    }
    if (item >= 0 && item < N_CROPS) {
      const int added = std::max(0, int(candidate.crop_targets[item])
                                      - own.field_crops[item]);
      units += added * crop_units_until_terminal(item);
    }
    return units;
  };
  auto planned_new_animal_days = [&](const AdaptivePlan& candidate) {
    double units = 0.0;
    for (int animal = 0; animal < N_ANIMALS; ++animal) {
      const int added = std::max(0, int(candidate.animal_targets[animal])
                                      - own.field_animals[animal]);
      units += added * std::max(0, TOTAL_DAYS - 1 - day);
    }
    return units;
  };
  for (int iteration = 0;
       iteration < (force_backbone_exact_ && backbone.enabled ? 0 : 96);
       ++iteration) {
    std::vector<Project> candidates;
    int animals = 0, crops = 0;
    for (auto value : plan.animal_targets) animals += value;
    for (auto value : plan.crop_targets) crops += value;

    if (day <= genome_.stop_new_animals_day && animals < genome_.max_total_animals) {
      const int animal_caps[N_ANIMALS] = {genome_.max_geese, genome_.max_cows, genome_.max_sheep};
      for (int a = 0; a < N_ANIMALS; ++a) {
        const int project = N_CROPS + a;
        const int effective_cap = local_project_cap[project] >= 0
            ? int(local_project_cap[project]) : animal_caps[a];
        if (plan.animal_targets[a] >= effective_cap) continue;
        const int first = ANIMAL_FIRST[a], interval = ANIMAL_INTERVAL[a];
        if (day + first >= TOTAL_DAYS) continue;
        const double units = animal_units_until_terminal(a);
        const int item = ANIMAL_PRODUCT[a];
        AdaptivePlan expanded = plan;
        expanded.animal_targets[a]++;
        expanded.animal_service_targets[a]++;
        const int average_lag = std::max(first, first + std::max(0, days_left - first) / 2);
        const double planned_product = planned_new_supply(expanded, item);
        const double revenue = units * future_price(item, average_lag, 0.5 * planned_product);
        const double planned_animal_days = planned_new_animal_days(expanded);
        const double collectible_fertilizer = std::max(0, TOTAL_DAYS - 1 - day);
        const double fertilizer_price = future_price(
            8, std::max(1, days_left / 2),
            0.5 * planned_animal_days * genome_.fertilizer_value_fraction);
        const double fert = collectible_fertilizer * fertilizer_price
                            * genome_.fertilizer_value_fraction;
        // Buying feed removes inventory and therefore raises, rather than
        // lowers, the price paid by the rest of the plan.
        const double feed_price = future_price(
            0, std::max(1, days_left / 2), -0.5 * planned_animal_days);
        const double feed = std::max(0, TOTAL_DAYS - 1 - day) * feed_price;
        const int cycles = 1 + std::max(0, (TOTAL_DAYS - 1 - (day + first)) / interval);
        const double actions = days_left * 3.0 + cycles + 5.0;
        const int hands_before = hands_for_workload(workload_for(plan));
        const int hands_after = hands_for_workload(workload_for(expanded));
        if (hands_after > genome_.max_hands) continue;
        const int service_days = std::max(0, TOTAL_DAYS - 1 - day);
        const double marginal_hire = service_days *
            std::max(0, daily_hire_cost(hands_after) - daily_hire_cost(hands_before));
        const int projected_tiles = animals + crops + 1;
        const int capacity = 25 * plan.quadrant_target;
        const int land_trigger = genome_.land_capacity_trigger_fraction > 0.0
            ? std::max(1, int(std::floor(
                  capacity * genome_.land_capacity_trigger_fraction)))
            : capacity - 2;
        const bool needs_land = projected_tiles > land_trigger &&
                                plan.quadrant_target < genome_.max_quadrants;
        const double marginal_land = needs_land
            ? LAND_COST[std::clamp(int(plan.quadrant_target) - 1, 0, 2)] : 0.0;
        const double npv = revenue + fert - ANIMAL_COST[a] - feed
                           - marginal_hire - marginal_land
                           - 0.25 * genome_.action_cost * actions
                           - 0.10 * genome_.move_cost * (2.0 * days_left);
        double score = npv / std::max(1.0, actions) /
            (1.0 + genome_.capital_lockup_weight * first);
        // Continuity is a tie-breaker, not a hard commitment.  Only units that
        // were planned yesterday and remain positive receive the bonus; a
        // materially better project can still replace them.
        if (genome_.autonomous_project_state_machine > 0.0 &&
            plan.animal_targets[a] < previous_animal_desired[a] &&
            previous_animal_positive[a]) {
          score *= 1.0 + genome_.autonomous_project_state_machine;
        }
        if (npv > 0.0) candidates.push_back({1, a, double(ANIMAL_COST[a]), npv, actions, score});
      }
    }

    {
      for (int crop = 0; crop < N_CROPS; ++crop) {
        if (!crop_project_open(genome_, crop, day)) continue;
        const int effective_cap = local_project_cap[crop] >= 0
            ? int(local_project_cap[crop]) : cap_for_crop(genome_, crop);
        if (plan.crop_targets[crop] >= effective_cap) continue;
        const int first = CROP_FIRST[crop], max_day = CROP_MAX_DAY[crop];
        if (day + first >= TOTAL_DAYS) continue;
        double units = CROP_MAX_YIELD[crop], actions = 0.0;
        double repeated_seed_cost = SEED_COST[crop];
        int lifecycle_days = std::min(days_left, max_day + 1);
        if (CROP_ONGOING[crop]) {
          const int cycles = 1 + std::max(0, (TOTAL_DAYS - 1 - (day + first)) /
                                                  std::max(1, CROP_INTERVAL[crop]));
          units = std::min<double>(CROP_MAX_YIELD[crop], cycles);
          lifecycle_days = std::min(days_left,
              first + CROP_INTERVAL[crop] * CROP_MAX_YIELD[crop] + 1);
          // Ongoing crops need survival watering roughly every other day, not
          // every day until the end of the whole 30-day match.
          actions = 1.0 + std::ceil(lifecycle_days / 2.0) + cycles;
        } else {
          actions = 2.0 + max_day + 1.0;
        }
        int average_lag = std::max(first, first +
            std::max(0, lifecycle_days - first) / 2);
        int service_days = std::min(
            days_left, std::max(first + 1, lifecycle_days));
        if (genome_.repeating_crop_economics > 0.0) {
          const auto horizon = crop_horizon(crop);
          units = horizon.units;
          actions = horizon.actions;
          repeated_seed_cost = horizon.seed_cost;
          average_lag = std::max(first, int(std::nearbyint(
              horizon.weighted_lag / std::max(1.0, horizon.units))));
          service_days = std::max(1, horizon.service_days);
        }
        AdaptivePlan expanded = plan;
        expanded.crop_targets[crop]++;
        const int hands_before = hands_for_workload(workload_for(plan));
        const int hands_after = hands_for_workload(workload_for(expanded));
        if (hands_after > genome_.max_hands) continue;
        const double planned_product = planned_new_supply(expanded, crop);
        const double revenue = units * future_price(crop, average_lag, 0.5 * planned_product);
        const double marginal_hire = service_days *
            std::max(0, daily_hire_cost(hands_after) - daily_hire_cost(hands_before));
        const int projected_tiles = animals + crops + 1;
        const int capacity = 25 * plan.quadrant_target;
        const int land_trigger = genome_.land_capacity_trigger_fraction > 0.0
            ? std::max(1, int(std::floor(
                  capacity * genome_.land_capacity_trigger_fraction)))
            : capacity - 2;
        const bool needs_land = projected_tiles > land_trigger &&
                                plan.quadrant_target < genome_.max_quadrants;
        const double marginal_land = needs_land
            ? LAND_COST[std::clamp(int(plan.quadrant_target) - 1, 0, 2)] : 0.0;
        const double npv = revenue - repeated_seed_cost
                           - marginal_hire - marginal_land
                           - 0.25 * genome_.action_cost * actions
                           - 0.10 * genome_.move_cost * std::max(2, max_day / 2);
        double score = npv / std::max(1.0, actions) /
            (1.0 + genome_.capital_lockup_weight * first);
        if (genome_.autonomous_project_state_machine > 0.0 &&
            plan.crop_targets[crop] < previous_crop_desired[crop] &&
            previous_crop_positive[crop]) {
          score *= 1.0 + genome_.autonomous_project_state_machine;
        }
        // Retaining an already committed crop slot needs no new cash now;
        // selecting beyond the current plants/seeds is a real purchase.
        const double cash_cost = plan.crop_targets[crop] < own.committed_crops[crop]
            ? 0.0 : double(SEED_COST[crop]);
        if (npv > 0.0) candidates.push_back({0, crop, cash_cost, npv, actions, score});
      }
    }
    if (candidates.empty()) break;
    std::sort(candidates.begin(), candidates.end(),
              [](const Project& a, const Project& b) { return a.score > b.score; });
    auto feasible = candidates.end();
    for (auto it = candidates.begin(); it != candidates.end(); ++it) {
      int next_animals = animals + (it->kind == 1 ? 1 : 0);
      int next_crops = crops + (it->kind == 0 ? 1 : 0);
      AdaptivePlan expanded = plan;
      if (it->kind == 1) {
        expanded.animal_targets[it->id]++;
        expanded.animal_service_targets[it->id]++;
      }
      else expanded.crop_targets[it->id]++;
      const int next_hands = hands_for_workload(workload_for(expanded));
      if (next_hands > genome_.max_hands) continue;
      int earliest_income = 8;
      for (int a = 0; a < N_ANIMALS; ++a)
        if (plan.animal_targets[a] + (it->kind == 1 && it->id == a) > 0)
          earliest_income = std::min(earliest_income, ANIMAL_FIRST[a]);
      for (int crop = 0; crop < N_CROPS; ++crop)
        if (plan.crop_targets[crop] + (it->kind == 0 && it->id == crop) > 0)
          earliest_income = std::min(earliest_income, CROP_FIRST[crop]);
      const double daily_feed = next_animals * market.prices[0];
      const double runway = std::min(8, std::max(2, earliest_income));
      const double liquidity_reserve = genome_.cash_reserve
          + runway * (daily_feed + daily_hire_cost(next_hands));
      if (committed_cost + it->cost + liquidity_reserve <= liquid_capital) {
        feasible = it;
        break;
      }
    }
    if (feasible == candidates.end() || feasible->npv <= 0.0) break;
    auto best = feasible;
    const int projected_tiles = animals + crops + 1;
    int available_tiles = 25 * plan.quadrant_target;
    const int land_trigger = genome_.land_capacity_trigger_fraction > 0.0
        ? std::max(1, int(std::floor(
              available_tiles * genome_.land_capacity_trigger_fraction)))
        : available_tiles - 2;
    if (projected_tiles > land_trigger &&
        plan.quadrant_target < genome_.max_quadrants) {
      const int extra = plan.quadrant_target - 1;
      const int land_cost = LAND_COST[std::clamp(extra, 0, 2)];
      if (committed_cost + land_cost <= budget) {
        committed_cost += land_cost;
        plan.quadrant_target++;
        available_tiles += 25;
      }
    }
    if (projected_tiles > 25 * plan.quadrant_target - 2) break;
    committed_cost += best->cost;
    plan.expected_incremental_value += best->npv;
    if (best->kind == 1) {
      plan.animal_targets[best->id]++;
      plan.animal_service_targets[best->id]++;
    }
    else plan.crop_targets[best->id]++;
  }

  // The marginal allocator above is intentionally cheap, but one-unit greedy
  // additions cannot cross a shared fixed-cost boundary.  For example, the
  // first strawberry slot that triggers another hand or quadrant may look
  // unprofitable even though a four-slot lot amortises that cost; conversely a
  // cow-heavy draft may be inferior to releasing a still-unbought animal
  // commitment and assigning the capacity to a crop lot.  When explicitly
  // enabled, compare a bounded set of complete, reversible portfolio edits.
  // This uses only the live public state and official economics.  There is no
  // Replay date, coordinate, opponent identity or fixed industry route.
  const bool switch_cooldown_ready = state.bundle_switches == 0 ||
      day - int(state.last_bundle_switch_day) >=
          genome_.portfolio_switch_cooldown_days;
  const bool offline_expose_all =
      genome_.offline_expose_all_feasible_switches;
  if (!backbone.enabled &&
      (genome_.portfolio_bundle_switch_margin > 0.0 || offline_expose_all) &&
      switch_cooldown_ready) {
    // Candidate8 searches official physical capacity, independently of the
    // conservative R6 prior.  Feasibility still enforces live cash, action
    // load, tiles and the four-quadrant board.
    const int animal_caps[N_ANIMALS] = {50, 50, 50};

    auto target_for = [&](const AdaptivePlan& candidate, int project) {
      return project < N_CROPS
          ? int(candidate.crop_targets[project])
          : int(candidate.animal_targets[project - N_CROPS]);
    };
    auto floor_for = [&](int project) {
      return project < N_CROPS
          // A seed already bought is an irreversible cash commitment even
          // though it has not reached a tile yet.  Letting SWITCH remove it
          // stranded shed inventory and made the whole downstream planting
          // chain disappear.  Only the still-unbought part of a crop target
          // is a reversible portfolio choice.
          ? own.committed_crops[project]
          : own.total_animals[project - N_CROPS];
    };
    auto cap_for = [&](int project) {
      return project < N_CROPS
          ? cap_for_crop(genome_, project)
          : animal_caps[project - N_CROPS];
    };
    auto apply_delta = [&](AdaptivePlan& candidate, int project, int delta) {
      if (project < N_CROPS) {
        candidate.crop_targets[project] = int16_t(
            int(candidate.crop_targets[project]) + delta);
      } else {
        const int animal = project - N_CROPS;
        candidate.animal_targets[animal] = int16_t(
            int(candidate.animal_targets[animal]) + delta);
        candidate.animal_service_targets[animal] = int16_t(std::max(
            own.field_animals[animal],
            int(candidate.animal_service_targets[animal]) + delta));
      }
    };
    auto normalise_and_feasible = [&](AdaptivePlan& candidate) {
      int animals = 0;
      int productive_tiles = 0;
      for (int crop = 0; crop < N_CROPS; ++crop) {
        const int target = int(candidate.crop_targets[crop]);
        if (target < 0 || target > cap_for_crop(genome_, crop)) return false;
        productive_tiles += std::max(target, own.field_crops[crop]);
      }
      for (int animal = 0; animal < N_ANIMALS; ++animal) {
        const int target = int(candidate.animal_targets[animal]);
        if (target < own.total_animals[animal] || target > animal_caps[animal])
          return false;
        candidate.animal_service_targets[animal] = int16_t(std::max(
            own.field_animals[animal],
            std::min(target, int(candidate.animal_service_targets[animal]))));
        animals += target;
        productive_tiles += std::max(target, own.field_animals[animal]);
      }
      if (animals > genome_.max_total_animals) return false;

      const int unlocked = popcount(farm.unlocked_mask);
      const int required_quadrants = std::max(
          unlocked, (productive_tiles + 2 + 24) / 25);
      if (required_quadrants > genome_.max_quadrants) return false;
      candidate.quadrant_target = int16_t(required_quadrants);

      const int candidate_hands = hands_for_workload(workload_for(candidate));
      if (candidate_hands > genome_.max_hands) return false;
      candidate.hand_target = int16_t(candidate_hands);

      double required_cash = 0.0;
      for (int crop = 0; crop < N_CROPS; ++crop)
        required_cash += std::max(
            0, int(candidate.crop_targets[crop]) - own.committed_crops[crop]) *
            SEED_COST[crop];
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        required_cash += std::max(
            0, int(candidate.animal_targets[animal]) - own.total_animals[animal]) *
            ANIMAL_COST[animal];
      for (int q = unlocked; q < required_quadrants && q <= 3; ++q)
        required_cash += LAND_COST[q - 1];

      int earliest_income = 8;
      for (int crop = 0; crop < N_CROPS; ++crop)
        if (candidate.crop_targets[crop] > 0)
          earliest_income = std::min(earliest_income, CROP_FIRST[crop]);
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        if (candidate.animal_targets[animal] > 0)
          earliest_income = std::min(earliest_income, ANIMAL_FIRST[animal]);
      const double daily_feed = animals * market.prices[0];
      const double runway = std::min(8, std::max(2, earliest_income));
      const double liquidity_reserve = genome_.cash_reserve +
          runway * (daily_feed + daily_hire_cost(candidate_hands));
      return required_cash + liquidity_reserve <= liquid_capital;
    };
    auto bundle_score = [&](const AdaptivePlan& candidate,
                            PlanValueBreakdown* breakdown) {
      // Existing field projects are real obligations and must contribute the
      // same baseline value even when the future replant target is zero.
      AdaptivePlan score_view = candidate;
      for (int crop = 0; crop < N_CROPS; ++crop)
        score_view.crop_targets[crop] = int16_t(std::max(
            int(score_view.crop_targets[crop]), own.field_crops[crop]));
      for (int animal = 0; animal < N_ANIMALS; ++animal) {
        score_view.animal_targets[animal] = int16_t(std::max(
            int(score_view.animal_targets[animal]), own.total_animals[animal]));
        score_view.animal_service_targets[animal] = int16_t(std::max(
            int(score_view.animal_service_targets[animal]),
            own.field_animals[animal]));
      }
      return score_plan_candidate(
          env, player, score_view, breakdown, &state);
    };

    const AdaptivePlan bundle_baseline_plan = plan;
    AdaptivePlan bundle_best = plan;
    PlanValueBreakdown bundle_baseline_breakdown;
    PlanValueBreakdown bundle_best_breakdown;
    const double bundle_baseline_score =
        bundle_score(plan, &bundle_baseline_breakdown);
    double bundle_best_score = bundle_baseline_score;
    int bundle_best_source = -1;
    int bundle_best_destination = -1;
    int bundle_best_remove = 0;
    int bundle_best_add = 0;
    struct BundleOption {
      AdaptivePlan plan;
      double score = 0.0;
      int source = -1;
      int destination = -1;
      int remove_count = 0;
      int add_count = 0;
    };
    std::vector<BundleOption> bundle_options;
    const double required_gain = genome_.portfolio_bundle_switch_margin *
        std::max(1.0, std::abs(bundle_baseline_score));
    const int batch_pairs[4][2] = {{2, 2}, {2, 4}, {4, 2}, {4, 4}};
    for (int source = 0; source < PROJECTS; ++source) {
      for (int destination = 0; destination < PROJECTS; ++destination) {
        if (source == destination) continue;
        for (const auto& pair : batch_pairs) {
          const int remove_count = pair[0];
          const int add_count = pair[1];
          if (target_for(plan, source) - remove_count < floor_for(source))
            continue;
          if (target_for(plan, destination) + add_count > cap_for(destination))
            continue;
          AdaptivePlan candidate = plan;
          apply_delta(candidate, source, -remove_count);
          apply_delta(candidate, destination, add_count);
          if (!normalise_and_feasible(candidate)) continue;
          const double score = bundle_score(candidate, nullptr);
          if (offline_expose_all ||
              score > bundle_baseline_score + required_gain)
            bundle_options.push_back({candidate, score, source, destination,
                                      remove_count, add_count});
        }
      }
    }
    std::sort(bundle_options.begin(), bundle_options.end(),
              [](const BundleOption& lhs, const BundleOption& rhs) {
                return lhs.score > rhs.score;
              });
    const int requested_rank = genome_.portfolio_switch_candidate_rank;
    if (requested_rank >= 0 &&
        requested_rank < int(bundle_options.size())) {
      const auto& selected = bundle_options[requested_rank];
      bundle_best = selected.plan;
      bundle_best_score = selected.score;
      bundle_best_source = selected.source;
      bundle_best_destination = selected.destination;
      bundle_best_remove = selected.remove_count;
      bundle_best_add = selected.add_count;
    }
    // The established marginal allocator already owns SCALE.  This second
    // stage is deliberately SWITCH-only: accepting extra standalone batches
    // duplicated the scaler and pushed otherwise feasible farms beyond their
    // realised logistics capacity.  Every candidate here must release an
    // outstanding, still-reversible commitment before adding another one.
    if (bundle_best_source >= 0 &&
        (offline_expose_all || bundle_best_score > bundle_baseline_score)) {
      // Recompute the selected plan once to expose the same score components
      // used by the analytic evaluator.  The scorer is deterministic and
      // reads only current public/self state, so this is diagnostic rather
      // than a second decision path.
      bundle_best_score = bundle_score(bundle_best, &bundle_best_breakdown);
      bundle_best.bundle_switch_applied = 1;
      bundle_best.bundle_predicted_gain =
          bundle_best_score - bundle_baseline_score;
      bundle_best.bundle_switch_source = int8_t(bundle_best_source);
      bundle_best.bundle_switch_destination = int8_t(bundle_best_destination);
      bundle_best.bundle_switch_source_target = int16_t(
          target_for(bundle_best, bundle_best_source));
      bundle_best.bundle_switch_destination_target = int16_t(
          target_for(bundle_best, bundle_best_destination));
      auto& features = bundle_best.bundle_switch_features;
      int feature = 0;
      features[feature++] = day;
      features[feature++] = env.hour();
      features[feature++] = int32_t(std::nearbyint(farm.money));
      features[feature++] = int32_t(farm.hands.size());
      features[feature++] = popcount(farm.unlocked_mask);
      features[feature++] = shed_sum(priv);
      int carried_units = 0;
      for (const auto& inventory : priv.inventories)
        carried_units += inventory_sum(inventory);
      features[feature++] = carried_units;
      features[feature++] = own.weeds;
      features[feature++] = own.ready_harvest;
      features[feature++] = own.hard_water;
      features[feature++] = own.hard_feed;
      features[feature++] = bundle_best_source;
      features[feature++] = bundle_best_destination;
      features[feature++] = bundle_best_remove;
      features[feature++] = bundle_best_add;
      features[feature++] = int32_t(std::nearbyint(bundle_baseline_score));
      features[feature++] = int32_t(std::nearbyint(bundle_best_score));
      features[feature++] = hands_for_workload(workload_for(bundle_baseline_plan));
      features[feature++] = hands_for_workload(workload_for(bundle_best));
      features[feature++] = bundle_baseline_plan.quadrant_target;
      features[feature++] = bundle_best.quadrant_target;
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        features[feature++] = own.total_animals[animal];
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        features[feature++] = own.field_animals[animal];
      for (int crop = 0; crop < N_CROPS; ++crop)
        features[feature++] = own.field_crops[crop];
      for (int crop = 0; crop < N_CROPS; ++crop)
        features[feature++] = own.committed_crops[crop];
      for (int crop = 0; crop < N_CROPS; ++crop)
        features[feature++] = bundle_baseline_plan.crop_targets[crop];
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        features[feature++] = bundle_baseline_plan.animal_targets[animal];
      for (int crop = 0; crop < N_CROPS; ++crop)
        features[feature++] = bundle_best.crop_targets[crop];
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        features[feature++] = bundle_best.animal_targets[animal];
      for (int item = 0; item < N_PRODUCTS; ++item)
        features[feature++] = market.prices[item];
      for (int item = 0; item < N_PRODUCTS; ++item)
        features[feature++] = market.inventory[item];
      int shop_mask = 0;
      for (const int shop : env.shops())
        if (shop >= 0 && shop < 8) shop_mask |= 1 << shop;
      features[feature++] = shop_mask;
      for (int crop = 0; crop < N_CROPS; ++crop)
        features[feature++] = int(bundle_best.crop_targets[crop]) -
                              int(bundle_baseline_plan.crop_targets[crop]);
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        features[feature++] = int(bundle_best.animal_targets[animal]) -
                              int(bundle_baseline_plan.animal_targets[animal]);
      for (int crop = 0; crop < N_CROPS; ++crop)
        features[feature++] = priv.seeds[crop];
      int fertilizer_units = priv.shed[8];
      int wheat_units = priv.shed[0];
      int shed_market_value = 0;
      int carried_market_value = 0;
      for (int item = 0; item < N_PRODUCTS; ++item)
        shed_market_value += priv.shed[item] * market.prices[item];
      for (const auto& inventory : priv.inventories) {
        fertilizer_units += inventory[8];
        wheat_units += inventory[0];
        for (int item = 0; item < N_PRODUCTS; ++item)
          carried_market_value += inventory[item] * market.prices[item];
      }
      features[feature++] = fertilizer_units;
      features[feature++] = wheat_units;
      features[feature++] = own.empty;
      features[feature++] = own.empty_pastures;
      features[feature++] = own.empty_coops;
      auto shed_distance = [&](Position pos) {
        int best = 100;
        for (const Position shed : SHED_POS)
          best = std::min(best, distance(pos, shed));
        return best;
      };
      int worker_distance = shed_distance(farm.farmer);
      for (const Position hand : farm.hands)
        worker_distance += shed_distance(hand);
      int crop_distance = 0, animal_distance = 0;
      int crop_yield = 0, animal_yield = 0;
      int fertilizer_ready = 0, unwatered = 0, unfed = 0;
      const int board = env.config().board_size;
      for (int cell = 0; cell < int(farm.tiles.size()); ++cell) {
        const auto& tile = farm.tiles[cell];
        const Position pos{int16_t(cell % board), int16_t(cell / board)};
        if (tile.kind == TileKind::PLANT) {
          crop_distance += shed_distance(pos);
          crop_yield += tile.yield_units;
          unwatered += !tile.watered_today;
        } else if (tile.kind == TileKind::ANIMAL) {
          animal_distance += shed_distance(pos);
          animal_yield += tile.yield_units;
          fertilizer_ready += tile.fertilizer_available;
          unfed += !tile.fed_today;
        }
      }
      features[feature++] = worker_distance;
      features[feature++] = crop_distance;
      features[feature++] = animal_distance;
      features[feature++] = crop_yield;
      features[feature++] = animal_yield;
      features[feature++] = fertilizer_ready;
      features[feature++] = unwatered;
      features[feature++] = unfed;
      int baseline_unmet_crops = 0, candidate_unmet_crops = 0;
      int baseline_unmet_animals = 0, candidate_unmet_animals = 0;
      int baseline_total_targets = 0, candidate_total_targets = 0;
      for (int crop = 0; crop < N_CROPS; ++crop) {
        baseline_unmet_crops += std::max(
            0, int(bundle_baseline_plan.crop_targets[crop]) -
                   own.committed_crops[crop]);
        candidate_unmet_crops += std::max(
            0, int(bundle_best.crop_targets[crop]) -
                   own.committed_crops[crop]);
        baseline_total_targets += bundle_baseline_plan.crop_targets[crop];
        candidate_total_targets += bundle_best.crop_targets[crop];
      }
      for (int animal = 0; animal < N_ANIMALS; ++animal) {
        baseline_unmet_animals += std::max(
            0, int(bundle_baseline_plan.animal_targets[animal]) -
                   own.total_animals[animal]);
        candidate_unmet_animals += std::max(
            0, int(bundle_best.animal_targets[animal]) -
                   own.total_animals[animal]);
        baseline_total_targets += bundle_baseline_plan.animal_targets[animal];
        candidate_total_targets += bundle_best.animal_targets[animal];
      }
      features[feature++] = baseline_unmet_crops;
      features[feature++] = candidate_unmet_crops;
      features[feature++] = baseline_unmet_animals;
      features[feature++] = candidate_unmet_animals;
      features[feature++] = baseline_total_targets;
      features[feature++] = candidate_total_targets;
      features[feature++] = int32_t(std::nearbyint(
          100.0 * workload_for(bundle_baseline_plan)));
      features[feature++] = int32_t(std::nearbyint(
          100.0 * workload_for(bundle_best)));
      features[feature++] = int32_t(std::nearbyint(liquid_capital));
      features[feature++] = int32_t(std::nearbyint(
          farm.money - genome_.cash_reserve));
      features[feature++] = shed_market_value;
      features[feature++] = carried_market_value;
      features[feature++] = state.plan.generated_day;
      features[feature++] = state.replans;

      // Public opponent production is essential context for a market switch.
      // The earlier 112-feature schema contained our own farm and the shared
      // market but omitted the visible opposing farm entirely.  As a result,
      // the same candidate looked identical to the learned value model whether
      // the opponent had no industry or twenty competing production tiles.
      // Scan only public tiles, public workforce, public land and public cash;
      // no opponent shed, seed inventory or identity is available here.
      const auto& opponent_farm = env.farms()[1 - player];
      std::array<int, N_CROPS> opponent_field_crops{};
      std::array<int, N_ANIMALS> opponent_field_animals{};
      std::array<int, N_CROPS> opponent_crop_yield{};
      std::array<int, N_ANIMALS> opponent_animal_yield{};
      int opponent_weeds = 0;
      int opponent_ready = 0;
      int opponent_hard_water = 0;
      int opponent_hard_feed = 0;
      for (const auto& tile : opponent_farm.tiles) {
        if (tile.kind == TileKind::PLANT) {
          const int crop = int(tile.crop);
          if (crop >= 0 && crop < N_CROPS) {
            opponent_field_crops[crop]++;
            opponent_crop_yield[crop] += tile.yield_units;
          }
          opponent_ready += tile.yield_units > 0;
          opponent_hard_water +=
              !tile.watered_today && tile.consecutive_unwatered >= 1;
        } else if (tile.kind == TileKind::ANIMAL) {
          const int animal = int(tile.animal) - 9;
          if (animal >= 0 && animal < N_ANIMALS) {
            opponent_field_animals[animal]++;
            opponent_animal_yield[animal] += tile.yield_units;
          }
          opponent_ready += tile.yield_units > 0;
          opponent_hard_feed +=
              !tile.fed_today && tile.consecutive_unfed >= 1;
        } else if (tile.kind == TileKind::WEED) {
          opponent_weeds++;
        }
      }
      for (const int value : opponent_field_crops)
        features[feature++] = value;
      for (const int value : opponent_field_animals)
        features[feature++] = value;
      for (const int value : opponent_crop_yield)
        features[feature++] = value;
      for (const int value : opponent_animal_yield)
        features[feature++] = value;
      features[feature++] = int32_t(opponent_farm.hands.size());
      features[feature++] = popcount(opponent_farm.unlocked_mask);
      features[feature++] = opponent_weeds;
      features[feature++] = opponent_ready;
      features[feature++] = opponent_hard_water;
      features[feature++] = opponent_hard_feed;
      for (const double value : opp_supply)
        features[feature++] = int32_t(std::nearbyint(100.0 * value));
      features[feature++] = int32_t(std::nearbyint(opponent_farm.money));
      features[feature++] = player;

      // Execution-feasibility context.  A target count alone does not reveal
      // whether its current assets are already mature, far from logistics,
      // one missed service away from loss, or still locked in inventory.  The
      // learned/analytic value layer needs those generic facts to distinguish
      // an executable project edit from the same edit in a superficially
      // similar but operationally incompatible state.
      for (int item = 0; item < N_PRODUCTS; ++item)
        features[feature++] = priv.shed[item];
      std::array<int, N_PRODUCTS> carried_by_product{};
      for (const auto& inventory : priv.inventories)
        for (int item = 0; item < N_PRODUCTS; ++item)
          carried_by_product[item] += inventory[item];
      for (const int value : carried_by_product)
        features[feature++] = value;

      std::array<int, N_CROPS> own_crop_yield{};
      std::array<int, N_ANIMALS> own_animal_yield{};
      std::array<int, N_CROPS> own_crop_age_sum{};
      std::array<int, N_ANIMALS> own_animal_age_sum{};
      std::array<int, N_CROPS> own_crop_stress{};
      std::array<int, N_ANIMALS> own_animal_stress{};
      std::array<int, N_CROPS> own_crop_distance{};
      std::array<int, N_ANIMALS> own_animal_distance{};
      for (int cell = 0; cell < int(farm.tiles.size()); ++cell) {
        const auto& tile = farm.tiles[cell];
        const Position pos{int16_t(cell % board), int16_t(cell / board)};
        if (tile.kind == TileKind::PLANT) {
          const int crop = int(tile.crop);
          if (crop < 0 || crop >= N_CROPS) continue;
          own_crop_yield[crop] += std::max(0, int(tile.yield_units));
          own_crop_age_sum[crop] += std::max(0, day - int(tile.planted_day));
          own_crop_stress[crop] +=
              !tile.watered_today && tile.consecutive_unwatered >= 1;
          own_crop_distance[crop] += shed_distance(pos);
        } else if (tile.kind == TileKind::ANIMAL) {
          const int animal = int(tile.animal) - 9;
          if (animal < 0 || animal >= N_ANIMALS) continue;
          own_animal_yield[animal] += std::max(0, int(tile.yield_units));
          own_animal_age_sum[animal] +=
              std::max(0, day - int(tile.placed_day));
          own_animal_stress[animal] +=
              !tile.fed_today && tile.consecutive_unfed >= 1;
          own_animal_distance[animal] += shed_distance(pos);
        }
      }
      for (const int value : own_crop_yield) features[feature++] = value;
      for (const int value : own_animal_yield) features[feature++] = value;
      for (const int value : own_crop_age_sum) features[feature++] = value;
      for (const int value : own_animal_age_sum) features[feature++] = value;
      for (const int value : own_crop_stress) features[feature++] = value;
      for (const int value : own_animal_stress) features[feature++] = value;
      for (const int value : own_crop_distance) features[feature++] = value;
      for (const int value : own_animal_distance) features[feature++] = value;

      // Horizon projections use only already-visible commitments.  They do
      // not inspect future RNG, private opponent inventory or route identity.
      // The three lags expose whether each market will receive supply before,
      // near, or after a proposed project's own cash-return window.
      constexpr int projection_lags[3] = {2, 4, 8};
      for (const int lag : projection_lags) {
        for (int item = 0; item < N_PRODUCTS; ++item)
          features[feature++] = int32_t(std::nearbyint(
              100.0 * committed_supply_until(player, item, lag)));
        for (int item = 0; item < N_PRODUCTS; ++item)
          features[feature++] = int32_t(std::nearbyint(
              100.0 * committed_supply_until(1 - player, item, lag)));
      }
      for (const int value : state.opponent_project_count_trend_x100)
        features[feature++] = value;
      for (const int value : state.opponent_project_yield_trend_x100)
        features[feature++] = value;
      features[feature++] = state.opponent_cash_trend;
      features[feature++] = state.opponent_hands_trend_x100;
      features[feature++] = state.opponent_quadrants_trend_x100;
      features[feature++] = state.opponent_history_samples;
      for (const double value : state.observed_daily_drift)
        features[feature++] = int32_t(std::nearbyint(100.0 * value));
      auto append_value_breakdown = [&](const PlanValueBreakdown& value) {
        features[feature++] = int32_t(std::nearbyint(value.crop_gross));
        features[feature++] = int32_t(std::nearbyint(value.animal_gross));
        features[feature++] = int32_t(std::nearbyint(value.fertilizer_gross));
        features[feature++] = int32_t(std::nearbyint(value.seed_cost));
        features[feature++] = int32_t(std::nearbyint(value.feed_cost));
        features[feature++] = int32_t(std::nearbyint(
            value.animal_purchase_cost));
        features[feature++] = int32_t(std::nearbyint(value.action_cost));
        features[feature++] = int32_t(std::nearbyint(value.move_cost));
        features[feature++] = int32_t(std::nearbyint(value.hire_cost));
        features[feature++] = int32_t(std::nearbyint(value.land_cost));
        features[feature++] = int32_t(std::nearbyint(value.lockup_cost));
        features[feature++] = int32_t(std::nearbyint(value.crop_units));
        features[feature++] = int32_t(std::nearbyint(
            value.animal_product_units));
        features[feature++] = int32_t(std::nearbyint(value.fertilizer_used));
        features[feature++] = int32_t(std::nearbyint(
            value.fertilizer_sellable));
        features[feature++] = int32_t(std::nearbyint(value.setup_turns));
        features[feature++] = int32_t(std::nearbyint(value.first_cash_lag));
        features[feature++] = int32_t(std::nearbyint(
            100.0 * value.immediate_commitment_actions));
        features[feature++] = int32_t(std::nearbyint(
            100.0 * value.today_action_capacity));
        features[feature++] = int32_t(std::nearbyint(
            100.0 * value.today_deadline_slack));
        features[feature++] = int32_t(std::nearbyint(
            value.peak_daily_utilization_x100));
        for (const double projected : value.window_action_demand)
          features[feature++] = int32_t(std::nearbyint(100.0 * projected));
        for (const double projected : value.window_action_capacity)
          features[feature++] = int32_t(std::nearbyint(100.0 * projected));
        for (const double projected : value.window_net_cash)
          features[feature++] = int32_t(std::nearbyint(projected));
        features[feature++] = int32_t(std::nearbyint(
            100.0 * value.minimum_window_slack));
        features[feature++] = int32_t(std::nearbyint(value.current_task_count));
        features[feature++] = int32_t(std::nearbyint(value.current_hard_task_count));
        features[feature++] = int32_t(std::nearbyint(value.assigned_task_count));
        features[feature++] = int32_t(std::nearbyint(value.assigned_hard_task_count));
        features[feature++] = int32_t(std::nearbyint(value.total_assignment_distance));
        features[feature++] = int32_t(std::nearbyint(value.total_assignment_steps));
        features[feature++] = int32_t(std::nearbyint(value.minimum_assignment_slack));
        features[feature++] = int32_t(std::nearbyint(
            value.deadline_infeasible_task_count));
        features[feature++] = int32_t(std::nearbyint(
            value.unassigned_hard_task_count));
        features[feature++] = int32_t(std::nearbyint(
            value.assigned_expected_cash_gain));
        features[feature++] = int32_t(std::nearbyint(
            value.unassigned_delayed_loss));
        features[feature++] = int32_t(std::nearbyint(
            100.0 * value.assigned_value_per_step));
        features[feature++] = int32_t(std::nearbyint(
            value.commissioning_job_count));
        features[feature++] = int32_t(std::nearbyint(
            value.commissioning_total_steps));
        features[feature++] = int32_t(std::nearbyint(
            value.commissioning_makespan_steps));
        features[feature++] = int32_t(std::nearbyint(
            value.commissioning_first_cash_slack_steps));
        features[feature++] = int32_t(std::nearbyint(
            value.commissioning_unreachable_count));
        features[feature++] = int32_t(std::nearbyint(
            value.new_service_distance));
        features[feature++] = int32_t(std::nearbyint(
            value.new_service_route_span));
        features[feature++] = int32_t(std::nearbyint(
            100.0 * value.commissioning_steps_per_added_unit));
      };
      append_value_breakdown(bundle_baseline_breakdown);
      append_value_breakdown(bundle_best_breakdown);
      // A counterfactual candidate is valued under the continuation planner,
      // not under the public board alone.  These public/self policy-state
      // fields make that continuation contract observable to the ranker.
      // Without them, identical boards generated with one- versus four-land
      // ceilings receive contradictory labels that no model can identify.
      features[feature++] = genome_.max_quadrants;
      features[feature++] = genome_.stop_new_crops_day;
      features[feature++] = genome_.stop_new_animals_day;
      features[feature++] = genome_.portfolio_local_edit_hold_days;
      features[feature++] = int32_t(std::nearbyint(
          100.0 * genome_.land_capacity_trigger_fraction));
      features[feature++] = int32_t(std::nearbyint(
          100.0 * genome_.proactive_land_investment));
      assert(feature == AdaptivePlan::SWITCH_FEATURE_DIM);
      plan = bundle_best;
      plan.expected_incremental_value = bundle_best_score;
    }
  }

  // Reapply sparse Candidate8 macro commitments before generating the next
  // edit.  Previously an edit disappeared at the next ordinary replan, so a
  // multi-stage search measured a chain of one-day patches rather than one
  // coherent operating plan.
  for (int crop = 0; crop < N_CROPS; ++crop) {
    const int locked = state.candidate8_persistent_target[crop];
    if (locked >= 0)
      plan.crop_targets[crop] = int16_t(std::clamp(
          std::max(locked, int(own.committed_crops[crop])), 0, 100));
  }
  for (int animal = 0; animal < N_ANIMALS; ++animal) {
    const int project = N_CROPS + animal;
    const int locked = state.candidate8_persistent_target[project];
    if (locked >= 0) {
      const int target = std::clamp(
          std::max(locked, int(own.total_animals[animal])), 0, 50);
      plan.animal_targets[animal] = int16_t(target);
      plan.animal_service_targets[animal] = int16_t(std::max(
          int(own.field_animals[animal]), target));
    }
  }
  if (state.candidate8_persistent_quadrant_target >= 0)
    plan.quadrant_target = int16_t(std::clamp(
        std::max(int(plan.quadrant_target),
                 int(state.candidate8_persistent_quadrant_target)),
        1, 4));
  if (state.candidate8_persistent_schedule_profile !=
      int8_t(ScheduleProfile::CURRENT))
    plan.candidate8_schedule_profile =
        state.candidate8_persistent_schedule_profile;

  // Offline broad-candidate bridge.  The generator is deliberately disabled
  // for every saved/runtime genome; counterfactual tooling enables exactly
  // one sparse PlanDelta at the first eligible public state.  This turns the
  // eight capability families into real executable continuations without
  // installing the cheap analytic screen as an online policy.
  if (genome_.offline_candidate8_enabled &&
      state.candidate8_decisions == 0 &&
      day >= genome_.offline_candidate8_minimum_day) {
    AdaptiveCandidateContext context;
    context.day = day;
    const int animal_caps[N_ANIMALS] = {
        genome_.max_geese, genome_.max_cows, genome_.max_sheep};
    for (int crop = 0; crop < N_CROPS; ++crop) {
      context.irreversible_floor[crop] = int16_t(own.committed_crops[crop]);
      context.targets[crop] = int16_t(std::max(
          int(plan.crop_targets[crop]),
          int(context.irreversible_floor[crop])));
      context.caps[crop] = int16_t(100);
      context.purchase_cost[crop] = SEED_COST[crop];
      context.daily_action_load[crop] = CROP_ONGOING[crop]
          ? 0.5 + 1.0 / std::max(1, CROP_INTERVAL[crop])
          : (2.0 + CROP_MAX_DAY[crop]) /
                std::max(1, CROP_MAX_DAY[crop] + 1);
      context.first_cash_lag_days[crop] = int16_t(CROP_FIRST[crop]);
    }
    for (int animal = 0; animal < N_ANIMALS; ++animal) {
      const int project = N_CROPS + animal;
      context.irreversible_floor[project] = int16_t(own.total_animals[animal]);
      context.targets[project] = int16_t(std::max(
          int(plan.animal_targets[animal]),
          int(context.irreversible_floor[project])));
      context.caps[project] = int16_t(animal_caps[animal]);
      context.purchase_cost[project] = ANIMAL_COST[animal];
      context.daily_action_load[project] =
          2.0 + 1.0 / std::max(1, ANIMAL_INTERVAL[animal]);
      context.first_cash_lag_days[project] = int16_t(ANIMAL_FIRST[animal]);
    }

    // Use the established full-plan scorer only to derive eight local
    // marginal screens.  The eventual labels still come from complete C++
    // continuations with common random numbers.
    const double baseline_score = score_plan_candidate(
        env, player, plan, nullptr, &state);
    for (int project = 0; project < ADAPTIVE_PROJECTS; ++project) {
      AdaptivePlan expanded = plan;
      if (project < N_CROPS) {
        expanded.crop_targets[project]++;
      } else {
        const int animal = project - N_CROPS;
        expanded.animal_targets[animal]++;
        expanded.animal_service_targets[animal]++;
      }
      const double expanded_score = score_plan_candidate(
          env, player, expanded, nullptr, &state);
      context.marginal_value[project] = expanded_score - baseline_score +
          context.purchase_cost[project];
    }

    context.liquid_cash = farm.money;
    context.protected_cash = genome_.cash_reserve;
    for (int item = 0; item < N_PRODUCTS; ++item)
      context.financeable_inventory_value +=
          priv.shed[item] * market.prices[item];
    context.unlocked_quadrants = popcount(farm.unlocked_mask);
    context.maximum_quadrants = 4;
    context.hands = int(farm.hands.size());
    context.maximum_hands = genome_.max_hands;
    context.next_hand_cost = std::max(
        0, daily_hire_cost(farm.hires_today + 1) -
               daily_hire_cost(farm.hires_today));
    context.next_quadrant_cost = context.unlocked_quadrants <= 3
        ? LAND_COST[context.unlocked_quadrants - 1] : 0.0;
    context.tiles_per_quadrant = 25;
    for (int crop = 0; crop < N_CROPS; ++crop)
      context.productive_tiles += std::max(
          int(plan.crop_targets[crop]), own.field_crops[crop]);
    for (int animal = 0; animal < N_ANIMALS; ++animal)
      context.productive_tiles += std::max(
          int(plan.animal_targets[animal]), own.field_animals[animal]);
    context.current_daily_action_load = workload_for(plan);
    context.hard_deadline_load = own.hard_water + own.hard_feed;
    context.estimated_travel_load =
        own.weeds + own.ready_harvest + own.hard_water + own.hard_feed;
    context.delayed_loss = 100.0 * (own.hard_water + own.hard_feed) +
        25.0 * own.ready_harvest;
    context.market_slots_available = env.config().max_market_orders;
    for (int item = 0; item < N_PRODUCTS; ++item) {
      context.sellable_inventory[item] = int16_t(std::clamp(
          int(priv.shed[item]), 0, int(std::numeric_limits<int16_t>::max())));
      context.market_prices[item] = int16_t(std::clamp(
          int(market.prices[item]), 0, int(std::numeric_limits<int16_t>::max())));
      context.demand_within_day[item] = int16_t(std::clamp(
          int(std::nearbyint(demand[item])), 0,
          int(std::numeric_limits<int16_t>::max())));
    }
    if (own.weeds > 0) context.recovery_issues |= RECOVERY_WEED;
    const auto& shortfalls = env.last_market_cash_shortfalls()[player];
    if (std::any_of(shortfalls.begin(), shortfalls.end(),
                    [](double value) { return value > 0.0; }))
      context.recovery_issues |= RECOVERY_PARTIAL_FILL;
    if ((own.hard_water + own.hard_feed) > 0)
      context.recovery_issues |= RECOVERY_BROKEN_CHAIN;
    if (!state.sticky_target.empty() && own.ready_harvest > 0)
      context.recovery_issues |= RECOVERY_UNIT_MISALIGNMENT;

    // Freeze the exact visible/self context used to generate this candidate
    // set.  Complete continuation labels vary substantially across seeds even
    // for an identical PlanDelta, so candidate metadata alone is an
    // under-specified learning problem.  Values derived from money or workload
    // use x100 fixed-point encoding; counts remain in official units.
    auto& context_features = plan.candidate8_context_features;
    int context_feature = 0;
    auto append_context_int = [&](int value) {
      context_features[context_feature++] = int32_t(value);
    };
    auto append_context_double = [&](double value) {
      context_features[context_feature++] =
          int32_t(std::nearbyint(100.0 * value));
    };
    append_context_int(context.day);
    for (const int value : context.targets) append_context_int(value);
    for (const int value : context.irreversible_floor) append_context_int(value);
    for (const int value : context.caps) append_context_int(value);
    for (const double value : context.marginal_value)
      append_context_double(value);
    for (const double value : context.purchase_cost)
      append_context_double(value);
    for (const double value : context.daily_action_load)
      append_context_double(value);
    for (const int value : context.first_cash_lag_days)
      append_context_int(value);
    append_context_double(context.liquid_cash);
    append_context_double(context.protected_cash);
    append_context_double(context.financeable_inventory_value);
    append_context_int(context.unlocked_quadrants);
    append_context_int(context.maximum_quadrants);
    append_context_int(context.hands);
    append_context_int(context.maximum_hands);
    append_context_double(context.next_hand_cost);
    append_context_double(context.next_quadrant_cost);
    append_context_int(context.tiles_per_quadrant);
    append_context_int(context.productive_tiles);
    append_context_double(context.current_daily_action_load);
    append_context_double(context.hard_deadline_load);
    append_context_double(context.estimated_travel_load);
    append_context_double(context.delayed_loss);
    append_context_int(context.market_slots_available);
    append_context_int(context.recovery_issues);
    for (const int value : context.sellable_inventory)
      append_context_int(value);
    for (const int value : context.market_prices) append_context_int(value);
    for (const int value : context.demand_within_day)
      append_context_int(value);
    // Public opponent state already maintained by the planner.  These fields
    // make competitive labels learnable without leaking a route id or any
    // private shed/carried inventory.
    for (const int value : state.opponent_last_project_counts)
      append_context_int(value);
    for (const int value : state.opponent_last_project_yield)
      append_context_int(value);
    for (const int value : state.opponent_project_count_trend_x100)
      append_context_int(value);
    for (const int value : state.opponent_project_yield_trend_x100)
      append_context_int(value);
    append_context_int(state.opponent_last_cash);
    append_context_int(state.opponent_cash_trend);
    append_context_int(state.opponent_last_hands);
    append_context_int(state.opponent_last_quadrants);
    append_context_int(state.opponent_hands_trend_x100);
    append_context_int(state.opponent_quadrants_trend_x100);

    // O1.1 deployable opponent-intent context.  Only public board, workforce,
    // cash and shared-market history are used.  In particular, the opponent's
    // shed, seeds and carried inventory are never read here.
    for (const int value : state.opponent_recent_project_count_delta)
      append_context_int(value);
    for (const int value : state.opponent_recent_project_yield_delta)
      append_context_int(value);
    append_context_int(state.opponent_recent_cash_delta);
    append_context_int(state.opponent_recent_hands_delta);
    append_context_int(state.opponent_recent_quadrants_delta);
    for (const double value : state.observed_daily_drift)
      append_context_double(value);

    const auto& opponent_farm = env.farms()[1 - player];
    std::vector<Position> opponent_units;
    opponent_units.reserve(1 + opponent_farm.hands.size());
    opponent_units.push_back(opponent_farm.farmer);
    opponent_units.insert(opponent_units.end(), opponent_farm.hands.begin(),
                          opponent_farm.hands.end());
    std::array<int16_t, N_CROPS + N_ANIMALS>
        opponent_min_unit_distance{};
    std::array<int16_t, N_CROPS + N_ANIMALS>
        opponent_estimated_sale_lead_steps{};
    opponent_min_unit_distance.fill(int16_t(999));
    opponent_estimated_sale_lead_steps.fill(int16_t(999));
    auto nearest_shed_distance = [&](Position pos) {
      int best = 100;
      for (const Position shed : SHED_POS)
        best = std::min(best, distance(pos, shed));
      return best;
    };
    const int board = env.config().board_size;
    for (int cell = 0; cell < int(opponent_farm.tiles.size()); ++cell) {
      const auto& tile = opponent_farm.tiles[cell];
      int project = -1;
      int wait_days = 0;
      if (tile.kind == TileKind::PLANT) {
        const int crop = int(tile.crop);
        if (crop < 0 || crop >= N_CROPS) continue;
        project = crop;
        if (tile.yield_units <= 0) {
          const int age = context.day - int(tile.planted_day);
          wait_days = std::max(0, CROP_FIRST[crop] - age);
        }
      } else if (tile.kind == TileKind::ANIMAL) {
        const int animal = int(tile.animal) - 9;
        if (animal < 0 || animal >= N_ANIMALS) continue;
        project = N_CROPS + animal;
        if (tile.yield_units <= 0) {
          const int age = context.day - int(tile.placed_day);
          wait_days = std::max(0, ANIMAL_FIRST[animal] - age);
        }
      } else {
        continue;
      }
      const Position pos{int16_t(cell % board), int16_t(cell / board)};
      int unit_distance = 100;
      for (const Position unit : opponent_units)
        unit_distance = std::min(unit_distance, distance(unit, pos));
      opponent_min_unit_distance[project] = int16_t(std::min(
          int(opponent_min_unit_distance[project]), unit_distance));
      // +2 represents collect/harvest and shed deposit actions.  This is a
      // public logistics estimate, not a claim that the opponent must sell.
      const int sale_lead = wait_days * env.config().turns_per_day +
          unit_distance + nearest_shed_distance(pos) + 2;
      opponent_estimated_sale_lead_steps[project] = int16_t(std::min(
          int(opponent_estimated_sale_lead_steps[project]), sale_lead));
    }
    for (const int value : opponent_min_unit_distance)
      append_context_int(value);
    for (const int value : opponent_estimated_sale_lead_steps)
      append_context_int(value);

    // The same public commitment projection already audited for the richer
    // SWITCH feature set.  It estimates supply, not private inventory and not
    // an opponent action.  Candidate8 receives only the opponent projection.
    constexpr int opponent_projection_lags[3] = {2, 4, 8};
    for (const int lag : opponent_projection_lags)
      for (int item = 0; item < N_PRODUCTS; ++item)
        append_context_double(
            committed_supply_until(1 - player, item, lag));
    assert(context_feature == AdaptivePlan::CANDIDATE8_CONTEXT_FEATURE_DIM);

    const AdaptiveCandidateSet candidates = generate_adaptive_candidates(
        context, 64, 1000);
    plan.candidate8_raw_count = int16_t(std::min(
        candidates.raw_count, int(std::numeric_limits<int16_t>::max())));
    plan.candidate8_feasible_count = int16_t(std::min(
        candidates.feasible_count, int(std::numeric_limits<int16_t>::max())));
    plan.candidate8_shortlist_count = int16_t(candidates.shortlist.size());
    const int rank = genome_.offline_candidate8_rank;
    const auto& candidate_pool = genome_.offline_candidate8_use_feasible_pool
        ? candidates.feasible : candidates.shortlist;
    if (rank >= 0 && rank < int(candidate_pool.size())) {
      const auto& delta = candidate_pool[rank];
      for (int crop = 0; crop < N_CROPS; ++crop)
        plan.crop_targets[crop] = int16_t(
            int(context.targets[crop]) + delta.target_delta[crop]);
      for (int animal = 0; animal < N_ANIMALS; ++animal) {
        const int project = N_CROPS + animal;
        plan.animal_targets[animal] = int16_t(
            int(context.targets[project]) + delta.target_delta[project]);
        plan.animal_service_targets[animal] = int16_t(std::max(
            own.field_animals[animal],
            int(plan.animal_service_targets[animal]) +
                int(delta.target_delta[project])));
      }
      plan.candidate8_family = int8_t(delta.family);
      plan.candidate8_schedule_profile = delta.schedule_profile;
      plan.candidate8_market_profile = delta.market_profile;
      plan.candidate8_recovery_profile = delta.recovery_profile;
      plan.candidate8_market_item = delta.market_item;
      plan.candidate8_recovery_issue = delta.recovery_issue;
      plan.candidate8_target_delta = delta.target_delta;
      plan.candidate8_hand_delta = delta.hand_delta;
      plan.candidate8_quadrant_delta = delta.quadrant_delta;
      plan.candidate8_effective_delay_days = delta.effective_delay_days;
      plan.candidate8_suffix_project = delta.suffix_project;
      plan.candidate8_estimated_value = delta.estimated_value;
      plan.candidate8_estimated_cash_cost = delta.estimated_cash_cost;
      plan.candidate8_estimated_daily_action_load =
          delta.estimated_daily_action_load;
      plan.candidate8_signature = delta.signature;
      plan.quadrant_target = int16_t(std::min(
          4,
          int(plan.quadrant_target) + std::max(0, int(delta.quadrant_delta))));
    }
  }

  double daily_actions = workload_for(plan);
  daily_actions += 0.5 * (own.weeds + own.ready_harvest) + 3.0 * (plan.quadrant_target - 1);
  int hand_target = std::clamp(
      int(std::ceil(daily_actions / unit_day_capacity)) - 1,
      0, genome_.max_hands);
  if (backbone.enabled)
    // FOLLOW owns the demonstrated workforce envelope.  Recomputing it from
    // the generic workload heuristic hired 2-3 unnecessary hands in the
    // opening and spent the cash needed for the next land/production jump.
    // Later project edits may explicitly change this calendar; the base
    // executor must not silently inflate it.
    hand_target = backbone.hand_target[std::clamp(day, 0, TOTAL_DAYS - 1)];
  hand_target = std::max(
      hand_target, int(farm.hands.size()) +
          std::max(0, int(plan.candidate8_hand_delta)));
  if (state.candidate8_persistent_hand_target >= 0)
    hand_target = std::max(
        hand_target, int(state.candidate8_persistent_hand_target));
  plan.hand_target = int16_t(std::clamp(hand_target, 0, genome_.max_hands));
  if (!backbone.enabled && genome_.autonomous_project_state_machine > 0.0) {
    bool new_crop_family = false;
    for (int crop = 0; crop < N_CROPS; ++crop)
      new_crop_family = new_crop_family ||
          (previous_crop_desired[crop] <= 0 && plan.crop_targets[crop] > 0);
    for (int animal = 0; animal < N_ANIMALS; ++animal) {
      const int selected = plan.animal_targets[animal];
      const int previous = previous_animal_desired[animal];
      const int owned = own.total_animals[animal];
      if (selected > previous) {
        plan.animal_project_modes[animal] = AdaptivePlan::SCALE;
        plan.deferred_animal_targets[animal] = int16_t(selected);
      } else if (selected >= previous && selected > owned) {
        plan.animal_project_modes[animal] = AdaptivePlan::KEEP;
        plan.deferred_animal_targets[animal] = int16_t(selected);
      } else if (previous > selected) {
        if (previous_animal_positive[animal]) {
          plan.animal_project_modes[animal] = AdaptivePlan::DEFER;
          plan.deferred_animal_targets[animal] = int16_t(previous);
        } else {
          plan.animal_project_modes[animal] = AdaptivePlan::CANCEL;
          plan.deferred_animal_targets[animal] = int16_t(selected);
        }
      } else if (owned > 0) {
        // Purchased animals are irreversible.  KEEP here means maintain the
        // real asset, not preserve an unfilled purchase request.
        plan.animal_project_modes[animal] = AdaptivePlan::KEEP;
        plan.deferred_animal_targets[animal] = int16_t(selected);
      }
    }
    for (int crop = 0; crop < N_CROPS; ++crop) {
      const int selected = plan.crop_targets[crop];
      const int previous = previous_crop_desired[crop];
      if (selected > previous) {
        plan.crop_project_modes[crop] = AdaptivePlan::SCALE;
        plan.deferred_crop_targets[crop] = int16_t(selected);
      } else if (selected == previous && selected > 0) {
        plan.crop_project_modes[crop] = AdaptivePlan::KEEP;
        plan.deferred_crop_targets[crop] = int16_t(selected);
      } else if (previous > selected) {
        if (new_crop_family) {
          plan.crop_project_modes[crop] = AdaptivePlan::SWITCH;
          plan.deferred_crop_targets[crop] = int16_t(selected);
        } else if (previous_crop_positive[crop]) {
          plan.crop_project_modes[crop] = AdaptivePlan::DEFER;
          plan.deferred_crop_targets[crop] = int16_t(previous);
        } else {
          plan.crop_project_modes[crop] = AdaptivePlan::CANCEL;
          plan.deferred_crop_targets[crop] = int16_t(selected);
        }
      }
    }
    const int unlocked = popcount(farm.unlocked_mask);
    const int selected_quadrants = plan.quadrant_target;
    if (selected_quadrants > previous_quadrant_desired) {
      plan.quadrant_project_mode = AdaptivePlan::SCALE;
      plan.deferred_quadrant_target = int16_t(selected_quadrants);
    } else if (selected_quadrants >= previous_quadrant_desired &&
               selected_quadrants > unlocked) {
      plan.quadrant_project_mode = AdaptivePlan::KEEP;
      plan.deferred_quadrant_target = int16_t(selected_quadrants);
    } else if (previous_quadrant_desired > selected_quadrants) {
      int projected_tiles = 0;
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        projected_tiles += plan.animal_targets[animal];
      for (int crop = 0; crop < N_CROPS; ++crop)
        projected_tiles += plan.crop_targets[crop];
      int required_quadrants = 1;
      for (; required_quadrants < genome_.max_quadrants;
           ++required_quadrants) {
        const int trigger = genome_.land_capacity_trigger_fraction > 0.0
            ? std::max(1, int(std::floor(25 * required_quadrants *
                                         genome_.land_capacity_trigger_fraction)))
            : 25 * required_quadrants - 2;
        if (projected_tiles <= trigger) break;
      }
      const bool still_needed =
          day <= std::max(genome_.stop_new_animals_day,
                          latest_generic_crop_start(genome_)) &&
          required_quadrants > selected_quadrants &&
          previous_quadrant_desired >= required_quadrants;
      if (still_needed) {
        plan.quadrant_project_mode = AdaptivePlan::DEFER;
        plan.deferred_quadrant_target = int16_t(previous_quadrant_desired);
      } else {
        plan.quadrant_project_mode = AdaptivePlan::CANCEL;
        plan.deferred_quadrant_target = int16_t(selected_quadrants);
      }
    } else if (unlocked > 1) {
      plan.quadrant_project_mode = AdaptivePlan::KEEP;
      plan.deferred_quadrant_target = int16_t(selected_quadrants);
    }
  }
  return plan;
}

double NativeAdaptivePlanner::score_plan_candidate(
    const Simulator& env, int player, const AdaptivePlan& plan,
    PlanValueBreakdown* breakdown,
    const AdaptivePlannerState* planner_state) const {
  PlanValueBreakdown parts;
  const int day = env.day();
  const int days_left = std::max(1, TOTAL_DAYS - day);
  const auto own = counts_for(env, player);
  const auto opponent_supply = visible_daily_supply(env, 1 - player);
  const auto own_supply = visible_daily_supply(env, player);
  const auto demand = shop_demand(env);
  const auto& market = env.market();
  const auto& farm = env.farms()[player];
  const auto& priv = env.privates()[player];

  std::array<double, N_PRODUCTS> candidate_supply{};
  // A target is a maintained production slot, not one isolated seed.  Walk
  // every lifecycle that can still cash out before the terminal day.  The old
  // whole-plan scorer treated max_yield as lifetime output and therefore
  // valued a fertilized strawberry lane at roughly half its official-rule
  // production while valuing a new animal through the whole remaining game.
  std::array<double, N_CROPS> crop_units{};
  std::array<double, N_CROPS> crop_seed_uses{};
  std::array<double, N_CROPS> crop_actions{};
  std::array<double, N_CROPS> crop_fertilizer_potential{};
  struct AnimalProjection {
    double product_units = 0.0;
    double product_lag_units = 0.0;
    double service_days = 0.0;
    double fertilizer_units = 0.0;
    double service_actions = 0.0;
    int new_purchases = 0;
  };
  std::array<AnimalProjection, N_ANIMALS> animal_projection{};
  for (int crop = 0; crop < N_CROPS; ++crop) {
    const int target = std::max(0, int(plan.crop_targets[crop]));
    if (target <= 0) continue;
    int start = day;
    while (start + CROP_FIRST[crop] < TOTAL_DAYS) {
      if (CROP_ONGOING[crop]) {
        const int interval = std::max(1, CROP_INTERVAL[crop]);
        const int production_events = std::min(
            CROP_MAX_YIELD[crop],
            1 + std::max(0, (TOTAL_DAYS - 1 - start -
                                  CROP_FIRST[crop]) / interval));
        if (production_events <= 0) break;
        crop_units[crop] += production_events;
        crop_fertilizer_potential[crop] += production_events;
        crop_seed_uses[crop] += 1.0;
        const int last_production_age = CROP_FIRST[crop] +
            (production_events - 1) * interval;
        const bool completes_lifecycle =
            production_events == CROP_MAX_YIELD[crop];
        const int service_days = completes_lifecycle
            ? last_production_age + 1
            : TOTAL_DAYS - start;
        crop_actions[crop] += 1.0 + std::ceil(service_days / 2.0) +
                              production_events + 1.0;
        if (!completes_lifecycle) break;
        start += last_production_age + 1;
      } else {
        const int harvest_age = std::min(
            CROP_MAX_DAY[crop], TOTAL_DAYS - 1 - start);
        if (harvest_age < CROP_FIRST[crop]) break;
        const int water_start = (CROP_MAX_DAY[crop] + 1) / 2;
        const int yield_waterings = std::max(
            0, harvest_age - water_start + 1);
        const int base_units = std::min(
            CROP_MAX_YIELD[crop], 1 + yield_waterings);
        const int fertilized_units = std::min(
            CROP_MAX_YIELD[crop], 1 + 2 * yield_waterings);
        crop_units[crop] += base_units;
        crop_fertilizer_potential[crop] +=
            std::max(0, fertilized_units - base_units);
        crop_seed_uses[crop] += 1.0;
        crop_actions[crop] += 1.0 + std::ceil((harvest_age + 1) / 2.0) +
                              2.0;
        if (harvest_age < CROP_MAX_DAY[crop]) break;
        start += std::max(CROP_FIRST[crop], harvest_age);
      }
    }
    crop_units[crop] *= target;
    crop_seed_uses[crop] *= target;
    crop_actions[crop] *= target;
    crop_fertilizer_potential[crop] *= target;
  }

  // Project animals from their *physical* state.  Existing field animals keep
  // their real age, held product and CARE bonus.  Animals in the shed, carried
  // by a unit, or not bought yet must first pass through the official
  // structure/pickup/placement pipeline.  The previous scorer treated every
  // target animal as if it were already placed today and immediately received
  // a full remaining-season production horizon.  That was the dominant source
  // of false strawberry -> animal switches in the W6 counterfactual oracle.
  auto add_animal_lifecycle = [&](int animal, int placed_day,
                                  int pending_care_bonus,
                                  int current_yield,
                                  int service_start_day,
                                  AnimalProjection& projection) {
    if (service_start_day >= TOTAL_DAYS - 1) return;
    if (current_yield > 0) {
      projection.product_units += current_yield;
      // Already-held output can be harvested and sold without waiting for a
      // new biological production event.
      projection.service_actions += 1.0;
    }
    int pending = std::max(0, pending_care_bonus);
    for (int service_day = service_start_day;
         service_day < TOTAL_DAYS - 1; ++service_day) {
      projection.service_days += 1.0;
      const int next_day = service_day + 1;
      const int delta = next_day - placed_day - ANIMAL_FIRST[animal];
      if (delta >= 0 &&
          delta % std::max(1, ANIMAL_INTERVAL[animal]) == 0) {
        // The official engine applies accumulated CARE before clamping the
        // newly available product to the animal-specific carry capacity.
        const int produced = std::min(
            ANIMAL_MAX_HELD[animal], 1 + pending);
        projection.product_units += produced;
        projection.product_lag_units +=
            produced * std::max(0, next_day - day);
        projection.service_actions += 1.0;  // HARVEST
        pending = 0;
      }
      // FEED and CARE are assumed executable by a feasible candidate.  CARE
      // performed on a production evening applies to the following cycle,
      // matching Simulator::end_of_day ordering.
      pending += 1;
    }
  };

  for (const auto& tile : farm.tiles) {
    if (tile.kind != TileKind::ANIMAL) continue;
    const int animal = int(tile.animal) - 9;
    if (animal < 0 || animal >= N_ANIMALS) continue;
    auto& projection = animal_projection[animal];
    add_animal_lifecycle(animal, tile.placed_day,
                         tile.pending_care_bonus, tile.yield_units,
                         day, projection);
    // A currently available fertilizer unit is distinct from the units that
    // can be collected after future end-of-day settlements.
    if (tile.fertilizer_available)
      projection.fertilizer_units += genome_.fertilizer_value_fraction;
  }

  const int new_goose_placements = std::max(
      0, int(plan.animal_service_targets[0]) - own.field_animals[0]);
  const int new_pasture_placements =
      std::max(0, int(plan.animal_service_targets[1]) -
                      own.field_animals[1]) +
      std::max(0, int(plan.animal_service_targets[2]) -
                      own.field_animals[2]);

  struct StructureSetup {
    int builds = 0;
    int digs = 0;
    int missing = 0;
    int max_distance = 0;
    double placement_distance = 0.0;
    double build_distance = 0.0;
  };
  auto structure_setup = [&](TileKind kind, int needed) {
    StructureSetup setup;
    if (needed <= 0) return setup;
    struct Slot {
      int distance = 0;
      bool build = false;
      bool dig = false;
    };
    std::vector<Slot> slots;
    auto shed_distance = [&](Position pos) {
      int best = 2 * env.config().board_size;
      for (const auto shed : SHED_POS)
        best = std::min(best, distance(pos, shed));
      return best;
    };
    const int board = env.config().board_size;
    for (int cell = 0; cell < int(farm.tiles.size()); ++cell) {
      const auto& tile = farm.tiles[cell];
      const Position pos{int16_t(cell % board), int16_t(cell / board)};
      if (tile.kind == kind && tile.animal == Item::NONE) {
        slots.push_back({shed_distance(pos), false, false});
      } else if (unlocked(farm, pos) &&
                 (tile.kind == TileKind::EMPTY ||
                  tile.kind == TileKind::WEED)) {
        slots.push_back({shed_distance(pos), true,
                         tile.kind == TileKind::WEED});
      }
    }
    std::sort(slots.begin(), slots.end(), [](const Slot& lhs,
                                             const Slot& rhs) {
      // The executor reuses an existing matching structure before creating a
      // new one.  Keep the scorer's setup path identical even when a fresh
      // empty tile happens to be closer to the shed.
      if (lhs.build != rhs.build) return lhs.build < rhs.build;
      const int lhs_effort = lhs.distance + (lhs.build ? 1 : 0) +
          (lhs.dig ? 1 : 0);
      const int rhs_effort = rhs.distance + (rhs.build ? 1 : 0) +
          (rhs.dig ? 1 : 0);
      return lhs_effort != rhs_effort
          ? lhs_effort < rhs_effort
          : lhs.distance < rhs.distance;
    });
    const int selected = std::min<int>(needed, slots.size());
    for (int i = 0; i < selected; ++i) {
      const auto& slot = slots[i];
      setup.builds += slot.build;
      setup.digs += slot.dig;
      setup.max_distance = std::max(setup.max_distance, slot.distance);
      setup.placement_distance += slot.distance;
      if (slot.build) setup.build_distance += slot.distance;
    }
    // A plan without enough legal structure cells is not truly executable.
    // Assign a large finite setup burden so it loses analytically while the
    // normal feasibility gate remains the authoritative legality check.
    if (selected < needed) {
      const int missing = needed - selected;
      setup.missing = missing;
      setup.builds += missing;
      setup.max_distance = std::max(
          setup.max_distance, 2 * env.config().board_size);
      setup.placement_distance +=
          missing * 2.0 * env.config().board_size;
      setup.build_distance +=
          missing * 2.0 * env.config().board_size;
    }
    return setup;
  };

  const StructureSetup coop_setup =
      structure_setup(TileKind::COOP, new_goose_placements);
  const StructureSetup pasture_setup =
      structure_setup(TileKind::PASTURE, new_pasture_placements);
  int carried_animals = 0;
  for (const auto& inventory : priv.inventories)
    for (int animal = 0; animal < N_ANIMALS; ++animal)
      carried_animals += inventory[9 + animal];
  const int total_new_placements =
      new_goose_placements + new_pasture_placements;
  const int pickup_actions = std::max(0,
      total_new_placements - carried_animals);
  const double setup_actions = pickup_actions + total_new_placements +
      coop_setup.builds + coop_setup.digs +
      pasture_setup.builds + pasture_setup.digs;
  const double setup_moves =
      coop_setup.placement_distance + coop_setup.build_distance +
      pasture_setup.placement_distance + pasture_setup.build_distance;
  // Derive the workforce from the complete portfolio for *both* KEEP and
  // SWITCH.  build_plan fills the ordinary plan.hand_target after bundle
  // scoring, while normalised SWITCH arms already carry their target.  Reading
  // that field directly therefore compared a one-worker KEEP setup with a
  // multi-worker SWITCH even when both feature rows reported the same workload.
  double planned_workload = 4.0;
  for (int animal = 0; animal < N_ANIMALS; ++animal)
    planned_workload += 3.1 * std::max(
        0, int(plan.animal_service_targets[animal]));
  for (int crop = 0; crop < N_CROPS; ++crop)
    planned_workload += (CROP_ONGOING[crop] ? 1.0 : 1.15) *
        std::max<int>(plan.crop_targets[crop], own.field_crops[crop]);
  const double projected_unit_day_capacity =
      12.0 / std::max(0.5, genome_.risk_multiplier);
  const int workload_hands = std::max(
      0, int(std::ceil(planned_workload / projected_unit_day_capacity)) - 1);
  const int planned_units = std::max(
      1, std::max(int(farm.hands.size()) + 1, workload_hands + 1));
  const int parallel_setup_turns = int(std::ceil(
      (setup_actions + setup_moves) / planned_units));
  const int critical_setup_turns = total_new_placements > 0
      ? 2 + std::max(coop_setup.max_distance,
                     pasture_setup.max_distance) +
          std::max(coop_setup.builds + coop_setup.digs > 0 ? 1 : 0,
                   pasture_setup.builds + pasture_setup.digs > 0 ? 1 : 0)
      : 0;
  const int setup_turns = std::max(parallel_setup_turns,
                                   critical_setup_turns);
  // Two extra turns reserve FEED/CARE capacity on the placement day.  Without
  // this guard an animal placed at hour 23 was scored as if it had received a
  // complete first service day.
  const int setup_delay_days = total_new_placements > 0
      ? std::max(0, (env.hour() + setup_turns + 2) /
                        env.config().turns_per_day)
      : 0;
  const int projected_placement_day = day + setup_delay_days;

  for (int animal = 0; animal < N_ANIMALS; ++animal) {
    auto& projection = animal_projection[animal];
    const int service_target = std::max(
        0, int(plan.animal_service_targets[animal]));
    const int placements = std::max(
        0, service_target - own.field_animals[animal]);
    projection.new_purchases = std::max(
        0, int(plan.animal_targets[animal]) - own.total_animals[animal]);
    for (int placed = 0; placed < placements; ++placed)
      add_animal_lifecycle(animal, projected_placement_day, 0, 0,
                           projected_placement_day, projection);
    // Each serviced animal may expose at most one fertilizer collection per
    // completed service day.  Realisation is discounted by the existing
    // searchable collection fraction, and the matching collection actions are
    // charged below.
    projection.fertilizer_units += projection.service_days *
        genome_.fertilizer_value_fraction;
    projection.service_actions += 2.0 * projection.service_days +
        projection.fertilizer_units;
    candidate_supply[ANIMAL_PRODUCT[animal]] +=
        projection.product_units;
    candidate_supply[0] -= projection.service_days;
  }

  // Fertilizer is a shared project resource.  A unit can either be consumed by
  // a crop or sold; it cannot earn both rewards.  Allocate it only when the
  // crop's current marginal quote exceeds the fertilizer quote plus the
  // FERTILIZE action cost.  This is deliberately conservative until the W3
  // oracle supports a more detailed timing model.
  double fertilizer_budget = priv.shed[8];
  for (const auto& inventory : priv.inventories)
    fertilizer_budget += inventory[8];
  for (const auto& projection : animal_projection)
    fertilizer_budget += projection.fertilizer_units;
  std::array<int, N_CROPS> fertilizer_order{0, 1, 2, 3, 4};
  std::sort(fertilizer_order.begin(), fertilizer_order.end(),
            [&](int lhs, int rhs) {
              const double lhs_gain = market.prices[lhs] - market.prices[8];
              const double rhs_gain = market.prices[rhs] - market.prices[8];
              return lhs_gain > rhs_gain;
            });
  double fertilizer_used = 0.0;
  for (const int crop : fertilizer_order) {
    if (market.prices[crop] <=
        market.prices[8] + 0.25 * genome_.action_cost) continue;
    const double used = std::min(
        std::max(0.0, fertilizer_budget - fertilizer_used),
        crop_fertilizer_potential[crop]);
    crop_units[crop] += used;
    crop_actions[crop] += used;
    fertilizer_used += used;
  }
  for (int crop = 0; crop < N_CROPS; ++crop)
    candidate_supply[crop] += crop_units[crop];
  const double sellable_fertilizer = std::max(
      0.0, fertilizer_budget - fertilizer_used);
  candidate_supply[8] += sellable_fertilizer;

  auto future_price = [&](int item, int lag) {
    const double public_demand = demand[item] * lag;
    double expected_future_shop_demand = 0.0;
    if (genome_.future_shop_expectation_weight > 0.0 &&
        item >= 0 && item < N_PRODUCTS && lag > 0) {
      double mean_shop_units_per_tick = 0.0;
      for (int shop = 0; shop < 8; ++shop)
        mean_shop_units_per_tick += SHOP_DEMAND[shop][item];
      mean_shop_units_per_tick /= 8.0;
      const int unlock_interval = std::max(
          1, env.config().town_shop_unlock_interval);
      const int sell_interval = std::max(
          1, env.config().town_shop_sell_interval);
      const double ticks_per_day =
          double(env.config().turns_per_day) / sell_interval;
      int remaining_unlocks = std::max(
          0, 8 - int(env.shops().size()));
      const int horizon_day = day + lag;
      for (int future_day = day + 1;
           future_day <= horizon_day && remaining_unlocks > 0;
           ++future_day) {
        if (future_day % unlock_interval != 0) continue;
        expected_future_shop_demand +=
            std::max(0, horizon_day - future_day) * ticks_per_day *
            mean_shop_units_per_tick;
        --remaining_unlocks;
      }
      expected_future_shop_demand *=
          genome_.future_shop_expectation_weight;
    }
    const double adverse_supply = genome_.opponent_supply_weight *
        opponent_supply[item] * lag;
    const double planned_impact = genome_.portfolio_supply_impact_weight > 0.0
        ? genome_.portfolio_supply_impact_weight
        : genome_.market_impact_weight;
    const double self_supply =
        genome_.market_impact_weight * own_supply[item] * lag +
        planned_impact * 0.5 * candidate_supply[item];
    const int inventory = int(std::nearbyint(
        market.inventory[item] - public_demand - expected_future_shop_demand +
        adverse_supply + self_supply));
    return predicted_price(item, std::max(1, inventory));
  };

  double value = 0.0;
  double workload = 4.0;
  for (int crop = 0; crop < N_CROPS; ++crop) {
    const int target = std::max(0, int(plan.crop_targets[crop]));
    if (target <= 0 || crop_units[crop] <= 0.0) continue;
    const int lag = std::max(CROP_FIRST[crop], CROP_MAX_DAY[crop] / 2);
    const double gross = crop_units[crop] * future_price(crop, lag);
    const double seed_purchases = std::max(
        0.0, crop_seed_uses[crop] -
                 std::min<double>(target, own.committed_crops[crop]));
    const double seed_cost = seed_purchases * SEED_COST[crop];
    const double action_cost =
        0.25 * genome_.action_cost * crop_actions[crop];
    const double move_cost = 0.10 * genome_.move_cost * target *
        std::max(2.0, crop_actions[crop] / std::max(1, target) / 2.0);
    value += gross - seed_cost - action_cost - move_cost;
    parts.crop_gross += gross;
    parts.seed_cost += seed_cost;
    parts.action_cost += action_cost;
    parts.move_cost += move_cost;
    parts.crop_units += crop_units[crop];
    workload += (CROP_ONGOING[crop] ? 1.0 : 1.15) * target;
  }
  for (int animal = 0; animal < N_ANIMALS; ++animal) {
    const int target = std::max(0, int(plan.animal_service_targets[animal]));
    const auto& projection = animal_projection[animal];
    if (target <= 0) continue;
    const int item = ANIMAL_PRODUCT[animal];
    const int lag = projection.product_units > 0.0
        ? std::max(1, int(std::nearbyint(
              projection.product_lag_units / projection.product_units)))
        : std::max(1, ANIMAL_FIRST[animal] + setup_delay_days);
    const double gross = projection.product_units * future_price(item, lag);
    const double feed = projection.service_days *
        future_price(0, std::max(1, days_left / 2));
    const double purchase_cost =
        projection.new_purchases * ANIMAL_COST[animal];
    const double action_cost =
        0.25 * genome_.action_cost * projection.service_actions;
    const double move_cost =
        0.10 * genome_.move_cost * 2.0 * projection.service_days;
    value += gross - feed - purchase_cost - action_cost - move_cost;
    parts.animal_gross += gross;
    parts.feed_cost += feed;
    parts.animal_purchase_cost += purchase_cost;
    parts.action_cost += action_cost;
    parts.move_cost += move_cost;
    parts.animal_product_units += projection.product_units;
    workload += 3.1 * target;
  }
  // Fertilizer not consumed by crops is valued once, after its own projected
  // market impact.  Setup logistics are also a single shared cost rather than
  // being replicated for every animal family.
  if (sellable_fertilizer > 0.0) {
    const double fertilizer_gross = sellable_fertilizer *
        future_price(8, std::max(1, days_left / 2));
    value += fertilizer_gross;
    parts.fertilizer_gross += fertilizer_gross;
  }
  const double setup_action_cost =
      0.25 * genome_.action_cost * setup_actions;
  const double setup_move_cost = 0.10 * genome_.move_cost * setup_moves;
  value -= setup_action_cost + setup_move_cost;
  parts.action_cost += setup_action_cost;
  parts.move_cost += setup_move_cost;
  parts.fertilizer_used = fertilizer_used;
  parts.fertilizer_sellable = sellable_fertilizer;
  parts.setup_turns = setup_turns;

  const int required_hands = std::max(
      int(plan.hand_target),
      std::max(0, int(std::ceil(
          workload / (12.0 / std::max(0.5, genome_.risk_multiplier)))) - 1));
  const double hire_cost =
      std::max(0, days_left - 1) * daily_hire_cost(required_hands);
  value -= hire_cost;
  parts.hire_cost = hire_cost;
  int quadrants = popcount(farm.unlocked_mask);
  for (int q = quadrants; q < int(plan.quadrant_target) && q <= 3; ++q) {
    value -= LAND_COST[q - 1];
    parts.land_cost += LAND_COST[q - 1];
  }
  // Prefer plans with better cash conversion when projected values are close.
  parts.lockup_cost = genome_.capital_lockup_weight *
      std::max(0, int(plan.quadrant_target) - quadrants) * 500.0;
  value -= parts.lockup_cost;

  // A total-season NPV cannot distinguish a profitable project that fits the
  // available work/cash windows from the same nominal project at an
  // unserviceable scale.  Expose conservative 2/4/8-day projections so the
  // learned calibrator can rank size without learning a replay calendar.
  const int projection_horizons[3] = {2, 4, 8};
  int new_crop_slots = 0;
  double initial_seed_bill = 0.0;
  for (int crop = 0; crop < N_CROPS; ++crop) {
    const int added = std::max(
        0, int(plan.crop_targets[crop]) - own.committed_crops[crop]);
    new_crop_slots += added;
    initial_seed_bill += added * SEED_COST[crop];
  }
  int new_animal_slots = 0;
  for (const auto& projection : animal_projection)
    new_animal_slots += projection.new_purchases;
  const double immediate_actions = own.hard_water + own.hard_feed +
      2.0 * new_crop_slots + setup_actions + setup_moves +
      2.0 * total_new_placements;
  const int projection_units = std::max(
      int(farm.hands.size()) + 1, required_hands + 1);
  const int turns_left_today = std::max(
      0, env.config().turns_per_day - env.hour());
  const double today_capacity = projection_units * turns_left_today;
  const double productive_capacity_per_day = projection_units *
      (12.0 / std::max(0.5, genome_.risk_multiplier));
  parts.immediate_commitment_actions = immediate_actions;
  parts.today_action_capacity = today_capacity;
  parts.today_deadline_slack = today_capacity - immediate_actions;

  // Cash already in the shed is identical for every candidate and therefore
  // cannot identify the payback of a new project.  Measure the first return of
  // the still-unrealised commitments selected by this plan instead.
  parts.first_cash_lag = 30.0;
  for (int crop = 0; crop < N_CROPS; ++crop)
    if (int(plan.crop_targets[crop]) > own.committed_crops[crop])
      parts.first_cash_lag = std::min<double>(
          parts.first_cash_lag,
          CROP_ONGOING[crop] ? CROP_FIRST[crop]
                             : std::max(CROP_FIRST[crop], CROP_MAX_DAY[crop]));
  for (int animal = 0; animal < N_ANIMALS; ++animal)
    if (int(plan.animal_targets[animal]) > own.total_animals[animal])
      parts.first_cash_lag = std::min<double>(
          parts.first_cash_lag, ANIMAL_FIRST[animal] + setup_delay_days);

  // Pack not-yet-materialised setup chains across the planned worker lanes.
  // This is a small deterministic list-scheduling projection, not an action
  // policy.  It lets equal-industry SCALE candidates reveal whether the added
  // units fit before their first cash deadline and how much logistics they add.
  std::vector<double> commissioning_jobs;
  auto append_animal_jobs = [&](int placements, const StructureSetup& setup) {
    if (placements <= 0) return;
    const double average_distance = setup.placement_distance /
        std::max(1, placements);
    const double average_structure_actions =
        (setup.builds + setup.digs) / double(std::max(1, placements));
    for (int i = 0; i < placements; ++i) {
      // Visit shed/pickup, travel to the slot, place, and reserve FEED+CARE.
      const double steps = 4.0 + average_structure_actions +
          2.0 * average_distance;
      commissioning_jobs.push_back(steps);
      parts.new_service_distance += average_distance;
      parts.new_service_route_span = std::max(
          parts.new_service_route_span, average_distance);
    }
    parts.commissioning_unreachable_count += setup.missing;
  };
  append_animal_jobs(new_goose_placements, coop_setup);
  append_animal_jobs(new_pasture_placements, pasture_setup);

  struct CropSlot {
    int shed_distance = 0;
    int preparation_actions = 0;
  };
  std::vector<CropSlot> crop_slots;
  const int board = env.config().board_size;
  auto nearest_shed_distance = [&](Position pos) {
    int best = 2 * board;
    for (const auto shed : SHED_POS)
      best = std::min(best, distance(pos, shed));
    return best;
  };
  for (int cell = 0; cell < int(farm.tiles.size()); ++cell) {
    const Position pos{int16_t(cell % board), int16_t(cell / board)};
    if (!unlocked(farm, pos)) continue;
    const auto& tile = farm.tiles[cell];
    if (tile.kind != TileKind::EMPTY && tile.kind != TileKind::WEED) continue;
    crop_slots.push_back({nearest_shed_distance(pos),
                          tile.kind == TileKind::WEED ? 1 : 0});
  }
  std::sort(crop_slots.begin(), crop_slots.end(),
            [](const CropSlot& lhs, const CropSlot& rhs) {
              const int lhs_cost = lhs.shed_distance + lhs.preparation_actions;
              const int rhs_cost = rhs.shed_distance + rhs.preparation_actions;
              return lhs_cost != rhs_cost ? lhs_cost < rhs_cost
                                           : lhs.shed_distance < rhs.shed_distance;
            });
  // Newly built animal structures consume empty/weed cells before crops.
  const int structure_cells = std::min<int>(
      crop_slots.size(), coop_setup.builds + pasture_setup.builds);
  int crop_cursor = structure_cells;
  for (int crop_slot = 0; crop_slot < new_crop_slots; ++crop_slot) {
    if (crop_cursor >= int(crop_slots.size())) {
      parts.commissioning_unreachable_count += 1.0;
      continue;
    }
    const auto& slot = crop_slots[crop_cursor++];
    // Visit shed/pickup, travel to cell, optional DIG, PLANT and first WATER.
    const double steps = 3.0 + slot.preparation_actions +
        2.0 * slot.shed_distance;
    commissioning_jobs.push_back(steps);
    parts.new_service_distance += slot.shed_distance;
    parts.new_service_route_span = std::max<double>(
        parts.new_service_route_span, slot.shed_distance);
  }
  std::sort(commissioning_jobs.begin(), commissioning_jobs.end(),
            std::greater<double>());
  std::vector<double> lane_load(std::max(1, planned_units), 0.0);
  for (const double job : commissioning_jobs) {
    const auto lane = std::min_element(lane_load.begin(), lane_load.end());
    *lane += job;
    parts.commissioning_total_steps += job;
  }
  parts.commissioning_job_count = commissioning_jobs.size();
  parts.commissioning_makespan_steps = lane_load.empty()
      ? 0.0 : *std::max_element(lane_load.begin(), lane_load.end());
  const double turns_until_first_cash = std::max(
      0.0, parts.first_cash_lag * env.config().turns_per_day - env.hour());
  parts.commissioning_first_cash_slack_steps =
      turns_until_first_cash - parts.commissioning_makespan_steps;
  parts.commissioning_steps_per_added_unit =
      parts.commissioning_total_steps /
      std::max(1.0, parts.commissioning_job_count);

  auto crop_units_by_lag = [&](int crop, int lag) {
    if (lag < CROP_FIRST[crop]) return 0.0;
    if (CROP_ONGOING[crop]) {
      const int events = std::min(
          CROP_MAX_YIELD[crop],
          1 + std::max(0, (lag - CROP_FIRST[crop]) /
                                std::max(1, CROP_INTERVAL[crop])));
      return double(events);
    }
    const int harvest_age = std::min(CROP_MAX_DAY[crop], lag);
    const int water_start = (CROP_MAX_DAY[crop] + 1) / 2;
    return double(std::min(
        CROP_MAX_YIELD[crop],
        1 + std::max(0, harvest_age - water_start + 1)));
  };
  auto animal_units_by_lag = [&](int animal, int lag) {
    const int effective_lag = lag - setup_delay_days;
    if (effective_lag < ANIMAL_FIRST[animal]) return 0.0;
    const int cycles = 1 + std::max(
        0, (effective_lag - ANIMAL_FIRST[animal]) /
               std::max(1, ANIMAL_INTERVAL[animal]));
    // FEED+CARE gives one base product plus the accumulated interval care.
    return double(cycles * (1 + ANIMAL_INTERVAL[animal]));
  };

  parts.minimum_window_slack = std::numeric_limits<double>::infinity();
  const double recurring_utilization = workload /
      std::max(1.0, productive_capacity_per_day);
  const double immediate_utilization = immediate_actions /
      std::max(1.0, today_capacity);
  parts.peak_daily_utilization_x100 =
      100.0 * std::max(recurring_utilization, immediate_utilization);
  for (int window = 0; window < 3; ++window) {
    const int horizon = std::min(days_left, projection_horizons[window]);
    const double action_demand = immediate_actions +
        workload * std::max(0, horizon - 1);
    const double action_capacity = productive_capacity_per_day * horizon;
    parts.window_action_demand[window] = action_demand;
    parts.window_action_capacity[window] = action_capacity;
    parts.minimum_window_slack = std::min(
        parts.minimum_window_slack, action_capacity - action_demand);

    double inflow = 0.0;
    for (int crop = 0; crop < N_CROPS; ++crop)
      inflow += std::max(0, int(plan.crop_targets[crop])) *
          crop_units_by_lag(crop, horizon) * future_price(crop, horizon);
    double feed_outflow = 0.0;
    for (int animal = 0; animal < N_ANIMALS; ++animal) {
      const int target = std::max(
          0, int(plan.animal_service_targets[animal]));
      inflow += target * animal_units_by_lag(animal, horizon) *
          future_price(ANIMAL_PRODUCT[animal], horizon);
      feed_outflow += target * horizon * future_price(0, horizon);
    }
    const double fertilizer_inflow =
        genome_.fertilizer_value_fraction *
        std::accumulate(plan.animal_service_targets.begin(),
                        plan.animal_service_targets.end(), 0.0) *
        horizon * future_price(8, horizon);
    const double hire_outflow = horizon * daily_hire_cost(required_hands);
    const double capital_outflow = initial_seed_bill +
        parts.animal_purchase_cost + parts.land_cost;
    parts.window_net_cash[window] = inflow + fertilizer_inflow -
        feed_outflow - hire_outflow - capital_outflow;
  }
  if (!std::isfinite(parts.minimum_window_slack))
    parts.minimum_window_slack = 0.0;
  if (breakdown != nullptr && planner_state != nullptr) {
    // Probe the candidate with the actual task generator and joint scheduler.
    // This is intentionally evaluated only for the baseline and the selected
    // offline candidate that request a breakdown, not for every analytic arm.
    // It therefore adds evidence to W3 without multiplying the hot candidate
    // enumeration cost by the full SWITCH lattice.
    const auto tasks = build_tasks(env, player, plan, *planner_state);
    AdaptivePlannerState probe_state = *planner_state;
    const auto actions = assign_tasks(
        env, player, tasks, probe_state);
    std::vector<Position> positions{farm.farmer};
    positions.insert(positions.end(), farm.hands.begin(), farm.hands.end());
    std::vector<uint8_t> matched(tasks.size(), uint8_t(0));
    constexpr int hard_priority = 950;
    parts.current_task_count = double(tasks.size());
    parts.minimum_assignment_slack = std::numeric_limits<double>::infinity();
    for (const auto& task : tasks) {
      if (task.priority >= hard_priority) parts.current_hard_task_count += 1.0;
      if (task.deadline_step >= 719) continue;
      int nearest = std::numeric_limits<int>::max();
      for (const auto& position : positions)
        nearest = std::min(nearest, distance(position, task.target));
      const int available = std::max(0, task.deadline_step - env.step_count());
      if (nearest + 1 > available)
        parts.deadline_infeasible_task_count += 1.0;
    }
    const int units = std::min<int>(positions.size(), actions.size());
    for (int unit = 0; unit < units; ++unit) {
      if (unit >= int(probe_state.sticky_target.size()) ||
          unit >= int(probe_state.sticky_cell.size()) ||
          unit >= int(probe_state.sticky_op.size()))
        continue;
      const int reservation = probe_state.sticky_target[unit];
      const int cell = probe_state.sticky_cell[unit];
      const int op = probe_state.sticky_op[unit];
      int selected = -1;
      for (int task_index = 0; task_index < int(tasks.size()); ++task_index) {
        const auto& task = tasks[task_index];
        if (matched[task_index] || task.reservation_key != reservation ||
            int(task.action.op) != op ||
            pos_index(task.target, env.config().board_size) != cell)
          continue;
        selected = task_index;
        break;
      }
      if (selected < 0) continue;
      matched[selected] = uint8_t(1);
      const auto& task = tasks[selected];
      const int move = distance(positions[unit], task.target);
      const int steps = move + 1;
      parts.assigned_task_count += 1.0;
      if (task.priority >= hard_priority)
        parts.assigned_hard_task_count += 1.0;
      parts.total_assignment_distance += move;
      parts.total_assignment_steps += steps;
      parts.assigned_expected_cash_gain += task.expected_cash_gain;
      parts.minimum_assignment_slack = std::min(
          parts.minimum_assignment_slack,
          double(task.deadline_step - env.step_count() - steps));
    }
    for (int task_index = 0; task_index < int(tasks.size()); ++task_index) {
      if (matched[task_index]) continue;
      const auto& task = tasks[task_index];
      if (task.priority >= hard_priority)
        parts.unassigned_hard_task_count += 1.0;
      parts.unassigned_delayed_loss += task.loss_if_delayed;
    }
    if (!std::isfinite(parts.minimum_assignment_slack))
      parts.minimum_assignment_slack = 0.0;
    parts.assigned_value_per_step = parts.assigned_expected_cash_gain /
        std::max(1.0, parts.total_assignment_steps);
  }
  if (breakdown != nullptr) *breakdown = parts;
  return value;
}

std::vector<AdaptiveTask> NativeAdaptivePlanner::build_tasks(
    const Simulator& env, int player, const AdaptivePlan& plan,
    const AdaptivePlannerState& state) const {
  const auto& backbone = active_backbone(state);
  std::vector<AdaptiveTask> tasks;
  const auto& farm = env.farms()[player];
  const auto& priv = env.privates()[player];
  const auto counts = counts_for(env, player);
  const int board = env.config().board_size;
  const int day = env.day(), hour = env.hour(), step = env.step_count();
  const int liquidation_day = backbone.enabled
      ? std::clamp(int(backbone.liquidation_start_step) / 24, 0, TOTAL_DAYS - 1)
      : genome_.liquidation_day;
  bool fertilizer_on_unit = false;
  for (const auto& inventory : priv.inventories)
    fertilizer_on_unit = fertilizer_on_unit || inventory[8] > 0;
  const bool fertilizer_available = priv.shed[8] > 0 || fertilizer_on_unit;
  int max_feed_priority = 0;
  int feed_task_count = 0;
  auto add = [&](Action action, Position target, int priority, int deadline,
                 int required = -1, int key = -1) {
    if (key < 0) key = pos_index(target, board);
    tasks.push_back({action, target, priority, deadline, required, key, 0.0, 0.0});
  };
  auto add_value = [&](Action action, Position target, int priority, int deadline,
                       int required, int key, double cash_gain,
                       double delayed_loss) {
    if (key < 0) key = pos_index(target, board);
    tasks.push_back({action, target, priority, deadline, required, key,
                     std::max(0.0, cash_gain), std::max(0.0, delayed_loss)});
  };

  std::array<int, N_CROPS> water_remaining{};
  std::array<int, N_CROPS> harvest_remaining{};
  std::array<int, N_CROPS> fertilize_remaining{};
  std::array<int, N_CROPS> water_pace_boost{};
  std::array<int, N_CROPS> harvest_pace_boost{};
  std::array<int, N_CROPS> fertilize_pace_boost{};
  std::array<int, N_CROPS> plant_pace_boost{};
  const auto crop_market_race = public_crop_market_race_pressure(env, player);
  std::array<int, N_CROPS> crop_market_race_boost{};
  for (int crop = 0; crop < N_CROPS; ++crop)
    crop_market_race_boost[crop] = int(std::nearbyint(
        300.0 * genome_.market_race_acceleration *
        crop_market_race[crop]));
  std::array<int, N_ANIMALS> feed_remaining{};
  std::array<int, N_ANIMALS> care_remaining{};
  std::array<int, N_ANIMALS> product_remaining{};
  std::array<int, N_ANIMALS> animal_fertilizer_remaining{};
  bool animal_flow_active = backbone.enabled &&
      genome_.animal_flow_control > 0.0;
  if (backbone.enabled) {
    auto effective_flow = [&](int crop, int raw) {
      if (state.crop_suffix == 1 && crop != 0 && crop != 1)
        return std::max(0, int(std::nearbyint(
            raw * (1.0 - genome_.pet_crop_flex))));
      return raw;
    };
    auto pace_boost = [&](int target, int completed) {
      const int due = (target * (hour + 1) + 23) / 24;
      const int backlog = std::max(0, due - completed);
      return int(std::nearbyint(std::min(
          300.0, genome_.flow_pace_priority * backlog)));
    };
    for (int crop = 0; crop < N_CROPS; ++crop) {
      const int water_target = effective_flow(
          crop, int(backbone.crop_water[day][crop]));
      const int harvest_target = effective_flow(
          crop, int(backbone.crop_harvest[day][crop]));
      const int fertilize_target = effective_flow(
          crop, int(backbone.crop_fertilize[day][crop]));
      const int plant_target = effective_flow(crop, int(std::nearbyint(
          backbone.crop_plant[day][crop] *
          (1.0 - genome_.backbone_crop_flex))));
      water_remaining[crop] = std::max(
          0, water_target - int(state.crop_water_actions[crop]));
      harvest_remaining[crop] = std::max(
          0, harvest_target - int(state.crop_harvest_actions[crop]));
      fertilize_remaining[crop] = std::max(
          0, fertilize_target - int(state.crop_fertilize_actions[crop]));
      water_pace_boost[crop] = pace_boost(
          water_target, int(state.crop_water_actions[crop]));
      harvest_pace_boost[crop] = pace_boost(
          harvest_target, int(state.crop_harvest_actions[crop]));
      fertilize_pace_boost[crop] = pace_boost(
          fertilize_target, int(state.crop_fertilize_actions[crop]));
      plant_pace_boost[crop] = pace_boost(
          plant_target, int(state.crop_plant_actions[crop]));
    }
    if (animal_flow_active && genome_.animal_flow_pressure_gate > 0.0) {
      int crop_backlog = 0;
      for (int crop = 0; crop < N_CROPS; ++crop)
        crop_backlog += water_remaining[crop] + harvest_remaining[crop] +
            fertilize_remaining[crop] + std::max(
                0, int(backbone.crop_plant[day][crop]) -
                       int(state.crop_plant_actions[crop]));
      const int worker_lanes = int(farm.hands.size()) + 1;
      const int remaining_lane_steps = std::max(
          1, worker_lanes * std::max(1, 24 - hour));
      // Use a generic movement-to-work logistics load estimate; no route
      // coordinate or action is imported.
      const double crop_pressure = 2.35 * crop_backlog /
          double(remaining_lane_steps);
      animal_flow_active =
          crop_pressure >= genome_.animal_flow_pressure_gate;
    }
    if (animal_flow_active && genome_.animal_flow_executable_gate > 0.0) {
      int executable_crop_actions = 0;
      int unlocked_empty = 0;
      int unlocked_weeds = 0;
      for (int cell = 0; cell < board * board; ++cell) {
        const Position pos{int16_t(cell % board), int16_t(cell / board)};
        if (!unlocked(farm, pos)) continue;
        const auto& tile = farm.tiles[cell];
        if (tile.kind == TileKind::EMPTY) {
          ++unlocked_empty;
          continue;
        }
        if (tile.kind == TileKind::WEED) {
          ++unlocked_weeds;
          continue;
        }
        if (tile.kind != TileKind::PLANT) continue;
        const int crop = int(tile.crop);
        if (crop < 0 || crop >= N_CROPS) continue;

        // Count at most the work that the task generator can issue now.  Hard
        // crop safety remains visible even when the reference calendar is
        // already satisfied.
        if (day < TOTAL_DAYS - 1 && !tile.watered_today &&
            (tile.consecutive_unwatered >= 1 ||
             water_remaining[crop] > 0))
          ++executable_crop_actions;
        if (tile.yield_units > 0) {
          const int age = day - tile.planted_day;
          const bool flow_harvest = harvest_remaining[crop] > 0 &&
              tile.yield_units >= std::max(
                  1, int(backbone.crop_harvest_min_yield[day][crop]));
          if (CROP_ONGOING[crop] || flow_harvest || day >= 28 ||
              (!CROP_ONGOING[crop] && age >= CROP_MAX_DAY[crop]))
            ++executable_crop_actions;
        }
        if (fertilizer_available && fertilize_remaining[crop] > 0 &&
            tile.fertilized_until_day < day)
          ++executable_crop_actions;
      }

      int plantable_now = 0;
      for (int crop = 0; crop < N_CROPS; ++crop) {
        if (day + CROP_FIRST[crop] >= TOTAL_DAYS) continue;
        const int stock_need = std::max(
            0, int(plan.crop_targets[crop]) - counts.field_crops[crop]);
        int flow_target = int(backbone.crop_plant[day][crop]);
        if (state.crop_suffix == 1 && crop != 0 && crop != 1)
          flow_target = std::max(0, int(std::nearbyint(
              flow_target * (1.0 - genome_.pet_crop_flex))));
        const int flow_need = std::max(
            0, flow_target - int(state.crop_plant_actions[crop]));
        plantable_now += std::min<int>(
            priv.seeds[crop], std::max(stock_need, flow_need));
      }
      executable_crop_actions += std::min(unlocked_empty, plantable_now);

      const int clear_need = std::max(
          0, int(backbone.crop_clear[day]) - int(state.crop_clear_actions));
      if (clear_need > 0 ||
          (unlocked_empty == 0 && plantable_now > 0))
        executable_crop_actions += std::min(
            unlocked_weeds,
            std::max(clear_need, plantable_now - unlocked_empty));

      const int worker_lanes = int(farm.hands.size()) + 1;
      const int remaining_lane_steps = std::max(
          1, worker_lanes * std::max(1, 24 - hour));
      const double executable_pressure =
          2.35 * executable_crop_actions / double(remaining_lane_steps);
      animal_flow_active = executable_pressure >=
          genome_.animal_flow_executable_gate;
    }
    if (animal_flow_active) {
      for (int animal = 0; animal < N_ANIMALS; ++animal) {
        feed_remaining[animal] = std::max(
            0, int(backbone.animal_feed[day][animal]) -
                   int(state.animal_feed_actions[animal]));
        care_remaining[animal] = std::max(
            0, int(backbone.animal_care[day][animal]) -
                   int(state.animal_care_actions[animal]));
        product_remaining[animal] = std::max(
            0, int(backbone.animal_product[day][animal]) -
                   int(state.animal_product_actions[animal]));
        animal_fertilizer_remaining[animal] = std::max(
            0, int(backbone.animal_fertilizer[day][animal]) -
                   int(state.animal_fertilizer_actions[animal]));
      }
    }
  }

  std::vector<int> candidate_cell_order;
  candidate_cell_order.reserve(board * board);
  if (genome_.local_candidate_order <= 0.0) {
    for (int cell = 0; cell < board * board; ++cell)
      candidate_cell_order.push_back(cell);
  } else {
    // Manhattan distance on a 10x10 board is at most 18.  Distance buckets
    // avoid a per-step comparison sort in the native hot loop.
    std::array<std::vector<int>, 32> buckets;
    std::vector<Position> workforce{farm.farmer};
    workforce.insert(workforce.end(), farm.hands.begin(), farm.hands.end());
    for (int cell = 0; cell < board * board; ++cell) {
      const Position pos{int16_t(cell % board), int16_t(cell / board)};
      int nearest = 2 * board;
      for (const auto worker : workforce)
        nearest = std::min(nearest, distance(pos, worker));
      buckets[std::clamp(nearest, 0, int(buckets.size()) - 1)]
          .push_back(cell);
    }
    for (const auto& bucket : buckets)
      candidate_cell_order.insert(
          candidate_cell_order.end(), bucket.begin(), bucket.end());
  }

  if (animal_flow_active) {
    // Daily service counts only become economically meaningful when the
    // selected animals are the ones whose production or survival depends on
    // today's work.  Preserve the surrounding candidate order and reorder
    // only animal cells by live urgency/value.
    std::vector<int> animal_slots;
    std::vector<int> animal_cells;
    for (int at = 0; at < int(candidate_cell_order.size()); ++at) {
      const int cell = candidate_cell_order[at];
      if (farm.tiles[cell].kind == TileKind::ANIMAL) {
        animal_slots.push_back(at);
        animal_cells.push_back(cell);
      }
    }
    std::vector<Position> workforce{farm.farmer};
    workforce.insert(workforce.end(), farm.hands.begin(), farm.hands.end());
    auto service_score = [&](int cell) {
      const auto& tile = farm.tiles[cell];
      const int animal = std::clamp(
          int(tile.animal) - 9, 0, N_ANIMALS - 1);
      const int delta = day + 1 - tile.placed_day - ANIMAL_FIRST[animal];
      const bool produces_tonight = delta >= 0 &&
          delta % std::max(1, ANIMAL_INTERVAL[animal]) == 0;
      const Position pos{int16_t(cell % board), int16_t(cell / board)};
      int nearest = 2 * board;
      for (const auto worker : workforce)
        nearest = std::min(nearest, distance(pos, worker));
      return (tile.consecutive_unfed >= 1 ? 100000 : 0) +
          (produces_tonight ? 10000 : 0) +
          1000 * int(tile.yield_units) +
          100 * int(tile.pending_care_bonus) +
          (tile.fertilizer_available ? 10 : 0) - nearest;
    };
    std::stable_sort(animal_cells.begin(), animal_cells.end(),
        [&](int a, int b) {
          const int sa = service_score(a), sb = service_score(b);
          return sa != sb ? sa > sb : a < b;
        });
    for (int at = 0; at < int(animal_slots.size()); ++at)
      candidate_cell_order[animal_slots[at]] = animal_cells[at];
  }

  for (const int cell : candidate_cell_order) {
    const int x = cell % board, y = cell / board;
    Position pos{int16_t(x), int16_t(y)};
    const auto& tile = farm.tiles[y * board + x];
    if (tile.kind == TileKind::ANIMAL) {
      const int animal = std::clamp(int(tile.animal) - 9, 0, N_ANIMALS - 1);
      const int product = ANIMAL_PRODUCT[animal];
      const double product_price = env.market().prices[product];
      const int tonight_delta = day + 1 - tile.placed_day -
          ANIMAL_FIRST[animal];
      const bool produces_tonight = tonight_delta >= 0 &&
          tonight_delta % std::max(1, ANIMAL_INTERVAL[animal]) == 0;
      bool care_bonus_realisable = true;
      if (genome_.terminal_animal_economics > 0.0) {
        care_bonus_realisable = false;
        // CARE is banked only after tonight's production and is consumed by a
        // later production.  The later product must appear by the start of the
        // final playable day so that a unit can still collect and sell it.
        for (int next_day = day + 2; next_day < TOTAL_DAYS; ++next_day) {
          const int delta = next_day - tile.placed_day -
              ANIMAL_FIRST[animal];
          if (delta >= 0 &&
              delta % std::max(1, ANIMAL_INTERVAL[animal]) == 0) {
            care_bonus_realisable = true;
            break;
          }
        }
      }
      const bool terminal_feed_realisable =
          genome_.terminal_animal_economics <= 0.0 ||
          day < TOTAL_DAYS - 2 ||
          (produces_tonight &&
           tile.yield_units < ANIMAL_MAX_HELD[animal]);
      const bool hard_feed = tile.consecutive_unfed >= 1;
      const bool flow_feed = !backbone.enabled ||
          !animal_flow_active || feed_remaining[animal] > 0;
      if (day < TOTAL_DAYS - 1 && !tile.fed_today &&
          terminal_feed_realisable && (hard_feed || flow_feed))
      {
        ++feed_task_count;
        const int feed_priority = hard_feed ? 1000 :
            (day >= liquidation_day ? 960 : 760);
        max_feed_priority = std::max(max_feed_priority, feed_priority);
        add_value(Action{Op::FEED}, pos, feed_priority,
                  (day + 1) * 24 - 1, 0, -1,
                  product_price / std::max(1, ANIMAL_INTERVAL[animal]),
                  hard_feed
                      ? ANIMAL_COST[animal] + tile.yield_units * product_price
                      : product_price / std::max(1, ANIMAL_INTERVAL[animal]));
        if (feed_remaining[animal] > 0) --feed_remaining[animal];
      }
      const bool flow_care = !backbone.enabled ||
          !animal_flow_active || care_remaining[animal] > 0;
      if (day < TOTAL_DAYS - 1 && !tile.cared_today && flow_care &&
          care_bonus_realisable) {
        add_value(Action{Op::CARE}, pos, 680, (day + 1) * 24 - 1, -1, -1,
                  product_price, product_price);
        if (care_remaining[animal] > 0) --care_remaining[animal];
      }
      const bool hard_product =
          tile.yield_units >= ANIMAL_MAX_HELD[animal] ||
          day >= liquidation_day;
      const bool flow_product = !backbone.enabled ||
          !animal_flow_active ||
          product_remaining[animal] > 0;
      if (tile.yield_units > 0 && (hard_product || flow_product)) {
        add_value(Action{Op::HARVEST}, pos,
                  (day >= liquidation_day ? 900 : 650) + 10 * tile.yield_units,
                  719, -1, -1, tile.yield_units * product_price,
                  tile.yield_units >= ANIMAL_MAX_HELD[animal]
                      ? product_price : 0.0);
        if (product_remaining[animal] > 0) --product_remaining[animal];
      }
      const bool flow_fertilizer = !backbone.enabled ||
          !animal_flow_active ||
          animal_fertilizer_remaining[animal] > 0;
      if (tile.fertilizer_available && flow_fertilizer) {
        add_value(Action{Op::COLLECT_FERTILIZER}, pos, 660,
                  (day + 1) * 24 - 1, -1, -1,
                  env.market().prices[8], env.market().prices[8]);
        if (animal_fertilizer_remaining[animal] > 0)
          --animal_fertilizer_remaining[animal];
      }
    } else if (tile.kind == TileKind::PLANT) {
      const int crop = int(tile.crop);
      bool fertilizer_yield_day = false;
      bool water_for_yield = false;
      if (crop >= 0 && crop < N_CROPS && day < TOTAL_DAYS - 1) {
        const int age = day - tile.planted_day;
        if (CROP_ONGOING[crop]) {
          const int next_day = day + 1;
          const int delta = next_day - tile.planted_day - CROP_FIRST[crop];
          const int production_count = delta >= 0
              ? delta / std::max(1, CROP_INTERVAL[crop]) + 1 : 0;
          fertilizer_yield_day = delta >= 0 &&
              delta % std::max(1, CROP_INTERVAL[crop]) == 0 &&
              production_count <= CROP_MAX_YIELD[crop];
        } else {
          const int water_start = (CROP_MAX_DAY[crop] + 1) / 2;
          fertilizer_yield_day = age >= water_start && age <= CROP_MAX_DAY[crop];
          water_for_yield = fertilizer_yield_day;
        }
      }
      const bool fertilizer_profitable = crop >= 0 && crop < N_CROPS &&
          env.market().prices[crop] >
              genome_.fertilizer_value_fraction * env.market().prices[8];
      const bool flow_fertilizer = crop >= 0 && crop < N_CROPS &&
          fertilize_remaining[crop] > 0;
      const bool needs_fertilizer = (fertilizer_yield_day || flow_fertilizer) &&
                                    fertilizer_available &&
                                    fertilizer_profitable &&
                                    tile.fertilized_until_day < day;
      if (needs_fertilizer) {
        add_value(Action{Op::FERTILIZE}, pos,
                   730 + (crop >= 0 && crop < N_CROPS
                              ? fertilize_pace_boost[crop] +
                                    crop_market_race_boost[crop] : 0),
                  (day + 1) * 24 - 2, 8, -1,
                  std::max(0, env.market().prices[crop] - env.market().prices[8]),
                  env.market().prices[crop]);
        if (crop >= 0 && crop < N_CROPS && fertilize_remaining[crop] > 0)
          fertilize_remaining[crop]--;
      }
      const bool flow_water = crop >= 0 && crop < N_CROPS &&
          water_remaining[crop] > 0;
      if (day < TOTAL_DAYS - 1 && !tile.watered_today &&
          (tile.consecutive_unwatered >= 1 || water_for_yield ||
           needs_fertilizer || flow_water)) {
        add_value(Action{Op::WATER}, pos,
                  tile.consecutive_unwatered >= 1 ? 980 :
                       620 + (crop >= 0 && crop < N_CROPS
                                  ? water_pace_boost[crop] +
                                        crop_market_race_boost[crop] : 0),
                  (day + 1) * 24 - 1, -1, -1,
                  water_for_yield ? env.market().prices[crop] : 0.0,
                  tile.consecutive_unwatered >= 1
                      ? std::max<double>(SEED_COST[crop],
                          tile.yield_units * env.market().prices[crop])
                      : env.market().prices[crop]);
        if (crop >= 0 && crop < N_CROPS && water_remaining[crop] > 0)
          water_remaining[crop]--;
      }
      if (tile.yield_units > 0 && crop >= 0 && crop < N_CROPS) {
        const int age = day - tile.planted_day;
        bool harvest = CROP_ONGOING[crop];
        int priority = day >= liquidation_day ? 900 : 640;
        const int min_flow_yield = std::max(
            1, int(backbone.crop_harvest_min_yield[day][crop]));
        const bool flow_harvest = backbone.enabled &&
            harvest_remaining[crop] > 0 &&
            tile.yield_units >= min_flow_yield;
        if (flow_harvest) {
          harvest = true;
          priority = std::max(priority,
              700 + harvest_pace_boost[crop] +
                  crop_market_race_boost[crop]);
        }
        if (!CROP_ONGOING[crop] &&
            (age > CROP_MAX_DAY[crop] ||
             (age == CROP_MAX_DAY[crop] && tile.watered_today))) {
          harvest = true;
          priority = 950;
        }
        if (day >= 28) harvest = true;
        if (harvest) {
          add_value(Action{Op::HARVEST}, pos, priority + 12 * tile.yield_units,
                    719, -1, -1,
                    tile.yield_units * env.market().prices[crop],
                    !CROP_ONGOING[crop] && age >= CROP_MAX_DAY[crop]
                        ? tile.yield_units * env.market().prices[crop] : 0.0);
          if (harvest_remaining[crop] > 0) harvest_remaining[crop]--;
        }
      }
    }
  }

  // Structure creation uses central unlocked cells first.  This is a generic
  // logistics heuristic, not a replay coordinate table.
  int pasture_deficit = std::max(0, int(plan.animal_targets[1] + plan.animal_targets[2])
                                      - counts.pastures);
  int coop_deficit = std::max(0, int(plan.animal_targets[0]) - counts.coops);
  int future_pasture_target = int(plan.animal_targets[1] + plan.animal_targets[2]);
  int future_coop_target = int(plan.animal_targets[0]);
  if (backbone.enabled && genome_.future_structure_reservation_fraction > 0.0) {
    for (int future_day = day; future_day < TOTAL_DAYS; ++future_day) {
      int cow_target = backbone.animal_targets[future_day][1];
      const int sheep_target = backbone.animal_targets[future_day][2];
      if (state.animal_branch == 1) {
        const int owned_cows = counts.total_animals[1];
        cow_target = owned_cows + int(std::nearbyint(
            std::max(0, cow_target - owned_cows) *
            (1.0 - genome_.yarn_animal_flex)));
      }
      future_pasture_target = std::max(
          future_pasture_target, cow_target + sheep_target);
      future_coop_target = std::max<int>(
          future_coop_target, backbone.animal_targets[future_day][0]);
    }
  }
  const int current_pasture_target =
      int(plan.animal_targets[1] + plan.animal_targets[2]);
  const int current_coop_target = int(plan.animal_targets[0]);
  const int reserved_pasture_target = current_pasture_target +
      int(std::nearbyint(genome_.future_structure_reservation_fraction *
          std::max(0, future_pasture_target - current_pasture_target)));
  const int reserved_coop_target = current_coop_target +
      int(std::nearbyint(genome_.future_structure_reservation_fraction *
          std::max(0, future_coop_target - current_coop_target)));
  const int pasture_reservation_deficit = std::max(
      pasture_deficit, reserved_pasture_target - counts.pastures);
  const int coop_reservation_deficit = std::max(
      coop_deficit, reserved_coop_target - counts.coops);
  std::vector<Position> empty;
  for (int y = 0; y < board; ++y) for (int x = 0; x < board; ++x) {
    Position pos{int16_t(x), int16_t(y)};
    const auto& tile = farm.tiles[y * board + x];
    if (unlocked(farm, pos) && (tile.kind == TileKind::EMPTY || tile.kind == TileKind::WEED))
      empty.push_back(pos);
  }
  std::sort(empty.begin(), empty.end(), [](Position a, Position b) {
    const int da = std::min({distance(a, SHED_POS[0]), distance(a, SHED_POS[1]),
                             distance(a, SHED_POS[2]), distance(a, SHED_POS[3])});
    const int db = std::min({distance(b, SHED_POS[0]), distance(b, SHED_POS[1]),
                             distance(b, SHED_POS[2]), distance(b, SHED_POS[3])});
    return da != db ? da < db : (a.y != b.y ? a.y < b.y : a.x < b.x);
  });
  // Reserve the exact cells selected for future animal structures.  The old
  // implementation advanced one cursor through EMPTY+WEED cells, then reused
  // that cursor as an offset into an EMPTY-only crop list.  Whenever a weed
  // appeared in the structure prefix this skipped an unrelated empty crop
  // cell and silently reduced productive capacity.
  std::vector<uint8_t> reserved_structure_cell(board * board, uint8_t(0));
  size_t cursor = 0;
  for (int i = 0; i < pasture_reservation_deficit && cursor < empty.size();
       ++i, ++cursor) {
    reserved_structure_cell[pos_index(empty[cursor], board)] = uint8_t(1);
    if (i < pasture_deficit) {
      const auto& tile = farm.tiles[pos_index(empty[cursor], board)];
      add(Action{tile.kind == TileKind::WEED ? Op::DIG : Op::BUILD_PASTURE},
          empty[cursor], 700, 719);
    }
  }
  for (int i = 0; i < coop_reservation_deficit && cursor < empty.size();
       ++i, ++cursor) {
    reserved_structure_cell[pos_index(empty[cursor], board)] = uint8_t(1);
    if (i < coop_deficit) {
      const auto& tile = farm.tiles[pos_index(empty[cursor], board)];
      add(Action{tile.kind == TileKind::WEED ? Op::DIG : Op::BUILD_COOP},
          empty[cursor], 690, 719);
    }
  }

  // Animal placement and shed pickup.
  for (int animal = 0; animal < N_ANIMALS; ++animal) {
    const Item item = Item(9 + animal);
    int place_need = std::max(0, int(plan.animal_service_targets[animal]) -
        counts.field_animals[animal]);
    if (place_need <= 0) continue;
    TileKind structure = animal == 0 ? TileKind::COOP : TileKind::PASTURE;
    for (int y = 0; y < board && place_need > 0; ++y) for (int x = 0; x < board && place_need > 0; ++x) {
      Position pos{int16_t(x), int16_t(y)};
      const auto& tile = farm.tiles[y * board + x];
      if (tile.kind == structure && tile.animal == Item::NONE) {
        add(Action{Op::PLACE, item, 1}, pos, 740, 719, 9 + animal);
        --place_need;
      }
    }
    if (priv.shed[9 + animal] > 0)
      for (auto pos : SHED_POS)
        add(Action{Op::PICKUP, item, 1}, pos, 720, 719, -1,
            1000 + animal * 10 + pos_index(pos, board));
  }

  // Resource pickup lanes.
  if (feed_task_count > 0 && priv.shed[0] > 0) {
    for (auto pos : SHED_POS)
      add(Action{Op::PICKUP, Item::WHEAT, 6}, pos,
          std::max(820, max_feed_priority + 5),
          (day + 1) * 24 - 1, -1, 2000 + pos_index(pos, board));
  }
  if (priv.shed[8] > 0)
    for (auto pos : SHED_POS) add(Action{Op::PICKUP, Item::FERTILIZER, 4}, pos, 740, 719,
                                  -1, 2100 + pos_index(pos, board));

  // Plant only as many cells as there are seeds, with the plan deciding the
  // product mix.  Crops are allocated to currently empty cells near the shed.
  bool any_crop_project_open = backbone.enabled;
  for (int crop = 0; crop < N_CROPS; ++crop)
    any_crop_project_open = any_crop_project_open ||
        crop_project_open(genome_, crop, day);
  if (hour < 23 && any_crop_project_open) {
    std::vector<Position> plantable;
    for (auto pos : empty) {
      const auto& tile = farm.tiles[pos_index(pos, board)];
      if (tile.kind == TileKind::EMPTY &&
          !reserved_structure_cell[pos_index(pos, board)])
        plantable.push_back(pos);
    }
    const bool candidate_compact_layout =
        plan.candidate8_schedule_profile ==
            int8_t(ScheduleProfile::COMPACT_LAYOUT);
    const bool candidate_service_lanes =
        plan.candidate8_schedule_profile ==
            int8_t(ScheduleProfile::SERVICE_LANES);
    if (genome_.crop_lane_layout_mode >= 0.5 ||
        candidate_compact_layout || candidate_service_lanes) {
      const bool serpentine = genome_.crop_lane_layout_mode >= 1.5 ||
          candidate_service_lanes;
      std::stable_sort(plantable.begin(), plantable.end(),
          [&](Position a, Position b) {
            if (a.y != b.y) return a.y < b.y;
            if (serpentine && (a.y & 1)) return a.x > b.x;
            return a.x < b.x;
          });
    }
    size_t at = 0;
    std::array<int, N_CROPS> remaining{};
    for (int crop = 0; crop < N_CROPS; ++crop) {
      const int stock_need = std::max(
          0, int(plan.crop_targets[crop]) - counts.field_crops[crop]);
      int flow_target = backbone.enabled
          ? int(std::nearbyint(backbone.crop_plant[day][crop] *
                (1.0 - genome_.backbone_crop_flex))) : 0;
      if (state.crop_suffix == 1 && crop != 0 && crop != 1)
        flow_target = std::max(0, int(std::nearbyint(
            flow_target * (1.0 - genome_.pet_crop_flex))));
      const int flow_need = std::max(
          0, flow_target - int(state.crop_plant_actions[crop]));
      remaining[crop] = (backbone.enabled ||
                         crop_project_open(genome_, crop, day))
          ? std::min<int>(priv.seeds[crop], std::max(stock_need, flow_need))
          : 0;
    }
    while (at < plantable.size()) {
      int best = -1;
      for (int crop = 0; crop < N_CROPS; ++crop)
        if (remaining[crop] > 0 &&
            (best < 0 || env.market().prices[crop] > env.market().prices[best]))
          best = crop;
      if (best < 0) break;
      const double lifecycle_gross = env.market().prices[best] *
          std::max(1, CROP_MAX_YIELD[best]);
      const Position target = plantable[at++];
      add_value(Action{Op::PLANT, Item(best), 1}, target,
                 int(std::nearbyint(genome_.plant_priority)) +
                     plant_pace_boost[best] +
                     crop_market_race_boost[best],
                (day + 1) * 24 - 2, -1, -1,
                0.0,
                std::max<double>(SEED_COST[best], 0.25 * lifecycle_gross));
      remaining[best]--;
    }
  }

  // Weed clearing is valuable only while there is unfinished productive work.
  // When enabled, derive its urgency from the selected portfolio's actual
  // land-capacity deficit.  Merely having one EMPTY tile is not enough when
  // dozens of seeds/projects are waiting behind weeds.
  const int field_crop_count = std::accumulate(
      counts.field_crops.begin(), counts.field_crops.end(), 0);
  const int flow_clear_need = backbone.enabled
      ? std::max(0, int(backbone.crop_clear[day]) -
                       int(state.crop_clear_actions))
      : 0;
  if (any_crop_project_open &&
      (field_crop_count < std::accumulate(
           plan.crop_targets.begin(), plan.crop_targets.end(), 0) ||
       flow_clear_need > 0)) {
    int weed_priority = counts.empty == 0 ? 700 : 540;
    int clear_remaining = flow_clear_need > 0 ? flow_clear_need : counts.weeds;
    if (genome_.weed_capacity_priority > 0.0) {
      int usable_empty_cells = 0;
      for (int cell = 0; cell < board * board; ++cell) {
        const Position pos{int16_t(cell % board), int16_t(cell / board)};
        if (unlocked(farm, pos) &&
            farm.tiles[cell].kind == TileKind::EMPTY &&
            !reserved_structure_cell[cell])
          usable_empty_cells++;
      }
      int pending_crop_capacity = 0;
      for (int crop = 0; crop < N_CROPS; ++crop) {
        if (day + CROP_FIRST[crop] >= TOTAL_DAYS) continue;
        const int stock_need = std::max(
            0, int(plan.crop_targets[crop]) - counts.field_crops[crop]);
        int flow_target = backbone.enabled
            ? int(std::nearbyint(backbone.crop_plant[day][crop] *
                  (1.0 - genome_.backbone_crop_flex))) : 0;
        if (state.crop_suffix == 1 && crop != 0 && crop != 1)
          flow_target = std::max(0, int(std::nearbyint(
              flow_target * (1.0 - genome_.pet_crop_flex))));
        const int flow_need = std::max(
            0, flow_target - int(state.crop_plant_actions[crop]));
        pending_crop_capacity += std::max(stock_need, flow_need);
      }
      const int capacity_deficit = std::max(
          0, pending_crop_capacity - usable_empty_cells);
      clear_remaining = std::max(flow_clear_need, capacity_deficit);
      if (capacity_deficit > 0)
        weed_priority = int(std::nearbyint(genome_.weed_capacity_priority));
    }
    clear_remaining = std::min(clear_remaining, counts.weeds);
    for (int y = 0; y < board; ++y) for (int x = 0; x < board; ++x) {
      Position pos{int16_t(x), int16_t(y)};
      if (clear_remaining > 0 && farm.tiles[y * board + x].kind == TileKind::WEED) {
        add(Action{Op::DIG}, pos, weed_priority, 719);
        clear_remaining--;
      }
    }
  }

  // Cargo return.  Multiple units may use the same shed cell in one step, so
  // expose one independently reservable lane per unit rather than silently
  // limiting logistics to four workers.  The final two hours prioritize
  // returning cargo before automatic end-of-day transfer can overflow shed.
  int total_carried = 0;
  for (const auto& inventory : priv.inventories)
    total_carried += inventory_sum(inventory);
  const int unit_count = int(farm.hands.size()) + 1;
  // During the day, carried cargo may legitimately exceed today's free shed
  // space because shed sales can create room before automatic return.  The
  // final ten steps need an evacuation guard based on stock that has actually
  // entered the logistics ledger.  Potential map output remains harvestable;
  // only an executing final-step collection is capacity-checked below.
  const bool capacity_risk = hour >= 14 &&
      shed_sum(priv) + total_carried >
          env.config().shed_capacity - unit_count;
  // Capacity evacuation must not steal an immediately expiring WATER or
  // HARVEST lane.  Start it early enough to cross the board and place it at
  // the same feasibility boundary as survival work.  Deadline slack, travel
  // distance and the hard latest-start reservation then decide which lane is
  // truly urgent instead of an unconditional higher priority stealing work.
  const int drop_priority = capacity_risk
      ? std::min(979, 950 + 5 * std::max(0, hour - 14))
      :
      (day >= liquidation_day ? 920 : (hour >= 22 ? 850 : 430));
  const int drop_deadline = capacity_risk
      ? (day + 1) * env.config().turns_per_day - 1 : 719;
  int drop_lanes = unit_count;
  if (capacity_risk) {
    // Every occupied carrier needs an independent route lane.  Limiting this
    // to the current numerical overflow was insufficient: routine harvesting
    // continues during the remaining hours and a single lane cannot evacuate
    // several workers in parallel.  Survival and capacity work share the same
    // hard-priority boundary and are resolved by their live deadlines.
    drop_lanes = 0;
    for (const auto& inventory : priv.inventories) {
      const int cargo = inventory_sum(inventory);
      if (cargo > 0) drop_lanes++;
    }
    drop_lanes = std::max(1, drop_lanes);
  }
  for (int lane = 0; lane < drop_lanes; ++lane)
    add(Action{Op::DROP}, SHED_POS[lane % SHED_POS.size()], drop_priority,
        drop_deadline, -2, 3000 + lane);
  return tasks;
}

std::vector<Action> NativeAdaptivePlanner::assign_tasks(
    const Simulator& env, int player, const std::vector<AdaptiveTask>& tasks,
    AdaptivePlannerState& state) const {
  const auto& backbone = active_backbone(state);
  const auto& farm = env.farms()[player];
  const auto& priv = env.privates()[player];
  const int liquidation_day = backbone.enabled
      ? std::clamp(int(backbone.liquidation_start_step) / 24, 0, TOTAL_DAYS - 1)
      : genome_.liquidation_day;
  const ScheduleProfile candidate_schedule = ScheduleProfile(
      state.plan.candidate8_schedule_profile);
  const bool candidate_region =
      candidate_schedule == ScheduleProfile::REGION_BALANCED;
  const bool candidate_global =
      candidate_schedule == ScheduleProfile::GLOBAL_MATCHING;
  const bool candidate_chain =
      candidate_schedule == ScheduleProfile::CHAIN_CONTINUITY;
  std::vector<Position> positions{farm.farmer};
  positions.insert(positions.end(), farm.hands.begin(), farm.hands.end());
  const int units = int(positions.size());
  if (int(state.sticky_target.size()) != units ||
      int(state.sticky_cell.size()) != units ||
      int(state.sticky_op.size()) != units ||
      int(state.sticky_group.size()) != units) {
    state.sticky_target.assign(units, -1);
    state.sticky_cell.assign(units, -1);
    state.sticky_op.assign(units, int8_t(Op::PASS));
    state.sticky_group.assign(units, -1);
    state.region_anchor_cell.assign(units, -1);
    state.region_owner_by_cell.assign(
        env.config().board_size * env.config().board_size, -1);
    state.region_anchor_day = -1;
  }

  if ((genome_.region_ownership_scale > 0.0 ||
       genome_.region_anchor_distance_scale > 0.0 || candidate_region) &&
      (state.region_anchor_day != env.day() ||
       int(state.region_anchor_cell.size()) != units)) {
    auto is_productive_region_task = [&](const AdaptiveTask& task) {
      const Op op = task.action.op;
      return op != Op::PASS && op != Op::PICKUP && op != Op::DROP;
    };
    std::vector<int> cells;
    std::vector<int> workload(
        env.config().board_size * env.config().board_size, 0);
    std::vector<uint8_t> seen(
        env.config().board_size * env.config().board_size, uint8_t(0));
    for (const auto& task : tasks) {
      if (task.priority >= 950 || !is_productive_region_task(task))
        continue;
      const int cell = pos_index(task.target, env.config().board_size);
      workload[cell]++;
      if (!seen[cell]) {
        seen[cell] = uint8_t(1);
        cells.push_back(cell);
      }
    }
    auto morton = [&](int cell) {
      const int x = cell % env.config().board_size;
      const int y = cell / env.config().board_size;
      int code = 0;
      for (int bit = 0; bit < 5; ++bit) {
        code |= ((x >> bit) & 1) << (2 * bit);
        code |= ((y >> bit) & 1) << (2 * bit + 1);
      }
      return code;
    };
    std::sort(cells.begin(), cells.end(), [&](int a, int b) {
      const int ma = morton(a), mb = morton(b);
      return ma != mb ? ma < mb : a < b;
    });
    state.region_anchor_cell.assign(units, -1);
    state.region_owner_by_cell.assign(
        env.config().board_size * env.config().board_size, -1);
    if (!cells.empty()) {
      std::vector<int> anchor_by_region(units, cells.front());
      for (int region = 0; region < units; ++region) {
        const int at = std::min<int>(
            cells.size() - 1,
            ((2 * region + 1) * int(cells.size())) / (2 * units));
        anchor_by_region[region] = cells[at];
      }
      std::vector<int> unit_for_region(units, -1);
      if ((genome_.region_assignment_mode > 0.0 || candidate_region) &&
          units > 1) {
        // Hungarian assignment, run only when the daily region map changes.
        // Rows are live workers and columns are spatial anchors.
        const int n = units;
        std::vector<int> u(n + 1), v(n + 1), p(n + 1), way(n + 1);
        for (int i = 1; i <= n; ++i) {
          p[0] = i;
          int j0 = 0;
          std::vector<int> minv(n + 1, 1 << 28);
          std::vector<uint8_t> used(n + 1, uint8_t(0));
          do {
            used[j0] = uint8_t(1);
            const int i0 = p[j0];
            int delta = 1 << 28, j1 = 0;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
              const int cell = anchor_by_region[j - 1];
              const Position anchor{
                  int16_t(cell % env.config().board_size),
                  int16_t(cell / env.config().board_size)};
              const int cur = distance(positions[i0 - 1], anchor) -
                  u[i0] - v[j];
              if (cur < minv[j]) {
                minv[j] = cur;
                way[j] = j0;
              }
              if (minv[j] < delta) {
                delta = minv[j];
                j1 = j;
              }
            }
            for (int j = 0; j <= n; ++j) {
              if (used[j]) {
                u[p[j]] += delta;
                v[j] -= delta;
              } else {
                minv[j] -= delta;
              }
            }
            j0 = j1;
          } while (p[j0] != 0);
          do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
          } while (j0 != 0);
        }
        for (int region = 0; region < n; ++region)
          unit_for_region[region] = p[region + 1] - 1;
      } else {
        for (int region = 0; region < units; ++region)
          unit_for_region[region] = region;
      }
      for (int region = 0; region < units; ++region) {
        const int unit = std::clamp(unit_for_region[region], 0, units - 1);
        state.region_anchor_cell[unit] = int16_t(anchor_by_region[region]);
      }
      if (genome_.workload_region_mode > 0.0 || candidate_region) {
        const int total_work = std::accumulate(
            workload.begin(), workload.end(), 0);
        int completed_work = 0;
        for (const int cell : cells) {
          const int weight = std::max(1, workload[cell]);
          const int region = std::clamp(
              ((2 * completed_work + weight) * units) /
                  std::max(1, 2 * total_work),
              0, units - 1);
          state.region_owner_by_cell[cell] = int8_t(
              std::clamp(unit_for_region[region], 0, units - 1));
          completed_work += weight;
        }
      }
    }
    state.region_anchor_day = int16_t(env.day());
  }

  std::vector<Action> actions(units, Action{});
  std::vector<int> assigned(units, -1);
  std::vector<bool> used(tasks.size(), false);
  // Some atomic operations exclusively mutate the occupancy of a map cell.
  // Distinct task records can still point at the same cell (for example a cow
  // and a sheep placement candidate), so task-index reservation alone is not
  // sufficient.  Reserve those physical targets across the whole unit batch.
  // FEED and CARE are intentionally excluded: two workers may legally perform
  // the two different maintenance actions on the same animal in one step.
  const int board_cells = env.config().board_size * env.config().board_size;
  std::vector<bool> used_exclusive_cell(board_cells, false);
  std::vector<bool> used_routine_cell(board_cells, false);
  auto requires_exclusive_cell = [](Op op) {
    return op == Op::BUILD_PASTURE || op == Op::BUILD_COOP ||
           op == Op::PLANT || op == Op::PLACE || op == Op::DIG;
  };
  auto exclusive_cell = [&](const AdaptiveTask& task) {
    return pos_index(task.target, env.config().board_size);
  };
  std::array<int, N_ITEMS> reserved_pickup{};
  int reserved_drop_capacity = 0;
  int reserved_final_collection_capacity = 0;
  const int available_drop_capacity =
      std::max(0, env.config().shed_capacity - shed_sum(priv));
  int carried_total = 0;
  for (const auto& inventory : priv.inventories)
    carried_total += inventory_sum(inventory);
  int pending_feed_tasks = 0;
  for (const auto& tile : farm.tiles) {
    pending_feed_tasks +=
        tile.kind == TileKind::ANIMAL && !tile.fed_today ? 1 : 0;
  }
  const bool end_of_day_capacity_risk = env.hour() >= 14 &&
      shed_sum(priv) + carried_total >
          env.config().shed_capacity - units;
  std::array<int, N_PRODUCTS> opponent_ready_supply{};
  for (const auto& tile : env.farms()[1 - player].tiles) {
    if (tile.yield_units > 0) {
      if (tile.kind == TileKind::PLANT) {
        const int item = int(tile.crop);
        if (item >= 0 && item < N_PRODUCTS)
          opponent_ready_supply[item] += tile.yield_units;
      } else if (tile.kind == TileKind::ANIMAL) {
        const int animal = int(tile.animal) - 9;
        if (animal >= 0 && animal < N_ANIMALS)
          opponent_ready_supply[ANIMAL_PRODUCT[animal]] += tile.yield_units;
      }
    }
    if (tile.kind == TileKind::ANIMAL && tile.fertilizer_available)
      opponent_ready_supply[8]++;
  }

  // A worker may be carrying the cash source needed to execute the current
  // economic plan.  Looking only at farm.money missed this dependency: the
  // planner would keep valuable wool/milk in the field until automatic EOD
  // return, while land, animals and seeds remained unfunded for a whole day.
  // Estimate the current public-state commitment bill and return only enough
  // high-value cargo to close its liquid-cash gap.
  const auto counts = counts_for(env, player);
  double liquid_cash = farm.money;
  for (int item = 0; item < N_PRODUCTS; ++item)
    liquid_cash += priv.shed[item] * env.market().prices[item];
  double immediate_commitment_bill = 0.0;
  const int hand_deficit = std::max(
      0, int(state.plan.hand_target) - int(farm.hands.size()));
  immediate_commitment_bill += std::max(
      0, daily_hire_cost(farm.hires_today + hand_deficit) -
             daily_hire_cost(farm.hires_today));
  int quadrants = popcount(farm.unlocked_mask);
  for (int quadrant = quadrants;
       quadrant < int(state.plan.quadrant_target) && quadrant <= 3;
       ++quadrant)
    immediate_commitment_bill += LAND_COST[quadrant - 1];
  for (int animal = 0; animal < N_ANIMALS; ++animal)
    immediate_commitment_bill += std::max(
        0, int(state.plan.animal_targets[animal]) -
               counts.total_animals[animal]) * ANIMAL_COST[animal];
  // Do not force a long shed trip merely to finance every replaceable seed in
  // today's aspirational flow.  Crop work can be delayed or resized without
  // stranding irreversible capital; land, animals, hands and feed cannot.
  const int owned_animals = std::accumulate(
      counts.total_animals.begin(), counts.total_animals.end(), 0);
  int physical_wheat = priv.shed[0];
  for (const auto& inventory : priv.inventories) physical_wheat += inventory[0];
  immediate_commitment_bill += std::max(
      0, std::max(owned_animals, int(state.plan.wheat_buffer)) -
             physical_wheat) * env.market().prices[0];
  const double financing_shortfall = std::max(
      0.0, immediate_commitment_bill + genome_.cash_reserve - liquid_cash);
  double reserved_drop_value = 0.0;

  auto immediate_collection_units = [&](int unit,
                                        const AdaptiveTask& task) {
    if (task.action.op != Op::HARVEST &&
        task.action.op != Op::COLLECT_FERTILIZER) return 0;
    if (unit < 0 || unit >= units ||
        !same(positions[unit], task.target)) return 0;
    const int cell = pos_index(task.target, env.config().board_size);
    if (cell < 0 || cell >= int(farm.tiles.size())) return 0;
    const auto& tile = farm.tiles[cell];
    if (task.action.op == Op::HARVEST)
      return std::max(0, int(tile.yield_units));
    return tile.fertilizer_available ? 1 : 0;
  };

  auto eligible = [&](int unit, const AdaptiveTask& task) {
    static const std::array<int32_t, N_ITEMS> zero{};
    const auto& inv = unit < int(priv.inventories.size()) ? priv.inventories[unit] : zero;
    if (task.action.op == Op::DROP) {
      int product_units = 0;
      double cargo_value = 0.0;
      bool visible_race = false;
      bool strong_market = false;
      for (int item = 0; item < N_PRODUCTS; ++item) {
        product_units += inv[item];
        cargo_value += inv[item] * env.market().prices[item];
        visible_race = visible_race ||
            (inv[item] > 0 && opponent_ready_supply[item] > 0);
        strong_market = strong_market ||
            (inv[item] > 0 &&
             env.market().prices[item] >= int(1.08 * BASE_PRICE[item]));
      }
      if (product_units <= 0) return false;
      // Cargo returns automatically at the end of every day.  A mid-day DROP
      // is useful only when it can finance a cash-starved plan or during the
      // liquidation phase.  A late capacity-risk return is the third case: it
      // gives the market compiler several steps to sell shed stock, make room,
      // and evacuate only the cargo that would otherwise overflow at day end.
      // Outside these cases DROP burns two long trips for no economic gain and
      // breaks the worker's local task chain.
      const bool financing_needed =
          farm.money < genome_.cash_reserve ||
          reserved_drop_value + 1e-9 < financing_shortfall;
      const bool liquidation = env.day() >= liquidation_day;
      const bool valuable_cargo = cargo_value >= genome_.drop_value_threshold;
      const bool preemption = product_units >= genome_.preempt_quantity &&
          (visible_race || strong_market);
      return financing_needed || liquidation || end_of_day_capacity_risk ||
             valuable_cargo || preemption;
    }
    // Once the late-day evacuation guard is active, do not create more cargo
    // than the shed plus all carriers can return at EOD.  The former hour-23
    // check was too late: several workers could harvest between hours 14-22,
    // fill their inventories beyond total capacity and remain too far from a
    // shed to repair the overflow on the final step.  This still rejects only
    // an immediately executing atomic collection; travelling toward a tile
    // and output left safely on the map remain legal.
    // Product left on the map is recoverable; product collected beyond total
    // shed-plus-carrier capacity is not.  Guard every immediately executing
    // collection, not only hour 14+, because a large early parallel harvest
    // can already make the end-of-day overflow unavoidable if later market
    // demand is weak.  Sales that create space simply make the task eligible
    // again on the next step.
    if (task.action.op == Op::HARVEST ||
        task.action.op == Op::COLLECT_FERTILIZER) {
      const int added = immediate_collection_units(unit, task);
      if (added > 0 && shed_sum(priv) + carried_total +
              reserved_final_collection_capacity + added >
              env.config().shed_capacity)
        return false;
    }
    if (task.action.op == Op::PICKUP) {
      const int item = int(task.action.item);
      return item >= 0 && item < N_ITEMS &&
             reserved_pickup[item] < priv.shed[item] && inv[item] == 0;
    }
    if (task.required_item >= 0 && (task.required_item >= N_ITEMS || inv[task.required_item] <= 0))
      return false;
    return true;
  };

  // Compact one-step route lookahead.  Count useful non-logistics tasks in a
  // three-cell Manhattan neighbourhood so the scheduler can prefer a district
  // where the worker will have follow-up work.  This is state-derived every
  // step and does not encode a crop type, replay coordinate, or worker lane.
  auto is_productive_local_task = [](Op op) {
    return op != Op::PASS && op != Op::PICKUP && op != Op::DROP;
  };
  std::vector<int> route_density(tasks.size(), 0);
  if (genome_.route_density_scale > 0.0 ||
      candidate_schedule == ScheduleProfile::VALUE_PER_STEP ||
      candidate_chain) {
    std::vector<int> productive_per_cell(board_cells, 0);
    for (const auto& task : tasks)
      if (is_productive_local_task(task.action.op))
        productive_per_cell[pos_index(task.target, env.config().board_size)]++;
    for (int t = 0; t < int(tasks.size()); ++t) {
      if (!is_productive_local_task(tasks[t].action.op)) continue;
      const Position center = tasks[t].target;
      for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx) {
        const int d = std::abs(dx) + std::abs(dy);
        const int x = center.x + dx, y = center.y + dy;
        if (d > 3 || x < 0 || y < 0 || x >= env.config().board_size ||
            y >= env.config().board_size) continue;
        route_density[t] += (4 - d) *
            productive_per_cell[y * env.config().board_size + x];
      }
      // Remove this task's self-count; same-cell follow-up operations remain.
      route_density[t] = std::max(0, route_density[t] - 4);
    }
  }

  std::vector<int> hard_nearest_distance(tasks.size(), 0);
  if (genome_.hard_latest_start_reservation > 0.0 ||
      candidate_schedule == ScheduleProfile::DEADLINE_FIRST) {
    for (int t = 0; t < int(tasks.size()); ++t) {
      if (tasks[t].priority < 950) continue;
      int nearest = 2 * env.config().board_size;
      for (int u = 0; u < units; ++u)
        if (eligible(u, tasks[t]))
          nearest = std::min(
              nearest, distance(positions[u], tasks[t].target));
      hard_nearest_distance[t] = nearest;
    }
  }

  auto semantic_group = [&](const AdaptiveTask& task) {
    if (task.action.op == Op::PLANT) {
      const int crop = int(task.action.item);
      return crop >= 0 && crop < N_CROPS ? crop : -1;
    }
    if (task.action.op == Op::PLACE) {
      const int animal = int(task.action.item) - 9;
      if (animal >= 0 && animal < N_ANIMALS) return 16 + animal;
    }
    const auto& tile = farm.tiles[pos_index(task.target, env.config().board_size)];
    if (tile.kind == TileKind::PLANT) {
      const int crop = int(tile.crop);
      return crop >= 0 && crop < N_CROPS ? crop : -1;
    }
    if (tile.kind == TileKind::ANIMAL) {
      const int animal = int(tile.animal) - 9;
      return animal >= 0 && animal < N_ANIMALS ? 16 + animal : -1;
    }
    return -1;
  };

  // R8 keeps a compact, public-state task ledger for the current day.  It is
  // intentionally reconciled from live tasks on every step: a vanished task
  // releases its owner immediately, while a still-valid obligation preserves
  // its age, deadline and soft ownership.  This is task memory, not an action
  // tape; no worker, coordinate or operation is forced after the state changes.
  std::vector<int> r8_node_for_task(tasks.size(), -1);
  if (genome_.r8_execution_enabled) {
    if (state.r8_day_plan_day != env.day()) {
      state.r8_day_plan_day = int16_t(env.day());
      state.r8_day_plan_start_step = int16_t(env.step_count());
      state.r8_day_plan_end_step = int16_t(std::min(
          719, env.step_count() +
              std::max(1, genome_.r8_day_horizon_steps) - 1));
      state.r8_task_nodes.clear();
      state.r8_day_plan_rebuilds++;
    }
    std::vector<uint8_t> r8_previously_active(
        state.r8_task_nodes.size(), uint8_t(0));
    for (int n = 0; n < int(state.r8_task_nodes.size()); ++n)
      r8_previously_active[n] = state.r8_task_nodes[n].active != 0;
    for (auto& node : state.r8_task_nodes) node.active = 0;
    for (int t = 0; t < int(tasks.size()); ++t) {
      const auto& task = tasks[t];
      const int cell = pos_index(task.target, env.config().board_size);
      int found = -1;
      for (int n = 0; n < int(state.r8_task_nodes.size()); ++n) {
        const auto& node = state.r8_task_nodes[n];
        if (node.cell == cell &&
            node.reservation_key == task.reservation_key &&
            node.op == int8_t(task.action.op)) {
          found = n;
          break;
        }
      }
      if (found < 0) {
        R8TaskNode node;
        node.id = state.r8_next_task_id++;
        node.cell = int16_t(cell);
        node.reservation_key = int16_t(task.reservation_key);
        node.first_seen_step = int16_t(env.step_count());
        node.last_seen_step = int16_t(env.step_count());
        node.deadline_step = int16_t(std::clamp(task.deadline_step, 0, 719));
        node.priority = int16_t(std::clamp(task.priority, -32768, 32767));
        node.op = int8_t(task.action.op);
        node.semantic_group = int8_t(semantic_group(task));
        node.active = 1;
        node.expected_cash_gain = task.expected_cash_gain;
        node.loss_if_delayed = task.loss_if_delayed;
        state.r8_task_nodes.push_back(node);
        found = int(state.r8_task_nodes.size()) - 1;
        state.r8_task_nodes_created++;
      } else {
        auto& node = state.r8_task_nodes[found];
        node.last_seen_step = int16_t(env.step_count());
        node.deadline_step = int16_t(std::clamp(task.deadline_step, 0, 719));
        node.priority = int16_t(std::clamp(task.priority, -32768, 32767));
        node.semantic_group = int8_t(semantic_group(task));
        node.active = 1;
        node.expected_cash_gain = task.expected_cash_gain;
        node.loss_if_delayed = task.loss_if_delayed;
      }
      r8_node_for_task[t] = found;
    }
    int active_nodes = 0;
    for (int n = 0; n < int(state.r8_task_nodes.size()); ++n) {
      const auto& node = state.r8_task_nodes[n];
      active_nodes += node.active != 0;
      if (n < int(r8_previously_active.size()) &&
          r8_previously_active[n] && !node.active) {
        const int latency = std::clamp(
            env.step_count() - int(node.first_seen_step), 0, 24);
        state.r8_resolution_latency_histogram[latency]++;
        state.r8_resolved_task_nodes++;
      }
    }
    state.r8_peak_active_tasks = std::max(
        state.r8_peak_active_tasks, active_nodes);
    state.r8_rolling_updates++;
  }

  std::vector<std::vector<double>> r8_lookahead_bonus(
      units, std::vector<double>(tasks.size(), 0.0));
  // R8 is a conservative refinement of the established R6 task selection.
  // Keep the ordinary assignment score bit-for-bit identical while R6 picks
  // which jobs are worth doing.  The persistent-plan and short-horizon terms
  // are enabled only inside the later exact worker-permutation pass.
  bool r8_refinement_scoring = false;

  auto pair_score = [&](int unit, int task_index) {
    const auto& task = tasks[task_index];
    static const std::array<int32_t, N_ITEMS> zero{};
    const auto& inv = unit < int(priv.inventories.size())
        ? priv.inventories[unit] : zero;
    const int dist = distance(positions[unit], task.target);
    const int slack = std::max(0, task.deadline_step - env.step_count());
    // Preserve absolute precedence for hard survival/expiry work, but compress
    // routine priorities so a small semantic rank difference cannot justify a
    // map-wide trip.  This lets travel cost and task continuity choose among
    // otherwise safe maintenance, harvest, and logistics work.
    const double priority_score = task.priority >= 950
        ? 10.0 * task.priority
        : 6000.0 + genome_.routine_priority_scale * (task.priority - 600);
    double score = priority_score - genome_.move_cost * dist;
    // Candidate-only scheduling overlays.  Every profile modifies the same
    // legal unit-task graph; none introduces route coordinates or bypasses
    // hard deadline/resource checks.
    if (candidate_schedule == ScheduleProfile::MIN_TOTAL_TRAVEL)
      score -= 3.0 * genome_.move_cost * dist;
    else if (candidate_schedule == ScheduleProfile::DEADLINE_FIRST)
      score += 4.0 * genome_.deadline_weight / double(1 + slack);
    else if (candidate_schedule == ScheduleProfile::VALUE_PER_STEP)
      score += 2.0 * (task.expected_cash_gain + task.loss_if_delayed) /
          double(1 + dist);
    else if (candidate_chain)
      score += 0.5 * genome_.action_cost * route_density[task_index];
    if (genome_.region_ownership_scale > 0.0 && task.priority < 950 &&
        env.day() < liquidation_day) {
      const int target_cell =
          pos_index(task.target, env.config().board_size);
      if (genome_.workload_region_mode > 0.0 &&
          target_cell < int(state.region_owner_by_cell.size()) &&
          state.region_owner_by_cell[target_cell] >= 0) {
        const int owner = state.region_owner_by_cell[target_cell];
        score += (owner == unit ? 1.0 : -1.0) *
                 genome_.region_ownership_scale * genome_.task_stickiness;
      } else if (unit < int(state.region_anchor_cell.size()) &&
                 state.region_anchor_cell[unit] >= 0) {
        const int anchor_cell = state.region_anchor_cell[unit];
        const Position anchor{
            int16_t(anchor_cell % env.config().board_size),
            int16_t(anchor_cell / env.config().board_size)};
        score -= genome_.region_ownership_scale * genome_.move_cost *
                 distance(anchor, task.target);
      }
    }
    if (genome_.region_anchor_distance_scale > 0.0 &&
        task.priority < 950 &&
        env.day() < liquidation_day &&
        unit < int(state.region_anchor_cell.size()) &&
        state.region_anchor_cell[unit] >= 0) {
      const int anchor_cell = state.region_anchor_cell[unit];
      const Position anchor{
          int16_t(anchor_cell % env.config().board_size),
          int16_t(anchor_cell / env.config().board_size)};
      score -= genome_.region_anchor_distance_scale * genome_.move_cost *
               distance(anchor, task.target);
    }
    // Monetary value breaks otherwise similar scheduling choices.  It must
    // remain an auxiliary signal: raw cash is much larger than one unit of
    // scheduling priority and would otherwise crowd out daily maintenance.
    score += genome_.task_value_scale *
             (task.expected_cash_gain + task.loss_if_delayed);
    score += genome_.deadline_weight / double(1 + slack);
    if (task.priority >= 950 && genome_.hard_travel_slack_scale > 0.0) {
      const int travel_slack = std::max(
          0, task.deadline_step - env.step_count() - dist);
      score += genome_.hard_travel_slack_scale * 10.0 *
               genome_.deadline_weight / double(1 + travel_slack);
    }
    if (task.priority >= 950 &&
        (genome_.hard_latest_start_reservation > 0.0 ||
         candidate_schedule == ScheduleProfile::DEADLINE_FIRST)) {
      const int latest_start_margin = slack - hard_nearest_distance[task_index];
      if (latest_start_margin <= 1) {
        const double reservation_strength = std::max(
            genome_.hard_latest_start_reservation,
            candidate_schedule == ScheduleProfile::DEADLINE_FIRST ? 1.0 : 0.0);
        score += reservation_strength *
            (latest_start_margin <= 0 ? 30000.0 : 15000.0);
        // A farther worker cannot complete the task by the deadline when the
        // nearest one is already at the latest-start boundary.
        if (dist > slack) score -= 60000.0;
      }
    }
    score += genome_.route_density_scale * genome_.action_cost *
             route_density[task_index];
    // An animal carried by a worker is already-paid capital that produces
    // nothing.  Treat PLACE as the continuation of that open commitment and
    // discourage unrelated routine work until it is closed.  Survival/expiry
    // tasks (priority >= 950) may still interrupt it.  This is state based and
    // applies to every day, animal mix, and layout; it is not an opening script.
    int carried_animal = -1;
    for (int animal = 0; animal < N_ANIMALS; ++animal)
      if (inv[9 + animal] > 0) { carried_animal = animal; break; }
    if (carried_animal >= 0) {
      const bool closes_commitment = task.action.op == Op::PLACE &&
          int(task.action.item) == 9 + carried_animal;
      if (closes_commitment)
        score += 20.0 * genome_.task_stickiness;
      else if (task.priority < 950)
        score -= 8.0 * genome_.task_stickiness;
    }
    // A small carried resource batch is an open service commitment.  Keep the
    // carrier on the corresponding local chain instead of sending it to an
    // unrelated routine task and creating a second shed trip later.  Survival
    // work remains free to interrupt because its priority is >= 950.
    const double prerequisite_strength = genome_.prerequisite_chain_strength;
    if (prerequisite_strength > 0.0 && inv[0] > 0 &&
        pending_feed_tasks > 0) {
      const bool feeds = task.action.op == Op::FEED;
      const auto& target_tile =
          farm.tiles[pos_index(task.target, env.config().board_size)];
      const bool same_animal_chain = target_tile.kind == TileKind::ANIMAL &&
          (task.action.op == Op::CARE ||
           task.action.op == Op::COLLECT_FERTILIZER ||
           task.action.op == Op::HARVEST);
      if (feeds)
        score += prerequisite_strength * 8.0 * genome_.task_stickiness;
      else if (!same_animal_chain && task.priority < 950)
        score -= prerequisite_strength * 4.0 * genome_.task_stickiness;
    }
    if (prerequisite_strength > 0.0 && inv[8] > 0) {
      if (task.action.op == Op::FERTILIZE)
        score += prerequisite_strength * 4.0 * genome_.task_stickiness;
      else if (task.priority < 950 && task.action.op != Op::WATER &&
               task.action.op != Op::HARVEST)
        score -= prerequisite_strength * 2.0 * genome_.task_stickiness;
    }
    if (task.action.op == Op::DROP && unit < int(priv.inventories.size())) {
      double cargo_value = 0.0;
      for (int item = 0; item < N_PRODUCTS; ++item)
        cargo_value += priv.inventories[unit][item] * env.market().prices[item];
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        cargo_value += priv.inventories[unit][9 + animal] * ANIMAL_COST[animal];
      score += genome_.task_value_scale * cargo_value;
    }
    if (state.sticky_target[unit] == task.reservation_key) {
      // Finish the useful action chain at one location before starting a
      // new map-wide sweep.  Hard deadlines still dominate via priority.
      score += 16.0 * genome_.task_stickiness;
      if (state.sticky_op[unit] == int8_t(task.action.op))
        score += genome_.task_stickiness;
      if (candidate_chain) score += 12.0 * genome_.task_stickiness;
    }
    if (state.sticky_cell[unit] >= 0) {
      const Position anchor{
          int16_t(state.sticky_cell[unit] % env.config().board_size),
          int16_t(state.sticky_cell[unit] / env.config().board_size)};
      const int local_distance = distance(anchor, task.target);
      // A task chain is spatial as well as semantic: after completing one
      // tile, keep the worker in the same small production district whenever
      // deadlines allow.  This is a soft score, so HARD survival tasks can
      // still pull any unit across the map.
      score += genome_.task_stickiness * std::max(0, 4 - local_distance);
      if (task.priority < 950 && genome_.quadrant_affinity_scale > 0.0) {
        const int anchor_quadrant =
            (anchor.y < 5 ? 0 : 2) + (anchor.x < 5 ? 0 : 1);
        const int target_quadrant =
            (task.target.y < 5 ? 0 : 2) + (task.target.x < 5 ? 0 : 1);
        if (anchor_quadrant == target_quadrant)
          score += genome_.quadrant_affinity_scale * genome_.task_stickiness;
      }
    }
    const int group = semantic_group(task);
    if (group >= 0 && state.sticky_group[unit] == group &&
        task.priority < 950)
      score += genome_.industry_affinity_scale * genome_.task_stickiness;
    if (candidate_chain && group >= 0 &&
        state.sticky_group[unit] == group && task.priority < 950)
      score += 8.0 * genome_.task_stickiness;
    if (state.plan.candidate8_recovery_profile ==
        int8_t(RecoveryProfile::REPAIR_NOW)) {
      const bool relevant_weed =
          (state.plan.candidate8_recovery_issue & RECOVERY_WEED) &&
          task.action.op == Op::DIG;
      const bool relevant_chain =
          (state.plan.candidate8_recovery_issue & RECOVERY_BROKEN_CHAIN) &&
          task.priority >= 950;
      if (relevant_weed || relevant_chain) score += 25000.0;
    } else if (state.plan.candidate8_recovery_profile ==
                   int8_t(RecoveryProfile::DEFER_LOW_VALUE) &&
               task.priority < 950) {
      score -= std::max(0.0, 200.0 - task.expected_cash_gain);
    } else if (state.plan.candidate8_recovery_profile ==
                   int8_t(RecoveryProfile::ABANDON_LOW_VALUE) &&
               task.priority < 950 && task.expected_cash_gain <= 0.0) {
      score -= 50000.0;
    }
    if (r8_refinement_scoring && genome_.r8_execution_enabled &&
        genome_.r8_feature_level >= 3 &&
        unit >= 0 && unit < int(r8_lookahead_bonus.size()) &&
        task_index >= 0 &&
        task_index < int(r8_lookahead_bonus[unit].size())) {
      score += r8_lookahead_bonus[unit][task_index];
      const int node_index = r8_node_for_task[task_index];
      if (node_index >= 0 &&
          node_index < int(state.r8_task_nodes.size())) {
        const auto& node = state.r8_task_nodes[node_index];
        const int age = std::max(
            0, env.step_count() - int(node.first_seen_step));
        const int remaining = std::max(
            0, int(node.deadline_step) - env.step_count());
        // Preserve a still-valid commitment without making it uninterruptible.
        // Hard work has already been reserved and can always preempt this soft
        // ownership bonus.
        if (node.assigned_unit == unit && task.priority < 950)
          score += 4.0 * genome_.task_stickiness;
        score += std::min(8, age) * 0.25 * genome_.task_stickiness;
        if (node.deadline_step <= state.r8_day_plan_end_step)
          score += genome_.deadline_weight / double(1 + remaining);
      }
    }
    return score;
  };

  // R8 short-horizon route valuation.  The current selected unit-task batch is
  // refined exactly by the conservative matcher below.  Each legal first edge is
  // augmented by the best bounded 4-8 step continuation through the current
  // public task graph.  Travel and action durations are exact Manhattan/action
  // steps; no hidden future shop/weed RNG is queried.  Only the first action is
  // committed and the graph is rebuilt after the real transition.
  if (genome_.r8_execution_enabled && genome_.r8_feature_level >= 4 &&
      genome_.r8_lookahead_scale > 0.0 && !tasks.empty()) {
    // A dense explicit day calendar already supplies long-range intent; a
    // a deep local horizon there tends to over-count interchangeable chores.  Use
    // four exact steps for that high-density mode and the configured 4-8 step
    // window for autonomous sparse plans.  This is based on plan structure,
    // never opponent identity, replay id, date, coordinate, or route author.
    const int horizon = backbone.enabled
        ? 4 : std::clamp(genome_.r8_joint_horizon_steps, 4, 8);
    const int limit = std::clamp(
        genome_.r8_joint_candidate_limit, 4, 32);
    auto task_utility = [&](const AdaptiveTask& task, int travel,
                            int finish_step) {
      if (task.deadline_step < env.step_count() + finish_step)
        return -1.0e9;
      const double semantic = std::max(0, task.priority - 500);
      const double value = 0.05 *
          (task.expected_cash_gain + task.loss_if_delayed);
      return (semantic + value) / double(1 + travel);
    };
    for (int unit = 0; unit < units; ++unit) {
      std::vector<std::pair<double, int>> ranked;
      ranked.reserve(tasks.size());
      for (int t = 0; t < int(tasks.size()); ++t) {
        if (tasks[t].priority >= 950 || !eligible(unit, tasks[t])) continue;
        ranked.push_back({pair_score(unit, t), t});
      }
      std::stable_sort(ranked.begin(), ranked.end(),
          [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
          });
      if (int(ranked.size()) > limit) ranked.resize(limit);
      const int m = int(ranked.size());
      if (m <= 1) continue;
      uint16_t base_item_mask = 0;
      if (unit < int(priv.inventories.size())) {
        for (int item = 0; item < N_ITEMS; ++item)
          if (priv.inventories[unit][item] > 0)
            base_item_mask |= uint16_t(1) << item;
      }
      const int mask_count = 1 << m;
      std::vector<uint16_t> acquired_item_mask(mask_count, uint16_t(0));
      for (int mask = 1; mask < mask_count; ++mask) {
        int bit = 0;
        while (!(mask & (1 << bit))) ++bit;
        const int previous = mask & (mask - 1);
        acquired_item_mask[mask] = acquired_item_mask[previous];
        const auto& selected = tasks[ranked[bit].second];
        if (selected.action.op == Op::PICKUP) {
          const int item = int(selected.action.item);
          if (item >= 0 && item < N_ITEMS)
            acquired_item_mask[mask] |= uint16_t(1) << item;
        }
      }
      for (int first_local = 0; first_local < m; ++first_local) {
        const int first_task = ranked[first_local].second;
        const int first_travel = distance(
            positions[unit], tasks[first_task].target);
        const int first_elapsed = first_travel + 1;
        if (first_elapsed >= horizon) continue;
        const int first_mask = 1 << first_local;
        double best_followup = 0.0;
        if (genome_.r8_sequence_solver == 0) {
          // High-throughput deterministic beam.  It keeps several distinct
          // partial task chains alive but avoids allocating and scanning the
          // full subset-DP tensor for every possible first edge.  Only the
          // low-level search algorithm changes; candidate tasks, utilities,
          // legality checks and the first-action commit rule remain identical.
          struct BeamState {
            int mask = 0;
            int last = 0;
            int elapsed = 0;
            double score = 0.0;
          };
          std::vector<BeamState> beam{{
              first_mask, first_local, first_elapsed, 0.0}};
          const int beam_width = std::clamp(genome_.r8_beam_width, 1, 64);
          for (int depth = 1; depth < horizon && !beam.empty(); ++depth) {
            std::vector<BeamState> children;
            children.reserve(size_t(beam.size()) * size_t(m));
            for (const auto& current : beam) {
              const uint16_t virtual_item_mask =
                  base_item_mask | acquired_item_mask[current.mask];
              const int last_task = ranked[current.last].second;
              for (int next_local = 0; next_local < m; ++next_local) {
                if (current.mask & (1 << next_local)) continue;
                const int next_task = ranked[next_local].second;
                const auto& task = tasks[next_task];
                if (task.required_item >= 0 &&
                    !(virtual_item_mask &
                      (uint16_t(1) << task.required_item)))
                  continue;
                const int travel = distance(
                    tasks[last_task].target, task.target);
                const int finish = current.elapsed + travel + 1;
                if (finish > horizon) continue;
                const double utility = task_utility(task, travel, finish);
                if (utility <= -1.0e8) continue;
                const double next_score = current.score + utility;
                children.push_back({
                    current.mask | (1 << next_local), next_local,
                    finish, next_score});
                best_followup = std::max(best_followup, next_score);
                state.r8_lookahead_evaluations++;
              }
            }
            std::stable_sort(children.begin(), children.end(),
                [](const BeamState& left, const BeamState& right) {
                  if (left.score != right.score)
                    return left.score > right.score;
                  if (left.elapsed != right.elapsed)
                    return left.elapsed < right.elapsed;
                  if (left.mask != right.mask)
                    return left.mask < right.mask;
                  return left.last < right.last;
                });
            // Duplicate states can be reached through different task orders.
            // Keep only the highest-scoring representative before truncation.
            std::vector<BeamState> unique;
            unique.reserve(std::min<int>(beam_width, children.size()));
            for (const auto& child : children) {
              bool duplicate = false;
              for (const auto& kept : unique) {
                if (kept.mask == child.mask && kept.last == child.last &&
                    kept.elapsed == child.elapsed) {
                  duplicate = true;
                  break;
                }
              }
              if (!duplicate) unique.push_back(child);
              if (int(unique.size()) >= beam_width) break;
            }
            beam = std::move(unique);
          }
        } else {
          constexpr double unreachable = -1.0e100;
          const int stride_elapsed = horizon + 1;
          const int stride_last = m * stride_elapsed;
          std::vector<double> dp(
              mask_count * stride_last, unreachable);
          auto dp_index = [&](int mask, int last, int elapsed) {
            return mask * stride_last + last * stride_elapsed + elapsed;
          };
          dp[dp_index(first_mask, first_local, first_elapsed)] = 0.0;
          // Exact subset DP over the bounded local candidate set.  State is
          // the completed-task mask, last task and elapsed action/travel
          // steps.  Acquired inventory is a deterministic subset function.
          for (int mask = 0; mask < mask_count; ++mask) {
            if (!(mask & first_mask)) continue;
            const uint16_t virtual_item_mask =
                base_item_mask | acquired_item_mask[mask];
            for (int last = 0; last < m; ++last) {
              if (!(mask & (1 << last))) continue;
              for (int elapsed = 1; elapsed <= horizon; ++elapsed) {
                const double current_score =
                    dp[dp_index(mask, last, elapsed)];
                if (current_score <= unreachable / 2) continue;
                for (int next_local = 0; next_local < m; ++next_local) {
                  if (mask & (1 << next_local)) continue;
                  const int next_task = ranked[next_local].second;
                  const auto& task = tasks[next_task];
                  if (task.required_item >= 0 &&
                      !(virtual_item_mask &
                        (uint16_t(1) << task.required_item)))
                    continue;
                  const int last_task = ranked[last].second;
                  const int travel = distance(
                      tasks[last_task].target, task.target);
                  const int finish = elapsed + travel + 1;
                  if (finish > horizon) continue;
                  const double utility = task_utility(task, travel, finish);
                  if (utility <= -1.0e8) continue;
                  const int next_mask = mask | (1 << next_local);
                  const double next_score = current_score + utility;
                  auto& destination =
                      dp[dp_index(next_mask, next_local, finish)];
                  destination = std::max(destination, next_score);
                  best_followup = std::max(best_followup, next_score);
                  state.r8_lookahead_evaluations++;
                }
              }
            }
          }
        }
        // A bounded continuation is a tie-breaker for the exact current
        // matching, not permission to override a truly better immediate job.
        r8_lookahead_bonus[unit][first_task] =
            std::min(6.0 * genome_.task_stickiness,
                     std::max(0.0, genome_.r8_lookahead_scale) *
                         best_followup);
      }
    }
  }

  auto reserve_assignment = [&](int unit, int task_index) {
    assigned[unit] = task_index;
    used[task_index] = true;
    if (requires_exclusive_cell(tasks[task_index].action.op))
      used_exclusive_cell[exclusive_cell(tasks[task_index])] = true;
    if (genome_.routine_cell_exclusivity > 0.0 &&
        tasks[task_index].priority < 950)
      used_routine_cell[exclusive_cell(tasks[task_index])] = true;
    if (tasks[task_index].action.op == Op::PICKUP) {
      const int item = int(tasks[task_index].action.item);
      if (item >= 0 && item < N_ITEMS) {
        const int available = std::max(
            0, priv.shed[item] - reserved_pickup[item]);
        reserved_pickup[item] += std::min(
            available, std::max(1, int(tasks[task_index].action.quantity)));
      }
    }
    reserved_final_collection_capacity +=
        immediate_collection_units(unit, tasks[task_index]);
    if (tasks[task_index].action.op == Op::DROP &&
        unit < int(priv.inventories.size())) {
      // Travelling towards a shed does not consume shed capacity yet.  The
      // old reservation blocked distant carriers from even starting home
      // whenever the shed was temporarily full, which caused the exact EOD
      // overflow that the evacuation task was meant to prevent.
      if (same(positions[unit], tasks[task_index].target))
        reserved_drop_capacity += inventory_sum(priv.inventories[unit]);
      for (int item = 0; item < N_PRODUCTS; ++item)
        reserved_drop_value +=
            priv.inventories[unit][item] * env.market().prices[item];
    }
  };

  auto drop_fits_current_shed = [&](int unit, int task_index) {
    if (tasks[task_index].action.op != Op::DROP) return true;
    // Capacity is checked only when DROP will execute on this step.  A worker
    // that is still travelling reserves a route lane, not storage space; the
    // market compiler can sell existing shed stock before its arrival.
    if (!same(positions[unit], tasks[task_index].target)) return true;
    const int cargo = unit < int(priv.inventories.size())
        ? inventory_sum(priv.inventories[unit]) : 0;
    return reserved_drop_capacity + cargo <= available_drop_capacity;
  };

  auto assign_one = [&](bool hard_only) {
    double best_score = -std::numeric_limits<double>::infinity();
    int best_u = -1, best_t = -1;
    for (int u = 0; u < units; ++u) {
      if (assigned[u] >= 0) continue;
      for (int t = 0; t < int(tasks.size()); ++t) {
        if (used[t] || !eligible(u, tasks[t])) continue;
        if (hard_only && tasks[t].priority < 950) continue;
        if (genome_.routine_cell_exclusivity > 0.0 &&
            tasks[t].priority < 950 &&
            used_routine_cell[exclusive_cell(tasks[t])])
          continue;
        if (requires_exclusive_cell(tasks[t].action.op) &&
            used_exclusive_cell[exclusive_cell(tasks[t])]) continue;
        if (!drop_fits_current_shed(u, t)) continue;
        const double score = pair_score(u, t);
        if (score > best_score) {
          best_score = score; best_u = u; best_t = t;
        }
      }
    }
    if (best_u < 0) return false;
    reserve_assignment(best_u, best_t);
    return true;
  };

  auto lock_local_water = [&](int unit, const AdaptiveTask& task) {
    if (unit < 0 || unit >= units || task.priority < 950 ||
        task.action.op != Op::WATER ||
        !same(positions[unit], task.target)) return false;
    if (task.deadline_step - env.step_count() <= 1) return true;
    const int cell = pos_index(task.target, env.config().board_size);
    if (cell < 0 || cell >= int(farm.tiles.size())) return false;
    const auto& tile = farm.tiles[cell];
    // PLANT creates a same-day watering obligation.  The planter is already
    // at zero travel cost; handing this WATER to a distant worker can leave
    // the tile unserved even when the batch had ample total actions.
    return tile.kind == TileKind::PLANT &&
           int(tile.planted_day) == env.day() && !tile.watered_today;
  };

  // A hard task already underneath a worker has zero travel cost and can be
  // completed this turn.  Reserve those actions before the global matching
  // pass.  Otherwise another worker may claim the tile while the local worker
  // is sent away, converting an immediately avoidable expiry into a deadline
  // miss.  This is a generic feasibility rule, not a route-specific priority.
  for (int u = 0; u < units; ++u) {
    double best_score = -std::numeric_limits<double>::infinity();
    int best_t = -1;
    for (int t = 0; t < int(tasks.size()); ++t) {
      if (used[t] || !lock_local_water(u, tasks[t]) ||
          !eligible(u, tasks[t]))
        continue;
      if (requires_exclusive_cell(tasks[t].action.op) &&
          used_exclusive_cell[exclusive_cell(tasks[t])])
        continue;
      if (!drop_fits_current_shed(u, t)) continue;
      const double score = pair_score(u, t);
      if (score > best_score) {
        best_score = score;
        best_t = t;
      }
    }
    if (best_t >= 0) reserve_assignment(u, best_t);
  }

  // Hard survival/expiry tasks are capacity reservations, not merely large
  // soft scores.  Lock them first so stickiness, cargo value or region affinity
  // can never consume the last feasible worker lane.
  for (int pick = 0; pick < units && assign_one(true); ++pick) {}

  // Receding-horizon task ownership.  Once a worker has started travelling to
  // a still-valid routine target, keep that target with the worker instead of
  // allowing the next global matching pass to hand it to somebody else.  The
  // hard pass above deliberately runs first, so survival/expiry work can still
  // preempt any route.  All resource, cell and shed reservations are checked
  // exactly as in the ordinary matcher.
  if (genome_.enroute_task_reservation > 0.0 || candidate_chain) {
    for (int u = 0; u < units; ++u) {
      if (assigned[u] >= 0 || state.sticky_target[u] < 0) continue;
      double best_score = -std::numeric_limits<double>::infinity();
      int best_t = -1;
      for (int t = 0; t < int(tasks.size()); ++t) {
        if (used[t] || tasks[t].priority >= 950 ||
            tasks[t].reservation_key != state.sticky_target[u] ||
            !eligible(u, tasks[t]))
          continue;
        if (genome_.routine_cell_exclusivity > 0.0 &&
            used_routine_cell[exclusive_cell(tasks[t])])
          continue;
        if (requires_exclusive_cell(tasks[t].action.op) &&
            used_exclusive_cell[exclusive_cell(tasks[t])])
          continue;
        if (!drop_fits_current_shed(u, t)) continue;
        const double score = pair_score(u, t);
        if (score > best_score) {
          best_score = score;
          best_t = t;
        }
      }
      if (best_t >= 0) reserve_assignment(u, best_t);
    }
  }

  // Stable local-work pass.  The owner map is generated from current public
  // task density rather than replay coordinates.  It is a preference pass,
  // not a hard partition: ineligible or missing local work falls through to
  // the unrestricted matcher, and all hard work was already assigned above.
  if ((genome_.region_owner_first > 0.0 || candidate_region) &&
      int(state.region_owner_by_cell.size()) == board_cells) {
    for (int u = 0; u < units; ++u) {
      if (assigned[u] >= 0) continue;
      double best_score = -std::numeric_limits<double>::infinity();
      int best_t = -1;
      for (int t = 0; t < int(tasks.size()); ++t) {
        if (used[t] || tasks[t].priority >= 950 ||
            !eligible(u, tasks[t]))
          continue;
        const int cell = exclusive_cell(tasks[t]);
        if (state.region_owner_by_cell[cell] != u) continue;
        if (genome_.routine_cell_exclusivity > 0.0 &&
            used_routine_cell[cell])
          continue;
        if (requires_exclusive_cell(tasks[t].action.op) &&
            used_exclusive_cell[cell])
          continue;
        if (!drop_fits_current_shed(u, t)) continue;
        const double score = pair_score(u, t);
        if (score > best_score) {
          best_score = score;
          best_t = t;
        }
      }
      if (best_t >= 0) reserve_assignment(u, best_t);
    }
  }

  // Maximum-total-score routine matching.  Each physical cell is one column
  // when routine cell exclusivity is active; the unit-specific best legal
  // action on that cell supplies the edge score.  Dummy columns allow PASS.
  // The selected edges are revalidated in deterministic unit order because
  // PICKUP stock and DROP capacity are shared resources.  Any edge invalidated
  // by an earlier reservation is simply left for the established greedy
  // fallback below.
  if (genome_.global_routine_matching > 0.0 || candidate_global) {
    std::vector<int> remaining_units;
    for (int u = 0; u < units; ++u)
      if (assigned[u] < 0) remaining_units.push_back(u);

    std::vector<int> slot_keys;
    std::vector<std::vector<int>> slot_tasks;
    for (int t = 0; t < int(tasks.size()); ++t) {
      if (used[t] || tasks[t].priority >= 950) continue;
      const int key = genome_.routine_cell_exclusivity > 0.0
          ? exclusive_cell(tasks[t]) : t;
      auto found = std::find(slot_keys.begin(), slot_keys.end(), key);
      int slot = 0;
      if (found == slot_keys.end()) {
        slot = int(slot_keys.size());
        slot_keys.push_back(key);
        slot_tasks.push_back({});
      } else {
        slot = int(found - slot_keys.begin());
      }
      slot_tasks[slot].push_back(t);
    }

    const int n = int(remaining_units.size());
    const int real_slots = int(slot_tasks.size());
    const int m = std::max(n, real_slots);
    if (n > 0 && m > 0) {
      constexpr double impossible = -1.0e12;
      std::vector<std::vector<double>> score(
          n, std::vector<double>(m, 0.0));
      std::vector<std::vector<int>> chosen_task(
          n, std::vector<int>(m, -1));
      for (int row = 0; row < n; ++row) {
        const int u = remaining_units[row];
        for (int slot = 0; slot < real_slots; ++slot) {
          double best = impossible;
          int best_t = -1;
          for (const int t : slot_tasks[slot]) {
            if (!eligible(u, tasks[t])) continue;
            if (requires_exclusive_cell(tasks[t].action.op) &&
                used_exclusive_cell[exclusive_cell(tasks[t])])
              continue;
            if (genome_.routine_cell_exclusivity > 0.0 &&
                used_routine_cell[exclusive_cell(tasks[t])])
              continue;
            if (!drop_fits_current_shed(u, t)) continue;
            const double candidate = pair_score(u, t);
            if (candidate > best) {
              best = candidate;
              best_t = t;
            }
          }
          score[row][slot] = best;
          chosen_task[row][slot] = best_t;
        }
      }

      // Rectangular Hungarian algorithm, minimizing -score.  Columns include
      // zero-cost dummy PASS lanes, so a unit is never forced onto an
      // impossible real edge.
      std::vector<double> u_potential(n + 1), v_potential(m + 1);
      std::vector<int> column_row(m + 1), way(m + 1);
      for (int i = 1; i <= n; ++i) {
        column_row[0] = i;
        int j0 = 0;
        std::vector<double> minv(
            m + 1, std::numeric_limits<double>::infinity());
        std::vector<uint8_t> column_used(m + 1, uint8_t(0));
        do {
          column_used[j0] = uint8_t(1);
          const int i0 = column_row[j0];
          double delta = std::numeric_limits<double>::infinity();
          int j1 = 0;
          for (int j = 1; j <= m; ++j) if (!column_used[j]) {
            const double edge_score = score[i0 - 1][j - 1];
            const double cost = edge_score <= impossible / 2
                ? 1.0e12 : -edge_score;
            const double cur = cost - u_potential[i0] - v_potential[j];
            if (cur < minv[j]) {
              minv[j] = cur;
              way[j] = j0;
            }
            if (minv[j] < delta) {
              delta = minv[j];
              j1 = j;
            }
          }
          for (int j = 0; j <= m; ++j) {
            if (column_used[j]) {
              u_potential[column_row[j]] += delta;
              v_potential[j] -= delta;
            } else {
              minv[j] -= delta;
            }
          }
          j0 = j1;
        } while (column_row[j0] != 0);
        do {
          const int j1 = way[j0];
          column_row[j0] = column_row[j1];
          j0 = j1;
        } while (j0 != 0);
      }

      std::vector<int> row_slot(n, -1);
      for (int j = 1; j <= m; ++j)
        if (column_row[j] > 0)
          row_slot[column_row[j] - 1] = j - 1;
      for (int row = 0; row < n; ++row) {
        const int u = remaining_units[row];
        const int slot = row_slot[row];
        if (slot < 0 || slot >= real_slots) continue;
        const int t = chosen_task[row][slot];
        if (t < 0 || used[t] || !eligible(u, tasks[t])) continue;
        if (requires_exclusive_cell(tasks[t].action.op) &&
            used_exclusive_cell[exclusive_cell(tasks[t])])
          continue;
        if (genome_.routine_cell_exclusivity > 0.0 &&
            used_routine_cell[exclusive_cell(tasks[t])])
          continue;
        if (!drop_fits_current_shed(u, t)) continue;
        reserve_assignment(u, t);
      }
    }
  }
  // Fill all remaining worker lanes with the best routine unit-task pairs.
  for (int pick = 0; pick < units && assign_one(false); ++pick) {}

  // One pair-swap pass reduces avoidable total movement without changing task
  // priority or deadlines.
  for (int a = 0; a < units; ++a) for (int b = a + 1; b < units; ++b) {
    if (assigned[a] < 0 || assigned[b] < 0) continue;
    const auto& ta = tasks[assigned[a]];
    const auto& tb = tasks[assigned[b]];
    // Do not undo an immediately executable hard reservation.  The assignment
    // pass above deliberately locked these to prevent needless deadline loss.
    if (lock_local_water(a, ta) || lock_local_water(b, tb))
      continue;
    if (!eligible(a, tb) || !eligible(b, ta)) continue;
    if (genome_.hard_latest_start_reservation > 0.0 ||
        candidate_schedule == ScheduleProfile::DEADLINE_FIRST) {
      const int slack_a = std::max(
          0, ta.deadline_step - env.step_count());
      const int slack_b = std::max(
          0, tb.deadline_step - env.step_count());
      if ((ta.priority >= 950 &&
           distance(positions[b], ta.target) > slack_a) ||
          (tb.priority >= 950 &&
           distance(positions[a], tb.target) > slack_b))
        continue;
    }
    const int current = distance(positions[a], ta.target) +
                        distance(positions[b], tb.target);
    const int swapped = distance(positions[a], tb.target) +
                        distance(positions[b], ta.target);
    if (swapped < current) {
      std::swap(assigned[a], assigned[b]);
    }
  }

  // R8 exact multi-worker refinement.  R6 remains the task selector and is
  // therefore the safety baseline.  R8 may only permute already-selected,
  // inventory-independent routine jobs among workers.  It cannot add/remove a
  // task, consume a different resource, or change an economic commitment.
  //
  // The maximum-weight assignment is exact for this selected task batch.  A
  // candidate permutation is accepted only when its immediate R6 score is not
  // lower, its total current travel is not longer, and the persistent 4-8 step
  // score is strictly better.  This makes lookahead a verified tie-breaker,
  // not a license to trade away current production for an optimistic future.
  const int r8_exact_cutoff_hour = backbone.enabled ? 12 : 16;
  if (genome_.r8_execution_enabled && genome_.r8_feature_level >= 2 &&
      env.hour() < r8_exact_cutoff_hour) {
    auto safe_to_permute = [&](const AdaptiveTask& task) {
      if (task.priority >= 950 || task.required_item >= 0) return false;
      switch (task.action.op) {
        case Op::WATER:
        case Op::HARVEST:
        case Op::CARE:
        case Op::COLLECT_FERTILIZER:
        case Op::DIG:
          return true;
        default:
          return false;
      }
    };
    std::vector<int> refine_units;
    std::vector<int> refine_tasks;
    for (int unit = 0; unit < units; ++unit) {
      if (assigned[unit] < 0) continue;
      const int task_index = assigned[unit];
      if (!safe_to_permute(tasks[task_index])) continue;
      refine_units.push_back(unit);
      refine_tasks.push_back(task_index);
    }
    const int n = int(refine_units.size());
    if (n >= 2) {
      constexpr double impossible = -1.0e12;
      std::vector<std::vector<double>> enhanced_score(
          n, std::vector<double>(n, impossible));
      r8_refinement_scoring = true;
      for (int row = 0; row < n; ++row) {
        const int unit = refine_units[row];
        for (int column = 0; column < n; ++column) {
          const int task_index = refine_tasks[column];
          if (eligible(unit, tasks[task_index]))
            enhanced_score[row][column] = pair_score(unit, task_index);
        }
      }
      r8_refinement_scoring = false;

      std::vector<double> u_potential(n + 1), v_potential(n + 1);
      std::vector<int> column_row(n + 1), way(n + 1);
      bool feasible = true;
      for (int i = 1; i <= n && feasible; ++i) {
        column_row[0] = i;
        int j0 = 0;
        std::vector<double> minv(
            n + 1, std::numeric_limits<double>::infinity());
        std::vector<uint8_t> column_used(n + 1, uint8_t(0));
        do {
          column_used[j0] = uint8_t(1);
          const int i0 = column_row[j0];
          double delta = std::numeric_limits<double>::infinity();
          int j1 = 0;
          for (int j = 1; j <= n; ++j) if (!column_used[j]) {
            const double edge = enhanced_score[i0 - 1][j - 1];
            const double cost = edge <= impossible / 2 ? 1.0e12 : -edge;
            const double cur = cost - u_potential[i0] - v_potential[j];
            if (cur < minv[j]) {
              minv[j] = cur;
              way[j] = j0;
            }
            if (minv[j] < delta) {
              delta = minv[j];
              j1 = j;
            }
          }
          if (!std::isfinite(delta) || j1 == 0) {
            feasible = false;
            break;
          }
          for (int j = 0; j <= n; ++j) {
            if (column_used[j]) {
              u_potential[column_row[j]] += delta;
              v_potential[j] -= delta;
            } else {
              minv[j] -= delta;
            }
          }
          j0 = j1;
        } while (column_row[j0] != 0);
        if (!feasible) break;
        do {
          const int j1 = way[j0];
          column_row[j0] = column_row[j1];
          j0 = j1;
        } while (j0 != 0);
      }

      if (feasible) {
        std::vector<int> row_column(n, -1);
        for (int column = 1; column <= n; ++column)
          if (column_row[column] > 0)
            row_column[column_row[column] - 1] = column - 1;
        double current_base = 0.0, candidate_base = 0.0;
        double current_enhanced = 0.0, candidate_enhanced = 0.0;
        int current_travel = 0, candidate_travel = 0;
        for (int row = 0; row < n && feasible; ++row) {
          const int column = row_column[row];
          if (column < 0 || enhanced_score[row][column] <= impossible / 2) {
            feasible = false;
            break;
          }
          const int unit = refine_units[row];
          const int current_task = assigned[unit];
          const int candidate_task = refine_tasks[column];
          current_base += pair_score(unit, current_task);
          candidate_base += pair_score(unit, candidate_task);
          current_enhanced += enhanced_score[row][
              int(std::find(refine_tasks.begin(), refine_tasks.end(),
                            current_task) - refine_tasks.begin())];
          candidate_enhanced += enhanced_score[row][column];
          current_travel += distance(positions[unit], tasks[current_task].target);
          candidate_travel +=
              distance(positions[unit], tasks[candidate_task].target);
        }
        if (feasible && candidate_base + 1.0e-9 >= current_base &&
            candidate_travel <= current_travel &&
            candidate_enhanced > current_enhanced +
                genome_.task_stickiness) {
          const auto old_assigned = assigned;
          for (int row = 0; row < n; ++row)
            assigned[refine_units[row]] = refine_tasks[row_column[row]];
          // Count only real owner changes; unchanged exact solves are cheap
          // diagnostics rather than artificial reassignment churn.
          for (const int unit : refine_units)
            if (assigned[unit] != old_assigned[unit])
              state.r8_task_reassignments++;
        }
      }
    }
  }

  if (genome_.r8_execution_enabled) {
    std::vector<int8_t> r8_seen_task(tasks.size(), int8_t(-1));
    std::vector<int8_t> r8_seen_exclusive_cell(
        board_cells, int8_t(-1));
    for (int unit = 0; unit < units; ++unit) {
      const int task_index = assigned[unit];
      if (task_index < 0) continue;
      if (r8_seen_task[task_index] >= 0)
        state.r8_duplicate_reservation_violations++;
      r8_seen_task[task_index] = int8_t(unit);
      if (requires_exclusive_cell(tasks[task_index].action.op)) {
        const int cell = exclusive_cell(tasks[task_index]);
        if (r8_seen_exclusive_cell[cell] >= 0)
          state.r8_duplicate_reservation_violations++;
        r8_seen_exclusive_cell[cell] = int8_t(unit);
      }
    }
  }

  for (int u = 0; u < units; ++u) {
    if (assigned[u] < 0) {
      state.sticky_target[u] = -1;
      state.sticky_op[u] = int8_t(Op::PASS);
      if (genome_.r8_execution_enabled) state.r8_idle_unit_actions++;
      continue;
    }
    const auto& task = tasks[assigned[u]];
    actions[u] = same(positions[u], task.target) ? task.action
                                                  : move_toward(positions[u], task.target);
    if (genome_.r8_execution_enabled &&
        actions[u].op >= Op::NORTH && actions[u].op <= Op::WEST)
      state.r8_move_unit_actions++;
    state.sticky_target[u] = int16_t(task.reservation_key);
    state.sticky_cell[u] = int16_t(pos_index(task.target, env.config().board_size));
    state.sticky_op[u] = int8_t(task.action.op);
    const int group = semantic_group(task);
    if (group >= 0) state.sticky_group[u] = int8_t(group);
    if (genome_.r8_execution_enabled) {
      const int node_index = r8_node_for_task[assigned[u]];
      if (node_index >= 0 && node_index < int(state.r8_task_nodes.size())) {
        auto& node = state.r8_task_nodes[node_index];
        if (node.assigned_unit >= 0 && node.assigned_unit != u)
          state.r8_task_reassignments++;
        node.assigned_unit = int8_t(u);
      }
    }
  }
  if (genome_.r8_execution_enabled && genome_.r8_feature_level >= 2)
    state.r8_joint_matches++;
  return actions;
}

std::vector<Action> NativeAdaptivePlanner::market_orders(
    const Simulator& env, int player, const AdaptivePlan& plan,
    const AdaptivePlannerState& state,
    const std::vector<Action>& unit_actions) const {
  const auto& backbone = active_backbone(state);
  std::vector<Action> orders;
  const auto& farm = env.farms()[player];
  const auto& priv = env.privates()[player];
  const auto counts = counts_for(env, player);
  const int day = env.day(), hour = env.hour(), step = env.step_count();
  const auto crop_market_race = public_crop_market_race_pressure(env, player);
  const MarketProfile candidate_market = MarketProfile(
      plan.candidate8_market_profile);
  const int candidate_market_item = int(plan.candidate8_market_item);
  const double strategic_reserve = genome_.cash_reserve *
      (day == 0 ? genome_.opening_cash_reserve_fraction : 1.0);
  const int liquidation_day = backbone.enabled
      ? std::clamp(int(backbone.liquidation_start_step) / 24, 0, TOTAL_DAYS - 1)
      : genome_.liquidation_day;
  double projected_money = farm.money;
  const int planned_hand_deficit = std::max(
      0, int(plan.hand_target) - int(farm.hands.size()));
  const int remaining_hire_cost = std::max(
      0, daily_hire_cost(farm.hires_today + planned_hand_deficit) -
             daily_hire_cost(farm.hires_today));
  int carried_total = 0;
  for (const auto& inventory : priv.inventories)
    carried_total += inventory_sum(inventory);
  int ready_to_collect = 0;
  for (const auto& tile : farm.tiles) {
    ready_to_collect += std::max(0, int(tile.yield_units));
    ready_to_collect += tile.fertilizer_available ? 1 : 0;
  }
  std::array<int, N_PRODUCTS> sellable{};
  for (int item = 0; item < N_PRODUCTS; ++item) sellable[item] = priv.shed[item];
  int projected_shed = shed_sum(priv);
  std::vector<Position> positions{farm.farmer};
  positions.insert(positions.end(), farm.hands.begin(), farm.hands.end());
  // Unit actions commit before market orders.  Include only cargo that this
  // exact step will successfully DROP, so a terminal transaction can sell it.
  for (int unit = 0; unit < int(unit_actions.size()) &&
                     unit < int(priv.inventories.size()) &&
                     unit < int(positions.size()); ++unit) {
    const auto& unit_action = unit_actions[unit];
    if (unit_action.op == Op::HARVEST) {
      const auto& tile = farm.tiles[pos_index(
          positions[unit], env.config().board_size)];
      carried_total += std::max(0, int(tile.yield_units));
    } else if (unit_action.op == Op::COLLECT_FERTILIZER) {
      const auto& tile = farm.tiles[pos_index(
          positions[unit], env.config().board_size)];
      carried_total += tile.fertilizer_available ? 1 : 0;
    }
    if (unit_actions[unit].op != Op::DROP || !shed_adjacent(positions[unit])) continue;
    for (int item = 0; item < N_PRODUCTS; ++item) {
      sellable[item] += priv.inventories[unit][item];
      projected_shed += priv.inventories[unit][item];
    }
  }
  const bool end_of_day_capacity_risk = hour >= 16 &&
      shed_sum(priv) + carried_total + ready_to_collect >
          env.config().shed_capacity;
  int capacity_relief_needed = std::max(
      0, shed_sum(priv) + carried_total + ready_to_collect -
             env.config().shed_capacity);

  auto push = [&](Action action) {
    if (orders.size() < size_t(env.config().max_market_orders)) orders.push_back(action);
  };
  auto sell_revenue = [&](int item, int quantity) {
    double value = 0.0;
    int inventory = env.market().inventory[item];
    for (int unit = 0; unit < quantity; ++unit) {
      const int price = predicted_price(item, inventory);
      value += price;
      if (price > 1) inventory++;
    }
    return value;
  };
  auto buy_product_cost = [&](int item, int quantity) {
    double value = 0.0;
    int inventory = env.market().inventory[item];
    for (int unit = 0; unit < quantity; ++unit)
      value += predicted_price(item, --inventory);
    return value;
  };

  // Forecast the next rules-defined town consumption window for every
  // product.  The market resolves before town consumption, so step k where
  // k%interval==0 is a hold window and k+1 is the first raised quote.  The
  // forecast includes only opponent supply already visible on the public
  // board; private inventories and opponent identity are unavailable.
  std::array<int, N_PRODUCTS> shop_tick_demand{};
  for (int shop : env.shops()) {
    if (shop < 0 || shop >= 8) continue;
    for (int item = 0; item < N_PRODUCTS; ++item)
      shop_tick_demand[item] += SHOP_DEMAND[shop][item];
  }
  std::array<int, N_PRODUCTS> opponent_ready_products{};
  for (const auto& tile : env.farms()[1 - player].tiles) {
    if (tile.yield_units <= 0) continue;
    if (tile.kind == TileKind::PLANT) {
      const int item = int(tile.crop);
      if (item >= 0 && item < N_PRODUCTS)
        opponent_ready_products[item] += tile.yield_units;
    } else if (tile.kind == TileKind::ANIMAL) {
      const int animal = int(tile.animal) - 9;
      if (animal >= 0 && animal < N_ANIMALS)
        opponent_ready_products[ANIMAL_PRODUCT[animal]] += tile.yield_units;
    }
  }
  auto demand_at = [&](int item, int at_step) {
    int demand = 0;
    if (at_step % std::max(1, env.config().town_shop_sell_interval) == 0)
      demand += shop_tick_demand[item];
    if (item < 8 &&
        at_step % std::max(1, env.config().town_center_sell_interval) == 0)
      demand++;
    return demand;
  };
  auto next_demand = [&](int item) {
    for (int wait = 0; wait <= 3; ++wait) {
      const int demand = demand_at(item, step + wait);
      if (demand > 0) return std::pair<int, int>{wait, demand};
    }
    return std::pair<int, int>{99, 0};
  };
  auto forecast_sell_revenue = [&](int item, int quantity, int inventory) {
    double value = 0.0;
    for (int unit = 0; unit < quantity; ++unit) {
      const int price = predicted_price(item, inventory);
      value += price;
      if (price > 1) inventory++;
    }
    return value;
  };
  auto public_market_race = [&](int item) {
    if (genome_.runtime_market_preempt_fraction <= 0.0 ||
        item < 0 || item >= N_PRODUCTS ||
        day >= liquidation_day)
      return false;
    if (opponent_ready_products[item] <
        genome_.runtime_market_preempt_min_ready)
      return false;
    const double minimum_quote = BASE_PRICE[item] *
        genome_.runtime_market_preempt_min_price_ratio;
    return env.market().prices[item] + 1e-9 >= minimum_quote;
  };

  // Town demand creates a public, rules-defined one-step working-capital
  // opportunity for WHEAT.  Shops consume after the market at steps divisible
  // by town_shop_sell_interval; inventory bought immediately before that tick
  // can be sold one step later at the post-consumption quote.  This is not a
  // replay timetable: the phase and demand are recomputed from the current
  // public shops, market inventory, cash and shed capacity every step.
  int wheat_shop_tick_demand = 0;
  for (int shop : env.shops()) {
    if (shop >= 0 && shop < 8) wheat_shop_tick_demand += SHOP_DEMAND[shop][0];
  }
  const int shop_interval = std::max(1, env.config().town_shop_sell_interval);
  const int center_interval = std::max(1, env.config().town_center_sell_interval);
  const int shop_phase = step % shop_interval;
  const int steps_to_next_shop_tick =
      (shop_interval - shop_phase) % shop_interval;
  const int next_shop_tick = step + steps_to_next_shop_tick;
  const int next_wheat_tick_demand = wheat_shop_tick_demand +
      (next_shop_tick % center_interval == 0 ? 1 : 0);
  const int previous_step = step - 1;
  const int previous_wheat_tick_demand = previous_step >= 0
      ? wheat_shop_tick_demand + (previous_step % center_interval == 0 ? 1 : 0)
      : 0;
  // Accumulate relay inventory over the non-sale phases, then sell one step
  // after the public town-consumption tick.  Spreading the purchase over the
  // whole cycle reduces self-inflicted price impact; the tranche size below
  // is derived from public shed capacity and the official interval.
  const bool wheat_relay_buy_phase =
      genome_.wheat_relay_capacity_fraction > 0.0 &&
      day < liquidation_day && shop_phase != 1 &&
      next_wheat_tick_demand > 0;
  const bool wheat_relay_sell_phase =
      genome_.wheat_relay_capacity_fraction > 0.0 &&
      day < liquidation_day && previous_step >= 0 &&
      previous_step % shop_interval == 0 && previous_wheat_tick_demand > 0;

  // Sell before buying so the official ordered market transaction can finance
  // new projects. Before liquidation, throttle only the largest price impact.
  std::vector<int> sell_items;
  for (int item = 0; item < N_PRODUCTS; ++item) if (sellable[item] > 0) sell_items.push_back(item);
  std::sort(sell_items.begin(), sell_items.end(), [&](int a, int b) {
    if (candidate_market == MarketProfile::SELL_TO_FINANCE) {
      if (a == candidate_market_item && b != candidate_market_item) return true;
      if (b == candidate_market_item && a != candidate_market_item) return false;
    }
    // A visible same-product market race is handled before unrelated routine
    // sales.  This changes only order/timing; inventory conservation remains
    // authoritative in the simulator and all financing/feed guards below are
    // still applied from the live post-order ledger.
    const bool a_race = public_market_race(a);
    const bool b_race = public_market_race(b);
    if (a_race != b_race) return a_race;
    return env.market().prices[a] * sellable[a] > env.market().prices[b] * sellable[b];
  });
  for (int item : sell_items) {
    if (orders.size() >= 5 && day < liquidation_day &&
        !end_of_day_capacity_risk) break;
    int quantity = sellable[item];
    if (item == 0 && day < TOTAL_DAYS - 1 &&
        !end_of_day_capacity_risk) {
      const int current_animals = std::accumulate(
          counts.total_animals.begin(), counts.total_animals.end(), 0);
      const int reserved_feed = current_animals + plan.wheat_buffer;
      quantity = std::max(0, quantity - reserved_feed);
    } else if (item == 8 && day < liquidation_day &&
               !end_of_day_capacity_risk) {
      // Fertilizer is working capital, not a fixed sacred stock.  Reserve it
      // only when cash is healthy and an already planted high-value crop can
      // actually consume it profitably in the next few days.
      int profitable_need = 0;
      if (farm.money >= 2.0 * strategic_reserve) {
        for (const auto& tile : farm.tiles) {
          if (tile.kind != TileKind::PLANT) continue;
          const int crop = int(tile.crop);
          if (crop < 0 || crop >= N_CROPS) continue;
          if (env.market().prices[crop] <=
              genome_.fertilizer_value_fraction * env.market().prices[8]) continue;
          if (tile.fertilized_until_day < day + 2) profitable_need++;
        }
      }
      const int reserved_fertilizer = std::min(8, profitable_need);
      quantity = std::max(0, quantity - reserved_fertilizer);
    }
    if (quantity <= 0) continue;
    const bool operating_cash_shortfall =
        projected_money + 1e-9 < remaining_hire_cost;
    const bool candidate_item = item == candidate_market_item;
    const bool public_race_preempt = public_market_race(item);
    const bool candidate_force_sell = (candidate_item &&
        (candidate_market == MarketProfile::SELL_NOW ||
         candidate_market == MarketProfile::SELL_TO_FINANCE)) ||
        public_race_preempt;
    const bool candidate_hold = candidate_item &&
        candidate_market == MarketProfile::HOLD_FOR_DEMAND;
    const bool candidate_partial = candidate_item &&
        candidate_market == MarketProfile::PARTIAL_SELL;
    // A capacity emergency is not a reason to dump the whole shed.  Sell only
    // the amount needed to make room for cargo that will automatically return
    // at day end.  This is recomputed from the live ledger and works for any
    // product mix; liquidation and a true cash shortfall may still sell more.
    if (end_of_day_capacity_risk && day < liquidation_day &&
        !operating_cash_shortfall) {
      quantity = std::min(quantity, capacity_relief_needed);
    }
    if (public_race_preempt && day < liquidation_day &&
        !end_of_day_capacity_risk && !operating_cash_shortfall) {
      const int visible_race_quantity = std::max(
          1, int(std::ceil(genome_.runtime_market_preempt_fraction *
                           opponent_ready_products[item])));
      quantity = std::min({
          quantity,
          visible_race_quantity,
          std::max(1, genome_.runtime_market_preempt_max_quantity)});
    }
    if (candidate_hold && day < liquidation_day &&
        !end_of_day_capacity_risk && !operating_cash_shortfall)
      continue;
    if (candidate_partial && day < liquidation_day &&
        !end_of_day_capacity_risk && !operating_cash_shortfall)
      quantity = std::max(1, quantity / 2);
    // Preserve relay stock until the first quote after the public consumption
    // tick.  Deadline liquidation, a capacity emergency or a cash shortfall
    // may still force an earlier sale.
    if (!candidate_force_sell && item == 0 &&
        genome_.wheat_relay_capacity_fraction > 0.0 &&
        wheat_shop_tick_demand > 0 &&
        !wheat_relay_sell_phase && day < liquidation_day &&
        !end_of_day_capacity_risk && !operating_cash_shortfall)
      continue;
    if (!candidate_force_sell && genome_.town_demand_sell_timing > 0.0 &&
        day < liquidation_day && !end_of_day_capacity_risk &&
        !operating_cash_shortfall) {
      const auto [wait, known_demand] = next_demand(item);
      const bool just_after_demand = step > 0 && demand_at(item, step - 1) > 0;
      if (!just_after_demand && wait <= 3 && known_demand > 0) {
        const int adverse_supply = int(std::nearbyint(
            genome_.opponent_supply_weight *
            opponent_ready_products[item]));
        const int future_inventory = std::max(
            1, env.market().inventory[item] + adverse_supply - known_demand);
        const double now_value = sell_revenue(item, quantity);
        const double future_value = forecast_sell_revenue(
            item, quantity, future_inventory);
        const double required_gain =
            0.05 * (1.0 - genome_.town_demand_sell_timing);
        if (future_value > now_value * (1.0 + required_gain))
          continue;
      }
    }
    if (!candidate_force_sell && day < liquidation_day &&
        !end_of_day_capacity_risk &&
        !operating_cash_shortfall &&
        !(item == 0 && wheat_relay_sell_phase)) {
      if (quantity < 4 && env.market().prices[item] < int(1.08 * BASE_PRICE[item])) continue;
      const int now = env.market().prices[item];
      int best = 1;
      for (int q = 1; q <= quantity; ++q) {
        const int after = predicted_price(item, env.market().inventory[item] + q);
        if (now <= 1 || (now - after) / double(now) <= genome_.sell_drop_limit) best = q;
        else break;
      }
      quantity = best;
    }
    if (quantity > 0) {
      push(Action{Op::SELL, Item(item), quantity});
      projected_money += sell_revenue(item, quantity);
      projected_shed = std::max(0, projected_shed - quantity);
      capacity_relief_needed = std::max(0, capacity_relief_needed - quantity);
    }
  }

  // Protect already committed animal projects before any new expansion.  Feed
  // is a sunk-capital deadline, so it may spend the normal cash reserve and it
  // must precede HIRE / land / animal / seed orders in the ten-slot sequence.
  int unfed_field_animals = 0;
  for (const auto& tile : farm.tiles)
    unfed_field_animals += tile.kind == TileKind::ANIMAL && !tile.fed_today;
  int pending_service_placements = 0;
  for (int animal = 0; animal < N_ANIMALS; ++animal)
    pending_service_placements += std::max(
        0, int(plan.animal_service_targets[animal]) - counts.field_animals[animal]);
  const int owned_animals = std::accumulate(
      counts.total_animals.begin(), counts.total_animals.end(), 0);
  // Keep one physical wheat reserve per already-owned animal.  This supports
  // just-in-time feeding and the next service cycle without adding the daily
  // buffer a second time.  Do not pre-buy feed for animals that have not yet
  // been purchased: the current turn's scarce cash and ten market slots must
  // first establish the selected capital/seed portfolio.  Once the animals
  // exist, the next market step replenishes their reserve from actual state.
  // There is no next service cycle after the final day.  Replenishing feed
  // after terminal liquidation creates a guaranteed losing buy/sell loop.
  const int feed_need = day >= TOTAL_DAYS - 1 || owned_animals == 0 ? 0 :
      std::max({owned_animals,
                unfed_field_animals + pending_service_placements,
                int(plan.wheat_buffer)});
  int wheat_have = priv.shed[0];
  for (const auto& inv : priv.inventories) wheat_have += inv[0];
  int buy_wheat = std::min(30, std::max(0, feed_need - wheat_have));
  if (hour >= 16) {
    // Do not immediately refill the exact shed space that the preceding
    // emergency sale created.  Returned cargo (often including wheat) becomes
    // available next day, when the feed ledger is re-evaluated from reality.
    const int safe_purchase_space = std::max(
        0, env.config().shed_capacity - projected_shed - carried_total -
               ready_to_collect);
    buy_wheat = std::min(buy_wheat, safe_purchase_space);
  }
  // Buying all desired feed while leaving zero cash for today's workforce is
  // self-defeating: the animals then cannot be serviced at all.  Preserve the
  // exact remaining hire bill and buy the largest feed quantity affordable
  // from the balance.  The prior sell pass may raise that balance first.
  const double feed_budget = std::max(
      0.0, projected_money - double(remaining_hire_cost));
  while (buy_wheat > 0 && buy_product_cost(0, buy_wheat) > feed_budget)
    --buy_wheat;
  if (buy_wheat > 0) {
    push(Action{Op::BUY_PRODUCT, Item::WHEAT, buy_wheat});
    projected_money = std::max(0.0,
        projected_money - buy_product_cost(0, buy_wheat));
    projected_shed += buy_wheat;
  }
  int projected_wheat_inventory = env.market().inventory[0] - buy_wheat;

  // Workforce is recomputed every day from the current plan.
  int hand_deficit = planned_hand_deficit;
  for (int i = 0; i < hand_deficit && int(orders.size()) < env.config().max_market_orders; ++i) {
    const int before = daily_hire_cost(farm.hires_today + i);
    const int after = daily_hire_cost(farm.hires_today + i + 1);
    const int hire_cost = after - before;
    // Hands are operating capacity, not optional capital expenditure.  Feed
    // has already been secured above; withholding all workers merely because
    // cash dipped below the strategic reserve destroys the projects that are
    // supposed to replenish that cash.
    if (projected_money < hire_cost) break;
    push(Action{Op::HIRE});
    projected_money -= hire_cost;
  }

  int quadrants = popcount(farm.unlocked_mask);
  if (quadrants < plan.quadrant_target && quadrants <= 3) {
    const int cost = LAND_COST[quadrants - 1];
    double land_reserve = strategic_reserve;
    if (backbone.enabled &&
        quadrants < backbone.quadrant_target[std::clamp(day, 0, TOTAL_DAYS - 1)]) {
      // Land is a deadline-bearing capacity commitment: postponing it can make
      // tomorrow's entire plant/animal flow infeasible.  Current-day feed and
      // hires have already been compiled above, so reserve the next day's
      // *derived operating bill* instead of an unrelated strategic constant.
      const int next_day = std::min(TOTAL_DAYS - 1, day + 1);
      int next_service_animals = 0;
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        next_service_animals += backbone.animal_service_targets[next_day][animal];
      const int next_feed_units = std::max(
          next_service_animals, int(backbone.wheat_buffer[next_day]));
      const double next_operating_bill =
          daily_hire_cost(backbone.hand_target[next_day]) +
          next_feed_units * env.market().prices[0];
      land_reserve = std::min(land_reserve, next_operating_bill);
    }
    if (projected_money >= cost + land_reserve) {
      push(Action{Op::BUY_LAND});
      projected_money -= cost;
    }
  }

  if (day <= genome_.stop_new_animals_day || backbone.enabled) {
    for (int animal : {1, 2, 0}) {
      int deficit = std::max(0, int(plan.animal_targets[animal]) - counts.total_animals[animal]);
      const int affordable = int(std::max(0.0, projected_money - strategic_reserve)) / ANIMAL_COST[animal];
      deficit = std::min({deficit, affordable, 4});
      if (hour >= 16) {
        const int safe_purchase_space = std::max(
            0, env.config().shed_capacity - projected_shed - carried_total -
                   ready_to_collect);
        deficit = std::min(deficit, safe_purchase_space);
      }
      if (deficit > 0) {
        push(Action{Op::BUY_ANIMAL, Item(9 + animal), deficit});
        projected_money -= deficit * ANIMAL_COST[animal];
        projected_shed += deficit;
      }
    }
  }

  bool any_crop_purchase_open = backbone.enabled;
  for (int crop = 0; crop < N_CROPS; ++crop)
    any_crop_purchase_open = any_crop_purchase_open ||
        crop_project_open(genome_, crop, day);
  if (any_crop_purchase_open) {
    // Buy only what can be planted in the near-term physical slots.  A desired
    // portfolio is allowed to change at the next replan; pre-buying the whole
    // future deficit stranded dozens of seeds whenever the mix changed.
    int pasture_target = int(plan.animal_targets[1] + plan.animal_targets[2]);
    int coop_target = int(plan.animal_targets[0]);
    if (backbone.enabled && genome_.future_structure_reservation_fraction > 0.0) {
      int future_pasture_target = pasture_target;
      int future_coop_target = coop_target;
      for (int future_day = day; future_day < TOTAL_DAYS; ++future_day) {
        int cow_target = backbone.animal_targets[future_day][1];
        const int sheep_target = backbone.animal_targets[future_day][2];
        if (state.animal_branch == 1) {
          const int owned_cows = counts.total_animals[1];
          cow_target = owned_cows + int(std::nearbyint(
              std::max(0, cow_target - owned_cows) *
              (1.0 - genome_.yarn_animal_flex)));
        }
        future_pasture_target = std::max(
            future_pasture_target, cow_target + sheep_target);
        future_coop_target = std::max<int>(
            future_coop_target, backbone.animal_targets[future_day][0]);
      }
      pasture_target += int(std::nearbyint(
          genome_.future_structure_reservation_fraction *
          std::max(0, future_pasture_target - pasture_target)));
      coop_target += int(std::nearbyint(
          genome_.future_structure_reservation_fraction *
          std::max(0, future_coop_target - coop_target)));
    }
    const int structure_need =
        std::max(0, pasture_target - counts.pastures) +
        std::max(0, coop_target - counts.coops);
    int planned_crop_releases = 0;
    if (backbone.enabled)
      for (int crop = 0; crop < N_CROPS; ++crop) {
        if (CROP_ONGOING[crop]) continue;
        int target = backbone.crop_harvest[day][crop];
        if (state.crop_suffix == 1 && crop != 0 && crop != 1)
          target = std::max(0, int(std::nearbyint(
              target * (1.0 - genome_.pet_crop_flex))));
        planned_crop_releases += target;
      }
    int plant_purchase_slots = std::max(
        0, counts.empty + counts.weeds + planned_crop_releases - structure_need);
    std::array<int, N_CROPS> planted_today{};
    for (const auto& tile : farm.tiles)
      if (tile.kind == TileKind::PLANT && tile.planted_day == day) {
        const int crop = int(tile.crop);
        if (crop >= 0 && crop < N_CROPS) planted_today[crop]++;
      }
    std::array<int, N_CROPS> seed_order{{4, 3, 2, 1, 0}};
    if (genome_.seed_transaction_value_order > 0.0) {
      auto seed_commitment_score = [&](int crop) {
        if (day + CROP_FIRST[crop] >= TOTAL_DAYS)
          return -std::numeric_limits<double>::infinity();
        const int stock_deficit = std::max(
            0, int(plan.crop_targets[crop]) - counts.committed_crops[crop]);
        int flow_target = backbone.enabled
            ? int(backbone.crop_plant[day][crop]) : 0;
        if (state.crop_suffix == 1 && crop != 0 && crop != 1)
          flow_target = std::max(0, int(std::nearbyint(
              flow_target * (1.0 - genome_.pet_crop_flex))));
        const int flow_deficit = std::max(
            0, flow_target - planted_today[crop] - int(priv.seeds[crop]));
        const int deficit = std::max(stock_deficit, flow_deficit);
        if (deficit <= 0) return -1.0e12;
        const double gross = env.market().prices[crop] *
            std::max(1, CROP_MAX_YIELD[crop]);
        const double net = std::max(
            1.0, gross - double(SEED_COST[crop]));
        const double cash_cycle = std::max(
            1, SEED_COST[crop] * CROP_FIRST[crop]);
        // Larger due quantities matter, but a slow expensive crop cannot
        // consume all working capital merely because its calendar deficit is
        // numerically larger.  The multiplier is deliberately sublinear.
        return std::sqrt(double(deficit)) * net / cash_cycle;
      };
      std::stable_sort(seed_order.begin(), seed_order.end(),
          [&](int a, int b) {
            const double sa = seed_commitment_score(a);
            const double sb = seed_commitment_score(b);
            return sa != sb ? sa > sb : a < b;
          });
    }
    for (int crop : seed_order) {
      if (!backbone.enabled &&
          !crop_project_open(genome_, crop, day)) continue;
      if (day + CROP_FIRST[crop] >= TOTAL_DAYS) continue;
      const int stock_deficit = std::max(
          0, int(plan.crop_targets[crop]) - counts.committed_crops[crop]);
      int flow_target = backbone.enabled
          ? int(backbone.crop_plant[day][crop]) : 0;
      if (state.crop_suffix == 1 && crop != 0 && crop != 1)
        flow_target = std::max(0, int(std::nearbyint(
            flow_target * (1.0 - genome_.pet_crop_flex))));
      const int flow_deficit = std::max(
          0, flow_target - planted_today[crop] - int(priv.seeds[crop]));
      int deficit = std::max(stock_deficit, flow_deficit);
      const double race_release = genome_.market_race_acceleration *
          crop_market_race[crop];
      const double due_release = flow_deficit > 0
          ? std::max(genome_.due_flow_cash_release, race_release) : 0.0;
      const double due_reserve = strategic_reserve * (1.0 - due_release);
      const int affordable = int(std::max(0.0, projected_money - due_reserve)) /
          SEED_COST[crop];
      deficit = std::min({deficit, affordable, plant_purchase_slots, 20});
      if (deficit > 0) {
        push(Action{Op::BUY_SEED, Item(crop), deficit});
        projected_money -= deficit * SEED_COST[crop];
        plant_purchase_slots -= deficit;
      }
    }
  }

  // Use only residual working capital after real production commitments.  At
  // most half of the shed is allowed to become relay stock so unit DROP and
  // animal placement remain possible on the following step.  Enumerating all
  // feasible quantities is cheap (shed capacity is 100) and avoids a fixed
  // gold-Replay quantity such as 16 or 32.
  if (wheat_relay_buy_phase &&
      int(orders.size()) < env.config().max_market_orders) {
    const int current_animals = std::accumulate(
        counts.total_animals.begin(), counts.total_animals.end(), 0);
    const int protected_wheat = current_animals + int(plan.wheat_buffer);
    const int current_excess_wheat = std::max(
        0, priv.shed[0] + buy_wheat - protected_wheat);
    const int relay_stock_cap = std::max(
        0, int(std::floor(env.config().shed_capacity *
                          genome_.wheat_relay_capacity_fraction)));
    const int relay_buy_phases = std::max(1, shop_interval - 1);
    const int tranche_cap = std::max(
        1, (relay_stock_cap + relay_buy_phases - 1) / relay_buy_phases);
    const int operational_space = std::max(
        0, env.config().shed_capacity - projected_shed -
           std::min(env.config().shed_capacity / 4,
                    carried_total + ready_to_collect));
    const int late_safe_space = hour < 16 ? operational_space : std::max(
        0, env.config().shed_capacity - projected_shed - carried_total -
               ready_to_collect);
    const int quantity_cap = std::max(
        0, std::min({operational_space, late_safe_space,
                     relay_stock_cap - current_excess_wheat,
                     tranche_cap}));
    const double relay_cash = std::max(
        0.0, projected_money - strategic_reserve);
    const int adverse_opponent_wheat = std::max(
        0, int(std::nearbyint(genome_.wheat_relay_opponent_risk *
                              visible_daily_supply(env, 1 - player)[0])));

    int best_quantity = 0;
    double best_profit = 0.0;
    double best_cost = 0.0;
    double running_cost = 0.0;
    for (int quantity = 1; quantity <= quantity_cap; ++quantity) {
      running_cost += predicted_price(0, projected_wheat_inventory - quantity);
      if (running_cost > relay_cash) break;
      double sale_value = 0.0;
      int sale_inventory = projected_wheat_inventory - quantity -
          next_wheat_tick_demand + adverse_opponent_wheat;
      for (int unit = 0; unit < quantity; ++unit) {
        const int price = predicted_price(0, sale_inventory);
        sale_value += price;
        if (price > 1) sale_inventory++;
      }
      const double profit = sale_value - running_cost;
      if (profit > best_profit) {
        best_profit = profit;
        best_quantity = quantity;
        best_cost = running_cost;
      }
    }
    if (best_quantity > 0 && best_profit >= 1.0) {
      push(Action{Op::BUY_PRODUCT, Item::WHEAT, best_quantity});
      projected_money -= best_cost;
      projected_shed += best_quantity;
    }
  }

  return orders;
}

PlayerAction NativeAdaptivePlanner::action(const Simulator& env, int player,
                                            AdaptivePlannerState& state) const {
  const int step = env.step_count();
  if (step == 0 || step < state.last_step) state.reset();

  // Maintain one compact public opponent-history sample per completed day.
  // No private opponent inventory, route identity or future action is stored.
  if (state.opponent_history_day != env.day()) {
    const auto& opponent = env.farms()[1 - player];
    std::array<int16_t, N_CROPS + N_ANIMALS> project_counts{};
    std::array<int16_t, N_CROPS + N_ANIMALS> project_yield{};
    for (const auto& tile : opponent.tiles) {
      if (tile.kind == TileKind::PLANT) {
        const int crop = int(tile.crop);
        if (crop >= 0 && crop < N_CROPS) {
          project_counts[crop]++;
          project_yield[crop] += std::max(0, int(tile.yield_units));
        }
      } else if (tile.kind == TileKind::ANIMAL) {
        const int animal = int(tile.animal) - 9;
        if (animal >= 0 && animal < N_ANIMALS) {
          const int project = N_CROPS + animal;
          project_counts[project]++;
          project_yield[project] += std::max(0, int(tile.yield_units));
        }
      }
    }
    const int cash = int(std::nearbyint(opponent.money));
    const int hands = int(opponent.hands.size());
    const int quadrants = popcount(opponent.unlocked_mask);
    if (state.opponent_history_day >= 0) {
      const int elapsed_days = std::max(
          1, env.day() - int(state.opponent_history_day));
      auto update_trend = [&](int old_trend, int delta_x100) {
        return int16_t(std::clamp(
            int(std::nearbyint(0.65 * old_trend + 0.35 * delta_x100)),
            -32000, 32000));
      };
      for (int project = 0; project < N_CROPS + N_ANIMALS; ++project) {
        state.opponent_recent_project_count_delta[project] = int16_t(
            int(project_counts[project]) -
            int(state.opponent_last_project_counts[project]));
        state.opponent_recent_project_yield_delta[project] = int16_t(
            int(project_yield[project]) -
            int(state.opponent_last_project_yield[project]));
        const int count_delta = 100 *
            (int(project_counts[project]) -
             int(state.opponent_last_project_counts[project])) /
            elapsed_days;
        const int yield_delta = 100 *
            (int(project_yield[project]) -
             int(state.opponent_last_project_yield[project])) /
            elapsed_days;
        state.opponent_project_count_trend_x100[project] = update_trend(
            state.opponent_project_count_trend_x100[project], count_delta);
        state.opponent_project_yield_trend_x100[project] = update_trend(
            state.opponent_project_yield_trend_x100[project], yield_delta);
      }
      state.opponent_cash_trend = int32_t(std::nearbyint(
          0.65 * state.opponent_cash_trend +
          0.35 * (cash - state.opponent_last_cash) / elapsed_days));
      state.opponent_recent_cash_delta = cash - state.opponent_last_cash;
      state.opponent_hands_trend_x100 = update_trend(
          state.opponent_hands_trend_x100,
          100 * (hands - int(state.opponent_last_hands)) / elapsed_days);
      state.opponent_recent_hands_delta = int16_t(
          hands - int(state.opponent_last_hands));
      state.opponent_quadrants_trend_x100 = update_trend(
          state.opponent_quadrants_trend_x100,
          100 * (quadrants - int(state.opponent_last_quadrants)) /
              elapsed_days);
      state.opponent_recent_quadrants_delta = int16_t(
          quadrants - int(state.opponent_last_quadrants));
      state.opponent_history_samples++;
    }
    state.opponent_last_project_counts = project_counts;
    state.opponent_last_project_yield = project_yield;
    state.opponent_last_cash = cash;
    state.opponent_last_hands = int16_t(hands);
    state.opponent_last_quadrants = int16_t(quadrants);
    state.opponent_history_day = int16_t(env.day());
  }
  if (state.flow_day != env.day()) {
    state.flow_day = env.day();
    state.crop_plant_actions.fill(0);
    state.crop_water_actions.fill(0);
    state.crop_harvest_actions.fill(0);
    state.crop_fertilize_actions.fill(0);
    state.crop_clear_actions = 0;
    state.animal_feed_actions.fill(0);
    state.animal_care_actions.fill(0);
    state.animal_product_actions.fill(0);
    state.animal_fertilizer_actions.fill(0);
  }
  if (state.last_step >= 0 && env.day() != state.plan.generated_day) {
    for (int item = 0; item < N_PRODUCTS; ++item) {
      const double delta = state.last_inventory[item] - env.market().inventory[item];
      state.observed_daily_drift[item] =
          0.65 * state.observed_daily_drift[item] + 0.35 * delta;
    }
  }
  // A local portfolio edit is meaningful only if the destination project is
  // allowed to reach production.  Conversely, a fixed N-day target lock is a
  // brittle mini-route.  Keep the generic source cap/destination floor until
  // the added destination is physically commissioned and has crossed its
  // official first-production lead time, then release it.  A bounded timeout
  // protects recovery when cash, weeds or shared-market fills prevent the
  // project from being commissioned.  No route identity, coordinate or
  // expert calendar participates in this transition.
  auto clear_bundle_lock = [&]() {
    state.bundle_locked_source = -1;
    state.bundle_locked_destination = -1;
    state.bundle_locked_source_target = -1;
    state.bundle_locked_destination_target = -1;
    state.bundle_lock_until_day = -1;
  };
  if (state.bundle_lock_until_day >= 0) {
    if (env.day() > state.bundle_lock_until_day) {
      clear_bundle_lock();
    } else if (state.last_bundle_switch_day >= 0 &&
               state.bundle_locked_destination >= 0) {
      const auto own = counts_for(env, player);
      const int source = int(state.bundle_locked_source);
      const int destination = int(state.bundle_locked_destination);
      bool source_edit_realised = true;
      if (source >= 0 && source < N_CROPS) {
        source_edit_realised = own.committed_crops[source] <=
            int(state.bundle_locked_source_target);
      } else if (source >= N_CROPS &&
                 source < N_CROPS + N_ANIMALS) {
        source_edit_realised = own.total_animals[source - N_CROPS] <=
            int(state.bundle_locked_source_target);
      }
      bool destination_commissioned = false;
      int destination_lead = 1;
      if (destination < N_CROPS) {
        destination_commissioned = own.field_crops[destination] >=
            int(state.bundle_locked_destination_target);
        destination_lead = CROP_FIRST[destination];
      } else if (destination < N_CROPS + N_ANIMALS) {
        const int animal = destination - N_CROPS;
        destination_commissioned = own.field_animals[animal] >=
            int(state.bundle_locked_destination_target);
        destination_lead = ANIMAL_FIRST[animal];
      }
      const bool first_cycle_reached = env.day() >=
          int(state.last_bundle_switch_day) + destination_lead;
      if (source_edit_realised && destination_commissioned &&
          first_cycle_reached) {
        clear_bundle_lock();
      }
    }
  }
  auto has_shop = [&](int shop) {
    return std::find(env.shops().begin(), env.shops().end(), shop) !=
           env.shops().end();
  };
  const auto& backbone = active_backbone(state);
  // Choose at the first animal-capital expansion encoded by the semantic
  // plan.  This keeps the decision before the irreversible purchase while
  // allowing public competitor capacity to become visible.  A crowded sheep
  // market suppresses the YARN branch; no opponent identity is used.
  int animal_expansion_day = 5;
  if (backbone.enabled) {
    const int opening_animals = std::accumulate(
        backbone.animal_targets[0].begin(),
        backbone.animal_targets[0].end(), 0);
    for (int candidate_day = 1; candidate_day < TOTAL_DAYS; ++candidate_day) {
      const int candidate_animals = std::accumulate(
          backbone.animal_targets[candidate_day].begin(),
          backbone.animal_targets[candidate_day].end(), 0);
      if (candidate_animals > opening_animals) {
        animal_expansion_day = candidate_day;
        break;
      }
    }
  }
  const int animal_decision_day = animal_expansion_day +
      (genome_.animal_branch_economic_selector > 0.0
           ? genome_.animal_branch_observation_days : 0);
  if (state.animal_branch < 0 && env.day() >= animal_decision_day) {
    const auto opponent_supply = visible_daily_supply(env, 1 - player);
    int visible_opponent_cows = 0, visible_opponent_sheep = 0;
    for (const auto& tile : env.farms()[1 - player].tiles)
      if (tile.kind == TileKind::ANIMAL) {
        visible_opponent_cows += tile.animal == Item::COW;
        visible_opponent_sheep += tile.animal == Item::SHEEP;
      }
    if (backbones_.size() == 4 &&
        genome_.animal_branch_economic_selector > 0.0) {
      const auto demand = shop_demand(env);
      const auto own = counts_for(env, player);
      const int horizon = std::max(
          1, std::min(12, TOTAL_DAYS - env.day()));
      auto branch_value = [&](int branch_index) {
        double total = 0.0;
        for (int animal : {1, 2}) {
          int target = own.total_animals[animal];
          const int last_day = std::min(
              TOTAL_DAYS - 1, env.day() + horizon);
          for (int future_day = env.day(); future_day <= last_day;
               ++future_day)
            target = std::max<int>(
                target,
                backbones_[branch_index].animal_targets[future_day][animal]);
          const int item = ANIMAL_PRODUCT[animal];
          const double production_rate =
              double(1 + ANIMAL_INTERVAL[animal]) /
              std::max(1, ANIMAL_INTERVAL[animal]);
          const double public_opponent_supply =
              genome_.animal_branch_economic_selector *
              genome_.opponent_supply_weight * opponent_supply[item];
          const double own_candidate_supply =
              genome_.market_impact_weight * target * production_rate;
          const int inventory = int(std::nearbyint(
              env.market().inventory[item] - demand[item] * horizon +
              public_opponent_supply * horizon +
              0.5 * own_candidate_supply * horizon));
          const double price = predicted_price(item, std::max(1, inventory));
          const int productive_days = std::max(
              0, horizon - ANIMAL_FIRST[animal]);
          const double gross = target * productive_days *
              production_rate * price;
          const double feed = target * horizon * env.market().prices[0];
          const double capital = std::max(
              0, target - own.total_animals[animal]) *
              ANIMAL_COST[animal];
          const double service_actions = target * horizon *
              (3.0 + 1.0 / std::max(1, ANIMAL_INTERVAL[animal]));
          total += gross - feed - capital -
              0.25 * genome_.action_cost * service_actions;
        }
        return total;
      };
      // Branch indices one and three share the wheat crop suffix, isolating
      // the animal portfolio comparison from the later crop decision.
      const double cow_value = branch_value(1);
      const double sheep_value = branch_value(3);
      const int crowding_delta =
          visible_opponent_cows - visible_opponent_sheep;
      if (crowding_delta >= genome_.animal_counter_crowding_gate)
        state.animal_branch = 1;
      else if (-crowding_delta >= genome_.animal_counter_crowding_gate)
        state.animal_branch = 0;
      else
        state.animal_branch = sheep_value > cow_value ? 1 : 0;
    } else if (has_shop(SHOP_YARN_STORE) &&
               env.market().prices[7] >=
                   genome_.yarn_wool_price_threshold) {
      state.animal_branch = visible_opponent_sheep <
              genome_.yarn_opponent_sheep_gate
          ? 1 : 0;
    } else if (env.day() >= std::max(9, animal_expansion_day + 2)) {
      state.animal_branch = 0;
    }
  }
  // The compatible crop/sales suffix is chosen in days 19-24.  Keep the
  // decision sticky and leave all already planted crops under hard care.
  if (state.crop_suffix < 0 && env.day() >= 19) {
    if (has_shop(SHOP_PET_CAFE) &&
        env.market().prices[1] >= genome_.pet_carrot_price_threshold)
      state.crop_suffix = 1;
    else if (env.day() >= 24)
      state.crop_suffix = 0;
  }
  if (genome_.operating_prior_selection > 0.0 && !backbones_.empty() &&
      state.operating_prior_selection_day != env.day()) {
    int best_index = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    double current_score = -std::numeric_limits<double>::infinity();
    const int current_index = int(state.operating_prior_index);
    for (int candidate_index = -1;
         candidate_index < int(backbones_.size()); ++candidate_index) {
      AdaptivePlannerState candidate_state = state;
      candidate_state.operating_prior_index = int8_t(candidate_index);
      const AdaptivePlan candidate_plan = build_plan(
          env, player, candidate_state);
      const double score = score_plan_candidate(
          env, player, candidate_plan, nullptr, &candidate_state);
      if (candidate_index == current_index) current_score = score;
      if (score > best_score) {
        best_score = score;
        best_index = candidate_index;
      }
    }
    if (current_index >= -1 && current_index < int(backbones_.size()) &&
        best_index != current_index) {
      const double required_gain = genome_.operating_prior_switch_margin *
          std::max(1000.0, std::abs(current_score));
      if (best_score <= current_score + required_gain)
        best_index = current_index;
    }
    if (current_index >= -1 && best_index != current_index)
      state.operating_prior_switches++;
    state.operating_prior_index = int8_t(best_index);
    state.operating_prior_selection_day = int16_t(env.day());
    // Selecting a different operating prior invalidates the cached plan even
    // if no ordinary price/shop replan trigger fired on this exact step.
    state.last_plan_step = -1;
  }
  if (should_replan(env, player, state)) {
    AdaptivePlan next_plan = build_plan(env, player, state);
    if (next_plan.candidate8_family >= 0 &&
        state.candidate8_decisions == 0) {
      state.candidate8_decisions = 1;
      state.first_candidate8_day = int16_t(env.day());
      state.first_candidate8_family = next_plan.candidate8_family;
      state.first_candidate8_signature = next_plan.candidate8_signature;
      state.first_candidate8_raw_count = next_plan.candidate8_raw_count;
      state.first_candidate8_feasible_count =
          next_plan.candidate8_feasible_count;
      state.first_candidate8_shortlist_count =
          next_plan.candidate8_shortlist_count;
      state.first_candidate8_target_delta = next_plan.candidate8_target_delta;
      state.first_candidate8_hand_delta = next_plan.candidate8_hand_delta;
      state.first_candidate8_quadrant_delta = next_plan.candidate8_quadrant_delta;
      state.first_candidate8_effective_delay_days =
          next_plan.candidate8_effective_delay_days;
      state.first_candidate8_schedule_profile =
          next_plan.candidate8_schedule_profile;
      state.first_candidate8_market_profile =
          next_plan.candidate8_market_profile;
      state.first_candidate8_recovery_profile =
          next_plan.candidate8_recovery_profile;
      state.first_candidate8_suffix_project =
          next_plan.candidate8_suffix_project;
      state.first_candidate8_market_item = next_plan.candidate8_market_item;
      state.first_candidate8_recovery_issue =
          next_plan.candidate8_recovery_issue;
      state.first_candidate8_estimated_value =
          next_plan.candidate8_estimated_value;
      state.first_candidate8_estimated_cash_cost =
          next_plan.candidate8_estimated_cash_cost;
      state.first_candidate8_estimated_daily_action_load =
          next_plan.candidate8_estimated_daily_action_load;
      state.first_candidate8_context_features =
          next_plan.candidate8_context_features;
      // Persist only the sparse fields explicitly changed by this edit.
      // Probes operate on copied planner state, so this mutation affects the
      // selected Beam child but never contaminates its siblings.
      for (int crop = 0; crop < N_CROPS; ++crop)
        if (next_plan.candidate8_target_delta[crop] != 0)
          state.candidate8_persistent_target[crop] =
              next_plan.crop_targets[crop];
      for (int animal = 0; animal < N_ANIMALS; ++animal) {
        const int project = N_CROPS + animal;
        if (next_plan.candidate8_target_delta[project] != 0)
          state.candidate8_persistent_target[project] =
              next_plan.animal_targets[animal];
      }
      if (next_plan.candidate8_hand_delta > 0)
        state.candidate8_persistent_hand_target = int16_t(std::max(
            int(state.candidate8_persistent_hand_target),
            int(env.farms()[player].hands.size()) +
                int(next_plan.candidate8_hand_delta)));
      if (next_plan.candidate8_quadrant_delta > 0)
        state.candidate8_persistent_quadrant_target = int16_t(std::max(
            int(state.candidate8_persistent_quadrant_target),
            popcount(env.farms()[player].unlocked_mask) +
                int(next_plan.candidate8_quadrant_delta)));
      if (next_plan.candidate8_schedule_profile !=
          int8_t(ScheduleProfile::CURRENT))
        state.candidate8_persistent_schedule_profile =
            next_plan.candidate8_schedule_profile;
    }
    if (next_plan.bundle_switch_applied) {
      state.bundle_switches++;
      state.bundle_predicted_gain += next_plan.bundle_predicted_gain;
      state.last_bundle_switch_day = int16_t(env.day());
      state.bundle_locked_source = next_plan.bundle_switch_source;
      state.bundle_locked_destination = next_plan.bundle_switch_destination;
      state.bundle_locked_source_target =
          next_plan.bundle_switch_source_target;
      state.bundle_locked_destination_target =
          next_plan.bundle_switch_destination_target;
      // portfolio_switch_cooldown_days controls only how soon another explicit
      // SWITCH may be proposed.  The independent hold parameter bounds a
      // project-level commitment which can release early after commissioning
      // and first production; zero leaves subsequent event-driven replanning
      // completely unconstrained.
      if (genome_.portfolio_local_edit_hold_days > 0) {
        state.bundle_lock_until_day = int16_t(std::min(
            TOTAL_DAYS - 1,
            env.day() + genome_.portfolio_local_edit_hold_days));
      } else {
        state.bundle_locked_source = -1;
        state.bundle_locked_destination = -1;
        state.bundle_locked_source_target = -1;
        state.bundle_locked_destination_target = -1;
        state.bundle_lock_until_day = -1;
      }
      if (state.first_bundle_switch_day < 0) {
        state.first_bundle_switch_day = int16_t(env.day());
        state.first_bundle_switch_features =
            next_plan.bundle_switch_features;
      }
    }
    state.plan = next_plan;
    state.last_plan_step = step;
    state.last_shop_count = int(env.shops().size());
    state.last_prices = env.market().prices;
    state.last_inventory = env.market().inventory;
    state.replans++;
  }
  const auto tasks = build_tasks(env, player, state.plan, state);
  PlayerAction out;
  out.units = assign_tasks(env, player, tasks, state);
  out.market = market_orders(env, player, state.plan, state, out.units);
  std::vector<Position> positions{env.farms()[player].farmer};
  positions.insert(positions.end(), env.farms()[player].hands.begin(),
                   env.farms()[player].hands.end());
  for (int unit = 0; unit < int(out.units.size()); ++unit) {
    const auto& unit_action = out.units[unit];
    state.override_actions += unit_action.op != Op::PASS;
    if (unit_action.op == Op::PLANT) {
      const int crop = int(unit_action.item);
      if (crop >= 0 && crop < N_CROPS) state.crop_plant_actions[crop]++;
    } else if (unit_action.op == Op::DIG) {
      state.crop_clear_actions++;
    } else if (unit < int(positions.size())) {
      const auto& tile = env.farms()[player].tiles[
          pos_index(positions[unit], env.config().board_size)];
      if (tile.kind == TileKind::PLANT) {
        const int crop = int(tile.crop);
        if (crop >= 0 && crop < N_CROPS) {
          if (unit_action.op == Op::WATER) state.crop_water_actions[crop]++;
          else if (unit_action.op == Op::HARVEST) state.crop_harvest_actions[crop]++;
          else if (unit_action.op == Op::FERTILIZE) state.crop_fertilize_actions[crop]++;
        }
      } else if (tile.kind == TileKind::ANIMAL) {
        const int animal = int(tile.animal) - 9;
        if (animal >= 0 && animal < N_ANIMALS) {
          if (unit_action.op == Op::FEED) state.animal_feed_actions[animal]++;
          else if (unit_action.op == Op::CARE) state.animal_care_actions[animal]++;
          else if (unit_action.op == Op::HARVEST) state.animal_product_actions[animal]++;
          else if (unit_action.op == Op::COLLECT_FERTILIZER)
            state.animal_fertilizer_actions[animal]++;
        }
      }
    }
  }
  state.override_actions += int(out.market.size());
  state.last_step = step;
  return out;
}

AdaptiveMatchResult NativeAdaptiveExecutor::play(const AdaptiveGenome& genome,
                                                 int opponent_route,
                                                 uint64_t seed,
                                                 int candidate_seat,
                                                 bool capture_trace) const {
  return play_blend(genome, -1, opponent_route, seed, candidate_seat, 0,
                    capture_trace);
}

AdaptiveMatchResult NativeAdaptiveExecutor::play_forced_backbone(
    const AdaptiveGenome& genome, int opponent_route, uint64_t seed,
    int candidate_seat, bool capture_trace) const {
  if (backbones_.empty())
    throw std::invalid_argument(
        "play_forced_backbone requires one semantic backbone");
  return play_blend(genome, -1, opponent_route, seed, candidate_seat, 0,
                    capture_trace, 1, true);
}

AdaptiveMatchResult NativeAdaptiveExecutor::play_blend(
    const AdaptiveGenome& genome, int base_route, int opponent_route,
    uint64_t seed, int candidate_seat, int blend_mode,
    bool capture_trace, int prefix_steps, bool force_backbone_exact) const {
  if (candidate_seat != 0 && candidate_seat != 1)
    throw std::invalid_argument("candidate_seat must be 0 or 1");
  if (opponent_route >= route_executor_.route_count())
    throw std::invalid_argument("invalid opponent route");
  if (blend_mode < 0 || blend_mode > 6)
    throw std::invalid_argument("blend_mode must be in [0, 6]");
  if (prefix_steps < 0 || prefix_steps > 719)
    throw std::invalid_argument("prefix_steps must be in [0, 719]");
  if (blend_mode != 0 &&
      (base_route < 0 || base_route >= route_executor_.route_count()))
    throw std::invalid_argument("invalid base route");
  Simulator env(Config{}, seed);
  NativeAdaptivePlanner planner(genome, backbones_, force_backbone_exact);
  AdaptivePlannerState adaptive_state;
  NativeAgentState base_state;
  base_state.reset();
  NativeAgentState opponent_state;
  opponent_state.reset();
  AdaptiveMatchResult result;
  result.candidate_seat = candidate_seat;
  if (capture_trace) result.trace.reserve(719);
  while (!env.done()) {
    std::array<PlayerAction, 2> joint;
    const int replans_before = adaptive_state.replans;
    const int prior_day_before = adaptive_state.operating_prior_selection_day;
    const bool base_prefix =
        blend_mode == 6 && env.step_count() < prefix_steps;
    if (base_prefix) {
      // Do not call planner.action here: doing so would mutate its persistent
      // plan as if the discarded adaptive action had actually been executed.
      // At the handoff step the planner initializes directly from the settled
      // public state produced by the frozen opening prefix.
      joint[candidate_seat] = route_executor_.action_for(
          env, candidate_seat, base_route, base_state);
    } else {
      joint[candidate_seat] = planner.action(
          env, candidate_seat, adaptive_state);
      if (capture_trace &&
          adaptive_state.operating_prior_selection_day != prior_day_before)
        result.operating_prior_trace.emplace_back(
            env.step_count(), int(adaptive_state.operating_prior_index));
    }
    if (blend_mode != 0 && blend_mode != 6) {
      PlayerAction base = route_executor_.action_for(
          env, candidate_seat, base_route, base_state);
      if (blend_mode == 1 || blend_mode == 3 || blend_mode == 4 ||
          blend_mode == 5)
        joint[candidate_seat].units = std::move(base.units);
      if (blend_mode == 2 || blend_mode == 3 || blend_mode == 4 ||
          blend_mode == 5)
        joint[candidate_seat].market = std::move(base.market);
      // Offline causal ablations only.  Mode 4 removes the public WHEAT
      // inventory shuttle; mode 5 caps each shuttle order at four units.  The
      // production planner never selects these modes, so this cannot become a
      // hidden route or opponent-specific policy branch.
      if (blend_mode == 4 || blend_mode == 5) {
        auto& market = joint[candidate_seat].market;
        for (auto& action : market) {
          const bool wheat_shuttle = action.item == Item::WHEAT &&
              (action.op == Op::BUY_PRODUCT || action.op == Op::SELL);
          if (!wheat_shuttle) continue;
          action.quantity = blend_mode == 4 ? 0 : std::min(action.quantity, 4);
        }
      }
    }
    if (adaptive_state.replans != replans_before) {
      for (int animal = 0; animal < N_ANIMALS; ++animal)
        result.max_animal_targets[animal] = std::max(
            result.max_animal_targets[animal], adaptive_state.plan.animal_targets[animal]);
      for (int crop = 0; crop < N_CROPS; ++crop)
        result.max_crop_targets[crop] = std::max(
            result.max_crop_targets[crop], adaptive_state.plan.crop_targets[crop]);
      if (capture_trace)
        result.plan_trace.emplace_back(env.step_count(), adaptive_state.plan);
    }
    if (opponent_route >= 0)
      joint[1 - candidate_seat] = route_executor_.action_for(
          env, 1 - candidate_seat, opponent_route, opponent_state);
    else {
      const auto& farm = env.farms()[1 - candidate_seat];
      joint[1 - candidate_seat].units.assign(farm.hands.size() + 1, Action{});
    }
    std::vector<int> risk_crop, risk_animal;
    if (env.hour() == 23) {
      const auto& farm = env.farms()[candidate_seat];
      for (int i = 0; i < int(farm.tiles.size()); ++i) {
        const auto& tile = farm.tiles[i];
        if (tile.kind == TileKind::PLANT && tile.consecutive_unwatered >= 1 && !tile.watered_today)
          risk_crop.push_back(i);
        if (tile.kind == TileKind::ANIMAL &&
            tile.consecutive_unfed >= 1 && !tile.fed_today) {
          bool planned_terminal_release = false;
          if (genome.terminal_animal_economics > 0.0 &&
              env.day() >= TOTAL_DAYS - 2) {
            const int animal = std::clamp(
                int(tile.animal) - 9, 0, N_ANIMALS - 1);
            const int delta = env.day() + 1 - tile.placed_day -
                ANIMAL_FIRST[animal];
            const bool produces_tonight = delta >= 0 &&
                delta % std::max(1, ANIMAL_INTERVAL[animal]) == 0;
            planned_terminal_release = !produces_tonight ||
                tile.yield_units >= ANIMAL_MAX_HELD[animal];
          }
          if (!planned_terminal_release) risk_animal.push_back(i);
        }
      }
    }
    if (capture_trace) result.trace.push_back(joint);
    env.step(joint);
    if (!risk_crop.empty() || !risk_animal.empty()) {
      const auto& farm = env.farms()[candidate_seat];
      for (int i : risk_crop) if (farm.tiles[i].kind == TileKind::WEED) result.avoidable_crop_losses++;
      for (int i : risk_animal)
        if (farm.tiles[i].kind == TileKind::PASTURE || farm.tiles[i].kind == TileKind::COOP)
          result.avoidable_animal_losses++;
    }
    result.end_overflow += env.last_end_of_day_overflow()[candidate_seat];
  }
  result.rewards = {env.farms()[0].money, env.farms()[1].money};
  const auto final_counts = counts_for(env, candidate_seat);
  for (int animal = 0; animal < N_ANIMALS; ++animal)
    result.final_animals[animal] = int16_t(final_counts.total_animals[animal]);
  result.replans = adaptive_state.replans;
  result.override_actions = adaptive_state.override_actions;
  result.final_operating_prior_index = int(adaptive_state.operating_prior_index);
  result.operating_prior_switches = adaptive_state.operating_prior_switches;
  result.bundle_switches = adaptive_state.bundle_switches;
  result.bundle_predicted_gain = adaptive_state.bundle_predicted_gain;
  result.first_bundle_switch_day = adaptive_state.first_bundle_switch_day;
  result.first_bundle_switch_features =
      adaptive_state.first_bundle_switch_features;
  result.candidate8_decisions = adaptive_state.candidate8_decisions;
  result.first_candidate8_day = adaptive_state.first_candidate8_day;
  result.first_candidate8_family = adaptive_state.first_candidate8_family;
  result.first_candidate8_signature =
      adaptive_state.first_candidate8_signature;
  result.first_candidate8_raw_count =
      adaptive_state.first_candidate8_raw_count;
  result.first_candidate8_feasible_count =
      adaptive_state.first_candidate8_feasible_count;
  result.first_candidate8_shortlist_count =
      adaptive_state.first_candidate8_shortlist_count;
  result.first_candidate8_target_delta =
      adaptive_state.first_candidate8_target_delta;
  result.first_candidate8_hand_delta =
      adaptive_state.first_candidate8_hand_delta;
  result.first_candidate8_quadrant_delta =
      adaptive_state.first_candidate8_quadrant_delta;
  result.first_candidate8_effective_delay_days =
      adaptive_state.first_candidate8_effective_delay_days;
  result.first_candidate8_schedule_profile =
      adaptive_state.first_candidate8_schedule_profile;
  result.first_candidate8_market_profile =
      adaptive_state.first_candidate8_market_profile;
  result.first_candidate8_recovery_profile =
      adaptive_state.first_candidate8_recovery_profile;
  result.first_candidate8_suffix_project =
      adaptive_state.first_candidate8_suffix_project;
  result.first_candidate8_market_item =
      adaptive_state.first_candidate8_market_item;
  result.first_candidate8_recovery_issue =
      adaptive_state.first_candidate8_recovery_issue;
  result.first_candidate8_estimated_value =
      adaptive_state.first_candidate8_estimated_value;
  result.first_candidate8_estimated_cash_cost =
      adaptive_state.first_candidate8_estimated_cash_cost;
  result.first_candidate8_estimated_daily_action_load =
      adaptive_state.first_candidate8_estimated_daily_action_load;
  result.first_candidate8_context_features =
      adaptive_state.first_candidate8_context_features;
  result.r8_day_plan_rebuilds = adaptive_state.r8_day_plan_rebuilds;
  result.r8_rolling_updates = adaptive_state.r8_rolling_updates;
  result.r8_task_nodes_created = adaptive_state.r8_task_nodes_created;
  result.r8_task_reassignments = adaptive_state.r8_task_reassignments;
  result.r8_joint_matches = adaptive_state.r8_joint_matches;
  result.r8_lookahead_evaluations =
      adaptive_state.r8_lookahead_evaluations;
  result.r8_idle_unit_actions = adaptive_state.r8_idle_unit_actions;
  result.r8_peak_active_tasks = adaptive_state.r8_peak_active_tasks;
  result.r8_move_unit_actions = adaptive_state.r8_move_unit_actions;
  result.r8_resolved_task_nodes = adaptive_state.r8_resolved_task_nodes;
  auto r8_latency_quantile = [&](int percentile) {
    if (adaptive_state.r8_resolved_task_nodes <= 0) return 0;
    const int target = std::max(
        1, (adaptive_state.r8_resolved_task_nodes * percentile + 99) / 100);
    int cumulative = 0;
    for (int latency = 0;
         latency < int(adaptive_state.r8_resolution_latency_histogram.size());
         ++latency) {
      cumulative +=
          adaptive_state.r8_resolution_latency_histogram[latency];
      if (cumulative >= target) return latency;
    }
    return 24;
  };
  result.r8_resolution_latency_p50 = r8_latency_quantile(50);
  result.r8_resolution_latency_p95 = r8_latency_quantile(95);
  // "Unresolved" is an outcome diagnostic, not a count of still-visible
  // alternative task records at a deadline.  Only a realized hard loss after
  // the state transition qualifies; optional DROP lanes and automatic EOD
  // return must not create false failures.
  result.r8_unresolved_hard_day_tasks =
      result.avoidable_crop_losses + result.avoidable_animal_losses;
  result.r8_duplicate_reservation_violations =
      adaptive_state.r8_duplicate_reservation_violations;
  return result;
}

AdaptiveMatchResult NativeAdaptiveExecutor::play_candidate8(
    const AdaptiveGenome& genome, int opponent_route, uint64_t seed,
    int candidate_seat, int candidate_rank, int minimum_decision_day,
    bool use_feasible_pool, bool capture_trace) const {
  if (candidate_rank < 0 || candidate_rank >= 4096)
    throw std::invalid_argument("candidate_rank must be in [0, 4095]");
  if (minimum_decision_day < 0 || minimum_decision_day >= TOTAL_DAYS)
    throw std::invalid_argument("minimum_decision_day must be in [0, 29]");
  AdaptiveGenome forced = genome;
  forced.offline_candidate8_enabled = true;
  forced.offline_candidate8_rank = candidate_rank;
  forced.offline_candidate8_minimum_day = minimum_decision_day;
  forced.offline_candidate8_use_feasible_pool = use_feasible_pool;
  return play(forced, opponent_route, seed, candidate_seat, capture_trace);
}

AdaptiveCandidate8CommittedSequenceResult
NativeAdaptiveExecutor::candidate8_committed_sequence(
    const AdaptiveGenome& genome, int opponent_route, uint64_t actual_seed,
    const std::vector<int>& decision_days,
    const std::vector<int>& selected_ranks, int candidate_seat,
    bool use_feasible_pool, int prefix_route, int prefix_steps,
    bool capture_trace) const {
  if (candidate_seat != 0 && candidate_seat != 1)
    throw std::invalid_argument("candidate_seat must be 0 or 1");
  if (opponent_route < -1 || opponent_route >= route_executor_.route_count())
    throw std::invalid_argument("invalid opponent route");
  if (prefix_route < -1 || prefix_route >= route_executor_.route_count())
    throw std::invalid_argument("invalid prefix route");
  if (prefix_steps < 0 || prefix_steps > TOTAL_DAYS * 24)
    throw std::invalid_argument("prefix_steps must be in [0, 720]");
  if ((prefix_route < 0) != (prefix_steps == 0))
    throw std::invalid_argument(
        "prefix_route and positive prefix_steps must be supplied together");
  if (decision_days.size() != selected_ranks.size())
    throw std::invalid_argument(
        "decision_days and selected_ranks must have equal length");
  for (size_t index = 0; index < decision_days.size(); ++index) {
    if (decision_days[index] < 0 || decision_days[index] >= TOTAL_DAYS)
      throw std::invalid_argument("decision days must be in [0, 29]");
    if (index > 0 && decision_days[index] <= decision_days[index - 1])
      throw std::invalid_argument(
          "decision days must be strictly increasing");
    if (selected_ranks[index] < 0 || selected_ranks[index] >= 4096)
      throw std::invalid_argument("selected ranks must be in [0, 4095]");
  }

  AdaptiveCandidate8CommittedSequenceResult result;
  result.candidate_seat = candidate_seat;
  Simulator env(Config{}, actual_seed);
  AdaptivePlannerState state;
  NativeAgentState prefix_state;
  prefix_state.reset();
  NativeAgentState opponent_state;
  opponent_state.reset();
  NativeAdaptivePlanner ordinary_planner(genome, backbones_);
  size_t stage = 0;
  while (!env.done()) {
    std::array<PlayerAction, 2> joint;
    const bool use_prefix = prefix_route >= 0 &&
        env.step_count() < prefix_steps;
    const bool commit = !use_prefix && stage < decision_days.size() &&
        env.day() >= decision_days[stage];
    if (use_prefix) {
      joint[candidate_seat] = route_executor_.action_for(
          env, candidate_seat, prefix_route, prefix_state);
    } else if (commit) {
      AdaptiveGenome selected_genome = genome;
      selected_genome.offline_candidate8_enabled = true;
      selected_genome.offline_candidate8_rank = selected_ranks[stage];
      selected_genome.offline_candidate8_minimum_day = decision_days[stage];
      selected_genome.offline_candidate8_use_feasible_pool =
          use_feasible_pool;
      NativeAdaptivePlanner selected_planner(selected_genome, backbones_);
      state.candidate8_decisions = 0;
      state.last_plan_step = -1;
      joint[candidate_seat] = selected_planner.action(
          env, candidate_seat, state);
      result.decision_day.push_back(int16_t(env.day()));
      result.selected_rank.push_back(int16_t(selected_ranks[stage]));
      result.selected_family.push_back(state.first_candidate8_family);
      result.selected_signature.push_back(state.first_candidate8_signature);
      ++stage;
    } else {
      joint[candidate_seat] = ordinary_planner.action(
          env, candidate_seat, state);
    }
    if (opponent_route >= 0)
      joint[1 - candidate_seat] = route_executor_.action_for(
          env, 1 - candidate_seat, opponent_route, opponent_state);
    else
      joint[1 - candidate_seat].units.assign(
          env.farms()[1 - candidate_seat].hands.size() + 1, Action{});

    std::vector<int> risk_crop, risk_animal;
    if (env.hour() == 23) {
      const auto& farm = env.farms()[candidate_seat];
      for (int tile_index = 0; tile_index < int(farm.tiles.size());
           ++tile_index) {
        const auto& tile = farm.tiles[tile_index];
        if (tile.kind == TileKind::PLANT &&
            tile.consecutive_unwatered >= 1 && !tile.watered_today)
          risk_crop.push_back(tile_index);
        if (tile.kind == TileKind::ANIMAL &&
            tile.consecutive_unfed >= 1 && !tile.fed_today) {
          bool planned_terminal_release = false;
          if (genome.terminal_animal_economics > 0.0 &&
              env.day() >= TOTAL_DAYS - 2) {
            const int animal = std::clamp(
                int(tile.animal) - 9, 0, N_ANIMALS - 1);
            const int delta = env.day() + 1 - tile.placed_day -
                ANIMAL_FIRST[animal];
            const bool produces_tonight = delta >= 0 &&
                delta % std::max(1, ANIMAL_INTERVAL[animal]) == 0;
            planned_terminal_release = !produces_tonight ||
                tile.yield_units >= ANIMAL_MAX_HELD[animal];
          }
          if (!planned_terminal_release) risk_animal.push_back(tile_index);
        }
      }
    }
    if (capture_trace) result.trace.push_back(joint);
    env.step(joint);
    if (!risk_crop.empty() || !risk_animal.empty()) {
      const auto& farm = env.farms()[candidate_seat];
      for (const int tile_index : risk_crop)
        if (farm.tiles[tile_index].kind == TileKind::WEED)
          ++result.avoidable_crop_losses;
      for (const int tile_index : risk_animal)
        if (farm.tiles[tile_index].kind == TileKind::PASTURE ||
            farm.tiles[tile_index].kind == TileKind::COOP)
          ++result.avoidable_animal_losses;
    }
    result.end_overflow +=
        env.last_end_of_day_overflow()[candidate_seat];
  }
  result.rewards = {env.farms()[0].money, env.farms()[1].money};
  return result;
}

AdaptiveCandidate8CounterfactualResult
NativeAdaptiveExecutor::candidate8_counterfactual(
    const AdaptiveGenome& genome, int opponent_route, uint64_t prefix_seed,
    const std::vector<uint64_t>& future_seeds, int candidate_seat,
    int minimum_decision_day, bool use_feasible_pool,
    int maximum_arms, const std::vector<int>& committed_days,
    const std::vector<int>& committed_ranks,
    bool include_response_scenarios) const {
  if (candidate_seat != 0 && candidate_seat != 1)
    throw std::invalid_argument("candidate_seat must be 0 or 1");
  if (opponent_route < -1 || opponent_route >= route_executor_.route_count())
    throw std::invalid_argument("invalid opponent route");
  if (future_seeds.empty())
    throw std::invalid_argument("at least one future seed is required");
  if (minimum_decision_day < 0 || minimum_decision_day >= TOTAL_DAYS)
    throw std::invalid_argument("minimum_decision_day must be in [0, 29]");
  if (maximum_arms <= 0 || maximum_arms > 4096)
    throw std::invalid_argument("maximum_arms must be in [1, 4096]");
  if (committed_days.size() != committed_ranks.size())
    throw std::invalid_argument(
        "committed_days and committed_ranks must have equal length");
  for (size_t index = 0; index < committed_days.size(); ++index) {
    if (committed_days[index] < 0 ||
        committed_days[index] >= minimum_decision_day)
      throw std::invalid_argument(
          "committed days must be in [0, minimum_decision_day)");
    if (index > 0 && committed_days[index] <= committed_days[index - 1])
      throw std::invalid_argument(
          "committed days must be strictly increasing");
    if (committed_ranks[index] < 0 || committed_ranks[index] >= 4096)
      throw std::invalid_argument("committed ranks must be in [0, 4095]");
  }

  AdaptiveCandidate8CounterfactualResult result;
  result.future_count = int(future_seeds.size());

  // Advance one ordinary policy to the first public-state replan where the
  // broad candidate grammar is available.  Probing uses a copied planner
  // state, so the prefix is uncontaminated by the candidate arm.
  Simulator prefix(Config{}, prefix_seed);
  NativeAdaptivePlanner keep_planner(genome, backbones_);
  AdaptivePlannerState prefix_state;
  NativeAgentState prefix_opponent_state;
  prefix_opponent_state.reset();
  Simulator checkpoint = prefix;
  AdaptivePlannerState checkpoint_state;
  NativeAgentState checkpoint_opponent_state;
  AdaptivePlannerState probe_candidate_state;
  size_t committed_index = 0;
  while (!prefix.done()) {
    if (committed_index < committed_days.size() &&
        prefix.day() >= committed_days[committed_index]) {
      AdaptiveGenome committed_genome = genome;
      committed_genome.offline_candidate8_enabled = true;
      committed_genome.offline_candidate8_rank =
          committed_ranks[committed_index];
      committed_genome.offline_candidate8_minimum_day =
          committed_days[committed_index];
      committed_genome.offline_candidate8_use_feasible_pool =
          use_feasible_pool;
      NativeAdaptivePlanner committed_planner(
          committed_genome, backbones_);
      prefix_state.candidate8_decisions = 0;
      prefix_state.last_plan_step = -1;
      std::array<PlayerAction, 2> joint;
      joint[candidate_seat] = committed_planner.action(
          prefix, candidate_seat, prefix_state);
      if (opponent_route >= 0)
        joint[1 - candidate_seat] = route_executor_.action_for(
            prefix, 1 - candidate_seat, opponent_route,
            prefix_opponent_state);
      else
        joint[1 - candidate_seat].units.assign(
            prefix.farms()[1 - candidate_seat].hands.size() + 1,
            Action{});
      prefix.step(joint);
      ++committed_index;
      continue;
    }
    if (prefix.day() >= minimum_decision_day) {
      AdaptiveGenome probe_genome = genome;
      probe_genome.offline_candidate8_enabled = true;
      probe_genome.offline_candidate8_rank = 0;
      probe_genome.offline_candidate8_minimum_day = minimum_decision_day;
      probe_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
      NativeAdaptivePlanner probe_planner(probe_genome, backbones_);
      probe_candidate_state = prefix_state;
      // A prior committed Candidate8 decision leaves this counter at one.
      // Reset only the decision trigger and force one fresh replan so the
      // requested later-day pool is generated from the inherited live state,
      // rather than accidentally reusing the first decision's metadata.
      probe_candidate_state.candidate8_decisions = 0;
      probe_candidate_state.last_plan_step = -1;
      probe_planner.action(prefix, candidate_seat, probe_candidate_state);
      if (probe_candidate_state.candidate8_decisions > 0) {
        checkpoint = prefix;
        checkpoint_state = prefix_state;
        checkpoint_opponent_state = prefix_opponent_state;
        result.decision_found = true;
        result.decision_step = prefix.step_count();
        break;
      }
    }
    std::array<PlayerAction, 2> joint;
    joint[candidate_seat] = keep_planner.action(
        prefix, candidate_seat, prefix_state);
    if (opponent_route >= 0)
      joint[1 - candidate_seat] = route_executor_.action_for(
          prefix, 1 - candidate_seat, opponent_route,
          prefix_opponent_state);
    else
      joint[1 - candidate_seat].units.assign(
          prefix.farms()[1 - candidate_seat].hands.size() + 1, Action{});
    prefix.step(joint);
  }
  if (!result.decision_found) return result;

  const int available = use_feasible_pool
      ? int(probe_candidate_state.first_candidate8_feasible_count)
      : int(probe_candidate_state.first_candidate8_shortlist_count);
  result.arms = std::min(maximum_arms, std::max(0, available));
  result.arm_family.assign(result.arms, int8_t(-1));
  result.arm_signature.assign(result.arms, uint64_t(0));
  result.arm_features.resize(result.arms);
  result.arm_consequence_features.resize(result.arms);
  result.context_features =
      probe_candidate_state.first_candidate8_context_features;

  auto capture_candidate = [&](int arm, const AdaptivePlannerState& state) {
    result.arm_family[arm] = state.first_candidate8_family;
    result.arm_signature[arm] = state.first_candidate8_signature;
    auto& features = result.arm_features[arm];
    int feature = 0;
    for (const int value : state.first_candidate8_target_delta)
      features[feature++] = value;
    features[feature++] = state.first_candidate8_hand_delta;
    features[feature++] = state.first_candidate8_quadrant_delta;
    features[feature++] = state.first_candidate8_effective_delay_days;
    features[feature++] = state.first_candidate8_schedule_profile;
    features[feature++] = state.first_candidate8_market_profile;
    features[feature++] = state.first_candidate8_recovery_profile;
    features[feature++] = state.first_candidate8_suffix_project;
    features[feature++] = state.first_candidate8_market_item;
    features[feature++] = state.first_candidate8_recovery_issue;
    features[feature++] = int32_t(std::nearbyint(
        100.0 * state.first_candidate8_estimated_value));
    features[feature++] = int32_t(std::nearbyint(
        100.0 * state.first_candidate8_estimated_cash_cost));
    features[feature++] = int32_t(std::nearbyint(
        100.0 * state.first_candidate8_estimated_daily_action_load));
    assert(feature ==
           AdaptiveCandidate8CounterfactualResult::CANDIDATE_FEATURE_DIM);
  };
  for (int arm = 0; arm < result.arms; ++arm) {
    AdaptiveGenome arm_genome = genome;
    arm_genome.offline_candidate8_enabled = true;
    arm_genome.offline_candidate8_rank = arm;
    arm_genome.offline_candidate8_minimum_day = minimum_decision_day;
    arm_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
    NativeAdaptivePlanner planner(arm_genome, backbones_);
    AdaptivePlannerState state = checkpoint_state;
    state.candidate8_decisions = 0;
    state.last_plan_step = -1;
    planner.action(checkpoint, candidate_seat, state);
    capture_candidate(arm, state);
  }

  // Execute every arm for two days in one canonical, non-clairvoyant world.
  // This is deliberately separate from the terminal label rollouts below.
  // The fixed seed is a contract constant; it never depends on prefix_seed,
  // future_seeds, opponent_route, rewards or candidate identity.
  constexpr uint64_t PREVIEW_SEED = 0xC81424048ULL;
  #pragma omp parallel for schedule(dynamic, 1)
  for (int arm = 0; arm < result.arms; ++arm) {
    Simulator env = checkpoint;
    env.reseed_future(PREVIEW_SEED);
    AdaptiveGenome arm_genome = genome;
    arm_genome.offline_candidate8_enabled = true;
    arm_genome.offline_candidate8_rank = arm;
    arm_genome.offline_candidate8_minimum_day = minimum_decision_day;
    arm_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
    NativeAdaptivePlanner planner(arm_genome, backbones_);
    AdaptivePlannerState state = checkpoint_state;
    state.candidate8_decisions = 0;
    state.last_plan_step = -1;
    CandidatePreviewAccumulator accumulator;
    accumulator.minimum_money = env.farms()[candidate_seat].money;
    int feature = 0;
    int offset = 0;
    while (!env.done()) {
      std::array<PlayerAction, 2> joint;
      joint[candidate_seat] = planner.action(env, candidate_seat, state);
      joint[1 - candidate_seat].units.assign(
          env.farms()[1 - candidate_seat].hands.size() + 1, Action{});
      preview_accumulate_action(joint[candidate_seat], accumulator);
      env.step(joint);
      preview_accumulate_effects(env, candidate_seat, accumulator);
      if (offset == 23 || offset == 47)
        capture_candidate_preview(
            env, candidate_seat, accumulator,
            result.arm_consequence_features[arm], feature);
      ++offset;
    }
    // Decision days 6/12/18 always have at least 48 remaining turns. Retain a
    // defensive fill for future uses near the terminal boundary, then append
    // the complete intrinsic consequence under this same synthetic world.
    if (feature == 76)
      capture_candidate_preview(
          env, candidate_seat, accumulator,
          result.arm_consequence_features[arm], feature);
    capture_candidate_preview(
        env, candidate_seat, accumulator,
        result.arm_consequence_features[arm], feature);
    assert(feature ==
        AdaptiveCandidate8CounterfactualResult::CONSEQUENCE_FEATURE_DIM);
  }

  if (include_response_scenarios) {
    constexpr int SCENARIOS =
        AdaptiveCandidate8CounterfactualResult::RESPONSE_SCENARIO_COUNT;
    const int scenario_rows = result.arms * SCENARIOS;
    result.arm_response_scenario_features.resize(scenario_rows);
    result.arm_response_scenario_outcomes.resize(scenario_rows);
    const int opponent = 1 - candidate_seat;
    const PrivateState public_belief = public_opponent_private_belief(
        checkpoint, opponent, checkpoint_state);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int row = 0; row < scenario_rows; ++row) {
      const int arm = row / SCENARIOS;
      const int scenario = row % SCENARIOS;
      Simulator env = checkpoint;
      env.reseed_future(PREVIEW_SEED);
      env.replace_private_for_public_counterfactual(opponent, public_belief);

      AdaptiveGenome arm_genome = genome;
      arm_genome.offline_candidate8_enabled = true;
      arm_genome.offline_candidate8_rank = arm;
      arm_genome.offline_candidate8_minimum_day = minimum_decision_day;
      arm_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
      NativeAdaptivePlanner planner(arm_genome, backbones_);
      AdaptivePlannerState state = checkpoint_state;
      state.candidate8_decisions = 0;
      state.last_plan_step = -1;

      const AdaptiveGenome rival_genome = response_scenario_genome(
          genome, scenario, env.day());
      NativeAdaptivePlanner rival_planner(rival_genome, {});
      AdaptivePlannerState rival_state;
      rival_state.reset();
      CandidatePreviewAccumulator accumulator;
      accumulator.minimum_money = env.farms()[candidate_seat].money;
      int feature = 0;
      int horizon = 0;
      int offset = 0;
      auto capture_outcome = [&]() {
        const int base = horizon * 3;
        const double own = env.farms()[candidate_seat].money;
        const double rival = env.farms()[opponent].money;
        result.arm_response_scenario_outcomes[row][base] =
            int32_t(std::nearbyint(100.0 * own));
        result.arm_response_scenario_outcomes[row][base + 1] =
            int32_t(std::nearbyint(100.0 * rival));
        result.arm_response_scenario_outcomes[row][base + 2] =
            int32_t(std::nearbyint(100.0 * (own - rival)));
        ++horizon;
      };
      while (!env.done()) {
        std::array<PlayerAction, 2> joint;
        joint[candidate_seat] = planner.action(env, candidate_seat, state);
        if (scenario == 0) {
          joint[opponent].units.assign(
              env.farms()[opponent].hands.size() + 1, Action{});
        } else {
          joint[opponent] = rival_planner.action(env, opponent, rival_state);
        }
        preview_accumulate_action(joint[candidate_seat], accumulator);
        env.step(joint);
        preview_accumulate_effects(env, candidate_seat, accumulator);
        if (offset == 23 || offset == 47) {
          capture_candidate_preview(
              env, candidate_seat, accumulator,
              result.arm_response_scenario_features[row], feature);
          capture_outcome();
        }
        ++offset;
      }
      while (feature < 152) {
        capture_candidate_preview(
            env, candidate_seat, accumulator,
            result.arm_response_scenario_features[row], feature);
        capture_outcome();
      }
      capture_candidate_preview(
          env, candidate_seat, accumulator,
          result.arm_response_scenario_features[row], feature);
      capture_outcome();
      assert(feature ==
          AdaptiveCandidate8CounterfactualResult::CONSEQUENCE_FEATURE_DIM);
      assert(horizon == 3);
    }
  }

  const int total = result.future_count * result.arms;
  result.own_rewards.assign(total, std::numeric_limits<double>::quiet_NaN());
  result.opponent_rewards.assign(
      total, std::numeric_limits<double>::quiet_NaN());
  result.end_overflow.assign(total, 0);
  std::vector<std::vector<uint64_t>> probe_action_trace(result.arms);
  std::vector<std::vector<uint64_t>> probe_state_trace(result.arms);
  #pragma omp parallel for schedule(dynamic, 1)
  for (int index = 0; index < total; ++index) {
    const int future = index / result.arms;
    const int arm = index % result.arms;
    Simulator env = checkpoint;
    env.reseed_future(future_seeds[future]);
    AdaptiveGenome arm_genome = genome;
    arm_genome.offline_candidate8_enabled = true;
    arm_genome.offline_candidate8_rank = arm;
    arm_genome.offline_candidate8_minimum_day = minimum_decision_day;
    arm_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
    NativeAdaptivePlanner planner(arm_genome, backbones_);
    AdaptivePlannerState state = checkpoint_state;
    state.candidate8_decisions = 0;
    state.last_plan_step = -1;
    NativeAgentState opponent_state = checkpoint_opponent_state;
    int overflow = 0;
    if (future == 0) {
      const int remaining = std::max(
          0, env.config().episode_steps - env.step_count());
      probe_action_trace[arm].reserve(remaining);
      probe_state_trace[arm].reserve(remaining);
    }
    while (!env.done()) {
      std::array<PlayerAction, 2> joint;
      joint[candidate_seat] = planner.action(env, candidate_seat, state);
      if (opponent_route >= 0)
        joint[1 - candidate_seat] = route_executor_.action_for(
            env, 1 - candidate_seat, opponent_route, opponent_state);
      else
        joint[1 - candidate_seat].units.assign(
            env.farms()[1 - candidate_seat].hands.size() + 1, Action{});
      if (future == 0)
        probe_action_trace[arm].push_back(
            player_action_signature(joint[candidate_seat]));
      env.step(joint);
      if (future == 0)
        probe_state_trace[arm].push_back(simulator_effect_signature(env));
      overflow += env.last_end_of_day_overflow()[candidate_seat];
    }
    result.own_rewards[index] = env.farms()[candidate_seat].money;
    result.opponent_rewards[index] = env.farms()[1 - candidate_seat].money;
    result.end_overflow[index] = overflow;
  }
  result.arm_first_action_change_offset.assign(result.arms, int16_t(-1));
  result.arm_action_change_count_24.assign(result.arms, int16_t(0));
  result.arm_action_change_count_full.assign(result.arms, int16_t(0));
  result.arm_first_state_change_offset.assign(result.arms, int16_t(-1));
  result.arm_state_change_count_24.assign(result.arms, int16_t(0));
  result.arm_state_change_count_full.assign(result.arms, int16_t(0));
  if (result.arms > 0) {
    const auto compare_trace = [](const std::vector<uint64_t>& keep,
                                  const std::vector<uint64_t>& candidate,
                                  int16_t& first_change,
                                  int16_t& first_day_count,
                                  int16_t& full_count) {
      const size_t length = std::max(keep.size(), candidate.size());
      int first = -1;
      int day_count = 0;
      int total_count = 0;
      for (size_t offset = 0; offset < length; ++offset) {
        const bool changed = offset >= keep.size() ||
            offset >= candidate.size() || keep[offset] != candidate[offset];
        if (!changed) continue;
        if (first < 0) first = int(offset);
        if (offset < 24) ++day_count;
        ++total_count;
      }
      first_change = int16_t(std::min(first, 32767));
      first_day_count = int16_t(std::min(day_count, 32767));
      full_count = int16_t(std::min(total_count, 32767));
    };
    for (int arm = 1; arm < result.arms; ++arm) {
      compare_trace(
          probe_action_trace[0], probe_action_trace[arm],
          result.arm_first_action_change_offset[arm],
          result.arm_action_change_count_24[arm],
          result.arm_action_change_count_full[arm]);
      compare_trace(
          probe_state_trace[0], probe_state_trace[arm],
          result.arm_first_state_change_offset[arm],
          result.arm_state_change_count_24[arm],
          result.arm_state_change_count_full[arm]);
    }
  }
  return result;
}

AdaptiveCandidate8RollingOracleResult
NativeAdaptiveExecutor::candidate8_rolling_oracle(
    const AdaptiveGenome& genome, int opponent_route, uint64_t actual_seed,
    const std::vector<int>& decision_days, uint64_t future_seed_base,
    int future_count, int candidate_seat, bool use_feasible_pool,
    int maximum_arms, bool clairvoyant_actual_future,
    bool competitive_objective) const {
  if (candidate_seat != 0 && candidate_seat != 1)
    throw std::invalid_argument("candidate_seat must be 0 or 1");
  if (opponent_route < -1 || opponent_route >= route_executor_.route_count())
    throw std::invalid_argument("invalid opponent route");
  if (future_count <= 0 || future_count > 256)
    throw std::invalid_argument("future_count must be in [1, 256]");
  if (maximum_arms <= 0 || maximum_arms > 4096)
    throw std::invalid_argument("maximum_arms must be in [1, 4096]");
  for (size_t index = 0; index < decision_days.size(); ++index) {
    if (decision_days[index] < 0 || decision_days[index] >= TOTAL_DAYS)
      throw std::invalid_argument("decision days must be in [0, 29]");
    if (index > 0 && decision_days[index] <= decision_days[index - 1])
      throw std::invalid_argument("decision days must be strictly increasing");
  }

  AdaptiveCandidate8RollingOracleResult result;
  result.candidate_seat = candidate_seat;
  Simulator actual(Config{}, actual_seed);
  AdaptivePlannerState actual_state;
  NativeAgentState actual_opponent_state;
  actual_opponent_state.reset();
  NativeAdaptivePlanner ordinary_planner(genome, backbones_);

  auto opponent_action = [&](const Simulator& env,
                             NativeAgentState& opponent_state) {
    PlayerAction action;
    if (opponent_route >= 0) {
      action = route_executor_.action_for(
          env, 1 - candidate_seat, opponent_route, opponent_state);
    } else {
      action.units.assign(
          env.farms()[1 - candidate_seat].hands.size() + 1, Action{});
    }
    return action;
  };

  auto advance_one = [&](Simulator& env, AdaptivePlannerState& state,
                         NativeAgentState& opponent_state,
                         const NativeAdaptivePlanner& planner) {
    std::array<PlayerAction, 2> joint;
    joint[candidate_seat] = planner.action(env, candidate_seat, state);
    joint[1 - candidate_seat] = opponent_action(env, opponent_state);
    env.step(joint);
    result.end_overflow +=
        env.last_end_of_day_overflow()[candidate_seat];
  };

  for (size_t stage = 0; stage < decision_days.size(); ++stage) {
    const int requested_day = decision_days[stage];
    while (!actual.done() && actual.day() < requested_day)
      advance_one(actual, actual_state, actual_opponent_state,
                  ordinary_planner);
    if (actual.done()) break;

    // Force a fresh plan exactly at this stage.  Resetting this diagnostic
    // counter does not reset the economic plan, task chains, observed market
    // history, or any official state carried from earlier selected arms.
    AdaptivePlannerState probe_state = actual_state;
    probe_state.candidate8_decisions = 0;
    probe_state.last_plan_step = -1;
    AdaptiveGenome probe_genome = genome;
    probe_genome.offline_candidate8_enabled = true;
    probe_genome.offline_candidate8_rank = 0;
    probe_genome.offline_candidate8_minimum_day = requested_day;
    probe_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
    NativeAdaptivePlanner probe_planner(probe_genome, backbones_);
    probe_planner.action(actual, candidate_seat, probe_state);
    if (probe_state.candidate8_decisions <= 0) continue;

    const int available = use_feasible_pool
        ? int(probe_state.first_candidate8_feasible_count)
        : int(probe_state.first_candidate8_shortlist_count);
    const int arms = std::min(maximum_arms, std::max(0, available));
    if (arms <= 0) continue;
    const int evaluation_futures = clairvoyant_actual_future ? 1 : future_count;
    const int total = evaluation_futures * arms;
    std::vector<double> own_rewards(
        total, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> opponent_rewards(
        total, std::numeric_limits<double>::quiet_NaN());

    #pragma omp parallel for schedule(dynamic, 1)
    for (int index = 0; index < total; ++index) {
      const int future = index / arms;
      const int arm = index % arms;
      Simulator env = actual;
      if (!clairvoyant_actual_future) {
        const uint64_t future_seed = future_seed_base +
            uint64_t(stage) * uint64_t(1000003) + uint64_t(future);
        env.reseed_future(future_seed);
      }
      AdaptivePlannerState state = actual_state;
      state.candidate8_decisions = 0;
      state.last_plan_step = -1;
      NativeAgentState opponent_state = actual_opponent_state;
      AdaptiveGenome arm_genome = genome;
      arm_genome.offline_candidate8_enabled = true;
      arm_genome.offline_candidate8_rank = arm;
      arm_genome.offline_candidate8_minimum_day = requested_day;
      arm_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
      NativeAdaptivePlanner arm_planner(arm_genome, backbones_);
      while (!env.done()) {
        std::array<PlayerAction, 2> joint;
        joint[candidate_seat] = arm_planner.action(
            env, candidate_seat, state);
        if (opponent_route >= 0) {
          joint[1 - candidate_seat] = route_executor_.action_for(
              env, 1 - candidate_seat, opponent_route, opponent_state);
        } else {
          joint[1 - candidate_seat].units.assign(
              env.farms()[1 - candidate_seat].hands.size() + 1, Action{});
        }
        env.step(joint);
      }
      own_rewards[index] = env.farms()[candidate_seat].money;
      opponent_rewards[index] = env.farms()[1 - candidate_seat].money;
    }
    result.complete_continuations += total;

    int selected = 0;
    double selected_mean = -std::numeric_limits<double>::infinity();
    double selected_win_rate = -std::numeric_limits<double>::infinity();
    double selected_margin = -std::numeric_limits<double>::infinity();
    double keep_mean = 0.0;
    double keep_win_rate = 0.0;
    double keep_margin = 0.0;
    for (int arm = 0; arm < arms; ++arm) {
      double own_sum = 0.0;
      double margin_sum = 0.0;
      double win_sum = 0.0;
      for (int future = 0; future < evaluation_futures; ++future) {
        const int index = future * arms + arm;
        const double own = own_rewards[index];
        const double opponent = opponent_rewards[index];
        own_sum += own;
        margin_sum += own - opponent;
        win_sum += own > opponent ? 1.0 : (own == opponent ? 0.5 : 0.0);
      }
      const double mean = own_sum / evaluation_futures;
      const double margin = margin_sum / evaluation_futures;
      const double win_rate = win_sum / evaluation_futures;
      if (arm == 0) {
        keep_mean = mean;
        keep_win_rate = win_rate;
        keep_margin = margin;
      }
      constexpr double EPSILON = 1e-12;
      const bool better_competitive =
          win_rate > selected_win_rate + EPSILON ||
          (std::abs(win_rate - selected_win_rate) <= EPSILON &&
           (margin > selected_margin + EPSILON ||
            (std::abs(margin - selected_margin) <= EPSILON &&
             mean > selected_mean + EPSILON)));
      const bool better_cash = mean > selected_mean + EPSILON;
      if ((competitive_objective && better_competitive) ||
          (!competitive_objective && better_cash)) {
        selected_mean = mean;
        selected_win_rate = win_rate;
        selected_margin = margin;
        selected = arm;
      }
    }

    AdaptiveGenome selected_genome = genome;
    selected_genome.offline_candidate8_enabled = true;
    selected_genome.offline_candidate8_rank = selected;
    selected_genome.offline_candidate8_minimum_day = requested_day;
    selected_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
    NativeAdaptivePlanner selected_planner(selected_genome, backbones_);
    actual_state.candidate8_decisions = 0;
    actual_state.last_plan_step = -1;
    std::array<PlayerAction, 2> selected_joint;
    selected_joint[candidate_seat] = selected_planner.action(
        actual, candidate_seat, actual_state);
    selected_joint[1 - candidate_seat] = opponent_action(
        actual, actual_opponent_state);

    result.decision_day.push_back(int16_t(actual.day()));
    result.decision_step.push_back(int16_t(actual.step_count()));
    result.feasible_count.push_back(int16_t(arms));
    result.selected_rank.push_back(int16_t(selected));
    result.selected_family.push_back(actual_state.first_candidate8_family);
    result.selected_signature.push_back(
        actual_state.first_candidate8_signature);
    result.selected_expected_reward.push_back(selected_mean);
    result.keep_expected_reward.push_back(keep_mean);
    result.stage_expected_gain.push_back(selected_mean - keep_mean);
    result.selected_expected_win_rate.push_back(selected_win_rate);
    result.keep_expected_win_rate.push_back(keep_win_rate);
    result.selected_expected_margin.push_back(selected_margin);
    result.keep_expected_margin.push_back(keep_margin);

    actual.step(selected_joint);
    result.end_overflow +=
        actual.last_end_of_day_overflow()[candidate_seat];
  }

  while (!actual.done())
    advance_one(actual, actual_state, actual_opponent_state,
                ordinary_planner);
  result.rewards[0] = actual.farms()[0].money;
  result.rewards[1] = actual.farms()[1].money;
  result.avoidable_crop_losses = actual_state.avoidable_crop_losses;
  result.avoidable_animal_losses = actual_state.avoidable_animal_losses;
  return result;
}

AdaptiveCandidate8SequenceOracleResult
NativeAdaptiveExecutor::candidate8_sequence_oracle(
    const AdaptiveGenome& genome, int opponent_route, uint64_t actual_seed,
    const std::vector<int>& decision_days, int candidate_seat,
    int beam_width, int per_node_arms, bool use_feasible_pool,
    bool competitive_objective, const std::vector<int>& committed_days,
    const std::vector<std::vector<int>>& committed_rank_sequences,
    int prefix_route, int prefix_steps) const {
  if (candidate_seat != 0 && candidate_seat != 1)
    throw std::invalid_argument("candidate_seat must be 0 or 1");
  if (opponent_route < -1 || opponent_route >= route_executor_.route_count())
    throw std::invalid_argument("invalid opponent route");
  if (prefix_route < -1 || prefix_route >= route_executor_.route_count())
    throw std::invalid_argument("invalid prefix route");
  if (prefix_steps < 0 || prefix_steps > TOTAL_DAYS * 24)
    throw std::invalid_argument("prefix_steps must be in [0, 720]");
  if ((prefix_route < 0) != (prefix_steps == 0))
    throw std::invalid_argument(
        "prefix_route and positive prefix_steps must be supplied together");
  if (decision_days.empty())
    throw std::invalid_argument("at least one decision day is required");
  if (beam_width <= 0 || beam_width > 512)
    throw std::invalid_argument("beam_width must be in [1, 512]");
  if (per_node_arms <= 0 || per_node_arms > 4096)
    throw std::invalid_argument("per_node_arms must be in [1, 4096]");
  for (size_t index = 0; index < decision_days.size(); ++index) {
    if (decision_days[index] < 0 || decision_days[index] >= TOTAL_DAYS)
      throw std::invalid_argument("decision days must be in [0, 29]");
    if (index > 0 && decision_days[index] <= decision_days[index - 1])
      throw std::invalid_argument("decision days must be strictly increasing");
  }
  if (committed_days.empty() != committed_rank_sequences.empty())
    throw std::invalid_argument(
        "committed days and rank sequences must both be empty or nonempty");
  for (size_t index = 0; index < committed_days.size(); ++index) {
    if (committed_days[index] < 0 ||
        committed_days[index] >= decision_days.front())
      throw std::invalid_argument(
          "committed days must precede the first search decision day");
    if (index > 0 && committed_days[index] <= committed_days[index - 1])
      throw std::invalid_argument(
          "committed days must be strictly increasing");
  }
  for (const auto& sequence : committed_rank_sequences) {
    if (sequence.size() != committed_days.size())
      throw std::invalid_argument(
          "every committed rank sequence must match committed days");
    for (const int rank : sequence)
      if (rank < 0 || rank >= 4096)
        throw std::invalid_argument("committed ranks must be in [0, 4095]");
  }

  struct BeamNode {
    Simulator env;
    AdaptivePlannerState state;
    NativeAgentState opponent_state;
    std::vector<int16_t> days;
    std::vector<int16_t> ranks;
    std::vector<int8_t> families;
    std::vector<uint64_t> signatures;
    std::vector<int16_t> feasible_counts;
    double terminal_own = 0.0;
    double terminal_opponent = 0.0;
  };

  AdaptiveCandidate8SequenceOracleResult result;
  result.candidate_seat = candidate_seat;
  NativeAdaptivePlanner ordinary_planner(genome, backbones_);

  auto opponent_action = [&](const Simulator& env,
                             NativeAgentState& opponent_state) {
    PlayerAction action;
    if (opponent_route >= 0) {
      action = route_executor_.action_for(
          env, 1 - candidate_seat, opponent_route, opponent_state);
    } else {
      action.units.assign(
          env.farms()[1 - candidate_seat].hands.size() + 1, Action{});
    }
    return action;
  };

  auto advance_to_day = [&](BeamNode& node, int requested_day) {
    while (!node.env.done() && node.env.day() < requested_day) {
      std::array<PlayerAction, 2> joint;
      joint[candidate_seat] = ordinary_planner.action(
          node.env, candidate_seat, node.state);
      joint[1 - candidate_seat] = opponent_action(
          node.env, node.opponent_state);
      node.env.step(joint);
    }
  };

  auto score_node = [&](BeamNode& node) {
    Simulator env = node.env;
    AdaptivePlannerState state = node.state;
    NativeAgentState opponent_state = node.opponent_state;
    while (!env.done()) {
      std::array<PlayerAction, 2> joint;
      joint[candidate_seat] = ordinary_planner.action(
          env, candidate_seat, state);
      joint[1 - candidate_seat] = opponent_action(env, opponent_state);
      env.step(joint);
    }
    node.terminal_own = env.farms()[candidate_seat].money;
    node.terminal_opponent = env.farms()[1 - candidate_seat].money;
  };

  auto outcome = [](double own, double opponent) {
    return own > opponent ? 1.0 : (own == opponent ? 0.5 : 0.0);
  };
  auto better = [&](const BeamNode& left, const BeamNode& right) {
    constexpr double EPSILON = 1e-12;
    if (competitive_objective) {
      const double left_outcome = outcome(
          left.terminal_own, left.terminal_opponent);
      const double right_outcome = outcome(
          right.terminal_own, right.terminal_opponent);
      if (std::abs(left_outcome - right_outcome) > EPSILON)
        return left_outcome > right_outcome;
      const double left_margin =
          left.terminal_own - left.terminal_opponent;
      const double right_margin =
          right.terminal_own - right.terminal_opponent;
      if (std::abs(left_margin - right_margin) > EPSILON)
        return left_margin > right_margin;
    }
    if (std::abs(left.terminal_own - right.terminal_own) > EPSILON)
      return left.terminal_own > right.terminal_own;
    return left.ranks < right.ranks;
  };

  std::vector<BeamNode> beam;
  const size_t prefix_count = committed_rank_sequences.empty()
      ? size_t(1) : committed_rank_sequences.size();
  beam.reserve(prefix_count);
  for (size_t prefix_index = 0; prefix_index < prefix_count; ++prefix_index) {
    BeamNode initial{
        Simulator(Config{}, actual_seed), AdaptivePlannerState{},
        NativeAgentState{}, {}, {}, {}, {}, {}};
    initial.opponent_state.reset();
    if (prefix_route >= 0) {
      NativeAgentState prefix_state;
      prefix_state.reset();
      while (!initial.env.done() &&
             initial.env.step_count() < prefix_steps) {
        std::array<PlayerAction, 2> joint;
        joint[candidate_seat] = route_executor_.action_for(
            initial.env, candidate_seat, prefix_route, prefix_state);
        joint[1 - candidate_seat] = opponent_action(
            initial.env, initial.opponent_state);
        initial.env.step(joint);
      }
    }
    if (!committed_rank_sequences.empty()) {
      const auto& ranks = committed_rank_sequences[prefix_index];
      for (size_t stage = 0; stage < committed_days.size(); ++stage) {
        advance_to_day(initial, committed_days[stage]);
        if (initial.env.done())
          throw std::runtime_error(
              "environment ended before committed opening completed");
        initial.state.candidate8_decisions = 0;
        initial.state.last_plan_step = -1;
        AdaptiveGenome selected_genome = genome;
        selected_genome.offline_candidate8_enabled = true;
        selected_genome.offline_candidate8_rank = ranks[stage];
        selected_genome.offline_candidate8_minimum_day = committed_days[stage];
        selected_genome.offline_candidate8_use_feasible_pool =
            use_feasible_pool;
        NativeAdaptivePlanner selected_planner(selected_genome, backbones_);
        std::array<PlayerAction, 2> joint;
        joint[candidate_seat] = selected_planner.action(
            initial.env, candidate_seat, initial.state);
        joint[1 - candidate_seat] = opponent_action(
            initial.env, initial.opponent_state);
        initial.days.push_back(int16_t(initial.env.day()));
        initial.ranks.push_back(int16_t(ranks[stage]));
        initial.families.push_back(initial.state.first_candidate8_family);
        initial.signatures.push_back(initial.state.first_candidate8_signature);
        initial.feasible_counts.push_back(0);
        initial.env.step(joint);
      }
    }
    beam.push_back(std::move(initial));
  }
  result.maximum_live_beam = int(beam.size());

  for (int requested_day : decision_days) {
    std::vector<BeamNode> children;
    for (BeamNode& parent : beam) {
      advance_to_day(parent, requested_day);
      if (parent.env.done()) {
        BeamNode child = parent;
        child.days.push_back(int16_t(requested_day));
        child.ranks.push_back(int16_t(-1));
        child.families.push_back(int8_t(-1));
        child.signatures.push_back(0);
        child.feasible_counts.push_back(0);
        children.push_back(std::move(child));
        continue;
      }

      AdaptivePlannerState probe_state = parent.state;
      probe_state.candidate8_decisions = 0;
      probe_state.last_plan_step = -1;
      AdaptiveGenome probe_genome = genome;
      probe_genome.offline_candidate8_enabled = true;
      probe_genome.offline_candidate8_rank = 0;
      probe_genome.offline_candidate8_minimum_day = requested_day;
      probe_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
      NativeAdaptivePlanner probe_planner(probe_genome, backbones_);
      probe_planner.action(parent.env, candidate_seat, probe_state);
      const int available = use_feasible_pool
          ? int(probe_state.first_candidate8_feasible_count)
          : int(probe_state.first_candidate8_shortlist_count);
      const int arms = std::min(per_node_arms, std::max(0, available));

      if (arms <= 0) {
        BeamNode child = parent;
        child.days.push_back(int16_t(requested_day));
        child.ranks.push_back(int16_t(-1));
        child.families.push_back(int8_t(-1));
        child.signatures.push_back(0);
        child.feasible_counts.push_back(0);
        children.push_back(std::move(child));
        continue;
      }

      children.reserve(children.size() + size_t(arms));
      for (int arm = 0; arm < arms; ++arm) {
        BeamNode child = parent;
        child.state.candidate8_decisions = 0;
        child.state.last_plan_step = -1;
        AdaptiveGenome arm_genome = genome;
        arm_genome.offline_candidate8_enabled = true;
        arm_genome.offline_candidate8_rank = arm;
        arm_genome.offline_candidate8_minimum_day = requested_day;
        arm_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
        NativeAdaptivePlanner arm_planner(arm_genome, backbones_);
        std::array<PlayerAction, 2> joint;
        joint[candidate_seat] = arm_planner.action(
            child.env, candidate_seat, child.state);
        joint[1 - candidate_seat] = opponent_action(
            child.env, child.opponent_state);
        child.days.push_back(int16_t(child.env.day()));
        child.ranks.push_back(int16_t(arm));
        child.families.push_back(child.state.first_candidate8_family);
        child.signatures.push_back(child.state.first_candidate8_signature);
        child.feasible_counts.push_back(int16_t(arms));
        child.env.step(joint);
        children.push_back(std::move(child));
      }
    }

    result.expanded_nodes += int(children.size());
    result.complete_continuations += int(children.size());
    #pragma omp parallel for schedule(dynamic, 1)
    for (int index = 0; index < int(children.size()); ++index)
      score_node(children[index]);
    if (requested_day == decision_days.back()) {
      result.final_path_sequence_length = int(
          committed_days.size() + decision_days.size());
      result.final_path_candidate_rewards.reserve(children.size());
      result.final_path_opponent_rewards.reserve(children.size());
      result.final_path_selected_ranks.reserve(
          children.size() * size_t(result.final_path_sequence_length));
      result.final_path_selected_families.reserve(
          children.size() * size_t(result.final_path_sequence_length));
      for (const BeamNode& child : children) {
        result.final_path_candidate_rewards.push_back(child.terminal_own);
        result.final_path_opponent_rewards.push_back(child.terminal_opponent);
        result.final_path_selected_ranks.insert(
            result.final_path_selected_ranks.end(),
            child.ranks.begin(), child.ranks.end());
        result.final_path_selected_families.insert(
            result.final_path_selected_families.end(),
            child.families.begin(), child.families.end());
      }
    }
    std::stable_sort(children.begin(), children.end(), better);
    if (int(children.size()) > beam_width)
      children.resize(size_t(beam_width));
    result.maximum_live_beam = std::max(
        result.maximum_live_beam, int(children.size()));
    beam = std::move(children);
  }

  if (beam.empty())
    throw std::runtime_error("Candidate8 sequence beam unexpectedly empty");
  std::stable_sort(beam.begin(), beam.end(), better);
  const BeamNode& best = beam.front();
  result.rewards[candidate_seat] = best.terminal_own;
  result.rewards[1 - candidate_seat] = best.terminal_opponent;
  result.decision_day = best.days;
  result.selected_rank = best.ranks;
  result.selected_family = best.families;
  result.selected_signature = best.signatures;
  result.feasible_count = best.feasible_counts;
  return result;
}

AdaptiveCandidate8MctsResult
NativeAdaptiveExecutor::candidate8_mcts_oracle(
    const AdaptiveGenome& genome, int opponent_route, uint64_t actual_seed,
    const std::vector<int>& decision_days, int candidate_seat,
    int simulation_budget, int per_node_arms, bool use_feasible_pool,
    bool competitive_objective, double exploration_constant,
    double progressive_widening_constant,
    double progressive_widening_alpha, int rollout_arms,
    uint64_t search_seed, int prefix_route, int prefix_steps) const {
  if (candidate_seat != 0 && candidate_seat != 1)
    throw std::invalid_argument("candidate_seat must be 0 or 1");
  if (opponent_route < -1 || opponent_route >= route_executor_.route_count())
    throw std::invalid_argument("invalid opponent route");
  if (prefix_route < -1 || prefix_route >= route_executor_.route_count())
    throw std::invalid_argument("invalid prefix route");
  if (prefix_steps < 0 || prefix_steps > TOTAL_DAYS * 24)
    throw std::invalid_argument("prefix_steps must be in [0, 720]");
  if ((prefix_route < 0) != (prefix_steps == 0))
    throw std::invalid_argument(
        "prefix_route and positive prefix_steps must be supplied together");
  if (decision_days.empty())
    throw std::invalid_argument("at least one decision day is required");
  if (simulation_budget <= 0 || simulation_budget > 1000000)
    throw std::invalid_argument("simulation_budget must be in [1, 1000000]");
  if (per_node_arms <= 0 || per_node_arms > 4096)
    throw std::invalid_argument("per_node_arms must be in [1, 4096]");
  if (!std::isfinite(exploration_constant) || exploration_constant < 0.0 ||
      exploration_constant > 100.0)
    throw std::invalid_argument(
        "exploration_constant must be finite and in [0, 100]");
  if (!std::isfinite(progressive_widening_constant) ||
      progressive_widening_constant <= 0.0 ||
      progressive_widening_constant > 4096.0)
    throw std::invalid_argument(
        "progressive_widening_constant must be finite and in (0, 4096]");
  if (!std::isfinite(progressive_widening_alpha) ||
      progressive_widening_alpha <= 0.0 ||
      progressive_widening_alpha > 1.0)
    throw std::invalid_argument(
        "progressive_widening_alpha must be finite and in (0, 1]");
  if (rollout_arms <= 0 || rollout_arms > 4096)
    throw std::invalid_argument("rollout_arms must be in [1, 4096]");
  for (size_t index = 0; index < decision_days.size(); ++index) {
    if (decision_days[index] < 0 || decision_days[index] >= TOTAL_DAYS)
      throw std::invalid_argument("decision days must be in [0, 29]");
    if (index > 0 && decision_days[index] <= decision_days[index - 1])
      throw std::invalid_argument("decision days must be strictly increasing");
  }

  struct SearchState {
    Simulator env;
    AdaptivePlannerState state;
    NativeAgentState opponent_state;
    std::vector<int16_t> days;
    std::vector<int16_t> ranks;
    std::vector<int8_t> families;
    std::vector<uint64_t> signatures;
    std::vector<int16_t> feasible_counts;
  };
  struct MctsNode {
    SearchState position;
    int stage = 0;
    int visits = 0;
    double value_sum = 0.0;
    int arms = -1;
    std::vector<std::unique_ptr<MctsNode>> children;
    explicit MctsNode(SearchState value, int next_stage)
        : position(std::move(value)), stage(next_stage) {}
  };

  AdaptiveCandidate8MctsResult result;
  result.candidate_seat = candidate_seat;
  NativeAdaptivePlanner ordinary_planner(genome, backbones_);

  auto opponent_action = [&](const Simulator& env,
                             NativeAgentState& opponent_state) {
    PlayerAction action;
    if (opponent_route >= 0) {
      action = route_executor_.action_for(
          env, 1 - candidate_seat, opponent_route, opponent_state);
    } else {
      action.units.assign(
          env.farms()[1 - candidate_seat].hands.size() + 1, Action{});
    }
    return action;
  };

  auto advance_to_day = [&](SearchState& position, int requested_day) {
    while (!position.env.done() && position.env.day() < requested_day) {
      std::array<PlayerAction, 2> joint;
      joint[candidate_seat] = ordinary_planner.action(
          position.env, candidate_seat, position.state);
      joint[1 - candidate_seat] = opponent_action(
          position.env, position.opponent_state);
      position.env.step(joint);
    }
  };

  auto available_arms = [&](const SearchState& position, int requested_day) {
    if (position.env.done()) return 0;
    AdaptivePlannerState probe_state = position.state;
    probe_state.candidate8_decisions = 0;
    probe_state.last_plan_step = -1;
    AdaptiveGenome probe_genome = genome;
    probe_genome.offline_candidate8_enabled = true;
    probe_genome.offline_candidate8_rank = 0;
    probe_genome.offline_candidate8_minimum_day = requested_day;
    probe_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
    NativeAdaptivePlanner probe_planner(probe_genome, backbones_);
    probe_planner.action(position.env, candidate_seat, probe_state);
    const int available = use_feasible_pool
        ? int(probe_state.first_candidate8_feasible_count)
        : int(probe_state.first_candidate8_shortlist_count);
    return std::min(per_node_arms, std::max(0, available));
  };

  auto apply_arm = [&](SearchState& position, int requested_day,
                       int rank, int arms) {
    position.days.push_back(int16_t(requested_day));
    position.feasible_counts.push_back(int16_t(arms));
    if (rank < 0 || position.env.done()) {
      position.ranks.push_back(int16_t(-1));
      position.families.push_back(int8_t(-1));
      position.signatures.push_back(0);
      return;
    }
    position.state.candidate8_decisions = 0;
    position.state.last_plan_step = -1;
    AdaptiveGenome arm_genome = genome;
    arm_genome.offline_candidate8_enabled = true;
    arm_genome.offline_candidate8_rank = rank;
    arm_genome.offline_candidate8_minimum_day = requested_day;
    arm_genome.offline_candidate8_use_feasible_pool = use_feasible_pool;
    NativeAdaptivePlanner arm_planner(arm_genome, backbones_);
    std::array<PlayerAction, 2> joint;
    joint[candidate_seat] = arm_planner.action(
        position.env, candidate_seat, position.state);
    joint[1 - candidate_seat] = opponent_action(
        position.env, position.opponent_state);
    position.ranks.push_back(int16_t(rank));
    position.families.push_back(position.state.first_candidate8_family);
    position.signatures.push_back(position.state.first_candidate8_signature);
    position.env.step(joint);
  };

  auto finish = [&](SearchState position) {
    while (!position.env.done()) {
      std::array<PlayerAction, 2> joint;
      joint[candidate_seat] = ordinary_planner.action(
          position.env, candidate_seat, position.state);
      joint[1 - candidate_seat] = opponent_action(
          position.env, position.opponent_state);
      position.env.step(joint);
    }
    return std::array<double, 2>{
        position.env.farms()[candidate_seat].money,
        position.env.farms()[1 - candidate_seat].money};
  };

  auto outcome = [](double own, double opponent) {
    return own > opponent ? 1.0 : (own == opponent ? 0.5 : 0.0);
  };
  auto utility = [&](double own, double opponent) {
    if (!competitive_objective)
      return std::tanh(own / 100000.0);
    const double score = outcome(own, opponent);
    const double margin = std::tanh((own - opponent) / 50000.0);
    return 2.0 * score + 0.49 * margin;
  };
  auto terminal_better = [&](double own, double opponent,
                             double best_own, double best_opponent) {
    constexpr double EPSILON = 1e-12;
    if (competitive_objective) {
      const double left_outcome = outcome(own, opponent);
      const double right_outcome = outcome(best_own, best_opponent);
      if (std::abs(left_outcome - right_outcome) > EPSILON)
        return left_outcome > right_outcome;
      const double margin = own - opponent;
      const double best_margin = best_own - best_opponent;
      if (std::abs(margin - best_margin) > EPSILON)
        return margin > best_margin;
    }
    if (std::abs(own - best_own) > EPSILON) return own > best_own;
    return false;
  };

  SearchState initial{
      Simulator(Config{}, actual_seed), AdaptivePlannerState{},
      NativeAgentState{}, {}, {}, {}, {}, {}};
  initial.opponent_state.reset();
  if (prefix_route >= 0) {
    NativeAgentState prefix_state;
    prefix_state.reset();
    while (!initial.env.done() && initial.env.step_count() < prefix_steps) {
      std::array<PlayerAction, 2> joint;
      joint[candidate_seat] = route_executor_.action_for(
          initial.env, candidate_seat, prefix_route, prefix_state);
      joint[1 - candidate_seat] = opponent_action(
          initial.env, initial.opponent_state);
      initial.env.step(joint);
    }
  }

  MctsNode root(std::move(initial), 0);
  int tree_nodes = 1;
  std::mt19937_64 rng(search_seed);
  std::unordered_set<uint64_t> sampled_paths;
  bool have_best = false;
  double best_own = 0.0;
  double best_opponent = 0.0;
  std::vector<int16_t> best_days;
  std::vector<int16_t> best_ranks;
  std::vector<int8_t> best_families;
  std::vector<uint64_t> best_signatures;
  std::vector<int16_t> best_counts;

  auto path_hash = [](const SearchState& position) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t index = 0; index < position.ranks.size(); ++index) {
      hash ^= uint64_t(uint16_t(position.ranks[index]) + 1U);
      hash *= 1099511628211ULL;
      hash ^= position.signatures[index];
      hash *= 1099511628211ULL;
    }
    return hash;
  };

  for (int simulation = 1; simulation <= simulation_budget; ++simulation) {
    std::vector<MctsNode*> visited;
    MctsNode* node = &root;
    visited.push_back(node);

    while (node->stage < int(decision_days.size())) {
      const int requested_day = decision_days[size_t(node->stage)];
      advance_to_day(node->position, requested_day);
      if (node->arms < 0)
        node->arms = available_arms(node->position, requested_day);
      const int branch_count = std::max(1, node->arms);
      const double visits = double(std::max(1, node->visits));
      const int allowed = std::min(
          branch_count, std::max(1, int(std::ceil(
              progressive_widening_constant *
              std::pow(visits, progressive_widening_alpha)))));

      if (int(node->children.size()) < allowed) {
        const int rank = node->arms > 0
            ? int(node->children.size()) : -1;
        SearchState child_position = node->position;
        apply_arm(child_position, requested_day, rank, node->arms);
        node->children.push_back(std::make_unique<MctsNode>(
            std::move(child_position), node->stage + 1));
        ++tree_nodes;
        node = node->children.back().get();
        visited.push_back(node);
        result.maximum_depth = std::max(result.maximum_depth, node->stage);
        break;
      }

      MctsNode* selected = nullptr;
      double selected_score = -std::numeric_limits<double>::infinity();
      for (const auto& child_pointer : node->children) {
        MctsNode* child = child_pointer.get();
        const double mean = child->visits > 0
            ? child->value_sum / double(child->visits) : 0.0;
        const double bonus = exploration_constant * std::sqrt(
            std::log(double(node->visits) + 1.0) /
            double(std::max(1, child->visits)));
        const double score = mean + bonus;
        if (selected == nullptr || score > selected_score + 1e-15) {
          selected = child;
          selected_score = score;
        }
      }
      node = selected;
      visited.push_back(node);
      result.maximum_depth = std::max(result.maximum_depth, node->stage);
    }

    SearchState rollout = node->position;
    for (int stage = node->stage;
         stage < int(decision_days.size()); ++stage) {
      const int requested_day = decision_days[size_t(stage)];
      advance_to_day(rollout, requested_day);
      const int arms = available_arms(rollout, requested_day);
      int rank = -1;
      if (arms > 0) {
        const int preferred = std::min(arms, rollout_arms);
        // Candidate8 shortlist order already interleaves the audited families.
        // Use it as a weak rollout prior, while reserving 20% of rollouts for
        // any legal rank so high-index multi-project edits remain reachable.
        const bool explore_all = arms > preferred && (rng() % 5ULL == 0ULL);
        const int bound = explore_all ? arms : preferred;
        rank = int(rng() % uint64_t(bound));
      }
      apply_arm(rollout, requested_day, rank, arms);
    }

    const auto terminal = finish(rollout);
    const double own = terminal[0];
    const double opponent = terminal[1];
    result.sampled_candidate_rewards.push_back(own);
    result.sampled_opponent_rewards.push_back(opponent);
    sampled_paths.insert(path_hash(rollout));
    if (own > opponent) {
      ++result.winning_simulations;
      if (result.first_win_simulation < 0)
        result.first_win_simulation = simulation;
    }
    if (!have_best || terminal_better(
            own, opponent, best_own, best_opponent)) {
      have_best = true;
      best_own = own;
      best_opponent = opponent;
      best_days = rollout.days;
      best_ranks = rollout.ranks;
      best_families = rollout.families;
      best_signatures = rollout.signatures;
      best_counts = rollout.feasible_counts;
      result.best_found_simulation = simulation;
    }

    const double value = utility(own, opponent);
    for (MctsNode* visited_node : visited) {
      ++visited_node->visits;
      visited_node->value_sum += value;
    }
  }

  result.simulations = simulation_budget;
  result.tree_nodes = tree_nodes;
  result.unique_sampled_paths = int(sampled_paths.size());
  result.rewards[candidate_seat] = best_own;
  result.rewards[1 - candidate_seat] = best_opponent;
  result.decision_day = std::move(best_days);
  result.selected_rank = std::move(best_ranks);
  result.selected_family = std::move(best_families);
  result.selected_signature = std::move(best_signatures);
  result.feasible_count = std::move(best_counts);
  result.root_rank.reserve(root.children.size());
  result.root_visits.reserve(root.children.size());
  result.root_mean_value.reserve(root.children.size());
  for (size_t rank = 0; rank < root.children.size(); ++rank) {
    const MctsNode& child = *root.children[rank];
    result.root_rank.push_back(int16_t(root.arms > 0 ? int(rank) : -1));
    result.root_visits.push_back(child.visits);
    result.root_mean_value.push_back(child.visits > 0
        ? child.value_sum / double(child.visits) : 0.0);
  }
  return result;
}

AdaptivePortfolioCounterfactualResult
NativeAdaptiveExecutor::portfolio_counterfactual(
    const AdaptiveGenome& genome, int opponent_route, uint64_t prefix_seed,
    const std::vector<uint64_t>& future_seeds, int candidate_seat,
    int candidate_ranks, double switch_margin,
    int minimum_decision_day) const {
  if (candidate_seat != 0 && candidate_seat != 1)
    throw std::invalid_argument("candidate_seat must be 0 or 1");
  if (opponent_route < -1 || opponent_route >= route_executor_.route_count())
    throw std::invalid_argument("invalid opponent route");
  if (future_seeds.empty())
    throw std::invalid_argument("at least one future seed is required");
  if (candidate_ranks <= 0 || candidate_ranks > 16)
    throw std::invalid_argument("candidate_ranks must be in [1, 16]");
  if (minimum_decision_day < 0 || minimum_decision_day >= TOTAL_DAYS)
    throw std::invalid_argument("minimum_decision_day must be in [0, 29]");

  AdaptivePortfolioCounterfactualResult result;
  result.arms = candidate_ranks + 1;
  result.future_count = int(future_seeds.size());
  result.own_rewards.assign(
      size_t(result.arms) * result.future_count,
      std::numeric_limits<double>::quiet_NaN());
  result.end_overflow.assign(
      size_t(result.arms) * result.future_count, 0);
  result.arm_available.assign(result.arms, 0);
  result.arm_features.resize(result.arms);

  // Generate the public prefix with the ordinary KEEP policy.  A separate
  // copied state probes whether a bundle is available only after the requested
  // minimum day.  This prevents an inconsequential opening edit from forcing
  // every W3 calibration sample to day zero, and guarantees that no candidate
  // action contaminates the checkpoint.
  AdaptiveGenome probe_genome = genome;
  probe_genome.portfolio_bundle_switch_margin = switch_margin;
  probe_genome.portfolio_switch_cooldown_days = 30;
  probe_genome.portfolio_switch_candidate_rank = 0;
  probe_genome.offline_expose_all_feasible_switches = true;
  AdaptiveGenome keep_genome = genome;
  keep_genome.portfolio_bundle_switch_margin = 0.0;
  Simulator prefix(Config{}, prefix_seed);
  NativeAdaptivePlanner probe_planner(probe_genome, backbones_);
  NativeAdaptivePlanner keep_planner(keep_genome, backbones_);
  AdaptivePlannerState probe_state;
  NativeAgentState prefix_opponent_state;
  prefix_opponent_state.reset();
  Simulator checkpoint = prefix;
  AdaptivePlannerState checkpoint_state;
  NativeAgentState checkpoint_opponent_state;
  while (!prefix.done()) {
    if (prefix.day() >= minimum_decision_day) {
      AdaptivePlannerState candidate_state = probe_state;
      probe_planner.action(prefix, candidate_seat, candidate_state);
      if (candidate_state.bundle_switches > probe_state.bundle_switches) {
        checkpoint = prefix;
        checkpoint_state = probe_state;
        checkpoint_opponent_state = prefix_opponent_state;
        result.decision_found = true;
        result.decision_step = prefix.step_count();
        break;
      }
    }
    std::array<PlayerAction, 2> joint;
    joint[candidate_seat] = keep_planner.action(
        prefix, candidate_seat, probe_state);
    if (opponent_route >= 0)
      joint[1 - candidate_seat] = route_executor_.action_for(
          prefix, 1 - candidate_seat, opponent_route,
          prefix_opponent_state);
    else
      joint[1 - candidate_seat].units.assign(
          prefix.farms()[1 - candidate_seat].hands.size() + 1, Action{});
    prefix.step(joint);
  }
  if (!result.decision_found) return result;

  // Probe availability and capture features once.  All future RNG forks share
  // this exact public/private decision state; only events after the checkpoint
  // are resampled.
  for (int arm = 0; arm < result.arms; ++arm) {
    AdaptiveGenome arm_genome = genome;
    arm_genome.portfolio_bundle_switch_margin =
        arm == 0 ? 0.0 : switch_margin;
    arm_genome.portfolio_switch_cooldown_days = 30;
    arm_genome.portfolio_switch_candidate_rank = std::max(0, arm - 1);
    arm_genome.offline_expose_all_feasible_switches = arm > 0;
    NativeAdaptivePlanner planner(arm_genome, backbones_);
    AdaptivePlannerState state = checkpoint_state;
    planner.action(checkpoint, candidate_seat, state);
    const bool available = arm == 0 || state.bundle_switches > 0;
    result.arm_available[arm] = available;
    if (arm > 0 && available)
      result.arm_features[arm] = state.first_bundle_switch_features;
  }

  const int total = result.future_count * result.arms;
  #pragma omp parallel for schedule(dynamic, 1)
  for (int index = 0; index < total; ++index) {
    const int future = index / result.arms;
    const int arm = index % result.arms;
    if (!result.arm_available[arm]) continue;
    Simulator env = checkpoint;
    env.reseed_future(future_seeds[future]);
    AdaptiveGenome arm_genome = genome;
    arm_genome.portfolio_bundle_switch_margin =
        arm == 0 ? 0.0 : switch_margin;
    arm_genome.portfolio_switch_cooldown_days = 30;
    arm_genome.portfolio_switch_candidate_rank = std::max(0, arm - 1);
    arm_genome.offline_expose_all_feasible_switches = arm > 0;
    NativeAdaptivePlanner planner(arm_genome, backbones_);
    AdaptivePlannerState state = checkpoint_state;
    NativeAgentState opponent_state = checkpoint_opponent_state;
    int overflow = 0;
    while (!env.done()) {
      std::array<PlayerAction, 2> joint;
      joint[candidate_seat] = planner.action(env, candidate_seat, state);
      if (opponent_route >= 0)
        joint[1 - candidate_seat] = route_executor_.action_for(
            env, 1 - candidate_seat, opponent_route, opponent_state);
      else
        joint[1 - candidate_seat].units.assign(
            env.farms()[1 - candidate_seat].hands.size() + 1, Action{});
      env.step(joint);
      overflow += env.last_end_of_day_overflow()[candidate_seat];
    }
    result.own_rewards[index] = env.farms()[candidate_seat].money;
    result.end_overflow[index] = overflow;
  }
  return result;
}

}  // namespace fastkag
