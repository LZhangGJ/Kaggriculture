#include "public_belief_runtime.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace public_belief_runtime {
namespace {

using g001::market::Product;
constexpr int kCropProducts = 5;

struct CropDefinition {
  int first_day;
  int max_day;
  int interval;
  int max_yield;
  bool ongoing;
};

struct AnimalDefinition {
  int first_day;
  int interval;
  int max_held;
  int product;
};

// Official kaggriculture 1.32.7 lifecycle constants.  This module deliberately
// keeps a small public-semantics table instead of depending on Simulator or an
// opponent route/private state.
constexpr std::array<CropDefinition, 5> kCrops{{
    {2, 4, 0, 6, false}, {2, 3, 0, 4, false},
    {8, 8, 1, 4, true}, {10, 10, 2, 4, true},
    {10, 12, 0, 6, false},
}};
constexpr std::array<AnimalDefinition, 3> kAnimals{{
    {4, 1, 4, 5}, {8, 2, 6, 6}, {6, 3, 6, 7},
}};

int saturated_add(int left, int right) {
  if (right <= 0) return left;
  return left > std::numeric_limits<int>::max() - right
      ? std::numeric_limits<int>::max()
      : left + right;
}

bool end_of_day_after_step(int step, int turns_per_day) {
  return (step + 1) % turns_per_day == 0;
}

int effective_supply_horizon(const Observation& observation, int requested) {
  const int capped = std::clamp(requested, 0, 24);
  const int remaining = std::max(0, observation.episode_steps - 1 - observation.step);
  return std::min(capped, remaining);
}

Inventory public_town_drain_upper(const Observation& observation, int horizon) {
  Inventory result{};
  auto per_tick = observation.town_shop_demand_per_tick;
  int unlocked = observation.town_unlocked_shop_count;
  // Per-product maxima over the eight public shop types.  A future random
  // unlock is not known at the current observation, so each product gets its
  // own conservative upper (the vector need not describe one joint shop).
  constexpr Inventory kMaximumNewShopDemand{{1, 2, 1, 1, 0, 1, 1, 2, 0}};
  for (int offset = 0; offset < horizon; ++offset) {
    const int step = observation.step + offset;
    if (step % observation.town_shop_interval == 0) {
      for (std::size_t p = 0; p < g001::market::product_count; ++p)
        result[p] = saturated_add(result[p], per_tick[p]);
    }
    if (step % observation.town_center_interval == 0) {
      for (std::size_t p = 0; p < 8; ++p)
        result[p] = saturated_add(result[p], 1);
    }
    if (end_of_day_after_step(step, observation.turns_per_day)) {
      const int next_day = step / observation.turns_per_day + 1;
      if (next_day > 0 &&
          next_day % observation.town_shop_unlock_interval == 0 &&
          unlocked < 8) {
        ++unlocked;
        for (std::size_t p = 0; p < g001::market::product_count; ++p)
          per_tick[p] = saturated_add(per_tick[p], kMaximumNewShopDemand[p]);
      }
    }
  }
  return result;
}

Inventory maximum_new_public_production(const Observation& observation,
                                        int horizon) {
  Inventory result{};
  const int current_day = observation.day;
  const int future_day = (observation.step + horizon) / observation.turns_per_day;

  for (const auto& tile : observation.opponent_farm.tiles) {
    if (tile.kind == TileKind::Plant) {
      const auto product = static_cast<std::size_t>(tile.product);
      const auto& crop = kCrops[product];
      if (crop.ongoing) {
        // Standing yield can be harvested before every rollover.  Optimal
        // public-only upper assumes WATER+FERTILIZE before each scheduled
        // production, so every event can add two fresh units without the
        // max-held cap suppressing a later event.
        for (int offset = 0; offset < horizon; ++offset) {
          const int step = observation.step + offset;
          if (!end_of_day_after_step(step, observation.turns_per_day)) continue;
          const int next_day = step / observation.turns_per_day + 1;
          const int delta = next_day - tile.planted_day - crop.first_day;
          if (delta < 0 || delta % crop.interval != 0) continue;
          const int production_count = delta / crop.interval + 1;
          if (production_count <= crop.max_yield)
            result[product] = saturated_add(result[product], 2);
        }
      } else if (future_day - tile.planted_day >= crop.first_day) {
        // A finite crop is harvested once.  Count every still-available daily
        // WATER opportunity, assuming fertilizer is obtainable.  Ignoring
        // travel/actor contention is intentional for a sound opponent upper.
        int possible_yield = std::max(0, tile.yield_units);
        const int water_start = (crop.max_day + 1) / 2;
        for (int day = current_day; day <= future_day; ++day) {
          const int age = day - tile.planted_day;
          if (age < water_start || age > crop.max_day) continue;
          if (day == current_day && tile.watered_today) continue;
          possible_yield = std::min(crop.max_yield, possible_yield + 2);
        }
        result[product] = saturated_add(
            result[product], std::max(0, possible_yield - tile.yield_units));
      }
      continue;
    }

    if (tile.kind != TileKind::Animal) continue;
    const int animal_index = tile.product - static_cast<int>(Product::Egg);
    if (animal_index < 0 || animal_index >= static_cast<int>(kAnimals.size()))
      continue;
    const auto& animal = kAnimals[static_cast<std::size_t>(animal_index)];
    int pending_care = std::max(0, tile.pending_care_bonus);
    for (int offset = 0; offset < horizon; ++offset) {
      const int step = observation.step + offset;
      if (!end_of_day_after_step(step, observation.turns_per_day)) continue;
      const int next_day = step / observation.turns_per_day + 1;
      const int delta = next_day - tile.placed_day - animal.first_day;
      if (delta >= 0 && delta % animal.interval == 0) {
        const int produced = std::min(animal.max_held, 1 + pending_care);
        result[static_cast<std::size_t>(animal.product)] = saturated_add(
            result[static_cast<std::size_t>(animal.product)], produced);
        pending_care = 0;
      }
      // Best-case FEED+CARE is public-feasible evidence for an upper.  The
      // official EOD applies production first and installs today's care bonus
      // afterwards, including on a production day.
      pending_care = saturated_add(pending_care, 1);
      result[static_cast<std::size_t>(Product::Fertilizer)] = saturated_add(
          result[static_cast<std::size_t>(Product::Fertilizer)], 1);
    }
  }
  return result;
}

void validate_inventory(const Inventory& values, const char* name) {
  for (int value : values) {
    if (value < 0) throw std::invalid_argument(name);
  }
}

void validate_interval(const InventoryInterval& interval, const char* name) {
  for (std::size_t i = 0; i < g001::market::product_count; ++i) {
    if (interval.lower[i] < 0 || interval.lower[i] > interval.point[i] ||
        interval.point[i] > interval.upper[i]) {
      throw std::invalid_argument(name);
    }
  }
}

void validate_observation(const Observation& observation) {
  if (observation.step < 0 || observation.day < 0 ||
      observation.turns_per_day <= 0 || observation.board_size <= 0 ||
      observation.episode_steps <= 1 || observation.step >= observation.episode_steps ||
      observation.farm_hand_cost_multiplier <= 0 ||
      observation.town_shop_unlock_interval <= 0 ||
      observation.town_shop_interval <= 0 ||
      observation.town_center_interval <= 0 ||
      observation.town_unlocked_shop_count < 0 ||
      observation.town_unlocked_shop_count > 8 ||
      observation.own_farm.unlocked_quadrants <= 0 ||
      observation.opponent_farm.unlocked_quadrants <= 0) {
    throw std::invalid_argument("invalid public observation scalar");
  }
  validate_inventory(observation.market_inventory, "negative public market inventory");
  validate_inventory(observation.market_price, "negative public market price");
  validate_inventory(observation.own_total, "negative own total inventory");
  validate_inventory(observation.town_shop_demand_per_tick,
                     "negative public town shop demand");
  auto validate_tiles = [](const std::vector<Tile>& tiles) {
    for (const auto& tile : tiles) {
      if (tile.yield_units < 0 || tile.fertilized_until_day < -1 ||
          tile.planted_day < 0 || tile.placed_day < 0 ||
          tile.pending_care_bonus < 0 || tile.consecutive_unwatered < 0 ||
          tile.consecutive_unfed < 0 || tile.max_lifespan_step < -1) {
        throw std::invalid_argument("invalid public tile scalar");
      }
      if (tile.kind == TileKind::Plant &&
          (tile.product < 0 || tile.product >= kCropProducts)) {
        throw std::invalid_argument("invalid public plant product");
      }
      const int egg = static_cast<int>(Product::Egg);
      const int wool = static_cast<int>(Product::Wool);
      if (tile.kind == TileKind::Animal &&
          (tile.product < egg || tile.product > wool)) {
        throw std::invalid_argument("invalid public animal product");
      }
    }
  };
  validate_tiles(observation.own_farm.tiles);
  validate_tiles(observation.opponent_farm.tiles);
}

bool at_shed(Position position, int board_size) {
  const int center = board_size / 2;
  return (position.x == center - 1 || position.x == center) &&
         (position.y == center - 1 || position.y == center);
}

bool public_shed_access(const FarmObservation& farm, int board_size) {
  if (at_shed(farm.farmer, board_size)) return true;
  return std::any_of(farm.hands.begin(), farm.hands.end(),
                     [&](Position position) { return at_shed(position, board_size); });
}

std::pair<int, int> visible_sell_bounds(int previous_price, int current_price,
                                       int fill_lower, int fill_upper) {
  if (previous_price <= 1) return {0, 0};
  if (current_price <= 1) return {0, fill_upper};
  return {fill_lower, fill_upper};
}

}  // namespace

Runtime::Runtime(Config config)
    : config_(config), filter_(config.shed_capacity) {
  if (config.shed_capacity < 0 || config.history_capacity == 0 ||
      config.opponent_supply_horizon_steps < 0 ||
      config.opponent_supply_horizon_steps > 24) {
    throw std::invalid_argument("invalid public belief runtime config");
  }
}

g001::market::PublicStateSummary Runtime::base_summary(
    const Observation& observation) const {
  g001::market::PublicStateSummary summary;
  summary.step = observation.step;
  summary.day = observation.day;
  summary.market_inventory = observation.market_inventory;
  summary.market_price = observation.market_price;
  summary.own_total = observation.own_total;
  summary.own_money = observation.own_farm.public_money;
  summary.opponent_money = observation.opponent_farm.public_money;
  summary.own_hands = static_cast<int>(observation.own_farm.hands.size());
  summary.opponent_hands =
      static_cast<int>(observation.opponent_farm.hands.size());
  summary.own_hires_today = observation.own_farm.hires_today;
  summary.opponent_hires_today = observation.opponent_farm.hires_today;
  summary.own_unlocked_quadrants = observation.own_farm.unlocked_quadrants;
  summary.opponent_unlocked_quadrants =
      observation.opponent_farm.unlocked_quadrants;
  summary.farm_hand_cost_multiplier = observation.farm_hand_cost_multiplier;
  // Match SharedBeliefTracker: public balances are retained but not used until
  // private purchase and all non-floor trade cash have causal bounds.
  summary.money_evidence_valid = false;
  return summary;
}

void Runtime::append_frame(g001::dump::PublicMarketFrame frame) {
  if (history_.size() == config_.history_capacity) history_.pop_front();
  history_.push_back(std::move(frame));
}

Snapshot Runtime::reset(const Observation& observation) {
  validate_observation(observation);
  history_.clear();
  const auto summary = base_summary(observation);
  belief_ = filter_.reset(summary);
  previous_ = observation;
  initialized_ = true;

  g001::dump::PublicMarketFrame frame;
  frame.step = observation.step;
  frame.day = observation.day;
  frame.market_inventory = observation.market_inventory;
  frame.market_price = observation.market_price;
  append_frame(std::move(frame));
  return snapshot();
}

Snapshot Runtime::update(const Observation& observation,
                         const TransitionEvidence& evidence) {
  if (!initialized_) {
    throw std::logic_error("reset must precede public belief runtime update");
  }
  validate_observation(observation);
  validate_inventory(evidence.own_sell_requested,
                     "negative submitted own SELL quantity");
  validate_inventory(evidence.public_town_drain,
                     "negative public town drain");
  validate_interval(evidence.confirmed_own_sell_fill,
                    "invalid confirmed own SELL interval");
  validate_interval(evidence.own_buy_product,
                    "invalid own BUY_PRODUCT interval");
  for (std::size_t p = 0; p < g001::market::product_count; ++p) {
    if (evidence.confirmed_own_sell_fill.upper[p] >
        evidence.own_sell_requested[p]) {
      throw std::invalid_argument("confirmed own SELL exceeds submitted quantity");
    }
    const bool purchasable = p == static_cast<std::size_t>(Product::Wheat) ||
        p == static_cast<std::size_t>(Product::Fertilizer);
    if (!purchasable && evidence.own_buy_product.upper[p] != 0) {
      throw std::invalid_argument("BUY_PRODUCT bound for non-tradable product");
    }
  }
  if (observation.step != previous_.step + 1) {
    throw std::invalid_argument("public observations must be consecutive");
  }
  if (observation.turns_per_day != previous_.turns_per_day ||
      observation.board_size != previous_.board_size ||
      observation.episode_steps != previous_.episode_steps ||
      observation.town_shop_unlock_interval !=
          previous_.town_shop_unlock_interval ||
      observation.town_shop_interval != previous_.town_shop_interval ||
      observation.town_center_interval != previous_.town_center_interval ||
      observation.own_farm.tiles.size() != previous_.own_farm.tiles.size() ||
      observation.opponent_farm.tiles.size() !=
          previous_.opponent_farm.tiles.size()) {
    throw std::invalid_argument("public observation geometry changed mid-episode");
  }

  auto summary = base_summary(observation);
  summary.town_consumption = evidence.public_town_drain;
  summary.own_market_net_flow_bounds_valid = true;
  for (std::size_t p = 0; p < g001::market::product_count; ++p) {
    const auto [visible_lower, visible_upper] = visible_sell_bounds(
        previous_.market_price[p], observation.market_price[p],
        evidence.confirmed_own_sell_fill.lower[p],
        evidence.confirmed_own_sell_fill.upper[p]);
    summary.own_market_net_flow_lower[p] =
        visible_lower - evidence.own_buy_product.upper[p];
    summary.own_market_net_flow_upper[p] =
        visible_upper - evidence.own_buy_product.lower[p];
    summary.own_market_net_flow_point[p] =
        (summary.own_market_net_flow_lower[p] +
         summary.own_market_net_flow_upper[p]) / 2;

    const int net_lower = observation.own_total[p] - previous_.own_total[p] +
        summary.own_market_net_flow_lower[p];
    const int net_point = observation.own_total[p] - previous_.own_total[p] +
        summary.own_market_net_flow_point[p];
    const int net_upper = observation.own_total[p] - previous_.own_total[p] +
        summary.own_market_net_flow_upper[p];
    summary.own_flow.harvest_point[p] = std::max(0, net_point);
    summary.own_flow.consumption_point[p] = std::max(0, -net_point);
    summary.own_flow.harvest_upper[p] =
        std::max(summary.own_flow.harvest_point[p], std::max(0, net_upper));
    summary.own_flow.consumption_upper[p] =
        std::max(summary.own_flow.consumption_point[p], std::max(0, -net_lower));
    summary.floor_sale_ambiguity[p] =
        previous_.market_price[p] == 1 || observation.market_price[p] == 1;
    summary.possible_private_discard[p] =
        (previous_.step + 1) % observation.turns_per_day == 0;
  }

  const auto& before_tiles = previous_.opponent_farm.tiles;
  const auto& after_tiles = observation.opponent_farm.tiles;
  for (std::size_t i = 0; i < before_tiles.size(); ++i) {
    const auto& before = before_tiles[i];
    const auto& after = after_tiles[i];
    if (before.kind == TileKind::Plant && before.yield_units > 0 &&
        (after.kind != TileKind::Plant ||
         after.yield_units < before.yield_units)) {
      summary.opponent_flow.harvest_upper[static_cast<std::size_t>(before.product)] +=
          before.yield_units -
          (after.kind == TileKind::Plant ? after.yield_units : 0);
    }
    if (before.kind == TileKind::Animal &&
        before.yield_units > after.yield_units) {
      summary.opponent_flow.harvest_upper[static_cast<std::size_t>(before.product)] +=
          before.yield_units - after.yield_units;
    }
    if (before.kind == TileKind::Animal && before.fertilizer_available &&
        !after.fertilizer_available) {
      ++summary.opponent_flow.harvest_upper[
          static_cast<std::size_t>(Product::Fertilizer)];
    }
    if (before.kind == TileKind::Animal && !before.fed_today &&
        after.fed_today) {
      ++summary.opponent_flow.consumption_upper[
          static_cast<std::size_t>(Product::Wheat)];
    }
    if (before.kind == TileKind::Plant && after.kind == TileKind::Plant &&
        after.fertilized_until_day > before.fertilized_until_day) {
      ++summary.opponent_flow.consumption_upper[
          static_cast<std::size_t>(Product::Fertilizer)];
    }
  }
  summary.opponent_shed_access_ambiguity =
      public_shed_access(previous_.opponent_farm, previous_.board_size);
  summary.day_rollover =
      (previous_.step + 1) % observation.turns_per_day == 0;

  belief_ = filter_.update(summary);

  g001::dump::PublicMarketFrame frame;
  frame.step = observation.step;
  frame.day = observation.day;
  frame.market_inventory = observation.market_inventory;
  frame.market_price = observation.market_price;
  frame.own_sell_requested = evidence.own_sell_requested;
  frame.own_sell_filled_lower = evidence.confirmed_own_sell_fill.lower;
  frame.own_sell_filled_point = evidence.confirmed_own_sell_fill.point;
  frame.own_sell_filled_upper = evidence.confirmed_own_sell_fill.upper;
  frame.known_town_drain = evidence.public_town_drain;
  frame.opponent_clearance_lower = belief_.recent_clearance_interval.lower;
  frame.opponent_clearance_point = belief_.recent_clearance;
  frame.opponent_clearance_upper = belief_.recent_clearance_interval.upper;
  frame.opponent_clearance_valid.fill(true);
  append_frame(std::move(frame));

  previous_ = observation;
  return snapshot();
}

Snapshot Runtime::snapshot() const {
  Snapshot result;
  result.belief = belief_;
  std::array<int, kCropProducts> crop_age_capacity{};
  for (const auto& tile : previous_.opponent_farm.tiles) {
    if (tile.kind == TileKind::Plant && tile.product >= 0 &&
        tile.product < kCropProducts) {
      const auto crop = static_cast<std::size_t>(tile.product);
      result.opponent_visible_standing_production[crop] = saturated_add(
          result.opponent_visible_standing_production[crop],
          std::max(0, tile.yield_units));
      const int age = std::max(0, previous_.day - tile.planted_day);
      crop_age_capacity[crop] = saturated_add(
          crop_age_capacity[crop],
          std::min(100, age * 100 / kCrops[crop].first_day));
    }
    if ((tile.kind == TileKind::Plant || tile.kind == TileKind::Animal) &&
        tile.product >= 0 && tile.product < static_cast<int>(g001::market::product_count)) {
      auto& upper = result.opponent_harvestable_now_upper[
          static_cast<std::size_t>(tile.product)];
      upper += std::max(0, tile.yield_units);
    }
    if (tile.kind == TileKind::Animal && tile.fertilizer_available) {
      ++result.opponent_harvestable_now_upper[
          static_cast<std::size_t>(Product::Fertilizer)];
    }
  }
  for (std::size_t crop = 0; crop < kCropProducts; ++crop)
    result.opponent_visible_standing_production[crop] = saturated_add(
        result.opponent_visible_standing_production[crop],
        crop_age_capacity[crop] / 100);
  result.opponent_supply_horizon_steps = effective_supply_horizon(
      previous_, config_.opponent_supply_horizon_steps);
  result.opponent_new_production_within_horizon_upper =
      maximum_new_public_production(
          previous_, result.opponent_supply_horizon_steps);
  for (std::size_t p = 0; p < g001::market::product_count; ++p) {
    int total = std::max(0, belief_.total_interval.upper[p]);
    total = saturated_add(total, result.opponent_harvestable_now_upper[p]);
    total = saturated_add(
        total, result.opponent_new_production_within_horizon_upper[p]);
    result.opponent_sellable_within_horizon_upper[p] = total;
  }

  const auto town = public_town_drain_upper(
      previous_, result.opponent_supply_horizon_steps);
  const auto wheat = static_cast<std::size_t>(Product::Wheat);
  const int wheat_inventory = previous_.market_inventory[wheat];
  const int wheat_cost_lower = g001::market::price(
      Product::Wheat, wheat_inventory - 1);
  const int wheat_cost_upper = g001::market::price(
      Product::Wheat, wheat_inventory - town[wheat] - 1);
  for (std::size_t animal = 0; animal < kAnimals.size(); ++animal) {
    const auto product = static_cast<std::size_t>(kAnimals[animal].product);
    auto& risk = result.feed_profitability_risk[animal];
    risk.product_price_lower = g001::market::price(
        static_cast<Product>(product),
        saturated_add(previous_.market_inventory[product],
                      result.opponent_sellable_within_horizon_upper[product]));
    risk.product_price_upper = g001::market::price(
        static_cast<Product>(product),
        previous_.market_inventory[product] - town[product]);
    risk.wheat_replacement_price_lower = wheat_cost_lower;
    risk.wheat_replacement_price_upper = wheat_cost_upper;
    risk.crosses_one_feed_one_output_threshold =
        risk.product_price_lower <= risk.wheat_replacement_price_upper &&
        risk.product_price_upper >= risk.wheat_replacement_price_lower;
  }
  result.public_market_history.assign(history_.begin(), history_.end());
  return result;
}

}  // namespace public_belief_runtime
