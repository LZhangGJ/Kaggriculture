#include "native_observation_adapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace native_observation_adapter {
namespace {

constexpr int kProducts = fastkag::N_PRODUCTS;

void validate_player(int player) {
  if (player < 0 || player >= 2)
    throw std::invalid_argument("player must be 0 or 1");
}

int unlocked_quadrants(std::uint8_t mask) {
  int count = 0;
  for (; mask != 0; mask = static_cast<std::uint8_t>(mask >> 1U))
    count += mask & 1U;
  return std::max(1, count);
}

bool shed_adjacent(fastkag::Position position, const fastkag::Config& config) {
  const int center = config.board_size / 2;
  return (position.x == center - 1 || position.x == center) &&
         (position.y == center - 1 || position.y == center);
}

int shed_total(const std::array<int32_t, fastkag::N_ITEMS>& shed) {
  return std::accumulate(shed.begin(), shed.end(), 0);
}

void add_carried(std::vector<std::array<int32_t, fastkag::N_ITEMS>>& carried,
                 std::vector<std::vector<int8_t>>& order, std::size_t actor,
                 int item, int quantity) {
  while (carried.size() <= actor) {
    carried.emplace_back();
    order.emplace_back();
  }
  if (quantity <= 0) return;
  if (carried[actor][static_cast<std::size_t>(item)] == 0)
    order[actor].push_back(static_cast<int8_t>(item));
  carried[actor][static_cast<std::size_t>(item)] += quantity;
}

int take_carried(std::vector<std::array<int32_t, fastkag::N_ITEMS>>& carried,
                 std::vector<std::vector<int8_t>>& order, std::size_t actor,
                 int item, int quantity) {
  while (carried.size() <= actor) {
    carried.emplace_back();
    order.emplace_back();
  }
  const int taken = std::min(
      std::max(0, quantity), carried[actor][static_cast<std::size_t>(item)]);
  carried[actor][static_cast<std::size_t>(item)] -= taken;
  if (carried[actor][static_cast<std::size_t>(item)] == 0) {
    auto& keys = order[actor];
    keys.erase(std::remove(keys.begin(), keys.end(), item), keys.end());
  }
  return taken;
}

public_belief_runtime::Inventory town_drain(
    const fastkag::Simulator& simulator, int transition_step) {
  public_belief_runtime::Inventory result{};
  const auto& config = simulator.config();
  if (transition_step % std::max(1, config.town_shop_sell_interval) == 0) {
    for (int shop : simulator.shops()) {
      auto add = [&](int product, int quantity = 1) {
        result[static_cast<std::size_t>(product)] += quantity;
      };
      switch (shop) {
        case 0: add(5); add(0); break;
        case 1: add(5); add(0); add(3); break;
        case 2: add(0); add(1); add(2); add(3); break;
        case 3: add(3); add(6); add(0); break;
        case 4: add(1, 2); break;
        case 5: add(6); add(2); add(0); break;
        case 6: add(3); add(6); break;
        case 7: add(7, 2); break;
        default: throw std::invalid_argument("invalid public town shop id");
      }
    }
  }
  if (transition_step % std::max(1, config.town_center_sell_interval) == 0)
    for (int product = 0; product < 8; ++product)
      ++result[static_cast<std::size_t>(product)];
  return result;
}

g001::dump::OwnShedTransition compile_shed_transition(
    const fastkag::Simulator& simulator, int player,
    const fastkag::PlayerAction& action,
    public_belief_runtime::TransitionEvidence& evidence) {
  g001::dump::OwnShedTransition result;
  const auto& own = simulator.privates()[static_cast<std::size_t>(player)];
  const auto& farm = simulator.farms()[static_cast<std::size_t>(player)];
  const auto& config = simulator.config();
  for (int product = 0; product < kProducts; ++product)
    result.previous_shed[static_cast<std::size_t>(product)] =
        own.shed[static_cast<std::size_t>(product)];

  auto shed = own.shed;
  auto carried = own.inventories;
  auto order = own.inventory_order;
  while (order.size() < carried.size()) order.emplace_back();
  const std::size_t actors = std::min(
      action.units.size(), farm.hands.size() + 1);
  while (carried.size() < actors) {
    carried.emplace_back();
    order.emplace_back();
  }
  for (std::size_t actor = 0; actor < actors; ++actor) {
    const auto position = actor == 0 ? farm.farmer : farm.hands[actor - 1];
    const auto& selected = action.units[actor];
    const int item = static_cast<int>(selected.item);
    const int quantity = std::max(0, selected.quantity);
    if (selected.op == fastkag::Op::DROP && shed_adjacent(position, config)) {
      const auto keys = order[actor];
      for (int key : keys) {
        if (key < 0 || key >= fastkag::N_ITEMS) continue;
        const int carried_amount =
            carried[actor][static_cast<std::size_t>(key)];
        const int room = std::max(0, config.shed_capacity - shed_total(shed));
        const int accepted = std::min(carried_amount, room);
        shed[static_cast<std::size_t>(key)] += accepted;
        // Official DROP clears the carried stack even when the shed cannot
        // accept all of it; overflow is discarded rather than retained.
        (void)take_carried(carried, order, actor, key, carried_amount);
        if (key < kProducts) {
          const auto product = static_cast<std::size_t>(key);
          result.known_nonmarket_inflow.lower[product] += accepted;
          result.known_nonmarket_inflow.point[product] += accepted;
          result.known_nonmarket_inflow.upper[product] += accepted;
        }
      }
    } else if (selected.op == fastkag::Op::PICKUP &&
               shed_adjacent(position, config) && item >= 0 &&
               item < fastkag::N_ITEMS && quantity > 0) {
      const int accepted = std::min(quantity, shed[static_cast<std::size_t>(item)]);
      shed[static_cast<std::size_t>(item)] -= accepted;
      add_carried(carried, order, actor, item, accepted);
      if (item < kProducts) {
        const auto product = static_cast<std::size_t>(item);
        result.known_nonmarket_outflow.lower[product] += accepted;
        result.known_nonmarket_outflow.point[product] += accepted;
        result.known_nonmarket_outflow.upper[product] += accepted;
      }
    } else if (selected.op == fastkag::Op::PLACE && item >= 0 &&
               item < fastkag::N_ITEMS && quantity > 0) {
      bool animal_placement = false;
      if (item >= 9 && item < 12 && position.x >= 0 && position.y >= 0 &&
          position.x < config.board_size && position.y < config.board_size) {
        constexpr std::array<fastkag::TileKind, 3> structure{
            fastkag::TileKind::COOP, fastkag::TileKind::PASTURE,
            fastkag::TileKind::PASTURE};
        const auto& tile = farm.tiles[static_cast<std::size_t>(
            position.y * config.board_size + position.x)];
        animal_placement = tile.kind == structure[static_cast<std::size_t>(item - 9)] &&
            tile.animal == fastkag::Item::NONE;
      }
      if (!animal_placement && shed_adjacent(position, config)) {
        const int room = std::max(0, config.shed_capacity - shed_total(shed));
        const int accepted = std::min(
            {quantity, carried[actor][static_cast<std::size_t>(item)], room});
        (void)take_carried(carried, order, actor, item, accepted);
        shed[static_cast<std::size_t>(item)] += accepted;
        if (item < kProducts) {
          const auto product = static_cast<std::size_t>(item);
          result.known_nonmarket_inflow.lower[product] += accepted;
          result.known_nonmarket_inflow.point[product] += accepted;
          result.known_nonmarket_inflow.upper[product] += accepted;
        }
      }
    }
  }

  const std::size_t market_slots = std::min(
      action.market.size(), static_cast<std::size_t>(config.max_market_orders));
  for (std::size_t slot = 0; slot < market_slots; ++slot) {
    const auto& market = action.market[slot];
    const int item = static_cast<int>(market.item);
    const int quantity = std::max(0, market.quantity);
    if (market.op == fastkag::Op::SELL) {
      if (item < 0 || item >= kProducts)
        throw std::invalid_argument("submitted SELL has invalid product");
      result.sell_requested[static_cast<std::size_t>(item)] += quantity;
      evidence.own_sell_requested[static_cast<std::size_t>(item)] += quantity;
    } else if (market.op == fastkag::Op::BUY_PRODUCT) {
      if (item != static_cast<int>(fastkag::Item::WHEAT) &&
          item != static_cast<int>(fastkag::Item::FERTILIZER)) {
        throw std::invalid_argument("BUY_PRODUCT is legal only for WHEAT/FERTILIZER");
      }
      const auto product = static_cast<std::size_t>(item);
      evidence.own_buy_product.upper[product] += quantity;
      evidence.own_buy_product.point[product] =
          evidence.own_buy_product.upper[product] / 2;
      result.known_nonmarket_inflow.upper[product] += quantity;
    }
  }

  if ((simulator.step_count() + 1) % config.turns_per_day == 0) {
    // Market fills and insertion-order overflow decide which carried units
    // return.  No positive per-product lower bound is safe without observing
    // the next shed, while capacity is a sound marginal upper bound.
    for (int product = 0; product < kProducts; ++product)
      result.known_nonmarket_inflow.upper[static_cast<std::size_t>(product)] +=
          config.shed_capacity;
  }
  return result;
}

public_belief_runtime::Tile map_tile(const fastkag::Tile& tile) {
  public_belief_runtime::Tile result;
  if (tile.kind == fastkag::TileKind::PLANT) {
    result.kind = public_belief_runtime::TileKind::Plant;
    result.product = static_cast<int>(tile.crop);
  } else if (tile.kind == fastkag::TileKind::ANIMAL) {
    result.kind = public_belief_runtime::TileKind::Animal;
    const int animal = static_cast<int>(tile.animal) - 9;
    constexpr std::array<int, 3> product{5, 6, 7};
    if (animal < 0 || animal >= 3)
      throw std::invalid_argument("public animal tile has invalid animal");
    result.product = product[static_cast<std::size_t>(animal)];
  }
  result.yield_units = std::max(0, static_cast<int>(tile.yield_units));
  result.fertilizer_available = tile.fertilizer_available;
  result.fed_today = tile.fed_today;
  result.fertilized_until_day = tile.fertilized_until_day;
  result.planted_day = tile.planted_day;
  result.placed_day = tile.placed_day;
  result.pending_care_bonus = tile.pending_care_bonus;
  result.consecutive_unwatered = tile.consecutive_unwatered;
  result.consecutive_unfed = tile.consecutive_unfed;
  result.max_lifespan_step = tile.max_lifespan_step;
  result.watered_today = tile.watered_today;
  result.cared_today = tile.cared_today;
  return result;
}

public_belief_runtime::FarmObservation map_farm(const fastkag::Farm& farm) {
  public_belief_runtime::FarmObservation result;
  result.public_money = static_cast<std::int64_t>(std::llround(farm.money));
  result.farmer = {farm.farmer.x, farm.farmer.y};
  result.hands.reserve(farm.hands.size());
  for (const auto hand : farm.hands) result.hands.push_back({hand.x, hand.y});
  result.hires_today = farm.hires_today;
  result.unlocked_quadrants = unlocked_quadrants(farm.unlocked_mask);
  result.tiles.reserve(farm.tiles.size());
  for (const auto& tile : farm.tiles) result.tiles.push_back(map_tile(tile));
  return result;
}

}  // namespace

public_belief_runtime::Observation map_observation(
    const fastkag::Simulator& simulator, int player) {
  validate_player(player);
  public_belief_runtime::Observation result;
  result.step = simulator.step_count();
  result.day = simulator.day();
  result.turns_per_day = simulator.config().turns_per_day;
  result.board_size = simulator.config().board_size;
  result.episode_steps = simulator.config().episode_steps;
  result.farm_hand_cost_multiplier = simulator.config().farm_hand_cost_mult;
  result.town_shop_unlock_interval =
      simulator.config().town_shop_unlock_interval;
  result.town_shop_interval = simulator.config().town_shop_sell_interval;
  result.town_center_interval = simulator.config().town_center_sell_interval;
  result.town_unlocked_shop_count = static_cast<int>(simulator.shops().size());
  for (int shop : simulator.shops()) {
    auto add = [&](int product, int quantity = 1) {
      result.town_shop_demand_per_tick[static_cast<std::size_t>(product)] +=
          quantity;
    };
    switch (shop) {
      case 0: add(5); add(0); break;
      case 1: add(5); add(0); add(3); break;
      case 2: add(0); add(1); add(2); add(3); break;
      case 3: add(3); add(6); add(0); break;
      case 4: add(1, 2); break;
      case 5: add(6); add(2); add(0); break;
      case 6: add(3); add(6); break;
      case 7: add(7, 2); break;
      default: throw std::invalid_argument("invalid public town shop id");
    }
  }
  for (int product = 0; product < kProducts; ++product) {
    const auto p = static_cast<std::size_t>(product);
    result.market_inventory[p] = simulator.market().inventory[p];
    result.market_price[p] = simulator.market().prices[p];
  }
  // The only private read: focal player's own shed and carried inventories.
  const auto& own = simulator.privates()[static_cast<std::size_t>(player)];
  for (int product = 0; product < kProducts; ++product) {
    const auto p = static_cast<std::size_t>(product);
    result.own_total[p] = own.shed[p];
    for (const auto& carried : own.inventories)
      result.own_total[p] += carried[p];
  }
  result.own_farm = map_farm(simulator.farms()[static_cast<std::size_t>(player)]);
  result.opponent_farm =
      map_farm(simulator.farms()[static_cast<std::size_t>(1 - player)]);
  return result;
}

Adapter::Adapter(Config config) : config_(config) {
  if (config.history_capacity == 0)
    throw std::invalid_argument("history capacity must be positive");
}

public_belief_runtime::Snapshot Adapter::reset(
    const fastkag::Simulator& simulator, int player) {
  validate_player(player);
  if (pending_) throw std::logic_error("cannot reset with a pending submission");
  if (simulator.step_count() != 0)
    throw std::invalid_argument("native observation adapter reset requires episode step 0");
  player_ = player;
  runtime_.emplace(public_belief_runtime::Config{
      simulator.config().shed_capacity, config_.history_capacity});
  initialized_ = true;
  last_observed_step_ = 0;
  return runtime_->reset(map_observation(simulator, player));
}

void Adapter::stage_submission(
    const fastkag::Simulator& simulator,
    const fastkag::PlayerAction& own_submission) {
  if (!initialized_ || !runtime_)
    throw std::logic_error("reset must precede stage_submission");
  if (pending_) throw std::logic_error("a submission is already pending");
  if (simulator.step_count() != last_observed_step_)
    throw std::invalid_argument("stage state does not match last observed step");
  Pending pending;
  pending.step = simulator.step_count();
  pending.evidence.public_town_drain = town_drain(simulator, pending.step);
  pending.shed_transition = compile_shed_transition(
      simulator, player_, own_submission, pending.evidence);
  pending_ = std::move(pending);
}

UpdateResult Adapter::observe(const fastkag::Simulator& simulator) {
  if (!initialized_ || !runtime_)
    throw std::logic_error("reset must precede observe");
  if (!pending_) throw std::logic_error("stage_submission must precede observe");
  if (simulator.step_count() != pending_->step + 1)
    throw std::invalid_argument("observe requires exactly the next official step");

  const auto& own = simulator.privates()[static_cast<std::size_t>(player_)];
  for (int product = 0; product < kProducts; ++product)
    pending_->shed_transition.current_shed[static_cast<std::size_t>(product)] =
        own.shed[static_cast<std::size_t>(product)];
  const auto fill = g001::dump::derive_own_sell_fill_interval(
      pending_->shed_transition);
  pending_->evidence.confirmed_own_sell_fill.lower = fill.lower;
  pending_->evidence.confirmed_own_sell_fill.point = fill.point;
  pending_->evidence.confirmed_own_sell_fill.upper = fill.upper;

  UpdateResult result;
  result.evidence = pending_->evidence;
  result.derived_fill = fill;
  result.snapshot = runtime_->update(
      map_observation(simulator, player_), result.evidence);
  last_observed_step_ = simulator.step_count();
  pending_.reset();
  return result;
}

}  // namespace native_observation_adapter
