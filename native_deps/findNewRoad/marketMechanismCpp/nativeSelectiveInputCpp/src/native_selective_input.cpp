#include "native_selective_input.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace native_selective_input {
namespace {

namespace po = production_obligation;
using g001::market::Animal;
using g001::market::Operation;
using g001::market::Product;

constexpr int kCompiledHorizon = 24;
// Native deployment must have a hard, single-core queue-compilation bound.
// Exhausting it is safe because protectedQueue returns the exact legacy queue;
// it must never submit a partially searched approximation.
constexpr std::size_t kNativeMaximumQueueSearchStates = 64;
constexpr int kNativeMaximumResourceDerivationIterations = 64;

bool same_native_action(const fastkag::Action& left,
                        const fastkag::Action& right) {
  return left.op == right.op && left.item == right.item &&
      left.quantity == right.quantity;
}

bool valid_item(fastkag::Item item) {
  const int raw = static_cast<int>(item);
  return raw >= -1 && raw < fastkag::N_ITEMS;
}

bool valid_product(Product product) {
  return static_cast<std::size_t>(product) < g001::market::product_count;
}

bool valid_animal(Animal animal) {
  const int raw = static_cast<int>(animal);
  return raw >= 0 && raw < fastkag::N_ANIMALS;
}

po::Item to_item(fastkag::Item item) {
  return static_cast<po::Item>(static_cast<int>(item));
}

bool convert_unit(const fastkag::Action& source, po::UnitAction& target,
                  std::string& reason) {
  const int op = static_cast<int>(source.op);
  if (op < static_cast<int>(fastkag::Op::PASS) ||
      op > static_cast<int>(fastkag::Op::CARE)) {
    reason = "unit plan contains a market operation or invalid op";
    return false;
  }
  if (!valid_item(source.item) || source.quantity < 0) {
    reason = "unit plan contains invalid item/quantity";
    return false;
  }
  target.op = static_cast<po::UnitOp>(op);
  target.item = to_item(source.item);
  target.quantity = source.quantity;
  return true;
}

bool convert_market_vector(const std::vector<fastkag::Action>& source,
                           int maximum_slots,
                           std::vector<g001::market::Order>& target,
                           std::string& reason) {
  if (source.size() > static_cast<std::size_t>(maximum_slots)) {
    reason = "market queue exceeds official slot count";
    return false;
  }
  target.clear();
  target.reserve(source.size());
  for (const auto& action : source) {
    auto converted = to_typed_market_order(action);
    if (!converted.accepted) {
      reason = converted.reason;
      return false;
    }
    const auto roundtrip = to_native_market_order(converted.order);
    if (!roundtrip.accepted || !same_native_action(roundtrip.action, action)) {
      reason = "typed market conversion is not byte-roundtrip exact";
      return false;
    }
    target.push_back(converted.order);
  }
  return true;
}

bool convert_unit_vector(const std::vector<fastkag::Action>& source,
                         std::vector<po::UnitAction>& target,
                         std::string& reason) {
  target.clear();
  target.reserve(source.size());
  for (const auto& action : source) {
    po::UnitAction converted;
    if (!convert_unit(action, converted, reason)) return false;
    target.push_back(converted);
  }
  return true;
}

g001::market::Inventory deterministic_town_drain(
    const fastkag::Simulator& simulator, int step) {
  g001::market::Inventory result{};
  const auto& config = simulator.config();
  auto add = [&](int product, int quantity = 1) {
    result[static_cast<std::size_t>(product)] += quantity;
  };
  if (step % std::max(1, config.town_shop_sell_interval) == 0) {
    for (const int shop : simulator.shops()) {
      switch (shop) {
        case 0: add(5); add(0); break;
        case 1: add(5); add(0); add(3); break;
        case 2: add(0); add(1); add(2); add(3); break;
        case 3: add(3); add(6); add(0); break;
        case 4: add(1, 2); break;
        case 5: add(6); add(2); add(0); break;
        case 6: add(3); add(6); break;
        case 7: add(7, 2); break;
        default: throw std::invalid_argument("public shop id is invalid");
      }
    }
  }
  if (step % std::max(1, config.town_center_sell_interval) == 0)
    for (int product = 0; product < 8; ++product) add(product);
  return result;
}

bool zero(const g001::market::Inventory& inventory) {
  return std::all_of(inventory.begin(), inventory.end(),
                     [](int quantity) { return quantity == 0; });
}

bool validate_public_snapshot(
    const public_belief_runtime::Snapshot& snapshot,
    const fastkag::Simulator& simulator, std::string& reason) {
  const int step = simulator.step_count();
  const int day = simulator.day();
  if (snapshot.belief.step != step ||
      snapshot.public_market_history.empty()) {
    reason = "public belief step/history does not match current observation";
    return false;
  }
  const auto& history = snapshot.public_market_history;
  for (std::size_t index = 0; index < history.size(); ++index) {
    const auto& frame = history[index];
    if (frame.step < 0 || frame.day != frame.step / 24 ||
        (index > 0 && frame.step != history[index - 1].step + 1)) {
      reason = "public market history is not an official continuous prefix";
      return false;
    }
    for (std::size_t product = 0;
         product < g001::market::product_count; ++product) {
      if (frame.market_inventory[product] < 0 ||
          frame.market_price[product] < 0 ||
          frame.known_town_drain[product] < 0 ||
          frame.own_sell_filled_lower[product] < 0 ||
          frame.own_sell_filled_lower[product] >
              frame.own_sell_filled_point[product] ||
          frame.own_sell_filled_point[product] >
              frame.own_sell_filled_upper[product] ||
          frame.own_sell_filled_upper[product] >
              frame.own_sell_requested[product] ||
          (frame.opponent_clearance_valid[product] &&
           (frame.opponent_clearance_lower[product] < 0 ||
            frame.opponent_clearance_lower[product] >
                frame.opponent_clearance_point[product] ||
            frame.opponent_clearance_point[product] >
                frame.opponent_clearance_upper[product]))) {
        reason = "public history contains an invalid inventory/fill interval";
        return false;
      }
    }
  }
  const auto& current = history.back();
  if (current.step != step || current.day != day) {
    reason = "final public history frame is stale";
    return false;
  }
  for (int product = 0; product < fastkag::N_PRODUCTS; ++product) {
    const auto index = static_cast<std::size_t>(product);
    if (current.market_inventory[index] != simulator.market().inventory[index] ||
        current.market_price[index] != simulator.market().prices[index]) {
      reason = "current public market snapshot is not exact";
      return false;
    }
    const auto& belief = snapshot.belief;
    if (belief.total_interval.lower[index] < 0 ||
        belief.total_interval.lower[index] > belief.total[index] ||
        belief.total[index] > belief.total_interval.upper[index] ||
        belief.shed_interval.lower[index] < 0 ||
        belief.shed_interval.lower[index] > belief.shed[index] ||
        belief.shed[index] > belief.shed_interval.upper[index] ||
        belief.shed_interval.upper[index] > simulator.config().shed_capacity) {
      reason = "public belief inventory interval is invalid";
      return false;
    }
  }
  auto sum = [](const g001::market::Inventory& inventory) {
    std::int64_t total = 0;
    for (int quantity : inventory) total += quantity;
    return total;
  };
  if (sum(snapshot.belief.shed) > simulator.config().shed_capacity ||
      sum(snapshot.belief.shed_interval.lower) >
          simulator.config().shed_capacity) {
    reason = "public belief shed point/lower exceeds official capacity";
    return false;
  }
  return true;
}

po::TileState convert_tile(const fastkag::Tile& source) {
  const int kind = static_cast<int>(source.kind);
  if (kind < static_cast<int>(fastkag::TileKind::EMPTY) ||
      kind > static_cast<int>(fastkag::TileKind::ANIMAL)) {
    throw std::invalid_argument("public own tile kind is invalid");
  }
  po::TileState target;
  target.kind = static_cast<po::TileKind>(kind);
  if (source.kind == fastkag::TileKind::PLANT) {
    const int crop = static_cast<int>(source.crop);
    if (crop < 0 || crop >= fastkag::N_CROPS)
      throw std::invalid_argument("public own crop item is invalid");
    target.item = to_item(source.crop);
  } else if (source.kind == fastkag::TileKind::ANIMAL) {
    const int animal = static_cast<int>(source.animal);
    if (animal < 9 || animal >= fastkag::N_ITEMS)
      throw std::invalid_argument("public own animal item is invalid");
    target.item = to_item(source.animal);
  }
  target.planted_day = source.planted_day;
  target.yield_units = source.yield_units;
  target.consecutive_unfed = source.consecutive_unfed;
  target.fed_today = source.fed_today;
  target.fertilizer_available = source.fertilizer_available;
  return target;
}

void fill_current_state(selective_runtime::Input& target,
                        const fastkag::Simulator& simulator, int player) {
  const auto& config = simulator.config();
  const auto& farm = simulator.farms()[static_cast<std::size_t>(player)];
  const auto& own = simulator.privates()[static_cast<std::size_t>(player)];
  auto& current = target.production.current;
  current.step = simulator.step_count();
  current.turns_per_day = config.turns_per_day;
  current.board_size = config.board_size;
  current.shed_capacity = config.shed_capacity;
  current.max_market_orders = config.max_market_orders;
  current.farm_hand_cost_mult = config.farm_hand_cost_mult;
  current.money = farm.money;
  current.unlocked_mask = farm.unlocked_mask;
  current.hires_today = farm.hires_today;
  current.shed = own.shed;
  current.seeds = own.seeds;
  current.product_price = simulator.market().prices;
  current.actor_positions.push_back({farm.farmer.x, farm.farmer.y});
  for (const auto position : farm.hands)
    current.actor_positions.push_back({position.x, position.y});
  if (own.inventories.size() != current.actor_positions.size() ||
      own.inventory_order.size() != own.inventories.size()) {
    throw std::invalid_argument("focal carried inventory does not match actors");
  }
  current.carried.resize(own.inventories.size());
  for (std::size_t actor = 0; actor < own.inventories.size(); ++actor) {
    current.carried[actor].quantity = own.inventories[actor];
    std::array<bool, fastkag::N_ITEMS> seen{};
    for (const int8_t raw : own.inventory_order[actor]) {
      if (raw < 0 || raw >= fastkag::N_ITEMS || seen[raw] ||
          own.inventories[actor][static_cast<std::size_t>(raw)] <= 0) {
        throw std::invalid_argument("focal carried insertion order is invalid");
      }
      seen[raw] = true;
      current.carried[actor].insertion_order.push_back(
          static_cast<po::Item>(raw));
    }
    for (int item = 0; item < fastkag::N_ITEMS; ++item) {
      if (own.inventories[actor][static_cast<std::size_t>(item)] < 0 ||
          (own.inventories[actor][static_cast<std::size_t>(item)] > 0 &&
           !seen[static_cast<std::size_t>(item)])) {
        throw std::invalid_argument("focal carried inventory/order is incomplete");
      }
    }
  }
  if (farm.tiles.size() !=
      static_cast<std::size_t>(config.board_size * config.board_size)) {
    throw std::invalid_argument("public own farm geometry is invalid");
  }
  current.tiles.reserve(farm.tiles.size());
  for (const auto& tile : farm.tiles) current.tiles.push_back(convert_tile(tile));
}

}  // namespace

TypedOrderResult to_typed_market_order(
    const fastkag::Action& action) noexcept {
  TypedOrderResult result;
  try {
    if (action.quantity < 0) {
      result.reason = "market order has negative quantity";
      return result;
    }
    const int item = static_cast<int>(action.item);
    auto product = [&]() {
      if (item < 0 || item >= fastkag::N_PRODUCTS)
        throw std::invalid_argument("market order product is invalid");
      return static_cast<Product>(item);
    };
    result.order.quantity = action.quantity;
    switch (action.op) {
      case fastkag::Op::PASS:
        if (action.item != fastkag::Item::NONE)
          throw std::invalid_argument("PASS market slot has noncanonical item");
        result.order.operation = Operation::Pass;
        break;
      case fastkag::Op::SELL:
        result.order.operation = Operation::Sell;
        result.order.product = product();
        break;
      case fastkag::Op::BUY_PRODUCT:
        if (action.item != fastkag::Item::WHEAT &&
            action.item != fastkag::Item::FERTILIZER)
          throw std::invalid_argument("BUY_PRODUCT is legal only for WHEAT/FERTILIZER");
        result.order.operation = Operation::BuyProduct;
        result.order.product = product();
        break;
      case fastkag::Op::BUY_SEED:
        if (item < 0 || item >= fastkag::N_CROPS)
          throw std::invalid_argument("BUY_SEED crop is invalid");
        result.order.operation = Operation::BuySeed;
        result.order.product = static_cast<Product>(item);
        break;
      case fastkag::Op::BUY_ANIMAL:
        if (item < 9 || item >= fastkag::N_ITEMS)
          throw std::invalid_argument("BUY_ANIMAL item is invalid");
        result.order.operation = Operation::BuyAnimal;
        result.order.animal = static_cast<Animal>(item - 9);
        break;
      case fastkag::Op::HIRE:
        if (action.item != fastkag::Item::NONE)
          throw std::invalid_argument("HIRE has noncanonical item");
        result.order.operation = Operation::Hire;
        break;
      case fastkag::Op::BUY_LAND:
        if (action.item != fastkag::Item::NONE)
          throw std::invalid_argument("BUY_LAND has noncanonical item");
        result.order.operation = Operation::BuyLand;
        break;
      default:
        throw std::invalid_argument("market vector contains unit/invalid op");
    }
    result.accepted = true;
    result.reason = "accepted canonical native market order";
  } catch (const std::exception& error) {
    result.reason = error.what();
  }
  return result;
}

NativeOrderResult to_native_market_order(
    const g001::market::Order& order) noexcept {
  NativeOrderResult result;
  try {
    if (order.quantity < 0 || !valid_product(order.product) ||
        !valid_animal(order.animal)) {
      throw std::invalid_argument("typed market order fields are invalid");
    }
    result.action.quantity = order.quantity;
    switch (order.operation) {
      case Operation::Pass:
        result.action.op = fastkag::Op::PASS;
        result.action.item = fastkag::Item::NONE;
        break;
      case Operation::Sell:
        result.action.op = fastkag::Op::SELL;
        result.action.item = static_cast<fastkag::Item>(order.product);
        break;
      case Operation::BuyProduct:
        if (order.product != Product::Wheat &&
            order.product != Product::Fertilizer)
          throw std::invalid_argument("typed BUY_PRODUCT is illegal");
        result.action.op = fastkag::Op::BUY_PRODUCT;
        result.action.item = static_cast<fastkag::Item>(order.product);
        break;
      case Operation::BuySeed:
        if (static_cast<int>(order.product) >= fastkag::N_CROPS)
          throw std::invalid_argument("typed BUY_SEED crop is invalid");
        result.action.op = fastkag::Op::BUY_SEED;
        result.action.item = static_cast<fastkag::Item>(order.product);
        break;
      case Operation::BuyAnimal:
        result.action.op = fastkag::Op::BUY_ANIMAL;
        result.action.item = static_cast<fastkag::Item>(
            9 + static_cast<int>(order.animal));
        break;
      case Operation::Hire:
        result.action.op = fastkag::Op::HIRE;
        result.action.item = fastkag::Item::NONE;
        break;
      case Operation::BuyLand:
        result.action.op = fastkag::Op::BUY_LAND;
        result.action.item = fastkag::Item::NONE;
        break;
      default:
        throw std::invalid_argument("typed market operation enum is invalid");
    }
    result.accepted = true;
    result.reason = "accepted typed market order";
  } catch (const std::exception& error) {
    result.reason = error.what();
  }
  return result;
}

Result build(const BuildInput& source) noexcept {
    Result result;
  try {
    if (!source.simulator)
      throw std::invalid_argument("simulator pointer is missing");
    if (source.player < 0 || source.player > 1)
      throw std::invalid_argument("focal player must be 0 or 1");
    const auto& simulator = *source.simulator;
    const auto& config = simulator.config();
    const int step = simulator.step_count();
    if (simulator.done())
      throw std::invalid_argument("episode is already terminal");
    if (config.episode_steps != 720 || config.turns_per_day != 24 ||
        config.board_size <= 1 || config.board_size % 2 != 0 ||
        config.max_market_orders <= 0 || config.max_market_orders > 10 ||
        config.shed_capacity < 0)
      throw std::invalid_argument("simulator geometry is outside selective planner ABI");
    // The official final actionable step is episode_steps-2. Keep the
    // downstream 24-tick ABI, but represent time beyond that step only as the
    // production forecaster's inert terminal padding. No fabricated action
    // or market window is accepted there.
    const int terminal_exclusive = config.episode_steps - 1;
    if (step < 0 || step >= terminal_exclusive)
      throw std::invalid_argument("current step is not actionable");
    const int live_horizon = std::min(kCompiledHorizon,
                                      terminal_exclusive - step);
    const int required_future_frames = live_horizon - 1;
    const int maximum_future_frames = required_future_frames +
        static_cast<int>(step + live_horizon < terminal_exclusive);
    if (source.future_plan.size() <
            static_cast<std::size_t>(required_future_frames) ||
        source.future_plan.size() >
            static_cast<std::size_t>(maximum_future_frames)) {
      throw std::invalid_argument(
          "future selected plan does not cover exactly the remaining live horizon");
    }

    result.audit.current_step = step;
    result.audit.provided_future_frames =
        static_cast<int>(source.future_plan.size());
    result.audit.compiled_horizon_ticks = kCompiledHorizon;
    result.audit.validated_lookahead_frames =
        static_cast<int>(source.future_plan.size()) - required_future_frames;

    if (!validate_public_snapshot(source.public_belief, simulator,
                                  result.reason))
      return result;
    result.audit.public_snapshot_exact = true;

    const auto& farm = simulator.farms()[static_cast<std::size_t>(source.player)];
    if (source.current_legacy.units.size() != farm.hands.size() + 1)
      throw std::invalid_argument("current unit vector is not bound to every focal actor");
    std::vector<po::UnitAction> current_units;
    if (!convert_unit_vector(source.current_legacy.units, current_units,
                             result.reason))
      return result;
    std::vector<g001::market::Order> current_market;
    if (!convert_market_vector(source.current_legacy.market,
                               config.max_market_orders, current_market,
                               result.reason))
      return result;
    result.audit.converted_current_market_orders =
        static_cast<int>(current_market.size());

    int expected_step = step + 1;
    for (const auto& frame : source.future_plan) {
      if (frame.step != expected_step++)
        throw std::invalid_argument("future selected plan is not consecutive from next step");
      if (frame.step >= config.episode_steps - 1)
        throw std::invalid_argument("future selected frame is not an actionable episode step");
      if (!frame.causal_queue_known)
        throw std::invalid_argument("future own queue is not causally known");
      if (frame.crosses_route_switch)
        throw std::invalid_argument("future selected plan crosses unresolved route switch");
      if (frame.selected.units.empty())
        throw std::invalid_argument("future unit frame omits the farmer action");
      std::vector<po::UnitAction> checked_units;
      if (!convert_unit_vector(frame.selected.units, checked_units, result.reason))
        return result;
      std::vector<g001::market::Order> checked_market;
      if (!convert_market_vector(frame.selected.market, config.max_market_orders,
                                 checked_market, result.reason))
        return result;
    }
    result.audit.typed_market_roundtrip_exact = true;

    result.input = selective_runtime::Input{};
    result.input.maximum_search_states = kNativeMaximumQueueSearchStates;
    result.input.production.maximum_resource_derivation_iterations =
        kNativeMaximumResourceDerivationIterations;
    result.input.planner_config =
        g001::general::conservative_deployment_config();
    result.input.planner_config.rolling.threads = 1;
    result.audit.planner_threads_one = true;
    result.input.episode_steps = config.episode_steps;
    result.input.allow_sale_deferral = false;
    result.input.allow_buyable_product_overlay = false;
    result.input.maximum_market_orders = config.max_market_orders;
    result.input.shed_capacity = config.shed_capacity;
    result.input.production_config.horizon_ticks = kCompiledHorizon;
    result.input.production_config.episode_steps = config.episode_steps;
    // A receding horizon is not the end of the game.  Treat inventory left at
    // the boundary as an asset by liquidating it through the same conservative
    // causal market scenarios used by the planner.  Zero terminal inventory
    // value systematically made any immediate low-price SELL look profitable.
    result.input.production_config.liquidate_own_at_end =
        live_horizon == kCompiledHorizon &&
        step + kCompiledHorizon < terminal_exclusive;
    result.input.public_belief = source.public_belief;
    result.input.continuation = source.continuation;
    result.input.confirmed_prior_fill = source.confirmed_prior_fill;

    fill_current_state(result.input, simulator, source.player);
    po::UnitFrame current_frame;
    current_frame.step = step;
    current_frame.actor_actions = std::move(current_units);
    result.input.production.future_units.push_back(std::move(current_frame));

    legacy_baseline_forecast::OwnMarketFrame current_legacy;
    current_legacy.step = step;
    current_legacy.orders = current_market;
    current_legacy.causal_queue_known = true;
    result.input.own_market_frames.push_back(std::move(current_legacy));

    for (int offset = 1; offset < kCompiledHorizon; ++offset) {
      legacy_baseline_forecast::OwnMarketFrame market_frame;
      market_frame.step = step + offset;
      market_frame.causal_queue_known = true;
      if (offset < live_horizon) {
        const auto& source_frame = source.future_plan[
            static_cast<std::size_t>(offset - 1)];
        po::UnitFrame unit_frame;
        unit_frame.step = source_frame.step;
        if (!convert_unit_vector(source_frame.selected.units,
                                 unit_frame.actor_actions, result.reason))
          return result;
        result.input.production.future_units.push_back(std::move(unit_frame));
        market_frame.causal_queue_known = source_frame.causal_queue_known;
        market_frame.crosses_route_switch = source_frame.crosses_route_switch;
        if (!convert_market_vector(source_frame.selected.market,
                                   config.max_market_orders,
                                   market_frame.orders, result.reason))
          return result;
        result.audit.converted_future_market_orders +=
            static_cast<int>(market_frame.orders.size());
      }
      result.input.own_market_frames.push_back(std::move(market_frame));
    }

    result.input.current.step = step;
    result.input.current.day = simulator.day();
    const double own_money =
        simulator.farms()[static_cast<std::size_t>(source.player)].money;
    const double opponent_money =
        simulator.farms()[static_cast<std::size_t>(1 - source.player)].money;
    if (std::fabs(own_money - std::llround(own_money)) > 1e-9 ||
        std::fabs(opponent_money - std::llround(opponent_money)) > 1e-9)
      throw std::invalid_argument("public money is not exactly integral");
    result.input.current.own_money = std::llround(own_money);
    result.input.current.opponent_public_money = std::llround(opponent_money);
    result.input.current.official_shed_capacity = config.shed_capacity;
    for (int product = 0; product < fastkag::N_PRODUCTS; ++product) {
      const auto p = static_cast<std::size_t>(product);
      result.input.current.own_shed[p] =
          simulator.privates()[static_cast<std::size_t>(source.player)].shed[p];
      result.input.current.public_market_inventory[p] =
          simulator.market().inventory[p];
      result.input.current.public_market_price[p] = simulator.market().prices[p];
    }

    for (int offset = 0; offset < live_horizon; ++offset) {
      const int forecast_step = step + offset;
      result.input.market_schedule.remaining_sale_windows.push_back(forecast_step);
      const auto drain = deterministic_town_drain(simulator, forecast_step);
      if (!zero(drain)) {
        result.input.market_schedule.future_town_drain.push_back(
            {forecast_step, drain});
        ++result.audit.deterministic_town_drain_steps;
      }
    }
    result.audit.sale_windows = live_horizon;

    if (source.confirmed_prior_fill) {
      const auto& fill = *source.confirmed_prior_fill;
      if (!source.continuation || !source.continuation->pending ||
          !fill.consistent || fill.lower < 0 || fill.lower > fill.point ||
          fill.point > fill.upper ||
          fill.upper > source.continuation->pending->requested) {
        throw std::invalid_argument("confirmed prior fill has no valid pending continuation");
      }
    }

    result.accepted = true;
    result.reason = "accepted exact live causal plan plus inert terminal padding";
  } catch (const std::exception& error) {
    result.accepted = false;
    result.reason = error.what();
  }
  return result;
}

}  // namespace native_selective_input
