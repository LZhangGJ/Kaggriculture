// Licensed under the Apache License, Version 2.0.
#include "native_general_market.hpp"

#include "production_obligation.hpp"
#include "unified_market_allocator.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace fastkag {
namespace {

namespace po = production_obligation;
namespace um = unified_market;

po::UnitOp unit_op(Op op) {
  if (op < Op::PASS || op > Op::CARE)
    throw std::invalid_argument("future unit plan contains a market operation");
  return static_cast<po::UnitOp>(static_cast<int>(op));
}

po::Item item(Item value) {
  const int raw = static_cast<int>(value);
  if (raw < -1 || raw >= N_ITEMS)
    throw std::invalid_argument("future unit plan contains an invalid item");
  return static_cast<po::Item>(raw);
}

po::TileKind tile_kind(TileKind value) {
  return static_cast<po::TileKind>(static_cast<int>(value));
}

Action action(const g001::market::Order& order) {
  Action result;
  result.quantity = order.quantity;
  switch (order.operation) {
    case g001::market::Operation::Sell:
      result.op = Op::SELL;
      result.item = static_cast<Item>(static_cast<int>(order.product));
      break;
    case g001::market::Operation::BuyProduct:
      result.op = Op::BUY_PRODUCT;
      result.item = static_cast<Item>(static_cast<int>(order.product));
      break;
    case g001::market::Operation::BuySeed:
      result.op = Op::BUY_SEED;
      result.item = static_cast<Item>(static_cast<int>(order.product));
      break;
    case g001::market::Operation::BuyAnimal:
      result.op = Op::BUY_ANIMAL;
      result.item = static_cast<Item>(9 + static_cast<int>(order.animal));
      break;
    case g001::market::Operation::Hire: result.op = Op::HIRE; break;
    case g001::market::Operation::BuyLand: result.op = Op::BUY_LAND; break;
    case g001::market::Operation::Pass: result.op = Op::PASS; break;
  }
  return result;
}

Action action(const po::UnitAction& source) {
  Action result;
  result.op = static_cast<Op>(static_cast<int>(source.op));
  result.item = static_cast<Item>(static_cast<int>(source.item));
  result.quantity = source.quantity;
  return result;
}

po::CompilerInput compiler_input(const Simulator& env, int player,
                                 const std::vector<Action>& current_units,
                                 const std::vector<NativeFutureUnitFrame>& frames) {
  if (player < 0 || player > 1) throw std::invalid_argument("invalid focal player");
  if (frames.size() > 24) throw std::invalid_argument("future unit horizon exceeds 24 steps");
  po::CompilerInput input;
  auto& current = input.current;
  current.step = env.step_count();
  current.turns_per_day = env.config().turns_per_day;
  current.board_size = env.config().board_size;
  current.shed_capacity = env.config().shed_capacity;
  current.max_market_orders = env.config().max_market_orders;
  current.farm_hand_cost_mult = env.config().farm_hand_cost_mult;
  current.money = env.farms()[player].money;
  current.unlocked_mask = env.farms()[player].unlocked_mask;
  current.hires_today = env.farms()[player].hires_today;
  for (int index = 0; index < N_ITEMS; ++index)
    current.shed[index] = env.privates()[player].shed[index];
  for (int crop = 0; crop < N_CROPS; ++crop)
    current.seeds[crop] = env.privates()[player].seeds[crop];
  for (int product = 0; product < N_PRODUCTS; ++product)
    current.product_price[product] = env.market().prices[product];
  current.actor_positions.push_back(
      {env.farms()[player].farmer.x, env.farms()[player].farmer.y});
  for (const auto position : env.farms()[player].hands)
    current.actor_positions.push_back({position.x, position.y});
  const auto& private_state = env.privates()[player];
  current.carried.resize(private_state.inventories.size());
  for (std::size_t actor = 0; actor < private_state.inventories.size(); ++actor) {
    current.carried[actor].quantity = private_state.inventories[actor];
    if (actor < private_state.inventory_order.size()) {
      for (const int8_t value : private_state.inventory_order[actor])
        current.carried[actor].insertion_order.push_back(item(static_cast<Item>(value)));
    }
  }
  current.tiles.reserve(env.farms()[player].tiles.size());
  for (const auto& source : env.farms()[player].tiles) {
    po::TileState tile;
    tile.kind = tile_kind(source.kind);
    if (source.kind == TileKind::PLANT) tile.item = item(source.crop);
    else if (source.kind == TileKind::ANIMAL) tile.item = item(source.animal);
    tile.planted_day = source.planted_day;
    tile.yield_units = source.yield_units;
    tile.consecutive_unfed = source.consecutive_unfed;
    tile.fed_today = source.fed_today;
    tile.fertilizer_available = source.fertilizer_available;
    current.tiles.push_back(tile);
  }

  input.future_units.reserve(frames.size() + 1);
  po::UnitFrame current_frame;
  current_frame.step = env.step_count();
  current_frame.actor_actions.reserve(current_units.size());
  for (const auto& source_action : current_units) {
    po::UnitAction converted;
    converted.op = unit_op(source_action.op);
    converted.item = item(source_action.item);
    converted.quantity = source_action.quantity;
    current_frame.actor_actions.push_back(converted);
  }
  input.future_units.push_back(std::move(current_frame));

  int expected = env.step_count() + 1;
  for (const auto& source : frames) {
    if (source.step != expected++)
      throw std::invalid_argument("future unit frames must be consecutive from next step");
    po::UnitFrame frame;
    frame.step = source.step;
    frame.actor_actions.reserve(source.units.size());
    for (const auto& source_action : source.units) {
      po::UnitAction converted;
      converted.op = unit_op(source_action.op);
      converted.item = item(source_action.item);
      converted.quantity = source_action.quantity;
      frame.actor_actions.push_back(converted);
    }
    input.future_units.push_back(std::move(frame));
  }
  return input;
}

}  // namespace

NativeProductionObligationResult compile_native_production_obligations(
    const Simulator& env,
    int player,
    const std::vector<Action>& current_units,
    const std::vector<NativeFutureUnitFrame>& future_units) {
  NativeProductionObligationResult result;
  if (current_units.empty() && future_units.empty()) {
    result.feasible = true;
    result.audit.compiler_feasible = true;
    result.audit.reason = "no future production step remains";
    return result;
  }
  const auto soft = po::compile_soft_current(
      compiler_input(env, player, current_units, future_units));
  const auto& compiled = soft.dag;
  result.audit.compiler_feasible = compiled.feasible;
  result.audit.obligation_nodes = static_cast<int>(compiled.nodes.size());
  result.nodes = compiled.nodes;
  result.edges = compiled.edges;
  result.audit.soft_misses = compiled.soft_misses;
  if (env.step_count() < 220)
    result.audit.production_protection_soft_misses = compiled.soft_misses;
  for (const auto& replacement : soft.current_unit_replacements) {
    NativeCurrentUnitReplacement converted;
    converted.actor = replacement.actor;
    converted.original = action(replacement.original);
    converted.replacement = action(replacement.replacement);
    converted.soft_miss_node = replacement.soft_miss_node;
    if (replacement.soft_miss_node >= 0 &&
        replacement.soft_miss_node < static_cast<int>(compiled.nodes.size())) {
      converted.downstream_loss_proxy =
          compiled.nodes[replacement.soft_miss_node].downstream_loss_proxy;
    }
    result.audit.soft_downstream_loss_proxy += converted.downstream_loss_proxy;
    if (converted.replacement.op == Op::PICKUP)
      ++result.audit.soft_pickup_replacements;
    else if (converted.replacement.op == Op::PASS)
      ++result.audit.soft_pass_replacements;
    result.current_unit_replacements.push_back(converted);
  }
  if (!compiled.feasible) {
    if (!compiled.diagnostics.empty()) {
      const auto& diagnostic = compiled.diagnostics.front();
      result.audit.diagnostic_code = po::diagnostic_code_name(diagnostic.code);
      result.audit.diagnostic_step = diagnostic.step;
      result.audit.diagnostic_actor = diagnostic.actor;
      result.audit.diagnostic_item = static_cast<int>(diagnostic.item);
      result.audit.diagnostic_quantity = diagnostic.quantity;
    }
    result.audit.reason = compiled.diagnostics.empty()
        ? "production obligation compiler rejected the unit plan"
        : compiled.diagnostics.front().message;
    return result;
  }
  result.feasible = true;
  result.audit.reason = "production obligation DAG compiled";
  return result;
}

NativeGeneralMarketResult compile_native_general_market(
    const Simulator& env,
    int player,
    const std::vector<Action>& current_units,
    const std::vector<NativeFutureUnitFrame>& future_units) {
  return compile_native_general_market(env, player, current_units,
                                       future_units, {});
}

NativeGeneralMarketResult compile_native_general_market(
    const Simulator& env,
    int player,
    const std::vector<Action>& current_units,
    const std::vector<NativeFutureUnitFrame>& future_units,
    std::span<const Action> hard_current_acquisitions) {
  NativeGeneralMarketResult result;
  auto production = compile_native_production_obligations(
      env, player, current_units, future_units);
  result.audit = production.audit;
  result.current_unit_replacements =
      std::move(production.current_unit_replacements);
  if (!production.feasible) return result;
  if (current_units.empty() && future_units.empty() &&
      hard_current_acquisitions.empty()) {
    result.feasible = true;
    result.audit.allocator_feasible = true;
    return result;
  }

  std::vector<Action> merged_acquisitions;
  for (const auto& demand : hard_current_acquisitions) {
    const bool seed = demand.op == Op::BUY_SEED &&
                      demand.item >= Item::WHEAT &&
                      demand.item <= Item::MELON;
    const bool product = demand.op == Op::BUY_PRODUCT &&
                         (demand.item == Item::WHEAT ||
                          demand.item == Item::FERTILIZER);
    const bool animal = demand.op == Op::BUY_ANIMAL &&
                        demand.item >= Item::GOOSE &&
                        demand.item <= Item::SHEEP;
    if ((!seed && !product && !animal) || demand.quantity <= 0)
      throw std::invalid_argument("invalid hard current acquisition");
    const auto found = std::find_if(
        merged_acquisitions.begin(), merged_acquisitions.end(),
        [&](const Action& existing) {
          return existing.op == demand.op && existing.item == demand.item;
        });
    if (found == merged_acquisitions.end()) {
      merged_acquisitions.push_back(demand);
    } else if (demand.quantity >
               std::numeric_limits<int>::max() - found->quantity) {
      throw std::invalid_argument("hard current acquisition overflows");
    } else {
      found->quantity += demand.quantity;
    }
  }

  int next_node_id = 0;
  int next_order_slot = 0;
  for (const auto& node : production.nodes) {
    next_node_id = std::max(next_node_id, node.id + 1);
    if (node.execution_step == env.step_count())
      next_order_slot = std::max(next_order_slot, node.order_slot + 1);
  }
  const auto kind_for = [](Op op) {
    if (op == Op::BUY_SEED) return po::NodeKind::BuySeed;
    if (op == Op::BUY_PRODUCT) return po::NodeKind::BuyProduct;
    return po::NodeKind::BuyAnimal;
  };
  for (const auto& demand : merged_acquisitions) {
    const auto kind = kind_for(demand.op);
    const auto typed_item = item(demand.item);
    int already_due = 0;
    for (const auto& node : production.nodes) {
      if (node.kind == kind && node.item == typed_item &&
          node.execution_step == env.step_count())
        already_due += std::max(0, node.quantity);
    }
    const int quantity = std::max(0, demand.quantity - already_due);
    if (quantity == 0) continue;
    po::ObligationNode node;
    node.id = next_node_id++;
    node.kind = kind;
    node.item = typed_item;
    node.quantity = quantity;
    node.cumulative_quantity = quantity;
    node.deadline_step = env.step_count();
    node.earliest_step = env.step_count();
    node.execution_step = env.step_count();
    node.order_slot = next_order_slot++;
    node.reason = "state-target hard acquisition";
    production.nodes.push_back(std::move(node));
  }
  result.audit.obligation_nodes = static_cast<int>(production.nodes.size());
  const auto& production_nodes = production.nodes;

  um::Input allocation;
  allocation.step = env.step_count();
  allocation.maximum_slots = env.config().max_market_orders;
  allocation.shed_capacity = env.config().shed_capacity;
  for (int product = 0; product < N_PRODUCTS; ++product) {
    allocation.market_inventory[product] = env.market().inventory[product];
    allocation.own.shed[product] = env.privates()[player].shed[product];
    allocation.own.seeds[product] = product < N_CROPS
        ? env.privates()[player].seeds[product] : 0;
  }
  for (int animal = 0; animal < N_ANIMALS; ++animal)
    allocation.own.animals[animal] = env.privates()[player].shed[9 + animal];
  allocation.own.money = static_cast<std::int64_t>(
      std::llround(env.farms()[player].money));
  allocation.own.hires_today = env.farms()[player].hires_today;
  allocation.own.hands = static_cast<int>(env.farms()[player].hands.size());
  allocation.own.unlocked_quadrants = std::popcount(
      static_cast<unsigned>(env.farms()[player].unlocked_mask));
  allocation.production_dag = production_nodes;

  const po::ObligationNode* first_due = nullptr;
  const auto acquisition = [](po::NodeKind kind) {
    return kind == po::NodeKind::BuySeed || kind == po::NodeKind::BuyProduct ||
           kind == po::NodeKind::BuyAnimal || kind == po::NodeKind::Hire ||
           kind == po::NodeKind::BuyLand;
  };
  for (const auto& node : production_nodes) {
    if (!acquisition(node.kind) || node.execution_step != env.step_count()) continue;
    if (first_due == nullptr ||
        std::tie(node.order_slot, node.id) <
            std::tie(first_due->order_slot, first_due->id))
      first_due = &node;
  }
  if (first_due != nullptr) {
    result.audit.first_due_kind = po::node_kind_name(first_due->kind);
    result.audit.first_due_item = static_cast<int>(first_due->item);
    result.audit.first_due_quantity = first_due->quantity;
    result.audit.first_due_cash_quote = first_due->cash_quote;
    result.audit.first_due_free_capacity = first_due->free_capacity_required;
  }

  std::array<int, N_PRODUCTS> protected_stock{};
  for (const auto& node : production_nodes) {
    if (node.kind != po::NodeKind::Pickup) continue;
    const int product = static_cast<int>(node.item);
    if (product >= 0 && product < N_PRODUCTS)
      protected_stock[product] += std::max(0, node.quantity);
  }
  for (int product = 0; product < N_PRODUCTS; ++product) {
    um::SellIntent intent;
    intent.product = static_cast<g001::market::Product>(product);
    intent.maximum_quantity = allocation.own.shed[product];
    intent.protected_quantity = std::min(
        allocation.own.shed[product], protected_stock[product]);
    intent.reservation_price = 1;
    intent.priority = env.market().prices[product];
    intent.purpose = um::SellPurpose::Liquidity;
    allocation.sell_intents.push_back(intent);
  }
  const auto allocated = um::allocate(allocation);
  result.audit.allocator_feasible = allocated.feasible;
  result.audit.purchase_orders_due = allocated.audit.due_purchase_orders;
  result.audit.funding_sell_orders = allocated.audit.funding_sell_orders;
  result.audit.funding_sell_units = allocated.audit.funding_sell_units;
  result.audit.allocated_orders = static_cast<int>(allocated.orders.size());
  result.audit.reason = allocated.audit.reason;
  if (!allocated.feasible) return result;
  result.market.reserve(allocated.orders.size());
  for (const auto& order : allocated.orders) result.market.push_back(action(order));
  result.feasible = true;
  return result;
}

}  // namespace fastkag
