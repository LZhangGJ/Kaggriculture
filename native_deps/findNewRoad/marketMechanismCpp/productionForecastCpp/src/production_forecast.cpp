#include "production_forecast.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace production_forecast {
namespace {

using production_obligation::CompilerInput;
using production_obligation::Inventory;
using production_obligation::Item;
using production_obligation::NodeKind;
using production_obligation::ObligationNode;
using production_obligation::Position;
using production_obligation::TileKind;
using production_obligation::TileState;
using production_obligation::UnitAction;
using production_obligation::UnitFrame;
using production_obligation::UnitOp;

constexpr std::array<int, production_obligation::kCrops> kFirstDay{2, 2, 8, 10, 10};
constexpr std::array<int, production_obligation::kCrops> kCropMaxYield{6, 4, 4, 4, 6};
constexpr std::array<int, production_obligation::kAnimals> kAnimalProduct{5, 6, 7};
constexpr std::array<int, production_obligation::kAnimals> kAnimalMaxHeld{4, 6, 6};

int as_int(Item item) { return static_cast<int>(item); }
bool valid_item(Item item) {
  return as_int(item) >= 0 && as_int(item) < production_obligation::kItems;
}
bool product_item(Item item) {
  return as_int(item) >= 0 && as_int(item) < production_obligation::kProducts;
}
bool crop_item(Item item) {
  return as_int(item) >= 0 && as_int(item) < production_obligation::kCrops;
}
bool animal_item(Item item) {
  return as_int(item) >= 9 && as_int(item) < 12;
}
bool acquisition(NodeKind kind) {
  return kind == NodeKind::BuySeed || kind == NodeKind::BuyProduct ||
         kind == NodeKind::BuyAnimal || kind == NodeKind::Hire ||
         kind == NodeKind::BuyLand;
}
bool route_purchase(NodeKind kind) {
  return kind == NodeKind::Hire || kind == NodeKind::BuyLand;
}
bool move_op(UnitOp op) {
  return op == UnitOp::North || op == UnitOp::South ||
         op == UnitOp::East || op == UnitOp::West;
}
bool inside(Position position, int size) {
  return position.x >= 0 && position.y >= 0 &&
         position.x < size && position.y < size;
}
void move(Position& position, UnitOp op, int size) {
  Position next = position;
  if (op == UnitOp::North) --next.y;
  if (op == UnitOp::South) ++next.y;
  if (op == UnitOp::West) --next.x;
  if (op == UnitOp::East) ++next.x;
  if (inside(next, size)) position = next;
}
bool shed_adjacent(Position position, int size) {
  const int half = size / 2;
  return (position.x == half - 1 || position.x == half) &&
         (position.y == half - 1 || position.y == half);
}
Position default_spawn(int size) {
  const int half = size / 2;
  return {half - 1, half - 1};
}
Position spawn_hand(const std::vector<Position>& positions, int size) {
  const int half = size / 2;
  const std::array<Position, 4> candidates{{
      {half - 1, half - 1}, {half, half - 1},
      {half - 1, half}, {half, half}}};
  std::array<int, 4> count{};
  for (const Position position : positions) {
    for (int index = 0; index < 4; ++index) {
      if (position.x == candidates[index].x &&
          position.y == candidates[index].y) ++count[index];
    }
  }
  int best = 0;
  for (int index = 1; index < 4; ++index) {
    if (count[index] < count[best]) best = index;
  }
  return candidates[best];
}
int tile_index(Position position, int size) {
  return position.y * size + position.x;
}
int quadrant(Position position, int size) {
  const int half = size / 2;
  return (position.y < half ? 0 : 2) + (position.x < half ? 0 : 1);
}

void normalize(Inventory& inventory) {
  std::array<bool, production_obligation::kItems> seen{};
  std::vector<Item> order;
  for (const Item item : inventory.insertion_order) {
    if (!valid_item(item)) continue;
    const int index = as_int(item);
    if (!seen[index] && inventory.quantity[index] > 0) {
      seen[index] = true;
      order.push_back(item);
    }
  }
  for (int index = 0; index < production_obligation::kItems; ++index) {
    if (!seen[index] && inventory.quantity[index] > 0)
      order.push_back(static_cast<Item>(index));
  }
  inventory.insertion_order = std::move(order);
}
void add(Inventory& inventory, Item item, int quantity) {
  if (!valid_item(item) || quantity <= 0) return;
  const int index = as_int(item);
  if (inventory.quantity[index] == 0) inventory.insertion_order.push_back(item);
  inventory.quantity[index] += quantity;
}
int take(Inventory& inventory, Item item, int quantity) {
  if (!valid_item(item) || quantity <= 0) return 0;
  const int index = as_int(item);
  const int amount = std::min(quantity, inventory.quantity[index]);
  inventory.quantity[index] -= amount;
  if (inventory.quantity[index] == 0) {
    inventory.insertion_order.erase(
        std::remove(inventory.insertion_order.begin(),
                    inventory.insertion_order.end(), item),
        inventory.insertion_order.end());
  }
  return amount;
}

int total(const std::array<int, production_obligation::kItems>& shed) {
  int value = 0;
  for (const int quantity : shed) value += std::max(0, quantity);
  return value;
}

std::uint64_t fingerprint(const CompilerInput& input) {
  std::uint64_t hash = 1469598103934665603ULL;
  auto mix = [&](std::uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ULL;
  };
  mix(static_cast<std::uint64_t>(input.current.step));
  for (const UnitFrame& frame : input.future_units) {
    mix(static_cast<std::uint64_t>(frame.step));
    mix(static_cast<std::uint64_t>(frame.actor_actions.size()));
    for (const UnitAction& action : frame.actor_actions) {
      mix(static_cast<std::uint64_t>(action.op));
      mix(static_cast<std::uint64_t>(as_int(action.item) + 1));
      mix(static_cast<std::uint64_t>(std::max(0, action.quantity)));
    }
  }
  return hash;
}

struct CashBucket {
  std::int64_t critical{};
  std::int64_t route{};
  std::int64_t reserve_increment{};
  std::int64_t reserve_cumulative{};
  bool has_reserve{};
};

struct WorkingTile {
  TileState state;
  // Current observed yield is a causal lower bound only at the current step for
  // crops.  Animal-held yield does not decay before the next day boundary.
  bool crop_yield_exact_now{};
  bool animal_yield_exact_until_boundary{};
  int causal_yield_lower{};
};

struct WorkingState {
  std::array<int, production_obligation::kItems> shed{};
  std::array<int, production_obligation::kCrops> seeds{};
  std::vector<Position> positions;
  std::vector<Inventory> carried;
  std::vector<WorkingTile> tiles;
};

void add_requirement(g001::rolling::ForecastTick& tick,
                     g001::rolling::RequirementKind kind,
                     std::int64_t cash, int product, int quantity,
                     Audit& audit, bool pre_market = false) {
  g001::rolling::Requirement value{kind, cash, product, quantity};
  if (pre_market)
    tick.pre_market_requirements.push_back(value);
  else
    tick.requirements.push_back(value);
  if (kind == g001::rolling::RequirementKind::CriticalPurchase)
    ++audit.critical_purchase_requirements;
  else if (kind == g001::rolling::RequirementKind::Feed)
    ++audit.feed_requirements;
  else
    ++audit.route_hard_requirements;
}

void mark_conservative(TickAudit& tick, Audit& audit, std::string reason) {
  tick.exact_production = false;
  tick.conservative_reasons.push_back(std::move(reason));
  audit.used_conservative_production_lower_bound = true;
}

}  // namespace

Result compile(const Input& input) {
  const auto& current = input.production.current;
  if (input.config.horizon_ticks < kMinimumHorizonTicks ||
      input.config.horizon_ticks > kMaximumHorizonTicks)
    throw std::invalid_argument(
        "productionForecastCpp horizon must be within 24..96 ticks");
  if (input.config.episode_steps <= 1 || current.step < 0 ||
      current.turns_per_day <= 0 || current.board_size <= 1 ||
      current.shed_capacity < 0) {
    throw std::invalid_argument("invalid production forecast scalar input");
  }
  if (input.config.require_complete_causal_unit_tape) {
    if (input.production.future_units.size() !=
        static_cast<std::size_t>(input.config.horizon_ticks))
      throw std::invalid_argument(
          "complete causal unit tape does not match forecast horizon");
    for (int offset = 0; offset < input.config.horizon_ticks; ++offset)
      if (input.production.future_units[static_cast<std::size_t>(offset)].step !=
          current.step + offset)
        throw std::invalid_argument(
            "complete causal unit tape is not ordered and contiguous");
  }

  Result result;
  result.audit.accepted = input.obligations.feasible;
  if (!input.obligations.feasible)
    result.audit.diagnostics.push_back("production obligation DAG is infeasible");
  result.audit.raw_unit_fingerprint = fingerprint(input.production);
  result.forecast.shed_capacity = current.shed_capacity;
  result.forecast.liquidate_own_at_end = input.config.liquidate_own_at_end;
  result.forecast.ticks.resize(input.config.horizon_ticks);
  result.audit.ticks.resize(input.config.horizon_ticks);

  std::map<int, std::size_t> tick_at;
  const int terminal_exclusive = input.config.episode_steps - 1;
  for (int offset = 0; offset < input.config.horizon_ticks; ++offset) {
    const int step = current.step + offset;
    auto& tick = result.forecast.ticks[offset];
    auto& audit_tick = result.audit.ticks[offset];
    tick.step = step;
    audit_tick.step = step;
    if (step >= terminal_exclusive) {
      audit_tick.terminal_padding = true;
      ++result.audit.terminal_padding_ticks;
      continue;
    }
    tick.sale_window = true;  // Every live official step has a market phase.
    tick.decision_epoch = offset == 0 || step % current.turns_per_day == 0;
    tick_at[step] = static_cast<std::size_t>(offset);
  }

  std::map<int, CashBucket> cash;
  std::map<int, std::vector<const ObligationNode*>> acquisitions;
  std::set<int> seen_node_ids;
  for (const ObligationNode& node : input.obligations.nodes) {
    if (node.id >= 0 && !seen_node_ids.insert(node.id).second) continue;
    if (node.execution_step < current.step) continue;
    CashBucket& bucket = cash[node.execution_step];
    if (node.kind == NodeKind::CashReserve) {
      bucket.has_reserve = true;
      bucket.reserve_increment = std::max<std::int64_t>(
          bucket.reserve_increment, std::max(0, node.cash_quote));
      bucket.reserve_cumulative = std::max<std::int64_t>(
          bucket.reserve_cumulative, std::max(0, node.cumulative_cash_quote));
    } else if (acquisition(node.kind)) {
      acquisitions[node.execution_step].push_back(&node);
      if (route_purchase(node.kind))
        bucket.route += std::max(0, node.cash_quote);
      else
        bucket.critical += std::max(0, node.cash_quote);
    }
  }
  std::int64_t previous_reserve = 0;
  for (auto& [step, bucket] : cash) {
    const std::int64_t acquisition_cash = bucket.critical + bucket.route;
    if (bucket.has_reserve) {
      const std::int64_t delta = std::max<std::int64_t>(
          0, bucket.reserve_cumulative - previous_reserve);
      const std::int64_t reserve = bucket.reserve_increment > 0
                                       ? bucket.reserve_increment
                                       : delta;
      if ((bucket.reserve_increment > 0 && bucket.reserve_cumulative > 0 &&
           bucket.reserve_increment != delta) ||
          (acquisition_cash > 0 && reserve != acquisition_cash)) {
        result.audit.cash_nodes_consistent = false;
        result.audit.accepted = false;
        result.audit.diagnostics.push_back(
            "CashReserve disagrees with de-duplicated acquisitions at step " +
            std::to_string(step));
      }
      previous_reserve = std::max(previous_reserve, bucket.reserve_cumulative);
      if (acquisition_cash == 0) bucket.critical = reserve;
    } else {
      previous_reserve += acquisition_cash;
    }
    const auto tick = tick_at.find(step);
    if (tick == tick_at.end()) continue;
    if (bucket.critical > 0)
      add_requirement(result.forecast.ticks[tick->second],
                      g001::rolling::RequirementKind::CriticalPurchase,
                      bucket.critical, -1, 0, result.audit);
    if (bucket.route > 0)
      add_requirement(result.forecast.ticks[tick->second],
                      g001::rolling::RequirementKind::RouteHard,
                      bucket.route, -1, 0, result.audit);
  }

  std::map<int, UnitFrame> frames;
  for (const UnitFrame& frame : input.production.future_units) {
    if (!frames.emplace(frame.step, frame).second)
      throw std::invalid_argument("duplicate future unit frame");
  }

  std::map<std::tuple<int, int, int>, g001::rolling::RequirementKind>
      pickup_requirement_kind;
  std::map<int, const ObligationNode*> node_by_id;
  for (const ObligationNode& node : input.obligations.nodes) {
    if (node.id >= 0) node_by_id[node.id] = &node;
  }
  for (const production_obligation::Edge& edge : input.obligations.edges) {
    const auto before = node_by_id.find(edge.before);
    const auto after = node_by_id.find(edge.after);
    if (before == node_by_id.end() || after == node_by_id.end() ||
        before->second->kind != NodeKind::Pickup ||
        after->second->kind != NodeKind::Consume) {
      continue;
    }
    const ObligationNode& pickup = *before->second;
    const ObligationNode& consume = *after->second;
    auto kind = g001::rolling::RequirementKind::RouteHard;
    const auto frame = frames.find(consume.consumer_step);
    if (frame != frames.end() && consume.actor >= 0 &&
        consume.actor < static_cast<int>(frame->second.actor_actions.size()) &&
        frame->second.actor_actions[consume.actor].op == UnitOp::Feed) {
      kind = g001::rolling::RequirementKind::Feed;
    }
    pickup_requirement_kind[{pickup.consumer_step, pickup.actor,
                             as_int(pickup.item)}] = kind;
  }

  WorkingState state;
  state.shed = current.shed;
  state.seeds = current.seeds;
  state.positions = current.actor_positions;
  if (state.positions.empty()) state.positions.push_back(default_spawn(current.board_size));
  state.carried = current.carried;
  state.carried.resize(state.positions.size());
  for (Inventory& inventory : state.carried) normalize(inventory);
  state.tiles.resize(current.board_size * current.board_size);
  if (current.tiles.size() == state.tiles.size()) {
    for (std::size_t index = 0; index < state.tiles.size(); ++index) {
      state.tiles[index].state = current.tiles[index];
      state.tiles[index].crop_yield_exact_now =
          current.tiles[index].kind == TileKind::Plant;
      state.tiles[index].animal_yield_exact_until_boundary =
          current.tiles[index].kind == TileKind::Animal;
      state.tiles[index].causal_yield_lower =
          std::max(0, current.tiles[index].yield_units);
    }
  } else {
    for (int y = 0; y < current.board_size; ++y) {
      for (int x = 0; x < current.board_size; ++x) {
        if ((current.unlocked_mask &
             (1u << quadrant({x, y}, current.board_size))) == 0) {
          state.tiles[y * current.board_size + x].state.kind = TileKind::Locked;
        }
      }
    }
  }

  auto animal_shed = [&]() {
    return std::max(0, state.shed[9]) + std::max(0, state.shed[10]) +
           std::max(0, state.shed[11]);
  };
  int maximum_animals = animal_shed();

  for (int offset = 0; offset < input.config.horizon_ticks; ++offset) {
    const int step = current.step + offset;
    auto& tick = result.forecast.ticks[offset];
    auto& audit_tick = result.audit.ticks[offset];
    if (audit_tick.terminal_padding) continue;

    const auto frame_it = frames.find(step);
    if (frame_it != frames.end()) {
      const UnitFrame& frame = frame_it->second;
      if (input.config.require_complete_causal_unit_tape &&
          frame.actor_actions.size() != state.positions.size())
        throw std::invalid_argument(
            "complete causal unit frame does not match live actor count");
      while (state.positions.size() < frame.actor_actions.size()) {
        state.positions.push_back(spawn_hand(state.positions, current.board_size));
        state.carried.emplace_back();
      }
      for (std::size_t actor = 0; actor < frame.actor_actions.size(); ++actor) {
        const UnitAction& action = frame.actor_actions[actor];
        Position& position = state.positions[actor];
        Inventory& carried = state.carried[actor];
        if (move_op(action.op)) {
          move(position, action.op, current.board_size);
          continue;
        }
        if (action.op == UnitOp::Pass || !inside(position, current.board_size)) continue;
        WorkingTile& working_tile = state.tiles[tile_index(position, current.board_size)];
        TileState& tile = working_tile.state;
        const int quantity = std::max(0, action.quantity);

        if (action.op == UnitOp::Pickup && shed_adjacent(position, current.board_size) &&
            valid_item(action.item) && quantity > 0) {
          const int item = as_int(action.item);
          const int available = std::min(quantity, std::max(0, state.shed[item]));
          state.shed[item] -= available;
          add(carried, action.item, available);
          if (product_item(action.item)) {
            tick.pre_market_unit_delta[item] -= quantity;
            audit_tick.shed_outflow[item] += quantity;
            const auto classified = pickup_requirement_kind.find(
                {step, static_cast<int>(actor), item});
            const auto kind = classified == pickup_requirement_kind.end()
                                  ? g001::rolling::RequirementKind::RouteHard
                                  : classified->second;
            add_requirement(tick, kind, 0, item, quantity, result.audit, true);
          }
        } else if (action.op == UnitOp::Drop &&
                   shed_adjacent(position, current.board_size)) {
          normalize(carried);
          const std::vector<Item> order = carried.insertion_order;
          for (const Item item : order) {
            const int index = as_int(item);
            const int amount = carried.quantity[index];
            if (product_item(item) && amount > 0) {
              tick.pre_market_unit_delta[index] += amount;
              audit_tick.shed_inflow_lower[index] += amount;
              audit_tick.shed_inflow_upper[index] += amount;
            }
            const int room = std::max(0, current.shed_capacity - total(state.shed));
            const int accepted = std::min(amount, room);
            state.shed[index] += accepted;
            take(carried, item, amount);
          }
        } else if (action.op == UnitOp::Place && animal_item(action.item)) {
          if ((action.item == Item::Goose && tile.kind == TileKind::Coop) ||
              (action.item != Item::Goose && tile.kind == TileKind::Pasture)) {
            if (take(carried, action.item, 1) == 1) {
              tile = TileState{};
              tile.kind = TileKind::Animal;
              tile.item = action.item;
              working_tile.animal_yield_exact_until_boundary = true;
              working_tile.causal_yield_lower = 0;
            }
          }
        } else if (action.op == UnitOp::Place && product_item(action.item) &&
                   shed_adjacent(position, current.board_size)) {
          const int item = as_int(action.item);
          const int amount = std::min(quantity, carried.quantity[item]);
          if (amount > 0) {
            tick.pre_market_unit_delta[item] += amount;
            audit_tick.shed_inflow_lower[item] += amount;
            audit_tick.shed_inflow_upper[item] += amount;
            const int room = std::max(0, current.shed_capacity - total(state.shed));
            const int accepted = std::min(amount, room);
            state.shed[item] += accepted;
            take(carried, action.item, amount);
          }
        } else if (tile.kind == TileKind::Locked) {
          continue;
        } else if (action.op == UnitOp::Plant && crop_item(action.item) &&
                   tile.kind == TileKind::Empty &&
                   state.seeds[as_int(action.item)] > 0) {
          --state.seeds[as_int(action.item)];
          tile = TileState{};
          tile.kind = TileKind::Plant;
          tile.item = action.item;
          tile.planted_day = step / current.turns_per_day;
          tile.yield_units = 0;  // stochastic weed/lifecycle-safe lower bound
          working_tile.crop_yield_exact_now = false;
          working_tile.causal_yield_lower = 0;
        } else if (action.op == UnitOp::Harvest &&
                   (tile.kind == TileKind::Plant || tile.kind == TileKind::Animal)) {
          bool exact = false;
          Item product = Item::None;
          int upper = std::max(0, tile.yield_units);
          if (tile.kind == TileKind::Plant) {
            const bool mature = crop_item(tile.item) &&
                step / current.turns_per_day - tile.planted_day >=
                    kFirstDay[as_int(tile.item)];
            if (!mature) continue;
            exact = mature && offset == 0 && working_tile.crop_yield_exact_now;
            product = tile.item;
            if (exact && tile.yield_units <= 0) continue;
            if (!exact) upper = kCropMaxYield[as_int(tile.item)];
          } else {
            exact = working_tile.animal_yield_exact_until_boundary;
            if (!animal_item(tile.item)) {
              mark_conservative(audit_tick, result.audit,
                                "ANIMAL tile has no valid animal item");
              continue;
            }
            const int animal = as_int(tile.item) - 9;
            product = static_cast<Item>(kAnimalProduct[animal]);
            if (exact && tile.yield_units <= 0) continue;
            upper = exact ? upper : kAnimalMaxHeld[animal];
          }
          const int lower = exact ? std::max(0, tile.yield_units)
                                  : (tile.kind == TileKind::Animal
                                         ? working_tile.causal_yield_lower
                                         : 0);
          if (product_item(product)) {
            audit_tick.production_yield_lower[as_int(product)] += lower;
            audit_tick.production_yield_upper[as_int(product)] += upper;
          }
          if (lower > 0) add(carried, product, lower);
          if (!exact) {
            mark_conservative(
                audit_tick, result.audit,
                "future HARVEST yield/lifespan is not fully represented by TileState; executable forecast uses its causal lower bound");
            if (product_item(product))
              audit_tick.shed_inflow_upper[as_int(product)] += upper;
          }
          working_tile.causal_yield_lower = 0;
          if (exact && lower > 0 && tile.kind == TileKind::Plant &&
              tile.item != Item::Tomato && tile.item != Item::Strawberry) {
            tile = TileState{};
          } else if (exact || tile.kind == TileKind::Animal) {
            tile.yield_units = 0;
          }
        } else if (action.op == UnitOp::CollectFertilizer &&
                   tile.kind == TileKind::Animal && tile.fertilizer_available) {
          tile.fertilizer_available = false;
          add(carried, Item::Fertilizer, 1);
          ++audit_tick.production_yield_lower[as_int(Item::Fertilizer)];
          ++audit_tick.production_yield_upper[as_int(Item::Fertilizer)];
        } else if (action.op == UnitOp::Feed && tile.kind == TileKind::Animal &&
                   !tile.fed_today) {
          if (take(carried, Item::Wheat, 1) == 1) tile.fed_today = true;
        } else if (action.op == UnitOp::Fertilize && tile.kind == TileKind::Plant) {
          (void)take(carried, Item::Fertilizer, 1);
        } else if (action.op == UnitOp::Dig && tile.kind != TileKind::Empty &&
                   tile.kind != TileKind::Animal) {
          tile = TileState{};
        } else if (action.op == UnitOp::BuildCoop && tile.kind == TileKind::Empty) {
          tile.kind = TileKind::Coop;
        } else if (action.op == UnitOp::BuildPasture && tile.kind == TileKind::Empty) {
          tile.kind = TileKind::Pasture;
        }
      }
    }

    // This is the sellable stock boundary.  Recording the simulated state
    // here avoids reconstructing it from requested PICKUP/DROP quantities and
    // keeps same-tick purchases and end-of-day returns out of the snapshot.
    audit_tick.pre_market_shed_valid = true;
    audit_tick.pre_market_shed = state.shed;

    // Market obligations execute after all unit actions.  Their cash was
    // emitted above exactly once per execution bucket.
    const auto purchases = acquisitions.find(step);
    if (purchases != acquisitions.end()) {
      for (const ObligationNode* node : purchases->second) {
        const int quantity = std::max(0, node->quantity);
        if (node->kind == NodeKind::BuySeed && crop_item(node->item)) {
          state.seeds[as_int(node->item)] += quantity;
        } else if ((node->kind == NodeKind::BuyProduct ||
                    node->kind == NodeKind::BuyAnimal) && valid_item(node->item)) {
          state.shed[as_int(node->item)] += quantity;
          if (node->kind == NodeKind::BuyProduct && product_item(node->item)) {
            tick.post_market_delta[as_int(node->item)] += quantity;
            audit_tick.shed_inflow_lower[as_int(node->item)] += quantity;
            audit_tick.shed_inflow_upper[as_int(node->item)] += quantity;
          }
        } else if (node->kind == NodeKind::Hire) {
          state.positions.push_back(spawn_hand(state.positions, current.board_size));
          state.carried.emplace_back();
        } else if (node->kind == NodeKind::BuyLand && node->quadrant >= 1 &&
                   node->quadrant <= 3) {
          for (int y = 0; y < current.board_size; ++y) {
            for (int x = 0; x < current.board_size; ++x) {
              WorkingTile& land = state.tiles[y * current.board_size + x];
              if (quadrant({x, y}, current.board_size) == node->quadrant &&
                  land.state.kind == TileKind::Locked) {
                land = WorkingTile{};
              }
            }
          }
        }
      }
    }
    maximum_animals = std::max(maximum_animals, animal_shed());

    if ((step + 1) % current.turns_per_day == 0) {
      for (Inventory& carried : state.carried) {
        normalize(carried);
        const std::vector<Item> order = carried.insertion_order;
        for (const Item item : order) {
          const int index = as_int(item);
          const int amount = carried.quantity[index];
          if (product_item(item) && amount > 0) {
            tick.end_of_day_return[index] += amount;
            audit_tick.shed_inflow_lower[index] += amount;
            audit_tick.shed_inflow_upper[index] += amount;
          }
          const int room = std::max(0, current.shed_capacity - total(state.shed));
          const int accepted = std::min(amount, room);
          state.shed[index] += accepted;
          take(carried, item, amount);
        }
      }
      state.positions.assign(1, default_spawn(current.board_size));
      state.carried.assign(1, Inventory{});
      for (WorkingTile& tile : state.tiles) {
        if (tile.state.kind == TileKind::Plant) {
          // Missing watered/fertilized/lifespan fields prohibit an optimistic
          // generated-yield forecast across the boundary.
          tile.state.yield_units = 0;
          tile.crop_yield_exact_now = false;
          tile.causal_yield_lower = 0;
        } else if (tile.state.kind == TileKind::Animal) {
          tile.state.consecutive_unfed = tile.state.fed_today
                                             ? 0
                                             : tile.state.consecutive_unfed + 1;
          if (tile.state.consecutive_unfed >= 2) {
            tile.state.kind = tile.state.item == Item::Goose
                                  ? TileKind::Coop
                                  : TileKind::Pasture;
            tile.state.item = Item::None;
            tile.state.yield_units = 0;
            tile.causal_yield_lower = 0;
            tile.animal_yield_exact_until_boundary = true;
            continue;
          }
          // Existing held yield remains a causal lower bound, while missing
          // placed-day/care fields make newly generated units uncertain.
          tile.animal_yield_exact_until_boundary = false;
          tile.state.fed_today = false;
          tile.state.fertilizer_available = true;
        }
      }
      maximum_animals = std::max(maximum_animals, animal_shed());
    }
  }

  result.audit.maximum_animals_in_shed = maximum_animals;
  result.forecast.official_shed_capacity = current.shed_capacity;
  result.forecast.hire_cost_multiplier = current.farm_hand_cost_mult;
  result.forecast.turns_per_day = current.turns_per_day;
  auto& initial_market = result.forecast.initial_own_market;
  for (int product = 0; product < production_obligation::kProducts; ++product)
    initial_market.shed[static_cast<std::size_t>(product)] = current.shed[product];
  for (int crop = 0; crop < production_obligation::kCrops; ++crop)
    initial_market.seeds[static_cast<std::size_t>(crop)] = current.seeds[crop];
  for (int animal = 0; animal < production_obligation::kAnimals; ++animal)
    initial_market.animals[static_cast<std::size_t>(animal)] = current.shed[9 + animal];
  initial_market.hires_today = current.hires_today;
  initial_market.hands = std::max(0, static_cast<int>(current.actor_positions.size()) - 1);
  initial_market.unlocked_quadrants = std::popcount(current.unlocked_mask & 0x0fU);
  result.forecast.shed_capacity = std::max(0, current.shed_capacity - maximum_animals);
  if (maximum_animals > 0) {
    result.audit.diagnostics.push_back(
        "rolling product capacity conservatively reserves the maximum causal animal occupancy");
  }
  return result;
}

}  // namespace production_forecast
