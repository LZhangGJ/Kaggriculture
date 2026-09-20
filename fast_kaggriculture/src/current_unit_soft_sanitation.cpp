// Licensed under the Apache License, Version 2.0.
#include "current_unit_soft_sanitation.hpp"

#include <algorithm>
#include <stdexcept>

namespace fastkag {
namespace {

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH ||
         op == Op::EAST || op == Op::WEST;
}

bool shed_adjacent(Position position, int board_size) {
  const int half = board_size / 2;
  return (position.x == half - 1 || position.x == half) &&
         (position.y == half - 1 || position.y == half);
}

int tile_index(Position position, int board_size) {
  return position.y * board_size + position.x;
}

int shed_sum(const std::array<int, N_ITEMS>& shed) {
  int total = 0;
  for (const int quantity : shed) total += quantity;
  return total;
}

TileKind animal_structure(Item item) {
  return item == Item::GOOSE ? TileKind::COOP : TileKind::PASTURE;
}

void erase_order_key(std::vector<int8_t>& order, int item) {
  order.erase(std::remove(order.begin(), order.end(), item), order.end());
}

void inventory_add(std::array<int, N_ITEMS>& inventory,
                   std::vector<int8_t>& order, int item, int quantity) {
  if (quantity <= 0) return;
  if (inventory[item] == 0) order.push_back(static_cast<int8_t>(item));
  inventory[item] += quantity;
}

void inventory_take(std::array<int, N_ITEMS>& inventory,
                    std::vector<int8_t>& order, int item, int quantity) {
  inventory[item] -= quantity;
  if (inventory[item] == 0) erase_order_key(order, item);
}

double loss_proxy(int shortfall, SoftRecovery recovery) {
  const int recovery_delay = recovery == SoftRecovery::Pickup ? 1 : 2;
  return 1.0 + std::max(1, shortfall) + recovery_delay;
}

}  // namespace

CurrentUnitSanitationInput current_unit_sanitation_input(
    const Simulator& env, int player, const std::vector<Action>& units) {
  if (player < 0 || player > 1) throw std::invalid_argument("invalid focal player");
  CurrentUnitSanitationInput input;
  input.step = env.step_count();
  input.board_size = env.config().board_size;
  input.shed_capacity = env.config().shed_capacity;
  input.units = units;
  const auto& farm = env.farms()[player];
  const auto& private_state = env.privates()[player];
  for (int item = 0; item < N_ITEMS; ++item) input.shed[item] = private_state.shed[item];
  for (int crop = 0; crop < N_CROPS; ++crop) input.seeds[crop] = private_state.seeds[crop];
  input.actor_positions.push_back(farm.farmer);
  input.actor_positions.insert(input.actor_positions.end(), farm.hands.begin(), farm.hands.end());
  input.carried.reserve(private_state.inventories.size());
  for (const auto& inventory : private_state.inventories) {
    std::array<int, N_ITEMS> converted{};
    for (int item = 0; item < N_ITEMS; ++item) converted[item] = inventory[item];
    input.carried.push_back(converted);
  }
  input.carried_order = private_state.inventory_order;
  input.tiles = farm.tiles;
  return input;
}

CurrentUnitSanitationResult sanitize_current_units(
    const CurrentUnitSanitationInput& input) {
  if (input.board_size <= 0 ||
      input.tiles.size() != static_cast<std::size_t>(input.board_size * input.board_size))
    throw std::invalid_argument("invalid sanitation board snapshot");
  if (input.shed_capacity < 0)
    throw std::invalid_argument("negative sanitation shed capacity");

  CurrentUnitSanitationResult result;
  result.production_protection_window = input.step < kProductionProtectionSteps;
  result.units = input.units;
  auto shed = input.shed;
  auto carried = input.carried;
  carried.resize(input.actor_positions.size());
  auto carried_order = input.carried_order;
  carried_order.resize(input.actor_positions.size());

  // Official Simulator blocks every same-crop PLANT when aggregate demand is
  // greater than the seed bank, rather than executing a first-fit prefix.
  std::array<int, N_CROPS> plant_demand{};
  for (const auto& action : input.units) {
    const int item = static_cast<int>(action.item);
    if (action.op == Op::PLANT && item >= 0 && item < N_CROPS) ++plant_demand[item];
  }
  std::array<bool, N_CROPS> plant_blocked{};
  for (int crop = 0; crop < N_CROPS; ++crop)
    plant_blocked[crop] = plant_demand[crop] > input.seeds[crop];

  auto miss = [&](std::size_t actor, SoftMissKind kind, int item,
                  int required, int available, std::string reason) {
    const Position position = input.actor_positions[actor];
    const bool can_pickup = kind != SoftMissKind::MissingSeed &&
        item >= 0 && item < N_ITEMS && shed_adjacent(position, input.board_size) &&
        shed[item] > 0;
    const SoftRecovery recovery = can_pickup ? SoftRecovery::Pickup : SoftRecovery::Pass;
    Action replacement;
    if (can_pickup) {
      replacement = {Op::PICKUP, static_cast<Item>(item), 1};
      --shed[item];
      inventory_add(carried[actor], carried_order[actor], item, 1);
      ++result.pickup_substitutions;
    } else {
      replacement = Action{};
      ++result.pass_substitutions;
    }
    result.units[actor] = replacement;
    SoftMiss value;
    value.step = input.step;
    value.actor = static_cast<int>(actor);
    value.original_op = input.units[actor].op;
    value.item = static_cast<Item>(item);
    value.required = required;
    value.available = available;
    value.shortfall = std::max(1, required - available);
    value.kind = kind;
    value.recovery = recovery;
    value.production_protection_window = result.production_protection_window;
    value.loss_proxy = loss_proxy(value.shortfall, recovery);
    value.reason = std::move(reason);
    result.total_loss_proxy += value.loss_proxy;
    ++result.unexpected_current_misses;
    if (value.production_protection_window) ++result.production_protection_misses;
    result.misses.push_back(std::move(value));
  };

  for (std::size_t actor = 0; actor < input.units.size(); ++actor) {
    const Action action = input.units[actor];
    if (is_move(action.op)) {
      ++result.move_actions_preserved;
      continue;
    }
    if (actor >= input.actor_positions.size()) continue;
    const Position position = input.actor_positions[actor];
    const Tile& tile = input.tiles[tile_index(position, input.board_size)];
    const int item = static_cast<int>(action.item);
    const int quantity = std::max(0, action.quantity);

    if (action.op == Op::PLANT && item >= 0 && item < N_CROPS &&
        plant_blocked[item]) {
      miss(actor, SoftMissKind::MissingSeed, item, plant_demand[item],
           input.seeds[item],
           "same-crop PLANT demand exceeds seeds before the unit phase");
      continue;
    }
    if (action.op == Op::FERTILIZE && tile.kind == TileKind::PLANT &&
        carried[actor][static_cast<int>(Item::FERTILIZER)] < 1) {
      miss(actor, SoftMissKind::MissingCarriedFertilizer,
           static_cast<int>(Item::FERTILIZER), 1,
           carried[actor][static_cast<int>(Item::FERTILIZER)],
           "FERTILIZE lacks carried fertilizer before the market phase");
      continue;
    }
    if (action.op == Op::FEED && tile.kind == TileKind::ANIMAL &&
        !tile.fed_today && carried[actor][static_cast<int>(Item::WHEAT)] < 1) {
      miss(actor, SoftMissKind::MissingCarriedWheat,
           static_cast<int>(Item::WHEAT), 1,
           carried[actor][static_cast<int>(Item::WHEAT)],
           "FEED lacks carried wheat before the market phase");
      continue;
    }
    if (action.op == Op::PLACE && item >= 0 && item < N_ITEMS) {
      const bool animal_place = item >= static_cast<int>(Item::GOOSE) &&
          tile.kind == animal_structure(action.item) && tile.animal == Item::NONE;
      const bool shed_place = item < static_cast<int>(Item::GOOSE) && quantity > 0 &&
          shed_adjacent(position, input.board_size);
      const int required = animal_place ? 1 : quantity;
      // Product PLACE is partial by official semantics, so only zero carried
      // stock is a complete resource miss. Animal PLACE is atomic.
      const bool missing = animal_place ? carried[actor][item] < 1
                                        : shed_place && carried[actor][item] == 0;
      if (missing) {
        miss(actor, SoftMissKind::MissingCarriedPlaceItem, item,
             std::max(1, required), carried[actor][item],
             "PLACE lacks carried stock before the market phase");
        continue;
      }
    }

    // Maintain exact-enough causal shed availability for later replacement
    // PICKUPs in actor execution order.
    if (action.op == Op::PICKUP && item >= 0 && item < N_ITEMS && quantity > 0 &&
        shed_adjacent(position, input.board_size)) {
      const int take = std::min(quantity, shed[item]);
      shed[item] -= take;
      inventory_add(carried[actor], carried_order[actor], item, take);
    } else if (action.op == Op::DROP && shed_adjacent(position, input.board_size)) {
      for (const int8_t raw : carried_order[actor]) {
        const int carried_item = raw;
        if (carried_item < 0 || carried_item >= N_ITEMS) continue;
        const int room = std::max(0, input.shed_capacity - shed_sum(shed));
        const int take = std::min(carried[actor][carried_item], room);
        shed[carried_item] += take;
        carried[actor][carried_item] = 0;
      }
      carried_order[actor].clear();
    } else if (action.op == Op::PLACE && item >= 0 && item < N_ITEMS) {
      if (item >= static_cast<int>(Item::GOOSE) &&
          tile.kind == animal_structure(action.item) && tile.animal == Item::NONE &&
          carried[actor][item] >= 1) {
        inventory_take(carried[actor], carried_order[actor], item, 1);
      } else if (item < static_cast<int>(Item::GOOSE) && quantity > 0 &&
                 shed_adjacent(position, input.board_size)) {
        const int room = std::max(0, input.shed_capacity - shed_sum(shed));
        const int take = std::min({quantity, carried[actor][item], room});
        if (take > 0) {
          inventory_take(carried[actor], carried_order[actor], item, take);
          shed[item] += take;
        }
      }
    } else if (action.op == Op::FERTILIZE && tile.kind == TileKind::PLANT &&
               carried[actor][static_cast<int>(Item::FERTILIZER)] >= 1) {
      inventory_take(carried[actor], carried_order[actor],
                     static_cast<int>(Item::FERTILIZER), 1);
    } else if (action.op == Op::FEED && tile.kind == TileKind::ANIMAL &&
               !tile.fed_today && carried[actor][static_cast<int>(Item::WHEAT)] >= 1) {
      inventory_take(carried[actor], carried_order[actor],
                     static_cast<int>(Item::WHEAT), 1);
    }
  }
  return result;
}

const char* soft_miss_kind_name(SoftMissKind kind) {
  switch (kind) {
    case SoftMissKind::MissingSeed: return "MISSING_SEED";
    case SoftMissKind::MissingCarriedWheat: return "MISSING_CARRIED_WHEAT";
    case SoftMissKind::MissingCarriedFertilizer: return "MISSING_CARRIED_FERTILIZER";
    case SoftMissKind::MissingCarriedPlaceItem: return "MISSING_CARRIED_PLACE_ITEM";
  }
  return "UNKNOWN";
}

const char* soft_recovery_name(SoftRecovery recovery) {
  switch (recovery) {
    case SoftRecovery::Pickup: return "PICKUP";
    case SoftRecovery::Pass: return "PASS";
  }
  return "UNKNOWN";
}

}  // namespace fastkag
