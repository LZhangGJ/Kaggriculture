#include "effect_certificate_semantics.hpp"

#include <algorithm>
#include <numeric>

namespace g001::effect_certificate {
namespace {

using fastkag::Item;
using fastkag::Op;
using fastkag::TileKind;

constexpr int crop_first_day[fastkag::N_CROPS] = {2, 2, 8, 10, 10};
constexpr bool crop_ongoing[fastkag::N_CROPS] = {
    false, false, true, true, false};
constexpr int animal_product[fastkag::N_ANIMALS] = {5, 6, 7};
constexpr TileKind animal_structure[fastkag::N_ANIMALS] = {
    TileKind::COOP, TileKind::PASTURE, TileKind::PASTURE};

int shed_total(const LocalState& state) {
  return std::accumulate(state.shed.begin(), state.shed.end(), 0);
}

void erase_order(LocalState& state, int item) {
  std::erase(state.inventory_order, static_cast<int8_t>(item));
}

void inventory_add(LocalState& state, int item, int quantity) {
  if (quantity <= 0) return;
  if (state.inventory[static_cast<std::size_t>(item)] == 0)
    state.inventory_order.push_back(static_cast<int8_t>(item));
  state.inventory[static_cast<std::size_t>(item)] += quantity;
}

bool inventory_take(LocalState& state, int item, int quantity) {
  if (quantity <= 0 ||
      state.inventory[static_cast<std::size_t>(item)] < quantity)
    return false;
  state.inventory[static_cast<std::size_t>(item)] -= quantity;
  if (state.inventory[static_cast<std::size_t>(item)] == 0)
    erase_order(state, item);
  return true;
}

Footprint footprint_for(Op operation) {
  Footprint result;
  result.lifecycle_public_certainty = PublicCertainty::Unobservable;
  result.public_limitation =
      "current typed lifecycle Observation omits inventory insertion order";
  switch (operation) {
    case Op::DROP:
      result.reads = {"actor.position", "actor.inventory[*]",
                      "actor.inventory_order", "shed[*]", "shed.capacity"};
      result.writes = {"actor.inventory[*]", "actor.inventory_order",
                       "shed[*]"};
      result.public_limitation =
          "inventory_order is required for capacity-limited DROP distribution";
      break;
    case Op::PICKUP:
      result.reads = {"actor.position", "shed[item]", "actor.inventory[item]",
                      "actor.inventory_order"};
      result.writes = {"shed[item]", "actor.inventory[item]",
                       "actor.inventory_order"};
      break;
    case Op::PLACE:
      result.reads = {"actor.position", "tile.kind", "tile.animal",
                      "actor.inventory[item]", "actor.inventory_order",
                      "shed[*]", "shed.capacity", "day"};
      result.writes = {"tile[*]", "actor.inventory[item]",
                       "actor.inventory_order", "shed[item]"};
      break;
    case Op::FERTILIZE:
      result.reads = {"tile.kind", "tile.fertilized_until_day", "day",
                      "actor.inventory[FERTILIZER]", "actor.inventory_order"};
      result.writes = {"tile.fertilized_until_day",
                       "actor.inventory[FERTILIZER]", "actor.inventory_order"};
      result.public_limitation =
          "typed lifecycle tile omits fertilized_until_day and inventory_order";
      break;
    case Op::FEED:
      result.reads = {"tile.kind", "tile.fed_today",
                      "actor.inventory[WHEAT]", "actor.inventory_order"};
      result.writes = {"tile.fed_today", "actor.inventory[WHEAT]",
                       "actor.inventory_order"};
      break;
    case Op::COLLECT_FERTILIZER:
      result.reads = {"tile.kind", "tile.fertilizer_available",
                      "actor.inventory[FERTILIZER]", "actor.inventory_order"};
      result.writes = {"tile.fertilizer_available",
                       "actor.inventory[FERTILIZER]", "actor.inventory_order"};
      result.public_limitation =
          "typed lifecycle tile omits fertilizer_available and inventory_order";
      break;
    case Op::HARVEST:
      result.reads = {"tile.kind", "tile.crop", "tile.animal",
                      "tile.yield_units", "tile.planted_day", "day",
                      "actor.inventory[product]", "actor.inventory_order"};
      result.writes = {"tile[*]", "actor.inventory[product]",
                       "actor.inventory_order"};
      result.public_limitation =
          "crop identity/planted_day/ongoing kind and inventory_order are not "
          "all present in typed lifecycle Observation";
      break;
    default:
      result.lifecycle_public_certainty = PublicCertainty::GlobalOpaque;
      result.public_limitation = "operation is outside this certificate ABI";
      break;
  }
  return result;
}

Prediction finish(const LocalState& before, LocalState after, Failure failure,
                  int transferred, Footprint footprint) {
  Prediction result;
  result.changed = !(after == before);
  result.after = std::move(after);
  result.failure = failure;
  result.transferred = transferred;
  result.footprint = std::move(footprint);
  return result;
}

}  // namespace

bool operator==(const LocalState& lhs, const LocalState& rhs) {
  const auto& a = lhs.config;
  const auto& b = rhs.config;
  const bool same_config =
      a.episode_steps == b.episode_steps && a.board_size == b.board_size &&
      a.starting_money == b.starting_money &&
      a.max_market_orders == b.max_market_orders &&
      a.turns_per_day == b.turns_per_day &&
      a.shed_capacity == b.shed_capacity &&
      a.weed_spawn_chance == b.weed_spawn_chance &&
      a.town_shop_unlock_interval == b.town_shop_unlock_interval &&
      a.town_shop_sell_interval == b.town_shop_sell_interval &&
      a.town_center_sell_interval == b.town_center_sell_interval &&
      a.farm_hand_cost_mult == b.farm_hand_cost_mult;
  const auto same_position = lhs.actor_position.x == rhs.actor_position.x &&
                             lhs.actor_position.y == rhs.actor_position.y;
  const auto& x = lhs.tile;
  const auto& y = rhs.tile;
  const bool same_tile =
      x.kind == y.kind && x.crop == y.crop && x.animal == y.animal &&
      x.planted_day == y.planted_day && x.placed_day == y.placed_day &&
      x.yield_units == y.yield_units &&
      x.consecutive_unwatered == y.consecutive_unwatered &&
      x.consecutive_unfed == y.consecutive_unfed &&
      x.fertilized_until_day == y.fertilized_until_day &&
      x.pending_care_bonus == y.pending_care_bonus &&
      x.max_lifespan_step == y.max_lifespan_step &&
      x.watered_today == y.watered_today && x.fed_today == y.fed_today &&
      x.cared_today == y.cared_today &&
      x.fertilizer_available == y.fertilizer_available;
  return same_config && lhs.day == rhs.day &&
         lhs.actor_slot == rhs.actor_slot &&
         lhs.actor_exists == rhs.actor_exists && same_position &&
         lhs.shed_adjacent == rhs.shed_adjacent && same_tile &&
         lhs.shed == rhs.shed && lhs.inventory == rhs.inventory &&
         lhs.inventory_order == rhs.inventory_order;
}

Prediction predict_unit_effect(const LocalState& before,
                               const fastkag::Action& action) {
  LocalState after = before;
  auto footprint = footprint_for(action.op);
  if (!before.actor_exists)
    return finish(before, std::move(after), Failure::ActorMissing, 0,
                  std::move(footprint));
  const int item = static_cast<int>(action.item);
  const int requested = std::max(0, action.quantity);

  if (action.op == Op::DROP) {
    if (!before.shed_adjacent)
      return finish(before, std::move(after), Failure::NotShedAdjacent, 0,
                    std::move(footprint));
    int transferred = 0;
    for (const int8_t key : before.inventory_order) {
      const int index = key;
      const int room = std::max(0, before.config.shed_capacity -
                                      shed_total(after));
      const int take = std::min(after.inventory[static_cast<std::size_t>(index)],
                                room);
      after.shed[static_cast<std::size_t>(index)] += take;
      after.inventory[static_cast<std::size_t>(index)] = 0;
      transferred += take;
    }
    after.inventory_order.clear();
    return finish(before, std::move(after), Failure::None, transferred,
                  std::move(footprint));
  }

  if (action.op == Op::PICKUP) {
    if (!before.shed_adjacent)
      return finish(before, std::move(after), Failure::NotShedAdjacent, 0,
                    std::move(footprint));
    if (item < 0 || item >= fastkag::N_ITEMS || requested <= 0)
      return finish(before, std::move(after), Failure::InvalidItemOrQuantity, 0,
                    std::move(footprint));
    const int take = std::min(requested,
                              before.shed[static_cast<std::size_t>(item)]);
    if (take <= 0)
      return finish(before, std::move(after), Failure::ShedItemAbsent, 0,
                    std::move(footprint));
    after.shed[static_cast<std::size_t>(item)] -= take;
    inventory_add(after, item, take);
    return finish(before, std::move(after), Failure::None, take,
                  std::move(footprint));
  }

  if (action.op == Op::PLACE) {
    const bool animal = item >= 9 && item < 12;
    if (animal && before.tile.kind == animal_structure[item - 9] &&
        before.tile.animal == Item::NONE) {
      if (!inventory_take(after, item, 1))
        return finish(before, std::move(after), Failure::ActorItemAbsent, 0,
                      std::move(footprint));
      after.tile = {};
      after.tile.kind = TileKind::ANIMAL;
      after.tile.animal = static_cast<Item>(item);
      after.tile.placed_day = static_cast<int16_t>(before.day);
      return finish(before, std::move(after), Failure::None, 1,
                    std::move(footprint));
    }
    if (!before.shed_adjacent)
      return finish(before, std::move(after),
                    animal ? Failure::WrongAnimalStructure
                           : Failure::NotShedAdjacent,
                    0, std::move(footprint));
    if (item < 0 || item >= fastkag::N_ITEMS || requested <= 0)
      return finish(before, std::move(after), Failure::InvalidItemOrQuantity, 0,
                    std::move(footprint));
    const int room = std::max(
        0, before.config.shed_capacity - shed_total(before));
    const int take = std::min(
        {requested, before.inventory[static_cast<std::size_t>(item)], room});
    if (take <= 0)
      return finish(before, std::move(after),
                    room <= 0 ? Failure::ShedFull : Failure::ActorItemAbsent,
                    0, std::move(footprint));
    static_cast<void>(inventory_take(after, item, take));
    after.shed[static_cast<std::size_t>(item)] += take;
    return finish(before, std::move(after), Failure::None, take,
                  std::move(footprint));
  }

  if (before.tile.kind == TileKind::LOCKED)
    return finish(before, std::move(after), Failure::LockedTile, 0,
                  std::move(footprint));

  if (action.op == Op::FERTILIZE) {
    if (before.tile.kind != TileKind::PLANT)
      return finish(before, std::move(after), Failure::TileNotPlant, 0,
                    std::move(footprint));
    if (!inventory_take(after, 8, 1))
      return finish(before, std::move(after), Failure::FertilizerAbsent, 0,
                    std::move(footprint));
    after.tile.fertilized_until_day = static_cast<int16_t>(std::max(
        static_cast<int>(before.tile.fertilized_until_day), before.day + 2));
    return finish(before, std::move(after), Failure::None, 1,
                  std::move(footprint));
  }

  if (action.op == Op::FEED) {
    if (before.tile.kind != TileKind::ANIMAL)
      return finish(before, std::move(after), Failure::TileNotAnimal, 0,
                    std::move(footprint));
    if (before.tile.fed_today)
      return finish(before, std::move(after), Failure::AlreadyFed, 0,
                    std::move(footprint));
    if (!inventory_take(after, 0, 1))
      return finish(before, std::move(after), Failure::WheatAbsent, 0,
                    std::move(footprint));
    after.tile.fed_today = true;
    return finish(before, std::move(after), Failure::None, 1,
                  std::move(footprint));
  }

  if (action.op == Op::COLLECT_FERTILIZER) {
    if (before.tile.kind != TileKind::ANIMAL)
      return finish(before, std::move(after), Failure::TileNotAnimal, 0,
                    std::move(footprint));
    if (!before.tile.fertilizer_available)
      return finish(before, std::move(after), Failure::FertilizerUnavailable,
                    0, std::move(footprint));
    after.tile.fertilizer_available = false;
    inventory_add(after, 8, 1);
    return finish(before, std::move(after), Failure::None, 1,
                  std::move(footprint));
  }

  if (action.op == Op::HARVEST) {
    if (before.tile.yield_units <= 0)
      return finish(before, std::move(after), Failure::YieldAbsent, 0,
                    std::move(footprint));
    if (before.tile.kind == TileKind::PLANT) {
      const int crop = static_cast<int>(before.tile.crop);
      if (crop < 0 || crop >= fastkag::N_CROPS)
        return finish(before, std::move(after), Failure::InvalidTileIdentity, 0,
                      std::move(footprint));
      if (before.day - before.tile.planted_day < crop_first_day[crop])
        return finish(before, std::move(after), Failure::CropImmature, 0,
                      std::move(footprint));
      inventory_add(after, crop, before.tile.yield_units);
      const int produced = before.tile.yield_units;
      after.tile.yield_units = 0;
      if (!crop_ongoing[crop]) after.tile = {};
      return finish(before, std::move(after), Failure::None, produced,
                    std::move(footprint));
    }
    if (before.tile.kind == TileKind::ANIMAL) {
      const int animal = static_cast<int>(before.tile.animal) - 9;
      if (animal < 0 || animal >= fastkag::N_ANIMALS)
        return finish(before, std::move(after), Failure::InvalidTileIdentity, 0,
                      std::move(footprint));
      inventory_add(after, animal_product[animal], before.tile.yield_units);
      const int produced = before.tile.yield_units;
      after.tile.yield_units = 0;
      return finish(before, std::move(after), Failure::None, produced,
                    std::move(footprint));
    }
    return finish(before, std::move(after), Failure::InvalidTileIdentity, 0,
                  std::move(footprint));
  }

  return finish(before, std::move(after), Failure::UnsupportedOperation, 0,
                std::move(footprint));
}

const char* failure_name(Failure failure) noexcept {
  switch (failure) {
    case Failure::None: return "none";
    case Failure::UnsupportedOperation: return "unsupported-operation";
    case Failure::ActorMissing: return "actor-missing";
    case Failure::LockedTile: return "locked-tile";
    case Failure::NotShedAdjacent: return "not-shed-adjacent";
    case Failure::InvalidItemOrQuantity: return "invalid-item-or-quantity";
    case Failure::ShedItemAbsent: return "shed-item-absent";
    case Failure::ActorItemAbsent: return "actor-item-absent";
    case Failure::ShedFull: return "shed-full";
    case Failure::WrongAnimalStructure: return "wrong-animal-structure";
    case Failure::TileNotPlant: return "tile-not-plant";
    case Failure::FertilizerAbsent: return "fertilizer-absent";
    case Failure::TileNotAnimal: return "tile-not-animal";
    case Failure::AlreadyFed: return "already-fed";
    case Failure::WheatAbsent: return "wheat-absent";
    case Failure::FertilizerUnavailable: return "fertilizer-unavailable";
    case Failure::YieldAbsent: return "yield-absent";
    case Failure::CropImmature: return "crop-immature";
    case Failure::InvalidTileIdentity: return "invalid-tile-identity";
  }
  return "unknown";
}

}  // namespace g001::effect_certificate
