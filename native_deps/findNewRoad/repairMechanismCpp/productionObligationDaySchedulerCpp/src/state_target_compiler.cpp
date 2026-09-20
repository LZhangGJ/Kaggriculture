#include "../include/state_target_compiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <utility>

namespace g001::state_target {
namespace {

using obligation_day::GoalKind;
using obligation_day::ProductionObligation;
using obligation_day::ResourceNeed;
using fastkag::Item;
using fastkag::Position;
using fastkag::Tile;
using fastkag::TileKind;

bool valid_position(const fastkag::Simulator& state, Position position) {
  return position.x >= 0 && position.y >= 0 &&
         position.x < state.config().board_size &&
         position.y < state.config().board_size;
}

bool shed_adjacent(const fastkag::Simulator& state, Position position) {
  const int half = state.config().board_size / 2;
  return (position.x == half - 1 || position.x == half) &&
         (position.y == half - 1 || position.y == half);
}

const Tile& tile_at(const fastkag::Simulator& state, int player,
                    Position position) {
  return state.farms()[player].tiles
      [position.y * state.config().board_size + position.x];
}

bool same_position(Position lhs, Position rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

int priority(double value, int rank) {
  if (!std::isfinite(value)) return rank;
  const double low = std::numeric_limits<int>::min() + rank;
  const double high = std::numeric_limits<int>::max() - rank;
  return static_cast<int>(std::llround(std::clamp(value, low, high))) + rank;
}

double share(double value, int weight, int total_weight) {
  return std::isfinite(value) && total_weight > 0
             ? value * weight / total_weight
             : 0.0;
}

int group_rank(TargetGroupKind kind) {
  switch (kind) {
    case TargetGroupKind::AnimalAcquisition:
      return 50;
    case TargetGroupKind::AnimalFeed:
      return 40;
    case TargetGroupKind::AnimalHarvest:
      return 30;
    case TargetGroupKind::AnimalCare:
      return 20;
    case TargetGroupKind::AnimalFertilizer:
      return 10;
    case TargetGroupKind::CropLifecycle:
      return 0;
  }
  return 0;
}

double resource_group_value(const fastkag::Simulator& state, int player,
                            const PlotTarget& target) {
  if (!std::isfinite(target.value) ||
      !valid_position(state, target.tile))
    return -std::numeric_limits<double>::infinity();
  if (target.kind == PlotTargetKind::Crop) return target.value;
  if (target.kind != PlotTargetKind::Animal)
    return -std::numeric_limits<double>::infinity();
  const auto& tile = tile_at(state, player, target.tile);
  const bool missing_animal = tile.kind != TileKind::ANIMAL;
  if (missing_animal)
    return share(target.value, 4, 4 + (target.care ? 1 : 0));
  const bool feed = target.feed && !tile.fed_today;
  const int total_weight = (feed ? 4 : 0) +
      (target.maintain && tile.yield_units > 0 ? 3 : 0) +
      (target.care && !tile.cared_today ? 2 : 0) +
      (target.maintain && tile.fertilizer_available ? 1 : 0);
  return feed ? share(target.value, 4, total_weight)
              : -std::numeric_limits<double>::infinity();
}

}  // namespace

CompileResult compile_state_targets(const fastkag::Simulator& observation,
                                    int player, int earliest_step,
                                    int deadline_step,
                                    std::span<const PlotTarget> targets) {
  CompileResult result;
  const int day_end = std::min(
      (observation.day() + 1) * observation.config().turns_per_day - 1,
      observation.config().episode_steps - 2);
  if (player < 0 || player >= static_cast<int>(observation.farms().size()) ||
      earliest_step < observation.step_count() ||
      earliest_step > deadline_step || deadline_step > day_end ||
      observation.done()) {
    result.diagnostics.push_back(
        {0, DiagnosticCode::InvalidRequest,
         "player or current-day compile range is invalid"});
    return result;
  }

  const auto& farm = observation.farms()[player];
  const auto& private_state = observation.privates()[player];
  const int actors = 1 + static_cast<int>(farm.hands.size());
  auto available_seeds = private_state.seeds;
  auto available_shed = private_state.shed;
  auto available_carried = private_state.inventories;
  available_carried.resize(static_cast<std::size_t>(actors));
  std::vector<const PlotTarget*> ordered_targets;
  ordered_targets.reserve(targets.size());
  for (const auto& target : targets) ordered_targets.push_back(&target);
  std::stable_sort(ordered_targets.begin(), ordered_targets.end(),
                   [&](const PlotTarget* left, const PlotTarget* right) {
                     const double left_value =
                         resource_group_value(observation, player, *left);
                     const double right_value =
                         resource_group_value(observation, player, *right);
                     if (left_value != right_value)
                       return left_value > right_value;
                     return left->id < right->id;
                   });
  std::set<std::uint64_t> target_ids;
  std::vector<Position> claimed_tiles;
  std::uint64_t next_obligation_id = 1;
  std::uint64_t next_group_id = 1;

  auto add_group = [&](const PlotTarget& target, TargetGroupKind kind,
                       double value) {
    result.groups.push_back(
        {next_group_id++, target.id, kind, value, {}, {}});
    return result.groups.size() - 1;
  };
  auto diagnose = [&](const PlotTarget& target, TargetGroupKind kind,
                      DiagnosticCode code, std::string message) {
    (void)kind;
    result.diagnostics.push_back({target.id, code, std::move(message)});
  };
  auto add_demand = [&](std::size_t group_index, Item item,
                        ResourceDemandReason reason, int action_count) {
    const auto& group = result.groups[group_index];
    result.demands.push_back(
        {group.id, group.parent_target_id, group.value, item, 1,
         earliest_step, deadline_step - action_count, reason});
  };

  // Resource ownership must follow the same value order as atomic admission;
  // caller input order is not an economic priority.
  // ponytail: this is one-pass matching; replace it with joint group/resource
  // matching if paired tests find a value-order counterexample.
  for (const PlotTarget* target_pointer : ordered_targets) {
    const auto& target = *target_pointer;
    const auto default_group = target.kind == PlotTargetKind::Crop
                                   ? TargetGroupKind::CropLifecycle
                                   : TargetGroupKind::AnimalAcquisition;
    DiagnosticCode invalid_code = DiagnosticCode::InvalidTarget;
    std::string invalid_message;
    if (target.id == 0) {
      invalid_message = "target id must be nonzero";
    } else if (!target_ids.insert(target.id).second) {
      invalid_code = DiagnosticCode::DuplicateTarget;
      invalid_message = "target id is duplicated";
    } else if (!std::isfinite(target.value)) {
      invalid_message = "target value must be finite";
    } else if (!valid_position(observation, target.tile)) {
      invalid_message = "target tile is outside the board";
    } else if (std::any_of(claimed_tiles.begin(), claimed_tiles.end(),
                           [&](Position tile) {
                             return same_position(tile, target.tile);
                           })) {
      invalid_code = DiagnosticCode::DuplicateTile;
      invalid_message = "two production targets claim the same tile";
    } else if (target.actor < -1 || target.actor >= actors) {
      invalid_code = DiagnosticCode::InvalidActor;
      invalid_message = "target actor does not exist";
    } else if (target.kind != PlotTargetKind::Crop &&
               target.kind != PlotTargetKind::Animal) {
      invalid_message = "target kind is invalid";
    }
    if (!invalid_message.empty()) {
      diagnose(target, default_group, invalid_code, std::move(invalid_message));
      continue;
    }

    const Tile& tile = tile_at(observation, player, target.tile);
    const int item_index = static_cast<int>(target.item);
    auto add_obligation = [&](std::size_t group_index, int actor,
                              GoalKind goal, Position position, Item item,
                              int quantity,
                              std::vector<std::uint64_t> dependencies,
                              ResourceNeed resource, bool repair) {
      const auto& group = result.groups[group_index];
      ProductionObligation obligation{
          next_obligation_id++, actor, position, goal, item, quantity,
          std::move(dependencies), resource, earliest_step, deadline_step,
          priority(group.value, group_rank(group.kind)), true, repair, -1,
          false};
      result.groups[group_index].obligation_ids.push_back(obligation.id);
      result.obligations.push_back(std::move(obligation));
      return result.groups[group_index].obligation_ids.back();
    };

    if (target.kind == PlotTargetKind::Crop) {
      if (item_index < 0 || item_index >= fastkag::N_CROPS) {
        diagnose(target, default_group, DiagnosticCode::InvalidTarget,
                 "crop target item is not a crop");
        continue;
      }
      if (tile.kind == TileKind::ANIMAL) {
        diagnose(target, default_group, DiagnosticCode::CannotRemoveAnimal,
                 "crop target cannot remove an existing animal");
        continue;
      }
      if (tile.kind == TileKind::LOCKED) {
        diagnose(target, default_group, DiagnosticCode::UnsupportedTileState,
                 "crop target tile is locked");
        continue;
      }
      claimed_tiles.push_back(target.tile);
      static constexpr std::array<int, fastkag::N_CROPS> first_day{
          2, 2, 8, 10, 10};
      static constexpr std::array<bool, fastkag::N_CROPS> ongoing{
          false, false, true, true, false};
      const int existing_crop = tile.kind == TileKind::PLANT
                                    ? static_cast<int>(tile.crop)
                                    : -1;
      const bool harvest_needed = target.maintain &&
          existing_crop >= 0 && existing_crop < fastkag::N_CROPS &&
          tile.yield_units > 0 &&
          observation.day() - tile.planted_day >= first_day[existing_crop];
      const auto group = add_group(target, TargetGroupKind::CropLifecycle,
                                   target.value);
      std::vector<std::uint64_t> lifecycle_dependencies;
      if (harvest_needed) {
        lifecycle_dependencies.push_back(add_obligation(
            group, target.actor, GoalKind::Harvest, target.tile, tile.crop,
            tile.yield_units, {}, {}, false));
      }
      const bool needs_seed =
          (harvest_needed &&
           (!ongoing[existing_crop] || tile.crop != target.item)) ||
          tile.kind != TileKind::PLANT ||
          tile.crop != target.item;
      add_obligation(group, target.actor, GoalKind::CropReady, target.tile,
                     target.item, 1, std::move(lifecycle_dependencies),
                     {target.item, needs_seed ? 1 : 0, 0, 0}, needs_seed);
      if (needs_seed) {
        if (available_seeds[item_index] > 0) {
          --available_seeds[item_index];
        } else {
          const int transitions = harvest_needed
              ? (ongoing[existing_crop] ? 4 : 3)
              : tile.kind == TileKind::EMPTY ? 2 : 3;
          add_demand(group, target.item, ResourceDemandReason::Seed,
                     transitions);
        }
      }
      continue;
    }

    if (item_index < static_cast<int>(Item::GOOSE) ||
        item_index > static_cast<int>(Item::SHEEP)) {
      diagnose(target, default_group, DiagnosticCode::InvalidTarget,
               "animal target item is not an animal");
      continue;
    }
    const bool animal_present = tile.kind == TileKind::ANIMAL;
    if (animal_present && tile.animal != target.item) {
      diagnose(target, default_group, DiagnosticCode::CannotRemoveAnimal,
               "animal target cannot remove a different existing animal");
      continue;
    }
    const bool missing_animal = !animal_present;
    if (missing_animal && tile.kind == TileKind::LOCKED) {
      diagnose(target, default_group, DiagnosticCode::UnsupportedTileState,
               "animal target tile is locked");
      continue;
    }
    const bool feed_needed =
        target.feed && (missing_animal || !tile.fed_today);
    int actor = target.actor;
    if (actor == -1) {
      actor = 0;
      bool found = false;
      if (missing_animal) {
        for (int candidate = 0; candidate < actors; ++candidate) {
          if (available_carried[candidate][item_index] > 0) {
            actor = candidate;
            found = true;
            break;
          }
        }
      }
      if (!found && feed_needed) {
        const int wheat = static_cast<int>(Item::WHEAT);
        for (int candidate = 0; candidate < actors; ++candidate) {
          if (available_carried[candidate][wheat] > 0) {
            actor = candidate;
            break;
          }
        }
      }
    }
    const bool needs_supply =
        (missing_animal && available_carried[actor][item_index] == 0) ||
        (feed_needed &&
         available_carried[actor][static_cast<int>(Item::WHEAT)] == 0);
    if (needs_supply &&
        (!valid_position(observation, target.supply_position) ||
         !shed_adjacent(observation, target.supply_position))) {
      diagnose(target, default_group, DiagnosticCode::InvalidSupplyPosition,
               "animal workflow supply position is not shed-adjacent");
      continue;
    }
    claimed_tiles.push_back(target.tile);

    if (missing_animal) {
      const bool care_needed = target.care;
      const int total_weight = 4 + (care_needed ? 1 : 0);
      const auto acquisition_group = add_group(
          target, TargetGroupKind::AnimalAcquisition,
          share(target.value, 4, total_weight));
      std::vector<std::uint64_t> place_dependencies;
      const TileKind structure = target.item == Item::GOOSE
                                     ? TileKind::COOP
                                     : TileKind::PASTURE;
      if (tile.kind != structure) {
        place_dependencies.push_back(add_obligation(
            acquisition_group, actor,
            target.item == Item::GOOSE ? GoalKind::BuildCoop
                                       : GoalKind::BuildPasture,
            target.tile, Item::NONE, 1, {}, {}, true));
      }
      if (available_carried[actor][item_index] > 0) {
        --available_carried[actor][item_index];
      } else {
        place_dependencies.push_back(add_obligation(
            acquisition_group, actor, GoalKind::Pickup,
            target.supply_position, target.item, 1, {},
            {target.item, 0, 1, 0}, true));
        if (available_shed[item_index] > 0) {
          --available_shed[item_index];
        } else {
          add_demand(acquisition_group, target.item,
                     ResourceDemandReason::Animal, 1);
        }
      }
      const auto place_id = add_obligation(
          acquisition_group, actor, GoalKind::Place, target.tile, target.item,
          1, std::move(place_dependencies), {target.item, 0, 0, 1}, true);
      if (feed_needed) {
        std::vector<std::uint64_t> feed_dependencies{place_id};
        const int wheat = static_cast<int>(Item::WHEAT);
        if (available_carried[actor][wheat] > 0) {
          --available_carried[actor][wheat];
        } else {
          feed_dependencies.push_back(add_obligation(
              acquisition_group, actor, GoalKind::Pickup,
              target.supply_position, Item::WHEAT, 1, {},
              {Item::WHEAT, 0, 1, 0}, true));
          if (available_shed[wheat] > 0) {
            --available_shed[wheat];
          } else {
            add_demand(acquisition_group, Item::WHEAT,
                       ResourceDemandReason::FeedWheat, 1);
          }
        }
        add_obligation(acquisition_group, actor, GoalKind::Feed, target.tile,
                       Item::WHEAT, 1, std::move(feed_dependencies),
                       {Item::WHEAT, 0, 0, 1}, true);
      }
      if (care_needed) {
        const auto care_group = add_group(
            target, TargetGroupKind::AnimalCare,
            share(target.value, 1, total_weight));
        result.groups[care_group].dependency_group_ids.push_back(
            result.groups[acquisition_group].id);
        add_obligation(care_group, actor, GoalKind::Care, target.tile,
                       target.item, 1, {place_id}, {}, true);
      }
      const int group_actions = std::accumulate(
          result.groups[acquisition_group].obligation_ids.begin(),
          result.groups[acquisition_group].obligation_ids.end(), 0,
          [&](int count, std::uint64_t id) {
            const auto found = std::find_if(
                result.obligations.begin(), result.obligations.end(),
                [&](const ProductionObligation& obligation) {
                  return obligation.id == id;
                });
            if (found == result.obligations.end()) return count;
            if (found->goal == GoalKind::BuildPasture ||
                found->goal == GoalKind::BuildCoop)
              return count + (tile.kind == TileKind::EMPTY ? 1 : 2);
            return count + 1;
          });
      for (auto& demand : result.demands) {
        if (demand.group_id == result.groups[acquisition_group].id)
          demand.latest_purchase_step = deadline_step - group_actions;
      }
      continue;
    }

    const bool care_needed = target.care && !tile.cared_today;
    const bool harvest_needed = target.maintain && tile.yield_units > 0;
    const bool fertilizer_needed =
        target.maintain && tile.fertilizer_available;
    const int total_weight = (feed_needed ? 4 : 0) +
                             (harvest_needed ? 3 : 0) +
                             (care_needed ? 2 : 0) +
                             (fertilizer_needed ? 1 : 0);
    std::uint64_t feed_id = 0;
    std::uint64_t feed_group_id = 0;
    if (feed_needed) {
      const auto group = add_group(target, TargetGroupKind::AnimalFeed,
                                   share(target.value, 4, total_weight));
      feed_group_id = result.groups[group].id;
      std::vector<std::uint64_t> dependencies;
      const int wheat = static_cast<int>(Item::WHEAT);
      if (available_carried[actor][wheat] > 0) {
        --available_carried[actor][wheat];
      } else {
        dependencies.push_back(add_obligation(
            group, actor, GoalKind::Pickup, target.supply_position,
            Item::WHEAT, 1, {}, {Item::WHEAT, 0, 1, 0}, false));
        if (available_shed[wheat] > 0) {
          --available_shed[wheat];
        } else {
          add_demand(group, Item::WHEAT,
                     ResourceDemandReason::FeedWheat, 2);
        }
      }
      feed_id = add_obligation(
          group, actor, GoalKind::Feed, target.tile, Item::WHEAT, 1,
          std::move(dependencies), {Item::WHEAT, 0, 0, 1}, false);
    }
    if (care_needed) {
      const auto group = add_group(target, TargetGroupKind::AnimalCare,
                                   share(target.value, 2, total_weight));
      std::vector<std::uint64_t> dependencies;
      if (feed_id != 0) {
        dependencies.push_back(feed_id);
        result.groups[group].dependency_group_ids.push_back(feed_group_id);
      }
      add_obligation(group, actor, GoalKind::Care, target.tile, target.item, 1,
                     std::move(dependencies), {}, false);
    }
    if (harvest_needed) {
      const auto group = add_group(target, TargetGroupKind::AnimalHarvest,
                                   share(target.value, 3, total_weight));
      add_obligation(group, actor, GoalKind::Harvest, target.tile, target.item,
                     tile.yield_units, {}, {}, false);
    }
    if (fertilizer_needed) {
      const auto group = add_group(
          target, TargetGroupKind::AnimalFertilizer,
          share(target.value, 1, total_weight));
      add_obligation(group, actor, GoalKind::CollectFertilizer, target.tile,
                     Item::NONE, 1, {}, {}, false);
    }
  }
  return result;
}

std::vector<obligation_day::AtomicGroup> atomic_groups(
    const CompileResult& compiled) {
  std::vector<obligation_day::AtomicGroup> groups;
  groups.reserve(compiled.groups.size());
  for (const auto& group : compiled.groups)
    groups.push_back({group.id, group.value, group.obligation_ids});
  return groups;
}

std::vector<fastkag::Action> current_acquisitions(
    const CompileResult& compiled,
    std::span<const std::uint64_t> approved_group_ids) {
  std::vector<fastkag::Action> actions;
  actions.reserve(compiled.demands.size());
  for (const auto& demand : compiled.demands) {
    if (demand.quantity <= 0 || demand.unfilled_value <= 0.0 ||
        std::find(approved_group_ids.begin(), approved_group_ids.end(),
                  demand.group_id) == approved_group_ids.end() ||
        demand.request_step > demand.latest_purchase_step)
      continue;
    fastkag::Op op = fastkag::Op::BUY_PRODUCT;
    if (demand.reason == ResourceDemandReason::Seed)
      op = fastkag::Op::BUY_SEED;
    else if (demand.reason == ResourceDemandReason::Animal)
      op = fastkag::Op::BUY_ANIMAL;
    actions.push_back({op, demand.item, demand.quantity});
  }
  return actions;
}

}  // namespace g001::state_target
