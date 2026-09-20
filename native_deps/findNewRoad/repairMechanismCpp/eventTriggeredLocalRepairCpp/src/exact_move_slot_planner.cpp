#include "../include/exact_move_slot_planner.hpp"

#include <algorithm>
#include <stdexcept>

namespace g001::day_horizon_repair {
namespace {

using event_local_repair::Op;
using event_local_repair::TileKind;

Position after_move(Position position, const Action& action) {
  if (action.op != Op::Move) return position;
  if (action.arg0 == 0) --position.row;
  else if (action.arg0 == 1) ++position.row;
  else if (action.arg0 == 2) --position.column;
  else if (action.arg0 == 3) ++position.column;
  else throw std::invalid_argument("invalid exact-slot MOVE direction");
  return position;
}

bool crop_source(const Action& action) {
  return action.op == Op::Dig || action.op == Op::Plant ||
      action.op == Op::Water || action.op == Op::Harvest;
}

}  // namespace

ExactMoveSlotResult compile_exact_move_slots(
    const std::vector<ExactMoveSlotActor>& actors,
    std::map<Position, event_local_repair::TileObservation> tiles,
    std::map<int, int> seeds, std::vector<PersistentPlotIntent> carried,
    int day) {
  if (actors.empty()) throw std::invalid_argument("empty exact-slot horizon");
  const std::size_t turns = actors.front().raw.size();
  if (turns == 0) throw std::invalid_argument("empty exact-slot horizon");
  std::set<int> actor_ids;
  std::set<std::uint64_t> intent_ids;
  ExactMoveSlotResult result;
  const auto initial_tiles = tiles;
  const auto initial_seeds = seeds;
  result.active_intents = std::move(carried);
  for (const auto& intent : result.active_intents)
    if (intent.id == 0 || !intent_ids.insert(intent.id).second)
      throw std::invalid_argument("invalid carried plot intent id");
  result.manifest.reserve(actors.size());
  result.positions_before.resize(actors.size());
  for (std::size_t actor_index = 0; actor_index < actors.size(); ++actor_index) {
    const auto& actor = actors[actor_index];
    if (actor.actor < 0 || actor.raw.size() != turns ||
        actor.service_slot.size() != turns ||
        (!actor.trigger_source.empty() &&
         actor.trigger_source.size() != turns) ||
        !actor_ids.insert(actor.actor).second)
      throw std::invalid_argument("inconsistent exact-slot actor horizon");
    result.manifest.push_back(actor.raw);
    auto position = actor.start;
    for (const auto& raw : actor.raw) {
      result.positions_before[actor_index].push_back(position);
      position = after_move(position, raw);
    }
  }

  std::uint64_t next_id = 1;
  for (const auto id : intent_ids) next_id = std::max(next_id, id + 1);
  std::map<std::uint64_t, PersistentPlotIntent> intent_records;
  for (const auto& intent : result.active_intents)
    intent_records[intent.id] = intent;
  for (std::size_t turn = 0; turn < turns; ++turn) {
    // First absorb every source intent without changing the manifest/state.
    // This makes same-tile arbitration independent of actor iteration order.
    for (std::size_t actor = 0; actor < actors.size(); ++actor) {
      const auto& raw = actors[actor].raw[turn];
      const auto tile = result.positions_before[actor][turn];
      if (raw.op == Op::Move) continue;
      const bool trigger = actors[actor].trigger_source.empty()
          ? crop_source(raw)
          : actors[actor].trigger_source[turn];
      const auto raw_state = tiles.find(tile);
      const bool legal_ongoing_harvest = raw.op == Op::Harvest &&
          raw_state != tiles.end() && raw_state->second.kind == TileKind::Crop &&
          (raw_state->second.item == 2 || raw_state->second.item == 3) &&
          raw_state->second.harvest_legal;
      if (trigger && !(actors[actor].trigger_source.empty() &&
                       legal_ongoing_harvest)) {
        auto intent = std::find_if(
            result.active_intents.begin(), result.active_intents.end(),
            [&](const auto& candidate) { return candidate.tile == tile; });
        int desired = raw.item;
        const auto state = tiles.find(tile);
        if (desired < 0 && intent != result.active_intents.end())
          desired = intent->desired_crop;
        if (desired < 0 && state != tiles.end() &&
            state->second.kind == TileKind::Crop)
          desired = state->second.item;
        if (desired < 0) {
          ++result.rejected[ExactSlotReject::MissingDesiredCrop];
        } else if (intent == result.active_intents.end()) {
          result.active_intents.push_back(
              {next_id++, tile, desired,
               raw.op == Op::Harvest && state != tiles.end() &&
                   state->second.kind == TileKind::Crop &&
                   state->second.harvest_legal,
               day});
          intent_ids.insert(result.active_intents.back().id);
          ++result.absorbed_raw_intents;
        } else {
          intent->desired_crop = desired;
          if (raw.op == Op::Harvest && state != tiles.end() &&
              state->second.kind == TileKind::Crop &&
              state->second.harvest_legal)
            intent->harvest_before_replant = true;
        }
      }
    }
    for (const auto& intent : result.active_intents)
      intent_records[intent.id] = intent;

    std::map<Position, int> contenders;
    std::set<Position> sealed_tiles;
    for (std::size_t actor = 0; actor < actors.size(); ++actor) {
      const auto& raw = actors[actor].raw[turn];
      const auto tile = result.positions_before[actor][turn];
      if (!actors[actor].service_slot[turn]) {
        if (raw.op != Op::Pass && raw.op != Op::Move)
          sealed_tiles.insert(tile);
        continue;
      }
      if (raw.op == Op::Move)
        continue;
      if (std::any_of(result.active_intents.begin(),
                      result.active_intents.end(), [&](const auto& intent) {
                        return intent.tile == tile;
                      }))
        ++contenders[tile];
    }

    for (std::size_t actor = 0; actor < actors.size(); ++actor) {
      const auto& raw = actors[actor].raw[turn];
      const auto tile = result.positions_before[actor][turn];
      if (raw.op == Op::Move) continue;
      if (!actors[actor].service_slot[turn]) {
        if (raw.op != Op::Pass && raw.op != Op::Move)
          tiles[tile] = {TileKind::Other, -1, false, false};
        continue;
      }
      auto intent = std::find_if(
          result.active_intents.begin(), result.active_intents.end(),
          [&](const auto& candidate) { return candidate.tile == tile; });
      if (intent == result.active_intents.end()) continue;
      if (contenders[tile] != 1 || sealed_tiles.contains(tile)) {
        ++result.rejected[ExactSlotReject::TileSerialized];
        continue;
      }
      auto& state = tiles[tile];
      Action chosen;
      if (state.kind == TileKind::Weed) {
        chosen.op = Op::Dig;
        state = {TileKind::Empty, -1, false, false};
      } else if (state.kind == TileKind::Empty) {
        if (intent->desired_crop < 0) {
          ++result.rejected[ExactSlotReject::MissingDesiredCrop];
          continue;
        }
        if (seeds[intent->desired_crop] <= 0) {
          ++result.rejected[ExactSlotReject::SeedUnavailable];
          continue;
        }
        --seeds[intent->desired_crop];
        chosen = {Op::Plant, intent->desired_crop, 1};
        state = {TileKind::Crop, intent->desired_crop, false, false};
      } else if (state.kind == TileKind::Crop) {
        if (state.item != intent->desired_crop) {
          chosen.op = Op::Dig;
          state = {TileKind::Empty, -1, false, false};
        } else if (intent->harvest_before_replant) {
          if (!state.harvest_legal) {
            ++result.rejected[ExactSlotReject::MaturityWait];
            continue;
          }
          if (state.item == 2 || state.item == 3) {
            ++result.rejected[ExactSlotReject::OngoingHarvestUnsupported];
            continue;
          }
          chosen = {Op::Harvest, intent->desired_crop, 1};
          state = {TileKind::Empty, -1, false, false};
          intent->harvest_before_replant = false;
        } else if (!state.watered_today) {
          chosen = {Op::Water, intent->desired_crop, 1};
          state.watered_today = true;
        } else {
          result.completed_intents.push_back(intent->id);
          // The only completed-slot replacement is a proven state no-op:
          // desired crop already watered and no pending harvest transition.
          result.manifest[actor][turn] = {};
          if (raw.op != Op::Pass) {
            ++result.state_equivalent_changes;
            ++result.certified_service_changes;
            result.affected_tiles.insert(tile);
            result.affected_actors.insert(actors[actor].actor);
            ++result.raw_nonmove_changes[raw.op];
          }
          continue;
        }
      } else {
        ++result.rejected[ExactSlotReject::TileUnsupported];
        continue;
      }
      result.manifest[actor][turn] = chosen;
      ++result.assignments;
      ++result.certified_service_changes;
      if (raw == chosen) ++result.state_equivalent_changes;
      result.affected_tiles.insert(tile);
      result.affected_actors.insert(actors[actor].actor);
      if (raw.op != Op::Move && !(raw == chosen))
        ++result.raw_nonmove_changes[raw.op];
      if (state.kind == TileKind::Crop &&
          state.item == intent->desired_crop && state.watered_today &&
          !intent->harvest_before_replant)
        result.completed_intents.push_back(intent->id);
    }
    if (!result.completed_intents.empty()) {
      std::sort(result.completed_intents.begin(),
                result.completed_intents.end());
      result.completed_intents.erase(
          std::unique(result.completed_intents.begin(),
                      result.completed_intents.end()),
          result.completed_intents.end());
      std::erase_if(result.active_intents, [&](const auto& intent) {
        return std::binary_search(result.completed_intents.begin(),
                                  result.completed_intents.end(), intent.id);
      });
    }
  }

  result.move_slots_exact = true;
  result.move_positions_exact = true;
  for (std::size_t actor = 0; actor < actors.size(); ++actor)
    for (std::size_t turn = 0; turn < turns; ++turn) {
      const bool raw_move = actors[actor].raw[turn].op == Op::Move;
      const bool final_move = result.manifest[actor][turn].op == Op::Move;
      if (raw_move != final_move) ++result.original_move_slot_changes;
      if (raw_move && final_move &&
          !(actors[actor].raw[turn] == result.manifest[actor][turn]))
        ++result.original_move_payload_changes;
      if ((raw_move || final_move) &&
          !(actors[actor].raw[turn] == result.manifest[actor][turn]))
        result.move_slots_exact = false;
    }
  for (std::size_t actor = 0; actor < actors.size(); ++actor) {
    auto final_position = actors[actor].start;
    for (std::size_t turn = 0; turn < turns; ++turn) {
      if (!(final_position == result.positions_before[actor][turn]))
        result.move_positions_exact = false;
      final_position = after_move(final_position, result.manifest[actor][turn]);
    }
  }
  std::map<int, int> final_plant_demand;
  for (const auto& manifest : result.manifest)
    for (const auto& action : manifest)
      if (action.op == Op::Plant && action.item >= 0)
        ++final_plant_demand[action.item];
  bool seed_safe = true;
  for (const auto& [item, demand] : final_plant_demand) {
    const auto found = initial_seeds.find(item);
    seed_safe = seed_safe && found != initial_seeds.end() &&
        demand <= found->second;
  }
  if (!seed_safe) {
    for (const auto id : result.completed_intents) {
      // Completion records are reconstructible from the input intent set only
      // when they predated this horizon. A newly absorbed-and-completed intent
      // is re-created from the source scan on the next call, so it is not
      // silently reported complete here.
      const auto old = intent_records.find(id);
      if (old != intent_records.end()) result.active_intents.push_back(old->second);
    }
    std::sort(result.active_intents.begin(), result.active_intents.end(),
              [](const auto& left, const auto& right) {
                return left.id < right.id;
              });
    result.active_intents.erase(
        std::unique(result.active_intents.begin(), result.active_intents.end(),
                    [](const auto& left, const auto& right) {
                      return left.id == right.id;
                    }),
        result.active_intents.end());
    for (std::size_t actor = 0; actor < actors.size(); ++actor)
      result.manifest[actor] = actors[actor].raw;
    result.assignments = 0;
    result.state_equivalent_changes = 0;
    result.certified_service_changes = 0;
    result.affected_tiles.clear();
    result.affected_actors.clear();
    result.raw_nonmove_changes.clear();
    result.completed_intents.clear();
    ++result.rejected[ExactSlotReject::GlobalSeedManifestUnsafe];
    tiles = initial_tiles;
    seeds = initial_seeds;
  }
  result.final_tiles = std::move(tiles);
  result.remaining_seeds = std::move(seeds);
  return result;
}

}  // namespace g001::day_horizon_repair
