#include "../include/component_scoped_route_planner.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace g001::day_horizon_repair {
namespace {

using event_local_repair::Op;

Position after_move(Position position, const Action& action) {
  if (action.op != Op::Move) return position;
  switch (action.arg0) {
    case 0: --position.row; break;
    case 1: ++position.row; break;
    case 2: --position.column; break;
    case 3: ++position.column; break;
    default: throw std::invalid_argument("unknown MOVE direction");
  }
  return position;
}

bool tile_effect(const Action& action) {
  return action.op != Op::Pass && action.op != Op::Move;
}

bool in_bounds(Position position, const BoardLegalityCertificate& board) {
  return position.row >= 0 && position.column >= 0 &&
      position.row < board.rows && position.column < board.columns;
}

bool plausible_raw_transition(
    const Action& action, const RawTileTransitionCertificate& certificate) {
  using event_local_repair::TileKind;
  switch (action.op) {
    case Op::Dig:
      return certificate.before.kind == TileKind::Weed &&
          certificate.after.kind == TileKind::Empty;
    case Op::Plant:
      return certificate.before.kind == TileKind::Empty &&
          certificate.after.kind == TileKind::Crop &&
          certificate.after.item == action.item;
    case Op::Build:
      return certificate.before.kind == TileKind::Empty &&
          certificate.after.kind == TileKind::Structure;
    case Op::Water:
      return certificate.before.kind == TileKind::Crop &&
          certificate.after.kind == TileKind::Crop &&
          certificate.before.item == certificate.after.item &&
          certificate.after.watered_today;
    case Op::Harvest:
      return certificate.before.kind == TileKind::Crop &&
          certificate.before.harvest_legal &&
          certificate.after.kind == TileKind::Empty;
    default: return false;
  }
}

bool apply_objective_transition(
    const Action& action, event_local_repair::TileObservation& state) {
  using event_local_repair::TileKind;
  switch (action.op) {
    case Op::Dig:
      if (state.kind != TileKind::Weed) return false;
      state = {TileKind::Empty, -1, false, false};
      return true;
    case Op::Plant:
      if (state.kind != TileKind::Empty || action.item < 0) return false;
      state = {TileKind::Crop, action.item, false, false};
      return true;
    case Op::Build:
      if (state.kind != TileKind::Empty) return false;
      state = {TileKind::Structure, action.item, false, false};
      return true;
    case Op::Water:
      if (state.kind != TileKind::Crop || state.watered_today ||
          (action.item >= 0 && state.item != action.item))
        return false;
      state.watered_today = true;
      return true;
    case Op::Harvest:
      if (state.kind != TileKind::Crop || !state.harvest_legal ||
          (action.item >= 0 && state.item != action.item))
        return false;
      state = {TileKind::Empty, -1, false, false};
      return true;
    default: return false;
  }
}

std::set<int> planted_items(const Objective& objective) {
  std::set<int> result;
  for (const auto& action : objective.remaining_transitions)
    if (action.op == Op::Plant && action.item >= 0) result.insert(action.item);
  return result;
}

}  // namespace

ComponentScopedResult compile_component_scoped_route(
    const std::vector<ComponentActorPlan>& input,
    std::vector<Objective> objectives, const ResourceSnapshot& resources,
    TimeExpandedConfig config) {
  if (input.empty()) throw std::invalid_argument("empty component horizon");
  const int turns = input.front().turns;
  if (turns <= 0) throw std::invalid_argument("empty component horizon");

  std::set<int> actor_ids;
  std::set<std::uint64_t> objective_ids;
  for (const auto& objective : objectives)
    if (objective.id == 0 || !objective_ids.insert(objective.id).second)
      throw std::invalid_argument("objective ids must be nonzero and unique");

  std::vector<ActorPlan> actors;
  std::vector<std::vector<std::uint64_t>> source_slots;
  std::vector<bool> actor_unsupported;
  std::vector<bool> actor_raw_tile_effect;
  std::vector<std::set<int>> actor_resource_items;
  std::vector<std::vector<std::optional<RawTileTransitionCertificate>>>
      raw_tile_certificates;
  bool global_opaque = false;
  bool route_geometry_safe = true;
  std::map<int, int> certified_resource_consumption;
  std::vector<std::set<Position>> reachable;
  actors.reserve(input.size());
  for (const auto& actor : input) {
    if (actor.actor < 0 || actor.turns != turns ||
        !actor_ids.insert(actor.actor).second)
      throw std::invalid_argument("inconsistent component actor horizon");
    ActorPlan plan{actor.actor, actor.start,
                   std::vector<Action>(static_cast<std::size_t>(turns)),
                   std::vector<bool>(static_cast<std::size_t>(turns), false)};
    std::vector<std::uint64_t> slots(static_cast<std::size_t>(turns));
    std::vector<std::optional<RawTileTransitionCertificate>> tile_certificates(
        static_cast<std::size_t>(turns));
    std::set<std::uint64_t> source_ids;
    int next_slot = 0;
    bool unsupported = false;
    bool has_raw_tile_effect = false;
    std::set<int> resource_items;
    std::set<Position> actor_reachable{actor.start};
    Position position = actor.start;
    for (const auto& source : actor.ordered_raw) {
      if (source.source_id == 0 || !source_ids.insert(source.source_id).second ||
          source.action.op == Op::Pass || source.earliest_turn < 0)
        throw std::invalid_argument("invalid ordered raw source");
      const int slot = std::max(next_slot, source.earliest_turn);
      if (slot >= turns)
        throw std::runtime_error("ordered raw source exceeds horizon");
      plan.raw[static_cast<std::size_t>(slot)] = source.action;
      slots[static_cast<std::size_t>(slot)] = source.source_id;
      tile_certificates[static_cast<std::size_t>(slot)] =
          source.tile_transition_certificate;
      next_slot = slot + 1;
      unsupported = unsupported || source.unsupported;
      if (source.effect_certificate.has_value()) {
        const auto& certificate = *source.effect_certificate;
        if (!source.unsupported || !certificate.no_unlisted_global_effects)
          throw std::invalid_argument("incomplete unsupported effect certificate");
        for (const auto& [item, quantity] : certificate.resource_consumption) {
          if (item < 0 || quantity < 0)
            throw std::invalid_argument("invalid certified resource consumption");
          certified_resource_consumption[item] += quantity;
          resource_items.insert(item);
        }
        for (const auto& [item, quantity] : certificate.resource_production) {
          if (item < 0 || quantity < 0)
            throw std::invalid_argument("invalid certified resource production");
          resource_items.insert(item);
        }
        for (const int item : certificate.resource_reads) {
          if (item < 0)
            throw std::invalid_argument("invalid certified resource read");
          resource_items.insert(item);
        }
      } else if (source.unsupported && source.action.op == Op::Other) {
        global_opaque = true;
      }
      if (source.action.op == Op::Move) {
        const Position destination = after_move(position, source.action);
        if (!actor.board.has_value() || !actor.board->complete ||
            actor.board->rows <= 0 || actor.board->columns <= 0 ||
            !in_bounds(position, *actor.board) ||
            !in_bounds(destination, *actor.board)) {
          route_geometry_safe = false;
        } else {
          const auto transition =
              actor.board->move_transitions.find(source.source_id);
          if (transition == actor.board->move_transitions.end() ||
              transition->second.first != position ||
              transition->second.second != destination ||
              actor.board->blocked_edges.contains({position, destination}))
            route_geometry_safe = false;
        }
        position = destination;
      }
      has_raw_tile_effect = has_raw_tile_effect || tile_effect(source.action);
      if (source.action.op == Op::Plant && source.action.item >= 0)
        resource_items.insert(source.action.item);
      actor_reachable.insert(position);
    }
    actors.push_back(std::move(plan));
    source_slots.push_back(std::move(slots));
    actor_unsupported.push_back(unsupported);
    actor_raw_tile_effect.push_back(has_raw_tile_effect);
    actor_resource_items.push_back(std::move(resource_items));
    raw_tile_certificates.push_back(std::move(tile_certificates));
    reachable.push_back(std::move(actor_reachable));
  }

  // DSU nodes: actors first, objectives second. Conservative shared
  // reachability connects actors because either may lease the same plot and
  // their effects cannot be proven independent.
  const std::size_t node_count = actors.size() + objectives.size();
  std::vector<std::size_t> parent(node_count);
  for (std::size_t node = 0; node < node_count; ++node) parent[node] = node;
  std::function<std::size_t(std::size_t)> root = [&](std::size_t node) {
    return parent[node] == node ? node : parent[node] = root(parent[node]);
  };
  auto unite = [&](std::size_t left, std::size_t right) {
    left = root(left);
    right = root(right);
    if (left != right) parent[right] = left;
  };
  for (std::size_t left = 0; left < actors.size(); ++left)
    for (std::size_t right = left + 1; right < actors.size(); ++right) {
      bool intersects = false;
      for (const auto& tile : reachable[left])
        intersects = intersects || reachable[right].contains(tile);
      bool shared_resource = false;
      for (const int item : actor_resource_items[left])
        shared_resource = shared_resource ||
            actor_resource_items[right].contains(item);
      if (intersects || shared_resource) unite(left, right);
    }
  for (std::size_t objective = 0; objective < objectives.size(); ++objective)
    for (std::size_t actor = 0; actor < actors.size(); ++actor)
      if (reachable[actor].contains(objectives[objective].tile))
        unite(actor, actors.size() + objective);
      else {
        const auto items = planted_items(objectives[objective]);
        bool resource_conflict = false;
        for (const int item : items)
          resource_conflict = resource_conflict ||
              actor_resource_items[actor].contains(item);
        if (resource_conflict) unite(actor, actors.size() + objective);
      }
  for (std::size_t left = 0; left < objectives.size(); ++left)
    for (std::size_t right = left + 1; right < objectives.size(); ++right) {
      const auto left_items = planted_items(objectives[left]);
      const auto right_items = planted_items(objectives[right]);
      bool shared_seed = false;
      for (const int item : left_items)
        shared_seed = shared_seed || right_items.contains(item);
      if (objectives[left].tile == objectives[right].tile || shared_seed)
        unite(actors.size() + left, actors.size() + right);
    }

  std::map<std::size_t, std::vector<std::size_t>> component_actors;
  std::map<std::size_t, std::vector<std::size_t>> component_objectives;
  for (std::size_t actor = 0; actor < actors.size(); ++actor)
    component_actors[root(actor)].push_back(actor);
  for (std::size_t objective = 0; objective < objectives.size(); ++objective)
    component_objectives[root(actors.size() + objective)].push_back(objective);
  std::set<std::size_t> component_roots;
  for (const auto& [component, unused] : component_actors) {
    (void)unused;
    component_roots.insert(component);
  }
  for (const auto& [component, unused] : component_objectives) {
    (void)unused;
    component_roots.insert(component);
  }

  ComponentScopedResult result;
  result.conflict_components = static_cast<int>(component_roots.size());
  result.manifest.reserve(actors.size());
  result.positions_before.resize(actors.size());
  result.raw_source_manifest = source_slots;
  for (const auto& actor : actors) result.manifest.push_back(actor.raw);
  std::set<std::pair<int, int>> repair_slots;
  std::map<int, int> sealed_resource_objective_demand;

  for (const auto component : component_roots) {
    const auto& actor_indices = component_actors[component];
    const auto& objective_indices = component_objectives[component];
    if (actor_indices.empty()) {
      for (const auto objective : objective_indices)
        result.unscheduled_objectives.push_back(objectives[objective].id);
      continue;
    }
    bool sealed = global_opaque;
    for (const auto actor : actor_indices)
      sealed = sealed || actor_unsupported[actor];
    if (sealed) {
      ++result.sealed_components;
      for (const auto objective : objective_indices) {
        result.unscheduled_objectives.push_back(objectives[objective].id);
        bool shares_raw_resource = false;
        for (const auto actor : actor_indices)
          for (const auto& action :
               objectives[objective].remaining_transitions)
            shares_raw_resource = shares_raw_resource ||
                (action.op == Op::Plant &&
                 actor_resource_items[actor].contains(action.item));
        if (shares_raw_resource)
          for (const auto& action :
               objectives[objective].remaining_transitions)
            if (action.op == Op::Plant)
              sealed_resource_objective_demand[action.item] += action.quantity;
      }
      continue;
    }
    bool has_raw_tile_effect = false;
    for (const auto actor : actor_indices)
      has_raw_tile_effect = has_raw_tile_effect || actor_raw_tile_effect[actor];
    if (has_raw_tile_effect) {
      // Raw production changes tile state. Without a full state transition
      // proof, mere tile/time serialization cannot make a later objective
      // legal. Keep the raw schedule fixed and admit only a fully certified,
      // consecutive suffix after the last raw effect on that tile.
      for (const auto objective_index : objective_indices) {
        const auto& objective = objectives[objective_index];
        bool scheduled = false;
        for (const auto actor_index : actor_indices) {
          Position position = actors[actor_index].start;
          int last_effect_turn = -1;
          std::optional<event_local_repair::TileObservation> state;
          bool certificate_chain = true;
          std::vector<Position> timeline;
          timeline.reserve(static_cast<std::size_t>(turns));
          for (int turn = 0; turn < turns; ++turn) {
            timeline.push_back(position);
            const auto& raw = actors[actor_index].raw[
                static_cast<std::size_t>(turn)];
            if (position == objective.tile && tile_effect(raw)) {
              last_effect_turn = turn;
              const auto& certificate = raw_tile_certificates[actor_index][
                  static_cast<std::size_t>(turn)];
              if (!certificate.has_value() ||
                  !plausible_raw_transition(raw, *certificate) ||
                  (state.has_value() && *state != certificate->before)) {
                certificate_chain = false;
              } else {
                state = certificate->after;
              }
            }
            position = after_move(position, raw);
          }
          if (last_effect_turn < 0 || !certificate_chain ||
              !state.has_value())
            continue;
          auto final_state = *state;
          for (const auto& transition : objective.remaining_transitions)
            certificate_chain = certificate_chain &&
                apply_objective_transition(transition, final_state);
          if (!certificate_chain) continue;
          const int first_turn = last_effect_turn + 1;
          const int final_turn = first_turn + static_cast<int>(
              objective.remaining_transitions.size()) - 1;
          if (objective.remaining_transitions.empty() ||
              final_turn >= turns || final_turn > objective.deadline_turn)
            continue;
          bool capacity = true;
          for (int turn = first_turn; turn <= final_turn; ++turn)
            capacity = capacity &&
                timeline[static_cast<std::size_t>(turn)] == objective.tile &&
                actors[actor_index].raw[static_cast<std::size_t>(turn)].op ==
                    Op::Pass;
          if (!capacity) continue;
          for (std::size_t transition = 0;
               transition < objective.remaining_transitions.size();
               ++transition) {
            const int turn = first_turn + static_cast<int>(transition);
            result.manifest[actor_index][static_cast<std::size_t>(turn)] =
                objective.remaining_transitions[transition];
            result.assignments.push_back(
                {objective.id, static_cast<int>(transition),
                 actors[actor_index].actor, turn,
                 objective.remaining_transitions[transition]});
            repair_slots.insert({actors[actor_index].actor, turn});
          }
          scheduled = true;
          break;
        }
        if (!scheduled)
          result.unscheduled_objectives.push_back(objective.id);
      }
      ++result.committed_components;
      continue;
    }
    std::vector<ActorPlan> local_actors;
    std::vector<Objective> local_objectives;
    for (const auto actor : actor_indices) local_actors.push_back(actors[actor]);
    for (const auto objective : objective_indices)
      local_objectives.push_back(objectives[objective]);
    if (local_objectives.empty()) {
      ++result.committed_components;
      continue;
    }
    const auto local = compile_time_expanded(
        local_actors, local_objectives, resources, {}, config);
    if (!local.ordered_move_sequences_exact ||
        local.terminal_unexecuted_raw_actions != 0) {
      for (const auto objective : objective_indices)
        result.unscheduled_objectives.push_back(objectives[objective].id);
      continue;
    }
    ++result.committed_components;
    for (std::size_t local_actor = 0; local_actor < actor_indices.size();
         ++local_actor)
      result.manifest[actor_indices[local_actor]] = local.manifest[local_actor];
    for (const auto& assignment : local.assignments) {
      result.assignments.push_back(assignment);
      repair_slots.insert({assignment.actor, assignment.turn});
    }
    result.unscheduled_objectives.insert(result.unscheduled_objectives.end(),
                                         local.unscheduled_objectives.begin(),
                                         local.unscheduled_objectives.end());
  }

  result.raw_source_manifest.assign(
      actors.size(), std::vector<std::uint64_t>(static_cast<std::size_t>(turns)));
  result.raw_source_order_exact = true;
  result.global_tile_serialization = true;
  result.global_resource_safe = true;
  result.global_route_geometry_safe = route_geometry_safe;
  std::map<int, int> plant_demand;
  for (const auto& [item, quantity] : certified_resource_consumption)
    plant_demand[item] += quantity;
  std::set<std::tuple<int, int, int>> occupied_effects;
  for (std::size_t actor = 0; actor < actors.size(); ++actor) {
    Position position = actors[actor].start;
    result.positions_before[actor].reserve(static_cast<std::size_t>(turns));
    std::size_t source_cursor = 0;
    for (int turn = 0; turn < turns; ++turn) {
      result.positions_before[actor].push_back(position);
      const auto& action = result.manifest[actor][static_cast<std::size_t>(turn)];
      const bool repair = repair_slots.contains({actors[actor].actor, turn});
      if (action.op != Op::Pass && !repair) {
        if (source_cursor >= input[actor].ordered_raw.size() ||
            !(action == input[actor].ordered_raw[source_cursor].action)) {
          result.raw_source_order_exact = false;
        } else {
          result.raw_source_manifest[actor][static_cast<std::size_t>(turn)] =
              input[actor].ordered_raw[source_cursor].source_id;
          ++source_cursor;
        }
      }
      if (tile_effect(action) &&
          !occupied_effects.insert(
              {position.row, position.column, turn}).second)
        result.global_tile_serialization = false;
      if (action.op == Op::Plant && action.item >= 0 && action.quantity > 0)
        plant_demand[action.item] += action.quantity;
      position = after_move(position, action);
    }
    if (source_cursor != input[actor].ordered_raw.size())
      result.raw_source_order_exact = false;
  }
  for (const auto& [item, demand] : plant_demand) {
    const auto found = resources.seeds.find(item);
    const int available = found == resources.seeds.end() ? 0 : found->second;
    const int blocked_candidate = sealed_resource_objective_demand[item];
    if (demand > available ||
        (blocked_candidate > 0 && demand + blocked_candidate > available))
      result.global_resource_safe = false;
  }
  for (const auto& [item, blocked_candidate] :
       sealed_resource_objective_demand)
    if (!plant_demand.contains(item)) {
      const auto found = resources.seeds.find(item);
      const int available = found == resources.seeds.end() ? 0 : found->second;
      if (blocked_candidate > available) result.global_resource_safe = false;
    }
  result.ordered_move_sequences_exact = result.raw_source_order_exact;
  result.terminal_unexecuted_raw_actions = result.raw_source_order_exact ? 0 : 1;
  result.terminal_unexecuted_moves = result.terminal_unexecuted_raw_actions;
  std::map<std::uint64_t, int> assignment_counts;
  for (const auto& assignment : result.assignments)
    ++assignment_counts[assignment.objective_id];
  for (const auto& objective : objectives)
    if (assignment_counts[objective.id] ==
        static_cast<int>(objective.remaining_transitions.size()))
      result.committed_objectives.push_back(objective.id);
    else
      result.unscheduled_objectives.push_back(objective.id);
  std::sort(result.committed_objectives.begin(),
            result.committed_objectives.end());
  result.committed_objectives.erase(
      std::unique(result.committed_objectives.begin(),
                  result.committed_objectives.end()),
      result.committed_objectives.end());
  std::sort(result.unscheduled_objectives.begin(),
            result.unscheduled_objectives.end());
  result.unscheduled_objectives.erase(
      std::unique(result.unscheduled_objectives.begin(),
                  result.unscheduled_objectives.end()),
      result.unscheduled_objectives.end());
  std::set<std::uint64_t> objective_partition(
      result.committed_objectives.begin(), result.committed_objectives.end());
  bool disjoint_partition = true;
  for (const auto id : result.unscheduled_objectives)
    disjoint_partition = objective_partition.insert(id).second &&
        disjoint_partition;
  result.objective_conservation = disjoint_partition &&
      objective_partition == objective_ids;
  result.merge_safe = result.raw_source_order_exact &&
      result.global_tile_serialization && result.global_resource_safe &&
      result.global_route_geometry_safe && result.objective_conservation;
  if (!result.merge_safe) {
    // A rejected merge exposes only the ordered raw proposal. No repair byte
    // or assignment may escape a failed global gate.
    result.manifest.clear();
    result.positions_before.assign(actors.size(), {});
    result.raw_source_manifest = source_slots;
    for (const auto& actor : actors) result.manifest.push_back(actor.raw);
    for (std::size_t actor = 0; actor < actors.size(); ++actor) {
      Position position = actors[actor].start;
      for (const auto& action : actors[actor].raw) {
        result.positions_before[actor].push_back(position);
        position = after_move(position, action);
      }
    }
    result.assignments.clear();
    result.committed_objectives.clear();
    result.unscheduled_objectives.clear();
    for (const auto& objective : objectives)
      result.unscheduled_objectives.push_back(objective.id);
    std::sort(result.unscheduled_objectives.begin(),
              result.unscheduled_objectives.end());
    result.objective_conservation =
        std::set<std::uint64_t>(result.unscheduled_objectives.begin(),
                                result.unscheduled_objectives.end()) ==
        objective_ids;
    result.committed_components = 0;
  }
  return result;
}

}  // namespace g001::day_horizon_repair
