#include "../include/production_obligation_day_scheduler.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <tuple>

namespace g001::obligation_day {
namespace {
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;

constexpr int kTurns = 24;

bool action_equal(Action lhs, Action rhs) {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool position_equal(Position lhs, Position rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
         op == Op::WEST;
}

void hash_value(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (8 * byte)) & 255U;
    hash *= 1099511628211ULL;
  }
}

void hash_action(std::uint64_t& hash, Action action) {
  hash_value(hash, static_cast<std::uint8_t>(action.op));
  hash_value(hash, static_cast<std::uint8_t>(action.item));
  hash_value(hash, static_cast<std::uint32_t>(action.quantity));
}

void hash_position(std::uint64_t& hash, Position position) {
  hash_value(hash, static_cast<std::uint16_t>(position.x));
  hash_value(hash, static_cast<std::uint16_t>(position.y));
}

Position actor_position(const Simulator& simulator, int player, int actor) {
  const auto& farm = simulator.farms()[player];
  return actor == 0 ? farm.farmer : farm.hands[actor - 1];
}

bool valid_position(const Simulator& simulator, Position position) {
  return position.x >= 0 && position.y >= 0 &&
         position.x < simulator.config().board_size &&
         position.y < simulator.config().board_size;
}

const Tile& tile_at(const Simulator& simulator, int player, Position position) {
  return simulator.farms()[player].tiles
      [position.y * simulator.config().board_size + position.x];
}

int inventory_count(const Simulator& simulator, int player, int actor,
                    Item item) {
  const auto& private_state = simulator.privates()[player];
  const int item_index = static_cast<int>(item);
  if (actor < 0 || actor >= static_cast<int>(private_state.inventories.size())) {
    return 0;
  }
  if (item_index < 0 || item_index >= fastkag::N_ITEMS) return 0;
  return private_state.inventories[actor][item_index];
}

bool valid_obligation_item(const ProductionObligation& obligation) {
  const int item = static_cast<int>(obligation.item);
  const bool product = item >= 0 && item < fastkag::N_PRODUCTS;
  const bool crop = item >= 0 && item < fastkag::N_CROPS;
  const bool animal = item >= static_cast<int>(Item::GOOSE) &&
      item <= static_cast<int>(Item::SHEEP);
  switch (obligation.goal) {
    case GoalKind::CropReady:
      return crop;
    case GoalKind::Pickup:
      return item >= 0 && item < fastkag::N_ITEMS;
    case GoalKind::Place:
      return animal;
    case GoalKind::Feed:
      return obligation.item == Item::WHEAT;
    case GoalKind::Care:
      return animal;
    case GoalKind::Harvest:
      return product || animal;
    case GoalKind::BuildPasture:
    case GoalKind::BuildCoop:
      return obligation.item == Item::NONE;
    case GoalKind::CollectFertilizer:
      return obligation.item == Item::NONE ||
          obligation.item == Item::FERTILIZER;
  }
  return false;
}

std::uint64_t request_input_hash(const DayPlanRequest& request) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_value(hash, request.player);
  hash_value(hash, request.issuer_generation);
  hash_value(hash, request.start_tick);
  hash_value(hash, request.moves.size());
  for (const auto& move : request.moves) {
    hash_value(hash, move.actor);
    hash_value(hash, move.source_step);
    hash_action(hash, move.action);
  }
  hash_value(hash, request.obligations.size());
  for (const auto& obligation : request.obligations) {
    hash_value(hash, obligation.id);
    hash_value(hash, obligation.actor);
    hash_position(hash, obligation.tile);
    hash_value(hash, static_cast<int>(obligation.goal));
    hash_value(hash, static_cast<int>(obligation.item));
    hash_value(hash, obligation.quantity);
    hash_value(hash, obligation.dependencies.size());
    for (auto dependency : obligation.dependencies) {
      hash_value(hash, dependency);
    }
    hash_value(hash, static_cast<int>(obligation.resource.item));
    hash_value(hash, obligation.resource.seed_quantity);
    hash_value(hash, obligation.resource.shed_quantity);
    hash_value(hash, obligation.resource.carried_quantity);
    hash_value(hash, obligation.earliest_step);
    hash_value(hash, obligation.deadline_step);
    hash_value(hash, obligation.priority);
    hash_value(hash, obligation.must_finish_today);
    hash_value(hash, obligation.repair);
    hash_value(hash, obligation.source_step);
    hash_value(hash, obligation.policy_deferred);
  }
  return hash;
}

bool crop_done(const Tile& tile, Item item) {
  return tile.kind == TileKind::PLANT && tile.crop == item &&
         tile.watered_today;
}

int transitions_remaining(const ProductionObligation& obligation,
                          const Simulator& state, int player) {
  if (obligation.goal == GoalKind::BuildPasture ||
      obligation.goal == GoalKind::BuildCoop) {
    const auto& tile = tile_at(state, player, obligation.tile);
    const auto wanted = obligation.goal == GoalKind::BuildCoop
                            ? TileKind::COOP
                            : TileKind::PASTURE;
    if (tile.kind == wanted) return 0;
    return tile.kind == TileKind::EMPTY ? 1 : 2;
  }
  if (obligation.goal != GoalKind::CropReady) {
    return 1;
  }
  const auto& tile = tile_at(state, player, obligation.tile);
  if (tile.kind == TileKind::WEED) {
    return 3;
  }
  if (tile.kind == TileKind::EMPTY) {
    return 2;
  }
  if (tile.kind != TileKind::PLANT || tile.crop != obligation.item) {
    return 3;
  }
  return tile.watered_today ? 0 : 1;
}

struct CompiledAction {
  std::optional<Action> action;
  bool blocked_on_receipt{};
  bool unsupported{};
};

CompiledAction compile_obligation(
    const ProductionObligation& obligation, const Simulator& state, int player,
    int actor, const std::array<int, fastkag::N_CROPS>& reserved_seeds,
    const std::array<int, fastkag::N_ITEMS>& reserved_shed) {
  const auto& tile = tile_at(state, player, obligation.tile);
  const auto& private_state = state.privates()[player];
  if (obligation.goal == GoalKind::CropReady) {
    const int item = static_cast<int>(obligation.item);
    if (item < 0 || item >= fastkag::N_CROPS) {
      return {{}, false, true};
    }
    if (tile.kind == TileKind::WEED) {
      return {Action{Op::DIG, Item::NONE, 1}};
    }
    if (tile.kind == TileKind::EMPTY) {
      if (private_state.seeds[item] - reserved_seeds[item] <= 0) {
        return {{}, true, false};
      }
      return {Action{Op::PLANT, obligation.item, 1}};
    }
    if (tile.kind == TileKind::PLANT && tile.crop == obligation.item) {
      return tile.watered_today
                 ? CompiledAction{}
                 : CompiledAction{Action{Op::WATER, obligation.item, 1}};
    }
    if (tile.kind == TileKind::ANIMAL || tile.kind == TileKind::LOCKED) {
      return {{}, false, true};
    }
    return {Action{Op::DIG, Item::NONE, 1}};
  }
  if (obligation.goal == GoalKind::Pickup) {
    const int item = static_cast<int>(obligation.item);
    if (item < 0 || item >= fastkag::N_ITEMS) {
      return {{}, false, true};
    }
    if (private_state.shed[item] - reserved_shed[item] < obligation.quantity) {
      return {{}, true, false};
    }
    return {Action{Op::PICKUP, obligation.item, obligation.quantity}};
  }
  if (obligation.goal == GoalKind::Place) {
    if (inventory_count(state, player, actor, obligation.item) <
        obligation.quantity) {
      return {{}, true, false};
    }
    return {Action{Op::PLACE, obligation.item, obligation.quantity}};
  }
  if (obligation.goal == GoalKind::Feed) {
    if (inventory_count(state, player, actor, Item::WHEAT) < 1) {
      return {{}, true, false};
    }
    return {Action{Op::FEED, Item::WHEAT, 1}};
  }
  if (obligation.goal == GoalKind::Care) {
    return {Action{Op::CARE, obligation.item, 1}};
  }
  if (obligation.goal == GoalKind::CollectFertilizer) {
    return {Action{Op::COLLECT_FERTILIZER, Item::NONE, 1}};
  }
  if (obligation.goal == GoalKind::Harvest) {
    return {Action{Op::HARVEST, obligation.item,
                   std::max(1, obligation.quantity)}};
  }
  if (obligation.goal == GoalKind::BuildPasture ||
      obligation.goal == GoalKind::BuildCoop) {
    const auto wanted = obligation.goal == GoalKind::BuildCoop
                            ? TileKind::COOP
                            : TileKind::PASTURE;
    if (tile.kind == wanted) return {};
    if (tile.kind == TileKind::WEED || tile.kind == TileKind::PLANT ||
        tile.kind == TileKind::COOP || tile.kind == TileKind::PASTURE)
      return {Action{Op::DIG, Item::NONE, 1}};
    if (tile.kind == TileKind::EMPTY)
      return {Action{obligation.goal == GoalKind::BuildCoop
                         ? Op::BUILD_COOP
                         : Op::BUILD_PASTURE,
                     Item::NONE, 1}};
    return {{}, false, true};
  }
  return {{}, false, true};
}

bool action_progressed(const ProductionObligation& obligation,
                       const Simulator& before, const Simulator& after,
                       int player, int actor, Action action) {
  const auto& before_tile = tile_at(before, player, obligation.tile);
  const auto& after_tile = tile_at(after, player, obligation.tile);
  if (obligation.goal == GoalKind::CropReady) {
    if (action.op == Op::DIG) {
      return after_tile.kind == TileKind::EMPTY;
    }
    if (action.op == Op::PLANT) {
      return after_tile.kind == TileKind::PLANT &&
             after_tile.crop == obligation.item;
    }
    if (action.op == Op::WATER) {
      return after_tile.kind == TileKind::PLANT && after_tile.watered_today;
    }
    return false;
  }
  if (obligation.goal == GoalKind::Pickup) {
    return inventory_count(after, player, actor, obligation.item) -
               inventory_count(before, player, actor, obligation.item) >=
           obligation.quantity;
  }
  if (obligation.goal == GoalKind::Place) {
    return after_tile.kind == TileKind::ANIMAL &&
           after_tile.animal == obligation.item;
  }
  if (obligation.goal == GoalKind::Feed) {
    return !before_tile.fed_today && after_tile.fed_today;
  }
  if (obligation.goal == GoalKind::Care) {
    return !before_tile.cared_today && after_tile.cared_today;
  }
  if (obligation.goal == GoalKind::CollectFertilizer) {
    return before_tile.fertilizer_available &&
           !after_tile.fertilizer_available &&
           inventory_count(after, player, actor, Item::FERTILIZER) >
               inventory_count(before, player, actor, Item::FERTILIZER);
  }
  if (obligation.goal == GoalKind::Harvest) {
    return inventory_count(after, player, actor, obligation.item) >
               inventory_count(before, player, actor, obligation.item) ||
           after_tile.yield_units < before_tile.yield_units;
  }
  if (obligation.goal == GoalKind::BuildPasture ||
      obligation.goal == GoalKind::BuildCoop) {
    if (action.op == Op::DIG)
      return before_tile.kind != TileKind::EMPTY &&
             before_tile.kind != TileKind::ANIMAL &&
             before_tile.kind != TileKind::LOCKED &&
             after_tile.kind == TileKind::EMPTY;
    if (action.op == Op::BUILD_PASTURE || action.op == Op::BUILD_COOP)
      return after_tile.kind == (obligation.goal == GoalKind::BuildCoop
                                     ? TileKind::COOP
                                     : TileKind::PASTURE);
    return false;
  }
  return false;
}

bool goal_done(const ProductionObligation& obligation, const Simulator& state,
               int player) {
  if (obligation.goal == GoalKind::CropReady)
    return crop_done(tile_at(state, player, obligation.tile), obligation.item);
  if (obligation.goal == GoalKind::BuildPasture ||
      obligation.goal == GoalKind::BuildCoop) {
    return tile_at(state, player, obligation.tile).kind ==
        (obligation.goal == GoalKind::BuildCoop ? TileKind::COOP
                                                : TileKind::PASTURE);
  }
  return false;
}

bool stateful_goal(GoalKind goal) {
  return goal == GoalKind::CropReady || goal == GoalKind::BuildPasture ||
      goal == GoalKind::BuildCoop;
}

void refresh_stateful_completion(
    const std::vector<ProductionObligation>& obligations,
    const std::map<std::uint64_t, std::size_t>& index,
    const Simulator& state, int player, std::vector<bool>& complete) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 0; i < obligations.size(); ++i) {
      if (complete[i] || !stateful_goal(obligations[i].goal)) continue;
      const bool dependencies_complete = std::all_of(
          obligations[i].dependencies.begin(),
          obligations[i].dependencies.end(),
          [&](std::uint64_t dependency) {
            return complete[index.at(dependency)];
          });
      if (dependencies_complete && goal_done(obligations[i], state, player)) {
        complete[i] = true;
        changed = true;
      }
    }
  }
}

ObligationDisposition disposition_for(DebtReason reason) {
  switch (reason) {
    case DebtReason::BlockedOnReceipt:
      return ObligationDisposition::BlockedOnReceipt;
    case DebtReason::Capacity:
      return ObligationDisposition::CapacityDebt;
    case DebtReason::Dependency:
      return ObligationDisposition::DependencyDebt;
    case DebtReason::Deadline:
      return ObligationDisposition::DeadlineDebt;
    case DebtReason::UnsupportedState:
      return ObligationDisposition::UnsupportedStateDebt;
    case DebtReason::PolicyDeferred:
      return ObligationDisposition::PolicyDeferredDebt;
  }
  return ObligationDisposition::UnsupportedStateDebt;
}

bool validate_dag(const DayPlanRequest& request, int actors, int day_start,
                  int day_end) {
  std::set<std::uint64_t> ids;
  for (const auto& obligation : request.obligations) {
    if (obligation.id == 0 || !ids.insert(obligation.id).second ||
        obligation.actor < -1 || obligation.actor >= actors ||
        obligation.quantity <= 0 || !valid_obligation_item(obligation) ||
        obligation.resource.seed_quantity < 0 ||
        obligation.resource.shed_quantity < 0 ||
        obligation.resource.carried_quantity < 0 ||
        obligation.earliest_step < day_start ||
        obligation.deadline_step > day_end ||
        obligation.earliest_step > obligation.deadline_step ||
        (obligation.source_step != -1 &&
         (obligation.source_step < day_start ||
          obligation.source_step > day_end)) ||
        !valid_position(*request.day_start, obligation.tile)) {
      return false;
    }
    for (auto dependency : obligation.dependencies) {
      if (dependency == obligation.id) {
        return false;
      }
    }
  }
  for (const auto& obligation : request.obligations) {
    for (auto dependency : obligation.dependencies) {
      if (!ids.contains(dependency)) {
        return false;
      }
    }
  }
  std::map<std::uint64_t, int> color;
  std::map<std::uint64_t, const ProductionObligation*> by_id;
  for (const auto& obligation : request.obligations) {
    by_id[obligation.id] = &obligation;
  }
  std::function<bool(std::uint64_t)> visit = [&](std::uint64_t id) {
    if (color[id] == 1) {
      return false;
    }
    if (color[id] == 2) {
      return true;
    }
    color[id] = 1;
    for (auto dependency : by_id[id]->dependencies) {
      if (!visit(dependency)) {
        return false;
      }
    }
    color[id] = 2;
    return true;
  };
  for (const auto& obligation : request.obligations) {
    if (!visit(obligation.id)) {
      return false;
    }
  }
  return true;
}

bool status_equal(const ObligationFinalStatus& lhs,
                  const ObligationFinalStatus& rhs) {
  return lhs.obligation_id == rhs.obligation_id &&
         lhs.disposition == rhs.disposition &&
         lhs.assigned_actor == rhs.assigned_actor &&
         lhs.remaining_transitions == rhs.remaining_transitions &&
         lhs.transition_steps == rhs.transition_steps;
}

}  // namespace

static DayPlanResult plan_unchecked(const DayPlanRequest& request,
                                    int start_tick) {
  DayPlanResult output;
  if (!request.day_start) {
    output.reject = PlanReject::InvalidState;
    return output;
  }
  if (request.player < 0 || request.player >= 2 ||
      request.issuer_generation == 0) {
    output.reject = PlanReject::InvalidIdentity;
    return output;
  }
  const auto& start = *request.day_start;
  if (start_tick < 0 || start_tick >= kTurns ||
      request.start_tick != start_tick || start.hour() != start_tick ||
      start.step_count() != start.day() * kTurns + start_tick ||
      start.config().turns_per_day != kTurns || start.done()) {
    output.reject = PlanReject::NotDayStart;
    return output;
  }

  const int actors =
      static_cast<int>(start.farms()[request.player].hands.size()) + 1;
  const int day_start = start.day() * kTurns;
  const int day_end = std::min(day_start + kTurns - 1,
                               start.config().episode_steps - 2);
  const int remaining_start = day_start + start_tick;
  if (remaining_start > day_end) {
    output.reject = PlanReject::NotDayStart;
    return output;
  }
  const int end_tick = day_end - day_start + 1;
  std::vector<std::vector<MoveSourceToken>> actor_moves(actors);
  std::set<std::pair<int, int>> move_keys;
  for (const auto& token : request.moves) {
    if (token.actor < 0 || token.actor >= actors ||
        token.source_step < day_start || token.source_step > day_end ||
        !is_move(token.action.op) ||
        !move_keys.insert({token.actor, token.source_step}).second) {
      output.reject = PlanReject::InvalidMoveToken;
      return output;
    }
    actor_moves[token.actor].push_back(token);
  }
  for (auto& moves : actor_moves) {
    std::sort(moves.begin(), moves.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.source_step < rhs.source_step;
    });
  }
  if (!validate_dag(request, actors, day_start, day_end)) {
    output.reject = PlanReject::InvalidDag;
    return output;
  }

  const int remaining_ticks = end_tick - start_tick;
  output.manifest.assign(actors, std::vector<Action>(remaining_ticks));
  output.sources.assign(
      actors, std::vector<repair_fork::SourceBinding>(remaining_ticks));
  std::vector<bool> complete(request.obligations.size());
  std::vector<bool> receipt_blocked(request.obligations.size());
  std::vector<bool> unsupported(request.obligations.size());
  std::vector<int> assigned_actor(request.obligations.size(), -1);
  std::vector<std::vector<int>> transition_steps(request.obligations.size());
  std::map<std::uint64_t, std::size_t> index;
  for (std::size_t i = 0; i < request.obligations.size(); ++i) {
    index[request.obligations[i].id] = i;
  }
  refresh_stateful_completion(request.obligations, index, start,
                              request.player, complete);
  for (std::size_t i = 0; i < request.obligations.size(); ++i) {
    if (complete[i]) {
      output.completed.push_back(request.obligations[i].id);
    }
  }

  Simulator state = start;
  std::vector<std::deque<MoveSourceToken>> pending_moves(actors);
  std::vector<std::size_t> move_cursor(actors);
  DayScheduleCertificate certificate;
  certificate.player = request.player;
  certificate.day = start.day();
  certificate.start_step = remaining_start;
  certificate.issuer_generation = request.issuer_generation;
  certificate.focal_start_fingerprint =
      production_suffix::focal_unit_state_fingerprint(start, request.player);
  certificate.input_hash = request_input_hash(request);

  for (int tick = start_tick; tick < end_tick; ++tick) {
    const int slot_index = tick - start_tick;
    const int step = day_start + tick;
    for (int actor = 0; actor < actors; ++actor) {
      while (move_cursor[actor] < actor_moves[actor].size() &&
             actor_moves[actor][move_cursor[actor]].source_step <= step) {
        pending_moves[actor].push_back(actor_moves[actor][move_cursor[actor]++]);
      }
    }

    std::vector<Action> joint_actions(actors);
    std::vector<std::uint64_t> chosen(actors);
    std::array<int, fastkag::N_CROPS> reserved_seeds{};
    std::array<int, fastkag::N_ITEMS> reserved_shed{};
    std::set<std::pair<int, int>> used_tiles;

    for (int actor = 0; actor < actors; ++actor) {
      std::vector<std::size_t> candidates;
      for (std::size_t i = 0; i < request.obligations.size(); ++i) {
        const auto& obligation = request.obligations[i];
        if (complete[i] || obligation.policy_deferred ||
            step < obligation.earliest_step ||
            step > obligation.deadline_step ||
            (obligation.actor >= 0 && obligation.actor != actor) ||
            (assigned_actor[i] >= 0 && assigned_actor[i] != actor) ||
            !position_equal(actor_position(state, request.player, actor),
                            obligation.tile)) {
          continue;
        }
        bool dependencies_complete = true;
        for (auto dependency : obligation.dependencies) {
          dependencies_complete =
              dependencies_complete && complete[index[dependency]];
        }
        if (dependencies_complete) {
          candidates.push_back(i);
        }
      }
      std::sort(candidates.begin(), candidates.end(),
                [&](std::size_t lhs, std::size_t rhs) {
                  const auto& a = request.obligations[lhs];
                  const auto& b = request.obligations[rhs];
                  if (a.deadline_step != b.deadline_step)
                    return a.deadline_step < b.deadline_step;
                  if (a.must_finish_today != b.must_finish_today)
                    return a.must_finish_today;
                  if (a.priority != b.priority)
                    return a.priority > b.priority;
                  return a.id < b.id;
                });

      const int remaining_slots = end_tick - tick;
      const int future_moves =
          static_cast<int>(actor_moves[actor].size() - move_cursor[actor]);
      const bool production_slack =
          remaining_slots >
          static_cast<int>(pending_moves[actor].size()) + future_moves;
      if (production_slack) {
        for (auto candidate : candidates) {
          const auto& obligation = request.obligations[candidate];
          const auto tile_key =
              std::pair{static_cast<int>(obligation.tile.x),
                        static_cast<int>(obligation.tile.y)};
          if (used_tiles.contains(tile_key)) {
            continue;
          }
          const auto compiled = compile_obligation(
              obligation, state, request.player, actor, reserved_seeds,
              reserved_shed);
          receipt_blocked[candidate] =
              receipt_blocked[candidate] || compiled.blocked_on_receipt;
          unsupported[candidate] =
              unsupported[candidate] || compiled.unsupported;
          if (!compiled.action) {
            continue;
          }
          joint_actions[actor] = *compiled.action;
          chosen[actor] = obligation.id;
          used_tiles.insert(tile_key);
          if (compiled.action->op == Op::PLANT) {
            ++reserved_seeds[static_cast<int>(compiled.action->item)];
          } else if (compiled.action->op == Op::PICKUP) {
            reserved_shed[static_cast<int>(compiled.action->item)] +=
                compiled.action->quantity;
          }
          break;
        }
      }

      if (chosen[actor] == 0 && !pending_moves[actor].empty()) {
        const auto token = pending_moves[actor].front();
        pending_moves[actor].pop_front();
        joint_actions[actor] = token.action;
        output.sources[actor][slot_index] =
            {actor, token.source_step, token.action};
        certificate.move_replays.push_back(
            {actor, token.source_step, step, token.action});
        output.move_delays += step != token.source_step;
      } else if (chosen[actor] != 0) {
        output.sources[actor][slot_index] =
            {actor, -1, joint_actions[actor]};
      } else {
        output.sources[actor][slot_index] = {actor, -1, {}};
      }
      output.manifest[actor][slot_index] = joint_actions[actor];
    }

    std::array<PlayerAction, 2> joint;
    joint[request.player].units = joint_actions;
    Simulator after = state.preview_unit_phase(joint);
    bool retry = false;
    for (int actor = 0; actor < actors; ++actor) {
      if (chosen[actor] == 0) {
        continue;
      }
      const auto obligation_index = index[chosen[actor]];
      if (!action_progressed(request.obligations[obligation_index], state,
                             after, request.player, actor,
                             joint_actions[actor])) {
        joint_actions[actor] = {};
        output.manifest[actor][slot_index] = {};
        output.sources[actor][slot_index] = {actor, -1, {}};
        chosen[actor] = 0;
        unsupported[obligation_index] = true;
        retry = true;
      }
    }
    if (retry) {
      joint = {};
      joint[request.player].units = joint_actions;
      after = state.preview_unit_phase(joint);
    }

    state = after;
    for (int actor = 0; actor < actors; ++actor) {
      if (chosen[actor] != 0) {
        const auto obligation_index = index[chosen[actor]];
        // Binding is evidence about an emitted, effectful transition.  A
        // candidate can compile but still be rejected by the simulator (for
        // example PICKUP away from the shed); do not retain that speculative
        // actor in the terminal certificate.
        assigned_actor[obligation_index] = actor;
        transition_steps[obligation_index].push_back(step);
      }
    }
    const auto complete_before = complete;
    for (std::size_t i = 0; i < request.obligations.size(); ++i) {
      if (!complete[i] && !stateful_goal(request.obligations[i].goal))
        complete[i] = std::find(chosen.begin(), chosen.end(),
                                request.obligations[i].id) != chosen.end();
    }
    refresh_stateful_completion(request.obligations, index, state,
                                request.player, complete);
    for (std::size_t i = 0; i < request.obligations.size(); ++i) {
      if (!complete_before[i] && complete[i]) {
        output.completed.push_back(request.obligations[i].id);
      }
    }

    DaySlotProof slot;
    slot.step = step;
    slot.actions = joint_actions;
    slot.obligation_ids = chosen;
    slot.focal_post_fingerprint =
        production_suffix::focal_unit_state_fingerprint(state, request.player,
                                                        step);
    for (int actor = 0; actor < actors; ++actor) {
      slot.sources.push_back(output.sources[actor][slot_index]);
    }
    certificate.slots.push_back(std::move(slot));
  }

  for (int actor = 0; actor < actors; ++actor) {
    if (!pending_moves[actor].empty() ||
        move_cursor[actor] != actor_moves[actor].size()) {
      output.reject = PlanReject::InternalProofFailure;
      return output;
    }
  }

  std::map<std::uint64_t, DebtReason> debt_reasons;
  for (std::size_t i = 0; i < request.obligations.size(); ++i) {
    if (complete[i]) {
      continue;
    }
    const auto& obligation = request.obligations[i];
    DebtReason reason = obligation.policy_deferred
                            ? DebtReason::PolicyDeferred
                            : receipt_blocked[i]
                            ? DebtReason::BlockedOnReceipt
                            : unsupported[i] ? DebtReason::UnsupportedState
                                             : DebtReason::Capacity;
    bool dependencies_complete = true;
    for (auto dependency : obligation.dependencies) {
      dependencies_complete =
          dependencies_complete && complete[index[dependency]];
    }
    if (obligation.policy_deferred) {
      reason = DebtReason::PolicyDeferred;
    } else if (!dependencies_complete) {
      reason = DebtReason::Dependency;
    } else if (obligation.must_finish_today &&
               obligation.deadline_step <= day_end && !receipt_blocked[i] &&
               !unsupported[i]) {
      reason = DebtReason::Deadline;
    }
    const int remaining =
        transitions_remaining(obligation, state, request.player);
    output.debts.push_back({obligation.id, reason, remaining});
    debt_reasons[obligation.id] = reason;
  }

  for (std::size_t i = 0; i < request.obligations.size(); ++i) {
    const auto& obligation = request.obligations[i];
    ObligationFinalStatus status;
    status.obligation_id = obligation.id;
    status.disposition = complete[i]
                             ? ObligationDisposition::Completed
                             : disposition_for(debt_reasons[obligation.id]);
    status.assigned_actor = assigned_actor[i];
    status.remaining_transitions =
        complete[i] ? 0
                    : transitions_remaining(obligation, state, request.player);
    status.transition_steps = transition_steps[i];
    certificate.obligation_statuses.push_back(std::move(status));
  }
  certificate.content_hash = day_schedule_certificate_hash(certificate);
  output.certificate = std::move(certificate);
  return output;
}

DayPlanResult internal::plan_day_unchecked(const DayPlanRequest& request) {
  return plan_unchecked(request, 0);
}

std::uint64_t day_schedule_certificate_hash(
    const DayScheduleCertificate& certificate) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_value(hash, certificate.player);
  hash_value(hash, certificate.day);
  hash_value(hash, certificate.start_step);
  hash_value(hash, certificate.issuer_generation);
  hash_value(hash, certificate.focal_start_fingerprint);
  hash_value(hash, certificate.input_hash);
  hash_value(hash, certificate.slots.size());
  for (const auto& slot : certificate.slots) {
    hash_value(hash, slot.step);
    hash_value(hash, slot.actions.size());
    for (auto action : slot.actions) {
      hash_action(hash, action);
    }
    hash_value(hash, slot.sources.size());
    for (const auto& source : slot.sources) {
      hash_value(hash, source.actor);
      hash_value(hash, source.source_step);
      hash_action(hash, source.source_action);
    }
    hash_value(hash, slot.obligation_ids.size());
    for (auto obligation_id : slot.obligation_ids) {
      hash_value(hash, obligation_id);
    }
    hash_value(hash, slot.focal_post_fingerprint);
  }
  hash_value(hash, certificate.move_replays.size());
  for (const auto& move : certificate.move_replays) {
    hash_value(hash, move.actor);
    hash_value(hash, move.source_step);
    hash_value(hash, move.emitted_step);
    hash_action(hash, move.action);
  }
  hash_value(hash, certificate.obligation_statuses.size());
  for (const auto& status : certificate.obligation_statuses) {
    hash_value(hash, status.obligation_id);
    hash_value(hash, static_cast<int>(status.disposition));
    hash_value(hash, status.assigned_actor);
    hash_value(hash, status.remaining_transitions);
    hash_value(hash, status.transition_steps.size());
    for (int step : status.transition_steps) {
      hash_value(hash, step);
    }
  }
  return hash;
}

static VerifyResult verify_schedule(const DayPlanRequest& request,
                                    const DayScheduleCertificate& certificate,
                                    int start_tick) {
  VerifyResult result;
  const auto reject = [&](VerifyFailureReason reason, int step = -1,
                          int actor = -1, std::uint64_t obligation_id = 0,
                          int index = -1) {
    result.valid = false;
    result.reject = PlanReject::InternalProofFailure;
    result.failure_reason = reason;
    result.failure_step = step;
    result.failure_actor = actor;
    result.failure_obligation_id = obligation_id;
    result.failure_index = index;
    return result;
  };
  if (!request.day_start || request.player < 0 || request.player >= 2 ||
      request.issuer_generation == 0 || start_tick < 0 ||
      start_tick >= kTurns || request.day_start->hour() != start_tick ||
      request.start_tick != start_tick ||
      request.day_start->step_count() !=
          request.day_start->day() * kTurns + start_tick ||
      request.day_start->config().turns_per_day != kTurns ||
      request.day_start->done()) {
    return reject(VerifyFailureReason::InvalidRequest);
  }
  const int actors = static_cast<int>(
                         request.day_start->farms()[request.player].hands.size()) +
                     1;
  const int day_start = request.day_start->day() * kTurns;
  const int day_end = std::min(day_start + kTurns - 1,
      request.day_start->config().episode_steps - 2);
  const int end_tick = day_end - day_start + 1;
  if (day_start + start_tick > day_end)
    return reject(VerifyFailureReason::InvalidRequest);
  if (!validate_dag(request, actors, day_start, day_end) ||
      certificate.content_hash != day_schedule_certificate_hash(certificate) ||
      certificate.player != request.player ||
      certificate.day != request.day_start->day() ||
      certificate.start_step != day_start + start_tick ||
      certificate.issuer_generation != request.issuer_generation ||
      certificate.focal_start_fingerprint !=
          production_suffix::focal_unit_state_fingerprint(*request.day_start,
                                                          request.player) ||
      certificate.input_hash != request_input_hash(request) ||
      certificate.slots.size() !=
          static_cast<std::size_t>(end_tick - start_tick) ||
      certificate.obligation_statuses.size() != request.obligations.size()) {
    return reject(VerifyFailureReason::CertificateEnvelope);
  }

  std::map<std::uint64_t, std::size_t> index;
  for (std::size_t i = 0; i < request.obligations.size(); ++i) {
    index[request.obligations[i].id] = i;
  }
  std::vector<bool> completed(request.obligations.size());
  std::vector<int> assigned_actor(request.obligations.size(), -1);
  std::vector<std::vector<int>> transition_steps(request.obligations.size());
  refresh_stateful_completion(request.obligations, index, *request.day_start,
                              request.player, completed);

  Simulator state = *request.day_start;
  std::vector<MoveReplay> actual_moves;
  for (int tick = start_tick; tick < end_tick; ++tick) {
    const auto& slot = certificate.slots[tick - start_tick];
    const int step = day_start + tick;
    if (slot.step != step ||
        slot.actions.size() != static_cast<std::size_t>(actors) ||
        slot.sources.size() != static_cast<std::size_t>(actors) ||
        slot.obligation_ids.size() != static_cast<std::size_t>(actors)) {
      return reject(VerifyFailureReason::SlotShape, step);
    }

    std::set<std::uint64_t> used_obligations;
    for (int actor = 0; actor < actors; ++actor) {
      const auto action = slot.actions[actor];
      const auto obligation_id = slot.obligation_ids[actor];
      const auto& source = slot.sources[actor];
      if (source.actor != actor ||
          !action_equal(source.source_action, action)) {
        return reject(VerifyFailureReason::SourceBinding, step, actor,
                      obligation_id);
      }
      if (obligation_id == 0) {
        if (action.op != Op::PASS && !is_move(action.op)) {
          return reject(VerifyFailureReason::UnexpectedUnboundAction, step,
                        actor);
        }
        continue;
      }
      if (!index.contains(obligation_id) || action.op == Op::PASS ||
          is_move(action.op) || source.source_step != -1 ||
          !used_obligations.insert(obligation_id).second) {
        return reject(VerifyFailureReason::ObligationBinding, step, actor,
                      obligation_id);
      }

      const auto obligation_index = index[obligation_id];
      const auto& obligation = request.obligations[obligation_index];
      if (completed[obligation_index] || step < obligation.earliest_step ||
          step > obligation.deadline_step ||
          (obligation.actor >= 0 && obligation.actor != actor) ||
          (assigned_actor[obligation_index] >= 0 &&
           assigned_actor[obligation_index] != actor) ||
          !position_equal(actor_position(state, request.player, actor),
                          obligation.tile)) {
        return reject(VerifyFailureReason::ObligationEligibility, step, actor,
                      obligation_id);
      }
      for (auto dependency : obligation.dependencies) {
        if (!completed[index[dependency]]) {
          return reject(VerifyFailureReason::DependencyIncomplete, step,
                        actor, obligation_id);
        }
      }
      const std::array<int, fastkag::N_CROPS> no_reserved_seeds{};
      const std::array<int, fastkag::N_ITEMS> no_reserved_shed{};
      const auto compiled = compile_obligation(
          obligation, state, request.player, actor, no_reserved_seeds,
          no_reserved_shed);
      if (!compiled.action || !action_equal(*compiled.action, action)) {
        return reject(VerifyFailureReason::CompiledActionMismatch, step, actor,
                      obligation_id);
      }
      assigned_actor[obligation_index] = actor;
    }

    std::array<PlayerAction, 2> joint;
    joint[request.player].units = slot.actions;
    Simulator after = state.preview_unit_phase(joint);
    for (int actor = 0; actor < actors; ++actor) {
      const auto obligation_id = slot.obligation_ids[actor];
      if (obligation_id == 0) {
        continue;
      }
      const auto obligation_index = index[obligation_id];
      const auto& obligation = request.obligations[obligation_index];
      if (!action_progressed(obligation, state, after, request.player, actor,
                             slot.actions[actor])) {
        return reject(VerifyFailureReason::ActionNoProgress, step, actor,
                      obligation_id);
      }
      transition_steps[obligation_index].push_back(step);
      if (!stateful_goal(obligation.goal)) completed[obligation_index] = true;
    }
    state = after;
    refresh_stateful_completion(request.obligations, index, state,
                                request.player, completed);
    if (production_suffix::focal_unit_state_fingerprint(state, request.player,
                                                        step) !=
        slot.focal_post_fingerprint) {
      return reject(VerifyFailureReason::PostFingerprintMismatch, step);
    }
    for (int actor = 0; actor < actors; ++actor) {
      if (is_move(slot.actions[actor].op)) {
        actual_moves.push_back({actor, slot.sources[actor].source_step, step,
                                slot.actions[actor]});
      }
    }
    ++result.checked_slots;
  }

  if (actual_moves.size() != certificate.move_replays.size()) {
    return reject(VerifyFailureReason::MoveReplayCount);
  }
  std::vector<int> last_source(actors, -1);
  for (std::size_t i = 0; i < actual_moves.size(); ++i) {
    const auto& actual = actual_moves[i];
    const auto& claimed = certificate.move_replays[i];
    if (actual.actor != claimed.actor ||
        actual.source_step != claimed.source_step ||
        actual.emitted_step != claimed.emitted_step ||
        !action_equal(actual.action, claimed.action) ||
        actual.source_step / kTurns != certificate.day ||
        actual.emitted_step / kTurns != certificate.day ||
        actual.emitted_step < certificate.start_step ||
        actual.emitted_step < actual.source_step ||
        actual.source_step <= last_source[actual.actor]) {
      return reject(VerifyFailureReason::MoveReplayBinding,
                    actual.emitted_step, actual.actor, 0,
                    static_cast<int>(i));
    }
    last_source[actual.actor] = actual.source_step;
  }

  std::vector<MoveSourceToken> expected_moves = request.moves;
  std::sort(expected_moves.begin(), expected_moves.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tie(lhs.actor, lhs.source_step) <
                     std::tie(rhs.actor, rhs.source_step);
            });
  auto sorted_actual = actual_moves;
  std::sort(sorted_actual.begin(), sorted_actual.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tie(lhs.actor, lhs.source_step) <
                     std::tie(rhs.actor, rhs.source_step);
            });
  if (sorted_actual.size() != expected_moves.size()) {
    return reject(VerifyFailureReason::MoveCoverageCount);
  }
  for (std::size_t i = 0; i < expected_moves.size(); ++i) {
    if (sorted_actual[i].actor != expected_moves[i].actor ||
        sorted_actual[i].source_step != expected_moves[i].source_step ||
        !action_equal(sorted_actual[i].action, expected_moves[i].action)) {
      return reject(VerifyFailureReason::MoveCoverageBinding,
                    sorted_actual[i].emitted_step, sorted_actual[i].actor, 0,
                    static_cast<int>(i));
    }
  }

  std::set<std::uint64_t> status_ids;
  for (const auto& status : certificate.obligation_statuses) {
    if (!index.contains(status.obligation_id) ||
        !status_ids.insert(status.obligation_id).second) {
      return reject(VerifyFailureReason::StatusIdentity, -1, -1,
                    status.obligation_id);
    }
    const auto obligation_index = index[status.obligation_id];
    if (status.assigned_actor != assigned_actor[obligation_index] ||
        status.transition_steps != transition_steps[obligation_index] ||
        (status.disposition == ObligationDisposition::Completed) !=
            completed[obligation_index] ||
        (completed[obligation_index] && status.remaining_transitions != 0) ||
        (!completed[obligation_index] &&
         status.remaining_transitions != transitions_remaining(
                                             request.obligations[obligation_index],
                                             state, request.player))) {
      return reject(VerifyFailureReason::StatusMismatch, -1,
                    status.assigned_actor, status.obligation_id);
    }
  }

  // Re-derive the deterministic canonical plan and every debt classification.
  // Recomputing content_hash after swallowing an obligation is insufficient.
  const auto canonical = plan_unchecked(request, start_tick);
  if (!canonical.planned() ||
      canonical.certificate->content_hash != certificate.content_hash ||
      canonical.certificate->obligation_statuses.size() !=
          certificate.obligation_statuses.size()) {
    return reject(VerifyFailureReason::CanonicalPlanMismatch);
  }
  for (std::size_t i = 0; i < certificate.obligation_statuses.size(); ++i) {
    if (!status_equal(canonical.certificate->obligation_statuses[i],
                      certificate.obligation_statuses[i])) {
      return reject(VerifyFailureReason::CanonicalStatusMismatch, -1,
                    certificate.obligation_statuses[i].assigned_actor,
                    certificate.obligation_statuses[i].obligation_id,
                    static_cast<int>(i));
    }
  }

  result.valid = true;
  result.reject = PlanReject::None;
  return result;
}

VerifyResult verify_day_schedule(const DayPlanRequest& request,
                                 const DayScheduleCertificate& certificate) {
  return verify_schedule(request, certificate, 0);
}

VerifyResult verify_remaining_day_schedule(
    const DayPlanRequest& request,
    const DayScheduleCertificate& certificate) {
  return verify_schedule(request, certificate, request.start_tick);
}

DayPlanResult plan_day(const DayPlanRequest& request) {
  auto output = internal::plan_day_unchecked(request);
  if (!output.planned()) {
    return output;
  }
  const auto verification = verify_day_schedule(request, *output.certificate);
  if (!verification.valid) {
    output.reject = PlanReject::InternalProofFailure;
    output.manifest.clear();
    output.sources.clear();
    output.debts.clear();
    output.completed.clear();
    output.certificate.reset();
    output.move_delays = 0;
  }
  return output;
}

DayPlanResult plan_remaining_day(const DayPlanRequest& request) {
  const int start_tick = request.start_tick;
  auto output = plan_unchecked(request, start_tick);
  if (!output.planned()) {
    return output;
  }
  const auto verification =
      verify_remaining_day_schedule(request, *output.certificate);
  if (!verification.valid) {
    output.reject = PlanReject::InternalProofFailure;
    output.manifest.clear();
    output.sources.clear();
    output.debts.clear();
    output.completed.clear();
    output.certificate.reset();
    output.move_delays = 0;
  }
  return output;
}

const char* debt_reason_name(DebtReason reason) {
  switch (reason) {
    case DebtReason::BlockedOnReceipt:
      return "blocked_on_receipt";
    case DebtReason::Capacity:
      return "capacity";
    case DebtReason::Dependency:
      return "dependency";
    case DebtReason::Deadline:
      return "deadline";
    case DebtReason::UnsupportedState:
      return "unsupported_state";
    case DebtReason::PolicyDeferred:
      return "policy_deferred";
  }
  return "unknown";
}

const char* verify_failure_reason_name(VerifyFailureReason reason) {
  switch (reason) {
    case VerifyFailureReason::None: return "none";
    case VerifyFailureReason::InvalidRequest: return "invalid_request";
    case VerifyFailureReason::CertificateEnvelope: return "certificate_envelope";
    case VerifyFailureReason::SlotShape: return "slot_shape";
    case VerifyFailureReason::SourceBinding: return "source_binding";
    case VerifyFailureReason::UnexpectedUnboundAction:
      return "unexpected_unbound_action";
    case VerifyFailureReason::ObligationBinding: return "obligation_binding";
    case VerifyFailureReason::ObligationEligibility:
      return "obligation_eligibility";
    case VerifyFailureReason::DependencyIncomplete:
      return "dependency_incomplete";
    case VerifyFailureReason::CompiledActionMismatch:
      return "compiled_action_mismatch";
    case VerifyFailureReason::ActionNoProgress: return "action_no_progress";
    case VerifyFailureReason::PostFingerprintMismatch:
      return "post_fingerprint_mismatch";
    case VerifyFailureReason::MoveReplayCount: return "move_replay_count";
    case VerifyFailureReason::MoveReplayBinding: return "move_replay_binding";
    case VerifyFailureReason::MoveCoverageCount: return "move_coverage_count";
    case VerifyFailureReason::MoveCoverageBinding:
      return "move_coverage_binding";
    case VerifyFailureReason::StatusIdentity: return "status_identity";
    case VerifyFailureReason::StatusMismatch: return "status_mismatch";
    case VerifyFailureReason::CanonicalPlanMismatch:
      return "canonical_plan_mismatch";
    case VerifyFailureReason::CanonicalStatusMismatch:
      return "canonical_status_mismatch";
  }
  return "unknown";
}

}  // namespace g001::obligation_day
