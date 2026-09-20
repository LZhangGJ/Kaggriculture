#include "repair_minimum_damage_selector.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace g001::minimum_damage {
namespace {

constexpr std::uint64_t kRawIdBase = 1ULL << 63;

struct MoveToken {
  int actor{-1};
  int source_slot{-1};
  Action action{};
};

struct Prepared {
  const Request* request{};
  std::vector<TypedObligation> obligations;
  std::vector<std::vector<MoveToken>> moves;
  std::map<std::uint64_t, int> obligation_by_id;
  bool valid{};
  bool hard_move_feasible{};
  std::string diagnostic;
};

bool valid_position(const DayStartSnapshot& snapshot, Position position) {
  return position.x >= 0 && position.y >= 0 &&
         position.x < snapshot.width && position.y < snapshot.height;
}

std::size_t tile_index(const DayStartSnapshot& snapshot, Position position) {
  return static_cast<std::size_t>(position.y * snapshot.width + position.x);
}

Position moved(Position value, ActionKind kind) {
  if (kind == ActionKind::North) --value.y;
  if (kind == ActionKind::South) ++value.y;
  if (kind == ActionKind::West) --value.x;
  if (kind == ActionKind::East) ++value.x;
  return value;
}

bool action_supported_on_unit(ActionKind kind) {
  return kind == ActionKind::Pass || is_move(kind) ||
         kind == ActionKind::Dig || kind == ActionKind::BuildPasture ||
         kind == ActionKind::BuildCoop ||
         kind == ActionKind::Pickup || kind == ActionKind::Plant ||
         kind == ActionKind::Water || kind == ActionKind::CollectFertilizer ||
         kind == ActionKind::PlaceAnimal ||
         kind == ActionKind::Feed || kind == ActionKind::Care;
}

// These effects are represented by per-day flags. The simulator clears them
// during the hour-23 transition, so a post-step receipt cannot prove success.
bool receipt_lost_at_day_boundary(ActionKind kind) {
  return kind == ActionKind::Water || kind == ActionKind::Feed ||
         kind == ActionKind::Care;
}

std::uint64_t raw_id(int actor, int slot, int horizon) {
  return kRawIdBase + static_cast<std::uint64_t>(actor * horizon + slot + 1);
}

Prepared prepare(const Request& request) {
  Prepared output;
  output.request = &request;
  output.valid = false;
  output.hard_move_feasible = false;
  if (request.horizon <= 0 || request.horizon > 64 ||
      request.maximum_states == 0 || request.day_start.width <= 0 ||
      request.day_start.height <= 0) {
    output.diagnostic = "invalid horizon/state limit/map dimensions";
    return output;
  }
  const auto& snapshot = request.day_start;
  if (snapshot.tiles.size() !=
          static_cast<std::size_t>(snapshot.width * snapshot.height) ||
      request.raw_sources.size() != snapshot.actor_positions.size()) {
    output.diagnostic = "snapshot tile/actor source shape mismatch";
    return output;
  }
  if (!snapshot.actor_inventory.empty()) {
    if (snapshot.actor_inventory.size() != snapshot.actor_positions.size()) {
      output.diagnostic = "actor carried-inventory shape mismatch";
      return output;
    }
    const auto width = snapshot.actor_inventory.front().size();
    if (std::any_of(snapshot.actor_inventory.begin(),
                    snapshot.actor_inventory.end(), [&](const auto& values) {
                      return values.size() != width;
                    })) {
      output.diagnostic = "ragged actor carried-inventory shape";
      return output;
    }
  }
  output.moves.resize(request.raw_sources.size());
  std::set<std::uint64_t> ids;
  std::vector<Position> projected = snapshot.actor_positions;
  for (std::size_t actor = 0; actor < request.raw_sources.size(); ++actor) {
    if (request.raw_sources[actor].size() !=
            static_cast<std::size_t>(request.horizon) ||
        !valid_position(snapshot, projected[actor])) {
      output.diagnostic = "invalid actor lane or day-start position";
      return output;
    }
    for (int slot = 0; slot < request.horizon; ++slot) {
      const auto& source =
          request.raw_sources[actor][static_cast<std::size_t>(slot)];
      if (source.cascade_risk < 0 || source.action.quantity < 0 ||
          is_market(source.action.kind) ||
          !action_supported_on_unit(source.action.kind)) {
        output.diagnostic = "unsupported raw unit source";
        return output;
      }
      if (is_move(source.action.kind)) {
        const auto next = moved(projected[actor], source.action.kind);
        if (!valid_position(snapshot, next)) {
          output.valid = true;
          output.diagnostic = "immutable MOVE route crosses map boundary";
          return output;
        }
        output.moves[actor].push_back(
            {static_cast<int>(actor), slot, source.action});
        projected[actor] = next;
      } else if (source.action.kind != ActionKind::Pass) {
        TypedObligation obligation;
        obligation.id = raw_id(static_cast<int>(actor), slot, request.horizon);
        obligation.kind = ObligationKind::RawProduction;
        obligation.actor = static_cast<int>(actor);
        obligation.release_slot = slot;
        obligation.deadline_slot = request.horizon - 1;
        obligation.source_slot = slot;
        obligation.must_finish = source.must_finish;
        obligation.important = source.important;
        obligation.cascade_risk = source.cascade_risk;
        obligation.transitions.push_back(
            {Channel::Unit, source.action, true, projected[actor], false, 0});
        ids.insert(obligation.id);
        output.obligations.push_back(std::move(obligation));
      }
    }
  }

  for (const auto& obligation : request.repair_obligations) {
    if (obligation.id == 0 || obligation.id >= kRawIdBase ||
        !ids.insert(obligation.id).second || obligation.transitions.empty() ||
        obligation.release_slot < 0 || obligation.deadline_slot < 0 ||
        obligation.release_slot >= request.horizon ||
        obligation.deadline_slot >= request.horizon ||
        obligation.release_slot > obligation.deadline_slot ||
        obligation.cascade_risk < 0) {
      output.diagnostic = "invalid or duplicate typed obligation identity";
      return output;
    }
    bool has_unit = false;
    bool has_market = false;
    for (const auto& transition : obligation.transitions) {
      has_unit = has_unit || transition.channel == Channel::Unit;
      has_market = has_market || transition.channel == Channel::Market;
      if (transition.action.quantity <= 0 ||
          (transition.channel == Channel::Unit &&
           (!action_supported_on_unit(transition.action.kind) ||
            is_market(transition.action.kind))) ||
          (transition.channel == Channel::Market &&
           !is_market(transition.action.kind)) ||
          (transition.requires_position &&
           !valid_position(snapshot, transition.tile)) ||
          transition.worst_case_cash_cost < 0) {
        output.diagnostic = "malformed typed transition";
        return output;
      }
    }
    if ((has_unit && has_market) ||
        (has_unit && (obligation.actor < 0 ||
                      obligation.actor >=
                          static_cast<int>(snapshot.actor_positions.size()))) ||
        (has_market && obligation.actor != -1)) {
      output.diagnostic = "obligation channel/actor mismatch";
      return output;
    }
    output.obligations.push_back(obligation);
  }
  for (std::size_t index = 0; index < output.obligations.size(); ++index)
    output.obligation_by_id.emplace(output.obligations[index].id,
                                    static_cast<int>(index));
  for (const auto& obligation : output.obligations) {
    for (const auto dependency : obligation.dependencies) {
      if (dependency == obligation.id ||
          !output.obligation_by_id.contains(dependency)) {
        output.diagnostic = "unknown/self obligation dependency";
        return output;
      }
    }
  }
  output.valid = true;
  output.hard_move_feasible = true;
  output.diagnostic = "prepared";
  return output;
}

struct State {
  int tick{};
  std::vector<Position> positions;
  std::vector<TileKind> tiles;
  std::vector<int> seed_inventory;
  std::vector<int> animal_inventory;
  std::vector<int> shed_inventory;
  std::vector<std::vector<int>> actor_inventory;
  int feed_inventory{};
  int cash{};
  std::vector<int> move_cursor;
  std::vector<int> progress;
};

struct Candidate {
  bool valid{};
  Objective objective{};
  std::vector<ScheduleSlot> slots;
  std::vector<DayBoundDebt> debts;
};

void append_int(std::string& key, std::int64_t value) {
  for (int byte = 0; byte < 8; ++byte)
    key.push_back(static_cast<char>(
        static_cast<std::uint64_t>(value) >> (byte * 8)));
}

std::string state_key(const State& state) {
  std::string key;
  key.reserve(64 + state.tiles.size() +
              (state.positions.size() + state.move_cursor.size() +
               state.progress.size() + state.seed_inventory.size() +
               state.animal_inventory.size()) *
                  8);
  append_int(key, state.tick);
  for (const auto value : state.positions) {
    append_int(key, value.x);
    append_int(key, value.y);
  }
  for (const auto value : state.tiles)
    key.push_back(static_cast<char>(value));
  for (const auto value : state.seed_inventory) append_int(key, value);
  for (const auto value : state.animal_inventory) append_int(key, value);
  for (const auto value : state.shed_inventory) append_int(key, value);
  for (const auto& inventory : state.actor_inventory)
    for (const auto value : inventory) append_int(key, value);
  append_int(key, state.feed_inventory);
  append_int(key, state.cash);
  for (const auto value : state.move_cursor) append_int(key, value);
  for (const auto value : state.progress) append_int(key, value);
  return key;
}

void add_damage(Objective& objective, int inserted_move_distance,
                int move_displacement,
                int production_disturbance) {
  objective.inserted_move_distance += inserted_move_distance;
  objective.move_slot_displacement += move_displacement;
  objective.production_slot_disturbance += production_disturbance;
}

bool complete(const Prepared& prepared, const State& state,
              std::uint64_t id) {
  const auto found = prepared.obligation_by_id.find(id);
  return found != prepared.obligation_by_id.end() &&
         state.progress[static_cast<std::size_t>(found->second)] ==
             static_cast<int>(prepared.obligations[
                                  static_cast<std::size_t>(found->second)]
                                  .transitions.size());
}

bool dependencies_complete(const Prepared& prepared, const State& state,
                           const TypedObligation& obligation) {
  return std::all_of(obligation.dependencies.begin(),
                     obligation.dependencies.end(), [&](const auto id) {
                       return complete(prepared, state, id);
                     });
}

bool inventory_has(const std::vector<int>& inventory, int item, int quantity) {
  return item >= 0 && item < static_cast<int>(inventory.size()) &&
         inventory[static_cast<std::size_t>(item)] >= quantity;
}

bool apply_unit(const Prepared& prepared, State& state, int actor,
                const Action& action) {
  const auto& snapshot = prepared.request->day_start;
  if (actor < 0 || actor >= static_cast<int>(state.positions.size()))
    return false;
  if (action.kind == ActionKind::Pass) return true;
  if (is_move(action.kind)) {
    const auto next = moved(state.positions[static_cast<std::size_t>(actor)],
                            action.kind);
    if (!valid_position(snapshot, next))
      return false;
    state.positions[static_cast<std::size_t>(actor)] = next;
    return true;
  }
  const auto position = state.positions[static_cast<std::size_t>(actor)];
  auto& tile = state.tiles[tile_index(snapshot, position)];
  const int quantity = action.quantity;
  switch (action.kind) {
    case ActionKind::Dig:
      if (tile != TileKind::Weed && tile != TileKind::Crop) return false;
      tile = TileKind::Empty;
      return true;
    case ActionKind::BuildPasture:
      if (tile != TileKind::Empty) return false;
      tile = TileKind::Pasture;
      return true;
    case ActionKind::BuildCoop:
      if (tile != TileKind::Empty) return false;
      tile = TileKind::OtherStructure;
      return true;
    case ActionKind::Pickup:
      if (!inventory_has(state.shed_inventory, action.item, quantity) ||
          actor >= static_cast<int>(state.actor_inventory.size()) ||
          action.item >= static_cast<int>(
                             state.actor_inventory[static_cast<std::size_t>(
                                 actor)]
                                 .size()))
        return false;
      state.shed_inventory[static_cast<std::size_t>(action.item)] -= quantity;
      state.actor_inventory[static_cast<std::size_t>(actor)]
                           [static_cast<std::size_t>(action.item)] += quantity;
      return true;
    case ActionKind::Plant:
      if ((tile != TileKind::Empty && tile != TileKind::Soil) ||
          !inventory_has(state.seed_inventory, action.item, quantity))
        return false;
      state.seed_inventory[static_cast<std::size_t>(action.item)] -= quantity;
      tile = TileKind::Crop;
      return true;
    case ActionKind::Water:
      return tile == TileKind::Crop;
    case ActionKind::CollectFertilizer:
      return tile == TileKind::Animal;
    case ActionKind::PlaceAnimal:
      if (tile != TileKind::Pasture) return false;
      if (actor < static_cast<int>(state.actor_inventory.size()) &&
          inventory_has(state.actor_inventory[static_cast<std::size_t>(actor)],
                        action.item, quantity)) {
        state.actor_inventory[static_cast<std::size_t>(actor)]
                             [static_cast<std::size_t>(action.item)] -=
            quantity;
      } else if (inventory_has(state.animal_inventory, action.item,
                               quantity)) {
        state.animal_inventory[static_cast<std::size_t>(action.item)] -=
            quantity;
      } else {
        return false;
      }
      tile = TileKind::Animal;
      return true;
    case ActionKind::Feed:
      if (tile != TileKind::Animal) return false;
      if (actor < static_cast<int>(state.actor_inventory.size()) &&
          action.item >= 0 &&
          inventory_has(state.actor_inventory[static_cast<std::size_t>(actor)],
                        action.item, quantity)) {
        state.actor_inventory[static_cast<std::size_t>(actor)]
                             [static_cast<std::size_t>(action.item)] -=
            quantity;
      } else if (state.feed_inventory >= quantity) {
        state.feed_inventory -= quantity;
      } else {
        return false;
      }
      return true;
    case ActionKind::Care:
      return tile == TileKind::Animal;
    default:
      return false;
  }
}

bool apply_transition_unit(const Prepared& prepared, State& state, int actor,
                           const Transition& transition) {
  if (transition.channel != Channel::Unit) return false;
  if (transition.requires_position &&
      state.positions[static_cast<std::size_t>(actor)] != transition.tile)
    return false;
  return apply_unit(prepared, state, actor, transition.action);
}

bool apply_market(State& state, const Transition& transition) {
  if (transition.channel != Channel::Market ||
      !transition.guaranteed_fill ||
      state.cash < transition.worst_case_cash_cost)
    return false;
  const auto& action = transition.action;
  if (action.item < 0 || action.quantity <= 0) return false;
  state.cash -= transition.worst_case_cash_cost;
  if (action.kind == ActionKind::BuySeed) {
    if (action.item >= static_cast<int>(state.seed_inventory.size()))
      return false;
    state.seed_inventory[static_cast<std::size_t>(action.item)] +=
        action.quantity;
  } else if (action.kind == ActionKind::BuyAnimal) {
    if (action.item < static_cast<int>(state.shed_inventory.size()))
      state.shed_inventory[static_cast<std::size_t>(action.item)] +=
          action.quantity;
    else if (action.item < static_cast<int>(state.animal_inventory.size()))
      state.animal_inventory[static_cast<std::size_t>(action.item)] +=
          action.quantity;
    else
      return false;
  } else if (action.kind == ActionKind::BuyFeed) {
    if (action.item >= 0 &&
        action.item < static_cast<int>(state.shed_inventory.size()))
      state.shed_inventory[static_cast<std::size_t>(action.item)] +=
          action.quantity;
    else
      state.feed_inventory += action.quantity;
  } else {
    return false;
  }
  return true;
}

bool route_contains(const Prepared& prepared, int actor, Position target) {
  auto position = prepared.request->day_start.actor_positions[
      static_cast<std::size_t>(actor)];
  if (position == target) return true;
  for (const auto& token : prepared.moves[static_cast<std::size_t>(actor)]) {
    position = moved(position, token.action.kind);
    if (position == target) return true;
  }
  return false;
}

DebtReason debt_reason(const Prepared& prepared, const State& state,
                       std::size_t index) {
  const auto& obligation = prepared.obligations[index];
  const int progress = state.progress[index];
  if (progress > 0) return DebtReason::PartialDayBound;
  for (const auto dependency_id : obligation.dependencies) {
    const auto dependency_found =
        prepared.obligation_by_id.find(dependency_id);
    if (dependency_found == prepared.obligation_by_id.end())
      return DebtReason::UnsupportedIdentity;
    const auto dependency_index =
        static_cast<std::size_t>(dependency_found->second);
    const auto& dependency = prepared.obligations[dependency_index];
    const int dependency_progress = state.progress[dependency_index];
    if (dependency_progress ==
        static_cast<int>(dependency.transitions.size()))
      continue;
    const auto& blocked = dependency.transitions[
        static_cast<std::size_t>(dependency_progress)];
    if ((blocked.channel == Channel::Market &&
         (!blocked.guaranteed_fill ||
          state.cash < blocked.worst_case_cash_cost)) ||
        (blocked.action.kind == ActionKind::Plant &&
         !inventory_has(state.seed_inventory, blocked.action.item,
                        blocked.action.quantity)) ||
        (blocked.action.kind == ActionKind::PlaceAnimal &&
         !inventory_has(state.animal_inventory, blocked.action.item,
                        blocked.action.quantity)) ||
        (blocked.action.kind == ActionKind::Feed &&
         state.feed_inventory < blocked.action.quantity))
      return DebtReason::ResourceUnavailable;
    return DebtReason::Capacity;
  }
  const auto& next = obligation.transitions[static_cast<std::size_t>(progress)];
  if (next.channel == Channel::Unit &&
      (obligation.actor < 0 ||
       obligation.actor >= static_cast<int>(state.positions.size())))
    return DebtReason::UnsupportedIdentity;
  if (next.channel == Channel::Market &&
      (!next.guaranteed_fill || state.cash < next.worst_case_cash_cost))
    return DebtReason::ResourceUnavailable;
  if (next.action.kind == ActionKind::Plant &&
      !inventory_has(state.seed_inventory, next.action.item,
                     next.action.quantity))
    return DebtReason::ResourceUnavailable;
  if (next.action.kind == ActionKind::Pickup &&
      !inventory_has(state.shed_inventory, next.action.item,
                     next.action.quantity))
    return DebtReason::ResourceUnavailable;
  if (next.action.kind == ActionKind::PlaceAnimal &&
      !(obligation.actor >= 0 &&
        obligation.actor < static_cast<int>(state.actor_inventory.size()) &&
        inventory_has(
            state.actor_inventory[static_cast<std::size_t>(obligation.actor)],
            next.action.item, next.action.quantity)) &&
      !inventory_has(state.animal_inventory, next.action.item,
                     next.action.quantity))
    return DebtReason::ResourceUnavailable;
  if (next.action.kind == ActionKind::Feed &&
      !(obligation.actor >= 0 &&
        obligation.actor < static_cast<int>(state.actor_inventory.size()) &&
        next.action.item >= 0 &&
        inventory_has(
            state.actor_inventory[static_cast<std::size_t>(obligation.actor)],
            next.action.item, next.action.quantity)) &&
      state.feed_inventory < next.action.quantity)
    return DebtReason::ResourceUnavailable;
  if (next.channel == Channel::Unit && next.requires_position &&
      !route_contains(prepared, obligation.actor, next.tile))
    return DebtReason::UnreachableTile;
  return DebtReason::Capacity;
}

Candidate terminal_candidate(const Prepared& prepared, const State& state) {
  Candidate output;
  for (std::size_t actor = 0; actor < prepared.moves.size(); ++actor) {
    if (state.move_cursor[actor] !=
        static_cast<int>(prepared.moves[actor].size()))
      return output;
  }
  output.valid = true;
  for (std::size_t index = 0; index < prepared.obligations.size(); ++index) {
    const auto& obligation = prepared.obligations[index];
    const int progress = state.progress[index];
    const int transitions = static_cast<int>(obligation.transitions.size());
    if (progress == transitions) continue;
    output.objective.unfinished_must_finish += obligation.must_finish;
    output.objective.cross_day_cascade_risk += obligation.cascade_risk;
    output.objective.deferred_important += obligation.important;
    output.debts.push_back(
        {obligation.id, debt_reason(prepared, state, index), progress,
         transitions - progress, obligation.must_finish, obligation.important,
         obligation.cascade_risk, 1});
  }
  return output;
}

struct UnitOption {
  ScheduledAction scheduled;
  int obligation_index{-1};
  bool move_token{};
  int damage{};
  int inserted_move_distance{};
};

struct MarketOption {
  std::optional<ScheduledAction> scheduled;
  int obligation_index{-1};
  int damage{};
};

class Solver {
 public:
  Solver(const Prepared& prepared, bool memoized, std::size_t limit)
      : prepared_(prepared), memoized_(memoized), limit_(limit) {}

  Candidate solve(const State& state) {
    if (limit_hit_) return {};
    const auto key = memoized_ ? state_key(state) : std::string{};
    if (memoized_) {
      const auto found = memo_.find(key);
      if (found != memo_.end()) return found->second;
    }
    if (++explored_ > limit_) {
      limit_hit_ = true;
      return {};
    }
    const int remaining_slots = prepared_.request->horizon - state.tick;
    for (std::size_t actor = 0; actor < prepared_.moves.size(); ++actor) {
      const int remaining_moves =
          static_cast<int>(prepared_.moves[actor].size()) -
          state.move_cursor[actor];
      if (remaining_moves > remaining_slots) return remember(key, {});
    }
    if (state.tick == prepared_.request->horizon)
      return remember(key, terminal_candidate(prepared_, state));

    ScheduleSlot slot;
    slot.units.resize(state.positions.size());
    Candidate best;
    enumerate_actor(0, state, slot, 0, 0, best);
    return remember(key, best);
  }

  [[nodiscard]] std::size_t explored() const noexcept { return explored_; }
  [[nodiscard]] bool limit_hit() const noexcept { return limit_hit_; }

 private:
  Candidate remember(const std::string& key, Candidate value) {
    if (memoized_ && !limit_hit_) memo_.emplace(key, value);
    return value;
  }

  std::vector<UnitOption> unit_options(const State& state, int actor) const {
    std::vector<UnitOption> output;
    output.push_back({ScheduledAction{}, -1, false, 0, 0});
    const auto cursor = state.move_cursor[static_cast<std::size_t>(actor)];
    const auto& tokens = prepared_.moves[static_cast<std::size_t>(actor)];
    if (cursor < static_cast<int>(tokens.size()) &&
        state.tick >= tokens[static_cast<std::size_t>(cursor)].source_slot) {
      const auto& token = tokens[static_cast<std::size_t>(cursor)];
      output.push_back(
          {{token.action, token.source_slot, 0, -1}, -1, true,
           std::abs(state.tick - token.source_slot), 0});
    }
    for (std::size_t index = 0; index < prepared_.obligations.size(); ++index) {
      const auto& obligation = prepared_.obligations[index];
      const int progress = state.progress[index];
      if (obligation.actor != actor ||
          progress >= static_cast<int>(obligation.transitions.size()) ||
          state.tick < obligation.release_slot ||
          state.tick > obligation.deadline_slot ||
          !dependencies_complete(prepared_, state, obligation))
        continue;
      const auto& transition =
          obligation.transitions[static_cast<std::size_t>(progress)];
      if (transition.channel != Channel::Unit) continue;
      if (state.tick + 1 == prepared_.request->horizon &&
          receipt_lost_at_day_boundary(transition.action.kind))
        continue;
      State preview = state;
      if (!apply_transition_unit(prepared_, preview, actor, transition))
        continue;
      const int damage = obligation.source_slot >= 0
                             ? std::abs(state.tick - obligation.source_slot)
                             : 1;
      output.push_back(
          {{transition.action, -1, obligation.id, progress},
           static_cast<int>(index), false, damage,
           is_move(transition.action.kind) ? 1 : 0});
    }
    return output;
  }

  std::vector<MarketOption> market_options(const State& state) const {
    std::vector<MarketOption> output{{std::nullopt, -1, 0}};
    for (std::size_t index = 0; index < prepared_.obligations.size(); ++index) {
      const auto& obligation = prepared_.obligations[index];
      const int progress = state.progress[index];
      if (obligation.actor != -1 ||
          progress >= static_cast<int>(obligation.transitions.size()) ||
          state.tick < obligation.release_slot ||
          state.tick > obligation.deadline_slot ||
          !dependencies_complete(prepared_, state, obligation))
        continue;
      const auto& transition =
          obligation.transitions[static_cast<std::size_t>(progress)];
      State preview = state;
      if (!apply_market(preview, transition)) continue;
      const int damage = obligation.source_slot >= 0
                             ? std::abs(state.tick - obligation.source_slot)
                             : 1;
      output.push_back(
          {ScheduledAction{transition.action, -1, obligation.id, progress},
           static_cast<int>(index), damage});
    }
    return output;
  }

  void consider(Candidate candidate, Candidate& best) const {
    if (!candidate.valid) return;
    if (!best.valid || objective_less(candidate.objective, best.objective))
      best = std::move(candidate);
  }

  void enumerate_market(const State& state, const ScheduleSlot& units,
                        int inserted_move_distance, int unit_damage,
                        Candidate& best) {
    for (const auto& option : market_options(state)) {
      State next = state;
      ScheduleSlot current = units;
      if (option.scheduled) {
        const auto index = static_cast<std::size_t>(option.obligation_index);
        const auto progress = next.progress[index];
        const auto& transition =
            prepared_.obligations[index].transitions[
                static_cast<std::size_t>(progress)];
        if (!apply_market(next, transition)) continue;
        ++next.progress[index];
        current.market = option.scheduled;
      }
      ++next.tick;
      auto child = solve(next);
      if (!child.valid) continue;
      add_damage(child.objective, inserted_move_distance, 0,
                 unit_damage + option.damage);
      child.slots.insert(child.slots.begin(), std::move(current));
      consider(std::move(child), best);
      if (limit_hit_) return;
    }
  }

  void enumerate_actor(std::size_t actor, const State& state,
                       ScheduleSlot slot, int inserted_move_distance,
                       int unit_damage, Candidate& best) {
    if (actor == state.positions.size()) {
      enumerate_market(state, slot, inserted_move_distance, unit_damage, best);
      return;
    }
    for (const auto& option : unit_options(state, static_cast<int>(actor))) {
      State next = state;
      if (option.move_token) {
        if (!apply_unit(prepared_, next, static_cast<int>(actor),
                        option.scheduled.action))
          continue;
        ++next.move_cursor[actor];
      } else if (option.obligation_index >= 0) {
        const auto index = static_cast<std::size_t>(option.obligation_index);
        const auto progress = next.progress[index];
        const auto& transition =
            prepared_.obligations[index].transitions[
                static_cast<std::size_t>(progress)];
        if (!apply_transition_unit(prepared_, next, static_cast<int>(actor),
                                   transition))
          continue;
        ++next.progress[index];
      }
      slot.units[actor] = option.scheduled;
      if (option.move_token) {
        Candidate branch;
        enumerate_actor(actor + 1, next, slot, inserted_move_distance,
                        unit_damage, branch);
        if (branch.valid)
          branch.objective.move_slot_displacement += option.damage;
        consider(std::move(branch), best);
      } else {
        enumerate_actor(actor + 1, next, slot,
                        inserted_move_distance +
                            option.inserted_move_distance,
                        unit_damage + option.damage, best);
      }
      if (limit_hit_) return;
    }
  }

  const Prepared& prepared_;
  bool memoized_{};
  std::size_t limit_{};
  std::size_t explored_{};
  bool limit_hit_{};
  std::unordered_map<std::string, Candidate> memo_;
};

State initial_state(const Prepared& prepared) {
  const auto& snapshot = prepared.request->day_start;
  State state;
  state.positions = snapshot.actor_positions;
  state.tiles = snapshot.tiles;
  state.seed_inventory = snapshot.seed_inventory;
  state.animal_inventory = snapshot.animal_inventory;
  state.shed_inventory = snapshot.shed_inventory;
  state.actor_inventory = snapshot.actor_inventory;
  std::size_t inventory_width = state.shed_inventory.size();
  inventory_width = std::max(inventory_width, state.seed_inventory.size());
  inventory_width = std::max(inventory_width, state.animal_inventory.size());
  if (state.actor_inventory.empty())
    state.actor_inventory.assign(snapshot.actor_positions.size(),
                                 std::vector<int>(inventory_width));
  state.feed_inventory = snapshot.feed_inventory;
  state.cash = snapshot.cash;
  state.move_cursor.resize(snapshot.actor_positions.size());
  state.progress.resize(prepared.obligations.size());
  return state;
}

PlanResult run_solver(const Request& request, bool memoized,
                      std::size_t limit) {
  PlanResult output;
  const auto prepared = prepare(request);
  if (!prepared.valid) {
    output.status = PlanStatus::InvalidRequest;
    output.diagnostic = prepared.diagnostic;
    return output;
  }
  if (!prepared.hard_move_feasible) {
    output.status = PlanStatus::HardMoveInfeasible;
    output.diagnostic = prepared.diagnostic;
    return output;
  }
  Solver solver(prepared, memoized, limit);
  auto candidate = solver.solve(initial_state(prepared));
  output.explored_states = solver.explored();
  if (solver.limit_hit()) {
    output.status = PlanStatus::SearchLimit;
    output.diagnostic = "exact search state limit reached";
    return output;
  }
  if (!candidate.valid) {
    output.status = PlanStatus::HardMoveInfeasible;
    output.diagnostic = "no schedule closes every immutable MOVE token";
    return output;
  }
  output.status = PlanStatus::Planned;
  output.objective = candidate.objective;
  output.slots = std::move(candidate.slots);
  output.debts = std::move(candidate.debts);
  output.diagnostic = memoized ? "memoized exact optimum" :
                                 "exhaustive exact optimum";
  return output;
}

}  // namespace

bool is_move(ActionKind kind) noexcept {
  return kind == ActionKind::North || kind == ActionKind::South ||
         kind == ActionKind::East || kind == ActionKind::West;
}

bool is_market(ActionKind kind) noexcept {
  return kind == ActionKind::BuySeed || kind == ActionKind::BuyAnimal ||
         kind == ActionKind::BuyFeed;
}

const char* action_kind_name(ActionKind kind) noexcept {
  switch (kind) {
    case ActionKind::Pass: return "pass";
    case ActionKind::North: return "north";
    case ActionKind::South: return "south";
    case ActionKind::East: return "east";
    case ActionKind::West: return "west";
    case ActionKind::Dig: return "dig";
    case ActionKind::BuildPasture: return "build_pasture";
    case ActionKind::BuildCoop: return "build_coop";
    case ActionKind::Pickup: return "pickup";
    case ActionKind::Plant: return "plant";
    case ActionKind::PlaceAnimal: return "place_animal";
    case ActionKind::Feed: return "feed";
    case ActionKind::Care: return "care";
    case ActionKind::BuySeed: return "buy_seed";
    case ActionKind::BuyAnimal: return "buy_animal";
    case ActionKind::BuyFeed: return "buy_feed";
    case ActionKind::Water: return "water";
    case ActionKind::CollectFertilizer: return "collect_fertilizer";
  }
  return "unknown";
}

TypedObligation make_weed_pasture_obligation(
    std::uint64_t id, int actor, Position tile, int release_slot,
    bool must_finish, bool important, int cascade_risk) {
  return {id,
          ObligationKind::WeedPastureRecovery,
          actor,
          release_slot,
          23,
          -1,
          must_finish,
          important,
          cascade_risk,
          {},
          {{Channel::Unit, {ActionKind::Dig, -1, 1}, true, tile, false, 0},
           {Channel::Unit, {ActionKind::BuildPasture, -1, 1}, true, tile,
            false, 0}}};
}

TypedObligation make_replant_obligation(
    std::uint64_t id, int actor, Position tile, int seed_item,
    bool dig_first, int release_slot, bool must_finish, bool important,
    int cascade_risk) {
  TypedObligation result{id, ObligationKind::ReplantRecovery, actor,
                         release_slot, 23, -1, must_finish, important,
                         cascade_risk, {}, {}};
  if (dig_first)
    result.transitions.push_back(
        {Channel::Unit, {ActionKind::Dig, -1, 1}, true, tile, false, 0});
  result.transitions.push_back(
      {Channel::Unit, {ActionKind::Plant, seed_item, 1}, true, tile, false, 0});
  return result;
}

TypedObligation make_seed_purchase_retry(
    std::uint64_t id, int seed_item, int quantity, int release_slot,
    int worst_case_cash_cost, bool guaranteed_fill, bool must_finish,
    bool important, int cascade_risk) {
  return {id,
          ObligationKind::PurchaseRetry,
          -1,
          release_slot,
          23,
          -1,
          must_finish,
          important,
          cascade_risk,
          {},
          {{Channel::Market, {ActionKind::BuySeed, seed_item, quantity}, false,
            {}, guaranteed_fill, worst_case_cash_cost}}};
}

TypedObligation make_animal_purchase_retry(
    std::uint64_t id, int animal_item, int quantity, int release_slot,
    int worst_case_cash_cost, bool guaranteed_fill, bool must_finish,
    bool important, int cascade_risk) {
  return {id,
          ObligationKind::PurchaseRetry,
          -1,
          release_slot,
          23,
          -1,
          must_finish,
          important,
          cascade_risk,
          {},
          {{Channel::Market, {ActionKind::BuyAnimal, animal_item, quantity},
            false, {}, guaranteed_fill, worst_case_cash_cost}}};
}

TypedObligation make_feed_purchase_retry(
    std::uint64_t id, int quantity, int release_slot,
    int worst_case_cash_cost, bool guaranteed_fill, bool must_finish,
    bool important, int cascade_risk) {
  return {id,
          ObligationKind::PurchaseRetry,
          -1,
          release_slot,
          23,
          -1,
          must_finish,
          important,
          cascade_risk,
          {},
          {{Channel::Market, {ActionKind::BuyFeed, 0, quantity}, false, {},
            guaranteed_fill, worst_case_cash_cost}}};
}

TypedObligation make_animal_recovery_obligation(
    std::uint64_t id, int actor, Position tile, int animal_item,
    bool build_pasture_first, bool feed_after_place, bool care_after_place,
    int release_slot, bool must_finish, bool important, int cascade_risk,
    std::vector<std::uint64_t> dependencies) {
  TypedObligation result{id, ObligationKind::AnimalRecovery, actor,
                         release_slot, 23, -1, must_finish, important,
                         cascade_risk, std::move(dependencies), {}};
  if (build_pasture_first)
    result.transitions.push_back(
        {Channel::Unit, {ActionKind::BuildPasture, -1, 1}, true, tile, false,
         0});
  result.transitions.push_back(
      {Channel::Unit, {ActionKind::PlaceAnimal, animal_item, 1}, true, tile,
       false, 0});
  if (feed_after_place)
    result.transitions.push_back(
        {Channel::Unit, {ActionKind::Feed, -1, 1}, true, tile, false, 0});
  if (care_after_place)
    result.transitions.push_back(
        {Channel::Unit, {ActionKind::Care, animal_item, 1}, true, tile, false,
         0});
  return result;
}

bool objective_less(const Objective& left, const Objective& right) noexcept {
  return std::tie(left.unfinished_must_finish,
                  left.cross_day_cascade_risk, left.deferred_important,
                  left.inserted_move_distance,
                  left.move_slot_displacement,
                  left.production_slot_disturbance) <
         std::tie(right.unfinished_must_finish,
                  right.cross_day_cascade_risk, right.deferred_important,
                  right.inserted_move_distance,
                  right.move_slot_displacement,
                  right.production_slot_disturbance);
}

PlanResult select_minimum_damage(const Request& request) {
  return run_solver(request, true, request.maximum_states);
}

PlanResult exhaustive_oracle_for_testing(const Request& request,
                                         std::size_t maximum_nodes) {
  return run_solver(request, false, maximum_nodes);
}

VerifyResult verify_schedule(const Request& request, const PlanResult& plan) {
  VerifyResult output;
  if (!plan.planned()) {
    output.diagnostic = "plan status is not planned";
    return output;
  }
  const auto prepared = prepare(request);
  if (!prepared.valid || !prepared.hard_move_feasible ||
      plan.slots.size() != static_cast<std::size_t>(request.horizon)) {
    output.diagnostic = "request/slot envelope invalid";
    return output;
  }
  State state = initial_state(prepared);
  Objective damage;
  for (int tick = 0; tick < request.horizon; ++tick) {
    const auto& slot = plan.slots[static_cast<std::size_t>(tick)];
    if (slot.units.size() != state.positions.size()) {
      output.diagnostic = "unit manifest actor count mismatch";
      return output;
    }
    for (std::size_t actor = 0; actor < slot.units.size(); ++actor) {
      const auto& scheduled = slot.units[actor];
      if (scheduled.raw_source_slot >= 0) {
        const int cursor = state.move_cursor[actor];
        const auto& tokens = prepared.moves[actor];
        if (scheduled.obligation_id != 0 ||
            cursor >= static_cast<int>(tokens.size()) ||
            tokens[static_cast<std::size_t>(cursor)].source_slot !=
                scheduled.raw_source_slot ||
            scheduled.raw_source_slot > tick ||
            tokens[static_cast<std::size_t>(cursor)].action !=
                scheduled.action ||
            !apply_unit(prepared, state, static_cast<int>(actor),
                        scheduled.action)) {
          output.diagnostic = "MOVE token identity/order/boundary violation";
          return output;
        }
        ++state.move_cursor[actor];
        ++output.checked_move_tokens;
        damage.move_slot_displacement +=
            std::abs(tick - scheduled.raw_source_slot);
      } else if (scheduled.obligation_id != 0) {
        const auto found =
            prepared.obligation_by_id.find(scheduled.obligation_id);
        if (found == prepared.obligation_by_id.end()) {
          output.diagnostic = "unknown unit obligation binding";
          return output;
        }
        const auto index = static_cast<std::size_t>(found->second);
        const auto& obligation = prepared.obligations[index];
        const int progress = state.progress[index];
        if (obligation.actor != static_cast<int>(actor) ||
            progress != scheduled.transition_index ||
            progress >= static_cast<int>(obligation.transitions.size()) ||
            tick < obligation.release_slot || tick > obligation.deadline_slot ||
            (tick + 1 == request.horizon &&
             receipt_lost_at_day_boundary(scheduled.action.kind)) ||
            !dependencies_complete(prepared, state, obligation) ||
            obligation.transitions[static_cast<std::size_t>(progress)].action !=
                scheduled.action ||
            !apply_transition_unit(
                prepared, state, static_cast<int>(actor),
                obligation.transitions[static_cast<std::size_t>(progress)])) {
          output.diagnostic = "unit obligation transition violation";
          return output;
        }
        ++state.progress[index];
        damage.inserted_move_distance +=
            is_move(scheduled.action.kind) ? 1 : 0;
        damage.production_slot_disturbance +=
            obligation.source_slot >= 0
                ? std::abs(tick - obligation.source_slot)
                : 1;
      } else if (scheduled.action.kind != ActionKind::Pass) {
        output.diagnostic = "unbound non-PASS unit action";
        return output;
      }
    }
    if (slot.market) {
      const auto& scheduled = *slot.market;
      const auto found =
          prepared.obligation_by_id.find(scheduled.obligation_id);
      if (scheduled.raw_source_slot >= 0 ||
          found == prepared.obligation_by_id.end()) {
        output.diagnostic = "unknown market binding";
        return output;
      }
      const auto index = static_cast<std::size_t>(found->second);
      const auto& obligation = prepared.obligations[index];
      const int progress = state.progress[index];
      if (obligation.actor != -1 || progress != scheduled.transition_index ||
          progress >= static_cast<int>(obligation.transitions.size()) ||
          tick < obligation.release_slot || tick > obligation.deadline_slot ||
          !dependencies_complete(prepared, state, obligation) ||
          obligation.transitions[static_cast<std::size_t>(progress)].action !=
              scheduled.action ||
          !apply_market(
              state,
              obligation.transitions[static_cast<std::size_t>(progress)])) {
        output.diagnostic = "market obligation transition violation";
        return output;
      }
      ++state.progress[index];
      damage.production_slot_disturbance +=
          obligation.source_slot >= 0
              ? std::abs(tick - obligation.source_slot)
              : 1;
    }
    ++state.tick;
  }
  auto terminal = terminal_candidate(prepared, state);
  if (!terminal.valid) {
    output.diagnostic = "not every MOVE token closed";
    return output;
  }
  terminal.objective.move_slot_displacement +=
      damage.move_slot_displacement;
  terminal.objective.inserted_move_distance +=
      damage.inserted_move_distance;
  terminal.objective.production_slot_disturbance +=
      damage.production_slot_disturbance;
  output.recomputed_objective = terminal.objective;
  if (terminal.objective != plan.objective ||
      terminal.debts.size() != plan.debts.size()) {
    output.diagnostic = "objective/debt certificate mismatch";
    return output;
  }
  for (std::size_t index = 0; index < terminal.debts.size(); ++index) {
    const auto& left = terminal.debts[index];
    const auto& right = plan.debts[index];
    if (left.obligation_id != right.obligation_id ||
        left.reason != right.reason ||
        left.completed_transitions != right.completed_transitions ||
        left.remaining_transitions != right.remaining_transitions ||
        right.bound_day_offset != 1) {
      output.diagnostic = "day-bound debt identity mismatch";
      return output;
    }
  }
  output.valid = true;
  output.diagnostic = "verified exact MOVE closure and typed schedule";
  return output;
}

const char* debt_reason_name(DebtReason reason) noexcept {
  switch (reason) {
    case DebtReason::Capacity: return "capacity";
    case DebtReason::ResourceUnavailable: return "resource_unavailable";
    case DebtReason::UnreachableTile: return "unreachable_tile";
    case DebtReason::UnsupportedIdentity: return "unsupported_identity";
    case DebtReason::PartialDayBound: return "partial_day_bound";
  }
  return "unknown";
}

const char* plan_status_name(PlanStatus status) noexcept {
  switch (status) {
    case PlanStatus::Planned: return "planned";
    case PlanStatus::InvalidRequest: return "invalid_request";
    case PlanStatus::HardMoveInfeasible: return "hard_move_infeasible";
    case PlanStatus::SearchLimit: return "search_limit";
  }
  return "unknown";
}

}  // namespace g001::minimum_damage
