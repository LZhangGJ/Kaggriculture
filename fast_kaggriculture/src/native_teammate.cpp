// Licensed under the Apache License, Version 2.0.
#include "native_teammate.hpp"

// A number of standalone C++ evaluators compile native_teammate.cpp directly
// and do not consume a shared native-core CMake target.  Embed the one canonical
// implementation here so the default-off backend links for every such
// consumer; jointFixedMoveOracleCpp builds the same source in its own target.
#include "../../native_deps/findNewRoad/jointFixedMoveOracleCpp/src/reactive_production.cpp"
#include "../../native_deps/findNewRoad/repairMechanismCpp/failureDebtSchedulerCpp/src/deferred_crop_scheduler.cpp"
// Plot state, objective coalescing, and receipt leases are embedded in this
// submission TU so standalone evaluators exercise the same implementation.
#include "../../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/src/event_triggered_local_repair.cpp"
#include "../../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/src/day_horizon_planner.cpp"
#include "../../native_deps/findNewRoad/repairMechanismCpp/purchaseRecoveryLedgerCpp/src/purchase_recovery_ledger.cpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace fastkag {
namespace {

constexpr std::array<int, 4> PREMIUM{3, 4, 6, 7};
constexpr std::array<int, 3> K320_PREMIUM{3, 6, 7};
constexpr std::array<int, 9> LIQUIDATION{1, 5, 8, 4, 6, 3, 2, 0, 7};
constexpr std::array<int, 9> ROOM_PRIORITY{7, 6, 5, 4, 3, 2, 1, 8, 0};
constexpr std::array<Position, 4> SHED_ACCESS{{{4, 4}, {5, 4}, {5, 5}, {4, 5}}};

int premium_slot(int item) {
  for (int i = 0; i < 4; ++i) if (PREMIUM[i] == item) return i;
  return -1;
}

int quantity(const Action& a) { return std::max(0, int(a.quantity)); }
bool sell(const Action& a) {
  return a.op == Op::SELL && int(a.item) >= 0 && int(a.item) < N_PRODUCTS;
}
bool at(Position a, Position b) { return a.x == b.x && a.y == b.y; }
int distance(Position a, Position b) {
  return std::abs(int(a.x) - int(b.x)) + std::abs(int(a.y) - int(b.y));
}
bool shed_adjacent(Position p) {
  return (p.x == 4 || p.x == 5) && (p.y == 4 || p.y == 5);
}

int shed_sum(const PrivateState& p) {
  return std::accumulate(p.shed.begin(), p.shed.end(), 0);
}

PlayerAction aligned(PlayerAction a, const Simulator& env, int player) {
  const size_t expected = env.farms()[player].hands.size() + 1;
  a.units.resize(expected);
  return a;
}

const Tile* tile_at(const Simulator& env, int player, Position p) {
  const int n = env.config().board_size;
  if (p.x < 0 || p.y < 0 || p.x >= n || p.y >= n) return nullptr;
  return &env.farms()[player].tiles[int(p.y) * n + int(p.x)];
}

std::vector<Position> positions(const Simulator& env, int player) {
  std::vector<Position> out{env.farms()[player].farmer};
  out.insert(out.end(), env.farms()[player].hands.begin(),
             env.farms()[player].hands.end());
  return out;
}

bool same_action(const Action& left, const Action& right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool movement(Op operation) {
  return operation == Op::NORTH || operation == Op::SOUTH ||
         operation == Op::EAST || operation == Op::WEST;
}

std::uint64_t event_local_actor_generation(int day, std::size_t actor) {
  // Farmer slot zero persists for the whole match.  Farm-hand slots are
  // destroyed at midnight and may be numerically reused by a later hire.
  if (actor == 0) return 1;
  return (static_cast<std::uint64_t>(day + 1) << 32U) |
      static_cast<std::uint64_t>(actor + 1);
}

bool crop_item(Item item) {
  const int index = static_cast<int>(item);
  return index >= 0 && index < N_CROPS;
}

namespace event_local = g001::event_local_repair;

event_local::Action to_event_local(Action action) {
  event_local::Action result;
  result.item = static_cast<int>(action.item);
  result.quantity = action.quantity;
  result.arg0 = static_cast<int>(action.op);
  switch (action.op) {
    case Op::PASS: result.op = event_local::Op::Pass; break;
    case Op::NORTH:
    case Op::SOUTH:
    case Op::EAST:
    case Op::WEST: result.op = event_local::Op::Move; break;
    case Op::DIG: result.op = event_local::Op::Dig; break;
    case Op::PLANT: result.op = event_local::Op::Plant; break;
    case Op::BUILD_COOP:
    case Op::BUILD_PASTURE:
      result.op = event_local::Op::Build;
      result.item = static_cast<int>(action.op);
      break;
    case Op::WATER: result.op = event_local::Op::Water; break;
    case Op::HARVEST: result.op = event_local::Op::Harvest; break;
    default: result.op = event_local::Op::Other; break;
  }
  return result;
}

Action from_event_local(const event_local::Action& action) {
  Action result;
  result.item = static_cast<Item>(action.item);
  result.quantity = action.quantity;
  switch (action.op) {
    case event_local::Op::Pass: result.op = Op::PASS; break;
    case event_local::Op::Move:
      result.op = static_cast<Op>(action.arg0);
      if (!movement(result.op))
        throw std::runtime_error("local repair lost MOVE direction");
      break;
    case event_local::Op::Dig:
      result.op = Op::DIG;
      result.item = Item::NONE;
      break;
    case event_local::Op::Plant: result.op = Op::PLANT; break;
    case event_local::Op::Build:
      result.op = static_cast<Op>(action.item);
      result.item = Item::NONE;
      if (result.op != Op::BUILD_COOP && result.op != Op::BUILD_PASTURE)
        throw std::runtime_error("local repair lost BUILD kind");
      break;
    case event_local::Op::Water:
      result.op = Op::WATER;
      if (!crop_item(result.item)) result.item = Item::NONE;
      break;
    case event_local::Op::Harvest:
      result.op = Op::HARVEST;
      if (!crop_item(result.item)) result.item = Item::NONE;
      break;
    case event_local::Op::Other:
      result.op = static_cast<Op>(action.arg0);
      break;
  }
  return result;
}

event_local::Position to_event_local(Position position) {
  return {position.y, position.x};
}

event_local::TileObservation to_event_local(const Simulator& env,
                                             const Tile* tile) {
  event_local::TileObservation result;
  if (!tile) return result;
  switch (tile->kind) {
    case TileKind::EMPTY: result.kind = event_local::TileKind::Empty; break;
    case TileKind::WEED: result.kind = event_local::TileKind::Weed; break;
    case TileKind::PLANT: result.kind = event_local::TileKind::Crop; break;
    case TileKind::COOP:
    case TileKind::PASTURE: result.kind = event_local::TileKind::Structure; break;
    default: result.kind = event_local::TileKind::Other; break;
  }
  result.item = tile->kind == TileKind::PLANT
      ? static_cast<int>(tile->crop) : -1;
  result.watered_today = tile->watered_today;
  constexpr std::array<int, N_CROPS> first_day{2, 2, 8, 10, 10};
  const int crop = static_cast<int>(tile->crop);
  result.harvest_legal = tile->kind == TileKind::PLANT && crop >= 0 &&
      crop < N_CROPS && tile->yield_units > 0 &&
      env.day() - tile->planted_day >= first_day[static_cast<std::size_t>(crop)];
  return result;
}

int actor_inventory(const Simulator& env, int player, std::size_t actor) {
  const auto& inventories = env.privates()[player].inventories;
  return actor < inventories.size()
      ? std::accumulate(inventories[actor].begin(), inventories[actor].end(), 0)
      : 0;
}

int actor_item_inventory(const Simulator& env, int player, std::size_t actor,
                         Item item) {
  const int index = static_cast<int>(item);
  const auto& inventories = env.privates()[player].inventories;
  return actor < inventories.size() && index >= 0 && index < N_ITEMS
      ? inventories[actor][static_cast<std::size_t>(index)] : 0;
}

bool route_unit_effect(const Simulator& env, int player, std::size_t actor,
                       const NativeAgentState::PendingRouteCursorCommit& pending) {
  if (actor >= pending.final_units.size() ||
      actor >= pending.positions_before.size() ||
      actor >= pending.tiles_before.size() ||
      actor >= pending.inventory_before.size()) return false;
  const auto current_positions = positions(env, player);
  const auto& action = pending.final_units[actor];
  const auto before_position = pending.positions_before[actor];
  const auto& before_tile = pending.tiles_before[actor];
  const Tile* after_tile = tile_at(env, player, before_position);
  const bool crossed_day =
      env.day() > pending.submitted_step / env.config().turns_per_day;
  if (movement(action.op)) {
    if (crossed_day) {
      int dx = 0, dy = 0;
      dx += action.op == Op::EAST;
      dx -= action.op == Op::WEST;
      dy += action.op == Op::SOUTH;
      dy -= action.op == Op::NORTH;
      const int x = before_position.x + dx;
      const int y = before_position.y + dy;
      // The official simulator has no cross-player unit collision. At the day
      // boundary the actor is reset/despawned before the receipt observation,
      // so exact staged manifest + legal prestate edge is the only persistent
      // deterministic lower bound; an off-board MOVE remains unconfirmed.
      return x >= 0 && y >= 0 && x < env.config().board_size &&
             y < env.config().board_size;
    }
    return actor < current_positions.size() &&
           actor < pending.positions_expected.size() &&
           at(current_positions[actor], pending.positions_expected[actor]);
  }
  switch (action.op) {
    case Op::PASS: return true;
    case Op::DIG:
      return after_tile && after_tile->kind == TileKind::EMPTY;
    case Op::PLANT:
      return after_tile && after_tile->kind == TileKind::PLANT &&
             after_tile->crop == action.item;
    case Op::WATER:
      return after_tile && after_tile->kind == TileKind::PLANT &&
             (after_tile->watered_today ||
              (crossed_day && before_tile.kind == TileKind::PLANT &&
               after_tile->crop == before_tile.crop &&
               after_tile->consecutive_unwatered == 0));
    case Op::HARVEST:
      return after_tile &&
             ((before_tile.kind == TileKind::PLANT &&
               (after_tile->kind != TileKind::PLANT ||
                after_tile->yield_units < before_tile.yield_units)) ||
              actor_inventory(env, player, actor) >
                  pending.inventory_before[actor]);
    case Op::BUILD_COOP:
      return after_tile && after_tile->kind == TileKind::COOP;
    case Op::BUILD_PASTURE:
      return after_tile && after_tile->kind == TileKind::PASTURE;
    case Op::PLACE:
      if (int(action.item) >= 9)
        return after_tile && after_tile->kind == TileKind::ANIMAL &&
               after_tile->animal == action.item;
      return actor_inventory(env, player, actor) <
             pending.inventory_before[actor];
    case Op::FEED:
      return after_tile && after_tile->kind == TileKind::ANIMAL &&
             (after_tile->fed_today ||
              (crossed_day && before_tile.kind == TileKind::ANIMAL &&
               after_tile->animal == before_tile.animal &&
               after_tile->consecutive_unfed == 0));
    case Op::CARE:
      if (!after_tile || before_tile.kind != TileKind::ANIMAL ||
          before_tile.cared_today)
        return false;
      if (!crossed_day)
        return after_tile->kind == TileKind::ANIMAL &&
               after_tile->cared_today;
      if (!before_tile.fed_today ||
          actor >= pending.tiles_expected.size())
        return false;
      {
        const auto& expected = pending.tiles_expected[actor];
        return after_tile->kind == expected.kind &&
               after_tile->animal == expected.animal &&
               after_tile->yield_units == expected.yield_units &&
               after_tile->pending_care_bonus ==
                   expected.pending_care_bonus;
      }
    case Op::FERTILIZE:
      return after_tile && after_tile->kind == TileKind::PLANT &&
             after_tile->fertilized_until_day >= env.day();
    case Op::DROP:
      return actor_inventory(env, player, actor) < pending.inventory_before[actor];
    case Op::PICKUP:
    case Op::COLLECT_FERTILIZER:
      return actor_inventory(env, player, actor) > pending.inventory_before[actor];
    default:
      // No canonical observation adapter exists for other unit effects. Do not
      // advance a source cursor from action-byte equality alone.
      return false;
  }
}

void accumulate_route_cursor_audit(
    const joint_fixed_move_oracle::production::RouteCursorAudit& before,
    const joint_fixed_move_oracle::production::RouteCursorAudit& after,
    NativeRepairAudit* audit) {
  if (!audit) return;
  audit->route_cursor_inserted_before_move +=
      after.inserted_before_move - before.inserted_before_move;
  audit->route_cursor_skipped_nonmoves +=
      after.skipped_nonmoves - before.skipped_nonmoves;
  audit->route_cursor_forced_moves += after.forced_moves - before.forced_moves;
  audit->route_cursor_deferred_nonmoves +=
      after.deferred_nonmoves - before.deferred_nonmoves;
  audit->route_cursor_weed_events +=
      after.production.weed_events - before.production.weed_events;
  audit->route_cursor_digs += after.production.digs - before.production.digs;
  audit->route_cursor_plants +=
      after.production.plants - before.production.plants;
  audit->route_cursor_waters +=
      after.production.waters - before.production.waters;
  audit->route_cursor_harvests +=
      after.production.harvests - before.production.harvests;
  audit->route_cursor_abandoned_cycles +=
      after.production.abandoned_cycles - before.production.abandoned_cycles;
  audit->route_cursor_completed_cycles +=
      after.production.completed_cycles - before.production.completed_cycles;
}

void settle_route_cursor_commit(const Simulator& env, int player,
                                NativeAgentState& state,
                                NativeRepairAudit* audit) {
  auto& pending = state.experimental_route_cursor_pending;
  if (!pending.active) return;
  const bool adjacent_observation =
      env.step_count() == pending.submitted_step + 1;
  std::vector<bool> confirmed(pending.final_units.size(),
                              adjacent_observation);
  bool all_confirmed = adjacent_observation;
  for (std::size_t actor = 0; actor < pending.final_units.size(); ++actor) {
    confirmed[actor] = adjacent_observation &&
                       route_unit_effect(env, player, actor, pending);
    all_confirmed = all_confirmed && confirmed[actor];
    if (!confirmed[actor] && audit) {
      const int operation = static_cast<int>(pending.final_units[actor].op);
      if (operation >= 0 && operation < static_cast<int>(
              audit->route_cursor_effect_failures_by_op.size()))
        ++audit->route_cursor_effect_failures_by_op[
            static_cast<std::size_t>(operation)];
      if (audit->route_cursor_first_effect_failure_step < 0) {
        audit->route_cursor_first_effect_failure_step = pending.submitted_step;
        audit->route_cursor_first_effect_failure_actor =
            static_cast<int>(actor);
        audit->route_cursor_first_effect_failure_op = operation;
      }
    }
  }
  const auto before = state.experimental_route_cursor_audit;
  bool committed = false;
  if (all_confirmed) {
    committed =
        joint_fixed_move_oracle::production::commit_reactive_route_cursor(
            pending.proposal, pending.final_units, true,
            state.experimental_route_cursor,
            state.experimental_route_cursor_audit);
  } else {
    // A failed production receipt invalidates the entire tentative production
    // transition. Successful actors may still commit an independently proven
    // ordered source MOVE. This prevents one worker's invalid non-MOVE from
    // replaying another worker's already-executed route edge.
    auto fallback = pending.source_base;
    fallback.action.units = pending.final_units;
    fallback.next_audit = fallback.prior_audit;
    int matched_moves = 0;
    int newly_deferred = 0;
    bool production_changed = false;
    const auto actors = std::min(
        {confirmed.size(), fallback.next_state.source_cursor.size(),
         pending.source_final.next_state.source_cursor.size()});
    for (std::size_t actor = 0; actor < actors; ++actor) {
      if (!confirmed[actor]) continue;
      const bool exact_actor = pending.exact_production_manifest &&
          actor < pending.proposal.action.units.size() &&
          same_action(pending.final_units[actor],
                      pending.proposal.action.units[actor]);
      const auto& actor_transition =
          exact_actor ? pending.proposal : pending.source_final;
      if (!exact_actor &&
          (!movement(pending.final_units[actor].op) ||
           fallback.next_state.source_cursor[actor] ==
               actor_transition.next_state.source_cursor[actor]))
        continue;
      const auto base_deferred =
          actor < fallback.next_state.deferred_nonmoves.size()
              ? fallback.next_state.deferred_nonmoves[actor].size()
              : 0;
      const auto final_deferred =
          actor < actor_transition.next_state.deferred_nonmoves.size()
              ? actor_transition.next_state.deferred_nonmoves[actor].size()
              : 0;
      fallback.next_state.source_cursor[actor] =
          actor_transition.next_state.source_cursor[actor];
      if (actor < fallback.next_state.deferred_nonmoves.size() &&
          actor < actor_transition.next_state.deferred_nonmoves.size())
        fallback.next_state.deferred_nonmoves[actor] =
            actor_transition.next_state.deferred_nonmoves[actor];
      newly_deferred +=
          static_cast<int>(final_deferred - base_deferred);
      matched_moves += movement(pending.final_units[actor].op);

      if (actor >= pending.positions_before.size()) continue;
      const auto position = pending.positions_before[actor];
      if (position.x < 0 || position.y < 0 ||
          position.x >= env.config().board_size ||
          position.y >= env.config().board_size)
        continue;
      const std::size_t tile = static_cast<std::size_t>(
          position.y * env.config().board_size + position.x);
      auto& target_production = fallback.next_state.production;
      const auto& source_production = actor_transition.next_state.production;
      if (tile < target_production.debts.size() &&
          tile < source_production.debts.size())
        target_production.debts[tile] = source_production.debts[tile];
      if (tile < target_production.last_crop.size() &&
          tile < source_production.last_crop.size())
        target_production.last_crop[tile] = source_production.last_crop[tile];
      target_production.last_step = pending.submitted_step;
      production_changed = true;

      auto& production_audit = fallback.next_audit.production;
      switch (pending.final_units[actor].op) {
        case Op::DIG: ++production_audit.digs; break;
        case Op::PLANT: ++production_audit.plants; break;
        case Op::WATER: ++production_audit.waters; break;
        case Op::HARVEST: {
          ++production_audit.harvests;
          const auto& before_debts = fallback.prior_state.production.debts;
          const auto& after_debts = fallback.next_state.production.debts;
          if (tile < before_debts.size() && tile < after_debts.size() &&
              before_debts[tile].active &&
              before_debts[tile].confirmed_harvests == 1 &&
              !after_debts[tile].active)
            ++production_audit.completed_cycles;
          break;
        }
        case Op::FERTILIZE: ++production_audit.fertilizes; break;
        case Op::BUILD_COOP:
        case Op::BUILD_PASTURE: ++production_audit.structures; break;
        default: break;
      }
    }
    // Preserve deterministic day-boundary deferrals already present in base.
    fallback.next_audit.deferred_nonmoves =
        pending.source_base.next_audit.deferred_nonmoves + newly_deferred;
    fallback.next_audit.skipped_nonmoves =
        pending.source_base.next_audit.skipped_nonmoves + newly_deferred;
    fallback.next_audit.forced_moves =
        pending.source_base.next_audit.forced_moves + matched_moves;
    const bool state_changed =
        fallback.next_state.day != fallback.prior_state.day ||
        fallback.next_state.source_cursor !=
            fallback.prior_state.source_cursor ||
        fallback.next_audit.deferred_nonmoves !=
            fallback.prior_audit.deferred_nonmoves || production_changed;
    if (state_changed)
      committed =
          joint_fixed_move_oracle::production::commit_reactive_route_cursor(
              fallback, pending.final_units, true,
              state.experimental_route_cursor,
              state.experimental_route_cursor_audit);
  }
  if (committed) {
    accumulate_route_cursor_audit(before,
                                  state.experimental_route_cursor_audit, audit);
    if (audit) ++audit->route_cursor_commits;
  }
  if (!all_confirmed && audit) {
    ++audit->route_cursor_effect_failures;
  }
  pending = {};
}

void settle_route_cursor_purchases(const Simulator& env, int player,
                                   NativeAgentState& state,
                                   NativeRepairAudit* audit) {
  const auto& fills = env.last_market_fills()[player];
  for (const auto& pending : state.experimental_purchase_receipts) {
    if (pending.operation != Op::BUY_SEED) continue;
    const int filled = pending.market_slot >= 0 &&
            pending.market_slot < static_cast<int>(fills.size())
        ? std::min(pending.requested,
                   std::max(0, static_cast<int>(fills[pending.market_slot])))
        : 0;
    for (const int tile : pending.debt_tiles) {
      auto& debts = state.experimental_route_cursor.production.debts;
      if (tile < 0 || tile >= static_cast<int>(debts.size())) continue;
      auto& debt = debts[static_cast<std::size_t>(tile)];
      if (!debt.active || debt.structure || debt.crop != pending.item) continue;
      ++debt.seed_retry_attempts;
      debt.last_seed_retry_step = pending.submitted_step;
    }
    if (audit) {
      ++audit->route_cursor_seed_retry_orders;
      audit->route_cursor_seed_retry_units += pending.requested;
      audit->route_cursor_seed_retry_fills += filled;
    }
  }
  state.experimental_purchase_receipts.clear();
}

void observe_route_crop_memory(const Simulator& env, int player,
                               NativeAgentState& state) {
  auto& production = state.experimental_route_cursor.production;
  const auto& tiles = env.farms()[player].tiles;
  if (production.debts.size() != tiles.size())
    production.reset(static_cast<int>(tiles.size()));
  for (std::size_t tile = 0; tile < tiles.size(); ++tile) {
    const int crop = static_cast<int>(tiles[tile].crop);
    if (tiles[tile].kind == TileKind::PLANT && crop >= 0 && crop < N_CROPS)
      production.last_crop[tile] = tiles[tile].crop;
  }
}

g001::failure_debt::deferred_crop::CropSnapshot crop_snapshot(
    const Simulator& env, int player, Position position) {
  g001::failure_debt::deferred_crop::CropSnapshot result;
  const Tile* tile = tile_at(env, player, position);
  if (!tile) return result;
  result.kind = tile->kind;
  result.crop = tile->crop;
  result.planted_day = tile->planted_day;
  result.yield_units = tile->yield_units;
  result.fertilized_until_day = tile->fertilized_until_day;
  result.watered_today = tile->watered_today;
  constexpr std::array<int, N_CROPS> first_day{2, 2, 8, 10, 10};
  const int crop = static_cast<int>(tile->crop);
  result.harvest_legal = tile->kind == TileKind::PLANT && crop >= 0 &&
      crop < N_CROPS && tile->yield_units > 0 &&
      env.day() - tile->planted_day >= first_day[static_cast<std::size_t>(crop)];
  return result;
}

void install_event_local_day(const std::vector<PlayerAction>& tape,
                             const Simulator& env,
                             NativeAgentState& state) {
  if (!state.experimental_event_local_repair)
    state.experimental_event_local_repair.emplace(
        event_local::Config{true, 0, 2});
  if (state.experimental_event_local_day == env.day()) return;

  const int turns = env.config().turns_per_day;
  std::size_t actors = 1;
  for (const auto& frame : tape) actors = std::max(actors, frame.units.size());
  event_local::DayPlan plan;
  plan.day = env.day();
  plan.turns = turns;
  plan.actors.resize(actors);
  const int day_begin = env.day() * turns;
  for (std::size_t actor = 0; actor < actors; ++actor) {
    auto& actor_plan = plan.actors[actor];
    actor_plan.actor = static_cast<int>(actor);
    actor_plan.turns.reserve(static_cast<std::size_t>(turns));
    for (int turn = 0; turn < turns; ++turn) {
      const int source = std::min(day_begin + turn,
                                  static_cast<int>(tape.size()) - 1);
      const Action action = actor < tape[static_cast<std::size_t>(source)].units.size()
          ? tape[static_cast<std::size_t>(source)].units[actor] : Action{};
      event_local::PlannedAction planned;
      planned.action = to_event_local(action);
      planned.certified_stationary_loss = action.op == Op::PASS ? 0 :
          std::numeric_limits<int>::max();
      actor_plan.turns.push_back(planned);
    }
  }
  state.experimental_event_local_repair->install_day(std::move(plan));
  state.experimental_event_local_day = env.day();
}

void remember_event_local_crops(const Simulator& env, int player,
                                NativeAgentState& state) {
  const auto& tiles = env.farms()[player].tiles;
  if (state.experimental_event_local_last_crop.size() != tiles.size())
    state.experimental_event_local_last_crop.assign(tiles.size(), Item::NONE);
  for (std::size_t index = 0; index < tiles.size(); ++index)
    if (tiles[index].kind == TileKind::PLANT && crop_item(tiles[index].crop))
      state.experimental_event_local_last_crop[index] = tiles[index].crop;
}

int planned_market_cash(const Simulator& env, int player,
                        const PlayerAction& action);

void settle_event_local_purchase_receipts(const Simulator& env, int player,
                                          NativeAgentState& state,
                                          NativeRepairAudit* audit) {
  g001::purchase_recovery::ReceiptObservation observation;
  observation.step = env.step_count();
  observation.slot_fills = env.last_market_fills()[player];
  observation.seeds_after = env.privates()[player].seeds;
  for (int animal = 0; animal < N_ANIMALS; ++animal)
    observation.animals_after[static_cast<std::size_t>(animal)] =
        env.privates()[player].shed[static_cast<std::size_t>(
            static_cast<int>(Item::GOOSE) + animal)];
  const auto settlements =
      state.experimental_event_local_purchase_ledger.observe(observation);
  if (audit) {
    for (const auto& settlement : settlements) {
      const auto debt = state.experimental_event_local_purchase_ledger.debt(
          settlement.debt_id);
      if (!debt || debt->obligation.operation != Op::BUY_SEED) continue;
      audit->local_repair_seed_fills += settlement.filled;
      audit->local_repair_seed_zero_fills +=
          settlement.status == g001::purchase_recovery::FillStatus::Zero;
    }
  }
  // InventoryOnly is intentionally the entire handoff contract.  Draining it
  // before decide() makes the newly observed inventory available to the unit
  // compiler, but never certifies PLANT or any later production effect.
  (void)state.experimental_event_local_purchase_ledger.drain_handoffs();
}

void settle_event_local_receipts(const Simulator& env, int player,
                                 NativeAgentState& state,
                                 NativeRepairAudit* audit) {
  if (!state.experimental_event_local_repair) return;
  for (const auto& pending : state.experimental_event_local_pending) {
    event_local::Receipt receipt;
    receipt.decision_id = pending.decision_id;
    receipt.actor = pending.actor;
    receipt.actor_generation = pending.actor_generation;
    receipt.tile = to_event_local(pending.tile);
    const Tile* after = tile_at(env, player, pending.tile);
    receipt.after = to_event_local(env, after);
    const int desired = static_cast<int>(pending.desired_item);
    receipt.desired_item_inventory_delta =
        desired >= 0 && desired < N_ITEMS
        ? actor_item_inventory(env, player,
                               static_cast<std::size_t>(pending.actor),
                               pending.desired_item) -
              pending.desired_inventory_before
        : 0;
    const bool crossed_day = env.day() >
        pending.submitted_step / env.config().turns_per_day;
    receipt.day_end_water_effect_lower_bound =
        pending.emitted == Op::WATER && crossed_day && after &&
        pending.before.kind == TileKind::PLANT &&
        after->kind == TileKind::PLANT &&
        after->crop == pending.before.crop &&
        after->consecutive_unwatered == 0;
    const bool adjacent = env.step_count() == pending.submitted_step + 1;
    const bool confirmed = adjacent &&
        state.experimental_event_local_repair->observe(receipt);
    if (audit) {
      if (confirmed) ++audit->local_repair_receipts_confirmed;
      else ++audit->local_repair_receipts_failed;
    }
  }
  state.experimental_event_local_pending.clear();
}

void compose_event_local_repair(const std::vector<PlayerAction>& tape,
                                const Simulator& env, int player,
                                NativeAgentState& state, PlayerAction& out,
                                NativeRepairAudit* audit,
                                bool owns_recovery_market_tail) {
  install_event_local_day(tape, env, state);
  remember_event_local_crops(env, player, state);
  event_local::TurnInput input;
  input.day = env.day();
  input.turn = env.hour();
  input.composed_baseline.reserve(out.units.size());
  input.actors.reserve(out.units.size());
  const auto current_positions = positions(env, player);
  state.experimental_event_local_actor_generations.resize(
      current_positions.size());
  for (std::size_t actor = 0; actor < current_positions.size(); ++actor)
    state.experimental_event_local_actor_generations[actor] =
        event_local_actor_generation(env.day(), actor);
  const auto source_step = static_cast<std::size_t>(std::min(
      env.step_count(), static_cast<int>(tape.size()) - 1));
  const auto& source_units = tape[source_step].units;

  // Plot transactions persist independently of farm-hand lifetime. Actor and
  // generation are only receipt-bound execution leases selected below.
  auto open_transactions =
      state.experimental_event_local_repair->open_transactions();
  const auto& seed_inventory = env.privates()[player].seeds;
  std::array<int, N_CROPS> baseline_plant_demand{};
  for (const auto& action : out.units)
    if (action.op == Op::PLANT && crop_item(action.item))
      ++baseline_plant_demand[static_cast<std::size_t>(action.item)];
  std::array<int, N_CROPS> repair_seed_budget{};
  for (int crop = 0; crop < N_CROPS; ++crop)
    repair_seed_budget[static_cast<std::size_t>(crop)] = std::max(
        0, seed_inventory[static_cast<std::size_t>(crop)] -
               baseline_plant_demand[static_cast<std::size_t>(crop)]);
  std::vector<const event_local::TransactionView*> actor_transactions(
      out.units.size(), nullptr);
  std::vector<bool> local_owns_slot(out.units.size(), false);
  std::vector<int> desired_items(out.units.size(), -1);
  std::vector<bool> repair_seed_reserved(out.units.size(), false);
  for (std::size_t actor = 0;
       actor < out.units.size() && actor < current_positions.size(); ++actor) {
    const Action source = actor < source_units.size()
        ? source_units[actor] : Action{};
    local_owns_slot[actor] = same_action(out.units[actor], source);
    const auto local_position = to_event_local(current_positions[actor]);
    const auto transaction = std::find_if(
        open_transactions.begin(), open_transactions.end(),
        [&](const auto& value) {
          return value.key.tile == local_position;
        });
    if (transaction != open_transactions.end())
      actor_transactions[actor] = &*transaction;
    const Tile* tile = tile_at(env, player, current_positions[actor]);
    int desired = tile && tile->kind == TileKind::PLANT
        ? static_cast<int>(tile->crop) : -1;
    if (transaction != open_transactions.end() &&
        transaction->desired_item >= 0)
      desired = transaction->desired_item;
    if (desired < 0) {
      const int tile_index = current_positions[actor].y * env.config().board_size +
          current_positions[actor].x;
      if (tile_index >= 0 && tile_index < static_cast<int>(
              state.experimental_event_local_last_crop.size()))
        desired = static_cast<int>(state.experimental_event_local_last_crop[
            static_cast<std::size_t>(tile_index)]);
    }
    if (desired < 0 && crop_item(out.units[actor].item))
      desired = static_cast<int>(out.units[actor].item);
    desired_items[actor] = desired;
    const bool repair_plant_candidate = transaction != open_transactions.end() &&
        transaction->goal == event_local::GoalKind::Crop &&
        local_owns_slot[actor] && tile && tile->kind == TileKind::EMPTY &&
        out.units[actor].op != Op::PLANT && desired >= 0 &&
        desired < N_CROPS;
    if (repair_plant_candidate &&
        repair_seed_budget[static_cast<std::size_t>(desired)] > 0) {
      repair_seed_reserved[actor] = true;
      --repair_seed_budget[static_cast<std::size_t>(desired)];
    }
  }
  std::map<std::pair<int, int>, std::pair<int, int>> effect_tile_owner;
  const auto is_tile_effect = [](Op operation) {
    return operation == Op::DIG || operation == Op::PLANT ||
           operation == Op::WATER || operation == Op::HARVEST ||
           operation == Op::BUILD_COOP || operation == Op::BUILD_PASTURE;
  };
  for (std::size_t actor = 0;
       actor < out.units.size() && actor < current_positions.size(); ++actor) {
    const bool has_transaction = actor_transactions[actor] != nullptr;
    if (!local_owns_slot[actor]) continue;
    const int score = has_transaction ? 2 :
        (is_tile_effect(out.units[actor].op) ? 1 : 0);
    if (score == 0) continue;
    const auto key = std::pair{static_cast<int>(current_positions[actor].x),
                               static_cast<int>(current_positions[actor].y)};
    const auto found = effect_tile_owner.find(key);
    if (found == effect_tile_owner.end() || score > found->second.first)
      effect_tile_owner[key] = {score, static_cast<int>(actor)};
  }
  for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
    input.composed_baseline.push_back(to_event_local(out.units[actor]));
    event_local::ActorObservation observation;
    observation.actor = static_cast<int>(actor);
    if (actor >= current_positions.size()) continue;
    observation.position = to_event_local(current_positions[actor]);
    observation.actor_generation =
        state.experimental_event_local_actor_generations[actor];
    const Tile* tile = tile_at(env, player, current_positions[actor]);
    const auto owner = effect_tile_owner.find(
        {static_cast<int>(current_positions[actor].x),
         static_cast<int>(current_positions[actor].y)});
    const bool owns_effect_tile = owner == effect_tile_owner.end() ||
        owner->second.second == static_cast<int>(actor);
    observation.tile = to_event_local(env, tile);
    observation.tile_effect_eligible = owns_effect_tile;
    const int desired = desired_items[actor];
    observation.desired_item_hint = desired;
    if (desired >= 0 && desired < N_CROPS && owns_effect_tile) {
      const bool baseline_plant = out.units[actor].op == Op::PLANT &&
          static_cast<int>(out.units[actor].item) == desired;
      observation.desired_item_inventory = baseline_plant
          ? (baseline_plant_demand[static_cast<std::size_t>(desired)] <=
                     seed_inventory[static_cast<std::size_t>(desired)]
                 ? 1 : 0)
          : (actor_transactions[actor] && tile &&
                     tile->kind == TileKind::EMPTY
                 ? (repair_seed_reserved[actor] ? 1 : 0)
                 : seed_inventory[static_cast<std::size_t>(desired)]);
    }
    input.actors.push_back(observation);
  }

  auto decision = state.experimental_event_local_repair->decide(input);
  if (audit) {
    ++audit->local_repair_decisions;
    audit->local_repair_fail_closed +=
        static_cast<int>(decision.fail_closed_actors.size());
  }
  if (!state.experimental_event_local_repair->commit(decision,
                                                      decision.actions))
    throw std::runtime_error("event-local final arbitration rejected proposal");
  for (std::size_t actor = 0; actor < out.units.size(); ++actor)
    out.units[actor] = from_event_local(decision.actions[actor]);
  for (const auto& binding : decision.bindings) {
    NativeAgentState::PendingEventLocalReceipt pending;
    pending.decision_id = decision.id;
    pending.transaction_id = binding.transaction_id;
    pending.actor = binding.actor;
    pending.actor_generation = binding.actor_generation;
    pending.tile = {static_cast<int16_t>(binding.tile.column),
                    static_cast<int16_t>(binding.tile.row)};
    pending.desired_item = static_cast<Item>(binding.action.item);
    pending.desired_inventory_before =
        crop_item(pending.desired_item)
        ? actor_item_inventory(env, player,
                               static_cast<std::size_t>(binding.actor),
                               pending.desired_item)
        : 0;
    const Tile* before = tile_at(env, player, pending.tile);
    pending.before = before ? *before : Tile{};
    pending.emitted = out.units[static_cast<std::size_t>(binding.actor)].op;
    pending.submitted_step = env.step_count();
    state.experimental_event_local_pending.push_back(pending);
  }

  // Open at most one unfinished acquisition debt per compiler transaction.
  // A completed debt may be followed by a later replant-generation debt, but
  // an empty/no-seed observation can never create one debt per tick.
  const auto transactions =
      state.experimental_event_local_repair->open_transactions();
  static constexpr std::array<int, N_CROPS> seed_cost{10, 20, 50, 100, 80};
  for (const auto& transaction : transactions) {
    if (transaction.goal != event_local::GoalKind::Crop ||
        transaction.desired_item < 0 ||
        transaction.desired_item >= N_CROPS)
      continue;
    const Position tile{
        static_cast<int16_t>(transaction.key.tile.column),
        static_cast<int16_t>(transaction.key.tile.row)};
    const Tile* observed = tile_at(env, player, tile);
    const auto crop = static_cast<std::size_t>(transaction.desired_item);
    if (!observed || observed->kind != TileKind::EMPTY ||
        env.privates()[player].seeds[crop] > 0)
      continue;
    const bool already_open = std::any_of(
        state.experimental_event_local_purchase_links.begin(),
        state.experimental_event_local_purchase_links.end(),
        [&](const NativeAgentState::EventLocalPurchaseLink& link) {
          if (link.transaction_id != transaction.id) return false;
          const auto debt =
              state.experimental_event_local_purchase_ledger.debt(link.debt_id);
          return debt && !debt->acquisition_complete;
        });
    if (already_open) continue;
    const auto item = static_cast<Item>(transaction.desired_item);
    const auto debt_id = state.experimental_event_local_purchase_ledger.open(
        {Op::BUY_SEED, item, 1, seed_cost[crop],
         "event-local crop transaction " + std::to_string(transaction.id)});
    state.experimental_event_local_purchase_links.push_back(
        {transaction.id, debt_id, item});
  }

  // Existing final/hard orders own their bytes and slots.  The ledger merely
  // describes a matching order and binds the exact slot after every upstream
  // market writer has finished.
  std::set<std::uint64_t> staged_debts;
  for (std::size_t slot = 0; slot < out.market.size(); ++slot) {
    const auto& order = out.market[slot];
    if (order.op != Op::BUY_SEED || quantity(order) <= 0) continue;
    for (const auto& link : state.experimental_event_local_purchase_links) {
      if (link.item != order.item || staged_debts.contains(link.debt_id))
        continue;
      const auto debt =
          state.experimental_event_local_purchase_ledger.debt(link.debt_id);
      if (!debt || debt->awaiting_receipt || debt->acquisition_complete ||
          debt->remaining < quantity(order))
        continue;
      const auto proposal =
          state.experimental_event_local_purchase_ledger.describe_hard_order(
              link.debt_id, quantity(order));
      if (!proposal) continue;
      const auto status =
          state.experimental_event_local_purchase_ledger.stage_final(
              *proposal,
              {env.step_count(), static_cast<int>(slot), out.market,
               env.privates()[player].seeds[static_cast<std::size_t>(
                   order.item)]});
      if (status != g001::purchase_recovery::StageStatus::Selected) continue;
      staged_debts.insert(link.debt_id);
      if (audit) {
        ++audit->local_repair_seed_orders;
        audit->local_repair_seed_units += quantity(order);
      }
      break;
    }
  }

  // Recovery has no market ownership under a general/phased takeover.  In
  // LegacyDefault it may use only the tail after all hard orders and only
  // cash left after conservatively protecting those orders.
  if (owns_recovery_market_tail) {
    int protected_cash = planned_market_cash(env, player, out);
    for (const auto& link : state.experimental_event_local_purchase_links) {
      if (out.market.size() >= 10) break;
      if (staged_debts.contains(link.debt_id)) continue;
      const auto proposal =
          state.experimental_event_local_purchase_ledger.propose_recovery(
              link.debt_id,
              {env.step_count(), static_cast<int>(env.farms()[player].money),
               protected_cash});
      if (!proposal) continue;
      const int purchase = quantity(proposal->order);
      if (purchase <= 0) continue;
      const int slot = static_cast<int>(out.market.size());
      out.market.push_back(proposal->order);
      const auto status =
          state.experimental_event_local_purchase_ledger.stage_final(
              *proposal,
              {env.step_count(), slot, out.market,
               env.privates()[player].seeds[static_cast<std::size_t>(
                   proposal->order.item)],
               static_cast<int>(env.farms()[player].money), protected_cash});
      if (status != g001::purchase_recovery::StageStatus::Selected) {
        out.market.pop_back();
        continue;
      }
      staged_debts.insert(link.debt_id);
      protected_cash += purchase * proposal->unit_cost;
      if (audit) {
        ++audit->local_repair_seed_orders;
        audit->local_repair_seed_units += purchase;
      }
    }
  }
  if (audit) ++audit->local_repair_commits;
}

namespace day_horizon = g001::day_horizon_repair;

event_local::Action to_day_horizon(Action action) {
  auto result = to_event_local(action);
  if (result.op != event_local::Op::Move) return result;
  if (action.op == Op::NORTH) result.arg0 = 0;
  if (action.op == Op::SOUTH) result.arg0 = 1;
  if (action.op == Op::WEST) result.arg0 = 2;
  if (action.op == Op::EAST) result.arg0 = 3;
  return result;
}

Action from_day_horizon(const event_local::Action& action) {
  if (action.op != event_local::Op::Move) return from_event_local(action);
  Action result;
  if (action.arg0 == 0) result.op = Op::NORTH;
  else if (action.arg0 == 1) result.op = Op::SOUTH;
  else if (action.arg0 == 2) result.op = Op::WEST;
  else if (action.arg0 == 3) result.op = Op::EAST;
  else throw std::runtime_error("day planner lost MOVE direction");
  result.item = static_cast<Item>(action.item);
  result.quantity = action.quantity;
  return result;
}

void compose_day_horizon_repair(const std::vector<PlayerAction>& tape,
                                const Simulator& env, int player,
                                NativeAgentState& state, PlayerAction& out,
                                NativeRepairAudit* audit) {
  auto& runtime = state.experimental_day_horizon;
  // Settle only the immediately preceding exact slot. A surprising receipt
  // disables every still-future experimental slot; past bytes remain an audit
  // fact and are never rewritten.
  for (const auto& pending : runtime.pending) {
    bool confirmed = env.step_count() == pending.submitted_step + 1;
    const Tile* after = tile_at(env, player, pending.tile);
    if (!after) confirmed = false;
    if (confirmed) {
      switch (pending.emitted) {
        case Op::DIG:
          confirmed = pending.before.kind == TileKind::WEED &&
              after->kind != TileKind::WEED;
          break;
        case Op::PLANT:
          confirmed = after->kind == TileKind::PLANT;
          break;
        case Op::WATER:
          confirmed = after->kind == TileKind::PLANT && after->watered_today;
          break;
        case Op::HARVEST:
          confirmed = pending.before.kind == TileKind::PLANT &&
              (after->kind == TileKind::EMPTY ||
               after->yield_units < pending.before.yield_units);
          break;
        default: confirmed = true; break;
      }
    }
    if (!confirmed) {
      runtime.active = false;
      runtime.failed_closed = true;
      runtime.failure_code = 4;
      if (audit) ++audit->day_horizon_fail_closed;
    }
  }
  runtime.pending.clear();

  const int step = env.step_count();
  const int day = env.day();
  const int actors_now = static_cast<int>(out.units.size());
  if (runtime.day != day) runtime = NativeAgentState::DayHorizonRuntime{};
  if (runtime.day < 0) runtime.day = day;
  if (runtime.failed_closed) return;

  if (!runtime.active) {
    const int day_end = std::min(
        (day + 1) * env.config().turns_per_day,
        static_cast<int>(tape.size()));
    const int turns = day_end - step;
    if (turns <= 0 || actors_now == 0) return;
    const auto current_positions = positions(env, player);
    if (current_positions.size() != out.units.size()) return;
    const auto& source_now = tape[static_cast<std::size_t>(
        std::min(step, static_cast<int>(tape.size()) - 1))].units;
    for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
      const Action source = actor < source_now.size()
          ? source_now[actor] : Action{};
      if (!same_action(out.units[actor], source)) {
        runtime.failed_closed = true;
        runtime.failure_code = 1;
        if (audit) ++audit->day_horizon_fail_closed;
        return;
      }
    }

    std::vector<day_horizon::ActorPlan> actor_plans;
    actor_plans.reserve(out.units.size());
    std::map<std::pair<int, int>, std::size_t> objective_by_tile;
    std::vector<day_horizon::Objective> objectives;
    for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
      day_horizon::ActorPlan plan;
      plan.actor = static_cast<int>(actor);
      plan.start = to_event_local(current_positions[actor]);
      Position simulated = current_positions[actor];
      for (int offset = 0; offset < turns; ++offset) {
        const auto source_index = static_cast<std::size_t>(step + offset);
        const Action source = actor < tape[source_index].units.size()
            ? tape[source_index].units[actor] : Action{};
        plan.raw.push_back(to_day_horizon(source));
        plan.blocked_equivalent_capacity.push_back(false);
        const Tile* observed = tile_at(env, player, simulated);
        const bool crop_source =
            (source.op == Op::PLANT || source.op == Op::WATER ||
             source.op == Op::HARVEST) && crop_item(source.item);
        const bool trigger = crop_source && observed &&
            (observed->kind == TileKind::WEED ||
             (observed->kind == TileKind::EMPTY && source.op != Op::PLANT));
        if (trigger) {
          // This source production is observation-proven invalid and is
          // represented by the lifecycle objective below. Keeping it in the
          // raw queue would replay stale HARVEST/WATER after DIG/PLANT.
          plan.raw.back() = to_day_horizon(Action{});
          const auto key = std::pair{static_cast<int>(simulated.x),
                                     static_cast<int>(simulated.y)};
          if (!objective_by_tile.contains(key)) {
            day_horizon::Objective objective;
            objective.id = runtime.next_objective_id++;
            objective.tile = to_event_local(simulated);
            objective.deadline_turn = turns - 1;
            // A source production proven invalid is a hard lifecycle repair,
            // not an optional one-point capacity fill. This dominates bounded
            // MOVE timing cost while MOVE order/terminal completion stay hard.
            objective.value = source.op == Op::HARVEST ? 1000 : 500;
            objective.critical = true;
            if (observed->kind == TileKind::WEED)
              objective.remaining_transitions.push_back(
                  {event_local::Op::Dig, -1, 1, 0, 0});
            objective.remaining_transitions.push_back(
                {event_local::Op::Plant, static_cast<int>(source.item), 1,
                 0, 0});
            objective.remaining_transitions.push_back(
                {event_local::Op::Water, static_cast<int>(source.item), 1,
                 0, 0});
            objective.deadline_turn = std::min(
                turns - 1, offset + static_cast<int>(
                    objective.remaining_transitions.size()) - 1);
            objective_by_tile[key] = objectives.size();
            objectives.push_back(std::move(objective));
          }
        }
        if (movement(source.op)) {
          if (source.op == Op::NORTH) --simulated.y;
          if (source.op == Op::SOUTH) ++simulated.y;
          if (source.op == Op::WEST) --simulated.x;
          if (source.op == Op::EAST) ++simulated.x;
        }
      }
      actor_plans.push_back(std::move(plan));
    }
    // No trigger means literal baseline ownership: do not even stage a day
    // manifest. This is the enabled/no-trigger bit-exact contract.
    if (objectives.empty()) return;
    day_horizon::ResourceSnapshot resources;
    for (int crop = 0; crop < N_CROPS; ++crop)
      resources.seeds[crop] =
          env.privates()[player].seeds[static_cast<std::size_t>(crop)];
    day_horizon::TimeExpandedResult planned;
    try {
      planned = day_horizon::compile_time_expanded(
          actor_plans, objectives, resources);
    } catch (const std::exception&) {
      runtime.failed_closed = true;
      runtime.failure_code = 2;
      if (audit) ++audit->day_horizon_fail_closed;
      return;
    }
    if (!planned.ordered_move_sequences_exact ||
        planned.terminal_unexecuted_raw_actions != 0 ||
        planned.manifest.size() != out.units.size()) {
      runtime.failed_closed = true;
      runtime.failure_code = 3;
      if (audit) {
        ++audit->day_horizon_fail_closed;
        audit->day_horizon_terminal_raw +=
            planned.terminal_unexecuted_raw_actions;
      }
      return;
    }
    runtime.manifest.assign(out.units.size(), {});
    runtime.objective_bindings.assign(out.units.size(),
                                      std::vector<std::uint64_t>(turns));
    for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
      runtime.manifest[actor].reserve(static_cast<std::size_t>(turns));
      for (const auto& action : planned.manifest[actor])
        runtime.manifest[actor].push_back(from_day_horizon(action));
    }
    for (const auto& assignment : planned.assignments) {
      if (assignment.actor >= 0 &&
          assignment.actor < static_cast<int>(runtime.objective_bindings.size()) &&
          assignment.turn >= 0 && assignment.turn < turns)
        runtime.objective_bindings[static_cast<std::size_t>(assignment.actor)]
                                  [static_cast<std::size_t>(assignment.turn)] =
            assignment.objective_id;
    }
    runtime.source_step = step;
    runtime.actor_count = actors_now;
    runtime.active = true;
    if (audit) {
      ++audit->day_horizon_plans;
      audit->day_horizon_assignments +=
          static_cast<int>(planned.assignments.size());
    }
  }

  const int turn = step - runtime.source_step;
  if (!runtime.active || runtime.actor_count != actors_now || turn < 0) {
    runtime.active = false;
    runtime.failed_closed = true;
    runtime.failure_code = 5;
    if (audit) ++audit->day_horizon_fail_closed;
    return;
  }
  for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
    if (turn >= static_cast<int>(runtime.manifest[actor].size())) continue;
    out.units[actor] = runtime.manifest[actor][static_cast<std::size_t>(turn)];
    const auto objective = runtime.objective_bindings[actor]
        [static_cast<std::size_t>(turn)];
    if (objective == 0) continue;
    NativeAgentState::DayHorizonPending pending;
    pending.objective_id = objective;
    pending.actor = static_cast<int>(actor);
    const auto now = positions(env, player);
    if (actor >= now.size()) continue;
    pending.tile = now[actor];
    pending.emitted = out.units[actor].op;
    const Tile* before = tile_at(env, player, pending.tile);
    pending.before = before ? *before : Tile{};
    pending.submitted_step = step;
    runtime.pending.push_back(pending);
  }
  if (audit) ++audit->day_horizon_commits;
}

namespace {

using DayV2Runtime = NativeAgentState::DayHorizonV2Runtime;
using DayV2Objective = NativeAgentState::DayHorizonV2Objective;

bool day_v2_tile_receipt(const Tile& before, const Tile* after, Op emitted) {
  if (!after) return false;
  switch (emitted) {
    case Op::DIG:
      return before.kind == TileKind::WEED && after->kind != TileKind::WEED;
    case Op::PLANT: return after->kind == TileKind::PLANT;
    case Op::WATER:
      return after->kind == TileKind::PLANT && after->watered_today;
    case Op::HARVEST:
      return before.kind == TileKind::PLANT &&
          (after->kind == TileKind::EMPTY ||
           after->yield_units < before.yield_units);
    default: return true;
  }
}

std::vector<day_horizon::ActorPlan> day_v2_tape_actors(
    const std::vector<PlayerAction>& tape, const Simulator& env, int player,
    std::size_t actors) {
  const int step = env.step_count();
  const int end = std::min((env.day() + 1) * env.config().turns_per_day,
                           static_cast<int>(tape.size()));
  const auto starts = positions(env, player);
  std::vector<day_horizon::ActorPlan> result;
  result.reserve(actors);
  for (std::size_t actor = 0; actor < actors; ++actor) {
    day_horizon::ActorPlan plan;
    plan.actor = static_cast<int>(actor);
    plan.start = to_event_local(starts[actor]);
    for (int source = step; source < end; ++source) {
      const auto& units = tape[static_cast<std::size_t>(source)].units;
      plan.raw.push_back(to_day_horizon(
          actor < units.size() ? units[actor] : Action{}));
      plan.blocked_equivalent_capacity.push_back(false);
    }
    result.push_back(std::move(plan));
  }
  return result;
}

std::vector<day_horizon::ActorPlan> day_v2_remaining_actors(
    const DayV2Runtime& runtime, const Simulator& env, int player) {
  const int frozen = env.step_count() - runtime.source_step;
  const auto starts = positions(env, player);
  if (frozen < 0 || starts.size() != runtime.actors.size() ||
      runtime.plan.manifest.size() != runtime.actors.size())
    throw std::runtime_error("day-v2 actor horizon changed");
  std::set<std::pair<int, int>> repair_slots;
  for (const auto& assignment : runtime.plan.assignments)
    repair_slots.insert({assignment.actor, assignment.turn});
  std::vector<day_horizon::ActorPlan> result;
  result.reserve(runtime.actors.size());
  for (std::size_t actor = 0; actor < runtime.actors.size(); ++actor) {
    const auto& prior_actor = runtime.actors[actor];
    std::vector<std::pair<int, event_local::Action>> raw_queue;
    for (std::size_t turn = 0; turn < prior_actor.raw.size(); ++turn)
      if (prior_actor.raw[turn].op != event_local::Op::Pass)
        raw_queue.push_back({static_cast<int>(turn), prior_actor.raw[turn]});
    std::size_t cursor = 0;
    const int prefix = std::min(
        frozen, static_cast<int>(runtime.plan.manifest[actor].size()));
    for (int turn = 0; turn < prefix; ++turn) {
      if (repair_slots.contains({static_cast<int>(actor), turn})) continue;
      const auto& emitted = runtime.plan.manifest[actor][
          static_cast<std::size_t>(turn)];
      if (emitted.op == event_local::Op::Pass) continue;
      if (cursor >= raw_queue.size() || !(raw_queue[cursor].second == emitted))
        throw std::runtime_error("day-v2 raw queue receipt diverged");
      ++cursor;
    }
    const int remaining_turns = static_cast<int>(prior_actor.raw.size()) - frozen;
    if (remaining_turns <= 0)
      throw std::runtime_error("day-v2 exhausted suffix");
    day_horizon::ActorPlan suffix;
    suffix.actor = static_cast<int>(actor);
    suffix.start = to_event_local(starts[actor]);
    suffix.raw.assign(static_cast<std::size_t>(remaining_turns), {});
    suffix.blocked_equivalent_capacity.assign(
        static_cast<std::size_t>(remaining_turns), false);
    int next_slot = 0;
    for (; cursor < raw_queue.size(); ++cursor) {
      next_slot = std::max(next_slot, raw_queue[cursor].first - frozen);
      if (next_slot >= remaining_turns)
        throw std::runtime_error("day-v2 terminal raw capacity exhausted");
      suffix.raw[static_cast<std::size_t>(next_slot)] =
          raw_queue[cursor].second;
      ++next_slot;
    }
    result.push_back(std::move(suffix));
  }
  return result;
}

DayV2Objective* day_v2_objective(DayV2Runtime& runtime,
                                 std::uint64_t id) {
  const auto found = std::find_if(
      runtime.objectives.begin(), runtime.objectives.end(),
      [&](const DayV2Objective& objective) { return objective.id == id; });
  return found == runtime.objectives.end() ? nullptr : &*found;
}

bool day_v2_open_objective(const DayV2Objective& objective) {
  return objective.waiting_maturity || !objective.remaining.empty();
}

void settle_day_horizon_v2_purchases(const Simulator& env, int player,
                                     NativeAgentState& state,
                                     NativeRepairAudit* audit) {
  auto& runtime = state.experimental_day_horizon_v2;
  g001::purchase_recovery::ReceiptObservation observation;
  observation.step = env.step_count();
  observation.slot_fills = env.last_market_fills()[player];
  observation.seeds_after = env.privates()[player].seeds;
  for (int animal = 0; animal < N_ANIMALS; ++animal)
    observation.animals_after[static_cast<std::size_t>(animal)] =
        env.privates()[player].shed[static_cast<std::size_t>(
            static_cast<int>(Item::GOOSE) + animal)];
  const auto settlements = runtime.purchase_ledger.observe(observation);
  for (const auto& settlement : settlements) {
    if (settlement.filled > 0) {
      runtime.resource_changed = true;
      if (audit) audit->day_horizon_v2_seed_fills += settlement.filled;
    }
    if (settlement.status == g001::purchase_recovery::FillStatus::Zero && audit)
      ++audit->day_horizon_v2_seed_zero_fills;
  }
  (void)runtime.purchase_ledger.drain_handoffs();
}

bool day_v2_confirm_and_retire(const Simulator& env, int player,
                               DayV2Runtime& runtime,
                               NativeRepairAudit* audit) {
  if (runtime.pending.empty()) return false;
  for (const auto& pending : runtime.pending) {
    const Tile* after = tile_at(env, player, pending.tile);
    const bool confirmed =
        env.step_count() == pending.submitted_step + 1 &&
        day_v2_tile_receipt(pending.before, after, pending.emitted);
    if (audit) {
      if (confirmed) ++audit->day_horizon_v2_receipts_confirmed;
      else ++audit->day_horizon_v2_receipts_failed;
    }
    auto* objective = day_v2_objective(runtime, pending.objective_id);
    if (!objective) continue;
    if (!confirmed) {
      if (pending.emitted == Op::HARVEST) {
        objective->remaining.clear();
        objective->waiting_maturity = true;
      }
      continue;
    }
    if (objective->remaining.empty() || pending.transition_index != 0 ||
        from_day_horizon(objective->remaining.front()).op != pending.emitted)
      throw std::runtime_error("day-v2 receipt/objective mismatch");
    objective->remaining.erase(objective->remaining.begin());
    if (!objective->remaining.empty()) continue;
    if (pending.emitted == Op::WATER && objective->requires_harvest) {
      objective->waiting_maturity = true;
      if (audit) ++audit->day_horizon_v2_maturity_waits;
    } else {
      if (audit) ++audit->day_horizon_v2_objectives_completed;
    }
  }
  runtime.pending.clear();
  runtime.objectives.erase(
      std::remove_if(runtime.objectives.begin(), runtime.objectives.end(),
                     [](const DayV2Objective& objective) {
                       return !day_v2_open_objective(objective);
                     }),
      runtime.objectives.end());
  return true;
}

void day_v2_open_trigger(DayV2Runtime& runtime, const Simulator& env,
                         int player, std::size_t actor, Action source,
                         std::vector<day_horizon::ActorPlan>& actors) {
  if (actor >= actors.size() || actors[actor].raw.empty()) return;
  const auto starts = positions(env, player);
  const Position tile_position = starts[actor];
  const Tile* tile = tile_at(env, player, tile_position);
  if (!tile || !crop_item(source.item)) return;
  const bool invalid = tile->kind == TileKind::WEED ||
      (tile->kind == TileKind::EMPTY && source.op != Op::PLANT);
  if (!invalid) return;
  const auto duplicate = std::find_if(
      runtime.objectives.begin(), runtime.objectives.end(),
      [&](const DayV2Objective& objective) {
        return objective.tile.x == tile_position.x &&
            objective.tile.y == tile_position.y &&
            objective.desired == source.item && day_v2_open_objective(objective);
      });
  if (duplicate == runtime.objectives.end()) {
    DayV2Objective objective;
    objective.id = runtime.next_objective_id++;
    objective.tile = tile_position;
    objective.desired = source.item;
    objective.source = source.op;
    objective.source_epoch = static_cast<std::uint64_t>(env.step_count() + 1);
    objective.requires_harvest = source.op == Op::HARVEST;
    if (tile->kind == TileKind::WEED)
      objective.remaining.push_back(to_day_horizon({Op::DIG}));
    objective.remaining.push_back(to_day_horizon(
        {Op::PLANT, source.item, 1}));
    objective.remaining.push_back(to_day_horizon(
        {Op::WATER, source.item, 1}));
    runtime.objectives.push_back(std::move(objective));
  }
  // Absorb exactly the observation-proven invalid source at the head of this
  // suffix. Every later raw action stays in the ordered queue.
  if (actors[actor].raw.front() == to_day_horizon(source))
    actors[actor].raw.front() = {};
}

bool day_v2_compile(const Simulator& env, int player,
                    DayV2Runtime& runtime,
                    std::vector<day_horizon::ActorPlan> actors,
                    NativeRepairAudit* audit, bool replan) {
  std::vector<day_horizon::Objective> objectives;
  day_horizon::ObservationReadiness readiness;
  readiness.observation_epoch = static_cast<std::uint64_t>(env.step_count() + 1);
  readiness.consumed_harvest_legal = runtime.consumed_maturity_proofs;
  for (auto& persistent : runtime.objectives) {
    if (persistent.waiting_maturity) {
      const Tile* tile = tile_at(env, player, persistent.tile);
      const auto observed = to_event_local(env, tile);
      if (tile && tile->kind == TileKind::PLANT &&
          tile->crop == persistent.desired && observed.harvest_legal &&
          tile->yield_units > 0) {
        persistent.waiting_maturity = false;
        persistent.remaining = {to_day_horizon(
            {Op::HARVEST, persistent.desired, 1})};
      } else if (tile && tile->kind == TileKind::PLANT &&
                 tile->crop == persistent.desired &&
                 !tile->watered_today) {
        // WATERED_IMMATURE is a wait node with a daily maintenance edge. It
        // cannot unlock HARVEST, but it prevents a cross-day carry from
        // silently dying while maturity is pending.
        persistent.waiting_maturity = false;
        persistent.remaining = {to_day_horizon(
            {Op::WATER, persistent.desired, 1})};
      } else if (tile && tile->kind == TileKind::WEED) {
        persistent.waiting_maturity = false;
        persistent.remaining = {
            to_day_horizon({Op::DIG}),
            to_day_horizon({Op::PLANT, persistent.desired, 1}),
            to_day_horizon({Op::WATER, persistent.desired, 1})};
      } else if (tile && tile->kind == TileKind::EMPTY) {
        persistent.waiting_maturity = false;
        persistent.remaining = {
            to_day_horizon({Op::PLANT, persistent.desired, 1}),
            to_day_horizon({Op::WATER, persistent.desired, 1})};
      } else {
        continue;
      }
    }
    if (persistent.remaining.empty()) continue;
    day_horizon::Objective objective;
    objective.id = persistent.id;
    objective.tile = to_event_local(persistent.tile);
    objective.remaining_transitions = persistent.remaining;
    // The invalid source was due at this observation. Its shortest legal
    // state suffix must complete before the route leaves the plot; otherwise
    // a cost-only optimizer would walk away first and defeat the promised
    // DIG->PLANT->WATER ordering.
    objective.deadline_turn = std::max(
        0, static_cast<int>(persistent.remaining.size()) - 1);
    objective.value = persistent.requires_harvest ? 1000 : 500;
    objective.critical = true;
    if (persistent.remaining.front().op == event_local::Op::Harvest) {
      const day_horizon::HarvestProofToken proof{
          readiness.observation_epoch, persistent.id, 0};
      readiness.harvest_legal_required.insert({persistent.id, 0});
      readiness.harvest_legal.insert(proof);
    }
    objectives.push_back(std::move(objective));
  }
  day_horizon::ResourceSnapshot resources;
  for (int crop = 0; crop < N_CROPS; ++crop)
    resources.seeds[crop] =
        env.privates()[player].seeds[static_cast<std::size_t>(crop)];
  try {
    auto planned = day_horizon::compile_time_expanded(
        actors, objectives, resources, readiness);
    if (!planned.ordered_move_sequences_exact ||
        planned.terminal_unexecuted_raw_actions != 0 ||
        planned.manifest.size() != actors.size())
      throw std::runtime_error("day-v2 route hard gate");
    runtime.actors = std::move(actors);
    runtime.plan = std::move(planned);
    runtime.source_step = env.step_count();
    runtime.actor_count = static_cast<int>(runtime.actors.size());
    runtime.active = true;
    runtime.resource_changed = false;
    const auto consumed_before = runtime.consumed_maturity_proofs.size();
    runtime.consumed_maturity_proofs.insert(
        runtime.plan.consumed_harvest_legal.begin(),
        runtime.plan.consumed_harvest_legal.end());
    if (audit) {
      if (replan) ++audit->day_horizon_v2_replans;
      else ++audit->day_horizon_v2_plans;
      audit->day_horizon_v2_assignments +=
          static_cast<int>(runtime.plan.assignments.size());
      audit->day_horizon_v2_exact_plans += runtime.plan.exact;
      audit->day_horizon_v2_fallback_plans +=
          runtime.plan.fell_back_to_greedy;
      audit->day_horizon_v2_budget_exhausted +=
          !runtime.plan.within_time_budget;
      audit->day_horizon_v2_terminal_raw +=
          runtime.plan.terminal_unexecuted_raw_actions;
      audit->day_horizon_v2_maturity_tokens_consumed += static_cast<int>(
          runtime.consumed_maturity_proofs.size() - consumed_before);
    }
    return true;
  } catch (const std::exception&) {
    runtime.active = false;
    if (audit) {
      ++audit->day_horizon_v2_fail_closed;
      ++audit->day_horizon_v2_fail_reasons[2];
    }
    return false;
  }
}

void day_v2_open_seed_debts(const Simulator& env, int player,
                            DayV2Runtime& runtime,
                            NativeRepairAudit* audit) {
  static constexpr std::array<int, N_CROPS> seed_cost{10, 20, 50, 100, 80};
  const std::set<std::uint64_t> unscheduled(
      runtime.plan.unscheduled_objectives.begin(),
      runtime.plan.unscheduled_objectives.end());
  for (const auto& objective : runtime.objectives) {
    if (!unscheduled.contains(objective.id) || objective.remaining.empty() ||
        objective.remaining.front().op != event_local::Op::Plant ||
        !crop_item(objective.desired))
      continue;
    const auto crop = static_cast<std::size_t>(objective.desired);
    if (env.privates()[player].seeds[crop] > 0) continue;
    if (audit) ++audit->day_horizon_v2_seed_unscheduled;
    const bool linked = std::any_of(
        runtime.purchase_links.begin(), runtime.purchase_links.end(),
        [&](const auto& link) {
          if (link.objective_id != objective.id) return false;
          const auto debt = runtime.purchase_ledger.debt(link.debt_id);
          return debt && !debt->acquisition_complete;
        });
    if (linked) continue;
    const auto debt = runtime.purchase_ledger.open(
        {Op::BUY_SEED, objective.desired, 1, seed_cost[crop],
         "day-horizon-v2 objective " + std::to_string(objective.id)});
    runtime.purchase_links.push_back(
        {objective.id, debt, objective.desired});
  }
}

void day_v2_compose_market(const Simulator& env, int player,
                           DayV2Runtime& runtime, PlayerAction& out,
                           NativeRepairAudit* audit,
                           bool owns_recovery_market_tail) {
  std::set<std::uint64_t> staged;
  for (std::size_t slot = 0; slot < out.market.size(); ++slot) {
    const auto& order = out.market[slot];
    if (order.op != Op::BUY_SEED || quantity(order) <= 0) continue;
    for (const auto& link : runtime.purchase_links) {
      if (link.item != order.item || staged.contains(link.debt_id)) continue;
      const auto debt = runtime.purchase_ledger.debt(link.debt_id);
      if (!debt || debt->awaiting_receipt || debt->acquisition_complete ||
          debt->remaining < quantity(order))
        continue;
      const auto proposal = runtime.purchase_ledger.describe_hard_order(
          link.debt_id, quantity(order));
      if (!proposal) continue;
      const auto status = runtime.purchase_ledger.stage_final(
          *proposal,
          {env.step_count(), static_cast<int>(slot), out.market,
           env.privates()[player].seeds[static_cast<std::size_t>(order.item)]});
      if (status != g001::purchase_recovery::StageStatus::Selected) continue;
      staged.insert(link.debt_id);
      if (audit) ++audit->day_horizon_v2_seed_orders;
      break;
    }
  }
  if (!owns_recovery_market_tail) return;
  int protected_cash = planned_market_cash(env, player, out);
  for (const auto& link : runtime.purchase_links) {
    if (out.market.size() >= 10 || staged.contains(link.debt_id)) continue;
    const auto proposal = runtime.purchase_ledger.propose_recovery(
        link.debt_id,
        {env.step_count(), static_cast<int>(env.farms()[player].money),
         protected_cash});
    if (!proposal) continue;
    const int slot = static_cast<int>(out.market.size());
    out.market.push_back(proposal->order);
    const auto status = runtime.purchase_ledger.stage_final(
        *proposal,
        {env.step_count(), slot, out.market,
         env.privates()[player].seeds[static_cast<std::size_t>(link.item)],
         static_cast<int>(env.farms()[player].money), protected_cash});
    if (status != g001::purchase_recovery::StageStatus::Selected) {
      out.market.pop_back();
      continue;
    }
    staged.insert(link.debt_id);
    protected_cash += quantity(proposal->order) * proposal->unit_cost;
    if (audit) ++audit->day_horizon_v2_seed_orders;
  }
}

void compose_day_horizon_repair_v2(
    const std::vector<PlayerAction>& tape, const Simulator& env, int player,
    NativeAgentState& state, PlayerAction& out, NativeRepairAudit* audit,
    bool owns_recovery_market_tail) {
  auto& runtime = state.experimental_day_horizon_v2;
  const int day = env.day();
  const bool had_receipt = day_v2_confirm_and_retire(
      env, player, runtime, audit);
  bool replan = had_receipt;
  std::vector<day_horizon::ActorPlan> actors;
  if (runtime.day != day) {
    if (runtime.day >= 0 && audit)
      audit->day_horizon_v2_objectives_carried +=
          static_cast<int>(runtime.objectives.size());
    runtime.day = day;
    runtime.active = false;
    replan = replan || !runtime.objectives.empty();
  }
  replan = replan || runtime.resource_changed;

  const auto current_positions = positions(env, player);
  if (current_positions.size() != out.units.size()) return;
  if (runtime.active) {
    try {
      actors = day_v2_remaining_actors(runtime, env, player);
    } catch (const std::exception&) {
      runtime.active = false;
      if (audit) {
        ++audit->day_horizon_v2_fail_closed;
        ++audit->day_horizon_v2_fail_reasons[1];
      }
      return;
    }
  } else {
    actors = day_v2_tape_actors(tape, env, player, out.units.size());
  }

  const auto source_index = static_cast<std::size_t>(std::min(
      env.step_count(), static_cast<int>(tape.size()) - 1));
  const auto& source_units = tape[source_index].units;
  for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
    const Action source = actor < source_units.size()
        ? source_units[actor] : Action{};
    if (!same_action(out.units[actor], source)) continue;
    const std::size_t before = runtime.objectives.size();
    if (source.op == Op::PLANT || source.op == Op::WATER ||
        source.op == Op::HARVEST)
      day_v2_open_trigger(runtime, env, player, actor, source, actors);
    if (runtime.objectives.size() != before ||
        (!actors[actor].raw.empty() &&
         actors[actor].raw.front().op == event_local::Op::Pass &&
         source.op != Op::PASS))
      replan = true;
  }
  for (const auto& objective : runtime.objectives)
    if (objective.waiting_maturity) {
      const Tile* tile = tile_at(env, player, objective.tile);
      if (tile && tile->kind == TileKind::PLANT &&
          tile->crop == objective.desired &&
          to_event_local(env, tile).harvest_legal &&
          tile->yield_units > 0)
        replan = true;
    }
  if (!runtime.active && runtime.objectives.empty()) return;
  if (replan || !runtime.active) {
    if (!day_v2_compile(env, player, runtime, std::move(actors), audit,
                        had_receipt || runtime.resource_changed ||
                            runtime.source_step >= 0))
      return;
    day_v2_open_seed_debts(env, player, runtime, audit);
  }
  const int turn = env.step_count() - runtime.source_step;
  if (turn < 0 || turn >= static_cast<int>(runtime.plan.manifest.front().size()))
    return;
  std::map<std::pair<int, int>, day_horizon::Assignment> bindings;
  for (const auto& assignment : runtime.plan.assignments)
    bindings[{assignment.actor, assignment.turn}] = assignment;
  for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
    out.units[actor] = from_day_horizon(
        runtime.plan.manifest[actor][static_cast<std::size_t>(turn)]);
    const auto binding = bindings.find({static_cast<int>(actor), turn});
    if (binding == bindings.end()) continue;
    NativeAgentState::DayHorizonV2Pending pending;
    pending.objective_id = binding->second.objective_id;
    pending.transition_index = binding->second.transition_index;
    pending.actor = static_cast<int>(actor);
    pending.actor_generation = event_local_actor_generation(
        env.day(), actor);
    pending.tile = current_positions[actor];
    pending.emitted = out.units[actor].op;
    const Tile* before = tile_at(env, player, pending.tile);
    pending.before = before ? *before : Tile{};
    pending.submitted_step = env.step_count();
    if (pending.emitted == Op::HARVEST)
      pending.maturity_proof = day_horizon::HarvestProofToken{
          static_cast<std::uint64_t>(env.step_count() + 1),
          pending.objective_id, pending.transition_index};
    runtime.pending.push_back(std::move(pending));
  }
  day_v2_compose_market(env, player, runtime, out, audit,
                        owns_recovery_market_tail);
}

}  // namespace

namespace {

using SkeletonRuntime = NativeAgentState::RouteSkeletonRuntime;
using SkeletonObjective = NativeAgentState::RouteSkeletonObjective;

SkeletonObjective* skeleton_objective(SkeletonRuntime& runtime,
                                      std::uint64_t id) {
  const auto found = std::find_if(
      runtime.objectives.begin(), runtime.objectives.end(),
      [&](const auto& objective) { return objective.id == id; });
  return found == runtime.objectives.end() ? nullptr : &*found;
}

bool skeleton_open(const SkeletonObjective& objective) {
  return objective.waiting_maturity || !objective.remaining.empty();
}

void settle_route_skeleton_receipts(const Simulator& env, int player,
                                    SkeletonRuntime& runtime,
                                    NativeRepairAudit* audit) {
  for (const auto& pending : runtime.pending) {
    const Tile* after = tile_at(env, player, pending.tile);
    const bool confirmed =
        env.step_count() == pending.submitted_step + 1 &&
        day_v2_tile_receipt(pending.before, after, pending.emitted);
    if (audit) {
      if (confirmed) ++audit->route_skeleton_v3_receipts_confirmed;
      else ++audit->route_skeleton_v3_receipts_failed;
    }
    auto* objective = skeleton_objective(runtime, pending.objective_id);
    if (!objective) continue;
    if (!confirmed) {
      if (pending.emitted == Op::HARVEST) {
        objective->remaining.clear();
        objective->waiting_maturity = true;
      }
      continue;
    }
    if (objective->remaining.empty() ||
        from_day_horizon(objective->remaining.front()).op != pending.emitted)
      throw std::runtime_error("route-skeleton receipt/objective mismatch");
    objective->remaining.erase(objective->remaining.begin());
    if (!objective->remaining.empty()) continue;
    if (pending.emitted == Op::WATER && objective->requires_harvest)
      objective->waiting_maturity = true;
    else if (audit)
      ++audit->route_skeleton_v3_objectives_completed;
  }
  runtime.pending.clear();
  runtime.objectives.erase(
      std::remove_if(runtime.objectives.begin(), runtime.objectives.end(),
                     [](const auto& objective) {
                       return !skeleton_open(objective);
                     }),
      runtime.objectives.end());
}

std::vector<NativeAgentState::RouteSkeletonMove> source_move_skeleton(
    const std::vector<PlayerAction>& tape, int actor, int begin, int end) {
  std::vector<NativeAgentState::RouteSkeletonMove> result;
  for (int source = begin; source < end; ++source) {
    const auto& units = tape[static_cast<std::size_t>(source)].units;
    if (actor >= static_cast<int>(units.size()) ||
        !movement(units[static_cast<std::size_t>(actor)].op))
      continue;
    result.push_back({source, units[static_cast<std::size_t>(actor)], false});
  }
  return result;
}

bool reconcile_route_skeleton(const std::vector<PlayerAction>& tape,
                              const Simulator& env, std::size_t actor_count,
                              SkeletonRuntime& runtime,
                              NativeRepairAudit* audit) {
  const int step = env.step_count();
  const int day = env.day();
  const int end = std::min((day + 1) * env.config().turns_per_day,
                           static_cast<int>(tape.size()));
  if (runtime.day != day) {
    if (runtime.day >= 0 && audit)
      audit->route_skeleton_v3_objectives_carried +=
          static_cast<int>(runtime.objectives.size());
    runtime.day = day;
    runtime.actors.clear();
  }
  for (std::size_t actor = 0; actor < actor_count; ++actor) {
    const auto generation = event_local_actor_generation(day, actor);
    const auto candidate = source_move_skeleton(
        tape, static_cast<int>(actor), step, end);
    auto found = std::find_if(
        runtime.actors.begin(), runtime.actors.end(), [&](const auto& state) {
          return state.actor == static_cast<int>(actor) &&
              state.generation == generation;
        });
    if (found == runtime.actors.end()) {
      NativeAgentState::RouteSkeletonActor state;
      state.actor = static_cast<int>(actor);
      state.generation = generation;
      state.moves = candidate;
      runtime.actors.push_back(std::move(state));
      continue;
    }
    std::vector<NativeAgentState::RouteSkeletonMove> overdue;
    std::vector<NativeAgentState::RouteSkeletonMove> future_old;
    for (std::size_t cursor = found->cursor; cursor < found->moves.size();
         ++cursor) {
      if (found->moves[cursor].source_step < step) {
        // A proposal rejection returns the real baseline action and, by
        // contract, cannot mutate the experimental runtime.  Thus an overdue
        // source MOVE which was never explicitly deferred by an accepted
        // manifest has already run as baseline and is acknowledged here.
        // Only accepted planner delays survive as overdue skeleton debt.
        if (found->moves[cursor].deferred_by_planner)
          overdue.push_back(found->moves[cursor]);
      }
      else
        future_old.push_back(found->moves[cursor]);
    }
    std::size_t scan = 0;
    int kept = 0;
    for (const auto& old : future_old) {
      while (scan < candidate.size() &&
             candidate[scan].source_step < old.source_step)
        ++scan;
      if (scan >= candidate.size() ||
          candidate[scan].source_step != old.source_step ||
          !same_action(candidate[scan].action, old.action)) {
        if (audit) ++audit->route_skeleton_v3_fail_reasons[1];
        return false;
      }
      ++kept;
      ++scan;
    }
    found->moves = std::move(overdue);
    found->moves.insert(found->moves.end(), candidate.begin(), candidate.end());
    found->cursor = 0;
    if (audit) {
      ++audit->route_skeleton_v3_rebases;
      audit->route_skeleton_v3_lcs_kept += kept;
    }
  }
  runtime.actors.erase(
      std::remove_if(runtime.actors.begin(), runtime.actors.end(),
                     [&](const auto& actor) {
                       return actor.generation !=
                                  event_local_actor_generation(day, actor.actor) ||
                           actor.actor >= static_cast<int>(actor_count);
                     }),
      runtime.actors.end());
  std::sort(runtime.actors.begin(), runtime.actors.end(),
            [](const auto& left, const auto& right) {
              return left.actor < right.actor;
            });
  return runtime.actors.size() == actor_count;
}

SkeletonObjective& ensure_skeleton_objective(
    SkeletonRuntime& runtime, Position tile, Item desired,
    bool requires_harvest) {
  const auto found = std::find_if(
      runtime.objectives.begin(), runtime.objectives.end(),
      [&](const auto& objective) {
        return objective.tile.x == tile.x && objective.tile.y == tile.y &&
            objective.desired == desired && skeleton_open(objective);
      });
  if (found != runtime.objectives.end()) return *found;
  SkeletonObjective objective;
  objective.id = runtime.next_objective_id++;
  objective.tile = tile;
  objective.desired = desired;
  objective.requires_harvest = requires_harvest;
  runtime.objectives.push_back(std::move(objective));
  return runtime.objectives.back();
}

bool absorb_current_crop_action(const Simulator& env, int player,
                                std::size_t actor, Action baseline,
                                SkeletonRuntime& runtime) {
  const auto current = positions(env, player);
  if (actor >= current.size()) return false;
  const Position position = current[actor];
  const Tile* tile = tile_at(env, player, position);
  if (!tile) return false;
  Item desired = crop_item(baseline.item) ? baseline.item : tile->crop;
  if (!crop_item(desired) && baseline.op != Op::DIG) return false;
  auto& objective = ensure_skeleton_objective(
      runtime, position, desired, baseline.op == Op::HARVEST);
  if (objective.waiting_maturity || !objective.remaining.empty()) return true;
  if (tile->kind == TileKind::WEED) {
    objective.remaining.push_back(to_day_horizon({Op::DIG}));
    if (crop_item(desired) && baseline.op != Op::DIG) {
      objective.remaining.push_back(
          to_day_horizon({Op::PLANT, desired, 1}));
      objective.remaining.push_back(
          to_day_horizon({Op::WATER, desired, 1}));
    }
  } else if (tile->kind == TileKind::EMPTY) {
    if (!crop_item(desired)) return false;
    objective.remaining.push_back(to_day_horizon({Op::PLANT, desired, 1}));
    objective.remaining.push_back(to_day_horizon({Op::WATER, desired, 1}));
  } else if (tile->kind == TileKind::PLANT) {
    desired = tile->crop;
    objective.desired = desired;
    if (baseline.op == Op::WATER && !tile->watered_today)
      objective.remaining.push_back(
          to_day_horizon({Op::WATER, desired, 1}));
    else if (baseline.op == Op::HARVEST) {
      if (to_event_local(env, tile).harvest_legal)
        objective.remaining.push_back(
            to_day_horizon({Op::HARVEST, desired, 1}));
      else
        objective.waiting_maturity = true;
    } else if (baseline.op == Op::PLANT ||
               (baseline.op == Op::WATER && tile->watered_today)) {
      const auto satisfied_id = objective.id;
      runtime.objectives.erase(std::remove_if(
          runtime.objectives.begin(), runtime.objectives.end(),
          [&](const auto& candidate) {
            return candidate.id == satisfied_id;
          }),
          runtime.objectives.end());
    }
  }
  return true;
}

void refresh_skeleton_wait_nodes(const Simulator& env, int player,
                                 SkeletonRuntime& runtime) {
  for (auto& objective : runtime.objectives) {
    if (!objective.waiting_maturity) continue;
    const Tile* tile = tile_at(env, player, objective.tile);
    if (tile && tile->kind == TileKind::PLANT &&
        tile->crop == objective.desired &&
        to_event_local(env, tile).harvest_legal && tile->yield_units > 0) {
      objective.waiting_maturity = false;
      objective.remaining = {
          to_day_horizon({Op::HARVEST, objective.desired, 1})};
    } else if (tile && tile->kind == TileKind::PLANT &&
               tile->crop == objective.desired && !tile->watered_today) {
      objective.waiting_maturity = false;
      objective.remaining = {
          to_day_horizon({Op::WATER, objective.desired, 1})};
    }
  }
}

void compose_rolling_route_skeleton_v3(
    const std::vector<PlayerAction>& tape, const Simulator& env, int player,
    NativeAgentState& state, PlayerAction& committed_out,
    NativeRepairAudit* audit) {
  auto& committed_runtime = state.experimental_route_skeleton_v3;
  // Receipts are observation commits and precede proposal arbitration. Every
  // other mutation below is staged and becomes visible only with the final
  // unit manifest.
  settle_route_skeleton_receipts(env, player, committed_runtime, audit);
  auto runtime = committed_runtime;
  auto out = committed_out;
  NativeRepairAudit* committed_audit = audit;
  NativeRepairAudit staged_audit = audit ? *audit : NativeRepairAudit{};
  audit = committed_audit ? &staged_audit : nullptr;
  struct ProposalGuard {
    NativeRepairAudit* committed{};
    NativeRepairAudit* staged{};
    bool accepted{};
    ~ProposalGuard() {
      if (accepted || !committed || !staged) return;
      const int fail_delta = staged->route_skeleton_v3_fail_closed -
          committed->route_skeleton_v3_fail_closed;
      committed->route_skeleton_v3_fail_closed += std::max(1, fail_delta);
      for (std::size_t reason = 0;
           reason < committed->route_skeleton_v3_fail_reasons.size(); ++reason)
        committed->route_skeleton_v3_fail_reasons[reason] += std::max(
            0, staged->route_skeleton_v3_fail_reasons[reason] -
                   committed->route_skeleton_v3_fail_reasons[reason]);
      committed->route_skeleton_v3_terminal_moves += std::max(
          0, staged->route_skeleton_v3_terminal_moves -
                 committed->route_skeleton_v3_terminal_moves);
    }
  } proposal_guard{committed_audit, audit};
  const auto current = positions(env, player);
  if (current.size() != out.units.size() ||
      !reconcile_route_skeleton(tape, env, out.units.size(), runtime, audit)) {
    if (audit) ++audit->route_skeleton_v3_fail_closed;
    return;
  }
  for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
    const Op operation = out.units[actor].op;
    if (operation == Op::PASS || movement(operation)) continue;
    if (operation == Op::DIG || operation == Op::PLANT ||
        operation == Op::WATER || operation == Op::HARVEST) {
      if (absorb_current_crop_action(
              env, player, actor, out.units[actor], runtime))
        continue;
    }
    // Carried inventory, animals, structures, fertilizer, and market-coupled
    // production have actor-specific state not represented by this planner.
    // Preserve the real baseline byte and decline ownership for this tick.
    if (audit) {
      ++audit->route_skeleton_v3_fail_closed;
      ++audit->route_skeleton_v3_fail_reasons[2];
    }
    return;
  }
  refresh_skeleton_wait_nodes(env, player, runtime);
  const int turns = std::min(
      static_cast<int>(tape.size()) - env.step_count(),
      env.config().turns_per_day - env.hour());
  if (turns <= 0) return;
  std::vector<day_horizon::ActorPlan> actors;
  actors.reserve(runtime.actors.size());
  for (const auto& skeleton : runtime.actors) {
    day_horizon::ActorPlan plan;
    plan.actor = skeleton.actor;
    plan.start = to_event_local(current[static_cast<std::size_t>(skeleton.actor)]);
    plan.raw.assign(static_cast<std::size_t>(turns), {});
    plan.blocked_equivalent_capacity.assign(
        static_cast<std::size_t>(turns), false);
    int slot = 0;
    for (std::size_t cursor = skeleton.cursor;
         cursor < skeleton.moves.size(); ++cursor) {
      slot = std::max(slot,
                      skeleton.moves[cursor].source_step - env.step_count());
      if (slot >= turns) {
        if (audit) {
          ++audit->route_skeleton_v3_fail_closed;
          ++audit->route_skeleton_v3_fail_reasons[3];
        }
        return;
      }
      plan.raw[static_cast<std::size_t>(slot)] =
          to_day_horizon(skeleton.moves[cursor].action);
      ++slot;
    }
    actors.push_back(std::move(plan));
  }
  std::vector<day_horizon::Objective> objectives;
  day_horizon::ObservationReadiness readiness;
  readiness.observation_epoch = static_cast<std::uint64_t>(env.step_count() + 1);
  readiness.consumed_harvest_legal = runtime.consumed_maturity_proofs;
  for (const auto& persistent : runtime.objectives) {
    if (persistent.waiting_maturity || persistent.remaining.empty()) continue;
    day_horizon::Objective objective;
    objective.id = persistent.id;
    objective.tile = to_event_local(persistent.tile);
    objective.remaining_transitions = persistent.remaining;
    objective.deadline_turn = std::max(
        0, static_cast<int>(persistent.remaining.size()) - 1);
    objective.value = persistent.requires_harvest ? 1000 : 500;
    objective.critical = true;
    if (persistent.remaining.front().op == event_local::Op::Harvest) {
      const day_horizon::HarvestProofToken proof{
          readiness.observation_epoch, persistent.id, 0};
      readiness.harvest_legal_required.insert({persistent.id, 0});
      readiness.harvest_legal.insert(proof);
    }
    objectives.push_back(std::move(objective));
  }
  day_horizon::ResourceSnapshot resources;
  for (int crop = 0; crop < N_CROPS; ++crop)
    resources.seeds[crop] =
        env.privates()[player].seeds[static_cast<std::size_t>(crop)];
  day_horizon::TimeExpandedResult planned;
  try {
    planned = day_horizon::compile_time_expanded(
        actors, objectives, resources, readiness);
  } catch (const std::exception&) {
    if (audit) {
      ++audit->route_skeleton_v3_fail_closed;
      ++audit->route_skeleton_v3_fail_reasons[4];
    }
    return;
  }
  if (!planned.ordered_move_sequences_exact ||
      planned.terminal_unexecuted_moves != 0 ||
      planned.terminal_unexecuted_raw_actions != 0) {
    if (audit) {
      ++audit->route_skeleton_v3_fail_closed;
      ++audit->route_skeleton_v3_fail_reasons[5];
      audit->route_skeleton_v3_terminal_moves +=
          planned.terminal_unexecuted_moves;
    }
    return;
  }
  if (audit) {
    if (runtime.planned_once) ++audit->route_skeleton_v3_replans;
    else ++audit->route_skeleton_v3_plans;
    audit->route_skeleton_v3_assignments +=
        static_cast<int>(planned.assignments.size());
    audit->route_skeleton_v3_fallback_plans += planned.fell_back_to_greedy;
    audit->route_skeleton_v3_budget_exhausted += !planned.within_time_budget;
  }
  runtime.planned_once = true;
  runtime.consumed_maturity_proofs.insert(
      planned.consumed_harvest_legal.begin(),
      planned.consumed_harvest_legal.end());
  std::map<int, day_horizon::Assignment> bindings;
  for (const auto& assignment : planned.assignments)
    if (assignment.turn == 0) bindings[assignment.actor] = assignment;
  for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
    out.units[actor] = from_day_horizon(planned.manifest[actor][0]);
    if (movement(out.units[actor].op)) {
      auto& skeleton = runtime.actors[actor];
      if (skeleton.cursor >= skeleton.moves.size() ||
          !same_action(out.units[actor], skeleton.moves[skeleton.cursor].action)) {
        if (audit) {
          ++audit->route_skeleton_v3_fail_closed;
          ++audit->route_skeleton_v3_fail_reasons[6];
        }
        return;
      }
      ++skeleton.cursor;
      if (audit) ++audit->route_skeleton_v3_moves_emitted;
    }
    const auto binding = bindings.find(static_cast<int>(actor));
    if (binding == bindings.end()) continue;
    NativeAgentState::RouteSkeletonPending pending;
    pending.objective_id = binding->second.objective_id;
    pending.actor = static_cast<int>(actor);
    pending.generation = runtime.actors[actor].generation;
    pending.tile = current[actor];
    pending.emitted = out.units[actor].op;
    const Tile* before = tile_at(env, player, pending.tile);
    pending.before = before ? *before : Tile{};
    pending.submitted_step = env.step_count();
    runtime.pending.push_back(std::move(pending));
  }
  // Make accepted postponement explicit.  This is staged with the proposal,
  // so a declined tick remains runtime-bit-identical and its baseline MOVE is
  // acknowledged on the next reconcile instead of being replayed.
  for (auto& skeleton : runtime.actors)
    for (std::size_t cursor = skeleton.cursor; cursor < skeleton.moves.size();
         ++cursor)
      if (skeleton.moves[cursor].source_step <= env.step_count())
        skeleton.moves[cursor].deferred_by_planner = true;
  committed_runtime = std::move(runtime);
  committed_out = std::move(out);
  if (committed_audit) *committed_audit = staged_audit;
  proposal_guard.accepted = true;
}

}  // namespace

std::uint64_t route_movement_fingerprint(
    const std::vector<PlayerAction>& tape, const Simulator& env,
    const joint_fixed_move_oracle::production::RouteCursorState& cursor) {
  std::uint64_t hash = 1469598103934665603ULL;
  const int turns = env.config().turns_per_day;
  const int end = std::min((env.day() + 1) * turns - 1,
                           static_cast<int>(tape.size()) - 1);
  for (std::size_t actor = 0; actor < cursor.source_cursor.size(); ++actor) {
    const int begin = std::max(env.day() * turns, cursor.source_cursor[actor]);
    for (int source = begin; source <= end; ++source) {
      if (actor >= tape[static_cast<std::size_t>(source)].units.size()) continue;
      const auto action = tape[static_cast<std::size_t>(source)].units[actor];
      if (!movement(action.op)) continue;
      hash ^= static_cast<std::uint64_t>(actor + 1);
      hash *= 1099511628211ULL;
      hash ^= static_cast<std::uint64_t>(source + 1);
      hash *= 1099511628211ULL;
      hash ^= static_cast<std::uint64_t>(static_cast<int>(action.op) + 1);
      hash *= 1099511628211ULL;
    }
  }
  return hash == 0 ? 1 : hash;
}

void enqueue_deferred_crop_sources(const Simulator& env,
                                   NativeAgentState& state) {
  if (!state.experimental_deferred_crop_scheduler) return;
  const auto& route = state.experimental_route_cursor;
  for (std::size_t actor = 0; actor < route.deferred_nonmoves.size(); ++actor) {
    for (const auto& deferred : route.deferred_nonmoves[actor]) {
      const auto key = std::pair{static_cast<int>(actor), deferred.source_step};
      if (state.experimental_deferred_crop_enqueued_sources.count(key)) continue;
      if (!g001::failure_debt::deferred_crop::is_crop_deferred_operation(
              deferred.action.op))
        continue;
      Item remembered = Item::NONE;
      const int tile = deferred.tile.y * env.config().board_size + deferred.tile.x;
      if (tile >= 0 && tile < static_cast<int>(route.production.last_crop.size()))
        remembered = route.production.last_crop[static_cast<std::size_t>(tile)];
      g001::failure_debt::deferred_crop::DeferredSource source;
      source.actor = static_cast<int>(actor);
      source.tile = deferred.tile;
      source.source_step = deferred.source_step;
      source.origin_day = deferred.source_step / env.config().turns_per_day;
      source.deadline_step = deferred.deadline_step;
      source.action = deferred.action;
      source.remembered_crop = remembered;
      source.critical = true;
      source.provenance = "native-route-cursor";
      if (state.experimental_deferred_crop_scheduler->enqueue(source))
        state.experimental_deferred_crop_enqueued_sources.insert(key);
    }
  }
}

void settle_deferred_crop_receipts(const Simulator& env, int player,
                                   NativeAgentState& state) {
  if (!state.experimental_deferred_crop_scheduler) return;
  for (const auto& pending : state.experimental_deferred_crop_pending) {
    if (!pending.active || env.step_count() != pending.submitted_step + 1)
      continue;
    g001::failure_debt::deferred_crop::Receipt receipt;
    receipt.step = env.step_count();
    receipt.actor = pending.proposal.actor;
    receipt.debt_id = pending.proposal.debt_id;
    receipt.obligation_id = pending.proposal.obligation_id;
    receipt.emitted = pending.emitted;
    receipt.before = pending.before;
    receipt.after = crop_snapshot(env, player, pending.proposal.tile);
    const auto actor = static_cast<std::size_t>(pending.proposal.actor);
    receipt.crop_inventory_delta = crop_item(pending.before.crop)
        ? actor_item_inventory(env, player, actor, pending.before.crop) -
              pending.crop_inventory_before
        : 0;
    receipt.fertilizer_inventory_delta =
        actor_item_inventory(env, player, actor, Item::FERTILIZER) -
        pending.fertilizer_inventory_before;
    receipt.day_end_water_effect_lower_bound =
        pending.day_end_water_effect_lower_bound;
    receipt.movement_hash = pending.movement_hash;
    receipt.provenance = "native-next-observation";
    (void)state.experimental_deferred_crop_scheduler->observe_receipt(receipt);
  }
  state.experimental_deferred_crop_pending.clear();

  std::set<std::pair<int, int>> completed_sources;
  for (const auto& debt : state.experimental_deferred_crop_scheduler->all_debts()) {
    if (!debt.completed) continue;
    for (const auto& obligation : debt.obligations)
      for (const int source_step : obligation.source_steps)
        completed_sources.insert({debt.actor, source_step});
  }
  for (std::size_t actor = 0;
       actor < state.experimental_route_cursor.deferred_nonmoves.size(); ++actor) {
    auto& actor_debts =
        state.experimental_route_cursor.deferred_nonmoves[actor];
    std::erase_if(actor_debts, [&](const auto& deferred) {
      return completed_sources.count(
                 {static_cast<int>(actor), deferred.source_step}) != 0;
    });
  }
}

std::array<int, N_ITEMS> projected_shed(const Simulator& env, int player,
                                         const PlayerAction& action) {
  auto projected = env.privates()[player].shed;
  const auto pos = positions(env, player);
  const auto& pr = env.privates()[player];
  for (size_t u = 0; u < action.units.size() && u < pos.size() &&
                     u < pr.inventories.size(); ++u) {
    if (!shed_adjacent(pos[u])) continue;
    const auto& a = action.units[u];
    const auto& inv = pr.inventories[u];
    if (a.op == Op::DROP) {
      // CPython dict insertion order is preserved by inventory_order.
      for (int item : pr.inventory_order[u]) {
        const int room = std::max(0, 100 - std::accumulate(
            projected.begin(), projected.end(), 0));
        projected[item] += std::min(std::max(0, inv[item]), room);
      }
    } else if (a.op == Op::PLACE) {
      const int item = int(a.item);
      if (item < 0 || item >= N_ITEMS) continue;
      const Tile* t = tile_at(env, player, pos[u]);
      if (item >= 9 && t &&
          ((item == 9 && t->kind == TileKind::COOP) ||
           (item >= 10 && t->kind == TileKind::PASTURE)) &&
          t->animal == Item::NONE) continue;
      const int room = std::max(0, 100 - std::accumulate(
          projected.begin(), projected.end(), 0));
      projected[item] += std::min({quantity(a), std::max(0, inv[item]), room});
    }
  }
  return projected;
}

struct PublicSignature {
  int hands = 0, lands = 0;
  std::array<int, 11> counts{};
};

PublicSignature signature(const Farm& farm) {
  PublicSignature s;
  s.hands = int(farm.hands.size());
  s.lands = std::popcount(unsigned(farm.unlocked_mask));
  for (const Tile& t : farm.tiles) {
    if (t.kind == TileKind::PLANT) s.counts[int(t.crop)]++;
    else if (t.kind == TileKind::ANIMAL) s.counts[5 + int(t.animal) - 9]++;
    else if (t.kind == TileKind::PASTURE) s.counts[8]++;
    else if (t.kind == TileKind::COOP) s.counts[9]++;
    else if (t.kind == TileKind::WEED) s.counts[10]++;
  }
  return s;
}

int clone_distance(const Simulator& env) {
  const auto a = signature(env.farms()[0]);
  const auto b = signature(env.farms()[1]);
  int d = std::abs(a.hands - b.hands) + 3 * std::abs(a.lands - b.lands);
  for (size_t i = 0; i < a.counts.size(); ++i) d += std::abs(a.counts[i] - b.counts[i]);
  return d;
}

int animal_count(const Farm& farm, Item item) {
  return std::count_if(farm.tiles.begin(), farm.tiles.end(),
                       [item](const Tile& t) {
                         return t.kind == TileKind::ANIMAL && t.animal == item;
                       });
}

int shop_demand(const Simulator& env, int item, int step) {
  int demand = item != 8 && step % 24 == 0 ? 1 : 0;
  if (step % 4 != 0) return demand;
  // BAKERY, BRUNCH, FARMERS, ICE_CREAM, PET, PIZZA, SMOOTHIE, YARN.
  static constexpr uint16_t masks[8] = {
      (1u << 5) | (1u << 0),
      (1u << 5) | (1u << 0) | (1u << 3),
      (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3),
      (1u << 3) | (1u << 6) | (1u << 0),
      (1u << 1),
      (1u << 6) | (1u << 2) | (1u << 0),
      (1u << 3) | (1u << 6),
      (1u << 7)};
  for (int sh : env.shops()) {
    if (sh >= 0 && sh < 8 && (masks[sh] & (1u << item)))
      demand += std::popcount(unsigned(masks[sh])) == 1 ? 2 : 1;
  }
  return demand;
}

double curve(const char* name, double value) {
  value = std::max(0.0, value);
  if (name[0] == 's' && name[1] == 'q' && name[2] == '\0') return value * value;
  if (name[0] == 's') return std::sqrt(value);
  if (name[0] == 'l' && name[3] == '\0') return std::log1p(value);
  return value;
}

int k320_market_price(int item, int inventory) {
  struct P { int base, eq, scale; const char *below, *above; double bt, at; };
  static constexpr P p[9] = {
      {25,10000,400,"sqrt","log",.8,.2}, {35,10000,450,"log","sqrt",.2,.7},
      {60,10000,200,"linear","sqrt",.4,.6}, {120,10000,100,"sqrt","linear",.7,1.6},
      {250,10000,300,"log","sq",.2,3.6}, {50,10000,332,"linear","log",.4,.2},
      {160,10000,122,"sqrt","linear",.6,1.6}, {200,10000,105,"log","sq",.2,3.2},
      {100,10000,200,"linear","linear",.4,.4}};
  const auto& x = p[item];
  double value;
  if (inventory < x.eq)
    value = x.base + x.bt * x.base / curve(x.below, x.scale) *
                         curve(x.below, x.eq - inventory);
  else
    value = x.base - x.at * x.base / curve(x.above, x.scale) *
                         curve(x.above, inventory - x.eq);
  return std::max(1, int(std::nearbyint(value)));
}

double order_score(const Simulator& env, const Action& a) {
  if (!sell(a)) return -std::numeric_limits<double>::infinity();
  const int item = int(a.item), q = quantity(a), inv = env.market().inventory[item];
  const double current = env.market().prices[item];
  const double later = k320_market_price(item, inv + q);
  double score = q * std::max(0.0, current - later);
  if (score <= 0) return score;
  static constexpr uint16_t masks[8] = {33,41,15,73,2,69,72,128};
  double demand = item == 8 ? 0.0 : 1.0;
  for (int sh : env.shops()) if (masks[sh] & (1u << item))
    demand += 6.0 * (std::popcount(unsigned(masks[sh])) == 1 ? 2 : 1);
  demand = std::max(0.25, demand);
  const double excess = std::max(0, inv + q - 10000);
  const double urgency = std::min(1.0, excess / demand / 10.0);
  return score * (1.0 + .25 * urgency);
}

void rank_sell_slots(const Simulator& env, PlayerAction& a) {
  struct Row { double score; int neg_index; Action action; };
  std::vector<Row> rows;
  for (size_t i = 0; i < a.market.size(); ++i)
    if (sell(a.market[i])) rows.push_back({order_score(env, a.market[i]), -int(i), a.market[i]});
  if (rows.size() < 2) return;
  std::sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) {
    return std::tie(x.score, x.neg_index) > std::tie(y.score, y.neg_index);
  });
  size_t r = 0;
  for (auto& order : a.market) if (sell(order)) order = rows[r++].action;
}

int planned_sell(const PlayerAction& a, int item) {
  int n = 0;
  for (const auto& x : a.market) if (sell(x) && int(x.item) == item) n += quantity(x);
  return n;
}

int pickup_reserve(const PlayerAction& a, int item) {
  int n = 0;
  for (const auto& x : a.units)
    if (x.op == Op::PICKUP && int(x.item) == item) n += quantity(x);
  return n;
}

Action tape_unit(const std::vector<PlayerAction>& tape, int step, int actor) {
  if (step < 0 || step >= int(tape.size()) || actor < 0 ||
      actor >= int(tape[step].units.size())) return {};
  return tape[step].units[actor];
}

std::vector<g001::repair::TimedAction> realignment_window(
    const std::vector<PlayerAction>& tape, int start_step, int actor,
    Position start_position, int maximum_lookahead) {
  const int day_end = (start_step / 24 + 1) * 24 - 1;
  const int last = std::min({int(tape.size()) - 1, day_end,
                             start_step + std::max(0, maximum_lookahead)});
  std::vector<g001::repair::TimedAction> result;
  Position position = start_position;
  for (int source_step = start_step; source_step <= last; ++source_step) {
    const Action action = tape_unit(tape, source_step, actor);
    g001::repair::TimedAction timed;
    timed.action = action;
    timed.required_position = position;
    timed.earliest_step = 0;
    timed.latest_step = std::max(0, last - start_step + 1);
    timed.economic_value = g001::repair::default_economic_value(action.op);
    timed.expected_fill = timed.economic_value > 0;
    result.push_back(timed);
    if (action.op == Op::NORTH) --position.y;
    else if (action.op == Op::SOUTH) ++position.y;
    else if (action.op == Op::EAST) ++position.x;
    else if (action.op == Op::WEST) --position.x;
  }
  return result;
}

std::optional<g001::repair::AlignmentPlan> minimum_loss_plan(
    const std::vector<PlayerAction>& tape, int start_step, int actor,
    Position start_position, int maximum_lookahead, int minimum_skip = 0) {
  auto window = realignment_window(tape, start_step, actor, start_position,
                                   maximum_lookahead);
  if (window.empty()) return std::nullopt;
  auto plan = g001::repair::search_minimum_loss_realign(
      window, start_position, int(window.size()) - 1, minimum_skip);
  if (plan.metrics.skipped_source_index < 0 ||
      plan.metrics.movement_edits != 0) return std::nullopt;
  return plan;
}

std::array<int, 3> placed_animals(const Farm& farm) {
  std::array<int, 3> result{};
  for (const auto& tile : farm.tiles) {
    if (tile.kind == TileKind::ANIMAL && int(tile.animal) >= 9 &&
        int(tile.animal) < 12) ++result[int(tile.animal) - 9];
  }
  return result;
}

std::array<int, 3> total_animals(const Simulator& env, int player) {
  std::array<int, 3> result = placed_animals(env.farms()[player]);
  const auto& private_state = env.privates()[player];
  for (int index = 0; index < 3; ++index) {
    result[index] += private_state.shed[index + 9];
    for (const auto& inventory : private_state.inventories)
      result[index] += inventory[index + 9];
  }
  return result;
}

struct AnimalRouteWindow {
  int pickup_step = -1;
  int place_step = -1;
  int actor = -1;
  bool pickup_realign_safe = false;
};

int fib_cost(int index);

AnimalRouteWindow next_animal_route_window(
    const std::vector<PlayerAction>& tape, int step, int animal_item,
    const std::vector<Position>& current_positions, int maximum_lookahead) {
  AnimalRouteWindow result;
  // Bought animals persist in the shed across the day boundary.  Unlike a
  // delayed unit transaction, its causal pickup window must therefore scan
  // into the next day's re-hire/reset programme.
  const int last = std::min(int(tape.size()) - 1,
                            step + std::max(1, maximum_lookahead));
  for (int source = step; source <= last; ++source) {
    for (int actor = 0; actor < int(tape[source].units.size()); ++actor) {
      const auto& action = tape[source].units[actor];
      if (action.op != Op::PICKUP || int(action.item) != animal_item) continue;
      for (int place = source + 1; place <= last; ++place) {
        if (actor >= int(tape[place].units.size())) continue;
        const auto& candidate = tape[place].units[actor];
        if (candidate.op != Op::PLACE || int(candidate.item) != animal_item) continue;
        result = {source, place, actor, false};
        if (source == step && actor < int(current_positions.size())) {
          result.pickup_realign_safe = minimum_loss_plan(
              tape, step, actor, current_positions[actor], maximum_lookahead, 1).has_value();
        }
        return result;
      }
    }
  }
  return result;
}

int planned_market_cash(const Simulator& env, int player,
                        const PlayerAction& action) {
  static constexpr int seed_cost[5] = {10, 20, 50, 100, 80};
  static constexpr int animal_cost[3] = {300, 400, 500};
  static constexpr int land_cost[3] = {1000, 2000, 4000};
  int reserve = 0;
  int hires = env.farms()[player].hires_today;
  int lands = std::popcount(unsigned(env.farms()[player].unlocked_mask));
  for (const auto& order : action.market) {
    const int item = int(order.item), q = std::max(0, quantity(order));
    if (order.op == Op::HIRE) reserve += fib_cost(hires++);
    else if (order.op == Op::BUY_LAND) {
      const int extra = std::max(0, lands - 1);
      if (extra < 3) reserve += land_cost[extra];
      ++lands;
    } else if (order.op == Op::BUY_SEED && item >= 0 && item < 5)
      reserve += q * seed_cost[item];
    else if (order.op == Op::BUY_PRODUCT && (item == 0 || item == 8))
      reserve += q * env.market().prices[item];
    else if (order.op == Op::BUY_ANIMAL && item >= 9 && item < 12)
      reserve += q * animal_cost[item - 9];
  }
  return reserve;
}

void merge_sale(PlayerAction& a, int item, int q) {
  if (q <= 0) return;
  for (auto& x : a.market) if (sell(x) && int(x.item) == item) {
    x.quantity = quantity(x) + q;
    return;
  }
  if (a.market.size() < 10)
    a.market.push_back(Action{Op::SELL, Item(item), q});
}

Action move_toward(Position p, Position target) {
  if (p.x < target.x) return Action{Op::EAST};
  if (p.x > target.x) return Action{Op::WEST};
  if (p.y < target.y) return Action{Op::SOUTH};
  if (p.y > target.y) return Action{Op::NORTH};
  return Action{};
}

int moon_route(const Simulator& env) {
  if (!env.shops().empty() && env.shops()[0] == 7) return 3;
  if (env.shops().size() >= 2 &&
      (env.shops()[0] == 7 || env.shops()[1] == 7)) return 4;
  if (env.shops().size() >= 3 &&
      (env.shops()[0] == 7 || env.shops()[1] == 7 || env.shops()[2] == 7)) return 2;
  for (size_t i = 0; i < std::min<size_t>(3, env.shops().size()); ++i)
    if (env.shops()[i] == 3 || env.shops()[i] == 5 || env.shops()[i] == 6) return 0;
  return 1;
}

int tape_sell(const std::vector<PlayerAction>& tape, int step, int item) {
  if (step < 0 || step >= int(tape.size())) return 0;
  return planned_sell(tape[step], item);
}

struct ThomasPrefixMarketState {
  struct CourierPlan {
    int start = -1;
    std::vector<Action> route;
  };
  struct CowSwapEntry {
    int step = -1;
    int actor = -1;
    Position site{};
    int quantity = 1;
  };
  struct PendingCattlePlace {
    int actor = -1;
    Position site{};
    int day = -1;
  };
  int due_step = -1;
  std::array<int, N_PRODUCTS> due{};
  std::array<std::array<int16_t, N_PRODUCTS>, 720> reserve_debts{};
  int cattle_confirmed = 0;
  int cattle_reserved = 0;
  bool cattle_pending_buy = false;
  int cattle_pending_before = 0;
  int cattle_pending_quantity = 0;
  std::array<int8_t, 32> cattle_carrying{};
  std::vector<PendingCattlePlace> cattle_pending_places;
  bool herd2_decided = false;
  Item herd2_mode = Item::NONE;
  bool cowswap_decided = false;
  bool cowswap_active = false;
  bool cowswap_broken = false;
  std::vector<CowSwapEntry> cowswap_builds;
  std::vector<CowSwapEntry> cowswap_pickups;
  std::vector<CowSwapEntry> cowswap_places;
  int cowswap_credit = 0;
  std::array<int, N_PRODUCTS> capharv_credit{};
  int courier_day = -1;
  std::array<CourierPlan, 32> courier_plans{};
  std::array<std::array<int16_t, 5>, 720> predictor_observed{};
  bool predictor_prev_valid = false;
  bool predictor_best_valid = false;
  int predictor_best = -1;
  int predictor_prev_step = -1;
  std::array<int, N_PRODUCTS> predictor_prev_inventory{};
  std::array<int, N_PRODUCTS> predictor_prev_prices{};
  std::array<int, 5> predictor_prev_own{};
  std::vector<int8_t> predictor_prev_shops;
};

struct ThomasHerdSpec {
  int cost;
  int first;
  int interval;
  double units;
  int product;
};

ThomasHerdSpec thomas_herd_spec(Item animal) {
  if (animal == Item::GOOSE) return {300, 4, 1, 1.8, int(Item::EGG)};
  if (animal == Item::COW) return {400, 8, 2, 2.6, int(Item::MILK)};
  return {500, 6, 3, 3.4, int(Item::WOOL)};
}

void add_thomas_herd_schedule(std::array<double, 30>& out, Item animal,
                              int placed_day, int day_from, double count = 1.0) {
  const auto spec = thomas_herd_spec(animal);
  for (int day = std::max(day_from, placed_day + spec.first); day < 30; ++day)
    if ((day - placed_day - spec.first) % spec.interval == 0)
      out[day] += spec.units * count;
}

bool thomas_family_similar(const Simulator& env, int player) {
  const auto& own = env.farms()[player];
  const auto& rival = env.farms()[1 - player];
  if (own.unlocked_mask != rival.unlocked_mask) return false;
  int matches = 0, total = 0;
  for (size_t i = 0; i < own.tiles.size() && i < rival.tiles.size(); ++i) {
    const auto& a = own.tiles[i];
    const auto& b = rival.tiles[i];
    const bool visible = a.crop != Item::NONE || a.animal != Item::NONE ||
                         b.crop != Item::NONE || b.animal != Item::NONE;
    if (!visible) continue;
    ++total;
    matches += a.crop == b.crop && a.animal == b.animal;
  }
  return total >= 8 && matches >= 0.9 * total;
}

double thomas_herd_ev(const Simulator& env, int player,
                      const std::vector<PlayerAction>& tape, Item option, int k) {
  const auto spec = thomas_herd_spec(option);
  const Item existing_animal = spec.product == int(Item::EGG) ? Item::GOOSE
      : spec.product == int(Item::MILK) ? Item::COW : Item::SHEEP;
  const int day = env.day();
  std::array<std::array<double, 30>, 2> existing{};
  for (int farm_index = 0; farm_index < 2; ++farm_index)
    for (const Tile& tile : env.farms()[farm_index].tiles)
      if (tile.animal == existing_animal)
        add_thomas_herd_schedule(existing[farm_index], existing_animal,
                                  tile.placed_day, day + 1);
  int held = env.privates()[player].shed[int(existing_animal)];
  for (const auto& inventory : env.privates()[player].inventories)
    held += inventory[int(existing_animal)];
  add_thomas_herd_schedule(existing[player], existing_animal,
                            day + 1, day + 1, held);
  const bool similar = thomas_family_similar(env, player);
  for (int step = env.step_count() + 1; step < int(tape.size()) && step < 696; ++step)
    for (const auto& order : tape[step].market)
      if (order.op == Op::BUY_ANIMAL && order.item == existing_animal) {
        const int qty = std::max(0, quantity(order));
        add_thomas_herd_schedule(existing[player], existing_animal,
                                  step / 24 + 1, day + 1, qty);
        if (similar)
          add_thomas_herd_schedule(existing[1 - player], existing_animal,
                                    step / 24 + 1, day + 1, qty);
      }
  std::array<double, 30> added{};
  add_thomas_herd_schedule(added, option, day + 1, day + 1, k);

  static constexpr uint16_t masks[8] = {33, 41, 15, 73, 2, 69, 72, 128};
  double daily_demand = 1.0;
  for (int shop : env.shops()) if (masks[shop] & (1u << spec.product))
    daily_demand += 6.0 * (std::popcount(unsigned(masks[shop])) == 1 ? 2 : 1);
  auto path = [&](bool with_new, std::array<int, 30>& prices) {
    double inventory = env.market().inventory[spec.product];
    double revenue = 0.0;
    for (int future_day = day + 1; future_day < 30; ++future_day) {
      inventory -= daily_demand;
      inventory += existing[player][future_day] + existing[1 - player][future_day];
      prices[future_day] = k320_market_price(
          spec.product, int(std::nearbyint(inventory)));
      if (!with_new) continue;
      const int units = int(std::nearbyint(added[future_day]));
      for (int unit = 0; unit < units; ++unit) {
        const int price = k320_market_price(
            spec.product, int(std::nearbyint(inventory)));
        revenue += price;
        if (price > 1) inventory += 1.0;
      }
    }
    return revenue;
  };
  std::array<int, 30> base_prices{}, new_prices{};
  path(false, base_prices);
  const double revenue = path(true, new_prices);
  double swing = 0.0;
  for (int future_day = day + 1; future_day < 30; ++future_day)
    swing += (new_prices[future_day] - base_prices[future_day]) *
             (existing[player][future_day] - existing[1 - player][future_day]);
  return revenue + swing - spec.cost * k;
}

void apply_thomas_herd2(const Simulator& env, int player,
                        const std::vector<PlayerAction>& tape,
                        PlayerAction& action, ThomasPrefixMarketState& state) {
  if (!state.herd2_decided) {
    int goose_buys = 0;
    for (const auto& order : action.market)
      if (order.op == Op::BUY_ANIMAL && order.item == Item::GOOSE)
        goose_buys += std::max(0, quantity(order));
    if (goose_buys > 0) {
      state.herd2_decided = true;
      bool coop_exists = false;
      for (const Tile& tile : env.farms()[player].tiles)
        coop_exists |= tile.kind == TileKind::COOP || tile.animal == Item::GOOSE;
      if (env.step_count() >= 192 && env.step_count() < 360 && !coop_exists) {
        const int planned = std::max(goose_buys, 3);
        const double goose = thomas_herd_ev(env, player, tape, Item::GOOSE, planned);
        const double cow = thomas_herd_ev(env, player, tape, Item::COW, planned);
        const double sheep = thomas_herd_ev(env, player, tape, Item::SHEEP, planned);
        const Item best = cow >= sheep ? Item::COW : Item::SHEEP;
        const double best_value = std::max(cow, sheep);
        const int extra_cost = (thomas_herd_spec(best).cost - 300) * goose_buys;
        if (best_value - goose >= 600.0 &&
            best_value >= 1.3 * std::max(goose, 1.0) &&
            env.farms()[player].money >= 300 * goose_buys + extra_cost + 50)
          state.herd2_mode = best;
      }
    }
  }
  if (state.herd2_mode == Item::NONE) return;
  const auto spec = thomas_herd_spec(state.herd2_mode);
  double cash = env.farms()[player].money;
  static constexpr int seed_cost[5] = {10, 20, 50, 100, 80};
  for (auto& order : action.market) {
    const int qty = std::max(0, quantity(order));
    if (order.op == Op::BUY_ANIMAL && order.item == Item::GOOSE) {
      order.item = state.herd2_mode;
      order.quantity = std::min(qty, int(std::max(0.0, cash) / spec.cost));
      cash -= order.quantity * spec.cost;
    } else if (order.op == Op::BUY_ANIMAL) {
      const int cost = order.item == Item::COW ? 400 : order.item == Item::SHEEP ? 500 : 300;
      cash -= qty * cost;
    } else if (order.op == Op::BUY_SEED && int(order.item) < N_CROPS) {
      cash -= qty * seed_cost[int(order.item)];
    } else if (order.op == Op::BUY_PRODUCT && int(order.item) < N_PRODUCTS) {
      cash -= qty * env.market().prices[int(order.item)];
    } else if (order.op == Op::BUY_LAND) {
      cash -= 4000;
    }
  }
  const Item product = Item(spec.product);
  for (auto& command : action.units) {
    if (command.op == Op::BUILD_COOP) command.op = Op::BUILD_PASTURE;
    else if ((command.op == Op::PICKUP || command.op == Op::PLACE) &&
             command.item == Item::GOOSE)
      command.item = state.herd2_mode;
    else if (command.op == Op::PLACE && command.item == Item::EGG)
      command.item = product;
  }
}

Position thomas_spawn(const std::vector<Position>& positions) {
  static constexpr std::array<Position, 4> access{{
      {4, 4}, {5, 4}, {4, 5}, {5, 5}}};
  Position best = access[0];
  int best_count = std::count_if(positions.begin(), positions.end(),
                                 [&](Position p) { return at(p, best); });
  for (Position candidate : access) {
    const int count = std::count_if(positions.begin(), positions.end(),
                                    [&](Position p) { return at(p, candidate); });
    if (count < best_count) { best = candidate; best_count = count; }
  }
  return best;
}

bool plan_thomas_cowswap(const Simulator& env, int player,
                         const std::vector<PlayerAction>& tape,
                         const PlayerAction& action, int wanted,
                         ThomasPrefixMarketState& state) {
  struct Seen { int step, actor, quantity; Position site; };
  std::vector<Seen> builds, pickups, places;
  auto pos = positions(env, player);
  const int step = env.step_count(), end = (step / 24 + 1) * 24 - 1;
  for (int future = step; future <= end && future < int(tape.size()); ++future) {
    const PlayerAction& frame = future == step ? action : tape[future];
    for (int actor = 0; actor < int(pos.size()); ++actor) {
      const Action command = actor < int(frame.units.size()) ? frame.units[actor] : Action{};
      const Position site = pos[actor];
      if (movement(command.op)) {
        if (command.op == Op::EAST) ++pos[actor].x;
        else if (command.op == Op::WEST) --pos[actor].x;
        else if (command.op == Op::SOUTH) ++pos[actor].y;
        else if (command.op == Op::NORTH) --pos[actor].y;
      } else if (command.op == Op::BUILD_PASTURE) {
        builds.push_back({future, actor, 1, site});
      } else if (command.op == Op::PICKUP && command.item == Item::COW) {
        pickups.push_back({future, actor, quantity(command), site});
      } else if (command.op == Op::PLACE && command.item == Item::COW) {
        places.push_back({future, actor, 1, site});
      }
    }
    for (const Action& order : frame.market)
      if (order.op == Op::HIRE) pos.push_back(thomas_spawn(pos));
  }
  std::vector<Seen> chosen;
  for (const Seen& place : places) {
    if (std::none_of(chosen.begin(), chosen.end(), [&](const Seen& row) {
          return at(row.site, place.site);
        }))
      chosen.push_back(place);
    if (int(chosen.size()) == wanted) break;
  }
  if (int(chosen.size()) < wanted) return false;
  for (const Seen& place : chosen) {
    const Tile* tile = tile_at(env, player, place.site);
    const bool buying_land = std::any_of(action.market.begin(), action.market.end(),
                                        [](Action order) { return order.op == Op::BUY_LAND; });
    if (!tile || (tile->kind != TileKind::EMPTY &&
                  !(tile->kind == TileKind::LOCKED && buying_land)))
      return false;
    auto build = std::find_if(builds.rbegin(), builds.rend(), [&](const Seen& row) {
      return row.step < place.step && at(row.site, place.site);
    });
    auto pickup = std::find_if(pickups.rbegin(), pickups.rend(), [&](const Seen& row) {
      return row.step < place.step && row.actor == place.actor;
    });
    if (build == builds.rend() || pickup == pickups.rend()) return false;
    state.cowswap_builds.push_back({build->step, build->actor, build->site, 1});
    state.cowswap_pickups.push_back({pickup->step, pickup->actor, pickup->site, 1});
    state.cowswap_places.push_back({place.step, place.actor, place.site, 1});
  }
  for (const Seen& pickup : pickups) {
    const int used = std::count_if(state.cowswap_pickups.begin(), state.cowswap_pickups.end(),
        [&](const auto& row) { return row.step == pickup.step && row.actor == pickup.actor; });
    if (used && used != pickup.quantity) return false;
  }
  return true;
}

void apply_thomas_cowswap(const Simulator& env, int player,
                          const std::vector<PlayerAction>& tape,
                          PlayerAction& action, ThomasPrefixMarketState& state) {
  const int step = env.step_count();
  if (!state.cowswap_decided && step >= 144 && step < 192) {
    int cow_buys = 0;
    for (const Action& order : action.market)
      if (order.op == Op::BUY_ANIMAL && order.item == Item::COW)
        cow_buys += quantity(order);
    if (cow_buys > 0) {
      state.cowswap_decided = true;
      bool shop_ok = true;
      for (int shop : env.shops())
        if (shop == 3 || shop == 5 || shop == 6 || shop == 7) shop_ok = false;
      if (shop_ok) {
        const double cow = thomas_herd_ev(env, player, tape, Item::COW, cow_buys);
        const double goose = thomas_herd_ev(env, player, tape, Item::GOOSE, cow_buys);
        if (goose - cow >= 600.0 && goose >= 1.3 * std::max(cow, 1.0) &&
            plan_thomas_cowswap(env, player, tape, action, cow_buys, state)) {
          state.cowswap_active = true;
          for (auto& order : action.market)
            if (order.op == Op::BUY_ANIMAL && order.item == Item::COW)
              order.item = Item::GOOSE;
        }
      }
    }
  }
  if (!state.cowswap_active || state.cowswap_broken) return;
  const auto current_positions = positions(env, player);
  const auto rewrite = [&](const std::vector<ThomasPrefixMarketState::CowSwapEntry>& rows,
                           Op expected, Op replacement, Item item) {
    for (const auto& row : rows) {
      if (row.step != step || row.actor < 0 || row.actor >= int(action.units.size()) ||
          row.actor >= int(current_positions.size())) continue;
      Action& command = action.units[row.actor];
      if (command.op != expected || !at(current_positions[row.actor], row.site) ||
          (item != Item::NONE && command.item != Item::COW)) {
        state.cowswap_broken = true;
        return;
      }
      command.op = replacement;
      if (item != Item::NONE) command.item = item;
    }
  };
  rewrite(state.cowswap_builds, Op::BUILD_PASTURE, Op::BUILD_COOP, Item::NONE);
  rewrite(state.cowswap_pickups, Op::PICKUP, Op::PICKUP, Item::GOOSE);
  rewrite(state.cowswap_places, Op::PLACE, Op::PLACE, Item::GOOSE);
}

void apply_thomas_capharv(const Simulator& env, int player,
                          const std::vector<PlayerAction>& tape,
                          PlayerAction& action, ThomasPrefixMarketState& state) {
  const int step = env.step_count(), day = env.day();
  const auto start_positions = positions(env, player);
  const auto& private_state = env.privates()[player];
  for (int actor = 0; actor < int(action.units.size()) &&
                      actor < int(start_positions.size()); ++actor) {
    Action& command = action.units[actor];
    if (command.op != Op::CARE && command.op != Op::COLLECT_FERTILIZER) continue;
    const Tile* tile = tile_at(env, player, start_positions[actor]);
    if (!tile || tile->kind != TileKind::ANIMAL) continue;
    int product = -1, cap = 0, first = 0, interval = 0;
    if (tile->animal == Item::GOOSE) {
      product = int(Item::EGG); cap = 4; first = 4; interval = 1;
    } else if (tile->animal == Item::COW) {
      product = int(Item::MILK); cap = 6; first = 8; interval = 2;
    } else if (tile->animal == Item::SHEEP) {
      product = int(Item::WOOL); cap = 6; first = 6; interval = 3;
    } else continue;
    const int since = day + 1 - tile->placed_day - first;
    if (since < 0 || since % interval != 0) continue;
    bool future_harvest = false, future_feed = false, future_collect = false;
    Position pos = start_positions[actor];
    for (int future = step + 1; future < (day + 1) * 24 &&
                              future < int(tape.size()); ++future) {
      const Action next = actor < int(tape[future].units.size())
          ? tape[future].units[actor] : Action{};
      if (movement(next.op)) {
        if (next.op == Op::EAST) ++pos.x;
        else if (next.op == Op::WEST) --pos.x;
        else if (next.op == Op::SOUTH) ++pos.y;
        else if (next.op == Op::NORTH) --pos.y;
      } else if (at(pos, start_positions[actor])) {
        future_harvest |= next.op == Op::HARVEST;
        future_feed |= next.op == Op::FEED;
        future_collect |= next.op == Op::COLLECT_FERTILIZER;
      }
    }
    if (future_harvest) continue;
    const bool fed = tile->fed_today || future_feed;
    const int production = 1 + (fed ? std::max(0, int(tile->pending_care_bonus)) : 0);
    const int overflow = tile->yield_units + production - cap;
    if (overflow <= 0 || tile->yield_units <= 0) continue;
    int saved = 0;
    if (command.op == Op::COLLECT_FERTILIZER) {
      if (overflow * env.market().prices[product] <=
          env.market().prices[int(Item::FERTILIZER)]) continue;
      saved = overflow;
    } else {
      if (future_collect || overflow <= 1) continue;
      saved = overflow - 1;
    }
    int carried = 0;
    for (const auto& inventory : private_state.inventories)
      carried += std::accumulate(inventory.begin(), inventory.end(), 0);
    if (shed_sum(private_state) + carried + tile->yield_units >= 90) continue;
    command = Action{Op::HARVEST};
    state.capharv_credit[product] += saved;
  }
}

void observe_thomas_cowswap_harvest(const Simulator& env, int player,
                                    const PlayerAction& action,
                                    ThomasPrefixMarketState& state) {
  if (!state.cowswap_active) return;
  const auto pos = positions(env, player);
  for (int actor = 0; actor < int(action.units.size()) && actor < int(pos.size()); ++actor) {
    if (action.units[actor].op != Op::HARVEST) continue;
    const bool swapped = std::any_of(state.cowswap_places.begin(), state.cowswap_places.end(),
        [&](const auto& row) { return at(row.site, pos[actor]); });
    const Tile* tile = tile_at(env, player, pos[actor]);
    if (swapped && tile && tile->kind == TileKind::ANIMAL && tile->animal == Item::GOOSE)
      state.cowswap_credit += std::max(0, int(tile->yield_units));
  }
}

void apply_thomas_overlay_credit_sales(const Simulator& env, int player,
                                       PlayerAction& action,
                                       ThomasPrefixMarketState& state) {
  const auto stock = projected_shed(env, player, action);
  const auto sell_credit = [&](int product, int& credit) {
    if (credit <= 0 || action.market.size() >= 10 || env.market().prices[product] < 2)
      return;
    const int extra = std::min(credit,
        std::max(0, stock[product] - planned_sell(action, product)));
    if (extra <= 0) return;
    for (auto& order : action.market)
      if (order.op == Op::SELL && int(order.item) == product) {
        order.quantity = quantity(order) + extra;
        credit -= extra;
        return;
      }
    action.market.insert(action.market.begin(), Action{Op::SELL, Item(product), extra});
    credit -= extra;
  };
  for (int product = 0; product < N_PRODUCTS; ++product)
    sell_credit(product, state.capharv_credit[product]);
  sell_credit(int(Item::EGG), state.cowswap_credit);
}

void apply_thomas_cattle_substitution(
    const Simulator& env, int player, PlayerAction& action,
    ThomasPrefixMarketState& state) {
  const auto& farm = env.farms()[player];
  const auto& private_state = env.privates()[player];
  const auto unit_positions = positions(env, player);
  if (state.cattle_pending_buy) {
    const int gained = std::max(0, private_state.shed[int(Item::COW)] -
                                      state.cattle_pending_before);
    const int confirmed = std::min(state.cattle_pending_quantity, gained);
    state.cattle_confirmed += confirmed;
    state.cattle_reserved += confirmed;
    state.cattle_pending_buy = false;
  }
  for (const auto& pending : state.cattle_pending_places) {
    const Tile* tile = tile_at(env, player, pending.site);
    if (tile && tile->animal == Item::COW && tile->placed_day == pending.day &&
        pending.actor >= 0 && pending.actor < int(state.cattle_carrying.size()))
      state.cattle_carrying[pending.actor] = std::max<int>(
          0, state.cattle_carrying[pending.actor] - 1);
  }
  state.cattle_pending_places.clear();

  int cow_available = private_state.shed[int(Item::COW)];
  std::set<std::pair<int, int>> occupied;
  for (int actor = 0; actor < int(action.units.size()) &&
                      actor < int(unit_positions.size()) &&
                      actor < int(private_state.inventories.size()); ++actor) {
    auto& command = action.units[actor];
    const auto& inventory = private_state.inventories[actor];
    const Position pos = unit_positions[actor];
    const Tile* tile = tile_at(env, player, pos);
    if (command.op == Op::PICKUP && command.item == Item::SHEEP) {
      const int qty = std::max(0, quantity(command));
      const int center = env.config().board_size / 2;
      int animal_cargo = 0;
      for (int animal = int(Item::GOOSE); animal <= int(Item::SHEEP); ++animal)
        animal_cargo += inventory[animal];
      if (qty > 0 && state.cattle_reserved >= qty && cow_available >= qty &&
          (pos.x == center - 1 || pos.x == center) &&
          (pos.y == center - 1 || pos.y == center) && animal_cargo == 0) {
        command.item = Item::COW;
        state.cattle_reserved -= qty;
        cow_available -= qty;
        if (actor < int(state.cattle_carrying.size()))
          state.cattle_carrying[actor] += int8_t(qty);
      }
    }
    if (command.op == Op::PLACE && command.item == Item::SHEEP &&
        actor < int(state.cattle_carrying.size()) &&
        state.cattle_carrying[actor] > 0 && inventory[int(Item::COW)] > 0 &&
        tile && tile->kind == TileKind::PASTURE && tile->animal == Item::NONE &&
        !occupied.count({pos.x, pos.y})) {
      command.item = Item::COW;
      state.cattle_pending_places.push_back({actor, pos, env.day()});
    }
    if (command.op == Op::PLACE &&
        int(command.item) >= int(Item::GOOSE) && int(command.item) <= int(Item::SHEEP) &&
        inventory[int(command.item)] > 0)
      occupied.insert({pos.x, pos.y});
  }

  const int step = env.step_count();
  if (step < 216 || step > 227 || env.shops().size() < 3 ||
      state.cattle_confirmed >= 4 || state.cattle_reserved != 0 ||
      !state.cattle_pending_places.empty())
    return;
  for (int carried : state.cattle_carrying) if (carried) return;
  int cargo = 0;
  for (const auto& inventory : private_state.inventories)
    for (int animal = int(Item::GOOSE); animal <= int(Item::SHEEP); ++animal)
      cargo += inventory[animal];
  for (int animal = int(Item::GOOSE); animal <= int(Item::SHEEP); ++animal)
    cargo += private_state.shed[animal];
  if (cargo != 0) return;

  Action* animal_order = nullptr;
  int animal_orders = 0;
  for (auto& order : action.market) if (order.op == Op::BUY_ANIMAL) {
    animal_order = &order;
    ++animal_orders;
  }
  if (animal_orders != 1 || !animal_order || animal_order->item != Item::SHEEP)
    return;
  int cows = 0, sheep = 0;
  for (const Tile& tile : farm.tiles) {
    cows += tile.animal == Item::COW;
    sheep += tile.animal == Item::SHEEP;
  }
  int milk_shops = 0;
  bool yarn = false;
  for (int shop : env.shops()) {
    milk_shops += shop == 3 || shop == 5 || shop == 6;
    yarn |= shop == 7;
  }
  const int qty = quantity(*animal_order);
  if (milk_shops < 2 || yarn ||
      env.market().prices[int(Item::MILK)] < env.market().prices[int(Item::WOOL)] ||
      cows < 4 || sheep < 2 || qty < 1 || qty > 2 ||
      qty > 4 - state.cattle_confirmed)
    return;
  animal_order->item = Item::COW;
  state.cattle_pending_buy = true;
  state.cattle_pending_before = private_state.shed[int(Item::COW)];
  state.cattle_pending_quantity = qty;
}

void apply_thomas_wheat_reservation(
    const Simulator& env, int player, const std::vector<PlayerAction>& tape,
    PlayerAction& action, ThomasPrefixMarketState& state) {
  const int step = env.step_count();
  const int item = int(Item::WHEAT);
  if (step >= 0 && step < int(state.reserve_debts.size())) {
    int debt = state.reserve_debts[step][item];
    for (auto& order : action.market) {
      if (debt <= 0) break;
      if (order.op != Op::SELL || int(order.item) != item) continue;
      const int removed = std::min(std::max(0, quantity(order)), debt);
      order.quantity = quantity(order) - removed;
      debt -= removed;
    }
  }
  if (step < 192 || step >= 288 || env.market().prices[item] <= 25 ||
      action.market.size() >= 10)
    return;
  for (const auto& order : action.market)
    if (int(order.item) == item &&
        (order.op == Op::SELL || order.op == Op::BUY_PRODUCT))
      return;
  for (const auto& command : action.units)
    if (command.op == Op::PICKUP && int(command.item) == item) return;

  int available = std::max(0, projected_shed(env, player, action)[item]);
  int reserved = 0;
  for (int due = step + 1;
       due < int(tape.size()) && due <= std::min(695, step + 40); ++due) {
    if (pickup_reserve(tape[due], item) > 0) break;
    bool future_buy = false;
    for (const auto& order : tape[due].market)
      if (order.op == Op::BUY_PRODUCT && int(order.item) == item) future_buy = true;
    if (future_buy) break;
    const int planned = planned_sell(tape[due], item);
    const int owed = due < int(state.reserve_debts.size())
        ? state.reserve_debts[due][item] : 0;
    const int amount = std::min(available, std::max(0, planned - owed));
    if (amount <= 0) continue;
    state.reserve_debts[due][item] += int16_t(amount);
    available -= amount;
    reserved += amount;
    if (available <= 0) break;
  }
  if (reserved > 0)
    action.market.push_back(Action{Op::SELL, Item::WHEAT, reserved});
}

void apply_thomas_wheat_replenishment_trim(
    const Simulator& env, int player, const std::vector<PlayerAction>& tape,
    PlayerAction& action) {
  // Thomas R95 retains enough physical grain for the next two complete days,
  // but trims the tape's fixed-size replenishment once that reserve is funded.
  const int step = env.step_count();
  if (step / 24 < 10 || step / 24 > 11) return;
  bool has_buy = false;
  for (const auto& order : action.market) {
    has_buy |= order.op == Op::BUY_PRODUCT && order.item == Item::WHEAT &&
               quantity(order) > 0;
    if (order.op == Op::SELL && order.item == Item::WHEAT) return;
  }
  if (!has_buy) return;

  int reserve = 6;
  for (int future = step + 1;
       future < int(tape.size()) && future < step + 49; ++future) {
    reserve += pickup_reserve(tape[future], int(Item::WHEAT));
    reserve += planned_sell(tape[future], int(Item::WHEAT));
  }
  if (std::count(env.shops().begin(), env.shops().end(), int8_t(7)) >= 2) {
    std::array<bool, 30> days{};
    for (int future = step + 1; future < step + 49; ++future)
      if (future / 24 >= 12 && future / 24 < int(days.size()))
        days[future / 24] = true;
    reserve += 6 * std::count(days.begin(), days.end(), true);
  }

  std::array<PlayerAction, 2> joint{};
  joint[player] = action;
  const Simulator after_units = env.preview_unit_phase(joint);
  int held = std::max(0, after_units.privates()[player].shed[int(Item::WHEAT)]);
  for (auto& order : action.market) {
    if (order.op != Op::BUY_PRODUCT || order.item != Item::WHEAT) continue;
    const int retained = std::min(quantity(order), std::max(0, reserve - held));
    order.quantity = retained;  // Keep the slot even when the quantity is zero.
    held += retained;
  }
}

void apply_thomas_egg_reservation(
    const Simulator& env, int player, const std::vector<PlayerAction>& tape,
    PlayerAction& action, ThomasPrefixMarketState& state) {
  const int step = env.step_count(), item = int(Item::EGG);
  if (step >= 0 && step < int(state.reserve_debts.size())) {
    int debt = state.reserve_debts[step][item];
    for (auto& order : action.market) {
      if (debt <= 0) break;
      if (order.op != Op::SELL || int(order.item) != item) continue;
      const int removed = std::min(quantity(order), debt);
      order.quantity = quantity(order) - removed;
      debt -= removed;
    }
  }
  if (step < 192 || step >= 288 || env.market().prices[item] <= 50 ||
      action.market.size() >= 10) return;
  for (const auto& order : action.market)
    if (int(order.item) == item &&
        (order.op == Op::SELL || order.op == Op::BUY_PRODUCT)) return;
  for (const auto& command : action.units)
    if (command.op == Op::PICKUP && int(command.item) == item) return;
  int available = std::max(0, projected_shed(env, player, action)[item]);
  int reserved = 0;
  for (int due = step + 1;
       due < int(tape.size()) && due <= std::min(695, step + 40); ++due) {
    if (pickup_reserve(tape[due], item) > 0) break;
    bool future_buy = false;
    for (const auto& order : tape[due].market)
      future_buy |= order.op == Op::BUY_PRODUCT && int(order.item) == item;
    if (future_buy) break;
    const int planned = planned_sell(tape[due], item);
    const int owed = state.reserve_debts[due][item];
    const int amount = std::min(available, std::max(0, planned - owed));
    if (amount <= 0) continue;
    state.reserve_debts[due][item] += int16_t(amount);
    available -= amount;
    reserved += amount;
    if (available <= 0) break;
  }
  if (reserved > 0)
    action.market.push_back(Action{Op::SELL, Item::EGG, reserved});
}

bool thomas_courier_idle(Action action) {
  return action.op == Op::PASS || movement(action.op) || action.op == Op::DROP;
}

void apply_thomas_courier(const Simulator& env, int player,
                          const std::vector<PlayerAction>& tape,
                          PlayerAction& action, ThomasPrefixMarketState& state) {
  const int step = env.step_count(), day = env.day();
  if (step >= 718 || step % 24 < 12) return;
  if (state.courier_day != day) {
    state.courier_day = day;
    for (auto& plan : state.courier_plans) plan = {};
  }
  const int end = day * 24 + 23;
  int crew = 1;
  for (int future = day * 24;
       future <= end && future < int(tape.size()); ++future)
    crew = std::max(crew, int(tape[future].units.size()));
  const auto unit_positions = positions(env, player);
  const auto& private_state = env.privates()[player];
  action.units.resize(unit_positions.size());
  static constexpr std::array<Position, 4> access{{
      {4, 4}, {5, 4}, {4, 5}, {5, 5}}};
  static constexpr std::array<int, 4> premium{{
      int(Item::STRAWBERRY), int(Item::MILK), int(Item::WOOL), int(Item::MELON)}};
  std::vector<std::pair<int, int>> delivered;
  bool changed = false;
  for (int actor = 0; actor < crew && actor < int(unit_positions.size()) &&
                      actor < int(state.courier_plans.size()); ++actor) {
    auto& plan = state.courier_plans[actor];
    bool cargo = false;
    if (actor < int(private_state.inventories.size()))
      for (int item : premium)
        cargo |= private_state.inventories[actor][item] > 0;
    if (plan.start < 0) {
      if (!cargo || !thomas_courier_idle(action.units[actor])) continue;
      bool idle = true;
      for (int future = step + 1; future <= end && future < int(tape.size()); ++future) {
        const Action command = actor < int(tape[future].units.size())
            ? tape[future].units[actor] : Action{};
        if (!thomas_courier_idle(command)) { idle = false; break; }
      }
      if (!idle) continue;
      const Position pos = unit_positions[actor];
      Position target = access[0];
      for (Position candidate : access)
        if (distance(pos, candidate) < distance(pos, target)) target = candidate;
      std::vector<Action> route;
      for (int x = pos.x; x < target.x; ++x) route.push_back(Action{Op::EAST});
      for (int x = pos.x; x > target.x; --x) route.push_back(Action{Op::WEST});
      for (int y = pos.y; y < target.y; ++y) route.push_back(Action{Op::SOUTH});
      for (int y = pos.y; y > target.y; --y) route.push_back(Action{Op::NORTH});
      route.push_back(Action{Op::DROP});
      if (int(route.size()) > end - step + 1) continue;
      plan.start = step;
      plan.route = std::move(route);
    }
    const int index = step - plan.start;
    if (index < 0 || index >= int(plan.route.size())) continue;
    const Action command = plan.route[index];
    if (command.op == Op::DROP) {
      if (!shed_adjacent(unit_positions[actor])) {
        plan.route.clear();
        plan.start = step;
        continue;
      }
      if (actor < int(private_state.inventories.size()))
        for (int item : premium) {
          const int quantity = private_state.inventories[actor][item];
          if (quantity > 0) delivered.push_back({item, quantity});
        }
    }
    action.units[actor] = command;
    changed = true;
  }
  if (!changed || delivered.empty()) return;
  std::stable_sort(delivered.begin(), delivered.end(), [&](auto left, auto right) {
    return env.market().prices[left.first] * left.second >
           env.market().prices[right.first] * right.second;
  });
  for (auto [item, qty] : delivered) {
    if (env.market().prices[item] < 2) continue;
    int existing = -1;
    for (int i = 0; i < int(action.market.size()); ++i)
      if (action.market[i].op == Op::SELL && int(action.market[i].item) == item) {
        existing = i;
        break;
      }
    if (existing >= 0) {
      Action sale = action.market[existing];
      sale.quantity = quantity(sale) + qty;
      action.market.erase(action.market.begin() + existing);
      action.market.insert(action.market.begin(), sale);
    } else if (action.market.size() < 10) {
      action.market.insert(action.market.begin(), Action{Op::SELL, Item(item), qty});
    }
  }
}

double thomas_quote_priority(const Simulator& env, int player,
                             const std::array<int, N_ITEMS>& stock,
                             const Action& order) {
  const int item = int(order.item);
  if (order.op != Op::SELL || item < 0 || item >= N_PRODUCTS) return 0.0;
  const int qty = std::min(quantity(order), std::max(0, stock[item]));
  if (qty <= 0) return 0.0;
  int standing = 0;
  for (const Tile& tile : env.farms()[1 - player].tiles) {
    if (item < N_CROPS && tile.kind == TileKind::PLANT && int(tile.crop) == item)
      standing += std::max(0, int(tile.yield_units));
    else if (item >= int(Item::EGG) && item <= int(Item::WOOL) &&
             tile.kind == TileKind::ANIMAL &&
             int(tile.animal) == item - int(Item::EGG) + int(Item::GOOSE))
      standing += std::max(0, int(tile.yield_units));
  }
  const int batch = std::min(24, std::max(8, standing));
  const int inventory = env.market().inventory[item];
  double score = 0.0;
  for (int unit = 0; unit < qty; ++unit)
    score += k320_market_price(item, inventory + unit) -
             k320_market_price(item, inventory + batch + unit);
  return score;
}

void apply_thomas_order_priority(const Simulator& env, int player,
                                 PlayerAction& action) {
  const auto stock = projected_shed(env, player, action);
  std::array<bool, N_ITEMS> bought{};
  struct Sale { int index; double score; Action order; };
  std::vector<Sale> movable;
  std::vector<std::pair<int, Action>> fixed;
  for (int index = 0; index < int(action.market.size()); ++index) {
    const Action order = action.market[index];
    const int item = int(order.item);
    if ((order.op == Op::BUY_PRODUCT || order.op == Op::BUY_ANIMAL) &&
        item >= 0 && item < N_ITEMS)
      bought[item] = true;
    if (order.op == Op::SELL && quantity(order) > 0 && item >= 0 &&
        item < N_ITEMS && !bought[item])
      movable.push_back({index, thomas_quote_priority(env, player, stock, order), order});
    else
      fixed.push_back({index, order});
  }
  if (movable.empty()) return;
  std::stable_sort(movable.begin(), movable.end(), [](const Sale& left, const Sale& right) {
    return left.score > right.score;
  });
  action.market.clear();
  for (const auto& sale : movable) action.market.push_back(sale.order);
  for (const auto& row : fixed) action.market.push_back(row.second);
}

constexpr int THOMAS_PREDICT_ITEMS[5] = {
    int(Item::MILK), int(Item::WOOL), int(Item::STRAWBERRY),
    int(Item::EGG), int(Item::MELON)};

int thomas_town_draw(const std::vector<int8_t>& shops, int step, int item) {
  int draw = step % 24 == 0 && item < int(Item::FERTILIZER) ? 1 : 0;
  if (step % 4 != 0) return draw;
  for (int shop : shops) {
    if (item == int(Item::MILK) && (shop == 3 || shop == 5 || shop == 6)) ++draw;
    else if (item == int(Item::WOOL) && shop == 7) draw += 2;
    else if (item == int(Item::STRAWBERRY) && (shop == 1 || shop == 3 || shop == 6)) ++draw;
    else if (item == int(Item::EGG) && (shop == 0 || shop == 1)) ++draw;
  }
  return draw;
}

bool thomas_stream_near(
    const std::vector<NativeTapeLibrary::ThomasPredictEvent>& stream,
    int step, int item) {
  for (const auto& event : stream)
    if (event.item == item && std::abs(int(event.step) - step) <= 1) return true;
  return false;
}

void apply_thomas_predict(
    const Simulator& env, int player, const std::vector<PlayerAction>& tape,
    const std::vector<std::vector<NativeTapeLibrary::ThomasPredictEvent>>& streams,
    PlayerAction& action, ThomasPrefixMarketState& state) {
  const int step = env.step_count();
  if (state.predictor_prev_valid && state.predictor_prev_step == step - 1) {
    for (int i = 0; i < 5; ++i) {
      const int item = THOMAS_PREDICT_ITEMS[i];
      if (state.predictor_prev_prices[item] <= 3) continue;
      const int sold = env.market().inventory[item] -
          state.predictor_prev_inventory[item] +
          thomas_town_draw(state.predictor_prev_shops,
                           state.predictor_prev_step, item) -
          state.predictor_prev_own[i];
      if (sold >= 2) state.predictor_observed[step - 1][i] =
          int16_t(std::min(sold, 32767));
    }
  }

  if (step >= 150 && step < 700 && !streams.empty()) {
    if (step % 3 == 0 || !state.predictor_best_valid) {
      const int lo = std::max(0, step - 240);
      int best_score = std::numeric_limits<int>::min();
      int best_index = -1;
      for (int index = 0; index < int(streams.size()); ++index) {
        const auto& stream = streams[index];
        int matches = 0, false_events = 0, misses = 0;
        for (const auto& event : stream) {
          if (event.step < lo || event.step >= step - 1) continue;
          bool seen = false;
          for (int offset = -1; offset <= 1; ++offset) {
            const int observed_step = int(event.step) + offset;
            if (observed_step >= 0 && observed_step < 720 &&
                state.predictor_observed[observed_step][event.item] > 0)
              seen = true;
          }
          seen ? ++matches : ++false_events;
        }
        for (int observed_step = lo; observed_step < step; ++observed_step)
          for (int item = 0; item < 5; ++item)
            if (state.predictor_observed[observed_step][item] > 0 &&
                !thomas_stream_near(stream, observed_step, item))
              ++misses;
        const int score = 2 * matches - false_events - misses;
        if (score > best_score) {
          best_score = score;
          best_index = index;
        }
      }
      state.predictor_best = best_index;
      state.predictor_best_valid = best_index >= 0;
    }
    if (state.predictor_best_valid) {
      const auto& best = streams[state.predictor_best];
      auto stock = projected_shed(env, player, action);
      for (int forecast_item = 0; forecast_item < 3; ++forecast_item) {
        const int item = THOMAS_PREDICT_ITEMS[forecast_item];
        bool blocked = false;
        for (const auto& order : action.market)
          if (int(order.item) == item &&
              (order.op == Op::SELL || order.op == Op::BUY_PRODUCT))
            blocked = true;
        if (blocked || action.market.size() >= 10) continue;
        int votes = 0;
        for (const auto& event : best)
          if (event.item == forecast_item &&
              (event.step == step + 1 || event.step == step + 2))
            votes += event.quantity;
        if (votes < 4) continue;
        int ours = 0;
        for (int future = step + 1;
             future < int(tape.size()) && future <= step + 48; ++future)
          ours += planned_sell(tape[future], item);
        const int qty = std::min(stock[item], ours);
        if (qty > 0)
          action.market.insert(action.market.begin(),
                               Action{Op::SELL, Item(item), qty});
      }
    }
  }

  state.predictor_prev_valid = true;
  state.predictor_prev_step = step;
  state.predictor_prev_inventory = env.market().inventory;
  state.predictor_prev_prices = env.market().prices;
  state.predictor_prev_shops = env.shops();
  state.predictor_prev_own.fill(0);
  auto stock = projected_shed(env, player, action);
  std::array<int, 5> used{};
  for (const auto& order : action.market) {
    if (order.op != Op::SELL) continue;
    for (int i = 0; i < 5; ++i) if (int(order.item) == THOMAS_PREDICT_ITEMS[i]) {
      const int filled = std::min(std::max(0, quantity(order)),
                                  std::max(0, stock[int(order.item)] - used[i]));
      used[i] += filled;
      state.predictor_prev_own[i] += filled;
    }
  }
}

struct ThomasMarketProjection {
  std::array<int, N_ITEMS> stock{};
  std::vector<int> buys;
  std::vector<int> sales;
  std::array<int, N_ITEMS> loss{};
};

ThomasMarketProjection thomas_market_projection(
    const PrivateState& private_state, const std::vector<Action>& orders,
    bool night) {
  ThomasMarketProjection out;
  out.stock = private_state.shed;
  out.buys.assign(orders.size(), 0);
  out.sales.assign(orders.size(), 0);
  auto stock_total = [&]() {
    return std::accumulate(out.stock.begin(), out.stock.end(), 0);
  };
  for (size_t index = 0; index < orders.size(); ++index) {
    const auto& order = orders[index];
    const int item = int(order.item);
    if (item < 0 || item >= N_ITEMS) continue;
    const int requested = std::max(0, quantity(order));
    if (order.op == Op::SELL) {
      const int filled = std::min(requested, std::max(0, out.stock[item]));
      out.stock[item] -= filled;
      out.sales[index] = filled;
    } else if (order.op == Op::BUY_PRODUCT || order.op == Op::BUY_ANIMAL) {
      const int filled = std::min(requested, std::max(0, 100 - stock_total()));
      out.stock[item] += filled;
      out.buys[index] = filled;
    }
  }
  if (night) {
    for (size_t actor = 0; actor < private_state.inventories.size(); ++actor) {
      for (int item : private_state.inventory_order[actor]) {
        const int held = std::max(0, private_state.inventories[actor][item]);
        const int deposited = std::min(held, std::max(0, 100 - stock_total()));
        out.stock[item] += deposited;
        out.loss[item] += held - deposited;
      }
    }
  }
  return out;
}

int thomas_fib(int n) {
  int a = 1, b = 1;
  while (n-- > 0) {
    const int next = a + b;
    a = b;
    b = next;
  }
  return a;
}

int thomas_conservative_product_price(const Simulator& env, int item) {
  const int inventory = env.market().inventory[item] - 2000;
  const int gap = std::max(0, 10000 - inventory);
  if (item == int(Item::WHEAT))
    return std::max(1, int(std::nearbyint(25.0 +
        (0.8 * 25.0 / std::sqrt(400.0)) * std::sqrt(double(gap)))));
  return std::max(1, int(std::nearbyint(100.0 +
      (0.4 * 100.0 / 200.0) * double(gap))));
}

bool thomas_market_budget(const Simulator& env, int player,
                          const std::vector<Action>& orders) {
  static constexpr int seed_cost[N_CROPS] = {10, 20, 50, 100, 80};
  static constexpr int animal_cost[N_ANIMALS] = {300, 400, 500};
  long long cost = 0;
  int hires = env.farms()[player].hires_today;
  for (const auto& order : orders) {
    const int item = int(order.item);
    const int requested = std::max(0, quantity(order));
    if (order.op == Op::HIRE) cost += thomas_fib(hires++);
    else if (order.op == Op::BUY_LAND) cost += 4000;
    else if (order.op == Op::BUY_PRODUCT &&
             (item == int(Item::WHEAT) || item == int(Item::FERTILIZER)))
      cost += requested * thomas_conservative_product_price(env, item);
    else if (order.op == Op::BUY_ANIMAL && item >= int(Item::GOOSE) &&
             item <= int(Item::SHEEP))
      cost += requested * animal_cost[item - int(Item::GOOSE)];
    else if (order.op == Op::BUY_SEED && item >= 0 && item < N_CROPS)
      cost += requested * seed_cost[item];
  }
  return cost <= env.farms()[player].money;
}

int thomas_pickup_wheat(const PlayerAction& action) {
  int demand = 0;
  for (const auto& command : action.units)
    if (command.op == Op::PICKUP && command.item == Item::WHEAT)
      demand += std::max(0, quantity(command));
  return demand;
}

void apply_thomas_supply_guard(const Simulator& env, int player,
                               const std::vector<PlayerAction>& tape,
                               PlayerAction& action) {
  const int step = env.step_count();
  if (step < 144 || step >= 695 || step + 2 >= int(tape.size())) return;
  const auto& future = tape[step + 1];
  const auto& following = tape[step + 2];
  int prefund = 0;
  bool wheat_trade = false;
  for (const auto& order : future.market)
    if (order.item == Item::WHEAT &&
        (order.op == Op::BUY_PRODUCT || order.op == Op::SELL))
      wheat_trade = true;
  if (future.market.size() == 10 && !wheat_trade &&
      thomas_pickup_wheat(following) > 0)
    prefund = thomas_pickup_wheat(future) + thomas_pickup_wheat(following);
  if (prefund == 0 && thomas_pickup_wheat(future) == 0) return;
  if (action.market.size() > 10 || !thomas_market_budget(env, player, action.market))
    return;

  std::array<PlayerAction, 2> joint{};
  joint[player] = action;
  const Simulator preview = env.preview_unit_phase(joint);
  const auto& farm = preview.farms()[player];
  const auto& private_state = preview.privates()[player];
  std::vector<Position> actor_positions;
  if (step % 24 == 23) {
    actor_positions.push_back(Position{4, 4});
  } else {
    actor_positions.push_back(farm.farmer);
    actor_positions.insert(actor_positions.end(), farm.hands.begin(), farm.hands.end());
    static constexpr Position access[4] = {{4,4}, {5,4}, {4,5}, {5,5}};
    for (const auto& order : action.market) {
      if (order.op != Op::HIRE) continue;
      int counts[4]{};
      for (const auto& position : actor_positions)
        for (int i = 0; i < 4; ++i)
          if (position.x == access[i].x && position.y == access[i].y) ++counts[i];
      int best = 0;
      for (int i = 1; i < 4; ++i) if (counts[i] < counts[best]) best = i;
      actor_positions.push_back(access[best]);
    }
  }
  int need = prefund;
  for (size_t actor = 0; actor < actor_positions.size() &&
                         actor < future.units.size(); ++actor) {
    const auto& position = actor_positions[actor];
    const auto& command = future.units[actor];
    if (shed_adjacent(position) && command.op == Op::PICKUP &&
        command.item == Item::WHEAT)
      need += prefund ? 0 : std::max(0, quantity(command));
  }
  if (need <= 0) return;

  const bool night = step % 24 == 23;
  const auto original = thomas_market_projection(private_state, action.market, night);
  if (original.stock[int(Item::WHEAT)] >= need) return;
  auto proposed = action.market;
  const auto safe = [&](const ThomasMarketProjection& candidate) {
    for (size_t i = 0; i < original.buys.size(); ++i)
      if (candidate.buys[i] < original.buys[i]) return false;
    for (int item = 0; item < N_ITEMS; ++item)
      if (candidate.loss[item] > original.loss[item]) return false;
    return true;
  };
  for (int index = int(proposed.size()) - 1; index >= 0; --index) {
    if (proposed[index].op != Op::SELL || proposed[index].item != Item::WHEAT)
      continue;
    const auto before = thomas_market_projection(private_state, proposed, night);
    const int shortage = std::max(0, need - before.stock[int(Item::WHEAT)]);
    if (shortage == 0) break;
    if (before.sales[index] == 0) continue;
    const int old = quantity(proposed[index]);
    proposed[index].quantity = std::max(0, before.sales[index] - shortage);
    const auto after = thomas_market_projection(private_state, proposed, night);
    if (!safe(after) || after.stock[int(Item::WHEAT)] <= before.stock[int(Item::WHEAT)])
      proposed[index].quantity = old;
  }
  auto final = thomas_market_projection(private_state, proposed, night);
  const int shortage = std::max(0, need - final.stock[int(Item::WHEAT)]);
  if (shortage > 0) {
    int last_sale = -1, buy = -1;
    for (int i = 0; i < int(proposed.size()); ++i)
      if (proposed[i].op == Op::SELL && proposed[i].item == Item::WHEAT)
        last_sale = i;
    for (int i = int(proposed.size()) - 1; i > last_sale; --i)
      if (proposed[i].op == Op::BUY_PRODUCT && proposed[i].item == Item::WHEAT) {
        buy = i;
        break;
      }
    if (buy >= 0) proposed[buy].quantity = std::max(0, quantity(proposed[buy])) + shortage;
    else if (proposed.size() < 10)
      proposed.push_back(Action{Op::BUY_PRODUCT, Item::WHEAT, shortage});
    else return;
  }
  final = thomas_market_projection(private_state, proposed, night);
  if (!safe(final) || final.stock[int(Item::WHEAT)] < need ||
      !thomas_market_budget(env, player, proposed))
    return;
  action.market = std::move(proposed);
}

void apply_thomas_prefix_market(const Simulator& env, int player,
                                const std::vector<PlayerAction>& tape,
                                const std::vector<std::vector<NativeTapeLibrary::ThomasPredictEvent>>& predict_streams,
                                PlayerAction& action,
                                ThomasPrefixMarketState& state) {
  const auto sales_first = [&action]() {
    for (size_t index = 0; index < action.market.size(); ++index) {
      if (!sell(action.market[index])) continue;
      size_t cursor = index;
      while (cursor > 0) {
        const Action& previous = action.market[cursor - 1];
        if (sell(previous) ||
            ((previous.op == Op::BUY_PRODUCT || previous.op == Op::BUY_ANIMAL) &&
             previous.item == action.market[cursor].item))
          break;
        std::swap(action.market[cursor - 1], action.market[cursor]);
        --cursor;
      }
    }
  };
  const int step = env.step_count();
  if (step == 0) {
    action.market = {{Op::BUY_PRODUCT, Item::WHEAT, 20},
                     {Op::SELL, Item::WHEAT, 15}};
  } else if (step == 1) {
    action.market.erase(std::remove_if(action.market.begin(), action.market.end(),
        [](const Action& order) {
          return order.item == Item::WHEAT &&
                 (order.op == Op::BUY_PRODUCT || order.op == Op::SELL);
        }), action.market.end());
  }

  apply_thomas_cattle_substitution(env, player, action, state);
  apply_thomas_herd2(env, player, tape, action, state);
  apply_thomas_cowswap(env, player, tape, action, state);
  apply_thomas_capharv(env, player, tape, action, state);
  observe_thomas_cowswap_harvest(env, player, action, state);
  if (state.due_step == step) {
    for (auto& order : action.market) {
      if (!sell(order)) continue;
      const int item = int(order.item);
      const int removed = std::min(quantity(order), state.due[item]);
      order.quantity = quantity(order) - removed;
      state.due[item] -= removed;
    }
    state.due_step = -1;
    state.due.fill(0);
  }

  apply_thomas_wheat_reservation(env, player, tape, action, state);
  apply_thomas_wheat_replenishment_trim(env, player, tape, action);
  apply_thomas_egg_reservation(env, player, tape, action, state);

  const int next = step + 1;
  if (step >= 288 || next >= int(tape.size()) || next % 72 == 0 || step % 4 == 0) {
    apply_thomas_supply_guard(env, player, tape, action);
    apply_thomas_courier(env, player, tape, action, state);
    apply_thomas_predict(env, player, tape, predict_streams, action, state);
    apply_thomas_order_priority(env, player, action);
    apply_thomas_overlay_credit_sales(env, player, action, state);
    sales_first();
    return;
  }
  static constexpr int bases[N_PRODUCTS] = {25, 35, 60, 120, 250, 50, 160, 200, 100};
  auto projected = projected_shed(env, player, action);
  for (int item = 0; item < N_PRODUCTS && action.market.size() < 10; ++item) {
    const int planned = planned_sell(tape[next], item);
    if (planned <= 0 || planned_sell(action, item) > 0 ||
        env.market().prices[item] <= bases[item])
      continue;
    const int quantity = std::min(projected[item], planned);
    if (quantity <= 0) continue;
    action.market.push_back(Action{Op::SELL, Item(item), quantity});
    projected[item] -= quantity;
    state.due[item] += quantity;
    state.due_step = next;
  }

  apply_thomas_supply_guard(env, player, tape, action);
  apply_thomas_courier(env, player, tape, action, state);
  apply_thomas_predict(env, player, tape, predict_streams, action, state);
  apply_thomas_order_priority(env, player, action);
  apply_thomas_overlay_credit_sales(env, player, action, state);

  // Thomas' final wrapper gives sales priority over unrelated purchases.  Keep
  // a same-product BUY_PRODUCT/BUY_ANIMAL barrier: crossing it would change the
  // quantity available to the sale.  Unlike the post-144 v224 helper, the
  // opening wrapper retains zero-quantity debt markers.
  sales_first();
}

int macro_unit_failures(const Simulator& env, int player,
                        const PlayerAction& actions) {
  auto farm = env.farms()[player];
  auto private_state = env.privates()[player];
  std::array<int, N_CROPS> demand{};
  for (const auto& action : actions.units)
    if (action.op == Op::PLANT && int(action.item) >= 0 && int(action.item) < N_CROPS)
      demand[int(action.item)]++;
  std::array<bool, N_CROPS> blocked{};
  for (int item = 0; item < N_CROPS; ++item)
    blocked[item] = demand[item] > private_state.seeds[item];
  int failures = 0;
  for (size_t actor = 0; actor < actions.units.size(); ++actor) {
    const auto& action = actions.units[actor];
    const bool macro = action.op == Op::PLANT || action.op == Op::BUILD_COOP ||
        action.op == Op::BUILD_PASTURE ||
        (action.op == Op::PLACE && int(action.item) >= int(Item::GOOSE) &&
         int(action.item) <= int(Item::SHEEP));
    if (!macro) continue;
    if (actor > farm.hands.size()) { failures++; continue; }
    const Position position = actor == 0 ? farm.farmer : farm.hands[actor - 1];
    const int tile_index = position.y * env.config().board_size + position.x;
    const Tile& tile = farm.tiles[tile_index];
    if (action.op == Op::PLANT) {
      const int item = int(action.item);
      if (item < 0 || item >= N_CROPS || blocked[item] || tile.kind != TileKind::EMPTY) {
        failures++;
      } else {
        private_state.seeds[item]--;
        farm.tiles[tile_index].kind = TileKind::PLANT;
      }
    } else if (action.op == Op::BUILD_COOP || action.op == Op::BUILD_PASTURE) {
      if (tile.kind != TileKind::EMPTY) failures++;
      else farm.tiles[tile_index].kind = action.op == Op::BUILD_COOP
          ? TileKind::COOP : TileKind::PASTURE;
    } else {
      const int item = int(action.item);
      const TileKind required = item == int(Item::GOOSE) ? TileKind::COOP : TileKind::PASTURE;
      if (tile.kind != required || tile.animal != Item::NONE ||
          actor >= private_state.inventories.size() ||
          private_state.inventories[actor][item] <= 0) {
        failures++;
      } else {
        private_state.inventories[actor][item]--;
        farm.tiles[tile_index].kind = TileKind::ANIMAL;
        farm.tiles[tile_index].animal = action.item;
      }
    }
  }
  return failures;
}

int macro_market_failures(const Simulator& env, int player,
                          const PlayerAction& actions) {
  const auto& fills = env.last_market_fills()[player];
  int failures = 0;
  for (size_t order = 0; order < actions.market.size(); ++order) {
    const auto& action = actions.market[order];
    if (action.op != Op::BUY_SEED && action.op != Op::BUY_ANIMAL &&
        action.op != Op::HIRE && action.op != Op::BUY_LAND) continue;
    const int requested = action.op == Op::HIRE || action.op == Op::BUY_LAND
        ? int(action.quantity > 0) : quantity(action);
    const int filled = order < fills.size() ? fills[order] : 0;
    failures += std::max(0, requested - filled);
  }
  return failures;
}

struct FeatureCounts {
  std::array<int, 5> crops{}, crop_yield{};
  std::array<int, 3> animals{}, animal_yield{};
  std::array<int, 3> structures{};
  int weeds = 0, empty = 0, crop_stress = 0, animal_stress = 0;
};

FeatureCounts feature_counts(const Farm& farm) {
  FeatureCounts c;
  for (const Tile& t : farm.tiles) {
    if (t.kind == TileKind::EMPTY) c.empty++;
    else if (t.kind == TileKind::WEED) c.weeds++;
    else if (t.kind == TileKind::COOP) c.structures[1]++;
    else if (t.kind == TileKind::PASTURE) c.structures[2]++;
    else if (t.kind == TileKind::PLANT) {
      const int item = int(t.crop); c.crops[item]++; c.crop_yield[item] += t.yield_units;
      c.crop_stress += !t.watered_today; c.crop_stress += t.consecutive_unwatered;
    } else if (t.kind == TileKind::ANIMAL) {
      const int item = int(t.animal) - 9; c.animals[item]++; c.animal_yield[item] += t.yield_units;
      c.structures[t.animal == Item::GOOSE ? 1 : 2]++;
      c.animal_stress += !t.fed_today; c.animal_stress += t.consecutive_unfed;
    }
  }
  return c;
}

struct FeatureHistory {
  int steps = 0, tile_samples = 0;
  std::array<double, 2> min_money{
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  std::array<int, 2> low100{}, low300{}, max_weeds{}, weed_area{}, max_crops{}, max_animals{};
  void update(const Simulator& env) {
    const int step = env.step_count();
    const bool sample = steps == 0 || step % 24 == 0;
    for (int p = 0; p < 2; ++p) {
      const auto& farm = env.farms()[p];
      min_money[p] = std::min(min_money[p], farm.money);
      low100[p] += farm.money < 100; low300[p] += farm.money < 300;
      if (sample) {
        const auto c = feature_counts(farm);
        const int crops = std::accumulate(c.crops.begin(), c.crops.end(), 0);
        const int animals = std::accumulate(c.animals.begin(), c.animals.end(), 0);
        max_weeds[p] = std::max(max_weeds[p], c.weeds); weed_area[p] += c.weeds;
        max_crops[p] = std::max(max_crops[p], crops);
        max_animals[p] = std::max(max_animals[p], animals);
      }
    }
    tile_samples += sample; steps++;
  }
};

int fib_cost(int index) {
  int left = 1, right = 1;
  while (index-- > 0) { const int next = left + right; left = right; right = next; }
  return left;
}

std::array<float, 147> build_features(const Simulator& env, int player,
                                      const FeatureHistory& history,
                                      const std::vector<PlayerAction>& tape) {
  std::array<float, 147> result{}; size_t at_index = 0;
  auto push = [&](double value) { result.at(at_index++) = float(value); };
  auto farm_vector = [&](int p) {
    const auto& farm = env.farms()[p]; const auto c = feature_counts(farm);
    push(farm.money); push(farm.hands.size()); push(std::popcount(unsigned(farm.unlocked_mask)));
    push(farm.hires_today); push(c.weeds); push(c.empty);
    for (int x : c.crops) push(x); for (int x : c.animals) push(x);
    for (int x : c.structures) push(x); for (int x : c.crop_yield) push(x);
    for (int x : c.animal_yield) push(x); push(c.crop_stress); push(c.animal_stress);
  };
  farm_vector(player); farm_vector(1 - player);
  const auto& pr = env.privates()[player];
  for (int x : pr.shed) push(x); for (int x : pr.seeds) push(x);
  std::array<int, N_ITEMS> carried{};
  for (const auto& inv : pr.inventories)
    for (int item = 0; item < N_ITEMS; ++item) carried[item] += inv[item];
  for (int x : carried) push(x);
  const int shed_total = shed_sum(pr);
  push(shed_total); push(std::accumulate(carried.begin(), carried.end(), 0)); push(100 - shed_total);
  for (int item = 0; item < N_PRODUCTS; ++item) {
    push(env.market().inventory[item]); push(env.market().prices[item]);
  }
  std::array<bool, 8> shops{}; for (int sh : env.shops()) shops[sh] = true;
  for (bool value : shops) push(value);
  const int step = env.step_count(); push(step); push(step / 24); push(step % 24);
  for (int p : {player, 1 - player}) {
    const auto current = feature_counts(env.farms()[p]);
    const int crops = std::accumulate(current.crops.begin(), current.crops.end(), 0);
    const int animals = std::accumulate(current.animals.begin(), current.animals.end(), 0);
    push(std::isfinite(history.min_money[p]) ? history.min_money[p] : 0);
    push(double(history.low100[p]) / std::max(1, history.steps));
    push(double(history.low300[p]) / std::max(1, history.steps));
    push(history.max_weeds[p]);
    push(double(history.weed_area[p]) / std::max(1, history.tile_samples));
    push(std::max(0, history.max_crops[p] - crops));
    push(std::max(0, history.max_animals[p] - animals));
  }
  static constexpr int seed_cost[5] = {10,20,50,100,80};
  static constexpr int animal_cost[3] = {300,400,500};
  static constexpr int land_cost[3] = {1000,2000,4000};
  for (int horizon : {24, 48, 72}) {
    double cumulative = 0, requirement = 0, expense = 0;
    int hires = env.farms()[player].hires_today;
    int land = std::popcount(unsigned(env.farms()[player].unlocked_mask));
    int hires_planned = 0, lands_planned = 0, animals_planned = 0;
    int previous_day = step / 24;
    for (int future = step; future < std::min(int(tape.size()), step + horizon); ++future) {
      const int day = future / 24; if (day != previous_day) { hires = 0; previous_day = day; }
      for (const auto& order : tape[future].market) {
        double spend = 0, revenue = 0;
        const int item = int(order.item), q = std::max(1, quantity(order));
        if (order.op == Op::HIRE) { spend = fib_cost(hires++); hires_planned++; }
        else if (order.op == Op::BUY_LAND) {
          const int extra = std::max(0, land - 1);
          if (extra < 3) { spend = land_cost[extra]; land++; lands_planned++; }
        } else if (order.op == Op::BUY_SEED && item >= 0 && item < 5) spend = q * seed_cost[item];
        else if (order.op == Op::BUY_ANIMAL && item >= 9 && item < 12) {
          spend = q * animal_cost[item - 9]; animals_planned += q;
        } else if (order.op == Op::BUY_PRODUCT && item >= 0 && item < N_PRODUCTS)
          spend = q * env.market().prices[item];
        else if (order.op == Op::SELL && item >= 0 && item < N_PRODUCTS)
          revenue = q * env.market().prices[item];
        expense += spend; cumulative += spend - revenue; requirement = std::max(requirement, cumulative);
      }
    }
    push(expense); push(requirement); push(env.farms()[player].money - requirement);
    push(hires_planned); push(lands_planned); push(animals_planned);
  }
  if (at_index != result.size()) throw std::runtime_error("native route feature dimension mismatch");
  return result;
}

}  // namespace

NativeRepairOptions native_repair_options_from_mask(int mask) {
  NativeRepairOptions result;
  result.weed_min_loss_realign = (mask & 1) != 0;
  result.animal_buy_retry = (mask & 2) != 0;
  result.empty_stall_reuse = (mask & 4) != 0;
  result.state_driven_local_repair = (mask & 8) != 0;
  result.day_horizon_repair = (mask & 16) != 0;
  result.day_horizon_repair_v2 = (mask & 32) != 0;
  result.rolling_route_skeleton_v3 = (mask & 64) != 0;
  return result;
}

void NativeAgentState::reset() {
  const auto evaluation_profile = phased_market.evaluation_profile;
  *this = NativeAgentState{};
  phased_market.evaluation_profile = evaluation_profile;
}

int native_macro_unit_failures(const Simulator& env, int player,
                               const PlayerAction& action) {
  return macro_unit_failures(env, player, action);
}

int native_macro_market_failures(const Simulator& env, int player,
                                 const PlayerAction& action) {
  return macro_market_failures(env, player, action);
}

PlayerAction NativeTeammateExecutor::action(const Simulator& env, int player,
                                             int route,
                                             NativeAgentState& state,
                                             bool neutral_special_economy,
                                             const NativeRepairOptions& repair_options,
                                             NativeRepairAudit* repair_audit,
                                             bool experimental_general_takeover,
                                             NativeGeneralMarketAudit* economy_audit,
                                             int future_switch_step,
                                             NativeMarketArm experimental_market_arm,
                                             NativePhasedMarketAudit* phased_market_audit,
                                             NativePhasedStepAudit* phased_step_audit,
                                             const PlayerAction* final_action_override,
                                             std::uint64_t final_action_owned_actor_mask,
                                             bool final_action_owns_market_tail,
                                             std::uint64_t final_action_authority_hash) const {
  const int step = env.step_count();
  if (step == 0 || step < state.last_step) state.reset();
  const int exclusive_repair_owners =
      (repair_options.route_cursor_production != 0) +
      repair_options.state_driven_local_repair +
      repair_options.day_horizon_repair +
      repair_options.day_horizon_repair_v2 +
      repair_options.rolling_route_skeleton_v3 +
      repair_options.weed_obligation_day_owner;
  if (exclusive_repair_owners > 1)
    throw std::invalid_argument(
        "RouteCursor and bit-8/16/32/64 repair owners are mutually exclusive");
  if (repair_options.state_driven_local_repair)
    settle_event_local_purchase_receipts(env, player, state, repair_audit);
  if (repair_options.day_horizon_repair_v2)
    settle_day_horizon_v2_purchases(env, player, state, repair_audit);
  if (repair_options.state_driven_local_repair)
    settle_event_local_receipts(env, player, state, repair_audit);
  if (repair_options.route_cursor_production != 0) {
    if (!state.experimental_deferred_crop_scheduler)
      state.experimental_deferred_crop_scheduler.emplace(
          g001::failure_debt::deferred_crop::Config{
              true, env.config().turns_per_day});
    settle_deferred_crop_receipts(env, player, state);
    settle_route_cursor_commit(env, player, state, repair_audit);
    settle_route_cursor_purchases(env, player, state, repair_audit);
    // This is an observation fact, not an emitted-action transition. Preserve
    // crop identity for every tile even when another actor's receipt failed.
    observe_route_crop_memory(env, player, state);
    enqueue_deferred_crop_sources(env, state);
  }
  state.last_step = step;
  if (route < 0 || route >= int(library_.routes.size()) ||
      library_.routes[route].empty()) return {};
  const auto& tape = library_.routes[route];
  PlayerAction out = aligned(tape[std::min(step, int(tape.size()) - 1)], env, player);
  std::optional<joint_fixed_move_oracle::production::RouteCursorProposal>
      route_cursor_proposal;
  std::vector<g001::failure_debt::deferred_crop::Proposal>
      deferred_crop_proposals;
  std::vector<NativeAgentState::PendingPurchaseReceipt>
      route_cursor_purchase_proposals;
  const auto pos = positions(env, player);

  // The fork owner asks native only for typed stationary work. At hour zero
  // this scans the real selected tape from the real observed actor positions;
  // it does not create a fixture state and it never exports a MOVE as a
  // stationary obligation. The external day scheduler is the sole MOVE owner.
  state.experimental_stationary_obligations.clear();
  if (repair_options.weed_obligation_day_owner && env.hour() == 0) {
    auto projected = pos;
    const int day_end = std::min(
        env.step_count() + env.config().turns_per_day - 1,
        static_cast<int>(tape.size()) - 1);
    for (int source = env.step_count(); source <= day_end; ++source) {
      const auto& units = tape[static_cast<std::size_t>(source)].units;
      for (std::size_t actor = 0;
           actor < projected.size() && actor < units.size(); ++actor) {
        const auto action = units[actor];
        if (action.op == Op::BUILD_PASTURE) {
          const Tile* target = tile_at(env, player, projected[actor]);
          if (target && target->kind == TileKind::WEED) {
            state.experimental_stationary_obligations.push_back(
                {static_cast<int>(actor), source, projected[actor], action});
          }
        }
        if (action.op == Op::NORTH) --projected[actor].y;
        else if (action.op == Op::SOUTH) ++projected[actor].y;
        else if (action.op == Op::WEST) --projected[actor].x;
        else if (action.op == Op::EAST) ++projected[actor].x;
      }
    }
  }

  // K320 weed repair and replay catch-up. The experiment absorbs DIG only at
  // a PASS before this actor's next MOVE, preserving every MOVE's absolute
  // step and order. The deployed legacy path remains unchanged when off.
  state.weed.resize(out.units.size());
  state.experimental_realign.resize(out.units.size());
  int hires = 0;
  for (int ahead = 0; ahead < 3; ++ahead) {
    const auto& x = tape[std::min(step + ahead, int(tape.size()) - 1)];
    hires += std::count_if(x.market.begin(), x.market.end(),
                           [](const Action& a) { return a.op == Op::HIRE; });
  }
  const bool farmer_barrier = hires >= 5;
  if (repair_options.route_cursor_production != 0 ||
      repair_options.state_driven_local_repair ||
      repair_options.day_horizon_repair ||
      repair_options.day_horizon_repair_v2 ||
      repair_options.rolling_route_skeleton_v3 ||
      repair_options.weed_obligation_day_owner) {
    // A state-driven compiler below owns weed recovery. Do not let the legacy
    // previous-tape replay pre-compose stale units (including MOVE) before
    // final arbitration. Unrelated downstream protection remains active.
    for (auto& transaction : state.weed) transaction = {};
    for (auto& transaction : state.experimental_realign) transaction = {};
  } else if (repair_options.weed_min_loss_realign) {
    for (size_t u = 0; u < state.experimental_realign.size(); ++u) {
      auto& transaction = state.experimental_realign[u];
      if (!transaction.active) continue;
      if (step <= transaction.skipped_source_step) {
        out.units[u] = tape_unit(tape, step - 1, int(u));
      } else {
        transaction.active = false;
      }
    }
    for (size_t u = 0; u < state.weed.size(); ++u) {
      auto& transaction = state.weed[u];
      if (state.experimental_realign[u].active) continue;
      if (farmer_barrier && u == 0) {
        transaction.active = false;
        continue;
      }
      if (!transaction.active) continue;
      const int age = step - transaction.start;
      if (age == 1) out.units[u] = transaction.intended;
      else if (age >= 2 && age <= 9)
        out.units[u] = tape_unit(tape, step - 1, static_cast<int>(u));
      else
        transaction.active = false;
    }
    for (size_t u = 0; u < out.units.size() && u < pos.size(); ++u) {
      auto& transaction = state.experimental_realign[u];
      auto& legacy = state.weed[u];
      if (transaction.active || legacy.active ||
          (farmer_barrier && u == 0))
        continue;
      const auto operation = out.units[u].op;
      if (operation != Op::BUILD_PASTURE && operation != Op::PLANT) continue;
      const Tile* tile = tile_at(env, player, pos[u]);
      if (!tile || tile->kind != TileKind::WEED) continue;
      const bool legacy_drops_move = g001::repair::is_movement(
          tape_unit(tape, step + 9, static_cast<int>(u)).op);
      const int safe_horizon = std::min({
          repair_options.maximum_realign_lookahead,
          env.config().turns_per_day - 1 - env.hour(),
          static_cast<int>(tape.size()) - 1 - step});
      const bool weed_realign_phase = operation == Op::BUILD_PASTURE ||
          step >= repair_options.minimum_weed_realign_step;
      int safe_pass = -1;
      for (int offset = 0;
           weed_realign_phase && offset <= safe_horizon; ++offset) {
        const auto op = tape_unit(tape, step + offset, static_cast<int>(u)).op;
        if (g001::repair::is_movement(op)) break;
        if (op == Op::PASS) {
          safe_pass = offset;
          break;
        }
      }
      const auto plan = safe_pass < 0 ? std::nullopt : minimum_loss_plan(
          tape, step, int(u), pos[u], safe_pass, safe_pass);
      const bool absorbs_pass = plan && tape_unit(
          tape, step + plan->metrics.skipped_source_index,
          static_cast<int>(u)).op == Op::PASS;
      if (!legacy_drops_move || !absorbs_pass) {
        legacy = {true, step, out.units[u]};
        out.units[u] = Action{Op::DIG};
        if (repair_audit) ++repair_audit->weed_no_safe_alignment;
        continue;
      }
      const int local_skip = plan->metrics.skipped_source_index;
      transaction = {true, step, step + local_skip, Action{Op::DIG}};
      out.units[u] = Action{Op::DIG};
      if (repair_audit) {
        ++repair_audit->weed_triggers;
        repair_audit->movement_edits += plan->metrics.movement_edits;
        const Action absorbed = tape_unit(tape, step + local_skip, int(u));
        if (absorbed.op == Op::PASS) ++repair_audit->weed_absorbed_pass;
        else ++repair_audit->weed_absorbed_productive;
        if (g001::repair::is_movement(tape_unit(tape, step + 9, int(u)).op))
          ++repair_audit->weed_legacy_would_drop_move;
      }
    }
  } else {
    if (repair_options.animal_buy_retry) {
      for (size_t u = 0; u < state.experimental_realign.size(); ++u) {
        auto& transaction = state.experimental_realign[u];
        if (!transaction.active) continue;
        if (step <= transaction.skipped_source_step)
          out.units[u] = tape_unit(tape, step - 1, int(u));
        else transaction.active = false;
      }
    }
    for (size_t u = 0; u < state.weed.size(); ++u) {
      auto& w = state.weed[u];
      if (u < state.experimental_realign.size() &&
          state.experimental_realign[u].active) continue;
      if (farmer_barrier && u == 0) { w.active = false; continue; }
      if (!w.active) continue;
      const int age = step - w.start;
      if (age == 1) out.units[u] = w.intended;
      else if (age >= 2 && age <= 9) {
        const auto& previous = tape[std::max(0, step - 1)];
        out.units[u] = u < previous.units.size() ? previous.units[u] : Action{};
      } else w.active = false;
    }
    for (size_t u = 0; u < out.units.size() && u < pos.size(); ++u) {
      auto& w = state.weed[u];
      if (w.active || (u < state.experimental_realign.size() &&
                       state.experimental_realign[u].active) ||
          (farmer_barrier && u == 0)) continue;
      const auto op = out.units[u].op;
      if (op != Op::BUILD_PASTURE && op != Op::PLANT) continue;
      const Tile* t = tile_at(env, player, pos[u]);
      if (t && t->kind == TileKind::WEED) {
        w = {true, step, out.units[u]};
        out.units[u] = Action{Op::DIG};
      }
    }
  }

  // Side-effect-free proposal only. RouteCursor state/audit remains unchanged
  // until the final unit vector is exact-staged below and its effects are
  // verified from the next Simulator observation.
  if (repair_options.route_cursor_production != 0) {
    joint_fixed_move_oracle::production::ReactiveOptions options;
    // WATER/HARVEST can be the first observed byte after an earlier crop has
    // decayed to WEED. The live last-crop observation is the canonical tile
    // identity; restricting admission to PLANT would emit a known-invalid
    // WATER and cascade the cursor for the rest of the day.
    options.recover_old_crop_weeds = true;
    options.manage_all_known_crops = false;
    options.recover_missing_seed_debt =
        repair_options.route_cursor_production >= 2;
    options.minimum_old_crop_market_price = 0;
    options.require_legacy_drop_move_for_new_debt = true;
    options.causal_cycle_guard = true;
    route_cursor_proposal =
        joint_fixed_move_oracle::production::propose_reactive_route_cursor(
            tape, env, player, state.experimental_route_cursor,
            state.experimental_route_cursor_audit, options);
    if (repair_audit) ++repair_audit->route_cursor_proposals;
    const auto& proposed = route_cursor_proposal->action.units;
    for (std::size_t actor = 0;
         actor < out.units.size() && actor < proposed.size(); ++actor) {
      // Absolute current MOVE retains ownership. A repair may wait and replan,
      // but cannot replace an already-selected MOVE byte.
      if (movement(out.units[actor].op) &&
          !same_action(out.units[actor], proposed[actor]))
        continue;
      out.units[actor] = proposed[actor];
    }
  }

  // Late capacity evacuation.
  if (step >= 648) {
    const int day = step / 24, hour = step % 24;
    if (state.room_evac.day != day) state.room_evac = {{}, -1, {}, day};
    if (hour >= 21) {
      const auto& pr = env.privates()[player];
      int total = shed_sum(pr);
      for (const auto& inv : pr.inventories)
        total += std::accumulate(inv.begin(), inv.end(), 0);
      if (hour == 21 && !state.room_evac.active && total > 100) {
        std::tuple<int, int, int> best{999, 0, 999};
        Position best_target{};
        for (size_t u = 0; u < pos.size() && u < pr.inventories.size(); ++u) {
          int saleable = std::accumulate(pr.inventories[u].begin(),
                                         pr.inventories[u].begin() + N_PRODUCTS, 0);
          if (saleable <= 0 || u >= out.units.size() || out.units[u].op != Op::PASS) continue;
          Position target = SHED_ACCESS[0];
          for (auto p : SHED_ACCESS) if (distance(pos[u], p) < distance(pos[u], target)) target = p;
          const int d = distance(pos[u], target);
          auto candidate = std::tuple{d, -saleable, int(u)};
          if (d <= 2 && candidate < best) { best = candidate; best_target = target; }
        }
        if (std::get<2>(best) != 999) {
          state.room_evac.active = true;
          state.room_evac.actor = std::get<2>(best);
          state.room_evac.target = best_target;
        }
      }
      if (state.room_evac.active) {
        const int u = state.room_evac.actor;
        if (u < 0 || u >= int(pos.size()) || u >= int(pr.inventories.size()))
          state.room_evac.active = false;
        else if (!at(pos[u], state.room_evac.target)) out.units[u] = move_toward(pos[u], state.room_evac.target);
        else if (hour == 23) {
          out.units[u] = Action{Op::DROP};
          int needed = std::max(0, total - 100);
          for (int item : ROOM_PRIORITY) {
            const int available = std::max(0, pr.inventories[u][item] - planned_sell(out, item));
            const int q = std::min(needed, available);
            merge_sale(out, item, q); needed -= q;
            if (needed <= 0) break;
          }
        }
      }
    }
  }

  // Route-special economic overlays are optional so a route-agnostic planner
  // can be measured against the same production/safety programme without
  // inheriting fixed-tape timing or opponent-family counters.
  if (!neutral_special_economy) for (auto it = out.market.begin(); it != out.market.end();) {
    const int slot = premium_slot(int(it->item));
    if (sell(*it) && slot >= 0 && state.k320_due[slot] > 0) {
      const int reduce = std::min(quantity(*it), state.k320_due[slot]);
      it->quantity = quantity(*it) - reduce; state.k320_due[slot] -= reduce;
      if (it->quantity <= 0) { it = out.market.erase(it); continue; }
    }
    ++it;
  }
  if (!neutral_special_economy) rank_sell_slots(env, out);

  // K320 four-step preemption.
  const int clone = clone_distance(env);
  const bool exact = clone == 0;
  if (!neutral_special_economy && step >= (exact ? 120 : 216) && step < 680 && clone <= (exact ? 6 : 100) &&
      std::accumulate(state.k320_due.begin(), state.k320_due.end(), 0) == 0 &&
      out.market.size() < 10) {
    auto remaining = projected_shed(env, player, out);
    for (const auto& a : out.market) if (sell(a)) remaining[int(a.item)] =
        std::max(0, remaining[int(a.item)] - quantity(a));
    const std::array<int, 4> choices{3, 4, 6, 7};
    for (int item : choices) {
      if (exact && item == 4) continue;
      int future = 0;
      for (int h = 1; h <= 4 && step + h < int(tape.size()); ++h)
        future += planned_sell(tape[step + h], item);
      if (future < 4 || out.market.size() >= 10) continue;
      const int q = std::min({remaining[item], future, exact ? 32 : 12});
      if (q > 0) {
        out.market.push_back(Action{Op::SELL, Item(item), q});
        state.k320_due[premium_slot(item)] = q;
      }
    }
  }

  // R5 and MD opponent-family counters.
  const Farm& opponent = env.farms()[1 - player];
  const int cows = animal_count(opponent, Item::COW);
  const int sheep = animal_count(opponent, Item::SHEEP);
  if (!neutral_special_economy && !state.r5_target && step >= 24 && sheep >= 4 && cows <= 3) state.r5_target = true;
  if (!neutral_special_economy && !state.md_target && step >= 160 &&
      ((std::popcount(unsigned(opponent.unlocked_mask)) >= 2 && cows >= 4 && sheep <= 2) || cows >= 9))
    state.md_target = true;
  auto reference_counter = [&](const std::vector<PlayerAction>& ref, int future,
                               double fraction, bool enabled) {
    if (!enabled || future < 0 || future >= int(ref.size())) return;
    const auto& pr = env.privates()[player];
    for (int item : PREMIUM) {
      const int target = planned_sell(ref[future], item);
      if (target <= 0) continue;
      if (fraction == .5 &&
          (shop_demand(env, item, step) > 0 || shop_demand(env, item, step + 1) > 0)) continue;
      const int available = std::max(0, pr.shed[item] - planned_sell(out, item) - pickup_reserve(out, item));
      const int desired = std::max(1, int(std::nearbyint(target * fraction)));
      merge_sale(out, item, std::min(available, desired));
    }
    if (out.market.size() > 10) out.market.resize(10);
  };
  if (!neutral_special_economy) {
    reference_counter(library_.r5_reference, step + 3, .5, state.r5_target);
    reference_counter(library_.md_reference, step + 1, 2.0, state.md_target);
  }

  // End-of-day capacity guard.
  if (step % 24 == 23) {
    const auto& pr = env.privates()[player];
    int carried = 0; for (const auto& inv : pr.inventories) carried += std::accumulate(inv.begin(), inv.end(), 0);
    int produced = 0, consumed = 0;
    for (size_t u = 0; u < out.units.size() && u < pos.size(); ++u) {
      const Tile* t = tile_at(env, player, pos[u]); const auto& x = out.units[u];
      if (x.op == Op::HARVEST && t) produced += std::max(0, int(t->yield_units));
      else if (x.op == Op::COLLECT_FERTILIZER && t && t->fertilizer_available) produced++;
      else if (x.op == Op::FEED || x.op == Op::FERTILIZE) consumed++;
      else if (x.op == Op::PLACE && int(x.item) >= 9) consumed++;
    }
    int actual_sells = 0, buys = 0;
    for (int item = 0; item < N_PRODUCTS; ++item)
      actual_sells += std::min(pr.shed[item], planned_sell(out, item));
    for (const auto& x : out.market)
      if (x.op == Op::BUY_PRODUCT || x.op == Op::BUY_ANIMAL) buys += quantity(x);
    int needed = std::max(0, shed_sum(pr) + carried + produced - consumed + buys - actual_sells - 100);
    for (int item : ROOM_PRIORITY) {
      const int q = std::min(needed, std::max(0, pr.shed[item] - planned_sell(out, item)));
      merge_sale(out, item, q); needed -= q;
      if (needed <= 0) break;
    }
    if (out.market.size() > 10) out.market.resize(10);
  }

  // Terminal liquidation.
  if (step >= 716) {
    const auto& shed = env.privates()[player].shed;
    for (int item : LIQUIDATION) {
      const int extra = step >= 718 ? shed[item] : std::max(0, shed[item] - planned_sell(out, item));
      if (extra > 0 && out.market.size() < 10)
        out.market.push_back(Action{Op::SELL, Item(item), extra});
    }
  }
  out = aligned(std::move(out), env, player);

  // Do not buy one-shot seeds that can no longer be planted and harvested.
  auto seeds = env.privates()[player].seeds;
  for (auto it = out.market.begin(); it != out.market.end();) {
    const int item = int(it->item);
    if (it->op == Op::BUY_SEED && (item == 0 || item == 1)) {
      const int last_day = 29 - 2;
      int need = std::count_if(out.units.begin(), out.units.end(), [item](const Action& a) {
        return a.op == Op::PLANT && int(a.item) == item;
      });
      for (int ahead = step + 1; ahead < int(tape.size()) && ahead / 24 <= last_day; ++ahead)
        need += std::count_if(tape[ahead].units.begin(), tape[ahead].units.end(), [item](const Action& a) {
          return a.op == Op::PLANT && int(a.item) == item;
        });
      const int q = std::min(quantity(*it), std::max(0, need - seeds[item]));
      if (q <= 0) { it = out.market.erase(it); continue; }
      it->quantity = q; seeds[item] += q;
    }
    ++it;
  }

  // FC15 residual premium sale.
  if (!neutral_special_economy && step >= 120 && clone <= 6) {
    auto projected = projected_shed(env, player, out);
    static constexpr int bases[9] = {25,35,60,120,250,50,160,200,100};
    for (int item : K320_PREMIUM) {
      const int available = std::max(0, projected[item] - planned_sell(out, item) - pickup_reserve(out, item));
      if (available > 0 && env.market().prices[item] >= bases[item]) merge_sale(out, item, 1);
    }
    if (out.market.size() > 10) out.market.resize(10);
  }

  // Moon opponent-market observer.  This is route-special because its future
  // reference is a frozen Moon tape, so the neutral baseline skips it wholly.
  if (!neutral_special_economy) {
  auto& race = state.moon_race;
  if (step == 0 || step < race.last_step) {
    race = NativeAgentState::MoonRace{};
    race.horizon = {1,1,1,1};
  }
  const int moon_label = moon_route(env);
  if (state.moon_layout < 0 && step >= 24 && step < 72) {
    const Farm& opp = env.farms()[1 - player];
    const auto sig = signature(opp);
    state.moon_layout = sig.counts[0] == 5 && sig.counts[4] == 5 &&
                        animal_count(opp, Item::COW) == 1 &&
                        animal_count(opp, Item::SHEEP) == 4 && sig.counts[8] == 0 &&
                        opp.money <= 12 ? 1 : 0;
  }
  const auto& moon_tape = (state.moon_layout == 1 ? library_.moon_legacy : library_.moon)[moon_label];
  auto moon_planned = [&](int at_step, int item) { return tape_sell(moon_tape, at_step, item); };
  for (double& x : race.evidence) x *= .999;
  for (auto& row : race.scores) for (double& x : row) x *= .999;
  race.policy_evidence *= .999; for (double& x : race.policy_scores) x *= .999;
  if (race.last_step == step - 1) {
    for (int item : PREMIUM) {
      const int slot = premium_slot(item);
      if (race.prices[item] <= 1 || env.market().prices[item] <= 1) continue;
      int drain = 0;
      if ((step - 1) % 4 == 0) {
        // Reconstruct prior-shop consumption from the stored shop list.
        Simulator const& same = env;
        (void)same;
        static constexpr uint16_t masks[8] = {
          33,41,15,73,2,69,72,128};
        for (int sh : race.shops) if (masks[sh] & (1u << item))
          drain += std::popcount(unsigned(masks[sh])) == 1 ? 2 : 1;
      }
      if ((step - 1) % 24 == 0) drain++;
      const int opponent_supply = env.market().inventory[item] - race.inventory[item] + drain - race.own_sells[slot];
      const int extra = opponent_supply - moon_planned(step - 1, item);
      if (extra < 4) continue;
      race.evidence[slot] += 1; race.policy_evidence += 1;
      for (int h = 1; h <= 6; ++h) {
        const int expected = moon_planned(step - 1 + h, item);
        const double delta = expected > 0 ? 1.0 + double(std::min(extra, expected)) / std::max(extra, expected) : -.15;
        race.scores[slot][h-1] += delta; race.policy_scores[h-1] += delta;
      }
      if (race.evidence[slot] >= 1.5) {
        std::array<int,6> order{0,1,2,3,4,5};
        std::sort(order.begin(), order.end(), [&](int a, int b) {
          if (race.scores[slot][a] != race.scores[slot][b])
            return race.scores[slot][a] > race.scores[slot][b];
          return a < b;
        });
        if (race.scores[slot][order[0]] >= race.scores[slot][order[1]] + .25)
          race.horizon[slot] = std::min(6, order[0] + 2);
      }
    }
  }
  if (race.policy_evidence >= 1.5) {
    std::array<int,6> order{0,1,2,3,4,5};
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      if (race.policy_scores[a] != race.policy_scores[b])
        return race.policy_scores[a] > race.policy_scores[b];
      return a < b;
    });
    if (race.policy_scores[order[0]] >= race.policy_scores[order[1]] + .25) {
      race.policy_horizon = std::min(6, order[0] + 2);
      for (int& h : race.horizon) if (h == 1) h = race.policy_horizon;
    }
  }
  race.last_step = step; race.inventory = env.market().inventory; race.prices = env.market().prices;
  race.shops = env.shops();

  // Moon repayment.
  for (auto it = out.market.begin(); it != out.market.end();) {
    const int slot = premium_slot(int(it->item));
    if (sell(*it) && slot >= 0 && state.moon_debts[step][slot] > 0) {
      const int r = std::min(quantity(*it), state.moon_debts[step][slot]);
      it->quantity = quantity(*it) - r; state.moon_debts[step][slot] -= r;
      if (it->quantity <= 0) { it = out.market.erase(it); continue; }
    }
    ++it;
  }
  // Embedded Moon uses [_PREEMPT_START, _PREEMPT_STOP) = [120, 680).
  // Its debt repayment remains active after the stop, but no new shift may be
  // opened in the terminal window.
  if (step >= 120 && step < 680 && clone <= 6 && out.market.size() < 10) {
    if (step >= 120 && clone <= 2) for (int& h : race.horizon) h = std::max(h, 4);
    auto remaining = projected_shed(env, player, out);
    for (const auto& x : out.market) if (sell(x)) remaining[int(x.item)] =
        std::max(0, remaining[int(x.item)] - quantity(x));
    struct Choice { double value; int item, q, horizon; };
    std::vector<Choice> choices;
    for (int item : PREMIUM) {
      const int slot = premium_slot(item);
      for (int h = race.horizon[slot]; h >= 1; --h) {
        const int future = moon_planned(step + h, item);
        if (future < 4) continue;
        const int q = std::min({remaining[item], future, 12});
        if (q > 0) choices.push_back({double(env.market().prices[item]) * q, item, q, h});
        break;
      }
    }
    std::vector<Choice> selected;
    for (auto x : choices) if (x.horizon > 1) selected.push_back(x);
    if (!selected.empty()) {
      const Choice best = *std::max_element(selected.begin(), selected.end(),
          [](auto a, auto b) { return std::tie(a.value,a.item,a.q,a.horizon) < std::tie(b.value,b.item,b.q,b.horizon); });
      selected.clear(); selected.push_back(best);
    }
    else selected = choices;
    for (auto x : selected) if (out.market.size() < 10) {
      out.market.push_back(Action{Op::SELL, Item(x.item), x.q});
      if (step + x.horizon < 720) state.moon_debts[step + x.horizon][premium_slot(x.item)] += x.q;
    }
  }
  // Record projected actual own premium sells for the next observation.
  race.own_sells.fill(0);
  {
    auto remaining = projected_shed(env, player, out);
    for (const auto& x : out.market) {
      const int slot = premium_slot(int(x.item));
      if (!sell(x) || slot < 0) continue;
      const int q = std::min(quantity(x), remaining[int(x.item)]);
      race.own_sells[slot] += q; remaining[int(x.item)] -= q;
    }
  }
  }

  // Opening and cash guards.
  for (auto& x : out.market) {
    if (step == 0 && x.op == Op::BUY_SEED && x.item == Item::WHEAT) x.quantity = 8;
    if (step > 72 && step < 192 && x.op == Op::BUY_SEED && x.item == Item::CARROT && quantity(x) >= 5)
      x.quantity = std::max(0, quantity(x) - 1);
  }
  if (out.market.size() > 10) out.market.resize(10);

  // Feed-value guard and wheat purchase credit.
  const auto& pr = env.privates()[player];
  int skipped = 0;
  if (step / 24 >= 10) for (size_t u = 0; u < out.units.size() && u < pos.size(); ++u) {
    if (out.units[u].op != Op::FEED) continue;
    const Tile* t = tile_at(env, player, pos[u]);
    if (!t || t->kind != TileKind::ANIMAL || t->fed_today || t->consecutive_unfed != 0) continue;
    const int product = t->animal == Item::GOOSE ? 5 : t->animal == Item::COW ? 6 : 7;
    if (env.market().prices[product] * (1 + t->pending_care_bonus) < env.market().prices[0]) {
      out.units[u] = Action{}; skipped++;
    }
  }
  state.wheat_credit += skipped;
  for (auto it = out.market.begin(); it != out.market.end();) {
    if (state.wheat_credit > 0 && it->op == Op::BUY_PRODUCT && it->item == Item::WHEAT) {
      const int take = std::min(quantity(*it), state.wheat_credit);
      it->quantity = quantity(*it) - take; state.wheat_credit -= take;
      if (it->quantity <= 0) { it = out.market.erase(it); continue; }
    }
    ++it;
  }

  // Terminal crop salvage overlay.
  if (!state.salvage.active && step >= 696) {
    long best = std::numeric_limits<long>::min();
    for (size_t u = 0; u < pos.size() && u < out.units.size() && u < pr.inventories.size(); ++u) {
      const Op op = out.units[u].op;
      if (op != Op::NORTH && op != Op::SOUTH && op != Op::EAST && op != Op::WEST) continue;
      const Tile* t = tile_at(env, player, pos[u]);
      if (!t || t->kind != TileKind::PLANT || t->yield_units <= 0) continue;
      static constexpr int first[5] = {2,2,8,10,10};
      if (step / 24 - t->planted_day < first[int(t->crop)]) continue;
      Position target = SHED_ACCESS[0];
      for (auto p : SHED_ACCESS) if (distance(pos[u], p) < distance(pos[u], target)) target = p;
      if (distance(pos[u], target) + 2 != 719 - step) continue;
      int carried_value = 0;
      for (int item = 0; item < N_PRODUCTS; ++item) carried_value += pr.inventories[u][item] * env.market().prices[item];
      const int crop_value = t->yield_units * env.market().prices[int(t->crop)];
      if (crop_value < 2 * carried_value) continue;
      const long score = long(crop_value) * 100 - int(u);
      if (score > best) {
        best = score; state.salvage = {true, int(u), target, t->crop, int(t->yield_units)};
      }
    }
    if (state.salvage.active) out.units[state.salvage.actor] = Action{Op::HARVEST};
  } else if (state.salvage.active && state.salvage.actor < int(pos.size())) {
    const int u = state.salvage.actor;
    out.units[u] = at(pos[u], state.salvage.target) ? Action{Op::DROP} : move_toward(pos[u], state.salvage.target);
  }
  if (state.salvage.active && state.salvage.actor < int(pos.size()) && step == 718 &&
      at(pos[state.salvage.actor], state.salvage.target)) {
    const auto projected = projected_shed(env, player, out);
    const int item = int(state.salvage.product);
    merge_sale(out, item, std::min(state.salvage.quantity,
               std::max(0, projected[item] - planned_sell(out, item))));
  }

  // Apply exact-source suppression for stationary work pulled forward earlier.
  // This is deliberately after the normal overlays: the future source must be
  // consumed once, not executed again because another overlay recreated it.
  if (repair_options.empty_stall_reuse) {
    for (auto it = state.experimental_suppressed.begin();
         it != state.experimental_suppressed.end();) {
      if (it->source_step < step) {
        it = state.experimental_suppressed.erase(it);
        continue;
      }
      if (it->source_step == step && it->actor >= 0 &&
          it->actor < int(out.units.size())) {
        out.units[it->actor] = Action{};
        it = state.experimental_suppressed.erase(it);
        continue;
      }
      ++it;
    }
  }

  // Causally confirm full/partial/zero animal fills from the next observation,
  // and retry only when the frozen movement programme still contains a usable
  // PICKUP -> PLACE chain.  No opponent identity or route is consulted.
  if (repair_options.animal_buy_retry) {
    g001::repair::AnimalObservation observation;
    observation.step = step;
    observation.own_total = total_animals(env, player);
    const auto currently_placed = placed_animals(env.farms()[player]);
    if (state.experimental_animals_initialized) {
      for (int animal = 0; animal < 3; ++animal)
        observation.known_losses[animal] = std::max(
            0, state.experimental_previous_placed[animal] - currently_placed[animal]);
    }
    state.experimental_previous_placed = currently_placed;
    state.experimental_animals_initialized = true;
    for (const auto& tile : env.farms()[player].tiles) {
      if (tile.kind == TileKind::COOP) ++observation.empty_structures[0];
      if (tile.kind == TileKind::PASTURE) {
        ++observation.empty_structures[1];
        ++observation.empty_structures[2];
      }
    }
    observation.shed_used = shed_sum(env.privates()[player]);
    observation.shed_capacity = env.config().shed_capacity;
    observation.money = std::max(0, int(env.farms()[player].money));
    observation.market_slots_used = int(out.market.size());
    observation.protected_cash_reserve = planned_market_cash(env, player, out);
    std::array<AnimalRouteWindow, 3> route_windows{};
    for (int animal = 0; animal < 3; ++animal) {
      route_windows[animal] = next_animal_route_window(
          tape, step, animal + 9, pos,
          std::max(96, repair_options.maximum_realign_lookahead));
      const auto& route_window = route_windows[animal];
      observation.next_pickup_step[animal] = route_window.pickup_step;
      observation.next_place_step[animal] = route_window.place_step;
      observation.pickup_place_feasible[animal] =
          route_window.pickup_step >= step &&
          route_window.place_step > route_window.pickup_step &&
          route_window.actor >= 0 && route_window.actor < int(pos.size());
      // A frozen future PLACE is a causal demand slot even when its pasture is
      // not built yet at the observation where the failed buy is detected.
      if (observation.pickup_place_feasible[animal])
        observation.empty_structures[animal] = std::max(
            1, observation.empty_structures[animal]);
      observation.allow_same_step_pickup_realign[animal] =
          repair_options.route_cursor_production == 0 &&
          route_window.pickup_realign_safe && route_window.actor >= 0 &&
          route_window.actor < int(state.experimental_realign.size()) &&
          !state.experimental_realign[route_window.actor].active;
    }
    for (const auto& order : out.market) {
      if (order.op == Op::BUY_ANIMAL && int(order.item) >= 9 &&
          int(order.item) < 12)
        observation.planned_buy[int(order.item) - 9] += quantity(order);
    }
    std::array<g001::repair::AnimalIntentSnapshot, 3> before{};
    for (int animal = 0; animal < 3; ++animal)
      before[animal] = state.experimental_animals.snapshot(
          static_cast<g001::repair::Animal>(animal));
    state.experimental_animals.observe(observation);
    if (repair_audit) for (int animal = 0; animal < 3; ++animal) {
      const auto& after = state.experimental_animals.snapshot(
          static_cast<g001::repair::Animal>(animal));
      const int fills = after.inferred_fills - before[animal].inferred_fills;
      const int failures = after.inferred_failures - before[animal].inferred_failures;
      repair_audit->animal_inferred_filled += fills;
      if (state.experimental_animal_retry_awaiting[animal])
        repair_audit->animal_retry_fills += fills;
      state.experimental_animal_retry_awaiting[animal] = false;
      if (fills > 0 && failures > 0) ++repair_audit->animal_inferred_partial;
      else if (fills == 0 && failures > 0) ++repair_audit->animal_inferred_zero;
    }

    const auto retry = state.experimental_animals.decide(observation);
    int retry_animal = -1;
    if (retry) {
      retry_animal = int(retry->animal);
      const auto& route_window = route_windows[retry_animal];
      bool admitted = true;
      std::optional<g001::repair::AlignmentPlan> pickup_plan;
      if (retry->requires_pickup_realign) {
        if (route_window.actor < 0 || route_window.actor >= int(pos.size()) ||
            state.experimental_realign[route_window.actor].active) {
          admitted = false;
        } else {
          pickup_plan = minimum_loss_plan(
              tape, step, route_window.actor, pos[route_window.actor],
              repair_options.maximum_realign_lookahead, 1);
          admitted = pickup_plan.has_value();
        }
      }
      if (admitted && out.market.size() < 10) {
        out.market.push_back(Action{Op::BUY_ANIMAL,
                                    Item(retry_animal + 9), retry->quantity});
        if (repair_audit) ++repair_audit->animal_retries_emitted;
        if (retry->requires_pickup_realign) {
          auto& transaction = state.experimental_realign[route_window.actor];
          transaction = {true, step,
                         step + pickup_plan->metrics.skipped_source_index,
                         Action{}};
          out.units[route_window.actor] = Action{};
          if (repair_audit) {
            ++repair_audit->animal_retry_pickup_realign;
            repair_audit->movement_edits += pickup_plan->metrics.movement_edits;
          }
        }
      } else if (repair_audit) {
        ++repair_audit->animal_retry_blocked_no_path;
      }
    }

    // Record aggregate emitted intent only after retry insertion.  A retry is
    // therefore observed against the same pre-market public state as the
    // original order and cannot be duplicated on the following turn.
    for (int animal = 0; animal < 3; ++animal) {
      int requested = 0;
      for (const auto& order : out.market)
        if (order.op == Op::BUY_ANIMAL && int(order.item) == animal + 9)
          requested += quantity(order);
      if (requested <= 0) continue;
      const int deadline = route_windows[animal].place_step;
      state.experimental_animals.record_attempt(
          static_cast<g001::repair::Animal>(animal), requested, deadline);
      state.experimental_animal_retry_awaiting[animal] =
          animal == retry_animal;
      if (repair_audit && animal != retry_animal)
        repair_audit->animal_original_attempted += requested;
    }
  }

  // Reuse only work that is already legal at this exact tile and position.
  // Empty animal work with no such candidate becomes an explicit PASS and is
  // counted as unresolved; no movement is manufactured or redirected.
  if (repair_options.empty_stall_reuse) {
    const auto& private_state = env.privates()[player];
    for (size_t actor = 0; actor < out.units.size() && actor < pos.size(); ++actor) {
      const Tile* tile = tile_at(env, player, pos[actor]);
      if (!tile || !g001::repair::suppress_empty_stall_work(
                       out.units[actor].op, tile->kind, false)) continue;
      auto future = realignment_window(
          tape, step + 1, int(actor), pos[actor],
          repair_options.stationary_reuse_lookahead);
      for (auto& candidate : future) {
        candidate.earliest_step = step;
        candidate.latest_step = step;
        candidate.expected_fill = false;
        const int item = int(candidate.action.item);
        if (candidate.action.op == Op::PLACE && item >= 9 && item < 12 &&
            actor < private_state.inventories.size() &&
            private_state.inventories[actor][item] > 0 &&
            ((item == 9 && tile->kind == TileKind::COOP) ||
             (item >= 10 && tile->kind == TileKind::PASTURE)))
          candidate.expected_fill = true;
        else if (candidate.action.op == Op::DROP && shed_adjacent(pos[actor]) &&
                 actor < private_state.inventories.size() &&
                 std::accumulate(private_state.inventories[actor].begin(),
                                 private_state.inventories[actor].end(), 0) > 0)
          candidate.expected_fill = true;
        else if (candidate.action.op == Op::PICKUP && shed_adjacent(pos[actor]) &&
                 item >= 0 && item < N_ITEMS && private_state.shed[item] > 0)
          candidate.expected_fill = true;
      }
      const auto recovery = g001::repair::search_stationary_work_replacement(
          pos[actor], step, future, repair_options.stationary_reuse_lookahead);
      if (recovery.recovered) {
        out.units[actor] = recovery.replacement;
        state.experimental_suppressed.push_back(
            {int(actor), step + 1 + recovery.source_index});
        if (repair_audit) ++repair_audit->empty_stall_reused;
      } else {
        out.units[actor] = Action{};
        if (repair_audit) ++repair_audit->empty_stall_unresolved;
      }
    }
  }

  // A later route overlay must not break an active delayed-source transaction.
  for (size_t actor = 0; actor < state.experimental_realign.size() &&
                         actor < out.units.size(); ++actor) {
    const auto& transaction = state.experimental_realign[actor];
    if (transaction.active && step > transaction.start_step &&
        step <= transaction.skipped_source_step)
      out.units[actor] = tape_unit(tape, step - 1, int(actor));
  }

  // Settle and emit the market half of online failed-seed repair after all
  // legacy mandatory market writes.  Unit repair was compiled before the
  // existing dynamic unit overlays, so this block must never replace units.
  if (repair_options.route_cursor_production != 0) {
    // Settle last step's exact simulator receipts before issuing another
    // retry.  No price inference or terminal outcome enters this ledger.
    // A semantic crop debt on an observed EMPTY tile is also the unified
    // failed-seed-purchase retry request.  Append only what current funds can
    // cover after preserving already composed mandatory market commitments.
    // Exact official simulator CropDef::seed values.
    static constexpr std::array<int, N_CROPS> seed_cost{10, 20, 50, 100, 80};
    std::array<int, N_CROPS> needed{};
    const auto& debts = state.experimental_route_cursor.production.debts;
    const auto& tiles = env.farms()[player].tiles;
    for (std::size_t tile = 0; tile < debts.size() && tile < tiles.size(); ++tile) {
      const auto& debt = debts[tile];
      const int crop = static_cast<int>(debt.crop);
      if (debt.active && !debt.structure && debt.seed_retry_attempts < 3 &&
          (debt.last_seed_retry_step < 0 ||
           step - debt.last_seed_retry_step >= env.config().turns_per_day) &&
          tiles[tile].kind == TileKind::EMPTY && crop >= 0 && crop < N_CROPS)
        ++needed[static_cast<std::size_t>(crop)];
    }
    std::array<int, N_CROPS> already_planned{};
    for (const auto& order : out.market)
      if (order.op == Op::BUY_SEED && static_cast<int>(order.item) >= 0 &&
          static_cast<int>(order.item) < N_CROPS)
        already_planned[static_cast<std::size_t>(order.item)] += quantity(order);
    int free_cash = std::max(0, static_cast<int>(env.farms()[player].money) -
                                planned_market_cash(env, player, out));
    const auto& seeds = env.privates()[player].seeds;
    for (int crop = 0; crop < N_CROPS && out.market.size() < 10; ++crop) {
      const int missing = std::max(
          0, needed[static_cast<std::size_t>(crop)] -
                 seeds[static_cast<std::size_t>(crop)] -
                 already_planned[static_cast<std::size_t>(crop)]);
      const int quantity_to_buy =
          std::min(missing, free_cash / seed_cost[static_cast<std::size_t>(crop)]);
      if (quantity_to_buy <= 0) continue;
      const int slot = static_cast<int>(out.market.size());
      out.market.push_back(
          Action{Op::BUY_SEED, Item(crop), quantity_to_buy});
      NativeAgentState::PendingPurchaseReceipt proposed_receipt;
      proposed_receipt.market_slot = slot;
      proposed_receipt.operation = Op::BUY_SEED;
      proposed_receipt.item = Item(crop);
      proposed_receipt.requested = quantity_to_buy;
      proposed_receipt.submitted_step = step;
      int attempts_to_mark=quantity_to_buy;
      for(std::size_t tile=0;tile<debts.size()&&tile<tiles.size()&&attempts_to_mark>0;++tile) {
        const auto& debt=state.experimental_route_cursor.production.debts[tile];
        if(debt.active&&!debt.structure&&static_cast<int>(debt.crop)==crop&&
           tiles[tile].kind==TileKind::EMPTY&&debt.seed_retry_attempts<3) {
          proposed_receipt.debt_tiles.push_back(static_cast<int>(tile));
          --attempts_to_mark;
        }
      }
      route_cursor_purchase_proposals.push_back(std::move(proposed_receipt));
      free_cash -= quantity_to_buy * seed_cost[static_cast<std::size_t>(crop)];
    }
  }

  // This is the single market overlay seam.  It is deliberately after
  // every tape/K320/R5/MD/Moon/capacity/terminal/FC15/repair write to market.
  // Default-off does not enter this branch.  arm1/2 protect this exact current
  // full action and use the raw core tape only as a caller-selected rolling
  // causal baseline.  Every next tick is rebuilt from the then-current exact
  // action; the raw preview is not claimed to predict future overlays.
  if (out.market.size() > 10) out.market.resize(10);
  if (experimental_market_arm != NativeMarketArm::LegacyDefault) {
    std::vector<native_selective_input::SelectedPlanFrame> future_plan;
    const int horizon_end = std::min({step + 24, 718, int(tape.size()) - 1});
    future_plan.reserve(std::max(0, horizon_end - step));
    const bool dynamic_transaction =
        repair_options.enabled() ||
        std::any_of(state.weed.begin(), state.weed.end(),
                    [](const auto& value) { return value.active; }) ||
        std::any_of(state.experimental_realign.begin(),
                    state.experimental_realign.end(),
                    [](const auto& value) { return value.active; }) ||
        state.room_evac.active || state.salvage.active;
    for (int future = step + 1; future <= horizon_end; ++future) {
      native_selective_input::SelectedPlanFrame frame;
      frame.step = future;
      frame.selected = tape[future];
      frame.causal_queue_known = !dynamic_transaction;
      frame.crosses_route_switch =
          future_switch_step > step && future_switch_step <= future;
      future_plan.push_back(std::move(frame));
    }
    auto phased = compose_native_phased_market(
        env, player, experimental_market_arm, out, future_plan,
        state.phased_market);
    out.market = std::move(phased.market);
    if (phased_market_audit)
      accumulate_native_phased_audit(*phased_market_audit, phased.audit);
    if (phased_step_audit) *phased_step_audit = std::move(phased.audit);
  }

  if (experimental_general_takeover) {
    if (repair_options.enabled())
      throw std::runtime_error(
          "general takeover cannot certify a future unit plan while experimental repair is enabled");
    const int horizon_end = std::min({step + 24, 718, int(tape.size()) - 1});
    if (future_switch_step > step && future_switch_step <= horizon_end)
      throw std::runtime_error(
          "general takeover future unit horizon crosses an unobserved route switch");
    const bool dynamic_transaction =
        std::any_of(state.weed.begin(), state.weed.end(),
                    [](const auto& value) { return value.active; }) ||
        std::any_of(state.experimental_realign.begin(), state.experimental_realign.end(),
                    [](const auto& value) { return value.active; }) ||
        state.room_evac.active || state.salvage.active;
    if (dynamic_transaction)
      throw std::runtime_error(
          "general takeover cannot certify raw future units during a dynamic transaction");
    std::vector<NativeFutureUnitFrame> future_units;
    future_units.reserve(std::max(0, horizon_end - step));
    for (int future = step + 1; future <= horizon_end; ++future)
      future_units.push_back({future, tape[future].units});
    auto general = compile_native_general_market(
        env, player, out.units, future_units);
    if (economy_audit) {
      economy_audit->obligation_nodes += general.audit.obligation_nodes;
      economy_audit->purchase_orders_due += general.audit.purchase_orders_due;
      economy_audit->funding_sell_orders += general.audit.funding_sell_orders;
      economy_audit->funding_sell_units += general.audit.funding_sell_units;
      economy_audit->allocated_orders += general.audit.allocated_orders;
      economy_audit->compiler_feasible = general.audit.compiler_feasible;
      economy_audit->allocator_feasible = general.audit.allocator_feasible;
      economy_audit->diagnostic_code = general.audit.diagnostic_code;
      economy_audit->diagnostic_step = general.audit.diagnostic_step;
      economy_audit->diagnostic_actor = general.audit.diagnostic_actor;
      economy_audit->diagnostic_item = general.audit.diagnostic_item;
      economy_audit->diagnostic_quantity = general.audit.diagnostic_quantity;
      economy_audit->soft_misses += general.audit.soft_misses;
      economy_audit->soft_pickup_replacements +=
          general.audit.soft_pickup_replacements;
      economy_audit->soft_pass_replacements +=
          general.audit.soft_pass_replacements;
      economy_audit->soft_downstream_loss_proxy +=
          general.audit.soft_downstream_loss_proxy;
      economy_audit->production_protection_soft_misses +=
          general.audit.production_protection_soft_misses;
      economy_audit->first_due_kind = general.audit.first_due_kind;
      economy_audit->first_due_item = general.audit.first_due_item;
      economy_audit->first_due_quantity = general.audit.first_due_quantity;
      economy_audit->first_due_cash_quote = general.audit.first_due_cash_quote;
      economy_audit->first_due_free_capacity =
          general.audit.first_due_free_capacity;
      economy_audit->reason = general.audit.reason;
    }
    if (!general.feasible)
      throw std::runtime_error(
          "general takeover fail-closed at step " + std::to_string(step) +
          ": " + general.audit.reason +
          " [nodes=" + std::to_string(general.audit.obligation_nodes) +
          ", due=" + std::to_string(general.audit.purchase_orders_due) +
          ", funding_sells=" +
          std::to_string(general.audit.funding_sell_orders) +
          ", funding_units=" +
          std::to_string(general.audit.funding_sell_units) +
          ", allocated_slots=" +
          std::to_string(general.audit.allocated_orders) +
          ", diagnostic=" + general.audit.diagnostic_code +
          ", diagnostic_step=" +
          std::to_string(general.audit.diagnostic_step) +
          ", actor=" + std::to_string(general.audit.diagnostic_actor) +
          ", item=" + std::to_string(general.audit.diagnostic_item) +
          ", quantity=" +
          std::to_string(general.audit.diagnostic_quantity) +
          ", first_due=" + general.audit.first_due_kind +
          ", due_item=" + std::to_string(general.audit.first_due_item) +
          ", due_quantity=" +
          std::to_string(general.audit.first_due_quantity) +
          ", due_cash=" +
          std::to_string(general.audit.first_due_cash_quote) +
          ", due_capacity=" +
          std::to_string(general.audit.first_due_free_capacity) + "]");
    for (const auto& replacement : general.current_unit_replacements) {
      if (replacement.actor < 0 ||
          replacement.actor >= static_cast<int>(out.units.size()))
        throw std::runtime_error(
            "general takeover soft replacement has an invalid actor");
      const auto move = [](Op op) {
        return op == Op::NORTH || op == Op::SOUTH ||
               op == Op::EAST || op == Op::WEST;
      };
      const Action& current = out.units[replacement.actor];
      if (move(current.op) || move(replacement.original.op) ||
          move(replacement.replacement.op) ||
          current.op != replacement.original.op ||
          current.item != replacement.original.item ||
          current.quantity != replacement.original.quantity)
        throw std::runtime_error(
            "general takeover refused a non-causal or MOVE soft replacement");
      out.units[replacement.actor] = replacement.replacement;
    }
    out.market = std::move(general.market);
  }

  // Isolated deferred crop executor: only certified PASS slack after all
  // existing unit overlays may be replaced. MOVE and productive base actions
  // remain read-only.
  if (repair_options.route_cursor_production != 0 &&
      state.experimental_deferred_crop_scheduler) {
    const auto movement_hash = route_movement_fingerprint(
        tape, env, state.experimental_route_cursor);
    const int turns = env.config().turns_per_day;
    const int day_end = std::min((env.day() + 1) * turns - 1,
                                 static_cast<int>(tape.size()) - 1);
    const int last_action = std::min(day_end, env.config().episode_steps - 2);
    for (std::size_t actor = 0; actor < out.units.size() && actor < pos.size();
         ++actor) {
      int cursor = step;
      if (actor < state.experimental_route_cursor.source_cursor.size())
        cursor = state.experimental_route_cursor.source_cursor[actor];
      int remaining_moves = 0;
      for (int source = std::max(step, cursor); source <= day_end; ++source)
        if (actor < tape[static_cast<std::size_t>(source)].units.size() &&
            movement(tape[static_cast<std::size_t>(source)].units[actor].op))
          ++remaining_moves;
      g001::failure_debt::deferred_crop::Visit visit;
      visit.step = step;
      visit.day = env.day();
      visit.actor = static_cast<int>(actor);
      visit.position = pos[actor];
      visit.tile = crop_snapshot(env, player, pos[actor]);
      visit.seeds = env.privates()[player].seeds;
      if (actor < env.privates()[player].inventories.size())
        visit.carried_fertilizer =
            env.privates()[player].inventories[actor][N_PRODUCTS - 1];
      visit.base_action = out.units[actor];
      const bool base_crop =
          g001::failure_debt::deferred_crop::is_crop_deferred_operation(
              out.units[actor].op);
      bool base_legal = false;
      bool base_satisfied = false;
      if (base_crop) {
        const int crop = static_cast<int>(out.units[actor].item);
        switch (out.units[actor].op) {
          case Op::DIG:
            base_legal = visit.tile.kind == TileKind::WEED;
            base_satisfied = visit.tile.kind == TileKind::EMPTY;
            break;
          case Op::PLANT:
            base_legal = visit.tile.kind == TileKind::EMPTY && crop >= 0 &&
                crop < N_CROPS &&
                visit.seeds[static_cast<std::size_t>(crop)] > 0;
            base_satisfied = visit.tile.kind == TileKind::PLANT &&
                visit.tile.crop == out.units[actor].item;
            break;
          case Op::WATER:
            base_legal = visit.tile.kind == TileKind::PLANT &&
                !visit.tile.watered_today;
            base_satisfied = visit.tile.kind == TileKind::PLANT &&
                visit.tile.watered_today;
            break;
          case Op::HARVEST:
            base_legal = visit.tile.kind == TileKind::PLANT &&
                visit.tile.harvest_legal && visit.tile.yield_units > 0;
            break;
          case Op::FERTILIZE:
            base_legal = visit.tile.kind == TileKind::PLANT &&
                visit.carried_fertilizer > 0 &&
                visit.tile.fertilized_until_day < visit.day + 2;
            base_satisfied = visit.tile.kind == TileKind::PLANT &&
                visit.tile.fertilized_until_day >= visit.day + 2;
            break;
          default: break;
        }
      }
      const bool replaceable_base = out.units[actor].op == Op::PASS ||
          (base_crop && (!base_legal || base_satisfied));
      visit.base_critical = !replaceable_base;
      visit.remaining_action_slots = std::max(0, last_action - step + 1);
      visit.remaining_moves = remaining_moves;
      visit.absorbable_slack = replaceable_base &&
          !movement(out.units[actor].op) &&
          visit.remaining_action_slots > remaining_moves;
      visit.movement_hash = movement_hash;
      auto proposal = state.experimental_deferred_crop_scheduler->propose(visit);
      if (!proposal.actionable()) continue;
      if (proposal.status ==
          g001::failure_debt::deferred_crop::ProposalStatus::EmitReplacement)
        out.units[actor] = proposal.action;
      deferred_crop_proposals.push_back(std::move(proposal));
    }
  }

  if (out.market.size() > 10) out.market.resize(10);
  out = aligned(std::move(out), env, player);

  // Final unit composer owner for the state-driven source ledger.  Every
  // legacy/dynamic overlay above is already present in `out`; the compiler
  // preserves slots it does not own, stages the exact full manifest, and
  // advances lifecycle only from the next observation settled above.
  if (repair_options.state_driven_local_repair)
    compose_event_local_repair(
        tape, env, player, state, out, repair_audit,
        !experimental_general_takeover &&
            experimental_market_arm == NativeMarketArm::LegacyDefault);
  else if (repair_options.day_horizon_repair)
    compose_day_horizon_repair(tape, env, player, state, out, repair_audit);
  else if (repair_options.day_horizon_repair_v2)
    compose_day_horizon_repair_v2(
        tape, env, player, state, out, repair_audit,
        !experimental_general_takeover &&
            experimental_market_arm == NativeMarketArm::LegacyDefault);
  else if (repair_options.rolling_route_skeleton_v3)
    compose_rolling_route_skeleton_v3(
        tape, env, player, state, out, repair_audit);

  if (final_action_override) {
    if (final_action_override->units.size() != out.units.size() ||
        final_action_override->market.size() < out.market.size() ||
        final_action_override->market.size() > 10 ||
        (final_action_owned_actor_mask == 0 &&
         !final_action_owns_market_tail) ||
        out.units.size() > 64 ||
        (out.units.size() < 64 &&
         (final_action_owned_actor_mask >> out.units.size()) != 0) ||
        (!final_action_owns_market_tail &&
         final_action_override->market.size() != out.market.size()) ||
        final_action_authority_hash == 0)
      throw std::runtime_error(
          "repair owner final action has an invalid envelope");
    for (std::size_t actor = 0; actor < out.units.size(); ++actor)
      if ((final_action_owned_actor_mask & (1ULL << actor)) == 0 &&
          !same_action(out.units[actor], final_action_override->units[actor]))
        throw std::runtime_error(
            "repair owner may not replace an unowned actor action");
    for (std::size_t slot = 0; slot < out.market.size(); ++slot)
      if (!same_action(out.market[slot], final_action_override->market[slot]))
        throw std::runtime_error(
            "repair owner may only append to native market output");
    out.units = final_action_override->units;
    out.market = final_action_override->market;
  }

  // Exact final-stage seam after every capacity/feed/salvage/stationary owner.
  // A changed manifest rejects every tentative production transition. The
  // only conservative reconciliation is an observed MOVE equal to that
  // actor's earliest remaining source MOVE; crossed non-MOVEs become debts.
  if (route_cursor_proposal) {
    bool exact = out.units.size() == route_cursor_proposal->action.units.size();
    for (std::size_t actor = 0; exact && actor < out.units.size(); ++actor)
      exact = same_action(out.units[actor],
                          route_cursor_proposal->action.units[actor]);
    auto source_base =
        joint_fixed_move_oracle::production::
            reconcile_reactive_route_cursor_final(
                tape, env, player, *route_cursor_proposal,
                std::vector<Action>(out.units.size()), false);
    auto source_final =
        joint_fixed_move_oracle::production::
            reconcile_reactive_route_cursor_final(
                tape, env, player, *route_cursor_proposal, out.units, false);
    auto staged_proposal = exact
        ? std::move(*route_cursor_proposal)
        : source_final;
    const bool reconciled_source =
        source_final.next_state.day != source_final.prior_state.day ||
        source_final.next_state.source_cursor !=
            source_final.prior_state.source_cursor ||
        source_final.next_audit.deferred_nonmoves !=
            source_final.prior_audit.deferred_nonmoves;
    if (exact || reconciled_source) {
      auto& pending = state.experimental_route_cursor_pending;
      pending = {};
      pending.active = true;
      pending.submitted_step = step;
      pending.proposal = std::move(staged_proposal);
      pending.exact_production_manifest = exact;
      pending.source_base = std::move(source_base);
      pending.source_final = std::move(source_final);
      pending.final_units = out.units;
      pending.positions_before = pos;
      {
        Simulator predicted = env;
        std::array<PlayerAction, 2> actions{};
        actions[static_cast<std::size_t>(player)] = out;
        predicted.step(actions);
        pending.positions_expected = positions(predicted, player);
        pending.tiles_expected.reserve(pos.size());
        for (const auto position : pos) {
          const Tile* tile = tile_at(predicted, player, position);
          pending.tiles_expected.push_back(tile ? *tile : Tile{});
        }
      }
      pending.tiles_before.reserve(pos.size());
      pending.inventory_before.reserve(pos.size());
      for (std::size_t actor = 0; actor < pos.size(); ++actor) {
        const Tile* tile = tile_at(env, player, pos[actor]);
        pending.tiles_before.push_back(tile ? *tile : Tile{});
        pending.inventory_before.push_back(actor_inventory(env, player, actor));
      }
      if (repair_audit) ++repair_audit->route_cursor_final_staged;
    }
    if (!exact && repair_audit) {
      ++repair_audit->route_cursor_overlay_overrides;
    }
  }
  for (const auto& proposal : deferred_crop_proposals) {
    if (proposal.actor < 0 ||
        proposal.actor >= static_cast<int>(out.units.size()))
      continue;
    const auto final_action = out.units[static_cast<std::size_t>(proposal.actor)];
    if (!state.experimental_deferred_crop_scheduler->commit(
            proposal, final_action, proposal.movement_hash))
      continue;
    NativeAgentState::PendingDeferredCropReceipt pending;
    pending.active = true;
    pending.submitted_step = step;
    pending.proposal = proposal;
    pending.before = crop_snapshot(env, player, proposal.tile);
    pending.emitted = final_action;
    pending.movement_hash = proposal.movement_hash;
    const auto actor = static_cast<std::size_t>(proposal.actor);
    pending.crop_inventory_before = crop_item(pending.before.crop)
        ? actor_item_inventory(env, player, actor, pending.before.crop) : 0;
    pending.fertilizer_inventory_before =
        actor_item_inventory(env, player, actor, Item::FERTILIZER);
    if (final_action.op == Op::WATER && step % env.config().turns_per_day ==
                                           env.config().turns_per_day - 1) {
      Simulator predicted = env;
      std::array<PlayerAction, 2> actions{};
      actions[static_cast<std::size_t>(player)] = out;
      predicted.step(actions);
      const Tile* after = tile_at(predicted, player, proposal.tile);
      pending.day_end_water_effect_lower_bound =
          pending.before.kind == TileKind::PLANT &&
          after && after->kind == TileKind::PLANT &&
          after->crop == pending.before.crop &&
          after->consecutive_unwatered == 0;
    }
    state.experimental_deferred_crop_pending.push_back(std::move(pending));
  }
  for (auto& receipt : route_cursor_purchase_proposals) {
    const bool exact = receipt.market_slot >= 0 &&
        receipt.market_slot < static_cast<int>(out.market.size()) &&
        out.market[static_cast<std::size_t>(receipt.market_slot)].op ==
            receipt.operation &&
        out.market[static_cast<std::size_t>(receipt.market_slot)].item ==
            receipt.item &&
        quantity(out.market[static_cast<std::size_t>(receipt.market_slot)]) ==
            receipt.requested;
    if (exact)
      state.experimental_purchase_receipts.push_back(std::move(receipt));
  }
  return out;
}

PlayerAction NativeTeammateExecutor::action_external(
    const Simulator& env, int player, int route, NativeAgentState& state,
    NativeMarketArm experimental_market_arm,
    NativePhasedMarketAudit* phased_market_audit,
    NativePhasedStepAudit* phased_step_audit,
    bool neutral_special_economy,
    NativeRepairOptions repair_options,
    NativeRepairAudit* repair_audit) const {
  return action(env, player, route, state, neutral_special_economy,
                repair_options, repair_audit,
                false, nullptr, -1, experimental_market_arm,
                phased_market_audit, phased_step_audit, nullptr, 0, false, 0);
}

PlayerAction NativeTeammateExecutor::action_external_weed_owner_finalized(
    const Simulator& env, int player, int route, NativeAgentState& state,
    int owned_actor, std::uint64_t owner_authority_hash,
    const PlayerAction& final_action) const {
  NativeRepairOptions options;
  options.weed_obligation_day_owner = true;
  const std::uint64_t mask = owned_actor >= 0 && owned_actor < 64
                                 ? 1ULL << owned_actor
                                 : 0;
  return action(env, player, route, state, false, options, nullptr, false,
                nullptr, -1, NativeMarketArm::LegacyDefault, nullptr,
                nullptr, &final_action, mask, false,
                owner_authority_hash);
}

PlayerAction NativeTeammateExecutor::action_external_repair_owner_finalized(
    const Simulator& env, int player, int route, NativeAgentState& state,
    NativeRepairOptions repair_options, std::uint64_t owned_actor_mask,
    bool owns_market_tail,
    std::uint64_t owner_authority_hash,
    const PlayerAction& final_action) const {
  return action(env, player, route, state, false, repair_options, nullptr, false,
                nullptr, -1, NativeMarketArm::LegacyDefault, nullptr,
                nullptr, &final_action, owned_actor_mask, owns_market_tail,
                owner_authority_hash);
}

std::uint64_t native_movement_commitment_hash(
    const NativeMovementCommitment& commitment) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto add = [&](std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & 255U;
      hash *= 1099511628211ULL;
    }
  };
  add(static_cast<std::uint32_t>(commitment.player));
  add(static_cast<std::uint32_t>(commitment.day));
  add(commitment.issuer_generation);
  add(static_cast<std::uint32_t>(commitment.issued_step));
  add(static_cast<std::uint32_t>(commitment.immutable_through_step));
  add(commitment.remaining_day_complete);
  add(commitment.revoked);
  add(commitment.moves.size());
  for (const auto& move : commitment.moves) {
    add(static_cast<std::uint32_t>(move.actor));
    add(static_cast<std::uint32_t>(move.source_step));
    add(static_cast<std::uint8_t>(move.action.op));
    add(static_cast<std::uint8_t>(move.action.item));
    add(static_cast<std::uint32_t>(move.action.quantity));
  }
  return hash;
}

NativeCommittedAction NativeTeammateExecutor::action_external_committed(
    const Simulator& env, int player, int route, NativeAgentState& state,
    std::uint64_t issuer_generation,
    NativeMarketArm experimental_market_arm,
    NativePhasedMarketAudit* phased_market_audit,
    NativePhasedStepAudit* phased_step_audit,
    bool neutral_special_economy,
    NativeRepairOptions repair_options,
    NativeRepairAudit* repair_audit) const {
  NativeCommittedAction output;
  output.action = action_external(
      env, player, route, state, experimental_market_arm,
      phased_market_audit, phased_step_audit, neutral_special_economy,
      repair_options, repair_audit);
  auto& commitment = output.movement;
  commitment.player = player;
  commitment.day = env.day();
  commitment.issuer_generation = issuer_generation;
  commitment.issued_step = env.step_count();
  commitment.immutable_through_step = env.step_count();
  commitment.remaining_day_complete = false;
  for (std::size_t actor = 0; actor < output.action.units.size(); ++actor) {
    const auto action = output.action.units[actor];
    if (action.op == Op::NORTH || action.op == Op::SOUTH ||
        action.op == Op::EAST || action.op == Op::WEST) {
      commitment.moves.push_back(
          {static_cast<int>(actor), env.step_count(), action});
    }
  }
  commitment.content_hash = native_movement_commitment_hash(commitment);
  return output;
}

NativeMatchResult NativeTeammateExecutor::play(int route0, int route1,
                                                uint64_t seed,
                                                int switch_step0,
                                                int switch_route0,
                                                int switch_step1,
                                                int switch_route1,
                                                bool capture_trace,
                                                bool capture_audit,
                                                bool neutral_special_economy,
                                                NativeRepairOptions repair_options,
                                                int repair_player,
                                                bool experimental_general_takeover,
                                                int general_takeover_player,
                                                int experimental_market_arm,
                                                int experimental_market_player,
                                                int stop_after_steps,
                                                int evaluation_compaction_minimum_step,
                                                int evaluation_compaction_evidence_mode,
                                                int thomas_prefix_player) const {
  if (experimental_market_arm < static_cast<int>(NativeMarketArm::LegacyDefault) ||
      experimental_market_arm > static_cast<int>(NativeMarketArm::Phased))
    throw std::invalid_argument("experimental market arm must be 0, 1, or 2");
  if (experimental_general_takeover && experimental_market_arm != 0)
    throw std::invalid_argument(
        "experimental general takeover and phased market arm are mutually exclusive");
  if (stop_after_steps == 0 || stop_after_steps < -1)
    throw std::invalid_argument("stop_after_steps must be -1 or positive");
  if (evaluation_compaction_minimum_step < -1 ||
      evaluation_compaction_minimum_step >= 720)
    throw std::invalid_argument(
        "evaluation compaction minimum step must be -1 or in [0,719]");
  if (evaluation_compaction_evidence_mode < 0 ||
      evaluation_compaction_evidence_mode >
          static_cast<int>(NativeCompactionEvidenceMode::RecentClearanceLower))
    throw std::invalid_argument(
        "evaluation compaction evidence mode must be in [0,6]");
  if (thomas_prefix_player < -2 || thomas_prefix_player > 1)
    throw std::invalid_argument("Thomas prefix player must be -2, -1, 0, or 1");
  Simulator env(Config{}, seed);
  NativeAgentState states[2];
  ThomasPrefixMarketState thomas_prefix[2];
  if (evaluation_compaction_minimum_step >= 0 ||
      evaluation_compaction_evidence_mode != 0) {
    for (int player = 0; player < 2; ++player) {
      if (experimental_market_player >= 0 &&
          experimental_market_player != player)
        continue;
      states[player].phased_market.evaluation_profile.enabled = true;
      states[player].phased_market.evaluation_profile.compaction_minimum_step =
          evaluation_compaction_minimum_step;
      states[player].phased_market.evaluation_profile.evidence_mode =
          static_cast<NativeCompactionEvidenceMode>(
              evaluation_compaction_evidence_mode);
    }
  }
  NativeMatchResult result;
  if (capture_trace) result.trace.reserve(719);
  while (!env.done() &&
         (stop_after_steps < 0 || env.step_count() < stop_after_steps)) {
    const int active0 = switch_step0 >= 0 && env.step_count() >= switch_step0
                            ? switch_route0 : route0;
    const int active1 = switch_step1 >= 0 && env.step_count() >= switch_step1
                            ? switch_route1 : route1;
    std::array<PlayerAction, 2> actions{
        action(env, 0, active0, states[0], neutral_special_economy,
               repair_player < 0 || repair_player == 0
                   ? repair_options : NativeRepairOptions{},
               &result.repair_audit[0],
               experimental_general_takeover &&
                   (general_takeover_player < 0 || general_takeover_player == 0),
               &result.economy_audit[0], switch_step0,
               experimental_market_player < 0 || experimental_market_player == 0
                   ? static_cast<NativeMarketArm>(experimental_market_arm)
                   : NativeMarketArm::LegacyDefault,
               &result.phased_market_audit[0]),
        action(env, 1, active1, states[1], neutral_special_economy,
               repair_player < 0 || repair_player == 1
                   ? repair_options : NativeRepairOptions{},
               &result.repair_audit[1],
               experimental_general_takeover &&
                   (general_takeover_player < 0 || general_takeover_player == 1),
               &result.economy_audit[1], switch_step1,
               experimental_market_player < 0 || experimental_market_player == 1
                   ? static_cast<NativeMarketArm>(experimental_market_arm)
                   : NativeMarketArm::LegacyDefault,
               &result.phased_market_audit[1])};
    if (thomas_prefix_player != -2) {
      static const std::vector<std::vector<NativeTapeLibrary::ThomasPredictEvent>>
          empty_predict_streams;
      const auto& shops = env.shops();
      const auto& predict_streams = shops.size() >= 2
          ? library_.thomas_predict_pairs[int(shops[0]) * 8 + int(shops[1])]
          : empty_predict_streams;
      if (thomas_prefix_player < 0 || thomas_prefix_player == 0)
        apply_thomas_prefix_market(env, 0, library_.routes[active0], predict_streams, actions[0],
                                   thomas_prefix[0]);
      if (thomas_prefix_player < 0 || thomas_prefix_player == 1)
        apply_thomas_prefix_market(env, 1, library_.routes[active1], predict_streams, actions[1],
                                   thomas_prefix[1]);
    }
    std::array<int, 2> unit_failures{};
    if (capture_audit) unit_failures = {
        macro_unit_failures(env, 0, actions[0]),
        macro_unit_failures(env, 1, actions[1])};
    if (capture_trace) result.trace.push_back(actions);
    const bool first200 = env.step_count() < 200;
    env.step(actions);
    if (capture_audit) for (int player = 0; player < 2; ++player) {
      const int market_failures = macro_market_failures(env, player, actions[player]);
      result.macro_unit_failures[player] += unit_failures[player];
      result.macro_market_failures[player] += market_failures;
      if (first200) {
        result.first200_unit_failures[player] += unit_failures[player];
        result.first200_market_failures[player] += market_failures;
      }
      if ((unit_failures[player] || market_failures) &&
          result.first_macro_failure_step[player] < 0)
        result.first_macro_failure_step[player] = env.step_count() - 1;
    }
  }
  result.rewards = {env.farms()[0].money, env.farms()[1].money};
  return result;
}

std::array<float, 147> NativeTeammateExecutor::features_at(
    int route0, int route1, uint64_t seed, int checkpoint, int player,
    int feature_route) const {
  if (player < 0 || player > 1 || feature_route < 0 ||
      feature_route >= int(library_.routes.size()))
    throw std::invalid_argument("invalid native feature player or route");
  Simulator env(Config{}, seed);
  NativeAgentState states[2]; FeatureHistory history;
  while (!env.done()) {
    history.update(env);
    if (env.step_count() == checkpoint)
      return build_features(env, player, history, library_.routes[feature_route]);
    std::array<PlayerAction, 2> actions{
        action(env, 0, route0, states[0], false, {}, nullptr),
        action(env, 1, route1, states[1], false, {}, nullptr)};
    env.step(actions);
  }
  throw std::invalid_argument("checkpoint is outside the episode");
}

}  // namespace fastkag
