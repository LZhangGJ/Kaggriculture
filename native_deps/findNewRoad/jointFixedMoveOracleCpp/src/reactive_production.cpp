#include "../include/reactive_production.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

namespace joint_fixed_move_oracle::production {
namespace {

bool is_move(fastkag::Op op) {
  return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
         op == fastkag::Op::EAST || op == fastkag::Op::WEST;
}

bool same_action(const fastkag::Action& left, const fastkag::Action& right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

std::vector<fastkag::Position> positions(const fastkag::Farm& farm) {
  std::vector<fastkag::Position> out{farm.farmer};
  out.insert(out.end(), farm.hands.begin(), farm.hands.end());
  return out;
}

bool valid_crop(fastkag::Item item) {
  const int value = static_cast<int>(item);
  return value >= 0 && value < fastkag::N_CROPS;
}

bool mature(const fastkag::Tile& tile, int day) {
  constexpr std::array<int, fastkag::N_CROPS> first_day{2, 2, 8, 10, 10};
  const int crop = static_cast<int>(tile.crop);
  return crop >= 0 && crop < fastkag::N_CROPS && tile.yield_units > 0 &&
         day - tile.planted_day >= first_day[static_cast<std::size_t>(crop)];
}

int first_maturity_days(fastkag::Item item) {
  constexpr std::array<int, fastkag::N_CROPS> days{2, 2, 8, 10, 10};
  const int crop = static_cast<int>(item);
  return crop >= 0 && crop < fastkag::N_CROPS
      ? days[static_cast<std::size_t>(crop)] : 1'000'000;
}

bool can_finish_new_cycle(const fastkag::Simulator& simulator,
                          fastkag::Item crop, int cycles = 1) {
  const int last_day = (simulator.config().episode_steps - 1) /
                       simulator.config().turns_per_day;
  return simulator.day() + first_maturity_days(crop) * std::max(1, cycles) <=
         last_day;
}

void defer_nonmove(RouteCursorState& state, RouteCursorAudit& audit,
                   std::size_t actor, int source,
                   const fastkag::Action& action, fastkag::Position tile,
                   int deadline_step) {
  if (state.deferred_nonmoves.size() <= actor)
    state.deferred_nonmoves.resize(actor + 1);
  state.deferred_nonmoves[actor].push_back(
      {source, action, tile, action.item, deadline_step});
  ++audit.deferred_nonmoves;
}

void prepare_route_day(const std::vector<fastkag::PlayerAction>& tape,
                       const fastkag::Simulator& simulator,
                       std::size_t actor_count, RouteCursorState& state,
                       RouteCursorAudit& audit) {
  const int step = simulator.step_count();
  const int turns = simulator.config().turns_per_day;
  const int day = step / turns;
  const int day_start = day * turns;
  if (state.day != day) {
    if (state.day >= 0) {
      const int previous_start = state.day * turns;
      const int previous_end = std::min(
          previous_start + turns - 1, static_cast<int>(tape.size()) - 1);
      for (std::size_t actor = 0; actor < state.source_cursor.size(); ++actor) {
        for (int source = std::max(previous_start, state.source_cursor[actor]);
             source <= previous_end; ++source) {
          const auto deferred =
              actor < tape[static_cast<std::size_t>(source)].units.size()
                  ? tape[static_cast<std::size_t>(source)].units[actor]
                  : fastkag::Action{};
          if (!is_move(deferred.op))
            defer_nonmove(state, audit, actor, source, deferred, {-1, -1},
                          previous_end);
        }
      }
    }
    state.day = day;
    state.source_cursor.assign(actor_count, day_start);
    // Hands disappear at every day boundary and are re-hired later. Their
    // deferred ledgers are cross-day transaction state, not active-actor
    // scratch; shrinking here silently deleted every debt for actor >= 1.
    if (state.deferred_nonmoves.size() < actor_count)
      state.deferred_nonmoves.resize(actor_count);
  } else if (state.source_cursor.size() < actor_count) {
    state.source_cursor.resize(actor_count, step);
    if (state.deferred_nonmoves.size() < actor_count)
      state.deferred_nonmoves.resize(actor_count);
  }
}

}  // namespace

void ReactiveState::reset(int tile_count) {
  debts.assign(static_cast<std::size_t>(std::max(0, tile_count)), {});
  last_crop.assign(static_cast<std::size_t>(std::max(0, tile_count)),
                   fastkag::Item::NONE);
  last_step = -1;
}

fastkag::PlayerAction apply_reactive_fixed_move(
    const fastkag::PlayerAction& raw, const fastkag::Simulator& simulator,
    int player, ReactiveState& state, ReactiveAudit& audit,
    const ReactiveOptions& options) {
  if (player < 0 || player > 1)
    throw std::invalid_argument("reactive production player must be 0 or 1");
  const auto& farm = simulator.farms()[static_cast<std::size_t>(player)];
  const auto& private_state =
      simulator.privates()[static_cast<std::size_t>(player)];
  if (simulator.step_count() == 0 || simulator.step_count() <= state.last_step ||
      state.debts.size() != farm.tiles.size())
    state.reset(static_cast<int>(farm.tiles.size()));
  state.last_step = simulator.step_count();
  for (std::size_t tile = 0; tile < farm.tiles.size(); ++tile)
    if (farm.tiles[tile].kind == fastkag::TileKind::PLANT &&
        valid_crop(farm.tiles[tile].crop))
      state.last_crop[tile] = farm.tiles[tile].crop;

  auto out = raw;
  const auto actor_positions = positions(farm);
  std::vector<bool> active_actor(out.units.size(), false);
  std::array<int, fastkag::N_CROPS> reserved_plants{};
  std::array<int, fastkag::N_CROPS> raw_plant_demand{};
  for (const auto& action : raw.units)
    if (action.op == fastkag::Op::PLANT && valid_crop(action.item))
      ++raw_plant_demand[static_cast<std::size_t>(action.item)];
  for (std::size_t actor = 0;
       actor < out.units.size() && actor < actor_positions.size(); ++actor) {
    const auto original = raw.units[actor];
    if (is_move(original.op)) continue;
    const auto position = actor_positions[actor];
    if (position.x < 0 || position.y < 0 ||
        position.x >= simulator.config().board_size ||
        position.y >= simulator.config().board_size)
      continue;
    const int tile_index =
        position.y * simulator.config().board_size + position.x;
    const auto& tile = farm.tiles[static_cast<std::size_t>(tile_index)];
    auto& debt = state.debts[static_cast<std::size_t>(tile_index)];

    // A failed seed purchase is observed causally as an EMPTY tile, a due
    // PLANT intent, and insufficient seeds for the simulator's all-or-none
    // crop batch.  Materialize the same semantic tile debt used by weed
    // recovery; the online owner may retry the missing market purchase.
    const bool due_plant = original.op == fastkag::Op::PLANT &&
                           valid_crop(original.item);
    if (options.recover_missing_seed_debt && !debt.active &&
        tile.kind == fastkag::TileKind::EMPTY && due_plant &&
        raw_plant_demand[static_cast<std::size_t>(original.item)] >
            private_state.seeds[static_cast<std::size_t>(original.item)] &&
        simulator.step_count() <= options.latest_new_debt_step) {
      debt = {true, false, original.item, fastkag::Op::PASS,
              simulator.step_count()};
      if (options.causal_cycle_guard &&
          !can_finish_new_cycle(simulator, debt.crop, 2)) {
        debt.active = false;
        ++audit.abandoned_cycles;
      }
    }

    if (debt.active && !debt.structure && options.causal_cycle_guard &&
        (tile.kind == fastkag::TileKind::WEED ||
         tile.kind == fastkag::TileKind::EMPTY) &&
        !can_finish_new_cycle(simulator, debt.crop,
                              debt.confirmed_harvests == 0 ? 2 : 1)) {
      debt.active = false;
      ++audit.abandoned_cycles;
    }

    if (tile.kind == fastkag::TileKind::WEED && !debt.active &&
        simulator.step_count() <= options.latest_new_debt_step) {
      const bool plant_event = original.op == fastkag::Op::PLANT &&
                               valid_crop(original.item);
      const bool structure_event = original.op == fastkag::Op::BUILD_PASTURE ||
                                   original.op == fastkag::Op::BUILD_COOP;
      const bool old_crop_event = options.recover_old_crop_weeds &&
          valid_crop(state.last_crop[static_cast<std::size_t>(tile_index)]) &&
          simulator.market().prices[static_cast<std::size_t>(
              state.last_crop[static_cast<std::size_t>(tile_index)])] >=
              options.minimum_old_crop_market_price;
      if (plant_event || structure_event || old_crop_event) {
        const int ordinal = audit.candidate_weed_events++;
        const bool enabled = options.allowed_event_ordinal < 0 ||
                             options.allowed_event_ordinal == ordinal ||
                             options.allowed_event_ordinal_second == ordinal ||
                             options.allowed_event_ordinal_third == ordinal;
        if (enabled) {
          ++audit.weed_events;
          if (plant_event) {
            debt = {true, false, original.item, fastkag::Op::PASS,
                    simulator.step_count(), 0, -1};
          } else if (structure_event) {
            debt = {true, true, fastkag::Item::NONE, original.op,
                    simulator.step_count(), 0, -1};
          } else {
            ++audit.recovered_old_crop_weeds;
            debt = {true, false,
                    state.last_crop[static_cast<std::size_t>(tile_index)],
                    fastkag::Op::PASS, simulator.step_count(), 0, -1};
          }
          if (debt.active && !debt.structure &&
              options.causal_cycle_guard &&
              !can_finish_new_cycle(simulator, debt.crop, 2)) {
            debt.active = false;
            ++audit.abandoned_cycles;
          }
        }
      }
    }
    const bool manage_known = options.manage_all_known_crops &&
        valid_crop(state.last_crop[static_cast<std::size_t>(tile_index)]) &&
        (tile.kind == fastkag::TileKind::PLANT ||
         (tile.kind == fastkag::TileKind::EMPTY &&
          simulator.step_count() <= options.latest_new_debt_step));
    if (!debt.active && !manage_known) continue;
    active_actor[actor] = true;
    ++audit.active_debt_steps;

    if (debt.structure) {
      if (tile.kind == fastkag::TileKind::WEED) {
        out.units[actor] = {fastkag::Op::DIG};
        ++audit.digs;
      } else if (tile.kind == fastkag::TileKind::EMPTY) {
        out.units[actor] = {debt.structure_op};
        ++audit.structures;
      } else if ((debt.structure_op == fastkag::Op::BUILD_PASTURE &&
                  tile.kind == fastkag::TileKind::PASTURE) ||
                 (debt.structure_op == fastkag::Op::BUILD_COOP &&
                  tile.kind == fastkag::TileKind::COOP)) {
        if (debt.active) ++audit.completed_cycles;
        debt.active = false;
      } else {
        out.units[actor] = {};
      }
      continue;
    }

    if (tile.kind == fastkag::TileKind::WEED) {
      out.units[actor] = {fastkag::Op::DIG};
      ++audit.digs;
    } else if (tile.kind == fastkag::TileKind::EMPTY) {
      const auto desired_crop = debt.active
          ? debt.crop
          : state.last_crop[static_cast<std::size_t>(tile_index)];
      const int crop = static_cast<int>(desired_crop);
      if (crop >= 0 && crop < fastkag::N_CROPS &&
          private_state.seeds[static_cast<std::size_t>(crop)] >
              reserved_plants[static_cast<std::size_t>(crop)]) {
        out.units[actor] = {fastkag::Op::PLANT, desired_crop, 1};
        ++reserved_plants[static_cast<std::size_t>(crop)];
        ++audit.plants;
      } else {
        out.units[actor] = {};
        ++audit.seed_waits;
      }
    } else if (tile.kind == fastkag::TileKind::PLANT) {
      if (valid_crop(tile.crop)) {
        if (debt.active) debt.crop = tile.crop;
        state.last_crop[static_cast<std::size_t>(tile_index)] = tile.crop;
      }
      if (mature(tile, simulator.day())) {
        out.units[actor] = {fastkag::Op::HARVEST};
        ++audit.harvests;
        // The delayed lifecycle has reached a legal terminal production
        // action.  Hand control back to the source program after this harvest
        // instead of permanently taking over the tile's future crop cycles.
        if (debt.active) ++audit.completed_cycles;
        debt.active = false;
      } else if (!tile.watered_today) {
        out.units[actor] = {fastkag::Op::WATER};
        ++audit.waters;
      } else if (original.op == fastkag::Op::FERTILIZE &&
                 tile.fertilized_until_day < simulator.day() &&
                 private_state.inventories.size() > actor &&
                 private_state.inventories[actor][fastkag::N_PRODUCTS - 1] > 0) {
        out.units[actor] = {fastkag::Op::FERTILIZE};
        ++audit.fertilizes;
      } else {
        out.units[actor] = {};
      }
    } else {
      out.units[actor] = {};
    }
  }

  // The simulator's PLANT guard is all-or-none by crop.  Cap aggregate raw +
  // repaired demand to observed seeds, dropping unaffected raw PLANT actions
  // first so a debt repayment cannot be nullified by an over-demand batch.
  for (int crop = 0; crop < fastkag::N_CROPS; ++crop) {
    std::vector<std::size_t> planters;
    for (std::size_t actor = 0; actor < out.units.size(); ++actor)
      if (out.units[actor].op == fastkag::Op::PLANT &&
          static_cast<int>(out.units[actor].item) == crop)
        planters.push_back(actor);
    int excess = static_cast<int>(planters.size()) -
                 private_state.seeds[static_cast<std::size_t>(crop)];
    if (excess <= 0) continue;
    std::stable_sort(planters.begin(), planters.end(), [&](std::size_t a,
                                                           std::size_t b) {
      return active_actor[a] < active_actor[b];
    });
    for (const auto actor : planters) {
      if (excess-- <= 0) break;
      out.units[actor] = {};
      ++audit.seed_demand_suppressed;
    }
  }

  for (std::size_t actor = 0;
       actor < raw.units.size() && actor < out.units.size(); ++actor) {
    if ((is_move(raw.units[actor].op) || is_move(out.units[actor].op)) &&
        (raw.units[actor].op != out.units[actor].op ||
         raw.units[actor].item != out.units[actor].item ||
         raw.units[actor].quantity != out.units[actor].quantity))
      ++audit.move_mismatches;
  }
  return out;
}

fastkag::PlayerAction apply_reactive_route_cursor(
    const std::vector<fastkag::PlayerAction>& tape,
    const fastkag::Simulator& simulator, int player, RouteCursorState& state,
    RouteCursorAudit& audit, const ReactiveOptions& options) {
  if (tape.empty()) return {};
  const int step = simulator.step_count();
  const int turns = simulator.config().turns_per_day;
  const int day = step / turns;
  const int day_start = day * turns;
  const int day_end = std::min(day_start + turns - 1,
                               static_cast<int>(tape.size()) - 1);
  const auto& absolute = tape[std::min<std::size_t>(
      static_cast<std::size_t>(step), tape.size() - 1)];
  const auto actor_positions = positions(simulator.farms()[player]);
  // Only workers that actually exist before this unit phase own a route
  // cursor.  A planned HIRE is processed after unit actions; replaying the
  // beginning of the day for a newly hired hand invents moves it never had.
  const std::size_t actor_count = actor_positions.size();
  prepare_route_day(tape, simulator, actor_count, state, audit);

  auto tape_action = [&](int source, std::size_t actor) {
    if (source < 0 || source >= static_cast<int>(tape.size()) ||
        actor >= tape[static_cast<std::size_t>(source)].units.size())
      return fastkag::Action{};
    return tape[static_cast<std::size_t>(source)].units[actor];
  };
  auto remaining_moves = [&](int cursor, std::size_t actor) {
    int count = 0;
    for (int source = std::max(cursor, day_start); source <= day_end; ++source)
      count += is_move(tape_action(source, actor).op);
    return count;
  };
  auto next_move = [&](int cursor, std::size_t actor) {
    for (int source = std::max(cursor, day_start); source <= day_end; ++source)
      if (is_move(tape_action(source, actor).op)) return source;
    return day_end + 1;
  };

  fastkag::PlayerAction probe = absolute;
  probe.units.resize(actor_count);
  std::vector<fastkag::Action> source_actions(actor_count);
  std::vector<bool> forced(actor_count, false);
  // FastKag's final actionable step is episode_steps-2.  Use the actual tape
  // / episode boundary rather than assuming the last day has 24 slots; the
  // G001 tape ends at step 718, so treating step 719 as usable can drop the
  // final daily MOVE.
  const int last_action_step = std::min(
      day_start + turns - 1, simulator.config().episode_steps - 2);
  const int slots_remaining = std::max(0, last_action_step - step + 1);
  for (std::size_t actor = 0; actor < actor_count; ++actor) {
    int& cursor = state.source_cursor[actor];
    cursor = std::clamp(cursor, day_start, day_end + 1);
    source_actions[actor] = tape_action(cursor, actor);
    const int move_count = remaining_moves(cursor, actor);
    if (move_count >= slots_remaining && move_count > 0) {
      const int target = next_move(cursor, actor);
      for (int source = cursor; source < target; ++source) {
        const auto deferred = tape_action(source, actor);
        if (is_move(deferred.op)) continue;
        defer_nonmove(state, audit, actor, source, deferred,
                      actor_positions[actor], last_action_step);
      }
      audit.skipped_nonmoves += std::max(0, target - cursor);
      cursor = target;
      source_actions[actor] = tape_action(cursor, actor);
      probe.units[actor] = source_actions[actor];
      forced[actor] = true;
      ++audit.forced_moves;
    } else if (is_move(source_actions[actor].op)) {
      // Probe the current tile for production debt before committing the MOVE.
      probe.units[actor] = {};
    } else {
      probe.units[actor] = source_actions[actor];
    }
  }
  probe.market = absolute.market;
  if (options.require_legacy_drop_move_for_new_debt) {
    const auto& farm = simulator.farms()[player];
    const auto& private_state = simulator.privates()[player];
    for (std::size_t actor = 0; actor < actor_count; ++actor) {
      const auto action = probe.units[actor];
      if (action.op != fastkag::Op::PLANT &&
          action.op != fastkag::Op::BUILD_PASTURE &&
          action.op != fastkag::Op::BUILD_COOP) continue;
      const auto position = actor_positions[actor];
      if (position.x < 0 || position.y < 0 ||
          position.x >= simulator.config().board_size ||
          position.y >= simulator.config().board_size) continue;
      const auto& tile = farm.tiles[static_cast<std::size_t>(
          position.y * simulator.config().board_size + position.x)];
      bool blocked = tile.kind == fastkag::TileKind::WEED;
      if (action.op == fastkag::Op::PLANT && valid_crop(action.item))
        blocked = blocked ||
            private_state.seeds[static_cast<std::size_t>(action.item)] <= 0;
      if (!blocked) continue;
      const int legacy_drop = state.source_cursor[actor] + 9;
      if (legacy_drop / turns == day &&
          !is_move(tape_action(legacy_drop, actor).op))
        probe.units[actor] = {};
    }
  }
  auto compiled = apply_reactive_fixed_move(
      probe, simulator, player, state.production, audit.production, options);
  compiled.units.resize(actor_count);
  compiled.market = absolute.market;

  for (std::size_t actor = 0; actor < actor_count; ++actor) {
    int& cursor = state.source_cursor[actor];
    const auto source = source_actions[actor];
    if (forced[actor]) {
      compiled.units[actor] = source;
      ++cursor;
      continue;
    }
    if (is_move(source.op)) {
      if (compiled.units[actor].op == fastkag::Op::PASS) {
        compiled.units[actor] = source;
        ++cursor;
      } else {
        ++audit.inserted_before_move;
      }
      continue;
    }
    const bool collision_insert =
        (source.op == fastkag::Op::PLANT ||
         source.op == fastkag::Op::BUILD_PASTURE ||
         source.op == fastkag::Op::BUILD_COOP) &&
        compiled.units[actor].op == fastkag::Op::DIG;
    if (!collision_insert && cursor <= day_end) ++cursor;
  }

  return compiled;
}

RouteCursorProposal propose_reactive_route_cursor(
    const std::vector<fastkag::PlayerAction>& tape,
    const fastkag::Simulator& simulator, int player,
    const RouteCursorState& state, const RouteCursorAudit& audit,
    const ReactiveOptions& options) {
  RouteCursorProposal proposal;
  proposal.prior_state = state;
  proposal.next_state = state;
  proposal.prior_audit = audit;
  proposal.next_audit = audit;
  proposal.action = apply_reactive_route_cursor(
      tape, simulator, player, proposal.next_state, proposal.next_audit,
      options);

  // `apply_reactive_fixed_move` historically cleared debt/counts when it
  // emitted HARVEST. Tentatively turn the first verified harvest into the
  // explicit replant suffix. Because this is only next_state, an overlay or a
  // failed effect cannot advance it.
  const auto actor_positions = positions(simulator.farms()[player]);
  for (std::size_t actor = 0;
       actor < proposal.action.units.size() && actor < actor_positions.size();
       ++actor) {
    if (proposal.action.units[actor].op != fastkag::Op::HARVEST) continue;
    const auto position = actor_positions[actor];
    if (position.x < 0 || position.y < 0 ||
        position.x >= simulator.config().board_size ||
        position.y >= simulator.config().board_size) continue;
    const std::size_t tile = static_cast<std::size_t>(
        position.y * simulator.config().board_size + position.x);
    if (tile >= proposal.prior_state.production.debts.size() ||
        tile >= proposal.next_state.production.debts.size()) continue;
    const auto& before = proposal.prior_state.production.debts[tile];
    auto& after = proposal.next_state.production.debts[tile];
    if (!before.active || before.structure || after.active ||
        before.confirmed_harvests != 0) continue;
    after = before;
    after.active = true;
    after.confirmed_harvests = 1;
    if (proposal.next_audit.production.completed_cycles >
        proposal.prior_audit.production.completed_cycles)
      --proposal.next_audit.production.completed_cycles;
  }

  return proposal;
}

RouteCursorProposal reconcile_reactive_route_cursor_final(
    const std::vector<fastkag::PlayerAction>& tape,
    const fastkag::Simulator& simulator, int player,
    const RouteCursorProposal& proposal,
    const std::vector<fastkag::Action>& final_units,
    bool accept_exact_production) {
  bool exact = final_units.size() == proposal.action.units.size();
  for (std::size_t actor = 0; exact && actor < final_units.size(); ++actor)
    exact = same_action(final_units[actor], proposal.action.units[actor]);
  if (exact && accept_exact_production) return proposal;

  RouteCursorProposal reconciled;
  reconciled.action = proposal.action;
  reconciled.action.units = final_units;
  reconciled.prior_state = proposal.prior_state;
  reconciled.next_state = proposal.prior_state;
  reconciled.prior_audit = proposal.prior_audit;
  reconciled.next_audit = proposal.prior_audit;
  if (tape.empty() || player < 0 || player > 1) return reconciled;

  const auto actor_positions = positions(simulator.farms()[player]);
  const std::size_t actor_count = actor_positions.size();
  prepare_route_day(tape, simulator, actor_count, reconciled.next_state,
                    reconciled.next_audit);
  const int step = simulator.step_count();
  const int turns = simulator.config().turns_per_day;
  const int day_start = (step / turns) * turns;
  const int day_end = std::min(day_start + turns - 1,
                               static_cast<int>(tape.size()) - 1);
  const int deadline = std::min(day_start + turns - 1,
                                simulator.config().episode_steps - 2);
  auto tape_action = [&](int source, std::size_t actor) {
    if (source < 0 || source >= static_cast<int>(tape.size()) ||
        actor >= tape[static_cast<std::size_t>(source)].units.size())
      return fastkag::Action{};
    return tape[static_cast<std::size_t>(source)].units[actor];
  };

  for (std::size_t actor = 0;
       actor < actor_count && actor < final_units.size(); ++actor) {
    if (!is_move(final_units[actor].op)) continue;
    int& cursor = reconciled.next_state.source_cursor[actor];
    cursor = std::clamp(cursor, day_start, day_end + 1);
    int next_move_source = day_end + 1;
    for (int source = cursor; source <= day_end; ++source) {
      if (is_move(tape_action(source, actor).op)) {
        next_move_source = source;
        break;
      }
    }
    // Ordered skeleton rule: never search past a mismatching first MOVE.
    if (next_move_source > day_end ||
        !same_action(final_units[actor],
                     tape_action(next_move_source, actor)))
      continue;
    for (int source = cursor; source < next_move_source; ++source) {
      const auto crossed = tape_action(source, actor);
      if (!is_move(crossed.op)) {
        defer_nonmove(reconciled.next_state, reconciled.next_audit, actor,
                      source, crossed, actor_positions[actor], deadline);
      }
    }
    reconciled.next_audit.skipped_nonmoves +=
        std::max(0, next_move_source - cursor);
    ++reconciled.next_audit.forced_moves;
    cursor = next_move_source + 1;
  }
  return reconciled;
}

bool commit_reactive_route_cursor(
    const RouteCursorProposal& proposal,
    const std::vector<fastkag::Action>& final_units,
    bool effects_confirmed, RouteCursorState& state, RouteCursorAudit& audit) {
  if (!effects_confirmed || final_units.size() != proposal.action.units.size())
    return false;
  for (std::size_t actor = 0; actor < final_units.size(); ++actor) {
    const auto& expected = proposal.action.units[actor];
    const auto& actual = final_units[actor];
    if (expected.op != actual.op || expected.item != actual.item ||
        expected.quantity != actual.quantity)
      return false;
  }
  state = proposal.next_state;
  audit = proposal.next_audit;
  return true;
}

}  // namespace joint_fixed_move_oracle::production
