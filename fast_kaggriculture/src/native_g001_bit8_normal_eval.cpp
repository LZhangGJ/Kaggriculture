#include "native_teammate.hpp"
#include "route_loader.hpp"
#include "../../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/include/day_horizon_planner.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using fastkag::Action;
using fastkag::NativeAgentState;
using fastkag::NativeRepairAudit;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
using Tape = std::vector<PlayerAction>;
constexpr std::uint64_t kCoverageSeed = 0x6001F0CEDULL;

struct Options {
  std::uint64_t seed_begin{970001};
  int seeds{32};
  int coverage_step{27 * 24 + 3};
  int treatment_mask{8};
  std::string output{"g001-bit8-normal-eval.json"};
  std::string tapes{NATIVE_G001_TAPES};
  std::string library{NATIVE_G001_LIBRARY};
};

struct Metrics {
  double own{};
  double opponent{};
  int hires{};
  int unit_failures{};
  int market_failures{};
  int move_overwrites{};
  int ordered_move_day_failures{};
  int move_timing_deviation{};
  int delayed_moves{};
  int maximum_move_delay{};
  int open_plot_transactions{};
  int minimal_plot_objectives{};
  int critical_source_transactions{};
  int critical_source_units{};
  int pending_unit_receipts{};
  int pending_purchase_receipts{};
  int terminal_identity_leases{};
  NativeRepairAudit audit;
};

struct Row {
  std::uint64_t seed{};
  int seat{};
  std::array<Metrics, 2> arms;
};

struct PairedSeed {
  std::uint64_t seed{};
  int games{};
  double own{};
  double margin{};
  double hires{};
  double unit_failures{};
  double market_failures{};
  double move_timing_deviation{};
};

struct Interval {
  double lower{};
  double upper{};
};

struct Coverage {
  std::uint64_t simulation_seed{};
  int step{};
  int day{};
  int start_hour{};
  int actors{};
  int turns{};
  int open_transactions{};
  int objectives{};
  int coalesced_duplicates{};
  int weed_tiles{};
  int weed_objectives{};
  int plant_objectives{};
  int objective_transition_actions{};
  int scheduled{};
  int unscheduled{};
  int assignments{};
  int move_edits{};
  int local_overlay_absolute_move_edits{};
  int raw_move_slots{};
  int raw_pass_slots{};
  int objective_tile_capacity_slots{};
  int inventory_unproven_assignments{};
  int unscheduled_no_transition{};
  int unscheduled_seed_constraint{};
  int unscheduled_no_tile_capacity{};
  int unscheduled_insufficient_suffix_capacity{};
  int unscheduled_contention_or_priority{};
  int route_sequence_completed_objectives{};
  int route_sequence_unscheduled_objectives{};
  int route_sequence_assignments{};
  int route_sequence_move_timing_deviation{};
  int route_sequence_derived_move_timing_deviation{};
  int route_sequence_delayed_moves{};
  int route_sequence_max_move_delay{};
  int route_sequence_terminal_unexecuted_moves{};
  int route_sequence_terminal_unexecuted_raw_actions{};
  int route_sequence_late_assignments{};
  int route_sequence_absolute_move_edits{};
  std::map<int, int> route_sequence_move_delay_histogram;
  bool route_sequence_ordered_moves_exact{};
  bool route_sequence_move_slots_exact{};
  bool route_sequence_positions_legal{};
  bool route_sequence_timing_accounting_exact{};
  int time_expanded_completed_objectives{};
  int time_expanded_completed_value{};
  int time_expanded_unscheduled_objectives{};
  int time_expanded_timing_cost{};
  int time_expanded_total_cost{};
  int time_expanded_terminal_raw{};
  std::size_t time_expanded_states{};
  std::uint64_t time_expanded_elapsed_us{};
  bool time_expanded_exact{};
  bool time_expanded_fallback{};
  bool time_expanded_within_budget{};
  bool time_expanded_ordered_moves{};
  int time_expanded_components{};
  int time_expanded_largest_component{};
  int time_expanded_exact_objectives{};
  int time_expanded_joint_objectives{};
  int time_expanded_single_exact_components{};
  int time_expanded_joint_components{};
  int time_expanded_fallback_components{};
  std::map<int, int> time_expanded_component_histogram;
  std::array<int, fastkag::N_CROPS> seed_snapshot{};
  std::array<int, fastkag::N_CROPS> raw_seed_reservation{};
  std::array<int, fastkag::N_CROPS> repair_seed_demand{};
  std::array<int, fastkag::N_CROPS> scheduled_seed_demand{};
  std::vector<std::array<int, 2>> start_positions;
  bool move_slots_exact{};
};

template <std::size_t Size>
void write_int_array(std::ostream& output,
                     const std::array<int, Size>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

void write_positions(std::ostream& output,
                     const std::vector<std::array<int, 2>>& positions) {
  output << '[';
  for (std::size_t index = 0; index < positions.size(); ++index) {
    if (index != 0) output << ',';
    output << "[" << positions[index][0] << ',' << positions[index][1]
           << ']';
  }
  output << ']';
}

void write_int_map(std::ostream& output, const std::map<int, int>& values) {
  output << '{';
  bool first = true;
  for (const auto& [key, value] : values) {
    if (!first) output << ',';
    first = false;
    output << '\"' << key << "\":" << value;
  }
  output << '}';
}

bool move(Op operation) {
  return operation == Op::NORTH || operation == Op::SOUTH ||
      operation == Op::EAST || operation == Op::WEST;
}

Action unit(const PlayerAction& action, std::size_t actor) {
  return actor < action.units.size() ? action.units[actor] : Action{};
}

bool same(const Action& left, const Action& right) {
  return left.op == right.op && left.item == right.item &&
      left.quantity == right.quantity;
}

g001::event_local_repair::Action local_action(Action action) {
  namespace local = g001::event_local_repair;
  local::Action result;
  result.item = static_cast<int>(action.item);
  result.quantity = action.quantity;
  result.arg0 = static_cast<int>(action.op);
  switch (action.op) {
    case Op::PASS: result.op = local::Op::Pass; break;
    case Op::NORTH: result.op = local::Op::Move; result.arg0 = 0; break;
    case Op::SOUTH: result.op = local::Op::Move; result.arg0 = 1; break;
    case Op::WEST: result.op = local::Op::Move; result.arg0 = 2; break;
    case Op::EAST: result.op = local::Op::Move; result.arg0 = 3; break;
    case Op::DIG: result.op = local::Op::Dig; break;
    case Op::PLANT: result.op = local::Op::Plant; break;
    case Op::BUILD_COOP:
    case Op::BUILD_PASTURE: result.op = local::Op::Build; break;
    case Op::WATER: result.op = local::Op::Water; break;
    case Op::HARVEST: result.op = local::Op::Harvest; break;
    default: result.op = local::Op::Other; break;
  }
  return result;
}

Coverage representative_day_coverage(const NativeTeammateExecutor& executor,
                                     const Tape& tape, int coverage_step) {
  namespace horizon = g001::day_horizon_repair;
  namespace local = g001::event_local_repair;
  fastkag::Config config;
  // Official G001 dynamics; the fixed seed/day below was selected only for
  // having observed weeds, many workers, and multiple crop objectives.
  Simulator simulator(config, kCoverageSeed);
  std::array<NativeAgentState, 2> states;
  NativeRepairAudit audit;
  const auto repair = fastkag::native_repair_options_from_mask(8);
  while (!simulator.done() && simulator.step_count() < coverage_step) {
    std::array<PlayerAction, 2> actions{
        executor.action_external(
            simulator, 0, 0, states[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true,
            repair, &audit),
        executor.action_external(simulator, 1, 0, states[1])};
    simulator.step(actions);
  }
  if (simulator.step_count() != coverage_step ||
      !states[0].experimental_event_local_repair)
    throw std::runtime_error("cannot reach G001 coverage boundary");

  // Resolve the receipt from the preceding turn and install this turn's raw
  // intent on a copy.  The real boundary state remains untouched for the
  // online-overlay counterfactual below.
  auto planner_state = states[0];
  auto planner_audit = audit;
  static_cast<void>(executor.action_external(
      simulator, 0, 0, planner_state,
      fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true, repair,
      &planner_audit));

  Coverage coverage;
  coverage.simulation_seed = kCoverageSeed;
  coverage.step = simulator.step_count();
  coverage.day = simulator.day();
  coverage.start_hour = simulator.hour();
  coverage.actors =
      static_cast<int>(simulator.farms()[0].hands.size() + 1);
  coverage.turns = std::min(
      static_cast<int>(tape.size()) - coverage.step,
      simulator.config().turns_per_day - coverage.start_hour);
  std::vector<fastkag::Position> starts{simulator.farms()[0].farmer};
  starts.insert(starts.end(), simulator.farms()[0].hands.begin(),
                simulator.farms()[0].hands.end());
  for (const auto position : starts)
    coverage.start_positions.push_back({position.y, position.x});
  std::vector<horizon::ActorPlan> actors;
  for (int actor = 0; actor < coverage.actors; ++actor) {
    horizon::ActorPlan plan;
    plan.actor = actor;
    plan.start = {starts[static_cast<std::size_t>(actor)].y,
                  starts[static_cast<std::size_t>(actor)].x};
    for (int step = coverage.step;
         step < coverage.step + coverage.turns; ++step)
      plan.raw.push_back(local_action(
          unit(tape[static_cast<std::size_t>(step)],
               static_cast<std::size_t>(actor))));
    for (const auto& action : plan.raw) {
      if (action.op == local::Op::Move) ++coverage.raw_move_slots;
      if (action.op == local::Op::Pass) ++coverage.raw_pass_slots;
      if (action.op == local::Op::Plant && action.item >= 0 &&
          action.item < fastkag::N_CROPS && action.quantity > 0)
        coverage.raw_seed_reservation[static_cast<std::size_t>(action.item)] +=
            action.quantity;
    }
    plan.blocked_equivalent_capacity.assign(plan.raw.size(), false);
    actors.push_back(std::move(plan));
  }

  // Observe the same remaining day with the online local overlay.  This is a
  // cloned counterfactual and cannot affect the offline planner input.
  {
    auto overlay_simulator = simulator;
    auto overlay_states = states;
    auto overlay_audit = audit;
    for (int turn = 0; turn < coverage.turns; ++turn) {
      const int step = overlay_simulator.step_count();
      const std::size_t active = overlay_simulator.farms()[0].hands.size() + 1;
      std::array<PlayerAction, 2> actions{
          executor.action_external(
              overlay_simulator, 0, 0, overlay_states[0],
              fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true,
              repair, &overlay_audit),
          executor.action_external(overlay_simulator, 1, 0,
                                   overlay_states[1])};
      for (std::size_t actor = 0; actor < active; ++actor) {
        const auto raw = unit(tape[static_cast<std::size_t>(step)], actor);
        const auto final = unit(actions[0], actor);
        if (!same(raw, final) && (move(raw.op) || move(final.op)))
          ++coverage.local_overlay_absolute_move_edits;
      }
      overlay_simulator.step(actions);
    }
  }

  const auto open =
      planner_state.experimental_event_local_repair->open_transactions();
  coverage.open_transactions = static_cast<int>(open.size());
  std::vector<horizon::Objective> objectives;
  std::map<std::uint64_t, int> suffix_sizes;
  std::set<std::tuple<int, int, int, int>> coalesced_keys;
  const auto& tiles = simulator.farms()[0].tiles;
  coverage.weed_tiles = static_cast<int>(std::count_if(
      tiles.begin(), tiles.end(), [](const auto& tile) {
        return tile.kind == fastkag::TileKind::WEED;
      }));
  for (const auto& transaction : open) {
    const auto key = std::make_tuple(
        transaction.key.tile.row, transaction.key.tile.column,
        transaction.desired_item, static_cast<int>(transaction.goal));
    if (!coalesced_keys.insert(key).second) {
      ++coverage.coalesced_duplicates;
      continue;
    }
    horizon::Objective objective;
    objective.id = transaction.id;
    objective.tile = transaction.key.tile;
    objective.deadline_turn = coverage.turns - 1;
    objective.value = 1;
    objective.critical = transaction.outstanding_source_actions > 0;
    const int index = objective.tile.row * simulator.config().board_size +
        objective.tile.column;
    if (index >= 0 && index < static_cast<int>(tiles.size())) {
      const auto& tile = tiles[static_cast<std::size_t>(index)];
      const auto desired = transaction.desired_item;
      if (transaction.goal == local::GoalKind::Structure) {
        if (tile.kind == fastkag::TileKind::WEED) {
          ++coverage.weed_objectives;
          objective.remaining_transitions.push_back(
              {local::Op::Dig, -1, 1, 0, 0});
        }
        if (tile.kind == fastkag::TileKind::WEED ||
            tile.kind == fastkag::TileKind::EMPTY)
          objective.remaining_transitions.push_back(
              {local::Op::Build, desired, 1, 0, 0});
      } else if (desired >= 0 && desired < fastkag::N_CROPS) {
        if (tile.kind == fastkag::TileKind::WEED) {
          ++coverage.weed_objectives;
          objective.remaining_transitions.push_back(
              {local::Op::Dig, -1, 1, 0, 0});
          objective.remaining_transitions.push_back(
              {local::Op::Plant, desired, 1, 0, 0});
        } else if (tile.kind == fastkag::TileKind::EMPTY) {
          objective.remaining_transitions.push_back(
              {local::Op::Plant, desired, 1, 0, 0});
        } else if (tile.kind == fastkag::TileKind::PLANT &&
                   static_cast<int>(tile.crop) == desired) {
          if (tile.yield_units > 0)
            objective.remaining_transitions.push_back(
                {local::Op::Harvest, desired, 1, 0, 0});
          else if (!tile.watered_today)
            objective.remaining_transitions.push_back(
                {local::Op::Water, desired, 1, 0, 0});
        }
      }
    }
    for (const auto& transition : objective.remaining_transitions) {
      ++coverage.objective_transition_actions;
      if (transition.op == local::Op::Plant && transition.item >= 0 &&
          transition.item < fastkag::N_CROPS && transition.quantity > 0) {
        ++coverage.plant_objectives;
        coverage.repair_seed_demand[
            static_cast<std::size_t>(transition.item)] += transition.quantity;
      }
    }
    suffix_sizes[objective.id] =
        static_cast<int>(objective.remaining_transitions.size());
    objectives.push_back(std::move(objective));
  }
  coverage.objectives = static_cast<int>(objectives.size());
  horizon::ResourceSnapshot resources;
  for (int crop = 0; crop < fastkag::N_CROPS; ++crop) {
    coverage.seed_snapshot[static_cast<std::size_t>(crop)] =
        simulator.privates()[0].seeds[static_cast<std::size_t>(crop)];
    resources.seeds[crop] =
        coverage.seed_snapshot[static_cast<std::size_t>(crop)];
  }
  const auto result = horizon::compile(actors, objectives, resources);
  // Exercise the production-facing default hard gate. Any unfinished raw
  // route makes this representative-day audit fail instead of being reported
  // as a schedulable candidate.
  const auto route_sequence =
      horizon::compile_route_sequence(actors, objectives, resources);
  const auto time_expanded =
      horizon::compile_time_expanded(actors, objectives, resources);
  coverage.assignments = static_cast<int>(result.assignments.size());
  coverage.unscheduled =
      static_cast<int>(result.unscheduled_objectives.size());
  std::map<std::uint64_t, int> assigned;
  for (const auto& assignment : result.assignments) {
    ++assigned[assignment.objective_id];
    if (assignment.action.op == local::Op::Plant) {
      if (assignment.action.item >= 0 &&
          assignment.action.item < fastkag::N_CROPS) {
        coverage.scheduled_seed_demand[
            static_cast<std::size_t>(assignment.action.item)] +=
            assignment.action.quantity;
        const auto crop = static_cast<std::size_t>(assignment.action.item);
        if (coverage.scheduled_seed_demand[crop] +
                coverage.raw_seed_reservation[crop] >
            coverage.seed_snapshot[crop])
          ++coverage.inventory_unproven_assignments;
      } else {
        ++coverage.inventory_unproven_assignments;
      }
    }
  }
  for (const auto& [id, count] : assigned)
    coverage.scheduled += count == suffix_sizes[id];
  coverage.move_slots_exact = result.move_slots_exact;
  for (std::size_t actor = 0; actor < actors.size(); ++actor)
    for (std::size_t turn = 0; turn < actors[actor].raw.size(); ++turn)
      if (!(actors[actor].raw[turn] == result.manifest[actor][turn]) &&
          (actors[actor].raw[turn].op == local::Op::Move ||
           result.manifest[actor][turn].op == local::Op::Move))
        ++coverage.move_edits;

  coverage.route_sequence_completed_objectives =
      route_sequence.completed_objectives;
  coverage.route_sequence_unscheduled_objectives = static_cast<int>(
      route_sequence.unscheduled_objectives.size());
  coverage.route_sequence_assignments =
      static_cast<int>(route_sequence.assignments.size());
  coverage.route_sequence_move_timing_deviation =
      route_sequence.move_timing_deviation;
  coverage.route_sequence_terminal_unexecuted_moves =
      route_sequence.terminal_unexecuted_moves;
  coverage.route_sequence_terminal_unexecuted_raw_actions =
      route_sequence.terminal_unexecuted_raw_actions;
  coverage.route_sequence_ordered_moves_exact =
      route_sequence.ordered_move_sequences_exact;
  coverage.route_sequence_move_slots_exact = route_sequence.move_slots_exact;
  coverage.route_sequence_positions_legal = true;
  for (const auto& assignment : route_sequence.assignments) {
    const auto objective = std::find_if(
        objectives.begin(), objectives.end(), [&](const auto& candidate) {
          return candidate.id == assignment.objective_id;
        });
    if (objective != objectives.end() &&
        assignment.turn > objective->deadline_turn)
      ++coverage.route_sequence_late_assignments;
  }
  for (std::size_t actor = 0; actor < actors.size(); ++actor) {
    auto expected = actors[actor].start;
    std::vector<std::pair<int, local::Action>> raw_moves;
    std::vector<std::pair<int, local::Action>> compiled_moves;
    for (int turn = 0; turn < coverage.turns; ++turn) {
      const auto index = static_cast<std::size_t>(turn);
      if (!(route_sequence.positions_before[actor][index] == expected) ||
          expected.row < 0 || expected.column < 0 ||
          expected.row >= simulator.config().board_size ||
          expected.column >= simulator.config().board_size)
        coverage.route_sequence_positions_legal = false;
      const auto raw = actors[actor].raw[index];
      const auto compiled = route_sequence.manifest[actor][index];
      if (raw.op == local::Op::Move) raw_moves.push_back({turn, raw});
      if (compiled.op == local::Op::Move) {
        compiled_moves.push_back({turn, compiled});
        switch (compiled.arg0) {
          case 0: --expected.row; break;
          case 1: ++expected.row; break;
          case 2: --expected.column; break;
          case 3: ++expected.column; break;
          default: coverage.route_sequence_positions_legal = false; break;
        }
      }
      if (!(raw == compiled) &&
          (raw.op == local::Op::Move || compiled.op == local::Op::Move))
        ++coverage.route_sequence_absolute_move_edits;
    }
    const auto matched = std::min(raw_moves.size(), compiled_moves.size());
    for (std::size_t index = 0; index < matched; ++index) {
      if (!(raw_moves[index].second == compiled_moves[index].second)) {
        coverage.route_sequence_positions_legal = false;
        continue;
      }
      const int delay = compiled_moves[index].first - raw_moves[index].first;
      if (delay < 0) coverage.route_sequence_positions_legal = false;
      coverage.route_sequence_derived_move_timing_deviation += delay;
      coverage.route_sequence_delayed_moves += delay > 0;
      coverage.route_sequence_max_move_delay =
          std::max(coverage.route_sequence_max_move_delay, delay);
      ++coverage.route_sequence_move_delay_histogram[delay];
    }
  }
  coverage.route_sequence_timing_accounting_exact =
      coverage.route_sequence_derived_move_timing_deviation ==
      coverage.route_sequence_move_timing_deviation;
  coverage.time_expanded_completed_objectives =
      time_expanded.completed_objectives;
  coverage.time_expanded_completed_value =
      time_expanded.completed_objective_value;
  coverage.time_expanded_unscheduled_objectives = static_cast<int>(
      time_expanded.unscheduled_objectives.size());
  coverage.time_expanded_timing_cost =
      time_expanded.timing_deviation_cost;
  coverage.time_expanded_total_cost = time_expanded.total_cost;
  coverage.time_expanded_terminal_raw =
      time_expanded.terminal_unexecuted_raw_actions;
  coverage.time_expanded_states = time_expanded.expanded_states;
  coverage.time_expanded_elapsed_us = time_expanded.planning_elapsed_us;
  coverage.time_expanded_exact = time_expanded.exact;
  coverage.time_expanded_fallback = time_expanded.fell_back_to_greedy;
  coverage.time_expanded_within_budget = time_expanded.within_time_budget;
  coverage.time_expanded_ordered_moves =
      time_expanded.ordered_move_sequences_exact;
  coverage.time_expanded_components = time_expanded.conflict_components;
  coverage.time_expanded_largest_component =
      time_expanded.largest_component_objectives;
  coverage.time_expanded_exact_objectives =
      time_expanded.exact_component_objectives;
  coverage.time_expanded_joint_objectives =
      time_expanded.joint_search_objectives;
  coverage.time_expanded_single_exact_components =
      time_expanded.single_actor_exact_components;
  coverage.time_expanded_joint_components =
      time_expanded.joint_search_components;
  coverage.time_expanded_fallback_components =
      time_expanded.fallback_components;
  coverage.time_expanded_component_histogram =
      time_expanded.component_objective_size_histogram;

  std::set<std::tuple<int, int, int>> objective_capacity;
  for (const auto& objective : objectives) {
    for (int turn = 0;
         turn < coverage.turns && turn <= objective.deadline_turn; ++turn) {
      bool has_capacity = false;
      for (std::size_t actor = 0; actor < actors.size(); ++actor) {
        if (result.positions_before[actor][static_cast<std::size_t>(turn)] ==
                objective.tile &&
            (actors[actor].raw[static_cast<std::size_t>(turn)].op ==
                 local::Op::Pass ||
             actors[actor].blocked_equivalent_capacity[
                 static_cast<std::size_t>(turn)])) {
          has_capacity = true;
          break;
        }
      }
      if (has_capacity)
        objective_capacity.insert(
            {objective.tile.row, objective.tile.column, turn});
    }
  }
  coverage.objective_tile_capacity_slots =
      static_cast<int>(objective_capacity.size());

  const std::set<std::uint64_t> unscheduled_ids(
      result.unscheduled_objectives.begin(),
      result.unscheduled_objectives.end());
  for (const auto& objective : objectives) {
    if (!unscheduled_ids.contains(objective.id)) continue;
    if (objective.remaining_transitions.empty()) {
      ++coverage.unscheduled_no_transition;
      continue;
    }
    bool seed_limited = false;
    std::array<int, fastkag::N_CROPS> objective_seed_demand{};
    for (const auto& transition : objective.remaining_transitions)
      if (transition.op == local::Op::Plant && transition.item >= 0 &&
          transition.item < fastkag::N_CROPS && transition.quantity > 0)
        objective_seed_demand[static_cast<std::size_t>(transition.item)] +=
            transition.quantity;
    for (int crop = 0; crop < fastkag::N_CROPS; ++crop) {
      const auto index = static_cast<std::size_t>(crop);
      const int post_raw = std::max(
          0, coverage.seed_snapshot[index] - coverage.raw_seed_reservation[index]);
      if (objective_seed_demand[index] > 0 &&
          coverage.scheduled_seed_demand[index] +
                  objective_seed_demand[index] >
              post_raw)
        seed_limited = true;
    }
    if (seed_limited) {
      ++coverage.unscheduled_seed_constraint;
      continue;
    }
    int candidate_turns = 0;
    for (int turn = 0;
         turn < coverage.turns && turn <= objective.deadline_turn; ++turn) {
      bool has_capacity = false;
      for (std::size_t actor = 0; actor < actors.size(); ++actor)
        if (result.positions_before[actor][static_cast<std::size_t>(turn)] ==
                objective.tile &&
            (actors[actor].raw[static_cast<std::size_t>(turn)].op ==
                 local::Op::Pass ||
             actors[actor].blocked_equivalent_capacity[
                 static_cast<std::size_t>(turn)])) {
          has_capacity = true;
          break;
        }
      candidate_turns += has_capacity;
    }
    if (candidate_turns == 0)
      ++coverage.unscheduled_no_tile_capacity;
    else if (candidate_turns <
             static_cast<int>(objective.remaining_transitions.size()))
      ++coverage.unscheduled_insufficient_suffix_capacity;
    else
      ++coverage.unscheduled_contention_or_priority;
  }
  return coverage;
}

int pending_purchase(const NativeAgentState& state, bool day_horizon_v2) {
  const auto debts = day_horizon_v2
      ? state.experimental_day_horizon_v2.purchase_ledger.debts()
      : state.experimental_event_local_purchase_ledger.debts();
  return static_cast<int>(std::count_if(
      debts.begin(), debts.end(),
      [](const auto& debt) { return debt.awaiting_receipt; }));
}

void certify_moves(const Tape& source, const Tape& emitted,
                   const std::vector<std::size_t>& active, Metrics& metrics,
                   int turns_per_day) {
  std::set<std::pair<int, int>> keys;
  const int steps = static_cast<int>(emitted.size());
  for (int step = 0; step < steps; ++step) {
    const std::size_t actors = active[static_cast<std::size_t>(step)];
    for (std::size_t actor = 0; actor < actors; ++actor) {
      const auto raw = unit(source[static_cast<std::size_t>(step)], actor);
      const auto final = unit(emitted[static_cast<std::size_t>(step)], actor);
      if (!same(raw, final) && (move(raw.op) || move(final.op)))
        ++metrics.move_overwrites;
      keys.insert({step / turns_per_day, static_cast<int>(actor)});
    }
  }
  for (const auto [day, actor] : keys) {
    std::vector<int> raw_moves;
    std::vector<int> final_moves;
    std::vector<int> raw_move_turns;
    std::vector<int> final_move_turns;
    const int begin = day * turns_per_day;
    const int end = std::min(steps, begin + turns_per_day);
    for (int step = begin; step < end; ++step) {
      if (actor >= static_cast<int>(active[static_cast<std::size_t>(step)]))
        continue;
      const auto raw = unit(source[static_cast<std::size_t>(step)], actor);
      const auto final = unit(emitted[static_cast<std::size_t>(step)], actor);
      if (move(raw.op)) {
        raw_moves.push_back(static_cast<int>(raw.op));
        raw_move_turns.push_back(step);
      }
      if (move(final.op)) {
        final_moves.push_back(static_cast<int>(final.op));
        final_move_turns.push_back(step);
      }
    }
    if (raw_moves != final_moves) {
      ++metrics.ordered_move_day_failures;
      continue;
    }
    for (std::size_t index = 0; index < raw_move_turns.size(); ++index) {
      const int delay = final_move_turns[index] - raw_move_turns[index];
      metrics.move_timing_deviation += delay;
      metrics.delayed_moves += delay > 0;
      metrics.maximum_move_delay = std::max(metrics.maximum_move_delay, delay);
    }
  }
}

Metrics run(const NativeTeammateExecutor& executor, const Tape& tape,
            std::uint64_t seed, int seat, int mask) {
  Simulator simulator({}, seed);
  std::array<NativeAgentState, 2> states;
  Metrics metrics;
  Tape emitted;
  std::vector<std::size_t> active;
  const auto repair = fastkag::native_repair_options_from_mask(mask);
  while (!simulator.done()) {
    std::array<PlayerAction, 2> actions;
    actions[seat] = executor.action_external(
        simulator, seat, 0, states[seat],
        fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, false,
        repair, &metrics.audit);
    actions[1 - seat] =
        executor.action_external(simulator, 1 - seat, 0, states[1 - seat]);
    active.push_back(simulator.farms()[seat].hands.size() + 1);
    emitted.push_back(actions[seat]);
    metrics.unit_failures +=
        fastkag::native_macro_unit_failures(simulator, seat, actions[seat]);
    const auto hands_before = simulator.farms()[seat].hands.size();
    simulator.step(actions);
    const auto hands_after = simulator.farms()[seat].hands.size();
    if (hands_after > hands_before)
      metrics.hires += static_cast<int>(hands_after - hands_before);
    metrics.market_failures +=
        fastkag::native_macro_market_failures(simulator, seat, actions[seat]);
  }
  metrics.own = simulator.farms()[seat].money;
  metrics.opponent = simulator.farms()[1 - seat].money;
  certify_moves(tape, emitted, active, metrics,
                simulator.config().turns_per_day);
  if (mask == 32) {
    const auto& runtime = states[seat].experimental_day_horizon_v2;
    metrics.open_plot_transactions =
        static_cast<int>(runtime.objectives.size());
    metrics.minimal_plot_objectives = metrics.open_plot_transactions;
    metrics.pending_unit_receipts = static_cast<int>(runtime.pending.size());
    // V2 actors are receipt leases, never plot identity. Any terminal pending
    // actor is therefore both an unfinished receipt and an identity lease.
    metrics.terminal_identity_leases =
        static_cast<int>(runtime.pending.size());
  } else if (states[seat].experimental_event_local_repair) {
    const auto open =
        states[seat].experimental_event_local_repair->open_transactions();
    metrics.open_plot_transactions = static_cast<int>(open.size());
    std::set<std::tuple<int, int, int, int>> objectives;
    for (const auto& transaction : open) {
      objectives.insert({transaction.key.tile.row,
                         transaction.key.tile.column,
                         transaction.desired_item,
                         static_cast<int>(transaction.goal)});
      if (transaction.outstanding_source_actions > 0) {
        ++metrics.critical_source_transactions;
        metrics.critical_source_units +=
            transaction.outstanding_source_actions;
      }
    }
    metrics.minimal_plot_objectives = static_cast<int>(objectives.size());
  }
  if (mask != 32)
    metrics.pending_unit_receipts = static_cast<int>(
        states[seat].experimental_event_local_pending.size());
  metrics.pending_purchase_receipts = pending_purchase(states[seat], mask == 32);
  return metrics;
}

Options parse(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&]() -> std::string {
      if (++index >= argc) throw std::invalid_argument("missing option value");
      return argv[index];
    };
    if (argument == "--seed-begin") options.seed_begin = std::stoull(next());
    else if (argument == "--seeds") options.seeds = std::stoi(next());
    else if (argument == "--coverage-step")
      options.coverage_step = std::stoi(next());
    else if (argument == "--treatment-mask")
      options.treatment_mask = std::stoi(next());
    else if (argument == "--output") options.output = next();
    else if (argument == "--tapes") options.tapes = next();
    else if (argument == "--library") options.library = next();
    else throw std::invalid_argument("unknown option: " + argument);
  }
  if (options.seeds <= 0) throw std::invalid_argument("seeds must be positive");
  if (options.coverage_step < 0 || options.coverage_step >= 719)
    throw std::invalid_argument("coverage-step outside G001 episode");
  if (options.treatment_mask != 8 && options.treatment_mask != 16 &&
      options.treatment_mask != 32)
    throw std::invalid_argument("treatment-mask must be 8, 16, or 32");
  return options;
}

struct Aggregate {
  int games{};
  double own{};
  double opponent{};
  long long hires{};
  long long unit_failures{};
  long long market_failures{};
  long long move_overwrites{};
  long long ordered_move_day_failures{};
  long long move_timing_deviation{};
  long long delayed_moves{};
  int maximum_move_delay{};
  long long day_horizon_plans{};
  long long day_horizon_commits{};
  long long day_horizon_recompiles{};
  long long day_horizon_fail_closed{};
  long long day_horizon_assignments{};
  long long day_horizon_terminal_raw{};
  long long open_plot_transactions{};
  long long minimal_plot_objectives{};
  long long critical_source_transactions{};
  long long critical_source_units{};
  long long pending_unit_receipts{};
  long long pending_purchase_receipts{};
  long long terminal_identity_leases{};
  long long day_horizon_v2_plans{};
  long long day_horizon_v2_replans{};
  long long day_horizon_v2_receipts_confirmed{};
  long long day_horizon_v2_receipts_failed{};
  long long day_horizon_v2_fail_closed{};
  long long day_horizon_v2_assignments{};
  long long day_horizon_v2_exact_plans{};
  long long day_horizon_v2_fallback_plans{};
  long long day_horizon_v2_budget_exhausted{};
  long long day_horizon_v2_objectives_completed{};
  long long day_horizon_v2_objectives_carried{};
  long long day_horizon_v2_maturity_waits{};
  long long day_horizon_v2_maturity_tokens_consumed{};
  long long day_horizon_v2_seed_unscheduled{};
  long long day_horizon_v2_seed_orders{};
  long long day_horizon_v2_seed_zero_fills{};
  long long day_horizon_v2_seed_fills{};
  long long day_horizon_v2_terminal_raw{};
  std::array<long long, 8> day_horizon_v2_fail_reasons{};
};

Aggregate aggregate(const std::vector<Row>& rows, int arm) {
  Aggregate result;
  for (const auto& row : rows) {
    const auto& value = row.arms[static_cast<std::size_t>(arm)];
    ++result.games;
    result.own += value.own;
    result.opponent += value.opponent;
    result.hires += value.hires;
    result.unit_failures += value.unit_failures;
    result.market_failures += value.market_failures;
    result.move_overwrites += value.move_overwrites;
    result.ordered_move_day_failures += value.ordered_move_day_failures;
    result.move_timing_deviation += value.move_timing_deviation;
    result.delayed_moves += value.delayed_moves;
    result.maximum_move_delay =
        std::max(result.maximum_move_delay, value.maximum_move_delay);
    result.day_horizon_plans += value.audit.day_horizon_plans;
    result.day_horizon_commits += value.audit.day_horizon_commits;
    result.day_horizon_recompiles += value.audit.day_horizon_recompiles;
    result.day_horizon_fail_closed += value.audit.day_horizon_fail_closed;
    result.day_horizon_assignments += value.audit.day_horizon_assignments;
    result.day_horizon_terminal_raw += value.audit.day_horizon_terminal_raw;
    result.open_plot_transactions += value.open_plot_transactions;
    result.minimal_plot_objectives += value.minimal_plot_objectives;
    result.critical_source_transactions += value.critical_source_transactions;
    result.critical_source_units += value.critical_source_units;
    result.pending_unit_receipts += value.pending_unit_receipts;
    result.pending_purchase_receipts += value.pending_purchase_receipts;
    result.terminal_identity_leases += value.terminal_identity_leases;
    result.day_horizon_v2_plans += value.audit.day_horizon_v2_plans;
    result.day_horizon_v2_replans += value.audit.day_horizon_v2_replans;
    result.day_horizon_v2_receipts_confirmed +=
        value.audit.day_horizon_v2_receipts_confirmed;
    result.day_horizon_v2_receipts_failed +=
        value.audit.day_horizon_v2_receipts_failed;
    result.day_horizon_v2_fail_closed +=
        value.audit.day_horizon_v2_fail_closed;
    result.day_horizon_v2_assignments +=
        value.audit.day_horizon_v2_assignments;
    result.day_horizon_v2_exact_plans +=
        value.audit.day_horizon_v2_exact_plans;
    result.day_horizon_v2_fallback_plans +=
        value.audit.day_horizon_v2_fallback_plans;
    result.day_horizon_v2_budget_exhausted +=
        value.audit.day_horizon_v2_budget_exhausted;
    result.day_horizon_v2_objectives_completed +=
        value.audit.day_horizon_v2_objectives_completed;
    result.day_horizon_v2_objectives_carried +=
        value.audit.day_horizon_v2_objectives_carried;
    result.day_horizon_v2_maturity_waits +=
        value.audit.day_horizon_v2_maturity_waits;
    result.day_horizon_v2_maturity_tokens_consumed +=
        value.audit.day_horizon_v2_maturity_tokens_consumed;
    result.day_horizon_v2_seed_unscheduled +=
        value.audit.day_horizon_v2_seed_unscheduled;
    result.day_horizon_v2_seed_orders +=
        value.audit.day_horizon_v2_seed_orders;
    result.day_horizon_v2_seed_zero_fills +=
        value.audit.day_horizon_v2_seed_zero_fills;
    result.day_horizon_v2_seed_fills +=
        value.audit.day_horizon_v2_seed_fills;
    result.day_horizon_v2_terminal_raw +=
        value.audit.day_horizon_v2_terminal_raw;
    for (std::size_t reason = 0;
         reason < result.day_horizon_v2_fail_reasons.size(); ++reason)
      result.day_horizon_v2_fail_reasons[reason] +=
          value.audit.day_horizon_v2_fail_reasons[reason];
  }
  return result;
}

std::vector<PairedSeed> pair_by_seed(const std::vector<Row>& rows) {
  std::map<std::uint64_t, PairedSeed> grouped;
  for (const auto& row : rows) {
    const auto& baseline = row.arms[0];
    const auto& treatment = row.arms[1];
    auto& pair = grouped[row.seed];
    pair.seed = row.seed;
    ++pair.games;
    pair.own += treatment.own - baseline.own;
    pair.margin += (treatment.own - treatment.opponent) -
        (baseline.own - baseline.opponent);
    pair.hires += treatment.hires - baseline.hires;
    pair.unit_failures +=
        treatment.unit_failures - baseline.unit_failures;
    pair.market_failures +=
        treatment.market_failures - baseline.market_failures;
    pair.move_timing_deviation +=
        treatment.move_timing_deviation - baseline.move_timing_deviation;
  }
  std::vector<PairedSeed> result;
  result.reserve(grouped.size());
  for (auto& [seed, pair] : grouped) {
    if (pair.games != 2)
      throw std::runtime_error("paired seed is missing one seat");
    const double games = pair.games;
    pair.own /= games;
    pair.margin /= games;
    pair.hires /= games;
    pair.unit_failures /= games;
    pair.market_failures /= games;
    pair.move_timing_deviation /= games;
    result.push_back(pair);
  }
  return result;
}

template <typename Projection>
Interval bootstrap_ci(const std::vector<PairedSeed>& pairs,
                      Projection projection, std::uint64_t salt) {
  constexpr int kResamples = 10000;
  if (pairs.empty()) return {};
  std::mt19937_64 generator(0xB1600325EEDULL ^ salt);
  std::uniform_int_distribution<std::size_t> choose(0, pairs.size() - 1);
  std::vector<double> estimates;
  estimates.reserve(kResamples);
  for (int sample = 0; sample < kResamples; ++sample) {
    double sum = 0.0;
    for (std::size_t draw = 0; draw < pairs.size(); ++draw)
      sum += projection(pairs[choose(generator)]);
    estimates.push_back(sum / pairs.size());
  }
  std::sort(estimates.begin(), estimates.end());
  // Deterministic paired-by-seed percentile interval. The seed is a cluster:
  // both seats always enter or leave a resample together.
  return {estimates[249], estimates[9749]};
}

void write_interval(std::ostream& output, Interval interval) {
  output << "{\"lower\":" << interval.lower
         << ",\"upper\":" << interval.upper << '}';
}

void write_aggregate(std::ostream& output, const Aggregate& value) {
  const double games = std::max(1, value.games);
  output << "{\"games\":" << value.games
         << ",\"own_money_mean\":" << value.own / games
         << ",\"opponent_money_mean\":" << value.opponent / games
         << ",\"margin_mean\":" << (value.own - value.opponent) / games
         << ",\"hires_mean\":" << value.hires / games
         << ",\"unit_failures_mean\":" << value.unit_failures / games
         << ",\"market_failures_mean\":" << value.market_failures / games
         << ",\"move_overwrites\":" << value.move_overwrites
         << ",\"ordered_move_day_failures\":"
         << value.ordered_move_day_failures
         << ",\"move_timing_deviation\":"
         << value.move_timing_deviation
         << ",\"delayed_moves\":" << value.delayed_moves
         << ",\"maximum_move_delay\":" << value.maximum_move_delay
         << ",\"day_horizon_plans\":" << value.day_horizon_plans
         << ",\"day_horizon_commits\":" << value.day_horizon_commits
         << ",\"day_horizon_recompiles\":"
         << value.day_horizon_recompiles
         << ",\"day_horizon_fail_closed\":"
         << value.day_horizon_fail_closed
         << ",\"day_horizon_assignments\":"
         << value.day_horizon_assignments
         << ",\"day_horizon_terminal_raw\":"
         << value.day_horizon_terminal_raw
         << ",\"planner_fallback_observable\":false"
         << ",\"day_horizon_v2_fallback_observable\":true"
         << ",\"open_plot_transactions_mean\":"
         << value.open_plot_transactions / games
         << ",\"minimal_plot_objectives_mean\":"
         << value.minimal_plot_objectives / games
         << ",\"critical_source_transactions_mean\":"
         << value.critical_source_transactions / games
         << ",\"critical_source_units_mean\":"
         << value.critical_source_units / games
         << ",\"terminal_pending_unit_receipts\":"
         << value.pending_unit_receipts
         << ",\"terminal_pending_purchase_receipts\":"
         << value.pending_purchase_receipts
         << ",\"terminal_identity_leases\":"
         << value.terminal_identity_leases
         << ",\"day_horizon_v2_plans\":" << value.day_horizon_v2_plans
         << ",\"day_horizon_v2_replans\":" << value.day_horizon_v2_replans
         << ",\"day_horizon_v2_receipts_confirmed\":"
         << value.day_horizon_v2_receipts_confirmed
         << ",\"day_horizon_v2_receipts_failed\":"
         << value.day_horizon_v2_receipts_failed
         << ",\"day_horizon_v2_fail_closed\":"
         << value.day_horizon_v2_fail_closed
         << ",\"day_horizon_v2_assignments\":"
         << value.day_horizon_v2_assignments
         << ",\"day_horizon_v2_exact_plans\":"
         << value.day_horizon_v2_exact_plans
         << ",\"day_horizon_v2_fallback_plans\":"
         << value.day_horizon_v2_fallback_plans
         << ",\"day_horizon_v2_budget_exhausted\":"
         << value.day_horizon_v2_budget_exhausted
         << ",\"day_horizon_v2_objectives_completed\":"
         << value.day_horizon_v2_objectives_completed
         << ",\"day_horizon_v2_objectives_carried\":"
         << value.day_horizon_v2_objectives_carried
         << ",\"day_horizon_v2_maturity_waits\":"
         << value.day_horizon_v2_maturity_waits
         << ",\"day_horizon_v2_maturity_tokens_consumed\":"
         << value.day_horizon_v2_maturity_tokens_consumed
         << ",\"day_horizon_v2_seed_unscheduled\":"
         << value.day_horizon_v2_seed_unscheduled
         << ",\"day_horizon_v2_seed_orders\":"
         << value.day_horizon_v2_seed_orders
         << ",\"day_horizon_v2_seed_zero_fills\":"
         << value.day_horizon_v2_seed_zero_fills
         << ",\"day_horizon_v2_seed_fills\":"
         << value.day_horizon_v2_seed_fills
         << ",\"day_horizon_v2_terminal_raw\":"
         << value.day_horizon_v2_terminal_raw
         << ",\"day_horizon_v2_fail_reasons\":[";
  for (std::size_t reason = 0;
       reason < value.day_horizon_v2_fail_reasons.size(); ++reason) {
    if (reason != 0) output << ',';
    output << value.day_horizon_v2_fail_reasons[reason];
  }
  output << "]}";
}

}  // namespace

int main(int argc, char** argv) try {
  const auto options = parse(argc, argv);
  const auto tape =
      g001::repair::load_route(options.tapes, options.library, "G001");
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  std::vector<Row> rows(static_cast<std::size_t>(options.seeds * 2));
#pragma omp parallel for schedule(dynamic, 1)
  for (int job = 0; job < options.seeds * 2; ++job) {
    Row row;
    row.seed = options.seed_begin + static_cast<std::uint64_t>(job / 2);
    row.seat = job % 2;
    row.arms[0] = run(executor, tape, row.seed, row.seat, 7);
    row.arms[1] = run(executor, tape, row.seed, row.seat,
                      options.treatment_mask);
    rows[static_cast<std::size_t>(job)] = std::move(row);
  }
  const auto mask7 = aggregate(rows, 0);
  const auto treatment = aggregate(rows, 1);
  const auto paired_seeds = pair_by_seed(rows);
  const auto coverage =
      representative_day_coverage(executor, tape, options.coverage_step);
  const auto parent = std::filesystem::path(options.output).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream output(options.output);
  if (!output) throw std::runtime_error("cannot open report");
  output << "{\n  \"schema\":\"native-g001-repair-normal-eval-v2\",\n"
         << "  \"causal\":true,\n  \"terminal_selection\":false,\n"
         << "  \"weed_spawn_chance\":0.005,\n"
         << "  \"seed_begin\":" << options.seed_begin << ",\n"
         << "  \"seeds\":" << options.seeds << ",\n"
         << "  \"treatment_mask\":" << options.treatment_mask << ",\n"
         << "  \"dual_seat_games\":" << rows.size() << ",\n"
         << "  \"arms\":{\n    \"mask7\":";
  write_aggregate(output, mask7);
  output << ",\n    \"treatment\":";
  write_aggregate(output, treatment);
  output << "\n  },\n  \"paired_delta\":{\"own_money_mean\":"
         << (treatment.own - mask7.own) / std::max(1, treatment.games)
         << ",\"margin_mean\":"
         << ((treatment.own - treatment.opponent) -
             (mask7.own - mask7.opponent)) /
                std::max(1, treatment.games)
         << ",\"hires_mean\":"
         << static_cast<double>(treatment.hires - mask7.hires) /
                std::max(1, treatment.games)
         << ",\"unit_failures_mean\":"
         << static_cast<double>(treatment.unit_failures -
                                mask7.unit_failures) /
                std::max(1, treatment.games)
         << ",\"market_failures_mean\":"
         << static_cast<double>(treatment.market_failures -
                                mask7.market_failures) /
                std::max(1, treatment.games)
         << ",\"move_timing_deviation\":"
         << treatment.move_timing_deviation - mask7.move_timing_deviation
         << ",\"terminal_route_raw\":"
         << (options.treatment_mask == 32
                 ? treatment.day_horizon_v2_terminal_raw
                 : treatment.day_horizon_terminal_raw)
         << "},\n  \"paired_by_seed\":[";
  for (std::size_t index = 0; index < paired_seeds.size(); ++index) {
    if (index != 0) output << ',';
    const auto& pair = paired_seeds[index];
    output << "{\"seed\":" << pair.seed
           << ",\"dual_seat_games\":" << pair.games
           << ",\"own_money_mean\":" << pair.own
           << ",\"margin_mean\":" << pair.margin
           << ",\"hires_mean\":" << pair.hires
           << ",\"unit_failures_mean\":" << pair.unit_failures
           << ",\"market_failures_mean\":" << pair.market_failures
           << ",\"move_timing_deviation_mean\":"
           << pair.move_timing_deviation << '}';
  }
  output << "],\n  \"paired_bootstrap_ci95\":{"
         << "\"method\":\"seed-cluster-percentile\","
         << "\"resamples\":10000,\"rng_seed\":12189120487149,"
         << "\"own_money_mean\":";
  write_interval(output, bootstrap_ci(
      paired_seeds, [](const PairedSeed& pair) { return pair.own; }, 1));
  output << ",\"margin_mean\":";
  write_interval(output, bootstrap_ci(
      paired_seeds, [](const PairedSeed& pair) { return pair.margin; }, 2));
  output << ",\"hires_mean\":";
  write_interval(output, bootstrap_ci(
      paired_seeds, [](const PairedSeed& pair) { return pair.hires; }, 3));
  output << ",\"unit_failures_mean\":";
  write_interval(output, bootstrap_ci(
      paired_seeds,
      [](const PairedSeed& pair) { return pair.unit_failures; }, 4));
  output << ",\"market_failures_mean\":";
  write_interval(output, bootstrap_ci(
      paired_seeds,
      [](const PairedSeed& pair) { return pair.market_failures; }, 5));
  output << ",\"move_timing_deviation_mean\":";
  write_interval(output, bootstrap_ci(
      paired_seeds,
      [](const PairedSeed& pair) { return pair.move_timing_deviation; }, 6));
  output << "},\n  \"offline_representative_day_coverage\":{"
         << "\"simulation_seed\":" << coverage.simulation_seed << ','
         << "\"weed_spawn_chance\":0.005,\"step\":"
         << coverage.step << ",\"day\":" << coverage.day
         << ",\"start_hour\":" << coverage.start_hour
         << ",\"actors\":" << coverage.actors
         << ",\"turns\":" << coverage.turns
         << ",\"open_transactions\":" << coverage.open_transactions
         << ",\"objectives\":" << coverage.objectives
         << ",\"coalesced_duplicates\":"
         << coverage.coalesced_duplicates
         << ",\"weed_tiles\":" << coverage.weed_tiles
         << ",\"weed_objectives\":" << coverage.weed_objectives
         << ",\"plant_objectives\":" << coverage.plant_objectives
         << ",\"objective_transition_actions\":"
         << coverage.objective_transition_actions
         << ",\"scheduled\":" << coverage.scheduled
         << ",\"unscheduled\":" << coverage.unscheduled
         << ",\"coverage_ratio\":"
         << (coverage.objectives == 0 ? 0.0 :
             static_cast<double>(coverage.scheduled) / coverage.objectives)
         << ",\"assignments\":" << coverage.assignments
         << ",\"planner_absolute_move_edits\":" << coverage.move_edits
         << ",\"local_overlay_absolute_move_edits\":"
         << coverage.local_overlay_absolute_move_edits
         << ",\"raw_move_slots\":" << coverage.raw_move_slots
         << ",\"raw_pass_slots\":" << coverage.raw_pass_slots
         << ",\"objective_tile_capacity_slots\":"
         << coverage.objective_tile_capacity_slots
         << ",\"move_slots_exact\":"
         << (coverage.move_slots_exact ? "true" : "false")
         << ",\"inventory_unproven_assignments\":"
         << coverage.inventory_unproven_assignments
         << ",\"unscheduled_reasons\":{\"no_transition\":"
         << coverage.unscheduled_no_transition
         << ",\"seed_constraint\":"
         << coverage.unscheduled_seed_constraint
         << ",\"no_tile_capacity\":"
         << coverage.unscheduled_no_tile_capacity
         << ",\"insufficient_suffix_capacity\":"
         << coverage.unscheduled_insufficient_suffix_capacity
         << ",\"contention_or_priority\":"
         << coverage.unscheduled_contention_or_priority << "},"
         << "\"seed_snapshot\":";
  write_int_array(output, coverage.seed_snapshot);
  output << ",\"raw_seed_reservation\":";
  write_int_array(output, coverage.raw_seed_reservation);
  output << ",\"repair_seed_demand\":";
  write_int_array(output, coverage.repair_seed_demand);
  output << ",\"scheduled_seed_demand\":";
  write_int_array(output, coverage.scheduled_seed_demand);
  output << ",\"start_positions_row_column\":";
  write_positions(output, coverage.start_positions);
  output << ",\"route_sequence\":{\"completed_objectives\":"
         << coverage.route_sequence_completed_objectives
         << ",\"unscheduled_objectives\":"
         << coverage.route_sequence_unscheduled_objectives
         << ",\"coverage_ratio\":"
         << (coverage.objectives == 0 ? 0.0 :
             static_cast<double>(
                 coverage.route_sequence_completed_objectives) /
                 coverage.objectives)
         << ",\"assignments\":" << coverage.route_sequence_assignments
         << ",\"ordered_move_sequences_exact\":"
         << (coverage.route_sequence_ordered_moves_exact ? "true" : "false")
         << ",\"move_slots_exact\":"
         << (coverage.route_sequence_move_slots_exact ? "true" : "false")
         << ",\"positions_legal\":"
         << (coverage.route_sequence_positions_legal ? "true" : "false")
         << ",\"timing_accounting_exact\":"
         << (coverage.route_sequence_timing_accounting_exact ? "true" :
                                                             "false")
         << ",\"move_timing_deviation\":"
         << coverage.route_sequence_move_timing_deviation
         << ",\"derived_move_timing_deviation\":"
         << coverage.route_sequence_derived_move_timing_deviation
         << ",\"delayed_moves\":"
         << coverage.route_sequence_delayed_moves
         << ",\"max_move_delay\":"
         << coverage.route_sequence_max_move_delay
         << ",\"absolute_move_edits\":"
         << coverage.route_sequence_absolute_move_edits
         << ",\"terminal_unexecuted_moves\":"
         << coverage.route_sequence_terminal_unexecuted_moves
         << ",\"terminal_unexecuted_raw_actions\":"
         << coverage.route_sequence_terminal_unexecuted_raw_actions
         << ",\"late_assignments\":"
         << coverage.route_sequence_late_assignments
         << ",\"move_delay_histogram\":";
  write_int_map(output, coverage.route_sequence_move_delay_histogram);
  output << "},\"time_expanded\":{\"completed_objectives\":"
         << coverage.time_expanded_completed_objectives
         << ",\"completed_value\":"
         << coverage.time_expanded_completed_value
         << ",\"unscheduled_objectives\":"
         << coverage.time_expanded_unscheduled_objectives
         << ",\"timing_cost\":" << coverage.time_expanded_timing_cost
         << ",\"total_cost\":" << coverage.time_expanded_total_cost
         << ",\"terminal_unexecuted_raw_actions\":"
         << coverage.time_expanded_terminal_raw
         << ",\"expanded_states\":" << coverage.time_expanded_states
         << ",\"planning_elapsed_us\":"
         << coverage.time_expanded_elapsed_us
         << ",\"exact\":"
         << (coverage.time_expanded_exact ? "true" : "false")
         << ",\"fell_back_to_greedy\":"
         << (coverage.time_expanded_fallback ? "true" : "false")
         << ",\"within_1000ms_budget\":"
         << (coverage.time_expanded_within_budget ? "true" : "false")
         << ",\"ordered_move_sequences_exact\":"
         << (coverage.time_expanded_ordered_moves ? "true" : "false")
         << ",\"conflict_components\":"
         << coverage.time_expanded_components
         << ",\"largest_component_objectives\":"
         << coverage.time_expanded_largest_component
         << ",\"exact_component_objectives\":"
         << coverage.time_expanded_exact_objectives
         << ",\"exact_objective_ratio\":"
         << (coverage.objectives == 0 ? 0.0 :
             static_cast<double>(coverage.time_expanded_exact_objectives) /
                 coverage.objectives)
         << ",\"joint_search_objectives\":"
         << coverage.time_expanded_joint_objectives
         << ",\"joint_search_objective_ratio\":"
         << (coverage.objectives == 0 ? 0.0 :
             static_cast<double>(coverage.time_expanded_joint_objectives) /
                 coverage.objectives)
         << ",\"single_actor_exact_components\":"
         << coverage.time_expanded_single_exact_components
         << ",\"joint_search_components\":"
         << coverage.time_expanded_joint_components
         << ",\"fallback_components\":"
         << coverage.time_expanded_fallback_components
         << ",\"component_objective_size_histogram\":";
  write_int_map(output, coverage.time_expanded_component_histogram);
  output << '}';
  output << "}\n}\n";
  std::cout << "native_g001_bit8_normal_eval games=" << rows.size()
            << " output=" << options.output << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "native_g001_bit8_normal_eval: " << error.what() << '\n';
  return 2;
}
