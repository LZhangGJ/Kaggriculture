#include "native_teammate.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fastkag::Action;
using fastkag::Config;
using fastkag::Item;
using fastkag::NativeAgentState;
using fastkag::NativeRepairAudit;
using fastkag::NativeRepairOptions;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

NativeRepairOptions enabled() {
  NativeRepairOptions options;
  options.rolling_route_skeleton_v3 = true;
  return options;
}

NativeTeammateExecutor executor_for(std::vector<PlayerAction> tape) {
  NativeTapeLibrary library;
  library.routes.push_back(std::move(tape));
  return NativeTeammateExecutor(std::move(library));
}

PlayerAction act(NativeTeammateExecutor& executor, Simulator& simulator,
                 NativeAgentState& state, NativeRepairAudit& audit) {
  return executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
}

NativeAgentState::RouteSkeletonActor actor_state(
    int actor, std::uint64_t generation,
    std::initializer_list<NativeAgentState::RouteSkeletonMove> moves) {
  NativeAgentState::RouteSkeletonActor result;
  result.actor = actor;
  result.generation = generation;
  result.moves.assign(moves);
  return result;
}

void duplicate_direction_is_matched_by_exact_source_id() {
  Config config;
  config.episode_steps = 24;
  config.weed_spawn_chance = 0.0;
  Simulator simulator(config, 0x640001ULL);
  simulator.step({});
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[2].units[0] = {Op::EAST, Item::NONE, 1};
  tape[3].units[0] = {Op::EAST, Item::NONE, 1};
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  state.experimental_route_skeleton_v3.day = 0;
  state.experimental_route_skeleton_v3.actors.push_back(
      actor_state(0, 1, {{3, {Op::EAST, Item::NONE, 1}}}));
  NativeRepairAudit audit;
  const auto output = act(executor, simulator, state, audit);
  const auto& moves = state.experimental_route_skeleton_v3.actors[0].moves;
  check(output.units[0].op == Op::PASS && moves.size() == 2 &&
            moves[0].source_step == 2 && moves[1].source_step == 3 &&
            audit.route_skeleton_v3_lcs_kept == 1 &&
            audit.route_skeleton_v3_fail_closed == 0,
        "duplicate EAST sources were coalesced or matched ambiguously");
}

void same_direction_at_different_source_fails_closed() {
  Config config;
  config.episode_steps = 24;
  config.weed_spawn_chance = 0.0;
  Simulator simulator(config, 0x640002ULL);
  simulator.step({});
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[3].units[0] = {Op::EAST, Item::NONE, 1};
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  state.experimental_route_skeleton_v3.day = 0;
  state.experimental_route_skeleton_v3.actors.push_back(
      actor_state(0, 1, {{2, {Op::EAST, Item::NONE, 1}}}));
  NativeRepairAudit audit;
  const auto output = act(executor, simulator, state, audit);
  check(output.units[0].op == Op::PASS &&
            audit.route_skeleton_v3_fail_closed == 1 &&
            audit.route_skeleton_v3_fail_reasons[1] == 1 &&
            state.experimental_route_skeleton_v3.actors[0].moves[0].source_step ==
                2,
        "same-direction different-source MOVE was incorrectly accepted");
}

void dynamic_add_is_rebased_but_dynamic_delete_fails_closed() {
  Config config;
  config.episode_steps = 24;
  config.weed_spawn_chance = 0.0;
  Simulator added_sim(config, 0x640003ULL);
  added_sim.step({});
  std::vector<PlayerAction> added_tape(config.episode_steps);
  for (auto& action : added_tape) action.units.resize(1);
  added_tape[2].units[0] = {Op::WEST, Item::NONE, 1};
  added_tape[3].units[0] = {Op::EAST, Item::NONE, 1};
  auto added_executor = executor_for(std::move(added_tape));
  NativeAgentState added_state;
  added_state.experimental_route_skeleton_v3.day = 0;
  added_state.experimental_route_skeleton_v3.actors.push_back(
      actor_state(0, 1, {{3, {Op::EAST, Item::NONE, 1}}}));
  NativeRepairAudit added_audit;
  static_cast<void>(act(added_executor, added_sim, added_state, added_audit));
  const auto& added = added_state.experimental_route_skeleton_v3.actors[0].moves;
  check(added.size() == 2 && added[0].source_step == 2 &&
            added[0].action.op == Op::WEST && added[1].source_step == 3 &&
            added_audit.route_skeleton_v3_fail_closed == 0,
        "new baseline MOVE was not inserted before an exact old source");

  Simulator deleted_sim(config, 0x640004ULL);
  deleted_sim.step({});
  std::vector<PlayerAction> deleted_tape(config.episode_steps);
  for (auto& action : deleted_tape) action.units.resize(1);
  auto deleted_executor = executor_for(std::move(deleted_tape));
  NativeAgentState deleted_state;
  deleted_state.experimental_route_skeleton_v3.day = 0;
  deleted_state.experimental_route_skeleton_v3.actors.push_back(
      actor_state(0, 1, {{2, {Op::EAST, Item::NONE, 1}}}));
  NativeRepairAudit deleted_audit;
  static_cast<void>(
      act(deleted_executor, deleted_sim, deleted_state, deleted_audit));
  check(deleted_audit.route_skeleton_v3_fail_closed == 1 &&
            deleted_audit.route_skeleton_v3_fail_reasons[1] == 1,
        "deleted baseline MOVE was silently retained or remapped");
}

void missed_move_is_prepended_and_position_is_conserved() {
  Config config;
  config.episode_steps = 24;
  config.weed_spawn_chance = 0.0;
  Simulator simulator(config, 0x640005ULL);
  simulator.step({});
  simulator.step({});
  const auto start = simulator.farms()[0].farmer;
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[3].units[0] = {Op::WEST, Item::NONE, 1};
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  state.experimental_route_skeleton_v3.day = 0;
  state.experimental_route_skeleton_v3.actors.push_back(
      actor_state(0, 1, {{0, {Op::EAST, Item::NONE, 1}}}));
  state.experimental_route_skeleton_v3.actors[0].moves[0].deferred_by_planner =
      true;
  NativeRepairAudit audit;
  std::array<PlayerAction, 2> first{};
  first[0] = act(executor, simulator, state, audit);
  check(first[0].units[0].op == Op::EAST,
        "overdue MOVE was not prepended to current route");
  simulator.step(first);
  std::array<PlayerAction, 2> second{};
  second[0] = act(executor, simulator, state, audit);
  check(second[0].units[0].op == Op::WEST,
        "future MOVE did not follow prepended overdue MOVE");
  simulator.step(second);
  check(simulator.farms()[0].farmer.x == start.x &&
            simulator.farms()[0].farmer.y == start.y &&
            audit.route_skeleton_v3_moves_emitted == 2 &&
            audit.route_skeleton_v3_terminal_moves == 0,
        "overdue prepend changed final position or stranded a MOVE");
}

void day_end_insufficient_capacity_retains_unemitted_overdue_moves() {
  Config config;
  config.episode_steps = 48;
  config.weed_spawn_chance = 0.0;
  Simulator simulator(config, 0x640006ULL);
  while (simulator.step_count() < 23) simulator.step({});
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  state.experimental_route_skeleton_v3.day = 0;
  state.experimental_route_skeleton_v3.actors.push_back(actor_state(
      0, 1, {{20, {Op::EAST, Item::NONE, 1}},
             {21, {Op::WEST, Item::NONE, 1}}}));
  for (auto& move : state.experimental_route_skeleton_v3.actors[0].moves)
    move.deferred_by_planner = true;
  NativeRepairAudit audit;
  const auto output = act(executor, simulator, state, audit);
  check(output.units[0].op == Op::PASS &&
            audit.route_skeleton_v3_fail_closed == 1 &&
            audit.route_skeleton_v3_fail_reasons[3] == 1 &&
            audit.route_skeleton_v3_moves_emitted == 0 &&
            state.experimental_route_skeleton_v3.actors[0].cursor == 0,
        "day-end capacity partially emitted an incomplete overdue route");
}

void despawned_hand_slot_is_not_reused_as_old_generation() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 0.0;
  Simulator simulator(config, 0x640007ULL);
  std::vector<PlayerAction> tape(config.episode_steps);
  tape[0].units.resize(1);
  tape[0].market.push_back({Op::HIRE, Item::NONE, 1});
  for (int step = 1; step < 24; ++step)
    tape[static_cast<std::size_t>(step)].units.resize(2);
  tape[23].units[1] = {Op::EAST, Item::NONE, 1};
  tape[24].units.resize(1);
  tape[24].market.push_back({Op::HIRE, Item::NONE, 1});
  for (int step = 25; step < config.episode_steps; ++step)
    tape[static_cast<std::size_t>(step)].units.resize(2);
  tape[25].units[1] = {Op::WEST, Item::NONE, 1};
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;
  while (simulator.step_count() <= 25) {
    std::array<PlayerAction, 2> actions{};
    actions[0] = act(executor, simulator, state, audit);
    if (simulator.step_count() == 25)
      check(actions[0].units.size() == 2 &&
                actions[0].units[1].op == Op::WEST &&
                state.experimental_route_skeleton_v3.actors[1].generation ==
                    ((std::uint64_t{2} << 32U) | 2U),
            "new-day actor slot reused old generation/MOVE debt");
    simulator.step(actions);
  }
}

void unsupported_actor_rollback_does_not_pollute_other_actor() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0x640008ULL);
  while (simulator.step_count() < 24) simulator.step({});
  std::array<PlayerAction, 2> hire{};
  hire[0].market.push_back({Op::HIRE, Item::NONE, 1});
  simulator.step(hire);
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(2);
  tape[25].units[0] = {Op::HARVEST, Item::WHEAT, 1};
  tape[25].units[1] = {Op::FEED, Item::WHEAT, 1};
  tape[26].units[0] = {Op::EAST, Item::NONE, 1};
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;
  std::array<PlayerAction, 2> rejected{};
  rejected[0] = act(executor, simulator, state, audit);
  check(rejected[0].units[0].op == Op::HARVEST &&
            rejected[0].units[1].op == Op::FEED &&
            state.experimental_route_skeleton_v3.objectives.empty(),
        "reason2 rejection leaked another actor's crop objective");
  simulator.step(rejected);
  const auto next = act(executor, simulator, state, audit);
  check(next.units[0].op == Op::EAST &&
            audit.route_skeleton_v3_fail_reasons[2] == 1,
        "reason2 rollback later overwrote another actor MOVE");
}

}  // namespace

int main() try {
  duplicate_direction_is_matched_by_exact_source_id();
  same_direction_at_different_source_fails_closed();
  dynamic_add_is_rebased_but_dynamic_delete_fails_closed();
  missed_move_is_prepended_and_position_is_conserved();
  day_end_insufficient_capacity_retains_unemitted_overdue_moves();
  despawned_hand_slot_is_not_reused_as_old_generation();
  unsupported_actor_rollback_does_not_pollute_other_actor();
  std::cout << "route skeleton v3 audit: 7 adversarial fixtures passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "route skeleton v3 audit failure: " << error.what() << '\n';
  return 1;
}
