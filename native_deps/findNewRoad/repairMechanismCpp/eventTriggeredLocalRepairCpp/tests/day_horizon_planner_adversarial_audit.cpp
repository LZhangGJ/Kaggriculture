#include "day_horizon_planner.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace g001::day_horizon_repair;
using g001::event_local_repair::Op;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Action pass() { return {Op::Pass, -1, 1, 0, 0}; }
Action other(int tag) { return {Op::Other, -1, 1, tag, 0}; }
Action move(int direction) { return {Op::Move, -1, 1, direction, 0}; }
Action dig() { return {Op::Dig, -1, 1, 0, 0}; }
Action plant(int crop) { return {Op::Plant, crop, 1, 0, 0}; }
Action water(int crop) { return {Op::Water, crop, 1, 0, 0}; }
Action harvest(int crop) { return {Op::Harvest, crop, 1, 0, 0}; }

void raw_move_and_position_timeline_are_exact() {
  ActorPlan actor{7, {3, 3},
                  {move(3), pass(), move(1), pass()},
                  {false, false, false, false}};
  const auto result = compile(
      {actor}, {{1, {3, 4}, {water(0)}, 1, 1, true},
                {2, {4, 4}, {harvest(0)}, 3, 1, true}});
  require(result.move_slots_exact, "raw MOVE exact flag failed");
  require(result.manifest[0][0] == move(3) &&
              result.manifest[0][2] == move(1),
          "raw MOVE byte changed");
  require(result.positions_before[0] ==
              std::vector<Position>{{3, 3}, {3, 4}, {3, 4}, {4, 4}},
          "raw position timeline changed");
  require(result.manifest[0][1] == water(0) &&
              result.manifest[0][3] == harvest(0),
          "objective did not use reached-tile capacity");
}

void suffix_is_atomic_and_capacity_shortage_is_unscheduled() {
  ActorPlan actor{0, {0, 0}, {pass(), other(1), pass()},
                  {false, false, false}};
  const auto result = compile(
      {actor}, {{10, {0, 0}, {dig(), plant(0), water(0)}, 2, 9, true}});
  require(result.assignments.empty(), "partial objective suffix escaped");
  require(result.manifest[0] == actor.raw,
          "failed suffix mutated the manifest");
  require(result.unscheduled_objectives == std::vector<std::uint64_t>{10},
          "capacity-short objective not reported unscheduled");
}

void same_tile_is_serial_and_priority_is_lexicographic() {
  ActorPlan left{4, {2, 2}, {pass(), pass(), pass()},
                 {false, false, false}};
  ActorPlan right{9, {2, 2}, {pass(), pass(), pass()},
                  {false, false, false}};
  const auto result = compile(
      {left, right},
      {{20, {2, 2}, {water(0)}, 2, 100, false},
       {21, {2, 2}, {harvest(0)}, 2, 1, true},
       {22, {2, 2}, {dig()}, 1, 5, true},
       {23, {2, 2}, {plant(0)}, 1, 4, true}},
      ResourceSnapshot{{{0, 1}}});
  require(result.assignments.size() == 3,
          "unexpected contested capacity cardinality");
  require(result.assignments[0].objective_id == 22 &&
              result.assignments[1].objective_id == 23 &&
              result.assignments[2].objective_id == 21,
          "critical/deadline/value ordering failed");
  for (std::size_t left_index = 0; left_index < result.assignments.size();
       ++left_index)
    for (std::size_t right_index = left_index + 1;
         right_index < result.assignments.size(); ++right_index)
      require(result.assignments[left_index].turn !=
                  result.assignments[right_index].turn,
              "same tile used twice on one turn");
  require(result.unscheduled_objectives == std::vector<std::uint64_t>{20},
          "lowest-priority objective should be unscheduled");
}

void worker_not_at_tile_cannot_act() {
  ActorPlan actor{0, {0, 0}, {pass(), pass()}, {false, false}};
  const auto result =
      compile({actor}, {{30, {9, 9}, {water(0)}, 1, 1, true}});
  require(result.assignments.empty() &&
              result.unscheduled_objectives ==
                  std::vector<std::uint64_t>{30},
          "planner acted while worker was not on objective tile");
}

bool objective_move_corrupts_future_position_witness() {
  ActorPlan actor{0, {5, 5}, {pass(), pass()}, {false, false}};
  const auto result = compile(
      {actor}, {{40, {5, 5}, {move(3), water(0)}, 1, 1, true}});
  return result.move_slots_exact && result.assignments.size() == 2 &&
      result.manifest[0][0] == move(3) &&
      result.manifest[0][1] == water(0) &&
      result.positions_before[0][1] == Position{5, 5};
}

bool future_inventory_is_unchecked_witness() {
  // No seed snapshot is an explicit zero-resource proof.
  ActorPlan actor{0, {1, 1}, {pass()}, {false}};
  const auto result =
      compile({actor}, {{50, {1, 1}, {plant(4)}, 0, 1, true}});
  return result.assignments.size() == 1 &&
      result.manifest[0][0] == plant(4);
}

bool route_sequence_completes_after_deadline_witness() {
  ActorPlan first{0, {2, 2}, {pass(), pass(), pass(), pass()},
                  {false, false, false, false}};
  ActorPlan second{1, {2, 2},
                   {pass(), move(3), move(2), pass()},
                   {false, false, false, false}};
  // Critical objective 60 wins actor 0. Actor 1 then claims objective 61,
  // but same-tile serialization stalls it through its deadline (turn 0).
  const auto result = compile_route_sequence(
      {first, second},
      {{60, {2, 2}, {dig(), plant(0)}, 1, 1, true},
       {61, {2, 2}, {water(0)}, 0, 100, false}},
      ResourceSnapshot{{{0, 1}}}, RouteSequenceConfig{1, 1000, false});
  const auto late = std::find_if(
      result.assignments.begin(), result.assignments.end(),
      [](const Assignment& assignment) {
        return assignment.objective_id == 61 && assignment.turn > 0;
      });
  return late != result.assignments.end() &&
      result.completed_objectives == 2 &&
      result.terminal_unexecuted_moves == 1 &&
      !result.ordered_move_sequences_exact;
}

bool raw_tile_effect_is_not_serialized_witness() {
  ActorPlan producer{0, {4, 4}, {harvest(0)}, {false}};
  ActorPlan repairer{1, {4, 4}, {pass()}, {false}};
  const Objective objective{70, {4, 4}, {water(0)}, 0, 1, true};
  const auto fixed = compile({producer, repairer}, {objective});
  const auto sequence =
      compile_route_sequence({producer, repairer}, {objective});
  return fixed.manifest[0][0] == harvest(0) &&
      fixed.manifest[1][0] == water(0) &&
      sequence.manifest[0][0] == harvest(0) &&
      sequence.manifest[1][0] == water(0);
}

bool route_sequence_partial_suffix_witness() {
  ActorPlan repairer{0, {6, 6}, {pass(), pass(), pass()},
                     {false, false, false}};
  ActorPlan producer{1, {6, 6},
                     {pass(), harvest(0), water(0)},
                     {false, false, false}};
  const auto result = compile_route_sequence(
      {repairer, producer},
      {{80, {6, 6}, {dig(), plant(0)}, 1, 10, true}},
      ResourceSnapshot{{{0, 1}}}, RouteSequenceConfig{1, 1000, false});
  return result.assignments.size() == 1 &&
      result.assignments[0].objective_id == 80 &&
      result.assignments[0].transition_index == 0 &&
      result.unscheduled_objectives == std::vector<std::uint64_t>{80};
}

bool time_expanded_shared_seed_component_witness() {
  ActorPlan left{0, {0, 0}, {pass()}, {false}};
  ActorPlan right{1, {5, 5}, {pass()}, {false}};
  const std::vector<Objective> objectives{
      {90, {0, 0}, {plant(0)}, 0, 6, true},
      {91, {0, 0}, {plant(0)}, 0, 6, true},
      {92, {5, 5}, {plant(0)}, 0, 10, false}};
  const auto result = compile_time_expanded(
      {left, right}, objectives, ResourceSnapshot{{{0, 1}}});
  // A globally exact solver spends the only seed on objective 92 (value 10).
  // If same-seed objectives do not share a conflict component, component 90/91
  // (aggregate value 12) runs first and spends it on only one value-6 item.
  return result.exact && result.completed_objective_value == 6;
}

bool maturity_proof_replay_witness() {
  ActorPlan actor{0, {3, 3}, {pass()}, {false}};
  const Objective objective{100, {3, 3}, {harvest(0)}, 0, 8, true};
  const HarvestProofToken token{77, 100, 0};
  const ObservationReadiness proof{77, {{{100, 0}}}, {token}, {}};
  const auto first = compile_time_expanded({actor}, {objective}, {}, proof);
  const ObservationReadiness replay_proof{
      77, {{{100, 0}}}, {token}, first.consumed_harvest_legal};
  const auto replay =
      compile_time_expanded({actor}, {objective}, {}, replay_proof);
  return first.completed_objectives == 1 &&
      replay.completed_objectives == 1;
}

bool maturity_duplicate_objective_id_witness() {
  ActorPlan actor{0, {3, 3}, {pass(), pass()}, {false, false}};
  const std::vector<Objective> objectives{
      {110, {3, 3}, {harvest(0)}, 1, 8, true},
      {110, {3, 3}, {harvest(0)}, 1, 7, true}};
  const HarvestProofToken token{88, 110, 0};
  try {
    static_cast<void>(compile_time_expanded(
        {actor}, objectives, {},
        ObservationReadiness{88, {{{110, 0}}}, {token}, {}}));
  } catch (const std::invalid_argument&) {
    return false;
  }
  return true;
}

Position moved(Position position, const Action& action) {
  if (action.op != Op::Move) return position;
  if (action.arg0 == 0) --position.row;
  if (action.arg0 == 1) ++position.row;
  if (action.arg0 == 2) --position.column;
  if (action.arg0 == 3) ++position.column;
  return position;
}

int brute_single_actor_cost(const ActorPlan& actor,
                            const std::vector<Objective>& objectives,
                            const ResourceSnapshot& resources) {
  struct Raw { Action action; int original_turn; };
  std::vector<Raw> raw;
  for (int turn = 0; turn < static_cast<int>(actor.raw.size()); ++turn)
    if (actor.raw[static_cast<std::size_t>(turn)].op != Op::Pass)
      raw.push_back({actor.raw[static_cast<std::size_t>(turn)], turn});
  constexpr int infinity = std::numeric_limits<int>::max() / 8;
  std::function<int(int, std::size_t, Position, std::uint64_t,
                    std::map<int, int>)> solve =
      [&](int turn, std::size_t cursor, Position position,
          std::uint64_t mask, std::map<int, int> seeds) -> int {
    if (turn == static_cast<int>(actor.raw.size())) {
      if (cursor != raw.size()) return infinity;
      int unfinished = 0;
      for (std::size_t index = 0; index < objectives.size(); ++index)
        if ((mask & (std::uint64_t{1} << index)) == 0)
          unfinished += std::max(0, objectives[index].value);
      return unfinished;
    }
    int best = solve(turn + 1, cursor, position, mask, seeds);
    if (cursor < raw.size() && raw[cursor].original_turn <= turn) {
      const int suffix = solve(turn + 1, cursor + 1,
                               moved(position, raw[cursor].action), mask,
                               seeds);
      const int delay = raw[cursor].action.op == Op::Move
          ? turn - raw[cursor].original_turn : 0;
      if (suffix < infinity) best = std::min(best, suffix + delay);
    }
    for (std::size_t index = 0; index < objectives.size(); ++index) {
      if ((mask & (std::uint64_t{1} << index)) != 0) continue;
      const auto& objective = objectives[index];
      const int length =
          static_cast<int>(objective.remaining_transitions.size());
      if (length <= 0 || !(objective.tile == position) ||
          turn + length > static_cast<int>(actor.raw.size()) ||
          turn + length - 1 > objective.deadline_turn ||
          static_cast<int>(actor.raw.size()) - (turn + length) <
              static_cast<int>(raw.size() - cursor))
        continue;
      auto next_seeds = seeds;
      bool valid = true;
      for (const auto& transition : objective.remaining_transitions) {
        if (transition.op != Op::Dig && transition.op != Op::Plant &&
            transition.op != Op::Build && transition.op != Op::Water &&
            transition.op != Op::Harvest) {
          valid = false;
          break;
        }
        if (transition.op == Op::Plant) {
          if (transition.item < 0 || transition.quantity <= 0 ||
              next_seeds[transition.item] < transition.quantity) {
            valid = false;
            break;
          }
          next_seeds[transition.item] -= transition.quantity;
        }
      }
      if (!valid) continue;
      best = std::min(best, solve(
          turn + length, cursor, position,
          mask | (std::uint64_t{1} << index), std::move(next_seeds)));
    }
    return best;
  };
  return solve(0, 0, actor.start, 0, resources.seeds);
}

void random_single_actor_matches_bruteforce() {
  std::mt19937 random(0xD00D1234U);
  for (int fixture = 0; fixture < 128; ++fixture) {
    const int turns = 4 + static_cast<int>(random() % 3);
    ActorPlan actor;
    actor.actor = 0;
    actor.start = {3, 3};
    std::vector<Position> reachable{actor.start};
    auto position = actor.start;
    for (int turn = 0; turn < turns; ++turn) {
      Action action = pass();
      if (random() % 3 == 0) action = move(static_cast<int>(random() % 4));
      actor.raw.push_back(action);
      position = moved(position, action);
      reachable.push_back(position);
    }
    actor.blocked_equivalent_capacity.assign(actor.raw.size(), false);
    std::vector<Objective> objectives;
    for (int index = 0; index < 3; ++index) {
      const auto tile = reachable[random() % reachable.size()];
      const bool needs_seed = random() % 2 == 0;
      objectives.push_back(
          {static_cast<std::uint64_t>(1000 + fixture * 3 + index), tile,
           {needs_seed ? plant(0) : water(0)},
           static_cast<int>(random() % static_cast<unsigned>(turns)),
           1 + static_cast<int>(random() % 9), random() % 2 == 0});
    }
    const ResourceSnapshot resources{{{0, static_cast<int>(random() % 3)}}};
    const auto result =
        compile_time_expanded({actor}, objectives, resources);
    const int oracle = brute_single_actor_cost(actor, objectives, resources);
    require(result.exact, "small random DP unexpectedly fell back");
    if (result.total_cost != oracle) {
      std::cerr << "BRUTE fixture=" << fixture << " turns=" << turns
                << " seeds=" << resources.seeds.at(0)
                << " result=" << result.total_cost << " oracle=" << oracle
                << " exact=" << result.exact
                << " value=" << result.completed_objective_value
                << " delay=" << result.move_timing_deviation
                << " raw=";
      for (const auto& action : actor.raw)
        std::cerr << '(' << static_cast<int>(action.op) << ',' << action.arg0
                  << ')';
      std::cerr << " objectives=";
      for (const auto& objective : objectives)
        std::cerr << '{' << objective.id << "@" << objective.tile.row << ','
                  << objective.tile.column << " op="
                  << static_cast<int>(objective.remaining_transitions[0].op)
                  << " d=" << objective.deadline_turn
                  << " v=" << objective.value
                  << " c=" << objective.critical << '}';
      std::cerr << " assignments=";
      for (const auto& assignment : result.assignments)
        std::cerr << '{' << assignment.objective_id << '@' << assignment.turn
                  << '}';
      std::cerr << '\n';
      throw std::runtime_error("time-expanded DP differs from brute-force "
                               "oracle at fixture " +
                               std::to_string(fixture));
    }
  }
}

void state_and_time_cap_fallback_is_greedy_equivalent() {
  ActorPlan actor;
  actor.actor = 0;
  actor.start = {2, 2};
  actor.raw.assign(32, pass());
  actor.blocked_equivalent_capacity.assign(32, false);
  std::vector<Objective> objectives;
  for (int index = 0; index < 20; ++index)
    objectives.push_back(
        {static_cast<std::uint64_t>(2000 + index), {2, 2}, {water(0)},
         31, 1 + index, false});
  const auto greedy = compile_route_sequence({actor}, objectives);
  const auto started = std::chrono::steady_clock::now();
  const auto bounded = compile_time_expanded({actor}, objectives);
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started).count();
  require(!bounded.exact && bounded.fell_back_to_greedy,
          "250k-state/1s fixture did not exercise fallback");
  require(bounded.manifest == greedy.manifest &&
              bounded.assignments == greedy.assignments &&
              bounded.unscheduled_objectives ==
                  greedy.unscheduled_objectives &&
              bounded.ordered_move_sequences_exact ==
                  greedy.ordered_move_sequences_exact &&
              bounded.terminal_unexecuted_raw_actions == 0,
          "bounded fallback was not completely greedy-equivalent");
  require(elapsed_ms < 1500,
          "nominal 1s fallback exceeded the bounded wall-clock allowance");
}

void mismatched_maturity_epoch_is_rejected() {
  ActorPlan actor{0, {1, 1}, {pass()}, {false}};
  const Objective objective{3000, {1, 1}, {harvest(0)}, 0, 1, true};
  bool rejected = false;
  try {
    static_cast<void>(compile_time_expanded(
        {actor}, {objective}, {},
        ObservationReadiness{
            90, {{{3000, 0}}}, {HarvestProofToken{89, 3000, 0}}, {}}));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "proof token from a stale observation epoch was accepted");
}

}  // namespace

int main() try {
  raw_move_and_position_timeline_are_exact();
  suffix_is_atomic_and_capacity_shortage_is_unscheduled();
  same_tile_is_serial_and_priority_is_lexicographic();
  worker_not_at_tile_cannot_act();
  random_single_actor_matches_bruteforce();
  state_and_time_cap_fallback_is_greedy_equivalent();
  mismatched_maturity_epoch_is_rejected();

  int bugs = 0;
  if (objective_move_corrupts_future_position_witness()) {
    ++bugs;
    std::cout << "BUG objective-MOVE accepted: fixed raw timeline is stale "
                 "and move_slots_exact remains true\n";
  }
  if (future_inventory_is_unchecked_witness()) {
    ++bugs;
    std::cout << "BUG PLANT accepted without current inventory/receipt proof\n";
  }
  if (route_sequence_completes_after_deadline_witness()) {
    ++bugs;
    std::cout << "BUG route-sequence objective completed after deadline; "
                 "contention also stranded a raw MOVE at day end\n";
  }
  if (raw_tile_effect_is_not_serialized_witness()) {
    ++bugs;
    std::cout << "BUG raw and repair tile effects share one tile/turn in "
                 "fixed and route-sequence planners\n";
  }
  if (route_sequence_partial_suffix_witness()) {
    ++bugs;
    std::cout << "BUG route-sequence emitted a partial objective suffix "
                 "before a future raw tile reservation forced cancellation\n";
  }
  if (time_expanded_shared_seed_component_witness()) {
    ++bugs;
    std::cout << "BUG time-expanded result marked exact but shared-seed "
                 "components return value 6 instead of global optimum 10\n";
  }
  if (maturity_proof_replay_witness()) {
    ++bugs;
    std::cout << "BUG identical nonzero maturity epoch/proof can be replayed "
                 "without a current-epoch or consumption token\n";
  }
  if (maturity_duplicate_objective_id_witness()) {
    ++bugs;
    std::cout << "BUG duplicate objective ids make one maturity proof token "
                 "ambiguous instead of rejecting the input\n";
  }
  std::cout << "day-horizon adversarial audit good_invariants=7 bugs="
            << bugs << '\n';
  return bugs == 0 ? 0 : 3;
} catch (const std::exception& error) {
  std::cerr << "FAIL adversarial fixture: " << error.what() << '\n';
  return 2;
}
