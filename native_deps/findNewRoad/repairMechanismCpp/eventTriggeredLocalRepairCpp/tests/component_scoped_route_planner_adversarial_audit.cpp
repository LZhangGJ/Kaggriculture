#include "component_scoped_route_planner.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace g001::day_horizon_repair;
using g001::event_local_repair::Op;

namespace {

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

Action move(int direction, int tag = 0) {
  return {Op::Move, -1, 1, direction, tag};
}
Action dig(int tag = 0) { return {Op::Dig, -1, 1, tag, 0}; }
Action plant(int item, int quantity = 1) {
  return {Op::Plant, item, quantity, 0, 0};
}
Action unknown(int tag) { return {Op::Other, -1, 1, tag, 0}; }

UnsupportedEffectCertificate reads_resource(int item) {
  UnsupportedEffectCertificate certificate;
  certificate.scope = UnsupportedEffectScope::ActorLocal;
  certificate.resource_reads.insert(item);
  certificate.no_unlisted_global_effects = true;
  return certificate;
}

Position after_move(Position position, const Action& action) {
  if (action.op != Op::Move) return position;
  if (action.arg0 == 0) --position.row;
  else if (action.arg0 == 1) ++position.row;
  else if (action.arg0 == 2) --position.column;
  else if (action.arg0 == 3) ++position.column;
  else throw std::runtime_error("invalid fixture direction");
  return position;
}

bool effect(const Action& action) {
  return action.op != Op::Pass && action.op != Op::Move;
}

void certify_moves(ComponentActorPlan& actor, int rows = 20,
                   int columns = 20) {
  BoardLegalityCertificate board;
  board.rows = rows;
  board.columns = columns;
  board.complete = true;
  Position position = actor.start;
  for (const auto& source : actor.ordered_raw) {
    if (source.action.op != Op::Move) continue;
    const Position destination = after_move(position, source.action);
    board.move_transitions[source.source_id] = {position, destination};
    position = destination;
  }
  actor.board = std::move(board);
}

void validate_safe_manifest(const std::vector<ComponentActorPlan>& actors,
                            const std::vector<Objective>& objectives,
                            const ResourceSnapshot& resources,
                            const ComponentScopedResult& result) {
  if (!result.merge_safe) return;
  check(result.manifest.size() == actors.size(), "safe merge actor width");
  std::map<std::pair<int, int>, Assignment> assignments;
  std::map<std::uint64_t, std::set<int>> objective_transitions;
  for (const auto& assignment : result.assignments) {
    check(assignments.emplace(
              std::pair{assignment.actor, assignment.turn}, assignment).second,
          "two assignments own one actor/turn");
    objective_transitions[assignment.objective_id].insert(
        assignment.transition_index);
  }
  std::map<int, int> seed_demand;
  std::set<std::tuple<int, int, int>> tile_turns;
  for (std::size_t index = 0; index < actors.size(); ++index) {
    const auto& actor = actors[index];
    check(result.manifest[index].size() == static_cast<std::size_t>(actor.turns),
          "safe merge turn width");
    Position position = actor.start;
    std::size_t raw = 0;
    for (int turn = 0; turn < actor.turns; ++turn) {
      const auto& action = result.manifest[index][static_cast<std::size_t>(turn)];
      const auto assigned = assignments.find({actor.actor, turn});
      if (assigned != assignments.end()) {
        check(assigned->second.action == action,
              "assignment does not own emitted action");
        const auto objective = std::find_if(
            objectives.begin(), objectives.end(), [&](const auto& candidate) {
              return candidate.id == assigned->second.objective_id;
            });
        check(objective != objectives.end() && objective->tile == position,
              "repair assignment is unreachable");
      } else if (action.op != Op::Pass) {
        check(raw < actor.ordered_raw.size() &&
                  action == actor.ordered_raw[raw].action,
              "raw source substituted or reordered");
        check(turn >= actor.ordered_raw[raw].earliest_turn,
              "raw source emitted before release");
        check(result.raw_source_manifest[index][static_cast<std::size_t>(turn)] ==
                  actor.ordered_raw[raw].source_id,
              "raw receipt names wrong source id");
        ++raw;
      }
      if (effect(action))
        check(tile_turns.insert({position.row, position.column, turn}).second,
              "two tile effects share tile/turn");
      if (action.op == Op::Plant)
        seed_demand[action.item] += action.quantity;
      position = after_move(position, action);
    }
    check(raw == actor.ordered_raw.size(), "terminal raw source debt");
  }
  for (const auto& objective : objectives) {
    const auto count = objective_transitions[objective.id].size();
    check(count == 0 || count == objective.remaining_transitions.size(),
          "objective suffix was partially emitted");
  }
  for (const auto& [item, demand] : seed_demand) {
    const auto found = resources.seeds.find(item);
    check(found != resources.seeds.end() && demand <= found->second,
          "safe merge overcommits seed resource");
  }
}

void failed_merge_marks_every_rolled_back_objective_unscheduled() {
  const ComponentActorPlan repair{0, {0, 0}, 1, {}};
  const ComponentActorPlan raw{
      1, {9, 9}, 1, {{20, 0, plant(3), true, std::nullopt}}};
  const Objective objective{30, {0, 0}, {plant(3)}, 0, 50, true};
  const auto result = compile_component_scoped_route(
      {repair, raw}, {objective}, ResourceSnapshot{{{3, 1}}});
  check(!result.merge_safe && result.assignments.empty() &&
            result.committed_components == 0,
        "resource-conflicting merge did not roll back actions");
  check(result.unscheduled_objectives == std::vector<std::uint64_t>{30},
        "rolled-back objective was falsely reported scheduled");
}

void unsupported_unknown_requires_an_effect_scope_certificate() {
  const ComponentActorPlan repair{0, {0, 0}, 2, {}};
  const ComponentActorPlan opaque{
      1, {9, 9}, 2, {{40, 0, unknown(77), true, std::nullopt}}};
  const Objective objective{41, {0, 0}, {plant(2)}, 1, 50, true};
  const auto result = compile_component_scoped_route(
      {repair, opaque}, {objective}, ResourceSnapshot{{{2, 1}}});
  // The ABI has no effect/resource footprint for Other. Therefore it cannot
  // prove that source 40 does not consume crop-2 seed or mutate objective 41.
  check(result.assignments.empty() && result.committed_components == 0,
        "opaque unsupported source contaminated a disjoint committed component");
}

void release_times_and_duplicate_directions_remain_exact() {
  ComponentActorPlan actor{
      7, {3, 3}, 5,
      {{50, 2, move(3, 1), false, std::nullopt},
       {51, 0, move(3, 2), false, std::nullopt},
       {52, 3, move(0, 3), false, std::nullopt}}};
  certify_moves(actor);
  const auto result = compile_component_scoped_route({actor}, {});
  check(result.merge_safe && result.raw_source_manifest[0][2] == 50 &&
            result.raw_source_manifest[0][3] == 51 &&
            result.raw_source_manifest[0][4] == 52,
        "source order/release allowed duplicate EAST overtaking");
  validate_safe_manifest({actor}, {}, {}, result);
}

void shared_tile_and_seed_edges_are_globally_safe() {
  const ComponentActorPlan left{0, {1, 1}, 3, {}};
  const ComponentActorPlan right{1, {1, 1}, 3, {}};
  const Objective first{60, {1, 1}, {plant(4)}, 2, 20, true};
  const Objective second{61, {1, 1}, {plant(4)}, 2, 19, true};
  const ResourceSnapshot resources{{{4, 1}}};
  const auto result = compile_component_scoped_route(
      {left, right}, {first, second}, resources);
  validate_safe_manifest({left, right}, {first, second}, resources, result);
  check(result.merge_safe && result.assignments.size() == 1 &&
            result.unscheduled_objectives.size() == 1,
        "shared tile/seed edge did not serialize scarce production");
}

void raw_tile_effect_requires_transition_compatibility() {
  const ComponentActorPlan actor{
      0, {0, 0}, 2, {{70, 0, dig(70), false, std::nullopt}}};
  const Objective water_after_dig{
      71, {0, 0}, {{Op::Water, 2, 1, 0, 0}}, 1, 50, true};
  const auto result = compile_component_scoped_route(
      {actor}, {water_after_dig});
  // Connectivity/serialization is insufficient: DIG invalidates the crop
  // state required by a later WATER. No tile transition snapshot/certificate
  // exists in this ABI, so the objective must remain debt.
  check(result.assignments.empty() &&
            result.unscheduled_objectives == std::vector<std::uint64_t>{71},
        "raw tile effect was serialized but its state dependency was ignored");
}

void movement_reachability_requires_board_legality() {
  const ComponentActorPlan actor{
      0, {0, 0}, 2, {{80, 0, move(0), false, std::nullopt}}};
  const Objective impossible{81, {-1, 0}, {dig()}, 1, 50, true};
  const auto result = compile_component_scoped_route({actor}, {impossible});
  // Native NORTH at row zero cannot establish position {-1,0}. The offline
  // ABI has no board bounds/blocked-edge proof and therefore must fail closed.
  check(!result.merge_safe || result.assignments.empty(),
        "off-board MOVE created fictitious objective reachability");
}

void certified_resource_read_forms_a_component_edge() {
  const ComponentActorPlan reader{
      0, {9, 9}, 2,
      {{90, 1, unknown(90), true, reads_resource(5)}}};
  const ComponentActorPlan repair{1, {0, 0}, 2, {}};
  const Objective consumes_before_read{
      91, {0, 0}, {plant(5)}, 0, 50, true};
  const auto result = compile_component_scoped_route(
      {reader, repair}, {consumes_before_read}, ResourceSnapshot{{{5, 1}}});
  // A raw source whose behavior reads item 5 cannot be proven independent of
  // a component that consumes item 5 before the raw source release.
  check(result.assignments.empty(),
        "certified resource read was omitted from the conflict graph");
}

// Brute-force maximum completed value for the deliberately small randomized
// oracle domain: one-transition objectives and MOVE-only raw queues.
int brute_value(const std::vector<ComponentActorPlan>& actors,
                const std::vector<Objective>& objectives,
                const ResourceSnapshot& resources) {
  const int turns = actors.front().turns;
  const std::size_t all = (std::size_t{1} << objectives.size()) - 1;
  int best = -1;
  struct State {
    std::vector<Position> positions;
    std::vector<std::size_t> raw;
    std::size_t completed{};
    std::map<int, int> seeds;
  };
  State initial;
  for (const auto& actor : actors) {
    initial.positions.push_back(actor.start);
    initial.raw.push_back(0);
  }
  auto value = [&](std::size_t mask) {
    int total = 0;
    for (std::size_t objective = 0; objective < objectives.size(); ++objective)
      if ((mask & (std::size_t{1} << objective)) != 0)
        total += std::max(0, objectives[objective].value);
    return total;
  };
  std::function<void(int, const State&)> search = [&](int turn,
                                                       const State& state) {
    if (turn == turns) {
      bool raw_done = true;
      for (std::size_t actor = 0; actor < actors.size(); ++actor)
        raw_done = raw_done &&
            state.raw[actor] == actors[actor].ordered_raw.size();
      if (raw_done) best = std::max(best, value(state.completed));
      return;
    }
    struct Choice { Action action; int objective{-1}; bool raw{}; };
    std::vector<std::vector<Choice>> choices(actors.size());
    for (std::size_t actor = 0; actor < actors.size(); ++actor) {
      choices[actor].push_back({});
      const auto cursor = state.raw[actor];
      if (cursor < actors[actor].ordered_raw.size() &&
          actors[actor].ordered_raw[cursor].earliest_turn <= turn)
        choices[actor].push_back(
            {actors[actor].ordered_raw[cursor].action, -1, true});
      for (std::size_t objective = 0; objective < objectives.size(); ++objective)
        if ((state.completed & (std::size_t{1} << objective)) == 0 &&
            objectives[objective].deadline_turn >= turn &&
            objectives[objective].tile == state.positions[actor])
          choices[actor].push_back(
              {objectives[objective].remaining_transitions[0],
               static_cast<int>(objective), false});
    }
    std::vector<Choice> selected(actors.size());
    std::function<void(std::size_t)> combine = [&](std::size_t actor) {
      if (actor != actors.size()) {
        for (const auto& choice : choices[actor]) {
          selected[actor] = choice;
          combine(actor + 1);
        }
        return;
      }
      State next = state;
      std::set<int> objective_once;
      std::set<Position> effect_tiles;
      for (std::size_t index = 0; index < selected.size(); ++index) {
        const auto& choice = selected[index];
        if (choice.objective >= 0 &&
            !objective_once.insert(choice.objective).second)
          return;
        if (effect(choice.action) &&
            !effect_tiles.insert(state.positions[index]).second)
          return;
        if (choice.action.op == Op::Plant) {
          next.seeds[choice.action.item] += choice.action.quantity;
          const auto found = resources.seeds.find(choice.action.item);
          if (found == resources.seeds.end() ||
              next.seeds[choice.action.item] > found->second)
            return;
        }
        if (choice.raw) ++next.raw[index];
        if (choice.objective >= 0)
          next.completed |= std::size_t{1} << choice.objective;
        next.positions[index] = after_move(next.positions[index], choice.action);
      }
      search(turn + 1, next);
    };
    combine(0);
  };
  search(0, initial);
  static_cast<void>(all);
  return best;
}

void random_small_instances_match_global_safe_oracle() {
  std::mt19937 random(0xC064A11U);
  for (int fixture = 0; fixture < 128; ++fixture) {
    const int actor_count = 1 + static_cast<int>(random() % 2);
    const int turns = 3 + static_cast<int>(random() % 2);
    std::vector<ComponentActorPlan> actors;
    for (int actor = 0; actor < actor_count; ++actor) {
      ComponentActorPlan plan{actor, {5 + actor * 4, 5}, turns, {}};
      const int raw_count = static_cast<int>(random() % 3);
      const int release_base = raw_count == 0 ? 0 :
          static_cast<int>(random() % (turns - raw_count + 1));
      for (int source = 0; source < raw_count; ++source)
        plan.ordered_raw.push_back(
            {static_cast<std::uint64_t>(1000 + fixture * 10 + actor * 3 + source),
             release_base + source,
             move(static_cast<int>(random() % 4), source), false,
             std::nullopt});
      certify_moves(plan);
      actors.push_back(std::move(plan));
    }
    std::vector<Objective> objectives;
    const int objective_count = static_cast<int>(random() % 3);
    for (int objective = 0; objective < objective_count; ++objective) {
      const auto& actor = actors[static_cast<std::size_t>(random() % actor_count)];
      Position tile = actor.start;
      for (const auto& source : actor.ordered_raw)
        if ((random() & 1U) != 0) tile = after_move(tile, source.action);
      const bool seeded = (random() & 1U) != 0;
      objectives.push_back(
          {static_cast<std::uint64_t>(5000 + fixture * 3 + objective), tile,
           {seeded ? plant(1) : dig(objective)}, turns - 1,
           20 + objective, true});
    }
    const ResourceSnapshot resources{{{1, static_cast<int>(random() % 3)}}};
    const auto result = compile_component_scoped_route(
        actors, objectives, resources);
    validate_safe_manifest(actors, objectives, resources, result);
    if (!result.merge_safe) continue;
    std::set<std::uint64_t> completed;
    for (const auto& assignment : result.assignments)
      completed.insert(assignment.objective_id);
    int actual = 0;
    for (const auto& objective : objectives)
      if (completed.contains(objective.id)) actual += objective.value;
    const int oracle = brute_value(actors, objectives, resources);
    check(actual == oracle,
          "random fixture " + std::to_string(fixture) +
              " differs from global brute-force value actual=" +
              std::to_string(actual) + " oracle=" + std::to_string(oracle));
  }
}

}  // namespace

int main() try {
  release_times_and_duplicate_directions_remain_exact();
  shared_tile_and_seed_edges_are_globally_safe();
  random_small_instances_match_global_safe_oracle();
  certified_resource_read_forms_a_component_edge();
  movement_reachability_requires_board_legality();
  raw_tile_effect_requires_transition_compatibility();
  unsupported_unknown_requires_an_effect_scope_certificate();
  failed_merge_marks_every_rolled_back_objective_unscheduled();
  std::cout << "component-scoped adversarial audit: 5 groups passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "component-scoped adversarial audit failure: "
            << error.what() << '\n';
  return 1;
}
