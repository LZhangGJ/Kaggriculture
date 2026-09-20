#include "component_scoped_route_planner.hpp"

#include <initializer_list>
#include <iostream>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

using namespace g001::day_horizon_repair;
using g001::event_local_repair::Op;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Action move(int direction, int opaque = 71) {
  return {Op::Move, -1, 1, direction, opaque};
}
Action other(int tag) { return {Op::Other, -1, 1, tag, 19}; }
Action dig() { return {Op::Dig, -1, 1, 0, 0}; }
Action water(int crop) { return {Op::Water, crop, 1, 0, 0}; }
Action plant(int crop) { return {Op::Plant, crop, 1, 0, 0}; }
Action harvest(int crop) { return {Op::Harvest, crop, 1, 0, 0}; }

BoardLegalityCertificate board(
    int rows, int columns,
    std::initializer_list<std::tuple<std::uint64_t, Position, Position>> moves,
    std::set<std::pair<Position, Position>> blocked = {}) {
  BoardLegalityCertificate certificate;
  certificate.rows = rows;
  certificate.columns = columns;
  certificate.complete = true;
  certificate.blocked_edges = std::move(blocked);
  for (const auto& [source, from, to] : moves)
    certificate.move_transitions[source] = {from, to};
  return certificate;
}

UnsupportedEffectCertificate tile_local_no_resource() {
  UnsupportedEffectCertificate certificate;
  certificate.scope = UnsupportedEffectScope::TileLocal;
  certificate.no_unlisted_global_effects = true;
  return certificate;
}

void reason2_on_other_actor_does_not_drop_old_south() {
  const ComponentActorPlan route{
      0, {0, 0}, 3,
      {{100, 0, move(1), false}, {101, 0, move(3), false}},
      board(10, 10, {{100, {0, 0}, {1, 0}}, {101, {1, 0}, {1, 1}}})};
  const ComponentActorPlan unsupported{
      1, {8, 8}, 3,
      {{200, 0, other(6), true, tile_local_no_resource()}}};
  const auto result = compile_component_scoped_route({route, unsupported}, {});
  check(result.merge_safe && result.conflict_components == 2 &&
            result.sealed_components == 1 &&
            result.manifest[0][0] == move(1) &&
            result.manifest[0][1] == move(3) &&
            result.raw_source_manifest[0][0] == 100 &&
            result.raw_source_manifest[0][1] == 101 &&
            result.manifest[1][0] == other(6),
        "d0 witness dropped old SOUTH during other-actor rejection");
}

void build_on_other_actor_cannot_swap_old_east_with_new_north() {
  const ComponentActorPlan route{
      0, {4, 4}, 3,
      {{300, 0, move(3, 81), false}, {301, 0, move(0, 82), false}},
      board(10, 10, {{300, {4, 4}, {4, 5}}, {301, {4, 5}, {3, 5}}})};
  const ComponentActorPlan build{
      1, {9, 9}, 3,
      {{400, 0, other(12), true, tile_local_no_resource()}}};
  const auto result = compile_component_scoped_route({route, build}, {});
  check(result.merge_safe && result.manifest[0][0] == move(3, 81) &&
            result.manifest[0][1] == move(0, 82) &&
            result.raw_source_manifest[0][0] == 300 &&
            result.raw_source_manifest[0][1] == 301 &&
            result.manifest[1][0] == other(12),
        "d6 witness allowed new NORTH to overtake deferred EAST");
}

void independent_crop_component_commits_while_non_crop_is_sealed() {
  const ComponentActorPlan crop{0, {1, 1}, 3, {}};
  const ComponentActorPlan animal{
      1, {7, 7}, 3,
      {{500, 0, other(4), true, tile_local_no_resource()}}};
  const Objective objective{600, {1, 1}, {dig(), water(0)}, 1, 50, true};
  const auto result = compile_component_scoped_route(
      {crop, animal}, {objective});
  check(result.merge_safe && result.conflict_components == 2 &&
            result.sealed_components == 1 &&
            result.committed_components == 1 &&
            result.assignments.size() == 2 &&
            result.manifest[0][0] == dig() &&
            result.manifest[0][1] == water(0) &&
            result.manifest[1][0] == other(4),
        "independent crop component was globally rejected by noncrop actor");
}

void same_tile_unknown_effect_seals_connected_crop_component() {
  const ComponentActorPlan crop{0, {2, 2}, 2, {}};
  const ComponentActorPlan unknown{
      1, {2, 2}, 2,
      {{700, 0, other(9), true, tile_local_no_resource()}}};
  const Objective objective{701, {2, 2}, {water(0)}, 1, 10, true};
  const auto result = compile_component_scoped_route(
      {crop, unknown}, {objective});
  check(result.merge_safe && result.conflict_components == 1 &&
            result.sealed_components == 1 && result.assignments.empty() &&
            result.unscheduled_objectives ==
                std::vector<std::uint64_t>{701} &&
            result.manifest[1][0] == other(9),
        "unknown same-tile effect was not conservatively isolated");
}

void failed_global_resource_gate_leaks_no_component_assignment() {
  const ComponentActorPlan repair{0, {1, 1}, 1, {}};
  const ComponentActorPlan protected_raw{
      1, {8, 8}, 1, {{800, 0, plant(2), true}}};
  const Objective objective{801, {1, 1}, {plant(2)}, 0, 20, true};
  const auto result = compile_component_scoped_route(
      {repair, protected_raw}, {objective}, ResourceSnapshot{{{2, 1}}});
  check(!result.merge_safe && !result.global_resource_safe &&
            result.assignments.empty() && result.committed_components == 0 &&
            result.committed_objectives.empty() &&
            result.unscheduled_objectives ==
                std::vector<std::uint64_t>{801} &&
            result.objective_conservation &&
            result.manifest[0][0].op == Op::Pass &&
            result.manifest[1][0] == plant(2) &&
            result.raw_source_manifest[1][0] == 800,
        "failed global resource gate leaked a partial component manifest");
}

void raw_tile_state_requires_proof_and_allows_certified_harvest_plant() {
  const ComponentActorPlan unproved{
      0, {3, 3}, 2, {{900, 0, dig(), false}}};
  const Objective illegal_water{901, {3, 3}, {water(2)}, 1, 20, true};
  const auto rejected = compile_component_scoped_route(
      {unproved}, {illegal_water});
  check(rejected.assignments.empty() && rejected.unscheduled_objectives ==
            std::vector<std::uint64_t>{901},
        "unproved DIG state allowed a later WATER");

  const RawTileTransitionCertificate harvest_to_empty{
      {g001::event_local_repair::TileKind::Crop, 2, true, true},
      {g001::event_local_repair::TileKind::Empty, -1, false, false}};
  const ComponentActorPlan proved{
      0, {3, 3}, 2,
      {{910, 0, harvest(2), false, std::nullopt, harvest_to_empty}}};
  const Objective replant{911, {3, 3}, {plant(2)}, 1, 20, true};
  const auto accepted = compile_component_scoped_route(
      {proved}, {replant}, ResourceSnapshot{{{2, 1}}});
  check(accepted.merge_safe && accepted.manifest[0][0] == harvest(2) &&
            accepted.manifest[0][1] == plant(2) &&
            accepted.committed_objectives ==
                std::vector<std::uint64_t>{911},
        "certified HARVEST->Empty->PLANT chain was not admitted");
}

void every_move_requires_exact_board_and_edge_proof() {
  const ComponentActorPlan missing{
      0, {0, 0}, 1, {{920, 0, move(3), false}}};
  check(!compile_component_scoped_route({missing}, {}).merge_safe,
        "MOVE without board certificate was accepted");

  const auto blocked_edge = std::pair{Position{0, 0}, Position{0, 1}};
  const ComponentActorPlan blocked{
      0, {0, 0}, 1, {{921, 0, move(3), false}},
      board(10, 10, {{921, {0, 0}, {0, 1}}}, {blocked_edge})};
  check(!compile_component_scoped_route({blocked}, {}).merge_safe,
        "blocked MOVE edge was accepted");

  const ComponentActorPlan legal{
      0, {0, 0}, 1, {{922, 0, move(3), false}},
      board(10, 10, {{922, {0, 0}, {0, 1}}})};
  check(compile_component_scoped_route({legal}, {}).merge_safe,
        "exact legal MOVE certificate was rejected");
}

}  // namespace

int main() {
  try {
    reason2_on_other_actor_does_not_drop_old_south();
    build_on_other_actor_cannot_swap_old_east_with_new_north();
    independent_crop_component_commits_while_non_crop_is_sealed();
    same_tile_unknown_effect_seals_connected_crop_component();
    failed_global_resource_gate_leaks_no_component_assignment();
    raw_tile_state_requires_proof_and_allows_certified_harvest_plant();
    every_move_requires_exact_board_and_edge_proof();
    std::cout << "component-scoped route planner: 7 fixtures passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
