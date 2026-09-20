#include "component_scoped_route_planner.hpp"
#include "exact_move_slot_planner.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace local = g001::day_horizon_repair;
using g001::event_local_repair::Action;
using g001::event_local_repair::Op;
using g001::event_local_repair::Position;
using g001::event_local_repair::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Action move(int direction, int tag = 0) {
  return {Op::Move, -1, 1, direction, tag};
}

void randomized_exact_move_actor_slot_payload_invariant() {
  std::mt19937 rng(20260829);
  for (int fixture = 0; fixture < 256; ++fixture) {
    constexpr int actors_count = 4;
    constexpr int turns = 24;
    std::vector<local::ExactMoveSlotActor> actors;
    for (int actor = 0; actor < actors_count; ++actor) {
      local::ExactMoveSlotActor input;
      input.actor = actor;
      input.start = {30 + actor * 100, 30 + actor * 100};
      input.raw.resize(turns);
      input.service_slot.resize(turns);
      for (int turn = 0; turn < turns; ++turn) {
        if ((rng() % 3U) == 0U)
          input.raw[turn] = move(static_cast<int>(rng() % 4U),
                                 fixture * 1000 + actor * 24 + turn);
        else
          input.service_slot[turn] = true;
      }
      actors.push_back(std::move(input));
    }
    const auto result = local::compile_exact_move_slots(actors, {}, {});
    check(result.move_slots_exact && result.move_positions_exact &&
              result.original_move_slot_changes == 0 &&
              result.original_move_payload_changes == 0,
          "randomized exact planner changed MOVE actor/slot/direction/tag");
    for (std::size_t actor = 0; actor < actors.size(); ++actor)
      for (int turn = 0; turn < turns; ++turn)
        if (actors[actor].raw[turn].op == Op::Move)
          check(result.manifest[actor][turn] == actors[actor].raw[turn],
                "MOVE payload escaped exact-slot postcondition");
  }
}

void ongoing_harvest_preserves_raw_without_leaking_plot_debt() {
  for (const int crop : {2, 3}) {
    const Position plot{4, 4};
    const local::ExactMoveSlotActor actor{
        0, plot, {{Op::Harvest, crop, 1}, move(3)}, {true, false}, {}};
    const auto result = local::compile_exact_move_slots(
        {actor}, {{plot, {TileKind::Crop, crop, true, true}}}, {});
    check(result.manifest[0][0] == actor.raw[0] &&
              result.manifest[0][1] == actor.raw[1],
          "ongoing tomato/strawberry HARVEST reject swallowed raw");
    check(result.active_intents.empty() && result.absorbed_raw_intents == 0,
          "ongoing HARVEST leaked a permanent replant debt");
  }
}

local::BoardLegalityCertificate east_board(std::uint64_t source) {
  local::BoardLegalityCertificate board;
  board.rows = 10;
  board.columns = 10;
  board.complete = true;
  board.move_transitions[source] = {{2, 2}, {2, 3}};
  return board;
}

void component_route_contract_can_shift_an_original_move_slot() {
  const std::uint64_t source = 100;
  const local::ComponentActorPlan actor{
      0, {2, 2}, 2, {{source, 0, move(3, 77), false}},
      east_board(source)};
  const local::Objective water{
      200, {2, 2}, {{Op::Water, 0, 1}}, 0, 1000, true};
  const auto result = local::compile_component_scoped_route({actor}, {water});
  check(result.merge_safe && result.assignments.size() == 1 &&
            result.manifest[0][0].op == Op::Water &&
            result.manifest[0][1] == move(3, 77) &&
            result.raw_source_manifest[0][1] == source &&
            !result.move_slots_exact,
        "component planner no longer demonstrates its non-exact-slot contract");
}

void global_opaque_seals_even_disjoint_repair_components() {
  const local::ComponentActorPlan opaque{
      0, {8, 8}, 2,
      {{300, 0, {Op::Other, -1, 1, 5, 0}, true}}};
  const local::ComponentActorPlan repair{1, {1, 1}, 2, {}};
  const local::Objective objective{
      301, {1, 1}, {{Op::Dig}}, 1, 100, true};
  const auto result = local::compile_component_scoped_route(
      {opaque, repair}, {objective});
  check(result.merge_safe && result.assignments.empty() &&
            result.committed_objectives.empty() &&
            result.unscheduled_objectives ==
                std::vector<std::uint64_t>{301} &&
            result.sealed_components == 2,
        "GlobalOpaque was relaxed to a local certificate");
}

}  // namespace

int main() try {
  randomized_exact_move_actor_slot_payload_invariant();
  ongoing_harvest_preserves_raw_without_leaking_plot_debt();
  component_route_contract_can_shift_an_original_move_slot();
  global_opaque_seals_even_disjoint_repair_components();
  std::cout << "repair integration adversarial audit: 4 groups passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
