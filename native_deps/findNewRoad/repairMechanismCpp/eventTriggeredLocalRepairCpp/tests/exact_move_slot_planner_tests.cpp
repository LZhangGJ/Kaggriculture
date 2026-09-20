#include "../include/exact_move_slot_planner.hpp"

#include <iostream>
#include <stdexcept>

namespace local = g001::day_horizon_repair;
using g001::event_local_repair::Action;
using g001::event_local_repair::Op;
using g001::event_local_repair::Position;
using g001::event_local_repair::TileKind;
using g001::event_local_repair::TileObservation;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Action move(int direction) { return {Op::Move, -1, 1, direction, 0}; }

void weed_water_recompiles_without_moving_move_slots_and_carries_debt() {
  const Position plot{2, 2};
  local::ExactMoveSlotActor day0{
      0, plot, {{Op::Water, 0, 1}, move(3), move(2), {}},
      {true, false, false, true}, {}};
  const auto first = local::compile_exact_move_slots(
      {day0}, {{plot, {TileKind::Weed, -1, false, false}}}, {{0, 1}}, {}, 0);
  check(first.move_slots_exact && first.move_positions_exact,
        "exact-slot planner changed MOVE geometry");
  check(first.manifest[0][0].op == Op::Dig &&
            first.manifest[0][1] == move(3) &&
            first.manifest[0][2] == move(2) &&
            first.manifest[0][3].op == Op::Plant,
        "WATER-on-weed did not become DIG, exact MOVE, return PLANT");
  check(first.active_intents.size() == 1 &&
            first.final_tiles.at(plot).kind == TileKind::Crop &&
            !first.final_tiles.at(plot).watered_today,
        "unfinished WATER lifecycle did not survive as plot debt");

  local::ExactMoveSlotActor day1{0, plot, {{}, {}}, {true, true}, {}};
  const auto second = local::compile_exact_move_slots(
      {day1}, first.final_tiles, first.remaining_seeds,
      first.active_intents, 1);
  check(second.manifest[0][0].op == Op::Water &&
            second.active_intents.empty() &&
            second.completed_intents.size() == 1,
        "cross-day plot debt did not WATER and retire");
}

void zero_seed_fails_closed_and_keeps_plot_debt() {
  const Position plot{1, 1};
  local::ExactMoveSlotActor actor{
      0, plot, {{Op::Plant, 2, 1}, move(3)}, {true, false}, {}};
  const auto result = local::compile_exact_move_slots(
      {actor}, {{plot, {TileKind::Empty, -1, false, false}}}, {{2, 0}});
  check(result.manifest[0][0] == actor.raw[0] &&
            result.manifest[0][1] == move(3) &&
            result.active_intents.size() == 1 &&
            result.rejected.at(local::ExactSlotReject::SeedUnavailable) == 1,
        "zero-seed PLANT did not fail closed with persistent debt");
}

void legal_ongoing_harvest_is_preserved_without_creating_replant_debt() {
  const Position plot{4, 4};
  local::ExactMoveSlotActor actor{
      0, plot, {{Op::Harvest, 2, 1}, move(3)}, {true, false}, {}};
  const auto result = local::compile_exact_move_slots(
      {actor}, {{plot, {TileKind::Crop, 2, true, true}}}, {});
  check(result.manifest[0][0] == actor.raw[0] &&
            result.manifest[0][1] == move(3) &&
            result.active_intents.empty() && result.rejected.empty() &&
            result.original_move_slot_changes == 0 &&
            result.original_move_payload_changes == 0 &&
            result.move_positions_exact,
        "legal ongoing HARVEST was rewritten or created replant debt");
}

void every_reject_path_preserves_the_raw_slot() {
  const Position plot{5, 5};
  {
    local::ExactMoveSlotActor actor{0, plot, {{Op::Dig}}, {true}, {}};
    const auto result = local::compile_exact_move_slots(
        {actor}, {{plot, {TileKind::Weed, -1, false, false}}}, {});
    check(result.manifest[0][0] == actor.raw[0] &&
              result.rejected.at(
                  local::ExactSlotReject::MissingDesiredCrop) == 1,
          "missing desired crop rejection changed raw");
  }
  {
    local::PersistentPlotIntent debt{9, plot, 0, true, 0};
    local::ExactMoveSlotActor actor{0, plot, {{}}, {true}, {}};
    const auto result = local::compile_exact_move_slots(
        {actor}, {{plot, {TileKind::Crop, 0, true, false}}}, {}, {debt}, 1);
    check(result.manifest[0][0] == actor.raw[0] &&
              result.rejected.at(local::ExactSlotReject::MaturityWait) == 1,
          "maturity rejection changed raw PASS");
  }
  {
    local::ExactMoveSlotActor actor{
        0, plot, {{Op::Water, 0, 1}}, {true}, {}};
    const auto result = local::compile_exact_move_slots(
        {actor}, {{plot, {TileKind::Structure, -1, false, false}}}, {});
    check(result.manifest[0][0] == actor.raw[0] &&
              result.rejected.at(local::ExactSlotReject::TileUnsupported) == 1,
          "unsupported tile rejection changed raw");
  }
  {
    local::ExactMoveSlotActor left{
        0, plot, {{Op::Water, 0, 1}}, {true}, {}};
    local::ExactMoveSlotActor right = left;
    right.actor = 1;
    const auto result = local::compile_exact_move_slots(
        {left, right}, {{plot, {TileKind::Crop, 0, false, false}}}, {});
    check(result.manifest[0][0] == left.raw[0] &&
              result.manifest[1][0] == right.raw[0] &&
              result.rejected.at(local::ExactSlotReject::TileSerialized) == 2,
          "same-tile serialization rejection partially rewrote raw");
  }
  {
    local::ExactMoveSlotActor repair{
        0, plot, {{Op::Water, 0, 1}}, {true}, {}};
    local::ExactMoveSlotActor typed_later{
        1, plot, {{Op::Other, 0, 1, 15, 0}}, {false}, {}};
    const auto result = local::compile_exact_move_slots(
        {repair, typed_later},
        {{plot, {TileKind::Crop, 0, false, false}}}, {});
    check(result.manifest[0][0] == repair.raw[0] &&
              result.manifest[1][0] == typed_later.raw[0] &&
              result.rejected.at(local::ExactSlotReject::TileSerialized) == 1,
          "later prefix-bound raw effect was not reserved before repair");
  }
}

void deviation_only_preserves_untriggered_production_bytes() {
  const Position plot{2, 2};
  local::ExactMoveSlotActor actor{
      0, plot,
      {{Op::Water, 0, 1}, move(3), move(2), {Op::Water, 0, 1}},
      {false, false, false, true},
      {false, false, false, true}};
  const auto result = local::compile_exact_move_slots(
      {actor}, {{plot, {TileKind::Crop, 0, false, false}}}, {});
  check(result.manifest[0][0] == actor.raw[0] &&
            result.manifest[0][1] == actor.raw[1] &&
            result.manifest[0][2] == actor.raw[2] &&
            result.absorbed_raw_intents == 1,
        "deviation-only mode absorbed or rewrote untriggered production");
}

void global_seed_manifest_failure_rolls_back_every_repair() {
  local::ExactMoveSlotActor left{
      0, {1, 1}, {{Op::Plant, 0, 1}}, {true}, {true}};
  local::ExactMoveSlotActor right{
      1, {8, 8}, {{Op::Plant, 0, 1}}, {true}, {true}};
  const auto result = local::compile_exact_move_slots(
      {left, right},
      {{{1, 1}, {TileKind::Empty, -1, false, false}},
       {{8, 8}, {TileKind::Empty, -1, false, false}}},
      {{0, 1}});
  check(result.assignments == 0 && result.manifest[0][0] == left.raw[0] &&
            result.manifest[1][0] == right.raw[0] &&
            result.active_intents.size() == 2 &&
            result.rejected.at(
                local::ExactSlotReject::GlobalSeedManifestUnsafe) == 1,
        "global seed failure leaked a partial exact-slot manifest or debt");
}

}  // namespace

int main() try {
  weed_water_recompiles_without_moving_move_slots_and_carries_debt();
  zero_seed_fails_closed_and_keeps_plot_debt();
  legal_ongoing_harvest_is_preserved_without_creating_replant_debt();
  every_reject_path_preserves_the_raw_slot();
  deviation_only_preserves_untriggered_production_bytes();
  global_seed_manifest_failure_rolls_back_every_repair();
  std::cout << "exact move slot planner: 6 fixtures passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
