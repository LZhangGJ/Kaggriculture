#include "joint_fixed_move_oracle.hpp"
#include "clairvoyant_economic_oracle.hpp"
#include "reactive_production.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace joint_fixed_move_oracle;

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
fastkag::PlayerAction frame(fastkag::Op op) {
  fastkag::PlayerAction out;
  out.units.push_back({op, fastkag::Item::NONE, 1});
  out.market.push_back({fastkag::Op::SELL, fastkag::Item::WHEAT, 3});
  return out;
}
void step_one(fastkag::Simulator& simulator,
              const fastkag::PlayerAction& player_zero = {}) {
  std::array<fastkag::PlayerAction, 2> actions{};
  actions[0] = player_zero;
  simulator.step(actions);
}
}  // namespace

int main() {
  Tape baseline{frame(fastkag::Op::NORTH), frame(fastkag::Op::PASS)};
  auto economic = baseline;
  economic[1].units[0] = {fastkag::Op::DIG};
  economic[0].market.clear();
  require(certify_absolute_moves(baseline, economic).valid,
          "non-MOVE and market edits must pass");

  auto shifted = baseline;
  shifted[0].units[0] = {fastkag::Op::PASS};
  shifted[1].units[0] = {fastkag::Op::NORTH};
  require(!certify_absolute_moves(baseline, shifted).valid,
          "a delayed identical MOVE must fail absolute identity");

  fastkag::Simulator simulator({}, 17);
  const auto held = apply_trade_policy(baseline[0], simulator, 0,
                                       TradePolicy::HoldAll);
  require(held.units.size() == baseline[0].units.size() &&
              action_equal(held.units[0], baseline[0].units[0]) &&
              held.market.empty(),
          "trade policy must be market-only");

  // Minimal counterexample for the old delayed replay.  The blocked PLANT
  // needs two stationary production slots: one for PLANT and one for WATER.
  // Moving either EAST/WEST by one turn changes the route; DIG+late PLANT
  // without WATER merely creates a crop that dies at day end.
  Tape production(6);
  for (auto& value : production) value.units.resize(1);
  production[0].units[0] = {fastkag::Op::PLANT, fastkag::Item::WHEAT, 1};
  production[1].units[0] = {fastkag::Op::EAST};
  production[2].units[0] = {fastkag::Op::WEST};
  production[3].units[0] = {fastkag::Op::PASS};
  production[4].units[0] = {fastkag::Op::PASS};
  production[5].units[0] = {fastkag::Op::EAST};
  std::vector<std::vector<fastkag::Position>> positions{
      {{0, 0}}, {{0, 0}}, {{1, 0}}, {{0, 0}}, {{0, 0}}, {{0, 0}}};
  WeedCollision collision{0, 0, {0, 0}, production[0].units[0]};
  const auto patches = enumerate_absolute_weed_patches(
      production, production, positions, collision, 24);
  require(!patches.empty(), "PLANT+WATER fixed-MOVE patch must exist");
  const auto& patch = patches.front();
  require(patch.replay_step == 3 && patch.water_step == 4,
          "repair must use both same-position stationary slots");
  require(patch.tape[0].units[0].op == fastkag::Op::DIG &&
              patch.tape[3].units[0].op == fastkag::Op::PLANT &&
              patch.tape[4].units[0].op == fastkag::Op::WATER,
          "repair must compile DIG, PLANT, WATER");
  require(certify_absolute_moves(production, patch.tape).valid,
          "compiled repair must retain every absolute MOVE");

  auto no_water = production;
  no_water[4].units[0] = {fastkag::Op::EAST};
  positions[5][0] = {1, 0};
  const auto impossible = enumerate_absolute_weed_patches(
      no_water, no_water, positions, collision, 24);
  require(impossible.empty(),
          "PLANT repair without a later same-position WATER slot must fail closed");

  // Simulator witness: a replanted crop starts with one unwatered day.  If
  // the compiler emits DIG+PLANT but omits same-day WATER, it is WEED again at
  // the next day boundary and the apparent repair cannot produce a harvest.
  fastkag::Config lifecycle_config;
  lifecycle_config.episode_steps = 72;
  lifecycle_config.weed_spawn_chance = 1.0;
  fastkag::Simulator dry(lifecycle_config, 42);
  for (int step = 0; step < 23; ++step) step_one(dry);
  fastkag::PlayerAction buy_seed;
  buy_seed.market.push_back(
      {fastkag::Op::BUY_SEED, fastkag::Item::WHEAT, 1});
  step_one(dry, buy_seed);
  const int spawn_tile = 4 * lifecycle_config.board_size + 4;
  require(dry.farms()[0].tiles[spawn_tile].kind == fastkag::TileKind::WEED,
          "fixture must spawn a weed at the farmer position");
  fastkag::Simulator watered = dry;
  step_one(dry, frame(fastkag::Op::DIG));
  step_one(watered, frame(fastkag::Op::DIG));
  auto plant = frame(fastkag::Op::PLANT);
  plant.units[0].item = fastkag::Item::WHEAT;
  step_one(dry, plant);
  step_one(watered, plant);
  step_one(dry);
  step_one(watered, frame(fastkag::Op::WATER));
  while (dry.step_count() < 48) step_one(dry);
  while (watered.step_count() < 48) step_one(watered);
  require(dry.farms()[0].tiles[spawn_tile].kind == fastkag::TileKind::WEED,
          "late PLANT without WATER must die at day end");
  require(watered.farms()[0].tiles[spawn_tile].kind ==
              fastkag::TileKind::PLANT,
          "DIG+PLANT+WATER must preserve the replanted crop");

  // Full lifecycle mode is state-driven: even when the next source byte is a
  // MOVE (and contains no WATER hint), it inserts required care before that
  // MOVE and leaves the daily route cursor on the same source byte.
  Tape state_driven_tape(72);
  for (auto& value : state_driven_tape) value.units.resize(1);
  state_driven_tape[48].units[0] = {fastkag::Op::EAST};
  production::RouteCursorState state_driven_cursor;
  production::RouteCursorAudit state_driven_audit;
  production::ReactiveOptions state_driven_options;
  state_driven_options.manage_all_known_crops = true;
  auto state_driven_action = production::apply_reactive_route_cursor(
      state_driven_tape, watered, 0, state_driven_cursor,
      state_driven_audit, state_driven_options);
  require(state_driven_action.units[0].op == fastkag::Op::WATER,
          "state-reactive compiler must WATER without a source WATER byte");
  step_one(watered, state_driven_action);
  state_driven_action = production::apply_reactive_route_cursor(
      state_driven_tape, watered, 0, state_driven_cursor,
      state_driven_audit, state_driven_options);
  require(state_driven_action.units[0].op == fastkag::Op::EAST,
          "state-driven care must delay, not delete, the following MOVE");

  // Cross-day debt witness: there is no same-position production slot left on
  // the collision day.  The fixed MOVE still executes, then the state machine
  // resumes at the next day's reset, waters across days, and harvests when the
  // crop is actually mature rather than at its obsolete tape deadline.
  auto multi_day_config = lifecycle_config;
  multi_day_config.episode_steps = 120;
  fastkag::Simulator multi_day(multi_day_config, 43);
  for (int step = 0; step < 23; ++step) step_one(multi_day);
  step_one(multi_day, buy_seed);
  production::ReactiveState reactive_state;
  production::ReactiveAudit reactive_audit;
  auto blocked_plant = frame(fastkag::Op::PLANT);
  blocked_plant.units[0].item = fastkag::Item::WHEAT;
  auto compiled = production::apply_reactive_fixed_move(
      blocked_plant, multi_day, 0, reactive_state, reactive_audit);
  require(compiled.units[0].op == fastkag::Op::DIG,
          "reactive debt must DIG the observed weed");
  step_one(multi_day, compiled);
  const auto move_east = frame(fastkag::Op::EAST);
  compiled = production::apply_reactive_fixed_move(
      move_east, multi_day, 0, reactive_state, reactive_audit);
  require(compiled.units[0].op == fastkag::Op::EAST,
          "reactive debt must not delay the next MOVE");
  step_one(multi_day, compiled);
  while (multi_day.step_count() < 48) step_one(multi_day);
  compiled = production::apply_reactive_fixed_move(
      frame(fastkag::Op::PASS), multi_day, 0, reactive_state, reactive_audit);
  require(compiled.units[0].op == fastkag::Op::DIG,
          "cross-day debt must re-DIG a weed that respawned overnight");
  step_one(multi_day, compiled);
  compiled = production::apply_reactive_fixed_move(
      frame(fastkag::Op::PASS), multi_day, 0, reactive_state, reactive_audit);
  require(compiled.units[0].op == fastkag::Op::PLANT,
          "cross-day debt must plant on the next stationary slot");
  step_one(multi_day, compiled);
  compiled = production::apply_reactive_fixed_move(
      frame(fastkag::Op::PASS), multi_day, 0, reactive_state, reactive_audit);
  require(compiled.units[0].op == fastkag::Op::WATER,
          "cross-day debt must water after planting");
  step_one(multi_day, compiled);
  while (multi_day.step_count() < 72) step_one(multi_day);
  compiled = production::apply_reactive_fixed_move(
      frame(fastkag::Op::PASS), multi_day, 0, reactive_state, reactive_audit);
  require(compiled.units[0].op == fastkag::Op::WATER,
          "multi-day crop debt must maintain lifecycle watering");
  step_one(multi_day, compiled);
  while (multi_day.step_count() < 96) step_one(multi_day);
  compiled = production::apply_reactive_fixed_move(
      frame(fastkag::Op::PASS), multi_day, 0, reactive_state, reactive_audit);
  require(compiled.units[0].op == fastkag::Op::HARVEST,
          "multi-day debt must harvest at actual maturity");
  step_one(multi_day, compiled);
  require(multi_day.privates()[0].inventories[0]
                   [static_cast<int>(fastkag::Item::WHEAT)] > 0 &&
              reactive_audit.move_mismatches == 0,
          "reactive recovery must yield product with zero MOVE edits");

  // RouteCursor version of the same chain.  Day 1 contains 23 MOVE actions,
  // so after inserting DIG there is no room to replay PLANT.  The cursor drops
  // that non-MOVE, completes the entire daily MOVE skeleton in order, carries
  // the tile debt across midnight, and resumes production on the next visit.
  Tape cursor_tape(120);
  for (auto& value : cursor_tape) value.units.resize(1);
  cursor_tape[24].units[0] =
      {fastkag::Op::PLANT, fastkag::Item::WHEAT, 1};
  std::vector<fastkag::Op> expected_moves;
  for (int step = 25; step < 48; ++step) {
    const auto op = step % 2 ? fastkag::Op::EAST : fastkag::Op::WEST;
    cursor_tape[step].units[0] = {op};
    expected_moves.push_back(op);
  }
  fastkag::Simulator cursor_sim(multi_day_config, 44);
  for (int step = 0; step < 23; ++step) step_one(cursor_sim);
  step_one(cursor_sim, buy_seed);
  production::RouteCursorState cursor_state;
  production::RouteCursorAudit cursor_audit;
  std::vector<fastkag::Op> emitted_moves;
  while (cursor_sim.step_count() < 48) {
    auto action = production::apply_reactive_route_cursor(
        cursor_tape, cursor_sim, 0, cursor_state, cursor_audit);
    if (!action.units.empty() && is_move(action.units[0].op))
      emitted_moves.push_back(action.units[0].op);
    step_one(cursor_sim, action);
  }
  require(emitted_moves == expected_moves,
          "time dilation must complete the ordered daily MOVE skeleton");
  auto cursor_action = production::apply_reactive_route_cursor(
      cursor_tape, cursor_sim, 0, cursor_state, cursor_audit);
  require(cursor_action.units[0].op == fastkag::Op::DIG,
          "cross-day cursor debt must react to overnight weed state");
  step_one(cursor_sim, cursor_action);
  cursor_action = production::apply_reactive_route_cursor(
      cursor_tape, cursor_sim, 0, cursor_state, cursor_audit);
  require(cursor_action.units[0].op == fastkag::Op::PLANT,
          "cross-day cursor debt must replant on the next slot");
  step_one(cursor_sim, cursor_action);
  cursor_action = production::apply_reactive_route_cursor(
      cursor_tape, cursor_sim, 0, cursor_state, cursor_audit);
  require(cursor_action.units[0].op == fastkag::Op::WATER,
          "cross-day cursor debt must water the replant");
  step_one(cursor_sim, cursor_action);
  while (cursor_sim.step_count() < 96) {
    auto action = production::apply_reactive_route_cursor(
        cursor_tape, cursor_sim, 0, cursor_state, cursor_audit);
    step_one(cursor_sim, action);
  }
  cursor_action = production::apply_reactive_route_cursor(
      cursor_tape, cursor_sim, 0, cursor_state, cursor_audit);
  require(cursor_action.units[0].op == fastkag::Op::HARVEST,
          "route cursor debt must harvest at realized maturity");
  step_one(cursor_sim, cursor_action);
  require(cursor_sim.privates()[0].inventories[0]
                   [static_cast<int>(fastkag::Item::WHEAT)] > 0 &&
              cursor_audit.skipped_nonmoves >= 1,
          "route cursor recovery must yield product after a certified skip");

  // The simulator has 719 actionable ticks (0..718), so its final day has
  // only 23 slots.  One inserted DIG must not rely on a phantom step 719:
  // skip one PASS and close all 14 source MOVEs by the real deadline.
  fastkag::Config final_day_config;
  final_day_config.weed_spawn_chance = 1.0;
  fastkag::Simulator final_day(final_day_config, 45);
  while (final_day.step_count() < 696) step_one(final_day);
  Tape final_tape(719);
  for (auto& value : final_tape) value.units.resize(1);
  final_tape[696].units[0] = {fastkag::Op::EAST};
  for (int source = 706; source <= 718; ++source)
    final_tape[source].units[0] =
        {source % 2 ? fastkag::Op::WEST : fastkag::Op::EAST};
  production::RouteCursorState final_cursor;
  final_cursor.production.debts.resize(final_day.farms()[0].tiles.size());
  final_cursor.production.last_crop.resize(final_day.farms()[0].tiles.size(),
                                            fastkag::Item::NONE);
  final_cursor.production.last_step = final_day.step_count() - 1;
  const auto final_position = final_day.farms()[0].farmer;
  const int final_tile = final_position.y * final_day.config().board_size +
                         final_position.x;
  final_cursor.production.debts[static_cast<std::size_t>(final_tile)] =
      {true, false, fastkag::Item::WHEAT, fastkag::Op::PASS,
       final_day.step_count()};
  production::RouteCursorAudit final_audit;
  std::vector<fastkag::Op> final_moves;
  while (!final_day.done()) {
    const auto action = production::apply_reactive_route_cursor(
        final_tape, final_day, 0, final_cursor, final_audit);
    if (!action.units.empty() && is_move(action.units[0].op))
      final_moves.push_back(action.units[0].op);
    step_one(final_day, action);
  }
  require(final_moves.size() == 14 && final_audit.skipped_nonmoves >= 1,
          "partial final day must close MOVE skeleton without phantom tick");

  const auto demand = economic::demand_at_step({2}, 4);
  require(demand[0] == 1 && demand[1] == 1 && demand[2] == 1 &&
              demand[3] == 1 && demand[4] == 0,
          "future-demand oracle must decode the realized farmers market");
  const auto center = economic::demand_at_step({}, 24);
  require(center[0] == 1 && center[7] == 1 && center[8] == 0,
          "town-center refresh demand must be represented exactly");
  std::vector<economic::ReferenceFrame> future(3);
  future[0].prices.fill(10);
  future[1].prices.fill(20);
  future[2].prices.fill(15);
  future[0].shops = {2};
  const auto forecast = economic::build_future_demand(future, 1, 1000);
  require(forecast.suffix_peak_price[0][3] == 20 &&
              forecast.suffix_peak_price[2][3] == 15 &&
              forecast.suffix_units[0][3] == 2,
          "suffix forecast must expose realized future peaks and demand");
  require(economic::sell_quantity(economic::SellPace::All, 11, 3) == 11 &&
              economic::sell_quantity(economic::SellPace::Half, 11, 3) == 6 &&
              economic::sell_quantity(economic::SellPace::Cap4, 11, 3) == 4 &&
              economic::sell_quantity(economic::SellPace::OpponentPressure,
                                      11, 3) == 3,
          "wide trade quantities must remain distinct");
  economic::CropPolicy all_tiles;
  all_tiles.selected_tile_percent = 100;
  for (int tile = 0; tile < 100; ++tile)
    require(economic::tile_selected(all_tiles, tile),
            "100 percent allocation must cover every field");
  std::cout << "joint_fixed_move_oracle_tests: ok\n";
}
