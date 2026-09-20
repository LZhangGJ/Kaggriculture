#include "event_triggered_local_repair.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace g001::event_local_repair;

namespace {

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

Action pass() { return {Op::Pass, 0, 1, 0, 0}; }
Action move(int direction) { return {Op::Move, 0, 1, direction, 0}; }
Action dig() { return {Op::Dig, 0, 1, 0, 0}; }
Action plant(int item) { return {Op::Plant, item, 1, 0, 0}; }
Action build(int item) { return {Op::Build, item, 1, 0, 0}; }
Action water(int item) { return {Op::Water, item, 1, 0, 0}; }
Action harvest(int item) { return {Op::Harvest, item, 1, 0, 0}; }
Action other(int tag) { return {Op::Other, tag, 1, tag + 10, tag + 20}; }

ActorDayPlan actor_plan(int actor, std::vector<Action> actions) {
  ActorDayPlan result;
  result.actor = actor;
  for (const auto& action : actions) result.turns.push_back({action, 999999});
  return result;
}

DayPlan day(int number, std::vector<ActorDayPlan> actors) {
  return {number, static_cast<int>(actors.front().turns.size()),
          std::move(actors)};
}

ActorObservation observed(int actor, Position position, TileKind kind,
                          int item, int inventory, bool watered = false,
                          bool harvest_legal = false,
                          bool failed_fill = false) {
  return {actor, position, {kind, item, watered, harvest_legal}, item,
          inventory, failed_fill};
}

Receipt receipt(const Decision& decision, int actor, Position tile,
                TileObservation after, int inventory_delta = 0,
                bool generic = false, bool day_end_water = false,
                std::uint64_t actor_generation = 0) {
  return {decision.id, actor, tile, after, inventory_delta, generic,
          day_end_water, actor_generation};
}

ActorObservation observed_generation(int actor, std::uint64_t generation,
                                     Position position, TileKind kind,
                                     int item, int inventory) {
  auto result = observed(actor, position, kind, item, inventory);
  result.actor_generation = generation;
  return result;
}

void no_trigger_parity() {
  Compiler disabled;
  disabled.install_day(day(3, {actor_plan(4, {move(2), other(7)}),
                               actor_plan(9, {water(11), move(3)})}));
  auto first = disabled.decide(
      {3, 0, {observed(4, {1, 1}, TileKind::Other, 0, 0),
              observed(9, {4, 5}, TileKind::Crop, 11, 2, true)}});
  check(first.actions == first.baseline, "disabled compiler changed baseline");
  check(first.bindings.empty(), "disabled compiler created repair binding");

  Compiler enabled({true, 0, 2});
  enabled.install_day(day(4, {actor_plan(4, {move(1), plant(11)}),
                              actor_plan(9, {other(99), move(4)})}));
  auto move_turn = enabled.decide(
      {4, 0, {observed(4, {2, 2}, TileKind::Other, 0, 0),
              observed(9, {7, 7}, TileKind::Other, 0, 0)}});
  check(move_turn.actions == move_turn.baseline,
        "enabled/no-event MOVE turn changed baseline");
  auto legal_plant = enabled.decide(
      {4, 1, {observed(4, {2, 3}, TileKind::Empty, 11, 1),
              observed(9, {7, 7}, TileKind::Other, 0, 0)}});
  check(legal_plant.actions == legal_plant.baseline,
        "legal production action changed without deviation");
  check(enabled.open_transactions().empty(),
        "legal production created a transaction");
}

void enabled_no_trigger_719_step_multi_actor_bit_parity() {
  constexpr int kTurns = 719;
  constexpr int kActors = 4;
  std::vector<ActorDayPlan> plans;
  for (int actor = 0; actor < kActors; ++actor) {
    std::vector<Action> actions;
    actions.reserve(kTurns);
    for (int turn = 0; turn < kTurns; ++turn) {
      const int tag = 1000 * actor + turn;
      switch ((turn + actor) % 8) {
        case 0: actions.push_back({Op::Pass, tag, turn + 1, tag + 2, tag + 3}); break;
        case 1: actions.push_back({Op::Move, tag, turn + 1, tag + 2, tag + 3}); break;
        case 2: actions.push_back({Op::Dig, tag, turn + 1, tag + 2, tag + 3}); break;
        case 3: actions.push_back({Op::Plant, actor, 1, tag + 2, tag + 3}); break;
        case 4: actions.push_back({Op::Build, actor + 30, 1, tag + 2, tag + 3}); break;
        case 5: actions.push_back({Op::Water, actor, 1, tag + 2, tag + 3}); break;
        case 6: actions.push_back({Op::Harvest, actor, 1, tag + 2, tag + 3}); break;
        case 7: actions.push_back({Op::Other, tag, turn + 1, tag + 2, tag + 3}); break;
      }
    }
    plans.push_back(actor_plan(actor, std::move(actions)));
  }

  Compiler compiler({true, 0, 2});
  compiler.install_day(day(20, std::move(plans)));
  for (int turn = 0; turn < kTurns; ++turn) {
    std::vector<ActorObservation> observations;
    for (int actor = 0; actor < kActors; ++actor) {
      const auto operation = static_cast<Op>((turn + actor) % 8);
      TileKind kind = TileKind::Other;
      int inventory = 0;
      bool watered = false;
      bool harvest_legal = false;
      if (operation == Op::Plant) {
        kind = TileKind::Empty;
        inventory = 1;
      } else if (operation == Op::Build) {
        kind = TileKind::Empty;
      } else if (operation == Op::Water) {
        kind = TileKind::Crop;
      } else if (operation == Op::Harvest) {
        kind = TileKind::Crop;
        watered = true;
        harvest_legal = true;
      }
      observations.push_back(
          observed(actor, {actor, turn}, kind, actor, inventory, watered,
                   harvest_legal));
    }
    auto decision = compiler.decide({20, turn, std::move(observations)});
    check(decision.actions == decision.baseline,
          "719-step enabled no-trigger bit parity diverged at turn " +
              std::to_string(turn));
    check(decision.bindings.empty() && decision.fail_closed_actors.empty(),
          "719-step parity unexpectedly entered repair path");
    check(compiler.commit(decision, decision.actions),
          "719-step parity manifest rejected");
  }
  check(compiler.open_transactions().empty(),
        "719-step parity created a repair transaction");
  for (int actor = 0; actor < kActors; ++actor)
    check(compiler.movement_invariant_holds(actor),
          "719-step parity changed a MOVE route");
}

void weed_with_pass_slack_and_multi_actor_isolation() {
  Compiler compiler({true, 0, 2});
  compiler.install_day(day(0, {actor_plan(0, {plant(5), pass()}),
                               actor_plan(1, {other(41), move(2)})}));
  const Position tile{3, 4};
  auto first = compiler.decide(
      {0, 0, {observed(0, tile, TileKind::Weed, 5, 1),
              observed(1, {8, 8}, TileKind::Other, 0, 0)}});
  check(first.actions[0] == dig(), "weed was not locally replaced by DIG");
  check(first.actions[1] == other(41), "unaffected actor changed");
  check(first.bindings.size() == 1 && first.bindings[0].actor == 0,
        "repair binding escaped affected actor");
  check(compiler.commit(first, first.actions), "DIG manifest rejected");
  check(compiler.observe(receipt(first, 0, tile,
                                 {TileKind::Empty, 0, false, false})),
        "DIG effect not confirmed");

  auto second = compiler.decide(
      {0, 1, {observed(0, tile, TileKind::Empty, 5, 1),
              observed(1, {8, 8}, TileKind::Other, 0, 0)}});
  check(second.actions[0] == plant(5), "intended PLANT was not replayed");
  check(second.actions[1] == move(2),
        "unaffected actor byte-exact action was not preserved");
  check(compiler.commit(second, second.actions), "PLANT manifest rejected");
  check(compiler.observe(receipt(second, 0, tile,
                                 {TileKind::Crop, 5, false, false})),
        "PLANT effect not confirmed");
  check(compiler.movement_invariant_holds(0),
        "PASS absorption altered affected actor MOVE route");
  check(compiler.movement_invariant_holds(1),
        "repair altered unaffected actor MOVE route");
}

void weed_before_move_never_leases_or_delays_move() {
  Compiler compiler({true, 0, 2});
  compiler.install_day(
      day(1, {actor_plan(2, {plant(7), move(3), pass()})}));
  const Position tile{6, 6};
  auto first = compiler.decide(
      {1, 0, {observed(2, tile, TileKind::Weed, 7, 1)}});
  check(first.actions[0] == dig(), "DIG insertion failed before MOVE");
  check(compiler.commit(first, first.actions), "DIG commit failed");
  check(compiler.observe(receipt(first, 2, tile,
                                 {TileKind::Empty, 0, false, false})),
        "DIG receipt failed");

  auto second = compiler.decide(
      {1, 1, {observed(2, tile, TileKind::Empty, 7, 1)}});
  check(second.actions[0] == move(3) && second.bindings.empty(),
        "plot debt leased or delayed an ordered MOVE");

  auto third = compiler.decide(
      {1, 2, {observed(2, tile, TileKind::Empty, 7, 1)}});
  check(third.actions[0] == plant(7),
        "later PASS did not lease the plot PLANT debt");
  check(compiler.movement_invariant_holds(2),
        "MOVE order/direction invariant failed");
  check(compiler.audit().delayed_move_rewrites == 0,
        "MOVE was rewritten under plot-owned semantics");
}

void harvest_on_weed_recompiles_from_observation_without_losing_route() {
  Compiler compiler({true, 0, 2});
  const Position tile{7, 8};
  compiler.install_day(day(
      2, {actor_plan(0, {harvest(17), move(4), pass(), pass(), pass()}),
          actor_plan(1, {other(90), other(91), other(92), other(93),
                         other(94)})}));

  auto dig_decision = compiler.decide(
      {2, 0, {observed(0, tile, TileKind::Weed, 17, 1),
              observed(1, {9, 9}, TileKind::Other, 0, 0)}});
  check(dig_decision.actions[0] == dig(),
        "weed-blocked HARVEST did not compile to DIG");
  check(dig_decision.actions[1] == other(90),
        "state repair changed another actor");
  check(compiler.commit(dig_decision, dig_decision.actions),
        "HARVEST/weed DIG commit failed");
  check(compiler.observe(receipt(dig_decision, 0, tile,
                                 {TileKind::Empty, -1, false, false})),
        "HARVEST/weed DIG receipt failed");

  auto plant_decision = compiler.decide(
      {2, 1, {observed(0, tile, TileKind::Empty, 17, 1),
              observed(1, {9, 9}, TileKind::Other, 0, 0)}});
  check(plant_decision.actions[0] == move(4) &&
            plant_decision.bindings.empty(),
        "post-DIG plot debt leased an ordered MOVE");
  check(plant_decision.actions[1] == other(91),
        "MOVE preservation changed another actor");

  auto water_decision = compiler.decide(
      {2, 2, {observed(0, tile, TileKind::Empty, 17, 1),
              observed(1, {9, 9}, TileKind::Other, 0, 0)}});
  check(water_decision.actions[0] == plant(17),
        "post-MOVE PASS did not lease PLANT");
  check(compiler.commit(water_decision, water_decision.actions),
        "state-recompiled PLANT commit failed");
  check(compiler.observe(receipt(water_decision, 0, tile,
                                 {TileKind::Crop, 17, false, false})),
        "state-recompiled PLANT receipt failed");

  auto harvest_decision = compiler.decide(
      {2, 3, {observed(0, tile, TileKind::Crop, 17, 0, false, false),
              observed(1, {9, 9}, TileKind::Other, 0, 0)}});
  check(harvest_decision.actions[0] == water(17),
        "unwatered crop did not lease WATER from PASS");
  check(compiler.commit(harvest_decision, harvest_decision.actions),
        "state-recompiled WATER commit failed");
  check(compiler.observe(receipt(harvest_decision, 0, tile,
                                 {TileKind::Crop, 17, true, false})),
        "state-recompiled WATER receipt failed");

  auto move_decision = compiler.decide(
      {2, 4, {observed(0, tile, TileKind::Crop, 17, 0, true, true),
              observed(1, {9, 9}, TileKind::Other, 0, 0)}});
  check(move_decision.actions[0] == harvest(17),
        "original HARVEST debt was not discharged when mature");
  check(move_decision.actions[1] == other(94),
        "final unaffected actor action changed");
  check(compiler.movement_invariant_holds(0),
        "state recompilation changed MOVE order or payload");
  check(compiler.protected_action_invariant_holds(0),
        "state recompilation swallowed an original production action");
  check(compiler.audit().state_recompiles >= 1,
        "blocked production observation recompile was not audited");
}

void water_on_weed_recompiles_and_preserves_other_critical_actions() {
  Compiler compiler({true, 0, 2});
  const Position tile{11, 4};
  compiler.install_day(day(
      14, {actor_plan(5, {water(23), other(44), move(1), pass(), pass()})}));

  auto run = [&](int turn, TileObservation before, int inventory,
                 const Action& expected, TileObservation after) {
    auto decision = compiler.decide(
        {14, turn,
         {observed(5, tile, before.kind, 23, inventory,
                   before.watered_today, before.harvest_legal)}});
    check(decision.actions[0] == expected,
          "WATER/weed state sequence diverged at turn " +
              std::to_string(turn));
    check(compiler.commit(decision, decision.actions),
          "WATER/weed sequence commit failed");
    check(compiler.observe(receipt(decision, 5, tile, after)),
          "WATER/weed sequence receipt failed");
  };

  run(0, {TileKind::Weed, 23, false, false}, 1, dig(),
      {TileKind::Empty, -1, false, false});
  auto other_decision = compiler.decide(
      {14, 1, {observed(5, tile, TileKind::Empty, 23, 1)}});
  check(other_decision.actions[0] == other(44) &&
            other_decision.bindings.empty(),
        "plot lease swallowed opaque critical action");
  auto move_decision = compiler.decide(
      {14, 2, {observed(5, tile, TileKind::Empty, 23, 1)}});
  check(move_decision.actions[0] == move(1) &&
            move_decision.bindings.empty(),
        "plot lease swallowed WATER-plan MOVE");
  run(3, {TileKind::Empty, -1, false, false}, 1, plant(23),
      {TileKind::Crop, 23, false, false});
  run(4, {TileKind::Crop, 23, false, false}, 0, water(23),
      {TileKind::Crop, 23, true, false});
  check(compiler.movement_invariant_holds(5),
        "WATER-plan MOVE invariant failed");
  check(compiler.protected_action_invariant_holds(5),
        "original WATER was not retained in effective plan");
}

void state_recompile_without_future_slack_fails_closed_across_day() {
  Compiler compiler({true, 0, 2});
  const Position tile{2, 14};
  compiler.install_day(day(15, {actor_plan(6, {harvest(31), move(2)})}));

  auto dig_decision = compiler.decide(
      {15, 0, {observed(6, tile, TileKind::Weed, 31, 1)}});
  check(dig_decision.actions[0] == dig(), "single-slack DIG was not emitted");
  check(compiler.commit(dig_decision, dig_decision.actions),
        "single-slack DIG commit failed");
  check(compiler.observe(receipt(dig_decision, 6, tile,
                                 {TileKind::Empty, -1, false, false})),
        "single-slack DIG receipt failed");

  auto no_room = compiler.decide(
      {15, 1, {observed(6, tile, TileKind::Empty, 31, 1)}});
  check(no_room.actions[0] == move(2) && no_room.bindings.empty(),
        "no-slack state recompile leased the final MOVE");
  check(compiler.open_transactions().size() == 1,
        "no-slack state recompile discarded repair debt");
  check(compiler.protected_action_invariant_holds(6),
        "fail-closed state recompile lost original HARVEST");

  compiler.install_day(day(16, {actor_plan(6, {pass(), pass()})}));
  auto retry = compiler.decide(
      {16, 0, {observed(6, tile, TileKind::Empty, 31, 1)}});
  check(retry.actions[0] == plant(31),
        "cross-day state-recompile debt did not retry PLANT");
}

void certified_production_is_never_an_absorption_sink() {
  Compiler compiler({true, 0, 2});
  ActorDayPlan plan;
  plan.actor = 8;
  plan.turns = {{water(37), 0}, {harvest(37), 0}};
  compiler.install_day(day(17, {std::move(plan)}));

  auto decision = compiler.decide(
      {17, 0, {observed(8, {3, 12}, TileKind::Weed, 37, 1)}});
  check(decision.actions[0] == dig() && decision.bindings.size() == 1,
        "blocked WATER was not state-recompiled in place");
  check(compiler.protected_action_invariant_holds(8),
        "protected production invariant failed");
  check(compiler.open_transactions().size() == 1,
        "protected-sink fail-closed path lost cross-day debt");
}

void post_replant_immature_harvest_remains_source_debt_until_mature() {
  Compiler compiler({true, 0, 2});
  const Position tile{15, 6};
  compiler.install_day(
      day(21, {actor_plan(4, {harvest(41), pass(), pass(), pass()})}));

  auto execute = [&](int day_number, int turn, TileObservation before,
                     int inventory, const Action& expected,
                     TileObservation after, int inventory_delta = 0) {
    auto decision = compiler.decide(
        {day_number, turn,
         {observed(4, tile, before.kind, 41, inventory,
                   before.watered_today, before.harvest_legal)}});
    check(decision.actions[0] == expected,
          "multi-day lifecycle action mismatch");
    check(compiler.commit(decision, decision.actions),
          "multi-day lifecycle commit failed");
    check(compiler.observe(
              receipt(decision, 4, tile, after, inventory_delta)),
          "multi-day lifecycle receipt failed");
  };

  execute(21, 0, {TileKind::Weed, 41, false, false}, 1, dig(),
          {TileKind::Empty, -1, false, false});
  execute(21, 1, {TileKind::Empty, -1, false, false}, 1, plant(41),
          {TileKind::Crop, 41, false, false});
  execute(21, 2, {TileKind::Crop, 41, false, false}, 0, water(41),
          {TileKind::Crop, 41, true, false});
  execute(21, 3, {TileKind::Crop, 41, true, true}, 0, harvest(41),
          {TileKind::Empty, -1, false, false}, 1);
  check(compiler.open_transactions()[0].confirmed_harvests == 1,
        "first generation did not remain open");

  compiler.install_day(
      day(22, {actor_plan(4, {pass(), pass(), pass(), harvest(41)})}));
  execute(22, 0, {TileKind::Empty, -1, false, false}, 1, plant(41),
          {TileKind::Crop, 41, false, false});
  execute(22, 1, {TileKind::Crop, 41, false, false}, 0, water(41),
          {TileKind::Crop, 41, true, false});

  auto growing_wait = compiler.decide(
      {22, 2, {observed(4, tile, TileKind::Crop, 41, 0, true, false)}});
  check(growing_wait.actions[0] == pass() && growing_wait.bindings.empty(),
        "ordinary immature wait should remain PASS");

  const int confirmed_before_invalid =
      compiler.open_transactions()[0].confirmed_actions;
  auto immature_source = compiler.decide(
      {22, 3, {observed(4, tile, TileKind::Crop, 41, 0, true, false)}});
  check(immature_source.actions[0] == harvest(41),
        "fail-closed immature source was silently replaced");
  check(immature_source.bindings.empty(),
        "immature HARVEST was staged as a successful effect");
  check(immature_source.fail_closed_actors == std::vector<int>{4},
        "immature HARVEST did not fail closed");
  check(compiler.commit(immature_source, immature_source.actions),
        "fail-closed immature manifest was rejected");
  check(!compiler.observe(receipt(immature_source, 4, tile,
                                  {TileKind::Empty, -1, false, false}, 1)),
        "unbound immature HARVEST accepted a fake success receipt");
  const auto debt_after_invalid = compiler.open_transactions()[0];
  check(debt_after_invalid.confirmed_actions == confirmed_before_invalid,
        "immature HARVEST advanced lifecycle success");
  check(debt_after_invalid.outstanding_source_actions == 1,
        "immature HARVEST source was not persisted as debt");

  compiler.install_day(day(23, {actor_plan(4, {pass(), move(2)})}));
  auto mature_retry = compiler.decide(
      {23, 0, {observed(4, tile, TileKind::Crop, 41, 0, true, true)}});
  check(mature_retry.actions[0] == harvest(41),
        "mature revisit did not replay HARVEST source debt");
  check(mature_retry.bindings.size() == 1 &&
            mature_retry.bindings[0].source_action_id != 0,
        "mature retry was not tied to the persisted source");
  check(compiler.commit(mature_retry, mature_retry.actions),
        "mature source retry commit failed");
  check(compiler.observe(receipt(mature_retry, 4, tile,
                                 {TileKind::Empty, -1, false, false}, 1)),
        "mature source retry receipt failed");
  check(compiler.open_transactions().empty(),
        "confirmed mature retry did not close lifecycle/source debt");

  auto retained_move = compiler.decide(
      {23, 1, {observed(4, tile, TileKind::Empty, 41, 0)}});
  check(retained_move.actions[0] == move(2),
        "mature source retry swallowed following MOVE");
}

void already_watered_and_nonempty_plant_are_not_false_successes() {
  const Position water_tile{5, 16};
  Compiler water_compiler({true, 0, 2});
  water_compiler.install_day(day(24, {actor_plan(1, {water(51)})}));
  auto redundant_water = water_compiler.decide(
      {24, 0,
       {observed(1, water_tile, TileKind::Crop, 51, 0, true, false)}});
  check(redundant_water.bindings.empty() &&
            redundant_water.fail_closed_actors == std::vector<int>{1},
        "already-watered WATER was treated as a successful action");
  check(water_compiler.open_transactions()[0].outstanding_source_actions == 1,
        "already-watered WATER source debt was not retained");
  water_compiler.install_day(day(25, {actor_plan(1, {pass()})}));
  auto next_day_water = water_compiler.decide(
      {25, 0,
       {observed(1, water_tile, TileKind::Crop, 51, 0, false, false)}});
  check(next_day_water.actions[0] == water(51) &&
            next_day_water.bindings.size() == 1 &&
            next_day_water.bindings[0].source_action_id != 0,
        "reset-day WATER did not discharge the original source debt");
  check(water_compiler.commit(next_day_water, next_day_water.actions),
        "reset-day WATER commit failed");
  check(water_compiler.observe(receipt(
            next_day_water, 1, water_tile,
            {TileKind::Crop, 51, true, false})),
        "reset-day WATER receipt failed");
  check(water_compiler.open_transactions()[0].outstanding_source_actions == 1,
        "intermediate WATER incorrectly closed the plot objective");

  const Position plant_tile{6, 16};
  Compiler plant_compiler({true, 0, 2});
  plant_compiler.install_day(day(26, {actor_plan(2, {plant(52)})}));
  auto nonempty_plant = plant_compiler.decide(
      {26, 0,
       {observed(2, plant_tile, TileKind::Crop, 52, 1, true, false)}});
  check(nonempty_plant.bindings.empty() &&
            nonempty_plant.fail_closed_actors == std::vector<int>{2},
        "PLANT on non-empty tile was treated as a successful action");
  check(plant_compiler.open_transactions()[0].outstanding_source_actions == 1,
        "non-empty PLANT source debt was not retained");

  plant_compiler.install_day(day(27, {actor_plan(2, {pass()})}));
  auto mature_harvest = plant_compiler.decide(
      {27, 0,
       {observed(2, plant_tile, TileKind::Crop, 52, 0, true, true)}});
  check(mature_harvest.actions[0] == harvest(52),
        "non-empty PLANT debt did not follow observed crop lifecycle");
  check(plant_compiler.commit(mature_harvest, mature_harvest.actions),
        "intermediate HARVEST commit failed");
  check(plant_compiler.observe(receipt(
            mature_harvest, 2, plant_tile,
            {TileKind::Empty, -1, false, false}, 1)),
        "intermediate HARVEST receipt failed");
  plant_compiler.install_day(day(28, {actor_plan(2, {pass()})}));
  auto plant_retry = plant_compiler.decide(
      {28, 0,
       {observed(2, plant_tile, TileKind::Empty, 52, 1)}});
  check(plant_retry.actions[0] == plant(52) &&
            plant_retry.bindings.size() == 1 &&
            plant_retry.bindings[0].source_action_id != 0,
        "empty revisit did not replay the original PLANT source debt");
  check(plant_compiler.commit(plant_retry, plant_retry.actions),
        "source-backed PLANT retry commit failed");
  check(plant_compiler.observe(receipt(
            plant_retry, 2, plant_tile,
            {TileKind::Crop, 52, false, false})),
        "source-backed PLANT retry receipt failed");
  check(plant_compiler.open_transactions()[0].outstanding_source_actions == 1,
        "intermediate PLANT incorrectly closed the plot objective");
}

void day_rollover_preserves_debt() {
  Compiler compiler({true, 0, 2});
  const Position tile{1, 9};
  compiler.install_day(day(5, {actor_plan(3, {plant(8), move(1)})}));
  auto blocked = compiler.decide(
      {5, 0, {observed(3, tile, TileKind::Weed, 8, 1)}});
  check(blocked.actions[0] == dig(),
        "blocked PLANT was not safely state-recompiled in place");
  check(compiler.open_transactions().size() == 1,
        "blocked repair debt was discarded");

  compiler.install_day(day(6, {actor_plan(3, {pass(), pass()})}));
  auto dig_retry = compiler.decide(
      {6, 0, {observed(3, tile, TileKind::Weed, 8, 1)}});
  check(dig_retry.actions[0] == dig(), "cross-day debt did not retry DIG");
  check(compiler.commit(dig_retry, dig_retry.actions),
        "cross-day DIG commit failed");
  check(compiler.observe(receipt(dig_retry, 3, tile,
                                 {TileKind::Empty, 0, false, false})),
        "cross-day DIG receipt failed");
  auto plant_retry = compiler.decide(
      {6, 1, {observed(3, tile, TileKind::Empty, 8, 1)}});
  check(plant_retry.actions[0] == plant(8),
        "cross-day debt did not restore PLANT");
}

void failed_seed_retries_on_later_slack() {
  Compiler compiler({true, 0, 2});
  const Position tile{10, 10};
  compiler.install_day(
      day(8, {actor_plan(6, {plant(12), pass(), pass()})}));
  auto no_seed = compiler.decide(
      {8, 0, {observed(6, tile, TileKind::Empty, 12, 0, false, false,
                              true)}});
  check(no_seed.actions[0] == pass() &&
            compiler.open_transactions()[0].outstanding_source_actions == 1,
        "missing seed source was not externalized into persistent debt");
  check(no_seed.bindings.empty(), "zero seed incorrectly staged success");
  check(compiler.commit(no_seed, no_seed.actions),
        "empty no-binding manifest rejected");

  auto first_retry = compiler.decide(
      {8, 1, {observed(6, tile, TileKind::Empty, 12, 1)}});
  check(first_retry.actions[0] == plant(12), "seed retry not emitted");
  check(compiler.commit(first_retry, first_retry.actions),
        "seed retry manifest rejected");
  check(!compiler.observe(receipt(first_retry, 6, tile,
                                  {TileKind::Empty, 0, false, false})),
        "failed PLANT receipt incorrectly completed");

  auto second_retry = compiler.decide(
      {8, 2, {observed(6, tile, TileKind::Empty, 12, 1)}});
  check(second_retry.actions[0] == plant(12),
        "failed seed action was not retried");
  check(compiler.commit(second_retry, second_retry.actions),
        "second seed retry manifest rejected");
  check(compiler.observe(receipt(second_retry, 6, tile,
                                 {TileKind::Crop, 12, false, false})),
        "successful seed retry was not confirmed");
  check(compiler.open_transactions()[0].failed_receipts == 1,
        "failed seed receipt audit missing");
}

void no_available_slack_is_strict_fail_closed() {
  Compiler compiler({true, 0, 2});
  compiler.install_day(
      day(10, {actor_plan(7, {plant(3), move(0), move(1), move(2)})}));
  auto decision = compiler.decide(
      {10, 0, {observed(7, {5, 5}, TileKind::Weed, 3, 1)}});
  check(decision.actions[0] == dig(),
        "blocked source was not state-recompiled without touching MOVE tail");
  check(compiler.movement_invariant_holds(7),
        "fail-closed path changed MOVE route");
  check(compiler.open_transactions().size() == 1,
        "fail-closed path discarded cross-day debt");
}

void wheat_zero_item_id_survives_weed_transaction() {
  Compiler compiler({true, 0, 2});
  const Position tile{13, 13};
  compiler.install_day(day(13, {actor_plan(0, {plant(0), pass()})}));
  auto dig_decision = compiler.decide(
      {13, 0, {observed(0, tile, TileKind::Weed, 0, 1)}});
  check(dig_decision.actions[0] == dig(), "WHEAT(0) weed DIG missing");
  check(compiler.open_transactions()[0].desired_item == 0,
        "WHEAT(0) was confused with unknown sentinel");
  check(compiler.commit(dig_decision, dig_decision.actions),
        "WHEAT DIG commit failed");
  check(compiler.observe(receipt(dig_decision, 0, tile,
                                 {TileKind::Empty, -1, false, false})),
        "WHEAT DIG receipt failed");
  auto plant_decision = compiler.decide(
      {13, 1, {observed(0, tile, TileKind::Empty, 0, 1)}});
  check(plant_decision.actions[0] == plant(0),
        "WHEAT(0) intended PLANT was not retained");
}

void structure_repair_is_local_and_receipt_driven() {
  Compiler compiler({true, 0, 2});
  const Position tile{12, 3};
  compiler.install_day(day(11, {actor_plan(0, {build(70), pass()}),
                                actor_plan(1, {other(5), other(6)})}));
  auto dig_decision = compiler.decide(
      {11, 0, {observed(0, tile, TileKind::Weed, 70, 1),
               observed(1, {2, 2}, TileKind::Other, 0, 0)}});
  check(dig_decision.actions[0] == dig(), "structure prerequisite DIG missing");
  check(dig_decision.actions[1] == other(5),
        "structure repair changed unrelated actor");
  check(compiler.commit(dig_decision, dig_decision.actions),
        "structure DIG commit failed");
  check(compiler.observe(receipt(dig_decision, 0, tile,
                                 {TileKind::Empty, 0, false, false})),
        "structure DIG receipt failed");
  auto build_decision = compiler.decide(
      {11, 1, {observed(0, tile, TileKind::Empty, 70, 1),
               observed(1, {2, 2}, TileKind::Other, 0, 0)}});
  check(build_decision.actions[0] == build(70), "delayed BUILD missing");
  check(compiler.commit(build_decision, build_decision.actions),
        "BUILD commit failed");
  check(compiler.observe(receipt(build_decision, 0, tile,
                                 {TileKind::Structure, 70, false, false})),
        "BUILD receipt failed");
  check(compiler.open_transactions().empty(),
        "confirmed structure transaction remained open");
}

void crop_repair_owns_water_harvest_and_replant_suffix() {
  Compiler compiler({true, 0, 2});
  const Position tile{4, 4};
  compiler.install_day(day(
      12, {actor_plan(0, {plant(21), pass(), pass(), pass(), pass(), pass(),
                          pass()})}));

  auto run = [&](int turn, TileObservation before, int inventory,
                 const Action& expected, TileObservation after,
                 int inventory_delta = 0) {
    auto decision = compiler.decide(
        {12, turn,
         {observed(0, tile, before.kind,
                   before.item == 0 ? 21 : before.item, inventory,
                   before.watered_today, before.harvest_legal)}});
    check(decision.actions[0] == expected,
          std::string("unexpected lifecycle action at turn ") +
              std::to_string(turn));
    check(compiler.commit(decision, decision.actions),
          "lifecycle action commit failed");
    check(compiler.observe(receipt(decision, 0, tile, after, inventory_delta)),
          "lifecycle action receipt failed");
  };

  run(0, {TileKind::Weed, 21, false, false}, 1, dig(),
      {TileKind::Empty, 0, false, false});
  run(1, {TileKind::Empty, 0, false, false}, 1, plant(21),
      {TileKind::Crop, 21, false, false});
  run(2, {TileKind::Crop, 21, false, false}, 0, water(21),
      {TileKind::Crop, 21, true, false});
  run(3, {TileKind::Crop, 21, true, true}, 0, harvest(21),
      {TileKind::Empty, 0, false, false}, 1);
  check(compiler.open_transactions()[0].confirmed_harvests == 1,
        "first harvest incorrectly closed crop repair");
  run(4, {TileKind::Empty, 0, false, false}, 1, plant(21),
      {TileKind::Crop, 21, false, false});
  run(5, {TileKind::Crop, 21, false, false}, 0, water(21),
      {TileKind::Crop, 21, true, false});
  run(6, {TileKind::Crop, 21, true, true}, 0, harvest(21),
      {TileKind::Empty, 0, false, false}, 1);
  check(compiler.open_transactions().empty(),
        "replant first-yield suffix did not close crop repair");
}

void plot_debt_survives_actor_recycling_with_receipt_bound_lease() {
  Compiler compiler({true, 0, 2});
  const Position tile{2, 0};
  constexpr std::uint64_t old_generation = 0x100000002ULL;
  constexpr std::uint64_t new_generation = 0x300000002ULL;
  compiler.install_day(day(1, {actor_plan(1, {harvest(0), pass()})}));
  auto dig_decision = compiler.decide(
      {1, 0, {observed_generation(1, old_generation, tile,
                                  TileKind::Weed, 0, 1)}});
  check(dig_decision.actions[0] == dig() &&
            dig_decision.bindings[0].actor_generation == old_generation,
        "old hand generation did not own its DIG binding");
  check(compiler.commit(dig_decision, dig_decision.actions),
        "old hand generation commit failed");
  check(compiler.observe(receipt(
            dig_decision, 1, tile, {TileKind::Empty, -1, false, false},
            0, false, false, old_generation)),
        "old hand generation DIG receipt failed");
  const auto transaction_id = compiler.open_transactions()[0].id;
  const auto source_epoch = compiler.open_transactions()[0].source_epoch;

  compiler.install_day(day(3, {actor_plan(1, {pass(), pass()})}));
  auto leased = compiler.decide(
      {3, 0, {observed_generation(1, new_generation, tile,
                                  TileKind::Empty, 0, 1)}});
  check(leased.actions[0] == plant(0) && leased.bindings.size() == 1 &&
            leased.bindings[0].transaction_id == transaction_id &&
            leased.bindings[0].actor_generation == new_generation &&
            compiler.open_transactions()[0].source_epoch == source_epoch &&
            compiler.open_transactions()[0].key.actor_generation ==
                old_generation,
        "new generation did not take a receipt-bound lease on plot debt");
  check(compiler.commit(leased, leased.actions),
        "new generation plot lease did not commit");
  check(compiler.audit().plot_leases == 2,
        "plot execution leases were not audited independently of actors");
  check(!compiler.observe(receipt(
            leased, 1, tile, {TileKind::Crop, 0, false, false}, 0,
            false, false, old_generation)),
        "stale generation receipt confirmed a new lease");
}

void incompatible_crop_intent_supersedes_old_plot_objective() {
  Compiler compiler({true, 0, 2});
  const Position tile{5, 5};
  compiler.install_day(day(1, {actor_plan(0, {harvest(0)})}));
  (void)compiler.decide(
      {1, 0, {observed(0, tile, TileKind::Weed, 0, 1)}});

  compiler.install_day(day(2, {actor_plan(0, {harvest(1)})}));
  (void)compiler.decide(
      {2, 0, {observed(0, tile, TileKind::Weed, 1, 1)}});
  const auto open = compiler.open_transactions();
  check(open.size() == 1 && open[0].source_epoch == 2 &&
            open[0].key.tile == tile && open[0].desired_item == 1 &&
            compiler.audit().objectives_superseded == 1,
        "new crop intent did not explicitly supersede the old objective");
}

void repeated_failed_sources_coalesce_to_one_plot_objective() {
  Compiler compiler({true, 0, 2});
  const Position tile{6, 6};
  constexpr int kDays = 12;
  for (int current_day = 0; current_day < kDays; ++current_day) {
    compiler.install_day(day(
        current_day, {actor_plan(0, {harvest(0), pass()})}));
    (void)compiler.decide(
        {current_day, 0, {observed(0, tile, TileKind::Weed, 0, 1)}});
  }
  const auto open = compiler.open_transactions();
  check(open.size() == 1 && open[0].outstanding_source_actions == 1 &&
            compiler.audit().source_debts_created == 1 &&
            compiler.audit().source_intents_coalesced == kDays - 1,
        "repeated failed sources accumulated instead of coalescing");
}

}  // namespace

int main() {
  try {
    no_trigger_parity();
    enabled_no_trigger_719_step_multi_actor_bit_parity();
    weed_with_pass_slack_and_multi_actor_isolation();
    weed_before_move_never_leases_or_delays_move();
    harvest_on_weed_recompiles_from_observation_without_losing_route();
    water_on_weed_recompiles_and_preserves_other_critical_actions();
    state_recompile_without_future_slack_fails_closed_across_day();
    certified_production_is_never_an_absorption_sink();
    post_replant_immature_harvest_remains_source_debt_until_mature();
    already_watered_and_nonempty_plant_are_not_false_successes();
    day_rollover_preserves_debt();
    failed_seed_retries_on_later_slack();
    no_available_slack_is_strict_fail_closed();
    wheat_zero_item_id_survives_weed_transaction();
    structure_repair_is_local_and_receipt_driven();
    crop_repair_owns_water_harvest_and_replant_suffix();
    plot_debt_survives_actor_recycling_with_receipt_bound_lease();
    incompatible_crop_intent_supersedes_old_plot_objective();
    repeated_failed_sources_coalesce_to_one_plot_objective();
    std::cout << "event-triggered local repair: 20 contract fixtures passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
