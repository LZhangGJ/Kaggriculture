#include "native_teammate.hpp"

#include <array>
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
using fastkag::TileKind;

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

PlayerAction frame(std::initializer_list<Action> units) {
  PlayerAction result;
  result.units.assign(units);
  return result;
}

NativeRepairOptions enabled() {
  NativeRepairOptions options;
  options.state_driven_local_repair = true;
  return options;
}

NativeRepairOptions day_enabled() {
  NativeRepairOptions options;
  options.day_horizon_repair = true;
  return options;
}

NativeRepairOptions day_v2_enabled() {
  NativeRepairOptions options;
  options.day_horizon_repair_v2 = true;
  return options;
}

NativeRepairOptions skeleton_v3_enabled() {
  NativeRepairOptions options;
  options.rolling_route_skeleton_v3 = true;
  return options;
}

NativeTeammateExecutor executor_for(std::vector<PlayerAction> tape) {
  NativeTapeLibrary library;
  library.routes.push_back(std::move(tape));
  return NativeTeammateExecutor(std::move(library));
}

void step_passes(Simulator& simulator, int target_step) {
  while (simulator.step_count() < target_step)
    simulator.step(std::array<PlayerAction, 2>{});
}

void minimum_loss_weed_uses_only_pass_or_legacy_fallback() {
  const auto run = [](Op trigger, bool pass_before_move, int start,
                      bool early_calibration, bool expect_safe) {
    Config config;
    config.episode_steps = start + 48;
    config.weed_spawn_chance = 1.0;
    Simulator simulator(config, (pass_before_move ? 0xA551ULL : 0xA552ULL) + start);
    std::array<PlayerAction, 2> buy{};
    buy[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
    simulator.step(buy);
    step_passes(simulator, start);
    std::vector<PlayerAction> tape(config.episode_steps);
    for (auto& action : tape) {
      action.units.resize(1);
      action.units[0] = {Op::WATER, Item::WHEAT, 1};
    }
    tape[start] = frame({{trigger, Item::WHEAT, 1}});
    tape[start + (pass_before_move ? 2 : 3)] =
        frame({{Op::PASS, Item::NONE, 1}});
    tape[start + (pass_before_move ? 3 : 2)] =
        frame({{Op::EAST, Item::NONE, 1}});
    tape[start + 9] = frame({{Op::EAST, Item::NONE, 1}});
    auto executor = executor_for(std::move(tape));
    NativeAgentState state;
    NativeRepairAudit audit;
    NativeRepairOptions options;
    options.weed_min_loss_realign = true;
    if (early_calibration) options.minimum_weed_realign_step = 0;
    std::vector<Op> emitted;
    for (int step = start; step <= start + 10; ++step) {
      std::array<PlayerAction, 2> action{};
      action[0] = executor.action_external(
          simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
          nullptr, nullptr, true, options, &audit);
      emitted.push_back(action[0].units[0].op);
      if (step == start) {
        check(action[0].units[0].op == Op::DIG,
              "minimum-loss weed branch did not open with DIG");
        check(expect_safe ? state.experimental_realign[0].active
                          : state.weed[0].active,
              "minimum-loss weed branch selected the wrong owner");
      }
      simulator.step(action);
    }
    check(emitted[1] == trigger,
          "weed repair did not replay the intended action");
    if (expect_safe)
      check(emitted[3] == Op::EAST,
            "PASS-before-MOVE absorption changed the MOVE's absolute step");
    else
      check(emitted[2] != Op::EAST && !state.experimental_realign[0].active,
            "PASS beyond the next MOVE was not rejected");
  };
  run(Op::PLANT, true, 24, true, true);
  run(Op::PLANT, false, 24, true, false);
  run(Op::PLANT, true, 24, false, false);
  run(Op::BUILD_PASTURE, true, 24, false, true);
  run(Op::PLANT, true, 240, false, true);

  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xA553ULL);
  step_passes(simulator, 24);
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) {
    action.units.resize(1);
    action.units[0] = {Op::WATER, Item::WHEAT, 1};
  }
  tape[24] = frame({{Op::PLANT, Item::WHEAT, 1}});
  tape[33] = frame({{Op::EAST, Item::NONE, 1}});
  tape[27].market.assign(5, {Op::HIRE, Item::NONE, 1});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairOptions options;
  options.weed_min_loss_realign = true;
  std::array<PlayerAction, 2> action{};
  action[0] = executor.action_external(simulator, 0, 0, state,
      fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true, options);
  check(options.minimum_weed_realign_step == 240 && state.weed[0].active &&
            !state.experimental_realign[0].active,
        "default early-stage guard did not select legacy fallback");
  simulator.step(action);
  action[0] = executor.action_external(simulator, 0, 0, state,
      fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true, options);
  check(!state.weed[0].active && action[0].units[0].op == Op::WATER,
        "mask-1 fallback did not honor the farmer barrier");
}

void forced_weed_never_leases_move_and_resumes_on_return_pass() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xD311A9ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  step_passes(simulator, 24);

  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  // HARVEST-on-weed bypasses the pre-existing legacy PLANT/BUILD weed shim,
  // forcing this final composer itself to own the transaction.
  tape[24] = frame({{Op::HARVEST, Item::WHEAT, 1}});
  tape[25] = frame({{Op::EAST, Item::NONE, 1}});
  tape[26] = frame({{Op::WEST, Item::NONE, 1}});
  tape[27] = frame({{Op::PASS, Item::NONE, 1}});
  tape[28] = frame({{Op::PASS, Item::NONE, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;

  std::array<PlayerAction, 2> actions{};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::DIG,
        "forced weed did not insert DIG");
  simulator.step(actions);

  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::EAST,
        "forced weed leased the first MOVE, got op=" +
            std::to_string(static_cast<int>(actions[0].units[0].op)) +
            " seeds=" + std::to_string(simulator.privates()[0].seeds[0]) +
            " open=" + std::to_string(
                state.experimental_event_local_repair->open_transactions().size()) +
            " receipts=" +
            std::to_string(audit.local_repair_receipts_confirmed));
  simulator.step(actions);

  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::WEST,
        "forced weed leased the return MOVE, got op=" +
            std::to_string(static_cast<int>(actions[0].units[0].op)));
  simulator.step(actions);

  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::PLANT &&
            state.experimental_event_local_repair->open_transactions()[0]
                    .outstanding_source_actions == 1,
        "return-tile PASS did not resume PLANT while keeping HARVEST debt; op=" +
            std::to_string(static_cast<int>(actions[0].units[0].op)) +
            " outstanding=" + std::to_string(
                state.experimental_event_local_repair->open_transactions()[0]
                    .outstanding_source_actions));
  simulator.step(actions);
  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::WATER,
        "return-tile lifecycle did not WATER after confirmed PLANT");
  simulator.step(actions);
  actions = {};
  (void)executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(audit.local_repair_decisions == 6 &&
            audit.local_repair_commits == 6 &&
            audit.local_repair_receipts_confirmed == 3,
        "real composer path did not expose nonzero local-repair receipts");
}

void insufficient_day_budget_carries_one_debt_across_midnight() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xC2055DA7ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::CARROT, 1});
  simulator.step(opening);
  step_passes(simulator, 46);

  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[46] = frame({{Op::HARVEST, Item::CARROT, 1}});
  tape[47] = frame({{Op::EAST, Item::NONE, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;

  std::array<PlayerAction, 2> actions{};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::DIG &&
            audit.local_repair_fail_closed == 0,
        "blocked HARVEST was not recompiled in place without MOVE slack");
  check(state.experimental_event_local_repair->open_transactions().size() == 1,
        "in-place recompile did not persist exactly one transaction");
  simulator.step(actions);

  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::EAST,
        "no-slack repair swallowed final route MOVE");
  simulator.step(actions);

  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::DIG,
        "midnight debt did not retry DIG after spawn reset");
  check(state.experimental_event_local_repair->open_transactions().size() == 1,
        "midnight duplicated or lost the transaction");
  simulator.step(actions);

  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::PLANT,
        "midnight debt did not replay PLANT after confirmed DIG");
}

void failed_seed_purchase_retries_after_cash_recovers() {
  Config config;
  config.episode_steps = 48;
  config.starting_money = 100;
  config.weed_spawn_chance = 0.0;
  Simulator simulator(config, 0x5EEDFA11ULL);

  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[0].units[0] = {Op::PLANT, Item::MELON, 1};
  tape[0].market = {{Op::BUY_PRODUCT, Item::WHEAT, 2},
                    {Op::BUY_SEED, Item::MELON, 1}};
  tape[1].market = {{Op::SELL, Item::WHEAT, 2}};
  // This hard slot must remain ahead of the recovery proposal after cash
  // returns; recovery owns only the final market tail.
  tape[2].market = {{Op::SELL, Item::WHEAT, 1}};
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;

  bool retry_seen = false;
  bool retry_was_affordable = false;
  bool hard_slots_preserved = false;
  std::string market_trace;
  while (!simulator.done() && simulator.step_count() < 30) {
    if (simulator.step_count() == 2)
      retry_was_affordable = simulator.farms()[0].money >= 80;
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, enabled(), &audit);
    if (simulator.step_count() == 0 || simulator.step_count() == 2) {
      market_trace += " step" + std::to_string(simulator.step_count()) + ":";
      for (const auto& order : actions[0].market)
        market_trace += "(" + std::to_string(static_cast<int>(order.op)) +
            "," + std::to_string(static_cast<int>(order.item)) + "," +
            std::to_string(order.quantity) + ")";
    }
    if (simulator.step_count() == 0)
      hard_slots_preserved = actions[0].market.size() == 2 &&
          actions[0].market[0].op == Op::BUY_PRODUCT &&
          actions[0].market[0].item == Item::WHEAT &&
          actions[0].market[1].op == Op::BUY_SEED &&
          actions[0].market[1].item == Item::MELON;
    if (simulator.step_count() == 2)
      hard_slots_preserved = hard_slots_preserved &&
          actions[0].market.size() == 2 &&
          actions[0].market[0].op == Op::SELL &&
          actions[0].market[0].item == Item::WHEAT &&
          actions[0].market[1].op == Op::BUY_SEED &&
          actions[0].market[1].item == Item::MELON;
    if (simulator.step_count() >= 2)
      for (const auto& order : actions[0].market)
        retry_seen = retry_seen ||
            (order.op == Op::BUY_SEED && order.item == Item::MELON);
    simulator.step(actions);
    if (simulator.step_count() == 1)
      check(simulator.last_market_fills()[0].size() == 2 &&
                simulator.last_market_fills()[0][1] == 0,
            "seed-purchase witness did not force an exact zero fill");
  }
  check(retry_was_affordable,
        "cash did not recover enough before the MELON retry decision");
  check(hard_slots_preserved,
        "purchase recovery replaced or reordered a hard market slot:" +
            market_trace);
  check(retry_seen,
        "zero-fill MELON purchase was not retried after cash recovered");
  check(audit.local_repair_seed_orders == 2 &&
            audit.local_repair_seed_units == 2 &&
            audit.local_repair_seed_fills == 1 &&
            audit.local_repair_seed_zero_fills == 1,
        "seed purchase receipts did not distinguish zero-fill from retry fill");
  check(state.experimental_event_local_repair->open_transactions().size() == 1 &&
            state.experimental_event_local_repair->open_transactions()[0]
                    .outstanding_source_actions == 1,
        "filled retry incorrectly closed the unfinished crop objective");
  const auto purchase_debts =
      state.experimental_event_local_purchase_ledger.debts();
  check(purchase_debts.size() == 1 &&
            purchase_debts[0].attempts == 2 &&
            purchase_debts[0].filled == 1 &&
            purchase_debts[0].acquisition_complete &&
            state.experimental_event_local_purchase_ledger.audit()
                    .inventory_units_handed_off == 1,
        "exact-slot fill did not produce one InventoryOnly unit handoff");
}

void two_workers_do_not_overbook_one_shared_seed() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0x2A11C701ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  step_passes(simulator, 24);

  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(2);
  tape[24].market.push_back({Op::HIRE, Item::NONE, 1});
  tape[25] = frame({{Op::PASS, Item::NONE, 1},
                    {Op::WEST, Item::NONE, 1}});
  tape[26] = frame({{Op::PASS, Item::NONE, 1},
                    {Op::NORTH, Item::NONE, 1}});
  tape[27] = frame({{Op::HARVEST, Item::WHEAT, 1},
                    {Op::HARVEST, Item::WHEAT, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;

  std::array<PlayerAction, 2> actions{};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  simulator.step(actions);
  check(simulator.farms()[0].hands.size() == 1,
        "two-worker witness failed to hire its second actor");

  for (int movement_step = 0; movement_step < 2; ++movement_step) {
    actions = {};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, enabled(), &audit);
    simulator.step(actions);
  }
  check(simulator.farms()[0].farmer.x != simulator.farms()[0].hands[0].x ||
            simulator.farms()[0].farmer.y != simulator.farms()[0].hands[0].y,
        "two-worker witness did not reach distinct unlocked tiles");

  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units.size() == 2 &&
            actions[0].units[0].op == Op::DIG &&
            actions[0].units[1].op == Op::DIG,
        "two-worker witness did not open two independent weed transactions");
  simulator.step(actions);

  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  const int plants = static_cast<int>(
      (actions[0].units[0].op == Op::PLANT) +
      (actions[0].units[1].op == Op::PLANT));
  check(plants == 1,
        "two local transactions must reserve the one shared seed exactly once; "
        "emitted PLANT count=" + std::to_string(plants));
  simulator.step(actions);

  const auto& farm = simulator.farms()[0];
  const auto farmer = farm.farmer;
  const auto hand = farm.hands[0];
  const auto at = [&](fastkag::Position position) -> const fastkag::Tile& {
    return farm.tiles[static_cast<std::size_t>(
        position.y * config.board_size + position.x)];
  };
  check((at(farmer).kind == TileKind::PLANT) +
            (at(hand).kind == TileKind::PLANT) == 1,
        "shared-seed arbitration did not make exactly one worker progress");
}

void colocated_workers_do_not_double_credit_one_tile_effect() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xC0110CA7ULL);
  step_passes(simulator, 24);

  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(2);
  tape[24].market.push_back({Op::HIRE, Item::NONE, 1});
  tape[25] = frame({{Op::PASS, Item::NONE, 1},
                    {Op::WEST, Item::NONE, 1}});
  tape[26] = frame({{Op::HARVEST, Item::WHEAT, 1},
                    {Op::HARVEST, Item::WHEAT, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;

  std::array<PlayerAction, 2> actions{};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  simulator.step(actions);
  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  simulator.step(actions);
  check(simulator.farms()[0].hands.size() == 1 &&
            simulator.farms()[0].farmer.x == simulator.farms()[0].hands[0].x &&
            simulator.farms()[0].farmer.y == simulator.farms()[0].hands[0].y,
        "colocated-worker witness did not converge on one tile");

  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  const int digs = static_cast<int>((actions[0].units[0].op == Op::DIG) +
                                    (actions[0].units[1].op == Op::DIG));
  check(digs == 1,
        "one weed effect cannot causally confirm two colocated DIG bindings; "
        "emitted DIG count=" + std::to_string(digs));
}

void baseline_plant_keeps_priority_over_repair_plant_seed_demand() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xBA5E11AEULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  step_passes(simulator, 24);

  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(2);
  tape[24].market.push_back({Op::HIRE, Item::NONE, 1});
  tape[25] = frame({{Op::PASS, Item::NONE, 1},
                    {Op::WEST, Item::NONE, 1}});
  tape[26] = frame({{Op::PASS, Item::NONE, 1},
                    {Op::NORTH, Item::NONE, 1}});
  tape[27] = frame({{Op::DIG, Item::NONE, 1},
                    {Op::HARVEST, Item::WHEAT, 1}});
  tape[28] = frame({{Op::PLANT, Item::WHEAT, 1},
                    {Op::PASS, Item::NONE, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;

  while (simulator.step_count() < 27) {
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, enabled(), &audit);
    simulator.step(actions);
  }
  std::array<PlayerAction, 2> actions{};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::DIG &&
            actions[0].units[1].op == Op::DIG,
        "mixed-demand witness did not open only the repair actor transaction");
  simulator.step(actions);

  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled(), &audit);
  check(actions[0].units[0].op == Op::PLANT &&
            actions[0].units[1].op != Op::PLANT,
        "repair PLANT overbooked or displaced the one-seed baseline PLANT");
  simulator.step(actions);
  const auto& farm = simulator.farms()[0];
  const auto& farmer_tile = farm.tiles[static_cast<std::size_t>(
      farm.farmer.y * config.board_size + farm.farmer.x)];
  const auto& hand_tile = farm.tiles[static_cast<std::size_t>(
      farm.hands[0].y * config.board_size + farm.hands[0].x)];
  check(farmer_tile.kind == TileKind::PLANT &&
            hand_tile.kind == TileKind::EMPTY,
        "mixed baseline+repair seed arbitration made the wrong actor progress");
}

void day_horizon_mask_and_no_trigger_719_parity() {
  const auto bit16 = fastkag::native_repair_options_from_mask(16);
  const auto bit8 = fastkag::native_repair_options_from_mask(8);
  check(bit16.day_horizon_repair && !bit16.state_driven_local_repair &&
            bit8.state_driven_local_repair && !bit8.day_horizon_repair &&
            !fastkag::native_repair_options_from_mask(7).day_horizon_repair,
        "bit16 native mask ABI is not isolated");
  Config config;
  config.episode_steps = 720;
  config.weed_spawn_chance = 0.0;
  Simulator simulator(config, 0x160016ULL);
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  auto executor = executor_for(std::move(tape));
  NativeAgentState baseline_state;
  NativeAgentState enabled_state;
  NativeAgentState v2_state;
  NativeAgentState v3_state;
  for (int step = 0; step < 719; ++step) {
    const auto baseline = executor.action_external(
        simulator, 0, 0, baseline_state,
        fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, {}, nullptr);
    NativeRepairAudit audit;
    const auto treatment = executor.action_external(
        simulator, 0, 0, enabled_state,
        fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, bit16, &audit);
    NativeRepairAudit v2_audit;
    const auto v2 = executor.action_external(
        simulator, 0, 0, v2_state,
        fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, day_v2_enabled(), &v2_audit);
    NativeRepairAudit v3_audit;
    const auto v3 = executor.action_external(
        simulator, 0, 0, v3_state,
        fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, skeleton_v3_enabled(), &v3_audit);
    check(baseline.units.size() == treatment.units.size() &&
              baseline.market.size() == treatment.market.size() &&
              baseline.units.size() == v2.units.size() &&
              baseline.market.size() == v2.market.size() &&
              baseline.units.size() == v3.units.size() &&
              baseline.market.size() == v3.market.size(),
          "bit16 no-trigger changed manifest width");
    for (std::size_t actor = 0; actor < baseline.units.size(); ++actor) {
      check(baseline.units[actor].op == treatment.units[actor].op &&
                baseline.units[actor].item == treatment.units[actor].item &&
                baseline.units[actor].quantity ==
                    treatment.units[actor].quantity &&
                baseline.units[actor].op == v2.units[actor].op &&
                baseline.units[actor].item == v2.units[actor].item &&
                baseline.units[actor].quantity == v2.units[actor].quantity &&
                baseline.units[actor].op == v3.units[actor].op &&
                baseline.units[actor].item == v3.units[actor].item &&
                baseline.units[actor].quantity == v3.units[actor].quantity,
            "bit16/32 no-trigger changed a unit field");
    }
    for (std::size_t slot = 0; slot < baseline.market.size(); ++slot)
      check(baseline.market[slot].op == treatment.market[slot].op &&
                baseline.market[slot].item == treatment.market[slot].item &&
                baseline.market[slot].quantity ==
                    treatment.market[slot].quantity &&
                baseline.market[slot].op == v2.market[slot].op &&
                baseline.market[slot].item == v2.market[slot].item &&
                baseline.market[slot].quantity == v2.market[slot].quantity &&
                baseline.market[slot].op == v3.market[slot].op &&
                baseline.market[slot].item == v3.market[slot].item &&
                baseline.market[slot].quantity == v3.market[slot].quantity,
            "bit16/32 no-trigger changed a market field");
    check(audit.day_horizon_plans == 0 &&
              audit.day_horizon_commits == 0 &&
              v2_audit.day_horizon_v2_plans == 0 &&
              v2_audit.day_horizon_v2_replans == 0 &&
              v3_audit.route_skeleton_v3_fail_closed == 0,
          "bit16/32 no-trigger staged an owner");
    simulator.step({baseline, PlayerAction{}});
  }
}

void day_horizon_forced_weed_compiles_production_before_moves() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xD416ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  step_passes(simulator, 24);
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[24] = frame({{Op::HARVEST, Item::WHEAT, 1}});
  tape[25] = frame({{Op::EAST, Item::NONE, 1}});
  tape[26] = frame({{Op::WEST, Item::NONE, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;
  const std::array<Op, 5> expected{
      Op::DIG, Op::PLANT, Op::WATER, Op::EAST, Op::WEST};
  for (const auto operation : expected) {
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state,
        fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, day_enabled(), &audit);
    check(actions[0].units[0].op == operation,
          "bit16 forced day manifest emitted op=" +
              std::to_string(static_cast<int>(actions[0].units[0].op)) +
              " expected=" + std::to_string(static_cast<int>(operation)) +
              " plans=" + std::to_string(audit.day_horizon_plans) +
              " fail=" + std::to_string(audit.day_horizon_fail_closed) +
              " code=" +
              std::to_string(state.experimental_day_horizon.failure_code) +
              " assignments=" +
              std::to_string(audit.day_horizon_assignments));
    simulator.step(actions);
  }
  check(audit.day_horizon_plans == 1 &&
            audit.day_horizon_assignments == 3 &&
            audit.day_horizon_fail_closed == 0,
        "bit16 forced day did not commit one safe compiled manifest");
}

void day_horizon_failed_receipt_cancels_suffix_without_cross_actor_replay() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xFA1616ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  step_passes(simulator, 24);
  std::array<PlayerAction, 2> hire{};
  hire[0].market.push_back({Op::HIRE, Item::NONE, 1});
  simulator.step(hire);
  check(simulator.farms()[0].hands.size() == 1,
        "failed-receipt witness could not hire second actor");

  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(2);
  tape[25] = frame({{Op::HARVEST, Item::WHEAT, 1},
                    {Op::EAST, Item::NONE, 1}});
  tape[26] = frame({{Op::EAST, Item::NONE, 1},
                    {Op::WEST, Item::NONE, 1}});
  tape[27] = frame({{Op::WEST, Item::NONE, 1},
                    {Op::EAST, Item::NONE, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;

  std::array<PlayerAction, 2> proposed{};
  proposed[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, day_enabled(), &audit);
  check(proposed[0].units[0].op == Op::DIG,
        "failed-receipt witness did not stage planner DIG");
  // Inject an exact receipt failure while leaving the unrelated actor action
  // untouched. The next observation must cancel the whole future suffix.
  proposed[0].units[0] = {};
  simulator.step(proposed);

  std::array<PlayerAction, 2> after_failure{};
  after_failure[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, day_enabled(), &audit);
  check(after_failure[0].units[0].op == Op::EAST,
        "failed DIG receipt allowed future PLANT/WATER suffix to continue");
  check(after_failure[0].units[0].op != Op::HARVEST,
        "failed invalid HARVEST source was replayed");
  check(after_failure[0].units[1].op == Op::WEST,
        "receipt failure polluted another actor/tile raw action");
  simulator.step(after_failure);

  const auto next = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, day_enabled(), &audit);
  check(next.units[0].op == Op::WEST && next.units[1].op == Op::EAST,
        "failed-closed runtime did not preserve the later raw suffix");
  check(state.experimental_day_horizon.failed_closed &&
            state.experimental_day_horizon.failure_code == 4 &&
            audit.day_horizon_plans == 1 &&
            audit.day_horizon_fail_closed == 1,
        "failed receipt did not latch one day-scoped fail-closed witness");
}

void day_horizon_v2_mask_and_receipt_replans_forced_suffix() {
  const auto bit32 = fastkag::native_repair_options_from_mask(32);
  check(bit32.day_horizon_repair_v2 && !bit32.day_horizon_repair &&
            !bit32.state_driven_local_repair &&
            !fastkag::native_repair_options_from_mask(7)
                 .day_horizon_repair_v2,
        "bit32 native mask ABI is not isolated");
  Config config;
  config.episode_steps = 96;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xD432ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  step_passes(simulator, 24);
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[24] = frame({{Op::HARVEST, Item::WHEAT, 1}});
  tape[25] = frame({{Op::EAST, Item::NONE, 1}});
  tape[26] = frame({{Op::WEST, Item::NONE, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;
  const std::array<Op, 5> expected{
      Op::DIG, Op::PLANT, Op::WATER, Op::EAST, Op::WEST};
  for (const auto operation : expected) {
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, day_v2_enabled(), &audit);
    check(actions[0].units[0].op == operation,
          "bit32 receipt-replanned suffix emitted op=" +
              std::to_string(static_cast<int>(actions[0].units[0].op)) +
              " expected=" + std::to_string(static_cast<int>(operation)) +
              " plans=" + std::to_string(audit.day_horizon_v2_plans) +
              " replans=" + std::to_string(audit.day_horizon_v2_replans) +
              " assignments=" +
              std::to_string(audit.day_horizon_v2_assignments) +
              " unscheduled=" + std::to_string(
                  state.experimental_day_horizon_v2.plan
                      .unscheduled_objectives.size()) +
              " fail=" + std::to_string(audit.day_horizon_v2_fail_closed));
    simulator.step(actions);
  }
  check(audit.day_horizon_v2_plans == 1 &&
            audit.day_horizon_v2_replans >= 3 &&
            audit.day_horizon_v2_receipts_confirmed == 3 &&
            audit.day_horizon_v2_receipts_failed == 0 &&
            audit.day_horizon_v2_maturity_waits == 1 &&
            audit.day_horizon_v2_terminal_raw == 0,
        "bit32 did not re-solve every confirmed production receipt");
}

void day_horizon_v2_carries_watered_immature_to_one_shot_harvest() {
  Config config;
  config.episode_steps = 120;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xD43277ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  step_passes(simulator, 24);
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[24] = frame({{Op::HARVEST, Item::WHEAT, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;
  for (const auto expected : {Op::DIG, Op::PLANT, Op::WATER}) {
    std::array<PlayerAction, 2> action{};
    action[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, day_v2_enabled(), &audit);
    check(action[0].units[0].op == expected,
          "bit32 maturity setup did not replant and water");
    simulator.step(action);
  }
  // No HARVEST proof exists at age zero. The objective must survive the day,
  // then maintain WATERED_IMMATURE once on day two.
  step_passes(simulator, 48);
  std::array<PlayerAction, 2> day_two{};
  day_two[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, day_v2_enabled(), &audit);
  check(day_two[0].units[0].op == Op::WATER,
        "bit32 WATERED_IMMATURE carry omitted daily maintenance");
  simulator.step(day_two);
  static_cast<void>(executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, day_v2_enabled(), &audit));
  step_passes(simulator, 72);
  std::array<PlayerAction, 2> mature{};
  mature[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, day_v2_enabled(), &audit);
  check(mature[0].units[0].op == Op::HARVEST,
        "bit32 scheduled HARVEST without/current maturity proof mismatch");
  check(audit.day_horizon_v2_maturity_tokens_consumed == 1,
        "bit32 maturity proof was not consumed exactly once");
  simulator.step(mature);
  static_cast<void>(executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, day_v2_enabled(), &audit));
  check(audit.day_horizon_v2_objectives_completed == 1 &&
            state.experimental_day_horizon_v2.objectives.empty(),
        "bit32 mature HARVEST receipt did not close carried objective");
}

void day_horizon_v2_seed_zero_fill_retries_and_replans_inventory() {
  Config config;
  config.episode_steps = 48;
  config.starting_money = 100;
  config.weed_spawn_chance = 0.0;
  Simulator simulator(config, 0xB3205EEDULL);
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[0].units[0] = {Op::HARVEST, Item::MELON, 1};
  tape[0].market = {{Op::BUY_PRODUCT, Item::WHEAT, 2},
                    {Op::BUY_SEED, Item::MELON, 1}};
  tape[1].market = {{Op::SELL, Item::WHEAT, 2}};
  tape[2].market = {{Op::SELL, Item::WHEAT, 1}};
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;
  bool zero_fill = false;
  bool retry_tail = false;
  bool planted = false;
  bool watered = false;
  for (int step = 0; step < 8; ++step) {
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, day_v2_enabled(), &audit);
    if (step == 0)
      check(actions[0].market.size() == 2 &&
                actions[0].market[0].op == Op::BUY_PRODUCT &&
                actions[0].market[1].op == Op::BUY_SEED,
            "bit32 purchase ledger changed initial hard market order/slot");
    if (step == 2)
      retry_tail = actions[0].market.size() == 2 &&
          actions[0].market[0].op == Op::SELL &&
          actions[0].market[1].op == Op::BUY_SEED &&
          actions[0].market[1].item == Item::MELON;
    planted = planted || actions[0].units[0].op == Op::PLANT;
    watered = watered || actions[0].units[0].op == Op::WATER;
    simulator.step(actions);
    if (step == 0)
      zero_fill = simulator.last_market_fills()[0].size() == 2 &&
          simulator.last_market_fills()[0][1] == 0;
  }
  const auto debts = state.experimental_day_horizon_v2.purchase_ledger.debts();
  check(zero_fill && retry_tail,
        "bit32 did not preserve exact zero-fill slot then append funded retry");
  check(debts.size() == 1 && debts[0].attempts == 2 &&
            debts[0].filled == 1 && debts[0].acquisition_complete,
        "bit32 BUY_SEED exact receipt did not complete one acquisition debt");
  check(planted && watered && audit.day_horizon_v2_seed_zero_fills == 1 &&
            audit.day_horizon_v2_seed_fills == 1 &&
            audit.day_horizon_v2_replans >= 2,
        "bit32 fill handoff did not refresh resources and re-solve unit suffix");
}

void day_horizon_v2_failed_receipt_replans_instead_of_cancelling_day() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xB32FA11ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  step_passes(simulator, 24);
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[24] = frame({{Op::HARVEST, Item::WHEAT, 1}});
  tape[25] = frame({{Op::EAST, Item::NONE, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;
  std::array<PlayerAction, 2> first{};
  first[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, day_v2_enabled(), &audit);
  check(first[0].units[0].op == Op::DIG,
        "bit32 failed-receipt witness did not stage DIG");
  first[0].units[0] = {};
  simulator.step(first);
  std::array<PlayerAction, 2> retry{};
  retry[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, day_v2_enabled(), &audit);
  check(retry[0].units[0].op == Op::DIG &&
            audit.day_horizon_v2_receipts_failed == 1 &&
            audit.day_horizon_v2_replans == 1 &&
            audit.day_horizon_v2_fail_closed == 0,
        "bit32 failed receipt cancelled the day instead of re-solving suffix");
}

void route_skeleton_v3_rebases_production_around_move_sources() {
  const auto bit64 = fastkag::native_repair_options_from_mask(64);
  check(bit64.rolling_route_skeleton_v3 &&
            !bit64.day_horizon_repair_v2 && !bit64.day_horizon_repair &&
            !bit64.state_driven_local_repair,
        "bit64 native mask ABI is not isolated");
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xB640001ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  step_passes(simulator, 24);
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(1);
  tape[24] = frame({{Op::HARVEST, Item::WHEAT, 1}});
  tape[25] = frame({{Op::EAST, Item::NONE, 1}});
  tape[26] = frame({{Op::WEST, Item::NONE, 1}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;
  for (const auto expected :
       {Op::DIG, Op::PLANT, Op::WATER, Op::EAST, Op::WEST}) {
    std::array<PlayerAction, 2> action{};
    action[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, skeleton_v3_enabled(), &audit);
    check(action[0].units[0].op == expected,
          "bit64 rolling skeleton lost production/MOVE ordering: got=" +
              std::to_string(static_cast<int>(action[0].units[0].op)) +
              " expected=" + std::to_string(static_cast<int>(expected)) +
              " fail=" + std::to_string(audit.route_skeleton_v3_fail_closed));
    simulator.step(action);
  }
  check(audit.route_skeleton_v3_plans == 1 &&
            audit.route_skeleton_v3_replans >= 4 &&
            audit.route_skeleton_v3_rebases >= 4 &&
            audit.route_skeleton_v3_receipts_confirmed == 3 &&
            audit.route_skeleton_v3_moves_emitted == 2 &&
            audit.route_skeleton_v3_terminal_moves == 0 &&
            audit.route_skeleton_v3_fail_closed == 0,
        "bit64 rebase audit did not conserve receipt and MOVE skeleton");
}

void route_skeleton_v3_hire_gets_new_generation_without_old_raw_debt() {
  Config config;
  config.episode_steps = 48;
  config.weed_spawn_chance = 0.0;
  Simulator simulator(config, 0xB64A11ULL);
  std::vector<PlayerAction> tape(config.episode_steps);
  tape[0].units.resize(1);
  tape[0].market.push_back({Op::HIRE, Item::NONE, 1});
  for (int step = 1; step < config.episode_steps; ++step)
    tape[static_cast<std::size_t>(step)].units.resize(2);
  tape[1].units[1] = {Op::EAST, Item::NONE, 1};
  tape[2].units[1] = {Op::WEST, Item::NONE, 1};
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;
  std::array<PlayerAction, 2> hire{};
  hire[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, skeleton_v3_enabled(), &audit);
  check(hire[0].market.size() == 1 && hire[0].market[0].op == Op::HIRE,
        "bit64 changed hard HIRE baseline slot");
  simulator.step(hire);
  check(simulator.farms()[0].hands.size() == 1,
        "bit64 HIRE witness did not create a new generation");
  for (const auto expected : {Op::EAST, Op::WEST}) {
    std::array<PlayerAction, 2> action{};
    action[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, skeleton_v3_enabled(), &audit);
    check(action[0].units.size() == 2 &&
              action[0].units[1].op == expected,
          "bit64 new hand did not receive its own MOVE source skeleton");
    simulator.step(action);
  }
  check(audit.route_skeleton_v3_fail_closed == 0 &&
            state.experimental_route_skeleton_v3.actors.size() == 2 &&
            state.experimental_route_skeleton_v3.actors[1].generation !=
                state.experimental_route_skeleton_v3.actors[0].generation,
        "bit64 actor-generation rebase inherited an old raw queue");
}

void route_skeleton_v3_rejection_rolls_back_all_actor_proposals() {
  Config config;
  config.episode_steps = 72;
  config.weed_spawn_chance = 1.0;
  Simulator simulator(config, 0xB64A701ULL);
  step_passes(simulator, 24);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::HIRE, Item::NONE, 1});
  simulator.step(opening);
  check(simulator.farms()[0].hands.size() == 1,
        "bit64 rollback witness could not create second actor");
  std::vector<PlayerAction> tape(config.episode_steps);
  for (auto& action : tape) action.units.resize(2);
  tape[25] = frame({{Op::HARVEST, Item::WHEAT, 1},
                    {Op::FEED, Item::WHEAT, 1}});
  tape[26] = frame({{Op::EAST, Item::NONE, 1}, {}});
  auto executor = executor_for(std::move(tape));
  NativeAgentState state;
  NativeRepairAudit audit;
  std::array<PlayerAction, 2> rejected{};
  rejected[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, skeleton_v3_enabled(), &audit);
  check(rejected[0].units[0].op == Op::HARVEST &&
            rejected[0].units[1].op == Op::FEED &&
            state.experimental_route_skeleton_v3.objectives.empty() &&
            state.experimental_route_skeleton_v3.actors.empty(),
        "bit64 rejected multi-actor proposal leaked staged runtime state op0=" +
            std::to_string(static_cast<int>(rejected[0].units[0].op)) +
            " op1=" +
            std::to_string(static_cast<int>(rejected[0].units[1].op)) +
            " objectives=" + std::to_string(
                state.experimental_route_skeleton_v3.objectives.size()) +
            " actors=" + std::to_string(
                state.experimental_route_skeleton_v3.actors.size()));
  simulator.step(rejected);
  const auto next = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, skeleton_v3_enabled(), &audit);
  check(next.units[0].op == Op::EAST &&
            state.experimental_route_skeleton_v3.objectives.empty() &&
            audit.route_skeleton_v3_fail_reasons[2] == 1,
        "bit64 rolled-back objective later overwrote a MOVE source");
}

}  // namespace

int main() {
  try {
    minimum_loss_weed_uses_only_pass_or_legacy_fallback();
    forced_weed_never_leases_move_and_resumes_on_return_pass();
    insufficient_day_budget_carries_one_debt_across_midnight();
    failed_seed_purchase_retries_after_cash_recovers();
    two_workers_do_not_overbook_one_shared_seed();
    colocated_workers_do_not_double_credit_one_tile_effect();
    baseline_plant_keeps_priority_over_repair_plant_seed_demand();
    day_horizon_mask_and_no_trigger_719_parity();
    day_horizon_forced_weed_compiles_production_before_moves();
    day_horizon_failed_receipt_cancels_suffix_without_cross_actor_replay();
    day_horizon_v2_mask_and_receipt_replans_forced_suffix();
    day_horizon_v2_carries_watered_immature_to_one_shot_harvest();
    day_horizon_v2_seed_zero_fill_retries_and_replans_inventory();
    day_horizon_v2_failed_receipt_replans_instead_of_cancelling_day();
    route_skeleton_v3_rebases_production_around_move_sources();
    route_skeleton_v3_hire_gets_new_generation_without_old_raw_debt();
    route_skeleton_v3_rejection_rolls_back_all_actor_proposals();
    std::cout << "native repair: 17 adversarial fixtures passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
