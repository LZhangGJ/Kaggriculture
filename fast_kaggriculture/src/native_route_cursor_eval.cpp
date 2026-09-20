#include "native_teammate.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using fastkag::Action;
using fastkag::NativeAgentState;
using fastkag::NativeRepairAudit;
using fastkag::NativeRepairOptions;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using Tape = std::vector<PlayerAction>;

struct Options {
  std::uint64_t seed_begin{993001};
  int seeds{32};
  int threads{static_cast<int>(std::max(1u,std::thread::hardware_concurrency()))};
  std::string output{"route-cursor-online-dev.json"};
  std::string tapes{NATIVE_RC_TAPES};
  std::string library{NATIVE_RC_LIBRARY};
};

struct Metrics {
  double own{}, opponent{};
  int unit_failures{}, market_failures{};
  int ordered_move_day_failures{}, absolute_move_mismatches{};
  int terminal_active_crop_debts{}, terminal_deferred_nonmoves{};
  int terminal_critical_deferred_nonmoves{};
  int terminal_expired_critical_deferred_nonmoves{};
  int terminal_pending_route_receipt{};
  int terminal_pending_critical_receipt{};
  int crop_scheduler_open_debts{},crop_scheduler_sources_enqueued{};
  int crop_scheduler_sources_coalesced{},crop_scheduler_replacements{};
  int crop_scheduler_existing_equivalents{};
  int crop_scheduler_receipt_successes{},crop_scheduler_receipt_failures{};
  int crop_scheduler_completed_obligations{},crop_scheduler_completed_debts{};
  int crop_scheduler_overdue_obligations{};
  std::array<int, 15> source_production_ops{};
  std::array<int, 15> emitted_production_ops{};
  std::array<int, 15> terminal_deferred_ops{};
  int first_ordered_failure_day{-1},first_ordered_failure_actor{-1};
  int first_expected_moves{},first_actual_moves{};
  std::uint64_t action_nanoseconds{}, action_calls{};
  std::uint64_t maximum_action_nanoseconds{};
  NativeRepairAudit audit{};
  int score() const { return own>opponent?2:own==opponent?1:0; }
  double margin() const { return own-opponent; }
  bool production_continuity_gate() const {
    return ordered_move_day_failures == 0 &&
           terminal_active_crop_debts == 0 &&
           terminal_critical_deferred_nonmoves == 0 &&
           terminal_pending_critical_receipt == 0;
  }
};

struct Row {
  std::string opponent;
  std::uint64_t seed{};
  int seat{};
  std::array<Metrics,4> arms{};
};

bool move(Op op) {
  return op==Op::NORTH||op==Op::SOUTH||op==Op::EAST||op==Op::WEST;
}

int production_op_index(Op op) {
  switch (op) {
    case Op::DIG: return 0;
    case Op::PLANT: return 1;
    case Op::WATER: return 2;
    case Op::HARVEST: return 3;
    case Op::FERTILIZE: return 4;
    case Op::BUILD_COOP: return 5;
    case Op::BUILD_PASTURE: return 6;
    case Op::PLACE: return 7;
    case Op::FEED: return 8;
    case Op::CARE: return 9;
    case Op::PICKUP: return 10;
    case Op::DROP: return 11;
    case Op::COLLECT_FERTILIZER: return 12;
    case Op::NORTH:
    case Op::SOUTH: return 13;
    case Op::EAST:
    case Op::WEST: return 14;
    default: return -1;
  }
}

bool critical_nonmove(Op op) {
  const int index = production_op_index(op);
  return index >= 0 && index < 13;
}

bool equal(const Action& a,const Action& b) {
  return a.op==b.op&&a.item==b.item&&a.quantity==b.quantity;
}

bool equal(const PlayerAction& a,const PlayerAction& b) {
  if(a.units.size()!=b.units.size()||a.market.size()!=b.market.size())return false;
  for(std::size_t i=0;i<a.units.size();++i)if(!equal(a.units[i],b.units[i]))return false;
  for(std::size_t i=0;i<a.market.size();++i)if(!equal(a.market[i],b.market[i]))return false;
  return true;
}

Action tape_unit(const Tape& tape,int step,std::size_t actor) {
  if(step<0||step>=static_cast<int>(tape.size())||actor>=tape[step].units.size())return {};
  return tape[step].units[actor];
}

void certify(const Tape& tape,const Tape& emitted,
             const std::vector<std::size_t>& active,Metrics& out,int turns) {
  for(std::size_t step=0;step<emitted.size()&&step<tape.size();++step)
    for(std::size_t actor=0;actor<active[step];++actor) {
      const auto expected=tape_unit(tape,static_cast<int>(step),actor);
      const auto actual=actor<emitted[step].units.size()?emitted[step].units[actor]:Action{};
      const int expected_index=production_op_index(expected.op);
      const int actual_index=production_op_index(actual.op);
      if(expected_index>=0)++out.source_production_ops[expected_index];
      if(actual_index>=0)++out.emitted_production_ops[actual_index];
      if((move(expected.op)||move(actual.op))&&!equal(expected,actual))
        ++out.absolute_move_mismatches;
    }
  for(int begin=0;begin<static_cast<int>(emitted.size());begin+=turns) {
    const int end=std::min(begin+turns,static_cast<int>(emitted.size()));
    std::size_t actors=0;
    for(int step=begin;step<end;++step)actors=std::max(actors,active[step]);
    for(std::size_t actor=0;actor<actors;++actor) {
      int first=end;
      for(int step=begin;step<end;++step)if(actor<active[step]){first=step;break;}
      std::vector<Op> expected,actual;
      for(int step=first;step<end;++step) {
        if(actor>=active[step])continue;
        const auto source=tape_unit(tape,step,actor);
        const auto actual_action=actor<emitted[step].units.size()?emitted[step].units[actor]:Action{};
        if(move(source.op))expected.push_back(source.op);
        if(move(actual_action.op))actual.push_back(actual_action.op);
      }
      if(expected!=actual) {
        ++out.ordered_move_day_failures;
        if(out.first_ordered_failure_day<0) {
          out.first_ordered_failure_day=begin/turns;
          out.first_ordered_failure_actor=static_cast<int>(actor);
          out.first_expected_moves=static_cast<int>(expected.size());
          out.first_actual_moves=static_cast<int>(actual.size());
        }
      }
    }
  }
}

Metrics run(const NativeTeammateExecutor& executor,const Tape& focal_tape,
            int opponent_route,std::uint64_t seed,int seat,int policy) {
  fastkag::Simulator sim({},seed);
  std::array<NativeAgentState,2> states;
  Metrics out;
  Tape emitted;
  std::vector<std::size_t> active;
  NativeRepairOptions repair;
  repair.route_cursor_production=policy;
  repair.animal_buy_retry=policy>=3;
  while(!sim.done()) {
    std::array<PlayerAction,2> actions;
    const auto start=std::chrono::steady_clock::now();
    actions[seat]=executor.action_external(
        sim,seat,0,states[seat],fastkag::NativeMarketArm::LegacyDefault,
        nullptr,nullptr,false,repair,&out.audit);
    const auto elapsed=static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start).count());
    out.action_nanoseconds+=elapsed;++out.action_calls;
    out.maximum_action_nanoseconds=std::max(out.maximum_action_nanoseconds,elapsed);
    actions[1-seat]=executor.action_external(sim,1-seat,opponent_route,states[1-seat]);
    active.push_back(sim.farms()[seat].hands.size()+1);
    emitted.push_back(actions[seat]);
    out.unit_failures+=fastkag::native_macro_unit_failures(sim,seat,actions[seat]);
    sim.step(actions);
    out.market_failures+=fastkag::native_macro_market_failures(sim,seat,actions[seat]);
  }
  out.own=sim.farms()[seat].money;
  out.opponent=sim.farms()[1-seat].money;
  certify(focal_tape,emitted,active,out,sim.config().turns_per_day);
  const auto& cursor=states[seat].experimental_route_cursor;
  for(const auto& debt:cursor.production.debts)
    out.terminal_active_crop_debts+=debt.active;
  for(const auto& actor_debts:cursor.deferred_nonmoves)
    for(const auto& debt:actor_debts) {
      ++out.terminal_deferred_nonmoves;
      const int index=production_op_index(debt.action.op);
      if(index>=0)++out.terminal_deferred_ops[static_cast<std::size_t>(index)];
      if(critical_nonmove(debt.action.op)) {
        ++out.terminal_critical_deferred_nonmoves;
        out.terminal_expired_critical_deferred_nonmoves+=
            debt.deadline_step>=0&&debt.deadline_step<sim.step_count();
      }
    }
  out.terminal_pending_route_receipt=
      states[seat].experimental_route_cursor_pending.active;
  if(states[seat].experimental_route_cursor_pending.active)
    out.terminal_pending_critical_receipt=std::any_of(
        states[seat].experimental_route_cursor_pending.final_units.begin(),
        states[seat].experimental_route_cursor_pending.final_units.end(),
        [](const Action& action){return critical_nonmove(action.op);});
  if(states[seat].experimental_deferred_crop_scheduler) {
    const auto& scheduler=*states[seat].experimental_deferred_crop_scheduler;
    const auto scheduler_audit=scheduler.audit(sim.step_count());
    out.crop_scheduler_open_debts=
        static_cast<int>(scheduler.open_debts(sim.step_count()).size());
    out.crop_scheduler_sources_enqueued=scheduler_audit.sources_enqueued;
    out.crop_scheduler_sources_coalesced=scheduler_audit.sources_coalesced;
    out.crop_scheduler_replacements=scheduler_audit.replacements;
    out.crop_scheduler_existing_equivalents=
        scheduler_audit.existing_equivalents;
    out.crop_scheduler_receipt_successes=scheduler_audit.receipt_successes;
    out.crop_scheduler_receipt_failures=scheduler_audit.receipt_failures;
    out.crop_scheduler_completed_obligations=
        scheduler_audit.completed_obligations;
    out.crop_scheduler_completed_debts=scheduler_audit.completed_tile_debts;
    out.crop_scheduler_overdue_obligations=
        scheduler_audit.overdue_open_obligations;
  }
  if(policy>0&&out.ordered_move_day_failures>0)
    std::cerr<<"route failure seed="<<seed<<" seat="<<seat
             <<" policy="<<policy<<" day="<<out.first_ordered_failure_day
             <<" actor="<<out.first_ordered_failure_actor
             <<" expected="<<out.first_expected_moves
             <<" actual="<<out.first_actual_moves<<'\n';
  return out;
}

void require_default_parity(const NativeTeammateExecutor& executor,
                            std::uint64_t seed) {
  const auto built_in=executor.play(0,1,seed,-1,-1,-1,-1,true);
  fastkag::Simulator sim({},seed);
  std::array<NativeAgentState,2> states;
  std::size_t step=0;
  while(!sim.done()) {
    std::array<PlayerAction,2> action{
        executor.action_external(sim,0,0,states[0]),
        executor.action_external(sim,1,1,states[1])};
    if(step>=built_in.trace.size()||!equal(action[0],built_in.trace[step][0])||
       !equal(action[1],built_in.trace[step][1]))
      throw std::runtime_error("default-off bit parity failed at step "+
                               std::to_string(step));
    sim.step(action);++step;
  }
  if(step!=built_in.trace.size())throw std::runtime_error("default-off trace length mismatch");
}

void require_event_local_enabled_no_trigger_719_multi_actor_parity() {
  fastkag::Config config;
  config.farm_hand_cost_mult = 0;
  config.weed_spawn_chance = 0.0;
  Tape tape(719);
  for (std::size_t step = 0; step < tape.size(); ++step) {
    auto& frame = tape[step];
    frame.units.resize(4);
    const int hour = static_cast<int>(step % 24);
    if (hour == 0)
      for (int hire = 0; hire < 3; ++hire)
        frame.market.push_back({Op::HIRE, Item::NONE, 1});
    for (std::size_t actor = 0; actor < frame.units.size(); ++actor) {
      if ((hour + static_cast<int>(actor)) % 5 == 0)
        frame.units[actor] = {Op::EAST, Item::WOOL,
                              static_cast<int>(actor) + 2};
      else if ((hour + static_cast<int>(actor)) % 7 == 0)
        frame.units[actor] = {Op::CARE, Item::MILK,
                              static_cast<int>(actor) + 3};
      else
        frame.units[actor] = {Op::PASS, Item::EGG,
                              static_cast<int>(actor) + 4};
    }
  }
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  fastkag::Simulator baseline_simulator(config, 0x719B17ULL);
  fastkag::Simulator enabled_simulator(config, 0x719B17ULL);
  NativeAgentState baseline_state, enabled_state;
  NativeRepairAudit audit;
  const auto enabled = fastkag::native_repair_options_from_mask(8);
  int turns = 0;
  std::size_t maximum_actors = 0;
  while (!baseline_simulator.done() && !enabled_simulator.done()) {
    std::array<PlayerAction, 2> baseline_actions{};
    std::array<PlayerAction, 2> enabled_actions{};
    baseline_actions[0] = executor.action_external(
        baseline_simulator, 0, 0, baseline_state,
        fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true);
    enabled_actions[0] = executor.action_external(
        enabled_simulator, 0, 0, enabled_state,
        fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true,
        enabled, &audit);
    maximum_actors = std::max(maximum_actors,
                              enabled_actions[0].units.size());
    if (!equal(baseline_actions[0], enabled_actions[0]))
      throw std::runtime_error(
          "enabled no-trigger native parity failed at step " +
          std::to_string(turns));
    baseline_simulator.step(baseline_actions);
    enabled_simulator.step(enabled_actions);
    ++turns;
  }
  if (turns != 719 || maximum_actors != 4 ||
      audit.local_repair_decisions != 719 ||
      audit.local_repair_commits != 719 ||
      !enabled_state.experimental_event_local_repair ||
      !enabled_state.experimental_event_local_repair->open_transactions().empty())
    throw std::runtime_error(
        "enabled no-trigger 719-step/multi-actor gate was not exercised");
}

void require_event_local_unknown_weed_source_item_drives_recompile() {
  fastkag::Config config;
  config.episode_steps = 32;
  config.weed_spawn_chance = 1.0;
  fastkag::Simulator simulator(config, 0xD3512EDULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  while (simulator.step_count() < 24) {
    std::array<PlayerAction, 2> actions{};
    simulator.step(actions);
  }
  Tape tape(static_cast<std::size_t>(config.episode_steps));
  for (auto& frame : tape) frame.units.resize(1);
  tape[24].units[0] = {Op::HARVEST, Item::WHEAT, 1};
  tape[25].units[0] = {Op::EAST, Item::NONE, 1};
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  NativeAgentState state;
  NativeRepairAudit audit;
  const auto enabled = fastkag::native_repair_options_from_mask(8);
  std::array<Op, 4> observed{};
  for (int offset = 0; offset < 4; ++offset) {
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, enabled, &audit);
    observed[static_cast<std::size_t>(offset)] = actions[0].units[0].op;
    simulator.step(actions);
  }
  const auto open = state.experimental_event_local_repair->open_transactions();
  if (observed != std::array<Op, 4>{Op::DIG, Op::EAST, Op::PASS,
                                    Op::PASS} ||
      audit.local_repair_receipts_confirmed != 1 || open.size() != 1 ||
      open[0].desired_item != static_cast<int>(Item::WHEAT) ||
      open[0].outstanding_source_actions != 1)
    throw std::runtime_error(
        "plot debt did not preserve immediate MOVE and remain durable");
}

void require_event_local_global_seed_reservation() {
  fastkag::Config config;
  config.episode_steps = 40;
  config.weed_spawn_chance = 1.0;
  config.farm_hand_cost_mult = 0;
  fastkag::Simulator simulator(config, 0x5EED2ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  while (simulator.step_count() < 24) {
    std::array<PlayerAction, 2> actions{};
    simulator.step(actions);
  }
  Tape tape(static_cast<std::size_t>(config.episode_steps));
  for (auto& frame : tape) frame.units.resize(2);
  tape[24].market.push_back({Op::HIRE, Item::NONE, 1});
  tape[25].units[1] = {Op::WEST, Item::NONE, 1};
  tape[26].units[1] = {Op::NORTH, Item::NONE, 1};
  tape[27].units[0] = {Op::HARVEST, Item::WHEAT, 1};
  tape[27].units[1] = {Op::HARVEST, Item::WHEAT, 1};
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  NativeAgentState state;
  NativeRepairAudit audit;
  const auto enabled = fastkag::native_repair_options_from_mask(8);
  std::array<PlayerAction, 2> actions{};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled, &audit);
  simulator.step(actions);  // HIRE
  for (int movement = 0; movement < 2; ++movement) {
    actions = {};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, enabled, &audit);
    simulator.step(actions);
  }
  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled, &audit);
  if (actions[0].units.size() != 2 || actions[0].units[0].op != Op::DIG ||
      actions[0].units[1].op != Op::DIG)
    throw std::runtime_error("two-actor seed reservation fixture did not DIG");
  simulator.step(actions);
  actions = {};
  actions[0] = executor.action_external(
      simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, true, enabled, &audit);
  const int proposed_plants = static_cast<int>(std::count_if(
      actions[0].units.begin(), actions[0].units.end(),
      [](const Action& action) { return action.op == Op::PLANT; }));
  simulator.step(actions);
  const int observed_plants = static_cast<int>(std::count_if(
      simulator.farms()[0].tiles.begin(), simulator.farms()[0].tiles.end(),
      [](const fastkag::Tile& tile) { return tile.kind == fastkag::TileKind::PLANT; }));
  const auto open = state.experimental_event_local_repair
      ? state.experimental_event_local_repair->open_transactions()
      : std::vector<g001::event_local_repair::TransactionView>{};
  // This assertion is deliberately scoped to the contested final manifest.
  // The market half may replenish the spent seed in the same step, but it
  // must never make two repair PLANT bindings compete for the one seed that
  // existed at arbitration time.
  if (proposed_plants != 1 || observed_plants != 1 || open.size() != 2)
    throw std::runtime_error(
        "global seed reservation did not advance exactly one actor");
}

void require_event_local_same_tile_serialization() {
  fastkag::Config config;
  config.episode_steps = 80;
  config.weed_spawn_chance = 1.0;
  config.farm_hand_cost_mult = 0;
  fastkag::Simulator simulator(config, 0x71E5EEDULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  while (simulator.step_count() < 24) {
    std::array<PlayerAction, 2> actions{};
    simulator.step(actions);
  }
  std::array<PlayerAction, 2> hire{};
  hire[0].market.push_back({Op::HIRE, Item::NONE, 1});
  simulator.step(hire);
  while (simulator.farms()[0].hands[0].x != simulator.farms()[0].farmer.x ||
         simulator.farms()[0].hands[0].y != simulator.farms()[0].farmer.y) {
    const auto farmer = simulator.farms()[0].farmer;
    const auto hand = simulator.farms()[0].hands[0];
    Action movement_action;
    if (hand.x < farmer.x) movement_action.op = Op::EAST;
    else if (hand.x > farmer.x) movement_action.op = Op::WEST;
    else if (hand.y < farmer.y) movement_action.op = Op::SOUTH;
    else movement_action.op = Op::NORTH;
    std::array<PlayerAction, 2> movement_actions{};
    movement_actions[0].units.resize(2);
    movement_actions[0].units[1] = movement_action;
    simulator.step(movement_actions);
  }
  const int trigger = simulator.step_count();
  Tape tape(static_cast<std::size_t>(config.episode_steps));
  for (auto& frame : tape) frame.units.resize(2);
  tape[static_cast<std::size_t>(trigger)].units[0] =
      {Op::HARVEST, Item::WHEAT, 1};
  tape[static_cast<std::size_t>(trigger)].units[1] =
      {Op::HARVEST, Item::WHEAT, 1};
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  NativeAgentState state;
  NativeRepairAudit audit;
  const auto enabled = fastkag::native_repair_options_from_mask(8);
  auto next = [&] {
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, enabled, &audit);
    const auto result = actions[0];
    simulator.step(actions);
    return result;
  };
  const auto dig = next();
  const auto plant = next();
  const auto water = next();
  (void)next();  // settle WATER
  const auto count = [](const PlayerAction& action, Op operation) {
    return static_cast<int>(std::count_if(
        action.units.begin(), action.units.end(),
        [operation](const Action& value) { return value.op == operation; }));
  };
  const auto open = state.experimental_event_local_repair
      ? state.experimental_event_local_repair->open_transactions()
      : std::vector<g001::event_local_repair::TransactionView>{};
  const int debts = std::accumulate(
      open.begin(), open.end(), 0,
      [](int total, const auto& value) {
        return total + value.outstanding_source_actions;
      });
  if (count(dig, Op::DIG) != 1 || count(plant, Op::PLANT) != 1 ||
      count(water, Op::WATER) != 1 ||
      audit.local_repair_receipts_confirmed != 3 || open.size() != 1 ||
      debts != 1)
    throw std::runtime_error(
        "same-tile effects were not serialized to one causal receipt: dig=" +
        std::to_string(count(dig, Op::DIG)) + " plant=" +
        std::to_string(count(plant, Op::PLANT)) + " water=" +
        std::to_string(count(water, Op::WATER)) + " receipts=" +
        std::to_string(audit.local_repair_receipts_confirmed) + " open=" +
        std::to_string(open.size()) + " objectives=" +
        std::to_string(debts));
}

void require_event_local_baseline_plant_precedes_repair_plant() {
  fastkag::Config config;
  config.episode_steps = 40;
  config.weed_spawn_chance = 1.0;
  config.farm_hand_cost_mult = 0;
  fastkag::Simulator simulator(config, 0xBA5E11EULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  simulator.step(opening);
  while (simulator.step_count() < 24) {
    std::array<PlayerAction, 2> actions{};
    simulator.step(actions);
  }
  Tape tape(static_cast<std::size_t>(config.episode_steps));
  for (auto& frame : tape) frame.units.resize(2);
  tape[24].market.push_back({Op::HIRE, Item::NONE, 1});
  tape[25].units[1] = {Op::WEST, Item::NONE, 1};
  tape[26].units[1] = {Op::NORTH, Item::NONE, 1};
  tape[27].units[0] = {Op::DIG, Item::NONE, 1};
  tape[27].units[1] = {Op::HARVEST, Item::WHEAT, 1};
  tape[28].units[0] = {Op::PLANT, Item::WHEAT, 1};
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  NativeAgentState state;
  NativeRepairAudit audit;
  const auto enabled = fastkag::native_repair_options_from_mask(8);
  auto next = [&] {
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, enabled, &audit);
    const auto emitted = actions[0];
    simulator.step(actions);
    return emitted;
  };
  (void)next();  // HIRE
  (void)next();  // hand WEST into unlocked quadrant
  (void)next();  // hand NORTH onto a distinct weed tile
  const auto digs = next();
  const auto plants = next();
  const int dig_count = static_cast<int>(std::count_if(
      digs.units.begin(), digs.units.end(),
      [](const Action& value) { return value.op == Op::DIG; }));
  const int plant_count = static_cast<int>(std::count_if(
      plants.units.begin(), plants.units.end(),
      [](const Action& value) { return value.op == Op::PLANT; }));
  const int observed_plants = static_cast<int>(std::count_if(
      simulator.farms()[0].tiles.begin(), simulator.farms()[0].tiles.end(),
      [](const fastkag::Tile& tile) { return tile.kind == fastkag::TileKind::PLANT; }));
  const auto open = state.experimental_event_local_repair
      ? state.experimental_event_local_repair->open_transactions()
      : std::vector<g001::event_local_repair::TransactionView>{};
  if (dig_count != 2 || plant_count != 1 || observed_plants != 1 ||
      open.size() != 1 || open.front().key.actor != 1)
    throw std::runtime_error(
        "baseline PLANT did not take priority over repair seed demand");
}

void require_route_cursor_preserves_terminal_salvage() {
  Tape pass_tape(719);
  for (auto& frame : pass_tape) frame.units.resize(1);
  NativeTapeLibrary library;
  library.routes.push_back(pass_tape);
  NativeTeammateExecutor executor(std::move(library));
  fastkag::Simulator simulator({}, 0x5A17A6EULL);
  while (simulator.step_count() < 714) {
    std::array<PlayerAction, 2> actions{};
    simulator.step(actions);
  }

  auto configured_state = [&] {
    NativeAgentState state;
    state.last_step = simulator.step_count() - 1;
    state.salvage.active = true;
    state.salvage.actor = 0;
    state.salvage.target = {0, 0};
    state.experimental_route_cursor.day = simulator.day();
    state.experimental_route_cursor.source_cursor = {simulator.step_count()};
    return state;
  };
  auto baseline_state = configured_state();
  auto repaired_state = configured_state();
  const auto baseline = executor.action_external(
      simulator, 0, 0, baseline_state);
  NativeRepairOptions repair;
  repair.route_cursor_production = 1;
  const auto repaired = executor.action_external(
      simulator, 0, 0, repaired_state, fastkag::NativeMarketArm::LegacyDefault,
      nullptr, nullptr, false, repair, nullptr);
  if (baseline.units.empty() || !move(baseline.units[0].op))
    throw std::runtime_error(
        "terminal-salvage fixture failed to produce its dynamic MOVE");
  if (repaired.units.empty() || !equal(baseline.units[0], repaired.units[0]))
    throw std::runtime_error(
        "route cursor erased the pre-existing terminal-salvage unit overlay");
  if (repaired_state.experimental_route_cursor_pending.active ||
      repaired_state.experimental_route_cursor.source_cursor !=
          std::vector<int>{simulator.step_count()})
    throw std::runtime_error(
        "overridden RouteCursor proposal consumed a MOVE/source cursor");
}

void require_event_local_weed_replant_lifecycle() {
  const auto mask = fastkag::native_repair_options_from_mask(8);
  if (!mask.state_driven_local_repair || mask.route_cursor_production != 0 ||
      mask.weed_min_loss_realign || mask.animal_buy_retry ||
      mask.empty_stall_reuse)
    throw std::runtime_error("repair mask bit 8 does not isolate local ledger");

  fastkag::Config config;
  config.episode_steps = 240;
  config.weed_spawn_chance = 0.0;
  fastkag::Simulator simulator(config, 0xE10CA1ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 3});
  simulator.step(opening);

  Tape tape(static_cast<std::size_t>(config.episode_steps));
  for (auto& frame : tape) frame.units.resize(1);
  tape[1].units[0] = {Op::PLANT, Item::WHEAT, 1};
  tape[2].units[0] = {Op::WATER, Item::NONE, 1};
  // The crop becomes WEED after two deliberately unwatered days. HARVEST is
  // durable plot debt; the following EAST remains at its exact raw slot.
  tape[72].units[0] = {Op::HARVEST, Item::NONE, 1};
  tape[73].units[0] = {Op::EAST, Item::NONE, 1};
  tape[95].units[0] = {Op::WEST, Item::NONE, 1};
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  NativeAgentState state;
  NativeRepairAudit audit;
  std::vector<Op> moves;
  std::array<Op, 3> trigger_ops{Op::PASS, Op::PASS, Op::PASS};
  std::vector<int> lifecycle_ops;
  int harvests = 0;
  while (!simulator.done()) {
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, mask, &audit);
    const int step = simulator.step_count();
    if (!actions[0].units.empty()) {
      const auto operation = actions[0].units[0].op;
      if (move(operation)) moves.push_back(operation);
      if (step >= 72 && step <= 74)
        trigger_ops[static_cast<std::size_t>(step - 72)] = operation;
      if (step >= 118 && step <= 146)
        lifecycle_ops.push_back(static_cast<int>(operation));
      harvests += operation == Op::HARVEST;
    }
    simulator.step(actions);
  }
  if (trigger_ops != std::array<Op, 3>{Op::DIG, Op::EAST, Op::PASS})
    throw std::runtime_error(
        "native local ledger did not preserve MOVE after state-driven DIG");
  if (moves != std::vector<Op>{Op::EAST, Op::WEST})
    throw std::runtime_error("native local ledger swallowed or reordered MOVE");
  if (harvests < 2 || !state.experimental_event_local_repair ||
      !state.experimental_event_local_repair->open_transactions().empty() ||
      !state.experimental_event_local_pending.empty() ||
      audit.local_repair_receipts_confirmed < 9 ||
      audit.local_repair_decisions == 0 || audit.local_repair_commits == 0)
    {
    const auto open = state.experimental_event_local_repair
        ? state.experimental_event_local_repair->open_transactions()
        : std::vector<g001::event_local_repair::TransactionView>{};
    std::string operation_trace;
    for (const int operation : lifecycle_ops)
      operation_trace += (operation_trace.empty() ? "" : ",") +
          std::to_string(operation);
    throw std::runtime_error(
        "native local ledger did not close cross-day replant lifecycle: harvests=" +
        std::to_string(harvests) + " open=" +
        std::to_string(state.experimental_event_local_repair
                           ? state.experimental_event_local_repair
                                 ->open_transactions().size()
                           : 999) + " pending=" +
        std::to_string(state.experimental_event_local_pending.size()) +
        " receipts=" +
        std::to_string(audit.local_repair_receipts_confirmed) + " failed=" +
        std::to_string(audit.local_repair_receipts_failed) + " closed=" +
        std::to_string(audit.local_repair_fail_closed) + " tx_harvests=" +
        std::to_string(open.empty() ? -1 : open.front().confirmed_harvests) +
        " tx_sources=" +
        std::to_string(open.empty() ? -1
                                    : open.front().outstanding_source_actions) +
        " desired=" +
        std::to_string(open.empty() ? -99 : open.front().desired_item) +
        " seeds=" +
        std::to_string(simulator.privates()[0].seeds[0]) +
        " ops118_146=" + operation_trace);
    }
}

void require_observation_committed_crop_continuation() {
  fastkag::Config config;
  config.episode_steps = 240;
  config.weed_spawn_chance = 0.0;
  fastkag::Simulator simulator(config, 0xC001CAFEULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 2});
  simulator.step(opening);

  Tape tape(static_cast<std::size_t>(config.episode_steps));
  for (auto& frame : tape) frame.units.resize(1);
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  NativeAgentState state;
  const auto position = simulator.farms()[0].farmer;
  const int tile = position.y * config.board_size + position.x;
  state.experimental_route_cursor.production.reset(config.board_size *
                                                    config.board_size);
  auto& debt = state.experimental_route_cursor.production.debts[
      static_cast<std::size_t>(tile)];
  debt.active = true;
  debt.crop = Item::WHEAT;
  debt.created_step = simulator.step_count();

  NativeRepairOptions repair;
  repair.route_cursor_production = 1;
  NativeRepairAudit audit;
  int emitted_plants = 0, emitted_waters = 0, emitted_harvests = 0;
  while (!simulator.done() && audit.route_cursor_completed_cycles == 0) {
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, repair, &audit);
    if (!actions[0].units.empty()) {
      emitted_plants += actions[0].units[0].op == Op::PLANT;
      emitted_waters += actions[0].units[0].op == Op::WATER;
      emitted_harvests += actions[0].units[0].op == Op::HARVEST;
    }
    simulator.step(actions);
  }
  // One extra observation/action call settles the terminal HARVEST receipt if
  // the loop stopped at its emitted step.
  if (!simulator.done() && audit.route_cursor_completed_cycles == 0)
    (void)executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, repair, &audit);
  if (emitted_plants < 2 || emitted_waters < 2 || emitted_harvests < 2 ||
      audit.route_cursor_plants < 2 || audit.route_cursor_waters < 2 ||
      audit.route_cursor_harvests < 2 ||
      audit.route_cursor_completed_cycles != 1 ||
      state.experimental_route_cursor.production.debts[
          static_cast<std::size_t>(tile)].active) {
    throw std::runtime_error(
        "crop transaction did not commit PLANT/WATER/HARVEST/replant first yield");
  }
}

void require_zero_fill_retries_next_day() {
  fastkag::Config config;
  config.episode_steps = 480;
  config.starting_money = 100;
  config.weed_spawn_chance = 0.0;
  fastkag::Simulator simulator(config, 0xB0F111ULL);
  std::array<PlayerAction, 2> opening{};
  opening[0].market = {
      {Op::BUY_PRODUCT, Item::WHEAT, 2},
      {Op::BUY_SEED, Item::TOMATO, 1},
  };
  simulator.step(opening);
  if (simulator.last_market_fills()[0].size() != 2 ||
      simulator.last_market_fills()[0][1] != 0)
    throw std::runtime_error("zero-fill seed fixture did not fail exactly");

  Tape tape(static_cast<std::size_t>(config.episode_steps));
  for (auto& frame : tape) frame.units.resize(1);
  tape[1].market.push_back({Op::SELL, Item::WHEAT, 1});
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  NativeAgentState state;
  const auto position = simulator.farms()[0].farmer;
  const int tile = position.y * config.board_size + position.x;
  state.experimental_route_cursor.production.reset(config.board_size *
                                                    config.board_size);
  auto& debt = state.experimental_route_cursor.production.debts[
      static_cast<std::size_t>(tile)];
  debt.active = true;
  debt.crop = Item::TOMATO;
  debt.created_step = 0;
  NativeAgentState::PendingPurchaseReceipt failed;
  failed.market_slot = 1;
  failed.operation = Op::BUY_SEED;
  failed.item = Item::TOMATO;
  failed.requested = 1;
  failed.submitted_step = 0;
  failed.debt_tiles = {tile};
  state.experimental_purchase_receipts.push_back(failed);

  NativeRepairOptions repair;
  repair.route_cursor_production = 2;
  NativeRepairAudit audit;
  bool retry_seen = false;
  int retry_fill_observed = -1;
  double money_before_retry = -1;
  while (!simulator.done() && simulator.step_count() <= 25) {
    std::array<PlayerAction, 2> actions{};
    actions[0] = executor.action_external(
        simulator, 0, 0, state, fastkag::NativeMarketArm::LegacyDefault,
        nullptr, nullptr, true, repair, &audit);
    const bool retry_step = simulator.step_count() == 24;
    if (retry_step) {
      money_before_retry = simulator.farms()[0].money;
      retry_seen = std::any_of(actions[0].market.begin(), actions[0].market.end(),
                               [](const Action& order) {
        return order.op == Op::BUY_SEED && order.item == Item::TOMATO &&
               order.quantity == 1;
      });
    }
    simulator.step(actions);
    if (retry_step && !simulator.last_market_fills()[0].empty())
      retry_fill_observed = simulator.last_market_fills()[0].back();
  }
  if (!retry_seen || audit.route_cursor_seed_retry_orders != 2 ||
      audit.route_cursor_seed_retry_units != 2 ||
      audit.route_cursor_seed_retry_fills != 1 ||
      state.experimental_route_cursor.production.debts[
          static_cast<std::size_t>(tile)].seed_retry_attempts != 2) {
    throw std::runtime_error(
        "zero-fill seed debt was not retained and retried on the next day: retry=" +
        std::to_string(retry_seen) + " orders=" +
        std::to_string(audit.route_cursor_seed_retry_orders) + " units=" +
        std::to_string(audit.route_cursor_seed_retry_units) + " fills=" +
        std::to_string(audit.route_cursor_seed_retry_fills) + " attempts=" +
        std::to_string(state.experimental_route_cursor.production.debts[
            static_cast<std::size_t>(tile)].seed_retry_attempts) + " direct_fill=" +
        std::to_string(retry_fill_observed) + " money=" +
        std::to_string(money_before_retry));
  }
}

Options parse(int argc,char** argv) {
  Options out;
  for(int i=1;i<argc;++i) {
    const std::string arg=argv[i];
    auto next=[&]{if(++i>=argc)throw std::invalid_argument("missing "+arg);return std::string(argv[i]);};
    if(arg=="--seed-begin")out.seed_begin=std::stoull(next());
    else if(arg=="--seeds")out.seeds=std::stoi(next());
    else if(arg=="--threads")out.threads=std::stoi(next());
    else if(arg=="--output")out.output=next();
    else if(arg=="--tapes")out.tapes=next();
    else if(arg=="--library")out.library=next();
    else throw std::invalid_argument("unknown option "+arg);
  }
  if(out.seeds<=0||out.threads<=0)throw std::invalid_argument("positive seeds/threads required");
  return out;
}

struct Aggregate {
  int games{},base_losses{},flips{},base_wins{},wins_retained{},ordered{},absolute{};
  int continuity_gate_failures{};
  int first_effect_failure_step{-1},first_effect_failure_actor{-1};
  int first_effect_failure_op{-1};
  long long unit_failures{},market_failures{},weed_events{},digs{},plants{},waters{},harvests{},abandoned{},completed{};
  long long active_crop_debts{},deferred_nonmoves{},critical_deferred_nonmoves{};
  long long cumulative_deferred_created{};
  long long expired_critical_deferred_nonmoves{},pending_route_receipts{};
  long long pending_critical_receipts{};
  long long crop_scheduler_open_debts{},crop_scheduler_sources_enqueued{};
  long long crop_scheduler_sources_coalesced{},crop_scheduler_replacements{};
  long long crop_scheduler_existing_equivalents{};
  long long crop_scheduler_receipt_successes{},crop_scheduler_receipt_failures{};
  long long crop_scheduler_completed_obligations{},crop_scheduler_completed_debts{};
  long long crop_scheduler_overdue_obligations{};
  std::array<long long,15> source_production_ops{},emitted_production_ops{};
  std::array<long long,15> terminal_deferred_ops{};
  std::array<long long,24> effect_failures_by_op{};
  long long seed_orders{},seed_units{},seed_fills{};
  long long animal_attempted{},animal_partial{},animal_zero{},animal_retries{},animal_retry_fills{};
  double own_delta{},margin_delta{},score{};
  std::uint64_t nanos{},calls{},max_nanos{};
};

Aggregate aggregate(const std::vector<Row>& rows,std::string_view opponent,int arm) {
  Aggregate out;
  for(const auto& row:rows) {
    if(!opponent.empty()&&row.opponent!=opponent)continue;
    const auto& base=row.arms[0];const auto& value=row.arms[arm];++out.games;
    const bool loss=base.score()==0,win=base.score()==2;
    out.base_losses+=loss;out.flips+=loss&&value.score()==2;
    out.base_wins+=win;out.wins_retained+=win&&value.score()==2;
    out.ordered+=value.ordered_move_day_failures;out.absolute+=value.absolute_move_mismatches;
    out.continuity_gate_failures+=!value.production_continuity_gate();
    out.active_crop_debts+=value.terminal_active_crop_debts;
    out.cumulative_deferred_created+=
        value.audit.route_cursor_deferred_nonmoves;
    out.deferred_nonmoves+=value.terminal_deferred_nonmoves;
    out.critical_deferred_nonmoves+=value.terminal_critical_deferred_nonmoves;
    out.expired_critical_deferred_nonmoves+=
        value.terminal_expired_critical_deferred_nonmoves;
    out.pending_route_receipts+=value.terminal_pending_route_receipt;
    out.pending_critical_receipts+=value.terminal_pending_critical_receipt;
    out.crop_scheduler_open_debts+=value.crop_scheduler_open_debts;
    out.crop_scheduler_sources_enqueued+=value.crop_scheduler_sources_enqueued;
    out.crop_scheduler_sources_coalesced+=value.crop_scheduler_sources_coalesced;
    out.crop_scheduler_replacements+=value.crop_scheduler_replacements;
    out.crop_scheduler_existing_equivalents+=
        value.crop_scheduler_existing_equivalents;
    out.crop_scheduler_receipt_successes+=
        value.crop_scheduler_receipt_successes;
    out.crop_scheduler_receipt_failures+=
        value.crop_scheduler_receipt_failures;
    out.crop_scheduler_completed_obligations+=
        value.crop_scheduler_completed_obligations;
    out.crop_scheduler_completed_debts+=
        value.crop_scheduler_completed_debts;
    out.crop_scheduler_overdue_obligations+=
        value.crop_scheduler_overdue_obligations;
    if(value.audit.route_cursor_first_effect_failure_step>=0&&
       (out.first_effect_failure_step<0||
        value.audit.route_cursor_first_effect_failure_step<
            out.first_effect_failure_step)) {
      out.first_effect_failure_step=
          value.audit.route_cursor_first_effect_failure_step;
      out.first_effect_failure_actor=
          value.audit.route_cursor_first_effect_failure_actor;
      out.first_effect_failure_op=value.audit.route_cursor_first_effect_failure_op;
    }
    for(std::size_t op=0;op<out.effect_failures_by_op.size();++op)
      out.effect_failures_by_op[op]+=
          value.audit.route_cursor_effect_failures_by_op[op];
    for(std::size_t op=0;op<out.source_production_ops.size();++op) {
      out.source_production_ops[op]+=value.source_production_ops[op];
      out.emitted_production_ops[op]+=value.emitted_production_ops[op];
      out.terminal_deferred_ops[op]+=value.terminal_deferred_ops[op];
    }
    out.unit_failures+=value.unit_failures;out.market_failures+=value.market_failures;
    out.weed_events+=value.audit.route_cursor_weed_events;
    out.digs+=value.audit.route_cursor_digs;
    out.plants+=value.audit.route_cursor_plants;
    out.waters+=value.audit.route_cursor_waters;
    out.harvests+=value.audit.route_cursor_harvests;
    out.abandoned+=value.audit.route_cursor_abandoned_cycles;
    out.completed+=value.audit.route_cursor_completed_cycles;
    out.seed_orders+=value.audit.route_cursor_seed_retry_orders;
    out.seed_units+=value.audit.route_cursor_seed_retry_units;
    out.seed_fills+=value.audit.route_cursor_seed_retry_fills;
    out.animal_attempted+=value.audit.animal_original_attempted;
    out.animal_partial+=value.audit.animal_inferred_partial;
    out.animal_zero+=value.audit.animal_inferred_zero;
    out.animal_retries+=value.audit.animal_retries_emitted;
    out.animal_retry_fills+=value.audit.animal_retry_fills;
    out.own_delta+=value.own-base.own;out.margin_delta+=value.margin()-base.margin();
    out.score+=value.score()/2.0;out.nanos+=value.action_nanoseconds;
    out.calls+=value.action_calls;out.max_nanos=std::max(out.max_nanos,value.maximum_action_nanoseconds);
  }
  return out;
}

} // namespace

int main(int argc,char** argv) try {
  const auto options=parse(argc,argv);
  const auto g001=g001::repair::load_route(options.tapes,options.library,"G001");
  const auto g096=g001::repair::load_route(options.tapes,options.library,"G096");
  NativeTapeLibrary library;library.routes={g001,g096};
  NativeTeammateExecutor executor(std::move(library));
  require_default_parity(executor,options.seed_begin);
  require_event_local_enabled_no_trigger_719_multi_actor_parity();
  require_event_local_unknown_weed_source_item_drives_recompile();
  require_event_local_global_seed_reservation();
  require_event_local_same_tile_serialization();
  require_event_local_baseline_plant_precedes_repair_plant();
  require_event_local_weed_replant_lifecycle();
  require_route_cursor_preserves_terminal_salvage();
  require_observation_committed_crop_continuation();
  require_zero_fill_retries_next_day();
  struct Job{int opponent{};std::uint64_t seed{};int seat{};};
  std::vector<Job> jobs;
  for(int opponent=0;opponent<2;++opponent)for(int s=0;s<options.seeds;++s)
    for(int seat=0;seat<2;++seat)jobs.push_back({opponent,options.seed_begin+static_cast<std::uint64_t>(s),seat});
  std::vector<Row> rows(jobs.size());std::atomic<std::size_t> cursor{};
  std::exception_ptr failure;std::mutex mutex;std::vector<std::thread> workers;
  for(int worker=0;worker<std::min<int>(options.threads,jobs.size());++worker)
    workers.emplace_back([&]{try{while(true){const auto i=cursor.fetch_add(1);if(i>=jobs.size())break;
      const auto& job=jobs[i];Row row;row.opponent=job.opponent?"G096":"G001";row.seed=job.seed;row.seat=job.seat;
      for(int arm=0;arm<4;++arm)row.arms[arm]=run(executor,g001,job.opponent,job.seed,job.seat,arm);
      rows[i]=std::move(row);}}catch(...){std::lock_guard lock(mutex);if(!failure)failure=std::current_exception();cursor.store(jobs.size());}});
  for(auto& worker:workers) worker.join();
  if(failure) std::rethrow_exception(failure);
  const auto parent=std::filesystem::path(options.output).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
  std::ofstream report(options.output);report<<std::fixed<<std::setprecision(6);
  report<<"{\n  \"schema\":\"native-route-cursor-continuity-v2\",\n  \"causal\":true,\n  \"terminal_selection\":false,\n  \"default_off_bit_parity\":true,\n  \"production_op_order\":[\"DIG\",\"PLANT\",\"WATER\",\"HARVEST\",\"FERTILIZE\",\"BUILD_COOP\",\"BUILD_PASTURE\",\"PLACE\",\"FEED\",\"CARE\",\"PICKUP\",\"DROP\",\"COLLECT_FERTILIZER\",\"NS_MOVE\",\"EW_MOVE\"],\n  \"seed_begin\":"<<options.seed_begin<<",\n  \"seeds_per_opponent\":"<<options.seeds<<",\n  \"arms\":{\n";
  constexpr std::array<const char*,4> names{
      "legacy","online-critical","online-unified","online-unified-animal"};
  for(int arm=0;arm<4;++arm){report<<"    \""<<names[arm]<<"\":{";
    for(int opponent=0;opponent<2;++opponent){const std::string name=opponent?"G096":"G001";const auto a=aggregate(rows,name,arm);const double n=std::max(1,a.games);
      report<<(opponent?",":"")<<"\""<<name<<"\":{\"games\":"<<a.games<<",\"baseline_losses\":"<<a.base_losses<<",\"loss_to_win\":"<<a.flips<<",\"baseline_wins\":"<<a.base_wins<<",\"wins_retained\":"<<a.wins_retained<<",\"own_delta_mean\":"<<a.own_delta/n<<",\"margin_delta_mean\":"<<a.margin_delta/n<<",\"score_rate\":"<<a.score/n<<",\"production_continuity_gate_failures\":"<<a.continuity_gate_failures<<",\"ordered_move_day_failures\":"<<a.ordered<<",\"absolute_move_mismatches\":"<<a.absolute<<",\"cumulative_deferred_nonmoves_created\":"<<a.cumulative_deferred_created<<",\"terminal_active_crop_debts\":"<<a.active_crop_debts<<",\"terminal_deferred_nonmoves\":"<<a.deferred_nonmoves<<",\"terminal_critical_deferred_nonmoves\":"<<a.critical_deferred_nonmoves<<",\"terminal_expired_critical_deferred_nonmoves\":"<<a.expired_critical_deferred_nonmoves<<",\"terminal_pending_route_receipts\":"<<a.pending_route_receipts<<",\"terminal_pending_critical_receipts\":"<<a.pending_critical_receipts<<",\"crop_scheduler_open_debts\":"<<a.crop_scheduler_open_debts<<",\"crop_scheduler_sources_enqueued\":"<<a.crop_scheduler_sources_enqueued<<",\"crop_scheduler_sources_coalesced\":"<<a.crop_scheduler_sources_coalesced<<",\"crop_scheduler_replacements\":"<<a.crop_scheduler_replacements<<",\"crop_scheduler_existing_equivalents\":"<<a.crop_scheduler_existing_equivalents<<",\"crop_scheduler_receipt_successes\":"<<a.crop_scheduler_receipt_successes<<",\"crop_scheduler_receipt_failures\":"<<a.crop_scheduler_receipt_failures<<",\"crop_scheduler_completed_obligations\":"<<a.crop_scheduler_completed_obligations<<",\"crop_scheduler_completed_debts\":"<<a.crop_scheduler_completed_debts<<",\"crop_scheduler_overdue_obligations\":"<<a.crop_scheduler_overdue_obligations<<",\"first_effect_failure_step\":"<<a.first_effect_failure_step<<",\"first_effect_failure_actor\":"<<a.first_effect_failure_actor<<",\"first_effect_failure_op\":"<<a.first_effect_failure_op<<",\"unit_failures\":"<<a.unit_failures<<",\"market_failures\":"<<a.market_failures<<",\"weed_events\":"<<a.weed_events<<",\"digs\":"<<a.digs<<",\"plants\":"<<a.plants<<",\"waters\":"<<a.waters<<",\"harvests\":"<<a.harvests<<",\"abandoned_cycles\":"<<a.abandoned<<",\"completed_cycles\":"<<a.completed<<",\"seed_retry_orders\":"<<a.seed_orders<<",\"seed_retry_units\":"<<a.seed_units<<",\"seed_retry_fills\":"<<a.seed_fills<<",\"animal_original_attempted\":"<<a.animal_attempted<<",\"animal_partial_failures\":"<<a.animal_partial<<",\"animal_zero_failures\":"<<a.animal_zero<<",\"animal_retries\":"<<a.animal_retries<<",\"animal_retry_fills\":"<<a.animal_retry_fills<<",\"effect_failures_by_op\":[";
      for(std::size_t op=0;op<a.effect_failures_by_op.size();++op)report<<(op?",":"")<<a.effect_failures_by_op[op];
      report<<"],\"source_production_ops\":[";
      for(std::size_t op=0;op<a.source_production_ops.size();++op)report<<(op?",":"")<<a.source_production_ops[op];
      report<<"],\"emitted_production_ops\":[";
      for(std::size_t op=0;op<a.emitted_production_ops.size();++op)report<<(op?",":"")<<a.emitted_production_ops[op];
      report<<"],\"terminal_deferred_ops\":[";
      for(std::size_t op=0;op<a.terminal_deferred_ops.size();++op)report<<(op?",":"")<<a.terminal_deferred_ops[op];
      report<<"],\"mean_action_us\":"<<(a.calls?double(a.nanos)/a.calls/1000.0:0)<<",\"max_action_us\":"<<double(a.max_nanos)/1000.0<<"}";}
    report<<"}"<<(arm==3?"\n":",\n");}
  report<<"  }\n}\n";
  std::cout<<"native route cursor online games="<<rows.size()<<" report="<<options.output<<'\n';
  return 0;
}catch(const std::exception& error){std::cerr<<"native_route_cursor_eval: "<<error.what()<<'\n';return 2;}
