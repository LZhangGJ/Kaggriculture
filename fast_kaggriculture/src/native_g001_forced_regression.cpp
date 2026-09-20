#include "native_teammate.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::NativeAgentState;
using fastkag::NativeRepairAudit;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using Tape = std::vector<PlayerAction>;

struct Options {
  std::uint64_t seed{0x6001F0CEDULL};
  int repair_mask{8};
  std::string output{"g001-forced-trigger-regression.json"};
  std::string tapes{NATIVE_G001_TAPES};
  std::string library{NATIVE_G001_LIBRARY};
};

struct Event {
  int step{};
  int day{};
  int hour{};
  int new_transactions{};
  int receipts_confirmed{};
  int receipts_failed{};
  int seed_orders{};
  int seed_fills{};
  int seed_zero_fills{};
  int open_transactions{};
  int open_source_debts{};
  int open_purchase_debts{};
  int pending_unit_receipts{};
  int pending_purchase_receipts{};
  int recycled_actor_collisions{};
  int stale_actor_pending{};
  int weeds_added_after_step{};
  int drop_deposit_units{};
  std::vector<int> baseline_ops;
  std::vector<int> enabled_ops;
};

struct RecycledWitness {
  int step{-1};
  int current_day{-1};
  int origin_day{-1};
  int actor{-1};
  int x{-1};
  int y{-1};
  std::uint64_t transaction_id{};
  bool pending_receipt{};
};

struct MoveMismatch {
  int day{};
  int actor{};
  std::vector<int> baseline;
  std::vector<int> enabled;
  std::vector<int> baseline_active_hours;
  std::vector<int> enabled_active_hours;
  std::vector<int> baseline_move_hours;
  std::vector<int> enabled_move_hours;
  std::vector<int> enabled_window_source;
  std::string reason;
};

struct OpenAttribution {
  int total{};
  int identity_polluted{};
  int terminal_pending{};
  int orphaned_worker_generation{};
  int desired_crop_impossible{};
  int source_debt_externalized{};
  int lifecycle_incomplete{};
  int old_worker_generation_risk{};
  int externalized_source_units{};
  int critical_source_transactions{};
  int critical_source_units{};
  int recoverable_critical_transactions{};
  int lifecycle_controller_only{};
};

struct OpenDetail {
  g001::event_local_repair::TransactionView transaction;
  int tile_kind{-1};
  int tile_crop{-1};
  int current_claimant{-1};
  bool pending{};
  bool critical{};
  bool recoverable_critical{};
  bool lifecycle_controller_only{};
  int last_receipt_step{-1};
  bool last_receipt_confirmed{};
  std::string desired_state;
};

bool movement(Op operation) {
  return operation == Op::NORTH || operation == Op::SOUTH ||
         operation == Op::EAST || operation == Op::WEST;
}

bool same_position(Position position,
                   g001::event_local_repair::Position tile) {
  return position.x == tile.column && position.y == tile.row;
}

int shed_total(const Simulator& simulator, int player) {
  return std::accumulate(simulator.privates()[player].shed.begin(),
                         simulator.privates()[player].shed.end(), 0);
}

int weed_count(const Simulator& simulator, int player) {
  return static_cast<int>(std::count_if(
      simulator.farms()[player].tiles.begin(),
      simulator.farms()[player].tiles.end(), [](const auto& tile) {
        return tile.kind == fastkag::TileKind::WEED;
      }));
}

std::vector<Position> positions(const Simulator& simulator, int player) {
  std::vector<Position> result{simulator.farms()[player].farmer};
  result.insert(result.end(), simulator.farms()[player].hands.begin(),
                simulator.farms()[player].hands.end());
  return result;
}

std::uint64_t actor_generation(int day, int actor) {
  if (actor == 0) return 1;
  return (static_cast<std::uint64_t>(day + 1) << 32U) |
      static_cast<std::uint64_t>(actor + 1);
}

int open_purchase_debts(const NativeAgentState& state) {
  const auto debts = state.experimental_event_local_purchase_ledger.debts();
  return static_cast<int>(std::count_if(
      debts.begin(), debts.end(),
      [](const auto& debt) { return !debt.acquisition_complete; }));
}

int pending_purchase_receipts(const NativeAgentState& state) {
  const auto debts = state.experimental_event_local_purchase_ledger.debts();
  return static_cast<int>(std::count_if(
      debts.begin(), debts.end(),
      [](const auto& debt) { return debt.awaiting_receipt; }));
}

bool same_config(const fastkag::Config& left, const fastkag::Config& right) {
  return left.episode_steps == right.episode_steps &&
      left.board_size == right.board_size &&
      left.starting_money == right.starting_money &&
      left.max_market_orders == right.max_market_orders &&
      left.turns_per_day == right.turns_per_day &&
      left.shed_capacity == right.shed_capacity &&
      left.weed_spawn_chance == right.weed_spawn_chance &&
      left.town_shop_unlock_interval == right.town_shop_unlock_interval &&
      left.town_shop_sell_interval == right.town_shop_sell_interval &&
      left.town_center_sell_interval == right.town_center_sell_interval &&
      left.farm_hand_cost_mult == right.farm_hand_cost_mult;
}

Options parse(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc) throw std::invalid_argument("missing option value");
      return argv[index];
    };
    if (argument == "--seed") options.seed = std::stoull(value());
    else if (argument == "--repair-mask")
      options.repair_mask = std::stoi(value());
    else if (argument == "--output") options.output = value();
    else if (argument == "--tapes") options.tapes = value();
    else if (argument == "--library") options.library = value();
    else throw std::invalid_argument("unknown option: " + argument);
  }
  if (options.repair_mask != 8 && options.repair_mask != 32 &&
      options.repair_mask != 64)
    throw std::invalid_argument("repair-mask must be 8, 32, or 64");
  return options;
}

template <class Values>
void write_numbers(std::ostream& output, const Values& values) {
  output << '[';
  bool first = true;
  for (const auto value : values) {
    if (!first) output << ',';
    first = false;
    output << value;
  }
  output << ']';
}

void append_moves(
    const PlayerAction& action, int day, int hour,
    std::map<std::pair<int, int>, std::vector<int>>& destination,
    std::map<std::pair<int, int>, std::vector<int>>& hours) {
  for (std::size_t actor = 0; actor < action.units.size(); ++actor)
    if (movement(action.units[actor].op)) {
      destination[{day, static_cast<int>(actor)}].push_back(
          static_cast<int>(action.units[actor].op));
      hours[{day, static_cast<int>(actor)}].push_back(hour);
    }
}

void append_active_hours(
    const PlayerAction& action, int day, int hour,
    std::map<std::pair<int, int>, std::vector<int>>& destination) {
  for (std::size_t actor = 0; actor < action.units.size(); ++actor)
    destination[{day, static_cast<int>(actor)}].push_back(hour);
}

void run_v2_forced(const Options& options, const Tape& tape,
                   const NativeTeammateExecutor& executor) {
  fastkag::Config config;
  config.episode_steps = 720;
  config.weed_spawn_chance = 1.0;
  Simulator baseline(config, options.seed);
  Simulator enabled(config, options.seed);
  std::array<NativeAgentState, 2> baseline_states;
  std::array<NativeAgentState, 2> enabled_states;
  NativeRepairAudit audit;
  const auto repair = fastkag::native_repair_options_from_mask(32);
  if (!repair.day_horizon_repair_v2 || repair.day_horizon_repair ||
      repair.state_driven_local_repair ||
      repair.route_cursor_production != 0)
    throw std::runtime_error("bit32 forced comparison mask is not isolated");
  std::map<std::pair<int, int>, std::vector<int>> raw_moves;
  std::map<std::pair<int, int>, std::vector<int>> final_moves;
  std::map<std::pair<int, int>, std::vector<int>> raw_hours;
  std::map<std::pair<int, int>, std::vector<int>> final_hours;
  int identity_violations = 0;
  int baseline_unit_failures = 0;
  int treatment_unit_failures = 0;
  int baseline_market_failures = 0;
  int treatment_market_failures = 0;
  int turns = 0;
  while (!baseline.done() && !enabled.done()) {
    const int step = enabled.step_count();
    const int day = enabled.day();
    const int hour = enabled.hour();
    for (const auto& pending :
         enabled_states[0].experimental_day_horizon_v2.pending)
      if (pending.actor < 0 || pending.actor_generation != actor_generation(
              pending.submitted_step / config.turns_per_day, pending.actor))
        ++identity_violations;
    std::array<PlayerAction, 2> baseline_actions{
        executor.action_external(
            baseline, 0, 0, baseline_states[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true),
        executor.action_external(baseline, 1, 0, baseline_states[1])};
    std::array<PlayerAction, 2> enabled_actions{
        executor.action_external(
            enabled, 0, 0, enabled_states[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true,
            repair, &audit),
        executor.action_external(enabled, 1, 0, enabled_states[1])};
    for (std::size_t actor = 0; actor < enabled_actions[0].units.size(); ++actor) {
      const auto& source_units = tape[static_cast<std::size_t>(step)].units;
      const Action source = actor < source_units.size()
          ? source_units[actor] : Action{};
      if (movement(source.op)) {
        raw_moves[{day, static_cast<int>(actor)}].push_back(
            static_cast<int>(source.op));
        raw_hours[{day, static_cast<int>(actor)}].push_back(hour);
      }
      if (movement(enabled_actions[0].units[actor].op)) {
        final_moves[{day, static_cast<int>(actor)}].push_back(
            static_cast<int>(enabled_actions[0].units[actor].op));
        final_hours[{day, static_cast<int>(actor)}].push_back(hour);
      }
    }
    baseline_unit_failures +=
        fastkag::native_macro_unit_failures(baseline, 0, baseline_actions[0]);
    treatment_unit_failures +=
        fastkag::native_macro_unit_failures(enabled, 0, enabled_actions[0]);
    baseline.step(baseline_actions);
    enabled.step(enabled_actions);
    baseline_market_failures += fastkag::native_macro_market_failures(
        baseline, 0, baseline_actions[0]);
    treatment_market_failures += fastkag::native_macro_market_failures(
        enabled, 0, enabled_actions[0]);
    ++turns;
  }
  std::set<std::pair<int, int>> keys;
  for (const auto& [key, unused] : raw_moves) {
    (void)unused;
    keys.insert(key);
  }
  for (const auto& [key, unused] : final_moves) {
    (void)unused;
    keys.insert(key);
  }
  int ordered_route_failures = 0;
  int move_timing_deviation = 0;
  for (const auto& key : keys) {
    if (raw_moves[key] != final_moves[key]) {
      ++ordered_route_failures;
      continue;
    }
    for (std::size_t index = 0; index < raw_hours[key].size(); ++index)
      move_timing_deviation +=
          final_hours[key][index] - raw_hours[key][index];
  }
  int pending_purchase = 0;
  for (const auto& debt :
       enabled_states[0].experimental_day_horizon_v2.purchase_ledger.debts())
    pending_purchase += debt.awaiting_receipt;
  const int pending_unit = static_cast<int>(
      enabled_states[0].experimental_day_horizon_v2.pending.size());
  const int active_objectives = static_cast<int>(
      enabled_states[0].experimental_day_horizon_v2.objectives.size());
  const bool gates = turns == 719 && ordered_route_failures == 0 &&
      audit.day_horizon_v2_terminal_raw == 0 && identity_violations == 0 &&
      pending_unit == 0 && pending_purchase == 0;
  const auto parent = std::filesystem::path(options.output).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream output(options.output);
  if (!output) throw std::runtime_error("cannot open bit32 forced report");
  output << "{\n  \"schema\":\"native-g001-day-horizon-v2-forced-v1\",\n"
         << "  \"repair_mask\":32,\n  \"seed\":" << options.seed
         << ",\n  \"turns\":" << turns
         << ",\n  \"gates_passed\":" << (gates ? "true" : "false")
         << ",\n  \"ordered_route_failures\":"
         << ordered_route_failures
         << ",\n  \"move_timing_deviation\":" << move_timing_deviation
         << ",\n  \"identity_violations\":" << identity_violations
         << ",\n  \"terminal_raw\":"
         << audit.day_horizon_v2_terminal_raw
         << ",\n  \"terminal_pending_unit_receipts\":" << pending_unit
         << ",\n  \"terminal_pending_purchase_receipts\":"
         << pending_purchase
         << ",\n  \"terminal_active_objectives\":" << active_objectives
         << ",\n  \"economics\":{\"baseline_money\":"
         << baseline.farms()[0].money << ",\"treatment_money\":"
         << enabled.farms()[0].money
         << ",\"baseline_unit_failures\":" << baseline_unit_failures
         << ",\"treatment_unit_failures\":" << treatment_unit_failures
         << ",\"baseline_market_failures\":" << baseline_market_failures
         << ",\"treatment_market_failures\":"
         << treatment_market_failures << "},\n  \"audit\":{"
         << "\"plans\":" << audit.day_horizon_v2_plans
         << ",\"replans\":" << audit.day_horizon_v2_replans
         << ",\"exact_plans\":" << audit.day_horizon_v2_exact_plans
         << ",\"fallback_plans\":"
         << audit.day_horizon_v2_fallback_plans
         << ",\"budget_exhausted\":"
         << audit.day_horizon_v2_budget_exhausted
         << ",\"assignments\":" << audit.day_horizon_v2_assignments
         << ",\"receipts_confirmed\":"
         << audit.day_horizon_v2_receipts_confirmed
         << ",\"receipts_failed\":"
         << audit.day_horizon_v2_receipts_failed
         << ",\"fail_closed\":" << audit.day_horizon_v2_fail_closed
         << ",\"objectives_completed\":"
         << audit.day_horizon_v2_objectives_completed
         << ",\"objectives_carried\":"
         << audit.day_horizon_v2_objectives_carried
         << ",\"maturity_waits\":"
         << audit.day_horizon_v2_maturity_waits
         << ",\"maturity_tokens_consumed\":"
         << audit.day_horizon_v2_maturity_tokens_consumed
         << ",\"seed_unscheduled\":"
         << audit.day_horizon_v2_seed_unscheduled
         << ",\"seed_orders\":" << audit.day_horizon_v2_seed_orders
         << ",\"seed_zero_fills\":"
         << audit.day_horizon_v2_seed_zero_fills
         << ",\"seed_fills\":" << audit.day_horizon_v2_seed_fills
         << ",\"fail_reasons\":";
  write_numbers(output, audit.day_horizon_v2_fail_reasons);
  output << "}\n}\n";
}

void run_v3_forced(const Options& options, const Tape& tape,
                   const NativeTeammateExecutor& executor) {
  fastkag::Config config;
  config.episode_steps = 720;
  config.weed_spawn_chance = 1.0;
  Simulator baseline(config, options.seed);
  Simulator enabled(config, options.seed);
  std::array<NativeAgentState, 2> baseline_states;
  std::array<NativeAgentState, 2> enabled_states;
  NativeRepairAudit audit;
  const auto repair = fastkag::native_repair_options_from_mask(64);
  if (!repair.rolling_route_skeleton_v3 || repair.day_horizon_repair_v2 ||
      repair.day_horizon_repair || repair.state_driven_local_repair ||
      repair.route_cursor_production != 0)
    throw std::runtime_error("bit64 forced comparison mask is not isolated");
  std::map<std::pair<int, int>, std::vector<int>> raw_moves;
  std::map<std::pair<int, int>, std::vector<int>> final_moves;
  int identity_violations = 0;
  int baseline_unit_failures = 0;
  int treatment_unit_failures = 0;
  int baseline_market_failures = 0;
  int treatment_market_failures = 0;
  int turns = 0;
  while (!baseline.done() && !enabled.done()) {
    const int step = enabled.step_count();
    const int day = enabled.day();
    for (const auto& pending :
         enabled_states[0].experimental_route_skeleton_v3.pending)
      if (pending.actor < 0 || pending.generation != actor_generation(
              pending.submitted_step / config.turns_per_day, pending.actor))
        ++identity_violations;
    std::array<PlayerAction, 2> baseline_actions{
        executor.action_external(
            baseline, 0, 0, baseline_states[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true),
        executor.action_external(baseline, 1, 0, baseline_states[1])};
    std::array<PlayerAction, 2> enabled_actions{
        executor.action_external(
            enabled, 0, 0, enabled_states[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true,
            repair, &audit),
        executor.action_external(enabled, 1, 0, enabled_states[1])};
    for (std::size_t actor = 0; actor < enabled_actions[0].units.size(); ++actor) {
      const auto& source_units = tape[static_cast<std::size_t>(step)].units;
      const Action source = actor < source_units.size()
          ? source_units[actor] : Action{};
      if (movement(source.op))
        raw_moves[{day, static_cast<int>(actor)}].push_back(
            static_cast<int>(source.op));
      if (movement(enabled_actions[0].units[actor].op))
        final_moves[{day, static_cast<int>(actor)}].push_back(
            static_cast<int>(enabled_actions[0].units[actor].op));
    }
    baseline_unit_failures +=
        fastkag::native_macro_unit_failures(baseline, 0, baseline_actions[0]);
    treatment_unit_failures +=
        fastkag::native_macro_unit_failures(enabled, 0, enabled_actions[0]);
    baseline.step(baseline_actions);
    enabled.step(enabled_actions);
    baseline_market_failures += fastkag::native_macro_market_failures(
        baseline, 0, baseline_actions[0]);
    treatment_market_failures += fastkag::native_macro_market_failures(
        enabled, 0, enabled_actions[0]);
    ++turns;
  }
  std::set<std::pair<int, int>> keys;
  for (const auto& [key, unused] : raw_moves) {
    (void)unused;
    keys.insert(key);
  }
  for (const auto& [key, unused] : final_moves) {
    (void)unused;
    keys.insert(key);
  }
  int ordered_route_failures = 0;
  std::vector<std::tuple<int, int, int, int>> ordered_route_mismatches;
  for (const auto& key : keys)
    if (raw_moves[key] != final_moves[key]) {
      ++ordered_route_failures;
      ordered_route_mismatches.push_back(
          {key.first, key.second, static_cast<int>(raw_moves[key].size()),
           static_cast<int>(final_moves[key].size())});
    }
  const auto& runtime =
      enabled_states[0].experimental_route_skeleton_v3;
  int terminal_skeleton = 0;
  for (const auto& actor : runtime.actors)
    terminal_skeleton +=
        static_cast<int>(actor.moves.size() - actor.cursor);
  const int pending_unit = static_cast<int>(runtime.pending.size());
  const bool gates = turns == 719 && ordered_route_failures == 0 &&
      terminal_skeleton == 0 && identity_violations == 0 &&
      pending_unit == 0;
  const auto parent = std::filesystem::path(options.output).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream output(options.output);
  if (!output) throw std::runtime_error("cannot open bit64 forced report");
  output << "{\n  \"schema\":\"native-g001-route-skeleton-v3-forced-v1\",\n"
         << "  \"repair_mask\":64,\n  \"seed\":" << options.seed
         << ",\n  \"turns\":" << turns
         << ",\n  \"gates_passed\":" << (gates ? "true" : "false")
         << ",\n  \"ordered_route_failures\":"
         << ordered_route_failures
         << ",\n  \"ordered_route_mismatch_details\":[";
  for (std::size_t index = 0; index < ordered_route_mismatches.size(); ++index) {
    if (index) output << ',';
    const auto [mismatch_day, mismatch_actor, raw_count, final_count] =
        ordered_route_mismatches[index];
    output << "{\"day\":" << mismatch_day << ",\"actor\":"
           << mismatch_actor << ",\"raw_count\":" << raw_count
           << ",\"final_count\":" << final_count << '}';
  }
  output << ']'
         << ",\n  \"identity_violations\":" << identity_violations
         << ",\n  \"terminal_skeleton_moves\":" << terminal_skeleton
         << ",\n  \"terminal_pending_unit_receipts\":" << pending_unit
         << ",\n  \"terminal_active_objectives\":"
         << runtime.objectives.size()
         << ",\n  \"economics\":{\"baseline_money\":"
         << baseline.farms()[0].money << ",\"treatment_money\":"
         << enabled.farms()[0].money
         << ",\"baseline_unit_failures\":" << baseline_unit_failures
         << ",\"treatment_unit_failures\":" << treatment_unit_failures
         << ",\"baseline_market_failures\":" << baseline_market_failures
         << ",\"treatment_market_failures\":"
         << treatment_market_failures << "},\n  \"audit\":{"
         << "\"plans\":" << audit.route_skeleton_v3_plans
         << ",\"replans\":" << audit.route_skeleton_v3_replans
         << ",\"rebases\":" << audit.route_skeleton_v3_rebases
         << ",\"lcs_kept\":" << audit.route_skeleton_v3_lcs_kept
         << ",\"assignments\":" << audit.route_skeleton_v3_assignments
         << ",\"receipts_confirmed\":"
         << audit.route_skeleton_v3_receipts_confirmed
         << ",\"receipts_failed\":"
         << audit.route_skeleton_v3_receipts_failed
         << ",\"fail_closed\":" << audit.route_skeleton_v3_fail_closed
         << ",\"objectives_completed\":"
         << audit.route_skeleton_v3_objectives_completed
         << ",\"objectives_carried\":"
         << audit.route_skeleton_v3_objectives_carried
         << ",\"moves_emitted\":"
         << audit.route_skeleton_v3_moves_emitted
         << ",\"fallback_plans\":"
         << audit.route_skeleton_v3_fallback_plans
         << ",\"budget_exhausted\":"
         << audit.route_skeleton_v3_budget_exhausted
         << ",\"fail_reasons\":";
  write_numbers(output, audit.route_skeleton_v3_fail_reasons);
  output << "}\n}\n";
}

void write_report(
    const Options& options, const std::vector<Event>& events,
    const std::vector<MoveMismatch>& movement_mismatches,
    const std::optional<RecycledWitness>& recycled_witness,
    const NativeRepairAudit& audit,
    const g001::event_local_repair::Audit& compiler_audit,
    const NativeAgentState& final_state, int turns,
    int boundary_receipts_confirmed, int boundary_receipts_failed,
    int day_boundaries_with_new_weeds, int last_new_weed_step,
    int last_harvest_step, int last_harvest_receipt_step,
    int last_drop_step, int last_drop_deposit_units,
    int last_emitted_harvest_step, const OpenAttribution& attribution,
    const std::vector<OpenDetail>& open_details,
    double baseline_final_money, double enabled_final_money,
    bool terminal_last_source_pass, bool gates_passed) {
  const auto parent = std::filesystem::path(options.output).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream output(options.output);
  if (!output) throw std::runtime_error("cannot open output report");
  const auto open = final_state.experimental_event_local_repair
      ? final_state.experimental_event_local_repair->open_transactions()
      : std::vector<g001::event_local_repair::TransactionView>{};
  output << "{\n"
         << "  \"schema\":\"native-g001-forced-trigger-regression-v1\",\n"
         << "  \"formal_data\":false,\n"
         << "  \"research_only\":true,\n"
         << "  \"promotable\":false,\n"
         << "  \"route\":\"G001\",\n"
         << "  \"turns\":" << turns << ",\n"
         << "  \"seed\":" << options.seed << ",\n"
         << "  \"weed_spawn_chance\":1.0,\n"
         << "  \"gates_passed\":" << (gates_passed ? "true" : "false")
         << ",\n"
         << "  \"summary\":{\n"
         << "    \"transactions_created\":"
         << compiler_audit.transactions_created << ",\n"
         << "    \"source_intents_coalesced\":"
         << compiler_audit.source_intents_coalesced << ",\n"
         << "    \"objectives_superseded\":"
         << compiler_audit.objectives_superseded << ",\n"
         << "    \"objectives_cancelled\":"
         << compiler_audit.objectives_cancelled << ",\n"
         << "    \"plot_leases\":" << compiler_audit.plot_leases << ",\n"
         << "    \"plot_execution_leases\":" << compiler_audit.plot_leases
         << ",\n"
         << "    \"source_intents_coalesced\":"
         << compiler_audit.source_intents_coalesced << ",\n"
         << "    \"objectives_superseded\":"
         << compiler_audit.objectives_superseded << ",\n"
         << "    \"objectives_cancelled\":"
         << compiler_audit.objectives_cancelled << ",\n"
         << "    \"receipts_confirmed\":" << audit.local_repair_receipts_confirmed
         << ",\n"
         << "    \"receipts_failed\":" << audit.local_repair_receipts_failed
         << ",\n"
         << "    \"seed_orders\":" << audit.local_repair_seed_orders << ",\n"
         << "    \"seed_fills\":" << audit.local_repair_seed_fills << ",\n"
         << "    \"seed_zero_fills\":"
         << audit.local_repair_seed_zero_fills << ",\n"
         << "    \"open_transactions\":" << open.size() << ",\n"
         << "    \"open_purchase_debts\":"
         << open_purchase_debts(final_state) << ",\n"
         << "    \"open_attribution\":{\"total\":" << attribution.total
         << ",\"identity_polluted\":" << attribution.identity_polluted
         << ",\"terminal_pending\":" << attribution.terminal_pending
         << ",\"orphaned_worker_generation\":"
         << attribution.orphaned_worker_generation
         << ",\"desired_crop_impossible\":"
         << attribution.desired_crop_impossible
         << ",\"source_debt_externalized\":"
         << attribution.source_debt_externalized
         << ",\"lifecycle_incomplete\":"
         << attribution.lifecycle_incomplete
         << ",\"old_worker_generation_risk\":"
         << attribution.old_worker_generation_risk
         << ",\"externalized_source_units\":"
         << attribution.externalized_source_units
         << ",\"critical_source_transactions\":"
         << attribution.critical_source_transactions
         << ",\"critical_source_units\":"
         << attribution.critical_source_units
         << ",\"recoverable_critical_transactions\":"
         << attribution.recoverable_critical_transactions
         << ",\"lifecycle_controller_only\":"
         << attribution.lifecycle_controller_only << "},\n"
         << "    \"baseline_final_money\":" << baseline_final_money << ",\n"
         << "    \"enabled_final_money\":" << enabled_final_money << ",\n"
         << "    \"terminal_pending_unit_receipts\":"
         << final_state.experimental_event_local_pending.size() << ",\n"
         << "    \"terminal_pending_purchase_receipts\":"
         << pending_purchase_receipts(final_state) << ",\n"
         << "    \"move_sequence_mismatches\":"
         << movement_mismatches.size() << ",\n"
         << "    \"recycled_actor_pollution\":"
         << (recycled_witness ? 1 : 0) << ",\n"
         << "    \"boundary_receipts_confirmed\":"
         << boundary_receipts_confirmed << ",\n"
         << "    \"boundary_receipts_failed\":"
         << boundary_receipts_failed << ",\n"
         << "    \"day_boundaries_with_new_weeds\":"
         << day_boundaries_with_new_weeds << ",\n"
         << "    \"last_new_weed_step\":" << last_new_weed_step << ",\n"
         << "    \"last_emitted_harvest_step\":"
         << last_emitted_harvest_step << ",\n"
         << "    \"last_bound_harvest_step\":" << last_harvest_step
         << ",\n"
         << "    \"last_bound_harvest_receipt_step\":"
         << last_harvest_receipt_step << ",\n"
         << "    \"last_drop_step\":" << last_drop_step << ",\n"
         << "    \"last_drop_deposit_units\":"
         << last_drop_deposit_units << ",\n"
         << "    \"terminal_last_source_pass\":"
         << (terminal_last_source_pass ? "true" : "false") << "\n"
         << "  },\n";
  output << "  \"recycled_actor_witness\":";
  if (!recycled_witness) output << "null";
  else {
    const auto& witness = *recycled_witness;
    output << "{\"step\":" << witness.step
           << ",\"current_day\":" << witness.current_day
           << ",\"origin_day\":" << witness.origin_day
           << ",\"actor\":" << witness.actor
           << ",\"tile\":[" << witness.x << ',' << witness.y << ']'
           << ",\"transaction_id\":" << witness.transaction_id
           << ",\"pending_receipt\":"
           << (witness.pending_receipt ? "true" : "false") << '}';
  }
  output << ",\n  \"movement_mismatches\":[";
  for (std::size_t index = 0; index < movement_mismatches.size(); ++index) {
    if (index) output << ',';
    const auto& mismatch = movement_mismatches[index];
    output << "{\"day\":" << mismatch.day << ",\"actor\":"
           << mismatch.actor << ",\"baseline\":";
    write_numbers(output, mismatch.baseline);
    output << ",\"enabled\":";
    write_numbers(output, mismatch.enabled);
    output << ",\"baseline_active_hours\":";
    write_numbers(output, mismatch.baseline_active_hours);
    output << ",\"enabled_active_hours\":";
    write_numbers(output, mismatch.enabled_active_hours);
    output << ",\"baseline_move_hours\":";
    write_numbers(output, mismatch.baseline_move_hours);
    output << ",\"enabled_move_hours\":";
    write_numbers(output, mismatch.enabled_move_hours);
    output << ",\"enabled_window_source\":";
    write_numbers(output, mismatch.enabled_window_source);
    output << ",\"reason\":\"" << mismatch.reason << '"';
    output << '}';
  }
  output << "],\n  \"events\":[\n";
  for (std::size_t index = 0; index < events.size(); ++index) {
    const auto& event = events[index];
    output << "    {\"step\":" << event.step << ",\"day\":" << event.day
           << ",\"hour\":" << event.hour
           << ",\"new_transactions\":" << event.new_transactions
           << ",\"receipts_confirmed\":" << event.receipts_confirmed
           << ",\"receipts_failed\":" << event.receipts_failed
           << ",\"seed_orders\":" << event.seed_orders
           << ",\"seed_fills\":" << event.seed_fills
           << ",\"seed_zero_fills\":" << event.seed_zero_fills
           << ",\"open_transactions\":" << event.open_transactions
           << ",\"open_source_debts\":" << event.open_source_debts
           << ",\"open_purchase_debts\":" << event.open_purchase_debts
           << ",\"pending_unit_receipts\":"
           << event.pending_unit_receipts
           << ",\"pending_purchase_receipts\":"
           << event.pending_purchase_receipts
           << ",\"recycled_actor_collisions\":"
           << event.recycled_actor_collisions
           << ",\"stale_actor_pending\":" << event.stale_actor_pending
           << ",\"weeds_added_after_step\":"
           << event.weeds_added_after_step
           << ",\"drop_deposit_units\":" << event.drop_deposit_units
           << ",\"baseline_ops\":";
    write_numbers(output, event.baseline_ops);
    output << ",\"enabled_ops\":";
    write_numbers(output, event.enabled_ops);
    output << '}' << (index + 1 == events.size() ? "\n" : ",\n");
  }
  output << "  ],\n  \"open_transaction_details\":[";
  for (std::size_t index = 0; index < open_details.size(); ++index) {
    if (index) output << ',';
    const auto& detail = open_details[index];
    const auto& tx = detail.transaction;
    output << "{\"id\":" << tx.id << ",\"source_epoch\":"
           << tx.source_epoch << ",\"origin_actor\":" << tx.key.actor
           << ",\"origin_actor_generation\":" << tx.key.actor_generation
           << ",\"origin_day\":" << tx.origin_day
           << ",\"origin_turn\":" << tx.origin_turn
           << ",\"tile\":[" << tx.key.tile.column << ','
           << tx.key.tile.row << "],\"goal\":"
           << static_cast<int>(tx.goal) << ",\"desired_item\":"
           << tx.desired_item << ",\"tile_kind\":" << detail.tile_kind
           << ",\"tile_crop\":" << detail.tile_crop
           << ",\"desired_state\":\"" << detail.desired_state << '"'
           << ",\"current_claimant\":" << detail.current_claimant
           << ",\"confirmed_actions\":" << tx.confirmed_actions
           << ",\"confirmed_harvests\":" << tx.confirmed_harvests
           << ",\"failed_receipts\":" << tx.failed_receipts
           << ",\"outstanding_source_actions\":"
           << tx.outstanding_source_actions
           << ",\"pending\":" << (detail.pending ? "true" : "false")
           << ",\"critical\":" << (detail.critical ? "true" : "false")
           << ",\"recoverable_critical\":"
           << (detail.recoverable_critical ? "true" : "false")
           << ",\"lifecycle_controller_only\":"
           << (detail.lifecycle_controller_only ? "true" : "false")
           << ",\"last_receipt_step\":" << detail.last_receipt_step
           << ",\"last_receipt_confirmed\":"
           << (detail.last_receipt_confirmed ? "true" : "false") << '}';
  }
  output << "]\n}\n";
}

}  // namespace

int main(int argc, char** argv) try {
  const auto options = parse(argc, argv);
  const auto g001_tape = g001::repair::load_route(
      options.tapes, options.library, "G001");
  if (g001_tape.size() != 719)
    throw std::runtime_error("G001 tape is not exactly 719 actions");
  NativeTapeLibrary library;
  library.routes.push_back(g001_tape);
  NativeTeammateExecutor executor(std::move(library));
  if (options.repair_mask == 32) {
    run_v2_forced(options, g001_tape, executor);
    std::cout << "native_g001_forced_regression bit32 output="
              << options.output << '\n';
    return 0;
  }
  if (options.repair_mask == 64) {
    run_v3_forced(options, g001_tape, executor);
    std::cout << "native_g001_forced_regression bit64 output="
              << options.output << '\n';
    return 0;
  }

  fastkag::Config config;
  config.episode_steps = 720;  // Simulator executes source steps 0..718.
  config.weed_spawn_chance = 1.0;
  Simulator baseline(config, options.seed);
  Simulator enabled(config, options.seed);
  if (!same_config(baseline.config(), enabled.config()))
    throw std::runtime_error("baseline/bit8 Config fingerprint mismatch");
  std::array<NativeAgentState, 2> baseline_states;
  std::array<NativeAgentState, 2> enabled_states;
  NativeRepairAudit audit;
  const auto repair = fastkag::native_repair_options_from_mask(8);
  const fastkag::NativeRepairOptions baseline_repair{};
  if (baseline_repair.enabled() || !repair.state_driven_local_repair ||
      repair.weed_min_loss_realign || repair.animal_buy_retry ||
      repair.empty_stall_reuse || repair.route_cursor_production != 0)
    throw std::runtime_error("comparison differs by more than bit8");
  constexpr bool neutral_special_economy = true;
  std::map<std::pair<int, int>, std::vector<int>> baseline_moves;
  std::map<std::pair<int, int>, std::vector<int>> enabled_moves;
  std::map<std::pair<int, int>, std::vector<int>> baseline_move_hours;
  std::map<std::pair<int, int>, std::vector<int>> enabled_move_hours;
  std::map<std::pair<int, int>, std::vector<int>> enabled_window_source;
  std::map<std::pair<int, int>, std::vector<int>> baseline_active_hours;
  std::map<std::pair<int, int>, std::vector<int>> enabled_active_hours;
  std::vector<Event> events;
  std::optional<RecycledWitness> recycled_witness;
  std::set<std::uint64_t> identity_polluted_transactions;
  std::set<std::pair<int, int>> identity_polluted_day_actors;
  std::map<std::uint64_t, int> last_receipt_steps;
  std::map<std::uint64_t, bool> last_receipt_results;
  int boundary_receipts_confirmed = 0;
  int boundary_receipts_failed = 0;
  int day_boundaries_with_new_weeds = 0;
  int last_new_weed_step = -1;
  int last_emitted_harvest_step = -1;
  int last_harvest_step = -1;
  int last_harvest_receipt_step = -1;
  int last_drop_step = -1;
  int last_drop_deposit_units = 0;
  int turns = 0;

  while (!baseline.done() && !enabled.done()) {
    const int step = enabled.step_count();
    const int day = enabled.day();
    const int hour = enabled.hour();
    const auto compiler_before = enabled_states[0].experimental_event_local_repair
        ? enabled_states[0].experimental_event_local_repair->audit()
        : g001::event_local_repair::Audit{};
    const auto open_before = enabled_states[0].experimental_event_local_repair
        ? enabled_states[0].experimental_event_local_repair->open_transactions()
        : std::vector<g001::event_local_repair::TransactionView>{};
    const auto audit_before = audit;
    const auto prior_pending =
        enabled_states[0].experimental_event_local_pending;

    std::array<PlayerAction, 2> baseline_actions{
        executor.action_external(
            baseline, 0, 0, baseline_states[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr,
            neutral_special_economy, baseline_repair),
        executor.action_external(baseline, 1, 0, baseline_states[1])};
    std::array<PlayerAction, 2> enabled_actions{
        executor.action_external(
            enabled, 0, 0, enabled_states[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr,
            neutral_special_economy, repair, &audit),
        executor.action_external(enabled, 1, 0, enabled_states[1])};
    append_moves(baseline_actions[0], day, hour, baseline_moves,
                 baseline_move_hours);
    append_moves(enabled_actions[0], day, hour, enabled_moves,
                 enabled_move_hours);
    append_active_hours(baseline_actions[0], day, hour,
                        baseline_active_hours);
    append_active_hours(enabled_actions[0], day, hour,
                        enabled_active_hours);
    for (std::size_t actor = 0; actor < enabled_actions[0].units.size();
         ++actor) {
      const Action source = actor < g001_tape[static_cast<std::size_t>(step)].units.size()
          ? g001_tape[static_cast<std::size_t>(step)].units[actor] : Action{};
      if (movement(source.op))
        enabled_window_source[{day, static_cast<int>(actor)}].push_back(
            static_cast<int>(source.op));
    }

    const auto compiler_after =
        enabled_states[0].experimental_event_local_repair->audit();
    const int new_transactions = compiler_after.transactions_created -
        compiler_before.transactions_created;
    const int confirmed = audit.local_repair_receipts_confirmed -
        audit_before.local_repair_receipts_confirmed;
    const int failed = audit.local_repair_receipts_failed -
        audit_before.local_repair_receipts_failed;
    if (!prior_pending.empty() && hour == 0) {
      boundary_receipts_confirmed += confirmed;
      boundary_receipts_failed += failed;
    }
    if (confirmed > 0 && std::any_of(
            prior_pending.begin(), prior_pending.end(), [](const auto& receipt) {
              return receipt.emitted == Op::HARVEST;
            }))
      last_harvest_receipt_step = step;

    const auto open = enabled_states[0].experimental_event_local_repair
        ->open_transactions();
    for (const auto& pending : prior_pending) {
      const auto before = std::find_if(
          open_before.begin(), open_before.end(), [&](const auto& value) {
            return value.id == pending.transaction_id;
          });
      const auto after = std::find_if(
          open.begin(), open.end(), [&](const auto& value) {
            return value.id == pending.transaction_id;
          });
      const bool receipt_confirmed = before != open_before.end() &&
          (after == open.end() ||
           after->confirmed_actions > before->confirmed_actions);
      last_receipt_steps[pending.transaction_id] = step;
      last_receipt_results[pending.transaction_id] = receipt_confirmed;
    }
    const auto current_positions = positions(enabled, 0);
    int open_source_debts = 0;
    int recycled_collisions = 0;
    int stale_pending = 0;
    // Transaction origin generation is provenance, not ownership. A plot
    // debt surviving its original hand is therefore not identity pollution.
    // Only a malformed receipt lease token can cross generations here: every
    // native pending entry is created after exact Compiler::commit().
    for (const auto& pending : prior_pending) {
      const int submitted_day = pending.submitted_step /
          config.turns_per_day;
      const bool invalid_lease = pending.actor_generation !=
          actor_generation(submitted_day, pending.actor);
      if (!invalid_lease) continue;
      ++recycled_collisions;
      if (confirmed > 0) {
        ++stale_pending;
        identity_polluted_transactions.insert(pending.transaction_id);
        identity_polluted_day_actors.insert({day, pending.actor});
        if (!recycled_witness)
          recycled_witness = RecycledWitness{
              step, day, submitted_day, pending.actor,
              pending.tile.x, pending.tile.y, pending.transaction_id, true};
      }
    }
    for (const auto& transaction : open) {
      open_source_debts += transaction.outstanding_source_actions;
    }

    Event event;
    event.step = step;
    event.day = day;
    event.hour = hour;
    event.new_transactions = new_transactions;
    event.receipts_confirmed = confirmed;
    event.receipts_failed = failed;
    event.seed_orders = audit.local_repair_seed_orders -
        audit_before.local_repair_seed_orders;
    event.seed_fills = audit.local_repair_seed_fills -
        audit_before.local_repair_seed_fills;
    event.seed_zero_fills = audit.local_repair_seed_zero_fills -
        audit_before.local_repair_seed_zero_fills;
    event.open_transactions = static_cast<int>(open.size());
    event.open_source_debts = open_source_debts;
    event.open_purchase_debts = open_purchase_debts(enabled_states[0]);
    event.pending_unit_receipts = static_cast<int>(
        enabled_states[0].experimental_event_local_pending.size());
    event.pending_purchase_receipts =
        pending_purchase_receipts(enabled_states[0]);
    event.recycled_actor_collisions = recycled_collisions;
    event.stale_actor_pending = stale_pending;
    for (const auto& unit : baseline_actions[0].units)
      event.baseline_ops.push_back(static_cast<int>(unit.op));
    bool emitted_drop = false;
    for (const auto& unit : enabled_actions[0].units) {
      event.enabled_ops.push_back(static_cast<int>(unit.op));
      if (unit.op == Op::HARVEST) last_emitted_harvest_step = step;
      if (unit.op == Op::DROP) {
        emitted_drop = true;
        last_drop_step = step;
      }
    }
    for (const auto& pending :
         enabled_states[0].experimental_event_local_pending)
      if (pending.emitted == Op::HARVEST) last_harvest_step = step;

    const int weeds_before = weed_count(enabled, 0);
    const int shed_before = shed_total(enabled, 0);
    const auto unit_preview = enabled.preview_unit_phase(enabled_actions);
    const int deposited = std::max(0, shed_total(unit_preview, 0) - shed_before);
    if (emitted_drop) last_drop_deposit_units = deposited;
    if (deposited > 0) {
      event.drop_deposit_units = deposited;
    }
    baseline.step(baseline_actions);
    enabled.step(enabled_actions);
    const int weeds_added = std::max(0, weed_count(enabled, 0) - weeds_before);
    if (weeds_added > 0) {
      event.weeds_added_after_step = weeds_added;
      last_new_weed_step = step;
      if (hour == config.turns_per_day - 1)
        ++day_boundaries_with_new_weeds;
    }
    const bool move_difference = [&] {
      const auto actors = std::max(event.baseline_ops.size(),
                                   event.enabled_ops.size());
      for (std::size_t actor = 0; actor < actors; ++actor) {
        const int baseline_op = actor < event.baseline_ops.size()
            ? event.baseline_ops[actor] : static_cast<int>(Op::PASS);
        const int enabled_op = actor < event.enabled_ops.size()
            ? event.enabled_ops[actor] : static_cast<int>(Op::PASS);
        if (baseline_op != enabled_op &&
            (movement(static_cast<Op>(baseline_op)) ||
             movement(static_cast<Op>(enabled_op)))) return true;
      }
      return false;
    }();
    const bool interesting = new_transactions || confirmed || failed ||
        event.seed_orders || event.seed_fills || event.seed_zero_fills ||
        recycled_collisions || stale_pending || weeds_added || deposited ||
        move_difference || hour == 0 ||
        hour == config.turns_per_day - 1 || step >= 707;
    if (interesting) events.push_back(std::move(event));
    ++turns;
  }

  std::vector<MoveMismatch> movement_mismatches;
  std::map<std::pair<int, int>, bool> keys;
  for (const auto& [key, value] : baseline_moves) {
    (void)value;
    keys[key] = true;
  }
  for (const auto& [key, value] : enabled_moves) {
    (void)value;
    keys[key] = true;
  }
  for (const auto& [key, unused] : keys) {
    (void)unused;
    const auto baseline_sequence = baseline_moves[key];
    const auto enabled_sequence = enabled_moves[key];
    if (baseline_sequence != enabled_sequence) {
      const auto baseline_hours = baseline_active_hours[key];
      const auto enabled_hours = enabled_active_hours[key];
      std::string reason;
      const auto source_sequence = enabled_window_source[key];
      if (baseline_hours != enabled_hours)
        reason = "economic-feasibility-actor-availability-diverged";
      else if (enabled_sequence == source_sequence)
        reason = "baseline-state-overlay-diverged-enabled-matches-source";
      else if (identity_polluted_day_actors.contains(key))
        reason = "recycled-actor-transaction-pollution";
      else
        reason = "enabled-same-active-window-move-invariant-breach";
      MoveMismatch mismatch;
      mismatch.day = key.first;
      mismatch.actor = key.second;
      mismatch.baseline = baseline_sequence;
      mismatch.enabled = enabled_sequence;
      mismatch.baseline_active_hours = baseline_hours;
      mismatch.enabled_active_hours = enabled_hours;
      mismatch.baseline_move_hours = baseline_move_hours[key];
      mismatch.enabled_move_hours = enabled_move_hours[key];
      mismatch.enabled_window_source = source_sequence;
      mismatch.reason = std::move(reason);
      movement_mismatches.push_back(std::move(mismatch));
    }
  }

  const auto compiler_audit = enabled_states[0].experimental_event_local_repair
      ->audit();
  const auto final_open = enabled_states[0].experimental_event_local_repair
      ->open_transactions();
  OpenAttribution attribution;
  std::vector<OpenDetail> open_details;
  attribution.total = static_cast<int>(final_open.size());
  const auto& final_tiles = enabled.farms()[0].tiles;
  const auto final_positions = positions(enabled, 0);
  for (const auto& transaction : final_open) {
    attribution.externalized_source_units +=
        transaction.outstanding_source_actions;
    const bool identity_polluted =
        identity_polluted_transactions.contains(transaction.id);
    const bool terminal_pending = std::any_of(
        enabled_states[0].experimental_event_local_pending.begin(),
        enabled_states[0].experimental_event_local_pending.end(),
        [&](const auto& pending) {
          return pending.transaction_id == transaction.id;
        });
    const int tile_index = transaction.key.tile.row * config.board_size +
        transaction.key.tile.column;
    const bool valid_tile = tile_index >= 0 &&
        tile_index < static_cast<int>(final_tiles.size());
    const bool valid_crop = transaction.desired_item >= 0 &&
        transaction.desired_item < fastkag::N_CROPS;
    const bool incompatible_tile = valid_tile &&
        (final_tiles[static_cast<std::size_t>(tile_index)].kind ==
             fastkag::TileKind::LOCKED ||
         final_tiles[static_cast<std::size_t>(tile_index)].kind ==
             fastkag::TileKind::COOP ||
         final_tiles[static_cast<std::size_t>(tile_index)].kind ==
             fastkag::TileKind::PASTURE ||
         final_tiles[static_cast<std::size_t>(tile_index)].kind ==
             fastkag::TileKind::ANIMAL ||
         (final_tiles[static_cast<std::size_t>(tile_index)].kind ==
              fastkag::TileKind::PLANT &&
          valid_crop &&
          static_cast<int>(final_tiles[static_cast<std::size_t>(tile_index)]
                               .crop) != transaction.desired_item));
    OpenDetail detail;
    detail.transaction = transaction;
    detail.pending = std::any_of(
        enabled_states[0].experimental_event_local_pending.begin(),
        enabled_states[0].experimental_event_local_pending.end(),
        [&](const auto& pending) {
          return pending.transaction_id == transaction.id;
        });
    detail.critical = transaction.outstanding_source_actions > 0;
    detail.recoverable_critical = detail.critical && valid_tile &&
        valid_crop && !incompatible_tile;
    detail.lifecycle_controller_only =
        transaction.outstanding_source_actions == 0;
    if (valid_tile) {
      const auto& tile = final_tiles[static_cast<std::size_t>(tile_index)];
      detail.tile_kind = static_cast<int>(tile.kind);
      detail.tile_crop = static_cast<int>(tile.crop);
      if (tile.kind == fastkag::TileKind::WEED)
        detail.desired_state = "weed";
      else if (tile.kind == fastkag::TileKind::EMPTY)
        detail.desired_state = "empty";
      else if (tile.kind == fastkag::TileKind::PLANT && valid_crop &&
               static_cast<int>(tile.crop) == transaction.desired_item)
        detail.desired_state = tile.yield_units > 0
            ? "desired-crop-live" : "desired-crop-exhausted";
      else if (tile.kind == fastkag::TileKind::PLANT)
        detail.desired_state = "wrong-crop";
      else
        detail.desired_state = "structure-or-animal";
    } else {
      detail.desired_state = "invalid-tile";
    }
    for (std::size_t actor = 0; actor < final_positions.size(); ++actor)
      if (same_position(final_positions[actor], transaction.key.tile)) {
        detail.current_claimant = static_cast<int>(actor);
        break;
      }
    if (last_receipt_steps.contains(transaction.id)) {
      detail.last_receipt_step = last_receipt_steps[transaction.id];
      detail.last_receipt_confirmed = last_receipt_results[transaction.id];
    }
    if (detail.critical) {
      ++attribution.critical_source_transactions;
      attribution.critical_source_units +=
          transaction.outstanding_source_actions;
    }
    if (detail.recoverable_critical)
      ++attribution.recoverable_critical_transactions;
    if (detail.lifecycle_controller_only)
      ++attribution.lifecycle_controller_only;
    open_details.push_back(std::move(detail));
    if (identity_polluted) {
      ++attribution.identity_polluted;
    } else if (terminal_pending) {
      ++attribution.terminal_pending;
    } else if (!valid_tile || !valid_crop || incompatible_tile) {
      ++attribution.desired_crop_impossible;
    } else if (transaction.outstanding_source_actions > 0) {
      ++attribution.source_debt_externalized;
    } else {
      ++attribution.lifecycle_incomplete;
    }
  }
  const bool terminal_last_source_pass = std::all_of(
      g001_tape.back().units.begin(), g001_tape.back().units.end(),
      [](const Action& action) { return action.op == Op::PASS; });
  const bool harvest_receipt_closed = last_harvest_step < 0 ||
      last_harvest_receipt_step == last_harvest_step + 1;
  const bool enabled_move_overwrite = std::any_of(
      movement_mismatches.begin(), movement_mismatches.end(),
      [](const MoveMismatch& mismatch) {
        return mismatch.reason ==
                   "enabled-same-active-window-move-invariant-breach" ||
            mismatch.reason == "recycled-actor-transaction-pollution";
      });
  const bool objective_count_bounded = std::all_of(
      final_open.begin(), final_open.end(), [](const auto& transaction) {
        return transaction.outstanding_source_actions == 0 ||
            transaction.outstanding_source_actions == 1;
      });
  const bool gates_passed = turns == 719 &&
      compiler_audit.transactions_created > 0 &&
      audit.local_repair_receipts_confirmed > 0 &&
      !enabled_move_overwrite && objective_count_bounded &&
      !recycled_witness &&
      enabled_states[0].experimental_event_local_pending.empty() &&
      pending_purchase_receipts(enabled_states[0]) == 0 &&
      harvest_receipt_closed && terminal_last_source_pass &&
      last_drop_step >= 0 && last_drop_deposit_units > 0 &&
      day_boundaries_with_new_weeds > 0 &&
      attribution.critical_source_transactions == 0;
  write_report(options, events, movement_mismatches, recycled_witness, audit,
               compiler_audit, enabled_states[0], turns,
               boundary_receipts_confirmed, boundary_receipts_failed,
               day_boundaries_with_new_weeds, last_new_weed_step,
               last_harvest_step, last_harvest_receipt_step,
               last_drop_step, last_drop_deposit_units,
               last_emitted_harvest_step, attribution, open_details,
               baseline.farms()[0].money, enabled.farms()[0].money,
               terminal_last_source_pass, gates_passed);
  std::cout << "native_g001_forced_regression turns=" << turns
            << " triggers=" << compiler_audit.transactions_created
            << " receipts=" << audit.local_repair_receipts_confirmed
            << " open="
            << enabled_states[0].experimental_event_local_repair
                   ->open_transactions().size()
            << " move_mismatches=" << movement_mismatches.size()
            << " critical_objectives="
            << attribution.critical_source_transactions
            << " recycled_pollution=" << (recycled_witness ? 1 : 0)
            << " report=" << options.output << '\n';
  return gates_passed ? 0 : 3;
} catch (const std::exception& error) {
  std::cerr << "native_g001_forced_regression: " << error.what() << '\n';
  return 2;
}
