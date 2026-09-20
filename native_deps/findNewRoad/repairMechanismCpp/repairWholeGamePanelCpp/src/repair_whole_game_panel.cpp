#include "repair_whole_game_panel.hpp"

#include "repair_fork_evaluator.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace g001::repair_whole_game_panel {
namespace {

using fastkag::Action;
using fastkag::NativeAgentState;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;

bool action_equal(const Action& lhs, const Action& rhs) {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool action_vector_equal(const std::vector<Action>& lhs,
                         const std::vector<Action>& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), action_equal);
}

bool player_action_equal(const PlayerAction& lhs, const PlayerAction& rhs) {
  return action_vector_equal(lhs.units, rhs.units) &&
         action_vector_equal(lhs.market, rhs.market);
}

bool joint_action_equal(const std::array<PlayerAction, 2>& lhs,
                        const std::array<PlayerAction, 2>& rhs) {
  return player_action_equal(lhs[0], rhs[0]) &&
         player_action_equal(lhs[1], rhs[1]);
}

bool tile_equal(const fastkag::Tile& lhs, const fastkag::Tile& rhs) {
  return lhs.kind == rhs.kind && lhs.crop == rhs.crop &&
         lhs.animal == rhs.animal && lhs.planted_day == rhs.planted_day &&
         lhs.placed_day == rhs.placed_day &&
         lhs.yield_units == rhs.yield_units &&
         lhs.consecutive_unwatered == rhs.consecutive_unwatered &&
         lhs.consecutive_unfed == rhs.consecutive_unfed &&
         lhs.fertilized_until_day == rhs.fertilized_until_day &&
         lhs.pending_care_bonus == rhs.pending_care_bonus &&
         lhs.max_lifespan_step == rhs.max_lifespan_step &&
         lhs.watered_today == rhs.watered_today &&
         lhs.fed_today == rhs.fed_today &&
         lhs.cared_today == rhs.cared_today &&
         lhs.fertilizer_available == rhs.fertilizer_available;
}

bool exact_state_equal(const Simulator& lhs, const Simulator& rhs) {
  if (lhs.step_count() != rhs.step_count() || lhs.done() != rhs.done() ||
      lhs.market().inventory != rhs.market().inventory ||
      lhs.market().prices != rhs.market().prices ||
      lhs.shops() != rhs.shops() ||
      lhs.last_market_fills() != rhs.last_market_fills() ||
      lhs.last_market_cash_shortfalls() !=
          rhs.last_market_cash_shortfalls() ||
      lhs.last_end_of_day_overflow() != rhs.last_end_of_day_overflow())
    return false;
  for (int player = 0; player < 2; ++player) {
    const auto& left_farm = lhs.farms()[player];
    const auto& right_farm = rhs.farms()[player];
    if (left_farm.money != right_farm.money ||
        left_farm.farmer.x != right_farm.farmer.x ||
        left_farm.farmer.y != right_farm.farmer.y ||
        left_farm.hands.size() != right_farm.hands.size() ||
        left_farm.tiles.size() != right_farm.tiles.size() ||
        left_farm.unlocked_mask != right_farm.unlocked_mask ||
        left_farm.hires_today != right_farm.hires_today)
      return false;
    for (std::size_t i = 0; i < left_farm.hands.size(); ++i) {
      if (left_farm.hands[i].x != right_farm.hands[i].x ||
          left_farm.hands[i].y != right_farm.hands[i].y)
        return false;
    }
    for (std::size_t i = 0; i < left_farm.tiles.size(); ++i)
      if (!tile_equal(left_farm.tiles[i], right_farm.tiles[i])) return false;
    const auto& left_private = lhs.privates()[player];
    const auto& right_private = rhs.privates()[player];
    if (left_private.shed != right_private.shed ||
        left_private.seeds != right_private.seeds ||
        left_private.inventories != right_private.inventories ||
        left_private.inventory_order != right_private.inventory_order)
      return false;
  }
  return true;
}

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
         op == Op::WEST;
}

int count_moves(const PlayerAction& action) {
  return static_cast<int>(std::count_if(
      action.units.begin(), action.units.end(),
      [](const auto& unit) { return is_move(unit.op); }));
}

int move_mismatches(const PlayerAction& lhs, const PlayerAction& rhs) {
  const auto common = std::min(lhs.units.size(), rhs.units.size());
  int mismatches =
      static_cast<int>(std::max(lhs.units.size(), rhs.units.size()) - common);
  for (std::size_t actor = 0; actor < common; ++actor) {
    const bool left_move = is_move(lhs.units[actor].op);
    const bool right_move = is_move(rhs.units[actor].op);
    mismatches += (left_move || right_move) &&
                  !action_equal(lhs.units[actor], rhs.units[actor]);
  }
  return mismatches;
}

double score(double own, double opponent) {
  return own > opponent ? 1.0 : own < opponent ? 0.0 : 0.5;
}

MoneyMetrics money_metrics(const Simulator& env, int seat) {
  const double own = env.farms()[seat].money;
  const double opponent = env.farms()[1 - seat].money;
  return {own, opponent, own - opponent, score(own, opponent)};
}

std::array<double, 2> reward(const Simulator& env) {
  if (!env.done()) return {};
  return {env.farms()[0].money, env.farms()[1].money};
}

fastkag::Config config_for(Scenario scenario) {
  fastkag::Config config;
  if (scenario != Scenario::Normal) config.weed_spawn_chance = 1.0;
  return config;
}

bool evidence_valid(const AdapterEvidence& value) {
  return value.receipt_checks >= 0 && value.receipts_accepted >= 0 &&
         value.receipts_failed >= 0 && value.moves_expected >= 0 &&
         value.moves_emitted >= 0 && value.move_failures >= 0 &&
         value.debts_opened >= 0 && value.debts_closed >= 0 &&
         value.failures >= 0 &&
         value.receipts_accepted + value.receipts_failed ==
             value.receipt_checks;
}

GameMetrics run_game(const NativeTeammateExecutor& executor,
                     std::uint64_t seed, int seat, Scenario scenario,
                     const FinalActionCallback& callback) {
  Simulator initial(config_for(scenario), seed);
  Simulator baseline = initial;
  Simulator repair = initial;
  std::array<NativeAgentState, 2> initial_provider_state;
  auto baseline_provider_state = initial_provider_state;
  auto repair_provider_state = initial_provider_state;
  GameMetrics out;
  out.seed = seed;
  out.seat = seat;
  out.scenario = scenario;

  while (!baseline.done()) {
    if (repair.done() || baseline.step_count() != repair.step_count())
      throw std::runtime_error("paired whole-game clocks diverged");
    const int step = baseline.step_count();
    const auto before_baseline_fingerprint =
        repair_fork::full_unit_phase_state_fingerprint(baseline);
    const auto before_repair_fingerprint =
        repair_fork::full_unit_phase_state_fingerprint(repair);
    if (before_baseline_fingerprint != before_repair_fingerprint ||
        !exact_state_equal(baseline, repair))
      throw std::runtime_error("paired whole-game pre-action states diverged");

    std::array<PlayerAction, 2> baseline_actions;
    std::array<PlayerAction, 2> repair_actions;
    std::array<NativeAgentState, 2> provider_before = repair_provider_state;
    for (int player = 0; player < 2; ++player) {
      baseline_actions[player] = executor.action_external(
          baseline, player, 0, baseline_provider_state[player]);
      repair_actions[player] = executor.action_external(
          repair, player, 0, repair_provider_state[player]);
    }

    bool final_receipt_accepted = true;
    AdapterEvidence adapter;
    if (callback) {
      const FinalActionContext context{
          repair, executor, seat, 0, step, repair_actions,
          repair_actions[seat],
          provider_before[seat], repair_provider_state[seat]};
      auto decision = callback(context);
      final_receipt_accepted =
          decision.final_action.units.size() ==
              repair_actions[seat].units.size() &&
          decision.final_action.market.size() <=
              static_cast<std::size_t>(repair.config().max_market_orders) &&
          evidence_valid(decision.evidence);
      if (final_receipt_accepted) {
        repair_actions[seat] = std::move(decision.final_action);
        adapter = decision.evidence;
        if (decision.committed_provider_state) {
          repair_provider_state[seat] =
              std::move(*decision.committed_provider_state);
          ++out.provider_state_commits;
        }
      } else {
        repair_actions[seat] = baseline_actions[seat];
        ++out.callback_failures;
      }
    }

    HandMetrics hand;
    hand.step = step;
    hand.callback_enabled = static_cast<bool>(callback);
    hand.final_action_receipt_accepted = final_receipt_accepted;
    ++out.final_action_receipt_checks;
    if (final_receipt_accepted)
      ++out.final_action_receipts_accepted;
    else
      ++out.final_action_receipts_failed;
    hand.action_equal = joint_action_equal(baseline_actions, repair_actions);
    out.action_mismatches += !hand.action_equal;
    hand.baseline_moves = count_moves(baseline_actions[seat]);
    hand.repair_moves = count_moves(repair_actions[seat]);
    hand.move_action_mismatches =
        move_mismatches(baseline_actions[seat], repair_actions[seat]);
    out.baseline_moves += hand.baseline_moves;
    out.repair_moves += hand.repair_moves;
    out.move_action_mismatches += hand.move_action_mismatches;

    hand.baseline_unit_failures = fastkag::native_macro_unit_failures(
        baseline, seat, baseline_actions[seat]);
    hand.repair_unit_failures = fastkag::native_macro_unit_failures(
        repair, seat, repair_actions[seat]);
    out.baseline_unit_failures += hand.baseline_unit_failures;
    out.repair_unit_failures += hand.repair_unit_failures;
    baseline.step(baseline_actions);
    repair.step(repair_actions);
    hand.baseline_market_failures = fastkag::native_macro_market_failures(
        baseline, seat, baseline_actions[seat]);
    hand.repair_market_failures = fastkag::native_macro_market_failures(
        repair, seat, repair_actions[seat]);
    out.baseline_market_failures += hand.baseline_market_failures;
    out.repair_market_failures += hand.repair_market_failures;

    hand.state_equal = exact_state_equal(baseline, repair) &&
                       repair_fork::full_unit_phase_state_fingerprint(
                           baseline) ==
                           repair_fork::full_unit_phase_state_fingerprint(
                               repair);
    out.state_mismatches += !hand.state_equal;
    hand.baseline_reward = reward(baseline);
    hand.repair_reward = reward(repair);
    hand.reward_equal = hand.baseline_reward == hand.repair_reward;
    out.reward_mismatches += !hand.reward_equal;

    out.adapter_receipt_checks += adapter.receipt_checks;
    out.adapter_receipts_accepted += adapter.receipts_accepted;
    out.adapter_receipts_failed += adapter.receipts_failed;
    out.adapter_moves_expected += adapter.moves_expected;
    out.adapter_moves_emitted += adapter.moves_emitted;
    out.adapter_move_failures += adapter.move_failures;
    out.debts_opened += adapter.debts_opened;
    out.debts_closed += adapter.debts_closed;
    out.callback_failures += adapter.failures;
    hand.outstanding_debts = out.debts_opened - out.debts_closed;
    hand.callback_failures = out.callback_failures;
    hand.adapter = adapter;
    hand.baseline = money_metrics(baseline, seat);
    hand.repair = money_metrics(repair, seat);
    out.hands.push_back(hand);
    ++out.steps;
  }

  if (!repair.done())
    throw std::runtime_error("repair whole-game arm did not terminate");
  out.terminal_debts = out.debts_opened - out.debts_closed;
  out.baseline_terminal = money_metrics(baseline, seat);
  out.repair_terminal = money_metrics(repair, seat);
  return out;
}

void write_money(std::ostream& out, const MoneyMetrics& value) {
  out << "{\"own\":" << value.own << ",\"opponent\":"
      << value.opponent << ",\"margin\":" << value.margin
      << ",\"score\":" << value.score << '}';
}

}  // namespace

const char* scenario_name(Scenario scenario) noexcept {
  switch (scenario) {
    case Scenario::Normal: return "normal";
    case Scenario::ForcedWeed: return "forced_weed";
    case Scenario::ForcedCropWeed: return "forced_crop_weed";
  }
  return "unknown";
}

bool GameMetrics::exact_parity() const noexcept {
  return action_mismatches == 0 && state_mismatches == 0 &&
         reward_mismatches == 0 &&
         final_action_receipts_failed == 0 &&
         baseline_terminal.own == repair_terminal.own &&
         baseline_terminal.opponent == repair_terminal.opponent &&
         baseline_terminal.margin == repair_terminal.margin &&
         baseline_terminal.score == repair_terminal.score;
}

bool Report::exact_parity() const noexcept {
  return games_with_exact_parity == static_cast<int>(games.size()) &&
         action_mismatches == 0 && state_mismatches == 0 &&
         reward_mismatches == 0;
}

Report evaluate(const Options& options, FinalActionCallback callback) {
  if (options.tapes.empty() || options.library.empty() ||
      options.route.empty() || options.seeds <= 0 || options.seats.empty() ||
      options.scenarios.empty())
    throw std::invalid_argument("whole-game panel options are incomplete");
  for (const int seat : options.seats)
    if (seat < 0 || seat >= 2)
      throw std::invalid_argument("whole-game panel seat must be 0 or 1");

  NativeTapeLibrary library;
  library.routes.push_back(
      repair::load_route(options.tapes, options.library, options.route));
  NativeTeammateExecutor executor(std::move(library));
  Report report;
  report.callback_enabled = static_cast<bool>(callback);
  for (int seed_offset = 0; seed_offset < options.seeds; ++seed_offset) {
    const auto seed = options.seed_begin +
                      static_cast<std::uint64_t>(seed_offset);
    for (const auto scenario : options.scenarios) {
      for (const int seat : options.seats) {
        auto game = run_game(executor, seed, seat, scenario, callback);
        report.games_with_exact_parity += game.exact_parity();
        report.action_mismatches += game.action_mismatches;
        report.state_mismatches += game.state_mismatches;
        report.reward_mismatches += game.reward_mismatches;
        report.terminal_debts += game.terminal_debts;
        report.failures += game.callback_failures +
                           game.adapter_receipts_failed +
                           game.adapter_move_failures;
        report.games.push_back(std::move(game));
      }
    }
  }
  return report;
}

std::string Report::json() const {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6)
      << "{\"schema\":\"repair-whole-game-panel-v1\","
      << "\"callback_enabled\":"
      << (callback_enabled ? "true" : "false")
      << ",\"benefit_claimed\":false,\"exact_parity\":"
      << (exact_parity() ? "true" : "false")
      << ",\"aggregate\":{\"games\":" << games.size()
      << ",\"games_with_exact_parity\":" << games_with_exact_parity
      << ",\"action_mismatches\":" << action_mismatches
      << ",\"state_mismatches\":" << state_mismatches
      << ",\"reward_mismatches\":" << reward_mismatches
      << ",\"terminal_debts\":" << terminal_debts
      << ",\"failures\":" << failures << "},\"games\":[";
  for (std::size_t game_index = 0; game_index < games.size(); ++game_index) {
    if (game_index) out << ',';
    const auto& game = games[game_index];
    out << "{\"seed\":" << game.seed << ",\"seat\":" << game.seat
        << ",\"scenario\":\"" << scenario_name(game.scenario)
        << "\",\"steps\":" << game.steps << ",\"exact_parity\":"
        << (game.exact_parity() ? "true" : "false")
        << ",\"parity\":{\"action_mismatches\":"
        << game.action_mismatches << ",\"state_mismatches\":"
        << game.state_mismatches << ",\"reward_mismatches\":"
        << game.reward_mismatches
        << "},\"final_action_receipts\":{\"checks\":"
        << game.final_action_receipt_checks << ",\"accepted\":"
        << game.final_action_receipts_accepted << ",\"failed\":"
        << game.final_action_receipts_failed
        << "},\"moves\":{\"baseline\":" << game.baseline_moves
        << ",\"repair\":" << game.repair_moves
        << ",\"action_mismatches\":" << game.move_action_mismatches
        << ",\"adapter_expected\":" << game.adapter_moves_expected
        << ",\"adapter_emitted\":" << game.adapter_moves_emitted
        << ",\"adapter_failures\":" << game.adapter_move_failures
        << "},\"debts\":{\"opened\":" << game.debts_opened
        << ",\"closed\":" << game.debts_closed << ",\"terminal\":"
        << game.terminal_debts << "},\"failures\":{\"callback\":"
        << game.callback_failures << ",\"baseline_unit\":"
        << game.baseline_unit_failures << ",\"repair_unit\":"
        << game.repair_unit_failures << ",\"baseline_market\":"
        << game.baseline_market_failures << ",\"repair_market\":"
        << game.repair_market_failures
        << "},\"adapter_receipts\":{\"checks\":"
        << game.adapter_receipt_checks << ",\"accepted\":"
        << game.adapter_receipts_accepted << ",\"failed\":"
        << game.adapter_receipts_failed
        << "},\"baseline_terminal\":";
    write_money(out, game.baseline_terminal);
    out << ",\"repair_terminal\":";
    write_money(out, game.repair_terminal);
    out << ",\"hands\":[";
    for (std::size_t hand_index = 0; hand_index < game.hands.size();
         ++hand_index) {
      if (hand_index) out << ',';
      const auto& hand = game.hands[hand_index];
      out << "{\"step\":" << hand.step
          << ",\"final_action_receipt_accepted\":"
          << (hand.final_action_receipt_accepted ? "true" : "false")
          << ",\"action_equal\":"
          << (hand.action_equal ? "true" : "false")
          << ",\"state_equal\":"
          << (hand.state_equal ? "true" : "false")
          << ",\"reward_equal\":"
          << (hand.reward_equal ? "true" : "false")
          << ",\"moves\":{\"baseline\":" << hand.baseline_moves
          << ",\"repair\":" << hand.repair_moves
          << ",\"mismatches\":" << hand.move_action_mismatches
          << "},\"outstanding_debts\":" << hand.outstanding_debts
          << ",\"callback_failures\":" << hand.callback_failures
          << ",\"failures\":{\"baseline_unit\":"
          << hand.baseline_unit_failures << ",\"repair_unit\":"
          << hand.repair_unit_failures << ",\"baseline_market\":"
          << hand.baseline_market_failures << ",\"repair_market\":"
          << hand.repair_market_failures << '}'
          << ",\"adapter_evidence\":{\"receipt_checks\":"
          << hand.adapter.receipt_checks << ",\"receipts_accepted\":"
          << hand.adapter.receipts_accepted << ",\"receipts_failed\":"
          << hand.adapter.receipts_failed << ",\"moves_expected\":"
          << hand.adapter.moves_expected << ",\"moves_emitted\":"
          << hand.adapter.moves_emitted << ",\"move_failures\":"
          << hand.adapter.move_failures << ",\"debts_opened\":"
          << hand.adapter.debts_opened << ",\"debts_closed\":"
          << hand.adapter.debts_closed << ",\"failures\":"
          << hand.adapter.failures << '}'
          << ",\"baseline_reward\":[" << hand.baseline_reward[0] << ','
          << hand.baseline_reward[1] << "],\"repair_reward\":["
          << hand.repair_reward[0] << ',' << hand.repair_reward[1]
          << "],\"baseline\":";
      write_money(out, hand.baseline);
      out << ",\"repair\":";
      write_money(out, hand.repair);
      out << '}';
    }
    out << "]}";
  }
  out << "]}";
  return out.str();
}

}  // namespace g001::repair_whole_game_panel
