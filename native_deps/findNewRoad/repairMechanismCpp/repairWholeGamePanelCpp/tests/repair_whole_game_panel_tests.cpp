#include "repair_whole_game_panel.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#ifndef REPAIR_WHOLE_GAME_TAPES
#error REPAIR_WHOLE_GAME_TAPES must be defined
#endif
#ifndef REPAIR_WHOLE_GAME_LIBRARY
#error REPAIR_WHOLE_GAME_LIBRARY must be defined
#endif

namespace panel = g001::repair_whole_game_panel;

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

panel::Options options() {
  panel::Options out;
  out.tapes = REPAIR_WHOLE_GAME_TAPES;
  out.library = REPAIR_WHOLE_GAME_LIBRARY;
  out.seed_begin = 970017;
  out.seeds = 1;
  out.seats = {0, 1};
  out.scenarios = {panel::Scenario::Normal, panel::Scenario::ForcedWeed};
  return out;
}

void disabled_default_off_exact_parity() {
  const auto report = panel::evaluate(options());
  require(!report.callback_enabled, "disabled panel enabled a callback");
  require(report.games.size() == 4U,
          "small-seed dual-seat dual-scenario matrix changed");
  require(report.exact_parity(), "disabled panel lost aggregate parity");
  for (const auto& game : report.games) {
    require(game.steps == 719, "whole game did not execute 719 hands");
    require(game.hands.size() == 719U, "per-hand metrics are incomplete");
    require(game.exact_parity(), "disabled game lost exact parity");
    require(game.final_action_receipt_checks == 719 &&
                game.final_action_receipts_accepted == 719 &&
                game.final_action_receipts_failed == 0,
            "per-hand final-action receipts are incomplete");
    require(game.baseline_moves == game.repair_moves &&
                game.move_action_mismatches == 0,
            "disabled MOVE parity failed");
    require(game.terminal_debts == 0 && game.callback_failures == 0,
            "disabled panel invented debt or callback failures");
    require(game.baseline_terminal.own == game.repair_terminal.own &&
                game.baseline_terminal.margin ==
                    game.repair_terminal.margin &&
                game.baseline_terminal.score == game.repair_terminal.score,
            "disabled terminal metrics diverged");
  }
  const auto json = report.json();
  require(json.find("\"benefit_claimed\":false") != std::string::npos,
          "JSON omitted no-benefit boundary");
  require(json.find("\"scenario\":\"normal\"") !=
              std::string::npos &&
              json.find("\"scenario\":\"forced_weed\"") !=
                  std::string::npos,
          "JSON omitted scenario coverage");
}

void callback_seam_pass_through() {
  int calls = 0;
  const panel::FinalActionCallback callback =
      [&](const panel::FinalActionContext& context) {
        ++calls;
        panel::FinalActionDecision out;
        out.final_action = context.provider_action;
        out.committed_provider_state = context.provider_state_proposal;
        out.evidence.receipt_checks = 1;
        out.evidence.receipts_accepted = 1;
        return out;
      };
  auto smoke = options();
  smoke.scenarios = {panel::Scenario::Normal};
  smoke.seats = {0};
  const auto report = panel::evaluate(smoke, callback);
  require(report.callback_enabled && report.exact_parity(),
          "pass-through callback changed the game");
  require(calls == 719, "final-action callback did not run once per hand");
  require(report.games.front().provider_state_commits == 719,
          "transactional provider-state seam was not exercised");
  require(report.games.front().adapter_receipt_checks == 719 &&
              report.games.front().adapter_receipts_accepted == 719 &&
              report.games.front().adapter_receipts_failed == 0,
          "adapter receipt telemetry did not accumulate per hand");
}

void invalid_callback_envelope_fails_closed() {
  const panel::FinalActionCallback invalid =
      [](const panel::FinalActionContext&) {
        return panel::FinalActionDecision{};
      };
  auto smoke = options();
  smoke.scenarios = {panel::Scenario::Normal};
  smoke.seats = {1};
  const auto report = panel::evaluate(smoke, invalid);
  const auto& game = report.games.front();
  require(game.action_mismatches == 0 && game.state_mismatches == 0 &&
              game.reward_mismatches == 0,
          "invalid callback envelope did not fail closed to provider");
  require(game.final_action_receipts_failed == 719 &&
              game.callback_failures == 719,
          "invalid callback envelope failures were not explicit");
}

}  // namespace

int main() {
  try {
    disabled_default_off_exact_parity();
    callback_seam_pass_through();
    invalid_callback_envelope_fails_closed();
    std::cout << "repair whole-game panel tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "repair whole-game panel tests failed: " << error.what()
              << '\n';
    return 1;
  }
}
