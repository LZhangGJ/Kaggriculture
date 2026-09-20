#include "minimum_damage_remaining_day_audit_adapter.hpp"

#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace audit = g001::minimum_damage_remaining_audit;
namespace bridge = g001::minimum_damage_bridge;
namespace issuer = g001::day_start_issuer;

namespace {

constexpr std::uint64_t kSeed = 970017;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::uint64_t generation(int day) {
  return (kSeed << 20) | (2ULL << 16) |
         static_cast<std::uint64_t>(day + 1);
}

void real_day7_default_off_rolling_audit() {
  fastkag::NativeTapeLibrary library;
  library.routes.push_back(g001::repair::load_route(
      ADAPTER_G001_TAPES, ADAPTER_G001_LIBRARY, "G001"));
  fastkag::NativeTeammateExecutor executor(std::move(library));
  fastkag::Simulator env({}, kSeed);
  std::array<fastkag::NativeAgentState, 2> states;
  while (env.step_count() < 168) {
    std::array<fastkag::PlayerAction, 2> actions;
    for (int player = 0; player < 2; ++player)
      actions[player] = executor.action_external(env, player, 0, states[player]);
    env.step(actions);
  }
  require(env.day() == 7 && env.hour() == 0,
          "real seed did not reach frozen day7 observation");

  issuer::PersistentRouteIntentRegistry lineage;
  const auto issued = issuer::issue_day_start(
      {&env, &executor.route_tape(0), 1, generation(7), &lineage, {}, {}});
  require(issued.issued(), "real day7 issuer rejected");
  std::vector<bridge::ObligationPolicy> policies;
  for (const auto& obligation : issued.obligations)
    if (obligation.actor == 0)
      policies.push_back(
          {obligation.id, true, 1 + (191 - obligation.source_step)});

  audit::Adapter adapter(issued, 1, generation(7), policies);
  auto audit_env = env;
  bool saw_signed_debt = false;
  for (int hour = 0; hour < 24; ++hour) {
    const int before_step = audit_env.step_count();
    const auto hand = adapter.plan(audit_env);
    require(hand.valid,
            "rolling hand " + std::to_string(before_step) + " rejected: " +
                hand.diagnostic);
    saw_signed_debt = saw_signed_debt || hand.signed_debts != 0;
    require(audit_env.step_count() == before_step,
            "default-off plan mutated the live simulator");

    // The real provider continues independently on the live environment.  It
    // never consumes the audit candidate.
    std::array<fastkag::PlayerAction, 2> actions;
    for (int player = 0; player < 2; ++player)
      actions[player] = executor.action_external(env, player, 0, states[player]);
    env.step(actions);

    std::array<fastkag::PlayerAction, 2> shadow_actions;
    shadow_actions[1].units.push_back(hand.candidate_unit);
    if (hand.candidate_market)
      shadow_actions[1].market.push_back(*hand.candidate_market);
    const auto before = audit_env;
    audit_env.step(shadow_actions);
    require(adapter.observe_final(before, hand.candidate_unit, audit_env),
            "signed shadow candidate could not be receipted at " +
                std::to_string(before_step));
  }

  const auto report = adapter.finish();
  require(report.default_off && report.hands == 24,
          "audit was not a 24-hand default-off run");
  require(report.initial_move_tokens == 7 &&
              report.observed_move_receipts == 7 &&
              report.duplicate_move_receipts == 0 &&
              report.failed_move_receipts == 0 &&
              report.move_early_violations == 0,
          "rolling MOVE closure/ordering failed");
  require(report.outstanding_omissions == 0,
          "an effect-failed typed obligation disappeared from suffix plans");
  require(report.outstanding_obligations > 0,
          "real weed failure fixture unexpectedly had no typed carry-in debt");
  require(saw_signed_debt,
          "rolling hand hid the signed certificate debt from its caller");
  require(report.terminal_debts.size() ==
              static_cast<std::size_t>(report.outstanding_obligations) &&
              std::all_of(report.terminal_debts.begin(),
                          report.terminal_debts.end(), [](const auto& debt) {
                            return debt.obligation_id != 0 &&
                                   debt.effect_evidence_hash != 0 &&
                                   debt.bound_day_offset == 1;
                          }),
          "outstanding effect failure lacks explicit day-bound terminal debt");
  std::cout << "remaining_day_default_off_hands=" << report.hands
            << " move_receipts=" << report.observed_move_receipts << '/'
            << report.initial_move_tokens
            << " completed_obligations=" << report.completed_obligations
            << " outstanding_obligations=" << report.outstanding_obligations
            << " terminal_debts=" << report.terminal_debts.size()
            << " selector_states_total=" << report.total_selector_states
            << " hand_us_min=" << report.min_hand_microseconds
            << " hand_us_median=" << report.median_hand_microseconds
            << " hand_us_max=" << report.max_hand_microseconds
            << " day_issue_us=" << report.total_issue_microseconds << '\n';
}

void hired_hand_receipt_uses_selected_actor() {
  fastkag::Simulator env({}, 991337);
  std::array<fastkag::PlayerAction, 2> hire;
  hire[1].market.push_back({fastkag::Op::HIRE, fastkag::Item::NONE, 1});
  env.step(hire);
  require(env.farms()[1].hands.size() == 1,
          "hired-hand fixture failed to hire actor 1");

  issuer::IssueResult issued;
  const auto hand_position = env.farms()[1].hands[0];
  const fastkag::Action move{
      hand_position.x + 1 < env.config().board_size ? fastkag::Op::EAST
                                                    : fastkag::Op::WEST,
      fastkag::Item::NONE, 1};
  for (int step = env.step_count(); step < 24; ++step)
    issued.raw_sources.push_back(
        {1, step, step == env.step_count() ? move : fastkag::Action{}});
  issued.moves.push_back({1, env.step_count(), move});

  audit::Adapter adapter(issued, 1, generation(0), {}, 1);
  bool move_seen = false;
  while (env.step_count() < 24) {
    const auto hand = adapter.plan(env);
    require(hand.valid, "actor-1 rolling suffix rejected at " +
                            std::to_string(env.step_count()) + ": " +
                            hand.diagnostic);
    move_seen = move_seen || hand.candidate_unit.op == move.op;
    auto after = env;
    std::array<fastkag::PlayerAction, 2> actions;
    actions[1].units.resize(2);
    actions[1].units[1] = hand.candidate_unit;
    after.step(actions);
    require(adapter.observe_final(env, hand.candidate_unit, after),
            "actor-1 concrete action receipt was rejected");
    env = std::move(after);
  }
  const auto report = adapter.finish();
  require(move_seen && report.initial_move_tokens == 1 &&
              report.observed_move_receipts == 1 &&
              report.failed_move_receipts == 0,
          "actor-1 MOVE receipt was attributed to actor 0");
}

}  // namespace

int main() {
  try {
    real_day7_default_off_rolling_audit();
    hired_hand_receipt_uses_selected_actor();
    std::cout << "minimum_damage_remaining_day_audit_adapter_tests: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "minimum_damage_remaining_day_audit_adapter_tests: "
              << error.what() << '\n';
    return 1;
  }
}
