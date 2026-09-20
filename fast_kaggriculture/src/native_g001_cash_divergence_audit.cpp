#include "native_teammate.hpp"
#include "route_loader.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifndef NATIVE_G001_TAPES
#error NATIVE_G001_TAPES is required
#endif
#ifndef NATIVE_G001_LIBRARY
#error NATIVE_G001_LIBRARY is required
#endif

namespace {

using fastkag::NativeAgentState;
using fastkag::NativeRepairAudit;
using fastkag::NativeTapeLibrary;
using fastkag::NativeTeammateExecutor;
using fastkag::PlayerAction;
using fastkag::Simulator;

void numbers(const std::vector<int32_t>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

void market(const PlayerAction& action) {
  std::cout << '[';
  for (std::size_t index = 0; index < action.market.size(); ++index) {
    if (index) std::cout << ',';
    const auto& order = action.market[index];
    std::cout << '{' << static_cast<int>(order.op) << ','
              << static_cast<int>(order.item) << ',' << order.quantity << '}';
  }
  std::cout << ']';
}

}  // namespace

int main() {
  const auto tape = g001::repair::load_route(
      NATIVE_G001_TAPES, NATIVE_G001_LIBRARY, "G001");
  NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  fastkag::Config config;
  config.episode_steps = 720;
  config.weed_spawn_chance = 1.0;
  constexpr std::uint64_t seed = 0x6001F0CEDULL;
  Simulator baseline(config, seed);
  Simulator enabled(config, seed);
  std::array<NativeAgentState, 2> baseline_states;
  std::array<NativeAgentState, 2> enabled_states;
  NativeRepairAudit audit;
  const auto repair = fastkag::native_repair_options_from_mask(8);
  bool cash_diverged = false;

  while (!baseline.done() && !enabled.done() && enabled.step_count() <= 266) {
    const int step = enabled.step_count();
    const double baseline_before = baseline.farms()[0].money;
    const double enabled_before = enabled.farms()[0].money;
    std::array<PlayerAction, 2> baseline_actions{
        executor.action_external(
            baseline, 0, 0, baseline_states[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true),
        executor.action_external(baseline, 1, 0, baseline_states[1])};
    const auto audit_before = audit;
    std::array<PlayerAction, 2> enabled_actions{
        executor.action_external(
            enabled, 0, 0, enabled_states[0],
            fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true,
            repair, &audit),
        executor.action_external(enabled, 1, 0, enabled_states[1])};
    baseline.step(baseline_actions);
    enabled.step(enabled_actions);
    const bool now_diverged =
        baseline.farms()[0].money != enabled.farms()[0].money;
    const bool hand_diverged = baseline.farms()[0].hands.size() !=
        enabled.farms()[0].hands.size();
    const bool recovery_order = audit.local_repair_seed_orders !=
        audit_before.local_repair_seed_orders;
    if ((!cash_diverged && now_diverged) || hand_diverged || recovery_order ||
        step >= 214) {
      std::cout << "step=" << step
                << " cash_before=" << baseline_before << '/' << enabled_before
                << " cash_after=" << baseline.farms()[0].money << '/'
                << enabled.farms()[0].money
                << " hands_after=" << baseline.farms()[0].hands.size() << '/'
                << enabled.farms()[0].hands.size()
                << " recovery_seed_orders="
                << audit.local_repair_seed_orders << " baseline_market=";
      market(baseline_actions[0]);
      std::cout << " enabled_market=";
      market(enabled_actions[0]);
      std::cout << " baseline_fills=";
      numbers(baseline.last_market_fills()[0]);
      std::cout << " enabled_fills=";
      numbers(enabled.last_market_fills()[0]);
      std::cout << '\n';
    }
    cash_diverged = cash_diverged || now_diverged;
  }
}
