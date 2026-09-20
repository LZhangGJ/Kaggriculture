#include "purchase_failure_day_rolling_owner.hpp"
#include "route_loader.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace rolling = g001::purchase_failure_rolling;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

bool same_action(fastkag::Action lhs, fastkag::Action rhs) {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool same_player_action(const fastkag::PlayerAction& lhs,
                        const fastkag::PlayerAction& rhs) {
  if (lhs.units.size() != rhs.units.size() ||
      lhs.market.size() != rhs.market.size())
    return false;
  for (std::size_t index = 0; index < lhs.units.size(); ++index)
    if (!same_action(lhs.units[index], rhs.units[index])) return false;
  for (std::size_t index = 0; index < lhs.market.size(); ++index)
    if (!same_action(lhs.market[index], rhs.market[index])) return false;
  return true;
}

fastkag::NativeTapeLibrary tapes() {
  fastkag::NativeTapeLibrary result;
  result.routes = {
      g001::repair::load_route(PURCHASE_ROLLING_G001_TAPES,
                               PURCHASE_ROLLING_G001_LIBRARY, "G001"),
      g001::repair::load_route(PURCHASE_ROLLING_G001_TAPES,
                               PURCHASE_ROLLING_G001_LIBRARY, "G096")};
  return result;
}

void default_off_is_exact_and_rejects_second_writer() {
  const rolling::Config defaults;
  require(defaults.retry_purchases && !defaults.retry_partial_purchases &&
              defaults.retry_seed_purchases &&
              !defaults.retry_animal_purchases &&
              defaults.require_route_seed_demand &&
              defaults.maximum_retry_quantity == 1 &&
              !defaults.block_unready_consumers &&
              !defaults.retry_unit_debts &&
              !defaults.monitor_natural_consumers,
          "minimal purchase-repair defaults changed");
  fastkag::NativeTeammateExecutor executor(tapes());
  rolling::Owner owner(executor, 1, 0, {});
  fastkag::Simulator owned({}, 990034);
  fastkag::Simulator mirror({}, 990034);
  fastkag::NativeAgentState mirror_focal;
  fastkag::NativeAgentState owned_opponent;
  fastkag::NativeAgentState mirror_opponent;
  bool tamper_checked = false;
  while (!owned.done()) {
    require(owner.observe(owned), "default-off receipt observation failed");
    const auto proposal = owner.propose(owned);
    const auto mirror_focal_action =
        executor.action_external(mirror, 1, 0, mirror_focal);
    require(same_player_action(proposal.base_action, mirror_focal_action),
            "default-off native proposal diverged");
    require(same_player_action(proposal.final_action, proposal.base_action),
            "default-off owner modified native action");
    if (!tamper_checked) {
      auto tampered = proposal.final_action;
      require(!tampered.units.empty(), "real proposal has no unit actor");
      tampered.units[0] = {fastkag::Op::PASS, fastkag::Item::WHEAT, 99};
      require(owner.finalize(proposal, tampered) ==
                  rolling::FinalizeStatus::SecondWriter,
              "full-action binding accepted a second writer");
      tamper_checked = true;
    }
    require(owner.finalize(proposal, proposal.final_action) ==
                rolling::FinalizeStatus::Selected,
            "default-off exact proposal was not finalized");
    if (owned.step_count() == 0)
      require(owner.finalize(proposal, proposal.final_action) ==
                  rolling::FinalizeStatus::StaleProposal,
              "already-selected proposal was accepted twice");
    const auto owned_opponent_action =
        executor.action_external(owned, 0, 1, owned_opponent);
    const auto mirror_opponent_action =
        executor.action_external(mirror, 0, 1, mirror_opponent);
    require(same_player_action(owned_opponent_action, mirror_opponent_action),
            "same-start opponent action diverged");
    std::array<fastkag::PlayerAction, 2> owned_actions;
    owned_actions[1] = proposal.final_action;
    owned_actions[0] = owned_opponent_action;
    std::array<fastkag::PlayerAction, 2> mirror_actions;
    mirror_actions[1] = mirror_focal_action;
    mirror_actions[0] = mirror_opponent_action;
    owned.step(owned_actions);
    mirror.step(mirror_actions);
    require(owned.step_count() == mirror.step_count() &&
                owned.farms()[1].money == mirror.farms()[1].money &&
                owned.privates()[1].seeds == mirror.privates()[1].seeds &&
                owned.privates()[1].shed == mirror.privates()[1].shed,
            "default-off same-start physical state diverged");
  }
  require(owner.observe(owned), "default-off terminal receipt failed");
  const auto& metrics = owner.metrics();
  require(metrics.purchase_failures == 0 && owner.debts().empty(),
          "default-off path created repair debt");
  require(metrics.move_source == metrics.move_final &&
              metrics.move_mismatch == 0 &&
              metrics.second_writer_rejections == 1 &&
              metrics.stale_finalize_rejections == 1,
          "default-off ownership/MOVE audit mismatch");
}

struct RealReport {
  std::uint64_t seed{};
  int seat{};
  rolling::Metrics metrics;
  int completed{};
  int outstanding{};
  int expired{};
  std::size_t diagnostics{};
  std::uint64_t wall_us{};
};

RealReport run_enabled(std::uint64_t seed, int seat) {
  fastkag::NativeTeammateExecutor executor(tapes());
  rolling::Config config;
  config.enabled = true;
  // Exercise the optional full debt-chain research path separately from the
  // safe minimal defaults.
  config.retry_partial_purchases = true;
  config.retry_animal_purchases = true;
  config.require_route_seed_demand = false;
  config.maximum_retry_quantity = 1000000;
  config.block_unready_consumers = true;
  config.retry_unit_debts = true;
  config.monitor_natural_consumers = true;
  config.maximum_attempts = 3;
  config.debt_days = 2;
  rolling::Owner owner(executor, seat, 0, config);
  fastkag::NativeAgentState opponent;
  fastkag::Simulator observation({}, seed);
  const auto begin = std::chrono::steady_clock::now();
  while (!observation.done()) {
    require(owner.observe(observation), "real rolling receipt failed");
    const auto proposal = owner.propose(observation);
    require(proposal.base_action.units.size() ==
                proposal.final_action.units.size(),
            "rolling owner changed actor shape");
    for (std::size_t actor = 0; actor < proposal.base_action.units.size(); ++actor) {
      const auto raw = proposal.base_action.units[actor];
      const bool move = raw.op == fastkag::Op::NORTH ||
                        raw.op == fastkag::Op::SOUTH ||
                        raw.op == fastkag::Op::EAST ||
                        raw.op == fastkag::Op::WEST;
      if (move)
        require(same_action(raw, proposal.final_action.units[actor]),
                "rolling owner edited a raw MOVE");
    }
    require(owner.finalize(proposal, proposal.final_action) ==
                rolling::FinalizeStatus::Selected,
            "real rolling finalization failed");
    std::array<fastkag::PlayerAction, 2> actions;
    actions[seat] = proposal.final_action;
    actions[1 - seat] =
        executor.action_external(observation, 1 - seat, 1, opponent);
    observation.step(actions);
  }
  require(owner.observe(observation), "real terminal receipt failed");
  RealReport report;
  report.seed = seed;
  report.seat = seat;
  report.metrics = owner.metrics();
  for (const auto& debt : owner.debts()) {
    if (debt.status == rolling::DebtStatus::Completed)
      ++report.completed;
    else if (debt.status == rolling::DebtStatus::Expired)
      ++report.expired;
    else
      ++report.outstanding;
  }
  report.diagnostics = owner.terminal_diagnostics(observation.step_count()).size();
  report.wall_us = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
  require(report.metrics.move_source == report.metrics.move_final &&
              report.metrics.move_mismatch == 0,
          "real MOVE source/final closure failed");
  return report;
}

void real_seed_and_animal_receipts_drive_repair() {
  const auto animal = run_enabled(990034, 1);
  const auto seed = run_enabled(990045, 1);
  require(animal.metrics.animal_purchase_failures >= 1,
          "real animal zero-fill trigger disappeared");
  require(seed.metrics.seed_purchase_failures >= 1,
          "real seed zero-fill trigger disappeared");
  require(animal.metrics.purchase_retries + seed.metrics.purchase_retries > 0,
          "real receipts never armed a purchase retry");
  require(animal.completed + seed.completed > 0,
          "no real typed acquisition/consumer debt completed");
  require(animal.metrics.causal_blocks + animal.metrics.unit_effect_failures +
              seed.metrics.causal_blocks + seed.metrics.unit_effect_failures >
              0,
          "real purchase/consumer causal edge was not observed");

  const auto print = [](const char* label, const RealReport& report) {
    const auto& m = report.metrics;
    const auto average_ns =
        m.proposals > 0 ? m.proposal_time_ns / m.proposals : 0;
    std::cout << label << " seed=" << report.seed << " seat=" << report.seat
              << " purchase_failures=" << m.purchase_failures
              << " seed_failures=" << m.seed_purchase_failures
              << " animal_failures=" << m.animal_purchase_failures
              << " retries=" << m.purchase_retries
              << " retry_fills=" << m.purchase_retry_fills
              << " causal_blocks=" << m.causal_blocks
              << " unit_effect_failures=" << m.unit_effect_failures
              << " unit_retries=" << m.unit_retries
              << " completed=" << report.completed
              << " outstanding=" << report.outstanding
              << " expired=" << report.expired
              << " omissions=" << m.move_mismatch
              << " unsupported_fail_closed=" << m.unsupported_displacements
              << " move=" << m.move_final << '/' << m.move_source
              << " terminal_diagnostics=" << report.diagnostics
              << " proposal_avg_ns=" << average_ns
              << " whole_game_us=" << report.wall_us << '\n';
  };
  print("real_animal", animal);
  print("real_seed", seed);
}

void seed_demand_scan_crosses_midnight_within_debt_window() {
  fastkag::Simulator observation({}, 990045);
  std::array<fastkag::PlayerAction, 2> pass;
  observation.step(pass);
  auto& private_state =
      const_cast<fastkag::PrivateState&>(observation.privates()[0]);
  private_state.seeds.fill(0);
  std::vector<fastkag::PlayerAction> tape(30);
  tape[25].units.push_back(
      {fastkag::Op::PLANT, fastkag::Item::STRAWBERRY, 1});
  rolling::Debt debt;
  debt.kind = rolling::DebtKind::AcquireSeed;
  debt.item = fastkag::Item::STRAWBERRY;
  debt.deadline_step = 48;
  require(rolling::detail::uncovered_seed_demand(observation, 0, tape,
                                                  debt) == 1,
          "same-debt-window next-day PLANT was hidden by midnight");
}

}  // namespace

int main() {
  default_off_is_exact_and_rejects_second_writer();
  seed_demand_scan_crosses_midnight_within_debt_window();
  real_seed_and_animal_receipts_drive_repair();
  std::cout << "purchase_failure_day_rolling_owner_tests: PASS\n";
  return 0;
}
