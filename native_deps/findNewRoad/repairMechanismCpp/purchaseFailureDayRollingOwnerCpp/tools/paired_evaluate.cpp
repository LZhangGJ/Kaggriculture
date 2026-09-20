#include "purchase_failure_day_rolling_owner.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace rolling = g001::purchase_failure_rolling;

namespace {

struct Result {
  std::uint64_t seed{};
  int seat{};
  double own_delta{};
  double opponent_delta{};
  double margin_delta{};
  double baseline_margin{};
  double repair_margin{};
  int purchase_failures{};
  int completed{};
  int expired{};
  int direct_move_edits{};
  int counterfactual_movement_sequence_mismatches{};
  int counterfactual_movement_lane_population_mismatches{};
  int causal_blocks{};
  int unit_retries{};
  int purchase_retries{};
};

bool move(fastkag::Op op) {
  return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
         op == fastkag::Op::EAST || op == fastkag::Op::WEST;
}

fastkag::NativeTapeLibrary tapes() {
  fastkag::NativeTapeLibrary result;
  result.routes = {
      g001::repair::load_route(PURCHASE_PAIRED_G001_TAPES,
                               PURCHASE_PAIRED_G001_LIBRARY, "G001"),
      g001::repair::load_route(PURCHASE_PAIRED_G001_TAPES,
                               PURCHASE_PAIRED_G001_LIBRARY, "G096")};
  return result;
}

Result run(const fastkag::NativeTeammateExecutor& executor,
           std::uint64_t seed, int seat, int feature_mask, int retry_mode,
           bool demand_guard, int maximum_retry_quantity,
           int forced_animal_step, int forced_animal_item, int debt_days) {
  fastkag::Simulator baseline({}, seed);
  fastkag::Simulator repair({}, seed);
  std::array<fastkag::NativeAgentState, 2> baseline_states;
  fastkag::NativeAgentState repair_opponent;
  rolling::Config config;
  config.enabled = true;
  config.retry_purchases = (feature_mask & 1) != 0;
  config.retry_partial_purchases = (retry_mode & 1) != 0;
  config.retry_seed_purchases = (retry_mode & 2) != 0;
  config.retry_animal_purchases = (retry_mode & 4) != 0;
  config.require_route_seed_demand = demand_guard;
  config.maximum_retry_quantity = maximum_retry_quantity;
  config.debt_days = debt_days;
  config.block_unready_consumers = (feature_mask & 2) != 0;
  config.retry_unit_debts = (feature_mask & 4) != 0;
  config.monitor_natural_consumers = (feature_mask & 8) != 0;
  rolling::Owner owner(executor, seat, 0, config);
  constexpr int kLaneStride = 128;
  const int movement_lanes =
      (baseline.config().episode_steps / baseline.config().turns_per_day + 1) *
      kLaneStride;
  std::vector<std::vector<int>> baseline_movement(
      static_cast<std::size_t>(movement_lanes));
  std::vector<std::vector<int>> repair_movement(
      static_cast<std::size_t>(movement_lanes));
  std::vector<bool> baseline_lane_seen(static_cast<std::size_t>(movement_lanes));
  std::vector<bool> repair_lane_seen(static_cast<std::size_t>(movement_lanes));
  const auto record_movement = [&](const fastkag::Simulator& simulator,
                                   const fastkag::PlayerAction& action,
                                   auto& lanes, auto& seen) {
    for (std::size_t actor = 0; actor < action.units.size(); ++actor) {
      const int lane = simulator.day() * kLaneStride +
                       static_cast<int>(actor);
      if (lane < 0 || lane >= movement_lanes) continue;
      seen[static_cast<std::size_t>(lane)] = true;
      if (move(action.units[actor].op))
        lanes[static_cast<std::size_t>(lane)].push_back(
            static_cast<int>(action.units[actor].op));
    }
  };
  const auto step_with_forced_animal_failure =
      [&](fastkag::Simulator& simulator,
          const std::array<fastkag::PlayerAction, 2>& actions) {
        const bool force = simulator.step_count() == forced_animal_step &&
            std::any_of(actions[seat].market.begin(),
                        actions[seat].market.end(), [&](const auto& order) {
                          return order.op == fastkag::Op::BUY_ANIMAL &&
                                 static_cast<int>(order.item) ==
                                     forced_animal_item;
                        });
        if (!force) {
          simulator.step(actions);
          return;
        }
        auto& private_state =
            const_cast<fastkag::PrivateState&>(simulator.privates()[seat]);
        const auto saved_shed = private_state.shed;
        int used = 0;
        for (const int quantity : private_state.shed) used += quantity;
        private_state.shed[0] += simulator.config().shed_capacity - used;
        simulator.step(actions);
        private_state.shed = saved_shed;
      };
  while (!baseline.done()) {
    std::array<fastkag::PlayerAction, 2> baseline_actions;
    baseline_actions[seat] = executor.action_external(
        baseline, seat, 0, baseline_states[seat]);
    baseline_actions[1 - seat] = executor.action_external(
        baseline, 1 - seat, 1, baseline_states[1 - seat]);
    record_movement(baseline, baseline_actions[seat], baseline_movement,
                    baseline_lane_seen);
    step_with_forced_animal_failure(baseline, baseline_actions);

    if (!owner.observe(repair))
      throw std::runtime_error("purchase owner rejected observation");
    const auto proposal = owner.propose(repair);
    if (owner.finalize(proposal, proposal.final_action) !=
        rolling::FinalizeStatus::Selected)
      throw std::runtime_error("purchase owner rejected final action");
    std::array<fastkag::PlayerAction, 2> repair_actions;
    repair_actions[seat] = proposal.final_action;
    repair_actions[1 - seat] = executor.action_external(
        repair, 1 - seat, 1, repair_opponent);
    record_movement(repair, repair_actions[seat], repair_movement,
                    repair_lane_seen);
    step_with_forced_animal_failure(repair, repair_actions);
  }
  if (!owner.observe(repair))
    throw std::runtime_error("purchase owner rejected terminal receipt");
  Result result;
  result.seed = seed;
  result.seat = seat;
  result.own_delta = repair.farms()[seat].money - baseline.farms()[seat].money;
  result.opponent_delta = repair.farms()[1 - seat].money -
                          baseline.farms()[1 - seat].money;
  result.baseline_margin = baseline.farms()[seat].money -
                           baseline.farms()[1 - seat].money;
  result.repair_margin = repair.farms()[seat].money -
                         repair.farms()[1 - seat].money;
  result.margin_delta = result.repair_margin - result.baseline_margin;
  const auto& metrics = owner.metrics();
  result.purchase_failures = metrics.purchase_failures;
  result.direct_move_edits = metrics.move_mismatch;
  for (int lane = 0; lane < movement_lanes; ++lane) {
    const auto index = static_cast<std::size_t>(lane);
    if (baseline_lane_seen[index] != repair_lane_seen[index])
      ++result.counterfactual_movement_lane_population_mismatches;
    else if (baseline_lane_seen[index] &&
             baseline_movement[index] != repair_movement[index])
      ++result.counterfactual_movement_sequence_mismatches;
  }
  result.causal_blocks = metrics.causal_blocks;
  result.unit_retries = metrics.unit_retries;
  result.purchase_retries = metrics.purchase_retries;
  for (const auto& debt : owner.debts()) {
    result.completed += debt.status == rolling::DebtStatus::Completed;
    result.expired += debt.status == rolling::DebtStatus::Expired;
  }
  return result;
}

int main_impl(int argc, char** argv) {
  const int seeds = argc > 1 ? std::stoi(argv[1]) : 256;
  const std::uint64_t seed_begin =
      argc > 2 ? std::stoull(argv[2]) : 990000ULL;
  const int requested_threads = argc > 3 ? std::stoi(argv[3]) :
      static_cast<int>(std::thread::hardware_concurrency());
  const int feature_mask = argc > 4 ? std::stoi(argv[4]) : 15;
  const int retry_mode = argc > 5 ? std::stoi(argv[5]) : 7;
  const bool demand_guard = argc > 6 ? std::stoi(argv[6]) != 0 : false;
  const int maximum_retry_quantity = argc > 7 ? std::stoi(argv[7]) : 1000000;
  const int forced_animal_step = argc > 8 ? std::stoi(argv[8]) : -1;
  const int forced_animal_item = argc > 9 ? std::stoi(argv[9]) : 10;
  const int debt_days = argc > 10 ? std::stoi(argv[10]) : 2;
  if (seeds <= 0 || requested_threads <= 0)
    throw std::invalid_argument("seeds and threads must be positive");
  const int games = seeds * 2;
  const fastkag::NativeTeammateExecutor executor(tapes());
  std::vector<Result> results(static_cast<std::size_t>(games));
  std::atomic<int> next{};
  std::atomic<bool> failed{};
  const int threads = std::min(games, requested_threads);
  std::vector<std::jthread> workers;
  workers.reserve(static_cast<std::size_t>(threads));
  for (int worker = 0; worker < threads; ++worker)
    workers.emplace_back([&] {
      try {
        for (;;) {
          const int index = next.fetch_add(1);
          if (index >= games) break;
          results[static_cast<std::size_t>(index)] =
              run(executor,
                  seed_begin + static_cast<std::uint64_t>(index / 2),
                  index % 2, feature_mask, retry_mode, demand_guard,
                  maximum_retry_quantity, forced_animal_step,
                  forced_animal_item, debt_days);
        }
      } catch (...) {
        failed = true;
      }
    });
  workers.clear();
  if (failed) throw std::runtime_error("paired worker failed");

  double own{};
  double opponent{};
  double margin{};
  double triggered_own{};
  double triggered_margin{};
  double min_own{};
  double min_margin{};
  int triggered{};
  int improved{};
  int worsened{};
  int unchanged{};
  int loss_to_win{};
  int win_to_loss{};
  int completed{};
  int expired{};
  int direct_move_edits{};
  int counterfactual_movement_sequence_mismatches{};
  int counterfactual_movement_lane_population_mismatches{};
  int causal_blocks{};
  int unit_retries{};
  int purchase_retries{};
  bool first = true;
  for (const auto& result : results) {
    own += result.own_delta;
    opponent += result.opponent_delta;
    margin += result.margin_delta;
    completed += result.completed;
    expired += result.expired;
    direct_move_edits += result.direct_move_edits;
    counterfactual_movement_sequence_mismatches +=
        result.counterfactual_movement_sequence_mismatches;
    counterfactual_movement_lane_population_mismatches +=
        result.counterfactual_movement_lane_population_mismatches;
    causal_blocks += result.causal_blocks;
    unit_retries += result.unit_retries;
    purchase_retries += result.purchase_retries;
    if (result.purchase_failures > 0) {
      ++triggered;
      triggered_own += result.own_delta;
      triggered_margin += result.margin_delta;
    }
    improved += result.own_delta > 0;
    worsened += result.own_delta < 0;
    unchanged += result.own_delta == 0;
    loss_to_win += result.baseline_margin <= 0 && result.repair_margin > 0;
    win_to_loss += result.baseline_margin > 0 && result.repair_margin <= 0;
    if (first || result.own_delta < min_own) min_own = result.own_delta;
    if (first || result.margin_delta < min_margin)
      min_margin = result.margin_delta;
    first = false;
  }
  std::cout << "games=" << games << " seeds=" << seeds
            << " threads=" << threads << " triggered=" << triggered
            << " feature_mask=" << feature_mask
            << " retry_mode=" << retry_mode
            << " demand_guard=" << demand_guard
            << " maximum_retry_quantity=" << maximum_retry_quantity
            << " forced_animal_step=" << forced_animal_step
            << " forced_animal_item=" << forced_animal_item
            << " debt_days=" << debt_days
            << " improved=" << improved << " worsened=" << worsened
            << " unchanged=" << unchanged
            << " own_mean=" << own / games
            << " opponent_mean=" << opponent / games
            << " margin_mean=" << margin / games
            << " triggered_own_mean="
            << (triggered ? triggered_own / triggered : 0.0)
            << " triggered_margin_mean="
            << (triggered ? triggered_margin / triggered : 0.0)
            << " min_own=" << min_own << " min_margin=" << min_margin
            << " loss_to_win=" << loss_to_win
            << " win_to_loss=" << win_to_loss
            << " completed_debts=" << completed
            << " expired_debts=" << expired
            << " causal_blocks=" << causal_blocks
            << " unit_retries=" << unit_retries
            << " purchase_retries=" << purchase_retries
            << " direct_current_move_edits=" << direct_move_edits
            << " counterfactual_movement_sequence_mismatches="
            << counterfactual_movement_sequence_mismatches
            << " counterfactual_movement_lane_population_mismatches="
            << counterfactual_movement_lane_population_mismatches << '\n';
  std::sort(results.begin(), results.end(),
            [](const Result& lhs, const Result& rhs) {
              return lhs.own_delta < rhs.own_delta;
            });
  for (std::size_t index = 0; index < std::min<std::size_t>(5, results.size());
       ++index) {
    const auto& result = results[index];
    std::cout << "worst rank=" << index + 1 << " seed=" << result.seed
              << " seat=" << result.seat
              << " own_delta=" << result.own_delta
              << " margin_delta=" << result.margin_delta
              << " failures=" << result.purchase_failures
              << " causal_blocks=" << result.causal_blocks
              << " unit_retries=" << result.unit_retries
              << " purchase_retries=" << result.purchase_retries
              << " completed=" << result.completed
              << " expired=" << result.expired
              << " direct_current_move_edits=" << result.direct_move_edits
              << " counterfactual_movement_sequence_mismatches="
              << result.counterfactual_movement_sequence_mismatches
              << " counterfactual_movement_lane_population_mismatches="
              << result.counterfactual_movement_lane_population_mismatches
              << '\n';
  }
  if (feature_mask == 0 &&
      (direct_move_edits != 0 ||
       counterfactual_movement_sequence_mismatches != 0 ||
       counterfactual_movement_lane_population_mismatches != 0))
    throw std::runtime_error("default no-op changed a movement lane");
  return direct_move_edits == 0 ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return main_impl(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "purchase_failure_paired_evaluate: " << error.what() << '\n';
    return 1;
  }
}
