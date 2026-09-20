#include "failure_debt_scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>

int main(int argc, char** argv) {
  using namespace g001::failure_debt;
  const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 100000;
  OwnObservation before;
  before.step = 10;
  before.money = 1000;
  PurchaseIntent intent;
  intent.attempt_id = 1;
  intent.operation = fastkag::Op::BUY_SEED;
  intent.item = fastkag::Item::STRAWBERRY;
  intent.quantity = 1;
  intent.deadline = 40;
  intent.economic_value = 300;
  intent.provenance = "benchmark";
  RecoveryChain chain;
  UnitRequirement plant;
  plant.action = {fastkag::Op::PLANT, fastkag::Item::STRAWBERRY, 1};
  plant.position = {2, 2};
  plant.earliest_step = 12;
  plant.deadline = 40;
  plant.provenance = "plant";
  chain.requirements.push_back(plant);
  intent.chains.push_back(chain);
  OwnObservation after = before;
  after.step = 11;
  PlanWindow window;
  for (int step = 11; step <= 35; ++step) {
    PlannedTurn turn;
    turn.step = step;
    PlannedUnitSlot slot;
    slot.position = step == 30 ? fastkag::Position{2, 2}
                               : fastkag::Position{0, 0};
    slot.original = {step == 30 ? fastkag::Op::PASS : fastkag::Op::EAST,
                     fastkag::Item::NONE, 1};
    slot.absorbable = step == 30;
    slot.expected_success = true;
    slot.legal_replacements.fill(true);
    turn.units.push_back(slot);
    window.turns.push_back(turn);
  }
  SchedulerConfig config;
  config.enabled = true;
  config.require_day_lifecycle = false;
  std::uint64_t checksum = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    FailureDebtScheduler scheduler(config);
    scheduler.observe(before);
    scheduler.record_attempt(intent, before);
    scheduler.observe(after);
    const auto transaction = scheduler.open_transactions().front();
    const auto plan = scheduler.plan(transaction, after, window);
    checksum += plan.accepted + plan.units.size() + plan.market.step;
  }
  const auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  std::cout << "{\"schema\":\"failure-debt-benchmark-v1\",\"iterations\":"
            << iterations << ",\"seconds\":" << elapsed
            << ",\"mean_us\":" << elapsed * 1e6 / iterations
            << ",\"checksum\":" << checksum << "}\n";
}
