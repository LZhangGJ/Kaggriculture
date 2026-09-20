#include "failure_debt_scheduler.hpp"
#include "static_dependencies.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace g001::failure_debt;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::Position;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

SchedulerConfig step_window_experiment() {
  SchedulerConfig config;
  config.enabled = true;
  config.require_day_lifecycle = false;
  return config;
}

PlannedUnitSlot slot(int actor, Position position, Action original,
                     int value = 0, bool absorbable = true) {
  PlannedUnitSlot result;
  result.actor = actor;
  result.position = position;
  result.original = original;
  result.economic_value = value;
  result.expected_success = true;
  result.absorbable = absorbable;
  result.legal_replacements.fill(true);
  return result;
}

UnitRequirement requirement(Op operation, Item item, int actor, Position position,
                            int earliest, int deadline, std::string provenance,
                            int maximum_loss = 4, int quantity = 1) {
  UnitRequirement result;
  result.action = {operation, item, quantity};
  result.actor = actor;
  result.position = position;
  result.earliest_step = earliest;
  result.deadline = deadline;
  result.maximum_absorbed_value = maximum_loss;
  result.provenance = std::move(provenance);
  return result;
}

PurchaseIntent seed_intent(std::uint64_t id, Item crop, int quantity, int step,
                           int plant_step, Position position) {
  PurchaseIntent result;
  result.attempt_id = id;
  result.operation = Op::BUY_SEED;
  result.item = crop;
  result.quantity = quantity;
  result.deadline = plant_step + 20;
  result.economic_value = quantity * 200;
  result.provenance = "fixture:seed-order";
  for (int unit = 0; unit < quantity; ++unit) {
    RecoveryChain chain;
    chain.purchase_unit_index = unit;
    chain.requirements.push_back(requirement(
        Op::PLANT, crop, 0, position, step + 1, plant_step + 20,
        "missed-plant-" + std::to_string(unit)));
    result.chains.push_back(std::move(chain));
  }
  return result;
}

PurchaseIntent animal_intent(std::uint64_t id, int step, bool include_harvest) {
  PurchaseIntent result;
  result.attempt_id = id;
  result.operation = Op::BUY_ANIMAL;
  result.item = Item::COW;
  result.quantity = 1;
  result.deadline = step + 40;
  result.economic_value = 1200;
  result.provenance = "fixture:cow-order";
  RecoveryChain chain;
  chain.requirements = {
      requirement(Op::PICKUP, Item::COW, 0, {4, 4}, step + 1, step + 5,
                  "pickup"),
      requirement(Op::PLACE, Item::COW, 0, {2, 2}, step + 2, step + 8,
                  "place"),
      requirement(Op::FEED, Item::WHEAT, 0, {2, 2}, step + 3, step + 20,
                  "first-feed"),
      requirement(Op::CARE, Item::NONE, 0, {2, 2}, step + 4, step + 22,
                  "first-care"),
  };
  if (include_harvest) {
    chain.requirements.push_back(requirement(
        Op::HARVEST, Item::NONE, 0, {2, 2}, step + 10, step + 40,
        "first-harvest"));
  }
  result.chains.push_back(std::move(chain));
  return result;
}

std::uint64_t settle_zero(FailureDebtScheduler& scheduler,
                          const PurchaseIntent& intent,
                          OwnObservation before) {
  scheduler.observe(before);
  require(scheduler.record_attempt(intent, before), "fixture attempt is recorded");
  ++before.step;
  const auto settlements = scheduler.observe(before);
  require(settlements.size() == 1 &&
              settlements[0].classification == FillClass::Zero,
          "zero fill is observation-confirmed");
  const auto open = scheduler.open_transactions();
  require(open.size() == 1, "zero fill materializes one transaction");
  return open[0];
}

void test_causal_full_partial_zero() {
  {
    FailureDebtScheduler scheduler;
    OwnObservation before; before.step = 10; before.money = 1000;
    const auto intent = seed_intent(1, Item::TOMATO, 2, 10, 20, {1, 1});
    scheduler.observe(before); require(scheduler.record_attempt(intent, before), "full recorded");
    OwnObservation after = before; after.step = 11; after.seeds[2] = 2;
    const auto result = scheduler.observe(after);
    require(result[0].classification == FillClass::Full &&
                result[0].filled == 2 && scheduler.open_transactions().empty(),
            "full seed fill creates no debt");
  }
  {
    FailureDebtScheduler scheduler;
    OwnObservation before; before.step = 20; before.money = 1000;
    const auto intent = seed_intent(2, Item::CARROT, 3, 20, 30, {1, 1});
    scheduler.observe(before); require(scheduler.record_attempt(intent, before), "partial recorded");
    OwnObservation after = before; after.step = 21;
    // One bought seed was consumed by an intervening known PLANT.
    after.known_outflow[1] = 1;
    const auto result = scheduler.observe(after);
    require(result[0].classification == FillClass::Partial &&
                result[0].filled == 1 && result[0].missing == 2,
            "known consumption preserves exact partial settlement");
    require(scheduler.debts().size() == 3,
            "partial fill creates one market debt and only two missing PLANT debts");
  }
  {
    FailureDebtScheduler scheduler;
    OwnObservation before; before.step = 30; before.money = 1000;
    const auto intent = seed_intent(3, Item::WHEAT, 1, 30, 40, {1, 1});
    scheduler.observe(before); require(scheduler.record_attempt(intent, before), "zero recorded");
    OwnObservation after = before; after.step = 31;
    // An unrelated known seed inflow must not be attributed to this order.
    after.seeds[0] = 1; after.known_other_inflow[0] = 1;
    const auto result = scheduler.observe(after);
    require(result[0].classification == FillClass::Zero,
            "known unrelated inflow is removed from causal settlement");
  }
}

void test_incomplete_animal_intent_is_rejected_before_settlement() {
  FailureDebtScheduler scheduler;
  OwnObservation before; before.step = 35; before.money = 1000;
  auto intent = animal_intent(31, 35, false);
  scheduler.observe(before);
  require(!scheduler.record_attempt(intent, before),
          "PICKUP/PLACE/FEED/CARE without HARVEST is not a complete animal intent");
  require(!scheduler.audit().empty() &&
              scheduler.audit().back().reason == "invalid_purchase_intent",
          "incomplete lifecycle is auditable at the intent boundary");
}

void test_seed_repair_waits_for_distant_slack_and_preserves_move() {
  auto config = step_window_experiment();
  FailureDebtScheduler scheduler(config);
  OwnObservation observation; observation.step = 50; observation.money = 1000;
  const auto transaction = settle_zero(
      scheduler, seed_intent(4, Item::STRAWBERRY, 1, 50, 80, {3, 3}),
      observation);
  ++observation.step;

  PlanWindow window;
  window.turns.push_back({51, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
  for (int step = 52; step < 70; ++step) {
    const Op move = step % 2 == 0 ? Op::EAST : Op::WEST;
    window.turns.push_back({step, {}, {slot(0, {0, 0}, {move, Item::NONE, 1}, 0, false)}});
  }
  window.turns.push_back({70, {}, {slot(0, {3, 3}, {Op::PASS, Item::NONE, 1})}});
  const auto plan = scheduler.plan(transaction, observation, window);
  require(plan.accepted && plan.market.step == 51 && plan.units.size() == 1 &&
              plan.units[0].step == 70 && plan.units[0].replacement.op == Op::PLANT,
          "scheduler crosses a nineteen-step gap instead of using a fixed 8-turn replay");
  require(!is_movement(plan.units[0].original.op),
          "only a non-MOVE slot is absorbed");
  for (const auto& patch : plan.units) {
    require(!is_movement(patch.original.op) && !is_movement(patch.replacement.op),
            "before/after MOVE subsequence is byte-identical by construction");
  }
  std::vector<Op> before_moves, after_moves;
  for (const auto& turn : window.turns) {
    for (int source = 0; source < static_cast<int>(turn.units.size()); ++source) {
      const auto original = turn.units[static_cast<std::size_t>(source)].original.op;
      if (is_movement(original)) before_moves.push_back(original);
      const auto found = std::find_if(plan.units.begin(), plan.units.end(),
          [&](const UnitPatch& patch) {
            return patch.step == turn.step && patch.source_slot == source;
          });
      const auto compiled = found == plan.units.end() ? original
                                                      : found->replacement.op;
      if (is_movement(compiled)) after_moves.push_back(compiled);
    }
  }
  require(before_moves == after_moves,
          "compiled repair preserves the exact MOVE byte sequence");
}

void test_same_step_market_cannot_repair_unit_phase() {
  auto config = step_window_experiment();
  FailureDebtScheduler scheduler(config);
  OwnObservation observation; observation.step = 75; observation.money = 1000;
  const auto transaction = settle_zero(
      scheduler, seed_intent(44, Item::CARROT, 1, 75, 76, {2, 2}), observation);
  ++observation.step;
  PlanWindow window;
  window.turns.push_back({76, {}, {slot(0, {2, 2}, {Op::PASS, Item::NONE, 1})}});
  const auto plan = scheduler.plan(transaction, observation, window);
  require(!plan.accepted && plan.reason == "no_complete_unit_chain",
          "same-step BUY_SEED is never credited to the preceding PLANT phase");
}

PlanWindow animal_window(int first_step, bool include_harvest = true) {
  PlanWindow window;
  window.turns.push_back({first_step, {},
                          {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
  window.turns.push_back({first_step + 1, {},
                          {slot(0, {4, 4}, {Op::PASS, Item::NONE, 1})}});
  window.turns.push_back({first_step + 2, {},
                          {slot(0, {2, 2}, {Op::PASS, Item::NONE, 1})}});
  window.turns.push_back({first_step + 3, {},
                          {slot(0, {2, 2}, {Op::PASS, Item::NONE, 1})}});
  // Existing CARE can discharge that debt without suppressing another action.
  window.turns.push_back({first_step + 4, {},
                          {slot(0, {2, 2}, {Op::CARE, Item::NONE, 1}, 6, false)}});
  if (include_harvest) {
    window.turns.push_back({first_step + 10, {},
                            {slot(0, {2, 2}, {Op::PASS, Item::NONE, 1})}});
  }
  return window;
}

void test_complete_animal_transaction() {
  auto config = step_window_experiment();
  FailureDebtScheduler scheduler(config);
  OwnObservation observation;
  observation.step = 100; observation.money = 2000;
  observation.actor_inventory.resize(1);
  observation.actor_inventory[0][0] = 3;
  const auto transaction = settle_zero(
      scheduler, animal_intent(5, 100, true), observation);
  ++observation.step;
  auto window = animal_window(101, true);
  const auto plan = scheduler.plan(transaction, observation, window);
  require(plan.accepted && plan.market.addition.op == Op::BUY_ANIMAL &&
              plan.units.size() == 4,
          "animal retry requires BUY->PICKUP->PLACE->FEED->CARE->HARVEST certificate");
  require(std::none_of(plan.units.begin(), plan.units.end(),
                       [](const UnitPatch& patch) {
                         return patch.replacement.op == Op::CARE;
                       }),
          "an existing valid CARE is reused rather than duplicated");
}

void test_cross_day_animal_transaction_requires_item_specific_pickup() {
  auto config = step_window_experiment();
  FailureDebtScheduler scheduler(config);
  OwnObservation observation;
  observation.step = 22; observation.money = 2000;
  observation.actor_inventory.resize(1);
  observation.actor_inventory[0][0] = 1;
  auto intent = animal_intent(55, 22, true);
  auto& chain = intent.chains[0].requirements;
  chain[0] = requirement(Op::PICKUP, Item::COW, 0, {4, 4}, 24, 25, "pickup-cow");
  chain[1] = requirement(Op::PLACE, Item::COW, 0, {2, 2}, 25, 26, "place-cow");
  chain.insert(chain.begin() + 2,
               requirement(Op::PICKUP, Item::WHEAT, 0, {4, 4}, 26, 27,
                           "pickup-feed"));
  chain[3] = requirement(Op::FEED, Item::WHEAT, 0, {2, 2}, 27, 28, "feed");
  chain[4] = requirement(Op::CARE, Item::NONE, 0, {2, 2}, 28, 30, "care");
  chain[5] = requirement(Op::HARVEST, Item::NONE, 0, {2, 2}, 35, 40, "harvest");
  const auto transaction = settle_zero(scheduler, intent, observation);
  ++observation.step;
  PlanWindow window;
  window.turns.push_back({23, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
  window.turns.push_back({24, {}, {slot(0, {4, 4}, {Op::PASS, Item::NONE, 1})}});
  window.turns.push_back({25, {}, {slot(0, {2, 2}, {Op::PASS, Item::NONE, 1})}});
  window.turns.push_back({26, {}, {slot(0, {4, 4}, {Op::PASS, Item::NONE, 1})}});
  window.turns.push_back({27, {}, {slot(0, {2, 2}, {Op::PASS, Item::NONE, 1})}});
  window.turns.push_back({28, {}, {slot(0, {2, 2}, {Op::PASS, Item::NONE, 1})}});
  window.turns.push_back({35, {}, {slot(0, {2, 2}, {Op::PASS, Item::NONE, 1})}});
  const auto plan = scheduler.plan(transaction, observation, window);
  require(plan.accepted && plan.units.size() == 6,
          "cross-day recovery deposits carried wheat then explicitly PICKUPs it before FEED");
}

void test_animal_rejects_incomplete_or_resource_unsafe_chain() {
  auto config = step_window_experiment();
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation observation; observation.step = 200; observation.money = 2000;
    observation.actor_inventory.resize(1);
    observation.actor_inventory[0][0] = 2;
    const auto transaction = settle_zero(
        scheduler, animal_intent(6, 200, true), observation);
    ++observation.step;
    const auto plan = scheduler.plan(transaction, observation,
                                     animal_window(201, false));
    require(!plan.accepted && plan.reason == "no_complete_unit_chain",
            "missing future HARVEST rejects the entire animal retry");
  }
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation observation; observation.step = 300; observation.money = 2000;
    const auto transaction = settle_zero(
        scheduler, animal_intent(7, 300, true), observation);
    ++observation.step;
    const auto plan = scheduler.plan(transaction, observation,
                                     animal_window(301, true));
    require(!plan.accepted && plan.reason == "animal_chain_missing_feed_wheat",
            "PICKUP/PLACE alone cannot certify an unfunded FEED chain");
  }
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation observation; observation.step = 400; observation.money = 600;
    observation.protected_cash = 100; observation.actor_inventory.resize(1);
    observation.actor_inventory[0][0] = 2;
    const auto transaction = settle_zero(
        scheduler, animal_intent(8, 400, true), observation);
    ++observation.step;
    auto window = animal_window(401, true);
    window.future_hard_purchase_reserve = 300;
    const auto plan = scheduler.plan(transaction, observation, window);
    require(!plan.accepted &&
                plan.reason == "repair_would_starve_existing_or_future_purchase",
            "animal retry cannot spend protected future seed/animal cash");
  }
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation observation; observation.step = 500; observation.money = 2000;
    observation.shed_used = 100; observation.actor_inventory.resize(1);
    observation.actor_inventory[0][0] = 2;
    const auto transaction = settle_zero(
        scheduler, animal_intent(9, 500, true), observation);
    ++observation.step;
    const auto plan = scheduler.plan(transaction, observation,
                                     animal_window(501, true));
    require(!plan.accepted && plan.reason == "repair_would_overflow_shed",
            "full shed rejects an animal transaction without proven prior release");
  }
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation observation; observation.step = 530; observation.money = 2000;
    observation.shed_used = 100; observation.actor_inventory.resize(1);
    observation.actor_inventory[0][0] = 2;
    const auto transaction = settle_zero(
        scheduler, animal_intent(90, 530, true), observation);
    ++observation.step;
    auto window = animal_window(531, true);
    window.turns.front().guaranteed_shed_delta_before_market = -1;
    const auto plan = scheduler.plan(transaction, observation, window);
    require(plan.accepted,
            "a proven pre-market DROP/PICKUP capacity release can fund full-shed recovery");
  }
}

void test_market_slots_and_future_purchase_reservation() {
  auto config = step_window_experiment();
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation observation; observation.step = 600; observation.money = 2000;
    const auto transaction = settle_zero(
        scheduler, seed_intent(10, Item::CARROT, 1, 600, 620, {1, 1}), observation);
    ++observation.step;
    PlanWindow window;
    PlannedTurn buy;
    buy.step = 601;
    buy.market.push_back({{Op::BUY_SEED, Item::CARROT, 2}, 1, 20, true});
    buy.units.push_back(slot(0, {0, 0}, {Op::PASS, Item::NONE, 1}));
    window.turns.push_back(buy);
    window.turns.push_back({602, {}, {slot(0, {1, 1}, {Op::PASS, Item::NONE, 1})}});
    const auto plan = scheduler.plan(transaction, observation, window);
    require(plan.accepted && plan.market.uses_existing_surplus &&
                plan.market.existing_slot == 0 && plan.purchase_cost == 0,
            "only explicitly unreserved future purchase surplus may discharge debt");
  }
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation observation; observation.step = 700; observation.money = 5000;
    const auto transaction = settle_zero(
        scheduler, seed_intent(11, Item::MELON, 1, 700, 720, {1, 1}), observation);
    ++observation.step;
    PlanWindow window;
    PlannedTurn full;
    full.step = 701;
    for (int slot_index = 0; slot_index < 10; ++slot_index) {
      full.market.push_back({{Op::BUY_SEED, Item::WHEAT, 1}, 1, 10, true});
    }
    full.units.push_back(slot(0, {0, 0}, {Op::PASS, Item::NONE, 1}));
    window.turns.push_back(full);
    PlannedTurn full_with_plant;
    full_with_plant.step = 702;
    for (int slot_index = 0; slot_index < 10; ++slot_index) {
      full_with_plant.market.push_back(
          {{Op::BUY_SEED, Item::WHEAT, 1}, 1, 10, true});
    }
    full_with_plant.units.push_back(
        slot(0, {1, 1}, {Op::PASS, Item::NONE, 1}));
    window.turns.push_back(std::move(full_with_plant));
    const auto plan = scheduler.plan(transaction, observation, window);
    require(!plan.accepted && plan.reason == "no_market_purchase_candidate",
            "ten occupied market slots cannot be overwritten by a repair order");
  }
}

void test_ambiguous_settlement_fails_closed() {
  FailureDebtScheduler scheduler;
  OwnObservation before; before.step = 950; before.money = 1000;
  const auto intent = seed_intent(14, Item::WHEAT, 1, 950, 960, {1, 1});
  scheduler.observe(before); require(scheduler.record_attempt(intent, before), "ambiguous recorded");
  auto after = before; after.step = 951; after.seeds[0] = 2;
  const auto result = scheduler.observe(after);
  require(result[0].classification == FillClass::Ambiguous &&
              scheduler.open_transactions().empty(),
          "inconsistent attribution unlocks no retry debt and cannot duplicate stock");
}

const Debt* find_debt(const FailureDebtScheduler& scheduler, DebtKind kind,
                      int unit_index = -2) {
  const auto found = std::find_if(scheduler.debts().begin(), scheduler.debts().end(),
      [&](const Debt& debt) {
        return debt.kind == kind &&
               (unit_index == -2 || debt.purchase_unit_index == unit_index);
      });
  return found == scheduler.debts().end() ? nullptr : &*found;
}

void test_explicit_plan_commit_settle_unit_lifecycle() {
  auto config = step_window_experiment();
  config.minimum_net_value = 1;
  FailureDebtScheduler scheduler(config);
  OwnObservation before; before.step = 10; before.money = 1000;
  const auto transaction = settle_zero(
      scheduler, seed_intent(70, Item::CARROT, 1, 10, 20, {1, 1}), before);
  OwnObservation at_plan = before; at_plan.step = 11;
  PlanWindow purchase_window;
  purchase_window.turns.push_back(
      {11, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
  purchase_window.turns.push_back(
      {12, {}, {slot(0, {1, 1}, {Op::PASS, Item::NONE, 1})}});

  const auto audit_before = scheduler.audit().size();
  const auto preview = scheduler.plan(transaction, at_plan, purchase_window);
  const auto preview_again = scheduler.plan(transaction, at_plan, purchase_window);
  require(preview.accepted && preview_again.accepted && preview.market.required &&
              scheduler.audit().size() == audit_before &&
              std::all_of(scheduler.debts().begin(), scheduler.debts().end(),
                          [](const Debt& debt) {
                            return debt.status == DebtStatus::Open;
                          }),
          "plan preview is deterministic and side-effect-free");
  OwnObservation too_early = at_plan; too_early.step = 10;
  require(!scheduler.commit(preview, too_early),
          "a future/current certificate cannot be committed before its market step");
  OwnObservation stale = at_plan; --stale.money;
  require(!scheduler.commit(preview, stale),
          "cash/capacity/slot/state drift invalidates a stale preview fingerprint");
  require(scheduler.commit(preview, at_plan),
          "only the actually emitted current market patch is committed");
  require(find_debt(scheduler, DebtKind::MarketPurchase)->status ==
              DebtStatus::Attempted &&
              find_debt(scheduler, DebtKind::UnitAction)->status == DebtStatus::Open,
          "purchase awaits observation while future unit debt remains unscheduled");

  OwnObservation filled = at_plan; filled.step = 12; filled.seeds[1] = 1;
  const auto settlement = scheduler.observe(filled);
  require(settlement.size() == 1 && settlement[0].classification == FillClass::Full &&
              find_debt(scheduler, DebtKind::MarketPurchase)->status ==
                  DebtStatus::Confirmed &&
              find_debt(scheduler, DebtKind::UnitAction)->status == DebtStatus::Open,
          "next observation confirms only acquisition, not a future PLANT");

  PlanWindow unit_window;
  unit_window.turns.push_back(
      {12, {}, {slot(0, {1, 1}, {Op::PASS, Item::NONE, 1})}});
  const auto unit_plan = scheduler.plan(transaction, filled, unit_window);
  require(unit_plan.accepted && !unit_plan.market.required &&
              unit_plan.unit_schedule.size() == 1,
          "confirmed seed produces a unit-only receding-horizon plan");
  require(scheduler.commit(unit_plan, filled) &&
              find_debt(scheduler, DebtKind::UnitAction)->status ==
                  DebtStatus::Scheduled,
          "only the due PLANT enters Scheduled");
  const auto plant_id = find_debt(scheduler, DebtKind::UnitAction)->id;
  UnitExecutionEvidence planted;
  planted.step = 13; planted.emitted = {Op::PLANT, Item::CARROT, 1};
  planted.observed_tile_kind = fastkag::TileKind::PLANT;
  planted.observed_tile_item = Item::CARROT;
  planted.provenance = "tile_observed_planted";
  require(scheduler.observe_unit_result(plant_id, planted) &&
              scheduler.open_transactions().empty(),
          "verified PLANT consumes the final debt and completes the transaction");
}

void test_future_market_preview_cannot_commit_or_mutate() {
  auto config = step_window_experiment();
  FailureDebtScheduler scheduler(config);
  OwnObservation before; before.step = 15; before.money = 2000;
  const auto transaction = settle_zero(
      scheduler, seed_intent(700, Item::CARROT, 1, 15, 30, {1,1}), before);
  OwnObservation current = before; current.step = 16;
  PlanWindow window;
  PlannedTurn full; full.step = 16;
  for (int index = 0; index < 10; ++index)
    full.market.push_back({{Op::BUY_SEED, Item::WHEAT, 1}, 1, 10, true});
  full.units.push_back(slot(0, {0,0}, {Op::PASS, Item::NONE, 1}));
  window.turns.push_back(full);
  window.turns.push_back({17, {}, {slot(0, {0,0}, {Op::PASS, Item::NONE, 1})}});
  window.turns.push_back({18, {}, {slot(0, {1,1}, {Op::PASS, Item::NONE, 1})}});
  const auto preview = scheduler.plan(transaction, current, window);
  require(preview.accepted && preview.market.step == 17 &&
              !scheduler.commit(preview, current) &&
              std::all_of(scheduler.debts().begin(), scheduler.debts().end(),
                          [](const Debt& debt) {
                            return debt.status == DebtStatus::Open;
                          }),
          "future market certificate remains preview-only until replanned at its due step");
}

void test_retry_zero_partial_and_failed_unit_reopen_without_duplicate_buy() {
  auto config = step_window_experiment();
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation before; before.step = 40; before.money = 1000;
    const auto transaction = settle_zero(
        scheduler, seed_intent(71, Item::TOMATO, 1, 40, 60, {1, 1}), before);
    OwnObservation current = before; current.step = 41;
    PlanWindow window;
    window.turns.push_back({41, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
    window.turns.push_back({42, {}, {slot(0, {1, 1}, {Op::PASS, Item::NONE, 1})}});
    const auto first = scheduler.plan(transaction, current, window);
    require(scheduler.commit(first, current), "zero retry committed");
    OwnObservation still_zero = current; still_zero.step = 42;
    const auto settled = scheduler.observe(still_zero);
    require(settled[0].classification == FillClass::Zero &&
                find_debt(scheduler, DebtKind::MarketPurchase)->status ==
                    DebtStatus::Open,
            "zero retry reopens only acquisition debt");
    PlanWindow retry_window;
    retry_window.turns.push_back({42, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
    retry_window.turns.push_back({43, {}, {slot(0, {1, 1}, {Op::PASS, Item::NONE, 1})}});
    const auto retry = scheduler.plan(transaction, still_zero, retry_window);
    require(retry.accepted && retry.market.required &&
                retry.market.addition.quantity == 1,
            "zero settlement requests exactly the outstanding unit, never duplicates it");
  }
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation before; before.step = 70; before.money = 2000;
    auto intent = seed_intent(72, Item::MELON, 2, 70, 90, {1, 1});
    intent.chains[0].requirements[0].actor = 0;
    intent.chains[0].requirements[0].position = {1, 1};
    intent.chains[1].requirements[0].actor = 1;
    intent.chains[1].requirements[0].position = {2, 2};
    const auto transaction = settle_zero(scheduler, intent, before);
    OwnObservation current = before; current.step = 71;
    PlanWindow window;
    window.turns.push_back({71, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
    window.turns.push_back({72, {}, {
        slot(0, {1, 1}, {Op::PASS, Item::NONE, 1}),
        slot(1, {2, 2}, {Op::PASS, Item::NONE, 1})}});
    const auto first = scheduler.plan(transaction, current, window);
    require(first.accepted && first.market.addition.quantity == 2 &&
                scheduler.commit(first, current), "two-unit retry committed");
    OwnObservation partial = current; partial.step = 72; partial.seeds[4] = 1;
    const auto settled = scheduler.observe(partial);
    require(settled[0].classification == FillClass::Partial &&
                find_debt(scheduler, DebtKind::MarketPurchase)->action.quantity == 1 &&
                std::all_of(scheduler.debts().begin(), scheduler.debts().end(),
                            [](const Debt& debt) {
                              return debt.kind != DebtKind::UnitAction ||
                                     debt.status == DebtStatus::Open;
                            }),
            "partial retry confirms inventory but schedules no future unit prematurely");
    PlanWindow remaining;
    remaining.turns.push_back({72, {}, {slot(0, {1, 1}, {Op::PASS, Item::NONE, 1})}});
    remaining.turns.push_back({73, {}, {slot(1, {2, 2}, {Op::PASS, Item::NONE, 1})}});
    const auto second = scheduler.plan(transaction, partial, remaining);
    const bool second_commit = scheduler.commit(second, partial);
    require(second.accepted && second.market.required &&
                second.market.addition.quantity == 1 &&
                second.scheduled_debt_ids.size() == 2 &&
                second_commit &&
                find_debt(scheduler, DebtKind::UnitAction, 0)->status ==
                    DebtStatus::Scheduled &&
                find_debt(scheduler, DebtKind::UnitAction, 1)->status ==
                    DebtStatus::Open,
            "partial joint plan emits confirmed current PLANT before retrying missing suffix");
  }
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation before; before.step = 100; before.money = 1000;
    const auto transaction = settle_zero(
        scheduler, seed_intent(73, Item::WHEAT, 1, 100, 120, {1, 1}), before);
    OwnObservation buy = before; buy.step = 101;
    PlanWindow purchase;
    purchase.turns.push_back({101, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
    purchase.turns.push_back({102, {}, {slot(0, {1, 1}, {Op::PASS, Item::NONE, 1})}});
    const auto p = scheduler.plan(transaction, buy, purchase);
    require(scheduler.commit(p, buy), "purchase committed before failed unit fixture");
    OwnObservation acquired = buy; acquired.step = 102; acquired.seeds[0] = 1;
    scheduler.observe(acquired);
    PlanWindow unit;
    unit.turns.push_back({102, {}, {slot(0, {1, 1}, {Op::PASS, Item::NONE, 1})}});
    const auto u = scheduler.plan(transaction, acquired, unit);
    require(scheduler.commit(u, acquired), "plant unit committed");
    const auto plant = find_debt(scheduler, DebtKind::UnitAction);
    UnitExecutionEvidence failed;
    failed.step = 103; failed.emitted = {Op::PLANT, Item::WHEAT, 1};
    failed.observed_tile_kind = fastkag::TileKind::WEED;
    failed.provenance = "weed_reappeared";
    require(scheduler.observe_unit_result(plant->id, failed) &&
                find_debt(scheduler, DebtKind::MarketPurchase)->status ==
                    DebtStatus::Confirmed &&
                find_debt(scheduler, DebtKind::UnitAction)->status ==
                    DebtStatus::Failed,
            "failed PLANT reopens unit suffix without reopening confirmed purchase");
    OwnObservation retry_state = acquired; retry_state.step = 103;
    PlanWindow retry_unit;
    retry_unit.turns.push_back({103, {}, {slot(0, {1, 1}, {Op::PASS, Item::NONE, 1})}});
    const auto retry = scheduler.plan(transaction, retry_state, retry_unit);
    require(retry.accepted && !retry.market.required,
            "failed unit is replanned without a duplicate seed buy");
  }
}

void test_multi_animal_chains_interleave_without_false_dependency() {
  auto config = step_window_experiment();
  FailureDebtScheduler scheduler(config);
  OwnObservation before; before.step = 120; before.money = 5000;
  before.actor_inventory.resize(2);
  before.actor_inventory[0][0] = 1;
  before.actor_inventory[1][0] = 1;
  auto intent = animal_intent(74, 120, true);
  intent.quantity = 2;
  intent.economic_value = 2400;
  auto second = intent.chains[0]; second.purchase_unit_index = 1;
  for (auto& requirement : second.requirements) requirement.actor = 1;
  intent.chains.push_back(second);
  const auto transaction = settle_zero(scheduler, intent, before);
  OwnObservation current = before; current.step = 121;
  PlanWindow window;
  window.turns.push_back({121, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
  const std::array<int, 5> offsets0{1, 3, 5, 7, 9};
  const std::array<int, 5> offsets1{2, 4, 6, 8, 10};
  const std::array<Position, 5> positions{{{4,4},{2,2},{2,2},{2,2},{2,2}}};
  for (int stage = 0; stage < 5; ++stage) {
    intent.chains[0].requirements[stage].earliest_step = 121 + offsets0[stage];
    intent.chains[0].requirements[stage].deadline = 121 + offsets0[stage];
    intent.chains[1].requirements[stage].earliest_step = 121 + offsets1[stage];
    intent.chains[1].requirements[stage].deadline = 121 + offsets1[stage];
    window.turns.push_back({121 + offsets0[stage], {},
                            {slot(0, positions[stage], {Op::PASS, Item::NONE, 1})}});
    window.turns.push_back({121 + offsets1[stage], {},
                            {slot(1, positions[stage], {Op::PASS, Item::NONE, 1})}});
  }
  // Re-record with interleaved deadlines; the first call above copied the old
  // intent, so use a fresh scheduler/transaction for the actual assertion.
  FailureDebtScheduler interleaved(config);
  const auto interleaved_transaction = settle_zero(interleaved, intent, before);
  std::sort(window.turns.begin(), window.turns.end(),
            [](const PlannedTurn& left, const PlannedTurn& right) {
              return left.step < right.step;
            });
  const auto plan = interleaved.plan(interleaved_transaction, current, window);
  require(plan.accepted && plan.market.addition.quantity == 2 &&
              plan.scheduled_debt_ids.size() == 10,
          "independent animal chains interleave; unit1 is not behind unit0 HARVEST");
  (void)transaction;
}

PurchaseIntent lifecycle_wheat_intent(std::uint64_t id, int plant_step,
                                      bool include_same_day_water) {
  PurchaseIntent intent;
  intent.attempt_id = id;
  intent.operation = Op::BUY_SEED;
  intent.item = Item::WHEAT;
  intent.quantity = 1;
  intent.deadline = plant_step + 60;
  intent.economic_value = 300;
  intent.provenance = "fixture:day-wheat";
  RecoveryChain chain;
  chain.requirements.push_back(requirement(
      Op::PLANT, Item::WHEAT, 0, {1, 1}, plant_step, plant_step, "plant"));
  if (include_same_day_water) {
    chain.requirements.push_back(requirement(
        Op::WATER, Item::NONE, 0, {1, 1}, plant_step + 1,
        plant_step + 1, "water-plant-day"));
  }
  chain.requirements.push_back(requirement(
      Op::WATER, Item::NONE, 0, {1, 1}, plant_step + 23,
      plant_step + 23, "water-next-day"));
  chain.requirements.push_back(requirement(
      Op::HARVEST, Item::NONE, 0, {1, 1}, plant_step + 46,
      plant_step + 46, "first-harvest"));
  intent.chains.push_back(std::move(chain));
  return intent;
}

PlanWindow dense_lifecycle_window(int begin, int end,
                                  const PurchaseIntent& intent,
                                  const DayLifecycleSpec& proof) {
  PlanWindow window;
  window.day_lifecycles.push_back(proof);
  for (int step = begin; step <= end; ++step) {
    PlannedTurn turn;
    turn.step = step;
    Position position{9, 9};
    for (const auto& action : intent.chains[0].requirements)
      if (action.earliest_step == step) position = action.position;
    turn.units.push_back(slot(0, position, {Op::PASS, Item::NONE, 1}));
    window.turns.push_back(std::move(turn));
  }
  return window;
}

void test_day_bucket_crop_lifecycle_and_fail_closed_edges() {
  SchedulerConfig config; config.enabled = true; config.require_day_lifecycle = true;
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation before; before.step = 0; before.money = 1000;
    const auto intent = lifecycle_wheat_intent(80, 2, true);
    const auto transaction = settle_zero(scheduler, intent, before);
    OwnObservation current = before; current.step = 1;
    DayLifecycleSpec proof;
    proof.purchase_unit_index = 0;
    proof.kind = LifecycleKind::Crop;
    proof.item = Item::WHEAT;
    proof.tile = {1, 1};
    proof.tile_observed_step = 1;
    proof.observed_kind = fastkag::TileKind::EMPTY;
    proof.required_harvest_cycles = 1;
    auto window = dense_lifecycle_window(1, 48, intent, proof);
    const auto plan = scheduler.plan(transaction, current, window);
    require(plan.accepted && plan.day_lifecycle_complete,
            "crop proof is rebuilt from actual plant day through WATER and maturity HARVEST");
  }
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation before; before.step = 0; before.money = 1000;
    const auto intent = lifecycle_wheat_intent(81, 2, false);
    const auto transaction = settle_zero(scheduler, intent, before);
    OwnObservation current = before; current.step = 1;
    DayLifecycleSpec proof;
    proof.purchase_unit_index = 0; proof.kind = LifecycleKind::Crop;
    proof.item = Item::WHEAT; proof.tile = {1, 1};
    proof.tile_observed_step = 1; proof.observed_kind = fastkag::TileKind::EMPTY;
    auto window = dense_lifecycle_window(1, 48, intent, proof);
    const auto plan = scheduler.plan(transaction, current, window);
    require(!plan.accepted &&
                plan.reason == "same_day_plant_missing_water_before_day_end",
            "late recovery PLANT without same-day WATER fails before day-end weed conversion");
  }
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation before; before.step = 0; before.money = 1000;
    const auto intent = lifecycle_wheat_intent(82, 25, true);
    const auto transaction = settle_zero(scheduler, intent, before);
    OwnObservation current = before; current.step = 1;
    DayLifecycleSpec proof;
    proof.purchase_unit_index = 0; proof.kind = LifecycleKind::Crop;
    proof.item = Item::WHEAT; proof.tile = {1, 1};
    proof.tile_observed_step = 1; proof.observed_kind = fastkag::TileKind::EMPTY;
    auto window = dense_lifecycle_window(1, 71, intent, proof);
    const auto plan = scheduler.plan(transaction, current, window);
    require(!plan.accepted && plan.reason == "future_empty_tile_weed_spawn_uncertified",
            "empty tile crossing day boundary cannot assume no random weed before PLANT");
  }
}

void test_day_bucket_animal_placed_day_feed_care_and_yield() {
  SchedulerConfig config; config.enabled = true; config.require_day_lifecycle = true;
  FailureDebtScheduler scheduler(config);
  OwnObservation before; before.step = 0; before.money = 3000;
  before.shed_used = 7; before.shed_items[0] = 7;
  before.actor_inventory.resize(1); before.actor_inventory[0][0] = 1;
  PurchaseIntent intent;
  intent.attempt_id = 86; intent.operation = Op::BUY_ANIMAL;
  intent.item = Item::COW; intent.quantity = 1; intent.deadline = 210;
  intent.economic_value = 1800; intent.provenance = "fixture:day-cow";
  RecoveryChain chain;
  chain.requirements.push_back(requirement(
      Op::PICKUP, Item::COW, 0, {4,4}, 2, 2, "pickup-cow"));
  chain.requirements.push_back(requirement(
      Op::PLACE, Item::COW, 0, {2,2}, 3, 3, "place-cow"));
  chain.requirements.push_back(requirement(
      Op::FEED, Item::WHEAT, 0, {2,2}, 4, 4, "feed-day0"));
  chain.requirements.push_back(requirement(
      Op::CARE, Item::NONE, 0, {2,2}, 5, 5, "care-day0"));
  for (int day = 1; day <= 7; ++day) {
    chain.requirements.push_back(requirement(
        Op::PICKUP, Item::WHEAT, 0, {4,4}, day * 24 + 1,
        day * 24 + 1, "pickup-feed-day" + std::to_string(day)));
    chain.requirements.push_back(requirement(
        Op::FEED, Item::WHEAT, 0, {2,2}, day * 24 + 2,
        day * 24 + 2, "feed-day" + std::to_string(day)));
  }
  chain.requirements.push_back(requirement(
      Op::HARVEST, Item::NONE, 0, {2,2}, 193, 193, "first-milk"));
  intent.chains.push_back(chain);
  const auto transaction = settle_zero(scheduler, intent, before);
  OwnObservation current = before; current.step = 1;
  PlanWindow window;
  DayLifecycleSpec proof;
  proof.purchase_unit_index = 0; proof.kind = LifecycleKind::Animal;
  proof.item = Item::COW; proof.tile = {2,2}; proof.tile_observed_step = 1;
  proof.observed_kind = fastkag::TileKind::PASTURE;
  proof.required_harvest_cycles = 1; proof.required_care_days = 1;
  window.day_lifecycles.push_back(proof);
  for (int step = 1; step <= 193; ++step) {
    PlannedTurn turn; turn.step = step;
    Position position{9,9};
    for (const auto& action : chain.requirements)
      if (action.earliest_step == step) position = action.position;
    turn.units.push_back(slot(0, position, {Op::PASS, Item::NONE, 1}));
    window.turns.push_back(std::move(turn));
  }
  const auto plan = scheduler.plan(transaction, current, window);
  require(plan.accepted && plan.day_lifecycle_complete,
          "animal proof rebuilds placed_day, daily FEED survival, fed+CARE bonus and first yield day");
}

void test_atomic_same_crop_plant_gate_and_false_expected_patch() {
  auto config = step_window_experiment();
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation before; before.step = 200; before.money = 1000;
    const auto intent = seed_intent(83, Item::CARROT, 1, 200, 220, {1, 1});
    const auto transaction = settle_zero(scheduler, intent, before);
    OwnObservation current = before; current.step = 201;
    PlanWindow window;
    window.turns.push_back({201, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
    window.turns.push_back({202, {}, {
        slot(0, {1, 1}, {Op::PASS, Item::NONE, 1}),
        slot(1, {2, 2}, {Op::PLANT, Item::CARROT, 1}, 9, false)}});
    const auto plan = scheduler.plan(transaction, current, window);
    require(!plan.accepted &&
                plan.reason == "same_tick_crop_plant_batch_underfunded_all_actions_would_pass",
            "same-crop PLANT demand is checked atomically, never sequentially funded");
  }
  {
    FailureDebtScheduler scheduler(config);
    OwnObservation before; before.step = 240; before.money = 2000;
    auto intent = animal_intent(84, 240, true);
    const auto transaction = settle_zero(scheduler, intent, before);
    OwnObservation current = before; current.step = 241;
    auto window = animal_window(241, true);
    // The FEED source itself is expected to fail, but its replacement still
    // consumes WHEAT and must be validated.
    window.turns[3].units[0].expected_success = false;
    const auto plan = scheduler.plan(transaction, current, window);
    require(!plan.accepted && plan.reason == "animal_chain_missing_feed_wheat",
            "expected_success=false source does not skip replacement resource accounting");
  }
}

void test_move_requirement_is_rejected_at_intent_boundary() {
  FailureDebtScheduler scheduler;
  OwnObservation before; before.step = 280; before.money = 1000;
  auto intent = seed_intent(87, Item::CARROT, 1, 280, 300, {1,1});
  intent.chains[0].requirements.push_back(requirement(
      Op::EAST, Item::NONE, 0, {1,1}, 301, 301, "illegal-move"));
  scheduler.observe(before);
  require(!scheduler.record_attempt(intent, before),
          "repair debt graph cannot contain an inserted/replaced MOVE requirement");
}

void test_idempotent_observe_and_expiry() {
  FailureDebtScheduler scheduler;
  OwnObservation before; before.step = 300; before.money = 1000;
  auto intent = seed_intent(85, Item::CARROT, 1, 300, 301, {1, 1});
  intent.deadline = 302;
  scheduler.observe(before);
  require(scheduler.record_attempt(intent, before), "expiry attempt recorded");
  require(scheduler.observe(before).empty() && scheduler.audit().size() == 1,
          "same-step observe is idempotent and does not reset pending attempts");
  OwnObservation failed = before; failed.step = 301;
  scheduler.observe(failed);
  require(!scheduler.open_transactions().empty(), "failed attempt creates debt before deadline");
  OwnObservation expired = failed; expired.step = 303;
  scheduler.observe(expired);
  require(scheduler.open_transactions().empty() &&
              std::all_of(scheduler.debts().begin(), scheduler.debts().end(),
                          [](const Debt& debt) {
                            return debt.status == DebtStatus::Retired;
                          }),
          "past-deadline debt retires deterministically instead of lingering forever");
}

void test_low_loss_and_default_off() {
  OwnObservation observation; observation.step = 800; observation.money = 1000;
  {
    FailureDebtScheduler scheduler;
    const auto transaction = settle_zero(
        scheduler, seed_intent(12, Item::TOMATO, 1, 800, 820, {1, 1}), observation);
    ++observation.step;
    PlanWindow window;
    window.turns.push_back({801, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
    window.turns.push_back({802, {}, {slot(0, {1, 1}, {Op::PASS, Item::NONE, 1})}});
    const auto plan = scheduler.plan(transaction, observation, window);
    require(!plan.accepted && plan.reason == "default_off",
            "deployment default remains byte-identical legacy behavior");
  }
  {
    auto config = step_window_experiment();
    FailureDebtScheduler scheduler(config);
    observation.step = 900;
    const auto transaction = settle_zero(
        scheduler, seed_intent(13, Item::TOMATO, 1, 900, 920, {1, 1}), observation);
    ++observation.step;
    PlanWindow window;
    window.turns.push_back({901, {}, {slot(0, {0, 0}, {Op::PASS, Item::NONE, 1})}});
    window.turns.push_back({902, {}, {slot(0, {1, 1}, {Op::HARVEST, Item::NONE, 1}, 12)}});
    const auto plan = scheduler.plan(transaction, observation, window);
    require(!plan.accepted && plan.reason == "no_complete_unit_chain",
            "valuable productive action is not silently absorbed");
  }
}

void test_static_dependency_fixture_and_audit_provenance() {
  std::vector<fastkag::PlayerAction> tape(5);
  tape[0].units = {{Op::PASS, Item::NONE, 1}};
  tape[0].market = {{Op::BUY_SEED, Item::CARROT, 1},
                    {Op::BUY_ANIMAL, Item::GOOSE, 1}};
  tape[1].units = {{Op::PLANT, Item::CARROT, 1}};
  tape[2].units = {{Op::PICKUP, Item::GOOSE, 1}};
  tape[3].units = {{Op::PLACE, Item::GOOSE, 1}};
  const auto dependencies = extract_static_dependencies(tape);
  require(dependencies.size() == 2 && dependencies[0].complete &&
              dependencies[0].terminal_unit_step == 1 &&
              dependencies[1].complete &&
              dependencies[1].first_unit_step == 2 &&
              dependencies[1].terminal_unit_step == 3,
          "static own-tape scanner extracts seed and animal phase dependencies");

  AuditEvent event{4, 7, "repair_rejected", "missing chain",
                   "G001:step0:slot1", 2, 1, 1, 0};
  const auto json = audit_json(event);
  require(json.find("G001:step0:slot1") != std::string::npos &&
              json.find("opponent") == std::string::npos,
          "audit is per-hand/provenance-rich and has no opponent identity field");
}

}  // namespace

int main() {
  test_causal_full_partial_zero();
  test_incomplete_animal_intent_is_rejected_before_settlement();
  test_seed_repair_waits_for_distant_slack_and_preserves_move();
  test_same_step_market_cannot_repair_unit_phase();
  test_complete_animal_transaction();
  test_cross_day_animal_transaction_requires_item_specific_pickup();
  test_animal_rejects_incomplete_or_resource_unsafe_chain();
  test_market_slots_and_future_purchase_reservation();
  test_low_loss_and_default_off();
  test_ambiguous_settlement_fails_closed();
  test_explicit_plan_commit_settle_unit_lifecycle();
  test_future_market_preview_cannot_commit_or_mutate();
  test_retry_zero_partial_and_failed_unit_reopen_without_duplicate_buy();
  test_multi_animal_chains_interleave_without_false_dependency();
  test_day_bucket_crop_lifecycle_and_fail_closed_edges();
  test_day_bucket_animal_placed_day_feed_care_and_yield();
  test_atomic_same_crop_plant_gate_and_false_expected_patch();
  test_move_requirement_is_rejected_at_intent_boundary();
  test_idempotent_observe_and_expiry();
  test_static_dependency_fixture_and_audit_provenance();
  std::cout << "failure debt scheduler tests passed\n";
}
