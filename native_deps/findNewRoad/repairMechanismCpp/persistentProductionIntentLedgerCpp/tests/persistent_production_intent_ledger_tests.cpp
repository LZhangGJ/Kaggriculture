#include "persistent_production_intent_ledger.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace persistent = g001::persistent_production;
namespace purchase = g001::purchase_recovery;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::TileKind;

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::size_t index(Item item) {
  return static_cast<std::size_t>(static_cast<int>(item));
}

std::size_t animal_index(Item item) { return index(item) - index(Item::GOOSE); }

persistent::ObjectiveSpec crop_spec(int day, persistent::TileKey tile,
                                    Item crop, int quantity = 1,
                                    int deadline = 8, int value = 50) {
  static constexpr std::array<int, 5> cost{10, 20, 50, 100, 80};
  return {{day, tile},
          persistent::Kind::Crop,
          crop,
          quantity,
          cost[index(crop)],
          deadline,
          true,
          value,
          "crop-intent"};
}

persistent::ObjectiveSpec animal_spec(int day, persistent::TileKey tile,
                                      Item animal, int deadline = 20) {
  const auto definition = g001::animal_lifecycle::definition(animal);
  return {{day, tile},
          persistent::Kind::Animal,
          animal,
          1,
          definition->purchase_cost,
          deadline,
          true,
          500,
          "animal-intent"};
}

persistent::Observation observation(int step, int day,
                                    persistent::TileKey target_key,
                                    persistent::ActorIdentity actor,
                                    persistent::TileKey actor_position,
                                    TileKind tile_kind) {
  persistent::Observation value;
  value.step = step;
  value.day = day;
  value.target_key = target_key;
  value.target.kind = tile_kind;
  value.actor.identity = actor;
  value.actor.position = actor_position;
  return value;
}

std::vector<persistent::PurchaseSettlement>
purchase_attempt(persistent::Ledger &ledger, std::uint64_t objective, int step,
                 int cash, int protected_cash, int holding_before, int fill,
                 int holding_after, Item item, bool animal) {
  const auto proposal =
      ledger.propose_purchase(objective, {step, cash, protected_cash});
  check(proposal.has_value(), "funded purchase was not proposed");
  std::vector<Action> final_market{{Op::SELL, Item::EGG, 1}, proposal->order};
  check(ledger.stage_purchase(
            objective, *proposal,
            {step, 1, final_market, holding_before, cash, protected_cash}) ==
            purchase::StageStatus::Selected,
        "exact final market slot was rejected");
  const std::array<std::int32_t, 2> fills{0, fill};
  purchase::ReceiptObservation receipt;
  receipt.step = step + 1;
  receipt.slot_fills = fills;
  if (animal)
    receipt.animals_after[animal_index(item)] = holding_after;
  else
    receipt.seeds_after[index(item)] = holding_after;
  return ledger.observe_purchases(receipt);
}

persistent::ReadyTransition only_ready(persistent::Ledger &ledger,
                                       std::uint64_t objective,
                                       persistent::ActorIdentity actor,
                                       const persistent::Observation &before,
                                       persistent::ResourceSnapshot resources) {
  const auto lease = ledger.lease_actor(objective, actor, before.step);
  check(lease.has_value(), "actor lease was not issued");
  const std::array<persistent::ReadyRequest, 1> requests{
      persistent::ReadyRequest{objective, *lease, before}};
  const auto ready = ledger.ready_transitions(requests, resources);
  check(ready.size() == 1, "expected exactly one ready transition");
  return ready.front();
}

std::uint64_t stage(persistent::Ledger &ledger,
                    const persistent::ReadyTransition &transition,
                    const persistent::Observation &before) {
  std::vector<Action> units(
      static_cast<std::size_t>(transition.lease.actor.actor_id + 1));
  units[static_cast<std::size_t>(transition.lease.actor.actor_id)] =
      transition.action;
  const auto prefix = persistent::make_prefix_authority(
      before.step, transition.lease.actor.actor_id, 1,
      persistent::observation_fingerprint(before), units, before);
  check(ledger.stage_transition(transition,
                                {before.step, transition.lease.actor.actor_id,
                                 units, before, prefix, before.seeds, true}) ==
            persistent::UnitStageStatus::Selected,
        "exact final unit slot was rejected");
  return persistent::prefix_authority_fingerprint(prefix);
}

void partial_and_zero_seed_receipts_remain_debt() {
  persistent::Ledger ledger;
  const auto opened = ledger.open(crop_spec(0, {1, 1}, Item::WHEAT, 2));
  check(opened.status == persistent::OpenStatus::Opened,
        "two-seed objective did not open");

  const auto zero = purchase_attempt(ledger, opened.objective_id, 0, 20, 0, 0,
                                     0, 0, Item::WHEAT, false);
  check(zero.size() == 1 &&
            zero.front().settlement.status == purchase::FillStatus::Zero &&
            !ledger.objective(opened.objective_id)->purchase_complete,
        "zero fill retired seed debt");

  const auto partial = purchase_attempt(ledger, opened.objective_id, 2, 20, 0,
                                        0, 1, 1, Item::WHEAT, false);
  check(partial.size() == 1 &&
            partial.front().settlement.status ==
                purchase::FillStatus::Partial &&
            partial.front().settlement.remaining == 1 &&
            !ledger.objective(opened.objective_id)->purchase_complete,
        "partial seed fill lost missing suffix");

  const auto full = purchase_attempt(ledger, opened.objective_id, 4, 10, 0, 1,
                                     1, 2, Item::WHEAT, false);
  check(full.size() == 1 &&
            full.front().settlement.status == purchase::FillStatus::Full &&
            ledger.objective(opened.objective_id)->purchase_complete &&
            ledger.seed_purchase_audit().zero_fills == 1 &&
            ledger.seed_purchase_audit().partial_fills == 1,
        "cash recovery did not close exact remaining seed suffix");
}

void cross_day_weed_dig_keeps_crop_then_plants() {
  persistent::Ledger ledger;
  const auto opened = ledger.open(crop_spec(0, {2, 3}, Item::CARROT));
  purchase_attempt(ledger, opened.objective_id, 0, 20, 0, 0, 1, 1, Item::CARROT,
                   false);

  const persistent::ActorIdentity actor{0, 11};
  auto weed = observation(25, 1, {2, 3}, actor, {2, 3}, TileKind::WEED);
  weed.seeds[index(Item::CARROT)] = 1;
  check(ledger.reconcile(opened.objective_id, weed) ==
                persistent::Status::Open &&
            ledger.objective(opened.objective_id)->spec.key.origin_day == 0 &&
            ledger.objective(opened.objective_id)->crop_stage ==
                persistent::CropStage::NeedDig,
        "midnight cleared or re-keyed plot debt");
  persistent::ResourceSnapshot resources;
  resources.seeds[index(Item::CARROT)] = 1;
  auto dig = only_ready(ledger, opened.objective_id, actor, weed, resources);
  check(dig.action.op == Op::DIG, "weed objective did not emit DIG");
  const auto dig_prefix = stage(ledger, dig, weed);
  auto empty = weed;
  empty.step = 26;
  empty.target = {};
  check(ledger.observe_transition(opened.objective_id, dig.lease.token,
                                  dig_prefix, empty)
                    .status == persistent::UnitReceiptStatus::Success &&
            ledger.objective(opened.objective_id)->crop_stage ==
                persistent::CropStage::NeedPlant &&
            ledger.objective(opened.objective_id)->spec.item == Item::CARROT,
        "DIG receipt lost desired crop continuation");

  auto plant = only_ready(ledger, opened.objective_id, actor, empty, resources);
  check(plant.action.op == Op::PLANT && plant.action.item == Item::CARROT,
        "post-DIG continuation did not emit desired PLANT");
  const auto plant_prefix = stage(ledger, plant, empty);
  auto planted = empty;
  planted.step = 27;
  planted.target.kind = TileKind::PLANT;
  planted.target.crop = Item::CARROT;
  planted.target.planted_day = 1;
  planted.seeds[index(Item::CARROT)] = 0;
  check(ledger.observe_transition(opened.objective_id, plant.lease.token,
                                  plant_prefix, planted)
                    .status == persistent::UnitReceiptStatus::Success &&
            ledger.objective(opened.objective_id)->status ==
                persistent::Status::Open &&
            ledger.objective(opened.objective_id)->crop_stage ==
                persistent::CropStage::NeedWater,
        "PLANT incorrectly retired objective before WATER");

  // The plot intent remains owned across another midnight. The next day's
  // WATER, not PLANT, is the minimum closed production suffix.
  auto next_day = planted;
  next_day.step = 48;
  next_day.day = 2;
  auto water =
      only_ready(ledger, opened.objective_id, actor, next_day, resources);
  check(water.action.op == Op::WATER,
        "unwatered planted crop did not expose WATER continuation");
  const auto water_prefix = stage(ledger, water, next_day);
  auto watered = next_day;
  watered.step = 49;
  watered.target.watered_today = true;
  check(ledger.observe_transition(opened.objective_id, water.lease.token,
                                  water_prefix, watered)
                    .status == persistent::UnitReceiptStatus::Success &&
            ledger.objective(opened.objective_id)->status ==
                persistent::Status::Complete,
        "WATER physical receipt did not retire closed crop suffix");
  check(ledger.objectives_for_day(0).size() == 1,
        "day-primary history disappeared after completion");
}

void duplicate_receipt_is_idempotent() {
  persistent::Ledger ledger;
  const auto opened = ledger.open(crop_spec(0, {0, 2}, Item::WHEAT));
  purchase_attempt(ledger, opened.objective_id, 0, 10, 0, 0, 1, 1, Item::WHEAT,
                   false);
  const persistent::ActorIdentity actor{1, 4};
  auto before = observation(2, 0, {0, 2}, actor, {0, 2}, TileKind::EMPTY);
  before.seeds[index(Item::WHEAT)] = 1;
  persistent::ResourceSnapshot resources;
  resources.seeds[index(Item::WHEAT)] = 1;
  auto plant =
      only_ready(ledger, opened.objective_id, actor, before, resources);
  const auto plant_prefix = stage(ledger, plant, before);
  auto after = before;
  after.step = 3;
  after.target.kind = TileKind::PLANT;
  after.target.crop = Item::WHEAT;
  after.seeds[index(Item::WHEAT)] = 0;
  const auto first = ledger.observe_transition(
      opened.objective_id, plant.lease.token, plant_prefix, after);
  const auto duplicate = ledger.observe_transition(
      opened.objective_id, plant.lease.token, plant_prefix, after);
  check(first.status == persistent::UnitReceiptStatus::Success &&
            duplicate.status ==
                persistent::UnitReceiptStatus::UnknownOrDuplicate &&
            ledger.objective(opened.objective_id)->status ==
                persistent::Status::Open &&
            ledger.objective(opened.objective_id)->crop_stage ==
                persistent::CropStage::NeedWater &&
            ledger.audit().completed == 0 &&
            ledger.audit().duplicate_receipts == 1,
        "duplicate receipt replayed or reverted objective");
}

void disappearing_worker_requires_failed_receipt_then_rebind() {
  persistent::Ledger ledger;
  const auto opened = ledger.open(crop_spec(0, {3, 3}, Item::MELON));
  purchase_attempt(ledger, opened.objective_id, 0, 80, 0, 0, 1, 1, Item::MELON,
                   false);
  const persistent::ActorIdentity old_worker{2, 7};
  auto before = observation(23, 0, {3, 3}, old_worker, {3, 3}, TileKind::WEED);
  before.seeds[index(Item::MELON)] = 1;
  persistent::ResourceSnapshot resources;
  resources.seeds[index(Item::MELON)] = 1;
  auto dig =
      only_ready(ledger, opened.objective_id, old_worker, before, resources);
  const auto dig_prefix = stage(ledger, dig, before);
  check(ledger.actor_disappeared(old_worker) == 0,
        "pending old-worker lease was silently discarded");

  const persistent::ActorIdentity replacement{2, 8};
  auto failed = before;
  failed.step = 24;
  failed.day = 1;
  failed.actor.identity = replacement;
  const auto settlement = ledger.observe_transition(
      opened.objective_id, dig.lease.token, dig_prefix, failed);
  check(settlement.status == persistent::UnitReceiptStatus::Ambiguous &&
            ledger.objective(opened.objective_id)->status ==
                persistent::Status::Open,
        "old generation mismatch cleared plot debt");

  const auto retry =
      only_ready(ledger, opened.objective_id, replacement, failed, resources);
  check(retry.action.op == Op::DIG &&
            retry.lease.actor.generation == replacement.generation,
        "new worker generation could not claim retained tile objective");
}

void resource_competition_returns_only_priority_feasible_prefix() {
  persistent::Ledger ledger;
  auto high_spec = crop_spec(0, {1, 4}, Item::TOMATO, 1, 5, 500);
  auto low_spec = crop_spec(0, {2, 4}, Item::TOMATO, 1, 6, 10);
  const auto high = ledger.open(high_spec);
  const auto low = ledger.open(low_spec);
  purchase_attempt(ledger, high.objective_id, 0, 50, 0, 0, 1, 1, Item::TOMATO,
                   false);
  purchase_attempt(ledger, low.objective_id, 2, 50, 0, 1, 1, 2, Item::TOMATO,
                   false);
  const persistent::ActorIdentity actor0{0, 1};
  const persistent::ActorIdentity actor1{1, 1};
  auto first = observation(4, 0, {1, 4}, actor0, {1, 4}, TileKind::EMPTY);
  auto second = observation(4, 0, {2, 4}, actor1, {2, 4}, TileKind::EMPTY);
  first.seeds[index(Item::TOMATO)] = 1;
  second.seeds[index(Item::TOMATO)] = 1;
  const auto lease0 = ledger.lease_actor(high.objective_id, actor0, 4);
  const auto lease1 = ledger.lease_actor(low.objective_id, actor1, 4);
  check(lease0 && lease1, "competition leases were not issued");
  const std::array<persistent::ReadyRequest, 2> requests{
      persistent::ReadyRequest{low.objective_id, *lease1, second},
      persistent::ReadyRequest{high.objective_id, *lease0, first}};
  persistent::ResourceSnapshot resources;
  resources.seeds[index(Item::TOMATO)] = 1;
  const auto ready = ledger.ready_transitions(requests, resources);
  check(ready.size() == 1 && ready.front().objective_id == high.objective_id &&
            ledger.audit().resource_deferred == 1,
        "shared seed was double-spent or priority order was ignored");

  const auto conflict = ledger.open(crop_spec(1, {1, 4}, Item::WHEAT));
  check(conflict.status == persistent::OpenStatus::ActiveTileOwner,
        "same tile was not serialized across origin days");
}

persistent::ReadyTransition
animal_ready(persistent::Ledger &ledger, std::uint64_t objective,
             persistent::ActorIdentity actor,
             const persistent::Observation &before) {
  persistent::ResourceSnapshot resources;
  resources.shed = before.shed;
  return only_ready(ledger, objective, actor, before, resources);
}

persistent::Observation
animal_settle(persistent::Ledger &ledger,
              const persistent::ReadyTransition &transition,
              persistent::Observation before, persistent::Observation after) {
  const auto prefix = stage(ledger, transition, before);
  check(ledger.observe_transition(transition.objective_id,
                                  transition.lease.token, prefix, after)
                .status == persistent::UnitReceiptStatus::Success,
        "animal exact unit receipt failed");
  return after;
}

void animal_zero_fill_recovery_through_feed_care_first_yield() {
  persistent::Ledger ledger;
  const persistent::TileKey target{4, 4};
  const auto opened = ledger.open(animal_spec(3, target, Item::COW));
  check(opened.status == persistent::OpenStatus::Opened,
        "cow objective did not open");
  const auto zero = purchase_attempt(ledger, opened.objective_id, 72, 500, 0, 0,
                                     0, 0, Item::COW, true);
  check(zero.front().settlement.status == purchase::FillStatus::Zero &&
            !ledger.objective(opened.objective_id)->purchase_complete,
        "zero animal fill retired acquisition");
  const auto full = purchase_attempt(ledger, opened.objective_id, 74, 600, 100,
                                     0, 1, 1, Item::COW, true);
  check(full.front().settlement.status == purchase::FillStatus::Full &&
            ledger.objective(opened.objective_id)->purchase_complete,
        "cash recovery did not hand animal inventory to lifecycle");

  const persistent::ActorIdentity actor{0, 31};
  fastkag::Tile pasture;
  pasture.kind = TileKind::PASTURE;
  auto pickup_before =
      observation(75, 3, target, actor, {5, 5}, TileKind::PASTURE);
  pickup_before.actor.shed_adjacent = true;
  pickup_before.shed[index(Item::COW)] = 1;
  auto pickup = animal_ready(ledger, opened.objective_id, actor, pickup_before);
  check(pickup.action.op == Op::PICKUP,
        "animal lifecycle did not request PICKUP");
  auto pickup_after = pickup_before;
  pickup_after.step = 76;
  pickup_after.shed[index(Item::COW)] = 0;
  pickup_after.actor.inventory[index(Item::COW)] = 1;
  animal_settle(ledger, pickup, pickup_before, pickup_after);

  auto place_before = pickup_after;
  place_before.actor.position = target;
  auto place = animal_ready(ledger, opened.objective_id, actor, place_before);
  check(place.action.op == Op::PLACE, "animal lifecycle skipped PLACE");
  auto place_after = place_before;
  place_after.step = 77;
  place_after.actor.inventory[index(Item::COW)] = 0;
  place_after.target = {};
  place_after.target.kind = TileKind::ANIMAL;
  place_after.target.animal = Item::COW;
  place_after.target.placed_day = 3;
  animal_settle(ledger, place, place_before, place_after);
  check(ledger.objective(opened.objective_id)->status ==
            persistent::Status::Open,
        "successful PLACE incorrectly retired lifecycle");

  auto feed_before = place_after;
  feed_before.actor.inventory[index(Item::WHEAT)] = 1;
  auto feed = animal_ready(ledger, opened.objective_id, actor, feed_before);
  check(feed.action.op == Op::FEED, "animal lifecycle skipped FEED");
  auto feed_after = feed_before;
  feed_after.step = 78;
  feed_after.target.fed_today = true;
  feed_after.actor.inventory[index(Item::WHEAT)] = 0;
  animal_settle(ledger, feed, feed_before, feed_after);

  auto care = animal_ready(ledger, opened.objective_id, actor, feed_after);
  check(care.action.op == Op::CARE, "animal lifecycle skipped CARE");
  auto care_after = feed_after;
  care_after.step = 79;
  care_after.target.cared_today = true;
  animal_settle(ledger, care, feed_after, care_after);

  auto yield_before = care_after;
  yield_before.step = 264;
  yield_before.day = 11;
  yield_before.target.yield_units = 1;
  auto harvest = animal_ready(ledger, opened.objective_id, actor, yield_before);
  check(harvest.action.op == Op::HARVEST && harvest.action.item == Item::MILK,
        "mature cow did not expose typed first-yield transition");
  auto yield_after = yield_before;
  yield_after.step = 265;
  yield_after.target.yield_units = 0;
  yield_after.actor.inventory[index(Item::MILK)] = 1;
  animal_settle(ledger, harvest, yield_before, yield_after);
  check(ledger.objective(opened.objective_id)->status ==
            persistent::Status::Complete,
        "first-yield receipt did not retire animal objective");
}

void expiration_occurs_only_after_unsatisfied_deadline() {
  persistent::Ledger ledger;
  auto spec = crop_spec(1, {5, 1}, Item::STRAWBERRY, 1, 2);
  const auto first = ledger.open(spec);
  auto satisfied = observation(80, 3, {5, 1}, {0, 1}, {0, 0}, TileKind::PLANT);
  satisfied.target.crop = Item::STRAWBERRY;
  satisfied.target.watered_today = true;
  check(ledger.reconcile(first.objective_id, satisfied) ==
            persistent::Status::Complete,
        "physical satisfaction after deadline was incorrectly expired");

  spec.key = {1, {5, 2}};
  const auto second = ledger.open(spec);
  auto missed = observation(80, 3, {5, 2}, {0, 1}, {0, 0}, TileKind::EMPTY);
  check(ledger.reconcile(second.objective_id, missed) ==
            persistent::Status::Expired,
        "unsatisfied objective did not expire after deadline");

  spec.key = {1, {5, 3}};
  spec.deadline_day = 5;
  const auto third = ledger.open(spec);
  auto mismatched = observation(81, 3, {9, 9}, {0, 1}, {0, 0}, TileKind::EMPTY);
  check(ledger.reconcile(third.objective_id, mismatched) ==
                persistent::Status::FailClosed &&
            ledger.active_objectives().empty(),
        "FailClosed objective leaked into default schedulable-active query");
}

void typed_animal_loss_reopens_same_persistent_objective() {
  persistent::Ledger ledger;
  const auto opened = ledger.open(animal_spec(0, {6, 6}, Item::GOOSE));
  purchase_attempt(ledger, opened.objective_id, 0, 300, 0, 0, 1, 1, Item::GOOSE,
                   true);
  const auto before = *ledger.objective(opened.objective_id);
  check(before.purchase_complete,
        "persistent loss fixture acquisition did not complete");
  check(ledger.reopen_animal_acquisition(
            opened.objective_id,
            {2, 0, {6, 6}, Item::GOOSE, 0, 1, 0, 0, false}) ==
            persistent::ReopenAnimalAcquisitionStatus::InventoryStillAvailable,
        "whole-player actor inventory was ignored by persistent proof");
  check(ledger.reopen_animal_acquisition(
            opened.objective_id,
            {2, 0, {6, 6}, Item::GOOSE, 0, 0, 1, 0, false}) ==
            persistent::ReopenAnimalAcquisitionStatus::InventoryStillAvailable,
        "whole-player board inventory was ignored by persistent proof");
  check(ledger.reopen_animal_acquisition(
            opened.objective_id,
            {2, 0, {6, 6}, Item::GOOSE, 0, 0, 1, 1, false}) ==
            persistent::ReopenAnimalAcquisitionStatus::Reopened,
        "persistent owner rejected item allocation loss proof");
  const auto reopened = *ledger.objective(opened.objective_id);
  check(!reopened.purchase_complete && reopened.id == before.id &&
            reopened.purchase_debt_id == before.purchase_debt_id &&
            reopened.revision == before.revision + 1,
        "reopen changed objective identity/debt or missed revision advance");
  check(ledger.reopen_animal_acquisition(
            opened.objective_id,
            {2, 0, {6, 6}, Item::GOOSE, 0, 0, 1, 1, false}) ==
            persistent::ReopenAnimalAcquisitionStatus::AlreadyOpen,
        "persistent repeated reopen was not idempotent");
  const auto retry = ledger.propose_purchase(opened.objective_id, {3, 300, 0});
  check(retry && retry->order.op == Op::BUY_ANIMAL &&
            retry->order.item == Item::GOOSE && retry->order.quantity == 1,
        "same persistent animal objective did not expose replacement BUY");
}

} // namespace

int main() {
  try {
    partial_and_zero_seed_receipts_remain_debt();
    cross_day_weed_dig_keeps_crop_then_plants();
    duplicate_receipt_is_idempotent();
    disappearing_worker_requires_failed_receipt_then_rebind();
    resource_competition_returns_only_priority_feasible_prefix();
    animal_zero_fill_recovery_through_feed_care_first_yield();
    expiration_occurs_only_after_unsatisfied_deadline();
    typed_animal_loss_reopens_same_persistent_objective();
    std::cout
        << "persistent production intent ledger: 8 adversarial groups passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "persistent production intent ledger FAIL: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
}
