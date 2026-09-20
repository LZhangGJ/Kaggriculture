#include "animal_lifecycle_repair.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace lifecycle = g001::animal_lifecycle;
namespace purchase = g001::purchase_recovery;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::TileKind;

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::size_t item(Item value) {
  return static_cast<std::size_t>(static_cast<int>(value));
}

std::size_t animal(Item value) {
  return static_cast<std::size_t>(static_cast<int>(value) -
                                  static_cast<int>(Item::GOOSE));
}

lifecycle::Observation observation(int step, int day,
                                   lifecycle::ActorIdentity identity,
                                   lifecycle::TileKey position,
                                   lifecycle::TileSnapshot target) {
  lifecycle::Observation value;
  value.step = step;
  value.day = day;
  value.target = target;
  value.actor.identity = identity;
  value.actor.position = position;
  return value;
}

void acquire(lifecycle::Ledger &ledger, std::uint64_t objective_id,
             Item animal_item, int cost, int step = 0) {
  const auto proposal =
      ledger.propose_acquisition(objective_id, {step, cost, 0});
  check(proposal.has_value(), "funded acquisition was not proposed");
  std::vector<Action> market{{Op::HIRE, Item::NONE, 1}, proposal->order};
  check(ledger.stage_acquisition(*proposal, {step, 1, market, 0, cost, 0}) ==
            purchase::StageStatus::Selected,
        "final composer exact animal slot was rejected");
  const std::array<std::int32_t, 2> fills{0, 1};
  purchase::ReceiptObservation receipt;
  receipt.step = step + 1;
  receipt.slot_fills = fills;
  receipt.animals_after[animal(animal_item)] = 1;
  const auto settlements = ledger.observe_acquisitions(receipt);
  check(settlements.size() == 1 &&
            settlements.front().status == purchase::FillStatus::Full,
        "exact animal fill did not settle acquisition");
}

lifecycle::UnitProposal propose(lifecycle::Ledger &ledger,
                                std::uint64_t objective_id,
                                lifecycle::ActorIdentity actor,
                                const lifecycle::Observation &before) {
  const auto lease = ledger.lease_actor(objective_id, actor, before.step);
  check(lease.has_value(), "actor lease was not issued");
  const auto decision = ledger.propose_unit(objective_id, *lease, before);
  check(decision.blocked == lifecycle::BlockReason::None &&
            decision.proposal.has_value(),
        "unit action was unexpectedly blocked");
  return *decision.proposal;
}

void stage(lifecycle::Ledger &ledger, const lifecycle::UnitProposal &proposal,
           int slot = 0, int width = 1) {
  std::vector<Action> units(static_cast<std::size_t>(width));
  units[static_cast<std::size_t>(slot)] = proposal.action;
  check(ledger.stage_unit(proposal, {proposal.exact_before.step, slot, units,
                                     proposal.exact_before}) ==
            lifecycle::UnitStageStatus::Selected,
        "exact final unit slot was rejected");
}

lifecycle::UnitSettlement settle(lifecycle::Ledger &ledger,
                                 const lifecycle::UnitProposal &proposal,
                                 lifecycle::Observation after) {
  return ledger.observe_unit({proposal.objective_id, after.step,
                              proposal.lease.token, std::move(after)});
}

void animal_definitions_and_tile_owner_are_explicit() {
  const auto goose = lifecycle::definition(Item::GOOSE);
  const auto cow = lifecycle::definition(Item::COW);
  const auto sheep = lifecycle::definition(Item::SHEEP);
  check(goose && goose->required_structure == TileKind::COOP &&
            goose->first_product == Item::EGG && goose->first_yield_days == 4 &&
            goose->purchase_cost == 300,
        "goose action/lifecycle ABI is wrong");
  check(cow && cow->required_structure == TileKind::PASTURE &&
            cow->first_product == Item::MILK && cow->first_yield_days == 8 &&
            cow->yield_interval_days == 2,
        "cow action/lifecycle ABI is wrong");
  check(sheep && sheep->required_structure == TileKind::PASTURE &&
            sheep->first_product == Item::WOOL &&
            sheep->first_yield_days == 6 && sheep->yield_interval_days == 3,
        "sheep action/lifecycle ABI is wrong");
  check(!lifecycle::definition(Item::WHEAT),
        "unsupported animal did not fail closed");

  lifecycle::Ledger ledger;
  const auto first = ledger.open({Item::GOOSE, {2, 3}, 300, "goose-A"});
  const auto conflict = ledger.open({Item::COW, {2, 3}, 400, "cow-B"});
  check(first.status == lifecycle::OpenStatus::Opened &&
            conflict.status == lifecycle::OpenStatus::TileAlreadyOwned &&
            ledger.audit().tile_owner_rejections == 1,
        "same target tile was not serialized by persistent ownership");
  check(ledger.open({Item::GOOSE, {3, 3}, 400, "wrong-cost"}).status ==
            lifecycle::OpenStatus::CostMismatch,
        "noncanonical animal cost did not fail closed");
}

void zero_fill_cash_recovery_retry_then_inventory_only_handoff() {
  lifecycle::Ledger ledger;
  const auto opened = ledger.open({Item::COW, {4, 4}, 400, "cash-retry"});
  check(opened.status == lifecycle::OpenStatus::Opened,
        "cash retry objective did not open");
  check(!ledger.propose_acquisition(opened.objective_id, {10, 399, 0}),
        "unfunded cow retry was proposed");

  const auto first =
      ledger.propose_acquisition(opened.objective_id, {10, 500, 0});
  check(first.has_value(), "funded first cow attempt was not proposed");
  std::vector<Action> market{{Op::SELL, Item::EGG, 1}, first->order};
  check(ledger.stage_acquisition(*first, {10, 1, market, 0, 500, 0}) ==
            purchase::StageStatus::Selected,
        "first cow attempt exact slot was rejected");
  const std::array<std::int32_t, 2> zero_fills{1, 0};
  purchase::ReceiptObservation zero;
  zero.step = 11;
  zero.slot_fills = zero_fills;
  const auto zero_result = ledger.observe_acquisitions(zero);
  check(zero_result.size() == 1 &&
            zero_result.front().status == purchase::FillStatus::Zero &&
            ledger.objective(opened.objective_id)->stage ==
                lifecycle::Stage::Acquire,
        "zero fill cleared animal acquisition debt");

  const auto retry =
      ledger.propose_acquisition(opened.objective_id, {12, 700, 200});
  check(retry && retry->order.quantity == 1,
        "cash recovery did not repropose one cow");
  std::vector<Action> retry_market{retry->order};
  check(ledger.stage_acquisition(*retry, {12, 0, retry_market, 0, 700, 200}) ==
            purchase::StageStatus::Selected,
        "cash recovery exact slot was rejected");
  const std::array<std::int32_t, 1> full_fill{1};
  purchase::ReceiptObservation full;
  full.step = 13;
  full.slot_fills = full_fill;
  full.animals_after[animal(Item::COW)] = 1;
  const auto full_result = ledger.observe_acquisitions(full);
  const auto view = ledger.objective(opened.objective_id);
  check(full_result.size() == 1 &&
            full_result.front().status == purchase::FillStatus::Full && view &&
            view->stage == lifecycle::Stage::PickupFromShed &&
            !view->complete && ledger.audit().purchase_handoffs == 1 &&
            ledger.purchase_audit().zero_fills == 1,
        "animal fill was confused with PLACE/lifecycle completion");
}

void exact_goose_lifecycle_reaches_first_yield_only_after_harvest_receipt() {
  lifecycle::Ledger ledger;
  const auto opened = ledger.open({Item::GOOSE, {2, 2}, 300, "goose-cycle"});
  acquire(ledger, opened.objective_id, Item::GOOSE, 300);
  const lifecycle::ActorIdentity first_actor{0, 7};

  lifecycle::TileSnapshot coop;
  coop.kind = TileKind::COOP;
  auto pickup_before = observation(1, 0, first_actor, {4, 4}, coop);
  pickup_before.actor.shed_adjacent = true;
  pickup_before.shed[item(Item::GOOSE)] = 1;
  const auto pickup =
      propose(ledger, opened.objective_id, first_actor, pickup_before);
  check(pickup.action.op == Op::PICKUP, "PICKUP stage emitted wrong op");
  auto forged = pickup;
  forged.action = {Op::HARVEST, Item::EGG, 1};
  std::vector<Action> forged_units{forged.action};
  check(ledger.stage_unit(forged, {forged.exact_before.step, 0, forged_units,
                                   forged.exact_before}) ==
            lifecycle::UnitStageStatus::StaleProposal,
        "caller-forged lifecycle action bypassed stage certification");
  stage(ledger, pickup);
  auto pickup_after = pickup_before;
  pickup_after.step = 2;
  pickup_after.shed[item(Item::GOOSE)] = 0;
  pickup_after.actor.inventory[item(Item::GOOSE)] = 1;
  check(settle(ledger, pickup, pickup_after).status ==
            lifecycle::UnitReceiptStatus::Success,
        "PICKUP exact receipt failed");

  // A new generation safely claims the plot-owned objective. The old actor
  // generation is not persistent ownership.
  const lifecycle::ActorIdentity replacement{0, 8};
  auto place_before = observation(2, 0, replacement, {2, 2}, coop);
  place_before.actor.inventory[item(Item::GOOSE)] = 1;
  const auto place =
      propose(ledger, opened.objective_id, replacement, place_before);
  check(place.action.op == Op::PLACE, "PLACE stage emitted wrong op");
  stage(ledger, place);
  auto place_after = place_before;
  place_after.step = 3;
  place_after.actor.inventory[item(Item::GOOSE)] = 0;
  place_after.target.kind = TileKind::ANIMAL;
  place_after.target.animal = Item::GOOSE;
  place_after.target.placed_day = 0;
  check(settle(ledger, place, place_after).status ==
            lifecycle::UnitReceiptStatus::Success,
        "PLACE exact receipt failed");
  check(!ledger.objective(opened.objective_id)->complete,
        "successful PLACE incorrectly cleared lifecycle debt");

  auto feed_before = place_after;
  feed_before.actor.inventory[item(Item::WHEAT)] = 1;
  const auto feed =
      propose(ledger, opened.objective_id, replacement, feed_before);
  check(feed.action.op == Op::FEED && feed.action.item == Item::WHEAT,
        "FEED stage did not require wheat");
  stage(ledger, feed);
  auto feed_after = feed_before;
  feed_after.step = 4;
  feed_after.target.fed_today = true;
  feed_after.actor.inventory[item(Item::WHEAT)] = 0;
  check(settle(ledger, feed, feed_after).status ==
            lifecycle::UnitReceiptStatus::Success,
        "FEED exact receipt failed");

  const auto care =
      propose(ledger, opened.objective_id, replacement, feed_after);
  check(care.action.op == Op::CARE, "CARE was not serialized after FEED");
  stage(ledger, care);
  auto care_after = feed_after;
  care_after.step = 5;
  care_after.target.cared_today = true;
  check(settle(ledger, care, care_after).status ==
            lifecycle::UnitReceiptStatus::Success,
        "CARE exact receipt failed");

  auto yield_before = care_after;
  yield_before.step = 100;
  yield_before.day = 4;
  yield_before.target.yield_units = 1;
  const auto harvest =
      propose(ledger, opened.objective_id, replacement, yield_before);
  check(harvest.proposed_stage == lifecycle::Stage::FirstYield &&
            harvest.action.op == Op::HARVEST &&
            harvest.action.item == Item::EGG,
        "mature goose did not propose typed first-yield harvest");
  stage(ledger, harvest);
  check(ledger.objective(opened.objective_id)->stage ==
            lifecycle::Stage::FirstYield,
        "staged first yield is not represented in persistent state");
  auto yield_after = yield_before;
  yield_after.step = 101;
  yield_after.target.yield_units = 0;
  yield_after.actor.inventory[item(Item::EGG)] = 1;
  check(settle(ledger, harvest, yield_after).status ==
                lifecycle::UnitReceiptStatus::Success &&
            ledger.objective(opened.objective_id)->complete,
        "first-yield debt cleared without exact product receipt");
}

void failed_place_receipt_preserves_objective_and_releases_actor_lease() {
  lifecycle::Ledger ledger;
  const auto opened = ledger.open({Item::SHEEP, {1, 1}, 500, "place-fail"});
  acquire(ledger, opened.objective_id, Item::SHEEP, 500);
  const lifecycle::ActorIdentity actor{3, 20};
  lifecycle::TileSnapshot pasture;
  pasture.kind = TileKind::PASTURE;
  auto pickup_before = observation(1, 0, actor, {4, 4}, pasture);
  pickup_before.actor.shed_adjacent = true;
  pickup_before.shed[item(Item::SHEEP)] = 1;
  const auto pickup =
      propose(ledger, opened.objective_id, actor, pickup_before);
  stage(ledger, pickup, actor.actor_id, actor.actor_id + 1);
  auto pickup_after = pickup_before;
  pickup_after.step = 2;
  pickup_after.shed[item(Item::SHEEP)] = 0;
  pickup_after.actor.inventory[item(Item::SHEEP)] = 1;
  settle(ledger, pickup, pickup_after);

  auto place_before = pickup_after;
  place_before.actor.position = {1, 1};
  const auto place = propose(ledger, opened.objective_id, actor, place_before);
  stage(ledger, place, actor.actor_id, actor.actor_id + 1);
  auto failed_after = place_before;
  failed_after.step = 3; // Exact step/token, but no PLACE effect.
  const auto failure = settle(ledger, place, failed_after);
  check(failure.status == lifecycle::UnitReceiptStatus::Failed &&
            ledger.objective(opened.objective_id)->stage ==
                lifecycle::Stage::PlaceOnTarget &&
            !ledger.objective(opened.objective_id)->complete,
        "failed PLACE receipt cleared or advanced the objective");

  const lifecycle::ActorIdentity new_actor{1, 21};
  const auto new_lease = ledger.lease_actor(opened.objective_id, new_actor, 3);
  check(new_lease.has_value(),
        "failed receipt did not release ephemeral actor lease");
  const auto stale =
      ledger.propose_unit(opened.objective_id, place.lease, place_before);
  check(stale.blocked == lifecycle::BlockReason::MissingOrStaleLease,
        "old actor lease token was accepted after failure");
  auto retry_before = place_before;
  retry_before.step = 3;
  retry_before.actor.identity = new_actor;
  const auto retry =
      ledger.propose_unit(opened.objective_id, *new_lease, retry_before);
  check(retry.proposal.has_value(), "new actor could not claim retained PLACE");
  stage(ledger, *retry.proposal, new_actor.actor_id, new_actor.actor_id + 1);
  auto ambiguous_after = retry_before;
  ambiguous_after.step = 4;
  ambiguous_after.actor.inventory[item(Item::SHEEP)] = 0;
  ambiguous_after.target.kind = TileKind::ANIMAL;
  ambiguous_after.target.animal = Item::SHEEP;
  ambiguous_after.target.placed_day = 0;
  const auto ambiguous = ledger.observe_unit(
      {opened.objective_id, 4, new_lease->token + 999, ambiguous_after});
  check(ambiguous.status == lifecycle::UnitReceiptStatus::Ambiguous &&
            ledger.objective(opened.objective_id)->stage ==
                lifecycle::Stage::PlaceOnTarget &&
            ledger.lease_actor(opened.objective_id, {2, 22}, 4).has_value(),
        "wrong-token receipt did not preserve objective and release lease");
}

void different_tiles_can_stage_same_turn_without_cross_actor_pollution() {
  lifecycle::Ledger ledger;
  const auto left = ledger.open({Item::GOOSE, {1, 1}, 300, "left"});
  const auto right = ledger.open({Item::COW, {2, 2}, 400, "right"});
  acquire(ledger, left.objective_id, Item::GOOSE, 300, 0);
  acquire(ledger, right.objective_id, Item::COW, 400, 2);
  lifecycle::TileSnapshot coop;
  coop.kind = TileKind::COOP;
  lifecycle::TileSnapshot pasture;
  pasture.kind = TileKind::PASTURE;
  auto left_before = observation(3, 0, {0, 30}, {4, 4}, coop);
  left_before.actor.shed_adjacent = true;
  left_before.shed[item(Item::GOOSE)] = 1;
  auto right_before = observation(3, 0, {1, 40}, {5, 4}, pasture);
  right_before.actor.shed_adjacent = true;
  right_before.shed[item(Item::COW)] = 1;
  const auto left_pickup = propose(ledger, left.objective_id,
                                   left_before.actor.identity, left_before);
  const auto right_pickup = propose(ledger, right.objective_id,
                                    right_before.actor.identity, right_before);
  stage(ledger, left_pickup, 0, 2);
  stage(ledger, right_pickup, 1, 2);
  auto left_after = left_before;
  left_after.step = 4;
  left_after.shed[item(Item::GOOSE)] = 0;
  left_after.actor.inventory[item(Item::GOOSE)] = 1;
  auto right_after = right_before;
  right_after.step = 4;
  right_after.shed[item(Item::COW)] = 0;
  right_after.actor.inventory[item(Item::COW)] = 1;
  check(settle(ledger, right_pickup, right_after).status ==
                lifecycle::UnitReceiptStatus::Success &&
            settle(ledger, left_pickup, left_after).status ==
                lifecycle::UnitReceiptStatus::Success,
        "independent actor/tile receipts polluted each other");
}

void typed_whole_player_loss_reopens_acquisition_idempotently() {
  lifecycle::Ledger ledger;
  const auto opened = ledger.open({Item::GOOSE, {3, 2}, 300, "lost-goose"});
  acquire(ledger, opened.objective_id, Item::GOOSE, 300);
  check(ledger.reopen_acquisition(
            opened.objective_id, {2, {3, 2}, Item::GOOSE, 1, 0, 0, 0, false}) ==
            lifecycle::ReopenAcquisitionStatus::InventoryStillAvailable,
        "shed-owned animal was incorrectly declared lost");
  check(ledger.reopen_acquisition(
            opened.objective_id, {2, {3, 2}, Item::GOOSE, 0, 1, 0, 0, false}) ==
            lifecycle::ReopenAcquisitionStatus::InventoryStillAvailable,
        "another actor inventory was ignored by loss proof");
  check(ledger.reopen_acquisition(
            opened.objective_id, {2, {3, 2}, Item::GOOSE, 0, 0, 1, 0, false}) ==
            lifecycle::ReopenAcquisitionStatus::InventoryStillAvailable,
        "wrong-board animal was ignored by loss proof");
  check(ledger.reopen_acquisition(
            opened.objective_id, {2, {3, 2}, Item::GOOSE, 0, 0, 1, 1, false}) ==
                lifecycle::ReopenAcquisitionStatus::Reopened &&
            ledger.objective(opened.objective_id)->stage ==
                lifecycle::Stage::Acquire,
        "one unit reserved by another objective did not reopen this debt");
  check(ledger.reopen_acquisition(
            opened.objective_id, {2, {3, 2}, Item::GOOSE, 0, 0, 1, 1, false}) ==
            lifecycle::ReopenAcquisitionStatus::AlreadyOpen,
        "repeated acquisition reopen was not idempotent");
  const auto retry =
      ledger.propose_acquisition(opened.objective_id, {3, 300, 0});
  check(retry && retry->order.quantity == 1,
        "reopened lifecycle did not restore one acquisition debt");
}

} // namespace

int main() try {
  animal_definitions_and_tile_owner_are_explicit();
  zero_fill_cash_recovery_retry_then_inventory_only_handoff();
  exact_goose_lifecycle_reaches_first_yield_only_after_harvest_receipt();
  failed_place_receipt_preserves_objective_and_releases_actor_lease();
  different_tiles_can_stage_same_turn_without_cross_actor_pollution();
  typed_whole_player_loss_reopens_acquisition_idempotently();
  std::cout << "animal lifecycle repair: 6 adversarial fixtures passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "animal lifecycle repair test failure: " << error.what() << '\n';
  return 1;
}
