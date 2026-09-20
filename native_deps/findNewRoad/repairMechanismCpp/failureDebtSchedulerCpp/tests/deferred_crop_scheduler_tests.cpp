#include "deferred_crop_scheduler.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace g001::failure_debt::deferred_crop;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::Position;
using fastkag::TileKind;

constexpr std::uint64_t kMovementHash = 0x9a7c'5512'40bb'12efULL;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

DeferredSource source(int actor, Position tile, int source_step, Action action,
                      int deadline = 23,
                      Item remembered_crop = Item::NONE) {
  return {actor, tile, source_step, source_step / 24, deadline, action,
          remembered_crop, true,
          "fixture:source:" + std::to_string(source_step)};
}

Visit visit(int step, int actor, Position tile, CropSnapshot snapshot,
            Action base = {}, bool slack = true) {
  Visit result;
  result.step = step;
  result.day = step / 24;
  result.actor = actor;
  result.position = tile;
  result.tile = snapshot;
  result.seeds.fill(4);
  result.carried_fertilizer = 2;
  result.base_action = base;
  result.absorbable_slack = slack;
  result.remaining_action_slots = 8;
  result.remaining_moves = 3;
  result.movement_hash = kMovementHash;
  return result;
}

Receipt receipt(const Proposal& proposal, CropSnapshot before,
                CropSnapshot after) {
  Receipt result;
  result.step = proposal.step + 1;
  result.actor = proposal.actor;
  result.debt_id = proposal.debt_id;
  result.obligation_id = proposal.obligation_id;
  result.emitted = proposal.status == ProposalStatus::EmitReplacement
      ? proposal.action : proposal.original;
  result.before = before;
  result.after = after;
  result.movement_hash = proposal.movement_hash;
  result.provenance = "fixture:next-observation";
  return result;
}

void test_default_off_and_source_boundary() {
  DeferredCropScheduler disabled;
  require(!disabled.enqueue(source(0, {1, 1}, 3, {Op::PLANT, Item::CARROT})),
          "default-off seam refuses to own a source");

  DeferredCropScheduler scheduler({true, 24});
  require(!scheduler.enqueue(source(0, {1, 1}, 3, {Op::EAST})),
          "MOVE can never enter deferred crop debt");
  require(!scheduler.enqueue(source(0, {-1, -1}, 3, {Op::WATER})),
          "day-boundary source without a causal tile fails closed");
  require(scheduler.audit(3).rejected_sources == 2,
          "rejected source count remains auditable");
}

void test_actor_tile_day_coalescing() {
  DeferredCropScheduler scheduler({true, 24});
  require(scheduler.enqueue(source(0, {2, 3}, 20, {Op::WATER}, 23,
                                           Item::TOMATO)), "first source");
  require(scheduler.enqueue(source(0, {2, 3}, 21, {Op::WATER}, 23,
                                           Item::TOMATO)), "duplicate source");
  require(scheduler.enqueue(source(0, {2, 3}, 22, {Op::HARVEST}, 23,
                                           Item::TOMATO)), "same tile chain");
  require(scheduler.enqueue(source(0, {2, 3}, 25, {Op::WATER}, 47,
                                           Item::TOMATO)), "next day source");
  const auto open = scheduler.open_debts(30);
  require(open.size() == 2,
          "same actor/tile/day is one debt but a later day is a new debt");
  require(open[0].origin_day == 0 && open[0].obligations.size() == 2 &&
              open[0].obligations[0].coalesced_sources == 2 &&
              open[0].obligations[0].source_steps == std::vector<int>({20, 21}),
          "consecutive equivalent WATER sources coalesce inside tile debt");
  const auto audit = scheduler.audit(30);
  require(audit.sources_enqueued == 4 && audit.sources_coalesced == 1 &&
              audit.overdue_open_obligations == 2,
          "overdue obligations are counted, not erased");
}

void test_weed_plant_water_chain_preserves_move_and_requires_receipts() {
  DeferredCropScheduler scheduler({true, 24});
  constexpr Position tile{2, 2};
  require(scheduler.enqueue(source(0, tile, 20,
                                   {Op::PLANT, Item::STRAWBERRY}, 23)),
          "plant debt enqueued");
  require(scheduler.enqueue(source(0, tile, 21, {Op::WATER}, 23,
                                   Item::STRAWBERRY)),
          "water debt grouped");

  CropSnapshot weed; weed.kind = TileKind::WEED;
  auto blocked = scheduler.propose(
      visit(24, 0, tile, weed, {Op::EAST}, true));
  require(blocked.status == ProposalStatus::Blocked &&
              blocked.reason == "move_is_read_only",
          "a due critical debt does not swallow MOVE");

  auto dig = scheduler.propose(visit(25, 0, tile, weed));
  require(dig.status == ProposalStatus::EmitReplacement &&
              dig.action.op == Op::DIG && dig.synthetic_prerequisite &&
              !dig.consumes_obligation_on_success,
          "weed causes a receipt-bound synthetic DIG before PLANT");
  require(!scheduler.commit(dig, {Op::DIG}, kMovementHash + 1),
          "movement fingerprint mismatch rejects commit");
  require(scheduler.commit(dig, {Op::DIG}, kMovementHash),
          "exact final non-MOVE manifest is staged");

  CropSnapshot still_weed = weed;
  auto failed_dig = receipt(dig, weed, still_weed);
  require(!scheduler.observe_receipt(failed_dig),
          "emitted action equality alone cannot consume DIG");
  require(scheduler.open_debts(26).size() == 1,
          "failed receipt reopens the same debt");

  dig = scheduler.propose(visit(26, 0, tile, weed));
  require(scheduler.commit(dig, dig.action, kMovementHash), "retry staged");
  CropSnapshot empty; empty.kind = TileKind::EMPTY;
  require(scheduler.observe_receipt(receipt(dig, weed, empty)),
          "weed-to-empty observation confirms DIG prerequisite");

  auto plant = scheduler.propose(visit(27, 0, tile, empty));
  require(plant.action.op == Op::PLANT && !plant.synthetic_prerequisite &&
              plant.consumes_obligation_on_success,
          "head PLANT becomes due after confirmed DIG");
  require(scheduler.commit(plant, plant.action, kMovementHash), "plant staged");
  CropSnapshot growing;
  growing.kind = TileKind::PLANT;
  growing.crop = Item::STRAWBERRY;
  growing.planted_day = 1;
  require(scheduler.observe_receipt(receipt(plant, empty, growing)),
          "empty-to-matching-crop confirms PLANT");

  auto water = scheduler.propose(visit(28, 0, tile, growing));
  require(water.action.op == Op::WATER &&
              water.consumes_obligation_on_success,
          "next grouped obligation executes at the same later visit");
  require(scheduler.commit(water, water.action, kMovementHash), "water staged");
  auto watered = growing;
  watered.watered_today = true;
  require(scheduler.observe_receipt(receipt(water, growing, watered)),
          "watered_today confirms WATER");
  require(scheduler.open_debts(29).empty(),
          "tile debt completes only after every receipt");

  const auto audit = scheduler.audit(29);
  require(audit.receipt_failures == 1 && audit.receipt_successes == 3 &&
              audit.completed_obligations == 2 &&
              audit.completed_tile_debts == 1,
          "synthetic prerequisite and source completions are distinguished");
}

void test_critical_base_action_is_not_absorbed() {
  DeferredCropScheduler scheduler({true, 24});
  constexpr Position tile{1, 3};
  require(scheduler.enqueue(source(0, tile, 12, {Op::WATER}, 20,
                                   Item::CARROT)), "water debt");
  CropSnapshot growing;
  growing.kind = TileKind::PLANT;
  growing.crop = Item::CARROT;
  auto current = visit(30, 0, tile, growing, {Op::FERTILIZE}, true);
  current.base_critical = true;
  const auto proposal = scheduler.propose(current);
  require(proposal.status == ProposalStatus::Blocked &&
              proposal.reason ==
                  "no_certified_nonmove_slack_after_move_reservation",
          "critical production action is never swallowed for repair");
  require(scheduler.audit(100).overdue_open_obligations == 1,
          "blocked overdue work remains visible and open");
}

void test_harvest_lifecycle_rebuild_and_day_end_water_receipt() {
  DeferredCropScheduler scheduler({true, 24});
  constexpr Position tile{3, 3};
  require(scheduler.enqueue(source(1, tile, 22, {Op::HARVEST}, 23,
                                   Item::TOMATO)), "harvest debt");
  CropSnapshot empty; empty.kind = TileKind::EMPTY;
  auto plant = scheduler.propose(visit(24, 1, tile, empty));
  require(plant.action.op == Op::PLANT && plant.synthetic_prerequisite,
          "lost crop HARVEST debt first restores remembered crop");
  require(scheduler.commit(plant, plant.action, kMovementHash), "plant staged");
  CropSnapshot growing;
  growing.kind = TileKind::PLANT;
  growing.crop = Item::TOMATO;
  growing.planted_day = 1;
  require(scheduler.observe_receipt(receipt(plant, empty, growing)),
          "restored crop observed");

  auto water = scheduler.propose(visit(47, 1, tile, growing));
  require(water.action.op == Op::WATER && water.synthetic_prerequisite,
          "immature harvest debt maintains restored crop");
  require(scheduler.commit(water, water.action, kMovementHash), "water staged");
  auto next_day = growing;
  auto water_receipt = receipt(water, growing, next_day);
  water_receipt.day_end_water_effect_lower_bound = true;
  require(scheduler.observe_receipt(water_receipt),
          "day-end WATER uses deterministic effect lower bound after reset");

  auto mature = growing;
  mature.harvest_legal = true;
  mature.yield_units = 4;
  auto harvest = scheduler.propose(visit(216, 1, tile, mature));
  require(harvest.action.op == Op::HARVEST &&
              !harvest.synthetic_prerequisite,
          "original HARVEST remains open until observed maturity");
  require(scheduler.commit(harvest, harvest.action, kMovementHash),
          "harvest staged");
  auto harvested = mature;
  harvested.kind = TileKind::EMPTY;
  harvested.crop = Item::NONE;
  harvested.yield_units = 0;
  auto harvest_receipt = receipt(harvest, mature, harvested);
  harvest_receipt.crop_inventory_delta = 4;
  require(scheduler.observe_receipt(harvest_receipt),
          "yield/inventory effect confirms HARVEST");
  require(scheduler.open_debts(217).empty(),
          "overdue lifecycle completes rather than expiring");
}

void test_existing_equivalent_keeps_route_action() {
  DeferredCropScheduler scheduler({true, 24});
  constexpr Position tile{4, 1};
  require(scheduler.enqueue(source(0, tile, 10,
                                   {Op::PLANT, Item::CARROT}, 20)),
          "plant debt");
  CropSnapshot planted;
  planted.kind = TileKind::PLANT;
  planted.crop = Item::CARROT;
  const auto proposal = scheduler.propose(
      visit(30, 0, tile, planted, {Op::WEST}, false));
  require(proposal.status == ProposalStatus::ExistingEquivalent &&
              proposal.original.op == Op::WEST &&
              proposal.action.op == Op::WEST,
          "observed equivalent crop preserves even a current MOVE byte");
  require(scheduler.commit(proposal, {Op::WEST}, kMovementHash),
          "unchanged route manifest staged for equivalent receipt");
  require(scheduler.observe_receipt(receipt(proposal, planted, planted)),
          "equivalent postcondition receipt consumes PLANT debt");
  require(scheduler.open_debts(31).empty(), "equivalent debt completed");
}

}  // namespace

int main() {
  test_default_off_and_source_boundary();
  test_actor_tile_day_coalescing();
  test_weed_plant_water_chain_preserves_move_and_requires_receipts();
  test_critical_base_action_is_not_absorbed();
  test_harvest_lifecycle_rebuild_and_day_end_water_receipt();
  test_existing_equivalent_keeps_route_action();
  std::cout << "deferred_crop_scheduler_tests: PASS\n";
}
