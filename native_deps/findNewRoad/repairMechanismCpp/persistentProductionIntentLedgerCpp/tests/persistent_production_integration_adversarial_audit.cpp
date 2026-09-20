#include "persistent_production_intent_ledger.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace persistent = g001::persistent_production;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::size_t item(Item value) {
  return static_cast<std::size_t>(static_cast<int>(value));
}

persistent::ObjectiveSpec crop_spec(int origin_day,
                                    persistent::TileKey tile,
                                    Item crop, int deadline_day) {
  static constexpr std::array<int, fastkag::N_CROPS> costs{
      10, 20, 50, 100, 80};
  return {{origin_day, tile}, persistent::Kind::Crop, crop, 1,
          costs[item(crop)], deadline_day, true, 100,
          "integration-adversarial-audit"};
}

persistent::Observation observation(
    int step, int day, persistent::TileKey tile,
    persistent::ActorIdentity actor, TileKind kind) {
  persistent::Observation result;
  result.step = step;
  result.day = day;
  result.target_key = tile;
  result.target.kind = kind;
  result.actor.identity = actor;
  result.actor.position = tile;
  return result;
}

void complete_seed_purchase(persistent::Ledger& ledger,
                            std::uint64_t objective, Item crop) {
  const auto proposal = ledger.propose_purchase(objective, {0, 1000, 0});
  check(proposal.has_value(), "seed purchase proposal absent");
  const std::array<Action, 1> market{proposal->order};
  check(ledger.stage_purchase(objective, *proposal,
                              {0, 0, market, 0, 1000, 0}) ==
            g001::purchase_recovery::StageStatus::Selected,
        "seed purchase staging failed");
  const std::array<std::int32_t, 1> fills{1};
  g001::purchase_recovery::ReceiptObservation receipt;
  receipt.step = 1;
  receipt.slot_fills = fills;
  receipt.seeds_after[item(crop)] = 1;
  const auto settled = ledger.observe_purchases(receipt);
  check(settled.size() == 1 &&
            ledger.objective(objective)->purchase_complete,
        "seed purchase did not hand off to production intent");
}

persistent::ReadyTransition ready(
    persistent::Ledger& ledger, std::uint64_t objective,
    persistent::ActorIdentity actor, const persistent::Observation& before,
    persistent::ResourceSnapshot resources = {}) {
  const auto lease = ledger.lease_actor(objective, actor, before.step);
  check(lease.has_value(), "actor lease absent");
  const std::array<persistent::ReadyRequest, 1> request{
      persistent::ReadyRequest{objective, *lease, before}};
  const auto transitions = ledger.ready_transitions(request, resources);
  check(transitions.size() == 1, "ready transition absent");
  return transitions.front();
}

void lower_slot_prefix_is_mandatory_authority() {
  persistent::Ledger ledger;
  const auto opened = ledger.open(crop_spec(0, {2, 2}, Item::WHEAT, 2));
  auto before = observation(8, 0, {2, 2}, {1, 7}, TileKind::PLANT);
  before.target.crop = Item::WHEAT;
  const auto water = ready(ledger, opened.objective_id, {1, 7}, before);
  const std::array<Action, 2> units{Action{}, water.action};
  const persistent::PrefixAuthority unbound_prefix{};
  const persistent::UnitFinalSelection unbound{
      before.step, 1, units, before, unbound_prefix, before.seeds, true};
  check(ledger.stage_transition(water, unbound) ==
            persistent::UnitStageStatus::BeforeWitnessMismatch,
        "actor1 accepted a whole-phase pre-state as a lower-prefix witness");
  auto tampered = persistent::make_prefix_authority(
      before.step, 1, 3, persistent::observation_fingerprint(before), units,
      before);
  tampered.lower_prefix_fingerprint ^= 1;
  check(ledger.stage_transition(
            water, {before.step, 1, units, before, tampered, before.seeds,
                    true}) ==
            persistent::UnitStageStatus::BeforeWitnessMismatch,
        "tampered lower-prefix hash was accepted");
}

void receipt_must_bind_the_staged_prefix_authority() {
  persistent::Ledger ledger;
  const auto opened = ledger.open(crop_spec(0, {2, 3}, Item::WHEAT, 2));
  auto before = observation(9, 0, {2, 3}, {0, 8}, TileKind::PLANT);
  before.target.crop = Item::WHEAT;
  const auto water = ready(ledger, opened.objective_id, {0, 8}, before);
  const std::array<Action, 1> units{water.action};
  const auto prefix = persistent::make_prefix_authority(
      before.step, 0, 4, persistent::observation_fingerprint(before), units,
      before);
  check(ledger.stage_transition(
            water, {before.step, 0, units, before, prefix, before.seeds,
                    true}) == persistent::UnitStageStatus::Selected,
        "valid prefix authority was not staged");
  auto after = before;
  ++after.step;
  after.target.watered_today = true;
  const auto settlement = ledger.observe_transition(
      opened.objective_id, water.lease.token,
      persistent::prefix_authority_fingerprint(prefix) ^ 1U, after);
  check(settlement.status == persistent::UnitReceiptStatus::Ambiguous &&
            settlement.objective_status == persistent::Status::Open &&
            ledger.objective(opened.objective_id)->status ==
                persistent::Status::Open,
        "mismatched receipt authority retired plot debt");
}

void whole_phase_plant_gate_prevents_shared_seed_false_credit() {
  persistent::Ledger ledger;
  const auto opened = ledger.open(crop_spec(0, {3, 3}, Item::TOMATO, 2));
  complete_seed_purchase(ledger, opened.objective_id, Item::TOMATO);
  auto before = observation(2, 0, {3, 3}, {1, 9}, TileKind::EMPTY);
  before.seeds[item(Item::TOMATO)] = 1;
  persistent::ResourceSnapshot resources;
  resources.seeds[item(Item::TOMATO)] = 1;
  const auto plant = ready(ledger, opened.objective_id, {1, 9}, before,
                           resources);
  const std::array<Action, 2> units{
      Action{Op::PLANT, Item::TOMATO, 1}, plant.action};
  const auto prefix = persistent::make_prefix_authority(
      before.step, 1, 5, persistent::observation_fingerprint(before),
      units, before);
  persistent::UnitFinalSelection selection{
      before.step, 1, units, before, prefix, before.seeds, true};
  check(ledger.stage_transition(plant, selection) ==
            persistent::UnitStageStatus::StaleTransition,
        "two same-crop PLANTs passed the simulator's all-or-none seed gate");
}

void hour23_full_step_receipt_is_ambiguous_and_keeps_debt() {
  fastkag::Simulator simulator({}, 123);
  const std::array<fastkag::PlayerAction, 2> pass{};
  for (int step = 0; step < 23; ++step) simulator.step(pass);
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  const auto native_position = farm.farmer;
  auto& native_tile = farm.tiles[static_cast<std::size_t>(
      native_position.y * simulator.config().board_size + native_position.x)];
  native_tile = {};
  native_tile.kind = TileKind::PLANT;
  native_tile.crop = Item::CARROT;

  persistent::Ledger ledger;
  const auto opened = ledger.open(crop_spec(0, {4, 4}, Item::CARROT, 0));
  auto before = observation(23, 0, {4, 4}, {0, 11}, TileKind::PLANT);
  before.target.crop = Item::CARROT;
  const auto water = ready(ledger, opened.objective_id, {0, 11}, before);
  const std::array<Action, 1> units{water.action};
  const auto prefix = persistent::make_prefix_authority(
      23, 0, 8, persistent::observation_fingerprint(before), units, before);
  check(ledger.stage_transition(
            water, {23, 0, units, before, prefix, before.seeds, true}) ==
            persistent::UnitStageStatus::Selected,
        "hour23 WATER was not staged");

  std::array<fastkag::PlayerAction, 2> action{};
  action[0].units = {water.action};
  simulator.step(action);
  check(simulator.step_count() == 24 && simulator.day() == 1 &&
            native_tile.kind == TileKind::PLANT &&
            native_tile.crop == Item::CARROT &&
            !native_tile.watered_today,
        "Simulator differential did not expose midnight WATER reset");
  auto after = before;
  after.step = 24;
  after.day = 1;
  after.target = native_tile;
  const auto receipt = ledger.observe_transition(
      opened.objective_id, water.lease.token,
      persistent::prefix_authority_fingerprint(prefix), after);
  check(receipt.status == persistent::UnitReceiptStatus::Ambiguous &&
            receipt.objective_status == persistent::Status::Open &&
            ledger.active_objectives().size() == 1 &&
            ledger.objective(opened.objective_id)->crop_stage ==
                persistent::CropStage::NeedWater,
        "hour23 full-step observation falsely completed or dropped WATER debt");
}

void stale_generation_cannot_stage_after_rebinding() {
  persistent::Ledger ledger;
  const auto opened = ledger.open(crop_spec(0, {1, 4}, Item::MELON, 2));
  auto before = observation(4, 0, {1, 4}, {2, 3}, TileKind::PLANT);
  before.target.crop = Item::MELON;
  const auto old = ready(ledger, opened.objective_id, {2, 3}, before);
  check(ledger.release_actor(old.lease), "old generation lease not released");
  auto replacement_before = before;
  replacement_before.actor.identity = {2, 4};
  const auto replacement = ready(ledger, opened.objective_id, {2, 4},
                                 replacement_before);
  const std::array<Action, 3> units{Action{}, Action{}, old.action};
  const auto stale_prefix = persistent::make_prefix_authority(
      before.step, 2, 9, persistent::observation_fingerprint(before), units,
      before);
  check(ledger.stage_transition(
            old, {4, 2, units, before, stale_prefix, before.seeds, true}) ==
            persistent::UnitStageStatus::MissingOrStaleLease,
        "stale actor generation staged after replacement binding");
  check(replacement.lease.actor.generation == 4,
        "replacement generation was not retained");
}

}  // namespace

int main() try {
  lower_slot_prefix_is_mandatory_authority();
  receipt_must_bind_the_staged_prefix_authority();
  whole_phase_plant_gate_prevents_shared_seed_false_credit();
  hour23_full_step_receipt_is_ambiguous_and_keeps_debt();
  stale_generation_cannot_stage_after_rebinding();
  std::cout << "persistent production integration audit: 5 groups passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
