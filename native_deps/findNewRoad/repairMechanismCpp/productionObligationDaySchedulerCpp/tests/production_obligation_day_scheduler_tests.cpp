#include "production_obligation_day_scheduler.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace day = g001::obligation_day;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Simulator weed_state(int seeds) {
  fastkag::Config config;
  config.weed_spawn_chance = 1;
  Simulator simulator(config, 77881);
  for (int step = 0; step < 24; ++step) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({});
    actions[1].units.push_back({});
    if (step == 0 && seeds > 0) {
      actions[0].market.push_back({Op::BUY_SEED, Item::WHEAT, seeds});
    }
    simulator.step(actions);
  }
  return simulator;
}

Position origin(const Simulator& simulator) {
  return simulator.farms()[0].farmer;
}

day::ProductionObligation crop(std::uint64_t id, Position position,
                               bool must_finish = true) {
  return {id,
          0,
          position,
          day::GoalKind::CropReady,
          Item::WHEAT,
          1,
          {},
          {},
          24,
          47,
          5,
          must_finish,
          true};
}

day::DayPlanRequest base_request(const Simulator& simulator) {
  return {&simulator, 0, 9001, {}, {crop(1, origin(simulator))}};
}

void rehash(day::DayScheduleCertificate& certificate) {
  certificate.content_hash = day::day_schedule_certificate_hash(certificate);
}

void recompute_post_fingerprints(const day::DayPlanRequest& request,
                                 day::DayScheduleCertificate& certificate) {
  Simulator state = *request.day_start;
  for (auto& slot : certificate.slots) {
    std::array<PlayerAction, 2> joint;
    joint[request.player].units = slot.actions;
    state = state.preview_unit_phase(joint);
    slot.focal_post_fingerprint =
        g001::production_suffix::focal_unit_state_fingerprint(
            state, request.player, slot.step);
  }
}

void set_pass(day::DaySlotProof& slot, int actor) {
  slot.actions[actor] = {};
  slot.sources[actor] = {actor, -1, {}};
  slot.obligation_ids[actor] = 0;
}

day::DayPlanRequest feed_request(Simulator& simulator) {
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  auto& private_state =
      const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
  const auto position = origin(simulator);
  auto& animal = farm.tiles[position.y * simulator.config().board_size +
                            position.x];
  animal = Tile{};
  animal.kind = TileKind::ANIMAL;
  animal.animal = Item::GOOSE;
  private_state.shed[0] = 1;
  Position west{static_cast<std::int16_t>(position.x - 1), position.y};
  farm.tiles[west.y * simulator.config().board_size + west.x].kind =
      TileKind::WEED;
  day::ProductionObligation pickup{
      10, 0, position, day::GoalKind::Pickup, Item::WHEAT, 1, {},
      {Item::WHEAT, 0, 1, 0}, 24, 24, 100, true, false};
  day::ProductionObligation feed{
      11, 0, position, day::GoalKind::Feed, Item::WHEAT, 1, {10},
      {Item::WHEAT, 0, 0, 1}, 24, 25, 100, true, false};
  return {&simulator,
          0,
          9002,
          {{0, 25, {Op::WEST, Item::NONE, 1}}},
          {pickup, feed, crop(12, west)}};
}

void weed_crop_keeps_two_moves() {
  auto simulator = weed_state(1);
  auto request = base_request(simulator);
  request.moves = {{0, 25, {Op::WEST, Item::NONE, 1}},
                   {0, 28, {Op::EAST, Item::NONE, 1}}};
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.debts.empty(), "crop plan failed");
  check(plan.manifest[0][0].op == Op::DIG &&
            plan.manifest[0][1].op == Op::PLANT &&
            plan.manifest[0][2].op == Op::WATER,
        "crop lifecycle wrong");
  check(plan.manifest[0][3].op == Op::WEST &&
            plan.sources[0][3].source_step == 25 &&
            plan.manifest[0][4].op == Op::EAST &&
            plan.sources[0][4].source_step == 28,
        "MOVE replay wrong");
  check(plan.certificate->obligation_statuses.size() == 1 &&
            plan.certificate->obligation_statuses[0].transition_steps ==
                std::vector<int>({24, 25, 26}) &&
            day::verify_day_schedule(request, *plan.certificate).valid,
        "rich day certificate invalid");
}

void feed_deadline_beats_crop_without_swallow() {
  auto simulator = weed_state(1);
  auto request = feed_request(simulator);
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.manifest[0][0].op == Op::PICKUP &&
            plan.manifest[0][1].op == Op::FEED,
        "baseline FEED deadline swallowed");
  check(plan.manifest[0][2].op == Op::WEST &&
            plan.sources[0][2].source_step == 25,
        "FEED lost MOVE");
  check(std::find(plan.completed.begin(), plan.completed.end(), 11) !=
                plan.completed.end() &&
            std::find(plan.completed.begin(), plan.completed.end(), 12) !=
                plan.completed.end(),
        "feed/crop not both completed");
  check(day::verify_day_schedule(request, *plan.certificate).valid,
        "feed certificate invalid");
}

void seed_receipt_unlocks_next_plan() {
  auto without_seed = weed_state(0);
  auto blocked_request = base_request(without_seed);
  const auto blocked = day::plan_day(blocked_request);
  check(blocked.planned() && blocked.debts.size() == 1 &&
            blocked.debts[0].reason == day::DebtReason::BlockedOnReceipt,
        "missing seed not blocked on receipt");
  check(day::verify_day_schedule(blocked_request, *blocked.certificate).valid,
        "blocked receipt certificate invalid");

  auto with_seed = weed_state(1);
  auto unlocked_request = base_request(with_seed);
  unlocked_request.issuer_generation = 9003;
  const auto unlocked = day::plan_day(unlocked_request);
  check(unlocked.planned() && unlocked.debts.empty(),
        "observed seed receipt did not unlock next plan");
}

void successful_buy_seed_reissues_only_remaining_crop_suffix() {
  auto simulator = weed_state(0);
  const auto start = origin(simulator);
  const Position west{static_cast<std::int16_t>(start.x - 1), start.y};
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  auto& target = farm.tiles[west.y * simulator.config().board_size + west.x];
  target = Tile{};
  target.kind = TileKind::WEED;

  day::DayPlanRequest before{
      &simulator,
      0,
      9300,
      {{0, 24, {Op::WEST, Item::NONE, 1}},
       {0, 28, {Op::EAST, Item::NONE, 1}}},
      {crop(300, west)}};
  before.obligations[0].source_step = 24;
  const auto blocked = day::plan_day(before);
  check(blocked.planned() && blocked.debts.size() == 1 &&
            blocked.debts[0].reason == day::DebtReason::BlockedOnReceipt,
        "pre-purchase crop was not explicit receipt debt");

  const double money_before = simulator.farms()[0].money;
  std::array<PlayerAction, 2> submitted;
  submitted[0].units.push_back({Op::WEST, Item::NONE, 1});
  submitted[0].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  submitted[1].units.push_back({});
  simulator.step(submitted);
  check(simulator.step_count() == 25 && simulator.hour() == 1 &&
            simulator.last_market_fills()[0] == std::vector<std::int32_t>({1}) &&
            simulator.privates()[0].seeds[static_cast<int>(Item::WHEAT)] == 1 &&
            simulator.farms()[0].money < money_before &&
            simulator.farms()[0].farmer.x == west.x &&
            simulator.farms()[0].farmer.y == west.y,
        "real BUY_SEED/prefix MOVE did not settle");

  day::DayPlanRequest remaining{&simulator,
                                0,
                                9300,
                                {{0, 28, {Op::EAST, Item::NONE, 1}}},
                                {crop(300, west)},
                                1};
  remaining.obligations[0].source_step = 24;
  const auto plan = day::plan_remaining_day(remaining);
  check(plan.planned() && plan.debts.empty() &&
            plan.manifest.size() == 1 && plan.manifest[0].size() == 23 &&
            plan.certificate->start_step == 25 &&
            plan.certificate->slots.size() == 23 &&
            plan.certificate->slots.front().step == 25,
        "post-purchase remaining-day envelope wrong");
  check(plan.manifest[0][0].op == Op::DIG &&
            plan.manifest[0][1].op == Op::PLANT &&
            plan.manifest[0][2].op == Op::WATER &&
            plan.manifest[0][3].op == Op::EAST &&
            plan.sources[0][3].source_step == 28 &&
            plan.certificate->move_replays.size() == 1 &&
            plan.certificate->move_replays[0].source_step == 28 &&
            plan.certificate->obligation_statuses[0].transition_steps ==
                std::vector<int>({25, 26, 27}),
        "BUY_SEED did not unlock DIG/PLANT/WATER plus exact suffix MOVE");
  const auto verified =
      day::verify_remaining_day_schedule(remaining, *plan.certificate);
  check(verified.valid && verified.checked_slots == 23,
        "remaining-day certificate failed canonical verification");
  check(!day::verify_remaining_day_schedule(remaining, *blocked.certificate)
             .valid,
        "pre-purchase certificate survived post-purchase reissue boundary");
  check(day::plan_day(remaining).reject == day::PlanReject::NotDayStart,
        "day-start entry accepted a remaining-day request");

  auto wrong_tick = remaining;
  wrong_tick.start_tick = 0;
  check(day::plan_remaining_day(wrong_tick).reject ==
            day::PlanReject::NotDayStart,
        "remaining-day request accepted the wrong start tick");
  auto wrong_start = *plan.certificate;
  ++wrong_start.start_step;
  rehash(wrong_start);
  check(!day::verify_remaining_day_schedule(remaining, wrong_start).valid,
        "tampered remaining-day start step was accepted");
}

void remaining_suffix_keeps_delayed_move() {
  auto simulator = weed_state(0);
  const std::vector<day::MoveSourceToken> originals{
      {0, 24, {Op::WEST, Item::NONE, 1}},
      {0, 28, {Op::EAST, Item::NONE, 1}}};
  std::array<PlayerAction, 2> pass;
  pass[0].units.push_back({});
  pass[1].units.push_back({});
  simulator.step(pass);
  day::DayPlanRequest delayed{&simulator, 0, 9302, originals, {}, 1};
  const auto plan = day::plan_remaining_day(delayed);
  check(plan.planned() && plan.manifest[0].size() == 23 &&
            plan.manifest[0][0].op == Op::WEST &&
            plan.sources[0][0].source_step == 24 &&
            plan.certificate->move_replays.size() == 2 &&
            plan.certificate->move_replays[0].source_step == 24 &&
            plan.certificate->move_replays[0].emitted_step == 25 &&
            day::verify_remaining_day_schedule(delayed, *plan.certificate)
                .valid,
        "unemitted past-source MOVE was not carried into the suffix");
}

void capacity_debt_preserves_all_moves() {
  auto simulator = weed_state(1);
  auto request = base_request(simulator);
  request.obligations[0].must_finish_today = false;
  for (int tick = 0; tick < 24; ++tick) {
    request.moves.push_back(
        {0, 24 + tick,
         {tick % 2 ? Op::EAST : Op::WEST, Item::NONE, 1}});
  }
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.debts.size() == 1 &&
            plan.debts[0].reason == day::DebtReason::Capacity,
        "capacity debt missing");
  for (int tick = 0; tick < 24; ++tick) {
    check(plan.sources[0][tick].source_step == 24 + tick &&
              plan.manifest[0][tick].op == request.moves[tick].action.op,
          "capacity plan changed MOVE");
  }
  check(day::verify_day_schedule(request, *plan.certificate).valid,
        "capacity certificate invalid");
}

void two_actor_shared_shed_conflict() {
  auto simulator = weed_state(0);
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  auto& private_state =
      const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
  farm.hands.push_back(farm.farmer);
  private_state.inventories.push_back({});
  private_state.inventory_order.push_back({});
  private_state.shed[0] = 1;
  const auto position = origin(simulator);
  day::ProductionObligation first{
      20, 0, position, day::GoalKind::Pickup, Item::WHEAT, 1, {},
      {Item::WHEAT, 0, 1, 0}, 24, 25, 10, true, false};
  auto second = first;
  second.id = 21;
  second.actor = 1;
  day::DayPlanRequest request{&simulator, 0, 9004, {}, {first, second}};
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.completed.size() == 1 &&
            plan.debts.size() == 1 &&
            plan.debts[0].reason == day::DebtReason::BlockedOnReceipt,
        "shared shed/wheat conflict not serialized");
  check(day::verify_day_schedule(request, *plan.certificate).valid,
        "two actor certificate invalid");
}

void hour23_and_crossday_reject() {
  auto hour23 = weed_state(1);
  for (int step = 0; step < 23; ++step) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({});
    actions[1].units.push_back({});
    hour23.step(actions);
  }
  auto request = base_request(hour23);
  check(day::plan_day(request).reject == day::PlanReject::NotDayStart,
        "hour23 accepted");

  auto simulator = weed_state(1);
  auto crossday = base_request(simulator);
  crossday.moves = {{0, 48, {Op::WEST, Item::NONE, 1}}};
  check(day::plan_day(crossday).reject == day::PlanReject::InvalidMoveToken,
        "cross-day MOVE accepted");
}

void tamper_all_pass_swallowing_and_unproved_debt() {
  auto simulator = weed_state(1);
  auto request = base_request(simulator);
  const auto plan = day::plan_day(request);

  auto swallowed = *plan.certificate;
  for (auto& slot : swallowed.slots) {
    set_pass(slot, 0);
  }
  recompute_post_fingerprints(request, swallowed);
  rehash(swallowed);
  check(!day::verify_day_schedule(request, swallowed).valid,
        "all-PASS swallowed a mandatory obligation");

  auto fake_capacity = swallowed;
  fake_capacity.obligation_statuses[0].disposition =
      day::ObligationDisposition::CapacityDebt;
  fake_capacity.obligation_statuses[0].assigned_actor = -1;
  fake_capacity.obligation_statuses[0].remaining_transitions = 3;
  fake_capacity.obligation_statuses[0].transition_steps.clear();
  rehash(fake_capacity);
  check(!day::verify_day_schedule(request, fake_capacity).valid,
        "unproved must-finish capacity debt accepted");
}

void tamper_fake_complete_and_hash() {
  auto simulator = weed_state(0);
  auto request = base_request(simulator);
  const auto plan = day::plan_day(request);
  auto fake_complete = *plan.certificate;
  fake_complete.obligation_statuses[0].disposition =
      day::ObligationDisposition::Completed;
  fake_complete.obligation_statuses[0].remaining_transitions = 0;
  rehash(fake_complete);
  check(!day::verify_day_schedule(request, fake_complete).valid,
        "fake Completed status accepted");

  auto hash_tamper = *plan.certificate;
  hash_tamper.content_hash ^= 0x55ULL;
  check(!day::verify_day_schedule(request, hash_tamper).valid,
        "certificate hash tamper accepted");
}

void tamper_unknown_and_duplicate_obligation_id() {
  auto simulator = weed_state(1);
  auto request = feed_request(simulator);
  const auto plan = day::plan_day(request);

  auto unknown = *plan.certificate;
  unknown.slots[0].obligation_ids[0] = 999999;
  rehash(unknown);
  check(!day::verify_day_schedule(request, unknown).valid,
        "unknown obligation id accepted");

  auto duplicate = *plan.certificate;
  duplicate.obligation_statuses[1].obligation_id =
      duplicate.obligation_statuses[0].obligation_id;
  rehash(duplicate);
  check(!day::verify_day_schedule(request, duplicate).valid,
        "duplicate obligation status accepted");
}

void tamper_wrong_actor_tile_and_action() {
  auto simulator = weed_state(1);
  auto request = base_request(simulator);
  const auto plan = day::plan_day(request);
  auto wrong_action = *plan.certificate;
  wrong_action.slots[0].actions[0] = {Op::CARE, Item::GOOSE, 1};
  wrong_action.slots[0].sources[0].source_action =
      wrong_action.slots[0].actions[0];
  recompute_post_fingerprints(request, wrong_action);
  rehash(wrong_action);
  check(!day::verify_day_schedule(request, wrong_action).valid,
        "wrong goal action accepted");

  auto two_actor = weed_state(1);
  auto& farm = const_cast<fastkag::Farm&>(two_actor.farms()[0]);
  auto& private_state =
      const_cast<fastkag::PrivateState&>(two_actor.privates()[0]);
  farm.hands.push_back(farm.farmer);
  private_state.inventories.push_back({});
  private_state.inventory_order.push_back({});
  auto actor_request = base_request(two_actor);
  actor_request.issuer_generation = 9010;
  const auto actor_plan = day::plan_day(actor_request);
  auto wrong_actor = *actor_plan.certificate;
  wrong_actor.slots[0].actions[1] = wrong_actor.slots[0].actions[0];
  wrong_actor.slots[0].sources[1] = {1, -1, wrong_actor.slots[0].actions[1]};
  wrong_actor.slots[0].obligation_ids[1] = 1;
  set_pass(wrong_actor.slots[0], 0);
  recompute_post_fingerprints(actor_request, wrong_actor);
  rehash(wrong_actor);
  check(!day::verify_day_schedule(actor_request, wrong_actor).valid,
        "wrong fixed actor accepted");

  auto tile_request = base_request(simulator);
  Position west{static_cast<std::int16_t>(origin(simulator).x - 1),
                origin(simulator).y};
  tile_request.obligations.push_back(crop(2, west));
  tile_request.issuer_generation = 9011;
  const auto tile_plan = day::plan_day(tile_request);
  auto wrong_tile = *tile_plan.certificate;
  wrong_tile.slots[0].obligation_ids[0] = 2;
  rehash(wrong_tile);
  check(!day::verify_day_schedule(tile_request, wrong_tile).valid,
        "obligation at wrong tile accepted");
}

void tamper_dependency_inversion() {
  auto simulator = weed_state(1);
  auto request = feed_request(simulator);
  const auto plan = day::plan_day(request);
  auto inverted = *plan.certificate;
  inverted.slots[0].actions[0] = {Op::FEED, Item::WHEAT, 1};
  inverted.slots[0].sources[0] = {0, -1, inverted.slots[0].actions[0]};
  inverted.slots[0].obligation_ids[0] = 11;
  recompute_post_fingerprints(request, inverted);
  rehash(inverted);
  check(!day::verify_day_schedule(request, inverted).valid,
        "dependency inversion accepted");
}

void tamper_before_earliest_and_after_deadline() {
  auto simulator = weed_state(1);
  auto early_request = base_request(simulator);
  early_request.obligations[0].earliest_step = 25;
  early_request.issuer_generation = 9012;
  const auto early_plan = day::plan_day(early_request);
  auto before_earliest = *early_plan.certificate;
  before_earliest.slots[0].actions[0] = before_earliest.slots[1].actions[0];
  before_earliest.slots[0].sources[0] =
      {0, -1, before_earliest.slots[0].actions[0]};
  before_earliest.slots[0].obligation_ids[0] = 1;
  set_pass(before_earliest.slots[1], 0);
  recompute_post_fingerprints(early_request, before_earliest);
  rehash(before_earliest);
  check(!day::verify_day_schedule(early_request, before_earliest).valid,
        "action before earliest accepted");

  auto deadline_state = weed_state(0);
  auto& private_state =
      const_cast<fastkag::PrivateState&>(deadline_state.privates()[0]);
  private_state.shed[0] = 1;
  const auto position = origin(deadline_state);
  day::ProductionObligation pickup{
      30, 0, position, day::GoalKind::Pickup, Item::WHEAT, 1, {},
      {Item::WHEAT, 0, 1, 0}, 24, 24, 100, true, false};
  day::DayPlanRequest deadline_request{&deadline_state, 0, 9013, {}, {pickup}};
  const auto deadline_plan = day::plan_day(deadline_request);
  auto after_deadline = *deadline_plan.certificate;
  after_deadline.slots[1].actions[0] = after_deadline.slots[0].actions[0];
  after_deadline.slots[1].sources[0] =
      {0, -1, after_deadline.slots[1].actions[0]};
  after_deadline.slots[1].obligation_ids[0] = 30;
  set_pass(after_deadline.slots[0], 0);
  recompute_post_fingerprints(deadline_request, after_deadline);
  rehash(after_deadline);
  check(!day::verify_day_schedule(deadline_request, after_deadline).valid,
        "action after deadline accepted");
}

void seed970017_day7_speculative_actor_binding_regression() {
  fastkag::Simulator simulator({}, 970017);
  for (int step = 0; step < 7 * 24; ++step) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({});
    actions[1].units.push_back({});
    simulator.step(actions);
  }
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  auto& private_state =
      const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
  farm.farmer = {0, 0};
  private_state.inventories[0][static_cast<int>(Item::GOOSE)] = 1;
  day::ProductionObligation place{
      63571038216003ULL,
      0,
      farm.farmer,
      day::GoalKind::Place,
      Item::GOOSE,
      1,
      {},
      {Item::GOOSE, 0, 0, 1},
      168,
      191,
      995,
      true,
      false};
  day::DayPlanRequest request{&simulator, 0, 63571038216ULL, {}, {place}};

  const auto unchecked = day::internal::plan_day_unchecked(request);
  check(unchecked.planned(), "real regression fixture did not produce a plan");
  check(unchecked.certificate->obligation_statuses.size() == 1 &&
            unchecked.certificate->obligation_statuses[0].assigned_actor ==
                -1 &&
            unchecked.certificate->obligation_statuses[0].transition_steps
                .empty(),
        "failed PICKUP retained its speculative actor binding");
  check(day::verify_day_schedule(request, *unchecked.certificate).valid,
        "seed970017/day7-style certificate is not self-consistent");
  check(day::plan_day(request).planned(),
        "public fail-closed planner rejected the repaired certificate");

  auto legacy_bug = *unchecked.certificate;
  legacy_bug.obligation_statuses[0].assigned_actor = 0;
  rehash(legacy_bug);
  const auto diagnostic = day::verify_day_schedule(request, legacy_bug);
  check(!diagnostic.valid &&
            diagnostic.checked_slots == 24 &&
            diagnostic.failure_reason == day::VerifyFailureReason::StatusMismatch &&
            diagnostic.failure_actor == 0 &&
            diagnostic.failure_obligation_id == place.id,
        "verifier did not diagnose the legacy speculative binding");
}

void build_pasture_two_phase_and_adversarial_proof() {
  Simulator simulator({}, 9917);
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  const auto position = origin(simulator);
  farm.tiles[position.y * simulator.config().board_size + position.x].kind =
      TileKind::WEED;
  day::ProductionObligation pasture{
      400, 0, position, day::GoalKind::BuildPasture, Item::NONE, 1, {}, {},
      0, 23, 1000, true, true};
  day::DayPlanRequest request{&simulator, 0, 9400, {}, {pasture}};
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.debts.empty() &&
            plan.manifest[0][0].op == Op::DIG &&
            plan.manifest[0][1].op == Op::BUILD_PASTURE &&
            plan.certificate->obligation_statuses[0].transition_steps ==
                std::vector<int>({0, 1}) &&
            plan.certificate->obligation_statuses[0].remaining_transitions == 0 &&
            day::verify_day_schedule(request, *plan.certificate).valid,
        "BUILD_PASTURE did not prove WEED->DIG->BUILD_PASTURE");

  auto changed_action = *plan.certificate;
  changed_action.slots[1].actions[0] = {Op::CARE, Item::NONE, 1};
  changed_action.slots[1].sources[0].source_action =
      changed_action.slots[1].actions[0];
  rehash(changed_action);
  check(!day::verify_day_schedule(request, changed_action).valid,
        "tampered BUILD_PASTURE second transition accepted");
  auto changed_fingerprint = *plan.certificate;
  changed_fingerprint.slots[1].focal_post_fingerprint ^= 1;
  rehash(changed_fingerprint);
  check(!day::verify_day_schedule(request, changed_fingerprint).valid,
        "tampered BUILD_PASTURE post fingerprint accepted");

  pasture.id = 401;
  pasture.earliest_step = pasture.deadline_step = 23;
  day::DayPlanRequest deadline{&simulator, 0, 9401, {}, {pasture}};
  const auto late = day::plan_day(deadline);
  check(late.planned() && late.manifest[0][23].op == Op::DIG &&
            late.completed.empty() && late.debts.size() == 1 &&
            late.debts[0].reason == day::DebtReason::Deadline &&
            late.debts[0].remaining_transitions == 1 &&
            day::verify_day_schedule(deadline, *late.certificate).valid,
        "one-slot BUILD_PASTURE was not explicit deadline debt");

  auto deferred_request = base_request(simulator);
  deferred_request.issuer_generation = 9402;
  deferred_request.obligations[0].earliest_step = 0;
  deferred_request.obligations[0].deadline_step = 23;
  deferred_request.obligations[0].policy_deferred = true;
  const auto deferred = day::plan_day(deferred_request);
  check(deferred.planned(), "policy-deferred plan rejected");
  check(deferred.completed.empty(), "policy-deferred obligation fake-completed");
  check(deferred.debts.size() == 1,
        "policy-deferred obligation debt count wrong");
  check(deferred.debts[0].reason == day::DebtReason::PolicyDeferred,
        "policy-deferred obligation debt reason wrong");
  check(deferred.certificate->obligation_statuses[0].disposition ==
            day::ObligationDisposition::PolicyDeferredDebt,
        "policy-deferred disposition wrong");
  check(deferred.certificate->obligation_statuses[0].transition_steps.empty(),
        "policy-deferred obligation emitted a transition");
  check(day::verify_day_schedule(deferred_request, *deferred.certificate).valid,
        "policy-deferred certificate invalid");
}

void build_coop_replaces_wrong_structure() {
  Simulator simulator({}, 9918);
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  const auto position = origin(simulator);
  farm.tiles[position.y * simulator.config().board_size + position.x].kind =
      TileKind::PASTURE;
  day::ProductionObligation coop{
      402, 0, position, day::GoalKind::BuildCoop, Item::NONE, 1, {}, {},
      0, 23, 1000, true, true};
  day::DayPlanRequest request{&simulator, 0, 9403, {}, {coop}};
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.debts.empty() &&
            plan.manifest[0][0].op == Op::DIG &&
            plan.manifest[0][1].op == Op::BUILD_COOP &&
            plan.certificate->obligation_statuses[0].transition_steps ==
                std::vector<int>({0, 1}) &&
            day::verify_day_schedule(request, *plan.certificate).valid,
        "BUILD_COOP did not replace a removable wrong structure");
}

void invalid_items_and_min_priority_fail_closed() {
  Simulator simulator({}, 9919);
  const auto position = origin(simulator);
  day::ProductionObligation invalid{
      410, 0, position, day::GoalKind::Place, Item::NONE, 1, {}, {},
      0, 23, 1, true, true};
  day::DayPlanRequest invalid_request{&simulator, 0, 9410, {}, {invalid}};
  check(day::plan_day(invalid_request).reject == day::PlanReject::InvalidDag,
        "invalid PLACE item reached inventory indexing");

  auto& private_state =
      const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
  private_state.shed[static_cast<int>(Item::WHEAT)] = 2;
  day::ProductionObligation low{
      411, 0, position, day::GoalKind::Pickup, Item::WHEAT, 1, {},
      {Item::WHEAT, 0, 1, 0}, 0, 23, std::numeric_limits<int>::min(),
      true, false};
  auto high = low;
  high.id = 412;
  high.priority = 0;
  day::DayPlanRequest priority_request{&simulator, 0, 9411, {}, {low, high}};
  const auto plan = day::plan_day(priority_request);
  check(plan.planned() && plan.completed.size() == 2 &&
            plan.certificate->slots[0].obligation_ids[0] == high.id,
        "INT_MIN priority overflowed or reversed candidate ordering");
}

void terminal_day_has_only_steps_696_through_718() {
  Simulator simulator({}, 9920);
  for (int step = 0; step < 696; ++step) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({});
    actions[1].units.push_back({});
    simulator.step(actions);
  }
  check(simulator.step_count() == 696 && !simulator.done(),
        "terminal-day fixture did not reach step 696");
  day::DayPlanRequest request{
      &simulator, 0, 9420,
      {{0, 718, {Op::WEST, Item::NONE, 1}}}, {}, 0};
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.manifest[0].size() == 23 &&
            plan.certificate->slots.size() == 23 &&
            plan.certificate->slots.back().step == 718 &&
            plan.certificate->move_replays.size() == 1 &&
            plan.certificate->move_replays[0].emitted_step == 718 &&
            day::verify_day_schedule(request, *plan.certificate).valid,
        "terminal day scheduled a synthetic step 719 or lost step 718");

  auto invalid = request;
  invalid.issuer_generation = 9421;
  invalid.moves[0].source_step = 719;
  check(day::plan_day(invalid).reject == day::PlanReject::InvalidMoveToken,
        "terminal observation step 719 was accepted as an action slot");
}

}  // namespace

int main() {
  try {
    weed_crop_keeps_two_moves();
    feed_deadline_beats_crop_without_swallow();
    seed_receipt_unlocks_next_plan();
    successful_buy_seed_reissues_only_remaining_crop_suffix();
    remaining_suffix_keeps_delayed_move();
    capacity_debt_preserves_all_moves();
    two_actor_shared_shed_conflict();
    hour23_and_crossday_reject();
    tamper_all_pass_swallowing_and_unproved_debt();
    tamper_fake_complete_and_hash();
    tamper_unknown_and_duplicate_obligation_id();
    tamper_wrong_actor_tile_and_action();
    tamper_dependency_inversion();
    tamper_before_earliest_and_after_deadline();
    seed970017_day7_speculative_actor_binding_regression();
    build_pasture_two_phase_and_adversarial_proof();
    build_coop_replaces_wrong_structure();
    invalid_items_and_min_priority_fail_closed();
    terminal_day_has_only_steps_696_through_718();
    std::cout << "production_obligation_day_scheduler_tests: 19 groups passed\n";
  } catch (const std::exception& error) {
    std::cerr << "production_obligation_day_scheduler_tests: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
