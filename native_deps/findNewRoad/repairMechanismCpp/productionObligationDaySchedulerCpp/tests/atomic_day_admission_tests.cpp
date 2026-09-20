#include "atomic_day_admission.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace day = g001::obligation_day;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Simulator day_one(int wheat_seeds) {
  fastkag::Config config;
  config.weed_spawn_chance = 1;
  Simulator simulator(config, 77881);
  for (int step = 0; step < 24; ++step) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({});
    actions[1].units.push_back({});
    if (step == 0 && wheat_seeds > 0)
      actions[0].market.push_back(
          {Op::BUY_SEED, Item::WHEAT, wheat_seeds});
    simulator.step(actions);
  }
  return simulator;
}

Position farmer(const Simulator& simulator) {
  return simulator.farms()[0].farmer;
}

day::ProductionObligation crop(std::uint64_t id,
                               const Simulator& simulator) {
  const int first = simulator.day() * 24;
  return {id,
          0,
          farmer(simulator),
          day::GoalKind::CropReady,
          Item::WHEAT,
          1,
          {},
          {},
          first,
          first + 23,
          1,
          true,
          true};
}

std::vector<day::MoveSourceToken> zigzag(int count, int first_step) {
  std::vector<day::MoveSourceToken> moves;
  for (int i = 0; i < count; ++i) {
    moves.push_back(
        {0, first_step + i,
         {i % 2 == 0 ? Op::WEST : Op::EAST, Item::NONE, 1}});
  }
  return moves;
}

void check_moves(const day::AtomicAdmissionResult& result,
                 const std::vector<day::MoveSourceToken>& expected) {
  const auto& replay = result.verified_plan.certificate->move_replays;
  check(replay.size() == expected.size(), "MOVE count changed");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    check(replay[i].actor == expected[i].actor &&
              replay[i].source_step == expected[i].source_step &&
              replay[i].action.op == expected[i].action.op,
          "MOVE identity/order changed");
  }
}

void two_slots_reject_whole_crop() {
  auto simulator = day_one(1);
  auto obligation = crop(1, simulator);
  auto moves = zigzag(22, 24);
  day::DayPlanRequest request{&simulator, 0, 7001, moves, {obligation}};
  const auto result = day::atomic_plan_day(request, {{101, 10.0, {1}}});

  check(result.planned() && result.planner_calls == 2,
        "two-slot atomic plan failed");
  check(!result.groups[0].admitted &&
            result.effective_request.obligations[0].policy_deferred,
        "partial crop group was admitted");
  for (const auto& actor : result.verified_plan.manifest) {
    check(std::none_of(actor.begin(), actor.end(), [](const auto& action) {
            return action.op == Op::DIG || action.op == Op::PLANT ||
                   action.op == Op::WATER;
          }),
          "rejected crop leaked a partial production action");
  }
  check_moves(result, moves);
}

void three_slots_complete_crop_and_moves() {
  auto simulator = day_one(1);
  auto obligation = crop(2, simulator);
  auto moves = zigzag(21, 24);
  day::DayPlanRequest request{&simulator, 0, 7002, moves, {obligation}};
  const auto result = day::atomic_plan_day(request, {{102, 10.0, {2}}});

  check(result.planned() && result.groups[0].admitted,
        "three-slot crop was not admitted");
  check(result.verified_plan.manifest[0][0].op == Op::DIG &&
            result.verified_plan.manifest[0][1].op == Op::PLANT &&
            result.verified_plan.manifest[0][2].op == Op::WATER,
        "admitted crop lifecycle is incomplete");
  check_moves(result, moves);
}

day::ProductionObligation pickup(std::uint64_t id,
                                 const Simulator& simulator, int priority) {
  const int first = simulator.day() * 24;
  return {id,
          0,
          farmer(simulator),
          day::GoalKind::Pickup,
          Item::WHEAT,
          1,
          {},
          {Item::WHEAT, 0, 1, 0},
          first,
          first + 23,
          priority,
          true,
          false};
}

void high_value_group_cannot_be_displaced() {
  auto simulator = day_one(0);
  auto& private_state =
      const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
  private_state.shed[static_cast<int>(Item::WHEAT)] = 2;
  auto high = pickup(10, simulator, 1);
  auto low = pickup(11, simulator, 1000);
  auto moves = zigzag(23, 24);
  day::DayPlanRequest request{&simulator, 0, 7010, moves, {high, low}};
  const std::vector<day::AtomicGroup> groups{{110, 100.0, {10}},
                                             {111, 1.0, {11}}};
  const auto result = day::atomic_plan_day(request, groups);

  check(result.planned() && result.groups[0].admitted &&
            !result.groups[1].admitted,
        "lower-value group displaced the admitted higher-value group");
  check(result.verified_plan.certificate->slots[0].obligation_ids[0] == 10 &&
            result.effective_request.obligations[1].policy_deferred,
        "final manifest does not contain only the high-value group");
  check_moves(result, moves);
}

void dependency_group_is_admitted_with_consumer() {
  auto simulator = day_one(0);
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  auto& private_state =
      const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
  const auto position = farmer(simulator);
  auto& animal = farm.tiles[position.y * simulator.config().board_size +
                            position.x];
  animal = Tile{};
  animal.kind = TileKind::ANIMAL;
  animal.animal = Item::GOOSE;
  private_state.shed[static_cast<int>(Item::WHEAT)] = 1;
  auto get_wheat = pickup(20, simulator, 1);
  auto feed = day::ProductionObligation{
      21,
      0,
      position,
      day::GoalKind::Feed,
      Item::WHEAT,
      1,
      {20},
      {Item::WHEAT, 0, 0, 1},
      24,
      47,
      1,
      true,
      false};
  day::DayPlanRequest request{&simulator, 0, 7020, {}, {get_wheat, feed}};
  const std::vector<day::AtomicGroup> groups{{120, -1.0, {20}},
                                             {121, 100.0, {21}}};
  const auto result = day::atomic_plan_day(request, groups);

  check(result.planned() && result.groups[0].admitted &&
            result.groups[1].admitted && result.planner_calls == 2,
        "cross-group dependency closure was not admitted atomically");
  check(result.verified_plan.manifest[0][0].op == Op::PICKUP &&
            result.verified_plan.manifest[0][1].op == Op::FEED,
        "dependency closure executed out of order");
}

void non_positive_group_is_not_an_admission_root() {
  auto simulator = day_one(1);
  const auto obligation = crop(22, simulator);
  day::DayPlanRequest request{&simulator, 0, 7022, {}, {obligation}};
  const auto result = day::atomic_plan_day(request, {{122, 0.0, {22}}});
  check(result.planned() && result.planner_calls == 1 &&
            !result.groups[0].admitted &&
            result.effective_request.obligations[0].policy_deferred &&
            result.verified_plan.manifest[0][0].op == Op::PASS,
        "non-positive advantage group changed the baseline");
}

void missing_seed_and_original_deferral_never_leak_dig() {
  auto simulator = day_one(0);
  auto obligation = crop(30, simulator);
  day::DayPlanRequest request{&simulator, 0, 7030, {}, {obligation}};
  auto missing = day::atomic_plan_day(request, {{130, 1.0, {30}}});
  check(missing.planned() && !missing.groups[0].admitted &&
            missing.groups[0].disposition ==
                day::ObligationDisposition::BlockedOnReceipt &&
            missing.verified_plan.manifest[0][0].op == Op::PASS,
        "missing seed leaked DIG from a rejected group");

  auto deferred_obligation = crop(31, simulator);
  deferred_obligation.policy_deferred = true;
  day::DayPlanRequest deferred_request{&simulator,
                                       0,
                                       7031,
                                       {},
                                       {deferred_obligation}};
  auto deferred =
      day::atomic_plan_day(deferred_request, {{131, 1.0, {31}}});
  check(deferred.planned() && deferred.planner_calls == 1 &&
            !deferred.groups[0].admitted &&
            deferred.effective_request.obligations[0].policy_deferred &&
            deferred.verified_plan.manifest[0][0].op == Op::PASS,
        "original policy deferral was activated");
}

void remaining_day_uses_same_atomic_rule() {
  auto simulator = day_one(1);
  std::array<PlayerAction, 2> pass;
  pass[0].units.push_back({});
  pass[1].units.push_back({});
  simulator.step(pass);
  auto obligation = crop(40, simulator);
  day::DayPlanRequest request{&simulator, 0, 7040, {}, {obligation}, 1};
  const auto result =
      day::atomic_plan_remaining_day(request, {{140, 1.0, {40}}});

  check(result.planned() && result.groups[0].admitted &&
            result.verified_plan.manifest[0].size() == 23 &&
            result.verified_plan.certificate->start_step == 25 &&
            result.verified_plan.manifest[0][0].op == Op::DIG &&
            result.verified_plan.manifest[0][1].op == Op::PLANT &&
            result.verified_plan.manifest[0][2].op == Op::WATER,
        "remaining-day atomic planning failed");
}

void membership_is_exact() {
  auto simulator = day_one(1);
  day::DayPlanRequest request{&simulator, 0, 7050, {}, {crop(50, simulator)}};
  const auto missing = day::atomic_plan_day(request, {});
  const auto duplicate = day::atomic_plan_day(
      request, {{150, 1.0, {50}}, {151, 0.0, {50}}});
  check(missing.reject == day::AtomicAdmissionReject::InvalidGroups &&
            duplicate.reject == day::AtomicAdmissionReject::InvalidGroups &&
            missing.planner_calls == 0 && duplicate.planner_calls == 0,
        "invalid group membership was accepted");
}

void every_daily_capacity_preserves_moves_and_crop_atomicity() {
  for (int move_count = 0; move_count <= 24; ++move_count) {
    auto simulator = day_one(1);
    const auto moves = zigzag(move_count, 24);
    const auto obligation = crop(60, simulator);
    day::DayPlanRequest request{&simulator, 0,
                                7060 + static_cast<std::uint64_t>(move_count),
                                moves, {obligation}};
    const auto result =
        day::atomic_plan_day(request, {{160, 1.0, {60}}});
    const bool has_full_crop_capacity = 24 - move_count >= 3;
    check(result.planned() &&
              result.groups[0].admitted == has_full_crop_capacity,
          "daily capacity threshold admitted a partial crop");
    check_moves(result, moves);
    int production_actions = 0;
    for (const auto& action : result.verified_plan.manifest[0])
      production_actions += action.op == Op::DIG || action.op == Op::PLANT ||
                            action.op == Op::WATER;
    check(production_actions == (has_full_crop_capacity ? 3 : 0),
          "capacity sweep leaked or dropped a crop transition");
  }
}

}  // namespace

int main() {
  try {
    two_slots_reject_whole_crop();
    three_slots_complete_crop_and_moves();
    high_value_group_cannot_be_displaced();
    dependency_group_is_admitted_with_consumer();
    non_positive_group_is_not_an_admission_root();
    missing_seed_and_original_deferral_never_leak_dig();
    remaining_day_uses_same_atomic_rule();
    membership_is_exact();
    every_daily_capacity_preserves_moves_and_crop_atomicity();
    std::cout << "atomic_day_admission_tests: 9 groups passed\n";
  } catch (const std::exception& error) {
    std::cerr << "atomic_day_admission_tests: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
