#include "state_target_compiler.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace target = g001::state_target;
namespace day = g001::obligation_day;
using fastkag::Item;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Position origin(const Simulator& simulator) {
  return simulator.farms()[0].farmer;
}

Tile& origin_tile(Simulator& simulator) {
  const Position position = origin(simulator);
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  return farm.tiles[position.y * simulator.config().board_size + position.x];
}

fastkag::PrivateState& private_state(Simulator& simulator) {
  return const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
}

target::PlotTarget crop_target(const Simulator& simulator) {
  return {1, origin(simulator), target::PlotTargetKind::Crop, Item::WHEAT,
          0, 100.0, origin(simulator)};
}

target::PlotTarget animal_target(const Simulator& simulator, Item animal) {
  return {2, origin(simulator), target::PlotTargetKind::Animal, animal,
          0, 200.0, origin(simulator), true, true, true};
}

const day::ProductionObligation* find_goal(
    const target::CompileResult& result, day::GoalKind goal) {
  const auto found = std::find_if(
      result.obligations.begin(), result.obligations.end(),
      [&](const auto& obligation) { return obligation.goal == goal; });
  return found == result.obligations.end() ? nullptr : &*found;
}

const target::TargetGroup* find_group(const target::CompileResult& result,
                                      target::TargetGroupKind kind) {
  const auto found = std::find_if(
      result.groups.begin(), result.groups.end(),
      [&](const auto& group) { return group.kind == kind; });
  return found == result.groups.end() ? nullptr : &*found;
}

const target::TargetGroup* find_group_for(
    const target::CompileResult& result, std::uint64_t parent,
    target::TargetGroupKind kind) {
  const auto found = std::find_if(
      result.groups.begin(), result.groups.end(), [&](const auto& group) {
        return group.parent_target_id == parent && group.kind == kind;
      });
  return found == result.groups.end() ? nullptr : &*found;
}

void weed_crop_is_one_atomic_state_goal() {
  Simulator simulator;
  origin_tile(simulator).kind = TileKind::WEED;
  private_state(simulator).seeds[static_cast<int>(Item::WHEAT)] = 1;
  const std::vector targets{crop_target(simulator)};
  const auto result = target::compile_state_targets(simulator, 0, 0, 23,
                                                     targets);
  check(result.obligations.size() == 1 &&
            result.obligations[0].goal == day::GoalKind::CropReady &&
            result.obligations[0].resource.seed_quantity == 1,
        "weed crop was not one CropReady obligation");
  check(result.groups.size() == 1 &&
            result.groups[0].parent_target_id == targets[0].id &&
            result.groups[0].kind ==
                target::TargetGroupKind::CropLifecycle &&
            result.groups[0].obligation_ids ==
                std::vector<std::uint64_t>{result.obligations[0].id} &&
            result.demands.empty(),
        "weed crop atomic group/resource allocation is wrong");
}

void missing_seed_is_a_fresh_resource_demand() {
  Simulator simulator;
  origin_tile(simulator).kind = TileKind::WEED;
  const std::vector targets{crop_target(simulator)};
  const auto result = target::compile_state_targets(simulator, 0, 0, 23,
                                                     targets);
  check(result.obligations.size() == 1 && result.demands.size() == 1 &&
            result.demands[0].item == Item::WHEAT &&
            result.demands[0].quantity == 1 &&
            result.demands[0].group_id == result.groups[0].id &&
            result.demands[0].target_id == targets[0].id &&
            result.demands[0].unfilled_value == result.groups[0].value &&
            result.demands[0].request_step == 0 &&
            result.demands[0].latest_purchase_step == 20 &&
            result.demands[0].reason ==
                target::ResourceDemandReason::Seed,
        "missing seed demand is wrong");
  const std::array approved{result.groups[0].id};
  const auto acquisitions = target::current_acquisitions(result, approved);
  check(acquisitions.size() == 1 &&
            acquisitions[0].op == fastkag::Op::BUY_SEED &&
            acquisitions[0].item == Item::WHEAT &&
            acquisitions[0].quantity == 1,
        "seed demand did not cross the native market boundary");

  const auto expired =
      target::compile_state_targets(simulator, 0, 0, 0, targets);
  check(expired.demands.size() == 1 &&
            target::current_acquisitions(expired, approved).empty(),
        "purchase was emitted after its last usable market phase");

  auto negative_target = targets[0];
  negative_target.value = -1.0;
  const std::vector negative_targets{negative_target};
  const auto negative = target::compile_state_targets(
      simulator, 0, 0, 23, negative_targets);
  const std::array negative_approved{negative.groups[0].id};
  check(target::current_acquisitions(negative, negative_approved).empty(),
        "non-positive target emitted a hard market order");
}

void mature_crop_harvest_and_replant_are_one_atomic_group() {
  Simulator simulator;
  auto& tile = origin_tile(simulator);
  tile.kind = TileKind::PLANT;
  tile.crop = Item::WHEAT;
  tile.planted_day = -2;
  tile.yield_units = 2;
  tile.watered_today = true;
  private_state(simulator).seeds[static_cast<int>(Item::WHEAT)] = 1;
  const std::vector targets{crop_target(simulator)};
  const auto result = target::compile_state_targets(simulator, 0, 0, 23,
                                                     targets);
  const auto* lifecycle =
      find_group(result, target::TargetGroupKind::CropLifecycle);
  const auto* harvest = find_goal(result, day::GoalKind::Harvest);
  const auto* ready = find_goal(result, day::GoalKind::CropReady);
  check(result.groups.size() == 1 && lifecycle != nullptr &&
            lifecycle->obligation_ids.size() == 2 && harvest != nullptr &&
            ready != nullptr &&
            ready->dependencies == std::vector<std::uint64_t>{harvest->id} &&
            ready->resource.seed_quantity == 1,
        "mature crop was not compiled as HARVEST -> replant lifecycle");

  day::DayPlanRequest request{&simulator, 0, 8801, {}, result.obligations};
  const auto plan = day::atomic_plan_day(request, target::atomic_groups(result));
  check(plan.planned() && plan.groups.size() == 1 && plan.groups[0].admitted &&
            plan.verified_plan.manifest[0][0].op == fastkag::Op::HARVEST &&
            plan.verified_plan.manifest[0][1].op == fastkag::Op::PLANT &&
            plan.verified_plan.manifest[0][2].op == fastkag::Op::WATER,
        "mature crop group was certified without its replant suffix");
  Simulator replay = simulator;
  for (std::size_t slot = 0;
       slot < plan.verified_plan.manifest[0].size(); ++slot) {
    std::array<PlayerAction, 2> actions;
    for (const auto& actor : plan.verified_plan.manifest)
      actions[0].units.push_back(actor[slot]);
    replay = replay.preview_unit_phase(actions);
  }
  const auto& final_tile = origin_tile(replay);
  check(final_tile.kind == TileKind::PLANT &&
            final_tile.crop == Item::WHEAT && final_tile.watered_today,
        "certified mature crop replay ended on an empty plot");
}

void ongoing_crop_harvest_does_not_buy_or_replant() {
  Simulator simulator;
  auto& tile = origin_tile(simulator);
  tile.kind = TileKind::PLANT;
  tile.crop = Item::TOMATO;
  tile.planted_day = -8;
  tile.yield_units = 1;
  tile.watered_today = true;
  auto desired = crop_target(simulator);
  desired.item = Item::TOMATO;
  const std::vector targets{desired};
  const auto result = target::compile_state_targets(simulator, 0, 0, 23,
                                                     targets);
  const auto* ready = find_goal(result, day::GoalKind::CropReady);
  const auto* harvest = find_goal(result, day::GoalKind::Harvest);
  check(result.groups.size() == 1 && result.demands.empty() &&
            ready != nullptr && harvest != nullptr &&
            ready->resource.seed_quantity == 0 &&
            ready->dependencies == std::vector<std::uint64_t>{harvest->id},
        "ongoing crop harvest reserved an unnecessary replacement seed");
  day::DayPlanRequest request{&simulator, 0, 8802, {}, result.obligations};
  const auto plan = day::atomic_plan_day(request, target::atomic_groups(result));
  check(plan.planned() && plan.groups[0].admitted &&
            plan.verified_plan.manifest[0][0].op == fastkag::Op::HARVEST &&
            std::count_if(plan.verified_plan.manifest[0].begin(),
                          plan.verified_plan.manifest[0].end(),
                          [](const auto& action) {
                            return action.op == fastkag::Op::PLANT;
                          }) == 0,
        "ongoing crop harvest incorrectly replanted the surviving crop");
}

void failed_animal_buy_is_rederived_without_a_ledger() {
  Simulator simulator;
  origin_tile(simulator).kind = TileKind::PASTURE;
  auto desired = animal_target(simulator, Item::COW);
  desired.maintain = desired.feed = desired.care = false;
  const std::vector targets{desired};
  const auto first = target::compile_state_targets(simulator, 0, 0, 23,
                                                    targets);
  const auto second = target::compile_state_targets(simulator, 0, 0, 23,
                                                     targets);
  for (const auto* result : {&first, &second}) {
    const auto* pickup = find_goal(*result, day::GoalKind::Pickup);
    const auto* place = find_goal(*result, day::GoalKind::Place);
    check(result->demands.size() == 1 &&
              result->demands[0].item == Item::COW &&
              result->demands[0].reason ==
                  target::ResourceDemandReason::Animal &&
              pickup != nullptr && place != nullptr &&
              place->dependencies == std::vector<std::uint64_t>{pickup->id},
          "failed BUY_ANIMAL was not rebuilt as PICKUP -> PLACE plus demand");
  }
}

void existing_animal_uses_real_maintenance_state() {
  Simulator simulator;
  Tile& tile = origin_tile(simulator);
  tile.kind = TileKind::ANIMAL;
  tile.animal = Item::COW;
  tile.yield_units = 2;
  tile.fertilizer_available = true;
  private_state(simulator).shed[static_cast<int>(Item::WHEAT)] = 1;
  const std::vector targets{animal_target(simulator, Item::COW)};
  const auto result = target::compile_state_targets(simulator, 0, 0, 23,
                                                     targets);
  const auto* pickup = find_goal(result, day::GoalKind::Pickup);
  const auto* feed = find_goal(result, day::GoalKind::Feed);
  const auto* care = find_goal(result, day::GoalKind::Care);
  const auto* feed_group =
      find_group(result, target::TargetGroupKind::AnimalFeed);
  const auto* care_group =
      find_group(result, target::TargetGroupKind::AnimalCare);
  const auto* harvest_group =
      find_group(result, target::TargetGroupKind::AnimalHarvest);
  const auto* fertilizer_group =
      find_group(result, target::TargetGroupKind::AnimalFertilizer);
  check(result.obligations.size() == 5 &&
            find_goal(result, day::GoalKind::Harvest) != nullptr &&
            find_goal(result, day::GoalKind::CollectFertilizer) != nullptr &&
            pickup != nullptr && feed != nullptr &&
            care != nullptr &&
            feed->dependencies == std::vector<std::uint64_t>{pickup->id} &&
            care->dependencies == std::vector<std::uint64_t>{feed->id} &&
            result.demands.empty(),
        "existing animal maintenance ignored real yield/feed/care state");
  check(result.groups.size() == 4 && feed_group != nullptr &&
            care_group != nullptr && harvest_group != nullptr &&
            fertilizer_group != nullptr &&
            feed_group->obligation_ids ==
                std::vector<std::uint64_t>({pickup->id, feed->id}) &&
            care_group->obligation_ids.size() == 1 &&
            harvest_group->obligation_ids.size() == 1 &&
            fertilizer_group->obligation_ids.size() == 1 &&
            feed_group->value > care_group->value &&
            feed_group->value > harvest_group->value &&
            feed_group->value > fertilizer_group->value,
        "capacity groups could still discard FEED with optional maintenance");
}

void unremovable_animal_fails_closed_and_goose_builds_coop() {
  Simulator simulator;
  Tile& tile = origin_tile(simulator);
  tile.kind = TileKind::ANIMAL;
  tile.animal = Item::GOOSE;
  const std::vector wrong{animal_target(simulator, Item::COW)};
  const auto result = target::compile_state_targets(simulator, 0, 0, 23,
                                                     wrong);
  check(result.obligations.empty() && result.groups.empty() &&
            result.diagnostics.size() == 1 &&
            result.diagnostics[0].code ==
                target::DiagnosticCode::CannotRemoveAnimal,
        "wrong existing animal was silently removed");

  tile = Tile{};
  auto goose_target = animal_target(simulator, Item::GOOSE);
  goose_target.maintain = goose_target.feed = goose_target.care = false;
  const std::vector goose{goose_target};
  const auto goose_result = target::compile_state_targets(
      simulator, 0, 0, 23, goose);
  const auto* build = find_goal(goose_result, day::GoalKind::BuildCoop);
  const auto* place = find_goal(goose_result, day::GoalKind::Place);
  check(goose_result.diagnostics.empty() && build != nullptr &&
            place != nullptr && goose_result.groups.size() == 1 &&
            goose_result.groups[0].kind ==
                target::TargetGroupKind::AnimalAcquisition &&
            std::find(place->dependencies.begin(), place->dependencies.end(),
                      build->id) != place->dependencies.end(),
        "missing goose structure was not compiled as BuildCoop -> Place");
}

void shared_resources_follow_value_not_input_order() {
  Simulator simulator;
  const auto first = origin(simulator);
  const Position second{static_cast<std::int16_t>(first.x - 1), first.y};
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  auto set_cow = [&](Position position) {
    auto& tile = farm.tiles[position.y * simulator.config().board_size +
                            position.x];
    tile = Tile{};
    tile.kind = TileKind::ANIMAL;
    tile.animal = Item::COW;
  };
  set_cow(first);
  set_cow(second);
  private_state(simulator).inventories[0]
      [static_cast<int>(Item::WHEAT)] = 1;

  auto low = animal_target(simulator, Item::COW);
  low.id = 10;
  low.tile = first;
  low.value = 1.0;
  low.maintain = low.care = false;
  auto high = low;
  high.id = 11;
  high.tile = second;
  high.value = 100.0;
  const std::vector targets{low, high};  // Deliberately low value first.
  const auto result = target::compile_state_targets(simulator, 0, 0, 23,
                                                     targets);
  const auto* high_group = find_group_for(
      result, high.id, target::TargetGroupKind::AnimalFeed);
  const auto* low_group = find_group_for(
      result, low.id, target::TargetGroupKind::AnimalFeed);
  check(high_group != nullptr && low_group != nullptr &&
            high_group->obligation_ids.size() == 1 &&
            low_group->obligation_ids.size() == 2 &&
            result.demands.size() == 1 &&
            result.demands[0].reason ==
                target::ResourceDemandReason::FeedWheat,
        "caller order stole a shared resource from the higher-value target");
}

void shared_resources_follow_split_group_value() {
  Simulator simulator;
  const auto first = origin(simulator);
  const Position second{static_cast<std::int16_t>(first.x - 1), first.y};
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  auto set_cow = [&](Position position, bool optional_yield) {
    auto& tile = farm.tiles[position.y * simulator.config().board_size +
                            position.x];
    tile = Tile{};
    tile.kind = TileKind::ANIMAL;
    tile.animal = Item::COW;
    tile.yield_units = optional_yield ? 1 : 0;
    tile.fertilizer_available = optional_yield;
  };
  set_cow(first, true);
  set_cow(second, false);
  private_state(simulator).inventories[0]
      [static_cast<int>(Item::WHEAT)] = 1;

  auto split_low = animal_target(simulator, Item::COW);
  split_low.id = 20;
  split_low.tile = first;
  split_low.value = 100.0;
  auto split_high = split_low;
  split_high.id = 21;
  split_high.tile = second;
  split_high.value = 50.0;
  split_high.maintain = false;
  split_high.care = false;
  const std::vector targets{split_low, split_high};
  const auto result = target::compile_state_targets(simulator, 0, 0, 23,
                                                     targets);
  const auto* low_feed = find_group_for(
      result, split_low.id, target::TargetGroupKind::AnimalFeed);
  const auto* high_feed = find_group_for(
      result, split_high.id, target::TargetGroupKind::AnimalFeed);
  check(low_feed != nullptr && high_feed != nullptr &&
            low_feed->value == 40.0 && high_feed->value == 50.0 &&
            low_feed->obligation_ids.size() == 2 &&
            high_feed->obligation_ids.size() == 1 &&
            result.demands.size() == 1 &&
            result.demands[0].group_id == low_feed->id,
        "shared wheat followed target total instead of atomic group value");
}

void terminal_observation_is_not_an_action_slot() {
  Simulator simulator;
  for (int step = 0; step < 696; ++step) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({});
    actions[1].units.push_back({});
    simulator.step(actions);
  }
  auto desired = crop_target(simulator);
  const std::vector targets{desired};
  const auto valid = target::compile_state_targets(simulator, 0, 696, 718,
                                                    targets);
  const auto invalid = target::compile_state_targets(simulator, 0, 696, 719,
                                                      targets);
  check(valid.diagnostics.empty() && !valid.obligations.empty() &&
            invalid.obligations.empty() && invalid.diagnostics.size() == 1 &&
            invalid.diagnostics[0].code ==
                target::DiagnosticCode::InvalidRequest,
        "state target compiler treated terminal observation 719 as a slot");
}

void compiled_target_runs_atomically_in_the_native_scheduler() {
  Simulator simulator;
  origin_tile(simulator).kind = TileKind::WEED;
  private_state(simulator).seeds[static_cast<int>(Item::WHEAT)] = 1;
  const std::vector targets{crop_target(simulator)};
  auto compiled =
      target::compile_state_targets(simulator, 0, 0, 23, targets);
  day::DayPlanRequest request{&simulator, 0, 9001, {}, compiled.obligations};
  const auto plan = day::atomic_plan_day(request,
                                          target::atomic_groups(compiled));
  check(plan.planned() && plan.groups.size() == 1 &&
            plan.groups[0].admitted &&
            plan.verified_plan.manifest[0][0].op == fastkag::Op::DIG &&
            plan.verified_plan.manifest[0][1].op == fastkag::Op::PLANT &&
            plan.verified_plan.manifest[0][2].op == fastkag::Op::WATER,
        "state target did not compile and schedule as one atomic lifecycle");
}

}  // namespace

int main() {
  try {
    weed_crop_is_one_atomic_state_goal();
    missing_seed_is_a_fresh_resource_demand();
    mature_crop_harvest_and_replant_are_one_atomic_group();
    ongoing_crop_harvest_does_not_buy_or_replant();
    failed_animal_buy_is_rederived_without_a_ledger();
    existing_animal_uses_real_maintenance_state();
    unremovable_animal_fails_closed_and_goose_builds_coop();
    shared_resources_follow_value_not_input_order();
    shared_resources_follow_split_group_value();
    terminal_observation_is_not_an_action_slot();
    compiled_target_runs_atomically_in_the_native_scheduler();
    std::cout << "state_target_compiler_tests: 11 groups passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
