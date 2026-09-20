#include "day_schedule_runtime_receipt.hpp"

#include "production_suffix_scheduler.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace runtime = g001::day_runtime_receipt;
namespace day = g001::obligation_day;
namespace suffix = g001::production_suffix;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;

namespace {

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

fastkag::Farm& farm(Simulator& simulator) {
  return const_cast<fastkag::Farm&>(simulator.farms()[0]);
}

fastkag::PrivateState& private_state(Simulator& simulator) {
  return const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
}

Tile& actor_tile(Simulator& simulator) {
  const auto position = simulator.farms()[0].farmer;
  return farm(simulator).tiles[static_cast<std::size_t>(
      position.y * simulator.config().board_size + position.x)];
}

void add_inventory(Simulator& simulator, Item item, int quantity) {
  auto& state = private_state(simulator);
  const auto index = static_cast<std::size_t>(static_cast<int>(item));
  state.inventories[0][index] = quantity;
  if (quantity > 0)
    state.inventory_order[0].push_back(static_cast<int8_t>(index));
}

day::ProductionObligation obligation(std::uint64_t id, day::GoalKind goal,
                                     Item item, Position tile) {
  day::ProductionObligation result;
  result.id = id;
  result.actor = 0;
  result.tile = tile;
  result.goal = goal;
  result.item = item;
  result.quantity = 1;
  result.earliest_step = 0;
  result.deadline_step = 23;
  result.priority = 100;
  result.must_finish_today = true;
  return result;
}

struct BuiltCase {
  std::unique_ptr<Simulator> day_start;
  day::DayPlanRequest request;
  day::DayScheduleCertificate certificate;
};

BuiltCase build_case(std::vector<day::MoveSourceToken> moves,
                     std::vector<day::ProductionObligation> obligations,
                     std::unique_ptr<Simulator> simulator) {
  BuiltCase result;
  result.day_start = std::move(simulator);
  result.request.day_start = result.day_start.get();
  result.request.player = 0;
  result.request.issuer_generation = 0x8a71;
  result.request.moves = std::move(moves);
  result.request.obligations = std::move(obligations);
  const auto planned = day::plan_day(result.request);
  check(planned.planned(), "fixture schedule was not planned");
  check(day::verify_day_schedule(result.request, *planned.certificate).valid,
        "fixture certificate did not verify");
  result.certificate = *planned.certificate;
  return result;
}

std::unique_ptr<Simulator> base_simulator() {
  fastkag::Config config;
  config.weed_spawn_chance = 0.0;
  return std::make_unique<Simulator>(config, 99117);
}

BuiltCase move_case() {
  auto simulator = base_simulator();
  std::vector<day::MoveSourceToken> moves{
      {0, 0, {Op::WEST, Item::NONE, 1}},
      {0, 1, {Op::NORTH, Item::NONE, 1}},
      {0, 2, {Op::EAST, Item::NONE, 1}},
      {0, 3, {Op::SOUTH, Item::NONE, 1}},
  };
  return build_case(std::move(moves), {}, std::move(simulator));
}

BuiltCase boundary_move_case() {
  auto simulator = base_simulator();
  farm(*simulator).farmer = {0, 0};
  return build_case({{0, 0, {Op::WEST, Item::NONE, 1}}}, {},
                    std::move(simulator));
}

BuiltCase crop_case() {
  auto simulator = base_simulator();
  actor_tile(*simulator).kind = TileKind::WEED;
  private_state(*simulator).seeds[0] = 1;
  const auto position = simulator->farms()[0].farmer;
  return build_case({},
                    {obligation(11, day::GoalKind::CropReady, Item::WHEAT,
                                position)},
                    std::move(simulator));
}

BuiltCase pickup_case() {
  auto simulator = base_simulator();
  private_state(*simulator).shed[0] = 1;
  const auto position = simulator->farms()[0].farmer;
  return build_case({},
                    {obligation(21, day::GoalKind::Pickup, Item::WHEAT,
                                position)},
                    std::move(simulator));
}

BuiltCase place_case() {
  auto simulator = base_simulator();
  actor_tile(*simulator).kind = TileKind::PASTURE;
  add_inventory(*simulator, Item::COW, 1);
  const auto position = simulator->farms()[0].farmer;
  return build_case({},
                    {obligation(31, day::GoalKind::Place, Item::COW,
                                position)},
                    std::move(simulator));
}

BuiltCase feed_case() {
  auto simulator = base_simulator();
  actor_tile(*simulator).kind = TileKind::ANIMAL;
  actor_tile(*simulator).animal = Item::GOOSE;
  add_inventory(*simulator, Item::WHEAT, 1);
  const auto position = simulator->farms()[0].farmer;
  return build_case({},
                    {obligation(41, day::GoalKind::Feed, Item::WHEAT,
                                position)},
                    std::move(simulator));
}

BuiltCase care_case() {
  auto simulator = base_simulator();
  actor_tile(*simulator).kind = TileKind::ANIMAL;
  actor_tile(*simulator).animal = Item::SHEEP;
  const auto position = simulator->farms()[0].farmer;
  return build_case({},
                    {obligation(51, day::GoalKind::Care, Item::SHEEP,
                                position)},
                    std::move(simulator));
}

BuiltCase harvest_case() {
  auto simulator = base_simulator();
  actor_tile(*simulator).kind = TileKind::PLANT;
  actor_tile(*simulator).crop = Item::WHEAT;
  actor_tile(*simulator).planted_day = -2;
  actor_tile(*simulator).yield_units = 2;
  const auto position = simulator->farms()[0].farmer;
  return build_case({},
                    {obligation(61, day::GoalKind::Harvest, Item::WHEAT,
                                position)},
                    std::move(simulator));
}

BuiltCase pasture_case() {
  auto simulator = base_simulator();
  actor_tile(*simulator).kind = TileKind::WEED;
  const auto position = simulator->farms()[0].farmer;
  return build_case({},
                    {obligation(71, day::GoalKind::BuildPasture, Item::NONE,
                                position)},
                    std::move(simulator));
}

runtime::StepExecution execution_for(
    const BuiltCase& built, int tick, const Simulator& before,
    const Simulator& after, std::vector<runtime::ActorExecution>& actors) {
  const auto& slot = built.certificate.slots[static_cast<std::size_t>(tick)];
  actors.clear();
  for (std::size_t actor = 0; actor < slot.actions.size(); ++actor) {
    actors.push_back({built.certificate.player,
                      built.certificate.day,
                      slot.step,
                      static_cast<int>(actor),
                      slot.sources[actor].source_step,
                      slot.obligation_ids[actor],
                      built.certificate.content_hash,
                      built.certificate.issuer_generation,
                      slot.sources[actor].source_action,
                      slot.actions[actor]});
  }
  return {&before,
          &after,
          suffix::focal_unit_state_fingerprint(before, 0, slot.step),
          suffix::focal_unit_state_fingerprint(after, 0, slot.step),
          actors};
}

Simulator preview_slot(const BuiltCase& built, int tick,
                       const Simulator& before,
                       const std::vector<Action>* override_actions = nullptr) {
  std::array<PlayerAction, 2> joint;
  joint[0].units = override_actions == nullptr
                       ? built.certificate.slots[static_cast<std::size_t>(tick)]
                             .actions
                       : *override_actions;
  return before.preview_unit_phase(joint);
}

std::set<Op> execute_day(BuiltCase& built) {
  runtime::RuntimeReceiptVerifier verifier(built.request, built.certificate);
  check(verifier.ready(), "runtime verifier did not open");
  Simulator state = *built.day_start;
  std::vector<runtime::ActorExecution> actor_records;
  std::set<Op> seen;
  for (int tick = 0; tick < 24; ++tick) {
    Simulator after = preview_slot(built, tick, state);
    auto receipt = execution_for(built, tick, state, after, actor_records);
    const auto accepted = verifier.accept(receipt);
    check(accepted.accepted,
          std::string("valid receipt rejected: ") +
              runtime::failure_name(accepted.failure));
    for (const auto& action : built.certificate.slots[tick].actions)
      seen.insert(action.op);
    state = std::move(after);
  }
  const auto closed = verifier.close_midnight();
  check(closed.accepted && closed.day_closed && closed.checked_steps == 24,
        "complete day did not close at midnight");
  return seen;
}

void all_supported_operations_physically_close() {
  std::set<Op> seen;
  auto collect = [&](BuiltCase built) {
    const auto local = execute_day(built);
    seen.insert(local.begin(), local.end());
  };
  collect(move_case());
  collect(crop_case());
  collect(pickup_case());
  collect(place_case());
  collect(feed_case());
  collect(care_case());
  collect(harvest_case());
  collect(pasture_case());

  const std::set<Op> required{
      Op::PASS,    Op::NORTH,   Op::SOUTH, Op::EAST,  Op::WEST,
      Op::DIG,     Op::PLANT,   Op::WATER, Op::HARVEST,
      Op::PICKUP,  Op::PLACE,   Op::FEED,  Op::CARE,
      Op::BUILD_PASTURE};
  check(seen == required, "positive suite missed a supported operation");
}

runtime::Result submit_first(BuiltCase& built, Simulator before,
                             bool forge_after_hash = false,
                             bool overlay_action = false) {
  runtime::RuntimeReceiptVerifier verifier(built.request, built.certificate);
  check(verifier.ready(), "negative verifier did not open");
  std::vector<Action> actions = built.certificate.slots[0].actions;
  if (overlay_action) actions[0] = {Op::EAST, Item::NONE, 1};
  Simulator after = preview_slot(built, 0, before, &actions);
  std::vector<runtime::ActorExecution> actor_records;
  auto receipt = execution_for(built, 0, before, after, actor_records);
  if (overlay_action) receipt.actors[0].emitted = actions[0];
  if (forge_after_hash) receipt.claimed_after_fingerprint ^= 1U;
  return verifier.accept(receipt);
}

void physical_and_overlay_negatives() {
  {
    // DayScheduleCertificate verification proves MOVE identity/coverage but
    // does not claim that a MOVE changes position.  Therefore a boundary WEST
    // certificate is valid input to this receipt layer and directly exercises
    // its independent physical no-effect rejection on the unchanged state.
    auto built = boundary_move_case();
    const auto result = submit_first(built, *built.day_start);
    check(!result.accepted &&
              result.failure == runtime::Failure::PhysicalNoEffect,
          "MOVE into boundary did not hit physical no-effect rejection");
  }
  {
    auto built = feed_case();
    Simulator fork = *built.day_start;
    actor_tile(fork) = Tile{};
    const auto result = submit_first(built, std::move(fork));
    check(!result.accepted &&
              result.failure == runtime::Failure::StateFingerprintFork,
          "FEED on empty tile was not rejected");
  }
  {
    auto built = pickup_case();
    Simulator fork = *built.day_start;
    farm(fork).farmer = {0, 0};
    const auto result = submit_first(built, std::move(fork));
    check(!result.accepted &&
              result.failure == runtime::Failure::StateFingerprintFork,
          "PICKUP at wrong location was not rejected");
  }
  {
    auto built = care_case();
    Simulator fork = *built.day_start;
    actor_tile(fork) = Tile{};
    const auto result = submit_first(built, std::move(fork));
    check(!result.accepted &&
              result.failure == runtime::Failure::StateFingerprintFork,
          "CARE without animal was not rejected");
  }
  {
    auto built = move_case();
    const auto result = submit_first(built, *built.day_start, false, true);
    check(!result.accepted &&
              result.failure == runtime::Failure::EmittedActionMismatch,
          "overlay action replacement was not rejected");
  }
  {
    auto built = move_case();
    const auto result = submit_first(built, *built.day_start, true);
    check(!result.accepted &&
              result.failure == runtime::Failure::AfterFingerprintForgery,
          "forged after fingerprint was not rejected");
  }
}

void ordering_binding_and_midnight_negatives() {
  {
    auto built = move_case();
    runtime::RuntimeReceiptVerifier verifier(built.request, built.certificate);
    Simulator before = *built.day_start;
    Simulator after = preview_slot(built, 0, before);
    std::vector<runtime::ActorExecution> actor_records;
    auto receipt = execution_for(built, 0, before, after, actor_records);
    check(verifier.accept(receipt).accepted, "first receipt failed");
    const auto duplicate = verifier.accept(receipt);
    check(!duplicate.accepted &&
              duplicate.failure == runtime::Failure::DuplicateStep,
          "duplicate step was not rejected");
  }
  {
    auto built = move_case();
    runtime::RuntimeReceiptVerifier verifier(built.request, built.certificate);
    Simulator before = *built.day_start;
    Simulator after = preview_slot(built, 1, before);
    std::vector<runtime::ActorExecution> actor_records;
    auto receipt = execution_for(built, 1, before, after, actor_records);
    const auto skipped = verifier.accept(receipt);
    check(!skipped.accepted &&
              skipped.failure == runtime::Failure::SkippedStep,
          "skipped step was not rejected");
  }
  {
    auto built = move_case();
    runtime::RuntimeReceiptVerifier verifier(built.request, built.certificate);
    const auto incomplete = verifier.close_midnight();
    check(!incomplete.accepted &&
              incomplete.failure == runtime::Failure::IncompleteDay,
          "incomplete day closed at midnight");
  }
  {
    auto built = feed_case();
    runtime::RuntimeReceiptVerifier verifier(built.request, built.certificate);
    Simulator before = *built.day_start;
    Simulator after = preview_slot(built, 0, before);
    std::vector<runtime::ActorExecution> actor_records;
    auto receipt = execution_for(built, 0, before, after, actor_records);
    receipt.actors[0].issuer_generation ^= 1U;
    const auto result = verifier.accept(receipt);
    check(!result.accepted &&
              result.failure == runtime::Failure::GenerationBinding,
          "generation substitution was not rejected");
  }
}

void invalid_certificate_never_opens() {
  auto built = move_case();
  ++built.certificate.content_hash;
  runtime::RuntimeReceiptVerifier verifier(built.request, built.certificate);
  check(!verifier.ready() &&
            verifier.opening_failure() == runtime::Failure::CertificateInvalid,
        "tampered certificate opened a runtime verifier");
}

}  // namespace

int main() {
  try {
    all_supported_operations_physically_close();
    physical_and_overlay_negatives();
    ordering_binding_and_midnight_negatives();
    invalid_certificate_never_opens();
    std::cout << "day schedule runtime receipt tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "day schedule runtime receipt tests failed: " << error.what()
              << '\n';
    return 1;
  }
}
