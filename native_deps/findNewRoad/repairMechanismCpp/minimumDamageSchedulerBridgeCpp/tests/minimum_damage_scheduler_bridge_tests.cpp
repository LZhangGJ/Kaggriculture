#include "minimum_damage_scheduler_bridge.hpp"

#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bridge = g001::minimum_damage_bridge;
namespace issuer = g001::day_start_issuer;
namespace scheduler = g001::obligation_day;

namespace {

constexpr std::uint64_t kSeed = 970017;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::uint64_t generation(int day) {
  return (kSeed << 20) | (2ULL << 16) |
         static_cast<std::uint64_t>(day + 1);
}

struct FrozenDay7 {
  fastkag::Simulator env{{}, kSeed};
  issuer::IssueResult issued;
  std::vector<bridge::ObligationPolicy> policies;
};

FrozenDay7 frozen_day7() {
  fastkag::NativeTapeLibrary library;
  library.routes.push_back(g001::repair::load_route(
      BRIDGE_G001_TAPES, BRIDGE_G001_LIBRARY, "G001"));
  fastkag::NativeTeammateExecutor executor(std::move(library));
  FrozenDay7 frozen;
  std::array<fastkag::NativeAgentState, 2> states;
  while (frozen.env.step_count() < 168) {
    std::array<fastkag::PlayerAction, 2> actions;
    for (int player = 0; player < 2; ++player)
      actions[player] =
          executor.action_external(frozen.env, player, 0, states[player]);
    frozen.env.step(actions);
  }
  issuer::PersistentRouteIntentRegistry lineage;
  frozen.issued = issuer::issue_day_start(
      {&frozen.env, &executor.route_tape(0), 1, generation(7), &lineage, {}, {}});
  require(frozen.issued.issued() && frozen.env.day() == 7 &&
              frozen.env.hour() == 0,
          "real frozen day7 issuer setup failed");
  for (const auto& obligation : frozen.issued.obligations) {
    if (obligation.actor != 0) continue;
    frozen.policies.push_back(
        {obligation.id, true, 1 + (191 - obligation.source_step)});
  }
  return frozen;
}

bridge::Request request_for(FrozenDay7& frozen) {
  return {&frozen.env, 1, 0, generation(7), &frozen.issued,
          frozen.policies, {}, 2'000'000, {}, {}};
}

void real_day7_bridge_and_existing_scheduler_both_verify() {
  auto frozen = frozen_day7();
  const auto request = request_for(frozen);
  const auto result = bridge::issue(request);
  require(result.issued(), std::string("real day7 bridge rejected: ") +
                               bridge::reject_reason_name(result.reject) +
                               ":" + result.diagnostic);
  const auto verified = bridge::verify(request, *result.certificate);
  require(verified.valid && result.certificate->raw_move_tokens == 7 &&
              result.certificate->emitted_raw_moves == 7 &&
              verified.checked_move_tokens == 7,
          "real day7 bridge certificate/MOVE closure failed");

  scheduler::DayPlanRequest native_request;
  native_request.day_start = &frozen.env;
  native_request.player = 1;
  native_request.issuer_generation = generation(7);
  for (const auto& move : frozen.issued.moves)
    if (move.actor == 0) native_request.moves.push_back(move);
  for (const auto& obligation : frozen.issued.obligations)
    if (obligation.actor == 0)
      native_request.obligations.push_back(obligation);
  const auto native_plan = scheduler::plan_day(native_request);
  require(native_plan.planned() && native_plan.certificate &&
              scheduler::verify_day_schedule(
                  native_request, *native_plan.certificate)
                  .valid &&
              native_plan.certificate->move_replays.size() == 7,
          "existing production scheduler verifier changed on same envelope");

  int typed_feed = 0;
  int typed_care = 0;
  for (const auto& slot : result.certificate->slots) {
    typed_feed += slot.unit.op == fastkag::Op::FEED &&
                  slot.unit.item == fastkag::Item::WHEAT;
    typed_care += slot.unit.op == fastkag::Op::CARE &&
                  slot.unit.item >= fastkag::Item::GOOSE &&
                  slot.unit.item <= fastkag::Item::SHEEP;
  }
  require(typed_feed > 0 && typed_care > 0,
          "item-less raw FEED/CARE did not regain typed signed payloads");
  if (std::getenv("MINIMUM_DAMAGE_BRIDGE_DUMP")) {
    for (const auto& slot : result.certificate->slots)
      std::cout << "signed step=" << slot.step
                << " unit=" << static_cast<int>(slot.unit.op)
                << " item=" << static_cast<int>(slot.unit.item)
                << " raw_move=" << slot.raw_move_source_step
                << " obligation=" << slot.obligation_id << '\n';
  }
}

void signed_mapping_tamper_is_rejected_even_after_rehash() {
  auto frozen = frozen_day7();
  const auto request = request_for(frozen);
  const auto result = bridge::issue(request);
  require(result.issued(), "tamper fixture bridge issue failed");
  auto tampered = *result.certificate;
  const auto move_slot = std::find_if(
      tampered.slots.begin(), tampered.slots.end(),
      [](const auto& slot) { return slot.raw_move_source_step >= 0; });
  require(move_slot != tampered.slots.end(), "tamper fixture has no MOVE");
  move_slot->raw_move_source_step += 1;
  tampered.content_hash = bridge::certificate_hash(tampered);
  require(!bridge::verify(request, tampered).valid,
          "rehash accepted changed MOVE source mapping");

  tampered = *result.certificate;
  const auto feed_slot = std::find_if(
      tampered.slots.begin(), tampered.slots.end(),
      [](const auto& slot) { return slot.unit.op == fastkag::Op::FEED; });
  require(feed_slot != tampered.slots.end(), "tamper fixture has no FEED");
  feed_slot->unit.item = fastkag::Item::NONE;
  tampered.content_hash = bridge::certificate_hash(tampered);
  require(!bridge::verify(request, tampered).valid,
          "rehash accepted erased FEED item signature");
}

void itemless_raw_requires_typed_feed_and_care_identity() {
  for (const auto goal : {scheduler::GoalKind::Feed,
                          scheduler::GoalKind::Care}) {
    auto frozen = frozen_day7();
    auto changed = frozen.issued;
    const auto found = std::find_if(
        changed.obligations.begin(), changed.obligations.end(),
        [&](const auto& obligation) {
          return obligation.actor == 0 && obligation.goal == goal;
        });
    require(found != changed.obligations.end(),
            "real day7 lacks item-less identity fixture");
    found->item = fastkag::Item::NONE;
    const bridge::Request request{&frozen.env, 1, 0, generation(7), &changed,
                                  frozen.policies, {}, 2'000'000, {}, {}};
    const auto result = bridge::issue(request);
    require(!result.issued() &&
                result.reject == bridge::RejectReason::MissingTypedIdentity,
            "item-less typed FEED/CARE identity was default-filled");
  }
}

void typed_policy_and_quantity_must_be_exact() {
  auto frozen = frozen_day7();
  auto policies = frozen.policies;
  policies.push_back({999'999, true, 1});
  bridge::Request extra_policy{&frozen.env, 1, 0, generation(7),
                               &frozen.issued, policies, {}, 2'000'000, {}, {}};
  const auto extra_result = bridge::issue(extra_policy);
  require(!extra_result.issued() &&
              extra_result.reject == bridge::RejectReason::MissingTypedIdentity,
          "extra obligation policy was silently ignored");

  auto changed = frozen.issued;
  const auto feed = std::find_if(
      changed.obligations.begin(), changed.obligations.end(),
      [](const auto& obligation) {
        return obligation.actor == 0 &&
               obligation.goal == scheduler::GoalKind::Feed;
      });
  require(feed != changed.obligations.end(), "real day7 lacks FEED fixture");
  feed->quantity = 2;
  bridge::Request wrong_quantity{&frozen.env, 1, 0, generation(7), &changed,
                                 frozen.policies, {}, 2'000'000, {}, {}};
  const auto quantity_result = bridge::issue(wrong_quantity);
  require(!quantity_result.issued() &&
              quantity_result.reject ==
                  bridge::RejectReason::MissingTypedIdentity,
          "typed FEED quantity was silently normalized");
}

struct SyntheticPlant {
  fastkag::Simulator env{{}, 991};
  issuer::IssueResult issued;
  std::vector<bridge::ObligationPolicy> policies;
  std::vector<bridge::PurchaseRetry> purchases;
};

SyntheticPlant synthetic_plant(bool guaranteed) {
  SyntheticPlant fixture;
  const auto tile = fixture.env.farms()[1].farmer;
  for (int step = 0; step < 24; ++step)
    fixture.issued.raw_sources.push_back({0, step, {}});
  fixture.issued.raw_sources[0].action =
      {fastkag::Op::PLANT, fastkag::Item::CARROT, 1};
  scheduler::ProductionObligation plant;
  plant.id = 100;
  plant.actor = 0;
  plant.tile = tile;
  plant.goal = scheduler::GoalKind::CropReady;
  plant.item = fastkag::Item::CARROT;
  plant.quantity = 1;
  plant.earliest_step = 0;
  plant.deadline_step = 23;
  plant.source_step = 0;
  plant.must_finish_today = true;
  fixture.issued.obligations.push_back(plant);
  fixture.policies.push_back({100, true, 8});
  fixture.purchases.push_back(
      {200, fastkag::Op::BUY_SEED, fastkag::Item::CARROT, 1, 0, 0,
       guaranteed, true, true, 5, {100}});
  return fixture;
}

void guaranteed_fill_is_causal_and_unguaranteed_fill_is_debt() {
  auto positive = synthetic_plant(true);
  bridge::Request positive_request{&positive.env, 1, 0, 1234,
                                   &positive.issued, positive.policies,
                                   positive.purchases, 200'000, {}, {}};
  const auto accepted = bridge::issue(positive_request);
  require(accepted.issued() &&
              bridge::verify(positive_request, *accepted.certificate).valid,
          "guaranteed seed retry bridge failed");
  int buy_step = -1;
  int plant_step = -1;
  for (const auto& slot : accepted.certificate->slots) {
    if (slot.market && slot.market->op == fastkag::Op::BUY_SEED)
      buy_step = slot.step;
    if (slot.unit.op == fastkag::Op::PLANT) plant_step = slot.step;
  }
  require(buy_step >= 0 && plant_step > buy_step,
          "same-slot future fill leaked into unit phase");

  auto negative = synthetic_plant(false);
  bridge::Request negative_request{&negative.env, 1, 0, 1234,
                                   &negative.issued, negative.policies,
                                   negative.purchases, 200'000, {}, {}};
  const auto deferred = bridge::issue(negative_request);
  require(deferred.issued() &&
              bridge::verify(negative_request, *deferred.certificate).valid,
          "unguaranteed fill did not produce a verifiable debt plan");
  const auto purchase_debt = std::find_if(
      deferred.certificate->debts.begin(), deferred.certificate->debts.end(),
      [](const auto& debt) { return debt.obligation_id == 200; });
  const auto plant_debt = std::find_if(
      deferred.certificate->debts.begin(), deferred.certificate->debts.end(),
      [](const auto& debt) { return debt.obligation_id == 100; });
  require(purchase_debt != deferred.certificate->debts.end() &&
              plant_debt != deferred.certificate->debts.end() &&
              purchase_debt->selector_reason ==
                  g001::minimum_damage::DebtReason::ResourceUnavailable &&
              plant_debt->selector_reason ==
                  g001::minimum_damage::DebtReason::ResourceUnavailable &&
              std::none_of(deferred.certificate->slots.begin(),
                           deferred.certificate->slots.end(),
                           [](const auto& slot) {
                             return slot.market &&
                                    slot.market->op == fastkag::Op::BUY_SEED;
                           }),
          "unguaranteed fill was treated as inventory evidence");
}

void guaranteed_animal_purchase_flows_through_shed_pickup_and_place() {
  fastkag::Simulator env({}, 992);
  issuer::IssueResult issued;
  for (int step = 0; step < 24; ++step)
    issued.raw_sources.push_back({0, step, {}});
  issued.raw_sources[0].action =
      {fastkag::Op::PICKUP, fastkag::Item::COW, 1};
  issued.raw_sources[1].action =
      {fastkag::Op::BUILD_PASTURE, fastkag::Item::NONE, 1};
  issued.raw_sources[2].action =
      {fastkag::Op::PLACE, fastkag::Item::COW, 1};
  const auto tile = env.farms()[1].farmer;
  scheduler::ProductionObligation pickup;
  pickup.id = 300;
  pickup.actor = 0;
  pickup.tile = tile;
  pickup.goal = scheduler::GoalKind::Pickup;
  pickup.item = fastkag::Item::COW;
  pickup.quantity = 1;
  pickup.earliest_step = 0;
  pickup.deadline_step = 23;
  pickup.source_step = 0;
  pickup.must_finish_today = true;
  scheduler::ProductionObligation build = pickup;
  build.id = 301;
  build.goal = scheduler::GoalKind::BuildPasture;
  build.item = fastkag::Item::NONE;
  build.source_step = 1;
  scheduler::ProductionObligation place = pickup;
  place.id = 302;
  place.goal = scheduler::GoalKind::Place;
  place.source_step = 2;
  place.dependencies = {300, 301};
  issued.obligations = {pickup, build, place};
  std::vector<bridge::ObligationPolicy> policies{
      {300, true, 5}, {301, true, 4}, {302, true, 7}};
  std::vector<bridge::PurchaseRetry> purchases{
      {400, fastkag::Op::BUY_ANIMAL, fastkag::Item::COW, 1, 0, 0, true,
       true, true, 6, {300}}};
  bridge::Request request{&env, 1, 0, 2222, &issued, policies, purchases,
                          200'000, {}, {}};
  const auto result = bridge::issue(request);
  require(result.issued() && bridge::verify(request, *result.certificate).valid,
          "animal purchase->shed->pickup->place bridge failed");
  int buy = -1;
  int pickup_step = -1;
  int place_step = -1;
  for (const auto& slot : result.certificate->slots) {
    if (slot.market && slot.market->op == fastkag::Op::BUY_ANIMAL)
      buy = slot.step;
    if (slot.unit.op == fastkag::Op::PICKUP) pickup_step = slot.step;
    if (slot.unit.op == fastkag::Op::PLACE) place_step = slot.step;
  }
  require(buy >= 0 && pickup_step > buy && place_step > pickup_step,
          "animal purchase bypassed real shed/carried phases");
}

void water_source_compiles_to_plant_then_water() {
  auto fixture = synthetic_plant(true);
  fixture.issued.raw_sources[0].action =
      {fastkag::Op::WATER, fastkag::Item::CARROT, 1};
  bridge::Request request{&fixture.env, 1, 0, 1234, &fixture.issued,
                          fixture.policies, fixture.purchases, 200'000, {}, {}};
  const auto result = bridge::issue(request);
  require(result.issued() && bridge::verify(request, *result.certificate).valid,
          "typed WATER source did not compile");
  int plant = -1;
  int water = -1;
  for (const auto& slot : result.certificate->slots) {
    if (slot.unit.op == fastkag::Op::PLANT) plant = slot.step;
    if (slot.unit.op == fastkag::Op::WATER) water = slot.step;
  }
  require(plant >= 0 && water > plant,
          "empty crop tile WATER recovery skipped PLANT->WATER order");
}

void build_coop_maps_to_build_coop() {
  fastkag::Simulator env({}, 994);
  issuer::IssueResult issued;
  for (int step = 0; step < 24; ++step)
    issued.raw_sources.push_back({0, step, {}});
  issued.raw_sources[0].action =
      {fastkag::Op::BUILD_COOP, fastkag::Item::NONE, 1};
  scheduler::ProductionObligation build;
  build.id = 500;
  build.actor = 0;
  build.tile = env.farms()[1].farmer;
  build.goal = scheduler::GoalKind::BuildCoop;
  build.item = fastkag::Item::NONE;
  build.quantity = 1;
  build.earliest_step = 0;
  build.deadline_step = 23;
  build.source_step = 0;
  build.must_finish_today = true;
  issued.obligations.push_back(build);
  std::vector<bridge::ObligationPolicy> policies{{500, true, 4}};
  bridge::Request request{&env, 1, 0, 4444, &issued, policies, {},
                          200'000, {}, {}};
  const auto result = bridge::issue(request);
  require(result.issued() && bridge::verify(request, *result.certificate).valid,
          "typed BuildCoop source did not compile");
  require(std::count_if(result.certificate->slots.begin(),
                        result.certificate->slots.end(), [](const auto& slot) {
                          return slot.unit.op == fastkag::Op::BUILD_COOP;
                        }) == 1 &&
              std::none_of(result.certificate->slots.begin(),
                           result.certificate->slots.end(), [](const auto& slot) {
                             return slot.unit.op == fastkag::Op::BUILD_PASTURE;
                           }),
          "BuildCoop was swallowed or mapped to BUILD_PASTURE");
}

void hired_hand_uses_its_own_lane() {
  fastkag::Simulator env({}, 993);
  auto& farm = const_cast<fastkag::Farm&>(env.farms()[1]);
  auto& private_state = const_cast<fastkag::PrivateState&>(env.privates()[1]);
  farm.hands.push_back({1, 1});
  private_state.inventories.emplace_back();
  issuer::IssueResult issued;
  for (int step = 0; step < 24; ++step)
    issued.raw_sources.push_back({1, step, {}});
  issued.raw_sources[0].action = {fastkag::Op::EAST};
  issued.moves.push_back({1, 0, issued.raw_sources[0].action});
  bridge::Request request{&env, 1, 1, 3333, &issued, {}, {}, 200'000, {}, {}};
  const auto result = bridge::issue(request);
  require(result.issued() && result.certificate->actor == 1 &&
              result.certificate->raw_move_tokens == 1 &&
              bridge::verify(request, *result.certificate).valid,
          std::string("hired-hand lane was not planned and verified: ") +
              bridge::reject_reason_name(result.reject) + ":" +
              result.diagnostic);
}

void report_single_core_frozen_timing() {
  auto frozen = frozen_day7();
  const auto request = request_for(frozen);
  std::vector<long long> times;
  std::size_t states = 0;
  for (int repeat = 0; repeat < 5; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    const auto result = bridge::issue(request);
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start);
    require(result.issued(), "frozen bridge timing issue failed");
    times.push_back(micros.count());
    states = result.certificate->selector_explored_states;
  }
  std::sort(times.begin(), times.end());
  std::cout << "bridge_frozen_day7_single_core_us_min=" << times.front()
            << " median=" << times[times.size() / 2]
            << " max=" << times.back() << " selector_states=" << states
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--build-coop-only") {
      build_coop_maps_to_build_coop();
      std::cout << "minimum_damage_scheduler_bridge_build_coop_test: PASS\n";
      return 0;
    }
    real_day7_bridge_and_existing_scheduler_both_verify();
    signed_mapping_tamper_is_rejected_even_after_rehash();
    itemless_raw_requires_typed_feed_and_care_identity();
    typed_policy_and_quantity_must_be_exact();
    guaranteed_fill_is_causal_and_unguaranteed_fill_is_debt();
    guaranteed_animal_purchase_flows_through_shed_pickup_and_place();
    water_source_compiles_to_plant_then_water();
    build_coop_maps_to_build_coop();
    hired_hand_uses_its_own_lane();
    report_single_core_frozen_timing();
    std::cout << "minimum_damage_scheduler_bridge_tests: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "minimum_damage_scheduler_bridge_tests: " << error.what()
              << '\n';
    return 1;
  }
}
