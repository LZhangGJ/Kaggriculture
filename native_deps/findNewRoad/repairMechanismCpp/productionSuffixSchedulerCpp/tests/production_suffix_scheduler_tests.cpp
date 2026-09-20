#include "production_suffix_scheduler.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace suffix = g001::production_suffix;
namespace online = g001::online_elastic;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool same_action(const Action& left, const Action& right) {
  return left.op == right.op && left.item == right.item &&
      left.quantity == right.quantity;
}

Simulator day_one_weed_with_seed(bool opponent_variant = false,
                                 int own_seed_quantity = 1) {
  fastkag::Config config;
  config.weed_spawn_chance = 1.0;
  Simulator env(config, 840021);
  for (int step = 0; step < 24; ++step) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({});
    actions[1].units.push_back({});
    if (step == 0) {
      actions[0].market.push_back(
          {Op::BUY_SEED, Item::WHEAT, own_seed_quantity});
      if (opponent_variant)
        actions[1].market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
    }
    if (opponent_variant && step == 1)
      actions[1].units[0] = {Op::PLANT, Item::WHEAT, 1};
    if (opponent_variant && step == 2)
      actions[1].units[0] = {Op::WATER, Item::WHEAT, 1};
    env.step(actions);
  }
  check(env.day() == 1 && env.hour() == 0,
        "fixture did not stop at day boundary");
  const auto position = env.farms()[0].farmer;
  const auto& tile = env.farms()[0].tiles[static_cast<std::size_t>(
      position.y * env.config().board_size + position.x)];
  check(tile.kind == TileKind::WEED &&
            env.privates()[0].seeds[0] == own_seed_quantity,
        "fixture lacks weed/current seed inventory");
  return env;
}

std::vector<std::vector<Action>> pass_route() {
  return std::vector<std::vector<Action>>(24, std::vector<Action>(1));
}

suffix::IssueRequest feasible_request(const Simulator& env) {
  suffix::IssueRequest request;
  request.phase_start = &env;
  request.player = 0;
  request.actor = 0;
  request.actor_generation = (std::uint64_t{2} << 32U) | 1U;
  request.issuer_generation = 70001;
  request.raw_units_by_tick = pass_route();
  request.raw_units_by_tick[0][0] = {Op::WATER, Item::WHEAT, 1};
  request.raw_units_by_tick[1][0] = {Op::WEST, Item::NONE, 1};
  request.raw_units_by_tick[4][0] = {Op::NORTH, Item::NONE, 1};
  request.raw_units_by_tick[5][0] = {Op::DIG, Item::NONE, 1};
  request.discardable_no_effect_steps.insert(24);
  const Position origin = env.farms()[0].farmer;
  request.insertions = {
      {101, origin, {Op::DIG, Item::NONE, 1}},
      {102, origin, {Op::PLANT, Item::WHEAT, 1}},
      {103, origin, {Op::WATER, Item::WHEAT, 1}},
  };
  request.budget = {6, 1, 1, 1};
  return request;
}

void plant_water_delays_move_and_rich_proof_reverifies() {
  const auto env = day_one_weed_with_seed();
  auto request = feasible_request(env);
  suffix::ProductionSuffixScheduler scheduler;
  const auto issued = scheduler.issue(request);
  if (!issued.issued())
    std::cerr << "feasible rejection: "
              << suffix::reject_reason_name(issued.reject) << '\n';
  check(issued.issued() && issued.owner_projection.has_value(),
        "feasible PLANT+WATER suffix was not certified");
  check(issued.delayed_moves == 1 && issued.maximum_move_delay == 2 &&
            issued.hard_obligations == 1 && issued.stamina_used == 6,
        "elastic witness accounting is wrong");
  check(issued.witness[0].emitted.op == Op::DIG &&
            issued.witness[1].emitted.op == Op::PLANT &&
            issued.witness[2].emitted.op == Op::WATER &&
            issued.witness[3].emitted.op == Op::WEST &&
            issued.witness[3].source_step == 25,
        "PLANT+WATER did not delay/replay the MOVE as planned");
  check(issued.witness[4].emitted.op == Op::NORTH &&
            issued.witness[4].source_step == 28 &&
            issued.witness[5].emitted.op == Op::DIG &&
            issued.witness[5].source_step == 29,
        "later MOVE/hard source changed slot or payload");
  check(issued.certificate->authorized_move_replays.size() == 2 &&
            issued.certificate->authorized_move_replays[0].source_step == 25 &&
            issued.certificate->authorized_move_replays[0].emitted_step == 27 &&
            issued.certificate->authorized_move_replays[1].source_step == 28 &&
            issued.certificate->authorized_move_replays[1].emitted_step == 28,
        "explicit source-to-emitted MOVE authorization is wrong");
  const auto verified = suffix::verify_certificate(
      *issued.certificate, env, request.raw_units_by_tick);
  check(verified.valid && verified.checked_slots == 24,
        "independent rich proof replay failed");
  check(issued.owner_projection->content_hash ==
            online::frozen_suffix_hash(*issued.owner_projection),
        "lossy owner projection hash is invalid");

  const auto opponent_changed = day_one_weed_with_seed(true, 1);
  check(suffix::focal_unit_state_fingerprint(env, 0) ==
            suffix::focal_unit_state_fingerprint(opponent_changed, 0),
        "opponent-private state leaked into focal fingerprint");
  check(suffix::verify_certificate(*issued.certificate, opponent_changed,
                                   request.raw_units_by_tick).valid,
        "opponent-private state invalidated a focal proof");
  auto opponent_request = request;
  opponent_request.phase_start = &opponent_changed;
  suffix::ProductionSuffixScheduler opponent_scheduler;
  const auto opponent_issued = opponent_scheduler.issue(opponent_request);
  check(opponent_issued.issued() &&
            opponent_issued.certificate->content_hash ==
                issued.certificate->content_hash,
        "same focal state/route produced opponent-dependent proof bytes");

  const auto own_resource_changed = day_one_weed_with_seed(false, 2);
  check(suffix::focal_unit_state_fingerprint(env, 0) !=
            suffix::focal_unit_state_fingerprint(own_resource_changed, 0) &&
            !suffix::verify_certificate(*issued.certificate,
                                        own_resource_changed,
                                        request.raw_units_by_tick).valid,
        "own resource mutation did not invalidate focal proof");

  auto tampered = *issued.certificate;
  tampered.proof_slots[2].actor_position_before.x++;
  tampered.content_hash = suffix::production_certificate_hash(tampered);
  check(!suffix::verify_certificate(tampered, env,
                                    request.raw_units_by_tick).valid,
        "self-rehashed tampered position proof was accepted");
  tampered = *issued.certificate;
  tampered.authorized_move_replays[0].emitted_step = 26;
  tampered.content_hash = suffix::production_certificate_hash(tampered);
  check(!suffix::verify_certificate(tampered, env,
                                    request.raw_units_by_tick).valid,
        "self-rehashed false MOVE authorization was accepted");
  tampered = *issued.certificate;
  tampered.budget.move_cost = -1;
  tampered.content_hash = suffix::production_certificate_hash(tampered);
  check(!suffix::verify_certificate(tampered, env,
                                    request.raw_units_by_tick).valid,
        "self-rehashed negative stamina cost was accepted");
}

void insufficient_capacity_rejects_without_crossing_midnight() {
  const auto env = day_one_weed_with_seed();
  auto request = feasible_request(env);
  request.issuer_generation = 70002;
  request.raw_units_by_tick = pass_route();
  request.raw_units_by_tick[0][0] = {Op::WATER, Item::WHEAT, 1};
  request.discardable_no_effect_steps = {24};
  request.budget.available_stamina = 100;
  for (int tick = 1; tick < 24; ++tick)
    request.raw_units_by_tick[static_cast<std::size_t>(tick)][0] =
        tick % 2 == 1 ? Action{Op::WEST, Item::NONE, 1}
                      : Action{Op::EAST, Item::NONE, 1};
  suffix::ProductionSuffixScheduler scheduler;
  const auto rejected = scheduler.issue(request);
  if (rejected.reject != suffix::RejectReason::SlotCapacity)
    std::cerr << "capacity rejection: "
              << suffix::reject_reason_name(rejected.reject) << '\n';
  check(!rejected.issued() &&
            rejected.reject == suffix::RejectReason::SlotCapacity,
        "capacity-short suffix did not fail closed");
}

void hour23_and_cross_day_requests_reject() {
  auto env = day_one_weed_with_seed();
  for (int tick = 0; tick < 23; ++tick) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({});
    actions[1].units.push_back({});
    env.step(actions);
  }
  auto request = feasible_request(env);
  suffix::ProductionSuffixScheduler scheduler;
  check(scheduler.issue(request).reject ==
            suffix::RejectReason::NotAtDayStart,
        "hour23 request was not rejected");

  const auto day_start = day_one_weed_with_seed();
  request = feasible_request(day_start);
  request.discardable_no_effect_steps.insert(48);
  check(scheduler.issue(request).reject == suffix::RejectReason::CrossDay,
        "cross-day source declaration was accepted");
}

void move_bytes_and_source_order_are_exact() {
  const auto env = day_one_weed_with_seed();
  auto request = feasible_request(env);
  suffix::ProductionSuffixScheduler scheduler;
  const auto issued = scheduler.issue(request);
  check(issued.issued(), "multi-MOVE fixture did not issue");
  std::vector<std::pair<int, Action>> moves;
  for (const auto& slot : issued.witness)
    if (slot.emitted.op == Op::EAST || slot.emitted.op == Op::SOUTH ||
        slot.emitted.op == Op::NORTH || slot.emitted.op == Op::WEST)
      moves.emplace_back(slot.source_step, slot.emitted);
  check(moves.size() == 2 && moves[0].first == 25 &&
            same_action(moves[0].second,
                        Action{Op::WEST, Item::NONE, 1}) &&
            moves[1].first == 28 &&
            same_action(moves[1].second,
                        Action{Op::NORTH, Item::NONE, 1}),
        "MOVE direction/source/order changed");
}

void failed_precondition_and_unsafe_sink_reject() {
  const auto env = day_one_weed_with_seed();
  auto request = feasible_request(env);
  request.issuer_generation = 70003;
  const Position origin = env.farms()[0].farmer;
  request.insertions = {
      {301, origin, {Op::PLANT, Item::WHEAT, 1}},
  };
  suffix::ProductionSuffixScheduler scheduler;
  check(scheduler.issue(request).reject ==
            suffix::RejectReason::PreconditionsUnsatisfied,
        "PLANT-on-weed precondition failure was not rejected");

  request = feasible_request(env);
  request.issuer_generation = 70004;
  request.raw_units_by_tick[0][0] = {Op::DIG, Item::NONE, 1};
  request.insertions.clear();
  check(scheduler.issue(request).reject == suffix::RejectReason::UnsafeSink,
        "effectful DIG was accepted as a no-effect sink");

  request = feasible_request(env);
  request.issuer_generation = 70005;
  request.raw_units_by_tick = pass_route();
  request.raw_units_by_tick[0][0] =
      {Op::FERTILIZE, Item::FERTILIZER, 1};
  request.discardable_no_effect_steps.clear();
  request.insertions.clear();
  request.budget.available_stamina = 24;
  check(scheduler.issue(request).reject ==
            suffix::RejectReason::PreconditionsUnsatisfied,
        "missing carried fertilizer did not reject a hard obligation");
}

void stamina_and_irrevocable_issuance_are_enforced() {
  const auto env = day_one_weed_with_seed();
  auto request = feasible_request(env);
  request.budget.available_stamina = 5;
  suffix::ProductionSuffixScheduler scheduler;
  check(scheduler.issue(request).reject ==
            suffix::RejectReason::StaminaExceeded,
        "insufficient stamina was accepted");

  request.budget.available_stamina = 6;
  const auto first = scheduler.issue(request);
  check(first.issued() && scheduler.was_issued(0, 1, 0),
        "valid certificate was not recorded");
  check(scheduler.issue(request).reject == suffix::RejectReason::AlreadyIssued,
        "issuer replaced an irrevocable day certificate");
}

}  // namespace

int main() {
  try {
    plant_water_delays_move_and_rich_proof_reverifies();
    insufficient_capacity_rejects_without_crossing_midnight();
    hour23_and_cross_day_requests_reject();
    move_bytes_and_source_order_are_exact();
    failed_precondition_and_unsafe_sink_reject();
    stamina_and_irrevocable_issuance_are_enforced();
    std::cout << "production_suffix_scheduler_tests: 6 groups passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "production_suffix_scheduler_tests: " << error.what()
              << '\n';
    return 1;
  }
}
