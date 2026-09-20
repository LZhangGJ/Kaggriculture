#include "certified_move_fork_audit.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace audit = g001::certified_move_audit;
namespace suffix = g001::production_suffix;
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

Simulator day_one_focal_weed(int focal_player,
                             bool opponent_private_variant = false) {
  fastkag::Config config;
  config.weed_spawn_chance = 1.0;
  Simulator env(config, 9917101);
  const int opponent = 1 - focal_player;
  for (int step = 0; step < 24; ++step) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({});
    actions[1].units.push_back({});
    if (step == 0) {
      actions[static_cast<std::size_t>(focal_player)].market.push_back(
          {Op::BUY_SEED, Item::WHEAT, 1});
      if (opponent_private_variant)
        actions[static_cast<std::size_t>(opponent)].market.push_back(
            {Op::BUY_SEED, Item::WHEAT, 1});
    }
    if (opponent_private_variant && step == 1)
      actions[static_cast<std::size_t>(opponent)].units[0] =
          {Op::PLANT, Item::WHEAT, 1};
    if (opponent_private_variant && step == 2)
      actions[static_cast<std::size_t>(opponent)].units[0] =
          {Op::WATER, Item::WHEAT, 1};
    env.step(actions);
  }
  const auto position = env.farms()[static_cast<std::size_t>(focal_player)].farmer;
  const auto& tile = env.farms()[static_cast<std::size_t>(focal_player)]
      .tiles[static_cast<std::size_t>(
          position.y * env.config().board_size + position.x)];
  check(env.day() == 1 && env.hour() == 0 &&
            tile.kind == TileKind::WEED &&
            env.privates()[static_cast<std::size_t>(focal_player)].seeds[0] == 1,
        "invalid focal day-start fixture");
  return env;
}

std::vector<std::vector<Action>> route() {
  std::vector<std::vector<Action>> raw(24, std::vector<Action>(1));
  raw[0][0] = {Op::WATER, Item::WHEAT, 1};
  raw[1][0] = {Op::WEST, Item::NONE, 1};
  raw[4][0] = {Op::NORTH, Item::NONE, 1};
  raw[5][0] = {Op::DIG, Item::NONE, 1};
  return raw;
}

suffix::IssueResult issue(const Simulator& env, int player,
                          std::uint64_t issuer_generation = 81001) {
  suffix::IssueRequest request;
  request.phase_start = &env;
  request.player = player;
  request.actor = 0;
  request.actor_generation = (std::uint64_t{2} << 32U) | 1U;
  request.issuer_generation = issuer_generation;
  request.raw_units_by_tick = route();
  request.discardable_no_effect_steps = {24};
  const Position origin =
      env.farms()[static_cast<std::size_t>(player)].farmer;
  request.insertions = {
      {101, origin, {Op::DIG, Item::NONE, 1}},
      {102, origin, {Op::PLANT, Item::WHEAT, 1}},
      {103, origin, {Op::WATER, Item::WHEAT, 1}},
  };
  request.budget = {6, 1, 1, 1};
  suffix::ProductionSuffixScheduler scheduler;
  return scheduler.issue(request);
}

struct Trace {
  std::vector<std::unique_ptr<Simulator>> states;
  std::vector<audit::ActualUnitSlot> slots;
};

Trace build_trace(
    const Simulator& initial, int player,
    const std::vector<std::vector<Action>>& raw,
    const suffix::ProductionFrozenDaySuffixCertificate* certificate,
    bool mutate_opponent_each_tick = false) {
  Trace trace;
  trace.states.reserve(24);
  trace.slots.reserve(24);
  Simulator state = initial;
  const int opponent = 1 - player;
  for (int tick = 0; tick < 24; ++tick) {
    const int step = 24 + tick;
    Action emitted = raw[static_cast<std::size_t>(tick)][0];
    int source_step = step;
    if (certificate != nullptr) {
      const auto& proof =
          certificate->proof_slots[static_cast<std::size_t>(tick)];
      emitted = proof.emitted;
      source_step = proof.emitted_source_step;
    }
    std::array<PlayerAction, 2> joint;
    joint[static_cast<std::size_t>(player)].units =
        raw[static_cast<std::size_t>(tick)];
    joint[static_cast<std::size_t>(player)].units[0] = emitted;
    if (mutate_opponent_each_tick) {
      joint[static_cast<std::size_t>(opponent)].units.push_back(
          {tick % 2 == 0 ? Op::WEST : Op::EAST, Item::NONE, 1});
    }
    auto after = std::make_unique<Simulator>(state.preview_unit_phase(joint));
    trace.slots.push_back(
        {step, emitted, {0, source_step, emitted}, after.get()});
    state = *after;
    trace.states.push_back(std::move(after));
  }
  return trace;
}

audit::AuditRequest request(
    const Simulator& env, int player,
    const suffix::ProductionFrozenDaySuffixCertificate* certificate,
    const g001::online_elastic::FrozenDaySuffixCertificate* projection,
    const Trace& trace) {
  return {&env, player, 0, route(), trace.slots, certificate, projection};
}

void pass_through_and_lossy_projection_use_exact_hour() {
  const auto env = day_one_focal_weed(0);
  const auto issued = issue(env, 0);
  check(issued.issued(), "certificate fixture did not issue");
  const auto pass = build_trace(env, 0, route(), nullptr);
  const auto no_certificate =
      audit::audit_candidate(request(env, 0, nullptr, nullptr, pass));
  check(no_certificate.accepted && no_certificate.fixed_hour_fallback &&
            !no_certificate.rich_certificate_verified,
        "pass-through without certificate did not use exact-hour fallback");
  const auto lossy_only = audit::audit_candidate(
      request(env, 0, nullptr, &*issued.owner_projection, pass));
  check(lossy_only.accepted && lossy_only.fixed_hour_fallback &&
            lossy_only.lossy_projection_ignored,
        "lossy projection gained authorization power");
}

void legal_25_to_27_shift_is_authorized() {
  const auto env = day_one_focal_weed(0);
  const auto issued = issue(env, 0);
  const auto shifted = build_trace(env, 0, route(), &*issued.certificate);
  const auto result = audit::audit_candidate(
      request(env, 0, &*issued.certificate, nullptr, shifted));
  check(result.accepted && result.rich_certificate_verified &&
            !result.fixed_hour_fallback && result.authorized_shifts == 1 &&
            result.raw_moves == 2 && result.emitted_moves == 2 &&
            result.pending_at_midnight == 0,
        "valid 25->27 elastic MOVE was rejected");
}

void absent_or_corrupt_rich_proof_cannot_shift() {
  const auto env = day_one_focal_weed(0);
  const auto issued = issue(env, 0);
  const auto shifted = build_trace(env, 0, route(), &*issued.certificate);
  const auto lossy_shift = audit::audit_candidate(
      request(env, 0, nullptr, &*issued.owner_projection, shifted));
  check(!lossy_shift.accepted && lossy_shift.fixed_hour_fallback &&
            lossy_shift.reject == audit::AuditReject::FixedHourMoveMismatch,
        "lossy projection authorized a shifted MOVE");

  auto corrupt = *issued.certificate;
  corrupt.content_hash ^= 1U;
  const auto pass = build_trace(env, 0, route(), nullptr);
  const auto corrupt_exact = audit::audit_candidate(
      request(env, 0, &corrupt, nullptr, pass));
  check(corrupt_exact.accepted && corrupt_exact.fixed_hour_fallback &&
            !corrupt_exact.rich_certificate_verified,
        "corrupt rich proof did not fail closed to exact-hour mode");
  const auto corrupt_shift = audit::audit_candidate(
      request(env, 0, &corrupt, nullptr, shifted));
  check(!corrupt_shift.accepted && corrupt_shift.fixed_hour_fallback,
        "corrupt rich proof authorized a shifted MOVE");
}

void duplicate_missing_crossday_and_direction_tamper_reject() {
  const auto env = day_one_focal_weed(0);
  const auto issued = issue(env, 0);

  auto duplicate = build_trace(env, 0, route(), &*issued.certificate);
  duplicate.slots[4].emitted = duplicate.slots[3].emitted;
  duplicate.slots[4].source = duplicate.slots[3].source;
  check(!audit::audit_candidate(
             request(env, 0, &*issued.certificate, nullptr, duplicate))
             .accepted,
        "duplicate MOVE source was accepted");

  auto missing = build_trace(env, 0, route(), &*issued.certificate);
  missing.slots[3].emitted = {};
  missing.slots[3].source = {0, 27, {}};
  check(!audit::audit_candidate(
             request(env, 0, &*issued.certificate, nullptr, missing))
             .accepted,
        "missing MOVE source was accepted");

  auto crossday = build_trace(env, 0, route(), &*issued.certificate);
  crossday.slots[3].source.source_step = 48;
  check(!audit::audit_candidate(
             request(env, 0, &*issued.certificate, nullptr, crossday))
             .accepted,
        "cross-day MOVE source was accepted");

  auto direction = build_trace(env, 0, route(), &*issued.certificate);
  direction.slots[3].emitted.op = Op::EAST;
  direction.slots[3].source.source_action.op = Op::EAST;
  check(!audit::audit_candidate(
             request(env, 0, &*issued.certificate, nullptr, direction))
             .accepted,
        "MOVE direction tamper was accepted");
}

void post_state_tamper_rejects() {
  const auto env = day_one_focal_weed(0);
  const auto issued = issue(env, 0);
  auto shifted = build_trace(env, 0, route(), &*issued.certificate);
  shifted.slots[2].post_unit_state = shifted.states[1].get();
  const auto result = audit::audit_candidate(
      request(env, 0, &*issued.certificate, nullptr, shifted));
  check(!result.accepted &&
            result.reject == audit::AuditReject::PostStateMismatch,
        "focal post-state tamper was accepted");
}

void player1_opponent_private_changes_do_not_break_focal_proof() {
  const auto issuer_env = day_one_focal_weed(1, false);
  const auto actual_env = day_one_focal_weed(1, true);
  const auto issued = issue(issuer_env, 1, 81002);
  check(issued.issued() &&
            suffix::verify_certificate(*issued.certificate, actual_env,
                                       route()).valid,
        "player1 proof depended on player0 private day-start state");
  const auto shifted = build_trace(actual_env, 1, route(),
                                   &*issued.certificate, true);
  const auto result = audit::audit_candidate(
      request(actual_env, 1, &*issued.certificate, nullptr, shifted));
  check(result.accepted && result.rich_certificate_verified &&
            result.authorized_shifts == 1,
        "player1 opponent-private unit changes broke focal audit");
}

}  // namespace

int main() {
  try {
    pass_through_and_lossy_projection_use_exact_hour();
    legal_25_to_27_shift_is_authorized();
    absent_or_corrupt_rich_proof_cannot_shift();
    duplicate_missing_crossday_and_direction_tamper_reject();
    post_state_tamper_rejects();
    player1_opponent_private_changes_do_not_break_focal_proof();
    std::cout << "certified_move_fork_audit_tests: 6 groups passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "certified_move_fork_audit_tests: " << error.what() << '\n';
    return 1;
  }
}
