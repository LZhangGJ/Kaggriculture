#include "g001_real_weed_move_owner.hpp"

#include "production_suffix_scheduler.hpp"

#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>

namespace g001::real_weed_move_owner {
namespace {

bool same_action(fastkag::Action lhs, fastkag::Action rhs) {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool movement(fastkag::Op op) {
  return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
         op == fastkag::Op::EAST || op == fastkag::Op::WEST;
}

void add_hash(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (8 * byte)) & 255U;
    hash *= 1099511628211ULL;
  }
}

fastkag::Position actor_position(const fastkag::Simulator& env, int player,
                                 int actor) {
  const auto& farm = env.farms()[player];
  return actor == 0 ? farm.farmer : farm.hands.at(actor - 1);
}

const fastkag::Tile& tile_at(const fastkag::Simulator& env, int player,
                             fastkag::Position position) {
  return env.farms()[player].tiles[
      position.y * env.config().board_size + position.x];
}

std::uint64_t receipt_fingerprint(const fastkag::Simulator& env, int player,
                                  int actor, fastkag::Action action) {
  if (actor < 0 || actor > static_cast<int>(env.farms()[player].hands.size()))
    return 0;
  std::uint64_t hash = 1469598103934665603ULL;
  const auto position = actor_position(env, player, actor);
  add_hash(hash, static_cast<std::uint16_t>(position.x));
  add_hash(hash, static_cast<std::uint16_t>(position.y));
  add_hash(hash, static_cast<std::uint8_t>(action.op));
  add_hash(hash, static_cast<std::uint8_t>(action.item));
  add_hash(hash, static_cast<std::uint32_t>(action.quantity));
  if (movement(action.op)) return hash;
  const auto& tile = tile_at(env, player, position);
  add_hash(hash, static_cast<std::uint8_t>(tile.kind));
  add_hash(hash, static_cast<std::uint8_t>(tile.crop));
  add_hash(hash, static_cast<std::uint8_t>(tile.animal));
  if (action.op == fastkag::Op::FEED) add_hash(hash, tile.fed_today);
  if (action.op == fastkag::Op::CARE) add_hash(hash, tile.cared_today);
  if (action.op == fastkag::Op::PICKUP || action.op == fastkag::Op::PLACE) {
    const int item = static_cast<int>(action.item);
    const auto& inventories = env.privates()[player].inventories;
    add_hash(hash, item >= 0 && item < fastkag::N_ITEMS &&
                           actor < static_cast<int>(inventories.size())
                       ? inventories[actor][item]
                       : 0);
  }
  return hash;
}

void settle_pending(const fastkag::Simulator& env, int player, State& state) {
  if (!state.pending.active) return;
  ++state.audit.receipt_checks;
  if (movement(state.pending.action.op)) ++state.audit.movement_receipts;
  else ++state.audit.production_receipts;
  const auto actual = receipt_fingerprint(
      env, player, state.pending.actor, state.pending.action);
  if (actual != state.pending.expected_fingerprint) {
    ++state.audit.receipt_failures;
    if (state.audit.first_receipt_failure_step < 0)
      state.audit.first_receipt_failure_step = state.pending.step;
  }
  state.pending = {};
}

bool has_day_start_weed_submission(
    const fastkag::NativeTeammateExecutor& executor,
    const fastkag::Simulator& env, int player, int route) {
  if (env.hour() != 0) return false;
  std::vector<fastkag::Position> projected{env.farms()[player].farmer};
  projected.insert(projected.end(), env.farms()[player].hands.begin(),
                   env.farms()[player].hands.end());
  const auto& tape = executor.route_tape(route);
  const int end = std::min(env.step_count() + 23,
                           static_cast<int>(tape.size()) - 1);
  for (int source = env.step_count(); source <= end; ++source) {
    const auto& units = tape[source].units;
    for (std::size_t actor = 0;
         actor < projected.size() && actor < units.size(); ++actor) {
      const auto action = units[actor];
      if (action.op == fastkag::Op::BUILD_PASTURE &&
          tile_at(env, player, projected[actor]).kind ==
              fastkag::TileKind::WEED)
        return true;
      if (action.op == fastkag::Op::NORTH) --projected[actor].y;
      else if (action.op == fastkag::Op::SOUTH) ++projected[actor].y;
      else if (action.op == fastkag::Op::WEST) --projected[actor].x;
      else if (action.op == fastkag::Op::EAST) ++projected[actor].x;
    }
  }
  return false;
}

void install_day(const fastkag::NativeTeammateExecutor& executor,
                 const fastkag::Simulator& env, int player, int route,
                 State& state, std::uint64_t generation,
                 const fastkag::PlayerAction& native) {
  finish_day(state);
  state.active_day = env.day();
  state.audit = {};
  state.audit.day = env.day();
  state.audit.start_fingerprint =
      production_suffix::focal_unit_state_fingerprint(env, player);
  const auto& submissions = state.native.experimental_stationary_obligations;
  state.audit.stationary_submissions = static_cast<int>(submissions.size());
  state.audit.stationary = submissions;
  if (submissions.empty()) return;

  std::set<int> submitted_actors;
  for (const auto& value : submissions) submitted_actors.insert(value.actor);
  // This fork intentionally admits one owner only. Multiple weed actors are
  // an unimplemented overlay and fail closed before any action is rewritten.
  if (submitted_actors.size() != 1) {
    state.audit.unowned_overlay_suppressed = true;
    return;
  }
  const int actor = *submitted_actors.begin();
  if (actor < 0 || actor >= static_cast<int>(native.units.size())) {
    state.audit.unowned_overlay_suppressed = true;
    return;
  }

  const auto issued = day_start_issuer::issue_day_start(
      {&env, &executor.route_tape(route), player, generation,
       &state.persistent_lineage, {}, {}});
  if (!issued.issued()) {
    state.audit.admission_rejected = true;
    state.audit.unowned_overlay_suppressed = true;
    return;
  }
  obligation_day::DayPlanRequest request;
  request.day_start = &env;
  request.player = player;
  request.issuer_generation = generation;
  for (const auto& move : issued.moves)
    if (move.actor == actor) request.moves.push_back(move);
  for (const auto& obligation : issued.obligations)
    if (obligation.actor == actor) request.obligations.push_back(obligation);
  if (state.policy_deferred_authorization) {
    const auto& authorization = *state.policy_deferred_authorization;
    const auto source = std::find_if(
        issued.raw_sources.begin(), issued.raw_sources.end(),
        [&](const auto& value) {
          return value.actor == authorization.actor &&
                 value.source_step == authorization.source_step;
        });
    const auto selected = std::find_if(
        request.obligations.begin(), request.obligations.end(),
        [&](const auto& value) {
          return value.id == authorization.obligation_id &&
                 value.source_step == authorization.source_step &&
                 value.actor == authorization.actor &&
                 value.goal == authorization.goal &&
                 value.item == authorization.item &&
                 value.tile.x == authorization.tile.x &&
                 value.tile.y == authorization.tile.y &&
                 value.quantity == authorization.quantity;
        });
    if (authorization.content_hash !=
            debt_selection_identity_hash(authorization) ||
        authorization.player != player || authorization.day != env.day() ||
        authorization.issuer_generation != generation ||
        source == issued.raw_sources.end() ||
        !same_action(source->action, authorization.source_action) ||
        selected == request.obligations.end() ||
        authorization.obligation_content_hash !=
            production_obligation_identity_hash(*selected) ||
        selected->goal == obligation_day::GoalKind::BuildPasture) {
      state.audit.admission_rejected = true;
      state.audit.unowned_overlay_suppressed = true;
      return;
    }
    selected->policy_deferred = true;
    state.audit.policy_deferred_source_step = authorization.source_step;
    state.audit.debt_selection_identity = authorization;
  }

  // Global fully_proven() is intentionally not the admission gate: future
  // actors that do not exist at this day start are outside this fork's
  // authority.  Instead prove an exact 24-source partition for the one owned
  // actor. Every source is uniquely MOVE, typed obligation, or explicit PASS.
  for (const auto& source : issued.raw_sources) {
    if (source.actor != actor) continue;
    ++state.audit.owned_source_coverage_total;
    const int move_matches = static_cast<int>(std::count_if(
        request.moves.begin(), request.moves.end(), [&](const auto& value) {
          return value.actor == actor &&
                 value.source_step == source.source_step &&
                 same_action(value.action, source.action);
        }));
    const int obligation_matches = static_cast<int>(std::count_if(
        request.obligations.begin(), request.obligations.end(),
        [&](const auto& value) {
          return value.actor == actor &&
                 value.source_step == source.source_step;
        }));
    const int unsupported_matches = static_cast<int>(std::count_if(
        issued.unsupported.begin(), issued.unsupported.end(),
        [&](const auto& value) {
          return value.actor == actor &&
                 value.source_step == source.source_step;
        }));
    const int pass_matches = source.action.op == fastkag::Op::PASS ? 1 : 0;
    const int classifications = move_matches + obligation_matches +
                                unsupported_matches + pass_matches;
    state.audit.owned_source_move_coverage += move_matches;
    state.audit.owned_source_obligation_coverage += obligation_matches;
    state.audit.owned_source_pass_coverage += pass_matches;
    state.audit.owned_source_unsupported += unsupported_matches;
    if (classifications == 0) ++state.audit.owned_source_missing;
    if (classifications > 1) ++state.audit.owned_source_duplicates;
  }
  if (state.audit.owned_source_coverage_total != 24 ||
      state.audit.owned_source_missing != 0 ||
      state.audit.owned_source_duplicates != 0 ||
      state.audit.owned_source_unsupported != 0 ||
      state.audit.owned_source_move_coverage +
              state.audit.owned_source_obligation_coverage +
              state.audit.owned_source_pass_coverage !=
          24) {
    state.audit.admission_rejected = true;
    state.audit.unowned_overlay_suppressed = true;
    return;
  }

  std::set<std::uint64_t> ids;
  for (const auto& obligation : request.obligations) ids.insert(obligation.id);
  for (const auto& obligation : request.obligations) {
    for (const auto dependency : obligation.dependencies) {
      if (!ids.contains(dependency)) {
        state.audit.admission_rejected = true;
        state.audit.unowned_overlay_suppressed = true;
        return;
      }
    }
  }
  const bool declared = std::all_of(
      submissions.begin(), submissions.end(), [&](const auto& submission) {
        return std::any_of(request.obligations.begin(),
                           request.obligations.end(), [&](const auto& value) {
          return value.actor == submission.actor &&
                 value.tile.x == submission.tile.x &&
                 value.tile.y == submission.tile.y &&
                 value.goal == obligation_day::GoalKind::BuildPasture;
        });
      });
  if (!declared) {
    state.audit.admission_rejected = true;
    state.audit.unowned_overlay_suppressed = true;
    return;
  }
  state.audit.scoped_manifest_proven = true;

  state.audit.owned_actor = actor;
  state.audit.move_tokens = static_cast<int>(request.moves.size());
  state.audit.raw_move_tokens = request.moves;
  state.audit.obligations = static_cast<int>(request.obligations.size());
  state.plan = obligation_day::plan_day(request);
  state.audit.planned = state.plan.planned();
  state.audit.debts = static_cast<int>(state.plan.debts.size());
  state.audit.explicit_debts = state.plan.debts;
  state.audit.completed = static_cast<int>(state.plan.completed.size());
  state.audit.move_delays = state.plan.move_delays;
  if (!state.plan.certificate) return;
  const auto verified = obligation_day::verify_day_schedule(
      request, *state.plan.certificate);
  state.audit.certificate_valid = verified.valid;
  state.audit.verify_failure = verified.failure_reason;
  state.audit.verify_failure_step = verified.failure_step;
  state.audit.move_replays = state.plan.certificate->move_replays;
  for (const auto& obligation : request.obligations) {
    const auto status = std::find_if(
        state.plan.certificate->obligation_statuses.begin(),
        state.plan.certificate->obligation_statuses.end(),
        [&](const auto& value) {
          return value.obligation_id == obligation.id;
        });
    if (status == state.plan.certificate->obligation_statuses.end()) continue;
    state.audit.obligation_audits.push_back(
        {obligation, status->disposition, status->assigned_actor,
         status->remaining_transitions, status->transition_steps});
  }
  std::set<std::uint64_t> stationary_ids;
  for (const auto& obligation : request.obligations) {
    if (obligation.goal != obligation_day::GoalKind::BuildPasture) continue;
    for (const auto& submission : submissions) {
      if (obligation.actor == submission.actor &&
          obligation.tile.x == submission.tile.x &&
          obligation.tile.y == submission.tile.y)
        stationary_ids.insert(obligation.id);
    }
  }
  for (const auto id : stationary_ids) {
    state.audit.stationary_completed +=
        std::find(state.plan.completed.begin(), state.plan.completed.end(), id) !=
        state.plan.completed.end();
    state.audit.stationary_debts +=
        std::any_of(state.plan.debts.begin(), state.plan.debts.end(),
                    [&](const auto& debt) { return debt.obligation_id == id; });
  }
  if (!verified.valid || state.audit.move_replays.size() != request.moves.size())
    return;
  state.active = true;
  state.owned_actor = actor;
  state.audit.exclusive_owner_admitted = true;
}

}  // namespace

std::uint64_t debt_selection_identity_hash(
    const DebtSelectionIdentity& authorization) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  add_hash(hash, authorization.player);
  add_hash(hash, authorization.day);
  add_hash(hash, authorization.issuer_generation);
  add_hash(hash, authorization.obligation_id);
  add_hash(hash, authorization.obligation_content_hash);
  add_hash(hash, authorization.actor);
  add_hash(hash, authorization.source_step);
  add_hash(hash, static_cast<std::uint8_t>(authorization.source_action.op));
  add_hash(hash, static_cast<std::uint8_t>(authorization.source_action.item));
  add_hash(hash, static_cast<std::uint32_t>(authorization.source_action.quantity));
  add_hash(hash, static_cast<std::uint8_t>(authorization.goal));
  add_hash(hash, static_cast<std::uint8_t>(authorization.item));
  add_hash(hash, static_cast<std::uint16_t>(authorization.tile.x));
  add_hash(hash, static_cast<std::uint16_t>(authorization.tile.y));
  add_hash(hash, authorization.quantity);
  return hash;
}

std::uint64_t production_obligation_identity_hash(
    const obligation_day::ProductionObligation& obligation) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  add_hash(hash, obligation.id);
  add_hash(hash, obligation.actor);
  add_hash(hash, static_cast<std::uint16_t>(obligation.tile.x));
  add_hash(hash, static_cast<std::uint16_t>(obligation.tile.y));
  add_hash(hash, static_cast<std::uint8_t>(obligation.goal));
  add_hash(hash, static_cast<std::uint8_t>(obligation.item));
  add_hash(hash, obligation.quantity);
  add_hash(hash, static_cast<std::uint8_t>(obligation.resource.item));
  add_hash(hash, obligation.resource.seed_quantity);
  add_hash(hash, obligation.resource.shed_quantity);
  add_hash(hash, obligation.resource.carried_quantity);
  add_hash(hash, obligation.dependencies.size());
  for (const auto dependency : obligation.dependencies)
    add_hash(hash, dependency);
  add_hash(hash, obligation.earliest_step);
  add_hash(hash, obligation.deadline_step);
  add_hash(hash, obligation.priority);
  add_hash(hash, obligation.must_finish_today);
  add_hash(hash, obligation.repair);
  add_hash(hash, obligation.source_step);
  add_hash(hash, obligation.policy_deferred);
  return hash;
}

std::uint64_t remaining_day_certificate_hash(
    const RemainingDayCertificate& certificate) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  add_hash(hash, certificate.player);
  add_hash(hash, certificate.day);
  add_hash(hash, certificate.actor);
  add_hash(hash, certificate.resign_step);
  add_hash(hash, certificate.issuer_generation);
  add_hash(hash, certificate.prior_certificate_hash);
  add_hash(hash, certificate.observation_fingerprint);
  add_hash(hash, certificate.slots.size());
  for (const auto& slot : certificate.slots) {
    add_hash(hash, slot.step);
    add_hash(hash, static_cast<std::uint8_t>(slot.action.op));
    add_hash(hash, static_cast<std::uint8_t>(slot.action.item));
    add_hash(hash, static_cast<std::uint32_t>(slot.action.quantity));
    add_hash(hash, slot.source.actor);
    add_hash(hash, slot.source.source_step);
    add_hash(hash, static_cast<std::uint8_t>(slot.source.source_action.op));
    add_hash(hash, static_cast<std::uint8_t>(slot.source.source_action.item));
    add_hash(hash,
             static_cast<std::uint32_t>(slot.source.source_action.quantity));
    add_hash(hash, slot.obligation_id);
    add_hash(hash, slot.pre_fingerprint);
    add_hash(hash, slot.post_fingerprint);
  }
  add_hash(hash, certificate.remaining_moves.size());
  for (const auto& move : certificate.remaining_moves) {
    add_hash(hash, move.actor);
    add_hash(hash, move.source_step);
    add_hash(hash, move.emitted_step);
    add_hash(hash, static_cast<std::uint8_t>(move.action.op));
    add_hash(hash, static_cast<std::uint8_t>(move.action.item));
    add_hash(hash, static_cast<std::uint32_t>(move.action.quantity));
  }
  add_hash(hash, certificate.terminal_statuses.size());
  for (const auto& status : certificate.terminal_statuses) {
    add_hash(hash, status.obligation_id);
    add_hash(hash, static_cast<std::uint8_t>(status.disposition));
    add_hash(hash, status.assigned_actor);
    add_hash(hash, status.remaining_transitions);
    add_hash(hash, status.transition_steps.size());
    for (const int step : status.transition_steps) add_hash(hash, step);
  }
  add_hash(hash, certificate.obligation_identity_hashes.size());
  for (const auto identity : certificate.obligation_identity_hashes)
    add_hash(hash, identity);
  return hash;
}

RemainingResignResult resign_remaining_day(const fastkag::Simulator& env,
                                           int player,
                                           const State& state) {
  RemainingResignResult result;
  if (!state.active || state.active_day != env.day() ||
      !state.plan.certificate || state.owned_actor < 0) {
    result.reject = RemainingResignReject::NoActiveOwner;
    return result;
  }
  const auto& prior = *state.plan.certificate;
  const int actor = state.owned_actor;
  const int step = env.step_count();
  const int tick = env.hour();
  if (prior.player != player || prior.day != env.day() ||
      prior.content_hash != obligation_day::day_schedule_certificate_hash(prior) ||
      prior.slots.size() != 24 || tick < 0 || tick >= 24 ||
      actor >= static_cast<int>(state.plan.manifest.size()) ||
      actor >= static_cast<int>(state.plan.sources.size()) ||
      state.plan.manifest[actor].size() != 24 ||
      state.plan.sources[actor].size() != 24) {
    result.reject = RemainingResignReject::InvalidEnvelope;
    return result;
  }

  RemainingDayCertificate certificate;
  certificate.player = player;
  certificate.day = env.day();
  certificate.actor = actor;
  certificate.resign_step = step;
  certificate.issuer_generation = prior.issuer_generation;
  certificate.prior_certificate_hash = prior.content_hash;
  certificate.observation_fingerprint =
      production_suffix::focal_unit_state_fingerprint(env, player);
  certificate.terminal_statuses = prior.obligation_statuses;
  for (const auto& audit : state.audit.obligation_audits)
    certificate.obligation_identity_hashes.push_back(
        production_obligation_identity_hash(audit.obligation));
  if (certificate.terminal_statuses.size() !=
          state.audit.obligation_audits.size() ||
      certificate.obligation_identity_hashes.size() !=
          state.audit.obligation_audits.size()) {
    result.reject = RemainingResignReject::ObligationIdentity;
    return result;
  }
  for (const auto& status : certificate.terminal_statuses) {
    const auto found = std::find_if(
        state.audit.obligation_audits.begin(),
        state.audit.obligation_audits.end(), [&](const auto& audit) {
          return audit.obligation.id == status.obligation_id &&
                 audit.disposition == status.disposition &&
                 audit.assigned_actor == status.assigned_actor &&
                 audit.remaining_transitions == status.remaining_transitions &&
                 audit.transition_steps == status.transition_steps;
        });
    if (found == state.audit.obligation_audits.end()) {
      result.reject = RemainingResignReject::ObligationIdentity;
      return result;
    }
  }

  for (const auto& token : state.audit.raw_move_tokens) {
    const auto matches = static_cast<int>(std::count_if(
        prior.move_replays.begin(), prior.move_replays.end(),
        [&](const auto& replay) {
          return replay.actor == token.actor &&
                 replay.source_step == token.source_step &&
                 same_action(replay.action, token.action);
        }));
    if (matches != 1) {
      result.reject = RemainingResignReject::MoveCoverage;
      return result;
    }
  }
  for (const auto& replay : prior.move_replays) {
    if (replay.actor == actor && replay.emitted_step >= step)
      certificate.remaining_moves.push_back(replay);
  }
  int last_source = -1;
  int last_emitted = step - 1;
  for (const auto& replay : certificate.remaining_moves) {
    if (replay.source_step <= last_source || replay.emitted_step <= last_emitted ||
        replay.emitted_step < replay.source_step) {
      result.reject = RemainingResignReject::MoveOrder;
      return result;
    }
    last_source = replay.source_step;
    last_emitted = replay.emitted_step;
  }

  fastkag::Simulator simulated = env;
  std::size_t remaining_move_cursor = 0;
  for (int current_tick = tick; current_tick < 24; ++current_tick) {
    const int logical_step = env.day() * 24 + current_tick;
    const auto action = state.plan.manifest[actor][current_tick];
    const auto source = state.plan.sources[actor][current_tick];
    const auto& old_slot = prior.slots[current_tick];
    if (old_slot.step != logical_step ||
        actor >= static_cast<int>(old_slot.actions.size()) ||
        actor >= static_cast<int>(old_slot.sources.size()) ||
        actor >= static_cast<int>(old_slot.obligation_ids.size()) ||
        !same_action(action, old_slot.actions[actor]) ||
        source.actor != old_slot.sources[actor].actor ||
        source.source_step != old_slot.sources[actor].source_step ||
        !same_action(source.source_action,
                     old_slot.sources[actor].source_action)) {
      result.reject = RemainingResignReject::SlotBinding;
      return result;
    }
    if (movement(action.op)) {
      if (remaining_move_cursor >= certificate.remaining_moves.size()) {
        result.reject = RemainingResignReject::MoveCoverage;
        return result;
      }
      const auto& move = certificate.remaining_moves[remaining_move_cursor++];
      if (move.emitted_step != logical_step ||
          move.source_step != source.source_step ||
          !same_action(move.action, action)) {
        result.reject = RemainingResignReject::MoveCoverage;
        return result;
      }
    } else if (source.source_step != -1) {
      result.reject = RemainingResignReject::SlotBinding;
      return result;
    }

    std::array<fastkag::PlayerAction, 2> joint;
    joint[player].units.resize(
        static_cast<std::size_t>(actor) + 1U);
    joint[player].units[actor] = action;
    std::array<fastkag::PlayerAction, 2> pass;
    pass[player].units.resize(static_cast<std::size_t>(actor) + 1U);
    const auto before_position = actor_position(simulated, player, actor);
    const auto after = simulated.preview_unit_phase(joint);
    const auto pass_after = simulated.preview_unit_phase(pass);
    if (action.op != fastkag::Op::PASS) {
      const auto effect = production_suffix::focal_unit_state_fingerprint(
                              after, player, logical_step) !=
                          production_suffix::focal_unit_state_fingerprint(
                              pass_after, player, logical_step);
      if (!effect) {
        result.reject = RemainingResignReject::PhysicalNoProgress;
        return result;
      }
      if (movement(action.op)) {
        auto expected = before_position;
        if (action.op == fastkag::Op::NORTH) --expected.y;
        else if (action.op == fastkag::Op::SOUTH) ++expected.y;
        else if (action.op == fastkag::Op::WEST) --expected.x;
        else if (action.op == fastkag::Op::EAST) ++expected.x;
        const auto actual = actor_position(after, player, actor);
        if (actual.x != expected.x || actual.y != expected.y) {
          result.reject = RemainingResignReject::PhysicalNoProgress;
          return result;
        }
      } else if (old_slot.obligation_ids[actor] == 0) {
        result.reject = RemainingResignReject::ObligationIdentity;
        return result;
      }
    }
    certificate.slots.push_back(
        {logical_step, action, source, old_slot.obligation_ids[actor],
         production_suffix::focal_unit_state_fingerprint(
             simulated, player, logical_step),
         production_suffix::focal_unit_state_fingerprint(after, player,
                                                         logical_step)});
    simulated = after;
  }
  if (remaining_move_cursor != certificate.remaining_moves.size()) {
    result.reject = RemainingResignReject::MoveCoverage;
    return result;
  }
  certificate.content_hash = remaining_day_certificate_hash(certificate);
  result.certificate = std::move(certificate);
  return result;
}

const char* remaining_resign_reject_name(
    RemainingResignReject reject) noexcept {
  switch (reject) {
    case RemainingResignReject::None: return "none";
    case RemainingResignReject::NoActiveOwner: return "no_active_owner";
    case RemainingResignReject::InvalidEnvelope: return "invalid_envelope";
    case RemainingResignReject::SlotBinding: return "slot_binding";
    case RemainingResignReject::MoveCoverage: return "move_coverage";
    case RemainingResignReject::MoveOrder: return "move_order";
    case RemainingResignReject::ObligationIdentity:
      return "obligation_identity";
    case RemainingResignReject::PhysicalNoProgress:
      return "physical_no_progress";
  }
  return "unknown";
}

void observe_final_provider_action(const fastkag::Simulator& env, int player,
                                   const fastkag::PlayerAction& action,
                                   State& state) {
  if (player < 0 || player >= 2) return;
  std::vector<fastkag::Position> positions{env.farms()[player].farmer};
  positions.insert(positions.end(), env.farms()[player].hands.begin(),
                   env.farms()[player].hands.end());
  for (std::size_t actor = 0;
       actor < positions.size() && actor < action.units.size(); ++actor) {
    (void)state.persistent_lineage.record_exact_source(
        static_cast<int>(actor), env.step_count(), positions[actor],
        action.units[actor]);
  }
}

void finish_day(State& state) {
  if (state.active_day >= 0) {
    state.completed_days.push_back(state.audit);
    if (state.policy_deferred_authorization &&
        state.policy_deferred_authorization->day == state.active_day)
      state.policy_deferred_authorization.reset();
  }
  state.active_day = -1;
  state.owned_actor = -1;
  state.active = false;
  state.plan = {};
  state.remaining_certificate.reset();
  state.audit = {};
}

fastkag::PlayerAction action_external(
    const fastkag::NativeTeammateExecutor& executor,
    const fastkag::Simulator& env, int player, int route, State& state,
    std::uint64_t issuer_generation) {
  settle_pending(env, player, state);
  fastkag::NativeRepairOptions options;
  options.weed_obligation_day_owner =
      (state.active && state.active_day == env.day()) ||
      has_day_start_weed_submission(executor, env, player, route);
  auto candidate_native = state.native;
  auto output = executor.action_external(
      env, player, route, candidate_native,
      fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, false,
      options, nullptr);
  if (state.overlay_test_injection &&
      state.overlay_test_injection->step == env.step_count() &&
      state.overlay_test_injection->actor >= 0 &&
      state.overlay_test_injection->actor <
          static_cast<int>(output.units.size()))
    output.units[state.overlay_test_injection->actor] =
        state.overlay_test_injection->action;
  if (state.overlay_test_injection &&
      state.overlay_test_injection->step == env.step_count() &&
      state.overlay_test_injection->mutate_candidate_native_before_resign)
    candidate_native.last_step = 999999;
  if (env.hour() == 0) {
    state.native = candidate_native;
    install_day(executor, env, player, route, state, issuer_generation, output);
  }
  if (!state.active || state.active_day != env.day()) {
    state.native = std::move(candidate_native);
    observe_final_provider_action(env, player, output, state);
    return output;
  }

  const int tick = env.hour();
  const int actor = state.owned_actor;
  const auto& raw = executor.route_tape(route)[env.step_count()].units;
  const bool overlay_conflict =
      actor >= static_cast<int>(output.units.size()) ||
      actor >= static_cast<int>(raw.size()) ||
      !same_action(output.units[actor], raw[actor]);
  State resign_view = state;
  if (overlay_conflict && state.overlay_test_injection &&
      state.overlay_test_injection->step == env.step_count() &&
      state.overlay_test_injection->tamper_move_source_before_resign &&
      actor < static_cast<int>(resign_view.plan.sources.size()) &&
      tick < static_cast<int>(resign_view.plan.sources[actor].size()))
    ++resign_view.plan.sources[actor][tick].source_step;
  // A future unit-only projection is not carried across market/town/decay.
  // Re-sign from every real pre-action observation, even without an overlay.
  const auto resigned = resign_remaining_day(env, player, resign_view);
  state.audit.last_remaining_resign_reject = resigned.reject;
  if (overlay_conflict) {
    ++state.audit.overlay_conflicts;
  }
  if (!resigned.issued()) {
    // The candidate native state and scheduler suffix have not been
    // committed. No action is returned: this is an explicit fail-stop, not
    // a claim that an already-applied action was rolled back.
    state.audit.overlay_conflict_fail_stop = overlay_conflict;
    state.audit.remaining_resign_fail_stop = true;
    throw std::runtime_error(
        "obligation-day remaining suffix resign rejected");
  }
  if (overlay_conflict) {
    // This owner ABI has no proposal/commit callback wired into the wrapper.
    // Even a valid suffix resign cannot prove that proposal-side native
    // mutations are independent of the replaced action. The independent
    // default-off final-action adapter is exercised by the continuous audit,
    // but a caller that enters here without that transaction still fail-stops.
    state.audit.native_proposal_discarded = true;
    state.audit.overlay_conflict_fail_stop = true;
    state.audit.remaining_resign_fail_stop = true;
    throw std::runtime_error(
        "native overlay requires caller final-action transaction");
  }
  state.native = std::move(candidate_native);
  state.remaining_certificate = resigned.certificate;
  ++state.audit.remaining_resigns;
  ++state.audit.interphase_resigns;
  if (actor < static_cast<int>(output.units.size()) &&
      actor < static_cast<int>(state.plan.manifest.size()) &&
      tick < static_cast<int>(state.plan.manifest[actor].size())) {
    output.units[actor] = state.plan.manifest[actor][tick];
    if (movement(output.units[actor].op)) ++state.audit.move_emitted;
    else if (output.units[actor].op != fastkag::Op::PASS)
      ++state.audit.production_emitted;
    if (output.units[actor].op != fastkag::Op::PASS) {
      const auto& slot = state.plan.certificate->slots[tick];
      const auto& source = state.plan.sources[actor][tick];
      if (!same_action(slot.actions[actor], output.units[actor]) ||
          !same_action(source.source_action, output.units[actor]))
        throw std::runtime_error("owner action lost certificate binding");
      std::array<fastkag::PlayerAction, 2> joint;
      joint[player] = output;
      const auto predicted = env.preview_unit_phase(joint);
      state.pending.active = true;
      state.pending.step = env.step_count();
      state.pending.actor = actor;
      state.pending.action = output.units[actor];
      state.pending.source_step = source.source_step;
      state.pending.obligation_id = slot.obligation_ids[actor];
      state.pending.expected_fingerprint = receipt_fingerprint(
          predicted, player, actor, output.units[actor]);
    }
  }
  observe_final_provider_action(env, player, output, state);
  return output;
}

}  // namespace g001::real_weed_move_owner
