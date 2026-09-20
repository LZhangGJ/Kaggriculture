#include "g001_continuous_rolling_runtime_receipt_adapter.hpp"

#include "production_suffix_scheduler.hpp"

#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace g001::continuous_rolling_runtime_receipt {
namespace {

bool same_action(fastkag::Action lhs, fastkag::Action rhs) {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool same_source(const repair_fork::SourceBinding& lhs,
                 const repair_fork::SourceBinding& rhs) {
  return lhs.actor == rhs.actor && lhs.source_step == rhs.source_step &&
         same_action(lhs.source_action, rhs.source_action);
}

bool movement(fastkag::Op op) {
  return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
         op == fastkag::Op::EAST || op == fastkag::Op::WEST;
}

fastkag::Position actor_position(const fastkag::Simulator& env, int player,
                                 int actor) {
  if (actor == 0) return env.farms()[player].farmer;
  return env.farms()[player].hands.at(static_cast<std::size_t>(actor - 1));
}

struct Projection {
  fastkag::Simulator actor_after;
  bool prefix_matches{};
  bool step_unchanged{};
};

Projection project_actual_actor_prefix(
    const fastkag::Simulator& before,
    const std::array<fastkag::PlayerAction, 2>& actual_joint, int player,
    int actor, int logical_step) {
  std::array<fastkag::PlayerAction, 2> isolated;
  isolated[player].units.resize(static_cast<std::size_t>(actor) + 1U);
  if (actor < static_cast<int>(actual_joint[player].units.size()))
    isolated[player].units[static_cast<std::size_t>(actor)] =
        actual_joint[player].units[static_cast<std::size_t>(actor)];

  std::array<fastkag::PlayerAction, 2> actual_prefix;
  actual_prefix[player].units.assign(
      actual_joint[player].units.begin(),
      actual_joint[player].units.begin() + std::min(
          actual_joint[player].units.size(),
          static_cast<std::size_t>(actor) + 1U));
  const auto actor_after = before.preview_unit_phase(isolated);
  const auto prefix_after = before.preview_unit_phase(actual_prefix);
  const auto actor_fingerprint =
      production_suffix::focal_unit_state_fingerprint(
          actor_after, player, logical_step);
  const auto prefix_fingerprint =
      production_suffix::focal_unit_state_fingerprint(
          prefix_after, player, logical_step);
  return {actor_after, actor_fingerprint == prefix_fingerprint,
          actor_after.step_count() == before.step_count() &&
              prefix_after.step_count() == before.step_count()};
}

bool same_status(const obligation_day::ObligationFinalStatus& lhs,
                 const obligation_day::ObligationFinalStatus& rhs) {
  return lhs.obligation_id == rhs.obligation_id &&
         lhs.disposition == rhs.disposition &&
         lhs.assigned_actor == rhs.assigned_actor &&
         lhs.remaining_transitions == rhs.remaining_transitions &&
         lhs.transition_steps == rhs.transition_steps;
}

}  // namespace

const char* failure_name(Failure failure) noexcept {
  switch (failure) {
    case Failure::None: return "none";
    case Failure::NotOpen: return "not_open";
    case Failure::StepSequence: return "step_sequence";
    case Failure::FullStepPredecessor: return "full_step_predecessor";
    case Failure::CertificateEnvelope: return "certificate_envelope";
    case Failure::CertificateHash: return "certificate_hash";
    case Failure::ObservationBinding: return "observation_binding";
    case Failure::SlotBinding: return "slot_binding";
    case Failure::ActionMismatch: return "action_mismatch";
    case Failure::ProjectorMismatch: return "projector_mismatch";
    case Failure::UnitPhaseStepChanged: return "unit_phase_step_changed";
    case Failure::PostFingerprintMismatch:
      return "post_fingerprint_mismatch";
    case Failure::PhysicalNoProgress: return "physical_no_progress";
    case Failure::MoveCoverage: return "move_coverage";
    case Failure::MoveOrder: return "move_order";
    case Failure::FullStepDidNotAdvance: return "full_step_did_not_advance";
    case Failure::MidnightIncomplete: return "midnight_incomplete";
  }
  return "unknown";
}

struct Adapter::Impl {
  Audit audit;
  std::optional<obligation_day::DayScheduleCertificate> prior;
  std::vector<std::uint64_t> obligation_identity_hashes;
  std::vector<obligation_day::MoveReplay> expected_moves;
  std::set<std::pair<int, int>> seen_moves;
  int last_move_source{-1};
  int next_step{-1};
  std::optional<int> full_step_predecessor_step;
  std::optional<std::uint64_t> full_step_predecessor_fingerprint;

  bool reject(int step, Failure failure, SlotAudit& slot) {
    slot.failure = failure;
    if (audit.first_failure == Failure::None) {
      audit.first_failure = failure;
      audit.first_failure_step = step;
    }
    audit.slots.push_back(slot);
    return false;
  }
};

Adapter::Adapter() : impl_(std::make_unique<Impl>()) {}
Adapter::~Adapter() = default;
Adapter::Adapter(Adapter&&) noexcept = default;
Adapter& Adapter::operator=(Adapter&&) noexcept = default;

void Adapter::open(
    int player, int actor, std::uint64_t issuer_generation,
    const obligation_day::DayScheduleCertificate& prior_certificate,
    const real_weed_move_owner::DayAudit& owner_day_audit) {
  if (impl_->audit.opened) throw std::runtime_error("rolling adapter reopened");
  if (prior_certificate.player != player || prior_certificate.day < 0 ||
      prior_certificate.issuer_generation != issuer_generation ||
      prior_certificate.slots.size() != 24U ||
      prior_certificate.content_hash == 0 ||
      prior_certificate.content_hash !=
          obligation_day::day_schedule_certificate_hash(prior_certificate))
    throw std::runtime_error("rolling adapter prior certificate invalid");
  impl_->audit.opened = true;
  impl_->audit.player = player;
  impl_->audit.day = prior_certificate.day;
  impl_->audit.actor = actor;
  impl_->audit.issuer_generation = issuer_generation;
  impl_->audit.prior_certificate_hash = prior_certificate.content_hash;
  impl_->audit.expected_slots = 24;
  impl_->prior = prior_certificate;
  impl_->next_step = prior_certificate.day * 24;
  for (const auto& replay : prior_certificate.move_replays) {
    if (replay.actor == actor) impl_->expected_moves.push_back(replay);
  }
  impl_->audit.expected_moves =
      static_cast<int>(impl_->expected_moves.size());
  for (const auto& obligation : owner_day_audit.obligation_audits)
    impl_->obligation_identity_hashes.push_back(
        real_weed_move_owner::production_obligation_identity_hash(
            obligation.obligation));
}

bool Adapter::observe_unit_step(
    const fastkag::Simulator& before,
    const std::array<fastkag::PlayerAction, 2>& actual_joint,
    const real_weed_move_owner::RemainingDayCertificate& certificate) {
  SlotAudit slot;
  slot.step = before.step_count();
  slot.remaining_certificate_hash = certificate.content_hash;
  slot.observation_fingerprint = certificate.observation_fingerprint;
  slot.remaining_slots = static_cast<int>(certificate.slots.size());
  slot.remaining_moves = static_cast<int>(certificate.remaining_moves.size());
  if (!impl_->audit.opened || !impl_->prior)
    return impl_->reject(slot.step, Failure::NotOpen, slot);
  if (slot.step != impl_->next_step)
    return impl_->reject(slot.step, Failure::StepSequence, slot);

  slot.actual_before_fingerprint =
      production_suffix::focal_unit_state_fingerprint(
          before, impl_->audit.player);
  if (impl_->full_step_predecessor_fingerprint) {
    slot.full_step_predecessor_checked = true;
    ++impl_->audit.predecessor_checks;
    slot.full_step_predecessor_matched =
        impl_->full_step_predecessor_step == slot.step &&
        *impl_->full_step_predecessor_fingerprint ==
            slot.actual_before_fingerprint;
    if (!slot.full_step_predecessor_matched) {
      ++impl_->audit.predecessor_failures;
      return impl_->reject(slot.step, Failure::FullStepPredecessor, slot);
    }
  }

  const int tick = slot.step - impl_->audit.day * 24;
  if (certificate.player != impl_->audit.player ||
      certificate.day != impl_->audit.day ||
      certificate.actor != impl_->audit.actor ||
      certificate.resign_step != slot.step ||
      certificate.issuer_generation != impl_->audit.issuer_generation ||
      certificate.prior_certificate_hash !=
          impl_->audit.prior_certificate_hash ||
      tick < 0 || tick >= 24 ||
      certificate.slots.size() != static_cast<std::size_t>(24 - tick) ||
      certificate.terminal_statuses.size() !=
          impl_->prior->obligation_statuses.size() ||
      certificate.obligation_identity_hashes !=
          impl_->obligation_identity_hashes)
    return impl_->reject(slot.step, Failure::CertificateEnvelope, slot);
  if (certificate.content_hash == 0 ||
      certificate.content_hash !=
          real_weed_move_owner::remaining_day_certificate_hash(certificate))
    return impl_->reject(slot.step, Failure::CertificateHash, slot);
  if (certificate.observation_fingerprint != slot.actual_before_fingerprint ||
      certificate.slots.empty() ||
      certificate.slots.front().pre_fingerprint !=
          slot.actual_before_fingerprint)
    return impl_->reject(slot.step, Failure::ObservationBinding, slot);

  for (std::size_t i = 0; i < certificate.terminal_statuses.size(); ++i) {
    if (!same_status(certificate.terminal_statuses[i],
                     impl_->prior->obligation_statuses[i]))
      return impl_->reject(slot.step, Failure::CertificateEnvelope, slot);
  }
  std::vector<obligation_day::MoveReplay> expected_remaining;
  for (const auto& move : impl_->expected_moves) {
    if (move.emitted_step >= slot.step) expected_remaining.push_back(move);
  }
  if (certificate.remaining_moves.size() != expected_remaining.size())
    return impl_->reject(slot.step, Failure::MoveCoverage, slot);
  for (std::size_t i = 0; i < expected_remaining.size(); ++i) {
    const auto& actual = certificate.remaining_moves[i];
    const auto& expected = expected_remaining[i];
    if (actual.actor != expected.actor ||
        actual.source_step != expected.source_step ||
        actual.emitted_step != expected.emitted_step ||
        !same_action(actual.action, expected.action))
      return impl_->reject(slot.step, Failure::MoveCoverage, slot);
    if (i > 0 &&
        (actual.source_step <= certificate.remaining_moves[i - 1].source_step ||
         actual.emitted_step <=
             certificate.remaining_moves[i - 1].emitted_step))
      return impl_->reject(slot.step, Failure::MoveOrder, slot);
  }

  fastkag::Simulator suffix_cursor = before;
  for (std::size_t i = 0; i < certificate.slots.size(); ++i) {
    const auto& current = certificate.slots[i];
    const auto logical_step = slot.step + static_cast<int>(i);
    const auto& prior_slot =
        impl_->prior->slots[static_cast<std::size_t>(tick) + i];
    if (current.step != logical_step || prior_slot.step != logical_step ||
        impl_->audit.actor >= static_cast<int>(prior_slot.actions.size()) ||
        impl_->audit.actor >= static_cast<int>(prior_slot.sources.size()) ||
        impl_->audit.actor >=
            static_cast<int>(prior_slot.obligation_ids.size()) ||
        !same_action(current.action,
                     prior_slot.actions[impl_->audit.actor]) ||
        !same_source(current.source,
                     prior_slot.sources[impl_->audit.actor]) ||
        current.obligation_id !=
            prior_slot.obligation_ids[impl_->audit.actor])
      return impl_->reject(slot.step, Failure::SlotBinding, slot);
    std::array<fastkag::PlayerAction, 2> suffix_action;
    suffix_action[impl_->audit.player].units.resize(
        static_cast<std::size_t>(impl_->audit.actor) + 1U);
    suffix_action[impl_->audit.player]
        .units[static_cast<std::size_t>(impl_->audit.actor)] = current.action;
    const auto suffix_after =
        suffix_cursor.preview_unit_phase(suffix_action);
    if (current.pre_fingerprint !=
            production_suffix::focal_unit_state_fingerprint(
                suffix_cursor, impl_->audit.player, logical_step) ||
        current.post_fingerprint !=
            production_suffix::focal_unit_state_fingerprint(
                suffix_after, impl_->audit.player, logical_step))
      return impl_->reject(slot.step, Failure::SlotBinding, slot);
    suffix_cursor = suffix_after;
  }

  const auto& current = certificate.slots.front();
  slot.source_step = current.source.source_step;
  slot.obligation_id = current.obligation_id;
  if (impl_->audit.actor >=
      static_cast<int>(actual_joint[impl_->audit.player].units.size()))
    return impl_->reject(slot.step, Failure::ActionMismatch, slot);
  slot.emitted = actual_joint[impl_->audit.player]
                              .units[static_cast<std::size_t>(impl_->audit.actor)];
  if (!same_action(slot.emitted, current.action))
    return impl_->reject(slot.step, Failure::ActionMismatch, slot);

  const auto projection = project_actual_actor_prefix(
      before, actual_joint, impl_->audit.player, impl_->audit.actor, slot.step);
  slot.actor_prefix_projector_matched = projection.prefix_matches;
  slot.unit_phase_step_unchanged = projection.step_unchanged;
  ++impl_->audit.projector_checks;
  if (!projection.prefix_matches) {
    ++impl_->audit.projector_failures;
    return impl_->reject(slot.step, Failure::ProjectorMismatch, slot);
  }
  if (!projection.step_unchanged)
    return impl_->reject(slot.step, Failure::UnitPhaseStepChanged, slot);
  slot.projected_after_fingerprint =
      production_suffix::focal_unit_state_fingerprint(
          projection.actor_after, impl_->audit.player, slot.step);
  slot.certificate_after_fingerprint = current.post_fingerprint;
  if (slot.projected_after_fingerprint != current.post_fingerprint)
    return impl_->reject(slot.step, Failure::PostFingerprintMismatch, slot);

  if (current.action.op != fastkag::Op::PASS) {
    std::array<fastkag::PlayerAction, 2> pass;
    pass[impl_->audit.player].units.resize(
        static_cast<std::size_t>(impl_->audit.actor) + 1U);
    const auto pass_after = before.preview_unit_phase(pass);
    const auto pass_fingerprint =
        production_suffix::focal_unit_state_fingerprint(
            pass_after, impl_->audit.player, slot.step);
    if (pass_fingerprint == slot.projected_after_fingerprint)
      return impl_->reject(slot.step, Failure::PhysicalNoProgress, slot);
    if (movement(current.action.op)) {
      auto expected =
          actor_position(before, impl_->audit.player, impl_->audit.actor);
      if (current.action.op == fastkag::Op::NORTH) --expected.y;
      if (current.action.op == fastkag::Op::SOUTH) ++expected.y;
      if (current.action.op == fastkag::Op::WEST) --expected.x;
      if (current.action.op == fastkag::Op::EAST) ++expected.x;
      const auto actual = actor_position(
          projection.actor_after, impl_->audit.player, impl_->audit.actor);
      if (actual.x != expected.x || actual.y != expected.y)
        return impl_->reject(slot.step, Failure::PhysicalNoProgress, slot);
    } else if (current.obligation_id == 0) {
      return impl_->reject(slot.step, Failure::SlotBinding, slot);
    }
  }

  if (movement(current.action.op)) {
    const auto expected = std::find_if(
        impl_->expected_moves.begin(), impl_->expected_moves.end(),
        [&](const auto& move) {
          return move.emitted_step == slot.step &&
                 move.source_step == current.source.source_step &&
                 same_action(move.action, current.action);
        });
    const auto identity =
        std::pair{impl_->audit.actor, current.source.source_step};
    if (expected == impl_->expected_moves.end() ||
        !impl_->seen_moves.insert(identity).second)
      return impl_->reject(slot.step, Failure::MoveCoverage, slot);
    if (current.source.source_step <= impl_->last_move_source)
      return impl_->reject(slot.step, Failure::MoveOrder, slot);
    impl_->last_move_source = current.source.source_step;
    ++impl_->audit.accepted_moves;
  }

  slot.accepted = true;
  ++impl_->audit.accepted_slots;
  ++impl_->next_step;
  impl_->full_step_predecessor_step.reset();
  impl_->full_step_predecessor_fingerprint.reset();
  impl_->audit.slots.push_back(slot);
  if (tick == 23) {
    impl_->audit.midnight_attempted = true;
    impl_->audit.move_sources_exactly_once_and_ordered =
        impl_->seen_moves.size() == impl_->expected_moves.size() &&
        impl_->audit.accepted_moves == impl_->audit.expected_moves;
    impl_->audit.day_closed =
        impl_->audit.accepted_slots == impl_->audit.expected_slots &&
        impl_->audit.move_sources_exactly_once_and_ordered;
    impl_->audit.midnight_failure =
        impl_->audit.day_closed ? Failure::None : Failure::MidnightIncomplete;
  }
  return true;
}

void Adapter::observe_full_step(int submitted_step,
                                const fastkag::Simulator& full_after) {
  if (impl_->audit.slots.empty() ||
      impl_->audit.slots.back().step != submitted_step ||
      !impl_->audit.slots.back().accepted)
    throw std::runtime_error(
        "rolling full-step predecessor lacks accepted unit receipt");
  auto& slot = impl_->audit.slots.back();
  if (slot.full_step_recorded)
    throw std::runtime_error("rolling full-step predecessor duplicated");
  slot.full_step_recorded = true;
  slot.full_step_advanced = full_after.step_count() == submitted_step + 1;
  slot.full_step_after_fingerprint =
      production_suffix::focal_unit_state_fingerprint(
          full_after, impl_->audit.player);
  ++impl_->audit.full_step_records;
  if (!slot.full_step_advanced) {
    ++impl_->audit.full_step_advance_failures;
    if (impl_->audit.first_failure == Failure::None) {
      impl_->audit.first_failure = Failure::FullStepDidNotAdvance;
      impl_->audit.first_failure_step = submitted_step;
    }
    return;
  }
  impl_->full_step_predecessor_step = submitted_step + 1;
  impl_->full_step_predecessor_fingerprint =
      slot.full_step_after_fingerprint;
}

const Audit& Adapter::audit() const noexcept { return impl_->audit; }

}  // namespace g001::continuous_rolling_runtime_receipt
