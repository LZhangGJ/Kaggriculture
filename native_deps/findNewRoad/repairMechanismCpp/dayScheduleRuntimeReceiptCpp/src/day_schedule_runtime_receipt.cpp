#include "day_schedule_runtime_receipt.hpp"

#include "production_suffix_scheduler.hpp"

#include <array>
#include <cstddef>

namespace g001::day_runtime_receipt {
namespace {

using fastkag::Action;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;

bool action_equal(const Action& lhs, const Action& rhs) {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
         op == Op::WEST;
}

bool supported(Op op) {
  switch (op) {
    case Op::PASS:
    case Op::NORTH:
    case Op::SOUTH:
    case Op::EAST:
    case Op::WEST:
    case Op::DIG:
    case Op::PLANT:
    case Op::WATER:
    case Op::HARVEST:
    case Op::PICKUP:
    case Op::PLACE:
    case Op::FEED:
    case Op::CARE:
    case Op::BUILD_PASTURE:
      return true;
    default:
      return false;
  }
}

std::uint64_t fingerprint(const Simulator& state, int player,
                          int logical_step) {
  return production_suffix::focal_unit_state_fingerprint(state, player,
                                                          logical_step);
}

}  // namespace

RuntimeReceiptVerifier::RuntimeReceiptVerifier(
    const obligation_day::DayPlanRequest& request,
    const obligation_day::DayScheduleCertificate& certificate)
    : certificate_(certificate), day_start_(certificate.day * 24) {
  const auto verified =
      obligation_day::verify_day_schedule(request, certificate);
  if (!verified.valid || certificate.content_hash == 0 ||
      certificate.content_hash !=
          obligation_day::day_schedule_certificate_hash(certificate) ||
      certificate.slots.size() != 24U) {
    failed_ = true;
    opening_failure_ = Failure::CertificateInvalid;
    return;
  }
  closed_slots_.assign(certificate.slots.size(), false);
}

bool RuntimeReceiptVerifier::ready() const noexcept {
  return opening_failure_ == Failure::None && !failed_ && !closed_;
}

Failure RuntimeReceiptVerifier::opening_failure() const noexcept {
  return opening_failure_;
}

Result RuntimeReceiptVerifier::snapshot_result(bool accepted,
                                               bool day_closed) const {
  Result result;
  result.accepted = accepted;
  result.day_closed = day_closed;
  result.checked_steps = next_tick_;
  result.checked_actor_slots = checked_actor_slots_;
  return result;
}

Result RuntimeReceiptVerifier::reject(Failure failure, int step, int actor) {
  failed_ = true;
  Result result = snapshot_result(false);
  result.failure = failure;
  result.failure_step = step;
  result.failure_actor = actor;
  return result;
}

Result RuntimeReceiptVerifier::accept(const StepExecution& execution) {
  if (opening_failure_ != Failure::None) {
    return reject(Failure::CertificateInvalid);
  }
  if (closed_) {
    Result result = snapshot_result(false, true);
    result.failure = Failure::AlreadyClosed;
    return result;
  }
  if (failed_) {
    Result result = snapshot_result(false);
    result.failure = Failure::SessionFailed;
    return result;
  }
  if (execution.before == nullptr || execution.after == nullptr) {
    return reject(Failure::NullSnapshot);
  }
  if (next_tick_ >= static_cast<int>(certificate_.slots.size())) {
    return reject(Failure::DuplicateStep, day_start_ + next_tick_);
  }

  const int expected_step = day_start_ + next_tick_;
  int received_step = -1;
  if (!execution.actors.empty()) received_step = execution.actors.front().step;
  if (received_step < expected_step) {
    return reject(Failure::DuplicateStep, received_step);
  }
  if (received_step > expected_step) {
    return reject(Failure::SkippedStep, received_step);
  }

  const auto& slot = certificate_.slots[static_cast<std::size_t>(next_tick_)];
  const std::size_t actors = slot.actions.size();
  if (slot.step != expected_step || actors == 0U ||
      slot.sources.size() != actors || slot.obligation_ids.size() != actors ||
      execution.actors.size() != actors) {
    return reject(Failure::SlotShape, expected_step);
  }

  const std::uint64_t actual_before =
      fingerprint(*execution.before, certificate_.player, expected_step);
  if (execution.claimed_before_fingerprint != actual_before) {
    return reject(Failure::BeforeFingerprintForgery, expected_step);
  }
  if (next_tick_ == 0) {
    if (actual_before != certificate_.focal_start_fingerprint) {
      return reject(Failure::StateFingerprintFork, expected_step);
    }
  } else if (fingerprint(*execution.before, certificate_.player,
                         expected_step - 1) !=
             previous_post_fingerprint_) {
    return reject(Failure::StateFingerprintFork, expected_step);
  }

  Simulator expected_after = *execution.before;
  for (std::size_t actor = 0; actor < actors; ++actor) {
    const auto& actual = execution.actors[actor];
    const auto& source = slot.sources[actor];
    if (actual.player != certificate_.player ||
        actual.day != certificate_.day || actual.step != expected_step ||
        actual.actor != static_cast<int>(actor)) {
      return reject(Failure::IdentityBinding, expected_step,
                    static_cast<int>(actor));
    }
    if (actual.certificate_hash != certificate_.content_hash) {
      return reject(Failure::CertificateHashBinding, expected_step,
                    static_cast<int>(actor));
    }
    if (actual.issuer_generation != certificate_.issuer_generation) {
      return reject(Failure::GenerationBinding, expected_step,
                    static_cast<int>(actor));
    }
    if (actual.source_step != source.source_step ||
        !action_equal(actual.source_action, source.source_action)) {
      return reject(Failure::SourceBinding, expected_step,
                    static_cast<int>(actor));
    }
    if (actual.obligation_id != slot.obligation_ids[actor]) {
      return reject(Failure::ObligationBinding, expected_step,
                    static_cast<int>(actor));
    }
    if (!action_equal(actual.emitted, slot.actions[actor])) {
      return reject(Failure::EmittedActionMismatch, expected_step,
                    static_cast<int>(actor));
    }
    if (!supported(actual.emitted.op)) {
      return reject(Failure::UnsupportedAction, expected_step,
                    static_cast<int>(actor));
    }

    // Re-run exactly this actor against the state left by lower actor slots.
    // This is the official simulator interpreter, not a parallel handwritten
    // action model.  It gives each action an attributable physical delta.
    std::array<PlayerAction, 2> isolated;
    isolated[static_cast<std::size_t>(certificate_.player)].units.resize(actors);
    isolated[static_cast<std::size_t>(certificate_.player)].units[actor] =
        actual.emitted;
    const std::uint64_t actor_before =
        fingerprint(expected_after, certificate_.player, expected_step);
    Simulator actor_after = expected_after.preview_unit_phase(isolated);
    const std::uint64_t actor_post =
        fingerprint(actor_after, certificate_.player, expected_step);
    if (actual.emitted.op == Op::PASS) {
      if (actor_before != actor_post) {
        return reject(Failure::PassHadEffect, expected_step,
                      static_cast<int>(actor));
      }
    } else if (actor_before == actor_post) {
      return reject(Failure::PhysicalNoEffect, expected_step,
                    static_cast<int>(actor));
    }
    expected_after = std::move(actor_after);

    if (is_move(actual.emitted.op) &&
        !seen_moves_.insert({actual.actor, actual.source_step}).second) {
      return reject(Failure::MoveNotExactlyOnce, expected_step,
                    static_cast<int>(actor));
    }
  }

  const std::uint64_t actual_after =
      fingerprint(*execution.after, certificate_.player, expected_step);
  if (execution.claimed_after_fingerprint != actual_after) {
    return reject(Failure::AfterFingerprintForgery, expected_step);
  }
  if (fingerprint(expected_after, certificate_.player, expected_step) !=
      actual_after) {
    return reject(Failure::PhysicalAfterMismatch, expected_step);
  }
  if (actual_after != slot.focal_post_fingerprint) {
    return reject(Failure::CertificatePostMismatch, expected_step);
  }

  previous_post_fingerprint_ = actual_after;
  closed_slots_[static_cast<std::size_t>(next_tick_)] = true;
  ++next_tick_;
  checked_actor_slots_ += static_cast<int>(actors);
  return snapshot_result(true);
}

Result RuntimeReceiptVerifier::close_midnight() {
  if (opening_failure_ != Failure::None) {
    return reject(Failure::CertificateInvalid);
  }
  if (closed_) {
    Result result = snapshot_result(false, true);
    result.failure = Failure::AlreadyClosed;
    return result;
  }
  if (failed_) {
    Result result = snapshot_result(false);
    result.failure = Failure::SessionFailed;
    return result;
  }
  if (next_tick_ != static_cast<int>(certificate_.slots.size())) {
    return reject(Failure::IncompleteDay, day_start_ + next_tick_);
  }
  for (std::size_t tick = 0; tick < closed_slots_.size(); ++tick) {
    if (!closed_slots_[tick]) {
      return reject(Failure::SlotNotClosed,
                    day_start_ + static_cast<int>(tick));
    }
  }
  std::set<std::pair<int, int>> expected_moves;
  for (const auto& move : certificate_.move_replays) {
    if (!expected_moves.insert({move.actor, move.source_step}).second) {
      return reject(Failure::MoveNotExactlyOnce, move.emitted_step,
                    move.actor);
    }
  }
  if (seen_moves_ != expected_moves) {
    return reject(Failure::MoveNotExactlyOnce, day_start_ + 23);
  }
  closed_ = true;
  return snapshot_result(true, true);
}

const char* failure_name(Failure failure) noexcept {
  switch (failure) {
    case Failure::None: return "none";
    case Failure::CertificateInvalid: return "certificate_invalid";
    case Failure::SessionFailed: return "session_failed";
    case Failure::AlreadyClosed: return "already_closed";
    case Failure::NullSnapshot: return "null_snapshot";
    case Failure::DuplicateStep: return "duplicate_step";
    case Failure::SkippedStep: return "skipped_step";
    case Failure::SlotShape: return "slot_shape";
    case Failure::IdentityBinding: return "identity_binding";
    case Failure::CertificateHashBinding: return "certificate_hash_binding";
    case Failure::GenerationBinding: return "generation_binding";
    case Failure::SourceBinding: return "source_binding";
    case Failure::ObligationBinding: return "obligation_binding";
    case Failure::EmittedActionMismatch: return "emitted_action_mismatch";
    case Failure::UnsupportedAction: return "unsupported_action";
    case Failure::BeforeFingerprintForgery:
      return "before_fingerprint_forgery";
    case Failure::AfterFingerprintForgery: return "after_fingerprint_forgery";
    case Failure::StateFingerprintFork: return "state_fingerprint_fork";
    case Failure::PhysicalNoEffect: return "physical_no_effect";
    case Failure::PassHadEffect: return "pass_had_effect";
    case Failure::PhysicalAfterMismatch: return "physical_after_mismatch";
    case Failure::CertificatePostMismatch:
      return "certificate_post_mismatch";
    case Failure::IncompleteDay: return "incomplete_day";
    case Failure::MoveNotExactlyOnce: return "move_not_exactly_once";
    case Failure::SlotNotClosed: return "slot_not_closed";
  }
  return "unknown";
}

}  // namespace g001::day_runtime_receipt
