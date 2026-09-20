#include "../include/certified_move_fork_audit.hpp"

#include <algorithm>
#include <deque>
#include <set>
#include <utility>

namespace g001::certified_move_audit {
namespace {

using fastkag::Action;
using fastkag::Op;
using production_suffix::MoveDelayAuthorization;

constexpr int kTurns = 24;

bool action_equal(const Action& left, const Action& right) {
  return left.op == right.op && left.item == right.item &&
      left.quantity == right.quantity;
}

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
      op == Op::WEST;
}

AuditResult reject(AuditResult result, AuditReject reason) {
  result.accepted = false;
  result.reject = reason;
  return result;
}

bool basic_shape(const AuditRequest& request, int& day_start) {
  if (request.day_start == nullptr || request.player < 0 ||
      request.player >= 2 || request.actor < 0 ||
      request.day_start->hour() != 0 ||
      request.day_start->config().turns_per_day != kTurns ||
      request.raw_units_by_tick.size() != kTurns ||
      request.actual_slots.size() != kTurns)
    return false;
  day_start = request.day_start->day() * kTurns;
  const auto& farm = request.day_start->farms()[
      static_cast<std::size_t>(request.player)];
  const int actors = static_cast<int>(farm.hands.size()) + 1;
  if (request.actor >= actors) return false;
  return std::all_of(request.raw_units_by_tick.begin(),
                     request.raw_units_by_tick.end(), [&](const auto& tick) {
                       return static_cast<int>(tick.size()) == actors;
                     });
}

AuditResult audit_fixed_hour(const AuditRequest& request, AuditResult result,
                             int day_start) {
  result.fixed_hour_fallback = true;
  std::set<int> seen_sources;
  for (int tick = 0; tick < kTurns; ++tick) {
    const int step = day_start + tick;
    const auto& raw = request.raw_units_by_tick[static_cast<std::size_t>(tick)]
                                             [static_cast<std::size_t>(request.actor)];
    const auto& actual = request.actual_slots[static_cast<std::size_t>(tick)];
    if (actual.step != step) return reject(result, AuditReject::InvalidRequest);
    const bool raw_move = is_move(raw.op);
    const bool emitted_move = is_move(actual.emitted.op);
    result.raw_moves += raw_move;
    result.emitted_moves += emitted_move;
    if (raw_move != emitted_move ||
        (raw_move && (!action_equal(raw, actual.emitted) ||
                      actual.source.actor != request.actor ||
                      actual.source.source_step != step ||
                      !action_equal(actual.source.source_action,
                                    actual.emitted))))
      return reject(result, AuditReject::FixedHourMoveMismatch);
    if (emitted_move && !seen_sources.insert(actual.source.source_step).second)
      return reject(result, AuditReject::DuplicateMoveSource);
    ++result.slots_checked;
  }
  result.accepted = true;
  result.reject = AuditReject::None;
  return result;
}

}  // namespace

AuditResult audit_candidate(const AuditRequest& request) {
  AuditResult result;
  result.rich_certificate_present = request.rich_certificate != nullptr;
  result.lossy_projection_ignored = request.lossy_projection != nullptr;
  int day_start = 0;
  if (!basic_shape(request, day_start))
    return reject(result, AuditReject::InvalidRequest);

  if (request.rich_certificate != nullptr) {
    const auto verified = production_suffix::verify_certificate(
        *request.rich_certificate, *request.day_start,
        request.raw_units_by_tick);
    result.certificate_reject = verified.reject;
    result.rich_certificate_verified = verified.valid &&
        request.rich_certificate->player == request.player &&
        request.rich_certificate->actor == request.actor &&
        request.rich_certificate->day == request.day_start->day();
    if (verified.valid && !result.rich_certificate_verified)
      result.certificate_reject = production_suffix::RejectReason::InvalidIdentity;
  }
  if (!result.rich_certificate_verified)
    return audit_fixed_hour(request, result, day_start);

  const auto& certificate = *request.rich_certificate;
  std::deque<int> pending;
  std::set<int> expected_move_sources;
  std::set<int> seen_move_sources;
  std::vector<MoveDelayAuthorization> actual_moves;
  int last_source = -1;
  for (int tick = 0; tick < kTurns; ++tick) {
    const int step = day_start + tick;
    const auto& raw = request.raw_units_by_tick[static_cast<std::size_t>(tick)]
                                             [static_cast<std::size_t>(request.actor)];
    const auto& actual = request.actual_slots[static_cast<std::size_t>(tick)];
    const auto& proof =
        certificate.proof_slots[static_cast<std::size_t>(tick)];
    if (actual.step != step || actual.post_unit_state == nullptr)
      return reject(result, AuditReject::InvalidRequest);
    if (!action_equal(actual.emitted, proof.emitted))
      return reject(result, AuditReject::ManifestMismatch);
    if (actual.source.actor != request.actor ||
        actual.source.source_step != proof.emitted_source_step ||
        !action_equal(actual.source.source_action, actual.emitted))
      return reject(result, AuditReject::SourceBindingMismatch);
    if (production_suffix::focal_unit_state_fingerprint(
            *actual.post_unit_state, request.player, step) !=
        proof.post_state_fingerprint)
      return reject(result, AuditReject::PostStateMismatch);

    if (is_move(raw.op)) {
      pending.push_back(step);
      expected_move_sources.insert(step);
      ++result.raw_moves;
    }
    if (is_move(actual.emitted.op)) {
      ++result.emitted_moves;
      const int source = actual.source.source_step;
      if (source / kTurns != request.day_start->day() ||
          step / kTurns != request.day_start->day())
        return reject(result, AuditReject::CrossDayMove);
      if (!expected_move_sources.contains(source))
        return reject(result, AuditReject::UnauthorizedMoveShift);
      if (!seen_move_sources.insert(source).second)
        return reject(result, AuditReject::DuplicateMoveSource);
      if (source <= last_source || pending.empty() ||
          pending.front() != source)
        return reject(result, AuditReject::MoveSourceOrder);
      if (source > step)
        return reject(result, AuditReject::UnauthorizedMoveShift);
      pending.pop_front();
      last_source = source;
      actual_moves.push_back({source, step, actual.emitted});
    }
    ++result.slots_checked;
  }
  result.pending_at_midnight = static_cast<int>(pending.size());
  if (!pending.empty())
    return reject(result, AuditReject::MidnightQueueNotEmpty);
  if (seen_move_sources.size() != expected_move_sources.size())
    return reject(result, AuditReject::MissingMoveSource);
  if (actual_moves.size() != certificate.authorized_move_replays.size())
    return reject(result, AuditReject::MissingMoveSource);
  for (std::size_t index = 0; index < actual_moves.size(); ++index) {
    const auto& actual = actual_moves[index];
    const auto& authorized = certificate.authorized_move_replays[index];
    if (actual.source_step != authorized.source_step ||
        actual.emitted_step != authorized.emitted_step ||
        !action_equal(actual.exact_move, authorized.exact_move))
      return reject(result, AuditReject::UnauthorizedMoveShift);
    result.authorized_shifts +=
        actual.source_step != actual.emitted_step;
  }
  result.accepted = true;
  result.reject = AuditReject::None;
  return result;
}

const char* audit_reject_name(AuditReject reject_reason) {
  switch (reject_reason) {
    case AuditReject::None: return "none";
    case AuditReject::InvalidRequest: return "invalid_request";
    case AuditReject::FixedHourMoveMismatch:
      return "fixed_hour_move_mismatch";
    case AuditReject::ManifestMismatch: return "manifest_mismatch";
    case AuditReject::SourceBindingMismatch: return "source_binding_mismatch";
    case AuditReject::UnauthorizedMoveShift: return "unauthorized_move_shift";
    case AuditReject::DuplicateMoveSource: return "duplicate_move_source";
    case AuditReject::MissingMoveSource: return "missing_move_source";
    case AuditReject::MoveSourceOrder: return "move_source_order";
    case AuditReject::CrossDayMove: return "cross_day_move";
    case AuditReject::MidnightQueueNotEmpty:
      return "midnight_queue_not_empty";
    case AuditReject::PostStateMismatch: return "post_state_mismatch";
  }
  return "unknown";
}

}  // namespace g001::certified_move_audit
