#pragma once

#include "production_suffix_scheduler.hpp"
#include "repair_fork_evaluator.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace g001::certified_move_audit {

struct ActualUnitSlot {
  int step{-1};
  fastkag::Action emitted{};
  // The isolated adapter extends SourceBinding with source_step=-2 for the
  // certificate's synthetic PASS over a certified no-effect source. -1 keeps
  // its existing inserted-repair meaning.
  repair_fork::SourceBinding source{};
  // Snapshot immediately after this tick's unit phase.  Market, decay, RNG,
  // and end-of-day must not yet have run.  Opponent private state is ignored
  // by the focal proof fingerprint. The evaluator, not the candidate owner,
  // must own and supply this snapshot.
  const fastkag::Simulator* post_unit_state{};
};

struct AuditRequest {
  const fastkag::Simulator* day_start{};
  int player{-1};
  int actor{-1};
  std::vector<std::vector<fastkag::Action>> raw_units_by_tick;
  std::vector<ActualUnitSlot> actual_slots;

  // Only a verified rich certificate can authorize a shifted MOVE.
  const production_suffix::ProductionFrozenDaySuffixCertificate*
      rich_certificate{};
  // Accepted only so callers can prove it is ignored.  This lossy projection
  // never authorizes a shift on its own.
  const online_elastic::FrozenDaySuffixCertificate* lossy_projection{};
};

enum class AuditReject : std::uint8_t {
  None = 0,
  InvalidRequest,
  FixedHourMoveMismatch,
  ManifestMismatch,
  SourceBindingMismatch,
  UnauthorizedMoveShift,
  DuplicateMoveSource,
  MissingMoveSource,
  MoveSourceOrder,
  CrossDayMove,
  MidnightQueueNotEmpty,
  PostStateMismatch,
};

struct AuditResult {
  bool accepted{};
  AuditReject reject{AuditReject::None};
  bool rich_certificate_present{};
  bool rich_certificate_verified{};
  bool fixed_hour_fallback{};
  bool lossy_projection_ignored{};
  production_suffix::RejectReason certificate_reject{
      production_suffix::RejectReason::None};
  int slots_checked{};
  int raw_moves{};
  int emitted_moves{};
  int authorized_shifts{};
  int pending_at_midnight{};
};

// Typed adapter for an isolated fork evaluator.  It performs no native writes
// and owns no route policy.  Invalid/missing rich proof falls back to exact raw
// MOVE hour; it never degrades into permissive slot checking.
[[nodiscard]] AuditResult audit_candidate(const AuditRequest& request);
[[nodiscard]] const char* audit_reject_name(AuditReject reject);

}  // namespace g001::certified_move_audit
