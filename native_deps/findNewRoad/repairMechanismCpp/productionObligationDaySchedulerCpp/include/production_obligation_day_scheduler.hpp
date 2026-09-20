#pragma once

#include "production_suffix_scheduler.hpp"
#include "repair_fork_evaluator.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace g001::obligation_day {

struct MoveSourceToken {
  int actor{-1};
  int source_step{-1};
  fastkag::Action action{};
};

enum class GoalKind : std::uint8_t {
  CropReady = 0,
  Pickup,
  Place,
  Feed,
  Care,
  Harvest,
  BuildPasture,
  CollectFertilizer,
  BuildCoop,
};

struct ResourceNeed {
  fastkag::Item item{fastkag::Item::NONE};
  int seed_quantity{};
  int shed_quantity{};
  int carried_quantity{};
};

struct ProductionObligation {
  std::uint64_t id{};
  // -1 permits any actor; otherwise the actor is fixed.
  int actor{-1};
  fastkag::Position tile{};
  GoalKind goal{GoalKind::CropReady};
  fastkag::Item item{fastkag::Item::NONE};
  int quantity{1};
  std::vector<std::uint64_t> dependencies;
  ResourceNeed resource;
  int earliest_step{-1};
  int deadline_step{-1};
  int priority{};
  bool must_finish_today{};
  bool repair{};
  // Optional immutable route provenance. -1 is accepted for non-route unit
  // requests; the G001 day-start issuer always supplies the exact source.
  int source_step{-1};
  // Offline-selected/default-off sacrifice. It remains in the certificate as
  // explicit debt and is never silently treated as completed.
  bool policy_deferred{};
};

struct DayPlanRequest {
  const fastkag::Simulator* day_start{};
  int player{-1};
  std::uint64_t issuer_generation{};
  std::vector<MoveSourceToken> moves;
  std::vector<ProductionObligation> obligations;
  // Zero preserves the day-start ABI. Remaining-day callers explicitly bind
  // this to day_start->hour().
  int start_tick{};
};

enum class DebtReason : std::uint8_t {
  BlockedOnReceipt = 0,
  Capacity,
  Dependency,
  Deadline,
  UnsupportedState,
  PolicyDeferred,
};

struct ObligationDebt {
  std::uint64_t obligation_id{};
  DebtReason reason{DebtReason::Capacity};
  int remaining_transitions{};
};

struct DaySlotProof {
  int step{-1};
  std::vector<fastkag::Action> actions;
  std::vector<repair_fork::SourceBinding> sources;
  std::vector<std::uint64_t> obligation_ids;
  std::uint64_t focal_post_fingerprint{};
};

struct MoveReplay {
  int actor{-1};
  int source_step{-1};
  int emitted_step{-1};
  fastkag::Action action{};
};

enum class ObligationDisposition : std::uint8_t {
  Completed = 0,
  BlockedOnReceipt,
  CapacityDebt,
  DependencyDebt,
  DeadlineDebt,
  UnsupportedStateDebt,
  PolicyDeferredDebt,
};

// A certificate contains exactly one terminal status for every input
// obligation.  transition_steps is the complete list of slots at which the
// obligation made a verified state transition.
struct ObligationFinalStatus {
  std::uint64_t obligation_id{};
  ObligationDisposition disposition{ObligationDisposition::CapacityDebt};
  int assigned_actor{-1};
  int remaining_transitions{};
  std::vector<int> transition_steps;
};

struct DayScheduleCertificate {
  int player{-1};
  int day{-1};
  int start_step{-1};
  std::uint64_t issuer_generation{};
  std::uint64_t focal_start_fingerprint{};
  std::uint64_t input_hash{};
  std::vector<DaySlotProof> slots;
  std::vector<MoveReplay> move_replays;
  std::vector<ObligationFinalStatus> obligation_statuses;
  std::uint64_t content_hash{};
};

enum class PlanReject : std::uint8_t {
  None = 0,
  InvalidState,
  NotDayStart,
  InvalidIdentity,
  InvalidMoveToken,
  InvalidDag,
  InternalProofFailure,
};

enum class VerifyFailureReason : std::uint8_t {
  None = 0,
  InvalidRequest,
  CertificateEnvelope,
  SlotShape,
  SourceBinding,
  UnexpectedUnboundAction,
  ObligationBinding,
  ObligationEligibility,
  DependencyIncomplete,
  CompiledActionMismatch,
  ActionNoProgress,
  PostFingerprintMismatch,
  MoveReplayCount,
  MoveReplayBinding,
  MoveCoverageCount,
  MoveCoverageBinding,
  StatusIdentity,
  StatusMismatch,
  CanonicalPlanMismatch,
  CanonicalStatusMismatch,
};

struct DayPlanResult {
  PlanReject reject{PlanReject::None};
  // [actor][tick]
  std::vector<std::vector<fastkag::Action>> manifest;
  std::vector<std::vector<repair_fork::SourceBinding>> sources;
  std::vector<ObligationDebt> debts;
  std::vector<std::uint64_t> completed;
  std::optional<DayScheduleCertificate> certificate;
  int move_delays{};

  [[nodiscard]] bool planned() const {
    return reject == PlanReject::None && certificate.has_value();
  }
};

struct VerifyResult {
  bool valid{};
  PlanReject reject{PlanReject::None};
  int checked_slots{};
  VerifyFailureReason failure_reason{VerifyFailureReason::None};
  int failure_step{-1};
  int failure_actor{-1};
  std::uint64_t failure_obligation_id{};
  int failure_index{-1};
};

[[nodiscard]] DayPlanResult plan_day(const DayPlanRequest& request);
// Re-signs only the active suffix from the supplied real observation. A MOVE
// may have an older source_step when the upper receipt owner proves it has not
// yet executed.
[[nodiscard]] DayPlanResult plan_remaining_day(const DayPlanRequest& request);
[[nodiscard]] VerifyResult verify_day_schedule(
    const DayPlanRequest& request,
    const DayScheduleCertificate& certificate);
[[nodiscard]] VerifyResult verify_remaining_day_schedule(
    const DayPlanRequest& request,
    const DayScheduleCertificate& certificate);
[[nodiscard]] std::uint64_t day_schedule_certificate_hash(
    const DayScheduleCertificate& certificate);
[[nodiscard]] const char* debt_reason_name(DebtReason reason);
[[nodiscard]] const char* verify_failure_reason_name(
    VerifyFailureReason reason);

// Deliberately bypasses the public fail-closed verification boundary.  This is
// exposed only for the independent verifier's canonical re-plan and for audit
// minimizers that must preserve an invalid certificate while shrinking it.
namespace internal {
[[nodiscard]] DayPlanResult plan_day_unchecked(const DayPlanRequest& request);
}

}  // namespace g001::obligation_day
