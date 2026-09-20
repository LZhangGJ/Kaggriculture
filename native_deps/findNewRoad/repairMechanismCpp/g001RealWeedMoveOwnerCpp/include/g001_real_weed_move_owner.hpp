#pragma once

#include "g001_day_start_obligation_issuer.hpp"
#include "native_teammate.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace g001::real_weed_move_owner {

struct ObligationAudit {
  obligation_day::ProductionObligation obligation;
  obligation_day::ObligationDisposition disposition{
      obligation_day::ObligationDisposition::CapacityDebt};
  int assigned_actor{-1};
  int remaining_transitions{};
  std::vector<int> transition_steps;
};

enum class RemainingResignReject : std::uint8_t {
  None = 0,
  NoActiveOwner,
  InvalidEnvelope,
  SlotBinding,
  MoveCoverage,
  MoveOrder,
  ObligationIdentity,
  PhysicalNoProgress,
};

struct RemainingSlotProof {
  int step{-1};
  fastkag::Action action{};
  repair_fork::SourceBinding source;
  std::uint64_t obligation_id{};
  std::uint64_t pre_fingerprint{};
  std::uint64_t post_fingerprint{};
};

struct RemainingDayCertificate {
  int player{-1};
  int day{-1};
  int actor{-1};
  int resign_step{-1};
  std::uint64_t issuer_generation{};
  std::uint64_t prior_certificate_hash{};
  std::uint64_t observation_fingerprint{};
  std::vector<RemainingSlotProof> slots;
  std::vector<obligation_day::MoveReplay> remaining_moves;
  std::vector<obligation_day::ObligationFinalStatus> terminal_statuses;
  std::vector<std::uint64_t> obligation_identity_hashes;
  std::uint64_t content_hash{};
};

struct RemainingResignResult {
  RemainingResignReject reject{RemainingResignReject::None};
  std::optional<RemainingDayCertificate> certificate;
  [[nodiscard]] bool issued() const noexcept {
    return reject == RemainingResignReject::None && certificate.has_value();
  }
};

// Complete, immutable identity for one offline-selected explicit debt.  A
// source offset alone is not authority: the issuer envelope, obligation id,
// typed target, and exact immutable route action must all still agree.
struct DebtSelectionIdentity {
  int player{-1};
  int day{-1};
  std::uint64_t issuer_generation{};
  std::uint64_t obligation_id{};
  std::uint64_t obligation_content_hash{};
  int actor{-1};
  int source_step{-1};
  fastkag::Action source_action{};
  obligation_day::GoalKind goal{obligation_day::GoalKind::CropReady};
  fastkag::Item item{fastkag::Item::NONE};
  fastkag::Position tile{};
  int quantity{};
  std::uint64_t content_hash{};
};

using PolicyDeferredAuthorization = DebtSelectionIdentity;

[[nodiscard]] std::uint64_t debt_selection_identity_hash(
    const DebtSelectionIdentity& identity) noexcept;
[[nodiscard]] std::uint64_t production_obligation_identity_hash(
    const obligation_day::ProductionObligation& obligation) noexcept;
[[nodiscard]] inline std::uint64_t policy_deferred_authorization_hash(
    const PolicyDeferredAuthorization& authorization) noexcept {
  return debt_selection_identity_hash(authorization);
}

struct DayAudit {
  int day{-1};
  std::uint64_t start_fingerprint{};
  int stationary_submissions{};
  int owned_actor{-1};
  int move_tokens{};
  int obligations{};
  int completed{};
  int debts{};
  int move_delays{};
  int move_emitted{};
  int production_emitted{};
  int runtime_unit_failures{};
  int stationary_completed{};
  int stationary_debts{};
  int policy_deferred_source_step{-1};
  int receipt_checks{};
  int receipt_failures{};
  int movement_receipts{};
  int production_receipts{};
  int first_receipt_failure_step{-1};
  // The in-provider fingerprint is observability only. Physical acceptance is
  // owned by dayScheduleRuntimeReceiptCpp and must be the release gate.
  bool receipt_fingerprint_diagnostic_only{true};
  int owned_source_coverage_total{};
  int owned_source_move_coverage{};
  int owned_source_obligation_coverage{};
  int owned_source_pass_coverage{};
  int owned_source_missing{};
  int owned_source_duplicates{};
  int owned_source_unsupported{};
  bool planned{};
  bool certificate_valid{};
  bool scoped_manifest_proven{};
  bool exclusive_owner_admitted{};
  bool admission_rejected{};
  bool overlay_conflict_fail_stop{};
  bool remaining_resign_fail_stop{};
  bool native_proposal_discarded{};
  int overlay_conflicts{};
  int remaining_resigns{};
  int interphase_resigns{};
  int overlay_resigns{};
  RemainingResignReject last_remaining_resign_reject{
      RemainingResignReject::None};
  bool unowned_overlay_suppressed{};
  obligation_day::VerifyFailureReason verify_failure{
      obligation_day::VerifyFailureReason::None};
  int verify_failure_step{-1};
  std::vector<obligation_day::ObligationDebt> explicit_debts;
  std::vector<obligation_day::MoveReplay> move_replays;
  std::vector<obligation_day::MoveSourceToken> raw_move_tokens;
  std::optional<DebtSelectionIdentity> debt_selection_identity;
  std::vector<fastkag::NativeAgentState::StationaryObligationSubmission>
      stationary;
  std::vector<ObligationAudit> obligation_audits;
};

struct State {
  fastkag::NativeAgentState native;
  day_start_issuer::PersistentRouteIntentRegistry persistent_lineage;
  std::optional<DebtSelectionIdentity> policy_deferred_authorization;
  int active_day{-1};
  int owned_actor{-1};
  bool active{};
  obligation_day::DayPlanResult plan;
  DayAudit audit;
  std::vector<DayAudit> completed_days;
  std::optional<RemainingDayCertificate> remaining_certificate;
  struct OverlayTestInjection {
    int step{-1};
    int actor{-1};
    fastkag::Action action{};
    bool tamper_move_source_before_resign{};
    bool mutate_candidate_native_before_resign{};
  };
  std::optional<OverlayTestInjection> overlay_test_injection;
  struct PendingReceipt {
    bool active{};
    int step{-1};
    int actor{-1};
    fastkag::Action action{};
    int source_step{-1};
    std::uint64_t obligation_id{};
    std::uint64_t expected_fingerprint{};
  } pending;
};

// Independent default-off provider. It calls the real native provider first,
// consumes only its stationary weed declaration, and lets the certified day
// scheduler exclusively emit that actor's MOVE/production suffix.
[[nodiscard]] fastkag::PlayerAction action_external(
    const fastkag::NativeTeammateExecutor& executor,
    const fastkag::Simulator& env, int player, int route, State& state,
    std::uint64_t issuer_generation);

void finish_day(State& state);

// Feed the registry the exact final provider action at the pre-action
// observation.  Replays which enter this fork mid-episode must call this for
// their untouched prefix; action_external records its own returned actions.
void observe_final_provider_action(const fastkag::Simulator& env, int player,
                                   const fastkag::PlayerAction& action,
                                   State& state);

// Pure transactional boundary: it reads current state and returns a newly
// signed remaining suffix without mutating State.
[[nodiscard]] RemainingResignResult resign_remaining_day(
    const fastkag::Simulator& env, int player, const State& state);
[[nodiscard]] std::uint64_t remaining_day_certificate_hash(
    const RemainingDayCertificate& certificate) noexcept;
[[nodiscard]] const char* remaining_resign_reject_name(
    RemainingResignReject reject) noexcept;

}  // namespace g001::real_weed_move_owner
