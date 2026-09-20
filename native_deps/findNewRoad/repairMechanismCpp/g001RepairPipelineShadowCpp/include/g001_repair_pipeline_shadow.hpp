#pragma once

#include "g001_day_start_obligation_issuer.hpp"
#include "g001_typed_intent_issuer.hpp"
#include "state_target_compiler.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace g001::repair_pipeline_shadow {

enum class TypedProof : std::uint8_t {
  CurrentIssuer = 0,
  PersistentCropLineage,
  Unresolved,
};

struct TypedEvidence {
  typed_intent::Evidence current;
  TypedProof proof{TypedProof::Unresolved};
  fastkag::Item resolved_item{fastkag::Item::NONE};
  bool exact{};
  bool emitted{};
};

struct CurrentStepRequest {
  const fastkag::Simulator* observation{};
  int player{-1};
  std::uint64_t issuer_generation{};
  std::span<const fastkag::Action> final_current_units;
  std::span<const production_obligation::ObligationNode> dag_nodes;
  bool dag_suffix_certified{};
  day_start_issuer::PersistentRouteIntentRegistry* persistent_lineage{};
};

struct CurrentStepResult {
  std::vector<transactional_crop_repair::TypedCropObligation> crops;
  std::vector<transactional_animal_repair::TypedAnimalObligation> animals;
  std::vector<TypedEvidence> evidence;
  int persistent_crop_recoveries{};
  int crop_lineage_records{};
  int animal_lineage_records{};
};

// Consumes only the real final current unit manifest. Persistent lineage is
// read before this step and updated afterward from exact PLANT/animal PLACE
// evidence. No action is edited or submitted.
[[nodiscard]] CurrentStepResult observe_final_current_units(
    const CurrentStepRequest& request);

struct DayStartRequest {
  const fastkag::Simulator* day_start{};
  const std::vector<fastkag::PlayerAction>* immutable_route_tape{};
  int player{-1};
  std::uint64_t issuer_generation{};
  const day_start_issuer::PersistentRouteIntentRegistry* persistent_lineage{};
};

struct DayStartResult {
  day_start_issuer::IssueResult issued;
  obligation_day::DayPlanResult plan;
  obligation_day::VerifyResult verification;
  // Current-step final evidence is not available for later slots at hour 0.
  // The adapter therefore never presents it as a day-start exact binding.
  bool current_evidence_temporally_unavailable{true};
  // Provider parity is deliberately excluded here because it is known only
  // after all 24 real final manifests have been observed.
  bool locally_full_certificate_eligible{};
  bool partial_plan{};
};

[[nodiscard]] DayStartResult plan_and_verify_day_shadow(
    const DayStartRequest& request);

struct UnifiedDayResult {
  day_start_issuer::IssueResult issued;
  std::vector<state_target::PlotTarget> targets;
  state_target::CompileResult compiled;
  obligation_day::AtomicAdmissionResult admission;
  bool takeover_eligible{};
};

// Default-off: derives final plot targets from G001's typed macro intent, then
// recompiles unit work from the real day-start state. No action is submitted.
[[nodiscard]] UnifiedDayResult plan_unified_day_shadow(
    const DayStartRequest& request);

enum class MovementAuthorityFailure : std::uint8_t {
  None = 0,
  MissingImmutableCommitment,
  IdentityMismatch,
  HashMismatch,
  Revoked,
  IncompleteHorizon,
  CrossDayToken,
  DuplicateToken,
  TokenOrderMismatch,
  TokenMismatch,
};

using FinalProviderMovementCommitment = fastkag::NativeMovementCommitment;

struct MovementSuffixAuthority {
  bool authorized{};
  MovementAuthorityFailure failure{
      MovementAuthorityFailure::MissingImmutableCommitment};
  int checked_remaining_moves{};
  // If false, a rolling repair may consume only PASS or an independently
  // proved no-effect slot. It may not postpone or replace a MOVE.
  bool production_slot_rewrite_allowed{};
};

enum class MissingLineageCause : std::uint8_t {
  ActorUnavailable = 0,
  ActorUnavailableAtDayStartOrHiredMidday,
  ResidualMoveDrift,
  ActiveDynamicMove,
  HiredActorNoPriorPlant,
  PlannedRouteTileWithoutPriorLineage,
  SuffixAuthorityOnly,
  NoPriorExactPlant,
};

struct MissingLineageContext {
  bool valid_actor{};
  bool present_at_day_start{};
  bool residual_route_position_drift{};
  bool dynamic_actor_overlay{};
  bool has_prior_exact_plant{};
  int actor{-1};
  typed_intent::Proof current_proof{typed_intent::Proof::UnsupportedAction};
};

[[nodiscard]] MissingLineageCause classify_missing_lineage(
    const MissingLineageContext& context) noexcept;
[[nodiscard]] const char* missing_lineage_cause_name(
    MissingLineageCause cause) noexcept;

[[nodiscard]] MovementSuffixAuthority verify_movement_suffix_authority(
    int player, int day, std::uint64_t issuer_generation, int current_step,
    std::span<const obligation_day::MoveSourceToken> issued_moves,
    const FinalProviderMovementCommitment* commitment);

[[nodiscard]] const char* typed_proof_name(TypedProof proof) noexcept;
[[nodiscard]] const char* movement_authority_failure_name(
    MovementAuthorityFailure failure) noexcept;

}  // namespace g001::repair_pipeline_shadow
