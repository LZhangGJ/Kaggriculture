#pragma once

#include "g001_real_weed_move_owner.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace g001::continuous_rolling_runtime_receipt {

enum class Failure : std::uint8_t {
  None = 0,
  NotOpen,
  StepSequence,
  FullStepPredecessor,
  CertificateEnvelope,
  CertificateHash,
  ObservationBinding,
  SlotBinding,
  ActionMismatch,
  ProjectorMismatch,
  UnitPhaseStepChanged,
  PostFingerprintMismatch,
  PhysicalNoProgress,
  MoveCoverage,
  MoveOrder,
  FullStepDidNotAdvance,
  MidnightIncomplete,
};

[[nodiscard]] const char* failure_name(Failure failure) noexcept;

struct SlotAudit {
  int step{-1};
  bool accepted{};
  Failure failure{Failure::None};
  std::uint64_t remaining_certificate_hash{};
  std::uint64_t observation_fingerprint{};
  std::uint64_t actual_before_fingerprint{};
  std::uint64_t projected_after_fingerprint{};
  std::uint64_t certificate_after_fingerprint{};
  std::uint64_t full_step_after_fingerprint{};
  bool full_step_predecessor_checked{};
  bool full_step_predecessor_matched{};
  bool actor_prefix_projector_matched{};
  bool unit_phase_step_unchanged{};
  bool full_step_recorded{};
  bool full_step_advanced{};
  fastkag::Action emitted{};
  int source_step{-1};
  std::uint64_t obligation_id{};
  int remaining_slots{};
  int remaining_moves{};
};

struct Audit {
  bool opened{};
  int player{-1};
  int day{-1};
  int actor{-1};
  std::uint64_t issuer_generation{};
  std::uint64_t prior_certificate_hash{};
  int expected_slots{};
  int accepted_slots{};
  int expected_moves{};
  int accepted_moves{};
  int predecessor_checks{};
  int predecessor_failures{};
  int projector_checks{};
  int projector_failures{};
  int full_step_records{};
  int full_step_advance_failures{};
  int first_failure_step{-1};
  Failure first_failure{Failure::None};
  bool move_sources_exactly_once_and_ordered{};
  bool midnight_attempted{};
  bool day_closed{};
  Failure midnight_failure{Failure::None};
  std::vector<SlotAudit> slots;
};

// Verifies one freshly resigned RemainingDayCertificate per real observation.
// Only its first slot is physical authority. Cross-observation continuity is
// bound to the evaluator-owned full-step output, never to a prior unit post.
class Adapter {
 public:
  Adapter();
  ~Adapter();
  Adapter(Adapter&&) noexcept;
  Adapter& operator=(Adapter&&) noexcept;
  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;

  void open(
      int player, int actor, std::uint64_t issuer_generation,
      const obligation_day::DayScheduleCertificate& prior_certificate,
      const real_weed_move_owner::DayAudit& owner_day_audit);
  [[nodiscard]] bool observe_unit_step(
      const fastkag::Simulator& before,
      const std::array<fastkag::PlayerAction, 2>& actual_joint,
      const real_weed_move_owner::RemainingDayCertificate& certificate);
  void observe_full_step(int submitted_step,
                         const fastkag::Simulator& full_after);

  [[nodiscard]] const Audit& audit() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g001::continuous_rolling_runtime_receipt
