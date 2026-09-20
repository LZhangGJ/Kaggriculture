#pragma once

#include "day_schedule_runtime_receipt.hpp"
#include "g001_day_start_obligation_issuer.hpp"
#include "g001_real_weed_move_owner.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace g001::continuous_runtime_receipt {

struct ComponentDiff {
  bool money{};
  bool farmer{};
  int hand_positions{};
  int tiles{};
  int shed_cells{};
  int seed_cells{};
  int inventory_cells{};
  int inventory_orders{};

  [[nodiscard]] bool empty() const;
};

struct SlotAudit {
  int step{-1};
  bool accepted{};
  day_runtime_receipt::Failure failure{
      day_runtime_receipt::Failure::None};
  bool player_projector_matches_joint{};
  bool unit_phase_step_unchanged{};
  bool full_step_advanced{};
  std::uint64_t actual_before{};
  std::uint64_t projected_after{};
  std::uint64_t certificate_after{};
  fastkag::Action emitted{};
  int source_step{-1};
  std::uint64_t obligation_id{};
  double money_before{};
  double money_after_full_step{};
  int actors_before{};
  int actors_after_full_step{};
  std::vector<fastkag::Action> submitted_market;
  std::vector<int32_t> market_fills;
  ComponentDiff expected_before_diff;
  ComponentDiff expected_after_diff;
};

struct Audit {
  bool opened{};
  day_runtime_receipt::Failure opening_failure{
      day_runtime_receipt::Failure::None};
  std::uint64_t certificate_hash{};
  std::uint64_t issuer_generation{};
  int expected_slots{};
  int accepted_slots{};
  int accepted_actor_slots{};
  int first_failure_step{-1};
  int first_failure_actor{-1};
  day_runtime_receipt::Failure first_failure{
      day_runtime_receipt::Failure::None};
  // Actual owner outputs with certificate source identities. This is
  // diagnostic execution evidence, not accepted runtime MOVE closure.
  int expected_moves{};
  int emitted_moves{};
  bool actual_move_sources_exactly_once{};
  bool runtime_move_closure_accepted{};
  bool midnight_attempted{};
  bool day_closed{};
  day_runtime_receipt::Failure midnight_failure{
      day_runtime_receipt::Failure::None};
  int projector_checks{};
  int projector_failures{};
  int unit_phase_step_failures{};
  int full_step_advance_failures{};
  std::vector<SlotAudit> slots;
};

// Independent bridge between the continuous owner's actual final joint
// manifest and dayScheduleRuntimeReceiptCpp. It never consumes the owner's
// diagnostic receipt counters. The only after-state submitted for physical
// acceptance is an explicit player-unit-phase projection validated against
// the actual joint unit phase.
class Adapter {
 public:
  Adapter();
  ~Adapter();
  Adapter(Adapter&&) noexcept;
  Adapter& operator=(Adapter&&) noexcept;
  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;

  void open(
      const fastkag::Simulator& day_start, int player, int owned_actor,
      std::uint64_t issuer_generation,
      const day_start_issuer::IssueResult& issued,
      const std::optional<real_weed_move_owner::DebtSelectionIdentity>&
          authorization,
      const obligation_day::DayScheduleCertificate& certificate);
  void observe_unit_step(
      const fastkag::Simulator& before,
      const std::array<fastkag::PlayerAction, 2>& actual_joint);
  void observe_full_step(int submitted_step,
                         const fastkag::Simulator& full_after);

  [[nodiscard]] const Audit& audit() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g001::continuous_runtime_receipt
