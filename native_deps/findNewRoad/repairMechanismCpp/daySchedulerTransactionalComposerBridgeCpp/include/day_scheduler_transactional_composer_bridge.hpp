#pragma once

#include "production_obligation_day_scheduler.hpp"
#include "transactional_owner_composer.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace g001::day_scheduler_transactional_bridge {

struct Audit {
  int decisions{};
  int certificate_rejections{};
  int must_finish_rejections{};
  int debt_authorization_rejections{};
  int authorized_debts{};
  int state_divergences{};
  int manifest_rejections{};
  int move_tokens_submitted{};
  int incomplete_day_closures{};
};

struct StagedDecision {
  std::uint64_t token{};
  repair_fork::RepairDecision decision;
};

// Explicit output of the upstream minimum-damage selector. The bridge checks
// identity/integrity only; it never decides that a FEED or other obligation is
// safe to sacrifice. source_step binds the original immutable route source
// carried by ProductionObligation, not merely its deadline or execution slot.
struct DebtAuthorization {
  std::uint64_t certificate_hash{};
  std::uint64_t obligation_id{};
  int source_step{-1};
  obligation_day::GoalKind goal{obligation_day::GoalKind::CropReady};
  int actor{-1};
  fastkag::Position tile{};
  fastkag::Item item{fastkag::Item::NONE};
  std::uint64_t damage_policy_version{};
  std::uint64_t damage_policy_hash{};
  std::uint64_t authorization_generation{};
  std::uint64_t content_hash{};
};

[[nodiscard]] std::uint64_t
debt_authorization_hash(const DebtAuthorization &authorization);

// Offline, non-native bridge. It is the only supported constructor of
// ScheduledStepAuthority: the complete DayScheduleCertificate is verified
// first, including canonical obligation status and exactly-once MOVE coverage.
class DaySchedulerTransactionalComposerBridge final
    : public repair_fork::RepairOwner {
public:
  DaySchedulerTransactionalComposerBridge(
      std::unique_ptr<repair_owner_composer::TransactionalRepairOwner> crop,
      std::unique_ptr<repair_owner_composer::TransactionalRepairOwner> animal,
      const obligation_day::DayPlanRequest &request,
      const obligation_day::DayScheduleCertificate &certificate);
  DaySchedulerTransactionalComposerBridge(
      std::unique_ptr<repair_owner_composer::TransactionalRepairOwner> crop,
      std::unique_ptr<repair_owner_composer::TransactionalRepairOwner> animal,
      const obligation_day::DayPlanRequest &request,
      const obligation_day::DayScheduleCertificate &certificate,
      std::span<const DebtAuthorization> debt_authorizations);
  ~DaySchedulerTransactionalComposerBridge() override;
  DaySchedulerTransactionalComposerBridge(
      DaySchedulerTransactionalComposerBridge &&) noexcept;
  DaySchedulerTransactionalComposerBridge &
  operator=(DaySchedulerTransactionalComposerBridge &&) noexcept;
  DaySchedulerTransactionalComposerBridge(
      const DaySchedulerTransactionalComposerBridge &) = delete;
  DaySchedulerTransactionalComposerBridge &
  operator=(const DaySchedulerTransactionalComposerBridge &) = delete;

  std::string name() const override;
  repair_fork::RepairDecision
  decide(const repair_fork::RepairContext &context) override;
  [[nodiscard]] StagedDecision prepare(
      const repair_fork::RepairContext &context,
      repair_owner_composer::TypedRepairInput input = {});
  [[nodiscard]] bool finalize(std::uint64_t token,
                              const fastkag::PlayerAction &selected);
  [[nodiscard]] bool abort(std::uint64_t token) noexcept;

  [[nodiscard]] bool certificate_valid() const noexcept;
  [[nodiscard]] bool fail_stopped() const noexcept;
  [[nodiscard]] const Audit &audit() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace g001::day_scheduler_transactional_bridge
