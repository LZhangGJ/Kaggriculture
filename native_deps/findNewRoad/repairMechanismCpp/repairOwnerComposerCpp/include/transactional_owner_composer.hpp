#pragma once

#include "transactional_repair_owner.hpp"

#include <memory>
#include <span>

namespace g001::repair_owner_composer {

struct TransactionalComposerAudit {
  int decisions{};
  int actor_conflicts{};
  int market_conflicts{};
  int invalid_proposals{};
  int aborted_proposals{};
  int committed_proposals{};
  int settlement_failures{};
  int commit_fail_stops{};
  int prepared_transactions{};
  int finalized_transactions{};
  int aborted_transactions{};
  int stale_transactions{};
  int second_writers{};
  int exact_action_rejections{};
};

struct StagedRepair {
  std::uint64_t token{};
  repair_fork::RepairDecision decision;
};

// Low-level authority accepted only from the verified day-scheduler bridge.
// The composer does not authenticate a DayScheduleCertificate itself; callers
// must use daySchedulerTransactionalComposerBridgeCpp, which verifies the
// complete certificate and exactly-once MOVE coverage before constructing this
// step projection.
struct ScheduledStepAuthority {
  int player{-1};
  int day{-1};
  int step{-1};
  std::uint64_t certificate_hash{};
  std::uint64_t issuer_generation{};
  std::vector<fastkag::Action> units;
  std::vector<repair_fork::SourceBinding> sources;
  std::vector<CertifiedMoveShift> displaced_moves;
};

// Offline transactional owner-of-owners. It is intentionally not registered
// in the native agent. Conflicting proposals are aborted before either child
// mutates; compatible proposals are committed against one final manifest.
class TransactionalOwnerComposer final : public repair_fork::RepairOwner {
public:
  TransactionalOwnerComposer(std::unique_ptr<TransactionalRepairOwner> crop,
                             std::unique_ptr<TransactionalRepairOwner> animal);
  ~TransactionalOwnerComposer() override;
  TransactionalOwnerComposer(TransactionalOwnerComposer &&) noexcept;
  TransactionalOwnerComposer &operator=(TransactionalOwnerComposer &&) noexcept;
  TransactionalOwnerComposer(const TransactionalOwnerComposer &) = delete;
  TransactionalOwnerComposer &
  operator=(const TransactionalOwnerComposer &) = delete;

  std::string name() const override;
  repair_fork::RepairDecision
  decide(const repair_fork::RepairContext &context) override;
  repair_fork::RepairDecision
  decide_with_intents(const repair_fork::RepairContext &context,
                      TypedRepairInput input);
  [[nodiscard]] StagedRepair
  prepare(const repair_fork::RepairContext &context,
          TypedRepairInput input = {});
  [[nodiscard]] StagedRepair
  prepare_certified(const repair_fork::RepairContext &context,
                    const ScheduledStepAuthority &authority,
                    TypedRepairInput input = {});
  [[nodiscard]] bool finalize(std::uint64_t token,
                              const fastkag::PlayerAction &selected);
  [[nodiscard]] bool abort(std::uint64_t token) noexcept;
  repair_fork::RepairDecision
  decide_certified(const repair_fork::RepairContext &context,
                   const ScheduledStepAuthority &authority);
  repair_fork::RepairDecision
  decide_certified(const repair_fork::RepairContext &context,
                   const ScheduledStepAuthority &authority,
                   TypedRepairInput input);
  [[nodiscard]] const TransactionalComposerAudit &audit() const noexcept;
  [[nodiscard]] bool fail_stopped() const noexcept;

private:
  StagedRepair prepare_impl(const repair_fork::RepairContext &context,
                            const ScheduledStepAuthority *authority,
                            TypedRepairInput input);
  repair_fork::RepairDecision decide_immediately(
      const repair_fork::RepairContext &context,
      const ScheduledStepAuthority *authority, TypedRepairInput input);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace g001::repair_owner_composer
