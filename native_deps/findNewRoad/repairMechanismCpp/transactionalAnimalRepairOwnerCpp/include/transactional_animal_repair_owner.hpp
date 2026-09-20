#pragma once

#include "persistent_production_intent_ledger.hpp"
#include "transactional_repair_owner.hpp"

#include <memory>

namespace g001::transactional_animal_repair {

struct Audit {
  int prepares{}, commits{}, aborts{}, rejected_commits{};
  int objectives_opened{}, purchase_zero{}, purchase_partial{}, purchase_full{};
  int unit_success{}, unit_failure{}, hour23_blocks{}, shared_blocks{};
  int unresolved_purchase_receipts{};
};

// Target provenance supplied by the production/route compiler. BUY_ANIMAL
// itself does not name the future PLACE tile, so opening an objective from a
// first matching empty structure is unsafe.
struct TypedAnimalObligation {
  std::uint64_t id{};
  int player{-1};
  int actor{-1};
  int place_step{-1};
  persistent_production::TileKey target{};
  fastkag::Action place_action{};
  fastkag::Item animal{fastkag::Item::NONE};
  // Optional exact evidence for a BUY_ANIMAL which failed one tick earlier.
  // Day-start bindings leave these defaults and still require a current raw
  // BUY_ANIMAL; cross-tick recovery is accepted only when every field matches
  // the live simulator receipt.
  int acquisition_step{-1};
  int acquisition_market_slot{-1};
  fastkag::Action acquisition_action{};
  int acquisition_fill{-1};
  std::uint64_t acquisition_seed{};
};

class TransactionalAnimalRepairOwner final
    : public repair_owner_composer::TransactionalRepairOwner {
public:
  struct Impl;
  TransactionalAnimalRepairOwner();
  ~TransactionalAnimalRepairOwner() override;
  TransactionalAnimalRepairOwner(TransactionalAnimalRepairOwner &&) noexcept;
  TransactionalAnimalRepairOwner &
  operator=(TransactionalAnimalRepairOwner &&) noexcept;
  TransactionalAnimalRepairOwner(const TransactionalAnimalRepairOwner &) =
      delete;
  TransactionalAnimalRepairOwner &
  operator=(const TransactionalAnimalRepairOwner &) = delete;

  std::string name() const override;
  repair_owner_composer::SettlementResult settle_owned(
      const repair_fork::RepairContext &,
      std::span<const repair_fork::ActionReceipt>,
      std::span<const repair_fork::PurchaseReceipt>) override;
  repair_owner_composer::SettlementResult validate_settle(
      const repair_fork::RepairContext &,
      std::span<const repair_fork::ActionReceipt>,
      std::span<const repair_fork::PurchaseReceipt>) const override;
  repair_owner_composer::PreparedRepair
  prepare(const repair_fork::RepairContext &) const override;
  repair_owner_composer::PreparedRepair
  prepare(const repair_fork::RepairContext &,
          std::span<const TypedAnimalObligation>) const;
  repair_owner_composer::PreparedRepair
  prepare(const repair_fork::RepairContext &,
          repair_owner_composer::TypedRepairInput) const override;
  bool commit(std::uint64_t,
              const repair_owner_composer::CommitGrant &) override;
  bool
  validate_commit(std::uint64_t,
                  const repair_owner_composer::CommitGrant &) const override;
  void abort(std::uint64_t) noexcept override;

  [[nodiscard]] const Audit &audit() const noexcept;
  [[nodiscard]] std::vector<persistent_production::ObjectiveView>
  objectives() const;
  [[nodiscard]] std::size_t expected_actions() const noexcept;
  [[nodiscard]] std::size_t expected_purchases() const noexcept;
  [[nodiscard]] std::size_t cached_proposals() const noexcept;

private:
  std::unique_ptr<Impl> impl_;
};
} // namespace g001::transactional_animal_repair
