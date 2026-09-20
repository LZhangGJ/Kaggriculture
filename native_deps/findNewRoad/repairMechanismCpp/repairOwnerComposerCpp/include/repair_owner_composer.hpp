#pragma once

#include "repair_fork_evaluator.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace g001::repair_owner_composer {

struct Audit {
  int decisions{};
  int unit_conflicts{};
  int market_conflicts{};
  int invalid_child_manifests{};
  int external_receipt_failures{};
  int child_ack_failures{};
  int fail_stops{};
  int composed_unit_actions{};
  int composed_purchases{};
  int translated_action_receipts{};
  int translated_purchase_receipts{};
  int raw_submission_witnesses{};
  int terminal_unacked_action_receipts{};
  int terminal_unacked_purchase_debts{};
};

// Offline-only owner-of-owners. Both children see the same immutable raw G001
// context and produce proposals independently. This class, and only this
// class, arbitrates actor and market-slot ownership and signs the final joint
// manifest. It is deliberately not registered in the native agent.
class RepairOwnerComposer final : public repair_fork::RepairOwner {
 public:
  RepairOwnerComposer();
  RepairOwnerComposer(std::unique_ptr<repair_fork::RepairOwner> crop,
                      std::unique_ptr<repair_fork::RepairOwner> animal);
  ~RepairOwnerComposer() override;
  RepairOwnerComposer(RepairOwnerComposer&&) noexcept;
  RepairOwnerComposer& operator=(RepairOwnerComposer&&) noexcept;
  RepairOwnerComposer(const RepairOwnerComposer&) = delete;
  RepairOwnerComposer& operator=(const RepairOwnerComposer&) = delete;

  std::string name() const override;
  repair_fork::RepairDecision decide(
      const repair_fork::RepairContext& context) override;

  [[nodiscard]] const Audit& audit() const noexcept;
  [[nodiscard]] bool fail_stopped() const noexcept;
  [[nodiscard]] std::size_t outstanding_action_receipts() const noexcept;
  [[nodiscard]] std::size_t outstanding_purchase_receipts() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace g001::repair_owner_composer
