#pragma once

#include "transactional_repair_owner.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace g001::transactional_crop_repair {

struct Audit {
  int prepares{};
  int commits{};
  int aborts{};
  int rejected_commits{};
  int plots_opened{};
  int plots_closed{};
  int purchase_zero_fills{};
  int purchase_partial_fills{};
  int purchase_full_fills{};
  int receipt_failures{};
  int move_shifts_committed{};
  int move_shifts_replayed{};
  int uncertified_move_rejections{};
  int midnight_failures{};
};

struct PlotDebtView {
  std::uint64_t id{};
  fastkag::Position tile{};
  fastkag::Item desired{fastkag::Item::NONE};
  int origin_day{-1};
  std::uint64_t purchase_debt_id{};
  std::uint64_t obligation_id{};
  int source_player{-1};
  int source_actor{-1};
  int source_step{-1};
  fastkag::Action source_action{};
};

// Exact production intent supplied by the route/production compiler. WATER
// and HARVEST do not encode a crop item in the official action ABI, so a
// first-time weed repair must bind to this typed source rather than guessing
// from Action::item. Existing plot debt is itself a persistent typed intent.
struct TypedCropObligation {
  std::uint64_t id{};
  int player{-1};
  int actor{-1};
  int source_step{-1};
  fastkag::Position tile{};
  fastkag::Action source_action{};
  fastkag::Item desired{fastkag::Item::NONE};
};

// A crop-only implementation of the two-phase repair ABI. prepare() may add
// an immutable entry to an ephemeral proposal cache, but cannot alter any
// production/purchase/route/receipt state. commit() is atomic and is the only
// method that may advance persistent state. abort() only drops cached bytes.
class TransactionalCropRepairOwner final
    : public repair_owner_composer::TransactionalRepairOwner {
public:
  struct Impl;

  TransactionalCropRepairOwner();
  ~TransactionalCropRepairOwner() override;
  TransactionalCropRepairOwner(TransactionalCropRepairOwner &&) noexcept;
  TransactionalCropRepairOwner &
  operator=(TransactionalCropRepairOwner &&) noexcept;
  TransactionalCropRepairOwner(const TransactionalCropRepairOwner &) = delete;
  TransactionalCropRepairOwner &
  operator=(const TransactionalCropRepairOwner &) = delete;

  std::string name() const override;
  repair_owner_composer::SettlementResult settle_owned(
      const repair_fork::RepairContext &context,
      std::span<const repair_fork::ActionReceipt> action_receipts,
      std::span<const repair_fork::PurchaseReceipt> purchase_receipts) override;
  repair_owner_composer::SettlementResult validate_settle(
      const repair_fork::RepairContext &context,
      std::span<const repair_fork::ActionReceipt> action_receipts,
      std::span<const repair_fork::PurchaseReceipt> purchase_receipts)
      const override;
  repair_owner_composer::PreparedRepair
  prepare(const repair_fork::RepairContext &context) const override;
  repair_owner_composer::PreparedRepair
  prepare(const repair_fork::RepairContext &context,
          std::span<const TypedCropObligation> obligations) const;
  repair_owner_composer::PreparedRepair
  prepare(const repair_fork::RepairContext &context,
          repair_owner_composer::TypedRepairInput input) const override;
  bool commit(std::uint64_t token,
              const repair_owner_composer::CommitGrant &grant) override;
  bool validate_commit(
      std::uint64_t token,
      const repair_owner_composer::CommitGrant &grant) const override;
  void abort(std::uint64_t token) noexcept override;

  [[nodiscard]] const Audit &audit() const noexcept;
  [[nodiscard]] std::vector<PlotDebtView> plot_debts() const;
  [[nodiscard]] std::uint64_t persistent_fingerprint() const noexcept;
  [[nodiscard]] std::size_t cached_proposals() const noexcept;
  [[nodiscard]] std::size_t expected_action_receipts() const noexcept;
  [[nodiscard]] std::size_t expected_purchase_receipts() const noexcept;
  [[nodiscard]] std::size_t delayed_moves() const noexcept;

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace g001::transactional_crop_repair
