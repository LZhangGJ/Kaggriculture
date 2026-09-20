#pragma once

#include "repair_fork_evaluator.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace g001::online_elastic {

enum class FrozenSlotKind : std::uint8_t {
  MoveToken,
  HardSemanticObligation,
  CertifiedSink,
};

struct FrozenSlot {
  int source_step{-1};
  fastkag::Action source_action{};
  FrozenSlotKind kind{FrozenSlotKind::HardSemanticObligation};
};

// A certificate is a commitment by the unique raw-production scheduler, not
// a prediction made by this owner.  Certified sinks must remain PASS or an
// exact no-effect source for the whole suffix.  The owner never manufactures
// a certificate from future Simulator observations.
struct FrozenDaySuffixCertificate {
  int player{-1};
  int day{-1};
  int actor{-1};
  std::uint64_t actor_generation{};
  int suffix_start_step{-1};
  std::uint64_t issuer_generation{};
  bool sinks_irrevocable{};
  std::vector<FrozenSlot> slots;
  std::uint64_t content_hash{};
};

[[nodiscard]] std::uint64_t frozen_suffix_hash(
    const FrozenDaySuffixCertificate& certificate);

struct OnlineAuditView {
  int triggers{};
  int plot_debts_opened{};
  int plot_debts_closed{};
  int unit_receipts_confirmed{};
  int unit_receipts_failed{};
  int purchase_zero_fills{};
  int purchase_partial_fills{};
  int purchase_full_fills{};
  int purchase_receipt_failures{};
  int move_tokens_delayed{};
  int move_tokens_replayed{};
  int missing_capacity_certificate{};
  int invalid_capacity_certificate{};
  int midnight_fail_closed{};
  int fail_closed{};
};

struct PlotDebtView {
  std::uint64_t id{};
  fastkag::Position tile{};
  fastkag::Item desired{fastkag::Item::NONE};
  int origin_day{-1};
  std::optional<std::uint64_t> purchase_debt;
};

// Conservative online owner.  Without an externally issued frozen suffix
// certificate it services only the current PASS/exact-no-effect slot and
// never delays MOVE.  It reads only RepairContext.phase_start, current raw,
// and the exact previous receipt supplied by the evaluator.
class OnlineElasticRepairOwner final : public repair_fork::RepairOwner {
 public:
  struct Impl;

  OnlineElasticRepairOwner();
  ~OnlineElasticRepairOwner() override;
  OnlineElasticRepairOwner(OnlineElasticRepairOwner&&) noexcept;
  OnlineElasticRepairOwner& operator=(OnlineElasticRepairOwner&&) noexcept;
  OnlineElasticRepairOwner(const OnlineElasticRepairOwner&) = delete;
  OnlineElasticRepairOwner& operator=(const OnlineElasticRepairOwner&) = delete;

  std::string name() const override;
  repair_fork::RepairDecision decide(
      const repair_fork::RepairContext& context) override;

  // Installs a read-only scheduler commitment. Invalid certificates are
  // rejected without changing owner state.
  [[nodiscard]] bool install_frozen_suffix(
      FrozenDaySuffixCertificate certificate);

  [[nodiscard]] OnlineAuditView audit() const;
  [[nodiscard]] std::vector<PlotDebtView> plot_debts() const;
  void reset();

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace g001::online_elastic
