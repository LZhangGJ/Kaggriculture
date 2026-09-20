#pragma once

#include "persistent_production_intent_ledger.hpp"
#include "repair_fork_evaluator.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace g001::online_animal_repair {

namespace persistent = g001::persistent_production;

struct Audit {
  int objectives_opened{};
  int purchase_requests{};
  int purchase_receipts{};
  int purchase_retries{};
  int unit_proposals{};
  int unit_commits{};
  int unit_receipt_successes{};
  int unit_receipt_failures{};
  int hour23_blocks{};
  int no_service_slot{};
  int raw_obligation_blocks{};
  int move_changes{};
  int external_reconciles{};
};

// Offline-only owner for the typed repair-fork ABI. It never moves a worker
// and only replaces an exact PASS slot (or preserves an already-identical
// lifecycle action). There is deliberately no native registration or option
// bit in this package.
class OnlineAnimalRepairOwner final : public repair_fork::RepairOwner {
 public:
  std::string name() const override;
  repair_fork::RepairDecision decide(
      const repair_fork::RepairContext& context) override;

  [[nodiscard]] persistent::OpenResult ensure_objective(
      int day, persistent::TileKey target, fastkag::Item animal,
      int deadline_day = 29, std::string provenance = "online-animal-repair");

  [[nodiscard]] const Audit& audit() const noexcept { return audit_; }
  [[nodiscard]] const persistent::Ledger& ledger() const noexcept {
    return ledger_;
  }

 private:
  struct PendingUnit {
    std::uint64_t objective_id{};
    std::uint64_t lease_token{};
    std::uint64_t prefix_fingerprint{};
    persistent::ActorIdentity actor;
    persistent::TileKey target;
    int submitted_step{-1};
    std::uint64_t manifest_generation{};
    std::uint64_t prefix_manifest_hash{};
    std::uint64_t post_prefix_state_fingerprint{};
    fastkag::Action emitted{};
  };
  struct PendingPurchase {
    std::uint64_t objective_id{};
    int submitted_step{-1};
    purchase_recovery::Proposal proposal;
    int holding_before{};
    int cash_before{};
    int protected_cash{};
  };

  [[nodiscard]] persistent::Observation observation(
      const fastkag::Simulator& simulator, int player,
      persistent::TileKey target, int actor) const;
  [[nodiscard]] persistent::ActorIdentity actor_identity(
      const fastkag::Simulator& simulator, int actor) const;
  void settle_previous(const repair_fork::RepairContext& context,
                       repair_fork::RepairTelemetry& telemetry);
  void discover_hard_animal(const repair_fork::RepairContext& context,
                            repair_fork::RepairTelemetry& telemetry);
  [[nodiscard]] bool raw_unit_neighborhood_is_sealed(
      const repair_fork::RepairContext& context, int selected_actor) const;
  [[nodiscard]] int protected_market_cash(
      const repair_fork::RepairContext& context) const;

  persistent::Ledger ledger_;
  std::vector<PendingUnit> pending_units_;
  std::vector<PendingPurchase> pending_purchases_;
  std::map<persistent::TileKey, std::uint64_t> target_owner_;
  Audit audit_;
  std::uint64_t final_manifest_generation_{1};
};

}  // namespace g001::online_animal_repair
