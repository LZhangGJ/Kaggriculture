#pragma once

#include "repair_fork_evaluator.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace g001::transactional_crop_repair {
struct TypedCropObligation;
}
namespace g001::transactional_animal_repair {
struct TypedAnimalObligation;
}

namespace g001::repair_owner_composer {

struct TypedRepairInput {
  std::span<const transactional_crop_repair::TypedCropObligation> crops;
  std::span<const transactional_animal_repair::TypedAnimalObligation> animals;
};

// The replacement ABI required for useful (non-fail-stop) arbitration. A
// proposal is immutable and prepare() is const: no production/purchase ledger,
// lease, generation, or expected-receipt state may advance before commit().
struct PreparedRepair {
  std::uint64_t token{};
  std::vector<fastkag::Action> units;
  std::vector<repair_fork::SourceBinding> sources;
  std::vector<int> claimed_actors;
  std::vector<repair_fork::RequiredPurchase> required_purchases;
  repair_fork::RepairTelemetry telemetry{};
};

struct CommittedAction {
  int actor{-1};
  fastkag::Action emitted{};
  repair_fork::ActorPrefixAuthority final_authority{};
};

struct CommittedPurchase {
  std::uint64_t local_debt_id{};
  std::uint64_t external_debt_id{};
  repair_fork::PurchaseBinding final_binding{};
};

// Optional authority supplied by the future unique day-suffix scheduler. It
// permits a MOVE to change hour within the same day, but never actor, source,
// direction, or relative per-actor MOVE order, and every deferred token must
// be replayed before midnight. Absence means exact raw-slot ownership.
struct CertifiedMoveShift {
  int actor{-1};
  int source_step{-1};
  int emitted_step{-1};
  fastkag::Action source_action{};
  std::uint64_t day_suffix_certificate_hash{};
  std::uint64_t issuer_generation{};
};

struct CommitGrant {
  int submitted_step{-1};
  // Complete focal unit manifest selected by the composer. The owner uses
  // this plus the raw joint manifest cached by prepare() to independently
  // verify final per-actor prefix authority (including player-0 effects when
  // the focal player is player 1).
  std::vector<fastkag::Action> final_units;
  std::vector<CommittedAction> actions;
  std::vector<CommittedPurchase> purchases;
  std::vector<CertifiedMoveShift> certified_move_shifts;
};

enum class SettlementResult : std::uint8_t {
  ProtocolInvalid = 0,
  AppliedSuccess = 1,
  AppliedPhysicalFailure = 2,
};

class TransactionalRepairOwner {
public:
  virtual ~TransactionalRepairOwner() = default;
  virtual std::string name() const = 0;

  // Settle only receipts explicitly routed to this owner. A protocol-valid
  // receipt is consumed even when the physical postcondition failed: that
  // failure transition is persistent state used to retry the debt.
  // validate_settle() executes the identical transition on a snapshot. Its
  // non-ProtocolInvalid result guarantees settle_owned() returns the same
  // result without throwing when no call to this owner intervenes.
  virtual SettlementResult settle_owned(
      const repair_fork::RepairContext &context,
      std::span<const repair_fork::ActionReceipt> action_receipts,
      std::span<const repair_fork::PurchaseReceipt> purchase_receipts) = 0;
  virtual SettlementResult validate_settle(
      const repair_fork::RepairContext &context,
      std::span<const repair_fork::ActionReceipt> action_receipts,
      std::span<const repair_fork::PurchaseReceipt> purchase_receipts) const = 0;

  // Must not mutate owner state. token must identify the exact proposal and
  // context facts that commit/abort will consume.
  virtual PreparedRepair
  prepare(const repair_fork::RepairContext &context) const = 0;
  virtual PreparedRepair prepare(const repair_fork::RepairContext &context,
                                 TypedRepairInput) const {
    return prepare(context);
  }

  // The composer calls exactly one of commit or abort once per nonzero token.
  // commit is the only operation allowed to stage leases, ledgers,
  // generations, purchases, or expected receipts. After validate_commit()
  // returns true, commit() for that token/grant must succeed without throwing
  // when no owner call intervenes; the composer enforces that single writer.
  virtual bool validate_commit(std::uint64_t token,
                               const CommitGrant &grant) const = 0;
  virtual bool commit(std::uint64_t token, const CommitGrant &grant) = 0;
  virtual void abort(std::uint64_t token) noexcept = 0;
};

} // namespace g001::repair_owner_composer
