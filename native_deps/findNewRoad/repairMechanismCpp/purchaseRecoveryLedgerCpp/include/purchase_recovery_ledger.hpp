#pragma once

#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace g001::purchase_recovery {

// A purchase debt is deliberately narrower than a production transaction.
// It ends when bought inventory is observed. PLANT/PLACE/FEED and all later
// effects remain owned by the unit-repair compiler.
struct Obligation {
  fastkag::Op operation{fastkag::Op::PASS};
  fastkag::Item item{fastkag::Item::NONE};
  int quantity{};
  int unit_cost{};
  std::string provenance;
};

struct FundingObservation {
  int step{-1};
  int cash{};
  // Cash already owed to the final composer's hard market intents. Recovery
  // may spend only cash-protected_cash and never edits those intents.
  int protected_cash{};
};

enum class ProposalOrigin : std::uint8_t {
  // Describes an order already required by the frozen route/final composer.
  // It does not claim that the order is affordable.
  ExistingHardObligation,
  // A new recovery request, bounded by cash remaining after hard obligations.
  Recovery,
};

struct Proposal {
  std::uint64_t debt_id{};
  std::uint64_t revision{};
  // -1 for an existing hard order; exact observation step for Recovery.
  int proposed_step{-1};
  ProposalOrigin origin{ProposalOrigin::ExistingHardObligation};
  fastkag::Action order{};
  int unit_cost{};

  // Intentionally no market slot: only the final composer owns slot choice.
};

struct FinalSelection {
  int submitted_step{-1};
  int selected_slot{-1};
  std::span<const fastkag::Action> final_market;
  // Seed count for BUY_SEED; available unplaced animal count for BUY_ANIMAL.
  // The same holding definition must be used in ReceiptObservation.
  int holding_before{};
  // Revalidated for Recovery proposals so a forged/stale proposal cannot
  // consume cash protected for the hard market queue.
  int cash_before{};
  int protected_cash{};
};

enum class StageStatus : std::uint8_t {
  Selected,
  UnknownDebt,
  StaleProposal,
  AlreadyPending,
  SlotAlreadyBound,
  ConcurrentHoldingWitness,
  InvalidSlot,
  FinalOrderMismatch,
  UnaffordableRecovery,
};

struct ReceiptObservation {
  // Exact next externally visible step after submitted_step.
  int step{-1};
  std::span<const std::int32_t> slot_fills;
  std::array<int, fastkag::N_CROPS> seeds_after{};
  // Available, unplaced GOOSE/COW/SHEEP inventory. This is intentionally not
  // placed-animal count: acquisition does not certify PLACE.
  std::array<int, fastkag::N_ANIMALS> animals_after{};
};

enum class FillStatus : std::uint8_t {
  Full,
  Partial,
  Zero,
  Ambiguous,
};

// Reopens only a previously completed acquisition debt after an upper,
// typed owner has proved that the acquired physical inventory was lost before
// its production handoff could be consumed.  This ledger deliberately does
// not attempt to manufacture that proof: it only enforces that no purchase
// receipt or undrained acquisition handoff is still in flight.
enum class ReopenStatus : std::uint8_t {
  Reopened,
  AlreadyOpen,
  UnknownDebt,
  AwaitingReceipt,
  PendingHandoff,
};

enum class HandoffScope : std::uint8_t {
  // Confirms only that inventory arrived. It never confirms PLANT, PICKUP,
  // PLACE, FEED, CARE, HARVEST, or lifecycle completion.
  InventoryOnly,
};

struct UnitRepairHandoff {
  std::uint64_t debt_id{};
  fastkag::Op acquisition{fastkag::Op::PASS};
  fastkag::Item item{fastkag::Item::NONE};
  int quantity{};
  int observed_step{-1};
  HandoffScope scope{HandoffScope::InventoryOnly};
  std::string provenance;
};

struct Settlement {
  std::uint64_t debt_id{};
  FillStatus status{FillStatus::Ambiguous};
  int requested{};
  int filled{};
  int remaining{};
  std::string reason;
};

struct DebtView {
  std::uint64_t id{};
  Obligation obligation;
  int remaining{};
  int attempts{};
  int filled{};
  bool awaiting_receipt{};
  bool acquisition_complete{};
};

struct Audit {
  int submissions{};
  int recovery_submissions{};
  int full_fills{};
  int partial_fills{};
  int zero_fills{};
  int ambiguous_receipts{};
  int inventory_units_handed_off{};
  int final_selection_rejections{};
};

// Exact-slot two-phase ledger. propose_* is side-effect free. stage_final()
// records only an order that survived the final composer byte-for-byte at the
// caller-selected slot. observe() is the sole acquisition commit authority.
class Ledger {
public:
  std::uint64_t open(Obligation obligation);

  // Use only to witness an order already present as a hard route obligation.
  // Robust frozen routes may use a quantity larger than the debt (for example
  // a buy-as-many-as-affordable sentinel). The exact final quantity is
  // witnessed, while only the required prefix can discharge this debt. This
  // API does not add the order to the market queue.
  [[nodiscard]] std::optional<Proposal>
  describe_hard_order(std::uint64_t debt_id, int requested_quantity) const;

  // Side-effect-free recovery request. The caller supplies hard-obligation
  // cash protection; the ledger returns no proposal until funds recover.
  [[nodiscard]] std::optional<Proposal>
  propose_recovery(std::uint64_t debt_id,
                   const FundingObservation &observation) const;

  // Does not modify final_market. selected_slot is exclusively chosen by the
  // caller's final composer and is accepted only on an exact action match.
  StageStatus stage_final(const Proposal &proposal,
                          const FinalSelection &selection);

  // Settles every submission from observation.step-1. Missing/misaligned
  // receipt ABI is ambiguous and preserves the debt. A zero fill also
  // preserves the debt and allows a later funded retry.
  [[nodiscard]] std::vector<Settlement>
  observe(const ReceiptObservation &observation);

  // One-shot inventory-arrival messages for unit repair. Draining these does
  // not claim any downstream unit action has executed.
  [[nodiscard]] std::vector<UnitRepairHandoff> drain_handoffs();

  // Commit-side authority used by typed lifecycle owners. Reopening restores
  // the original obligation quantity as debt, preserves historical filled and
  // attempt counters for audit, and advances the private proposal revision.
  // Repeating the call while the debt is already open is idempotent.
  ReopenStatus reopen_acquisition(std::uint64_t debt_id);

  [[nodiscard]] std::optional<DebtView> debt(std::uint64_t debt_id) const;
  [[nodiscard]] std::vector<DebtView> debts() const;
  [[nodiscard]] const Audit &audit() const noexcept { return audit_; }
  void reset();

private:
  struct DebtState {
    DebtView view;
    std::uint64_t revision{1};
  };
  struct Pending {
    std::uint64_t debt_id{};
    int submitted_step{-1};
    int slot{-1};
    int requested{};
    int holding_before{};
    fastkag::Action exact_order{};
  };

  [[nodiscard]] DebtState *find(std::uint64_t debt_id);
  [[nodiscard]] const DebtState *find(std::uint64_t debt_id) const;
  [[nodiscard]] static bool supported(const Obligation &obligation);
  [[nodiscard]] static bool same_action(const fastkag::Action &left,
                                        const fastkag::Action &right);

  std::vector<DebtState> debts_;
  std::vector<Pending> pending_;
  std::vector<UnitRepairHandoff> handoffs_;
  Audit audit_;
  std::uint64_t next_debt_id_{1};
};

[[nodiscard]] const char *stage_status_name(StageStatus status) noexcept;
[[nodiscard]] const char *fill_status_name(FillStatus status) noexcept;
[[nodiscard]] const char *reopen_status_name(ReopenStatus status) noexcept;

} // namespace g001::purchase_recovery
