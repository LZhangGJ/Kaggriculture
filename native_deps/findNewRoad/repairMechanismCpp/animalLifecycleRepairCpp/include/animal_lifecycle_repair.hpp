#pragma once

#include "purchase_recovery_ledger.hpp"
#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace g001::animal_lifecycle {

struct TileKey {
  int row{-1};
  int column{-1};
  auto operator<=>(const TileKey &) const = default;
};

struct AnimalDefinition {
  fastkag::Item animal{fastkag::Item::NONE};
  fastkag::TileKind required_structure{fastkag::TileKind::EMPTY};
  fastkag::Item first_product{fastkag::Item::NONE};
  int purchase_cost{};
  int first_yield_days{};
  int yield_interval_days{};
};

[[nodiscard]] std::optional<AnimalDefinition>
definition(fastkag::Item animal) noexcept;

enum class Stage : std::uint8_t {
  Acquire,
  PickupFromShed,
  PlaceOnTarget,
  FeedCare,
  FirstYield,
  Complete,
};

struct ObjectiveSpec {
  fastkag::Item animal{fastkag::Item::NONE};
  TileKey target;
  int purchase_cost{};
  std::string provenance;
};

enum class OpenStatus : std::uint8_t {
  Opened,
  UnsupportedAnimal,
  InvalidTarget,
  CostMismatch,
  TileAlreadyOwned,
};

struct OpenResult {
  OpenStatus status{OpenStatus::UnsupportedAnimal};
  std::uint64_t objective_id{};
};

struct ActorIdentity {
  int actor_id{-1};
  std::uint64_t generation{};
  auto operator<=>(const ActorIdentity &) const = default;
};

struct ActorLease {
  std::uint64_t objective_id{};
  std::uint64_t token{};
  ActorIdentity actor;
  int leased_step{-1};
};

struct TileSnapshot {
  fastkag::TileKind kind{fastkag::TileKind::EMPTY};
  fastkag::Item animal{fastkag::Item::NONE};
  int placed_day{-1};
  int yield_units{};
  int consecutive_unfed{};
  bool fed_today{};
  bool cared_today{};
  auto operator<=>(const TileSnapshot &) const = default;
};

struct ActorSnapshot {
  ActorIdentity identity;
  TileKey position;
  bool shed_adjacent{};
  std::array<int, fastkag::N_ITEMS> inventory{};
  auto operator<=>(const ActorSnapshot &) const = default;
};

// Public observations deliberately contain only facts available to a final
// composer. The lifecycle never resolves actor ids into persistent owners.
struct Observation {
  int step{-1};
  int day{-1};
  TileSnapshot target;
  std::array<int, fastkag::N_ITEMS> shed{};
  ActorSnapshot actor;
  auto operator<=>(const Observation &) const = default;
};

enum class BlockReason : std::uint8_t {
  None,
  UnknownObjective,
  AcquisitionPending,
  Complete,
  MissingOrStaleLease,
  ObservationMismatch,
  NotAtShed,
  AnimalAbsentFromShed,
  NotAtTarget,
  AnimalAbsentFromActor,
  WrongStructure,
  TargetAnimalLost,
  WheatAbsentFromActor,
  AwaitingDayBoundary,
  YieldNotMature,
};

struct UnitProposal {
  std::uint64_t objective_id{};
  std::uint64_t revision{};
  Stage proposed_stage{Stage::Acquire};
  ActorLease lease;
  fastkag::Action action;
  Observation exact_before;
};

struct ProposalResult {
  BlockReason blocked{BlockReason::UnknownObjective};
  std::optional<UnitProposal> proposal;
};

struct UnitFinalSelection {
  int submitted_step{-1};
  int selected_actor_slot{-1};
  std::span<const fastkag::Action> final_units;
  Observation exact_before;
};

enum class UnitStageStatus : std::uint8_t {
  Selected,
  UnknownObjective,
  StaleProposal,
  MissingOrStaleLease,
  AlreadyPending,
  InvalidSlot,
  FinalActionMismatch,
  BeforeWitnessMismatch,
};

struct UnitReceipt {
  std::uint64_t objective_id{};
  int step{-1};
  std::uint64_t lease_token{};
  Observation exact_after;
};

enum class UnitReceiptStatus : std::uint8_t {
  Success,
  Failed,
  Ambiguous,
  UnknownObjective,
};

struct UnitSettlement {
  std::uint64_t objective_id{};
  Stage attempted_stage{Stage::Acquire};
  UnitReceiptStatus status{UnitReceiptStatus::UnknownObjective};
  Stage retained_or_next_stage{Stage::Acquire};
  std::string reason;
};

struct ObjectiveView {
  std::uint64_t id{};
  ObjectiveSpec spec;
  std::uint64_t purchase_debt_id{};
  Stage stage{Stage::Acquire};
  std::uint64_t revision{1};
  int placed_day{-1};
  bool pending_unit_receipt{};
  bool complete{};
  bool fail_closed{};
  std::string last_block_reason;
};

struct Audit {
  int objectives_opened{};
  int tile_owner_rejections{};
  int purchase_handoffs{};
  int leases_issued{};
  int stale_lease_rejections{};
  int unit_submissions{};
  int unit_receipt_successes{};
  int unit_receipt_failures{};
  int ambiguous_unit_receipts{};
  int lifecycles_completed{};
  int fail_closed_observations{};
  int external_inventory_reconciles{};
};

enum class ExternalReconcileStatus : std::uint8_t {
  NoChange,
  AdvancedToPlace,
  ReturnedToPickup,
  AdvancedToFeedCare,
  PendingReceipt,
  UnknownObjective,
};

// Whole-player physical witness supplied by the persistent owner.  Unlike a
// normal unit Observation, this proof must include every own actor inventory
// and every owned board tile; otherwise a temporarily held or wrong-target
// placed animal could trigger a duplicate purchase.
struct AcquisitionLossProof {
  int step{-1};
  TileKey target;
  fastkag::Item animal{fastkag::Item::NONE};
  int shed_count{};
  int all_actor_inventory_count{};
  int all_owned_board_count{};
  int units_reserved_for_other_objectives{};
  bool intended_target_has_animal{};
};

enum class ReopenAcquisitionStatus : std::uint8_t {
  Reopened,
  AlreadyOpen,
  UnknownObjective,
  InvalidProof,
  LifecycleAlreadyPlaced,
  PendingUnitReceipt,
  ActiveLease,
  InventoryStillAvailable,
  PurchaseLedgerRejected,
};

// Persistent plot-owned lifecycle compiler. The embedded purchase ledger owns
// BUY_ANIMAL only; this class owns every downstream exact unit receipt.
class Ledger {
public:
  [[nodiscard]] OpenResult open(ObjectiveSpec spec);

  [[nodiscard]] std::optional<purchase_recovery::Proposal>
  describe_hard_acquisition(std::uint64_t objective_id,
                            int requested_quantity = 1) const;
  [[nodiscard]] std::optional<purchase_recovery::Proposal> propose_acquisition(
      std::uint64_t objective_id,
      const purchase_recovery::FundingObservation &funding) const;
  purchase_recovery::StageStatus
  stage_acquisition(const purchase_recovery::Proposal &proposal,
                    const purchase_recovery::FinalSelection &selection);
  [[nodiscard]] std::vector<purchase_recovery::Settlement>
  observe_acquisitions(const purchase_recovery::ReceiptObservation &receipt);

  [[nodiscard]] std::optional<ActorLease>
  lease_actor(std::uint64_t objective_id, ActorIdentity actor, int step);
  bool release_actor(const ActorLease &lease);
  [[nodiscard]] ProposalResult propose_unit(std::uint64_t objective_id,
                                            const ActorLease &lease,
                                            const Observation &before);
  UnitStageStatus stage_unit(const UnitProposal &proposal,
                             const UnitFinalSelection &selection);
  [[nodiscard]] UnitSettlement observe_unit(const UnitReceipt &receipt);
  [[nodiscard]] ExternalReconcileStatus
  reconcile_external(std::uint64_t objective_id,
                     const Observation &observation);
  ReopenAcquisitionStatus reopen_acquisition(std::uint64_t objective_id,
                                             const AcquisitionLossProof &proof);

  [[nodiscard]] std::optional<ObjectiveView>
  objective(std::uint64_t objective_id) const;
  [[nodiscard]] std::vector<ObjectiveView> objectives() const;
  [[nodiscard]] const Audit &audit() const noexcept { return audit_; }
  [[nodiscard]] const purchase_recovery::Audit &
  purchase_audit() const noexcept {
    return purchases_.audit();
  }

private:
  struct ObjectiveState {
    ObjectiveView view;
    std::optional<ActorLease> lease;
  };
  struct PendingUnit {
    std::uint64_t objective_id{};
    Stage prior_stage{Stage::Acquire};
    Stage attempted_stage{Stage::Acquire};
    ActorLease lease;
    int submitted_step{-1};
    int slot{-1};
    fastkag::Action exact_action;
    Observation before;
  };

  [[nodiscard]] ObjectiveState *find(std::uint64_t objective_id);
  [[nodiscard]] const ObjectiveState *find(std::uint64_t objective_id) const;
  [[nodiscard]] ObjectiveState *find_purchase(std::uint64_t debt_id);
  [[nodiscard]] static bool same_action(const fastkag::Action &left,
                                        const fastkag::Action &right);
  [[nodiscard]] static const char *block_name(BlockReason reason) noexcept;

  purchase_recovery::Ledger purchases_;
  std::vector<ObjectiveState> objectives_;
  std::vector<PendingUnit> pending_;
  Audit audit_;
  std::uint64_t next_objective_id_{1};
  std::uint64_t next_lease_token_{1};
};

[[nodiscard]] const char *stage_name(Stage stage) noexcept;
[[nodiscard]] const char *open_status_name(OpenStatus status) noexcept;
[[nodiscard]] const char *
unit_stage_status_name(UnitStageStatus status) noexcept;
[[nodiscard]] const char *
unit_receipt_status_name(UnitReceiptStatus status) noexcept;

} // namespace g001::animal_lifecycle
