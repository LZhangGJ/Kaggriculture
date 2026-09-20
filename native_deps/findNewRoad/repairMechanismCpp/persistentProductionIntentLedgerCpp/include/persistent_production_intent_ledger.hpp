#pragma once

#include "animal_lifecycle_repair.hpp"
#include "purchase_recovery_ledger.hpp"
#include "simulator.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace g001::persistent_production {

using TileKey = animal_lifecycle::TileKey;
using ActorIdentity = animal_lifecycle::ActorIdentity;
using ActorSnapshot = animal_lifecycle::ActorSnapshot;

struct DayPlotKey {
  int origin_day{-1};
  TileKey tile;
  auto operator<=>(const DayPlotKey &) const = default;
};

enum class Kind : std::uint8_t { Crop, Animal };
enum class Status : std::uint8_t { Open, Complete, Expired, FailClosed };
enum class CropStage : std::uint8_t {
  Acquire,
  NeedDig,
  NeedPlant,
  NeedWater,
  Satisfied,
};

struct ObjectiveSpec {
  DayPlotKey key;
  Kind kind{Kind::Crop};
  fastkag::Item item{fastkag::Item::NONE};
  int purchase_quantity{1};
  int unit_cost{};
  int deadline_day{-1};
  bool critical{true};
  int economic_value{};
  std::string provenance;
};

enum class OpenStatus : std::uint8_t {
  Opened,
  InvalidDayOrTile,
  UnsupportedItem,
  InvalidQuantityOrCost,
  ActiveTileOwner,
  DelegateRejected,
};

struct OpenResult {
  OpenStatus status{OpenStatus::UnsupportedItem};
  std::uint64_t objective_id{};
};

struct Observation {
  int step{-1};
  int day{-1};
  TileKey target_key;
  fastkag::Tile target;
  std::array<int, fastkag::N_ITEMS> shed{};
  std::array<int, fastkag::N_CROPS> seeds{};
  ActorSnapshot actor;
  friend bool operator==(const Observation &, const Observation &);
};

struct ActorLease {
  std::uint64_t objective_id{};
  std::uint64_t token{};
  ActorIdentity actor;
  int leased_step{-1};
  auto operator<=>(const ActorLease &) const = default;
};

struct CropProposal {
  std::uint64_t objective_id{};
  std::uint64_t revision{};
  ActorLease lease;
  fastkag::Action action;
  Observation exact_before;
};

using TransitionAuthority =
    std::variant<CropProposal, animal_lifecycle::UnitProposal>;

struct ReadyTransition {
  std::uint64_t objective_id{};
  DayPlotKey key;
  Kind kind{Kind::Crop};
  ActorLease lease;
  fastkag::Position required_position{-1, -1};
  fastkag::Action action;
  int deadline_day{-1};
  bool critical{};
  int economic_value{};
  TransitionAuthority authority;
};

struct ReadyRequest {
  std::uint64_t objective_id{};
  ActorLease lease;
  Observation exact_before;
};

struct ResourceSnapshot {
  std::array<int, fastkag::N_CROPS> seeds{};
  std::array<int, fastkag::N_ITEMS> shed{};
};

// Opaque binding supplied by the owner of the exact Simulator. The ledger
// verifies the manifest-prefix hash and its projected post-prefix Observation
// hash. The owner must derive phase_start_fingerprint from the complete
// private Simulator state; this local ledger cannot reconstruct that state.
struct PrefixAuthority {
  int submitted_step{-1};
  int selected_actor_slot{-1};
  std::uint64_t final_manifest_generation{};
  std::uint64_t phase_start_fingerprint{};
  std::uint64_t lower_prefix_fingerprint{};
  std::uint64_t exact_before_fingerprint{};
  auto operator<=>(const PrefixAuthority &) const = default;
};

[[nodiscard]] std::uint64_t
observation_fingerprint(const Observation &observation) noexcept;
[[nodiscard]] PrefixAuthority
make_prefix_authority(int submitted_step, int selected_actor_slot,
                      std::uint64_t final_manifest_generation,
                      std::uint64_t phase_start_fingerprint,
                      std::span<const fastkag::Action> final_units,
                      const Observation &exact_before);
[[nodiscard]] std::uint64_t
prefix_authority_fingerprint(const PrefixAuthority &authority) noexcept;

struct UnitFinalSelection {
  int submitted_step{-1};
  int selected_actor_slot{-1};
  std::span<const fastkag::Action> final_units;
  // exact_before is the selected actor's state after every lower-numbered
  // unit slot has been applied. A whole-phase pre-state is not a causal
  // witness for actor slots > 0.
  Observation exact_before;
  PrefixAuthority prefix_authority;
  // PLANT has a whole-phase all-or-none seed demand gate in Simulator.  This
  // snapshot must be the seed state before any unit slot was applied, not the
  // post-prefix exact_before.seeds value.
  std::array<int, fastkag::N_CROPS> phase_seeds_before{};
  bool phase_seed_snapshot_bound{};
};

enum class UnitStageStatus : std::uint8_t {
  Selected,
  UnknownObjective,
  RetiredObjective,
  StaleTransition,
  MissingOrStaleLease,
  AlreadyPending,
  SlotAlreadyBound,
  InvalidSlot,
  FinalActionMismatch,
  BeforeWitnessMismatch,
  DelegateRejected,
};

enum class UnitReceiptStatus : std::uint8_t {
  Success,
  Failed,
  Ambiguous,
  UnknownOrDuplicate,
};

struct UnitSettlement {
  std::uint64_t objective_id{};
  UnitReceiptStatus status{UnitReceiptStatus::UnknownOrDuplicate};
  Status objective_status{Status::Open};
  std::string reason;
};

struct PurchaseSettlement {
  std::uint64_t objective_id{};
  purchase_recovery::Settlement settlement;
};

struct AnimalAcquisitionLossProof {
  int step{-1};
  int day{-1};
  TileKey target;
  fastkag::Item animal{fastkag::Item::NONE};
  int shed_count{};
  int all_actor_inventory_count{};
  int all_owned_board_count{};
  int units_reserved_for_other_objectives{};
  bool intended_target_has_animal{};
};

enum class ReopenAnimalAcquisitionStatus : std::uint8_t {
  Reopened,
  AlreadyOpen,
  UnknownObjective,
  NotOpenAnimal,
  InvalidProof,
  PendingUnitReceipt,
  PendingPurchaseReceipt,
  ActiveLease,
  InventoryStillAvailable,
  LifecycleAlreadyPlaced,
  DelegateRejected,
};

struct ObjectiveView {
  std::uint64_t id{};
  ObjectiveSpec spec;
  Status status{Status::Open};
  CropStage crop_stage{CropStage::Acquire};
  std::uint64_t purchase_debt_id{};
  std::uint64_t animal_delegate_id{};
  bool purchase_complete{};
  bool pending_unit_receipt{};
  std::optional<ActorLease> lease;
  std::uint64_t revision{1};
  std::string last_reason;
};

struct Audit {
  int objectives_opened{};
  int active_tile_rejections{};
  int purchase_handoffs{};
  int leases_issued{};
  int stale_lease_rejections{};
  int actor_rebindings{};
  int ready_transitions{};
  int resource_deferred{};
  int unit_submissions{};
  int unit_receipt_successes{};
  int unit_receipt_failures{};
  int ambiguous_receipts{};
  int duplicate_receipts{};
  int completed{};
  int expired{};
};

// Cross-day, plot-owned production intent. Actor ids are ephemeral execution
// leases. This library proposes actions but never chooses/edits final slots.
class Ledger {
public:
  [[nodiscard]] OpenResult open(ObjectiveSpec spec);

  [[nodiscard]] std::optional<purchase_recovery::Proposal>
  describe_hard_purchase(std::uint64_t objective_id,
                         int requested_quantity) const;
  [[nodiscard]] std::optional<purchase_recovery::Proposal>
  propose_purchase(std::uint64_t objective_id,
                   const purchase_recovery::FundingObservation &funding) const;
  purchase_recovery::StageStatus
  stage_purchase(std::uint64_t objective_id,
                 const purchase_recovery::Proposal &proposal,
                 const purchase_recovery::FinalSelection &selection);
  [[nodiscard]] std::vector<PurchaseSettlement>
  observe_purchases(const purchase_recovery::ReceiptObservation &receipt);

  // Commit-side typed recovery. The caller must prove absence across the
  // intended target, shed, every own actor inventory and every owned board
  // tile. This never infers a target from board order and never runs while any
  // receipt/lease is live.
  ReopenAnimalAcquisitionStatus
  reopen_animal_acquisition(std::uint64_t objective_id,
                            const AnimalAcquisitionLossProof &proof);

  // Reconcile never clears an open intent at midnight. It retires only on an
  // exact physical satisfaction witness or after deadline_day has passed.
  Status reconcile(std::uint64_t objective_id, const Observation &observation);

  [[nodiscard]] std::optional<ActorLease>
  lease_actor(std::uint64_t objective_id, ActorIdentity actor, int step);
  bool release_actor(const ActorLease &lease);
  // Releases only a non-pending matching generation. A pending old worker
  // must first receive an exact failed/ambiguous receipt.
  int actor_disappeared(ActorIdentity actor);

  // Returns a resource-feasible prefix ordered by critical, deadline, value,
  // origin day and id. It reserves each actor, tile, seed and shed animal at
  // most once. It never assigns or moves an actor.
  [[nodiscard]] std::vector<ReadyTransition>
  ready_transitions(std::span<const ReadyRequest> requests,
                    ResourceSnapshot resources);
  UnitStageStatus stage_transition(const ReadyTransition &transition,
                                   const UnitFinalSelection &selection);
  [[nodiscard]] UnitSettlement
  observe_transition(std::uint64_t objective_id, std::uint64_t lease_token,
                     std::uint64_t prefix_fingerprint,
                     const Observation &exact_after);

  [[nodiscard]] std::optional<ObjectiveView>
  objective(std::uint64_t objective_id) const;
  [[nodiscard]] std::optional<ObjectiveView> objective(DayPlotKey key) const;
  [[nodiscard]] std::vector<ObjectiveView> objectives_for_day(int day) const;
  [[nodiscard]] std::vector<ObjectiveView> active_objectives() const;
  [[nodiscard]] const Audit &audit() const noexcept { return audit_; }
  [[nodiscard]] const purchase_recovery::Audit &
  seed_purchase_audit() const noexcept {
    return seed_purchases_.audit();
  }
  [[nodiscard]] const animal_lifecycle::Audit &animal_audit() const noexcept {
    return animals_.audit();
  }

private:
  struct State {
    ObjectiveView view;
    std::optional<animal_lifecycle::ActorLease> animal_lease;
    std::optional<PrefixAuthority> pending_prefix_authority;
    // One-day retry window when the exact unit effect is erased by end-of-day
    // normalization before the next public/full-step observation.
    int receipt_grace_day{-1};
  };
  struct PendingCrop {
    CropProposal proposal;
    int slot{-1};
    PrefixAuthority prefix_authority;
  };
  struct PendingSlot {
    std::uint64_t objective_id{};
    int submitted_step{-1};
    int slot{-1};
    bool market{};
  };

  [[nodiscard]] State *find(std::uint64_t objective_id);
  [[nodiscard]] const State *find(std::uint64_t objective_id) const;
  [[nodiscard]] State *find_seed_debt(std::uint64_t debt_id);
  [[nodiscard]] State *find_animal_debt(std::uint64_t debt_id);
  [[nodiscard]] static bool same_action(const fastkag::Action &left,
                                        const fastkag::Action &right);
  [[nodiscard]] static animal_lifecycle::Observation
  animal_observation(const Observation &observation);
  void sync_animal(State &state);
  void release_slot(std::uint64_t objective_id);

  std::map<DayPlotKey, State> objectives_;
  purchase_recovery::Ledger seed_purchases_;
  animal_lifecycle::Ledger animals_;
  std::vector<PendingCrop> pending_crops_;
  std::vector<PendingSlot> pending_slots_;
  Audit audit_;
  std::uint64_t next_objective_id_{1};
  std::uint64_t next_lease_token_{1};
};

[[nodiscard]] const char *open_status_name(OpenStatus status) noexcept;
[[nodiscard]] const char *
unit_stage_status_name(UnitStageStatus status) noexcept;

} // namespace g001::persistent_production
