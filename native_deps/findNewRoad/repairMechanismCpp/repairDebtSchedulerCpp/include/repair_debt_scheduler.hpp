#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace repair_debt {

enum class ActionKind : std::uint8_t { Pass = 0, Move, Obligation };
enum class DebtState : std::uint8_t {
  Active = 0,
  Completed,
  Deferred,
  DroppedInfeasible,
  DroppedDominated,
  ExpiredTerminal,
};
enum class DropReason : std::uint8_t {
  None = 0,
  SlotCapacity,
  ActorMissing,
  TargetMissing,
  ResourceImpossible,
  DominatedDamage,
  AncestorDropped,
  DeadlineElapsed,
  Terminal,
};
enum class Reject : std::uint8_t {
  None = 0,
  InvalidObservation,
  InvalidLedger,
  InvalidMoveTokens,
  MoveDebtAtMidnight,
  PendingProposal,
  StaleProposal,
  FinalActionMismatch,
  ReceiptMismatch,
  FailStopped,
};

// All fields participate in authority and identity. origin_key is the typed
// issuer identity; it is deliberately not a route step number.
struct DebtIdentity {
  std::uint64_t id{};
  std::uint64_t origin_key{};
  int actor{-1};
  int goal{};
  int item{};
  int tile_x{};
  int tile_y{};
  std::uint64_t dependency_set_hash{};
  friend bool operator==(const DebtIdentity&, const DebtIdentity&) = default;
};

struct TypedProductionAction {
  int op{};
  int item{-1};
  int quantity{1};
  int tile_x{};
  int tile_y{};
  friend bool operator==(const TypedProductionAction&,
                         const TypedProductionAction&) = default;
};

enum class ResourceScope : std::uint8_t {
  ActorCarried = 0, FarmSeed, Shed, Cash, ShedCapacity, MarketSlot,
};

struct ResourceKey {
  ResourceScope scope{ResourceScope::ActorCarried};
  int owner{-1};
  int item{};
  friend bool operator==(const ResourceKey&, const ResourceKey&) = default;
  friend bool operator<(const ResourceKey& a, const ResourceKey& b) {
    return std::tie(a.scope, a.owner, a.item) <
           std::tie(b.scope, b.owner, b.item);
  }
};

struct ResourceRequirement {
  ResourceKey key;
  int quantity{};
};

struct ResourceProduction {
  ResourceKey key;
  int quantity{};
};

struct DebtNode {
  DebtIdentity identity;
  TypedProductionAction required_action;
  std::vector<std::uint64_t> dependencies;
  std::vector<ResourceRequirement> resources;
  std::vector<ResourceProduction> produces;
  int admitted_day{};
  int expires_day{};
  int priority{};
  int value{};
  int cascade_value{};
  int estimated_repair_damage{};
  bool target_must_exist{};
  std::uint64_t authorization_hash{};
};

struct DebtRecord {
  DebtNode node;
  DebtState state{DebtState::Active};
  int rebases{};
  int attempts{};
  int completed_day{-1};
  std::uint64_t last_receipt_hash{};
  DropReason drop_reason{DropReason::None};
  int estimated_drop_damage{};
  std::vector<std::uint64_t> cancelled_descendants;
  std::uint64_t physical_evidence_hash{};
};

struct DebtLedger {
  std::uint64_t registry_generation{};
  std::map<std::uint64_t, DebtRecord> records;
  std::uint64_t content_hash{};
};

struct MoveToken {
  std::uint64_t identity{};
  int day{};
  int actor{-1};
  int ordinal{};
  int direction{};
};

struct ResourceCertificate {
  std::uint64_t observation_hash{};
  std::uint64_t issuer_generation{};
  std::map<ResourceKey, int> holdings;
  // Optional future availability claims, signed as part of content_hash.
  std::map<ResourceKey, int> certified_available_day;
  std::uint64_t physical_evidence_hash{};
  std::uint64_t content_hash{};
};

struct Observation {
  int day{-1};
  int hour{-1};
  int actor{-1};
  bool terminal{};
  std::set<std::pair<int, int>> existing_targets;
  std::set<int> existing_actors;
  std::uint64_t physical_hash{};
};

struct TickInput {
  Observation observation;
  std::vector<MoveToken> immutable_day_moves;
  std::vector<DebtNode> production_dag;
  ResourceCertificate resources;
};

struct Action {
  ActionKind kind{ActionKind::Pass};
  int actor{-1};
  int direction{};
  std::uint64_t move_identity{};
  std::uint64_t debt_id{};
  TypedProductionAction production;
  friend bool operator==(const Action&, const Action&) = default;
};

struct Proposal {
  bool accepted{};
  Reject reject{Reject::None};
  Action action;
  std::uint64_t observation_hash{};
  std::uint64_t ledger_before_hash{};
  std::uint64_t day_move_set_hash{};
  std::uint64_t binding_hash{};
};

struct PhysicalReceipt {
  std::uint64_t proposal_binding_hash{};
  Action committed_action;
  bool physically_applied{};
  bool target_effect_observed{};
  std::map<ResourceKey, int> resource_deltas;
  std::uint64_t before_hash{};
  std::uint64_t after_hash{};
};

struct FinalizeResult {
  bool committed{};
  Reject reject{Reject::None};
  std::optional<std::uint64_t> completed_debt;
  std::optional<std::uint64_t> emitted_move;
};

struct Metrics {
  int iterations{};
  int day_rebases{};
  int completed{};
  int deferred{};
  int dropped_infeasible{};
  int dropped_dominated{};
  int expired_terminal{};
  int moves_emitted{};
  int move_early{};
  int move_duplicate{};
  int move_drop{};
  int fail_stops{};
};

[[nodiscard]] std::uint64_t debt_authorization_hash(const DebtNode&) noexcept;
[[nodiscard]] std::uint64_t ledger_hash(const DebtLedger&) noexcept;
[[nodiscard]] std::uint64_t observation_hash(const Observation&) noexcept;
[[nodiscard]] std::uint64_t move_set_hash(const std::vector<MoveToken>&) noexcept;
[[nodiscard]] std::uint64_t dependency_set_hash(
    const std::vector<std::uint64_t>&) noexcept;
[[nodiscard]] std::uint64_t resource_certificate_hash(
    const ResourceCertificate&) noexcept;

class RepairDebtScheduler {
 public:
  explicit RepairDebtScheduler(DebtLedger ledger = {});

  [[nodiscard]] Proposal prepare(const TickInput& input);
  [[nodiscard]] FinalizeResult finalize(const TickInput& input,
                                        const Proposal& proposal,
                                        const Action& final_action,
                                        const PhysicalReceipt& receipt);

  [[nodiscard]] const DebtLedger& ledger() const noexcept { return ledger_; }
  [[nodiscard]] const Metrics& metrics() const noexcept { return metrics_; }
  [[nodiscard]] Reject failure() const noexcept { return failure_; }
  [[nodiscard]] bool fail_stopped() const noexcept {
    return failure_ != Reject::None;
  }

 private:
  [[nodiscard]] Reject validate_and_roll(const TickInput& input);
  void reduce_soft_constraints(const TickInput& input);
  void drop_subgraph(std::uint64_t root, DebtState disposition,
                     DropReason reason, int damage,
                     std::uint64_t physical_evidence);
  [[nodiscard]] Action choose(const TickInput& input) const;
  void close_terminal();
  void restore_staged();

  DebtLedger ledger_;
  Metrics metrics_;
  Reject failure_{Reject::None};
  int active_day_{-1};
  std::uint64_t active_move_hash_{};
  std::vector<MoveToken> active_moves_;
  std::set<std::uint64_t> emitted_moves_;
  std::map<int, std::uint64_t> current_assignments_;
  std::set<int> current_move_actors_;
  std::optional<Proposal> pending_;
  std::optional<DebtLedger> rollback_ledger_;
  Metrics rollback_metrics_;
  int rollback_active_day_{-1};
  std::uint64_t rollback_move_hash_{};
  std::vector<MoveToken> rollback_moves_;
  std::set<std::uint64_t> rollback_emitted_moves_;
};

[[nodiscard]] const char* reject_name(Reject) noexcept;
[[nodiscard]] const char* debt_state_name(DebtState) noexcept;

}  // namespace repair_debt
