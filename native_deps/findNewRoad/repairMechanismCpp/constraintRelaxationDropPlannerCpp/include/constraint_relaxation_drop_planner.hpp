#pragma once

#include "simulator.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace g001::constraint_relaxation {

enum class Disposition : std::uint8_t {
  Kept,
  Deferred,
  DroppedInfeasible,
  DroppedDominated,
};

enum class DropReason : std::uint8_t {
  None,
  DependencyDropped,
  DependencyDeferred,
  DeadlineExpired,
  NoEligibleSlot,
  MoveCommittedSlot,
  ResourceUnavailable,
  LedgerAgeBound,
  LedgerCapacityBound,
  DominatedByLowerDamageSet,
  UnsupportedInput,
};

struct ResourceCertificate {
  std::string key;
  int available{};
  std::uint64_t evidence_hash{};
};

struct ResourceClaim {
  std::string key;
  int amount{};
};

// A certified unit-phase effect that becomes available only after this
// obligation's assigned (step, actor) execution position. PICKUP, HARVEST, and
// other observed unit effects can produce typed tokens. Market fills are out of
// domain and may enter only as a later observation's ResourceCertificate.
struct ResourceProduction {
  std::string key;
  int amount{};
};

struct Slot {
  int step{-1};
  int actor{-1};
  bool move_committed{};
  int move_source_step{-1};
  std::uint64_t move_token{};
  std::uint64_t final_owner_generation{};
};

struct Obligation {
  std::uint64_t id{};
  fastkag::Action action{};
  fastkag::Position tile{-1, -1};
  int actor{-1};  // -1 means any actor represented by a remaining slot.
  int earliest_step{-1};
  int deadline{-1};
  std::vector<std::uint64_t> dependencies;
  std::vector<ResourceClaim> resources;
  std::vector<ResourceProduction> produces;
  int value_loss{};
  // Node-local additional cascade damage.  It excludes every descendant's
  // value/cascade fields; each recursively dropped descendant is charged once.
  int cascade_damage{};
  int defer_damage{};
  bool allow_defer{};
  int carry_days{};
  std::uint64_t lineage_hash{};
};

struct Request {
  int current_step{-1};
  int horizon_end_step{-1};
  int turns_per_day{24};
  std::uint64_t observation_hash{};
  std::uint64_t final_owner_generation{};
  std::vector<Obligation> obligations;
  std::vector<Slot> remaining_slots;
  std::vector<ResourceCertificate> resources;
  int maximum_carry_days{2};
  int maximum_deferred_ledger{32};
  std::size_t exact_node_limit{16};
  std::size_t maximum_search_states{1'000'000};
};

struct Assignment {
  std::uint64_t obligation_id{};
  int slot_index{-1};
  int step{-1};
  int actor{-1};
};

struct DropCertificate {
  std::uint64_t obligation_id{};
  Disposition disposition{Disposition::DroppedInfeasible};
  DropReason reason{DropReason::UnsupportedInput};
  std::uint64_t cancelled_by_dependency{};
  std::vector<std::uint64_t> conflicting_kept_obligations;
  int value_loss{};
  int cascade_damage{};
  std::uint64_t observation_hash{};
  std::uint64_t resource_evidence_hash{};
  std::uint64_t lineage_hash{};
  int deadline{-1};
  std::uint64_t content_hash{};
  std::string diagnostic;
};

struct DeferredDebt {
  std::uint64_t obligation_id{};
  int prior_carry_days{};
  int next_carry_days{};
  int deadline{-1};
  int charged_damage{};
  std::uint64_t lineage_hash{};
};

struct Decision {
  std::uint64_t obligation_id{};
  Disposition disposition{Disposition::DroppedInfeasible};
  std::optional<Assignment> assignment;
  std::optional<DropCertificate> drop;
  std::optional<DeferredDebt> deferred;
};

struct Objective {
  std::int64_t total_damage{};
  int dropped_count{};
  int deferred_count{};
  std::int64_t slot_perturbation{};
  std::uint64_t deterministic_signature{};
};

enum class PlanStatus : std::uint8_t {
  Accepted,
  InvalidInput,
};

struct Plan {
  PlanStatus status{PlanStatus::InvalidInput};
  bool exact{};
  bool bounded_fallback{};
  std::size_t explored_states{};
  std::size_t state_limit{};
  Objective objective;
  std::vector<std::uint64_t> kept_subgraph;
  std::vector<Assignment> assignments;
  std::vector<Decision> decisions;
  std::vector<DropCertificate> drops;
  std::vector<DeferredDebt> deferred_ledger;
  int preserved_move_commitments{};
  std::uint64_t move_commitment_hash{};
  std::uint64_t observation_hash{};
  std::uint64_t final_owner_generation{};
  std::uint64_t request_content_hash{};
  std::uint64_t content_hash{};
  std::string diagnostic;
  [[nodiscard]] bool accepted() const noexcept {
    return status == PlanStatus::Accepted;
  }
};

struct VerifyResult {
  bool valid{};
  std::string diagnostic;
};

[[nodiscard]] Plan plan(const Request& request);
[[nodiscard]] VerifyResult verify(const Request& request,
                                  const Plan& candidate);
[[nodiscard]] std::uint64_t drop_certificate_hash(
    const DropCertificate& certificate) noexcept;
[[nodiscard]] std::uint64_t plan_hash(const Plan& plan) noexcept;
[[nodiscard]] const char* disposition_name(Disposition disposition) noexcept;
[[nodiscard]] const char* drop_reason_name(DropReason reason) noexcept;

}  // namespace g001::constraint_relaxation
