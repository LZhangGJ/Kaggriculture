#pragma once

#include "online_elastic_repair_owner.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace g001::production_suffix {

// Extra work committed by the production scheduler.  The sequence is strict:
// action i may run only after every earlier insertion has succeeded.  Position
// is checked immediately before execution.
struct RequiredInsertion {
  std::uint64_t semantic_id{};
  fastkag::Position position{};
  fastkag::Action action{};
};

struct WorkBudget {
  // Kaggriculture itself has one unit action per actor/tick and no separate
  // stamina field.  This caller-owned budget makes any external work/energy
  // limit explicit rather than silently assuming it away.
  int available_stamina{};
  int move_cost{1};
  int production_cost{1};
  int other_cost{1};
};

struct IssueRequest {
  const fastkag::Simulator* phase_start{};
  int player{-1};
  int actor{-1};
  std::uint64_t actor_generation{};
  std::uint64_t issuer_generation{};

  // Full own-unit day skeleton, indexed [day tick][actor].  It must begin at
  // hour zero and contain exactly turns_per_day ticks.  No future market state,
  // opponent action, fill, or receipt is accepted by this ABI.
  std::vector<std::vector<fastkag::Action>> raw_units_by_tick;

  // Non-PASS/non-MOVE sources are hard by default.  A source can become a
  // CertifiedSink only when listed here AND independently proven to be an
  // exact no-effect action in the final witness state.
  std::set<int> discardable_no_effect_steps;
  std::vector<RequiredInsertion> insertions;
  WorkBudget budget;
};

enum class RejectReason : std::uint8_t {
  None = 0,
  NullState,
  NotAtDayStart,
  UnsupportedTurnsPerDay,
  TerminalDay,
  InvalidIdentity,
  InvalidRouteShape,
  InvalidSourceAction,
  InvalidInsertion,
  UnsafeSink,
  PreconditionsUnsatisfied,
  SlotCapacity,
  StaminaExceeded,
  CrossDay,
  AlreadyIssued,
  InternalProofFailure,
};

struct WitnessSlot {
  int emitted_step{-1};
  fastkag::Action emitted{};
  // -1 identifies inserted repair work. -2 identifies a synthetic PASS which
  // consumes a certified exact-no-effect raw source. Non-negative values name
  // the immutable raw source step whose exact action byte was replayed.
  int source_step{-1};
  std::uint64_t semantic_id{};
};

struct HardSemanticObligation {
  std::uint64_t obligation_id{};
  int source_step{-1};
  fastkag::Action action{};
  fastkag::Position position_before{};
  std::uint64_t pre_state_fingerprint{};
  std::uint64_t post_state_fingerprint{};
};

struct ProofSlot {
  int step{-1};
  fastkag::Action raw_source{};
  online_elastic::FrozenSlotKind source_kind{
      online_elastic::FrozenSlotKind::HardSemanticObligation};
  fastkag::Position actor_position_before{};
  fastkag::Action emitted{};
  int emitted_source_step{-1};
  std::uint64_t semantic_id{};
  int stamina_before{};
  int stamina_after{};
  std::uint64_t pre_state_fingerprint{};
  std::uint64_t post_state_fingerprint{};
};

struct MoveDelayAuthorization {
  int source_step{-1};
  int emitted_step{-1};
  fastkag::Action exact_move{};
};

// This is the authoritative proof object.  owner_projection is only a lossy
// compatibility view for the current OnlineElasticRepairOwner ABI; consumers
// must verify this richer certificate before installing that projection.
struct ProductionFrozenDaySuffixCertificate {
  int player{-1};
  int day{-1};
  int actor{-1};
  std::uint64_t actor_generation{};
  std::uint64_t issuer_generation{};
  std::uint64_t focal_phase_start_fingerprint{};
  std::uint64_t raw_day_route_hash{};
  WorkBudget budget;
  std::vector<RequiredInsertion> insertions;
  std::vector<HardSemanticObligation> hard_obligations;
  std::vector<ProofSlot> proof_slots;
  // This is the evaluator-facing route contract.  emitted_step may differ
  // from source_step, but both must be in the same day, sources stay strictly
  // ordered, payload bytes are exact, and every token is emitted once.
  std::vector<MoveDelayAuthorization> authorized_move_replays;
  online_elastic::FrozenDaySuffixCertificate owner_projection;
  std::uint64_t content_hash{};
};

struct VerificationResult {
  bool valid{};
  RejectReason reject{RejectReason::None};
  int checked_slots{};
};

struct IssueResult {
  RejectReason reject{RejectReason::None};
  std::optional<ProductionFrozenDaySuffixCertificate> certificate;
  std::optional<online_elastic::FrozenDaySuffixCertificate> owner_projection;
  std::vector<WitnessSlot> witness;
  std::uint64_t focal_phase_start_fingerprint{};
  int delayed_moves{};
  int maximum_move_delay{};
  int hard_obligations{};
  int certified_sinks{};
  int stamina_used{};

  [[nodiscard]] bool issued() const {
    return reject == RejectReason::None && certificate.has_value();
  }
};

// Stateful, monotonic issuer.  A (player, day, actor) key can be signed only
// once for the lifetime of this object.  There is deliberately no revoke or
// replacement API.
class ProductionSuffixScheduler {
 public:
  ProductionSuffixScheduler() = default;
  ProductionSuffixScheduler(const ProductionSuffixScheduler&) = delete;
  ProductionSuffixScheduler& operator=(const ProductionSuffixScheduler&) =
      delete;

  [[nodiscard]] IssueResult issue(const IssueRequest& request);
  [[nodiscard]] bool was_issued(int player, int day, int actor) const;

 private:
  std::map<std::tuple<int, int, int>, std::uint64_t> issued_hashes_;
};

[[nodiscard]] const char* reject_reason_name(RejectReason reason);
[[nodiscard]] std::uint64_t production_certificate_hash(
    const ProductionFrozenDaySuffixCertificate& certificate);
// Unit-local proof scope: selected player's complete Farm and PrivateState,
// plus immutable Config/step metadata.  Opponent private state is excluded.
// This is intentionally different from evaluator PrefixAuthority's full-joint
// fingerprint.
[[nodiscard]] std::uint64_t focal_unit_state_fingerprint(
    const fastkag::Simulator& state, int player);
[[nodiscard]] std::uint64_t focal_unit_state_fingerprint(
    const fastkag::Simulator& state, int player, int logical_step);
[[nodiscard]] VerificationResult verify_certificate(
    const ProductionFrozenDaySuffixCertificate& certificate,
    const fastkag::Simulator& phase_start,
    const std::vector<std::vector<fastkag::Action>>& raw_units_by_tick);

}  // namespace g001::production_suffix
