#pragma once

#include "native_teammate.hpp"
#include "production_suffix_scheduler.hpp"

#include <cstdint>
#include <optional>

namespace g001::native_final_commit {

enum class OverlayReplayBehavior : std::uint8_t {
  None = 0,
  ProposalOnly,
  RepeatsOnReplay,
};

struct StatefulOverlayInjection {
  int actor{-1};
  fastkag::Action proposal_action{};
  int wheat_credit_delta{};
  OverlayReplayBehavior replay_behavior{OverlayReplayBehavior::None};
};

struct Request {
  const fastkag::NativeTeammateExecutor* executor{};
  const fastkag::Simulator* observation{};
  int player{-1};
  int route{-1};
  fastkag::NativeRepairOptions repair_options{};
  std::optional<StatefulOverlayInjection> injection;
};

struct Proposal {
  int player{-1};
  int route{-1};
  int step{-1};
  std::uint64_t observation_fingerprint{};
  fastkag::NativeRepairOptions repair_options{};
  fastkag::NativeAgentState base_state;
  fastkag::NativeAgentState candidate_state;
  fastkag::PlayerAction action;
  std::optional<StatefulOverlayInjection> injection;
};

// Caller-created authorization for one complete final PlayerAction on the
// exact proposal observation. A source_step alone is deliberately not an
// identity: player, route, clock, observation and the entire units+market
// payload all enter content_hash.
struct FinalActionBinding {
  int player{-1};
  int route{-1};
  int step{-1};
  int owned_actor{-1};
  std::uint64_t owned_actor_mask{};
  bool owns_market_tail{};
  std::uint64_t owner_generation{};
  std::uint64_t owner_certificate_hash{};
  std::uint64_t observation_fingerprint{};
  std::uint64_t action_fingerprint{};
  std::uint64_t content_hash{};
};

enum class CommitReject : std::uint8_t {
  None = 0,
  InvalidRequest,
  ObservationChanged,
  FinalShape,
  BindingMismatch,
  ReplayMismatch,
};

struct CommitResult {
  CommitReject reject{CommitReject::None};
  bool committed{};
  bool candidate_discarded{};
  bool proposal_stateful_mutation{};
  fastkag::PlayerAction replayed_action;
};

[[nodiscard]] Proposal propose(const Request& request,
                               const fastkag::NativeAgentState& state);
[[nodiscard]] std::uint64_t action_fingerprint(
    const fastkag::PlayerAction& action) noexcept;
[[nodiscard]] FinalActionBinding bind_final_action(
    const Proposal& proposal, int owned_actor,
    std::uint64_t owner_generation, std::uint64_t owner_certificate_hash,
    const fastkag::PlayerAction& final_action);
[[nodiscard]] FinalActionBinding bind_repair_final_action(
    const Proposal& proposal, std::uint64_t owned_actor_mask,
    bool owns_market_tail, std::uint64_t owner_generation,
    std::uint64_t owner_certificate_hash,
    const fastkag::PlayerAction& final_action);

// Replays from Proposal::base_state and commits only when the replayed full
// PlayerAction is byte-identical to final_action. No field-level whitelist is
// used: unknown NativeAgentState mutations are therefore either reconstructed
// by the exact replay or rejected with the candidate left uncommitted.
[[nodiscard]] CommitResult commit(
    const Request& request, const Proposal& proposal,
    const fastkag::PlayerAction& final_action,
    fastkag::NativeAgentState& state);

// Uses the native default-off final-composer seam. The replay runs from the
// saved base state with the caller final action injected after all composers
// and before final-action-dependent staging. State is assigned only after the
// returned full action is exact.
[[nodiscard]] CommitResult commit_weed_owner_finalized(
    const Request& request, const Proposal& proposal,
    const FinalActionBinding& binding,
    const fastkag::PlayerAction& final_action,
    fastkag::NativeAgentState& state);

[[nodiscard]] CommitResult commit_repair_owner_finalized(
    const Request& request, const Proposal& proposal,
    const FinalActionBinding& binding,
    const fastkag::PlayerAction& final_action,
    fastkag::NativeAgentState& state);

[[nodiscard]] bool same_action(const fastkag::PlayerAction& lhs,
                               const fastkag::PlayerAction& rhs) noexcept;
[[nodiscard]] const char* commit_reject_name(CommitReject reject) noexcept;

}  // namespace g001::native_final_commit
