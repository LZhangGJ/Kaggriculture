#pragma once

#include "g001_day_start_obligation_issuer.hpp"
#include "repair_minimum_damage_selector.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace g001::minimum_damage_bridge {

struct ObligationPolicy {
  std::uint64_t obligation_id{};
  bool important{};
  int cross_day_cascade_risk{};
};

struct PurchaseRetry {
  std::uint64_t id{};
  fastkag::Op operation{fastkag::Op::PASS};
  fastkag::Item item{fastkag::Item::NONE};
  int quantity{};
  int release_step{-1};
  int worst_case_cash_cost{};
  bool guaranteed_fill{};
  bool must_finish{};
  bool important{};
  int cross_day_cascade_risk{};
  std::vector<std::uint64_t> unlocks_obligations;
};

// A rolling request must account for every selected actor's production source that is
// older than the observation.  `completed` may only be asserted by an
// observation adapter with concrete effect evidence.  An uncompleted entry is
// carried into the suffix selector with its original typed identity.
struct PriorObligationProgress {
  std::uint64_t obligation_id{};
  int original_source_step{-1};
  bool completed{};
  std::uint64_t effect_evidence_hash{};
};

struct PriorMoveProgress {
  int original_source_step{-1};
  bool completed{};
  std::uint64_t effect_evidence_hash{};
};

struct Request {
  const fastkag::Simulator* remaining_day_start{};
  int player{-1};
  int actor{};
  std::uint64_t issuer_generation{};
  const day_start_issuer::IssueResult* issued{};
  std::span<const ObligationPolicy> policies;
  std::span<const PurchaseRetry> purchase_retries;
  std::size_t maximum_states{2'000'000};
  std::span<const PriorObligationProgress> prior_obligation_progress;
  std::span<const PriorMoveProgress> prior_move_progress;
};

enum class RejectReason : std::uint8_t {
  None = 0,
  InvalidInput,
  ActorScope,
  IssuerRejected,
  PriorSourceRequiresReceipt,
  UnsupportedSource,
  MissingTypedIdentity,
  AmbiguousSourceMapping,
  UnsupportedSnapshotState,
  SelectorRejected,
  InternalVerificationFailure,
};

struct SignedSlot {
  int step{-1};
  fastkag::Action unit{};
  std::optional<fastkag::Action> market;
  int raw_move_source_step{-1};
  std::uint64_t obligation_id{};
  int transition_index{-1};
  std::uint64_t market_obligation_id{};
  int market_transition_index{-1};
};

struct SignedDebt {
  std::uint64_t obligation_id{};
  minimum_damage::DebtReason selector_reason{
      minimum_damage::DebtReason::Capacity};
  obligation_day::DebtReason scheduler_reason{
      obligation_day::DebtReason::Capacity};
  int completed_transitions{};
  int remaining_transitions{};
  int bound_day_offset{1};
};

struct Certificate {
  int player{-1};
  int actor{-1};
  int day{-1};
  int start_step{-1};
  int end_step{-1};
  std::uint64_t issuer_generation{};
  std::uint64_t input_hash{};
  minimum_damage::Objective objective{};
  std::vector<SignedSlot> slots;
  std::vector<SignedDebt> debts;
  int raw_move_tokens{};
  int emitted_raw_moves{};
  std::size_t selector_explored_states{};
  std::uint64_t content_hash{};
};

struct IssueResult {
  RejectReason reject{RejectReason::None};
  std::optional<Certificate> certificate;
  std::string diagnostic;
  [[nodiscard]] bool issued() const noexcept {
    return reject == RejectReason::None && certificate.has_value();
  }
};

struct VerifyResult {
  bool valid{};
  RejectReason reject{RejectReason::None};
  std::string diagnostic;
  int checked_slots{};
  int checked_move_tokens{};
};

[[nodiscard]] IssueResult issue(const Request& request);
[[nodiscard]] VerifyResult verify(const Request& request,
                                  const Certificate& certificate);
[[nodiscard]] std::uint64_t certificate_hash(
    const Certificate& certificate) noexcept;
[[nodiscard]] const char* reject_reason_name(RejectReason reason) noexcept;

}  // namespace g001::minimum_damage_bridge
