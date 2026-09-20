#pragma once

#include "g001_real_weed_move_owner.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace g001::cross_day_debt_carry {

struct AuthorizedDependency {
  obligation_day::ProductionObligation obligation;
  std::uint64_t identity_hash{};
};

// Immutable cross-day authority. The original source remains part of this
// identity; the rebased day-8 DAG node deliberately has source_step=-1.
struct DebtAuthorization {
  int player{-1};
  int original_day{-1};
  int valid_day{-1};
  int expires_day{-1};
  std::uint64_t original_issuer_generation{};
  std::uint64_t original_certificate_hash{};
  std::uint64_t upstream_selection_hash{};
  obligation_day::ProductionObligation obligation;
  fastkag::Action original_source_action{};
  std::uint64_t obligation_identity_hash{};
  std::vector<AuthorizedDependency> dependencies;
  std::uint64_t day7_noncompletion_evidence_hash{};
  std::uint64_t authorization_generation{};
  std::uint64_t content_hash{};
};

enum class RevalidationReject : std::uint8_t {
  None = 0,
  InvalidEnvelope,
  Integrity,
  RegistryMismatch,
  Stale,
  RouteSourceMismatch,
  ActorUnavailable,
  TargetMismatch,
  DependencyMismatch,
  ResourceUnavailable,
  AlreadySatisfied,
};

struct RevalidationResult {
  RevalidationReject reject{RevalidationReject::None};
  bool admitted{};
  obligation_day::ProductionObligation carried;
  std::vector<obligation_day::ProductionObligation> dependency_witnesses;
};

class PersistentDebtRegistry {
 public:
  [[nodiscard]] bool enrolled() const noexcept { return canonical_hash_ != 0; }
  [[nodiscard]] std::uint64_t canonical_hash() const noexcept {
    return canonical_hash_;
  }

 private:
  std::uint64_t canonical_hash_{};
  friend std::optional<PersistentDebtRegistry> enroll_authorization(
      const DebtAuthorization& authorization);
};

[[nodiscard]] std::optional<PersistentDebtRegistry> enroll_authorization(
    const DebtAuthorization& authorization);

[[nodiscard]] std::uint64_t authorization_hash(
    const DebtAuthorization& authorization) noexcept;
[[nodiscard]] RevalidationResult revalidate(
    const DebtAuthorization& authorization,
    const PersistentDebtRegistry& registry,
    const fastkag::NativeTeammateExecutor& executor,
    const fastkag::Simulator& observation, int route);
[[nodiscard]] const char* reject_name(RevalidationReject reject) noexcept;

enum class FinishDisposition : std::uint8_t { Completed = 0, Carried, Expired };
struct FinishResult {
  FinishDisposition disposition{FinishDisposition::Expired};
  bool explicitly_closed{};
  DebtAuthorization next;
};
[[nodiscard]] FinishResult finish_authorization(
    const DebtAuthorization& authorization, int finished_day,
    bool physically_completed);

struct Metrics {
  std::uint64_t seed{};
  int seat{-1};
  int steps{};
  int action_divergences{};
  int paired_action_divergences{};
  int day7_commits{};
  int day8_commits{};
  int commit_failures{};
  int provider_state_leaks{};
  bool day7_debt_authorized{};
  std::uint64_t authorization_hash{};
  std::uint64_t debt_id{};
  int debt_source_step{-1};
  bool day8_revalidated{};
  RevalidationReject day8_revalidation_reject{RevalidationReject::None};
  int day8_start_x{-1};
  int day8_start_y{-1};
  int source191_emitted_step{-1};
  bool source191_physical_receipt{};
  bool source191_completed{};
  FinishDisposition source191_finish_disposition{FinishDisposition::Expired};
  int day8_receipt_checks{};
  int day8_receipt_failures{};
  int day8_move_expected{};
  int day8_move_emitted{};
  int day8_move_early{};
  int day8_move_duplicates{};
  int day8_move_drops{};
  int day8_new_debts{};
  std::vector<std::uint64_t> day8_new_debt_ids;
  std::vector<int> day8_new_debt_source_steps;
  std::vector<int> day8_new_debt_goals;
  std::vector<int> day8_new_debt_items;
  std::vector<int> day8_new_debt_x;
  std::vector<int> day8_new_debt_y;
  std::vector<int> day8_new_debt_dispositions;
  std::vector<int> day8_new_debt_remaining;
  int baseline_unit_failures{};
  int repair_unit_failures{};
  int baseline_market_failures{};
  int repair_market_failures{};
  std::vector<int> baseline_unit_failure_steps;
  std::vector<int> repair_unit_failure_steps;
  std::vector<int> baseline_market_failure_steps;
  std::vector<int> repair_market_failure_steps;
  double baseline_own{};
  double baseline_opponent{};
  double repair_own{};
  double repair_opponent{};
  double own_delta{};
  double margin_delta{};
  double score_delta{};
};

struct Report {
  bool default_off_parity{};
  bool stale_rejected{};
  bool tamper_rejected{};
  bool wrong_tile_rejected{};
  bool wrong_item_rejected{};
  bool midnight_duplicate_rejected{};
  bool no_terminal_oracle{};
  Metrics game;
  [[nodiscard]] std::string json() const;
};

[[nodiscard]] Report evaluate(std::string tapes, std::string library,
                              std::uint64_t seed = 970017, int seat = 1);
[[nodiscard]] Metrics evaluate_focused(std::string tapes, std::string library,
                                       std::uint64_t seed = 970017,
                                       int seat = 1);

}  // namespace g001::cross_day_debt_carry
