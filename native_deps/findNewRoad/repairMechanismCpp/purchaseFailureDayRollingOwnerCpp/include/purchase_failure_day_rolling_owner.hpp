#pragma once

#include "native_teammate.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace g001::purchase_failure_rolling {

enum class DebtKind : std::uint8_t {
  AcquireSeed,
  AcquireAnimal,
  PickupAnimal,
  PickupInput,
  Plant,
  Place,
  Water,
  Feed,
  Care,
  Harvest,
};

enum class DebtStatus : std::uint8_t {
  Open,
  AwaitingReceipt,
  Completed,
  Expired,
};

struct Debt {
  std::uint64_t id{};
  std::uint64_t root_id{};
  DebtKind kind{DebtKind::AcquireSeed};
  DebtStatus status{DebtStatus::Open};
  fastkag::Item item{fastkag::Item::NONE};
  fastkag::Action action{};
  int actor{-1};
  fastkag::Position tile{-1, -1};
  int source_step{-1};
  int earliest_step{-1};
  int deadline_step{-1};
  int attempts{};
  std::string provenance;
};

struct Config {
  // Research path is opt-in. Disabled mode is an exact native provider pass
  // through. Enabled defaults retry only a debt-window-needed, zero-filled
  // seed purchase once; the speculative cross-action chain remains opt-in.
  bool enabled{false};
  bool retry_purchases{true};
  bool retry_partial_purchases{false};
  bool retry_seed_purchases{true};
  bool retry_animal_purchases{false};
  bool require_route_seed_demand{true};
  bool require_route_animal_demand{true};
  int maximum_retry_quantity{1};
  bool block_unready_consumers{false};
  bool retry_unit_debts{false};
  bool monitor_natural_consumers{false};
  int maximum_attempts{3};
  int debt_days{2};
};

struct Metrics {
  int observations{};
  int proposals{};
  int finalized{};
  int purchase_failures{};
  int seed_purchase_failures{};
  int animal_purchase_failures{};
  int purchase_retries{};
  int purchase_retry_fills{};
  int unit_effect_failures{};
  int causal_blocks{};
  int unit_retries{};
  int completed_debts{};
  int expired_debts{};
  int displaced_typed_actions{};
  int unsupported_displacements{};
  int move_source{};
  int move_final{};
  int move_mismatch{};
  int stale_finalize_rejections{};
  int second_writer_rejections{};
  std::uint64_t proposal_time_ns{};
};

enum class FinalizeStatus : std::uint8_t {
  Selected,
  StaleProposal,
  SecondWriter,
};

// Immutable full-action authorization emitted by the sole owner. The caller
// may submit this action or reject it; it may not patch one field after this
// boundary and still commit owner/native state.
struct Proposal {
  struct ReceiptBinding {
    std::uint64_t debt_id{};
    bool market{};
    int index{-1};
  };
  int step{-1};
  std::uint64_t generation{};
  std::uint64_t observation_hash{};
  std::uint64_t base_action_hash{};
  std::uint64_t final_action_hash{};
  std::uint64_t binding_hash{};
  fastkag::PlayerAction base_action;
  fastkag::PlayerAction final_action;
  // Exact current physical state used only to construct the next receipt
  // checkpoint after finalize. No future state is present in a proposal.
  fastkag::Simulator observation_before;
  fastkag::NativeAgentState next_native_state;
  std::vector<Debt> debts_after_finalize;
  std::vector<std::uint64_t> attempted_debt_ids;
  std::vector<ReceiptBinding> receipt_bindings;
  std::vector<std::string> diagnostics;
};

class Owner {
 public:
  Owner(const fastkag::NativeTeammateExecutor& executor, int player,
        int route, Config config = {});

  // Settles only the exact previously finalized action against the current
  // physical simulator receipt. It must be called before propose().
  bool observe(const fastkag::Simulator& observation);

  // Side-effect free with respect to owner state. Native is run on a clone;
  // its candidate state is committed only by finalize().
  [[nodiscard]] Proposal propose(const fastkag::Simulator& observation) const;

  // Single-owner boundary: selected_action must equal the full bound action.
  // A partial or second writer is rejected without advancing either state.
  FinalizeStatus finalize(const Proposal& proposal,
                          const fastkag::PlayerAction& selected_action);

  // Finalizes after the repair composer and native final-action seam have
  // authorized additional, non-purchase unit edits. Every purchase-owned diff
  // must remain exact; committed_native_state must come from that exact replay.
  FinalizeStatus finalize_composed(
      const Proposal& proposal,
      const fastkag::PlayerAction& selected_action,
      fastkag::NativeAgentState committed_native_state);

  [[nodiscard]] const std::vector<Debt>& debts() const noexcept {
    return debts_;
  }
  [[nodiscard]] const Metrics& metrics() const noexcept { return metrics_; }
  [[nodiscard]] std::vector<std::string> terminal_diagnostics(
      int terminal_step) const;

 private:
  struct Pending {
    int step{-1};
    fastkag::Simulator before;
    fastkag::PlayerAction base_action;
    fastkag::PlayerAction final_action;
    std::vector<Proposal::ReceiptBinding> receipt_bindings;

    explicit Pending(const fastkag::Simulator& simulator) : before(simulator) {}
  };

  const fastkag::NativeTeammateExecutor* executor_{};
  int player_{-1};
  int route_{-1};
  Config config_{};
  fastkag::NativeAgentState native_state_;
  std::vector<Debt> debts_;
  std::optional<Pending> pending_;
  mutable Metrics metrics_;
  std::uint64_t generation_{1};
  std::uint64_t next_debt_id_{1};
  int last_observed_step_{-1};

  FinalizeStatus finalize_impl(
      const Proposal& proposal,
      const fastkag::PlayerAction& selected_action,
      fastkag::NativeAgentState committed_native_state,
      bool composed);
};

[[nodiscard]] const char* debt_kind_name(DebtKind kind) noexcept;
[[nodiscard]] const char* debt_status_name(DebtStatus status) noexcept;
[[nodiscard]] const char* finalize_status_name(FinalizeStatus status) noexcept;
[[nodiscard]] std::uint64_t action_hash(
    const fastkag::PlayerAction& action) noexcept;

namespace detail {
int uncovered_seed_demand(const fastkag::Simulator& observation, int player,
                          const std::vector<fastkag::PlayerAction>& tape,
                          const Debt& debt);
}

}  // namespace g001::purchase_failure_rolling
