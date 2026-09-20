#pragma once

#include "failure_debt_scheduler.hpp"
#include "modular_agent_core.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace g001::failure_debt::modular_bridge {

enum class ProposalStatus : std::uint8_t {
  Disabled,
  Accepted,
  Deferred,
  Unrepaired,
  InvalidInput,
};

// Exact base-unit ownership at the current step. The modular core uses this
// binding to reject a repair that targets a different Route/Production action.
struct CurrentUnitBinding {
  int actor{-1};
  int source_slot{-1};
  fastkag::Action action{};
  modular_agent_core::IntentRef owner{};
};

// Caller-certified original action on one tile. The list starts at PLANT or
// PLACE and includes every later production action on that tile through the
// caller's chosen lifecycle/occupancy boundary.
struct TileContinuationAction {
  std::uint64_t debt_id{};
  int original_step{-1};
  fastkag::Action action{};
  std::string provenance;
};

struct TileContinuation {
  std::uint64_t transaction_id{};
  int purchase_unit_index{-1};
  fastkag::Position tile{-1, -1};
  LifecycleKind kind{LifecycleKind::Crop};
  std::vector<TileContinuationAction> actions;
  std::string provenance;
};

struct ShiftedTileAction {
  std::uint64_t debt_id{};
  int purchase_unit_index{-1};
  fastkag::Position tile{-1, -1};
  int original_step{-1};
  int shifted_step{-1};
  int delay_steps{};
  int delay_days{};
  fastkag::Action action{};
  std::string provenance;
};

struct UnrepairedDebt {
  std::uint64_t debt_id{};
  std::uint64_t transaction_id{};
  int purchase_unit_index{-1};
  std::string reason;
  std::string provenance;
};

struct RepairDeltaRequest {
  // Separate from FailureDebtScheduler::enabled. Both seams remain default-off.
  bool enabled{false};
  modular_agent_core::SharedKey key;
  std::uint64_t movement_hash{};
  std::uint64_t base_obligation_dag_hash{};
  int turns_per_day{24};
  int episode_steps{720};
  // Delay is measured as ceil(delay_steps / turns_per_day). Zero forbids any
  // delay; this is a day budget, not a free step horizon.
  int maximum_delay_days{};
  RepairPlan plan;
  PlanWindow plan_window;
  std::vector<Debt> debts;
  std::vector<CurrentUnitBinding> current_units;
  std::vector<TileContinuation> tile_continuations;
};

// This is a proposal only. It contains no PlayerAction, MOVE directive, final
// market queue, compose call, or scheduler commit side effect.
struct RepairDeltaProposal {
  ProposalStatus status{ProposalStatus::Disabled};
  std::string reason;
  modular_agent_core::RepairDelta delta;
  std::vector<ShiftedTileAction> shifted_tile_actions;
  std::vector<UnrepairedDebt> unrepaired;
  std::vector<std::string> audit;
  [[nodiscard]] bool accepted() const {
    return status == ProposalStatus::Accepted ||
           status == ProposalStatus::Deferred;
  }
};

[[nodiscard]] RepairDeltaProposal propose_repair_delta(
    const RepairDeltaRequest& request);

[[nodiscard]] const char* proposal_status_name(ProposalStatus status) noexcept;

}  // namespace g001::failure_debt::modular_bridge
