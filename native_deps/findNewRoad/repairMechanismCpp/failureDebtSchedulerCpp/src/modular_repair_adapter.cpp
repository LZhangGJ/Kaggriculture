#include "modular_repair_adapter.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

namespace g001::failure_debt::modular_bridge {
namespace {

bool same_action(const fastkag::Action& left, const fastkag::Action& right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool same_position(const fastkag::Position& left,
                   const fastkag::Position& right) {
  return left.x == right.x && left.y == right.y;
}

bool acquisition(fastkag::Op operation) {
  return operation == fastkag::Op::BUY_SEED ||
         operation == fastkag::Op::BUY_ANIMAL;
}

bool anchor_action(LifecycleKind kind, fastkag::Op operation) {
  return kind == LifecycleKind::Crop ? operation == fastkag::Op::PLANT
                                     : operation == fastkag::Op::PLACE;
}

bool active(DebtStatus status) {
  return status == DebtStatus::Open || status == DebtStatus::Attempted ||
         status == DebtStatus::Confirmed || status == DebtStatus::Scheduled ||
         status == DebtStatus::Failed;
}

int ceil_days(int delay_steps, int turns_per_day) {
  return delay_steps == 0 ? 0 : (delay_steps + turns_per_day - 1) / turns_per_day;
}

const PlannedTurn* find_turn(const PlanWindow& window, int step) {
  const auto found = std::find_if(window.turns.begin(), window.turns.end(),
      [&](const PlannedTurn& turn) { return turn.step == step; });
  return found == window.turns.end() ? nullptr : &*found;
}

const Debt* find_debt(const std::vector<Debt>& debts, std::uint64_t id) {
  const auto found = std::find_if(debts.begin(), debts.end(),
      [&](const Debt& debt) { return debt.id == id; });
  return found == debts.end() ? nullptr : &*found;
}

bool depends_on(std::uint64_t debt_id, std::uint64_t ancestor_id,
                const std::vector<Debt>& debts, std::set<std::uint64_t>& visited) {
  if (debt_id == ancestor_id) return true;
  if (!visited.insert(debt_id).second) return false;
  const auto* debt = find_debt(debts, debt_id);
  if (debt == nullptr) return false;
  for (const auto dependency : debt->dependencies) {
    if (depends_on(dependency, ancestor_id, debts, visited)) return true;
  }
  return false;
}

bool depends_on(std::uint64_t debt_id, std::uint64_t ancestor_id,
                const std::vector<Debt>& debts) {
  std::set<std::uint64_t> visited;
  return depends_on(debt_id, ancestor_id, debts, visited);
}

const UnitSchedule* find_schedule(const RepairPlan& plan, std::uint64_t id) {
  const auto found = std::find_if(plan.unit_schedule.begin(),
                                  plan.unit_schedule.end(),
      [&](const UnitSchedule& schedule) { return schedule.debt_id == id; });
  return found == plan.unit_schedule.end() ? nullptr : &*found;
}

void append_unrepaired(RepairDeltaProposal& proposal,
                       const RepairDeltaRequest& request,
                       const std::string& reason,
                       std::optional<int> purchase_unit_index = std::nullopt) {
  std::set<std::uint64_t> present;
  for (const auto& item : proposal.unrepaired) present.insert(item.debt_id);
  for (const auto& debt : request.debts) {
    if (debt.transaction_id != request.plan.transaction_id || !active(debt.status) ||
        (purchase_unit_index.has_value() &&
         debt.purchase_unit_index != *purchase_unit_index &&
         debt.kind != DebtKind::MarketPurchase) ||
        !present.insert(debt.id).second) {
      continue;
    }
    proposal.unrepaired.push_back({debt.id, debt.transaction_id,
                                   debt.purchase_unit_index, reason,
                                   debt.provenance});
  }
  if (proposal.unrepaired.empty()) {
    proposal.unrepaired.push_back({0, request.plan.transaction_id,
                                   purchase_unit_index.value_or(-1), reason,
                                   "proposal:unmapped-debt"});
  }
}

RepairDeltaProposal reject(const RepairDeltaRequest& request,
                           ProposalStatus status, std::string reason,
                           std::optional<int> purchase_unit_index = std::nullopt) {
  RepairDeltaProposal result;
  result.status = status;
  result.reason = std::move(reason);
  result.delta.key = request.key;
  result.delta.movement_hash = request.movement_hash;
  result.delta.base_obligation_dag_hash = request.base_obligation_dag_hash;
  append_unrepaired(result, request, result.reason, purchase_unit_index);
  result.audit.push_back("reject:" + result.reason);
  return result;
}

const CurrentUnitBinding* find_current_binding(
    const RepairDeltaRequest& request, const UnitPatch& patch) {
  const auto found = std::find_if(request.current_units.begin(),
                                  request.current_units.end(),
      [&](const CurrentUnitBinding& binding) {
        return binding.actor == patch.actor &&
               binding.source_slot == patch.source_slot;
      });
  return found == request.current_units.end() ? nullptr : &*found;
}

bool valid_base_owner(const modular_agent_core::IntentRef& owner) {
  return owner.intent_id != 0 &&
         (owner.owner == modular_agent_core::Owner::Route ||
          owner.owner == modular_agent_core::Owner::Production);
}

bool schedule_matches_window(const RepairDeltaRequest& request,
                             const Debt& debt,
                             const UnitSchedule& schedule,
                             std::string& reason) {
  const auto* turn = find_turn(request.plan_window, schedule.step);
  if (turn == nullptr || schedule.source_slot < 0 ||
      static_cast<std::size_t>(schedule.source_slot) >= turn->units.size()) {
    reason = "shifted_action_has_no_exact_plan_slot";
    return false;
  }
  const auto& slot = turn->units[static_cast<std::size_t>(schedule.source_slot)];
  if (slot.actor != schedule.actor || !same_position(slot.position, debt.position) ||
      is_movement(slot.original.op)) {
    reason = "shifted_action_would_change_move_or_tile";
    return false;
  }
  if (schedule.uses_existing_equivalent) {
    if (!same_action(slot.original, debt.action)) {
      reason = "existing_shifted_action_is_not_equivalent";
      return false;
    }
  } else {
    const int operation = static_cast<int>(debt.action.op);
    if (!slot.absorbable || slot.critical || operation < 0 ||
        operation >= static_cast<int>(kOperationCount) ||
        !slot.legal_replacements[static_cast<std::size_t>(operation)]) {
      reason = "shifted_action_slot_cannot_absorb_replacement";
      return false;
    }
  }
  return true;
}

}  // namespace

RepairDeltaProposal propose_repair_delta(const RepairDeltaRequest& request) {
  RepairDeltaProposal result;
  result.delta.key = request.key;
  result.delta.movement_hash = request.movement_hash;
  result.delta.base_obligation_dag_hash = request.base_obligation_dag_hash;
  if (!request.enabled) {
    result.status = ProposalStatus::Disabled;
    result.reason = "repair_delta_adapter_default_off";
    result.audit.push_back("disabled:no_output");
    return result;
  }
  if (request.key.abi_version != modular_agent_core::kAbiVersion ||
      request.key.step < 0 || request.key.observation_fingerprint == 0 ||
      request.movement_hash == 0 || request.base_obligation_dag_hash == 0 ||
      request.turns_per_day <= 0 || request.episode_steps <= 0 ||
      request.maximum_delay_days < 0) {
    return reject(request, ProposalStatus::InvalidInput,
                  "invalid_modular_repair_request");
  }
  if (!request.plan.accepted || !request.plan.day_lifecycle_complete) {
    return reject(request, ProposalStatus::Unrepaired,
                  request.plan.accepted ? "missing_complete_day_lifecycle"
                                        : "scheduler_plan_rejected");
  }
  if (request.plan.transaction_id == 0 ||
      request.plan.observation_fingerprint != request.key.observation_fingerprint) {
    return reject(request, ProposalStatus::InvalidInput,
                  "stale_or_unbound_scheduler_plan");
  }

  // Revalidate the scheduler certificate before translating any due node. No
  // orphan schedule/patch may disappear merely because it is in the future.
  std::set<std::uint64_t> schedule_ids;
  for (const auto& schedule : request.plan.unit_schedule) {
    const auto* debt = find_debt(request.debts, schedule.debt_id);
    std::string slot_reason;
    if (debt == nullptr || debt->transaction_id != request.plan.transaction_id ||
        debt->kind != DebtKind::UnitAction || debt->actor != schedule.actor ||
        !schedule_ids.insert(schedule.debt_id).second ||
        !schedule_matches_window(request, *debt, schedule, slot_reason)) {
      return reject(request, ProposalStatus::InvalidInput,
                    !slot_reason.empty() ? slot_reason
                                         : "invalid_unit_schedule_certificate");
    }
    const auto patch_count = std::count_if(
        request.plan.units.begin(), request.plan.units.end(),
        [&](const UnitPatch& patch) { return patch.debt_id == schedule.debt_id; });
    if ((schedule.uses_existing_equivalent && patch_count != 0) ||
        (!schedule.uses_existing_equivalent && patch_count != 1)) {
      return reject(request, ProposalStatus::InvalidInput,
                    "schedule_patch_cardinality_mismatch");
    }
  }
  const std::set<std::uint64_t> declared_schedule_ids(
      request.plan.scheduled_debt_ids.begin(),
      request.plan.scheduled_debt_ids.end());
  if (declared_schedule_ids.size() != request.plan.scheduled_debt_ids.size() ||
      declared_schedule_ids != schedule_ids) {
    return reject(request, ProposalStatus::InvalidInput,
                  "scheduled_debt_manifest_mismatch");
  }
  for (const auto& patch : request.plan.units) {
    const auto* debt = find_debt(request.debts, patch.debt_id);
    const auto* schedule = find_schedule(request.plan, patch.debt_id);
    if (debt == nullptr || schedule == nullptr ||
        patch.step != schedule->step || patch.actor != schedule->actor ||
        patch.source_slot != schedule->source_slot ||
        !same_action(patch.replacement, debt->action) ||
        is_movement(patch.original.op) || is_movement(patch.replacement.op)) {
      return reject(request, ProposalStatus::InvalidInput,
                    "unit_patch_does_not_match_scheduled_debt");
    }
  }

  std::set<std::pair<std::uint64_t, int>> continuation_keys;
  for (const auto& continuation : request.tile_continuations) {
    const auto key = std::pair{continuation.transaction_id,
                               continuation.purchase_unit_index};
    if (continuation.transaction_id != request.plan.transaction_id ||
        continuation.purchase_unit_index < 0 ||
        continuation.actions.empty() ||
        !continuation_keys.insert(key).second) {
      return reject(request, ProposalStatus::InvalidInput,
                    "invalid_or_duplicate_tile_continuation",
                    continuation.purchase_unit_index);
    }
    const auto& anchor = continuation.actions.front();
    const auto* anchor_debt = find_debt(request.debts, anchor.debt_id);
    const auto* anchor_schedule = find_schedule(request.plan, anchor.debt_id);
    if (anchor_debt == nullptr || anchor_schedule == nullptr ||
        anchor_debt->transaction_id != continuation.transaction_id ||
        anchor_debt->purchase_unit_index != continuation.purchase_unit_index ||
        anchor_debt->kind != DebtKind::UnitAction ||
        !anchor_action(continuation.kind, anchor_debt->action.op) ||
        !same_action(anchor.action, anchor_debt->action) ||
        !same_position(anchor_debt->position, continuation.tile) ||
        anchor.original_step < 0 || anchor_schedule->step < anchor.original_step) {
      return reject(request, ProposalStatus::Unrepaired,
                    "tile_anchor_not_proven", continuation.purchase_unit_index);
    }
    const int delay_steps = anchor_schedule->step - anchor.original_step;
    const int delay_days = ceil_days(delay_steps, request.turns_per_day);
    if (delay_days > request.maximum_delay_days) {
      return reject(request, ProposalStatus::Unrepaired,
                    "tile_delay_exceeds_day_budget",
                    continuation.purchase_unit_index);
    }

    int previous_original = -1;
    int previous_shifted = -1;
    std::uint64_t previous_debt_id{};
    std::set<std::uint64_t> chain_debt_ids;
    for (const auto& action : continuation.actions) {
      const auto* debt = find_debt(request.debts, action.debt_id);
      const auto* schedule = find_schedule(request.plan, action.debt_id);
      const int shifted_step = action.original_step + delay_steps;
      std::string slot_reason;
      if (debt == nullptr || schedule == nullptr ||
          debt->transaction_id != continuation.transaction_id ||
          debt->purchase_unit_index != continuation.purchase_unit_index ||
          debt->kind != DebtKind::UnitAction ||
          !same_position(debt->position, continuation.tile) ||
          !same_action(debt->action, action.action) ||
          action.original_step <= previous_original ||
          shifted_step <= previous_shifted || shifted_step >= request.episode_steps ||
          schedule->step != shifted_step ||
          (previous_debt_id != 0 &&
           !depends_on(action.debt_id, previous_debt_id, request.debts)) ||
          !chain_debt_ids.insert(action.debt_id).second ||
          !schedule_matches_window(request, *debt, *schedule, slot_reason)) {
        const std::string reason = !slot_reason.empty()
            ? slot_reason : "tile_continuation_not_shifted_as_a_whole";
        return reject(request, ProposalStatus::Unrepaired, reason,
                      continuation.purchase_unit_index);
      }
      previous_original = action.original_step;
      previous_shifted = shifted_step;
      previous_debt_id = action.debt_id;
      result.shifted_tile_actions.push_back(
          {action.debt_id, continuation.purchase_unit_index, continuation.tile,
           action.original_step, shifted_step, delay_steps, delay_days,
           action.action, continuation.provenance + ":" + action.provenance});
    }

    // No post-anchor unit debt on the tile may be silently omitted from the
    // caller's continuation certificate.
    for (const auto& debt : request.debts) {
      if (debt.transaction_id != continuation.transaction_id ||
          debt.purchase_unit_index != continuation.purchase_unit_index ||
          debt.kind != DebtKind::UnitAction ||
          !same_position(debt.position, continuation.tile) ||
          !depends_on(debt.id, anchor.debt_id, request.debts)) {
        continue;
      }
      if (!chain_debt_ids.count(debt.id)) {
        return reject(request, ProposalStatus::Unrepaired,
                      "tile_continuation_omits_downstream_debt",
                      continuation.purchase_unit_index);
      }
    }
  }

  // Every scheduled PLANT/PLACE chain must have a whole-tile continuation.
  for (const auto& schedule : request.plan.unit_schedule) {
    const auto* debt = find_debt(request.debts, schedule.debt_id);
    if (debt == nullptr || debt->transaction_id != request.plan.transaction_id ||
        (debt->action.op != fastkag::Op::PLANT &&
         debt->action.op != fastkag::Op::PLACE)) {
      continue;
    }
    if (!continuation_keys.count({debt->transaction_id,
                                  debt->purchase_unit_index})) {
      return reject(request, ProposalStatus::Unrepaired,
                    "plant_or_place_lacks_whole_tile_continuation",
                    debt->purchase_unit_index);
    }
  }

  std::set<std::uint64_t> emitted_ids;
  for (const auto& patch : request.plan.units) {
    if (patch.step != request.key.step) continue;
    const auto* debt = find_debt(request.debts, patch.debt_id);
    const auto* binding = find_current_binding(request, patch);
    if (debt == nullptr || binding == nullptr ||
        debt->transaction_id != request.plan.transaction_id ||
        debt->kind != DebtKind::UnitAction || debt->actor != patch.actor ||
        !same_action(binding->action, patch.original) ||
        !valid_base_owner(binding->owner) || is_movement(patch.original.op) ||
        is_movement(patch.replacement.op) ||
        !emitted_ids.insert(patch.debt_id).second) {
      return reject(request, ProposalStatus::InvalidInput,
                    "current_unit_patch_not_exactly_bound");
    }
    result.delta.unit_patches.push_back(
        {patch.actor, patch.original, binding->owner, patch.replacement,
         {modular_agent_core::Owner::Repair, patch.debt_id}});
  }

  if (request.plan.market.required &&
      !request.plan.market.uses_existing_surplus &&
      request.plan.market.step == request.key.step) {
    const auto found = std::find_if(request.debts.begin(), request.debts.end(),
        [&](const Debt& debt) {
          return debt.transaction_id == request.plan.transaction_id &&
                 debt.kind == DebtKind::MarketPurchase && active(debt.status);
        });
    if (found == request.debts.end() || !acquisition(request.plan.market.addition.op) ||
        request.plan.market.addition.quantity <= 0 ||
        !emitted_ids.insert(found->id).second) {
      return reject(request, ProposalStatus::InvalidInput,
                    "current_market_patch_not_exactly_bound");
    }
    result.delta.hard_market.push_back(
        {{modular_agent_core::Owner::Repair, found->id},
         request.plan.market.addition, found->economic_value, -1});
  }

  const bool emitted = !result.delta.unit_patches.empty() ||
                       !result.delta.hard_market.empty();
  result.delta.transaction_id = emitted ? request.plan.transaction_id : 0;
  result.status = emitted ? ProposalStatus::Accepted : ProposalStatus::Deferred;
  result.reason = emitted ? "current_repair_delta_proposed"
                          : "whole_tile_repair_certified_but_not_due";
  result.audit.push_back("proposal:" + result.reason);
  result.audit.push_back("movement_hash_bound_read_only");
  result.audit.push_back("market_queue_not_materialized");
  return result;
}

const char* proposal_status_name(ProposalStatus status) noexcept {
  switch (status) {
    case ProposalStatus::Disabled: return "disabled";
    case ProposalStatus::Accepted: return "accepted";
    case ProposalStatus::Deferred: return "deferred";
    case ProposalStatus::Unrepaired: return "unrepaired";
    case ProposalStatus::InvalidInput: return "invalid-input";
  }
  return "unknown";
}

}  // namespace g001::failure_debt::modular_bridge
