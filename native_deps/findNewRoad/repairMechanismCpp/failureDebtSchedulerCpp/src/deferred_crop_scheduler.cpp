#include "../include/deferred_crop_scheduler.hpp"

#include <algorithm>
#include <tuple>

namespace g001::failure_debt::deferred_crop {
namespace {

bool same_position(fastkag::Position left, fastkag::Position right) {
  return left.x == right.x && left.y == right.y;
}

bool same_action(const fastkag::Action& left, const fastkag::Action& right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool is_move(fastkag::Op operation) {
  return operation == fastkag::Op::NORTH ||
         operation == fastkag::Op::SOUTH ||
         operation == fastkag::Op::EAST ||
         operation == fastkag::Op::WEST;
}

bool valid_crop(fastkag::Item item) {
  const int value = static_cast<int>(item);
  return value >= 0 && value < fastkag::N_CROPS;
}

bool already_satisfied(const fastkag::Action& action,
                       const CropSnapshot& tile, int day) {
  switch (action.op) {
    case fastkag::Op::DIG:
      return tile.kind == fastkag::TileKind::EMPTY;
    case fastkag::Op::PLANT:
      return tile.kind == fastkag::TileKind::PLANT &&
             tile.crop == action.item;
    case fastkag::Op::WATER:
      return tile.kind == fastkag::TileKind::PLANT && tile.watered_today;
    case fastkag::Op::FERTILIZE:
      return tile.kind == fastkag::TileKind::PLANT &&
             tile.fertilized_until_day >= day + 2;
    case fastkag::Op::HARVEST:
      // EMPTY is ambiguous (harvest, decay, DIG, or another owner), so a
      // deferred HARVEST is never discharged from snapshot shape alone.
      return false;
    default:
      return false;
  }
}

bool exact_action_legal(const fastkag::Action& action,
                        const Visit& visit) {
  switch (action.op) {
    case fastkag::Op::DIG:
      return visit.tile.kind == fastkag::TileKind::WEED;
    case fastkag::Op::PLANT: {
      const int crop = static_cast<int>(action.item);
      return visit.tile.kind == fastkag::TileKind::EMPTY && valid_crop(action.item) &&
             visit.seeds[static_cast<std::size_t>(crop)] > 0;
    }
    case fastkag::Op::WATER:
      return visit.tile.kind == fastkag::TileKind::PLANT &&
             !visit.tile.watered_today;
    case fastkag::Op::HARVEST:
      return visit.tile.kind == fastkag::TileKind::PLANT &&
             visit.tile.harvest_legal && visit.tile.yield_units > 0;
    case fastkag::Op::FERTILIZE:
      return visit.tile.kind == fastkag::TileKind::PLANT &&
             visit.carried_fertilizer > 0 &&
             visit.tile.fertilized_until_day < visit.day + 2;
    default:
      return false;
  }
}

bool receipt_proves(const Proposal& proposal, const Receipt& receipt) {
  if (receipt.generic_effect_verified) return true;
  switch (proposal.action.op) {
    case fastkag::Op::DIG:
      return receipt.before.kind == fastkag::TileKind::WEED &&
             receipt.after.kind == fastkag::TileKind::EMPTY;
    case fastkag::Op::PLANT:
      return receipt.before.kind == fastkag::TileKind::EMPTY &&
             receipt.after.kind == fastkag::TileKind::PLANT &&
             receipt.after.crop == proposal.action.item;
    case fastkag::Op::WATER:
      return receipt.before.kind == fastkag::TileKind::PLANT &&
             !receipt.before.watered_today &&
             ((receipt.after.kind == fastkag::TileKind::PLANT &&
               receipt.after.watered_today) ||
              receipt.day_end_water_effect_lower_bound);
    case fastkag::Op::HARVEST:
      return receipt.before.kind == fastkag::TileKind::PLANT &&
             receipt.before.harvest_legal && receipt.before.yield_units > 0 &&
             (receipt.crop_inventory_delta > 0 ||
              receipt.after.kind == fastkag::TileKind::EMPTY ||
              (receipt.after.kind == fastkag::TileKind::PLANT &&
               receipt.after.yield_units < receipt.before.yield_units));
    case fastkag::Op::FERTILIZE:
      return receipt.before.kind == fastkag::TileKind::PLANT &&
             receipt.after.kind == fastkag::TileKind::PLANT &&
             receipt.after.fertilized_until_day >
                 receipt.before.fertilized_until_day &&
             receipt.fertilizer_inventory_delta < 0;
    default:
      return false;
  }
}

}  // namespace

bool is_crop_deferred_operation(fastkag::Op operation) noexcept {
  return operation == fastkag::Op::DIG || operation == fastkag::Op::PLANT ||
         operation == fastkag::Op::WATER ||
         operation == fastkag::Op::HARVEST ||
         operation == fastkag::Op::FERTILIZE;
}

DeferredCropScheduler::DeferredCropScheduler(Config config) : config_(config) {}

void DeferredCropScheduler::reset() {
  debts_.clear();
  pending_.clear();
  audit_ = {};
  next_debt_id_ = 1;
  next_obligation_id_ = 1;
}

bool DeferredCropScheduler::enqueue(const DeferredSource& source) {
  if (!config_.enabled || source.actor < 0 || source.source_step < 0 ||
      source.tile.x < 0 || source.tile.y < 0 ||
      !is_crop_deferred_operation(source.action.op) ||
      is_move(source.action.op) ||
      (source.action.op == fastkag::Op::PLANT &&
       !valid_crop(source.action.item))) {
    ++audit_.rejected_sources;
    return false;
  }
  const int origin_day = source.origin_day >= 0
      ? source.origin_day
      : source.source_step / std::max(1, config_.turns_per_day);
  auto debt = std::find_if(debts_.begin(), debts_.end(), [&](const TileDebt& item) {
    return !item.completed && item.actor == source.actor &&
           same_position(item.tile, source.tile) && item.origin_day == origin_day;
  });
  if (debt == debts_.end()) {
    TileDebt created;
    created.id = next_debt_id_++;
    created.actor = source.actor;
    created.tile = source.tile;
    created.origin_day = origin_day;
    created.created_step = source.source_step;
    created.earliest_deadline = source.deadline_step;
    created.desired_crop = source.action.op == fastkag::Op::PLANT
        ? source.action.item : source.remembered_crop;
    debts_.push_back(std::move(created));
    debt = std::prev(debts_.end());
  } else {
    debt->created_step = std::min(debt->created_step, source.source_step);
    if (source.deadline_step >= 0 &&
        (debt->earliest_deadline < 0 ||
         source.deadline_step < debt->earliest_deadline))
      debt->earliest_deadline = source.deadline_step;
    if (!valid_crop(debt->desired_crop)) {
      debt->desired_crop = source.action.op == fastkag::Op::PLANT
          ? source.action.item : source.remembered_crop;
    }
  }
  if (!debt->obligations.empty()) {
    auto& tail = debt->obligations.back();
    if (tail.state == ObligationState::Pending &&
        same_action(tail.action, source.action)) {
      ++tail.coalesced_sources;
      tail.deadline_step = tail.deadline_step < 0 ? source.deadline_step
          : (source.deadline_step < 0 ? tail.deadline_step
                                      : std::min(tail.deadline_step,
                                                 source.deadline_step));
      tail.critical = tail.critical || source.critical;
      tail.source_steps.push_back(source.source_step);
      tail.source_provenances.push_back(source.provenance);
      ++audit_.sources_enqueued;
      ++audit_.sources_coalesced;
      return true;
    }
  }
  Obligation obligation;
  obligation.id = next_obligation_id_++;
  obligation.source_step = source.source_step;
  obligation.deadline_step = source.deadline_step;
  obligation.action = source.action;
  obligation.critical = source.critical;
  obligation.provenance = source.provenance;
  obligation.source_steps.push_back(source.source_step);
  obligation.source_provenances.push_back(source.provenance);
  debt->obligations.push_back(std::move(obligation));
  ++audit_.sources_enqueued;
  return true;
}

Proposal DeferredCropScheduler::propose(const Visit& visit) const {
  Proposal result;
  result.step = visit.step;
  result.actor = visit.actor;
  result.tile = visit.position;
  result.original = visit.base_action;
  result.movement_hash = visit.movement_hash;
  if (!config_.enabled) {
    result.reason = "deferred_crop_scheduler_default_off";
    return result;
  }
  const bool actor_pending = std::any_of(
      pending_.begin(), pending_.end(), [&](const PendingReceipt& pending) {
        return pending.proposal.actor == visit.actor;
      });
  if (actor_pending) {
    result.status = ProposalStatus::WaitingForReceipt;
    result.reason = "actor_has_unsettled_receipt";
    return result;
  }
  std::vector<const TileDebt*> candidates;
  for (const auto& debt : debts_) {
    if (!debt.completed && debt.actor == visit.actor &&
        same_position(debt.tile, visit.position))
      candidates.push_back(&debt);
  }
  if (candidates.empty()) {
    result.status = ProposalStatus::NoDebtAtVisit;
    result.reason = "no_open_tile_debt_at_actor_visit";
    return result;
  }
  std::sort(candidates.begin(), candidates.end(), [&](const TileDebt* left,
                                                       const TileDebt* right) {
    const bool left_overdue = left->earliest_deadline >= 0 &&
                              visit.step > left->earliest_deadline;
    const bool right_overdue = right->earliest_deadline >= 0 &&
                               visit.step > right->earliest_deadline;
    return std::tuple{!left_overdue, left->created_step, left->id} <
           std::tuple{!right_overdue, right->created_step, right->id};
  });
  const auto& debt = *candidates.front();
  const auto head = std::find_if(debt.obligations.begin(), debt.obligations.end(),
      [](const Obligation& item) {
        return item.state != ObligationState::Confirmed;
      });
  if (head == debt.obligations.end()) {
    result.status = ProposalStatus::Blocked;
    result.reason = "completed_debt_awaits_compaction";
    return result;
  }
  result.debt_id = debt.id;
  result.obligation_id = head->id;

  fastkag::Action desired = head->action;
  bool synthetic = false;
  // State-derived weed/lifecycle prerequisites.  They never discharge the
  // source obligation; each needs its own receipt before the head is retried.
  if (head->action.op != fastkag::Op::DIG &&
      visit.tile.kind == fastkag::TileKind::WEED) {
    desired = {fastkag::Op::DIG};
    synthetic = true;
  } else if (head->action.op != fastkag::Op::PLANT &&
             visit.tile.kind == fastkag::TileKind::EMPTY &&
             valid_crop(debt.desired_crop)) {
    desired = {fastkag::Op::PLANT, debt.desired_crop, 1};
    synthetic = true;
  } else if (head->action.op == fastkag::Op::HARVEST &&
             visit.tile.kind == fastkag::TileKind::PLANT &&
             !visit.tile.harvest_legal && !visit.tile.watered_today) {
    desired = {fastkag::Op::WATER};
    synthetic = true;
  }

  result.action = desired;
  result.synthetic_prerequisite = synthetic;
  result.consumes_obligation_on_success = !synthetic;
  if (!synthetic && already_satisfied(desired, visit.tile, visit.day)) {
    result.status = ProposalStatus::ExistingEquivalent;
    result.reason = "current_snapshot_is_operation_equivalent";
    // No replacement is necessary.  The caller retains the base action and
    // commits an observation receipt for the equivalent postcondition.
    result.action = visit.base_action;
    return result;
  }
  if (!exact_action_legal(desired, visit)) {
    result.status = ProposalStatus::Blocked;
    if ((desired.op == fastkag::Op::PLANT) && visit.tile.kind == fastkag::TileKind::EMPTY)
      result.reason = "plant_waits_for_seed";
    else if (desired.op == fastkag::Op::HARVEST)
      result.reason = "harvest_waits_for_observed_maturity";
    else if (desired.op == fastkag::Op::FERTILIZE)
      result.reason = "fertilize_waits_for_inventory_or_crop";
    else
      result.reason = "head_obligation_not_legal_in_current_tile_state";
    return result;
  }
  if (is_move(visit.base_action.op) || visit.base_critical ||
      !visit.absorbable_slack ||
      visit.remaining_action_slots <= visit.remaining_moves) {
    result.status = ProposalStatus::Blocked;
    result.reason = is_move(visit.base_action.op)
        ? "move_is_read_only"
        : "no_certified_nonmove_slack_after_move_reservation";
    return result;
  }
  result.status = ProposalStatus::EmitReplacement;
  result.reason = synthetic ? "emit_crop_lifecycle_prerequisite"
                            : "emit_deferred_crop_obligation";
  return result;
}

bool DeferredCropScheduler::commit(const Proposal& proposal,
                                   const fastkag::Action& final_action,
                                   std::uint64_t final_movement_hash) {
  if (!proposal.actionable() || proposal.debt_id == 0 ||
      proposal.obligation_id == 0 || proposal.movement_hash == 0 ||
      final_movement_hash != proposal.movement_hash ||
      std::any_of(pending_.begin(), pending_.end(),
                  [&](const PendingReceipt& pending) {
                    return pending.proposal.actor == proposal.actor;
                  }))
    return false;
  auto* debt = find_debt(proposal.debt_id);
  if (debt == nullptr || debt->completed) return false;
  auto* obligation = find_obligation(*debt, proposal.obligation_id);
  if (obligation == nullptr || obligation->state != ObligationState::Pending)
    return false;
  if (proposal.status == ProposalStatus::EmitReplacement &&
      !same_action(final_action, proposal.action))
    return false;
  if (proposal.status == ProposalStatus::ExistingEquivalent &&
      !same_action(final_action, proposal.original))
    return false;
  obligation->state = ObligationState::InFlight;
  ++obligation->attempts;
  PendingReceipt pending;
  pending.proposal = proposal;
  pending_.push_back(std::move(pending));
  ++audit_.proposals;
  if (proposal.status == ProposalStatus::EmitReplacement) {
    ++audit_.replacements;
    if (proposal.synthetic_prerequisite) {
      if (proposal.action.op == fastkag::Op::DIG) ++audit_.synthetic_digs;
      if (proposal.action.op == fastkag::Op::PLANT) ++audit_.synthetic_plants;
      if (proposal.action.op == fastkag::Op::WATER) ++audit_.synthetic_waters;
    }
  } else {
    ++audit_.existing_equivalents;
  }
  return true;
}

bool DeferredCropScheduler::observe_receipt(const Receipt& receipt) {
  const auto found = std::find_if(pending_.begin(), pending_.end(),
      [&](const PendingReceipt& pending) {
        return pending.proposal.actor == receipt.actor &&
               pending.proposal.debt_id == receipt.debt_id &&
               pending.proposal.obligation_id == receipt.obligation_id;
      });
  if (found == pending_.end()) return false;
  auto* debt = find_debt(receipt.debt_id);
  if (debt == nullptr) return false;
  auto* obligation = find_obligation(*debt, receipt.obligation_id);
  if (obligation == nullptr || obligation->state != ObligationState::InFlight)
    return false;
  const Proposal proposal = found->proposal;
  const bool manifest_matches = receipt.movement_hash == proposal.movement_hash &&
      ((proposal.status == ProposalStatus::EmitReplacement &&
        same_action(receipt.emitted, proposal.action)) ||
       (proposal.status == ProposalStatus::ExistingEquivalent &&
        same_action(receipt.emitted, proposal.original)));
  bool success = false;
  if (manifest_matches) {
    if (proposal.status == ProposalStatus::ExistingEquivalent) {
      success = already_satisfied(obligation->action, receipt.before,
                                  receipt.step / std::max(1, config_.turns_per_day)) ||
                already_satisfied(obligation->action, receipt.after,
                                  receipt.step / std::max(1, config_.turns_per_day));
    } else {
      success = receipt_proves(proposal, receipt);
    }
  }
  if (success) {
    ++audit_.receipt_successes;
    if (proposal.consumes_obligation_on_success) {
      obligation->state = ObligationState::Confirmed;
      ++audit_.completed_obligations;
    } else {
      obligation->state = ObligationState::Pending;
    }
    const bool complete = std::all_of(
        debt->obligations.begin(), debt->obligations.end(),
        [](const Obligation& item) {
          return item.state == ObligationState::Confirmed;
        });
    if (complete && !debt->completed) {
      debt->completed = true;
      ++audit_.completed_tile_debts;
    }
  } else {
    obligation->state = ObligationState::Pending;
    ++audit_.receipt_failures;
  }
  pending_.erase(found);
  return success;
}

TileDebt* DeferredCropScheduler::find_debt(std::uint64_t id) {
  const auto found = std::find_if(debts_.begin(), debts_.end(),
      [&](const TileDebt& debt) { return debt.id == id; });
  return found == debts_.end() ? nullptr : &*found;
}

const TileDebt* DeferredCropScheduler::find_debt(std::uint64_t id) const {
  const auto found = std::find_if(debts_.begin(), debts_.end(),
      [&](const TileDebt& debt) { return debt.id == id; });
  return found == debts_.end() ? nullptr : &*found;
}

Obligation* DeferredCropScheduler::find_obligation(TileDebt& debt,
                                                    std::uint64_t id) {
  const auto found = std::find_if(debt.obligations.begin(), debt.obligations.end(),
      [&](const Obligation& item) { return item.id == id; });
  return found == debt.obligations.end() ? nullptr : &*found;
}

std::vector<TileDebt> DeferredCropScheduler::open_debts(int current_step) const {
  std::vector<TileDebt> result;
  for (const auto& debt : debts_) if (!debt.completed) result.push_back(debt);
  std::sort(result.begin(), result.end(), [&](const TileDebt& left,
                                             const TileDebt& right) {
    const bool left_overdue = left.earliest_deadline >= 0 &&
                              current_step > left.earliest_deadline;
    const bool right_overdue = right.earliest_deadline >= 0 &&
                               current_step > right.earliest_deadline;
    return std::tuple{!left_overdue, left.created_step, left.id} <
           std::tuple{!right_overdue, right.created_step, right.id};
  });
  return result;
}

AuditCounters DeferredCropScheduler::audit(int current_step) const {
  auto result = audit_;
  result.overdue_open_obligations = 0;
  for (const auto& debt : debts_) {
    if (debt.completed) continue;
    for (const auto& obligation : debt.obligations)
      if (obligation.state != ObligationState::Confirmed &&
          obligation.deadline_step >= 0 &&
          current_step > obligation.deadline_step)
        ++result.overdue_open_obligations;
  }
  return result;
}

const char* proposal_status_name(ProposalStatus status) noexcept {
  switch (status) {
    case ProposalStatus::Disabled: return "disabled";
    case ProposalStatus::NoDebtAtVisit: return "no-debt-at-visit";
    case ProposalStatus::WaitingForReceipt: return "waiting-for-receipt";
    case ProposalStatus::Blocked: return "blocked";
    case ProposalStatus::ExistingEquivalent: return "existing-equivalent";
    case ProposalStatus::EmitReplacement: return "emit-replacement";
  }
  return "unknown";
}

}  // namespace g001::failure_debt::deferred_crop
