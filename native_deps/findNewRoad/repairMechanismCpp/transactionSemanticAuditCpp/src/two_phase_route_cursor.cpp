#include "two_phase_route_cursor.hpp"

#include <algorithm>
#include <stdexcept>

namespace g001::repair_audit::two_phase {
namespace {

bool same_action(const fastkag::Action& left,
                 const fastkag::Action& right) noexcept {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool contains_ordered(const std::vector<LifecycleStage>& stages,
                      const std::vector<fastkag::Op>& required) {
  std::size_t cursor = 0;
  for (const auto& stage : stages)
    if (cursor < required.size() && stage.action.op == required[cursor])
      ++cursor;
  return cursor == required.size();
}

void validate(const LifecycleContract& contract) {
  using fastkag::Op;
  using modular_agent_core::Owner;
  if (contract.transaction_id == 0 || contract.stages.empty())
    throw std::invalid_argument("lifecycle contract lacks transaction/stages");
  for (std::size_t index = 0; index < contract.stages.size(); ++index) {
    const auto& stage = contract.stages[index];
    if (stage.intent.owner != Owner::Repair || stage.intent.intent_id == 0 ||
        stage.action.quantity <= 0)
      throw std::invalid_argument("lifecycle stage lacks stable repair intent");
    if (std::count_if(contract.stages.begin(), contract.stages.end(),
                      [&](const LifecycleStage& value) {
                        return value.intent == stage.intent;
                      }) != 1)
      throw std::invalid_argument("duplicate lifecycle intent");
  }
  const auto& terminal = contract.stages.back();
  if (!terminal.first_yield || terminal.action.op != Op::HARVEST ||
      terminal.phase != StagePhase::Unit)
    throw std::invalid_argument("terminal lifecycle stage is not first yield");
  if (contract.kind == failure_debt::LifecycleKind::Crop) {
    // A repaired crop transaction is not closed by the first delayed harvest:
    // it owns replant, water and the replanted crop's first verified yield.
    if (!contains_ordered(contract.stages,
                          {Op::PLANT, Op::WATER, Op::HARVEST,
                           Op::PLANT, Op::WATER, Op::HARVEST}))
      throw std::invalid_argument("crop contract omits replant first-yield suffix");
  } else {
    if (!contains_ordered(contract.stages,
                          {Op::BUY_ANIMAL, Op::PICKUP, Op::PLACE, Op::FEED,
                           Op::CARE, Op::HARVEST}))
      throw std::invalid_argument("animal contract omits acquisition/maintenance/first-yield suffix");
    if (contract.stages.front().phase != StagePhase::Market ||
        contract.stages.front().action.op != Op::BUY_ANIMAL)
      throw std::invalid_argument("animal lifecycle must start at acquisition");
  }
}

void account(const LifecycleStage& stage, CommitAudit& audit) {
  using fastkag::Op;
  switch (stage.action.op) {
    case Op::BUY_SEED:
    case Op::BUY_ANIMAL: ++audit.acquisitions; break;
    case Op::PICKUP: ++audit.pickups; break;
    case Op::PLACE: ++audit.places; break;
    case Op::PLANT: ++audit.plants; break;
    case Op::WATER: ++audit.waters; break;
    case Op::FEED: ++audit.feeds; break;
    case Op::CARE: ++audit.cares; break;
    case Op::HARVEST: ++audit.harvest_effects; break;
    default: break;
  }
}

}  // namespace

Coordinator::Coordinator(LifecycleContract contract, int source_cursor)
    : contract_(std::move(contract)), source_cursor_(source_cursor) {
  validate(contract_);
}

Proposal Coordinator::propose(
    const failure_debt::modular_bridge::RepairDeltaProposal& repair) const {
  Proposal out;
  if (completed_) {
    out.status = ProposalStatus::Completed;
    out.reason = "transaction already completed from first-yield receipt";
    return out;
  }
  if (pending_) {
    out.status = ProposalStatus::AwaitingObservation;
    out.reason = "previous final submission awaits next observation";
    return out;
  }
  if (!repair.accepted() || repair.delta.transaction_id != contract_.transaction_id ||
      next_stage_ >= contract_.stages.size()) {
    out.status = ProposalStatus::InvalidRepairDelta;
    out.reason = "repair proposal does not bind the open lifecycle transaction";
    return out;
  }
  const auto& stage = contract_.stages[next_stage_];
  bool found = false;
  if (stage.phase == StagePhase::Unit) {
    found = std::count_if(repair.delta.unit_patches.begin(),
                         repair.delta.unit_patches.end(), [&](const auto& patch) {
      return patch.intent == stage.intent && same_action(patch.replacement, stage.action);
    }) == 1;
  } else {
    found = std::count_if(repair.delta.hard_market.begin(),
                         repair.delta.hard_market.end(), [&](const auto& intent) {
      return intent.intent == stage.intent && same_action(intent.action, stage.action);
    }) == 1;
  }
  if (!found) {
    out.status = ProposalStatus::InvalidRepairDelta;
    out.reason = "current lifecycle intent is absent or ambiguous in RepairDelta";
    return out;
  }
  out.status = ProposalStatus::Accepted;
  out.reason = "side-effect-free lifecycle proposal";
  out.delta = repair.delta;
  out.expected = stage;
  out.stage_index = next_stage_;
  out.source_cursor_before = source_cursor_;
  out.source_cursor_after = source_cursor_ + (stage.consumes_source ? 1 : 0);
  return out;
}

FinalStatus Coordinator::stage_final(
    const Proposal& proposal,
    const modular_agent_core::ExecutionManifest& manifest) {
  if (!proposal.accepted() || proposal.stage_index != next_stage_ || pending_ ||
      proposal.source_cursor_before != source_cursor_)
    return FinalStatus::StaleProposal;
  if (!(manifest.key == proposal.delta.key) ||
      manifest.action_hash == 0)
    return FinalStatus::InvalidManifest;

  bool selected = false;
  if (proposal.expected.phase == StagePhase::Unit) {
    selected = std::count_if(manifest.units.begin(), manifest.units.end(),
                             [&](const auto& binding) {
      return binding.owner == proposal.expected.intent &&
             same_action(binding.emitted, proposal.expected.action);
    }) == 1;
  } else {
    selected = std::count_if(manifest.market.begin(), manifest.market.end(),
                             [&](const auto& binding) {
      return binding.owner == proposal.expected.intent &&
             same_action(binding.emitted, proposal.expected.action);
    }) == 1;
  }
  if (!selected) {
    ++audit_.overlay_overrides;
    return FinalStatus::Overridden;
  }
  if (manifest.repair_transaction_id != contract_.transaction_id)
    return FinalStatus::InvalidManifest;
  pending_ = Pending{proposal.stage_index, proposal.expected,
                     manifest.action_hash, manifest.key};
  return FinalStatus::Selected;
}

CommitStatus Coordinator::observe(
    const modular_agent_core::ExecutionReceipt& receipt) {
  if (!pending_) return CommitStatus::NoPendingSubmission;
  if (!(receipt.key == pending_->key) ||
      receipt.submitted_action_hash != pending_->action_hash ||
      receipt.repair_transaction_id != contract_.transaction_id)
    return CommitStatus::StaleReceipt;

  bool confirmed = false;
  if (pending_->expected.phase == StagePhase::Unit) {
    confirmed = std::count_if(receipt.units.begin(), receipt.units.end(),
                              [&](const auto& unit) {
      return unit.intent == pending_->expected.intent &&
             same_action(unit.submitted, pending_->expected.action) &&
             unit.effect_confirmed;
    }) == 1;
  } else {
    confirmed = std::count_if(receipt.market.begin(), receipt.market.end(),
                              [&](const auto& market) {
      return market.intent == pending_->expected.intent &&
             market.submitted == pending_->expected.action.quantity &&
             market.fill.lower >= pending_->expected.action.quantity;
    }) == 1;
  }
  if (!confirmed) {
    ++audit_.effect_failures;
    pending_.reset();
    return CommitStatus::EffectNotConfirmed;
  }

  const auto committed = pending_->expected;
  if (committed.consumes_source) ++source_cursor_;
  account(committed, audit_);
  ++next_stage_;
  pending_.reset();
  if (next_stage_ == contract_.stages.size()) {
    // validate() guarantees that only a verified terminal HARVEST receipt can
    // cross this boundary.
    completed_ = true;
    ++audit_.completed_transactions;
  }
  return CommitStatus::Committed;
}

StateSnapshot Coordinator::snapshot() const noexcept {
  return {source_cursor_, next_stage_, pending_.has_value(), completed_, audit_};
}

const char* proposal_status_name(ProposalStatus status) noexcept {
  switch (status) {
    case ProposalStatus::Accepted: return "accepted";
    case ProposalStatus::Completed: return "completed";
    case ProposalStatus::AwaitingObservation: return "awaiting-observation";
    case ProposalStatus::InvalidRepairDelta: return "invalid-repair-delta";
  }
  return "unknown";
}

const char* final_status_name(FinalStatus status) noexcept {
  switch (status) {
    case FinalStatus::Selected: return "selected";
    case FinalStatus::Overridden: return "overridden";
    case FinalStatus::StaleProposal: return "stale-proposal";
    case FinalStatus::InvalidManifest: return "invalid-manifest";
  }
  return "unknown";
}

const char* commit_status_name(CommitStatus status) noexcept {
  switch (status) {
    case CommitStatus::Committed: return "committed";
    case CommitStatus::EffectNotConfirmed: return "effect-not-confirmed";
    case CommitStatus::NoPendingSubmission: return "no-pending-submission";
    case CommitStatus::StaleReceipt: return "stale-receipt";
  }
  return "unknown";
}

}  // namespace g001::repair_audit::two_phase
