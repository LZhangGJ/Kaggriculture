#include "purchase_recovery_ledger.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace g001::purchase_recovery {
namespace {

int positive(int value) { return std::max(0, value); }

} // namespace

std::uint64_t Ledger::open(Obligation obligation) {
  if (!supported(obligation))
    throw std::invalid_argument(
        "purchase recovery supports only valid BUY_SEED/BUY_ANIMAL debts");
  DebtState state;
  state.view.id = next_debt_id_++;
  state.view.obligation = std::move(obligation);
  state.view.remaining = state.view.obligation.quantity;
  debts_.push_back(std::move(state));
  return debts_.back().view.id;
}

std::optional<Proposal>
Ledger::describe_hard_order(std::uint64_t debt_id,
                            int requested_quantity) const {
  const auto *state = find(debt_id);
  if (!state || state->view.awaiting_receipt ||
      state->view.acquisition_complete || requested_quantity <= 0 ||
      requested_quantity > std::numeric_limits<std::int32_t>::max())
    return std::nullopt;
  return Proposal{state->view.id,
                  state->revision,
                  -1,
                  ProposalOrigin::ExistingHardObligation,
                  {state->view.obligation.operation,
                   state->view.obligation.item, requested_quantity},
                  state->view.obligation.unit_cost};
}

std::optional<Proposal>
Ledger::propose_recovery(std::uint64_t debt_id,
                         const FundingObservation &observation) const {
  const auto *state = find(debt_id);
  if (!state || state->view.awaiting_receipt ||
      state->view.acquisition_complete || observation.step < 0)
    return std::nullopt;
  const auto raw_spendable = static_cast<std::int64_t>(observation.cash) -
                             positive(observation.protected_cash);
  const int spendable = static_cast<int>(
      std::min<std::int64_t>(std::numeric_limits<int>::max(),
                             std::max<std::int64_t>(0, raw_spendable)));
  const int affordable = spendable / state->view.obligation.unit_cost;
  const int requested = std::min(state->view.remaining, affordable);
  if (requested <= 0)
    return std::nullopt;
  return Proposal{state->view.id,
                  state->revision,
                  observation.step,
                  ProposalOrigin::Recovery,
                  {state->view.obligation.operation,
                   state->view.obligation.item, requested},
                  state->view.obligation.unit_cost};
}

StageStatus Ledger::stage_final(const Proposal &proposal,
                                const FinalSelection &selection) {
  auto reject = [&](StageStatus status) {
    ++audit_.final_selection_rejections;
    return status;
  };
  auto *state = find(proposal.debt_id);
  if (!state)
    return reject(StageStatus::UnknownDebt);
  if (proposal.revision != state->revision ||
      proposal.order.op != state->view.obligation.operation ||
      proposal.order.item != state->view.obligation.item ||
      proposal.unit_cost != state->view.obligation.unit_cost ||
      proposal.order.quantity <= 0 ||
      (proposal.origin == ProposalOrigin::Recovery &&
       proposal.order.quantity > state->view.remaining))
    return reject(StageStatus::StaleProposal);
  if (state->view.awaiting_receipt)
    return reject(StageStatus::AlreadyPending);
  if (selection.submitted_step < 0 || selection.selected_slot < 0 ||
      selection.selected_slot >=
          static_cast<int>(selection.final_market.size()))
    return reject(StageStatus::InvalidSlot);
  if (!same_action(selection.final_market[static_cast<std::size_t>(
                       selection.selected_slot)],
                   proposal.order))
    return reject(StageStatus::FinalOrderMismatch);
  if (std::any_of(pending_.begin(), pending_.end(), [&](const Pending &value) {
        return value.submitted_step == selection.submitted_step &&
               value.slot == selection.selected_slot;
      }))
    return reject(StageStatus::SlotAlreadyBound);

  // One inventory-delta witness cannot safely attribute two independent
  // orders for the same acquisition/item in the same turn. The composer may
  // aggregate them under one debt or defer one proposal.
  if (std::any_of(pending_.begin(), pending_.end(), [&](const Pending &value) {
        return value.submitted_step == selection.submitted_step &&
               value.exact_order.op == proposal.order.op &&
               value.exact_order.item == proposal.order.item;
      }))
    return reject(StageStatus::ConcurrentHoldingWitness);

  // A recovery proposal must still be the exact quantity returned by the
  // funding calculation. The final composer may reject or omit it, but may
  // not silently enlarge it under the same proposal revision.
  if (proposal.origin == ProposalOrigin::Recovery) {
    if (proposal.proposed_step != selection.submitted_step)
      return reject(StageStatus::StaleProposal);
    const auto spendable = std::max<std::int64_t>(
        0, static_cast<std::int64_t>(selection.cash_before) -
               positive(selection.protected_cash));
    const auto required =
        static_cast<std::int64_t>(proposal.order.quantity) * proposal.unit_cost;
    if (required > spendable)
      return reject(StageStatus::UnaffordableRecovery);
  }

  pending_.push_back(Pending{proposal.debt_id, selection.submitted_step,
                             selection.selected_slot, proposal.order.quantity,
                             selection.holding_before, proposal.order});
  state->view.awaiting_receipt = true;
  ++state->view.attempts;
  ++audit_.submissions;
  if (proposal.origin == ProposalOrigin::Recovery)
    ++audit_.recovery_submissions;
  return StageStatus::Selected;
}

std::vector<Settlement> Ledger::observe(const ReceiptObservation &observation) {
  std::vector<Settlement> settlements;
  std::vector<Pending> retained;
  retained.reserve(pending_.size());
  for (const auto &pending : pending_) {
    if (pending.submitted_step >= observation.step) {
      retained.push_back(pending);
      continue;
    }
    auto *state = find(pending.debt_id);
    if (!state)
      continue;

    Settlement settlement;
    settlement.debt_id = pending.debt_id;
    settlement.requested = pending.requested;
    settlement.remaining = state->view.remaining;
    const bool adjacent = observation.step == pending.submitted_step + 1;
    const bool slot_present =
        pending.slot >= 0 &&
        pending.slot < static_cast<int>(observation.slot_fills.size());
    if (!adjacent || !slot_present) {
      settlement.status = FillStatus::Ambiguous;
      settlement.reason = adjacent ? "exact market slot absent from receipt"
                                   : "receipt is not from the next step";
      ++audit_.ambiguous_receipts;
    } else {
      const int exact_filled = std::min(
          pending.requested,
          positive(static_cast<int>(
              observation.slot_fills[static_cast<std::size_t>(pending.slot)])));
      const int item = static_cast<int>(pending.exact_order.item);
      int holding_after = 0;
      if (pending.exact_order.op == fastkag::Op::BUY_SEED && item >= 0 &&
          item < fastkag::N_CROPS) {
        holding_after = observation.seeds_after[static_cast<std::size_t>(item)];
      } else if (pending.exact_order.op == fastkag::Op::BUY_ANIMAL &&
                 item >= static_cast<int>(fastkag::Item::GOOSE) &&
                 item <= static_cast<int>(fastkag::Item::SHEEP)) {
        holding_after = observation.animals_after[static_cast<std::size_t>(
            item - static_cast<int>(fastkag::Item::GOOSE))];
      }
      const int observed_delta = holding_after - pending.holding_before;
      if (exact_filled > 0 && observed_delta < exact_filled) {
        settlement.status = FillStatus::Ambiguous;
        settlement.reason = "fill lacks matching inventory-arrival witness";
        settlement.filled = 0;
        ++audit_.ambiguous_receipts;
      } else if (exact_filled == 0) {
        settlement.status = FillStatus::Zero;
        settlement.reason = "exact slot reported zero fill; debt preserved";
        ++audit_.zero_fills;
      } else {
        const int credited = std::min(state->view.remaining, exact_filled);
        settlement.filled = exact_filled;
        state->view.remaining -= credited;
        state->view.filled += credited;
        state->view.acquisition_complete = state->view.remaining == 0;
        settlement.remaining = state->view.remaining;
        settlement.status =
            state->view.remaining == 0 ? FillStatus::Full : FillStatus::Partial;
        settlement.reason = state->view.remaining == 0
                                ? "inventory acquisition confirmed"
                                : "partial inventory acquisition confirmed; "
                                  "suffix remains debt";
        if (settlement.status == FillStatus::Full)
          ++audit_.full_fills;
        else
          ++audit_.partial_fills;
        audit_.inventory_units_handed_off += credited;
        handoffs_.push_back(UnitRepairHandoff{
            state->view.id, state->view.obligation.operation,
            state->view.obligation.item, credited, observation.step,
            HandoffScope::InventoryOnly, state->view.obligation.provenance});
      }
    }
    state->view.awaiting_receipt = false;
    ++state->revision;
    settlement.remaining = state->view.remaining;
    settlements.push_back(std::move(settlement));
  }
  pending_ = std::move(retained);
  return settlements;
}

std::vector<UnitRepairHandoff> Ledger::drain_handoffs() {
  auto result = std::move(handoffs_);
  handoffs_.clear();
  return result;
}

ReopenStatus Ledger::reopen_acquisition(std::uint64_t debt_id) {
  auto *state = find(debt_id);
  if (!state)
    return ReopenStatus::UnknownDebt;
  if (state->view.awaiting_receipt ||
      std::any_of(
          pending_.begin(), pending_.end(),
          [&](const Pending &pending) { return pending.debt_id == debt_id; }))
    return ReopenStatus::AwaitingReceipt;
  if (std::any_of(handoffs_.begin(), handoffs_.end(),
                  [&](const UnitRepairHandoff &handoff) {
                    return handoff.debt_id == debt_id;
                  }))
    return ReopenStatus::PendingHandoff;
  if (!state->view.acquisition_complete)
    return ReopenStatus::AlreadyOpen;

  state->view.remaining = state->view.obligation.quantity;
  state->view.acquisition_complete = false;
  ++state->revision;
  return ReopenStatus::Reopened;
}

std::optional<DebtView> Ledger::debt(std::uint64_t debt_id) const {
  const auto *state = find(debt_id);
  if (!state)
    return std::nullopt;
  return state->view;
}

std::vector<DebtView> Ledger::debts() const {
  std::vector<DebtView> result;
  result.reserve(debts_.size());
  for (const auto &state : debts_)
    result.push_back(state.view);
  return result;
}

void Ledger::reset() {
  debts_.clear();
  pending_.clear();
  handoffs_.clear();
  audit_ = {};
  next_debt_id_ = 1;
}

Ledger::DebtState *Ledger::find(std::uint64_t debt_id) {
  const auto found =
      std::find_if(debts_.begin(), debts_.end(), [&](const DebtState &state) {
        return state.view.id == debt_id;
      });
  return found == debts_.end() ? nullptr : &*found;
}

const Ledger::DebtState *Ledger::find(std::uint64_t debt_id) const {
  const auto found =
      std::find_if(debts_.begin(), debts_.end(), [&](const DebtState &state) {
        return state.view.id == debt_id;
      });
  return found == debts_.end() ? nullptr : &*found;
}

bool Ledger::supported(const Obligation &obligation) {
  const int item = static_cast<int>(obligation.item);
  const bool seed = obligation.operation == fastkag::Op::BUY_SEED &&
                    item >= 0 && item < fastkag::N_CROPS;
  const bool animal = obligation.operation == fastkag::Op::BUY_ANIMAL &&
                      item >= static_cast<int>(fastkag::Item::GOOSE) &&
                      item <= static_cast<int>(fastkag::Item::SHEEP);
  return (seed || animal) && obligation.quantity > 0 &&
         obligation.unit_cost > 0;
}

bool Ledger::same_action(const fastkag::Action &left,
                         const fastkag::Action &right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

const char *stage_status_name(StageStatus status) noexcept {
  switch (status) {
  case StageStatus::Selected:
    return "selected";
  case StageStatus::UnknownDebt:
    return "unknown-debt";
  case StageStatus::StaleProposal:
    return "stale-proposal";
  case StageStatus::AlreadyPending:
    return "already-pending";
  case StageStatus::SlotAlreadyBound:
    return "slot-already-bound";
  case StageStatus::ConcurrentHoldingWitness:
    return "concurrent-holding-witness";
  case StageStatus::InvalidSlot:
    return "invalid-slot";
  case StageStatus::FinalOrderMismatch:
    return "final-order-mismatch";
  case StageStatus::UnaffordableRecovery:
    return "unaffordable-recovery";
  }
  return "unknown";
}

const char *fill_status_name(FillStatus status) noexcept {
  switch (status) {
  case FillStatus::Full:
    return "full";
  case FillStatus::Partial:
    return "partial";
  case FillStatus::Zero:
    return "zero";
  case FillStatus::Ambiguous:
    return "ambiguous";
  }
  return "unknown";
}

const char *reopen_status_name(ReopenStatus status) noexcept {
  switch (status) {
  case ReopenStatus::Reopened:
    return "reopened";
  case ReopenStatus::AlreadyOpen:
    return "already-open";
  case ReopenStatus::UnknownDebt:
    return "unknown-debt";
  case ReopenStatus::AwaitingReceipt:
    return "awaiting-receipt";
  case ReopenStatus::PendingHandoff:
    return "pending-handoff";
  }
  return "unknown";
}

} // namespace g001::purchase_recovery
