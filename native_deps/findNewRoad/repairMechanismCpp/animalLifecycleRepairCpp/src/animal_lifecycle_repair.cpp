#include "animal_lifecycle_repair.hpp"

#include <algorithm>
#include <utility>

namespace g001::animal_lifecycle {
namespace {

int index(fastkag::Item item) { return static_cast<int>(item); }

bool valid_target(TileKey target) {
  return target.row >= 0 && target.column >= 0;
}

} // namespace

std::optional<AnimalDefinition> definition(fastkag::Item animal) noexcept {
  switch (animal) {
  case fastkag::Item::GOOSE:
    return AnimalDefinition{
        animal, fastkag::TileKind::COOP, fastkag::Item::EGG, 300, 4, 1};
  case fastkag::Item::COW:
    return AnimalDefinition{
        animal, fastkag::TileKind::PASTURE, fastkag::Item::MILK, 400, 8, 2};
  case fastkag::Item::SHEEP:
    return AnimalDefinition{
        animal, fastkag::TileKind::PASTURE, fastkag::Item::WOOL, 500, 6, 3};
  default:
    return std::nullopt;
  }
}

OpenResult Ledger::open(ObjectiveSpec spec) {
  const auto animal = definition(spec.animal);
  if (!animal)
    return {OpenStatus::UnsupportedAnimal, 0};
  if (!valid_target(spec.target))
    return {OpenStatus::InvalidTarget, 0};
  if (spec.purchase_cost != animal->purchase_cost)
    return {OpenStatus::CostMismatch, 0};
  if (std::any_of(objectives_.begin(), objectives_.end(),
                  [&](const ObjectiveState &candidate) {
                    return candidate.view.spec.target == spec.target;
                  })) {
    ++audit_.tile_owner_rejections;
    return {OpenStatus::TileAlreadyOwned, 0};
  }
  const auto debt = purchases_.open({fastkag::Op::BUY_ANIMAL, spec.animal, 1,
                                     spec.purchase_cost, spec.provenance});
  ObjectiveState state;
  state.view.id = next_objective_id_++;
  state.view.spec = std::move(spec);
  state.view.purchase_debt_id = debt;
  objectives_.push_back(std::move(state));
  ++audit_.objectives_opened;
  return {OpenStatus::Opened, objectives_.back().view.id};
}

std::optional<purchase_recovery::Proposal>
Ledger::describe_hard_acquisition(std::uint64_t objective_id,
                                  int requested_quantity) const {
  const auto *state = find(objective_id);
  if (!state || state->view.stage != Stage::Acquire)
    return std::nullopt;
  return purchases_.describe_hard_order(state->view.purchase_debt_id,
                                        requested_quantity);
}

std::optional<purchase_recovery::Proposal> Ledger::propose_acquisition(
    std::uint64_t objective_id,
    const purchase_recovery::FundingObservation &funding) const {
  const auto *state = find(objective_id);
  if (!state || state->view.stage != Stage::Acquire)
    return std::nullopt;
  return purchases_.propose_recovery(state->view.purchase_debt_id, funding);
}

purchase_recovery::StageStatus
Ledger::stage_acquisition(const purchase_recovery::Proposal &proposal,
                          const purchase_recovery::FinalSelection &selection) {
  const auto *state = find_purchase(proposal.debt_id);
  if (!state || state->view.stage != Stage::Acquire)
    return purchase_recovery::StageStatus::UnknownDebt;
  return purchases_.stage_final(proposal, selection);
}

std::vector<purchase_recovery::Settlement> Ledger::observe_acquisitions(
    const purchase_recovery::ReceiptObservation &receipt) {
  auto settlements = purchases_.observe(receipt);
  for (const auto &handoff : purchases_.drain_handoffs()) {
    auto *state = find_purchase(handoff.debt_id);
    if (!state || state->view.stage != Stage::Acquire ||
        handoff.acquisition != fastkag::Op::BUY_ANIMAL ||
        handoff.item != state->view.spec.animal || handoff.quantity != 1)
      continue;
    const auto purchase = purchases_.debt(handoff.debt_id);
    if (!purchase || !purchase->acquisition_complete)
      continue;
    state->view.stage = Stage::PickupFromShed;
    ++state->view.revision;
    ++audit_.purchase_handoffs;
  }
  return settlements;
}

std::optional<ActorLease> Ledger::lease_actor(std::uint64_t objective_id,
                                              ActorIdentity actor, int step) {
  auto *state = find(objective_id);
  if (!state || state->view.stage == Stage::Acquire ||
      state->view.stage == Stage::Complete || state->lease ||
      state->view.pending_unit_receipt || step < 0 || actor.actor_id < 0)
    return std::nullopt;
  state->lease = ActorLease{objective_id, next_lease_token_++, actor, step};
  ++audit_.leases_issued;
  return state->lease;
}

bool Ledger::release_actor(const ActorLease &lease) {
  auto *state = find(lease.objective_id);
  if (!state || state->view.pending_unit_receipt || !state->lease ||
      state->lease->token != lease.token || state->lease->actor != lease.actor)
    return false;
  state->lease.reset();
  return true;
}

ProposalResult Ledger::propose_unit(std::uint64_t objective_id,
                                    const ActorLease &lease,
                                    const Observation &before) {
  auto blocked = [&](BlockReason reason, bool fail_closed = false) {
    if (auto *state = find(objective_id)) {
      state->view.last_block_reason = block_name(reason);
      if (fail_closed && !state->view.fail_closed) {
        state->view.fail_closed = true;
        ++audit_.fail_closed_observations;
      }
    }
    return ProposalResult{reason, std::nullopt};
  };
  auto *state = find(objective_id);
  if (!state)
    return blocked(BlockReason::UnknownObjective);
  if (state->view.stage == Stage::Acquire)
    return blocked(BlockReason::AcquisitionPending);
  if (state->view.stage == Stage::Complete)
    return blocked(BlockReason::Complete);
  if (!state->lease || state->lease->token != lease.token ||
      state->lease->actor != lease.actor ||
      before.actor.identity != lease.actor || before.step < lease.leased_step) {
    ++audit_.stale_lease_rejections;
    return blocked(BlockReason::MissingOrStaleLease);
  }
  const auto animal = definition(state->view.spec.animal);
  if (!animal)
    return blocked(BlockReason::ObservationMismatch, true);

  UnitProposal proposal{
      state->view.id, state->view.revision, state->view.stage, lease, {},
      before};
  const auto animal_item = index(state->view.spec.animal);
  const bool at_target = before.actor.position == state->view.spec.target;
  switch (state->view.stage) {
  case Stage::PickupFromShed:
    if (!before.actor.shed_adjacent)
      return blocked(BlockReason::NotAtShed);
    if (before.shed[static_cast<std::size_t>(animal_item)] < 1)
      return blocked(BlockReason::AnimalAbsentFromShed);
    proposal.action = {fastkag::Op::PICKUP, state->view.spec.animal, 1};
    break;
  case Stage::PlaceOnTarget:
    if (!at_target)
      return blocked(BlockReason::NotAtTarget);
    if (before.actor.inventory[static_cast<std::size_t>(animal_item)] < 1)
      return blocked(BlockReason::AnimalAbsentFromActor);
    if (before.target.kind != animal->required_structure ||
        before.target.animal != fastkag::Item::NONE)
      return blocked(BlockReason::WrongStructure);
    proposal.action = {fastkag::Op::PLACE, state->view.spec.animal, 1};
    break;
  case Stage::FeedCare: {
    if (!at_target)
      return blocked(BlockReason::NotAtTarget);
    if (before.target.kind != fastkag::TileKind::ANIMAL ||
        before.target.animal != state->view.spec.animal)
      return blocked(BlockReason::TargetAnimalLost, true);
    const int first_day = state->view.placed_day + animal->first_yield_days;
    if (before.target.yield_units > 0) {
      if (state->view.placed_day < 0 || before.day < first_day)
        return blocked(BlockReason::YieldNotMature, true);
      proposal.proposed_stage = Stage::FirstYield;
      proposal.action = {fastkag::Op::HARVEST, animal->first_product, 1};
    } else if (!before.target.fed_today) {
      if (before.actor
              .inventory[static_cast<std::size_t>(fastkag::Item::WHEAT)] < 1)
        return blocked(BlockReason::WheatAbsentFromActor);
      proposal.action = {fastkag::Op::FEED, fastkag::Item::WHEAT, 1};
    } else if (!before.target.cared_today) {
      proposal.action = {fastkag::Op::CARE, fastkag::Item::NONE, 1};
    } else {
      return blocked(BlockReason::AwaitingDayBoundary);
    }
    break;
  }
  case Stage::FirstYield:
    return blocked(BlockReason::ObservationMismatch, true);
  case Stage::Acquire:
  case Stage::Complete:
    break;
  }
  state->view.last_block_reason.clear();
  return {BlockReason::None, std::move(proposal)};
}

UnitStageStatus Ledger::stage_unit(const UnitProposal &proposal,
                                   const UnitFinalSelection &selection) {
  auto *state = find(proposal.objective_id);
  if (!state)
    return UnitStageStatus::UnknownObjective;
  if (proposal.revision != state->view.revision)
    return UnitStageStatus::StaleProposal;
  if (!state->lease || state->lease->token != proposal.lease.token ||
      state->lease->actor != proposal.lease.actor) {
    ++audit_.stale_lease_rejections;
    return UnitStageStatus::MissingOrStaleLease;
  }
  // UnitProposal is a public transport type, not authority. Re-derive it from
  // persistent stage + exact observation so a caller cannot forge HARVEST,
  // skip PICKUP/PLACE, or change the animal item under a valid lease token.
  const auto certified = propose_unit(proposal.objective_id, proposal.lease,
                                      proposal.exact_before);
  if (!certified.proposal ||
      certified.proposal->revision != proposal.revision ||
      certified.proposal->proposed_stage != proposal.proposed_stage ||
      !same_action(certified.proposal->action, proposal.action))
    return UnitStageStatus::StaleProposal;
  if (state->view.pending_unit_receipt ||
      std::any_of(pending_.begin(), pending_.end(),
                  [&](const PendingUnit &pending) {
                    return pending.objective_id == state->view.id;
                  }))
    return UnitStageStatus::AlreadyPending;
  if (selection.submitted_step != proposal.exact_before.step ||
      selection.selected_actor_slot < 0 ||
      selection.selected_actor_slot >=
          static_cast<int>(selection.final_units.size()) ||
      selection.selected_actor_slot != proposal.lease.actor.actor_id)
    return UnitStageStatus::InvalidSlot;
  if (selection.exact_before != proposal.exact_before)
    return UnitStageStatus::BeforeWitnessMismatch;
  if (!same_action(selection.final_units[static_cast<std::size_t>(
                       selection.selected_actor_slot)],
                   proposal.action))
    return UnitStageStatus::FinalActionMismatch;
  if (std::any_of(pending_.begin(), pending_.end(),
                  [&](const PendingUnit &pending) {
                    return pending.submitted_step == selection.submitted_step &&
                           pending.slot == selection.selected_actor_slot;
                  }))
    return UnitStageStatus::AlreadyPending;
  pending_.push_back(PendingUnit{
      state->view.id, state->view.stage, proposal.proposed_stage,
      proposal.lease, selection.submitted_step, selection.selected_actor_slot,
      proposal.action, proposal.exact_before});
  state->view.stage = proposal.proposed_stage;
  state->view.pending_unit_receipt = true;
  ++audit_.unit_submissions;
  return UnitStageStatus::Selected;
}

UnitSettlement Ledger::observe_unit(const UnitReceipt &receipt) {
  const auto found = std::find_if(
      pending_.begin(), pending_.end(), [&](const PendingUnit &pending) {
        return pending.objective_id == receipt.objective_id;
      });
  if (found == pending_.end())
    return {0, Stage::Acquire, UnitReceiptStatus::UnknownObjective,
            Stage::Acquire, "no pending unit receipt"};
  const auto pending = *found;
  auto *state = find(pending.objective_id);
  if (!state) {
    pending_.erase(found);
    return {pending.objective_id, pending.attempted_stage,
            UnitReceiptStatus::UnknownObjective, Stage::Acquire,
            "pending objective disappeared"};
  }
  UnitSettlement result{state->view.id,
                        pending.attempted_stage,
                        UnitReceiptStatus::Failed,
                        state->view.stage,
                        {}};
  const bool exact_receipt =
      receipt.step == pending.submitted_step + 1 &&
      receipt.exact_after.step == receipt.step &&
      receipt.lease_token == pending.lease.token &&
      receipt.exact_after.actor.identity == pending.lease.actor;
  bool success = false;
  if (!exact_receipt) {
    result.status = UnitReceiptStatus::Ambiguous;
    result.reason = "receipt step or actor lease token is not exact";
    ++audit_.ambiguous_unit_receipts;
  } else {
    const auto animal = definition(state->view.spec.animal);
    const auto animal_item =
        static_cast<std::size_t>(index(state->view.spec.animal));
    const auto product_item =
        static_cast<std::size_t>(index(animal->first_product));
    const auto &before = pending.before;
    const auto &after = receipt.exact_after;
    const bool actor_stayed = after.actor.position == before.actor.position;
    switch (pending.exact_action.op) {
    case fastkag::Op::PICKUP:
      success = actor_stayed &&
                after.shed[animal_item] <= before.shed[animal_item] - 1 &&
                after.actor.inventory[animal_item] >=
                    before.actor.inventory[animal_item] + 1;
      if (success)
        state->view.stage = Stage::PlaceOnTarget;
      break;
    case fastkag::Op::PLACE:
      success = actor_stayed &&
                after.actor.inventory[animal_item] <=
                    before.actor.inventory[animal_item] - 1 &&
                after.target.kind == fastkag::TileKind::ANIMAL &&
                after.target.animal == state->view.spec.animal &&
                after.target.placed_day == before.day;
      if (success) {
        state->view.stage = Stage::FeedCare;
        state->view.placed_day = after.target.placed_day;
      }
      break;
    case fastkag::Op::FEED:
      success =
          actor_stayed && after.target.kind == fastkag::TileKind::ANIMAL &&
          after.target.animal == state->view.spec.animal &&
          after.target.fed_today &&
          after.actor
                  .inventory[static_cast<std::size_t>(fastkag::Item::WHEAT)] <=
              before.actor.inventory[static_cast<std::size_t>(
                  fastkag::Item::WHEAT)] -
                  1;
      break;
    case fastkag::Op::CARE:
      success = actor_stayed &&
                after.target.kind == fastkag::TileKind::ANIMAL &&
                after.target.animal == state->view.spec.animal &&
                after.target.cared_today;
      break;
    case fastkag::Op::HARVEST:
      success = actor_stayed && pending.attempted_stage == Stage::FirstYield &&
                after.target.kind == fastkag::TileKind::ANIMAL &&
                after.target.animal == state->view.spec.animal &&
                after.target.yield_units < before.target.yield_units &&
                after.actor.inventory[product_item] >=
                    before.actor.inventory[product_item] + 1;
      if (success) {
        state->view.stage = Stage::Complete;
        state->view.complete = true;
        ++audit_.lifecycles_completed;
      }
      break;
    default:
      success = false;
      break;
    }
    if (success) {
      result.status = UnitReceiptStatus::Success;
      result.reason = "exact before/after lifecycle effect confirmed";
      ++state->view.revision;
      ++audit_.unit_receipt_successes;
    } else {
      result.reason = "exact receipt lacks required lifecycle effect";
      ++audit_.unit_receipt_failures;
    }
  }
  if (!success)
    state->view.stage = pending.prior_stage;
  state->view.pending_unit_receipt = false;
  state->lease.reset();
  pending_.erase(found);
  result.retained_or_next_stage = state->view.stage;
  return result;
}

ExternalReconcileStatus
Ledger::reconcile_external(std::uint64_t objective_id,
                           const Observation &observation) {
  auto *state = find(objective_id);
  if (!state)
    return ExternalReconcileStatus::UnknownObjective;
  if (state->view.pending_unit_receipt)
    return ExternalReconcileStatus::PendingReceipt;
  const auto animal_index =
      static_cast<std::size_t>(index(state->view.spec.animal));
  const bool actor_has = observation.actor.inventory[animal_index] > 0;
  const bool shed_has = observation.shed[animal_index] > 0;
  const bool target_has =
      observation.target.kind == fastkag::TileKind::ANIMAL &&
      observation.target.animal == state->view.spec.animal;
  ExternalReconcileStatus status = ExternalReconcileStatus::NoChange;
  if (state->view.stage == Stage::PickupFromShed && actor_has) {
    state->view.stage = Stage::PlaceOnTarget;
    status = ExternalReconcileStatus::AdvancedToPlace;
  } else if (state->view.stage == Stage::PlaceOnTarget && target_has) {
    state->view.stage = Stage::FeedCare;
    state->view.placed_day = observation.target.placed_day;
    status = ExternalReconcileStatus::AdvancedToFeedCare;
  } else if (state->view.stage == Stage::PlaceOnTarget && !actor_has &&
             !target_has && shed_has) {
    state->view.stage = Stage::PickupFromShed;
    status = ExternalReconcileStatus::ReturnedToPickup;
  }
  if (status != ExternalReconcileStatus::NoChange) {
    ++state->view.revision;
    state->lease.reset();
    state->view.last_block_reason.clear();
    ++audit_.external_inventory_reconciles;
  }
  return status;
}

ReopenAcquisitionStatus
Ledger::reopen_acquisition(std::uint64_t objective_id,
                           const AcquisitionLossProof &proof) {
  auto *state = find(objective_id);
  if (!state)
    return ReopenAcquisitionStatus::UnknownObjective;
  if (proof.step < 0 || proof.target != state->view.spec.target ||
      proof.animal != state->view.spec.animal || proof.shed_count < 0 ||
      proof.all_actor_inventory_count < 0 || proof.all_owned_board_count < 0)
    return ReopenAcquisitionStatus::InvalidProof;
  const int total_owned = proof.shed_count + proof.all_actor_inventory_count +
                          proof.all_owned_board_count;
  if (proof.units_reserved_for_other_objectives < 0 ||
      proof.units_reserved_for_other_objectives > total_owned)
    return ReopenAcquisitionStatus::InvalidProof;
  if (state->view.pending_unit_receipt)
    return ReopenAcquisitionStatus::PendingUnitReceipt;
  if (state->lease)
    return ReopenAcquisitionStatus::ActiveLease;
  if (state->view.stage == Stage::Acquire)
    return ReopenAcquisitionStatus::AlreadyOpen;
  if (state->view.stage != Stage::PickupFromShed &&
      state->view.stage != Stage::PlaceOnTarget)
    return ReopenAcquisitionStatus::LifecycleAlreadyPlaced;
  if (proof.intended_target_has_animal ||
      total_owned > proof.units_reserved_for_other_objectives)
    return ReopenAcquisitionStatus::InventoryStillAvailable;

  const auto reopened =
      purchases_.reopen_acquisition(state->view.purchase_debt_id);
  if (reopened == purchase_recovery::ReopenStatus::AlreadyOpen) {
    state->view.stage = Stage::Acquire;
    return ReopenAcquisitionStatus::AlreadyOpen;
  }
  if (reopened != purchase_recovery::ReopenStatus::Reopened)
    return ReopenAcquisitionStatus::PurchaseLedgerRejected;
  state->view.stage = Stage::Acquire;
  state->view.last_block_reason =
      "acquired animal inventory was lost before PLACE";
  ++state->view.revision;
  return ReopenAcquisitionStatus::Reopened;
}

std::optional<ObjectiveView>
Ledger::objective(std::uint64_t objective_id) const {
  const auto *state = find(objective_id);
  return state ? std::optional<ObjectiveView>(state->view) : std::nullopt;
}

std::vector<ObjectiveView> Ledger::objectives() const {
  std::vector<ObjectiveView> result;
  result.reserve(objectives_.size());
  for (const auto &state : objectives_)
    result.push_back(state.view);
  return result;
}

Ledger::ObjectiveState *Ledger::find(std::uint64_t objective_id) {
  const auto found = std::find_if(objectives_.begin(), objectives_.end(),
                                  [&](const ObjectiveState &state) {
                                    return state.view.id == objective_id;
                                  });
  return found == objectives_.end() ? nullptr : &*found;
}

const Ledger::ObjectiveState *Ledger::find(std::uint64_t objective_id) const {
  const auto found = std::find_if(objectives_.begin(), objectives_.end(),
                                  [&](const ObjectiveState &state) {
                                    return state.view.id == objective_id;
                                  });
  return found == objectives_.end() ? nullptr : &*found;
}

Ledger::ObjectiveState *Ledger::find_purchase(std::uint64_t debt_id) {
  const auto found = std::find_if(
      objectives_.begin(), objectives_.end(), [&](const ObjectiveState &state) {
        return state.view.purchase_debt_id == debt_id;
      });
  return found == objectives_.end() ? nullptr : &*found;
}

bool Ledger::same_action(const fastkag::Action &left,
                         const fastkag::Action &right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

const char *Ledger::block_name(BlockReason reason) noexcept {
  switch (reason) {
  case BlockReason::None:
    return "none";
  case BlockReason::UnknownObjective:
    return "unknown-objective";
  case BlockReason::AcquisitionPending:
    return "acquisition-pending";
  case BlockReason::Complete:
    return "complete";
  case BlockReason::MissingOrStaleLease:
    return "missing-or-stale-lease";
  case BlockReason::ObservationMismatch:
    return "observation-mismatch";
  case BlockReason::NotAtShed:
    return "not-at-shed";
  case BlockReason::AnimalAbsentFromShed:
    return "animal-absent-from-shed";
  case BlockReason::NotAtTarget:
    return "not-at-target";
  case BlockReason::AnimalAbsentFromActor:
    return "animal-absent-from-actor";
  case BlockReason::WrongStructure:
    return "wrong-structure";
  case BlockReason::TargetAnimalLost:
    return "target-animal-lost";
  case BlockReason::WheatAbsentFromActor:
    return "wheat-absent-from-actor";
  case BlockReason::AwaitingDayBoundary:
    return "awaiting-day-boundary";
  case BlockReason::YieldNotMature:
    return "yield-not-mature";
  }
  return "unknown";
}

const char *stage_name(Stage stage) noexcept {
  switch (stage) {
  case Stage::Acquire:
    return "acquire";
  case Stage::PickupFromShed:
    return "pickup-from-shed";
  case Stage::PlaceOnTarget:
    return "place-on-target";
  case Stage::FeedCare:
    return "feed-care";
  case Stage::FirstYield:
    return "first-yield";
  case Stage::Complete:
    return "complete";
  }
  return "unknown";
}

const char *open_status_name(OpenStatus status) noexcept {
  switch (status) {
  case OpenStatus::Opened:
    return "opened";
  case OpenStatus::UnsupportedAnimal:
    return "unsupported-animal";
  case OpenStatus::InvalidTarget:
    return "invalid-target";
  case OpenStatus::CostMismatch:
    return "cost-mismatch";
  case OpenStatus::TileAlreadyOwned:
    return "tile-already-owned";
  }
  return "unknown";
}

const char *unit_stage_status_name(UnitStageStatus status) noexcept {
  switch (status) {
  case UnitStageStatus::Selected:
    return "selected";
  case UnitStageStatus::UnknownObjective:
    return "unknown-objective";
  case UnitStageStatus::StaleProposal:
    return "stale-proposal";
  case UnitStageStatus::MissingOrStaleLease:
    return "missing-or-stale-lease";
  case UnitStageStatus::AlreadyPending:
    return "already-pending";
  case UnitStageStatus::InvalidSlot:
    return "invalid-slot";
  case UnitStageStatus::FinalActionMismatch:
    return "final-action-mismatch";
  case UnitStageStatus::BeforeWitnessMismatch:
    return "before-witness-mismatch";
  }
  return "unknown";
}

const char *unit_receipt_status_name(UnitReceiptStatus status) noexcept {
  switch (status) {
  case UnitReceiptStatus::Success:
    return "success";
  case UnitReceiptStatus::Failed:
    return "failed";
  case UnitReceiptStatus::Ambiguous:
    return "ambiguous";
  case UnitReceiptStatus::UnknownObjective:
    return "unknown-objective";
  }
  return "unknown";
}

} // namespace g001::animal_lifecycle
