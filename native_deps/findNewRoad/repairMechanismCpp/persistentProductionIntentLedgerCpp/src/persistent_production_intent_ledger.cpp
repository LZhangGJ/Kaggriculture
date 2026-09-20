#include "persistent_production_intent_ledger.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace g001::persistent_production {
namespace {

constexpr std::array<int, fastkag::N_CROPS> crop_cost{10, 20, 50, 100, 80};

int item_index(fastkag::Item item) { return static_cast<int>(item); }

bool valid_tile(TileKey tile) { return tile.row >= 0 && tile.column >= 0; }

bool crop_item(fastkag::Item item) {
  const int index = item_index(item);
  return index >= 0 && index < fastkag::N_CROPS;
}

bool animal_item(fastkag::Item item) {
  return animal_lifecycle::definition(item).has_value();
}

fastkag::Position native_position(TileKey tile) {
  return {static_cast<std::int16_t>(tile.column),
          static_cast<std::int16_t>(tile.row)};
}

void mix(std::uint64_t &hash, std::uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
}

void mix_action(std::uint64_t &hash, const fastkag::Action &action) noexcept {
  mix(hash, static_cast<std::uint64_t>(static_cast<int>(action.op) + 1));
  mix(hash, static_cast<std::uint64_t>(static_cast<int>(action.item) + 2));
  mix(hash,
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(action.quantity)));
}

} // namespace

std::uint64_t observation_fingerprint(const Observation &observation) noexcept {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  const auto add_int = [&](int value) {
    mix(hash, static_cast<std::uint64_t>(static_cast<std::uint32_t>(value)));
  };
  add_int(observation.step);
  add_int(observation.day);
  add_int(observation.target_key.row);
  add_int(observation.target_key.column);
  const auto &tile = observation.target;
  add_int(static_cast<int>(tile.kind));
  add_int(static_cast<int>(tile.crop));
  add_int(static_cast<int>(tile.animal));
  add_int(tile.planted_day);
  add_int(tile.placed_day);
  add_int(tile.yield_units);
  add_int(tile.consecutive_unwatered);
  add_int(tile.consecutive_unfed);
  add_int(tile.fertilized_until_day);
  add_int(tile.pending_care_bonus);
  add_int(tile.max_lifespan_step);
  add_int(tile.watered_today);
  add_int(tile.fed_today);
  add_int(tile.cared_today);
  add_int(tile.fertilizer_available);
  for (const int value : observation.shed)
    add_int(value);
  for (const int value : observation.seeds)
    add_int(value);
  add_int(observation.actor.identity.actor_id);
  mix(hash, observation.actor.identity.generation);
  add_int(observation.actor.position.row);
  add_int(observation.actor.position.column);
  add_int(observation.actor.shed_adjacent);
  for (const int value : observation.actor.inventory)
    add_int(value);
  return hash == 0 ? 1 : hash;
}

PrefixAuthority
make_prefix_authority(int submitted_step, int selected_actor_slot,
                      std::uint64_t final_manifest_generation,
                      std::uint64_t phase_start_fingerprint,
                      std::span<const fastkag::Action> final_units,
                      const Observation &exact_before) {
  if (submitted_step < 0 || selected_actor_slot < 0 ||
      selected_actor_slot >= static_cast<int>(final_units.size()) ||
      final_manifest_generation == 0 || phase_start_fingerprint == 0)
    throw std::invalid_argument("invalid unit prefix authority input");
  std::uint64_t lower = phase_start_fingerprint;
  mix(lower, final_manifest_generation);
  mix(lower, static_cast<std::uint64_t>(submitted_step));
  mix(lower, static_cast<std::uint64_t>(selected_actor_slot));
  for (int slot = 0; slot < selected_actor_slot; ++slot)
    mix_action(lower, final_units[static_cast<std::size_t>(slot)]);
  if (lower == 0)
    lower = 1;
  return {submitted_step,
          selected_actor_slot,
          final_manifest_generation,
          phase_start_fingerprint,
          lower,
          observation_fingerprint(exact_before)};
}

std::uint64_t
prefix_authority_fingerprint(const PrefixAuthority &authority) noexcept {
  std::uint64_t hash = 0x84222325cbf29ce4ULL;
  mix(hash, static_cast<std::uint64_t>(authority.submitted_step));
  mix(hash, static_cast<std::uint64_t>(authority.selected_actor_slot));
  mix(hash, authority.final_manifest_generation);
  mix(hash, authority.phase_start_fingerprint);
  mix(hash, authority.lower_prefix_fingerprint);
  mix(hash, authority.exact_before_fingerprint);
  return hash == 0 ? 1 : hash;
}

bool operator==(const Observation &left, const Observation &right) {
  const auto &a = left.target;
  const auto &b = right.target;
  const bool same_tile =
      a.kind == b.kind && a.crop == b.crop && a.animal == b.animal &&
      a.planted_day == b.planted_day && a.placed_day == b.placed_day &&
      a.yield_units == b.yield_units &&
      a.consecutive_unwatered == b.consecutive_unwatered &&
      a.consecutive_unfed == b.consecutive_unfed &&
      a.fertilized_until_day == b.fertilized_until_day &&
      a.pending_care_bonus == b.pending_care_bonus &&
      a.max_lifespan_step == b.max_lifespan_step &&
      a.watered_today == b.watered_today && a.fed_today == b.fed_today &&
      a.cared_today == b.cared_today &&
      a.fertilizer_available == b.fertilizer_available;
  return left.step == right.step && left.day == right.day &&
         left.target_key == right.target_key && same_tile &&
         left.shed == right.shed && left.seeds == right.seeds &&
         left.actor == right.actor;
}

OpenResult Ledger::open(ObjectiveSpec spec) {
  if (spec.key.origin_day < 0 || !valid_tile(spec.key.tile) ||
      spec.deadline_day < spec.key.origin_day)
    return {OpenStatus::InvalidDayOrTile, 0};
  const bool crop = spec.kind == Kind::Crop && crop_item(spec.item);
  const bool animal = spec.kind == Kind::Animal && animal_item(spec.item);
  if (!crop && !animal)
    return {OpenStatus::UnsupportedItem, 0};
  const int canonical_cost =
      crop ? crop_cost[static_cast<std::size_t>(item_index(spec.item))]
           : animal_lifecycle::definition(spec.item)->purchase_cost;
  if (spec.purchase_quantity <= 0 || spec.unit_cost != canonical_cost ||
      (animal && spec.purchase_quantity != 1))
    return {OpenStatus::InvalidQuantityOrCost, 0};
  if (std::any_of(objectives_.begin(), objectives_.end(),
                  [&](const auto &entry) {
                    return entry.first.tile == spec.key.tile &&
                           entry.second.view.status != Status::Complete &&
                           entry.second.view.status != Status::Expired;
                  })) {
    ++audit_.active_tile_rejections;
    return {OpenStatus::ActiveTileOwner, 0};
  }

  State state;
  state.view.id = next_objective_id_++;
  state.view.spec = spec;
  if (crop) {
    state.view.purchase_debt_id = seed_purchases_.open(
        {fastkag::Op::BUY_SEED, spec.item, spec.purchase_quantity,
         spec.unit_cost, spec.provenance});
  } else {
    const auto opened = animals_.open(
        {spec.item, spec.key.tile, spec.unit_cost, spec.provenance});
    if (opened.status != animal_lifecycle::OpenStatus::Opened)
      return {OpenStatus::DelegateRejected, 0};
    state.view.animal_delegate_id = opened.objective_id;
    state.view.purchase_debt_id =
        animals_.objective(opened.objective_id)->purchase_debt_id;
  }
  const auto id = state.view.id;
  objectives_.emplace(spec.key, std::move(state));
  ++audit_.objectives_opened;
  return {OpenStatus::Opened, id};
}

std::optional<purchase_recovery::Proposal>
Ledger::describe_hard_purchase(std::uint64_t objective_id,
                               int requested_quantity) const {
  const auto *state = find(objective_id);
  if (!state || state->view.status == Status::Complete ||
      state->view.status == Status::Expired)
    return std::nullopt;
  if (state->view.spec.kind == Kind::Crop)
    return seed_purchases_.describe_hard_order(state->view.purchase_debt_id,
                                               requested_quantity);
  return animals_.describe_hard_acquisition(state->view.animal_delegate_id,
                                            requested_quantity);
}

std::optional<purchase_recovery::Proposal> Ledger::propose_purchase(
    std::uint64_t objective_id,
    const purchase_recovery::FundingObservation &funding) const {
  const auto *state = find(objective_id);
  if (!state || state->view.status == Status::Complete ||
      state->view.status == Status::Expired)
    return std::nullopt;
  if (state->view.spec.kind == Kind::Crop)
    return seed_purchases_.propose_recovery(state->view.purchase_debt_id,
                                            funding);
  return animals_.propose_acquisition(state->view.animal_delegate_id, funding);
}

purchase_recovery::StageStatus
Ledger::stage_purchase(std::uint64_t objective_id,
                       const purchase_recovery::Proposal &proposal,
                       const purchase_recovery::FinalSelection &selection) {
  auto *state = find(objective_id);
  if (!state)
    return purchase_recovery::StageStatus::UnknownDebt;
  if (state->view.status == Status::Complete ||
      state->view.status == Status::Expired)
    return purchase_recovery::StageStatus::StaleProposal;
  if (std::any_of(pending_slots_.begin(), pending_slots_.end(),
                  [&](const PendingSlot &pending) {
                    return pending.market &&
                           pending.submitted_step == selection.submitted_step &&
                           pending.slot == selection.selected_slot;
                  }))
    return purchase_recovery::StageStatus::SlotAlreadyBound;
  const auto status = state->view.spec.kind == Kind::Crop
                          ? seed_purchases_.stage_final(proposal, selection)
                          : animals_.stage_acquisition(proposal, selection);
  if (status == purchase_recovery::StageStatus::Selected)
    pending_slots_.push_back({objective_id, selection.submitted_step,
                              selection.selected_slot, true});
  return status;
}

std::vector<PurchaseSettlement> Ledger::observe_purchases(
    const purchase_recovery::ReceiptObservation &receipt) {
  std::vector<PurchaseSettlement> result;
  for (auto settlement : seed_purchases_.observe(receipt)) {
    if (auto *state = find_seed_debt(settlement.debt_id))
      result.push_back({state->view.id, std::move(settlement)});
  }
  for (const auto &handoff : seed_purchases_.drain_handoffs()) {
    auto *state = find_seed_debt(handoff.debt_id);
    if (!state)
      continue;
    ++audit_.purchase_handoffs;
    const auto debt = seed_purchases_.debt(handoff.debt_id);
    state->view.purchase_complete = debt && debt->acquisition_complete;
  }

  for (auto settlement : animals_.observe_acquisitions(receipt)) {
    if (auto *state = find_animal_debt(settlement.debt_id)) {
      result.push_back({state->view.id, std::move(settlement)});
      sync_animal(*state);
    }
  }
  for (auto &[key, state] : objectives_) {
    static_cast<void>(key);
    if (state.view.spec.kind != Kind::Animal)
      continue;
    const bool before = state.view.purchase_complete;
    sync_animal(state);
    if (!before && state.view.purchase_complete)
      ++audit_.purchase_handoffs;
  }
  std::erase_if(pending_slots_, [&](const PendingSlot &pending) {
    return pending.market && pending.submitted_step < receipt.step;
  });
  return result;
}

ReopenAnimalAcquisitionStatus
Ledger::reopen_animal_acquisition(std::uint64_t objective_id,
                                  const AnimalAcquisitionLossProof &proof) {
  auto *state = find(objective_id);
  if (!state)
    return ReopenAnimalAcquisitionStatus::UnknownObjective;
  if (state->view.status != Status::Open ||
      state->view.spec.kind != Kind::Animal)
    return ReopenAnimalAcquisitionStatus::NotOpenAnimal;
  if (proof.step < 0 || proof.day < state->view.spec.key.origin_day ||
      proof.target != state->view.spec.key.tile ||
      proof.animal != state->view.spec.item || proof.shed_count < 0 ||
      proof.all_actor_inventory_count < 0 || proof.all_owned_board_count < 0)
    return ReopenAnimalAcquisitionStatus::InvalidProof;
  const int total_owned = proof.shed_count + proof.all_actor_inventory_count +
                          proof.all_owned_board_count;
  if (proof.units_reserved_for_other_objectives < 0 ||
      proof.units_reserved_for_other_objectives > total_owned)
    return ReopenAnimalAcquisitionStatus::InvalidProof;
  if (state->view.pending_unit_receipt)
    return ReopenAnimalAcquisitionStatus::PendingUnitReceipt;
  if (state->view.lease || state->animal_lease)
    return ReopenAnimalAcquisitionStatus::ActiveLease;
  if (std::any_of(pending_slots_.begin(), pending_slots_.end(),
                  [&](const PendingSlot &pending) {
                    return pending.objective_id == objective_id &&
                           pending.market;
                  }))
    return ReopenAnimalAcquisitionStatus::PendingPurchaseReceipt;
  if (proof.intended_target_has_animal ||
      total_owned > proof.units_reserved_for_other_objectives)
    return ReopenAnimalAcquisitionStatus::InventoryStillAvailable;

  const auto status = animals_.reopen_acquisition(
      state->view.animal_delegate_id,
      {proof.step, proof.target, proof.animal, proof.shed_count,
       proof.all_actor_inventory_count, proof.all_owned_board_count,
       proof.units_reserved_for_other_objectives,
       proof.intended_target_has_animal});
  switch (status) {
  case animal_lifecycle::ReopenAcquisitionStatus::Reopened:
    sync_animal(*state);
    ++state->view.revision;
    state->view.last_reason =
        "animal acquisition reopened after exact whole-player loss proof";
    return ReopenAnimalAcquisitionStatus::Reopened;
  case animal_lifecycle::ReopenAcquisitionStatus::AlreadyOpen:
    sync_animal(*state);
    return ReopenAnimalAcquisitionStatus::AlreadyOpen;
  case animal_lifecycle::ReopenAcquisitionStatus::PendingUnitReceipt:
    return ReopenAnimalAcquisitionStatus::PendingUnitReceipt;
  case animal_lifecycle::ReopenAcquisitionStatus::ActiveLease:
    return ReopenAnimalAcquisitionStatus::ActiveLease;
  case animal_lifecycle::ReopenAcquisitionStatus::InventoryStillAvailable:
    return ReopenAnimalAcquisitionStatus::InventoryStillAvailable;
  case animal_lifecycle::ReopenAcquisitionStatus::LifecycleAlreadyPlaced:
    return ReopenAnimalAcquisitionStatus::LifecycleAlreadyPlaced;
  case animal_lifecycle::ReopenAcquisitionStatus::InvalidProof:
    return ReopenAnimalAcquisitionStatus::InvalidProof;
  case animal_lifecycle::ReopenAcquisitionStatus::UnknownObjective:
  case animal_lifecycle::ReopenAcquisitionStatus::PurchaseLedgerRejected:
    return ReopenAnimalAcquisitionStatus::DelegateRejected;
  }
  return ReopenAnimalAcquisitionStatus::DelegateRejected;
}

Status Ledger::reconcile(std::uint64_t objective_id,
                         const Observation &observation) {
  auto *state = find(objective_id);
  if (!state)
    return Status::FailClosed;
  auto &view = state->view;
  if (view.status == Status::Complete || view.status == Status::Expired)
    return view.status;
  if (observation.step < 0 || observation.day < view.spec.key.origin_day ||
      observation.target_key != view.spec.key.tile) {
    view.status = Status::FailClosed;
    view.last_reason = "observation does not match objective day/tile";
    return view.status;
  }
  view.status = Status::Open;
  view.last_reason.clear();
  if (view.spec.kind == Kind::Crop &&
      observation.target.kind == fastkag::TileKind::PLANT &&
      observation.target.crop == view.spec.item &&
      observation.target.watered_today) {
    view.status = Status::Complete;
    view.crop_stage = CropStage::Satisfied;
    view.lease.reset();
    ++audit_.completed;
    return view.status;
  }
  if (view.spec.kind == Kind::Animal) {
    static_cast<void>(animals_.reconcile_external(
        view.animal_delegate_id, animal_observation(observation)));
    sync_animal(*state);
    if (view.status == Status::Complete)
      return view.status;
  }
  if (observation.day >
      std::max(view.spec.deadline_day, state->receipt_grace_day)) {
    view.status = Status::Expired;
    view.lease.reset();
    ++audit_.expired;
    return view.status;
  }
  if (view.spec.kind == Kind::Crop) {
    const auto debt = seed_purchases_.debt(view.purchase_debt_id);
    view.purchase_complete = debt && debt->acquisition_complete;
    if (observation.target.kind == fastkag::TileKind::WEED)
      view.crop_stage = CropStage::NeedDig;
    else if (observation.target.kind == fastkag::TileKind::PLANT &&
             observation.target.crop == view.spec.item &&
             !observation.target.watered_today)
      view.crop_stage = CropStage::NeedWater;
    else if (observation.target.kind == fastkag::TileKind::EMPTY &&
             view.purchase_complete)
      view.crop_stage = CropStage::NeedPlant;
    else
      view.crop_stage = CropStage::Acquire;
  }
  return view.status;
}

std::optional<ActorLease> Ledger::lease_actor(std::uint64_t objective_id,
                                              ActorIdentity actor, int step) {
  auto *state = find(objective_id);
  if (!state || state->view.status != Status::Open ||
      state->view.pending_unit_receipt || state->view.lease ||
      actor.actor_id < 0 || step < 0)
    return std::nullopt;
  ActorLease lease{objective_id, next_lease_token_++, actor, step};
  if (state->view.spec.kind == Kind::Animal) {
    auto delegate =
        animals_.lease_actor(state->view.animal_delegate_id, actor, step);
    if (!delegate)
      return std::nullopt;
    lease.token = delegate->token;
    state->animal_lease = *delegate;
  }
  state->view.lease = lease;
  ++audit_.leases_issued;
  return lease;
}

bool Ledger::release_actor(const ActorLease &lease) {
  auto *state = find(lease.objective_id);
  if (!state || state->view.pending_unit_receipt || !state->view.lease ||
      *state->view.lease != lease)
    return false;
  if (state->view.spec.kind == Kind::Animal) {
    if (!state->animal_lease || !animals_.release_actor(*state->animal_lease))
      return false;
    state->animal_lease.reset();
  }
  state->view.lease.reset();
  return true;
}

int Ledger::actor_disappeared(ActorIdentity actor) {
  int released = 0;
  for (auto &[key, state] : objectives_) {
    static_cast<void>(key);
    if (!state.view.lease || state.view.lease->actor != actor ||
        state.view.pending_unit_receipt)
      continue;
    const auto lease = *state.view.lease;
    if (release_actor(lease)) {
      ++released;
      ++audit_.actor_rebindings;
    }
  }
  return released;
}

std::vector<ReadyTransition>
Ledger::ready_transitions(std::span<const ReadyRequest> requests,
                          ResourceSnapshot resources) {
  std::vector<ReadyTransition> candidates;
  for (const auto &request : requests) {
    auto *state = find(request.objective_id);
    if (!state ||
        reconcile(request.objective_id, request.exact_before) != Status::Open ||
        state->view.pending_unit_receipt || !state->view.lease ||
        *state->view.lease != request.lease ||
        request.exact_before.actor.identity != request.lease.actor ||
        request.exact_before.step < request.lease.leased_step) {
      if (state && state->view.lease && *state->view.lease != request.lease)
        ++audit_.stale_lease_rejections;
      continue;
    }
    ReadyTransition candidate;
    candidate.objective_id = state->view.id;
    candidate.key = state->view.spec.key;
    candidate.kind = state->view.spec.kind;
    candidate.lease = request.lease;
    candidate.required_position = native_position(state->view.spec.key.tile);
    candidate.deadline_day = state->view.spec.deadline_day;
    candidate.critical = state->view.spec.critical;
    candidate.economic_value = state->view.spec.economic_value;

    if (state->view.spec.kind == Kind::Crop) {
      if (request.exact_before.actor.position != state->view.spec.key.tile)
        continue;
      CropProposal proposal;
      proposal.objective_id = state->view.id;
      proposal.revision = state->view.revision;
      proposal.lease = request.lease;
      proposal.exact_before = request.exact_before;
      if (state->view.crop_stage == CropStage::NeedDig)
        proposal.action = {fastkag::Op::DIG, fastkag::Item::NONE, 1};
      else if (state->view.crop_stage == CropStage::NeedPlant)
        proposal.action = {fastkag::Op::PLANT, state->view.spec.item, 1};
      else if (state->view.crop_stage == CropStage::NeedWater)
        proposal.action = {fastkag::Op::WATER, state->view.spec.item, 1};
      else
        continue;
      candidate.action = proposal.action;
      candidate.authority = proposal;
    } else {
      if (!state->animal_lease)
        continue;
      const auto proposal = animals_.propose_unit(
          state->view.animal_delegate_id, *state->animal_lease,
          animal_observation(request.exact_before));
      if (!proposal.proposal)
        continue;
      candidate.action = proposal.proposal->action;
      candidate.authority = *proposal.proposal;
    }
    candidates.push_back(std::move(candidate));
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ReadyTransition &left, const ReadyTransition &right) {
              return std::tuple{!left.critical, left.deadline_day,
                                -left.economic_value, left.key.origin_day,
                                left.objective_id} <
                     std::tuple{!right.critical, right.deadline_day,
                                -right.economic_value, right.key.origin_day,
                                right.objective_id};
            });
  std::set<ActorIdentity> actors;
  std::set<TileKey> tiles;
  std::vector<ReadyTransition> ready;
  for (auto &candidate : candidates) {
    if (actors.contains(candidate.lease.actor) ||
        tiles.contains(candidate.key.tile)) {
      ++audit_.resource_deferred;
      continue;
    }
    const int item = item_index(candidate.action.item);
    if (candidate.action.op == fastkag::Op::PLANT) {
      if (item < 0 || item >= fastkag::N_CROPS ||
          resources.seeds[static_cast<std::size_t>(item)] <= 0) {
        ++audit_.resource_deferred;
        continue;
      }
      --resources.seeds[static_cast<std::size_t>(item)];
    } else if (candidate.action.op == fastkag::Op::PICKUP && item >= 0 &&
               item < fastkag::N_ITEMS) {
      if (resources.shed[static_cast<std::size_t>(item)] <= 0) {
        ++audit_.resource_deferred;
        continue;
      }
      --resources.shed[static_cast<std::size_t>(item)];
    }
    actors.insert(candidate.lease.actor);
    tiles.insert(candidate.key.tile);
    ready.push_back(std::move(candidate));
    ++audit_.ready_transitions;
  }
  return ready;
}

UnitStageStatus Ledger::stage_transition(const ReadyTransition &transition,
                                         const UnitFinalSelection &selection) {
  auto *state = find(transition.objective_id);
  if (!state)
    return UnitStageStatus::UnknownObjective;
  if (state->view.status != Status::Open)
    return UnitStageStatus::RetiredObjective;
  if (!state->view.lease || *state->view.lease != transition.lease) {
    ++audit_.stale_lease_rejections;
    return UnitStageStatus::MissingOrStaleLease;
  }
  if (state->view.pending_unit_receipt)
    return UnitStageStatus::AlreadyPending;
  if (selection.submitted_step != selection.exact_before.step ||
      selection.selected_actor_slot != transition.lease.actor.actor_id ||
      selection.selected_actor_slot < 0 ||
      selection.selected_actor_slot >=
          static_cast<int>(selection.final_units.size()))
    return UnitStageStatus::InvalidSlot;
  if (selection.prefix_authority.final_manifest_generation == 0 ||
      selection.prefix_authority.phase_start_fingerprint == 0)
    return UnitStageStatus::BeforeWitnessMismatch;
  const auto expected_prefix = make_prefix_authority(
      selection.submitted_step, selection.selected_actor_slot,
      selection.prefix_authority.final_manifest_generation,
      selection.prefix_authority.phase_start_fingerprint, selection.final_units,
      selection.exact_before);
  if (expected_prefix != selection.prefix_authority)
    return UnitStageStatus::BeforeWitnessMismatch;
  if (!same_action(selection.final_units[static_cast<std::size_t>(
                       selection.selected_actor_slot)],
                   transition.action))
    return UnitStageStatus::FinalActionMismatch;
  if (std::any_of(pending_slots_.begin(), pending_slots_.end(),
                  [&](const PendingSlot &pending) {
                    return !pending.market &&
                           pending.submitted_step == selection.submitted_step &&
                           pending.slot == selection.selected_actor_slot;
                  }))
    return UnitStageStatus::SlotAlreadyBound;

  if (const auto *crop = std::get_if<CropProposal>(&transition.authority)) {
    if (state->view.spec.kind != Kind::Crop ||
        crop->objective_id != state->view.id ||
        crop->revision != state->view.revision ||
        crop->lease != transition.lease ||
        crop->exact_before != selection.exact_before ||
        !same_action(crop->action, transition.action))
      return UnitStageStatus::StaleTransition;
    const bool dig = crop->action.op == fastkag::Op::DIG &&
                     crop->exact_before.target.kind == fastkag::TileKind::WEED;
    const bool plant =
        crop->action.op == fastkag::Op::PLANT &&
        crop->action.item == state->view.spec.item &&
        crop->exact_before.target.kind == fastkag::TileKind::EMPTY &&
        state->view.purchase_complete &&
        crop->exact_before.seeds[static_cast<std::size_t>(
            item_index(state->view.spec.item))] > 0;
    const bool water =
        crop->action.op == fastkag::Op::WATER &&
        crop->exact_before.target.kind == fastkag::TileKind::PLANT &&
        crop->exact_before.target.crop == state->view.spec.item &&
        !crop->exact_before.target.watered_today;
    bool phase_seed_safe = true;
    if (plant) {
      phase_seed_safe = selection.phase_seed_snapshot_bound;
      int phase_demand = 0;
      for (const auto &final : selection.final_units)
        if (final.op == fastkag::Op::PLANT &&
            final.item == state->view.spec.item)
          ++phase_demand;
      const auto crop_index =
          static_cast<std::size_t>(item_index(state->view.spec.item));
      phase_seed_safe =
          phase_seed_safe && phase_demand > 0 &&
          phase_demand <= selection.phase_seeds_before[crop_index];
    }
    if ((!dig && !plant && !water) || !phase_seed_safe)
      return UnitStageStatus::StaleTransition;
    pending_crops_.push_back(
        {*crop, selection.selected_actor_slot, selection.prefix_authority});
  } else {
    const auto *animal =
        std::get_if<animal_lifecycle::UnitProposal>(&transition.authority);
    if (!animal || state->view.spec.kind != Kind::Animal ||
        animal->objective_id != state->view.animal_delegate_id ||
        animal->exact_before != animal_observation(selection.exact_before))
      return UnitStageStatus::StaleTransition;
    std::vector<fastkag::Action> final(selection.final_units.begin(),
                                       selection.final_units.end());
    const auto status = animals_.stage_unit(
        *animal, {selection.submitted_step, selection.selected_actor_slot,
                  final, animal_observation(selection.exact_before)});
    if (status != animal_lifecycle::UnitStageStatus::Selected)
      return UnitStageStatus::DelegateRejected;
  }
  state->view.pending_unit_receipt = true;
  state->pending_prefix_authority = selection.prefix_authority;
  pending_slots_.push_back({state->view.id, selection.submitted_step,
                            selection.selected_actor_slot, false});
  ++audit_.unit_submissions;
  return UnitStageStatus::Selected;
}

UnitSettlement Ledger::observe_transition(std::uint64_t objective_id,
                                          std::uint64_t lease_token,
                                          std::uint64_t prefix_fingerprint,
                                          const Observation &exact_after) {
  auto *state = find(objective_id);
  if (!state)
    return {objective_id, UnitReceiptStatus::UnknownOrDuplicate,
            Status::FailClosed, "unknown objective"};
  if (!state->view.pending_unit_receipt) {
    ++audit_.duplicate_receipts;
    return {objective_id, UnitReceiptStatus::UnknownOrDuplicate,
            state->view.status, "no pending receipt; duplicate ignored"};
  }
  UnitSettlement result{
      objective_id, UnitReceiptStatus::Failed, state->view.status, {}};
  if (state->view.spec.kind == Kind::Crop) {
    const auto found =
        std::find_if(pending_crops_.begin(), pending_crops_.end(),
                     [&](const PendingCrop &pending) {
                       return pending.proposal.objective_id == objective_id;
                     });
    if (found == pending_crops_.end()) {
      ++audit_.duplicate_receipts;
      return {objective_id, UnitReceiptStatus::UnknownOrDuplicate,
              state->view.status, "crop pending record absent"};
    }
    const auto pending = *found;
    const auto &before = pending.proposal.exact_before;
    const bool exact =
        exact_after.step == before.step + 1 &&
        exact_after.target_key == state->view.spec.key.tile &&
        exact_after.actor.identity == pending.proposal.lease.actor &&
        lease_token == pending.proposal.lease.token &&
        prefix_fingerprint ==
            prefix_authority_fingerprint(pending.prefix_authority);
    bool success = false;
    if (!exact) {
      result.status = UnitReceiptStatus::Ambiguous;
      result.reason = "receipt step/tile/actor lease is not exact";
      ++audit_.ambiguous_receipts;
    } else if (pending.proposal.action.op == fastkag::Op::DIG) {
      success = before.target.kind == fastkag::TileKind::WEED &&
                exact_after.target.kind == fastkag::TileKind::EMPTY;
    } else if (pending.proposal.action.op == fastkag::Op::PLANT) {
      const auto crop =
          static_cast<std::size_t>(item_index(state->view.spec.item));
      success = before.target.kind == fastkag::TileKind::EMPTY &&
                exact_after.target.kind == fastkag::TileKind::PLANT &&
                exact_after.target.crop == state->view.spec.item &&
                exact_after.seeds[crop] <= before.seeds[crop] - 1;
    } else if (pending.proposal.action.op == fastkag::Op::WATER) {
      success = before.target.kind == fastkag::TileKind::PLANT &&
                before.target.crop == state->view.spec.item &&
                !before.target.watered_today &&
                exact_after.target.kind == fastkag::TileKind::PLANT &&
                exact_after.target.crop == state->view.spec.item &&
                exact_after.target.watered_today;
    }
    const bool midnight_daily_reset =
        exact && pending.proposal.action.op == fastkag::Op::WATER &&
        exact_after.day == before.day + 1 &&
        exact_after.target.kind == fastkag::TileKind::PLANT &&
        exact_after.target.crop == state->view.spec.item &&
        !exact_after.target.watered_today;
    if (midnight_daily_reset) {
      result.status = UnitReceiptStatus::Ambiguous;
      result.reason = "midnight reset erased WATER effect; objective retained";
      state->receipt_grace_day =
          std::max(state->receipt_grace_day, exact_after.day);
      ++audit_.ambiguous_receipts;
    } else if (exact && success) {
      result.status = UnitReceiptStatus::Success;
      result.reason = "exact crop transition confirmed";
      ++state->view.revision;
      ++audit_.unit_receipt_successes;
    } else if (exact) {
      result.reason =
          "exact receipt lacks required crop effect; objective retained";
      ++audit_.unit_receipt_failures;
    }
    state->view.pending_unit_receipt = false;
    state->view.lease.reset();
    state->pending_prefix_authority.reset();
    pending_crops_.erase(found);
    release_slot(objective_id);
    if (result.status == UnitReceiptStatus::Success)
      static_cast<void>(reconcile(objective_id, exact_after));
    else {
      // A failed/ambiguous receipt cannot retire debt by replaying the same
      // untrusted after-image through the generic physical reconciler.
      state->view.status = Status::Open;
      if (midnight_daily_reset)
        state->view.crop_stage = CropStage::NeedWater;
    }
  } else {
    const bool prefix_exact =
        state->pending_prefix_authority &&
        prefix_fingerprint ==
            prefix_authority_fingerprint(*state->pending_prefix_authority);
    if (state->animal_lease) {
      // Always forward the receipt, including a stale token/generation.  The
      // delegate classifies it Ambiguous and, crucially, releases its pending
      // lease so a replacement generation can claim the plot safely.
      const auto settlement = animals_.observe_unit(
          {state->view.animal_delegate_id, exact_after.step,
           prefix_exact ? lease_token : 0, animal_observation(exact_after)});
      switch (settlement.status) {
      case animal_lifecycle::UnitReceiptStatus::Success:
        result.status = UnitReceiptStatus::Success;
        ++audit_.unit_receipt_successes;
        break;
      case animal_lifecycle::UnitReceiptStatus::Failed:
        result.status = UnitReceiptStatus::Failed;
        ++audit_.unit_receipt_failures;
        break;
      case animal_lifecycle::UnitReceiptStatus::Ambiguous:
        result.status = UnitReceiptStatus::Ambiguous;
        ++audit_.ambiguous_receipts;
        break;
      case animal_lifecycle::UnitReceiptStatus::UnknownObjective:
        result.status = UnitReceiptStatus::UnknownOrDuplicate;
        ++audit_.duplicate_receipts;
        break;
      }
      result.reason = settlement.reason;
    } else {
      result.status = UnitReceiptStatus::Ambiguous;
      result.reason = "wrapper lost animal lease while receipt was pending";
      ++audit_.ambiguous_receipts;
    }
    state->view.pending_unit_receipt = false;
    state->view.lease.reset();
    state->animal_lease.reset();
    state->pending_prefix_authority.reset();
    release_slot(objective_id);
    sync_animal(*state);
  }
  result.objective_status = state->view.status;
  return result;
}

std::optional<ObjectiveView>
Ledger::objective(std::uint64_t objective_id) const {
  const auto *state = find(objective_id);
  return state ? std::optional<ObjectiveView>(state->view) : std::nullopt;
}

std::optional<ObjectiveView> Ledger::objective(DayPlotKey key) const {
  const auto found = objectives_.find(key);
  return found == objectives_.end()
             ? std::nullopt
             : std::optional<ObjectiveView>(found->second.view);
}

std::vector<ObjectiveView> Ledger::objectives_for_day(int day) const {
  std::vector<ObjectiveView> result;
  for (const auto &[key, state] : objectives_)
    if (key.origin_day == day)
      result.push_back(state.view);
  return result;
}

std::vector<ObjectiveView> Ledger::active_objectives() const {
  std::vector<ObjectiveView> result;
  for (const auto &[key, state] : objectives_) {
    static_cast<void>(key);
    if (state.view.status == Status::Open)
      result.push_back(state.view);
  }
  return result;
}

Ledger::State *Ledger::find(std::uint64_t objective_id) {
  const auto found =
      std::find_if(objectives_.begin(), objectives_.end(), [&](auto &entry) {
        return entry.second.view.id == objective_id;
      });
  return found == objectives_.end() ? nullptr : &found->second;
}

const Ledger::State *Ledger::find(std::uint64_t objective_id) const {
  const auto found = std::find_if(
      objectives_.begin(), objectives_.end(),
      [&](const auto &entry) { return entry.second.view.id == objective_id; });
  return found == objectives_.end() ? nullptr : &found->second;
}

Ledger::State *Ledger::find_seed_debt(std::uint64_t debt_id) {
  const auto found =
      std::find_if(objectives_.begin(), objectives_.end(), [&](auto &entry) {
        return entry.second.view.spec.kind == Kind::Crop &&
               entry.second.view.purchase_debt_id == debt_id;
      });
  return found == objectives_.end() ? nullptr : &found->second;
}

Ledger::State *Ledger::find_animal_debt(std::uint64_t debt_id) {
  const auto found =
      std::find_if(objectives_.begin(), objectives_.end(), [&](auto &entry) {
        return entry.second.view.spec.kind == Kind::Animal &&
               entry.second.view.purchase_debt_id == debt_id;
      });
  return found == objectives_.end() ? nullptr : &found->second;
}

bool Ledger::same_action(const fastkag::Action &left,
                         const fastkag::Action &right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

animal_lifecycle::Observation
Ledger::animal_observation(const Observation &observation) {
  animal_lifecycle::Observation result;
  result.step = observation.step;
  result.day = observation.day;
  result.target = {observation.target.kind,
                   observation.target.animal,
                   observation.target.placed_day,
                   observation.target.yield_units,
                   observation.target.consecutive_unfed,
                   observation.target.fed_today,
                   observation.target.cared_today};
  result.shed = observation.shed;
  result.actor = observation.actor;
  return result;
}

void Ledger::sync_animal(State &state) {
  const auto delegate = animals_.objective(state.view.animal_delegate_id);
  if (!delegate) {
    state.view.status = Status::FailClosed;
    state.view.last_reason = "animal lifecycle delegate disappeared";
    return;
  }
  state.view.purchase_complete =
      delegate->stage != animal_lifecycle::Stage::Acquire;
  if (delegate->complete && state.view.status != Status::Complete) {
    state.view.status = Status::Complete;
    state.view.lease.reset();
    ++audit_.completed;
  }
}

void Ledger::release_slot(std::uint64_t objective_id) {
  std::erase_if(pending_slots_, [&](const PendingSlot &pending) {
    return !pending.market && pending.objective_id == objective_id;
  });
}

const char *open_status_name(OpenStatus status) noexcept {
  switch (status) {
  case OpenStatus::Opened:
    return "opened";
  case OpenStatus::InvalidDayOrTile:
    return "invalid-day-or-tile";
  case OpenStatus::UnsupportedItem:
    return "unsupported-item";
  case OpenStatus::InvalidQuantityOrCost:
    return "invalid-quantity-or-cost";
  case OpenStatus::ActiveTileOwner:
    return "active-tile-owner";
  case OpenStatus::DelegateRejected:
    return "delegate-rejected";
  }
  return "unknown";
}

const char *unit_stage_status_name(UnitStageStatus status) noexcept {
  switch (status) {
  case UnitStageStatus::Selected:
    return "selected";
  case UnitStageStatus::UnknownObjective:
    return "unknown-objective";
  case UnitStageStatus::RetiredObjective:
    return "retired-objective";
  case UnitStageStatus::StaleTransition:
    return "stale-transition";
  case UnitStageStatus::MissingOrStaleLease:
    return "missing-or-stale-lease";
  case UnitStageStatus::AlreadyPending:
    return "already-pending";
  case UnitStageStatus::SlotAlreadyBound:
    return "slot-already-bound";
  case UnitStageStatus::InvalidSlot:
    return "invalid-slot";
  case UnitStageStatus::FinalActionMismatch:
    return "final-action-mismatch";
  case UnitStageStatus::BeforeWitnessMismatch:
    return "before-witness-mismatch";
  case UnitStageStatus::DelegateRejected:
    return "delegate-rejected";
  }
  return "unknown";
}

} // namespace g001::persistent_production
