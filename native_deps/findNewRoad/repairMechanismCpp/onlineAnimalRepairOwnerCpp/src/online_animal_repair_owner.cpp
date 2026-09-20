#include "online_animal_repair_owner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace g001::online_animal_repair {
namespace {

int index(fastkag::Item item) { return static_cast<int>(item); }

bool movement(fastkag::Op op) {
  return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
      op == fastkag::Op::EAST || op == fastkag::Op::WEST;
}

bool same_action(const fastkag::Action& left, const fastkag::Action& right) {
  return left.op == right.op && left.item == right.item &&
      left.quantity == right.quantity;
}

fastkag::Position actor_position(const fastkag::Simulator& simulator,
                                 int player, int actor) {
  const auto& farm = simulator.farms()[player];
  if (actor == 0) return farm.farmer;
  if (actor < 0 || actor > static_cast<int>(farm.hands.size()))
    return {-1, -1};
  return farm.hands[static_cast<std::size_t>(actor - 1)];
}

bool shed_adjacent(const fastkag::Simulator& simulator,
                   fastkag::Position position) {
  const int half = simulator.config().board_size / 2;
  return (position.x == half - 1 || position.x == half) &&
      (position.y == half - 1 || position.y == half);
}

const fastkag::Tile* tile_at(const fastkag::Simulator& simulator, int player,
                             persistent::TileKey key) {
  const int size = simulator.config().board_size;
  if (key.row < 0 || key.column < 0 || key.row >= size ||
      key.column >= size)
    return nullptr;
  return &simulator.farms()[player].tiles[static_cast<std::size_t>(
      key.row * size + key.column)];
}

int animal_cost(fastkag::Item animal) {
  const auto definition = animal_lifecycle::definition(animal);
  return definition ? definition->purchase_cost : 0;
}

}  // namespace

std::string OnlineAnimalRepairOwner::name() const {
  return "online_animal_repair_typed_offline";
}

persistent::OpenResult OnlineAnimalRepairOwner::ensure_objective(
    int day, persistent::TileKey target, fastkag::Item animal,
    int deadline_day, std::string provenance) {
  if (const auto found = target_owner_.find(target);
      found != target_owner_.end()) {
    const auto current = ledger_.objective(found->second);
    if (current && current->status != persistent::Status::Complete &&
        current->status != persistent::Status::Expired)
      return {persistent::OpenStatus::ActiveTileOwner, found->second};
  }
  const int cost = animal_cost(animal);
  auto opened = ledger_.open({{day, target}, persistent::Kind::Animal, animal,
                              1, cost, deadline_day, true, 500,
                              std::move(provenance)});
  if (opened.status == persistent::OpenStatus::Opened) {
    target_owner_[target] = opened.objective_id;
    ++audit_.objectives_opened;
  }
  return opened;
}

persistent::ActorIdentity OnlineAnimalRepairOwner::actor_identity(
    const fastkag::Simulator& simulator, int actor) const {
  const auto generation =
      (static_cast<std::uint64_t>(simulator.day() + 1) << 32U) |
      static_cast<std::uint32_t>(actor + 1);
  return {actor, generation};
}

persistent::Observation OnlineAnimalRepairOwner::observation(
    const fastkag::Simulator& simulator, int player,
    persistent::TileKey target, int actor) const {
  persistent::Observation result;
  result.step = simulator.step_count();
  result.day = simulator.day();
  result.target_key = target;
  if (const auto* tile = tile_at(simulator, player, target))
    result.target = *tile;
  const auto& private_state = simulator.privates()[player];
  for (int item = 0; item < fastkag::N_ITEMS; ++item)
    result.shed[static_cast<std::size_t>(item)] =
        private_state.shed[static_cast<std::size_t>(item)];
  for (int crop = 0; crop < fastkag::N_CROPS; ++crop)
    result.seeds[static_cast<std::size_t>(crop)] =
        private_state.seeds[static_cast<std::size_t>(crop)];
  result.actor.identity = actor_identity(simulator, actor);
  const auto position = actor_position(simulator, player, actor);
  result.actor.position = {position.y, position.x};
  result.actor.shed_adjacent = shed_adjacent(simulator, position);
  if (actor >= 0 &&
      actor < static_cast<int>(private_state.inventories.size()))
    for (int item = 0; item < fastkag::N_ITEMS; ++item)
      result.actor.inventory[static_cast<std::size_t>(item)] =
          private_state.inventories[static_cast<std::size_t>(actor)]
                                   [static_cast<std::size_t>(item)];
  return result;
}

bool OnlineAnimalRepairOwner::raw_unit_neighborhood_is_sealed(
    const repair_fork::RepairContext& context, int selected_actor) const {
  for (int actor = 0;
       actor < static_cast<int>(context.raw_g001.units.size()); ++actor) {
    if (actor == selected_actor) continue;
    const auto op = context.raw_g001.units[static_cast<std::size_t>(actor)].op;
    if (op != fastkag::Op::PASS && !movement(op)) return false;
  }
  return true;
}

int OnlineAnimalRepairOwner::protected_market_cash(
    const repair_fork::RepairContext& context) const {
  std::int64_t protected_cash = 0;
  for (const auto& action : context.raw_g001.market) {
    const int quantity = std::max(0, action.quantity);
    if (action.op == fastkag::Op::BUY_SEED) {
      static constexpr std::array<int, fastkag::N_CROPS> costs{
          10, 20, 50, 100, 80};
      const int item = index(action.item);
      if (item >= 0 && item < fastkag::N_CROPS)
        protected_cash += static_cast<std::int64_t>(quantity) * costs[item];
    } else if (action.op == fastkag::Op::BUY_ANIMAL) {
      protected_cash += static_cast<std::int64_t>(quantity) *
          animal_cost(action.item);
    } else if (action.op == fastkag::Op::BUY_PRODUCT) {
      const int item = index(action.item);
      if (item >= 0 && item < fastkag::N_PRODUCTS)
        protected_cash += static_cast<std::int64_t>(quantity) *
            context.phase_start.market().prices[static_cast<std::size_t>(item)];
    }
  }
  return static_cast<int>(std::min<std::int64_t>(
      std::numeric_limits<int>::max(), protected_cash));
}

void OnlineAnimalRepairOwner::discover_hard_animal(
    const repair_fork::RepairContext& context,
    repair_fork::RepairTelemetry& telemetry) {
  for (const auto& action : context.raw_g001.market) {
    if (action.op != fastkag::Op::BUY_ANIMAL || action.quantity <= 0 ||
        !animal_lifecycle::definition(action.item))
      continue;
    const auto definition = *animal_lifecycle::definition(action.item);
    const auto& farm = context.phase_start.farms()[context.player];
    const int size = context.phase_start.config().board_size;
    for (int tile_index = 0; tile_index < static_cast<int>(farm.tiles.size());
         ++tile_index) {
      const auto& tile = farm.tiles[static_cast<std::size_t>(tile_index)];
      if (tile.kind != definition.required_structure ||
          tile.animal != fastkag::Item::NONE)
        continue;
      const persistent::TileKey key{tile_index / size, tile_index % size};
      if (target_owner_.contains(key)) continue;
      const auto opened = ensure_objective(
          context.phase_start.day(), key, action.item,
          std::max(context.phase_start.day(),
                   context.phase_start.config().episode_steps /
                       context.phase_start.config().turns_per_day - 1),
          "observed-hard-g001-animal");
      if (opened.status == persistent::OpenStatus::Opened) {
        ++telemetry.debts_opened;
        break;
      }
    }
  }
}

void OnlineAnimalRepairOwner::settle_previous(
    const repair_fork::RepairContext& context,
    repair_fork::RepairTelemetry& telemetry) {
  for (const auto& pending : pending_units_) {
    if (pending.submitted_step + 1 != context.step) continue;
    const auto receipt = std::find_if(
        context.previous_action_receipts.begin(),
        context.previous_action_receipts.end(),
        [&](const repair_fork::ActionReceipt& candidate) {
          return candidate.submitted_step == pending.submitted_step &&
              candidate.actor == pending.actor.actor_id;
        });
    const bool receipt_exact =
        receipt != context.previous_action_receipts.end() &&
        receipt->manifest_generation == pending.manifest_generation &&
        receipt->prefix_manifest_hash == pending.prefix_manifest_hash &&
        receipt->post_prefix_state_fingerprint ==
            pending.post_prefix_state_fingerprint &&
        same_action(receipt->emitted, pending.emitted);
    const auto after = observation(context.phase_start, context.player,
                                   pending.target, pending.actor.actor_id);
    const auto settlement = ledger_.observe_transition(
        pending.objective_id, pending.lease_token,
        receipt_exact ? pending.prefix_fingerprint : 0, after);
    if (settlement.status == persistent::UnitReceiptStatus::Success) {
      ++audit_.unit_receipt_successes;
      ++telemetry.receipts_confirmed;
      if (settlement.objective_status == persistent::Status::Complete) {
        ++telemetry.debts_closed;
      }
    } else {
      ++audit_.unit_receipt_failures;
      ++telemetry.receipts_failed;
    }
  }
  std::erase_if(pending_units_, [&](const PendingUnit& pending) {
    return pending.submitted_step < context.step;
  });

  for (const auto& receipt : context.previous_purchase_receipts) {
    const auto found = std::find_if(
        pending_purchases_.begin(), pending_purchases_.end(),
        [&](const PendingPurchase& pending) {
          return pending.objective_id == receipt.debt_id &&
              pending.submitted_step == receipt.submitted_step;
        });
    if (found == pending_purchases_.end()) continue;
    ++audit_.purchase_receipts;
    const bool compiled =
        receipt.compile_status == repair_fork::PurchaseCompileStatus::BoundExisting ||
        receipt.compile_status == repair_fork::PurchaseCompileStatus::Appended;
    if (!compiled || receipt.market_slot < 0 ||
        receipt.operation != found->proposal.order.op ||
        receipt.item != found->proposal.order.item ||
        receipt.requested != found->proposal.order.quantity) {
      ++audit_.purchase_retries;
      ++telemetry.receipts_failed;
      pending_purchases_.erase(found);
      continue;
    }
    std::vector<fastkag::Action> final_market(
        static_cast<std::size_t>(receipt.market_slot + 1));
    final_market[static_cast<std::size_t>(receipt.market_slot)] =
        found->proposal.order;
    const auto staged = ledger_.stage_purchase(
        found->objective_id, found->proposal,
        {found->submitted_step, receipt.market_slot, final_market,
         found->holding_before, found->cash_before, found->protected_cash});
    if (staged != purchase_recovery::StageStatus::Selected) {
      ++audit_.purchase_retries;
      ++telemetry.receipts_failed;
      pending_purchases_.erase(found);
      continue;
    }
    std::vector<std::int32_t> fills(
        static_cast<std::size_t>(receipt.market_slot + 1));
    fills[static_cast<std::size_t>(receipt.market_slot)] = receipt.filled;
    purchase_recovery::ReceiptObservation observed;
    observed.step = context.step;
    observed.slot_fills = fills;
    const auto& private_state = context.phase_start.privates()[context.player];
    for (int animal = 0; animal < fastkag::N_ANIMALS; ++animal) {
      const int item = static_cast<int>(fastkag::Item::GOOSE) + animal;
      int unplaced = private_state.shed[static_cast<std::size_t>(item)];
      for (const auto& inventory : private_state.inventories)
        unplaced += inventory[static_cast<std::size_t>(item)];
      observed.animals_after[static_cast<std::size_t>(animal)] = unplaced;
    }
    const auto settlements = ledger_.observe_purchases(observed);
    bool confirmed = false;
    for (const auto& settlement : settlements) {
      if (settlement.objective_id != found->objective_id) continue;
      confirmed = settlement.settlement.status ==
              purchase_recovery::FillStatus::Full ||
          settlement.settlement.status ==
              purchase_recovery::FillStatus::Partial;
      if (settlement.settlement.status ==
              purchase_recovery::FillStatus::Zero ||
          settlement.settlement.status ==
              purchase_recovery::FillStatus::Ambiguous)
        ++audit_.purchase_retries;
    }
    if (confirmed) ++telemetry.receipts_confirmed;
    else ++telemetry.receipts_failed;
    pending_purchases_.erase(found);
  }
}

repair_fork::RepairDecision OnlineAnimalRepairOwner::decide(
    const repair_fork::RepairContext& context) {
  if (context.player < 0 || context.player > 1 ||
      context.step != context.phase_start.step_count())
    throw std::invalid_argument("invalid online animal repair context");
  repair_fork::RepairDecision decision;
  decision.units = context.raw_g001.units;
  // The evaluator owns these receipts. Echoing is only an acknowledgement;
  // settle_previous independently validates every token against local pending
  // state before allowing it to confirm an effect.
  decision.receipt_acks.assign(context.previous_action_receipts.begin(),
                               context.previous_action_receipts.end());
  decision.purchase_receipt_acks.assign(
      context.previous_purchase_receipts.begin(),
      context.previous_purchase_receipts.end());
  decision.sources.reserve(decision.units.size());
  for (std::size_t actor = 0; actor < decision.units.size(); ++actor)
    decision.sources.push_back(
        {static_cast<int>(actor), context.step, decision.units[actor]});

  settle_previous(context, decision.telemetry);
  discover_hard_animal(context, decision.telemetry);

  const auto phase_fingerprint =
      repair_fork::phase_start_fingerprint(context.phase_start, context.player);
  std::set<std::uint64_t> purchase_pending;
  for (const auto& pending : pending_purchases_)
    purchase_pending.insert(pending.objective_id);
  for (const auto& objective : ledger_.active_objectives()) {
    if (objective.spec.kind != persistent::Kind::Animal ||
        objective.purchase_complete ||
        purchase_pending.contains(objective.id))
      continue;
    std::optional<purchase_recovery::Proposal> proposal;
    for (const auto& raw : context.raw_g001.market) {
      if (raw.op == fastkag::Op::BUY_ANIMAL &&
          raw.item == objective.spec.item && raw.quantity > 0) {
        proposal = ledger_.describe_hard_purchase(objective.id, raw.quantity);
        break;
      }
    }
    const int protected_cash = protected_market_cash(context);
    if (!proposal)
      proposal = ledger_.propose_purchase(
          objective.id,
          {context.step,
           static_cast<int>(std::floor(
               context.phase_start.farms()[context.player].money)),
           protected_cash});
    if (!proposal) continue;
    const int animal_item = index(objective.spec.item);
    const auto& private_state = context.phase_start.privates()[context.player];
    int holding = private_state.shed[static_cast<std::size_t>(animal_item)];
    for (const auto& inventory : private_state.inventories)
      holding += inventory[static_cast<std::size_t>(animal_item)];
    decision.required_purchases.push_back(
        {objective.id, proposal->order.op, proposal->order.item,
         proposal->order.quantity, context.step});
    pending_purchases_.push_back(
        {objective.id, context.step, *proposal, holding,
         static_cast<int>(std::floor(
             context.phase_start.farms()[context.player].money)),
         protected_cash});
    ++audit_.purchase_requests;
  }

  // Full-step hour-23 receipts erase daily FEED/CARE evidence and retire all
  // hired generations. Purchases are safe because market runs after units and
  // bought animals remain in the shed; unit takeover is intentionally closed.
  const bool hour23 = context.phase_start.hour() ==
      context.phase_start.config().turns_per_day - 1;
  bool unit_committed = false;
  for (const auto& objective : ledger_.active_objectives()) {
    if (unit_committed || objective.spec.kind != persistent::Kind::Animal ||
        !objective.purchase_complete ||
        objective.pending_unit_receipt)
      continue;
    for (int actor = 0;
         actor < static_cast<int>(decision.units.size()); ++actor) {
      if (hour23) {
        ++audit_.hour23_blocks;
        ++decision.telemetry.fail_closed;
        break;
      }
      if (!raw_unit_neighborhood_is_sealed(context, actor)) {
        ++audit_.raw_obligation_blocks;
        break;
      }
      const auto& raw = context.raw_g001.units[static_cast<std::size_t>(actor)];
      if (movement(raw.op)) continue;

      auto lower = context.raw_joint;
      lower[static_cast<std::size_t>(context.player)].units = decision.units;
      for (int slot = actor;
           slot < static_cast<int>(lower[static_cast<std::size_t>(
               context.player)].units.size()); ++slot)
        lower[static_cast<std::size_t>(context.player)]
            .units[static_cast<std::size_t>(slot)] = {};
      if (context.player == 0)
        for (auto& action : lower[1].units) action = {};
      const auto prefix = context.phase_start.preview_unit_phase(lower);
      auto before = observation(prefix, context.player,
                                objective.spec.key.tile, actor);
      const int external_before =
          ledger_.animal_audit().external_inventory_reconciles;
      static_cast<void>(ledger_.reconcile(objective.id, before));
      audit_.external_reconciles +=
          ledger_.animal_audit().external_inventory_reconciles -
          external_before;
      const auto lease = ledger_.lease_actor(
          objective.id, before.actor.identity, context.step);
      if (!lease) continue;
      const std::array<persistent::ReadyRequest, 1> request{
          persistent::ReadyRequest{objective.id, *lease, before}};
      persistent::ResourceSnapshot resources;
      resources.seeds = before.seeds;
      resources.shed = before.shed;
      const auto ready = ledger_.ready_transitions(request, resources);
      if (ready.size() != 1) {
        static_cast<void>(ledger_.release_actor(*lease));
        continue;
      }
      const auto& transition = ready.front();
      if (raw.op != fastkag::Op::PASS &&
          !same_action(raw, transition.action)) {
        ++audit_.raw_obligation_blocks;
        static_cast<void>(ledger_.release_actor(*lease));
        continue;
      }
      decision.units[static_cast<std::size_t>(actor)] = transition.action;
      const auto manifest_generation = final_manifest_generation_++;
      const auto authority = persistent::make_prefix_authority(
          context.step, actor, manifest_generation, phase_fingerprint,
          decision.units, before);
      const auto stage = ledger_.stage_transition(
          transition,
          {context.step, actor, decision.units, before, authority,
           context.phase_start.privates()[context.player].seeds, true});
      if (stage != persistent::UnitStageStatus::Selected) {
        decision.units[static_cast<std::size_t>(actor)] = raw;
        static_cast<void>(ledger_.release_actor(*lease));
        ++decision.telemetry.fail_closed;
        continue;
      }
      decision.sources[static_cast<std::size_t>(actor)] =
          same_action(raw, transition.action)
          ? repair_fork::SourceBinding{actor, context.step, raw}
          : repair_fork::SourceBinding{actor, -1, {}};
      auto final_joint = context.raw_joint;
      final_joint[static_cast<std::size_t>(context.player)].units =
          decision.units;
      const repair_fork::ActorPrefixAuthority fork_authority{
          actor, manifest_generation,
          repair_fork::unit_prefix_manifest_hash(decision.units, actor),
          repair_fork::post_unit_prefix_state_fingerprint(
              context.phase_start, context.player, final_joint, actor)};
      decision.prefix_authority.push_back(fork_authority);
      pending_units_.push_back(
          {objective.id, lease->token,
           persistent::prefix_authority_fingerprint(authority),
           lease->actor, objective.spec.key.tile, context.step,
           fork_authority.manifest_generation,
           fork_authority.prefix_manifest_hash,
           fork_authority.post_prefix_state_fingerprint,
           transition.action});
      ++audit_.unit_proposals;
      ++audit_.unit_commits;
      ++decision.telemetry.triggers;
      unit_committed = true;
      break;
    }
    if (!unit_committed) ++audit_.no_service_slot;
  }

  for (std::size_t actor = 0; actor < decision.units.size(); ++actor)
    if (movement(context.raw_g001.units[actor].op) &&
        !same_action(context.raw_g001.units[actor], decision.units[actor]))
      ++audit_.move_changes;
  return decision;
}

}  // namespace g001::online_animal_repair
