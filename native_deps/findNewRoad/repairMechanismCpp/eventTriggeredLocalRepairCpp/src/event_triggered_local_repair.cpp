#include "../include/event_triggered_local_repair.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace g001::event_local_repair {

namespace {

bool is_crop_operation(Op operation) noexcept {
  return operation == Op::Plant || operation == Op::Water ||
         operation == Op::Harvest;
}

bool is_protected_production(Op operation) noexcept {
  return operation == Op::Dig || operation == Op::Plant ||
         operation == Op::Build || operation == Op::Water ||
         operation == Op::Harvest;
}

bool same_source_effect(const Action& source, const Action& required) noexcept {
  const bool native_crop_wildcard =
      (source.op == Op::Water || source.op == Op::Harvest) &&
      (source.item == -1 || required.item == -1);
  return source.op == required.op &&
         (source.item == required.item || native_crop_wildcard) &&
         source.quantity == required.quantity;
}

bool same_moves(const std::vector<PlannedAction>& baseline,
                const std::vector<std::optional<Action>>& overlays) {
  std::vector<Action> before;
  std::vector<Action> after;
  for (std::size_t turn = 0; turn < baseline.size(); ++turn) {
    if (baseline[turn].action.op == Op::Move)
      before.push_back(baseline[turn].action);
    const auto& effective = overlays[turn].has_value()
                                ? *overlays[turn]
                                : baseline[turn].action;
    if (effective.op == Op::Move) after.push_back(effective);
  }
  return before == after;
}

}  // namespace

Compiler::Compiler(Config config) : config_(config) {
  if (config_.maximum_stationary_sink_loss < 0)
    throw std::invalid_argument("negative stationary sink loss");
  if (config_.crop_harvests_to_close < 1)
    throw std::invalid_argument("crop_harvests_to_close must be positive");
}

void Compiler::install_day(DayPlan plan) {
  if (plan.day < 0 || plan.turns <= 0 || plan.actors.empty())
    throw std::invalid_argument("invalid day plan header");
  for (const auto& actor : plan.actors) {
    if (actor.actor < 0 || static_cast<int>(actor.turns.size()) != plan.turns)
      throw std::invalid_argument("invalid actor day plan");
  }
  for (std::size_t left = 0; left < plan.actors.size(); ++left)
    for (std::size_t right = left + 1; right < plan.actors.size(); ++right)
      if (plan.actors[left].actor == plan.actors[right].actor)
        throw std::invalid_argument("duplicate actor in day plan");
  day_plan_ = std::move(plan);
  coalesced_current_day_sources_.clear();
  overlays_.assign(day_plan_->actors.size(),
                   std::vector<std::optional<Action>>(day_plan_->turns));
  overlay_sink_losses_.assign(
      day_plan_->actors.size(),
      std::vector<std::optional<int>>(day_plan_->turns));
  baseline_source_ids_.assign(
      day_plan_->actors.size(),
      std::vector<std::uint64_t>(day_plan_->turns));
  overlay_source_ids_.assign(
      day_plan_->actors.size(),
      std::vector<std::optional<std::uint64_t>>(day_plan_->turns));
  for (auto& actor_sources : baseline_source_ids_)
    for (auto& source : actor_sources) source = next_source_action_id_++;
}

int Compiler::actor_index(int actor) const {
  if (!day_plan_) return -1;
  for (std::size_t index = 0; index < day_plan_->actors.size(); ++index)
    if (day_plan_->actors[index].actor == actor)
      return static_cast<int>(index);
  return -1;
}

const ActorObservation* Compiler::observation_for(const TurnInput& input,
                                                   int actor) const {
  const auto found = std::find_if(input.actors.begin(), input.actors.end(),
                                  [actor](const auto& observation) {
                                    return observation.actor == actor;
                                  });
  return found == input.actors.end() ? nullptr : &*found;
}

Compiler::Transaction* Compiler::find_plot_transaction(Position tile) {
  const auto found = std::find_if(
      transactions_.begin(), transactions_.end(), [&](const auto& entry) {
        return !entry.second.view.completed &&
            entry.second.view.key.tile == tile;
      });
  return found == transactions_.end() ? nullptr : &found->second;
}

const Compiler::Transaction* Compiler::find_plot_transaction(
    Position tile) const {
  const auto found = std::find_if(
      transactions_.begin(), transactions_.end(), [&](const auto& entry) {
        return !entry.second.view.completed &&
            entry.second.view.key.tile == tile;
      });
  return found == transactions_.end() ? nullptr : &found->second;
}

Compiler::Transaction& Compiler::ensure_transaction(
    const TransactionKey& key, GoalKind goal, int desired_item, int day,
    int turn) {
  const auto compatible = std::find_if(
      transactions_.begin(), transactions_.end(), [&](const auto& entry) {
        const auto& view = entry.second.view;
        return !view.completed && view.key.tile == key.tile &&
            view.goal == goal &&
            (view.desired_item == desired_item || view.desired_item == -1 ||
             desired_item == -1);
      });
  if (compatible != transactions_.end()) {
    if (compatible->second.view.desired_item == -1 && desired_item != -1)
      compatible->second.view.desired_item = desired_item;
    return compatible->second;
  }
  // A plot has one active objective. A later incompatible desired crop or
  // goal explicitly supersedes the old objective instead of accumulating an
  // impossible parallel lifecycle.
  for (auto& [id, entry] : transactions_) {
    (void)id;
    if (entry.view.completed || !(entry.view.key.tile == key.tile)) continue;
    entry.view.completed = true;
    entry.view.superseded = true;
    entry.view.outstanding_source_actions = 0;
    entry.source_debts.clear();
    ++audit_.objectives_superseded;
  }
  Transaction transaction;
  transaction.view.id = next_transaction_id_++;
  transaction.view.source_epoch = transaction.view.id;
  transaction.view.key = key;
  transaction.view.goal = goal;
  transaction.view.desired_item = desired_item;
  transaction.view.origin_day = day;
  transaction.view.origin_turn = turn;
  const auto id = transaction.view.id;
  auto [inserted, ok] = transactions_.emplace(id, std::move(transaction));
  (void)ok;
  ++audit_.transactions_created;
  return inserted->second;
}

std::optional<Action> Compiler::required_action(
    const Transaction& transaction,
    const ActorObservation& observation) const {
  const auto& view = transaction.view;
  if (view.completed) return std::nullopt;
  if (view.goal == GoalKind::Structure) {
    if (observation.tile.kind == TileKind::Weed)
      return Action{Op::Dig, 0, 1, 0, 0};
    if (observation.tile.kind == TileKind::Empty)
      return Action{Op::Build, view.desired_item, 1, 0, 0};
    return std::nullopt;
  }

  if (observation.tile.kind == TileKind::Weed)
    return Action{Op::Dig, 0, 1, 0, 0};
  if (observation.tile.kind == TileKind::Empty) {
    if (observation.desired_item_inventory <= 0) return std::nullopt;
    return Action{Op::Plant, view.desired_item, 1, 0, 0};
  }
  if (observation.tile.kind != TileKind::Crop ||
      (view.desired_item != -1 && observation.tile.item != view.desired_item))
    return std::nullopt;
  if (observation.tile.harvest_legal)
    return Action{Op::Harvest, view.desired_item, 1, 0, 0};
  if (!observation.tile.watered_today)
    return Action{Op::Water, view.desired_item, 1, 0, 0};
  return std::nullopt;
}

bool Compiler::action_is_blocked(const Action& action,
                                 const ActorObservation& observation,
                                 GoalKind& goal,
                                 int& desired_item) const {
  desired_item = action.item != -1 ? action.item : observation.desired_item_hint;
  if (action.op == Op::Build) {
    goal = GoalKind::Structure;
    return observation.tile.kind != TileKind::Empty ||
           observation.failed_fill_observed;
  }
  if (!is_crop_operation(action.op)) return false;
  goal = GoalKind::Crop;
  const bool desired_crop =
      observation.tile.kind == TileKind::Crop &&
      (action.item == -1 || observation.tile.item == action.item);
  if (action.op == Op::Plant)
    return observation.failed_fill_observed ||
           observation.desired_item_inventory <= 0 ||
           observation.tile.kind != TileKind::Empty;
  if (action.op == Op::Water)
    return !desired_crop || observation.tile.watered_today;
  return !desired_crop || !observation.tile.harvest_legal;
}

bool Compiler::is_sink(int index, int turn) const {
  const auto& action = effective_action(index, turn);
  if (action.op == Op::Move || is_protected_production(action.op)) return false;
  if (action.op == Op::Pass) return true;
  return effective_sink_loss(index, turn) <=
         config_.maximum_stationary_sink_loss;
}

const Action& Compiler::effective_action(int index, int turn) const {
  const auto& overlay = overlays_[static_cast<std::size_t>(index)]
                                 [static_cast<std::size_t>(turn)];
  if (overlay.has_value()) return *overlay;
  return day_plan_->actors[static_cast<std::size_t>(index)]
      .turns[static_cast<std::size_t>(turn)]
      .action;
}

std::uint64_t Compiler::effective_source_id(int index, int turn) const {
  const auto& overlay_source =
      overlay_source_ids_[static_cast<std::size_t>(index)]
                         [static_cast<std::size_t>(turn)];
  if (overlay_source.has_value()) return *overlay_source;
  return baseline_source_ids_[static_cast<std::size_t>(index)]
                             [static_cast<std::size_t>(turn)];
}

int Compiler::effective_sink_loss(int index, int turn) const {
  const auto& overlay_loss =
      overlay_sink_losses_[static_cast<std::size_t>(index)]
                          [static_cast<std::size_t>(turn)];
  if (overlay_loss.has_value()) return *overlay_loss;
  return day_plan_->actors[static_cast<std::size_t>(index)]
      .turns[static_cast<std::size_t>(turn)]
      .certified_stationary_loss;
}

bool Compiler::install_insertion(int index, int turn, const Action& inserted,
                                 std::uint64_t inserted_source_id) {
  auto& actor_overlays = overlays_[static_cast<std::size_t>(index)];
  auto& actor_overlay_losses =
      overlay_sink_losses_[static_cast<std::size_t>(index)];
  auto& actor_overlay_sources =
      overlay_source_ids_[static_cast<std::size_t>(index)];
  const auto& actor_plan = day_plan_->actors[static_cast<std::size_t>(index)];

  int sink = -1;
  for (int candidate = turn + 1; candidate < day_plan_->turns; ++candidate) {
    if (is_sink(index, candidate)) {
      sink = candidate;
      break;
    }
  }
  if (sink < 0) return false;

  std::vector<std::optional<Action>> before = actor_overlays;
  std::vector<std::optional<int>> losses_before = actor_overlay_losses;
  std::vector<std::optional<std::uint64_t>> sources_before =
      actor_overlay_sources;
  std::vector<Action> displaced_actions;
  std::vector<int> displaced_losses;
  std::vector<std::uint64_t> displaced_sources;
  displaced_actions.reserve(static_cast<std::size_t>(sink - turn));
  displaced_losses.reserve(static_cast<std::size_t>(sink - turn));
  displaced_sources.reserve(static_cast<std::size_t>(sink - turn));
  for (int slot = turn; slot < sink; ++slot) {
    displaced_actions.push_back(effective_action(index, slot));
    displaced_losses.push_back(effective_sink_loss(index, slot));
    displaced_sources.push_back(effective_source_id(index, slot));
  }

  actor_overlays[static_cast<std::size_t>(turn)] = inserted;
  actor_overlay_losses[static_cast<std::size_t>(turn)] =
      std::numeric_limits<int>::max();
  actor_overlay_sources[static_cast<std::size_t>(turn)] = inserted_source_id;
  for (int slot = turn + 1; slot <= sink; ++slot)
    actor_overlays[static_cast<std::size_t>(slot)] =
        displaced_actions[static_cast<std::size_t>(slot - turn - 1)];
  for (int slot = turn + 1; slot <= sink; ++slot)
    actor_overlay_losses[static_cast<std::size_t>(slot)] =
        displaced_losses[static_cast<std::size_t>(slot - turn - 1)];
  for (int slot = turn + 1; slot <= sink; ++slot)
    actor_overlay_sources[static_cast<std::size_t>(slot)] =
        displaced_sources[static_cast<std::size_t>(slot - turn - 1)];
  if (!same_moves(actor_plan.turns, actor_overlays) ||
      !protected_action_invariant_holds(actor_plan.actor)) {
    actor_overlays = std::move(before);
    actor_overlay_losses = std::move(losses_before);
    actor_overlay_sources = std::move(sources_before);
    return false;
  }
  ++audit_.local_rewrites;
  for (const auto& displaced : displaced_actions)
    if (displaced.op == Op::Move) {
      ++audit_.delayed_move_rewrites;
      break;
    }
  return true;
}

bool Compiler::externalize_blocked_source(int index, int turn,
                                          std::uint64_t source_id) {
  if (source_id == 0 || !is_protected_production(
          effective_action(index, turn).op))
    return false;
  int sink = -1;
  for (int candidate = turn + 1; candidate < day_plan_->turns; ++candidate)
    if (is_sink(index, candidate)) {
      sink = candidate;
      break;
    }
  if (sink < 0) return false;

  auto& actions = overlays_[static_cast<std::size_t>(index)];
  auto& losses = overlay_sink_losses_[static_cast<std::size_t>(index)];
  auto& sources = overlay_source_ids_[static_cast<std::size_t>(index)];
  const auto before_actions = actions;
  const auto before_losses = losses;
  const auto before_sources = sources;
  std::vector<Action> shifted_actions;
  std::vector<int> shifted_losses;
  std::vector<std::uint64_t> shifted_sources;
  for (int slot = turn + 1; slot <= sink; ++slot) {
    shifted_actions.push_back(effective_action(index, slot));
    shifted_losses.push_back(effective_sink_loss(index, slot));
    shifted_sources.push_back(effective_source_id(index, slot));
  }
  for (int slot = turn; slot < sink; ++slot) {
    const auto offset = static_cast<std::size_t>(slot - turn);
    actions[static_cast<std::size_t>(slot)] = shifted_actions[offset];
    losses[static_cast<std::size_t>(slot)] = shifted_losses[offset];
    sources[static_cast<std::size_t>(slot)] = shifted_sources[offset];
  }
  actions[static_cast<std::size_t>(sink)] = Action{};
  losses[static_cast<std::size_t>(sink)] = 0;
  sources[static_cast<std::size_t>(sink)] = 0;
  const auto& actor_plan = day_plan_->actors[static_cast<std::size_t>(index)];
  if (!same_moves(actor_plan.turns, actions) ||
      !protected_action_invariant_holds(actor_plan.actor)) {
    actions = before_actions;
    losses = before_losses;
    sources = before_sources;
    return false;
  }
  ++audit_.source_debts_externalized;
  return true;
}

void Compiler::register_source_debt(Transaction& transaction,
                                    const Action& source,
                                    std::uint64_t source_id) {
  if (source_id == 0 || !is_protected_production(source.op)) return;
  if (!transaction.source_debts.empty()) {
    coalesced_current_day_sources_[source_id] = true;
    ++audit_.source_intents_coalesced;
    return;
  }
  transaction.source_debts.emplace(source_id, source);
  transaction.view.outstanding_source_actions = 1;
  ++audit_.source_debts_created;
}

std::uint64_t Compiler::source_for_required(
    const Transaction& transaction, const Action& required) const {
  for (const auto& [source_id, source] : transaction.source_debts)
    if (same_source_effect(source, required)) return source_id;
  return 0;
}

Decision Compiler::decide(const TurnInput& input) {
  if (!day_plan_ || input.day != day_plan_->day || input.turn < 0 ||
      input.turn >= day_plan_->turns)
    throw std::invalid_argument("turn input does not match installed day");

  if (!input.composed_baseline.empty() &&
      input.composed_baseline.size() > day_plan_->actors.size())
    throw std::invalid_argument("composed baseline has unknown actors");

  Decision decision;
  decision.id = next_decision_id_++;
  decision.day = input.day;
  decision.turn = input.turn;
  const std::size_t actor_count = input.composed_baseline.empty()
      ? day_plan_->actors.size() : input.composed_baseline.size();
  for (std::size_t index = 0; index < actor_count; ++index) {
    const auto& actor_plan = day_plan_->actors[index];
    const auto baseline =
        actor_plan.turns[static_cast<std::size_t>(input.turn)].action;
    const auto composed = input.composed_baseline.empty()
        ? baseline : input.composed_baseline[index];
    decision.actor_ids.push_back(actor_plan.actor);
    decision.baseline.push_back(composed);
    decision.actions.push_back(composed);

    if (!config_.enabled) continue;
    const auto* observation = observation_for(input, actor_plan.actor);
    if (observation == nullptr) continue;

    const auto scheduled = overlays_[index][static_cast<std::size_t>(input.turn)];
    const auto effective = scheduled.has_value() ? *scheduled : baseline;
    // A dynamic native owner changed this actor after the source plan was
    // selected.  Preserve it byte-for-byte and leave any local debt open.
    const bool local_owns_slot = composed == baseline || composed == effective;
    if (!local_owns_slot) continue;
    const auto effective_source =
        effective_source_id(static_cast<int>(index), input.turn);
    decision.actions[index] = effective;
    const TransactionKey key{actor_plan.actor, observation->position,
                             observation->actor_generation};
    GoalKind detected_goal{GoalKind::Crop};
    int detected_item{};
    const bool effective_is_blocked =
        action_is_blocked(effective, *observation, detected_goal,
                          detected_item);
    Transaction* transaction = effective_is_blocked
        ? &ensure_transaction(key, detected_goal, detected_item,
                              input.day, input.turn)
        : find_plot_transaction(observation->position);

    if (transaction == nullptr || transaction->view.completed) continue;
    if (effective_is_blocked)
      register_source_debt(*transaction, effective, effective_source);
    if (!observation->tile_effect_eligible) continue;

    const auto required = required_action(*transaction, *observation);
    if (!required.has_value()) {
      const bool objective_invalidated =
          transaction->view.goal == GoalKind::Crop &&
          (observation->tile.kind == TileKind::Other ||
           (observation->tile.kind == TileKind::Crop &&
            transaction->view.desired_item != -1 &&
            observation->tile.item != transaction->view.desired_item));
      if (effective_is_blocked) {
        if (externalize_blocked_source(static_cast<int>(index), input.turn,
                                       effective_source)) {
          decision.actions[index] =
              effective_action(static_cast<int>(index), input.turn);
          if (objective_invalidated) {
            transaction->view.completed = true;
            transaction->view.cancelled = true;
            transaction->view.outstanding_source_actions = 0;
            transaction->source_debts.clear();
            ++audit_.objectives_cancelled;
          }
        } else {
          decision.fail_closed_actors.push_back(actor_plan.actor);
          ++audit_.fail_closed;
        }
      } else if (objective_invalidated) {
        transaction->view.completed = true;
        transaction->view.cancelled = true;
        transaction->view.outstanding_source_actions = 0;
        transaction->source_debts.clear();
        ++audit_.objectives_cancelled;
      }
      continue;
    }
    // Plot debt may lease only this exact execution slot. MOVE and unrelated
    // production/critical actions are never displaced or shifted. A blocked
    // production source is already durable debt and can be state-recompiled
    // in place; a PASS is explicit slack.
    if (effective.op == Op::Move ||
        (!effective_is_blocked && effective.op != Op::Pass &&
         effective != *required))
      continue;
    std::uint64_t required_source =
        source_for_required(*transaction, *required);

    if (effective == *required) {
      if (transaction->source_debts.contains(effective_source))
        required_source = effective_source;
      decision.bindings.push_back(
          {transaction->view.id, required_source, actor_plan.actor,
           observation->actor_generation,
           observation->position, *required, observation->tile,
           *required != baseline});
      continue;
    }

    if (effective.op == Op::Pass || effective_is_blocked) {
      overlays_[index][static_cast<std::size_t>(input.turn)] = *required;
      overlay_sink_losses_[index][static_cast<std::size_t>(input.turn)] =
          std::numeric_limits<int>::max();
      overlay_source_ids_[index][static_cast<std::size_t>(input.turn)] =
          required_source;
      decision.actions[index] = *required;
      decision.bindings.push_back(
          {transaction->view.id, required_source, actor_plan.actor,
           observation->actor_generation,
           observation->position, *required, observation->tile, true});
      if (effective.op == Op::Pass) ++audit_.direct_slack_uses;
      else ++audit_.state_recompiles;
      continue;
    }
    decision.fail_closed_actors.push_back(actor_plan.actor);
    ++audit_.fail_closed;
  }
  return decision;
}

bool Compiler::commit(const Decision& decision,
                      const std::vector<Action>& final_actions) {
  if (final_actions != decision.actions ||
      final_actions.size() != decision.actor_ids.size())
    return false;
  for (const auto& binding : decision.bindings)
    if (pending_by_actor_.contains(
            {binding.actor, binding.actor_generation}))
      return false;
  std::set<std::uint64_t> leased_transactions;
  for (const auto& binding : decision.bindings)
    if (!leased_transactions.insert(binding.transaction_id).second)
      return false;
  for (const auto& binding : decision.bindings)
    pending_by_actor_.emplace(
                              std::pair{binding.actor,
                                        binding.actor_generation},
                              Pending{decision.id, binding});
  audit_.plot_leases += static_cast<int>(decision.bindings.size());
  return true;
}

bool Compiler::action_confirms(const Pending& pending,
                               const Receipt& receipt) const {
  const auto& action = pending.binding.action;
  const auto& before = pending.binding.before;
  if (receipt.generic_effect_confirmed) return true;
  switch (action.op) {
    case Op::Dig:
      return before.kind == TileKind::Weed &&
             receipt.after.kind == TileKind::Empty;
    case Op::Plant:
      return before.kind == TileKind::Empty &&
             receipt.after.kind == TileKind::Crop &&
             (action.item == -1 || receipt.after.item == action.item);
    case Op::Build:
      return receipt.after.kind == TileKind::Structure;
    case Op::Water:
      return before.kind == TileKind::Crop &&
             (receipt.after.watered_today ||
              receipt.day_end_water_effect_lower_bound);
    case Op::Harvest:
      return before.kind == TileKind::Crop && before.harvest_legal &&
             (receipt.desired_item_inventory_delta > 0 ||
              receipt.after.kind == TileKind::Empty);
    default:
      return false;
  }
}

bool Compiler::observe(const Receipt& receipt) {
  const auto found = pending_by_actor_.find(
      {receipt.actor, receipt.actor_generation});
  if (found == pending_by_actor_.end() ||
      found->second.decision_id != receipt.decision_id ||
      !(found->second.binding.tile == receipt.tile))
    return false;
  const Pending pending = found->second;
  pending_by_actor_.erase(found);

  auto transaction_it = std::find_if(
      transactions_.begin(), transactions_.end(), [&](const auto& entry) {
        return entry.second.view.id == pending.binding.transaction_id;
      });
  if (transaction_it == transactions_.end()) return false;
  auto& transaction_state = transaction_it->second;
  auto& transaction = transaction_state.view;
  if (!action_confirms(pending, receipt)) {
    ++transaction.failed_receipts;
    ++audit_.receipts_failed;
    return false;
  }

  ++transaction.confirmed_actions;
  ++audit_.receipts_confirmed;
  if (pending.binding.action.op == Op::Harvest)
    ++transaction.confirmed_harvests;
  const bool lifecycle_complete =
      (transaction.goal == GoalKind::Structure &&
       pending.binding.action.op == Op::Build) ||
      (transaction.goal == GoalKind::Crop &&
       transaction.confirmed_harvests >= config_.crop_harvests_to_close);
  if (lifecycle_complete) {
    if (!transaction_state.source_debts.empty()) {
      transaction_state.source_debts.clear();
      transaction.outstanding_source_actions = 0;
      ++audit_.source_debts_confirmed;
    }
    transaction.completed = true;
    ++audit_.completed_transactions;
  }
  return true;
}

std::vector<TransactionView> Compiler::open_transactions() const {
  std::vector<TransactionView> result;
  for (const auto& [key, transaction] : transactions_) {
    (void)key;
    if (!transaction.view.completed) result.push_back(transaction.view);
  }
  return result;
}

bool Compiler::movement_invariant_holds(int actor) const {
  const int index = actor_index(actor);
  if (index < 0) return false;
  return same_moves(day_plan_->actors[static_cast<std::size_t>(index)].turns,
                    overlays_[static_cast<std::size_t>(index)]);
}

bool Compiler::protected_action_invariant_holds(int actor) const {
  const int index = actor_index(actor);
  if (index < 0) return false;
  const auto& plan = day_plan_->actors[static_cast<std::size_t>(index)].turns;
  std::vector<std::uint64_t> expected;
  for (std::size_t turn = 0; turn < plan.size(); ++turn) {
    if (!is_protected_production(plan[turn].action.op)) continue;
    const auto source = baseline_source_ids_[static_cast<std::size_t>(index)][turn];
    bool externalized = false;
    for (const auto& [key, transaction] : transactions_) {
      (void)key;
      if (transaction.source_debts.contains(source)) {
        externalized = true;
        break;
      }
    }
    if (coalesced_current_day_sources_.contains(source)) externalized = true;
    if (!externalized) expected.push_back(source);
  }
  std::size_t next = 0;
  for (int turn = 0; turn < day_plan_->turns && next < expected.size(); ++turn)
    if (effective_source_id(index, turn) == expected[next]) ++next;
  return next == expected.size();
}

const char* op_name(Op operation) noexcept {
  switch (operation) {
    case Op::Pass: return "PASS";
    case Op::Move: return "MOVE";
    case Op::Dig: return "DIG";
    case Op::Plant: return "PLANT";
    case Op::Build: return "BUILD";
    case Op::Water: return "WATER";
    case Op::Harvest: return "HARVEST";
    case Op::Other: return "OTHER";
  }
  return "UNKNOWN";
}

}  // namespace g001::event_local_repair
