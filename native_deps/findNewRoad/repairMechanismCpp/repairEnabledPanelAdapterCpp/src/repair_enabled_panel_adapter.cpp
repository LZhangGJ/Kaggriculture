#include "repair_enabled_panel_adapter.hpp"

#include "day_scheduler_transactional_composer_bridge.hpp"
#include "g001_day_start_obligation_issuer.hpp"
#include "minimum_damage_remaining_day_audit_adapter.hpp"
#include "native_final_action_commit.hpp"
#include "purchase_failure_day_rolling_owner.hpp"
#include "repair.hpp"
#include "repair_fork_evaluator.hpp"
#include "route_loader.hpp"
#include "transactional_crop_repair_owner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>

namespace g001::repair_enabled_panel_adapter {

bool detail::disrupted_crop_source(
    fastkag::Action source, fastkag::Item desired,
    const fastkag::Tile& tile, int source_step, int current_step) {
  if (source.op == fastkag::Op::PLANT)
    return (source_step < current_step &&
            tile.kind == fastkag::TileKind::EMPTY) ||
           tile.kind == fastkag::TileKind::WEED ||
           (tile.kind == fastkag::TileKind::PLANT && tile.crop != desired);
  return source.op == fastkag::Op::WATER &&
         (tile.kind != fastkag::TileKind::PLANT || tile.crop != desired);
}

bool detail::crop_recovery_fits_pass_capacity(
    fastkag::Action source, fastkag::Item desired,
    const fastkag::Tile& tile, int source_step, int current_step,
    int remaining_raw_pass_slots) {
  if (remaining_raw_pass_slots < 0) return false;
  if (!disrupted_crop_source(source, desired, tile, source_step, current_step))
    return true;
  const int recovery_actions = source.op == fastkag::Op::PLANT
      ? 1 + (tile.kind != fastkag::TileKind::EMPTY)
      : source.op == fastkag::Op::WATER
          ? 2 + (tile.kind != fastkag::TileKind::EMPTY)
          : 0;
  if (recovery_actions == 0) return false;
  const int reusable_source_slot = source_step >= current_step;
  return recovery_actions - reusable_source_slot <= remaining_raw_pass_slots;
}

namespace {

namespace panel = repair_whole_game_panel;
namespace issuer = day_start_issuer;
namespace rolling = minimum_damage_remaining_audit;
namespace purchase = purchase_failure_rolling;
namespace bridge = minimum_damage_bridge;
namespace commit = native_final_commit;
namespace transactional_bridge = day_scheduler_transactional_bridge;
namespace transactional_crop = transactional_crop_repair;
namespace transactional_owner = repair_owner_composer;
using fastkag::Action;
using fastkag::NativeAgentState;
using fastkag::NativeTeammateExecutor;
using fastkag::PlayerAction;
using fastkag::Simulator;

bool same(Action lhs, Action rhs) {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool same(const std::vector<Action>& lhs, const std::vector<Action>& rhs) {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                    [](Action left, Action right) { return same(left, right); });
}

bool same(const PlayerAction& lhs, const PlayerAction& rhs) {
  return same(lhs.units, rhs.units) && same(lhs.market, rhs.market);
}

bool same(const fastkag::Tile& a, const fastkag::Tile& b) {
  return a.kind == b.kind && a.crop == b.crop && a.animal == b.animal &&
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
}

bool move(fastkag::Op op) {
  return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
         op == fastkag::Op::EAST || op == fastkag::Op::WEST;
}

int move_mismatches(const PlayerAction& lhs, const PlayerAction& rhs) {
  const auto common = std::min(lhs.units.size(), rhs.units.size());
  int mismatches = static_cast<int>(
      std::max(lhs.units.size(), rhs.units.size()) - common);
  for (std::size_t actor = 0; actor < common; ++actor)
    mismatches += (move(lhs.units[actor].op) || move(rhs.units[actor].op)) &&
                  !same(lhs.units[actor], rhs.units[actor]);
  return mismatches;
}

int common_move_mismatches(const PlayerAction& lhs, const PlayerAction& rhs) {
  const auto common = std::min(lhs.units.size(), rhs.units.size());
  int mismatches = 0;
  for (std::size_t actor = 0; actor < common; ++actor)
    mismatches += (move(lhs.units[actor].op) || move(rhs.units[actor].op)) &&
                  !same(lhs.units[actor], rhs.units[actor]);
  return mismatches;
}

const fastkag::Tile* tile_at(const Simulator& state, int player,
                             fastkag::Position position) {
  const int size = state.config().board_size;
  if (player < 0 || player >= 2 || position.x < 0 || position.y < 0 ||
      position.x >= size || position.y >= size)
    return nullptr;
  const auto index = static_cast<std::size_t>(position.y * size + position.x);
  const auto& tiles = state.farms()[static_cast<std::size_t>(player)].tiles;
  return index < tiles.size() ? &tiles[index] : nullptr;
}

std::uint64_t generation(std::uint64_t seed, int day) {
  return (seed << 20) | (2ULL << 16) |
         static_cast<std::uint64_t>(day + 1);
}

panel::MoneyMetrics money(const Simulator& env, int seat) {
  const double own = env.farms()[seat].money;
  const double opponent = env.farms()[1 - seat].money;
  return {own, opponent, own - opponent,
          own > opponent ? 1.0 : own < opponent ? 0.0 : 0.5};
}

purchase::Config purchase_config(bool enabled) {
  purchase::Config config;
  config.enabled = enabled;
  return config;
}

void record_lineage(const Simulator& observation, int player,
                    const PlayerAction& action,
                    issuer::PersistentRouteIntentRegistry& registry) {
  std::vector<fastkag::Position> positions{
      observation.farms()[player].farmer};
  positions.insert(positions.end(), observation.farms()[player].hands.begin(),
                   observation.farms()[player].hands.end());
  for (std::size_t actor = 0;
       actor < positions.size() && actor < action.units.size(); ++actor)
    (void)registry.record_exact_source(
        static_cast<int>(actor), observation.step_count(), positions[actor],
        action.units[actor]);
}

const Action* source_action(
    const issuer::IssueResult& issued,
    const obligation_day::ProductionObligation& obligation) {
  const auto found = std::find_if(
      issued.raw_sources.begin(), issued.raw_sources.end(),
      [&](const auto& source) {
        return source.actor == obligation.actor &&
               source.source_step == obligation.source_step;
      });
  return found == issued.raw_sources.end() ? nullptr : &found->action;
}

bool disrupted_stateful_goal(
    const issuer::IssueResult& issued,
    const obligation_day::ProductionObligation& obligation,
    const fastkag::Tile& tile, int current_step) {
  if (obligation.goal == obligation_day::GoalKind::BuildPasture)
    return tile.kind == fastkag::TileKind::WEED;
  if (obligation.goal != obligation_day::GoalKind::CropReady) return false;
  const auto* source = source_action(issued, obligation);
  if (!source) return false;
  return detail::disrupted_crop_source(*source, obligation.item, tile,
                                       obligation.source_step, current_step);
}

bool inject_tile_weed(Simulator& env, int player,
                      fastkag::Position position) {
  auto& farms = const_cast<std::array<fastkag::Farm, 2>&>(env.farms());
  auto& tiles = farms[static_cast<std::size_t>(player)].tiles;
  const auto index = static_cast<std::size_t>(
      position.y * env.config().board_size + position.x);
  if (position.x < 0 || position.y < 0 ||
      position.x >= env.config().board_size ||
      position.y >= env.config().board_size || index >= tiles.size())
    return false;
  tiles[index] = {};
  tiles[index].kind = fastkag::TileKind::WEED;
  return true;
}

bool inject_frozen_weed(Simulator& env, int player, int step) {
  if (step != 168) return false;
  constexpr fastkag::Position target{6, 1};
  const auto* tile = tile_at(env, player, target);
  return tile && tile->kind == fastkag::TileKind::EMPTY &&
         inject_tile_weed(env, player, target);
}

class Callback {
 public:
  struct CropInjection {
    fastkag::Position tile{};
    int step{-1};
    bool weed{true};
  };

  Callback(const NativeTeammateExecutor& executor, std::uint64_t seed,
           int seat, Metrics& metrics,
           bool inject_commit_failure, bool purchase_enabled,
           bool force_crop_completion = false)
      : seed_(seed), seat_(seat), metrics_(&metrics),
        purchase_owner_(executor, seat, 0,
                        purchase_config(purchase_enabled)),
        inject_commit_failure_(inject_commit_failure),
        force_crop_completion_(force_crop_completion) {}

  panel::FinalActionDecision operator()(
      const panel::FinalActionContext& context) {
    const auto start = std::chrono::steady_clock::now();
    panel::FinalActionDecision decision;
    decision.final_action = context.provider_action;
    decision.committed_provider_state = context.provider_state_proposal;

    if (!purchase_owner_.observe(context.observation))
      throw std::runtime_error("combined purchase receipt rejected");
    sync_purchase_metrics();
    const auto purchase_proposal = purchase_owner_.propose(context.observation);
    if (!same(purchase_proposal.base_action, context.provider_action))
      throw std::runtime_error("purchase/native provider state diverged");

    ensure_day(context);
    const auto weed = day_adapter_ ? prepare_weed(context, decision)
                                   : std::nullopt;
    bool weed_selected = weed.has_value();
    const bool purchase_changed = !same(purchase_proposal.base_action,
                                        purchase_proposal.final_action);
    auto final_action = purchase_proposal.final_action;
    if (weed)
      final_action.units[static_cast<std::size_t>(owned_actor_)] =
          weed->hand.candidate_unit;

    std::optional<rolling::Adapter> prepared_weed_adapter;
    std::optional<std::uint64_t> prepared_weed_fingerprint;
    if (weed_selected) {
      auto shadow = context.observation;
      auto joint = context.provider_joint;
      joint[static_cast<std::size_t>(context.player)] = final_action;
      shadow.step(joint);
      auto candidate_adapter = weed->adapter;
      if (!candidate_adapter.observe_final(
              context.observation,
              final_action.units[static_cast<std::size_t>(owned_actor_)],
              shadow)) {
        fallback(decision);
        weed_selected = false;
        final_action = purchase_proposal.final_action;
      } else {
        prepared_weed_adapter = std::move(candidate_adapter);
        prepared_weed_fingerprint =
            repair_fork::full_unit_phase_state_fingerprint(shadow);
      }
    }

    auto committed_state = context.provider_state_proposal;
    if (weed_selected || purchase_changed) {
      fastkag::NativeRepairOptions repair;
      repair.weed_obligation_day_owner = weed_selected;
      commit::Request request{&context.executor, &context.observation,
                              context.player, context.route, repair,
                              std::nullopt};
      const auto native_proposal = commit::propose(
          request, context.provider_state_before);
      bool valid = native_proposal.action.units.size() ==
                       context.provider_action.units.size() &&
                   final_action.market.size() >=
                       native_proposal.action.market.size();
      for (std::size_t slot = 0;
           valid && slot < native_proposal.action.market.size(); ++slot)
        valid = same(native_proposal.action.market[slot],
                     final_action.market[slot]);
      for (std::size_t actor = 0;
           valid && actor < native_proposal.action.units.size(); ++actor)
        if (!weed_selected || actor != static_cast<std::size_t>(owned_actor_))
          valid = same(native_proposal.action.units[actor],
                       final_action.units[actor]);
      if (!valid) {
        ++metrics_->market_exact_failures;
        throw std::runtime_error("combined repair changed an unowned slot");
      }
      const auto binding = commit::bind_repair_final_action(
          native_proposal,
          weed_selected ? (1ULL << static_cast<unsigned>(owned_actor_)) : 0ULL,
          final_action.market.size() > native_proposal.action.market.size(),
          purchase_proposal.generation,
          purchase_proposal.binding_hash ^
              (weed_selected ? weed->hand.certificate_hash : 0ULL),
          final_action);
      if (inject_commit_failure_ && context.step == 168)
        request.repair_options.weed_min_loss_realign = true;
      committed_state = context.provider_state_before;
      const auto committed = commit::commit_repair_owner_finalized(
          request, native_proposal, binding, final_action, committed_state);
      if (!committed.committed ||
          !commit::same_action(committed.replayed_action, final_action)) {
        ++metrics_->commit_failures;
        if (purchase_changed)
          throw std::runtime_error("combined purchase commit failed");
        fallback(decision);
        weed_selected = false;
        final_action = context.provider_action;
        committed_state = context.provider_state_proposal;
      }
    }

    auto staged_purchase = purchase_owner_;
    if (staged_purchase.finalize_composed(
            purchase_proposal, final_action, committed_state) !=
        purchase::FinalizeStatus::Selected) {
      ++metrics_->purchase_finalize_failures;
      throw std::runtime_error("combined purchase owner rejected final action");
    }

    if (weed_selected) {
      *day_adapter_ = std::move(*prepared_weed_adapter);
      expected_after_ = *prepared_weed_fingerprint;
      ++metrics_->commits;
      if (purchase_changed) ++metrics_->joint_weed_purchase_commits;
      if (const auto* before = tile_at(context.observation, context.player,
                                       trigger_tile_)) {
        pending_trigger_receipt_ = true;
        pending_trigger_before_kind_ = before->kind;
        pending_trigger_op_ =
            final_action.units[static_cast<std::size_t>(owned_actor_)].op;
      }
      decision.evidence.receipt_checks = 1;
      decision.evidence.receipts_accepted = 1;
      if (context.observation.hour() == 23) close_day(&decision.evidence);
    }
    if (weed_selected || purchase_changed) ++metrics_->provider_state_commits;
    purchase_owner_ = std::move(staged_purchase);
    decision.final_action = std::move(final_action);
    decision.committed_provider_state = std::move(committed_state);
    if (!same(decision.final_action, context.provider_action)) {
      ++metrics_->action_divergences;
      metrics_->action_divergence_steps.push_back(context.step);
      const auto actor = owned_actor_ >= 0
          ? static_cast<std::size_t>(owned_actor_) : 0U;
      metrics_->provider_unit_ops.push_back(
          actor < context.provider_action.units.size()
              ? static_cast<int>(context.provider_action.units[actor].op) : -1);
      metrics_->repair_unit_ops.push_back(
          actor < decision.final_action.units.size()
              ? static_cast<int>(decision.final_action.units[actor].op) : -1);
    }
    record_lineage(context.observation, context.player, decision.final_action,
                   lineage_);
    pending_history_ = HistoryReceipt{context.observation, std::nullopt,
                                      decision.final_action};
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    if (day_causally_activated_) {
      timings_.push_back(elapsed);
      metrics_->hand_us_total += elapsed;
    }
    return decision;
  }

  void after_step(const Simulator& actual) {
    if (pending_history_) {
      pending_history_->after = actual;
      history_.push_back(std::move(*pending_history_));
      pending_history_.reset();
    }
    if (actual.done()) {
      if (!purchase_owner_.observe(actual))
        throw std::runtime_error("terminal purchase receipt rejected");
      sync_purchase_metrics();
    }
    if (expected_after_) {
      ++metrics_->receipt_checks;
      const bool exact = *expected_after_ ==
          repair_fork::full_unit_phase_state_fingerprint(actual);
      metrics_->receipts_accepted += exact;
      metrics_->receipts_failed += !exact;
      if (!exact) ++metrics_->provider_state_leaks;
      expected_after_.reset();
    }
    if (pending_trigger_receipt_) {
      const auto* after = tile_at(actual, seat_, trigger_tile_);
      if (after) {
        metrics_->trigger_final_tile_kind = static_cast<int>(after->kind);
        if (after->kind != pending_trigger_before_kind_) {
          metrics_->trigger_transition_steps.push_back(actual.step_count() - 1);
          metrics_->trigger_transition_ops.push_back(
              static_cast<int>(pending_trigger_op_));
          metrics_->trigger_transition_before_kinds.push_back(
              static_cast<int>(pending_trigger_before_kind_));
          metrics_->trigger_transition_after_kinds.push_back(
              static_cast<int>(after->kind));
        }
        const bool completed =
            (trigger_goal_ == obligation_day::GoalKind::BuildPasture &&
             after->kind == fastkag::TileKind::PASTURE) ||
            (trigger_goal_ == obligation_day::GoalKind::CropReady &&
             after->kind == fastkag::TileKind::PLANT &&
             after->crop == trigger_item_ && after->watered_today);
        metrics_->trigger_effect_observed =
            metrics_->trigger_effect_observed || completed;
      }
      pending_trigger_receipt_ = false;
    }
  }

  void finish() {
    close_day();
    if (!timings_.empty()) {
      std::sort(timings_.begin(), timings_.end());
      metrics_->hand_us_min = timings_.front();
      metrics_->hand_us_median = timings_[timings_.size() / 2];
      metrics_->hand_us_max = timings_.back();
    }
  }

  [[nodiscard]] std::optional<CropInjection> crop_injection_target(
      const Simulator& observation,
      const NativeTeammateExecutor& executor) const {
    if (observation.hour() != 0) return std::nullopt;
    const auto issued = issuer::issue_day_start(
        {&observation, &executor.route_tape(0), seat_,
         generation(seed_, observation.day()), &lineage_, {}, {}});
    for (const auto& obligation : issued.obligations) {
      const auto* source = source_action(issued, obligation);
      const auto* tile = tile_at(observation, seat_, obligation.tile);
      if (obligation.goal == obligation_day::GoalKind::CropReady && source &&
          source->op == fastkag::Op::WATER && tile &&
          obligation.source_step + 2 <=
              observation.step_count() +
                  observation.config().turns_per_day - 1 &&
          tile->kind == fastkag::TileKind::PLANT &&
          tile->crop == obligation.item)
        return CropInjection{obligation.tile, obligation.source_step, true};
    }
    return std::nullopt;
  }

 private:
  void ensure_day(const panel::FinalActionContext& context) {
    if (active_day_ != context.observation.day()) {
      close_day();
      active_day_ = context.observation.day();
      day_causally_activated_ = false;
      history_.clear();
      issued_ = issuer::issue_day_start(
          {&context.observation, &context.executor.route_tape(context.route),
           context.player, generation(seed_, active_day_), &lineage_, {}, {}});
    }
    if (day_adapter_ || day_causally_activated_) return;
    if (!issued_.issued()) return;
    const obligation_day::ProductionObligation* trigger = nullptr;
    for (const auto& obligation : issued_.obligations) {
      const auto* tile = tile_at(context.observation, context.player,
                                 obligation.tile);
      if (obligation.actor >= 0 && tile &&
          disrupted_stateful_goal(issued_, obligation, *tile,
                                   context.observation.step_count())) {
        trigger = &obligation;
        break;
      }
    }
    // Causal activation is observation-only. Merely issuing a day certificate,
    // knowing the seed/day, or seeing a future route BUILD is insufficient.
    if (!trigger) return;
    day_causally_activated_ = true;
    ++metrics_->causal_activations;
    metrics_->trigger_obligation_id = trigger->id;
    metrics_->trigger_source_step = trigger->source_step;
    metrics_->trigger_x = trigger->tile.x;
    metrics_->trigger_y = trigger->tile.y;
    const auto* trigger_state = tile_at(context.observation, context.player,
                                        trigger->tile);
    metrics_->trigger_observed_weed =
        trigger_state && trigger_state->kind == fastkag::TileKind::WEED;
    metrics_->crop_repair_activations +=
        trigger->goal == obligation_day::GoalKind::CropReady;
    owned_actor_ = trigger->actor;
    metrics_->owned_actor = owned_actor_;
    metrics_->trigger_goal = static_cast<int>(trigger->goal);
    metrics_->trigger_item = static_cast<int>(trigger->item);
    metrics_->trigger_day = context.observation.day();
    metrics_->trigger_remaining_slots =
        context.observation.config().turns_per_day - context.observation.hour();
    metrics_->trigger_own_money =
        context.observation.farms()[context.player].money;
    metrics_->trigger_opponent_money =
        context.observation.farms()[1 - context.player].money;
    const int product = static_cast<int>(trigger->item);
    if (product >= 0 && product < fastkag::N_PRODUCTS) {
      metrics_->trigger_market_price = context.observation.market().prices[
          static_cast<std::size_t>(product)];
      metrics_->trigger_market_inventory =
          context.observation.market().inventory[static_cast<std::size_t>(
              product)];
      if (product < fastkag::N_CROPS)
        metrics_->trigger_seed_inventory =
            context.observation.privates()[context.player].seeds[
                static_cast<std::size_t>(product)];
      metrics_->trigger_shop_demand = static_cast<int>(std::count(
          context.observation.shops().begin(),
          context.observation.shops().end(), product));
    }
    if (trigger_state) {
      metrics_->trigger_crop_yield = trigger_state->yield_units;
      metrics_->trigger_crop_age = context.observation.day() -
                                   trigger_state->planted_day;
      metrics_->trigger_unwatered = trigger_state->consecutive_unwatered;
    }
    metrics_->trigger_future_ops.assign(24, 0);
    for (const auto& source : issued_.raw_sources)
      if (source.actor == owned_actor_ &&
          source.source_step >= context.observation.step_count()) {
        const auto op = static_cast<int>(source.action.op);
        if (op >= 0 && op < 24)
          ++metrics_->trigger_future_ops[static_cast<std::size_t>(op)];
      }
    metrics_->trigger_final_tile_kind = trigger_state
        ? static_cast<int>(trigger_state->kind)
        : -1;
    trigger_tile_ = trigger->tile;
    trigger_goal_ = trigger->goal;
    trigger_item_ = trigger->item;
    const auto* trigger_source = source_action(issued_, *trigger);
    const int remaining_raw_pass_slots = static_cast<int>(std::count_if(
        issued_.raw_sources.begin(), issued_.raw_sources.end(),
        [&](const auto& source) {
          return source.actor == owned_actor_ &&
                 source.source_step >= context.observation.step_count() &&
                 source.action.op == fastkag::Op::PASS;
        }));
    if (force_crop_completion_ &&
        trigger_goal_ == obligation_day::GoalKind::CropReady &&
        (!trigger_source || !trigger_state ||
         !detail::crop_recovery_fits_pass_capacity(
             *trigger_source, trigger_item_, *trigger_state,
             trigger->source_step, context.observation.step_count(),
             remaining_raw_pass_slots)))
      return;
    if (force_crop_completion_ &&
        trigger_goal_ == obligation_day::GoalKind::CropReady)
      for (auto& obligation : issued_.obligations)
        if (obligation.actor == owned_actor_)
          obligation.must_finish_today = obligation.id == trigger->id;
    std::vector<bridge::ObligationPolicy> policies;
    for (const auto& obligation : issued_.obligations)
      if (obligation.actor == owned_actor_) {
        const auto* raw = source_action(issued_, obligation);
        const int value = raw ? repair::default_economic_value(raw->op) : 0;
        policies.push_back(
            {obligation.id, true,
             1 + (force_crop_completion_ ? value * 100 : 0) +
                 (active_day_ * 24 + 23 - obligation.source_step)});
      }
    day_adapter_ = std::make_unique<rolling::Adapter>(
        issued_, context.player, generation(seed_, active_day_),
        std::move(policies), owned_actor_);
    for (const auto& receipt : history_) {
      if (!receipt.after ||
          static_cast<std::size_t>(owned_actor_) >= receipt.action.units.size() ||
          !day_adapter_->observe_passthrough(
              receipt.before,
              receipt.action.units[static_cast<std::size_t>(owned_actor_)],
              *receipt.after)) {
        ++metrics_->fallbacks;
        day_adapter_.reset();
        return;
      }
    }
  }

  void fallback(panel::FinalActionDecision& decision) {
    ++metrics_->fallbacks;
    ++decision.evidence.failures;
    day_adapter_.reset();
  }

  struct WeedAttempt {
    rolling::Adapter adapter;
    rolling::HandAudit hand;
  };

  struct HistoryReceipt {
    Simulator before;
    std::optional<Simulator> after;
    PlayerAction action;
  };

  std::optional<WeedAttempt> prepare_weed(
      const panel::FinalActionContext& context,
      panel::FinalActionDecision& decision) {
    auto candidate_adapter = *day_adapter_;
    const auto hand = candidate_adapter.plan(context.observation);
    if (!hand.valid || context.provider_action.units.empty() ||
        (force_crop_completion_ && hand.signed_debts != 0)) {
      fallback(decision);
      return std::nullopt;
    }
    ++metrics_->triggers;
    ++metrics_->attempts;
    return WeedAttempt{std::move(candidate_adapter), hand};
  }

  void sync_purchase_metrics() {
    const auto& audit = purchase_owner_.metrics();
    metrics_->purchase_failures = audit.purchase_failures;
    metrics_->seed_purchase_failures = audit.seed_purchase_failures;
    metrics_->purchase_retries = audit.purchase_retries;
    metrics_->purchase_retry_fills = audit.purchase_retry_fills;
    metrics_->purchase_move_mismatches = audit.move_mismatch;
  }

  void close_day(panel::AdapterEvidence* evidence = nullptr) {
    if (!day_adapter_) return;
    const auto day = day_adapter_->finish();
    metrics_->moves_expected += day.initial_move_tokens;
    metrics_->moves_emitted += day.observed_move_receipts;
    metrics_->move_failures += day.duplicate_move_receipts +
        day.failed_move_receipts + day.move_early_violations +
        std::max(0, day.initial_move_tokens - day.observed_move_receipts);
    metrics_->move_early += day.move_early_violations;
    metrics_->move_duplicates += day.duplicate_move_receipts;
    metrics_->move_drops +=
        std::max(0, day.initial_move_tokens - day.observed_move_receipts);
    metrics_->terminal_debts +=
        static_cast<int>(day.terminal_debts.size());
    for (const auto& debt : day.terminal_debts) {
      metrics_->terminal_debt_ids.push_back(debt.obligation_id);
      const auto found = std::find_if(
          issued_.obligations.begin(), issued_.obligations.end(),
          [&](const auto& obligation) {
            return obligation.id == debt.obligation_id;
          });
      metrics_->terminal_debt_source_steps.push_back(
          found == issued_.obligations.end() ? -1 : found->source_step);
    }
    if (evidence) {
      evidence->moves_expected = day.initial_move_tokens;
      evidence->moves_emitted = day.observed_move_receipts;
      evidence->move_failures = day.duplicate_move_receipts +
          day.failed_move_receipts + day.move_early_violations +
          std::max(0, day.initial_move_tokens -
                          day.observed_move_receipts);
      evidence->debts_opened =
          static_cast<int>(day.terminal_debts.size());
    }
    day_adapter_.reset();
    owned_actor_ = -1;
  }

  std::uint64_t seed_{};
  int seat_{};
  Metrics* metrics_{};
  int active_day_{-1};
  int owned_actor_{-1};
  issuer::PersistentRouteIntentRegistry lineage_;
  issuer::IssueResult issued_;
  std::unique_ptr<rolling::Adapter> day_adapter_;
  purchase::Owner purchase_owner_;
  std::optional<std::uint64_t> expected_after_;
  std::vector<HistoryReceipt> history_;
  std::optional<HistoryReceipt> pending_history_;
  std::vector<long long> timings_;
  bool inject_commit_failure_{};
  bool force_crop_completion_{};
  bool day_causally_activated_{};
  fastkag::Position trigger_tile_{};
  obligation_day::GoalKind trigger_goal_{
      obligation_day::GoalKind::BuildPasture};
  fastkag::Item trigger_item_{fastkag::Item::NONE};
  bool pending_trigger_receipt_{};
  fastkag::TileKind pending_trigger_before_kind_{fastkag::TileKind::EMPTY};
  fastkag::Op pending_trigger_op_{fastkag::Op::PASS};
};

Metrics run_game(const NativeTeammateExecutor& executor, std::uint64_t seed,
                 int seat, panel::Scenario scenario,
                 bool inject_commit_failure = false,
                 std::optional<Callback::CropInjection> forced_crop = {},
                 bool force_crop_completion = false,
                 int opponent_route = 0) {
  fastkag::Config config;
  if (scenario != panel::Scenario::Normal)
    config.weed_spawn_chance = 0.0;
  Simulator baseline(config, seed);
  Simulator repair(config, seed);
  std::array<NativeAgentState, 2> baseline_states;
  std::array<NativeAgentState, 2> repair_states;
  Metrics metrics;
  metrics.seed = seed;
  metrics.seat = seat;
  metrics.scenario = scenario;
  Callback callback(executor, seed, seat, metrics, inject_commit_failure,
                    !inject_commit_failure &&
                        scenario != panel::Scenario::ForcedCropWeed,
                    force_crop_completion);
  constexpr int kLaneStride = 128;
  const int movement_lanes =
      (config.episode_steps / config.turns_per_day + 1) * kLaneStride;
  std::vector<std::vector<int>> baseline_movement(
      static_cast<std::size_t>(movement_lanes));
  std::vector<std::vector<int>> repair_movement(
      static_cast<std::size_t>(movement_lanes));
  std::vector<bool> baseline_lane_seen(static_cast<std::size_t>(movement_lanes));
  std::vector<bool> repair_lane_seen(static_cast<std::size_t>(movement_lanes));

  while (!baseline.done() && !repair.done()) {
    if (baseline.step_count() != repair.step_count())
      throw std::runtime_error("divergence-aware paired clocks differ");
    const int step = repair.step_count();
    if (scenario == panel::Scenario::ForcedWeed && step == 168) {
      ++metrics.forced_events;
      metrics.forced_baseline_applied += inject_frozen_weed(baseline, seat, step);
      metrics.forced_repair_applied += inject_frozen_weed(repair, seat, step);
    }
    if (scenario == panel::Scenario::ForcedCropWeed &&
        metrics.forced_events == 0 && step >= 24 && repair.hour() == 0 &&
        (!forced_crop || forced_crop->step == step)) {
      const auto target = forced_crop
          ? forced_crop : callback.crop_injection_target(repair, executor);
      const auto* baseline_tile = target
          ? tile_at(baseline, seat, target->tile) : nullptr;
      const auto* repair_tile = target
          ? tile_at(repair, seat, target->tile) : nullptr;
      if (baseline_tile && repair_tile &&
          baseline_tile->kind == fastkag::TileKind::PLANT &&
          repair_tile->kind == fastkag::TileKind::PLANT &&
          baseline_tile->crop == repair_tile->crop) {
        ++metrics.forced_events;
        metrics.forced_step = step;
        metrics.forced_x = target->tile.x;
        metrics.forced_y = target->tile.y;
        metrics.forced_kind = target->weed
            ? static_cast<int>(fastkag::TileKind::WEED)
            : static_cast<int>(fastkag::TileKind::EMPTY);
        if (target->weed) {
          metrics.forced_baseline_applied +=
              inject_tile_weed(baseline, seat, target->tile);
          metrics.forced_repair_applied +=
              inject_tile_weed(repair, seat, target->tile);
        } else {
          auto clear = [&](Simulator& env) {
            auto& farms = const_cast<std::array<fastkag::Farm, 2>&>(env.farms());
            auto& tile = farms[static_cast<std::size_t>(seat)].tiles[
                static_cast<std::size_t>(target->tile.y *
                    env.config().board_size + target->tile.x)];
            tile = {};
            return 1;
          };
          metrics.forced_baseline_applied += clear(baseline);
          metrics.forced_repair_applied += clear(repair);
        }
      }
    }

    std::array<PlayerAction, 2> baseline_actions;
    std::array<PlayerAction, 2> repair_actions;
    std::array<NativeAgentState, 2> repair_before = repair_states;
    for (int player = 0; player < 2; ++player) {
      const int route = player == seat ? 0 : opponent_route;
      baseline_actions[player] = executor.action_external(
          baseline, player, route, baseline_states[player]);
      repair_actions[player] = executor.action_external(
          repair, player, route, repair_states[player]);
    }
    const panel::FinalActionContext context{
        repair, executor, seat, 0, step, repair_actions,
        repair_actions[static_cast<std::size_t>(seat)],
        repair_before[static_cast<std::size_t>(seat)],
        repair_states[static_cast<std::size_t>(seat)]};
    auto decision = callback(context);
    if (decision.final_action.units.size() !=
            repair_actions[static_cast<std::size_t>(seat)].units.size() ||
        !decision.committed_provider_state)
      throw std::runtime_error("adapter returned an invalid transaction");
    repair_actions[static_cast<std::size_t>(seat)] = decision.final_action;
    repair_states[static_cast<std::size_t>(seat)] =
        std::move(*decision.committed_provider_state);

    if (metrics.first_legacy_trade_exposure_step < 0 &&
        same(baseline_actions[static_cast<std::size_t>(seat)].market,
             repair_actions[static_cast<std::size_t>(seat)].market)) {
      const auto baseline_unit = baseline.preview_unit_phase(baseline_actions);
      const auto repair_unit = repair.preview_unit_phase(repair_actions);
      const auto& baseline_private =
          baseline_unit.privates()[static_cast<std::size_t>(seat)];
      const auto& repair_private =
          repair_unit.privates()[static_cast<std::size_t>(seat)];
      for (int product = 0; product < fastkag::N_PRODUCTS; ++product) {
        int requested = 0;
        for (const auto& order :
             baseline_actions[static_cast<std::size_t>(seat)].market)
          if (order.op == fastkag::Op::SELL &&
              static_cast<int>(order.item) == product)
            requested += std::max(0, order.quantity);
        const auto index = static_cast<std::size_t>(product);
        const int baseline_shed = baseline_private.shed[index];
        const int repair_shed = repair_private.shed[index];
        const int incremental_fill =
            std::min(requested, repair_shed) -
            std::min(requested, baseline_shed);
        if (incremental_fill == 0) continue;
        metrics.first_legacy_trade_exposure_step = step;
        metrics.first_legacy_trade_exposure_product = product;
        metrics.first_legacy_trade_requested = requested;
        metrics.first_legacy_trade_baseline_shed = baseline_shed;
        metrics.first_legacy_trade_repair_shed = repair_shed;
        metrics.first_legacy_trade_incremental_fill = incremental_fill;
        metrics.first_legacy_trade_price = baseline.market().prices[index];
        const auto animals = [](const Simulator& state, int player) {
          return static_cast<int>(std::count_if(
              state.farms()[static_cast<std::size_t>(player)].tiles.begin(),
              state.farms()[static_cast<std::size_t>(player)].tiles.end(),
              [](const auto& tile) {
                return tile.kind == fastkag::TileKind::ANIMAL;
              }));
        };
        metrics.first_legacy_trade_baseline_animals = animals(baseline_unit, seat);
        metrics.first_legacy_trade_repair_animals = animals(repair_unit, seat);
        const auto asset_value = [&](const Simulator& state) {
          const auto& own = state.privates()[static_cast<std::size_t>(seat)];
          double value = state.farms()[static_cast<std::size_t>(seat)].money;
          for (int item = 0; item < fastkag::N_PRODUCTS; ++item) {
            int quantity = own.shed[static_cast<std::size_t>(item)];
            for (const auto& carried : own.inventories)
              quantity += carried[static_cast<std::size_t>(item)];
            value += quantity * baseline.market().prices[
                                    static_cast<std::size_t>(item)];
          }
          return value;
        };
        metrics.first_legacy_trade_baseline_asset_value =
            asset_value(baseline_unit);
        metrics.first_legacy_trade_repair_asset_value = asset_value(repair_unit);
        break;
      }
    }

    const int opponent = 1 - seat;
    if (metrics.first_focal_unit_action_divergence_step < 0 &&
        !same(baseline_actions[static_cast<std::size_t>(seat)].units,
              repair_actions[static_cast<std::size_t>(seat)].units))
      metrics.first_focal_unit_action_divergence_step = step;
    if (metrics.first_focal_market_action_divergence_step < 0 &&
        !same(baseline_actions[static_cast<std::size_t>(seat)].market,
              repair_actions[static_cast<std::size_t>(seat)].market))
      metrics.first_focal_market_action_divergence_step = step;
    if (metrics.first_opponent_action_divergence_step < 0 &&
        !same(baseline_actions[static_cast<std::size_t>(opponent)],
              repair_actions[static_cast<std::size_t>(opponent)]))
      metrics.first_opponent_action_divergence_step = step;

    const auto record_movement = [&](const PlayerAction& action,
                                     auto& lanes, auto& seen) {
      for (std::size_t actor = 0; actor < action.units.size(); ++actor) {
        const int lane = repair.day() * kLaneStride +
                         static_cast<int>(actor);
        if (lane < 0 || lane >= movement_lanes) continue;
        seen[static_cast<std::size_t>(lane)] = true;
        if (move(action.units[actor].op))
          lanes[static_cast<std::size_t>(lane)].push_back(
              static_cast<int>(action.units[actor].op));
      }
    };
    record_movement(baseline_actions[static_cast<std::size_t>(seat)],
                    baseline_movement, baseline_lane_seen);
    record_movement(repair_actions[static_cast<std::size_t>(seat)],
                    repair_movement, repair_lane_seen);

    metrics.paired_action_divergences += !same(
        baseline_actions[static_cast<std::size_t>(seat)],
        repair_actions[static_cast<std::size_t>(seat)]);
    const int paired_move_mismatch = move_mismatches(
        baseline_actions[static_cast<std::size_t>(seat)],
        repair_actions[static_cast<std::size_t>(seat)]);
    metrics.paired_move_mismatches += paired_move_mismatch;
    if (paired_move_mismatch)
      metrics.paired_move_mismatch_steps.push_back(step);
    metrics.paired_actor_count_mismatches +=
        baseline_actions[static_cast<std::size_t>(seat)].units.size() !=
        repair_actions[static_cast<std::size_t>(seat)].units.size();
    const int common_mismatch = common_move_mismatches(
        baseline_actions[static_cast<std::size_t>(seat)],
        repair_actions[static_cast<std::size_t>(seat)]);
    metrics.paired_common_move_mismatches += common_mismatch;
    if (!baseline_actions[static_cast<std::size_t>(seat)].units.empty() &&
        !repair_actions[static_cast<std::size_t>(seat)].units.empty()) {
      const auto baseline_farmer =
          baseline_actions[static_cast<std::size_t>(seat)].units.front();
      const auto repair_farmer =
          repair_actions[static_cast<std::size_t>(seat)].units.front();
      metrics.paired_farmer_move_mismatches +=
          (move(baseline_farmer.op) || move(repair_farmer.op)) &&
          !same(baseline_farmer, repair_farmer);
    }
    if (common_mismatch && metrics.first_common_move_mismatch_step < 0) {
      const auto common = std::min(
          baseline_actions[static_cast<std::size_t>(seat)].units.size(),
          repair_actions[static_cast<std::size_t>(seat)].units.size());
      for (std::size_t actor = 0; actor < common; ++actor) {
        const auto baseline_op = baseline_actions[static_cast<std::size_t>(seat)]
                                     .units[actor].op;
        const auto repair_op = repair_actions[static_cast<std::size_t>(seat)]
                                   .units[actor].op;
        if ((move(baseline_op) || move(repair_op)) &&
            !same(baseline_actions[static_cast<std::size_t>(seat)].units[actor],
                  repair_actions[static_cast<std::size_t>(seat)].units[actor])) {
          metrics.first_common_move_mismatch_step = step;
          metrics.first_common_move_mismatch_actor = static_cast<int>(actor);
          metrics.first_common_move_baseline_op = static_cast<int>(baseline_op);
          metrics.first_common_move_repair_op = static_cast<int>(repair_op);
          break;
        }
      }
    }
    auto route_action = executor.route_tape(0)[static_cast<std::size_t>(step)];
    route_action.units.resize(
        repair_actions[static_cast<std::size_t>(seat)].units.size());
    const int route_move_mismatch = move_mismatches(
        route_action, repair_actions[static_cast<std::size_t>(seat)]);
    metrics.route_move_mismatches += route_move_mismatch;
    if (route_move_mismatch)
      metrics.route_move_mismatch_steps.push_back(step);
    const int baseline_unit = fastkag::native_macro_unit_failures(
        baseline, seat, baseline_actions[static_cast<std::size_t>(seat)]);
    const int repair_unit = fastkag::native_macro_unit_failures(
        repair, seat, repair_actions[static_cast<std::size_t>(seat)]);
    metrics.baseline_unit_failures += baseline_unit;
    metrics.repair_unit_failures += repair_unit;
    if (baseline_unit) metrics.baseline_unit_failure_steps.push_back(step);
    if (repair_unit) metrics.repair_unit_failure_steps.push_back(step);
    baseline.step(baseline_actions);
    repair.step(repair_actions);
    if (metrics.first_state_divergence_step < 0 &&
        repair_fork::full_unit_phase_state_fingerprint(baseline) !=
            repair_fork::full_unit_phase_state_fingerprint(repair)) {
      metrics.first_state_divergence_step = step;
      for (const auto& action :
           baseline_actions[static_cast<std::size_t>(seat)].units)
        metrics.first_state_baseline_unit_ops.push_back(
            static_cast<int>(action.op));
      for (const auto& action :
           repair_actions[static_cast<std::size_t>(seat)].units)
        metrics.first_state_repair_unit_ops.push_back(
            static_cast<int>(action.op));
      const auto& baseline_tiles =
          baseline.farms()[static_cast<std::size_t>(seat)].tiles;
      const auto& repair_tiles =
          repair.farms()[static_cast<std::size_t>(seat)].tiles;
      for (std::size_t index = 0;
           index < std::min(baseline_tiles.size(), repair_tiles.size());
           ++index) {
        if (same(baseline_tiles[index], repair_tiles[index])) continue;
        metrics.first_tile_divergence_index = static_cast<int>(index);
        metrics.first_tile_baseline_kind =
            static_cast<int>(baseline_tiles[index].kind);
        metrics.first_tile_repair_kind =
            static_cast<int>(repair_tiles[index].kind);
        metrics.first_tile_baseline_animal =
            static_cast<int>(baseline_tiles[index].animal);
        metrics.first_tile_repair_animal =
            static_cast<int>(repair_tiles[index].animal);
        break;
      }
    }
    if (metrics.first_own_money_divergence_step < 0 &&
        baseline.farms()[static_cast<std::size_t>(seat)].money !=
            repair.farms()[static_cast<std::size_t>(seat)].money)
      metrics.first_own_money_divergence_step = step;
    if (metrics.first_opponent_money_divergence_step < 0 &&
        baseline.farms()[static_cast<std::size_t>(opponent)].money !=
            repair.farms()[static_cast<std::size_t>(opponent)].money)
      metrics.first_opponent_money_divergence_step = step;
    if (metrics.first_market_state_divergence_step < 0 &&
        (baseline.market().inventory != repair.market().inventory ||
         baseline.market().prices != repair.market().prices)) {
      metrics.first_market_state_divergence_step = step;
      for (int product = 0; product < fastkag::N_PRODUCTS; ++product) {
        const auto index = static_cast<std::size_t>(product);
        if (baseline.market().inventory[index] == repair.market().inventory[index] &&
            baseline.market().prices[index] == repair.market().prices[index])
          continue;
        metrics.first_market_divergence_product = product;
        metrics.first_market_baseline_inventory = baseline.market().inventory[index];
        metrics.first_market_repair_inventory = repair.market().inventory[index];
        metrics.first_market_baseline_price = baseline.market().prices[index];
        metrics.first_market_repair_price = repair.market().prices[index];
        break;
      }
      const auto record_orders = [](const std::vector<Action>& orders,
                                    std::vector<int>& output) {
        for (const auto& action : orders) {
          output.push_back(static_cast<int>(action.op));
          output.push_back(static_cast<int>(action.item));
          output.push_back(action.quantity);
        }
      };
      record_orders(baseline_actions[static_cast<std::size_t>(seat)].market,
                    metrics.first_market_focal_orders);
      record_orders(baseline_actions[static_cast<std::size_t>(opponent)].market,
                    metrics.first_market_opponent_orders);
      metrics.first_market_baseline_focal_fills =
          baseline.last_market_fills()[static_cast<std::size_t>(seat)];
      metrics.first_market_repair_focal_fills =
          repair.last_market_fills()[static_cast<std::size_t>(seat)];
      metrics.first_market_baseline_opponent_fills =
          baseline.last_market_fills()[static_cast<std::size_t>(opponent)];
      metrics.first_market_repair_opponent_fills =
          repair.last_market_fills()[static_cast<std::size_t>(opponent)];
    }
    callback.after_step(repair);
    const int baseline_market = fastkag::native_macro_market_failures(
        baseline, seat, baseline_actions[static_cast<std::size_t>(seat)]);
    const int repair_market = fastkag::native_macro_market_failures(
        repair, seat, repair_actions[static_cast<std::size_t>(seat)]);
    metrics.baseline_market_failures += baseline_market;
    metrics.repair_market_failures += repair_market;
    if (baseline_market) metrics.baseline_market_failure_steps.push_back(step);
    if (repair_market) metrics.repair_market_failure_steps.push_back(step);
    ++metrics.steps;
  }
  if (!baseline.done() || !repair.done())
    throw std::runtime_error("paired arm terminated early");
  callback.finish();
  for (int lane = 0; lane < movement_lanes; ++lane) {
    const auto index = static_cast<std::size_t>(lane);
    if (baseline_lane_seen[index] != repair_lane_seen[index]) {
      ++metrics.movement_lane_population_mismatches;
      metrics.movement_lane_population_mismatch_lanes.push_back(lane);
    } else if (baseline_lane_seen[index] &&
               baseline_movement[index] != repair_movement[index]) {
      ++metrics.movement_sequence_mismatches;
      metrics.movement_sequence_mismatch_lanes.push_back(lane);
    }
  }
  metrics.baseline_terminal = money(baseline, seat);
  metrics.repair_terminal = money(repair, seat);
  return metrics;
}

class PassiveTransactionalOwner final
    : public transactional_owner::TransactionalRepairOwner {
 public:
  std::string name() const override { return "crop_fixture_passive"; }

  transactional_owner::SettlementResult settle_owned(
      const repair_fork::RepairContext&,
      std::span<const repair_fork::ActionReceipt> actions,
      std::span<const repair_fork::PurchaseReceipt> purchases) override {
    return actions.empty() && purchases.empty()
               ? transactional_owner::SettlementResult::AppliedSuccess
               : transactional_owner::SettlementResult::ProtocolInvalid;
  }

  transactional_owner::SettlementResult validate_settle(
      const repair_fork::RepairContext&,
      std::span<const repair_fork::ActionReceipt> actions,
      std::span<const repair_fork::PurchaseReceipt> purchases) const override {
    return actions.empty() && purchases.empty()
               ? transactional_owner::SettlementResult::AppliedSuccess
               : transactional_owner::SettlementResult::ProtocolInvalid;
  }

  transactional_owner::PreparedRepair prepare(
      const repair_fork::RepairContext& context) const override {
    transactional_owner::PreparedRepair out;
    out.token = 1;
    out.units = context.raw_g001.units;
    for (std::size_t actor = 0; actor < out.units.size(); ++actor)
      out.sources.push_back({static_cast<int>(actor), context.step,
                             out.units[actor]});
    return out;
  }

  bool validate_commit(
      std::uint64_t,
      const transactional_owner::CommitGrant&) const override {
    return false;
  }
  bool commit(std::uint64_t,
              const transactional_owner::CommitGrant&) override {
    return false;
  }
  void abort(std::uint64_t) noexcept override {}
};

bool same(const repair_fork::ActionReceipt& left,
          const repair_fork::ActionReceipt& right) {
  return left.submitted_step == right.submitted_step &&
         left.actor == right.actor &&
         left.manifest_generation == right.manifest_generation &&
         left.prefix_manifest_hash == right.prefix_manifest_hash &&
         left.post_prefix_state_fingerprint ==
             right.post_prefix_state_fingerprint &&
         same(left.emitted, right.emitted);
}

bool same(const repair_fork::PurchaseReceipt& left,
          const repair_fork::PurchaseReceipt& right) {
  return left.debt_id == right.debt_id &&
         left.submitted_step == right.submitted_step &&
         left.operation == right.operation && left.item == right.item &&
         left.requested == right.requested && left.filled == right.filled &&
         left.market_slot == right.market_slot &&
         left.compile_status == right.compile_status;
}

std::vector<PlayerAction> crop_smoke_tape(int steps) {
  std::vector<PlayerAction> tape(static_cast<std::size_t>(steps));
  for (auto& action : tape) action.units.resize(1);
  tape[0].units[0] = {fastkag::Op::PLANT, fastkag::Item::WHEAT, 1};
  tape[1].units[0] = {fastkag::Op::WATER, fastkag::Item::NONE, 1};
  tape[24].units[0] = {fastkag::Op::WATER, fastkag::Item::NONE, 1};
  tape[25].units[0] = {fastkag::Op::EAST, fastkag::Item::NONE, 1};
  tape[26].units[0] = {fastkag::Op::WEST, fastkag::Item::NONE, 1};
  return tape;
}

class TransactionalCropCallback {
 public:
  TransactionalCropCallback(const std::vector<PlayerAction>& tape,
                            std::uint64_t seed,
                            TransactionalCropSmoke& metrics)
      : tape_(&tape), seed_(seed), metrics_(&metrics) {}

  void force_past_plant_weed(Simulator& observation) {
    if (observation.step_count() != 24 || metrics_->forced_events != 0)
      return;
    const auto tile = observation.farms()[0].farmer;
    const auto prior = lineage_.crop_at(tile);
    if (!prior || prior->item != fastkag::Item::WHEAT ||
        prior->exact_source.op != fastkag::Op::PLANT ||
        prior->source_step >= observation.step_count())
      throw std::runtime_error("crop smoke lacks exact past-PLANT lineage");
    metrics_->lineage_from_past_plant = true;
    auto& farms = const_cast<std::array<fastkag::Farm, 2>&>(
        observation.farms());
    auto& private_states = const_cast<std::array<fastkag::PrivateState, 2>&>(
        observation.privates());
    const auto index = static_cast<std::size_t>(
        tile.y * observation.config().board_size + tile.x);
    if (farms[0].tiles[index].kind != fastkag::TileKind::PLANT ||
        farms[0].tiles[index].crop != fastkag::Item::WHEAT)
      throw std::runtime_error("crop smoke source tile is not the past crop");
    farms[0].tiles[index] = {};
    farms[0].tiles[index].kind = fastkag::TileKind::WEED;
    // A zero-fill production purchase exercises debt receipts without making
    // the scheduler's unit-state certificate depend on a market transition.
    farms[0].money = 0.0;
    private_states[0].seeds[static_cast<std::size_t>(fastkag::Item::WHEAT)] =
        std::max(1, private_states[0].seeds[
                        static_cast<std::size_t>(fastkag::Item::WHEAT)]);
    target_ = tile;
    ++metrics_->forced_events;
  }

  panel::FinalActionDecision operator()(
      const panel::FinalActionContext& context) {
    panel::FinalActionDecision output;
    output.final_action = context.provider_action;
    output.committed_provider_state = context.provider_state_proposal;
    if (context.step == 24 && !owner_) start_day(context);
    if (!owner_) {
      record_lineage(context.observation, context.player,
                     output.final_action, lineage_);
      return output;
    }

    repair_fork::RepairContext repair_context{
        context.observation, context.player, context.step,
        context.provider_action, context.provider_joint,
        previous_purchase_receipts_, previous_action_receipts_};
    const transactional_owner::TypedRepairInput typed{
        issued_.crop_owner_obligations, {}};
    auto staged = owner_->prepare(repair_context, typed);
    if (staged.token == 0)
      throw std::runtime_error("crop bridge rejected a certified hand");
    check_acks(staged.decision, true);

    auto compiled = repair_fork::ExactMarketCompiler().compile(
        context.observation, context.player, context.provider_action.market,
        staged.decision.required_purchases);
    auto selected = context.provider_action;
    selected.units = staged.decision.units;
    selected.market = compiled.market;

    std::uint64_t actor_mask = changed_actor_mask(
        context.provider_action.units, selected.units);
    bool owns_market_tail =
        selected.market.size() > context.provider_action.market.size();
    if (actor_mask != 0 || owns_market_tail) {
      commit::Request request{&context.executor, &context.observation,
                              context.player, context.route, {}, {}};
      auto native = commit::propose(request, context.provider_state_before);
      if (!same(native.action, context.provider_action))
        throw std::runtime_error("crop smoke native proposal drifted");

      if (!native_reject_injected_ &&
          selected.units[0].op == fastkag::Op::PLANT) {
        metrics_->debt_continuations += crop_->plot_debts().size() == 1;
        const auto fingerprint = crop_->persistent_fingerprint();
        const auto expected_actions = crop_->expected_action_receipts();
        const auto expected_purchases = crop_->expected_purchase_receipts();
        const auto commits = crop_->audit().commits;
        auto invalid = commit::bind_repair_final_action(
            native, actor_mask, owns_market_tail, generation(seed_, 1),
            certificate_hash_, selected);
        invalid.content_hash ^= 1ULL;
        auto rejected_state = context.provider_state_before;
        const auto rejected = commit::commit_repair_owner_finalized(
            request, native, invalid, selected, rejected_state);
        ++metrics_->native_rejections;
        if (rejected.committed ||
            !owner_->abort(staged.token) ||
            crop_->persistent_fingerprint() != fingerprint ||
            crop_->expected_action_receipts() != expected_actions ||
            crop_->expected_purchase_receipts() != expected_purchases ||
            crop_->audit().commits != commits ||
            crop_->cached_proposals() != 0)
          ++metrics_->native_reject_ledger_leaks;
        native_reject_injected_ = true;
        ++metrics_->same_hand_retries;
        staged = owner_->prepare(repair_context, typed);
        if (staged.token == 0)
          throw std::runtime_error("crop bridge rejected same-hand retry");
        check_acks(staged.decision, false);
        compiled = repair_fork::ExactMarketCompiler().compile(
            context.observation, context.player,
            context.provider_action.market,
            staged.decision.required_purchases);
        selected = context.provider_action;
        selected.units = staged.decision.units;
        selected.market = compiled.market;
        actor_mask = changed_actor_mask(context.provider_action.units,
                                        selected.units);
        owns_market_tail =
            selected.market.size() > context.provider_action.market.size();
        native = commit::propose(request, context.provider_state_before);
      }

      const auto binding = commit::bind_repair_final_action(
          native, actor_mask, owns_market_tail, generation(seed_, 1),
          certificate_hash_, selected);
      auto committed_state = context.provider_state_before;
      const auto committed = commit::commit_repair_owner_finalized(
          request, native, binding, selected, committed_state);
      if (!committed.committed ||
          !commit::same_action(committed.replayed_action, selected)) {
        (void)owner_->abort(staged.token);
        throw std::runtime_error("crop smoke native exact commit rejected");
      }
      if (!owner_->finalize(staged.token, selected))
        throw std::runtime_error("crop bridge rejected native-selected action");
      output.committed_provider_state = std::move(committed_state);
      ++metrics_->native_commits;
    } else if (!owner_->finalize(staged.token, selected)) {
      throw std::runtime_error("crop bridge rejected provider passthrough");
    }

    output.final_action = selected;
    record_committed_actions(context.step, context.provider_action,
                             staged.decision);
    pending_action_receipts_.clear();
    for (const auto& authority : staged.decision.prefix_authority)
      pending_action_receipts_.push_back(
          {context.step, authority.actor, authority.manifest_generation,
           authority.prefix_manifest_hash,
           authority.post_prefix_state_fingerprint,
           staged.decision.units[static_cast<std::size_t>(authority.actor)]});
    pending_purchase_bindings_ = std::move(compiled.bindings);
    pending_purchase_step_ = context.step;
    record_lineage(context.observation, context.player, output.final_action,
                   lineage_);
    return output;
  }

  void after_step(const Simulator& actual) {
    previous_action_receipts_ = std::move(pending_action_receipts_);
    metrics_->action_receipts_issued +=
        static_cast<int>(previous_action_receipts_.size());
    previous_purchase_receipts_.clear();
    const auto& fills = actual.last_market_fills()[0];
    std::vector<int> remaining(fills.begin(), fills.end());
    for (const auto& binding : pending_purchase_bindings_) {
      int filled = 0;
      if (binding.market_slot >= 0 &&
          binding.market_slot < static_cast<int>(remaining.size())) {
        auto& available = remaining[
            static_cast<std::size_t>(binding.market_slot)];
        filled = std::min(binding.requested, available);
        available -= filled;
      }
      previous_purchase_receipts_.push_back(
          {binding.debt_id, pending_purchase_step_, binding.operation,
           binding.item, binding.requested, filled, binding.market_slot,
           binding.status});
      metrics_->purchase_zero_fills += filled == 0;
    }
    metrics_->purchase_receipts_issued +=
        static_cast<int>(previous_purchase_receipts_.size());
    pending_purchase_bindings_.clear();

    const auto* crop_tile = tile_at(actual, 0, target_);
    if (crop_tile && crop_tile->kind == fastkag::TileKind::PLANT &&
        crop_tile->crop == fastkag::Item::WHEAT &&
        crop_tile->watered_today)
      metrics_->crop_effect_observed = true;
    if (actual.step_count() == 48) close_day();
  }

  void finish() {
    if (owner_) close_day();
  }

 private:
  static std::uint64_t changed_actor_mask(
      std::span<const Action> raw, std::span<const Action> selected) {
    if (raw.size() != selected.size() || raw.size() > 64)
      throw std::runtime_error("crop smoke actor envelope changed");
    std::uint64_t mask = 0;
    for (std::size_t actor = 0; actor < raw.size(); ++actor)
      if (!same(raw[actor], selected[actor])) mask |= 1ULL << actor;
    return mask;
  }

  void start_day(const panel::FinalActionContext& context) {
    if (metrics_->forced_events != 1)
      throw std::runtime_error("crop smoke day started without forced weed");
    issued_ = issuer::issue_day_start(
        {&context.observation, tape_, context.player,
         generation(seed_, context.observation.day()), &lineage_, {}, {}});
    if (!issued_.fully_proven() || issued_.crop_owner_obligations.size() != 1 ||
        issued_.crop_owner_obligations.front().source_step != 24 ||
        issued_.crop_owner_obligations.front().desired !=
            fastkag::Item::WHEAT)
      throw std::runtime_error("crop smoke typed day-start issue failed");
    day_request_ = {&context.observation, context.player,
                    generation(seed_, context.observation.day()),
                    issued_.moves, issued_.obligations};
    const auto plan = obligation_day::plan_day(day_request_);
    if (!plan.planned() || !plan.debts.empty() ||
        !obligation_day::verify_day_schedule(
             day_request_, *plan.certificate).valid)
      throw std::runtime_error("crop smoke certified day did not plan");
    expected_moves_ = plan.certificate->move_replays;
    metrics_->moves_expected = static_cast<int>(expected_moves_.size());
    certificate_hash_ = plan.certificate->content_hash;
    auto crop = std::make_unique<
        transactional_crop::TransactionalCropRepairOwner>();
    crop_ = crop.get();
    owner_ = std::make_unique<
        transactional_bridge::DaySchedulerTransactionalComposerBridge>(
        std::move(crop), std::make_unique<PassiveTransactionalOwner>(),
        day_request_, *plan.certificate);
    if (!owner_->certificate_valid())
      throw std::runtime_error("crop smoke bridge rejected its certificate");
  }

  void check_acks(const repair_fork::RepairDecision& decision,
                  bool count) {
    bool exact = decision.receipt_acks.size() ==
                     previous_action_receipts_.size() &&
                 decision.purchase_receipt_acks.size() ==
                     previous_purchase_receipts_.size();
    for (std::size_t index = 0;
         exact && index < previous_action_receipts_.size(); ++index)
      exact = same(decision.receipt_acks[index],
                   previous_action_receipts_[index]);
    for (std::size_t index = 0;
         exact && index < previous_purchase_receipts_.size(); ++index)
      exact = same(decision.purchase_receipt_acks[index],
                   previous_purchase_receipts_[index]);
    if (!exact) throw std::runtime_error("crop smoke receipt ack mismatch");
    if (count) {
      metrics_->action_receipts_acked +=
          static_cast<int>(previous_action_receipts_.size());
      metrics_->purchase_receipts_acked +=
          static_cast<int>(previous_purchase_receipts_.size());
    }
  }

  void record_committed_actions(
      int step, const PlayerAction& raw,
      const repair_fork::RepairDecision& decision) {
    for (std::size_t actor = 0; actor < decision.units.size(); ++actor) {
      if (same(raw.units[actor], decision.units[actor])) continue;
      metrics_->committed_ops.push_back(
          static_cast<int>(decision.units[actor].op));
      if (!move(decision.units[actor].op)) continue;
      ++metrics_->moves_emitted;
      const auto found = std::find_if(
          expected_moves_.begin(), expected_moves_.end(),
          [&](const auto& expected) {
            return expected.actor == static_cast<int>(actor) &&
                   expected.source_step == decision.sources[actor].source_step &&
                   expected.emitted_step == step &&
                   same(expected.action, decision.units[actor]);
          });
      metrics_->move_mismatches += found == expected_moves_.end();
    }
  }

  void close_day() {
    if (!owner_ || !crop_) return;
    metrics_->crop_commits = crop_->audit().commits;
    metrics_->crop_plots_closed = crop_->audit().plots_closed;
    metrics_->terminal_plot_debts =
        static_cast<int>(crop_->plot_debts().size());
    metrics_->terminal_expected_receipts = static_cast<int>(
        crop_->expected_action_receipts() +
        crop_->expected_purchase_receipts());
    metrics_->terminal_delayed_moves =
        static_cast<int>(crop_->delayed_moves());
    metrics_->moves_emitted = owner_->audit().move_tokens_submitted;
    metrics_->move_mismatches +=
        metrics_->moves_emitted != metrics_->moves_expected;
    owner_.reset();
    crop_ = nullptr;
  }

  const std::vector<PlayerAction>* tape_{};
  std::uint64_t seed_{};
  TransactionalCropSmoke* metrics_{};
  issuer::PersistentRouteIntentRegistry lineage_;
  issuer::IssueResult issued_;
  obligation_day::DayPlanRequest day_request_;
  std::vector<obligation_day::MoveReplay> expected_moves_;
  std::unique_ptr<transactional_bridge::
                      DaySchedulerTransactionalComposerBridge> owner_;
  transactional_crop::TransactionalCropRepairOwner* crop_{};
  std::vector<repair_fork::ActionReceipt> previous_action_receipts_;
  std::vector<repair_fork::PurchaseReceipt> previous_purchase_receipts_;
  std::vector<repair_fork::ActionReceipt> pending_action_receipts_;
  std::vector<repair_fork::PurchaseBinding> pending_purchase_bindings_;
  int pending_purchase_step_{-1};
  std::uint64_t certificate_hash_{};
  fastkag::Position target_{};
  bool native_reject_injected_{};
};

void write_money(std::ostream& out, const panel::MoneyMetrics& value) {
  out << "{\"own\":" << value.own << ",\"opponent\":"
      << value.opponent << ",\"margin\":" << value.margin
      << ",\"score\":" << value.score << '}';
}

void write_steps(std::ostream& out, const std::vector<int>& values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) out << ',';
    out << values[index];
  }
  out << ']';
}

void write_ids(std::ostream& out, const std::vector<std::uint64_t>& values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) out << ',';
    out << values[index];
  }
  out << ']';
}

const char* op_name(int value) {
  constexpr std::array<const char*, 24> names{
      "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
      "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
      "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER",
      "CARE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT",
      "BUY_ANIMAL", "SELL"};
  return value >= 0 && value < static_cast<int>(names.size())
             ? names[static_cast<std::size_t>(value)]
             : "UNKNOWN";
}

const char* goal_name(int value) {
  constexpr std::array<const char*, 8> names{
      "CROP_READY", "PICKUP", "PLACE", "FEED", "CARE", "HARVEST",
      "BUILD_PASTURE", "COLLECT_FERTILIZER"};
  return value >= 0 && value < static_cast<int>(names.size())
             ? names[static_cast<std::size_t>(value)]
             : "UNKNOWN";
}

const char* tile_kind_name(int value) {
  constexpr std::array<const char*, 7> names{
      "EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE", "ANIMAL"};
  return value >= 0 && value < static_cast<int>(names.size())
             ? names[static_cast<std::size_t>(value)]
             : "UNKNOWN";
}

}  // namespace

bool TransactionalCropSmoke::passed() const noexcept {
  return steps == 719 && forced_events == 1 && native_rejections == 1 &&
         native_reject_ledger_leaks == 0 && same_hand_retries == 1 &&
         native_commits == 5 && action_receipts_issued == 5 &&
         action_receipts_acked == action_receipts_issued &&
         purchase_receipts_issued == 2 &&
         purchase_receipts_acked == purchase_receipts_issued &&
         purchase_zero_fills == 2 && debt_continuations == 1 &&
         moves_expected == 2 &&
         moves_emitted == moves_expected && move_mismatches == 0 &&
         crop_commits == 5 && crop_plots_closed == 1 &&
         terminal_plot_debts == 0 && terminal_expected_receipts == 0 &&
         terminal_delayed_moves == 0 && lineage_from_past_plant &&
         dig_plant_water_committed && crop_effect_observed;
}

TransactionalCropSmoke evaluate_transactional_crop_smoke(
    std::uint64_t seed) {
  fastkag::Config config;
  config.weed_spawn_chance = 0.0;
  auto tape = crop_smoke_tape(config.episode_steps);
  fastkag::NativeTapeLibrary library;
  library.routes.push_back(tape);
  NativeTeammateExecutor executor(std::move(library));
  Simulator observation(config, seed);
  auto& private_states = const_cast<std::array<fastkag::PrivateState, 2>&>(
      observation.privates());
  private_states[0].seeds[static_cast<std::size_t>(fastkag::Item::WHEAT)] = 2;
  std::array<NativeAgentState, 2> states;
  TransactionalCropSmoke metrics;
  TransactionalCropCallback callback(tape, seed, metrics);

  while (!observation.done()) {
    callback.force_past_plant_weed(observation);
    const int step = observation.step_count();
    auto before = states;
    std::array<PlayerAction, 2> actions{
        executor.action_external(observation, 0, 0, states[0]),
        executor.action_external(observation, 1, 0, states[1])};
    const panel::FinalActionContext context{
        observation, executor, 0, 0, step, actions, actions[0], before[0],
        states[0]};
    auto selected = callback(context);
    if (!selected.committed_provider_state ||
        selected.final_action.units.size() != actions[0].units.size())
      throw std::runtime_error("crop smoke callback returned invalid envelope");
    actions[0] = std::move(selected.final_action);
    states[0] = std::move(*selected.committed_provider_state);
    observation.step(actions);
    callback.after_step(observation);
    ++metrics.steps;
  }
  callback.finish();
  const std::array<int, 3> chain{
      static_cast<int>(fastkag::Op::DIG),
      static_cast<int>(fastkag::Op::PLANT),
      static_cast<int>(fastkag::Op::WATER)};
  metrics.dig_plant_water_committed =
      metrics.committed_ops.size() >= chain.size() &&
      std::equal(chain.begin(), chain.end(), metrics.committed_ops.begin());
  return metrics;
}

Report evaluate_with_opponent(std::string tapes, std::string library,
                              std::uint64_t seed, bool versus_g096) {
  panel::Options disabled;
  disabled.tapes = tapes;
  disabled.library = library;
  disabled.seed_begin = seed;
  disabled.seeds = 1;
  const auto disabled_report = panel::evaluate(disabled);

  fastkag::NativeTapeLibrary tapes_library;
  tapes_library.routes.push_back(
      repair::load_route(tapes, library, "G001"));
  if (versus_g096)
    tapes_library.routes.push_back(
        repair::load_route(tapes, library, "G096"));
  NativeTeammateExecutor executor(std::move(tapes_library));
  const int opponent_route = versus_g096 ? 1 : 0;
  Report report;
  report.disabled_parity = disabled_report.exact_parity() &&
      std::all_of(disabled_report.games.begin(), disabled_report.games.end(),
                  [](const auto& game) { return game.steps == 719; });
  for (const auto scenario : {panel::Scenario::Normal,
                              panel::Scenario::ForcedWeed,
                              panel::Scenario::ForcedCropWeed})
    for (const int seat : {0, 1})
      report.games.push_back(
          run_game(executor, seed, seat, scenario, false, {}, false,
                   opponent_route));
  report.no_event_metrics = report.games.front();
  const auto& no_event = report.no_event_metrics;
  report.no_event_strict_parity = no_event.steps == 719 &&
      no_event.causal_activations == 0 && no_event.attempts == 0 &&
      no_event.commits == 0 && no_event.action_divergences == 0 &&
      no_event.paired_action_divergences == 0 &&
      no_event.baseline_terminal.own == no_event.repair_terminal.own &&
      no_event.baseline_terminal.margin == no_event.repair_terminal.margin &&
      no_event.baseline_terminal.score == no_event.repair_terminal.score;
  const auto fail_stop = run_game(executor, seed, 1,
                                  panel::Scenario::ForcedWeed,
                                  true, {}, false, opponent_route);
  report.fail_stop_metrics = fail_stop;
  report.fail_stop_probe = fail_stop.steps == 719 &&
      fail_stop.attempts == 1 && fail_stop.commits == 0 &&
      fail_stop.commit_failures == 1 && fail_stop.fallbacks == 1 &&
      fail_stop.provider_state_commits == 0 &&
      fail_stop.provider_state_leaks == 0 &&
      fail_stop.baseline_terminal.own == fail_stop.repair_terminal.own &&
      fail_stop.baseline_terminal.margin == fail_stop.repair_terminal.margin &&
      fail_stop.baseline_terminal.score == fail_stop.repair_terminal.score;
  return report;
}

Report evaluate(std::string tapes, std::string library, std::uint64_t seed) {
  return evaluate_with_opponent(std::move(tapes), std::move(library), seed,
                                false);
}

Report evaluate_vs_g096(std::string tapes, std::string library,
                        std::uint64_t seed) {
  return evaluate_with_opponent(std::move(tapes), std::move(library), seed,
                                true);
}

Metrics evaluate_focused(std::string tapes, std::string library,
                         std::uint64_t seed) {
  fastkag::NativeTapeLibrary tapes_library;
  tapes_library.routes.push_back(
      repair::load_route(tapes, library, "G001"));
  NativeTeammateExecutor executor(std::move(tapes_library));
  return run_game(executor, seed, 0, panel::Scenario::ForcedWeed);
}

Report evaluate_forced_crop_at(std::string tapes, std::string library,
                               std::uint64_t seed, int step, int x, int y,
                               bool weed) {
  fastkag::NativeTapeLibrary tapes_library;
  tapes_library.routes.push_back(
      repair::load_route(tapes, library, "G001"));
  NativeTeammateExecutor executor(std::move(tapes_library));
  Report report;
  const Callback::CropInjection forced{{static_cast<std::int16_t>(x),
                                        static_cast<std::int16_t>(y)}, step,
                                       weed};
  for (const int seat : {0, 1})
    report.games.push_back(run_game(
        executor, seed, seat, panel::Scenario::ForcedCropWeed, false, forced,
        true));
  return report;
}

std::string Report::json() const {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6)
      << "{\"schema\":\"repair-enabled-panel-adapter-v1\","
      << "\"scope\":{\"small_smoke_only\":true,"
         "\"benefit_claimed\":false,\"terminal_oracle_used\":false,"
         "\"future_observation_used\":false,"
         "\"candidate_terminal_ranking_used\":false,"
         "\"selection_inputs\":[\"current_observation\","
         "\"immutable_route\",\"prior_typed_lineage\"]},"
      << "\"disabled_parity\":" << (disabled_parity ? "true" : "false")
      << ",\"fail_stop_probe\":" << (fail_stop_probe ? "true" : "false")
      << ",\"no_event_strict_parity\":"
      << (no_event_strict_parity ? "true" : "false")
      << ",\"no_event_evidence\":{\"seed\":" << no_event_metrics.seed
      << ",\"seat\":" << no_event_metrics.seat
      << ",\"steps\":" << no_event_metrics.steps
      << ",\"causal_activations\":"
      << no_event_metrics.causal_activations
      << ",\"action_divergences\":"
      << no_event_metrics.action_divergences
      << ",\"paired_action_divergences\":"
      << no_event_metrics.paired_action_divergences << "}"
      << ",\"fail_stop_evidence\":{\"steps\":"
      << fail_stop_metrics.steps << ",\"attempts\":"
      << fail_stop_metrics.attempts << ",\"commits\":"
      << fail_stop_metrics.commits << ",\"fallbacks\":"
      << fail_stop_metrics.fallbacks << ",\"commit_failures\":"
      << fail_stop_metrics.commit_failures
      << ",\"provider_state_commits\":"
      << fail_stop_metrics.provider_state_commits
      << ",\"provider_state_leaks\":"
      << fail_stop_metrics.provider_state_leaks
      << ",\"baseline_own\":" << fail_stop_metrics.baseline_terminal.own
      << ",\"repair_own\":" << fail_stop_metrics.repair_terminal.own
      << "}"
      << ",\"games\":[";
  for (std::size_t i = 0; i < games.size(); ++i) {
    if (i) out << ',';
    const auto& game = games[i];
    out << "{\"seed\":" << game.seed << ",\"seat\":" << game.seat
        << ",\"scenario\":\"" << panel::scenario_name(game.scenario)
        << "\",\"steps\":" << game.steps
        << ",\"action_divergences\":" << game.action_divergences
        << ",\"action_divergence_steps\":";
    write_steps(out, game.action_divergence_steps);
    out << ",\"provider_unit_ops\":";
    write_steps(out, game.provider_unit_ops);
    out << ",\"repair_unit_ops\":";
    write_steps(out, game.repair_unit_ops);
    out
        << ",\"paired_action_divergences\":"
        << game.paired_action_divergences
        << ",\"paired_move_mismatches\":"
        << game.paired_move_mismatches
        << ",\"paired_common_move_mismatches\":"
        << game.paired_common_move_mismatches
        << ",\"paired_farmer_move_mismatches\":"
        << game.paired_farmer_move_mismatches
        << ",\"paired_actor_count_mismatches\":"
        << game.paired_actor_count_mismatches
        << ",\"paired_move_mismatch_steps\":";
    write_steps(out, game.paired_move_mismatch_steps);
    out
        << ",\"causal_divergence\":{\"focal_unit_action\":"
        << game.first_focal_unit_action_divergence_step
        << ",\"focal_market_action\":"
        << game.first_focal_market_action_divergence_step
        << ",\"opponent_action\":"
        << game.first_opponent_action_divergence_step
        << ",\"state_after_step\":"
        << game.first_state_divergence_step
        << ",\"own_money_after_step\":"
        << game.first_own_money_divergence_step
        << ",\"opponent_money_after_step\":"
        << game.first_opponent_money_divergence_step
        << ",\"market_after_step\":"
        << game.first_market_state_divergence_step
        << ",\"tile_index\":" << game.first_tile_divergence_index
        << ",\"tile_baseline_kind\":" << game.first_tile_baseline_kind
        << ",\"tile_repair_kind\":" << game.first_tile_repair_kind
        << ",\"tile_baseline_animal\":" << game.first_tile_baseline_animal
        << ",\"tile_repair_animal\":" << game.first_tile_repair_animal
        << ",\"baseline_unit_ops\":";
    write_steps(out, game.first_state_baseline_unit_ops);
    out << ",\"repair_unit_ops\":";
    write_steps(out, game.first_state_repair_unit_ops);
    out << ",\"market_product\":" << game.first_market_divergence_product
        << ",\"market_baseline_inventory\":"
        << game.first_market_baseline_inventory
        << ",\"market_repair_inventory\":"
        << game.first_market_repair_inventory
        << ",\"market_baseline_price\":" << game.first_market_baseline_price
        << ",\"market_repair_price\":" << game.first_market_repair_price
        << ",\"focal_orders\":";
    write_steps(out, game.first_market_focal_orders);
    out << ",\"opponent_orders\":";
    write_steps(out, game.first_market_opponent_orders);
    out << ",\"baseline_focal_fills\":";
    write_steps(out, game.first_market_baseline_focal_fills);
    out << ",\"repair_focal_fills\":";
    write_steps(out, game.first_market_repair_focal_fills);
    out << ",\"baseline_opponent_fills\":";
    write_steps(out, game.first_market_baseline_opponent_fills);
    out << ",\"repair_opponent_fills\":";
    write_steps(out, game.first_market_repair_opponent_fills);
    out << ",\"legacy_trade_exposure\":{\"step\":"
        << game.first_legacy_trade_exposure_step << ",\"product\":"
        << game.first_legacy_trade_exposure_product << ",\"requested\":"
        << game.first_legacy_trade_requested << ",\"baseline_shed\":"
        << game.first_legacy_trade_baseline_shed << ",\"repair_shed\":"
        << game.first_legacy_trade_repair_shed << ",\"incremental_fill\":"
        << game.first_legacy_trade_incremental_fill << ",\"price\":"
        << game.first_legacy_trade_price << ",\"baseline_animals\":"
        << game.first_legacy_trade_baseline_animals << ",\"repair_animals\":"
        << game.first_legacy_trade_repair_animals
        << ",\"baseline_asset_value\":"
        << game.first_legacy_trade_baseline_asset_value
        << ",\"repair_asset_value\":"
        << game.first_legacy_trade_repair_asset_value << "}}"
        << ",\"first_common_move_mismatch\":{\"step\":"
        << game.first_common_move_mismatch_step << ",\"actor\":"
        << game.first_common_move_mismatch_actor << ",\"baseline_op\":"
        << game.first_common_move_baseline_op << ",\"repair_op\":"
        << game.first_common_move_repair_op << "}"
        << ",\"route_move_mismatches\":"
        << game.route_move_mismatches
        << ",\"route_move_mismatch_steps\":";
    write_steps(out, game.route_move_mismatch_steps);
    out
        << ",\"movement_sequence_mismatches\":"
        << game.movement_sequence_mismatches
        << ",\"movement_lane_population_mismatches\":"
        << game.movement_lane_population_mismatches
        << ",\"movement_sequence_mismatch_lanes\":";
    write_steps(out, game.movement_sequence_mismatch_lanes);
    out << ",\"movement_lane_population_mismatch_lanes\":";
    write_steps(out, game.movement_lane_population_mismatch_lanes);
    out
        << ",\"causal_trigger\":{\"activations\":"
        << game.causal_activations << ",\"obligation_id\":"
        << game.trigger_obligation_id << ",\"goal\":\""
        << goal_name(game.trigger_goal) << "\",\"owned_actor\":"
        << game.owned_actor << ",\"crop_activations\":"
        << game.crop_repair_activations
        << ",\"features\":{\"item\":" << game.trigger_item
        << ",\"day\":" << game.trigger_day
        << ",\"remaining_slots\":" << game.trigger_remaining_slots
        << ",\"own_money\":" << game.trigger_own_money
        << ",\"opponent_money\":" << game.trigger_opponent_money
        << ",\"market_price\":" << game.trigger_market_price
        << ",\"market_inventory\":" << game.trigger_market_inventory
        << ",\"seed_inventory\":" << game.trigger_seed_inventory
        << ",\"crop_yield\":" << game.trigger_crop_yield
        << ",\"crop_age\":" << game.trigger_crop_age
        << ",\"unwatered\":" << game.trigger_unwatered
        << ",\"shop_demand\":" << game.trigger_shop_demand
        << ",\"future_ops\":";
    write_steps(out, game.trigger_future_ops);
    out << '}'
        << ",\"source_step\":" << game.trigger_source_step
        << ",\"tile\":[" << game.trigger_x << ',' << game.trigger_y << ']'
        << ",\"observed_weed\":"
        << (game.trigger_observed_weed ? "true" : "false")
        << ",\"effect_observed\":"
        << (game.trigger_effect_observed ? "true" : "false")
        << ",\"final_tile_kind\":" << game.trigger_final_tile_kind
        << ",\"final_tile_kind_name\":\""
        << tile_kind_name(game.trigger_final_tile_kind) << "\""
        << ",\"transition_steps\":";
    write_steps(out, game.trigger_transition_steps);
    out << ",\"transition_ops\":";
    write_steps(out, game.trigger_transition_ops);
    out << ",\"transition_before_kinds\":";
    write_steps(out, game.trigger_transition_before_kinds);
    out << ",\"transition_after_kinds\":";
    write_steps(out, game.trigger_transition_after_kinds);
    out << ",\"transitions\":[";
    for (std::size_t transition = 0;
         transition < game.trigger_transition_steps.size(); ++transition) {
      if (transition) out << ',';
      out << "{\"step\":" << game.trigger_transition_steps[transition]
          << ",\"op\":\""
          << op_name(game.trigger_transition_ops[transition])
          << "\",\"before\":\""
          << tile_kind_name(game.trigger_transition_before_kinds[transition])
          << "\",\"after\":\""
          << tile_kind_name(game.trigger_transition_after_kinds[transition])
          << "\"}";
    }
    out << ']';
    out << "}"
        << ",\"transaction\":{\"triggers\":" << game.triggers
        << ",\"attempts\":" << game.attempts
        << ",\"commits\":" << game.commits
        << ",\"fallbacks\":" << game.fallbacks
        << ",\"provider_state_commits\":" << game.provider_state_commits
        << ",\"provider_state_leaks\":" << game.provider_state_leaks
        << ",\"commit_failures\":" << game.commit_failures
        << ",\"unowned_unit_failures\":" << game.unowned_unit_failures
        << ",\"market_exact_failures\":" << game.market_exact_failures
        << "},\"purchase\":{\"failures\":" << game.purchase_failures
        << ",\"seed_failures\":" << game.seed_purchase_failures
        << ",\"retries\":" << game.purchase_retries
        << ",\"retry_fills\":" << game.purchase_retry_fills
        << ",\"finalize_failures\":" << game.purchase_finalize_failures
        << ",\"move_mismatches\":" << game.purchase_move_mismatches
        << ",\"joint_weed_commits\":"
        << game.joint_weed_purchase_commits
        << "},\"receipts\":{\"checks\":" << game.receipt_checks
        << ",\"accepted\":" << game.receipts_accepted
        << ",\"failed\":" << game.receipts_failed
        << "},\"moves\":{\"expected\":" << game.moves_expected
        << ",\"emitted\":" << game.moves_emitted
        << ",\"failures\":" << game.move_failures
        << ",\"early\":" << game.move_early
        << ",\"duplicates\":" << game.move_duplicates
        << ",\"drops\":" << game.move_drops
        << "},\"terminal_debts\":{\"count\":" << game.terminal_debts
        << ",\"obligation_ids\":";
    write_ids(out, game.terminal_debt_ids);
    out << ",\"source_steps\":";
    write_steps(out, game.terminal_debt_source_steps);
    out << "}"
        << ",\"macro_failures\":{\"baseline_unit\":"
        << game.baseline_unit_failures << ",\"repair_unit\":"
        << game.repair_unit_failures << ",\"baseline_market\":"
        << game.baseline_market_failures << ",\"repair_market\":"
        << game.repair_market_failures << ",\"chains\":{"
        << "\"baseline_unit_steps\":";
    write_steps(out, game.baseline_unit_failure_steps);
    out << ",\"repair_unit_steps\":";
    write_steps(out, game.repair_unit_failure_steps);
    out << ",\"baseline_market_steps\":";
    write_steps(out, game.baseline_market_failure_steps);
    out << ",\"repair_market_steps\":";
    write_steps(out, game.repair_market_failure_steps);
    out << "}},\"forced_schedule\":{\"events\":" << game.forced_events
        << ",\"baseline_applied\":" << game.forced_baseline_applied
        << ",\"repair_applied\":" << game.forced_repair_applied
        << ",\"step\":" << game.forced_step
        << ",\"tile\":[" << game.forced_x << ',' << game.forced_y << ']'
        << ",\"kind\":" << game.forced_kind
        << "},\"hand_us\":{\"total\":" << game.hand_us_total
        << ",\"min\":" << game.hand_us_min
        << ",\"median\":" << game.hand_us_median
        << ",\"max\":" << game.hand_us_max << "},\"terminal\":{"
        << "\"baseline\":";
    write_money(out, game.baseline_terminal);
    out << ",\"repair\":";
    write_money(out, game.repair_terminal);
    out << ",\"delta\":{\"own\":"
        << game.repair_terminal.own - game.baseline_terminal.own
        << ",\"margin\":"
        << game.repair_terminal.margin - game.baseline_terminal.margin
        << ",\"score\":"
        << game.repair_terminal.score - game.baseline_terminal.score
        << "}}}";
  }
  out << "]}";
  return out.str();
}

}  // namespace g001::repair_enabled_panel_adapter
