#include "minimum_damage_remaining_day_audit_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <utility>

namespace g001::minimum_damage_remaining_audit {
namespace {

using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::Position;
using fastkag::Tile;
using fastkag::TileKind;

bool same_action(Action left, Action right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool same_position(Position left, Position right) {
  return left.x == right.x && left.y == right.y;
}

bool movement(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
         op == Op::WEST;
}

Position moved(Position position, Op op) {
  position.x += op == Op::EAST;
  position.x -= op == Op::WEST;
  position.y += op == Op::SOUTH;
  position.y -= op == Op::NORTH;
  return position;
}

const Tile* tile_at(const fastkag::Simulator& env, int player,
                    Position position) {
  if (player < 0 || player >= static_cast<int>(env.farms().size()) ||
      position.x < 0 || position.y < 0 ||
      position.x >= env.config().board_size ||
      position.y >= env.config().board_size)
    return nullptr;
  const auto& tiles = env.farms()[static_cast<std::size_t>(player)].tiles;
  const auto index = static_cast<std::size_t>(position.y) *
                         static_cast<std::size_t>(env.config().board_size) +
                     static_cast<std::size_t>(position.x);
  return index < tiles.size() ? &tiles[index] : nullptr;
}

Position actor_position(const fastkag::Simulator& env, int player,
                        int actor) {
  const auto& farm = env.farms()[static_cast<std::size_t>(player)];
  return actor == 0 ? farm.farmer
                    : farm.hands[static_cast<std::size_t>(actor - 1)];
}

int actor_inventory(const fastkag::Simulator& env, int player, int actor,
                    Item item) {
  const int index = static_cast<int>(item);
  if (player < 0 || player >= static_cast<int>(env.privates().size()) ||
      actor < 0 || index < 0 || index >= fastkag::N_ITEMS)
    return -1;
  const auto& inventories =
      env.privates()[static_cast<std::size_t>(player)].inventories;
  if (static_cast<std::size_t>(actor) >= inventories.size() ||
      static_cast<std::size_t>(index) >=
          inventories[static_cast<std::size_t>(actor)].size())
    return -1;
  return inventories[static_cast<std::size_t>(actor)]
                    [static_cast<std::size_t>(index)];
}

bool concrete_effect(const fastkag::Simulator& before, int player,
                     int actor, Action action,
                     const fastkag::Simulator& after) {
  const auto before_position = actor_position(before, player, actor);
  const auto after_position = actor_position(after, player, actor);
  if (movement(action.op))
    return same_position(after_position, moved(before_position, action.op));
  const auto* before_tile = tile_at(before, player, before_position);
  const auto* after_tile = tile_at(after, player, before_position);
  if (!before_tile || !after_tile) return false;
  switch (action.op) {
    case Op::PASS: return true;
    case Op::DIG:
      return before_tile->kind != TileKind::EMPTY &&
             after_tile->kind == TileKind::EMPTY;
    case Op::PLANT:
      return after_tile->kind == TileKind::PLANT &&
             after_tile->crop == action.item;
    case Op::WATER:
      return before_tile->kind == TileKind::PLANT &&
             !before_tile->watered_today && after_tile->watered_today;
    case Op::COLLECT_FERTILIZER:
      return before_tile->kind == TileKind::ANIMAL &&
             before_tile->fertilizer_available &&
             !after_tile->fertilizer_available &&
             actor_inventory(after, player, actor, Item::FERTILIZER) >
                 actor_inventory(before, player, actor, Item::FERTILIZER);
    case Op::BUILD_PASTURE:
      return after_tile->kind == TileKind::PASTURE;
    case Op::PICKUP:
      return actor_inventory(after, player, actor, action.item) >
             actor_inventory(before, player, actor, action.item);
    case Op::PLACE:
      return after_tile->kind == TileKind::ANIMAL &&
             after_tile->animal == action.item;
    case Op::FEED:
      return before_tile->kind == TileKind::ANIMAL &&
             !before_tile->fed_today && after_tile->fed_today;
    case Op::CARE:
      return before_tile->kind == TileKind::ANIMAL &&
             !before_tile->cared_today && after_tile->cared_today;
    default: return false;
  }
}

void mix(std::uint64_t& hash, std::uint64_t value) {
  hash ^= value;
  hash *= 1099511628211ULL;
}

std::uint64_t evidence_hash(const fastkag::Simulator& before, int player,
                            int actor, Action action,
                            const fastkag::Simulator& after, bool effect) {
  std::uint64_t hash = 1469598103934665603ULL;
  mix(hash, static_cast<std::uint64_t>(before.step_count()));
  mix(hash, static_cast<std::uint64_t>(after.step_count()));
  mix(hash, static_cast<std::uint64_t>(actor));
  mix(hash, static_cast<std::uint64_t>(action.op));
  mix(hash, static_cast<std::uint64_t>(static_cast<int>(action.item) + 1));
  mix(hash, static_cast<std::uint64_t>(action.quantity));
  mix(hash, static_cast<std::uint64_t>(effect));
  const auto before_position = actor_position(before, player, actor);
  const auto after_position = actor_position(after, player, actor);
  mix(hash, static_cast<std::uint64_t>(before_position.x));
  mix(hash, static_cast<std::uint64_t>(before_position.y));
  mix(hash, static_cast<std::uint64_t>(after_position.x));
  mix(hash, static_cast<std::uint64_t>(after_position.y));
  if (const auto* tile = tile_at(before, player, before_position)) {
    mix(hash, static_cast<std::uint64_t>(tile->kind));
    mix(hash, static_cast<std::uint64_t>(static_cast<int>(tile->crop) + 1));
    mix(hash, static_cast<std::uint64_t>(static_cast<int>(tile->animal) + 1));
    mix(hash, static_cast<std::uint64_t>(tile->fed_today));
    mix(hash, static_cast<std::uint64_t>(tile->cared_today));
  }
  if (const auto* tile = tile_at(after, player, before_position)) {
    mix(hash, static_cast<std::uint64_t>(tile->kind));
    mix(hash, static_cast<std::uint64_t>(static_cast<int>(tile->crop) + 1));
    mix(hash, static_cast<std::uint64_t>(static_cast<int>(tile->animal) + 1));
    mix(hash, static_cast<std::uint64_t>(tile->fed_today));
    mix(hash, static_cast<std::uint64_t>(tile->cared_today));
  }
  return hash == 0 ? 1 : hash;
}

bool durable_goal_satisfied(
    const fastkag::Simulator& observation, int player,
    const obligation_day::ProductionObligation& obligation,
    Action original_raw) {
  const auto* tile = tile_at(observation, player, obligation.tile);
  if (!tile) return false;
  using Goal = obligation_day::GoalKind;
  switch (obligation.goal) {
    case Goal::BuildPasture:
      return tile->kind == TileKind::PASTURE || tile->kind == TileKind::ANIMAL;
    case Goal::BuildCoop:
      return tile->kind == TileKind::COOP ||
             (tile->kind == TileKind::ANIMAL &&
              tile->animal == fastkag::Item::GOOSE);
    case Goal::Place:
      return tile->kind == TileKind::ANIMAL &&
             tile->animal == obligation.item;
    case Goal::Feed:
      return tile->kind == TileKind::ANIMAL &&
             tile->animal == obligation.item && tile->fed_today;
    case Goal::Care:
      return tile->kind == TileKind::ANIMAL &&
             tile->animal == obligation.item && tile->cared_today;
    case Goal::CropReady:
      if (original_raw.op == Op::DIG)
        return tile->kind == TileKind::EMPTY;
      if (original_raw.op == Op::PLANT)
        return tile->kind == TileKind::PLANT &&
               tile->crop == obligation.item;
      return original_raw.op == Op::WATER &&
             tile->kind == TileKind::PLANT &&
             tile->crop == obligation.item && tile->watered_today;
    case Goal::Pickup:
    case Goal::Harvest:
    case Goal::CollectFertilizer: return false;
  }
  return false;
}

std::uint64_t goal_evidence_hash(
    const fastkag::Simulator& observation, int player,
    const obligation_day::ProductionObligation& obligation) {
  std::uint64_t hash = 1469598103934665603ULL;
  mix(hash, static_cast<std::uint64_t>(observation.step_count()));
  mix(hash, obligation.id);
  mix(hash, static_cast<std::uint64_t>(obligation.goal));
  mix(hash, static_cast<std::uint64_t>(static_cast<int>(obligation.item) + 1));
  mix(hash, static_cast<std::uint64_t>(obligation.tile.x));
  mix(hash, static_cast<std::uint64_t>(obligation.tile.y));
  if (const auto* tile = tile_at(observation, player, obligation.tile)) {
    mix(hash, static_cast<std::uint64_t>(tile->kind));
    mix(hash, static_cast<std::uint64_t>(static_cast<int>(tile->crop) + 1));
    mix(hash, static_cast<std::uint64_t>(static_cast<int>(tile->animal) + 1));
    mix(hash, static_cast<std::uint64_t>(tile->fed_today));
    mix(hash, static_cast<std::uint64_t>(tile->cared_today));
  }
  return hash == 0 ? 1 : hash;
}

}  // namespace

Adapter::Adapter(
    const day_start_issuer::IssueResult& issued, int player,
    std::uint64_t issuer_generation,
    std::vector<minimum_damage_bridge::ObligationPolicy> policies,
    int actor,
    std::size_t maximum_states)
    : issued_(&issued),
      player_(player),
      actor_(actor),
      issuer_generation_(issuer_generation),
      policies_(std::move(policies)),
      maximum_states_(maximum_states) {}

HandAudit Adapter::plan(const fastkag::Simulator& observation) {
  HandAudit audit;
  audit.step = observation.step_count();
  pending_slot_.reset();
  if (!issued_ || player_ < 0 || player_ >= 2 || actor_ < 0 ||
      actor_ > static_cast<int>(observation.farms()[player_].hands.size()) ||
      actor_ >= static_cast<int>(
                    observation.privates()[player_].inventories.size()) ||
      observation.done()) {
    audit.diagnostic = "invalid/terminal observation";
    hands_.push_back(audit);
    return audit;
  }
  std::vector<minimum_damage_bridge::PriorObligationProgress> prior;
  std::vector<minimum_damage_bridge::PriorMoveProgress> prior_moves;
  for (const auto& token : issued_->moves) {
    if (token.actor != actor_ || token.source_step >= audit.step) continue;
    const auto evidence = passed_move_evidence_.find(token.source_step);
    if (evidence == passed_move_evidence_.end() || evidence->second == 0) {
      audit.diagnostic = "prior hard MOVE lacks observation evidence";
      hands_.push_back(audit);
      return audit;
    }
    prior_moves.push_back({token.source_step,
                           observed_move_steps_.contains(token.source_step),
                           evidence->second});
  }
  std::set<std::uint64_t> active;
  for (const auto& obligation : issued_->obligations) {
    if (obligation.actor != actor_) continue;
    if (obligation.source_step < audit.step) {
      const auto found = progress_.find(obligation.id);
      if (found == progress_.end() || !found->second.observed ||
          found->second.evidence_hash == 0) {
        audit.diagnostic = "prior obligation lacks concrete observation receipt";
        hands_.push_back(audit);
        return audit;
      }
      prior.push_back({obligation.id, obligation.source_step,
                       found->second.completed, found->second.evidence_hash});
      if (!found->second.completed) active.insert(obligation.id);
    } else {
      active.insert(obligation.id);
    }
  }

  std::vector<minimum_damage_bridge::ObligationPolicy> policies;
  for (const auto id : active) {
    const auto matches = std::count_if(
        policies_.begin(), policies_.end(),
        [id](const auto& policy) { return policy.obligation_id == id; });
    if (matches != 1) {
      audit.diagnostic = "active obligation lacks unique policy";
      hands_.push_back(audit);
      return audit;
    }
    policies.push_back(*std::find_if(
        policies_.begin(), policies_.end(),
        [id](const auto& policy) { return policy.obligation_id == id; }));
  }

  minimum_damage_bridge::Request request;
  request.remaining_day_start = &observation;
  request.player = player_;
  request.actor = actor_;
  request.issuer_generation = issuer_generation_;
  request.issued = issued_;
  request.policies = policies;
  request.maximum_states = maximum_states_;
  request.prior_obligation_progress = prior;
  request.prior_move_progress = prior_moves;

  const auto start = std::chrono::steady_clock::now();
  const auto issued = minimum_damage_bridge::issue(request);
  audit.elapsed_microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
  if (!issued.issued()) {
    audit.diagnostic = std::string(minimum_damage_bridge::reject_reason_name(
                                      issued.reject)) +
                       ":" + issued.diagnostic;
    hands_.push_back(audit);
    return audit;
  }
  const auto verified =
      minimum_damage_bridge::verify(request, *issued.certificate);
  if (!verified.valid) {
    audit.diagnostic = "bridge verifier rejected rolling certificate:" +
                       verified.diagnostic;
    hands_.push_back(audit);
    return audit;
  }

  audit.certificate_hash = issued.certificate->content_hash;
  audit.selector_states = issued.certificate->selector_explored_states;
  audit.signed_debts = issued.certificate->debts.size();
  audit.signed_move_tokens = issued.certificate->emitted_raw_moves;
  if (issued.certificate->slots.empty() ||
      issued.certificate->slots.front().step != audit.step) {
    audit.diagnostic = "certificate lacks current signed slot";
    hands_.push_back(audit);
    return audit;
  }
  pending_slot_ = issued.certificate->slots.front();
  audit.candidate_unit = pending_slot_->unit;
  audit.candidate_market = pending_slot_->market;
  for (const auto& token : issued_->moves)
    if (token.actor == actor_ &&
        (token.source_step >= audit.step ||
         !observed_move_steps_.contains(token.source_step)))
      ++audit.remaining_move_tokens;
  for (const auto& slot : issued.certificate->slots) {
    if (slot.raw_move_source_step >= 0 &&
        slot.step < slot.raw_move_source_step)
      ++move_early_violations_;
  }

  std::set<std::uint64_t> represented;
  for (const auto& slot : issued.certificate->slots) {
    if (slot.obligation_id) represented.insert(slot.obligation_id);
    if (slot.market_obligation_id)
      represented.insert(slot.market_obligation_id);
  }
  for (const auto& debt : issued.certificate->debts)
    represented.insert(debt.obligation_id);
  for (const auto& [id, progress] : progress_) {
    if (!progress.observed || progress.completed) continue;
    ++audit.outstanding_obligations;
    if (!represented.contains(id)) ++audit.outstanding_omissions;
  }
  outstanding_omissions_ += audit.outstanding_omissions;
  audit.valid = audit.remaining_move_tokens == audit.signed_move_tokens &&
                audit.outstanding_omissions == 0;
  audit.diagnostic = audit.valid ? "signed default-off suffix candidate"
                                 : "MOVE/debt closure mismatch";
  hands_.push_back(audit);
  return audit;
}

bool Adapter::observe_final(const fastkag::Simulator& before,
                            Action final_actor_action,
                            const fastkag::Simulator& after) {
  if (!issued_ || !pending_slot_ ||
      pending_slot_->step != before.step_count() ||
      !same_action(pending_slot_->unit, final_actor_action) ||
      before.step_count() + 1 != after.step_count())
    return false;
  const int move_source_step = pending_slot_->raw_move_source_step;
  const auto obligation_id = pending_slot_->obligation_id;
  pending_slot_.reset();
  return record_observation(before, final_actor_action, after,
                            move_source_step, obligation_id);
}

bool Adapter::observe_passthrough(const fastkag::Simulator& before,
                                  Action provider_action,
                                  const fastkag::Simulator& after) {
  const int step = before.step_count();
  const auto source = std::find_if(
      issued_->raw_sources.begin(), issued_->raw_sources.end(),
      [&](const auto& candidate) {
        return candidate.actor == actor_ && candidate.source_step == step;
      });
  if (source == issued_->raw_sources.end() ||
      !same_action(source->action, provider_action) ||
      step + 1 != after.step_count())
    return false;
  const auto move = std::find_if(
      issued_->moves.begin(), issued_->moves.end(), [&](const auto& token) {
        return token.actor == actor_ && token.source_step == step;
      });
  const auto obligation = std::find_if(
      issued_->obligations.begin(), issued_->obligations.end(),
      [&](const auto& candidate) {
        return candidate.actor == actor_ && candidate.source_step == step;
      });
  return record_observation(
      before, provider_action, after,
      move == issued_->moves.end() ? -1 : move->source_step,
      obligation == issued_->obligations.end() ? 0 : obligation->id);
}

bool Adapter::record_observation(const fastkag::Simulator& before,
                                 Action final_actor_action,
                                 const fastkag::Simulator& after,
                                 int move_source_step,
                                 std::uint64_t obligation_id) {
  const int step = before.step_count();
  const bool effect = concrete_effect(before, player_, actor_,
                                      final_actor_action, after);
  if (move_source_step >= 0) {
    if (!effect) {
      ++failed_move_receipts_;
      return false;
    }
    if (move_source_step > step) {
      ++move_early_violations_;
      return false;
    }
    if (!observed_move_steps_.insert(move_source_step).second)
      ++duplicate_move_receipts_;
    passed_move_evidence_[move_source_step] = evidence_hash(
        before, player_, actor_, final_actor_action, after, effect);
  }

  if (obligation_id) {
    const auto obligation = std::find_if(
        issued_->obligations.begin(), issued_->obligations.end(),
        [&](const auto& candidate) {
          return candidate.id == obligation_id &&
                 candidate.actor == actor_;
        });
    if (obligation == issued_->obligations.end()) return false;
    auto& progress = progress_[obligation->id];
    progress.observed = true;
    progress.evidence_hash = evidence_hash(
        before, player_, actor_, final_actor_action, after, effect);
    if (effect && (obligation->goal == obligation_day::GoalKind::Pickup ||
                   obligation->goal ==
                       obligation_day::GoalKind::CollectFertilizer))
      progress.completed = true;
  }

  // Every source whose timestamp has now passed gets an explicit observation
  // record, even when the rolling candidate chose different work in this slot.
  // It remains incomplete unless a concrete typed goal predicate proves it.
  for (const auto& obligation : issued_->obligations) {
    if (obligation.actor != actor_ || obligation.source_step > step) continue;
    auto& progress = progress_[obligation.id];
    if (!progress.observed) {
      progress.observed = true;
      progress.completed = false;
      progress.evidence_hash =
          goal_evidence_hash(after, player_, obligation);
    }
  }

  for (const auto& token : issued_->moves) {
    if (token.actor == actor_ && token.source_step <= step &&
        !passed_move_evidence_.contains(token.source_step))
      passed_move_evidence_[token.source_step] = evidence_hash(
          before, player_, actor_, final_actor_action, after, false);
  }

  // A scheduled repair or later action may satisfy an older stateful goal.
  // Settle only from the exact typed target predicate.  PICKUP is deliberately
  // excluded here because present inventory alone has no provenance.
  for (const auto& obligation : issued_->obligations) {
    if (obligation.actor != actor_ || obligation.source_step > step) continue;
    auto found = progress_.find(obligation.id);
    if (found == progress_.end() || !found->second.observed ||
        found->second.completed)
      continue;
    const auto raw_source = std::find_if(
        issued_->raw_sources.begin(), issued_->raw_sources.end(),
        [&](const auto& source) {
          return source.actor == actor_ &&
                 source.source_step == obligation.source_step;
        });
    if (raw_source != issued_->raw_sources.end() &&
        durable_goal_satisfied(after, player_, obligation,
                               raw_source->action)) {
      found->second.completed = true;
      found->second.evidence_hash =
          goal_evidence_hash(after, player_, obligation);
    }
  }
  return true;
}

DayAudit Adapter::finish() const {
  DayAudit audit;
  audit.hands = static_cast<int>(hands_.size());
  audit.observed_move_receipts =
      static_cast<int>(observed_move_steps_.size());
  audit.duplicate_move_receipts = duplicate_move_receipts_;
  audit.failed_move_receipts = failed_move_receipts_;
  audit.move_early_violations = move_early_violations_;
  audit.outstanding_omissions = outstanding_omissions_;
  audit.hand_audits = hands_;
  for (const auto& token : issued_->moves)
    audit.initial_move_tokens += token.actor == actor_;
  std::vector<long long> timings;
  for (const auto& hand : hands_) {
    audit.total_selector_states += hand.selector_states;
    audit.total_issue_microseconds += hand.elapsed_microseconds;
    timings.push_back(hand.elapsed_microseconds);
  }
  for (const auto& [id, progress] : progress_) {
    audit.completed_obligations += progress.observed && progress.completed;
    audit.outstanding_obligations += progress.observed && !progress.completed;
    if (progress.observed && !progress.completed)
      audit.terminal_debts.push_back({id, progress.evidence_hash, 1});
  }
  if (!timings.empty()) {
    std::sort(timings.begin(), timings.end());
    audit.min_hand_microseconds = timings.front();
    audit.median_hand_microseconds = timings[timings.size() / 2];
    audit.max_hand_microseconds = timings.back();
  }
  return audit;
}

}  // namespace g001::minimum_damage_remaining_audit
