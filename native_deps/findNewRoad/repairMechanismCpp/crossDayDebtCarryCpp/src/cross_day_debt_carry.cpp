#include "cross_day_debt_carry.hpp"

#include "g001_day_start_obligation_issuer.hpp"
#include "native_final_action_commit.hpp"
#include "production_suffix_scheduler.hpp"
#include "repair_fork_evaluator.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace g001::cross_day_debt_carry {
namespace {

namespace issuer = day_start_issuer;
namespace owner = real_weed_move_owner;
namespace commit = native_final_commit;
using fastkag::Action;
using fastkag::Item;
using fastkag::NativeAgentState;
using fastkag::NativeTeammateExecutor;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Position;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;

void mix(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8)) & 255U;
    hash *= 1099511628211ULL;
  }
}

void mix_action(std::uint64_t& hash, Action action) {
  mix(hash, static_cast<std::uint8_t>(action.op));
  mix(hash, static_cast<std::uint8_t>(action.item));
  mix(hash, static_cast<std::uint32_t>(action.quantity));
}

bool same(Action lhs, Action rhs) {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool same(const PlayerAction& lhs, const PlayerAction& rhs) {
  if (lhs.units.size() != rhs.units.size() ||
      lhs.market.size() != rhs.market.size())
    return false;
  for (std::size_t i = 0; i < lhs.units.size(); ++i)
    if (!same(lhs.units[i], rhs.units[i])) return false;
  for (std::size_t i = 0; i < lhs.market.size(); ++i)
    if (!same(lhs.market[i], rhs.market[i])) return false;
  return true;
}

bool movement(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
         op == Op::WEST;
}

const Tile* tile_at(const Simulator& state, int player, Position position) {
  const int size = state.config().board_size;
  if (player < 0 || player >= 2 || position.x < 0 || position.y < 0 ||
      position.x >= size || position.y >= size)
    return nullptr;
  const auto index = static_cast<std::size_t>(position.y * size + position.x);
  const auto& tiles = state.farms()[static_cast<std::size_t>(player)].tiles;
  return index < tiles.size() ? &tiles[index] : nullptr;
}

Position actor_position(const Simulator& state, int player, int actor) {
  if (actor == 0) return state.farms()[player].farmer;
  return state.farms()[player].hands.at(static_cast<std::size_t>(actor - 1));
}

std::uint64_t generation(std::uint64_t seed, int day) {
  // Match the sealed repair-enabled adapter's issuer envelope exactly.
  return (seed << 20) | (2ULL << 16) |
         static_cast<std::uint64_t>(day + 1);
}

std::uint64_t evidence_hash(const Simulator& state, int player,
                            const obligation_day::ProductionObligation& value,
                            std::uint64_t certificate_hash) {
  std::uint64_t hash = 1469598103934665603ULL;
  mix(hash, state.seed());
  mix(hash, state.step_count());
  mix(hash, player);
  mix(hash, owner::production_obligation_identity_hash(value));
  mix(hash, certificate_hash);
  if (const auto* tile = tile_at(state, player, value.tile)) {
    mix(hash, static_cast<std::uint8_t>(tile->kind));
    mix(hash, static_cast<std::uint8_t>(tile->animal));
    mix(hash, tile->fed_today);
    mix(hash, tile->cared_today);
  }
  return hash == 0 ? 1 : hash;
}

bool goal_satisfied(const Simulator& state, int player,
                    const obligation_day::ProductionObligation& value) {
  const auto* tile = tile_at(state, player, value.tile);
  if (!tile) return false;
  using Goal = obligation_day::GoalKind;
  switch (value.goal) {
    case Goal::Place:
      return tile->kind == TileKind::ANIMAL && tile->animal == value.item;
    case Goal::BuildPasture:
      return tile->kind == TileKind::PASTURE ||
             (tile->kind == TileKind::ANIMAL && tile->animal != Item::NONE);
    case Goal::BuildCoop:
      return tile->kind == TileKind::COOP ||
             (tile->kind == TileKind::ANIMAL && tile->animal == Item::GOOSE);
    case Goal::Feed:
      return tile->kind == TileKind::ANIMAL && tile->animal == value.item &&
             tile->fed_today;
    case Goal::Care:
      return tile->kind == TileKind::ANIMAL && tile->animal == value.item &&
             tile->cared_today;
    case Goal::CropReady:
      return tile->kind == TileKind::PLANT && tile->crop == value.item;
    case Goal::Pickup:
    case Goal::Harvest:
    case Goal::CollectFertilizer: return false;
  }
  return false;
}

bool target_applicable(const Simulator& state, int player,
                       const obligation_day::ProductionObligation& value) {
  const auto* tile = tile_at(state, player, value.tile);
  if (!tile) return false;
  using Goal = obligation_day::GoalKind;
  switch (value.goal) {
    case Goal::Feed:
      return tile->kind == TileKind::ANIMAL && tile->animal == value.item &&
             !tile->fed_today;
    case Goal::Care:
      return tile->kind == TileKind::ANIMAL && tile->animal == value.item &&
             !tile->cared_today;
    case Goal::Place:
      return tile->kind == TileKind::PASTURE && tile->animal == Item::NONE;
    case Goal::BuildPasture:
      return tile->kind == TileKind::EMPTY || tile->kind == TileKind::WEED;
    case Goal::BuildCoop:
      return tile->kind == TileKind::EMPTY || tile->kind == TileKind::WEED;
    case Goal::CropReady:
      return tile->kind == TileKind::EMPTY || tile->kind == TileKind::WEED ||
             (tile->kind == TileKind::PLANT && tile->crop == value.item);
    case Goal::Pickup:
    case Goal::Harvest: return true;
    case Goal::CollectFertilizer:
      return tile->kind == TileKind::ANIMAL && tile->fertilizer_available;
  }
  return false;
}

bool base_resource_available(const Simulator& state, int player,
                             const obligation_day::ProductionObligation& value) {
  const int item = static_cast<int>(value.resource.item);
  if (item < 0 || item >= fastkag::N_ITEMS) {
    return value.resource.seed_quantity == 0 &&
           value.resource.shed_quantity == 0 &&
           value.resource.carried_quantity == 0;
  }
  const auto& private_state = state.privates()[player];
  if (value.resource.seed_quantity > private_state.seeds[item] ||
      value.resource.shed_quantity > private_state.shed[item])
    return false;
  return true;
}

void rebase(obligation_day::ProductionObligation& value, int day_start,
            int day_end) {
  value.earliest_step = day_start;
  value.deadline_step = day_end;
  value.source_step = -1;
  value.policy_deferred = false;
  value.repair = true;
}

DebtAuthorization make_authorization(
    int player, int day, std::uint64_t issuer_generation,
    std::uint64_t certificate_hash,
    const owner::DebtSelectionIdentity& upstream,
    const issuer::IssueResult& issued, const Simulator& midnight) {
  const auto found = std::find_if(
      issued.obligations.begin(), issued.obligations.end(),
      [&](const auto& value) { return value.id == upstream.obligation_id; });
  if (found == issued.obligations.end())
    throw std::runtime_error("day7 debt lost typed obligation identity");
  DebtAuthorization output;
  output.player = player;
  output.original_day = day;
  output.valid_day = day + 1;
  output.expires_day = day + 2;
  output.original_issuer_generation = issuer_generation;
  output.original_certificate_hash = certificate_hash;
  output.upstream_selection_hash = upstream.content_hash;
  output.obligation = *found;
  output.original_source_action = upstream.source_action;
  output.obligation_identity_hash =
      owner::production_obligation_identity_hash(*found);
  for (const auto dependency_id : found->dependencies) {
    const auto dependency = std::find_if(
        issued.obligations.begin(), issued.obligations.end(),
        [&](const auto& value) { return value.id == dependency_id; });
    if (dependency == issued.obligations.end())
      throw std::runtime_error("day7 debt dependency identity missing");
    output.dependencies.push_back(
        {*dependency, owner::production_obligation_identity_hash(*dependency)});
  }
  output.day7_noncompletion_evidence_hash =
      evidence_hash(midnight, player, *found, certificate_hash);
  output.authorization_generation = issuer_generation ^ 0xc4a771d3ULL;
  output.content_hash = authorization_hash(output);
  return output;
}

struct MidnightGuard {
  int day{-1};
  bool closed{};
  bool close(int value) {
    if (closed || value != day) return false;
    closed = true;
    return true;
  }
};

struct Day8Install {
  issuer::IssueResult issued;
  obligation_day::DayPlanRequest request;
  obligation_day::DayPlanResult plan;
  RevalidationResult validation;
};

Day8Install install_day8(const Simulator& state, int player,
                         std::uint64_t issuer_generation,
                         const NativeTeammateExecutor& executor, int route,
                         const issuer::PersistentRouteIntentRegistry& lineage,
                         const DebtAuthorization& authorization,
                         const PersistentDebtRegistry& registry) {
  Day8Install output;
  output.validation = revalidate(authorization, registry, executor,
                                 state, route);
  if (!output.validation.admitted) return output;
  output.issued = issuer::issue_day_start(
      {&state, &executor.route_tape(route), player, issuer_generation,
       &lineage, {}, {}});
  if (!output.issued.issued()) return output;
  output.request.day_start = &state;
  output.request.player = player;
  output.request.issuer_generation = issuer_generation;
  for (const auto& move : output.issued.moves)
    if (move.actor == authorization.obligation.actor)
      output.request.moves.push_back(move);
  for (const auto& obligation : output.issued.obligations) {
    if (obligation.actor != authorization.obligation.actor) continue;
    const bool dependencies_in_scope = std::all_of(
        obligation.dependencies.begin(), obligation.dependencies.end(),
        [&](std::uint64_t id) {
          return std::any_of(output.issued.obligations.begin(),
                             output.issued.obligations.end(),
                             [&](const auto& candidate) {
                               return candidate.id == id &&
                                      candidate.actor == obligation.actor;
                             });
        });
    if (!dependencies_in_scope) {
      output.validation.admitted = false;
      output.validation.reject = RevalidationReject::DependencyMismatch;
      return output;
    }
    output.request.obligations.push_back(obligation);
  }
  for (const auto& unsupported : output.issued.unsupported) {
    if (unsupported.actor == authorization.obligation.actor) {
      output.validation.admitted = false;
      output.validation.reject = RevalidationReject::InvalidEnvelope;
      return output;
    }
  }
  auto carried = output.validation.carried;
  // Dependencies were revalidated against the current observation. They are
  // discharged witnesses, not silently erased unknown cross-actor work.
  carried.dependencies.clear();
  for (const auto& prerequisite : output.validation.dependency_witnesses) {
    output.request.obligations.push_back(prerequisite);
    carried.dependencies.push_back(prerequisite.id);
  }
  output.request.obligations.push_back(carried);
  output.plan = obligation_day::plan_day(output.request);
  return output;
}

bool physical_effect(const Simulator& before,
                     const std::array<PlayerAction, 2>& joint, int player,
                     int actor, Action action, Simulator& unit_after) {
  const auto position = actor_position(before, player, actor);
  const auto* before_tile = tile_at(before, player, position);
  unit_after = before.preview_unit_phase(joint);
  const auto after_position = actor_position(unit_after, player, actor);
  const auto* after_tile = tile_at(unit_after, player, position);
  if (movement(action.op)) {
    Position expected = position;
    expected.x += action.op == Op::EAST;
    expected.x -= action.op == Op::WEST;
    expected.y += action.op == Op::SOUTH;
    expected.y -= action.op == Op::NORTH;
    return after_position.x == expected.x && after_position.y == expected.y;
  }
  if (action.op == Op::PASS) return true;
  if (!before_tile || !after_tile) return false;
  switch (action.op) {
    case Op::FEED:
      return before_tile->kind == TileKind::ANIMAL &&
             !before_tile->fed_today && after_tile->fed_today;
    case Op::CARE:
      return before_tile->kind == TileKind::ANIMAL &&
             !before_tile->cared_today && after_tile->cared_today;
    case Op::DIG:
      return before_tile->kind != TileKind::EMPTY &&
             after_tile->kind == TileKind::EMPTY;
    case Op::BUILD_PASTURE:
      return before_tile->kind == TileKind::EMPTY &&
             after_tile->kind == TileKind::PASTURE;
    case Op::PLACE:
      return after_tile->kind == TileKind::ANIMAL &&
             after_tile->animal == action.item;
    case Op::PLANT:
      return after_tile->kind == TileKind::PLANT &&
             after_tile->crop == action.item;
    case Op::WATER:
      return after_tile->kind == TileKind::PLANT &&
             after_tile->watered_today;
    case Op::PICKUP: {
      const int item = static_cast<int>(action.item);
      return item >= 0 && item < fastkag::N_ITEMS &&
             before.privates()[player].inventories[actor][item] <
                 unit_after.privates()[player].inventories[actor][item];
    }
    case Op::HARVEST:
      return before_tile->yield_units > after_tile->yield_units;
    default:
      return production_suffix::focal_unit_state_fingerprint(before, player) !=
             production_suffix::focal_unit_state_fingerprint(unit_after,
                                                              player);
  }
}

bool run_default_off(const NativeTeammateExecutor& executor,
                     std::uint64_t seed) {
  Simulator left({}, seed);
  Simulator right({}, seed);
  std::array<NativeAgentState, 2> left_state;
  std::array<NativeAgentState, 2> right_state;
  int steps = 0;
  while (!left.done() && !right.done()) {
    std::array<PlayerAction, 2> lhs;
    std::array<PlayerAction, 2> rhs;
    for (int player = 0; player < 2; ++player) {
      lhs[player] = executor.action_external(left, player, 0,
                                             left_state[player]);
      rhs[player] = executor.action_external(right, player, 0,
                                             right_state[player]);
      if (!same(lhs[player], rhs[player])) return false;
    }
    left.step(lhs);
    right.step(rhs);
    if (repair_fork::full_unit_phase_state_fingerprint(left) !=
        repair_fork::full_unit_phase_state_fingerprint(right))
      return false;
    ++steps;
  }
  return steps == 719 && left.done() && right.done();
}

struct RunOutput {
  Metrics metrics;
  DebtAuthorization authorization;
  PersistentDebtRegistry registry;
  Simulator day8_start;
  const NativeTeammateExecutor* executor{};
};

RunOutput run_game(const NativeTeammateExecutor& executor, std::uint64_t seed,
                   int seat, bool stop_after_day8 = false) {
  RunOutput output;
  auto& metrics = output.metrics;
  metrics.seed = seed;
  metrics.seat = seat;
  output.executor = &executor;
  Simulator baseline({}, seed);
  Simulator repair({}, seed);
  std::array<NativeAgentState, 2> baseline_states;
  NativeAgentState repair_opponent;
  owner::State repair_owner;
  issuer::IssueResult day7_issued;
  std::optional<DebtAuthorization> authorization;
  std::optional<Day8Install> day8;
  std::set<int> day8_seen_moves;
  int last_move_source = -1;
  MidnightGuard midnight{8, false};

  while (!baseline.done() && !repair.done()) {
    const int step = repair.step_count();
    if (baseline.step_count() != step)
      throw std::runtime_error("paired clock diverged");
    std::array<PlayerAction, 2> baseline_actions;
    for (int player = 0; player < 2; ++player)
      baseline_actions[player] = executor.action_external(
          baseline, player, 0, baseline_states[player]);

    std::array<PlayerAction, 2> repair_actions;
    repair_actions[1 - seat] = executor.action_external(
        repair, 1 - seat, 0, repair_opponent);
    const auto provider_before = repair_owner.native;

    if (step < 168) {
      repair_actions[seat] = executor.action_external(
          repair, seat, 0, repair_owner.native);
      owner::observe_final_provider_action(repair, seat, repair_actions[seat],
                                          repair_owner);
    } else if (repair.day() == 7) {
      if (step == 168) {
        day7_issued = issuer::issue_day_start(
            {&repair, &executor.route_tape(0), seat, generation(seed, 7),
             &repair_owner.persistent_lineage, {}, {}});
        if (!day7_issued.issued())
          throw std::runtime_error("real day7 issuer rejected");
        const auto deferred = std::find_if(
            day7_issued.obligations.begin(), day7_issued.obligations.end(),
            [](const auto& value) {
              return value.actor == 0 && value.source_step == 191 &&
                     value.goal == obligation_day::GoalKind::Feed;
            });
        if (deferred == day7_issued.obligations.end())
          throw std::runtime_error("typed source191 FEED missing");
        owner::DebtSelectionIdentity selected;
        selected.player = seat;
        selected.day = 7;
        selected.issuer_generation = generation(seed, 7);
        selected.obligation_id = deferred->id;
        selected.obligation_content_hash =
            owner::production_obligation_identity_hash(*deferred);
        selected.actor = deferred->actor;
        selected.source_step = deferred->source_step;
        selected.source_action = executor.route_tape(0)
                                     .at(static_cast<std::size_t>(191))
                                     .units.at(0);
        selected.goal = deferred->goal;
        selected.item = deferred->item;
        selected.tile = deferred->tile;
        selected.quantity = deferred->quantity;
        selected.content_hash = owner::debt_selection_identity_hash(selected);
        repair_owner.policy_deferred_authorization = selected;
      }
      fastkag::NativeRepairOptions options;
      options.weed_obligation_day_owner = true;
      commit::Request request{&executor, &repair, seat, 0, options,
                              std::nullopt};
      const auto proposal = commit::propose(request, repair_owner.native);
      repair_actions[seat] = owner::action_external(
          executor, repair, seat, 0, repair_owner, generation(seed, 7));
      if (!repair_owner.plan.certificate)
        throw std::runtime_error("day7 owner certificate missing");
      const auto binding = commit::bind_final_action(
          proposal, 0, generation(seed, 7),
          repair_owner.plan.certificate->content_hash, repair_actions[seat]);
      auto committed_native = proposal.base_state;
      const auto result = commit::commit_weed_owner_finalized(
          request, proposal, binding, repair_actions[seat], committed_native);
      if (!result.committed) {
        ++metrics.commit_failures;
        if (committed_native.last_step != provider_before.last_step)
          ++metrics.provider_state_leaks;
        throw std::runtime_error("day7 final action commit rejected");
      }
      repair_owner.native = std::move(committed_native);
      ++metrics.day7_commits;
    } else if (repair.day() == 8) {
      if (step == 192) {
        // Settle the final real day7 receipt on a wrapper clone, then retain
        // only evaluator-visible completion bookkeeping. Native day8 proposal
        // state is still produced transactionally below from the untouched
        // pre-action NativeAgentState.
        auto settled = repair_owner;
        (void)owner::action_external(executor, repair, seat, 0, settled,
                                     generation(seed, 8));
        if (settled.completed_days.empty())
          throw std::runtime_error("day7 owner did not close at day8 boundary");
        const auto& day7_audit = settled.completed_days.back();
        const auto debt = std::find_if(
            day7_audit.explicit_debts.begin(), day7_audit.explicit_debts.end(),
            [&](const auto& value) {
              return value.obligation_id ==
                     repair_owner.policy_deferred_authorization->obligation_id;
            });
        if (debt == day7_audit.explicit_debts.end())
          throw std::runtime_error("source191 did not close as explicit debt");
        const auto upstream = *repair_owner.policy_deferred_authorization;
        const auto certificate_hash =
            repair_owner.plan.certificate->content_hash;
        authorization = make_authorization(
            seat, 7, generation(seed, 7), certificate_hash, upstream,
            day7_issued, repair);
        metrics.day7_debt_authorized = true;
        metrics.authorization_hash = authorization->content_hash;
        metrics.debt_id = authorization->obligation.id;
        metrics.debt_source_step = authorization->obligation.source_step;
        const auto enrolled = enroll_authorization(*authorization);
        if (!enrolled)
          throw std::runtime_error("day7 authorization registry enrollment failed");
        output.registry = *enrolled;
        output.authorization = *authorization;
        output.day8_start = repair;

        repair_owner.pending = {};
        repair_owner.completed_days = settled.completed_days;
        repair_owner.active_day = -1;
        repair_owner.owned_actor = -1;
        repair_owner.active = false;
        repair_owner.plan = {};
        repair_owner.audit = {};
        repair_owner.remaining_certificate.reset();
        repair_owner.policy_deferred_authorization.reset();

        day8.emplace(install_day8(
            repair, seat, generation(seed, 8), executor, 0,
            repair_owner.persistent_lineage, *authorization,
            output.registry));
        metrics.day8_revalidated = day8->validation.admitted;
        metrics.day8_revalidation_reject = day8->validation.reject;
        metrics.day8_start_x = repair.farms()[seat].farmer.x;
        metrics.day8_start_y = repair.farms()[seat].farmer.y;
        const auto day8_verification = day8->plan.certificate
            ? obligation_day::verify_day_schedule(
                  day8->request, *day8->plan.certificate)
            : obligation_day::VerifyResult{};
        if (!day8->validation.admitted || !day8->plan.planned() ||
            !day8->plan.certificate || !day8_verification.valid)
          throw std::runtime_error(
              "day8 carry DAG did not verify validation=" +
              std::string(reject_name(day8->validation.reject)) +
              " plan_reject=" +
              std::to_string(static_cast<int>(day8->plan.reject)) +
              " verify_failure=" +
              std::to_string(static_cast<int>(
                  day8_verification.failure_reason)) +
              " step=" + std::to_string(day8_verification.failure_step) +
              " obligation=" +
              std::to_string(day8_verification.failure_obligation_id) +
              " carried_wheat=" +
              std::to_string(repair.privates()[seat].inventories[0][0]) +
              " shed_wheat=" +
              std::to_string(repair.privates()[seat].shed[0]));
        for (const auto& replay : day8->plan.certificate->move_replays)
          if (replay.actor == 0) ++metrics.day8_move_expected;
        for (const auto& status :
             day8->plan.certificate->obligation_statuses) {
          if (status.disposition !=
              obligation_day::ObligationDisposition::Completed) {
            ++metrics.day8_new_debts;
            metrics.day8_new_debt_ids.push_back(status.obligation_id);
            const auto obligation = std::find_if(
                day8->request.obligations.begin(),
                day8->request.obligations.end(), [&](const auto& value) {
                  return value.id == status.obligation_id;
                });
            metrics.day8_new_debt_source_steps.push_back(
                obligation == day8->request.obligations.end()
                    ? -2
                    : obligation->source_step);
            metrics.day8_new_debt_goals.push_back(
                obligation == day8->request.obligations.end()
                    ? -1
                    : static_cast<int>(obligation->goal));
            metrics.day8_new_debt_items.push_back(
                obligation == day8->request.obligations.end()
                    ? -2
                    : static_cast<int>(obligation->item));
            metrics.day8_new_debt_x.push_back(
                obligation == day8->request.obligations.end()
                    ? -1
                    : obligation->tile.x);
            metrics.day8_new_debt_y.push_back(
                obligation == day8->request.obligations.end()
                    ? -1
                    : obligation->tile.y);
            metrics.day8_new_debt_dispositions.push_back(
                static_cast<int>(status.disposition));
            metrics.day8_new_debt_remaining.push_back(
                status.remaining_transitions);
          }
        }
      }
      if (!day8 || !day8->plan.certificate)
        throw std::runtime_error("day8 carry owner unavailable");
      const int tick = step - 192;
      fastkag::NativeRepairOptions options;
      options.weed_obligation_day_owner = true;
      commit::Request request{&executor, &repair, seat, 0, options,
                              std::nullopt};
      const auto proposal = commit::propose(request, repair_owner.native);
      auto final_action = proposal.action;
      final_action.units[0] = day8->plan.manifest[0][tick];
      const auto binding = commit::bind_final_action(
          proposal, 0, generation(seed, 8),
          day8->plan.certificate->content_hash, final_action);
      auto committed_native = proposal.base_state;
      const auto result = commit::commit_weed_owner_finalized(
          request, proposal, binding, final_action, committed_native);
      if (!result.committed) {
        ++metrics.commit_failures;
        if (committed_native.last_step != provider_before.last_step)
          ++metrics.provider_state_leaks;
        throw std::runtime_error("day8 final action commit rejected");
      }
      repair_owner.native = std::move(committed_native);
      repair_actions[seat] = final_action;
      ++metrics.day8_commits;
      metrics.action_divergences += !same(final_action, proposal.action);

      Simulator unit_after;
      ++metrics.day8_receipt_checks;
      const bool effect = physical_effect(repair, repair_actions, seat, 0,
                                          final_action.units[0], unit_after);
      if (!effect) {
        ++metrics.day8_receipt_failures;
        throw std::runtime_error("day8 physical precommit receipt rejected");
      }
      const auto& slot = day8->plan.certificate->slots[tick];
      if (movement(final_action.units[0].op)) {
        const int source = slot.sources[0].source_step;
        const auto expected = std::find_if(
            day8->plan.certificate->move_replays.begin(),
            day8->plan.certificate->move_replays.end(),
            [&](const auto& value) {
              return value.actor == 0 && value.source_step == source &&
                     value.emitted_step == step &&
                     same(value.action, final_action.units[0]);
            });
        if (expected == day8->plan.certificate->move_replays.end())
          throw std::runtime_error("day8 MOVE lost certificate source");
        metrics.day8_move_early += step < source;
        metrics.day8_move_duplicates += !day8_seen_moves.insert(source).second;
        if (source <= last_move_source)
          throw std::runtime_error("day8 MOVE source order changed");
        last_move_source = source;
        ++metrics.day8_move_emitted;
      }
      if (slot.obligation_ids[0] == authorization->obligation.id) {
        metrics.source191_emitted_step = step;
        metrics.source191_physical_receipt =
            final_action.units[0].op == Op::FEED && effect;
      }
    } else {
      repair_actions[seat] = executor.action_external(
          repair, seat, 0, repair_owner.native);
    }

    metrics.paired_action_divergences +=
        !same(baseline_actions[seat], repair_actions[seat]);
    const int baseline_unit = fastkag::native_macro_unit_failures(
        baseline, seat, baseline_actions[seat]);
    const int repair_unit = fastkag::native_macro_unit_failures(
        repair, seat, repair_actions[seat]);
    metrics.baseline_unit_failures += baseline_unit;
    metrics.repair_unit_failures += repair_unit;
    if (baseline_unit) metrics.baseline_unit_failure_steps.push_back(step);
    if (repair_unit) metrics.repair_unit_failure_steps.push_back(step);

    auto repair_shadow = repair;
    repair_shadow.step(repair_actions);
    baseline.step(baseline_actions);
    repair.step(repair_actions);
    if (repair_fork::full_unit_phase_state_fingerprint(repair_shadow) !=
        repair_fork::full_unit_phase_state_fingerprint(repair)) {
      ++metrics.day8_receipt_failures;
      throw std::runtime_error("actual full step differed from receipt shadow");
    }
    const int baseline_market = fastkag::native_macro_market_failures(
        baseline, seat, baseline_actions[seat]);
    const int repair_market = fastkag::native_macro_market_failures(
        repair, seat, repair_actions[seat]);
    metrics.baseline_market_failures += baseline_market;
    metrics.repair_market_failures += repair_market;
    if (baseline_market) metrics.baseline_market_failure_steps.push_back(step);
    if (repair_market) metrics.repair_market_failure_steps.push_back(step);
    ++metrics.steps;

    if (step == 215) {
      metrics.day8_move_drops =
          std::max(0, metrics.day8_move_expected -
                          metrics.day8_move_emitted);
      if (!midnight.close(8))
        throw std::runtime_error("day8 midnight closure duplicated");
      const auto status = std::find_if(
          day8->plan.certificate->obligation_statuses.begin(),
          day8->plan.certificate->obligation_statuses.end(),
          [&](const auto& value) {
            return value.obligation_id == authorization->obligation.id;
          });
      metrics.source191_completed =
          status != day8->plan.certificate->obligation_statuses.end() &&
          status->disposition ==
              obligation_day::ObligationDisposition::Completed &&
          metrics.source191_physical_receipt;
      const auto finish = finish_authorization(*authorization, 8,
                                               metrics.source191_completed);
      metrics.source191_finish_disposition = finish.disposition;
      if (!finish.explicitly_closed)
        throw std::runtime_error("day8 carry lacked explicit finish status");
      if (stop_after_day8) break;
    }
  }
  if (!stop_after_day8 && (!baseline.done() || !repair.done()))
    throw std::runtime_error("whole-game pair terminated early");
  metrics.baseline_own = baseline.farms()[seat].money;
  metrics.baseline_opponent = baseline.farms()[1 - seat].money;
  metrics.repair_own = repair.farms()[seat].money;
  metrics.repair_opponent = repair.farms()[1 - seat].money;
  metrics.own_delta = metrics.repair_own - metrics.baseline_own;
  const double baseline_margin =
      metrics.baseline_own - metrics.baseline_opponent;
  const double repair_margin = metrics.repair_own - metrics.repair_opponent;
  metrics.margin_delta = repair_margin - baseline_margin;
  const double baseline_score = baseline_margin > 0 ? 1 : baseline_margin < 0 ? 0 : .5;
  const double repair_score = repair_margin > 0 ? 1 : repair_margin < 0 ? 0 : .5;
  metrics.score_delta = repair_score - baseline_score;
  return output;
}

void write_ints(std::ostream& out, const std::vector<int>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

void write_ids(std::ostream& out, const std::vector<std::uint64_t>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

const char* finish_name(FinishDisposition disposition) {
  switch (disposition) {
    case FinishDisposition::Completed: return "completed";
    case FinishDisposition::Carried: return "carried";
    case FinishDisposition::Expired: return "expired";
  }
  return "unknown";
}

const char* goal_name(int goal) {
  using Goal = obligation_day::GoalKind;
  switch (static_cast<Goal>(goal)) {
    case Goal::CropReady: return "crop_ready";
    case Goal::Pickup: return "pickup";
    case Goal::Place: return "place";
    case Goal::Feed: return "feed";
    case Goal::Care: return "care";
    case Goal::Harvest: return "harvest";
    case Goal::BuildPasture: return "build_pasture";
    case Goal::CollectFertilizer: return "collect_fertilizer";
    case Goal::BuildCoop: return "build_coop";
  }
  return "unknown";
}

const char* disposition_name(int disposition) {
  using Value = obligation_day::ObligationDisposition;
  switch (static_cast<Value>(disposition)) {
    case Value::Completed: return "completed";
    case Value::BlockedOnReceipt: return "blocked_on_receipt";
    case Value::CapacityDebt: return "capacity_debt";
    case Value::DependencyDebt: return "dependency_debt";
    case Value::DeadlineDebt: return "deadline_debt";
    case Value::UnsupportedStateDebt: return "unsupported_state_debt";
    case Value::PolicyDeferredDebt: return "policy_deferred_debt";
  }
  return "unknown";
}

}  // namespace

std::uint64_t authorization_hash(
    const DebtAuthorization& authorization) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  mix(hash, authorization.player);
  mix(hash, authorization.original_day);
  mix(hash, authorization.valid_day);
  mix(hash, authorization.expires_day);
  mix(hash, authorization.original_issuer_generation);
  mix(hash, authorization.original_certificate_hash);
  mix(hash, authorization.upstream_selection_hash);
  mix(hash, authorization.obligation_identity_hash);
  mix_action(hash, authorization.original_source_action);
  mix(hash, authorization.obligation.dependencies.size());
  for (const auto dependency : authorization.obligation.dependencies)
    mix(hash, dependency);
  mix(hash, static_cast<std::uint8_t>(authorization.obligation.resource.item));
  mix(hash, authorization.obligation.resource.seed_quantity);
  mix(hash, authorization.obligation.resource.shed_quantity);
  mix(hash, authorization.obligation.resource.carried_quantity);
  mix(hash, authorization.dependencies.size());
  for (const auto& dependency : authorization.dependencies) {
    mix(hash, dependency.identity_hash);
    mix(hash, dependency.obligation.id);
  }
  mix(hash, authorization.day7_noncompletion_evidence_hash);
  mix(hash, authorization.authorization_generation);
  return hash == 0 ? 1 : hash;
}

std::optional<PersistentDebtRegistry> enroll_authorization(
    const DebtAuthorization& authorization) {
  if (authorization.content_hash == 0 ||
      authorization.content_hash != authorization_hash(authorization))
    return std::nullopt;
  PersistentDebtRegistry registry;
  registry.canonical_hash_ = authorization.content_hash;
  return registry;
}

RevalidationResult revalidate(
    const DebtAuthorization& authorization,
    const PersistentDebtRegistry& registry,
    const NativeTeammateExecutor& executor, const Simulator& observation,
    int route) {
  RevalidationResult output;
  const auto reject = [&](RevalidationReject reason) {
    output.reject = reason;
    return output;
  };
  if (authorization.player < 0 || authorization.player >= 2 ||
      authorization.original_day < 0 || authorization.valid_day < 0 ||
      authorization.expires_day < authorization.valid_day ||
      authorization.original_issuer_generation == 0 ||
      authorization.original_certificate_hash == 0 ||
      authorization.upstream_selection_hash == 0 ||
      authorization.obligation.id == 0 ||
      authorization.obligation.source_step < 0 ||
      authorization.obligation.actor < 0 ||
      authorization.day7_noncompletion_evidence_hash == 0 ||
      authorization.authorization_generation == 0)
    return reject(RevalidationReject::InvalidEnvelope);
  if (authorization.content_hash != authorization_hash(authorization) ||
      authorization.obligation_identity_hash !=
          owner::production_obligation_identity_hash(
              authorization.obligation))
    return reject(RevalidationReject::Integrity);
  if (!registry.enrolled() ||
      authorization.content_hash != registry.canonical_hash())
    return reject(RevalidationReject::RegistryMismatch);
  if (observation.day() != authorization.valid_day ||
      observation.day() > authorization.expires_day || observation.hour() != 0)
    return reject(RevalidationReject::Stale);
  if (route < 0 || route >= executor.route_count() ||
      authorization.obligation.source_step >=
          static_cast<int>(executor.route_tape(route).size()) ||
      authorization.obligation.actor >=
          static_cast<int>(executor.route_tape(route)
                               .at(static_cast<std::size_t>(
                                   authorization.obligation.source_step))
                               .units.size()) ||
      !same(executor.route_tape(route)
                .at(static_cast<std::size_t>(
                    authorization.obligation.source_step))
                .units.at(static_cast<std::size_t>(
                    authorization.obligation.actor)),
            authorization.original_source_action))
    return reject(RevalidationReject::RouteSourceMismatch);
  if (authorization.obligation.actor >
      static_cast<int>(observation.farms()[authorization.player].hands.size()))
    return reject(RevalidationReject::ActorUnavailable);
  if (goal_satisfied(observation, authorization.player,
                     authorization.obligation))
    return reject(RevalidationReject::AlreadySatisfied);
  if (!target_applicable(observation, authorization.player,
                         authorization.obligation))
    return reject(RevalidationReject::TargetMismatch);
  if (authorization.dependencies.size() !=
      authorization.obligation.dependencies.size())
    return reject(RevalidationReject::DependencyMismatch);
  for (std::size_t i = 0; i < authorization.dependencies.size(); ++i) {
    const auto& dependency = authorization.dependencies[i];
    if (dependency.obligation.id != authorization.obligation.dependencies[i] ||
        dependency.identity_hash !=
            owner::production_obligation_identity_hash(
                dependency.obligation) ||
        !goal_satisfied(observation, authorization.player,
                        dependency.obligation))
      return reject(RevalidationReject::DependencyMismatch);
  }
  if (!base_resource_available(observation, authorization.player,
                               authorization.obligation))
    return reject(RevalidationReject::ResourceUnavailable);
  output.carried = authorization.obligation;
  rebase(output.carried, observation.step_count(),
         observation.step_count() + 23);
  // An admitted prior-day must-finish debt outranks same-day stationary work;
  // immutable MOVE tokens remain hard and are merely delayed.
  output.carried.priority += 2000;
  const int resource_item = static_cast<int>(authorization.obligation.resource.item);
  const int required = authorization.obligation.resource.carried_quantity;
  const auto& private_state = observation.privates()[authorization.player];
  const int carried_now = resource_item >= 0 && resource_item < fastkag::N_ITEMS &&
                                  authorization.obligation.actor >= 0 &&
                                  authorization.obligation.actor <
                                      static_cast<int>(private_state.inventories.size())
                              ? private_state.inventories[
                                    authorization.obligation.actor][resource_item]
                              : 0;
  if (required > carried_now) {
    const int missing = required - carried_now;
    if (resource_item < 0 || resource_item >= fastkag::N_ITEMS ||
        private_state.shed[resource_item] < missing)
      return reject(RevalidationReject::ResourceUnavailable);
    obligation_day::ProductionObligation pickup;
    pickup.id = authorization.obligation.id ^ 0x4000000000000000ULL;
    pickup.actor = authorization.obligation.actor;
    pickup.tile = authorization.obligation.tile;
    pickup.goal = obligation_day::GoalKind::Pickup;
    pickup.item = authorization.obligation.resource.item;
    pickup.quantity = missing;
    pickup.resource = {authorization.obligation.resource.item, 0, missing, 0};
    pickup.earliest_step = observation.step_count();
    pickup.deadline_step = observation.step_count() + 23;
    pickup.priority = output.carried.priority + 1;
    pickup.must_finish_today = true;
    pickup.repair = true;
    pickup.source_step = -1;
    output.dependency_witnesses.push_back(std::move(pickup));
  }
  output.admitted = true;
  return output;
}

const char* reject_name(RevalidationReject reject) noexcept {
  switch (reject) {
    case RevalidationReject::None: return "none";
    case RevalidationReject::InvalidEnvelope: return "invalid_envelope";
    case RevalidationReject::Integrity: return "integrity";
    case RevalidationReject::RegistryMismatch: return "registry_mismatch";
    case RevalidationReject::Stale: return "stale";
    case RevalidationReject::RouteSourceMismatch: return "route_source_mismatch";
    case RevalidationReject::ActorUnavailable: return "actor_unavailable";
    case RevalidationReject::TargetMismatch: return "target_mismatch";
    case RevalidationReject::DependencyMismatch: return "dependency_mismatch";
    case RevalidationReject::ResourceUnavailable: return "resource_unavailable";
    case RevalidationReject::AlreadySatisfied: return "already_satisfied";
  }
  return "unknown";
}

FinishResult finish_authorization(const DebtAuthorization& authorization,
                                  int finished_day,
                                  bool physically_completed) {
  FinishResult output;
  output.explicitly_closed = true;
  if (physically_completed) {
    output.disposition = FinishDisposition::Completed;
    return output;
  }
  if (finished_day >= authorization.expires_day) {
    output.disposition = FinishDisposition::Expired;
    return output;
  }
  output.disposition = FinishDisposition::Carried;
  output.next = authorization;
  output.next.valid_day = finished_day + 1;
  ++output.next.authorization_generation;
  output.next.content_hash = authorization_hash(output.next);
  return output;
}

Report evaluate(std::string tapes, std::string library, std::uint64_t seed,
                int seat) {
  fastkag::NativeTapeLibrary tape_library;
  tape_library.routes.push_back(
      repair::load_route(tapes, library, "G001"));
  NativeTeammateExecutor executor(std::move(tape_library));
  Report report;
  report.default_off_parity = run_default_off(executor, seed);
  auto run = run_game(executor, seed, seat);
  report.game = run.metrics;
  auto tampered = run.authorization;
  tampered.content_hash ^= 1;
  report.tamper_rejected =
      revalidate(tampered, run.registry, executor, run.day8_start, 0)
          .reject == RevalidationReject::Integrity;
  auto wrong_tile = run.authorization;
  ++wrong_tile.obligation.tile.x;
  wrong_tile.obligation_identity_hash =
      owner::production_obligation_identity_hash(wrong_tile.obligation);
  wrong_tile.content_hash = authorization_hash(wrong_tile);
  report.wrong_tile_rejected =
      revalidate(wrong_tile, run.registry, executor, run.day8_start, 0)
          .reject == RevalidationReject::RegistryMismatch;
  auto wrong_item = run.authorization;
  wrong_item.obligation.item = Item::SHEEP;
  wrong_item.obligation_identity_hash =
      owner::production_obligation_identity_hash(wrong_item.obligation);
  wrong_item.content_hash = authorization_hash(wrong_item);
  report.wrong_item_rejected =
      revalidate(wrong_item, run.registry, executor, run.day8_start, 0)
          .reject == RevalidationReject::RegistryMismatch;
  auto stale_state = run.day8_start;
  std::array<PlayerAction, 2> pass;
  while (stale_state.day() == 8) stale_state.step(pass);
  report.stale_rejected =
      revalidate(run.authorization, run.registry, executor, stale_state, 0)
          .reject == RevalidationReject::Stale;
  MidnightGuard guard{8, false};
  report.midnight_duplicate_rejected = guard.close(8) && !guard.close(8);
  report.no_terminal_oracle = true;
  return report;
}

Metrics evaluate_focused(std::string tapes, std::string library,
                         std::uint64_t seed, int seat) {
  fastkag::NativeTapeLibrary tape_library;
  tape_library.routes.push_back(
      repair::load_route(tapes, library, "G001"));
  NativeTeammateExecutor executor(std::move(tape_library));
  return run_game(executor, seed, seat, true).metrics;
}

std::string Report::json() const {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6)
      << "{\"schema\":\"cross-day-debt-carry-v1\",\"scope\":{"
         "\"default_off\":true,\"terminal_oracle_used\":false,"
         "\"market_takeover\":false,"
         "\"day8_fixed_certificate_smoke\":true,"
         "\"intraday_rolling_deployment_ready\":false,"
         "\"persistent_registry_required\":true},\"tests\":{"
      << "\"default_off_parity\":"
      << (default_off_parity ? "true" : "false")
      << ",\"stale_rejected\":" << (stale_rejected ? "true" : "false")
      << ",\"tamper_rejected\":" << (tamper_rejected ? "true" : "false")
      << ",\"wrong_tile_rejected\":"
      << (wrong_tile_rejected ? "true" : "false")
      << ",\"wrong_item_rejected\":"
      << (wrong_item_rejected ? "true" : "false")
      << ",\"midnight_duplicate_rejected\":"
      << (midnight_duplicate_rejected ? "true" : "false")
      << "},\"game\":{\"seed\":" << game.seed << ",\"seat\":"
      << game.seat << ",\"steps\":" << game.steps
      << ",\"actions\":{\"same_observation_divergences\":"
      << game.action_divergences << ",\"paired_divergences\":"
      << game.paired_action_divergences << "},\"commits\":{\"day7\":"
      << game.day7_commits << ",\"day8\":" << game.day8_commits
      << ",\"failures\":" << game.commit_failures
      << ",\"state_leaks\":" << game.provider_state_leaks
      << "},\"authorization\":{\"created\":"
      << (game.day7_debt_authorized ? "true" : "false")
      << ",\"hash\":" << game.authorization_hash
      << ",\"obligation_id\":" << game.debt_id
      << ",\"original_source_step\":" << game.debt_source_step
      << ",\"day8_revalidated\":"
      << (game.day8_revalidated ? "true" : "false")
      << ",\"reject\":\""
      << reject_name(game.day8_revalidation_reject) << "\"}"
      << ",\"source191\":{\"emitted_step\":"
      << game.source191_emitted_step << ",\"physical_receipt\":"
      << (game.source191_physical_receipt ? "true" : "false")
      << ",\"completed\":"
      << (game.source191_completed ? "true" : "false")
      << ",\"finish_disposition\":\""
      << finish_name(game.source191_finish_disposition) << "\""
      << "},\"day8_receipts\":{\"checks\":"
      << game.day8_receipt_checks << ",\"failures\":"
      << game.day8_receipt_failures << "},\"day8_moves\":{\"expected\":"
      << game.day8_move_expected << ",\"emitted\":"
      << game.day8_move_emitted << ",\"early\":" << game.day8_move_early
      << ",\"duplicates\":" << game.day8_move_duplicates
      << ",\"drops\":" << game.day8_move_drops
      << "},\"day8_terminal_debts\":{\"count\":"
      << game.day8_new_debts << ",\"ids\":";
  write_ids(out, game.day8_new_debt_ids);
  out << ",\"source_steps\":";
  write_ints(out, game.day8_new_debt_source_steps);
  out << ",\"details\":[";
  for (std::size_t i = 0; i < game.day8_new_debt_ids.size(); ++i) {
    if (i) out << ',';
    out << "{\"id\":" << game.day8_new_debt_ids[i]
        << ",\"source_step\":" << game.day8_new_debt_source_steps[i]
        << ",\"goal\":\"" << goal_name(game.day8_new_debt_goals[i])
        << "\",\"item\":" << game.day8_new_debt_items[i]
        << ",\"tile\":[" << game.day8_new_debt_x[i] << ','
        << game.day8_new_debt_y[i] << "],\"disposition\":\""
        << disposition_name(game.day8_new_debt_dispositions[i])
        << "\",\"remaining_transitions\":"
        << game.day8_new_debt_remaining[i] << '}';
  }
  out << "]},\"failure_chains\":{\"baseline_unit\":"
      << game.baseline_unit_failures << ",\"repair_unit\":"
      << game.repair_unit_failures << ",\"baseline_market\":"
      << game.baseline_market_failures << ",\"repair_market\":"
      << game.repair_market_failures << ",\"baseline_unit_steps\":";
  write_ints(out, game.baseline_unit_failure_steps);
  out << ",\"repair_unit_steps\":";
  write_ints(out, game.repair_unit_failure_steps);
  out << ",\"baseline_market_steps\":";
  write_ints(out, game.baseline_market_failure_steps);
  out << ",\"repair_market_steps\":";
  write_ints(out, game.repair_market_failure_steps);
  out << "},\"terminal\":{\"baseline\":{\"own\":"
      << game.baseline_own << ",\"opponent\":" << game.baseline_opponent
      << "},\"repair\":{\"own\":" << game.repair_own
      << ",\"opponent\":" << game.repair_opponent
      << "},\"delta\":{\"own\":" << game.own_delta
      << ",\"margin\":" << game.margin_delta
      << ",\"score\":" << game.score_delta << "}}}}";
  return out.str();
}

}  // namespace g001::cross_day_debt_carry
