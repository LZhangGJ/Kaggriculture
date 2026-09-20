#include "minimum_damage_remaining_day_audit_adapter.hpp"
#include "native_final_action_commit.hpp"
#include "purchase_failure_day_rolling_owner.hpp"
#include "repair_composer_arbiter.hpp"
#include "route_loader.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace arb = g001::repair_composer_arbiter;
namespace rolling = g001::purchase_failure_rolling;
namespace final_commit = g001::native_final_commit;
namespace weed = g001::minimum_damage_remaining_audit;
namespace bridge = g001::minimum_damage_bridge;
namespace issuer = g001::day_start_issuer;
namespace obligation_day = g001::obligation_day;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
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

const fastkag::Tile* tile_at(const fastkag::Simulator& env, int player,
                             fastkag::Position position) {
  const int board = env.config().board_size;
  if (position.x < 0 || position.y < 0 || position.x >= board ||
      position.y >= board)
    return nullptr;
  const auto index = static_cast<std::size_t>(position.y * board + position.x);
  const auto& tiles = env.farms()[player].tiles;
  return index < tiles.size() ? &tiles[index] : nullptr;
}

arb::Intent unit_intent(arb::Source source, std::uint64_t id, int actor,
                        Action base, Action replacement, bool must,
                        int cascade, int damage) {
  arb::Intent result;
  result.source = source;
  result.id = id;
  result.debt_id = id + 1000;
  result.domain = arb::Domain::Unit;
  result.actor = actor;
  result.expected_base = base;
  result.replacement = replacement;
  result.deadline = 30;
  result.must_finish = must;
  result.important = must;
  result.cascade_risk = cascade;
  result.displacement_damage = damage;
  result.provenance = "adversarial_typed_unit";
  return result;
}

arb::Intent market_intent(arb::Source source, std::uint64_t id, int slot,
                          Action replacement, int cash, int capacity,
                          bool must, int cascade) {
  arb::Intent result;
  result.source = source;
  result.id = id;
  result.debt_id = id + 1000;
  result.domain = arb::Domain::Market;
  result.market_slot = slot;
  result.expected_base = {};
  result.replacement = replacement;
  result.cash_cost = cash;
  result.shed_capacity_cost = capacity;
  result.deadline = 30;
  result.must_finish = must;
  result.important = must;
  result.cascade_risk = cascade;
  result.displacement_damage = 1;
  result.provenance = "adversarial_typed_market";
  return result;
}

arb::ParticipantProposal proposal(arb::Source source, std::uint64_t token,
                                  std::uint64_t observation_hash,
                                  const PlayerAction& base,
                                  std::vector<arb::Intent> intents) {
  arb::ParticipantProposal result;
  result.source = source;
  result.generation = 1;
  result.observation_hash = observation_hash;
  result.commit_token = token;
  result.base_action = base;
  result.proposed_action = base;
  result.intents = std::move(intents);
  for (const auto& intent : result.intents) {
    if (intent.domain == arb::Domain::Unit) {
      result.proposed_action.units[intent.actor] = intent.replacement;
    } else {
      while (result.proposed_action.market.size() <=
             static_cast<std::size_t>(intent.market_slot))
        result.proposed_action.market.push_back({});
      result.proposed_action.market[intent.market_slot] = intent.replacement;
    }
  }
  return result;
}

struct TxAudit {
  int preflight{};
  int commit{};
  int abort{};
};

arb::ParticipantTransaction tx(arb::Source source, std::uint64_t token,
                               TxAudit& audit, bool accept = true) {
  return {source,
          1,
          token,
          [&audit, accept] {
            ++audit.preflight;
            return accept;
          },
          [&audit]() noexcept { ++audit.commit; },
          [&audit]() noexcept { ++audit.abort; }};
}

arb::PrepareRequest joint_request() {
  arb::PrepareRequest request;
  request.enabled = true;
  request.step = 12;
  request.observation_hash = 0xabc;
  request.base_action.units = {{Op::PASS}, {Op::PASS}};
  auto weed_intent = unit_intent(
      arb::Source::WeedMinimumDamage, 101, 0, {Op::PASS},
      {Op::BUILD_PASTURE}, true, 8, 2);
  request.weed = proposal(arb::Source::WeedMinimumDamage, 11,
                          request.observation_hash, request.base_action,
                          {weed_intent});
  auto purchase_intent = market_intent(
      arb::Source::PurchaseFailure, 201, 0,
      {Op::BUY_ANIMAL, Item::COW, 1}, 400, 1, true, 5);
  request.purchase = proposal(arb::Source::PurchaseFailure, 22,
                              request.observation_hash, request.base_action,
                              {purchase_intent});
  request.available_cash = 500;
  request.free_shed_capacity = 2;
  return request;
}

void synthetic_joint_atomicity_and_tamper() {
  arb::Composer composer;
  const auto before_generation = composer.generation();
  const auto prepared = composer.prepare(joint_request());
  require(prepared.status == arb::PrepareStatus::Accepted &&
              prepared.selected_sources.size() == 2 &&
              prepared.final_action.units[0].op == Op::BUILD_PASTURE &&
              prepared.final_action.market.size() == 1 &&
              prepared.final_action.market[0].op == Op::BUY_ANIMAL,
          "non-conflicting joint set was not composed");
  require(composer.generation() == before_generation,
          "prepare mutated composer generation");

  TxAudit weed_audit;
  TxAudit purchase_audit;
  auto tampered = prepared.final_action;
  tampered.market[0].quantity = 2;
  require(composer.finalize(prepared, tampered,
                            {tx(arb::Source::WeedMinimumDamage, 11, weed_audit),
                             tx(arb::Source::PurchaseFailure, 22,
                                purchase_audit)}) ==
              arb::FinalizeStatus::Tampered &&
              weed_audit.commit == 0 && purchase_audit.commit == 0,
          "tamper did not abort both participants");

  weed_audit = {};
  purchase_audit = {};
  require(composer.finalize(
              prepared, prepared.final_action,
              {tx(arb::Source::WeedMinimumDamage, 11, weed_audit)}) ==
              arb::FinalizeStatus::PartialCommit &&
              weed_audit.commit == 0,
          "partial participant set committed");

  weed_audit = {};
  purchase_audit = {};
  require(composer.finalize(
              prepared, prepared.final_action,
              {tx(arb::Source::WeedMinimumDamage, 11, weed_audit, false),
               tx(arb::Source::PurchaseFailure, 22, purchase_audit)}) ==
              arb::FinalizeStatus::ParticipantRejected &&
              weed_audit.commit == 0 && purchase_audit.commit == 0,
          "failed preflight caused partial commit");

  weed_audit = {};
  purchase_audit = {};
  TxAudit rogue;
  require(composer.finalize(
              prepared, prepared.final_action,
              {tx(arb::Source::WeedMinimumDamage, 11, weed_audit),
               tx(arb::Source::PurchaseFailure, 22, purchase_audit),
               tx(arb::Source::WeedMinimumDamage, 999, rogue)}) ==
              arb::FinalizeStatus::Selected &&
              weed_audit.commit == 1 && purchase_audit.commit == 1 &&
              rogue.commit == 0 && rogue.abort == 1,
          "exact joint selection did not atomically commit");
  require(composer.finalize(prepared, prepared.final_action, {}) ==
              arb::FinalizeStatus::AlreadyFinalized,
          "double finalize was not rejected");
}

bool graph_has(const arb::Prepared& prepared, arb::ConflictReason reason) {
  return std::any_of(prepared.conflict_graph.begin(),
                     prepared.conflict_graph.end(),
                     [reason](const auto& conflict) {
                       return conflict.reason == reason;
                     });
}

void adversarial_conflict_graph_and_default_off() {
  auto request = joint_request();
  auto purchase_unit = unit_intent(
      arb::Source::PurchaseFailure, 202, 0, {Op::PASS},
      {Op::PLANT, Item::WHEAT, 1}, true, 2, 1);
  purchase_unit.resource_item = Item::WHEAT;
  purchase_unit.resource_consumption = 1;
  request.purchase = proposal(arb::Source::PurchaseFailure, 23,
                              request.observation_hash, request.base_action,
                              {purchase_unit});
  request.weed->intents[0].resource_item = Item::WHEAT;
  request.weed->intents[0].resource_consumption = 1;
  request.available_items[Item::WHEAT] = 1;
  arb::Composer composer;
  const auto conflict = composer.prepare(request);
  require(conflict.accepted() && graph_has(conflict, arb::ConflictReason::SameActor) &&
              graph_has(conflict, arb::ConflictReason::ItemResource) &&
              conflict.selected_sources.size() == 1 &&
              conflict.selected_sources[0] == arb::Source::WeedMinimumDamage,
          "same actor/item conflict did not deterministically yield to weed");

  auto capacity = joint_request();
  capacity.available_cash = 399;
  capacity.free_shed_capacity = 0;
  const auto constrained = composer.prepare(capacity);
  require(graph_has(constrained, arb::ConflictReason::Cash) &&
              graph_has(constrained, arb::ConflictReason::ShedCapacity),
          "cash/capacity conflicts missing from graph");

  auto dependency = joint_request();
  dependency.purchase->intents[0].dependencies = {9999};
  dependency.purchase->intents[0].deadline = 11;
  const auto blocked = composer.prepare(dependency);
  require(graph_has(blocked, arb::ConflictReason::Dependency) &&
              graph_has(blocked, arb::ConflictReason::Deadline),
          "dependency/deadline conflicts missing from graph");

  arb::PrepareRequest move;
  move.enabled = true;
  move.step = 1;
  move.observation_hash = 9;
  move.base_action.units = {{Op::EAST}};
  auto illegal = unit_intent(arb::Source::PurchaseFailure, 301, 0,
                             {Op::EAST}, {Op::PLANT, Item::WHEAT}, true, 1, 1);
  move.purchase = proposal(arb::Source::PurchaseFailure, 33,
                           move.observation_hash, move.base_action, {illegal});
  move.available_cash = 1000;
  move.free_shed_capacity = 10;
  const auto move_reject = composer.prepare(move);
  require(graph_has(move_reject, arb::ConflictReason::PurchaseMoveEdit) &&
              same(move_reject.final_action, move.base_action),
          "purchase MOVE edit was not fail-closed");

  auto disabled = joint_request();
  disabled.enabled = false;
  const auto off = composer.prepare(disabled);
  require(off.status == arb::PrepareStatus::Disabled &&
              same(off.final_action, disabled.base_action) &&
              off.selected_sources.empty(),
          "default-off was not exact/no-owner");
  require(composer.finalize(off, disabled.base_action, {}) ==
              arb::FinalizeStatus::Selected,
          "default-off exact action did not finalize");

  arb::Composer slot_composer;
  auto slot = joint_request();
  auto weed_market = market_intent(arb::Source::WeedMinimumDamage, 401, 0,
                                   {Op::BUY_SEED, Item::WHEAT, 1}, 10, 0,
                                   true, 9);
  slot.weed = proposal(arb::Source::WeedMinimumDamage, 41,
                       slot.observation_hash, slot.base_action, {weed_market});
  const auto slot_conflict = slot_composer.prepare(slot);
  require(graph_has(slot_conflict, arb::ConflictReason::SameMarketSlot),
          "same market slot conflict missing from graph");

  auto unbound = joint_request();
  unbound.purchase->proposed_action.units[1] = {Op::CARE};
  const auto invalid = slot_composer.prepare(unbound);
  require(invalid.status == arb::PrepareStatus::InvalidInput,
          "untyped action diff was admitted");

  auto zero_token = joint_request();
  zero_token.purchase->commit_token = 0;
  require(slot_composer.prepare(zero_token).status ==
              arb::PrepareStatus::InvalidInput,
          "zero participant token was admitted");

  auto aliased = joint_request();
  aliased.purchase->intents[0].id = aliased.weed->intents[0].id;
  require(slot_composer.prepare(aliased).status ==
              arb::PrepareStatus::InvalidInput,
          "cross-source intent id alias was admitted");

  auto wrong_market_base = joint_request();
  wrong_market_base.purchase->intents[0].expected_base = {Op::SELL, Item::MILK, 1};
  require(slot_composer.prepare(wrong_market_base).status ==
              arb::PrepareStatus::InvalidInput,
          "market intent with wrong expected base was admitted");

  auto shrink = joint_request();
  shrink.base_action.market = {{Op::SELL, Item::MILK, 1},
                               {Op::SELL, Item::EGG, 1}};
  shrink.weed.reset();
  shrink.purchase->base_action = shrink.base_action;
  shrink.purchase->proposed_action.market = {
      {Op::SELL, Item::MILK, 1}};
  shrink.purchase->intents.clear();
  require(slot_composer.prepare(shrink).status ==
              arb::PrepareStatus::InvalidInput,
          "market shrink was admitted without lossless slot identity");

  auto reorder = joint_request();
  reorder.base_action.market = {{Op::SELL, Item::MILK, 1},
                                {Op::SELL, Item::EGG, 1}};
  reorder.weed.reset();
  auto first = market_intent(arb::Source::PurchaseFailure, 501, 0,
                             {Op::SELL, Item::EGG, 1}, 0, 0, true, 1);
  auto second = market_intent(arb::Source::PurchaseFailure, 502, 1,
                              {Op::SELL, Item::MILK, 1}, 0, 0, true, 1);
  first.expected_base = reorder.base_action.market[0];
  second.expected_base = reorder.base_action.market[1];
  reorder.purchase = proposal(arb::Source::PurchaseFailure, 51,
                              reorder.observation_hash, reorder.base_action,
                              {first, second});
  require(slot_composer.prepare(reorder).selected_intents.size() == 2,
          "fully typed market reorder was not admitted");
}

std::uint64_t weed_generation(std::uint64_t seed, int day) {
  return (seed << 20) | (2ULL << 16) |
         static_cast<std::uint64_t>(day + 1);
}

fastkag::NativeTapeLibrary library() {
  fastkag::NativeTapeLibrary result;
  result.routes = {
      g001::repair::load_route(ARBITER_G001_TAPES, ARBITER_G001_LIBRARY,
                               "G001"),
      g001::repair::load_route(ARBITER_G001_TAPES, ARBITER_G001_LIBRARY,
                               "G096")};
  return result;
}

void real_weed_minimum_damage_shadow() {
  constexpr std::uint64_t seed = 970017;
  fastkag::NativeTeammateExecutor executor(library());
  fastkag::Simulator env({}, seed);
  std::array<fastkag::NativeAgentState, 2> states;
  while (env.step_count() < 168) {
    std::array<PlayerAction, 2> actions;
    for (int player = 0; player < 2; ++player)
      actions[player] = executor.action_external(env, player, 0, states[player]);
    env.step(actions);
  }
  issuer::PersistentRouteIntentRegistry lineage;
  const auto issued = issuer::issue_day_start(
      {&env, &executor.route_tape(0), 1, weed_generation(seed, 7), &lineage,
       {}, {}});
  require(issued.issued(), "real weed day7 issuer rejected");
  int weed_obligations = 0;
  for (const auto& obligation : issued.obligations) {
    const auto* tile = tile_at(env, 1, obligation.tile);
    if (obligation.goal == obligation_day::GoalKind::BuildPasture && tile &&
        tile->kind == fastkag::TileKind::WEED)
      ++weed_obligations;
  }
  require(weed_obligations > 0,
          "seed970017 day7 did not expose a physical weed obligation");
  std::vector<bridge::ObligationPolicy> policies;
  for (const auto& obligation : issued.obligations)
    if (obligation.actor == 0)
      policies.push_back({obligation.id, true, 2});
  weed::Adapter adapter(issued, 1, weed_generation(seed, 7), policies);
  const auto hand = adapter.plan(env);
  require(hand.valid, "real minimum-damage hand rejected");
  auto provider_state = states[1];
  const auto base = executor.action_external(env, 1, 0, provider_state);
  require(!base.units.empty(), "real weed base has no actor0");

  arb::PrepareRequest request;
  request.enabled = true;
  request.step = env.step_count();
  request.observation_hash = hand.certificate_hash;
  request.base_action = base;
  auto intent = unit_intent(arb::Source::WeedMinimumDamage,
                            hand.certificate_hash, 0, base.units[0],
                            hand.candidate_unit, true, 4, 1);
  intent.deadline = 191;
  intent.certified_weed_move_owner = true;
  request.weed = proposal(arb::Source::WeedMinimumDamage,
                          hand.certificate_hash, request.observation_hash,
                          base, {intent});
  request.weed->generation = weed_generation(seed, 7);
  request.available_cash = static_cast<int>(env.farms()[1].money);
  request.free_shed_capacity = env.config().shed_capacity;
  arb::Composer composer;
  const auto prepared = composer.prepare(request);
  require(prepared.accepted() && prepared.selected_sources.size() == 1 &&
              same(prepared.final_action.units[0], hand.candidate_unit),
          "real weed proposal was not wired into arbiter");
  std::optional<weed::Adapter> staged_adapter;
  bool adapter_committed = false;
  auto opponent_state = states[0];
  const auto opponent = executor.action_external(env, 0, 0, opponent_state);
  arb::ParticipantTransaction transaction;
  transaction.source = arb::Source::WeedMinimumDamage;
  transaction.generation = weed_generation(seed, 7);
  transaction.commit_token = hand.certificate_hash;
  transaction.preflight = [&] {
    staged_adapter = adapter;
    auto shadow = env;
    std::array<PlayerAction, 2> actions;
    actions[0] = opponent;
    actions[1] = prepared.final_action;
    shadow.step(actions);
    return staged_adapter->observe_final(env, prepared.final_action.units[0],
                                         shadow);
  };
  transaction.commit = [&]() noexcept {
    adapter = std::move(*staged_adapter);
    adapter_committed = true;
  };
  transaction.abort = [&]() noexcept { staged_adapter.reset(); };
  require(composer.finalize(prepared, prepared.final_action,
                            {std::move(transaction)}) ==
                  arb::FinalizeStatus::Selected &&
              adapter_committed && adapter.finish().hands == 1,
          "real weed adapter state did not commit through arbiter");
  std::cout << "real_weed_shadow seed=970017 step=" << env.step_count()
            << " actor=0 candidate_op="
            << static_cast<int>(hand.candidate_unit.op)
            << " weed_obligations=" << weed_obligations
            << " conflicts=" << prepared.conflict_graph.size() << '\n';
}

std::uint64_t purchase_token(const rolling::Proposal& proposal) {
  return proposal.binding_hash;
}

void real_purchase_shadow() {
  constexpr std::uint64_t seed = 990045;
  fastkag::NativeTeammateExecutor executor(library());
  fastkag::Simulator env({}, seed);
  rolling::Config config;
  config.enabled = true;
  config.maximum_retry_quantity = 1000000;
  config.require_route_seed_demand = false;
  rolling::Owner owner(executor, 1, 0, config);
  fastkag::NativeAgentState opponent;
  fastkag::NativeAgentState provider;
  bool captured = false;
  int captured_step = -1;
  std::size_t captured_intents = 0;
  while (!env.done() && !captured) {
    require(owner.observe(env), "real purchase owner receipt failed");
    const auto candidate = owner.propose(env);
    fastkag::NativeRepairOptions repair_options;
    final_commit::Request final_request{
        &executor, &env, 1, 0, repair_options, std::nullopt};
    const auto native_proposal = final_commit::propose(final_request, provider);
    require(same(native_proposal.action, candidate.base_action),
            "purchase/native providers diverged before composition");
    const bool purchase_triggered = owner.metrics().purchase_failures > 0;
    if (purchase_triggered && !same(candidate.base_action, candidate.final_action)) {
      arb::PrepareRequest request;
      request.enabled = true;
      request.step = env.step_count();
      request.observation_hash = candidate.observation_hash;
      request.base_action = candidate.base_action;
      std::vector<arb::Intent> intents;
      std::uint64_t fallback_id = candidate.binding_hash;
      for (std::size_t actor = 0; actor < candidate.base_action.units.size(); ++actor) {
        if (same(candidate.base_action.units[actor],
                 candidate.final_action.units[actor]))
          continue;
        const auto binding = std::find_if(
            candidate.receipt_bindings.begin(),
            candidate.receipt_bindings.end(), [&](const auto& value) {
              return !value.market && value.index == static_cast<int>(actor);
            });
        const auto id = binding == candidate.receipt_bindings.end()
                            ? ++fallback_id
                            : binding->debt_id;
        auto intent = unit_intent(
            arb::Source::PurchaseFailure, id, static_cast<int>(actor),
            candidate.base_action.units[actor],
            candidate.final_action.units[actor], true, 3, 1);
        intent.deadline = env.step_count() + 48;
        intents.push_back(std::move(intent));
      }
      for (std::size_t slot = 0; slot < candidate.final_action.market.size(); ++slot) {
        const Action base = slot < candidate.base_action.market.size()
                                ? candidate.base_action.market[slot]
                                : Action{};
        if (same(base, candidate.final_action.market[slot])) continue;
        const auto binding = std::find_if(
            candidate.receipt_bindings.begin(),
            candidate.receipt_bindings.end(), [&](const auto& value) {
              return value.market && value.index == static_cast<int>(slot);
            });
        const auto id = binding == candidate.receipt_bindings.end()
                            ? ++fallback_id
                            : binding->debt_id;
        auto intent = market_intent(
            arb::Source::PurchaseFailure, id, static_cast<int>(slot),
            candidate.final_action.market[slot], 400, 1, true, 4);
        intent.expected_base = base;
        intent.deadline = env.step_count() + 48;
        intents.push_back(std::move(intent));
      }
      require(!intents.empty(), "real purchase diff lacked typed intents");
      request.purchase = proposal(arb::Source::PurchaseFailure,
                                  purchase_token(candidate),
                                  request.observation_hash,
                                  candidate.base_action, std::move(intents));
      request.purchase->generation = candidate.generation;
      // Preserve the exact sealed proposal payload; proposal() reconstructs it
      // from typed diffs and must be byte-identical.
      require(same(request.purchase->proposed_action, candidate.final_action),
              "real purchase typed mapping was lossy");
      const auto free_actor = std::find_if(
          candidate.base_action.units.begin(), candidate.base_action.units.end(),
          [&](const Action& action) {
            const auto actor = static_cast<std::size_t>(
                &action - candidate.base_action.units.data());
            return same(action, candidate.final_action.units[actor]);
          });
      require(free_actor != candidate.base_action.units.end(),
              "real purchase proposal owns every actor");
      const int weed_actor = static_cast<int>(
          std::distance(candidate.base_action.units.begin(), free_actor));
      Action weed_action{Op::PASS};
      if (same(*free_actor, weed_action)) weed_action = {Op::NORTH};
      std::uint64_t weed_id = candidate.binding_hash ^ 0x9e3779b97f4a7c15ULL;
      if (weed_id == 0) weed_id = 1;
      auto weed_intent = unit_intent(
          arb::Source::WeedMinimumDamage, weed_id, weed_actor, *free_actor,
          weed_action, true, 3, 1);
      weed_intent.certified_weed_move_owner = true;
      weed_intent.deadline = env.step_count() + 48;
      request.weed = proposal(arb::Source::WeedMinimumDamage, weed_id,
                              request.observation_hash,
                              candidate.base_action, {weed_intent});
      request.weed->generation = candidate.generation + 1000;
      // This test certifies ownership composition, not the economic allocator;
      // the real owner already performed physical affordability checks.
      request.available_cash = 1'000'000;
      request.free_shed_capacity = env.config().shed_capacity;
      arb::Composer composer;
      const auto prepared = composer.prepare(request);
      require(prepared.accepted() && prepared.selected_sources.size() == 2 &&
                  same(prepared.final_action.units[weed_actor], weed_action),
              "real joint purchase/worker proposal was not composed");
      std::uint64_t actor_mask = 0;
      for (std::size_t actor = 0;
           actor < prepared.final_action.units.size(); ++actor)
        if (!same(native_proposal.action.units[actor],
                  prepared.final_action.units[actor]))
          actor_mask |= 1ULL << actor;
      const bool owns_market_tail = prepared.final_action.market.size() >
                                    native_proposal.action.market.size();
      const auto final_binding = final_commit::bind_repair_final_action(
          native_proposal, actor_mask, owns_market_tail,
          candidate.generation, prepared.binding_hash,
          prepared.final_action);
      auto committed_provider = provider;
      const auto native_commit = final_commit::commit_repair_owner_finalized(
          final_request, native_proposal, final_binding,
          prepared.final_action, committed_provider);
      require(native_commit.committed && final_commit::same_action(
                  native_commit.replayed_action, prepared.final_action),
              "real joint action did not pass native final commit");
      std::optional<rolling::Owner> staged_owner;
      rolling::FinalizeStatus purchase_commit =
          rolling::FinalizeStatus::StaleProposal;
      bool weed_committed = false;
      arb::ParticipantTransaction weed_transaction;
      weed_transaction.source = arb::Source::WeedMinimumDamage;
      weed_transaction.generation = request.weed->generation;
      weed_transaction.commit_token = weed_id;
      weed_transaction.preflight = [] { return true; };
      weed_transaction.commit = [&]() noexcept { weed_committed = true; };
      arb::ParticipantTransaction transaction;
      transaction.source = arb::Source::PurchaseFailure;
      transaction.generation = candidate.generation;
      transaction.commit_token = purchase_token(candidate);
      transaction.preflight = [&] {
        staged_owner.emplace(owner);
        return staged_owner->finalize_composed(
                   candidate, prepared.final_action,
                   committed_provider) ==
               rolling::FinalizeStatus::Selected;
      };
      transaction.commit = [&]() noexcept {
        owner = std::move(*staged_owner);
        purchase_commit = rolling::FinalizeStatus::Selected;
      };
      transaction.abort = [&]() noexcept { staged_owner.reset(); };
      require(composer.finalize(prepared, prepared.final_action,
                                {std::move(weed_transaction),
                                 std::move(transaction)}) ==
                      arb::FinalizeStatus::Selected &&
                  purchase_commit == rolling::FinalizeStatus::Selected &&
                  weed_committed,
              "real joint purchase/worker owners did not commit atomically");
      provider = std::move(committed_provider);
      std::array<PlayerAction, 2> actions;
      actions[1] = prepared.final_action;
      actions[0] = executor.action_external(env, 0, 1, opponent);
      env.step(actions);
      require(owner.observe(env), "joint action receipt was rejected");
      const auto next_candidate = owner.propose(env);
      auto next_provider = provider;
      const auto next_native = executor.action_external(
          env, 1, 0, next_provider);
      require(same(next_candidate.base_action, next_native),
              "joint commit left native owner stale on next hand");
      captured = true;
      captured_step = env.step_count();
      captured_intents = prepared.selected_intents.size();
      break;
    }
    if (same(candidate.base_action, candidate.final_action)) {
      const auto native_commit = final_commit::commit(
          final_request, native_proposal, candidate.final_action, provider);
      require(native_commit.committed &&
                  owner.finalize(candidate, candidate.final_action) ==
                      rolling::FinalizeStatus::Selected,
              "exact purchase prefix did not commit");
    } else {
      std::uint64_t actor_mask = 0;
      for (std::size_t actor = 0;
           actor < candidate.final_action.units.size(); ++actor)
        if (!same(candidate.base_action.units[actor],
                  candidate.final_action.units[actor]))
          actor_mask |= 1ULL << actor;
      const bool owns_market_tail = candidate.final_action.market.size() >
                                    candidate.base_action.market.size();
      const auto binding = final_commit::bind_repair_final_action(
          native_proposal, actor_mask, owns_market_tail,
          candidate.generation, candidate.binding_hash,
          candidate.final_action);
      const auto native_commit = final_commit::commit_repair_owner_finalized(
          final_request, native_proposal, binding,
          candidate.final_action, provider);
      require(native_commit.committed &&
                  owner.finalize_composed(candidate, candidate.final_action,
                                          provider) ==
                      rolling::FinalizeStatus::Selected,
              "repairing purchase prefix did not commit");
    }
    std::array<PlayerAction, 2> actions;
    actions[1] = candidate.final_action;
    actions[0] = executor.action_external(env, 0, 1, opponent);
    env.step(actions);
  }
  require(captured, "real purchase proposal was not observed");
  std::cout << "real_purchase_shadow seed=" << seed << " step=" << captured_step
            << " typed_intents=" << captured_intents
            << " joint_trigger_observed=true atomic_commit=true\n";
}

}  // namespace

int main() {
  try {
    synthetic_joint_atomicity_and_tamper();
    adversarial_conflict_graph_and_default_off();
    real_weed_minimum_damage_shadow();
    real_purchase_shadow();
    std::cout << "repair_composer_arbiter_tests: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "repair_composer_arbiter_tests: " << error.what() << '\n';
    return 1;
  }
}
