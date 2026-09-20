#include "transactional_crop_repair_owner.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace owner_ns = g001::transactional_crop_repair;
namespace fork_ns = g001::repair_fork;
namespace tx = g001::repair_owner_composer;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
using fastkag::TileKind;

namespace {

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

bool same_action(const Action &left, const Action &right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

fastkag::Config config(int money = 3000, int episode_steps = 240) {
  fastkag::Config cfg;
  cfg.starting_money = money;
  cfg.episode_steps = episode_steps;
  cfg.weed_spawn_chance = 0.0;
  return cfg;
}

PlayerAction raw(int actors = 1) {
  PlayerAction out;
  out.units.resize(static_cast<std::size_t>(actors));
  return out;
}

std::array<PlayerAction, 2> joint(PlayerAction focal, int player = 0) {
  std::array<PlayerAction, 2> out{raw(), raw()};
  out[static_cast<std::size_t>(player)] = std::move(focal);
  return out;
}

fastkag::Tile &actor_tile(Simulator &env, int player = 0) {
  auto &farm = const_cast<fastkag::Farm &>(env.farms()[player]);
  const auto position = farm.farmer;
  return farm.tiles[static_cast<std::size_t>(
      position.y * env.config().board_size + position.x)];
}

fastkag::PrivateState &private_state(Simulator &env, int player = 0) {
  return const_cast<fastkag::PrivateState &>(env.privates()[player]);
}

void set_farmer(Simulator &env, int row, int column, int player = 0) {
  auto &farm = const_cast<fastkag::Farm &>(env.farms()[player]);
  farm.farmer = {static_cast<std::int16_t>(column),
                 static_cast<std::int16_t>(row)};
}

fork_ns::RepairContext context(const Simulator &env, int player,
                               const std::array<PlayerAction, 2> &raw_joint) {
  return {env,
          player,
          env.step_count(),
          raw_joint[static_cast<std::size_t>(player)],
          raw_joint,
          {},
          {}};
}

tx::PreparedRepair typed_prepare(owner_ns::TransactionalCropRepairOwner &owner,
                                 const Simulator &env, int player,
                                 const std::array<PlayerAction, 2> &raw_joint,
                                 Item desired, std::uint64_t obligation_id = 1,
                                 int actor = 0) {
  const auto &farm = env.farms()[static_cast<std::size_t>(player)];
  const auto position = actor == 0
                            ? farm.farmer
                            : farm.hands[static_cast<std::size_t>(actor - 1)];
  const auto &source = raw_joint[static_cast<std::size_t>(player)]
                           .units[static_cast<std::size_t>(actor)];
  const owner_ns::TypedCropObligation obligation{
      obligation_id, player, actor,  env.step_count(),
      position,      source, desired};
  return owner.prepare(context(env, player, raw_joint),
                       std::span{&obligation, std::size_t{1}});
}

struct Granted {
  tx::CommitGrant grant;
  fork_ns::MarketCompileResult compiled;
};

Granted grant(const Simulator &env, int player,
              const std::array<PlayerAction, 2> &raw_joint,
              const tx::PreparedRepair &proposal,
              std::uint64_t generation = 1) {
  Granted out;
  out.grant.submitted_step = env.step_count();
  out.grant.final_units = proposal.units;
  auto final_joint = raw_joint;
  final_joint[static_cast<std::size_t>(player)].units = proposal.units;
  for (int actor : proposal.claimed_actors) {
    const fork_ns::ActorPrefixAuthority authority{
        actor, generation,
        fork_ns::unit_prefix_manifest_hash(proposal.units, actor),
        fork_ns::post_unit_prefix_state_fingerprint(env, player, final_joint,
                                                    actor)};
    out.grant.actions.push_back(
        {actor, proposal.units[static_cast<std::size_t>(actor)], authority});
  }
  std::vector<fork_ns::RequiredPurchase> external;
  for (std::size_t index = 0; index < proposal.required_purchases.size();
       ++index) {
    auto request = proposal.required_purchases[index];
    request.debt_id = (1ULL << 62U) + request.debt_id + index;
    external.push_back(request);
  }
  fork_ns::ExactMarketCompiler compiler;
  out.compiled = compiler.compile(
      env, player, raw_joint[static_cast<std::size_t>(player)].market,
      external);
  for (std::size_t index = 0; index < external.size(); ++index)
    out.grant.purchases.push_back({proposal.required_purchases[index].debt_id,
                                   external[index].debt_id,
                                   out.compiled.bindings[index]});
  return out;
}

struct Receipts {
  std::vector<fork_ns::ActionReceipt> actions;
  std::vector<fork_ns::PurchaseReceipt> purchases;
};

Receipts execute(Simulator &env, int player,
                 std::array<PlayerAction, 2> raw_joint,
                 const tx::PreparedRepair &proposal, const Granted &selected) {
  raw_joint[static_cast<std::size_t>(player)].units = proposal.units;
  raw_joint[static_cast<std::size_t>(player)].market = selected.compiled.market;
  const int submitted = env.step_count();
  env.step(raw_joint);
  Receipts out;
  for (const auto &committed : selected.grant.actions)
    out.actions.push_back(
        {submitted, committed.actor,
         committed.final_authority.manifest_generation,
         committed.final_authority.prefix_manifest_hash,
         committed.final_authority.post_prefix_state_fingerprint,
         committed.emitted});
  const auto &fills = env.last_market_fills()[player];
  std::vector<int> remaining(fills.begin(), fills.end());
  for (const auto &binding : selected.compiled.bindings) {
    int filled = 0;
    if (binding.market_slot >= 0 &&
        binding.market_slot < static_cast<int>(remaining.size())) {
      auto &slot = remaining[static_cast<std::size_t>(binding.market_slot)];
      filled = std::min(binding.requested, slot);
      slot -= filled;
    }
    out.purchases.push_back({binding.debt_id, submitted, binding.operation,
                             binding.item, binding.requested, filled,
                             binding.market_slot, binding.status});
  }
  return out;
}

bool settle(owner_ns::TransactionalCropRepairOwner &owner, const Simulator &env,
            int player, const std::array<PlayerAction, 2> &raw_joint,
            const Receipts &receipts = {}) {
  const auto ctx = context(env, player, raw_joint);
  return owner.settle_owned(ctx, receipts.actions, receipts.purchases) ==
         g001::repair_owner_composer::SettlementResult::AppliedSuccess;
}

void consecutive_aborts_are_pure_then_commit_succeeds() {
  Simulator env(config(), 2001);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction focal = raw();
  focal.units[0] = {Op::WATER, Item::NONE, 1};
  auto actions = joint(focal);
  const auto initial = owner.persistent_fingerprint();
  for (int round = 0; round < 3; ++round) {
    const auto proposal = typed_prepare(owner, env, 0, actions, Item::WHEAT);
    check(proposal.units[0].op == Op::DIG && proposal.token != 0,
          "abort fixture did not prepare deterministic DIG");
    owner.abort(proposal.token);
    check(owner.persistent_fingerprint() == initial &&
              owner.plot_debts().empty() &&
              owner.expected_action_receipts() == 0 &&
              owner.expected_purchase_receipts() == 0 &&
              owner.delayed_moves() == 0,
          "abort changed persistent crop state");
    std::array<PlayerAction, 2> pass{raw(), raw()};
    env.step(pass);
    actions = joint(focal);
  }
  const auto proposal = typed_prepare(owner, env, 0, actions, Item::WHEAT);
  const auto selected = grant(env, 0, actions, proposal);
  check(owner.commit(proposal.token, selected.grant) &&
            owner.plot_debts().size() == 1 &&
            owner.expected_action_receipts() == 1,
        "valid commit after repeated aborts did not advance state");
  const auto receipts = execute(env, 0, actions, proposal, selected);
  check(settle(owner, env, 0, joint(raw()), receipts),
        "committed DIG did not settle after repeated aborts");
}

void abort_never_creates_fake_debt_or_receipt() {
  Simulator env(config(), 2002);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction focal = raw();
  focal.units[0] = {Op::WATER, Item::NONE, 1};
  const auto actions = joint(focal);
  const auto before = owner.persistent_fingerprint();
  const auto first = typed_prepare(owner, env, 0, actions, Item::STRAWBERRY);
  const auto second = typed_prepare(owner, env, 0, actions, Item::STRAWBERRY);
  check(first.token == second.token,
        "same state/context did not produce a recomputable token");
  owner.abort(first.token);
  check(owner.persistent_fingerprint() == before &&
            owner.cached_proposals() == 0 && owner.plot_debts().empty() &&
            owner.expected_action_receipts() == 0 &&
            owner.expected_purchase_receipts() == 0,
        "abort manufactured debt or expected receipt");
}

void zero_and_partial_seed_fills_are_exact() {
  for (const int money : {0, 10}) {
    Simulator env(config(money), static_cast<std::uint64_t>(2100 + money));
    owner_ns::TransactionalCropRepairOwner owner;
    PlayerAction focal = raw();
    focal.units[0] = {Op::PLANT, Item::WHEAT, 1};
    focal.market.push_back({Op::BUY_SEED, Item::WHEAT, 2});
    auto actions = joint(focal);
    const auto proposal = owner.prepare(context(env, 0, actions));
    check(proposal.required_purchases.size() == 1 &&
              proposal.required_purchases[0].quantity == 2,
          "hard seed quantity was not preserved transactionally");
    const auto selected = grant(env, 0, actions, proposal);
    check(owner.commit(proposal.token, selected.grant),
          "seed purchase commit failed");
    const auto receipts = execute(env, 0, actions, proposal, selected);
    check(receipts.purchases.size() == 1 &&
              receipts.purchases[0].filled == (money == 0 ? 0 : 1),
          "fixture did not produce requested zero/partial fill");
    check(settle(owner, env, 0, joint(raw()), receipts),
          "exact seed receipt failed settlement");
    check(owner.plot_debts().size() == 1 &&
              (money == 0 ? owner.audit().purchase_zero_fills == 1
                          : owner.audit().purchase_partial_fills == 1),
          "zero/partial fill retired debt or was misclassified");
    const auto retry = owner.prepare(context(env, 0, joint(raw())));
    check(retry.required_purchases.size() == 1 &&
              retry.required_purchases[0].debt_id ==
                  proposal.required_purchases[0].debt_id &&
              retry.required_purchases[0].quantity == (money == 0 ? 2 : 1),
          "zero/partial fill did not propose the exact remaining seed debt");
    owner.abort(retry.token);
  }
}

void weed_zero_fill_retries_after_dig_to_empty() {
  Simulator env(config(0), 2111);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction focal = raw();
  focal.units[0] = {Op::WATER, Item::NONE, 1};
  auto actions = joint(focal);
  const auto first = typed_prepare(owner, env, 0, actions, Item::WHEAT);
  check(first.units[0].op == Op::DIG && first.required_purchases.size() == 1,
        "WEED zero-fill setup lacked DIG+BUY");
  const auto selected = grant(env, 0, actions, first);
  check(owner.commit(first.token, selected.grant),
        "WEED DIG+BUY commit failed");
  const auto receipts = execute(env, 0, actions, first, selected);
  check(receipts.purchases[0].filled == 0 &&
            settle(owner, env, 0, joint(raw()), receipts) &&
            actor_tile(env).kind == TileKind::EMPTY,
        "WEED zero-fill setup did not settle to EMPTY+0 seed");
  const auto retry = owner.prepare(context(env, 0, joint(raw())));
  check(retry.required_purchases.size() == 1 &&
            retry.required_purchases[0].debt_id ==
                first.required_purchases[0].debt_id,
        "EMPTY+0 seed lost the stable local purchase debt");
  owner.abort(retry.token);
}

void rejected_no_slot_retries_stable_debt() {
  auto cfg = config();
  cfg.max_market_orders = 1;
  Simulator env(cfg, 2112);
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction focal = raw();
  focal.units[0] = {Op::PLANT, Item::CARROT, 1};
  focal.market.push_back({});
  auto actions = joint(focal);
  const auto first = owner.prepare(context(env, 0, actions));
  const auto selected = grant(env, 0, actions, first);
  check(selected.compiled.bindings[0].status ==
                fork_ns::PurchaseCompileStatus::RejectedNoSlot &&
            owner.commit(first.token, selected.grant),
        "RejectedNoSlot setup failed");
  const auto receipts = execute(env, 0, actions, first, selected);
  check(settle(owner, env, 0, joint(raw()), receipts),
        "RejectedNoSlot receipt was not accepted as retryable");
  const auto retry = owner.prepare(context(env, 0, joint(raw())));
  check(retry.required_purchases.size() == 1 &&
            retry.required_purchases[0].debt_id ==
                first.required_purchases[0].debt_id,
        "RejectedNoSlot abandoned or renumbered seed debt");
  owner.abort(retry.token);
}

void crop_debt_survives_midnight() {
  Simulator env(config(), 2003);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction focal = raw();
  focal.units[0] = {Op::WATER, Item::NONE, 1};
  auto actions = joint(focal);
  const auto proposal = typed_prepare(owner, env, 0, actions, Item::TOMATO);
  const auto selected = grant(env, 0, actions, proposal);
  check(owner.commit(proposal.token, selected.grant),
        "cross-day setup commit failed");
  auto receipts = execute(env, 0, actions, proposal, selected);
  check(settle(owner, env, 0, joint(raw()), receipts),
        "cross-day setup receipt failed");
  while (env.step_count() < 24) {
    std::array<PlayerAction, 2> pass{raw(), raw()};
    env.step(pass);
  }
  check(settle(owner, env, 0, joint(raw())) && owner.plot_debts().size() == 1 &&
            owner.plot_debts()[0].origin_day == 0,
        "plot-owned crop debt did not survive midnight");
}

void uncertified_move_shift_is_rejected_without_state_change() {
  Simulator env(config(), 2004);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction trigger = raw();
  trigger.units[0] = {Op::WATER, Item::NONE, 1};
  auto first_joint = joint(trigger);
  auto first = typed_prepare(owner, env, 0, first_joint, Item::WHEAT);
  auto first_grant = grant(env, 0, first_joint, first);
  check(owner.commit(first.token, first_grant.grant), "DIG setup failed");
  auto receipts = execute(env, 0, first_joint, first, first_grant);
  PlayerAction move = raw();
  move.units[0] = {Op::EAST, Item::NONE, 1};
  auto move_joint = joint(move);
  check(settle(owner, env, 0, move_joint, receipts), "DIG settle failed");
  const auto before = owner.persistent_fingerprint();
  const auto proposal = owner.prepare(context(env, 0, move_joint));
  check(proposal.units[0].op == Op::PLANT,
        "existing debt did not propose MOVE-slot PLANT");
  const auto selected = grant(env, 0, move_joint, proposal, 2);
  check(!owner.commit(proposal.token, selected.grant) &&
            owner.persistent_fingerprint() == before &&
            owner.delayed_moves() == 0 &&
            owner.audit().uncertified_move_rejections == 1,
        "uncertified MOVE shift changed persistent state");
}

void certified_move_shift_replays_same_day() {
  Simulator env(config(), 2005);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction trigger = raw();
  trigger.units[0] = {Op::WATER, Item::NONE, 1};
  auto trigger_joint = joint(trigger);
  auto dig = typed_prepare(owner, env, 0, trigger_joint, Item::WHEAT);
  auto dig_grant = grant(env, 0, trigger_joint, dig);
  check(owner.commit(dig.token, dig_grant.grant), "certified setup DIG failed");
  auto receipts = execute(env, 0, trigger_joint, dig, dig_grant);

  PlayerAction move = raw();
  move.units[0] = {Op::EAST, Item::NONE, 1};
  auto move_joint = joint(move);
  check(settle(owner, env, 0, move_joint, receipts),
        "certified setup settle failed");
  auto plant = owner.prepare(context(env, 0, move_joint));
  auto plant_grant = grant(env, 0, move_joint, plant, 2);
  plant_grant.grant.certified_move_shifts.push_back(
      {0, env.step_count(), env.step_count() + 2, move.units[0], 991, 17});
  check(owner.commit(plant.token, plant_grant.grant) &&
            owner.delayed_moves() == 1,
        "trusted same-day MOVE shift was not committed");
  receipts = execute(env, 0, move_joint, plant, plant_grant);

  auto pass_joint = joint(raw());
  check(settle(owner, env, 0, pass_joint, receipts),
        "shifted PLANT did not settle");
  auto water = owner.prepare(context(env, 0, pass_joint));
  auto water_grant = grant(env, 0, pass_joint, water, 3);
  check(owner.commit(water.token, water_grant.grant),
        "intervening WATER commit failed");
  receipts = execute(env, 0, pass_joint, water, water_grant);

  check(settle(owner, env, 0, pass_joint, receipts),
        "intervening WATER did not settle");
  auto replay = owner.prepare(context(env, 0, pass_joint));
  check(replay.units[0].op == Op::EAST && replay.sources[0].source_step == 1 &&
            same_action(replay.sources[0].source_action, move.units[0]),
        "deferred MOVE source/direction was not replayed exactly");
  auto replay_grant = grant(env, 0, pass_joint, replay, 4);
  check(owner.commit(replay.token, replay_grant.grant) &&
            owner.delayed_moves() == 1 &&
            owner.audit().move_shifts_committed == 1 &&
            owner.audit().move_shifts_replayed == 0,
        "MOVE token retired before its owned receipt");
  receipts = execute(env, 0, pass_joint, replay, replay_grant);
  check(settle(owner, env, 0, joint(raw()), receipts) &&
            owner.delayed_moves() == 0 &&
            owner.audit().move_shifts_replayed == 1,
        "same-day deferred MOVE did not close on exact receipt");
}

void missing_replay_receipt_cannot_move_to_another_slot() {
  Simulator env(config(), 2008);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction trigger = raw();
  trigger.units[0] = {Op::WATER, Item::NONE, 1};
  auto trigger_joint = joint(trigger);
  auto dig = typed_prepare(owner, env, 0, trigger_joint, Item::WHEAT);
  auto dig_grant = grant(env, 0, trigger_joint, dig);
  check(owner.commit(dig.token, dig_grant.grant), "missing-receipt DIG failed");
  auto receipts = execute(env, 0, trigger_joint, dig, dig_grant);
  PlayerAction move = raw();
  move.units[0] = {Op::EAST, Item::NONE, 1};
  auto move_joint = joint(move);
  check(settle(owner, env, 0, move_joint, receipts),
        "missing-receipt DIG settle failed");
  auto plant = owner.prepare(context(env, 0, move_joint));
  auto plant_grant = grant(env, 0, move_joint, plant, 2);
  plant_grant.grant.certified_move_shifts.push_back(
      {0, env.step_count(), env.step_count() + 1, move.units[0], 992, 18});
  check(owner.commit(plant.token, plant_grant.grant),
        "missing-receipt shift failed");
  receipts = execute(env, 0, move_joint, plant, plant_grant);
  auto pass_joint = joint(raw());
  check(settle(owner, env, 0, pass_joint, receipts),
        "missing-receipt PLANT settle failed");
  auto replay = owner.prepare(context(env, 0, pass_joint));
  auto replay_grant = grant(env, 0, pass_joint, replay, 3);
  check(replay.units[0].op == Op::EAST &&
            owner.commit(replay.token, replay_grant.grant),
        "authorized replay slot did not submit MOVE");
  receipts = execute(env, 0, pass_joint, replay, replay_grant);
  check(!settle(owner, env, 0, pass_joint) && owner.delayed_moves() == 1 &&
            owner.expected_action_receipts() == 1,
        "missing replay receipt silently retired MOVE debt");
  bool prepare_blocked = false;
  try {
    static_cast<void>(owner.prepare(context(env, 0, pass_joint)));
  } catch (const std::logic_error &) {
    prepare_blocked = true;
  }
  check(prepare_blocked && settle(owner, env, 0, pass_joint, receipts) &&
            owner.delayed_moves() == 0 &&
            owner.expected_action_receipts() == 0,
        "late exact replay receipt did not close the unchanged protocol debt");
}

void player1_commit_verifies_final_prefix_including_player0() {
  Simulator env(config(), 2006);
  actor_tile(env, 1).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction focal = raw();
  focal.units[0] = {Op::WATER, Item::NONE, 1};
  auto actions = joint(focal, 1);
  actions[0].units[0] = {Op::EAST, Item::NONE, 1};
  const auto proposal = typed_prepare(owner, env, 1, actions, Item::MELON);
  auto selected = grant(env, 1, actions, proposal);
  check(owner.commit(proposal.token, selected.grant),
        "player1 final authority including player0 was rejected");

  owner_ns::TransactionalCropRepairOwner rejected;
  const auto bad_proposal =
      typed_prepare(rejected, env, 1, actions, Item::MELON);
  auto bad = grant(env, 1, actions, bad_proposal);
  ++bad.grant.actions[0].final_authority.post_prefix_state_fingerprint;
  const auto before = rejected.persistent_fingerprint();
  check(!rejected.commit(bad_proposal.token, bad.grant) &&
            rejected.persistent_fingerprint() == before,
        "player1 corrupt final prefix authority was accepted");
}

void itemless_harvest_requires_typed_intent_then_recovers() {
  Simulator env(config(), 2010);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction focal = raw();
  focal.units[0] = {Op::HARVEST, Item::NONE, 1};
  auto actions = joint(focal);

  const auto untyped = owner.prepare(context(env, 0, actions));
  check(untyped.units[0].op == Op::HARVEST && untyped.claimed_actors.empty() &&
            untyped.required_purchases.empty(),
        "itemless HARVEST guessed a crop without typed intent");
  owner.abort(untyped.token);

  const auto typed = typed_prepare(owner, env, 0, actions, Item::CARROT, 901);
  check(typed.units[0].op == Op::DIG && typed.required_purchases.size() == 1 &&
            typed.required_purchases[0].item == Item::CARROT,
        "typed itemless HARVEST did not compile exact DIG+seed recovery");
  const auto selected = grant(env, 0, actions, typed);
  check(owner.commit(typed.token, selected.grant),
        "typed HARVEST recovery commit failed");
  const auto receipts = execute(env, 0, actions, typed, selected);
  check(settle(owner, env, 0, joint(raw()), receipts) &&
            owner.plot_debts().size() == 1 &&
            owner.plot_debts()[0].desired == Item::CARROT &&
            owner.plot_debts()[0].obligation_id == 901 &&
            owner.plot_debts()[0].source_player == 0 &&
            owner.plot_debts()[0].source_actor == 0 &&
            owner.plot_debts()[0].source_step == 0 &&
            same_action(owner.plot_debts()[0].source_action,
                        Action{Op::HARVEST, Item::NONE, 1}),
        "typed HARVEST recovery lost desired crop provenance");
}

void ambiguous_typed_intents_fail_closed_in_both_orders() {
  Simulator env(config(), 2011);
  actor_tile(env).kind = TileKind::WEED;
  PlayerAction focal = raw();
  focal.units[0] = {Op::WATER, Item::NONE, 1};
  const auto actions = joint(focal);
  const auto position = env.farms()[0].farmer;
  std::array<owner_ns::TypedCropObligation, 2> intents{
      owner_ns::TypedCropObligation{11, 0, 0, env.step_count(), position,
                                    focal.units[0], Item::WHEAT},
      owner_ns::TypedCropObligation{12, 0, 0, env.step_count(), position,
                                    focal.units[0], Item::CARROT}};
  for (int reversed = 0; reversed < 2; ++reversed) {
    if (reversed)
      std::swap(intents[0], intents[1]);
    owner_ns::TransactionalCropRepairOwner owner;
    const auto proposal = owner.prepare(context(env, 0, actions), intents);
    check(proposal.claimed_actors.empty() &&
              proposal.units[0].op == Op::WATER &&
              proposal.required_purchases.empty(),
          "ambiguous typed crop intents were resolved by input order");
    owner.abort(proposal.token);
  }
}

void missing_purchase_receipt_retains_owned_expectation() {
  Simulator env(config(0), 2012);
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction focal = raw();
  focal.units[0] = {Op::PLANT, Item::WHEAT, 1};
  auto actions = joint(focal);
  const auto proposal = owner.prepare(context(env, 0, actions));
  const auto selected = grant(env, 0, actions, proposal);
  check(owner.commit(proposal.token, selected.grant),
        "missing purchase receipt setup commit failed");
  const auto receipts = execute(env, 0, actions, proposal, selected);
  check(!settle(owner, env, 0, joint(raw())) &&
            owner.expected_purchase_receipts() == 1,
        "missing purchase receipt was cleared as retryable");
  bool prepare_blocked = false;
  try {
    static_cast<void>(owner.prepare(context(env, 0, joint(raw()))));
  } catch (const std::logic_error &) {
    prepare_blocked = true;
  }
  check(prepare_blocked,
        "owner prepared a duplicate BUY with an unresolved receipt");
  check(settle(owner, env, 0, joint(raw()), receipts) &&
            owner.expected_purchase_receipts() == 0,
        "late exact owned purchase receipt did not close expectation");
}

void active_debt_rotation_resets_item_scoped_purchase_provenance() {
  Simulator env(config(), 2013);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction wheat = raw();
  wheat.units[0] = {Op::WATER, Item::NONE, 1};
  wheat.market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  auto actions = joint(wheat);
  auto proposal = typed_prepare(owner, env, 0, actions, Item::WHEAT, 1001);
  auto selected = grant(env, 0, actions, proposal);
  check(owner.commit(proposal.token, selected.grant),
        "WHEAT rotation setup commit failed");
  auto receipts = execute(env, 0, actions, proposal, selected);
  check(settle(owner, env, 0, joint(raw()), receipts) &&
            owner.audit().purchase_full_fills == 1,
        "WHEAT rotation setup purchase did not settle");

  actor_tile(env).kind = TileKind::WEED;
  PlayerAction carrot = raw();
  carrot.units[0] = {Op::WATER, Item::NONE, 1};
  carrot.market.push_back({Op::BUY_SEED, Item::CARROT, 1});
  actions = joint(carrot);
  proposal = typed_prepare(owner, env, 0, actions, Item::CARROT, 1002);
  check(proposal.units[0].op == Op::DIG &&
            proposal.required_purchases.size() == 1 &&
            proposal.required_purchases[0].item == Item::CARROT &&
            proposal.required_purchases[0].quantity == 1,
        "old WHEAT seed credit suppressed CARROT recovery purchase");
  selected = grant(env, 0, actions, proposal, 2);
  check(owner.commit(proposal.token, selected.grant) &&
            owner.plot_debts().size() == 1 &&
            owner.plot_debts()[0].desired == Item::CARROT &&
            owner.plot_debts()[0].obligation_id == 1002 &&
            owner.plot_debts()[0].source_step == env.step_count() &&
            same_action(owner.plot_debts()[0].source_action,
                        Action{Op::WATER, Item::NONE, 1}),
        "same-tile rotation did not atomically replace provenance");
}

void same_crop_new_obligation_preserves_purchase_progress() {
  Simulator env(config(), 2014);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction wheat = raw();
  wheat.units[0] = {Op::WATER, Item::NONE, 1};
  wheat.market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  auto actions = joint(wheat);
  auto proposal = typed_prepare(owner, env, 0, actions, Item::WHEAT, 2001);
  auto selected = grant(env, 0, actions, proposal);
  check(owner.commit(proposal.token, selected.grant),
        "same-crop setup commit failed");
  auto receipts = execute(env, 0, actions, proposal, selected);
  check(settle(owner, env, 0, joint(raw()), receipts) &&
            owner.audit().purchase_full_fills == 1,
        "same-crop setup purchase failed");

  actor_tile(env).kind = TileKind::WEED;
  PlayerAction refresh = raw();
  refresh.units[0] = {Op::WATER, Item::NONE, 1};
  actions = joint(refresh);
  proposal = typed_prepare(owner, env, 0, actions, Item::WHEAT, 2002);
  check(proposal.units[0].op == Op::DIG && proposal.required_purchases.empty(),
        "same crop/new source id reset acquired seed credit");
  selected = grant(env, 0, actions, proposal, 2);
  check(owner.commit(proposal.token, selected.grant) &&
            owner.plot_debts()[0].obligation_id == 2002 &&
            owner.plot_debts()[0].desired == Item::WHEAT,
        "same-crop provenance did not refresh atomically");
  receipts = execute(env, 0, actions, proposal, selected);
  check(settle(owner, env, 0, joint(raw()), receipts) &&
            owner.audit().purchase_full_fills == 1,
        "same-crop refresh manufactured another purchase settlement");
}

void diverted_seed_restores_acquisition_debt() {
  Simulator env(config(), 2015);
  actor_tile(env).kind = TileKind::WEED;
  owner_ns::TransactionalCropRepairOwner owner;
  PlayerAction wheat = raw();
  wheat.units[0] = {Op::WATER, Item::NONE, 1};
  wheat.market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  auto actions = joint(wheat);
  auto proposal = typed_prepare(owner, env, 0, actions, Item::WHEAT, 3001);
  auto selected = grant(env, 0, actions, proposal);
  check(owner.commit(proposal.token, selected.grant),
        "diverted-seed setup commit failed");
  auto receipts = execute(env, 0, actions, proposal, selected);
  check(settle(owner, env, 0, joint(raw()), receipts) &&
            private_state(env).seeds[static_cast<int>(Item::WHEAT)] == 1,
        "diverted-seed setup did not acquire seed");

  // Exact next state after another raw plot consumed the fungible WHEAT seed.
  private_state(env).seeds[static_cast<int>(Item::WHEAT)] = 0;
  proposal = owner.prepare(context(env, 0, joint(raw())));
  check(proposal.required_purchases.size() == 1 &&
            proposal.required_purchases[0].item == Item::WHEAT &&
            proposal.required_purchases[0].quantity == 1,
        "lost physical seed left durable credit and silently stuck EMPTY plot");
  selected = grant(env, 0, joint(raw()), proposal, 2);
  check(owner.commit(proposal.token, selected.grant),
        "restored acquisition debt did not commit");
}

void two_wheat_credits_allocate_one_physical_seed_once() {
  Simulator env(config(), 2016);
  owner_ns::TransactionalCropRepairOwner owner;
  actor_tile(env).kind = TileKind::WEED;
  PlayerAction first = raw();
  first.units[0] = {Op::WATER, Item::NONE, 1};
  first.market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  auto actions = joint(first);
  auto proposal = typed_prepare(owner, env, 0, actions, Item::WHEAT, 4001);
  auto selected = grant(env, 0, actions, proposal);
  check(owner.commit(proposal.token, selected.grant),
        "first WHEAT reservation commit failed");
  auto receipts = execute(env, 0, actions, proposal, selected);
  check(settle(owner, env, 0, joint(raw()), receipts),
        "first WHEAT reservation settle failed");

  set_farmer(env, 0, 0);
  actor_tile(env).kind = TileKind::WEED;
  PlayerAction second = raw();
  second.units[0] = {Op::WATER, Item::NONE, 1};
  second.market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  actions = joint(second);
  proposal = typed_prepare(owner, env, 0, actions, Item::WHEAT, 4002);
  selected = grant(env, 0, actions, proposal, 2);
  check(owner.commit(proposal.token, selected.grant),
        "second WHEAT reservation commit failed");
  receipts = execute(env, 0, actions, proposal, selected);
  check(settle(owner, env, 0, joint(raw()), receipts) &&
            owner.plot_debts().size() == 2 &&
            private_state(env).seeds[static_cast<int>(Item::WHEAT)] == 2,
        "two WHEAT reservation setup failed");

  // One physical seed cannot back both historical acquisition credits. Visit
  // the earlier plot first and abort, then visit the later plot: allocation
  // remains stable by plot id rather than prepare order, and exactly one
  // replacement BUY is required.
  private_state(env).seeds[static_cast<int>(Item::WHEAT)] = 1;
  set_farmer(env, 4, 4);
  proposal = owner.prepare(context(env, 0, joint(raw())));
  check(proposal.required_purchases.empty(),
        "stable first plot lost the sole physical seed reservation");
  owner.abort(proposal.token);
  set_farmer(env, 0, 0);
  proposal = owner.prepare(context(env, 0, joint(raw())));
  check(proposal.required_purchases.size() == 1 &&
            proposal.required_purchases[0].operation == Op::BUY_SEED &&
            proposal.required_purchases[0].item == Item::WHEAT &&
            proposal.required_purchases[0].quantity == 1,
        "one WHEAT seed was double-claimed by two plot credits");
  owner.abort(proposal.token);

  // Once the earlier plot is already planted, its seed credit was physically
  // consumed and must not reserve the remaining inventory seed. The later
  // EMPTY plot can PLANT without a duplicate BUY.
  set_farmer(env, 4, 4);
  actor_tile(env) = {};
  actor_tile(env).kind = TileKind::PLANT;
  actor_tile(env).crop = Item::WHEAT;
  actor_tile(env).watered_today = false;
  set_farmer(env, 0, 0);
  proposal = owner.prepare(context(env, 0, joint(raw())));
  check(proposal.units[0].op == Op::PLANT &&
            proposal.units[0].item == Item::WHEAT &&
            proposal.required_purchases.empty(),
        "already-planted early plot stole later EMPTY plot seed reservation");
  owner.abort(proposal.token);
}

} // namespace

int main() {
  try {
    consecutive_aborts_are_pure_then_commit_succeeds();
    abort_never_creates_fake_debt_or_receipt();
    zero_and_partial_seed_fills_are_exact();
    weed_zero_fill_retries_after_dig_to_empty();
    rejected_no_slot_retries_stable_debt();
    crop_debt_survives_midnight();
    uncertified_move_shift_is_rejected_without_state_change();
    certified_move_shift_replays_same_day();
    missing_replay_receipt_cannot_move_to_another_slot();
    player1_commit_verifies_final_prefix_including_player0();
    itemless_harvest_requires_typed_intent_then_recovers();
    ambiguous_typed_intents_fail_closed_in_both_orders();
    missing_purchase_receipt_retains_owned_expectation();
    same_crop_new_obligation_preserves_purchase_progress();
    active_debt_rotation_resets_item_scoped_purchase_provenance();
    diverted_seed_restores_acquisition_debt();
    two_wheat_credits_allocate_one_physical_seed_once();
    std::cout << "transactional crop owner: 17 deterministic groups passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "transactional crop owner test failure: " << error.what()
              << '\n';
    return 1;
  }
}
