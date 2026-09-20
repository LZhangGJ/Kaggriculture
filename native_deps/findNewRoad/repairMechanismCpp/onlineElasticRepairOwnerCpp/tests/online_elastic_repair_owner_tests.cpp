#include "online_elastic_repair_owner.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace online = g001::online_elastic;
namespace fork_api = g001::repair_fork;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
using fastkag::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool same_action(const Action& left, const Action& right) {
  return left.op == right.op && left.item == right.item &&
      left.quantity == right.quantity;
}

Simulator weed_state(int money) {
  fastkag::Config config;
  config.starting_money = money;
  config.weed_spawn_chance = 1.0;
  Simulator env(config, 77101);
  for (int step = 0; step < 24; ++step) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({});
    actions[1].units.push_back({});
    env.step(actions);
  }
  const auto position = env.farms()[0].farmer;
  const auto& tile = env.farms()[0].tiles[static_cast<std::size_t>(
      position.y * env.config().board_size + position.x)];
  check(tile.kind == TileKind::WEED, "fixture did not create spawn weed");
  return env;
}

struct Applied {
  std::vector<fork_api::PurchaseReceipt> receipts;
  std::vector<fork_api::ActionReceipt> action_receipts;
};

Applied apply(Simulator& env, int player, const PlayerAction& raw,
              const fork_api::RepairDecision& decision) {
  fork_api::ExactMarketCompiler compiler;
  const auto compiled = compiler.compile(env, player, raw.market,
                                         decision.required_purchases);
  std::array<PlayerAction, 2> actions;
  actions[static_cast<std::size_t>(player)].units = decision.units;
  actions[static_cast<std::size_t>(player)].market = compiled.market;
  actions[static_cast<std::size_t>(1 - player)].units.push_back({});
  const int submitted = env.step_count();
  env.step(actions);
  Applied out;
  for (const auto& authority : decision.prefix_authority) {
    out.action_receipts.push_back(
        {submitted, authority.actor, authority.manifest_generation,
         authority.prefix_manifest_hash,
         authority.post_prefix_state_fingerprint,
         decision.units[static_cast<std::size_t>(authority.actor)]});
  }
  const auto& fills = env.last_market_fills()[player];
  std::vector<int> remaining(fills.begin(), fills.end());
  for (const auto& binding : compiled.bindings) {
    int filled = 0;
    if (binding.market_slot >= 0 &&
        binding.market_slot < static_cast<int>(remaining.size())) {
      auto& slot = remaining[static_cast<std::size_t>(binding.market_slot)];
      filled = std::min(binding.requested, slot);
      slot -= filled;
    }
    out.receipts.push_back({binding.debt_id, submitted, binding.operation,
                            binding.item, binding.requested, filled,
                            binding.market_slot, binding.status});
  }
  return out;
}

fork_api::RepairDecision decide(online::OnlineElasticRepairOwner& owner,
                            const Simulator& env, const PlayerAction& raw,
                            const std::vector<fork_api::PurchaseReceipt>& receipts =
                                {},
                            const std::vector<fork_api::ActionReceipt>&
                                action_receipts = {}) {
  std::array<PlayerAction, 2> joint;
  joint[0] = raw;
  return owner.decide({env, 0, env.step_count(), raw, joint, receipts,
                       action_receipts});
}

PlayerAction trigger(int hard_quantity) {
  PlayerAction raw;
  raw.units.push_back({Op::WATER, Item::WHEAT, 1});
  if (hard_quantity > 0)
    raw.market.push_back({Op::BUY_SEED, Item::WHEAT, hard_quantity});
  return raw;
}

void zero_fill_preserves_move_and_debt_without_certificate() {
  auto env = weed_state(0);
  online::OnlineElasticRepairOwner owner;
  const auto raw0 = trigger(1);
  const auto first = decide(owner, env, raw0);
  check(first.units.size() == 1 && first.units[0].op == Op::DIG &&
            first.sources[0].source_step == -1 &&
            first.required_purchases.size() == 1 &&
            first.prefix_authority.size() == 1 &&
            first.prefix_authority[0].actor == 0,
        "trigger did not produce certified DIG and bound hard seed order");
  const auto applied = apply(env, 0, raw0, first);
  check(applied.receipts.size() == 1 && applied.receipts[0].filled == 0,
        "fixture did not produce exact zero fill");

  PlayerAction raw1;
  raw1.units.push_back({Op::EAST, Item::NONE, 1});
  const auto second = decide(owner, env, raw1, applied.receipts,
                             applied.action_receipts);
  check(same_action(second.units[0], raw1.units[0]) &&
            second.sources[0].source_step == env.step_count() &&
            owner.plot_debts().size() == 1 &&
            owner.audit().purchase_zero_fills == 1 &&
            owner.audit().move_tokens_delayed == 0,
        "zero fill delayed/lost MOVE or retired persistent debt");
}

void partial_fill_is_exact_and_current_pass_closes_lifecycle() {
  auto env = weed_state(10);
  online::OnlineElasticRepairOwner owner;
  const auto raw0 = trigger(2);
  const auto first = decide(owner, env, raw0);
  const auto applied0 = apply(env, 0, raw0, first);
  check(applied0.receipts.size() == 1 &&
            applied0.receipts[0].requested == 2 &&
            applied0.receipts[0].filled == 1,
        "fixture did not produce an exact partial market fill");

  PlayerAction pass;
  pass.units.push_back({});
  const auto plant = decide(owner, env, pass, applied0.receipts,
                            applied0.action_receipts);
  check(plant.units[0].op == Op::PLANT &&
            plant.sources[0].source_step == -1 &&
            owner.audit().purchase_partial_fills == 1,
        "partial fill was not handed to current-slot PLANT");
  const auto applied_plant = apply(env, 0, pass, plant);
  const auto water = decide(owner, env, pass, {},
                            applied_plant.action_receipts);
  check(water.units[0].op == Op::WATER,
        "post-PLANT receipt did not compile WATER");
  const auto applied_water = apply(env, 0, pass, water);
  const auto closed = decide(owner, env, pass, {},
                             applied_water.action_receipts);
  check(closed.units[0].op == Op::PASS && owner.plot_debts().empty() &&
            owner.audit().plot_debts_closed == 1,
        "watered crop did not retire plot-owned debt");
}

void full_fill_debt_survives_midnight_and_revisit() {
  auto env = weed_state(20);
  online::OnlineElasticRepairOwner owner;
  const auto raw0 = trigger(2);
  const auto first = decide(owner, env, raw0);
  const auto applied0 = apply(env, 0, raw0, first);
  auto receipts = applied0.receipts;
  check(receipts.size() == 1 && receipts[0].filled == 2,
        "fixture did not produce full fill");

  PlayerAction east;
  east.units.push_back({Op::EAST, Item::NONE, 1});
  const auto move_away = decide(owner, env, east, receipts,
                                applied0.action_receipts);
  check(same_action(move_away.units[0], east.units[0]) &&
            owner.audit().purchase_full_fills == 1,
        "full receipt improperly delayed uncertified MOVE");
  static_cast<void>(apply(env, 0, east, move_away));
  PlayerAction pass;
  pass.units.push_back({});
  while (env.step_count() < 48) {
    const auto unchanged = decide(owner, env, pass);
    check(unchanged.units[0].op == Op::PASS,
          "off-plot debt changed a raw PASS");
    static_cast<void>(apply(env, 0, pass, unchanged));
  }
  check(owner.plot_debts().size() == 1,
        "plot debt did not survive midnight");

  const auto leave_again = decide(owner, env, east);
  check(same_action(leave_again.units[0], east.units[0]),
        "cross-day debt replaced an uncertified MOVE");
  static_cast<void>(apply(env, 0, east, leave_again));
  PlayerAction west;
  west.units.push_back({Op::WEST, Item::NONE, 1});
  const auto return_move = decide(owner, env, west);
  check(same_action(return_move.units[0], west.units[0]),
        "cross-day debt replaced revisit MOVE");
  static_cast<void>(apply(env, 0, west, return_move));
  const auto dig = decide(owner, env, pass);
  check(dig.units[0].op == Op::DIG,
        "revisited cross-day weed did not recompile DIG");
  const auto applied_dig = apply(env, 0, pass, dig);
  const auto plant = decide(owner, env, pass, {},
                            applied_dig.action_receipts);
  check(plant.units[0].op == Op::PLANT,
        "cross-day DIG receipt did not recompile PLANT");
  const auto applied_plant = apply(env, 0, pass, plant);
  const auto water = decide(owner, env, pass, {},
                            applied_plant.action_receipts);
  check(water.units[0].op == Op::WATER,
        "cross-day PLANT receipt did not recompile WATER");
  const auto applied_water = apply(env, 0, pass, water);
  static_cast<void>(decide(owner, env, pass, {},
                           applied_water.action_receipts));
  check(owner.plot_debts().empty(), "cross-day lifecycle debt was not closed");
}

online::FrozenDaySuffixCertificate certificate() {
  online::FrozenDaySuffixCertificate out;
  out.player = 0;
  out.day = 1;
  out.actor = 0;
  out.actor_generation = (std::uint64_t{2} << 32U) | 1U;
  out.suffix_start_step = 24;
  out.issuer_generation = 9001;
  out.sinks_irrevocable = true;
  for (int step = 24; step < 48; ++step) {
    online::FrozenSlot slot;
    slot.source_step = step;
    if (step == 24) {
      slot.source_action = {Op::WATER, Item::WHEAT, 1};
      slot.kind = online::FrozenSlotKind::CertifiedSink;
    } else if (step == 25) {
      slot.source_action = {Op::EAST, Item::NONE, 1};
      slot.kind = online::FrozenSlotKind::MoveToken;
    } else {
      slot.source_action = {};
      slot.kind = step <= 27
          ? online::FrozenSlotKind::CertifiedSink
          : online::FrozenSlotKind::HardSemanticObligation;
    }
    out.slots.push_back(slot);
  }
  out.content_hash = online::frozen_suffix_hash(out);
  return out;
}

void trusted_frozen_suffix_allows_atomic_elastic_close() {
  auto env = weed_state(10);
  online::OnlineElasticRepairOwner owner;
  auto frozen = certificate();
  check(owner.install_frozen_suffix(frozen),
        "valid frozen suffix certificate was rejected");
  frozen.content_hash ^= 1U;
  check(!owner.install_frozen_suffix(frozen),
        "tampered frozen suffix certificate was accepted");

  const auto raw0 = trigger(1);
  const auto dig = decide(owner, env, raw0);
  const auto applied_dig = apply(env, 0, raw0, dig);
  auto receipts = applied_dig.receipts;
  PlayerAction east;
  east.units.push_back({Op::EAST, Item::NONE, 1});
  const auto plant = decide(owner, env, east, receipts,
                            applied_dig.action_receipts);
  check(plant.units[0].op == Op::PLANT &&
            plant.sources[0].source_step == -1,
        "trusted capacity did not delay MOVE for PLANT");
  const auto applied_plant = apply(env, 0, east, plant);

  PlayerAction pass;
  pass.units.push_back({});
  const auto water = decide(owner, env, pass, {},
                            applied_plant.action_receipts);
  check(water.units[0].op == Op::WATER,
        "trusted sink did not finish WATER before MOVE");
  const auto applied_water = apply(env, 0, pass, water);
  const auto replay = decide(owner, env, pass, {},
                             applied_water.action_receipts);
  check(replay.units[0].op == Op::EAST &&
            replay.sources[0].source_step == 25 &&
            same_action(replay.sources[0].source_action, east.units[0]) &&
            owner.audit().move_tokens_delayed == 1 &&
            owner.audit().move_tokens_replayed == 1,
        "deferred MOVE lost exact source identity/order");
}

void final_hour_prioritizes_deferred_move_replay() {
  auto env = weed_state(10);
  online::OnlineElasticRepairOwner owner;
  const auto raw0 = trigger(1);
  const auto dig = decide(owner, env, raw0);
  const auto applied_dig = apply(env, 0, raw0, dig);
  PlayerAction pass;
  pass.units.push_back({});
  const auto plant = decide(owner, env, pass, applied_dig.receipts,
                            applied_dig.action_receipts);
  const auto applied_plant = apply(env, 0, pass, plant);
  PlayerAction east;
  east.units.push_back({Op::EAST, Item::NONE, 1});
  const auto leave = decide(owner, env, east, {},
                            applied_plant.action_receipts);
  check(same_action(leave.units[0], east.units[0]),
        "uncertified early MOVE was delayed");
  static_cast<void>(apply(env, 0, east, leave));
  while (env.step_count() < 45) {
    const auto unchanged = decide(owner, env, pass);
    static_cast<void>(apply(env, 0, pass, unchanged));
  }
  PlayerAction west;
  west.units.push_back({Op::WEST, Item::NONE, 1});
  const auto return_to_plot = decide(owner, env, west);
  check(same_action(return_to_plot.units[0], west.units[0]),
        "return MOVE was changed without certificate");
  static_cast<void>(apply(env, 0, west, return_to_plot));

  online::FrozenDaySuffixCertificate frozen;
  frozen.player = 0;
  frozen.day = 1;
  frozen.actor = 0;
  frozen.actor_generation = (std::uint64_t{2} << 32U) | 1U;
  frozen.suffix_start_step = 24;
  frozen.issuer_generation = 9002;
  frozen.sinks_irrevocable = true;
  for (int step = 24; step < 48; ++step) {
    online::FrozenSlot slot;
    slot.source_step = step;
    if (step == 24)
      slot.source_action = {Op::WATER, Item::WHEAT, 1};
    else if (step == 26 || step == 46)
      slot.source_action = {Op::EAST, Item::NONE, 1};
    else if (step == 45)
      slot.source_action = {Op::WEST, Item::NONE, 1};
    else
      slot.source_action = {};
    if (step == 26 || step == 45 || step == 46)
      slot.kind = online::FrozenSlotKind::MoveToken;
    else if (step == 47)
      slot.kind = online::FrozenSlotKind::CertifiedSink;
    else
      slot.kind = online::FrozenSlotKind::HardSemanticObligation;
    frozen.slots.push_back(slot);
  }
  frozen.content_hash = online::frozen_suffix_hash(frozen);
  check(owner.install_frozen_suffix(frozen),
        "hour23 fixture certificate was rejected");

  const auto delayed_water = decide(owner, env, east);
  check(env.hour() == 22 && delayed_water.units[0].op == Op::WATER,
        "t22 certified capacity did not delay MOVE for WATER");
  const auto applied_water = apply(env, 0, east, delayed_water);
  const auto final_replay = decide(owner, env, pass, {},
                                   applied_water.action_receipts);
  check(env.hour() == 23 && final_replay.units[0].op == Op::EAST &&
            final_replay.sources[0].source_step == 46,
        "hour23 did not prioritize exact deferred MOVE replay");
  const auto applied_replay = apply(env, 0, pass, final_replay);
  const auto after_midnight = decide(owner, env, pass, {},
                                     applied_replay.action_receipts);
  check(after_midnight.units[0].op == Op::PASS &&
            owner.audit().midnight_fail_closed == 0 &&
            owner.plot_debts().empty(),
        "hour23 replay left token/debt across midnight");
}

void full_fingerprint_and_player1_prefix_are_authoritative() {
  Simulator base({}, 8821);
  {
    auto changed = base;
    auto& tile = const_cast<fastkag::Farm&>(changed.farms()[0]).tiles[0];
    tile.fertilized_until_day = 19;
    check(fork_api::full_unit_phase_state_fingerprint(base) !=
              fork_api::full_unit_phase_state_fingerprint(changed),
          "FERTILIZE hidden field was absent from owner authority helper");
  }
  {
    auto changed = base;
    auto& order = const_cast<fastkag::PrivateState&>(
        changed.privates()[0]).inventory_order[0];
    order.push_back(static_cast<std::int8_t>(Item::WHEAT));
    check(fork_api::full_unit_phase_state_fingerprint(base) !=
              fork_api::full_unit_phase_state_fingerprint(changed),
          "DROP inventory_order-only state was absent from authority helper");
  }

  auto env = weed_state(10);
  online::OnlineElasticRepairOwner owner;
  std::array<PlayerAction, 2> joint;
  joint[0].units.push_back({Op::NORTH, Item::NONE, 1});
  joint[1] = trigger(1);
  const auto decision = owner.decide(
      {env, 1, env.step_count(), joint[1], joint, {}, {}});
  check(decision.units[0].op == Op::DIG &&
            decision.prefix_authority.size() == 1,
        "player1 trigger did not produce an authorized repair");
  auto final_joint = joint;
  final_joint[1].units = decision.units;
  const auto expected = fork_api::post_unit_prefix_state_fingerprint(
      env, 1, final_joint, 0);
  check(decision.prefix_authority[0].post_prefix_state_fingerprint == expected,
        "player1 authority omitted official player0 prefix");
  final_joint[0].units[0] = {};
  check(expected != fork_api::post_unit_prefix_state_fingerprint(
                        env, 1, final_joint, 0),
        "player1 prefix token ignored earlier player0 unit effects");
}

}  // namespace

int main() try {
  zero_fill_preserves_move_and_debt_without_certificate();
  partial_fill_is_exact_and_current_pass_closes_lifecycle();
  full_fill_debt_survives_midnight_and_revisit();
  trusted_frozen_suffix_allows_atomic_elastic_close();
  final_hour_prioritizes_deferred_move_replay();
  full_fingerprint_and_player1_prefix_are_authoritative();
  std::cout << "online elastic repair owner: 6 fixture groups passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
