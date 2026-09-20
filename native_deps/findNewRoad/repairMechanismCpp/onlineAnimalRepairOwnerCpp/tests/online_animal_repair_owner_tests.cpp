#include "online_animal_repair_owner.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace owner_ns = g001::online_animal_repair;
namespace fork_ns = g001::repair_fork;
namespace persistent = g001::persistent_production;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
using fastkag::TileKind;

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::size_t item(Item value) {
  return static_cast<std::size_t>(static_cast<int>(value));
}

bool same_action(const Action& left, const Action& right) {
  return left.op == right.op && left.item == right.item &&
      left.quantity == right.quantity;
}

bool same_action_receipt(const fork_ns::ActionReceipt& left,
                         const fork_ns::ActionReceipt& right) {
  return left.submitted_step == right.submitted_step &&
      left.actor == right.actor &&
      left.manifest_generation == right.manifest_generation &&
      left.prefix_manifest_hash == right.prefix_manifest_hash &&
      left.post_prefix_state_fingerprint ==
          right.post_prefix_state_fingerprint &&
      same_action(left.emitted, right.emitted);
}

bool same_purchase_receipt(const fork_ns::PurchaseReceipt& left,
                           const fork_ns::PurchaseReceipt& right) {
  return left.debt_id == right.debt_id &&
      left.submitted_step == right.submitted_step &&
      left.operation == right.operation && left.item == right.item &&
      left.requested == right.requested && left.filled == right.filled &&
      left.market_slot == right.market_slot &&
      left.compile_status == right.compile_status;
}

fastkag::Config config() {
  fastkag::Config value;
  value.weed_spawn_chance = 0.0;
  value.episode_steps = 240;
  return value;
}

fastkag::Farm& farm(Simulator& simulator, int player = 0) {
  return const_cast<fastkag::Farm&>(simulator.farms()[player]);
}

fastkag::PrivateState& private_state(Simulator& simulator, int player = 0) {
  return const_cast<fastkag::PrivateState&>(simulator.privates()[player]);
}

fastkag::Tile& tile(Simulator& simulator, persistent::TileKey target,
                    int player = 0) {
  return farm(simulator, player).tiles[static_cast<std::size_t>(
      target.row * simulator.config().board_size + target.column)];
}

PlayerAction raw_with_units(int units = 1) {
  PlayerAction raw;
  raw.units.resize(static_cast<std::size_t>(units));
  return raw;
}

struct Receipts {
  std::vector<fork_ns::PurchaseReceipt> purchases;
  std::vector<fork_ns::ActionReceipt> actions;
};

struct TickResult {
  fork_ns::RepairDecision decision;
  fork_ns::MarketCompileResult compiled;
};

fork_ns::RepairDecision decide(owner_ns::OnlineAnimalRepairOwner& owner,
                               const Simulator& simulator,
                               const std::array<PlayerAction, 2>& raw_joint,
                               const Receipts& previous = {}, int player = 0) {
  const auto& raw = raw_joint[static_cast<std::size_t>(player)];
  const fork_ns::RepairContext context{
      simulator, player, simulator.step_count(), raw, raw_joint,
      previous.purchases, previous.actions};
  return owner.decide(context);
}

Receipts execute(Simulator& simulator, int player,
                 const std::array<PlayerAction, 2>& raw_joint,
                 const fork_ns::RepairDecision& decision,
                 fork_ns::MarketCompileResult* compiled_out = nullptr) {
  fork_ns::ExactMarketCompiler compiler;
  const auto compiled = compiler.compile(
      simulator, player, raw_joint[static_cast<std::size_t>(player)].market,
      decision.required_purchases);
  auto final_joint = raw_joint;
  auto& final = final_joint[static_cast<std::size_t>(player)];
  final.units = decision.units;
  final.market = compiled.market;
  const int submitted_step = simulator.step_count();
  simulator.step(final_joint);

  Receipts receipts;
  std::vector<int> remaining_fill(
      simulator.last_market_fills()[static_cast<std::size_t>(player)].begin(),
      simulator.last_market_fills()[static_cast<std::size_t>(player)].end());
  for (const auto& binding : compiled.bindings) {
    int filled = 0;
    if (binding.market_slot >= 0 &&
        binding.market_slot < static_cast<int>(remaining_fill.size())) {
      auto& available =
          remaining_fill[static_cast<std::size_t>(binding.market_slot)];
      filled = std::min(binding.requested, available);
      available -= filled;
    }
    receipts.purchases.push_back(
        {binding.debt_id, submitted_step, binding.operation, binding.item,
         binding.requested, filled, binding.market_slot, binding.status});
  }
  for (const auto& authority : decision.prefix_authority) {
    check(authority.actor >= 0 &&
              authority.actor < static_cast<int>(decision.units.size()),
          "test executor received invalid actor authority");
    receipts.actions.push_back(
        {submitted_step, authority.actor, authority.manifest_generation,
         authority.prefix_manifest_hash,
         authority.post_prefix_state_fingerprint,
         decision.units[static_cast<std::size_t>(authority.actor)]});
  }
  if (compiled_out) *compiled_out = compiled;
  return receipts;
}

TickResult tick(owner_ns::OnlineAnimalRepairOwner& owner, Simulator& simulator,
                const std::array<PlayerAction, 2>& raw_joint,
                Receipts& receipts, int player = 0) {
  TickResult result;
  result.decision = decide(owner, simulator, raw_joint, receipts, player);
  receipts = execute(simulator, player, raw_joint, result.decision,
                     &result.compiled);
  return result;
}

std::array<PlayerAction, 2> joint(PlayerAction focal) {
  std::array<PlayerAction, 2> result{};
  result[0] = std::move(focal);
  result[1] = raw_with_units();
  return result;
}

void purchase_zero_fill_and_no_slot_both_retry() {
  const persistent::TileKey target{4, 4};
  {
    Simulator simulator(config(), 1001);
    tile(simulator, target).kind = TileKind::COOP;
    owner_ns::OnlineAnimalRepairOwner owner;
    const auto opened = owner.ensure_objective(0, target, Item::GOOSE);
    check(opened.status == persistent::OpenStatus::Opened,
          "zero-fill objective did not open");
    Receipts receipts;
    const auto raw = joint(raw_with_units());
    const auto first = decide(owner, simulator, raw);
    check(first.required_purchases.size() == 1,
          "funded animal recovery did not request a purchase");
    farm(simulator).money = 0;
    receipts = execute(simulator, 0, raw, first);
    check(receipts.purchases.size() == 1 &&
              receipts.purchases[0].filled == 0,
          "fixture did not create an exact zero fill");
    farm(simulator).money = 3000;
    const auto retry = decide(owner, simulator, raw, receipts);
    check(retry.required_purchases.size() == 1 &&
              retry.required_purchases[0].item == Item::GOOSE &&
              owner.audit().purchase_retries == 1 &&
              retry.purchase_receipt_acks.size() == receipts.purchases.size() &&
              same_purchase_receipt(retry.purchase_receipt_acks[0],
                                    receipts.purchases[0]),
          "zero fill permanently abandoned or failed to ack animal debt");
  }

  {
    Simulator simulator(config(), 1002);
    tile(simulator, target).kind = TileKind::COOP;
    owner_ns::OnlineAnimalRepairOwner owner;
    check(owner.ensure_objective(0, target, Item::GOOSE).status ==
              persistent::OpenStatus::Opened,
          "no-slot objective did not open");
    auto full = raw_with_units();
    full.market.assign(
        static_cast<std::size_t>(simulator.config().max_market_orders),
        Action{Op::SELL, Item::EGG, 1});
    Receipts receipts;
    const auto first = tick(owner, simulator, joint(full), receipts);
    check(first.compiled.bindings.size() == 1 &&
              first.compiled.bindings[0].status ==
                  fork_ns::PurchaseCompileStatus::RejectedNoSlot,
          "fixture did not exercise compiler RejectedNoSlot");
    const auto retry = decide(owner, simulator, joint(raw_with_units()),
                              receipts);
    check(retry.required_purchases.size() == 1 &&
              owner.audit().purchase_retries == 1,
          "RejectedNoSlot silently consumed the animal debt");
  }

  {
    Simulator simulator(config(), 1003);
    const persistent::TileKey second_target{4, 3};
    tile(simulator, target).kind = TileKind::COOP;
    tile(simulator, second_target).kind = TileKind::COOP;
    owner_ns::OnlineAnimalRepairOwner owner;
    check(owner.ensure_objective(0, target, Item::GOOSE).status ==
              persistent::OpenStatus::Opened &&
              owner.ensure_objective(0, second_target, Item::GOOSE).status ==
                  persistent::OpenStatus::Opened,
          "partial-fill objectives did not open");
    auto hard = raw_with_units();
    hard.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 2});
    farm(simulator).money = 300;
    Receipts receipts;
    const auto first = tick(owner, simulator, joint(hard), receipts);
    check(first.decision.required_purchases.size() == 2 &&
              receipts.purchases.size() == 2 &&
              receipts.purchases[0].requested == 2 &&
              receipts.purchases[0].filled == 1 &&
              receipts.purchases[1].filled == 0,
          "fixture did not create partial-fill plus unfilled target debt");
    farm(simulator).money = 300;
    auto retry_raw = raw_with_units();
    retry_raw.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
    const auto retry = decide(owner, simulator, joint(retry_raw), receipts);
    check(retry.required_purchases.size() == 1 &&
              retry.required_purchases[0].item == Item::GOOSE,
          "partial order fill lost the still-unfilled target objective");
  }
}

void move_is_byte_exact_and_lower_slot_prefix_is_authoritative() {
  Simulator simulator(config(), 2001);
  const persistent::TileKey target{4, 4};
  tile(simulator, target).kind = TileKind::COOP;
  owner_ns::OnlineAnimalRepairOwner owner;
  check(owner.ensure_objective(0, target, Item::GOOSE).status ==
            persistent::OpenStatus::Opened,
        "move fixture objective did not open");
  Receipts receipts;
  tick(owner, simulator, joint(raw_with_units()), receipts);

  farm(simulator).hands.push_back({4, 4});
  private_state(simulator).inventories.push_back({});
  private_state(simulator).inventory_order.push_back({});
  auto raw = raw_with_units(2);
  raw.units[0] = {Op::EAST, Item::NONE, 1};
  const auto before = simulator.farms()[0].farmer;
  const auto decision = decide(owner, simulator, joint(raw), receipts);
  check(decision.units.size() == 2 &&
            same_action(decision.units[0], raw.units[0]) &&
            decision.units[1].op == Op::PICKUP &&
            decision.prefix_authority.size() == 1 &&
            decision.prefix_authority[0].actor == 1,
        "repair did not preserve MOVE while using a later PASS slot");
  auto final_joint = joint(raw);
  final_joint[0].units = decision.units;
  const auto expected_post = fork_ns::post_unit_prefix_state_fingerprint(
      simulator, 0, final_joint, 1);
  check(decision.prefix_authority[0].prefix_manifest_hash ==
            fork_ns::unit_prefix_manifest_hash(decision.units, 1) &&
            decision.prefix_authority[0].post_prefix_state_fingerprint ==
                expected_post,
        "actor PrefixAuthority was not derived from exact lower MOVE");
  receipts = execute(simulator, 0, joint(raw), decision);
  check(simulator.farms()[0].farmer.x == before.x + 1 &&
            simulator.farms()[0].farmer.y == before.y &&
            owner.audit().move_changes == 0,
        "MOVE direction/slot changed under animal repair");

  auto follow = raw_with_units(2);
  follow.units[0] = {Op::WEST, Item::NONE, 1};
  const auto next = decide(owner, simulator, joint(follow), receipts);
  check(next.receipt_acks.size() == receipts.actions.size() &&
            same_action_receipt(next.receipt_acks[0], receipts.actions[0]) &&
            owner.audit().unit_receipt_successes == 1,
        "evaluator-issued action receipt was not exactly acknowledged");
}

void shared_shed_or_wheat_obligation_blocks_takeover() {
  Simulator simulator(config(), 3001);
  const persistent::TileKey target{4, 4};
  tile(simulator, target).kind = TileKind::COOP;
  owner_ns::OnlineAnimalRepairOwner owner;
  check(owner.ensure_objective(0, target, Item::GOOSE).status ==
            persistent::OpenStatus::Opened,
        "shared-resource objective did not open");
  Receipts receipts;
  tick(owner, simulator, joint(raw_with_units()), receipts);
  farm(simulator).hands.push_back({4, 4});
  private_state(simulator).inventories.push_back({});
  private_state(simulator).inventory_order.push_back({});
  private_state(simulator).shed[item(Item::WHEAT)] = 1;
  auto raw = raw_with_units(2);
  raw.units[1] = {Op::PICKUP, Item::WHEAT, 1};
  const auto decision = decide(owner, simulator, joint(raw), receipts);
  check(decision.units.size() == raw.units.size() &&
            std::equal(decision.units.begin(), decision.units.end(),
                       raw.units.begin(), same_action) &&
            decision.prefix_authority.empty() &&
            owner.audit().raw_obligation_blocks > 0,
        "shared shed/wheat action was overwritten by lifecycle repair");
}

void wrong_structure_place_fallback_returns_to_pickup() {
  Simulator simulator(config(), 4001);
  const persistent::TileKey target{4, 4};
  tile(simulator, target).kind = TileKind::COOP;
  owner_ns::OnlineAnimalRepairOwner owner;
  const auto opened = owner.ensure_objective(0, target, Item::GOOSE);
  check(opened.status == persistent::OpenStatus::Opened,
        "wrong-structure objective did not open");
  Receipts receipts;
  tick(owner, simulator, joint(raw_with_units()), receipts);
  const auto pickup = tick(owner, simulator, joint(raw_with_units()), receipts);
  check(pickup.decision.units[0].op == Op::PICKUP,
        "purchase handoff did not reach PICKUP");

  const auto place = decide(owner, simulator, joint(raw_with_units()), receipts);
  check(place.units[0].op == Op::PLACE,
        "successful PICKUP did not reach PLACE");
  tile(simulator, target) = {};
  receipts = execute(simulator, 0, joint(raw_with_units()), place);
  check(private_state(simulator).shed[item(Item::GOOSE)] == 1 &&
            private_state(simulator).inventories[0][item(Item::GOOSE)] == 0,
        "official wrong-structure PLACE did not exercise shed fallback");

  const auto recovered = decide(owner, simulator, joint(raw_with_units()),
                                receipts);
  check(recovered.units[0].op == Op::PICKUP &&
            owner.ledger().animal_audit().external_inventory_reconciles > 0 &&
            owner.ledger().objective(opened.objective_id)->status ==
                persistent::Status::Open,
        "shed fallback lost PLACE debt instead of returning to PICKUP");
}

void hour23_refuses_unit_commit_and_rebinds_next_day() {
  Simulator simulator(config(), 5001);
  const persistent::TileKey target{4, 4};
  tile(simulator, target).kind = TileKind::COOP;
  owner_ns::OnlineAnimalRepairOwner owner;
  const auto opened = owner.ensure_objective(0, target, Item::GOOSE);
  check(opened.status == persistent::OpenStatus::Opened,
        "hour23 objective did not open");
  Receipts receipts;
  tick(owner, simulator, joint(raw_with_units()), receipts);
  auto moving = raw_with_units();
  moving.units[0] = {Op::EAST, Item::NONE, 1};
  tick(owner, simulator, joint(moving), receipts);
  while (simulator.step_count() < 23) {
    std::array<PlayerAction, 2> pass{};
    pass[0] = raw_with_units();
    pass[1] = raw_with_units();
    simulator.step(pass);
  }
  const auto at_boundary = decide(owner, simulator, joint(raw_with_units()));
  check(at_boundary.units[0].op == Op::PASS &&
            at_boundary.prefix_authority.empty() &&
            owner.audit().hour23_blocks > 0 &&
            owner.ledger().objective(opened.objective_id)->status ==
                persistent::Status::Open,
        "hour23 ambiguity was treated as an exact unit receipt");
  Receipts none;
  tick(owner, simulator, joint(raw_with_units()), none);
  check(simulator.day() == 1, "fixture did not cross midnight");
  const auto next_day = decide(owner, simulator, joint(raw_with_units()));
  check(next_day.units[0].op == Op::PICKUP &&
            next_day.prefix_authority.size() == 1 &&
            next_day.prefix_authority[0].manifest_generation != 0,
        "cross-day debt did not rebind to the new farmer generation");
}

void exact_simulator_lifecycle_reaches_first_yield() {
  Simulator simulator(config(), 6001);
  const persistent::TileKey target{4, 4};
  tile(simulator, target).kind = TileKind::COOP;
  private_state(simulator).shed[item(Item::WHEAT)] = 8;
  owner_ns::OnlineAnimalRepairOwner owner;
  const auto opened = owner.ensure_objective(0, target, Item::GOOSE);
  check(opened.status == persistent::OpenStatus::Opened,
        "lifecycle objective did not open");
  Receipts receipts;

  const auto buy = tick(owner, simulator, joint(raw_with_units()), receipts);
  check(buy.decision.required_purchases.size() == 1,
        "lifecycle BUY was not emitted");
  const auto pickup = tick(owner, simulator, joint(raw_with_units()), receipts);
  check(pickup.decision.units[0].op == Op::PICKUP,
        "lifecycle PICKUP was not emitted");
  const auto place = tick(owner, simulator, joint(raw_with_units()), receipts);
  check(place.decision.units[0].op == Op::PLACE,
        "lifecycle PLACE was not emitted");

  bool saw_feed = false;
  bool saw_care = false;
  bool saw_harvest = false;
  while (simulator.step_count() <= 97) {
    auto raw = raw_with_units();
    const auto& target_tile = tile(simulator, target);
    const bool animal_present = target_tile.kind == TileKind::ANIMAL &&
        target_tile.animal == Item::GOOSE;
    if (animal_present && simulator.hour() == 0 &&
        target_tile.yield_units == 0)
      raw.units[0] = {Op::PICKUP, Item::WHEAT, 1};
    const auto result = tick(owner, simulator, joint(raw), receipts);
    saw_feed = saw_feed || result.decision.units[0].op == Op::FEED;
    saw_care = saw_care || result.decision.units[0].op == Op::CARE;
    saw_harvest = saw_harvest || result.decision.units[0].op == Op::HARVEST;
    if (owner.ledger().objective(opened.objective_id)->status ==
        persistent::Status::Complete)
      break;
  }
  const auto final = owner.ledger().objective(opened.objective_id);
  check(saw_feed && saw_care && saw_harvest && final &&
            final->status == persistent::Status::Complete &&
            private_state(simulator).inventories[0][item(Item::EGG)] > 0,
        "Simulator chain did not confirm FEED->CARE->FIRST_YIELD");
  check(owner.audit().move_changes == 0,
        "full lifecycle fixture changed a MOVE");
}

}  // namespace

int main() {
  using Test = std::pair<const char*, void (*)()>;
  const std::array<Test, 6> tests{
      Test{"purchase_zero_fill_and_no_slot_both_retry",
                purchase_zero_fill_and_no_slot_both_retry},
      Test{"move_is_byte_exact_and_lower_slot_prefix_is_authoritative",
                move_is_byte_exact_and_lower_slot_prefix_is_authoritative},
      Test{"shared_shed_or_wheat_obligation_blocks_takeover",
                shared_shed_or_wheat_obligation_blocks_takeover},
      Test{"wrong_structure_place_fallback_returns_to_pickup",
                wrong_structure_place_fallback_returns_to_pickup},
      Test{"hour23_refuses_unit_commit_and_rebinds_next_day",
                hour23_refuses_unit_commit_and_rebinds_next_day},
      Test{"exact_simulator_lifecycle_reaches_first_yield",
                exact_simulator_lifecycle_reaches_first_yield},
  };
  int passed = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      ++passed;
      std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
      std::cerr << "FAIL " << name << ": " << error.what() << '\n';
      return 1;
    }
  }
  std::cout << "PASS " << passed << '/' << tests.size()
            << " typed online animal repair fixtures\n";
  return 0;
}
