#include "repair_owner_composer.hpp"
#include "transactional_animal_repair_owner.hpp"
#include "transactional_crop_repair_owner.hpp"
#include "transactional_owner_composer.hpp"
#include "transactional_repair_owner.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace composer_ns = g001::repair_owner_composer;
namespace fork_ns = g001::repair_fork;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;

namespace {

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

bool same_action(const Action &left, const Action &right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

struct Probe {
  int calls{};
  std::vector<std::size_t> action_receipt_counts;
  std::vector<std::size_t> purchase_receipt_counts;
};

struct TxProbe {
  int prepares{};
  int commits{};
  int aborts{};
  int settles{};
  bool validation_result{true};
  composer_ns::SettlementResult settlement_result{
      composer_ns::SettlementResult::AppliedSuccess};
  std::uint64_t generation{};
  int ledger_version{};
  int pending_receipts{};
  std::vector<composer_ns::CommitGrant> grants;
  std::vector<std::size_t> settled_actions;
  std::vector<std::size_t> settled_purchases;
};

class TxMock final : public composer_ns::TransactionalRepairOwner {
public:
  using Script = std::function<composer_ns::PreparedRepair(
      const fork_ns::RepairContext &, int)>;
  TxMock(std::shared_ptr<TxProbe> probe, Script script)
      : probe_(std::move(probe)), script_(std::move(script)) {}
  std::string name() const override { return "tx-mock"; }
  composer_ns::SettlementResult settle_owned(
      const fork_ns::RepairContext &,
      std::span<const fork_ns::ActionReceipt> actions,
      std::span<const fork_ns::PurchaseReceipt> purchases) override {
    ++probe_->settles;
    probe_->settled_actions.push_back(actions.size());
    probe_->settled_purchases.push_back(purchases.size());
    const auto consumed = static_cast<int>(actions.size() + purchases.size());
    if (consumed > 0) {
      ++probe_->ledger_version;
      probe_->pending_receipts -= consumed;
    }
    return probe_->settlement_result;
  }
  composer_ns::SettlementResult validate_settle(
      const fork_ns::RepairContext &,
      std::span<const fork_ns::ActionReceipt>,
      std::span<const fork_ns::PurchaseReceipt>) const override {
    return probe_->settlement_result;
  }
  composer_ns::PreparedRepair
  prepare(const fork_ns::RepairContext &context) const override {
    return script_(context, probe_->prepares++);
  }
  bool commit(std::uint64_t, const composer_ns::CommitGrant &grant) override {
    ++probe_->commits;
    probe_->grants.push_back(grant);
    if (!grant.actions.empty())
      probe_->generation =
          grant.actions.back().final_authority.manifest_generation;
    probe_->pending_receipts +=
        static_cast<int>(grant.actions.size() + grant.purchases.size());
    return true;
  }
  bool validate_commit(std::uint64_t,
                       const composer_ns::CommitGrant &) const override {
    return probe_->validation_result;
  }
  void abort(std::uint64_t) noexcept override { ++probe_->aborts; }

private:
  std::shared_ptr<TxProbe> probe_;
  Script script_;
};

class SettlementGate final : public composer_ns::TransactionalRepairOwner {
public:
  SettlementGate(
      std::unique_ptr<composer_ns::TransactionalRepairOwner> owner,
      std::shared_ptr<bool> reject)
      : owner_(std::move(owner)), reject_(std::move(reject)) {}
  std::string name() const override { return owner_->name(); }
  composer_ns::SettlementResult settle_owned(
      const fork_ns::RepairContext &context,
      std::span<const fork_ns::ActionReceipt> actions,
      std::span<const fork_ns::PurchaseReceipt> purchases) override {
    return owner_->settle_owned(context, actions, purchases);
  }
  composer_ns::SettlementResult validate_settle(
      const fork_ns::RepairContext &context,
      std::span<const fork_ns::ActionReceipt> actions,
      std::span<const fork_ns::PurchaseReceipt> purchases) const override {
    return *reject_ ? composer_ns::SettlementResult::ProtocolInvalid
                    : owner_->validate_settle(context, actions, purchases);
  }
  composer_ns::PreparedRepair
  prepare(const fork_ns::RepairContext &context) const override {
    return owner_->prepare(context);
  }
  composer_ns::PreparedRepair
  prepare(const fork_ns::RepairContext &context,
          composer_ns::TypedRepairInput input) const override {
    return owner_->prepare(context, input);
  }
  bool validate_commit(
      std::uint64_t token,
      const composer_ns::CommitGrant &grant) const override {
    return owner_->validate_commit(token, grant);
  }
  bool commit(std::uint64_t token,
              const composer_ns::CommitGrant &grant) override {
    return owner_->commit(token, grant);
  }
  void abort(std::uint64_t token) noexcept override { owner_->abort(token); }

private:
  std::unique_ptr<composer_ns::TransactionalRepairOwner> owner_;
  std::shared_ptr<bool> reject_;
};

composer_ns::PreparedRepair tx_raw(const fork_ns::RepairContext &context,
                                   std::uint64_t token) {
  composer_ns::PreparedRepair out;
  out.token = token;
  out.units = context.raw_g001.units;
  for (std::size_t actor = 0; actor < out.units.size(); ++actor)
    out.sources.push_back(
        {static_cast<int>(actor), context.step, out.units[actor]});
  return out;
}

composer_ns::PreparedRepair tx_unit(const fork_ns::RepairContext &context,
                                    std::uint64_t token, int actor,
                                    Action action) {
  auto out = tx_raw(context, token);
  out.units[static_cast<std::size_t>(actor)] = action;
  out.sources[static_cast<std::size_t>(actor)] = {actor, -1, action};
  out.claimed_actors.push_back(actor);
  return out;
}

class MockOwner final : public fork_ns::RepairOwner {
public:
  using Script = std::function<fork_ns::RepairDecision(
      const fork_ns::RepairContext &, int)>;

  MockOwner(std::string label, std::shared_ptr<Probe> probe, Script script)
      : label_(std::move(label)), probe_(std::move(probe)),
        script_(std::move(script)) {}

  std::string name() const override { return label_; }

  fork_ns::RepairDecision
  decide(const fork_ns::RepairContext &context) override {
    const int call = probe_->calls++;
    probe_->action_receipt_counts.push_back(
        context.previous_action_receipts.size());
    probe_->purchase_receipt_counts.push_back(
        context.previous_purchase_receipts.size());
    auto out = script_(context, call);
    out.receipt_acks.assign(context.previous_action_receipts.begin(),
                            context.previous_action_receipts.end());
    out.purchase_receipt_acks.assign(context.previous_purchase_receipts.begin(),
                                     context.previous_purchase_receipts.end());
    return out;
  }

private:
  std::string label_;
  std::shared_ptr<Probe> probe_;
  Script script_;
};

fork_ns::RepairDecision raw_proposal(const fork_ns::RepairContext &context) {
  fork_ns::RepairDecision out;
  out.units = context.raw_g001.units;
  for (std::size_t actor = 0; actor < out.units.size(); ++actor)
    out.sources.push_back(
        {static_cast<int>(actor), context.step, out.units[actor]});
  return out;
}

fork_ns::RepairDecision unit_proposal(const fork_ns::RepairContext &context,
                                      int actor, Action action,
                                      std::uint64_t generation) {
  auto out = raw_proposal(context);
  out.units[static_cast<std::size_t>(actor)] = action;
  out.sources[static_cast<std::size_t>(actor)] = {actor, -1, action};
  out.prefix_authority.push_back(
      {actor, generation, 1000 + generation, 2000 + generation});
  return out;
}

std::unique_ptr<MockOwner> mock(std::string name, std::shared_ptr<Probe> probe,
                                MockOwner::Script script) {
  return std::make_unique<MockOwner>(std::move(name), std::move(probe),
                                     std::move(script));
}

fastkag::Config config(int episode_steps = 240) {
  fastkag::Config cfg;
  cfg.episode_steps = episode_steps;
  cfg.weed_spawn_chance = 0.0;
  return cfg;
}

PlayerAction units(int count) {
  PlayerAction out;
  out.units.resize(static_cast<std::size_t>(count));
  return out;
}

fork_ns::RepairDecision
decide(composer_ns::RepairOwnerComposer &composer, const Simulator &env,
       int player, const std::array<PlayerAction, 2> &joint,
       const std::vector<fork_ns::PurchaseReceipt> &purchases = {},
       const std::vector<fork_ns::ActionReceipt> &actions = {}) {
  return composer.decide({env, player, env.step_count(),
                          joint[static_cast<std::size_t>(player)], joint,
                          purchases, actions});
}

std::vector<fork_ns::ActionReceipt>
action_receipts(int step, const fork_ns::RepairDecision &decision) {
  std::vector<fork_ns::ActionReceipt> out;
  for (const auto &authority : decision.prefix_authority)
    out.push_back({step, authority.actor, authority.manifest_generation,
                   authority.prefix_manifest_hash,
                   authority.post_prefix_state_fingerprint,
                   decision.units[static_cast<std::size_t>(authority.actor)]});
  return out;
}

PlayerAction selected_action(const Simulator &env, int player,
                             const PlayerAction &raw,
                             const fork_ns::RepairDecision &decision) {
  auto out = raw;
  out.units = decision.units;
  out.market = fork_ns::ExactMarketCompiler()
                   .compile(env, player, raw.market,
                            decision.required_purchases)
                   .market;
  return out;
}

void advance(Simulator &env, std::array<PlayerAction, 2> joint,
             const fork_ns::RepairDecision *decision = nullptr,
             int player = 0) {
  if (decision)
    joint[static_cast<std::size_t>(player)].units = decision->units;
  env.step(joint);
}

void same_actor_conflict_is_raw_and_permanent() {
  Simulator env(config(), 1001);
  auto crop_probe = std::make_shared<Probe>();
  auto animal_probe = std::make_shared<Probe>();
  composer_ns::RepairOwnerComposer composer(
      mock("crop", crop_probe,
           [](const auto &context, int) {
             return unit_proposal(context, 0, {Op::DIG, Item::NONE, 1}, 11);
           }),
      mock("animal", animal_probe, [](const auto &context, int) {
        return unit_proposal(context, 0, {Op::PICKUP, Item::GOOSE, 1}, 22);
      }));
  std::array<PlayerAction, 2> joint{units(1), units(1)};
  const auto result = decide(composer, env, 0, joint);
  check(result.units.size() == 1 && result.units[0].op == Op::PASS &&
            result.required_purchases.empty() &&
            result.prefix_authority.empty() && composer.fail_stopped() &&
            composer.audit().unit_conflicts == 1,
        "same actor did not fail closed to the complete raw manifest");
  advance(env, joint);
  const auto again = decide(composer, env, 0, joint);
  check(again.units[0].op == Op::PASS && crop_probe->calls == 1 &&
            animal_probe->calls == 1,
        "fail-stopped composer called a child with an orphaned ledger");
}

void different_actors_coexist_and_receipts_are_owned() {
  Simulator env(config(), 1002);
  auto crop_probe = std::make_shared<Probe>();
  auto animal_probe = std::make_shared<Probe>();
  composer_ns::RepairOwnerComposer composer(
      mock("crop", crop_probe,
           [](const auto &context, int call) {
             if (call == 0)
               return unit_proposal(context, 0, {Op::DIG, Item::NONE, 1}, 31);
             return raw_proposal(context);
           }),
      mock("animal", animal_probe, [](const auto &context, int call) {
        if (call == 0)
          return unit_proposal(context, 1, {Op::PICKUP, Item::GOOSE, 1}, 41);
        return raw_proposal(context);
      }));
  std::array<PlayerAction, 2> joint{units(2), units(1)};
  const int submitted = env.step_count();
  const auto result = decide(composer, env, 0, joint);
  check(result.units[0].op == Op::DIG && result.units[1].op == Op::PICKUP &&
            result.prefix_authority.size() == 2 &&
            result.prefix_authority[0].actor == 0 &&
            result.prefix_authority[1].actor == 1 &&
            result.sources[0].source_step == -1 &&
            result.sources[1].source_step == -1,
        "different actor proposals did not coexist deterministically");
  const auto receipts = action_receipts(submitted, result);
  advance(env, joint, &result);
  std::array<PlayerAction, 2> next{units(2), units(1)};
  const auto settled = decide(composer, env, 0, next, {}, receipts);
  check(settled.receipt_acks.size() == 2 &&
            crop_probe->action_receipt_counts ==
                std::vector<std::size_t>({0, 1}) &&
            animal_probe->action_receipt_counts ==
                std::vector<std::size_t>({0, 1}) &&
            composer.audit().translated_action_receipts == 2,
        "final receipts were not translated to their unique child owners");
}

void seed_animal_last_market_slot_conflict_fails_closed() {
  fastkag::Config cfg = config();
  cfg.max_market_orders = 10;
  Simulator env(cfg, 1003);
  auto crop_probe = std::make_shared<Probe>();
  auto animal_probe = std::make_shared<Probe>();
  composer_ns::RepairOwnerComposer composer(
      mock("crop", crop_probe,
           [](const auto &context, int) {
             auto out = raw_proposal(context);
             out.required_purchases.push_back(
                 {1, Op::BUY_SEED, Item::WHEAT, 1, context.step});
             return out;
           }),
      mock("animal", animal_probe, [](const auto &context, int) {
        auto out = raw_proposal(context);
        out.required_purchases.push_back(
            {1, Op::BUY_ANIMAL, Item::GOOSE, 1, context.step});
        return out;
      }));
  std::array<PlayerAction, 2> joint{units(1), units(1)};
  joint[0].units[0] = {Op::EAST, Item::NONE, 1};
  joint[0].market.resize(9);
  const auto result = decide(composer, env, 0, joint);
  check(result.required_purchases.empty() &&
            same_action(result.units[0], joint[0].units[0]) &&
            result.sources[0].source_step == env.step_count() &&
            same_action(result.sources[0].source_action, joint[0].units[0]) &&
            composer.fail_stopped() && composer.audit().market_conflicts == 1,
        "seed/animal sole-slot collision changed raw MOVE or was submitted");
}

void player1_authority_contains_player0_prefix() {
  Simulator env(config(), 1004);
  auto crop_probe = std::make_shared<Probe>();
  auto animal_probe = std::make_shared<Probe>();
  composer_ns::RepairOwnerComposer composer(
      mock("crop", crop_probe,
           [](const auto &context, int) {
             return unit_proposal(context, 0, {Op::DIG, Item::NONE, 1}, 51);
           }),
      mock("animal", animal_probe,
           [](const auto &context, int) { return raw_proposal(context); }));
  std::array<PlayerAction, 2> joint{units(1), units(1)};
  joint[0].units[0] = {Op::EAST, Item::NONE, 1};
  const auto result = decide(composer, env, 1, joint);
  auto final_joint = joint;
  final_joint[1].units = result.units;
  check(result.prefix_authority.size() == 1 &&
            result.prefix_authority[0].post_prefix_state_fingerprint ==
                fork_ns::post_unit_prefix_state_fingerprint(env, 1, final_joint,
                                                            0),
        "player1 authority omitted the official player0 unit phase");
}

void terminal_purchase_debt_is_not_silently_acked() {
  Simulator env(config(1), 1005);
  auto crop_probe = std::make_shared<Probe>();
  auto animal_probe = std::make_shared<Probe>();
  composer_ns::RepairOwnerComposer composer(
      mock("crop", crop_probe,
           [](const auto &context, int) {
             auto out = raw_proposal(context);
             out.required_purchases.push_back(
                 {9, Op::BUY_SEED, Item::CARROT, 1, context.step});
             return out;
           }),
      mock("animal", animal_probe,
           [](const auto &context, int) { return raw_proposal(context); }));
  std::array<PlayerAction, 2> joint{units(1), units(1)};
  const auto result = decide(composer, env, 0, joint);
  check(result.required_purchases.size() == 1 &&
            composer.outstanding_purchase_receipts() == 1 &&
            composer.audit().terminal_unacked_purchase_debts == 1 &&
            result.purchase_receipt_acks.empty(),
        "terminal purchase debt was lost or falsely acknowledged");
}

void full_raw_pass_through_is_byte_exact() {
  Simulator env(config(), 1006);
  auto probe0 = std::make_shared<Probe>();
  auto probe1 = std::make_shared<Probe>();
  composer_ns::RepairOwnerComposer composer(
      mock("crop", probe0,
           [](const auto &context, int) { return raw_proposal(context); }),
      mock("animal", probe1,
           [](const auto &context, int) { return raw_proposal(context); }));
  std::array<PlayerAction, 2> joint{units(3), units(1)};
  joint[0].units[0] = {Op::EAST, Item::NONE, 1};
  joint[0].units[1] = {Op::WATER, Item::STRAWBERRY, 1};
  joint[0].market.push_back({Op::SELL, Item::STRAWBERRY, 3});
  const auto result = decide(composer, env, 0, joint);
  check(result.units.size() == joint[0].units.size() &&
            std::equal(result.units.begin(), result.units.end(),
                       joint[0].units.begin(), same_action) &&
            result.required_purchases.empty() &&
            result.prefix_authority.empty() &&
            result.sources.size() == result.units.size() &&
            result.sources[0].source_step == env.step_count() &&
            same_action(result.sources[0].source_action, joint[0].units[0]) &&
            !composer.fail_stopped(),
        "raw pass-through changed action/source bytes");
}

void sealed_default_owners_are_linked_but_not_native_registered() {
  Simulator env(config(), 1007);
  composer_ns::RepairOwnerComposer composer;
  std::array<PlayerAction, 2> joint{units(1), units(1)};
  joint[0].units[0] = {Op::EAST, Item::NONE, 1};
  const auto result = decide(composer, env, 0, joint);
  check(result.units.size() == 1 &&
            same_action(result.units[0], joint[0].units[0]) &&
            result.sources.size() == 1 &&
            result.sources[0].source_step == env.step_count() &&
            !composer.fail_stopped(),
        "sealed default owner wiring did not preserve a normal raw MOVE");
}

void transactional_conflict_aborts_without_poisoning_next_prepare() {
  Simulator env(config(), 1101);
  auto crop = std::make_shared<TxProbe>();
  auto animal = std::make_shared<TxProbe>();
  composer_ns::TransactionalOwnerComposer composer(
      std::make_unique<TxMock>(crop,
                               [](const auto &context, int call) {
                                 return tx_unit(context, 100 + call, 0,
                                                {Op::DIG, Item::NONE, 1});
                               }),
      std::make_unique<TxMock>(animal, [](const auto &context, int call) {
        if (call == 0)
          return tx_unit(context, 200, 0, {Op::PICKUP, Item::GOOSE, 1});
        return tx_raw(context, 201);
      }));
  std::array<PlayerAction, 2> joint{units(1), units(1)};
  auto result =
      composer.decide({env, 0, env.step_count(), joint[0], joint, {}, {}});
  check(result.units[0].op == Op::PASS && crop->aborts == 1 &&
            animal->aborts == 1 && crop->commits == 0 && animal->commits == 0 &&
            !composer.fail_stopped(),
        "unscheduled transactional conflict committed without authority");
  advance(env, joint);
  result = composer.decide({env, 0, env.step_count(), joint[0], joint, {}, {}});
  check(result.units[0].op == Op::DIG && crop->commits == 1 &&
            animal->aborts == 2 && !composer.fail_stopped(),
        "aborted transactional conflict poisoned the next commit");
}

void transactional_different_actors_share_one_final_manifest() {
  Simulator env(config(), 1102);
  auto crop = std::make_shared<TxProbe>();
  auto animal = std::make_shared<TxProbe>();
  composer_ns::TransactionalOwnerComposer composer(
      std::make_unique<TxMock>(crop,
                               [](const auto &context, int) {
                                 return tx_unit(context, 301, 0,
                                                {Op::DIG, Item::NONE, 1});
                               }),
      std::make_unique<TxMock>(animal, [](const auto &context, int) {
        return tx_unit(context, 302, 1, {Op::PICKUP, Item::GOOSE, 1});
      }));
  std::array<PlayerAction, 2> joint{units(2), units(1)};
  const auto result =
      composer.decide({env, 0, env.step_count(), joint[0], joint, {}, {}});
  check(result.units[0].op == Op::DIG && result.units[1].op == Op::PICKUP &&
            result.prefix_authority.size() == 2 && crop->commits == 1 &&
            animal->commits == 1 &&
            crop->grants[0].final_units.size() ==
                animal->grants[0].final_units.size() &&
            std::equal(crop->grants[0].final_units.begin(),
                       crop->grants[0].final_units.end(),
                       animal->grants[0].final_units.begin(), same_action),
        "compatible transactional actors did not share final manifest");
}

void transactional_second_child_rejection_is_prevalidated_atomically() {
  Simulator env(config(), 1103);
  auto crop = std::make_shared<TxProbe>();
  auto animal = std::make_shared<TxProbe>();
  animal->validation_result = false;
  composer_ns::TransactionalOwnerComposer composer(
      std::make_unique<TxMock>(crop,
                               [](const auto &context, int) {
                                 return tx_unit(context, 401, 0,
                                                {Op::DIG, Item::NONE, 1});
                               }),
      std::make_unique<TxMock>(animal, [](const auto &context, int) {
        return tx_unit(context, 402, 1, {Op::PICKUP, Item::GOOSE, 1});
      }));
  std::array<PlayerAction, 2> joint{units(2), units(1)};
  const auto result =
      composer.decide({env, 0, env.step_count(), joint[0], joint, {}, {}});
  check(result.units[0].op == Op::PASS && result.units[1].op == Op::PASS &&
            crop->commits == 0 && animal->commits == 0 && crop->aborts == 1 &&
            animal->aborts == 1 && !composer.fail_stopped(),
        "second-child validation rejection half-committed first child");
}

void staged_native_reject_abort_retry_and_owned_receipts() {
  Simulator env(config(), 1107);
  auto crop = std::make_shared<TxProbe>();
  auto animal = std::make_shared<TxProbe>();
  composer_ns::TransactionalOwnerComposer composer(
      std::make_unique<TxMock>(crop, [](const auto &context, int call) {
        return tx_unit(context, 800 + call, 0,
                       {Op::DIG, Item::NONE, 1});
      }),
      std::make_unique<TxMock>(animal, [](const auto &context, int call) {
        return tx_unit(context, 900 + call, 1,
                       {Op::PICKUP, Item::GOOSE, 1});
      }));
  std::array<PlayerAction, 2> joint{units(2), units(1)};
  const fork_ns::RepairContext initial{env, 0, env.step_count(), joint[0],
                                       joint, {}, {}};

  const auto rejected = composer.prepare(initial);
  check(rejected.token != 0 && crop->commits == 0 && animal->commits == 0,
        "prepare published child state before native selection");
  const auto second_writer = composer.prepare(initial);
  check(second_writer.token == 0 && second_writer.decision.telemetry.fail_closed,
        "second writer did not fail closed");
  auto wrong = selected_action(env, 0, joint[0], rejected.decision);
  wrong.units[0] = {};
  check(!composer.finalize(rejected.token, wrong) && crop->commits == 0 &&
            animal->commits == 0 && composer.abort(rejected.token) &&
            !composer.abort(rejected.token),
        "native rejection changed state or stale abort succeeded");

  const auto retry = composer.prepare(initial);
  const auto exact = selected_action(env, 0, joint[0], retry.decision);
  check(retry.token != 0 &&
            retry.decision.prefix_authority.size() ==
                rejected.decision.prefix_authority.size() &&
            std::equal(retry.decision.prefix_authority.begin(),
                       retry.decision.prefix_authority.end(),
                       rejected.decision.prefix_authority.begin(),
                       [](const auto &left, const auto &right) {
                         return left.actor == right.actor &&
                                left.manifest_generation ==
                                    right.manifest_generation &&
                                left.prefix_manifest_hash ==
                                    right.prefix_manifest_hash &&
                                left.post_prefix_state_fingerprint ==
                                    right.post_prefix_state_fingerprint;
                       }) &&
            composer.finalize(retry.token, exact) &&
            crop->commits == 1 && animal->commits == 1 &&
            !composer.finalize(retry.token, exact),
        "exact retry did not publish both staged children exactly once");
  const int submitted = env.step_count();
  auto executed = joint;
  executed[0] = exact;
  env.step(executed);
  const auto receipts = action_receipts(submitted, retry.decision);
  const fork_ns::RepairContext next_context{
      env, 0, env.step_count(), joint[0], joint, {}, receipts};
  const auto crop_generation = crop->generation;
  const auto animal_generation = animal->generation;
  animal->settlement_result = composer_ns::SettlementResult::ProtocolInvalid;
  const auto blocked_settlement = composer.prepare(next_context);
  check(blocked_settlement.token == 0 &&
            blocked_settlement.decision.receipt_acks.empty() &&
            blocked_settlement.decision.telemetry.fail_closed &&
            crop->generation == crop_generation &&
            animal->generation == animal_generation &&
            crop->ledger_version == 0 && animal->ledger_version == 0 &&
            crop->pending_receipts == 1 && animal->pending_receipts == 1 &&
            !composer.fail_stopped(),
        "second-child settlement rejection partially advanced crop state");
  animal->settlement_result =
      composer_ns::SettlementResult::AppliedPhysicalFailure;
  const auto next = composer.prepare(next_context);
  check(next.token != 0 && next.decision.units[1].op == Op::PICKUP &&
            next.decision.receipt_acks.size() == 2 &&
            next.decision.telemetry.receipts_failed == 1 &&
            crop->settled_actions.back() == 1 &&
            animal->settled_actions.back() == 1 &&
            crop->ledger_version == 1 && animal->ledger_version == 1 &&
            crop->pending_receipts == 0 && animal->pending_receipts == 0 &&
            composer.abort(next.token),
        "physical failure was not published, acked, and offered for retry");
  const auto settle_calls =
      std::pair{crop->settles, animal->settles};
  const auto same_step_retry = composer.prepare(next_context);
  check(same_step_retry.decision.receipt_acks.size() == 2 &&
            crop->settles == settle_calls.first &&
            animal->settles == settle_calls.second &&
            composer.abort(same_step_retry.token),
        "same-hand retry settled owned receipts more than once");
}

struct ActualGrant {
  composer_ns::CommitGrant grant;
  fork_ns::MarketCompileResult market;
};

ActualGrant actual_grant(const Simulator &env, int player,
                         const std::array<PlayerAction, 2> &joint,
                         const composer_ns::PreparedRepair &proposal,
                         std::uint64_t generation) {
  ActualGrant out;
  out.grant.submitted_step = env.step_count();
  out.grant.final_units = proposal.units;
  auto final_joint = joint;
  final_joint[player].units = proposal.units;
  for (int actor : proposal.claimed_actors) {
    fork_ns::ActorPrefixAuthority authority{
        actor, generation,
        fork_ns::unit_prefix_manifest_hash(proposal.units, actor),
        fork_ns::post_unit_prefix_state_fingerprint(env, player, final_joint,
                                                    actor)};
    out.grant.actions.push_back({actor, proposal.units[actor], authority});
  }
  std::vector<fork_ns::RequiredPurchase> external;
  for (std::size_t i = 0; i < proposal.required_purchases.size(); ++i) {
    auto request = proposal.required_purchases[i];
    request.debt_id += (1ULL << 61U) + i;
    external.push_back(request);
  }
  fork_ns::ExactMarketCompiler compiler;
  out.market = compiler.compile(env, player, joint[player].market, external);
  for (std::size_t i = 0; i < external.size(); ++i)
    out.grant.purchases.push_back({proposal.required_purchases[i].debt_id,
                                   external[i].debt_id,
                                   out.market.bindings[i]});
  return out;
}

struct ActualReceipts {
  std::vector<fork_ns::ActionReceipt> actions;
  std::vector<fork_ns::PurchaseReceipt> purchases;
};

ActualReceipts actual_execute(Simulator &env, int player,
                              std::array<PlayerAction, 2> joint,
                              const composer_ns::PreparedRepair &proposal,
                              const ActualGrant &selected) {
  const int submitted = env.step_count();
  joint[player].units = proposal.units;
  joint[player].market = selected.market.market;
  env.step(joint);
  ActualReceipts receipts;
  for (const auto &action : selected.grant.actions)
    receipts.actions.push_back(
        {submitted, action.actor, action.final_authority.manifest_generation,
         action.final_authority.prefix_manifest_hash,
         action.final_authority.post_prefix_state_fingerprint, action.emitted});
  const auto &fills = env.last_market_fills()[player];
  for (const auto &binding : selected.market.bindings) {
    const int filled =
        binding.market_slot >= 0 &&
                binding.market_slot < static_cast<int>(fills.size())
            ? std::min(binding.requested,
                       static_cast<int>(fills[binding.market_slot]))
            : 0;
    receipts.purchases.push_back({binding.debt_id, submitted, binding.operation,
                                  binding.item, binding.requested, filled,
                                  binding.market_slot, binding.status});
  }
  return receipts;
}

void actual_transactional_children_joint_commit_and_settle() {
  Simulator env(config(), 1104);
  auto &farm = const_cast<fastkag::Farm &>(env.farms()[0]);
  auto &private_state = const_cast<fastkag::PrivateState &>(env.privates()[0]);
  farm.hands.push_back({0, 0});
  private_state.inventories.push_back({});
  private_state.inventory_order.push_back({});
  farm.tiles[0].kind = fastkag::TileKind::WEED;
  farm.tiles[2 * env.config().board_size + 2].kind = fastkag::TileKind::COOP;

  auto crop = std::make_unique<
      g001::transactional_crop_repair::TransactionalCropRepairOwner>();
  auto *crop_state = crop.get();
  auto animal = std::make_unique<
      g001::transactional_animal_repair::TransactionalAnimalRepairOwner>();

  PlayerAction crop_raw = units(2);
  crop_raw.units[1] = {Op::WATER, Item::NONE, 1};
  crop_raw.market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  auto joint = std::array<PlayerAction, 2>{crop_raw, units(1)};
  const g001::transactional_crop_repair::TypedCropObligation crop_source{
      5001, 0, 1, env.step_count(), {0, 0}, crop_raw.units[1], Item::WHEAT};
  auto crop_proposal =
      crop->prepare({env, 0, env.step_count(), joint[0], joint, {}, {}},
                    std::span{&crop_source, std::size_t{1}});
  auto selected = actual_grant(env, 0, joint, crop_proposal, 1);
  check(crop->validate_commit(crop_proposal.token, selected.grant) &&
            crop->commit(crop_proposal.token, selected.grant),
        "actual crop setup commit failed");
  auto crop_receipts = actual_execute(env, 0, joint, crop_proposal, selected);
  joint = {units(2), units(1)};
  check(crop->settle_owned({env, 0, env.step_count(), joint[0], joint, {}, {}},
                           crop_receipts.actions, crop_receipts.purchases) ==
            composer_ns::SettlementResult::AppliedSuccess,
        "actual crop setup settle failed");

  PlayerAction animal_raw = units(2);
  animal_raw.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  joint = {animal_raw, units(1)};
  const g001::transactional_animal_repair::TypedAnimalObligation animal_source{
      5002,       0,
      0,          env.step_count() + 1,
      {2, 2},     {Op::PLACE, Item::GOOSE, 1},
      Item::GOOSE};
  auto animal_proposal =
      animal->prepare({env, 0, env.step_count(), joint[0], joint, {}, {}},
                      std::span{&animal_source, std::size_t{1}});
  selected = actual_grant(env, 0, joint, animal_proposal, 2);
  check(animal->validate_commit(animal_proposal.token, selected.grant) &&
            animal->commit(animal_proposal.token, selected.grant),
        "actual animal setup commit failed");
  auto animal_receipts =
      actual_execute(env, 0, joint, animal_proposal, selected);
  joint = {units(2), units(1)};
  check(
      animal->settle_owned({env, 0, env.step_count(), joint[0], joint, {}, {}},
                           animal_receipts.actions,
                           animal_receipts.purchases) ==
          composer_ns::SettlementResult::AppliedSuccess,
      "actual animal setup settle failed");

  auto reject_animal_settlement = std::make_shared<bool>(false);
  composer_ns::TransactionalOwnerComposer composer(
      std::move(crop),
      std::make_unique<SettlementGate>(std::move(animal),
                                       reject_animal_settlement));
  auto move_conflict = joint;
  move_conflict[0].units[1] = {Op::EAST, Item::NONE, 1};
  const auto blocked = composer.decide(
      {env, 0, env.step_count(), move_conflict[0], move_conflict, {}, {}});
  check(blocked.units[0].op == Op::PASS && blocked.units[1].op == Op::EAST &&
            blocked.sources[1].source_step == env.step_count() &&
            same_action(blocked.sources[1].source_action,
                        move_conflict[0].units[1]) &&
            blocked.prefix_authority.empty() &&
            blocked.required_purchases.empty() &&
            composer.audit().invalid_proposals == 1 &&
            composer.audit().aborted_proposals == 2 && !composer.fail_stopped(),
        "actual crop/animal repair swallowed an uncertified raw MOVE");

  const int submitted = env.step_count();
  const auto decision =
      composer.decide({env, 0, submitted, joint[0], joint, {}, {}});
  check(decision.units[0].op == Op::PICKUP &&
            decision.units[1].op == Op::PLANT &&
            decision.prefix_authority.size() == 2 && !composer.fail_stopped(),
        "actual crop/animal children did not jointly commit");
  auto executed = joint;
  executed[0].units = decision.units;
  env.step(executed);
  const auto receipts = action_receipts(submitted, decision);
  // Keep the receipt exact but model a physically failed PLANT: the debt must
  // consume the receipt, retain its plot, and offer PLANT again.
  farm.tiles[0].kind = fastkag::TileKind::EMPTY;
  farm.tiles[0].crop = Item::NONE;
  farm.tiles[0].watered_today = false;
  ++private_state.seeds[static_cast<int>(Item::WHEAT)];
  const fork_ns::RepairContext next_context{
      env, 0, env.step_count(), joint[0], joint, {}, receipts};
  const auto crop_before = crop_state->persistent_fingerprint();
  const auto actions_before = crop_state->expected_action_receipts();
  const auto purchases_before = crop_state->expected_purchase_receipts();
  *reject_animal_settlement = true;
  const auto rejected_settlement = composer.prepare(next_context);
  check(rejected_settlement.token == 0 &&
            rejected_settlement.decision.receipt_acks.empty() &&
            crop_state->persistent_fingerprint() == crop_before &&
            crop_state->expected_action_receipts() == actions_before &&
            crop_state->expected_purchase_receipts() == purchases_before &&
            !composer.fail_stopped(),
        "real crop state advanced before second-child settlement validation");
  *reject_animal_settlement = false;
  const auto next = composer.prepare(next_context);
  check(next.decision.receipt_acks.size() == 2 &&
            next.decision.telemetry.receipts_failed == 1 &&
            next.decision.units[1].op == Op::PLANT &&
            crop_state->plot_debts().size() == 1 &&
            crop_state->expected_action_receipts() == 0 &&
            crop_state->expected_purchase_receipts() == 0 &&
            (next.token == 0 || composer.abort(next.token)) &&
            !composer.fail_stopped(),
        "physical crop failure was not acked and retained for retry");
}

void actual_transactional_children_sole_market_slot_abort_cleanly() {
  auto cfg = config();
  cfg.max_market_orders = 1;
  Simulator env(cfg, 1105);
  auto &farm = const_cast<fastkag::Farm &>(env.farms()[0]);
  farm.farmer = {0, 0};
  farm.tiles[0].kind = fastkag::TileKind::WEED;
  farm.tiles[2 * env.config().board_size + 2].kind = fastkag::TileKind::COOP;

  auto crop = std::make_unique<
      g001::transactional_crop_repair::TransactionalCropRepairOwner>();
  auto animal = std::make_unique<
      g001::transactional_animal_repair::TransactionalAnimalRepairOwner>();

  // Open a real crop debt while an unrelated raw order occupies the sole
  // market slot. DIG commits, but the required seed binding is RejectedNoSlot.
  PlayerAction crop_raw = units(1);
  crop_raw.units[0] = {Op::WATER, Item::NONE, 1};
  crop_raw.market.push_back({Op::BUY_PRODUCT, Item::WHEAT, 1});
  auto joint = std::array<PlayerAction, 2>{crop_raw, units(1)};
  const g001::transactional_crop_repair::TypedCropObligation crop_source{
      6001, 0, 0, env.step_count(), {0, 0}, crop_raw.units[0], Item::WHEAT};
  auto crop_proposal =
      crop->prepare({env, 0, env.step_count(), joint[0], joint, {}, {}},
                    std::span{&crop_source, std::size_t{1}});
  auto selected = actual_grant(env, 0, joint, crop_proposal, 11);
  check(crop_proposal.required_purchases.size() == 1,
        "actual crop no-slot setup did not propose one seed debt");
  check(selected.market.bindings.size() == 1,
        "actual crop no-slot setup lost its seed binding");
  check(selected.market.bindings[0].status ==
            fork_ns::PurchaseCompileStatus::RejectedNoSlot,
        "actual crop seed binding was not rejected at the sole occupied slot");
  check(crop->validate_commit(crop_proposal.token, selected.grant),
        "actual crop no-slot grant failed side-effect-free validation");
  check(crop->commit(crop_proposal.token, selected.grant),
        "actual crop no-slot grant failed commit");
  auto receipts = actual_execute(env, 0, joint, crop_proposal, selected);
  joint = {units(1), units(1)};
  check(crop->settle_owned({env, 0, env.step_count(), joint[0], joint, {}, {}},
                           receipts.actions, receipts.purchases) ==
            composer_ns::SettlementResult::AppliedSuccess,
        "actual crop no-slot debt did not settle");

  // Open a real animal debt through a byte-exact raw hard order. Zero cash
  // yields a real BoundExisting receipt with fill zero, preserving the debt.
  farm.money = 0.0;
  PlayerAction animal_raw = units(1);
  animal_raw.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  joint = {animal_raw, units(1)};
  const g001::transactional_animal_repair::TypedAnimalObligation animal_source{
      6002,       0,
      0,          env.step_count() + 1,
      {2, 2},     {Op::PLACE, Item::GOOSE, 1},
      Item::GOOSE};
  auto animal_proposal =
      animal->prepare({env, 0, env.step_count(), joint[0], joint, {}, {}},
                      std::span{&animal_source, std::size_t{1}});
  selected = actual_grant(env, 0, joint, animal_proposal, 12);
  check(animal_proposal.required_purchases.size() == 1 &&
            selected.market.bindings.size() == 1 &&
            selected.market.bindings[0].status ==
                fork_ns::PurchaseCompileStatus::BoundExisting &&
            animal->validate_commit(animal_proposal.token, selected.grant) &&
            animal->commit(animal_proposal.token, selected.grant),
        "actual animal zero-fill debt setup failed");
  receipts = actual_execute(env, 0, joint, animal_proposal, selected);
  joint = {units(1), units(1)};
  check(receipts.purchases.size() == 1 && receipts.purchases[0].filled == 0 &&
            animal->settle_owned(
                {env, 0, env.step_count(), joint[0], joint, {}, {}},
                receipts.actions, receipts.purchases) ==
                composer_ns::SettlementResult::AppliedSuccess,
        "actual animal zero-fill debt did not settle");
  farm.money = 3000.0;

  composer_ns::TransactionalOwnerComposer composer(std::move(crop),
                                                   std::move(animal));
  const auto decision =
      composer.decide({env, 0, env.step_count(), joint[0], joint, {}, {}});
  check(decision.required_purchases.empty() &&
            composer.audit().market_conflicts == 1 &&
            composer.audit().aborted_proposals == 2 &&
            composer.audit().committed_proposals == 0 &&
            !composer.fail_stopped(),
        "actual crop/animal sole-slot conflict did not atomically abort");

  // Execute the unchanged raw step. Both aborted real proposals must be
  // prepare-able again; a second identical conflict proves neither child was
  // left waiting for a purchase receipt that was never submitted.
  advance(env, joint);
  const auto retry =
      composer.decide({env, 0, env.step_count(), joint[0], joint, {}, {}});
  check(retry.required_purchases.empty() &&
            composer.audit().market_conflicts == 2 &&
            composer.audit().aborted_proposals == 4 &&
            composer.audit().committed_proposals == 0 &&
            !composer.fail_stopped(),
        "actual sole-slot abort poisoned a child purchase ledger");
}

void typed_obligations_reach_transactional_children() {
  Simulator env(config(), 1106);
  auto &farm = const_cast<fastkag::Farm &>(env.farms()[0]);
  farm.farmer = {0, 0};
  farm.tiles[0].kind = fastkag::TileKind::WEED;
  auto crop = std::make_unique<
      g001::transactional_crop_repair::TransactionalCropRepairOwner>();
  auto animal = std::make_unique<
      g001::transactional_animal_repair::TransactionalAnimalRepairOwner>();
  composer_ns::TransactionalOwnerComposer composer(std::move(crop),
                                                   std::move(animal));
  auto joint = std::array<PlayerAction, 2>{units(1), units(1)};
  joint[0].units[0] = {Op::WATER, Item::NONE, 1};
  const g001::transactional_crop_repair::TypedCropObligation source{
      7001, 0, 0, env.step_count(), {0, 0}, joint[0].units[0], Item::WHEAT};
  const auto decision = composer.decide_with_intents(
      {env, 0, env.step_count(), joint[0], joint, {}, {}},
      {std::span{&source, std::size_t{1}}, {}});
  check(decision.units.size() == 1 && decision.units[0].op == Op::DIG &&
            decision.required_purchases.size() == 1 &&
            decision.required_purchases[0].operation == Op::BUY_SEED &&
            decision.required_purchases[0].item == Item::WHEAT,
        "typed crop obligation did not reach the transactional owner");
}

} // namespace

int main() {
  try {
    same_actor_conflict_is_raw_and_permanent();
    different_actors_coexist_and_receipts_are_owned();
    seed_animal_last_market_slot_conflict_fails_closed();
    player1_authority_contains_player0_prefix();
    terminal_purchase_debt_is_not_silently_acked();
    full_raw_pass_through_is_byte_exact();
    sealed_default_owners_are_linked_but_not_native_registered();
    transactional_conflict_aborts_without_poisoning_next_prepare();
    transactional_different_actors_share_one_final_manifest();
    transactional_second_child_rejection_is_prevalidated_atomically();
    staged_native_reject_abort_retry_and_owned_receipts();
    actual_transactional_children_joint_commit_and_settle();
    actual_transactional_children_sole_market_slot_abort_cleanly();
    typed_obligations_reach_transactional_children();
    std::cout << "repair owner composer: 14 deterministic groups passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "repair owner composer test failure: " << error.what() << '\n';
    return 1;
  }
}
