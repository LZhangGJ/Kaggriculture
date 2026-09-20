#include "day_scheduler_transactional_composer_bridge.hpp"
#include "transactional_animal_repair_owner.hpp"
#include "transactional_crop_repair_owner.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bridge = g001::day_scheduler_transactional_bridge;
namespace composer = g001::repair_owner_composer;
namespace crop_ns = g001::transactional_crop_repair;
namespace animal_ns = g001::transactional_animal_repair;
namespace day = g001::obligation_day;
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

bool same(const Action &left, const Action &right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

fastkag::Config config() {
  fastkag::Config value;
  value.episode_steps = 72;
  value.weed_spawn_chance = 0.0;
  return value;
}

PlayerAction units(int count) {
  PlayerAction output;
  output.units.resize(static_cast<std::size_t>(count));
  return output;
}

struct ProbeState {
  int prepares{};
  int commits{};
  int aborts{};
  int validations_until_reject{};
  std::set<std::uint64_t> cached;
};

class ProbeOwner final : public composer::TransactionalRepairOwner {
public:
  ProbeOwner(std::shared_ptr<ProbeState> state, int step = -1,
             int actor = -1, Action action = {},
             fork_ns::SourceBinding source = {}, int active_from = 1,
             int purchase_step = -1)
      : state_(std::move(state)), step_(step), actor_(actor), action_(action),
        source_(source), active_from_(active_from),
        purchase_step_(purchase_step) {}

  std::string name() const override { return "bridge_gate_probe"; }
  composer::SettlementResult settle_owned(
      const fork_ns::RepairContext &,
      std::span<const fork_ns::ActionReceipt> actions,
      std::span<const fork_ns::PurchaseReceipt> purchases) override {
    static_cast<void>(actions);
    static_cast<void>(purchases);
    return composer::SettlementResult::AppliedSuccess;
  }
  composer::SettlementResult validate_settle(
      const fork_ns::RepairContext &,
      std::span<const fork_ns::ActionReceipt> actions,
      std::span<const fork_ns::PurchaseReceipt> purchases) const override {
    static_cast<void>(actions);
    static_cast<void>(purchases);
    return composer::SettlementResult::AppliedSuccess;
  }
  composer::PreparedRepair
  prepare(const fork_ns::RepairContext &context) const override {
    composer::PreparedRepair out;
    out.token = 0x70000000ULL + static_cast<std::uint64_t>(++state_->prepares);
    state_->cached.insert(out.token);
    out.units = context.raw_g001.units;
    for (std::size_t index = 0; index < out.units.size(); ++index)
      out.sources.push_back({static_cast<int>(index), context.step,
                             out.units[index]});
    if (context.step == step_ && state_->prepares >= active_from_) {
      out.units[static_cast<std::size_t>(actor_)] = action_;
      out.sources[static_cast<std::size_t>(actor_)] = source_;
      out.claimed_actors.push_back(actor_);
    }
    if (context.step == purchase_step_)
      out.required_purchases.push_back(
          {0x71000000ULL, Op::BUY_SEED, Item::WHEAT, 1, context.step});
    return out;
  }
  bool validate_commit(std::uint64_t token,
                       const composer::CommitGrant &) const override {
    if (!state_->cached.contains(token))
      return false;
    if (state_->validations_until_reject > 0 &&
        --state_->validations_until_reject == 0)
      return false;
    return true;
  }
  bool commit(std::uint64_t token,
              const composer::CommitGrant &) override {
    if (state_->cached.erase(token) == 0)
      return false;
    ++state_->commits;
    return true;
  }
  void abort(std::uint64_t token) noexcept override {
    if (state_->cached.erase(token) != 0)
      ++state_->aborts;
  }

private:
  std::shared_ptr<ProbeState> state_;
  int step_{};
  int actor_{};
  Action action_{};
  fork_ns::SourceBinding source_{};
  int active_from_{};
  int purchase_step_{};
};

struct Grant {
  composer::CommitGrant commit;
  fork_ns::MarketCompileResult market;
};

Grant grant(const Simulator &env, int player,
            const std::array<PlayerAction, 2> &joint,
            const composer::PreparedRepair &proposal,
            std::uint64_t generation) {
  Grant output;
  output.commit.submitted_step = env.step_count();
  output.commit.final_units = proposal.units;
  auto final_joint = joint;
  final_joint[player].units = proposal.units;
  for (int actor : proposal.claimed_actors) {
    fork_ns::ActorPrefixAuthority authority{
        actor, generation,
        fork_ns::unit_prefix_manifest_hash(proposal.units, actor),
        fork_ns::post_unit_prefix_state_fingerprint(env, player, final_joint,
                                                    actor)};
    output.commit.actions.push_back(
        {actor, proposal.units[static_cast<std::size_t>(actor)], authority});
  }
  std::vector<fork_ns::RequiredPurchase> external;
  for (std::size_t index = 0; index < proposal.required_purchases.size();
       ++index) {
    auto request = proposal.required_purchases[index];
    request.debt_id += (1ULL << 60U) + index;
    external.push_back(request);
  }
  fork_ns::ExactMarketCompiler compiler;
  output.market = compiler.compile(env, player, joint[player].market, external);
  for (std::size_t index = 0; index < external.size(); ++index)
    output.commit.purchases.push_back(
        {proposal.required_purchases[index].debt_id, external[index].debt_id,
         output.market.bindings[index]});
  return output;
}

struct Receipts {
  std::vector<fork_ns::ActionReceipt> actions;
  std::vector<fork_ns::PurchaseReceipt> purchases;
};

Receipts execute(Simulator &env, int player, std::array<PlayerAction, 2> joint,
                 const composer::PreparedRepair &proposal,
                 const Grant &selected) {
  const int submitted = env.step_count();
  joint[player].units = proposal.units;
  joint[player].market = selected.market.market;
  env.step(joint);
  Receipts output;
  for (const auto &action : selected.commit.actions)
    output.actions.push_back(
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
    output.purchases.push_back({binding.debt_id, submitted, binding.operation,
                                binding.item, binding.requested, filled,
                                binding.market_slot, binding.status});
  }
  return output;
}

std::vector<fork_ns::ActionReceipt>
action_receipts(int submitted, const fork_ns::RepairDecision &decision) {
  std::vector<fork_ns::ActionReceipt> output;
  for (const auto &authority : decision.prefix_authority)
    output.push_back(
        {submitted, authority.actor, authority.manifest_generation,
         authority.prefix_manifest_hash,
         authority.post_prefix_state_fingerprint,
         decision.units[static_cast<std::size_t>(authority.actor)]});
  return output;
}

struct Fixture {
  Simulator env;
  std::unique_ptr<crop_ns::TransactionalCropRepairOwner> crop;
  std::unique_ptr<animal_ns::TransactionalAnimalRepairOwner> animal;
};

Fixture prepared_day_one() {
  Fixture fixture{
      Simulator(config(), 73001),
      std::make_unique<crop_ns::TransactionalCropRepairOwner>(),
      std::make_unique<animal_ns::TransactionalAnimalRepairOwner>()};
  auto &farm = const_cast<fastkag::Farm &>(fixture.env.farms()[0]);
  auto &private_state =
      const_cast<fastkag::PrivateState &>(fixture.env.privates()[0]);
  farm.hands.push_back({0, 0});
  private_state.inventories.push_back({});
  private_state.inventory_order.push_back({});
  farm.tiles[0].kind = fastkag::TileKind::WEED;
  farm.tiles[2 * fixture.env.config().board_size + 2].kind =
      fastkag::TileKind::COOP;

  PlayerAction crop_raw = units(2);
  crop_raw.units[1] = {Op::WATER, Item::NONE, 1};
  crop_raw.market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  auto joint = std::array<PlayerAction, 2>{crop_raw, units(1)};
  const crop_ns::TypedCropObligation crop_source{
      8001,       0, 1, fixture.env.step_count(), {0, 0}, crop_raw.units[1],
      Item::WHEAT};
  const auto crop_proposal = fixture.crop->prepare(
      {fixture.env, 0, fixture.env.step_count(), joint[0], joint, {}, {}},
      std::span{&crop_source, std::size_t{1}});
  const auto crop_grant = grant(fixture.env, 0, joint, crop_proposal, 1);
  check(fixture.crop->validate_commit(crop_proposal.token, crop_grant.commit) &&
            fixture.crop->commit(crop_proposal.token, crop_grant.commit),
        "crop day-zero setup commit failed");
  const auto crop_receipts =
      execute(fixture.env, 0, joint, crop_proposal, crop_grant);
  joint = {units(2), units(1)};
  check(fixture.crop->settle_owned(
            {fixture.env, 0, fixture.env.step_count(), joint[0], joint, {}, {}},
            crop_receipts.actions, crop_receipts.purchases) ==
            composer::SettlementResult::AppliedSuccess,
        "crop day-zero setup settle failed");

  PlayerAction animal_raw = units(2);
  animal_raw.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
  joint = {animal_raw, units(1)};
  const animal_ns::TypedAnimalObligation animal_source{
      8002, 0, 0, 30, {2, 2}, {Op::PLACE, Item::GOOSE, 1}, Item::GOOSE};
  const auto animal_proposal = fixture.animal->prepare(
      {fixture.env, 0, fixture.env.step_count(), joint[0], joint, {}, {}},
      std::span{&animal_source, std::size_t{1}});
  const auto animal_grant = grant(fixture.env, 0, joint, animal_proposal, 2);
  check(fixture.animal->validate_commit(animal_proposal.token,
                                        animal_grant.commit) &&
            fixture.animal->commit(animal_proposal.token, animal_grant.commit),
        "animal day-zero setup commit failed");
  const auto animal_receipts =
      execute(fixture.env, 0, joint, animal_proposal, animal_grant);
  joint = {units(2), units(1)};
  check(fixture.animal->settle_owned(
            {fixture.env, 0, fixture.env.step_count(), joint[0], joint, {}, {}},
            animal_receipts.actions, animal_receipts.purchases) ==
            composer::SettlementResult::AppliedSuccess,
        "animal day-zero setup settle failed");

  while (fixture.env.step_count() < 24)
    fixture.env.step(joint);
  auto &day_one_farm = const_cast<fastkag::Farm &>(fixture.env.farms()[0]);
  auto &day_one_private =
      const_cast<fastkag::PrivateState &>(fixture.env.privates()[0]);
  day_one_farm.hands.push_back({0, 0});
  day_one_private.inventories.push_back({});
  day_one_private.inventory_order.push_back({});
  check(fixture.env.hour() == 0, "fixture did not reach day-one boundary");
  return fixture;
}

day::DayPlanRequest day_request(const Simulator &env) {
  const auto &farm = env.farms()[0];
  day::ProductionObligation pickup{
      8101,        0,  farm.farmer, day::GoalKind::Pickup,
      Item::GOOSE, 1,  {},          {Item::GOOSE, 0, 1, 0},
      24,          24, 200,         true,
      true};
  day::ProductionObligation crop{
      8102,        1,  {0, 0}, day::GoalKind::CropReady,
      Item::WHEAT, 1,  {},     {Item::WHEAT, 1, 0, 0},
      24,          47, 100,    true,
      true};
  return {&env, 0, 99001, {{1, 24, {Op::EAST, Item::NONE, 1}}}, {pickup, crop}};
}

std::unique_ptr<crop_ns::TransactionalCropRepairOwner> fresh_crop() {
  return std::make_unique<crop_ns::TransactionalCropRepairOwner>();
}

std::unique_ptr<animal_ns::TransactionalAnimalRepairOwner> fresh_animal() {
  return std::make_unique<animal_ns::TransactionalAnimalRepairOwner>();
}

void real_children_follow_signed_day_and_close_move_once() {
  auto fixture = prepared_day_one();
  auto request = day_request(fixture.env);
  const auto plan = day::plan_day(request);
  check(plan.planned(), "real bridge day was rejected by planner code " +
                            std::to_string(static_cast<int>(plan.reject)));
  check(plan.debts.empty(), "real bridge day retained obligation debt");
  check(day::verify_day_schedule(request, *plan.certificate).valid,
        "real bridge day certificate did not verify");
  check(plan.manifest[0][0].op == Op::PICKUP &&
            plan.manifest[1][0].op == Op::PLANT &&
            plan.manifest[1][1].op == Op::WATER &&
            plan.manifest[1][2].op == Op::EAST &&
            plan.sources[1][2].source_step == 24,
        "day scheduler did not produce expected animal/crop/MOVE chain");

  bridge::DaySchedulerTransactionalComposerBridge owner(
      std::move(fixture.crop), std::move(fixture.animal), request,
      *plan.certificate);
  check(owner.certificate_valid(), "valid signed bridge was rejected");

  std::vector<fork_ns::ActionReceipt> previous;
  for (int step = 24; step <= 47; ++step) {
    std::array<PlayerAction, 2> joint{units(2), units(1)};
    if (step == 24)
      joint[0].units[1] = {Op::EAST, Item::NONE, 1};
    const auto decision =
        owner.decide({fixture.env, 0, step, joint[0], joint, {}, previous});
    check(decision.units.size() == 2 &&
              same(decision.units[0], plan.manifest[0][step - 24]) &&
              same(decision.units[1], plan.manifest[1][step - 24]),
          "real child decision diverged from signed day manifest");
    if (step == 24)
      check(decision.units[0].op == Op::PICKUP &&
                decision.units[1].op == Op::PLANT,
            "crop+animal did not jointly displace signed MOVE");
    if (step == 26)
      check(decision.units[1].op == Op::EAST &&
                decision.sources[1].source_step == 24,
            "signed delayed MOVE did not replay with original source");
    const int submitted = fixture.env.step_count();
    joint[0].units = decision.units;
    fixture.env.step(joint);
    previous = action_receipts(submitted, decision);
  }
  check(!owner.fail_stopped() && owner.audit().move_tokens_submitted == 1 &&
            owner.audit().incomplete_day_closures == 0,
        "signed day did not close every MOVE exactly once");
}

void tamper_omission_reorder_and_cross_day_fail_closed() {
  auto fixture = prepared_day_one();
  auto request = day_request(fixture.env);
  request.moves.push_back({1, 28, {Op::WEST, Item::NONE, 1}});
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.certificate->move_replays.size() == 2,
        "two-MOVE adversarial certificate setup failed");

  auto rejected = [&](day::DayScheduleCertificate certificate,
                      const std::string &label) {
    certificate.content_hash = day::day_schedule_certificate_hash(certificate);
    bridge::DaySchedulerTransactionalComposerBridge owner(
        fresh_crop(), fresh_animal(), request, certificate);
    check(!owner.certificate_valid() &&
              owner.audit().certificate_rejections == 1,
          label + " certificate was accepted");
  };

  auto tampered = *plan.certificate;
  tampered.slots[0].actions[1] = {};
  tampered.slots[0].sources[1] = {1, -1, {}};
  tampered.slots[0].obligation_ids[1] = 0;
  rejected(tampered, "tampered action");

  auto omitted = *plan.certificate;
  omitted.move_replays.erase(omitted.move_replays.begin());
  rejected(omitted, "omitted MOVE");

  auto reordered = *plan.certificate;
  std::swap(reordered.move_replays[0], reordered.move_replays[1]);
  rejected(reordered, "reordered MOVE");

  auto cross_day = *plan.certificate;
  cross_day.move_replays[0].emitted_step = 48;
  rejected(cross_day, "cross-day MOVE");
}

void must_finish_debt_rejects_otherwise_valid_certificate() {
  Simulator env(config(), 73002);
  std::array<PlayerAction, 2> joint{units(1), units(1)};
  while (env.step_count() < 24)
    env.step(joint);
  auto &farm = const_cast<fastkag::Farm &>(env.farms()[0]);
  auto &private_state = const_cast<fastkag::PrivateState &>(env.privates()[0]);
  const auto position = farm.farmer;
  farm.tiles[position.y * env.config().board_size + position.x].kind =
      fastkag::TileKind::WEED;
  private_state.seeds[static_cast<int>(Item::WHEAT)] = 0;
  day::ProductionObligation crop{
      8201,        0,  position, day::GoalKind::CropReady,
      Item::WHEAT, 1,  {},       {Item::WHEAT, 1, 0, 0},
      24,          47, 100,      true,
      true};
  crop.source_step = 24;
  crop.policy_deferred = true;
  day::DayPlanRequest request{&env, 0, 99002, {}, {crop}};
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.debts.size() == 1 &&
            day::verify_day_schedule(request, *plan.certificate).valid,
        "must-finish debt certificate setup was not verifier-valid");
  bridge::DaySchedulerTransactionalComposerBridge owner(
      fresh_crop(), fresh_animal(), request, *plan.certificate);
  check(!owner.certificate_valid() &&
            owner.audit().must_finish_rejections == 1 &&
            owner.audit().debt_authorization_rejections == 1,
        "must-finish debt authorized MOVE scheduling");

  bridge::DebtAuthorization authorization{plan.certificate->content_hash,
                                          crop.id,
                                          crop.source_step,
                                          crop.goal,
                                          crop.actor,
                                          crop.tile,
                                          crop.item,
                                          3,
                                          0xabcddcbaULL,
                                          77,
                                          0};
  authorization.content_hash = bridge::debt_authorization_hash(authorization);
  bridge::DaySchedulerTransactionalComposerBridge explicitly_authorized(
      fresh_crop(), fresh_animal(), request, *plan.certificate,
      std::span{&authorization, std::size_t{1}});
  check(explicitly_authorized.certificate_valid() &&
            explicitly_authorized.audit().authorized_debts == 1,
        "exact upstream debt authorization was not admitted");

  auto tampered = authorization;
  tampered.damage_policy_hash ^= 1ULL;
  bridge::DaySchedulerTransactionalComposerBridge rejected_tamper(
      fresh_crop(), fresh_animal(), request, *plan.certificate,
      std::span{&tampered, std::size_t{1}});
  check(!rejected_tamper.certificate_valid() &&
            rejected_tamper.audit().debt_authorization_rejections == 1,
        "tampered damage-policy authorization was admitted");

  auto rebound = authorization;
  rebound.goal = day::GoalKind::Feed;
  rebound.content_hash = bridge::debt_authorization_hash(rebound);
  bridge::DaySchedulerTransactionalComposerBridge rejected_rebind(
      fresh_crop(), fresh_animal(), request, *plan.certificate,
      std::span{&rebound, std::size_t{1}});
  check(!rejected_rebind.certificate_valid() &&
            rejected_rebind.audit().debt_authorization_rejections == 1,
        "rehash rebound an authorization to another goal");
}

Simulator day_one(std::uint64_t seed, bool with_hand) {
  Simulator env(config(), seed);
  std::array<PlayerAction, 2> joint{units(1), units(1)};
  while (env.step_count() < 24)
    env.step(joint);
  if (with_hand) {
    auto &farm = const_cast<fastkag::Farm &>(env.farms()[0]);
    auto &private_state = const_cast<fastkag::PrivateState &>(env.privates()[0]);
    farm.hands.push_back({0, 0});
    private_state.inventories.push_back({});
    private_state.inventory_order.push_back({});
  }
  return env;
}

day::DayPlanRequest pasture_request(const Simulator &env, int step,
                                    bool with_move) {
  day::ProductionObligation pasture{
      8301,       0,  env.farms()[0].farmer, day::GoalKind::BuildPasture,
      Item::NONE, 1,  {},                       {},
      step,       step, 100,                    true,
      true};
  pasture.source_step = step;
  day::DayPlanRequest request{&env, 0, 99003, {}, {pasture}};
  if (with_move)
    request.moves.push_back({1, 24, {Op::EAST, Item::NONE, 1}});
  return request;
}

std::unique_ptr<ProbeOwner>
probe_for_slot(std::shared_ptr<ProbeState> state,
               const day::DayScheduleCertificate &certificate, int step,
               int active_from = 1, int purchase_step = -1) {
  const auto &slot = certificate.slots[static_cast<std::size_t>(step - 24)];
  return std::make_unique<ProbeOwner>(
      std::move(state), step, 0, slot.actions[0], slot.sources[0], active_from,
      purchase_step);
}

void gate_rejection_aborts_before_publish() {
  auto env = day_one(73003, true);
  auto request = pasture_request(env, 47, true);
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.certificate->move_replays.size() == 1 &&
            plan.certificate->slots.back().actions[0].op ==
                Op::BUILD_PASTURE,
        "gate-abort fixture did not produce its signed terminal action");
  const auto active = std::make_shared<ProbeState>();
  const auto passive = std::make_shared<ProbeState>();
  bridge::DaySchedulerTransactionalComposerBridge owner(
      probe_for_slot(active, *plan.certificate, 47),
      std::make_unique<ProbeOwner>(passive), request, *plan.certificate);

  for (int step = 24; step < 47; ++step) {
    std::array<PlayerAction, 2> joint{units(2), units(1)};
    joint[0].units = plan.certificate->slots[step - 24].actions;
    env.step(joint);
  }
  std::array<PlayerAction, 2> raw{units(2), units(1)};
  const auto rejected = owner.decide({env, 0, 47, raw[0], raw, {}, {}});
  check(rejected.units[0].op == Op::PASS && active->commits == 0 &&
            active->aborts == 1 && !owner.fail_stopped() &&
            owner.audit().incomplete_day_closures == 1 &&
            owner.audit().move_tokens_submitted == 0,
        "post-prepare day-closure gate published child state");
}

void gate_rejection_allows_same_hand_retry() {
  auto env = day_one(73004, false);
  auto request = pasture_request(env, 24, false);
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.certificate->slots.front().actions[0].op ==
                              Op::BUILD_PASTURE,
        "same-hand retry fixture did not produce its signed action");
  const auto active = std::make_shared<ProbeState>();
  const auto passive = std::make_shared<ProbeState>();
  bridge::DaySchedulerTransactionalComposerBridge owner(
      probe_for_slot(active, *plan.certificate, 24, 2),
      std::make_unique<ProbeOwner>(passive), request, *plan.certificate);
  std::array<PlayerAction, 2> raw{units(1), units(1)};
  const auto first = owner.decide({env, 0, 24, raw[0], raw, {}, {}});
  check(first.units[0].op == Op::PASS && active->commits == 0 &&
            !owner.fail_stopped(),
        "manifest gate rejection published or fail-stopped the owner");
  const auto second = owner.decide({env, 0, 24, raw[0], raw, {}, {}});
  check(second.units[0].op == Op::BUILD_PASTURE && active->commits == 1 &&
            owner.audit().manifest_rejections == 1 && !owner.fail_stopped(),
        "corrected same-hand proposal did not finalize exactly once");
}

void finalize_failure_preserves_already_settled_receipts() {
  auto env = day_one(73005, false);
  auto request = pasture_request(env, 24, false);
  const auto plan = day::plan_day(request);
  check(plan.planned(), "finalize-failure fixture did not plan");
  const auto active = std::make_shared<ProbeState>();
  const auto passive = std::make_shared<ProbeState>();
  bridge::DaySchedulerTransactionalComposerBridge owner(
      probe_for_slot(active, *plan.certificate, 24, 1, 25),
      std::make_unique<ProbeOwner>(passive), request, *plan.certificate);

  std::array<PlayerAction, 2> raw{units(1), units(1)};
  const int submitted = env.step_count();
  const auto first = owner.decide({env, 0, submitted, raw[0], raw, {}, {}});
  check(first.units[0].op == Op::BUILD_PASTURE && active->commits == 1,
        "finalize-failure setup did not commit its first action");
  raw[0].units = first.units;
  env.step(raw);
  const auto receipts = action_receipts(submitted, first);

  active->validations_until_reject = 2; // prepare preflight passes, finalize fails.
  raw = {units(1), units(1)};
  const auto failed = owner.decide(
      {env, 0, env.step_count(), raw[0], raw, {}, receipts});
  check(failed.units[0].op == Op::PASS && failed.receipt_acks.size() == 1 &&
            failed.receipt_acks[0].manifest_generation ==
                receipts[0].manifest_generation &&
            active->commits == 1 && owner.fail_stopped(),
        "finalize failure dropped the receipt settled during prepare");
}

void external_abort_keeps_crop_ledger_and_allows_retry() {
  auto env = day_one(73006, false);
  auto &farm = const_cast<fastkag::Farm &>(env.farms()[0]);
  auto &private_state = const_cast<fastkag::PrivateState &>(env.privates()[0]);
  const auto tile = farm.farmer;
  farm.tiles[static_cast<std::size_t>(tile.y * env.config().board_size +
                                      tile.x)] = {};
  farm.tiles[static_cast<std::size_t>(tile.y * env.config().board_size +
                                      tile.x)]
      .kind = fastkag::TileKind::WEED;
  private_state.seeds[static_cast<std::size_t>(Item::WHEAT)] = 1;

  day::ProductionObligation crop_obligation{
      8401,        0,  tile, day::GoalKind::CropReady,
      Item::WHEAT, 1,  {},   {Item::WHEAT, 1, 0, 0},
      24,          47, 100,  true,
      true};
  crop_obligation.source_step = 24;
  day::DayPlanRequest request{&env, 0, 99004, {}, {crop_obligation}};
  const auto plan = day::plan_day(request);
  check(plan.planned() && plan.debts.empty(),
        "external-abort fixture did not plan the crop chain");

  auto crop = fresh_crop();
  auto *crop_probe = crop.get();
  bridge::DaySchedulerTransactionalComposerBridge owner(
      std::move(crop), fresh_animal(), request, *plan.certificate);
  std::array<PlayerAction, 2> raw{units(1), units(1)};
  raw[0].units[0] = {Op::WATER, Item::NONE, 1};
  const crop_ns::TypedCropObligation typed{
      crop_obligation.id, 0, 0, crop_obligation.source_step, tile,
      {Op::WATER, Item::NONE, 1}, Item::WHEAT};
  const fork_ns::RepairContext context{env, 0, 24, raw[0], raw, {}, {}};
  const auto fingerprint = crop_probe->persistent_fingerprint();
  const auto expected_actions = crop_probe->expected_action_receipts();
  const auto expected_purchases = crop_probe->expected_purchase_receipts();
  const auto commits = crop_probe->audit().commits;

  const auto rejected = owner.prepare(
      context, composer::TypedRepairInput{
                   std::span{&typed, std::size_t{1}}, {}});
  check(rejected.token != 0 && rejected.decision.units[0].op == Op::DIG,
        "external-abort fixture did not stage DIG");
  check(owner.abort(rejected.token), "external native rejection did not abort");
  check(crop_probe->persistent_fingerprint() == fingerprint &&
            crop_probe->expected_action_receipts() == expected_actions &&
            crop_probe->expected_purchase_receipts() == expected_purchases &&
            crop_probe->audit().commits == commits &&
            crop_probe->cached_proposals() == 0,
        "external native rejection leaked crop ledger state");

  const auto retry = owner.prepare(
      context, composer::TypedRepairInput{
                   std::span{&typed, std::size_t{1}}, {}});
  auto selected = raw[0];
  selected.units = retry.decision.units;
  selected.market = fork_ns::ExactMarketCompiler()
                        .compile(env, 0, raw[0].market,
                                 retry.decision.required_purchases)
                        .market;
  check(retry.token != 0 && owner.finalize(retry.token, selected) &&
            crop_probe->audit().commits == commits + 1 &&
            crop_probe->expected_action_receipts() == expected_actions + 1,
        "same-hand retry did not commit exactly once after native rejection");
}

} // namespace

int main() {
  try {
    real_children_follow_signed_day_and_close_move_once();
    tamper_omission_reorder_and_cross_day_fail_closed();
    must_finish_debt_rejects_otherwise_valid_certificate();
    gate_rejection_aborts_before_publish();
    gate_rejection_allows_same_hand_retry();
    finalize_failure_preserves_already_settled_receipts();
    external_abort_keeps_crop_ledger_and_allows_retry();
    std::cout
        << "day scheduler transactional composer bridge: 7 groups passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "day scheduler transactional composer bridge: " << error.what()
              << '\n';
    return 1;
  }
}
