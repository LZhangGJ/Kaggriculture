#include "transactional_owner_composer.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

namespace g001::repair_owner_composer {
namespace {
using fastkag::Action;
using fastkag::Op;
using repair_fork::PurchaseCompileStatus;

bool same(const Action &a, const Action &b) {
  return a.op == b.op && a.item == b.item && a.quantity == b.quantity;
}
bool same(const fastkag::PlayerAction &a, const fastkag::PlayerAction &b) {
  return a.units.size() == b.units.size() &&
         a.market.size() == b.market.size() &&
         std::equal(a.units.begin(), a.units.end(), b.units.begin(),
                    [](const auto &left, const auto &right) {
                      return same(left, right);
                    }) &&
         std::equal(a.market.begin(), a.market.end(), b.market.begin(),
                    [](const auto &left, const auto &right) {
                      return same(left, right);
                    });
}
bool move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST || op == Op::WEST;
}
bool accepted(PurchaseCompileStatus status) {
  return status == PurchaseCompileStatus::BoundExisting ||
         status == PurchaseCompileStatus::Appended;
}
repair_fork::RepairDecision raw(const repair_fork::RepairContext &c) {
  repair_fork::RepairDecision out;
  out.units = c.raw_g001.units;
  for (std::size_t actor = 0; actor < out.units.size(); ++actor)
    out.sources.push_back({static_cast<int>(actor), c.step, out.units[actor]});
  return out;
}
bool same_action_receipt(const repair_fork::ActionReceipt &a,
                         const repair_fork::ActionReceipt &b) {
  return a.submitted_step == b.submitted_step && a.actor == b.actor &&
         a.manifest_generation == b.manifest_generation &&
         a.prefix_manifest_hash == b.prefix_manifest_hash &&
         a.post_prefix_state_fingerprint == b.post_prefix_state_fingerprint &&
         same(a.emitted, b.emitted);
}
bool same_purchase_shape(const repair_fork::PurchaseReceipt &a,
                         const repair_fork::PurchaseReceipt &b) {
  return a.debt_id == b.debt_id && a.submitted_step == b.submitted_step &&
         a.operation == b.operation && a.item == b.item &&
         a.requested == b.requested && a.market_slot == b.market_slot &&
         a.compile_status == b.compile_status;
}
} // namespace

struct TransactionalOwnerComposer::Impl {
  struct PendingAction {
    int child{};
    repair_fork::ActionReceipt shape;
  };
  struct PendingPurchase {
    int child{};
    repair_fork::PurchaseReceipt shape;
  };
  struct Stage {
    std::uint64_t token{};
    std::array<PreparedRepair, 2> proposals;
    std::array<CommitGrant, 2> grants;
    std::array<bool, 2> active{};
    fastkag::PlayerAction expected;
    std::map<int, std::uint64_t> generations;
    std::map<std::pair<int, std::uint64_t>, std::uint64_t> debts;
    std::uint64_t next_debt{};
    std::vector<PendingAction> pending_actions;
    std::vector<PendingPurchase> pending_purchases;
  };
  std::array<std::unique_ptr<TransactionalRepairOwner>, 2> owners;
  std::array<std::vector<repair_fork::ActionReceipt>, 2> routed_actions;
  std::array<std::vector<repair_fork::PurchaseReceipt>, 2> routed_purchases;
  std::vector<PendingAction> pending_actions;
  std::vector<PendingPurchase> pending_purchases;
  std::map<int, std::uint64_t> generations;
  std::map<std::pair<int, std::uint64_t>, std::uint64_t> debts;
  std::uint64_t next_debt{1ULL << 62U};
  std::uint64_t next_transaction{1};
  std::optional<Stage> stage;
  int settled_player{-1};
  int settled_step{-1};
  std::vector<repair_fork::ActionReceipt> settled_action_acks;
  std::vector<repair_fork::PurchaseReceipt> settled_purchase_acks;
  TransactionalComposerAudit audit;
  bool stopped{};

  Impl(std::unique_ptr<TransactionalRepairOwner> crop,
       std::unique_ptr<TransactionalRepairOwner> animal)
      : owners{std::move(crop), std::move(animal)} {
    if (!owners[0] || !owners[1])
      throw std::invalid_argument("transactional composer requires two owners");
  }
  static std::uint64_t external_debt(
      int child, std::uint64_t local,
      std::map<std::pair<int, std::uint64_t>, std::uint64_t> &debt_map,
      std::uint64_t &next) {
    const auto key = std::make_pair(child, local);
    if (const auto found = debt_map.find(key); found != debt_map.end())
      return found->second;
    if (local == 0 || next == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("invalid/exhausted composer debt namespace");
    const auto result = next++;
    debt_map.emplace(key, result);
    return result;
  }
};

TransactionalOwnerComposer::TransactionalOwnerComposer(
    std::unique_ptr<TransactionalRepairOwner> crop,
    std::unique_ptr<TransactionalRepairOwner> animal)
    : impl_(std::make_unique<Impl>(std::move(crop), std::move(animal))) {}
TransactionalOwnerComposer::~TransactionalOwnerComposer() = default;
TransactionalOwnerComposer::TransactionalOwnerComposer(
    TransactionalOwnerComposer &&) noexcept = default;
TransactionalOwnerComposer &TransactionalOwnerComposer::operator=(
    TransactionalOwnerComposer &&) noexcept = default;
std::string TransactionalOwnerComposer::name() const {
  return "transactional_owner_composer_v1";
}

repair_fork::RepairDecision
TransactionalOwnerComposer::decide(const repair_fork::RepairContext &context) {
  return decide_immediately(context, nullptr, {});
}

repair_fork::RepairDecision TransactionalOwnerComposer::decide_with_intents(
    const repair_fork::RepairContext &context, TypedRepairInput input) {
  return decide_immediately(context, nullptr, input);
}

StagedRepair TransactionalOwnerComposer::prepare(
    const repair_fork::RepairContext &context, TypedRepairInput input) {
  return prepare_impl(context, nullptr, input);
}

StagedRepair TransactionalOwnerComposer::prepare_certified(
    const repair_fork::RepairContext &context,
    const ScheduledStepAuthority &authority, TypedRepairInput input) {
  return prepare_impl(context, &authority, input);
}

repair_fork::RepairDecision TransactionalOwnerComposer::decide_certified(
    const repair_fork::RepairContext &context,
    const ScheduledStepAuthority &authority) {
  return decide_immediately(context, &authority, {});
}

repair_fork::RepairDecision TransactionalOwnerComposer::decide_certified(
    const repair_fork::RepairContext &context,
    const ScheduledStepAuthority &authority, TypedRepairInput input) {
  return decide_immediately(context, &authority, input);
}

repair_fork::RepairDecision TransactionalOwnerComposer::decide_immediately(
    const repair_fork::RepairContext &context,
    const ScheduledStepAuthority *authority, TypedRepairInput input) {
  auto prepared = prepare_impl(context, authority, input);
  if (prepared.token == 0)
    return prepared.decision;
  fastkag::PlayerAction selected = context.raw_g001;
  selected.units = prepared.decision.units;
  selected.market = repair_fork::ExactMarketCompiler()
                        .compile(context.phase_start, context.player,
                                 context.raw_g001.market,
                                 prepared.decision.required_purchases)
                        .market;
  if (finalize(prepared.token, selected))
    return prepared.decision;
  auto fallback = raw(context);
  fallback.receipt_acks = prepared.decision.receipt_acks;
  fallback.purchase_receipt_acks =
      prepared.decision.purchase_receipt_acks;
  fallback.telemetry.fail_closed = 1;
  return fallback;
}

StagedRepair TransactionalOwnerComposer::prepare_impl(
    const repair_fork::RepairContext &context,
    const ScheduledStepAuthority *authority, TypedRepairInput input) {
  if (context.player < 0 || context.player > 1 || context.step < 0 ||
      context.step != context.phase_start.step_count())
    throw std::invalid_argument("transactional composer context invalid");
  ++impl_->audit.decisions;
  StagedRepair result;
  auto &out = result.decision;
  out = raw(context);
  if (impl_->stage) {
    ++impl_->audit.second_writers;
    out.receipt_acks = impl_->settled_action_acks;
    out.purchase_receipt_acks = impl_->settled_purchase_acks;
    out.telemetry.fail_closed = 1;
    return result;
  }
  if (impl_->stopped) {
    out.receipt_acks = impl_->settled_action_acks;
    out.purchase_receipt_acks = impl_->settled_purchase_acks;
    out.telemetry.fail_closed = 1;
    return result;
  }

  if (impl_->settled_player == context.player &&
      impl_->settled_step == context.step) {
    out.receipt_acks = impl_->settled_action_acks;
    out.purchase_receipt_acks = impl_->settled_purchase_acks;
  } else {
    const auto clear_routed = [&]() {
      for (auto &values : impl_->routed_actions)
        values.clear();
      for (auto &values : impl_->routed_purchases)
        values.clear();
    };
    std::vector<bool> used_actions(context.previous_action_receipts.size());
    for (const auto &pending : impl_->pending_actions) {
      std::size_t match = context.previous_action_receipts.size();
      for (std::size_t i = 0; i < context.previous_action_receipts.size(); ++i)
        if (!used_actions[i] &&
            same_action_receipt(context.previous_action_receipts[i],
                                pending.shape)) {
          match = i;
          break;
        }
      if (match == context.previous_action_receipts.size()) {
        ++impl_->audit.settlement_failures;
        clear_routed();
        out.telemetry.fail_closed = 1;
        return result;
      }
      used_actions[match] = true;
      impl_->routed_actions[pending.child].push_back(
          context.previous_action_receipts[match]);
    }
    std::vector<bool> used_purchases(
        context.previous_purchase_receipts.size());
    for (const auto &pending : impl_->pending_purchases) {
      std::size_t match = context.previous_purchase_receipts.size();
      for (std::size_t i = 0; i < context.previous_purchase_receipts.size();
           ++i) {
        auto expected = pending.shape;
        expected.filled = context.previous_purchase_receipts[i].filled;
        if (!used_purchases[i] &&
            same_purchase_shape(context.previous_purchase_receipts[i],
                                expected)) {
          match = i;
          break;
        }
      }
      if (match == context.previous_purchase_receipts.size()) {
        ++impl_->audit.settlement_failures;
        clear_routed();
        out.telemetry.fail_closed = 1;
        return result;
      }
      used_purchases[match] = true;
      impl_->routed_purchases[pending.child].push_back(
          context.previous_purchase_receipts[match]);
    }
    std::array<SettlementResult, 2> settlement_preview;
    for (int child = 0; child < 2; ++child) {
      settlement_preview[child] = impl_->owners[child]->validate_settle(
          context, impl_->routed_actions[child],
          impl_->routed_purchases[child]);
      if (settlement_preview[child] == SettlementResult::ProtocolInvalid) {
        ++impl_->audit.settlement_failures;
        clear_routed();
        out.telemetry.fail_closed = 1;
        return result;
      }
    }
    for (int child = 0; child < 2; ++child) {
      const auto applied = impl_->owners[child]->settle_owned(
          context, impl_->routed_actions[child],
          impl_->routed_purchases[child]);
      if (applied == SettlementResult::ProtocolInvalid ||
          applied != settlement_preview[child]) {
        // The snapshot promised this exact single-writer result. A contract
        // violation may have partially advanced one child.
        ++impl_->audit.settlement_failures;
        impl_->stopped = true;
        clear_routed();
        out.telemetry.fail_closed = 1;
        return result;
      }
      out.telemetry.receipts_failed +=
          applied == SettlementResult::AppliedPhysicalFailure;
    }
    for (std::size_t i = 0; i < used_actions.size(); ++i)
      if (used_actions[i])
        out.receipt_acks.push_back(context.previous_action_receipts[i]);
    for (std::size_t i = 0; i < used_purchases.size(); ++i)
      if (used_purchases[i])
        out.purchase_receipt_acks.push_back(
            context.previous_purchase_receipts[i]);
    clear_routed();
    impl_->pending_actions.clear();
    impl_->pending_purchases.clear();
    impl_->settled_player = context.player;
    impl_->settled_step = context.step;
    impl_->settled_action_acks = out.receipt_acks;
    impl_->settled_purchase_acks = out.purchase_receipt_acks;
  }

  std::array<PreparedRepair, 2> proposals{
      impl_->owners[0]->prepare(context, input),
      impl_->owners[1]->prepare(context, input)};
  bool invalid = authority && (authority->player != context.player ||
                               authority->day != context.phase_start.day() ||
                               authority->step != context.step ||
                               authority->certificate_hash == 0 ||
                               authority->issuer_generation == 0 ||
                               authority->units.size() != out.units.size() ||
                               authority->sources.size() != out.sources.size());
  std::map<int, int> actor_owner;
  for (int child = 0; child < 2; ++child) {
    const auto &p = proposals[child];
    std::set<int> local;
    invalid = invalid || p.token == 0 || p.units.size() != out.units.size() ||
              p.sources.size() != out.sources.size();
    for (int actor : p.claimed_actors) {
      if (actor < 0 || actor >= static_cast<int>(out.units.size()) ||
          !local.insert(actor).second ||
          same(p.units[actor], context.raw_g001.units[actor]) ||
          (!authority && (move(p.units[actor].op) ||
                          move(context.raw_g001.units[actor].op))) ||
          p.sources[actor].actor != actor ||
          (!authority && p.sources[actor].source_step != -1))
        invalid = true;
      if (!actor_owner.emplace(actor, child).second)
        ++impl_->audit.actor_conflicts;
    }
  }
  bool duplicate_actor = false;
  for (int actor : proposals[0].claimed_actors)
    duplicate_actor = duplicate_actor ||
                      std::find(proposals[1].claimed_actors.begin(),
                                proposals[1].claimed_actors.end(),
                                actor) != proposals[1].claimed_actors.end();
  if (authority && !invalid) {
    std::set<std::pair<int, int>> displaced;
    for (const auto &shift : authority->displaced_moves) {
      const bool valid_shift =
          shift.actor >= 0 &&
          shift.actor < static_cast<int>(out.units.size()) &&
          shift.source_step == context.step &&
          shift.emitted_step > shift.source_step &&
          shift.source_step / context.phase_start.config().turns_per_day ==
              shift.emitted_step / context.phase_start.config().turns_per_day &&
          shift.day_suffix_certificate_hash == authority->certificate_hash &&
          shift.issuer_generation == authority->issuer_generation &&
          actor_owner.contains(shift.actor) &&
          same(shift.source_action,
               context.raw_g001.units[static_cast<std::size_t>(shift.actor)]) &&
          move(shift.source_action.op) &&
          displaced.insert({shift.actor, shift.source_step}).second;
      invalid = invalid || !valid_shift;
    }
    for (std::size_t actor = 0; actor < out.units.size(); ++actor) {
      const auto &scheduled_source = authority->sources[actor];
      invalid = invalid || scheduled_source.actor != static_cast<int>(actor) ||
                !same(scheduled_source.source_action, authority->units[actor]);
      const auto owner = actor_owner.find(static_cast<int>(actor));
      const bool schedule_differs =
          !same(authority->units[actor], context.raw_g001.units[actor]) ||
          (move(authority->units[actor].op) &&
           scheduled_source.source_step != context.step);
      if (owner == actor_owner.end()) {
        invalid = invalid || schedule_differs;
      } else {
        const auto &proposal = proposals[owner->second];
        invalid = invalid ||
                  !same(proposal.units[actor], authority->units[actor]) ||
                  proposal.sources[actor].actor != scheduled_source.actor ||
                  proposal.sources[actor].source_step !=
                      scheduled_source.source_step ||
                  !same(proposal.sources[actor].source_action,
                        scheduled_source.source_action);
      }
      if (move(context.raw_g001.units[actor].op) &&
          !same(authority->units[actor], context.raw_g001.units[actor]))
        invalid = invalid ||
                  !displaced.contains({static_cast<int>(actor), context.step});
    }
  }
  if (invalid || duplicate_actor) {
    if (invalid)
      ++impl_->audit.invalid_proposals;
    for (int child = 0; child < 2; ++child) {
      impl_->owners[child]->abort(proposals[child].token);
      ++impl_->audit.aborted_proposals;
    }
    return result;
  }

  for (const auto &[actor, child] : actor_owner) {
    out.units[actor] = proposals[child].units[actor];
    out.sources[actor] = proposals[child].sources[actor];
  }
  struct Work {
    int child{};
    repair_fork::RequiredPurchase local;
    repair_fork::RequiredPurchase external;
  };
  std::vector<Work> work;
  auto staged_debts = impl_->debts;
  auto staged_next_debt = impl_->next_debt;
  for (int child = 0; child < 2; ++child)
    for (const auto &local : proposals[child].required_purchases) {
      auto external = local;
      external.debt_id = Impl::external_debt(
          child, local.debt_id, staged_debts, staged_next_debt);
      work.push_back({child, local, external});
    }
  repair_fork::ExactMarketCompiler compiler;
  std::array<std::vector<repair_fork::PurchaseBinding>, 2> individual;
  for (int child = 0; child < 2; ++child)
    individual[child] = compiler
                            .compile(context.phase_start, context.player,
                                     context.raw_g001.market,
                                     proposals[child].required_purchases)
                            .bindings;
  std::vector<repair_fork::RequiredPurchase> external;
  for (const auto &value : work)
    external.push_back(value.external);
  const auto compiled = compiler.compile(context.phase_start, context.player,
                                         context.raw_g001.market, external);
  std::array<std::size_t, 2> local_index{};
  bool market_conflict =
      compiled.bindings.size() != work.size() ||
      individual[0].size() != proposals[0].required_purchases.size() ||
      individual[1].size() != proposals[1].required_purchases.size();
  for (std::size_t i = 0; i < work.size() && !market_conflict; ++i) {
    const auto local = individual[work[i].child][local_index[work[i].child]++];
    if (accepted(local.status) && !accepted(compiled.bindings[i].status))
      market_conflict = true;
  }
  if (market_conflict) {
    ++impl_->audit.market_conflicts;
    for (int child = 0; child < 2; ++child) {
      impl_->owners[child]->abort(proposals[child].token);
      ++impl_->audit.aborted_proposals;
    }
    out = raw(context);
    out.receipt_acks = impl_->settled_action_acks;
    out.purchase_receipt_acks = impl_->settled_purchase_acks;
    return result;
  }
  out.required_purchases = external;

  auto final_joint = context.raw_joint;
  final_joint[context.player].units = out.units;
  std::array<CommitGrant, 2> grants;
  auto staged_generations = impl_->generations;
  std::vector<Impl::PendingAction> staged_actions;
  std::vector<Impl::PendingPurchase> staged_purchases;
  for (auto &grant : grants) {
    grant.submitted_step = context.step;
    grant.final_units = out.units;
  }
  for (const auto &[actor, child] : actor_owner) {
    auto &generation = staged_generations[actor];
    generation = generation == 0 ? (1ULL << 32U) : generation + 1;
    repair_fork::ActorPrefixAuthority authority{
        actor, generation,
        repair_fork::unit_prefix_manifest_hash(out.units, actor),
        repair_fork::post_unit_prefix_state_fingerprint(
            context.phase_start, context.player, final_joint, actor)};
    out.prefix_authority.push_back(authority);
    grants[child].actions.push_back({actor, out.units[actor], authority});
    staged_actions.push_back(
        {child,
         {context.step, actor, authority.manifest_generation,
          authority.prefix_manifest_hash,
          authority.post_prefix_state_fingerprint, out.units[actor]}});
  }
  if (authority)
    for (const auto &shift : authority->displaced_moves) {
      grants[actor_owner.at(shift.actor)].certified_move_shifts.push_back(
          shift);
    }
  local_index = {};
  for (std::size_t i = 0; i < work.size(); ++i) {
    const auto &binding = compiled.bindings[i];
    grants[work[i].child].purchases.push_back(
        {work[i].local.debt_id, work[i].external.debt_id, binding});
    staged_purchases.push_back(
        {work[i].child,
         {work[i].external.debt_id, context.step, binding.operation,
          binding.item, binding.requested, 0, binding.market_slot,
          binding.status}});
  }
  std::array<bool, 2> active{};
  for (int child = 0; child < 2; ++child) {
    active[child] = !proposals[child].claimed_actors.empty() ||
                    !proposals[child].required_purchases.empty();
    if (active[child] && !impl_->owners[child]->validate_commit(
                             proposals[child].token, grants[child])) {
      for (int owner = 0; owner < 2; ++owner) {
        impl_->owners[owner]->abort(proposals[owner].token);
        ++impl_->audit.aborted_proposals;
      }
      ++impl_->audit.invalid_proposals;
      out = raw(context);
      out.receipt_acks = impl_->settled_action_acks;
      out.purchase_receipt_acks = impl_->settled_purchase_acks;
      return result;
    }
  }
  bool any_active = false;
  for (int child = 0; child < 2; ++child) {
    any_active = any_active || active[child];
    if (!active[child] && proposals[child].token != 0) {
      impl_->owners[child]->abort(proposals[child].token);
      proposals[child].token = 0;
      ++impl_->audit.aborted_proposals;
    }
  }
  if (!any_active)
    return result;

  if (impl_->next_transaction == 0 ||
      impl_->next_transaction == std::numeric_limits<std::uint64_t>::max()) {
    for (int child = 0; child < 2; ++child)
      if (proposals[child].token != 0) {
        impl_->owners[child]->abort(proposals[child].token);
        ++impl_->audit.aborted_proposals;
      }
    impl_->stopped = true;
    out = raw(context);
    out.telemetry.fail_closed = 1;
    return result;
  }
  result.token = impl_->next_transaction++;
  fastkag::PlayerAction expected = context.raw_g001;
  expected.units = out.units;
  expected.market = compiled.market;
  impl_->stage.emplace(Impl::Stage{
      result.token,
      std::move(proposals),
      std::move(grants),
      active,
      std::move(expected),
      std::move(staged_generations),
      std::move(staged_debts),
      staged_next_debt,
      std::move(staged_actions),
      std::move(staged_purchases)});
  ++impl_->audit.prepared_transactions;
  return result;
}

bool TransactionalOwnerComposer::finalize(
    std::uint64_t token, const fastkag::PlayerAction &selected) {
  if (!impl_->stage || token == 0 || impl_->stage->token != token) {
    ++impl_->audit.stale_transactions;
    return false;
  }
  auto &stage = *impl_->stage;
  if (!same(stage.expected, selected)) {
    ++impl_->audit.exact_action_rejections;
    return false;
  }
  for (int child = 0; child < 2; ++child)
    if (stage.active[child] &&
        !impl_->owners[child]->validate_commit(stage.proposals[child].token,
                                               stage.grants[child])) {
      for (int owner = 0; owner < 2; ++owner)
        if (stage.proposals[owner].token != 0) {
          impl_->owners[owner]->abort(stage.proposals[owner].token);
          ++impl_->audit.aborted_proposals;
        }
      ++impl_->audit.invalid_proposals;
      impl_->stage.reset();
      return false;
    }

  // No other writer can reach either child while this stage is outstanding.
  // validate_commit runs the identical transition on an owner snapshot, so
  // these two publishes are no-fail under the TransactionalRepairOwner ABI.
  for (int child = 0; child < 2; ++child)
    if (stage.active[child]) {
      if (!impl_->owners[child]->commit(stage.proposals[child].token,
                                        stage.grants[child])) {
        for (int owner = child + 1; owner < 2; ++owner)
          if (stage.proposals[owner].token != 0)
            impl_->owners[owner]->abort(stage.proposals[owner].token);
        ++impl_->audit.commit_fail_stops;
        impl_->stopped = true;
        impl_->stage.reset();
        return false;
      }
      ++impl_->audit.committed_proposals;
    }
  impl_->generations = std::move(stage.generations);
  impl_->debts = std::move(stage.debts);
  impl_->next_debt = stage.next_debt;
  impl_->pending_actions = std::move(stage.pending_actions);
  impl_->pending_purchases = std::move(stage.pending_purchases);
  ++impl_->audit.finalized_transactions;
  impl_->stage.reset();
  return true;
}

bool TransactionalOwnerComposer::abort(std::uint64_t token) noexcept {
  if (!impl_->stage || token == 0 || impl_->stage->token != token) {
    ++impl_->audit.stale_transactions;
    return false;
  }
  for (int child = 0; child < 2; ++child)
    if (impl_->stage->proposals[child].token != 0) {
      impl_->owners[child]->abort(impl_->stage->proposals[child].token);
      ++impl_->audit.aborted_proposals;
    }
  impl_->stage.reset();
  ++impl_->audit.aborted_transactions;
  return true;
}

const TransactionalComposerAudit &
TransactionalOwnerComposer::audit() const noexcept {
  return impl_->audit;
}
bool TransactionalOwnerComposer::fail_stopped() const noexcept {
  return impl_->stopped;
}

} // namespace g001::repair_owner_composer
