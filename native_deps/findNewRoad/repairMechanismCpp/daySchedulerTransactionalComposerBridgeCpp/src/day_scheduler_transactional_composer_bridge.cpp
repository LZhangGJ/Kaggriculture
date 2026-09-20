#include "day_scheduler_transactional_composer_bridge.hpp"

#include "production_suffix_scheduler.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace g001::day_scheduler_transactional_bridge {
namespace {

bool same(const fastkag::Action &left, const fastkag::Action &right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

bool same(const fastkag::PlayerAction &left,
          const fastkag::PlayerAction &right) {
  return left.units.size() == right.units.size() &&
         left.market.size() == right.market.size() &&
         std::equal(left.units.begin(), left.units.end(), right.units.begin(),
                    [](const auto &a, const auto &b) { return same(a, b); }) &&
         std::equal(left.market.begin(), left.market.end(),
                    right.market.begin(),
                    [](const auto &a, const auto &b) { return same(a, b); });
}

bool move(fastkag::Op op) {
  return op == fastkag::Op::NORTH || op == fastkag::Op::SOUTH ||
         op == fastkag::Op::EAST || op == fastkag::Op::WEST;
}

template <class Value> void hash_value(std::uint64_t &hash, Value value) {
  const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
  for (std::size_t index = 0; index < sizeof(Value); ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
}

repair_fork::RepairDecision raw(const repair_fork::RepairContext &context) {
  repair_fork::RepairDecision output;
  output.units = context.raw_g001.units;
  for (std::size_t actor = 0; actor < output.units.size(); ++actor)
    output.sources.push_back(
        {static_cast<int>(actor), context.step, output.units[actor]});
  return output;
}

} // namespace

std::uint64_t debt_authorization_hash(const DebtAuthorization &authorization) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_value(hash, authorization.certificate_hash);
  hash_value(hash, authorization.obligation_id);
  hash_value(hash, authorization.source_step);
  hash_value(hash, static_cast<std::uint8_t>(authorization.goal));
  hash_value(hash, authorization.actor);
  hash_value(hash, authorization.tile.x);
  hash_value(hash, authorization.tile.y);
  hash_value(hash, static_cast<std::uint8_t>(authorization.item));
  hash_value(hash, authorization.damage_policy_version);
  hash_value(hash, authorization.damage_policy_hash);
  hash_value(hash, authorization.authorization_generation);
  return hash;
}

struct DaySchedulerTransactionalComposerBridge::Impl {
  struct Stage {
    std::uint64_t token{};
    std::uint64_t composer_token{};
    repair_fork::RepairDecision decision;
    fastkag::PlayerAction expected;
    std::set<std::pair<int, int>> submitted_moves;
    int submitted_this_step{};
    int step{-1};
  };
  fastkag::Simulator day_start;
  obligation_day::DayPlanRequest request;
  obligation_day::DayScheduleCertificate certificate;
  repair_owner_composer::TransactionalOwnerComposer composer;
  std::map<int, fastkag::Simulator> expected_pre_states;
  std::set<std::pair<int, int>> submitted_moves;
  Audit audit;
  bool verified{};
  bool stopped{};
  int last_step{-1};
  std::uint64_t next_stage{1};
  std::optional<Stage> stage;

  Impl(std::unique_ptr<repair_owner_composer::TransactionalRepairOwner> crop,
       std::unique_ptr<repair_owner_composer::TransactionalRepairOwner> animal,
       const obligation_day::DayPlanRequest &source_request,
       const obligation_day::DayScheduleCertificate &source_certificate,
       std::span<const DebtAuthorization> debt_authorizations)
      : day_start(source_request.day_start ? *source_request.day_start
                                           : fastkag::Simulator{}),
        request(source_request), certificate(source_certificate),
        composer(std::move(crop), std::move(animal)) {
    request.day_start = source_request.day_start ? &day_start : nullptr;
    const auto verification =
        obligation_day::verify_day_schedule(request, certificate);
    if (!verification.valid || !request.day_start ||
        certificate.move_replays.size() != request.moves.size()) {
      ++audit.certificate_rejections;
      return;
    }
    std::map<std::uint64_t, obligation_day::ObligationDisposition> status;
    for (const auto &value : certificate.obligation_statuses)
      status.emplace(value.obligation_id, value.disposition);
    std::set<std::uint64_t> used_authorizations;
    for (const auto &obligation : request.obligations) {
      if (!obligation.must_finish_today ||
          status[obligation.id] ==
              obligation_day::ObligationDisposition::Completed)
        continue;
      const auto matches = [&](const DebtAuthorization &authorization) {
        return authorization.certificate_hash == certificate.content_hash &&
               authorization.obligation_id == obligation.id &&
               obligation.policy_deferred && obligation.source_step >= 0 &&
               authorization.source_step == obligation.source_step &&
               authorization.goal == obligation.goal &&
               authorization.actor == obligation.actor &&
               authorization.tile.x == obligation.tile.x &&
               authorization.tile.y == obligation.tile.y &&
               authorization.item == obligation.item &&
               authorization.damage_policy_version != 0 &&
               authorization.damage_policy_hash != 0 &&
               authorization.authorization_generation != 0 &&
               authorization.content_hash ==
                   debt_authorization_hash(authorization);
      };
      const auto count = static_cast<int>(std::count_if(
          debt_authorizations.begin(), debt_authorizations.end(), matches));
      if (count != 1 || !used_authorizations.insert(obligation.id).second) {
        ++audit.must_finish_rejections;
        ++audit.debt_authorization_rejections;
        return;
      }
      ++audit.authorized_debts;
    }
    for (const auto &authorization : debt_authorizations)
      if (!used_authorizations.contains(authorization.obligation_id)) {
        ++audit.debt_authorization_rejections;
        return;
      }

    auto state = day_start;
    for (const auto &slot : certificate.slots) {
      expected_pre_states.emplace(slot.step, state);
      std::array<fastkag::PlayerAction, 2> joint;
      joint[request.player].units = slot.actions;
      state = state.preview_unit_phase(joint);
    }
    verified = true;
  }

  bool closes_moves(const std::set<std::pair<int, int>> &moves) const {
    if (moves.size() != certificate.move_replays.size())
      return false;
    return std::all_of(certificate.move_replays.begin(),
                       certificate.move_replays.end(), [&](const auto &value) {
                         return moves.contains(
                             {value.actor, value.source_step});
                       });
  }
};

DaySchedulerTransactionalComposerBridge::
    DaySchedulerTransactionalComposerBridge(
        std::unique_ptr<repair_owner_composer::TransactionalRepairOwner> crop,
        std::unique_ptr<repair_owner_composer::TransactionalRepairOwner> animal,
        const obligation_day::DayPlanRequest &request,
        const obligation_day::DayScheduleCertificate &certificate)
    : DaySchedulerTransactionalComposerBridge(
          std::move(crop), std::move(animal), request, certificate,
          std::span<const DebtAuthorization>{}) {}

DaySchedulerTransactionalComposerBridge::
    DaySchedulerTransactionalComposerBridge(
        std::unique_ptr<repair_owner_composer::TransactionalRepairOwner> crop,
        std::unique_ptr<repair_owner_composer::TransactionalRepairOwner> animal,
        const obligation_day::DayPlanRequest &request,
        const obligation_day::DayScheduleCertificate &certificate,
        std::span<const DebtAuthorization> debt_authorizations)
    : impl_(std::make_unique<Impl>(std::move(crop), std::move(animal), request,
                                   certificate, debt_authorizations)) {}

DaySchedulerTransactionalComposerBridge::
    ~DaySchedulerTransactionalComposerBridge() = default;
DaySchedulerTransactionalComposerBridge::
    DaySchedulerTransactionalComposerBridge(
        DaySchedulerTransactionalComposerBridge &&) noexcept = default;
DaySchedulerTransactionalComposerBridge &
DaySchedulerTransactionalComposerBridge::operator=(
    DaySchedulerTransactionalComposerBridge &&) noexcept = default;

std::string DaySchedulerTransactionalComposerBridge::name() const {
  return "day_scheduler_transactional_composer_bridge_v1";
}

repair_fork::RepairDecision DaySchedulerTransactionalComposerBridge::decide(
    const repair_fork::RepairContext &context) {
  auto staged = prepare(context);
  if (staged.token == 0)
    return staged.decision;
  fastkag::PlayerAction selected = context.raw_g001;
  selected.units = staged.decision.units;
  selected.market = repair_fork::ExactMarketCompiler()
                        .compile(context.phase_start, context.player,
                                 context.raw_g001.market,
                                 staged.decision.required_purchases)
                        .market;
  if (finalize(staged.token, selected))
    return staged.decision;
  (void)abort(staged.token);
  auto fallback = raw(context);
  fallback.receipt_acks = staged.decision.receipt_acks;
  fallback.purchase_receipt_acks = staged.decision.purchase_receipt_acks;
  return fallback;
}

StagedDecision DaySchedulerTransactionalComposerBridge::prepare(
    const repair_fork::RepairContext &context,
    repair_owner_composer::TypedRepairInput input) {
  ++impl_->audit.decisions;
  auto fallback = raw(context);
  if (!impl_->verified || impl_->stopped || impl_->stage)
    return {0, std::move(fallback)};
  const int turns = context.phase_start.config().turns_per_day;
  const int day_start = impl_->certificate.day * turns;
  const int day_end = day_start + turns - 1;
  if (context.player != impl_->certificate.player || context.step < day_start ||
      context.step > day_end || context.step <= impl_->last_step) {
    ++impl_->audit.state_divergences;
    impl_->stopped = true;
    return {0, std::move(fallback)};
  }
  const auto expected = impl_->expected_pre_states.find(context.step);
  if (expected == impl_->expected_pre_states.end() ||
      production_suffix::focal_unit_state_fingerprint(
          context.phase_start, context.player, context.step) !=
          production_suffix::focal_unit_state_fingerprint(
              expected->second, context.player, context.step)) {
    ++impl_->audit.state_divergences;
    impl_->stopped = true;
    return {0, std::move(fallback)};
  }

  const auto &slot = impl_->certificate.slots[context.step - day_start];
  repair_owner_composer::ScheduledStepAuthority authority;
  authority.player = context.player;
  authority.day = impl_->certificate.day;
  authority.step = context.step;
  authority.certificate_hash = impl_->certificate.content_hash;
  authority.issuer_generation = impl_->certificate.issuer_generation;
  authority.units = slot.actions;
  authority.sources = slot.sources;
  for (const auto &replay : impl_->certificate.move_replays)
    if (replay.source_step == context.step &&
        replay.emitted_step > replay.source_step)
      authority.displaced_moves.push_back(
          {replay.actor, replay.source_step, replay.emitted_step, replay.action,
           impl_->certificate.content_hash,
           impl_->certificate.issuer_generation});

  auto staged = impl_->composer.prepare_certified(context, authority, input);
  const auto reject_staged = [&]() -> StagedDecision {
    auto rejected = raw(context);
    rejected.receipt_acks = staged.decision.receipt_acks;
    rejected.purchase_receipt_acks = staged.decision.purchase_receipt_acks;
    if (staged.token != 0 && !impl_->composer.abort(staged.token))
      impl_->stopped = true;
    return {0, std::move(rejected)};
  };
  bool exact = staged.decision.units.size() == slot.actions.size() &&
               staged.decision.sources.size() == slot.sources.size();
  for (std::size_t actor = 0; actor < slot.actions.size() && exact; ++actor) {
    exact = same(staged.decision.units[actor], slot.actions[actor]);
    if (move(slot.actions[actor].op)) {
      exact = exact &&
              staged.decision.sources[actor].actor ==
                  slot.sources[actor].actor &&
              staged.decision.sources[actor].source_step ==
                  slot.sources[actor].source_step &&
              same(staged.decision.sources[actor].source_action,
                   slot.sources[actor].source_action);
    }
  }
  if (!exact) {
    ++impl_->audit.manifest_rejections;
    return reject_staged();
  }
  auto submitted_moves = impl_->submitted_moves;
  int submitted_this_step = 0;
  for (std::size_t actor = 0; actor < slot.actions.size(); ++actor)
    if (move(slot.actions[actor].op)) {
      if (!submitted_moves
               .insert(
                   {static_cast<int>(actor), slot.sources[actor].source_step})
               .second) {
        ++impl_->audit.manifest_rejections;
        return reject_staged();
      }
      ++submitted_this_step;
    }
  if (context.step == day_end && !impl_->closes_moves(submitted_moves)) {
    ++impl_->audit.incomplete_day_closures;
    return reject_staged();
  }
  if (impl_->next_stage == 0) {
    if (staged.token != 0)
      (void)impl_->composer.abort(staged.token);
    impl_->stopped = true;
    fallback.receipt_acks = staged.decision.receipt_acks;
    fallback.purchase_receipt_acks = staged.decision.purchase_receipt_acks;
    return {0, std::move(fallback)};
  }
  fastkag::PlayerAction selected = context.raw_g001;
  selected.units = staged.decision.units;
  selected.market = repair_fork::ExactMarketCompiler()
                        .compile(context.phase_start, context.player,
                                 context.raw_g001.market,
                                 staged.decision.required_purchases)
                        .market;
  const auto token = impl_->next_stage++;
  impl_->stage.emplace(Impl::Stage{token, staged.token, staged.decision,
                                  std::move(selected),
                                  std::move(submitted_moves),
                                  submitted_this_step, context.step});
  return {token, staged.decision};
}

bool DaySchedulerTransactionalComposerBridge::finalize(
    std::uint64_t token, const fastkag::PlayerAction &selected) {
  if (!impl_->stage || token == 0 || impl_->stage->token != token ||
      !same(impl_->stage->expected, selected))
    return false;
  if (impl_->stage->composer_token != 0 &&
      !impl_->composer.finalize(impl_->stage->composer_token, selected)) {
    (void)impl_->composer.abort(impl_->stage->composer_token);
    ++impl_->audit.manifest_rejections;
    impl_->stopped = true;
    impl_->stage.reset();
    return false;
  }
  impl_->submitted_moves = std::move(impl_->stage->submitted_moves);
  impl_->audit.move_tokens_submitted += impl_->stage->submitted_this_step;
  impl_->last_step = impl_->stage->step;
  impl_->stage.reset();
  return true;
}

bool DaySchedulerTransactionalComposerBridge::abort(
    std::uint64_t token) noexcept {
  if (!impl_->stage || token == 0 || impl_->stage->token != token)
    return false;
  const bool okay = impl_->stage->composer_token == 0 ||
                    impl_->composer.abort(impl_->stage->composer_token);
  impl_->stage.reset();
  if (!okay)
    impl_->stopped = true;
  return okay;
}

bool DaySchedulerTransactionalComposerBridge::certificate_valid()
    const noexcept {
  return impl_->verified;
}

bool DaySchedulerTransactionalComposerBridge::fail_stopped() const noexcept {
  return impl_->stopped || impl_->composer.fail_stopped();
}

const Audit &DaySchedulerTransactionalComposerBridge::audit() const noexcept {
  return impl_->audit;
}

} // namespace g001::day_scheduler_transactional_bridge
