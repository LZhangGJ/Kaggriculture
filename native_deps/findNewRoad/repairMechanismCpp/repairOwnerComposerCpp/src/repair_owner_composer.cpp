#include "repair_owner_composer.hpp"

#include "online_animal_repair_owner.hpp"
#include "online_elastic_repair_owner.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace g001::repair_owner_composer {
namespace {

using repair_fork::ActionReceipt;
using repair_fork::ActorPrefixAuthority;
using repair_fork::PurchaseBinding;
using repair_fork::PurchaseCompileStatus;
using repair_fork::PurchaseReceipt;
using repair_fork::RepairContext;
using repair_fork::RepairDecision;
using repair_fork::RequiredPurchase;
using fastkag::Action;
using fastkag::Op;

enum class Child : std::uint8_t { Crop = 0, Animal = 1 };

bool same_action(const Action& left, const Action& right) {
  return left.op == right.op && left.item == right.item &&
      left.quantity == right.quantity;
}

bool same_action_receipt(const ActionReceipt& left,
                         const ActionReceipt& right) {
  return left.submitted_step == right.submitted_step &&
      left.actor == right.actor &&
      left.manifest_generation == right.manifest_generation &&
      left.prefix_manifest_hash == right.prefix_manifest_hash &&
      left.post_prefix_state_fingerprint ==
          right.post_prefix_state_fingerprint &&
      same_action(left.emitted, right.emitted);
}

bool same_purchase_receipt(const PurchaseReceipt& left,
                           const PurchaseReceipt& right) {
  return left.debt_id == right.debt_id &&
      left.submitted_step == right.submitted_step &&
      left.operation == right.operation && left.item == right.item &&
      left.requested == right.requested && left.filled == right.filled &&
      left.market_slot == right.market_slot &&
      left.compile_status == right.compile_status;
}

bool is_move(Op op) {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
      op == Op::WEST;
}

bool accepted(PurchaseCompileStatus status) {
  return status == PurchaseCompileStatus::BoundExisting ||
      status == PurchaseCompileStatus::Appended;
}

RepairDecision raw_decision(const RepairContext& context) {
  RepairDecision out;
  out.units = context.raw_g001.units;
  out.sources.reserve(out.units.size());
  for (std::size_t actor = 0; actor < out.units.size(); ++actor)
    out.sources.push_back({static_cast<int>(actor), context.step,
                           out.units[actor]});
  out.receipt_acks.assign(context.previous_action_receipts.begin(),
                          context.previous_action_receipts.end());
  out.purchase_receipt_acks.assign(
      context.previous_purchase_receipts.begin(),
      context.previous_purchase_receipts.end());
  return out;
}

}  // namespace

struct RepairOwnerComposer::Impl {
  struct ActionLink {
    Child child{Child::Crop};
    ActionReceipt local;
    std::optional<ActionReceipt> external;
  };
  struct PurchaseLink {
    Child child{Child::Crop};
    PurchaseReceipt local_shape;
    PurchaseReceipt external_shape;
  };
  struct ChildInputs {
    std::vector<ActionReceipt> actions;
    std::vector<PurchaseReceipt> purchases;
  };
  struct Proposal {
    Child child{Child::Crop};
    RepairDecision decision;
    std::vector<int> claims;
  };

  std::unique_ptr<repair_fork::RepairOwner> crop;
  std::unique_ptr<repair_fork::RepairOwner> animal;
  Audit audit;
  bool stopped{};
  int last_step{-1};
  std::map<int, std::uint64_t> actor_generations;
  std::map<std::pair<Child, std::uint64_t>, std::uint64_t> external_debts;
  std::uint64_t next_external_debt{1ULL << 62};
  std::vector<ActionLink> pending_actions;
  std::vector<PurchaseLink> pending_purchases;

  Impl(std::unique_ptr<repair_fork::RepairOwner> crop_owner,
       std::unique_ptr<repair_fork::RepairOwner> animal_owner)
      : crop(std::move(crop_owner)), animal(std::move(animal_owner)) {
    if (!crop || !animal)
      throw std::invalid_argument("repair composer requires two owners");
  }

  repair_fork::RepairOwner& owner(Child child) {
    return child == Child::Crop ? *crop : *animal;
  }

  void stop() {
    if (!stopped) ++audit.fail_stops;
    stopped = true;
    pending_actions.clear();
    pending_purchases.clear();
  }

  std::uint64_t map_debt(Child child, std::uint64_t local) {
    if (local == 0) return 0;
    const auto key = std::make_pair(child, local);
    const auto found = external_debts.find(key);
    if (found != external_debts.end()) return found->second;
    if (next_external_debt == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("repair composer debt namespace exhausted");
    const auto external = next_external_debt++;
    external_debts.emplace(key, external);
    return external;
  }

  bool translate_previous(const RepairContext& context,
                          std::array<ChildInputs, 2>& translated) {
    std::vector<const ActionReceipt*> external_actions;
    external_actions.reserve(context.previous_action_receipts.size());
    for (const auto& receipt : context.previous_action_receipts)
      external_actions.push_back(&receipt);
    std::vector<bool> used_actions(external_actions.size());
    for (const auto& link : pending_actions) {
      if (!link.external) {
        translated[static_cast<std::size_t>(link.child)].actions.push_back(
            link.local);
        ++audit.raw_submission_witnesses;
        continue;
      }
      std::size_t match = external_actions.size();
      for (std::size_t index = 0; index < external_actions.size(); ++index) {
        if (!used_actions[index] &&
            same_action_receipt(*external_actions[index], *link.external)) {
          match = index;
          break;
        }
      }
      if (match == external_actions.size()) return false;
      used_actions[match] = true;
      translated[static_cast<std::size_t>(link.child)].actions.push_back(
          link.local);
      ++audit.translated_action_receipts;
    }
    if (std::any_of(used_actions.begin(), used_actions.end(),
                    [](bool value) { return !value; }))
      return false;

    std::vector<bool> used_purchases(
        context.previous_purchase_receipts.size());
    for (const auto& link : pending_purchases) {
      std::size_t match = context.previous_purchase_receipts.size();
      for (std::size_t index = 0;
           index < context.previous_purchase_receipts.size(); ++index) {
        auto expected = link.external_shape;
        expected.filled = context.previous_purchase_receipts[index].filled;
        if (!used_purchases[index] && same_purchase_receipt(
                context.previous_purchase_receipts[index], expected)) {
          match = index;
          break;
        }
      }
      if (match == context.previous_purchase_receipts.size()) return false;
      used_purchases[match] = true;
      auto local = link.local_shape;
      local.filled = context.previous_purchase_receipts[match].filled;
      translated[static_cast<std::size_t>(link.child)].purchases.push_back(
          local);
      ++audit.translated_purchase_receipts;
    }
    return !std::any_of(used_purchases.begin(), used_purchases.end(),
                        [](bool value) { return !value; });
  }

  bool valid_child(const RepairContext& context, Proposal& proposal) {
    const auto& decision = proposal.decision;
    if (decision.units.size() != context.raw_g001.units.size() ||
        decision.sources.size() != decision.units.size())
      return false;
    std::set<int> claims;
    for (const auto& authority : decision.prefix_authority) {
      if (authority.actor < 0 ||
          authority.actor >= static_cast<int>(decision.units.size()) ||
          !claims.insert(authority.actor).second)
        return false;
      proposal.claims.push_back(authority.actor);
    }
    for (int actor = 0; actor < static_cast<int>(decision.units.size());
         ++actor) {
      const auto& raw = context.raw_g001.units[static_cast<std::size_t>(actor)];
      const auto& emitted = decision.units[static_cast<std::size_t>(actor)];
      if (same_action(raw, emitted)) continue;
      if (!claims.contains(actor)) return false;
      // This MVP never accepts a child's MOVE rewrite. Exact raw MOVE source,
      // slot and direction remain owned by G001 until a single scheduler owns
      // the real FrozenDaySuffixCertificate.
      if (is_move(raw.op) || is_move(emitted.op)) return false;
      const auto& source = decision.sources[static_cast<std::size_t>(actor)];
      if (source.actor != actor || source.source_step != -1) return false;
    }
    for (int actor : proposal.claims) {
      const auto found = std::find_if(
          decision.prefix_authority.begin(), decision.prefix_authority.end(),
          [&](const ActorPrefixAuthority& value) {
            return value.actor == actor;
          });
      if (found == decision.prefix_authority.end()) return false;
    }
    return true;
  }

  bool child_acked(const RepairDecision& decision,
                   const ChildInputs& inputs) const {
    if (decision.receipt_acks.size() != inputs.actions.size() ||
        decision.purchase_receipt_acks.size() != inputs.purchases.size())
      return false;
    for (std::size_t index = 0; index < inputs.actions.size(); ++index)
      if (!same_action_receipt(decision.receipt_acks[index],
                               inputs.actions[index]))
        return false;
    for (std::size_t index = 0; index < inputs.purchases.size(); ++index)
      if (!same_purchase_receipt(decision.purchase_receipt_acks[index],
                                 inputs.purchases[index]))
        return false;
    return true;
  }
};

RepairOwnerComposer::RepairOwnerComposer()
    : impl_(std::make_unique<Impl>(
          std::make_unique<online_elastic::OnlineElasticRepairOwner>(),
          std::make_unique<online_animal_repair::OnlineAnimalRepairOwner>())) {}

RepairOwnerComposer::RepairOwnerComposer(
    std::unique_ptr<repair_fork::RepairOwner> crop,
    std::unique_ptr<repair_fork::RepairOwner> animal)
    : impl_(std::make_unique<Impl>(std::move(crop), std::move(animal))) {}

RepairOwnerComposer::~RepairOwnerComposer() = default;
RepairOwnerComposer::RepairOwnerComposer(RepairOwnerComposer&&) noexcept =
    default;
RepairOwnerComposer& RepairOwnerComposer::operator=(
    RepairOwnerComposer&&) noexcept = default;

std::string RepairOwnerComposer::name() const {
  return "repair_owner_composer_v1";
}

RepairDecision RepairOwnerComposer::decide(const RepairContext& context) {
  if (context.player < 0 || context.player > 1 || context.step < 0 ||
      context.step != context.phase_start.step_count())
    throw std::invalid_argument("repair composer context is not phase-aligned");
  auto out = raw_decision(context);
  ++impl_->audit.decisions;
  if (impl_->last_step >= 0 && context.step != impl_->last_step + 1) {
    ++impl_->audit.external_receipt_failures;
    impl_->stop();
  }
  impl_->last_step = context.step;
  if (impl_->stopped) {
    out.telemetry.fail_closed = 1;
    return out;
  }

  std::array<Impl::ChildInputs, 2> child_inputs;
  if (!impl_->translate_previous(context, child_inputs)) {
    ++impl_->audit.external_receipt_failures;
    impl_->stop();
    out.telemetry.fail_closed = 1;
    return out;
  }
  impl_->pending_actions.clear();
  impl_->pending_purchases.clear();

  std::array<Impl::Proposal, 2> proposals;
  for (std::size_t index = 0; index < proposals.size(); ++index) {
    const auto child = static_cast<Child>(index);
    const auto& inputs = child_inputs[index];
    const RepairContext child_context{
        context.phase_start, context.player, context.step, context.raw_g001,
        context.raw_joint, inputs.purchases, inputs.actions};
    proposals[index].child = child;
    proposals[index].decision = impl_->owner(child).decide(child_context);
    if (!impl_->child_acked(proposals[index].decision, inputs)) {
      ++impl_->audit.child_ack_failures;
      impl_->stop();
      out.telemetry.fail_closed = 1;
      return out;
    }
    if (!impl_->valid_child(context, proposals[index])) {
      ++impl_->audit.invalid_child_manifests;
      impl_->stop();
      out.telemetry.fail_closed = 1;
      return out;
    }
  }

  std::map<int, std::size_t> actor_owner;
  for (std::size_t index = 0; index < proposals.size(); ++index) {
    for (int actor : proposals[index].claims) {
      if (!actor_owner.emplace(actor, index).second) {
        ++impl_->audit.unit_conflicts;
        impl_->stop();
        out.telemetry.fail_closed = 1;
        return out;
      }
    }
  }

  for (const auto& [actor, owner_index] : actor_owner) {
    const auto& proposal = proposals[owner_index].decision;
    out.units[static_cast<std::size_t>(actor)] =
        proposal.units[static_cast<std::size_t>(actor)];
    out.sources[static_cast<std::size_t>(actor)] =
        proposal.sources[static_cast<std::size_t>(actor)];
  }

  struct PurchaseWork {
    Child child{Child::Crop};
    RequiredPurchase local;
    RequiredPurchase external;
    PurchaseBinding local_binding;
  };
  std::vector<PurchaseWork> work;
  repair_fork::ExactMarketCompiler compiler;
  for (const auto& proposal : proposals) {
    const auto& required = proposal.decision.required_purchases;
    const auto local_compiled = compiler.compile(
        context.phase_start, context.player, context.raw_g001.market, required);
    if (local_compiled.bindings.size() != required.size()) {
      ++impl_->audit.invalid_child_manifests;
      impl_->stop();
      out = raw_decision(context);
      out.telemetry.fail_closed = 1;
      return out;
    }
    std::set<std::uint64_t> local_ids;
    for (std::size_t index = 0; index < required.size(); ++index) {
      if (required[index].debt_id == 0 ||
          !local_ids.insert(required[index].debt_id).second) {
        ++impl_->audit.invalid_child_manifests;
        impl_->stop();
        out = raw_decision(context);
        out.telemetry.fail_closed = 1;
        return out;
      }
      auto external = required[index];
      external.debt_id = impl_->map_debt(proposal.child,
                                         required[index].debt_id);
      work.push_back({proposal.child, required[index], external,
                      local_compiled.bindings[index]});
    }
  }
  std::vector<RequiredPurchase> external_required;
  external_required.reserve(work.size());
  for (const auto& value : work) external_required.push_back(value.external);
  const auto joint_compiled = compiler.compile(
      context.phase_start, context.player, context.raw_g001.market,
      external_required);
  if (joint_compiled.bindings.size() != work.size()) {
    ++impl_->audit.invalid_child_manifests;
    impl_->stop();
    out = raw_decision(context);
    out.telemetry.fail_closed = 1;
    return out;
  }
  for (std::size_t index = 0; index < work.size(); ++index) {
    if (accepted(work[index].local_binding.status) &&
        !accepted(joint_compiled.bindings[index].status)) {
      ++impl_->audit.market_conflicts;
      impl_->stop();
      out = raw_decision(context);
      out.telemetry.fail_closed = 1;
      return out;
    }
  }
  out.required_purchases = std::move(external_required);

  auto final_joint = context.raw_joint;
  final_joint[static_cast<std::size_t>(context.player)].units = out.units;
  for (const auto& [actor, owner_index] : actor_owner) {
    const auto& proposal = proposals[owner_index];
    const auto local_authority = std::find_if(
        proposal.decision.prefix_authority.begin(),
        proposal.decision.prefix_authority.end(),
        [&](const ActorPrefixAuthority& value) { return value.actor == actor; });
    if (local_authority == proposal.decision.prefix_authority.end()) {
      ++impl_->audit.invalid_child_manifests;
      impl_->stop();
      out = raw_decision(context);
      out.telemetry.fail_closed = 1;
      return out;
    }
    const ActionReceipt local_receipt{
        context.step, actor, local_authority->manifest_generation,
        local_authority->prefix_manifest_hash,
        local_authority->post_prefix_state_fingerprint,
        proposal.decision.units[static_cast<std::size_t>(actor)]};
    std::optional<ActionReceipt> external_receipt;
    if (!same_action(out.units[static_cast<std::size_t>(actor)],
                     context.raw_g001.units[static_cast<std::size_t>(actor)])) {
      auto& generation = impl_->actor_generations[actor];
      ++generation;
      const ActorPrefixAuthority authority{
          actor, generation,
          repair_fork::unit_prefix_manifest_hash(out.units, actor),
          repair_fork::post_unit_prefix_state_fingerprint(
              context.phase_start, context.player, final_joint, actor)};
      out.prefix_authority.push_back(authority);
      external_receipt = ActionReceipt{
          context.step, actor, authority.manifest_generation,
          authority.prefix_manifest_hash,
          authority.post_prefix_state_fingerprint,
          out.units[static_cast<std::size_t>(actor)]};
      ++impl_->audit.composed_unit_actions;
    }
    impl_->pending_actions.push_back(
        {proposal.child, local_receipt, external_receipt});
  }

  for (std::size_t index = 0; index < work.size(); ++index) {
    const auto& local_binding = work[index].local_binding;
    const auto& external_binding = joint_compiled.bindings[index];
    impl_->pending_purchases.push_back(
        {work[index].child,
         {work[index].local.debt_id, context.step,
          local_binding.operation, local_binding.item,
          local_binding.requested, 0, local_binding.market_slot,
          local_binding.status},
         {work[index].external.debt_id, context.step,
          external_binding.operation, external_binding.item,
          external_binding.requested, 0, external_binding.market_slot,
          external_binding.status}});
    ++impl_->audit.composed_purchases;
  }

  for (const auto& proposal : proposals) {
    out.telemetry.triggers += proposal.decision.telemetry.triggers;
    out.telemetry.debts_opened += proposal.decision.telemetry.debts_opened;
    out.telemetry.debts_closed += proposal.decision.telemetry.debts_closed;
    out.telemetry.receipts_confirmed +=
        proposal.decision.telemetry.receipts_confirmed;
    out.telemetry.receipts_failed +=
        proposal.decision.telemetry.receipts_failed;
    out.telemetry.fail_closed += proposal.decision.telemetry.fail_closed;
  }
  if (context.step == context.phase_start.config().episode_steps - 1) {
    impl_->audit.terminal_unacked_action_receipts += static_cast<int>(
        std::count_if(impl_->pending_actions.begin(),
                      impl_->pending_actions.end(),
                      [](const Impl::ActionLink& link) {
                        return link.external.has_value();
                      }));
    impl_->audit.terminal_unacked_purchase_debts +=
        static_cast<int>(impl_->pending_purchases.size());
  }
  return out;
}

const Audit& RepairOwnerComposer::audit() const noexcept {
  return impl_->audit;
}

bool RepairOwnerComposer::fail_stopped() const noexcept {
  return impl_->stopped;
}

std::size_t RepairOwnerComposer::outstanding_action_receipts() const noexcept {
  return impl_->pending_actions.size();
}

std::size_t RepairOwnerComposer::outstanding_purchase_receipts() const noexcept {
  return impl_->pending_purchases.size();
}

}  // namespace g001::repair_owner_composer
