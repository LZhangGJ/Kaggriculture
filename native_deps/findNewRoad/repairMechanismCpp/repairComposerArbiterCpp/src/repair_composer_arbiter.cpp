#include "repair_composer_arbiter.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>

namespace g001::repair_composer_arbiter {
namespace {

using fastkag::Action;
using fastkag::Op;
using fastkag::PlayerAction;

void hash_add(std::uint64_t& hash, std::uint64_t value) noexcept {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (8 * byte)) & 255U;
    hash *= 1099511628211ULL;
  }
}

bool same(Action lhs, Action rhs) noexcept {
  return lhs.op == rhs.op && lhs.item == rhs.item &&
         lhs.quantity == rhs.quantity;
}

bool same(const PlayerAction& lhs, const PlayerAction& rhs) noexcept {
  if (lhs.units.size() != rhs.units.size() ||
      lhs.market.size() != rhs.market.size())
    return false;
  for (std::size_t index = 0; index < lhs.units.size(); ++index)
    if (!same(lhs.units[index], rhs.units[index])) return false;
  for (std::size_t index = 0; index < lhs.market.size(); ++index)
    if (!same(lhs.market[index], rhs.market[index])) return false;
  return true;
}

bool is_move(Op op) noexcept {
  return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST ||
         op == Op::WEST;
}

std::uint64_t binding_hash(const Prepared& prepared) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, prepared.step);
  hash_add(hash, prepared.generation);
  hash_add(hash, prepared.observation_hash);
  hash_add(hash, prepared.base_action_hash);
  hash_add(hash, prepared.final_action_hash);
  for (const auto source : prepared.selected_sources)
    hash_add(hash, static_cast<std::uint8_t>(source));
  for (const auto& participant : prepared.selected_participants) {
    hash_add(hash, static_cast<std::uint8_t>(participant.source));
    hash_add(hash, participant.generation);
    hash_add(hash, participant.commit_token);
  }
  for (const auto id : prepared.selected_intents) hash_add(hash, id);
  return hash;
}

bool objective_less(const Objective& lhs, const Objective& rhs) {
  return std::tie(lhs.missed_must_finish, lhs.cascade_risk,
                  lhs.deferred_important, lhs.total_displacement_damage,
                  lhs.deterministic_mask) <
         std::tie(rhs.missed_must_finish, rhs.cascade_risk,
                  rhs.deferred_important, rhs.total_displacement_damage,
                  rhs.deterministic_mask);
}

const ParticipantProposal* proposal_for(const PrepareRequest& request,
                                        Source source) {
  if (source == Source::WeedMinimumDamage)
    return request.weed ? &*request.weed : nullptr;
  return request.purchase ? &*request.purchase : nullptr;
}

std::vector<const Intent*> selected_intents(const PrepareRequest& request,
                                            unsigned mask) {
  std::vector<const Intent*> result;
  if ((mask & 1U) && request.weed)
    for (const auto& intent : request.weed->intents) result.push_back(&intent);
  if ((mask & 2U) && request.purchase)
    for (const auto& intent : request.purchase->intents)
      result.push_back(&intent);
  return result;
}

void edge(std::vector<ConflictEdge>& graph, const Intent& left,
          const Intent& right, ConflictReason reason,
          std::string diagnostic, bool excludes = true) {
  graph.push_back(
      {left.id, right.id, reason, excludes, std::move(diagnostic)});
}

bool contains(const std::vector<std::uint64_t>& values, std::uint64_t value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

bool validate_proposal(const PrepareRequest& request,
                       const ParticipantProposal& proposal,
                       std::vector<ConflictEdge>& graph) {
  if (proposal.generation == 0 || proposal.commit_token == 0 ||
      proposal.observation_hash != request.observation_hash ||
      !same(proposal.base_action, request.base_action) ||
      proposal.proposed_action.units.size() != request.base_action.units.size() ||
      proposal.proposed_action.market.size() < request.base_action.market.size())
    return false;
  std::set<int> unit_bound;
  std::set<int> market_bound;
  for (const auto& intent : proposal.intents) {
    if (intent.id == 0 || intent.source != proposal.source) return false;
    if (intent.domain == Domain::Unit) {
      if (intent.actor < 0 ||
          intent.actor >= static_cast<int>(request.base_action.units.size()) ||
          !same(request.base_action.units[intent.actor], intent.expected_base) ||
          !same(proposal.proposed_action.units[intent.actor], intent.replacement))
        return false;
      if (!unit_bound.insert(intent.actor).second) return false;
      if (is_move(request.base_action.units[intent.actor].op) &&
          !same(intent.expected_base, intent.replacement)) {
        if (proposal.source == Source::PurchaseFailure)
          edge(graph, intent, intent, ConflictReason::PurchaseMoveEdit,
               "purchase repair may not rewrite MOVE");
        else if (!intent.certified_weed_move_owner)
          edge(graph, intent, intent, ConflictReason::UncertifiedWeedMoveEdit,
               "weed MOVE edit lacks owner certificate");
      }
    } else {
      if (intent.market_slot < 0 ||
          intent.market_slot >=
              static_cast<int>(proposal.proposed_action.market.size()) ||
          !same(intent.expected_base,
                intent.market_slot <
                        static_cast<int>(request.base_action.market.size())
                    ? request.base_action.market[intent.market_slot]
                    : Action{}) ||
          !same(proposal.proposed_action.market[intent.market_slot],
                intent.replacement))
        return false;
      if (!market_bound.insert(intent.market_slot).second) return false;
    }
  }
  for (std::size_t actor = 0; actor < request.base_action.units.size(); ++actor)
    if (!same(request.base_action.units[actor],
              proposal.proposed_action.units[actor]) &&
        !unit_bound.contains(static_cast<int>(actor)))
      return false;
  const auto max_market = std::max(request.base_action.market.size(),
                                   proposal.proposed_action.market.size());
  for (std::size_t slot = 0; slot < max_market; ++slot) {
    const Action base = slot < request.base_action.market.size()
                            ? request.base_action.market[slot]
                            : Action{};
    const Action proposed = slot < proposal.proposed_action.market.size()
                                ? proposal.proposed_action.market[slot]
                                : Action{};
    if (!same(base, proposed) && !market_bound.contains(static_cast<int>(slot)))
      return false;
  }
  return true;
}

bool mask_feasible(const PrepareRequest& request, unsigned mask,
                   const std::vector<ConflictEdge>& graph) {
  const auto intents = selected_intents(request, mask);
  const auto is_selected = [&](std::uint64_t id) {
    return std::any_of(intents.begin(), intents.end(),
                       [id](const Intent* intent) { return intent->id == id; });
  };
  for (const auto& conflict : graph)
    if (conflict.excludes_joint_selection && is_selected(conflict.left) &&
        is_selected(conflict.right))
      return false;
  int cash = 0;
  int shed = 0;
  std::map<fastkag::Item, int> item_use;
  for (const auto* intent : intents) {
    cash += intent->cash_cost;
    shed += intent->shed_capacity_cost;
    item_use[intent->resource_item] += intent->resource_consumption;
    if (intent->deadline >= 0 && request.step > intent->deadline) return false;
    for (const auto dependency : intent->dependencies)
      if (!contains(request.satisfied_dependencies, dependency) &&
          !is_selected(dependency))
        return false;
  }
  if (cash > request.available_cash || shed > request.free_shed_capacity)
    return false;
  for (const auto& [item, used] : item_use) {
    if (item == fastkag::Item::NONE || used <= 0) continue;
    const auto found = request.available_items.find(item);
    if (found == request.available_items.end() || used > found->second)
      return false;
  }
  return true;
}

Objective score(const PrepareRequest& request, unsigned mask) {
  Objective result;
  result.deterministic_mask = mask;
  for (const auto source : {Source::WeedMinimumDamage,
                            Source::PurchaseFailure}) {
    const auto* proposal = proposal_for(request, source);
    if (!proposal) continue;
    const bool selected = (source == Source::WeedMinimumDamage)
                              ? (mask & 1U)
                              : (mask & 2U);
    for (const auto& intent : proposal->intents) {
      if (!selected) {
        result.missed_must_finish += intent.must_finish;
        result.cascade_risk += intent.cascade_risk;
        result.deferred_important += intent.important;
      } else {
        result.total_displacement_damage += intent.displacement_damage;
      }
    }
  }
  return result;
}

PlayerAction compose(const PrepareRequest& request, unsigned mask) {
  auto result = request.base_action;
  for (const auto source : {Source::WeedMinimumDamage,
                            Source::PurchaseFailure}) {
    const unsigned bit = source == Source::WeedMinimumDamage ? 1U : 2U;
    const auto* proposal = proposal_for(request, source);
    if (!(mask & bit) || !proposal) continue;
    for (const auto& intent : proposal->intents) {
      if (intent.domain == Domain::Unit) {
        result.units[intent.actor] = intent.replacement;
      } else {
        while (result.market.size() <=
               static_cast<std::size_t>(intent.market_slot))
          result.market.push_back({});
        result.market[intent.market_slot] = intent.replacement;
      }
    }
  }
  return result;
}

}  // namespace

Prepared Composer::prepare(const PrepareRequest& request) const {
  Prepared result;
  result.step = request.step;
  result.generation = generation_;
  result.observation_hash = request.observation_hash;
  result.base_action_hash = action_hash(request.base_action);
  result.final_action = request.base_action;
  if (request.step < 0 || request.observation_hash == 0) {
    result.status = PrepareStatus::InvalidInput;
    result.diagnostic = "invalid observation envelope";
    return result;
  }
  if (!request.enabled) {
    result.status = PrepareStatus::Disabled;
    result.final_action_hash = action_hash(result.final_action);
    result.binding_hash = binding_hash(result);
    result.diagnostic = "default-off exact base action";
    return result;
  }
  if ((request.weed && !validate_proposal(request, *request.weed,
                                          result.conflict_graph)) ||
      (request.purchase && !validate_proposal(request, *request.purchase,
                                              result.conflict_graph))) {
    result.status = PrepareStatus::InvalidInput;
    result.diagnostic = "participant diff is not fully typed/bound";
    return result;
  }
  std::set<std::uint64_t> global_intent_ids;
  const auto insert_ids = [&](const ParticipantProposal& proposal) {
    return std::all_of(proposal.intents.begin(), proposal.intents.end(),
                       [&](const Intent& intent) {
                         return global_intent_ids.insert(intent.id).second;
                       });
  };
  if ((request.weed && !insert_ids(*request.weed)) ||
      (request.purchase && !insert_ids(*request.purchase))) {
    result.status = PrepareStatus::InvalidInput;
    result.diagnostic = "intent ids are not globally unique";
    return result;
  }
  std::vector<const Intent*> all;
  if (request.weed)
    for (const auto& intent : request.weed->intents) all.push_back(&intent);
  if (request.purchase)
    for (const auto& intent : request.purchase->intents) all.push_back(&intent);
  for (std::size_t left = 0; left < all.size(); ++left) {
    const auto& a = *all[left];
    for (const auto dependency : a.dependencies) {
      const auto dependency_intent = std::find_if(
          all.begin(), all.end(), [dependency](const Intent* candidate) {
            return candidate->id == dependency;
          });
      if (!contains(request.satisfied_dependencies, dependency)) {
        if (dependency_intent == all.end())
          edge(result.conflict_graph, a, a, ConflictReason::Dependency,
               "dependency is neither satisfied nor proposed");
        else
          edge(result.conflict_graph, a, **dependency_intent,
               ConflictReason::Dependency,
               "dependency requires joint selection", false);
      }
    }
    if (a.deadline >= 0 && request.step > a.deadline)
      edge(result.conflict_graph, a, a, ConflictReason::Deadline,
           "intent deadline expired");
    if (a.cash_cost > request.available_cash)
      edge(result.conflict_graph, a, a, ConflictReason::Cash,
           "intent exceeds current cash");
    if (a.shed_capacity_cost > request.free_shed_capacity)
      edge(result.conflict_graph, a, a, ConflictReason::ShedCapacity,
           "intent exceeds current shed capacity");
    if (a.resource_item != fastkag::Item::NONE &&
        a.resource_consumption > 0) {
      const auto available = request.available_items.find(a.resource_item);
      if (available == request.available_items.end() ||
          a.resource_consumption > available->second)
        edge(result.conflict_graph, a, a, ConflictReason::ItemResource,
             "intent exceeds current item resource");
    }
    for (std::size_t right = left + 1; right < all.size(); ++right) {
      const auto& b = *all[right];
      if (a.source == b.source) continue;
      if (a.domain == Domain::Unit && b.domain == Domain::Unit &&
          a.actor == b.actor)
        edge(result.conflict_graph, a, b, ConflictReason::SameActor,
             "weed owned actor and purchase actor overlap");
      if (a.domain == Domain::Market && b.domain == Domain::Market &&
          a.market_slot == b.market_slot)
        edge(result.conflict_graph, a, b, ConflictReason::SameMarketSlot,
             "market slot has two owners");
      if (a.resource_item != fastkag::Item::NONE &&
          a.resource_item == b.resource_item) {
        const auto available = request.available_items.find(a.resource_item);
        const int limit = available == request.available_items.end()
                              ? 0
                              : available->second;
        if (a.resource_consumption + b.resource_consumption > limit)
          edge(result.conflict_graph, a, b, ConflictReason::ItemResource,
               "same item resource overcommitted");
      }
      if (a.cash_cost + b.cash_cost > request.available_cash)
        edge(result.conflict_graph, a, b, ConflictReason::Cash,
             "joint cash overcommitted");
      if (a.shed_capacity_cost + b.shed_capacity_cost >
          request.free_shed_capacity)
        edge(result.conflict_graph, a, b, ConflictReason::ShedCapacity,
             "joint shed capacity overcommitted");
    }
  }

  bool found = false;
  Objective best;
  unsigned best_mask = 0;
  for (unsigned mask = 0; mask < 4; ++mask) {
    if ((mask & 1U) && !request.weed) continue;
    if ((mask & 2U) && !request.purchase) continue;
    if (!mask_feasible(request, mask, result.conflict_graph)) continue;
    const auto candidate = score(request, mask);
    if (!found || objective_less(candidate, best)) {
      found = true;
      best = candidate;
      best_mask = mask;
    }
  }
  if (!found) {
    result.status = PrepareStatus::NoFeasibleJointSet;
    result.diagnostic = "no feasible atomic participant subset";
    return result;
  }
  result.status = PrepareStatus::Accepted;
  result.objective = best;
  result.final_action = compose(request, best_mask);
  if (best_mask & 1U) {
    result.selected_sources.push_back(Source::WeedMinimumDamage);
    result.selected_participants.push_back(
        {Source::WeedMinimumDamage, request.weed->generation,
         request.weed->commit_token});
  }
  if (best_mask & 2U) {
    result.selected_sources.push_back(Source::PurchaseFailure);
    result.selected_participants.push_back(
        {Source::PurchaseFailure, request.purchase->generation,
         request.purchase->commit_token});
  }
  for (const auto* intent : selected_intents(request, best_mask))
    result.selected_intents.push_back(intent->id);
  result.final_action_hash = action_hash(result.final_action);
  result.binding_hash = binding_hash(result);
  result.diagnostic = "deterministic lexicographic atomic joint set";
  return result;
}

FinalizeStatus Composer::finalize(
    const Prepared& prepared, const PlayerAction& exact_selected_action,
    std::vector<ParticipantTransaction> participants) {
  const auto abort_all = [&] {
    for (auto& participant : participants)
      if (participant.abort) participant.abort();
  };
  if (prepared.generation < generation_) {
    abort_all();
    return FinalizeStatus::AlreadyFinalized;
  }
  if (prepared.generation != generation_ || prepared.step < last_step_ ||
      prepared.binding_hash != binding_hash(prepared)) {
    abort_all();
    return FinalizeStatus::Stale;
  }
  if (!same(prepared.final_action, exact_selected_action) ||
      action_hash(exact_selected_action) != prepared.final_action_hash) {
    abort_all();
    return FinalizeStatus::Tampered;
  }
  for (const auto& selected : prepared.selected_participants) {
    const auto count = std::count_if(
        participants.begin(), participants.end(),
        [&selected](const ParticipantTransaction& value) {
          return value.source == selected.source &&
                 value.generation == selected.generation &&
                 value.commit_token == selected.commit_token &&
                 value.preflight && value.commit;
        });
    if (count != 1) {
      abort_all();
      return FinalizeStatus::PartialCommit;
    }
  }
  const auto exactly_selected = [&](const ParticipantTransaction& participant) {
    return std::any_of(
        prepared.selected_participants.begin(),
        prepared.selected_participants.end(),
        [&](const ParticipantSelection& selected) {
          return participant.source == selected.source &&
                 participant.generation == selected.generation &&
                 participant.commit_token == selected.commit_token;
        });
  };
  for (auto& participant : participants) {
    if (!exactly_selected(participant)) continue;
    bool accepted = false;
    try {
      accepted = participant.preflight();
    } catch (...) {
      accepted = false;
    }
    if (!accepted) {
      abort_all();
      return FinalizeStatus::ParticipantRejected;
    }
  }
  for (auto& participant : participants)
    if (exactly_selected(participant))
      participant.commit();
  for (auto& participant : participants)
    if (!exactly_selected(participant) && participant.abort)
      participant.abort();
  last_step_ = prepared.step;
  ++generation_;
  return FinalizeStatus::Selected;
}

std::uint64_t action_hash(const PlayerAction& action) noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_add(hash, action.units.size());
  for (const auto value : action.units) {
    hash_add(hash, static_cast<std::uint8_t>(value.op));
    hash_add(hash, static_cast<std::uint8_t>(value.item));
    hash_add(hash, value.quantity);
  }
  hash_add(hash, action.market.size());
  for (const auto value : action.market) {
    hash_add(hash, static_cast<std::uint8_t>(value.op));
    hash_add(hash, static_cast<std::uint8_t>(value.item));
    hash_add(hash, value.quantity);
  }
  return hash;
}

const char* conflict_reason_name(ConflictReason reason) noexcept {
  switch (reason) {
    case ConflictReason::SameActor: return "same_actor";
    case ConflictReason::SameMarketSlot: return "same_market_slot";
    case ConflictReason::ItemResource: return "item_resource";
    case ConflictReason::Cash: return "cash";
    case ConflictReason::ShedCapacity: return "shed_capacity";
    case ConflictReason::Dependency: return "dependency";
    case ConflictReason::Deadline: return "deadline";
    case ConflictReason::PurchaseMoveEdit: return "purchase_move_edit";
    case ConflictReason::UncertifiedWeedMoveEdit: return "uncertified_weed_move_edit";
    case ConflictReason::UnboundActionDiff: return "unbound_action_diff";
  }
  return "unknown";
}

const char* prepare_status_name(PrepareStatus status) noexcept {
  switch (status) {
    case PrepareStatus::Disabled: return "disabled";
    case PrepareStatus::Accepted: return "accepted";
    case PrepareStatus::InvalidInput: return "invalid_input";
    case PrepareStatus::NoFeasibleJointSet: return "no_feasible_joint_set";
  }
  return "unknown";
}

const char* finalize_status_name(FinalizeStatus status) noexcept {
  switch (status) {
    case FinalizeStatus::Selected: return "selected";
    case FinalizeStatus::Stale: return "stale";
    case FinalizeStatus::Tampered: return "tampered";
    case FinalizeStatus::PartialCommit: return "partial_commit";
    case FinalizeStatus::ParticipantRejected: return "participant_rejected";
    case FinalizeStatus::AlreadyFinalized: return "already_finalized";
  }
  return "unknown";
}

}  // namespace g001::repair_composer_arbiter
