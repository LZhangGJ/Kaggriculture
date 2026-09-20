#include "repair_debt_scheduler.hpp"
#include "constraint_relaxation_drop_planner.hpp"

#include <algorithm>
#include <limits>
#include <tuple>

namespace repair_debt {
namespace {

void mix(std::uint64_t& h, std::uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
}

std::uint64_t action_hash(const Action& a) {
  std::uint64_t h = 0xd1b54a32d192ed03ULL;
  mix(h, static_cast<unsigned>(a.kind));
  mix(h, static_cast<unsigned>(a.actor));
  mix(h, static_cast<unsigned>(a.direction));
  mix(h, a.move_identity);
  mix(h, a.debt_id);
  mix(h, static_cast<unsigned>(a.production.op));
  mix(h, static_cast<unsigned>(a.production.item));
  mix(h, static_cast<unsigned>(a.production.quantity));
  mix(h, static_cast<unsigned>(a.production.tile_x));
  mix(h, static_cast<unsigned>(a.production.tile_y));
  return h;
}

bool resources_available(const DebtNode& n, const TickInput& in) {
  if (in.resources.observation_hash != observation_hash(in.observation) ||
      in.resources.content_hash != resource_certificate_hash(in.resources)) return false;
  for (const auto& requirement : n.resources) {
    const auto it = in.resources.holdings.find(requirement.key);
    if (it == in.resources.holdings.end() || it->second < requirement.quantity)
      return false;
  }
  return true;
}

bool target_exists(const DebtNode& n, const Observation& o) {
  return !n.target_must_exist ||
         o.existing_targets.contains({n.identity.tile_x, n.identity.tile_y});
}

bool valid_resource_key(const ResourceKey& key, int actor) {
  switch (key.scope) {
    case ResourceScope::ActorCarried:
      return key.owner == actor && key.item >= 0 && key.item <= 11;
    case ResourceScope::FarmSeed:
      return key.owner >= 0 && key.item >= 0 && key.item <= 4;
    case ResourceScope::Shed:
      return key.owner >= 0 && key.item >= 0 && key.item <= 11;
    case ResourceScope::Cash:
    case ResourceScope::ShedCapacity:
    case ResourceScope::MarketSlot:
      return key.owner >= 0 && key.item == -1;
  }
  return false;
}

}  // namespace

std::uint64_t debt_authorization_hash(const DebtNode& n) noexcept {
  std::uint64_t h = 0x243f6a8885a308d3ULL;
  mix(h, n.identity.id); mix(h, n.identity.origin_key);
  mix(h, static_cast<unsigned>(n.identity.actor));
  mix(h, static_cast<unsigned>(n.identity.goal));
  mix(h, static_cast<unsigned>(n.identity.item));
  mix(h, static_cast<unsigned>(n.identity.tile_x));
  mix(h, static_cast<unsigned>(n.identity.tile_y));
  mix(h, n.identity.dependency_set_hash);
  mix(h, static_cast<unsigned>(n.required_action.op));
  mix(h, static_cast<unsigned>(n.required_action.item));
  mix(h, static_cast<unsigned>(n.required_action.quantity));
  mix(h, static_cast<unsigned>(n.required_action.tile_x));
  mix(h, static_cast<unsigned>(n.required_action.tile_y));
  for (auto d : n.dependencies) mix(h, d);
  for (const auto& requirement : n.resources) {
    mix(h, static_cast<unsigned>(requirement.key.scope));
    mix(h, static_cast<unsigned>(requirement.key.owner));
    mix(h, static_cast<unsigned>(requirement.key.item));
    mix(h, static_cast<unsigned>(requirement.quantity));
  }
  for (const auto& produced : n.produces) {
    mix(h, static_cast<unsigned>(produced.key.scope));
    mix(h, static_cast<unsigned>(produced.key.owner));
    mix(h, static_cast<unsigned>(produced.key.item));
    mix(h, static_cast<unsigned>(produced.quantity));
  }
  mix(h, static_cast<unsigned>(n.admitted_day));
  mix(h, static_cast<unsigned>(n.expires_day));
  mix(h, static_cast<unsigned>(n.priority));
  mix(h, static_cast<unsigned>(n.value));
  mix(h, static_cast<unsigned>(n.cascade_value));
  mix(h, static_cast<unsigned>(n.estimated_repair_damage));
  mix(h, n.target_must_exist);
  return h;
}

std::uint64_t ledger_hash(const DebtLedger& l) noexcept {
  std::uint64_t h = 0x13198a2e03707344ULL;
  mix(h, l.registry_generation);
  for (const auto& [id, r] : l.records) {
    mix(h, id); mix(h, r.node.authorization_hash);
    mix(h, static_cast<unsigned>(r.state)); mix(h, r.rebases);
    mix(h, r.attempts); mix(h, static_cast<unsigned>(r.completed_day));
    mix(h, r.last_receipt_hash);
    mix(h, static_cast<unsigned>(r.drop_reason));
    mix(h, static_cast<unsigned>(r.estimated_drop_damage));
    for (auto child : r.cancelled_descendants) mix(h, child);
    mix(h, r.physical_evidence_hash);
  }
  return h;
}

std::uint64_t observation_hash(const Observation& o) noexcept {
  std::uint64_t h = 0xa4093822299f31d0ULL;
  mix(h, static_cast<unsigned>(o.day)); mix(h, static_cast<unsigned>(o.hour));
  mix(h, static_cast<unsigned>(o.actor)); mix(h, o.terminal);
  mix(h, o.physical_hash);
  for (auto [x, y] : o.existing_targets) {
    mix(h, static_cast<unsigned>(x)); mix(h, static_cast<unsigned>(y));
  }
  for (auto actor : o.existing_actors) mix(h, static_cast<unsigned>(actor));
  return h;
}

std::uint64_t move_set_hash(const std::vector<MoveToken>& moves) noexcept {
  std::uint64_t h = 0x082efa98ec4e6c89ULL;
  for (const auto& m : moves) {
    mix(h, m.identity); mix(h, static_cast<unsigned>(m.day));
    mix(h, static_cast<unsigned>(m.actor)); mix(h, static_cast<unsigned>(m.ordinal));
    mix(h, static_cast<unsigned>(m.direction));
  }
  return h;
}

std::uint64_t dependency_set_hash(const std::vector<std::uint64_t>& input) noexcept {
  auto deps = input;
  std::ranges::sort(deps);
  std::uint64_t h = 0x452821e638d01377ULL;
  for (auto id : deps) mix(h, id);
  return h;
}

std::uint64_t resource_certificate_hash(const ResourceCertificate& c) noexcept {
  std::uint64_t h = 0xbe5466cf34e90c6cULL;
  mix(h, c.observation_hash); mix(h, c.issuer_generation);
  for (const auto& [key, amount] : c.holdings) {
    mix(h, static_cast<unsigned>(key.scope)); mix(h, static_cast<unsigned>(key.owner));
    mix(h, static_cast<unsigned>(key.item)); mix(h, static_cast<unsigned>(amount));
  }
  for (const auto& [key, day] : c.certified_available_day) {
    mix(h, static_cast<unsigned>(key.scope)); mix(h, static_cast<unsigned>(key.owner));
    mix(h, static_cast<unsigned>(key.item)); mix(h, static_cast<unsigned>(day));
  }
  mix(h, c.physical_evidence_hash);
  return h;
}

RepairDebtScheduler::RepairDebtScheduler(DebtLedger ledger) : ledger_(std::move(ledger)) {
  if (ledger_.registry_generation == 0) failure_ = Reject::InvalidLedger;
  if (ledger_.content_hash == 0) ledger_.content_hash = ledger_hash(ledger_);
}

void RepairDebtScheduler::close_terminal() {
  for (auto& [_, r] : ledger_.records) {
    if (r.state == DebtState::Active || r.state == DebtState::Deferred) {
      r.state = DebtState::ExpiredTerminal;
      r.drop_reason = DropReason::Terminal;
      ++metrics_.expired_terminal;
    }
  }
  ledger_.content_hash = ledger_hash(ledger_);
}

void RepairDebtScheduler::drop_subgraph(std::uint64_t root,
                                        DebtState disposition,
                                        DropReason reason, int damage,
                                        std::uint64_t evidence) {
  std::vector<std::uint64_t> queue{root};
  std::set<std::uint64_t> seen;
  while (!queue.empty()) {
    const auto id = queue.back(); queue.pop_back();
    if (!seen.insert(id).second) continue;
    auto it = ledger_.records.find(id);
    if (it == ledger_.records.end() ||
        (it->second.state != DebtState::Active &&
         it->second.state != DebtState::Deferred)) continue;
    it->second.state = disposition;
    it->second.drop_reason = id == root ? reason : DropReason::AncestorDropped;
    it->second.estimated_drop_damage = damage;
    it->second.physical_evidence_hash = evidence;
    if (disposition == DebtState::DroppedInfeasible) ++metrics_.dropped_infeasible;
    else ++metrics_.dropped_dominated;
    for (auto& [child_id, child] : ledger_.records) {
      if ((child.state == DebtState::Active || child.state == DebtState::Deferred) &&
          std::ranges::find(child.node.dependencies, id) != child.node.dependencies.end()) {
        it->second.cancelled_descendants.push_back(child_id);
        queue.push_back(child_id);
      }
    }
  }
}

void RepairDebtScheduler::reduce_soft_constraints(const TickInput& in) {
  const auto& o = in.observation;
  // Physical/current-observation impossibility is resolved before scheduling.
  std::vector<std::tuple<std::uint64_t, DropReason, int>> impossible;
  for (const auto& [id, r] : ledger_.records) {
    if (r.state != DebtState::Active && r.state != DebtState::Deferred) continue;
    DropReason why = DropReason::None;
    if (!o.existing_actors.contains(r.node.identity.actor)) why = DropReason::ActorMissing;
    else if (!target_exists(r.node, o)) why = DropReason::TargetMissing;
    else if (o.day > r.node.expires_day) why = DropReason::DeadlineElapsed;
    else {
      for (const auto& requirement : r.node.resources) {
        const auto forecast = in.resources.certified_available_day.find(requirement.key);
        if (forecast != in.resources.certified_available_day.end() &&
            forecast->second > r.node.expires_day) {
          why = DropReason::ResourceImpossible;
          break;
        }
      }
    }
    if (why != DropReason::None)
      impossible.emplace_back(id, why, r.node.value + r.node.cascade_value);
  }
  for (auto [id, why, loss] : impossible)
    drop_subgraph(id, DebtState::DroppedInfeasible, why, loss, o.physical_hash);

  // If repair itself costs more than abandoning its entire typed subtree, the
  // node is dominated. This is a soft-policy decision, never a hard receipt.
  std::vector<std::pair<std::uint64_t, int>> dominated;
  for (const auto& [id, r] : ledger_.records) {
    if ((r.state == DebtState::Active || r.state == DebtState::Deferred) &&
        r.node.estimated_repair_damage > r.node.value + r.node.cascade_value)
      dominated.emplace_back(id, r.node.value + r.node.cascade_value);
  }
  for (auto [id, loss] : dominated)
    drop_subgraph(id, DebtState::DroppedDominated, DropReason::DominatedDamage,
                  loss, o.physical_hash);

  namespace cr = g001::constraint_relaxation;
  cr::Request request;
  request.current_step = o.day * 24 + o.hour;
  request.horizon_end_step = o.day * 24 + 23;
  request.observation_hash = observation_hash(o);
  request.final_owner_generation = ledger_.registry_generation;
  request.maximum_carry_days = 5;
  request.maximum_deferred_ledger = 64;
  request.exact_node_limit = 16;
  request.maximum_search_states = 1'000'000;

  auto resource_key = [](const ResourceKey& key) {
    return std::to_string(static_cast<unsigned>(key.scope)) + ":" +
           std::to_string(key.owner) + ":" + std::to_string(key.item);
  };
  std::map<ResourceKey, int> resource_index;
  for (const auto& [key, amount] : in.resources.holdings) {
    cr::ResourceCertificate certificate;
    certificate.key = resource_key(key);
    certificate.available = amount;
    certificate.evidence_hash = in.resources.physical_evidence_hash;
    resource_index[key] = static_cast<int>(request.resources.size());
    request.resources.push_back(std::move(certificate));
  }
  // Claims require a certificate even when current availability is zero.
  for (const auto& [_, record] : ledger_.records) {
    for (const auto& requirement : record.node.resources) {
      if (!resource_index.contains(requirement.key)) {
        cr::ResourceCertificate certificate;
        certificate.key = resource_key(requirement.key);
        certificate.available = 0;
        certificate.evidence_hash = in.resources.physical_evidence_hash;
        resource_index[requirement.key] = static_cast<int>(request.resources.size());
        request.resources.push_back(std::move(certificate));
      }
    }
    for (const auto& produced : record.node.produces) {
      if (!resource_index.contains(produced.key)) {
        cr::ResourceCertificate certificate;
        certificate.key = resource_key(produced.key);
        certificate.available = 0;
        certificate.evidence_hash = in.resources.physical_evidence_hash;
        resource_index[produced.key] = static_cast<int>(request.resources.size());
        request.resources.push_back(std::move(certificate));
      }
    }
  }

  std::set<int> actors = o.existing_actors;
  for (const auto& m : active_moves_) actors.insert(m.actor);
  for (const auto& [_, r] : ledger_.records) actors.insert(r.node.identity.actor);
  std::map<int, std::vector<const MoveToken*>> remaining_by_actor;
  for (const auto& m : active_moves_)
    if (!emitted_moves_.contains(m.identity)) remaining_by_actor[m.actor].push_back(&m);
  for (auto& [_, tokens] : remaining_by_actor)
    std::ranges::sort(tokens, {}, [](const MoveToken* m) { return m->ordinal; });
  for (int step = request.current_step; step <= request.horizon_end_step; ++step) {
    for (int actor : actors) {
      cr::Slot slot;
      slot.step = step; slot.actor = actor;
      slot.final_owner_generation = request.final_owner_generation;
      const auto& tokens = remaining_by_actor[actor];
      const int offset = request.horizon_end_step - step;
      if (offset < static_cast<int>(tokens.size())) {
        const auto* token = tokens[tokens.size() - 1U - static_cast<std::size_t>(offset)];
        slot.move_committed = true;
        slot.move_source_step = token->ordinal;
        slot.move_token = token->identity;
      }
      request.remaining_slots.push_back(slot);
    }
  }

  std::set<std::uint64_t> included;
  for (const auto& [id, r] : ledger_.records)
    if (r.state == DebtState::Active || r.state == DebtState::Deferred) included.insert(id);
  for (const auto& [_, r] : ledger_.records) {
    if (r.state != DebtState::Active && r.state != DebtState::Deferred) continue;
    cr::Obligation obligation;
    obligation.id = r.node.identity.id;
    obligation.action.op = static_cast<fastkag::Op>(r.node.required_action.op);
    obligation.action.item = static_cast<fastkag::Item>(r.node.required_action.item);
    obligation.action.quantity = r.node.required_action.quantity;
    obligation.tile = {static_cast<std::int16_t>(r.node.required_action.tile_x),
                       static_cast<std::int16_t>(r.node.required_action.tile_y)};
    obligation.actor = r.node.identity.actor;
    obligation.earliest_step = request.current_step;
    obligation.deadline = r.node.expires_day * 24 + 23;
    for (auto dep : r.node.dependencies)
      if (included.contains(dep)) obligation.dependencies.push_back(dep);
    for (const auto& requirement : r.node.resources)
      obligation.resources.push_back({resource_key(requirement.key),
                                      requirement.quantity});
    for (const auto& produced : r.node.produces)
      obligation.produces.push_back({resource_key(produced.key),
                                     produced.quantity});
    obligation.value_loss = std::max(0, r.node.value);
    obligation.cascade_damage = std::max(0, r.node.cascade_value);
    obligation.defer_damage = std::max(0, r.node.priority);
    obligation.allow_defer = true;
    obligation.carry_days = r.rebases;
    obligation.lineage_hash = r.node.authorization_hash;
    request.obligations.push_back(std::move(obligation));
  }
  current_assignments_.clear(); current_move_actors_.clear();
  const auto plan = cr::plan(request);
  if (!plan.accepted() || !cr::verify(request, plan).valid) {
    // Planner rejection is not permission to invent a partial schedule. The
    // enclosing prepare will observe an empty assignment and keep soft debt
    // explicit; hard MOVE slots remain locally enforced.
    return;
  }
  for (const auto& slot : request.remaining_slots)
    if (slot.step == request.current_step && slot.move_committed)
      current_move_actors_.insert(slot.actor);
  for (const auto& decision : plan.decisions) {
    auto it = ledger_.records.find(decision.obligation_id);
    if (it == ledger_.records.end()) continue;
    if (decision.disposition == cr::Disposition::Kept) {
      if (decision.assignment && decision.assignment->step == request.current_step)
        current_assignments_[decision.assignment->actor] = decision.obligation_id;
      if (it->second.state == DebtState::Deferred) it->second.state = DebtState::Active;
    } else if (decision.disposition == cr::Disposition::Deferred) {
      if (it->second.state != DebtState::Deferred) ++metrics_.deferred;
      it->second.state = DebtState::Deferred;
    } else {
      const auto disposition = decision.disposition == cr::Disposition::DroppedInfeasible
                                   ? DebtState::DroppedInfeasible
                                   : DebtState::DroppedDominated;
      const auto reason = decision.drop &&
                                  decision.drop->reason == cr::DropReason::DependencyDropped
                              ? DropReason::AncestorDropped
                              : (disposition == DebtState::DroppedDominated
                                     ? DropReason::DominatedDamage
                                     : DropReason::SlotCapacity);
      const int damage = decision.drop ? decision.drop->value_loss +
                                             decision.drop->cascade_damage : 0;
      drop_subgraph(decision.obligation_id, disposition, reason, damage,
                    decision.drop ? decision.drop->content_hash : o.physical_hash);
    }
  }
}

Reject RepairDebtScheduler::validate_and_roll(const TickInput& in) {
  const auto& o = in.observation;
  if (o.day < 0 || o.hour < 0 || o.hour >= 24 || o.actor < 0 ||
      in.resources.observation_hash != observation_hash(o) ||
      in.resources.issuer_generation == 0 ||
      in.resources.physical_evidence_hash == 0 ||
      in.resources.content_hash != resource_certificate_hash(in.resources))
    return Reject::InvalidObservation;
  for (const auto& [key, amount] : in.resources.holdings)
    if (!valid_resource_key(key, key.owner) || amount < 0)
      return Reject::InvalidObservation;
  for (const auto& [key, day] : in.resources.certified_available_day)
    if (!valid_resource_key(key, key.owner) || day < o.day)
      return Reject::InvalidObservation;
  if (ledger_.content_hash != ledger_hash(ledger_)) return Reject::InvalidLedger;

  for (const auto& [id, r] : ledger_.records) {
    if (id != r.node.identity.id ||
        r.node.authorization_hash != debt_authorization_hash(r.node) ||
        r.node.identity.dependency_set_hash != dependency_set_hash(r.node.dependencies) ||
        r.node.required_action.op < 0 || r.node.required_action.op > 17 ||
        r.node.required_action.item < -1 || r.node.required_action.item > 11 ||
        r.node.required_action.quantity <= 0 ||
        r.node.required_action.tile_x != r.node.identity.tile_x ||
        r.node.required_action.tile_y != r.node.identity.tile_y)
      return Reject::InvalidLedger;
    for (const auto& requirement : r.node.resources)
      if (!valid_resource_key(requirement.key, r.node.identity.actor) ||
          requirement.quantity <= 0)
        return Reject::InvalidLedger;
    for (const auto& produced : r.node.produces)
      if (!valid_resource_key(produced.key, r.node.identity.actor) ||
          produced.quantity <= 0)
        return Reject::InvalidLedger;
    for (auto dependency : r.node.dependencies)
      if (!ledger_.records.contains(dependency)) return Reject::InvalidLedger;
  }
  // Complete DAG validation: no self edge, no cycles, and each child deadline
  // leaves at least its complete parent chain's certified horizon.
  std::map<std::uint64_t, int> color;
  auto visit = [&](auto&& self, std::uint64_t id) -> bool {
    if (color[id] == 1) return false;
    if (color[id] == 2) return true;
    color[id] = 1;
    const auto& child = ledger_.records.at(id).node;
    for (auto dep : child.dependencies) {
      if (dep == id) return false;
      const auto& parent = ledger_.records.at(dep).node;
      if (parent.expires_day > child.expires_day ||
          parent.admitted_day > child.expires_day || !self(self, dep)) return false;
    }
    color[id] = 2;
    return true;
  };
  for (const auto& [id, _] : ledger_.records)
    if (!visit(visit, id)) return Reject::InvalidLedger;

  std::set<std::uint64_t> unique_identities;
  std::set<std::tuple<int, int, int>> unique_ordinals;
  std::map<int, std::set<int>> actor_ordinals;
  for (const auto& m : in.immutable_day_moves) {
    if (m.identity == 0 || m.day != o.day || m.actor < 0 || m.ordinal < 0 ||
        m.direction < 0 || m.direction > 3 ||
        !unique_identities.insert(m.identity).second ||
        !unique_ordinals.emplace(m.day, m.actor, m.ordinal).second)
      return Reject::InvalidMoveTokens;
    actor_ordinals[m.actor].insert(m.ordinal);
  }
  for (const auto& [_, ordinals] : actor_ordinals) {
    int expected = 0;
    for (int ordinal : ordinals)
      if (ordinal != expected++) return Reject::InvalidMoveTokens;
  }
  const auto move_hash = move_set_hash(in.immutable_day_moves);
  if (active_day_ < 0) {
    active_day_ = o.day;
    active_move_hash_ = move_hash;
    active_moves_ = in.immutable_day_moves;
  } else if (o.day == active_day_) {
    if (move_hash != active_move_hash_) return Reject::InvalidMoveTokens;
  } else {
    if (o.day < active_day_) return Reject::InvalidObservation;
    if (emitted_moves_.size() != active_moves_.size()) {
      metrics_.move_drop += static_cast<int>(active_moves_.size() - emitted_moves_.size());
      return Reject::MoveDebtAtMidnight;
    }
    for (auto& [_, r] : ledger_.records) {
      if (r.state == DebtState::Active || r.state == DebtState::Deferred) {
        r.rebases += o.day - active_day_;
        metrics_.day_rebases += o.day - active_day_;
      }
    }
    active_day_ = o.day;
    active_move_hash_ = move_hash;
    active_moves_ = in.immutable_day_moves;
    emitted_moves_.clear();
  }

  for (const auto& n : in.production_dag) {
    if (n.authorization_hash != debt_authorization_hash(n) ||
        n.identity.dependency_set_hash != dependency_set_hash(n.dependencies) ||
        n.required_action.op < 0 || n.required_action.op > 17 ||
        n.required_action.quantity <= 0 ||
        n.required_action.tile_x != n.identity.tile_x ||
        n.required_action.tile_y != n.identity.tile_y)
      return Reject::InvalidLedger;
    for (const auto& requirement : n.resources)
      if (!valid_resource_key(requirement.key, n.identity.actor) ||
          requirement.quantity <= 0)
        return Reject::InvalidLedger;
    for (const auto& produced : n.produces)
      if (!valid_resource_key(produced.key, n.identity.actor) ||
          produced.quantity <= 0)
        return Reject::InvalidLedger;
    DebtRecord record; record.node = n;
    auto [it, inserted] = ledger_.records.emplace(n.identity.id, std::move(record));
    if (!inserted && it->second.node.authorization_hash != n.authorization_hash)
      return Reject::InvalidLedger;
  }
  // Revalidate the combined persistent+current DAG, including obligations that
  // arrived in this observation.
  std::map<std::uint64_t, int> combined_color;
  auto combined_visit = [&](auto&& self, std::uint64_t id) -> bool {
    if (combined_color[id] == 1) return false;
    if (combined_color[id] == 2) return true;
    combined_color[id] = 1;
    const auto& child = ledger_.records.at(id).node;
    for (auto dep : child.dependencies) {
      auto found = ledger_.records.find(dep);
      if (dep == id || found == ledger_.records.end() ||
          found->second.node.expires_day > child.expires_day ||
          !self(self, dep)) return false;
    }
    combined_color[id] = 2; return true;
  };
  for (const auto& [id, _] : ledger_.records)
    if (!combined_visit(combined_visit, id)) return Reject::InvalidLedger;

  reduce_soft_constraints(in);
  for (auto& [_, r] : ledger_.records) {
    if (r.state != DebtState::Active && r.state != DebtState::Deferred) continue;
    const bool available = resources_available(r.node, in);
    if (!available && r.state == DebtState::Active) {
      r.state = DebtState::Deferred; ++metrics_.deferred;
    } else if (available && r.state == DebtState::Deferred) {
      r.state = DebtState::Active;
    }
  }
  if (o.terminal) close_terminal();
  ledger_.content_hash = ledger_hash(ledger_);
  return Reject::None;
}

Action RepairDebtScheduler::choose(const TickInput& in) const {
  const int actor = in.observation.actor;
  std::vector<const MoveToken*> remaining;
  for (const auto& m : active_moves_)
    if (m.actor == actor && !emitted_moves_.contains(m.identity)) remaining.push_back(&m);
  std::ranges::sort(remaining, {}, [](const MoveToken* m) { return m->ordinal; });
  if (!remaining.empty() && current_move_actors_.contains(actor)) {
    const auto& m = *remaining.front();
    return {ActionKind::Move, actor, m.direction, m.identity, 0, {}};
  }

  if (const auto assignment = current_assignments_.find(actor);
      assignment != current_assignments_.end())
    return {ActionKind::Obligation, actor, 0, 0, assignment->second,
            ledger_.records.at(assignment->second).node.required_action};
  return {ActionKind::Pass, actor, 0, 0, 0, {}};
}

Proposal RepairDebtScheduler::prepare(const TickInput& in) {
  ++metrics_.iterations;
  if (failure_ != Reject::None) {
    Proposal p; p.reject = Reject::FailStopped; return p;
  }
  if (pending_) { Proposal p; p.reject = Reject::PendingProposal; return p; }
  rollback_ledger_ = ledger_;
  rollback_metrics_ = metrics_;
  rollback_active_day_ = active_day_;
  rollback_move_hash_ = active_move_hash_;
  rollback_moves_ = active_moves_;
  rollback_emitted_moves_ = emitted_moves_;
  if (const auto reject = validate_and_roll(in); reject != Reject::None) {
    restore_staged(); failure_ = reject; ++metrics_.fail_stops;
    Proposal p; p.reject = reject; return p;
  }
  Proposal p;
  p.accepted = true;
  if (in.observation.terminal) {
    p.action.kind = ActionKind::Pass; p.action.actor = in.observation.actor;
  } else {
    p.action = choose(in);
  }
  p.observation_hash = observation_hash(in.observation);
  p.ledger_before_hash = ledger_.content_hash;
  p.day_move_set_hash = active_move_hash_;
  p.binding_hash = p.observation_hash;
  mix(p.binding_hash, p.ledger_before_hash); mix(p.binding_hash, p.day_move_set_hash);
  mix(p.binding_hash, action_hash(p.action));
  pending_ = p;
  return p;
}

void RepairDebtScheduler::restore_staged() {
  if (!rollback_ledger_) return;
  ledger_ = std::move(*rollback_ledger_);
  metrics_ = rollback_metrics_;
  active_day_ = rollback_active_day_;
  active_move_hash_ = rollback_move_hash_;
  active_moves_ = std::move(rollback_moves_);
  emitted_moves_ = std::move(rollback_emitted_moves_);
  current_assignments_.clear(); current_move_actors_.clear();
  rollback_ledger_.reset();
}

FinalizeResult RepairDebtScheduler::finalize(const TickInput& in, const Proposal& p,
                                             const Action& final_action,
                                             const PhysicalReceipt& receipt) {
  auto reject = [&](Reject r) {
    restore_staged(); failure_ = r; pending_.reset(); ++metrics_.fail_stops;
    FinalizeResult result; result.reject = r; return result;
  };
  if (failure_ != Reject::None) {
    FinalizeResult result; result.reject = Reject::FailStopped; return result;
  }
  if (!pending_ || !p.accepted || pending_->binding_hash != p.binding_hash ||
      observation_hash(in.observation) != p.observation_hash ||
      ledger_.content_hash != p.ledger_before_hash)
    return reject(Reject::StaleProposal);
  if (!(final_action == p.action) || !(receipt.committed_action == final_action) ||
      receipt.proposal_binding_hash != p.binding_hash)
    return reject(Reject::FinalActionMismatch);
  if (receipt.before_hash != in.observation.physical_hash ||
      (final_action.kind != ActionKind::Pass &&
       (!receipt.physically_applied || receipt.after_hash == receipt.before_hash)))
    return reject(Reject::ReceiptMismatch);

  FinalizeResult result; result.committed = true;
  if (final_action.kind == ActionKind::Move) {
    auto it = std::ranges::find(active_moves_, final_action.move_identity,
                                &MoveToken::identity);
    if (it == active_moves_.end() || it->actor != final_action.actor)
      return reject(Reject::ReceiptMismatch);
    int expected = std::numeric_limits<int>::max();
    for (const auto& m : active_moves_)
      if (m.actor == it->actor && !emitted_moves_.contains(m.identity))
        expected = std::min(expected, m.ordinal);
    if (emitted_moves_.contains(it->identity)) {
      ++metrics_.move_duplicate; return reject(Reject::ReceiptMismatch);
    }
    if (it->ordinal != expected) {
      ++metrics_.move_early; return reject(Reject::ReceiptMismatch);
    }
    emitted_moves_.insert(it->identity); ++metrics_.moves_emitted;
    result.emitted_move = it->identity;
  } else if (final_action.kind == ActionKind::Obligation) {
    auto it = ledger_.records.find(final_action.debt_id);
    if (it == ledger_.records.end() ||
        (it->second.state != DebtState::Active && it->second.state != DebtState::Deferred) ||
        !receipt.target_effect_observed)
      return reject(Reject::ReceiptMismatch);
    if (!(final_action.production == it->second.node.required_action))
      return reject(Reject::ReceiptMismatch);
    std::map<ResourceKey, int> expected_deltas;
    for (const auto& need : it->second.node.resources)
      expected_deltas[need.key] -= need.quantity;
    for (const auto& produced : it->second.node.produces)
      expected_deltas[produced.key] += produced.quantity;
    if (receipt.resource_deltas != expected_deltas)
      return reject(Reject::ReceiptMismatch);
    it->second.state = DebtState::Completed;
    it->second.completed_day = in.observation.day;
    it->second.last_receipt_hash = receipt.after_hash;
    ++it->second.attempts; ++metrics_.completed;
    result.completed_debt = it->first;
  }
  ledger_.content_hash = ledger_hash(ledger_);
  pending_.reset();
  rollback_ledger_.reset();
  return result;
}

const char* reject_name(Reject r) noexcept {
  switch (r) {
    case Reject::None: return "none"; case Reject::InvalidObservation: return "invalid_observation";
    case Reject::InvalidLedger: return "invalid_ledger"; case Reject::InvalidMoveTokens: return "invalid_move_tokens";
    case Reject::MoveDebtAtMidnight: return "move_debt_at_midnight"; case Reject::PendingProposal: return "pending_proposal";
    case Reject::StaleProposal: return "stale_proposal"; case Reject::FinalActionMismatch: return "final_action_mismatch";
    case Reject::ReceiptMismatch: return "receipt_mismatch"; case Reject::FailStopped: return "fail_stopped";
  }
  return "unknown";
}

const char* debt_state_name(DebtState s) noexcept {
  switch (s) {
    case DebtState::Active: return "active"; case DebtState::Completed: return "completed";
    case DebtState::Deferred: return "deferred";
    case DebtState::DroppedInfeasible: return "dropped_infeasible";
    case DebtState::DroppedDominated: return "dropped_dominated";
    case DebtState::ExpiredTerminal: return "expired_terminal";
  }
  return "unknown";
}

}  // namespace repair_debt
