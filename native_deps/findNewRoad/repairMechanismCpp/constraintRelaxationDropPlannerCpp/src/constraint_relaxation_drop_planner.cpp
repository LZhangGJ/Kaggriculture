#include "constraint_relaxation_drop_planner.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace g001::constraint_relaxation {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void mix(std::uint64_t& hash, std::uint64_t value) noexcept {
  for (unsigned int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8U)) & 0xffU;
    hash *= kFnvPrime;
  }
}

void mix(std::uint64_t& hash, const std::string& value) noexcept {
  for (const unsigned char byte : value) {
    hash ^= byte;
    hash *= kFnvPrime;
  }
  hash ^= 0xffU;
  hash *= kFnvPrime;
}

std::uint64_t as_u64(const std::int64_t value) noexcept {
  return static_cast<std::uint64_t>(value);
}

struct Validation {
  bool valid{};
  std::string diagnostic;
  std::vector<std::size_t> topo;
  std::unordered_map<std::uint64_t, std::size_t> by_id;
  std::unordered_map<std::string, std::size_t> resource_by_key;
};

Validation validate(const Request& request) {
  Validation result;
  if (request.current_step < 0 ||
      request.horizon_end_step < request.current_step ||
      request.turns_per_day <= 0 || request.observation_hash == 0 ||
      request.final_owner_generation == 0 || request.maximum_carry_days < 0 ||
      request.maximum_deferred_ledger < 0 || request.exact_node_limit == 0 ||
      request.maximum_search_states == 0) {
    result.diagnostic = "invalid request bounds or authority identity";
    return result;
  }

  for (std::size_t index = 0; index < request.resources.size(); ++index) {
    const auto& resource = request.resources[index];
    if (resource.key.empty() || resource.available < 0 ||
        resource.evidence_hash == 0 ||
        !result.resource_by_key.emplace(resource.key, index).second) {
      result.diagnostic = "invalid or duplicate resource certificate";
      return result;
    }
  }

  std::set<std::pair<int, int>> slot_keys;
  std::set<std::pair<int, int>> move_sources;
  std::unordered_set<std::uint64_t> move_tokens;
  for (const auto& slot : request.remaining_slots) {
    if (slot.step < request.current_step ||
        slot.step > request.horizon_end_step || slot.actor < 0 ||
        slot.final_owner_generation != request.final_owner_generation ||
        !slot_keys.emplace(slot.step, slot.actor).second) {
      result.diagnostic = "invalid, duplicate, or stale remaining slot";
      return result;
    }
    if (slot.move_committed) {
      if (slot.move_source_step < 0 || slot.move_token == 0) {
        result.diagnostic = "MOVE commitment lacks source identity";
        return result;
      }
      if (!move_sources.emplace(slot.actor, slot.move_source_step).second ||
          !move_tokens.insert(slot.move_token).second) {
        result.diagnostic = "duplicate MOVE source or token identity";
        return result;
      }
    } else if (slot.move_source_step != -1 || slot.move_token != 0) {
      result.diagnostic = "non-MOVE slot carries MOVE identity";
      return result;
    }
  }

  const int max_op = static_cast<int>(fastkag::Op::CARE);
  const int min_item = static_cast<int>(fastkag::Item::NONE);
  const int max_item = static_cast<int>(fastkag::Item::SHEEP);
  for (std::size_t index = 0; index < request.obligations.size(); ++index) {
    const auto& obligation = request.obligations[index];
    const int op = static_cast<int>(obligation.action.op);
    const int item = static_cast<int>(obligation.action.item);
    if (obligation.id == 0 || obligation.actor < -1 ||
        obligation.earliest_step < 0 || obligation.deadline < 0 ||
        obligation.deadline < obligation.earliest_step ||
        obligation.action.quantity <= 0 || op < 0 || op > max_op ||
        item < min_item || item > max_item || obligation.value_loss < 0 ||
        obligation.cascade_damage < 0 || obligation.defer_damage < 0 ||
        obligation.carry_days < 0 || obligation.lineage_hash == 0 ||
        !result.by_id.emplace(obligation.id, index).second) {
      result.diagnostic = "invalid, duplicate, or non-unit typed obligation";
      return result;
    }
    std::unordered_set<std::string> local_resources;
    for (const auto& claim : obligation.resources) {
      if (claim.key.empty() || claim.amount <= 0 ||
          !local_resources.insert(claim.key).second ||
          !result.resource_by_key.contains(claim.key)) {
        result.diagnostic = "invalid, duplicate, or uncertified resource claim";
        return result;
      }
    }
    std::unordered_set<std::string> local_productions;
    for (const auto& production : obligation.produces) {
      if (production.key.empty() || production.amount <= 0 ||
          !local_productions.insert(production.key).second ||
          !result.resource_by_key.contains(production.key)) {
        result.diagnostic = "invalid, duplicate, or uncertified resource production";
        return result;
      }
    }
  }

  std::vector<int> indegree(request.obligations.size(), 0);
  std::vector<std::vector<std::size_t>> successors(request.obligations.size());
  for (std::size_t index = 0; index < request.obligations.size(); ++index) {
    std::unordered_set<std::uint64_t> local_dependencies;
    for (const auto dependency : request.obligations[index].dependencies) {
      const auto found = result.by_id.find(dependency);
      if (dependency == request.obligations[index].id ||
          found == result.by_id.end() ||
          !local_dependencies.insert(dependency).second) {
        result.diagnostic = "invalid, duplicate, or unknown dependency";
        return result;
      }
      ++indegree[index];
      successors[found->second].push_back(index);
    }
  }

  using Ready = std::pair<std::uint64_t, std::size_t>;
  std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> ready;
  for (std::size_t index = 0; index < indegree.size(); ++index) {
    if (indegree[index] == 0) {
      ready.emplace(request.obligations[index].id, index);
    }
  }
  while (!ready.empty()) {
    const auto index = ready.top().second;
    ready.pop();
    result.topo.push_back(index);
    for (const auto successor : successors[index]) {
      if (--indegree[successor] == 0) {
        ready.emplace(request.obligations[successor].id, successor);
      }
    }
  }
  if (result.topo.size() != request.obligations.size()) {
    result.diagnostic = "obligation dependency graph contains a cycle";
    return result;
  }
  result.valid = true;
  return result;
}

enum class WorkingDisposition : std::uint8_t { Unset, Kept, Deferred, Dropped };

struct SearchState {
  std::vector<WorkingDisposition> disposition;
  std::vector<int> assignment_slot;
  std::vector<bool> used_slot;
  Objective objective;
};

bool objective_less(const Objective& left, const Objective& right) {
  return std::tie(left.total_damage, left.dropped_count,
                  left.deferred_count, left.slot_perturbation,
                  left.deterministic_signature) <
         std::tie(right.total_damage, right.dropped_count,
                  right.deferred_count, right.slot_perturbation,
                  right.deterministic_signature);
}

std::vector<int> sorted_slots(const Request& request) {
  std::vector<int> slots(request.remaining_slots.size());
  std::iota(slots.begin(), slots.end(), 0);
  std::stable_sort(slots.begin(), slots.end(), [&](const int left,
                                                   const int right) {
    const auto& a = request.remaining_slots[static_cast<std::size_t>(left)];
    const auto& b = request.remaining_slots[static_cast<std::size_t>(right)];
    return std::tie(a.step, a.actor, left) < std::tie(b.step, b.actor, right);
  });
  return slots;
}

bool execution_before(const Slot& left, const Slot& right) {
  return std::tie(left.step, left.actor) < std::tie(right.step, right.actor);
}

bool dependencies_execute_before(const Request& request,
                                 const Validation& validation,
                                 const SearchState& state,
                                 const Obligation& obligation,
                                 const Slot& candidate) {
  for (const auto dependency : obligation.dependencies) {
    const auto dep_index = validation.by_id.at(dependency);
    if (state.disposition[dep_index] == WorkingDisposition::Kept) {
      const int slot_index = state.assignment_slot[dep_index];
      if (!execution_before(
              request.remaining_slots[static_cast<std::size_t>(slot_index)],
              candidate)) {
        return false;
      }
    }
  }
  return true;
}

bool resources_fit_at(const Request& request, const Validation& validation,
                      const SearchState& state,
                      const Obligation& obligation, const Slot& candidate) {
  std::vector<std::int64_t> balances;
  balances.reserve(request.resources.size());
  for (const auto& certificate : request.resources) {
    balances.push_back(certificate.available);
  }
  for (std::size_t index = 0; index < request.obligations.size(); ++index) {
    if (state.disposition[index] != WorkingDisposition::Kept) continue;
    const auto& assigned = request.remaining_slots[static_cast<std::size_t>(
        state.assignment_slot[index])];
    if (!execution_before(assigned, candidate)) continue;
    for (const auto& claim : request.obligations[index].resources) {
      balances[validation.resource_by_key.at(claim.key)] -= claim.amount;
    }
    for (const auto& production : request.obligations[index].produces) {
      balances[validation.resource_by_key.at(production.key)] +=
          production.amount;
    }
  }
  for (const auto& claim : obligation.resources) {
    if (balances[validation.resource_by_key.at(claim.key)] < claim.amount) {
      return false;
    }
  }
  return true;
}

bool all_resource_prefixes_feasible(const Request& request,
                                    const Validation& validation,
                                    const SearchState& state) {
  std::vector<std::size_t> kept;
  for (std::size_t index = 0; index < request.obligations.size(); ++index) {
    if (state.disposition[index] == WorkingDisposition::Kept) {
      kept.push_back(index);
    }
  }
  std::sort(kept.begin(), kept.end(), [&](const auto left, const auto right) {
    const auto& left_slot = request.remaining_slots[static_cast<std::size_t>(
        state.assignment_slot[left])];
    const auto& right_slot = request.remaining_slots[static_cast<std::size_t>(
        state.assignment_slot[right])];
    return std::tie(left_slot.step, left_slot.actor,
                    request.obligations[left].id) <
           std::tie(right_slot.step, right_slot.actor,
                    request.obligations[right].id);
  });
  std::vector<std::int64_t> balances;
  for (const auto& certificate : request.resources) {
    balances.push_back(certificate.available);
  }
  for (const auto index : kept) {
    const auto& item = request.obligations[index];
    for (const auto& claim : item.resources) {
      auto& balance = balances[validation.resource_by_key.at(claim.key)];
      if (balance < claim.amount) return false;
      balance -= claim.amount;
    }
    for (const auto& production : item.produces) {
      balances[validation.resource_by_key.at(production.key)] +=
          production.amount;
    }
  }
  return true;
}

bool dependency_was_dropped(const Validation& validation,
                            const SearchState& state,
                            const Obligation& obligation) {
  return std::any_of(obligation.dependencies.begin(),
                     obligation.dependencies.end(), [&](const auto id) {
    return state.disposition[validation.by_id.at(id)] ==
           WorkingDisposition::Dropped;
  });
}

bool dependency_was_deferred(const Validation& validation,
                             const SearchState& state,
                             const Obligation& obligation) {
  return std::any_of(obligation.dependencies.begin(),
                     obligation.dependencies.end(), [&](const auto id) {
    return state.disposition[validation.by_id.at(id)] ==
           WorkingDisposition::Deferred;
  });
}

int deferred_count(const SearchState& state) {
  return static_cast<int>(std::count(state.disposition.begin(),
                                     state.disposition.end(),
                                     WorkingDisposition::Deferred));
}

bool may_defer(const Request& request, const SearchState& state,
               const Obligation& obligation) {
  return obligation.allow_defer &&
         obligation.deadline > request.horizon_end_step &&
         obligation.carry_days < request.maximum_carry_days &&
         deferred_count(state) < request.maximum_deferred_ledger;
}

std::uint64_t state_signature(const Request& request,
                              const SearchState& state) {
  std::uint64_t hash = kFnvOffset;
  std::vector<std::size_t> indices(request.obligations.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](const auto left,
                                                const auto right) {
    return request.obligations[left].id < request.obligations[right].id;
  });
  for (const auto index : indices) {
    mix(hash, request.obligations[index].id);
    mix(hash, static_cast<std::uint64_t>(state.disposition[index]));
    mix(hash, as_u64(state.assignment_slot[index]));
  }
  return hash;
}

struct SolverResult {
  bool complete{};
  bool limit_hit{};
  std::size_t explored{};
  SearchState best;
};

SolverResult solve_exact(const Request& request, const Validation& validation,
                         const std::vector<int>& slot_order) {
  SolverResult result;
  SearchState state;
  state.disposition.assign(request.obligations.size(),
                           WorkingDisposition::Unset);
  state.assignment_slot.assign(request.obligations.size(), -1);
  state.used_slot.assign(request.remaining_slots.size(), false);
  Objective best_objective;
  best_objective.total_damage = std::numeric_limits<std::int64_t>::max();

  std::function<void(std::size_t)> search = [&](const std::size_t topo_offset) {
    if (result.limit_hit) {
      return;
    }
    if (++result.explored > request.maximum_search_states) {
      result.limit_hit = true;
      return;
    }
    if (state.objective.total_damage > best_objective.total_damage) {
      return;
    }
    if (topo_offset == validation.topo.size()) {
      if (!all_resource_prefixes_feasible(request, validation, state)) {
        return;
      }
      state.objective.deterministic_signature = state_signature(request, state);
      if (!result.complete || objective_less(state.objective, best_objective)) {
        result.complete = true;
        best_objective = state.objective;
        result.best = state;
      }
      return;
    }

    const auto index = validation.topo[topo_offset];
    const auto& obligation = request.obligations[index];
    if (dependency_was_dropped(validation, state, obligation)) {
      state.disposition[index] = WorkingDisposition::Dropped;
      state.objective.total_damage +=
          obligation.value_loss + obligation.cascade_damage;
      ++state.objective.dropped_count;
      search(topo_offset + 1);
      --state.objective.dropped_count;
      state.objective.total_damage -=
          obligation.value_loss + obligation.cascade_damage;
      state.disposition[index] = WorkingDisposition::Unset;
      return;
    }

    const bool dependency_deferred =
        dependency_was_deferred(validation, state, obligation);
    if (!dependency_deferred) {
      const int earliest = std::max(request.current_step,
                                    obligation.earliest_step);
      for (const int slot_index : slot_order) {
        const auto& slot =
            request.remaining_slots[static_cast<std::size_t>(slot_index)];
        if (state.used_slot[static_cast<std::size_t>(slot_index)] ||
            slot.move_committed || slot.step < earliest ||
            slot.step > obligation.deadline ||
            (obligation.actor >= 0 && obligation.actor != slot.actor) ||
            !dependencies_execute_before(request, validation, state,
                                         obligation, slot)) {
          continue;
        }
        state.disposition[index] = WorkingDisposition::Kept;
        state.assignment_slot[index] = slot_index;
        state.used_slot[static_cast<std::size_t>(slot_index)] = true;
        const auto perturbation =
            static_cast<std::int64_t>(slot.step - obligation.earliest_step) +
            (obligation.actor < 0 ? slot.actor : 0);
        state.objective.slot_perturbation += perturbation;
        search(topo_offset + 1);
        state.objective.slot_perturbation -= perturbation;
        state.used_slot[static_cast<std::size_t>(slot_index)] = false;
        state.assignment_slot[index] = -1;
        state.disposition[index] = WorkingDisposition::Unset;
      }
    }

    if (may_defer(request, state, obligation)) {
      state.disposition[index] = WorkingDisposition::Deferred;
      state.objective.total_damage += obligation.defer_damage;
      ++state.objective.deferred_count;
      search(topo_offset + 1);
      --state.objective.deferred_count;
      state.objective.total_damage -= obligation.defer_damage;
      state.disposition[index] = WorkingDisposition::Unset;
    }

    state.disposition[index] = WorkingDisposition::Dropped;
    state.objective.total_damage +=
        obligation.value_loss + obligation.cascade_damage;
    ++state.objective.dropped_count;
    search(topo_offset + 1);
    --state.objective.dropped_count;
    state.objective.total_damage -=
        obligation.value_loss + obligation.cascade_damage;
    state.disposition[index] = WorkingDisposition::Unset;
  };
  search(0);
  return result;
}

SearchState solve_fallback(const Request& request, const Validation& validation,
                           const std::vector<int>& slot_order,
                           std::size_t& explored) {
  SearchState state;
  state.disposition.assign(request.obligations.size(),
                           WorkingDisposition::Unset);
  state.assignment_slot.assign(request.obligations.size(), -1);
  state.used_slot.assign(request.remaining_slots.size(), false);
  explored = 0;

  for (const auto index : validation.topo) {
    ++explored;
    const auto& obligation = request.obligations[index];
    bool kept = false;
    if (!dependency_was_dropped(validation, state, obligation) &&
        !dependency_was_deferred(validation, state, obligation)) {
      const int earliest = std::max(request.current_step,
                                    obligation.earliest_step);
      for (const int slot_index : slot_order) {
        const auto& slot =
            request.remaining_slots[static_cast<std::size_t>(slot_index)];
        if (!state.used_slot[static_cast<std::size_t>(slot_index)] &&
            !slot.move_committed && slot.step >= earliest &&
            slot.step <= obligation.deadline &&
            (obligation.actor < 0 || obligation.actor == slot.actor) &&
            dependencies_execute_before(request, validation, state,
                                        obligation, slot) &&
            resources_fit_at(request, validation, state, obligation, slot)) {
          state.disposition[index] = WorkingDisposition::Kept;
          state.assignment_slot[index] = slot_index;
          state.used_slot[static_cast<std::size_t>(slot_index)] = true;
          if (!all_resource_prefixes_feasible(request, validation, state)) {
            state.used_slot[static_cast<std::size_t>(slot_index)] = false;
            state.assignment_slot[index] = -1;
            state.disposition[index] = WorkingDisposition::Unset;
            continue;
          }
          state.objective.slot_perturbation +=
              static_cast<std::int64_t>(slot.step - obligation.earliest_step) +
              (obligation.actor < 0 ? slot.actor : 0);
          kept = true;
          break;
        }
      }
    }
    if (kept) {
      continue;
    }
    if (!dependency_was_dropped(validation, state, obligation) &&
        may_defer(request, state, obligation)) {
      state.disposition[index] = WorkingDisposition::Deferred;
      state.objective.total_damage += obligation.defer_damage;
      ++state.objective.deferred_count;
    } else {
      state.disposition[index] = WorkingDisposition::Dropped;
      state.objective.total_damage +=
          obligation.value_loss + obligation.cascade_damage;
      ++state.objective.dropped_count;
    }
  }
  state.objective.deterministic_signature = state_signature(request, state);
  return state;
}

std::uint64_t resource_evidence_hash(const Request& request,
                                     const Obligation& obligation) {
  std::vector<const ResourceCertificate*> certificates;
  for (const auto& claim : obligation.resources) {
    const auto found = std::find_if(
        request.resources.begin(), request.resources.end(),
        [&](const auto& certificate) { return certificate.key == claim.key; });
    if (found != request.resources.end()) {
      certificates.push_back(&*found);
    }
  }
  for (const auto& production : obligation.produces) {
    const auto found = std::find_if(
        request.resources.begin(), request.resources.end(),
        [&](const auto& certificate) {
          return certificate.key == production.key;
        });
    if (found != request.resources.end() &&
        std::find(certificates.begin(), certificates.end(), &*found) ==
            certificates.end()) {
      certificates.push_back(&*found);
    }
  }
  std::sort(certificates.begin(), certificates.end(),
            [](const auto* left, const auto* right) {
              return left->key < right->key;
            });
  std::uint64_t hash = kFnvOffset;
  mix(hash, request.observation_hash);
  for (const auto* certificate : certificates) {
    mix(hash, certificate->key);
    mix(hash, as_u64(certificate->available));
    mix(hash, certificate->evidence_hash);
  }
  return hash;
}

std::uint64_t first_dropped_dependency(const Validation& validation,
                                       const SearchState& state,
                                       const Obligation& obligation) {
  std::uint64_t first = 0;
  for (const auto id : obligation.dependencies) {
    if (state.disposition[validation.by_id.at(id)] ==
            WorkingDisposition::Dropped &&
        (first == 0 || id < first)) {
      first = id;
    }
  }
  return first;
}

std::uint64_t first_deferred_dependency(const Validation& validation,
                                        const SearchState& state,
                                        const Obligation& obligation) {
  std::uint64_t first = 0;
  for (const auto id : obligation.dependencies) {
    if (state.disposition[validation.by_id.at(id)] ==
            WorkingDisposition::Deferred &&
        (first == 0 || id < first)) {
      first = id;
    }
  }
  return first;
}

bool individually_resource_feasible(const Request& request,
                                    const Validation& validation,
                                    const Obligation& obligation) {
  std::vector<std::int64_t> possible;
  for (const auto& certificate : request.resources) {
    possible.push_back(certificate.available);
  }
  for (const auto& candidate : request.obligations) {
    if (candidate.id == obligation.id) continue;
    for (const auto& production : candidate.produces) {
      possible[validation.resource_by_key.at(production.key)] +=
          production.amount;
    }
  }
  for (const auto& claim : obligation.resources) {
    if (claim.amount > possible[validation.resource_by_key.at(claim.key)]) {
      return false;
    }
  }
  return true;
}

std::vector<int> individually_eligible_slots(const Request& request,
                                             const Obligation& obligation,
                                             const bool include_moves) {
  std::vector<int> result;
  const int earliest = std::max(request.current_step, obligation.earliest_step);
  for (std::size_t index = 0; index < request.remaining_slots.size(); ++index) {
    const auto& slot = request.remaining_slots[index];
    if ((include_moves || !slot.move_committed) && slot.step >= earliest &&
        slot.step <= obligation.deadline &&
        (obligation.actor < 0 || obligation.actor == slot.actor)) {
      result.push_back(static_cast<int>(index));
    }
  }
  return result;
}

DropCertificate make_drop(const Request& request, const Validation& validation,
                          const SearchState& state, const std::size_t index,
                          const int final_deferred_count) {
  const auto& obligation = request.obligations[index];
  DropCertificate certificate;
  certificate.obligation_id = obligation.id;
  certificate.value_loss = obligation.value_loss;
  certificate.cascade_damage = obligation.cascade_damage;
  certificate.observation_hash = request.observation_hash;
  certificate.resource_evidence_hash =
      resource_evidence_hash(request, obligation);
  certificate.lineage_hash = obligation.lineage_hash;
  certificate.deadline = obligation.deadline;

  const auto cancelled_by =
      first_dropped_dependency(validation, state, obligation);
  if (cancelled_by != 0) {
    certificate.disposition = Disposition::DroppedInfeasible;
    certificate.reason = DropReason::DependencyDropped;
    certificate.cancelled_by_dependency = cancelled_by;
    certificate.diagnostic = "dependency was explicitly dropped";
  } else if (const auto deferred_dependency =
                 first_deferred_dependency(validation, state, obligation);
             deferred_dependency != 0) {
    certificate.disposition = Disposition::DroppedInfeasible;
    certificate.reason = DropReason::DependencyDeferred;
    certificate.cancelled_by_dependency = deferred_dependency;
    certificate.diagnostic = "dependency was deferred beyond this horizon";
  } else if (obligation.deadline < request.current_step) {
    certificate.disposition = Disposition::DroppedInfeasible;
    certificate.reason = DropReason::DeadlineExpired;
    certificate.diagnostic = "deadline precedes current observation";
  } else if (obligation.allow_defer &&
             obligation.deadline > request.horizon_end_step &&
             obligation.carry_days >= request.maximum_carry_days) {
    certificate.disposition = Disposition::DroppedInfeasible;
    certificate.reason = DropReason::LedgerAgeBound;
    certificate.diagnostic = "explicit debt reached maximum carry age";
  } else if (obligation.allow_defer &&
             obligation.deadline > request.horizon_end_step &&
             obligation.carry_days < request.maximum_carry_days &&
             final_deferred_count >= request.maximum_deferred_ledger) {
    certificate.disposition = Disposition::DroppedInfeasible;
    certificate.reason = DropReason::LedgerCapacityBound;
    certificate.diagnostic = "bounded deferred ledger is full";
  } else if (!individually_resource_feasible(request, validation,
                                             obligation)) {
    certificate.disposition = Disposition::DroppedInfeasible;
    certificate.reason = DropReason::ResourceUnavailable;
    certificate.diagnostic = "current certificate cannot cover claim";
  } else {
    const auto including_moves =
        individually_eligible_slots(request, obligation, true);
    const auto without_moves =
        individually_eligible_slots(request, obligation, false);
    if (including_moves.empty()) {
      certificate.disposition = Disposition::DroppedInfeasible;
      certificate.reason = DropReason::NoEligibleSlot;
      certificate.diagnostic = "no remaining actor/deadline slot";
    } else if (without_moves.empty()) {
      certificate.disposition = Disposition::DroppedInfeasible;
      certificate.reason = DropReason::MoveCommittedSlot;
      certificate.diagnostic = "all individually eligible slots are MOVE commitments";
    } else {
      certificate.disposition = Disposition::DroppedDominated;
      certificate.reason = DropReason::DominatedByLowerDamageSet;
      certificate.diagnostic = "excluded by deterministic minimum-damage set";
      for (std::size_t kept_index = 0;
           kept_index < request.obligations.size(); ++kept_index) {
        if (state.disposition[kept_index] != WorkingDisposition::Kept) {
          continue;
        }
        const int kept_slot = state.assignment_slot[kept_index];
        const bool slot_conflict = std::find(
            without_moves.begin(), without_moves.end(), kept_slot) !=
            without_moves.end();
        bool resource_conflict = false;
        for (const auto& claim : obligation.resources) {
          resource_conflict = resource_conflict || std::any_of(
              request.obligations[kept_index].resources.begin(),
              request.obligations[kept_index].resources.end(),
              [&](const auto& kept_claim) {
                return kept_claim.key == claim.key;
              });
        }
        if (slot_conflict || resource_conflict) {
          certificate.conflicting_kept_obligations.push_back(
              request.obligations[kept_index].id);
        }
      }
      std::sort(certificate.conflicting_kept_obligations.begin(),
                certificate.conflicting_kept_obligations.end());
    }
  }
  certificate.content_hash = drop_certificate_hash(certificate);
  return certificate;
}

std::uint64_t move_hash(const Request& request) {
  std::vector<const Slot*> moves;
  for (const auto& slot : request.remaining_slots) {
    if (slot.move_committed) {
      moves.push_back(&slot);
    }
  }
  std::sort(moves.begin(), moves.end(), [](const auto* left, const auto* right) {
    return std::tie(left->step, left->actor, left->move_source_step,
                    left->move_token) <
           std::tie(right->step, right->actor, right->move_source_step,
                    right->move_token);
  });
  std::uint64_t hash = kFnvOffset;
  for (const auto* slot : moves) {
    mix(hash, as_u64(slot->step));
    mix(hash, as_u64(slot->actor));
    mix(hash, as_u64(slot->move_source_step));
    mix(hash, slot->move_token);
    mix(hash, slot->final_owner_generation);
  }
  return hash;
}

std::uint64_t request_hash(const Request& request) {
  std::uint64_t hash = kFnvOffset;
  mix(hash, as_u64(request.current_step));
  mix(hash, as_u64(request.horizon_end_step));
  mix(hash, as_u64(request.turns_per_day));
  mix(hash, request.observation_hash);
  mix(hash, request.final_owner_generation);
  mix(hash, as_u64(request.maximum_carry_days));
  mix(hash, as_u64(request.maximum_deferred_ledger));
  mix(hash, request.exact_node_limit);
  mix(hash, request.maximum_search_states);

  std::vector<const ResourceCertificate*> resources;
  for (const auto& item : request.resources) resources.push_back(&item);
  std::sort(resources.begin(), resources.end(), [](const auto* a, const auto* b) {
    return a->key < b->key;
  });
  for (const auto* item : resources) {
    mix(hash, item->key);
    mix(hash, as_u64(item->available));
    mix(hash, item->evidence_hash);
  }

  std::vector<const Slot*> slots;
  for (const auto& item : request.remaining_slots) slots.push_back(&item);
  std::sort(slots.begin(), slots.end(), [](const auto* a, const auto* b) {
    return std::tie(a->step, a->actor) < std::tie(b->step, b->actor);
  });
  for (const auto* item : slots) {
    mix(hash, as_u64(item->step));
    mix(hash, as_u64(item->actor));
    mix(hash, item->move_committed ? 1U : 0U);
    mix(hash, as_u64(item->move_source_step));
    mix(hash, item->move_token);
    mix(hash, item->final_owner_generation);
  }

  std::vector<const Obligation*> obligations;
  for (const auto& item : request.obligations) obligations.push_back(&item);
  std::sort(obligations.begin(), obligations.end(), [](const auto* a,
                                                       const auto* b) {
    return a->id < b->id;
  });
  for (const auto* item : obligations) {
    mix(hash, item->id);
    mix(hash, as_u64(static_cast<int>(item->action.op)));
    mix(hash, as_u64(static_cast<int>(item->action.item)));
    mix(hash, as_u64(item->action.quantity));
    mix(hash, as_u64(item->tile.x));
    mix(hash, as_u64(item->tile.y));
    mix(hash, as_u64(item->actor));
    mix(hash, as_u64(item->earliest_step));
    mix(hash, as_u64(item->deadline));
    auto dependencies = item->dependencies;
    std::sort(dependencies.begin(), dependencies.end());
    for (const auto dependency : dependencies) mix(hash, dependency);
    std::vector<ResourceClaim> claims = item->resources;
    std::sort(claims.begin(), claims.end(), [](const auto& a, const auto& b) {
      return a.key < b.key;
    });
    for (const auto& claim : claims) {
      mix(hash, claim.key);
      mix(hash, as_u64(claim.amount));
    }
    std::vector<ResourceProduction> productions = item->produces;
    std::sort(productions.begin(), productions.end(),
              [](const auto& a, const auto& b) { return a.key < b.key; });
    for (const auto& production : productions) {
      mix(hash, production.key);
      mix(hash, as_u64(production.amount));
    }
    mix(hash, as_u64(item->value_loss));
    mix(hash, as_u64(item->cascade_damage));
    mix(hash, as_u64(item->defer_damage));
    mix(hash, item->allow_defer ? 1U : 0U);
    mix(hash, as_u64(item->carry_days));
    mix(hash, item->lineage_hash);
  }
  return hash;
}

Plan materialize(const Request& request, const Validation& validation,
                 const SearchState& state, const bool exact,
                 const bool fallback, const std::size_t explored) {
  Plan result;
  result.status = PlanStatus::Accepted;
  result.exact = exact;
  result.bounded_fallback = fallback;
  result.explored_states = explored;
  result.state_limit = request.maximum_search_states;
  result.objective = state.objective;
  result.preserved_move_commitments = static_cast<int>(std::count_if(
      request.remaining_slots.begin(), request.remaining_slots.end(),
      [](const auto& slot) { return slot.move_committed; }));
  result.move_commitment_hash = move_hash(request);
  result.observation_hash = request.observation_hash;
  result.final_owner_generation = request.final_owner_generation;
  result.request_content_hash = request_hash(request);
  const int final_deferred_count = deferred_count(state);

  for (const auto index : validation.topo) {
    const auto& obligation = request.obligations[index];
    Decision decision;
    decision.obligation_id = obligation.id;
    switch (state.disposition[index]) {
      case WorkingDisposition::Kept: {
        decision.disposition = Disposition::Kept;
        const int slot_index = state.assignment_slot[index];
        const auto& slot =
            request.remaining_slots[static_cast<std::size_t>(slot_index)];
        Assignment assignment{obligation.id, slot_index, slot.step, slot.actor};
        decision.assignment = assignment;
        result.assignments.push_back(assignment);
        result.kept_subgraph.push_back(obligation.id);
        break;
      }
      case WorkingDisposition::Deferred: {
        decision.disposition = Disposition::Deferred;
        DeferredDebt debt{obligation.id,
                          obligation.carry_days,
                          obligation.carry_days + 1,
                          obligation.deadline,
                          obligation.defer_damage,
                          obligation.lineage_hash};
        decision.deferred = debt;
        result.deferred_ledger.push_back(debt);
        break;
      }
      case WorkingDisposition::Dropped: {
        auto certificate = make_drop(request, validation, state, index,
                                     final_deferred_count);
        decision.disposition = certificate.disposition;
        decision.drop = certificate;
        result.drops.push_back(certificate);
        break;
      }
      case WorkingDisposition::Unset:
        result.status = PlanStatus::InvalidInput;
        result.diagnostic = "internal omission: obligation has no disposition";
        return result;
    }
    result.decisions.push_back(std::move(decision));
  }
  result.diagnostic = exact ? "exact minimum-damage feasible subgraph"
                            : "deterministic bounded fallback subgraph";
  result.content_hash = plan_hash(result);
  return result;
}

}  // namespace

Plan plan(const Request& request) {
  const auto validation = validate(request);
  if (!validation.valid) {
    Plan result;
    result.status = PlanStatus::InvalidInput;
    result.state_limit = request.maximum_search_states;
    result.observation_hash = request.observation_hash;
    result.final_owner_generation = request.final_owner_generation;
    result.request_content_hash = request_hash(request);
    result.diagnostic = validation.diagnostic;
    result.content_hash = plan_hash(result);
    return result;
  }
  const auto slot_order = sorted_slots(request);
  if (request.obligations.size() <= request.exact_node_limit) {
    const auto exact = solve_exact(request, validation, slot_order);
    if (!exact.limit_hit && exact.complete) {
      return materialize(request, validation, exact.best, true, false,
                         exact.explored);
    }
  }
  std::size_t explored = 0;
  const auto fallback = solve_fallback(request, validation, slot_order, explored);
  return materialize(request, validation, fallback, false, true, explored);
}

VerifyResult verify(const Request& request, const Plan& candidate) {
  if (plan_hash(candidate) != candidate.content_hash) {
    return {false, "candidate content hash mismatch"};
  }
  const auto validation = validate(request);
  if (!validation.valid) {
    const auto expected = plan(request);
    return {expected.content_hash == candidate.content_hash,
            expected.content_hash == candidate.content_hash
                ? "deterministic invalid-input rejection verified"
                : "invalid request rejection mismatch"};
  }
  if (!candidate.accepted() ||
      candidate.observation_hash != request.observation_hash ||
      candidate.final_owner_generation != request.final_owner_generation ||
      candidate.request_content_hash != request_hash(request) ||
      candidate.preserved_move_commitments !=
          static_cast<int>(std::count_if(
              request.remaining_slots.begin(), request.remaining_slots.end(),
              [](const auto& slot) { return slot.move_committed; })) ||
      candidate.move_commitment_hash != move_hash(request)) {
    return {false, "candidate authority or MOVE binding mismatch"};
  }
  SearchState audited;
  audited.disposition.assign(request.obligations.size(),
                              WorkingDisposition::Unset);
  audited.assignment_slot.assign(request.obligations.size(), -1);
  audited.used_slot.assign(request.remaining_slots.size(), false);
  std::unordered_set<std::uint64_t> seen;
  for (const auto& item : candidate.decisions) {
    const auto found = validation.by_id.find(item.obligation_id);
    if (found == validation.by_id.end() || !seen.insert(item.obligation_id).second) {
      return {false, "unknown or duplicate decision identity"};
    }
    const auto index = found->second;
    const auto& obligation = request.obligations[index];
    if (item.disposition == Disposition::Kept) {
      if (!item.assignment || item.drop || item.deferred ||
          item.assignment->obligation_id != item.obligation_id ||
          item.assignment->slot_index < 0 ||
          static_cast<std::size_t>(item.assignment->slot_index) >=
              request.remaining_slots.size()) {
        return {false, "malformed kept decision"};
      }
      const auto slot_index =
          static_cast<std::size_t>(item.assignment->slot_index);
      const auto& assigned = request.remaining_slots[slot_index];
      if (assigned.move_committed || audited.used_slot[slot_index] ||
          assigned.step != item.assignment->step ||
          assigned.actor != item.assignment->actor ||
          assigned.step < std::max(request.current_step,
                                   obligation.earliest_step) ||
          assigned.step > obligation.deadline ||
          (obligation.actor >= 0 && obligation.actor != assigned.actor)) {
        return {false, "kept assignment violates actor/slot/MOVE bounds"};
      }
      audited.disposition[index] = WorkingDisposition::Kept;
      audited.assignment_slot[index] = item.assignment->slot_index;
      audited.used_slot[slot_index] = true;
    } else if (item.disposition == Disposition::Deferred) {
      if (item.assignment || item.drop || !item.deferred ||
          !obligation.allow_defer ||
          obligation.deadline <= request.horizon_end_step ||
          obligation.carry_days >= request.maximum_carry_days ||
          item.deferred->next_carry_days != obligation.carry_days + 1) {
        return {false, "malformed or unbounded deferred decision"};
      }
      audited.disposition[index] = WorkingDisposition::Deferred;
    } else {
      if (item.assignment || !item.drop || item.deferred ||
          item.drop->obligation_id != item.obligation_id ||
          item.drop->disposition != item.disposition ||
          item.drop->content_hash != drop_certificate_hash(*item.drop)) {
        return {false, "malformed dropped decision or certificate"};
      }
      audited.disposition[index] = WorkingDisposition::Dropped;
    }
  }
  if (seen.size() != request.obligations.size() ||
      static_cast<int>(candidate.deferred_ledger.size()) >
          request.maximum_deferred_ledger) {
    return {false, "candidate omitted a node or exceeded ledger capacity"};
  }
  for (const auto index : validation.topo) {
    if (audited.disposition[index] != WorkingDisposition::Kept) continue;
    const auto& item = request.obligations[index];
    const auto& assigned = request.remaining_slots[static_cast<std::size_t>(
        audited.assignment_slot[index])];
    for (const auto dependency : item.dependencies) {
      const auto dep_index = validation.by_id.at(dependency);
      if (audited.disposition[dep_index] != WorkingDisposition::Kept) {
        return {false, "kept node has non-kept dependency"};
      }
      const auto& dep_slot = request.remaining_slots[static_cast<std::size_t>(
          audited.assignment_slot[dep_index])];
      if (!execution_before(dep_slot, assigned)) {
        return {false, "dependency execution order is not strict"};
      }
    }
  }
  if (!all_resource_prefixes_feasible(request, validation, audited)) {
    return {false, "resource balance is negative at an execution prefix"};
  }
  const auto expected = plan(request);
  if (expected.status != candidate.status ||
      expected.content_hash != candidate.content_hash) {
    return {false, "candidate differs from deterministic planner result"};
  }
  if (candidate.accepted() &&
      candidate.decisions.size() != request.obligations.size()) {
    return {false, "candidate silently omitted an obligation"};
  }
  return {true, "deterministic plan and certificates verified"};
}

std::uint64_t drop_certificate_hash(
    const DropCertificate& certificate) noexcept {
  std::uint64_t hash = kFnvOffset;
  mix(hash, certificate.obligation_id);
  mix(hash, static_cast<std::uint64_t>(certificate.disposition));
  mix(hash, static_cast<std::uint64_t>(certificate.reason));
  mix(hash, certificate.cancelled_by_dependency);
  for (const auto id : certificate.conflicting_kept_obligations) {
    mix(hash, id);
  }
  mix(hash, as_u64(certificate.value_loss));
  mix(hash, as_u64(certificate.cascade_damage));
  mix(hash, certificate.observation_hash);
  mix(hash, certificate.resource_evidence_hash);
  mix(hash, certificate.lineage_hash);
  mix(hash, as_u64(certificate.deadline));
  mix(hash, certificate.diagnostic);
  return hash;
}

std::uint64_t plan_hash(const Plan& plan_value) noexcept {
  std::uint64_t hash = kFnvOffset;
  mix(hash, static_cast<std::uint64_t>(plan_value.status));
  mix(hash, plan_value.exact ? 1U : 0U);
  mix(hash, plan_value.bounded_fallback ? 1U : 0U);
  mix(hash, plan_value.explored_states);
  mix(hash, plan_value.state_limit);
  mix(hash, as_u64(plan_value.objective.total_damage));
  mix(hash, as_u64(plan_value.objective.dropped_count));
  mix(hash, as_u64(plan_value.objective.deferred_count));
  mix(hash, as_u64(plan_value.objective.slot_perturbation));
  mix(hash, plan_value.objective.deterministic_signature);
  for (const auto id : plan_value.kept_subgraph) mix(hash, id);
  for (const auto& assignment : plan_value.assignments) {
    mix(hash, assignment.obligation_id);
    mix(hash, as_u64(assignment.slot_index));
    mix(hash, as_u64(assignment.step));
    mix(hash, as_u64(assignment.actor));
  }
  for (const auto& decision : plan_value.decisions) {
    mix(hash, decision.obligation_id);
    mix(hash, static_cast<std::uint64_t>(decision.disposition));
    if (decision.assignment) {
      mix(hash, decision.assignment->obligation_id);
      mix(hash, as_u64(decision.assignment->slot_index));
      mix(hash, as_u64(decision.assignment->step));
      mix(hash, as_u64(decision.assignment->actor));
    } else {
      mix(hash, 0U);
    }
    mix(hash, decision.drop.has_value() ? decision.drop->content_hash : 0U);
    if (decision.deferred) {
      mix(hash, decision.deferred->obligation_id);
      mix(hash, as_u64(decision.deferred->prior_carry_days));
      mix(hash, as_u64(decision.deferred->next_carry_days));
      mix(hash, as_u64(decision.deferred->deadline));
      mix(hash, as_u64(decision.deferred->charged_damage));
      mix(hash, decision.deferred->lineage_hash);
    } else {
      mix(hash, 0U);
    }
  }
  for (const auto& drop : plan_value.drops) mix(hash, drop.content_hash);
  for (const auto& debt : plan_value.deferred_ledger) {
    mix(hash, debt.obligation_id);
    mix(hash, as_u64(debt.prior_carry_days));
    mix(hash, as_u64(debt.next_carry_days));
    mix(hash, as_u64(debt.deadline));
    mix(hash, as_u64(debt.charged_damage));
    mix(hash, debt.lineage_hash);
  }
  mix(hash, as_u64(plan_value.preserved_move_commitments));
  mix(hash, plan_value.move_commitment_hash);
  mix(hash, plan_value.observation_hash);
  mix(hash, plan_value.final_owner_generation);
  mix(hash, plan_value.request_content_hash);
  mix(hash, plan_value.diagnostic);
  return hash;
}

const char* disposition_name(const Disposition disposition) noexcept {
  switch (disposition) {
    case Disposition::Kept: return "Kept";
    case Disposition::Deferred: return "Deferred";
    case Disposition::DroppedInfeasible: return "DroppedInfeasible";
    case Disposition::DroppedDominated: return "DroppedDominated";
  }
  return "Unknown";
}

const char* drop_reason_name(const DropReason reason) noexcept {
  switch (reason) {
    case DropReason::None: return "None";
    case DropReason::DependencyDropped: return "DependencyDropped";
    case DropReason::DependencyDeferred: return "DependencyDeferred";
    case DropReason::DeadlineExpired: return "DeadlineExpired";
    case DropReason::NoEligibleSlot: return "NoEligibleSlot";
    case DropReason::MoveCommittedSlot: return "MoveCommittedSlot";
    case DropReason::ResourceUnavailable: return "ResourceUnavailable";
    case DropReason::LedgerAgeBound: return "LedgerAgeBound";
    case DropReason::LedgerCapacityBound: return "LedgerCapacityBound";
    case DropReason::DominatedByLowerDamageSet:
      return "DominatedByLowerDamageSet";
    case DropReason::UnsupportedInput: return "UnsupportedInput";
  }
  return "Unknown";
}

}  // namespace g001::constraint_relaxation
