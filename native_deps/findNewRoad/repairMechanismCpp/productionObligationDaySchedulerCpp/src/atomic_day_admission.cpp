#include "../include/atomic_day_admission.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace g001::obligation_day {
namespace {

struct GroupIndex {
  std::map<std::uint64_t, std::size_t> obligation_by_id;
  std::vector<std::size_t> obligation_group;
};

bool index_groups(const DayPlanRequest& request,
                  const std::vector<AtomicGroup>& groups, GroupIndex& index) {
  std::map<std::uint64_t, std::size_t> group_by_id;
  for (std::size_t i = 0; i < request.obligations.size(); ++i) {
    const auto id = request.obligations[i].id;
    if (id == 0 || !index.obligation_by_id.emplace(id, i).second) return false;
  }
  index.obligation_group.assign(request.obligations.size(), groups.size());
  for (std::size_t group = 0; group < groups.size(); ++group) {
    const auto& definition = groups[group];
    if (definition.id == 0 || definition.obligation_ids.empty() ||
        !std::isfinite(definition.value) ||
        !group_by_id.emplace(definition.id, group).second) {
      return false;
    }
    for (const auto id : definition.obligation_ids) {
      const auto found = index.obligation_by_id.find(id);
      if (found == index.obligation_by_id.end() ||
          index.obligation_group[found->second] != groups.size()) {
        return false;
      }
      index.obligation_group[found->second] = group;
    }
  }
  if (std::find(index.obligation_group.begin(), index.obligation_group.end(),
                groups.size()) != index.obligation_group.end()) {
    return false;
  }
  for (const auto& obligation : request.obligations) {
    for (const auto dependency : obligation.dependencies) {
      if (!index.obligation_by_id.contains(dependency)) return false;
    }
  }
  return true;
}

DayPlanRequest effective_request(const DayPlanRequest& original,
                                 const GroupIndex& index,
                                 const std::vector<bool>& active) {
  DayPlanRequest request = original;
  for (std::size_t i = 0; i < request.obligations.size(); ++i) {
    request.obligations[i].policy_deferred =
        original.obligations[i].policy_deferred ||
        !active[index.obligation_group[i]];
  }
  return request;
}

std::vector<std::size_t> dependency_closure(
    std::size_t root, const DayPlanRequest& request,
    const std::vector<AtomicGroup>& groups, const GroupIndex& index) {
  std::vector<std::size_t> stack{root};
  std::vector<bool> included(groups.size());
  included[root] = true;
  while (!stack.empty()) {
    const auto group = stack.back();
    stack.pop_back();
    for (const auto obligation_id : groups[group].obligation_ids) {
      const auto& obligation =
          request.obligations[index.obligation_by_id.at(obligation_id)];
      for (const auto dependency : obligation.dependencies) {
        const auto dependency_group = index.obligation_group[
            index.obligation_by_id.at(dependency)];
        if (!included[dependency_group]) {
          included[dependency_group] = true;
          stack.push_back(dependency_group);
        }
      }
    }
  }
  std::vector<std::size_t> closure;
  for (std::size_t group = 0; group < groups.size(); ++group) {
    if (included[group]) closure.push_back(group);
  }
  return closure;
}

DayPlanResult run_planner(const DayPlanRequest& request, bool remaining_day) {
  return remaining_day ? plan_remaining_day(request) : plan_day(request);
}

VerifyResult verify_plan(const DayPlanRequest& request,
                         const DayPlanResult& plan, bool remaining_day) {
  if (!plan.certificate) return {};
  return remaining_day
             ? verify_remaining_day_schedule(request, *plan.certificate)
             : verify_day_schedule(request, *plan.certificate);
}

const ObligationFinalStatus* first_incomplete(
    const DayPlanResult& plan, const GroupIndex& index,
    const std::vector<bool>& active) {
  if (!plan.certificate) return nullptr;
  for (const auto& status : plan.certificate->obligation_statuses) {
    const auto found = index.obligation_by_id.find(status.obligation_id);
    if (found != index.obligation_by_id.end() &&
        active[index.obligation_group[found->second]] &&
        status.disposition != ObligationDisposition::Completed) {
      return &status;
    }
  }
  return nullptr;
}

AtomicAdmissionResult atomic_plan(const DayPlanRequest& request,
                                  const std::vector<AtomicGroup>& groups,
                                  bool remaining_day) {
  AtomicAdmissionResult output;
  output.effective_request = request;
  output.groups.reserve(groups.size());
  for (const auto& group : groups) output.groups.push_back({group.id});

  GroupIndex index;
  if (!index_groups(request, groups, index)) {
    output.reject = AtomicAdmissionReject::InvalidGroups;
    return output;
  }

  std::vector<bool> forbidden(groups.size());
  for (std::size_t i = 0; i < request.obligations.size(); ++i) {
    forbidden[index.obligation_group[i]] =
        forbidden[index.obligation_group[i]] ||
        request.obligations[i].policy_deferred;
  }
  std::vector<bool> active(groups.size());
  output.effective_request = effective_request(request, index, active);
  output.verified_plan = run_planner(output.effective_request, remaining_day);
  ++output.planner_calls;
  if (!output.verified_plan.planned()) {
    output.reject = AtomicAdmissionReject::PlannerRejected;
    return output;
  }

  std::vector<std::size_t> order(groups.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    if (groups[lhs].value != groups[rhs].value)
      return groups[lhs].value > groups[rhs].value;
    return groups[lhs].id < groups[rhs].id;
  });

  for (const auto candidate : order) {
    if (active[candidate]) continue;
    // Group value is advantage over the submitted baseline. Non-positive
    // groups are never roots, but may still enter as a prerequisite of a
    // positive group through dependency_closure().
    if (groups[candidate].value <= 0.0) continue;
    const auto closure =
        dependency_closure(candidate, request, groups, index);
    const auto blocked =
        std::find_if(closure.begin(), closure.end(),
                     [&](std::size_t group) { return forbidden[group]; });
    if (blocked != closure.end()) {
      const auto failed = std::find_if(
          groups[*blocked].obligation_ids.begin(),
          groups[*blocked].obligation_ids.end(), [&](std::uint64_t id) {
            return request.obligations[index.obligation_by_id.at(id)]
                .policy_deferred;
          });
      output.groups[candidate].failed_obligation_id =
          failed == groups[*blocked].obligation_ids.end() ? 0 : *failed;
      continue;
    }

    auto trial_active = active;
    for (const auto group : closure) trial_active[group] = true;
    auto trial_request = effective_request(request, index, trial_active);
    auto trial_plan = run_planner(trial_request, remaining_day);
    ++output.planner_calls;
    if (!trial_plan.planned()) {
      output.groups[candidate].disposition =
          ObligationDisposition::UnsupportedStateDebt;
      output.groups[candidate].planner_reject = trial_plan.reject;
      continue;
    }
    const auto* failure = first_incomplete(trial_plan, index, trial_active);
    if (failure) {
      output.groups[candidate].disposition = failure->disposition;
      output.groups[candidate].failed_obligation_id = failure->obligation_id;
      continue;
    }

    active = std::move(trial_active);
    output.effective_request = std::move(trial_request);
    output.verified_plan = std::move(trial_plan);
    for (const auto group : closure) {
      output.groups[group].admitted = true;
      output.groups[group].disposition = ObligationDisposition::Completed;
      output.groups[group].failed_obligation_id = 0;
      output.groups[group].planner_reject = PlanReject::None;
    }
  }

  output.verification =
      verify_plan(output.effective_request, output.verified_plan, remaining_day);
  if (!output.verification.valid)
    output.reject = AtomicAdmissionReject::VerificationFailed;
  return output;
}

}  // namespace

AtomicAdmissionResult atomic_plan_day(
    const DayPlanRequest& request, const std::vector<AtomicGroup>& groups) {
  return atomic_plan(request, groups, false);
}

AtomicAdmissionResult atomic_plan_remaining_day(
    const DayPlanRequest& request, const std::vector<AtomicGroup>& groups) {
  return atomic_plan(request, groups, true);
}

}  // namespace g001::obligation_day
