#pragma once

#include "production_obligation_day_scheduler.hpp"

#include <cstdint>
#include <vector>

namespace g001::obligation_day {

struct AtomicGroup {
  std::uint64_t id{};
  // Predicted advantage over leaving the provider baseline unchanged.
  // Non-positive groups are admitted only as dependencies of a positive root.
  double value{};
  std::vector<std::uint64_t> obligation_ids;
};

enum class AtomicAdmissionReject : std::uint8_t {
  None = 0,
  InvalidGroups,
  PlannerRejected,
  VerificationFailed,
};

struct AtomicGroupResult {
  std::uint64_t group_id{};
  bool admitted{};
  // Completed when admitted; otherwise the first disposition which prevented
  // the candidate closure from completing.
  ObligationDisposition disposition{
      ObligationDisposition::PolicyDeferredDebt};
  std::uint64_t failed_obligation_id{};
  PlanReject planner_reject{PlanReject::None};
};

struct AtomicAdmissionResult {
  AtomicAdmissionReject reject{AtomicAdmissionReject::None};
  DayPlanRequest effective_request;
  DayPlanResult verified_plan;
  VerifyResult verification;
  std::vector<AtomicGroupResult> groups;
  int planner_calls{};

  [[nodiscard]] bool planned() const {
    return reject == AtomicAdmissionReject::None && verified_plan.planned() &&
           verification.valid;
  }
};

// Every input obligation must occur in exactly one non-empty group. A group
// transitively activates the groups containing all of its dependencies.
// Obligations already marked policy_deferred in the input are never activated.
[[nodiscard]] AtomicAdmissionResult atomic_plan_day(
    const DayPlanRequest& request, const std::vector<AtomicGroup>& groups);
[[nodiscard]] AtomicAdmissionResult atomic_plan_remaining_day(
    const DayPlanRequest& request, const std::vector<AtomicGroup>& groups);

}  // namespace g001::obligation_day
