#pragma once
#include "production_obligation_day_scheduler.hpp"
#include "native_teammate.hpp"

namespace g001::movement_plan_owner {
// Isolated experiment seam.  Callers must supply a captured complete day-start
// request; the owner does not claim that a synthetic seed/step fixture is a
// replay of NativeTeammate's real provider state.
struct Request {
  obligation_day::DayPlanRequest day;
  bool unsupported_overlay{};
};
struct Result {
  obligation_day::DayPlanResult plan;
  fastkag::NativeMovementCommitment commitment;
  bool authorized{};
};
[[nodiscard]] Result plan(const Request& request);
}
