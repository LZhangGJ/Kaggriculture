#include "movement_plan_owner.hpp"

namespace g001::movement_plan_owner {
Result plan(const Request& request) {
  Result output;
  if (request.unsupported_overlay || !request.day.day_start) return output;
  output.plan = obligation_day::plan_day(request.day);
  if (!output.plan.planned()) return output;
  const auto verification = obligation_day::verify_day_schedule(
      request.day, *output.plan.certificate);
  if (!verification.valid ||
      output.plan.certificate->move_replays.size() != request.day.moves.size())
    return output;
  auto& commitment = output.commitment;
  commitment.player = request.day.player;
  commitment.day = request.day.day_start->day();
  commitment.issuer_generation = request.day.issuer_generation;
  commitment.issued_step = request.day.day_start->step_count();
  commitment.immutable_through_step = commitment.day * 24 + 23;
  commitment.remaining_day_complete = true;
  for (const auto& move : request.day.moves)
    commitment.moves.push_back({move.actor, move.source_step, move.action});
  commitment.content_hash = fastkag::native_movement_commitment_hash(commitment);
  output.authorized = true;
  return output;
}
}
