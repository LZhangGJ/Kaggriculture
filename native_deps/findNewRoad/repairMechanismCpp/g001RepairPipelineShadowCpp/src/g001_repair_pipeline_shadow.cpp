#include "g001_repair_pipeline_shadow.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace g001::repair_pipeline_shadow {
namespace {

bool crop(fastkag::Item item) {
  const int value = static_cast<int>(item);
  return value >= 0 && value < fastkag::N_CROPS;
}

bool animal(fastkag::Item item) {
  const int value = static_cast<int>(item);
  return value >= static_cast<int>(fastkag::Item::GOOSE) &&
         value <= static_cast<int>(fastkag::Item::SHEEP);
}

std::uint64_t lineage_id(const CurrentStepRequest& request,
                         const typed_intent::Evidence& evidence,
                         fastkag::Item item) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto add = [&](std::uint64_t value, std::uint64_t& target) {
    for (int byte = 0; byte < 8; ++byte) {
      target ^= static_cast<std::uint8_t>(value >> (byte * 8));
      target *= 1099511628211ULL;
    }
  };
  add(request.issuer_generation, hash);
  add(static_cast<std::uint64_t>(evidence.source_step), hash);
  add(static_cast<std::uint64_t>(evidence.player), hash);
  add(static_cast<std::uint64_t>(evidence.actor), hash);
  add(static_cast<std::uint64_t>(evidence.tile.x), hash);
  add(static_cast<std::uint64_t>(evidence.tile.y), hash);
  add(static_cast<std::uint8_t>(evidence.source_action.op), hash);
  add(static_cast<std::uint8_t>(item), hash);
  return hash == 0 ? 1 : hash;
}

}  // namespace

CurrentStepResult observe_final_current_units(
    const CurrentStepRequest& request) {
  if (!request.observation || !request.persistent_lineage ||
      request.player < 0 || request.player >= 2 ||
      request.issuer_generation == 0) {
    throw std::invalid_argument("invalid repair pipeline current-step input");
  }
  const auto current = typed_intent::issue(
      {request.observation, request.player, request.issuer_generation,
       request.final_current_units, request.dag_nodes,
       request.dag_suffix_certified});
  CurrentStepResult output;
  output.crops = current.crops;
  output.animals = current.animals;
  output.evidence.reserve(current.evidence.size());

  for (const auto& evidence : current.evidence) {
    TypedEvidence combined;
    combined.current = evidence;
    combined.resolved_item = evidence.resolved_item;
    combined.exact = evidence.exact;
    combined.emitted = evidence.emitted;
    combined.proof = evidence.exact ? TypedProof::CurrentIssuer
                                    : TypedProof::Unresolved;
    const bool crop_work = evidence.source_action.op == fastkag::Op::WATER ||
                           evidence.source_action.op == fastkag::Op::HARVEST;
    if (!combined.exact && crop_work) {
      const auto lineage = request.persistent_lineage->crop_at(evidence.tile);
      if (lineage && crop(lineage->item) &&
          lineage->exact_source.op == fastkag::Op::PLANT &&
          crop(lineage->exact_source.item)) {
        combined.proof = TypedProof::PersistentCropLineage;
        combined.resolved_item = lineage->item;
        combined.exact = combined.emitted = true;
        output.crops.push_back(
            {lineage_id(request, evidence, lineage->item), request.player,
             evidence.actor, evidence.source_step, evidence.tile,
             evidence.source_action, lineage->item});
        ++output.persistent_crop_recoveries;
      }
    }
    output.evidence.push_back(combined);
  }

  // Lineage is an intent receipt: it is written only from exact typed final
  // PLANT/animal PLACE actions, but it does not assert that the physical
  // action succeeded. Physical success belongs to the transactional ledger.
  for (const auto& evidence : output.evidence) {
    if (evidence.current.source_action.op == fastkag::Op::PLANT &&
        evidence.current.exact && crop(evidence.current.resolved_item) &&
        request.persistent_lineage->record_exact_source(
            evidence.current.actor, evidence.current.source_step,
            evidence.current.tile, evidence.current.source_action)) {
      ++output.crop_lineage_records;
    } else if (evidence.current.source_action.op == fastkag::Op::PLACE &&
               evidence.current.exact &&
               animal(evidence.current.resolved_item) &&
               request.persistent_lineage->record_exact_source(
                   evidence.current.actor, evidence.current.source_step,
                   evidence.current.tile, evidence.current.source_action)) {
      ++output.animal_lineage_records;
    }
  }
  return output;
}

DayStartResult plan_and_verify_day_shadow(const DayStartRequest& request) {
  DayStartResult output;
  output.issued = day_start_issuer::issue_day_start(
      {request.day_start, request.immutable_route_tape, request.player,
       request.issuer_generation, request.persistent_lineage, {}, {}});
  if (!output.issued.issued()) {
    return output;
  }
  obligation_day::DayPlanRequest plan_request{
      request.day_start, request.player, request.issuer_generation,
      output.issued.moves, output.issued.obligations};
  output.plan = obligation_day::plan_day(plan_request);
  if (output.plan.certificate) {
    output.verification = obligation_day::verify_day_schedule(
        plan_request, *output.plan.certificate);
  }
  bool must_finish_debt = false;
  for (const auto& debt : output.plan.debts) {
    const auto found = std::find_if(
        output.issued.obligations.begin(), output.issued.obligations.end(),
        [&](const auto& obligation) {
          return obligation.id == debt.obligation_id;
        });
    must_finish_debt =
        must_finish_debt ||
        (found != output.issued.obligations.end() &&
         found->must_finish_today);
  }
  output.locally_full_certificate_eligible =
      output.issued.unsupported.empty() && output.plan.planned() &&
      output.verification.valid && !must_finish_debt;
  output.partial_plan = output.plan.planned() &&
                        !output.locally_full_certificate_eligible;
  return output;
}

UnifiedDayResult plan_unified_day_shadow(const DayStartRequest& request) {
  UnifiedDayResult output;
  output.issued = day_start_issuer::issue_day_start(
      {request.day_start, request.immutable_route_tape, request.player,
       request.issuer_generation, request.persistent_lineage, {}, {}});
  if (!output.issued.issued() || !request.day_start) return output;

  struct Candidate {
    state_target::PlotTarget target;
    int source_step{-1};
  };
  using TileKey = std::pair<int, int>;
  std::map<TileKey, Candidate> candidates;
  std::map<std::pair<int, int>, fastkag::Position> supply;
  const auto key = [](fastkag::Position tile) {
    return TileKey{tile.x, tile.y};
  };
  const auto is_crop = [](fastkag::Item item) {
    const int value = static_cast<int>(item);
    return value >= 0 && value < fastkag::N_CROPS;
  };
  const auto is_animal = [](fastkag::Item item) {
    return item >= fastkag::Item::GOOSE && item <= fastkag::Item::SHEEP;
  };

  for (const auto& obligation : output.issued.obligations) {
    if (obligation.goal == obligation_day::GoalKind::Pickup)
      supply[{obligation.actor, static_cast<int>(obligation.item)}] =
          obligation.tile;
    const bool crop_goal =
        obligation.goal == obligation_day::GoalKind::CropReady &&
        is_crop(obligation.item);
    const bool animal_goal = obligation.goal == obligation_day::GoalKind::Place &&
        is_animal(obligation.item);
    if (!crop_goal && !animal_goal) continue;

    auto& candidate = candidates[key(obligation.tile)];
    if (obligation.source_step >= candidate.source_step) {
      candidate.source_step = obligation.source_step;
      candidate.target.id = obligation.id;
      candidate.target.tile = obligation.tile;
      candidate.target.kind = crop_goal ? state_target::PlotTargetKind::Crop
                                        : state_target::PlotTargetKind::Animal;
      candidate.target.item = obligation.item;
      candidate.target.actor = obligation.actor;
      candidate.target.maintain = false;
      candidate.target.feed = false;
      candidate.target.care = false;
    }
    candidate.target.value = std::max(
        candidate.target.value, static_cast<double>(obligation.priority));
  }

  output.targets.reserve(candidates.size());
  for (auto& [tile, candidate] : candidates) {
    (void)tile;
    auto& target = candidate.target;
    if (target.kind == state_target::PlotTargetKind::Animal) {
      for (const auto& obligation : output.issued.obligations) {
        if (key(obligation.tile) != key(target.tile) ||
            obligation.item != target.item ||
            obligation.actor != target.actor)
          continue;
        target.feed = target.feed ||
            obligation.goal == obligation_day::GoalKind::Feed;
        target.care = target.care ||
            obligation.goal == obligation_day::GoalKind::Care;
      }
      const auto animal_supply =
          supply.find({target.actor, static_cast<int>(target.item)});
      const auto wheat_supply =
          supply.find({target.actor, static_cast<int>(fastkag::Item::WHEAT)});
      if (animal_supply != supply.end())
        target.supply_position = animal_supply->second;
      else if (wheat_supply != supply.end())
        target.supply_position = wheat_supply->second;
    }
    output.targets.push_back(target);
  }

  const int deadline = std::min(
      (request.day_start->day() + 1) *
              request.day_start->config().turns_per_day - 1,
      request.day_start->config().episode_steps - 2);
  output.compiled = state_target::compile_state_targets(
      *request.day_start, request.player, request.day_start->step_count(),
      deadline, output.targets);
  obligation_day::DayPlanRequest plan_request{
      request.day_start, request.player, request.issuer_generation,
      output.issued.moves, output.compiled.obligations};
  output.admission = obligation_day::atomic_plan_day(
      plan_request, state_target::atomic_groups(output.compiled));
  const bool all_groups_admitted =
      output.admission.groups.size() == output.compiled.groups.size() &&
      std::all_of(output.admission.groups.begin(), output.admission.groups.end(),
                  [](const auto& group) { return group.admitted; });
  const bool no_semantic_unsupported = std::all_of(
      output.issued.unsupported.begin(), output.issued.unsupported.end(),
      [](const auto& unsupported) {
        return unsupported.reason ==
            day_start_issuer::UnsupportedReason::ActorUnavailableAtDayStart;
      });
  output.takeover_eligible = no_semantic_unsupported &&
      output.compiled.diagnostics.empty() && output.admission.planned() &&
      all_groups_admitted;
  return output;
}

MovementSuffixAuthority verify_movement_suffix_authority(
    int player, int day, std::uint64_t issuer_generation, int current_step,
    std::span<const obligation_day::MoveSourceToken> issued_moves,
    const FinalProviderMovementCommitment* commitment) {
  MovementSuffixAuthority output;
  if (!commitment) return output;
  if (player < 0 || player >= 2 || day < 0 || issuer_generation == 0 ||
      current_step / 24 != day || commitment->player != player ||
      commitment->day != day ||
      commitment->issuer_generation != issuer_generation ||
      commitment->issued_step != current_step) {
    output.failure = MovementAuthorityFailure::IdentityMismatch;
    return output;
  }
  if (commitment->content_hash !=
      fastkag::native_movement_commitment_hash(*commitment)) {
    output.failure = MovementAuthorityFailure::HashMismatch;
    return output;
  }
  if (commitment->revoked) {
    output.failure = MovementAuthorityFailure::Revoked;
    return output;
  }
  const int day_end = day * 24 + 23;
  if (!commitment->remaining_day_complete ||
      commitment->immutable_through_step != day_end) {
    output.failure = MovementAuthorityFailure::IncompleteHorizon;
    return output;
  }
  std::vector<obligation_day::MoveSourceToken> expected;
  for (const auto& move : issued_moves) {
    if (move.source_step >= current_step) expected.push_back(move);
  }
  output.checked_remaining_moves = static_cast<int>(expected.size());
  std::map<int, std::vector<const obligation_day::MoveSourceToken*>>
      expected_by_actor;
  std::map<int, std::vector<const fastkag::NativeMovementToken*>>
      committed_by_actor;
  std::set<std::pair<int, int>> keys;
  for (const auto& move : expected) expected_by_actor[move.actor].push_back(&move);
  for (const auto& move : commitment->moves) {
    if (move.source_step / 24 != day || move.source_step < current_step ||
        move.source_step > day_end) {
      output.failure = MovementAuthorityFailure::CrossDayToken;
      return output;
    }
    if (!keys.insert({move.actor, move.source_step}).second) {
      output.failure = MovementAuthorityFailure::DuplicateToken;
      return output;
    }
    auto& actor_moves = committed_by_actor[move.actor];
    if (!actor_moves.empty() &&
        actor_moves.back()->source_step >= move.source_step) {
      output.failure = MovementAuthorityFailure::TokenOrderMismatch;
      return output;
    }
    actor_moves.push_back(&move);
  }
  if (expected.size() != commitment->moves.size()) {
    output.failure = MovementAuthorityFailure::TokenMismatch;
    return output;
  }
  if (expected_by_actor.size() != committed_by_actor.size()) {
    output.failure = MovementAuthorityFailure::TokenMismatch;
    return output;
  }
  for (const auto& [actor, left_moves] : expected_by_actor) {
    const auto found = committed_by_actor.find(actor);
    if (found == committed_by_actor.end() ||
        left_moves.size() != found->second.size()) {
      output.failure = MovementAuthorityFailure::TokenMismatch;
      return output;
    }
    for (std::size_t index = 0; index < left_moves.size(); ++index) {
      const auto& left = *left_moves[index];
      const auto& right = *found->second[index];
      if (left.source_step != right.source_step ||
          left.action.op != right.action.op ||
          left.action.item != right.action.item ||
          left.action.quantity != right.action.quantity) {
        output.failure = MovementAuthorityFailure::TokenMismatch;
        return output;
      }
    }
  }
  for (const auto& move : commitment->moves) {
    if (move.action.op != fastkag::Op::NORTH &&
        move.action.op != fastkag::Op::SOUTH &&
        move.action.op != fastkag::Op::EAST &&
        move.action.op != fastkag::Op::WEST) {
      output.failure = MovementAuthorityFailure::TokenMismatch;
      return output;
    }
  }
  output.authorized = true;
  output.production_slot_rewrite_allowed = true;
  output.failure = MovementAuthorityFailure::None;
  return output;
}

MissingLineageCause classify_missing_lineage(
    const MissingLineageContext& context) noexcept {
  if (!context.valid_actor) return MissingLineageCause::ActorUnavailable;
  if (!context.present_at_day_start) {
    return MissingLineageCause::ActorUnavailableAtDayStartOrHiredMidday;
  }
  if (context.residual_route_position_drift &&
      context.has_prior_exact_plant) {
    return MissingLineageCause::ResidualMoveDrift;
  }
  if (context.dynamic_actor_overlay && context.has_prior_exact_plant) {
    return MissingLineageCause::ActiveDynamicMove;
  }
  if (context.actor > 0 && !context.has_prior_exact_plant) {
    return MissingLineageCause::HiredActorNoPriorPlant;
  }
  if (context.has_prior_exact_plant) {
    return MissingLineageCause::PlannedRouteTileWithoutPriorLineage;
  }
  if (context.current_proof ==
      typed_intent::Proof::DynamicSuffixUncertified) {
    return MissingLineageCause::SuffixAuthorityOnly;
  }
  return MissingLineageCause::NoPriorExactPlant;
}

const char* missing_lineage_cause_name(MissingLineageCause cause) noexcept {
  switch (cause) {
    case MissingLineageCause::ActorUnavailable:
      return "actor_unavailable";
    case MissingLineageCause::ActorUnavailableAtDayStartOrHiredMidday:
      return "actor_unavailable_at_day_start_or_hired_midday";
    case MissingLineageCause::ResidualMoveDrift:
      return "lineage_tile_mismatch_from_residual_move_drift";
    case MissingLineageCause::ActiveDynamicMove:
      return "dynamic_move_lineage_tile_mismatch";
    case MissingLineageCause::HiredActorNoPriorPlant:
      return "hired_actor_no_prior_exact_plant";
    case MissingLineageCause::PlannedRouteTileWithoutPriorLineage:
      return "planned_route_tile_without_prior_lineage";
    case MissingLineageCause::SuffixAuthorityOnly:
      return "suffix_authority_only_no_lineage";
    case MissingLineageCause::NoPriorExactPlant:
      return "no_prior_exact_plant_lineage";
  }
  return "unknown";
}

const char* typed_proof_name(TypedProof proof) noexcept {
  switch (proof) {
    case TypedProof::CurrentIssuer:
      return "current_issuer";
    case TypedProof::PersistentCropLineage:
      return "persistent_crop_lineage";
    case TypedProof::Unresolved:
      return "unresolved";
  }
  return "unknown";
}

const char* movement_authority_failure_name(
    MovementAuthorityFailure failure) noexcept {
  switch (failure) {
    case MovementAuthorityFailure::None:
      return "none";
    case MovementAuthorityFailure::MissingImmutableCommitment:
      return "missing_immutable_commitment";
    case MovementAuthorityFailure::IdentityMismatch:
      return "identity_mismatch";
    case MovementAuthorityFailure::HashMismatch:
      return "hash_mismatch";
    case MovementAuthorityFailure::Revoked:
      return "revoked";
    case MovementAuthorityFailure::IncompleteHorizon:
      return "incomplete_horizon";
    case MovementAuthorityFailure::CrossDayToken:
      return "cross_day_token";
    case MovementAuthorityFailure::DuplicateToken:
      return "duplicate_token";
    case MovementAuthorityFailure::TokenOrderMismatch:
      return "token_order_mismatch";
    case MovementAuthorityFailure::TokenMismatch:
      return "token_mismatch";
  }
  return "unknown";
}

}  // namespace g001::repair_pipeline_shadow
