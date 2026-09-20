#include "g001_repair_pipeline_shadow.hpp"
#include "movement_plan_owner.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace shadow = g001::repair_pipeline_shadow;
namespace issuer = g001::day_start_issuer;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
using fastkag::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void persistent_lineage_completes_itemless_weed_work() {
  Simulator simulator({}, 7001);
  const auto tile = simulator.farms()[0].farmer;
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  farm.tiles[tile.y * simulator.config().board_size + tile.x].kind =
      TileKind::WEED;
  issuer::PersistentRouteIntentRegistry lineage;
  check(lineage.record_exact_source(
            0, 0, tile, {Op::PLANT, Item::STRAWBERRY, 1}),
        "fixture lineage was rejected");
  const std::array<Action, 1> units{{Op::WATER, Item::NONE, 1}};
  const auto result = shadow::observe_final_current_units(
      {&simulator, 0, 101, units, {}, true, &lineage});
  check(result.crops.size() == 1 &&
            result.crops[0].desired == Item::STRAWBERRY &&
            result.persistent_crop_recoveries == 1 &&
            result.evidence.size() == 1 && result.evidence[0].exact &&
            result.evidence[0].proof ==
                shadow::TypedProof::PersistentCropLineage,
        "persistent lineage did not complete item-less weed WATER");
}

void final_manifest_is_the_only_lineage_writer() {
  Simulator simulator({}, 7002);
  issuer::PersistentRouteIntentRegistry lineage;
  const std::array<Action, 1> plant{{Op::PLANT, Item::MELON, 1}};
  const auto planted = shadow::observe_final_current_units(
      {&simulator, 0, 102, plant, {}, true, &lineage});
  check(planted.crop_lineage_records == 1 &&
            lineage.crop_at(simulator.farms()[0].farmer)->item == Item::MELON,
        "exact final PLANT did not write lineage");

  const std::array<Action, 1> non_animal_place{
      {Op::PLACE, Item::FERTILIZER, 1}};
  const auto placed = shadow::observe_final_current_units(
      {&simulator, 0, 103, non_animal_place, {}, true, &lineage});
  check(placed.animal_lineage_records == 0 && placed.animals.empty(),
        "non-animal PLACE wrote animal lineage");
}

void day_start_pipeline_plans_and_verifies_without_future_binding() {
  Simulator simulator({}, 7003);
  const auto tile = simulator.farms()[0].farmer;
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  auto& private_state =
      const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
  farm.tiles[tile.y * simulator.config().board_size + tile.x].kind =
      TileKind::WEED;
  private_state.seeds[static_cast<int>(Item::STRAWBERRY)] = 1;
  issuer::PersistentRouteIntentRegistry lineage;
  check(lineage.record_exact_source(
            0, 0, tile, {Op::PLANT, Item::STRAWBERRY, 1}),
        "day lineage rejected");
  std::vector<PlayerAction> tape(24);
  for (auto& action : tape) action.units.push_back({});
  tape[0].units[0] = {Op::WATER, Item::NONE, 1};
  tape[4].units[0] = {Op::WEST, Item::NONE, 1};
  tape[5].units[0] = {Op::EAST, Item::NONE, 1};
  const auto result = shadow::plan_and_verify_day_shadow(
      {&simulator, &tape, 0, 104, &lineage});
  check(result.issued.issued() && result.issued.unsupported.empty() &&
            result.issued.crop_owner_obligations.size() == 1 &&
            result.plan.planned() && result.verification.valid &&
            result.plan.certificate->move_replays.size() == 2 &&
            result.current_evidence_temporally_unavailable &&
            result.locally_full_certificate_eligible && !result.partial_plan,
        "day-start shadow pipeline failed certificate closure");
}

void unsupported_source_can_only_form_a_partial_plan() {
  Simulator simulator({}, 7004);
  std::vector<PlayerAction> tape(24);
  for (auto& action : tape) action.units.push_back({});
  tape[0].units[0] = {Op::FERTILIZE, Item::FERTILIZER, 1};
  issuer::PersistentRouteIntentRegistry lineage;
  const auto result = shadow::plan_and_verify_day_shadow(
      {&simulator, &tape, 0, 105, &lineage});
  check(result.issued.issued() && result.issued.unsupported.size() == 1 &&
            result.plan.planned() && result.verification.valid &&
            !result.locally_full_certificate_eligible && result.partial_plan,
        "omitted unsupported source counted as complete closure");
}

void unified_shadow_recompiles_macro_intent_from_real_state() {
  Simulator simulator({}, 7005);
  const auto tile = simulator.farms()[0].farmer;
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  auto& private_state =
      const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
  farm.tiles[tile.y * simulator.config().board_size + tile.x].kind =
      TileKind::WEED;
  private_state.seeds[static_cast<int>(Item::STRAWBERRY)] = 1;
  issuer::PersistentRouteIntentRegistry lineage;
  check(lineage.record_exact_source(
            0, 0, tile, {Op::PLANT, Item::STRAWBERRY, 1}),
        "unified fixture lineage was rejected");
  std::vector<PlayerAction> tape(24);
  for (auto& action : tape) action.units.push_back({});
  tape[0].units[0] = {Op::WATER, Item::NONE, 1};
  tape[4].units[0] = {Op::WEST, Item::NONE, 1};
  tape[5].units[0] = {Op::EAST, Item::NONE, 1};
  const auto result = shadow::plan_unified_day_shadow(
      {&simulator, &tape, 0, 106, &lineage});
  check(result.takeover_eligible && result.targets.size() == 1 &&
            result.targets[0].item == Item::STRAWBERRY &&
            result.admission.planned() && result.admission.groups.size() == 1 &&
            result.admission.groups[0].admitted &&
            result.admission.verified_plan.manifest[0][0].op == Op::DIG &&
            result.admission.verified_plan.manifest[0][1].op == Op::PLANT &&
            result.admission.verified_plan.manifest[0][2].op == Op::WATER &&
            result.admission.verified_plan.certificate->move_replays.size() == 2,
        "unified shadow did not preserve MOVE while rebuilding weed lifecycle");
}

void movement_suffix_requires_exact_final_provider_commitment() {
  const std::array<g001::obligation_day::MoveSourceToken, 2> moves{{
      {0, 25, {Op::WEST, Item::NONE, 1}},
      {0, 28, {Op::EAST, Item::NONE, 1}},
  }};
  const auto missing = shadow::verify_movement_suffix_authority(
      0, 1, 700, 24, moves, nullptr);
  check(!missing.authorized && !missing.production_slot_rewrite_allowed,
        "missing runtime commitment authorized MOVE rewrite");
  shadow::FinalProviderMovementCommitment commitment;
  commitment.player = 0;
  commitment.day = 1;
  commitment.issuer_generation = 700;
  commitment.issued_step = 24;
  commitment.immutable_through_step = 47;
  commitment.remaining_day_complete = true;
  for (const auto& move : moves) {
    commitment.moves.push_back({move.actor, move.source_step, move.action});
  }
  const auto rehash = [&]() {
    commitment.content_hash =
        fastkag::native_movement_commitment_hash(commitment);
  };
  rehash();
  const auto exact = shadow::verify_movement_suffix_authority(
      0, 1, 700, 24, moves, &commitment);
  check(exact.authorized && exact.production_slot_rewrite_allowed &&
            exact.checked_remaining_moves == 2,
        "exact movement commitment was rejected");
  commitment.moves[1].action.op = Op::WEST;
  rehash();
  const auto mismatch = shadow::verify_movement_suffix_authority(
      0, 1, 700, 24, moves, &commitment);
  check(!mismatch.authorized && !mismatch.production_slot_rewrite_allowed &&
            mismatch.failure ==
                shadow::MovementAuthorityFailure::TokenMismatch,
        "changed final-provider MOVE was authorized");

  commitment.moves[1].action.op = Op::EAST;
  rehash();
  commitment.content_hash ^= 1;
  check(shadow::verify_movement_suffix_authority(
            0, 1, 700, 24, moves, &commitment).failure ==
            shadow::MovementAuthorityFailure::HashMismatch,
        "tampered commitment hash was accepted");

  rehash();
  commitment.moves.push_back(commitment.moves.back());
  rehash();
  check(shadow::verify_movement_suffix_authority(
            0, 1, 700, 24, moves, &commitment).failure ==
            shadow::MovementAuthorityFailure::DuplicateToken,
        "duplicate commitment token was accepted");

  commitment.moves.pop_back();
  commitment.moves.pop_back();
  rehash();
  check(shadow::verify_movement_suffix_authority(
            0, 1, 700, 24, moves, &commitment).failure ==
            shadow::MovementAuthorityFailure::TokenMismatch,
        "omitted commitment token was accepted");

  commitment.moves = {{0, 48, {Op::EAST, Item::NONE, 1}},
                      {0, 25, {Op::WEST, Item::NONE, 1}}};
  rehash();
  check(shadow::verify_movement_suffix_authority(
            0, 1, 700, 24, moves, &commitment).failure ==
            shadow::MovementAuthorityFailure::CrossDayToken,
        "midnight-crossing commitment token was accepted");

  commitment.moves = {{0, 28, {Op::EAST, Item::NONE, 1}},
                      {0, 25, {Op::WEST, Item::NONE, 1}}};
  rehash();
  check(shadow::verify_movement_suffix_authority(
            0, 1, 700, 24, moves, &commitment).failure ==
            shadow::MovementAuthorityFailure::TokenOrderMismatch,
        "per-actor reordered commitment was accepted");

  commitment.moves = {{0, 25, {Op::WEST, Item::NONE, 1}},
                      {0, 28, {Op::EAST, Item::NONE, 1}}};
  commitment.remaining_day_complete = false;
  commitment.immutable_through_step = 24;
  rehash();
  check(shadow::verify_movement_suffix_authority(
            0, 1, 700, 24, moves, &commitment).failure ==
            shadow::MovementAuthorityFailure::IncompleteHorizon,
        "current-only commitment authorized a delayed MOVE");

  commitment.remaining_day_complete = true;
  commitment.immutable_through_step = 47;
  commitment.revoked = true;
  rehash();
  check(shadow::verify_movement_suffix_authority(
            0, 1, 700, 24, moves, &commitment).failure ==
            shadow::MovementAuthorityFailure::Revoked,
        "revoked commitment survived a later weed-overlay attack");
}

void missing_lineage_causes_are_mutually_exclusive() {
  using Cause = shadow::MissingLineageCause;
  using Proof = g001::typed_intent::Proof;
  check(shadow::classify_missing_lineage(
            {true, false, false, false, false, 3,
             Proof::LiveTileHasNoCrop}) ==
            Cause::ActorUnavailableAtDayStartOrHiredMidday,
        "midday hire cause was not dominant");
  check(shadow::classify_missing_lineage(
            {true, true, true, false, true, 0,
             Proof::LiveTileHasNoCrop}) == Cause::ResidualMoveDrift,
        "residual MOVE drift was not separated");
  check(shadow::classify_missing_lineage(
            {true, true, false, true, true, 0,
             Proof::DynamicSuffixUncertified}) == Cause::ActiveDynamicMove,
        "active dynamic MOVE mismatch was not separated");
  check(shadow::classify_missing_lineage(
            {true, true, false, false, false, 0,
             Proof::DynamicSuffixUncertified}) ==
            Cause::SuffixAuthorityOnly,
        "suffix-only gap was not separated");
  check(shadow::classify_missing_lineage(
            {true, true, false, false, false, 0,
             Proof::LiveTileHasNoCrop}) == Cause::NoPriorExactPlant,
        "no-prior-lineage gap was not separated");
}

void synthetic_movement_owner_weed_move_collision() {
  Simulator simulator({}, 970017);
  for (int step = 0; step < 168; ++step) {
    std::array<PlayerAction, 2> actions;
    actions[0].units.push_back({}); actions[1].units.push_back({});
    simulator.step(actions);
  }
  auto& tile = const_cast<fastkag::Farm&>(simulator.farms()[1]).tiles[
      simulator.farms()[1].farmer.y * simulator.config().board_size +
      simulator.farms()[1].farmer.x];
  tile.kind = fastkag::TileKind::WEED;
  auto& seeds = const_cast<fastkag::PrivateState&>(simulator.privates()[1]).seeds;
  seeds[static_cast<int>(Item::WHEAT)] = 1;
  g001::obligation_day::ProductionObligation weed{
      970017187, 0, simulator.farms()[1].farmer,
      g001::obligation_day::GoalKind::CropReady, Item::WHEAT, 1, {}, {},
      183, 191, 1000, true, true};
  const std::vector<g001::obligation_day::MoveSourceToken> moves{
      {0, 187, {Op::SOUTH, Item::NONE, 1}}};
  g001::obligation_day::DayPlanRequest day{
      &simulator, 1, 970017187, moves, {weed}};
  const auto owned = g001::movement_plan_owner::plan({day, false});
  check(owned.authorized && owned.plan.planned() &&
            owned.plan.certificate->move_replays.size() == 1 &&
            owned.plan.certificate->move_replays[0].action.op == Op::SOUTH &&
            owned.plan.certificate->move_replays[0].source_step == 187,
        "step187 SOUTH was swallowed by weed work");
  check(!g001::movement_plan_owner::plan({day, true}).authorized,
        "unowned overlay did not fail closed");

  day.obligations.clear();
  const auto normal = g001::movement_plan_owner::plan({day, false});
  check(normal.authorized && normal.plan.manifest[0][19].op == Op::SOUTH &&
            normal.plan.sources[0][19].source_step == 187,
        "normal default-off MOVE parity changed");
}

}  // namespace

int main() {
  try {
    persistent_lineage_completes_itemless_weed_work();
    final_manifest_is_the_only_lineage_writer();
    day_start_pipeline_plans_and_verifies_without_future_binding();
    unsupported_source_can_only_form_a_partial_plan();
    unified_shadow_recompiles_macro_intent_from_real_state();
    movement_suffix_requires_exact_final_provider_commitment();
    synthetic_movement_owner_weed_move_collision();
    missing_lineage_causes_are_mutually_exclusive();
    std::cout << "g001_repair_pipeline_shadow_tests: 8 groups passed\n";
  } catch (const std::exception& error) {
    std::cerr << "g001_repair_pipeline_shadow_tests: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
