#include "two_phase_route_cursor.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace core = modular_agent_core;
namespace bridge = g001::failure_debt::modular_bridge;
namespace two = g001::repair_audit::two_phase;
namespace po = production_obligation;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

Action action(Op operation, Item item = Item::NONE, int quantity = 1) {
  return {operation, item, quantity};
}

two::LifecycleStage unit_stage(std::uint64_t id, Op operation,
                               Item item = Item::NONE,
                               bool first_yield = false) {
  return {{core::Owner::Repair, id}, action(operation, item),
          two::StagePhase::Unit, true, first_yield,
          "fixture:unit:" + std::to_string(id)};
}

two::LifecycleContract crop_contract() {
  two::LifecycleContract result;
  result.transaction_id = 7001;
  result.kind = g001::failure_debt::LifecycleKind::Crop;
  result.stages = {
      unit_stage(101, Op::PLANT, Item::WHEAT),
      unit_stage(102, Op::WATER),
      unit_stage(103, Op::HARVEST),
      unit_stage(104, Op::PLANT, Item::WHEAT),
      unit_stage(105, Op::WATER),
      unit_stage(106, Op::HARVEST, Item::NONE, true),
  };
  return result;
}

two::LifecycleContract animal_contract() {
  two::LifecycleContract result;
  result.transaction_id = 8001;
  result.kind = g001::failure_debt::LifecycleKind::Animal;
  result.stages = {
      {{core::Owner::Repair, 201}, action(Op::BUY_ANIMAL, Item::COW),
       two::StagePhase::Market, false, false, "fixture:buy-cow"},
      unit_stage(202, Op::PICKUP, Item::COW),
      unit_stage(203, Op::PLACE, Item::COW),
      unit_stage(204, Op::FEED, Item::WHEAT),
      unit_stage(205, Op::CARE),
      unit_stage(206, Op::HARVEST, Item::NONE, true),
  };
  return result;
}

void install_keys(core::ComposeInput& input, int step) {
  input.movement.future_visit_calendar_hash = 0x900d;
  input.movement.movement_hash = core::hash_movement_payload(input.movement.actors);
  input.production.movement_hash = input.movement.movement_hash;
  input.production.obligation_dag_hash =
      core::hash_production_dag(input.production.obligation_dag);
  input.repair.movement_hash = input.movement.movement_hash;
  input.repair.base_obligation_dag_hash =
      input.production.obligation_dag_hash;

  core::SharedKey key;
  key.episode_nonce = 77;
  key.step = step;
  key.observation_fingerprint = 0xabcde + static_cast<std::uint64_t>(step);
  key.route_plan_hash = core::hash_route_plan(
      input.movement.movement_hash,
      input.movement.future_visit_calendar_hash);
  input.production.key = key;
  key.production_plan_hash = core::hash_production_payload(input.production);
  input.snapshot.key = key;
  input.movement.key = key;
  input.production.key = key;
  input.repair.key = key;
  input.trade.key = key;
}

core::ComposeInput base_input(int step) {
  core::ComposeInput input;
  input.snapshot.actor_count = 1;
  input.snapshot.actor_positions = {{1, 1}};
  input.snapshot.maximum_market_slots = 10;
  input.snapshot.shed_capacity = 100;
  input.snapshot.pre_market_own.money = 10000;
  core::MarketStressScenario scenario;
  scenario.scenario_id = 1;
  input.snapshot.stress_scenarios.push_back(scenario);
  input.movement.actors.push_back(
      {0, {1, 1}, action(Op::PASS), 9001});
  // A typed dynamic overlay/base owner. Omitting RepairDelta leaves this PASS;
  // selecting RepairDelta replaces it with the lifecycle action.
  input.production.current_units.push_back(
      {0, action(Op::PASS), {core::Owner::Production, 9002}});
  input.production.obligation_dag.feasible = true;
  install_keys(input, step);
  return input;
}

bridge::RepairDeltaProposal bridge_proposal(
    const two::LifecycleContract& contract, const two::LifecycleStage& stage,
    const core::ComposeInput& input) {
  bridge::RepairDeltaProposal result;
  result.status = bridge::ProposalStatus::Accepted;
  result.reason = "fixture accepted whole-continuation certificate";
  result.delta.key = input.snapshot.key;
  result.delta.movement_hash = input.movement.movement_hash;
  result.delta.base_obligation_dag_hash = input.production.obligation_dag_hash;
  result.delta.transaction_id = contract.transaction_id;
  if (stage.phase == two::StagePhase::Unit) {
    result.delta.unit_patches.push_back(
        {0, action(Op::PASS), {core::Owner::Production, 9002}, stage.action,
         stage.intent});
  } else {
    result.delta.hard_market.push_back(
        {stage.intent, stage.action, 100, -1});
  }
  return result;
}

struct Prepared {
  core::ComposeInput input;
  bridge::RepairDeltaProposal bridge;
  two::Proposal proposal;
};

Prepared prepare(two::Coordinator& coordinator,
                 const two::LifecycleContract& contract,
                 bool select_repair, int step) {
  auto input = base_input(step);
  const auto& stage = contract.stages[coordinator.snapshot().next_stage];
  auto modular = bridge_proposal(contract, stage, input);
  auto proposal = coordinator.propose(modular);
  require(proposal.accepted(), proposal.reason);
  if (select_repair) {
    input.repair = proposal.delta;
  } else {
    input.repair.transaction_id = 0;
    input.repair.unit_patches.clear();
    input.repair.hard_market.clear();
  }
  const Action final_unit = select_repair && stage.phase == two::StagePhase::Unit
      ? stage.action : action(Op::PASS);
  input.snapshot.projected_unit_action_hash = core::hash_unit_actions(
      std::array<Action, 1>{final_unit});
  return {std::move(input), std::move(modular), std::move(proposal)};
}

core::ExecutionReceipt receipt(const core::ExecutionManifest& manifest,
                               const two::LifecycleStage& stage,
                               bool confirmed) {
  core::ReceiptEvidence evidence;
  evidence.key = manifest.key;
  evidence.submitted_action_hash = manifest.action_hash;
  evidence.next_observation_fingerprint = 0xbeef + manifest.key.step;
  evidence.market_fill.resize(manifest.market.size());
  evidence.market_cash.resize(manifest.market.size());
  evidence.unit_effect_confirmed.assign(manifest.units.size(), false);
  if (stage.phase == two::StagePhase::Unit) {
    for (std::size_t index = 0; index < manifest.units.size(); ++index)
      if (manifest.units[index].owner == stage.intent)
        evidence.unit_effect_confirmed[index] = confirmed;
  } else {
    for (std::size_t index = 0; index < manifest.market.size(); ++index)
      if (manifest.market[index].owner == stage.intent) {
        const int fill = confirmed ? stage.action.quantity : 0;
        evidence.market_fill[index] = {fill, fill, fill};
      }
  }
  const auto result = core::make_execution_receipt(manifest, evidence);
  require(result.accepted, result.reason);
  return result.receipt;
}

two::CommitStatus select_and_observe(two::Coordinator& coordinator,
                                     const two::LifecycleContract& contract,
                                     int step, bool confirmed = true) {
  auto prepared = prepare(coordinator, contract, true, step);
  const auto composed = core::compose(prepared.input);
  require(composed.accepted(), composed.reason);
  require(coordinator.stage_final(prepared.proposal, composed.manifest) ==
              two::FinalStatus::Selected,
          "modular final composer must select exact repair intent");
  const auto observed = receipt(composed.manifest, prepared.proposal.expected,
                                confirmed);
  return coordinator.observe(observed);
}

void proposal_is_side_effect_free_and_override_keeps_source() {
  const auto contract = crop_contract();
  two::Coordinator coordinator(contract, 41);
  const auto before = coordinator.snapshot();
  auto prepared = prepare(coordinator, contract, false, 7);
  require(coordinator.snapshot() == before,
          "proposal must not advance cursor or lifecycle audit");
  const auto composed = core::compose(prepared.input);
  require(composed.accepted(), composed.reason);
  require(coordinator.stage_final(prepared.proposal, composed.manifest) ==
              two::FinalStatus::Overridden,
          "dynamic final owner must be observable as override");
  const auto after_override = coordinator.snapshot();
  require(after_override.source_cursor == 41 && after_override.next_stage == 0 &&
              !after_override.awaiting_observation &&
              after_override.audit.plants == 0 &&
              after_override.audit.overlay_overrides == 1,
          "override retains debt/source and never counts PLANT");

  prepared = prepare(coordinator, contract, true, 8);
  const auto selected = core::compose(prepared.input);
  require(selected.accepted(), selected.reason);
  require(coordinator.stage_final(prepared.proposal, selected.manifest) ==
              two::FinalStatus::Selected,
          "repair must stage only after final selection");
  require(coordinator.snapshot().source_cursor == 41 &&
              coordinator.snapshot().audit.plants == 0,
          "final selection still does not commit before observation");
  require(coordinator.observe(
              receipt(selected.manifest, prepared.proposal.expected, false)) ==
              two::CommitStatus::EffectNotConfirmed,
          "failed observed effect must reopen the same stage");
  const auto after_failure = coordinator.snapshot();
  require(after_failure.source_cursor == 41 && after_failure.next_stage == 0 &&
              after_failure.audit.plants == 0 &&
              after_failure.audit.effect_failures == 1,
          "failed effect retains cursor/debt and no productive audit");
}

void crop_completes_only_after_replant_first_yield_receipt() {
  const auto contract = crop_contract();
  two::Coordinator coordinator(contract, 50);
  for (int index = 0; index < 3; ++index)
    require(select_and_observe(coordinator, contract, 20 + index) ==
                two::CommitStatus::Committed,
            "first crop cycle stage must commit");
  auto middle = coordinator.snapshot();
  require(!middle.completed && middle.next_stage == 3 &&
              middle.audit.harvest_effects == 1 &&
              middle.audit.completed_transactions == 0,
          "first HARVEST effect is not whole crop transaction completion");

  require(select_and_observe(coordinator, contract, 23) ==
              two::CommitStatus::Committed,
          "replant commits from observation");
  require(select_and_observe(coordinator, contract, 24) ==
              two::CommitStatus::Committed,
          "replant water commits from observation");
  auto final_prepared = prepare(coordinator, contract, true, 25);
  const auto final_composed = core::compose(final_prepared.input);
  require(final_composed.accepted(), final_composed.reason);
  require(coordinator.stage_final(final_prepared.proposal,
                                  final_composed.manifest) ==
              two::FinalStatus::Selected,
          "first-yield action selected");
  require(!coordinator.snapshot().completed,
          "emitted terminal HARVEST must not complete before observation");
  require(coordinator.observe(receipt(final_composed.manifest,
                                      final_prepared.proposal.expected, true)) ==
              two::CommitStatus::Committed,
          "verified replanted first yield commits");
  const auto done = coordinator.snapshot();
  require(done.completed && done.source_cursor == 56 &&
              done.audit.plants == 2 && done.audit.waters == 2 &&
              done.audit.harvest_effects == 2 &&
              done.audit.completed_transactions == 1,
          "crop completes exactly at full continuation first yield");
}

void animal_fill_is_not_completion_and_requires_first_yield() {
  const auto contract = animal_contract();
  two::Coordinator coordinator(contract, 70);
  require(select_and_observe(coordinator, contract, 30, false) ==
              two::CommitStatus::EffectNotConfirmed,
          "zero BUY fill retains animal acquisition debt");
  require(coordinator.snapshot().source_cursor == 70 &&
              coordinator.snapshot().next_stage == 0,
          "market failure does not consume unit source");
  require(select_and_observe(coordinator, contract, 31, true) ==
              two::CommitStatus::Committed,
          "full BUY fill confirms acquisition stage");
  require(!coordinator.snapshot().completed &&
              coordinator.snapshot().audit.acquisitions == 1 &&
              coordinator.snapshot().source_cursor == 70,
          "BUY fill alone is not animal transaction completion");
  for (int index = 1; index < 5; ++index)
    require(select_and_observe(coordinator, contract, 31 + index) ==
                two::CommitStatus::Committed,
            "animal continuation stage must commit");
  const auto before_yield = coordinator.snapshot();
  require(!before_yield.completed && before_yield.audit.places == 1 &&
              before_yield.audit.feeds == 1 && before_yield.audit.cares == 1 &&
              before_yield.audit.harvest_effects == 0,
          "PLACE/FEED/CARE still await first yield");
  require(select_and_observe(coordinator, contract, 36) ==
              two::CommitStatus::Committed,
          "animal first yield commits");
  const auto done = coordinator.snapshot();
  require(done.completed && done.source_cursor == 75 &&
              done.audit.pickups == 1 && done.audit.places == 1 &&
              done.audit.harvest_effects == 1 &&
              done.audit.completed_transactions == 1,
          "animal completes only after full continuation first yield");
}

void incomplete_contracts_fail_closed() {
  auto crop = crop_contract();
  crop.stages.resize(3);
  crop.stages.back().first_yield = true;
  bool rejected = false;
  try {
    two::Coordinator ignored(std::move(crop));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "single-harvest crop contract must fail closed");

  auto animal = animal_contract();
  animal.stages.erase(animal.stages.begin() + 4);
  rejected = false;
  try {
    two::Coordinator ignored(std::move(animal));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "animal contract without CARE must fail closed");
}

}  // namespace

int main() {
  proposal_is_side_effect_free_and_override_keeps_source();
  crop_completes_only_after_replant_first_yield_receipt();
  animal_fill_is_not_completion_and_requires_first_yield();
  incomplete_contracts_fail_closed();
  std::cout << "two-phase RouteCursor transaction fixtures passed\n";
  return 0;
}
