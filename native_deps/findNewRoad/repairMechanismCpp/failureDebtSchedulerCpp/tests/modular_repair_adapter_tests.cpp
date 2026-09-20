#include "modular_repair_adapter.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
using namespace g001::failure_debt;
using namespace g001::failure_debt::modular_bridge;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

fastkag::Action action(fastkag::Op op, fastkag::Item item = fastkag::Item::NONE,
                       int quantity = 1) {
  return {op, item, quantity};
}

Debt debt(std::uint64_t id, DebtKind kind, int unit_index,
          fastkag::Action value, int actor, fastkag::Position position) {
  Debt result;
  result.id = id;
  result.transaction_id = 7;
  result.kind = kind;
  result.status = DebtStatus::Open;
  result.purchase_unit_index = unit_index;
  result.action = value;
  result.actor = actor;
  result.position = position;
  result.economic_value = 500;
  result.provenance = "fixture:debt:" + std::to_string(id);
  return result;
}

PlannedTurn turn(int step, int actor, fastkag::Position position,
                 fastkag::Op replacement) {
  PlannedTurn result;
  result.step = step;
  result.certified_actor_count = 2;
  result.actor_availability_certified = true;
  PlannedUnitSlot slot;
  slot.actor = actor;
  slot.position = position;
  slot.original = action(fastkag::Op::PASS);
  slot.absorbable = true;
  slot.legal_replacements[static_cast<std::size_t>(replacement)] = true;
  result.units.push_back(slot);
  return result;
}

UnitPatch patch(std::uint64_t debt_id, int step, int actor,
                fastkag::Action replacement) {
  return {step, actor, 0, action(fastkag::Op::PASS), replacement, 1, debt_id};
}

RepairDeltaRequest crop_request() {
  RepairDeltaRequest request;
  request.enabled = true;
  request.key = {modular_agent_core::kAbiVersion, 9, 44, 0xabc,
                 0x111, 0x222};
  request.movement_hash = 0x333;
  request.base_obligation_dag_hash = 0x444;
  request.turns_per_day = 24;
  request.episode_steps = 720;
  request.maximum_delay_days = 1;
  request.plan.accepted = true;
  request.plan.day_lifecycle_complete = true;
  request.plan.transaction_id = 7;
  request.plan.observation_fingerprint = request.key.observation_fingerprint;
  request.plan.market = {true, 44,
                         action(fastkag::Op::BUY_SEED,
                                fastkag::Item::CARROT),
                         false, -1};
  constexpr fastkag::Position tile{2, 2};
  request.debts = {
      debt(100, DebtKind::MarketPurchase, -1,
           action(fastkag::Op::BUY_SEED, fastkag::Item::CARROT), -1, {-1, -1}),
      debt(101, DebtKind::UnitAction, 0,
           action(fastkag::Op::PLANT, fastkag::Item::CARROT), 0, tile),
      debt(102, DebtKind::UnitAction, 0, action(fastkag::Op::WATER), 1, tile),
      debt(103, DebtKind::UnitAction, 0, action(fastkag::Op::HARVEST), 0, tile),
  };
  request.debts[1].dependencies = {100};
  request.debts[2].dependencies = {101};
  request.debts[3].dependencies = {102};
  request.plan.unit_schedule = {
      {101, 44, 0, 0, false},
      {102, 45, 1, 0, false},
      {103, 92, 0, 0, false},
  };
  request.plan.scheduled_debt_ids = {101, 102, 103};
  request.plan.units = {
      patch(101, 44, 0,
            action(fastkag::Op::PLANT, fastkag::Item::CARROT)),
      patch(102, 45, 1, action(fastkag::Op::WATER)),
      patch(103, 92, 0, action(fastkag::Op::HARVEST)),
  };
  request.plan_window.turns = {
      turn(44, 0, tile, fastkag::Op::PLANT),
      turn(45, 1, tile, fastkag::Op::WATER),
      turn(92, 0, tile, fastkag::Op::HARVEST),
  };
  request.current_units = {
      {0, 0, action(fastkag::Op::PASS),
       {modular_agent_core::Owner::Route, 700}},
  };
  TileContinuation continuation;
  continuation.transaction_id = 7;
  continuation.purchase_unit_index = 0;
  continuation.tile = tile;
  continuation.kind = LifecycleKind::Crop;
  continuation.provenance = "fixture:crop-tile-2-2";
  continuation.actions = {
      {101, 20, action(fastkag::Op::PLANT, fastkag::Item::CARROT), "plant"},
      {102, 21, action(fastkag::Op::WATER), "water"},
      {103, 68, action(fastkag::Op::HARVEST), "harvest"},
  };
  request.tile_continuations.push_back(std::move(continuation));
  return request;
}

void test_crop_whole_tile_shift_to_pure_delta() {
  const auto request = crop_request();
  const auto proposal = propose_repair_delta(request);
  require(proposal.status == ProposalStatus::Accepted, proposal.reason);
  require(proposal.unrepaired.empty(), "accepted proposal has no unrepaired debt");
  require(proposal.delta.key == request.key &&
              proposal.delta.movement_hash == request.movement_hash &&
              proposal.delta.base_obligation_dag_hash ==
                  request.base_obligation_dag_hash,
          "RepairDelta binds exact key/MOVE/DAG");
  require(proposal.delta.transaction_id == 7 &&
              proposal.delta.unit_patches.size() == 1 &&
              proposal.delta.hard_market.size() == 1,
          "only current unit and acquisition become current RepairDelta");
  require(proposal.delta.unit_patches[0].intent.owner ==
              modular_agent_core::Owner::Repair &&
              proposal.delta.unit_patches[0].intent.intent_id == 101,
          "unit proposal keeps stable debt intent");
  require(proposal.delta.hard_market[0].intent.intent_id == 100 &&
              proposal.delta.hard_market[0].obligation_node_id == -1,
          "market proposal is repair-owned, not a production DAG binding");
  require(proposal.shifted_tile_actions.size() == 3,
          "entire tile continuation certified");
  require(proposal.shifted_tile_actions[0].shifted_step == 44 &&
              proposal.shifted_tile_actions[1].shifted_step == 45 &&
              proposal.shifted_tile_actions[2].shifted_step == 92,
          "all downstream production shifted by identical 24 steps");
  for (const auto& shifted : proposal.shifted_tile_actions) {
    require(shifted.delay_steps == 24 && shifted.delay_days == 1,
            "delay budget measured in days");
    require(!is_movement(shifted.action.op), "continuation never contains MOVE");
  }

  auto shuffled = request;
  std::swap(shuffled.debts[1], shuffled.debts[3]);
  const auto reordered = propose_repair_delta(shuffled);
  require(reordered.status == ProposalStatus::Accepted &&
              reordered.shifted_tile_actions.size() == 3,
          "whole-chain proof follows dependency edges, not vector order");
}

void test_default_off_and_future_deferred() {
  auto request = crop_request();
  request.enabled = false;
  auto proposal = propose_repair_delta(request);
  require(proposal.status == ProposalStatus::Disabled &&
              proposal.delta.unit_patches.empty() &&
              proposal.delta.hard_market.empty() &&
              proposal.unrepaired.empty(),
          "adapter is default-off without pretending debt was repaired");

  request = crop_request();
  request.key.step = 43;
  request.key.observation_fingerprint = 0xdef;
  request.plan.observation_fingerprint = 0xdef;
  proposal = propose_repair_delta(request);
  require(proposal.status == ProposalStatus::Deferred && proposal.accepted() &&
              proposal.delta.transaction_id == 0 &&
              proposal.delta.unit_patches.empty() &&
              proposal.delta.hard_market.empty() &&
              proposal.shifted_tile_actions.size() == 3,
          "future certificate does not prematurely emit or reserve");
}

void test_incomplete_or_over_budget_chain_returns_debt() {
  {
    auto request = crop_request();
    request.plan.unit_schedule[2].step = 91;
    request.plan.units[2].step = 91;
    request.plan_window.turns[2].step = 91;
    const auto proposal = propose_repair_delta(request);
    require(proposal.status == ProposalStatus::Unrepaired &&
                !proposal.accepted() && proposal.delta.unit_patches.empty() &&
                !proposal.unrepaired.empty() &&
                proposal.reason == "tile_continuation_not_shifted_as_a_whole",
            "one early downstream action rejects whole proposal");
  }
  {
    auto request = crop_request();
    request.tile_continuations[0].actions.pop_back();
    const auto proposal = propose_repair_delta(request);
    require(proposal.status == ProposalStatus::Unrepaired &&
                proposal.reason == "tile_continuation_omits_downstream_debt" &&
                !proposal.unrepaired.empty(),
            "omitted later tile work remains explicit unrepaired debt");
  }
  {
    auto request = crop_request();
    request.maximum_delay_days = 0;
    const auto proposal = propose_repair_delta(request);
    require(proposal.status == ProposalStatus::Unrepaired &&
                proposal.reason == "tile_delay_exceeds_day_budget" &&
                !proposal.unrepaired.empty(),
            "day budget, not arbitrary long step horizon, is enforced");
  }
}

void test_animal_place_shifts_all_later_care() {
  auto request = crop_request();
  constexpr fastkag::Position tile{4, 4};
  request.plan.market = {};
  request.debts = {
      debt(200, DebtKind::UnitAction, 0,
           action(fastkag::Op::PICKUP, fastkag::Item::COW), 0, {0, 0}),
      debt(201, DebtKind::UnitAction, 0,
           action(fastkag::Op::PLACE, fastkag::Item::COW), 0, tile),
      debt(202, DebtKind::UnitAction, 0,
           action(fastkag::Op::FEED, fastkag::Item::WHEAT), 1, tile),
      debt(203, DebtKind::UnitAction, 0, action(fastkag::Op::CARE), 0, tile),
      debt(204, DebtKind::UnitAction, 0, action(fastkag::Op::HARVEST), 1, tile),
  };
  request.debts[1].dependencies = {200};
  request.debts[2].dependencies = {201};
  request.debts[3].dependencies = {202};
  request.debts[4].dependencies = {203};
  request.plan.unit_schedule = {
      {201, 44, 0, 0, false}, {202, 45, 1, 0, false},
      {203, 46, 0, 0, false}, {204, 92, 1, 0, false},
  };
  request.plan.scheduled_debt_ids = {201, 202, 203, 204};
  request.plan.units = {
      patch(201, 44, 0, action(fastkag::Op::PLACE, fastkag::Item::COW)),
      patch(202, 45, 1, action(fastkag::Op::FEED, fastkag::Item::WHEAT)),
      patch(203, 46, 0, action(fastkag::Op::CARE)),
      patch(204, 92, 1, action(fastkag::Op::HARVEST)),
  };
  request.plan_window.turns = {
      turn(44, 0, tile, fastkag::Op::PLACE),
      turn(45, 1, tile, fastkag::Op::FEED),
      turn(46, 0, tile, fastkag::Op::CARE),
      turn(92, 1, tile, fastkag::Op::HARVEST),
  };
  TileContinuation continuation;
  continuation.transaction_id = 7;
  continuation.purchase_unit_index = 0;
  continuation.tile = tile;
  continuation.kind = LifecycleKind::Animal;
  continuation.provenance = "fixture:cow-tile-4-4";
  continuation.actions = {
      {201, 20, action(fastkag::Op::PLACE, fastkag::Item::COW), "place"},
      {202, 21, action(fastkag::Op::FEED, fastkag::Item::WHEAT), "feed"},
      {203, 22, action(fastkag::Op::CARE), "care"},
      {204, 68, action(fastkag::Op::HARVEST), "harvest"},
  };
  request.tile_continuations = {continuation};
  const auto proposal = propose_repair_delta(request);
  require(proposal.status == ProposalStatus::Accepted &&
              proposal.delta.unit_patches.size() == 1 &&
              proposal.shifted_tile_actions.size() == 4 &&
              proposal.shifted_tile_actions.back().shifted_step == 92,
          "PLACE forces FEED/CARE/HARVEST whole-tile shift");
}

void test_move_or_stale_binding_fails_closed() {
  {
    auto request = crop_request();
    request.plan.units[0].original = action(fastkag::Op::NORTH);
    request.current_units[0].action = action(fastkag::Op::NORTH);
    const auto proposal = propose_repair_delta(request);
    require(proposal.status == ProposalStatus::InvalidInput &&
                proposal.delta.unit_patches.empty() &&
                !proposal.unrepaired.empty(),
            "RepairDelta adapter cannot replace MOVE");
  }
  {
    auto request = crop_request();
    ++request.key.observation_fingerprint;
    const auto proposal = propose_repair_delta(request);
    require(proposal.status == ProposalStatus::InvalidInput &&
                proposal.reason == "stale_or_unbound_scheduler_plan" &&
                proposal.delta.unit_patches.empty(),
            "stale plan cannot become current delta");
  }
}

}  // namespace

int main() {
  test_crop_whole_tile_shift_to_pure_delta();
  test_default_off_and_future_deferred();
  test_incomplete_or_over_budget_chain_returns_debt();
  test_animal_place_shifts_all_later_care();
  test_move_or_stale_binding_fails_closed();
  std::cout << "modular_repair_adapter_tests: PASS\n";
}
