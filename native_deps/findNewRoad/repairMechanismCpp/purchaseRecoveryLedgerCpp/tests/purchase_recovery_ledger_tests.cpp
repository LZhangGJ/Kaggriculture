#include "purchase_recovery_ledger.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fastkag::Action;
using fastkag::Config;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
using g001::purchase_recovery::FillStatus;
using g001::purchase_recovery::FinalSelection;
using g001::purchase_recovery::FundingObservation;
using g001::purchase_recovery::HandoffScope;
using g001::purchase_recovery::Ledger;
using g001::purchase_recovery::Obligation;
using g001::purchase_recovery::ReceiptObservation;
using g001::purchase_recovery::ReopenStatus;
using g001::purchase_recovery::StageStatus;

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::span<const Action> actions(const std::vector<Action> &value) {
  return {value.data(), value.size()};
}

std::span<const std::int32_t> fills(const std::vector<std::int32_t> &value) {
  return {value.data(), value.size()};
}

ReceiptObservation seed_receipt(int step,
                                const std::vector<std::int32_t> &slot_fills,
                                Item item, int holding) {
  ReceiptObservation result;
  result.step = step;
  result.slot_fills = fills(slot_fills);
  result.seeds_after[static_cast<std::size_t>(item)] = holding;
  return result;
}

ReceiptObservation animal_receipt(int step,
                                  const std::vector<std::int32_t> &slot_fills,
                                  Item item, int holding) {
  ReceiptObservation result;
  result.step = step;
  result.slot_fills = fills(slot_fills);
  result.animals_after[static_cast<std::size_t>(
      static_cast<int>(item) - static_cast<int>(Item::GOOSE))] = holding;
  return result;
}

void exact_slot_zero_fill_cash_recovery_retry_and_inventory_handoff() {
  Config config;
  config.starting_money = 100;
  config.episode_steps = 8;
  config.weed_spawn_chance = 0.0;
  Simulator simulator(config, 0x5EEDULL);

  Ledger ledger;
  const auto debt_id = ledger.open(Obligation{Op::BUY_SEED, Item::MELON, 1, 80,
                                              "blocked MELON PLANT source"});

  // The ledger describes the frozen hard BUY_SEED but cannot select its slot
  // or mutate the final queue. The final composer retains its earlier hard
  // BUY_PRODUCT order in slot 0 and selects slot 1 for this binding.
  const auto hard = ledger.describe_hard_order(debt_id, 1);
  check(hard.has_value(), "hard purchase proposal missing");
  std::array<PlayerAction, 2> turn{};
  turn[0].market = {{Op::BUY_PRODUCT, Item::WHEAT, 2},
                    {Op::BUY_SEED, Item::MELON, 1}};
  check(ledger.stage_final(*hard, FinalSelection{simulator.step_count(), 1,
                                                 actions(turn[0].market), 0}) ==
            StageStatus::Selected,
        "final composer slot was not selected exactly");
  check(turn[0].market[0].op == Op::BUY_PRODUCT,
        "ledger overwrote the earlier hard market obligation");

  simulator.step(turn);
  check(simulator.last_market_fills()[0].size() == 2 &&
            simulator.last_market_fills()[0][0] == 2 &&
            simulator.last_market_fills()[0][1] == 0,
        "simulator witness did not produce exact slot-1 zero fill");
  auto settled = ledger.observe(seed_receipt(
      simulator.step_count(), simulator.last_market_fills()[0], Item::MELON,
      simulator.privates()[0].seeds[static_cast<int>(Item::MELON)]));
  check(settled.size() == 1 && settled[0].status == FillStatus::Zero &&
            settled[0].remaining == 1,
        "zero fill did not preserve one unit of purchase debt");
  check(ledger.drain_handoffs().empty(),
        "zero fill incorrectly handed inventory to unit repair");

  // The bought WHEAT is sold to recover cash. No recovery proposal exists
  // before the sale because MELON costs 80 and current cash is only 50.
  check(
      !ledger.propose_recovery(
          debt_id,
          FundingObservation{simulator.step_count(),
                             static_cast<int>(simulator.farms()[0].money), 0}),
      "unfunded retry was proposed");
  turn = {};
  turn[0].market = {{Op::SELL, Item::WHEAT, 2}};
  simulator.step(turn);
  check(simulator.farms()[0].money >= 90,
        "fixture did not recover sufficient cash");

  // Ten cash is protected for a new hard WHEAT seed obligation. The ledger
  // proposes only from the remaining budget and still leaves slot choice to
  // the composer, which retains WHEAT at slot 0 and puts retry at slot 1.
  const auto retry = ledger.propose_recovery(
      debt_id,
      FundingObservation{simulator.step_count(),
                         static_cast<int>(simulator.farms()[0].money), 10});
  check(retry.has_value() && retry->order.op == Op::BUY_SEED &&
            retry->order.item == Item::MELON && retry->order.quantity == 1,
        "funded MELON retry was not proposed");
  turn = {};
  turn[0].market = {{Op::BUY_SEED, Item::WHEAT, 1}, retry->order};
  check(ledger.stage_final(
            *retry,
            FinalSelection{simulator.step_count(), 1, actions(turn[0].market),
                           0, static_cast<int>(simulator.farms()[0].money),
                           10}) == StageStatus::Selected,
        "composer-selected retry slot was not staged");
  check(turn[0].market[0].op == Op::BUY_SEED &&
            turn[0].market[0].item == Item::WHEAT,
        "recovery replaced a protected hard order");

  simulator.step(turn);
  check(simulator.last_market_fills()[0].size() == 2 &&
            simulator.last_market_fills()[0][1] == 1 &&
            simulator.privates()[0].seeds[static_cast<int>(Item::MELON)] == 1,
        "retry did not produce an exact fill plus inventory arrival");
  settled = ledger.observe(seed_receipt(
      simulator.step_count(), simulator.last_market_fills()[0], Item::MELON,
      simulator.privates()[0].seeds[static_cast<int>(Item::MELON)]));
  check(settled.size() == 1 && settled[0].status == FillStatus::Full &&
            settled[0].filled == 1 && settled[0].remaining == 0,
        "filled retry did not retire purchase debt");
  const auto handoffs = ledger.drain_handoffs();
  check(handoffs.size() == 1 && handoffs[0].debt_id == debt_id &&
            handoffs[0].quantity == 1 &&
            handoffs[0].scope == HandoffScope::InventoryOnly,
        "inventory arrival was not handed back exactly once");
  const auto debt = ledger.debt(debt_id);
  check(debt && debt->acquisition_complete && debt->attempts == 2 &&
            debt->filled == 1,
        "debt/audit state does not reflect zero-fill plus successful retry");
}

void final_composer_override_never_advances_debt() {
  Ledger ledger;
  const auto debt_id = ledger.open(
      Obligation{Op::BUY_SEED, Item::CARROT, 1, 20, "CARROT PLANT"});
  const auto proposal = ledger.describe_hard_order(debt_id, 1);
  check(proposal.has_value(), "hard proposal missing");
  const std::vector<Action> final_market{{Op::HIRE, Item::NONE, 1}};
  check(ledger.stage_final(*proposal,
                           FinalSelection{9, 0, actions(final_market), 0}) ==
            StageStatus::FinalOrderMismatch,
        "overridden proposal was staged");
  const auto debt = ledger.debt(debt_id);
  check(debt && debt->remaining == 1 && debt->attempts == 0 &&
            !debt->awaiting_receipt,
        "composer override mutated purchase debt");
}

void positive_fill_without_inventory_witness_is_ambiguous() {
  Ledger ledger;
  const auto debt_id = ledger.open(
      Obligation{Op::BUY_SEED, Item::TOMATO, 1, 50, "TOMATO PLANT"});
  const auto proposal = ledger.describe_hard_order(debt_id, 1);
  const std::vector<Action> final_market{proposal->order};
  check(ledger.stage_final(*proposal,
                           FinalSelection{3, 0, actions(final_market), 0}) ==
            StageStatus::Selected,
        "valid proposal did not stage");
  const std::vector<std::int32_t> exact_fills{1};
  const auto settled =
      ledger.observe(seed_receipt(4, exact_fills, Item::TOMATO, 0));
  check(settled.size() == 1 && settled[0].status == FillStatus::Ambiguous &&
            ledger.debt(debt_id)->remaining == 1 &&
            ledger.drain_handoffs().empty(),
        "uncorroborated fill incorrectly retired debt");
}

void partial_fill_hands_off_prefix_and_preserves_suffix() {
  Ledger ledger;
  const auto debt_id = ledger.open(
      Obligation{Op::BUY_SEED, Item::MELON, 2, 80, "two MELON PLANTs"});
  const auto proposal = ledger.describe_hard_order(debt_id, 2);
  const std::vector<Action> final_market{proposal->order};
  check(ledger.stage_final(*proposal,
                           FinalSelection{11, 0, actions(final_market), 0}) ==
            StageStatus::Selected,
        "two-unit proposal did not stage");
  const std::vector<std::int32_t> exact_fills{1};
  const auto settled =
      ledger.observe(seed_receipt(12, exact_fills, Item::MELON, 1));
  const auto handoffs = ledger.drain_handoffs();
  check(settled.size() == 1 && settled[0].status == FillStatus::Partial &&
            settled[0].remaining == 1 && handoffs.size() == 1 &&
            handoffs[0].quantity == 1,
        "partial fill did not split confirmed prefix from open suffix");
}

void animal_handoff_is_acquisition_only() {
  Ledger ledger;
  const auto debt_id = ledger.open(Obligation{Op::BUY_ANIMAL, Item::GOOSE, 1,
                                              300, "future animal lifecycle"});
  const auto proposal = ledger.describe_hard_order(debt_id, 1);
  const std::vector<Action> final_market{proposal->order};
  check(ledger.stage_final(*proposal,
                           FinalSelection{20, 0, actions(final_market), 0}) ==
            StageStatus::Selected,
        "animal acquisition did not stage");
  const std::vector<std::int32_t> exact_fills{1};
  const auto settled =
      ledger.observe(animal_receipt(21, exact_fills, Item::GOOSE, 1));
  const auto handoffs = ledger.drain_handoffs();
  check(settled.size() == 1 && settled[0].status == FillStatus::Full &&
            handoffs.size() == 1 && handoffs[0].acquisition == Op::BUY_ANIMAL &&
            handoffs[0].scope == HandoffScope::InventoryOnly,
        "animal fill claimed more than inventory acquisition");
}

void completed_acquisition_reopens_idempotently_after_handoff() {
  Ledger ledger;
  const auto debt_id = ledger.open(
      Obligation{Op::BUY_ANIMAL, Item::GOOSE, 1, 300, "lost goose"});
  const auto proposal = ledger.describe_hard_order(debt_id, 1);
  const std::vector<Action> final_market{proposal->order};
  check(ledger.stage_final(*proposal,
                           FinalSelection{30, 0, actions(final_market), 0}) ==
            StageStatus::Selected,
        "reopen fixture did not stage");
  const std::vector<std::int32_t> exact_fills{1};
  const auto settled =
      ledger.observe(animal_receipt(31, exact_fills, Item::GOOSE, 1));
  check(settled.size() == 1 && settled[0].status == FillStatus::Full &&
            ledger.reopen_acquisition(debt_id) == ReopenStatus::PendingHandoff,
        "undrained inventory handoff was reopened ambiguously");
  check(ledger.drain_handoffs().size() == 1 &&
            ledger.reopen_acquisition(debt_id) == ReopenStatus::Reopened,
        "completed acquisition did not reopen after typed handoff ownership");
  const auto reopened = ledger.debt(debt_id);
  check(reopened && reopened->remaining == 1 &&
            !reopened->acquisition_complete && reopened->filled == 1 &&
            reopened->attempts == 1 &&
            ledger.reopen_acquisition(debt_id) == ReopenStatus::AlreadyOpen,
        "reopen was not idempotent or erased historical acquisition audit");
  check(ledger.stage_final(*proposal,
                           FinalSelection{31, 0, actions(final_market), 0}) ==
            StageStatus::StaleProposal,
        "pre-reopen purchase proposal survived revision change");
}

} // namespace

int main() {
  try {
    exact_slot_zero_fill_cash_recovery_retry_and_inventory_handoff();
    final_composer_override_never_advances_debt();
    positive_fill_without_inventory_witness_is_ambiguous();
    partial_fill_hands_off_prefix_and_preserves_suffix();
    animal_handoff_is_acquisition_only();
    completed_acquisition_reopens_idempotently_after_handoff();
    std::cout << "purchase_recovery_ledger_tests: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "purchase_recovery_ledger_tests: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
