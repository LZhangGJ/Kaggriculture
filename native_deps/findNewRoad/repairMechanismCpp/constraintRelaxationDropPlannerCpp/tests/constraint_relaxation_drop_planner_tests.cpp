#include "constraint_relaxation_drop_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace cr = g001::constraint_relaxation;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

cr::Slot slot(const int step, const int actor) {
  return {step, actor, false, -1, 0, 73};
}

cr::Slot move_slot(const int step, const int actor, const int source,
                   const std::uint64_t token) {
  return {step, actor, true, source, token, 73};
}

cr::Obligation obligation(const std::uint64_t id, const int actor,
                           const int earliest, const int deadline,
                           const int loss) {
  cr::Obligation result;
  result.id = id;
  result.action = {fastkag::Op::PLANT, fastkag::Item::WHEAT, 1};
  result.tile = {static_cast<std::int16_t>(id % 7),
                 static_cast<std::int16_t>(id % 5)};
  result.actor = actor;
  result.earliest_step = earliest;
  result.deadline = deadline;
  result.value_loss = loss;
  result.cascade_damage = 0;
  result.defer_damage = 1;
  result.lineage_hash = 1000 + id;
  return result;
}

cr::Request base_request() {
  cr::Request request;
  request.current_step = 100;
  request.horizon_end_step = 123;
  request.turns_per_day = 24;
  request.observation_hash = 0xabc123;
  request.final_owner_generation = 73;
  request.maximum_carry_days = 2;
  request.maximum_deferred_ledger = 32;
  request.exact_node_limit = 16;
  request.maximum_search_states = 1'000'000;
  return request;
}

const cr::Decision& decision(const cr::Plan& plan, const std::uint64_t id) {
  const auto found = std::find_if(
      plan.decisions.begin(), plan.decisions.end(),
      [&](const auto& item) { return item.obligation_id == id; });
  require(found != plan.decisions.end(), "missing explicit decision");
  return *found;
}

void require_valid(const cr::Request& request, const cr::Plan& plan) {
  require(plan.accepted(), "plan rejected: " + plan.diagnostic);
  const auto verified = cr::verify(request, plan);
  require(verified.valid, "plan failed verifier: " + verified.diagnostic);
  require(plan.decisions.size() == request.obligations.size(),
          "planner silently omitted an obligation");
}

void test_actor_slot_bipartite_and_exact_global_minimum() {
  auto request = base_request();
  request.remaining_slots = {slot(100, 0), slot(100, 1), slot(101, 0)};
  auto a = obligation(10, 0, 100, 101, 20);
  auto b = obligation(20, 1, 100, 100, 30);
  auto c = obligation(30, 0, 100, 101, 5);
  request.obligations = {c, b, a};
  const auto result = cr::plan(request);
  require_valid(request, result);
  require(result.exact && !result.bounded_fallback, "small plan not exact");
  require(result.assignments.size() == 3,
          "actor x step slots were incorrectly collapsed");
  require(result.objective.total_damage == 0, "feasible plan lost value");
  std::map<std::pair<int, int>, int> occupied;
  for (const auto& assignment : result.assignments) {
    ++occupied[{assignment.step, assignment.actor}];
  }
  require(occupied[{100, 0}] == 1 && occupied[{100, 1}] == 1 &&
              occupied[{101, 0}] == 1,
          "actor-slot matching was not one-to-one");
}

void test_capacity_shortage_drops_dominated_lower_value() {
  auto request = base_request();
  request.remaining_slots = {slot(100, 0)};
  request.obligations = {obligation(1, 0, 100, 100, 5),
                         obligation(2, 0, 100, 100, 50)};
  const auto result = cr::plan(request);
  require_valid(request, result);
  require(decision(result, 2).disposition == cr::Disposition::Kept,
          "exact solver did not keep minimum-damage choice");
  const auto& low = decision(result, 1);
  require(low.disposition == cr::Disposition::DroppedDominated && low.drop &&
              low.drop->reason == cr::DropReason::DominatedByLowerDamageSet,
          "capacity loss was not explicit DroppedDominated");
  require(low.drop->conflicting_kept_obligations ==
              std::vector<std::uint64_t>{2},
          "domination certificate omitted slot conflict");
}

void test_resource_exclusion_and_evidence() {
  auto request = base_request();
  request.remaining_slots = {slot(100, 0), slot(101, 0)};
  request.resources = {{"seed:wheat", 1, 0x5511}};
  auto low = obligation(4, 0, 100, 101, 10);
  low.resources = {{"seed:wheat", 1}};
  auto high = obligation(5, 0, 100, 101, 40);
  high.resources = {{"seed:wheat", 1}};
  request.obligations = {low, high};
  const auto result = cr::plan(request);
  require_valid(request, result);
  require(decision(result, 5).disposition == cr::Disposition::Kept,
          "resource arbiter kept higher-damage exclusion");
  const auto& dropped = decision(result, 4);
  require(dropped.disposition == cr::Disposition::DroppedDominated &&
              dropped.drop && dropped.drop->resource_evidence_hash != 0 &&
              dropped.drop->content_hash ==
                  cr::drop_certificate_hash(*dropped.drop),
          "resource conflict lacks bound certificate");

  auto unavailable = base_request();
  unavailable.remaining_slots = {slot(100, 0)};
  unavailable.resources = {{"animal:cow", 0, 0x9911}};
  auto animal = obligation(9, 0, 100, 100, 12);
  animal.resources = {{"animal:cow", 1}};
  unavailable.obligations = {animal};
  const auto unavailable_plan = cr::plan(unavailable);
  require_valid(unavailable, unavailable_plan);
  require(decision(unavailable_plan, 9).drop->reason ==
              cr::DropReason::ResourceUnavailable,
          "physical resource infeasibility mislabeled as domination");

  request.resources[0].evidence_hash = 0;
  require(!cr::plan(request).accepted(),
          "zero resource evidence was accepted");
}

void test_dependency_drop_recursively_cancels_successors() {
  auto request = base_request();
  request.remaining_slots = {slot(100, 0), slot(101, 0), slot(102, 0)};
  request.resources = {{"pasture", 0, 0x7788}};
  auto root = obligation(10, 0, 100, 102, 7);
  root.resources = {{"pasture", 1}};
  auto child = obligation(20, 0, 100, 102, 9);
  child.dependencies = {10};
  auto grandchild = obligation(30, 0, 100, 102, 11);
  grandchild.dependencies = {20};
  request.obligations = {grandchild, root, child};
  const auto result = cr::plan(request);
  require_valid(request, result);
  require(decision(result, 10).drop->reason ==
              cr::DropReason::ResourceUnavailable,
          "root physical infeasibility reason changed");
  require(decision(result, 20).drop->reason ==
              cr::DropReason::DependencyDropped &&
              decision(result, 20).drop->cancelled_by_dependency == 10,
          "child did not cite dropped parent");
  require(decision(result, 30).drop->reason ==
              cr::DropReason::DependencyDropped &&
              decision(result, 30).drop->cancelled_by_dependency == 20,
          "recursive successor cancellation was not explicit");
  require(result.objective.total_damage == 27,
          "recursive drop loss omitted descendant value");
}

void test_strict_dependency_order_and_typed_resource_production() {
  auto request = base_request();
  request.remaining_slots = {slot(100, 0), slot(100, 1), slot(101, 0)};
  request.resources = {{"cash", 1, 0xc001}, {"seed:wheat", 0, 0x5eed}};
  auto purchase = obligation(10, 1, 100, 100, 80);
  purchase.action = {fastkag::Op::BUY_SEED, fastkag::Item::WHEAT, 1};
  purchase.resources = {{"cash", 1}};
  purchase.produces = {{"seed:wheat", 1}};
  auto plant = obligation(20, 0, 100, 101, 70);
  plant.dependencies = {10};
  plant.resources = {{"seed:wheat", 1}};
  request.obligations = {plant, purchase};
  const auto result = cr::plan(request);
  require_valid(request, result);
  require(decision(result, 10).assignment->step == 100 &&
              decision(result, 10).assignment->actor == 1,
          "purchase assignment changed unexpectedly");
  require(decision(result, 20).assignment->step == 101 &&
              decision(result, 20).assignment->actor == 0,
          "child ran before high-actor parent in same simulator step");
  require(result.objective.total_damage == 0,
          "typed purchase production did not fund later consumer");
}

void test_every_assignment_rechecks_all_resource_prefixes() {
  auto request = base_request();
  request.remaining_slots = {slot(100, 0), slot(101, 1)};
  request.resources = {{"cash", 1, 0xca55}};
  auto late_high = obligation(1, 1, 101, 101, 50);
  late_high.resources = {{"cash", 1}};
  auto early_low = obligation(2, 0, 100, 100, 10);
  early_low.resources = {{"cash", 1}};
  request.obligations = {late_high, early_low};
  const auto result = cr::plan(request);
  require_valid(request, result);
  require(decision(result, 1).disposition == cr::Disposition::Kept &&
              decision(result, 2).disposition ==
                  cr::Disposition::DroppedDominated,
          "later insertion made an already kept prefix resource-negative");
  require(result.objective.total_damage == 10,
          "prefix resource conflict did not choose global minimum damage");
}

void test_exact_can_insert_later_enumerated_producer_before_consumer() {
  auto request = base_request();
  request.remaining_slots = {slot(100, 0), slot(101, 1)};
  request.resources = {{"seed:carrot", 0, 0xcafe}};
  auto consumer = obligation(1, 1, 101, 101, 70);
  consumer.action = {fastkag::Op::PLANT, fastkag::Item::CARROT, 1};
  consumer.resources = {{"seed:carrot", 1}};
  auto producer = obligation(99, 0, 100, 100, 60);
  producer.action = {fastkag::Op::BUY_SEED, fastkag::Item::CARROT, 1};
  producer.produces = {{"seed:carrot", 1}};
  request.obligations = {consumer, producer};
  const auto result = cr::plan(request);
  require_valid(request, result);
  require(result.exact && result.assignments.size() == 2 &&
              result.objective.total_damage == 0,
          "exact search pruned a later-enumerated earlier producer");
}

void test_move_commitment_and_owner_generation_are_hard() {
  auto request = base_request();
  request.remaining_slots = {move_slot(100, 0, 88, 0xbeef)};
  request.obligations = {obligation(1, 0, 100, 100, 9)};
  const auto result = cr::plan(request);
  require_valid(request, result);
  require(result.assignments.empty(), "planner consumed a MOVE slot");
  require(result.preserved_move_commitments == 1 &&
              result.move_commitment_hash != 0,
          "MOVE commitment was not preserved and bound");
  require(decision(result, 1).drop->reason ==
              cr::DropReason::MoveCommittedSlot,
          "MOVE-only infeasibility was not diagnosed");

  request.remaining_slots[0].final_owner_generation = 74;
  require(!cr::plan(request).accepted(),
          "stale final-owner slot was accepted");

  auto multi_actor = base_request();
  multi_actor.remaining_slots = {move_slot(100, 0, 88, 0x1001),
                                 move_slot(100, 1, 88, 0x1002)};
  require(cr::plan(multi_actor).accepted(),
          "same raw source ordinal on distinct actors was rejected");
  multi_actor.remaining_slots[1].move_token = 0x1001;
  require(!cr::plan(multi_actor).accepted(),
          "duplicate global MOVE token was accepted");
  multi_actor.remaining_slots[1] = move_slot(101, 0, 88, 0x1002);
  require(!cr::plan(multi_actor).accepted(),
          "duplicate actor/source MOVE identity was accepted");
}

void test_deferred_ledger_is_explicit_and_bounded() {
  auto request = base_request();
  request.maximum_deferred_ledger = 2;
  request.maximum_carry_days = 2;
  for (std::uint64_t id = 1; id <= 5; ++id) {
    auto item = obligation(id, 0, 100, 130, static_cast<int>(id * 10));
    item.allow_defer = true;
    item.defer_damage = 1;
    request.obligations.push_back(item);
  }
  const auto result = cr::plan(request);
  require_valid(request, result);
  require(result.deferred_ledger.size() == 2,
          "deferred ledger exceeded or underused explicit bound");
  require(decision(result, 5).disposition == cr::Disposition::Deferred &&
              decision(result, 4).disposition == cr::Disposition::Deferred,
          "minimum-damage ledger did not retain highest-loss debts");
  for (const std::uint64_t id : {1U, 2U, 3U}) {
    require(decision(result, id).drop->reason ==
                cr::DropReason::LedgerCapacityBound,
            "ledger eviction was not an explicit capacity drop");
  }

  auto aged = base_request();
  aged.maximum_carry_days = 2;
  auto old = obligation(90, 0, 100, 130, 100);
  old.allow_defer = true;
  old.carry_days = 2;
  aged.obligations = {old};
  const auto aged_plan = cr::plan(aged);
  require_valid(aged, aged_plan);
  require(decision(aged_plan, 90).drop->reason ==
              cr::DropReason::LedgerAgeBound,
          "aged debt could grow across another day");
}

void test_date_translation_property() {
  auto original = base_request();
  original.remaining_slots = {slot(101, 0), slot(102, 1)};
  original.obligations = {obligation(1, 0, 100, 102, 20),
                          obligation(2, 1, 101, 102, 30),
                          obligation(3, 0, 100, 100, 4)};
  const auto before = cr::plan(original);
  require_valid(original, before);

  auto translated = original;
  constexpr int delta = 24 * 17;
  translated.current_step += delta;
  translated.horizon_end_step += delta;
  for (auto& item : translated.obligations) {
    item.earliest_step += delta;
    item.deadline += delta;
  }
  for (auto& item : translated.remaining_slots) item.step += delta;
  const auto after = cr::plan(translated);
  require_valid(translated, after);
  require(before.objective.total_damage == after.objective.total_damage &&
              before.objective.slot_perturbation ==
                  after.objective.slot_perturbation,
          "date translation changed optimization result");
  for (const auto& before_decision : before.decisions) {
    require(decision(after, before_decision.obligation_id).disposition ==
                before_decision.disposition,
            "date translation changed disposition");
  }
}

void test_id_and_tile_permutation_property() {
  auto left = base_request();
  left.remaining_slots = {slot(100, 0), slot(101, 0)};
  auto parent = obligation(11, 0, 100, 101, 50);
  auto child = obligation(22, 0, 100, 101, 20);
  child.dependencies = {11};
  auto excluded = obligation(33, 0, 100, 101, 3);
  left.obligations = {excluded, child, parent};
  const auto first = cr::plan(left);
  require_valid(left, first);

  auto right = left;
  const std::map<std::uint64_t, std::uint64_t> ids{
      {11, 901}, {22, 707}, {33, 505}};
  for (auto& item : right.obligations) {
    item.id = ids.at(item.id);
    for (auto& dependency : item.dependencies) dependency = ids.at(dependency);
    item.tile = {static_cast<std::int16_t>(9 - item.tile.x),
                 static_cast<std::int16_t>(8 - item.tile.y)};
    item.lineage_hash += 0x10000;
  }
  std::reverse(right.obligations.begin(), right.obligations.end());
  const auto second = cr::plan(right);
  require_valid(right, second);
  require(first.objective.total_damage == second.objective.total_damage &&
              first.objective.dropped_count == second.objective.dropped_count &&
              first.kept_subgraph.size() == second.kept_subgraph.size(),
          "ID/tile permutation changed feasible damage subgraph");
  for (const auto& [old_id, new_id] : ids) {
    require(decision(first, old_id).disposition ==
                decision(second, new_id).disposition,
            "ID/tile permutation changed typed disposition");
  }
}

void test_invalid_graph_id_and_slot_certificates() {
  auto cycle = base_request();
  auto a = obligation(1, 0, 100, 100, 1);
  auto b = obligation(2, 0, 100, 100, 1);
  a.dependencies = {2};
  b.dependencies = {1};
  cycle.obligations = {a, b};
  require(!cr::plan(cycle).accepted(), "DAG cycle was accepted");

  auto self = base_request();
  a.dependencies = {1};
  self.obligations = {a};
  require(!cr::plan(self).accepted(), "self dependency was accepted");

  auto duplicate = base_request();
  a.dependencies.clear();
  b.id = 1;
  duplicate.obligations = {a, b};
  require(!cr::plan(duplicate).accepted(), "duplicate obligation ID accepted");

  auto duplicate_slot = base_request();
  duplicate_slot.remaining_slots = {slot(100, 0), slot(100, 0)};
  require(!cr::plan(duplicate_slot).accepted(),
          "duplicate actor x step slot accepted");
}

void test_fallback_is_bounded_deterministic_and_never_omits() {
  auto request = base_request();
  request.exact_node_limit = 4;
  request.maximum_search_states = 30;
  for (int i = 0; i < 18; ++i) {
    request.remaining_slots.push_back(slot(100 + i / 2, i % 2));
    request.obligations.push_back(
        obligation(static_cast<std::uint64_t>(i + 1), i % 2, 100,
                   123, 20 + i));
  }
  const auto first = cr::plan(request);
  const auto second = cr::plan(request);
  require_valid(request, first);
  require(first.bounded_fallback && !first.exact,
          "oversized graph masqueraded as exact");
  require(first.explored_states <= request.maximum_search_states,
          "fallback exceeded deterministic state bound");
  require(first.decisions.size() == 18 && first.drops.empty() &&
              first.deferred_ledger.empty(),
          "fallback omitted or unnecessarily discarded a feasible node");
  require(first.content_hash == second.content_hash,
          "bounded fallback is nondeterministic");
}

void test_verifier_rejects_tamper() {
  auto request = base_request();
  request.remaining_slots = {slot(100, 0)};
  request.obligations = {obligation(1, 0, 100, 100, 10)};
  auto result = cr::plan(request);
  require_valid(request, result);
  result.assignments[0].step = 101;
  require(!cr::verify(request, result).valid,
          "assignment tamper survived verifier");

  result = cr::plan(request);
  request.obligations[0].action.op = fastkag::Op::DIG;
  require(!cr::verify(request, result).valid,
          "typed request action tamper survived authority binding");
}

}  // namespace

int main() {
  const auto started = std::chrono::steady_clock::now();
  try {
    test_actor_slot_bipartite_and_exact_global_minimum();
    test_capacity_shortage_drops_dominated_lower_value();
    test_resource_exclusion_and_evidence();
    test_dependency_drop_recursively_cancels_successors();
    test_strict_dependency_order_and_typed_resource_production();
    test_every_assignment_rechecks_all_resource_prefixes();
    test_exact_can_insert_later_enumerated_producer_before_consumer();
    test_move_commitment_and_owner_generation_are_hard();
    test_deferred_ledger_is_explicit_and_bounded();
    test_date_translation_property();
    test_id_and_tile_permutation_property();
    test_invalid_graph_id_and_slot_certificates();
    test_fallback_is_bounded_deterministic_and_never_omits();
    test_verifier_rejects_tamper();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  const auto elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started);
  std::cout << "PASS constraint-relaxation/drop-planner properties"
            << " elapsed_ms=" << elapsed.count() << '\n';
  return EXIT_SUCCESS;
}
