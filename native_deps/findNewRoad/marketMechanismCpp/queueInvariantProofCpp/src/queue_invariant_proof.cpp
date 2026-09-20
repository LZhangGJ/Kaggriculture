#include "queue_invariant_proof.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace queue_invariant_proof {
namespace {

using g001::market::Operation;
using g001::market::Order;
using protected_queue::SlotOrigin;

constexpr std::size_t kOfficialSlots = 10;

bool valid_product(g001::market::Product product) {
  return static_cast<std::size_t>(product) < g001::market::product_count;
}
bool active_non_sell(const Order& order) {
  return order.operation != Operation::Sell &&
         order.operation != Operation::Pass;
}
bool byte_equal(const Order& left, const Order& right) {
  return left.operation == right.operation && left.product == right.product &&
         left.animal == right.animal && left.quantity == right.quantity;
}
bool checked_add(int& target, int quantity) {
  if (quantity < 0 || target > std::numeric_limits<int>::max() - quantity)
    return false;
  target += quantity;
  return true;
}
int failure_total(const std::vector<int>& failures, bool& valid) {
  int total = 0;
  for (const int failure : failures) {
    if (!checked_add(total, failure)) {
      valid = false;
      return 0;
    }
  }
  return total;
}

void add_check(Result& result, std::string name, bool passed,
               std::string reason) {
  result.audit.checks.push_back(
      {std::move(name), passed, std::move(reason)});
}

}  // namespace

Result derive(const Input& input) {
  Result result;
  bool valid = true;
  auto fail = [&](std::string name, std::string reason) {
    valid = false;
    add_check(result, std::move(name), false, std::move(reason));
  };
  auto pass = [&](std::string name, std::string reason) {
    add_check(result, std::move(name), true, std::move(reason));
  };

  if (!valid_product(input.selected_product))
    fail("selected_product", "selected product is out of range");
  else
    pass("selected_product", "selected product is a valid sellable product");
  if (input.bridge_emitted_total_target < 0)
    fail("bridge_target", "bridge emitted a negative total target");
  else
    pass("bridge_target", "bridge target is nonnegative");
  if (input.legacy_queue.size() > kOfficialSlots ||
      input.candidate.slots.size() > kOfficialSlots)
    fail("slot_limit", "legacy or candidate queue exceeds ten official slots");
  else
    pass("slot_limit", "legacy and candidate queues fit ten slots");
  if (input.candidate.exact_legacy_fallback ||
      input.candidate.audit.used_legacy_fallback)
    fail("fallback", "protected queue returned exact legacy fallback");
  else
    pass("fallback", "candidate is not a fallback");
  if (!input.candidate.audit.minimum_proven ||
      !input.candidate.audit.certificate_passed)
    fail("protected_certificate",
         "protected queue minimum/final exact certificate is incomplete");
  else
    pass("protected_certificate",
         "protected queue minimum and final certificate passed");

  int active_non_sell_count = 0;
  bool legacy_bytes_ok = true;
  bool nonselected_sells_ok = true;
  for (std::size_t slot = 0; slot < input.legacy_queue.size(); ++slot) {
    const Order& legacy = input.legacy_queue[slot];
    if (legacy.quantity < 0) {
      legacy_bytes_ok = false;
      continue;
    }
    if (active_non_sell(legacy)) {
      ++active_non_sell_count;
      if (slot >= input.candidate.slots.size() ||
          input.candidate.slots[slot].origin != SlotOrigin::LegacyNonSell ||
          !byte_equal(input.candidate.slots[slot].order, legacy)) {
        legacy_bytes_ok = false;
      }
    } else if (legacy.operation == Operation::Sell &&
               legacy.product != input.selected_product) {
      if (slot >= input.candidate.slots.size() ||
          input.candidate.slots[slot].origin !=
              SlotOrigin::PreservedLegacySell ||
          !byte_equal(input.candidate.slots[slot].order, legacy)) {
        nonselected_sells_ok = false;
      } else if (!checked_add(result.audit.preserved_nonselected_sell_units,
                              legacy.quantity)) {
        nonselected_sells_ok = false;
      }
    } else if (legacy.operation == Operation::Sell &&
               legacy.product == input.selected_product &&
               !checked_add(result.audit.selected_legacy_sell_units,
                            legacy.quantity)) {
      legacy_bytes_ok = false;
    }
  }
  result.audit.protected_non_sell_slots = active_non_sell_count;
  if (input.candidate.audit.protected_non_sell_orders != active_non_sell_count)
    legacy_bytes_ok = false;
  if (legacy_bytes_ok)
    pass("legacy_non_sell_bytes",
         "every active legacy non-SELL retains slot/op/item/animal/quantity");
  else
    fail("legacy_non_sell_bytes",
         "an active legacy non-SELL changed, moved, vanished, or has invalid quantity");
  if (nonselected_sells_ok)
    pass("nonselected_legacy_sells",
         "all nonselected legacy SELLs are byte-identical PreservedLegacySell slots");
  else
    fail("nonselected_legacy_sells",
         "a nonselected legacy SELL changed byte, slot, or provenance");

  bool candidate_provenance_ok = true;
  int required_orders = 0;
  int optional_orders = 0;
  int preserved_orders = 0;
  for (std::size_t slot = 0; slot < input.candidate.slots.size(); ++slot) {
    const auto& candidate = input.candidate.slots[slot];
    const Order& order = candidate.order;
    if (order.quantity < 0) {
      candidate_provenance_ok = false;
      continue;
    }
    switch (candidate.origin) {
      case SlotOrigin::Pass:
        candidate_provenance_ok &= order.operation == Operation::Pass;
        break;
      case SlotOrigin::LegacyNonSell:
        candidate_provenance_ok &= slot < input.legacy_queue.size() &&
            active_non_sell(input.legacy_queue[slot]) &&
            byte_equal(order, input.legacy_queue[slot]);
        break;
      case SlotOrigin::PreservedLegacySell:
        // Selected-product preserved legacy volume is outside bridge target
        // provenance and therefore cannot be silently added to it.
        candidate_provenance_ok &= slot < input.legacy_queue.size() &&
            order.operation == Operation::Sell &&
            order.product != input.selected_product &&
            byte_equal(order, input.legacy_queue[slot]);
        ++preserved_orders;
        break;
      case SlotOrigin::RequiredFundingSell:
        candidate_provenance_ok &= order.operation == Operation::Sell &&
            order.product == input.selected_product && order.quantity > 0;
        if (!checked_add(result.audit.required_funding_units, order.quantity))
          candidate_provenance_ok = false;
        ++required_orders;
        break;
      case SlotOrigin::OptionalSell:
        candidate_provenance_ok &= order.operation == Operation::Sell &&
            order.product == input.selected_product && order.quantity > 0;
        if (!checked_add(result.audit.optional_units, order.quantity))
          candidate_provenance_ok = false;
        ++optional_orders;
        break;
      case SlotOrigin::ExactLegacyFallback:
        candidate_provenance_ok = false;
        break;
    }
  }
  if (!checked_add(result.audit.selected_actual_units,
                   result.audit.required_funding_units) ||
      !checked_add(result.audit.selected_actual_units,
                   result.audit.optional_units)) {
    candidate_provenance_ok = false;
  }
  if (candidate_provenance_ok)
    pass("candidate_provenance",
         "candidate slots have valid PASS/preserved/required/optional provenance");
  else
    fail("candidate_provenance",
         "candidate contains an invalid, unrelated, unproven, or fallback slot");
  if (input.candidate.audit.required_funding_sell_units !=
          result.audit.required_funding_units ||
      input.candidate.audit.required_funding_sell_orders != required_orders ||
      input.candidate.audit.optional_replacement_units !=
          result.audit.optional_units ||
      input.candidate.audit.optional_replacement_orders != optional_orders ||
      input.candidate.audit.preserved_legacy_sell_units !=
          result.audit.preserved_nonselected_sell_units ||
      input.candidate.audit.preserved_legacy_sell_orders != preserved_orders) {
    fail("provenance_totals",
         "ResultV2 audit SELL totals disagree with per-slot provenance");
  } else {
    pass("provenance_totals",
         "ResultV2 audit SELL totals reconcile with per-slot provenance");
  }
  if (result.audit.selected_actual_units ==
      input.bridge_emitted_total_target) {
    pass("bridge_target_exact",
         "RequiredFundingSell + OptionalSell equals bridge emitted total target");
  } else {
    fail("bridge_target_exact",
         "actual selected SELL total differs from bridge target; certificate reuse forbidden");
  }

  bool bindings_ok = true;
  std::set<int> obligation_ids;
  std::map<int, int> first_binding_by_slot;
  for (const auto& binding : input.candidate.due_legacy_bindings) {
    if (binding.obligation_node_id < 0 || binding.legacy_slot < 0 ||
        binding.required_quantity <= 0 ||
        !obligation_ids.insert(binding.obligation_node_id).second ||
        binding.legacy_slot >= static_cast<int>(input.legacy_queue.size()) ||
        binding.legacy_slot >= static_cast<int>(input.candidate.slots.size()) ||
        !active_non_sell(input.legacy_queue[binding.legacy_slot]) ||
        input.candidate.slots[binding.legacy_slot].origin !=
            SlotOrigin::LegacyNonSell) {
      bindings_ok = false;
      continue;
    }
    first_binding_by_slot.emplace(binding.legacy_slot,
                                  binding.obligation_node_id);
  }
  for (const auto& [slot, first_node] : first_binding_by_slot) {
    if (input.candidate.slots[slot].obligation_node_id != first_node)
      bindings_ok = false;
  }
  result.audit.due_binding_count =
      static_cast<int>(input.candidate.due_legacy_bindings.size());
  if (bindings_ok)
    pass("due_dag_bindings",
         "ResultV2 due bindings are unique, positive, in-range, and attached to protected slots");
  else
    fail("due_dag_bindings",
         "ResultV2 due DAG binding provenance is incomplete or inconsistent");

  bool scenario_audits_ok = !input.candidate.audit.scenarios.empty();
  int baseline_slot_failures = 0;
  int candidate_slot_failures = 0;
  for (const auto& scenario : input.candidate.audit.scenarios) {
    if (scenario.legacy_failure_vector.size() !=
            static_cast<std::size_t>(active_non_sell_count) ||
        scenario.candidate_failure_vector.size() !=
            static_cast<std::size_t>(active_non_sell_count) ||
        scenario.required_sell_unfilled != 0 ||
        scenario.optional_sell_unfilled != 0 ||
        scenario.optional_price_band_failures != 0 ||
        scenario.required_funding_requested_units !=
            result.audit.required_funding_units ||
        scenario.required_funding_committed_units !=
            result.audit.required_funding_units ||
        (result.audit.required_funding_units > 0 &&
         scenario.committed_required_funding_revenue <= 0)) {
      scenario_audits_ok = false;
    }
    bool vector_valid = true;
    const int legacy_failures =
        failure_total(scenario.legacy_failure_vector, vector_valid);
    const int candidate_failures =
        failure_total(scenario.candidate_failure_vector, vector_valid);
    if (!vector_valid) scenario_audits_ok = false;
    baseline_slot_failures = std::max(baseline_slot_failures, legacy_failures);
    candidate_slot_failures =
        std::max(candidate_slot_failures, candidate_failures);
  }
  if (scenario_audits_ok)
    pass("exact_fill_and_band",
         "every exact scenario fully fills required/optional SELLs within price band");
  else
    fail("exact_fill_and_band",
         "scenario failure vectors, required fill, optional fill, or price band are incomplete");

  bool pressure_facts_ok = input.exact_pressure_facts.size() ==
      input.candidate.audit.scenarios.size();
  std::vector<bool> seen(input.candidate.audit.scenarios.size());
  bool all_abstract_opponent_first = true;
  for (const ExactPressureFact& fact : input.exact_pressure_facts) {
    if (fact.scenario_index >= seen.size() || seen[fact.scenario_index] ||
        !fact.exact_replay_completed) {
      pressure_facts_ok = false;
      continue;
    }
    seen[fact.scenario_index] = true;
    if (fact.relation == ExactRelation::NoPressure) {
      ++result.audit.no_pressure_scenarios;
      if (fact.abstract_opponent_first_represented)
        pressure_facts_ok = false;
    } else if (fact.relation == ExactRelation::OpponentBefore) {
      ++result.audit.opponent_before_scenarios;
      if (!fact.abstract_opponent_first_represented)
        pressure_facts_ok = false;
    } else {
      ++result.audit.lockstep_scenarios;
      all_abstract_opponent_first = false;
      if (fact.abstract_opponent_first_represented)
        pressure_facts_ok = false;
    }
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    pressure_facts_ok = false;
  result.audit.all_pressure_scenarios_abstract_opponent_first =
      pressure_facts_ok && all_abstract_opponent_first;
  if (pressure_facts_ok)
    pass("exact_pressure_relations",
         "all exact relations are complete; Lockstep remains explicitly non-OpponentFirst");
  else
    fail("exact_pressure_relations",
         "exact relation facts are missing, duplicated, incomplete, or mislabeled");

  if (input.movement_unchanged)
    pass("movement", "caller boundary proves unit movement unchanged");
  else
    fail("movement", "movement unchanged proof is missing");
  if (input.previous_execution_observation_confirmed)
    pass("observation", "previous market execution is observation-confirmed");
  else
    fail("observation", "previous market execution remains unconfirmed");

  result.audit.accepted = valid;
  if (valid) {
    result.execution.movement_unchanged = true;
    result.execution.only_optional_sell_changed = true;
    result.execution.legacy_non_sell_orders_unchanged = true;
    result.execution.purchase_timing_unchanged = bindings_ok;
    result.execution.land_hire_timing_unchanged = bindings_ok;
    result.execution.required_funding_sells_unchanged = scenario_audits_ok;
    result.execution.previous_execution_observation_confirmed = true;
    result.execution.replacement_materialized =
        input.bridge_emitted_total_target > 0 ||
        result.audit.selected_legacy_sell_units > 0;
    result.execution.selected_total = input.bridge_emitted_total_target;
    result.execution.optional_sell_intent_materialized =
        input.bridge_emitted_total_target > 0;
    result.execution.optional_sell_requested_units =
        input.bridge_emitted_total_target;
    result.slots.evaluated = true;
    result.slots.baseline_failures = baseline_slot_failures;
    result.slots.candidate_failures = candidate_slot_failures;
    std::ostringstream reason;
    reason << "accepted queue invariant proof: target="
           << input.bridge_emitted_total_target << " required="
           << result.audit.required_funding_units << " optional="
           << result.audit.optional_units << " protected non-SELL slots="
           << active_non_sell_count << " due bindings="
           << result.audit.due_binding_count << " worst slot failures "
           << baseline_slot_failures << " -> " << candidate_slot_failures;
    result.audit.reason = reason.str();
  } else {
    result.audit.reason = "rejected: one or more queue invariants lack exact factual proof";
  }
  return result;
}

}  // namespace queue_invariant_proof
