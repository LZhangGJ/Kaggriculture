#include "robust_certificate.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace robust_certificate {
namespace {

using g001::rolling::FailureCounts;

phased_takeover::FailureVector map_failures(const FailureCounts& failures,
                                            int market_slot_failures) {
  if (failures.critical_purchase < 0 || failures.feed < 0 ||
      failures.overflow < 0 || failures.route_hard < 0 ||
      market_slot_failures < 0) {
    throw std::invalid_argument("negative robust-certificate failure count");
  }
  auto checked_sum = [](int left, int right, int third = 0) {
    const auto value = static_cast<long long>(left) + right + third;
    if (value > std::numeric_limits<int>::max())
      throw std::overflow_error("robust-certificate failure count overflow");
    return static_cast<int>(value);
  };
  phased_takeover::FailureVector mapped;
  // A missed critical purchase or feed breaks production directly. RouteHard
  // is deliberately charged to both production and liquidity: its compact ABI
  // does not identify whether cash or a fixed productive action was the first
  // cause, so assigning it to only one axis would be optimistic.
  mapped.production = checked_sum(failures.critical_purchase, failures.feed,
                                  failures.route_hard);
  mapped.liquidity = checked_sum(failures.critical_purchase,
                                 failures.route_hard);
  mapped.capacity = failures.overflow;
  mapped.market_slots = market_slot_failures;
  return mapped;
}

bool no_worse(const phased_takeover::FailureVector& candidate,
              const phased_takeover::FailureVector& baseline) {
  return candidate.production <= baseline.production &&
         candidate.liquidity <= baseline.liquidity &&
         candidate.capacity <= baseline.capacity &&
         candidate.market_slots <= baseline.market_slots;
}

bool same_option(const g001::rolling::PersistentOption& left,
                 const g001::rolling::PersistentOption& right) {
  return std::tie(left.kind, left.product, left.quota, left.window_steps,
                  left.reservation_price, left.target_inventory,
                  left.impact_limit) ==
         std::tie(right.kind, right.product, right.quota, right.window_steps,
                  right.reservation_price, right.target_inventory,
                  right.impact_limit);
}

bool sell_option(g001::option::Kind kind) {
  return kind == g001::option::Kind::Drip ||
         kind == g001::option::Kind::PriceTarget ||
         kind == g001::option::Kind::InventoryTarget ||
         kind == g001::option::Kind::PreDump ||
         kind == g001::option::Kind::Clear;
}

}  // namespace

Result compile_selective_sell(const g001::general::Result& planner,
                              const Proof& proof) {
  if (proof.execution.selected_total < 0 ||
      proof.execution.optional_sell_requested_units < 0 ||
      proof.slots.baseline_failures < 0 || proof.slots.candidate_failures < 0) {
    throw std::invalid_argument("negative robust-certificate proof field");
  }

  Result result;
  auto& certificate = result.certificate;
  auto& audit = result.audit;
  const auto& candidate_score = planner.plan.score;
  const auto& baseline_score = planner.plan.baseline_reference;

  audit.planner_scenario_count = planner.dump_audit.scenarios.size();
  const bool explicit_bands = proof.scenarios.public_observations_only &&
      proof.scenarios.lower_band_evaluated &&
      proof.scenarios.point_band_evaluated &&
      proof.scenarios.upper_band_evaluated &&
      proof.scenarios.exact_planner_scenario_set_replayed;
  const bool count_matches = proof.scenarios.replayed_scenario_count ==
      planner.dump_audit.scenarios.size();
  audit.scenario_proof_passed = explicit_bands && count_matches &&
      !planner.dump_audit.scenarios.empty();

  audit.all_scenarios_opponent_first =
      audit.scenario_proof_passed;
  double weight_sum = 0.0;
  for (const g001::dump::WeightedScenario& value :
       planner.dump_audit.scenarios) {
    const auto& scenario = value.scenario;
    if (scenario.same_tick_order != g001::rolling::SameTickOrder::OpponentFirst ||
        scenario.realized_future || !std::isfinite(scenario.weight) ||
        scenario.weight <= 0.0 || !std::isfinite(value.weight) ||
        value.weight <= 0.0 || std::abs(scenario.weight - value.weight) > 1e-12) {
      audit.all_scenarios_opponent_first = false;
    }
    weight_sum += scenario.weight;
  }
  if (std::abs(weight_sum - 1.0) > 1e-9)
    audit.all_scenarios_opponent_first = false;

  const auto& execution = proof.execution;
  audit.execution_proof_passed = execution.movement_unchanged &&
      execution.only_optional_sell_changed &&
      execution.legacy_non_sell_orders_unchanged &&
      execution.purchase_timing_unchanged &&
      execution.land_hire_timing_unchanged &&
      execution.required_funding_sells_unchanged &&
      execution.previous_execution_observation_confirmed;

  const bool selected_sequence_consistent =
      !planner.plan.planned_sequence.empty() &&
      same_option(planner.plan.first, planner.plan.planned_sequence.front());
  const bool explicit_replacement = execution.replacement_materialized;
  const bool legacy_positive_replacement =
      execution.optional_sell_intent_materialized &&
      execution.optional_sell_requested_units > 0;
  const bool replacement_fields_consistent =
      !(explicit_replacement && legacy_positive_replacement) ||
      execution.selected_total == execution.optional_sell_requested_units;
  audit.replacement_proof_passed = replacement_fields_consistent &&
      (explicit_replacement || legacy_positive_replacement);
  audit.selected_total = explicit_replacement
      ? execution.selected_total : execution.optional_sell_requested_units;
  const bool explicit_zero_replacement =
      explicit_replacement && execution.selected_total == 0;
  const bool sell_mechanism_action_proven =
      audit.selected_total > 0 || execution.zero_option_action_replayed;
  const bool selected_mechanism =
      (sell_option(planner.plan.first.kind) &&
       audit.replacement_proof_passed && sell_mechanism_action_proven) ||
      (planner.plan.first.kind == g001::option::Kind::Hold &&
       explicit_zero_replacement && audit.replacement_proof_passed);
  audit.actionable_optional_sell = selected_sequence_consistent &&
      planner.plan.first.kind != g001::option::Kind::Baseline &&
      selected_mechanism;

  const auto valid_margin = [](std::int64_t margin) {
    return margin != std::numeric_limits<std::int64_t>::min() &&
           margin != std::numeric_limits<std::int64_t>::max();
  };
  audit.strict_margin_improvement =
      valid_margin(candidate_score.worst_margin) &&
      valid_margin(baseline_score.worst_margin) &&
      candidate_score.worst_margin > baseline_score.worst_margin;
  audit.margin_no_worse_in_every_scenario =
      candidate_score.worst_margin_delta_vs_baseline >= 0;
  audit.own_money_no_worse =
      candidate_score.worst_own_terminal_delta_vs_baseline >= 0;

  audit.baseline_failures = map_failures(
      baseline_score.worst_failures, proof.slots.baseline_failures);
  audit.candidate_failures = map_failures(
      candidate_score.worst_failures, proof.slots.candidate_failures);
  audit.failures_no_worse = proof.slots.evaluated &&
      no_worse(audit.candidate_failures, audit.baseline_failures);

  certificate.opponent_first_checked =
      audit.scenario_proof_passed && audit.all_scenarios_opponent_first;
  certificate.observation_confirmed =
      execution.previous_execution_observation_confirmed;
  certificate.movement_unchanged = execution.movement_unchanged;
  certificate.purchase_timing_unchanged =
      execution.purchase_timing_unchanged &&
      execution.legacy_non_sell_orders_unchanged &&
      execution.only_optional_sell_changed;
  certificate.land_hire_timing_unchanged =
      execution.land_hire_timing_unchanged &&
      execution.legacy_non_sell_orders_unchanged &&
      execution.only_optional_sell_changed;
  certificate.baseline_failures = audit.baseline_failures;
  certificate.candidate_failures = audit.candidate_failures;
  certificate.baseline_worst_margin = baseline_score.worst_margin;
  certificate.candidate_worst_margin = candidate_score.worst_margin;

  audit.accepted = certificate.opponent_first_checked &&
      audit.execution_proof_passed && audit.actionable_optional_sell &&
      audit.strict_margin_improvement &&
      audit.margin_no_worse_in_every_scenario && audit.own_money_no_worse &&
      audit.failures_no_worse &&
      planner.plan.selected_inside_envelope && planner.plan.evaluated_sequences > 0;
  certificate.evaluated = audit.accepted;

  std::ostringstream reason;
  if (audit.accepted) {
    reason << "accepted selective SELL: " << audit.planner_scenario_count
           << " public causal OpponentFirst scenarios; margin "
           << baseline_score.worst_margin << " -> "
           << candidate_score.worst_margin
           << "; paired margin/own-money deltas >= 0; failures and market slots no worse; execution invariants confirmed";
  } else if (!audit.scenario_proof_passed) {
    reason << "rejected: public lower/point/upper exact scenario-set proof missing"
           << " [lower=" << proof.scenarios.lower_band_evaluated
           << " point=" << proof.scenarios.point_band_evaluated
           << " upper=" << proof.scenarios.upper_band_evaluated
           << " exact=" << proof.scenarios.exact_planner_scenario_set_replayed
           << " replayed=" << proof.scenarios.replayed_scenario_count
           << " planner=" << planner.dump_audit.scenarios.size() << ']';
  } else if (!audit.all_scenarios_opponent_first) {
    reason << "rejected: scenario set is not normalized, causal, and entirely OpponentFirst";
  } else if (!audit.execution_proof_passed) {
    reason << "rejected: movement/non-SELL/purchase/land-hire/funding or observation proof missing";
  } else if (!audit.actionable_optional_sell) {
    reason << "rejected: Baseline, inconsistent sequence, or no exact materialized selected-product replacement";
  } else if (!audit.strict_margin_improvement) {
    reason << "rejected: candidate worst margin is not strictly better than baseline";
  } else if (!audit.margin_no_worse_in_every_scenario) {
    reason << "rejected: candidate lowers margin in a paired causal scenario";
  } else if (!audit.own_money_no_worse) {
    reason << "rejected: candidate lowers own terminal money in a paired causal scenario";
  } else if (!audit.failures_no_worse) {
    reason << "rejected: mapped production/liquidity/capacity/market-slot failures worsened";
  } else if (!planner.plan.selected_inside_envelope) {
    reason << "rejected: planner selection lies outside its baseline-relative failure envelope";
  } else {
    reason << "rejected: planner did not evaluate a candidate sequence";
  }
  audit.reason = reason.str();
  return result;
}

}  // namespace robust_certificate
