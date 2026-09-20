#include "phase_input.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace phase_input {
namespace {

using production_obligation::Item;
using production_obligation::NodeKind;
using production_obligation::ObligationNode;

bool acquisition(NodeKind kind) {
  return kind == NodeKind::BuySeed || kind == NodeKind::BuyProduct ||
         kind == NodeKind::BuyAnimal || kind == NodeKind::Hire ||
         kind == NodeKind::BuyLand;
}

bool capacity_acquisition(NodeKind kind) {
  return kind == NodeKind::BuyProduct || kind == NodeKind::BuyAnimal;
}

int item_index(Item item) { return static_cast<int>(item); }

struct CashAtStep {
  std::int64_t acquisition_sum{};
  std::int64_t reserve_increment{};
  std::int64_t reserve_cumulative{};
  bool has_reserve{};
};

std::map<int, CashAtStep> collect_cash(const Input& input, Audit& audit) {
  std::map<int, CashAtStep> by_step;
  std::set<int> seen_ids;
  for (std::size_t index = 0; index < input.production.nodes.size(); ++index) {
    const ObligationNode& node = input.production.nodes[index];
    if (node.execution_step < input.step) continue;
    if (node.id >= 0 && !seen_ids.insert(node.id).second) continue;
    CashAtStep& at = by_step[node.execution_step];
    if (node.kind == NodeKind::CashReserve) {
      at.has_reserve = true;
      // Duplicate reserve summaries describe the same execution bucket.
      at.reserve_increment = std::max<std::int64_t>(
          at.reserve_increment, std::max(0, node.cash_quote));
      at.reserve_cumulative = std::max<std::int64_t>(
          at.reserve_cumulative, std::max(0, node.cumulative_cash_quote));
    } else if (acquisition(node.kind)) {
      at.acquisition_sum += std::max(0, node.cash_quote);
    }
  }

  std::int64_t previous_cumulative = 0;
  for (auto& [step, at] : by_step) {
    std::int64_t amount = at.acquisition_sum;
    if (at.has_reserve) {
      const std::int64_t cumulative_delta = std::max<std::int64_t>(
          0, at.reserve_cumulative - previous_cumulative);
      amount = at.reserve_increment > 0 ? at.reserve_increment : cumulative_delta;
      if (at.reserve_increment > 0 && at.reserve_cumulative > 0 &&
          at.reserve_increment != cumulative_delta) {
        audit.cumulative_cash_consistent = false;
        // Do not silently understate a malformed DAG.
        amount = std::max(at.reserve_increment, cumulative_delta);
      }
      if (at.acquisition_sum > 0 && amount != at.acquisition_sum) {
        audit.cumulative_cash_consistent = false;
        // The reserve summary and the acquisition nodes are redundant views of
        // one bucket.  Count the bucket once, conservatively using the larger
        // view when a malformed externally-constructed DAG disagrees.
        amount = std::max(amount, at.acquisition_sum);
      }
      previous_cumulative = std::max(previous_cumulative, at.reserve_cumulative);
    } else {
      previous_cumulative += amount;
    }
    audit.cash_by_step.push_back(
        {step, amount, previous_cumulative, at.has_reserve});
  }
  return by_step;
}

int projected_peak(const Input& input, Audit& audit) {
  std::array<int, production_obligation::kItems> stock{};
  for (int product = 0; product < production_obligation::kProducts; ++product)
    stock[product] = std::max(0, input.own.shed[product]);
  for (int animal = 0; animal < production_obligation::kAnimals; ++animal)
    stock[9 + animal] = std::max(0, input.own.animals[animal]);

  auto total = [&]() {
    int value = 0;
    for (const int quantity : stock) value += std::max(0, quantity);
    return value;
  };
  audit.current_shed_used = total();
  int peak = audit.current_shed_used;

  std::map<int, std::vector<const ObligationNode*>> unit_pickups;
  std::map<int, std::vector<const ObligationNode*>> market_buys;
  std::set<int> seen_ids;
  for (const ObligationNode& node : input.production.nodes) {
    if (node.id >= 0 && !seen_ids.insert(node.id).second) continue;
    if (node.kind == NodeKind::Pickup && node.consumer_step >= input.step)
      unit_pickups[node.consumer_step].push_back(&node);
    if (capacity_acquisition(node.kind) && node.execution_step >= input.step)
      market_buys[node.execution_step].push_back(&node);
  }
  std::set<int> steps;
  for (const auto& [step, nodes] : unit_pickups) {
    (void)nodes;
    steps.insert(step);
  }
  for (const auto& [step, nodes] : market_buys) {
    (void)nodes;
    steps.insert(step);
  }
  for (const int step : steps) {
    // Official unit phase precedes market: PICKUP frees room first.
    for (const ObligationNode* node : unit_pickups[step]) {
      const int item = item_index(node->item);
      if (item >= 0 && item < production_obligation::kItems)
        stock[item] = std::max(0, stock[item] - std::max(0, node->quantity));
    }
    for (const ObligationNode* node : market_buys[step]) {
      const int item = item_index(node->item);
      if (item >= 0 && item < production_obligation::kItems)
        stock[item] += std::max(0, node->quantity);
    }
    peak = std::max(peak, total());
  }
  for (const FundingScenarioAudit& scenario : input.protected_queue.scenarios) {
    if (scenario.evaluated)
      peak = std::max(peak, std::max(0, scenario.post_queue_shed_used));
  }
  audit.projected_peak_shed = peak;
  return peak;
}

std::int64_t certified_funding(const Input& input, Audit& audit) {
  audit.funding_scenarios = static_cast<int>(input.protected_queue.scenarios.size());
  audit.all_funding_scenarios_opponent_first =
      input.protected_queue.evaluated && input.protected_queue.certificate_passed &&
      !input.protected_queue.scenarios.empty();
  audit.all_required_funding_filled = audit.all_funding_scenarios_opponent_first;
  std::int64_t lower_bound = std::numeric_limits<std::int64_t>::max();
  for (const FundingScenarioAudit& scenario : input.protected_queue.scenarios) {
    if (!scenario.evaluated || !scenario.opponent_first ||
        scenario.required_funding_requested_units < 0 ||
        scenario.required_funding_committed_units < 0 ||
        scenario.committed_required_funding_revenue < 0 ||
        scenario.post_queue_shed_used < 0) {
      audit.all_funding_scenarios_opponent_first = false;
      audit.all_required_funding_filled = false;
    }
    if (scenario.required_funding_committed_units <
        scenario.required_funding_requested_units) {
      audit.all_required_funding_filled = false;
    }
    lower_bound = std::min(lower_bound,
                           std::max<std::int64_t>(
                               0, scenario.committed_required_funding_revenue));
  }
  if (!audit.all_funding_scenarios_opponent_first ||
      !audit.all_required_funding_filled ||
      lower_bound == std::numeric_limits<std::int64_t>::max()) {
    return 0;
  }
  audit.used_required_funding_lower_bound = true;
  return lower_bound;
}

}  // namespace

Result build(const Input& input) {
  if (input.step < 0 || input.relaxation_step < 0 || input.shed_capacity < 0 ||
      input.maximum_market_orders < 0 || input.uncertainty_cash_buffer < 0 ||
      input.capacity_buffer < 0 || input.own.money < 0) {
    throw std::invalid_argument("negative phase-input field");
  }
  Result result;
  result.phased.step = input.step;
  result.phased.relaxation_step = input.relaxation_step;
  result.phased.current_cash = input.own.money;
  result.phased.uncertainty_cash_buffer = input.uncertainty_cash_buffer;
  result.phased.shed_capacity = input.shed_capacity;
  result.phased.capacity_buffer = input.capacity_buffer;
  result.phased.maximum_market_orders = input.maximum_market_orders;
  result.phased.selective_sell = input.selective_sell;
  result.phased.full_takeover = input.full_takeover;

  collect_cash(input, result.audit);
  for (const StepCashAudit& cash : result.audit.cash_by_step) {
    if (cash.execution_step == input.step)
      result.phased.hard_cash_due_now += cash.incremental_cash;
    else if (cash.execution_step > input.step)
      result.phased.hard_cash_horizon += cash.incremental_cash;
  }

  std::set<int> seen_ids;
  for (const ObligationNode& node : input.production.nodes) {
    if (node.id >= 0 && !seen_ids.insert(node.id).second) continue;
    if (acquisition(node.kind) && node.execution_step == input.step)
      ++result.phased.due_market_orders;
    if (node.kind == NodeKind::SoftMiss) ++result.audit.soft_miss_nodes;
  }
  result.audit.due_market_orders = result.phased.due_market_orders;
  result.phased.current_irreversible_soft_miss =
      input.production.soft_misses > 0 || result.audit.soft_miss_nodes > 0;
  result.phased.pending_unconfirmed_execution =
      input.pending_unconfirmed_execution ||
      input.protected_queue.pending_unconfirmed_execution;
  result.phased.future_plan_certified =
      input.future_plan_certified && input.production.feasible &&
      result.audit.cumulative_cash_consistent;
  result.phased.projected_peak_shed = projected_peak(input, result.audit);
  result.phased.worst_case_required_funding_revenue =
      certified_funding(input, result.audit);

  std::ostringstream reason;
  reason << "cash due now=" << result.phased.hard_cash_due_now
         << " future-only horizon=" << result.phased.hard_cash_horizon
         << " peak shed=" << result.phased.projected_peak_shed
         << " due orders=" << result.phased.due_market_orders
         << " funding lower bound="
         << result.phased.worst_case_required_funding_revenue
         << " soft misses=" << result.audit.soft_miss_nodes
         << " pending=" << result.phased.pending_unconfirmed_execution;
  result.audit.reason = reason.str();
  return result;
}

}  // namespace phase_input
