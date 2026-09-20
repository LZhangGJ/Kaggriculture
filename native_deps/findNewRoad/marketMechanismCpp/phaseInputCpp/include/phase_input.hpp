#pragma once

#include "market.hpp"
#include "phased_takeover.hpp"
#include "production_obligation.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace phase_input {

// Produced by protectedQueue's exact OpponentFirst scenario replays. Revenue
// means committed proceeds from RequiredFundingSell slots only, never requested
// quantity or optional/legacy sale revenue.
struct FundingScenarioAudit {
  bool evaluated{};
  bool opponent_first{};
  int required_funding_requested_units{};
  int required_funding_committed_units{};
  std::int64_t committed_required_funding_revenue{};
  int post_queue_shed_used{};  // products + animals
};

struct ProtectedQueueAuditSummary {
  bool evaluated{};
  bool certificate_passed{};
  bool pending_unconfirmed_execution{};
  std::vector<FundingScenarioAudit> scenarios;
};

// No legacy orders/tape and no route/opponent identity are accepted here.
struct Input {
  int step{};
  int relaxation_step{220};
  int shed_capacity{100};
  int maximum_market_orders{10};
  std::int64_t uncertainty_cash_buffer{};
  int capacity_buffer{4};
  bool pending_unconfirmed_execution{};
  bool future_plan_certified{};

  g001::market::PlayerMarketState own{};
  production_obligation::CompileResult production;
  ProtectedQueueAuditSummary protected_queue;
  phased_takeover::RobustCertificate selective_sell;
  phased_takeover::RobustCertificate full_takeover;
};

struct StepCashAudit {
  int execution_step{-1};
  std::int64_t incremental_cash{};
  std::int64_t cumulative_cash{};
  bool used_cash_reserve_node{};
};

struct Audit {
  std::vector<StepCashAudit> cash_by_step;
  int current_shed_used{};
  int projected_peak_shed{};
  int due_market_orders{};
  int soft_miss_nodes{};
  int funding_scenarios{};
  bool all_funding_scenarios_opponent_first{};
  bool all_required_funding_filled{};
  bool used_required_funding_lower_bound{};
  bool cumulative_cash_consistent{true};
  std::string reason;
};

struct Result {
  phased_takeover::Input phased;
  Audit audit;
};

[[nodiscard]] Result build(const Input& input);

}  // namespace phase_input
