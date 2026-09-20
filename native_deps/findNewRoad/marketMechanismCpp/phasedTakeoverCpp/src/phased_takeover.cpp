#include "phased_takeover.hpp"

#include <algorithm>
#include <stdexcept>

namespace phased_takeover {
namespace {

bool no_worse(const FailureVector& candidate, const FailureVector& baseline) {
  return candidate.production <= baseline.production &&
         candidate.liquidity <= baseline.liquidity &&
         candidate.capacity <= baseline.capacity &&
         candidate.market_slots <= baseline.market_slots;
}

bool valid_certificate(const RobustCertificate& value,
                       bool require_purchase_timing) {
  if (!value.evaluated || !value.opponent_first_checked ||
      !value.observation_confirmed || !value.movement_unchanged ||
      !no_worse(value.candidate_failures, value.baseline_failures) ||
      value.candidate_worst_margin <= value.baseline_worst_margin) {
    return false;
  }
  if (require_purchase_timing &&
      (!value.purchase_timing_unchanged || !value.land_hire_timing_unchanged)) {
    return false;
  }
  return true;
}

}  // namespace

Decision decide(const Input& input) {
  if (input.step < 0 || input.relaxation_step < 0 || input.current_cash < 0 ||
      input.hard_cash_due_now < 0 || input.hard_cash_horizon < 0 ||
      input.worst_case_required_funding_revenue < 0 ||
      input.uncertainty_cash_buffer < 0 || input.shed_capacity < 0 ||
      input.projected_peak_shed < 0 || input.capacity_buffer < 0 ||
      input.due_market_orders < 0 || input.maximum_market_orders < 0) {
    throw std::invalid_argument("negative phased-takeover input");
  }

  Decision out;
  const std::int64_t reserve = input.hard_cash_horizon +
                               input.uncertainty_cash_buffer;
  out.cash_coverage = input.current_cash - reserve;
  out.capacity_slack = input.shed_capacity - input.projected_peak_shed;

  if (!input.future_plan_certified) {
    out.reason = "future production obligations are not certified";
    return out;
  }
  if (input.current_irreversible_soft_miss) {
    out.reason = "current production miss requires recovery before takeover";
    return out;
  }
  if (input.pending_unconfirmed_execution) {
    out.reason = "previous market execution is not observation-confirmed";
    return out;
  }
  if (input.current_cash + input.worst_case_required_funding_revenue <
      input.hard_cash_due_now) {
    out.reason = "cash cannot cover current hard production orders";
    return out;
  }
  if (input.due_market_orders > input.maximum_market_orders) {
    out.reason = "hard production orders exceed the market slot budget";
    return out;
  }

  const bool selective_safe = valid_certificate(input.selective_sell, true);
  const bool financially_independent = out.cash_coverage >= 0 &&
      out.capacity_slack >= input.capacity_buffer &&
      input.due_market_orders < input.maximum_market_orders;
  const bool full_safe = valid_certificate(input.full_takeover, false);

  if (input.step >= input.relaxation_step && financially_independent && full_safe) {
    out.mode = Mode::FullTakeover;
    out.preserve_original_non_sell_orders = false;
    out.preserve_required_funding_sells = false;
    out.allow_optional_sell_replacement = true;
    out.allow_purchase_replacement = true;
    out.reason = "solvency/capacity coverage and robust full-takeover certificate passed";
    return out;
  }

  // Before the incumbent route's relaxation point, even a sell-only edit can
  // change public prices and therefore a rival's competing seed/animal buys.
  // Preserving our non-SELL bytes is not enough to prove that those buys will
  // still fill.  Keep the complete G001 market queue until the protected
  // expansion phase is over; relaxation_step is permission to intervene, not
  // a forced takeover.
  if (input.step >= input.relaxation_step && selective_safe) {
    out.mode = Mode::SelectiveSellOverlay;
    out.allow_optional_sell_replacement = true;
    out.reason = "full takeover not ready; sell-only intervention remains certified";
    return out;
  }

  out.reason = input.step < input.relaxation_step
      ? "protected expansion phase preserves the complete incumbent market queue"
      : "financial coverage or robust takeover certificate is insufficient";
  return out;
}

const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::ProtectedProduction: return "protected-production";
    case Mode::SelectiveSellOverlay: return "selective-sell-overlay";
    case Mode::FullTakeover: return "full-takeover";
  }
  return "unknown";
}

}  // namespace phased_takeover
