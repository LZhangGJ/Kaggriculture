#pragma once

#include <cstdint>
#include <string>

namespace phased_takeover {

// This gate sees no opponent identity, route/family identity, clone distance,
// fixed opponent tape, private opponent stock, or realized future state.
enum class Mode : std::uint8_t {
  ProtectedProduction,
  SelectiveSellOverlay,
  FullTakeover,
};

struct FailureVector {
  int production{};
  int liquidity{};
  int capacity{};
  int market_slots{};
};

struct RobustCertificate {
  bool evaluated{};
  bool opponent_first_checked{};
  bool observation_confirmed{};
  bool movement_unchanged{};
  bool purchase_timing_unchanged{};
  bool land_hire_timing_unchanged{};
  FailureVector baseline_failures{};
  FailureVector candidate_failures{};
  std::int64_t baseline_worst_margin{};
  std::int64_t candidate_worst_margin{};
};

struct Input {
  int step{};
  int relaxation_step{220};
  std::int64_t current_cash{};
  std::int64_t hard_cash_due_now{};
  // Lower-bound proceeds from the protected funding SELL prefix after replay
  // against every supplied opponent-first causal stress scenario.
  std::int64_t worst_case_required_funding_revenue{};
  std::int64_t hard_cash_horizon{};
  std::int64_t uncertainty_cash_buffer{};
  int shed_capacity{100};
  int projected_peak_shed{};
  int capacity_buffer{4};
  int due_market_orders{};
  int maximum_market_orders{10};
  bool current_irreversible_soft_miss{};
  bool pending_unconfirmed_execution{};
  bool future_plan_certified{};
  RobustCertificate selective_sell{};
  RobustCertificate full_takeover{};
};

struct Decision {
  Mode mode{Mode::ProtectedProduction};
  bool preserve_original_non_sell_orders{true};
  bool preserve_required_funding_sells{true};
  bool allow_optional_sell_replacement{};
  bool allow_purchase_replacement{};
  std::int64_t cash_coverage{};
  int capacity_slack{};
  std::string reason;
};

[[nodiscard]] Decision decide(const Input& input);
[[nodiscard]] const char* mode_name(Mode mode);

}  // namespace phased_takeover
