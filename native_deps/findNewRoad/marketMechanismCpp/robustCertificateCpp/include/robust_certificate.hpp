#pragma once

#include "general_planner.hpp"
#include "phased_takeover.hpp"

#include <cstddef>
#include <string>

namespace robust_certificate {

// Explicit proof that the planner scores came from the same public causal
// scenario set inspected below.  These facts are not inferred from a good
// score and are deliberately supplied independently by the caller.
struct ScenarioProof {
  bool public_observations_only{};
  bool lower_band_evaluated{};
  bool point_band_evaluated{};
  bool upper_band_evaluated{};
  bool exact_planner_scenario_set_replayed{};
  std::size_t replayed_scenario_count{};
};

// Typed execution invariants required for a sell-only overlay.  No field is
// defaulted to true: an omitted proof fails closed.
struct ExecutionProof {
  bool movement_unchanged{};
  bool only_optional_sell_changed{};
  bool legacy_non_sell_orders_unchanged{};
  bool purchase_timing_unchanged{};
  bool land_hire_timing_unchanged{};
  bool required_funding_sells_unchanged{};
  bool previous_execution_observation_confirmed{};
  // Preferred ABI: ReplaceTotal(q) was materialized and the exact selected-
  // product total after queue composition is q. q may be zero.
  bool replacement_materialized{};
  int selected_total{};
  // Required when a SELL-class planner option is certified as ReplaceTotal(0):
  // the caller independently replayed the selected option's current action and
  // observed an exact zero (for example ReservationPause/ImpactPause).
  // Queue composition alone cannot prove this planner-action fact. Hold is
  // intrinsically zero and does not require it.
  bool zero_option_action_replayed{};
  // Backward-compatible positive-SELL proof. It cannot represent ReplaceTotal(0).
  bool optional_sell_intent_materialized{};
  int optional_sell_requested_units{};
};

struct SlotProof {
  bool evaluated{};
  int baseline_failures{};
  int candidate_failures{};
};

struct Proof {
  ScenarioProof scenarios;
  ExecutionProof execution;
  SlotProof slots;
};

struct Audit {
  bool accepted{};
  bool scenario_proof_passed{};
  bool all_scenarios_opponent_first{};
  bool execution_proof_passed{};
  bool actionable_optional_sell{};
  bool replacement_proof_passed{};
  int selected_total{};
  bool strict_margin_improvement{};
  bool margin_no_worse_in_every_scenario{};
  bool own_money_no_worse{};
  bool failures_no_worse{};
  std::size_t planner_scenario_count{};
  phased_takeover::FailureVector baseline_failures{};
  phased_takeover::FailureVector candidate_failures{};
  std::string reason;
};

struct Result {
  phased_takeover::RobustCertificate certificate;
  Audit audit;
};

// Compiles only a selective-SELL certificate.  Full-takeover certification is
// a separate problem because this proof requires every legacy purchase and
// land/hire timing to remain unchanged.
[[nodiscard]] Result compile_selective_sell(
    const g001::general::Result& planner, const Proof& proof);

}  // namespace robust_certificate
