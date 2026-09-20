#pragma once

#include "economic_intent_bridge.hpp"
#include "legacy_baseline_forecast.hpp"
#include "phase_input.hpp"
#include "planner_input.hpp"
#include "production_forecast.hpp"
#include "protected_queue.hpp"
#include "queue_invariant_proof.hpp"
#include "robust_certificate.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace selective_runtime {

enum class CompactionRivalEvidence : std::uint8_t {
  Disabled,
  ProvedShedLowerBound,
  CausalShedPointEstimate,
  NoRivalEvidenceRequired,
  RecentClearancePointEstimate,
  RecentClearanceProvedLowerBound,
};

// Deliberately contains neither route/opponent identity nor a realized future
// tape.  future_units and own_market_frames are the focal player's already
// selected causal plans only; current step is explicit in both.
struct Input {
  production_obligation::CompilerInput production;
  production_forecast::Config production_config;
  planner_input::CurrentOwnPublicState current;
  public_belief_runtime::Snapshot public_belief;
  planner_input::CausalMarketSchedule market_schedule;
  std::vector<legacy_baseline_forecast::OwnMarketFrame> own_market_frames;

  std::optional<economic_intent_bridge::Continuation> continuation;
  std::optional<g001::general_econ::QuantityInterval> confirmed_prior_fill;
  // A fully observation-settled active option may be paused (never silently
  // cancelled) when the current planner selects a different mechanism.
  bool allow_observation_confirmed_pause_switch{true};

  g001::general::Config planner_config{
      g001::general::conservative_deployment_config()};
  int episode_steps{720};
  int relaxation_step{220};
  int compaction_minimum_step{220};
  // Evaluator-only fast-path mode. When enabled, terminal liquidation and
  // sell compaction are the only candidates; otherwise return exact legacy
  // without invoking the general planner or creating execution continuation.
  bool compaction_only{};
  // A 24-tick causal rollout cannot certify that an incumbent sale deferred
  // beyond its horizon will remain safe under adaptive public-state policies.
  // Native deployment disables real deferrals; offline/unit proof surfaces
  // may opt in explicitly.
  bool allow_sale_deferral{true};
  // WHEAT/FERTILIZER are both buyable production inputs.  A sell-only dump
  // scenario cannot certify the later BUY/production feedback they create.
  bool allow_buyable_product_overlay{true};
  // The point-estimate mode is an offline-search surface.  Deployment keeps
  // the proved lower bound unless a frozen multi-family/held-out evaluation
  // promotes the broader causal rule.
  CompactionRivalEvidence compaction_rival_evidence{
      CompactionRivalEvidence::Disabled};
  int maximum_market_orders{10};
  int shed_capacity{100};
  std::int64_t uncertainty_cash_buffer{};
  int capacity_buffer{4};
  std::size_t maximum_search_states{500000};
  // A very wide public conservation band is uncertainty, not evidence that
  // the opponent owns that many immediately dumpable units.  Exact rollout
  // cost is linear in this upper bound, so deployment fails closed to legacy
  // instead of turning an uninformative band into an unbounded computation.
  int maximum_opponent_units_for_exact_search{384};
};

enum class Fallback : std::uint8_t {
  None,
  InvalidInput,
  ProductionUncertified,
  LegacyForecastUncertified,
  PlannerNoStrictImprovement,
  AwaitingObservation,
  OptionSwitchUncertified,
  NoOptionalIntent,
  ProvisionalQueueRejected,
  OptionalSlotAmbiguous,
  ExactPressureUnrepresentable,
  FinalQueueRejected,
  OptionalQuantityChanged,
  QueueInvariantFailed,
  RobustCertificateRejected,
  PhaseGateRejected,
  OpponentBeliefTooWide,
  AccelerationOpportunityCostRejected,
  CompactionOnlyNoCandidate,
  FullModeInvariant,
};

// A causal, opponent-identity-free guard for increasing the incumbent sale.
// The future quote is deliberately town-only: it ignores all future player
// orders and therefore represents an optimistic wait opportunity, not a
// guaranteed execution price.
struct OpportunityCostAudit {
  bool evaluated{};
  bool has_future_sale_window{};
  bool used_equilibrium_boundary_value{};
  bool passed{true};
  int current_marginal_quote{};
  int best_town_only_future_quote{};
  int best_future_step{-1};
};

[[nodiscard]] OpportunityCostAudit evaluate_acceleration_opportunity_cost(
    g001::market::Product product,
    int desired_total,
    int legacy_total,
    const g001::market::Inventory& current_market_inventory,
    const g001::rolling::FixedForecast& forecast,
    bool require_equilibrium_boundary_value = false);

struct StageTiming {
  std::int64_t obligation_us{};
  std::int64_t production_forecast_us{};
  std::int64_t legacy_forecast_us{};
  std::int64_t planner_input_us{};
  std::int64_t planner_us{};
  std::int64_t bridge_us{};
  std::int64_t provisional_queue_us{};
  std::int64_t pressure_materialize_us{};
  std::int64_t final_queue_us{};
  std::int64_t certificate_us{};
  std::int64_t phase_gate_us{};
  std::int64_t total_us{};
};

struct PressureAudit {
  std::uint32_t source_scenario_id{};
  g001::dump::BeliefBand belief_band{g001::dump::BeliefBand::Point};
  // Abstract planner proof and exact official queue relation are separate.
  g001::rolling::SameTickOrder abstract_order{
      g001::rolling::SameTickOrder::OpponentFirst};
  planner_input::MaterializeStatus status{
      planner_input::MaterializeStatus::Unrepresentable};
  std::vector<planner_input::ExactRelation> exact_relations;
  std::string reason;
};

struct QueueInvariantAudit {
  bool only_selected_product_sell_changed{};
  bool legacy_non_sell_byte_identical{};
  bool purchase_timing_byte_identical{};
  bool land_hire_timing_byte_identical{};
  bool required_funding_sells_identical{};
  bool non_target_sells_byte_identical{};
  bool non_target_sell_origins_preserved{};
  int optional_slot{-1};
  int bridge_quantity{};
  int final_optional_quantity{};
  int baseline_slot_failures{};
  int candidate_slot_failures{};
};

struct Audit {
  bool selected{};
  Fallback fallback{Fallback::InvalidInput};
  std::string reason;
  bool previous_execution_observation_confirmed{};
  bool tentative_pending_committed{};
  bool full_mode_observed{};
  bool production_certified{};
  bool legacy_forecast_certified{};
  bool planner_scenario_set_replayed{};
  economic_intent_bridge::IntentMode intent_mode{
      economic_intent_bridge::IntentMode::NoChange};
  int desired_total{};
  OpportunityCostAudit opportunity_cost;
  bool exact_terminal_liquidation_evaluated{};
  bool exact_terminal_liquidation_selected{};
  int exact_terminal_liquidation_orders{};
  int exact_terminal_liquidation_units{};
  std::int64_t exact_terminal_nominal_revenue{};
  bool monotone_sell_compaction_evaluated{};
  bool monotone_sell_compaction_selected{};
  int monotone_sell_compaction_products{};
  int monotone_sell_compaction_orders{};
  int monotone_sell_compaction_units{};
  bool state_only_continuation_committed{};
  bool virtual_no_sell_slot{};
  std::size_t expected_planner_scenarios{};
  std::size_t actual_planner_scenarios{};
  production_forecast::Audit production;
  legacy_baseline_forecast::Audit legacy;
  planner_input::Audit planner_input;
  g001::dump::Result planner_scenarios;
  economic_intent_bridge::Audit bridge;
  protected_queue::Audit provisional_queue;
  protected_queue::Audit final_queue;
  robust_certificate::Audit robust;
  phase_input::Audit phase_input;
  phased_takeover::Decision phase_decision;
  std::vector<PressureAudit> pressures;
  QueueInvariantAudit invariants;
  queue_invariant_proof::Audit queue_proof;
  StageTiming timing;
};

struct Result {
  // Exact legacy unless every layer certifies SelectiveSellOverlay.
  std::vector<g001::market::Order> orders;
  // Tentative bridge pending is committed only when orders is the certified
  // candidate at its exact requested quantity.
  std::optional<economic_intent_bridge::Continuation> continuation;
  Audit audit;
};

[[nodiscard]] Result run(const Input& input) noexcept;
[[nodiscard]] const char* fallback_name(Fallback fallback);

}  // namespace selective_runtime
