// Licensed under the Apache License, Version 2.0.
#pragma once

#include "economic_intent_bridge.hpp"
#include "native_observation_adapter.hpp"
#include "native_selective_input.hpp"
#include "simulator.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fastkag {

enum class NativeMarketArm : std::uint8_t {
  LegacyDefault = 0,
  SelectiveProtected = 1,
  Phased = 2,
};

enum class NativeCompactionEvidenceMode : std::uint8_t {
  ArmDefault,
  Disabled,
  ProvedLower,
  CausalPoint,
  NoEvidenceRequired,
  RecentClearancePoint,
  RecentClearanceLower,
};

// Offline evaluator-only overrides. Deployment and the existing Python ABI do
// not enable this object, so arm defaults remain byte-identical.
struct NativePhasedEvaluationProfile {
  bool enabled{};
  int compaction_minimum_step{-1};  // -1 preserves the selected arm default.
  // Evaluator-only: values <= 0 preserve selective_runtime's deployment cap.
  int maximum_opponent_units_for_exact_search{-1};
  // Evaluator-only tri-state: -1 preserves native, 0 disables, 1 enables.
  int allow_sale_deferral{-1};
  // Evaluator-only tri-state: terminal/compaction fast paths without planner.
  int compaction_only{-1};
  NativeCompactionEvidenceMode evidence_mode{
      NativeCompactionEvidenceMode::ArmDefault};
  bool capture_step_diagnostics{};
};

struct NativePhasedRuntimeState {
  int last_mode = 0;  // phased_takeover::Mode without leaking the dependency.
  int mode_since_step = -1;
  std::uint64_t generation = 0;
  bool v2_audit_completed = false;
  // arm0 never emplaces or calls either object.  arm1/2 own one instance per
  // seat so the official stage_submission -> next observe boundary is exact.
  std::optional<native_observation_adapter::Adapter> observation;
  std::optional<economic_intent_bridge::Continuation> continuation;
  bool selective_disabled = false;
  NativePhasedEvaluationProfile evaluation_profile{};
};

inline constexpr int kNativeRuntimeFallbackKinds =
    static_cast<int>(selective_runtime::Fallback::FullModeInvariant) + 1;
inline constexpr int kNativeBridgeReasonKinds =
    static_cast<int>(economic_intent_bridge::Reason::SellIntent) + 1;
inline constexpr int kNativeIntentModeKinds =
    static_cast<int>(economic_intent_bridge::IntentMode::StateOnlyContinuation) + 1;

struct NativePhasedStepAudit {
  int step = -1;
  int arm = 0;
  int mode = 0;
  bool protected_queue_called = false;
  bool phase_gate_called = false;
  bool no_candidate = false;
  bool uncertified = false;
  bool exact_legacy_fallback = false;
  bool protected_certificate_passed = false;
  bool production_dag_certified = false;
  bool v2_exact_legacy_fallback = false;
  bool full_selected = false;
  int production_obligation_nodes = 0;
  int due_legacy_bindings = 0;
  int protected_non_sell_orders = 0;
  int required_funding_sell_orders = 0;
  int required_funding_sell_units = 0;
  int optional_replacement_orders = 0;
  int optional_replacement_units = 0;
  std::size_t search_states = 0;
  bool observation_reset = false;
  bool observation_observed = false;
  bool observation_staged = false;
  bool observation_failure = false;
  bool builder_called = false;
  bool builder_accepted = false;
  bool runtime_called = false;
  bool runtime_selected = false;
  bool confirmation_supplied = false;
  bool confirmed_fill_clamped = false;
  bool continuation_pending_before = false;
  bool continuation_pending_after = false;
  int runtime_fallback = -1;
  int bridge_reason = -1;
  int intent_mode = -1;
  int desired_total = 0;
  bool state_only_continuation_committed = false;
  bool virtual_no_sell_slot = false;
  int market_changed_slots = 0;
  int opponent_total_upper_units = 0;
  int maximum_scenario_dump_units = 0;
  bool planner_lower_band = false;
  bool planner_point_band = false;
  bool planner_upper_band = false;
  bool planner_scenario_set_replayed = false;
  std::size_t expected_planner_scenarios = 0;
  std::size_t actual_planner_scenarios = 0;
  std::int64_t runtime_total_us = 0;
  std::int64_t runtime_planner_us = 0;
  std::int64_t observation_us = 0;
  std::int64_t builder_us = 0;
  std::int64_t native_overlay_total_us = 0;
  std::string runtime_fallback_name;
  std::string observation_reason;
  std::string reason;
  bool selected_general_robust{};
  bool selected_exact_terminal_liquidation{};
  bool selected_sell_compaction{};
  int selected_product{-1};
  std::uint16_t changed_product_mask{};
  std::array<int, N_PRODUCTS> diagnostic_belief_lower{};
  std::array<int, N_PRODUCTS> diagnostic_belief_point{};
  std::array<int, N_PRODUCTS> diagnostic_belief_upper{};
  std::vector<Action> diagnostic_legacy_market;
  std::vector<Action> diagnostic_candidate_market;
};

struct NativePhasedMarketAudit {
  int arm = 0;
  int calls = 0;
  int protected_steps = 0;
  int selective_steps = 0;
  int full_steps = 0;
  int protected_queue_calls = 0;
  int phase_gate_calls = 0;
  int no_candidate_steps = 0;
  int uncertified_steps = 0;
  int exact_legacy_fallbacks = 0;
  int production_dag_certified_steps = 0;
  int v2_exact_legacy_fallbacks = 0;
  int production_obligation_nodes = 0;
  int due_legacy_bindings = 0;
  int protected_non_sell_orders = 0;
  int required_funding_sell_orders = 0;
  int required_funding_sell_units = 0;
  int optional_replacement_orders = 0;
  int optional_replacement_units = 0;
  std::uint64_t search_states = 0;
  int observation_resets = 0;
  int observation_updates = 0;
  int observation_stages = 0;
  int observation_failures = 0;
  int builder_calls = 0;
  int builder_rejections = 0;
  int runtime_calls = 0;
  int runtime_selected_steps = 0;
  int sell_compaction_selected_steps = 0;
  int exact_terminal_liquidation_selected_steps = 0;
  int general_robust_selected_steps = 0;
  int runtime_fallback_steps = 0;
  int market_changed_steps = 0;
  int market_changed_slots = 0;
  int maximum_opponent_total_upper_units = 0;
  int maximum_scenario_dump_units = 0;
  std::int64_t runtime_total_us = 0;
  std::int64_t runtime_planner_us = 0;
  std::int64_t observation_us = 0;
  std::int64_t builder_us = 0;
  std::int64_t native_overlay_total_us = 0;
  std::array<int, kNativeRuntimeFallbackKinds> runtime_fallback_counts{};
  std::array<int, kNativeRuntimeFallbackKinds> zero_target_fallback_counts{};
  std::array<int, kNativeRuntimeFallbackKinds> virtual_no_sell_fallback_counts{};
  std::array<int, kNativeBridgeReasonKinds> bridge_reason_counts{};
  std::array<int, kNativeBridgeReasonKinds> virtual_no_sell_bridge_reason_counts{};
  std::array<std::array<int, kNativeRuntimeFallbackKinds>,
             kNativeBridgeReasonKinds> bridge_reason_fallback_counts{};
  std::array<int, kNativeIntentModeKinds> intent_mode_counts{};
  int zero_target_steps = 0;
  int state_only_continuation_steps = 0;
  int virtual_no_sell_steps = 0;
  int first_virtual_no_sell_rejection_step = -1;
  int first_sell_intent_rejection_step = -1;
  std::string first_virtual_no_sell_rejection_reason;
  std::string first_sell_intent_rejection_reason;
  bool full_ever_selected = false;
  int last_mode = 0;
  int last_runtime_fallback = -1;
  std::string last_runtime_fallback_name;
  std::string last_observation_reason;
  std::string last_reason;
};

struct NativePhasedMarketResult {
  std::vector<Action> market;
  NativePhasedStepAudit audit;
};

[[nodiscard]] NativePhasedMarketResult compose_native_phased_market(
    const Simulator& env,
    int player,
    NativeMarketArm arm,
    const PlayerAction& legacy_action,
    const std::vector<native_selective_input::SelectedPlanFrame>& future_plan,
    NativePhasedRuntimeState& state);

void accumulate_native_phased_audit(
    NativePhasedMarketAudit& total,
    const NativePhasedStepAudit& step);

const char* native_market_arm_name(NativeMarketArm arm);
const char* native_phased_mode_name(int mode);

}  // namespace fastkag
