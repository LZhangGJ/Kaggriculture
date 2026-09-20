// Licensed under the Apache License, Version 2.0.
#include "native_phased_market.hpp"

#include "phased_takeover.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <optional>
#include <string>
#include <utility>

namespace fastkag {
namespace {

namespace phase = phased_takeover;
using Clock = std::chrono::steady_clock;

bool same_action(const Action& lhs, const Action& rhs) noexcept {
  return lhs.op == rhs.op && lhs.item == rhs.item && lhs.quantity == rhs.quantity;
}

int changed_slots(const std::vector<Action>& lhs,
                  const std::vector<Action>& rhs) noexcept {
  int changed = 0;
  const std::size_t count = std::max(lhs.size(), rhs.size());
  for (std::size_t i = 0; i < count; ++i) {
    if (i >= lhs.size() || i >= rhs.size() || !same_action(lhs[i], rhs[i]))
      ++changed;
  }
  return changed;
}

std::int64_t elapsed_us(Clock::time_point begin) noexcept {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin)
      .count();
}

bool continuation_pending(
    const std::optional<economic_intent_bridge::Continuation>& value) noexcept {
  return value.has_value() && value->pending.has_value();
}

void copy_queue_audit(const protected_queue::Audit& source,
                      NativePhasedStepAudit& target) noexcept {
  target.protected_certificate_passed = source.certificate_passed;
  target.protected_non_sell_orders = source.protected_non_sell_orders;
  target.required_funding_sell_orders = source.required_funding_sell_orders;
  target.required_funding_sell_units = source.required_funding_sell_units;
  target.optional_replacement_orders = source.optional_replacement_orders;
  target.optional_replacement_units = source.optional_replacement_units;
  target.search_states = source.search_states;
}

}  // namespace

NativePhasedMarketResult compose_native_phased_market(
    const Simulator& env,
    int player,
    NativeMarketArm arm,
    const PlayerAction& legacy_action,
    const std::vector<native_selective_input::SelectedPlanFrame>& future_plan,
    NativePhasedRuntimeState& state) {
  NativePhasedMarketResult result;
  result.market = legacy_action.market;
  result.audit.step = env.step_count();
  result.audit.arm = static_cast<int>(arm);
  result.audit.mode = static_cast<int>(phase::Mode::ProtectedProduction);
  if (state.evaluation_profile.capture_step_diagnostics)
    result.audit.diagnostic_legacy_market = legacy_action.market;

  // arm0 returns before constructing/calling any new runtime component.
  if (arm == NativeMarketArm::LegacyDefault) {
    result.audit.reason = "legacy default bypass";
    return result;
  }

  const auto total_begin = Clock::now();
  if (state.selective_disabled && env.step_count() != 0) {
    result.audit.no_candidate = true;
    result.audit.uncertified = true;
    result.audit.exact_legacy_fallback = true;
    result.audit.reason =
        "selective runtime permanently disabled for this episode after a causal boundary failure";
    result.audit.native_overlay_total_us = elapsed_us(total_begin);
    return result;
  }
  const auto continuation_before = state.continuation;
  result.audit.continuation_pending_before = continuation_pending(continuation_before);
  std::optional<public_belief_runtime::Snapshot> snapshot;
  std::optional<g001::general_econ::QuantityInterval> confirmed;

  auto record_observation_failure = [&](std::string reason) {
    result.audit.observation_failure = true;
    if (!result.audit.observation_reason.empty())
      result.audit.observation_reason += "; ";
    result.audit.observation_reason += reason;
    result.audit.exact_legacy_fallback = true;
    result.audit.uncertified = true;
  };

  // Stage exactly the returned action. Candidate-stage failure rolls back both
  // the action and the tentative continuation before attempting legacy stage.
  auto stage_actual = [&] {
    if (!state.observation.has_value()) return;
    PlayerAction submission = legacy_action;
    submission.market = result.market;
    try {
      state.observation->stage_submission(env, submission);
      result.audit.observation_staged = true;
      return;
    } catch (const std::exception& error) {
      const bool candidate_was_selected = result.audit.runtime_selected;
      record_observation_failure(std::string("submission stage failed: ") + error.what());
      state.selective_disabled = true;
      if (!candidate_was_selected) return;
      result.market = legacy_action.market;
      result.audit.runtime_selected = false;
      result.audit.mode = static_cast<int>(phase::Mode::ProtectedProduction);
      result.audit.exact_legacy_fallback = true;
      state.continuation = continuation_before;
      // stage_submission validates fully before installing pending_. Retry
      // legacy only when that invariant is observably intact.
      if (state.observation->awaiting_observation()) {
        result.audit.observation_reason +=
            "; legacy retry suppressed because adapter is already pending";
        return;
      }
      try {
        state.observation->stage_submission(env, legacy_action);
        result.audit.observation_staged = true;
      } catch (const std::exception& legacy_error) {
        result.audit.observation_reason +=
            std::string("; legacy stage also failed: ") + legacy_error.what();
      }
    }
  };

  const auto observation_begin = Clock::now();
  try {
    if (env.step_count() == 0) {
      state.observation.emplace();
      snapshot = state.observation->reset(env, player);
      state.continuation.reset();
      state.selective_disabled = false;
      result.audit.observation_reset = true;
    } else if (!state.observation.has_value() || !state.observation->initialized()) {
      record_observation_failure("nonzero step reached without a step0 observation reset");
    } else if (!state.observation->awaiting_observation()) {
      record_observation_failure("previous submitted action was not staged for observation");
    } else {
      auto update = state.observation->observe(env);
      snapshot = std::move(update.snapshot);
      result.audit.observation_observed = true;
      if (continuation_pending(state.continuation)) {
        const int product = static_cast<int>(state.continuation->option.product);
        if (product < 0 || product >= N_PRODUCTS) {
          record_observation_failure("pending continuation has an invalid product");
          state.selective_disabled = true;
        } else {
          g001::general_econ::QuantityInterval interval;
          interval.lower = update.derived_fill.lower[product];
          interval.point = update.derived_fill.point[product];
          interval.upper = update.derived_fill.upper[product];
          interval.consistent = !update.derived_fill.clamped[product];
          result.audit.confirmed_fill_clamped = update.derived_fill.clamped[product];
          if (!interval.consistent) {
            record_observation_failure(
                "own-shed conservation fill interval required clamping");
            state.selective_disabled = true;
          } else {
            confirmed = interval;
            result.audit.confirmation_supplied = true;
          }
        }
      }
    }
  } catch (const std::exception& error) {
    record_observation_failure(std::string("observation exception: ") + error.what());
    if (continuation_pending(state.continuation)) state.selective_disabled = true;
  }
  result.audit.observation_us = elapsed_us(observation_begin);

  if (!snapshot.has_value() || state.selective_disabled) {
    result.audit.no_candidate = true;
    result.audit.exact_legacy_fallback = true;
    result.audit.reason = state.selective_disabled
        ? "selective runtime disabled after an unsettled causal boundary"
        : "public observation unavailable; exact legacy fallback";
    stage_actual();
    result.audit.continuation_pending_after = continuation_pending(state.continuation);
    result.audit.native_overlay_total_us = elapsed_us(total_begin);
    return result;
  }

  native_selective_input::BuildInput build_source;
  build_source.simulator = &env;
  build_source.player = player;
  build_source.current_legacy = legacy_action;
  build_source.future_plan = future_plan;
  build_source.public_belief = *snapshot;
  build_source.continuation = state.continuation;
  build_source.confirmed_prior_fill = confirmed;
  result.audit.builder_called = true;
  const auto builder_begin = Clock::now();
  auto built = native_selective_input::build(build_source);
  result.audit.builder_us = elapsed_us(builder_begin);
  result.audit.builder_accepted = built.accepted;
  if (!built.accepted) {
    result.audit.no_candidate = true;
    result.audit.uncertified = true;
    result.audit.exact_legacy_fallback = true;
    result.audit.reason = "selective input rejected: " + built.reason;
    if (continuation_pending(state.continuation) && confirmed.has_value())
      state.selective_disabled = true;
    stage_actual();
    result.audit.continuation_pending_after = continuation_pending(state.continuation);
    result.audit.native_overlay_total_us = elapsed_us(total_begin);
    return result;
  }
  if (state.evaluation_profile.capture_step_diagnostics) {
    result.audit.diagnostic_belief_lower =
        built.input.public_belief.belief.total_interval.lower;
    result.audit.diagnostic_belief_point =
        built.input.public_belief.belief.total;
    result.audit.diagnostic_belief_upper =
        built.input.public_belief.belief.total_interval.upper;
  }

  // Arm 2 is deliberately an offline-only causal generalization probe.  It
  // tests late same-product SELL compaction without requiring a positive
  // rival-stock estimate; arm 1 keeps compaction disabled.  Neither mode
  // receives opponent identity, route identity, or realized future state.
  if (arm == NativeMarketArm::Phased) {
    built.input.compaction_rival_evidence =
        selective_runtime::CompactionRivalEvidence::NoRivalEvidenceRequired;
    built.input.compaction_minimum_step = 600;
  }
  if (state.evaluation_profile.enabled) {
    if (state.evaluation_profile.compaction_minimum_step >= 0)
      built.input.compaction_minimum_step =
          state.evaluation_profile.compaction_minimum_step;
    if (state.evaluation_profile.maximum_opponent_units_for_exact_search > 0)
      built.input.maximum_opponent_units_for_exact_search =
          state.evaluation_profile.maximum_opponent_units_for_exact_search;
    if (state.evaluation_profile.allow_sale_deferral >= 0)
      built.input.allow_sale_deferral =
          state.evaluation_profile.allow_sale_deferral != 0;
    if (state.evaluation_profile.compaction_only >= 0)
      built.input.compaction_only =
          state.evaluation_profile.compaction_only != 0;
    switch (state.evaluation_profile.evidence_mode) {
      case NativeCompactionEvidenceMode::ArmDefault:
        break;
      case NativeCompactionEvidenceMode::Disabled:
        built.input.compaction_rival_evidence =
            selective_runtime::CompactionRivalEvidence::Disabled;
        break;
      case NativeCompactionEvidenceMode::ProvedLower:
        built.input.compaction_rival_evidence =
            selective_runtime::CompactionRivalEvidence::ProvedShedLowerBound;
        break;
      case NativeCompactionEvidenceMode::CausalPoint:
        built.input.compaction_rival_evidence =
            selective_runtime::CompactionRivalEvidence::CausalShedPointEstimate;
        break;
      case NativeCompactionEvidenceMode::NoEvidenceRequired:
        built.input.compaction_rival_evidence =
            selective_runtime::CompactionRivalEvidence::NoRivalEvidenceRequired;
        break;
      case NativeCompactionEvidenceMode::RecentClearancePoint:
        built.input.compaction_rival_evidence = selective_runtime::
            CompactionRivalEvidence::RecentClearancePointEstimate;
        break;
      case NativeCompactionEvidenceMode::RecentClearanceLower:
        built.input.compaction_rival_evidence = selective_runtime::
            CompactionRivalEvidence::RecentClearanceProvedLowerBound;
        break;
    }
  }

  result.audit.runtime_called = true;
  const auto runtime = selective_runtime::run(built.input);
  for (const int quantity : built.input.public_belief.belief.total_interval.upper)
    result.audit.opponent_total_upper_units += std::max(0, quantity);
  for (const auto& weighted : runtime.audit.planner_scenarios.scenarios) {
    int dumped = 0;
    for (const auto& dump : weighted.scenario.dumps)
      dumped += std::max(0, dump.quantity);
    result.audit.maximum_scenario_dump_units =
        std::max(result.audit.maximum_scenario_dump_units, dumped);
  }
  result.audit.runtime_fallback = static_cast<int>(runtime.audit.fallback);
  result.audit.runtime_fallback_name = selective_runtime::fallback_name(runtime.audit.fallback);
  result.audit.bridge_reason = static_cast<int>(runtime.audit.bridge.reason);
  result.audit.intent_mode = static_cast<int>(runtime.audit.intent_mode);
  result.audit.desired_total = runtime.audit.desired_total;
  result.audit.state_only_continuation_committed =
      runtime.audit.state_only_continuation_committed;
  result.audit.virtual_no_sell_slot = runtime.audit.virtual_no_sell_slot;
  result.audit.runtime_planner_us = runtime.audit.timing.planner_us;
  result.audit.runtime_total_us = runtime.audit.timing.total_us;
  for (const auto& scenario : runtime.audit.planner_scenarios.scenarios) {
    result.audit.planner_lower_band |=
        scenario.belief_band == g001::dump::BeliefBand::Lower;
    result.audit.planner_point_band |=
        scenario.belief_band == g001::dump::BeliefBand::Point;
    result.audit.planner_upper_band |=
        scenario.belief_band == g001::dump::BeliefBand::Upper;
  }
  result.audit.planner_scenario_set_replayed =
      runtime.audit.planner_scenario_set_replayed;
  result.audit.expected_planner_scenarios =
      runtime.audit.expected_planner_scenarios;
  result.audit.actual_planner_scenarios = runtime.audit.actual_planner_scenarios;
  result.audit.selected_exact_terminal_liquidation =
      runtime.audit.exact_terminal_liquidation_selected;
  result.audit.selected_sell_compaction =
      runtime.audit.monotone_sell_compaction_selected;
  result.audit.selected_general_robust =
      runtime.audit.selected &&
      !result.audit.selected_exact_terminal_liquidation &&
      !result.audit.selected_sell_compaction;
  // The bridge product is authoritative only for the general single-option
  // path. Terminal liquidation and compaction can rewrite other/multiple
  // products, so derive those from the actual queue diff below.
  if (result.audit.selected_general_robust &&
      runtime.audit.bridge.option.product >= 0 &&
      runtime.audit.bridge.option.product < N_PRODUCTS)
    result.audit.selected_product = runtime.audit.bridge.option.product;
  result.audit.production_dag_certified = runtime.audit.production_certified;
  result.audit.phase_gate_called =
      runtime.audit.selected ||
      runtime.audit.fallback == selective_runtime::Fallback::PhaseGateRejected ||
      runtime.audit.fallback == selective_runtime::Fallback::FullModeInvariant;
  result.audit.protected_queue_called =
      runtime.audit.selected || runtime.audit.provisional_queue.search_states > 0 ||
      runtime.audit.final_queue.search_states > 0;
  copy_queue_audit(runtime.audit.final_queue.certificate_passed
                       ? runtime.audit.final_queue
                       : runtime.audit.provisional_queue,
                   result.audit);

  const bool full_observed =
      runtime.audit.full_mode_observed ||
      runtime.audit.phase_decision.mode == phase::Mode::FullTakeover ||
      runtime.audit.fallback == selective_runtime::Fallback::FullModeInvariant;
  result.audit.full_selected = full_observed;
  if (full_observed) {
    result.audit.uncertified = true;
    result.audit.exact_legacy_fallback = true;
    result.audit.reason = "FullTakeover is disabled at the native deployment boundary";
    state.continuation = continuation_before;
    state.selective_disabled = continuation_pending(continuation_before);
  } else if (runtime.audit.selected) {
    std::vector<Action> candidate;
    candidate.reserve(runtime.orders.size());
    bool converted = true;
    std::string conversion_reason;
    for (const auto& order : runtime.orders) {
      const auto native = native_selective_input::to_native_market_order(order);
      if (!native.accepted) {
        converted = false;
        conversion_reason = native.reason;
        break;
      }
      candidate.push_back(native.action);
    }
    if (converted) {
      result.market = std::move(candidate);
      result.audit.runtime_selected = true;
      result.audit.mode = static_cast<int>(phase::Mode::SelectiveSellOverlay);
      result.audit.reason = runtime.audit.reason;
      state.continuation = runtime.continuation;
    } else {
      result.audit.uncertified = true;
      result.audit.exact_legacy_fallback = true;
      result.audit.reason = "typed result conversion failed: " + conversion_reason;
      state.selective_disabled = continuation_pending(continuation_before);
    }
  } else {
    result.audit.exact_legacy_fallback = true;
    result.audit.no_candidate =
        runtime.audit.fallback == selective_runtime::Fallback::PlannerNoStrictImprovement ||
        runtime.audit.fallback == selective_runtime::Fallback::NoOptionalIntent;
    result.audit.uncertified = !result.audit.no_candidate;
    result.audit.reason = runtime.audit.reason;
    // Only runtime may settle/pause/replace continuation state.
    state.continuation = runtime.continuation;
  }

  result.audit.market_changed_slots = changed_slots(legacy_action.market, result.market);
  stage_actual();
  result.audit.market_changed_slots = changed_slots(legacy_action.market, result.market);
  if (state.evaluation_profile.capture_step_diagnostics) {
    result.audit.diagnostic_candidate_market = result.market;
    const std::size_t slots = std::max(legacy_action.market.size(), result.market.size());
    for (std::size_t slot = 0; slot < slots; ++slot) {
      const Action* before = slot < legacy_action.market.size()
          ? &legacy_action.market[slot] : nullptr;
      const Action* after = slot < result.market.size() ? &result.market[slot] : nullptr;
      if (before && after && same_action(*before, *after)) continue;
      for (const Action* order : {before, after}) {
        if (!order || order->op != Op::SELL) continue;
        const int product = static_cast<int>(order->item);
        if (product >= 0 && product < N_PRODUCTS)
          result.audit.changed_product_mask |=
              static_cast<std::uint16_t>(1U << product);
      }
    }
    if (!result.audit.selected_general_robust &&
        result.audit.changed_product_mask != 0 &&
        (result.audit.changed_product_mask &
         static_cast<std::uint16_t>(result.audit.changed_product_mask - 1)) == 0) {
      for (int product = 0; product < N_PRODUCTS; ++product)
        if (result.audit.changed_product_mask &
            static_cast<std::uint16_t>(1U << product)) {
          result.audit.selected_product = product;
          break;
        }
    }
  }
  result.audit.continuation_pending_after = continuation_pending(state.continuation);
  state.v2_audit_completed = state.v2_audit_completed || result.audit.runtime_called;
  if (state.mode_since_step < 0 || state.last_mode != result.audit.mode) {
    state.last_mode = result.audit.mode;
    state.mode_since_step = env.step_count();
    ++state.generation;
  }
  result.audit.native_overlay_total_us = elapsed_us(total_begin);
  return result;
}

void accumulate_native_phased_audit(NativePhasedMarketAudit& total,
                                    const NativePhasedStepAudit& step) {
  total.arm = step.arm;
  ++total.calls;
  if (step.mode == static_cast<int>(phase::Mode::ProtectedProduction))
    ++total.protected_steps;
  else if (step.mode == static_cast<int>(phase::Mode::SelectiveSellOverlay))
    ++total.selective_steps;
  else if (step.mode == static_cast<int>(phase::Mode::FullTakeover))
    ++total.full_steps;
  total.protected_queue_calls += step.protected_queue_called;
  total.phase_gate_calls += step.phase_gate_called;
  total.no_candidate_steps += step.no_candidate;
  total.uncertified_steps += step.uncertified;
  total.exact_legacy_fallbacks += step.exact_legacy_fallback;
  total.production_dag_certified_steps += step.production_dag_certified;
  total.v2_exact_legacy_fallbacks += step.v2_exact_legacy_fallback;
  total.production_obligation_nodes += step.production_obligation_nodes;
  total.due_legacy_bindings += step.due_legacy_bindings;
  total.protected_non_sell_orders += step.protected_non_sell_orders;
  total.required_funding_sell_orders += step.required_funding_sell_orders;
  total.required_funding_sell_units += step.required_funding_sell_units;
  total.optional_replacement_orders += step.optional_replacement_orders;
  total.optional_replacement_units += step.optional_replacement_units;
  total.search_states += step.search_states;
  total.observation_resets += step.observation_reset;
  total.observation_updates += step.observation_observed;
  total.observation_stages += step.observation_staged;
  total.observation_failures += step.observation_failure;
  total.builder_calls += step.builder_called;
  total.builder_rejections += step.builder_called && !step.builder_accepted;
  total.runtime_calls += step.runtime_called;
  total.runtime_selected_steps += step.runtime_selected;
  total.sell_compaction_selected_steps += step.selected_sell_compaction;
  total.exact_terminal_liquidation_selected_steps +=
      step.selected_exact_terminal_liquidation;
  total.general_robust_selected_steps += step.selected_general_robust;
  total.runtime_fallback_steps +=
      step.runtime_called &&
      step.runtime_fallback != static_cast<int>(selective_runtime::Fallback::None);
  total.market_changed_steps += step.market_changed_slots > 0;
  total.market_changed_slots += step.market_changed_slots;
  total.maximum_opponent_total_upper_units = std::max(
      total.maximum_opponent_total_upper_units,
      step.opponent_total_upper_units);
  total.maximum_scenario_dump_units = std::max(
      total.maximum_scenario_dump_units,
      step.maximum_scenario_dump_units);
  total.runtime_total_us += step.runtime_total_us;
  total.runtime_planner_us += step.runtime_planner_us;
  total.observation_us += step.observation_us;
  total.builder_us += step.builder_us;
  total.native_overlay_total_us += step.native_overlay_total_us;
  if (step.runtime_fallback >= 0 && step.runtime_fallback < kNativeRuntimeFallbackKinds) {
    ++total.runtime_fallback_counts[step.runtime_fallback];
    const bool zero_target =
        step.intent_mode == static_cast<int>(
            economic_intent_bridge::IntentMode::ReplaceSelectedProductTotal) &&
        step.desired_total == 0;
    if (zero_target)
      ++total.zero_target_fallback_counts[step.runtime_fallback];
    if (step.virtual_no_sell_slot)
      ++total.virtual_no_sell_fallback_counts[step.runtime_fallback];
  }
  if (step.bridge_reason >= 0 && step.bridge_reason < kNativeBridgeReasonKinds) {
    ++total.bridge_reason_counts[step.bridge_reason];
    if (step.runtime_fallback >= 0 &&
        step.runtime_fallback < kNativeRuntimeFallbackKinds)
      ++total.bridge_reason_fallback_counts[step.bridge_reason]
                                            [step.runtime_fallback];
    if (step.virtual_no_sell_slot)
      ++total.virtual_no_sell_bridge_reason_counts[step.bridge_reason];
  }
  if (step.intent_mode >= 0 && step.intent_mode < kNativeIntentModeKinds)
    ++total.intent_mode_counts[step.intent_mode];
  total.zero_target_steps +=
      step.intent_mode == static_cast<int>(
          economic_intent_bridge::IntentMode::ReplaceSelectedProductTotal) &&
      step.desired_total == 0;
  total.state_only_continuation_steps +=
      step.state_only_continuation_committed;
  total.virtual_no_sell_steps += step.virtual_no_sell_slot;
  if (step.virtual_no_sell_slot &&
      step.runtime_fallback != static_cast<int>(selective_runtime::Fallback::None) &&
      total.first_virtual_no_sell_rejection_step < 0) {
    total.first_virtual_no_sell_rejection_step = step.step;
    total.first_virtual_no_sell_rejection_reason = step.reason;
  }
  if (step.bridge_reason == static_cast<int>(
          economic_intent_bridge::Reason::SellIntent) &&
      step.runtime_fallback != static_cast<int>(selective_runtime::Fallback::None) &&
      total.first_sell_intent_rejection_step < 0) {
    total.first_sell_intent_rejection_step = step.step;
    total.first_sell_intent_rejection_reason = step.reason;
  }
  total.full_ever_selected = total.full_ever_selected || step.full_selected;
  total.last_mode = step.mode;
  total.last_runtime_fallback = step.runtime_fallback;
  total.last_runtime_fallback_name = step.runtime_fallback_name;
  if (!step.observation_reason.empty())
    total.last_observation_reason = step.observation_reason;
  total.last_reason = step.reason;
}

const char* native_market_arm_name(NativeMarketArm arm) {
  switch (arm) {
    case NativeMarketArm::LegacyDefault: return "legacy-default";
    case NativeMarketArm::SelectiveProtected: return "selective-protected";
    case NativeMarketArm::Phased: return "phased";
  }
  return "invalid";
}

const char* native_phased_mode_name(int mode) {
  return phase::mode_name(static_cast<phase::Mode>(mode));
}

}  // namespace fastkag
