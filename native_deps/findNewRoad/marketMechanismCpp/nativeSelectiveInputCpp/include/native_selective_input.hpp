#pragma once

#include "selective_runtime.hpp"
#include "simulator.hpp"

#include <optional>
#include <string>
#include <vector>

namespace native_selective_input {

// A caller-selected causal plan frame.  It is not fetched from a route table
// or replay by this adapter.
struct SelectedPlanFrame {
  int step{-1};
  fastkag::PlayerAction selected;
  bool causal_queue_known{};
  bool crosses_route_switch{};
};

struct BuildInput {
  const fastkag::Simulator* simulator{};
  int player{-1};
  // Final current action after all non-market overlays.  It is the exact
  // legacy queue protected by selective_runtime.
  fastkag::PlayerAction current_legacy;
  // Consecutive live frames beginning at current step + 1. Normally 23 form
  // the downstream 24-tick horizon and an optional 24th is switch lookahead.
  // Near the official terminal, only remaining live frames are supplied; the
  // adapter creates inert, non-actionable padding for the 24-tick ABI.
  std::vector<SelectedPlanFrame> future_plan;
  public_belief_runtime::Snapshot public_belief;
  std::optional<economic_intent_bridge::Continuation> continuation;
  std::optional<g001::general_econ::QuantityInterval> confirmed_prior_fill;
};

struct Audit {
  int current_step{-1};
  int provided_future_frames{};
  int compiled_horizon_ticks{};
  int validated_lookahead_frames{};
  int sale_windows{};
  int deterministic_town_drain_steps{};
  int converted_current_market_orders{};
  int converted_future_market_orders{};
  bool public_snapshot_exact{};
  bool typed_market_roundtrip_exact{};
  bool planner_threads_one{};
};

struct Result {
  bool accepted{};
  std::string reason;
  selective_runtime::Input input;
  Audit audit;
};

struct TypedOrderResult {
  bool accepted{};
  std::string reason;
  g001::market::Order order;
};

struct NativeOrderResult {
  bool accepted{};
  std::string reason;
  fastkag::Action action;
};

// Canonical native<->typed conversion. Accepted native actions round-trip
// exactly in op/item/quantity; unused item fields must therefore be NONE.
[[nodiscard]] TypedOrderResult to_typed_market_order(
    const fastkag::Action& action) noexcept;
[[nodiscard]] NativeOrderResult to_native_market_order(
    const g001::market::Order& order) noexcept;

// No exception crosses this native boundary. Invalid/missing evidence returns
// accepted=false with an exact legacy-safe reason.
[[nodiscard]] Result build(const BuildInput& source) noexcept;

}  // namespace native_selective_input
