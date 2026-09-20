#pragma once

#include "adaptive_execution.hpp"
#include "general_planner.hpp"
#include "protected_queue.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace economic_intent_bridge {

// The bridge owns no market queue.  A pending proposal is retained until the
// next observation attributes a fill interval to this option.
struct SavedExecution {
  g001::rolling::PersistentOption option{};
  g001::general_econ::ExecutionState execution{};
};

struct Continuation {
  bool active{};
  g001::rolling::PersistentOption option{};
  g001::general_econ::ExecutionState execution{};
  std::optional<g001::general_econ::ExecutionProposal> pending;
  std::vector<SavedExecution> paused;
};

struct Input {
  g001::market::Inventory own_stock{};
  g001::market::Inventory protected_stock{};
  g001::market::Inventory public_market_inventory{};
  bool sale_window{};
  // Counted from the focal player's causal calendar, including the current
  // window when sale_window is true. It is not inferred from realized future.
  int remaining_sale_windows{};
  // A changed selected option is rejected by default.  Pause/Cancel must be
  // chosen explicitly after any pending fill has been observed.
  g001::general_econ::SwitchDirective switch_directive{
      g001::general_econ::SwitchDirective::RequireSame};
  std::optional<Continuation> continuation;
  // Required exactly when continuation.pending exists.  The interval must be
  // produced from observed own-shed change and causal own non-SELL deltas.
  std::optional<g001::general_econ::QuantityInterval> confirmed_prior_fill;
};

enum class Reason : std::uint8_t {
  Baseline,
  Hold,
  NoSaleWindow,
  AwaitingConfirmation,
  AwaitingExplicitSwitch,
  NoSellableStock,
  Completed,
  ReservationPause,
  ImpactPause,
  SellIntent,
};

// A missing OptionalSell is not sufficient to describe the selected economic
// action: zero can mean either "leave legacy untouched" or "replace this
// product's legacy SELL total with zero". Keep that distinction typed so the
// queue/runtime boundary can certify zero-quantity suppression separately.
enum class IntentMode : std::uint8_t {
  NoChange,
  ReplaceSelectedProductTotal,
  StateOnlyContinuation,
};

struct Provenance {
  g001::option::Kind kind{g001::option::Kind::Baseline};
  int product{};
  int quota{};
  int window_steps{};
  int reservation_price{};
  int target_inventory{};
  int impact_limit{};
};

struct Audit {
  Provenance option{};
  Reason reason{Reason::Baseline};
  bool continued{};
  bool switched{};
  bool paused_previous{};
  bool resumed{};
  bool confirmation_applied{};
  bool zero_fill_settled{};
  int cancelled_outstanding{};
  g001::general_econ::QuantityInterval confirmed_fill{};

  int own_stock{};
  int protected_stock{};
  int sellable_stock{};
  int requested_by_option{};
  int requested_after_stock_cap{};
  int emitted_quantity{};
  int base_due{};
  int debt_before{};
  int outstanding_before{};
  int remaining_windows_before{};

  int initial_quote{};
  int last_accepted_quote{-1};
  int first_rejected_quote{-1};
  int market_inventory_before{};
  int market_inventory_if_emitted{};
  std::vector<int> accepted_marginal_quotes;
  std::string explanation;
};

struct Result {
  IntentMode mode{IntentMode::NoChange};
  // Baseline has no selected product. Every product-local option carries its
  // product even when desired_total is zero and optional_sell is absent.
  std::optional<g001::market::Product> selected_product;
  int desired_total{};
  // The only executable output.  No BUY or legacy order is represented by
  // this compatibility field; protected_queue remains the sole queue
  // compositor. It is populated only for a positive replacement total.
  std::optional<protected_queue::OptionalSell> optional_sell;
  Continuation continuation;
  Audit audit;
};

[[nodiscard]] Result make_sell_intent(
    const g001::rolling::PersistentOption& selected,
    const Input& input);

// Convenience boundary for the current general planner ABI.  The selected
// online option is general::Result.plan.first.
[[nodiscard]] Result make_sell_intent(
    const g001::general::Result& selected,
    const Input& input);

[[nodiscard]] const char* reason_name(Reason reason);
[[nodiscard]] const char* intent_mode_name(IntentMode mode);

}  // namespace economic_intent_bridge
