#include "economic_intent_bridge.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace economic_intent_bridge {
namespace {

using g001::general_econ::ExecutionProposal;
using g001::general_econ::ExecutionReason;
using g001::general_econ::ExecutionState;
using g001::market::Product;
using g001::option::Kind;
using g001::rolling::PersistentOption;

constexpr int kBridgePriority = 100;

bool same_option(const PersistentOption& left, const PersistentOption& right) {
  return std::tie(left.kind, left.product, left.quota, left.window_steps,
                  left.reservation_price, left.target_inventory,
                  left.impact_limit) ==
         std::tie(right.kind, right.product, right.quota, right.window_steps,
                  right.reservation_price, right.target_inventory,
                  right.impact_limit);
}

int ceil_div(int numerator, int denominator) {
  return denominator <= 0 ? 0 : (numerator + denominator - 1) / denominator;
}

std::pair<int, int> outstanding_bounds(const ExecutionState& state) {
  if (state.outstanding_lower == 0 && state.outstanding_upper == 0 &&
      state.outstanding_units > 0) {
    return {state.outstanding_units, state.outstanding_units};
  }
  return {state.outstanding_lower, state.outstanding_upper};
}

Provenance provenance(const PersistentOption& option) {
  return {option.kind, option.product, option.quota, option.window_steps,
          option.reservation_price, option.target_inventory,
          option.impact_limit};
}

Continuation initialize(const PersistentOption& option, const Input& input,
                        int sellable) {
  Continuation result;
  result.active = true;
  result.option = option;
  const int windows = std::max(1, input.remaining_sale_windows);
  int units = 0;
  switch (option.kind) {
    case Kind::Drip:
    case Kind::PreDump:
      units = option.quota > 0 ? option.quota : sellable;
      break;
    case Kind::PriceTarget:
    case Kind::Clear:
      units = sellable;
      break;
    case Kind::InventoryTarget:
      units = std::max(0, sellable -
          std::max(0, option.target_inventory -
              input.protected_stock[static_cast<std::size_t>(option.product)]));
      break;
    case Kind::Baseline:
    case Kind::Hold:
      break;
  }
  result.execution.outstanding_units = units;
  result.execution.outstanding_lower = units;
  result.execution.outstanding_upper = units;
  result.execution.remaining_windows = windows;
  return result;
}

struct MarginalResult {
  int quantity{};
  int final_inventory{};
  int last_accepted{-1};
  int first_rejected{-1};
  bool reservation_blocked{};
  bool impact_blocked{};
  std::vector<int> quotes;
};

MarginalResult apply_marginal_band(Product product, int market_inventory,
                                   int limit, int reservation_price,
                                   int impact_limit) {
  MarginalResult result;
  result.final_inventory = market_inventory;
  const int initial_quote = g001::market::price(product, market_inventory);
  while (result.quantity < limit) {
    const int quote = g001::market::price(product, result.final_inventory);
    if (reservation_price > 0 && quote < reservation_price) {
      result.first_rejected = quote;
      result.reservation_blocked = true;
      break;
    }
    if (initial_quote - quote > impact_limit) {
      result.first_rejected = quote;
      result.impact_blocked = true;
      break;
    }
    result.quotes.push_back(quote);
    result.last_accepted = quote;
    ++result.quantity;
    // Official parity: a $1 sale consumes own stock but does not add public
    // market inventory.
    if (quote > 1) ++result.final_inventory;
  }
  return result;
}

void validate(const PersistentOption& option, const Input& input) {
  if (input.remaining_sale_windows < 0)
    throw std::invalid_argument("negative remaining sale windows");
  for (std::size_t product = 0; product < g001::market::product_count; ++product) {
    if (input.own_stock[product] < 0 || input.protected_stock[product] < 0 ||
        input.public_market_inventory[product] < 0) {
      throw std::invalid_argument("negative observable inventory");
    }
  }
  if (option.kind != Kind::Baseline &&
      (option.product < 0 ||
       option.product >= static_cast<int>(g001::market::product_count))) {
    throw std::invalid_argument("option product outside market range");
  }
  if (option.quota < 0 || option.window_steps < 0 ||
      option.reservation_price < 0 || option.target_inventory < 0 ||
      option.impact_limit < 0) {
    throw std::invalid_argument("negative persistent option field");
  }
}

}  // namespace

const char* reason_name(Reason reason) {
  switch (reason) {
    case Reason::Baseline: return "baseline";
    case Reason::Hold: return "hold";
    case Reason::NoSaleWindow: return "no-sale-window";
    case Reason::AwaitingConfirmation: return "awaiting-confirmation";
    case Reason::AwaitingExplicitSwitch: return "awaiting-explicit-switch";
    case Reason::NoSellableStock: return "no-sellable-stock";
    case Reason::Completed: return "completed";
    case Reason::ReservationPause: return "reservation-pause";
    case Reason::ImpactPause: return "impact-pause";
    case Reason::SellIntent: return "sell-intent";
  }
  return "unknown";
}

const char* intent_mode_name(IntentMode mode) {
  switch (mode) {
    case IntentMode::NoChange: return "no-change";
    case IntentMode::ReplaceSelectedProductTotal:
      return "replace-selected-product-total";
    case IntentMode::StateOnlyContinuation: return "state-only-continuation";
  }
  return "unknown";
}

Result make_sell_intent(const PersistentOption& selected, const Input& input) {
  validate(selected, input);
  Result result;
  result.audit.option = provenance(selected);
  if (selected.kind != Kind::Baseline)
    result.selected_product = static_cast<Product>(selected.product);

  Continuation prior;
  if (input.continuation) prior = *input.continuation;
  if (prior.pending) {
    if (!input.confirmed_prior_fill) {
      result.continuation = prior;
      result.audit.reason = Reason::AwaitingConfirmation;
      result.audit.explanation =
          "a submitted SELL remains pending; no second intent is issued before observation";
      return result;
    }
    prior.execution = g001::general_econ::settle_interval(
        prior.execution, *prior.pending, *input.confirmed_prior_fill);
    prior.pending.reset();
    result.audit.confirmation_applied = true;
    result.audit.confirmed_fill = *input.confirmed_prior_fill;
  } else if (input.confirmed_prior_fill) {
    throw std::invalid_argument("confirmed fill supplied without a pending bridge SELL");
  }

  const std::size_t product_index = selected.kind == Kind::Baseline
      ? 0U : static_cast<std::size_t>(selected.product);
  const int own_stock = input.own_stock[product_index];
  const int protected_stock = input.protected_stock[product_index];
  const int sellable = std::max(0, own_stock - protected_stock);
  result.audit.own_stock = own_stock;
  result.audit.protected_stock = protected_stock;
  result.audit.sellable_stock = sellable;

  if (prior.active && same_option(prior.option, selected)) {
    result.continuation = prior;
    result.audit.continued = true;
  } else {
    if (prior.active) {
      if (input.switch_directive ==
          g001::general_econ::SwitchDirective::RequireSame) {
        result.continuation = prior;
        result.audit.reason = Reason::AwaitingExplicitSwitch;
        result.audit.explanation =
            "selected option differs; explicit PausePrevious or CancelPrevious required";
        return result;
      }
      result.audit.switched = true;
      if (input.switch_directive ==
          g001::general_econ::SwitchDirective::PausePrevious) {
        prior.paused.push_back({prior.option, prior.execution});
        result.audit.paused_previous = true;
      } else {
        result.audit.cancelled_outstanding =
            outstanding_bounds(prior.execution).second;
      }
    }
    auto saved = std::find_if(
        prior.paused.begin(), prior.paused.end(), [&](const SavedExecution& candidate) {
          return same_option(candidate.option, selected);
        });
    if (saved != prior.paused.end()) {
      result.continuation.active = true;
      result.continuation.option = saved->option;
      result.continuation.execution = saved->execution;
      result.continuation.paused = prior.paused;
      const auto index = static_cast<std::size_t>(saved - prior.paused.begin());
      result.continuation.paused.erase(result.continuation.paused.begin() +
                                       static_cast<std::ptrdiff_t>(index));
      result.audit.resumed = true;
    } else {
      result.continuation = initialize(selected, input, sellable);
      result.continuation.paused = std::move(prior.paused);
    }
  }

  if (selected.kind == Kind::Baseline) {
    result.continuation.active = false;
    result.mode = IntentMode::NoChange;
    result.audit.reason = Reason::Baseline;
    result.audit.explanation = "Baseline belongs to the protected legacy layer";
    return result;
  }
  if (!input.sale_window) {
    result.mode = IntentMode::StateOnlyContinuation;
    result.audit.reason = Reason::NoSaleWindow;
    result.audit.explanation =
        "the current observation is not a sale window; retain option state without replacing legacy";
    return result;
  }
  if (selected.kind == Kind::Hold) {
    result.mode = IntentMode::ReplaceSelectedProductTotal;
    result.desired_total = 0;
    result.audit.reason = Reason::Hold;
    result.audit.explanation =
        "Hold explicitly replaces the selected product SELL total with zero";
    return result;
  }

  auto& state = result.continuation.execution;
  const auto [outstanding_lower, outstanding_upper] = outstanding_bounds(state);
  (void)outstanding_upper;
  result.audit.debt_before = state.arrears;
  result.audit.outstanding_before = state.outstanding_units;
  result.audit.remaining_windows_before = state.remaining_windows;

  int requested = 0;
  int reservation = 0;
  switch (selected.kind) {
    case Kind::Drip: {
      // remaining outstanding contains both nominal future slices and debt.
      // Removing debt before dividing reproduces settle_drip's earlier catch-up.
      const int nominal = std::max(0, state.outstanding_units - state.arrears);
      result.audit.base_due = ceil_div(nominal, state.remaining_windows);
      requested = std::min(state.outstanding_units,
                           result.audit.base_due + state.arrears);
      // With uncertain attribution, emit only units guaranteed outstanding.
      requested = std::min(requested, outstanding_lower);
      reservation = selected.reservation_price;
      break;
    }
    case Kind::PriceTarget:
      requested = sellable;
      reservation = selected.reservation_price;
      break;
    case Kind::InventoryTarget:
      requested = std::max(
          0, own_stock - std::max(protected_stock, selected.target_inventory));
      break;
    case Kind::PreDump:
      requested = std::min(state.outstanding_units, outstanding_lower);
      break;
    case Kind::Clear:
      requested = sellable;
      break;
    case Kind::Baseline:
    case Kind::Hold:
      break;
  }
  result.audit.requested_by_option = requested;
  requested = std::min(requested, sellable);
  result.audit.requested_after_stock_cap = requested;

  const auto product = static_cast<Product>(selected.product);
  const int market_inventory = input.public_market_inventory[product_index];
  result.audit.market_inventory_before = market_inventory;
  result.audit.initial_quote = g001::market::price(product, market_inventory);
  const auto marginals = apply_marginal_band(
      product, market_inventory, requested, reservation, selected.impact_limit);
  result.audit.emitted_quantity = marginals.quantity;
  result.audit.last_accepted_quote = marginals.last_accepted;
  result.audit.first_rejected_quote = marginals.first_rejected;
  result.audit.market_inventory_if_emitted = marginals.final_inventory;
  result.audit.accepted_marginal_quotes = marginals.quotes;

  if (marginals.quantity <= 0) {
    result.mode = IntentMode::ReplaceSelectedProductTotal;
    result.desired_total = 0;
    if (requested <= 0) {
      result.audit.reason = sellable <= 0 ? Reason::NoSellableStock : Reason::Completed;
      result.audit.explanation = sellable <= 0
          ? "own stock minus protected stock is zero"
          : "the selected option currently requires no sale";
    } else if (marginals.reservation_blocked) {
      result.audit.reason = Reason::ReservationPause;
      result.audit.explanation = "the first marginal quote is below reservation";
    } else {
      result.audit.reason = Reason::ImpactPause;
      result.audit.explanation = "the first marginal quote exceeds the impact band";
    }

    // A zero request/fill is exact and needs no next-observation attribution.
    // For DRIP/PreDump it still consumes the causal sale window and carries
    // scheduled debt forward.
    if ((selected.kind == Kind::Drip || selected.kind == Kind::PreDump) &&
        state.outstanding_units > 0) {
      ExecutionProposal zero;
      zero.requested = 0;
      zero.scheduled_due = selected.kind == Kind::Drip
          ? std::min(state.outstanding_units,
                     result.audit.base_due + state.arrears)
          : std::min(state.outstanding_units, outstanding_lower);
      zero.quote = result.audit.initial_quote;
      zero.market_inventory_if_filled = market_inventory;
      zero.reason = ExecutionReason::FeedbackPause;
      state = g001::general_econ::settle(state, zero, 0);
      result.audit.zero_fill_settled = true;
    }
    return result;
  }

  protected_queue::OptionalSell intent;
  intent.product = product;
  intent.maximum_quantity = marginals.quantity;
  intent.priority = kBridgePriority;
  // Preserve the planner's marginal economics through queue composition.
  // protected_queue replays every public pressure scenario and may reduce the
  // quantity further, but it must never execute a unit below these bounds.
  intent.reservation_price = reservation;
  intent.impact_limit = selected.impact_limit;
  result.optional_sell = intent;
  result.mode = IntentMode::ReplaceSelectedProductTotal;
  result.desired_total = marginals.quantity;

  ExecutionProposal proposal;
  proposal.requested = marginals.quantity;
  proposal.scheduled_due = selected.kind == Kind::Drip
      ? std::min(state.outstanding_units,
                 result.audit.base_due + state.arrears)
      : marginals.quantity;
  proposal.quote = result.audit.initial_quote;
  proposal.market_inventory_if_filled = marginals.final_inventory;
  proposal.reason = selected.kind == Kind::Clear
      ? ExecutionReason::LastWindow : ExecutionReason::ScheduledDrip;
  result.continuation.pending = proposal;
  result.audit.reason = Reason::SellIntent;
  result.audit.explanation =
      "bounded SELL intent; protected_queue must still certify queue feasibility";
  return result;
}

Result make_sell_intent(const g001::general::Result& selected,
                        const Input& input) {
  return make_sell_intent(selected.plan.first, input);
}

}  // namespace economic_intent_bridge
