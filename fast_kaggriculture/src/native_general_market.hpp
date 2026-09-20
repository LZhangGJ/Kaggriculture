// Licensed under the Apache License, Version 2.0.
#pragma once

#include "production_obligation.hpp"
#include "simulator.hpp"

#include <span>
#include <string>
#include <vector>

namespace fastkag {

// Contains unit actions only.  There is intentionally no market vector,
// route/opponent identity, clone feature, or realized-future field.
struct NativeFutureUnitFrame {
  int step = 0;
  std::vector<Action> units;
};

struct NativeGeneralMarketAudit {
  int obligation_nodes = 0;
  int purchase_orders_due = 0;
  int funding_sell_orders = 0;
  int funding_sell_units = 0;
  int allocated_orders = 0;
  bool compiler_feasible = false;
  bool allocator_feasible = false;
  std::string diagnostic_code;
  int diagnostic_step = -1;
  int diagnostic_actor = -1;
  int diagnostic_item = -1;
  int diagnostic_quantity = 0;
  int soft_misses = 0;
  int soft_pickup_replacements = 0;
  int soft_pass_replacements = 0;
  int soft_downstream_loss_proxy = 0;
  int production_protection_soft_misses = 0;
  std::string first_due_kind;
  int first_due_item = -1;
  int first_due_quantity = 0;
  int first_due_cash_quote = 0;
  int first_due_free_capacity = 0;
  std::string reason;
};

struct NativeCurrentUnitReplacement {
  int actor = -1;
  Action original;
  Action replacement;
  int soft_miss_node = -1;
  int downstream_loss_proxy = 0;
};

struct NativeGeneralMarketResult {
  bool feasible = false;
  std::vector<Action> market;
  std::vector<NativeCurrentUnitReplacement> current_unit_replacements;
  NativeGeneralMarketAudit audit;
};

// Shared causal unit-plan boundary for both the experimental full allocator
// and the phased protected-queue seam.  Keeping this conversion here ensures
// there is only one native interpretation of official state and unit actions.
struct NativeProductionObligationResult {
  bool feasible = false;
  std::vector<production_obligation::ObligationNode> nodes;
  std::vector<production_obligation::Edge> edges;
  std::vector<NativeCurrentUnitReplacement> current_unit_replacements;
  NativeGeneralMarketAudit audit;
};

[[nodiscard]] NativeProductionObligationResult
compile_native_production_obligations(
    const Simulator& env,
    int player,
    const std::vector<Action>& current_units,
    const std::vector<NativeFutureUnitFrame>& future_units);

// Recompiles a bounded 1..24-step causal unit plan from the current own
// observation, then allocates only hard production orders.  Optional economic
// sells are outside this MVP.
[[nodiscard]] NativeGeneralMarketResult compile_native_general_market(
    const Simulator& env,
    int player,
    const std::vector<Action>& current_units,
    const std::vector<NativeFutureUnitFrame>& future_units);

// Adds current-step hard buys derived from state targets which cannot yet
// appear in the feasible unit plan. Duplicate operation/item demands are
// merged and any quantity already derived from future units is not bought
// twice.
[[nodiscard]] NativeGeneralMarketResult compile_native_general_market(
    const Simulator& env,
    int player,
    const std::vector<Action>& current_units,
    const std::vector<NativeFutureUnitFrame>& future_units,
    std::span<const Action> hard_current_acquisitions);

}  // namespace fastkag
