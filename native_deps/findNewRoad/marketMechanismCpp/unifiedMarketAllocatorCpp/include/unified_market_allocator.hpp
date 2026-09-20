#pragma once

#include "market.hpp"
#include "production_obligation.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace unified_market {

enum class SellPurpose : std::uint8_t {
  Liquidity,
  Capacity,
  Terminal,
  Feedback,
};

enum class FailureCause : std::uint8_t {
  None,
  Cash,
  Capacity,
  ProtectedStock,
  SellSlot,
  HardOrderSlotBudget,
  InvalidOrder,
};

struct SellIntent {
  g001::market::Product product{g001::market::Product::Wheat};
  int maximum_quantity{};
  int protected_quantity{};
  int reservation_price{1};
  int priority{};
  SellPurpose purpose{SellPurpose::Feedback};
};

// Deliberately has no legacy market tape, opponent identity, route identity,
// fixed opponent timing, private state, last_fills, or future realized field.
struct Input {
  int step{};
  int maximum_slots{10};
  int shed_capacity{100};
  g001::market::Inventory market_inventory{};
  g001::market::PlayerMarketState own{};
  std::vector<production_obligation::ObligationNode> production_dag;
  std::vector<SellIntent> sell_intents;
};

struct Audit {
  int due_purchase_orders{};
  int funding_sell_orders{};
  int funding_sell_units{};
  int unmet_purchase_orders{};
  int slots_used{};
  bool exact_queue_verified{};
  bool has_first_unfilled_hard_order{};
  production_obligation::NodeKind first_unfilled_kind{
      production_obligation::NodeKind::Consume};
  production_obligation::Item first_unfilled_item{
      production_obligation::Item::None};
  int first_unfilled_quantity{};
  int first_unfilled_committed{};
  int first_unfilled_remaining{};
  std::int64_t cash_shortfall{};
  int capacity_shortfall{};
  std::array<int, g001::market::product_count> sellable_unprotected_by_product{};
  int sellable_unprotected_inventory{};
  int protected_blocked_inventory{};
  int remaining_sell_slots{};
  FailureCause failure_cause{FailureCause::None};
  std::string reason;
};

struct Result {
  bool feasible{};
  std::vector<g001::market::Order> orders;
  g001::market::PlayerMarketState predicted_own{};
  g001::market::Inventory predicted_market{};
  Audit audit;
};

[[nodiscard]] Result allocate(const Input& input);
[[nodiscard]] const char* failure_cause_name(FailureCause cause);

}  // namespace unified_market
