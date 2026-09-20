#pragma once

#include "market.hpp"
#include "production_obligation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace protected_queue {

struct OptionalSell {
  g001::market::Product product{g001::market::Product::Wheat};
  int maximum_quantity{};
  int priority{};
  int reservation_price{1};
  int impact_limit{std::numeric_limits<int>::max()};
};

struct Input {
  int maximum_slots{10};
  // Optional SELLs may only occupy this slot or a later one. This leaves an
  // explicit serial prefix for exact earlier pressure. RequiredFundingSell and
  // byte-preserved legacy orders keep their original slots.
  int minimum_optional_slot{0};
  int shed_capacity{100};
  int hire_cost_multiplier{1};
  g001::market::Inventory market_inventory{};
  g001::market::PlayerMarketState own{};
  // Future feed/PICKUP/production reserve. Neither retained legacy SELL nor
  // optional SELL may consume these units.
  g001::market::Inventory protected_stock{};

  // The early-phase exception: legacy orders are visible only to protect the
  // production programme. Every non-SELL remains byte-equivalent and in its
  // original slot. Removed SELL slots stay as PASS until optionally replaced.
  std::vector<g001::market::Order> legacy_orders;
  std::vector<OptionalSell> optional_sells;
  // Default false preserves the original all-SELL experimental behavior.
  // The product-local general planner sets this true and marks only its
  // selected product, so unrelated G001 SELLs remain byte-identical.
  bool restrict_legacy_sell_replacement{};
  std::array<bool, g001::market::product_count> replaceable_legacy_sell{};

  // Each queue is a causal public stress assumption, not observed/private
  // opponent stock. Only SELL is accepted. Slots are replayed exactly: same
  // slot uses official pre-commit lockstep quotes; a true opponent-before-own
  // stress places the rival SELL in a slot earlier than the optional SELL.
  // The implementation derives the minimum rival shed needed to execute it.
  std::vector<std::vector<g001::market::Order>> opponent_first_pressure;

  // Exact branch-and-bound guard. Exhaustion is fail-safe: return legacy.
  std::size_t maximum_search_states{500000};
};

enum class SlotOrigin : std::uint8_t {
  Pass,
  LegacyNonSell,
  PreservedLegacySell,
  RequiredFundingSell,
  OptionalSell,
  ExactLegacyFallback,
};

struct ScenarioAudit {
  // One entry per protected legacy non-SELL, in legacy slot order.
  std::vector<int> legacy_failure_vector;
  std::vector<int> candidate_failure_vector;
  int required_sell_unfilled{};
  int optional_sell_unfilled{};
  int optional_price_band_failures{};
  int required_funding_requested_units{};
  int required_funding_committed_units{};
  // Exact positive cash flow of slots whose provenance is
  // RequiredFundingSell. Optional/legacy-surplus proceeds are excluded.
  std::int64_t committed_required_funding_revenue{};
  std::int64_t legacy_final_money{};
  std::int64_t candidate_final_money{};
  int legacy_capacity_margin{};
  int candidate_capacity_margin{};
  std::int64_t marked_margin_vs_legacy{};
};

struct Audit {
  int protected_non_sell_orders{};
  int preserved_legacy_sell_orders{};
  int preserved_legacy_sell_units{};
  int required_funding_sell_orders{};
  int required_funding_sell_units{};
  int optional_replacement_orders{};
  int optional_replacement_units{};
  std::size_t search_states{};
  bool minimum_proven{};
  bool certificate_passed{};
  bool used_legacy_fallback{};
  std::int64_t worst_marked_margin{};
  std::vector<ScenarioAudit> scenarios;
  std::string reason;
};

struct Result {
  std::vector<g001::market::Order> orders;
  std::vector<SlotOrigin> origins;
  Audit audit;
};

// Scenario 0 is always the no-pressure case. Supplied pressure queues follow
// with official slot/pre-commit ordering; player index does not turn a same-slot
// lockstep quote into a serial opponent-first quote.
[[nodiscard]] Result compose(const Input& input);

struct DueLegacyBinding {
  int obligation_node_id{-1};
  int legacy_slot{-1};
  int required_quantity{};
};

struct InputV2 {
  static constexpr int supported_abi_version = 2;
  int abi_version{supported_abi_version};
  int step{};
  Input queue;
  std::vector<production_obligation::ObligationNode> production_dag;
};

struct ProtectedSlotV2 {
  g001::market::Order order;
  SlotOrigin origin{SlotOrigin::Pass};
  int obligation_node_id{-1};
};

struct ResultV2 {
  std::vector<ProtectedSlotV2> slots;
  std::vector<DueLegacyBinding> due_legacy_bindings;
  Audit audit;
  bool exact_legacy_fallback{};
  std::string reason;
};

// Versioned native boundary. Every current-step hard acquisition is
// stable-bound to an adequate legacy non-SELL before any SELL may change.
// Ambiguity or missing coverage returns the exact legacy vector.
[[nodiscard]] ResultV2 compose_v2(const InputV2& input);

}  // namespace protected_queue
