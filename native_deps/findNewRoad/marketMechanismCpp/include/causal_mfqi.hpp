#pragma once

#include "belief.hpp"
#include "fqi_teacher.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace g001::causal_fqi {

// These offsets are part of the generated feature contract.  In particular,
// continuation state is observable by the next FQI row instead of living only
// in label metadata.
enum Feature : std::size_t {
    Step = 0,
    Day,
    Hour,
    OwnMoney,
    OpponentMoney,
    CapacityUsed,
    CapacityFree,
    ActiveKind,
    RemainingQuota,
    RemainingWindows,
    Debt,
    HorizonRemaining,
    CausalOverflow,
    DumpEarliestDistance,
    DumpLikelyDistance,
    MarketInventoryBegin,
    MarketPriceBegin = MarketInventoryBegin + market::product_count,
    OwnStockBegin = MarketPriceBegin + market::product_count,
    OwnFarmReadyBegin = OwnStockBegin + market::product_count,
    OpponentFarmReadyBegin = OwnFarmReadyBegin + market::product_count,
    BeliefPoint = OpponentFarmReadyBegin + market::product_count,
    BeliefLower,
    BeliefUpper,
    RecentClearance,
    FeatureCount
};

struct RouteCapacityForecast {
    int current_shed_items{};
    int current_carried_items{};
    int visible_ready_harvest{};
    int planned_product_buys{};
    int planned_animal_buys{};
    int shed_capacity{};
};

struct CapacityDecision {
    int predicted_peak{};
    int overflow{};
    int target_inventory{};
    bool valid{};
};

// This function intentionally has no realized-future/opponent-private input.
[[nodiscard]] CapacityDecision causal_inventory_target(
    const RouteCapacityForecast& forecast,
    int selected_product_stock
);

struct DumpWindow {
    bool valid{};
    int earliest_step{-1};
    int likely_step{-1};
    int observed_clearance{};
};

// history_end is inclusive. Only history[0..history_end] may affect output.
[[nodiscard]] DumpWindow public_dump_window(
    const std::vector<market::InventoryBelief>& history,
    std::size_t history_end,
    std::size_t product,
    int current_step,
    int episode_steps,
    int lookback_steps = 48
);

[[nodiscard]] int predump_quantity(int stock, int remaining_safe_windows);
[[nodiscard]] int clear_quantity(int stock);

// Public market conservation. Unit-side FEED/DROP/HARVEST never enters this
// formula and therefore cannot be mistaken for an opponent sale.
[[nodiscard]] int causal_rival_net_market_flow(
    int market_before, int market_after, int town_drain,
    int own_sell_fill, int own_buy_fill
);

// Validates ABI-independent graph invariants before serialization.  All
// nonterminal edges must be forward, closed, same episode and same product.
void validate_graph(const std::vector<fqi::Transition>& rows,
                    std::size_t option_count);

void write_mfqi(const std::string& path,
                const std::vector<fqi::Transition>& rows,
                std::size_t feature_count,
                std::size_t option_count,
                std::uint64_t option_schema_hash);

}  // namespace g001::causal_fqi
