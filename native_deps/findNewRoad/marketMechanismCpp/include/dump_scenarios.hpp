#pragma once

#include "rolling_optimizer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace g001::dump {

using Inventory = g001::market::Inventory;

struct InventoryFlowInterval {
    Inventory lower{};
    Inventory point{};
    Inventory upper{};
};

struct OwnShedTransition {
    Inventory previous_shed{};
    Inventory current_shed{};
    Inventory sell_requested{};
    InventoryFlowInterval known_nonmarket_inflow{};  // harvest/DROP interval
    InventoryFlowInterval known_nonmarket_outflow{}; // feed/PICKUP/discard interval
};

struct DerivedOwnSellFill {
    Inventory lower{};
    Inventory point{};
    Inventory upper{};
    std::array<bool, g001::market::product_count> clamped{};
};

// Causal shed conservation only: fill = previous + nonmarket inflow -
// nonmarket outflow - current, intersected with [0, requested] and available
// own stock. This API deliberately has no simulator last_fills input.
[[nodiscard]] DerivedOwnSellFill derive_own_sell_fill_interval(
    const OwnShedTransition& transition
);

// One public observation. Flow fields describe the transition from the
// preceding frame to this frame. No opponent action or private inventory is
// accepted by this ABI.
struct PublicMarketFrame {
    int step{};
    int day{};
    Inventory market_inventory{};
    Inventory market_price{};
    Inventory own_sell_requested{};
    // Must be derived causally from consecutive own inventories plus known
    // own unit operations. Do not source these fields from a simulator's
    // post-transition/last_fills private API at runtime.
    Inventory own_sell_filled_lower{};
    Inventory own_sell_filled_point{};
    Inventory own_sell_filled_upper{};
    Inventory known_town_drain{};

    // Optional causal opponent-clearance interval produced by the public
    // inventory belief filter.  In particular, that filter may use public
    // money conservation to constrain SELL units that are invisible at the
    // official $1 floor.  The scenario generator intersects this stronger
    // evidence with market-flow evidence instead of treating a floor event as
    // wholly unknowable.
    Inventory opponent_clearance_lower{};
    Inventory opponent_clearance_point{};
    Inventory opponent_clearance_upper{};
    std::array<bool, g001::market::product_count> opponent_clearance_valid{};
};

struct ProductFlowEvidence {
    int opponent_net_flow_point{}; // sales minus possible BUY_PRODUCT
    int sale_lower{};
    int sale_point{};
    int sale_upper{};
    bool buy_ambiguity{};          // WHEAT/FERTILIZER can have negative flow
    // True when either shed conservation leaves the own fill uncertain or the
    // official $1 rule makes some filled units invisible in market.inventory.
    bool fill_ambiguity{};
    // In the official interpreter a SELL quoted at $1 pays the seller and
    // removes the item from their shed, but does not add it to public market
    // inventory.  A floor transition therefore hides gross player flow; it is
    // not merely a price-inversion ambiguity.
    bool price_floor{};
    bool price_inconsistent{};
    int price_change{};
    bool inconsistent{};
};

struct TransitionEvidence {
    int previous_step{};
    int current_step{};
    std::array<ProductFlowEvidence, g001::market::product_count> product{};
};

enum class BeliefBand : std::uint8_t { Lower, Point, Upper };
enum class DumpTiming : std::uint8_t { None, Now, Next, Late, Distributed };

struct WeightedScenario {
    g001::rolling::Scenario scenario;
    double weight{};
    BeliefBand belief_band{BeliefBand::Point};
    DumpTiming timing{DumpTiming::None};
    std::uint32_t id{};
};

struct Input {
    std::vector<PublicMarketFrame> public_history;
    Inventory belief_lower{};
    Inventory belief_point{};
    Inventory belief_upper{};
    int current_step{};
    int current_day{};
    int steps_remaining{};
    int horizon_steps{24};
    std::vector<int> remaining_sale_windows;
};

struct Result {
    struct IntentSignals {
        int recent_sale_point{};
        double pulse_frequency{};
        double recent_quantity_ratio{};
        double terminal_urgency{};
        double stock_pressure{};
        double dump_propensity{};
    } intent;
    std::vector<TransitionEvidence> flow_history;
    std::vector<WeightedScenario> scenarios;
};

// At most three distinct nonempty belief bands crossed with now/next/late and
// a horizon-distributed clearance, plus one explicit no-dump scenario.
inline constexpr std::size_t max_scenario_count = 13;

[[nodiscard]] Result generate(const Input& input);

}  // namespace g001::dump
