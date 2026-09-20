#pragma once

#include "market.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace g001::market {

using ProductFlags = std::array<bool, product_count>;

struct InventoryInterval {
    Inventory lower{};
    Inventory upper{};
};

// Public effects belonging to the transition from the previous summary to
// this one.  `point` is the visible effect; `upper` also contains effects that
// can be hidden by a same-transition refresh (harvest/feed/fertilize).
struct PublicFlow {
    Inventory harvest_point{};
    Inventory harvest_upper{};
    Inventory consumption_point{};
    Inventory consumption_upper{};
};

// A compact, legal-observation-only input.  `market_inventory` is the primary
// exact integer signal; rounded price is used only for floor detection and a
// consistency check.  The input deliberately has no opponent private stock or
// opponent action field (farm money is public in the official observation).
struct PublicStateSummary {
    int step = 0;
    int day = 0;
    Inventory market_inventory{};
    Inventory market_price{};
    Inventory own_total{};  // own shed plus all own carried inventories

    // Optional causal accounting of our successful market net flow for this
    // transition: non-floor SELL minus BUY_PRODUCT.  Exact bounds let the
    // filter distinguish an opponent's floor BUY/rebase from our own invisible
    // floor sale.  Deployment callers derive this from own actions and own
    // inventory conservation; no opponent action or simulator fill is needed.
    Inventory own_market_net_flow_lower{};
    Inventory own_market_net_flow_point{};
    Inventory own_market_net_flow_upper{};
    bool own_market_net_flow_bounds_valid = false;

    // Both balances are public in the official observation.  Money evidence
    // is optional because a caller may not yet have bounded private seed
    // purchases and non-floor trade proceeds soundly.
    std::int64_t own_money = 0;
    std::int64_t opponent_money = 0;
    bool money_evidence_valid = false;

    // Public farm fields used to deduct successful HIRE / BUY_LAND directly.
    int own_hands = 0;
    int opponent_hands = 0;
    int own_hires_today = 0;
    int opponent_hires_today = 0;
    int own_unlocked_quadrants = 1;
    int opponent_unlocked_quadrants = 1;
    int farm_hand_cost_multiplier = 1;

    // Additional signed cash change explained by other public non-market
    // events.  HIRE/LAND are derived from the fields above and need not be
    // included here.
    std::int64_t opponent_public_cash_delta = 0;
    std::int64_t opponent_private_purchase_cost_lower = 0;
    std::int64_t opponent_private_purchase_cost_upper = 0;
    // Legal bounds on cash from all non-floor trades in this transition.
    // These can come from exact action compilation / price-path enumeration.
    std::int64_t opponent_nonfloor_trade_cash_lower = 0;
    std::int64_t opponent_nonfloor_trade_cash_upper = 0;
    bool trade_cash_bounds_valid = false;

    PublicFlow own_flow{};
    PublicFlow opponent_flow{};
    Inventory town_consumption{};

    // Publicly attributable opponent harvest put into unit inventories and
    // public consumption taken from those units.  These drive the shed point
    // estimate without reading private state.
    Inventory opponent_carried_gain{};
    Inventory opponent_carried_use{};

    // True when the opponent could have PICKUP/DROP/PASSed at shed access.
    bool opponent_shed_access_ambiguity = false;
    bool day_rollover = false;

    // A floor sale does not enter market inventory.  A private discard also
    // changes loose stock without any market evidence.  Both destroy exact
    // identifiability for that product.
    ProductFlags floor_sale_ambiguity{};
    ProductFlags possible_private_discard{};
};

struct InitialOpponentStock {
    Inventory point{};
    Inventory lower{};
    Inventory upper{};
};

struct InventoryBelief {
    int step = 0;
    Inventory total{};
    InventoryInterval total_interval{};
    Inventory shed{};
    InventoryInterval shed_interval{};
    Inventory recent_clearance{};
    InventoryInterval recent_clearance_interval{};
    // Gross BUY_PRODUCT quantity at a saturated $1 market.  These marginals
    // are jointly correlated with recent_clearance through
    //     floor_sales - floor_purchases = public cash residual.
    // They intentionally remain intervals when buy/sell churn is not uniquely
    // identifiable from the public market and money deltas.
    Inventory recent_floor_purchase{};
    InventoryInterval recent_floor_purchase_interval{};
    ProductFlags likely_cleared{};
    ProductFlags saturated{};
    ProductFlags price_inconsistent{};
    // True when the public-money joint hidden-floor-sale interval was
    // projected onto this product.  Several products may be true on one step;
    // their marginal intervals are correlated and must not be multiplied as
    // independent beliefs.
    ProductFlags cash_constraint_applied{};
    ProductFlags floor_purchase_constraint_applied{};
    ProductFlags ambiguous{};
};

// Causal resource-conservation filter.  Consecutive summaries must be ordered
// by increasing step.  reset() may be given a non-zero prior when attaching to
// a trace after its initial state; the default prior is exact zero.
class OpponentInventoryBelief {
public:
    explicit OpponentInventoryBelief(int shed_capacity = 100);

    [[nodiscard]] InventoryBelief reset(
        const PublicStateSummary& state,
        const InitialOpponentStock& initial = {}
    );
    [[nodiscard]] InventoryBelief update(const PublicStateSummary& state);

    [[nodiscard]] bool initialized() const noexcept { return initialized_; }

private:
    [[nodiscard]] InventoryBelief snapshot(
        const PublicStateSummary& state,
        const Inventory& clearance,
        const InventoryInterval& clearance_interval,
        const ProductFlags& saturated
    ) const;

    int shed_capacity_;
    bool initialized_ = false;
    bool shed_partition_ambiguous_ = false;
    PublicStateSummary previous_{};
    Inventory joint_point_{};
    Inventory joint_lower_{};
    Inventory joint_upper_{};
    Inventory carried_point_{};
    ProductFlags ambiguous_{};
};

}  // namespace g001::market
