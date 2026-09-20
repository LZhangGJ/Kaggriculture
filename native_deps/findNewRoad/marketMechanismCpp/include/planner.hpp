#pragma once

#include "market.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace g001::market {

enum class Phase : std::uint8_t { CapitalConstrained, CapacityConstrained, Liquidation };

// These are deliberately mechanism switches, not tunable numeric thresholds.
// They make causal ablations possible without turning the search into parameter
// fitting against a fixed opponent set.
enum class Mechanism : std::uint32_t {
    ProtectPurchaseCash = 1u << 0,
    RespectPurchaseDeadlines = 1u << 1,
    ForecastProduction = 1u << 2,
    PreventCapacityLoss = 1u << 3,
    ModelTownDrain = 1u << 4,
    ModelOwnPriceImpact = 1u << 5,
    InferRivalInventory = 1u << 6,
    RaceRivalLiquidation = 1u << 7,
    SearchSafeSaleWindow = 1u << 8,
};

using MechanismMask = std::uint32_t;

[[nodiscard]] constexpr MechanismMask mechanism_bit(Mechanism mechanism) {
    return static_cast<MechanismMask>(mechanism);
}

[[nodiscard]] constexpr MechanismMask all_mechanisms() {
    return (mechanism_bit(Mechanism::SearchSafeSaleWindow) << 1u) - 1u;
}

struct CashCommitment {
    int step = 0;
    std::int64_t amount = 0;
    bool route_critical = true;
};

struct ProductionLot {
    int step = 0;
    Product product = Product::Wheat;
    int quantity = 0;
};

struct RivalSaleWindow {
    int earliest_step = -1;
    int likely_step = -1;
    int latest_step = -1;
    int likely_quantity = 0;
};

struct PublicBelief {
    Inventory rival_stock{};
    Inventory rival_lower{};
    Inventory rival_upper{};
    Inventory recent_clearance{};
    std::array<RivalSaleWindow, product_count> liquidation{};
};

struct PlanningState {
    int step = 0;
    int last_action_step = 718;
    int shed_capacity = 100;
    std::int64_t money = 0;
    std::int64_t purchase_cash_required = 0;
    Inventory market_inventory{};
    Inventory own_shed{};
    Inventory own_carried{};
    Inventory baseline_sale{};
    PublicBelief belief{};
    TownState town{};
    std::vector<CashCommitment> cash_commitments;
    std::vector<ProductionLot> future_production;
    // Exact steps at which the fixed movement route can execute a market sale.
    // Empty means every step is available, which preserves the old interface.
    std::vector<int> sale_opportunity_steps;
    MechanismMask mechanisms = all_mechanisms();
};

struct SalePlan {
    Phase phase = Phase::CapitalConstrained;
    // Quantity to execute at state.step (the legacy deployment-facing field).
    Inventory quantity{};
    // Full scheduled sale, including quantity that should be held until later.
    Inventory scheduled_quantity{};
    std::array<int, product_count> execution_step{};
    std::int64_t expected_revenue = 0;
    std::int64_t reserved_cash = 0;
    int forced_units = 0;
    int projected_peak_storage = 0;
    bool production_plan_funded = true;
    bool capacity_feasible = true;
};

[[nodiscard]] Phase classify_phase(const PlanningState& state);
[[nodiscard]] SalePlan plan_sales(const PlanningState& state);
[[nodiscard]] SalePlan plan_sales(const PlanningState& state, MechanismMask mechanisms);

}  // namespace g001::market
