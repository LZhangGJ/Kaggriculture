#pragma once

#include "planner.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace g001::market {

struct MarketScenario {
    PlanningState state{};
    Inventory actual_rival_quantity{};
    std::array<int, product_count> actual_rival_sale_step{};
    int group = 0;
    int weight = 1;
};

struct ScenarioOutcome {
    std::int64_t revenue = 0;
    int production_failures = 0;
    int overflow_units = 0;
    int early_clear_units = 0;
    int sold_after_rival_units = 0;
};

struct AblationResult {
    MechanismMask mechanisms = 0;
    std::int64_t weighted_revenue = 0;
    double mean_revenue = 0.0;
    double worst_group_mean_revenue = 0.0;
    std::int64_t weighted_production_failures = 0;
    std::int64_t weighted_overflow_units = 0;
    std::int64_t weighted_early_clear_units = 0;
    std::int64_t weighted_sold_after_rival_units = 0;
};

[[nodiscard]] std::vector<MechanismMask> mechanism_ablations();

[[nodiscard]] ScenarioOutcome evaluate_scenario(
    const MarketScenario& scenario, MechanismMask mechanisms
);

[[nodiscard]] std::vector<AblationResult> evaluate_mechanisms_parallel(
    const std::vector<MarketScenario>& scenarios,
    const std::vector<MechanismMask>& candidates,
    std::size_t thread_count = 0
);

void rank_ablations(std::vector<AblationResult>& results);

}  // namespace g001::market
