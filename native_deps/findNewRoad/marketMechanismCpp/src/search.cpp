#include "search.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <numeric>
#include <thread>
#include <tuple>

namespace g001::market {
namespace {

[[nodiscard]] int sum(const Inventory& values) {
    return std::accumulate(values.begin(), values.end(), 0);
}

[[nodiscard]] int realized_inventory_at(
    const MarketScenario& scenario, Product product, int step
) {
    const auto i = static_cast<std::size_t>(product);
    return scenario.state.market_inventory[i] - town_drain(
        product, scenario.state.step + 1, step + 1, scenario.state.town
    );
}

}  // namespace

std::vector<MechanismMask> mechanism_ablations() {
    std::vector<MechanismMask> result{all_mechanisms()};
    for (std::uint32_t bit = 1; bit <= mechanism_bit(Mechanism::SearchSafeSaleWindow);
         bit <<= 1u) {
        result.push_back(all_mechanisms() & ~bit);
    }
    const auto opening = mechanism_bit(Mechanism::ProtectPurchaseCash) |
        mechanism_bit(Mechanism::RespectPurchaseDeadlines) |
        mechanism_bit(Mechanism::ModelOwnPriceImpact);
    const auto middle = mechanism_bit(Mechanism::ForecastProduction) |
        mechanism_bit(Mechanism::PreventCapacityLoss) |
        mechanism_bit(Mechanism::ModelTownDrain) |
        mechanism_bit(Mechanism::ModelOwnPriceImpact);
    const auto ending = mechanism_bit(Mechanism::ModelTownDrain) |
        mechanism_bit(Mechanism::ModelOwnPriceImpact) |
        mechanism_bit(Mechanism::InferRivalInventory) |
        mechanism_bit(Mechanism::RaceRivalLiquidation) |
        mechanism_bit(Mechanism::SearchSafeSaleWindow);
    result.push_back(opening);
    result.push_back(middle);
    result.push_back(ending);
    result.push_back(0);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

ScenarioOutcome evaluate_scenario(
    const MarketScenario& scenario, MechanismMask mechanisms
) {
    const auto plan = plan_sales(scenario.state, mechanisms);
    ScenarioOutcome outcome;

    const auto immediate_storage = sum(scenario.state.own_shed) +
        sum(scenario.state.own_carried) - sum(plan.quantity);
    int incoming = 0;
    auto next_sale = scenario.state.last_action_step + 1;
    for (const auto step : scenario.state.sale_opportunity_steps) {
        if (step > scenario.state.step) next_sale = std::min(next_sale, step);
    }
    for (const auto& lot : scenario.state.future_production) {
        if (lot.step >= scenario.state.step && lot.step < next_sale) {
            incoming += std::max(0, lot.quantity);
        }
    }
    outcome.overflow_units = std::max(
        0, immediate_storage + incoming - scenario.state.shed_capacity
    );

    std::int64_t immediate_revenue = 0;
    for (std::size_t i = 0; i < product_count; ++i) {
        const auto quantity = plan.scheduled_quantity[i];
        const auto own_step = plan.execution_step[i];
        if (quantity <= 0 || own_step < scenario.state.step) continue;
        const auto product = static_cast<Product>(i);
        const auto inventory = realized_inventory_at(scenario, product, own_step);
        const auto rival_quantity = std::max(0, scenario.actual_rival_quantity[i]);
        const auto rival_step = scenario.actual_rival_sale_step[i];
        if (rival_quantity > 0 && rival_step >= 0 && rival_step < own_step) {
            outcome.revenue += sell_after_rival_revenue(
                product, inventory, quantity, rival_quantity
            );
            outcome.sold_after_rival_units += quantity;
        } else if (rival_quantity > 0 && rival_step == own_step) {
            outcome.revenue += lockstep_sell_revenue(
                product, inventory, quantity, rival_quantity
            );
        } else {
            outcome.revenue += sell_revenue(product, inventory, quantity);
        }
        if (own_step == scenario.state.step) {
            // At the current step no prior hidden rival action is possible;
            // using the public quote path is the cash actually made available
            // to subsequent same-turn purchases.
            immediate_revenue += sell_revenue(
                product, scenario.state.market_inventory[i], quantity
            );
        }
        if (plan.phase == Phase::Liquidation && rival_quantity == 0 &&
            own_step < scenario.state.last_action_step) {
            outcome.early_clear_units += quantity;
        }
    }

    // Ground-truth route feasibility is independent of which mechanism the
    // candidate enabled.  An ablation cannot improve its score by disabling
    // both the protection and the referee that detects the missed purchase.
    std::int64_t route_cash_target = std::max<std::int64_t>(
        0, scenario.state.purchase_cash_required
    );
    std::int64_t commitments_before_next_sale = 0;
    for (const auto& commitment : scenario.state.cash_commitments) {
        if (commitment.route_critical && commitment.step < next_sale) {
            commitments_before_next_sale += std::max<std::int64_t>(0, commitment.amount);
        }
    }
    route_cash_target = std::max(route_cash_target, commitments_before_next_sale);
    outcome.production_failures =
        scenario.state.money + immediate_revenue < route_cash_target;
    return outcome;
}

std::vector<AblationResult> evaluate_mechanisms_parallel(
    const std::vector<MarketScenario>& scenarios,
    const std::vector<MechanismMask>& candidates,
    std::size_t thread_count
) {
    if (scenarios.empty() || candidates.empty()) return {};
    const auto task_count = scenarios.size() * candidates.size();
    std::vector<ScenarioOutcome> outcomes(task_count);
    std::atomic<std::size_t> next{0};
    if (thread_count == 0) thread_count = std::thread::hardware_concurrency();
    thread_count = std::clamp<std::size_t>(thread_count, 1, task_count);

    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t worker = 0; worker < thread_count; ++worker) {
        workers.emplace_back([&] {
            constexpr std::size_t chunk = 32;
            while (true) {
                const auto begin = next.fetch_add(chunk, std::memory_order_relaxed);
                if (begin >= task_count) break;
                const auto end = std::min(task_count, begin + chunk);
                for (auto task = begin; task < end; ++task) {
                    const auto candidate = task / scenarios.size();
                    const auto scenario = task % scenarios.size();
                    outcomes[task] = evaluate_scenario(
                        scenarios[scenario], candidates[candidate]
                    );
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();

    std::vector<AblationResult> results(candidates.size());
    for (std::size_t candidate = 0; candidate < candidates.size(); ++candidate) {
        auto& result = results[candidate];
        result.mechanisms = candidates[candidate];
        std::int64_t total_weight = 0;
        std::map<int, std::pair<std::int64_t, std::int64_t>> groups;
        for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
            const auto weight = std::max(0, scenarios[scenario].weight);
            const auto& outcome = outcomes[candidate * scenarios.size() + scenario];
            total_weight += weight;
            result.weighted_revenue += outcome.revenue * weight;
            result.weighted_production_failures += outcome.production_failures * weight;
            result.weighted_overflow_units += outcome.overflow_units * weight;
            result.weighted_early_clear_units += outcome.early_clear_units * weight;
            result.weighted_sold_after_rival_units += outcome.sold_after_rival_units * weight;
            auto& group = groups[scenarios[scenario].group];
            group.first += outcome.revenue * weight;
            group.second += weight;
        }
        result.mean_revenue = total_weight == 0 ? 0.0 :
            static_cast<double>(result.weighted_revenue) / total_weight;
        result.worst_group_mean_revenue = std::numeric_limits<double>::infinity();
        for (const auto& [id, group] : groups) {
            (void)id;
            if (group.second > 0) {
                result.worst_group_mean_revenue = std::min(
                    result.worst_group_mean_revenue,
                    static_cast<double>(group.first) / group.second
                );
            }
        }
        if (result.worst_group_mean_revenue == std::numeric_limits<double>::infinity()) {
            result.worst_group_mean_revenue = 0.0;
        }
    }
    return results;
}

void rank_ablations(std::vector<AblationResult>& results) {
    std::stable_sort(results.begin(), results.end(), [](const auto& left, const auto& right) {
        if (left.weighted_production_failures != right.weighted_production_failures) {
            return left.weighted_production_failures < right.weighted_production_failures;
        }
        if (left.weighted_overflow_units != right.weighted_overflow_units) {
            return left.weighted_overflow_units < right.weighted_overflow_units;
        }
        if (left.weighted_early_clear_units != right.weighted_early_clear_units) {
            return left.weighted_early_clear_units < right.weighted_early_clear_units;
        }
        if (left.worst_group_mean_revenue != right.worst_group_mean_revenue) {
            return left.worst_group_mean_revenue > right.worst_group_mean_revenue;
        }
        return left.mean_revenue > right.mean_revenue;
    });
}

}  // namespace g001::market
