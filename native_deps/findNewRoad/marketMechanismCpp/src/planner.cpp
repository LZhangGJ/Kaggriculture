#include "planner.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

namespace g001::market {
namespace {

constexpr int no_step = -1;

[[nodiscard]] bool enabled(MechanismMask mask, Mechanism mechanism) {
    return (mask & mechanism_bit(mechanism)) != 0;
}

[[nodiscard]] int sum(const Inventory& values) {
    return std::accumulate(values.begin(), values.end(), 0);
}

[[nodiscard]] std::vector<int> sale_windows(const PlanningState& state) {
    std::vector<int> result;
    if (state.sale_opportunity_steps.empty()) {
        result.reserve(static_cast<std::size_t>(state.last_action_step - state.step + 1));
        for (int step = state.step; step <= state.last_action_step; ++step) {
            result.push_back(step);
        }
        return result;
    }
    for (const auto step : state.sale_opportunity_steps) {
        if (step >= state.step && step <= state.last_action_step) result.push_back(step);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

[[nodiscard]] int next_window_after_now(const PlanningState& state) {
    const auto windows = sale_windows(state);
    const auto it = std::upper_bound(windows.begin(), windows.end(), state.step);
    return it == windows.end() ? state.last_action_step + 1 : *it;
}

[[nodiscard]] std::int64_t cash_reservation(
    const PlanningState& state, MechanismMask mask
) {
    if (!enabled(mask, Mechanism::ProtectPurchaseCash)) return 0;
    std::int64_t reserve = std::max<std::int64_t>(0, state.purchase_cash_required);
    if (!enabled(mask, Mechanism::RespectPurchaseDeadlines)) return reserve;

    const auto next_sale = next_window_after_now(state);
    std::int64_t due_before_next_sale = 0;
    for (const auto& commitment : state.cash_commitments) {
        if (commitment.route_critical && commitment.step < next_sale) {
            due_before_next_sale += std::max<std::int64_t>(0, commitment.amount);
        }
    }
    return std::max(reserve, due_before_next_sale);
}

[[nodiscard]] int market_inventory_at(
    const PlanningState& state, Product product, int step, MechanismMask mask
) {
    const auto i = static_cast<std::size_t>(product);
    auto inventory = state.market_inventory[i];
    if (enabled(mask, Mechanism::ModelTownDrain)) {
        inventory -= town_drain(product, state.step + 1, step + 1, state.town);
    }
    return inventory;
}

[[nodiscard]] int incoming_before_next_sale(
    const PlanningState& state, MechanismMask mask
) {
    if (!enabled(mask, Mechanism::ForecastProduction)) return 0;
    const auto next_sale = next_window_after_now(state);
    int incoming = 0;
    for (const auto& lot : state.future_production) {
        if (lot.step >= state.step && lot.step < next_sale) {
            incoming += std::max(0, lot.quantity);
        }
    }
    return incoming;
}

[[nodiscard]] int projected_peak_storage(
    const PlanningState& state, int immediate_sale, MechanismMask mask
) {
    auto level = std::max(0, sum(state.own_shed) + sum(state.own_carried) - immediate_sale);
    auto peak = level;
    if (!enabled(mask, Mechanism::ForecastProduction)) return peak;
    auto lots = state.future_production;
    std::sort(lots.begin(), lots.end(), [](const auto& left, const auto& right) {
        return left.step < right.step;
    });
    for (const auto& lot : lots) {
        if (lot.step < state.step || lot.step > state.last_action_step) continue;
        level += std::max(0, lot.quantity);
        peak = std::max(peak, level);
    }
    return peak;
}

// Adds exactly enough current sale units to satisfy a hard cash or capacity
// requirement. Every unit is ranked by the opportunity cost of selling now.
void add_required_units(
    const PlanningState& state,
    MechanismMask mask,
    Inventory& sale,
    int required_units,
    std::int64_t required_cash,
    std::int64_t& raised
) {
    const auto next_sale = std::min(state.last_action_step, next_window_after_now(state));
    while (required_units > 0 || raised < required_cash) {
        std::size_t best = product_count;
        std::tuple<std::int64_t, int, int> best_key{
            std::numeric_limits<std::int64_t>::min(), -1, -1
        };
        for (std::size_t i = 0; i < product_count; ++i) {
            if (sale[i] >= std::max(0, state.own_shed[i])) continue;
            const auto product = static_cast<Product>(i);
            const auto impact = enabled(mask, Mechanism::ModelOwnPriceImpact) ? sale[i] : 0;
            const auto now_quote = price(product, state.market_inventory[i] + impact);
            const auto later_quote = price(
                product, market_inventory_at(state, product, next_sale, mask)
            );
            const auto key = std::tuple<std::int64_t, int, int>{
                static_cast<std::int64_t>(now_quote) - later_quote,
                now_quote,
                -static_cast<int>(i)
            };
            if (key > best_key) {
                best_key = key;
                best = i;
            }
        }
        if (best == product_count) break;
        const auto product = static_cast<Product>(best);
        const auto inventory = state.market_inventory[best] +
            (enabled(mask, Mechanism::ModelOwnPriceImpact) ? sale[best] : 0);
        raised += price(product, inventory);
        ++sale[best];
        if (required_units > 0) --required_units;
    }
}

struct WindowValue {
    std::int64_t worst = 0;
    std::int64_t expected = 0;
    int step = no_step;
};

[[nodiscard]] WindowValue value_terminal_window(
    const PlanningState& state,
    Product product,
    int quantity,
    int candidate_step,
    MechanismMask mask
) {
    const auto i = static_cast<std::size_t>(product);
    const auto inventory = market_inventory_at(state, product, candidate_step, mask);
    auto rival_quantity = 0;
    auto rival_upper = 0;
    if (enabled(mask, Mechanism::InferRivalInventory)) {
        rival_quantity = std::max({
            0,
            state.belief.rival_stock[i],
            state.belief.liquidation[i].likely_quantity
        });
        rival_upper = std::max(rival_quantity, state.belief.rival_upper[i]);
    }
    if (!enabled(mask, Mechanism::RaceRivalLiquidation) || rival_upper == 0) {
        const auto revenue = sell_revenue(product, inventory, quantity);
        return {revenue, revenue, candidate_step};
    }

    auto window = state.belief.liquidation[i];
    if (window.earliest_step < state.step) window.earliest_step = state.step + 1;
    if (window.likely_step < window.earliest_step) window.likely_step = window.earliest_step;
    if (window.latest_step < window.likely_step) window.latest_step = state.last_action_step;

    const auto before = sell_revenue(product, inventory, quantity);
    const auto after = sell_after_rival_revenue(product, inventory, quantity, rival_quantity);
    const auto worst_after = sell_after_rival_revenue(
        product, inventory, quantity, rival_upper
    );
    const auto simultaneous = lockstep_sell_revenue(product, inventory, quantity, rival_quantity);
    if (candidate_step < window.earliest_step) return {before, before, candidate_step};
    if (candidate_step > window.latest_step) return {worst_after, after, candidate_step};
    return {
        std::min({before, simultaneous, worst_after}),
        (before + simultaneous + after) / 3,
        candidate_step
    };
}

[[nodiscard]] WindowValue choose_terminal_window(
    const PlanningState& state,
    Product product,
    int quantity,
    MechanismMask mask
) {
    const auto windows = sale_windows(state);
    if (windows.empty()) return {0, 0, no_step};
    if (!enabled(mask, Mechanism::SearchSafeSaleWindow)) {
        return value_terminal_window(state, product, quantity, windows.front(), mask);
    }

    WindowValue best{std::numeric_limits<std::int64_t>::min(),
                     std::numeric_limits<std::int64_t>::min(), no_step};
    for (const auto step : windows) {
        const auto value = value_terminal_window(state, product, quantity, step, mask);
        // If revenues tie, keep waiting. This removes unnecessary early clears.
        if (std::tie(value.worst, value.expected, value.step) >
            std::tie(best.worst, best.expected, best.step)) {
            best = value;
        }
    }
    return best;
}

}  // namespace

Phase classify_phase(const PlanningState& state) {
    if (state.step >= state.last_action_step - 10) return Phase::Liquidation;
    if (state.money < cash_reservation(state, state.mechanisms)) {
        return Phase::CapitalConstrained;
    }
    return Phase::CapacityConstrained;
}

SalePlan plan_sales(const PlanningState& state) {
    return plan_sales(state, state.mechanisms);
}

SalePlan plan_sales(const PlanningState& state, MechanismMask mask) {
    SalePlan result;
    result.execution_step.fill(no_step);

    PlanningState phased = state;
    phased.mechanisms = mask;
    result.phase = classify_phase(phased);
    result.reserved_cash = cash_reservation(state, mask);

    if (state.step >= state.last_action_step) {
        result.quantity = state.own_shed;
        result.scheduled_quantity = state.own_shed;
        result.execution_step.fill(state.step);
    } else if (result.phase == Phase::CapitalConstrained) {
        const auto shortfall = std::max<std::int64_t>(0, result.reserved_cash - state.money);
        std::int64_t raised = 0;
        add_required_units(state, mask, result.quantity, 0, shortfall, raised);
        result.scheduled_quantity = result.quantity;
        for (std::size_t i = 0; i < product_count; ++i) {
            if (result.quantity[i] > 0) result.execution_step[i] = state.step;
        }
        result.forced_units = sum(result.quantity);
        result.production_plan_funded = raised >= shortfall;
    } else if (result.phase == Phase::CapacityConstrained) {
        const auto current_storage = sum(state.own_shed) + sum(state.own_carried);
        const auto incoming = incoming_before_next_sale(state, mask);
        const auto overflow = enabled(mask, Mechanism::PreventCapacityLoss)
            ? std::max(0, current_storage + incoming - state.shed_capacity)
            : 0;
        std::int64_t ignored_cash = 0;
        add_required_units(state, mask, result.quantity, overflow, 0, ignored_cash);
        result.scheduled_quantity = result.quantity;
        for (std::size_t i = 0; i < product_count; ++i) {
            if (result.quantity[i] > 0) result.execution_step[i] = state.step;
        }
        result.forced_units = sum(result.quantity);
    } else {
        for (std::size_t i = 0; i < product_count; ++i) {
            const auto held = std::max(0, state.own_shed[i]);
            if (held == 0) continue;
            const auto product = static_cast<Product>(i);
            const auto window = choose_terminal_window(state, product, held, mask);
            if (window.step == no_step) continue;
            result.scheduled_quantity[i] = held;
            result.execution_step[i] = window.step;
            if (window.step == state.step) result.quantity[i] = held;
            result.expected_revenue += window.expected;
        }
    }

    for (std::size_t i = 0; i < product_count; ++i) {
        result.quantity[i] = std::clamp(result.quantity[i], 0, state.own_shed[i]);
        result.scheduled_quantity[i] = std::clamp(
            result.scheduled_quantity[i], 0, state.own_shed[i]
        );
        if (result.phase != Phase::Liquidation) {
            result.expected_revenue += sell_revenue(
                static_cast<Product>(i), state.market_inventory[i], result.quantity[i]
            );
        }
    }
    result.projected_peak_storage = projected_peak_storage(
        state, sum(result.quantity), mask
    );
    const auto next_peak = sum(state.own_shed) + sum(state.own_carried) -
        sum(result.quantity) + incoming_before_next_sale(state, mask);
    result.capacity_feasible = !enabled(mask, Mechanism::PreventCapacityLoss) ||
        next_peak <= state.shed_capacity;
    return result;
}

}  // namespace g001::market
