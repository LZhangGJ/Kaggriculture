// SPDX-License-Identifier: Apache-2.0
// Fund a 144-turn tape segment by selling only inventory above its static reserve.
#pragma once

#include "sim.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace kag::native {

inline constexpr int SIX_DAY_TURNS = 144;

struct SixDayRequirements {
    double purchase_budget = 0.0;
    std::array<int, N_CROPS> starting_seeds{};
    std::array<int, N_ITEMS> starting_items{};
};

struct SixDayBudgetGuardSettings {
    int interval_turns = SIX_DAY_TURNS;
    int minimum_unit_price = 2;
    bool sales_first = true;
    bool protect_static_consumption = true;
};

inline int planned_quantity(int quantity) noexcept {
    return std::max(1, quantity);
}

template <typename ActionAt>
SixDayRequirements calculate_six_day_requirements(
    const State& state,
    const Config& config,
    int seat,
    int start,
    int end,
    ActionAt action_at) {
    SixDayRequirements result;
    std::array<int, N_CROPS> seed_balance{};
    std::array<int, N_ITEMS> item_balance{};
    std::array<int, 6> hires_by_day{};
    int quadrants = state.farms[seat].n_quadrants;

    for (int step = start; step < end; ++step) {
        const Action action = action_at(step);
        for (int index = 0; index < action.n_units; ++index) {
            const UnitAction& operation = action.units[index];
            const int quantity = planned_quantity(operation.n);
            if (operation.op == OP_PLANT && operation.arg < N_CROPS) {
                --seed_balance[operation.arg];
                result.starting_seeds[operation.arg] = std::max(
                    result.starting_seeds[operation.arg],
                    -seed_balance[operation.arg]);
            } else if (operation.op == OP_FEED) {
                --item_balance[WHEAT];
                result.starting_items[WHEAT] = std::max(
                    result.starting_items[WHEAT], -item_balance[WHEAT]);
            } else if (operation.op == OP_FERTILIZE) {
                --item_balance[FERTILIZER];
                result.starting_items[FERTILIZER] = std::max(
                    result.starting_items[FERTILIZER], -item_balance[FERTILIZER]);
            } else if (operation.op == OP_PLACE && operation.arg < N_ITEMS) {
                item_balance[operation.arg] -= quantity;
                result.starting_items[operation.arg] = std::max(
                    result.starting_items[operation.arg],
                    -item_balance[operation.arg]);
            }
        }
        for (int index = 0; index < action.n_orders; ++index) {
            const Order& order = action.orders[index];
            const int quantity = planned_quantity(order.n);
            if (order.op == M_HIRE) {
                const int day = std::min(5, std::max(0, (step - start) / 24));
                ++hires_by_day[day];
            } else if (order.op == M_BUY_LAND) {
                const int extra = quadrants - 1;
                if (extra >= 0 && extra < 3) {
                    result.purchase_budget += LAND_PRICES[extra];
                    ++quadrants;
                }
            } else if (order.op == M_BUY_SEED && order.item < N_CROPS) {
                result.purchase_budget += CROPS[order.item].seed * quantity;
                seed_balance[order.item] += quantity;
            } else if (
                order.op == M_BUY_PRODUCT &&
                (order.item == WHEAT || order.item == FERTILIZER)) {
                result.purchase_budget += state.market.prices[order.item] * quantity;
                item_balance[order.item] += quantity;
            } else if (order.op == M_BUY_ANIMAL && is_animal(order.item)) {
                result.purchase_budget += ANIMALS[order.item - GOOSE].cost * quantity;
                item_balance[order.item] += quantity;
            }
        }
    }
    for (int day = 0; day < static_cast<int>(hires_by_day.size()); ++day) {
        const int first_hire = day == 0 ? state.farms[seat].hires_today : 0;
        for (int index = 0; index < hires_by_day[day]; ++index) {
            result.purchase_budget += config.hire_mult * fib(first_hire + index);
        }
    }
    return result;
}

inline int owned_in_hands(const Farm& farm, int item) noexcept {
    int result = 0;
    for (int unit = 0; unit < std::min(farm.n_units, MAX_UNITS); ++unit) {
        result += std::max(0, static_cast<int>(farm.inv[unit][item]));
    }
    return result;
}

inline int existing_sale(const Action& action, int item) noexcept {
    int result = 0;
    for (int index = 0; index < action.n_orders; ++index) {
        const Order& order = action.orders[index];
        if (order.op == M_SELL && order.item == item && order.n > 0) {
            result += order.n;
        }
    }
    return result;
}

inline bool add_budget_sale(
    Action& action,
    const Config& config,
    int item,
    int quantity) noexcept {
    if (quantity <= 0) return true;
    for (int index = 0; index < action.n_orders; ++index) {
        Order& order = action.orders[index];
        if (order.op == M_SELL && order.item == item) {
            order.n += quantity;
            return true;
        }
    }
    const int limit = std::max(0, std::min(16, config.max_orders));
    if (action.n_orders >= limit) return false;
    action.orders[action.n_orders++] = {
        M_SELL,
        static_cast<std::uint8_t>(item),
        quantity,
    };
    return true;
}

inline void budget_sales_first(Action& action) noexcept {
    std::array<Order, 16> ordered{};
    int output = 0;
    for (int index = 0; index < action.n_orders; ++index) {
        if (action.orders[index].op == M_SELL) ordered[output++] = action.orders[index];
    }
    for (int index = 0; index < action.n_orders; ++index) {
        if (action.orders[index].op != M_SELL) ordered[output++] = action.orders[index];
    }
    std::copy(ordered.begin(), ordered.end(), action.orders);
}

inline Action apply_six_day_budget_guard(
    const State& state,
    const Config& config,
    int seat,
    const Action& input,
    const SixDayRequirements& requirements,
    const SixDayBudgetGuardSettings& settings = {}) noexcept {
    Action result = input;
    if (seat < 0 || seat > 1 || settings.interval_turns <= 0 ||
        state.step % settings.interval_turns != 0) {
        return result;
    }
    const Farm& farm = state.farms[seat];
    double available_cash = farm.money;
    for (int item = 0; item < N_PRODUCTS; ++item) {
        const int sold = std::min(
            std::max(0, static_cast<int>(farm.shed[item])),
            existing_sale(result, item));
        available_cash += sold * state.market.prices[item];
    }
    double shortfall = requirements.purchase_budget - available_cash;
    if (shortfall <= 0.0) return result;

    struct Candidate {
        int item = 0;
        int quantity = 0;
        int price = 0;
    };
    std::array<Candidate, N_PRODUCTS> candidates{};
    int count = 0;
    for (int item = 0; item < N_PRODUCTS; ++item) {
        const int price = state.market.prices[item];
        if (price < settings.minimum_unit_price) continue;
        const int protected_total = settings.protect_static_consumption
            ? requirements.starting_items[item]
            : 0;
        const int protected_shed = std::max(
            0,
            protected_total - owned_in_hands(farm, item));
        const int available = std::max(
            0,
            static_cast<int>(farm.shed[item]) - protected_shed -
                existing_sale(result, item));
        if (available > 0) candidates[count++] = {item, available, price};
    }
    std::stable_sort(
        candidates.begin(),
        candidates.begin() + count,
        [](const Candidate& left, const Candidate& right) {
            if (left.price != right.price) return left.price > right.price;
            return left.item < right.item;
        });

    int added = 0;
    for (int index = 0; index < count && shortfall > 0.0; ++index) {
        const Candidate& candidate = candidates[index];
        const int needed = static_cast<int>(std::ceil(shortfall / candidate.price));
        const int quantity = std::min(candidate.quantity, needed);
        if (!add_budget_sale(result, config, candidate.item, quantity)) continue;
        shortfall -= static_cast<double>(quantity * candidate.price);
        ++added;
    }
    if (added > 0 && settings.sales_first) budget_sales_first(result);
    return result;
}

}  // namespace kag::native
