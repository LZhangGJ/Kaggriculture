#include "market.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace g001::market {
namespace {

constexpr std::array<MarketParameters, product_count> params{{
    {25, 10000, 400, "sqrt", 0.8, "log", 0.2},
    {35, 10000, 450, "hinge", 1.0, "sqrt", 0.7},
    {60, 10000, 200, "hinge", 0.4, "sqrt", 0.6},
    {120, 10000, 100, "sqrt", 0.7, "linear", 1.6},
    {250, 10000, 300, "log", 0.2, "sq", 3.6},
    {50, 10000, 332, "hinge", 0.4, "log", 0.2},
    {160, 10000, 122, "sqrt", 0.6, "linear", 1.6},
    {200, 10000, 105, "log", 0.2, "sq", 3.2},
    {100, 10000, 200, "linear", 0.4, "linear", 0.4},
}};

double shape(std::string_view kind, double value, double transition) {
    value = std::max(0.0, value);
    if (kind == "linear") return value;
    if (kind == "sq") return value * value;
    if (kind == "sqrt") return std::sqrt(value);
    if (kind == "log") return std::log1p(value);
    if (kind == "hinge") {
        const auto unit = value / transition;
        const auto excess = std::max(0.0, unit - 1.0);
        return unit + 8.0 * excess * excess;
    }
    throw std::invalid_argument("unknown market shape");
}

int shed_used(const PlayerMarketState& player) {
    int total = 0;
    for (const auto value : player.shed) total += std::max(0, value);
    for (const auto value : player.animals) total += std::max(0, value);
    return total;
}

int fibonacci(int index) {
    int left = 1;
    int right = 1;
    for (int i = 0; i < std::max(0, index); ++i) {
        const auto next = left + right;
        left = right;
        right = next;
    }
    return left;
}

constexpr std::array<int, 5> seed_costs{10, 20, 50, 100, 80};
constexpr std::array<int, 3> animal_costs{300, 400, 500};
constexpr std::array<int, 3> land_costs{1000, 2000, 4000};

}  // namespace

const MarketParameters& parameters(Product product) {
    return params.at(static_cast<std::size_t>(product));
}

int price(Product product, int inventory) {
    const auto& p = parameters(product);
    double value = 0.0;
    if (inventory < p.equilibrium) {
        const auto amplitude = p.below_target * p.base /
            shape(p.below_shape, p.transition, p.transition);
        value = p.base + amplitude *
            shape(p.below_shape, p.equilibrium - inventory, p.transition);
    } else {
        const auto amplitude = p.above_target * p.base /
            shape(p.above_shape, p.transition, p.transition);
        value = p.base - amplitude *
            shape(p.above_shape, inventory - p.equilibrium, p.transition);
    }
    // std::nearbyint follows the default ties-to-even mode used by Python.
    return std::max(1, static_cast<int>(std::nearbyint(value)));
}

std::int64_t sell_revenue(Product product, int inventory, int quantity) {
    std::int64_t revenue = 0;
    for (int unit = 0; unit < std::max(0, quantity); ++unit) {
        const auto quote = price(product, inventory);
        revenue += quote;
        if (quote > 1) ++inventory;
    }
    return revenue;
}

std::int64_t lockstep_sell_revenue(
    Product product, int inventory, int own_quantity, int rival_quantity
) {
    std::int64_t revenue = 0;
    own_quantity = std::max(0, own_quantity);
    rival_quantity = std::max(0, rival_quantity);
    while (own_quantity-- > 0) {
        const auto quote = price(product, inventory);
        revenue += quote;
        const auto commits = 1 + static_cast<int>(rival_quantity > 0);
        if (rival_quantity > 0) --rival_quantity;
        if (quote > 1) inventory += commits;
    }
    return revenue;
}

QueueResult simulate_queue(
    const Inventory& initial_inventory,
    const std::array<PlayerMarketState, 2>& initial_players,
    const std::array<std::vector<Order>, 2>& queues,
    int max_orders,
    int shed_capacity,
    int hire_cost_multiplier
) {
    QueueResult result;
    result.players = initial_players;
    result.market_inventory = initial_inventory;
    for (int player = 0; player < 2; ++player) {
        const auto size = std::min<int>(max_orders, queues[player].size());
        result.committed[player].assign(size, 0);
        result.committed_cash[player].assign(size, 0);
        result.minimum_committed_quote[player].assign(size, -1);
        result.maximum_committed_quote[player].assign(size, -1);
    }
    const auto slots = std::min(
        max_orders,
        static_cast<int>(std::max(queues[0].size(), queues[1].size()))
    );
    for (int slot = 0; slot < slots; ++slot) {
        std::array<const Order*, 2> current{};
        std::array<int, 2> remaining{};
        std::array<bool, 2> active{};
        for (int player = 0; player < 2; ++player) {
            if (slot < static_cast<int>(queues[player].size())) {
                current[player] = &queues[player][slot];
                remaining[player] = std::max(0, current[player]->quantity);
                active[player] = current[player]->operation != Operation::Pass &&
                    remaining[player] > 0;
            }
        }

        for (int player = 0; player < 2; ++player) {
            if (!active[player] || current[player] == nullptr) continue;
            const auto& order = *current[player];
            auto& state = result.players[player];
            if (order.operation == Operation::Hire) {
                const auto cost = hire_cost_multiplier * fibonacci(state.hires_today);
                if (state.money >= cost) {
                    state.money -= cost;
                    ++state.hires_today;
                    ++state.hands;
                    result.committed[player][slot] = 1;
                    result.committed_cash[player][slot] = -cost;
                    result.minimum_committed_quote[player][slot] = cost;
                    result.maximum_committed_quote[player][slot] = cost;
                }
                active[player] = false;
            } else if (order.operation == Operation::BuyLand) {
                const auto index = state.unlocked_quadrants - 1;
                if (index >= 0 && index < static_cast<int>(land_costs.size()) &&
                    state.money >= land_costs[index]) {
                    state.money -= land_costs[index];
                    ++state.unlocked_quadrants;
                    result.committed[player][slot] = 1;
                    result.committed_cash[player][slot] = -land_costs[index];
                    result.minimum_committed_quote[player][slot] = land_costs[index];
                    result.maximum_committed_quote[player][slot] = land_costs[index];
                }
                active[player] = false;
            }
        }

        while (active[0] || active[1]) {
            std::array<int, 2> quotes{};
            std::array<bool, 2> quoted{};
            for (int player = 0; player < 2; ++player) {
                if (!active[player] || current[player] == nullptr || remaining[player] <= 0) {
                    active[player] = false;
                    continue;
                }
                const auto& order = *current[player];
                const auto index = static_cast<std::size_t>(order.product);
                switch (order.operation) {
                    case Operation::Sell:
                        quotes[player] = price(order.product, result.market_inventory[index]);
                        quoted[player] = true;
                        break;
                    case Operation::BuyProduct:
                        if (order.product == Product::Wheat || order.product == Product::Fertilizer) {
                            quotes[player] = price(order.product, result.market_inventory[index] - 1);
                            quoted[player] = true;
                        }
                        break;
                    case Operation::BuySeed:
                        if (index < seed_costs.size()) {
                            quotes[player] = seed_costs[index];
                            quoted[player] = true;
                        }
                        break;
                    case Operation::BuyAnimal:
                        quotes[player] = animal_costs[static_cast<std::size_t>(order.animal)];
                        quoted[player] = true;
                        break;
                    default:
                        break;
                }
                if (!quoted[player]) active[player] = false;
            }
            if (!quoted[0] && !quoted[1]) break;

            bool any_commit = false;
            // Both quotes are pre-commit; commits occur in player order.
            for (int player = 0; player < 2; ++player) {
                if (!quoted[player] || current[player] == nullptr) continue;
                const auto& order = *current[player];
                auto& state = result.players[player];
                const auto index = static_cast<std::size_t>(order.product);
                const auto quote = quotes[player];
                bool success = false;
                switch (order.operation) {
                    case Operation::Sell:
                        if (state.shed[index] > 0) {
                            --state.shed[index];
                            state.money += quote;
                            if (quote > 1) ++result.market_inventory[index];
                            success = true;
                        }
                        break;
                    case Operation::BuyProduct:
                        if (state.money >= quote && shed_used(state) < shed_capacity) {
                            state.money -= quote;
                            ++state.shed[index];
                            --result.market_inventory[index];
                            success = true;
                        }
                        break;
                    case Operation::BuySeed:
                        if (state.money >= quote) {
                            state.money -= quote;
                            ++state.seeds[index];
                            success = true;
                        }
                        break;
                    case Operation::BuyAnimal:
                        if (state.money >= quote && shed_used(state) < shed_capacity) {
                            state.money -= quote;
                            ++state.animals[static_cast<std::size_t>(order.animal)];
                            success = true;
                        }
                        break;
                    default:
                        break;
                }
                if (success) {
                    --remaining[player];
                    ++result.committed[player][slot];
                    const auto signed_quote = order.operation == Operation::Sell
                        ? static_cast<std::int64_t>(quote)
                        : -static_cast<std::int64_t>(quote);
                    result.committed_cash[player][slot] += signed_quote;
                    auto& minimum = result.minimum_committed_quote[player][slot];
                    auto& maximum = result.maximum_committed_quote[player][slot];
                    minimum = minimum < 0 ? quote : std::min(minimum, quote);
                    maximum = std::max(maximum, quote);
                    any_commit = true;
                    if (remaining[player] <= 0) active[player] = false;
                } else {
                    active[player] = false;
                }
            }
            if (!any_commit) break;
        }
    }
    return result;
}

std::int64_t sell_after_rival_revenue(
    Product product, int inventory, int own_quantity, int rival_quantity
) {
    rival_quantity = std::max(0, rival_quantity);
    // Rival units still enter the public inventory even though their revenue is
    // irrelevant to us.  At the one-coin floor the engine stops adding supply.
    for (int unit = 0; unit < rival_quantity; ++unit) {
        if (price(product, inventory) > 1) ++inventory;
    }
    return sell_revenue(product, inventory, own_quantity);
}

int town_drain(Product product, int first_step, int stop_step, const TownState& town) {
    int total = 0;
    const auto index = static_cast<std::size_t>(product);
    for (int step = std::max(0, first_step); step < std::max(0, stop_step); ++step) {
        if (step % std::max(1, town.shop_interval) == 0) {
            total += town.shop_demand_per_tick[index];
        }
        if (product != Product::Fertilizer && step % std::max(1, town.center_interval) == 0) {
            ++total;
        }
    }
    return total;
}

}  // namespace g001::market
