#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace g001::market {

enum class Product : std::uint8_t {
    Wheat, Carrot, Tomato, Strawberry, Melon, Egg, Milk, Wool, Fertilizer, Count
};

constexpr std::size_t product_count = static_cast<std::size_t>(Product::Count);
using Inventory = std::array<int, product_count>;
using AnimalInventory = std::array<int, 3>;

enum class Operation : std::uint8_t {
    Pass, Sell, BuyProduct, BuySeed, BuyAnimal, Hire, BuyLand
};

enum class Animal : std::uint8_t { Goose, Cow, Sheep };

struct Order {
    Operation operation = Operation::Pass;
    Product product = Product::Wheat;
    Animal animal = Animal::Goose;
    int quantity = 1;
};

struct PlayerMarketState {
    std::int64_t money = 0;
    Inventory shed{};
    AnimalInventory animals{};
    Inventory seeds{};
    int hires_today = 0;
    int hands = 0;
    int unlocked_quadrants = 1;
};

struct QueueResult {
    std::array<PlayerMarketState, 2> players{};
    Inventory market_inventory{};
    std::array<std::vector<int>, 2> committed{};
    // Exact per-slot cash flow and quote extrema for committed units. These
    // are simulator outputs of the causal candidate replay, not observations
    // available to a deployed policy.  A slot with no fill keeps quote -1.
    std::array<std::vector<std::int64_t>, 2> committed_cash{};
    std::array<std::vector<int>, 2> minimum_committed_quote{};
    std::array<std::vector<int>, 2> maximum_committed_quote{};
};

struct MarketParameters {
    int base;
    int equilibrium;
    int transition;
    std::string_view below_shape;
    double below_target;
    std::string_view above_shape;
    double above_target;
};

[[nodiscard]] const MarketParameters& parameters(Product product);
[[nodiscard]] int price(Product product, int inventory);
[[nodiscard]] std::int64_t sell_revenue(Product product, int inventory, int quantity);
[[nodiscard]] std::int64_t lockstep_sell_revenue(
    Product product, int inventory, int own_quantity, int rival_quantity
);
[[nodiscard]] QueueResult simulate_queue(
    const Inventory& market_inventory,
    const std::array<PlayerMarketState, 2>& players,
    const std::array<std::vector<Order>, 2>& queues,
    int max_orders = 10,
    int shed_capacity = 100,
    int hire_cost_multiplier = 1
);

// Revenue when the rival dump is committed before our order in the same market
// action.  Together with lockstep_sell_revenue this brackets ordering risk.
[[nodiscard]] std::int64_t sell_after_rival_revenue(
    Product product, int inventory, int own_quantity, int rival_quantity
);

struct TownState {
    std::array<int, product_count> shop_demand_per_tick{};
    int shop_interval = 4;
    int center_interval = 24;
};

[[nodiscard]] int town_drain(
    Product product, int first_step, int stop_step, const TownState& town
);

}  // namespace g001::market
