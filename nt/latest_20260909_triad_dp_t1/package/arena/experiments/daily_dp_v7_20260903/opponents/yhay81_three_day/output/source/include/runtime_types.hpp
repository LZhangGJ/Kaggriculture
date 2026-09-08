// SPDX-License-Identifier: Apache-2.0
// Minimal Kaggriculture types required by the submitted tape router.
#pragma once

#include <cstdint>

namespace kag {

enum Item : std::uint8_t {
    WHEAT = 0,
    CARROT,
    TOMATO,
    STRAWBERRY,
    MELON,
    EGG,
    MILK,
    WOOL,
    FERTILIZER,
    GOOSE,
    COW,
    SHEEP,
    N_ITEMS,
};

inline constexpr int N_PRODUCTS = 9;
inline constexpr int N_CROPS = 5;
inline constexpr int N_ANIMALS = 3;
inline constexpr int MAX_UNITS = 40;
inline constexpr int BOARD = 10;
inline constexpr int MAX_SHOP_INSTANCES = 8;

inline bool is_animal(std::uint8_t item) {
    return item >= GOOSE && item < N_ITEMS;
}

enum Op : std::uint8_t {
    OP_PASS = 0,
    OP_NORTH,
    OP_SOUTH,
    OP_EAST,
    OP_WEST,
    OP_PICKUP,
    OP_DROP,
    OP_PLACE,
    OP_PLANT,
    OP_WATER,
    OP_HARVEST,
    OP_FERTILIZE,
    OP_DIG,
    OP_BUILD_COOP,
    OP_BUILD_PASTURE,
    OP_FEED,
    OP_COLLECT_FERTILIZER,
    OP_CARE,
};

enum MOp : std::uint8_t {
    M_NONE = 0,
    M_HIRE,
    M_BUY_LAND,
    M_BUY_SEED,
    M_BUY_PRODUCT,
    M_BUY_ANIMAL,
    M_SELL,
};

enum ShopId : std::uint8_t {
    SHOP_BAKERY = 0,
    SHOP_BRUNCH_SPOT,
    SHOP_FARMERS_MARKET,
    SHOP_ICE_CREAM_SHOP,
    SHOP_PET_CAFE,
    SHOP_PIZZA_SHOP,
    SHOP_SMOOTHIE_SHOP,
    SHOP_YARN_STORE,
};

enum TileKind : std::uint8_t {
    T_EMPTY = 0,
    T_LOCKED,
    T_WEED,
    T_COOP,
    T_PASTURE,
    T_PLANT,
};

struct CropDef {
    int seed;
};

inline constexpr CropDef CROPS[N_CROPS] = {{10}, {20}, {50}, {100}, {80}};

struct AnimalDef {
    int cost;
};

inline constexpr AnimalDef ANIMALS[N_ANIMALS] = {{300}, {400}, {500}};
inline constexpr int LAND_PRICES[3] = {1000, 2000, 4000};

inline int fib(int n) {
    int previous = 1;
    int current = 1;
    for (int index = 0; index < n; ++index) {
        const int next = previous + current;
        previous = current;
        current = next;
    }
    return previous;
}

struct Config {
    int episode_steps = 720;
    int max_orders = 10;
    int hire_mult = 1;
};

struct Tile {
    TileKind kind = T_EMPTY;
};

struct Farm {
    double money = 0;
    Tile tiles[BOARD][BOARD]{};
    int n_units = 1;
    int n_quadrants = 1;
    int hires_today = 0;
    std::int16_t shed[N_ITEMS]{};
    std::int16_t inv[MAX_UNITS][N_ITEMS]{};
};

struct Market {
    std::int32_t inventory[N_PRODUCTS]{};
    std::int32_t prices[N_PRODUCTS]{};
};

struct State {
    Farm farms[2]{};
    Market market{};
    std::uint8_t shops[MAX_SHOP_INSTANCES]{};
    int n_shops = 0;
    int step = 0;
};

struct UnitAction {
    std::uint8_t op = OP_PASS;
    std::uint8_t arg = 0;
    std::int16_t n = 1;
};

struct Order {
    std::uint8_t op = M_NONE;
    std::uint8_t item = 0;
    std::int32_t n = 0;
};

struct Action {
    UnitAction units[MAX_UNITS]{};
    int n_units = 1;
    Order orders[16]{};
    int n_orders = 0;
};

}  // namespace kag
