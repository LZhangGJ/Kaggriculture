// SPDX-License-Identifier: Apache-2.0
// Packed Python-observation bridge linked directly with ShopForge SixDay Guard R1.
#include "policy_plugin_abi.hpp"
#include "runtime_types.hpp"

#include <algorithm>
#include <cstdint>

extern "C" std::uint32_t kag_policy_abi_version();
extern "C" void* kag_policy_create();
extern "C" void kag_policy_destroy(void* context);
extern "C" int kag_policy_act(
    void* context,
    const kag::State* state,
    const kag::Config* config,
    int seat,
    kag::Action* output);

namespace {

#pragma pack(push, 1)
struct PackedTile {
    std::uint8_t kind = 0;
};

struct PackedFarm {
    double money = 0;
    PackedTile tiles[kag::BOARD][kag::BOARD]{};
    std::int32_t n_units = 1;
    std::int32_t n_quadrants = 1;
    std::int32_t hires_today = 0;
    std::int16_t shed[kag::N_ITEMS]{};
    std::int16_t inv[kag::MAX_UNITS][kag::N_ITEMS]{};
};

struct PackedObservation {
    std::int32_t step = 0;
    std::int32_t n_shops = 0;
    std::int32_t market_inventory[kag::N_PRODUCTS]{};
    std::int32_t market_prices[kag::N_PRODUCTS]{};
    std::uint8_t shops[kag::MAX_SHOP_INSTANCES]{};
    PackedFarm farms[2]{};
};

struct PackedAction {
    std::uint8_t unit_ops[kag::MAX_UNITS]{};
    std::uint8_t unit_args[kag::MAX_UNITS]{};
    std::int16_t unit_ns[kag::MAX_UNITS]{};
    std::int32_t n_units = 1;
    std::uint8_t order_ops[16]{};
    std::uint8_t order_items[16]{};
    std::int32_t order_ns[16]{};
    std::int32_t n_orders = 0;
};
#pragma pack(pop)

void fill_state(const PackedObservation& observation, kag::State& state) {
    state = kag::State{};
    state.step = observation.step;
    state.n_shops = std::max(0, std::min(observation.n_shops, kag::MAX_SHOP_INSTANCES));
    for (int index = 0; index < state.n_shops; ++index)
        state.shops[index] = observation.shops[index];
    for (int item = 0; item < kag::N_PRODUCTS; ++item) {
        state.market.inventory[item] = observation.market_inventory[item];
        state.market.prices[item] = observation.market_prices[item];
    }
    for (int player = 0; player < 2; ++player) {
        const PackedFarm& source = observation.farms[player];
        kag::Farm& farm = state.farms[player];
        farm.money = source.money;
        farm.n_units = std::max(1, std::min(source.n_units, kag::MAX_UNITS));
        farm.n_quadrants = std::max(0, std::min(source.n_quadrants, 4));
        farm.hires_today = source.hires_today;
        for (int y = 0; y < kag::BOARD; ++y) {
            for (int x = 0; x < kag::BOARD; ++x) {
                farm.tiles[y][x].kind =
                    static_cast<kag::TileKind>(source.tiles[y][x].kind);
            }
        }
        for (int item = 0; item < kag::N_ITEMS; ++item) {
            farm.shed[item] = source.shed[item];
        }
        for (int unit = 0; unit < kag::MAX_UNITS; ++unit)
            for (int item = 0; item < kag::N_ITEMS; ++item)
                farm.inv[unit][item] = source.inv[unit][item];
    }
}
void pack_action(const kag::Action& action, PackedAction& packed) {
    packed = PackedAction{};
    packed.n_units = std::max(1, std::min(action.n_units, kag::MAX_UNITS));
    for (int unit = 0; unit < packed.n_units; ++unit) {
        packed.unit_ops[unit] = action.units[unit].op;
        packed.unit_args[unit] = action.units[unit].arg;
        packed.unit_ns[unit] = action.units[unit].n;
    }
    packed.n_orders = std::max(0, std::min(action.n_orders, 16));
    for (int index = 0; index < packed.n_orders; ++index) {
        packed.order_ops[index] = action.orders[index].op;
        packed.order_items[index] = action.orders[index].item;
        packed.order_ns[index] = action.orders[index].n;
    }
}

struct Session {
    void* context[2]{};
    int last_step[2]{-1, -1};

    ~Session() {
        for (void*& value : context) {
            if (value != nullptr) kag_policy_destroy(value);
            value = nullptr;
        }
    }

    void reset(int seat) {
        if (context[seat] != nullptr) kag_policy_destroy(context[seat]);
        context[seat] = kag_policy_create();
        last_step[seat] = -1;
    }
};

Session session;

}  // namespace

extern "C" std::uint32_t kag_submission_abi_version() {
    return kag_policy_abi_version() == kag::native::POLICY_PLUGIN_ABI_VERSION ? 1u : 0u;
}

extern "C" int kag_submission_act(
    const PackedObservation* observation,
    int seat,
    int episode_steps,
    PackedAction* output) {
    if (observation == nullptr || output == nullptr || seat < 0 || seat > 1)
        return 1;
    if (session.context[seat] == nullptr || observation->step == 0 ||
        observation->step < session.last_step[seat]) {
        session.reset(seat);
    }
    if (session.context[seat] == nullptr) return 1;
    kag::State state;
    fill_state(*observation, state);
    kag::Config config;
    if (episode_steps > 0) config.episode_steps = episode_steps;
    kag::Action action;
    if (kag_policy_act(session.context[seat], &state, &config, seat, &action) != 0)
        return 1;
    pack_action(action, *output);
    session.last_step[seat] = observation->step;
    return 0;
}
