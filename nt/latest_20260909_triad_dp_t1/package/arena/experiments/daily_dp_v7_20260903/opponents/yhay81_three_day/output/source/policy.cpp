// SPDX-License-Identifier: Apache-2.0
#include "policy_plugin_abi.hpp"
#include "six_day_budget_guard.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int kSegmentTurns = 72;
constexpr int kDecisionStep = 360;
#include "tape.inc"

kag::Action decode_action(const char* encoded) {
    kag::Action action{};
    std::istringstream input(encoded);
    if (!(input >> action.n_units >> action.n_orders) ||
        action.n_units < 0 || action.n_units > kag::MAX_UNITS ||
        action.n_orders < 0 || action.n_orders > 16) {
        throw std::runtime_error("invalid encoded tape action counts");
    }
    for (int index = 0; index < action.n_units; ++index) {
        int operation = 0, argument = 0, quantity = 0;
        if (!(input >> operation >> argument >> quantity))
            throw std::runtime_error("invalid encoded unit action");
        action.units[index] = {
            static_cast<std::uint8_t>(operation),
            static_cast<std::uint8_t>(argument),
            static_cast<std::int16_t>(quantity),
        };
    }
    for (int index = 0; index < action.n_orders; ++index) {
        int operation = 0, item = 0, quantity = 0;
        if (!(input >> operation >> item >> quantity))
            throw std::runtime_error("invalid encoded market order");
        action.orders[index] = {
            static_cast<std::uint8_t>(operation),
            static_cast<std::uint8_t>(item),
            quantity,
        };
    }
    int trailing = 0;
    if (input >> trailing) throw std::runtime_error("trailing encoded tape value");
    return action;
}



int plant_tiles(const kag::Farm& farm) {
    int result = 0;
    for (int y = 0; y < kag::BOARD; ++y)
        for (int x = 0; x < kag::BOARD; ++x)
            result += farm.tiles[y][x].kind == kag::T_PLANT;
    return result;
}


int select_route(const kag::State& state, int seat) {
    static_cast<void>(seat);
    // Route 1: segment72_b01_0091_0ebdd1a079
    if (state.n_shops >= 1 &&
        state.shops[0] == kag::SHOP_BAKERY &&
        static_cast<double>(state.market.inventory[kag::FERTILIZER]) <= 10232.5) return 1;
    // Route 1: segment72_b01_0091_0ebdd1a079
    if (state.n_shops >= 1 &&
        state.shops[0] == kag::SHOP_PET_CAFE &&
        static_cast<double>(plant_tiles(state.farms[1 - seat])) <= 64.5) return 1;
    return 0;
}

struct Context {
    std::array<std::array<kag::Action, kTurns>, kRoutes> actions{};
    int selected_route = 0;

    Context() {
        for (int route = 0; route < kRoutes; ++route)
            for (int step = 0; step < kTurns; ++step)
                actions[route][step] = decode_action(kEncodedTapes[route][step]);
    }

    kag::Action action_for(int step) const {
        if (step < 0 || step >= kTurns) return kag::Action{};
        return actions[selected_route][static_cast<std::size_t>(step)];
    }

    kag::Action act(const kag::State& state, const kag::Config& config, int seat) {
        if (state.step < 0 || state.step >= kTurns) return kag::Action{};
        if (state.step == 0) selected_route = 0;
        if (state.step == kDecisionStep) selected_route = select_route(state, seat);

        const kag::Action input = action_for(state.step);
        if (state.step % kSegmentTurns != 0) return input;
        const int end = std::min(kTurns, state.step + kSegmentTurns);
        const auto requirements = kag::native::calculate_six_day_requirements(
            state,
            config,
            seat,
            state.step,
            end,
            [&](int step) { return action_for(step); });
        kag::native::SixDayBudgetGuardSettings settings;
        settings.interval_turns = kSegmentTurns;
        return kag::native::apply_six_day_budget_guard(
            state, config, seat, input, requirements, settings);
    }
};

}  // namespace

extern "C" std::uint32_t kag_policy_abi_version() {
    return kag::native::POLICY_PLUGIN_ABI_VERSION;
}
extern "C" void* kag_policy_create() {
    try { return new Context{}; } catch (...) { return nullptr; }
}
extern "C" void kag_policy_destroy(void* context) {
    delete static_cast<Context*>(context);
}
extern "C" int kag_policy_act(
    void* raw_context,
    const kag::State* state,
    const kag::Config* config,
    int seat,
    kag::Action* output) {
    if (!raw_context || !state || !config || !output || seat < 0 || seat > 1) return 1;
    try {
        *output = static_cast<Context*>(raw_context)->act(*state, *config, seat);
        return 0;
    } catch (...) {
        return 2;
    }
}
