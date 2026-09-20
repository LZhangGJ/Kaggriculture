#include "native_general_market.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <utility>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

void step(fastkag::Simulator& env, fastkag::PlayerAction left = {}) {
  env.step({std::move(left), fastkag::PlayerAction{}});
}

void advance_to(fastkag::Simulator& env, int target) {
  while (env.step_count() < target) step(env);
}

}  // namespace

int main() {
  using namespace fastkag;
  {
    Simulator env(Config{}, 123);
    std::vector<NativeFutureUnitFrame> future{
        {1, {{Op::PLANT, Item::STRAWBERRY, 1}}}};
    const auto result = compile_native_general_market(env, 0, {}, future);
    require(result.feasible && result.market.size() == 1 &&
                result.market[0].op == Op::BUY_SEED &&
                result.market[0].item == Item::STRAWBERRY &&
                result.market[0].quantity == 1,
            "next-step PLANT compiles an independent current BUY_SEED");
  }
  {
    Simulator env(Config{}, 124);
    const std::vector<Action> demands{
        {Op::BUY_SEED, Item::TOMATO, 1},
        {Op::BUY_SEED, Item::TOMATO, 2}};
    const auto result = compile_native_general_market(
        env, 0, {{Op::PLANT, Item::TOMATO, 1}}, {}, demands);
    require(result.feasible && result.market.size() == 1 &&
                result.market[0].op == Op::BUY_SEED &&
                result.market[0].item == Item::TOMATO &&
                result.market[0].quantity == 3 &&
                result.current_unit_replacements.size() == 1 &&
                result.current_unit_replacements[0].replacement.op == Op::PASS,
            "current PLANT waits while state-target seed demands merge");
    PlayerAction buy;
    buy.units.push_back(result.current_unit_replacements[0].replacement);
    buy.market = result.market;
    step(env, std::move(buy));
    require(env.privates()[0].seeds[static_cast<int>(Item::TOMATO)] == 3 &&
                env.farms()[0].tiles[44].kind == TileKind::EMPTY,
            "market fill becomes inventory only after the current unit phase");
    step(env, PlayerAction{{{Op::PLANT, Item::TOMATO, 1}}, {}});
    require(env.privates()[0].seeds[static_cast<int>(Item::TOMATO)] == 2 &&
                env.farms()[0].tiles[44].kind == TileKind::PLANT,
            "confirmed seed is usable by the next observation");
  }
  {
    Simulator env(Config{}, 125);
    const std::vector<NativeFutureUnitFrame> future{
        {1, {{Op::PLANT, Item::STRAWBERRY, 1}}}};
    const std::vector<Action> same_demand{
        {Op::BUY_SEED, Item::STRAWBERRY, 1}};
    const auto result =
        compile_native_general_market(env, 0, {}, future, same_demand);
    require(result.feasible && result.market.size() == 1 &&
                result.market[0].op == Op::BUY_SEED &&
                result.market[0].quantity == 1,
            "explicit and unit-derived hard buys are not duplicated");
  }
  {
    Config config;
    config.starting_money = 100000;
    Simulator env(config, 456);
    PlayerAction fill;
    fill.units.push_back({Op::BUILD_COOP});
    fill.market.push_back({Op::BUY_PRODUCT, Item::WHEAT, 100});
    env.step({fill, PlayerAction{}});
    require(env.privates()[0].shed[0] == 100,
            "official fixture fills the shed before an animal obligation");
    std::vector<NativeFutureUnitFrame> future{
        {2, {{Op::PICKUP, Item::GOOSE, 1}}},
        {3, {{Op::PLACE, Item::GOOSE, 1}}},
    };
    const auto result = compile_native_general_market(env, 0, {Action{}}, future);
    if (!(result.feasible && result.market.size() == 2 &&
          result.market[0].op == Op::SELL &&
          result.market[1].op == Op::BUY_ANIMAL &&
          result.audit.funding_sell_units == 1)) {
      std::cerr << "capacity debug feasible=" << result.feasible
                << " market=" << result.market.size()
                << " buy_due=" << result.audit.purchase_orders_due
                << " sell_orders=" << result.audit.funding_sell_orders
                << " sell_units=" << result.audit.funding_sell_units
                << " reason=" << result.audit.reason << '\n';
    }
    require(result.feasible && result.market.size() == 2 &&
                result.market[0].op == Op::SELL &&
                result.market[1].op == Op::BUY_ANIMAL &&
                result.audit.funding_sell_units == 1,
            "capacity allocator emits SELL before the hard animal BUY");
  }
  {
    Simulator env(Config{}, 789);
    PlayerAction setup;
    setup.units.push_back({Op::BUILD_COOP});
    env.step({setup, PlayerAction{}});
    const std::vector<Action> current{{Op::PICKUP, Item::GOOSE, 1}};
    const std::vector<NativeFutureUnitFrame> future{
        {2, {{Op::PLACE, Item::GOOSE, 1}}}};
    const auto result = compile_native_general_market(env, 0, current, future);
    if (!(!result.feasible && !result.audit.compiler_feasible &&
          result.audit.diagnostic_code == "MISSED_ACQUISITION_DEADLINE" &&
          result.audit.diagnostic_step == 2 &&
          result.audit.diagnostic_actor == 0 &&
          result.audit.diagnostic_item == static_cast<int>(Item::GOOSE) &&
          result.audit.diagnostic_quantity == 1)) {
      std::cerr << "current-phase diagnostic: feasible=" << result.feasible
                << " compiler=" << result.audit.compiler_feasible
                << " code=" << result.audit.diagnostic_code
                << " step=" << result.audit.diagnostic_step
                << " actor=" << result.audit.diagnostic_actor
                << " item=" << result.audit.diagnostic_item
                << " quantity=" << result.audit.diagnostic_quantity
                << " reason=" << result.audit.reason << '\n';
    }
    require(!result.feasible && !result.audit.compiler_feasible &&
                result.audit.diagnostic_code == "MISSED_ACQUISITION_DEADLINE" &&
                result.audit.diagnostic_step == 2 &&
                result.audit.diagnostic_actor == 0 &&
                result.audit.diagnostic_item == static_cast<int>(Item::GOOSE) &&
                result.audit.diagnostic_quantity == 1,
            "a current-phase PICKUP cannot be funded by the later market phase");
  }
  {
    Config config;
    config.starting_money = 100000;
    Simulator env(config, 790);
    PlayerAction setup;
    setup.units.push_back({Op::BUILD_COOP});
    setup.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
    env.step({setup, PlayerAction{}});
    require(env.privates()[0].shed[static_cast<int>(Item::GOOSE)] == 1,
            "official fixture acquires the animal before the current unit phase");
    const std::vector<Action> current{{Op::PICKUP, Item::GOOSE, 1}};
    const std::vector<NativeFutureUnitFrame> future{
        {2, {{Op::PLACE, Item::GOOSE, 1}}}};
    const auto result = compile_native_general_market(env, 0, current, future);
    require(result.feasible && result.market.empty(),
            "current PICKUP is simulated before a future PLACE without a duplicate buy");
  }
  {
    Simulator env(Config{}, 791);
    bool rejected = false;
    try {
      (void)compile_native_general_market(
          env, 0, {}, {{1, {{Op::BUY_SEED, Item::WHEAT, 1}}}});
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "market operations cannot enter the future-unit ABI");
  }
  {
    Config config;
    config.starting_money = 100000;
    config.weed_spawn_chance = 0.0;
    Simulator env(config, 792);
    PlayerAction hire;
    hire.units.push_back({});
    hire.market.push_back({Op::HIRE});
    step(env, std::move(hire));
    const std::vector<Action> current{
        {Op::PLANT, Item::WHEAT, 1},
        {Op::EAST, Item::MELON, 7},
    };
    const auto result = compile_native_general_market(env, 0, current, {});
    require(result.feasible && result.market.empty() &&
                result.current_unit_replacements.size() == 1 &&
                result.current_unit_replacements[0].actor == 0 &&
                result.current_unit_replacements[0].original.op == Op::PLANT &&
                result.current_unit_replacements[0].replacement.op == Op::PASS &&
                result.audit.soft_misses == 1 &&
                result.audit.soft_pass_replacements == 1 &&
                result.audit.production_protection_soft_misses == 1,
            "irreversible current PLANT miss softens only that non-MOVE action");
    require(result.current_unit_replacements[0].actor != 1 &&
                current[1].op == Op::EAST && current[1].item == Item::MELON &&
                current[1].quantity == 7,
            "MOVE remains byte-identical and absent from compiler replacements");
  }
  {
    Config config;
    config.starting_money = 100000;
    config.weed_spawn_chance = 0.0;
    Simulator env(config, 793);
    advance_to(env, 47);
    PlayerAction buy_seed;
    buy_seed.units.push_back({});
    buy_seed.market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
    step(env, std::move(buy_seed));
    require(env.step_count() == 48, "maturity fixture reaches plant step");
    step(env, PlayerAction{{{Op::PLANT, Item::WHEAT, 1}}, {}});
    step(env, PlayerAction{{{Op::WATER}}, {}});
    advance_to(env, 72);
    step(env, PlayerAction{{{Op::WATER}}, {}});
    advance_to(env, 96);
    step(env, PlayerAction{{{Op::WATER}}, {}});
    advance_to(env, 106);
    const auto& planted = env.farms()[0].tiles[44];
    require(planted.kind == TileKind::PLANT && planted.crop == Item::WHEAT &&
                planted.planted_day == 2 && planted.yield_units > 0,
            "official fixture has mature day-2 WHEAT at step 106");
    const std::vector<Action> current{{Op::HARVEST}};
    const std::vector<NativeFutureUnitFrame> future{
        {107, {{Op::PLANT, Item::WHEAT, 1}}}};
    const auto result = compile_native_general_market(env, 0, current, future);
    require(result.feasible && result.current_unit_replacements.empty() &&
                result.market.size() == 1 &&
                result.market[0].op == Op::BUY_SEED &&
                result.market[0].item == Item::WHEAT &&
                result.market[0].quantity == 1,
            "mature step-106 WHEAT HARVEST clears for hard step-107 BUY_SEED/PLANT");
  }
  std::cout << "native_general_market_tests: PASS\n";
}
