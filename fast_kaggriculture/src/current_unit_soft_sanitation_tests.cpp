#include "current_unit_soft_sanitation.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

bool same_action(const fastkag::Action& left, const fastkag::Action& right) {
  return left.op == right.op && left.item == right.item &&
         left.quantity == right.quantity;
}

fastkag::CurrentUnitSanitationInput base_input(int actors = 1) {
  fastkag::CurrentUnitSanitationInput input;
  input.step = 77;
  input.board_size = 10;
  input.shed_capacity = 100;
  input.tiles.resize(100);
  input.actor_positions.resize(actors, {4, 4});
  input.carried.resize(actors);
  input.carried_order.resize(actors);
  input.units.resize(actors);
  return input;
}

int tile_index(fastkag::Position position) {
  return position.y * 10 + position.x;
}

}  // namespace

int main() {
  using namespace fastkag;
  {
    auto input = base_input();
    input.units[0] = {Op::NORTH, Item::MELON, 7};
    const auto result = sanitize_current_units(input);
    require(same_action(result.units[0], input.units[0]) &&
                result.move_actions_preserved == 1 && result.misses.empty(),
            "MOVE is preserved byte-for-byte");
  }
  {
    auto input = base_input();
    input.units[0] = {Op::PLANT, Item::STRAWBERRY, 1};
    const auto result = sanitize_current_units(input);
    require(result.units[0].op == Op::PASS && result.pass_substitutions == 1 &&
                result.unexpected_current_misses == 1 &&
                result.production_protection_window &&
                result.production_protection_misses == 1 &&
                result.misses[0].kind == SoftMissKind::MissingSeed &&
                result.misses[0].unexpected_current_miss &&
                result.misses[0].production_protection_window &&
                result.misses[0].loss_proxy == 4.0,
            "missing current seed becomes an audited PASS, never same-tick market hope");
  }
  {
    auto input = base_input();
    input.step = kProductionProtectionSteps;
    input.units[0] = {Op::PLANT, Item::WHEAT, 1};
    const auto result = sanitize_current_units(input);
    require(!result.production_protection_window &&
                result.production_protection_misses == 0 &&
                !result.misses[0].production_protection_window,
            "step 220 is outside the explicitly audited production-protection window");
  }
  {
    auto input = base_input(2);
    input.seeds[static_cast<int>(Item::WHEAT)] = 1;
    input.units[0] = {Op::PLANT, Item::WHEAT, 1};
    input.units[1] = {Op::PLANT, Item::WHEAT, 1};
    const auto result = sanitize_current_units(input);
    require(result.units[0].op == Op::PASS && result.units[1].op == Op::PASS &&
                result.misses.size() == 2 && result.misses[0].required == 2 &&
                result.misses[0].available == 1,
            "aggregate same-crop shortage mirrors official all-PLANT blocking");
  }
  {
    auto input = base_input();
    input.tiles[tile_index(input.actor_positions[0])].kind = TileKind::ANIMAL;
    input.shed[static_cast<int>(Item::WHEAT)] = 1;
    input.units[0] = {Op::FEED};
    const auto result = sanitize_current_units(input);
    require(result.units[0].op == Op::PICKUP &&
                result.units[0].item == Item::WHEAT &&
                result.units[0].quantity == 1 &&
                result.misses[0].recovery == SoftRecovery::Pickup &&
                result.misses[0].loss_proxy == 3.0,
            "legal shed-adjacent FEED miss becomes PICKUP wheat");
  }
  {
    auto input = base_input(2);
    input.actor_positions[1] = {5, 4};
    input.tiles[tile_index(input.actor_positions[1])].kind = TileKind::ANIMAL;
    input.shed[static_cast<int>(Item::WHEAT)] = 1;
    input.units[0] = {Op::PICKUP, Item::WHEAT, 1};
    input.units[1] = {Op::FEED};
    const auto result = sanitize_current_units(input);
    require(same_action(result.units[0], input.units[0]) &&
                result.units[1].op == Op::PASS &&
                result.misses[0].recovery == SoftRecovery::Pass,
            "an earlier original PICKUP reserves shed stock before sanitation");
  }
  {
    auto input = base_input();
    input.actor_positions[0] = {3, 4};
    input.tiles[tile_index(input.actor_positions[0])].kind = TileKind::ANIMAL;
    input.shed[static_cast<int>(Item::WHEAT)] = 2;
    input.units[0] = {Op::FEED};
    const auto result = sanitize_current_units(input);
    require(result.units[0].op == Op::PASS &&
                result.misses[0].recovery == SoftRecovery::Pass,
            "non-adjacent actor cannot synthesize an illegal PICKUP");
  }
  {
    auto input = base_input();
    input.tiles[tile_index(input.actor_positions[0])].kind = TileKind::PLANT;
    input.shed[static_cast<int>(Item::FERTILIZER)] = 1;
    input.units[0] = {Op::FERTILIZE};
    const auto result = sanitize_current_units(input);
    require(result.units[0].op == Op::PICKUP &&
                result.units[0].item == Item::FERTILIZER &&
                result.misses[0].kind == SoftMissKind::MissingCarriedFertilizer,
            "legal FERTILIZE miss becomes PICKUP fertilizer");
  }
  {
    auto input = base_input();
    input.tiles[tile_index(input.actor_positions[0])].kind = TileKind::COOP;
    input.shed[static_cast<int>(Item::GOOSE)] = 1;
    input.units[0] = {Op::PLACE, Item::GOOSE, 1};
    const auto result = sanitize_current_units(input);
    require(result.units[0].op == Op::PICKUP &&
                result.units[0].item == Item::GOOSE &&
                result.misses[0].kind == SoftMissKind::MissingCarriedPlaceItem,
            "legal animal PLACE miss becomes PICKUP animal");
  }
  {
    auto input = base_input();
    input.tiles[tile_index(input.actor_positions[0])].kind = TileKind::ANIMAL;
    input.carried[0][static_cast<int>(Item::WHEAT)] = 1;
    input.carried_order[0].push_back(static_cast<int8_t>(Item::WHEAT));
    input.units[0] = {Op::FEED};
    const auto result = sanitize_current_units(input);
    require(same_action(result.units[0], input.units[0]) && result.misses.empty(),
            "resource-satisfied production action remains unchanged");
  }
  {
    auto input = base_input();
    input.units[0] = {Op::FEED};
    const auto result = sanitize_current_units(input);
    require(same_action(result.units[0], input.units[0]) && result.misses.empty(),
            "non-resource invalidity is outside this narrow sanitation layer");
  }
  {
    Config config;
    config.starting_money = 100000;
    Simulator env(config, 9001);
    PlayerAction setup;
    setup.units.push_back({Op::BUILD_COOP});
    setup.market.push_back({Op::BUY_ANIMAL, Item::GOOSE, 1});
    setup.market.push_back({Op::BUY_PRODUCT, Item::WHEAT, 1});
    env.step({setup, PlayerAction{}});
    env.step({PlayerAction{{{Op::PICKUP, Item::GOOSE, 1}}, {}}, PlayerAction{}});
    env.step({PlayerAction{{{Op::PLACE, Item::GOOSE, 1}}, {}}, PlayerAction{}});
    const std::vector<Action> feed{{Op::FEED}};
    const auto result = sanitize_current_units(
        current_unit_sanitation_input(env, 0, feed));
    require(result.units[0].op == Op::PICKUP && result.units[0].item == Item::WHEAT,
            "official trace adapter emits the expected recovery PICKUP");
    env.step({PlayerAction{result.units, {}}, PlayerAction{}});
    require(env.privates()[0].inventories[0][static_cast<int>(Item::WHEAT)] == 1,
            "official simulator executes the sanitation PICKUP");
  }
  {
    Simulator env(Config{}, 9002);
    const std::vector<Action> plant{{Op::PLANT, Item::WHEAT, 1}};
    const auto result = sanitize_current_units(
        current_unit_sanitation_input(env, 0, plant));
    PlayerAction action{result.units, {{Op::BUY_SEED, Item::WHEAT, 1}}};
    env.step({action, PlayerAction{}});
    require(env.farms()[0].tiles[44].kind == TileKind::EMPTY &&
                env.privates()[0].seeds[static_cast<int>(Item::WHEAT)] == 1,
            "same-tick BUY_SEED cannot rescue the sanitized current PLANT");
  }
  std::cout << "current_unit_soft_sanitation_tests: PASS\n";
}
