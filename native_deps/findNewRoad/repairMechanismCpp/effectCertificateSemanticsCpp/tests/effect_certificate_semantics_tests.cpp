#include "effect_certificate_semantics.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

namespace {

using g001::effect_certificate::Failure;
using g001::effect_certificate::LocalState;
using g001::effect_certificate::Prediction;
using fastkag::Action;
using fastkag::Config;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;

int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: "          \
                << #condition << '\n';                                         \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

fastkag::Farm& farm(Simulator& simulator) {
  return const_cast<fastkag::Farm&>(simulator.farms()[0]);
}

fastkag::PrivateState& private_state(Simulator& simulator) {
  return const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
}

Tile& actor_tile(Simulator& simulator) {
  const auto position = simulator.farms()[0].farmer;
  return farm(simulator).tiles[static_cast<std::size_t>(
      position.y * simulator.config().board_size + position.x)];
}

bool shed_adjacent(const Config& config, fastkag::Position position) {
  const int half = config.board_size / 2;
  return (position.x == half - 1 || position.x == half) &&
         (position.y == half - 1 || position.y == half);
}

LocalState snapshot(const Simulator& simulator) {
  LocalState result;
  result.config = simulator.config();
  result.day = simulator.day();
  result.actor_slot = 0;
  result.actor_exists = true;
  result.actor_position = simulator.farms()[0].farmer;
  result.shed_adjacent =
      shed_adjacent(simulator.config(), result.actor_position);
  const auto index = static_cast<std::size_t>(
      result.actor_position.y * simulator.config().board_size +
      result.actor_position.x);
  result.tile = simulator.farms()[0].tiles[index];
  result.shed = simulator.privates()[0].shed;
  result.inventory = simulator.privates()[0].inventories[0];
  result.inventory_order = simulator.privates()[0].inventory_order[0];
  return result;
}

Prediction compare_once(const std::string& name, Simulator simulator,
                        Action action, Failure expected_failure) {
  const auto before = snapshot(simulator);
  const auto prediction =
      g001::effect_certificate::predict_unit_effect(before, action);
  std::array<PlayerAction, 2> manifests;
  manifests[0].units.push_back(action);
  const auto actual = simulator.preview_unit_phase(manifests);
  if (!(prediction.after == snapshot(actual))) {
    std::cerr << name << ": predicted after-state differs from Simulator\n";
    ++failures;
  }
  if (prediction.failure != expected_failure) {
    std::cerr << name << ": expected failure "
              << g001::effect_certificate::failure_name(expected_failure)
              << ", got "
              << g001::effect_certificate::failure_name(prediction.failure)
              << '\n';
    ++failures;
  }
  CHECK(prediction.changed == !(prediction.after == before));
  return prediction;
}

Simulator make_simulator(int capacity = 100) {
  Config config;
  config.shed_capacity = capacity;
  config.weed_spawn_chance = 0.0;
  return Simulator(config, 4139);
}

void put_inventory(Simulator& simulator, int item, int quantity) {
  auto& state = private_state(simulator);
  if (state.inventories[0][static_cast<std::size_t>(item)] == 0 &&
      quantity > 0) {
    state.inventory_order[0].push_back(static_cast<int8_t>(item));
  }
  state.inventories[0][static_cast<std::size_t>(item)] = quantity;
}

void test_drop() {
  auto simulator = make_simulator(3);
  auto& state = private_state(simulator);
  state.shed[0] = 2;
  put_inventory(simulator, 1, 2);
  put_inventory(simulator, 8, 2);
  const auto result = compare_once(
      "drop-capacity-order-and-overflow-loss", std::move(simulator),
      Action{Op::DROP, Item::MELON, 99}, Failure::None);
  CHECK(result.transferred == 1);
  CHECK(result.after.shed[1] == 1);
  CHECK(result.after.inventory[1] == 0);
  CHECK(result.after.inventory[8] == 0);
  CHECK(result.after.inventory_order.empty());

  simulator = make_simulator();
  farm(simulator).farmer = {0, 0};
  put_inventory(simulator, 0, 2);
  compare_once("drop-not-adjacent", std::move(simulator),
               Action{Op::DROP, Item::NONE, -9},
               Failure::NotShedAdjacent);
}

void test_pickup() {
  auto simulator = make_simulator();
  private_state(simulator).shed[10] = 2;
  const auto result = compare_once("pickup-clamps-to-shed", simulator,
                                   Action{Op::PICKUP, Item::COW, 5},
                                   Failure::None);
  CHECK(result.transferred == 2);
  CHECK(result.after.inventory[10] == 2);
  CHECK(result.after.inventory_order == std::vector<int8_t>{10});
  compare_once("pickup-zero-quantity", std::move(simulator),
               Action{Op::PICKUP, Item::COW, 0},
               Failure::InvalidItemOrQuantity);
}

void test_place() {
  auto simulator = make_simulator();
  actor_tile(simulator).kind = TileKind::COOP;
  put_inventory(simulator, 9, 1);
  const auto animal = compare_once(
      "place-animal-consumes-exactly-one", std::move(simulator),
      Action{Op::PLACE, Item::GOOSE, 77}, Failure::None);
  CHECK(animal.transferred == 1);
  CHECK(animal.after.tile.kind == TileKind::ANIMAL);
  CHECK(animal.after.tile.animal == Item::GOOSE);
  CHECK(animal.after.tile.placed_day == 0);

  simulator = make_simulator();
  actor_tile(simulator).kind = TileKind::EMPTY;
  put_inventory(simulator, 9, 1);
  const auto fallback = compare_once(
      "place-wrong-animal-structure-falls-back-to-shed", simulator,
      Action{Op::PLACE, Item::GOOSE, 1}, Failure::None);
  CHECK(fallback.after.tile.kind == TileKind::EMPTY);
  CHECK(fallback.after.shed[9] == 1);

  simulator = make_simulator(2);
  private_state(simulator).shed[5] = 1;
  put_inventory(simulator, 0, 3);
  const auto product = compare_once(
      "place-product-capacity-clamp", std::move(simulator),
      Action{Op::PLACE, Item::WHEAT, 3}, Failure::None);
  CHECK(product.transferred == 1);
  CHECK(product.after.inventory[0] == 2);
  CHECK(product.after.shed[0] == 1);

  simulator = make_simulator();
  actor_tile(simulator).kind = TileKind::COOP;
  compare_once("place-animal-structure-without-item-does-not-fallback",
               std::move(simulator), Action{Op::PLACE, Item::GOOSE, 1},
               Failure::ActorItemAbsent);
}

void test_fertilize() {
  auto simulator = make_simulator();
  actor_tile(simulator).kind = TileKind::PLANT;
  actor_tile(simulator).crop = Item::WHEAT;
  put_inventory(simulator, 8, 1);
  const auto result = compare_once(
      "fertilize-consumes-one", simulator,
      Action{Op::FERTILIZE, Item::COW, 0}, Failure::None);
  CHECK(result.after.tile.fertilized_until_day == 2);
  CHECK(result.after.inventory[8] == 0);

  private_state(simulator).inventories[0][8] = 0;
  private_state(simulator).inventory_order[0].clear();
  compare_once("fertilize-missing-item", std::move(simulator),
               Action{Op::FERTILIZE, Item::NONE, 999},
               Failure::FertilizerAbsent);
}

void test_feed() {
  auto simulator = make_simulator();
  actor_tile(simulator).kind = TileKind::ANIMAL;
  actor_tile(simulator).animal = Item::GOOSE;
  put_inventory(simulator, 0, 1);
  const auto result = compare_once("feed-ignores-action-item-and-quantity",
                                   simulator,
                                   Action{Op::FEED, Item::MELON, 0},
                                   Failure::None);
  CHECK(result.after.tile.fed_today);
  CHECK(result.after.inventory[0] == 0);

  actor_tile(simulator).fed_today = true;
  compare_once("feed-already-fed", std::move(simulator),
               Action{Op::FEED, Item::NONE, 1}, Failure::AlreadyFed);
}

void test_collect_fertilizer() {
  auto simulator = make_simulator();
  actor_tile(simulator).kind = TileKind::ANIMAL;
  actor_tile(simulator).animal = Item::SHEEP;
  actor_tile(simulator).fertilizer_available = true;
  const auto result = compare_once(
      "collect-fertilizer", simulator,
      Action{Op::COLLECT_FERTILIZER, Item::MILK, -4}, Failure::None);
  CHECK(!result.after.tile.fertilizer_available);
  CHECK(result.after.inventory[8] == 1);
  CHECK(result.after.inventory_order == std::vector<int8_t>{8});

  actor_tile(simulator).fertilizer_available = false;
  compare_once("collect-fertilizer-unavailable", std::move(simulator),
               Action{Op::COLLECT_FERTILIZER, Item::NONE, 1},
               Failure::FertilizerUnavailable);
}

void test_harvest() {
  auto simulator = make_simulator();
  auto& wheat = actor_tile(simulator);
  wheat.kind = TileKind::PLANT;
  wheat.crop = Item::WHEAT;
  wheat.planted_day = -2;
  wheat.yield_units = 3;
  const auto terminal = compare_once(
      "harvest-terminal-crop", std::move(simulator),
      Action{Op::HARVEST, Item::SHEEP, 0}, Failure::None);
  CHECK(terminal.transferred == 3);
  CHECK(terminal.after.tile.kind == TileKind::EMPTY);
  CHECK(terminal.after.inventory[0] == 3);

  simulator = make_simulator();
  auto& tomato = actor_tile(simulator);
  tomato.kind = TileKind::PLANT;
  tomato.crop = Item::TOMATO;
  tomato.planted_day = -8;
  tomato.yield_units = 2;
  const auto ongoing = compare_once(
      "harvest-ongoing-crop", std::move(simulator),
      Action{Op::HARVEST, Item::NONE, -1}, Failure::None);
  CHECK(ongoing.after.tile.kind == TileKind::PLANT);
  CHECK(ongoing.after.tile.crop == Item::TOMATO);
  CHECK(ongoing.after.tile.yield_units == 0);
  CHECK(ongoing.after.inventory[2] == 2);

  simulator = make_simulator();
  auto& immature = actor_tile(simulator);
  immature.kind = TileKind::PLANT;
  immature.crop = Item::WHEAT;
  immature.planted_day = 0;
  immature.yield_units = 1;
  compare_once("harvest-immature", std::move(simulator),
               Action{Op::HARVEST, Item::NONE, 1}, Failure::CropImmature);

  simulator = make_simulator();
  auto& cow = actor_tile(simulator);
  cow.kind = TileKind::ANIMAL;
  cow.animal = Item::COW;
  cow.yield_units = 2;
  const auto animal = compare_once("harvest-animal", std::move(simulator),
                                   Action{Op::HARVEST, Item::WHEAT, 99},
                                   Failure::None);
  CHECK(animal.after.tile.kind == TileKind::ANIMAL);
  CHECK(animal.after.tile.yield_units == 0);
  CHECK(animal.after.inventory[6] == 2);

  simulator = make_simulator();
  actor_tile(simulator).kind = TileKind::ANIMAL;
  actor_tile(simulator).animal = Item::GOOSE;
  compare_once("harvest-zero-yield", std::move(simulator),
               Action{Op::HARVEST, Item::NONE, 1}, Failure::YieldAbsent);
}

void test_locked_precedence() {
  auto simulator = make_simulator();
  actor_tile(simulator).kind = TileKind::LOCKED;
  put_inventory(simulator, 8, 1);
  compare_once("locked-fertilize", std::move(simulator),
               Action{Op::FERTILIZE, Item::NONE, 1}, Failure::LockedTile);
}

}  // namespace

int main() {
  test_drop();
  test_pickup();
  test_place();
  test_fertilize();
  test_feed();
  test_collect_fertilizer();
  test_harvest();
  test_locked_precedence();
  if (failures != 0) {
    std::cerr << failures << " effect-certificate differential checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "effect-certificate differential checks passed: 20 fixtures\n";
  return EXIT_SUCCESS;
}
