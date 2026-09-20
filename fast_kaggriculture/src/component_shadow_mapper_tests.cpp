#include "component_shadow_mapper.hpp"

#include <array>
#include <iostream>
#include <set>
#include <stdexcept>

namespace shadow = fastkag::component_shadow;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;
using fastkag::Tile;
using fastkag::TileKind;

namespace {

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Tile& current_tile(Simulator& simulator) {
  const auto position = simulator.farms()[0].farmer;
  const int size = simulator.config().board_size;
  return const_cast<Tile&>(simulator.farms()[0].tiles[
      static_cast<std::size_t>(position.y * size + position.x)]);
}

fastkag::PrivateState& private_state(Simulator& simulator) {
  return const_cast<fastkag::PrivateState&>(simulator.privates()[0]);
}

shadow::MappedUnit map(Simulator& simulator, Action action,
                       std::uint64_t source = 1) {
  PlayerAction raw;
  raw.units.push_back(action);
  return shadow::map_current_unit(simulator, 0, 0, raw, source, 0);
}

void every_native_unit_op_has_a_semantic_class() {
  std::set<Op> covered;
  {
    Simulator simulator;
    check(map(simulator, {}).kind == shadow::CertificateKind::ExactNoEffect,
          "PASS certificate");
    covered.insert(Op::PASS);
  }
  for (const auto op : {Op::NORTH, Op::SOUTH, Op::EAST, Op::WEST}) {
    Simulator simulator;
    check(map(simulator, {op}).kind ==
              shadow::CertificateKind::MoveTransition,
          "legal MOVE certificate");
    covered.insert(op);
  }
  {
    Simulator simulator;
    auto& state = private_state(simulator);
    state.inventories[0][0] = 1;
    state.inventory_order[0].push_back(0);
    check(map(simulator, {Op::DROP}).kind ==
              shadow::CertificateKind::TypedEffect,
          "DROP must expose an exact typed inventory/shed effect");
    covered.insert(Op::DROP);
  }
  {
    Simulator simulator;
    private_state(simulator).shed[0] = 1;
    check(map(simulator, {Op::PICKUP, Item::WHEAT, 1}).kind ==
              shadow::CertificateKind::TypedEffect,
          "PICKUP must expose an exact typed inventory/shed effect");
    covered.insert(Op::PICKUP);
  }
  {
    Simulator simulator;
    auto& state = private_state(simulator);
    state.inventories[0][9] = 1;
    state.inventory_order[0].push_back(9);
    current_tile(simulator).kind = TileKind::COOP;
    check(map(simulator, {Op::PLACE, Item::GOOSE, 1}).kind ==
              shadow::CertificateKind::TypedEffect,
          "PLACE must expose an exact typed inventory/animal effect");
    covered.insert(Op::PLACE);
  }
  {
    Simulator simulator;
    private_state(simulator).seeds[0] = 1;
    check(map(simulator, {Op::PLANT, Item::WHEAT, 1}).kind ==
              shadow::CertificateKind::TileTransition,
          "PLANT transition certificate");
    covered.insert(Op::PLANT);
  }
  {
    Simulator simulator;
    auto& tile = current_tile(simulator);
    tile.kind = TileKind::PLANT;
    tile.crop = Item::WHEAT;
    check(map(simulator, {Op::WATER, Item::WHEAT, 1}).kind ==
              shadow::CertificateKind::TileTransition,
          "WATER transition certificate");
    covered.insert(Op::WATER);
  }
  {
    Simulator simulator;
    auto& tile = current_tile(simulator);
    tile.kind = TileKind::PLANT;
    tile.crop = Item::WHEAT;
    tile.planted_day = -100;
    tile.yield_units = 1;
    const auto harvested = map(simulator, {Op::HARVEST, Item::WHEAT, 1});
    check(harvested.kind == shadow::CertificateKind::TypedEffect &&
              harvested.typed_effect.has_value() &&
              harvested.raw_source.has_value() &&
              harvested.raw_source->tile_transition_certificate.has_value(),
          "terminal HARVEST typed/tile transition certificate");
    covered.insert(Op::HARVEST);
  }
  {
    Simulator simulator;
    auto& tile = current_tile(simulator);
    tile.kind = TileKind::PLANT;
    tile.crop = Item::WHEAT;
    private_state(simulator).inventories[0][8] = 1;
    check(map(simulator, {Op::FERTILIZE, Item::FERTILIZER, 1}).kind ==
              shadow::CertificateKind::TypedEffect,
          "FERTILIZE must expose an exact typed effect");
    covered.insert(Op::FERTILIZE);
  }
  {
    Simulator simulator;
    current_tile(simulator).kind = TileKind::WEED;
    check(map(simulator, {Op::DIG}).kind ==
              shadow::CertificateKind::TileTransition,
          "DIG transition certificate");
    covered.insert(Op::DIG);
  }
  for (const auto op : {Op::BUILD_COOP, Op::BUILD_PASTURE}) {
    Simulator simulator;
    check(map(simulator, {op}).kind ==
              shadow::CertificateKind::TileTransition,
          "BUILD transition certificate");
    covered.insert(op);
  }
  {
    Simulator simulator;
    auto& tile = current_tile(simulator);
    tile.kind = TileKind::ANIMAL;
    tile.animal = Item::GOOSE;
    private_state(simulator).inventories[0][0] = 1;
    check(map(simulator, {Op::FEED}).kind ==
              shadow::CertificateKind::TypedEffect,
          "FEED must expose an exact typed effect");
    covered.insert(Op::FEED);
  }
  {
    Simulator simulator;
    auto& tile = current_tile(simulator);
    tile.kind = TileKind::ANIMAL;
    tile.animal = Item::GOOSE;
    tile.fertilizer_available = true;
    check(map(simulator, {Op::COLLECT_FERTILIZER}).kind ==
              shadow::CertificateKind::TypedEffect,
          "COLLECT_FERTILIZER must expose an exact typed effect");
    covered.insert(Op::COLLECT_FERTILIZER);
  }
  {
    Simulator simulator;
    auto& tile = current_tile(simulator);
    tile.kind = TileKind::ANIMAL;
    tile.animal = Item::GOOSE;
    check(map(simulator, {Op::CARE}).kind ==
              shadow::CertificateKind::ScopedEffect,
          "CARE tile-local certificate");
    covered.insert(Op::CARE);
  }
  check(covered.size() == 18, "not every native unit op was covered");
}

void invalid_crop_action_becomes_observation_driven_objective() {
  Simulator simulator;
  current_tile(simulator).kind = TileKind::WEED;
  const auto mapped = map(
      simulator, {Op::HARVEST, Item::WHEAT, 1}, 77);
  check(mapped.kind == shadow::CertificateKind::ReplaceableObjective &&
            !mapped.raw_source.has_value() && mapped.objective.has_value() &&
            mapped.objective->transitions.size() == 3 &&
            mapped.objective->transitions[0].op ==
                g001::event_local_repair::Op::Dig &&
            mapped.objective->transitions[1].op ==
                g001::event_local_repair::Op::Plant,
        "weed HARVEST was not recompiled to DIG/PLANT/WATER objective");
}

void concurrent_seed_overbooking_is_not_falsely_actor_causal() {
  Simulator simulator;
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  farm.hands.push_back(farm.farmer);
  auto& state = private_state(simulator);
  state.inventories.resize(2);
  state.inventory_order.resize(2);
  state.seeds[0] = 1;
  PlayerAction raw;
  raw.units = {{Op::PLANT, Item::WHEAT, 1},
               {Op::PLANT, Item::WHEAT, 1}};
  const auto mapped = shadow::map_current_unit(simulator, 0, 0, raw, 88, 0);
  check(mapped.kind == shadow::CertificateKind::GlobalOpaque,
        "two-actor seed rejection was falsely certified to actor0");
}

void shared_shed_effect_is_bound_to_lower_actor_prefix() {
  Simulator simulator;
  auto& farm = const_cast<fastkag::Farm&>(simulator.farms()[0]);
  farm.hands.push_back(farm.farmer);
  auto& state = private_state(simulator);
  state.inventories.resize(2);
  state.inventory_order.resize(2);
  state.shed[0] = 1;
  PlayerAction raw;
  raw.units = {{Op::PICKUP, Item::WHEAT, 1},
               {Op::PICKUP, Item::WHEAT, 1}};
  const auto first = shadow::map_current_unit(simulator, 0, 0, raw, 91, 0);
  const auto second = shadow::map_current_unit(simulator, 0, 1, raw, 92, 0);
  check(first.kind == shadow::CertificateKind::TypedEffect &&
            first.typed_effect.has_value() &&
            first.typed_effect->lower_slot_prefix_bound &&
            first.typed_effect->prefix_hash != 0 &&
            first.typed_effect->actor_generation != 0,
        "lower actor PICKUP lacked an exact prefix certificate");
  check(second.kind == shadow::CertificateKind::ExactNoEffect &&
            second.effect_failure ==
                g001::effect_certificate::Failure::ShedItemAbsent,
        "later actor did not observe lower-slot shed consumption");

  Simulator shared;
  auto& shared_farm = const_cast<fastkag::Farm&>(shared.farms()[0]);
  shared_farm.hands.push_back(shared_farm.farmer);
  auto& shared_state = private_state(shared);
  shared_state.inventories.resize(2);
  shared_state.inventory_order.resize(2);
  shared_state.shed[0] = 2;
  const auto left = shadow::map_current_unit(shared, 0, 0, raw, 93, 0);
  const auto right = shadow::map_current_unit(shared, 0, 1, raw, 94, 0);
  check(left.raw_source.has_value() && right.raw_source.has_value() &&
            left.raw_source->effect_certificate.has_value() &&
            right.raw_source->effect_certificate.has_value() &&
            left.raw_source->effect_certificate->resource_reads.contains(200000) &&
            right.raw_source->effect_certificate->resource_reads.contains(200000),
        "same shed item did not produce a shared typed dependency key");
}

}  // namespace

int main() try {
  every_native_unit_op_has_a_semantic_class();
  invalid_crop_action_becomes_observation_driven_objective();
  concurrent_seed_overbooking_is_not_falsely_actor_causal();
  shared_shed_effect_is_bound_to_lower_actor_prefix();
  std::cout << "component shadow mapper: 4 fixture groups passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "FAIL: " << error.what() << '\n';
  return 1;
}
