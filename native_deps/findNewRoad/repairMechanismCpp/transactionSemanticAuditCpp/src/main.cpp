#include "reactive_production.hpp"
#include "repair.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace production = joint_fixed_move_oracle::production;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;

namespace {

PlayerAction unit(Op operation, Item item = Item::NONE) {
  PlayerAction result;
  result.units.push_back({operation, item, 1});
  return result;
}

void step(fastkag::Simulator& simulator,
          const PlayerAction& focal = {}) {
  std::array<PlayerAction, 2> actions{};
  actions[0] = focal;
  simulator.step(actions);
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct CropWitness {
  int completed_cycles{};
  int remaining_seeds{};
  bool debt_active_after_harvest{};
  Op next_operation{Op::PASS};
  fastkag::TileKind tile_after_harvest{fastkag::TileKind::EMPTY};
};

CropWitness crop_witness() {
  fastkag::Config config;
  config.episode_steps = 144;
  config.weed_spawn_chance = 1.0;
  fastkag::Simulator simulator(config, 42001);
  while (simulator.step_count() < 23) step(simulator);
  PlayerAction buy;
  buy.market.push_back({Op::BUY_SEED, Item::WHEAT, 2});
  step(simulator, buy);

  production::ReactiveState state;
  production::ReactiveAudit audit;
  production::ReactiveOptions options;
  // These are the exact crop ownership switches used by the integrated
  // route_cursor_production path in native_teammate.cpp.
  options.recover_old_crop_weeds = false;
  options.manage_all_known_crops = false;
  options.recover_missing_seed_debt = false;
  options.require_legacy_drop_move_for_new_debt = true;
  options.causal_cycle_guard = true;

  auto compiled = production::apply_reactive_fixed_move(
      unit(Op::PLANT, Item::WHEAT), simulator, 0, state, audit, options);
  require(compiled.units[0].op == Op::DIG, "blocked PLANT must create DIG debt");
  step(simulator, compiled);
  compiled = production::apply_reactive_fixed_move(
      unit(Op::PASS), simulator, 0, state, audit, options);
  require(compiled.units[0].op == Op::PLANT, "debt must restore PLANT");
  step(simulator, compiled);
  compiled = production::apply_reactive_fixed_move(
      unit(Op::PASS), simulator, 0, state, audit, options);
  require(compiled.units[0].op == Op::WATER, "debt must restore first WATER");
  step(simulator, compiled);

  for (const int boundary : {48, 72}) {
    while (simulator.step_count() < boundary) step(simulator);
    compiled = production::apply_reactive_fixed_move(
        unit(Op::PASS), simulator, 0, state, audit, options);
    step(simulator, compiled);
  }
  require(compiled.units[0].op == Op::HARVEST,
          "crop witness must reach the integrated completed-cycle boundary");
  const auto position = simulator.farms()[0].farmer;
  const auto tile_index = position.y * config.board_size + position.x;
  require(simulator.farms()[0].tiles[static_cast<std::size_t>(tile_index)].kind ==
              fastkag::TileKind::EMPTY,
          "non-ongoing WHEAT harvest must leave the tile empty");
  require(simulator.privates()[0].seeds[0] == 1,
          "a feasible replant seed must remain in the witness");

  const auto next = production::apply_reactive_fixed_move(
      unit(Op::PASS), simulator, 0, state, audit, options);
  return {audit.completed_cycles, simulator.privates()[0].seeds[0],
          state.debts[static_cast<std::size_t>(tile_index)].active,
          next.units[0].op,
          simulator.farms()[0].tiles[static_cast<std::size_t>(tile_index)].kind};
}

struct AnimalWitness {
  int inferred_retry_fills{};
  int outstanding{};
  bool retired{};
  bool followup_decision{};
};

AnimalWitness animal_witness() {
  using g001::repair::Animal;
  using g001::repair::AnimalObservation;
  using g001::repair::AnimalRetryController;
  AnimalRetryController controller;
  AnimalObservation original;
  original.step = 10;
  original.money = 2000;
  original.shed_capacity = 100;
  original.empty_structures[1] = 1;
  original.next_pickup_step[1] = 12;
  original.next_place_step[1] = 13;
  original.pickup_place_feasible[1] = true;
  controller.observe(original);
  controller.record_attempt(Animal::Cow, 1, 13);

  auto failed = original;
  failed.step = 11;
  controller.observe(failed);
  const auto retry = controller.decide(failed);
  require(retry && retry->quantity == 1, "fixture must emit one animal retry");
  controller.record_attempt(Animal::Cow, retry->quantity, retry->deadline);

  auto acquired_but_chain_gone = failed;
  acquired_but_chain_gone.step = 12;
  acquired_but_chain_gone.own_total[1] = 1;
  acquired_but_chain_gone.empty_structures[1] = 0;
  acquired_but_chain_gone.next_pickup_step[1] = -1;
  acquired_but_chain_gone.next_place_step[1] = -1;
  acquired_but_chain_gone.pickup_place_feasible[1] = false;
  controller.observe(acquired_but_chain_gone);
  const auto snapshot = controller.snapshot(Animal::Cow);
  return {snapshot.inferred_fills, snapshot.outstanding, snapshot.retired,
          controller.decide(acquired_but_chain_gone).has_value()};
}

struct CompositionWitness {
  Op raw_operation{Op::PASS};
  Op overlay_operation{Op::PASS};
  Op cursor_operation{Op::PASS};
  Op submitted_operation{Op::PASS};
};

CompositionWitness composition_witness() {
  fastkag::Config config;
  config.episode_steps = 48;
  fastkag::Simulator simulator(config, 42002);

  // Smallest representation of native_teammate.cpp's current ownership
  // order: an earlier dynamic overlay has composed DROP, but RouteCursor is
  // called with only the raw tape and replaces the complete vector at 1328.
  std::vector<PlayerAction> tape(
      static_cast<std::size_t>(config.episode_steps), unit(Op::PASS));
  PlayerAction composed = tape.front();
  composed.units.front() = Action{Op::DROP};

  production::RouteCursorState state;
  production::RouteCursorAudit audit;
  const auto compiled = production::apply_reactive_route_cursor(
      tape, simulator, 0, state, audit, {});
  require(compiled.units.size() == 1,
          "composition witness must retain the live actor");
  const Op cursor_operation = compiled.units.front().op;
  composed.units = compiled.units;
  return {tape.front().units.front().op, Op::DROP, cursor_operation,
          composed.units.front().op};
}

}  // namespace

int main() try {
  const auto crop = crop_witness();
  const auto animal = animal_witness();
  const auto composition = composition_witness();
  require(crop.completed_cycles == 1 && !crop.debt_active_after_harvest &&
              crop.next_operation == Op::PASS && crop.remaining_seeds == 1,
          "crop counterexample did not expose harvest-boundary retirement");
  require(animal.inferred_retry_fills == 1 && animal.outstanding == 0 &&
              !animal.retired && !animal.followup_decision,
          "animal counterexample did not expose acquisition-only settlement");
  require(composition.raw_operation == Op::PASS &&
              composition.overlay_operation == Op::DROP &&
              composition.cursor_operation == Op::PASS &&
              composition.submitted_operation == Op::PASS,
          "composition counterexample did not expose unit-vector replacement");
  std::cout
      << "{\"schema\":\"repair-transaction-semantic-audit-v1\","
      << "\"crop\":{\"completed_cycles\":" << crop.completed_cycles
      << ",\"remaining_seed\":" << crop.remaining_seeds
      << ",\"tile_after_harvest\":"
      << static_cast<int>(crop.tile_after_harvest)
      << ",\"debt_active_after_harvest\":"
      << (crop.debt_active_after_harvest ? "true" : "false")
      << ",\"next_operation\":" << static_cast<int>(crop.next_operation)
      << "},\"animal\":{\"retry_fill_counted\":"
      << animal.inferred_retry_fills << ",\"outstanding\":"
      << animal.outstanding << ",\"retired\":"
      << (animal.retired ? "true" : "false")
      << ",\"followup_decision\":"
      << (animal.followup_decision ? "true" : "false")
      << "},\"composition\":{\"raw_operation\":"
      << static_cast<int>(composition.raw_operation)
      << ",\"overlay_operation\":"
      << static_cast<int>(composition.overlay_operation)
      << ",\"cursor_operation\":"
      << static_cast<int>(composition.cursor_operation)
      << ",\"submitted_operation\":"
      << static_cast<int>(composition.submitted_operation) << "}}\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "transaction semantic audit failed: " << error.what() << '\n';
  return 1;
}
