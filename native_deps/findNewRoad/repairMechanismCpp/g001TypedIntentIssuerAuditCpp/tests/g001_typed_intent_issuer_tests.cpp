#include "g001_typed_intent_issuer.hpp"

#include <array>
#include <cassert>

namespace ti = g001::typed_intent;
namespace po = production_obligation;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;

namespace {

void step(fastkag::Simulator& simulator, PlayerAction focal = {}) {
  std::array<PlayerAction, 2> actions;
  actions[0] = std::move(focal);
  simulator.step(actions);
}

void buy_and_plant_wheat(fastkag::Simulator& simulator) {
  PlayerAction buy;
  buy.market.push_back({Op::BUY_SEED, Item::WHEAT, 1});
  step(simulator, buy);
  PlayerAction plant;
  plant.units.push_back({Op::PLANT, Item::WHEAT, 1});
  step(simulator, plant);
}

ti::Result issue(fastkag::Simulator& simulator, Action action,
                 std::span<const po::ObligationNode> nodes = {},
                 bool certified = true) {
  std::array<Action, 1> units{action};
  return ti::issue({&simulator, 0, 7, units, nodes, certified});
}

}  // namespace

int main() {
  {
    fastkag::Config config;
    config.weed_spawn_chance = 0;
    fastkag::Simulator simulator(config, 1);
    buy_and_plant_wheat(simulator);
    const auto result = issue(simulator, {Op::WATER, Item::NONE, 1});
    assert(result.crops.size() == 1);
    assert(result.crops[0].desired == Item::WHEAT);
    assert(result.evidence[0].proof == ti::Proof::LivePlantTile);
  }
  {
    fastkag::Config config;
    config.weed_spawn_chance = 0;
    fastkag::Simulator simulator(config, 2);
    buy_and_plant_wheat(simulator);
    while (simulator.step_count() < 48) step(simulator);
    const auto result = issue(simulator, {Op::HARVEST, Item::NONE, 1});
    assert(result.crops.empty());
    assert(result.evidence[0].proof == ti::Proof::LiveTileHasNoCrop);
  }
  {
    fastkag::Simulator simulator({}, 3);
    const auto position = simulator.farms()[0].farmer;
    po::ObligationNode node;
    node.kind = po::NodeKind::DownstreamAction;
    node.item = po::Item::Tomato;
    node.actor = 0;
    node.consumer_step = simulator.step_count();
    node.position = {position.x, position.y};
    node.original_unit_op = po::UnitOp::Water;
    std::array<po::ObligationNode, 1> nodes{node};
    const auto result = issue(simulator, {Op::WATER, Item::NONE, 1}, nodes);
    assert(result.crops.size() == 1);
    assert(result.crops[0].desired == Item::TOMATO);
    assert(result.evidence[0].proof == ti::Proof::DagTypedPosition);
  }
  {
    fastkag::Simulator simulator({}, 4);
    const auto result = issue(simulator, {Op::PLACE, Item::GOOSE, 1});
    assert(result.animals.size() == 1);
    const auto position = simulator.farms()[0].farmer;
    assert(result.animals[0].target.row == position.y);
    assert(result.animals[0].target.column == position.x);
  }
  {
    fastkag::Simulator simulator({}, 5);
    const auto result = issue(simulator, {Op::HARVEST, Item::NONE, 1}, {}, false);
    assert(result.crops.empty());
    assert(result.evidence[0].proof == ti::Proof::DynamicSuffixUncertified);
  }
}
